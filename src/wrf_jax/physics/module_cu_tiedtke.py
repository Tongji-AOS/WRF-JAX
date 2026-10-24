"""Translations from WRF ``module_cu_tiedtke``.

The functions in this module keep WRF/Tiedtke's top-to-bottom vertical
ordering.  Integer level arguments such as ``kk`` are 1-based, matching the
Fortran source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jax import lax

from wrf_jax.core.column_scan import scan_columns
from wrf_jax.core.optimization import semantic_call
from wrf_jax.core.real_arithmetic import real_assignment, real_divide, column_binary, column_barrier
from wrf_jax.core.wrf_math import sqrt_for_wrf


G = jnp.float32(9.81)
RD = jnp.float32(287.0)
RV = jnp.float32(461.6)
CPD = jnp.float32(7.0 * 287.0 / 2.0)
ALV = jnp.float32(2.5e6)
ALS = jnp.float32(2.85e6)
ALF = jnp.float32(3.50e5)

RCPD = jnp.float32(1.0) / CPD
TMELT = jnp.float32(273.16)
T000 = jnp.float32(273.15)
HGFR = jnp.float32(233.15)
C1ES = jnp.float32(610.78)
C2ES = C1ES * RD / RV
C3LES = jnp.float32(17.269)
C3IES = jnp.float32(21.875)
C4LES = jnp.float32(35.86)
C4IES = jnp.float32(7.66)
C5LES = C3LES * (TMELT - C4LES)
C5IES = C3IES * (TMELT - C4IES)
VTMPc1 = RV / RD - jnp.float32(1.0)
ZRG = jnp.float32(1.0) / G
ZDNOPRC = jnp.float32(1.5e4)
ZBUO0 = jnp.float32(0.50)
RHOH2O = jnp.float32(1.0e3)
RHC = jnp.float32(0.80)
RHM = jnp.float32(1.0)
FDBK = jnp.float32(1.0)
ENTRPEN = jnp.float32(1.0e-4)
ENTRSCV = jnp.float32(1.2e-3)
ENTRMID = jnp.float32(1.0e-4)
ENTRDD = jnp.float32(2.0e-4)
CMFDEPS = jnp.float32(0.30)
CMFCTOP = jnp.float32(0.30)
CMFCMAX = jnp.float32(1.0)
CMFCMIN = jnp.float32(1.0e-10)
CPRCON = jnp.float32(1.1e-3 / 9.81)
CRIRH = jnp.float32(0.70)
ZTAU = jnp.float32(2400.0)
CEVAPCU1 = jnp.float32(1.93e-6 * 261.0 * 0.5 / 9.81)
CEVAPCU2 = jnp.float32(1.0e3 / (38.3 * 0.293))
LMFDD = True
LMFMID = True
LMFDUDV = True

_TIEDTKE_PROFILE_PATH = os.environ.get("WRF_PY_PROFILE_TIEDTKE", "").strip()
_JIT_CUASC = os.environ.get("WRF_PY_JIT_CUASC", "0").strip().lower() in {"1", "true", "yes", "on"}
_USE_CUASC_SCAN = os.environ.get("WRF_PY_USE_CUASC_SCAN", "0").strip().lower() in {"1", "true", "yes", "on"}
_USE_CUASC_SECOND_SCAN = os.environ.get("WRF_PY_USE_CUASC_SECOND_SCAN", "0").strip().lower() in {"1", "true", "yes", "on"}
_CUADJTQ_COMPACT_CAPACITY = int(os.environ.get("WRF_PY_TIEDTKE_CUADJTQ_COMPACT_CAPACITY", "0").strip() or "0")


def _profile_ready(value: Any) -> None:
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _profile_ready(getattr(value, name))
        return
    if isinstance(value, dict):
        for item in value.values():
            _profile_ready(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _profile_ready(item)
        return
    block = getattr(value, "block_until_ready", None)
    if block is not None:
        block()


def _profile_tiedtke_stage(name: str, fn: Any) -> Any:
    if not _TIEDTKE_PROFILE_PATH:
        return fn()
    start = perf_counter()
    result = fn()
    _profile_ready(result)
    with open(_TIEDTKE_PROFILE_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{perf_counter() - start:.6f} {name}\n")
    return result
LMFSCV = True


def _is_traced(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def _concrete_int_or_none(value: Any) -> int | None:
    if _is_traced(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _concrete_bool_or_none(value: Any) -> bool | None:
    if _is_traced(value):
        return None
    try:
        return bool(jnp.asarray(value).item())
    except (TypeError, ValueError):
        return None


def _level_index_pair(level: Any) -> tuple[Any, Any]:
    concrete = _concrete_int_or_none(level)
    if concrete is not None:
        return concrete, concrete - 1
    level_i = jnp.asarray(level, dtype=jnp.int32)
    return level_i, level_i - jnp.asarray(1, dtype=jnp.int32)


def _row_reduce_broadcast(values: Any, row_size: int | None, reducer: Any) -> Any:
    arr = jnp.asarray(values)
    if row_size is None:
        return reducer(arr)
    row_size_i = int(row_size)
    if row_size_i <= 0 or row_size_i >= int(arr.shape[0]) or int(arr.shape[0]) % row_size_i != 0:
        return reducer(arr)
    grouped = arr.reshape((int(arr.shape[0]) // row_size_i, row_size_i) + arr.shape[1:])
    reduced = reducer(grouped, axis=1)
    return jnp.broadcast_to(jnp.expand_dims(reduced, axis=1), grouped.shape).reshape(arr.shape)


def _row_any_broadcast(values: Any, row_size: int | None) -> Any:
    return _row_reduce_broadcast(values, row_size, jnp.any)


def _row_min_broadcast(values: Any, row_size: int | None) -> Any:
    return _row_reduce_broadcast(values, row_size, jnp.min)


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    return real_assignment(value, rounding_token)


def _wrf_real(value: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    array = jnp.asarray(value, dtype=dtype)
    if rounding_token is not None:
        return _store_wrf_real(array, rounding_token)
    return lax.optimization_barrier(array)


def _column_buoyancy_step(carry: Any, inputs: Any, token: Any) -> tuple[Any, Any]:
    """Keep the first threshold crossing in descending WRF level order."""
    ihmin, zhmin = carry
    level, cloud_base, eligible, increment, threshold = inputs
    active = (eligible != 0) & (ihmin == cloud_base)
    accumulated = real_assignment(zhmin + increment, token)
    current = (
        jnp.where(active & (accumulated > threshold), level, ihmin),
        jnp.where(active, accumulated, zhmin),
    )
    return current, current


@jax.custom_jvp
def _real_spacing(value):
    absolute = jnp.abs(value)
    return column_binary("sub", jnp.nextafter(absolute, jnp.float32(jnp.inf)), absolute)


@_real_spacing.defjvp
def _real_spacing_jvp(primals, tangents):
    # The representable-number spacing is a piecewise-constant branch threshold.
    result = _real_spacing(primals[0])
    return result, jnp.zeros_like(result)


def _column_precipitation_step(carry: Any, inputs: Any, token: Any) -> tuple[Any, Any]:
    """Accumulate rain and snow, then melt snow, without reordering sums."""
    prain, prfl, psfl, _ = carry
    active, warm, can_melt, up, down, melt_potential, pdpmel = inputs
    active, warm = active != 0, warm != 0
    store = lambda value: real_assignment(value, token)
    prain = store(prain + jnp.where(active, up, jnp.float32(0.0)))
    precip_inc = store(up + down)
    prfl = store(prfl + jnp.where(active & warm, precip_inc, jnp.float32(0.0)))
    psfl = store(psfl + jnp.where(active & ~warm, precip_inc, jnp.float32(0.0)))
    melt_mask = active & warm & (psfl > 0.0) & (can_melt != 0)
    melt = store(jnp.minimum(psfl, melt_potential))
    pdpmel = jnp.where(melt_mask, melt, pdpmel)
    psfl = store(psfl - jnp.where(melt_mask, melt, jnp.float32(0.0)))
    prfl = store(prfl + jnp.where(melt_mask, melt, jnp.float32(0.0)))
    current = (prain, prfl, psfl, pdpmel)
    return current, current


_EXP2F_TABLE_BITS = 5
_EXP2F_TABLE_SIZE = 1 << _EXP2F_TABLE_BITS
# Constants from musl src/math/exp2f_data.c, Copyright (c) 2017-2018 Arm Limited,
# SPDX-License-Identifier: MIT.  This reproduces the C expf path used by the
# Fortran reference over Tiedtke's saturation-adjustment temperature range.
_EXP2F_TABLE = jnp.asarray(
    [
        0x3FF0000000000000,
        0x3FEFD9B0D3158574,
        0x3FEFB5586CF9890F,
        0x3FEF9301D0125B51,
        0x3FEF72B83C7D517B,
        0x3FEF54873168B9AA,
        0x3FEF387A6E756238,
        0x3FEF1E9DF51FDEE1,
        0x3FEF06FE0A31B715,
        0x3FEEF1A7373AA9CB,
        0x3FEEDEA64C123422,
        0x3FEECE086061892D,
        0x3FEEBFDAD5362A27,
        0x3FEEB42B569D4F82,
        0x3FEEAB07DD485429,
        0x3FEEA47EB03A5585,
        0x3FEEA09E667F3BCD,
        0x3FEE9F75E8EC5F74,
        0x3FEEA11473EB0187,
        0x3FEEA589994CCE13,
        0x3FEEACE5422AA0DB,
        0x3FEEB737B0CDC5E5,
        0x3FEEC49182A3F090,
        0x3FEED503B23E255D,
        0x3FEEE89F995AD3AD,
        0x3FEEFF76F2FB5E47,
        0x3FEF199BDD85529C,
        0x3FEF3720DCEF9069,
        0x3FEF5818DCFBA487,
        0x3FEF7C97337B9B5F,
        0x3FEFA4AFA2A490DA,
        0x3FEFD0765B6E4540,
    ],
    dtype=jnp.uint64,
)
_EXP2F_SHIFT = jnp.float64(float.fromhex("0x1.8p+52"))
_EXP2F_INV_LN2_SCALED = jnp.float64(float.fromhex("0x1.71547652b82fep+0") * _EXP2F_TABLE_SIZE)
_EXP2F_C0 = jnp.float64(float.fromhex("0x1.c6af84b912394p-5") / _EXP2F_TABLE_SIZE / _EXP2F_TABLE_SIZE / _EXP2F_TABLE_SIZE)
_EXP2F_C1 = jnp.float64(float.fromhex("0x1.ebfce50fac4f3p-3") / _EXP2F_TABLE_SIZE / _EXP2F_TABLE_SIZE)
_EXP2F_C2 = jnp.float64(float.fromhex("0x1.62e42ff0c52d6p-1") / _EXP2F_TABLE_SIZE)


class _TiedtkeConstants(NamedTuple):
    als: Any
    alv: Any
    c2es: Any
    c3ies: Any
    c3les: Any
    c4ies: Any
    c4les: Any
    c5ies: Any
    c5les: Any
    cmfcmax: Any
    cmfcmin: Any
    cpd: Any
    cprcon: Any
    entrmid: Any
    g: Any
    rcpd: Any
    rd: Any
    rv: Any
    tmelt: Any
    vtmpc1: Any
    zbuo0: Any
    zdnoprc: Any
    zrg: Any
    exp2f_c0: Any
    exp2f_c1: Any
    exp2f_c2: Any
    exp2f_inv_ln2_scaled: Any
    exp2f_shift: Any
    exp2f_table_bits: Any
    exp2f_table_size: Any


def _tiedtke_constants() -> _TiedtkeConstants:
    return _TiedtkeConstants(
        ALS,
        ALV,
        C2ES,
        C3IES,
        C3LES,
        C4IES,
        C4LES,
        C5IES,
        C5LES,
        CMFCMAX,
        CMFCMIN,
        CPD,
        CPRCON,
        ENTRMID,
        G,
        RCPD,
        RD,
        RV,
        TMELT,
        VTMPc1,
        ZBUO0,
        ZDNOPRC,
        ZRG,
        _EXP2F_C0,
        _EXP2F_C1,
        _EXP2F_C2,
        _EXP2F_INV_LN2_SCALED,
        _EXP2F_SHIFT,
        _EXP2F_TABLE_BITS,
        _EXP2F_TABLE_SIZE,
    )


@jax.custom_jvp
def _expf_values(x: Any, exp_table, constants) -> Any:
    x32 = jnp.asarray(x, dtype=jnp.float32)
    z = column_binary("mul", constants.exp2f_inv_ln2_scaled, x32.astype(jnp.float64))
    kd = column_barrier(column_binary("add", z, constants.exp2f_shift).astype(jnp.float64))
    ki = lax.bitcast_convert_type(kd, jnp.uint64)
    kd = column_binary("sub", kd, constants.exp2f_shift)
    r = column_binary("sub", z, kd)
    table_index = (ki & jnp.uint64(column_binary("sub", constants.exp2f_table_size, 1))).astype(jnp.int32)
    table_bits = exp_table[table_index]
    scale_bits = column_binary(
        "add",
        table_bits,
        jnp.left_shift(ki, jnp.uint64(column_binary("sub", 52, constants.exp2f_table_bits))),
    )
    s = lax.bitcast_convert_type(scale_bits, jnp.float64)
    z_poly = column_binary("add", column_binary("mul", constants.exp2f_c0, r), constants.exp2f_c1)
    r2 = column_binary("mul", r, r)
    y = column_binary("add", column_binary("mul", constants.exp2f_c2, r), jnp.float64(1.0))
    y = column_binary("add", column_binary("mul", z_poly, r2), y)
    return column_binary("mul", y, s).astype(jnp.float32)


def _tlucua_values(tt: Any, exp_table, constants) -> Any:
    """WRF ``tlucua`` expression used by repeated ``cuadjtq`` calls."""
    tt_arr = jnp.asarray(tt, dtype=jnp.float32)
    zcvm3 = jnp.where(column_binary("sub", tt_arr, constants.tmelt) > 0.0, constants.c3les, constants.c3ies)
    zcvm4 = jnp.where(column_binary("sub", tt_arr, constants.tmelt) > 0.0, constants.c4les, constants.c4ies)
    arg = column_binary("mul", zcvm3, column_binary("sub", tt_arr, constants.tmelt))
    arg = column_binary(
        "mul", arg, column_binary("div", jnp.float32(1.0), column_binary("sub", tt_arr, zcvm4))
    )
    return column_binary("mul", constants.c2es, _expf_values(arg, exp_table, constants))


def _tlucub_values(tt: Any, constants) -> Any:
    """WRF ``tlucub`` latent-heat correction lookup formula."""
    tt_arr = jnp.asarray(tt, dtype=jnp.float32)
    z5alvcp = column_binary("div", column_binary("mul", constants.c5les, constants.alv), constants.cpd)
    z5alscp = column_binary("div", column_binary("mul", constants.c5ies, constants.als), constants.cpd)
    zcvm4 = jnp.where(column_binary("sub", tt_arr, constants.tmelt) > 0.0, constants.c4les, constants.c4ies)
    zcvm5 = jnp.where(column_binary("sub", tt_arr, constants.tmelt) > 0.0, z5alvcp, z5alscp)
    inv = column_binary("div", jnp.float32(1.0), column_binary("sub", tt_arr, zcvm4))
    return column_binary("mul", zcvm5, inv**2)


def _tlucuc_values(tt: Any, constants) -> Any:
    """WRF ``tlucuc`` latent heat over heat-capacity lookup formula."""
    tt_arr = jnp.asarray(tt, dtype=jnp.float32)
    return jnp.where(
        column_binary("sub", tt_arr, constants.tmelt) > 0.0,
        column_binary("div", constants.alv, constants.cpd),
        column_binary("div", constants.als, constants.cpd),
    )


def _cuadjtq_values(
    pp: Any, pt_col: Any, pq_col: Any, ldflag: Any, kcall: int, exp_table, constants
) -> tuple[Any, Any]:
    """Adjust one vertical level for WRF ``cuadjtq``.

    Hot callers already work level by level.  Keeping this helper column-wide
    avoids updating an entire two-dimensional work array when only one level is
    physically changed.
    """
    pt_out = jnp.asarray(pt_col, dtype=jnp.float32)
    pq_out = jnp.asarray(pq_col, dtype=jnp.float32)
    pp_arr = jnp.asarray(pp, dtype=jnp.float32)
    flags = jnp.asarray(ldflag, dtype=bool)

    def condensation_column(pt_value: Any, pq_value: Any) -> Any:
        zqp = column_binary("div", jnp.float32(1.0), pp_arr)
        zqsat = jnp.minimum(
            jnp.float32(0.5), column_binary("mul", _tlucua_values(pt_value, exp_table, constants), zqp)
        )
        zcor = column_binary(
            "div",
            jnp.float32(1.0),
            column_binary("sub", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, zqsat)),
        )
        zqsat = column_binary("mul", zqsat, zcor)
        return column_binary(
            "div",
            column_binary("sub", pq_value, zqsat),
            column_binary(
                "add",
                jnp.float32(1.0),
                column_binary("mul", column_binary("mul", zqsat, zcor), _tlucub_values(pt_value, constants)),
            ),
        )

    def apply_delta(pt_value: Any, pq_value: Any, mask: Any, zcond: Any) -> tuple[Any, Any]:
        delta = jnp.where(mask, zcond, jnp.float32(0.0))
        return (
            column_binary("add", pt_value, column_binary("mul", _tlucuc_values(pt_value, constants), delta)),
            column_binary("sub", pq_value, delta),
        )

    if int(kcall) == 1:
        zcond = jnp.maximum(condensation_column(pt_out, pq_out), jnp.float32(0.0))
        pt_out, pq_out = apply_delta(pt_out, pq_out, flags, zcond)
        second_mask = flags & (zcond != 0.0)
        pt_out, pq_out = apply_delta(pt_out, pq_out, second_mask, condensation_column(pt_out, pq_out))
    elif int(kcall) == 2:
        zcond = jnp.minimum(condensation_column(pt_out, pq_out), jnp.float32(0.0))
        pt_out, pq_out = apply_delta(pt_out, pq_out, flags, zcond)
        second_mask = flags & (zcond != 0.0)
        pt_out, pq_out = apply_delta(pt_out, pq_out, second_mask, condensation_column(pt_out, pq_out))
    elif int(kcall) == 0:
        all_columns = jnp.ones_like(flags)
        pt_out, pq_out = apply_delta(pt_out, pq_out, all_columns, condensation_column(pt_out, pq_out))
        pt_out, pq_out = apply_delta(pt_out, pq_out, all_columns, condensation_column(pt_out, pq_out))
    elif int(kcall) == 4:
        all_columns = jnp.ones_like(flags)
        pt_out, pq_out = apply_delta(pt_out, pq_out, all_columns, condensation_column(pt_out, pq_out))
        pt_out, pq_out = apply_delta(pt_out, pq_out, all_columns, condensation_column(pt_out, pq_out))
    else:
        raise NotImplementedError(f"cuadjtq kcall={int(kcall)} is not present in WRF module_cu_tiedtke")
    return (pt_out, pq_out)


def _cuentr_values(
    kk_i,
    ptenh_next,
    paph_next,
    paph_py,
    pap_py,
    klwmin_arr,
    ldcum_arr,
    ktype_arr,
    kcbot_arr,
    kctop0_arr,
    pmfu_next_arr,
    pentr_arr,
    khmin_arr,
    pgeoh_py,
    zpbase_out,
    paph_kctop0,
    pgeoh_khmin,
    pgeoh_kctop0,
    constants,
) -> CuentrLevelResult:
    zrrho = column_binary("mul", constants.rd, ptenh_next)
    zrrho = column_binary("div", zrrho, paph_next)
    zdprho = column_binary("mul", column_binary("sub", paph_next, paph_py), constants.zrg)
    zpmid = column_binary("mul", jnp.float32(0.5), column_binary("add", zpbase_out, paph_kctop0))
    zentr = column_binary("mul", pentr_arr, pmfu_next_arr)
    zentr = column_binary("mul", zentr, zdprho)
    zentr = column_binary("mul", zentr, zrrho)
    llo1 = (kk_i < kcbot_arr) & ldcum_arr
    zdmfde_out = jnp.where(llo1, zentr, jnp.float32(0.0))
    zdmfen_tmp = jnp.zeros_like(zdmfde_out)
    llo2 = (
        llo1
        & (ktype_arr == 2)
        & ((column_binary("sub", zpbase_out, paph_py) < constants.zdnoprc) | (paph_py > zpmid))
    )
    zdmfen_tmp = jnp.where(llo2, zentr, zdmfen_tmp)
    iklwmin = jnp.maximum(klwmin_arr, column_binary("add", kctop0_arr, 2))
    llo2 = llo1 & (ktype_arr == 3) & ((kk_i >= iklwmin) | (pap_py > zpmid))
    zdmfen_tmp = jnp.where(llo2, zentr, zdmfen_tmp)
    llo_deep = llo1 & (ktype_arr == 1)
    zdmfen_out = jnp.where(llo_deep, zentr, zdmfen_tmp)
    zzmzk = column_binary("mul", -column_binary("sub", pgeoh_khmin, pgeoh_py), constants.zrg)
    ztmzk = column_binary("mul", -column_binary("sub", pgeoh_khmin, pgeoh_kctop0), constants.zrg)
    arg = column_binary(
        "mul", column_binary("mul", jnp.float32(3.1415), column_binary("div", zzmzk, ztmzk)), jnp.float32(0.5)
    )
    zorgde = column_binary(
        "div",
        column_binary("mul", column_binary("mul", jnp.tan(arg), jnp.float32(3.1415)), jnp.float32(0.5)),
        ztmzk,
    )
    zdprho_det = column_binary(
        "mul", column_binary("sub", paph_next, paph_py), column_binary("mul", constants.zrg, zrrho)
    )
    zodetr_val = column_binary("mul", jnp.minimum(zorgde, jnp.float32(0.001)), pmfu_next_arr)
    zodetr_val = column_binary("mul", zodetr_val, zdprho_det)
    detrain_mask = llo_deep & (kk_i <= khmin_arr) & (kk_i >= kctop0_arr) & (khmin_arr > kctop0_arr)
    zodetr_col = jnp.where(detrain_mask, zodetr_val, jnp.float32(0.0))
    return CuentrLevelResult(zpbase=zpbase_out, zdmfen=zdmfen_out, zdmfde=zdmfde_out, zodetr_col=zodetr_col)


@_expf_values.defjvp
def _expf_values_jvp(primals, tangents):
    x, table, constants = primals
    x_dot, _, _ = tangents
    y = _expf_values(x, table, constants)
    return (y, y * x_dot)


def _expf_fortran_compatible_primal(x: Any) -> Any:
    return _expf_values(x, _EXP2F_TABLE, _tiedtke_constants())


@jax.custom_jvp
def _expf_fortran_compatible(x: Any) -> Any:
    return _expf_fortran_compatible_primal(x)


@_expf_fortran_compatible.defjvp
def _expf_fortran_compatible_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
    (x,) = primals
    (x_dot,) = tangents
    y = _expf_fortran_compatible_primal(x)
    return y, y * x_dot


@dataclass(frozen=True)
class CuiniResult:
    pgeoh: Any
    ptenh: Any
    pqenh: Any
    pqsenh: Any
    klwmin: Any
    ptu: Any
    pqu: Any
    ptd: Any
    pqd: Any
    puu: Any
    pvu: Any
    pud: Any
    pvd: Any
    pmfu: Any
    pmfd: Any
    pmfus: Any
    pmfds: Any
    pmfuq: Any
    pmfdq: Any
    pdmfup: Any
    pdmfdp: Any
    pdpmel: Any
    plu: Any
    plude: Any
    klab: Any


@dataclass(frozen=True)
class CubaseResult:
    ptu: Any
    pqu: Any
    plu: Any
    puu: Any
    pvu: Any
    ldcum: Any
    kcbot: Any
    klab: Any


@dataclass(frozen=True)
class CubasmcResult:
    ldcum: Any
    ktype: Any
    klab: Any
    pmfu: Any
    pmfub: Any
    pentr: Any
    kcbot: Any
    ptu: Any
    pqu: Any
    plu: Any
    puu: Any
    pvu: Any
    pmfus: Any
    pmfuq: Any
    pmful: Any
    pdmfup: Any
    pmfuu: Any
    pmfuv: Any


@dataclass(frozen=True)
class CuentrResult:
    zpbase: Any
    zdmfen: Any
    zdmfde: Any
    zodetr: Any


@dataclass(frozen=True)
class CuentrLevelResult:
    zpbase: Any
    zdmfen: Any
    zdmfde: Any
    zodetr_col: Any


@dataclass(frozen=True)
class CududvResult:
    pvom: Any
    pvol: Any
    psdiss: Any


@dataclass(frozen=True)
class CudtdqResult:
    ptte: Any
    pqte: Any
    pcte: Any
    prsfc: Any
    pssfc: Any
    paprc: Any
    paprs: Any
    psheat: Any
    psrain: Any
    psevap: Any
    psmelt: Any


@dataclass(frozen=True)
class CudlfsResult:
    prfl: Any
    ptd: Any
    pqd: Any
    pud: Any
    pvd: Any
    pmfd: Any
    pmfds: Any
    pmfdq: Any
    pdmfdp: Any
    kdtop: Any
    lddraf: Any


@dataclass(frozen=True)
class CuddrafResult:
    prfl: Any
    ptd: Any
    pqd: Any
    pud: Any
    pvd: Any
    pmfd: Any
    pmfds: Any
    pmfdq: Any
    pdmfdp: Any


@dataclass(frozen=True)
class CuflxResult:
    ldcum: Any
    lddraf: Any
    ktype: Any
    pmfu: Any
    pmfd: Any
    pmfus: Any
    pmfds: Any
    pmfuq: Any
    pmfdq: Any
    pmful: Any
    plude: Any
    pdmfup: Any
    pdmfdp: Any
    prfl: Any
    prain: Any
    psfl: Any
    pdpmel: Any
    ktopm2: int


@dataclass(frozen=True)
class CuascResult:
    ldcum: Any
    phcbase: Any
    ktype: Any
    klab: Any
    ptu: Any
    pqu: Any
    plu: Any
    puu: Any
    pvu: Any
    pmfu: Any
    pmfub: Any
    pentr: Any
    pmfus: Any
    pmfuq: Any
    pmful: Any
    plude: Any
    pdmfup: Any
    kcbot: Any
    kctop: Any
    kctop0: Any
    kcum: int
    khmin: Any


CuascResult = jax.tree_util.register_dataclass(
    CuascResult,
    data_fields=(
        "ldcum",
        "phcbase",
        "ktype",
        "klab",
        "ptu",
        "pqu",
        "plu",
        "puu",
        "pvu",
        "pmfu",
        "pmfub",
        "pentr",
        "pmfus",
        "pmfuq",
        "pmful",
        "plude",
        "pdmfup",
        "kcbot",
        "kctop",
        "kctop0",
        "kcum",
        "khmin",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class _CuascLoopState:
    ldcum: Any
    ktype: Any
    klab_next: Any
    ptu: Any
    pqu: Any
    plu_next: Any
    puu: Any
    pvu: Any
    pmfu_next: Any
    pmfub: Any
    pentr: Any
    kcbot: Any
    kctop: Any
    kctop0: Any
    khmin: Any
    zmfuu: Any
    zmfuv: Any
    zbuoy: Any
    zoentr_value: Any
    zoentr_valid: Any
    base_pressure: Any


_CuascLoopState = jax.tree_util.register_dataclass(
    _CuascLoopState,
    data_fields=(
        "ldcum",
        "ktype",
        "klab_next",
        "ptu",
        "pqu",
        "plu_next",
        "puu",
        "pvu",
        "pmfu_next",
        "pmfub",
        "pentr",
        "kcbot",
        "kctop",
        "kctop0",
        "khmin",
        "zmfuu",
        "zmfuv",
        "zbuoy",
        "zoentr_value",
        "zoentr_valid",
        "base_pressure",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class TiecnvResult:
    pu: Any
    pv: Any
    pt: Any
    pqv: Any
    pqc: Any
    pqi: Any
    zprecc: Any
    ktype: Any


@dataclass(frozen=True)
class CuTiedtkeResult:
    raincv: Any
    pratec: Any
    cu_act_flag: Any
    rthcuten: Any
    rqvcuten: Any
    rqccuten: Any
    rqicuten: Any
    rucuten: Any
    rvcuten: Any


@dataclass(frozen=True)
class CumastrResult:
    ptte: Any
    pqte: Any
    pvom: Any
    pvol: Any
    prsfc: Any
    pssfc: Any
    paprc: Any
    paprs: Any
    ldcum: Any
    ktype: Any
    kcbot: Any
    kctop: Any
    ptu: Any
    pqu: Any
    plu: Any
    plude: Any
    pmfu: Any
    pmfd: Any
    prain: Any
    psrain: Any
    psevap: Any
    psheat: Any
    psdiss: Any
    psmelt: Any
    pcte: Any


def _register_result_dataclass(cls: Any) -> Any:
    return jax.tree_util.register_dataclass(
        cls,
        data_fields=tuple(cls.__dataclass_fields__),
        meta_fields=(),
    )


CuiniResult = _register_result_dataclass(CuiniResult)
CubaseResult = _register_result_dataclass(CubaseResult)
CubasmcResult = _register_result_dataclass(CubasmcResult)
CuentrResult = _register_result_dataclass(CuentrResult)
CuentrLevelResult = _register_result_dataclass(CuentrLevelResult)
CududvResult = _register_result_dataclass(CududvResult)
CudtdqResult = _register_result_dataclass(CudtdqResult)
CudlfsResult = _register_result_dataclass(CudlfsResult)
CuddrafResult = _register_result_dataclass(CuddrafResult)
CuflxResult = _register_result_dataclass(CuflxResult)
TiecnvResult = _register_result_dataclass(TiecnvResult)
CuTiedtkeResult = _register_result_dataclass(CuTiedtkeResult)
CumastrResult = _register_result_dataclass(CumastrResult)


def cu_tiedtke_column_core(
    u: Any,
    v: Any,
    t: Any,
    qv: Any,
    qc: Any,
    qi: Any,
    qvften: Any,
    qvpblten: Any,
    geopotential_height: Any,
    omega: Any,
    pressure: Any,
    pressure_interface: Any,
    surface_evap: Any,
    land_mask: Any,
    sigma: Any,
    dt: Any,
    *,
    column_count: int,
    level_count: int,
    row_size: int | None = None,
    use_cuasc_jit: bool | None = None,
    use_cuasc_scan: bool | None = None,
    use_cuasc_second_scan: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> TiecnvResult:
    """Run the Tiedtke mass-flux core on batched vertical columns.

    The public ``cu_tiedtke`` wrapper still preserves WRF's 3-D memory layout
    contract.  This helper is the production-core boundary: inputs are already
    packed as ``(column, level)`` arrays ordered top-to-bottom, so future
    production stages can avoid re-entering the WRF memory-layout wrapper.
    """

    column_count_i = int(column_count)
    level_count_i = int(level_count)
    return tiecnv(
        u,
        v,
        t,
        qv,
        qc,
        qi,
        qvften,
        qvpblten,
        geopotential_height,
        omega,
        pressure,
        pressure_interface,
        surface_evap,
        jnp.zeros((column_count_i,), dtype=jnp.float32),
        land_mask,
        jnp.zeros((column_count_i,), dtype=jnp.int32),
        column_count_i,
        level_count_i,
        level_count_i + 1,
        sigma,
        dt,
        row_size=row_size,
        use_cuasc_jit=use_cuasc_jit,
        use_cuasc_scan=use_cuasc_scan,
        use_cuasc_second_scan=use_cuasc_second_scan,
        use_row_conditional_cuadjtq=use_row_conditional_cuadjtq,
        rounding_token=rounding_token,
    )


def cu_tiedtke(
    dt: Any,
    itimestep: Any,
    stepcu: int,
    raincv: Any,
    pratec: Any,
    qfx: Any,
    znu: Any,
    u3d: Any,
    v3d: Any,
    w: Any,
    t3d: Any,
    qv3d: Any,
    qc3d: Any,
    qi3d: Any,
    pi3d: Any,
    rho3d: Any,
    qvften: Any,
    qvpblten: Any,
    dz8w: Any,
    pcps: Any,
    p8w: Any,
    xland: Any,
    cu_act_flag: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqicuten: Any,
    rucuten: Any,
    rvcuten: Any,
    f_qv: bool = True,
    f_qc: bool = True,
    f_qr: bool = True,
    f_qi: bool = True,
    f_qs: bool = True,
    use_cuasc_jit: bool | None = None,
    use_cuasc_scan: bool | None = None,
    use_cuasc_second_scan: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> CuTiedtkeResult:
    """Translate WRF ``cu_tiedtke`` outer driver."""

    del ids, ide, jds, jde, kds, kde, kme, f_qv, f_qr, f_qs
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)
    dt_arr = store(jnp.asarray(dt, dtype=jnp.float32))
    stepcu_i = int(stepcu)
    stepcu_arr = store(jnp.asarray(stepcu_i, dtype=jnp.float32))
    delt = store(dt_arr * stepcu_arr)
    rdelt = store(jnp.float32(1.0) / store(delt))

    raincv_out = jnp.asarray(raincv, dtype=jnp.float32)
    pratec_out = jnp.asarray(pratec, dtype=jnp.float32)
    qfx_arr = jnp.asarray(qfx, dtype=jnp.float32)
    znu_arr = jnp.asarray(znu, dtype=jnp.float32)
    u3d_arr = jnp.asarray(u3d, dtype=jnp.float32)
    v3d_arr = jnp.asarray(v3d, dtype=jnp.float32)
    w_arr = jnp.asarray(w, dtype=jnp.float32)
    t3d_arr = jnp.asarray(t3d, dtype=jnp.float32)
    qv3d_arr = jnp.asarray(qv3d, dtype=jnp.float32)
    qc3d_arr = jnp.asarray(qc3d, dtype=jnp.float32)
    qi3d_arr = jnp.asarray(qi3d, dtype=jnp.float32)
    pi3d_arr = jnp.asarray(pi3d, dtype=jnp.float32)
    rho3d_arr = jnp.asarray(rho3d, dtype=jnp.float32)
    qvften_arr = jnp.asarray(qvften, dtype=jnp.float32)
    qvpblten_arr = jnp.asarray(qvpblten, dtype=jnp.float32)
    dz8w_arr = jnp.asarray(dz8w, dtype=jnp.float32)
    pcps_arr = jnp.asarray(pcps, dtype=jnp.float32)
    p8w_arr = jnp.asarray(p8w, dtype=jnp.float32)
    xland_arr = jnp.asarray(xland, dtype=jnp.float32)
    cu_act_flag_out = jnp.asarray(cu_act_flag, dtype=jnp.int32)
    rthcuten_out = jnp.asarray(rthcuten, dtype=jnp.float32)
    rqvcuten_out = jnp.asarray(rqvcuten, dtype=jnp.float32)
    rqccuten_out = jnp.asarray(rqccuten, dtype=jnp.float32)
    rqicuten_out = jnp.asarray(rqicuten, dtype=jnp.float32)
    rucuten_out = jnp.asarray(rucuten, dtype=jnp.float32)
    rvcuten_out = jnp.asarray(rvcuten, dtype=jnp.float32)

    i0 = int(its) - int(ims)
    i1 = int(ite) - int(ims) + 1
    k0 = int(kts) - int(kms)
    k1 = int(kte) - int(kms) + 1
    ke1 = int(kte) - int(kms) + 2
    im = int(ite) - int(its) + 1
    kx = int(kte) - int(kts) + 1
    i_slice = slice(i0, i1)
    k_slice = slice(k0, k1)
    ke_slice = slice(k0, ke1)

    j0 = int(jts) - int(jms)
    j1 = int(jte) - int(jms) + 1
    j_slice = slice(j0, j1)
    jm = int(jte) - int(jts) + 1
    ncol = im * jm

    def flatten_mass(block: Any) -> Any:
        return jnp.transpose(block, (2, 0, 1)).reshape((ncol, kx))

    def flatten_edge(block: Any) -> Any:
        return jnp.transpose(block, (2, 0, 1)).reshape((ncol, kx + 1))

    def flatten_2d(block: Any) -> Any:
        return jnp.transpose(block, (1, 0)).reshape((ncol,))

    def unflatten_mass(block: Any) -> Any:
        return block.reshape((jm, im, kx)).transpose((1, 2, 0))

    def unflatten_2d(block: Any) -> Any:
        return block.reshape((jm, im)).T

    cu_act_flag_out = cu_act_flag_out.at[i_slice, j_slice].set(jnp.ones((im, jm), dtype=jnp.int32))

    dz_local = flatten_mass(dz8w_arr[i_slice, k_slice, j_slice])
    if kx > 1:
        # This fixed-size prefix avoids a late scan compilation observed in long
        # direct trajectories while preserving the WRF vertical accumulation order.
        previous = jnp.zeros((ncol,), dtype=jnp.float32)
        zi_columns = [previous]
        for kk in range(kx - 1):
            previous = store(previous + dz_local[:, kk])
            zi_columns.append(previous)
        zi = jnp.stack(zi_columns, axis=1)
        zl_core = store(zi[:, 1:kx] + zi[:, : kx - 1])
        zl_core = store(zl_core * jnp.float32(0.5))
        zl_top = store(jnp.float32(2.0) * zi[:, kx - 1])
        zl_top = store(zl_top - zl_core[:, kx - 2])[:, None]
        zl = jnp.concatenate((zl_core, zl_top), axis=1)
    else:
        zl = jnp.zeros((ncol, 1), dtype=jnp.float32)

    slimsk = jnp.abs(flatten_2d(xland_arr[i_slice, j_slice]) - jnp.float32(2.0)).astype(jnp.int32)
    w_edges = flatten_edge(w_arr[i_slice, ke_slice, j_slice])
    rho_local = flatten_mass(rho3d_arr[i_slice, k_slice, j_slice])
    dot = store(w_edges[:, :kx] + w_edges[:, 1 : kx + 1])
    dot = store(rho_local * dot)
    dot = store(G * dot)
    dot = store(-jnp.float32(0.5) * dot)

    u1 = flatten_mass(u3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    v1 = flatten_mass(v3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    t1 = flatten_mass(t3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    q1 = flatten_mass(qv3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    first_step = jnp.asarray(itimestep, dtype=jnp.int32) == jnp.asarray(1, dtype=jnp.int32)
    q1b_later = flatten_mass(qvften_arr[i_slice, k_slice, j_slice])[:, ::-1]
    q1bl_later = flatten_mass(qvpblten_arr[i_slice, k_slice, j_slice])[:, ::-1]
    q1b = jnp.where(first_step, jnp.zeros_like(q1b_later), q1b_later)
    q1bl = jnp.where(first_step, jnp.zeros_like(q1bl_later), q1bl_later)
    q2 = flatten_mass(qc3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    q3 = flatten_mass(qi3d_arr[i_slice, k_slice, j_slice])[:, ::-1]
    omg = dot[:, ::-1]
    ght = zl[:, ::-1]
    prsl = flatten_mass(pcps_arr[i_slice, k_slice, j_slice])[:, ::-1]
    prsi = flatten_edge(p8w_arr[i_slice, ke_slice, j_slice])[:, ::-1]
    sig1_col = znu_arr[k_slice][::-1]
    evap = flatten_2d(qfx_arr[i_slice, j_slice])

    tied = cu_tiedtke_column_core(
        u1,
        v1,
        t1,
        q1,
        q2,
        q3,
        q1b,
        q1bl,
        ght,
        omg,
        prsl,
        prsi,
        evap,
        slimsk,
        sig1_col,
        delt,
        column_count=ncol,
        level_count=kx,
        row_size=im,
        use_cuasc_jit=use_cuasc_jit,
        use_cuasc_scan=use_cuasc_scan,
        use_cuasc_second_scan=use_cuasc_second_scan,
        use_row_conditional_cuadjtq=use_row_conditional_cuadjtq,
        rounding_token=rounding_token,
    )

    rn = tied.zprecc
    full_local = (
        i0 == 0
        and j0 == 0
        and k0 == 0
        and i1 == int(raincv_out.shape[0])
        and j1 == int(raincv_out.shape[1])
        and k1 == int(rthcuten_out.shape[1])
    )
    t1_model = unflatten_mass(tied.pt[:, ::-1])
    q1_model = unflatten_mass(tied.pqv[:, ::-1])
    u1_model = unflatten_mass(tied.pu[:, ::-1])
    v1_model = unflatten_mass(tied.pv[:, ::-1])
    q2_model = unflatten_mass(tied.pqc[:, ::-1])
    q3_model = unflatten_mass(tied.pqi[:, ::-1])
    t3d_block = t3d_arr[i_slice, k_slice, j_slice]
    qv3d_block = qv3d_arr[i_slice, k_slice, j_slice]
    u3d_block = u3d_arr[i_slice, k_slice, j_slice]
    v3d_block = v3d_arr[i_slice, k_slice, j_slice]
    if full_local:
        raincv_direct = unflatten_2d(store(rn / stepcu_arr))
        pratec_denom = store(stepcu_arr * dt_arr)
        pratec_direct = unflatten_2d(store(rn / pratec_denom))
        rqccuten_direct = (
            store(store(q2_model - qc3d_arr[i_slice, k_slice, j_slice]) * rdelt)
            if bool(f_qc)
            else rqccuten_out
        )
        rqicuten_direct = (
            store(store(q3_model - qi3d_arr[i_slice, k_slice, j_slice]) * rdelt)
            if bool(f_qi)
            else rqicuten_out
        )
        rthcuten_direct = store(t1_model - t3d_block)
        rthcuten_direct = store(rthcuten_direct / store(pi3d_arr[i_slice, k_slice, j_slice]))
        rthcuten_direct = store(rthcuten_direct * rdelt)
        return CuTiedtkeResult(
            raincv=raincv_direct,
            pratec=pratec_direct,
            cu_act_flag=jnp.ones_like(cu_act_flag_out, dtype=jnp.int32),
            rthcuten=rthcuten_direct,
            rqvcuten=store(store(q1_model - qv3d_block) * rdelt),
            rqccuten=rqccuten_direct,
            rqicuten=rqicuten_direct,
            rucuten=store(store(u1_model - u3d_block) * rdelt),
            rvcuten=store(store(v1_model - v3d_block) * rdelt),
        )

    raincv_out = raincv_out.at[i_slice, j_slice].set(unflatten_2d(rn / stepcu_arr))
    pratec_out = pratec_out.at[i_slice, j_slice].set(unflatten_2d(rn / (stepcu_arr * dt_arr)))
    rthcuten_out = rthcuten_out.at[i_slice, k_slice, j_slice].set(
        (t1_model - t3d_block) / pi3d_arr[i_slice, k_slice, j_slice] * rdelt
    )
    rqvcuten_out = rqvcuten_out.at[i_slice, k_slice, j_slice].set((q1_model - qv3d_block) * rdelt)
    rucuten_out = rucuten_out.at[i_slice, k_slice, j_slice].set((u1_model - u3d_block) * rdelt)
    rvcuten_out = rvcuten_out.at[i_slice, k_slice, j_slice].set((v1_model - v3d_block) * rdelt)
    if bool(f_qc):
        rqccuten_out = rqccuten_out.at[i_slice, k_slice, j_slice].set((q2_model - qc3d_arr[i_slice, k_slice, j_slice]) * rdelt)
    if bool(f_qi):
        rqicuten_out = rqicuten_out.at[i_slice, k_slice, j_slice].set((q3_model - qi3d_arr[i_slice, k_slice, j_slice]) * rdelt)

    return CuTiedtkeResult(
        raincv=raincv_out,
        pratec=pratec_out,
        cu_act_flag=cu_act_flag_out,
        rthcuten=rthcuten_out,
        rqvcuten=rqvcuten_out,
        rqccuten=rqccuten_out,
        rqicuten=rqicuten_out,
        rucuten=rucuten_out,
        rvcuten=rvcuten_out,
    )


def tiecnv(
    pu: Any,
    pv: Any,
    pt: Any,
    pqv: Any,
    pqc: Any,
    pqi: Any,
    pqvf: Any,
    pqvbl: Any,
    poz: Any,
    pomg: Any,
    pap: Any,
    paph: Any,
    evap: Any,
    zprecc: Any,
    lndj: Any,
    ktype: Any,
    lq: int,
    km: int,
    km1: int,
    sig1: Any,
    dt: Any,
    row_size: int | None = None,
    use_cuasc_jit: bool | None = None,
    use_cuasc_scan: bool | None = None,
    use_cuasc_second_scan: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> TiecnvResult:
    """Translate WRF ``tiecnv`` interface around the Tiedtke mass-flux scheme."""

    lq_i = int(lq)
    km_i = int(km)
    km1_i = int(km1)
    ztmst = jnp.asarray(dt, dtype=jnp.float32)

    pu_out = jnp.asarray(pu, dtype=jnp.float32)
    pv_out = jnp.asarray(pv, dtype=jnp.float32)
    pt_arr = jnp.asarray(pt, dtype=jnp.float32)
    pqv_arr = jnp.asarray(pqv, dtype=jnp.float32)
    pqc_out = jnp.asarray(pqc, dtype=jnp.float32)
    pqi_out = jnp.asarray(pqi, dtype=jnp.float32)
    pqvf_arr = jnp.asarray(pqvf, dtype=jnp.float32)
    pqvbl_arr = jnp.asarray(pqvbl, dtype=jnp.float32)
    poz_arr = jnp.asarray(poz, dtype=jnp.float32)
    pomg_arr = jnp.asarray(pomg, dtype=jnp.float32)
    pap_arr = jnp.asarray(pap, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    evap_arr = jnp.asarray(evap, dtype=jnp.float32)
    zprecc_out = jnp.asarray(zprecc, dtype=jnp.float32)
    lndj_arr = jnp.asarray(lndj, dtype=jnp.int32)
    ktype_out = jnp.asarray(ktype, dtype=jnp.int32)
    sig1_arr = jnp.asarray(sig1, dtype=jnp.float32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    zeros = jnp.zeros((lq_i, km_i), dtype=jnp.float32)
    ptte = zeros
    pcte = zeros
    pvom = zeros
    pvol = zeros
    ztp1 = pt_arr
    zqp1_denom = store(jnp.float32(1.0) + pqv_arr)
    zqp1 = store(pqv_arr / zqp1_denom)
    pum1 = pu_out
    pvm1 = pv_out
    pverv = pomg_arr
    pgeo = store(G * poz_arr)
    zqsat = store(tlucua(store(ztp1)))
    zqsat = store(zqsat / store(pap_arr))
    zqsat = store(jnp.minimum(jnp.float32(0.5), zqsat))
    zqsat_denom = store(jnp.float32(1.0) - store(VTMPc1 * zqsat))
    zqsat = store(zqsat / zqsat_denom)
    pqte = store(pqvf_arr + pqvbl_arr)
    zqq = pqte

    zrain = jnp.zeros((lq_i,), dtype=jnp.float32)
    locum = jnp.zeros((lq_i,), dtype=bool)
    prsfc = jnp.zeros((lq_i,), dtype=jnp.float32)
    pssfc = jnp.zeros((lq_i,), dtype=jnp.float32)
    paprc = jnp.zeros((lq_i,), dtype=jnp.float32)
    paprs = jnp.zeros((lq_i,), dtype=jnp.float32)
    paprsm = jnp.zeros((lq_i,), dtype=jnp.float32)

    conv = cumastr_new(
        lq_i,
        km_i,
        km1_i,
        km_i - 1,
        ztp1,
        zqp1,
        pum1,
        pvm1,
        pverv,
        zqsat,
        evap_arr,
        ztmst,
        pap_arr,
        paph_arr,
        pgeo,
        ptte,
        pqte,
        pvom,
        pvol,
        prsfc,
        pssfc,
        paprc,
        paprsm,
        paprs,
        locum,
        ktype_out,
        jnp.zeros((lq_i,), dtype=jnp.int32),
        jnp.zeros((lq_i,), dtype=jnp.int32),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zrain,
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        pcte,
        sig1_arr,
        lndj_arr,
        row_size=row_size,
        use_cuasc_jit=use_cuasc_jit,
        use_cuasc_scan=use_cuasc_scan,
        use_cuasc_second_scan=use_cuasc_second_scan,
        use_row_conditional_cuadjtq=use_row_conditional_cuadjtq,
        rounding_token=rounding_token,
    )

    ptte = conv.ptte
    pqte = conv.pqte
    pvom = conv.pvom
    pvol = conv.pvol
    prsfc = conv.prsfc
    pssfc = conv.pssfc
    ktype_out = conv.ktype
    pcte = conv.pcte

    ztpp1 = store(ptte * ztmst)
    ztpp1 = store(pt_arr + ztpp1)
    liquid = ztpp1 >= T000
    ice = ztpp1 <= HGFR
    ztc = store(ztpp1 - T000)
    fliq_exponent = store(-jnp.float32(0.003102) * ztc)
    fliq_exponent = store(fliq_exponent * ztc)
    fliq_mixed = store(jnp.exp(fliq_exponent))
    fliq_mixed = store(jnp.float32(0.9941) * fliq_mixed)
    fliq_mixed = store(jnp.float32(0.0059) + fliq_mixed)
    fliq = jnp.where(liquid, jnp.float32(1.0), jnp.where(ice, jnp.float32(0.0), fliq_mixed))
    zalf = jnp.where(liquid, jnp.float32(0.0), ALF)
    feedback_mask = pcte > 0.0
    fice = store(jnp.float32(1.0) - fliq)
    cloud_feedback = store(fliq * pcte)
    cloud_feedback = store(cloud_feedback * ztmst)
    ice_feedback = store(fice * pcte)
    ice_feedback = store(ice_feedback * ztmst)
    pqc_out = store(pqc_out + jnp.where(feedback_mask, cloud_feedback, jnp.float32(0.0)))
    pqi_out = store(pqi_out + jnp.where(feedback_mask, ice_feedback, jnp.float32(0.0)))
    thermal_feedback = store(zalf * RCPD)
    thermal_feedback = store(thermal_feedback * fliq)
    thermal_feedback = store(thermal_feedback * pcte)
    ptte = store(ptte - jnp.where(feedback_mask, thermal_feedback, jnp.float32(0.0)))

    pt_out = store(ztp1 + store(ptte * ztmst))
    zqp1_delta = store(pqte - zqq)
    zqp1_delta = store(zqp1_delta * ztmst)
    zqp1 = store(zqp1 + zqp1_delta)
    pqv_out = store(zqp1 / store(jnp.float32(1.0) - zqp1))
    zprecc_out = store(prsfc + pssfc)
    zprecc_out = store(zprecc_out * ztmst)
    zprecc_out = store(jnp.maximum(jnp.float32(0.0), zprecc_out))
    if LMFDUDV:
        pu_out = store(pu_out + store(pvom * ztmst))
        pv_out = store(pv_out + store(pvol * ztmst))

    return TiecnvResult(pu=pu_out, pv=pv_out, pt=pt_out, pqv=pqv_out, pqc=pqc_out, pqi=pqi_out, zprecc=zprecc_out, ktype=ktype_out)


def tlucua(tt: Any) -> Any:
    """WRF ``tlucua`` saturation lookup formula."""
    return _tlucua_values(tt, _EXP2F_TABLE, _tiedtke_constants())


def _tlucua_cuadjtq(tt: Any) -> Any:
    """WRF ``tlucua`` expression used by repeated ``cuadjtq`` calls."""
    return _tlucua_values(tt, _EXP2F_TABLE, _tiedtke_constants())


def tlucub(tt: Any) -> Any:
    """WRF ``tlucub`` latent-heat correction lookup formula."""
    return _tlucub_values(tt, _tiedtke_constants())


def tlucuc(tt: Any) -> Any:
    """WRF ``tlucuc`` latent heat over heat-capacity lookup formula."""
    return _tlucuc_values(tt, _tiedtke_constants())


@partial(jax.jit, static_argnums=(4,))
def _cuadjtq_level(pp: Any, pt_col: Any, pq_col: Any, ldflag: Any, kcall: int) -> tuple[Any, Any]:
    """Adjust one vertical level for WRF ``cuadjtq``.

    Hot callers already work level by level.  Keeping this helper column-wide
    avoids updating an entire two-dimensional work array when only one level is
    physically changed.
    """
    return _cuadjtq_values(pp, pt_col, pq_col, ldflag, kcall, _EXP2F_TABLE, _tiedtke_constants())


def _cuadjtq_level_compact(
    pp: Any,
    pt_col: Any,
    pq_col: Any,
    ldflag: Any,
    kcall: int,
    *,
    capacity: int,
) -> tuple[Any, Any]:
    """Run level saturation adjustment on a fixed active-column subset.

    ``cuasc_new`` often has sparse active masks at each vertical level.  When
    the active count fits in ``capacity``, gather those columns, run the same
    per-column adjustment, and scatter them back.  If the active count is too
    large, fall back to the full-column path.
    """

    capacity_i = int(capacity)
    pp_arr = jnp.asarray(pp, dtype=jnp.float32)
    pt_arr = jnp.asarray(pt_col, dtype=jnp.float32)
    pq_arr = jnp.asarray(pq_col, dtype=jnp.float32)
    flags = jnp.asarray(ldflag, dtype=bool)
    ncol_i = int(pt_arr.shape[0])
    if capacity_i <= 0 or capacity_i >= ncol_i:
        return _cuadjtq_level(pp_arr, pt_arr, pq_arr, flags, kcall)

    active_count = jnp.sum(flags.astype(jnp.int32))
    slots = jnp.arange(capacity_i, dtype=jnp.int32)

    def compact(_: Any) -> tuple[Any, Any]:
        active_idx = jnp.nonzero(flags, size=capacity_i, fill_value=0)[0]
        slot_active = slots < active_count
        compact_pt, compact_pq = _cuadjtq_level(
            pp_arr[active_idx],
            pt_arr[active_idx],
            pq_arr[active_idx],
            slot_active,
            kcall,
        )
        scatter_idx = jnp.where(slot_active, active_idx, jnp.asarray(ncol_i, dtype=jnp.int32))
        pt_out = pt_arr.at[scatter_idx].set(compact_pt, mode="drop")
        pq_out = pq_arr.at[scatter_idx].set(compact_pq, mode="drop")
        return pt_out, pq_out

    def full(_: Any) -> tuple[Any, Any]:
        return _cuadjtq_level(pp_arr, pt_arr, pq_arr, flags, kcall)

    return lax.cond(active_count <= capacity_i, compact, full, operand=None)


def _cuadjtq_level_row_conditional(
    pp: Any,
    pt_col: Any,
    pq_col: Any,
    ldflag: Any,
    kcall: int,
    *,
    row_size: int | None,
    enabled: bool,
) -> tuple[Any, Any]:
    """Run ``cuadjtq`` only for WRF rows with active columns.

    The flattened Tiedtke production layout preserves WRF j-rows as contiguous
    chunks.  Several hot callers have sparse active masks; row-level
    ``lax.cond`` keeps inactive rows from evaluating the saturation-adjustment
    exponentials while preserving the original full-column result.
    """

    if not enabled or row_size is None:
        if _CUADJTQ_COMPACT_CAPACITY <= 0:
            return _cuadjtq_level(pp, pt_col, pq_col, ldflag, kcall)
        return _cuadjtq_level_compact(
            pp,
            pt_col,
            pq_col,
            ldflag,
            kcall,
            capacity=_CUADJTQ_COMPACT_CAPACITY,
        )
    pp_arr = jnp.asarray(pp, dtype=jnp.float32)
    pt_arr = jnp.asarray(pt_col, dtype=jnp.float32)
    pq_arr = jnp.asarray(pq_col, dtype=jnp.float32)
    flags = jnp.asarray(ldflag, dtype=bool)
    row_size_i = int(row_size)
    ncol_i = int(pt_arr.shape[0])
    if row_size_i <= 0 or row_size_i >= ncol_i or ncol_i % row_size_i != 0:
        if _CUADJTQ_COMPACT_CAPACITY <= 0:
            return _cuadjtq_level(pp_arr, pt_arr, pq_arr, flags, kcall)
        return _cuadjtq_level_compact(
            pp_arr,
            pt_arr,
            pq_arr,
            flags,
            kcall,
            capacity=_CUADJTQ_COMPACT_CAPACITY,
        )
    nrow_i = ncol_i // row_size_i

    def reshape_rows(value: Any) -> Any:
        return jnp.reshape(value, (nrow_i, row_size_i))

    def row_fn(row_pp: Any, row_pt: Any, row_pq: Any, row_flags: Any) -> tuple[Any, Any]:
        def active(args: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
            pp_i, pt_i, pq_i, flags_i = args
            if _CUADJTQ_COMPACT_CAPACITY <= 0:
                return _cuadjtq_level(pp_i, pt_i, pq_i, flags_i, kcall)
            return _cuadjtq_level_compact(
                pp_i,
                pt_i,
                pq_i,
                flags_i,
                kcall,
                capacity=_CUADJTQ_COMPACT_CAPACITY,
            )

        def inactive(args: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
            _, pt_i, pq_i, _ = args
            return pt_i, pq_i

        return lax.cond(jnp.any(row_flags), active, inactive, (row_pp, row_pt, row_pq, row_flags))

    pt_rows, pq_rows = jax.vmap(row_fn)(
        reshape_rows(pp_arr),
        reshape_rows(pt_arr),
        reshape_rows(pq_arr),
        reshape_rows(flags),
    )
    return jnp.reshape(pt_rows, pt_arr.shape), jnp.reshape(pq_rows, pq_arr.shape)


def cuadjtq(klon: int, klev: int, kk: int, pp: Any, pt: Any, pq: Any, ldflag: Any, kcall: int) -> tuple[Any, Any]:
    """Adjust Tiedtke cloud-ascent temperature and humidity.

    ``kk`` is 1-based, as in WRF.  ``pt`` and ``pq`` are returned after the
    in-place Fortran updates have been represented functionally.
    """

    del klon, klev
    level = int(kk) - 1
    pt_out = jnp.asarray(pt, dtype=jnp.float32)
    pq_out = jnp.asarray(pq, dtype=jnp.float32)
    pt_col, pq_col = _cuadjtq_level(pp, pt_out[:, level], pq_out[:, level], ldflag, kcall)
    pt_out = pt_out.at[:, level].set(pt_col)
    pq_out = pq_out.at[:, level].set(pq_col)
    return pt_out, pq_out


def _column_static_energy_step(next_temperature, inputs, parameters):
    """Advance the descending WRF dry-static-energy temperature recurrence."""
    temperature, geopotential, next_geopotential = inputs
    token, cpd, rcpd = parameters

    def store(value):
        value = jnp.asarray(value, jnp.float32)
        return column_barrier(value) if token is None else real_assignment(value, token)

    static_left = store(column_binary("mul", cpd, temperature))
    static_left = store(column_binary("add", static_left, geopotential))
    static_right = store(column_binary("mul", cpd, next_temperature))
    static_right = store(column_binary("add", static_right, next_geopotential))
    zzs = store(jnp.maximum(static_left, static_right))
    delta = store(column_binary("sub", zzs, geopotential))
    current = store(column_binary("mul", delta, rcpd))
    return current, current


def cuini(
    klon: int,
    klev: int,
    klevp1: int,
    klevm1: int,
    pten: Any,
    pqen: Any,
    pqsen: Any,
    puen: Any,
    pven: Any,
    pverv: Any,
    pgeo: Any,
    paph: Any,
    rounding_token: Any | None = None,
) -> CuiniResult:
    """Translate WRF ``cuini`` initialization for the Tiedtke scheme."""

    del klevp1
    klon_i = int(klon)
    klev_i = int(klev)
    klevm1_i = int(klevm1)

    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pverv_arr = jnp.asarray(pverv, dtype=jnp.float32)
    pgeo_arr = jnp.asarray(pgeo, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    zeros = jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
    izeros = jnp.zeros((klon_i, klev_i), dtype=jnp.int32)

    pgeoh = zeros
    ptenh = zeros
    pqenh = zeros
    pqsenh = zeros

    pgeo_delta = store(pgeo_arr[:, :-1] - pgeo_arr[:, 1:])
    pgeo_half_delta = store(pgeo_delta * jnp.float32(0.5))
    pgeoh_cols = store(pgeo_arr[:, 1:] + pgeo_half_delta)
    dry_static_left = store(CPD * pten_arr[:, :-1])
    dry_static_left = store(dry_static_left + pgeo_arr[:, :-1])
    dry_static_right = store(CPD * pten_arr[:, 1:])
    dry_static_right = store(dry_static_right + pgeo_arr[:, 1:])
    dry_static_max = store(jnp.maximum(dry_static_left, dry_static_right))
    ptenh_delta = store(dry_static_max - pgeoh_cols)
    ptenh_cols = store(ptenh_delta * RCPD)
    pqsenh_cols = store(pqsen_arr[:, :-1])
    adjust_pressure = store(paph_arr[:, 1:klev_i])
    adjust_ptenh = store(ptenh_cols)
    adjust_pqsenh = store(pqsenh_cols)
    ptenh_cols, pqsenh_cols = _cuadjtq_level(
        adjust_pressure,
        adjust_ptenh,
        adjust_pqsenh,
        jnp.ones_like(ptenh_cols, dtype=bool),
        0,
    )
    ptenh_cols = store(ptenh_cols)
    pqsenh_cols = store(pqsenh_cols)
    pqenh_min = store(jnp.minimum(pqen_arr[:, :-1], pqsen_arr[:, :-1]))
    pqsenh_delta = store(pqsenh_cols - pqsen_arr[:, :-1])
    pqenh_cols = store(pqenh_min + pqsenh_delta)
    pqenh_cols = store(jnp.maximum(pqenh_cols, jnp.float32(0.0)))
    pgeoh = pgeoh.at[:, 1:].set(pgeoh_cols)
    ptenh = ptenh.at[:, 1:].set(ptenh_cols)
    pqsenh = pqsenh.at[:, 1:].set(pqsenh_cols)
    pqenh = pqenh.at[:, 1:].set(pqenh_cols)

    bottom = klev_i - 1
    bottom_static = store(CPD * pten_arr[:, bottom])
    bottom_static = store(bottom_static + pgeo_arr[:, bottom])
    bottom_static = store(bottom_static - pgeoh[:, bottom])
    bottom_ptenh = store(bottom_static * RCPD)
    ptenh = ptenh.at[:, bottom].set(bottom_ptenh)
    pqenh = pqenh.at[:, bottom].set(pqen_arr[:, bottom])
    ptenh = ptenh.at[:, 0].set(pten_arr[:, 0])
    pqenh = pqenh.at[:, 0].set(pqen_arr[:, 0])
    pgeoh = pgeoh.at[:, 0].set(pgeo_arr[:, 0])

    klwmin = jnp.full((klon_i,), klev_i, dtype=jnp.int32)

    if klevm1_i > 1:
        levels = jnp.arange(klevm1_i - 1, 0, -1, dtype=jnp.int32)
        inputs = tuple(
            jnp.swapaxes(value, 0, 1)
            for value in (ptenh[:, levels], pgeoh[:, levels], pgeoh[:, levels + 1])
        )
        _, temperature_history = scan_columns(
            _column_static_energy_step,
            ptenh[:, klevm1_i],
            inputs,
            (rounding_token, CPD, RCPD),
        )
        ptenh = ptenh.at[:, levels].set(jnp.swapaxes(temperature_history, 0, 1))

    if klev_i > 2:
        # The original bottom-up strict comparison keeps the first minimum,
        # ignores NaNs and retains klev when no negative value is found.
        descending = pverv_arr[:, 2:klev_i][:, ::-1]
        negative = jnp.where(descending < 0, descending, jnp.float32(0.0))
        minimum_index = jnp.argmin(negative, axis=1).astype(jnp.int32)
        klwmin = klev_i - minimum_index

    ptu = zeros
    pqu = zeros
    ptd = zeros
    pqd = zeros
    puu = zeros
    pvu = zeros
    pud = zeros
    pvd = zeros
    pmfu = zeros
    pmfd = zeros
    pmfus = zeros
    pmfds = zeros
    pmfuq = zeros
    pmfdq = zeros
    pdmfup = zeros
    pdmfdp = zeros
    pdpmel = zeros
    plu = zeros
    plude = zeros
    klab = izeros

    wind_source = jnp.maximum(jnp.arange(klev_i, dtype=jnp.int32) - jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    ptu = ptenh
    ptd = ptenh
    pqu = pqenh
    pqd = pqenh
    puu = puen_arr[:, wind_source]
    pud = puu
    pvu = pven_arr[:, wind_source]
    pvd = pvu

    return CuiniResult(
        pgeoh=pgeoh,
        ptenh=ptenh,
        pqenh=pqenh,
        pqsenh=pqsenh,
        klwmin=klwmin,
        ptu=ptu,
        pqu=pqu,
        ptd=ptd,
        pqd=pqd,
        puu=puu,
        pvu=pvu,
        pud=pud,
        pvd=pvd,
        pmfu=pmfu,
        pmfd=pmfd,
        pmfus=pmfus,
        pmfds=pmfds,
        pmfuq=pmfuq,
        pmfdq=pmfdq,
        pdmfup=pdmfup,
        pdmfdp=pdmfdp,
        pdpmel=pdpmel,
        plu=plu,
        plude=plude,
        klab=klab,
    )


def cubase(
    klon: int,
    klev: int,
    klevp1: int,
    klevm1: int,
    ptenh: Any,
    pqenh: Any,
    pgeoh: Any,
    paph: Any,
    ptu: Any,
    pqu: Any,
    plu: Any,
    puen: Any,
    pven: Any,
    puu: Any,
    pvu: Any,
    klab: Any,
    row_size: int | None = None,
) -> CubaseResult:
    """Translate WRF ``cubase`` cloud-base initialization."""
    klon_i = int(klon)
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    klevm1_i = int(klevm1)
    bottom = klev_i - 1
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    pqenh_arr = jnp.asarray(pqenh, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    ptu_out = jnp.asarray(ptu, dtype=jnp.float32)
    pqu_out = jnp.asarray(pqu, dtype=jnp.float32)
    plu_out = jnp.asarray(plu, dtype=jnp.float32)
    puu_out = jnp.asarray(puu, dtype=jnp.float32)
    pvu_out = jnp.asarray(pvu, dtype=jnp.float32)
    klab_out = jnp.asarray(klab, dtype=jnp.int32)
    kcbot = jnp.full((klon_i,), klevm1_i, dtype=jnp.int32)
    ldcum = jnp.zeros((klon_i,), dtype=bool)
    ldbase = jnp.zeros((klon_i,), dtype=bool)
    klab_out = klab_out.at[:, bottom].set(1)
    surface_dp = paph_arr[:, klevp1_i - 1] - paph_arr[:, bottom]
    puu_out = puu_out.at[:, bottom].set(puen_arr[:, bottom] * surface_dp)
    pvu_out = pvu_out.at[:, bottom].set(pven_arr[:, bottom] * surface_dp)

    # Carry the adjacent level and wind budget; write each complete profile once.
    def column_step(carry, layer_inputs, parameters):
        (
            next_temperature,
            next_humidity,
            next_label,
            bottom_u,
            bottom_v,
            kcbot_acc,
            ldcum_acc,
            ldbase_acc,
        ) = carry
        (
            jk,
            pressure_next,
            pressure,
            geopotential_next,
            geopotential,
            temperature,
            humidity,
            environment_u,
            environment_v,
            ptu_level,
            pqu_level,
            plu_level,
            klab_level,
        ) = layer_inputs
        exp_table, constants, initial_buoyancy = parameters
        loflag = (next_label == 1) | ldcum_acc & (
            kcbot_acc == column_binary("add", jk, jnp.asarray(1, dtype=jnp.int32))
        )
        if LMFDUDV:
            dp = column_binary("sub", pressure_next, pressure)
            avg_mask = ~ldbase_acc
            bottom_u = column_binary(
                "add",
                bottom_u,
                jnp.where(
                    avg_mask, column_binary("mul", environment_u, dp), jnp.float32(0.0)
                ),
            )
            bottom_v = column_binary(
                "add",
                bottom_v,
                jnp.where(
                    avg_mask, column_binary("mul", environment_v, dp), jnp.float32(0.0)
                ),
            )
        pqu_col = jnp.where(loflag, next_humidity, pqu_level)
        ptu_col = jnp.where(
            loflag,
            column_binary(
                "mul",
                column_binary(
                    "sub",
                    column_binary(
                        "add",
                        column_binary("mul", constants.cpd, next_temperature),
                        geopotential_next,
                    ),
                    geopotential,
                ),
                constants.rcpd,
            ),
            ptu_level,
        )
        zqold_col = pqu_col
        ptu_col, pqu_col = _cuadjtq_values(
            pressure, ptu_col, pqu_col, loflag, 1, exp_table, constants
        )
        pqu_level = pqu_col
        ptu_level = ptu_col
        same_q = pqu_col == zqold_col
        zbuo = column_binary(
            "add",
            column_binary(
                "sub",
                column_binary(
                    "mul",
                    ptu_col,
                    column_binary(
                        "add",
                        jnp.float32(1.0),
                        column_binary("mul", constants.vtmpc1, pqu_col),
                    ),
                ),
                column_binary(
                    "mul",
                    temperature,
                    column_binary(
                        "add",
                        jnp.float32(1.0),
                        column_binary("mul", constants.vtmpc1, humidity),
                    ),
                ),
            ),
            initial_buoyancy,
        )
        condensed = loflag & ~same_q
        set_subcloud = loflag & same_q & (zbuo > 0.0)
        klab_col = jnp.where(set_subcloud, 1, jnp.where(condensed, 2, klab_level))
        klab_level = klab_col
        plu_level = column_binary(
            "add",
            plu_level,
            jnp.where(
                condensed, column_binary("sub", zqold_col, pqu_col), jnp.float32(0.0)
            ),
        )
        new_base = condensed & (zbuo > 0.0) & (next_label == 1)
        kcbot_acc = jnp.where(new_base, jk, kcbot_acc)
        ldcum_acc = jnp.where(new_base, True, ldcum_acc)
        ldbase_acc = jnp.where(new_base, True, ldbase_acc)
        return (
            (
                ptu_level,
                pqu_level,
                klab_level,
                bottom_u,
                bottom_v,
                kcbot_acc,
                ldcum_acc,
                ldbase_acc,
            ),
            (ptu_level, pqu_level, plu_level, klab_level),
        )

    jk = jnp.arange(klevm1_i, 1, -1, dtype=jnp.int32)
    py = jk - 1
    next_py = jk
    inputs = (
        jk,
        jnp.swapaxes(paph_arr[:, py + 1], 0, 1),
        jnp.swapaxes(paph_arr[:, py], 0, 1),
        jnp.swapaxes(pgeoh_arr[:, next_py], 0, 1),
        jnp.swapaxes(pgeoh_arr[:, py], 0, 1),
        jnp.swapaxes(ptenh_arr[:, py], 0, 1),
        jnp.swapaxes(pqenh_arr[:, py], 0, 1),
        jnp.swapaxes(puen_arr[:, py], 0, 1),
        jnp.swapaxes(pven_arr[:, py], 0, 1),
        jnp.swapaxes(ptu_out[:, py], 0, 1),
        jnp.swapaxes(pqu_out[:, py], 0, 1),
        jnp.swapaxes(plu_out[:, py], 0, 1),
        jnp.swapaxes(klab_out[:, py], 0, 1),
    )
    initial = (
        ptu_out[:, klevm1_i],
        pqu_out[:, klevm1_i],
        klab_out[:, klevm1_i],
        puu_out[:, bottom],
        pvu_out[:, bottom],
        kcbot,
        ldcum,
        ldbase,
    )
    current, history = scan_columns(
        column_step, initial, inputs, (_EXP2F_TABLE, _tiedtke_constants(), ZBUO0)
    )
    puu_out = puu_out.at[:, bottom].set(current[3])
    pvu_out = pvu_out.at[:, bottom].set(current[4])
    kcbot, ldcum, ldbase = current[5:]
    ptu_out = ptu_out.at[:, py].set(jnp.swapaxes(history[0], 0, 1))
    pqu_out = pqu_out.at[:, py].set(jnp.swapaxes(history[1], 0, 1))
    plu_out = plu_out.at[:, py].set(jnp.swapaxes(history[2], 0, 1))
    klab_out = klab_out.at[:, py].set(jnp.swapaxes(history[3], 0, 1))
    if LMFDUDV:
        ikb_idx = kcbot - 1
        paph_ikb = jnp.take_along_axis(paph_arr, ikb_idx[:, None], axis=1)[:, 0]
        zz = jnp.float32(1.0) / (paph_arr[:, klevp1_i - 1] - paph_ikb)
        puu_bottom = jnp.where(
            ldcum, puu_out[:, bottom] * zz, puen_arr[:, klevm1_i - 1]
        )
        pvu_bottom = jnp.where(
            ldcum, pvu_out[:, bottom] * zz, pven_arr[:, klevm1_i - 1]
        )
        puu_out = puu_out.at[:, bottom].set(puu_bottom)
        pvu_out = pvu_out.at[:, bottom].set(pvu_bottom)
    return CubaseResult(
        ptu=ptu_out,
        pqu=pqu_out,
        plu=plu_out,
        puu=puu_out,
        pvu=pvu_out,
        ldcum=ldcum,
        kcbot=kcbot,
        klab=klab_out,
    )


def cubasmc(
    klon: int,
    klev: int,
    klevm1: int,
    kk: int,
    pten: Any,
    pqen: Any,
    pqsen: Any,
    puen: Any,
    pven: Any,
    pverv: Any,
    pgeo: Any,
    pgeoh: Any,
    ldcum: Any,
    ktype: Any,
    klab: Any,
    pmfu: Any,
    pmfub: Any,
    pentr: Any,
    kcbot: Any,
    ptu: Any,
    pqu: Any,
    plu: Any,
    puu: Any,
    pvu: Any,
    pmfus: Any,
    pmfuq: Any,
    pmful: Any,
    pdmfup: Any,
    pmfuu: Any,
    pmfuv: Any,
) -> CubasmcResult:
    """Translate WRF ``cubasmc`` midlevel cloud-base initialization."""

    del klon, klev, klevm1
    kk_i, kk_py = _level_index_pair(kk)
    out_py = kk_i

    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pverv_arr = jnp.asarray(pverv, dtype=jnp.float32)
    pgeo_arr = jnp.asarray(pgeo, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    ldcum_out = jnp.asarray(ldcum, dtype=bool)
    ktype_out = jnp.asarray(ktype, dtype=jnp.int32)
    klab_out = jnp.asarray(klab, dtype=jnp.int32)
    pmfu_out = jnp.asarray(pmfu, dtype=jnp.float32)
    pmfub_out = jnp.asarray(pmfub, dtype=jnp.float32)
    pentr_out = jnp.asarray(pentr, dtype=jnp.float32)
    kcbot_out = jnp.asarray(kcbot, dtype=jnp.int32)
    ptu_out = jnp.asarray(ptu, dtype=jnp.float32)
    pqu_out = jnp.asarray(pqu, dtype=jnp.float32)
    plu_out = jnp.asarray(plu, dtype=jnp.float32)
    puu_out = jnp.asarray(puu, dtype=jnp.float32)
    pvu_out = jnp.asarray(pvu, dtype=jnp.float32)
    pmfus_out = jnp.asarray(pmfus, dtype=jnp.float32)
    pmfuq_out = jnp.asarray(pmfuq, dtype=jnp.float32)
    pmful_out = jnp.asarray(pmful, dtype=jnp.float32)
    pdmfup_out = jnp.asarray(pdmfup, dtype=jnp.float32)
    pmfuu_out = jnp.asarray(pmfuu, dtype=jnp.float32)
    pmfuv_out = jnp.asarray(pmfuv, dtype=jnp.float32)

    mask = (~ldcum_out) & (klab_out[:, out_py] == 0) & (pqen_arr[:, kk_py] > jnp.float32(0.80) * pqsen_arr[:, kk_py])
    ptu_col = (CPD * pten_arr[:, kk_py] + pgeo_arr[:, kk_py] - pgeoh_arr[:, out_py]) * RCPD
    pqu_col = pqen_arr[:, kk_py]
    zzzmb = jnp.maximum(CMFCMIN, -pverv_arr[:, kk_py] / G)
    zzzmb = jnp.minimum(zzzmb, CMFCMAX)

    ptu_out = ptu_out.at[:, out_py].set(jnp.where(mask, ptu_col, ptu_out[:, out_py]))
    pqu_out = pqu_out.at[:, out_py].set(jnp.where(mask, pqu_col, pqu_out[:, out_py]))
    plu_out = plu_out.at[:, out_py].set(jnp.where(mask, jnp.float32(0.0), plu_out[:, out_py]))
    pmfub_out = jnp.where(mask, zzzmb, pmfub_out)
    pmfu_out = pmfu_out.at[:, out_py].set(jnp.where(mask, pmfub_out, pmfu_out[:, out_py]))
    pmfus_col = pmfub_out * (CPD * ptu_out[:, out_py] + pgeoh_arr[:, out_py])
    pmfuq_col = pmfub_out * pqu_out[:, out_py]
    pmfus_out = pmfus_out.at[:, out_py].set(jnp.where(mask, pmfus_col, pmfus_out[:, out_py]))
    pmfuq_out = pmfuq_out.at[:, out_py].set(jnp.where(mask, pmfuq_col, pmfuq_out[:, out_py]))
    pmful_out = pmful_out.at[:, out_py].set(jnp.where(mask, jnp.float32(0.0), pmful_out[:, out_py]))
    pdmfup_out = pdmfup_out.at[:, out_py].set(jnp.where(mask, jnp.float32(0.0), pdmfup_out[:, out_py]))
    kcbot_out = jnp.where(mask, kk_i, kcbot_out)
    klab_out = klab_out.at[:, out_py].set(jnp.where(mask, 1, klab_out[:, out_py]))
    ktype_out = jnp.where(mask, 3, ktype_out)
    pentr_out = jnp.where(mask, ENTRMID, pentr_out)

    if LMFDUDV:
        puu_out = puu_out.at[:, out_py].set(jnp.where(mask, puen_arr[:, kk_py], puu_out[:, out_py]))
        pvu_out = pvu_out.at[:, out_py].set(jnp.where(mask, pven_arr[:, kk_py], pvu_out[:, out_py]))
        pmfuu_out = jnp.where(mask, pmfub_out * puu_out[:, out_py], pmfuu_out)
        pmfuv_out = jnp.where(mask, pmfub_out * pvu_out[:, out_py], pmfuv_out)

    return CubasmcResult(
        ldcum=ldcum_out,
        ktype=ktype_out,
        klab=klab_out,
        pmfu=pmfu_out,
        pmfub=pmfub_out,
        pentr=pentr_out,
        kcbot=kcbot_out,
        ptu=ptu_out,
        pqu=pqu_out,
        plu=plu_out,
        puu=puu_out,
        pvu=pvu_out,
        pmfus=pmfus_out,
        pmfuq=pmfuq_out,
        pmful=pmful_out,
        pdmfup=pdmfup_out,
        pmfuu=pmfuu_out,
        pmfuv=pmfuv_out,
    )


def cuentr_new_level(
    kk: int,
    ptenh: Any,
    paph: Any,
    pap: Any,
    pgeoh: Any,
    klwmin: Any,
    ldcum: Any,
    ktype: Any,
    kcbot: Any,
    kctop0: Any,
    pmfu_next: Any,
    pentr: Any,
    khmin: Any,
) -> CuentrLevelResult:
    """Compute one ``cuentr_new`` level with vector inputs and outputs."""
    kk_i, kk_py = _level_index_pair(kk)
    kk_next_py = kk_i
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    pap_arr = jnp.asarray(pap, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    klwmin_arr = jnp.asarray(klwmin, dtype=jnp.int32)
    ldcum_arr = jnp.asarray(ldcum, dtype=bool)
    ktype_arr = jnp.asarray(ktype, dtype=jnp.int32)
    kcbot_arr = jnp.asarray(kcbot, dtype=jnp.int32)
    kctop0_arr = jnp.asarray(kctop0, dtype=jnp.int32)
    pmfu_next_arr = jnp.asarray(pmfu_next, dtype=jnp.float32)
    pentr_arr = jnp.asarray(pentr, dtype=jnp.float32)
    khmin_arr = jnp.asarray(khmin, dtype=jnp.int32)
    kcbot_idx = kcbot_arr - 1
    kctop0_idx = kctop0_arr - 1
    khmin_idx = khmin_arr - 1
    return _cuentr_values(
        kk_i,
        ptenh_arr[:, kk_next_py],
        paph_arr[:, kk_next_py],
        paph_arr[:, kk_py],
        pap_arr[:, kk_py],
        klwmin_arr,
        ldcum_arr,
        ktype_arr,
        kcbot_arr,
        kctop0_arr,
        pmfu_next_arr,
        pentr_arr,
        khmin_arr,
        pgeoh_arr[:, kk_py],
        jnp.take_along_axis(paph_arr, kcbot_idx[:, None], axis=1)[:, 0],
        jnp.take_along_axis(paph_arr, kctop0_idx[:, None], axis=1)[:, 0],
        jnp.take_along_axis(pgeoh_arr, khmin_idx[:, None], axis=1)[:, 0],
        jnp.take_along_axis(pgeoh_arr, kctop0_idx[:, None], axis=1)[:, 0],
        _tiedtke_constants(),
    )


def cuentr_new(
    klon: int,
    klev: int,
    klevp1: int,
    kk: int,
    ptenh: Any,
    paph: Any,
    pap: Any,
    pgeoh: Any,
    klwmin: Any,
    ldcum: Any,
    ktype: Any,
    kcbot: Any,
    kctop0: Any,
    zpbase: Any,
    pmfu: Any,
    pentr: Any,
    zdmfen: Any,
    zdmfde: Any,
    zodetr: Any,
    khmin: Any,
) -> CuentrResult:
    """Translate WRF ``cuentr_new`` for one 1-based vertical level ``kk``."""

    del klon, klev, klevp1
    kk_next_py, kk_py = _level_index_pair(kk)
    zodetr_out = jnp.asarray(zodetr, dtype=jnp.float32)
    pmfu_arr = jnp.asarray(pmfu, dtype=jnp.float32)
    level = cuentr_new_level(
        kk,
        ptenh,
        paph,
        pap,
        pgeoh,
        klwmin,
        ldcum,
        ktype,
        kcbot,
        kctop0,
        pmfu_arr[:, kk_next_py],
        pentr,
        khmin,
    )
    zpbase_out = level.zpbase
    zdmfen_out = level.zdmfen
    zdmfde_out = level.zdmfde
    zodetr_col = level.zodetr_col
    zodetr_out = zodetr_out.at[:, kk_py].set(zodetr_col)

    return CuentrResult(zpbase=zpbase_out, zdmfen=zdmfen_out, zdmfde=zdmfde_out, zodetr=zodetr_out)


def cududv(
    klon: int,
    klev: int,
    klevp1: int,
    ktopm2: int,
    ktype: Any,
    kcbot: Any,
    paph: Any,
    ldcum: Any,
    puen: Any,
    pven: Any,
    pvom: Any,
    pvol: Any,
    puu: Any,
    pud: Any,
    pvu: Any,
    pvd: Any,
    pmfu: Any,
    pmfd: Any,
    psdiss: Any,
    rounding_token: Any | None = None,
) -> CududvResult:
    """Translate WRF ``cududv`` momentum tendency update."""

    del klevp1
    klon_i = int(klon)
    klev_i = int(klev)
    ktopm2_arr = jnp.asarray(ktopm2, dtype=jnp.int32)
    ktopm2_col = jnp.broadcast_to(ktopm2_arr, (klon_i,))

    ktype_arr = jnp.asarray(ktype, dtype=jnp.int32)
    kcbot_arr = jnp.asarray(kcbot, dtype=jnp.int32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    ldcum_arr = jnp.asarray(ldcum, dtype=bool)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pvom_out = jnp.asarray(pvom, dtype=jnp.float32)
    pvol_out = jnp.asarray(pvol, dtype=jnp.float32)
    puu_arr = jnp.asarray(puu, dtype=jnp.float32)
    pud_arr = jnp.asarray(pud, dtype=jnp.float32)
    pvu_arr = jnp.asarray(pvu, dtype=jnp.float32)
    pvd_arr = jnp.asarray(pvd, dtype=jnp.float32)
    pmfu_arr = jnp.asarray(pmfu, dtype=jnp.float32)
    pmfd_arr = jnp.asarray(pmfd, dtype=jnp.float32)
    psdiss_out = jnp.asarray(psdiss, dtype=jnp.float32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    levels = jnp.arange(1, klev_i + 1, dtype=jnp.int32)
    py_idx = levels - jnp.asarray(1, dtype=jnp.int32)
    ik_idx = jnp.maximum(levels - jnp.asarray(2, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    active = (levels[None, :] >= ktopm2_col[:, None]) & ldcum_arr[:, None]
    zmfuu = jnp.where(active, pmfu_arr[:, py_idx] * (puu_arr[:, py_idx] - puen_arr[:, ik_idx]), jnp.float32(0.0))
    zmfuv = jnp.where(active, pmfu_arr[:, py_idx] * (pvu_arr[:, py_idx] - pven_arr[:, ik_idx]), jnp.float32(0.0))
    zmfdu = jnp.where(active, pmfd_arr[:, py_idx] * (pud_arr[:, py_idx] - puen_arr[:, ik_idx]), jnp.float32(0.0))
    zmfdv = jnp.where(active, pmfd_arr[:, py_idx] * (pvd_arr[:, py_idx] - pven_arr[:, ik_idx]), jnp.float32(0.0))

    above_base = active & (levels[None, :] > kcbot_arr[:, None])
    # WRF skips this extrapolation when there is no sub-cloud momentum flux.
    # Its inactive cloud-base index can be outside the mass-level array.
    kcbot_idx = jnp.where(jnp.any(above_base, axis=1), kcbot_arr - 1, 0)
    paph_ikb = jnp.take_along_axis(paph_arr, kcbot_idx[:, None], axis=1)[:, 0]
    pressure_span = jnp.where(above_base, paph_arr[:, klev_i, None] - paph_ikb[:, None], jnp.float32(1.0))
    zzp = (paph_arr[:, klev_i, None] - paph_arr[:, py_idx]) / pressure_span
    zzp = jnp.where(ktype_arr[:, None] == 3, zzp**2, zzp)
    base_zmfuu = jnp.take_along_axis(zmfuu, kcbot_idx[:, None], axis=1)
    base_zmfuv = jnp.take_along_axis(zmfuv, kcbot_idx[:, None], axis=1)
    base_zmfdu = jnp.take_along_axis(zmfdu, kcbot_idx[:, None], axis=1)
    base_zmfdv = jnp.take_along_axis(zmfdv, kcbot_idx[:, None], axis=1)
    zmfuu = jnp.where(above_base, base_zmfuu * zzp, zmfuu)
    zmfuv = jnp.where(above_base, base_zmfuv * zzp, zmfuv)
    zmfdu = jnp.where(above_base, base_zmfdu * zzp, zmfdu)
    zmfdv = jnp.where(above_base, base_zmfdv * zzp, zmfdv)

    denom = paph_arr[:, 1 : klev_i + 1] - paph_arr[:, :klev_i]
    du_flux_mid = zmfuu[:, 1:klev_i] - zmfuu[:, : klev_i - 1] + zmfdu[:, 1:klev_i] - zmfdu[:, : klev_i - 1]
    dv_flux_mid = zmfuv[:, 1:klev_i] - zmfuv[:, : klev_i - 1] + zmfdv[:, 1:klev_i] - zmfdv[:, : klev_i - 1]
    du_flux_last = zmfuu[:, klev_i - 1 : klev_i] + zmfdu[:, klev_i - 1 : klev_i]
    dv_flux_last = zmfuv[:, klev_i - 1 : klev_i] + zmfdv[:, klev_i - 1 : klev_i]
    du_flux = jnp.concatenate((du_flux_mid, du_flux_last), axis=1)
    dv_flux = jnp.concatenate((dv_flux_mid, dv_flux_last), axis=1)
    last_level = levels[None, :] == jnp.asarray(klev_i, dtype=jnp.int32)
    zdudt = jnp.where(last_level, -(G / denom) * du_flux, (G / denom) * du_flux)
    zdvdt = jnp.where(last_level, -(G / denom) * dv_flux, (G / denom) * dv_flux)
    zdiss_inc = store(puen_arr[:, py_idx] * du_flux)
    zdiss_inc = store(zdiss_inc + store(pven_arr[:, py_idx] * dv_flux))
    zdiss_inc = jnp.where(last_level, -zdiss_inc, zdiss_inc)
    pvom_out = pvom_out + jnp.where(active, zdudt, jnp.float32(0.0))
    pvol_out = pvol_out + jnp.where(active, zdvdt, jnp.float32(0.0))
    zdiss = store(jnp.sum(jnp.where(active, zdiss_inc, jnp.float32(0.0)), axis=1))
    psdiss_out = store(psdiss_out + store(jnp.sum(zdiss)))
    return CududvResult(pvom=pvom_out, pvol=pvol_out, psdiss=psdiss_out)


def cudlfs(
    klon: int,
    klev: int,
    klevp1: int,
    ptenh: Any,
    pqenh: Any,
    puen: Any,
    pven: Any,
    pgeoh: Any,
    paph: Any,
    ptu: Any,
    pqu: Any,
    puu: Any,
    pvu: Any,
    ldcum: Any,
    kcbot: Any,
    kctop: Any,
    pmfub: Any,
    prfl: Any,
    ptd: Any,
    pqd: Any,
    pud: Any,
    pvd: Any,
    pmfd: Any,
    pmfds: Any,
    pmfdq: Any,
    pdmfdp: Any,
    row_size: int | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
) -> CudlfsResult:
    """Translate WRF ``cudlfs`` level-of-free-sinking search."""

    klon_i = int(klon)
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    pqenh_arr = jnp.asarray(pqenh, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    ptu_arr = jnp.asarray(ptu, dtype=jnp.float32)
    pqu_arr = jnp.asarray(pqu, dtype=jnp.float32)
    puu_arr = jnp.asarray(puu, dtype=jnp.float32)
    pvu_arr = jnp.asarray(pvu, dtype=jnp.float32)
    ldcum_arr = jnp.asarray(ldcum, dtype=bool)
    kcbot_arr = jnp.asarray(kcbot, dtype=jnp.int32)
    kctop_arr = jnp.asarray(kctop, dtype=jnp.int32)
    pmfub_arr = jnp.asarray(pmfub, dtype=jnp.float32)
    prfl_out = jnp.asarray(prfl, dtype=jnp.float32)
    ptd_out = jnp.asarray(ptd, dtype=jnp.float32)
    pqd_out = jnp.asarray(pqd, dtype=jnp.float32)
    pud_out = jnp.asarray(pud, dtype=jnp.float32)
    pvd_out = jnp.asarray(pvd, dtype=jnp.float32)
    pmfd_out = jnp.asarray(pmfd, dtype=jnp.float32)
    pmfds_out = jnp.asarray(pmfds, dtype=jnp.float32)
    pmfdq_out = jnp.asarray(pmfdq, dtype=jnp.float32)
    pdmfdp_out = jnp.asarray(pdmfdp, dtype=jnp.float32)

    lddraf = jnp.zeros((klon_i,), dtype=bool)
    kdtop = jnp.full((klon_i,), klevp1_i, dtype=jnp.int32)
    if not LMFDD:
        return CudlfsResult(
            prfl=prfl_out,
            ptd=ptd_out,
            pqd=pqd_out,
            pud=pud_out,
            pvd=pvd_out,
            pmfd=pmfd_out,
            pmfds=pmfds_out,
            pmfdq=pmfdq_out,
            pdmfdp=pdmfdp_out,
            kdtop=kdtop,
            lddraf=lddraf,
        )

    ke = klev_i - 3
    if ke >= 3:
        # Only the rain budget and trigger state recur; profiles are emitted once.
        def column_step(carry, layer_inputs, parameters):
            prfl_acc, lddraf_acc, kdtop_acc = carry
            (
                jk,
                pressure,
                temperature,
                humidity,
                up_temperature,
                up_humidity,
                geopotential,
                up_u,
                up_v,
                environment_u,
                environment_v,
                ptd_level,
                pqd_level,
                pud_level,
                pvd_level,
                pmfd_level,
                pmfds_level,
                pmfdq_level,
                pdmfdp_level,
                ldcum_arr,
                kcbot_arr,
                kctop_arr,
                pmfub_arr,
            ) = layer_inputs
            exp_table, constants, cmfdeps = parameters
            llo2 = ldcum_arr & (prfl_acc > 0.0) & ~lddraf_acc & (jk < kcbot_arr) & (jk > kctop_arr)
            ztenwb_col, zqenwb_col = _cuadjtq_values(
                pressure, temperature, humidity, llo2, 2, exp_table, constants
            )
            zttest = column_binary(
                "mul", jnp.float32(0.5), column_binary("add", up_temperature, ztenwb_col)
            )
            zqtest = column_binary(
                "mul", jnp.float32(0.5), column_binary("add", up_humidity, zqenwb_col)
            )
            zbuo = column_binary(
                "sub",
                column_binary(
                    "mul",
                    zttest,
                    column_binary(
                        "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, zqtest)
                    ),
                ),
                column_binary(
                    "mul",
                    temperature,
                    column_binary(
                        "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, humidity)
                    ),
                ),
            )
            zcond = column_binary("sub", humidity, zqenwb_col)
            zmftop = column_binary("mul", -cmfdeps, pmfub_arr)
            trigger = (
                llo2
                & (zbuo < 0.0)
                & (
                    prfl_acc
                    > column_binary("mul", column_binary("mul", jnp.float32(10.0), zmftop), zcond)
                )
            )
            kdtop_acc = jnp.where(trigger, jk, kdtop_acc)
            lddraf_acc = jnp.where(trigger, True, lddraf_acc)
            ptd_level = jnp.where(trigger, zttest, ptd_level)
            pqd_level = jnp.where(trigger, zqtest, pqd_level)
            pmfd_level = jnp.where(trigger, zmftop, pmfd_level)
            pmfds_col = column_binary(
                "mul",
                pmfd_level,
                column_binary("add", column_binary("mul", constants.cpd, ptd_level), geopotential),
            )
            pmfdq_col = column_binary("mul", pmfd_level, pqd_level)
            pdmfdp_col = column_binary(
                "mul", column_binary("mul", jnp.float32(-0.5), pmfd_level), zcond
            )
            pmfds_level = jnp.where(trigger, pmfds_col, pmfds_level)
            pmfdq_level = jnp.where(trigger, pmfdq_col, pmfdq_level)
            pdmfdp_level = jnp.where(trigger, pdmfdp_col, pdmfdp_level)
            prfl_acc = column_binary("add", prfl_acc, jnp.where(trigger, pdmfdp_col, jnp.float32(0.0)))
            if LMFDUDV:
                wind_mask = llo2 & (pmfd_level < 0.0)
                pud_col = column_binary(
                    "mul", jnp.float32(0.5), column_binary("add", up_u, environment_u)
                )
                pvd_col = column_binary(
                    "mul", jnp.float32(0.5), column_binary("add", up_v, environment_v)
                )
                pud_level = jnp.where(wind_mask, pud_col, pud_level)
                pvd_level = jnp.where(wind_mask, pvd_col, pvd_level)
            return (
                (prfl_acc, lddraf_acc, kdtop_acc),
                (
                    ptd_level,
                    pqd_level,
                    pud_level,
                    pvd_level,
                    pmfd_level,
                    pmfds_level,
                    pmfdq_level,
                    pdmfdp_level,
                ),
            )

        jk = jnp.arange(3, ke + 1, dtype=jnp.int32)
        py = jk - 1
        inputs = (
            jk,
            jnp.swapaxes(paph_arr[:, py], 0, 1),
            jnp.swapaxes(ptenh_arr[:, py], 0, 1),
            jnp.swapaxes(pqenh_arr[:, py], 0, 1),
            jnp.swapaxes(ptu_arr[:, py], 0, 1),
            jnp.swapaxes(pqu_arr[:, py], 0, 1),
            jnp.swapaxes(pgeoh_arr[:, py], 0, 1),
            jnp.swapaxes(puu_arr[:, py], 0, 1),
            jnp.swapaxes(pvu_arr[:, py], 0, 1),
            jnp.swapaxes(puen_arr[:, py - 1], 0, 1),
            jnp.swapaxes(pven_arr[:, py - 1], 0, 1),
            jnp.swapaxes(ptd_out[:, py], 0, 1),
            jnp.swapaxes(pqd_out[:, py], 0, 1),
            jnp.swapaxes(pud_out[:, py], 0, 1),
            jnp.swapaxes(pvd_out[:, py], 0, 1),
            jnp.swapaxes(pmfd_out[:, py], 0, 1),
            jnp.swapaxes(pmfds_out[:, py], 0, 1),
            jnp.swapaxes(pmfdq_out[:, py], 0, 1),
            jnp.swapaxes(pdmfdp_out[:, py - 1], 0, 1),
            jnp.swapaxes(jnp.broadcast_to(ldcum_arr[:, None], (klon_i, jk.size)), 0, 1),
            jnp.swapaxes(jnp.broadcast_to(kcbot_arr[:, None], (klon_i, jk.size)), 0, 1),
            jnp.swapaxes(jnp.broadcast_to(kctop_arr[:, None], (klon_i, jk.size)), 0, 1),
            jnp.swapaxes(jnp.broadcast_to(pmfub_arr[:, None], (klon_i, jk.size)), 0, 1),
        )
        current, history = scan_columns(
            column_step,
            (prfl_out, lddraf, kdtop),
            inputs,
            (_EXP2F_TABLE, _tiedtke_constants(), CMFDEPS),
        )
        prfl_out, lddraf, kdtop = current
        ptd_out = ptd_out.at[:, py].set(jnp.swapaxes(history[0], 0, 1))
        pqd_out = pqd_out.at[:, py].set(jnp.swapaxes(history[1], 0, 1))
        pud_out = pud_out.at[:, py].set(jnp.swapaxes(history[2], 0, 1))
        pvd_out = pvd_out.at[:, py].set(jnp.swapaxes(history[3], 0, 1))
        pmfd_out = pmfd_out.at[:, py].set(jnp.swapaxes(history[4], 0, 1))
        pmfds_out = pmfds_out.at[:, py].set(jnp.swapaxes(history[5], 0, 1))
        pmfdq_out = pmfdq_out.at[:, py].set(jnp.swapaxes(history[6], 0, 1))
        pdmfdp_out = pdmfdp_out.at[:, py - 1].set(jnp.swapaxes(history[7], 0, 1))

    return CudlfsResult(
        prfl=prfl_out,
        ptd=ptd_out,
        pqd=pqd_out,
        pud=pud_out,
        pvd=pvd_out,
        pmfd=pmfd_out,
        pmfds=pmfds_out,
        pmfdq=pmfdq_out,
        pdmfdp=pdmfdp_out,
        kdtop=kdtop,
        lddraf=lddraf,
    )


def _column_downdraft_wind_step(previous, inputs, minimum_mass):
    """Transport downdraft momentum in the original vertical order."""
    previous_u, previous_v, previous_mass = previous
    entrainment, detrainment, eligible, mass, environment_u, environment_v, old_u, old_v = inputs
    active = eligible & (mass < jnp.float32(0.0))
    momentum_u = (
        column_binary("mul", previous_mass, previous_u, separate=True)
        + entrainment * environment_u - detrainment * previous_u
    )
    momentum_v = (
        column_binary("mul", previous_mass, previous_v, separate=True)
        + entrainment * environment_v - detrainment * previous_v
    )
    inverse_mass = column_binary("div", jnp.float32(1.0), jnp.minimum(-minimum_mass, mass))
    u = jnp.where(active, momentum_u * inverse_mass, old_u)
    v = jnp.where(active, momentum_v * inverse_mass, old_v)
    return (u, v, mass), (u, v)


def cuddraf(
    klon: int,
    klev: int,
    klevp1: int,
    ptenh: Any,
    pqenh: Any,
    puen: Any,
    pven: Any,
    pgeoh: Any,
    paph: Any,
    prfl: Any,
    lddraf: Any,
    ptd: Any,
    pqd: Any,
    pud: Any,
    pvd: Any,
    pmfd: Any,
    pmfds: Any,
    pmfdq: Any,
    pdmfdp: Any,
    row_size: int | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
) -> CuddrafResult:
    """Translate WRF ``cuddraf`` moist downdraft descent."""
    klon_i = int(klon)
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    pqenh_arr = jnp.asarray(pqenh, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    prfl_out = jnp.asarray(prfl, dtype=jnp.float32)
    lddraf_arr = jnp.asarray(lddraf, dtype=bool)
    ptd_out = jnp.asarray(ptd, dtype=jnp.float32)
    pqd_out = jnp.asarray(pqd, dtype=jnp.float32)
    pud_out = jnp.asarray(pud, dtype=jnp.float32)
    pvd_out = jnp.asarray(pvd, dtype=jnp.float32)
    pmfd_out = jnp.asarray(pmfd, dtype=jnp.float32)
    pmfds_out = jnp.asarray(pmfds, dtype=jnp.float32)
    pmfdq_out = jnp.asarray(pmfdq, dtype=jnp.float32)
    pdmfdp_out = jnp.asarray(pdmfdp, dtype=jnp.float32)
    itopde = klev_i - 2

    # Descent decisions use only the preceding layer and the top mass flux.
    def column_step(carry, layer_inputs, parameters):
        (
            prfl_acc,
            previous_ptd,
            previous_pqd,
            previous_pmfd,
            previous_pmfds,
            previous_pmfdq,
            top_mass_flux,
        ) = carry
        (
            jk,
            previous_temperature,
            previous_pressure,
            pressure,
            surface_pressure,
            top_pressure,
            previous_geopotential,
            previous_humidity,
            geopotential,
            temperature,
            humidity,
            ptd_level,
            pqd_level,
            pmfd_level,
            pmfds_level,
            pmfdq_level,
            pdmfdp_level,
            lddraf_arr,
        ) = layer_inputs
        exp_table, constants, entrdd, top_level = parameters
        llo2 = lddraf_arr & (previous_pmfd < 0.0)
        zentr = column_binary(
            "mul",
            column_binary(
                "div",
                column_binary(
                    "mul",
                    column_binary("mul", column_binary("mul", entrdd, previous_pmfd), constants.rd),
                    previous_temperature,
                ),
                column_binary("mul", constants.g, previous_pressure),
            ),
            column_binary("sub", pressure, previous_pressure),
        )
        zdmfen = jnp.where(llo2, zentr, jnp.float32(0.0))
        zdmfde = jnp.where(llo2, zentr, jnp.float32(0.0))
        tail = jk > jnp.asarray(top_level, dtype=jnp.int32)
        zdmfde_tail = column_binary(
            "div",
            column_binary("mul", top_mass_flux, column_binary("sub", pressure, previous_pressure)),
            column_binary("sub", surface_pressure, top_pressure),
        )
        zdmfen = jnp.where(tail & llo2, jnp.float32(0.0), zdmfen)
        zdmfde = jnp.where(tail & llo2, zdmfde_tail, zdmfde)
        pmfd_col = column_binary("sub", column_binary("add", previous_pmfd, zdmfen), zdmfde)
        zseen = column_binary(
            "mul",
            column_binary(
                "add",
                column_binary("mul", constants.cpd, previous_temperature),
                previous_geopotential,
            ),
            zdmfen,
        )
        zqeen = column_binary("mul", previous_humidity, zdmfen)
        zsdde = column_binary(
            "mul",
            column_binary(
                "add", column_binary("mul", constants.cpd, previous_ptd), previous_geopotential
            ),
            zdmfde,
        )
        zqdde = column_binary("mul", previous_pqd, zdmfde)
        zmfdsk = column_binary("sub", column_binary("add", previous_pmfds, zseen), zsdde)
        zmfdqk = column_binary("sub", column_binary("add", previous_pmfdq, zqeen), zqdde)
        inv_mass = column_binary(
            "div",
            jnp.float32(1.0),
            jnp.minimum(column_binary("mul", jnp.float32(-1), constants.cmfcmin), pmfd_col),
        )
        pqd_col = column_binary("mul", zmfdqk, inv_mass)
        ptd_col = column_binary(
            "mul",
            column_binary("sub", column_binary("mul", zmfdsk, inv_mass), geopotential),
            constants.rcpd,
        )
        ptd_col = jnp.minimum(jnp.float32(400.0), jnp.maximum(jnp.float32(100.0), ptd_col))
        zcond_before = pqd_col
        ptd_col, pqd_col = _cuadjtq_values(
            pressure, ptd_col, pqd_col, llo2, 2, exp_table, constants
        )
        pmfd_level = jnp.where(llo2, pmfd_col, pmfd_level)
        pqd_level = jnp.where(llo2, pqd_col, pqd_level)
        ptd_level = jnp.where(llo2, ptd_col, ptd_level)
        zcond = column_binary("sub", zcond_before, pqd_col)
        zbuo = column_binary(
            "sub",
            column_binary(
                "mul",
                ptd_col,
                column_binary(
                    "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, pqd_col)
                ),
            ),
            column_binary(
                "mul",
                temperature,
                column_binary(
                    "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, humidity)
                ),
            ),
        )
        kill = llo2 & ((zbuo >= 0.0) | (prfl_acc <= column_binary("mul", pmfd_level, zcond)))
        pmfd_level = jnp.where(kill, jnp.float32(0.0), pmfd_level)
        pmfds_col = column_binary(
            "mul",
            column_binary("add", column_binary("mul", constants.cpd, ptd_col), geopotential),
            pmfd_level,
        )
        pmfdq_col = column_binary("mul", pqd_col, pmfd_level)
        zdmfdp = column_binary("mul", column_binary("mul", jnp.float32(-1), pmfd_level), zcond)
        pmfds_level = jnp.where(llo2, pmfds_col, pmfds_level)
        pmfdq_level = jnp.where(llo2, pmfdq_col, pmfdq_level)
        pdmfdp_level = jnp.where(llo2, zdmfdp, pdmfdp_level)
        prfl_acc = column_binary("add", prfl_acc, jnp.where(llo2, zdmfdp, jnp.float32(0.0)))
        top_mass_flux = jnp.where(jk == top_level, pmfd_level, top_mass_flux)
        return (
            (prfl_acc, ptd_level, pqd_level, pmfd_level, pmfds_level, pmfdq_level, top_mass_flux),
            (
                ptd_level,
                pqd_level,
                pmfd_level,
                pmfds_level,
                pmfdq_level,
                pdmfdp_level,
                zdmfen,
                zdmfde,
                llo2,
            ),
        )

    jk = jnp.arange(3, klev_i + 1, dtype=jnp.int32)
    py = jk - 1
    prev_py = py - 1
    top_py = itopde - 1
    inputs = (
        jk,
        jnp.swapaxes(ptenh_arr[:, prev_py], 0, 1),
        jnp.swapaxes(paph_arr[:, prev_py], 0, 1),
        jnp.swapaxes(paph_arr[:, py], 0, 1),
        jnp.swapaxes(jnp.broadcast_to(paph_arr[:, klevp1_i - 1][:, None], (klon_i, jk.size)), 0, 1),
        jnp.swapaxes(jnp.broadcast_to(paph_arr[:, top_py][:, None], (klon_i, jk.size)), 0, 1),
        jnp.swapaxes(pgeoh_arr[:, prev_py], 0, 1),
        jnp.swapaxes(pqenh_arr[:, prev_py], 0, 1),
        jnp.swapaxes(pgeoh_arr[:, py], 0, 1),
        jnp.swapaxes(ptenh_arr[:, py], 0, 1),
        jnp.swapaxes(pqenh_arr[:, py], 0, 1),
        jnp.swapaxes(ptd_out[:, py], 0, 1),
        jnp.swapaxes(pqd_out[:, py], 0, 1),
        jnp.swapaxes(pmfd_out[:, py], 0, 1),
        jnp.swapaxes(pmfds_out[:, py], 0, 1),
        jnp.swapaxes(pmfdq_out[:, py], 0, 1),
        jnp.swapaxes(pdmfdp_out[:, prev_py], 0, 1),
        jnp.swapaxes(jnp.broadcast_to(lddraf_arr[:, None], (klon_i, jk.size)), 0, 1),
    )
    current, history = scan_columns(
        column_step,
        (
            prfl_out,
            ptd_out[:, 1],
            pqd_out[:, 1],
            pmfd_out[:, 1],
            pmfds_out[:, 1],
            pmfdq_out[:, 1],
            pmfd_out[:, top_py],
        ),
        inputs,
        (_EXP2F_TABLE, _tiedtke_constants(), ENTRDD, jnp.int32(itopde)),
    )
    prfl_out = current[0]
    ptd_out = ptd_out.at[:, py].set(jnp.swapaxes(history[0], 0, 1))
    pqd_out = pqd_out.at[:, py].set(jnp.swapaxes(history[1], 0, 1))
    pmfd_out = pmfd_out.at[:, py].set(jnp.swapaxes(history[2], 0, 1))
    pmfds_out = pmfds_out.at[:, py].set(jnp.swapaxes(history[3], 0, 1))
    pmfdq_out = pmfdq_out.at[:, py].set(jnp.swapaxes(history[4], 0, 1))
    pdmfdp_out = pdmfdp_out.at[:, prev_py].set(jnp.swapaxes(history[5], 0, 1))
    # Winds do not feed the descent; preserve their profile-wise recurrence.
    if LMFDUDV and klev_i > 2:
        wind_inputs = (
            *history[-3:],
            jnp.swapaxes(pmfd_out[:, 2:klev_i], 0, 1),
            jnp.swapaxes(puen_arr[:, 1:klev_i - 1], 0, 1),
            jnp.swapaxes(pven_arr[:, 1:klev_i - 1], 0, 1),
            jnp.swapaxes(pud_out[:, 2:klev_i], 0, 1),
            jnp.swapaxes(pvd_out[:, 2:klev_i], 0, 1),
        )
        _, wind_history = semantic_call(
            lambda initial, inputs: scan_columns(_column_downdraft_wind_step, initial, inputs, CMFCMIN),
            (pud_out[:, 1], pvd_out[:, 1], jnp.asarray(pmfd, dtype=jnp.float32)[:, 1]),
            wind_inputs,
        )
        pud_out = pud_out.at[:, 2:klev_i].set(jnp.swapaxes(wind_history[0], 0, 1))
        pvd_out = pvd_out.at[:, 2:klev_i].set(jnp.swapaxes(wind_history[1], 0, 1))
    return CuddrafResult(
        prfl=prfl_out,
        ptd=ptd_out,
        pqd=pqd_out,
        pud=pud_out,
        pvd=pvd_out,
        pmfd=pmfd_out,
        pmfds=pmfds_out,
        pmfdq=pmfdq_out,
        pdmfdp=pdmfdp_out,
    )


def _column_evaporation_step(zpsubcl, inputs, parameters):
    """Advance one layer of the ordered below-cloud evaporation budget."""
    sig1, eligible, pressure_lower, pressure, saturation, humidity, pdmfup = inputs
    token, zcucov, zcons2, cevapcu1, cevapcu2 = parameters

    def store(value):
        value = jnp.asarray(value, jnp.float32)
        return column_barrier(value) if token is None else real_assignment(value, token)

    active = eligible & (zpsubcl > jnp.float32(1.0e-20))
    # No-rain columns bypass this square root in the Fortran branch.
    zrfl = store(jnp.where(active, zpsubcl, jnp.float32(1.0)))
    sig1_root = store(sqrt_for_wrf(sig1))
    evaporation_root = store(column_binary("mul", cevapcu2, sig1_root))
    evaporation_root = store(sqrt_for_wrf(evaporation_root))
    cevapcu = store(column_binary("mul", cevapcu1, evaporation_root))
    zrnew_root = store(column_binary("div", zrfl, zcucov))
    zrnew_root = store(sqrt_for_wrf(zrnew_root))
    pressure_depth = store(column_binary("sub", pressure_lower, pressure))
    humidity_deficit = store(column_binary("sub", saturation, humidity))
    humidity_deficit = store(jnp.maximum(jnp.float32(0.0), humidity_deficit))
    evaporation_loss = store(column_binary("mul", cevapcu, pressure_depth))
    evaporation_loss = store(column_binary("mul", evaporation_loss, humidity_deficit))
    zrnew_root = store(column_binary("sub", zrnew_root, evaporation_loss))
    zrnew_root = store(jnp.maximum(jnp.float32(0.0), zrnew_root))
    zrnew = store(column_binary("mul", zrnew_root, zrnew_root))
    zrnew = store(column_binary("mul", zrnew, zcucov))
    zrmin_humidity = store(column_binary("mul", jnp.float32(0.8), saturation))
    zrmin_humidity = store(column_binary("sub", zrmin_humidity, humidity))
    zrmin_humidity = store(jnp.maximum(jnp.float32(0.0), zrmin_humidity))
    zrmin_loss = store(column_binary("mul", zcucov, zrmin_humidity))
    zrmin_loss = store(column_binary("mul", zrmin_loss, zcons2))
    zrmin_loss = store(column_binary("mul", zrmin_loss, pressure_depth))
    zrmin = store(column_binary("sub", zrfl, zrmin_loss))
    zrnew = store(jnp.maximum(zrnew, zrmin))
    zrfln = store(jnp.maximum(zrnew, jnp.float32(0.0)))
    zdrfl = store(column_binary("sub", zrfln, zrfl))
    zdrfl = store(jnp.minimum(jnp.float32(0.0), zdrfl))
    delta = jnp.where(active, zdrfl, jnp.float32(0.0))
    return jnp.where(active, zrfln, zpsubcl), store(column_binary("add", pdmfup, delta))


def cuflx(
    klon: int,
    klev: int,
    klevp1: int,
    pqen: Any,
    pqsen: Any,
    ptenh: Any,
    pqenh: Any,
    paph: Any,
    pgeoh: Any,
    kcbot: Any,
    kctop: Any,
    kdtop: Any,
    ktype: Any,
    lddraf: Any,
    ldcum: Any,
    pmfu: Any,
    pmfd: Any,
    pmfus: Any,
    pmfds: Any,
    pmfuq: Any,
    pmfdq: Any,
    pmful: Any,
    plude: Any,
    pdmfup: Any,
    pdmfdp: Any,
    prfl: Any,
    prain: Any,
    pten: Any,
    psfl: Any,
    pdpmel: Any,
    ztmst: Any,
    sig1: Any,
    row_size: int | None = None,
    rounding_token: Any | None = None,
) -> CuflxResult:
    """Translate WRF ``cuflx`` final convective flux adjustment."""

    del klon
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    pqenh_arr = jnp.asarray(pqenh, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    kcbot_arr = jnp.asarray(kcbot, dtype=jnp.int32)
    kctop_arr = jnp.asarray(kctop, dtype=jnp.int32)
    kdtop_arr = jnp.asarray(kdtop, dtype=jnp.int32)
    ktype_out = jnp.asarray(ktype, dtype=jnp.int32)
    lddraf_out = jnp.asarray(lddraf, dtype=bool)
    ldcum_out = jnp.asarray(ldcum, dtype=bool)
    pmfu_out = jnp.asarray(pmfu, dtype=jnp.float32)
    pmfd_out = jnp.asarray(pmfd, dtype=jnp.float32)
    pmfus_out = jnp.asarray(pmfus, dtype=jnp.float32)
    pmfds_out = jnp.asarray(pmfds, dtype=jnp.float32)
    pmfuq_out = jnp.asarray(pmfuq, dtype=jnp.float32)
    pmfdq_out = jnp.asarray(pmfdq, dtype=jnp.float32)
    pmful_out = jnp.asarray(pmful, dtype=jnp.float32)
    plude_out = jnp.asarray(plude, dtype=jnp.float32)
    pdmfup_out = jnp.asarray(pdmfup, dtype=jnp.float32)
    pdmfdp_out = jnp.asarray(pdmfdp, dtype=jnp.float32)
    prfl_out = jnp.asarray(prfl, dtype=jnp.float32)
    prain_out = jnp.asarray(prain, dtype=jnp.float32)
    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    psfl_out = jnp.asarray(psfl, dtype=jnp.float32)
    pdpmel_out = jnp.asarray(pdpmel, dtype=jnp.float32)
    sig1_arr = jnp.asarray(sig1, dtype=jnp.float32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    ztmst_arr = store(jnp.asarray(ztmst, dtype=jnp.float32))
    zcons1_denom = store(ALF * G)
    zcons1_denom = store(zcons1_denom * ztmst_arr)
    zcons1 = store(CPD / zcons1_denom)
    zcons2_denom = store(G * ztmst_arr)
    zcons2 = store(jnp.float32(1.0) / zcons2_denom)
    zcucov = store(jnp.float32(0.05))
    ztmelp2 = TMELT + jnp.float32(2.0)

    prfl_out = jnp.zeros_like(prfl_out)
    psfl_out = jnp.zeros_like(psfl_out)
    prain_out = jnp.zeros_like(prain_out)
    if not LMFSCV:
        shallow = ktype_out == 2
        ldcum_out = jnp.where(shallow, False, ldcum_out)
        lddraf_out = jnp.where(shallow, False, lddraf_out)
    itop = _row_min_broadcast(kctop_arr, row_size)
    lddraf_out = jnp.where((~ldcum_out) | (kdtop_arr < kctop_arr), False, lddraf_out)
    ktype_out = jnp.where(ldcum_out, ktype_out, 0)
    ktopm2 = itop - 2
    ktopm2_start = jnp.min(ktopm2)
    loop_start = 1 if _is_traced(ktopm2_start) else int(jnp.asarray(ktopm2_start).item())

    if loop_start >= 1:
        jk = jnp.arange(loop_start, klev_i + 1, dtype=jnp.int32)
        py = jk - 1
        level_active = jk[None, :] >= jnp.broadcast_to(ktopm2, ldcum_out.shape)[:, None]
        active_cloud = level_active & ldcum_out[:, None] & (jk[None, :] >= kctop_arr[:, None] - 1)
        lddraf_level = level_active & lddraf_out[:, None] & (jk[None, :] >= kdtop_arr[:, None])
        dry_static = CPD * ptenh_arr[:, py] + pgeoh_arr[:, py]
        pmfus_new = jnp.where(active_cloud, pmfus_out[:, py] - pmfu_out[:, py] * dry_static, jnp.float32(0.0))
        pmfuq_new = jnp.where(active_cloud, pmfuq_out[:, py] - pmfu_out[:, py] * pqenh_arr[:, py], jnp.float32(0.0))
        pmfus_out = pmfus_out.at[:, py].set(jnp.where(level_active, pmfus_new, pmfus_out[:, py]))
        pmfuq_out = pmfuq_out.at[:, py].set(jnp.where(level_active, pmfuq_new, pmfuq_out[:, py]))
        pmfds_level = pmfds_out[:, py] - pmfd_out[:, py] * dry_static
        pmfdq_level = pmfdq_out[:, py] - pmfd_out[:, py] * pqenh_arr[:, py]
        downdraft = active_cloud & lddraf_level
        pmfd_out = pmfd_out.at[:, py].set(jnp.where(level_active, jnp.where(downdraft, pmfd_out[:, py], jnp.float32(0.0)), pmfd_out[:, py]))
        pmfds_out = pmfds_out.at[:, py].set(jnp.where(level_active, jnp.where(downdraft, pmfds_level, jnp.float32(0.0)), pmfds_out[:, py]))
        pmfdq_out = pmfdq_out.at[:, py].set(jnp.where(level_active, jnp.where(downdraft, pmfdq_level, jnp.float32(0.0)), pmfdq_out[:, py]))
        inactive = level_active & ~active_cloud
        pmfu_out = pmfu_out.at[:, py].set(jnp.where(inactive, jnp.float32(0.0), pmfu_out[:, py]))
        pmful_out = pmful_out.at[:, py].set(jnp.where(inactive, jnp.float32(0.0), pmful_out[:, py]))
        previous_py = jnp.arange(max(loop_start, 2) - 2, klev_i - 1, dtype=jnp.int32)
        offset = max(0, 2 - loop_start)
        inactive_previous = inactive[:, offset:]
        reset_previous = (level_active & ~downdraft)[:, offset:]
        pdmfdp_out = pdmfdp_out.at[:, previous_py].set(jnp.where(reset_previous, jnp.float32(0.0), pdmfdp_out[:, previous_py]))
        pdmfup_out = pdmfup_out.at[:, previous_py].set(jnp.where(inactive_previous, jnp.float32(0.0), pdmfup_out[:, previous_py]))
        pdmfdp_out = pdmfdp_out.at[:, previous_py].set(jnp.where(inactive_previous, jnp.float32(0.0), pdmfdp_out[:, previous_py]))
        plude_out = plude_out.at[:, previous_py].set(jnp.where(inactive_previous, jnp.float32(0.0), plude_out[:, previous_py]))
    else:

        def cuflx_prepare_body(loop_index: Any, carry: tuple[Any, ...]) -> tuple[Any, ...]:
            pmfu_acc, pmfd_acc, pmfus_acc, pmfds_acc, pmfuq_acc, pmfdq_acc, pmful_acc, plude_acc, pdmfup_acc, pdmfdp_acc = carry
            jk = jnp.asarray(loop_start, dtype=jnp.int32) + jnp.asarray(loop_index, dtype=jnp.int32)
            py = jk - jnp.asarray(1, dtype=jnp.int32)
            prev_py = jnp.maximum(py - jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
            has_prev = py > 0
            level_active = jk >= ktopm2
            active_cloud = level_active & ldcum_out & (jk >= kctop_arr - 1)
            lddraf_level = level_active & lddraf_out & (jk >= kdtop_arr)
            dry_static = CPD * ptenh_arr[:, py] + pgeoh_arr[:, py]
            pmfus_new = jnp.where(active_cloud, pmfus_acc[:, py] - pmfu_acc[:, py] * dry_static, jnp.float32(0.0))
            pmfuq_new = jnp.where(active_cloud, pmfuq_acc[:, py] - pmfu_acc[:, py] * pqenh_arr[:, py], jnp.float32(0.0))
            pmfus_acc = pmfus_acc.at[:, py].set(jnp.where(level_active, pmfus_new, pmfus_acc[:, py]))
            pmfuq_acc = pmfuq_acc.at[:, py].set(jnp.where(level_active, pmfuq_new, pmfuq_acc[:, py]))
            pmfds_level = pmfds_acc[:, py] - pmfd_acc[:, py] * dry_static
            pmfdq_level = pmfdq_acc[:, py] - pmfd_acc[:, py] * pqenh_arr[:, py]
            pmfd_new = jnp.where(active_cloud & lddraf_level, pmfd_acc[:, py], jnp.float32(0.0))
            pmfds_new = jnp.where(active_cloud & lddraf_level, pmfds_level, jnp.float32(0.0))
            pmfdq_new = jnp.where(active_cloud & lddraf_level, pmfdq_level, jnp.float32(0.0))
            pmfd_acc = pmfd_acc.at[:, py].set(jnp.where(level_active, pmfd_new, pmfd_acc[:, py]))
            pmfds_acc = pmfds_acc.at[:, py].set(jnp.where(level_active, pmfds_new, pmfds_acc[:, py]))
            pmfdq_acc = pmfdq_acc.at[:, py].set(jnp.where(level_active, pmfdq_new, pmfdq_acc[:, py]))
            reset_prev = has_prev & level_active & ~(active_cloud & lddraf_level)
            pdmfdp_acc = pdmfdp_acc.at[:, prev_py].set(jnp.where(reset_prev, jnp.float32(0.0), pdmfdp_acc[:, prev_py]))
            inactive = level_active & ~active_cloud
            pmfu_acc = pmfu_acc.at[:, py].set(jnp.where(inactive, jnp.float32(0.0), pmfu_acc[:, py]))
            pmful_acc = pmful_acc.at[:, py].set(jnp.where(inactive, jnp.float32(0.0), pmful_acc[:, py]))
            pdmfup_acc = pdmfup_acc.at[:, prev_py].set(jnp.where(has_prev & inactive, jnp.float32(0.0), pdmfup_acc[:, prev_py]))
            pdmfdp_acc = pdmfdp_acc.at[:, prev_py].set(jnp.where(has_prev & inactive, jnp.float32(0.0), pdmfdp_acc[:, prev_py]))
            plude_acc = plude_acc.at[:, prev_py].set(jnp.where(has_prev & inactive, jnp.float32(0.0), plude_acc[:, prev_py]))
            return (pmfu_acc, pmfd_acc, pmfus_acc, pmfds_acc, pmfuq_acc, pmfdq_acc, pmful_acc, plude_acc, pdmfup_acc, pdmfdp_acc)
        pmfu_out, pmfd_out, pmfus_out, pmfds_out, pmfuq_out, pmfdq_out, pmful_out, plude_out, pdmfup_out, pdmfdp_out = lax.fori_loop(0, klev_i - loop_start + 1, cuflx_prepare_body, (pmfu_out, pmfd_out, pmfus_out, pmfds_out, pmfuq_out, pmfdq_out, pmful_out, plude_out, pdmfup_out, pdmfdp_out))

    kcbot_idx = kcbot_arr - 1
    if loop_start >= 1:
        jk = jnp.arange(loop_start, klev_i + 1, dtype=jnp.int32)
        py = jk - 1
        level_active = jk[None, :] >= jnp.broadcast_to(ktopm2, ldcum_out.shape)[:, None]
        extend_mask = level_active & ldcum_out[:, None] & (jk[None, :] > kcbot_arr[:, None])
        paph_ikb = jnp.take_along_axis(paph_arr, kcbot_idx[:, None], axis=1)
        zzp = (paph_arr[:, klevp1_i - 1, None] - paph_arr[:, py]) / (paph_arr[:, klevp1_i - 1, None] - paph_ikb)
        zzp = jnp.where(ktype_out[:, None] == 3, zzp ** 2, zzp)
        base_pmfu = jnp.take_along_axis(pmfu_out, kcbot_idx[:, None], axis=1)
        base_pmfus = jnp.take_along_axis(pmfus_out, kcbot_idx[:, None], axis=1)
        base_pmfuq = jnp.take_along_axis(pmfuq_out, kcbot_idx[:, None], axis=1)
        base_pmful = jnp.take_along_axis(pmful_out, kcbot_idx[:, None], axis=1)
        pmfu_out = pmfu_out.at[:, py].set(jnp.where(extend_mask, base_pmfu * zzp, pmfu_out[:, py]))
        pmfus_out = pmfus_out.at[:, py].set(jnp.where(extend_mask, base_pmfus * zzp, pmfus_out[:, py]))
        pmfuq_out = pmfuq_out.at[:, py].set(jnp.where(extend_mask, base_pmfuq * zzp, pmfuq_out[:, py]))
        pmful_out = pmful_out.at[:, py].set(jnp.where(extend_mask, base_pmful * zzp, pmful_out[:, py]))
        active = level_active & ldcum_out[:, None]
        warm = pten_arr[:, py] > TMELT
        zfac = store(zcons1 * store(paph_arr[:, py + 1] - paph_arr[:, py]))
        melt_potential = store(zfac * store(pten_arr[:, py] - ztmelp2))
        inputs = (active.astype(jnp.float32), warm.astype(jnp.float32), (pten_arr[:, py] > ztmelp2).astype(jnp.float32), pdmfup_out[:, py], pdmfdp_out[:, py], melt_potential, pdpmel_out[:, py])
        token = jnp.uint32(0) if rounding_token is None else rounding_token
        current, history = scan_columns(_column_precipitation_step, (prain_out, prfl_out, psfl_out, jnp.zeros_like(prain_out)), tuple((jnp.swapaxes(value, 0, 1) for value in inputs)), token)
        prain_out, prfl_out, psfl_out, _ = current
        pdpmel_out = pdpmel_out.at[:, py].set(jnp.swapaxes(history[3], 0, 1))
    else:

        def cuflx_flux_body(loop_index: Any, carry: tuple[Any, ...]) -> tuple[Any, ...]:
            pmfu_acc, pmfus_acc, pmfuq_acc, pmful_acc, prain_acc, prfl_acc, psfl_acc, pdpmel_acc = carry
            jk = jnp.asarray(loop_start, dtype=jnp.int32) + jnp.asarray(loop_index, dtype=jnp.int32)
            py = jk - jnp.asarray(1, dtype=jnp.int32)
            level_active = jk >= ktopm2
            extend_mask = level_active & ldcum_out & (jk > kcbot_arr)
            paph_ikb = jnp.take_along_axis(paph_arr, kcbot_idx[:, None], axis=1)[:, 0]
            zzp = (paph_arr[:, klevp1_i - 1] - paph_arr[:, py]) / (paph_arr[:, klevp1_i - 1] - paph_ikb)
            zzp = jnp.where(ktype_out == 3, zzp ** 2, zzp)
            base_pmfu = jnp.take_along_axis(pmfu_acc, kcbot_idx[:, None], axis=1)[:, 0]
            base_pmfus = jnp.take_along_axis(pmfus_acc, kcbot_idx[:, None], axis=1)[:, 0]
            base_pmfuq = jnp.take_along_axis(pmfuq_acc, kcbot_idx[:, None], axis=1)[:, 0]
            base_pmful = jnp.take_along_axis(pmful_acc, kcbot_idx[:, None], axis=1)[:, 0]
            pmfu_acc = pmfu_acc.at[:, py].set(jnp.where(extend_mask, base_pmfu * zzp, pmfu_acc[:, py]))
            pmfus_acc = pmfus_acc.at[:, py].set(jnp.where(extend_mask, base_pmfus * zzp, pmfus_acc[:, py]))
            pmfuq_acc = pmfuq_acc.at[:, py].set(jnp.where(extend_mask, base_pmfuq * zzp, pmfuq_acc[:, py]))
            pmful_acc = pmful_acc.at[:, py].set(jnp.where(extend_mask, base_pmful * zzp, pmful_acc[:, py]))
            active = level_active & ldcum_out
            prain_acc = store(prain_acc + jnp.where(active, pdmfup_out[:, py], jnp.float32(0.0)))
            warm = pten_arr[:, py] > TMELT
            precip_inc = store(pdmfup_out[:, py] + pdmfdp_out[:, py])
            prfl_acc = store(prfl_acc + jnp.where(active & warm, precip_inc, jnp.float32(0.0)))
            psfl_acc = store(psfl_acc + jnp.where(active & ~warm, precip_inc, jnp.float32(0.0)))
            melt_mask = active & warm & (psfl_acc > 0.0) & (pten_arr[:, py] > ztmelp2)
            zfac = store(zcons1 * store(paph_arr[:, py + 1] - paph_arr[:, py]))
            zsnmlt = store(zfac * store(pten_arr[:, py] - ztmelp2))
            zsnmlt = store(jnp.minimum(psfl_acc, zsnmlt))
            pdpmel_acc = pdpmel_acc.at[:, py].set(jnp.where(melt_mask, zsnmlt, pdpmel_acc[:, py]))
            psfl_acc = store(psfl_acc - jnp.where(melt_mask, zsnmlt, jnp.float32(0.0)))
            prfl_acc = store(prfl_acc + jnp.where(melt_mask, zsnmlt, jnp.float32(0.0)))
            return (pmfu_acc, pmfus_acc, pmfuq_acc, pmful_acc, prain_acc, prfl_acc, psfl_acc, pdpmel_acc)
        pmfu_out, pmfus_out, pmfuq_out, pmful_out, prain_out, prfl_out, psfl_out, pdpmel_out = lax.fori_loop(0, klev_i - loop_start + 1, cuflx_flux_body, (pmfu_out, pmfus_out, pmfuq_out, pmful_out, prain_out, prfl_out, psfl_out, pdpmel_out))

    prfl_out = jnp.maximum(prfl_out, jnp.float32(0.0))
    psfl_out = jnp.maximum(psfl_out, jnp.float32(0.0))
    zpsubcl = store(prfl_out + psfl_out)

    evaporation_parameters = (rounding_token, zcucov, zcons2, CEVAPCU1, CEVAPCU2)
    if loop_start >= 1:
        jk = jnp.arange(loop_start, klev_i + 1, dtype=jnp.int32)
        py = jk - 1
        level_active = jk[None, :] >= jnp.broadcast_to(ktopm2, ldcum_out.shape)[:, None]
        eligible = level_active & ldcum_out[:, None] & (jk[None, :] >= kcbot_arr[:, None])
        profiles = (
            eligible, paph_arr[:, py + 1], paph_arr[:, py],
            pqsen_arr[:, py], pqen_arr[:, py], pdmfup_out[:, py],
        )
        inputs = (sig1_arr[py], *(jnp.swapaxes(value, 0, 1) for value in profiles))
        zpsubcl, history = scan_columns(
            _column_evaporation_step, zpsubcl, inputs, evaporation_parameters,
        )
        pdmfup_out = pdmfup_out.at[:, py].set(jnp.swapaxes(history, 0, 1))
    else:
        # Nonpositive concrete starts revisit wrapped profile indices. Keep
        # their update order, using the same layer arithmetic as the scan.
        def cuflx_evap_body(loop_index, carry):
            profile, budget = carry
            jk = jnp.int32(loop_start) + loop_index
            py = jk - jnp.int32(1)
            eligible = (jk >= ktopm2) & ldcum_out & (jk >= kcbot_arr)
            inputs = (
                sig1_arr[py], eligible, paph_arr[:, py + 1], paph_arr[:, py],
                pqsen_arr[:, py], pqen_arr[:, py], profile[:, py],
            )
            budget, value = _column_evaporation_step(budget, inputs, evaporation_parameters)
            return profile.at[:, py].set(value), budget

        pdmfup_out, zpsubcl = lax.fori_loop(
            0, klev_i - loop_start + 1, cuflx_evap_body, (pdmfup_out, zpsubcl),
        )

    total_surface_flux = store(prfl_out + psfl_out)
    zdpevap = store(zpsubcl - total_surface_flux)
    denom = store(jnp.maximum(jnp.float32(1.0e-20), total_surface_flux))
    rain_evap = store(zdpevap * prfl_out)
    rain_evap = store(real_divide(rain_evap, denom))
    snow_evap = store(zdpevap * psfl_out)
    snow_evap = store(real_divide(snow_evap, denom))
    prfl_out = store(prfl_out + rain_evap)
    psfl_out = store(psfl_out + snow_evap)

    return CuflxResult(
        ldcum=ldcum_out,
        lddraf=lddraf_out,
        ktype=ktype_out,
        pmfu=pmfu_out,
        pmfd=pmfd_out,
        pmfus=pmfus_out,
        pmfds=pmfds_out,
        pmfuq=pmfuq_out,
        pmfdq=pmfdq_out,
        pmful=pmful_out,
        plude=plude_out,
        pdmfup=pdmfup_out,
        pdmfdp=pdmfdp_out,
        prfl=prfl_out,
        prain=prain_out,
        psfl=psfl_out,
        pdpmel=pdpmel_out,
        ktopm2=ktopm2,
    )


def _restore_ascent_history(original: Any, current: Any, following: Any) -> Any:
    """Assemble descending layer writes, with the second write taking precedence."""
    if not current.shape[0]:
        return original
    # The two histories overlap; only the last current-layer value survives.
    return jnp.concatenate(
        (original[:, :1], current[-1, :, None],
         jnp.swapaxes(following[::-1], 0, 1), original[:, following.shape[0] + 2:]),
        axis=1,
    )


def cuasc_new(
    klon: int,
    klev: int,
    klevp1: int,
    klevm1: int,
    ptenh: Any,
    pqenh: Any,
    puen: Any,
    pven: Any,
    pten: Any,
    pqen: Any,
    pqsen: Any,
    pgeo: Any,
    pgeoh: Any,
    pap: Any,
    paph: Any,
    pqte: Any,
    pverv: Any,
    klwmin: Any,
    ldcum: Any,
    phcbase: Any,
    ktype: Any,
    klab: Any,
    ptu: Any,
    pqu: Any,
    plu: Any,
    puu: Any,
    pvu: Any,
    pmfu: Any,
    pmfub: Any,
    pentr: Any,
    pmfus: Any,
    pmfuq: Any,
    pmful: Any,
    plude: Any,
    pdmfup: Any,
    kcbot: Any,
    kctop: Any,
    kctop0: Any,
    ztmst: Any,
    khmin: Any,
    phhatt: Any,
    pqsenh: Any,
    row_size: int | None = None,
    use_scan: bool | None = None,
    scan_zero_small_buoyancy: bool | None = None,
    scan_statement_rounding: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> CuascResult:
    """Translate WRF ``cuasc_new`` cloud-ascent calculation."""
    del row_size, use_row_conditional_cuadjtq
    klon_i = int(klon)
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    klevm1_i = int(klevm1)
    use_scan_i = True if use_scan is None else bool(use_scan)
    scan_zero_small_buoyancy_i = (
        use_scan_i if scan_zero_small_buoyancy is None else bool(scan_zero_small_buoyancy)
    )
    scan_statement_rounding_i = (
        use_scan_i if scan_statement_rounding is None else bool(scan_statement_rounding)
    )
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)
    cols = jnp.arange(klon_i)
    ptenh_arr = jnp.asarray(ptenh, dtype=jnp.float32)
    pqenh_arr = jnp.asarray(pqenh, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    pgeo_arr = jnp.asarray(pgeo, dtype=jnp.float32)
    pgeoh_arr = jnp.asarray(pgeoh, dtype=jnp.float32)
    pap_arr = jnp.asarray(pap, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    pqte_arr = jnp.asarray(pqte, dtype=jnp.float32)
    pverv_arr = jnp.asarray(pverv, dtype=jnp.float32)
    klwmin_arr = jnp.asarray(klwmin, dtype=jnp.int32)
    ldcum_out = jnp.asarray(ldcum, dtype=bool)
    phcbase_arr = jnp.asarray(phcbase, dtype=jnp.float32)
    ktype_out = jnp.asarray(ktype, dtype=jnp.int32)
    klab_out = jnp.asarray(klab, dtype=jnp.int32)
    ptu_out = jnp.asarray(ptu, dtype=jnp.float32)
    pqu_out = jnp.asarray(pqu, dtype=jnp.float32)
    plu_out = jnp.asarray(plu, dtype=jnp.float32)
    puu_out = jnp.asarray(puu, dtype=jnp.float32)
    pvu_out = jnp.asarray(pvu, dtype=jnp.float32)
    pmfu_out = jnp.asarray(pmfu, dtype=jnp.float32)
    pmfub_out = jnp.asarray(pmfub, dtype=jnp.float32)
    pentr_out = jnp.asarray(pentr, dtype=jnp.float32)
    del pmfus, pmfuq, pmful
    plude_out = jnp.asarray(plude, dtype=jnp.float32)
    pdmfup_out = jnp.asarray(pdmfup, dtype=jnp.float32)
    kcbot_out = jnp.asarray(kcbot, dtype=jnp.int32)
    kctop_out = jnp.asarray(kctop, dtype=jnp.int32)
    kctop0_out = jnp.asarray(kctop0, dtype=jnp.int32)
    khmin_arr = jnp.asarray(khmin, dtype=jnp.int32)
    phhatt_arr = jnp.asarray(phhatt, dtype=jnp.float32)
    pqsenh_arr = jnp.asarray(pqsenh, dtype=jnp.float32)
    gravity_dt = store(G * jnp.asarray(ztmst, dtype=jnp.float32))
    zcons2 = store(jnp.float32(1.0) / gravity_dt)
    zmfuu = jnp.zeros((klon_i,), dtype=jnp.float32)
    zmfuv = jnp.zeros((klon_i,), dtype=jnp.float32)
    zbuoy = jnp.zeros((klon_i,), dtype=jnp.float32)
    ktype_out = jnp.where(ldcum_out, ktype_out, 0)
    zeros = jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
    pmfu_out = zeros
    plude_out = zeros
    pdmfup_out = zeros
    plu_out = zeros
    zoentr_seed = zeros
    clear_klab = ~ldcum_out[:, None] | (ktype_out[:, None] == 3)
    klab_out = jnp.where(clear_klab, 0, klab_out)
    levels = jnp.arange(1, klev_i + 1, dtype=jnp.int32)
    top_candidates = jnp.where(
        ~ldcum_out[:, None] & (paph_arr[:, :klev_i] < jnp.float32(40000.0)),
        levels[None, :],
        jnp.asarray(-1, dtype=jnp.int32),
    )
    kctop0_candidate = jnp.max(top_candidates, axis=1)
    kctop0_out = jnp.where(kctop0_candidate >= 0, kctop0_candidate, kctop0_out)
    kctop_out = jnp.full_like(kctop_out, klevm1_i)
    inactive = ~ldcum_out
    kcbot_out = jnp.where(inactive, klevm1_i, kcbot_out)
    pmfub_out = jnp.where(inactive, jnp.float32(0.0), pmfub_out)
    pqu_out = pqu_out.at[:, klev_i - 1].set(jnp.where(inactive, jnp.float32(0.0), pqu_out[:, klev_i - 1]))
    pmfu_out = pmfu_out.at[:, klev_i - 1].set(pmfub_out)
    if LMFDUDV:
        zmfuu = store(pmfub_out * puu_out[:, klev_i - 1])
        zmfuv = store(pmfub_out * pvu_out[:, klev_i - 1])
    ldcum_out = jnp.zeros_like(ldcum_out)
    deep_initial = ktype_out == 1
    ikb_idx = kcbot_out - 1
    ptu_ikb = jnp.take_along_axis(ptu_out, ikb_idx[:, None], axis=1)[:, 0]
    ptenh_ikb = jnp.take_along_axis(ptenh_arr, ikb_idx[:, None], axis=1)[:, 0]
    pqu_ikb = jnp.take_along_axis(pqu_out, ikb_idx[:, None], axis=1)[:, 0]
    pqenh_ikb = jnp.take_along_axis(pqenh_arr, ikb_idx[:, None], axis=1)[:, 0]
    zbuoy_thermal = store(ptu_ikb - ptenh_ikb)
    zbuoy_thermal = store(zbuoy_thermal / ptenh_ikb)
    zbuoy_moisture = store(pqu_ikb - pqenh_ikb)
    zbuoy_moisture = store(jnp.float32(0.608) * zbuoy_moisture)
    zbuoy_base = store(G * store(zbuoy_thermal + zbuoy_moisture))
    zbuoy = jnp.where(deep_initial, zbuoy_base, zbuoy)
    positive = deep_initial & (zbuoy > 0.0)
    prev_idx = jnp.maximum(ikb_idx - 1, 0)
    pgeo_prev = jnp.take_along_axis(pgeo_arr, prev_idx[:, None], axis=1)[:, 0]
    pgeo_ikb = jnp.take_along_axis(pgeo_arr, ikb_idx[:, None], axis=1)[:, 0]
    pten_prev = jnp.take_along_axis(pten_arr, prev_idx[:, None], axis=1)[:, 0]
    pten_ikb = jnp.take_along_axis(pten_arr, ikb_idx[:, None], axis=1)[:, 0]
    zdz = store(store(pgeo_prev - pgeo_ikb) * ZRG)
    temperature_ratio = store(pten_prev / pten_ikb)
    zdrodz = store(-jnp.log(temperature_ratio))
    zdrodz = store(zdrodz / zdz)
    density_term = store(RD * ptenh_ikb)
    density_term = store(G / density_term)
    zdrodz = store(zdrodz - density_term)
    zoentr_denominator = store(jnp.float32(1.0) + store(zbuoy * zdz))
    zoentr_base = store(store(zbuoy * jnp.float32(0.5)) / zoentr_denominator)
    zoentr_base = store(zoentr_base + zdrodz)
    zoentr_base = store(jnp.minimum(jnp.float32(0.001), zoentr_base))
    zoentr_base = store(jnp.maximum(jnp.float32(0.0), zoentr_base))
    zoentr_seed = zoentr_seed.at[cols, prev_idx].set(
        jnp.where(positive, zoentr_base, zoentr_seed[cols, prev_idx])
    )
    loop_state = _CuascLoopState(
        ldcum=ldcum_out,
        ktype=ktype_out,
        klab_next=klab_out[:, klevm1_i],
        ptu=ptu_out[:, klevm1_i],
        pqu=pqu_out[:, klevm1_i],
        plu_next=plu_out[:, klevm1_i],
        puu=puu_out[:, klevm1_i],
        pvu=pvu_out[:, klevm1_i],
        pmfu_next=pmfu_out[:, klevm1_i],
        pmfub=pmfub_out,
        pentr=pentr_out,
        kcbot=kcbot_out,
        kctop=kctop_out,
        kctop0=kctop0_out,
        khmin=khmin_arr,
        zmfuu=zmfuu,
        zmfuv=zmfuv,
        zbuoy=zbuoy,
        zoentr_value=jnp.zeros((klon_i,), dtype=jnp.float32),
        zoentr_valid=jnp.zeros((klon_i,), dtype=bool),
        base_pressure=jnp.take_along_axis(paph_arr, (kcbot_out - 1)[:, None], axis=1)[:, 0],
    )
    scan_steps = klevm1_i - 1
    scan_indices = jnp.arange(scan_steps, dtype=jnp.int32)
    ascent_indices = jnp.asarray(klevm1_i, jnp.int32) - scan_indices - 1
    fixed_top_pressure = jnp.take_along_axis(paph_arr, (kctop0_out - 1)[:, None], axis=1)[:, 0]
    fixed_min_geo = jnp.take_along_axis(pgeoh_arr, (khmin_arr - 1)[:, None], axis=1)[:, 0]
    fixed_top_geo = jnp.take_along_axis(pgeoh_arr, jnp.maximum(kctop0_out - 1, 0)[:, None], axis=1)[:, 0]

    def column_step(st, xs, parameters):
        """Advance a whole column with adjacent-layer state and explicit forcing."""
        zcons2, rounding_token, exp_table, constants = parameters
        store = lambda value: (
            column_barrier(jnp.asarray(value, jnp.float32))
            if rounding_token is None
            else _store_wrf_real(value, rounding_token)
        )
        (
            jk,
            pqen_current,
            pqsen_current,
            pten_current,
            pgeo_current,
            pgeoh_lower,
            pverv_current,
            puen_current,
            pven_current,
            paph_current,
            klab_initial_current,
            paph_upper,
            ptenh_lower,
            paph_lower,
            pap_current,
            pgeoh_current,
            pmfu_initial_current,
            zoentr_seed_current,
            phhatt_lower,
            pqenh_lower,
            pqsenh_lower,
            plu_initial_current,
            pqu_initial_current,
            ptu_initial_current,
            puu_initial_current,
            pvu_initial_current,
            pqenh_current,
            ptenh_current,
            pgeo_upper,
            pten_upper,
            klwmin_arr,
            phcbase_arr,
            fixed_top_pressure,
            fixed_min_geo,
            fixed_top_geo,
        ) = xs

        def _cuasc_level_transition(jk: Any, st: _CuascLoopState) -> tuple[_CuascLoopState, tuple[Any, ...]]:
            jk = jnp.asarray(jk, dtype=jnp.int32)

            def _midlevel(inner: _CuascLoopState) -> _CuascLoopState:
                mask = (
                    ~inner.ldcum
                    & (inner.klab_next == 0)
                    & (pqen_current > column_binary("mul", jnp.float32(0.8), pqsen_current))
                )
                ptu_col = store(column_binary("mul", constants.cpd, pten_current))
                ptu_col = store(column_binary("add", ptu_col, pgeo_current))
                ptu_col = store(column_binary("sub", ptu_col, pgeoh_lower))
                ptu_col = store(column_binary("mul", ptu_col, constants.rcpd))
                pqu_col = pqen_current
                zzzmb = store(column_binary("div", -pverv_current, constants.g))
                zzzmb = store(jnp.maximum(constants.cmfcmin, zzzmb))
                zzzmb = store(jnp.minimum(zzzmb, constants.cmfcmax))
                ptu_mid = jnp.where(mask, ptu_col, inner.ptu)
                pqu_mid = jnp.where(mask, pqu_col, inner.pqu)
                plu_next_mid = jnp.where(mask, jnp.float32(0.0), inner.plu_next)
                pmfub_mid = jnp.where(mask, zzzmb, inner.pmfub)
                pmfu_next_mid = jnp.where(mask, pmfub_mid, inner.pmfu_next)
                kcbot_mid = jnp.where(mask, jk, inner.kcbot)
                klab_next_mid = jnp.where(mask, 1, inner.klab_next)
                ktype_mid = jnp.where(mask, 3, inner.ktype)
                pentr_mid = jnp.where(mask, constants.entrmid, inner.pentr)
                if LMFDUDV:
                    puu_mid = jnp.where(mask, puen_current, inner.puu)
                    pvu_mid = jnp.where(mask, pven_current, inner.pvu)
                    zmfuu_mid = jnp.where(mask, store(column_binary("mul", pmfub_mid, puu_mid)), inner.zmfuu)
                    zmfuv_mid = jnp.where(mask, store(column_binary("mul", pmfub_mid, pvu_mid)), inner.zmfuv)
                else:
                    puu_mid = inner.puu
                    pvu_mid = inner.pvu
                    zmfuu_mid = inner.zmfuu
                    zmfuv_mid = inner.zmfuv
                return _CuascLoopState(
                    ldcum=inner.ldcum,
                    ktype=ktype_mid,
                    klab_next=klab_next_mid,
                    ptu=ptu_mid,
                    pqu=pqu_mid,
                    plu_next=plu_next_mid,
                    puu=puu_mid,
                    pvu=pvu_mid,
                    pmfu_next=pmfu_next_mid,
                    pmfub=pmfub_mid,
                    pentr=pentr_mid,
                    kcbot=kcbot_mid,
                    kctop=inner.kctop,
                    kctop0=inner.kctop0,
                    khmin=inner.khmin,
                    zmfuu=zmfuu_mid,
                    zmfuv=zmfuv_mid,
                    zbuoy=inner.zbuoy,
                    zoentr_value=inner.zoentr_value,
                    zoentr_valid=inner.zoentr_valid,
                    base_pressure=jnp.where(mask, paph_current, inner.base_pressure),
                )

            if LMFMID:
                do_mid = (jk < jnp.asarray(klevm1_i, dtype=jnp.int32)) & (
                    jk > jnp.asarray(column_binary("sub", klev_i, 13), dtype=jnp.int32)
                )
                do_mid_concrete = _concrete_bool_or_none(do_mid)
                if do_mid_concrete is None:
                    st = lax.cond(do_mid, _midlevel, lambda inner: inner, st)
                elif do_mid_concrete:
                    st = _midlevel(st)
            return _cuasc_level_transition_body(jk, st)

        def _cuasc_level_transition_body(
            jk: Any, st: _CuascLoopState
        ) -> tuple[_CuascLoopState, tuple[Any, ...]]:
            zqold = jnp.zeros((st.ldcum.shape[0],), dtype=jnp.float32)
            next_label = st.klab_next
            klab_py_l = jnp.where(next_label == 0, 0, klab_initial_current)
            loflag = next_label > 0
            any_loflag = jnp.max(loflag.astype(jnp.int32)) != 0
            level_active = loflag
            layer_pressure = store(column_binary("sub", paph_current, paph_upper))
            zmfmax = store(column_binary("mul", layer_pressure, zcons2))
            cap_mask = (st.ktype == 3) & (jk == st.kcbot) & (st.pmfub > zmfmax)
            cap_mass = jnp.where(cap_mask, st.pmfub, jnp.float32(1.0))
            zfac = store(column_binary("div", zmfmax, cap_mass))
            pmfu_next_l = jnp.where(cap_mask, store(column_binary("mul", st.pmfu_next, zfac)), st.pmfu_next)
            zmfuu_l = jnp.where(cap_mask, store(column_binary("mul", st.zmfuu, zfac)), st.zmfuu)
            zmfuv_l = jnp.where(cap_mask, store(column_binary("mul", st.zmfuv, zfac)), st.zmfuv)
            pmfub_out_l = jnp.where(cap_mask, zmfmax, st.pmfub)

            def _run_cuentr(_: Any) -> CuentrLevelResult:
                return _cuentr_values(
                    jk,
                    ptenh_lower,
                    paph_lower,
                    paph_current,
                    pap_current,
                    klwmin_arr,
                    jnp.where(level_active, st.ldcum, False),
                    st.ktype,
                    st.kcbot,
                    st.kctop0,
                    pmfu_next_l,
                    st.pentr,
                    st.khmin,
                    pgeoh_current,
                    st.base_pressure,
                    fixed_top_pressure,
                    fixed_min_geo,
                    fixed_top_geo,
                    constants,
                )

            def _skip_cuentr(_: Any) -> CuentrLevelResult:
                return CuentrLevelResult(
                    zpbase=jnp.zeros((st.ldcum.shape[0],), dtype=jnp.float32),
                    zdmfen=jnp.zeros((st.ldcum.shape[0],), dtype=jnp.float32),
                    zdmfde=jnp.zeros((st.ldcum.shape[0],), dtype=jnp.float32),
                    zodetr_col=jnp.zeros((st.ldcum.shape[0],), dtype=jnp.float32),
                )

            entr = lax.cond(any_loflag, _run_cuentr, _skip_cuentr, operand=None)
            zpbase = entr.zpbase
            zdmfen = entr.zdmfen
            zdmfde = entr.zdmfde
            zodetr_col_l = entr.zodetr_col
            below_base = jk < st.kcbot
            zmftest = store(column_binary("add", pmfu_next_l, zdmfen))
            zmftest = store(column_binary("sub", zmftest, zdmfde))
            layer_mass_limit = store(
                column_binary("mul", store(column_binary("sub", paph_current, paph_upper)), zcons2)
            )
            zmfmax = store(jnp.minimum(zmftest, layer_mass_limit))
            excess_mass = store(column_binary("sub", zmftest, zmfmax))
            excess_mass = store(jnp.maximum(excess_mass, jnp.float32(0.0)))
            zdmfen_limited = store(column_binary("sub", zdmfen, excess_mass))
            zdmfen_limited = store(jnp.maximum(zdmfen_limited, jnp.float32(0.0)))
            zdmfen = jnp.where(loflag & below_base, zdmfen_limited, zdmfen)
            detrain_limit = store(column_binary("mul", jnp.float32(0.75), pmfu_next_l))
            zdmfde = jnp.where(loflag, store(jnp.minimum(zdmfde, detrain_limit)), zdmfde)
            pmfu_col = store(column_binary("add", pmfu_next_l, zdmfen))
            pmfu_col = store(column_binary("sub", pmfu_col, zdmfde))
            pmfu_py_l = jnp.where(loflag, pmfu_col, pmfu_initial_current)
            zdprho = store(
                column_binary("mul", store(column_binary("sub", pgeoh_current, pgeoh_lower)), constants.zrg)
            )
            zoentr_current = jnp.where(st.zoentr_valid, st.zoentr_value, zoentr_seed_current)
            zoentr_mass_col = store(column_binary("mul", zoentr_current, zdprho))
            zoentr_mass_col = store(column_binary("mul", zoentr_mass_col, pmfu_next_l))
            zmftest = store(column_binary("add", pmfu_py_l, zoentr_mass_col))
            zmftest = store(column_binary("sub", zmftest, zodetr_col_l))
            zmfmax = store(jnp.minimum(zmftest, layer_mass_limit))
            organized_excess = store(column_binary("sub", zmftest, zmfmax))
            organized_excess = store(jnp.maximum(organized_excess, jnp.float32(0.0)))
            zoentr_mass_col = store(column_binary("sub", zoentr_mass_col, organized_excess))
            zoentr_mass_col = store(jnp.maximum(zoentr_mass_col, jnp.float32(0.0)))
            zoentr_col_l = jnp.where(loflag & below_base, zoentr_mass_col, zoentr_current)
            deep_det = loflag & (st.ktype == 1) & below_base & (jk <= st.khmin)
            zmse = column_binary(
                "add",
                column_binary(
                    "add",
                    column_binary("mul", constants.cpd, st.ptu),
                    column_binary("mul", constants.alv, st.pqu),
                ),
                pgeoh_lower,
            )
            pgeoh_ikt = fixed_top_geo
            phhatt_next = phhatt_lower
            znevn = column_binary(
                "mul",
                column_binary(
                    "mul",
                    column_binary("sub", pgeoh_ikt, pgeoh_lower),
                    column_binary("sub", zmse, phhatt_next),
                ),
                constants.zrg,
            )
            znevn = jnp.where(znevn <= 0.0, jnp.float32(1.0), znevn)
            zodmax = column_binary("div", column_binary("sub", phcbase_arr, zmse), znevn)
            zodmax = column_binary("mul", zodmax, zdprho)
            zodmax = column_binary("mul", zodmax, pmfu_next_l)
            zodmax = jnp.maximum(zodmax, jnp.float32(0.0))
            zodetr_col = jnp.where(deep_det, jnp.minimum(zodetr_col_l, zodmax), zodetr_col_l)
            organized_detrain_limit = store(column_binary("mul", jnp.float32(0.75), pmfu_py_l))
            zodetr_col = jnp.where(
                loflag, store(jnp.minimum(zodetr_col, organized_detrain_limit)), zodetr_col
            )
            zodetr_col_l = zodetr_col
            pmfu_organized = store(column_binary("add", pmfu_py_l, zoentr_col_l))
            pmfu_organized = store(column_binary("sub", pmfu_organized, zodetr_col_l))
            pmfu_py_l = jnp.where(loflag, pmfu_organized, pmfu_py_l)
            zqeen = store(column_binary("mul", pqenh_lower, zdmfen))
            zqeen = store(column_binary("add", zqeen, store(column_binary("mul", pqenh_lower, zoentr_col_l))))
            environment_static = store(column_binary("mul", constants.cpd, ptenh_lower))
            environment_static = store(column_binary("add", environment_static, pgeoh_lower))
            zseen = store(column_binary("mul", environment_static, zdmfen))
            zseen = store(
                column_binary("add", zseen, store(column_binary("mul", environment_static, zoentr_col_l)))
            )
            cloud_static = store(column_binary("mul", constants.cpd, st.ptu))
            cloud_static = store(column_binary("add", cloud_static, pgeoh_lower))
            zscde = store(column_binary("mul", cloud_static, zdmfde))
            zga = column_binary(
                "div",
                column_binary("mul", constants.alv, pqsenh_lower),
                column_binary("mul", constants.rv, ptenh_lower**2),
            )
            plu_next_l = st.plu_next
            zdt = column_binary(
                "div",
                column_binary(
                    "sub",
                    plu_next_l,
                    column_binary("mul", jnp.float32(0.608), column_binary("sub", pqsenh_lower, pqenh_lower)),
                ),
                column_binary(
                    "add",
                    column_binary("div", jnp.float32(1.0), ptenh_lower),
                    column_binary("mul", jnp.float32(0.608), zga),
                ),
            )
            zscod = column_binary("add", column_binary("mul", constants.cpd, ptenh_lower), pgeoh_lower)
            zscod = column_binary("add", zscod, column_binary("mul", constants.cpd, zdt))
            zscde = store(column_binary("add", zscde, store(column_binary("mul", zodetr_col_l, zscod))))
            zqude = store(column_binary("mul", st.pqu, zdmfde))
            zqcod = store(column_binary("add", pqsenh_lower, store(column_binary("mul", zga, zdt))))
            zqude = store(column_binary("add", zqude, store(column_binary("mul", zodetr_col_l, zqcod))))
            plude_col = store(column_binary("mul", plu_next_l, zdmfde))
            plude_col = store(
                column_binary("add", plude_col, store(column_binary("mul", plu_next_l, zodetr_col_l)))
            )
            plude_level = jnp.where(loflag, plude_col, jnp.float32(0.0))
            pmfus_next = store(column_binary("mul", cloud_static, pmfu_next_l))
            pmfuq_next = store(column_binary("mul", st.pqu, pmfu_next_l))
            pmful_next = store(column_binary("mul", plu_next_l, pmfu_next_l))
            zmfusk = store(column_binary("add", pmfus_next, zseen))
            zmfusk = store(column_binary("sub", zmfusk, zscde))
            zmfuqk = store(column_binary("add", pmfuq_next, zqeen))
            zmfuqk = store(column_binary("sub", zmfuqk, zqude))
            zmfulk = store(column_binary("sub", pmful_next, plude_level))
            denom_up = jnp.maximum(constants.cmfcmin, pmfu_py_l)
            if scan_statement_rounding_i:
                inv_denom_up = store(column_binary("div", jnp.float32(1.0), denom_up))
                plu_col = store(column_binary("mul", zmfulk, inv_denom_up))
                pqu_col = store(column_binary("mul", zmfuqk, inv_denom_up))
                ptu_energy = store(column_binary("mul", zmfusk, inv_denom_up))
                ptu_energy, ptu_geo = column_barrier((ptu_energy, pgeoh_current))
                ptu_delta = store(
                    column_binary("sub", ptu_energy.astype(jnp.float64), ptu_geo.astype(jnp.float64)).astype(
                        jnp.float32
                    )
                )
                ptu_col = store(
                    column_binary(
                        "mul", ptu_delta.astype(jnp.float64), constants.rcpd.astype(jnp.float64)
                    ).astype(jnp.float32)
                )
            else:
                inv_denom_up = column_binary("div", jnp.float32(1.0), denom_up)
                plu_col = column_binary("mul", zmfulk, inv_denom_up)
                pqu_col = column_binary("mul", zmfuqk, inv_denom_up)
                ptu_col = column_binary(
                    "mul",
                    column_binary("sub", column_binary("mul", zmfusk, inv_denom_up), pgeoh_current),
                    constants.rcpd,
                )
            ptu_col = jnp.minimum(jnp.float32(400.0), jnp.maximum(jnp.float32(100.0), ptu_col))
            zqold = jnp.where(loflag, pqu_col, zqold)

            def _run_cuadjtq(_: Any) -> tuple[Any, Any]:
                return _cuadjtq_values(paph_current, ptu_col, pqu_col, loflag, 1, exp_table, constants)

            ptu_col, pqu_col = lax.cond(any_loflag, _run_cuadjtq, lambda _: (ptu_col, pqu_col), operand=None)
            if scan_statement_rounding_i:
                ptu_col = store(ptu_col)
                pqu_col = store(pqu_col)
            plu_py_l = jnp.where(loflag, plu_col, plu_initial_current)
            pqu_out_l = jnp.where(loflag, pqu_col, pqu_initial_current)
            ptu_out_l = jnp.where(loflag, ptu_col, ptu_initial_current)
            puu_out_l = puu_initial_current
            pvu_out_l = pvu_initial_current
            condensation = column_binary("sub", zqold, pqu_col)
            if scan_statement_rounding_i:
                condensation = store(condensation)
            changed = loflag & (pqu_col != zqold)
            klab_py_l = jnp.where(changed, 2, klab_py_l)
            if scan_statement_rounding_i:
                plu_after_condensation = store(column_binary("add", plu_py_l, condensation))
                plu_py_l = jnp.where(changed, plu_after_condensation, plu_py_l)
                zbuo_moist = store(
                    column_binary(
                        "sub",
                        column_binary(
                            "add", jnp.float32(1.0), store(column_binary("mul", constants.vtmpc1, pqu_col))
                        ),
                        plu_py_l,
                    )
                )
                zbuo_cloud = store(column_binary("mul", ptu_col, zbuo_moist))
                zbuo_env = store(
                    column_binary(
                        "add", jnp.float32(1.0), store(column_binary("mul", constants.vtmpc1, pqenh_current))
                    )
                )
                zbuo_ref = store(column_binary("mul", ptenh_current, zbuo_env))
                zbuo_cloud, zbuo_ref = column_barrier((zbuo_cloud, zbuo_ref))
                zbuo = store(column_binary("sub", zbuo_cloud, zbuo_ref))
                if scan_zero_small_buoyancy_i:
                    zbuo = jnp.where(zbuo_cloud == zbuo_ref, jnp.float32(0.0), zbuo)
                    zbuo_ulp = store(_real_spacing(zbuo_ref))
                    zbuo = jnp.where(jnp.abs(zbuo) <= zbuo_ulp, jnp.float32(0.0), zbuo)
                zbuo = jnp.where(next_label == 1, store(column_binary("add", zbuo, constants.zbuo0)), zbuo)
            else:
                plu_py_l = column_binary("add", plu_py_l, jnp.where(changed, condensation, jnp.float32(0.0)))
                zbuo = column_binary(
                    "sub",
                    column_binary(
                        "mul",
                        ptu_col,
                        column_binary(
                            "sub",
                            column_binary(
                                "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, pqu_col)
                            ),
                            plu_py_l,
                        ),
                    ),
                    column_binary(
                        "mul",
                        ptenh_current,
                        column_binary(
                            "add", jnp.float32(1.0), column_binary("mul", constants.vtmpc1, pqenh_current)
                        ),
                    ),
                )
                zbuo = jnp.where(next_label == 1, column_binary("add", zbuo, constants.zbuo0), zbuo)
            keep = (
                changed
                & (zbuo > 0.0)
                & (pmfu_py_l > column_binary("mul", jnp.float32(0.01), pmfub_out_l))
                & (jk >= st.kctop0)
            )
            kctop_out_l = jnp.where(keep, jk, st.kctop)
            ldcum_out_l = jnp.where(keep, True, st.ldcum)
            zprcon = jnp.where(
                column_binary("sub", zpbase, paph_current) >= constants.zdnoprc,
                constants.cprcon,
                jnp.float32(0.0),
            )
            zlnew = column_binary(
                "div",
                plu_py_l,
                column_binary(
                    "add",
                    jnp.float32(1.0),
                    column_binary("mul", zprcon, column_binary("sub", pgeoh_current, pgeoh_lower)),
                ),
            )
            pdmfup_new = jnp.maximum(
                jnp.float32(0.0), column_binary("mul", column_binary("sub", plu_py_l, zlnew), pmfu_py_l)
            )
            pdmfup_level = jnp.where(keep, pdmfup_new, jnp.float32(0.0))
            plu_py_l = jnp.where(keep, zlnew, plu_py_l)
            fail = changed & ~keep
            klab_py_l = jnp.where(fail, 0, klab_py_l)
            pmfu_py_l = jnp.where(fail, jnp.float32(0.0), pmfu_py_l)
            if LMFDUDV:
                zdmfen_total = column_binary("add", zdmfen, zoentr_col_l)
                zdmfde_total = column_binary("add", zdmfde, zodetr_col_l)
                zz_deep = jnp.where(zdmfen_total <= jnp.float32(1e-20), jnp.float32(3.0), jnp.float32(2.0))
                zz_other = jnp.where(zdmfen_total <= jnp.float32(1e-20), jnp.float32(1.0), jnp.float32(0.0))
                zz = jnp.where((st.ktype == 1) | (st.ktype == 3), zz_deep, zz_other)
                zdmfeu = column_binary("add", zdmfen_total, column_binary("mul", zz, zdmfde_total))
                zdmfdu = jnp.minimum(
                    column_binary("add", zdmfde_total, column_binary("mul", zz, zdmfde_total)),
                    column_binary("mul", jnp.float32(0.75), pmfu_next_l),
                )
                zmfuu_l = column_binary(
                    "add",
                    zmfuu_l,
                    jnp.where(
                        loflag,
                        column_binary(
                            "sub",
                            column_binary("mul", zdmfeu, puen_current),
                            column_binary("mul", zdmfdu, st.puu),
                        ),
                        jnp.float32(0.0),
                    ),
                )
                zmfuv_l = column_binary(
                    "add",
                    zmfuv_l,
                    jnp.where(
                        loflag,
                        column_binary(
                            "sub",
                            column_binary("mul", zdmfeu, pven_current),
                            column_binary("mul", zdmfdu, st.pvu),
                        ),
                        jnp.float32(0.0),
                    ),
                )
                wind_update = loflag & (pmfu_py_l > 0.0)
                wind_mass = jnp.where(wind_update, pmfu_py_l, jnp.float32(1.0))
                puu_out_l = jnp.where(wind_update, column_binary("div", zmfuu_l, wind_mass), puu_out_l)
                pvu_out_l = jnp.where(wind_update, column_binary("div", zmfuv_l, wind_mass), pvu_out_l)
            org_mask = loflag & (st.ktype == 1)
            zbuoyz = column_binary(
                "mul",
                constants.g,
                column_binary(
                    "sub",
                    column_binary(
                        "add",
                        column_binary("div", column_binary("sub", ptu_out_l, ptenh_current), ptenh_current),
                        column_binary(
                            "mul", jnp.float32(0.608), column_binary("sub", pqu_out_l, pqenh_current)
                        ),
                    ),
                    plu_py_l,
                ),
            )
            zbuoyz = jnp.maximum(zbuoyz, jnp.float32(0.0))
            zdz = column_binary("mul", column_binary("sub", pgeo_upper, pgeo_current), constants.zrg)
            zdrodz = column_binary(
                "sub",
                column_binary("div", -jnp.log(column_binary("div", pten_upper, pten_current)), zdz),
                column_binary("div", constants.g, column_binary("mul", constants.rd, ptenh_current)),
            )
            zbuoy_l = column_binary(
                "add", st.zbuoy, jnp.where(org_mask, column_binary("mul", zbuoyz, zdz), jnp.float32(0.0))
            )
            zo_next = jnp.minimum(
                jnp.float32(0.001),
                jnp.maximum(
                    jnp.float32(0.0),
                    column_binary(
                        "add",
                        column_binary(
                            "div",
                            column_binary("mul", zbuoyz, jnp.float32(0.5)),
                            column_binary("add", jnp.float32(1.0), zbuoy_l),
                        ),
                        zdrodz,
                    ),
                ),
            )
            zoentr_value_l = zo_next
            zoentr_valid_l = org_mask
            next_state = _CuascLoopState(
                ldcum=ldcum_out_l,
                ktype=st.ktype,
                klab_next=klab_py_l,
                ptu=ptu_out_l,
                pqu=pqu_out_l,
                plu_next=plu_py_l,
                puu=puu_out_l,
                pvu=pvu_out_l,
                pmfu_next=pmfu_py_l,
                pmfub=pmfub_out_l,
                pentr=st.pentr,
                kcbot=st.kcbot,
                kctop=kctop_out_l,
                kctop0=st.kctop0,
                khmin=st.khmin,
                zmfuu=zmfuu_l,
                zmfuv=zmfuv_l,
                zbuoy=zbuoy_l,
                zoentr_value=zoentr_value_l,
                zoentr_valid=zoentr_valid_l,
                base_pressure=st.base_pressure,
            )
            return (
                next_state,
                (
                    plude_level,
                    pdmfup_level,
                    klab_py_l,
                    st.klab_next,
                    pmfu_py_l,
                    pmfu_next_l,
                    plu_py_l,
                    plu_next_l,
                    ptu_out_l,
                    pqu_out_l,
                    puu_out_l,
                    pvu_out_l,
                    st.ptu,
                    st.pqu,
                    st.puu,
                    st.pvu,
                ),
            )

        return _cuasc_level_transition(jk, st)

    loop_state, scan_outputs = scan_columns(
        column_step,
        loop_state,
        (
            jnp.asarray(klevm1_i, jnp.int32) - scan_indices,
            jnp.swapaxes(pqen_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pqsen_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pten_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pgeo_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pgeoh_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(pverv_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(puen_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pven_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(paph_arr[:, ascent_indices], 0, 1),
            # Adjacent labels avoid folding a full-column constant gather.
            jnp.swapaxes(klab_out[:, 1:klevm1_i][:, ::-1], 0, 1),
            jnp.swapaxes(paph_arr[:, ascent_indices - 1], 0, 1),
            jnp.swapaxes(ptenh_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(paph_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(pap_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pgeoh_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pmfu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(zoentr_seed[:, ascent_indices], 0, 1),
            jnp.swapaxes(phhatt_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(pqenh_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(pqsenh_arr[:, ascent_indices + 1], 0, 1),
            jnp.swapaxes(plu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(pqu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(ptu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(puu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(pvu_out[:, ascent_indices], 0, 1),
            jnp.swapaxes(pqenh_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(ptenh_arr[:, ascent_indices], 0, 1),
            jnp.swapaxes(pgeo_arr[:, ascent_indices - 1], 0, 1),
            jnp.swapaxes(pten_arr[:, ascent_indices - 1], 0, 1),
            jnp.broadcast_to(klwmin_arr[None, :], (scan_steps, klon_i)),
            jnp.broadcast_to(phcbase_arr[None, :], (scan_steps, klon_i)),
            jnp.broadcast_to(fixed_top_pressure[None, :], (scan_steps, klon_i)),
            jnp.broadcast_to(fixed_min_geo[None, :], (scan_steps, klon_i)),
            jnp.broadcast_to(fixed_top_geo[None, :], (scan_steps, klon_i)),
        ),
        (zcons2, rounding_token, _EXP2F_TABLE, _tiedtke_constants()),
    )
    (
        plude_scan,
        pdmfup_scan,
        klab_py_scan,
        klab_next_scan,
        pmfu_py_scan,
        pmfu_next_scan,
        plu_py_scan,
        plu_next_scan,
        ptu_py_scan,
        pqu_py_scan,
        puu_py_scan,
        pvu_py_scan,
        ptu_next_scan,
        pqu_next_scan,
        puu_next_scan,
        pvu_next_scan,
    ) = scan_outputs
    plude_out = (
        jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
        .at[:, 1:klevm1_i]
        .set(jnp.swapaxes(plude_scan[::-1], 0, 1))
    )
    pdmfup_out = (
        jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
        .at[:, 1:klevm1_i]
        .set(jnp.swapaxes(pdmfup_scan[::-1], 0, 1))
    )
    klab_out = _restore_ascent_history(klab_out, klab_py_scan, klab_next_scan)
    pmfu_out = _restore_ascent_history(pmfu_out, pmfu_py_scan, pmfu_next_scan)
    plu_out = _restore_ascent_history(plu_out, plu_py_scan, plu_next_scan)
    ldcum_out = loop_state.ldcum
    ktype_out = loop_state.ktype
    ptu_out = _restore_ascent_history(ptu_out, ptu_py_scan, ptu_next_scan)
    pqu_out = _restore_ascent_history(pqu_out, pqu_py_scan, pqu_next_scan)
    puu_out = _restore_ascent_history(puu_out, puu_py_scan, puu_next_scan)
    pvu_out = _restore_ascent_history(pvu_out, pvu_py_scan, pvu_next_scan)
    pmfub_out = loop_state.pmfub
    pentr_out = loop_state.pentr
    kcbot_out = loop_state.kcbot
    kctop_out = loop_state.kctop
    kctop0_out = loop_state.kctop0
    khmin_arr = loop_state.khmin
    zmfuu = loop_state.zmfuu
    zmfuv = loop_state.zmfuv
    zbuoy = loop_state.zbuoy
    ldcum_out = jnp.where(kctop_out == klevm1_i, False, ldcum_out)
    kcbot_out = jnp.maximum(kcbot_out, kctop_out)
    kcum = jnp.sum(ldcum_out.astype(jnp.int32))
    cloud_static_all = store(CPD * ptu_out)
    cloud_static_all = store(cloud_static_all + pgeoh_arr)
    pmfus_out = store(cloud_static_all * pmfu_out)
    pmfuq_out = store(pqu_out * pmfu_out)
    pmful_out = store(plu_out * pmfu_out)
    if not _is_traced(kcum):
        kcum_int = int(jnp.asarray(kcum).item())
        if kcum_int == 0:
            return CuascResult(
                ldcum=ldcum_out,
                phcbase=phcbase_arr,
                ktype=ktype_out,
                klab=klab_out,
                ptu=ptu_out,
                pqu=pqu_out,
                plu=plu_out,
                puu=puu_out,
                pvu=pvu_out,
                pmfu=pmfu_out,
                pmfub=pmfub_out,
                pentr=pentr_out,
                pmfus=pmfus_out,
                pmfuq=pmfuq_out,
                pmful=pmful_out,
                plude=plude_out,
                pdmfup=pdmfup_out,
                kcbot=kcbot_out,
                kctop=kctop_out,
                kctop0=kctop0_out,
                kcum=kcum_int,
                khmin=khmin_arr,
            )
    active = ldcum_out
    jk_top = kctop_out - 1
    py_top = jk_top - 1
    next_top = jk_top
    zdmfde_top = store(store(jnp.float32(1.0) - CMFCTOP) * pmfu_out[cols, next_top])
    plude_top = store(zdmfde_top * plu_out[cols, next_top])
    pmfu_top = store(pmfu_out[cols, next_top] - zdmfde_top)
    pmful_top = store(plu_out[cols, py_top] * pmfu_top)
    pmfu_out = pmfu_out.at[cols, py_top].set(jnp.where(active, pmfu_top, pmfu_out[cols, py_top]))
    plude_out = plude_out.at[cols, py_top].set(jnp.where(active, plude_top, plude_out[cols, py_top]))
    prev_top = jnp.maximum(py_top - 1, 0)
    plude_out = plude_out.at[cols, prev_top].set(
        jnp.where(active & (py_top > 0), pmful_top, plude_out[cols, prev_top])
    )
    pdmfup_out = pdmfup_out.at[cols, py_top].set(
        jnp.where(active, jnp.float32(0.0), pdmfup_out[cols, py_top])
    )
    if LMFDUDV:
        puu_out = puu_out.at[cols, py_top].set(
            jnp.where(active, puu_out[cols, next_top], puu_out[cols, py_top])
        )
        pvu_out = pvu_out.at[cols, py_top].set(
            jnp.where(active, pvu_out[cols, next_top], pvu_out[cols, py_top])
        )
    cloud_static_all = store(CPD * ptu_out)
    cloud_static_all = store(cloud_static_all + pgeoh_arr)
    pmfus_out = store(cloud_static_all * pmfu_out)
    pmfuq_out = store(pqu_out * pmfu_out)
    pmful_out = store(plu_out * pmfu_out)
    return CuascResult(
        ldcum=ldcum_out,
        phcbase=phcbase_arr,
        ktype=ktype_out,
        klab=klab_out,
        ptu=ptu_out,
        pqu=pqu_out,
        plu=plu_out,
        puu=puu_out,
        pvu=pvu_out,
        pmfu=pmfu_out,
        pmfub=pmfub_out,
        pentr=pentr_out,
        pmfus=pmfus_out,
        pmfuq=pmfuq_out,
        pmful=pmful_out,
        plude=plude_out,
        pdmfup=pdmfup_out,
        kcbot=kcbot_out,
        kctop=kctop_out,
        kctop0=kctop0_out,
        kcum=kcum,
        khmin=khmin_arr,
    )


_cuasc_new_jit = jax.jit(
    cuasc_new,
    static_argnums=(0, 1, 2, 3),
    static_argnames=(
        "row_size",
        "use_scan",
        "scan_zero_small_buoyancy",
        "scan_statement_rounding",
        "use_row_conditional_cuadjtq",
    ),
)


def _run_cuasc_new(
    *args: Any,
    row_size: int | None = None,
    use_jit: bool | None = None,
    use_scan: bool | None = None,
    scan_zero_small_buoyancy: bool | None = None,
    scan_statement_rounding: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> CuascResult:
    use_jit_i = _JIT_CUASC if use_jit is None else bool(use_jit)
    if use_jit_i:
        return _cuasc_new_jit(
            *args,
            row_size=row_size,
            use_scan=use_scan,
            scan_zero_small_buoyancy=scan_zero_small_buoyancy,
            scan_statement_rounding=scan_statement_rounding,
            use_row_conditional_cuadjtq=use_row_conditional_cuadjtq,
            rounding_token=rounding_token,
        )
    return cuasc_new(
        *args,
        row_size=row_size,
        use_scan=use_scan,
        scan_zero_small_buoyancy=scan_zero_small_buoyancy,
        scan_statement_rounding=scan_statement_rounding,
        use_row_conditional_cuadjtq=use_row_conditional_cuadjtq,
        rounding_token=rounding_token,
    )


def cumastr_new(
    klon: int,
    klev: int,
    klevp1: int,
    klevm1: int,
    pten: Any,
    pqen: Any,
    puen: Any,
    pven: Any,
    pverv: Any,
    pqsen: Any,
    pqhfl: Any,
    ztmst: Any,
    pap: Any,
    paph: Any,
    pgeo: Any,
    ptte: Any,
    pqte: Any,
    pvom: Any,
    pvol: Any,
    prsfc: Any,
    pssfc: Any,
    paprc: Any,
    paprsm: Any,
    paprs: Any,
    ldcum: Any,
    ktype: Any,
    kcbot: Any,
    kctop: Any,
    ptu: Any,
    pqu: Any,
    plu: Any,
    plude: Any,
    pmfu: Any,
    pmfd: Any,
    prain: Any,
    psrain: Any,
    psevap: Any,
    psheat: Any,
    psdiss: Any,
    psmelt: Any,
    pcte: Any,
    sig1: Any,
    lndj: Any,
    row_size: int | None = None,
    use_cuasc_jit: bool | None = None,
    use_cuasc_scan: bool | None = None,
    use_cuasc_second_scan: bool | None = None,
    use_row_conditional_cuadjtq: bool | None = None,
    rounding_token: Any | None = None,
) -> CumastrResult:
    """Translate WRF ``cumastr_new`` mass-flux master routine."""

    klon_i = int(klon)
    klev_i = int(klev)
    klevp1_i = int(klevp1)
    klevm1_i = int(klevm1)
    use_cuasc_scan_i = _USE_CUASC_SCAN if use_cuasc_scan is None else bool(use_cuasc_scan)
    use_cuasc_second_scan_i = _USE_CUASC_SECOND_SCAN if use_cuasc_second_scan is None else bool(use_cuasc_second_scan)
    use_row_conditional_i = False if use_row_conditional_cuadjtq is None else bool(use_row_conditional_cuadjtq)
    cols = jnp.arange(klon_i)

    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    puen_arr = jnp.asarray(puen, dtype=jnp.float32)
    pven_arr = jnp.asarray(pven, dtype=jnp.float32)
    pverv_arr = jnp.asarray(pverv, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    pqhfl_arr = jnp.asarray(pqhfl, dtype=jnp.float32)
    pap_arr = jnp.asarray(pap, dtype=jnp.float32)
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    pgeo_arr = jnp.asarray(pgeo, dtype=jnp.float32)
    ptte_out = jnp.asarray(ptte, dtype=jnp.float32)
    pqte_out = jnp.asarray(pqte, dtype=jnp.float32)
    pvom_out = jnp.asarray(pvom, dtype=jnp.float32)
    pvol_out = jnp.asarray(pvol, dtype=jnp.float32)
    prsfc_out = jnp.asarray(prsfc, dtype=jnp.float32)
    pssfc_out = jnp.asarray(pssfc, dtype=jnp.float32)
    paprc_out = jnp.asarray(paprc, dtype=jnp.float32)
    paprsm_arr = jnp.asarray(paprsm, dtype=jnp.float32)
    paprs_out = jnp.asarray(paprs, dtype=jnp.float32)
    ldcum_out = jnp.asarray(ldcum, dtype=bool)
    ktype_out = jnp.asarray(ktype, dtype=jnp.int32)
    ptu_out = jnp.asarray(ptu, dtype=jnp.float32)
    pqu_out = jnp.asarray(pqu, dtype=jnp.float32)
    plu_out = jnp.asarray(plu, dtype=jnp.float32)
    plude_out = jnp.asarray(plude, dtype=jnp.float32)
    pmfu_out = jnp.asarray(pmfu, dtype=jnp.float32)
    pmfd_out = jnp.asarray(pmfd, dtype=jnp.float32)
    prain_out = jnp.asarray(prain, dtype=jnp.float32)
    psrain_out = jnp.asarray(psrain, dtype=jnp.float32)
    psevap_out = jnp.asarray(psevap, dtype=jnp.float32)
    psheat_out = jnp.asarray(psheat, dtype=jnp.float32)
    psdiss_out = jnp.asarray(psdiss, dtype=jnp.float32)
    psmelt_out = jnp.asarray(psmelt, dtype=jnp.float32)
    pcte_out = jnp.asarray(pcte, dtype=jnp.float32)
    sig1_arr = jnp.asarray(sig1, dtype=jnp.float32)
    lndj_arr = jnp.asarray(lndj, dtype=jnp.int32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    gravity_dt = store(G * jnp.asarray(ztmst, dtype=jnp.float32))
    zcons2 = store(jnp.float32(1.0) / gravity_dt)
    init = _profile_tiedtke_stage(
        "cumastr.cuini",
        lambda: cuini(
            klon_i,
            klev_i,
            klevp1_i,
            klevm1_i,
            pten_arr,
            pqen_arr,
            pqsen_arr,
            puen_arr,
            pven_arr,
            pverv_arr,
            pgeo_arr,
            paph_arr,
            rounding_token=rounding_token,
        ),
    )
    zgeoh = init.pgeoh
    ztenh = init.ptenh
    zqenh = init.pqenh
    zqsenh = init.pqsenh
    ilwmin = init.klwmin
    ptu_out = init.ptu
    pqu_out = init.pqu
    ztd = init.ptd
    zqd = init.pqd
    zuu = init.puu
    zvu = init.pvu
    zud = init.pud
    zvd = init.pvd
    pmfu_out = init.pmfu
    pmfd_out = init.pmfd
    zmfus = init.pmfus
    zmfds = init.pmfds
    zmfuq = init.pmfuq
    zmfdq = init.pmfdq
    zmful = jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
    zdmfup = init.pdmfup
    zdmfdp = init.pdmfdp
    zdpmel = init.pdpmel
    plu_out = init.plu
    plude_out = init.plude
    ilab = init.klab

    base = _profile_tiedtke_stage(
        "cumastr.cubase",
        lambda: cubase(
            klon_i,
            klev_i,
            klevp1_i,
            klevm1_i,
            ztenh,
            zqenh,
            zgeoh,
            paph_arr,
            ptu_out,
            pqu_out,
            plu_out,
            puen_arr,
            pven_arr,
            zuu,
            zvu,
            ilab,
            row_size=row_size,
        ),
    )
    ptu_out = base.ptu
    pqu_out = base.pqu
    plu_out = base.plu
    zuu = base.puu
    zvu = base.pvu
    ldcum_out = base.ldcum
    kcbot_out = base.kcbot
    ilab = base.klab

    layer_levels = jnp.arange(2, klev_i + 1, dtype=jnp.int32)
    layer_pressure = store(paph_arr[:, 2 : klev_i + 1] - paph_arr[:, 1:klev_i])
    pqte_layers = store(pqte_out[:, 1:klev_i] * layer_pressure)
    first_layer_pressure = store(paph_arr[:, 1] - paph_arr[:, 0])
    zdqcv = store(pqte_out[:, 0] * first_layer_pressure)
    zdqcv = store(zdqcv + store(jnp.sum(pqte_layers, axis=1)))
    zdqpbl = store(
        jnp.sum(
            jnp.where(layer_levels[None, :] >= kcbot_out[:, None], pqte_layers, jnp.float32(0.0)),
            axis=1,
        )
    )
    idtop = jnp.zeros((klon_i,), dtype=jnp.int32)
    convection_threshold = store(jnp.float32(1.1) * pqhfl_arr)
    convection_threshold = store(convection_threshold * G)
    convection_threshold = store(jnp.maximum(jnp.float32(0.0), convection_threshold))
    ktype_out = jnp.where(zdqcv > convection_threshold, 1, 2)

    ikb_idx = kcbot_out - 1
    pqu_ikb = jnp.take_along_axis(pqu_out, ikb_idx[:, None], axis=1)[:, 0]
    plu_ikb = jnp.take_along_axis(plu_out, ikb_idx[:, None], axis=1)[:, 0]
    zqenh_ikb = jnp.take_along_axis(zqenh, ikb_idx[:, None], axis=1)[:, 0]
    zqumqe = store(pqu_ikb + plu_ikb)
    zqumqe = store(zqumqe - zqenh_ikb)
    zdqmin = store(jnp.float32(0.01) * zqenh_ikb)
    zdqmin = store(jnp.maximum(zdqmin, jnp.float32(1.0e-10)))
    mass_ok = (zdqpbl > 0.0) & (zqumqe > zdqmin) & ldcum_out
    mass_denom = store(jnp.maximum(zqumqe, zdqmin))
    mass_denom = store(G * mass_denom)
    zmfub_candidate = store(zdqpbl / mass_denom)
    zmfub = jnp.where(mass_ok, zmfub_candidate, jnp.float32(0.01))
    ldcum_out = jnp.where(mass_ok, ldcum_out, False)
    paph_ikb = jnp.take_along_axis(paph_arr, ikb_idx[:, None], axis=1)[:, 0]
    paph_ikb_prev = jnp.take_along_axis(paph_arr, jnp.maximum(ikb_idx - 1, 0)[:, None], axis=1)[:, 0]
    zmfmax = store(paph_ikb - paph_ikb_prev)
    zmfmax = store(zmfmax * zcons2)
    zmfub = store(jnp.minimum(zmfub, zmfmax))
    zhcbase = store(CPD * jnp.take_along_axis(ptu_out, ikb_idx[:, None], axis=1)[:, 0])
    zhcbase = store(zhcbase + jnp.take_along_axis(zgeoh, ikb_idx[:, None], axis=1)[:, 0])
    zhcbase = store(zhcbase + store(ALV * pqu_ikb))
    ictop0 = kcbot_out - 1

    zalvdcp = store(store(ALV) / store(CPD))
    zqalv = store(jnp.float32(1.0) / store(ALV))
    zhhatt = jnp.zeros((klon_i, klev_i), dtype=jnp.float32)
    if klevm1_i > 2:
        # Layer enthalpies are independent; only the integer top selection
        # combines levels. Keep every floating-point statement unchanged.
        jk = jnp.arange(klevm1_i, 2, -1, dtype=jnp.int32)
        py = jk - 1
        zhsat = store(CPD * ztenh[:, py])
        zhsat = store(zhsat + zgeoh[:, py])
        zhsat = store(zhsat + store(ALV * zqsenh[:, py]))
        zgam = store(C5LES * zalvdcp)
        zgam = store(zgam * zqsenh[:, py])
        zgam_denom_left = store(jnp.float32(1.0) - store(VTMPc1 * zqsenh[:, py]))
        zgam_denom_right = store(ztenh[:, py] - C4LES)
        zgam_denom_right = store(zgam_denom_right**2)
        zgam_denom = store(zgam_denom_left * zgam_denom_right)
        zgam = store(zgam / zgam_denom)
        zzz = store(CPD * ztenh[:, py])
        zzz = store(zzz * jnp.float32(0.608))
        zhhat_num = store(zzz + store(zgam * zzz))
        zhhat_denom = store(zgam * zzz)
        zhhat_denom = store(zhhat_denom * zqalv)
        zhhat_denom = store(jnp.float32(1.0) + zhhat_denom)
        zhhat_correction = store(zhhat_num / zhhat_denom)
        humidity_deficit = store(zqsenh[:, py] - zqenh[:, py])
        humidity_deficit = store(jnp.maximum(humidity_deficit, jnp.float32(0.0)))
        zhhat_correction = store(zhhat_correction * humidity_deficit)
        zhhat = store(zhsat - zhhat_correction)
        zhhatt = zhhatt.at[:, py].set(zhhat)
        eligible = zhcbase[:, None] > zhhat
        ictop0 = jnp.minimum(
            ictop0, jnp.min(jnp.where(eligible, jk[None, :], ictop0[:, None]), axis=1),
        )

    ikb_idx = kcbot_out - 1
    ztenh_ikb = jnp.take_along_axis(ztenh, ikb_idx[:, None], axis=1)[:, 0]
    zgeoh_ikb = jnp.take_along_axis(zgeoh, ikb_idx[:, None], axis=1)[:, 0]
    zqsenh_ikb = jnp.take_along_axis(zqsenh, ikb_idx[:, None], axis=1)[:, 0]
    zqenh_ikb = jnp.take_along_axis(zqenh, ikb_idx[:, None], axis=1)[:, 0]
    zhsat = store(CPD * ztenh_ikb)
    zhsat = store(zhsat + zgeoh_ikb)
    zhsat = store(zhsat + store(ALV * zqsenh_ikb))
    zgam = store(C5LES * zalvdcp)
    zgam = store(zgam * zqsenh_ikb)
    zgam_denom_left = store(jnp.float32(1.0) - store(VTMPc1 * zqsenh_ikb))
    zgam_denom_right = store(ztenh_ikb - C4LES)
    zgam_denom_right = store(zgam_denom_right**2)
    zgam = store(zgam / store(zgam_denom_left * zgam_denom_right))
    zzz = store(CPD * ztenh_ikb)
    zzz = store(zzz * jnp.float32(0.608))
    zhhat_num = store(zzz + store(zgam * zzz))
    zhhat_denom = store(zgam * zzz)
    zhhat_denom = store(zhhat_denom * zqalv)
    zhhat_denom = store(jnp.float32(1.0) + zhhat_denom)
    zhhat_correction = store(zhhat_num / zhhat_denom)
    humidity_deficit = store(zqsenh_ikb - zqenh_ikb)
    humidity_deficit = store(jnp.maximum(humidity_deficit, jnp.float32(0.0)))
    zhhat = store(zhsat - store(zhhat_correction * humidity_deficit))
    zhhatt = zhhatt.at[cols, ikb_idx].set(zhhat)

    ihmin = jnp.where(ldcum_out & (ktype_out == 1), kcbot_out, -1)
    zhmin = jnp.zeros((klon_i,), dtype=jnp.float32)
    zbi = store(jnp.float32(1.0) / store(jnp.float32(25.0) * G))
    jk = jnp.arange(klev_i, 0, -1, dtype=jnp.int32)
    py = jk - 1
    prev_py = jnp.maximum(py - 1, 0)
    zro = store(RD * ztenh[:, py])
    zro = store(zro / store(G * paph_arr[:, py]))
    zdz = store(paph_arr[:, py] - paph_arr[:, prev_py])
    zdz = store(zdz * zro)
    thermal_gradient = store(pten_arr[:, prev_py] - pten_arr[:, py])
    thermal_gradient = store(CPD * thermal_gradient)
    moisture_gradient = store(pqen_arr[:, prev_py] - pqen_arr[:, py])
    moisture_gradient = store(ALV * moisture_gradient)
    geo_gradient = store(pgeo_arr[:, prev_py] - pgeo_arr[:, py])
    zdhdz = store(thermal_gradient + moisture_gradient)
    zdhdz = store(zdhdz + geo_gradient)
    zdhdz = store(zdhdz * G)
    zdhdz = store(zdhdz / geo_gradient)
    zdepth = store(zgeoh[:, py] - zgeoh_ikb[:, None])
    zfac = store(zdepth * zbi)
    zfac = store(sqrt_for_wrf(store(jnp.float32(1.0) + zfac)))
    zhmin_increment = store(zdhdz * zfac)
    zhmin_increment = store(zhmin_increment * zdz)
    zrh = store(zqsenh[:, py] - zqenh[:, py])
    zrh = store(-ALV * zrh)
    zrh = store(zrh * zfac)
    eligible = ldcum_out[:, None] & (ktype_out[:, None] == 1) & (jk[None, :] < kcbot_out[:, None]) & (jk[None, :] >= ictop0[:, None])
    shape = zhmin_increment.shape
    inputs = (jnp.broadcast_to(jk[None, :], shape).astype(jnp.float32), jnp.broadcast_to(kcbot_out[:, None], shape).astype(jnp.float32), eligible.astype(jnp.float32), zhmin_increment, zrh)
    token = jnp.uint32(0) if rounding_token is None else rounding_token
    (ihmin_float, zhmin), _ = scan_columns(_column_buoyancy_step, (ihmin.astype(jnp.float32), zhmin), tuple((jnp.swapaxes(value, 0, 1) for value in inputs)), token)
    ihmin = ihmin_float.astype(jnp.int32)

    ihmin = jnp.where(ldcum_out & (ktype_out == 1) & (ihmin < ictop0), ictop0, ihmin)
    zentr = jnp.where(ktype_out == 1, ENTRPEN, ENTRSCV)
    zentr = jnp.where(lndj_arr == 1, store(zentr * jnp.float32(1.1)), zentr)

    asc = _profile_tiedtke_stage(
        "cumastr.cuasc_first",
        lambda: _run_cuasc_new(
            klon_i,
            klev_i,
            klevp1_i,
            klevm1_i,
            ztenh,
            zqenh,
            puen_arr,
            pven_arr,
            pten_arr,
            pqen_arr,
            pqsen_arr,
            pgeo_arr,
            zgeoh,
            pap_arr,
            paph_arr,
            pqte_out,
            pverv_arr,
            ilwmin,
            ldcum_out,
            zhcbase,
            ktype_out,
            ilab,
            ptu_out,
            pqu_out,
            plu_out,
            zuu,
            zvu,
            pmfu_out,
            zmfub,
            zentr,
            zmfus,
            zmfuq,
            zmful,
            plude_out,
            zdmfup,
            kcbot_out,
            jnp.asarray(kctop, dtype=jnp.int32),
            ictop0,
            ztmst,
            ihmin,
            zhhatt,
            zqsenh,
            row_size=row_size,
            use_jit=use_cuasc_jit,
            use_scan=use_cuasc_scan_i,
            scan_zero_small_buoyancy=False,
            use_row_conditional_cuadjtq=use_row_conditional_i,
            rounding_token=rounding_token,
        ),
    )
    ldcum_out = asc.ldcum
    ktype_out = asc.ktype
    ilab = asc.klab
    ptu_out = asc.ptu
    pqu_out = asc.pqu
    plu_out = asc.plu
    zuu = asc.puu
    zvu = asc.pvu
    pmfu_out = asc.pmfu
    zmfub = asc.pmfub
    zentr = asc.pentr
    zmfus = asc.pmfus
    zmfuq = asc.pmfuq
    zmful = asc.pmful
    plude_out = asc.plude
    zdmfup = asc.pdmfup
    kcbot_out = asc.kcbot
    kctop_out = asc.kctop
    ictop0 = asc.kctop0
    ihmin = asc.khmin
    if not _is_traced(asc.kcum) and int(jnp.asarray(asc.kcum).item()) == 0:
        return CumastrResult(ptte_out, pqte_out, pvom_out, pvol_out, prsfc_out, pssfc_out, paprc_out, paprs_out, ldcum_out, ktype_out, kcbot_out, kctop_out, ptu_out, pqu_out, plu_out, plude_out, pmfu_out, pmfd_out, prain_out, psrain_out, psevap_out, psheat_out, psdiss_out, psmelt_out, pcte_out)

    zpbmpt = store(
        jnp.take_along_axis(paph_arr, (kcbot_out - 1)[:, None], axis=1)[:, 0]
        - jnp.take_along_axis(paph_arr, (kctop_out - 1)[:, None], axis=1)[:, 0]
    )
    ictop0 = jnp.where(ldcum_out, kctop_out, ictop0)
    ktype_out = jnp.where(ldcum_out & (ktype_out == 1) & (zpbmpt < ZDNOPRC), 2, ktype_out)
    zentr = jnp.where(ktype_out == 2, ENTRSCV, zentr)
    zentr = jnp.where((ktype_out == 2) & (lndj_arr == 1), store(zentr * jnp.float32(1.1)), zentr)
    zrfl = store(jnp.sum(zdmfup, axis=1))

    ztd = ztd
    zqd = zqd
    if LMFDD:
        lfs = _profile_tiedtke_stage(
            "cumastr.cudlfs",
            lambda: cudlfs(
                klon_i,
                klev_i,
                klevp1_i,
                ztenh,
                zqenh,
                puen_arr,
                pven_arr,
                zgeoh,
                paph_arr,
                ptu_out,
                pqu_out,
                zuu,
                zvu,
                ldcum_out,
                kcbot_out,
                kctop_out,
                zmfub,
                zrfl,
                ztd,
                zqd,
                zud,
                zvd,
                pmfd_out,
                zmfds,
                zmfdq,
                zdmfdp,
                row_size=row_size,
                use_row_conditional_cuadjtq=use_row_conditional_i,
            ),
        )
        zrfl = lfs.prfl
        ztd = lfs.ptd
        zqd = lfs.pqd
        zud = lfs.pud
        zvd = lfs.pvd
        pmfd_out = lfs.pmfd
        zmfds = lfs.pmfds
        zmfdq = lfs.pmfdq
        zdmfdp = lfs.pdmfdp
        idtop = lfs.kdtop
        loddraf = lfs.lddraf
        dd = _profile_tiedtke_stage(
            "cumastr.cuddraf",
            lambda: cuddraf(
                klon_i,
                klev_i,
                klevp1_i,
                ztenh,
                zqenh,
                puen_arr,
                pven_arr,
                zgeoh,
                paph_arr,
                zrfl,
                loddraf,
                ztd,
                zqd,
                zud,
                zvd,
                pmfd_out,
                zmfds,
                zmfdq,
                zdmfdp,
                row_size=row_size,
                use_row_conditional_cuadjtq=use_row_conditional_i,
            ),
        )
        zrfl = dd.prfl
        ztd = dd.ptd
        zqd = dd.pqd
        zud = dd.pud
        zvd = dd.pvd
        pmfd_out = dd.pmfd
        zmfds = dd.pmfds
        zmfdq = dd.pmfdq
        zdmfdp = dd.pdmfdp
    else:
        idtop = jnp.full((klon_i,), klevp1_i, dtype=jnp.int32)
        loddraf = jnp.zeros((klon_i,), dtype=bool)

    zheat = jnp.zeros((klon_i,), dtype=jnp.float32)
    zcape = jnp.zeros((klon_i,), dtype=jnp.float32)
    zrelh = jnp.zeros((klon_i,), dtype=jnp.float32)
    zmfub1 = zmfub
    ktop0_arr = jnp.maximum(jnp.int32(12), kctop_out)
    ktop0_paph_idx = jnp.minimum(ktop0_arr, klev_i)
    ktop0_denom = store(paph_arr[cols, kcbot_out] - paph_arr[cols, ktop0_paph_idx])
    heat_levels = jnp.arange(2, klev_i + 1, dtype=jnp.int32)
    py_idx = jnp.arange(1, klev_i, dtype=jnp.int32)
    prev_idx = py_idx - jnp.asarray(1, dtype=jnp.int32)
    deep_mask = ldcum_out & (ktype_out == 1)
    layer_mask = deep_mask[:, None] & (heat_levels[None, :] <= kcbot_out[:, None]) & (heat_levels[None, :] > kctop_out[:, None])
    zro_denom = store(RD * ztenh[:, py_idx])
    zro_layers = store(paph_arr[:, py_idx] / zro_denom)
    zdz_layers = store(paph_arr[:, py_idx] - paph_arr[:, prev_idx])
    zdz_layers = store(zdz_layers / store(G * zro_layers))
    thermal_lapse = store(pten_arr[:, prev_idx] - pten_arr[:, py_idx])
    hydro_lapse = store(G * zdz_layers)
    hydro_lapse = store(hydro_lapse / store(CPD))
    thermal_lapse = store(thermal_lapse + hydro_lapse)
    thermal_lapse = store(thermal_lapse / ztenh[:, py_idx])
    moisture_lapse = store(pqen_arr[:, prev_idx] - pqen_arr[:, py_idx])
    moisture_lapse = store(jnp.float32(0.608) * moisture_lapse)
    zheat_terms = store(thermal_lapse + moisture_lapse)
    zheat_terms = store(zheat_terms * store(pmfu_out[:, py_idx] + pmfd_out[:, py_idx]))
    zheat_terms = store(zheat_terms * G)
    zheat_terms = store(zheat_terms / zro_layers)

    cloud_virtual = store(jnp.float32(0.608) * pqu_out[:, py_idx])
    cloud_virtual = store(jnp.float32(1.0) + cloud_virtual)
    cloud_virtual = store(cloud_virtual - plu_out[:, py_idx])
    cloud_virtual = store(ptu_out[:, py_idx] * cloud_virtual)
    environment_virtual = store(jnp.float32(0.608) * zqenh[:, py_idx])
    environment_virtual = store(jnp.float32(1.0) + environment_virtual)
    environment_virtual = store(ztenh[:, py_idx] * environment_virtual)
    zcape_terms = store(cloud_virtual / environment_virtual)
    zcape_terms = store(zcape_terms - jnp.float32(1.0))
    zcape_terms = store(G * zcape_terms)
    zcape_terms = store(zcape_terms * zdz_layers)
    zheat = store(jnp.sum(jnp.where(layer_mask, zheat_terms, jnp.float32(0.0)), axis=1))
    zcape = store(jnp.sum(jnp.where(layer_mask, zcape_terms, jnp.float32(0.0)), axis=1))
    relh_mask = deep_mask[:, None] & (heat_levels[None, :] <= kcbot_out[:, None]) & (heat_levels[None, :] > ktop0_arr[:, None])
    dept = store(paph_arr[:, py_idx + 1] - paph_arr[:, py_idx])
    dept = store(dept / ktop0_denom[:, None])
    relative_humidity = store(dept * pqen_arr[:, py_idx])
    relative_humidity = store(relative_humidity / pqsen_arr[:, py_idx])
    zrelh = store(jnp.sum(jnp.where(relh_mask, relative_humidity, jnp.float32(0.0)), axis=1))

    deep_mask = ldcum_out & (ktype_out == 1)
    zht_denom = store(ZTAU * jnp.where(zheat != 0.0, zheat, jnp.float32(1.0)))
    zht = store(jnp.maximum(jnp.float32(0.0), zcape))
    zht = store(zht / zht_denom)
    ikb_idx = kcbot_out - 1
    paph_ikb = jnp.take_along_axis(paph_arr, ikb_idx[:, None], axis=1)[:, 0]
    paph_ikb_prev = jnp.take_along_axis(paph_arr, jnp.maximum(ikb_idx - 1, 0)[:, None], axis=1)[:, 0]
    zmfmax = store(paph_ikb - paph_ikb_prev)
    zmfmax = store(zmfmax * zcons2)
    relh_ok = zrelh >= CRIRH
    deep_mass = store(zmfub * zht)
    deep_mass = store(jnp.maximum(deep_mass, jnp.float32(0.01)))
    deep_mass = store(jnp.minimum(deep_mass, zmfmax))
    zmfub1 = jnp.where(deep_mask & relh_ok, deep_mass, zmfub1)
    zmfub1 = jnp.where(deep_mask & (~relh_ok), jnp.float32(0.01), zmfub1)
    zmfub = jnp.where(deep_mask & (~relh_ok), jnp.float32(0.01), zmfub)
    ldcum_out = jnp.where(deep_mask & (~relh_ok), False, ldcum_out)

    nondeep = ktype_out != 1
    ikb_idx = kcbot_out - 1
    pmfd_ikb = jnp.take_along_axis(pmfd_out, ikb_idx[:, None], axis=1)[:, 0]
    zqd_ikb = jnp.take_along_axis(zqd, ikb_idx[:, None], axis=1)[:, 0]
    zeps = jnp.where((pmfd_ikb < 0.0) & loddraf, CMFDEPS, jnp.float32(0.0))
    pqu_ikb = jnp.take_along_axis(pqu_out, ikb_idx[:, None], axis=1)[:, 0]
    plu_ikb = jnp.take_along_axis(plu_out, ikb_idx[:, None], axis=1)[:, 0]
    zqenh_ikb = jnp.take_along_axis(zqenh, ikb_idx[:, None], axis=1)[:, 0]
    zqumqe = store(pqu_ikb + plu_ikb)
    zqumqe = store(zqumqe - store(zeps * zqd_ikb))
    environment_fraction = store(jnp.float32(1.0) - zeps)
    zqumqe = store(zqumqe - store(environment_fraction * zqenh_ikb))
    zdqmin = store(jnp.float32(0.01) * zqenh_ikb)
    zdqmin = store(jnp.maximum(zdqmin, jnp.float32(1.0e-10)))
    candidate_denom = store(jnp.maximum(zqumqe, zdqmin))
    candidate_denom = store(G * candidate_denom)
    candidate = store(zdqpbl / candidate_denom)
    use_candidate = (zdqpbl > 0.0) & (zqumqe > zdqmin) & ldcum_out & (zmfub < zmfmax)
    zmfub1_non = jnp.where(use_candidate, candidate, zmfub)
    mass_delta = store(zmfub1_non - zmfub)
    mass_tolerance = store(jnp.float32(0.2) * zmfub)
    llo1 = (ktype_out == 2) & (jnp.abs(mass_delta) < mass_tolerance)
    zmfub1_non = jnp.where(llo1, zmfub1_non, zmfub)
    zmfub1_non = jnp.minimum(zmfub1_non, zmfmax)
    zmfub1 = jnp.where(nondeep, zmfub1_non, zmfub1)

    zfac_denom = store(jnp.maximum(zmfub, jnp.float32(1.0e-10)))
    zfac = store(zmfub1 / zfac_denom)
    pmfd_out = jnp.where(ldcum_out[:, None], store(pmfd_out * zfac[:, None]), jnp.float32(0.0))
    zmfds = jnp.where(ldcum_out[:, None], store(zmfds * zfac[:, None]), jnp.float32(0.0))
    zmfdq = jnp.where(ldcum_out[:, None], store(zmfdq * zfac[:, None]), jnp.float32(0.0))
    zdmfdp = jnp.where(ldcum_out[:, None], store(zdmfdp * zfac[:, None]), jnp.float32(0.0))
    zmfub = jnp.where(ldcum_out, zmfub1, jnp.float32(0.0))

    asc = _profile_tiedtke_stage(
        "cumastr.cuasc_second",
        lambda: _run_cuasc_new(
            klon_i,
            klev_i,
            klevp1_i,
            klevm1_i,
            ztenh,
            zqenh,
            puen_arr,
            pven_arr,
            pten_arr,
            pqen_arr,
            pqsen_arr,
            pgeo_arr,
            zgeoh,
            pap_arr,
            paph_arr,
            pqte_out,
            pverv_arr,
            ilwmin,
            ldcum_out,
            zhcbase,
            ktype_out,
            ilab,
            ptu_out,
            pqu_out,
            plu_out,
            zuu,
            zvu,
            pmfu_out,
            zmfub,
            zentr,
            zmfus,
            zmfuq,
            zmful,
            plude_out,
            zdmfup,
            kcbot_out,
            kctop_out,
            ictop0,
            ztmst,
            ihmin,
            zhhatt,
            zqsenh,
            row_size=row_size,
            use_jit=use_cuasc_jit,
            use_scan=use_cuasc_scan_i or use_cuasc_second_scan_i,
            scan_zero_small_buoyancy=True,
            use_row_conditional_cuadjtq=use_row_conditional_i,
            rounding_token=rounding_token,
        ),
    )
    ldcum_out = asc.ldcum
    ktype_out = asc.ktype
    ptu_out = asc.ptu
    pqu_out = asc.pqu
    plu_out = asc.plu
    zuu = asc.puu
    zvu = asc.pvu
    pmfu_out = asc.pmfu
    zmfus = asc.pmfus
    zmfuq = asc.pmfuq
    zmful = asc.pmful
    plude_out = asc.plude
    zdmfup = asc.pdmfup
    kcbot_out = asc.kcbot
    kctop_out = asc.kctop

    flux = _profile_tiedtke_stage(
        "cumastr.cuflx",
        lambda: cuflx(
            klon_i,
            klev_i,
            klevp1_i,
            pqen_arr,
            pqsen_arr,
            ztenh,
            zqenh,
            paph_arr,
            zgeoh,
            kcbot_out,
            kctop_out,
            idtop,
            ktype_out,
            loddraf,
            ldcum_out,
            pmfu_out,
            pmfd_out,
            zmfus,
            zmfds,
            zmfuq,
            zmfdq,
            zmful,
            plude_out,
            zdmfup,
            zdmfdp,
            zrfl,
            prain_out,
            pten_arr,
            jnp.zeros((klon_i,), dtype=jnp.float32),
            zdpmel,
            ztmst,
            sig1_arr,
            row_size=row_size,
            rounding_token=rounding_token,
        ),
    )
    ldcum_out = flux.ldcum
    ktype_out = flux.ktype
    pmfu_out = flux.pmfu
    pmfd_out = flux.pmfd
    zmfus = flux.pmfus
    zmfds = flux.pmfds
    zmfuq = flux.pmfuq
    zmfdq = flux.pmfdq
    zmful = flux.pmful
    plude_out = flux.plude
    zdmfup = flux.pdmfup
    zdmfdp = flux.pdmfdp
    zrfl = flux.prfl
    prain_out = flux.prain
    zsfl = flux.psfl
    zdpmel = flux.pdpmel
    itopm2 = flux.ktopm2

    tq = _profile_tiedtke_stage(
        "cumastr.cudtdq",
        lambda: cudtdq(
            klon_i,
            klev_i,
            klevp1_i,
            itopm2,
            paph_arr,
            ldcum_out,
            pten_arr,
            ptte_out,
            pqte_out,
            zmfus,
            zmfds,
            zmfuq,
            zmfdq,
            zmful,
            zdmfup,
            zdmfdp,
            ztmst,
            zdpmel,
            prain_out,
            zrfl,
            zsfl,
            psrain_out,
            psevap_out,
            psheat_out,
            psmelt_out,
            prsfc_out,
            pssfc_out,
            paprc_out,
            paprsm_arr,
            paprs_out,
            pqen_arr,
            pqsen_arr,
            plude_out,
            pcte_out,
            rounding_token=rounding_token,
        ),
    )
    ptte_out = tq.ptte
    pqte_out = tq.pqte
    pcte_out = tq.pcte
    prsfc_out = tq.prsfc
    pssfc_out = tq.pssfc
    paprc_out = tq.paprc
    paprs_out = tq.paprs
    psheat_out = tq.psheat
    psrain_out = tq.psrain
    psevap_out = tq.psevap
    psmelt_out = tq.psmelt

    if LMFDUDV:
        uv = _profile_tiedtke_stage(
            "cumastr.cududv",
            lambda: cududv(
                klon_i,
                klev_i,
                klevp1_i,
                itopm2,
                ktype_out,
                kcbot_out,
                paph_arr,
                ldcum_out,
                puen_arr,
                pven_arr,
                pvom_out,
                pvol_out,
                zuu,
                zud,
                zvu,
                zvd,
                pmfu_out,
                pmfd_out,
                psdiss_out,
                rounding_token=rounding_token,
            ),
        )
        pvom_out = uv.pvom
        pvol_out = uv.pvol
        psdiss_out = uv.psdiss

    return CumastrResult(ptte_out, pqte_out, pvom_out, pvol_out, prsfc_out, pssfc_out, paprc_out, paprs_out, ldcum_out, ktype_out, kcbot_out, kctop_out, ptu_out, pqu_out, plu_out, plude_out, pmfu_out, pmfd_out, prain_out, psrain_out, psevap_out, psheat_out, psdiss_out, psmelt_out, pcte_out)


def cudtdq(
    klon: int,
    klev: int,
    klevp1: int,
    ktopm2: int,
    paph: Any,
    ldcum: Any,
    pten: Any,
    ptte: Any,
    pqte: Any,
    pmfus: Any,
    pmfds: Any,
    pmfuq: Any,
    pmfdq: Any,
    pmful: Any,
    pdmfup: Any,
    pdmfdp: Any,
    ztmst: Any,
    pdpmel: Any,
    prain: Any,
    prfl: Any,
    psfl: Any,
    psrain: Any,
    psevap: Any,
    psheat: Any,
    psmelt: Any,
    prsfc: Any,
    pssfc: Any,
    paprc: Any,
    paprsm: Any,
    paprs: Any,
    pqen: Any,
    pqsen: Any,
    plude: Any,
    pcte: Any,
    rounding_token: Any | None = None,
) -> CudtdqResult:
    """Translate WRF ``cudtdq`` temperature/moisture tendency update."""

    del klon, klevp1
    klev_i = int(klev)
    ktopm2_arr = jnp.asarray(ktopm2, dtype=jnp.int32)
    ktopm2_start = jnp.min(ktopm2_arr)
    ktopm2_i = 1 if _is_traced(ktopm2_start) else int(jnp.asarray(ktopm2_start).item())
    paph_arr = jnp.asarray(paph, dtype=jnp.float32)
    ldcum_arr = jnp.asarray(ldcum, dtype=bool)
    pten_arr = jnp.asarray(pten, dtype=jnp.float32)
    ptte_out = jnp.asarray(ptte, dtype=jnp.float32)
    pqte_out = jnp.asarray(pqte, dtype=jnp.float32)
    pmfus_arr = jnp.asarray(pmfus, dtype=jnp.float32)
    pmfds_arr = jnp.asarray(pmfds, dtype=jnp.float32)
    pmfuq_arr = jnp.asarray(pmfuq, dtype=jnp.float32)
    pmfdq_arr = jnp.asarray(pmfdq, dtype=jnp.float32)
    pmful_arr = jnp.asarray(pmful, dtype=jnp.float32)
    pdmfup_arr = jnp.asarray(pdmfup, dtype=jnp.float32)
    pdmfdp_arr = jnp.asarray(pdmfdp, dtype=jnp.float32)
    pdpmel_arr = jnp.asarray(pdpmel, dtype=jnp.float32)
    prain_arr = jnp.asarray(prain, dtype=jnp.float32)
    prfl_arr = jnp.asarray(prfl, dtype=jnp.float32)
    psfl_arr = jnp.asarray(psfl, dtype=jnp.float32)
    prsfc_out = jnp.asarray(prsfc, dtype=jnp.float32)
    pssfc_out = jnp.asarray(pssfc, dtype=jnp.float32)
    paprc_out = jnp.asarray(paprc, dtype=jnp.float32)
    paprsm_arr = jnp.asarray(paprsm, dtype=jnp.float32)
    paprs_out = jnp.asarray(paprs, dtype=jnp.float32)
    pqen_arr = jnp.asarray(pqen, dtype=jnp.float32)
    pqsen_arr = jnp.asarray(pqsen, dtype=jnp.float32)
    plude_arr = jnp.asarray(plude, dtype=jnp.float32)
    pcte_out = jnp.asarray(pcte, dtype=jnp.float32)
    store = lambda value: _wrf_real(value, rounding_token=rounding_token)

    rhoh2o = store(RHOH2O)
    zdiagw = store(jnp.asarray(ztmst, dtype=jnp.float32) / rhoh2o)
    psrain_out = jnp.asarray(psrain, dtype=jnp.float32)
    psevap_out = jnp.asarray(psevap, dtype=jnp.float32)
    psheat_out = jnp.asarray(psheat, dtype=jnp.float32)
    psmelt_out = jnp.asarray(psmelt, dtype=jnp.float32)

    del ktopm2_i
    ktopm2_vec = jnp.broadcast_to(ktopm2_arr, prain_arr.shape)
    denom = store(paph_arr[:, 1 : klev_i + 1] - paph_arr[:, :klev_i])
    zalv = jnp.where(pten_arr > TMELT, ALV, ALS)
    rhk = store(pqen_arr / pqsen_arr)
    rhk = store(jnp.minimum(jnp.float32(1.0), rhk))
    rhcoe = store(rhk - RHC)
    rhcoe = store(rhcoe / store(RHM - RHC))
    rhcoe = store(jnp.maximum(jnp.float32(0.0), rhcoe))
    pldfd = store(rhcoe * FDBK)
    pldfd = store(pldfd * plude_arr)
    pldfd = store(jnp.maximum(jnp.float32(0.0), pldfd))
    condensate_sink = store(pdmfup_arr + pdmfdp_arr)
    pressure_factor = store(G / denom)
    thermal_factor = store(pressure_factor * RCPD)

    interior = slice(0, klev_i - 1)
    next_levels = slice(1, klev_i)
    thermal_flux = store(pmfus_arr[:, next_levels] - pmfus_arr[:, interior])
    thermal_flux = store(thermal_flux + pmfds_arr[:, next_levels])
    thermal_flux = store(thermal_flux - pmfds_arr[:, interior])
    thermal_flux = store(thermal_flux - store(ALF * pdpmel_arr[:, interior]))
    cloud_flux = store(pmful_arr[:, next_levels] - pmful_arr[:, interior])
    cloud_flux = store(cloud_flux - pldfd[:, interior])
    cloud_flux = store(cloud_flux - condensate_sink[:, interior])
    thermal_flux = store(thermal_flux - store(zalv[:, interior] * cloud_flux))
    zdtdt_interior = store(thermal_factor[:, interior] * thermal_flux)

    moisture_flux = store(pmfuq_arr[:, next_levels] - pmfuq_arr[:, interior])
    moisture_flux = store(moisture_flux + pmfdq_arr[:, next_levels])
    moisture_flux = store(moisture_flux - pmfdq_arr[:, interior])
    moisture_flux = store(moisture_flux + pmful_arr[:, next_levels])
    moisture_flux = store(moisture_flux - pmful_arr[:, interior])
    moisture_flux = store(moisture_flux - pldfd[:, interior])
    moisture_flux = store(moisture_flux - condensate_sink[:, interior])
    zdqdt_interior = store(pressure_factor[:, interior] * moisture_flux)

    bottom = klev_i - 1
    thermal_bottom = store(pmfus_arr[:, bottom] + pmfds_arr[:, bottom])
    thermal_bottom = store(thermal_bottom + store(ALF * pdpmel_arr[:, bottom]))
    cloud_bottom = store(pmful_arr[:, bottom] + condensate_sink[:, bottom])
    cloud_bottom = store(cloud_bottom + pldfd[:, bottom])
    thermal_bottom = store(thermal_bottom - store(zalv[:, bottom] * cloud_bottom))
    zdtdt_bottom = store(-thermal_factor[:, bottom])
    zdtdt_bottom = store(zdtdt_bottom * thermal_bottom)

    moisture_bottom = store(pmfuq_arr[:, bottom] + pmfdq_arr[:, bottom])
    moisture_bottom = store(moisture_bottom + pldfd[:, bottom])
    moisture_bottom = store(moisture_bottom + pmful_arr[:, bottom])
    moisture_bottom = store(moisture_bottom + condensate_sink[:, bottom])
    zdqdt_bottom = store(-pressure_factor[:, bottom])
    zdqdt_bottom = store(zdqdt_bottom * moisture_bottom)

    zdtdt = jnp.concatenate((zdtdt_interior, zdtdt_bottom[:, None]), axis=1)
    zdqdt = jnp.concatenate((zdqdt_interior, zdqdt_bottom[:, None]), axis=1)
    pcte_new = store(pressure_factor * pldfd)
    jk_levels = jnp.arange(1, klev_i + 1, dtype=jnp.int32)
    active = (jk_levels[None, :] >= ktopm2_vec[:, None]) & ldcum_arr[:, None]

    ptte_out = store(ptte_out + jnp.where(active, zdtdt, jnp.float32(0.0)))
    pqte_out = store(pqte_out + jnp.where(active, zdqdt, jnp.float32(0.0)))
    pcte_out = jnp.where(active, pcte_new, pcte_out)
    zsheat_terms = store(zalv * condensate_sink)
    zsheat = store(jnp.sum(jnp.where(active, zsheat_terms, jnp.float32(0.0)), axis=1))
    zmelt = store(jnp.sum(jnp.where(active, pdpmel_arr, jnp.float32(0.0)), axis=1))

    prsfc_out = prfl_arr
    pssfc_out = psfl_arr
    surface_precip = store(prfl_arr + psfl_arr)
    paprc_base, paprc_weight, paprc_precip = lax.optimization_barrier(
        (store(paprc_out), store(zdiagw), surface_precip)
    )
    paprc_increment = store(paprc_weight * paprc_precip)
    paprc_base, paprc_increment = lax.optimization_barrier((paprc_base, paprc_increment))
    paprc_out = store(paprc_base + paprc_increment)
    paprs_out = store(paprsm_arr + store(zdiagw * psfl_arr))
    psheat_out = store(psheat_out + store(jnp.sum(zsheat)))
    psrain_out = store(psrain_out + store(jnp.sum(prain_arr)))
    psevap_out = store(psevap_out - store(jnp.sum(surface_precip)))
    psmelt_out = store(psmelt_out + store(jnp.sum(zmelt)))
    psevap_out = store(psevap_out + psrain_out)

    return CudtdqResult(
        ptte=ptte_out,
        pqte=pqte_out,
        pcte=pcte_out,
        prsfc=prsfc_out,
        pssfc=pssfc_out,
        paprc=paprc_out,
        paprs=paprs_out,
        psheat=psheat_out,
        psrain=psrain_out,
        psevap=psevap_out,
        psmelt=psmelt_out,
    )


__all__ = [
    "CubaseResult",
    "CubasmcResult",
    "CuentrResult",
    "CudtdqResult",
    "CudlfsResult",
    "CuddrafResult",
    "CuflxResult",
    "CuascResult",
    "CududvResult",
    "CuiniResult",
    "cuadjtq",
    "cu_tiedtke",
    "cu_tiedtke_column_core",
    "cubase",
    "cubasmc",
    "cuentr_new",
    "cudlfs",
    "cuddraf",
    "cuflx",
    "cuasc_new",
    "cudtdq",
    "cududv",
    "cuini",
    "tlucua",
    "tlucub",
    "tlucuc",
]
