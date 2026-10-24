"""Vectorized runtime kernels for WRF diffusion/deformation stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from wrf_jax.core.real_arithmetic import real_assignment
import numpy as np


from wrf_jax.core.wrf_math import sqrt_for_wrf
from wrf_jax.dyn_em.module_diffusion_em import (
    CP,
    EP_2,
    G,
    PARAM_FIRST_SCALAR,
    P1000MB,
    PRANDTL,
    P_QC,
    P_QI,
    P_QV,
    R_D,
    SVP1,
    SVP2,
    SVP3,
    SVPT0,
    XLV,
    CalculateKmKhResult,
    DeformAndDivResult,
    HorizontalDiffusion2Result,
    HorizontalDiffusionU2Result,
    HorizontalDiffusionV2Result,
    HorizontalDiffusionW2Result,
    P_M11,
    P_M12,
    P_M13,
    P_M22,
    P_M23,
    TiTauResult,
)


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _flag(config: Any, name: str) -> bool:
    return bool(_read(config, name, False))


def _int_value(config: Any, name: str, default: int = 0) -> int:
    return int(_read(config, name, default))


def _max_f(arr: Any, axis: int, lower: int) -> int:
    return int(lower) + int(arr.shape[axis]) - 1


def _frange(start: int, end: int, low: int, high: int) -> Any:
    first = max(int(start), int(low))
    last = min(int(end), int(high))
    return np.arange(first, max(first, last + 1), dtype=np.int32)


def _span(vals: Any, lower: int) -> slice:
    """Translate a static contiguous Fortran range to a Python slice."""
    if not vals.size:
        return slice(0, 0)
    return slice(int(vals[0]) - int(lower), int(vals[-1]) - int(lower) + 1)


def _take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, ims: int, kms: int, jms: int) -> Any:
    return arr[_span(i_vals, ims), _span(k_vals, kms), _span(j_vals, jms)]


def _take2(arr: Any, i_vals: Any, j_vals: Any, ims: int, jms: int) -> Any:
    return arr[_span(i_vals, ims), _span(j_vals, jms)]


def _set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any, ims: int, kms: int, jms: int) -> Any:
    if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
        return arr
    return arr.at[_span(i_vals, ims), _span(k_vals, kms), _span(j_vals, jms)].set(values)


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    """Materialize a WRF REAL assignment without leaving the JIT graph."""

    return real_assignment(value, rounding_token)


def _real_add(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(jnp.asarray(left, dtype=jnp.float32) + jnp.asarray(right, dtype=jnp.float32), rounding_token)


def _real_sub(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(jnp.asarray(left, dtype=jnp.float32) - jnp.asarray(right, dtype=jnp.float32), rounding_token)


def _real_mul(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(jnp.asarray(left, dtype=jnp.float32) * jnp.asarray(right, dtype=jnp.float32), rounding_token)


def _real_div(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(jnp.asarray(left, dtype=jnp.float32) / jnp.asarray(right, dtype=jnp.float32), rounding_token)


def _take_titau(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, i_lower: int, k_lower: int, j_lower: int) -> Any:
    return _take3(arr, i_vals, k_vals, j_vals, i_lower, k_lower, j_lower)


def _nba_component_index(config_flags: Any, name: str, default_zero_based: int, n_nba_mij: int) -> int:
    fortran_index = int(_read(config_flags, name, default_zero_based + 1))
    index = fortran_index - 1
    if index < 0 or index >= int(n_nba_mij):
        raise ValueError(f"{name}={fortran_index} is outside nba_mij component bounds")
    return index


def _normal_titau_bounds(
    config_flags: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
) -> tuple[int, int, int, int]:
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)
    return i_start, i_end, j_start, j_end


def _u_bounds(config_flags: Any, ids: int, ide: int, jds: int, jde: int, its: int, ite: int, jts: int, jte: int) -> tuple[int, int, int, int]:
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)
    return i_start, i_end, j_start, j_end


def _v_bounds(config_flags: Any, ids: int, ide: int, jds: int, jde: int, its: int, ite: int, jts: int, jte: int) -> tuple[int, int, int, int]:
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)
    return i_start, i_end, j_start, j_end


def cal_deform_and_div_vectorized(
    config_flags: Any,
    u: Any,
    v: Any,
    w: Any,
    div: Any,
    defor11: Any,
    defor22: Any,
    defor33: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    nba_rij: Any | None,
    n_nba_rij: int,
    u_base: Any,
    v_base: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    rdx: float,
    rdy: float,
    dn: Any,
    dnw: Any,
    rdz: Any,
    rdzw: Any,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
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
    rounding_token: Any | None = None,
) -> DeformAndDivResult:
    """Vectorized equivalent of the non-SFS WRF ``cal_deform_and_div`` path."""

    del kds, ime, jme, kme, n_nba_rij
    if _int_value(config_flags, "sfs_opt", 0) > 0:
        raise NotImplementedError("vectorized cal_deform_and_div does not implement NBA/SFS outputs")
    if _flag(config_flags, "polar") or _flag(config_flags, "periodic_x") or _flag(config_flags, "periodic_y"):
        raise NotImplementedError("vectorized cal_deform_and_div currently targets the configured non-periodic real case")
    if int(kts) != 1:
        raise NotImplementedError("vectorized cal_deform_and_div expects WRF bottom kts=1")

    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    w_arr = jnp.asarray(w)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdz_arr = jnp.asarray(rdz)
    rdzw_arr = jnp.asarray(rdzw)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    dn_arr = jnp.asarray(dn)
    dnw_arr = jnp.asarray(dnw)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    u_base_arr = jnp.asarray(u_base)
    v_base_arr = jnp.asarray(v_base)

    div_out = jnp.asarray(div)
    defor11_out = jnp.asarray(defor11)
    defor22_out = jnp.asarray(defor22)
    defor33_out = jnp.asarray(defor33)
    defor12_out = jnp.asarray(defor12)
    defor13_out = jnp.asarray(defor13)
    defor23_out = jnp.asarray(defor23)
    nba_out = None if nba_rij is None else jnp.asarray(nba_rij)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
        store = lambda value: jnp.asarray(value)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)
        store = lambda value: _store_wrf_real(value, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    if ktf < int(kts) or int(kte) < 3:
        return DeformAndDivResult(div_out, defor11_out, defor22_out, defor33_out, defor12_out, defor13_out, defor23_out, nba_out)

    def max_f(arr: Any, axis: int, lower: int) -> int:
        return int(lower) + int(arr.shape[axis]) - 1

    out_i_max = max_f(div_out, 0, ims)
    out_k_max = max_f(div_out, 1, kms)
    out_j_max = max_f(div_out, 2, jms)

    frange = _frange

    def take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        return _take3(arr, i_vals, k_vals, j_vals, ims, kms, jms)

    def take2(arr: Any, i_vals: Any, j_vals: Any) -> Any:
        return _take2(arr, i_vals, j_vals, ims, jms)

    def set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any) -> Any:
        return _set3(arr, i_vals, k_vals, j_vals, values, ims, kms, jms)

    def set3_ij(arr: Any, i_vals: Any, k_value: int, j_vals: Any, values: Any) -> Any:
        if int(i_vals.size) == 0 or int(j_vals.size) == 0:
            return arr
        return arr.at[_span(i_vals, ims), int(k_value) - int(kms), _span(j_vals, jms)].set(values)

    def vertical_average(left: Any, right: Any) -> Any:
        bottom_sum = mul(cf1, left[:, 0, :])
        bottom_sum = add(bottom_sum, mul(cf2, left[:, 1, :]))
        bottom_sum = add(bottom_sum, mul(cf3, left[:, 2, :]))
        bottom_sum = add(bottom_sum, mul(cf1, right[:, 0, :]))
        bottom_sum = add(bottom_sum, mul(cf2, right[:, 1, :]))
        bottom_sum = add(bottom_sum, mul(cf3, right[:, 2, :]))
        bottom = store(mul(0.5, bottom_sum))
        pieces = [bottom[:, None, :]]
        if ktf >= 2:
            mid = jnp.arange(2, ktf + 1, dtype=jnp.int32)
            mid_hat = mid - int(kts)
            current = mul(
                fnm_arr[mid - int(kms)][None, :, None],
                add(left[:, mid_hat, :], right[:, mid_hat, :]),
            )
            previous = mul(
                fnp_arr[mid - int(kms)][None, :, None],
                add(left[:, mid_hat - 1, :], right[:, mid_hat - 1, :]),
            )
            pieces.append(store(mul(0.5, add(current, previous))))
        ktes1 = int(kte) - 1
        ktes2 = int(kte) - 2
        cft2 = store(
            div_real(
                mul(-0.5, dnw_arr[ktes1 - int(kms)]),
                dn_arr[ktes1 - int(kms)],
            )
        )
        cft1 = store(sub(1.0, cft2))
        top_current = mul(
            cft1,
            add(left[:, ktes1 - int(kts), :], right[:, ktes1 - int(kts), :]),
        )
        top_previous = mul(
            cft2,
            add(left[:, ktes2 - int(kts), :], right[:, ktes2 - int(kts), :]),
        )
        top = store(mul(0.5, add(top_current, top_previous)))
        pieces.append(top[:, None, :])
        return jnp.concatenate(pieces, axis=1)

    i_mass = frange(int(its), min(int(ite), int(ide) - 1), int(ims), min(out_i_max, max_f(w_arr, 0, ims)))
    j_mass = frange(int(jts), min(int(jte), int(jde) - 1), int(jms), min(out_j_max, max_f(w_arr, 2, jms)))
    k_mass = frange(int(kts), ktf, int(kms), min(out_k_max, max_f(w_arr, 1, kms) - 1))
    if int(i_mass.size) and int(j_mass.size) and int(k_mass.size):
        i_u = frange(int(its), min(int(ite), int(ide) - 1) + 1, int(ims), max_f(u_arr, 0, ims))
        hat_u = div_real(
            take3(u_arr, i_u, k_mass, j_mass),
            take2(msfuy_arr, i_u, j_mass)[:, None, :],
        )
        havg_u = vertical_average(hat_u[:-1, :, :], hat_u[1:, :, :])
        mm = mul(
            take2(msftx_arr, i_mass, j_mass),
            take2(msfty_arr, i_mass, j_mass),
        )
        tmpzx = add(
            take3(zx_arr, i_mass, k_mass, j_mass),
            take3(zx_arr, i_mass + 1, k_mass, j_mass),
        )
        tmpzx = add(tmpzx, take3(zx_arr, i_mass, k_mass + 1, j_mass))
        tmpzx = add(tmpzx, take3(zx_arr, i_mass + 1, k_mass + 1, j_mass))
        tmpzx = mul(0.25, tmpzx)
        tmp = sub(
            havg_u[:, 1 : int(k_mass.size) + 1, :],
            havg_u[:, : int(k_mass.size), :],
        )
        tmp = mul(tmp, tmpzx)
        tmp = mul(tmp, take3(rdzw_arr, i_mass, k_mass, j_mass))
        horizontal = mul(float(rdx), sub(hat_u[1:, :, :], hat_u[:-1, :, :]))
        val11 = mul(mm[:, None, :], sub(horizontal, tmp))
        defor11_out = set3(defor11_out, i_mass, k_mass, j_mass, mul(2.0, val11))
        div_out = set3(div_out, i_mass, k_mass, j_mass, val11)

        j_v = frange(int(jts), min(int(jte), int(jde) - 1) + 1, int(jms), max_f(v_arr, 2, jms))
        hat_v = div_real(
            take3(v_arr, i_mass, k_mass, j_v),
            take2(msfvx_arr, i_mass, j_v)[:, None, :],
        )
        havg_v = vertical_average(hat_v[:, :, :-1], hat_v[:, :, 1:])
        tmpzy = add(
            take3(zy_arr, i_mass, k_mass, j_mass),
            take3(zy_arr, i_mass, k_mass, j_mass + 1),
        )
        tmpzy = add(tmpzy, take3(zy_arr, i_mass, k_mass + 1, j_mass))
        tmpzy = add(tmpzy, take3(zy_arr, i_mass, k_mass + 1, j_mass + 1))
        tmpzy = mul(0.25, tmpzy)
        tmp = sub(
            havg_v[:, 1 : int(k_mass.size) + 1, :],
            havg_v[:, : int(k_mass.size), :],
        )
        tmp = mul(tmp, tmpzy)
        tmp = mul(tmp, take3(rdzw_arr, i_mass, k_mass, j_mass))
        horizontal = mul(float(rdy), sub(hat_v[:, :, 1:], hat_v[:, :, :-1]))
        val22 = mul(mm[:, None, :], sub(horizontal, tmp))
        defor22_out = set3(defor22_out, i_mass, k_mass, j_mass, mul(2.0, val22))
        div_out = set3(
            div_out,
            i_mass,
            k_mass,
            j_mass,
            add(take3(div_out, i_mass, k_mass, j_mass), val22),
        )

        val33 = mul(
            sub(
                take3(w_arr, i_mass, k_mass + 1, j_mass),
                take3(w_arr, i_mass, k_mass, j_mass),
            ),
            take3(rdzw_arr, i_mass, k_mass, j_mass),
        )
        defor33_out = set3(defor33_out, i_mass, k_mass, j_mass, mul(2.0, val33))
        div_out = set3(
            div_out,
            i_mass,
            k_mass,
            j_mass,
            add(take3(div_out, i_mass, k_mass, j_mass), val33),
        )

    ci0 = int(its)
    ci1 = int(ite)
    cj0 = int(jts)
    cj1 = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        ci0 = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        ci1 = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        cj0 = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        cj1 = min(int(jde) - 1, int(jte))
    i_cross = frange(ci0, ci1, max(int(ims) + 1, int(ims)), min(out_i_max, max_f(v_arr, 0, ims)))
    j_cross = frange(cj0, cj1, max(int(jms) + 1, int(jms)), min(out_j_max, max_f(u_arr, 2, jms)))
    k_cross = frange(int(kts), ktf, int(kms), min(out_k_max, max_f(w_arr, 1, kms) - 1))
    if int(i_cross.size) and int(j_cross.size) and int(k_cross.size):
        mm12 = mul(
            0.25,
            add(
                take2(msfux_arr, i_cross, j_cross - 1),
                take2(msfux_arr, i_cross, j_cross),
            ),
        )
        mm12 = mul(
            mm12,
            add(
                take2(msfvy_arr, i_cross - 1, j_cross),
                take2(msfvy_arr, i_cross, j_cross),
            ),
        )
        j_hat = frange(cj0 - 1, cj1, int(jms), max_f(u_arr, 2, jms))
        hat_u = div_real(
            take3(u_arr, i_cross, k_cross, j_hat),
            take2(msfux_arr, i_cross, j_hat)[:, None, :],
        )
        havg = vertical_average(hat_u[:, :, :-1], hat_u[:, :, 1:])
        rdzw_sum = add(
            take3(rdzw_arr, i_cross, k_cross, j_cross),
            take3(rdzw_arr, i_cross - 1, k_cross, j_cross),
        )
        rdzw_sum = add(rdzw_sum, take3(rdzw_arr, i_cross - 1, k_cross, j_cross - 1))
        rdzw_sum = add(rdzw_sum, take3(rdzw_arr, i_cross, k_cross, j_cross - 1))
        tmpzy = add(
            take3(zy_arr, i_cross - 1, k_cross, j_cross),
            take3(zy_arr, i_cross, k_cross, j_cross),
        )
        tmpzy = add(tmpzy, take3(zy_arr, i_cross - 1, k_cross + 1, j_cross))
        tmpzy = mul(0.25, add(tmpzy, take3(zy_arr, i_cross, k_cross + 1, j_cross)))
        tmp = mul(
            sub(
                havg[:, 1 : int(k_cross.size) + 1, :],
                havg[:, : int(k_cross.size), :],
            ),
            0.25,
        )
        tmp = mul(tmp, tmpzy)
        tmp = mul(tmp, rdzw_sum)
        horizontal = mul(rdy, sub(hat_u[:, :, 1:], hat_u[:, :, :-1]))
        val12 = mul(mm12[:, None, :], sub(horizontal, tmp))
        defor12_out = set3(defor12_out, i_cross, k_cross, j_cross, val12)

        i_hat = frange(ci0 - 1, ci1, int(ims), max_f(v_arr, 0, ims))
        hat_v = div_real(
            take3(v_arr, i_hat, k_cross, j_cross),
            take2(msfvy_arr, i_hat, j_cross)[:, None, :],
        )
        havg = vertical_average(hat_v[:-1, :, :], hat_v[1:, :, :])
        tmpzx = add(
            take3(zx_arr, i_cross, k_cross, j_cross - 1),
            take3(zx_arr, i_cross, k_cross, j_cross),
        )
        tmpzx = add(tmpzx, take3(zx_arr, i_cross, k_cross + 1, j_cross - 1))
        tmpzx = mul(0.25, add(tmpzx, take3(zx_arr, i_cross, k_cross + 1, j_cross)))
        rdzw_sum_v = add(
            take3(rdzw_arr, i_cross, k_cross, j_cross),
            take3(rdzw_arr, i_cross, k_cross, j_cross - 1),
        )
        rdzw_sum_v = add(
            rdzw_sum_v,
            take3(rdzw_arr, i_cross - 1, k_cross, j_cross - 1),
        )
        rdzw_sum_v = add(
            rdzw_sum_v,
            take3(rdzw_arr, i_cross - 1, k_cross, j_cross),
        )
        tmp = mul(
            sub(
                havg[:, 1 : int(k_cross.size) + 1, :],
                havg[:, : int(k_cross.size), :],
            ),
            0.25,
        )
        tmp = mul(tmp, tmpzx)
        tmp = mul(tmp, rdzw_sum_v)
        horizontal = mul(rdx, sub(hat_v[1:, :, :], hat_v[:-1, :, :]))
        second = mul(mm12[:, None, :], sub(horizontal, tmp))
        defor12_out = set3(
            defor12_out,
            i_cross,
            k_cross,
            j_cross,
            add(take3(defor12_out, i_cross, k_cross, j_cross), second),
        )

    k_all = frange(int(kts), int(kte), int(kms), out_k_max)
    j_all = frange(int(jts), int(jte), int(jms), out_j_max)
    i_all = frange(int(its), int(ite), int(ims), out_i_max)
    if ci0 == int(ids) + 1:
        defor12_out = set3(defor12_out, frange(int(ids), int(ids), int(ims), out_i_max), k_all, j_all, take3(defor12_out, frange(int(ids) + 1, int(ids) + 1, int(ims), out_i_max), k_all, j_all))
    if cj0 == int(jds) + 1:
        defor12_out = set3(defor12_out, i_all, k_all, frange(int(jds), int(jds), int(jms), out_j_max), take3(defor12_out, i_all, k_all, frange(int(jds) + 1, int(jds) + 1, int(jms), out_j_max)))
    if ci1 == int(ide) - 1:
        defor12_out = set3(defor12_out, frange(int(ide), int(ide), int(ims), out_i_max), k_all, j_all, take3(defor12_out, frange(int(ide) - 1, int(ide) - 1, int(ims), out_i_max), k_all, j_all))
    if cj1 == int(jde) - 1:
        defor12_out = set3(defor12_out, i_all, k_all, frange(int(jde), int(jde), int(jms), out_j_max), take3(defor12_out, i_all, k_all, frange(int(jde) - 1, int(jde) - 1, int(jms), out_j_max)))

    si0 = int(its)
    si1 = min(int(ite), int(ide) - 1)
    sj0 = int(jts)
    sj1 = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        si0 = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        sj0 = max(int(jds) + 1, int(jts))
    i_shear = frange(si0, si1, max(int(ims) + 1, int(ims)), min(out_i_max, max_f(w_arr, 0, ims)))
    j_shear = frange(sj0, sj1, max(int(jms) + 1, int(jms)), min(out_j_max, max_f(w_arr, 2, jms)))
    k_mid = frange(int(kts) + 1, ktf, int(kms), min(out_k_max, max_f(w_arr, 1, kms) - 1))
    if int(i_shear.size) and int(j_shear.size):
        mm13 = store(
            mul(
                take2(msfux_arr, i_shear, j_shear),
                take2(msfuy_arr, i_shear, j_shear),
            )
        )
        i_hat = frange(si0 - 1, si1, int(ims), max_f(w_arr, 0, ims))
        k_hat = frange(int(kts), int(kte), int(kms), max_f(w_arr, 1, kms))
        hat_w = store(
            div_real(
                take3(w_arr, i_hat, k_hat, j_shear),
                take2(msfty_arr, i_hat, j_shear)[:, None, :],
            )
        )
        hatavg_sum = add(hat_w[1:, :-1, :], hat_w[1:, 1:, :])
        hatavg_sum = add(hatavg_sum, hat_w[:-1, :-1, :])
        hatavg_sum = add(hatavg_sum, hat_w[:-1, 1:, :])
        hatavg = store(mul(0.25, hatavg_sum))
        if int(k_mid.size):
            tmp = sub(
                hatavg[:, 1 : int(k_mid.size) + 1, :],
                hatavg[:, : int(k_mid.size), :],
            )
            tmp = mul(tmp, take3(zx_arr, i_shear, k_mid, j_shear))
            tmp = mul(tmp, 0.5)
            tmp = store(
                mul(
                    tmp,
                    add(
                        take3(rdz_arr, i_shear, k_mid, j_shear),
                        take3(rdz_arr, i_shear - 1, k_mid, j_shear),
                    ),
                )
            )
            horizontal13 = mul(
                rdx,
                sub(
                    hat_w[1:, 1 : int(k_mid.size) + 1, :],
                    hat_w[:-1, 1 : int(k_mid.size) + 1, :],
                ),
            )
            val13 = store(mul(mm13[:, None, :], sub(horizontal13, tmp)))
            if _flag(config_flags, "mix_full_fields"):
                shear13 = sub(
                    take3(u_arr, i_shear, k_mid, j_shear),
                    take3(u_arr, i_shear, k_mid - 1, j_shear),
                )
            else:
                shear13 = sub(
                    take3(u_arr, i_shear, k_mid, j_shear),
                    u_base_arr[k_mid - int(kms)][None, :, None],
                )
                shear13 = sub(
                    shear13,
                    take3(u_arr, i_shear, k_mid - 1, j_shear),
                )
                shear13 = add(
                    shear13,
                    u_base_arr[k_mid - 1 - int(kms)][None, :, None],
                )
            shear13 = mul(shear13, 0.5)
            shear13 = store(
                mul(
                    shear13,
                    add(
                        take3(rdz_arr, i_shear, k_mid, j_shear),
                        take3(rdz_arr, i_shear - 1, k_mid, j_shear),
                    ),
                )
            )
            defor13_out = set3(
                defor13_out,
                i_shear,
                k_mid,
                j_shear,
                store(add(val13, shear13)),
            )
        defor13_out = set3_ij(defor13_out, i_shear, int(kts), j_shear, jnp.zeros((int(i_shear.size), int(j_shear.size)), dtype=defor13_out.dtype))
        defor13_out = set3_ij(defor13_out, i_shear, ktf + 1, j_shear, jnp.zeros((int(i_shear.size), int(j_shear.size)), dtype=defor13_out.dtype))

        mm23 = store(
            mul(
                take2(msfvx_arr, i_shear, j_shear),
                take2(msfvy_arr, i_shear, j_shear),
            )
        )
        j_hat = frange(sj0 - 1, sj1, int(jms), max_f(w_arr, 2, jms))
        hat_w = store(
            div_real(
                take3(w_arr, i_shear, k_hat, j_hat),
                take2(msftx_arr, i_shear, j_hat)[:, None, :],
            )
        )
        hatavg_sum = add(hat_w[:, :-1, 1:], hat_w[:, 1:, 1:])
        hatavg_sum = add(hatavg_sum, hat_w[:, :-1, :-1])
        hatavg_sum = add(hatavg_sum, hat_w[:, 1:, :-1])
        hatavg = store(mul(0.25, hatavg_sum))
        if int(k_mid.size):
            tmp = sub(
                hatavg[:, 1 : int(k_mid.size) + 1, :],
                hatavg[:, : int(k_mid.size), :],
            )
            tmp = mul(tmp, take3(zy_arr, i_shear, k_mid, j_shear))
            tmp = mul(tmp, 0.5)
            tmp = store(
                mul(
                    tmp,
                    add(
                        take3(rdz_arr, i_shear, k_mid, j_shear),
                        take3(rdz_arr, i_shear, k_mid, j_shear - 1),
                    ),
                )
            )
            horizontal23 = mul(
                rdy,
                sub(
                    hat_w[:, 1 : int(k_mid.size) + 1, 1:],
                    hat_w[:, 1 : int(k_mid.size) + 1, :-1],
                ),
            )
            val23 = store(mul(mm23[:, None, :], sub(horizontal23, tmp)))
            if _flag(config_flags, "mix_full_fields"):
                shear23 = sub(
                    take3(v_arr, i_shear, k_mid, j_shear),
                    take3(v_arr, i_shear, k_mid - 1, j_shear),
                )
            else:
                shear23 = sub(
                    take3(v_arr, i_shear, k_mid, j_shear),
                    v_base_arr[k_mid - int(kms)][None, :, None],
                )
                shear23 = sub(
                    shear23,
                    take3(v_arr, i_shear, k_mid - 1, j_shear),
                )
                shear23 = add(
                    shear23,
                    v_base_arr[k_mid - 1 - int(kms)][None, :, None],
                )
            shear23 = mul(shear23, 0.5)
            shear23 = store(
                mul(
                    shear23,
                    add(
                        take3(rdz_arr, i_shear, k_mid, j_shear),
                        take3(rdz_arr, i_shear, k_mid, j_shear - 1),
                    ),
                )
            )
            defor23_out = set3(
                defor23_out,
                i_shear,
                k_mid,
                j_shear,
                store(add(val23, shear23)),
            )
        defor23_out = set3_ij(defor23_out, i_shear, int(kts), j_shear, jnp.zeros((int(i_shear.size), int(j_shear.size)), dtype=defor23_out.dtype))
        defor23_out = set3_ij(defor23_out, i_shear, ktf + 1, j_shear, jnp.zeros((int(i_shear.size), int(j_shear.size)), dtype=defor23_out.dtype))

    k_all = frange(int(kts), int(kte), int(kms), out_k_max)
    j_all = frange(int(jts), int(jte), int(jms), out_j_max)
    i_all = frange(int(its), int(ite), int(ims), out_i_max)
    if si0 == int(ids) + 1:
        ib = frange(int(ids), int(ids), int(ims), out_i_max)
        src13 = take3(defor13_out, frange(int(ids) + 1, int(ids) + 1, int(ims), out_i_max), k_all, j_all)
        src23 = take3(defor23_out, frange(int(ids) + 1, int(ids) + 1, int(ims), out_i_max), k_all, j_all)
        defor13_out = set3(defor13_out, ib, k_all, j_all, src13)
        defor23_out = set3(defor23_out, ib, k_all, j_all, src23)
    if sj0 == int(jds) + 1:
        jb = frange(int(jds), int(jds), int(jms), out_j_max)
        src13 = take3(defor13_out, i_all, k_all, frange(int(jds) + 1, int(jds) + 1, int(jms), out_j_max))
        src23 = take3(defor23_out, i_all, k_all, frange(int(jds) + 1, int(jds) + 1, int(jms), out_j_max))
        defor13_out = set3(defor13_out, i_all, k_all, jb, src13)
        defor23_out = set3(defor23_out, i_all, k_all, jb, src23)
    if si1 == int(ide) - 1:
        ib = frange(int(ide), int(ide), int(ims), out_i_max)
        src13 = take3(defor13_out, frange(int(ide) - 1, int(ide) - 1, int(ims), out_i_max), k_all, j_all)
        src23 = take3(defor23_out, frange(int(ide) - 1, int(ide) - 1, int(ims), out_i_max), k_all, j_all)
        defor13_out = set3(defor13_out, ib, k_all, j_all, src13)
        defor23_out = set3(defor23_out, ib, k_all, j_all, src23)
    if sj1 == int(jde) - 1:
        jb = frange(int(jde), int(jde), int(jms), out_j_max)
        src13 = take3(defor13_out, i_all, k_all, frange(int(jde) - 1, int(jde) - 1, int(jms), out_j_max))
        src23 = take3(defor23_out, i_all, k_all, frange(int(jde) - 1, int(jde) - 1, int(jms), out_j_max))
        defor13_out = set3(defor13_out, i_all, k_all, jb, src13)
        defor23_out = set3(defor23_out, i_all, k_all, jb, src23)

    return DeformAndDivResult(
        div=div_out,
        defor11=defor11_out,
        defor22=defor22_out,
        defor33=defor33_out,
        defor12=defor12_out,
        defor13=defor13_out,
        defor23=defor23_out,
        nba_rij=nba_out,
    )


def calculate_km_kh_vectorized(
    config_flags: Any,
    dt: float,
    dampcoef: float,
    zdamp: float,
    damp_opt: int,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    xkhv: Any,
    bn2: Any,
    khdif: float,
    kvdif: float,
    div: Any,
    defor11: Any,
    defor22: Any,
    defor33: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    tke: Any,
    p8w: Any,
    t8w: Any,
    theta: Any,
    t: Any,
    p: Any,
    moist: Any,
    dn: Any,
    dnw: Any,
    dx: float,
    dy: float,
    rdz: Any,
    rdzw: Any,
    isotropic: int,
    n_moist: int,
    cf1: float,
    cf2: float,
    cf3: float,
    warm_rain: bool,
    mix_upper_bound: float,
    msftx: Any,
    msfty: Any,
    zx: Any,
    zy: Any,
    hpbl: Any,
    dlk: Any,
    xkmv_meso: Any,
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
    rounding_token: Any | None = None,
) -> CalculateKmKhResult:
    """Vectorized WRF ``calculate_km_kh`` for the configured km_opt=4 branch."""

    del dt, dampcoef, zdamp, khdif, kvdif, div, defor33, defor13, defor23, isotropic, warm_rain, mix_upper_bound, dlk
    del kds, ime, jme, kme
    if _int_value(config_flags, "km_opt", 0) != 4:
        raise NotImplementedError("vectorized calculate_km_kh implements km_opt=4")
    if int(damp_opt) == 1:
        raise NotImplementedError("vectorized calculate_km_kh does not implement damp_opt=1")
    if _flag(config_flags, "periodic_x"):
        raise NotImplementedError("vectorized calculate_km_kh currently targets the configured non-periodic real case")
    if int(kts) != 1:
        raise NotImplementedError("vectorized calculate_km_kh expects WRF bottom kts=1")

    xkmh_out = jnp.asarray(xkmh)
    xkmv_out = jnp.asarray(xkmv)
    xkhh_out = jnp.asarray(xkhh)
    xkhv_out = jnp.asarray(xkhv)
    bn2_out = jnp.asarray(bn2)
    moist_arr = jnp.asarray(moist)
    theta_arr = jnp.asarray(theta)
    t_arr = jnp.asarray(t)
    p_arr = jnp.asarray(p)
    p8w_arr = jnp.asarray(p8w)
    t8w_arr = jnp.asarray(t8w)
    dnw_arr = jnp.asarray(dnw)
    dn_arr = jnp.asarray(dn)
    rdz_arr = jnp.asarray(rdz)
    rdzw_arr = jnp.asarray(rdzw)
    defor11_arr = jnp.asarray(defor11)
    defor22_arr = jnp.asarray(defor22)
    defor12_arr = jnp.asarray(defor12)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
        store = lambda value: jnp.asarray(value)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)
        store = lambda value: _store_wrf_real(value, rounding_token)

    def max_f(arr: Any, axis: int, lower: int) -> int:
        return int(lower) + int(arr.shape[axis]) - 1

    out_i_max = max_f(bn2_out, 0, ims)
    out_k_max = max_f(bn2_out, 1, kms)
    out_j_max = max_f(bn2_out, 2, jms)

    frange = _frange

    def kx(values: Any) -> Any:
        return values - int(kms)

    def take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        return _take3(arr, i_vals, k_vals, j_vals, ims, kms, jms)

    def take2(arr: Any, i_vals: Any, j_vals: Any) -> Any:
        return _take2(arr, i_vals, j_vals, ims, jms)

    def take4(species: int, i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        idx = int(species) - 1
        if idx < 0 or idx >= int(moist_arr.shape[3]) or int(species) > int(n_moist):
            return jnp.zeros((int(i_vals.size), int(k_vals.size), int(j_vals.size)), dtype=moist_arr.dtype)
        return moist_arr[_span(i_vals, ims), _span(k_vals, kms), _span(j_vals, jms), idx]

    def set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any) -> Any:
        return _set3(arr, i_vals, k_vals, j_vals, values, ims, kms, jms)

    ktf = min(int(kte), int(kde) - 1)
    i0 = int(its)
    i1 = min(int(ite), int(ide) - 1)
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i0 = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i1 = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j0 = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j1 = min(int(jde) - 2, int(jte))

    i_vals = frange(i0, i1, int(ims), min(out_i_max, max_f(t_arr, 0, ims)))
    j_vals = frange(j0, j1, int(jms), min(out_j_max, max_f(t_arr, 2, jms)))
    k_vals = frange(int(kts), ktf, int(kms), min(out_k_max, max_f(t_arr, 1, kms)))
    if int(i_vals.size) and int(j_vals.size) and int(k_vals.size):
        qv = take4(P_QV, i_vals, k_vals, j_vals)
        qc = take4(P_QC, i_vals, k_vals, j_vals) if P_QC > PARAM_FIRST_SCALAR else jnp.zeros_like(qv)
        tmp1 = jnp.zeros_like(qv)
        for species in (P_QV, P_QC, P_QI):
            if PARAM_FIRST_SCALAR <= int(species) <= int(n_moist):
                tmp1 = tmp1 + take4(species, i_vals, k_vals, j_vals)

        tc = take3(t_arr, i_vals, k_vals, j_vals) - SVPT0
        es = 1000.0 * SVP1 * jnp.exp(SVP2 * tc / (take3(t_arr, i_vals, k_vals, j_vals) - SVP3))
        qvs = EP_2 * es / (take3(p_arr, i_vals, k_vals, j_vals) - es)

        k_mid = frange(int(kts) + 1, ktf - 1, int(kms), min(out_k_max - 1, max_f(t_arr, 1, kms) - 1))
        if int(k_mid.size):
            theta_k = take3(theta_arr, i_vals, k_mid, j_vals)
            t_k = take3(t_arr, i_vals, k_mid, j_vals)
            qv_k = take4(P_QV, i_vals, k_mid, j_vals)
            qc_k = take4(P_QC, i_vals, k_mid, j_vals) if P_QC > PARAM_FIRST_SCALAR else jnp.zeros((int(i_vals.size), int(k_mid.size), int(j_vals.size)), dtype=moist_arr.dtype)
            qvs_k = qvs[:, kx(k_mid), :]
            tmpdz = 1.0 / take3(rdz_arr, i_vals, k_mid, j_vals) + 1.0 / take3(rdz_arr, i_vals, k_mid + 1, j_vals)
            saturated = (qv_k >= qvs_k) | (qc_k >= 0.00001)
            xlvqv = XLV * qv_k
            coefa = (1.0 + xlvqv / R_D / t_k) / (1.0 + XLV * xlvqv / CP / 461.6 / t_k / t_k) / theta_k
            thetaep1 = take3(theta_arr, i_vals, k_mid + 1, j_vals) * (1.0 + XLV * qvs[:, kx(k_mid + 1), :] / CP / take3(t_arr, i_vals, k_mid + 1, j_vals))
            thetaem1 = take3(theta_arr, i_vals, k_mid - 1, j_vals) * (1.0 + XLV * qvs[:, kx(k_mid - 1), :] / CP / take3(t_arr, i_vals, k_mid - 1, j_vals))
            saturated_bn2 = G * (coefa * (thetaep1 - thetaem1) / tmpdz - (tmp1[:, kx(k_mid + 1), :] - tmp1[:, kx(k_mid - 1), :]) / tmpdz)
            dry_bn2 = G * (
                (take3(theta_arr, i_vals, k_mid + 1, j_vals) - take3(theta_arr, i_vals, k_mid - 1, j_vals)) / theta_k / tmpdz
                + 1.61 * (take4(P_QV, i_vals, k_mid + 1, j_vals) - take4(P_QV, i_vals, k_mid - 1, j_vals)) / tmpdz
                - (tmp1[:, kx(k_mid + 1), :] - tmp1[:, kx(k_mid - 1), :]) / tmpdz
            )
            bn2_out = set3(bn2_out, i_vals, k_mid, j_vals, jnp.where(saturated, saturated_bn2, dry_bn2))

        k_bottom = frange(int(kts), int(kts), int(kms), out_k_max)
        if int(k_bottom.size):
            qv1 = take4(P_QV, i_vals, k_bottom, j_vals)
            qc1 = take4(P_QC, i_vals, k_bottom, j_vals) if P_QC > PARAM_FIRST_SCALAR else jnp.zeros_like(qv1)
            tmpdz_moist = 1.0 / take3(rdz_arr, i_vals, k_bottom + 1, j_vals) + 0.5 / take3(rdzw_arr, i_vals, k_bottom, j_vals)
            thetasfc = take3(t8w_arr, i_vals, k_bottom, j_vals) / (take3(p8w_arr, i_vals, k_bottom, j_vals) / P1000MB) ** (R_D / CP)
            saturated = (qv1 >= qvs[:, 0:1, :]) | (qc1 >= 0.00001)
            qvsfc_sat = float(cf1) * qvs[:, 0:1, :] + float(cf2) * qvs[:, 1:2, :] + float(cf3) * qvs[:, 2:3, :]
            xlvqv = XLV * qv1
            coefa = (1.0 + xlvqv / R_D / take3(t_arr, i_vals, k_bottom, j_vals)) / (
                1.0 + XLV * xlvqv / CP / 461.6 / take3(t_arr, i_vals, k_bottom, j_vals) / take3(t_arr, i_vals, k_bottom, j_vals)
            ) / take3(theta_arr, i_vals, k_bottom, j_vals)
            thetaep1 = take3(theta_arr, i_vals, k_bottom + 1, j_vals) * (
                1.0 + XLV * qvs[:, 1:2, :] / CP / take3(t_arr, i_vals, k_bottom + 1, j_vals)
            )
            thetaesfc = thetasfc * (1.0 + XLV * qvsfc_sat / CP / take3(t8w_arr, i_vals, k_bottom, j_vals))
            tmp1sfc = jnp.zeros_like(qv1)
            for species in (P_QV, P_QC, P_QI):
                if PARAM_FIRST_SCALAR <= int(species) <= int(n_moist):
                    tmp1sfc = tmp1sfc + float(cf1) * take4(species, i_vals, k_bottom, j_vals) + float(cf2) * take4(species, i_vals, k_bottom + 1, j_vals) + float(cf3) * take4(species, i_vals, k_bottom + 2, j_vals)
            saturated_bn2 = G * (coefa * (thetaep1 - thetaesfc) / tmpdz_moist - (tmp1[:, 1:2, :] - tmp1sfc) / tmpdz_moist)
            qvsfc_dry = float(cf1) * take4(P_QV, i_vals, k_bottom, j_vals) + float(cf2) * take4(P_QV, i_vals, k_bottom + 1, j_vals) + float(cf3) * take4(P_QV, i_vals, k_bottom + 2, j_vals)
            tmpdz_dry = 1.0 / take3(rdzw_arr, i_vals, k_bottom, j_vals)
            dry_bn2 = G * (
                (take3(theta_arr, i_vals, k_bottom + 1, j_vals) - take3(theta_arr, i_vals, k_bottom, j_vals))
                / take3(theta_arr, i_vals, k_bottom, j_vals)
                / tmpdz_dry
                + 1.61 * (take4(P_QV, i_vals, k_bottom + 1, j_vals) - qvsfc_dry) / tmpdz_dry
                - (tmp1[:, 1:2, :] - tmp1sfc) / tmpdz_dry
            )
            bn2_out = set3(bn2_out, i_vals, k_bottom, j_vals, jnp.where(saturated, saturated_bn2, dry_bn2))

        if ktf - 1 >= int(kts):
            bn2_out = set3(
                bn2_out,
                i_vals,
                frange(ktf, ktf, int(kms), out_k_max),
                j_vals,
                take3(bn2_out, i_vals, frange(ktf - 1, ktf - 1, int(kms), out_k_max), j_vals),
            )

    i_smag = i_vals
    j_smag = j_vals
    k_smag = k_vals
    if int(i_smag.size) and int(j_smag.size) and int(k_smag.size):
        d11 = take3(defor11_arr, i_smag, k_smag, j_smag)
        d22 = take3(defor22_arr, i_smag, k_smag, j_smag)
        d12_sum = add(
            take3(defor12_arr, i_smag, k_smag, j_smag),
            take3(defor12_arr, i_smag, k_smag, j_smag + 1),
        )
        d12_sum = add(d12_sum, take3(defor12_arr, i_smag + 1, k_smag, j_smag))
        d12_sum = add(d12_sum, take3(defor12_arr, i_smag + 1, k_smag, j_smag + 1))
        d12_avg = store(mul(0.25, d12_sum))
        d11_minus_d22 = sub(d11, d22)
        def2 = store(mul(0.25, mul(d11_minus_d22, d11_minus_d22)))
        def2 = store(add(def2, mul(d12_avg, d12_avg)))
        dxm = store(div_real(dx, take2(msftx_arr, i_smag, j_smag)))
        dym = store(div_real(dy, take2(msfty_arr, i_smag, j_smag)))
        mlen_h_sq = div_real(dx, take2(msftx_arr, i_smag, j_smag))
        mlen_h_sq = mul(mlen_h_sq, dy)
        mlen_h_sq = div_real(mlen_h_sq, take2(msfty_arr, i_smag, j_smag))
        mlen_h = store(sqrt_for_wrf(mlen_h_sq))
        strain = store(sqrt_for_wrf(def2))
        c_s = float(_read(config_flags, "c_s", 0.25))
        xkmh_val = mul(c_s, c_s)
        xkmh_val = mul(xkmh_val, mlen_h[:, None, :])
        xkmh_val = mul(xkmh_val, mlen_h[:, None, :])
        xkmh_val = store(mul(xkmh_val, strain))
        xkmh_val = store(jnp.minimum(xkmh_val, mul(10.0, mlen_h[:, None, :])))
        if _int_value(config_flags, "diff_opt", 0) == 2:
            tmpzx = add(
                jnp.abs(take3(zx_arr, i_smag, k_smag, j_smag)),
                jnp.abs(take3(zx_arr, i_smag + 1, k_smag, j_smag)),
            )
            tmpzx = add(tmpzx, jnp.abs(take3(zx_arr, i_smag, k_smag + 1, j_smag)))
            tmpzx = add(tmpzx, jnp.abs(take3(zx_arr, i_smag + 1, k_smag + 1, j_smag)))
            tmpzx = mul(0.25, tmpzx)
            tmpzx = mul(tmpzx, take3(rdzw_arr, i_smag, k_smag, j_smag))
            tmpzx = store(mul(tmpzx, dxm[:, None, :]))
            tmpzy = add(
                jnp.abs(take3(zy_arr, i_smag, k_smag, j_smag)),
                jnp.abs(take3(zy_arr, i_smag, k_smag, j_smag + 1)),
            )
            tmpzy = add(tmpzy, jnp.abs(take3(zy_arr, i_smag, k_smag + 1, j_smag)))
            tmpzy = add(tmpzy, jnp.abs(take3(zy_arr, i_smag, k_smag + 1, j_smag + 1)))
            tmpzy = mul(0.25, tmpzy)
            tmpzy = mul(tmpzy, take3(rdzw_arr, i_smag, k_smag, j_smag))
            tmpzy = store(mul(tmpzy, dym[:, None, :]))
            alpha_sq = add(mul(tmpzx, tmpzx), mul(tmpzy, tmpzy))
            alpha = store(jnp.maximum(sqrt_for_wrf(alpha_sq), 1.0))
            def_limit = store(jnp.maximum(div_real(10.0, mlen_h[:, None, :]), 1.0e-3))
            reduced_large = div_real(xkmh_val, mul(alpha, alpha))
            reduced_normal = div_real(xkmh_val, alpha)
            xkmh_val = store(jnp.where(strain > def_limit, reduced_large, reduced_normal))
        xkmh_out = set3(xkmh_out, i_smag, k_smag, j_smag, xkmh_val)
        xkmv_out = set3(xkmv_out, i_smag, k_smag, j_smag, xkmh_val)
        prandtl = store(PRANDTL)
        xkhh_out = set3(
            xkhh_out,
            i_smag,
            k_smag,
            j_smag,
            store(div_real(xkmh_val, prandtl)),
        )
        xkhv_out = set3(xkhv_out, i_smag, k_smag, j_smag, jnp.zeros_like(xkmh_val))

    return CalculateKmKhResult(
        xkmh=xkmh_out,
        xkmv=xkmv_out,
        xkhh=xkhh_out,
        xkhv=xkhv_out,
        bn2=bn2_out,
        moist=moist_arr,
        tke=jnp.asarray(tke),
        hpbl=jnp.asarray(hpbl),
        xkmv_meso=jnp.asarray(xkmv_meso),
    )


def _configured_mass_bounds(config_flags: Any, ids: int, ide: int, jds: int, jde: int, its: int, ite: int, jts: int, jte: int) -> tuple[int, int, int, int]:
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)
    return i_start, i_end, j_start, j_end


def cal_titau_11_22_33_vectorized(
    config_flags: Any,
    titau: Any,
    tke: Any,
    xkx: Any,
    defor: Any,
    mtau: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
    rounding_token: Any | None = None,
) -> TiTauResult:
    """Vectorized WRF ``cal_titau_11_22_33``."""

    del tke, kds, ime, jme, kme
    titau_out = jnp.asarray(titau)
    mtau_out = jnp.asarray(mtau)
    xkx_arr = jnp.asarray(xkx)
    defor_arr = jnp.asarray(defor)
    rho_arr = jnp.asarray(rho)

    ktf = min(int(kte), int(kde) - 1)
    i0, i1, j0, j1 = _normal_titau_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i0 -= int(is_ext)
    i1 += int(ie_ext)
    j0 -= int(js_ext)
    j1 += int(je_ext)

    i_vals = _frange(i0, i1, int(ims), min(_max_f(xkx_arr, 0, ims), _max_f(titau_out, 0, int(its) - 1)))
    j_vals = _frange(j0, j1, int(jms), min(_max_f(xkx_arr, 2, jms), _max_f(titau_out, 2, int(jts) - 1)))
    k_vals = _frange(int(kts), ktf, int(kms), min(_max_f(xkx_arr, 1, kms), _max_f(titau_out, 1, int(kts))))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return TiTauResult(titau=titau_out, mtau=mtau_out)

    rho_block = _take3(rho_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    xkx_block = _take3(xkx_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    defor_block = _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    if _int_value(config_flags, "sfs_opt", 0) > 0:
        mtau_block = _take3(mtau_out, i_vals, k_vals, j_vals, ims, kms, jms)
        titau_values = (
            rho_block * mtau_block
            if rounding_token is None
            else _real_mul(rho_block, mtau_block, rounding_token)
        )
    elif _int_value(config_flags, "m_opt", 0) == 1:
        if rounding_token is None:
            mtau_values = -xkx_block * defor_block
            titau_values = -rho_block * xkx_block * defor_block
        else:
            mtau_values = _real_mul(-xkx_block, defor_block, rounding_token)
            titau_values = _real_mul(-rho_block, xkx_block, rounding_token)
            titau_values = _real_mul(titau_values, defor_block, rounding_token)
        mtau_out = _set3(mtau_out, i_vals, k_vals, j_vals, mtau_values, ims, kms, jms)
    else:
        if rounding_token is None:
            titau_values = -rho_block * xkx_block * defor_block
        else:
            titau_values = _real_mul(-rho_block, xkx_block, rounding_token)
            titau_values = _real_mul(titau_values, defor_block, rounding_token)
    titau_out = _set3(titau_out, i_vals, k_vals, j_vals, titau_values, int(its) - 1, int(kts), int(jts) - 1)
    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_12_21_vectorized(
    config_flags: Any,
    titau: Any,
    xkx: Any,
    defor: Any,
    mtau: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
    rounding_token: Any | None = None,
) -> TiTauResult:
    """Vectorized WRF ``cal_titau_12_21``."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau)
    mtau_out = jnp.asarray(mtau)
    xkx_arr = jnp.asarray(xkx)
    defor_arr = jnp.asarray(defor)
    rho_arr = jnp.asarray(rho)

    ktf = min(int(kte), int(kde) - 1)
    i0, i1, j0, j1 = _normal_titau_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i0 -= int(is_ext)
    i1 += int(ie_ext)
    j0 -= int(js_ext)
    j1 += int(je_ext)

    i_vals = _frange(i0, i1, int(ims) + 1, min(_max_f(xkx_arr, 0, ims), _max_f(titau_out, 0, int(its) - 1)))
    j_vals = _frange(j0, j1, int(jms) + 1, min(_max_f(xkx_arr, 2, jms), _max_f(titau_out, 2, int(jts) - 1)))
    k_vals = _frange(int(kts), ktf, int(kms), min(_max_f(xkx_arr, 1, kms), _max_f(titau_out, 1, int(kts))))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return TiTauResult(titau=titau_out, mtau=mtau_out)

    rho_im_j = _take3(rho_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms)
    rho_i_j = _take3(rho_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    rho_im_jm = _take3(rho_arr, i_vals - 1, k_vals, j_vals - 1, ims, kms, jms)
    rho_i_jm = _take3(rho_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms)
    xkx_im_j = _take3(xkx_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms)
    xkx_i_j = _take3(xkx_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    xkx_im_jm = _take3(xkx_arr, i_vals - 1, k_vals, j_vals - 1, ims, kms, jms)
    xkx_i_jm = _take3(xkx_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms)
    if rounding_token is None:
        rhoavg = 0.25 * (rho_im_j + rho_i_j + rho_im_jm + rho_i_jm)
        xkxavg = rhoavg * 0.25 * (xkx_im_j + xkx_i_j + xkx_im_jm + xkx_i_jm)
    else:
        rho_sum = _real_add(rho_im_j, rho_i_j, rounding_token)
        rho_sum = _real_add(rho_sum, rho_im_jm, rounding_token)
        rho_sum = _real_add(rho_sum, rho_i_jm, rounding_token)
        rhoavg = _real_mul(0.25, rho_sum, rounding_token)
        xkx_sum = _real_add(xkx_im_j, xkx_i_j, rounding_token)
        xkx_sum = _real_add(xkx_sum, xkx_im_jm, rounding_token)
        xkx_sum = _real_add(xkx_sum, xkx_i_jm, rounding_token)
        xkxavg = _real_mul(rhoavg, 0.25, rounding_token)
        xkxavg = _real_mul(xkxavg, xkx_sum, rounding_token)
    if _int_value(config_flags, "sfs_opt", 0) > 0:
        mtau_block = _take3(mtau_out, i_vals, k_vals, j_vals, ims, kms, jms)
        titau_values = (
            rhoavg * mtau_block
            if rounding_token is None
            else _real_mul(rhoavg, mtau_block, rounding_token)
        )
    else:
        defor_block = _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms)
        titau_values = (
            -xkxavg * defor_block
            if rounding_token is None
            else _real_mul(-xkxavg, defor_block, rounding_token)
        )
        if _int_value(config_flags, "m_opt", 0) == 1:
            mtau_values = (
                titau_values / rhoavg
                if rounding_token is None
                else _real_div(titau_values, rhoavg, rounding_token)
            )
            mtau_out = _set3(mtau_out, i_vals, k_vals, j_vals, mtau_values, ims, kms, jms)
    titau_out = _set3(titau_out, i_vals, k_vals, j_vals, titau_values, int(its) - 1, int(kts), int(jts) - 1)
    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_13_31_vectorized(
    config_flags: Any,
    titau: Any,
    defor: Any,
    mtau: Any,
    xkx: Any,
    fnm: Any,
    fnp: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
    rounding_token: Any | None = None,
) -> TiTauResult:
    """Vectorized WRF ``cal_titau_13_31``."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau)
    mtau_out = jnp.asarray(mtau)
    defor_arr = jnp.asarray(defor)
    xkx_arr = jnp.asarray(xkx)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    i0, i1, j0, j1 = _u_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i0 -= int(is_ext)
    i1 += int(ie_ext)
    j0 -= int(js_ext)
    j1 += int(je_ext)

    i_vals = _frange(i0, i1, int(ims) + 1, min(_max_f(xkx_arr, 0, ims), _max_f(titau_out, 0, int(its) - 1)))
    j_vals = _frange(j0, j1, int(jms), min(_max_f(xkx_arr, 2, jms), _max_f(titau_out, 2, int(jts) - 1)))
    k_vals = _frange(int(kts) + 1, ktf, int(kms) + 1, min(_max_f(xkx_arr, 1, kms), _max_f(titau_out, 1, int(kts))))
    if int(i_vals.size) and int(j_vals.size) and int(k_vals.size):
        fnm_values = fnm_arr[k_vals - int(kms)][None, :, None]
        fnp_values = fnp_arr[k_vals - int(kms)][None, :, None]
        rho_current = mul(
            fnm_values,
            add(
                _take3(rho_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
                _take3(rho_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
        )
        rho_previous = mul(
            fnp_values,
            add(
                _take3(rho_arr, i_vals - 1, k_vals - 1, j_vals, ims, kms, jms),
                _take3(rho_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
            ),
        )
        rhoavg = mul(0.5, add(rho_current, rho_previous))
        xkx_current = mul(
            fnm_values,
            add(
                _take3(xkx_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                _take3(xkx_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
            ),
        )
        xkx_previous = mul(
            fnp_values,
            add(
                _take3(xkx_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
                _take3(xkx_arr, i_vals - 1, k_vals - 1, j_vals, ims, kms, jms),
            ),
        )
        xkxavg = mul(mul(rhoavg, 0.5), add(xkx_current, xkx_previous))
        if _int_value(config_flags, "sfs_opt", 0) > 0:
            titau_values = mul(rhoavg, _take3(mtau_out, i_vals, k_vals, j_vals, ims, kms, jms))
        else:
            titau_values = mul(-xkxavg, _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms))
            if _int_value(config_flags, "m_opt", 0) == 1:
                mtau_values = div_real(
                    mul(-xkxavg, _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms)),
                    rhoavg,
                )
                mtau_out = _set3(mtau_out, i_vals, k_vals, j_vals, mtau_values, ims, kms, jms)
        titau_out = _set3(titau_out, i_vals, k_vals, j_vals, titau_values, int(its) - 1, int(kts), int(jts) - 1)

    i_zero = _frange(i0, i1, int(ims) + 1, _max_f(titau_out, 0, int(its) - 1))
    j_zero = _frange(j0, j1, int(jms), _max_f(titau_out, 2, int(jts) - 1))
    for k_zero in (int(kts), ktf + 1):
        k_arr = _frange(k_zero, k_zero, int(kts), _max_f(titau_out, 1, int(kts)))
        titau_out = _set3(titau_out, i_zero, k_arr, j_zero, jnp.zeros((int(i_zero.size), int(k_arr.size), int(j_zero.size)), dtype=titau_out.dtype), int(its) - 1, int(kts), int(jts) - 1)
    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_23_32_vectorized(
    config_flags: Any,
    titau: Any,
    defor: Any,
    mtau: Any,
    xkx: Any,
    fnm: Any,
    fnp: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
    rounding_token: Any | None = None,
) -> TiTauResult:
    """Vectorized WRF ``cal_titau_23_32``."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau)
    mtau_out = jnp.asarray(mtau)
    defor_arr = jnp.asarray(defor)
    xkx_arr = jnp.asarray(xkx)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    i0, i1, j0, j1 = _v_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i0 -= int(is_ext)
    i1 += int(ie_ext)
    j0 -= int(js_ext)
    j1 += int(je_ext)

    i_vals = _frange(i0, i1, int(ims), min(_max_f(xkx_arr, 0, ims), _max_f(titau_out, 0, int(its) - 1)))
    j_vals = _frange(j0, j1, int(jms) + 1, min(_max_f(xkx_arr, 2, jms), _max_f(titau_out, 2, int(jts) - 1)))
    k_vals = _frange(int(kts) + 1, ktf, int(kms) + 1, min(_max_f(xkx_arr, 1, kms), _max_f(titau_out, 1, int(kts))))
    if int(i_vals.size) and int(j_vals.size) and int(k_vals.size):
        fnm_values = fnm_arr[k_vals - int(kms)][None, :, None]
        fnp_values = fnp_arr[k_vals - int(kms)][None, :, None]
        rho_current = mul(
            fnm_values,
            add(
                _take3(rho_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                _take3(rho_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
            ),
        )
        rho_previous = mul(
            fnp_values,
            add(
                _take3(rho_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
                _take3(rho_arr, i_vals, k_vals - 1, j_vals - 1, ims, kms, jms),
            ),
        )
        rhoavg = mul(0.5, add(rho_current, rho_previous))
        xkx_current = mul(
            fnm_values,
            add(
                _take3(xkx_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                _take3(xkx_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
            ),
        )
        xkx_previous = mul(
            fnp_values,
            add(
                _take3(xkx_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
                _take3(xkx_arr, i_vals, k_vals - 1, j_vals - 1, ims, kms, jms),
            ),
        )
        xkxavg = mul(mul(rhoavg, 0.5), add(xkx_current, xkx_previous))
        if _int_value(config_flags, "sfs_opt", 0) > 0:
            titau_values = mul(rhoavg, _take3(mtau_out, i_vals, k_vals, j_vals, ims, kms, jms))
        else:
            titau_values = mul(-xkxavg, _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms))
            if _int_value(config_flags, "m_opt", 0) == 1:
                mtau_values = div_real(
                    mul(-xkxavg, _take3(defor_arr, i_vals, k_vals, j_vals, ims, kms, jms)),
                    rhoavg,
                )
                mtau_out = _set3(mtau_out, i_vals, k_vals, j_vals, mtau_values, ims, kms, jms)
        titau_out = _set3(titau_out, i_vals, k_vals, j_vals, titau_values, int(its) - 1, int(kts), int(jts) - 1)

    i_zero = _frange(i0, i1, int(ims), _max_f(titau_out, 0, int(its) - 1))
    j_zero = _frange(j0, j1, int(jms) + 1, _max_f(titau_out, 2, int(jts) - 1))
    for k_zero in (int(kts), ktf + 1):
        k_arr = _frange(k_zero, k_zero, int(kts), _max_f(titau_out, 1, int(kts)))
        titau_out = _set3(titau_out, i_zero, k_arr, j_zero, jnp.zeros((int(i_zero.size), int(k_arr.size), int(j_zero.size)), dtype=titau_out.dtype), int(its) - 1, int(kts), int(jts) - 1)
    return TiTauResult(titau=titau_out, mtau=mtau_out)


def horizontal_diffusion_s_vectorized(
    tendency: Any,
    config_flags: Any,
    var: Any,
    msftx: Any,
    msfty: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    xkhh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
    rdz: Any,
    rdzw: Any,
    dnw: Any,
    dn: Any,
    rho: Any,
    doing_tke: bool,
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
    rounding_token: Any | None = None,
) -> Any:
    """Vectorized WRF ``horizontal_diffusion_s`` for non-periodic mass fields."""

    del msfuy, msfvx, rdz, kds, ime, jme, kme
    if _flag(config_flags, "periodic_x"):
        raise NotImplementedError("vectorized horizontal_diffusion_s currently targets the configured non-periodic real case")

    tend = jnp.asarray(tendency)
    var_arr = jnp.asarray(var)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    msfux_arr = jnp.asarray(msfux)
    msfvy_arr = jnp.asarray(msfvy)
    xkhh_arr = jnp.asarray(xkhh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    dnw_arr = jnp.asarray(dnw)
    dn_arr = jnp.asarray(dn)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div = lambda left, right: _real_div(left, right, rounding_token)
    store = (
        (lambda value: jnp.asarray(value))
        if rounding_token is None
        else (lambda value: _store_wrf_real(value, rounding_token))
    )

    def max_f(arr: Any, axis: int, lower: int) -> int:
        return int(lower) + int(arr.shape[axis]) - 1

    def frange(start: int, end: int, low: int, high: int) -> Any:
        first = max(int(start), int(low))
        last = min(int(end), int(high))
        return np.arange(first, max(first, last + 1), dtype=np.int32)

    def span(vals: Any, lower: int) -> slice:
        # These ranges depend only on static tile bounds, not model arrays.
        return slice(int(vals[0]) - int(lower), int(vals[-1]) - int(lower) + 1)

    def take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        return arr[span(i_vals, ims), span(k_vals, kms), span(j_vals, jms)]

    def take2(arr: Any, i_vals: Any, j_vals: Any) -> Any:
        return arr[span(i_vals, ims), span(j_vals, jms)]

    def set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any) -> Any:
        if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
            return arr
        return arr.at[span(i_vals, ims), span(k_vals, kms), span(j_vals, jms)].set(values)

    ktf = min(int(kte), int(kde) - 1)
    i0, i1, j0, j1 = _configured_mass_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i_vals = frange(i0, i1, int(ims), min(max_f(tend, 0, ims), max_f(var_arr, 0, ims)))
    j_vals = frange(j0, j1, int(jms), min(max_f(tend, 2, jms), max_f(var_arr, 2, jms)))
    k_vals = frange(int(kts), ktf, int(kms), min(max_f(tend, 1, kms), max_f(var_arr, 1, kms)))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return tend

    ktes1 = int(kte) - 1
    ktes2 = int(kte) - 2

    i_edge = frange(i0, i1 + 1, int(ims) + 1, max_f(var_arr, 0, ims))
    j_edge = frange(j0, j1 + 1, int(jms) + 1, max_f(var_arr, 2, jms))

    bottom = np.asarray([int(kts)], dtype=np.int32)
    var_i_bottom = take3(var_arr, i_edge, bottom, j_vals)
    var_im_bottom = take3(var_arr, i_edge - 1, bottom, j_vals)
    var_i_bottom_p1 = take3(var_arr, i_edge, bottom + 1, j_vals)
    var_im_bottom_p1 = take3(var_arr, i_edge - 1, bottom + 1, j_vals)
    var_i_bottom_p2 = take3(var_arr, i_edge, bottom + 2, j_vals)
    var_im_bottom_p2 = take3(var_arr, i_edge - 1, bottom + 2, j_vals)
    h1_bottom_sum = mul(cf1, var_i_bottom)
    h1_bottom_sum = add(h1_bottom_sum, mul(cf2, var_i_bottom_p1))
    h1_bottom_sum = add(h1_bottom_sum, mul(cf3, var_i_bottom_p2))
    h1_bottom_sum = add(h1_bottom_sum, mul(cf1, var_im_bottom))
    h1_bottom_sum = add(h1_bottom_sum, mul(cf2, var_im_bottom_p1))
    h1_bottom_sum = add(h1_bottom_sum, mul(cf3, var_im_bottom_p2))
    h1_bottom = store(mul(0.5, h1_bottom_sum))
    top_k = np.asarray([ktes1], dtype=np.int32)
    top_m1_k = np.asarray([ktes2], dtype=np.int32)
    var_i_top = take3(var_arr, i_edge, top_k, j_vals)
    var_i_top_m1 = take3(var_arr, i_edge, top_m1_k, j_vals)
    var_im_top = take3(var_arr, i_edge - 1, top_k, j_vals)
    var_im_top_m1 = take3(var_arr, i_edge - 1, top_m1_k, j_vals)
    top_dnw = dnw_arr[ktes1 - int(kms)]
    top_dn = dn_arr[ktes1 - int(kms)]
    i_top_extrap = div(mul(mul(sub(var_i_top, var_i_top_m1), 0.5), top_dnw), top_dn)
    im_top_extrap = div(mul(mul(sub(var_im_top, var_im_top_m1), 0.5), top_dnw), top_dn)
    h1_top_sum = add(var_i_top, i_top_extrap)
    h1_top_sum = add(h1_top_sum, var_im_top)
    h1_top_sum = add(h1_top_sum, im_top_extrap)
    h1_top = store(mul(0.5, h1_top_sum))
    var_j_bottom = take3(var_arr, i_vals, bottom, j_edge)
    var_jm_bottom = take3(var_arr, i_vals, bottom, j_edge - 1)
    var_j_bottom_p1 = take3(var_arr, i_vals, bottom + 1, j_edge)
    var_jm_bottom_p1 = take3(var_arr, i_vals, bottom + 1, j_edge - 1)
    var_j_bottom_p2 = take3(var_arr, i_vals, bottom + 2, j_edge)
    var_jm_bottom_p2 = take3(var_arr, i_vals, bottom + 2, j_edge - 1)
    h2_bottom_sum = mul(cf1, var_j_bottom)
    h2_bottom_sum = add(h2_bottom_sum, mul(cf2, var_j_bottom_p1))
    h2_bottom_sum = add(h2_bottom_sum, mul(cf3, var_j_bottom_p2))
    h2_bottom_sum = add(h2_bottom_sum, mul(cf1, var_jm_bottom))
    h2_bottom_sum = add(h2_bottom_sum, mul(cf2, var_jm_bottom_p1))
    h2_bottom_sum = add(h2_bottom_sum, mul(cf3, var_jm_bottom_p2))
    h2_bottom = store(mul(0.5, h2_bottom_sum))
    var_j_top = take3(var_arr, i_vals, top_k, j_edge)
    var_j_top_m1 = take3(var_arr, i_vals, top_m1_k, j_edge)
    var_jm_top = take3(var_arr, i_vals, top_k, j_edge - 1)
    var_jm_top_m1 = take3(var_arr, i_vals, top_m1_k, j_edge - 1)
    j_top_extrap = div(mul(mul(sub(var_j_top, var_j_top_m1), 0.5), top_dnw), top_dn)
    jm_top_extrap = div(mul(mul(sub(var_jm_top, var_jm_top_m1), 0.5), top_dnw), top_dn)
    h2_top_sum = add(var_j_top, j_top_extrap)
    h2_top_sum = add(h2_top_sum, var_jm_top)
    h2_top_sum = add(h2_top_sum, jm_top_extrap)
    h2_top = store(mul(0.5, h2_top_sum))

    if ktf >= int(kts) + 1:
        mid = np.arange(int(kts) + 1, ktf + 1, dtype=np.int32)
        h1_mid_current = mul(
            fnm_arr[mid - int(kms)][None, :, None],
            add(take3(var_arr, i_edge, mid, j_vals), take3(var_arr, i_edge - 1, mid, j_vals)),
        )
        h1_mid_previous = mul(
            fnp_arr[mid - int(kms)][None, :, None],
            add(take3(var_arr, i_edge, mid - 1, j_vals), take3(var_arr, i_edge - 1, mid - 1, j_vals)),
        )
        h1_mid = store(mul(0.5, add(h1_mid_current, h1_mid_previous)))
        h2_mid_current = mul(
            fnm_arr[mid - int(kms)][None, :, None],
            add(take3(var_arr, i_vals, mid, j_edge), take3(var_arr, i_vals, mid, j_edge - 1)),
        )
        h2_mid_previous = mul(
            fnp_arr[mid - int(kms)][None, :, None],
            add(take3(var_arr, i_vals, mid - 1, j_edge), take3(var_arr, i_vals, mid - 1, j_edge - 1)),
        )
        h2_mid = store(mul(0.5, add(h2_mid_current, h2_mid_previous)))
        h1avg = jnp.concatenate([h1_bottom, h1_mid, h1_top], axis=1)
        h2avg = jnp.concatenate([h2_bottom, h2_mid, h2_top], axis=1)
    else:
        h1avg = jnp.concatenate([h1_bottom, h1_top], axis=1)
        h2avg = jnp.concatenate([h2_bottom, h2_top], axis=1)

    xkhh_h1_im = take3(xkhh_arr, i_edge - 1, k_vals, j_vals)
    xkhh_h1_i = take3(xkhh_arr, i_edge, k_vals, j_vals)
    rho_h1_im = take3(rho_arr, i_edge - 1, k_vals, j_vals)
    rho_h1_i = take3(rho_arr, i_edge, k_vals, j_vals)
    xkxavg_h1 = store(mul(mul(0.5, add(xkhh_h1_im, xkhh_h1_i)), mul(0.5, add(rho_h1_im, rho_h1_i))))
    tmpzx_h1 = store(mul(0.5, add(take3(zx_arr, i_edge, k_vals, j_vals), take3(zx_arr, i_edge, k_vals + 1, j_vals))))
    rdzw_h1_i = take3(rdzw_arr, i_edge, k_vals, j_vals)
    rdzw_h1_im = take3(rdzw_arr, i_edge - 1, k_vals, j_vals)
    rdzu = store(div(2.0, add(div(1.0, rdzw_h1_i), div(1.0, rdzw_h1_im))))
    var_h1_i = take3(var_arr, i_edge, k_vals, j_vals)
    var_h1_im = take3(var_arr, i_edge - 1, k_vals, j_vals)
    h1_horizontal = mul(rdx, sub(var_h1_i, var_h1_im))
    h1_vertical = mul(tmpzx_h1, sub(h1avg[:, 1 : int(k_vals.size) + 1, :], h1avg[:, : int(k_vals.size), :]))
    h1_vertical = mul(h1_vertical, rdzu)
    h1 = store(mul(mul(-take2(msfux_arr, i_edge, j_vals)[:, None, :], xkxavg_h1), sub(h1_horizontal, h1_vertical)))

    xkhh_h2_jm = take3(xkhh_arr, i_vals, k_vals, j_edge - 1)
    xkhh_h2_j = take3(xkhh_arr, i_vals, k_vals, j_edge)
    rho_h2_jm = take3(rho_arr, i_vals, k_vals, j_edge - 1)
    rho_h2_j = take3(rho_arr, i_vals, k_vals, j_edge)
    xkxavg_h2 = store(mul(mul(0.5, add(xkhh_h2_jm, xkhh_h2_j)), mul(0.5, add(rho_h2_jm, rho_h2_j))))
    tmpzy_h2 = store(mul(0.5, add(take3(zy_arr, i_vals, k_vals, j_edge), take3(zy_arr, i_vals, k_vals + 1, j_edge))))
    rdzw_h2_j = take3(rdzw_arr, i_vals, k_vals, j_edge)
    rdzw_h2_jm = take3(rdzw_arr, i_vals, k_vals, j_edge - 1)
    rdzv = store(div(2.0, add(div(1.0, rdzw_h2_j), div(1.0, rdzw_h2_jm))))
    var_h2_j = take3(var_arr, i_vals, k_vals, j_edge)
    var_h2_jm = take3(var_arr, i_vals, k_vals, j_edge - 1)
    h2_horizontal = mul(rdy, sub(var_h2_j, var_h2_jm))
    h2_vertical = mul(tmpzy_h2, sub(h2avg[:, 1 : int(k_vals.size) + 1, :], h2avg[:, : int(k_vals.size), :]))
    h2_vertical = mul(h2_vertical, rdzv)
    h2 = store(mul(mul(-take2(msfvy_arr, i_vals, j_edge)[:, None, :], xkxavg_h2), sub(h2_horizontal, h2_vertical)))

    zero_mass_edge = jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=tend.dtype)
    if ktf >= int(kts) + 1:
        mid = np.arange(int(kts) + 1, ktf + 1, dtype=np.int32)
        h1mass_current = mul(
            fnm_arr[mid - int(kms)][None, :, None],
            add(h1[1:, mid - int(kts), :], h1[:-1, mid - int(kts), :]),
        )
        h1mass_previous = mul(
            fnp_arr[mid - int(kms)][None, :, None],
            add(h1[1:, mid - 1 - int(kts), :], h1[:-1, mid - 1 - int(kts), :]),
        )
        h1mass_mid = store(mul(0.5, add(h1mass_current, h1mass_previous)))
        h2mass_current = mul(
            fnm_arr[mid - int(kms)][None, :, None],
            add(h2[:, mid - int(kts), 1:], h2[:, mid - int(kts), :-1]),
        )
        h2mass_previous = mul(
            fnp_arr[mid - int(kms)][None, :, None],
            add(h2[:, mid - 1 - int(kts), 1:], h2[:, mid - 1 - int(kts), :-1]),
        )
        h2mass_mid = store(mul(0.5, add(h2mass_current, h2mass_previous)))
        h1mass = jnp.concatenate([zero_mass_edge, h1mass_mid, zero_mass_edge], axis=1)
        h2mass = jnp.concatenate([zero_mass_edge, h2mass_mid, zero_mass_edge], axis=1)
    else:
        h1mass = jnp.concatenate([zero_mass_edge, zero_mass_edge], axis=1)
        h2mass = h1mass

    zx_at_m_sum = add(
        take3(zx_arr, i_vals, k_vals, j_vals),
        take3(zx_arr, i_vals + 1, k_vals, j_vals),
    )
    zx_at_m_sum = add(zx_at_m_sum, take3(zx_arr, i_vals, k_vals + 1, j_vals))
    zx_at_m_sum = add(zx_at_m_sum, take3(zx_arr, i_vals + 1, k_vals + 1, j_vals))
    zx_at_m = store(mul(0.25, zx_at_m_sum))
    zy_at_m_sum = add(
        take3(zy_arr, i_vals, k_vals, j_vals),
        take3(zy_arr, i_vals, k_vals, j_vals + 1),
    )
    zy_at_m_sum = add(zy_at_m_sum, take3(zy_arr, i_vals, k_vals + 1, j_vals))
    zy_at_m_sum = add(zy_at_m_sum, take3(zy_arr, i_vals, k_vals + 1, j_vals + 1))
    zy_at_m = store(mul(0.25, zy_at_m_sum))
    rdzw_mass = take3(rdzw_arr, i_vals, k_vals, j_vals)
    msftx_mass = take2(msftx_arr, i_vals, j_vals)[:, None, :]
    msfty_mass = take2(msfty_arr, i_vals, j_vals)[:, None, :]
    old = take3(tend, i_vals, k_vals, j_vals)
    if rounding_token is None:
        increment = G / (dnw_arr[k_vals - int(kms)][None, :, None] * rdzw_mass) * (
            msftx_mass * float(rdx) * (h1[1:, :, :] - h1[:-1, :, :])
            + msfty_mass * float(rdy) * (h2[:, :, 1:] - h2[:, :, :-1])
            - msftx_mass
            * zx_at_m
            * (h1mass[:, 1 : int(k_vals.size) + 1, :] - h1mass[:, : int(k_vals.size), :])
            * rdzw_mass
            - msfty_mass
            * zy_at_m
            * (h2mass[:, 1 : int(k_vals.size) + 1, :] - h2mass[:, : int(k_vals.size), :])
            * rdzw_mass
        )
        new_values = old + increment
    else:
        mrdx = _real_mul(msftx_mass, rdx, rounding_token)
        mrdy = _real_mul(msfty_mass, rdy, rounding_token)
        h1_delta = _real_sub(h1[1:, :, :], h1[:-1, :, :], rounding_token)
        h2_delta = _real_sub(h2[:, :, 1:], h2[:, :, :-1], rounding_token)
        h1mass_delta = _real_sub(
            h1mass[:, 1 : int(k_vals.size) + 1, :],
            h1mass[:, : int(k_vals.size), :],
            rounding_token,
        )
        h2mass_delta = _real_sub(
            h2mass[:, 1 : int(k_vals.size) + 1, :],
            h2mass[:, : int(k_vals.size), :],
            rounding_token,
        )
        term1 = _real_mul(mrdx, h1_delta, rounding_token)
        term2 = _real_mul(mrdy, h2_delta, rounding_token)
        term3 = _real_mul(msftx_mass, zx_at_m, rounding_token)
        term3 = _real_mul(term3, h1mass_delta, rounding_token)
        term3 = _real_mul(term3, rdzw_mass, rounding_token)
        term4 = _real_mul(msfty_mass, zy_at_m, rounding_token)
        term4 = _real_mul(term4, h2mass_delta, rounding_token)
        term4 = _real_mul(term4, rdzw_mass, rounding_token)
        bracket = _real_add(term1, term2, rounding_token)
        bracket = _real_sub(bracket, term3, rounding_token)
        bracket = _real_sub(bracket, term4, rounding_token)
        denominator = _real_mul(
            dnw_arr[k_vals - int(kms)][None, :, None], rdzw_mass, rounding_token
        )
        increment = _real_mul(_real_div(G, denominator, rounding_token), bracket, rounding_token)
        new_values = _real_add(old, increment, rounding_token)
    if bool(doing_tke):
        new_values = old + 2.0 * increment
    return set3(tend, i_vals, k_vals, j_vals, new_values)


def horizontal_diffusion_u_2_vectorized(
    tendency: Any,
    config_flags: Any,
    defor11: Any,
    defor12: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msfux: Any,
    msfuy: Any,
    xkmh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dnw: Any,
    zx: Any,
    zy: Any,
    rdzw: Any,
    rho: Any,
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
    rounding_token: Any | None = None,
) -> HorizontalDiffusionU2Result:
    """Vectorized WRF ``horizontal_diffusion_u_2``."""

    del div, kds, ime, jme, kme
    tend = jnp.asarray(tendency)
    nba_out = jnp.asarray(nba_mij)
    defor11_arr = jnp.asarray(defor11)
    defor12_arr = jnp.asarray(defor12)
    tke_arr = jnp.asarray(tke)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    xkmh_arr = jnp.asarray(xkmh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dnw_arr = jnp.asarray(dnw)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    p_m11 = _nba_component_index(config_flags, "p_m11", P_M11, int(n_nba_mij))
    p_m12 = _nba_component_index(config_flags, "p_m12", P_M12 if int(n_nba_mij) > P_M12 else P_M11, int(n_nba_mij))

    tau11 = cal_titau_11_22_33_vectorized(
        config_flags,
        titau1_init,
        tke_arr,
        xkmh_arr,
        defor11_arr,
        nba_out[:, :, :, p_m11],
        rho_arr,
        1,
        0,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau1 = tau11.titau
    nba_out = nba_out.at[:, :, :, p_m11].set(tau11.mtau)
    tau12 = cal_titau_12_21_vectorized(
        config_flags,
        titau2_init,
        xkmh_arr,
        defor12_arr,
        nba_out[:, :, :, p_m12],
        rho_arr,
        0,
        0,
        0,
        1,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau2 = tau12.titau
    nba_out = nba_out.at[:, :, :, p_m12].set(tau12.mtau)

    i0, i1, j0, j1 = _u_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i_vals = _frange(i0, i1, int(ims) + 1, min(_max_f(tend, 0, ims), _max_f(rdzw_arr, 0, ims)))
    j_vals = _frange(j0, j1, int(jms), min(_max_f(tend, 2, jms), _max_f(rdzw_arr, 2, jms) - 1))
    k_vals = _frange(int(kts), ktf, int(kms), min(_max_f(tend, 1, kms), _max_f(rdzw_arr, 1, kms)))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return HorizontalDiffusionU2Result(tendency=tend, nba_mij=nba_out)

    i_tlb = int(its) - 1
    j_tlb = int(jts) - 1
    zero_titau_edge = jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=tend.dtype)
    if ktf >= int(kts) + 1:
        mid = np.arange(int(kts) + 1, ktf + 1, dtype=np.int32)
        tau1_mid = mul(
            0.5,
            add(
                mul(
                    fnm_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau1, i_vals - 1, mid, j_vals, i_tlb, int(kts), j_tlb),
                        _take_titau(titau1, i_vals, mid, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
                mul(
                    fnp_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau1, i_vals - 1, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                        _take_titau(titau1, i_vals, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
            ),
        )
        tau2_mid = mul(
            0.5,
            add(
                mul(
                    fnm_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau2, i_vals, mid, j_vals + 1, i_tlb, int(kts), j_tlb),
                        _take_titau(titau2, i_vals, mid, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
                mul(
                    fnp_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau2, i_vals, mid - 1, j_vals + 1, i_tlb, int(kts), j_tlb),
                        _take_titau(titau2, i_vals, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
            ),
        )
        titau1avg = jnp.concatenate([zero_titau_edge, tau1_mid, zero_titau_edge], axis=1)
        titau2avg = jnp.concatenate([zero_titau_edge, tau2_mid, zero_titau_edge], axis=1)
    else:
        titau1avg = jnp.concatenate([zero_titau_edge, zero_titau_edge], axis=1)
        titau2avg = titau1avg

    zx_at_u = mul(
        0.5,
        add(
            _take3(zx_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            _take3(zx_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
        ),
    )
    zy_sum = _take3(zy_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms)
    for zy_term in (
        _take3(zy_arr, i_vals, k_vals, j_vals, ims, kms, jms),
        _take3(zy_arr, i_vals - 1, k_vals, j_vals + 1, ims, kms, jms),
        _take3(zy_arr, i_vals, k_vals, j_vals + 1, ims, kms, jms),
        _take3(zy_arr, i_vals - 1, k_vals + 1, j_vals, ims, kms, jms),
        _take3(zy_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
        _take3(zy_arr, i_vals - 1, k_vals + 1, j_vals + 1, ims, kms, jms),
        _take3(zy_arr, i_vals, k_vals + 1, j_vals + 1, ims, kms, jms),
    ):
        zy_sum = add(zy_sum, zy_term)
    zy_at_u = mul(0.125, zy_sum)
    rdzw_u = _take3(rdzw_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    rdzw_um1 = _take3(rdzw_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms)
    tmpdz = div_real(add(div_real(1.0, rdzw_u), div_real(1.0, rdzw_um1)), 2.0)
    msfux_u = _take2(msfux_arr, i_vals, j_vals, ims, jms)[:, None, :]
    msfuy_u = _take2(msfuy_arr, i_vals, j_vals, ims, jms)[:, None, :]
    mrdx = mul(msfux_u, float(rdx))
    mrdy = mul(msfuy_u, float(rdy))
    term1 = mul(
        mrdx,
        sub(
            _take_titau(titau1, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
            _take_titau(titau1, i_vals - 1, k_vals, j_vals, i_tlb, int(kts), j_tlb),
        ),
    )
    term2 = mul(
        mrdy,
        sub(
            _take_titau(titau2, i_vals, k_vals, j_vals + 1, i_tlb, int(kts), j_tlb),
            _take_titau(titau2, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
        ),
    )
    term3 = div_real(
        mul(
            mul(msfux_u, zx_at_u),
            sub(titau1avg[:, 1 : int(k_vals.size) + 1, :], titau1avg[:, : int(k_vals.size), :]),
        ),
        tmpdz,
    )
    term4 = div_real(
        mul(
            mul(msfuy_u, zy_at_u),
            sub(titau2avg[:, 1 : int(k_vals.size) + 1, :], titau2avg[:, : int(k_vals.size), :]),
        ),
        tmpdz,
    )
    bracket = sub(sub(add(term1, term2), term3), term4)
    prefactor = div_real(mul(G, tmpdz), dnw_arr[k_vals - int(kms)][None, :, None])
    increment = mul(prefactor, bracket)
    updated = add(_take3(tend, i_vals, k_vals, j_vals, ims, kms, jms), increment)
    tend = _set3(tend, i_vals, k_vals, j_vals, updated, ims, kms, jms)
    return HorizontalDiffusionU2Result(tendency=tend, nba_mij=nba_out)


def horizontal_diffusion_v_2_vectorized(
    tendency: Any,
    config_flags: Any,
    defor12: Any,
    defor22: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msfvx: Any,
    msfvy: Any,
    xkmh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dnw: Any,
    zx: Any,
    zy: Any,
    rdzw: Any,
    rho: Any,
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
    rounding_token: Any | None = None,
) -> HorizontalDiffusionV2Result:
    """Vectorized WRF ``horizontal_diffusion_v_2``."""

    del div, kds, ime, jme, kme
    tend = jnp.asarray(tendency)
    nba_out = jnp.asarray(nba_mij)
    defor12_arr = jnp.asarray(defor12)
    defor22_arr = jnp.asarray(defor22)
    tke_arr = jnp.asarray(tke)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    xkmh_arr = jnp.asarray(xkmh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dnw_arr = jnp.asarray(dnw)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    p_m12 = _nba_component_index(config_flags, "p_m12", P_M12 if int(n_nba_mij) > P_M12 else P_M11, int(n_nba_mij))
    p_m22 = _nba_component_index(config_flags, "p_m22", P_M22 if int(n_nba_mij) > P_M22 else P_M11, int(n_nba_mij))

    tau12 = cal_titau_12_21_vectorized(
        config_flags,
        titau1_init,
        xkmh_arr,
        defor12_arr,
        nba_out[:, :, :, p_m12],
        rho_arr,
        0,
        1,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau1 = tau12.titau
    nba_out = nba_out.at[:, :, :, p_m12].set(tau12.mtau)
    tau22 = cal_titau_11_22_33_vectorized(
        config_flags,
        titau2_init,
        tke_arr,
        xkmh_arr,
        defor22_arr,
        nba_out[:, :, :, p_m22],
        rho_arr,
        0,
        0,
        1,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau2 = tau22.titau
    nba_out = nba_out.at[:, :, :, p_m22].set(tau22.mtau)

    i0, i1, j0, j1 = _v_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i_vals = _frange(i0, i1, int(ims), min(_max_f(tend, 0, ims), _max_f(rdzw_arr, 0, ims) - 1))
    j_vals = _frange(j0, j1, int(jms) + 1, min(_max_f(tend, 2, jms), _max_f(rdzw_arr, 2, jms)))
    k_vals = _frange(int(kts), ktf, int(kms), min(_max_f(tend, 1, kms), _max_f(rdzw_arr, 1, kms)))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return HorizontalDiffusionV2Result(tendency=tend, nba_mij=nba_out)

    i_tlb = int(its) - 1
    j_tlb = int(jts) - 1
    zero_titau_edge = jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=tend.dtype)
    if ktf >= int(kts) + 1:
        mid = np.arange(int(kts) + 1, ktf + 1, dtype=np.int32)
        tau1_mid = mul(
            0.5,
            add(
                mul(
                    fnm_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau1, i_vals + 1, mid, j_vals, i_tlb, int(kts), j_tlb),
                        _take_titau(titau1, i_vals, mid, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
                mul(
                    fnp_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau1, i_vals + 1, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                        _take_titau(titau1, i_vals, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
            ),
        )
        tau2_mid = mul(
            0.5,
            add(
                mul(
                    fnm_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau2, i_vals, mid, j_vals - 1, i_tlb, int(kts), j_tlb),
                        _take_titau(titau2, i_vals, mid, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
                mul(
                    fnp_arr[mid - int(kms)][None, :, None],
                    add(
                        _take_titau(titau2, i_vals, mid - 1, j_vals - 1, i_tlb, int(kts), j_tlb),
                        _take_titau(titau2, i_vals, mid - 1, j_vals, i_tlb, int(kts), j_tlb),
                    ),
                ),
            ),
        )
        titau1avg = jnp.concatenate([zero_titau_edge, tau1_mid, zero_titau_edge], axis=1)
        titau2avg = jnp.concatenate([zero_titau_edge, tau2_mid, zero_titau_edge], axis=1)
    else:
        titau1avg = jnp.concatenate([zero_titau_edge, zero_titau_edge], axis=1)
        titau2avg = titau1avg

    zx_sum = _take3(zx_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    for zx_term in (
        _take3(zx_arr, i_vals + 1, k_vals, j_vals, ims, kms, jms),
        _take3(zx_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
        _take3(zx_arr, i_vals + 1, k_vals, j_vals - 1, ims, kms, jms),
        _take3(zx_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
        _take3(zx_arr, i_vals + 1, k_vals + 1, j_vals, ims, kms, jms),
        _take3(zx_arr, i_vals, k_vals + 1, j_vals - 1, ims, kms, jms),
        _take3(zx_arr, i_vals + 1, k_vals + 1, j_vals - 1, ims, kms, jms),
    ):
        zx_sum = add(zx_sum, zx_term)
    zx_at_v = mul(0.125, zx_sum)
    zy_at_v = mul(
        0.5,
        add(
            _take3(zy_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            _take3(zy_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
        ),
    )
    rdzw_v = _take3(rdzw_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    rdzw_vm1 = _take3(rdzw_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms)
    tmpdz = div_real(add(div_real(1.0, rdzw_v), div_real(1.0, rdzw_vm1)), 2.0)
    msfvx_v = _take2(msfvx_arr, i_vals, j_vals, ims, jms)[:, None, :]
    msfvy_v = _take2(msfvy_arr, i_vals, j_vals, ims, jms)[:, None, :]
    mrdy = mul(msfvy_v, float(rdy))
    mrdx = mul(msfvx_v, float(rdx))
    term1 = mul(
        mrdy,
        sub(
            _take_titau(titau2, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
            _take_titau(titau2, i_vals, k_vals, j_vals - 1, i_tlb, int(kts), j_tlb),
        ),
    )
    term2 = mul(
        mrdx,
        sub(
            _take_titau(titau1, i_vals + 1, k_vals, j_vals, i_tlb, int(kts), j_tlb),
            _take_titau(titau1, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
        ),
    )
    term3 = div_real(
        mul(
            mul(msfvx_v, zx_at_v),
            sub(titau1avg[:, 1 : int(k_vals.size) + 1, :], titau1avg[:, : int(k_vals.size), :]),
        ),
        tmpdz,
    )
    term4 = div_real(
        mul(
            mul(msfvy_v, zy_at_v),
            sub(titau2avg[:, 1 : int(k_vals.size) + 1, :], titau2avg[:, : int(k_vals.size), :]),
        ),
        tmpdz,
    )
    bracket = sub(sub(add(term1, term2), term3), term4)
    prefactor = div_real(mul(G, tmpdz), dnw_arr[k_vals - int(kms)][None, :, None])
    increment = mul(prefactor, bracket)
    updated = add(_take3(tend, i_vals, k_vals, j_vals, ims, kms, jms), increment)
    tend = _set3(tend, i_vals, k_vals, j_vals, updated, ims, kms, jms)
    return HorizontalDiffusionV2Result(tendency=tend, nba_mij=nba_out)


def horizontal_diffusion_w_2_vectorized(
    tendency: Any,
    config_flags: Any,
    defor13: Any,
    defor23: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msftx: Any,
    msfty: Any,
    xkmv: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dn: Any,
    zx: Any,
    zy: Any,
    rdz: Any,
    rho: Any,
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
    rounding_token: Any | None = None,
) -> HorizontalDiffusionW2Result:
    """Vectorized WRF ``horizontal_diffusion_w_2``."""

    del div, tke, kds, ime, jme, kme
    tend = jnp.asarray(tendency)
    nba_out = jnp.asarray(nba_mij)
    defor13_arr = jnp.asarray(defor13)
    defor23_arr = jnp.asarray(defor23)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    xkmv_arr = jnp.asarray(xkmv)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dn_arr = jnp.asarray(dn)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdz_arr = jnp.asarray(rdz)
    rho_arr = jnp.asarray(rho)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        div_real = lambda left, right: jnp.asarray(left) / jnp.asarray(right)
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div_real = lambda left, right: _real_div(left, right, rounding_token)

    ktf = min(int(kte), int(kde) - 1)
    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tend.dtype)
    p_m13 = _nba_component_index(config_flags, "p_m13", P_M13 if int(n_nba_mij) > P_M13 else P_M11, int(n_nba_mij))
    p_m23 = _nba_component_index(config_flags, "p_m23", P_M23 if int(n_nba_mij) > P_M23 else P_M11, int(n_nba_mij))

    tau13 = cal_titau_13_31_vectorized(
        config_flags,
        titau1_init,
        defor13_arr,
        nba_out[:, :, :, p_m13],
        xkmv_arr,
        fnm_arr,
        fnp_arr,
        rho_arr,
        0,
        1,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau1 = tau13.titau
    nba_out = nba_out.at[:, :, :, p_m13].set(tau13.mtau)
    tau23 = cal_titau_23_32_vectorized(
        config_flags,
        titau2_init,
        defor23_arr,
        nba_out[:, :, :, p_m23],
        xkmv_arr,
        fnm_arr,
        fnp_arr,
        rho_arr,
        0,
        0,
        0,
        1,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    titau2 = tau23.titau
    nba_out = nba_out.at[:, :, :, p_m23].set(tau23.mtau)

    i0, i1, j0, j1 = _configured_mass_bounds(config_flags, ids, ide, jds, jde, its, ite, jts, jte)
    i_vals = _frange(i0, i1, int(ims), min(_max_f(tend, 0, ims), _max_f(rdz_arr, 0, ims) - 1))
    j_vals = _frange(j0, j1, int(jms), min(_max_f(tend, 2, jms), _max_f(rdz_arr, 2, jms) - 1))
    k_vals = _frange(int(kts) + 1, ktf, int(kms) + 1, min(_max_f(tend, 1, kms), _max_f(rdz_arr, 1, kms)))
    if int(i_vals.size) == 0 or int(j_vals.size) == 0 or int(k_vals.size) == 0:
        return HorizontalDiffusionW2Result(tendency=tend, nba_mij=nba_out)

    i_tlb = int(its) - 1
    j_tlb = int(jts) - 1
    k_avg = np.arange(int(kts), ktf + 1, dtype=np.int32)
    titau1avg_sum = add(
        _take_titau(titau1, i_vals + 1, k_avg + 1, j_vals, i_tlb, int(kts), j_tlb),
        _take_titau(titau1, i_vals, k_avg + 1, j_vals, i_tlb, int(kts), j_tlb),
    )
    titau1avg_sum = add(
        titau1avg_sum,
        _take_titau(titau1, i_vals + 1, k_avg, j_vals, i_tlb, int(kts), j_tlb),
    )
    titau1avg_sum = add(
        titau1avg_sum,
        _take_titau(titau1, i_vals, k_avg, j_vals, i_tlb, int(kts), j_tlb),
    )
    titau1avg = mul(0.25, titau1avg_sum)
    titau2avg_sum = add(
        _take_titau(titau2, i_vals, k_avg + 1, j_vals + 1, i_tlb, int(kts), j_tlb),
        _take_titau(titau2, i_vals, k_avg + 1, j_vals, i_tlb, int(kts), j_tlb),
    )
    titau2avg_sum = add(
        titau2avg_sum,
        _take_titau(titau2, i_vals, k_avg, j_vals + 1, i_tlb, int(kts), j_tlb),
    )
    titau2avg_sum = add(
        titau2avg_sum,
        _take_titau(titau2, i_vals, k_avg, j_vals, i_tlb, int(kts), j_tlb),
    )
    titau2avg = mul(0.25, titau2avg_sum)
    zx_at_w = mul(
        0.5,
        add(
            _take3(zx_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            _take3(zx_arr, i_vals + 1, k_vals, j_vals, ims, kms, jms),
        ),
    )
    zy_at_w = mul(
        0.5,
        add(
            _take3(zy_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            _take3(zy_arr, i_vals, k_vals, j_vals + 1, ims, kms, jms),
        ),
    )
    rdz_w = _take3(rdz_arr, i_vals, k_vals, j_vals, ims, kms, jms)
    msftx_w = _take2(msftx_arr, i_vals, j_vals, ims, jms)[:, None, :]
    msfty_w = _take2(msfty_arr, i_vals, j_vals, ims, jms)[:, None, :]
    k_pos = k_vals - int(kts)
    if rounding_token is None:
        increment = G / (dn_arr[k_vals - int(kms)][None, :, None] * rdz_w) * (
            msftx_w * float(rdx)
            * (_take_titau(titau1, i_vals + 1, k_vals, j_vals, i_tlb, int(kts), j_tlb) - _take_titau(titau1, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb))
            + msfty_w * float(rdy)
            * (_take_titau(titau2, i_vals, k_vals, j_vals + 1, i_tlb, int(kts), j_tlb) - _take_titau(titau2, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb))
            - msfty_w
            * rdz_w
            * (zx_at_w * (titau1avg[:, k_pos, :] - titau1avg[:, k_pos - 1, :]) + zy_at_w * (titau2avg[:, k_pos, :] - titau2avg[:, k_pos - 1, :]))
        )
        updated = _take3(tend, i_vals, k_vals, j_vals, ims, kms, jms) + increment
    else:
        mrdx = mul(msftx_w, rdx)
        mrdy = mul(msfty_w, rdy)
        term1 = mul(
            mrdx,
            sub(
                _take_titau(titau1, i_vals + 1, k_vals, j_vals, i_tlb, int(kts), j_tlb),
                _take_titau(titau1, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
            ),
        )
        term2 = mul(
            mrdy,
            sub(
                _take_titau(titau2, i_vals, k_vals, j_vals + 1, i_tlb, int(kts), j_tlb),
                _take_titau(titau2, i_vals, k_vals, j_vals, i_tlb, int(kts), j_tlb),
            ),
        )
        vertical = add(
            mul(
                zx_at_w,
                sub(titau1avg[:, k_pos, :], titau1avg[:, k_pos - 1, :]),
            ),
            mul(
                zy_at_w,
                sub(titau2avg[:, k_pos, :], titau2avg[:, k_pos - 1, :]),
            ),
        )
        term3 = mul(mul(msfty_w, rdz_w), vertical)
        bracket = sub(add(term1, term2), term3)
        denominator = mul(dn_arr[k_vals - int(kms)][None, :, None], rdz_w)
        increment = mul(div_real(G, denominator), bracket)
        updated = add(_take3(tend, i_vals, k_vals, j_vals, ims, kms, jms), increment)
    tend = _set3(tend, i_vals, k_vals, j_vals, updated, ims, kms, jms)
    return HorizontalDiffusionW2Result(tendency=tend, nba_mij=nba_out)


def horizontal_diffusion_2_vectorized(
    rt_tendf: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    rw_tendf: Any,
    tke_tendf: Any,
    moist_tendf: Any,
    n_moist: int,
    chem_tendf: Any,
    n_chem: int,
    scalar_tendf: Any,
    n_scalar: int,
    tracer_tendf: Any,
    n_tracer: int,
    thp: Any,
    theta: Any,
    tke: Any,
    config_flags: Any,
    defor11: Any,
    defor22: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    nba_mij: Any,
    n_nba_mij: int,
    div: Any,
    moist: Any,
    chem: Any,
    scalar: Any,
    tracer: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    km_opt: int,
    rdx: float,
    rdy: float,
    rdz: Any,
    rdzw: Any,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
    dn: Any,
    dnw: Any,
    rho: Any,
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
    rounding_token: Any | None = None,
    *,
    _diffuse_theta: bool = True,
) -> HorizontalDiffusion2Result:
    """Vectorized WRF ``horizontal_diffusion_2`` driver."""

    del theta
    u_result = horizontal_diffusion_u_2_vectorized(
        ru_tendf,
        config_flags,
        defor11,
        defor12,
        div,
        nba_mij,
        n_nba_mij,
        tke,
        msfux,
        msfuy,
        xkmh,
        rdx,
        rdy,
        fnm,
        fnp,
        dnw,
        zx,
        zy,
        rdzw,
        rho,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    v_result = horizontal_diffusion_v_2_vectorized(
        rv_tendf,
        config_flags,
        defor12,
        defor22,
        div,
        u_result.nba_mij,
        n_nba_mij,
        tke,
        msfvx,
        msfvy,
        xkmh,
        rdx,
        rdy,
        fnm,
        fnp,
        dnw,
        zx,
        zy,
        rdzw,
        rho,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    w_result = horizontal_diffusion_w_2_vectorized(
        rw_tendf,
        config_flags,
        defor13,
        defor23,
        div,
        v_result.nba_mij,
        n_nba_mij,
        tke,
        msftx,
        msfty,
        xkmv,
        rdx,
        rdy,
        fnm,
        fnp,
        dn,
        zx,
        zy,
        rdz,
        rho,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    if _diffuse_theta:
        rt_out = horizontal_diffusion_s_vectorized(
            rt_tendf,
            config_flags,
            thp,
            msftx,
            msfty,
            msfux,
            msfuy,
            msfvx,
            msfvy,
            xkhh,
            rdx,
            rdy,
            fnm,
            fnp,
            cf1,
            cf2,
            cf3,
            zx,
            zy,
            rdz,
            rdzw,
            dnw,
            dn,
            rho,
            False,
            ids,
            ide,
            jds,
            jde,
            kds,
            kde,
            ims,
            ime,
            jms,
            jme,
            kms,
            kme,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
            rounding_token,
        )
    else:
        rt_out = jnp.asarray(rt_tendf)
    if ((int(km_opt) == 2 and not _flag(config_flags, "tke_mix2_off")) or int(km_opt) == 5):
        tke_out = horizontal_diffusion_s_vectorized(
            tke_tendf,
            config_flags,
            tke,
            msftx,
            msfty,
            msfux,
            msfuy,
            msfvx,
            msfvy,
            xkhh,
            rdx,
            rdy,
            fnm,
            fnp,
            cf1,
            cf2,
            cf3,
            zx,
            zy,
            rdz,
            rdzw,
            dnw,
            dn,
            rho,
            True,
            ids,
            ide,
            jds,
            jde,
            kds,
            kde,
            ims,
            ime,
            jms,
            jme,
            kms,
            kme,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
            rounding_token,
        )
    else:
        tke_out = jnp.asarray(tke_tendf)

    def diffuse_tracers(tend: Any, count: int, values: Any, disabled_flag: str) -> Any:
        if int(count) < PARAM_FIRST_SCALAR or _flag(config_flags, disabled_flag):
            return jnp.asarray(tend)
        out = jnp.asarray(tend)
        vals = jnp.asarray(values)

        def diffuse_one(tendency: Any, field: Any) -> Any:
            return horizontal_diffusion_s_vectorized(
                tendency,
                config_flags,
                field,
                msftx,
                msfty,
                msfux,
                msfuy,
                msfvx,
                msfvy,
                xkhh,
                rdx,
                rdy,
                fnm,
                fnp,
                cf1,
                cf2,
                cf3,
                zx,
                zy,
                rdz,
                rdzw,
                dnw,
                dn,
                rho,
                False,
                ids,
                ide,
                jds,
                jde,
                kds,
                kde,
                ims,
                ime,
                jms,
                jme,
                kms,
                kme,
                its,
                ite,
                jts,
                jte,
                kts,
                kte,
                rounding_token,
            )

        first, stop = PARAM_FIRST_SCALAR - 1, int(count)
        updated = jax.vmap(diffuse_one, in_axes=(-1, -1), out_axes=-1)(
            out[..., first:stop], vals[..., first:stop],
        )
        return jnp.concatenate((out[..., :first], updated, out[..., stop:]), axis=-1)

    return HorizontalDiffusion2Result(
        rt_tendf=rt_out,
        ru_tendf=u_result.tendency,
        rv_tendf=v_result.tendency,
        rw_tendf=w_result.tendency,
        tke_tendf=tke_out,
        moist_tendf=diffuse_tracers(moist_tendf, n_moist, moist, "moist_mix2_off"),
        chem_tendf=diffuse_tracers(chem_tendf, n_chem, chem, "chem_mix2_off"),
        scalar_tendf=diffuse_tracers(scalar_tendf, n_scalar, scalar, "scalar_mix2_off"),
        tracer_tendf=diffuse_tracers(tracer_tendf, n_tracer, tracer, "tracer_mix2_off"),
        nba_mij=w_result.nba_mij,
    )


def horizontal_diffusion_2_without_theta_vectorized(
    *args: Any,
    **kwargs: Any,
) -> HorizontalDiffusion2Result:
    """Run the coupled diffusion driver while leaving theta tendency unchanged."""

    return horizontal_diffusion_2_vectorized(
        *args,
        **kwargs,
        _diffuse_theta=False,
    )


__all__ = [
    "cal_deform_and_div_vectorized",
    "calculate_km_kh_vectorized",
    "cal_titau_11_22_33_vectorized",
    "cal_titau_12_21_vectorized",
    "cal_titau_13_31_vectorized",
    "cal_titau_23_32_vectorized",
    "horizontal_diffusion_s_vectorized",
    "horizontal_diffusion_u_2_vectorized",
    "horizontal_diffusion_v_2_vectorized",
    "horizontal_diffusion_w_2_vectorized",
    "horizontal_diffusion_2_vectorized",
    "horizontal_diffusion_2_without_theta_vectorized",
]
