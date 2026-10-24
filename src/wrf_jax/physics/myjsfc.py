"""Translations from ``module_sf_myjsfc``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.optimization import optimization_barrier
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.real_arithmetic import column_barrier, column_binary
from wrf_jax.core.wrf_math import (
    glibc_expf_tables, scalar_powf_tables,
    sqrt_for_wrf, scalar_powf_for_wrf, wrf_glibc_expf_finite,
)


ITRMX = 5
KZTM = 10001
KZTM2 = KZTM - 2
_TABLE_PATH = Path(__file__).with_name("data") / "myjsfc_similarity_tables.npz"

_REAL = np.float32

G = _REAL(9.81)
R_D = _REAL(287.0)
CP = _REAL(_REAL(7.0) * R_D / _REAL(2.0))
R_V = _REAL(461.6)
XLV = _REAL(2.5e6)
_RVOVRD = _REAL(R_V / R_D)
P608 = _REAL(_RVOVRD - _REAL(1.0))
PQ0 = _REAL(379.90516)
EPSQ2 = _REAL(0.2)
A2S = _REAL(17.2693882)
A3S = _REAL(273.16)
A4S = _REAL(35.86)
VKARMAN = _REAL(0.4)
ELOCP = _REAL(_REAL(2.72e6) / CP)
EPSU2 = _REAL(1.0e-6)
EPSUST = _REAL(1.0e-9)
EPSZT = _REAL(1.0e-28)
SEAFC = _REAL(0.98)
PQ0SEA = _REAL(PQ0 * SEAFC)
EXCML = _REAL(0.0001)
EXCMS = _REAL(0.0001)
GLKBR = _REAL(10.0)
GLKBS = _REAL(30.0)
QVISC = _REAL(2.1e-5)
RIC = _REAL(0.505)
SMALL = _REAL(0.35)
SQPR = _REAL(0.84)
SQSC = _REAL(0.84)
SQVISC = _REAL(258.2)
TVISC = _REAL(2.1e-5)
USTC = _REAL(0.7)
USTR = _REAL(0.225)
VISC = _REAL(1.5e-5)
FH = _REAL(1.01)
WWST = _REAL(1.2)
ZTFC = _REAL(1.0)
CZIV = _REAL(SMALL * GLKBS)
GRRS = _REAL(GLKBR / GLKBS)
RTVISC = _REAL(_REAL(1.0) / TVISC)
RVISC = _REAL(_REAL(1.0) / VISC)
ZQRZT = _REAL(SQSC / SQPR)
FZQ1 = _REAL(RTVISC * QVISC * ZQRZT)
FZQ2 = _REAL(RTVISC * QVISC * ZQRZT)
FZT1 = _REAL(RVISC * TVISC * SQPR)
FZT2 = _REAL(CZIV * GRRS * TVISC * SQPR)
FZU1 = _REAL(CZIV * VISC)
RQVISC = _REAL(_REAL(1.0) / QVISC)
USTFC = _REAL(_REAL(0.018) / G)
WWST2 = _REAL(WWST * WWST)


@dataclass(frozen=True)
class MyjSfcTables:
    dzeta1: np.float32
    dzeta2: np.float32
    fh01: np.float32
    fh02: np.float32
    ztmax1: np.float32
    ztmax2: np.float32
    ztmin1: np.float32
    ztmin2: np.float32
    psim1: np.ndarray
    psih1: np.ndarray
    psim2: np.ndarray
    psih2: np.ndarray


@dataclass(frozen=True)
class MyjSfcInitResult:
    lowlyr: Any
    ustar: Any
    z0: Any
    tables: MyjSfcTables


@dataclass(frozen=True)
class SfcdifResult:
    chs: Any
    chs2: Any
    cpm: Any
    cqs2: Any
    ct: Any
    flhc: Any
    flqc: Any
    flx_lh: Any
    hfx: Any
    pshltr: Any
    q02: Any
    q10: Any
    qfx: Any
    qgh: Any
    rib: Any
    rlmo: Any
    th02: Any
    th10: Any
    u10: Any
    v10: Any
    u10e: Any
    v10e: Any
    akhs: Any
    akms: Any
    qz0: Any
    thz0: Any
    ustar: Any
    uz0: Any
    vz0: Any
    z0: Any
    qs: Any


@dataclass(frozen=True)
class MyjSfcResult:
    qsfc: Any
    thz0: Any
    qz0: Any
    uz0: Any
    vz0: Any
    ustar: Any
    znt: Any
    pblh: Any
    rmol: Any
    akhs: Any
    akms: Any
    rib: Any
    chs: Any
    chs2: Any
    cqs2: Any
    hfx: Any
    qfx: Any
    flx_lh: Any
    flhc: Any
    flqc: Any
    qgh: Any
    cpm: Any
    ct: Any
    u10: Any
    v10: Any
    t02: Any
    th02: Any
    tshltr: Any
    th10: Any
    q02: Any
    qshltr: Any
    q10: Any
    pshltr: Any
    u10e: Any
    v10e: Any


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


@lru_cache(maxsize=1)
def _build_tables() -> MyjSfcTables:
    scalar_names = (
        "dzeta1",
        "dzeta2",
        "fh01",
        "fh02",
        "ztmax1",
        "ztmax2",
        "ztmin1",
        "ztmin2",
    )
    table_names = ("psim1", "psih1", "psim2", "psih2")
    with np.load(_TABLE_PATH, allow_pickle=False) as archive:
        missing = sorted(set((*scalar_names, *table_names)) - set(archive.files))
        if missing:
            raise ValueError(f"MYJSFC table asset is missing: {', '.join(missing)}")
        scalars = {
            name: np.asarray(archive[name], dtype=np.float32).reshape(())[()]
            for name in scalar_names
        }
        tables = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in table_names
        }
    for name, values in tables.items():
        if values.shape != (KZTM,):
            raise ValueError(f"MYJSFC table {name} has shape {values.shape}, expected {(KZTM,)}")
    return MyjSfcTables(**scalars, **tables)


def _wrf_real_statement(value: Any, dtype: Any) -> Any:
    """Materialize one WRF ``REAL`` statement result inside compiled graphs."""

    return column_barrier(jnp.asarray(value, dtype=dtype))


@jax.custom_jvp
def _materialize_real_operation(value: Any) -> Any:
    """Expose a REAL operation result so XLA cannot contract across it."""

    return jnp.nextafter(value, value)


@_materialize_real_operation.defjvp
def _materialize_real_operation_jvp(
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Any, Any]:
    return _materialize_real_operation(primals[0]), tangents[0]


def _wrf_add(left: Any, right: Any, dtype: Any) -> Any:
    return _wrf_real_statement(column_binary("add", jnp.asarray(left, dtype=dtype), jnp.asarray(right, dtype=dtype)), dtype)


def _wrf_sub(left: Any, right: Any, dtype: Any) -> Any:
    difference = column_binary("sub", jnp.asarray(left, dtype=dtype), jnp.asarray(right, dtype=dtype))
    return _materialize_real_operation(difference)


def _wrf_mul(left: Any, right: Any, dtype: Any) -> Any:
    product = column_binary("mul", jnp.asarray(left, dtype=dtype), jnp.asarray(right, dtype=dtype))
    return _materialize_real_operation(product)


def _wrf_div(left: Any, right: Any, dtype: Any) -> Any:
    numerator = jnp.asarray(left, dtype=dtype)
    denominator = column_barrier(jnp.asarray(right, dtype=dtype))
    return _wrf_real_statement(column_binary("div", numerator, denominator), dtype)


def _wrf_sqrt(value: Any, dtype: Any) -> Any:
    return _wrf_real_statement(sqrt_for_wrf(jnp.asarray(value, dtype=dtype)), dtype)


def _wrf_log(value: Any, dtype: Any) -> Any:
    value_real = jnp.asarray(value, dtype=dtype)
    if jnp.dtype(dtype) == jnp.dtype(jnp.float32) and jax.config.x64_enabled:
        value_real = value_real.astype(jnp.float64)
    return _wrf_real_statement(jnp.log(value_real), dtype)


def _wrf_exp(value: Any, dtype: Any, *, tables: Any = None) -> Any:
    value_real = jnp.asarray(value, dtype=dtype)
    if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
        return _wrf_real_statement(wrf_glibc_expf_finite(value_real, tables), dtype)
    return _wrf_real_statement(jnp.exp(value_real), dtype)


def _wrf_power(value: Any, exponent: Any, dtype: Any, *, tables: Any = None) -> Any:
    if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
        return _wrf_real_statement(scalar_powf_for_wrf(value, exponent, tables), dtype)
    return _wrf_real_statement(
        jnp.asarray(value, dtype=dtype) ** jnp.asarray(exponent, dtype=dtype),
        dtype,
    )


def _interp_similarity(values: Any, zeta: Any, ztmin: Any, dzeta: Any) -> Any:
    dtype = jnp.asarray(zeta).dtype
    table = values
    rz = _wrf_div(_wrf_sub(zeta, ztmin, dtype), dzeta, dtype)
    k_unclipped = rz.astype(jnp.int32)
    rdzt = _wrf_sub(rz, k_unclipped.astype(dtype), dtype)
    k = jnp.clip(k_unclipped, 0, KZTM2)
    delta = _wrf_sub(table[k + 1], table[k], dtype)
    return _wrf_add(_wrf_mul(delta, rdzt, dtype), table[k], dtype)


def _dthv(thelow: Any, thz0: Any, qlow: Any, qz0: Any, cwmlow: Any, t_low: Any, tz0: Any) -> Any:
    dtype = jnp.result_type(thelow, thz0, qlow, qz0, cwmlow, t_low, tz0, jnp.float32)
    tem = _wrf_mul(_wrf_add(t_low, tz0, dtype), 0.5, dtype)
    thm = _wrf_mul(_wrf_add(thelow, thz0, dtype), 0.5, dtype)
    a = _wrf_mul(thm, P608, dtype)
    b = _wrf_mul(_wrf_sub(_wrf_sub(_wrf_div(ELOCP, tem, dtype), 1.0, dtype), P608, dtype), thm, dtype)
    moisture = _wrf_add(_wrf_add(qlow, qz0, dtype), cwmlow, dtype)
    moisture_factor = _wrf_add(_wrf_mul(moisture, _wrf_mul(0.5, P608, dtype), dtype), 1.0, dtype)
    first = _wrf_mul(_wrf_sub(thelow, thz0, dtype), moisture_factor, dtype)
    second_moisture = _wrf_add(_wrf_sub(qlow, qz0, dtype), cwmlow, dtype)
    second = _wrf_mul(second_moisture, a, dtype)
    third = _wrf_mul(cwmlow, b, dtype)
    return _wrf_add(_wrf_add(first, second, dtype), third, dtype)


def _surface_similarity(
    psim: Any,
    psih: Any,
    ztmin: Any,
    ztmax: Any,
    dzeta: Any,
    fh0: Any,
    cxch: Any,
    zu: Any,
    zt: Any,
    rlmo: Any,
    psmz: Any,
    pshz: Any,
    ustark: Any,
) -> tuple[Any, Any, Any]:
    zu10 = zu + 10.0
    zt02 = zt + 2.0
    zt10 = zt + 10.0
    rlnu10 = jnp.log(column_binary('div', zu10, zu))
    rlnt02 = jnp.log(column_binary('div', zt02, zt))
    rlnt10 = jnp.log(column_binary('div', zt10, zt))

    ztau10 = jnp.clip(zu10 * rlmo, ztmin, ztmax)
    ztat02 = jnp.clip(zt02 * rlmo, ztmin, ztmax)
    ztat10 = jnp.clip(zt10 * rlmo, ztmin, ztmax)

    psm10 = _interp_similarity(psim, ztau10, ztmin, dzeta)
    simm10 = psm10 - psmz + rlnu10
    psh02 = _interp_similarity(psih, ztat02, ztmin, dzeta)
    simh02 = (psh02 - pshz + rlnt02) * fh0
    psh10 = _interp_similarity(psih, ztat10, ztmin, dzeta)
    simh10 = (psh10 - pshz + rlnt10) * fh0

    akms10 = jnp.maximum(column_binary('div', ustark, simm10), cxch)
    akhs02 = jnp.maximum(column_binary('div', ustark, simh02), cxch)
    akhs10 = jnp.maximum(column_binary('div', ustark, simh10), cxch)
    return akms10, akhs02, akhs10


def sfcdif(
    ntsd: int,
    seamask: Any,
    ths: Any,
    qs: Any,
    psfc: Any,
    uz0: Any,
    vz0: Any,
    tz0: Any,
    tsk: Any,
    thz0: Any,
    qz0: Any,
    ustar: Any,
    z0: Any,
    z0base: Any,
    akms: Any,
    akhs: Any,
    pblh: Any,
    wetm: Any,
    vegtyp: int,
    isurban: int,
    iz0tlnd: int,
    ulow: Any,
    vlow: Any,
    tlow: Any,
    thlow: Any,
    thelow: Any,
    qlow: Any,
    cwmlow: Any,
    zsl: Any,
    plow: Any,
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
    i: int,
    j: int,
    zsfc: Any,
    tables: MyjSfcTables | None = None,
) -> SfcdifResult:
    """Return WRF ``SFCDIF`` scalar surface-layer outputs for one grid point."""

    del vegtyp, isurban, iz0tlnd, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
    del its, ite, jts, jte, kts, kte, i, j, zsfc

    tables = _build_tables() if tables is None else tables
    inputs = tuple(jnp.asarray(value) for value in (
        ntsd, seamask, ths, qs, psfc, uz0, vz0, tz0, tsk, thz0, qz0,
        ustar, z0, z0base, akms, akhs, pblh, wetm, ulow, vlow, tlow,
        thlow, thelow, qlow, cwmlow, zsl, plow,
    ))
    dtype = jnp.result_type(*inputs[1:], jnp.float32)
    parameters = {
        "similarity": {name: jnp.asarray(value, dtype) for name, value in vars(tables).items()},
        "exponential": glibc_expf_tables(),
        "power": scalar_powf_tables(),
    }
    outputs = map_elements(_sfcdif_elements, inputs, parameters)
    return SfcdifResult(ct=jnp.asarray(0.0, dtype), **outputs)


def _sfcdif_elements(inputs: tuple[Any, ...], parameters: dict[str, Any]) -> dict[str, Any]:
    """The same ordered surface science for ordinary and resident execution."""
    (
        ntsd, seamask, ths, qs, psfc, uz0, vz0, tz0, tsk, thz0, qz0,
        ustar, z0, z0base, akms, akhs, pblh, wetm, ulow, vlow, tlow,
        thlow, thelow, qlow, cwmlow, zsl, plow,
    ) = inputs
    tables = MyjSfcTables(**parameters["similarity"])
    dtype = jnp.result_type(
        seamask,
        ths,
        qs,
        psfc,
        uz0,
        vz0,
        tz0,
        tsk,
        thz0,
        qz0,
        ustar,
        z0,
        z0base,
        akms,
        akhs,
        pblh,
        wetm,
        ulow,
        vlow,
        tlow,
        thlow,
        thelow,
        qlow,
        cwmlow,
        zsl,
        plow,
        jnp.float32,
    )
    seamask = jnp.asarray(seamask, dtype=dtype)
    ths = jnp.asarray(ths, dtype=dtype)
    qs = jnp.asarray(qs, dtype=dtype)
    psfc = jnp.asarray(psfc, dtype=dtype)
    uz0 = jnp.asarray(uz0, dtype=dtype)
    vz0 = jnp.asarray(vz0, dtype=dtype)
    tz0 = jnp.asarray(tz0, dtype=dtype)
    tsk = jnp.asarray(tsk, dtype=dtype)
    thz0 = jnp.asarray(thz0, dtype=dtype)
    qz0 = jnp.asarray(qz0, dtype=dtype)
    ustar = jnp.asarray(ustar, dtype=dtype)
    z0 = jnp.asarray(z0, dtype=dtype)
    z0base = jnp.asarray(z0base, dtype=dtype)
    akms = jnp.asarray(akms, dtype=dtype)
    akhs = jnp.asarray(akhs, dtype=dtype)
    pblh = jnp.asarray(pblh, dtype=dtype)
    wetm = jnp.asarray(wetm, dtype=dtype)
    ulow = jnp.asarray(ulow, dtype=dtype)
    vlow = jnp.asarray(vlow, dtype=dtype)
    tlow = jnp.asarray(tlow, dtype=dtype)
    thlow = jnp.asarray(thlow, dtype=dtype)
    thelow = jnp.asarray(thelow, dtype=dtype)
    qlow = jnp.asarray(qlow, dtype=dtype)
    cwmlow = jnp.asarray(cwmlow, dtype=dtype)
    zsl = jnp.asarray(zsl, dtype=dtype)
    plow = jnp.asarray(plow, dtype=dtype)

    rd_z = _wrf_div(1.0, zsl, dtype)
    cxchl = _wrf_mul(EXCML, rd_z, dtype)
    cxchs = _wrf_mul(EXCMS, rd_z, dtype)
    btgx = _wrf_div(G, thlow, dtype)
    elfc = _wrf_mul(VKARMAN, btgx, dtype)
    btgh = _wrf_mul(
        btgx,
        jnp.where(pblh > 1000.0, pblh, jnp.asarray(1000.0, dtype=dtype)),
        dtype,
    )

    psim1 = tables.psim1
    psih1 = tables.psih1
    psim2 = tables.psim2
    psih2 = tables.psih2
    dzeta1 = jnp.asarray(tables.dzeta1, dtype=dtype)
    dzeta2 = jnp.asarray(tables.dzeta2, dtype=dtype)
    ztmin1 = jnp.asarray(tables.ztmin1, dtype=dtype)
    ztmin2 = jnp.asarray(tables.ztmin2, dtype=dtype)
    ztmax1 = jnp.asarray(tables.ztmax1, dtype=dtype)
    ztmax2 = jnp.asarray(tables.ztmax2, dtype=dtype)
    fh01 = jnp.asarray(tables.fh01, dtype=dtype)
    fh02 = jnp.asarray(tables.fh02, dtype=dtype)
    first_step = jnp.asarray(ntsd, dtype=jnp.int32) == 1

    def sea_branch() -> tuple[Any, ...]:
        qs_s = qs
        uz0_s = uz0
        vz0_s = vz0
        thz0_s = thz0
        qz0_s = qz0
        ustar_s = ustar
        z0_s = z0
        akms_s = akms
        akhs_s = akhs
        rib_s = jnp.asarray(0.0, dtype=dtype)
        rlmo_s = jnp.asarray(0.0, dtype=dtype)
        zu_s = _wrf_real_statement(jnp.maximum(z0_s, EPSZT), dtype)
        zt_s = zu_s
        psmz_s = jnp.asarray(0.0, dtype=dtype)
        pshz_s = jnp.asarray(0.0, dtype=dtype)
        ustark_s = _wrf_mul(ustar_s, VKARMAN, dtype)
        wstar2_s = jnp.asarray(0.0, dtype=dtype)

        for _ in range(ITRMX):
            z0_s = _wrf_real_statement(
                jnp.maximum(
                    _wrf_mul(_wrf_mul(USTFC, ustar_s, dtype), ustar_s, dtype),
                    jnp.asarray(1.59e-5, dtype=dtype),
                ),
                dtype,
            )

            akms_very = jnp.where(first_step, jnp.asarray(cxchs, dtype=dtype), akms_s)
            akhs_very = jnp.where(first_step, jnp.asarray(cxchs, dtype=dtype), akhs_s)
            qs_very = jnp.where(first_step, qlow, qs_s)
            viscous_root = _wrf_sqrt(
                _wrf_sqrt(
                    _wrf_mul(_wrf_mul(z0_s, ustar_s, dtype), RVISC, dtype),
                    dtype,
                ),
                dtype,
            )
            zu_very = _wrf_div(_wrf_mul(FZU1, viscous_root, dtype), ustar_s, dtype)
            wght = _wrf_mul(_wrf_mul(akms_very, zu_very, dtype), RVISC, dtype)
            rwgh = _wrf_div(wght, _wrf_add(wght, 1.0, dtype), dtype)
            uz0_very = _wrf_mul(
                _wrf_add(_wrf_mul(ulow, rwgh, dtype), uz0_s, dtype),
                0.5,
                dtype,
            )
            vz0_very = _wrf_mul(
                _wrf_add(_wrf_mul(vlow, rwgh, dtype), vz0_s, dtype),
                0.5,
                dtype,
            )
            zt_very = _wrf_mul(FZT1, zu_very, dtype)
            zq_very = _wrf_mul(FZQ1, zt_very, dtype)
            wghtt = _wrf_mul(_wrf_mul(akhs_very, zt_very, dtype), RTVISC, dtype)
            wghtq = _wrf_mul(_wrf_mul(akhs_very, zq_very, dtype), RQVISC, dtype)
            thz0_direct = _wrf_div(
                _wrf_add(_wrf_mul(wghtt, thlow, dtype), ths, dtype),
                _wrf_add(wghtt, 1.0, dtype),
                dtype,
            )
            qz0_direct = _wrf_div(
                _wrf_add(_wrf_mul(wghtq, qlow, dtype), qs_very, dtype),
                _wrf_add(wghtq, 1.0, dtype),
                dtype,
            )
            thz0_very = jnp.where(
                first_step,
                thz0_direct,
                _wrf_mul(_wrf_add(thz0_direct, thz0_s, dtype), 0.5, dtype),
            )
            qz0_very = jnp.where(
                first_step,
                qz0_direct,
                _wrf_mul(_wrf_add(qz0_direct, qz0_s, dtype), 0.5, dtype),
            )

            zu_mid = z0_s
            uz0_mid = jnp.asarray(0.0, dtype=dtype)
            vz0_mid = jnp.asarray(0.0, dtype=dtype)
            zt_mid = _wrf_div(_wrf_mul(FZT2, viscous_root, dtype), ustar_s, dtype)
            zq_mid = _wrf_mul(FZQ2, zt_mid, dtype)
            wghtt_mid = _wrf_mul(_wrf_mul(akhs_s, zt_mid, dtype), RTVISC, dtype)
            wghtq_mid = _wrf_mul(_wrf_mul(akhs_s, zq_mid, dtype), RQVISC, dtype)
            thz0_mid_direct = _wrf_div(
                _wrf_add(_wrf_mul(wghtt_mid, thlow, dtype), ths, dtype),
                _wrf_add(wghtt_mid, 1.0, dtype),
                dtype,
            )
            qz0_mid_direct = _wrf_div(
                _wrf_add(_wrf_mul(wghtq_mid, qlow, dtype), qs_s, dtype),
                _wrf_add(wghtq_mid, 1.0, dtype),
                dtype,
            )
            thz0_mid = jnp.where(
                first_step,
                thz0_mid_direct,
                _wrf_mul(_wrf_add(thz0_mid_direct, thz0_s, dtype), 0.5, dtype),
            )
            qz0_mid = jnp.where(
                first_step,
                qz0_mid_direct,
                _wrf_mul(_wrf_add(qz0_mid_direct, qz0_s, dtype), 0.5, dtype),
            )

            zu_high = z0_s
            zt_high = z0_s
            very = ustar_s < USTR
            mid = (ustar_s >= USTR) & (ustar_s < USTC)
            qs_s = jnp.where(very, qs_very, qs_s)
            akms_s = jnp.where(very, akms_very, akms_s)
            akhs_s = jnp.where(very, akhs_very, akhs_s)
            zu_s = jnp.where(very, zu_very, jnp.where(mid, zu_mid, zu_high))
            uz0_s = jnp.where(very, uz0_very, jnp.where(mid, uz0_mid, 0.0))
            vz0_s = jnp.where(very, vz0_very, jnp.where(mid, vz0_mid, 0.0))
            zt_s = jnp.where(very, zt_very, jnp.where(mid, zt_mid, zt_high))
            thz0_s = jnp.where(very, thz0_very, jnp.where(mid, thz0_mid, ths))
            qz0_s = jnp.where(very, qz0_very, jnp.where(mid, qz0_mid, qs_s))

            dthv = _dthv(thelow, thz0_s, qlow, qz0_s, cwmlow, tlow, tz0)
            du = _wrf_sub(ulow, uz0_s, dtype)
            dv = _wrf_sub(vlow, vz0_s, dtype)
            du2 = _wrf_real_statement(
                jnp.maximum(
                    _wrf_add(_wrf_mul(du, du, dtype), _wrf_mul(dv, dv, dtype), dtype),
                    jnp.asarray(EPSU2, dtype=dtype),
                ),
                dtype,
            )
            rib_s = _wrf_div(_wrf_mul(_wrf_mul(btgx, dthv, dtype), zsl, dtype), du2, dtype)
            zslu = _wrf_add(zsl, zu_s, dtype)
            zslt = _wrf_add(zsl, zt_s, dtype)
            rzsu = _wrf_div(zslu, zu_s, dtype)
            rzst = _wrf_div(zslt, zt_s, dtype)
            rlogu = _wrf_log(rzsu, dtype)
            rlogt = _wrf_log(rzst, dtype)
            ustar_squared = _wrf_mul(ustar_s, ustar_s, dtype)
            ustar_cubed = _wrf_mul(ustar_squared, ustar_s, dtype)
            rlmo_s = _wrf_div(_wrf_mul(_wrf_mul(elfc, akhs_s, dtype), dthv, dtype), ustar_cubed, dtype)

            zetalu = _wrf_real_statement(jnp.clip(_wrf_mul(zslu, rlmo_s, dtype), ztmin1, ztmax1), dtype)
            zetalt = _wrf_real_statement(jnp.clip(_wrf_mul(zslt, rlmo_s, dtype), ztmin1, ztmax1), dtype)
            zetau = _wrf_real_statement(
                jnp.clip(
                    _wrf_mul(zu_s, rlmo_s, dtype),
                    _wrf_div(ztmin1, rzsu, dtype),
                    _wrf_div(ztmax1, rzsu, dtype),
                ),
                dtype,
            )
            zetat = _wrf_real_statement(
                jnp.clip(
                    _wrf_mul(zt_s, rlmo_s, dtype),
                    _wrf_div(ztmin1, rzst, dtype),
                    _wrf_div(ztmax1, rzst, dtype),
                ),
                dtype,
            )
            psmz_s = _interp_similarity(psim1, zetau, ztmin1, dzeta1)
            psmzl = _interp_similarity(psim1, zetalu, ztmin1, dzeta1)
            simm = _wrf_add(_wrf_sub(psmzl, psmz_s, dtype), rlogu, dtype)
            pshz_s = _interp_similarity(psih1, zetat, ztmin1, dzeta1)
            pshzl = _interp_similarity(psih1, zetalt, ztmin1, dzeta1)
            simh = _wrf_mul(_wrf_add(_wrf_sub(pshzl, pshz_s, dtype), rlogt, dtype), fh01, dtype)

            ustark_s = _wrf_mul(ustar_s, VKARMAN, dtype)
            akms_s = _wrf_real_statement(jnp.maximum(_wrf_div(ustark_s, simm, dtype), cxchs), dtype)
            akhs_s = _wrf_real_statement(jnp.maximum(_wrf_div(ustark_s, simh, dtype), cxchs), dtype)
            buoyancy = _wrf_mul(_wrf_mul(btgh, akhs_s, dtype), dthv, dtype)
            convective = _wrf_mul(
                WWST2,
                _wrf_power(jnp.abs(buoyancy), _wrf_div(2.0, 3.0, dtype), dtype, tables=parameters['power']),
                dtype,
            )
            wstar2_s = _wrf_real_statement(jnp.where(dthv <= 0.0, convective, 0.0), dtype)
            ustar_s = _wrf_real_statement(
                jnp.maximum(
                    _wrf_sqrt(_wrf_mul(akms_s, _wrf_sqrt(_wrf_add(du2, wstar2_s, dtype), dtype), dtype), dtype),
                    jnp.asarray(EPSUST, dtype=dtype),
                ),
                dtype,
            )
            qs_s = _wrf_real_statement(qs_s, dtype)
            uz0_s = _wrf_real_statement(uz0_s, dtype)
            vz0_s = _wrf_real_statement(vz0_s, dtype)
            thz0_s = _wrf_real_statement(thz0_s, dtype)
            qz0_s = _wrf_real_statement(qz0_s, dtype)
            ustar_s = _wrf_real_statement(ustar_s, dtype)
            z0_s = _wrf_real_statement(z0_s, dtype)
            akms_s = _wrf_real_statement(akms_s, dtype)
            akhs_s = _wrf_real_statement(akhs_s, dtype)

        return qs_s, uz0_s, vz0_s, thz0_s, qz0_s, ustar_s, z0_s, akms_s, akhs_s, rib_s, rlmo_s, zu_s, zt_s, psmz_s, pshz_s, ustark_s, wstar2_s

    def land_branch() -> tuple[Any, ...]:
        qs_l = _wrf_real_statement(jnp.where(first_step, qlow, qs), dtype)
        uz0_l = jnp.asarray(0.0, dtype=dtype)
        vz0_l = jnp.asarray(0.0, dtype=dtype)
        thz0_l = _wrf_real_statement(ths, dtype)
        qz0_l = qs_l
        ustar_l = _wrf_real_statement(ustar, dtype)
        z0_l = _wrf_real_statement(z0, dtype)
        akms_l = _wrf_real_statement(akms, dtype)
        akhs_l = _wrf_real_statement(akhs, dtype)
        zu_l = z0_l
        zt_l = _wrf_mul(zu_l, ZTFC, dtype)
        dthv = _dthv(thelow, thz0_l, qlow, qz0_l, cwmlow, tlow, tz0)
        du = _wrf_sub(ulow, uz0_l, dtype)
        dv = _wrf_sub(vlow, vz0_l, dtype)
        du2 = _wrf_real_statement(
            jnp.maximum(
                _wrf_add(_wrf_mul(du, du, dtype), _wrf_mul(dv, dv, dtype), dtype),
                jnp.asarray(EPSU2, dtype=dtype),
            ),
            dtype,
        )
        rib_l = _wrf_div(_wrf_mul(_wrf_mul(btgx, dthv, dtype), zsl, dtype), du2, dtype)
        zslu = _wrf_add(zsl, zu_l, dtype)
        rzsu = _wrf_div(zslu, zu_l, dtype)
        rlogu = _wrf_log(rzsu, dtype)
        zslt = _wrf_add(zsl, zu_l, dtype)
        zilfc = _wrf_mul(_wrf_mul(jnp.asarray(-0.1, dtype=dtype), VKARMAN, dtype), SQVISC, dtype)
        czetmax = jnp.asarray(10.0, dtype=dtype)
        rib_ratio = _wrf_div(rib_l, RIC, dtype)
        stable_below_ric = _wrf_mul(
            zilfc,
            _wrf_add(
                1.0,
                _wrf_mul(_wrf_mul(rib_ratio, rib_ratio, dtype), czetmax, dtype),
                dtype,
            ),
            dtype,
        )
        stable_above_ric = _wrf_mul(zilfc, _wrf_add(1.0, czetmax, dtype), dtype)
        zzil = jnp.where(
            dthv > 0.0,
            jnp.where(rib_l < RIC, stable_below_ric, stable_above_ric),
            zilfc,
        )
        zzil = _wrf_real_statement(zzil, dtype)
        rlmo_l = jnp.asarray(0.0, dtype=dtype)
        psmz_l = jnp.asarray(0.0, dtype=dtype)
        pshz_l = jnp.asarray(0.0, dtype=dtype)
        ustark_l = _wrf_mul(ustar_l, VKARMAN, dtype)
        wstar2_l = jnp.asarray(0.0, dtype=dtype)

        for _ in range(ITRMX):
            zt_l = _wrf_real_statement(
                jnp.maximum(
                    _wrf_mul(
                        _wrf_exp(_wrf_mul(zzil, _wrf_sqrt(_wrf_mul(ustar_l, z0base, dtype), dtype), dtype), dtype, tables=parameters['exponential']),
                        z0base,
                        dtype,
                    ),
                    jnp.asarray(EPSZT, dtype=dtype),
                ),
                dtype,
            )
            rzst = _wrf_div(zslt, zt_l, dtype)
            rlogt = _wrf_log(rzst, dtype)
            ustar_squared = _wrf_mul(ustar_l, ustar_l, dtype)
            ustar_cubed = _wrf_mul(ustar_squared, ustar_l, dtype)
            rlmo_l = _wrf_div(
                _wrf_mul(_wrf_mul(elfc, akhs_l, dtype), dthv, dtype),
                ustar_cubed,
                dtype,
            )
            zetalu = _wrf_real_statement(
                jnp.clip(_wrf_mul(zslu, rlmo_l, dtype), ztmin2, ztmax2),
                dtype,
            )
            zetalt = _wrf_real_statement(
                jnp.clip(_wrf_mul(zslt, rlmo_l, dtype), ztmin2, ztmax2),
                dtype,
            )
            zetau = _wrf_real_statement(
                jnp.clip(
                    _wrf_mul(zu_l, rlmo_l, dtype),
                    _wrf_div(ztmin2, rzsu, dtype),
                    _wrf_div(ztmax2, rzsu, dtype),
                ),
                dtype,
            )
            zetat = _wrf_real_statement(
                jnp.clip(
                    _wrf_mul(zt_l, rlmo_l, dtype),
                    _wrf_div(ztmin2, rzst, dtype),
                    _wrf_div(ztmax2, rzst, dtype),
                ),
                dtype,
            )
            psmz_l = _interp_similarity(psim2, zetau, ztmin2, dzeta2)
            psmzl = _interp_similarity(psim2, zetalu, ztmin2, dzeta2)
            simm = _wrf_add(_wrf_sub(psmzl, psmz_l, dtype), rlogu, dtype)
            pshz_l = _interp_similarity(psih2, zetat, ztmin2, dzeta2)
            pshzl = _interp_similarity(psih2, zetalt, ztmin2, dzeta2)
            simh = _wrf_mul(
                _wrf_add(_wrf_sub(pshzl, pshz_l, dtype), rlogt, dtype),
                fh02,
                dtype,
            )
            ustark_l = _wrf_mul(ustar_l, VKARMAN, dtype)
            akms_l = _wrf_real_statement(
                jnp.maximum(_wrf_div(ustark_l, simm, dtype), cxchl),
                dtype,
            )
            akhs_l = _wrf_real_statement(
                jnp.maximum(_wrf_div(ustark_l, simh, dtype), cxchl),
                dtype,
            )
            buoyancy = _wrf_mul(_wrf_mul(btgh, akhs_l, dtype), dthv, dtype)
            convective = _wrf_mul(
                WWST2,
                _wrf_power(jnp.abs(buoyancy), _wrf_div(2.0, 3.0, dtype), dtype, tables=parameters['power']),
                dtype,
            )
            wstar2_l = _wrf_real_statement(jnp.where(dthv <= 0.0, convective, 0.0), dtype)
            ustar_l = _wrf_real_statement(
                jnp.maximum(
                    _wrf_sqrt(
                        _wrf_mul(
                            akms_l,
                            _wrf_sqrt(_wrf_add(du2, wstar2_l, dtype), dtype),
                            dtype,
                        ),
                        dtype,
                    ),
                    jnp.asarray(EPSUST, dtype=dtype),
                ),
                dtype,
            )
        return qs_l, uz0_l, vz0_l, thz0_l, qz0_l, ustar_l, z0_l, akms_l, akhs_l, rib_l, rlmo_l, zu_l, zt_l, psmz_l, pshz_l, ustark_l, wstar2_l

    sea_values = sea_branch()
    land_values = land_branch()
    sea = seamask > 0.5
    (
        qs_out,
        uz0_out,
        vz0_out,
        thz0_out,
        qz0_out,
        ustar_out,
        z0_out,
        akms_out,
        akhs_out,
        rib_out,
        rlmo_out,
        zu,
        zt,
        psmz,
        pshz,
        ustark,
        wstar2,
    ) = tuple(jnp.where(sea, sea_value, land_value) for sea_value, land_value in zip(sea_values, land_values, strict=True))

    del wstar2
    umflx = akms_out * (ulow - uz0_out)
    vmflx = akms_out * (vlow - vz0_out)
    hsflx = akhs_out * (thlow - thz0_out)
    hlflx = akhs_out * (qlow - qz0_out)

    sea_akms10, sea_akhs02, sea_akhs10 = _surface_similarity(
        psim1, psih1, ztmin1, ztmax1, dzeta1, fh01, cxchs, zu, zt, rlmo_out, psmz, pshz, ustark
    )
    land_akms10, land_akhs02, land_akhs10 = _surface_similarity(
        psim2, psih2, ztmin2, ztmax2, dzeta2, fh02, cxchl, zu, zt, rlmo_out, psmz, pshz, ustark
    )
    akms10 = jnp.where(sea, sea_akms10, land_akms10)
    akhs02 = jnp.where(sea, sea_akhs02, land_akhs02)
    akhs10 = jnp.where(sea, sea_akhs10, land_akhs10)

    u10 = column_binary('div', umflx, akms10) + uz0_out
    v10 = column_binary('div', vmflx, akms10) + vz0_out
    th02 = column_binary('div', hsflx, akhs02) + thz0_out
    th02_linear = thz0_out + 2.0 * rd_z * (thlow - thz0_out)
    th02_bad = ((thlow > thz0_out) & ((th02 < thz0_out) | (th02 > thlow))) | (
        (thlow < thz0_out) & ((th02 > thz0_out) | (th02 < thlow))
    )
    th02 = jnp.where(th02_bad, th02_linear, th02)

    th10 = column_binary('div', hsflx, akhs10) + thz0_out
    th10_linear = thz0_out + 10.0 * rd_z * (thlow - thz0_out)
    th10_bad = ((thlow > thz0_out) & ((th10 < thz0_out) | (th10 > thlow))) | (
        (thlow < thz0_out) & ((th10 > thz0_out) | (th10 < thlow))
    )
    th10 = jnp.where(th10_bad, th10_linear, th10)

    q02 = column_binary('div', hlflx, akhs02) + qz0_out
    q10 = column_binary('div', hlflx, akhs10) + qz0_out
    pshltr = psfc * _wrf_exp(column_binary('div', -0.068283, tlow), dtype, tables=parameters['exponential'])
    u10e = u10
    v10e = v10

    zu_effective = jnp.maximum(zu * 0.35, jnp.minimum(zu * 0.50, 0.18))
    zu10_effective = zu_effective + 10.0
    rzsu_effective = column_binary('div', zu10_effective, zu_effective)
    rlnu10_effective = jnp.log(rzsu_effective)
    zetau_effective = jnp.clip(zu_effective * rlmo_out, column_binary('div', ztmin2, rzsu_effective), column_binary('div', ztmax2, rzsu_effective))
    ztau10_effective = jnp.clip(zu10_effective * rlmo_out, ztmin2, ztmax2)
    psm10_effective = _interp_similarity(psim2, ztau10_effective, ztmin2, dzeta2)
    simm10_effective = psm10_effective - psmz + rlnu10_effective
    ekms10 = jnp.maximum(column_binary('div', ustark, simm10_effective), cxchl)
    land_u10e = column_binary('div', umflx, ekms10) + uz0_out
    land_v10e = column_binary('div', vmflx, ekms10) + vz0_out
    u10e = jnp.where(seamask < 0.5, land_u10e, u10e)
    v10e = jnp.where(seamask < 0.5, land_v10e, v10e)
    del zetau_effective

    rlow = column_binary('div', plow, R_D * tlow)
    chs = akhs_out
    chs2 = akhs02
    cqs2 = akhs02
    hfx = -rlow * CP * hsflx
    qfx = -rlow * hlflx * wetm
    flx_lh = XLV * qfx
    flhc = rlow * CP * akhs_out
    flqc = rlow * akhs_out * wetm
    qgh = column_binary('div', (1.0 - seamask) * PQ0 + seamask * PQ0SEA, plow) * _wrf_exp(column_binary('div', A2S * (tlow - A3S), tlow - A4S), dtype, tables=parameters['exponential'])
    qgh = column_binary('div', qgh, 1.0 - qgh)
    cpm = CP * (1.0 + 0.8 * qlow)
    qs_sea = column_binary('div', PQ0SEA, psfc) * _wrf_exp(column_binary('div', A2S * (tsk - A3S), tsk - A4S), dtype, tables=parameters['exponential'])
    qs_sea = column_binary('div', qs_sea, 1.0 - qs_sea)
    qs_out = jnp.where(sea, qs_sea, qs_out)

    result = dict(
        chs=chs,
        chs2=chs2,
        cpm=cpm,
        cqs2=cqs2,
        flhc=flhc,
        flqc=flqc,
        flx_lh=flx_lh,
        hfx=hfx,
        pshltr=pshltr,
        q02=q02,
        q10=q10,
        qfx=qfx,
        qgh=qgh,
        rib=rib_out,
        rlmo=rlmo_out,
        th02=th02,
        th10=th10,
        u10=u10,
        v10=v10,
        u10e=u10e,
        v10e=v10e,
        akhs=akhs_out,
        akms=akms_out,
        qz0=qz0_out,
        thz0=thz0_out,
        ustar=ustar_out,
        uz0=uz0_out,
        vz0=vz0_out,
        z0=z0_out,
        qs=qs_out,
    )
    shape = jnp.broadcast_shapes(*(value.shape for value in inputs))
    return {name: jnp.broadcast_to(value, shape) for name, value in result.items()}


def myjsfc(
    itimestep: int,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    qv: Any,
    qc: Any,
    u: Any,
    v: Any,
    q2: Any,
    tsk: Any,
    qsfc: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    ivgtyp: Any,
    isurban: int,
    iz0tlnd: int,
    ustar: Any,
    znt: Any,
    z0base: Any,
    pblh: Any,
    mavail: Any,
    rmol: Any,
    akhs: Any,
    akms: Any,
    rib: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    hfx: Any,
    qfx: Any,
    flx_lh: Any,
    flhc: Any,
    flqc: Any,
    qgh: Any,
    cpm: Any,
    ct: Any,
    u10: Any,
    v10: Any,
    t02: Any,
    th02: Any,
    tshltr: Any,
    th10: Any,
    q02: Any,
    qshltr: Any,
    q10: Any,
    pshltr: Any,
    p1000mb: float,
    u10e: Any,
    v10e: Any,
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
    *,
    tables: MyjSfcTables | None = None,
) -> MyjSfcResult:
    """Run WRF ``MYJSFC`` over the active horizontal tile."""

    del ids, jds, kds, kde
    tables = _build_tables() if tables is None else tables

    ht_arr = jnp.asarray(ht)
    dz_arr = jnp.asarray(dz)
    pmid_arr = jnp.asarray(pmid)
    pint_arr = jnp.asarray(pint)
    th_arr = jnp.asarray(th)
    t_arr = jnp.asarray(t)
    qv_arr = jnp.asarray(qv)
    qc_arr = jnp.asarray(qc)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    q2_arr = jnp.asarray(q2)
    tsk_arr = jnp.asarray(tsk)
    lowlyr_arr = jnp.asarray(lowlyr)
    xland_arr = jnp.asarray(xland)
    ivgtyp_arr = jnp.asarray(ivgtyp)
    z0base_arr = jnp.asarray(z0base)
    mavail_arr = jnp.asarray(mavail)

    qsfc_out = jnp.asarray(qsfc).copy()
    thz0_out = jnp.asarray(thz0).copy()
    qz0_out = jnp.asarray(qz0).copy()
    uz0_out = jnp.asarray(uz0).copy()
    vz0_out = jnp.asarray(vz0).copy()
    ustar_out = jnp.asarray(ustar).copy()
    znt_out = jnp.asarray(znt).copy()
    pblh_out = jnp.asarray(pblh).copy()
    rmol_out = jnp.asarray(rmol).copy()
    akhs_out = jnp.asarray(akhs).copy()
    akms_out = jnp.asarray(akms).copy()
    rib_out = jnp.asarray(rib).copy()
    chs_out = jnp.asarray(chs).copy()
    chs2_out = jnp.asarray(chs2).copy()
    cqs2_out = jnp.asarray(cqs2).copy()
    hfx_out = jnp.asarray(hfx).copy()
    qfx_out = jnp.asarray(qfx).copy()
    flx_lh_out = jnp.asarray(flx_lh).copy()
    flhc_out = jnp.asarray(flhc).copy()
    flqc_out = jnp.asarray(flqc).copy()
    qgh_out = jnp.asarray(qgh).copy()
    cpm_out = jnp.asarray(cpm).copy()
    ct_out = jnp.asarray(ct).copy()
    u10_out = jnp.asarray(u10).copy()
    v10_out = jnp.asarray(v10).copy()
    t02_out = jnp.asarray(t02).copy()
    th02_out = jnp.asarray(th02).copy()
    tshltr_out = jnp.asarray(tshltr).copy()
    th10_out = jnp.asarray(th10).copy()
    q02_out = jnp.asarray(q02).copy()
    qshltr_out = jnp.asarray(qshltr).copy()
    q10_out = jnp.asarray(q10).copy()
    pshltr_out = jnp.asarray(pshltr).copy()
    u10e_out = jnp.asarray(u10e).copy()
    v10e_out = jnp.asarray(v10e).copy()

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set2(name: str, arr: Any, i: int, j: int, value: Any) -> Any:
        del name
        return arr.at[_idx(i, ims), _idx(j, jms)].set(value)

    zint: dict[tuple[int, int, int], Any] = {}
    for j in range(int(jts), int(jte) + 1):
        for k in range(int(kts), int(kte) + 2):
            for i in range(int(its), int(ite) + 1):
                zint[(i, k, j)] = jnp.asarray(0.0, dtype=ht_arr.dtype)

    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            zint[(i, int(kte) + 1, j)] = a2(ht_arr, i, j)
            pblh_out = set2("pblh", pblh_out, i, j, -1.0)

    for j in range(int(jts), int(jte) + 1):
        for k in range(int(kte), int(kts) - 1, -1):
            kflip = int(kte) + 1 - k
            for i in range(int(its), int(ite) + 1):
                zint[(i, k, j)] = zint[(i, k + 1, j)] + a3(dz_arr, i, kflip, j)

    ntsd = int(itimestep)
    if ntsd == 1:
        for j in range(int(jts), int(jte) + 1):
            for i in range(int(its), int(ite) + 1):
                ustar_out = set2("ustar", ustar_out, i, j, 0.1)

    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            ct_out = set2("ct", ct_out, i, j, 0.0)

    capa = R_D / CP
    rcap = 1.0 / capa
    gocp02 = G / CP * 2.0
    gocp10 = G / CP * 10.0

    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            low = int(a2(lowlyr_arr, i, j))
            lmh = int(kte) - low + 1
            psfc = a3(pint_arr, i, low, j)
            thsk = a2(tsk_arr, i, j) / (psfc / float(p1000mb)) ** capa
            seamask = a2(xland_arr, i, j) - 1.0

            cwmk: dict[int, Any] = {}
            pk: dict[int, Any] = {}
            q2k: dict[int, Any] = {}
            qk: dict[int, Any] = {}
            thek: dict[int, Any] = {}
            thk: dict[int, Any] = {}
            tk: dict[int, Any] = {}
            uk: dict[int, Any] = {}
            vk: dict[int, Any] = {}
            zhk: dict[int, Any] = {}

            for k in range(int(kte), int(kts) - 1, -1):
                kflip = int(kte) + 1 - k
                thk[k] = a3(th_arr, i, kflip, j)
                tk[k] = a3(t_arr, i, kflip, j)
                ratiomx = a3(qv_arr, i, kflip, j)
                qk[k] = ratiomx / (1.0 + ratiomx)
                pk[k] = a3(pmid_arr, i, kflip, j)
                cwmk[k] = a3(qc_arr, i, kflip, j)
                thek[k] = (cwmk[k] * (-ELOCP / tk[k]) + 1.0) * thk[k]
                q2k[k] = 2.0 * a3(q2_arr, i, kflip, j)
                zhk[k] = zint[(i, k, j)]
            zhk[int(kte) + 1] = a2(ht_arr, i, j)

            for k in range(int(kte), int(kts) - 1, -1):
                kflip = int(kte) + 1 - k
                uk[k] = a3(u_arr, i, kflip, j)
                vk[k] = a3(v_arr, i, kflip, j)

            lpbl = lmh
            found = False
            for k in range(lmh - 1, 0, -1):
                if bool(q2k[k] <= EPSQ2 * FH):
                    lpbl = k
                    found = True
                    break
            if not found:
                lpbl = 1

            pblh_value = zhk[lpbl] - zhk[lmh + 1]
            pblh_out = set2("pblh", pblh_out, i, j, pblh_value)

            plow = pk[lmh]
            tlow = tk[lmh]
            thlow = thk[lmh]
            thelow = thek[lmh]
            qlow = qk[lmh]
            cwmlow = cwmk[lmh]
            ulow = uk[lmh]
            vlow = vk[lmh]
            zsl = (zhk[lmh] - zhk[lmh + 1]) * 0.5
            apesfc = (psfc / float(p1000mb)) ** capa
            tz0 = a2(tsk_arr, i, j) if ntsd == 1 else a2(thz0_out, i, j) * apesfc

            surface = sfcdif(
                ntsd,
                seamask,
                thsk,
                a2(qsfc_out, i, j),
                psfc,
                a2(uz0_out, i, j),
                a2(vz0_out, i, j),
                tz0,
                a2(tsk_arr, i, j),
                a2(thz0_out, i, j),
                a2(qz0_out, i, j),
                a2(ustar_out, i, j),
                a2(znt_out, i, j),
                a2(z0base_arr, i, j),
                a2(akms_out, i, j),
                a2(akhs_out, i, j),
                pblh_value,
                a2(mavail_arr, i, j),
                int(a2(ivgtyp_arr, i, j)),
                int(isurban),
                int(iz0tlnd),
                ulow,
                vlow,
                tlow,
                thlow,
                thelow,
                qlow,
                cwmlow,
                zsl,
                plow,
                0,
                int(ide),
                0,
                int(jde),
                0,
                0,
                int(ims),
                int(ime),
                int(jms),
                int(jme),
                int(kms),
                int(kme),
                int(its),
                int(ite),
                int(jts),
                int(jte),
                int(kts),
                int(kte),
                i,
                j,
                zhk[lmh + 1],
                tables,
            )

            qsfc_out = set2("qsfc", qsfc_out, i, j, surface.qs)
            uz0_out = set2("uz0", uz0_out, i, j, surface.uz0)
            vz0_out = set2("vz0", vz0_out, i, j, surface.vz0)
            thz0_out = set2("thz0", thz0_out, i, j, surface.thz0)
            qz0_out = set2("qz0", qz0_out, i, j, surface.qz0)
            ustar_out = set2("ustar", ustar_out, i, j, surface.ustar)
            znt_out = set2("znt", znt_out, i, j, surface.z0)
            ct_out = set2("ct", ct_out, i, j, surface.ct)
            rmol_out = set2("rmol", rmol_out, i, j, surface.rlmo)
            akms_out = set2("akms", akms_out, i, j, surface.akms)
            akhs_out = set2("akhs", akhs_out, i, j, surface.akhs)
            rib_out = set2("rib", rib_out, i, j, surface.rib)
            chs_out = set2("chs", chs_out, i, j, surface.chs)
            chs2_out = set2("chs2", chs2_out, i, j, surface.chs2)
            cqs2_out = set2("cqs2", cqs2_out, i, j, surface.cqs2)
            hfx_out = set2("hfx", hfx_out, i, j, surface.hfx)
            qfx_out = set2("qfx", qfx_out, i, j, surface.qfx)
            flx_lh_out = set2("flx_lh", flx_lh_out, i, j, surface.flx_lh)
            flhc_out = set2("flhc", flhc_out, i, j, surface.flhc)
            flqc_out = set2("flqc", flqc_out, i, j, surface.flqc)
            qgh_out = set2("qgh", qgh_out, i, j, surface.qgh)
            cpm_out = set2("cpm", cpm_out, i, j, surface.cpm)
            u10_out = set2("u10", u10_out, i, j, surface.u10)
            v10_out = set2("v10", v10_out, i, j, surface.v10)
            tshltr_value = surface.th02
            th10_value = surface.th10
            qshltr_value = surface.q02
            q10_value = surface.q10
            tshltr_out = set2("tshltr", tshltr_out, i, j, tshltr_value)
            th10_out = set2("th10", th10_out, i, j, th10_value)
            pshltr_out = set2("pshltr", pshltr_out, i, j, surface.pshltr)
            u10e_out = set2("u10e", u10e_out, i, j, surface.u10e)
            v10e_out = set2("v10e", v10e_out, i, j, surface.v10e)

            rapa = apesfc
            th02p = tshltr_value
            th10p = th10_value
            th02_out = set2("th02", th02_out, i, j, tshltr_value)
            rapa02 = rapa - gocp02 / th02p
            rapa10 = rapa - gocp10 / th10p
            t02p = th02p * rapa02
            t10p = th10p * rapa10
            t02_out = set2("t02", t02_out, i, j, tshltr_value * apesfc)
            p02p = rapa02**rcap * float(p1000mb)
            p10p = rapa10**rcap * float(p1000mb)
            qs02 = PQ0 / p02p * _wrf_exp(A2S * (t02p - A3S) / (t02p - A4S), jnp.asarray(t02p).dtype)
            qs10 = PQ0 / p10p * _wrf_exp(A2S * (t10p - A3S) / (t10p - A4S), jnp.asarray(t10p).dtype)
            qshltr_value = jnp.where(qshltr_value > qs02, qs02, qshltr_value)
            q10_value = jnp.where(q10_value > qs10, qs10, q10_value)
            qshltr_out = set2("qshltr", qshltr_out, i, j, qshltr_value)
            q10_out = set2("q10", q10_out, i, j, q10_value)
            q02_out = set2("q02", q02_out, i, j, qshltr_value / (1.0 - qshltr_value))

    return MyjSfcResult(
        qsfc=qsfc_out,
        thz0=thz0_out,
        qz0=qz0_out,
        uz0=uz0_out,
        vz0=vz0_out,
        ustar=ustar_out,
        znt=znt_out,
        pblh=pblh_out,
        rmol=rmol_out,
        akhs=akhs_out,
        akms=akms_out,
        rib=rib_out,
        chs=chs_out,
        chs2=chs2_out,
        cqs2=cqs2_out,
        hfx=hfx_out,
        qfx=qfx_out,
        flx_lh=flx_lh_out,
        flhc=flhc_out,
        flqc=flqc_out,
        qgh=qgh_out,
        cpm=cpm_out,
        ct=ct_out,
        u10=u10_out,
        v10=v10_out,
        t02=t02_out,
        th02=th02_out,
        tshltr=tshltr_out,
        th10=th10_out,
        q02=q02_out,
        qshltr=qshltr_out,
        q10=q10_out,
        pshltr=pshltr_out,
        u10e=u10e_out,
        v10e=v10e_out,
    )


def _myjsfc_vectorized_impl(
    itimestep: int,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    qv: Any,
    qc: Any,
    u: Any,
    v: Any,
    q2: Any,
    tsk: Any,
    qsfc: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    ivgtyp: Any,
    isurban: int,
    iz0tlnd: int,
    ustar: Any,
    znt: Any,
    z0base: Any,
    pblh: Any,
    mavail: Any,
    rmol: Any,
    akhs: Any,
    akms: Any,
    rib: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    hfx: Any,
    qfx: Any,
    flx_lh: Any,
    flhc: Any,
    flqc: Any,
    qgh: Any,
    cpm: Any,
    ct: Any,
    u10: Any,
    v10: Any,
    t02: Any,
    th02: Any,
    tshltr: Any,
    th10: Any,
    q02: Any,
    qshltr: Any,
    q10: Any,
    pshltr: Any,
    p1000mb: float,
    u10e: Any,
    v10e: Any,
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
    *,
    tables: MyjSfcTables | None = None,
) -> MyjSfcResult:
    """Vectorized WRF ``MYJSFC`` for the active horizontal tile."""

    del ids, jds, kds, kde
    tables = _build_tables() if tables is None else tables

    i_slice = slice(int(its) - int(ims), int(ite) - int(ims) + 1)
    j_slice = slice(int(jts) - int(jms), int(jte) - int(jms) + 1)
    k_slice = slice(int(kts) - int(kms), int(kte) - int(kms) + 1)
    nlev = int(kte) - int(kts) + 1

    def tile2(value: Any) -> Any:
        return jnp.asarray(value)[i_slice, j_slice]

    def tile3(value: Any) -> Any:
        return jnp.asarray(value)[i_slice, k_slice, j_slice]

    def slice_covers_axis(slc: slice, size: int) -> bool:
        start = 0 if slc.start is None else int(slc.start)
        stop = size if slc.stop is None else int(slc.stop)
        step = 1 if slc.step is None else int(slc.step)
        return start == 0 and stop == int(size) and step == 1

    def put2(target: Any, value: Any) -> Any:
        target_arr = jnp.asarray(target)
        value_arr = jnp.asarray(value, dtype=target_arr.dtype)
        if (
            value_arr.shape == target_arr.shape
            and slice_covers_axis(i_slice, target_arr.shape[0])
            and slice_covers_axis(j_slice, target_arr.shape[1])
        ):
            return value_arr
        return target_arr.at[i_slice, j_slice].set(value_arr)

    def gather_level(arr: Any, level: Any) -> Any:
        level_idx = jnp.clip(jnp.asarray(level, dtype=jnp.int32) - int(kts), 0, nlev - 1)
        return jnp.take_along_axis(arr, level_idx[:, None, :], axis=1)[:, 0, :]

    def gather_fortran_k(arr: Any, level: Any) -> Any:
        level_idx = jnp.clip(jnp.asarray(level, dtype=jnp.int32) - int(kts), 0, nlev)
        return jnp.take_along_axis(arr, level_idx[:, None, :], axis=1)[:, 0, :]

    ht_tile = tile2(ht)
    dz_tile = tile3(dz)
    pmid_tile = tile3(pmid)
    pint_tile = tile3(pint)
    th_tile = tile3(th)
    t_tile = tile3(t)
    qv_tile = tile3(qv)
    qc_tile = tile3(qc)
    u_tile = tile3(u)
    v_tile = tile3(v)
    q2_tile = tile3(q2)
    tsk_tile = tile2(tsk)
    lowlyr_tile = tile2(lowlyr).astype(jnp.int32)
    xland_tile = tile2(xland)
    ivgtyp_tile = tile2(ivgtyp).astype(jnp.int32)

    cumsum_dz = jnp.cumsum(dz_tile[:, :nlev, :], axis=1)
    zint_without_surface = ht_tile[:, None, :] + jnp.flip(cumsum_dz, axis=1)
    zint = jnp.concatenate([zint_without_surface, ht_tile[:, None, :]], axis=1)

    q2_fortran_k = 2.0 * jnp.flip(q2_tile[:, :nlev, :], axis=1)
    k_values = jnp.arange(int(kts), int(kte) + 1, dtype=jnp.int32)[None, :, None]
    lmh = (int(kte) - lowlyr_tile + 1).astype(jnp.int32)
    weak_tke = q2_fortran_k <= jnp.asarray(EPSQ2 * FH, dtype=q2_fortran_k.dtype)
    valid_pbl = weak_tke & (k_values < lmh[:, None, :])
    lpbl = jnp.max(jnp.where(valid_pbl, k_values, int(kts)), axis=1).astype(jnp.int32)
    pblh_tile = gather_fortran_k(zint, lpbl) - gather_fortran_k(zint, lmh + 1)

    psfc = gather_level(pint_tile, lowlyr_tile)
    dtype = tsk_tile.dtype
    store = lambda value: _wrf_real_statement(value, dtype)
    capa = store(R_D / CP)
    rcap = store(1.0 / capa)
    pressure_ratio = store(psfc / store(float(p1000mb)))
    apesfc = _wrf_power(pressure_ratio, capa, dtype)
    thsk = store(tsk_tile / apesfc)
    seamask = store(xland_tile - 1.0)

    plow = gather_level(pmid_tile, lowlyr_tile)
    tlow = gather_level(t_tile, lowlyr_tile)
    thlow = gather_level(th_tile, lowlyr_tile)
    ratiomx = gather_level(qv_tile, lowlyr_tile)
    qlow = store(ratiomx / store(1.0 + ratiomx))
    cwmlow = gather_level(qc_tile, lowlyr_tile)
    thelow = store(store(cwmlow * store(-ELOCP / tlow)) + 1.0)
    thelow = store(thelow * thlow)
    ulow = gather_level(u_tile, lowlyr_tile)
    vlow = gather_level(v_tile, lowlyr_tile)
    zsl = store(gather_fortran_k(zint, lmh) - gather_fortran_k(zint, lmh + 1))
    zsl = store(zsl * 0.5)

    thz0_tile = tile2(thz0)
    first_step = jnp.asarray(itimestep, dtype=jnp.int32) == 1
    ustar_tile = jnp.where(first_step, jnp.full_like(tile2(ustar), 0.1), tile2(ustar))
    tz0 = jnp.where(first_step, tsk_tile, thz0_tile * apesfc)

    surface = sfcdif(
        itimestep,
        seamask,
        thsk,
        tile2(qsfc),
        psfc,
        tile2(uz0),
        tile2(vz0),
        tz0,
        tsk_tile,
        thz0_tile,
        tile2(qz0),
        ustar_tile,
        tile2(znt),
        tile2(z0base),
        tile2(akms),
        tile2(akhs),
        pblh_tile,
        tile2(mavail),
        ivgtyp_tile,
        int(isurban),
        int(iz0tlnd),
        ulow,
        vlow,
        tlow,
        thlow,
        thelow,
        qlow,
        cwmlow,
        zsl,
        plow,
        0,
        int(ide),
        0,
        int(jde),
        0,
        0,
        int(ims),
        int(ime),
        int(jms),
        int(jme),
        int(kms),
        int(kme),
        int(its),
        int(ite),
        int(jts),
        int(jte),
        int(kts),
        int(kte),
        0,
        0,
        gather_fortran_k(zint, lmh + 1),
        tables,
    )

    gocp02 = G / CP * 2.0
    gocp10 = G / CP * 10.0
    th02p = surface.th02
    th10p = surface.th10
    rapa02 = apesfc - gocp02 / th02p
    rapa10 = apesfc - gocp10 / th10p
    t02p = th02p * rapa02
    t10p = th10p * rapa10
    p02p = rapa02**rcap * float(p1000mb)
    p10p = rapa10**rcap * float(p1000mb)
    qs02 = PQ0 / p02p * _wrf_exp(A2S * (t02p - A3S) / (t02p - A4S), dtype)
    qs10 = PQ0 / p10p * _wrf_exp(A2S * (t10p - A3S) / (t10p - A4S), dtype)
    qshltr_tile = jnp.where(surface.q02 > qs02, qs02, surface.q02)
    q10_tile = jnp.where(surface.q10 > qs10, qs10, surface.q10)

    return MyjSfcResult(
        qsfc=put2(qsfc, surface.qs),
        thz0=put2(thz0, surface.thz0),
        qz0=put2(qz0, surface.qz0),
        uz0=put2(uz0, surface.uz0),
        vz0=put2(vz0, surface.vz0),
        ustar=put2(ustar, surface.ustar),
        znt=put2(znt, surface.z0),
        pblh=put2(pblh, pblh_tile),
        rmol=put2(rmol, surface.rlmo),
        akhs=put2(akhs, surface.akhs),
        akms=put2(akms, surface.akms),
        rib=put2(rib, surface.rib),
        chs=put2(chs, surface.chs),
        chs2=put2(chs2, surface.chs2),
        cqs2=put2(cqs2, surface.cqs2),
        hfx=put2(hfx, surface.hfx),
        qfx=put2(qfx, surface.qfx),
        flx_lh=put2(flx_lh, surface.flx_lh),
        flhc=put2(flhc, surface.flhc),
        flqc=put2(flqc, surface.flqc),
        qgh=put2(qgh, surface.qgh),
        cpm=put2(cpm, surface.cpm),
        ct=put2(ct, surface.ct),
        u10=put2(u10, surface.u10),
        v10=put2(v10, surface.v10),
        t02=put2(t02, surface.th02 * apesfc),
        th02=put2(th02, surface.th02),
        tshltr=put2(tshltr, surface.th02),
        th10=put2(th10, surface.th10),
        q02=put2(q02, qshltr_tile / (1.0 - qshltr_tile)),
        qshltr=put2(qshltr, qshltr_tile),
        q10=put2(q10, q10_tile),
        pshltr=put2(pshltr, surface.pshltr),
        u10e=put2(u10e, surface.u10e),
        v10e=put2(v10e, surface.v10e),
    )


def _myjsfc_result_tuple(result: MyjSfcResult) -> tuple[Any, ...]:
    return (
        result.qsfc,
        result.thz0,
        result.qz0,
        result.uz0,
        result.vz0,
        result.ustar,
        result.znt,
        result.pblh,
        result.rmol,
        result.akhs,
        result.akms,
        result.rib,
        result.chs,
        result.chs2,
        result.cqs2,
        result.hfx,
        result.qfx,
        result.flx_lh,
        result.flhc,
        result.flqc,
        result.qgh,
        result.cpm,
        result.ct,
        result.u10,
        result.v10,
        result.t02,
        result.th02,
        result.tshltr,
        result.th10,
        result.q02,
        result.qshltr,
        result.q10,
        result.pshltr,
        result.u10e,
        result.v10e,
    )


def _myjsfc_result_from_tuple(values: tuple[Any, ...]) -> MyjSfcResult:
    return MyjSfcResult(*values)


_MYJSFC_STATIC_ARGNUMS = (21, 22, 53, *range(56, 74))
_MYJSFC_DYNAMIC_ARGNUMS = tuple(index for index in range(74) if index not in _MYJSFC_STATIC_ARGNUMS)
_JIT_MYJSFC_VECTORIZED = os.environ.get("WRF_PY_JIT_MYJSFC_VECTORIZED", "1").strip().lower() in {"1", "true", "yes", "on"}
_INLINE_COLUMN_PHYSICS_JITS = os.environ.get("WRF_PY_INLINE_COLUMN_PHYSICS_JITS", "0").strip().lower() in {"1", "true", "yes", "on"}


@partial(jax.jit, static_argnums=_MYJSFC_STATIC_ARGNUMS, inline=_INLINE_COLUMN_PHYSICS_JITS)
def _myjsfc_vectorized_jit_tuple(
    itimestep: int,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    qv: Any,
    qc: Any,
    u: Any,
    v: Any,
    q2: Any,
    tsk: Any,
    qsfc: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    ivgtyp: Any,
    isurban: int,
    iz0tlnd: int,
    ustar: Any,
    znt: Any,
    z0base: Any,
    pblh: Any,
    mavail: Any,
    rmol: Any,
    akhs: Any,
    akms: Any,
    rib: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    hfx: Any,
    qfx: Any,
    flx_lh: Any,
    flhc: Any,
    flqc: Any,
    qgh: Any,
    cpm: Any,
    ct: Any,
    u10: Any,
    v10: Any,
    t02: Any,
    th02: Any,
    tshltr: Any,
    th10: Any,
    q02: Any,
    qshltr: Any,
    q10: Any,
    pshltr: Any,
    p1000mb: float,
    u10e: Any,
    v10e: Any,
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
    table_dzeta1: Any,
    table_dzeta2: Any,
    table_fh01: Any,
    table_fh02: Any,
    table_ztmax1: Any,
    table_ztmax2: Any,
    table_ztmin1: Any,
    table_ztmin2: Any,
    table_psim1: Any,
    table_psih1: Any,
    table_psim2: Any,
    table_psih2: Any,
) -> tuple[Any, ...]:
    call_args = (
        itimestep,
        ht,
        dz,
        pmid,
        pint,
        th,
        t,
        qv,
        qc,
        u,
        v,
        q2,
        tsk,
        qsfc,
        thz0,
        qz0,
        uz0,
        vz0,
        lowlyr,
        xland,
        ivgtyp,
        isurban,
        iz0tlnd,
        ustar,
        znt,
        z0base,
        pblh,
        mavail,
        rmol,
        akhs,
        akms,
        rib,
        chs,
        chs2,
        cqs2,
        hfx,
        qfx,
        flx_lh,
        flhc,
        flqc,
        qgh,
        cpm,
        ct,
        u10,
        v10,
        t02,
        th02,
        tshltr,
        th10,
        q02,
        qshltr,
        q10,
        pshltr,
        p1000mb,
        u10e,
        v10e,
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
    )
    dynamic_values = optimization_barrier(
        tuple(call_args[index] for index in _MYJSFC_DYNAMIC_ARGNUMS)
    )
    materialized_args = list(call_args)
    for index, value in zip(_MYJSFC_DYNAMIC_ARGNUMS, dynamic_values, strict=True):
        materialized_args[index] = value
    table_values = optimization_barrier(
        (
            table_dzeta1,
            table_dzeta2,
            table_fh01,
            table_fh02,
            table_ztmax1,
            table_ztmax2,
            table_ztmin1,
            table_ztmin2,
            table_psim1,
            table_psih1,
            table_psim2,
            table_psih2,
        )
    )
    tables = MyjSfcTables(
        dzeta1=table_values[0],
        dzeta2=table_values[1],
        fh01=table_values[2],
        fh02=table_values[3],
        ztmax1=table_values[4],
        ztmax2=table_values[5],
        ztmin1=table_values[6],
        ztmin2=table_values[7],
        psim1=table_values[8],
        psih1=table_values[9],
        psim2=table_values[10],
        psih2=table_values[11],
    )
    return optimization_barrier(
        _myjsfc_result_tuple(
            _myjsfc_vectorized_impl(
                *materialized_args,
                tables=tables,
            )
        )
    )


def myjsfc_vectorized(
    itimestep: int,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    qv: Any,
    qc: Any,
    u: Any,
    v: Any,
    q2: Any,
    tsk: Any,
    qsfc: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    ivgtyp: Any,
    isurban: int,
    iz0tlnd: int,
    ustar: Any,
    znt: Any,
    z0base: Any,
    pblh: Any,
    mavail: Any,
    rmol: Any,
    akhs: Any,
    akms: Any,
    rib: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    hfx: Any,
    qfx: Any,
    flx_lh: Any,
    flhc: Any,
    flqc: Any,
    qgh: Any,
    cpm: Any,
    ct: Any,
    u10: Any,
    v10: Any,
    t02: Any,
    th02: Any,
    tshltr: Any,
    th10: Any,
    q02: Any,
    qshltr: Any,
    q10: Any,
    pshltr: Any,
    p1000mb: float,
    u10e: Any,
    v10e: Any,
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
    *,
    tables: MyjSfcTables | None = None,
) -> MyjSfcResult:
    """Vectorized WRF ``MYJSFC`` for the active horizontal tile."""

    tables = _build_tables() if tables is None else tables
    if not _JIT_MYJSFC_VECTORIZED:
        return _myjsfc_vectorized_impl(
            itimestep,
            ht,
            dz,
            pmid,
            pint,
            th,
            t,
            qv,
            qc,
            u,
            v,
            q2,
            tsk,
            qsfc,
            thz0,
            qz0,
            uz0,
            vz0,
            lowlyr,
            xland,
            ivgtyp,
            isurban,
            iz0tlnd,
            ustar,
            znt,
            z0base,
            pblh,
            mavail,
            rmol,
            akhs,
            akms,
            rib,
            chs,
            chs2,
            cqs2,
            hfx,
            qfx,
            flx_lh,
            flhc,
            flqc,
            qgh,
            cpm,
            ct,
            u10,
            v10,
            t02,
            th02,
            tshltr,
            th10,
            q02,
            qshltr,
            q10,
            pshltr,
            p1000mb,
            u10e,
            v10e,
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
            tables=tables,
        )
    return _myjsfc_result_from_tuple(
        _myjsfc_vectorized_jit_tuple(
            itimestep,
            ht,
            dz,
            pmid,
            pint,
            th,
            t,
            qv,
            qc,
            u,
            v,
            q2,
            tsk,
            qsfc,
            thz0,
            qz0,
            uz0,
            vz0,
            lowlyr,
            xland,
            ivgtyp,
            isurban,
            iz0tlnd,
            ustar,
            znt,
            z0base,
            pblh,
            mavail,
            rmol,
            akhs,
            akms,
            rib,
            chs,
            chs2,
            cqs2,
            hfx,
            qfx,
            flx_lh,
            flhc,
            flqc,
            qgh,
            cpm,
            ct,
            u10,
            v10,
            t02,
            th02,
            tshltr,
            th10,
            q02,
            qshltr,
            q10,
            pshltr,
            p1000mb,
            u10e,
            v10e,
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
            tables.dzeta1,
            tables.dzeta2,
            tables.fh01,
            tables.fh02,
            tables.ztmax1,
            tables.ztmax2,
            tables.ztmin1,
            tables.ztmin2,
            tables.psim1,
            tables.psih1,
            tables.psim2,
            tables.psih2,
        )
    )


def myjsfcinit(
    lowlyr: Any,
    ustar: Any,
    z0: Any,
    seamask: Any,
    xice: Any,
    ivgtyp: Any,
    restart: bool,
    allowed_to_read: bool,
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
) -> MyjSfcInitResult:
    """Initialize MYJ surface-layer low-level indices and lookup tables."""

    del seamask, xice, ivgtyp, allowed_to_read, ids, jds, kds, kde, kms, kme, kts, kte
    lowlyr_out = jnp.asarray(lowlyr)
    ustar_out = jnp.asarray(ustar)
    z0_out = jnp.asarray(z0)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    i_slice = slice(int(its) - int(ims), itf - int(ims) + 1)
    low_j_slice = slice(int(jts) - int(jms), jtf - int(jms) + 1)
    ustar_j_slice = slice(int(jts) - int(jms), int(jte) - int(jms) + 1)

    lowlyr_out = lowlyr_out.at[i_slice, low_j_slice].set(jnp.asarray(1, dtype=lowlyr_out.dtype))
    if not bool(restart):
        ustar_out = ustar_out.at[i_slice, ustar_j_slice].set(jnp.asarray(0.1, dtype=ustar_out.dtype))

    del ime, jme
    return MyjSfcInitResult(lowlyr=lowlyr_out, ustar=ustar_out, z0=z0_out, tables=_build_tables())


__all__ = [
    "EPSQ2",
    "ITRMX",
    "KZTM",
    "MyjSfcInitResult",
    "MyjSfcResult",
    "MyjSfcTables",
    "SfcdifResult",
    "myjsfc",
    "myjsfc_vectorized",
    "myjsfcinit",
    "sfcdif",
]
