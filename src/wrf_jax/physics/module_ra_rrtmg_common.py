"""Shared translations from WRF RRTMG radiation modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.column_scan import scan_columns


@jax.custom_jvp
def pade_transmittance(optical_depth, sampled_value):
    """Keep WRF's lookup value; differentiate the represented exponential.

    The integer table index has zero AD sensitivity. As with rounded libm
    functions, derivatives refer to the continuous function approximated by
    the table, not to its piecewise-constant quantization.
    """
    return sampled_value


@pade_transmittance.defjvp
def _pade_transmittance_jvp(primals, tangents):
    depth, value = primals
    ddepth, dvalue = tangents
    slope = jnp.where(value > jnp.asarray(1.e-20, value.dtype), -value, 0.)
    return value, dvalue + slope * jnp.asarray(ddepth, value.dtype)


@jax.custom_jvp
def pade_source_factor(optical_depth, sampled_value):
    """Preserve the tabulated source factor and its continuous sensitivity."""
    return sampled_value


@pade_source_factor.defjvp
def _pade_source_factor_jvp(primals, tangents):
    depth, value = primals
    ddepth, dvalue = tangents
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    tau = jnp.asarray(depth, dtype)
    safe = jnp.maximum(tau, .5)
    trans = jnp.exp(-safe)
    slope = 2./safe**2 - 2.*trans/(1.-trans)**2
    # Avoid subtracting nearly equal terms in the optically thin limit.
    squared = tau*tau
    series = 1./6. - squared/120. + squared**2/3024. - squared**3/86400.
    slope = jnp.where(tau < .5, series, slope).astype(value.dtype)
    return value, dvalue + slope * jnp.asarray(ddepth, value.dtype)


@jax.custom_jvp
def pade_optical_depth(optical_depth, sampled_value):
    """The rounded optical-depth table represents the identity function."""
    return sampled_value


@pade_optical_depth.defjvp
def _pade_optical_depth_jvp(primals, tangents):
    return primals[1], jnp.asarray(tangents[0], primals[1].dtype) + tangents[1]


@jax.custom_jvp
def wrf_real_pa_to_hpa(value: Any) -> Any:
    """Round WRF's REAL ``pressure / 100.`` identically on CPU and GPU.

    Keep the quotient wide until its final REAL assignment. The shared XLA
    configuration disables excess precision, preserving that conversion.
    Avoid identity nextafter operations here: when combined with the runtime's
    transpose/reshape, their integer-bit lowering corrupts GPU pressure values.
    """

    pressure = jnp.asarray(value, dtype=jnp.float32)
    quotient = pressure.astype(jnp.float64) / jnp.float64(100.0)
    return quotient.astype(jnp.float32)


@wrf_real_pa_to_hpa.defjvp
def _wrf_real_pa_to_hpa_jvp(
    primals: tuple[Any], tangents: tuple[Any],
) -> tuple[Any, Any]:
    (value,), (tangent,) = primals, tangents
    result = wrf_real_pa_to_hpa(value)
    return result, jnp.asarray(tangent, dtype=result.dtype) / jnp.float32(100.0)


RETAB = jnp.asarray(
    (
        5.92779,
        6.26422,
        6.61973,
        6.99539,
        7.39234,
        7.81177,
        8.25496,
        8.72323,
        9.21800,
        9.74075,
        10.2930,
        10.8765,
        11.4929,
        12.1440,
        12.8317,
        13.5581,
        14.2319,
        15.0351,
        15.8799,
        16.7674,
        17.6986,
        18.6744,
        19.6955,
        20.7623,
        21.8757,
        23.0364,
        24.2452,
        25.5034,
        26.8125,
        27.7895,
        28.6450,
        29.4167,
        30.1088,
        30.7306,
        31.2943,
        31.8151,
        32.3077,
        32.7870,
        33.2657,
        33.7540,
        34.2601,
        34.7892,
        35.3442,
        35.9255,
        36.5316,
        37.1602,
        37.8078,
        38.4720,
        39.1508,
        39.8442,
        40.5552,
        41.2912,
        42.0635,
        42.8876,
        43.7863,
        44.7853,
        45.9170,
        47.2165,
        48.7221,
        50.4710,
        52.4980,
        54.8315,
        57.4898,
        60.4785,
        63.7898,
        65.5604,
        71.2885,
        75.4113,
        79.7368,
        84.2351,
        88.8833,
        93.6658,
        98.5739,
        103.603,
        108.752,
        114.025,
        119.424,
        124.954,
        130.630,
        136.457,
        142.446,
        148.608,
        154.956,
        161.503,
        168.262,
        175.248,
        182.473,
        189.952,
        197.699,
        205.728,
        214.055,
        222.694,
        231.661,
        240.971,
        250.639,
    ),
    dtype=jnp.float32,
)

_O3SUM = jnp.asarray(
    (
        5.297e-8,
        5.852e-8,
        6.579e-8,
        7.505e-8,
        8.577e-8,
        9.895e-8,
        1.175e-7,
        1.399e-7,
        1.677e-7,
        2.003e-7,
        2.571e-7,
        3.325e-7,
        4.438e-7,
        6.255e-7,
        8.168e-7,
        1.036e-6,
        1.366e-6,
        1.855e-6,
        2.514e-6,
        3.240e-6,
        4.033e-6,
        4.854e-6,
        5.517e-6,
        6.089e-6,
        6.689e-6,
        1.106e-5,
        1.462e-5,
        1.321e-5,
        9.856e-6,
        5.960e-6,
        5.960e-6,
    ),
    dtype=jnp.float32,
)

_PPSUM = jnp.asarray(
    (
        955.890,
        850.532,
        754.599,
        667.742,
        589.841,
        519.421,
        455.480,
        398.085,
        347.171,
        301.735,
        261.310,
        225.360,
        193.419,
        165.490,
        141.032,
        120.125,
        102.689,
        87.829,
        75.123,
        64.306,
        55.086,
        47.209,
        40.535,
        34.795,
        29.865,
        19.122,
        9.277,
        4.660,
        2.421,
        1.294,
        0.647,
    ),
    dtype=jnp.float32,
)

_O3WIN = jnp.asarray(
    (
        4.629e-8,
        4.686e-8,
        5.017e-8,
        5.613e-8,
        6.871e-8,
        8.751e-8,
        1.138e-7,
        1.516e-7,
        2.161e-7,
        3.264e-7,
        4.968e-7,
        7.338e-7,
        1.017e-6,
        1.308e-6,
        1.625e-6,
        2.011e-6,
        2.516e-6,
        3.130e-6,
        3.840e-6,
        4.703e-6,
        5.486e-6,
        6.289e-6,
        6.993e-6,
        7.494e-6,
        8.197e-6,
        9.632e-6,
        1.113e-5,
        1.146e-5,
        9.389e-6,
        6.135e-6,
        6.135e-6,
    ),
    dtype=jnp.float32,
)

_PPWIN = jnp.asarray(
    (
        955.747,
        841.783,
        740.199,
        649.538,
        568.404,
        495.815,
        431.069,
        373.464,
        322.354,
        277.190,
        237.635,
        203.433,
        174.070,
        148.949,
        127.408,
        108.915,
        93.114,
        79.551,
        67.940,
        58.072,
        49.593,
        42.318,
        36.138,
        30.907,
        26.362,
        16.423,
        7.583,
        3.620,
        1.807,
        0.938,
        0.469,
    ),
    dtype=jnp.float32,
)


def _o3_work_profile() -> tuple[Any, Any]:
    ppann = _PPSUM
    interp = _O3WIN[:-1] + (_O3WIN[1:] - _O3WIN[:-1]) / (_PPWIN[1:] - _PPWIN[:-1]) * (
        _PPSUM[1:] - _PPWIN[:-1]
    )
    o3ann = jnp.concatenate(
        (
            jnp.asarray([0.5 * (_O3SUM[0] + _O3WIN[0])], dtype=jnp.float32),
            0.5 * (interp + _O3SUM[1:]),
        )
    )
    return o3ann, ppann


def o3data(plev: Any, *, kte: int | None = None) -> Any:
    """WRF ``module_ra_rrtmg_lw:O3DATA`` pressure-weighted ozone profile."""

    plev_arr = jnp.asarray(plev, dtype=jnp.float32)
    if kte is None:
        out_layers = int(plev_arr.shape[0]) - 1
    else:
        out_layers = int(kte) + 1
    plev_used = plev_arr[: out_layers + 1]
    o3wrk, ppwrk = _o3_work_profile()
    ppwrkh = jnp.concatenate(
        (
            jnp.asarray([1100.0], dtype=jnp.float32),
            0.5 * (ppwrk[1:] + ppwrk[:-1]),
            jnp.asarray([0.0], dtype=jnp.float32),
        )
    )
    p_top = plev_used[:-1, None]
    p_bottom = plev_used[1:, None]
    bin_top = ppwrkh[:-1][None, :]
    bin_bottom = ppwrkh[1:][None, :]
    pb1 = jnp.maximum(p_top - bin_top, 0.0)
    pb2 = jnp.maximum(p_top - bin_bottom, 0.0)
    pt1 = jnp.maximum(p_bottom - bin_top, 0.0)
    pt2 = jnp.maximum(p_bottom - bin_bottom, 0.0)
    overlap = lax.optimization_barrier(
        lax.optimization_barrier(
            lax.optimization_barrier(pb2 - pb1) - pt2
        )
        + pt1
    )
    terms = lax.optimization_barrier(overlap * o3wrk[None, :])

    def add_bin(total: Any, term: Any) -> tuple[Any, None]:
        updated = lax.optimization_barrier(
            jnp.asarray(total + term, dtype=jnp.float32)
        )
        return updated, None

    o3prof, _ = lax.scan(
        add_bin,
        jnp.zeros_like(plev_used[:-1]),
        jnp.moveaxis(terms, 1, 0),
    )
    pressure_depth = lax.optimization_barrier(
        plev_used[:-1] - plev_used[1:]
    )
    return lax.optimization_barrier(o3prof / pressure_depth)


def inirad(plev: Any, *, kte: int | None = None) -> Any:
    """WRF ``module_ra_rrtmg_lw:INIRAD`` ozone profile initialization."""

    return o3data(plev, kte=kte)


class RrtmgCloudRadii(NamedTuple):
    """Cloud-size inputs shared by WRF RRTMG longwave and shortwave paths."""

    inflg: int
    iceflg: int
    liqflg: int
    recloud: Any
    reice: Any
    resnow: Any
    qc: Any
    qi: Any
    qs: Any


class RrtmgCloudPathInputs(NamedTuple):
    """Cloud path and effective-radius inputs passed into RRTMG MCICA."""

    cldfrac: Any
    clwpth: Any
    ciwpth: Any
    cswpth: Any
    rel: Any
    rei: Any
    res: Any


class McicaStochasticClouds(NamedTuple):
    """MCICA stochastic subcolumn cloud fields."""

    cld_stoch: Any
    clwp_stoch: Any
    ciwp_stoch: Any
    cswp_stoch: Any
    tauc_stoch: Any


class McicaLwSubcolumns(NamedTuple):
    """Outputs of WRF ``mcica_subcol_lw`` for the supported overlap path."""

    cldfmcl: Any
    ciwpmcl: Any
    clwpmcl: Any
    cswpmcl: Any
    reicmcl: Any
    relqmcl: Any
    resnmcl: Any
    taucmcl: Any


class McicaSwSubcolumns(NamedTuple):
    """Outputs of WRF ``mcica_subcol_sw`` for the supported overlap path."""

    cldfmcl: Any
    ciwpmcl: Any
    clwpmcl: Any
    cswpmcl: Any
    reicmcl: Any
    relqmcl: Any
    resnmcl: Any
    taucmcl: Any
    ssacmcl: Any
    asmcmcl: Any
    fsfcmcl: Any


class RrtmgLwPadeTables(NamedTuple):
    """Longwave Pade lookup tables from WRF ``rrtmg_lw_ini``."""

    bpade: Any
    tau_tbl: Any
    exp_tbl: Any
    tfn_tbl: Any


class RrtmgSwPadeTables(NamedTuple):
    """Shortwave Pade lookup table from WRF ``rrtmg_sw_ini``."""

    bpade: Any
    tau_tbl: Any
    exp_tbl: Any


class RrtmgSpectralInit(NamedTuple):
    """Static band metadata and constants initialized by WRF RRTMG."""

    wavenum1: Any
    wavenum2: Any
    delwave: Any
    ng: Any
    nspa: Any
    nspb: Any
    ngc: Any
    ngs: Any
    ngm: Any
    ngn: Any
    ngb: Any
    wt: Any
    rwgt: Any
    heatfac: Any
    grav: Any
    planck: Any
    boltz: Any
    clight: Any
    avogad: Any
    alosmt: Any
    gascon: Any
    radcn1: Any
    radcn2: Any
    sbcnst: Any
    secdy: Any
    oneminus: Any
    pi: Any
    fluxfac: Any | None
    nxmol: Any | None
    ixindx: Any | None


_LW_GPOINT_COUNTS = (10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2)
_SW_GPOINT_COUNTS = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)
RRTMG_MG = 16
RRTMG_LW_NBND = 16
RRTMG_LW_NGPT = 140
RRTMG_SW_NBND = 14
RRTMG_SW_NGPT = 112
NGB_LW = jnp.asarray(
    tuple(band for band, count in enumerate(_LW_GPOINT_COUNTS, start=1) for _ in range(count)),
    dtype=jnp.int32,
)
NGB_SW_LOCAL = jnp.asarray(
    tuple(band for band, count in enumerate(_SW_GPOINT_COUNTS, start=1) for _ in range(count)),
    dtype=jnp.int32,
)
NGB_SW = jnp.asarray(
    tuple(band for band, count in enumerate(_SW_GPOINT_COUNTS, start=16) for _ in range(count)),
    dtype=jnp.int32,
)

_RRTMG_WT_VALUES = (
    0.1527534276,
    0.1491729617,
    0.1420961469,
    0.1316886544,
    0.1181945205,
    0.1019300893,
    0.0832767040,
    0.0626720116,
    0.0424925000,
    0.0046269894,
    0.0038279891,
    0.0030260086,
    0.0022199750,
    0.0014140010,
    0.0005330000,
    0.0000750000,
)

_LW_NG_VALUES = (16,) * RRTMG_LW_NBND
_LW_NSPA_VALUES = (1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9)
_LW_NSPB_VALUES = (1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0)
_LW_NGC_VALUES = _LW_GPOINT_COUNTS
_LW_NGS_VALUES = (10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138, 140)
_LW_NGM_VALUES = (
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
    1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8,
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
    1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
)
_LW_NGN_VALUES = (
    1, 1, 2, 2, 2, 2, 2, 2, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    2, 2, 2, 2, 4, 4,
    1, 1, 2, 2, 2, 2, 3, 3,
    1, 1, 1, 1, 2, 2, 4, 4,
    3, 3, 4, 6,
    8, 8,
    8, 8,
    4, 12,
)

_SW_NG_VALUES = (16,) * RRTMG_SW_NBND
_SW_NSPA_VALUES = (9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1)
_SW_NSPB_VALUES = (1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1)
_SW_NGC_VALUES = _SW_GPOINT_COUNTS
_SW_NGS_VALUES = (6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 112)
_SW_NGM_VALUES = (
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
    1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12,
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
    1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12,
)
_SW_NGN_VALUES = (
    2, 2, 2, 2, 4, 4,
    1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 1, 1, 1, 2, 2, 4, 4,
    1, 1, 1, 1, 2, 2, 4, 4,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 6,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 6,
    8, 8,
    2, 2, 1, 1, 1, 1, 1, 1, 2, 4,
    2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 2, 2, 4, 6,
    1, 1, 2, 2, 4, 6,
    1, 1, 1, 1, 1, 1, 4, 6,
    1, 1, 2, 2, 4, 6,
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
)


def _compute_rrtmg_rwgt(
    *,
    ng: tuple[int, ...],
    ngc: tuple[int, ...],
    ngm: tuple[int, ...],
    ngn: tuple[int, ...],
    wt: tuple[float, ...],
) -> Any:
    # WRF initializes ``wtsm`` and ``rwgt`` as default REAL arrays.  Keep
    # every loop-carried sum and division in binary32: evaluating this small
    # setup loop with Python floats first changes the compressed absorption
    # tables by up to two ulps before the first radiation call.
    wt_real = tuple(np.float32(value) for value in wt)
    rwgt = np.zeros(len(ng) * RRTMG_MG, dtype=np.float32)
    igcsm = 0
    for ibnd, compressed_count in enumerate(ngc):
        iprsm = 0
        if compressed_count < RRTMG_MG:
            wtsm = np.zeros(compressed_count, dtype=np.float32)
            for igc in range(compressed_count):
                original_count = ngn[igcsm]
                igcsm += 1
                wtsum = np.float32(0.0)
                for _ in range(original_count):
                    wtsum = np.add(wtsum, wt_real[iprsm], dtype=np.float32)
                    iprsm += 1
                wtsm[igc] = wtsum
            for ig in range(ng[ibnd]):
                ind = ibnd * RRTMG_MG + ig
                rwgt[ind] = np.divide(
                    wt_real[ig],
                    wtsm[ngm[ind] - 1],
                    dtype=np.float32,
                )
        else:
            for ig in range(ng[ibnd]):
                igcsm += 1
                ind = ibnd * RRTMG_MG + ig
                rwgt[ind] = np.float32(1.0)
    return jnp.asarray(rwgt, dtype=jnp.float32)


LW_RWGT = _compute_rrtmg_rwgt(
    ng=_LW_NG_VALUES,
    ngc=_LW_NGC_VALUES,
    ngm=_LW_NGM_VALUES,
    ngn=_LW_NGN_VALUES,
    wt=_RRTMG_WT_VALUES,
)
SW_RWGT = _compute_rrtmg_rwgt(
    ng=_SW_NG_VALUES,
    ngc=_SW_NGC_VALUES,
    ngm=_SW_NGM_VALUES,
    ngn=_SW_NGN_VALUES,
    wt=_RRTMG_WT_VALUES,
)


def _rrtmg_common_physical_constants(*, cpdair: float, lw: bool) -> dict[str, Any]:
    grav = jnp.asarray(9.8066, dtype=jnp.float32)
    secdy = jnp.asarray(8.6400e4, dtype=jnp.float32)
    pi = jnp.asarray(3.1415926535897932384626433832795028841971693993751, dtype=jnp.float32)
    fluxfac = pi * jnp.asarray(2.0e4, dtype=jnp.float32) if lw else None
    return {
        "heatfac": grav * secdy / (jnp.asarray(cpdair, dtype=jnp.float32) * jnp.asarray(1.0e2, dtype=jnp.float32)),
        "grav": grav,
        "planck": jnp.asarray(6.62606876e-27, dtype=jnp.float32),
        "boltz": jnp.asarray(1.3806503e-16, dtype=jnp.float32),
        "clight": jnp.asarray(2.99792458e10, dtype=jnp.float32),
        "avogad": jnp.asarray(6.02214199e23, dtype=jnp.float32),
        "alosmt": jnp.asarray(2.6867775e19, dtype=jnp.float32),
        "gascon": jnp.asarray(8.31447200e7, dtype=jnp.float32),
        "radcn1": jnp.asarray(1.191042722e-12 if lw else 1.191042772e-12, dtype=jnp.float32),
        "radcn2": jnp.asarray(1.4387752, dtype=jnp.float32),
        "sbcnst": jnp.asarray(5.670400e-4, dtype=jnp.float32),
        "secdy": secdy,
        "oneminus": jnp.asarray(1.0 - 1.0e-6, dtype=jnp.float32),
        "pi": pi,
        "fluxfac": fluxfac,
    }


def lwdatinit(cpdair: float = 1004.5) -> RrtmgSpectralInit:
    """WRF ``module_ra_rrtmg_lw:lwdatinit`` and ``lwcmbdat`` metadata."""

    const = _rrtmg_common_physical_constants(cpdair=cpdair, lw=True)
    return RrtmgSpectralInit(
        wavenum1=jnp.asarray((10.0, 350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0, 1180.0, 1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2380.0, 2600.0), dtype=jnp.float32),
        wavenum2=jnp.asarray((350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0, 1180.0, 1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2380.0, 2600.0, 3250.0), dtype=jnp.float32),
        delwave=jnp.asarray((340.0, 150.0, 130.0, 70.0, 120.0, 160.0, 100.0, 100.0, 210.0, 90.0, 320.0, 280.0, 170.0, 130.0, 220.0, 650.0), dtype=jnp.float32),
        ng=jnp.asarray(_LW_NG_VALUES, dtype=jnp.int32),
        nspa=jnp.asarray(_LW_NSPA_VALUES, dtype=jnp.int32),
        nspb=jnp.asarray(_LW_NSPB_VALUES, dtype=jnp.int32),
        ngc=jnp.asarray(_LW_NGC_VALUES, dtype=jnp.int32),
        ngs=jnp.asarray(_LW_NGS_VALUES, dtype=jnp.int32),
        ngm=jnp.asarray(_LW_NGM_VALUES, dtype=jnp.int32),
        ngn=jnp.asarray(_LW_NGN_VALUES, dtype=jnp.int32),
        ngb=NGB_LW,
        wt=jnp.asarray(_RRTMG_WT_VALUES, dtype=jnp.float32),
        rwgt=LW_RWGT,
        nxmol=jnp.asarray(4, dtype=jnp.int32),
        ixindx=jnp.asarray((1, 2, 3, 4) + (0,) * 34, dtype=jnp.int32),
        **const,
    )


def swdatinit(cpdair: float = 1004.5) -> RrtmgSpectralInit:
    """WRF ``module_ra_rrtmg_sw:swdatinit`` and ``swcmbdat`` metadata."""

    const = _rrtmg_common_physical_constants(cpdair=cpdair, lw=False)
    return RrtmgSpectralInit(
        wavenum1=jnp.asarray((2600.0, 3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0, 12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 820.0), dtype=jnp.float32),
        wavenum2=jnp.asarray((3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0, 12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 50000.0, 2600.0), dtype=jnp.float32),
        delwave=jnp.asarray((650.0, 750.0, 650.0, 500.0, 1000.0, 1550.0, 350.0, 4800.0, 3150.0, 6650.0, 6350.0, 9000.0, 12000.0, 1780.0), dtype=jnp.float32),
        ng=jnp.asarray(_SW_NG_VALUES, dtype=jnp.int32),
        nspa=jnp.asarray(_SW_NSPA_VALUES, dtype=jnp.int32),
        nspb=jnp.asarray(_SW_NSPB_VALUES, dtype=jnp.int32),
        ngc=jnp.asarray(_SW_NGC_VALUES, dtype=jnp.int32),
        ngs=jnp.asarray(_SW_NGS_VALUES, dtype=jnp.int32),
        ngm=jnp.asarray(_SW_NGM_VALUES, dtype=jnp.int32),
        ngn=jnp.asarray(_SW_NGN_VALUES, dtype=jnp.int32),
        ngb=NGB_SW,
        wt=jnp.asarray(_RRTMG_WT_VALUES, dtype=jnp.float32),
        rwgt=SW_RWGT,
        nxmol=None,
        ixindx=None,
        **const,
    )


def rrtmg_lw_spectral_init(cpdair: float = 1004.5) -> RrtmgSpectralInit:
    """Initialize longwave spectral metadata used by pure Python/JAX RRTMG kernels."""

    return lwdatinit(cpdair=cpdair)


def rrtmg_sw_spectral_init(cpdair: float = 1004.5) -> RrtmgSpectralInit:
    """Initialize shortwave spectral metadata used by pure Python/JAX RRTMG kernels."""

    return swdatinit(cpdair=cpdair)



def _retab_ice_radius_for_cloud(temp: Any) -> Any:
    temp_arr = jnp.asarray(temp, dtype=jnp.float32)
    temp_int = temp_arr.astype(jnp.int32)
    index = jnp.clip((temp_arr - 179.0).astype(jnp.int32), 1, 75)
    corr = temp_arr - temp_int.astype(temp_arr.dtype)
    return RETAB[index - 1] * (1.0 - corr) + RETAB[index] * corr


def prepare_cloud_radii(
    *,
    t3d: Any,
    cldfra3d: Any,
    xland: float,
    qc3d: Any,
    qi3d: Any,
    qs3d: Any,
    re_cloud: Any,
    re_ice: Any,
    re_snow: Any,
    has_reqc: int,
    has_reqi: int,
    has_reqs: int,
    icloud: int = 1,
) -> RrtmgCloudRadii:
    """Prepare WRF RRTMG cloud particle radii for one vertical column."""

    t = jnp.asarray(t3d, dtype=jnp.float32)
    cldfra = jnp.asarray(cldfra3d, dtype=jnp.float32)
    qc = jnp.asarray(qc3d, dtype=jnp.float32)
    qi = jnp.asarray(qi3d, dtype=jnp.float32)
    qs = jnp.asarray(qs3d, dtype=jnp.float32)
    xland_arr = jnp.asarray(xland, dtype=jnp.float32)
    re_cloud_arr = jnp.asarray(re_cloud, dtype=jnp.float32)
    re_ice_arr = jnp.asarray(re_ice, dtype=jnp.float32)
    re_snow_arr = jnp.asarray(re_snow, dtype=jnp.float32)

    inflg = 2
    iceflg = 3
    liqflg = 1
    recloud = jnp.full_like(t, 5.0)
    reice = jnp.full_like(t, 10.0)
    resnow = jnp.full_like(t, 10.0)

    if int(icloud) != 0:
        if int(has_reqc) != 0:
            inflg = 3
            recloud = jnp.maximum(2.5, re_cloud_arr * 1.0e6)
            cloudy_min = (recloud <= 2.5) & (cldfra > 0.0)
            recloud = jnp.where(cloudy_min & (xland_arr - 1.5 > 0.0), 10.5, recloud)
            recloud = jnp.where(cloudy_min & (xland_arr - 1.5 < 0.0), 7.5, recloud)

        if int(has_reqi) != 0:
            inflg = 4
            iceflg = 4
            reice = jnp.maximum(5.0, re_ice_arr * 1.0e6)
            fallback = (reice <= 5.0) & (cldfra > 0.0)
            reice = jnp.where(fallback, jnp.maximum(_retab_ice_radius_for_cloud(t), 5.0), reice)

        if int(has_reqs) != 0:
            inflg = 5
            iceflg = 5
            resnow = jnp.maximum(10.0, re_snow_arr * 1.0e6)

        if int(has_reqs) == 0 and int(has_reqi) != 0 and int(has_reqc) != 0:
            inflg = 5
            iceflg = 5
            resnow = jnp.maximum(10.0, re_ice_arr * 1.0e6)
            qs = qi
            qi = jnp.zeros_like(qi)
            reice = jnp.full_like(reice, 10.0)

    return RrtmgCloudRadii(
        inflg=inflg,
        iceflg=iceflg,
        liqflg=liqflg,
        recloud=recloud,
        reice=reice,
        resnow=resnow,
        qc=qc,
        qi=qi,
        qs=qs,
    )


def prepare_cloud_path_inputs(
    *,
    tlay: Any,
    pdel: Any,
    cldfra: Any,
    qc: Any,
    qi: Any,
    qs: Any,
    xland: float,
    xice: float,
    snow: float,
    recloud: Any,
    reice: Any,
    resnow: Any,
    inflg: int,
    iceflg: int,
    g: float,
) -> RrtmgCloudPathInputs:
    """Prepare WRF RRTMG cloud path and radius inputs for one column."""

    temp = jnp.asarray(tlay, dtype=jnp.float32)
    pdel_arr = jnp.asarray(pdel, dtype=jnp.float32)
    cldfrac = jnp.asarray(cldfra, dtype=jnp.float32)
    qc_arr = jnp.asarray(qc, dtype=jnp.float32)
    qi_arr = jnp.asarray(qi, dtype=jnp.float32)
    qs_arr = jnp.asarray(qs, dtype=jnp.float32)
    denom = jnp.maximum(0.01, cldfrac)
    gravmks = jnp.asarray(g, dtype=jnp.float32)

    gicewp = (qi_arr + qs_arr) * pdel_arr * 100.0 / gravmks * 1000.0
    gliqwp = qc_arr * pdel_arr * 100.0 / gravmks * 1000.0
    cicewp = gicewp / denom
    cliqwp = gliqwp / denom

    if int(iceflg) >= 4:
        gicewp = qi_arr * pdel_arr * 100.0 / gravmks * 1000.0
        cicewp = gicewp / denom

    csnowp = jnp.zeros_like(temp)
    res_adj = jnp.asarray(resnow, dtype=jnp.float32)
    if int(iceflg) == 5:
        snow_mass_factor = jnp.full_like(temp, 0.99)
        over_limit = res_adj > 130.0
        limited_factor = jnp.minimum(snow_mass_factor, (130.0 / res_adj) * (130.0 / res_adj))
        snow_mass_factor = jnp.where(over_limit, limited_factor, snow_mass_factor)
        res_adj = jnp.where(over_limit, 130.0, res_adj)
        gsnowp = qs_arr * snow_mass_factor * pdel_arr * 100.0 / gravmks * 1000.0
        csnowp = gsnowp / denom

    pver = int(temp.shape[0])
    xland_arr = jnp.asarray(xland, dtype=jnp.float32)
    xice_arr = jnp.asarray(xice, dtype=jnp.float32)
    snow_arr = jnp.asarray(snow, dtype=jnp.float32)
    landfrac = jnp.reshape(2.0 - xland_arr, (1,))
    landm = landfrac
    icefrac = jnp.reshape(xice_arr, (1,))
    snowh = jnp.reshape(0.001 * snow_arr, (1,))
    rel = relcalc(ncol=1, pcols=1, pver=pver, t=temp[None, :], landfrac=landfrac, landm=landm, icefrac=icefrac, snowh=snowh)[0]
    rei = reicalc(ncol=1, pcols=1, pver=pver, t=temp[None, :])[0]

    if int(inflg) >= 3:
        rel = jnp.asarray(recloud, dtype=jnp.float32)
    if int(iceflg) >= 4:
        rei = jnp.asarray(reice, dtype=jnp.float32)
    if int(iceflg) == 3:
        rei = jnp.minimum(140.0, rei * 1.0315)

    clwpth = cliqwp
    ciwpth = cicewp
    if int(inflg) == 5:
        cswpth = csnowp
        res = res_adj
    else:
        cswpth = jnp.zeros_like(temp)
        res = jnp.full_like(temp, 10.0)

    return RrtmgCloudPathInputs(
        cldfrac=cldfrac,
        clwpth=clwpth,
        ciwpth=ciwpth,
        cswpth=cswpth,
        rel=rel,
        rei=rei,
        res=res,
    )


def _u32(value: Any) -> Any:
    return jnp.asarray(value, dtype=jnp.uint32)


def _mwc_xor_shift(value: Any, shift: int) -> Any:
    if shift >= 0:
        shifted = jnp.left_shift(value, jnp.uint32(shift))
    else:
        shifted = jnp.right_shift(value, jnp.uint32(-shift))
    return jnp.bitwise_xor(value, shifted)


def kissvec(seed1: Any, seed2: Any, seed3: Any, seed4: Any) -> tuple[Any, Any, Any, Any, Any]:
    """WRF RRTMG MCICA ``kissvec`` random-number update."""

    s1 = _u32(seed1)
    s2 = _u32(seed2)
    s3 = _u32(seed3)
    s4 = _u32(seed4)

    s1 = jnp.uint32(69069) * s1 + jnp.uint32(1327217885)
    s2 = _mwc_xor_shift(_mwc_xor_shift(_mwc_xor_shift(s2, 13), -17), 5)
    s3 = jnp.uint32(18000) * jnp.bitwise_and(s3, jnp.uint32(65535)) + jnp.right_shift(s3, jnp.uint32(16))
    s4 = jnp.uint32(30903) * jnp.bitwise_and(s4, jnp.uint32(65535)) + jnp.right_shift(s4, jnp.uint32(16))
    kiss = s1 + s2 + jnp.left_shift(s3, jnp.uint32(16)) + s4
    kiss_i32 = lax.bitcast_convert_type(kiss, jnp.int32)
    ran_arr = kiss_i32.astype(jnp.float32) * jnp.float32(2.328306e-10) + jnp.float32(0.5)
    return (
        lax.bitcast_convert_type(s1, jnp.int32),
        lax.bitcast_convert_type(s2, jnp.int32),
        lax.bitcast_convert_type(s3, jnp.int32),
        lax.bitcast_convert_type(s4, jnp.int32),
        ran_arr,
    )


def _initial_kiss_seeds(pmid: Any) -> tuple[Any, Any, Any, Any]:
    pmid_arr = jnp.asarray(pmid, dtype=jnp.float32)
    fractions = pmid_arr[:, :4] - jnp.trunc(pmid_arr[:, :4])
    seeds = (fractions * jnp.float32(1000000000.0)).astype(jnp.int32)
    return seeds[:, 0], seeds[:, 1], seeds[:, 2], seeds[:, 3]


@lru_cache(maxsize=16)
def _kiss_jump_coefficients(offsets: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Integer jump operators for the original four WRF KISS recurrences."""
    multiplier, increment = 1, 0
    basis = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    affine, xorshift = [], []
    current = 0
    for offset in offsets:
        while current < offset:
            multiplier = (69069 * multiplier) & 0xFFFFFFFF
            increment = (69069 * increment + 1327217885) & 0xFFFFFFFF
            basis ^= basis << np.uint32(13)
            basis ^= basis >> np.uint32(17)
            basis ^= basis << np.uint32(5)
            current += 1
        affine.append((multiplier, increment))
        xorshift.append(basis.copy())
    mwc = [
        [pow(a, max(0, offset - 2), a * 65536 - 1) for offset in offsets]
        for a in (18000, 30903)
    ]
    arrays = (np.asarray(affine, dtype=np.uint32), np.asarray(xorshift, dtype=np.uint32),
              np.asarray(mwc, dtype=np.uint32))
    for array in arrays:
        array.flags.writeable = False
    return arrays


def _kiss_advance_seeds(seeds: tuple[Any, ...], offsets: tuple[int, ...]) -> tuple[Any, ...]:
    """Advance each seed to each offset without changing the random stream.

    LCG and xorshift jumps use exact uint32 arithmetic. For MWC, two ordinary
    steps put every uint32 input in [0, a*65536-1]. Further steps are modular
    multiplication by a. The modulus itself, like zero, is a fixed point and
    must remain distinct from zero. uint64 products fit below 2**62.
    """
    affine, matrix, mwc = _kiss_jump_coefficients(offsets)
    s1, s2, s3, s4 = map(_u32, seeds)
    shape = (len(offsets),) + (1,) * s1.ndim
    jumped1 = jnp.asarray(affine[:, 0]).reshape(shape) * s1 + jnp.asarray(affine[:, 1]).reshape(shape)
    bit_shape = (1, 32) + (1,) * s2.ndim
    bits = (s2[None, None, ...] >> jnp.arange(32, dtype=jnp.uint32).reshape(bit_shape)) & jnp.uint32(1)
    terms = jnp.where(bits != 0, jnp.asarray(matrix).reshape((len(offsets), 32) + (1,) * s2.ndim), jnp.uint32(0))
    jumped2 = jnp.bitwise_xor.reduce(terms, axis=1)
    jump_indices = jnp.asarray(offsets).reshape(shape)

    def advance_mwc(seed: Any, a: int, powers: np.ndarray) -> Any:
        def step(value: Any) -> Any:
            return jnp.uint32(a) * (value & jnp.uint32(65535)) + (value >> jnp.uint32(16))
        once = step(seed)
        twice = step(once)
        modulus = jnp.uint64(a * 65536 - 1)
        product = twice.astype(jnp.uint64) * jnp.asarray(powers, dtype=jnp.uint64).reshape(shape)
        advanced = (product % modulus).astype(jnp.uint32)
        advanced = jnp.where(twice == jnp.uint32(a * 65536 - 1), twice, advanced)
        return jnp.where(jump_indices == 0, seed, jnp.where(jump_indices == 1, once, advanced))

    return jumped1, jumped2, advance_mwc(s3, 18000, mwc[0]), advance_mwc(s4, 30903, mwc[1])


def _kiss_random_sequence(seeds: tuple[Any, ...], count: int, skip: int) -> Any:
    """Generate the unchanged WRF stream in parallel blocks of 64 draws.

    Blocks start at exact jumped seeds, then use the existing kissvec update.
    Only integer state is jumped; floating random values and their ordering
    are unchanged. Padded draws in the last block are discarded; MCICA does
    not retain the final seeds between calls.
    """
    if count == 0:
        return jnp.empty((0,) + jnp.shape(seeds[0]), dtype=jnp.float32)
    length = min(64, count)
    blocks = (count + length - 1) // length
    offsets = tuple(max(0, skip) + block * length for block in range(blocks))
    starts = tuple(lax.bitcast_convert_type(value, jnp.int32)
                   for value in _kiss_advance_seeds(seeds, offsets))

    def step(current: tuple[Any, ...], _index: Any, _parameter: Any) -> tuple[tuple[Any, ...], Any]:
        s1, s2, s3, s4, random = kissvec(*current)
        return (s1, s2, s3, s4), random

    _last, by_draw = scan_columns(step, starts, (jnp.arange(length, dtype=jnp.int32),))
    by_block = jnp.swapaxes(by_draw, 0, 1)
    return by_block.reshape((blocks * length,) + jnp.shape(seeds[0]))[:count]


def generate_stochastic_clouds_overlap2(
    *,
    pmid: Any,
    cld: Any,
    clwp: Any,
    ciwp: Any,
    cswp: Any,
    tauc: Any,
    change_seed: int,
    ngb: Any,
) -> McicaStochasticClouds:
    """WRF MCICA ``generate_stochastic_clouds`` for overlap case 2, ``irng=0``."""

    pmid_arr = jnp.asarray(pmid, dtype=jnp.float32)
    cld_raw = jnp.asarray(cld, dtype=jnp.float32)
    cld_arr = jnp.where(cld_raw < 1.0e-20, 0.0, cld_raw)
    clwp_arr = jnp.asarray(clwp, dtype=jnp.float32)
    ciwp_arr = jnp.asarray(ciwp, dtype=jnp.float32)
    cswp_arr = jnp.asarray(cswp, dtype=jnp.float32)
    tauc_arr = jnp.asarray(tauc, dtype=jnp.float32)
    ngb_idx = jnp.asarray(ngb, dtype=jnp.int32) - 1
    nsubcol = int(ngb_idx.shape[0])
    ncol = int(pmid_arr.shape[0])
    nlay = int(pmid_arr.shape[1])

    seeds = _initial_kiss_seeds(pmid_arr)
    rand_flat = _kiss_random_sequence(seeds, nsubcol * nlay, int(change_seed))
    cdf = jnp.reshape(rand_flat, (nsubcol, nlay, ncol)).transpose(0, 2, 1)

    if nlay > 1:
        def overlap_step(previous: Any, inputs: tuple[Any, Any], thresholds: Any) -> tuple[Any, Any]:
            previous_cdf, column_index = previous
            current_cdf, level = inputs
            previous_threshold = thresholds[level * ncol + column_index]
            adjusted = jnp.where(
                previous_cdf > previous_threshold,
                previous_cdf,
                current_cdf * previous_threshold,
            )
            return (adjusted, column_index), adjusted

        current_by_level = jnp.moveaxis(cdf[:, :, 1:], 2, 0)
        threshold_by_level = (1.0 - cld_arr[:, :-1]).T
        # Share each atmospheric threshold across subcolumns without copying it.
        column_index = jnp.broadcast_to(jnp.arange(ncol, dtype=jnp.int32), (nsubcol, ncol))
        _last, adjusted_by_level = scan_columns(
            overlap_step,
            (cdf[:, :, 0], column_index),
            (current_by_level, jnp.arange(nlay - 1, dtype=jnp.int32)),
            threshold_by_level.reshape(-1),
        )
        cdf = jnp.concatenate((cdf[:, :, :1], jnp.moveaxis(adjusted_by_level, 0, 2)), axis=2)

    iscloudy = cdf >= (1.0 - cld_arr[None, :, :])
    tauc_by_subcol = jnp.take(tauc_arr, ngb_idx, axis=0)
    return McicaStochasticClouds(
        cld_stoch=iscloudy.astype(jnp.float32),
        clwp_stoch=jnp.where(iscloudy, clwp_arr[None, :, :], 0.0),
        ciwp_stoch=jnp.where(iscloudy, ciwp_arr[None, :, :], 0.0),
        cswp_stoch=jnp.where(iscloudy, cswp_arr[None, :, :], 0.0),
        tauc_stoch=jnp.where(iscloudy, tauc_by_subcol, 0.0),
    )


def mcica_subcol_lw_overlap2(
    *,
    play: Any,
    cldfrac: Any,
    ciwp: Any,
    clwp: Any,
    cswp: Any,
    rei: Any,
    rel: Any,
    res: Any,
    tauc: Any,
    change_seed: int = 150,
    ngb: Any = NGB_LW,
) -> McicaLwSubcolumns:
    """WRF ``mcica_subcol_lw`` for default maximum-random overlap."""

    sampled = generate_stochastic_clouds_overlap2(
        pmid=jnp.asarray(play, dtype=jnp.float32) * 100.0,
        cld=cldfrac,
        clwp=clwp,
        ciwp=ciwp,
        cswp=cswp,
        tauc=tauc,
        change_seed=change_seed,
        ngb=ngb,
    )
    return McicaLwSubcolumns(
        cldfmcl=sampled.cld_stoch,
        ciwpmcl=sampled.ciwp_stoch,
        clwpmcl=sampled.clwp_stoch,
        cswpmcl=sampled.cswp_stoch,
        reicmcl=jnp.asarray(rei, dtype=jnp.float32),
        relqmcl=jnp.asarray(rel, dtype=jnp.float32),
        resnmcl=jnp.asarray(res, dtype=jnp.float32),
        taucmcl=sampled.tauc_stoch,
    )


def mcica_subcol_sw_overlap2(
    *,
    play: Any,
    cldfrac: Any,
    ciwp: Any,
    clwp: Any,
    cswp: Any,
    rei: Any,
    rel: Any,
    res: Any,
    tauc: Any,
    ssac: Any,
    asmc: Any,
    fsfc: Any,
    change_seed: int = 1,
    ngb: Any = NGB_SW_LOCAL,
) -> McicaSwSubcolumns:
    """WRF ``mcica_subcol_sw`` for default maximum-random overlap."""

    sampled = generate_stochastic_clouds_overlap2(
        pmid=jnp.asarray(play, dtype=jnp.float32) * 100.0,
        cld=cldfrac,
        clwp=clwp,
        ciwp=ciwp,
        cswp=cswp,
        tauc=tauc,
        change_seed=change_seed,
        ngb=ngb,
    )
    iscloudy = sampled.cld_stoch > 0.0
    ngb_idx = jnp.asarray(ngb, dtype=jnp.int32) - 1
    ssac_by_subcol = jnp.take(jnp.asarray(ssac, dtype=jnp.float32), ngb_idx, axis=0)
    asmc_by_subcol = jnp.take(jnp.asarray(asmc, dtype=jnp.float32), ngb_idx, axis=0)
    fsfc_by_subcol = jnp.take(jnp.asarray(fsfc, dtype=jnp.float32), ngb_idx, axis=0)
    return McicaSwSubcolumns(
        cldfmcl=sampled.cld_stoch,
        ciwpmcl=sampled.ciwp_stoch,
        clwpmcl=sampled.clwp_stoch,
        cswpmcl=sampled.cswp_stoch,
        reicmcl=jnp.asarray(rei, dtype=jnp.float32),
        relqmcl=jnp.asarray(rel, dtype=jnp.float32),
        resnmcl=jnp.asarray(res, dtype=jnp.float32),
        taucmcl=sampled.tauc_stoch,
        ssacmcl=jnp.where(iscloudy, ssac_by_subcol, 1.0),
        asmcmcl=jnp.where(iscloudy, asmc_by_subcol, 0.0),
        fsfcmcl=jnp.where(iscloudy, fsfc_by_subcol, 0.0),
    )


def rrtmg_lw_nlayers(*, p_top: float, kme: int, deltap: float = 4.0) -> int:
    """WRF ``rrtmg_lwinit`` extended-layer count."""

    layers_above = float(p_top) * 0.01 / float(deltap)
    return int(kme) + int(layers_above + 0.5) - 1


@lru_cache(maxsize=1)
def _load_default_rrtmg_lw_pade_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table_path = Path(__file__).with_name("data") / "rrtmg_lw_pade_tables.npz"
    with np.load(table_path) as data:
        return (
            np.asarray(data["bpade"], dtype=np.float32),
            np.asarray(data["tau_tbl"], dtype=np.float32),
            np.asarray(data["exp_tbl"], dtype=np.float32),
            np.asarray(data["tfn_tbl"], dtype=np.float32),
        )


def rrtmg_lw_pade_tables(*, ntbl: int = 10000, pade: float = 0.278) -> RrtmgLwPadeTables:
    """Load the exact default WRF longwave Pade transmittance tables.

    WRF initializes these arrays with single-precision Fortran arithmetic and
    ``EXP``. JAX reconstruction changes both ``bpade`` and thousands of table
    entries, so the production defaults are stored as immutable model data.
    """

    ntbl_i = int(ntbl)
    if ntbl_i == 10000 and float(pade) == 0.278:
        bpade_data, tau_tbl_data, exp_tbl_data, tfn_tbl_data = _load_default_rrtmg_lw_pade_table()
        return RrtmgLwPadeTables(
            bpade=jnp.asarray(bpade_data, dtype=jnp.float32),
            tau_tbl=jnp.asarray(tau_tbl_data, dtype=jnp.float32),
            exp_tbl=jnp.asarray(exp_tbl_data, dtype=jnp.float32),
            tfn_tbl=jnp.asarray(tfn_tbl_data, dtype=jnp.float32),
        )

    expeps = jnp.asarray(1.0e-20, dtype=jnp.float32)
    bpade = jnp.asarray(1.0, dtype=jnp.float32) / jnp.asarray(pade, dtype=jnp.float32)
    idx = jnp.arange(ntbl_i + 1, dtype=jnp.float32)
    tfn = idx / jnp.asarray(float(ntbl_i), dtype=jnp.float32)
    tau_tbl = bpade * tfn / (1.0 - tfn)
    exp_tbl = jnp.exp(-tau_tbl)
    tfn_tbl = jnp.where(tau_tbl < 0.06, tau_tbl / 6.0, 1.0 - 2.0 * ((1.0 / tau_tbl) - (exp_tbl / (1.0 - exp_tbl))))
    tau_tbl = tau_tbl.at[0].set(0.0).at[ntbl_i].set(1.0e10)
    exp_tbl = jnp.maximum(exp_tbl, expeps).at[0].set(1.0).at[ntbl_i].set(expeps)
    tfn_tbl = tfn_tbl.at[0].set(0.0).at[ntbl_i].set(1.0)
    return RrtmgLwPadeTables(bpade=bpade, tau_tbl=tau_tbl, exp_tbl=exp_tbl, tfn_tbl=tfn_tbl)


@lru_cache(maxsize=1)
def _load_default_rrtmg_sw_pade_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_path = Path(__file__).with_name("data") / "rrtmg_sw_pade_tables.npz"
    with np.load(table_path) as data:
        return (
            np.asarray(data["bpade"], dtype=np.float32),
            np.asarray(data["tau_tbl"], dtype=np.float32),
            np.asarray(data["exp_tbl"], dtype=np.float32),
        )


def rrtmg_sw_pade_tables(*, ntbl: int = 10000, pade: float = 0.278) -> RrtmgSwPadeTables:
    """Load the exact WRF shortwave Pade transmittance lookup table.

    WRF initializes the default table with single-precision Fortran ``EXP``.
    Reconstructing it with JAX changes thousands of entries by one ULP, so the
    production default is stored as immutable model data.  Non-default table
    dimensions retain a generated path for diagnostic use.
    """

    ntbl_i = int(ntbl)
    if ntbl_i == 10000 and float(pade) == 0.278:
        bpade_data, tau_tbl_data, exp_tbl_data = _load_default_rrtmg_sw_pade_table()
        return RrtmgSwPadeTables(
            bpade=jnp.asarray(bpade_data, dtype=jnp.float32),
            tau_tbl=jnp.asarray(tau_tbl_data, dtype=jnp.float32),
            exp_tbl=jnp.asarray(exp_tbl_data, dtype=jnp.float32),
        )

    expeps = jnp.asarray(1.0e-20, dtype=jnp.float32)
    bpade = jnp.asarray(1.0, dtype=jnp.float32) / jnp.asarray(pade, dtype=jnp.float32)
    idx = jnp.arange(ntbl_i + 1, dtype=jnp.float32)
    tfn = idx / jnp.asarray(float(ntbl_i), dtype=jnp.float32)
    tau_values = bpade * tfn / (1.0 - tfn)
    exp_tbl = jnp.maximum(jnp.exp(-tau_values), expeps).at[0].set(1.0).at[ntbl_i].set(expeps)
    tau_tbl = tau_values[ntbl_i - 1]
    return RrtmgSwPadeTables(bpade=bpade, tau_tbl=tau_tbl, exp_tbl=exp_tbl)


def relcalc(
    *,
    ncol: int,
    pcols: int,
    pver: int,
    t: Any,
    landfrac: Any,
    landm: Any,
    icefrac: Any,
    snowh: Any,
) -> Any:
    """WRF ``module_ra_rrtmg_lw:relcalc`` liquid effective radius."""

    del pcols
    ncol_i = int(ncol)
    pver_i = int(pver)
    temp = jnp.asarray(t, dtype=jnp.float32)[:ncol_i, :pver_i]
    landm_arr = jnp.asarray(landm, dtype=jnp.float32)[:ncol_i, None]
    icefrac_arr = jnp.asarray(icefrac, dtype=jnp.float32)[:ncol_i, None]
    snowh_arr = jnp.asarray(snowh, dtype=jnp.float32)[:ncol_i, None]
    del landfrac

    tmelt = jnp.asarray(273.16, dtype=jnp.float32)
    rliqocean = jnp.asarray(14.0, dtype=jnp.float32)
    rliqice = jnp.asarray(14.0, dtype=jnp.float32)
    rliqland = jnp.asarray(8.0, dtype=jnp.float32)

    rel = rliqland + (rliqocean - rliqland) * jnp.minimum(
        1.0, jnp.maximum(0.0, (tmelt - temp) * 0.05)
    )
    rel = rel + (rliqocean - rel) * jnp.minimum(1.0, jnp.maximum(0.0, snowh_arr * 10.0))
    rel = rel + (rliqocean - rel) * jnp.minimum(1.0, jnp.maximum(0.0, 1.0 - landm_arr))
    rel = rel + (rliqice - rel) * jnp.minimum(1.0, jnp.maximum(0.0, icefrac_arr))
    return rel


def reicalc(*, ncol: int, pcols: int, pver: int, t: Any) -> Any:
    """WRF ``module_ra_rrtmg_lw:reicalc`` ice effective radius."""

    del pcols
    ncol_i = int(ncol)
    pver_i = int(pver)
    temp = jnp.asarray(t, dtype=jnp.float32)[:ncol_i, :pver_i]
    temp_int = temp.astype(jnp.int32)
    index = jnp.clip((temp - 179.0).astype(jnp.int32), 1, 94)
    corr = temp - temp_int.astype(temp.dtype)
    lower = RETAB[index - 1]
    upper = RETAB[index]
    return lower * (1.0 - corr) + upper * corr


__all__ = [
    "McicaLwSubcolumns",
    "McicaStochasticClouds",
    "McicaSwSubcolumns",
    "NGB_LW",
    "NGB_SW",
    "NGB_SW_LOCAL",
    "RETAB",
    "RRTMG_LW_NBND",
    "RRTMG_LW_NGPT",
    "RRTMG_MG",
    "RRTMG_SW_NBND",
    "RRTMG_SW_NGPT",
    "RrtmgCloudPathInputs",
    "RrtmgCloudRadii",
    "RrtmgLwPadeTables",
    "RrtmgSpectralInit",
    "RrtmgSwPadeTables",
    "lwdatinit",
    "generate_stochastic_clouds_overlap2",
    "inirad",
    "kissvec",
    "mcica_subcol_lw_overlap2",
    "mcica_subcol_sw_overlap2",
    "o3data",
    "prepare_cloud_path_inputs",
    "prepare_cloud_radii",
    "relcalc",
    "reicalc",
    "rrtmg_lw_spectral_init",
    "rrtmg_lw_nlayers",
    "rrtmg_lw_pade_tables",
    "rrtmg_sw_spectral_init",
    "rrtmg_sw_pade_tables",
    "swdatinit",
]
