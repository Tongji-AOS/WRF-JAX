"""Shortwave RRTMG translations from WRF ``module_ra_rrtmg_sw``."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.column_scan import scan_columns, scan_vmapped_columns
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.real_arithmetic import column_barrier, column_binary, real_assignment, real_divide, real_multiply
from wrf_jax.core.wrf_math import (
    sqrt_for_wrf,
    wrf_glibc_expf_finite,
    wrf_glibc_logf_positive_normal,
)
from wrf_jax.physics.module_ra_rrtmg_common import (
    NGB_SW_LOCAL,
    RRTMG_SW_NBND,
    RRTMG_SW_NGPT,
    RrtmgSpectralInit,
    RrtmgSwPadeTables,
    SW_RWGT,
    _SW_NGC_VALUES,
    _SW_NGN_VALUES,
    _SW_NGS_VALUES,
    inirad,
    mcica_subcol_sw_overlap2,
    pade_transmittance,
    prepare_cloud_path_inputs,
    prepare_cloud_radii,
    rrtmg_sw_pade_tables,
    swdatinit,
    wrf_real_pa_to_hpa as _wrf_real_pa_to_hpa,
)
from wrf_jax.runtime_config import configured_path


_PROFILE_RRTMG_SW_PATH = os.environ.get("WRF_PY_PROFILE_RRTMG_SW", "").strip()
_CAPTURE_RRTMG_SW_TARGET_PATH = os.environ.get("WRF_PY_CAPTURE_RRTMG_SW_TARGET", "").strip()
_CAPTURE_RRTMG_SW_TARGET_COSZEN = np.float32(
    os.environ.get("WRF_PY_CAPTURE_RRTMG_SW_TARGET_COSZEN", "0.3232190012931824")
)
_CAPTURE_RRTMG_SW_TRANSPORT_PATH = os.environ.get(
    "WRF_PY_CAPTURE_RRTMG_SW_TRANSPORT", ""
).strip()


def _wrf_real_store(value: Any, rounding_token: Any | None = None) -> Any:
    """Materialize one WRF default-REAL assignment inside an XLA graph."""

    array = jnp.asarray(value, dtype=jnp.float32)
    if rounding_token is not None:
        return real_assignment(array, rounding_token)
    return jax.lax.optimization_barrier(array)


def _wrf_real_add(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64) + jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_sub(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64) - jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_mul(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64) * jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_div(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    # Hide constant denominators behind assignment barriers so XLA does not
    # replace WRF's binary32 divide with reciprocal multiplication.
    return _wrf_real_store(
        real_divide(_wrf_real_store(left, rounding_token), _wrf_real_store(right, rounding_token)),
        rounding_token,
    )


def _wrf_real_linear_combination(
    coefficients: tuple[Any, ...],
    values: tuple[Any, ...],
    rounding_token: Any | None = None,
) -> Any:
    """Evaluate a WRF REAL sum of products in source order."""

    if len(coefficients) != len(values) or not coefficients:
        raise ValueError("WRF linear combination requires equally sized non-empty inputs")
    total = _wrf_real_mul(coefficients[0], values[0], rounding_token)
    for coefficient, value in zip(coefficients[1:], values[1:], strict=True):
        total = _wrf_real_add(
            total,
            _wrf_real_mul(coefficient, value, rounding_token),
            rounding_token,
        )
    return total


def _wrf_inatm_coldry(
    pdp: Any,
    h2o: Any,
    amm: Any,
    rounding_token: Any | None = None,
) -> Any:
    """Evaluate WRF ``inatm_sw`` dry-column arithmetic in compiler order."""

    avogad = jnp.asarray(6.02214199e23, dtype=jnp.float32)
    grav = jnp.asarray(9.8066, dtype=jnp.float32)
    numerator = _wrf_real_mul(
        _wrf_real_mul(
            pdp,
            jnp.asarray(1.0e3, dtype=jnp.float32),
            rounding_token,
        ),
        avogad,
        rounding_token,
    )
    denominator = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(
                jnp.asarray(1.0e2, dtype=jnp.float32),
                grav,
                rounding_token,
            ),
            amm,
            rounding_token,
        ),
        _wrf_real_add(
            jnp.asarray(1.0, dtype=jnp.float32),
            h2o,
            rounding_token,
        ),
        rounding_token,
    )
    return _wrf_real_div(numerator, denominator, rounding_token)


def _wrf_ordered_weighted_sum(values: Any, weights: Any) -> Any:
    """Accumulate spectral fluxes in WRF's increasing g-point order."""

    products = _wrf_real_store(
        jnp.asarray(values, dtype=jnp.float32) * jnp.asarray(weights, dtype=jnp.float32)
    )

    def add_one(total: Any, inputs: Any, _parameter: Any) -> tuple[Any, None]:
        return column_barrier(column_binary("add", total, inputs[0])), None

    total, _ = scan_vmapped_columns(
        add_one,
        jnp.zeros(products.shape[:-1], dtype=jnp.float32),
        (jnp.moveaxis(products, -1, 0),),
    )
    return total


def _wrf_reversed_flux_sum(values: Any, weights: Any) -> Any:
    """Sum top-down spectral fluxes, then return the bottom-up level order.

    The flux weights are independent of level. Reversing after the ordered
    sum avoids copying the much larger level-by-g-point arrays.
    """
    return _wrf_ordered_weighted_sum(values, weights)[..., ::-1]


def _wrf_ordered_cumprod(
    values: Any,
    *,
    axis: int,
    rounding_token: Any | None = None,
) -> Any:
    """Accumulate a WRF layer product in source-loop order."""

    array = jnp.asarray(values, dtype=jnp.float32)
    values_first = jnp.moveaxis(array, axis, 0)

    def multiply_one(total: Any, inputs: Any, token: Any) -> tuple[Any, Any]:
        if token is None:
            updated = column_barrier(column_binary("mul", total, inputs[0]))
        else:
            updated = real_multiply(total, inputs[0], token)
        return updated, updated

    _final, history = scan_vmapped_columns(
        multiply_one,
        jnp.ones(values_first.shape[1:], dtype=jnp.float32),
        (values_first,),
        rounding_token,
    )
    return jnp.moveaxis(history, 0, axis)


def _profile_rrtmg_sw_ready(value: Any) -> None:
    if not _PROFILE_RRTMG_SW_PATH:
        return
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _profile_rrtmg_sw_event(name: str) -> None:
    if not _PROFILE_RRTMG_SW_PATH:
        return
    with open(_PROFILE_RRTMG_SW_PATH, "a", encoding="ascii") as handle:
        handle.write(f"0.000000 {name}\n")


def _profile_rrtmg_sw_write(name: str, start: float, value: Any) -> None:
    if not _PROFILE_RRTMG_SW_PATH:
        return
    _profile_rrtmg_sw_ready(value)
    with open(_PROFILE_RRTMG_SW_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{perf_counter() - start:.6f} {name}\n")


def _capture_rrtmg_sw_target_host(
    coszen: Any,
    pavel: Any,
    pz: Any,
    tavel: Any,
    tz: Any,
    coldry: Any,
    wkl: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colo3: Any,
    sfluxzen: Any,
    taug: Any,
    taur: Any,
    cldfmc: Any,
    swdflx: Any,
    swuflx: Any,
) -> None:
    values = np.asarray(coszen, dtype=np.float32)
    matches = np.flatnonzero(values == _CAPTURE_RRTMG_SW_TARGET_COSZEN)
    if matches.size != 1:
        return
    index = int(matches[0])
    path = Path(_CAPTURE_RRTMG_SW_TARGET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        index=np.asarray(index, dtype=np.int32),
        coszen=values[index],
        pavel=np.asarray(pavel, dtype=np.float32)[index],
        pz=np.asarray(pz, dtype=np.float32)[index],
        tavel=np.asarray(tavel, dtype=np.float32)[index],
        tz=np.asarray(tz, dtype=np.float32)[index],
        coldry=np.asarray(coldry, dtype=np.float32)[index],
        wkl=np.asarray(wkl, dtype=np.float32)[index],
        laytrop=np.asarray(laytrop)[index],
        jp=np.asarray(jp)[index],
        jt=np.asarray(jt)[index],
        jt1=np.asarray(jt1)[index],
        colo3=np.asarray(colo3, dtype=np.float32)[index],
        sfluxzen=np.asarray(sfluxzen, dtype=np.float32)[index],
        taug=np.asarray(taug, dtype=np.float32)[index],
        taur=np.asarray(taur, dtype=np.float32)[index],
        cldfmc=np.asarray(cldfmc, dtype=np.float32)[index],
        swdflx=np.asarray(swdflx, dtype=np.float32)[index],
        swuflx=np.asarray(swuflx, dtype=np.float32)[index],
    )


def _capture_rrtmg_sw_o3_context_host(
    coszen: Any,
    plev: Any,
    o3mmr: Any,
    o3vmr: Any,
) -> None:
    values = np.asarray(coszen, dtype=np.float32)
    matches = np.flatnonzero(values == _CAPTURE_RRTMG_SW_TARGET_COSZEN)
    if matches.size != 1:
        return
    index = int(matches[0])
    target = Path(_CAPTURE_RRTMG_SW_TARGET_PATH)
    path = target.with_name(f"{target.stem}-o3-context{target.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        index=np.asarray(index, dtype=np.int32),
        coszen=values[index],
        plev=np.asarray(plev, dtype=np.float32)[index],
        o3mmr=np.asarray(o3mmr, dtype=np.float32)[index],
        o3vmr=np.asarray(o3vmr, dtype=np.float32)[index],
    )


def _capture_rrtmg_sw_transport_host(
    zincflx: Any,
    total_pfd: Any,
    total_pfu: Any,
    clear_pfd: Any,
    clear_pfu: Any,
    ztau_clear_raw: Any,
    zom_clear_raw: Any,
    zg_clear_raw: Any,
    prmu0: Any,
    ze1_nodel: Any,
    zdbtc_nodel_layer: Any,
    ztauc: Any,
    zomcc: Any,
    zgcc: Any,
    ze1: Any,
    zdbtc_layer: Any,
    ztdbt_nodel: Any,
    ztdbtc_nodel: Any,
) -> None:
    path = Path(_CAPTURE_RRTMG_SW_TRANSPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        zincflx=np.asarray(zincflx, dtype=np.float32),
        total_pfd=np.asarray(total_pfd, dtype=np.float32),
        total_pfu=np.asarray(total_pfu, dtype=np.float32),
        clear_pfd=np.asarray(clear_pfd, dtype=np.float32),
        clear_pfu=np.asarray(clear_pfu, dtype=np.float32),
        ztau_clear_raw=np.asarray(ztau_clear_raw, dtype=np.float32),
        zom_clear_raw=np.asarray(zom_clear_raw, dtype=np.float32),
        zg_clear_raw=np.asarray(zg_clear_raw, dtype=np.float32),
        prmu0=np.asarray(prmu0, dtype=np.float32),
        ze1_nodel=np.asarray(ze1_nodel, dtype=np.float32),
        zdbtc_nodel_layer=np.asarray(zdbtc_nodel_layer, dtype=np.float32),
        ztauc=np.asarray(ztauc, dtype=np.float32),
        zomcc=np.asarray(zomcc, dtype=np.float32),
        zgcc=np.asarray(zgcc, dtype=np.float32),
        ze1=np.asarray(ze1, dtype=np.float32),
        zdbtc_layer=np.asarray(zdbtc_layer, dtype=np.float32),
        ztdbt_nodel=np.asarray(ztdbt_nodel, dtype=np.float32),
        ztdbtc_nodel=np.asarray(ztdbtc_nodel, dtype=np.float32),
    )


_SW_NGC_VALUES = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)
_RRTMG_SW_JPBAND = 29
_RRTMG_SW_NMOL = 7
_RRTMG_SW_SOLAR_CONSTANT = 1.36822e3


_PREFLOG_VALUES = (
    6.9600,
    6.7600,
    6.5600,
    6.3600,
    6.1600,
    5.9600,
    5.7600,
    5.5600,
    5.3600,
    5.1600,
    4.9600,
    4.7600,
    4.5600,
    4.3600,
    4.1600,
    3.9600,
    3.7600,
    3.5600,
    3.3600,
    3.1600,
    2.9600,
    2.7600,
    2.5600,
    2.3600,
    2.1600,
    1.9600,
    1.7600,
    1.5600,
    1.3600,
    1.1600,
    0.9600,
    0.7600,
    0.5600,
    0.3600,
    0.1600,
    -0.0400,
    -0.2400,
    -0.4400,
    -0.6400,
    -0.8400,
    -1.0400,
    -1.2400,
    -1.4400,
    -1.6400,
    -1.8400,
    -2.0400,
    -2.2400,
    -2.4400,
    -2.6400,
    -2.8400,
    -3.0400,
    -3.2400,
    -3.4400,
    -3.6400,
    -3.8400,
    -4.0400,
    -4.2400,
    -4.4400,
    -4.6400,
)

_TREF_VALUES = (
    294.20,
    287.99,
    278.94,
    269.25,
    259.83,
    250.17,
    240.77,
    231.79,
    223.06,
    215.78,
    215.70,
    215.70,
    215.70,
    217.06,
    218.58,
    220.18,
    221.74,
    223.28,
    224.79,
    226.55,
    228.34,
    231.13,
    234.01,
    237.03,
    240.22,
    243.71,
    247.26,
    250.85,
    254.57,
    258.32,
    262.16,
    266.06,
    269.99,
    273.40,
    275.36,
    275.68,
    273.72,
    271.63,
    269.55,
    265.93,
    262.11,
    258.28,
    253.60,
    248.54,
    243.48,
    238.09,
    232.06,
    226.03,
    220.00,
    214.35,
    208.87,
    203.40,
    197.92,
    192.90,
    188.09,
    183.29,
    178.49,
    173.94,
    172.12,
)


class RrtmgSwReferenceAtmosphere(NamedTuple):
    """Reference pressure and temperature tables from WRF ``swatmref``."""

    preflog: Any
    tref: Any


class RrtmgSwSetcoefResult(NamedTuple):
    """Outputs from WRF ``setcoef_sw`` for one atmospheric column."""

    laytrop: Any
    layswtch: Any
    laylow: Any
    jp: Any
    jt: Any
    jt1: Any
    co2mult: Any
    colch4: Any
    colco2: Any
    colh2o: Any
    colmol: Any
    coln2o: Any
    colo2: Any
    colo3: Any
    fac00: Any
    fac01: Any
    fac10: Any
    fac11: Any
    selffac: Any
    selffrac: Any
    indself: Any
    forfac: Any
    forfrac: Any
    indfor: Any


class RrtmgSwTaumolBandResult(NamedTuple):
    """Shortwave optical depth, Rayleigh depth, and solar flux for one band."""

    sfluxzen: Any
    taug: Any
    taur: Any


class RrtmgSwReftraResult(NamedTuple):
    """Layer reflectance/transmittance outputs from WRF ``reftra_sw``."""

    pref: Any
    prefd: Any
    ptra: Any
    ptrad: Any


class RrtmgSwVrtqdrResult(NamedTuple):
    """Vertical quadrature flux outputs from WRF ``vrtqdr_sw`` for one g-point."""

    prdnd: Any
    prup: Any
    prupd: Any
    pfd: Any
    pfu: Any


class RrtmgSwSpcvmcResult(NamedTuple):
    """Spectrally integrated shortwave fluxes from WRF ``spcvmc_sw``."""

    pbbfd: Any
    pbbfu: Any
    pbbcd: Any
    pbbcu: Any
    puvfd: Any
    puvcd: Any
    pnifd: Any
    pnicd: Any
    pbbfddir: Any
    pbbcddir: Any
    puvfddir: Any
    puvcddir: Any
    pnifddir: Any
    pnicddir: Any


class RrtmgSwCloudTables(NamedTuple):
    """WRF ``swcldpr`` cloud optical property lookup tables."""

    abari: Any
    bbari: Any
    cbari: Any
    dbari: Any
    ebari: Any
    fbari: Any
    extliq1: Any
    ssaliq1: Any
    asyliq1: Any
    extice2: Any
    ssaice2: Any
    asyice2: Any
    extice3: Any
    ssaice3: Any
    asyice3: Any
    fdlice3: Any


class RrtmgSwCldprmcResult(NamedTuple):
    """Cloud optical properties from WRF ``cldprmc_sw``."""

    taormc: Any
    taucmc: Any
    ssacmc: Any
    asmcmc: Any


class RrtmgSwColumnResult(NamedTuple):
    """Single-column flux outputs from WRF ``rrtmg_sw``."""

    swuflx: Any
    swdflx: Any
    swhr: Any
    swuflxc: Any
    swdflxc: Any
    swhrc: Any
    swuflxcln: Any
    swdflxcln: Any
    sibvisdir: Any
    sibvisdif: Any
    sibnirdir: Any
    sibnirdif: Any
    swdkdir: Any
    swdkdif: Any
    swdkdirc: Any
    sw_zbbcddir: Any
    sw_dirdflux: Any
    sw_difdflux: Any


class RrtmgSwWrfColumnResult(NamedTuple):
    """WRF ``rrtmg_swrad`` outputs mapped back to model grid fields."""

    rthratensw: Any
    rthratenswc: Any
    swupt: Any
    swuptc: Any
    swuptcln: Any
    swdnt: Any
    swdntc: Any
    swdntcln: Any
    swupb: Any
    swupbc: Any
    swupbcln: Any
    swdnb: Any
    swdnbc: Any
    swdnbcln: Any
    swcf: Any
    gsw: Any
    swddir: Any
    swddni: Any
    swddif: Any
    swdownc: Any
    swddnic: Any
    swddirc: Any
    coszr: Any


class RrtmgSwInatmResult(NamedTuple):
    """Column inputs prepared by WRF ``inatm_sw``."""

    nlayers: int
    pavel: Any
    pz: Any
    pdp: Any
    tavel: Any
    tz: Any
    tbound: Any
    coldry: Any
    wkl: Any
    adjflux: Any
    solvar: Any
    inflag: int
    iceflag: int
    liqflag: int
    cldfmc: Any
    taucmc: Any
    ssacmc: Any
    asmcmc: Any
    fsfcmc: Any
    ciwpmc: Any
    clwpmc: Any
    cswpmc: Any
    reicmc: Any
    relqmc: Any
    resnmc: Any
    taua: Any
    ssaa: Any
    asma: Any


class RrtmgSwBandData(NamedTuple):
    """One raw band record read from WRF ``RRTMG_SW_DATA``."""

    band: int
    rayl: Any | None = None
    raylo: Any | None = None
    raylao: Any | None = None
    raylbo: Any | None = None
    strrat: Any | None = None
    strrat1: Any | None = None
    givfac: Any | None = None
    scalekur: Any | None = None
    layreffr: Any | None = None
    kao: Any | None = None
    kbo: Any | None = None
    selfrefo: Any | None = None
    forrefo: Any | None = None
    sfluxrefo: Any | None = None
    absch4o: Any | None = None
    abso3ao: Any | None = None
    abso3bo: Any | None = None
    absh2oo: Any | None = None
    absco2o: Any | None = None


class RrtmgSwLookupTables(NamedTuple):
    """Raw WRF shortwave RRTMG lookup tables keyed by band number."""

    band16: RrtmgSwBandData
    band17: RrtmgSwBandData
    band18: RrtmgSwBandData
    band19: RrtmgSwBandData
    band20: RrtmgSwBandData
    band21: RrtmgSwBandData
    band22: RrtmgSwBandData
    band23: RrtmgSwBandData
    band24: RrtmgSwBandData
    band25: RrtmgSwBandData
    band26: RrtmgSwBandData
    band27: RrtmgSwBandData
    band28: RrtmgSwBandData
    band29: RrtmgSwBandData


class RrtmgSwCompressedBandData(NamedTuple):
    """One shortwave RRTMG band after WRF ``cmbgb*`` g-point reduction."""

    band: int
    rayl: Any | None = None
    rayla: Any | None = None
    raylb: Any | None = None
    strrat: Any | None = None
    strrat1: Any | None = None
    givfac: Any | None = None
    scalekur: Any | None = None
    layreffr: Any | None = None
    ka: Any | None = None
    kb: Any | None = None
    selfref: Any | None = None
    forref: Any | None = None
    sfluxref: Any | None = None
    absch4: Any | None = None
    abso3a: Any | None = None
    abso3b: Any | None = None
    absh2o: Any | None = None
    absco2: Any | None = None


class RrtmgSwCompressedLookupTables(NamedTuple):
    """Compressed shortwave RRTMG lookup tables used by ``taumol_sw``."""

    band16: RrtmgSwCompressedBandData
    band17: RrtmgSwCompressedBandData
    band18: RrtmgSwCompressedBandData
    band19: RrtmgSwCompressedBandData
    band20: RrtmgSwCompressedBandData
    band21: RrtmgSwCompressedBandData
    band22: RrtmgSwCompressedBandData
    band23: RrtmgSwCompressedBandData
    band24: RrtmgSwCompressedBandData
    band25: RrtmgSwCompressedBandData
    band26: RrtmgSwCompressedBandData
    band27: RrtmgSwCompressedBandData
    band28: RrtmgSwCompressedBandData
    band29: RrtmgSwCompressedBandData


def swatmref() -> RrtmgSwReferenceAtmosphere:
    """WRF ``module_ra_rrtmg_sw:swatmref`` reference atmosphere."""

    return RrtmgSwReferenceAtmosphere(
        preflog=jnp.asarray(_PREFLOG_VALUES, dtype=jnp.float32),
        tref=jnp.asarray(_TREF_VALUES, dtype=jnp.float32),
    )


@lru_cache(maxsize=1)
def load_rrtmg_sw_cloud_tables(path: str | Path | None = None) -> RrtmgSwCloudTables:
    """Load WRF ``swcldpr`` shortwave cloud optical tables."""

    table_path = Path(path) if path is not None else Path(__file__).with_name("data") / "rrtmg_sw_cloud_tables.npz"
    with np.load(table_path) as data:
        return RrtmgSwCloudTables(
            abari=jnp.asarray(data["abari"], dtype=jnp.float32),
            bbari=jnp.asarray(data["bbari"], dtype=jnp.float32),
            cbari=jnp.asarray(data["cbari"], dtype=jnp.float32),
            dbari=jnp.asarray(data["dbari"], dtype=jnp.float32),
            ebari=jnp.asarray(data["ebari"], dtype=jnp.float32),
            fbari=jnp.asarray(data["fbari"], dtype=jnp.float32),
            extliq1=jnp.asarray(data["extliq1"], dtype=jnp.float32),
            ssaliq1=jnp.asarray(data["ssaliq1"], dtype=jnp.float32),
            asyliq1=jnp.asarray(data["asyliq1"], dtype=jnp.float32),
            extice2=jnp.asarray(data["extice2"], dtype=jnp.float32),
            ssaice2=jnp.asarray(data["ssaice2"], dtype=jnp.float32),
            asyice2=jnp.asarray(data["asyice2"], dtype=jnp.float32),
            extice3=jnp.asarray(data["extice3"], dtype=jnp.float32),
            ssaice3=jnp.asarray(data["ssaice3"], dtype=jnp.float32),
            asyice3=jnp.asarray(data["asyice3"], dtype=jnp.float32),
            fdlice3=jnp.asarray(data["fdlice3"], dtype=jnp.float32),
        )


def _read_record_payload(raw: bytes, offset: int, path: Path) -> tuple[memoryview, int]:
    if offset + 8 > len(raw):
        raise ValueError(f"Unexpected end of Fortran records in {path}")
    leading = int(np.frombuffer(raw, dtype=">i4", count=1, offset=offset)[0])
    offset += 4
    end = offset + leading
    if end + 4 > len(raw):
        raise ValueError(f"Truncated Fortran record in {path}")
    trailing = int(np.frombuffer(raw, dtype=">i4", count=1, offset=end)[0])
    if trailing != leading:
        raise ValueError(f"Mismatched Fortran record markers in {path}: {leading} != {trailing}")
    return memoryview(raw)[offset:end], end + 4


def _parse_record_fields(payload: memoryview, fields: tuple[tuple[str, str, tuple[int, ...] | None], ...]) -> dict[str, Any]:
    offset = 0
    values: dict[str, Any] = {}
    for name, kind, shape in fields:
        if kind == "r":
            dtype = np.dtype(">f4")
        elif kind == "i":
            dtype = np.dtype(">i4")
        else:
            raise ValueError(f"Unsupported RRTMG field kind: {kind}")
        count = 1 if shape is None else int(np.prod(shape))
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ValueError(f"RRTMG record too short while reading {name}")
        parsed = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += nbytes
        if shape is None:
            value: Any = parsed.astype(np.float32 if kind == "r" else np.int32)[0]
            values[name] = value
        else:
            values[name] = np.array(
                parsed.reshape(shape, order="F"),
                dtype=np.float32 if kind == "r" else np.int32,
                copy=True,
            )
    if offset != len(payload):
        raise ValueError(f"Unexpected extra bytes in RRTMG record: {len(payload) - offset}")
    return values


_SW_KGB_RECORDS: tuple[tuple[int, tuple[tuple[str, str, tuple[int, ...] | None], ...]], ...] = (
    (
        16,
        (
            ("rayl", "r", None),
            ("strrat1", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
    (
        17,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (4, 16)),
            ("sfluxrefo", "r", (16, 5)),
        ),
    ),
    (
        18,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16, 9)),
        ),
    ),
    (
        19,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16, 9)),
        ),
    ),
    (
        20,
        (
            ("rayl", "r", None),
            ("layreffr", "i", None),
            ("absch4o", "r", (16,)),
            ("kao", "r", (5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (4, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
    (
        21,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (4, 16)),
            ("sfluxrefo", "r", (16, 9)),
        ),
    ),
    (
        22,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16, 9)),
        ),
    ),
    (
        23,
        (
            ("raylo", "r", (16,)),
            ("givfac", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (5, 13, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
    (
        24,
        (
            ("raylao", "r", (16, 9)),
            ("raylbo", "r", (16,)),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("abso3ao", "r", (16,)),
            ("abso3bo", "r", (16,)),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (3, 16)),
            ("sfluxrefo", "r", (16, 9)),
        ),
    ),
    (
        25,
        (
            ("raylo", "r", (16,)),
            ("layreffr", "i", None),
            ("abso3ao", "r", (16,)),
            ("abso3bo", "r", (16,)),
            ("kao", "r", (5, 13, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
    (26, (("raylo", "r", (16,)), ("sfluxrefo", "r", (16,)))),
    (
        27,
        (
            ("raylo", "r", (16,)),
            ("scalekur", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
    (
        28,
        (
            ("rayl", "r", None),
            ("strrat", "r", None),
            ("layreffr", "i", None),
            ("kao", "r", (9, 5, 13, 16)),
            ("kbo", "r", (5, 5, 47, 16)),
            ("sfluxrefo", "r", (16, 5)),
        ),
    ),
    (
        29,
        (
            ("rayl", "r", None),
            ("layreffr", "i", None),
            ("absh2oo", "r", (16,)),
            ("absco2o", "r", (16,)),
            ("kao", "r", (5, 13, 16)),
            ("kbo", "r", (5, 47, 16)),
            ("selfrefo", "r", (10, 16)),
            ("forrefo", "r", (4, 16)),
            ("sfluxrefo", "r", (16,)),
        ),
    ),
)


@lru_cache(maxsize=4)
def load_rrtmg_sw_data(path: str | Path) -> RrtmgSwLookupTables:
    """Read WRF ``RRTMG_SW_DATA`` without using the Fortran bridge."""

    file_path = Path(path)
    raw = file_path.read_bytes()
    offset = 0
    bands: list[RrtmgSwBandData] = []
    for band_number, fields in _SW_KGB_RECORDS:
        payload, offset = _read_record_payload(raw, offset, file_path)
        values = _parse_record_fields(payload, fields)
        bands.append(RrtmgSwBandData(band=band_number, **values))
    if offset != len(raw):
        raise ValueError(f"Unexpected extra bytes in {file_path}: {len(raw) - offset}")
    return RrtmgSwLookupTables(*bands)


def _rrtmg_sw_groups(spectral: RrtmgSpectralInit, band_index: int) -> list[tuple[slice, np.ndarray]]:
    try:
        ngc = np.asarray(spectral.ngc, dtype=np.int32)
        ngs = np.asarray(spectral.ngs, dtype=np.int32)
        ngn = np.asarray(spectral.ngn, dtype=np.int32)
        rwgt = np.asarray(spectral.rwgt, dtype=np.float32)
    except Exception:
        ngc = np.asarray(_SW_NGC_VALUES, dtype=np.int32)
        ngs = np.asarray(_SW_NGS_VALUES, dtype=np.int32)
        ngn = np.asarray(_SW_NGN_VALUES, dtype=np.int32)
        rwgt = np.asarray(SW_RWGT, dtype=np.float32)
    group_count = int(ngc[band_index])
    compressed_start = 0 if band_index == 0 else int(ngs[band_index - 1])
    original_start = band_index * 16
    old_pos = 0
    groups: list[tuple[slice, np.ndarray]] = []
    for igc in range(group_count):
        count = int(ngn[compressed_start + igc])
        old_slice = slice(old_pos, old_pos + count)
        weight_slice = slice(original_start + old_pos, original_start + old_pos + count)
        groups.append((old_slice, rwgt[weight_slice]))
        old_pos += count
    if old_pos != 16:
        raise ValueError(f"RRTMG SW band {band_index + 16} does not cover 16 original g-points")
    return groups


def _combine_old_last(values: Any, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty(arr.shape[:-1] + (len(groups),), dtype=np.float32)
    for igc, (old_slice, weights) in enumerate(groups):
        chunk = arr[..., old_slice]
        terms = np.asarray(
            chunk * weights.reshape((1,) * (chunk.ndim - 1) + (weights.size,)) if weighted else chunk,
            dtype=np.float32,
        )
        out[..., igc] = np.add.accumulate(terms, axis=-1, dtype=np.float32)[..., -1]
    return out


def _combine_old_first(values: Any, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty((len(groups),) + arr.shape[1:], dtype=np.float32)
    for igc, (old_slice, weights) in enumerate(groups):
        chunk = arr[old_slice, ...]
        terms = np.asarray(
            chunk * weights.reshape((weights.size,) + (1,) * (chunk.ndim - 1)) if weighted else chunk,
            dtype=np.float32,
        )
        out[igc, ...] = np.add.accumulate(terms, axis=0, dtype=np.float32)[-1, ...]
    return out


def _combine_optional_old_last(values: Any | None, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray | None:
    if values is None:
        return None
    return _combine_old_last(values, groups, weighted=weighted)


def _combine_optional_old_first(values: Any | None, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray | None:
    if values is None:
        return None
    return _combine_old_first(values, groups, weighted=weighted)


def _compress_rrtmg_sw_band(
    band: RrtmgSwBandData,
    *,
    spectral: RrtmgSpectralInit,
    band_index: int,
) -> RrtmgSwCompressedBandData:
    groups = _rrtmg_sw_groups(spectral, band_index)
    scalar_rayl: Any | None = band.rayl
    compressed_rayl: Any | None = None
    if band.raylo is not None:
        compressed_rayl = _combine_old_first(band.raylo, groups, weighted=True)
    return RrtmgSwCompressedBandData(
        band=band.band,
        rayl=scalar_rayl if scalar_rayl is not None else compressed_rayl,
        rayla=_combine_optional_old_first(band.raylao, groups, weighted=True),
        raylb=_combine_optional_old_first(band.raylbo, groups, weighted=True),
        strrat=band.strrat,
        strrat1=band.strrat1,
        givfac=band.givfac,
        scalekur=band.scalekur,
        layreffr=band.layreffr,
        ka=_combine_optional_old_last(band.kao, groups, weighted=True),
        kb=_combine_optional_old_last(band.kbo, groups, weighted=True),
        selfref=_combine_optional_old_last(band.selfrefo, groups, weighted=True),
        forref=_combine_optional_old_last(band.forrefo, groups, weighted=True),
        sfluxref=_combine_optional_old_first(band.sfluxrefo, groups, weighted=False),
        absch4=_combine_optional_old_first(band.absch4o, groups, weighted=True),
        abso3a=_combine_optional_old_first(band.abso3ao, groups, weighted=True),
        abso3b=_combine_optional_old_first(band.abso3bo, groups, weighted=True),
        absh2o=_combine_optional_old_first(band.absh2oo, groups, weighted=True),
        absco2=_combine_optional_old_first(band.absco2o, groups, weighted=True),
    )


def combine_rrtmg_sw_lookup_tables(
    raw: RrtmgSwLookupTables,
    *,
    spectral: RrtmgSpectralInit | None = None,
) -> RrtmgSwCompressedLookupTables:
    """Apply WRF ``cmbgb16s`` through ``cmbgb29`` g-point reduction."""

    spec = swdatinit() if spectral is None else spectral
    raw_bands = (
        raw.band16,
        raw.band17,
        raw.band18,
        raw.band19,
        raw.band20,
        raw.band21,
        raw.band22,
        raw.band23,
        raw.band24,
        raw.band25,
        raw.band26,
        raw.band27,
        raw.band28,
        raw.band29,
    )
    return RrtmgSwCompressedLookupTables(
        *(
            _compress_rrtmg_sw_band(band, spectral=spec, band_index=band_index)
            for band_index, band in enumerate(raw_bands)
        )
    )


@lru_cache(maxsize=4)
def load_rrtmg_sw_compressed_lookup_tables(path: str | Path | None = None) -> RrtmgSwCompressedLookupTables:
    """Load and compress WRF RRTMG shortwave lookup tables once per path."""

    default_path = Path(__file__).with_name("data") / "rrtmg" / "RRTMG_SW_DATA"
    path = configured_path("rrtmg_sw_data", env="WRF_PY_RRTMG_SW_DATA", fallback=default_path) if path is None else path
    return combine_rrtmg_sw_lookup_tables(load_rrtmg_sw_data(path))


def _sw_exp_lookup(
    value: Any,
    tables: RrtmgSwPadeTables,
    *,
    od_lo: float = 0.06,
    rounding_token: Any | None = None,
) -> Any:
    """WRF shortwave Pade exponential lookup with the small-optical-depth branch."""

    value_arr = jnp.asarray(value, dtype=jnp.float32)
    exp_tbl = jnp.asarray(tables.exp_tbl, dtype=jnp.float32)
    ntbl = exp_tbl.shape[0] - 1
    tblint = jnp.asarray(float(ntbl), dtype=jnp.float32)
    bpade = jnp.asarray(tables.bpade, dtype=jnp.float32)
    if rounding_token is None:
        tblind = value_arr / (bpade + value_arr)
        itind = jnp.clip((tblint * tblind + 0.5).astype(jnp.int32), 0, ntbl)
        approx = 1.0 - value_arr + 0.5 * value_arr * value_arr
    else:
        def lookup(values, parameters):
            table, pade, token = parameters
            return _sw_exp_lookup_real(values[0], table, pade, token, od_lo=od_lo)

        return map_elements(lookup, (value_arr,), (exp_tbl, bpade, rounding_token))
    return jnp.where(value_arr <= jnp.asarray(od_lo, dtype=jnp.float32), approx,
                     pade_transmittance(value_arr, exp_tbl[itind]))


def _sw_exp_lookup_real(value_arr, exp_tbl, bpade, rounding_token, *, od_lo=0.06):
    """The same REAL lookup for array programs and read-only kernel tables."""
    from wrf_jax.core.real_arithmetic import (
        real_add as _wrf_real_add,
        real_subtract as _wrf_real_sub,
        real_multiply as _wrf_real_mul,
        real_divide_assigned as _wrf_real_div,
    )

    ntbl = exp_tbl.shape[0] - 1
    tblint = jnp.asarray(float(ntbl), dtype=jnp.float32)
    tblind = _wrf_real_div(
        value_arr,
        _wrf_real_add(bpade, value_arr, rounding_token),
        rounding_token,
    )
    itind_real = _wrf_real_add(
        _wrf_real_mul(tblint, tblind, rounding_token),
        jnp.float32(0.5),
        rounding_token,
    )
    itind = jnp.clip(itind_real.astype(jnp.int32), 0, ntbl)
    approx = _wrf_real_add(
        _wrf_real_sub(jnp.float32(1.0), value_arr, rounding_token),
        _wrf_real_mul(
            _wrf_real_mul(jnp.float32(0.5), value_arr, rounding_token),
            value_arr,
            rounding_token,
        ),
        rounding_token,
    )
    return jnp.where(value_arr <= jnp.asarray(od_lo, dtype=jnp.float32), approx,
                     pade_transmittance(value_arr, exp_tbl[itind]))


def earth_sun(day_of_year: Any) -> Any:
    """WRF ``earth_sun`` Earth-Sun distance correction."""

    day = jnp.asarray(day_of_year, dtype=jnp.float32)
    pi = jnp.asarray(np.pi, dtype=jnp.float32)
    gamma = 2.0 * pi * (day - 1.0) / 365.0
    return (
        1.000110
        + 0.034221 * jnp.cos(gamma)
        + 0.001289 * jnp.sin(gamma)
        + 0.000719 * jnp.cos(2.0 * gamma)
        + 0.000077 * jnp.sin(2.0 * gamma)
    )


def inatm_sw_column(
    *,
    nlay: int,
    icld: int,
    iaer: int,
    play: Any,
    plev: Any,
    tlay: Any,
    tlev: Any,
    tsfc: Any,
    h2ovmr: Any,
    o3vmr: Any,
    co2vmr: Any,
    ch4vmr: Any,
    n2ovmr: Any,
    o2vmr: Any,
    adjes: Any,
    dyofyr: Any,
    scon: Any,
    inflgsw: int,
    iceflgsw: int,
    liqflgsw: int,
    cldfmcl: Any | None = None,
    taucmcl: Any | None = None,
    ssacmcl: Any | None = None,
    asmcmcl: Any | None = None,
    fsfcmcl: Any | None = None,
    ciwpmcl: Any | None = None,
    clwpmcl: Any | None = None,
    cswpmcl: Any | None = None,
    reicmcl: Any | None = None,
    relqmcl: Any | None = None,
    resnmcl: Any | None = None,
    tauaer: Any | None = None,
    ssaaer: Any | None = None,
    asmaer: Any | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwInatmResult:
    """WRF ``inatm_sw`` for one atmospheric column."""

    n = int(nlay)
    play_arr = jnp.asarray(play, dtype=jnp.float32)[:n]
    plev_arr = jnp.asarray(plev, dtype=jnp.float32)[: n + 1]
    tlay_arr = jnp.asarray(tlay, dtype=jnp.float32)[:n]
    tlev_arr = jnp.asarray(tlev, dtype=jnp.float32)[: n + 1]
    h2o = jnp.asarray(h2ovmr, dtype=jnp.float32)[:n]
    o3 = jnp.asarray(o3vmr, dtype=jnp.float32)[:n]
    co2 = jnp.asarray(co2vmr, dtype=jnp.float32)[:n]
    ch4 = jnp.asarray(ch4vmr, dtype=jnp.float32)[:n]
    n2o = jnp.asarray(n2ovmr, dtype=jnp.float32)[:n]
    o2 = jnp.asarray(o2vmr, dtype=jnp.float32)[:n]

    adjflx = jnp.where(jnp.asarray(dyofyr, dtype=jnp.int32) > 0, earth_sun(dyofyr), jnp.asarray(adjes, dtype=jnp.float32))
    solvar_band = jnp.asarray(scon, dtype=jnp.float32) / jnp.asarray(_RRTMG_SW_SOLAR_CONSTANT, dtype=jnp.float32)
    solvar = jnp.zeros((_RRTMG_SW_JPBAND,), dtype=jnp.float32).at[15:29].set(solvar_band)
    adjflux = jnp.zeros((_RRTMG_SW_JPBAND,), dtype=jnp.float32).at[15:29].set(adjflx * solvar_band)

    pavel = play_arr
    tavel = tlay_arr
    pz = plev_arr
    tz = tlev_arr
    pdp = _wrf_real_sub(pz[:-1], pz[1:], rounding_token)

    amd = jnp.asarray(28.9660, dtype=jnp.float32)
    amw = jnp.asarray(18.0160, dtype=jnp.float32)
    amm = _wrf_real_add(
        _wrf_real_mul(
            _wrf_real_sub(
                jnp.asarray(1.0, dtype=jnp.float32),
                h2o,
                rounding_token,
            ),
            amd,
            rounding_token,
        ),
        _wrf_real_mul(h2o, amw, rounding_token),
        rounding_token,
    )
    coldry = _wrf_inatm_coldry(pdp, h2o, amm, rounding_token)

    wkl = jnp.zeros((_RRTMG_SW_NMOL, n), dtype=jnp.float32)
    wkl = wkl.at[0].set(h2o)
    wkl = wkl.at[1].set(co2)
    wkl = wkl.at[2].set(o3)
    wkl = wkl.at[3].set(n2o)
    wkl = wkl.at[5].set(ch4)
    wkl = wkl.at[6].set(o2)
    wkl = _wrf_real_mul(wkl, coldry[None, :], rounding_token)

    taua = jnp.zeros((n, RRTMG_SW_NBND), dtype=jnp.float32)
    ssaa = jnp.ones((n, RRTMG_SW_NBND), dtype=jnp.float32)
    asma = jnp.zeros((n, RRTMG_SW_NBND), dtype=jnp.float32)
    if int(iaer) >= 1:
        taua = jnp.asarray(tauaer, dtype=jnp.float32)[:n, :RRTMG_SW_NBND] if tauaer is not None else taua
        ssaa = jnp.asarray(ssaaer, dtype=jnp.float32)[:n, :RRTMG_SW_NBND] if ssaaer is not None else ssaa
        asma = jnp.asarray(asmaer, dtype=jnp.float32)[:n, :RRTMG_SW_NBND] if asmaer is not None else asma

    cldfmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    taucmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    ssacmc = jnp.ones((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    asmcmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    fsfcmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    ciwpmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    clwpmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    cswpmc = jnp.zeros((RRTMG_SW_NGPT, n), dtype=jnp.float32)
    reicmc = jnp.zeros((n,), dtype=jnp.float32)
    relqmc = jnp.zeros((n,), dtype=jnp.float32)
    resnmc = jnp.zeros((n,), dtype=jnp.float32)

    inflag = int(inflgsw)
    iceflag = int(iceflgsw)
    liqflag = int(liqflgsw)
    if int(icld) >= 1:
        cldfmc = jnp.asarray(cldfmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if cldfmcl is not None else cldfmc
        taucmc = jnp.asarray(taucmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if taucmcl is not None else taucmc
        ssacmc = jnp.asarray(ssacmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if ssacmcl is not None else ssacmc
        asmcmc = jnp.asarray(asmcmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if asmcmcl is not None else asmcmc
        fsfcmc = jnp.asarray(fsfcmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if fsfcmcl is not None else fsfcmc
        ciwpmc = jnp.asarray(ciwpmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if ciwpmcl is not None else ciwpmc
        clwpmc = jnp.asarray(clwpmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n] if clwpmcl is not None else clwpmc
        if iceflag == 5 and cswpmcl is not None:
            cswpmc = jnp.asarray(cswpmcl, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
        reicmc = jnp.asarray(reicmcl, dtype=jnp.float32)[:n] if reicmcl is not None else reicmc
        relqmc = jnp.asarray(relqmcl, dtype=jnp.float32)[:n] if relqmcl is not None else relqmc
        if iceflag == 5 and resnmcl is not None:
            resnmc = jnp.asarray(resnmcl, dtype=jnp.float32)[:n]

    return RrtmgSwInatmResult(
        nlayers=n,
        pavel=pavel,
        pz=pz,
        pdp=pdp,
        tavel=tavel,
        tz=tz,
        tbound=jnp.asarray(tsfc, dtype=jnp.float32),
        coldry=coldry,
        wkl=wkl,
        adjflux=adjflux,
        solvar=solvar,
        inflag=inflag,
        iceflag=iceflag,
        liqflag=liqflag,
        cldfmc=cldfmc,
        taucmc=taucmc,
        ssacmc=ssacmc,
        asmcmc=asmcmc,
        fsfcmc=fsfcmc,
        ciwpmc=ciwpmc,
        clwpmc=clwpmc,
        cswpmc=cswpmc,
        reicmc=reicmc,
        relqmc=relqmc,
        resnmc=resnmc,
        taua=taua,
        ssaa=ssaa,
        asma=asma,
    )


def cldprmc_sw(
    *,
    nlayers: int,
    inflag: int,
    iceflag: int,
    liqflag: int,
    cldfmc: Any,
    ciwpmc: Any,
    clwpmc: Any,
    cswpmc: Any,
    reicmc: Any,
    relqmc: Any,
    resnmc: Any,
    taucmc: Any,
    ssacmc: Any,
    asmcmc: Any,
    fsfcmc: Any,
    spectral: RrtmgSpectralInit | None = None,
    cloud_tables: RrtmgSwCloudTables | None = None,
) -> RrtmgSwCldprmcResult:
    """WRF ``cldprmc_sw`` cloud optical processing for one column.

    Inputs and outputs use WRF's ``(ngptsw, nlayers)`` orientation.
    """

    n = int(nlayers)
    inflag_i = int(inflag)
    iceflag_i = int(iceflag)
    liqflag_i = int(liqflag)
    if inflag_i == 1:
        raise NotImplementedError("WRF cldprmc_sw stops for INFLAG = 1 with MCICA")
    if inflag_i >= 2 and liqflag_i != 1:
        raise NotImplementedError(f"WRF cldprmc_sw liquid option {liqflag_i} is not translated")

    spec = swdatinit() if spectral is None else spectral
    tables = load_rrtmg_sw_cloud_tables() if cloud_tables is None else cloud_tables
    cldmin = jnp.asarray(1.0e-20, dtype=jnp.float32)
    eps = jnp.asarray(1.0e-6, dtype=jnp.float32)

    cldf = jnp.asarray(cldfmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    ciwp = jnp.asarray(ciwpmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    clwp = jnp.asarray(clwpmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    cswp = jnp.asarray(cswpmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    tau_in = jnp.asarray(taucmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    ssa_in = jnp.asarray(ssacmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    asm_in = jnp.asarray(asmcmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    fsf = jnp.asarray(fsfcmc, dtype=jnp.float32)[:RRTMG_SW_NGPT, :n]
    radice = jnp.asarray(reicmc, dtype=jnp.float32)[:n][None, :]
    radliq = jnp.asarray(relqmc, dtype=jnp.float32)[:n][None, :]
    radsno = jnp.asarray(resnmc, dtype=jnp.float32)[:n][None, :]
    cwp = ciwp + clwp + cswp
    active = (cldf >= cldmin) & ((cwp >= cldmin) | (tau_in >= cldmin))
    taor_out = tau_in

    if inflag_i == 0:
        ffp = fsf
        ffp1 = 1.0 - ffp
        ffpssa = 1.0 - ffp * ssa_in
        tau_new = ffpssa * tau_in
        ssa_new = ffp1 * ssa_in / ffpssa
        asm_new = (asm_in - ffp) / ffp1
        return RrtmgSwCldprmcResult(
            taormc=jnp.where(active, tau_in, taor_out),
            taucmc=jnp.where(active, tau_new, tau_in),
            ssacmc=jnp.where(active, ssa_new, ssa_in),
            asmcmc=jnp.where(active, asm_new, asm_in),
        )

    ngb = jnp.asarray(spec.ngb, dtype=jnp.int32)[:RRTMG_SW_NGPT]
    band = ngb - 16
    band2 = band[:, None]

    def _table_lookup(table: Any, fortran_index: Any) -> Any:
        row0 = fortran_index.astype(jnp.int32) - 1
        return table[row0, band2]

    def _interp(table: Any, fortran_index: Any, fraction: Any) -> Any:
        row0 = fortran_index.astype(jnp.int32) - 1
        lower = table[row0, band2]
        upper = table[row0 + 1, band2]
        return lower + fraction * (upper - lower)

    has_ice_or_snow = (ciwp + cswp) != 0.0
    zeros = jnp.zeros_like(tau_in)
    if iceflag_i == 1:
        wavenum2 = jnp.asarray(spec.wavenum2, dtype=jnp.float32)[band][:, None]
        icx = jnp.where(
            wavenum2 > 1.43e4,
            1,
            jnp.where(wavenum2 > 7.7e3, 2, jnp.where(wavenum2 > 5.3e3, 3, jnp.where(wavenum2 > 4.0e3, 4, 5))),
        )
        icx0 = icx - 1
        extice = tables.abari[icx0] + tables.bbari[icx0] / radice
        ssaice = 1.0 - tables.cbari[icx0] - tables.dbari[icx0] * radice
        gice = tables.ebari[icx0] + tables.fbari[icx0] * radice
        gice = jnp.where(gice >= 1.0, 1.0 - eps, gice)
        forwice = gice * gice
        fdelta_ice = zeros
    elif iceflag_i == 2:
        factor = (radice - 2.0) / 3.0
        index = _fortran_int(factor)
        index = jnp.where(index == 43, 42, index)
        fint = factor - index.astype(jnp.float32)
        extice = _interp(tables.extice2, index, fint)
        ssaice = _interp(tables.ssaice2, index, fint)
        gice = _interp(tables.asyice2, index, fint)
        forwice = gice * gice
        fdelta_ice = zeros
    elif iceflag_i >= 3:
        factor = (radice - 2.0) / 3.0
        index = _fortran_int(factor)
        index = jnp.where(index == 46, 45, index)
        fint = factor - index.astype(jnp.float32)
        extice = _interp(tables.extice3, index, fint)
        ssaice = _interp(tables.ssaice3, index, fint)
        gice = _interp(tables.asyice3, index, fint)
        fdelta_ice = _interp(tables.fdlice3, index, fint)
        forwice = jnp.minimum(fdelta_ice + 0.5 / ssaice, gice)
    else:
        raise NotImplementedError(f"WRF cldprmc_sw ice option {iceflag_i} is not translated")

    extice = jnp.where(has_ice_or_snow, extice, 0.0)
    ssaice = jnp.where(has_ice_or_snow, ssaice, 0.0)
    gice = jnp.where(has_ice_or_snow, gice, 0.0)
    forwice = jnp.where(has_ice_or_snow, forwice, 0.0)
    del fdelta_ice

    if iceflag_i == 5:
        factor_snow = (radsno - 2.0) / 3.0
        index_snow = _fortran_int(factor_snow)
        index_snow = jnp.where(index_snow == 46, 45, index_snow)
        fint_snow = factor_snow - index_snow.astype(jnp.float32)
        extsno_raw = _interp(tables.extice3, index_snow, fint_snow)
        ssasno_raw = _interp(tables.ssaice3, index_snow, fint_snow)
        gsno_raw = _interp(tables.asyice3, index_snow, fint_snow)
        fdelta_snow = _interp(tables.fdlice3, index_snow, fint_snow)
        forwsno_raw = jnp.minimum(fdelta_snow + 0.5 / ssasno_raw, gsno_raw)
        has_snow = cswp > 0.0
        extsno = jnp.where(has_snow, extsno_raw, 0.0)
        ssasno = jnp.where(has_snow, ssasno_raw, 0.0)
        gsno = jnp.where(has_snow, gsno_raw, 0.0)
        forwsno = jnp.where(has_snow, forwsno_raw, 0.0)
    else:
        extsno = zeros
        ssasno = zeros
        gsno = zeros
        forwsno = zeros

    factor_liq = radliq - 1.5
    index_liq = _fortran_int(factor_liq)
    index_liq = jnp.where(index_liq == 0, 1, index_liq)
    index_liq = jnp.where(index_liq == 58, 57, index_liq)
    fint_liq = radliq - 1.5 - index_liq.astype(jnp.float32)
    extliq_raw = _interp(tables.extliq1, index_liq, fint_liq)
    ssaliq_raw = _interp(tables.ssaliq1, index_liq, fint_liq)
    ssaliq_raw = jnp.where((fint_liq < 0.0) & (ssaliq_raw > 1.0), _table_lookup(tables.ssaliq1, index_liq), ssaliq_raw)
    gliq_raw = _interp(tables.asyliq1, index_liq, fint_liq)
    has_liquid = clwp != 0.0
    extliq = jnp.where(has_liquid, extliq_raw, 0.0)
    ssaliq_coeff = jnp.where(has_liquid, ssaliq_raw, 0.0)
    gliq = jnp.where(has_liquid, gliq_raw, 0.0)
    forwliq = gliq * gliq

    tauliqorig = clwp * extliq
    tauiceorig = ciwp * extice
    ssaliq_delta = ssaliq_coeff * (1.0 - forwliq) / (1.0 - forwliq * ssaliq_coeff)
    tauliq = (1.0 - forwliq * ssaliq_coeff) * tauliqorig
    ssaice_delta = ssaice * (1.0 - forwice) / (1.0 - forwice * ssaice)
    tauice = (1.0 - forwice * ssaice) * tauiceorig
    scatliq = ssaliq_delta * tauliq
    scatice = ssaice_delta * tauice

    if iceflag_i == 5:
        tausnoorig = cswp * extsno
        ssasno_delta = ssasno * (1.0 - forwsno) / (1.0 - forwsno * ssasno)
        tausno = (1.0 - forwsno * ssasno) * tausnoorig
        scatsno = ssasno_delta * tausno
        tau_new = tauliq + tauice + tausno
        taor_new = tauliqorig + tauiceorig + tausnoorig
    else:
        tausno = zeros
        scatsno = zeros
        tau_new = tauliq + tauice
        taor_new = tauliqorig + tauiceorig
    del tausno

    tau_new = jnp.where(tau_new == 0.0, cldmin, tau_new)
    scatice = jnp.where(scatice == 0.0, cldmin, scatice)
    scatsno = jnp.where(scatsno == 0.0, cldmin, scatsno)
    if iceflag_i == 5:
        scatter_sum = scatliq + scatice + scatsno
        ssa_new = real_divide(scatter_sum, tau_new)
        asm_new = real_divide((
            scatliq * (gliq - forwliq) / (1.0 - forwliq)
            + scatice * (gice - forwice) / (1.0 - forwice)
            + scatsno * (gsno - forwsno) / (1.0 - forwsno)
        ), scatter_sum)
    elif iceflag_i in (3, 4):
        scatter_sum = scatliq + scatice
        ssa_new = real_divide(scatter_sum, tau_new)
        asm_new = real_divide((
            scatliq * (gliq - forwliq) / (1.0 - forwliq)
            + scatice * (gice - forwice) / (1.0 - forwice)
        ), scatter_sum)
    else:
        scatter_sum = scatliq + scatice
        ssa_new = real_divide(scatter_sum, tau_new)
        asm_new = real_divide((
            scatliq * (gliq - forwliq) / (1.0 - forwliq)
            + scatice * (gice - forwice) / (1.0 - forwice)
        ), scatter_sum)

    return RrtmgSwCldprmcResult(
        taormc=jnp.where(active, taor_new, taor_out),
        taucmc=jnp.where(active, tau_new, tau_in),
        ssacmc=jnp.where(active, ssa_new, ssa_in),
        asmcmc=jnp.where(active, asm_new, asm_in),
    )


def rrtmg_sw_column(
    *,
    nlay: int,
    icld: int,
    iaer: int,
    play: Any,
    plev: Any,
    tlay: Any,
    tlev: Any,
    tsfc: Any,
    h2ovmr: Any,
    o3vmr: Any,
    co2vmr: Any,
    ch4vmr: Any,
    n2ovmr: Any,
    o2vmr: Any,
    asdir: Any,
    asdif: Any,
    aldir: Any,
    aldif: Any,
    coszen: Any,
    adjes: Any,
    dyofyr: Any,
    scon: Any,
    inflgsw: int,
    iceflgsw: int,
    liqflgsw: int,
    lookup_tables: RrtmgSwCompressedLookupTables,
    cldfmcl: Any | None = None,
    taucmcl: Any | None = None,
    ssacmcl: Any | None = None,
    asmcmcl: Any | None = None,
    fsfcmcl: Any | None = None,
    ciwpmcl: Any | None = None,
    clwpmcl: Any | None = None,
    cswpmcl: Any | None = None,
    reicmcl: Any | None = None,
    relqmcl: Any | None = None,
    resnmcl: Any | None = None,
    tauaer: Any | None = None,
    ssaaer: Any | None = None,
    asmaer: Any | None = None,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgSwPadeTables | None = None,
    cloud_tables: RrtmgSwCloudTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwColumnResult:
    """Pure single-column WRF ``rrtmg_sw`` orchestration."""

    n = int(nlay)
    spec = swdatinit() if spectral is None else spectral
    inatm = inatm_sw_column(
        nlay=n,
        icld=icld,
        iaer=iaer,
        play=play,
        plev=plev,
        tlay=tlay,
        tlev=tlev,
        tsfc=tsfc,
        h2ovmr=h2ovmr,
        o3vmr=o3vmr,
        co2vmr=co2vmr,
        ch4vmr=ch4vmr,
        n2ovmr=n2ovmr,
        o2vmr=o2vmr,
        adjes=adjes,
        dyofyr=dyofyr,
        scon=scon,
        inflgsw=inflgsw,
        iceflgsw=iceflgsw,
        liqflgsw=liqflgsw,
        cldfmcl=cldfmcl,
        taucmcl=taucmcl,
        ssacmcl=ssacmcl,
        asmcmcl=asmcmcl,
        fsfcmcl=fsfcmcl,
        ciwpmcl=ciwpmcl,
        clwpmcl=clwpmcl,
        cswpmcl=cswpmcl,
        reicmcl=reicmcl,
        relqmcl=relqmcl,
        resnmcl=resnmcl,
        tauaer=tauaer,
        ssaaer=ssaaer,
        asmaer=asmaer,
        rounding_token=rounding_token,
    )

    if int(icld) == 0:
        zcldfmc = jnp.zeros((n, RRTMG_SW_NGPT), dtype=jnp.float32)
        ztaucmc = jnp.zeros((n, RRTMG_SW_NGPT), dtype=jnp.float32)
        ztaormc = jnp.zeros((n, RRTMG_SW_NGPT), dtype=jnp.float32)
        zasycmc = jnp.zeros((n, RRTMG_SW_NGPT), dtype=jnp.float32)
        zomgcmc = jnp.ones((n, RRTMG_SW_NGPT), dtype=jnp.float32)
    else:
        cloud = cldprmc_sw(
            nlayers=n,
            inflag=inatm.inflag,
            iceflag=inatm.iceflag,
            liqflag=inatm.liqflag,
            cldfmc=inatm.cldfmc,
            ciwpmc=inatm.ciwpmc,
            clwpmc=inatm.clwpmc,
            cswpmc=inatm.cswpmc,
            reicmc=inatm.reicmc,
            relqmc=inatm.relqmc,
            resnmc=inatm.resnmc,
            taucmc=inatm.taucmc,
            ssacmc=inatm.ssacmc,
            asmcmc=inatm.asmcmc,
            fsfcmc=inatm.fsfcmc,
            spectral=spec,
            cloud_tables=cloud_tables,
        )
        zcldfmc = jnp.swapaxes(inatm.cldfmc, 0, 1)
        ztaucmc = jnp.swapaxes(cloud.taucmc, 0, 1)
        ztaormc = jnp.swapaxes(cloud.taormc, 0, 1)
        zasycmc = jnp.swapaxes(cloud.asmcmc, 0, 1)
        zomgcmc = jnp.swapaxes(cloud.ssacmc, 0, 1)

    if int(iaer) == 0:
        ztaua = jnp.zeros((n, RRTMG_SW_NBND), dtype=jnp.float32)
        zasya = jnp.zeros((n, RRTMG_SW_NBND), dtype=jnp.float32)
        zomga = jnp.ones((n, RRTMG_SW_NBND), dtype=jnp.float32)
    elif int(iaer) == 10:
        ztaua = inatm.taua
        zasya = inatm.asma
        zomga = inatm.ssaa
    else:
        raise NotImplementedError(f"WRF rrtmg_sw aerosol option iaer={int(iaer)} is not translated")

    albdir = jnp.zeros((RRTMG_SW_NBND,), dtype=jnp.float32)
    albdif = jnp.zeros((RRTMG_SW_NBND,), dtype=jnp.float32)
    albdir = albdir.at[:9].set(jnp.asarray(aldir, dtype=jnp.float32))
    albdif = albdif.at[:9].set(jnp.asarray(aldif, dtype=jnp.float32))
    albdir = albdir.at[13].set(jnp.asarray(aldir, dtype=jnp.float32))
    albdif = albdif.at[13].set(jnp.asarray(aldif, dtype=jnp.float32))
    albdir = albdir.at[9:13].set(jnp.asarray(asdir, dtype=jnp.float32))
    albdif = albdif.at[9:13].set(jnp.asarray(asdif, dtype=jnp.float32))

    coef = setcoef_sw(
        nlayers=n,
        pavel=inatm.pavel,
        tavel=inatm.tavel,
        pz=inatm.pz,
        tz=inatm.tz,
        tbound=inatm.tbound,
        coldry=inatm.coldry,
        wkl=inatm.wkl,
        rounding_token=rounding_token,
    )
    cossza = jnp.maximum(jnp.asarray(coszen, dtype=jnp.float32), jnp.asarray(1.0e-10, dtype=jnp.float32))
    transfer = spcvmc_sw(
        nlayers=n,
        istart=16,
        iend=29,
        icpr=1,
        iout=0,
        pavel=inatm.pavel,
        tavel=inatm.tavel,
        pz=inatm.pz,
        tz=inatm.tz,
        tbound=inatm.tbound,
        palbd=albdif,
        palbp=albdir,
        pcldfmc=zcldfmc,
        ptaucmc=ztaucmc,
        pasycmc=zasycmc,
        pomgcmc=zomgcmc,
        ptaormc=ztaormc,
        ptaua=ztaua,
        pasya=zasya,
        pomga=zomga,
        prmu0=cossza,
        coldry=inatm.coldry,
        wkl=inatm.wkl,
        adjflux=inatm.adjflux,
        laytrop=coef.laytrop,
        layswtch=coef.layswtch,
        laylow=coef.laylow,
        jp=coef.jp,
        jt=coef.jt,
        jt1=coef.jt1,
        co2mult=coef.co2mult,
        colch4=coef.colch4,
        colco2=coef.colco2,
        colh2o=coef.colh2o,
        colmol=coef.colmol,
        coln2o=coef.coln2o,
        colo2=coef.colo2,
        colo3=coef.colo3,
        fac00=coef.fac00,
        fac01=coef.fac01,
        fac10=coef.fac10,
        fac11=coef.fac11,
        selffac=coef.selffac,
        selffrac=coef.selffrac,
        indself=coef.indself,
        forfac=coef.forfac,
        forfrac=coef.forfrac,
        indfor=coef.indfor,
        lookup_tables=lookup_tables,
        spectral=spec,
        pade_tables=pade_tables,
        rounding_token=rounding_token,
    )

    swuflx = transfer.pbbfu
    swdflx = transfer.pbbfd
    swuflxc = transfer.pbbcu
    swdflxc = transfer.pbbcd
    dirdflux = transfer.pbbfddir
    difdflux = swdflx - dirdflux
    dirdnuv = transfer.puvfddir
    difdnuv = transfer.puvfd - dirdnuv
    dirdnir = transfer.pnifddir
    difdnir = transfer.pnifd - dirdnir

    # WRF stores the net flux, pressure conversion factor, and heating rate
    # in three separate default-REAL assignments.  Keeping those boundaries
    # matters here: XLA otherwise contracts the whole expression and changes
    # the shortwave theta tendency by several ulps.
    swnflx = _wrf_real_sub(swdflx, swuflx, rounding_token)
    swnflxc = _wrf_real_sub(swdflxc, swuflxc, rounding_token)
    heatfac = jnp.asarray(spec.heatfac, dtype=jnp.float32)
    pdp = jnp.asarray(inatm.pdp, dtype=jnp.float32)[:n]
    zdpgcp = _wrf_real_div(heatfac, pdp, rounding_token)
    swhr = _wrf_real_mul(
        _wrf_real_sub(swnflx[1 : n + 1], swnflx[:n], rounding_token),
        zdpgcp,
        rounding_token,
    )
    swhrc = _wrf_real_mul(
        _wrf_real_sub(swnflxc[1 : n + 1], swnflxc[:n], rounding_token),
        zdpgcp,
        rounding_token,
    )
    swhr = swhr.at[n - 1].set(0.0)
    swhrc = swhrc.at[n - 1].set(0.0)
    zeros_flux = jnp.zeros_like(swuflx)
    return RrtmgSwColumnResult(
        swuflx=swuflx,
        swdflx=swdflx,
        swhr=swhr,
        swuflxc=swuflxc,
        swdflxc=swdflxc,
        swhrc=swhrc,
        swuflxcln=zeros_flux,
        swdflxcln=zeros_flux,
        sibvisdir=dirdnuv,
        sibvisdif=difdnuv,
        sibnirdir=dirdnir,
        sibnirdif=difdnir,
        swdkdir=dirdflux,
        swdkdif=difdflux,
        swdkdirc=transfer.pbbcddir,
        sw_zbbcddir=transfer.pbbcddir[0],
        sw_dirdflux=dirdflux[0],
        sw_difdflux=difdflux[0],
    )


def rrtmg_sw_columns(
    *,
    nlay: int,
    icld: int,
    iaer: int,
    play: Any,
    plev: Any,
    tlay: Any,
    tlev: Any,
    tsfc: Any,
    h2ovmr: Any,
    o3vmr: Any,
    co2vmr: Any,
    ch4vmr: Any,
    n2ovmr: Any,
    o2vmr: Any,
    asdir: Any,
    asdif: Any,
    aldir: Any,
    aldif: Any,
    coszen: Any,
    adjes: Any,
    dyofyr: Any,
    scon: Any,
    inflgsw: int,
    iceflgsw: int,
    liqflgsw: int,
    lookup_tables: RrtmgSwCompressedLookupTables,
    cldfmcl: Any,
    taucmcl: Any,
    ssacmcl: Any,
    asmcmcl: Any,
    fsfcmcl: Any,
    ciwpmcl: Any,
    clwpmcl: Any,
    cswpmcl: Any,
    reicmcl: Any,
    relqmcl: Any,
    resnmcl: Any,
    tauaer: Any,
    ssaaer: Any,
    asmaer: Any,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgSwPadeTables | None = None,
    cloud_tables: RrtmgSwCloudTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwColumnResult:
    """Batch WRF ``rrtmg_sw`` over horizontal columns with JAX ``vmap``."""

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0

    def _inatm_one(
        play_i,
        plev_i,
        tlay_i,
        tlev_i,
        tsfc_i,
        h2ovmr_i,
        o3vmr_i,
        co2vmr_i,
        ch4vmr_i,
        n2ovmr_i,
        o2vmr_i,
        asdir_i,
        asdif_i,
        aldir_i,
        aldif_i,
        coszen_i,
        cldfmcl_i,
        taucmcl_i,
        ssacmcl_i,
        asmcmcl_i,
        fsfcmcl_i,
        ciwpmcl_i,
        clwpmcl_i,
        cswpmcl_i,
        reicmcl_i,
        relqmcl_i,
        resnmcl_i,
        tauaer_i,
        ssaaer_i,
        asmaer_i,
    ):
        return inatm_sw_column(
            nlay=nlay,
            icld=icld,
            iaer=iaer,
            play=play_i,
            plev=plev_i,
            tlay=tlay_i,
            tlev=tlev_i,
            tsfc=tsfc_i,
            h2ovmr=h2ovmr_i,
            o3vmr=o3vmr_i,
            co2vmr=co2vmr_i,
            ch4vmr=ch4vmr_i,
            n2ovmr=n2ovmr_i,
            o2vmr=o2vmr_i,
            adjes=adjes,
            dyofyr=dyofyr,
            scon=scon,
            inflgsw=inflgsw,
            iceflgsw=iceflgsw,
            liqflgsw=liqflgsw,
            cldfmcl=cldfmcl_i,
            taucmcl=taucmcl_i,
            ssacmcl=ssacmcl_i,
            asmcmcl=asmcmcl_i,
            fsfcmcl=fsfcmcl_i,
            ciwpmcl=ciwpmcl_i,
            clwpmcl=clwpmcl_i,
            cswpmcl=cswpmcl_i,
            reicmcl=reicmcl_i,
            relqmcl=relqmcl_i,
            resnmcl=resnmcl_i,
            tauaer=tauaer_i,
            ssaaer=ssaaer_i,
            asmaer=asmaer_i,
            rounding_token=rounding_token,
        )

    inatm = jax.vmap(_inatm_one)(
        play,
        plev,
        tlay,
        tlev,
        tsfc,
        h2ovmr,
        o3vmr,
        co2vmr,
        ch4vmr,
        n2ovmr,
        o2vmr,
        asdir,
        asdif,
        aldir,
        aldif,
        coszen,
        cldfmcl,
        taucmcl,
        ssacmcl,
        asmcmcl,
        fsfcmcl,
        ciwpmcl,
        clwpmcl,
        cswpmcl,
        reicmcl,
        relqmcl,
        resnmcl,
        tauaer,
        ssaaer,
        asmaer,
    )
    _profile_rrtmg_sw_write("columns_inatm", profile_start, inatm)

    n = int(nlay)
    ncol = int(jnp.asarray(play).shape[0])
    spec = swdatinit() if spectral is None else spectral

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0
    if int(icld) == 0:
        zcldfmc = jnp.zeros((ncol, n, RRTMG_SW_NGPT), dtype=jnp.float32)
        ztaucmc = jnp.zeros((ncol, n, RRTMG_SW_NGPT), dtype=jnp.float32)
        ztaormc = jnp.zeros((ncol, n, RRTMG_SW_NGPT), dtype=jnp.float32)
        zasycmc = jnp.zeros((ncol, n, RRTMG_SW_NGPT), dtype=jnp.float32)
        zomgcmc = jnp.ones((ncol, n, RRTMG_SW_NGPT), dtype=jnp.float32)
    else:
        cloud_inflag = int(inflgsw) if int(icld) >= 1 else 0
        cloud_iceflag = int(iceflgsw) if int(icld) >= 1 else 0
        cloud_liqflag = int(liqflgsw) if int(icld) >= 1 else 0

        def _cloud_one(cldf_i, ciwp_i, clwp_i, cswp_i, rei_i, rel_i, res_i, tau_i, ssa_i, asm_i, fsf_i):
            return cldprmc_sw(
                nlayers=n,
                inflag=cloud_inflag,
                iceflag=cloud_iceflag,
                liqflag=cloud_liqflag,
                cldfmc=cldf_i,
                ciwpmc=ciwp_i,
                clwpmc=clwp_i,
                cswpmc=cswp_i,
                reicmc=rei_i,
                relqmc=rel_i,
                resnmc=res_i,
                taucmc=tau_i,
                ssacmc=ssa_i,
                asmcmc=asm_i,
                fsfcmc=fsf_i,
                spectral=spec,
                cloud_tables=cloud_tables,
            )

        cloud = jax.vmap(_cloud_one)(
            inatm.cldfmc,
            inatm.ciwpmc,
            inatm.clwpmc,
            inatm.cswpmc,
            inatm.reicmc,
            inatm.relqmc,
            inatm.resnmc,
            inatm.taucmc,
            inatm.ssacmc,
            inatm.asmcmc,
            inatm.fsfcmc,
        )
        zcldfmc = jnp.swapaxes(inatm.cldfmc, 1, 2)
        ztaucmc = jnp.swapaxes(cloud.taucmc, 1, 2)
        ztaormc = jnp.swapaxes(cloud.taormc, 1, 2)
        zasycmc = jnp.swapaxes(cloud.asmcmc, 1, 2)
        zomgcmc = jnp.swapaxes(cloud.ssacmc, 1, 2)
    _profile_rrtmg_sw_write("columns_cloud", profile_start, (zcldfmc, ztaucmc, ztaormc, zasycmc, zomgcmc))

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0
    if int(iaer) == 0:
        ztaua = None
        zasya = None
        zomga = None
    elif int(iaer) == 10:
        ztaua = inatm.taua
        zasya = inatm.asma
        zomga = inatm.ssaa
    else:
        raise NotImplementedError(f"WRF rrtmg_sw aerosol option iaer={int(iaer)} is not translated")

    asdir_arr = jnp.asarray(asdir, dtype=jnp.float32)
    asdif_arr = jnp.asarray(asdif, dtype=jnp.float32)
    aldir_arr = jnp.asarray(aldir, dtype=jnp.float32)
    aldif_arr = jnp.asarray(aldif, dtype=jnp.float32)
    albdir = jnp.zeros((ncol, RRTMG_SW_NBND), dtype=jnp.float32)
    albdif = jnp.zeros((ncol, RRTMG_SW_NBND), dtype=jnp.float32)
    albdir = albdir.at[:, :9].set(aldir_arr[:, None])
    albdif = albdif.at[:, :9].set(aldif_arr[:, None])
    albdir = albdir.at[:, 13].set(aldir_arr)
    albdif = albdif.at[:, 13].set(aldif_arr)
    albdir = albdir.at[:, 9:13].set(asdir_arr[:, None])
    albdif = albdif.at[:, 9:13].set(asdif_arr[:, None])
    _profile_rrtmg_sw_write("columns_aerosol_albedo", profile_start, (ztaua, zasya, zomga, albdir, albdif))

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0

    coef = setcoef_sw(
        nlayers=n,
        pavel=inatm.pavel,
        tavel=inatm.tavel,
        pz=inatm.pz,
        tz=inatm.tz,
        tbound=inatm.tbound,
        coldry=inatm.coldry,
        wkl=inatm.wkl,
        rounding_token=rounding_token,
    )
    _profile_rrtmg_sw_write("columns_setcoef", profile_start, coef)

    cossza = jnp.maximum(jnp.asarray(coszen, dtype=jnp.float32), jnp.asarray(1.0e-10, dtype=jnp.float32))

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0

    def _taumol_one(
        laytrop_i,
        jp_i,
        jt_i,
        jt1_i,
        colch4_i,
        colco2_i,
        colh2o_i,
        colmol_i,
        colo2_i,
        colo3_i,
        fac00_i,
        fac01_i,
        fac10_i,
        fac11_i,
        selffac_i,
        selffrac_i,
        indself_i,
        forfac_i,
        forfrac_i,
        indfor_i,
    ):
        return taumol_sw(
            nlayers=n,
            colh2o=colh2o_i,
            colco2=colco2_i,
            colch4=colch4_i,
            colo2=colo2_i,
            colo3=colo3_i,
            colmol=colmol_i,
            laytrop=laytrop_i,
            jp=jp_i,
            jt=jt_i,
            jt1=jt1_i,
            fac00=fac00_i,
            fac01=fac01_i,
            fac10=fac10_i,
            fac11=fac11_i,
            selffac=selffac_i,
            selffrac=selffrac_i,
            indself=indself_i,
            forfac=forfac_i,
            forfrac=forfrac_i,
            indfor=indfor_i,
            tables=lookup_tables,
            rounding_token=rounding_token,
        )

    optical = jax.vmap(_taumol_one)(
        coef.laytrop,
        coef.jp,
        coef.jt,
        coef.jt1,
        coef.colch4,
        coef.colco2,
        coef.colh2o,
        coef.colmol,
        coef.colo2,
        coef.colo3,
        coef.fac00,
        coef.fac01,
        coef.fac10,
        coef.fac11,
        coef.selffac,
        coef.selffrac,
        coef.indself,
        coef.forfac,
        coef.forfrac,
        coef.indfor,
    )
    _profile_rrtmg_sw_write("columns_taumol", profile_start, optical)

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0
    transfer = spcvmc_sw_from_optical_depths_columns(
        nlayers=n,
        istart=16,
        iend=29,
        icpr=1,
        iout=0,
        palbd=albdif,
        palbp=albdir,
        pcldfmc=zcldfmc,
        ptaucmc=ztaucmc,
        pasycmc=zasycmc,
        pomgcmc=zomgcmc,
        ptaormc=ztaormc,
        ptaua=ztaua,
        pasya=zasya,
        pomga=zomga,
        prmu0=cossza,
        adjflux=inatm.adjflux,
        sfluxzen=optical.sfluxzen,
        taug=optical.taug,
        taur=optical.taur,
        spectral=spec,
        tables=pade_tables,
        rounding_token=rounding_token,
        clear_only=int(icld) == 0,
    )
    _profile_rrtmg_sw_write("columns_spcvmc_transport", profile_start, transfer)
    if _CAPTURE_RRTMG_SW_TARGET_PATH:
        jax.debug.callback(
            _capture_rrtmg_sw_target_host,
            coszen,
            inatm.pavel,
            inatm.pz,
            inatm.tavel,
            inatm.tz,
            inatm.coldry,
            inatm.wkl,
            coef.laytrop,
            coef.jp,
            coef.jt,
            coef.jt1,
            coef.colo3,
            optical.sfluxzen,
            optical.taug,
            optical.taur,
            zcldfmc,
            transfer.pbbfd,
            transfer.pbbfu,
            ordered=True,
        )

    profile_start = perf_counter() if _PROFILE_RRTMG_SW_PATH else 0.0
    swuflx = transfer.pbbfu
    swdflx = transfer.pbbfd
    swuflxc = transfer.pbbcu
    swdflxc = transfer.pbbcd
    dirdflux = transfer.pbbfddir
    difdflux = swdflx - dirdflux
    dirdnuv = transfer.puvfddir
    difdnuv = transfer.puvfd - dirdnuv
    dirdnir = transfer.pnifddir
    difdnir = transfer.pnifd - dirdnir

    swnflx = _wrf_real_sub(swdflx, swuflx, rounding_token)
    swnflxc = _wrf_real_sub(swdflxc, swuflxc, rounding_token)
    heatfac = jnp.asarray(spec.heatfac, dtype=jnp.float32)
    pdp = jnp.asarray(inatm.pdp, dtype=jnp.float32)[:, :n]
    zdpgcp = _wrf_real_div(heatfac, pdp, rounding_token)
    swhr = _wrf_real_mul(
        _wrf_real_sub(swnflx[:, 1 : n + 1], swnflx[:, :n], rounding_token),
        zdpgcp,
        rounding_token,
    )
    swhrc = _wrf_real_mul(
        _wrf_real_sub(swnflxc[:, 1 : n + 1], swnflxc[:, :n], rounding_token),
        zdpgcp,
        rounding_token,
    )
    swhr = swhr.at[:, n - 1].set(0.0)
    swhrc = swhrc.at[:, n - 1].set(0.0)
    zeros_flux = jnp.zeros_like(swuflx)
    result = RrtmgSwColumnResult(
        swuflx=swuflx,
        swdflx=swdflx,
        swhr=swhr,
        swuflxc=swuflxc,
        swdflxc=swdflxc,
        swhrc=swhrc,
        swuflxcln=zeros_flux,
        swdflxcln=zeros_flux,
        sibvisdir=dirdnuv,
        sibvisdif=difdnuv,
        sibnirdir=dirdnir,
        sibnirdif=difdnir,
        swdkdir=dirdflux,
        swdkdif=difdflux,
        swdkdirc=transfer.pbbcddir,
        sw_zbbcddir=transfer.pbbcddir[:, 0],
        sw_dirdflux=dirdflux[:, 0],
        sw_difdflux=difdflux[:, 0],
    )
    _profile_rrtmg_sw_write("columns_outputs", profile_start, result)
    return result


def _flatten_wrf_mass_columns(field: Any) -> Any:
    arr = jnp.asarray(field, dtype=jnp.float32)
    return jnp.reshape(jnp.swapaxes(arr, 1, 2), (-1, arr.shape[1]))


def _flatten_wrf_2d(field: Any) -> Any:
    arr = jnp.asarray(field, dtype=jnp.float32)
    return jnp.reshape(arr, (-1,))


def _unflatten_wrf_2d(values: Any, *, nx: int, ny: int) -> Any:
    return jnp.reshape(jnp.asarray(values, dtype=jnp.float32), (nx, ny))


def _unflatten_wrf_mass_columns(values: Any, *, nx: int, ny: int) -> Any:
    arr = jnp.reshape(jnp.asarray(values, dtype=jnp.float32), (nx, ny, -1))
    return jnp.swapaxes(arr, 1, 2)


def _extend_top_mass(values: Any, top_values: Any) -> Any:
    return jnp.concatenate(
        (
            jnp.asarray(values, dtype=jnp.float32),
            jnp.asarray(top_values, dtype=jnp.float32)[:, None],
        ),
        axis=1,
    )


def _sw_cloud_flags(*, has_reqc: int, has_reqi: int, has_reqs: int, icloud: int) -> tuple[int, int, int]:
    inflg = 2
    iceflg = 3
    liqflg = 1
    if int(icloud) != 0:
        if int(has_reqc) != 0:
            inflg = 3
        if int(has_reqi) != 0:
            inflg = 4
            iceflg = 4
        if int(has_reqs) != 0:
            inflg = 5
            iceflg = 5
        if int(has_reqs) == 0 and int(has_reqi) != 0 and int(has_reqc) != 0:
            inflg = 5
            iceflg = 5
    return inflg, iceflg, liqflg


def _sw_greenhouse_scalars(*, yr: int, ghg_input: int) -> tuple[float, float, float]:
    if int(ghg_input) == 0:
        co2 = (280.0 + 90.0 * np.exp(0.02 * (int(yr) - 2000))) * 1.0e-6
    else:
        # Runtime callers that need CAM greenhouse interpolation can pass the
        # exact scalars later; this default matches WRF's common present-day
        # magnitude and keeps the JAX adapter self-contained for unit tests.
        co2 = 379.0e-6
    return float(co2), 319.0e-9, 1774.0e-9


def rrtmg_swrad_wrf_columns(
    *,
    p3d: Any,
    p8w: Any,
    pi3d: Any,
    t3d: Any,
    t8w: Any,
    tsk: Any,
    dz8w: Any,
    qv3d: Any,
    qc3d: Any,
    qi3d: Any,
    qs3d: Any,
    cldfra3d: Any,
    o33d: Any,
    albedo: Any,
    coszr: Any,
    xland: Any,
    xice: Any,
    snow: Any,
    re_cloud: Any,
    re_ice: Any,
    re_snow: Any,
    solcon: Any,
    yr: int,
    julian: Any,
    ghg_input: int = 1,
    cldovrlp: int = 2,
    idcor: int = 0,
    icloud: int = 1,
    has_reqc: int = 1,
    has_reqi: int = 1,
    has_reqs: int = 1,
    sf_surface_physics: int = 2,
    alswvisdir: Any | None = None,
    alswvisdif: Any | None = None,
    alswnirdir: Any | None = None,
    alswnirdif: Any | None = None,
    co2vmr: float | None = None,
    n2ovmr: float | None = None,
    ch4vmr: float | None = None,
    lookup_tables: RrtmgSwCompressedLookupTables | None = None,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgSwPadeTables | None = None,
    cloud_tables: RrtmgSwCloudTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwWrfColumnResult:
    """Batched JAX translation of WRF ``rrtmg_swrad`` for one model domain.

    Arrays use WRF-JAX runtime layout ``(west_east, bottom_top, south_north)``
    for three-dimensional fields and ``(west_east, south_north)`` for surface
    fields. The horizontal dimension is flattened into independent atmospheric
    columns before calling the pure RRTMG shortwave column kernel.
    """

    t3d_arr = jnp.asarray(t3d, dtype=jnp.float32)
    nx = int(t3d_arr.shape[0])
    nk = int(t3d_arr.shape[1])
    ny = int(t3d_arr.shape[2])
    nlay = nk + 1
    ncol = nx * ny

    lookup = load_rrtmg_sw_compressed_lookup_tables() if lookup_tables is None else lookup_tables
    clouds = load_rrtmg_sw_cloud_tables() if cloud_tables is None else cloud_tables
    spec = swdatinit() if spectral is None else spectral
    pade = rrtmg_sw_pade_tables() if pade_tables is None else pade_tables

    p3d_c = _wrf_real_pa_to_hpa(_flatten_wrf_mass_columns(p3d))
    p8w_c = _wrf_real_pa_to_hpa(_flatten_wrf_mass_columns(p8w))
    t3d_c = _flatten_wrf_mass_columns(t3d_arr)
    t8w_c = _flatten_wrf_mass_columns(t8w)
    pi3d_c = _flatten_wrf_mass_columns(pi3d)
    qv_c = jnp.maximum(_flatten_wrf_mass_columns(qv3d), jnp.float32(1.0e-12))
    qc_c = jnp.maximum(_flatten_wrf_mass_columns(qc3d), jnp.float32(0.0))
    qi_c = jnp.maximum(_flatten_wrf_mass_columns(qi3d), jnp.float32(0.0))
    qs_c = jnp.maximum(_flatten_wrf_mass_columns(qs3d), jnp.float32(0.0))
    cldfra_c = _flatten_wrf_mass_columns(cldfra3d)
    o33d_c = _flatten_wrf_mass_columns(o33d)
    re_cloud_c = _flatten_wrf_mass_columns(re_cloud)
    re_ice_c = _flatten_wrf_mass_columns(re_ice)
    re_snow_c = _flatten_wrf_mass_columns(re_snow)
    dz8w_c = _flatten_wrf_mass_columns(dz8w)
    del dz8w_c

    plev = jnp.concatenate((p8w_c[:, : nk + 1], jnp.full((ncol, 1), 1.0e-5, dtype=jnp.float32)), axis=1)
    tlev_top = t8w_c[:, nk]
    tlev = jnp.concatenate((t8w_c[:, : nk + 1], tlev_top[:, None]), axis=1)
    play = _extend_top_mass(p3d_c[:, :nk], 0.5 * p8w_c[:, nk])
    tlay = _extend_top_mass(t3d_c[:, :nk], tlev_top)
    pdel = plev[:, :nk] - plev[:, 1 : nk + 1]

    h2ovmr = _extend_top_mass(qv_c[:, :nk] * jnp.float32(1.607793), qv_c[:, nk - 1] * jnp.float32(1.607793))
    if co2vmr is not None and n2ovmr is not None and ch4vmr is not None:
        co2_scalar = jnp.asarray(co2vmr, dtype=jnp.float32)
        n2o_scalar = jnp.asarray(n2ovmr, dtype=jnp.float32)
        ch4_scalar = jnp.asarray(ch4vmr, dtype=jnp.float32)
    else:
        co2_default, n2o_default, ch4_default = _sw_greenhouse_scalars(yr=int(yr), ghg_input=int(ghg_input))
        co2_scalar = co2_default if co2vmr is None else float(co2vmr)
        n2o_scalar = n2o_default if n2ovmr is None else float(n2ovmr)
        ch4_scalar = ch4_default if ch4vmr is None else float(ch4vmr)
    co2 = jnp.full((ncol, nlay), co2_scalar, dtype=jnp.float32)
    n2o = jnp.full((ncol, nlay), n2o_scalar, dtype=jnp.float32)
    ch4 = jnp.full((ncol, nlay), ch4_scalar, dtype=jnp.float32)
    o2 = jnp.full((ncol, nlay), 0.209488, dtype=jnp.float32)

    o3mmr = jax.vmap(lambda col_plev: inirad(col_plev, kte=nk))(plev)
    amdo = jnp.float32(0.603461)
    o3_top = _wrf_real_add(
        _wrf_real_sub(
            o33d_c[:, nk - 1],
            _wrf_real_mul(o3mmr[:, nk - 1], amdo, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(o3mmr[:, nk], amdo, rounding_token),
        rounding_token,
    )
    o3_top = jnp.where(
        o3_top <= 0.0,
        _wrf_real_mul(o3mmr[:, nk], amdo, rounding_token),
        o3_top,
    )
    o3vmr = _extend_top_mass(o33d_c[:, :nk], o3_top)
    if _CAPTURE_RRTMG_SW_TARGET_PATH:
        jax.debug.callback(
            _capture_rrtmg_sw_o3_context_host,
            _flatten_wrf_2d(coszr),
            plev,
            o3mmr,
            o3vmr,
            ordered=True,
        )

    surface_albedo = _flatten_wrf_2d(albedo)
    xland_c = _flatten_wrf_2d(xland)
    use_sf8 = (int(sf_surface_physics) == 8) & (alswvisdir is not None) & (alswvisdif is not None) & (alswnirdir is not None) & (alswnirdif is not None)
    if use_sf8:
        water = xland_c < 1.5
        asdir = jnp.where(water, _flatten_wrf_2d(alswvisdir), surface_albedo)
        asdif = jnp.where(water, _flatten_wrf_2d(alswvisdif), surface_albedo)
        aldir = jnp.where(water, _flatten_wrf_2d(alswnirdir), surface_albedo)
        aldif = jnp.where(water, _flatten_wrf_2d(alswnirdif), surface_albedo)
    else:
        asdir = surface_albedo
        asdif = surface_albedo
        aldir = surface_albedo
        aldif = surface_albedo

    inflgsw, iceflgsw, liqflgsw = _sw_cloud_flags(
        has_reqc=int(has_reqc),
        has_reqi=int(has_reqi),
        has_reqs=int(has_reqs),
        icloud=int(icloud),
    )

    radii = jax.vmap(
        lambda t, cld, land, qc, qi, qs, rec, rei, res: prepare_cloud_radii(
            t3d=t,
            cldfra3d=cld,
            xland=land,
            qc3d=qc,
            qi3d=qi,
            qs3d=qs,
            re_cloud=rec,
            re_ice=rei,
            re_snow=res,
            has_reqc=int(has_reqc),
            has_reqi=int(has_reqi),
            has_reqs=int(has_reqs),
            icloud=int(icloud),
        )
    )(t3d_c[:, :nk], cldfra_c[:, :nk], xland_c, qc_c[:, :nk], qi_c[:, :nk], qs_c[:, :nk], re_cloud_c[:, :nk], re_ice_c[:, :nk], re_snow_c[:, :nk])

    xice_c = _flatten_wrf_2d(xice)
    snow_c = _flatten_wrf_2d(snow)
    paths = jax.vmap(
        lambda t, dp, cld, qc, qi, qs, land, ice, sn, rec, rei, res: prepare_cloud_path_inputs(
            tlay=t,
            pdel=dp,
            cldfra=cld,
            qc=qc,
            qi=qi,
            qs=qs,
            xland=land,
            xice=ice,
            snow=sn,
            recloud=rec,
            reice=rei,
            resnow=res,
            inflg=inflgsw,
            iceflg=iceflgsw,
            g=9.8066,
        )
    )(tlay[:, :nk], pdel, cldfra_c[:, :nk], radii.qc, radii.qi, radii.qs, xland_c, xice_c, snow_c, radii.recloud, radii.reice, radii.resnow)

    zero_top = jnp.zeros((ncol, 1), dtype=jnp.float32)
    ten_top = jnp.full((ncol, 1), 10.0, dtype=jnp.float32)
    cldfrac = jnp.concatenate((paths.cldfrac, zero_top), axis=1)
    clwpth = jnp.concatenate((paths.clwpth, zero_top), axis=1)
    ciwpth = jnp.concatenate((paths.ciwpth, zero_top), axis=1)
    cswpth = jnp.concatenate((paths.cswpth, zero_top), axis=1)
    rel = jnp.concatenate((paths.rel, ten_top), axis=1)
    rei = jnp.concatenate((paths.rei, ten_top), axis=1)
    res = jnp.concatenate((paths.res, ten_top), axis=1)

    # MCICA broadcasts band constants over columns/layers after spectral lookup.
    # Keeping these axes singleton avoids folding a full-grid constant gather.
    taucld = jnp.zeros((RRTMG_SW_NBND, 1, 1), dtype=jnp.float32)
    ssacld = jnp.ones((RRTMG_SW_NBND, 1, 1), dtype=jnp.float32)
    asmcld = jnp.zeros((RRTMG_SW_NBND, 1, 1), dtype=jnp.float32)
    fsfcld = jnp.zeros((RRTMG_SW_NBND, 1, 1), dtype=jnp.float32)
    mcica = mcica_subcol_sw_overlap2(
        play=play,
        cldfrac=cldfrac,
        ciwp=ciwpth,
        clwp=clwpth,
        cswp=cswpth,
        rei=rei,
        rel=rel,
        res=res,
        tauc=taucld,
        ssac=ssacld,
        asmc=asmcld,
        fsfc=fsfcld,
        change_seed=1,
        ngb=NGB_SW_LOCAL,
    )

    tauaer = jnp.zeros((ncol,), dtype=jnp.float32)
    ssaaer = jnp.ones((ncol,), dtype=jnp.float32)
    asmaer = jnp.zeros((ncol,), dtype=jnp.float32)
    coszen = _flatten_wrf_2d(coszr)
    day = coszen > 0.0

    column = rrtmg_sw_columns(
        nlay=nlay,
        icld=0 if int(icloud) == 0 else int(cldovrlp),
        iaer=0,
        play=play,
        plev=plev,
        tlay=tlay,
        tlev=tlev,
        tsfc=_flatten_wrf_2d(tsk),
        h2ovmr=h2ovmr,
        o3vmr=o3vmr,
        co2vmr=co2,
        ch4vmr=ch4,
        n2ovmr=n2o,
        o2vmr=o2,
        asdir=asdir,
        asdif=asdif,
        aldir=aldir,
        aldif=aldif,
        coszen=coszen,
        adjes=jnp.asarray(1.0, dtype=jnp.float32),
        dyofyr=jnp.asarray(0, dtype=jnp.int32),
        scon=jnp.asarray(solcon, dtype=jnp.float32),
        inflgsw=inflgsw,
        iceflgsw=iceflgsw,
        liqflgsw=liqflgsw,
        lookup_tables=lookup,
        cldfmcl=jnp.swapaxes(mcica.cldfmcl, 0, 1),
        taucmcl=jnp.swapaxes(mcica.taucmcl, 0, 1),
        ssacmcl=jnp.swapaxes(mcica.ssacmcl, 0, 1),
        asmcmcl=jnp.swapaxes(mcica.asmcmcl, 0, 1),
        fsfcmcl=jnp.swapaxes(mcica.fsfcmcl, 0, 1),
        ciwpmcl=jnp.swapaxes(mcica.ciwpmcl, 0, 1),
        clwpmcl=jnp.swapaxes(mcica.clwpmcl, 0, 1),
        cswpmcl=jnp.swapaxes(mcica.cswpmcl, 0, 1),
        reicmcl=mcica.reicmcl,
        relqmcl=mcica.relqmcl,
        resnmcl=mcica.resnmcl,
        tauaer=tauaer,
        ssaaer=ssaaer,
        asmaer=asmaer,
        spectral=spec,
        pade_tables=pade,
        cloud_tables=clouds,
        rounding_token=rounding_token,
    )

    def mask2(values: Any) -> Any:
        return jnp.where(day, jnp.asarray(values, dtype=jnp.float32), 0.0)

    def mask3(values: Any) -> Any:
        return jnp.where(day[:, None], jnp.asarray(values, dtype=jnp.float32), 0.0)

    tten1d = _wrf_real_div(column.swhr[:, :nk], jnp.float32(86400.0), rounding_token)
    tten1dc = _wrf_real_div(column.swhrc[:, :nk], jnp.float32(86400.0), rounding_token)
    rthratensw_c = mask3(_wrf_real_div(tten1d, pi3d_c[:, :nk], rounding_token))
    rthratenswc_c = mask3(_wrf_real_div(tten1dc, pi3d_c[:, :nk], rounding_token))
    swupt = mask2(column.swuflx[:, -1])
    swuptc = mask2(column.swuflxc[:, -1])
    swdnt = mask2(column.swdflx[:, -1])
    swdntc = mask2(column.swdflxc[:, -1])
    swupb = mask2(column.swuflx[:, 0])
    swupbc = mask2(column.swuflxc[:, 0])
    swdnb = mask2(column.swdflx[:, 0])
    swdnbc = mask2(column.swdflxc[:, 0])
    swcf = mask2((column.swdflx[:, -1] - column.swuflx[:, -1]) - (column.swdflxc[:, -1] - column.swuflxc[:, -1]))
    gsw = mask2(column.swdflx[:, 0] - column.swuflx[:, 0])
    swddir = mask2(column.swdkdir[:, 0])
    swddni = mask2(jnp.where(coszen > 0.0, column.swdkdir[:, 0] / coszen, 0.0))
    swddif = mask2(column.swdkdif[:, 0])
    swdownc = mask2(column.swdflxc[:, 0])
    swddirc = mask2(column.swdkdirc[:, 0])
    swddnic = mask2(jnp.where(coszen > 0.0, column.swdkdirc[:, 0] / coszen, 0.0))
    zeros2 = jnp.zeros((ncol,), dtype=jnp.float32)

    return RrtmgSwWrfColumnResult(
        rthratensw=_unflatten_wrf_mass_columns(rthratensw_c, nx=nx, ny=ny),
        rthratenswc=_unflatten_wrf_mass_columns(rthratenswc_c, nx=nx, ny=ny),
        swupt=_unflatten_wrf_2d(swupt, nx=nx, ny=ny),
        swuptc=_unflatten_wrf_2d(swuptc, nx=nx, ny=ny),
        swuptcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        swdnt=_unflatten_wrf_2d(swdnt, nx=nx, ny=ny),
        swdntc=_unflatten_wrf_2d(swdntc, nx=nx, ny=ny),
        swdntcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        swupb=_unflatten_wrf_2d(swupb, nx=nx, ny=ny),
        swupbc=_unflatten_wrf_2d(swupbc, nx=nx, ny=ny),
        swupbcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        swdnb=_unflatten_wrf_2d(swdnb, nx=nx, ny=ny),
        swdnbc=_unflatten_wrf_2d(swdnbc, nx=nx, ny=ny),
        swdnbcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        swcf=_unflatten_wrf_2d(swcf, nx=nx, ny=ny),
        gsw=_unflatten_wrf_2d(gsw, nx=nx, ny=ny),
        swddir=_unflatten_wrf_2d(swddir, nx=nx, ny=ny),
        swddni=_unflatten_wrf_2d(swddni, nx=nx, ny=ny),
        swddif=_unflatten_wrf_2d(swddif, nx=nx, ny=ny),
        swdownc=_unflatten_wrf_2d(swdownc, nx=nx, ny=ny),
        swddnic=_unflatten_wrf_2d(swddnic, nx=nx, ny=ny),
        swddirc=_unflatten_wrf_2d(swddirc, nx=nx, ny=ny),
        coszr=_unflatten_wrf_2d(coszen, nx=nx, ny=ny),
    )


def _reftra_sw_wrf_real(
    *,
    lrt: Any,
    zg: Any,
    zto1: Any,
    zw: Any,
    prmuz_arr: Any,
    tables: RrtmgSwPadeTables,
    rounding_token: Any,
) -> RrtmgSwReftraResult:
    """Evaluate each independent layer/g-point without global intermediates."""

    def body(inputs, parameters):
        exp_tbl, bpade, token = parameters
        return _reftra_sw_real_values(
            lrt=inputs[0], zg=inputs[1], zto1=inputs[2], zw=inputs[3],
            prmuz_arr=inputs[4], exp_tbl=exp_tbl, bpade=bpade, rounding_token=token,
        )

    inputs = tuple(jnp.asarray(value) for value in (lrt, zg, zto1, zw, prmuz_arr))
    return map_elements(body, inputs, (
        jnp.asarray(tables.exp_tbl, jnp.float32),
        jnp.asarray(tables.bpade, jnp.float32), rounding_token,
    ))


def _reftra_sw_real_values(
    *, lrt, zg, zto1, zw, prmuz_arr, exp_tbl, bpade, rounding_token,
) -> RrtmgSwReftraResult:
    """Shared ``reftra_sw`` equations with WRF REAL assignment boundaries."""
    from wrf_jax.core.real_arithmetic import (
        real_add as _wrf_real_add,
        real_subtract as _wrf_real_sub,
        real_multiply as _wrf_real_mul,
        real_divide_assigned as _wrf_real_div,
        real_assignment as _wrf_real_store,
    )

    zero = jnp.float32(0.0)
    one = jnp.float32(1.0)
    eps = np.float32(1.0e-8)
    zwcrit = jnp.float32(0.9999995)

    # Non-radiating layers return identity transfer, but their intermediates
    # must remain defined when the reverse pass sends them zero cotangents.
    zg = jnp.where(lrt, zg, zero)
    zto1 = jnp.where(lrt, zto1, zero)
    zw = jnp.where(lrt, zw, zero)
    prmuz_arr = jnp.where(lrt, prmuz_arr, one)

    zg3 = _wrf_real_mul(jnp.float32(3.0), zg, rounding_token)
    zgamma1 = _wrf_real_mul(
        _wrf_real_sub(
            jnp.float32(8.0),
            _wrf_real_mul(
                zw,
                _wrf_real_add(jnp.float32(5.0), zg3, rounding_token),
                rounding_token,
            ),
            rounding_token,
        ),
        jnp.float32(0.25),
        rounding_token,
    )
    zgamma2 = _wrf_real_mul(
        _wrf_real_mul(
            jnp.float32(3.0),
            _wrf_real_mul(
                zw,
                _wrf_real_sub(one, zg, rounding_token),
                rounding_token,
            ),
            rounding_token,
        ),
        jnp.float32(0.25),
        rounding_token,
    )
    zgamma3 = _wrf_real_mul(
        _wrf_real_sub(
            jnp.float32(2.0),
            _wrf_real_mul(zg3, prmuz_arr, rounding_token),
            rounding_token,
        ),
        jnp.float32(0.25),
        rounding_token,
    )
    zgamma4 = _wrf_real_sub(one, zgamma3, rounding_token)

    one_minus_g = _wrf_real_sub(one, zg, rounding_token)
    ratio = _wrf_real_div(zg, jnp.where(zg != one, one_minus_g, one), rounding_token)
    ratio = jnp.where(zg != one, ratio, zero)
    denom = _wrf_real_sub(
        one,
        _wrf_real_mul(
            _wrf_real_sub(one, zw, rounding_token),
            _wrf_real_mul(ratio, ratio, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    denom = jnp.where(zg != one, denom, one)
    zwo_active = (zw > zero) & (denom != zero)
    zwo = jnp.where(
        zwo_active,
        _wrf_real_div(zw, jnp.where(zwo_active, denom, one), rounding_token),
        zero,
    )
    conservative = zwo >= zwcrit

    za = _wrf_real_mul(zgamma1, prmuz_arr, rounding_token)
    za1_cons = _wrf_real_sub(za, zgamma3, rounding_token)
    zgt = _wrf_real_mul(zgamma1, zto1, rounding_token)
    ze_cons = jnp.minimum(
        _wrf_real_div(zto1, prmuz_arr, rounding_token),
        jnp.float32(500.0),
    )
    exp_cons = _sw_exp_lookup_real(ze_cons, exp_tbl, bpade, rounding_token)
    pref_cons = _wrf_real_div(
        _wrf_real_sub(
            zgt,
            _wrf_real_mul(
                za1_cons,
                _wrf_real_sub(one, exp_cons, rounding_token),
                rounding_token,
            ),
            rounding_token,
        ),
        _wrf_real_add(one, zgt, rounding_token),
        rounding_token,
    )
    ptra_cons = _wrf_real_sub(one, pref_cons, rounding_token)
    prefd_cons = _wrf_real_div(
        zgt,
        _wrf_real_add(one, zgt, rounding_token),
        rounding_token,
    )
    ptrad_cons = _wrf_real_sub(one, prefd_cons, rounding_token)
    clear_cons = exp_cons == one
    pref_cons = jnp.where(clear_cons, zero, pref_cons)
    ptra_cons = jnp.where(clear_cons, one, ptra_cons)
    prefd_cons = jnp.where(clear_cons, zero, prefd_cons)
    ptrad_cons = jnp.where(clear_cons, one, ptrad_cons)

    za1 = _wrf_real_add(
        _wrf_real_mul(zgamma1, zgamma4, rounding_token),
        _wrf_real_mul(zgamma2, zgamma3, rounding_token),
        rounding_token,
    )
    za2 = _wrf_real_add(
        _wrf_real_mul(zgamma1, zgamma3, rounding_token),
        _wrf_real_mul(zgamma2, zgamma4, rounding_token),
        rounding_token,
    )
    zrk_radicand = _wrf_real_sub(
        _wrf_real_mul(zgamma1, zgamma1, rounding_token),
        _wrf_real_mul(zgamma2, zgamma2, rounding_token),
        rounding_token,
    )
    zrk = _wrf_real_store(sqrt_for_wrf(jnp.where(conservative, one, zrk_radicand)), rounding_token)
    zrp = _wrf_real_mul(zrk, prmuz_arr, rounding_token)
    zrp1 = _wrf_real_add(one, zrp, rounding_token)
    zrm1 = _wrf_real_sub(one, zrp, rounding_token)
    zrk2 = _wrf_real_mul(jnp.float32(2.0), zrk, rounding_token)
    zrpp = _wrf_real_sub(
        one,
        _wrf_real_mul(zrp, zrp, rounding_token),
        rounding_token,
    )
    zrkg = _wrf_real_add(zrk, zgamma1, rounding_token)
    zr1 = _wrf_real_mul(
        zrm1,
        _wrf_real_add(
            za2,
            _wrf_real_mul(zrk, zgamma3, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zr2 = _wrf_real_mul(
        zrp1,
        _wrf_real_sub(
            za2,
            _wrf_real_mul(zrk, zgamma3, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zr3 = _wrf_real_mul(
        zrk2,
        _wrf_real_sub(
            zgamma3,
            _wrf_real_mul(za2, prmuz_arr, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zr4 = _wrf_real_mul(zrpp, zrkg, rounding_token)
    zr5 = _wrf_real_mul(
        zrpp,
        _wrf_real_sub(zrk, zgamma1, rounding_token),
        rounding_token,
    )
    zt1 = _wrf_real_mul(
        zrp1,
        _wrf_real_add(
            za1,
            _wrf_real_mul(zrk, zgamma4, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zt2 = _wrf_real_mul(
        zrm1,
        _wrf_real_sub(
            za1,
            _wrf_real_mul(zrk, zgamma4, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zt3 = _wrf_real_mul(
        zrk2,
        _wrf_real_add(
            zgamma4,
            _wrf_real_mul(za1, prmuz_arr, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    zbeta = _wrf_real_div(
        _wrf_real_sub(zgamma1, zrk, rounding_token),
        zrkg,
        rounding_token,
    )

    ze1 = jnp.minimum(
        _wrf_real_mul(zrk, zto1, rounding_token),
        jnp.float32(500.0),
    )
    ze2 = jnp.minimum(
        _wrf_real_div(zto1, prmuz_arr, rounding_token),
        jnp.float32(500.0),
    )
    zem1 = _sw_exp_lookup_real(ze1, exp_tbl, bpade, rounding_token)
    zep1 = _wrf_real_div(one, zem1, rounding_token)
    zem2 = _sw_exp_lookup_real(ze2, exp_tbl, bpade, rounding_token)
    zep2 = _wrf_real_div(one, zem2, rounding_token)

    zdenr = _wrf_real_add(
        _wrf_real_mul(zr4, zep1, rounding_token),
        _wrf_real_mul(zr5, zem1, rounding_token),
        rounding_token,
    )
    zdent = _wrf_real_add(
        _wrf_real_mul(zr4, zep1, rounding_token),
        _wrf_real_mul(zr5, zem1, rounding_token),
        rounding_token,
    )
    small_den = (zdenr >= -eps) & (zdenr <= eps)
    pref_terms = _wrf_real_sub(
        _wrf_real_sub(
            _wrf_real_mul(zr1, zep1, rounding_token),
            _wrf_real_mul(zr2, zem1, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(zr3, zem2, rounding_token),
        rounding_token,
    )
    pref_general = _wrf_real_div(
        _wrf_real_mul(zw, pref_terms, rounding_token),
        jnp.where(small_den, one, zdenr),
        rounding_token,
    )
    ptra_terms = _wrf_real_sub(
        _wrf_real_sub(
            _wrf_real_mul(zt1, zep1, rounding_token),
            _wrf_real_mul(zt2, zem1, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(zt3, zep2, rounding_token),
        rounding_token,
    )
    ptra_general = _wrf_real_sub(
        zem2,
        _wrf_real_div(
            _wrf_real_mul(
                _wrf_real_mul(zem2, zw, rounding_token),
                ptra_terms,
                rounding_token,
            ),
            jnp.where(small_den, one, zdent),
            rounding_token,
        ),
        rounding_token,
    )
    pref_general = jnp.where(small_den, eps, pref_general)
    ptra_general = jnp.where(small_den, zem2, ptra_general)

    zemm = _wrf_real_mul(zem1, zem1, rounding_token)
    zdend = _wrf_real_div(
        one,
        _wrf_real_mul(
            _wrf_real_sub(
                one,
                _wrf_real_mul(zbeta, zemm, rounding_token),
                rounding_token,
            ),
            zrkg,
            rounding_token,
        ),
        rounding_token,
    )
    prefd_general = _wrf_real_mul(
        _wrf_real_mul(
            zgamma2,
            _wrf_real_sub(one, zemm, rounding_token),
            rounding_token,
        ),
        zdend,
        rounding_token,
    )
    ptrad_general = _wrf_real_mul(
        _wrf_real_mul(zrk2, zem1, rounding_token),
        zdend,
        rounding_token,
    )

    pref_active = jnp.where(conservative, pref_cons, pref_general)
    ptra_active = jnp.where(conservative, ptra_cons, ptra_general)
    prefd_active = jnp.where(conservative, prefd_cons, prefd_general)
    ptrad_active = jnp.where(conservative, ptrad_cons, ptrad_general)
    return RrtmgSwReftraResult(
        pref=jnp.where(lrt, pref_active, zero),
        prefd=jnp.where(lrt, prefd_active, zero),
        ptra=jnp.where(lrt, ptra_active, one),
        ptrad=jnp.where(lrt, ptrad_active, one),
    )


def reftra_sw(
    *,
    nlayers: int,
    lrtchk: Any,
    pgg: Any,
    prmuz: Any,
    ptau: Any,
    pw: Any,
    tables: RrtmgSwPadeTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwReftraResult:
    """WRF ``reftra_sw`` layer reflectance/transmittance kernel."""

    nlay = int(nlayers)
    tab = rrtmg_sw_pade_tables() if tables is None else tables
    lrt = jnp.asarray(lrtchk, dtype=bool)[:nlay]
    zg = jnp.asarray(pgg, dtype=jnp.float32)[:nlay]
    zto1 = jnp.asarray(ptau, dtype=jnp.float32)[:nlay]
    zw = jnp.asarray(pw, dtype=jnp.float32)[:nlay]
    prmuz_arr = jnp.asarray(prmuz, dtype=jnp.float32)

    if rounding_token is not None:
        return _reftra_sw_wrf_real(
            lrt=lrt,
            zg=zg,
            zto1=zto1,
            zw=zw,
            prmuz_arr=prmuz_arr,
            tables=tab,
            rounding_token=rounding_token,
        )

    eps = jnp.asarray(1.0e-8, dtype=jnp.float32)
    zwcrit = jnp.asarray(0.9999995, dtype=jnp.float32)
    zsr3 = sqrt_for_wrf(jnp.asarray(3.0, dtype=jnp.float32))
    zg3 = 3.0 * zg
    # WRF sets kmodts=2 in this routine.
    zgamma1 = (8.0 - zw * (5.0 + zg3)) * 0.25
    zgamma2 = 3.0 * (zw * (1.0 - zg)) * 0.25
    zgamma3 = (2.0 - zg3 * prmuz_arr) * 0.25
    del zsr3
    zgamma4 = 1.0 - zgamma3

    safe_ratio = jnp.where(zg != 1.0, zg / (1.0 - zg), 0.0)
    denom = jnp.where(zg != 1.0, 1.0 - (1.0 - zw) * safe_ratio * safe_ratio, 1.0)
    zwo = jnp.where((zw > 0.0) & (denom != 0.0), zw / denom, 0.0)
    conservative = zwo >= zwcrit

    za = zgamma1 * prmuz_arr
    za1_cons = za - zgamma3
    zgt = zgamma1 * zto1
    ze_cons = jnp.minimum(zto1 / prmuz_arr, 500.0)
    exp_cons = _sw_exp_lookup(ze_cons, tab)
    pref_cons = (zgt - za1_cons * (1.0 - exp_cons)) / (1.0 + zgt)
    ptra_cons = 1.0 - pref_cons
    prefd_cons = zgt / (1.0 + zgt)
    ptrad_cons = 1.0 - prefd_cons
    clear_cons = exp_cons == 1.0
    pref_cons = jnp.where(clear_cons, 0.0, pref_cons)
    ptra_cons = jnp.where(clear_cons, 1.0, ptra_cons)
    prefd_cons = jnp.where(clear_cons, 0.0, prefd_cons)
    ptrad_cons = jnp.where(clear_cons, 1.0, ptrad_cons)

    za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3
    za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4
    zrk = sqrt_for_wrf(jnp.maximum(zgamma1 * zgamma1 - zgamma2 * zgamma2, 0.0))
    zrp = zrk * prmuz_arr
    zrp1 = 1.0 + zrp
    zrm1 = 1.0 - zrp
    zrk2 = 2.0 * zrk
    zrpp = 1.0 - zrp * zrp
    zrkg = zrk + zgamma1
    zr1 = zrm1 * (za2 + zrk * zgamma3)
    zr2 = zrp1 * (za2 - zrk * zgamma3)
    zr3 = zrk2 * (zgamma3 - za2 * prmuz_arr)
    zr4 = zrpp * zrkg
    zr5 = zrpp * (zrk - zgamma1)
    zt1 = zrp1 * (za1 + zrk * zgamma4)
    zt2 = zrm1 * (za1 - zrk * zgamma4)
    zt3 = zrk2 * (zgamma4 + za1 * prmuz_arr)
    zt4 = zr4
    zt5 = zr5
    zbeta = (zgamma1 - zrk) / zrkg

    ze1 = jnp.minimum(zrk * zto1, 500.0)
    ze2 = jnp.minimum(zto1 / prmuz_arr, 500.0)
    zem1 = _sw_exp_lookup(ze1, tab)
    zep1 = 1.0 / zem1
    zem2 = _sw_exp_lookup(ze2, tab)
    zep2 = 1.0 / zem2

    zdenr = zr4 * zep1 + zr5 * zem1
    zdent = zt4 * zep1 + zt5 * zem1
    small_den = (zdenr >= -eps) & (zdenr <= eps)
    pref_general = jnp.where(small_den, eps, zw * (zr1 * zep1 - zr2 * zem1 - zr3 * zem2) / zdenr)
    ptra_general = jnp.where(
        small_den,
        zem2,
        zem2 - zem2 * zw * (zt1 * zep1 - zt2 * zem1 - zt3 * zep2) / zdent,
    )
    zemm = zem1 * zem1
    zdend = 1.0 / ((1.0 - zbeta * zemm) * zrkg)
    prefd_general = zgamma2 * (1.0 - zemm) * zdend
    ptrad_general = zrk2 * zem1 * zdend

    pref_active = jnp.where(conservative, pref_cons, pref_general)
    ptra_active = jnp.where(conservative, ptra_cons, ptra_general)
    prefd_active = jnp.where(conservative, prefd_cons, prefd_general)
    ptrad_active = jnp.where(conservative, ptrad_cons, ptrad_general)

    pref = jnp.where(lrt, pref_active, 0.0)
    ptra = jnp.where(lrt, ptra_active, 1.0)
    prefd = jnp.where(lrt, prefd_active, 0.0)
    ptrad = jnp.where(lrt, ptrad_active, 1.0)
    return RrtmgSwReftraResult(pref=pref, prefd=prefd, ptra=ptra, ptrad=ptrad)


def _vrtqdr_sw_wrf_real(
    *,
    klev: int,
    pref: Any,
    prefd: Any,
    ptra: Any,
    ptrad: Any,
    pdbt: Any,
    prdnd: Any,
    prup: Any,
    prupd: Any,
    ptdbt: Any,
    rounding_token: Any,
) -> RrtmgSwVrtqdrResult:
    """Evaluate ``vrtqdr_sw`` with WRF default-REAL statement boundaries."""

    from wrf_jax.core.real_arithmetic import (
        real_add as _wrf_real_add,
        real_subtract as _wrf_real_sub,
        real_multiply as _wrf_real_mul,
        real_divide_assigned as _wrf_real_div,
    )

    n = int(klev)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    pref_arr = jnp.asarray(pref, dtype=jnp.float32)[: n + 1]
    prefd_arr = jnp.asarray(prefd, dtype=jnp.float32)[: n + 1]
    ptra_arr = jnp.asarray(ptra, dtype=jnp.float32)[: n + 1]
    ptrad_arr = jnp.asarray(ptrad, dtype=jnp.float32)[: n + 1]
    pdbt_arr = jnp.asarray(pdbt, dtype=jnp.float32)[: n + 1]
    ptdbt_arr = jnp.asarray(ptdbt, dtype=jnp.float32)[: n + 1]
    prdnd_arr = jnp.asarray(prdnd, dtype=jnp.float32)[: n + 1]
    prup_arr = jnp.asarray(prup, dtype=jnp.float32)[: n + 1]
    prupd_arr = jnp.asarray(prupd, dtype=jnp.float32)[: n + 1]

    zreflect = _wrf_real_div(
        one,
        _wrf_real_sub(
            one,
            _wrf_real_mul(prefd_arr[n], prefd_arr[n - 1], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    direct = _wrf_real_mul(
        _wrf_real_sub(ptra_arr[n - 1], pdbt_arr[n - 1], rounding_token),
        prefd_arr[n],
        rounding_token,
    )
    reflected = _wrf_real_mul(pdbt_arr[n - 1], pref_arr[n], rounding_token)
    linked = _wrf_real_mul(
        _wrf_real_mul(
            ptrad_arr[n - 1],
            _wrf_real_add(direct, reflected, rounding_token),
            rounding_token,
        ),
        zreflect,
        rounding_token,
    )
    prup_arr = prup_arr.at[n - 1].set(
        _wrf_real_add(pref_arr[n - 1], linked, rounding_token)
    )
    diffuse = _wrf_real_mul(ptrad_arr[n - 1], ptrad_arr[n - 1], rounding_token)
    diffuse = _wrf_real_mul(diffuse, prefd_arr[n], rounding_token)
    diffuse = _wrf_real_mul(diffuse, zreflect, rounding_token)
    prupd_arr = prupd_arr.at[n - 1].set(
        _wrf_real_add(prefd_arr[n - 1], diffuse, rounding_token)
    )

    def upward_body(carry: tuple[Any, Any], inputs: tuple[Any, ...], rounding_token: Any):
        prup_c, prupd_c = carry
        pref_k, prefd_k, ptra_k, ptrad_k, pdbt_k = inputs
        one = jnp.full_like(prup_c, 1.0)
        reflect = _wrf_real_div(
            one,
            _wrf_real_sub(
                one,
                _wrf_real_mul(prupd_c, prefd_k, rounding_token),
                rounding_token,
            ),
            rounding_token,
        )
        direct_c = _wrf_real_mul(
            _wrf_real_sub(ptra_k, pdbt_k, rounding_token),
            prupd_c,
            rounding_token,
        )
        reflected_c = _wrf_real_mul(pdbt_k, prup_c, rounding_token)
        linked_c = _wrf_real_mul(
            _wrf_real_mul(
                ptrad_k,
                _wrf_real_add(direct_c, reflected_c, rounding_token),
                rounding_token,
            ),
            reflect,
            rounding_token,
        )
        prup_c = _wrf_real_add(pref_k, linked_c, rounding_token)
        diffuse_c = _wrf_real_mul(ptrad_k, ptrad_k, rounding_token)
        diffuse_c = _wrf_real_mul(diffuse_c, prupd_c, rounding_token)
        diffuse_c = _wrf_real_mul(diffuse_c, reflect, rounding_token)
        prupd_c = _wrf_real_add(prefd_k, diffuse_c, rounding_token)
        return (prup_c, prupd_c), (prup_c, prupd_c)

    # Retain only the preceding level in each recurrence. The same body runs
    # inside one CUDA column kernel or an ordinary CPU scan.
    _, up_values = scan_columns(
        upward_body, (prup_arr[n - 1], prupd_arr[n - 1]),
        tuple(value[:n - 1] for value in
              (pref_arr, prefd_arr, ptra_arr, ptrad_arr, pdbt_arr)), rounding_token,
        reverse=True,
    )
    prup_arr = prup_arr.at[:n - 1].set(up_values[0])
    prupd_arr = prupd_arr.at[:n - 1].set(up_values[1])

    ztdn = jnp.zeros_like(pref_arr).at[0].set(one)
    prdnd_arr = prdnd_arr.at[0].set(jnp.asarray(0.0, dtype=jnp.float32))
    ztdn = ztdn.at[1].set(ptra_arr[0])
    prdnd_arr = prdnd_arr.at[1].set(prefd_arr[0])

    def downward_body(carry: tuple[Any, Any], inputs: tuple[Any, ...], rounding_token: Any):
        ztdn_c, prdnd_c = carry
        pref_k, prefd_k, ptra_k, ptrad_k, ptdbt_k = inputs
        one = jnp.full_like(ztdn_c, 1.0)
        reflect = _wrf_real_div(
            one,
            _wrf_real_sub(
                one,
                _wrf_real_mul(prefd_k, prdnd_c, rounding_token),
                rounding_token,
            ),
            rounding_token,
        )
        diffuse_c = _wrf_real_sub(ztdn_c, ptdbt_k, rounding_token)
        reflected_c = _wrf_real_mul(ptdbt_k, pref_k, rounding_token)
        reflected_c = _wrf_real_mul(reflected_c, prdnd_c, rounding_token)
        linked_c = _wrf_real_add(diffuse_c, reflected_c, rounding_token)
        linked_c = _wrf_real_mul(ptrad_k, linked_c, rounding_token)
        linked_c = _wrf_real_mul(linked_c, reflect, rounding_token)
        direct_c = _wrf_real_mul(ptdbt_k, ptra_k, rounding_token)
        ztdn_c = _wrf_real_add(direct_c, linked_c, rounding_token)
        diffuse_reflect = _wrf_real_mul(
            ptrad_k, ptrad_k, rounding_token
        )
        diffuse_reflect = _wrf_real_mul(diffuse_reflect, prdnd_c, rounding_token)
        diffuse_reflect = _wrf_real_mul(diffuse_reflect, reflect, rounding_token)
        prdnd_c = _wrf_real_add(prefd_k, diffuse_reflect, rounding_token)
        return (ztdn_c, prdnd_c), (ztdn_c, prdnd_c)

    _, down_values = scan_columns(
        downward_body, (ztdn[1], prdnd_arr[1]),
        tuple(value[1:n] for value in
              (pref_arr, prefd_arr, ptra_arr, ptrad_arr, ptdbt_arr)), rounding_token,
    )
    ztdn = ztdn.at[2:n + 1].set(down_values[0])
    prdnd_arr = prdnd_arr.at[2:n + 1].set(down_values[1])

    def flux_body(inputs, rounding_token):
        prdnd_arr, prup_arr, prupd_arr, ptdbt_arr, ztdn = inputs
        one = jnp.float32(1.0)
        reflect = _wrf_real_div(
            one,
            _wrf_real_sub(
                one,
                _wrf_real_mul(prdnd_arr, prupd_arr, rounding_token),
                rounding_token,
            ),
            rounding_token,
        )
        up_direct = _wrf_real_mul(ptdbt_arr, prup_arr, rounding_token)
        up_diffuse = _wrf_real_sub(ztdn, ptdbt_arr, rounding_token)
        up_diffuse = _wrf_real_mul(up_diffuse, prupd_arr, rounding_token)
        pfu = _wrf_real_add(up_direct, up_diffuse, rounding_token)
        pfu = _wrf_real_mul(pfu, reflect, rounding_token)

        down_diffuse = _wrf_real_sub(ztdn, ptdbt_arr, rounding_token)
        down_reflected = _wrf_real_mul(ptdbt_arr, prup_arr, rounding_token)
        down_reflected = _wrf_real_mul(down_reflected, prdnd_arr, rounding_token)
        pfd = _wrf_real_add(down_diffuse, down_reflected, rounding_token)
        pfd = _wrf_real_mul(pfd, reflect, rounding_token)
        pfd = _wrf_real_add(ptdbt_arr, pfd, rounding_token)
        return pfd, pfu

    # Keep the row/level/g-point memory order at the pointwise call.
    level_axis = max(prdnd_arr.ndim - 2, 0)
    pfd, pfu = map_elements(
        flux_body,
        tuple(jnp.moveaxis(value, 0, level_axis) for value in
              (prdnd_arr, prup_arr, prupd_arr, ptdbt_arr, ztdn)),
        rounding_token,
    )
    pfd = jnp.moveaxis(pfd, level_axis, 0)
    pfu = jnp.moveaxis(pfu, level_axis, 0)
    return RrtmgSwVrtqdrResult(
        prdnd=prdnd_arr,
        prup=prup_arr,
        prupd=prupd_arr,
        pfd=pfd,
        pfu=pfu,
    )


def vrtqdr_sw(
    *,
    klev: int,
    pref: Any,
    prefd: Any,
    ptra: Any,
    ptrad: Any,
    pdbt: Any,
    prdnd: Any,
    prup: Any,
    prupd: Any,
    ptdbt: Any,
    rounding_token: Any | None = None,
) -> RrtmgSwVrtqdrResult:
    """WRF ``vrtqdr_sw`` vertical quadrature kernel for one shortwave g-point.

    WRF writes one column ``kw`` of the two-dimensional ``pfd`` and ``pfu``
    arrays.  This pure translation returns that column as one-dimensional
    arrays and makes the three ``intent(inout)`` arrays explicit.
    """

    if rounding_token is not None:
        return _vrtqdr_sw_wrf_real(
            klev=klev,
            pref=pref,
            prefd=prefd,
            ptra=ptra,
            ptrad=ptrad,
            pdbt=pdbt,
            prdnd=prdnd,
            prup=prup,
            prupd=prupd,
            ptdbt=ptdbt,
            rounding_token=rounding_token,
        )

    n = int(klev)
    pref_arr = jnp.asarray(pref, dtype=jnp.float32)[: n + 1]
    prefd_arr = jnp.asarray(prefd, dtype=jnp.float32)[: n + 1]
    ptra_arr = jnp.asarray(ptra, dtype=jnp.float32)[: n + 1]
    ptrad_arr = jnp.asarray(ptrad, dtype=jnp.float32)[: n + 1]
    pdbt_arr = jnp.asarray(pdbt, dtype=jnp.float32)[: n + 1]
    ptdbt_arr = jnp.asarray(ptdbt, dtype=jnp.float32)[: n + 1]
    prdnd_arr = jnp.asarray(prdnd, dtype=jnp.float32)[: n + 1]
    prup_arr = jnp.asarray(prup, dtype=jnp.float32)[: n + 1]
    prupd_arr = jnp.asarray(prupd, dtype=jnp.float32)[: n + 1]

    zreflect = 1.0 / (1.0 - prefd_arr[n] * prefd_arr[n - 1])
    prup_arr = prup_arr.at[n - 1].set(
        pref_arr[n - 1]
        + (
            ptrad_arr[n - 1]
            * ((ptra_arr[n - 1] - pdbt_arr[n - 1]) * prefd_arr[n] + pdbt_arr[n - 1] * pref_arr[n])
        )
        * zreflect
    )
    prupd_arr = prupd_arr.at[n - 1].set(
        prefd_arr[n - 1] + ptrad_arr[n - 1] * ptrad_arr[n - 1] * prefd_arr[n] * zreflect
    )

    def _upward_body(jk: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        prup_c, prupd_c = carry
        ikp = n - 1 - jk
        ikx = ikp - 1
        zreflect_c = 1.0 / (1.0 - prupd_c[ikp] * prefd_arr[ikx])
        prup_c = prup_c.at[ikx].set(
            pref_arr[ikx]
            + (
                ptrad_arr[ikx]
                * ((ptra_arr[ikx] - pdbt_arr[ikx]) * prupd_c[ikp] + pdbt_arr[ikx] * prup_c[ikp])
            )
            * zreflect_c
        )
        prupd_c = prupd_c.at[ikx].set(
            prefd_arr[ikx] + ptrad_arr[ikx] * ptrad_arr[ikx] * prupd_c[ikp] * zreflect_c
        )
        return prup_c, prupd_c

    prup_arr, prupd_arr = lax.fori_loop(0, max(n - 1, 0), _upward_body, (prup_arr, prupd_arr))

    ztdn = jnp.zeros_like(pref_arr)
    ztdn = ztdn.at[0].set(1.0)
    prdnd_arr = prdnd_arr.at[0].set(0.0)
    ztdn = ztdn.at[1].set(ptra_arr[0])
    prdnd_arr = prdnd_arr.at[1].set(prefd_arr[0])

    def _downward_body(jk0: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        ztdn_c, prdnd_c = carry
        zreflect_c = 1.0 / (1.0 - prefd_arr[jk0] * prdnd_c[jk0])
        ztdn_c = ztdn_c.at[jk0 + 1].set(
            ptdbt_arr[jk0] * ptra_arr[jk0]
            + (
                ptrad_arr[jk0]
                * ((ztdn_c[jk0] - ptdbt_arr[jk0]) + ptdbt_arr[jk0] * pref_arr[jk0] * prdnd_c[jk0])
            )
            * zreflect_c
        )
        prdnd_c = prdnd_c.at[jk0 + 1].set(
            prefd_arr[jk0] + ptrad_arr[jk0] * ptrad_arr[jk0] * prdnd_c[jk0] * zreflect_c
        )
        return ztdn_c, prdnd_c

    ztdn, prdnd_arr = lax.fori_loop(1, n, _downward_body, (ztdn, prdnd_arr))

    zreflect_all = 1.0 / (1.0 - prdnd_arr * prupd_arr)
    pfu = (ptdbt_arr * prup_arr + (ztdn - ptdbt_arr) * prupd_arr) * zreflect_all
    pfd = ptdbt_arr + (ztdn - ptdbt_arr + ptdbt_arr * prup_arr * prdnd_arr) * zreflect_all
    return RrtmgSwVrtqdrResult(prdnd=prdnd_arr, prup=prup_arr, prupd=prupd_arr, pfd=pfd, pfu=pfu)


def reftra_sw_columns(
    *,
    nlayers: int,
    lrtchk: Any,
    pgg: Any,
    prmuz: Any,
    ptau: Any,
    pw: Any,
    tables: RrtmgSwPadeTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwReftraResult:
    """Batch ``reftra_sw`` over horizontal columns and g-points."""

    nlay = int(nlayers)
    tab = rrtmg_sw_pade_tables() if tables is None else tables
    lrt = jnp.asarray(lrtchk, dtype=bool)[:, :nlay, :]
    zg = jnp.asarray(pgg, dtype=jnp.float32)[:, :nlay, :]
    zto1 = jnp.asarray(ptau, dtype=jnp.float32)[:, :nlay, :]
    zw = jnp.asarray(pw, dtype=jnp.float32)[:, :nlay, :]
    prmuz_arr = jnp.asarray(prmuz, dtype=jnp.float32)[:, None, None]

    if rounding_token is not None:
        return _reftra_sw_wrf_real(
            lrt=lrt,
            zg=zg,
            zto1=zto1,
            zw=zw,
            prmuz_arr=prmuz_arr,
            tables=tab,
            rounding_token=rounding_token,
        )

    eps = jnp.asarray(1.0e-8, dtype=jnp.float32)
    zwcrit = jnp.asarray(0.9999995, dtype=jnp.float32)
    zg3 = 3.0 * zg
    zgamma1 = (8.0 - zw * (5.0 + zg3)) * 0.25
    zgamma2 = 3.0 * (zw * (1.0 - zg)) * 0.25
    zgamma3 = (2.0 - zg3 * prmuz_arr) * 0.25
    zgamma4 = 1.0 - zgamma3

    safe_ratio = jnp.where(zg != 1.0, zg / (1.0 - zg), 0.0)
    denom = jnp.where(zg != 1.0, 1.0 - (1.0 - zw) * safe_ratio * safe_ratio, 1.0)
    zwo = jnp.where((zw > 0.0) & (denom != 0.0), zw / denom, 0.0)
    conservative = zwo >= zwcrit

    za = zgamma1 * prmuz_arr
    za1_cons = za - zgamma3
    zgt = zgamma1 * zto1
    ze_cons = jnp.minimum(zto1 / prmuz_arr, 500.0)
    exp_cons = _sw_exp_lookup(ze_cons, tab)
    pref_cons = (zgt - za1_cons * (1.0 - exp_cons)) / (1.0 + zgt)
    ptra_cons = 1.0 - pref_cons
    prefd_cons = zgt / (1.0 + zgt)
    ptrad_cons = 1.0 - prefd_cons
    clear_cons = exp_cons == 1.0
    pref_cons = jnp.where(clear_cons, 0.0, pref_cons)
    ptra_cons = jnp.where(clear_cons, 1.0, ptra_cons)
    prefd_cons = jnp.where(clear_cons, 0.0, prefd_cons)
    ptrad_cons = jnp.where(clear_cons, 1.0, ptrad_cons)

    za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3
    za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4
    zrk = sqrt_for_wrf(jnp.maximum(zgamma1 * zgamma1 - zgamma2 * zgamma2, 0.0))
    zrp = zrk * prmuz_arr
    zrp1 = 1.0 + zrp
    zrm1 = 1.0 - zrp
    zrk2 = 2.0 * zrk
    zrpp = 1.0 - zrp * zrp
    zrkg = zrk + zgamma1
    zr1 = zrm1 * (za2 + zrk * zgamma3)
    zr2 = zrp1 * (za2 - zrk * zgamma3)
    zr3 = zrk2 * (zgamma3 - za2 * prmuz_arr)
    zr4 = zrpp * zrkg
    zr5 = zrpp * (zrk - zgamma1)
    zt1 = zrp1 * (za1 + zrk * zgamma4)
    zt2 = zrm1 * (za1 - zrk * zgamma4)
    zt3 = zrk2 * (zgamma4 + za1 * prmuz_arr)
    zt4 = zr4
    zt5 = zr5
    zbeta = (zgamma1 - zrk) / zrkg

    ze1 = jnp.minimum(zrk * zto1, 500.0)
    ze2 = jnp.minimum(zto1 / prmuz_arr, 500.0)
    zem1 = _sw_exp_lookup(ze1, tab)
    zep1 = 1.0 / zem1
    zem2 = _sw_exp_lookup(ze2, tab)
    zep2 = 1.0 / zem2

    zdenr = zr4 * zep1 + zr5 * zem1
    zdent = zt4 * zep1 + zt5 * zem1
    small_den = (zdenr >= -eps) & (zdenr <= eps)
    pref_general = jnp.where(small_den, eps, zw * (zr1 * zep1 - zr2 * zem1 - zr3 * zem2) / zdenr)
    ptra_general = jnp.where(
        small_den,
        zem2,
        zem2 - zem2 * zw * (zt1 * zep1 - zt2 * zem1 - zt3 * zep2) / zdent,
    )
    zemm = zem1 * zem1
    zdend = 1.0 / ((1.0 - zbeta * zemm) * zrkg)
    prefd_general = zgamma2 * (1.0 - zemm) * zdend
    ptrad_general = zrk2 * zem1 * zdend

    pref_active = jnp.where(conservative, pref_cons, pref_general)
    ptra_active = jnp.where(conservative, ptra_cons, ptra_general)
    prefd_active = jnp.where(conservative, prefd_cons, prefd_general)
    ptrad_active = jnp.where(conservative, ptrad_cons, ptrad_general)

    pref = jnp.where(lrt, pref_active, 0.0)
    ptra = jnp.where(lrt, ptra_active, 1.0)
    prefd = jnp.where(lrt, prefd_active, 0.0)
    ptrad = jnp.where(lrt, ptrad_active, 1.0)
    return RrtmgSwReftraResult(pref=pref, prefd=prefd, ptra=ptra, ptrad=ptrad)


def vrtqdr_sw_columns(
    *,
    klev: int,
    pref: Any,
    prefd: Any,
    ptra: Any,
    ptrad: Any,
    pdbt: Any,
    prdnd: Any,
    prup: Any,
    prupd: Any,
    ptdbt: Any,
    rounding_token: Any | None = None,
) -> RrtmgSwVrtqdrResult:
    """Batch ``vrtqdr_sw`` over horizontal columns and g-points."""

    if rounding_token is not None:
        n = int(klev)

        def level_first(value: Any) -> Any:
            return jnp.moveaxis(
                jnp.asarray(value, dtype=jnp.float32)[:, : n + 1, :], 1, 0
            )

        pref_level = level_first(pref)
        prefd_level = level_first(prefd)
        prup_level = level_first(prup).at[n].set(pref_level[n])
        prupd_level = level_first(prupd).at[n].set(prefd_level[n])
        result = _vrtqdr_sw_wrf_real(
            klev=n,
            pref=pref_level,
            prefd=prefd_level,
            ptra=level_first(ptra),
            ptrad=level_first(ptrad),
            pdbt=level_first(pdbt),
            prdnd=level_first(prdnd),
            prup=prup_level,
            prupd=prupd_level,
            ptdbt=level_first(ptdbt),
            rounding_token=rounding_token,
        )
        return RrtmgSwVrtqdrResult(
            prdnd=jnp.moveaxis(result.prdnd, 0, 1),
            prup=jnp.moveaxis(result.prup, 0, 1),
            prupd=jnp.moveaxis(result.prupd, 0, 1),
            pfd=jnp.moveaxis(result.pfd, 0, 1),
            pfu=jnp.moveaxis(result.pfu, 0, 1),
        )

    del prdnd, prup, prupd
    n = int(klev)
    pref_arr = jnp.asarray(pref, dtype=jnp.float32)[:, : n + 1, :]
    prefd_arr = jnp.asarray(prefd, dtype=jnp.float32)[:, : n + 1, :]
    ptra_arr = jnp.asarray(ptra, dtype=jnp.float32)[:, : n + 1, :]
    ptrad_arr = jnp.asarray(ptrad, dtype=jnp.float32)[:, : n + 1, :]
    pdbt_arr = jnp.asarray(pdbt, dtype=jnp.float32)[:, : n + 1, :]
    ptdbt_arr = jnp.asarray(ptdbt, dtype=jnp.float32)[:, : n + 1, :]

    def upward_step(carry: tuple[Any, Any], inputs: tuple[Any, ...]) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        prup_next, prupd_next = carry
        pref_i, prefd_i, ptra_i, ptrad_i, pdbt_i = inputs
        zreflect = 1.0 / (1.0 - prupd_next * prefd_i)
        prup_i = pref_i + ptrad_i * ((ptra_i - pdbt_i) * prupd_next + pdbt_i * prup_next) * zreflect
        prupd_i = prefd_i + ptrad_i * ptrad_i * prupd_next * zreflect
        return (prup_i, prupd_i), (prup_i, prupd_i)

    up_inputs = (
        jnp.moveaxis(pref_arr[:, :n, :][:, ::-1, :], 1, 0),
        jnp.moveaxis(prefd_arr[:, :n, :][:, ::-1, :], 1, 0),
        jnp.moveaxis(ptra_arr[:, :n, :][:, ::-1, :], 1, 0),
        jnp.moveaxis(ptrad_arr[:, :n, :][:, ::-1, :], 1, 0),
        jnp.moveaxis(pdbt_arr[:, :n, :][:, ::-1, :], 1, 0),
    )
    _up_carry, up_outputs = lax.scan(upward_step, (pref_arr[:, n, :], prefd_arr[:, n, :]), up_inputs)
    prup_layers, prupd_layers = up_outputs
    prup_arr = jnp.concatenate((jnp.moveaxis(prup_layers[::-1], 0, 1), pref_arr[:, n : n + 1, :]), axis=1)
    prupd_arr = jnp.concatenate((jnp.moveaxis(prupd_layers[::-1], 0, 1), prefd_arr[:, n : n + 1, :]), axis=1)

    zeros = jnp.zeros_like(pref_arr[:, 0, :])
    ones = jnp.ones_like(pref_arr[:, 0, :])
    if n > 1:
        def downward_step(carry: tuple[Any, Any], inputs: tuple[Any, ...]) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
            ztdn_prev, prdnd_prev = carry
            pref_i, prefd_i, ptra_i, ptrad_i, pdbt_i, ptdbt_i = inputs
            zreflect = 1.0 / (1.0 - prefd_i * prdnd_prev)
            ztdn_i = ptdbt_i * ptra_i + ptrad_i * (
                (ztdn_prev - ptdbt_i) + ptdbt_i * pref_i * prdnd_prev
            ) * zreflect
            prdnd_i = prefd_i + ptrad_i * ptrad_i * prdnd_prev * zreflect
            return (ztdn_i, prdnd_i), (ztdn_i, prdnd_i)

        down_inputs = (
            jnp.moveaxis(pref_arr[:, 1:n, :], 1, 0),
            jnp.moveaxis(prefd_arr[:, 1:n, :], 1, 0),
            jnp.moveaxis(ptra_arr[:, 1:n, :], 1, 0),
            jnp.moveaxis(ptrad_arr[:, 1:n, :], 1, 0),
            jnp.moveaxis(pdbt_arr[:, 1:n, :], 1, 0),
            jnp.moveaxis(ptdbt_arr[:, 1:n, :], 1, 0),
        )
        _down_carry, down_outputs = lax.scan(downward_step, (ptra_arr[:, 0, :], prefd_arr[:, 0, :]), down_inputs)
        ztdn_tail, prdnd_tail = down_outputs
        ztdn_arr = jnp.concatenate(
            (ones[:, None, :], ptra_arr[:, :1, :], jnp.moveaxis(ztdn_tail, 0, 1)),
            axis=1,
        )
        prdnd_arr = jnp.concatenate(
            (zeros[:, None, :], prefd_arr[:, :1, :], jnp.moveaxis(prdnd_tail, 0, 1)),
            axis=1,
        )
    else:
        ztdn_arr = jnp.concatenate((ones[:, None, :], ptra_arr[:, :1, :]), axis=1)
        prdnd_arr = jnp.concatenate((zeros[:, None, :], prefd_arr[:, :1, :]), axis=1)

    zreflect_all = 1.0 / (1.0 - prdnd_arr * prupd_arr)
    pfu = (ptdbt_arr * prup_arr + (ztdn_arr - ptdbt_arr) * prupd_arr) * zreflect_all
    pfd = ptdbt_arr + (ztdn_arr - ptdbt_arr + ptdbt_arr * prup_arr * prdnd_arr) * zreflect_all
    return RrtmgSwVrtqdrResult(prdnd=prdnd_arr, prup=prup_arr, prupd=prupd_arr, pfd=pfd, pfu=pfu)


def _sw_boundary_with_layers(layer_values: Any, surface_value: Any, n: int) -> Any:
    layers = jnp.asarray(layer_values, dtype=jnp.float32)[:n]
    out = jnp.zeros((n + 1,) + layers.shape[1:], dtype=jnp.float32)
    out = out.at[:n].set(layers)
    return out.at[n].set(jnp.asarray(surface_value, dtype=jnp.float32))


def spcvmc_sw_from_optical_depths(
    *,
    nlayers: int,
    istart: int = 16,
    iend: int = 29,
    icpr: int = 1,
    iout: int = 0,
    palbd: Any,
    palbp: Any,
    pcldfmc: Any,
    ptaucmc: Any,
    pasycmc: Any,
    pomgcmc: Any,
    ptaormc: Any,
    ptaua: Any | None,
    pasya: Any | None,
    pomga: Any | None,
    prmu0: Any,
    adjflux: Any,
    sfluxzen: Any,
    taug: Any,
    taur: Any,
    spectral: RrtmgSpectralInit | None = None,
    tables: RrtmgSwPadeTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwSpcvmcResult:
    """WRF ``spcvmc_sw`` spectral integration with precomputed optical depths."""

    n = int(nlayers)
    tab = rrtmg_sw_pade_tables() if tables is None else tables
    if spectral is None:
        ngc = _SW_NGC_VALUES
    else:
        try:
            ngc = tuple(int(x) for x in np.asarray(spectral.ngc))
        except Exception:
            ngc = _SW_NGC_VALUES
    prmu0_arr = jnp.asarray(prmu0, dtype=jnp.float32)
    repclc = jnp.asarray(1.0e-12, dtype=jnp.float32)

    palbd_arr = jnp.asarray(palbd, dtype=jnp.float32)[:RRTMG_SW_NBND]
    palbp_arr = jnp.asarray(palbp, dtype=jnp.float32)[:RRTMG_SW_NBND]
    pcldfmc_arr = jnp.asarray(pcldfmc, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    ptaucmc_arr = jnp.asarray(ptaucmc, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    pasycmc_arr = jnp.asarray(pasycmc, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    pomgcmc_arr = jnp.asarray(pomgcmc, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    ptaormc_arr = jnp.asarray(ptaormc, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    ptaua_arr = jnp.asarray(ptaua, dtype=jnp.float32)[:n, :RRTMG_SW_NBND]
    pasya_arr = jnp.asarray(pasya, dtype=jnp.float32)[:n, :RRTMG_SW_NBND]
    pomga_arr = jnp.asarray(pomga, dtype=jnp.float32)[:n, :RRTMG_SW_NBND]
    sfluxzen_arr = jnp.asarray(sfluxzen, dtype=jnp.float32)[:RRTMG_SW_NGPT]
    taug_arr = jnp.asarray(taug, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]
    taur_arr = jnp.asarray(taur, dtype=jnp.float32)[:n, :RRTMG_SW_NGPT]

    band_for_gpoint = np.asarray(
        tuple(band for band, count in enumerate(ngc) for _ in range(count)),
        dtype=np.int32,
    )
    active_band = np.asarray(
        [(int(istart) - 16) <= band <= (int(iend) - 16) for band in band_for_gpoint],
        dtype=bool,
    )
    band_for_gpoint_arr = jnp.asarray(band_for_gpoint, dtype=jnp.int32)
    active = jnp.asarray(active_band, dtype=jnp.float32)

    adjflux_arr = jnp.asarray(adjflux, dtype=jnp.float32)
    if adjflux_arr.shape[0] >= 29:
        adjflux_by_band = adjflux_arr[15:29]
    else:
        adjflux_by_band = adjflux_arr[:RRTMG_SW_NBND]
    adjflux_g = adjflux_by_band[band_for_gpoint_arr]
    palbd_g = palbd_arr[band_for_gpoint_arr]
    palbp_g = palbp_arr[band_for_gpoint_arr]

    cloud = pcldfmc_arr[::-1, :]
    clear = 1.0 - cloud
    ptaucmc_rev = ptaucmc_arr[::-1, :]
    pasycmc_rev = pasycmc_arr[::-1, :]
    pomgcmc_rev = pomgcmc_arr[::-1, :]
    ptaormc_rev = ptaormc_arr[::-1, :]
    taug_rev = taug_arr[::-1, :]
    taur_rev = taur_arr[::-1, :]
    ptaua_rev = ptaua_arr[::-1, :][:, band_for_gpoint_arr]
    pasya_rev = pasya_arr[::-1, :][:, band_for_gpoint_arr]
    pomga_rev = pomga_arr[::-1, :][:, band_for_gpoint_arr]

    ztau_clear_raw = taur_rev + taug_rev + ptaua_rev
    zom_clear_raw = taur_rev + ptaua_rev * pomga_rev
    zg_clear_raw = pasya_rev * pomga_rev * ptaua_rev / zom_clear_raw
    zom_clear_raw = zom_clear_raw / ztau_clear_raw

    ze1_nodel = _wrf_real_div(ztau_clear_raw, prmu0_arr, rounding_token)
    zdbtc_nodel_layer = _sw_exp_lookup(
        ze1_nodel,
        tab,
        rounding_token=rounding_token,
    )
    tauorig = ztau_clear_raw + ptaormc_rev
    zdbtmo_nodel = _sw_exp_lookup(
        _wrf_real_div(tauorig, prmu0_arr, rounding_token),
        tab,
        rounding_token=rounding_token,
    )
    zdbt_nodel_layer = clear * zdbtc_nodel_layer + cloud * zdbtmo_nodel
    ones_g = jnp.ones((1, RRTMG_SW_NGPT), dtype=jnp.float32)
    ztdbtc_nodel = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbtc_nodel_layer, axis=0, rounding_token=rounding_token
            ),
        ],
        axis=0,
    )
    ztdbt_nodel = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbt_nodel_layer, axis=0, rounding_token=rounding_token
            ),
        ],
        axis=0,
    )

    zf = zg_clear_raw * zg_clear_raw
    zwf = zom_clear_raw * zf
    ztauc = (1.0 - zwf) * ztau_clear_raw
    zomcc = (zom_clear_raw - zwf) / (1.0 - zwf)
    zgcc = (zg_clear_raw - zf) / (1.0 - zf)

    if int(icpr) >= 1:
        ztauo = ztauc + ptaucmc_rev
        zomco = ztauc * zomcc + ptaucmc_rev * pomgcmc_rev
        zgco = (ptaucmc_rev * pomgcmc_rev * pasycmc_rev + ztauc * zomcc * zgcc) / zomco
        zomco = zomco / ztauo
    else:
        ztauo = taur_rev + taug_rev + ptaua_rev + ptaucmc_rev
        zomco = ptaua_rev * pomga_rev + ptaucmc_rev * pomgcmc_rev + taur_rev
        zgco = (ptaucmc_rev * pomgcmc_rev * pasycmc_rev + ptaua_rev * pomga_rev * pasya_rev) / zomco
        zomco = zomco / ztauo
        zf_o = zgco * zgco
        zwf_o = zomco * zf_o
        ztauo = (1.0 - zwf_o) * ztauo
        zomco = (zomco - zwf_o) / (1.0 - zwf_o)
        zgco = (zgco - zf_o) / (1.0 - zf_o)

    clear_ref = reftra_sw(
        nlayers=n,
        lrtchk=jnp.ones((n, RRTMG_SW_NGPT), dtype=bool),
        pgg=zgcc,
        prmuz=prmu0_arr,
        ptau=ztauc,
        pw=zomcc,
        tables=tab,
        rounding_token=rounding_token,
    )
    cloudy_ref = reftra_sw(
        nlayers=n,
        lrtchk=cloud > repclc,
        pgg=zgco,
        prmuz=prmu0_arr,
        ptau=ztauo,
        pw=zomco,
        tables=tab,
        rounding_token=rounding_token,
    )

    zrefc = _sw_boundary_with_layers(clear_ref.pref, palbp_g, n)
    zrefdc = _sw_boundary_with_layers(clear_ref.prefd, palbd_g, n)
    ztrac = _sw_boundary_with_layers(clear_ref.ptra, jnp.zeros_like(palbp_g), n)
    ztradc = _sw_boundary_with_layers(clear_ref.ptrad, jnp.zeros_like(palbp_g), n)
    zrefo = _sw_boundary_with_layers(cloudy_ref.pref, palbp_g, n)
    zrefdo = _sw_boundary_with_layers(cloudy_ref.prefd, palbd_g, n)
    ztrao = _sw_boundary_with_layers(cloudy_ref.ptra, jnp.zeros_like(palbp_g), n)
    ztrado = _sw_boundary_with_layers(cloudy_ref.ptrad, jnp.zeros_like(palbp_g), n)

    zref = _sw_boundary_with_layers(clear * zrefc[:n] + cloud * zrefo[:n], palbp_g, n)
    zrefd = _sw_boundary_with_layers(clear * zrefdc[:n] + cloud * zrefdo[:n], palbd_g, n)
    ztra = _sw_boundary_with_layers(clear * ztrac[:n] + cloud * ztrao[:n], jnp.zeros_like(palbp_g), n)
    ztrad = _sw_boundary_with_layers(clear * ztradc[:n] + cloud * ztrado[:n], jnp.zeros_like(palbp_g), n)

    ze1 = _wrf_real_div(ztauc, prmu0_arr, rounding_token)
    zdbtc_layer = _sw_exp_lookup(
        ze1,
        tab,
        rounding_token=rounding_token,
    )
    zdbtmo = _sw_exp_lookup(
        _wrf_real_div(ztauo, prmu0_arr, rounding_token),
        tab,
        rounding_token=rounding_token,
    )
    zdbt_layer = clear * zdbtc_layer + cloud * zdbtmo
    zdbtc = _sw_boundary_with_layers(zdbtc_layer, jnp.zeros_like(palbp_g), n)
    zdbt = _sw_boundary_with_layers(zdbt_layer, jnp.zeros_like(palbp_g), n)
    ztdbtc = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbtc_layer, axis=0, rounding_token=rounding_token
            ),
        ],
        axis=0,
    )
    ztdbt = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbt_layer, axis=0, rounding_token=rounding_token
            ),
        ],
        axis=0,
    )

    zrupc = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32).at[n].set(palbp_g)
    zrupdc = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32).at[n].set(palbd_g)
    zrup = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32).at[n].set(palbp_g)
    zrupd = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32).at[n].set(palbd_g)
    zrdndc = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32)
    zrdnd = jnp.zeros((n + 1, RRTMG_SW_NGPT), dtype=jnp.float32)

    clear_flux = vrtqdr_sw(
        klev=n,
        pref=zrefc,
        prefd=zrefdc,
        ptra=ztrac,
        ptrad=ztradc,
        pdbt=zdbtc,
        prdnd=zrdndc,
        prup=zrupc,
        prupd=zrupdc,
        ptdbt=ztdbtc,
        rounding_token=rounding_token,
    )
    total_flux = vrtqdr_sw(
        klev=n,
        pref=zref,
        prefd=zrefd,
        ptra=ztra,
        ptrad=ztrad,
        pdbt=zdbt,
        prdnd=zrdnd,
        prup=zrup,
        prupd=zrupd,
        ptdbt=ztdbt,
        rounding_token=rounding_token,
    )

    zincflx = adjflux_g * sfluxzen_arr * prmu0_arr * active
    if _CAPTURE_RRTMG_SW_TRANSPORT_PATH:
        jax.debug.callback(
            _capture_rrtmg_sw_transport_host,
            zincflx,
            total_flux.pfd,
            total_flux.pfu,
            clear_flux.pfd,
            clear_flux.pfu,
            ztau_clear_raw,
            zom_clear_raw,
            zg_clear_raw,
            prmu0_arr,
            ze1_nodel,
            zdbtc_nodel_layer,
            ztauc,
            zomcc,
            zgcc,
            ze1,
            zdbtc_layer,
            ztdbt_nodel,
            ztdbtc_nodel,
            ordered=True,
        )
    broadband = zincflx[None, :]
    ibm_g = band_for_gpoint_arr + 1
    visible = ((ibm_g >= 10) & (ibm_g <= 13)).astype(jnp.float32) * active
    near_ir = ((ibm_g == 14) | (ibm_g <= 9)).astype(jnp.float32) * active
    visible_w = zincflx[None, :] * visible[None, :]
    near_ir_w = zincflx[None, :] * near_ir[None, :]

    pbbfu = _wrf_reversed_flux_sum(total_flux.pfu, broadband)
    pbbfd = _wrf_reversed_flux_sum(total_flux.pfd, broadband)
    pbbcu = _wrf_reversed_flux_sum(clear_flux.pfu, broadband)
    pbbcd = _wrf_reversed_flux_sum(clear_flux.pfd, broadband)
    pbbfddir = _wrf_reversed_flux_sum(ztdbt_nodel, broadband)
    pbbcddir = _wrf_reversed_flux_sum(ztdbtc_nodel, broadband)

    puvcd = _wrf_reversed_flux_sum(clear_flux.pfd, visible_w)
    puvfd = _wrf_reversed_flux_sum(total_flux.pfd, visible_w)
    puvcddir = _wrf_reversed_flux_sum(ztdbtc_nodel, visible_w)
    puvfddir = _wrf_reversed_flux_sum(ztdbt_nodel, visible_w)
    pnicd = _wrf_reversed_flux_sum(clear_flux.pfd, near_ir_w)
    pnifd = _wrf_reversed_flux_sum(total_flux.pfd, near_ir_w)
    pnicddir = _wrf_reversed_flux_sum(ztdbtc_nodel, near_ir_w)
    pnifddir = _wrf_reversed_flux_sum(ztdbt_nodel, near_ir_w)

    return RrtmgSwSpcvmcResult(
        pbbfd=pbbfd,
        pbbfu=pbbfu,
        pbbcd=pbbcd,
        pbbcu=pbbcu,
        puvfd=puvfd,
        puvcd=puvcd,
        pnifd=pnifd,
        pnicd=pnicd,
        pbbfddir=pbbfddir,
        pbbcddir=pbbcddir,
        puvfddir=puvfddir,
        puvcddir=puvcddir,
        pnifddir=pnifddir,
        pnicddir=pnicddir,
    )


def taumol16(
    *,
    nlayers: int,
    colh2o: Any,
    colch4: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol16`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colch4_arr = jnp.asarray(colch4, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat1 = jnp.asarray(band.strrat1, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    trop_mask = layer_index < jnp.asarray(laytrop, dtype=jnp.int32)

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat1, colch4_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js = 1 + specmult.astype(jnp.int32)
    fs = _wrf_real_sub(specmult, jnp.floor(specmult), rounding_token)
    compfs = _wrf_real_sub(jnp.float32(1.0), fs, rounding_token)

    fac000 = _wrf_real_mul(compfs, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs, fac11_arr, rounding_token)

    js0 = jnp.clip(js - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1

    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 1)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
        continuum,
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_mul(colch4_arr[:, None], strat_abs, rounding_token)

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )
    has_stratosphere = jnp.asarray(laytrop, dtype=jnp.int32) < nlay
    sfluxzen = jnp.where(has_stratosphere, jnp.asarray(band.sfluxref, dtype=jnp.float32), jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol17(
    *,
    nlayers: int,
    colh2o: Any,
    colco2: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol17`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat, colco2_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult_trop = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js_trop = 1 + specmult_trop.astype(jnp.int32)
    fs_trop = _wrf_real_sub(specmult_trop, jnp.floor(specmult_trop), rounding_token)
    compfs_trop = _wrf_real_sub(jnp.float32(1.0), fs_trop, rounding_token)
    fac000 = _wrf_real_mul(compfs_trop, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs_trop, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs_trop, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs_trop, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs_trop, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs_trop, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs_trop, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs_trop, fac11_arr, rounding_token)

    js0 = jnp.clip(js_trop - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    trop_continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
        trop_continuum,
        rounding_token,
    )

    specmult_strat = _wrf_real_mul(jnp.float32(4.0), specparm, rounding_token)
    js_strat = 1 + specmult_strat.astype(jnp.int32)
    fs_strat = _wrf_real_sub(specmult_strat, jnp.floor(specmult_strat), rounding_token)
    compfs_strat = _wrf_real_sub(jnp.float32(1.0), fs_strat, rounding_token)
    s000 = _wrf_real_mul(compfs_strat, fac00_arr, rounding_token)
    s010 = _wrf_real_mul(compfs_strat, fac10_arr, rounding_token)
    s100 = _wrf_real_mul(fs_strat, fac00_arr, rounding_token)
    s110 = _wrf_real_mul(fs_strat, fac10_arr, rounding_token)
    s001 = _wrf_real_mul(compfs_strat, fac01_arr, rounding_token)
    s011 = _wrf_real_mul(compfs_strat, fac11_arr, rounding_token)
    s101 = _wrf_real_mul(fs_strat, fac01_arr, rounding_token)
    s111 = _wrf_real_mul(fs_strat, fac11_arr, rounding_token)
    sjs0 = jnp.clip(js_strat - 1, 0, 3)
    sjs1 = sjs0 + 1
    sjt0 = jnp.clip(jt_arr - 1, 0, 3)
    sjt1 = sjt0 + 1
    sjt10 = jnp.clip(jt1_arr - 1, 0, 3)
    sjt11 = sjt10 + 1
    sjp0 = jnp.clip(jp_arr - 13, 0, 45)
    sjp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            s000[:, None],
            s100[:, None],
            s010[:, None],
            s110[:, None],
            s001[:, None],
            s101[:, None],
            s011[:, None],
            s111[:, None],
        ),
        (
            kb[sjs0, sjt0, sjp0, :],
            kb[sjs1, sjt0, sjp0, :],
            kb[sjs0, sjt1, sjp0, :],
            kb[sjs1, sjt1, sjp0, :],
            kb[sjs0, sjt10, sjp1, :],
            kb[sjs1, sjt10, sjp1, :],
            kb[sjs0, sjt11, sjp1, :],
            kb[sjs1, sjt11, sjp1, :],
        ),
        rounding_token,
    )
    strat_continuum = _wrf_real_mul(
        _wrf_real_mul(colh2o_arr[:, None], forfac_arr[:, None], rounding_token),
        foreign_interp,
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], strat_abs, rounding_token),
        strat_continuum,
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    prev_jp = jnp.concatenate((jp_arr[:1], jp_arr[:-1]))
    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    crossing = (layer_index > 0) & (layer_index >= laytrop_i) & (prev_jp < layreffr) & (jp_arr >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.argmax(crossing.astype(jnp.int32)), nlay - 1)
    flux_js0 = jnp.clip(js_strat[laysolfr_idx] - 1, 0, 3)
    flux_fs = fs_strat[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i < nlay, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol18(
    *,
    nlayers: int,
    colh2o: Any,
    colch4: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol18`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colch4_arr = jnp.asarray(colch4, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat, colch4_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js = 1 + specmult.astype(jnp.int32)
    fs = _wrf_real_sub(specmult, jnp.floor(specmult), rounding_token)
    compfs = _wrf_real_sub(jnp.float32(1.0), fs, rounding_token)

    fac000 = _wrf_real_mul(compfs, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs, fac11_arr, rounding_token)

    js0 = jnp.clip(js - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
        continuum,
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_mul(colch4_arr[:, None], strat_abs, rounding_token)

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    # WRF initializes laysolfr=laytrop and then uses min(lay+1, laytrop).
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    flux_js0 = jnp.clip(js[laysolfr_idx] - 1, 0, 7)
    flux_fs = fs[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i > 0, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol19(
    *,
    nlayers: int,
    colh2o: Any,
    colco2: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol19`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat, colco2_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js = 1 + specmult.astype(jnp.int32)
    fs = _wrf_real_sub(specmult, jnp.floor(specmult), rounding_token)
    compfs = _wrf_real_sub(jnp.float32(1.0), fs, rounding_token)

    fac000 = _wrf_real_mul(compfs, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs, fac11_arr, rounding_token)

    js0 = jnp.clip(js - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
        continuum,
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_mul(colco2_arr[:, None], strat_abs, rounding_token)

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    flux_js0 = jnp.clip(js[laysolfr_idx] - 1, 0, 7)
    flux_fs = fs[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i > 0, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol20(
    *,
    nlayers: int,
    colh2o: Any,
    colch4: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol20`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colch4_arr = jnp.asarray(colch4, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    absch4 = jnp.asarray(band.absch4, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            ka[jt0, jp0, :],
            ka[jt1p, jp0, :],
            ka[jt10, jp1p, :],
            ka[jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    trop_inner = _wrf_real_add(
        _wrf_real_add(
            trop_abs,
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], trop_inner, rounding_token),
        _wrf_real_mul(colch4_arr[:, None], absch4[None, :], rounding_token),
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    strat_inner = _wrf_real_add(
        strat_abs,
        _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], strat_inner, rounding_token),
        _wrf_real_mul(colch4_arr[:, None], absch4[None, :], rounding_token),
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    sfluxzen = jnp.where(laytrop_i > 0, sfluxref, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol21(
    *,
    nlayers: int,
    colh2o: Any,
    colco2: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol21`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat, colco2_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult_trop = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js_trop = 1 + specmult_trop.astype(jnp.int32)
    fs_trop = _wrf_real_sub(
        specmult_trop,
        jnp.floor(specmult_trop),
        rounding_token,
    )
    compfs_trop = _wrf_real_sub(jnp.float32(1.0), fs_trop, rounding_token)

    fac000 = _wrf_real_mul(compfs_trop, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs_trop, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs_trop, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs_trop, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs_trop, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs_trop, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs_trop, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs_trop, fac11_arr, rounding_token)

    js0 = jnp.clip(js_trop - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    trop_continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
        trop_continuum,
        rounding_token,
    )

    specmult_strat = _wrf_real_mul(jnp.float32(4.0), specparm, rounding_token)
    js_strat = 1 + specmult_strat.astype(jnp.int32)
    fs_strat = _wrf_real_sub(
        specmult_strat,
        jnp.floor(specmult_strat),
        rounding_token,
    )
    compfs_strat = _wrf_real_sub(jnp.float32(1.0), fs_strat, rounding_token)
    s000 = _wrf_real_mul(compfs_strat, fac00_arr, rounding_token)
    s010 = _wrf_real_mul(compfs_strat, fac10_arr, rounding_token)
    s100 = _wrf_real_mul(fs_strat, fac00_arr, rounding_token)
    s110 = _wrf_real_mul(fs_strat, fac10_arr, rounding_token)
    s001 = _wrf_real_mul(compfs_strat, fac01_arr, rounding_token)
    s011 = _wrf_real_mul(compfs_strat, fac11_arr, rounding_token)
    s101 = _wrf_real_mul(fs_strat, fac01_arr, rounding_token)
    s111 = _wrf_real_mul(fs_strat, fac11_arr, rounding_token)
    sjs0 = jnp.clip(js_strat - 1, 0, 3)
    sjs1 = sjs0 + 1
    sjt0 = jnp.clip(jt_arr - 1, 0, 3)
    sjt1 = sjt0 + 1
    sjt10 = jnp.clip(jt1_arr - 1, 0, 3)
    sjt11 = sjt10 + 1
    sjp0 = jnp.clip(jp_arr - 13, 0, 45)
    sjp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            s000[:, None],
            s100[:, None],
            s010[:, None],
            s110[:, None],
            s001[:, None],
            s101[:, None],
            s011[:, None],
            s111[:, None],
        ),
        (
            kb[sjs0, sjt0, sjp0, :],
            kb[sjs1, sjt0, sjp0, :],
            kb[sjs0, sjt1, sjp0, :],
            kb[sjs1, sjt1, sjp0, :],
            kb[sjs0, sjt10, sjp1, :],
            kb[sjs1, sjt10, sjp1, :],
            kb[sjs0, sjt11, sjp1, :],
            kb[sjs1, sjt11, sjp1, :],
        ),
        rounding_token,
    )
    strat_continuum = _wrf_real_mul(
        _wrf_real_mul(colh2o_arr[:, None], forfac_arr[:, None], rounding_token),
        foreign_interp,
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(speccomb[:, None], strat_abs, rounding_token),
        strat_continuum,
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    flux_js0 = jnp.clip(js_trop[laysolfr_idx] - 1, 0, 7)
    flux_fs = fs_trop[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i > 0, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol22(
    *,
    nlayers: int,
    colh2o: Any,
    colo2: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol22`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colo2_arr = jnp.asarray(colo2, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    o2adj = jnp.asarray(1.6, dtype=jnp.float32)
    o2cont = _wrf_real_div(
        _wrf_real_mul(jnp.float32(4.35e-4), colo2_arr, rounding_token),
        _wrf_real_mul(jnp.float32(350.0), jnp.float32(2.0), rounding_token),
        rounding_token,
    )
    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(
            _wrf_real_mul(o2adj, strrat, rounding_token),
            colo2_arr,
            rounding_token,
        ),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js = 1 + specmult.astype(jnp.int32)
    fs = _wrf_real_sub(specmult, jnp.floor(specmult), rounding_token)

    compfs = _wrf_real_sub(jnp.float32(1.0), fs, rounding_token)
    fac000 = _wrf_real_mul(compfs, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs, fac11_arr, rounding_token)

    js0 = jnp.clip(js - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
            continuum,
            rounding_token,
        ),
        o2cont[:, None],
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(
            _wrf_real_mul(colo2_arr[:, None], o2adj, rounding_token),
            strat_abs,
            rounding_token,
        ),
        o2cont[:, None],
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    flux_js0 = jnp.clip(js[laysolfr_idx] - 1, 0, 7)
    flux_fs = fs[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i > 0, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol23(
    *,
    nlayers: int,
    colh2o: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol23`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    givfac = jnp.asarray(band.givfac, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    base_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            ka[jt0, jp0, :],
            ka[jt1p, jp0, :],
            ka[jt10, jp1p, :],
            ka[jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 1)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    optical_sum = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(givfac, base_abs, rounding_token),
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
        rounding_token,
    )
    taug_trop = _wrf_real_mul(
        colh2o_arr[:, None],
        optical_sum,
        rounding_token,
    )
    taug = jnp.where(trop_mask[:, None], taug_trop, jnp.zeros((nlay, ng), dtype=jnp.float32))
    taur = _wrf_real_mul(
        colmol_arr[:, None],
        rayl[None, :],
        rounding_token,
    )

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    sfluxzen = jnp.where(laytrop_i > 0, sfluxref, jnp.zeros((ng,), dtype=jnp.float32))
    del laysolfr_idx
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol24(
    *,
    nlayers: int,
    colh2o: Any,
    colo2: Any,
    colo3: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol24`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colo2_arr = jnp.asarray(colo2, dtype=jnp.float32)[:nlay]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    rayla = jnp.asarray(band.rayla, dtype=jnp.float32)
    raylb = jnp.asarray(band.raylb, dtype=jnp.float32)
    abso3a = jnp.asarray(band.abso3a, dtype=jnp.float32)
    abso3b = jnp.asarray(band.abso3b, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colh2o_arr,
        _wrf_real_mul(strrat, colo2_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colh2o_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )
    specmult = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js = 1 + specmult.astype(jnp.int32)
    fs = _wrf_real_sub(specmult, jnp.floor(specmult), rounding_token)

    compfs = _wrf_real_sub(jnp.float32(1.0), fs, rounding_token)
    fac000 = _wrf_real_mul(compfs, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs, fac11_arr, rounding_token)

    js0 = jnp.clip(js - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 1)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    continuum = _wrf_real_mul(
        colh2o_arr[:, None],
        _wrf_real_add(
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token),
            _wrf_real_mul(colo3_arr[:, None], abso3a[None, :], rounding_token),
            rounding_token,
        ),
        continuum,
        rounding_token,
    )
    rayleigh_trop = _wrf_real_add(
        rayla[:, js0].T,
        _wrf_real_mul(
            fs[:, None],
            _wrf_real_sub(rayla[:, js1].T, rayla[:, js0].T, rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    taur_trop = _wrf_real_mul(
        colmol_arr[:, None],
        rayleigh_trop,
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(colo2_arr[:, None], strat_abs, rounding_token),
        _wrf_real_mul(colo3_arr[:, None], abso3b[None, :], rounding_token),
        rounding_token,
    )
    taur_strat = _wrf_real_mul(
        colmol_arr[:, None],
        raylb[None, :],
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.where(trop_mask[:, None], taur_trop, taur_strat)

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    flux_js0 = jnp.clip(js[laysolfr_idx] - 1, 0, 7)
    flux_fs = fs[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i > 0, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol25(
    *,
    nlayers: int,
    colh2o: Any,
    colo3: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    band: RrtmgSwCompressedBandData,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol25`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    abso3a = jnp.asarray(band.abso3a, dtype=jnp.float32)
    abso3b = jnp.asarray(band.abso3b, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            ka[jt0, jp0, :],
            ka[jt1p, jp0, :],
            ka[jt10, jp1p, :],
            ka[jt11, jp1p, :],
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], trop_abs, rounding_token),
        _wrf_real_mul(colo3_arr[:, None], abso3a[None, :], rounding_token),
        rounding_token,
    )
    taug_strat = _wrf_real_mul(colo3_arr[:, None], abso3b[None, :], rounding_token)
    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = _wrf_real_mul(colmol_arr[:, None], rayl[None, :], rounding_token)

    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    next_jp = jnp.concatenate((jp_arr[1:], jp_arr[-1:]))
    crossing = trop_mask & (jp_arr < layreffr) & (next_jp >= layreffr)
    has_crossing = jnp.any(crossing)
    _laysolfr_idx = jnp.where(has_crossing, jnp.minimum(jnp.argmax(crossing.astype(jnp.int32)) + 1, laytrop_i - 1), laytrop_i - 1)
    sfluxzen = jnp.where(laytrop_i > 0, sfluxref, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol26(
    *,
    nlayers: int,
    colmol: Any,
    laytrop: Any,
    band: RrtmgSwCompressedBandData,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol26`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    ng = int(rayl.shape[0])
    taug = jnp.zeros((nlay, ng), dtype=jnp.float32)
    taur = colmol_arr[:, None] * rayl[None, :]
    sfluxzen = jnp.where(jnp.asarray(laytrop, dtype=jnp.int32) > 0, sfluxref, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol27(
    *,
    nlayers: int,
    colo3: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    band: RrtmgSwCompressedBandData,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol27`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    scalekur = jnp.asarray(band.scalekur, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            ka[jt0, jp0, :],
            ka[jt1p, jp0, :],
            ka[jt10, jp1p, :],
            ka[jt11, jp1p, :],
        ),
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )

    taug = _wrf_real_mul(
        colo3_arr[:, None],
        jnp.where(trop_mask[:, None], trop_abs, strat_abs),
        rounding_token,
    )
    taur = _wrf_real_mul(
        colmol_arr[:, None],
        rayl[None, :],
        rounding_token,
    )
    sfluxzen = jnp.where(
        laytrop_i < nlay,
        _wrf_real_mul(scalekur, sfluxref, rounding_token),
        jnp.zeros((ng,), dtype=jnp.float32),
    )
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol28(
    *,
    nlayers: int,
    colo2: Any,
    colo3: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    band: RrtmgSwCompressedBandData,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol28`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colo2_arr = jnp.asarray(colo2, dtype=jnp.float32)[:nlay]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    strrat = jnp.asarray(band.strrat, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    speccomb = _wrf_real_add(
        colo3_arr,
        _wrf_real_mul(strrat, colo2_arr, rounding_token),
        rounding_token,
    )
    specparm = jnp.minimum(
        _wrf_real_div(colo3_arr, speccomb, rounding_token),
        jnp.asarray(oneminus, dtype=jnp.float32),
    )

    specmult_trop = _wrf_real_mul(jnp.float32(8.0), specparm, rounding_token)
    js_trop = 1 + specmult_trop.astype(jnp.int32)
    fs_trop = _wrf_real_sub(specmult_trop, jnp.floor(specmult_trop), rounding_token)
    compfs_trop = _wrf_real_sub(jnp.float32(1.0), fs_trop, rounding_token)
    fac000 = _wrf_real_mul(compfs_trop, fac00_arr, rounding_token)
    fac010 = _wrf_real_mul(compfs_trop, fac10_arr, rounding_token)
    fac100 = _wrf_real_mul(fs_trop, fac00_arr, rounding_token)
    fac110 = _wrf_real_mul(fs_trop, fac10_arr, rounding_token)
    fac001 = _wrf_real_mul(compfs_trop, fac01_arr, rounding_token)
    fac011 = _wrf_real_mul(compfs_trop, fac11_arr, rounding_token)
    fac101 = _wrf_real_mul(fs_trop, fac01_arr, rounding_token)
    fac111 = _wrf_real_mul(fs_trop, fac11_arr, rounding_token)
    js0 = jnp.clip(js_trop - 1, 0, 7)
    js1 = js0 + 1
    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
            fac001[:, None],
            fac101[:, None],
            fac011[:, None],
            fac111[:, None],
        ),
        (
            ka[js0, jt0, jp0, :],
            ka[js1, jt0, jp0, :],
            ka[js0, jt1p, jp0, :],
            ka[js1, jt1p, jp0, :],
            ka[js0, jt10, jp1p, :],
            ka[js1, jt10, jp1p, :],
            ka[js0, jt11, jp1p, :],
            ka[js1, jt11, jp1p, :],
        ),
        rounding_token,
    )
    taug_trop = _wrf_real_mul(speccomb[:, None], trop_abs, rounding_token)

    specmult_strat = _wrf_real_mul(jnp.float32(4.0), specparm, rounding_token)
    js_strat = 1 + specmult_strat.astype(jnp.int32)
    fs_strat = _wrf_real_sub(specmult_strat, jnp.floor(specmult_strat), rounding_token)
    compfs_strat = _wrf_real_sub(jnp.float32(1.0), fs_strat, rounding_token)
    s000 = _wrf_real_mul(compfs_strat, fac00_arr, rounding_token)
    s010 = _wrf_real_mul(compfs_strat, fac10_arr, rounding_token)
    s100 = _wrf_real_mul(fs_strat, fac00_arr, rounding_token)
    s110 = _wrf_real_mul(fs_strat, fac10_arr, rounding_token)
    s001 = _wrf_real_mul(compfs_strat, fac01_arr, rounding_token)
    s011 = _wrf_real_mul(compfs_strat, fac11_arr, rounding_token)
    s101 = _wrf_real_mul(fs_strat, fac01_arr, rounding_token)
    s111 = _wrf_real_mul(fs_strat, fac11_arr, rounding_token)
    sjs0 = jnp.clip(js_strat - 1, 0, 3)
    sjs1 = sjs0 + 1
    sjt0 = jnp.clip(jt_arr - 1, 0, 3)
    sjt1 = sjt0 + 1
    sjt10 = jnp.clip(jt1_arr - 1, 0, 3)
    sjt11 = sjt10 + 1
    sjp0 = jnp.clip(jp_arr - 13, 0, 45)
    sjp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            s000[:, None],
            s100[:, None],
            s010[:, None],
            s110[:, None],
            s001[:, None],
            s101[:, None],
            s011[:, None],
            s111[:, None],
        ),
        (
            kb[sjs0, sjt0, sjp0, :],
            kb[sjs1, sjt0, sjp0, :],
            kb[sjs0, sjt1, sjp0, :],
            kb[sjs1, sjt1, sjp0, :],
            kb[sjs0, sjt10, sjp1, :],
            kb[sjs1, sjt10, sjp1, :],
            kb[sjs0, sjt11, sjp1, :],
            kb[sjs1, sjt11, sjp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_mul(speccomb[:, None], strat_abs, rounding_token)
    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )

    prev_jp = jnp.concatenate((jp_arr[:1], jp_arr[:-1]))
    layreffr = jnp.asarray(band.layreffr, dtype=jnp.int32)
    crossing = (layer_index > 0) & (layer_index >= laytrop_i) & (prev_jp < layreffr) & (jp_arr >= layreffr)
    has_crossing = jnp.any(crossing)
    laysolfr_idx = jnp.where(has_crossing, jnp.argmax(crossing.astype(jnp.int32)), nlay - 1)
    flux_js0 = jnp.clip(js_strat[laysolfr_idx] - 1, 0, 3)
    flux_fs = fs_strat[laysolfr_idx]
    flux_table = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    flux = _wrf_real_add(
        flux_table[:, flux_js0],
        _wrf_real_mul(
            flux_fs,
            _wrf_real_sub(
                flux_table[:, flux_js0 + 1],
                flux_table[:, flux_js0],
                rounding_token,
            ),
            rounding_token,
        ),
        rounding_token,
    )
    sfluxzen = jnp.where(laytrop_i < nlay, flux, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol29(
    *,
    nlayers: int,
    colh2o: Any,
    colco2: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    band: RrtmgSwCompressedBandData,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol29`` shortwave optical-depth kernel for one column."""

    nlay = int(nlayers)
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:nlay]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:nlay]
    colmol_arr = jnp.asarray(colmol, dtype=jnp.float32)[:nlay]
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:nlay]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:nlay]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:nlay]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:nlay]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:nlay]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:nlay]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:nlay]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:nlay]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:nlay]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:nlay]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:nlay]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:nlay]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:nlay]

    ka = jnp.asarray(band.ka, dtype=jnp.float32)
    kb = jnp.asarray(band.kb, dtype=jnp.float32)
    selfref = jnp.asarray(band.selfref, dtype=jnp.float32)
    forref = jnp.asarray(band.forref, dtype=jnp.float32)
    absh2o = jnp.asarray(band.absh2o, dtype=jnp.float32)
    absco2 = jnp.asarray(band.absco2, dtype=jnp.float32)
    rayl = jnp.asarray(band.rayl, dtype=jnp.float32)
    sfluxref = jnp.asarray(band.sfluxref, dtype=jnp.float32)
    ng = int(ka.shape[-1])

    layer_index = jnp.arange(nlay, dtype=jnp.int32)
    laytrop_i = jnp.asarray(laytrop, dtype=jnp.int32)
    trop_mask = layer_index < laytrop_i

    jt0 = jnp.clip(jt_arr - 1, 0, 3)
    jt1p = jt0 + 1
    jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    jt11 = jt10 + 1
    jp0 = jnp.clip(jp_arr - 1, 0, 11)
    jp1p = jp0 + 1
    trop_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            ka[jt0, jp0, :],
            ka[jt1p, jp0, :],
            ka[jt10, jp1p, :],
            ka[jt11, jp1p, :],
        ),
        rounding_token,
    )
    inds = jnp.clip(indself_arr - 1, 0, 8)
    indsp = inds + 1
    indf = jnp.clip(indfor_arr - 1, 0, 2)
    indfp = indf + 1
    self_interp = _wrf_real_add(
        selfref[inds, :],
        _wrf_real_mul(
            selffrac_arr[:, None],
            _wrf_real_sub(selfref[indsp, :], selfref[inds, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    foreign_interp = _wrf_real_add(
        forref[indf, :],
        _wrf_real_mul(
            forfrac_arr[:, None],
            _wrf_real_sub(forref[indfp, :], forref[indf, :], rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    water_optical = _wrf_real_add(
        _wrf_real_add(
            trop_abs,
            _wrf_real_mul(selffac_arr[:, None], self_interp, rounding_token),
            rounding_token,
        ),
        _wrf_real_mul(forfac_arr[:, None], foreign_interp, rounding_token),
        rounding_token,
    )
    taug_trop = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], water_optical, rounding_token),
        _wrf_real_mul(colco2_arr[:, None], absco2[None, :], rounding_token),
        rounding_token,
    )

    kb_jt0 = jnp.clip(jt_arr - 1, 0, 3)
    kb_jt1 = kb_jt0 + 1
    kb_jt10 = jnp.clip(jt1_arr - 1, 0, 3)
    kb_jt11 = kb_jt10 + 1
    kb_jp0 = jnp.clip(jp_arr - 13, 0, 45)
    kb_jp1 = jnp.clip(jp_arr - 12, 0, 46)
    strat_abs = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            kb[kb_jt0, kb_jp0, :],
            kb[kb_jt1, kb_jp0, :],
            kb[kb_jt10, kb_jp1, :],
            kb[kb_jt11, kb_jp1, :],
        ),
        rounding_token,
    )
    taug_strat = _wrf_real_add(
        _wrf_real_mul(colco2_arr[:, None], strat_abs, rounding_token),
        _wrf_real_mul(colh2o_arr[:, None], absh2o[None, :], rounding_token),
        rounding_token,
    )

    taug = jnp.where(trop_mask[:, None], taug_trop, taug_strat)
    taur = jnp.broadcast_to(
        _wrf_real_mul(colmol_arr, rayl, rounding_token)[:, None],
        (nlay, ng),
    )
    sfluxzen = jnp.where(laytrop_i < nlay, sfluxref, jnp.zeros((ng,), dtype=jnp.float32))
    return RrtmgSwTaumolBandResult(sfluxzen=sfluxzen, taug=taug, taur=taur)


def taumol_sw(
    *,
    nlayers: int,
    colh2o: Any,
    colco2: Any,
    colch4: Any,
    colo2: Any,
    colo3: Any,
    colmol: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    tables: RrtmgSwCompressedLookupTables,
    oneminus: float = 1.0 - 1.0e-6,
    rounding_token: Any | None = None,
) -> RrtmgSwTaumolBandResult:
    """WRF ``taumol_sw`` shortwave optical-depth wrapper for one column."""

    results = (
        taumol16(
            nlayers=nlayers,
            colh2o=colh2o,
            colch4=colch4,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band16,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol17(
            nlayers=nlayers,
            colh2o=colh2o,
            colco2=colco2,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band17,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol18(
            nlayers=nlayers,
            colh2o=colh2o,
            colch4=colch4,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band18,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol19(
            nlayers=nlayers,
            colh2o=colh2o,
            colco2=colco2,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band19,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol20(
            nlayers=nlayers,
            colh2o=colh2o,
            colch4=colch4,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band20,
            rounding_token=rounding_token,
        ),
        taumol21(
            nlayers=nlayers,
            colh2o=colh2o,
            colco2=colco2,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band21,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol22(
            nlayers=nlayers,
            colh2o=colh2o,
            colo2=colo2,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band22,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol23(
            nlayers=nlayers,
            colh2o=colh2o,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band23,
            rounding_token=rounding_token,
        ),
        taumol24(
            nlayers=nlayers,
            colh2o=colh2o,
            colo2=colo2,
            colo3=colo3,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band24,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol25(
            nlayers=nlayers,
            colh2o=colh2o,
            colo3=colo3,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            band=tables.band25,
            rounding_token=rounding_token,
        ),
        taumol26(
            nlayers=nlayers,
            colmol=colmol,
            laytrop=laytrop,
            band=tables.band26,
        ),
        taumol27(
            nlayers=nlayers,
            colo3=colo3,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            band=tables.band27,
            rounding_token=rounding_token,
        ),
        taumol28(
            nlayers=nlayers,
            colo2=colo2,
            colo3=colo3,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            band=tables.band28,
            oneminus=oneminus,
            rounding_token=rounding_token,
        ),
        taumol29(
            nlayers=nlayers,
            colh2o=colh2o,
            colco2=colco2,
            colmol=colmol,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            band=tables.band29,
            rounding_token=rounding_token,
        ),
    )
    return RrtmgSwTaumolBandResult(
        sfluxzen=jnp.concatenate([result.sfluxzen for result in results], axis=0),
        taug=jnp.concatenate([result.taug for result in results], axis=1),
        taur=jnp.concatenate([result.taur for result in results], axis=1),
    )


def spcvmc_sw(
    *,
    nlayers: int,
    istart: int = 16,
    iend: int = 29,
    icpr: int = 1,
    iout: int = 0,
    pavel: Any | None = None,
    tavel: Any | None = None,
    pz: Any | None = None,
    tz: Any | None = None,
    tbound: Any | None = None,
    palbd: Any,
    palbp: Any,
    pcldfmc: Any,
    ptaucmc: Any,
    pasycmc: Any,
    pomgcmc: Any,
    ptaormc: Any,
    ptaua: Any,
    pasya: Any,
    pomga: Any,
    prmu0: Any,
    coldry: Any | None = None,
    wkl: Any | None = None,
    adjflux: Any,
    laytrop: Any,
    layswtch: Any | None = None,
    laylow: Any | None = None,
    jp: Any,
    jt: Any,
    jt1: Any,
    co2mult: Any | None = None,
    colch4: Any,
    colco2: Any,
    colh2o: Any,
    colmol: Any,
    coln2o: Any | None = None,
    colo2: Any,
    colo3: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    lookup_tables: RrtmgSwCompressedLookupTables,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgSwPadeTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwSpcvmcResult:
    """WRF ``spcvmc_sw`` wrapper using the pure ``taumol_sw`` optical depths."""

    del pavel, tavel, pz, tz, tbound, coldry, wkl, layswtch, laylow, co2mult, coln2o
    optical = taumol_sw(
        nlayers=nlayers,
        colh2o=colh2o,
        colco2=colco2,
        colch4=colch4,
        colo2=colo2,
        colo3=colo3,
        colmol=colmol,
        laytrop=laytrop,
        jp=jp,
        jt=jt,
        jt1=jt1,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        selffac=selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=forfac,
        forfrac=forfrac,
        indfor=indfor,
        tables=lookup_tables,
        rounding_token=rounding_token,
    )
    return spcvmc_sw_from_optical_depths(
        nlayers=nlayers,
        istart=istart,
        iend=iend,
        icpr=icpr,
        iout=iout,
        palbd=palbd,
        palbp=palbp,
        pcldfmc=pcldfmc,
        ptaucmc=ptaucmc,
        pasycmc=pasycmc,
        pomgcmc=pomgcmc,
        ptaormc=ptaormc,
        ptaua=ptaua,
        pasya=pasya,
        pomga=pomga,
        prmu0=prmu0,
        adjflux=adjflux,
        sfluxzen=optical.sfluxzen,
        taug=optical.taug,
        taur=optical.taur,
        spectral=spectral,
        tables=pade_tables,
        rounding_token=rounding_token,
    )


def spcvmc_sw_from_optical_depths_columns(
    *,
    nlayers: int,
    istart: int = 16,
    iend: int = 29,
    icpr: int = 1,
    iout: int = 0,
    palbd: Any,
    palbp: Any,
    pcldfmc: Any,
    ptaucmc: Any,
    pasycmc: Any,
    pomgcmc: Any,
    ptaormc: Any,
    ptaua: Any,
    pasya: Any,
    pomga: Any,
    prmu0: Any,
    adjflux: Any,
    sfluxzen: Any,
    taug: Any,
    taur: Any,
    spectral: RrtmgSpectralInit | None = None,
    tables: RrtmgSwPadeTables | None = None,
    rounding_token: Any | None = None,
    clear_only: bool = False,
) -> RrtmgSwSpcvmcResult:
    """Batch ``spcvmc_sw_from_optical_depths`` over horizontal columns."""

    n = int(nlayers)
    tab = rrtmg_sw_pade_tables() if tables is None else tables
    if spectral is None:
        ngc = _SW_NGC_VALUES
    else:
        try:
            ngc = tuple(int(x) for x in np.asarray(spectral.ngc))
        except Exception:
            ngc = _SW_NGC_VALUES
    prmu0_arr = jnp.asarray(prmu0, dtype=jnp.float32)
    repclc = jnp.asarray(1.0e-12, dtype=jnp.float32)

    palbd_arr = jnp.asarray(palbd, dtype=jnp.float32)[:, :RRTMG_SW_NBND]
    palbp_arr = jnp.asarray(palbp, dtype=jnp.float32)[:, :RRTMG_SW_NBND]
    pcldfmc_arr = jnp.asarray(pcldfmc, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    ptaucmc_arr = jnp.asarray(ptaucmc, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    pasycmc_arr = jnp.asarray(pasycmc, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    pomgcmc_arr = jnp.asarray(pomgcmc, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    ptaormc_arr = jnp.asarray(ptaormc, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    has_aerosol = ptaua is not None and pasya is not None and pomga is not None
    if has_aerosol:
        ptaua_arr = jnp.asarray(ptaua, dtype=jnp.float32)[:, :n, :RRTMG_SW_NBND]
        pasya_arr = jnp.asarray(pasya, dtype=jnp.float32)[:, :n, :RRTMG_SW_NBND]
        pomga_arr = jnp.asarray(pomga, dtype=jnp.float32)[:, :n, :RRTMG_SW_NBND]
    sfluxzen_arr = jnp.asarray(sfluxzen, dtype=jnp.float32)[:, :RRTMG_SW_NGPT]
    taug_arr = jnp.asarray(taug, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]
    taur_arr = jnp.asarray(taur, dtype=jnp.float32)[:, :n, :RRTMG_SW_NGPT]

    band_for_gpoint = np.asarray(
        tuple(band for band, count in enumerate(ngc) for _ in range(count)),
        dtype=np.int32,
    )
    active_band = np.asarray(
        [(int(istart) - 16) <= band <= (int(iend) - 16) for band in band_for_gpoint],
        dtype=bool,
    )
    band_for_gpoint_arr = jnp.asarray(band_for_gpoint, dtype=jnp.int32)
    active = jnp.asarray(active_band, dtype=jnp.float32)

    adjflux_arr = jnp.asarray(adjflux, dtype=jnp.float32)
    if adjflux_arr.shape[1] >= 29:
        adjflux_by_band = adjflux_arr[:, 15:29]
    else:
        adjflux_by_band = adjflux_arr[:, :RRTMG_SW_NBND]
    adjflux_g = jnp.take(adjflux_by_band, band_for_gpoint_arr, axis=1)
    palbd_g = jnp.take(palbd_arr, band_for_gpoint_arr, axis=1)
    palbp_g = jnp.take(palbp_arr, band_for_gpoint_arr, axis=1)

    cloud = pcldfmc_arr[:, ::-1, :]
    clear = 1.0 - cloud
    ptaucmc_rev = ptaucmc_arr[:, ::-1, :]
    pasycmc_rev = pasycmc_arr[:, ::-1, :]
    pomgcmc_rev = pomgcmc_arr[:, ::-1, :]
    ptaormc_rev = ptaormc_arr[:, ::-1, :]
    taug_rev = taug_arr[:, ::-1, :]
    taur_rev = taur_arr[:, ::-1, :]
    if has_aerosol:
        ptaua_rev = jnp.take(ptaua_arr[:, ::-1, :], band_for_gpoint_arr, axis=2)
        pasya_rev = jnp.take(pasya_arr[:, ::-1, :], band_for_gpoint_arr, axis=2)
        pomga_rev = jnp.take(pomga_arr[:, ::-1, :], band_for_gpoint_arr, axis=2)
    else:
        ptaua_rev = jnp.asarray(0.0, dtype=jnp.float32)
        pasya_rev = jnp.asarray(0.0, dtype=jnp.float32)
        pomga_rev = jnp.asarray(1.0, dtype=jnp.float32)

    ztau_clear_raw = taur_rev + taug_rev + ptaua_rev
    zom_clear_raw = taur_rev + ptaua_rev * pomga_rev
    zg_clear_raw = pasya_rev * pomga_rev * ptaua_rev / zom_clear_raw
    zom_clear_raw = zom_clear_raw / ztau_clear_raw

    mu = prmu0_arr[:, None, None]
    zdbtc_nodel_layer = _sw_exp_lookup(
        _wrf_real_div(ztau_clear_raw, mu, rounding_token),
        tab,
        rounding_token=rounding_token,
    )
    ones_g = jnp.ones((ztau_clear_raw.shape[0], 1, RRTMG_SW_NGPT), dtype=jnp.float32)
    ztdbtc_nodel = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbtc_nodel_layer, axis=1, rounding_token=rounding_token
            ),
        ],
        axis=1,
    )
    if clear_only:
        ztdbt_nodel = ztdbtc_nodel
    else:
        tauorig = ztau_clear_raw + ptaormc_rev
        zdbtmo_nodel = _sw_exp_lookup(
            _wrf_real_div(tauorig, mu, rounding_token),
            tab,
            rounding_token=rounding_token,
        )
        zdbt_nodel_layer = clear * zdbtc_nodel_layer + cloud * zdbtmo_nodel
        ztdbt_nodel = jnp.concatenate(
            [
                ones_g,
                _wrf_ordered_cumprod(
                    zdbt_nodel_layer, axis=1, rounding_token=rounding_token
                ),
            ],
            axis=1,
        )

    zf = zg_clear_raw * zg_clear_raw
    zwf = zom_clear_raw * zf
    ztauc = (1.0 - zwf) * ztau_clear_raw
    zomcc = (zom_clear_raw - zwf) / (1.0 - zwf)
    zgcc = (zg_clear_raw - zf) / (1.0 - zf)

    clear_ref = reftra_sw_columns(
        nlayers=n,
        lrtchk=jnp.ones((ztauc.shape[0], n, RRTMG_SW_NGPT), dtype=bool),
        pgg=zgcc,
        prmuz=prmu0_arr,
        ptau=ztauc,
        pw=zomcc,
        tables=tab,
        rounding_token=rounding_token,
    )

    def boundary(layer_values: Any, surface_value: Any) -> Any:
        return jnp.concatenate(
            (jnp.asarray(layer_values, dtype=jnp.float32)[:, :n, :], jnp.asarray(surface_value, dtype=jnp.float32)[:, None, :]),
            axis=1,
        )

    zrefc = boundary(clear_ref.pref, palbp_g)
    zrefdc = boundary(clear_ref.prefd, palbd_g)
    ztrac = boundary(clear_ref.ptra, jnp.zeros_like(palbp_g))
    ztradc = boundary(clear_ref.ptrad, jnp.zeros_like(palbp_g))

    zdbtc_layer = _sw_exp_lookup(
        _wrf_real_div(ztauc, mu, rounding_token),
        tab,
        rounding_token=rounding_token,
    )
    zdbtc = boundary(zdbtc_layer, jnp.zeros_like(palbp_g))
    ztdbtc = jnp.concatenate(
        [
            ones_g,
            _wrf_ordered_cumprod(
                zdbtc_layer, axis=1, rounding_token=rounding_token
            ),
        ],
        axis=1,
    )
    if clear_only:
        zref = zrefc
        zrefd = zrefdc
        ztra = ztrac
        ztrad = ztradc
        zdbt = zdbtc
        ztdbt = ztdbtc
    else:
        if int(icpr) >= 1:
            ztauo = ztauc + ptaucmc_rev
            zomco = ztauc * zomcc + ptaucmc_rev * pomgcmc_rev
            zgco = (ptaucmc_rev * pomgcmc_rev * pasycmc_rev + ztauc * zomcc * zgcc) / zomco
            zomco = zomco / ztauo
        else:
            ztauo = taur_rev + taug_rev + ptaua_rev + ptaucmc_rev
            zomco = ptaua_rev * pomga_rev + ptaucmc_rev * pomgcmc_rev + taur_rev
            zgco = (ptaucmc_rev * pomgcmc_rev * pasycmc_rev + ptaua_rev * pomga_rev * pasya_rev) / zomco
            zomco = zomco / ztauo
            zf_o = zgco * zgco
            zwf_o = zomco * zf_o
            ztauo = (1.0 - zwf_o) * ztauo
            zomco = (zomco - zwf_o) / (1.0 - zwf_o)
            zgco = (zgco - zf_o) / (1.0 - zf_o)
        cloudy_ref = reftra_sw_columns(
            nlayers=n,
            lrtchk=cloud > repclc,
            pgg=zgco,
            prmuz=prmu0_arr,
            ptau=ztauo,
            pw=zomco,
            tables=tab,
            rounding_token=rounding_token,
        )
        zrefo = boundary(cloudy_ref.pref, palbp_g)
        zrefdo = boundary(cloudy_ref.prefd, palbd_g)
        ztrao = boundary(cloudy_ref.ptra, jnp.zeros_like(palbp_g))
        ztrado = boundary(cloudy_ref.ptrad, jnp.zeros_like(palbp_g))
        zref = boundary(clear * zrefc[:, :n, :] + cloud * zrefo[:, :n, :], palbp_g)
        zrefd = boundary(clear * zrefdc[:, :n, :] + cloud * zrefdo[:, :n, :], palbd_g)
        ztra = boundary(clear * ztrac[:, :n, :] + cloud * ztrao[:, :n, :], jnp.zeros_like(palbp_g))
        ztrad = boundary(clear * ztradc[:, :n, :] + cloud * ztrado[:, :n, :], jnp.zeros_like(palbp_g))
        zdbtmo = _sw_exp_lookup(
            _wrf_real_div(ztauo, mu, rounding_token),
            tab,
            rounding_token=rounding_token,
        )
        zdbt_layer = clear * zdbtc_layer + cloud * zdbtmo
        zdbt = boundary(zdbt_layer, jnp.zeros_like(palbp_g))
        ztdbt = jnp.concatenate(
            [
                ones_g,
                _wrf_ordered_cumprod(
                    zdbt_layer, axis=1, rounding_token=rounding_token
                ),
            ],
            axis=1,
        )

    zeros_flux = jnp.zeros((ztauc.shape[0], n + 1, RRTMG_SW_NGPT), dtype=jnp.float32)
    clear_flux = vrtqdr_sw_columns(
        klev=n,
        pref=zrefc,
        prefd=zrefdc,
        ptra=ztrac,
        ptrad=ztradc,
        pdbt=zdbtc,
        prdnd=zeros_flux,
        prup=zeros_flux,
        prupd=zeros_flux,
        ptdbt=ztdbtc,
        rounding_token=rounding_token,
    )
    if clear_only:
        total_flux = clear_flux
    else:
        total_flux = vrtqdr_sw_columns(
            klev=n,
            pref=zref,
            prefd=zrefd,
            ptra=ztra,
            ptrad=ztrad,
            pdbt=zdbt,
            prdnd=zeros_flux,
            prup=zeros_flux,
            prupd=zeros_flux,
            ptdbt=ztdbt,
            rounding_token=rounding_token,
        )

    zincflx = adjflux_g * sfluxzen_arr * prmu0_arr[:, None] * active[None, :]
    broadband = zincflx[:, None, :]
    ibm_g = band_for_gpoint_arr + 1
    visible = ((ibm_g >= 10) & (ibm_g <= 13)).astype(jnp.float32) * active
    near_ir = ((ibm_g == 14) | (ibm_g <= 9)).astype(jnp.float32) * active
    visible_w = zincflx[:, None, :] * visible[None, None, :]
    near_ir_w = zincflx[:, None, :] * near_ir[None, None, :]

    pbbfu = _wrf_reversed_flux_sum(total_flux.pfu, broadband)
    pbbfd = _wrf_reversed_flux_sum(total_flux.pfd, broadband)
    pbbcu = _wrf_reversed_flux_sum(clear_flux.pfu, broadband)
    pbbcd = _wrf_reversed_flux_sum(clear_flux.pfd, broadband)
    pbbfddir = _wrf_reversed_flux_sum(ztdbt_nodel, broadband)
    pbbcddir = _wrf_reversed_flux_sum(ztdbtc_nodel, broadband)

    puvcd = _wrf_reversed_flux_sum(clear_flux.pfd, visible_w)
    puvfd = _wrf_reversed_flux_sum(total_flux.pfd, visible_w)
    puvcddir = _wrf_reversed_flux_sum(ztdbtc_nodel, visible_w)
    puvfddir = _wrf_reversed_flux_sum(ztdbt_nodel, visible_w)
    pnicd = _wrf_reversed_flux_sum(clear_flux.pfd, near_ir_w)
    pnifd = _wrf_reversed_flux_sum(total_flux.pfd, near_ir_w)
    pnicddir = _wrf_reversed_flux_sum(ztdbtc_nodel, near_ir_w)
    pnifddir = _wrf_reversed_flux_sum(ztdbt_nodel, near_ir_w)

    return RrtmgSwSpcvmcResult(
        pbbfd=pbbfd,
        pbbfu=pbbfu,
        pbbcd=pbbcd,
        pbbcu=pbbcu,
        puvfd=puvfd,
        puvcd=puvcd,
        pnifd=pnifd,
        pnicd=pnicd,
        pbbfddir=pbbfddir,
        pbbcddir=pbbcddir,
        puvfddir=puvfddir,
        puvcddir=puvcddir,
        pnifddir=pnifddir,
        pnicddir=pnicddir,
    )


def _fortran_int(value: Any) -> Any:
    return jnp.asarray(value).astype(jnp.int32)


def setcoef_sw(
    *,
    nlayers: int,
    pavel: Any,
    tavel: Any,
    pz: Any,
    tz: Any,
    tbound: Any,
    coldry: Any,
    wkl: Any,
    reference: RrtmgSwReferenceAtmosphere | None = None,
    rounding_token: Any | None = None,
) -> RrtmgSwSetcoefResult:
    """WRF ``setcoef_sw`` over one column or leading column batches."""

    del pz, tz, tbound
    nlay = int(nlayers)
    ref = swatmref() if reference is None else reference
    preflog = jnp.asarray(ref.preflog, dtype=jnp.float32)
    tref = jnp.asarray(ref.tref, dtype=jnp.float32)
    pavel_arr = jnp.asarray(pavel, dtype=jnp.float32)[..., :nlay]
    tavel_arr = jnp.asarray(tavel, dtype=jnp.float32)[..., :nlay]
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[..., :nlay]
    wkl_arr = jnp.asarray(wkl, dtype=jnp.float32)[..., :, :nlay]

    stpfac = _wrf_real_div(
        jnp.float32(296.0),
        jnp.float32(1013.0),
        rounding_token,
    )
    plog = wrf_glibc_logf_positive_normal(pavel_arr)
    jp_coordinate = _wrf_real_sub(
        jnp.float32(36.0),
        _wrf_real_mul(
            jnp.float32(5.0),
            _wrf_real_add(plog, jnp.float32(0.04), rounding_token),
            rounding_token,
        ),
        rounding_token,
    )
    jp = jnp.clip(_fortran_int(jp_coordinate), 1, 58)
    jp_idx = jp - 1
    jp1_idx = jp
    fp = _wrf_real_mul(
        jnp.float32(5.0),
        _wrf_real_sub(preflog[jp_idx], plog, rounding_token),
        rounding_token,
    )

    temperature_offset = _wrf_real_div(
        _wrf_real_sub(tavel_arr, tref[jp_idx], rounding_token),
        jnp.float32(15.0),
        rounding_token,
    )
    jt = jnp.clip(
        _fortran_int(_wrf_real_add(jnp.float32(3.0), temperature_offset, rounding_token)),
        1,
        4,
    )
    ft = _wrf_real_sub(temperature_offset, (jt - 3).astype(jnp.float32), rounding_token)
    temperature_offset_1 = _wrf_real_div(
        _wrf_real_sub(tavel_arr, tref[jp1_idx], rounding_token),
        jnp.float32(15.0),
        rounding_token,
    )
    jt1 = jnp.clip(
        _fortran_int(_wrf_real_add(jnp.float32(3.0), temperature_offset_1, rounding_token)),
        1,
        4,
    )
    ft1 = _wrf_real_sub(temperature_offset_1, (jt1 - 3).astype(jnp.float32), rounding_token)

    water = _wrf_real_div(wkl_arr[..., 0, :], coldry_arr, rounding_token)
    scalefac = _wrf_real_div(
        _wrf_real_mul(pavel_arr, stpfac, rounding_token),
        tavel_arr,
        rounding_token,
    )
    troposphere = plog > 4.56
    laytrop = jnp.sum(troposphere.astype(jnp.int32), axis=-1)
    laylow = jnp.sum((plog >= 6.62).astype(jnp.int32), axis=-1)
    layswtch = jnp.zeros(pavel_arr.shape[:-1], dtype=jnp.int32)

    forfac = _wrf_real_div(
        scalefac,
        _wrf_real_add(jnp.float32(1.0), water, rounding_token),
        rounding_token,
    )
    factor_trop_for = _wrf_real_div(
        _wrf_real_sub(jnp.float32(332.0), tavel_arr, rounding_token),
        jnp.float32(36.0),
        rounding_token,
    )
    indfor_trop = jnp.clip(_fortran_int(factor_trop_for), 1, 2)
    forfrac_trop = _wrf_real_sub(
        factor_trop_for,
        indfor_trop.astype(jnp.float32),
        rounding_token,
    )
    factor_strat_for = _wrf_real_div(
        _wrf_real_sub(tavel_arr, jnp.float32(188.0), rounding_token),
        jnp.float32(36.0),
        rounding_token,
    )
    indfor_strat = jnp.full_like(jp, 3, dtype=jnp.int32)
    forfrac_strat = _wrf_real_sub(factor_strat_for, jnp.float32(1.0), rounding_token)
    indfor = jnp.where(troposphere, indfor_trop, indfor_strat)
    forfrac = jnp.where(troposphere, forfrac_trop, forfrac_strat)

    selffac_trop = _wrf_real_mul(water, forfac, rounding_token)
    factor_self = _wrf_real_div(
        _wrf_real_sub(tavel_arr, jnp.float32(188.0), rounding_token),
        jnp.float32(7.2),
        rounding_token,
    )
    indself_trop = jnp.clip(_fortran_int(factor_self) - 7, 1, 9)
    selffrac_trop = _wrf_real_sub(
        factor_self,
        (indself_trop + 7).astype(jnp.float32),
        rounding_token,
    )
    indself = jnp.where(troposphere, indself_trop, 0)
    selffac = jnp.where(troposphere, selffac_trop, 0.0)
    selffrac = jnp.where(troposphere, selffrac_trop, 0.0)

    column_scale = jnp.float32(1.0e-20)
    colh2o = _wrf_real_mul(column_scale, wkl_arr[..., 0, :], rounding_token)
    colco2 = _wrf_real_mul(column_scale, wkl_arr[..., 1, :], rounding_token)
    colo3 = _wrf_real_mul(column_scale, wkl_arr[..., 2, :], rounding_token)
    coln2o = _wrf_real_mul(column_scale, wkl_arr[..., 3, :], rounding_token)
    colch4 = _wrf_real_mul(column_scale, wkl_arr[..., 5, :], rounding_token)
    colo2 = _wrf_real_mul(column_scale, wkl_arr[..., 6, :], rounding_token)
    colmol = _wrf_real_add(
        _wrf_real_mul(column_scale, coldry_arr, rounding_token),
        colh2o,
        rounding_token,
    )
    fallback = _wrf_real_mul(jnp.float32(1.0e-32), coldry_arr, rounding_token)
    colco2 = jnp.where(colco2 == 0.0, fallback, colco2)
    coln2o = jnp.where(coln2o == 0.0, fallback, coln2o)
    colch4 = jnp.where(colch4 == 0.0, fallback, colch4)
    colo2 = jnp.where(colo2 == 0.0, fallback, colo2)

    co2reg = _wrf_real_mul(jnp.float32(3.55e-24), coldry_arr, rounding_token)
    co2_delta = _wrf_real_sub(colco2, co2reg, rounding_token)
    co2_exponent = _wrf_real_div(
        jnp.float32(-1919.4),
        tavel_arr,
        rounding_token,
    )
    co2_numerator = _wrf_real_mul(
        _wrf_real_mul(co2_delta, jnp.float32(272.63), rounding_token),
        wrf_glibc_expf_finite(co2_exponent),
        rounding_token,
    )
    co2_denominator = _wrf_real_mul(
        jnp.float32(8.7604e-4),
        tavel_arr,
        rounding_token,
    )
    co2mult = _wrf_real_div(
        co2_numerator,
        co2_denominator,
        rounding_token,
    )

    compfp = _wrf_real_sub(jnp.float32(1.0), fp, rounding_token)
    fac10 = _wrf_real_mul(compfp, ft, rounding_token)
    fac00 = _wrf_real_mul(
        compfp,
        _wrf_real_sub(jnp.float32(1.0), ft, rounding_token),
        rounding_token,
    )
    fac11 = _wrf_real_mul(fp, ft1, rounding_token)
    fac01 = _wrf_real_mul(
        fp,
        _wrf_real_sub(jnp.float32(1.0), ft1, rounding_token),
        rounding_token,
    )

    return RrtmgSwSetcoefResult(
        laytrop=laytrop,
        layswtch=layswtch,
        laylow=laylow,
        jp=jp,
        jt=jt,
        jt1=jt1,
        co2mult=co2mult,
        colch4=colch4,
        colco2=colco2,
        colh2o=colh2o,
        colmol=colmol,
        coln2o=coln2o,
        colo2=colo2,
        colo3=colo3,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        selffac=selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=forfac,
        forfrac=forfrac,
        indfor=indfor,
    )


__all__ = [
    "RrtmgSwReferenceAtmosphere",
    "RrtmgSwBandData",
    "RrtmgSwCldprmcResult",
    "RrtmgSwCloudTables",
    "RrtmgSwColumnResult",
    "RrtmgSwCompressedBandData",
    "RrtmgSwCompressedLookupTables",
    "RrtmgSwInatmResult",
    "RrtmgSwLookupTables",
    "RrtmgSwReftraResult",
    "RrtmgSwSetcoefResult",
    "RrtmgSwSpcvmcResult",
    "RrtmgSwTaumolBandResult",
    "RrtmgSwVrtqdrResult",
    "RrtmgSwWrfColumnResult",
    "combine_rrtmg_sw_lookup_tables",
    "cldprmc_sw",
    "earth_sun",
    "inatm_sw_column",
    "load_rrtmg_sw_cloud_tables",
    "load_rrtmg_sw_compressed_lookup_tables",
    "load_rrtmg_sw_data",
    "reftra_sw",
    "rrtmg_sw_column",
    "rrtmg_sw_columns",
    "rrtmg_swrad_wrf_columns",
    "setcoef_sw",
    "spcvmc_sw",
    "spcvmc_sw_from_optical_depths",
    "swatmref",
    "taumol16",
    "taumol17",
    "taumol18",
    "taumol19",
    "taumol20",
    "taumol21",
    "taumol22",
    "taumol23",
    "taumol24",
    "taumol25",
    "taumol26",
    "taumol27",
    "taumol28",
    "taumol29",
    "taumol_sw",
    "vrtqdr_sw",
]
