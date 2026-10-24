"""Longwave RRTMG translations from WRF ``module_ra_rrtmg_lw``."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from wrf_jax.core.column_scan import scan_columns
from wrf_jax.core.real_arithmetic import column_binary, real_assignment, real_reduced_assignment
from wrf_jax.core.wrf_math import (
    wrf_glibc_expf_finite,
    wrf_glibc_logf_positive_normal,
)
from wrf_jax.physics.module_ra_rrtmg_common import (
    LW_RWGT,
    NGB_LW,
    RRTMG_LW_NBND,
    RRTMG_LW_NGPT,
    RrtmgLwPadeTables,
    RrtmgSpectralInit,
    _LW_NGC_VALUES,
    _LW_NGN_VALUES,
    _LW_NGS_VALUES,
    inirad,
    lwdatinit,
    mcica_subcol_lw_overlap2,
    pade_optical_depth,
    pade_source_factor,
    pade_transmittance,
    prepare_cloud_path_inputs,
    prepare_cloud_radii,
    rrtmg_lw_nlayers,
    rrtmg_lw_pade_tables,
    wrf_real_pa_to_hpa as _wrf_real_pa_to_hpa,
)
from wrf_jax.physics.module_ra_clwrf_support import read_camgases
from wrf_jax.runtime_config import configured_path


_PROFILE_RRTMG_LW_PATH = os.environ.get("WRF_PY_PROFILE_RRTMG_LW", "").strip()


def _wrf_real_store(value: Any, rounding_token: Any | None = None) -> Any:
    """Materialize one WRF default-REAL assignment inside an XLA graph."""

    array = real_reduced_assignment(jnp.asarray(value, dtype=jnp.float64))
    if rounding_token is not None:
        return real_assignment(array, rounding_token)
    return array


def _wrf_real_add(
    left: Any,
    right: Any,
    rounding_token: Any | None = None,
) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64)
        + jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_sub(
    left: Any,
    right: Any,
    rounding_token: Any | None = None,
) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64)
        - jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_mul(
    left: Any,
    right: Any,
    rounding_token: Any | None = None,
) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64)
        * jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_div(
    left: Any,
    right: Any,
    rounding_token: Any | None = None,
) -> Any:
    return _wrf_real_store(
        jnp.asarray(left, dtype=jnp.float64)
        / jnp.asarray(right, dtype=jnp.float64),
        rounding_token,
    )


def _wrf_real_interp(lower: Any, upper: Any, fraction: Any) -> Any:
    return _wrf_real_add(
        lower,
        _wrf_real_mul(fraction, _wrf_real_sub(upper, lower)),
    )


def _wrf_real_species_interpolation(
    primary: Any,
    ratio: Any,
    secondary: Any,
    scale: float,
    one_minus: Any,
) -> tuple[Any, Any, Any, Any]:
    speccomb = _wrf_real_add(primary, _wrf_real_mul(ratio, secondary))
    specparm = _wrf_real_store(
        jnp.minimum(_wrf_real_div(primary, speccomb), one_minus)
    )
    specmult = _wrf_real_mul(jnp.float32(scale), specparm)
    index = jnp.int32(1) + specmult.astype(jnp.int32)
    fraction = _wrf_real_sub(specmult, jnp.floor(specmult))
    return speccomb, specparm, index, fraction


def _wrf_real_pow(base: Any, exponent: float) -> Any:
    return _wrf_real_store(
        jnp.power(
            jnp.asarray(base, dtype=jnp.float32),
            jnp.float32(exponent),
        )
    )


def _wrf_real_active_pow(base: Any, exponent: float, active: Any) -> Any:
    """Keep unused gas corrections away from singular fractional powers."""
    base = jnp.asarray(base, jnp.float32)
    return _wrf_real_pow(jnp.where(active, base, jnp.ones_like(base)), exponent)


def _wrf_real_linear_factors(
    fraction: Any,
    first: Any,
    second: Any,
) -> tuple[Any, Any, Any, Any]:
    complement = _wrf_real_sub(jnp.float32(1.0), fraction)
    return (
        _wrf_real_mul(complement, first),
        _wrf_real_mul(fraction, first),
        _wrf_real_mul(complement, second),
        _wrf_real_mul(fraction, second),
    )


def _wrf_real_linear_combination(coefficients: tuple[Any, ...], values: tuple[Any, ...]) -> Any:
    if len(coefficients) != len(values) or not coefficients:
        raise ValueError("WRF linear combination requires equally sized non-empty inputs")
    total = _wrf_real_mul(coefficients[0], values[0])
    for coefficient, value in zip(coefficients[1:], values[1:], strict=True):
        total = _wrf_real_add(total, _wrf_real_mul(coefficient, value))
    return total


def _wrf_lw_broadband_flux(radiance: Any, delwave: Any, fluxfac: Any) -> Any:
    """Reproduce WRF ``rtrnmc``'s g-point and band accumulation order."""

    values = jnp.asarray(radiance, dtype=jnp.float32)
    weights = jnp.asarray(delwave, dtype=jnp.float32)
    ends = np.asarray(_LW_NGS_VALUES)
    values = values[..., :int(ends[-1])]
    band_indices = np.searchsorted(ends, np.arange(ends[-1]), side="right")
    end_flags = np.isin(np.arange(ends[-1]) + 1, ends)
    zero = jnp.zeros(values.shape[:-1], dtype=jnp.float32)

    def accumulate(carry: Any, inputs: Any, _parameter: Any) -> Any:
        band, total = carry
        term, end_of_band, weight = inputs
        band = _wrf_real_add(band, term)
        band_flux = _wrf_real_store(column_binary("mul", band, jnp.float32(0.5)))
        weighted = _wrf_real_store(column_binary("mul", band_flux, weight))
        updated = _wrf_real_store(column_binary("add", total, weighted))
        return (
            jnp.where(end_of_band, jnp.zeros_like(band), band),
            jnp.where(end_of_band, updated, total),
        ), None

    # Keep both ordered sums local; no per-g-point history is required.
    (_, total), _ = scan_columns(
        accumulate,
        (zero, zero),
        (jnp.moveaxis(values, -1, 0), jnp.asarray(end_flags), weights[band_indices]),
    )
    return _wrf_real_store(total * jnp.asarray(fluxfac, dtype=jnp.float32))


def _wrf_ordered_difference(minuend: Any, subtrahend: Any) -> Any:
    """Materialize two WRF REAL inputs before one subtraction under XLA."""

    def subtract_one(total: Any, term: Any) -> tuple[Any, None]:
        return _wrf_real_store(total - term), None

    terms = jnp.stack((jnp.asarray(subtrahend, dtype=jnp.float32), jnp.zeros_like(subtrahend)), axis=0)
    difference, _ = jax.lax.scan(subtract_one, jnp.asarray(minuend, dtype=jnp.float32), terms)
    return difference


def _wrf_ordered_divisions(
    dividend: Any,
    *divisors: Any,
    rounding_token: Any | None = None,
) -> Any:
    """Apply WRF REAL divisions in source order without XLA reassociation."""

    dividend_real = jnp.asarray(dividend, dtype=jnp.float32)
    divisor_arrays = [
        jnp.broadcast_to(
            jnp.asarray(value, dtype=jnp.float32),
            dividend_real.shape,
        )
        for value in divisors
    ]
    divisor_arrays.append(jnp.ones_like(dividend_real))

    def divide_one(value: Any, divisor: Any) -> tuple[Any, None]:
        return _wrf_real_store(value / divisor, rounding_token), None

    quotient, _ = jax.lax.scan(divide_one, dividend_real, jnp.stack(divisor_arrays, axis=0))
    return quotient


def _profile_rrtmg_lw_ready(value: Any) -> None:
    if not _PROFILE_RRTMG_LW_PATH:
        return
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _profile_rrtmg_lw_event(name: str) -> None:
    if not _PROFILE_RRTMG_LW_PATH:
        return
    with open(_PROFILE_RRTMG_LW_PATH, "a", encoding="ascii") as handle:
        handle.write(f"0.000000 {name}\n")


def _profile_rrtmg_lw_write(name: str, start: float, value: Any) -> None:
    if not _PROFILE_RRTMG_LW_PATH:
        return
    _profile_rrtmg_lw_ready(value)
    with open(_PROFILE_RRTMG_LW_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{perf_counter() - start:.6f} {name}\n")


class RrtmgLwCloudTables(NamedTuple):
    """WRF longwave cloud absorption coefficient tables."""

    abscld1: Any
    absice0: Any
    absice1: Any
    absice2: Any
    absice3: Any
    absliq0: Any
    absliq1: Any


class RrtmgLwSetcoefTables(NamedTuple):
    """WRF longwave reference-atmosphere and Planck tables."""

    pref: Any
    preflog: Any
    tref: Any
    chi_mls: Any
    totplnk: Any
    totplk16: Any


class RrtmgLwBandData(NamedTuple):
    """One raw band record read from WRF ``RRTMG_LW_DATA``."""

    band: int
    fracrefao: Any | None = None
    fracrefbo: Any | None = None
    kao: Any | None = None
    kbo: Any | None = None
    kao_mn2: Any | None = None
    kbo_mn2: Any | None = None
    kao_mn2o: Any | None = None
    kbo_mn2o: Any | None = None
    kao_mo3: Any | None = None
    ccl4o: Any | None = None
    kao_mco2: Any | None = None
    kbo_mco2: Any | None = None
    cfc11adjo: Any | None = None
    cfc12o: Any | None = None
    cfc22adjo: Any | None = None
    kao_mo2: Any | None = None
    kbo_mo2: Any | None = None
    kao_mco: Any | None = None
    kbo_mo3: Any | None = None
    selfrefo: Any | None = None
    forrefo: Any | None = None


class RrtmgLwLookupTables(NamedTuple):
    """Raw WRF longwave RRTMG lookup tables keyed by band number."""

    band1: RrtmgLwBandData
    band2: RrtmgLwBandData
    band3: RrtmgLwBandData
    band4: RrtmgLwBandData
    band5: RrtmgLwBandData
    band6: RrtmgLwBandData
    band7: RrtmgLwBandData
    band8: RrtmgLwBandData
    band9: RrtmgLwBandData
    band10: RrtmgLwBandData
    band11: RrtmgLwBandData
    band12: RrtmgLwBandData
    band13: RrtmgLwBandData
    band14: RrtmgLwBandData
    band15: RrtmgLwBandData
    band16: RrtmgLwBandData


class RrtmgLwCompressedBandData(NamedTuple):
    """One longwave RRTMG band after WRF ``cmbgb*`` g-point reduction."""

    band: int
    fracrefa: Any | None = None
    fracrefb: Any | None = None
    ka: Any | None = None
    kb: Any | None = None
    ka_mn2: Any | None = None
    kb_mn2: Any | None = None
    ka_mn2o: Any | None = None
    kb_mn2o: Any | None = None
    ka_mo3: Any | None = None
    ccl4: Any | None = None
    ka_mco2: Any | None = None
    kb_mco2: Any | None = None
    cfc11adj: Any | None = None
    cfc12: Any | None = None
    cfc22adj: Any | None = None
    ka_mo2: Any | None = None
    kb_mo2: Any | None = None
    ka_mco: Any | None = None
    kb_mo3: Any | None = None
    selfref: Any | None = None
    forref: Any | None = None


class RrtmgLwCompressedLookupTables(NamedTuple):
    """Compressed longwave RRTMG lookup tables used by ``taumol_lw``."""

    band1: RrtmgLwCompressedBandData
    band2: RrtmgLwCompressedBandData
    band3: RrtmgLwCompressedBandData
    band4: RrtmgLwCompressedBandData
    band5: RrtmgLwCompressedBandData
    band6: RrtmgLwCompressedBandData
    band7: RrtmgLwCompressedBandData
    band8: RrtmgLwCompressedBandData
    band9: RrtmgLwCompressedBandData
    band10: RrtmgLwCompressedBandData
    band11: RrtmgLwCompressedBandData
    band12: RrtmgLwCompressedBandData
    band13: RrtmgLwCompressedBandData
    band14: RrtmgLwCompressedBandData
    band15: RrtmgLwCompressedBandData
    band16: RrtmgLwCompressedBandData


class RrtmgLwCldprmcResult(NamedTuple):
    """Cloud optical-depth outputs from WRF ``cldprmc``."""

    ncbands: int
    taucmc: Any


class RrtmgLwInatmResult(NamedTuple):
    """Column inputs prepared by WRF longwave ``inatm``."""

    nlayers: int
    pavel: Any
    pz: Any
    tavel: Any
    tz: Any
    tbound: Any
    semiss: Any
    coldry: Any
    wbrodl: Any
    wkl: Any
    wx: Any
    pwvcm: Any
    inflag: int
    iceflag: int
    liqflag: int
    cldfmc: Any
    taucmc: Any
    ciwpmc: Any
    clwpmc: Any
    cswpmc: Any
    reicmc: Any
    relqmc: Any
    resnmc: Any
    taua: Any


class RrtmgLwSetcoefResult(NamedTuple):
    """Outputs from WRF longwave ``setcoef`` for one atmospheric column."""

    laytrop: Any
    jp: Any
    jt: Any
    jt1: Any
    planklay: Any
    planklev: Any
    plankbnd: Any
    colh2o: Any
    colco2: Any
    colo3: Any
    coln2o: Any
    colco: Any
    colch4: Any
    colo2: Any
    colbrd: Any
    fac00: Any
    fac01: Any
    fac10: Any
    fac11: Any
    rat_h2oco2: Any
    rat_h2oco2_1: Any
    rat_h2oo3: Any
    rat_h2oo3_1: Any
    rat_h2on2o: Any
    rat_h2on2o_1: Any
    rat_h2och4: Any
    rat_h2och4_1: Any
    rat_n2oco2: Any
    rat_n2oco2_1: Any
    rat_o3co2: Any
    rat_o3co2_1: Any
    selffac: Any
    selffrac: Any
    indself: Any
    forfac: Any
    forfrac: Any
    indfor: Any
    minorfrac: Any
    scaleminor: Any
    scaleminorn2: Any
    indminor: Any


class RrtmgLwTaumolBandResult(NamedTuple):
    """Longwave optical depth and Planck fraction output for one band."""

    fracs: Any
    taug: Any


class RrtmgLwRtrnmcResult(NamedTuple):
    """Longwave radiative-transfer fluxes and heating rates from ``rtrnmc``."""

    totuflux: Any
    totdflux: Any
    fnet: Any
    htr: Any
    totuclfl: Any
    totdclfl: Any
    fnetc: Any
    htrc: Any


class RrtmgLwDriverResult(NamedTuple):
    """Column outputs from WRF longwave ``rrtmg_lw``."""

    uflx: Any
    dflx: Any
    hr: Any
    uflxc: Any
    dflxc: Any
    hrc: Any
    uflxcln: Any
    dflxcln: Any
    fracs: Any
    taug: Any
    taut: Any
    taucmc: Any


class RrtmgLwWrfColumnResult(NamedTuple):
    """WRF ``RRTMG_LWRAD`` outputs mapped back to model grid fields."""

    rthratenlw: Any
    rthratenlwc: Any
    lwupt: Any
    lwuptc: Any
    lwuptcln: Any
    lwdnt: Any
    lwdntc: Any
    lwdntcln: Any
    lwupb: Any
    lwupbc: Any
    lwupbcln: Any
    lwdnb: Any
    lwdnbc: Any
    lwdnbcln: Any
    glw: Any
    olr: Any
    lwcf: Any


@lru_cache(maxsize=1)
def load_rrtmg_lw_cloud_tables(path: str | Path | None = None) -> RrtmgLwCloudTables:
    """Load WRF ``rrtmg_lw_init:lwcldpr`` cloud absorption tables."""

    table_path = Path(path) if path is not None else Path(__file__).with_name("data") / "rrtmg_lw_cloud_tables.npz"
    with np.load(table_path) as data:
        return RrtmgLwCloudTables(
            abscld1=jnp.asarray(data["abscld1"], dtype=jnp.float32),
            absice0=jnp.asarray(data["absice0"], dtype=jnp.float32),
            absice1=jnp.asarray(data["absice1"], dtype=jnp.float32),
            absice2=jnp.asarray(data["absice2"], dtype=jnp.float32),
            absice3=jnp.asarray(data["absice3"], dtype=jnp.float32),
            absliq0=jnp.asarray(data["absliq0"], dtype=jnp.float32),
            absliq1=jnp.asarray(data["absliq1"], dtype=jnp.float32),
        )


@lru_cache(maxsize=1)
def load_rrtmg_lw_setcoef_tables(path: str | Path | None = None) -> RrtmgLwSetcoefTables:
    """Load WRF ``lwatmref`` and ``lwavplank`` tables used by ``setcoef``."""

    table_path = Path(path) if path is not None else Path(__file__).with_name("data") / "rrtmg_lw_setcoef_tables.npz"
    with np.load(table_path) as data:
        return RrtmgLwSetcoefTables(
            pref=jnp.asarray(data["pref"], dtype=jnp.float32),
            preflog=jnp.asarray(data["preflog"], dtype=jnp.float32),
            tref=jnp.asarray(data["tref"], dtype=jnp.float32),
            chi_mls=jnp.asarray(data["chi_mls"], dtype=jnp.float32),
            totplnk=jnp.asarray(data["totplnk"], dtype=jnp.float32),
            totplk16=jnp.asarray(data["totplk16"], dtype=jnp.float32),
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


def _parse_record_fields(payload: memoryview, fields: tuple[tuple[str, tuple[int, ...]], ...]) -> dict[str, Any]:
    offset = 0
    values: dict[str, Any] = {}
    dtype = np.dtype(">f4")
    for name, shape in fields:
        count = int(np.prod(shape))
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ValueError(f"RRTMG LW record too short while reading {name}")
        parsed = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        offset += nbytes
        values[name] = np.array(parsed.reshape(shape, order="F"), dtype=np.float32, copy=True)
    if offset != len(payload):
        raise ValueError(f"Unexpected extra bytes in RRTMG LW record: {len(payload) - offset}")
    return values


_LW_KGB_RECORDS: tuple[tuple[int, tuple[tuple[str, tuple[int, ...]], ...]], ...] = (
    (
        1,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("kao_mn2", (19, 16)),
            ("kbo_mn2", (19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        2,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        3,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16, 5)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 5, 47, 16)),
            ("kao_mn2o", (9, 19, 16)),
            ("kbo_mn2o", (5, 19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        4,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16, 5)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 5, 47, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        5,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16, 5)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 5, 47, 16)),
            ("kao_mo3", (9, 19, 16)),
            ("ccl4o", (16,)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        6,
        (
            ("fracrefao", (16,)),
            ("kao", (5, 13, 16)),
            ("kao_mco2", (19, 16)),
            ("cfc11adjo", (16,)),
            ("cfc12o", (16,)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        7,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16,)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("kao_mco2", (9, 19, 16)),
            ("kbo_mco2", (19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        8,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("kao_mco2", (19, 16)),
            ("kbo_mco2", (19, 16)),
            ("kao_mn2o", (19, 16)),
            ("kbo_mn2o", (19, 16)),
            ("kao_mo3", (19, 16)),
            ("cfc12o", (16,)),
            ("cfc22adjo", (16,)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        9,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16,)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("kao_mn2o", (9, 19, 16)),
            ("kbo_mn2o", (19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        10,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        11,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("kao_mo2", (19, 16)),
            ("kbo_mo2", (19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        12,
        (
            ("fracrefao", (16, 9)),
            ("kao", (9, 5, 13, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        13,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16,)),
            ("kao", (9, 5, 13, 16)),
            ("kao_mco2", (9, 19, 16)),
            ("kao_mco", (9, 19, 16)),
            ("kbo_mo3", (19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        14,
        (
            ("fracrefao", (16,)),
            ("fracrefbo", (16,)),
            ("kao", (5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        15,
        (
            ("fracrefao", (16, 9)),
            ("kao", (9, 5, 13, 16)),
            ("kao_mn2", (9, 19, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
    (
        16,
        (
            ("fracrefao", (16, 9)),
            ("fracrefbo", (16,)),
            ("kao", (9, 5, 13, 16)),
            ("kbo", (5, 47, 16)),
            ("selfrefo", (10, 16)),
            ("forrefo", (4, 16)),
        ),
    ),
)


@lru_cache(maxsize=4)
def load_rrtmg_lw_data(path: str | Path | None = None) -> RrtmgLwLookupTables:
    """Read WRF ``RRTMG_LW_DATA`` without using the Fortran bridge."""

    default_path = Path(__file__).with_name("data") / "rrtmg" / "RRTMG_LW_DATA"
    file_path = configured_path("rrtmg_lw_data", env="WRF_PY_RRTMG_LW_DATA", fallback=default_path) if path is None else Path(path)
    raw = file_path.read_bytes()
    offset = 0
    bands: list[RrtmgLwBandData] = []
    for band_number, fields in _LW_KGB_RECORDS:
        payload, offset = _read_record_payload(raw, offset, file_path)
        values = _parse_record_fields(payload, fields)
        bands.append(RrtmgLwBandData(band=band_number, **values))
    if offset != len(raw):
        raise ValueError(f"Unexpected extra bytes in {file_path}: {len(raw) - offset}")
    return RrtmgLwLookupTables(*bands)


def _rrtmg_lw_groups(spectral: RrtmgSpectralInit, band_index: int) -> list[tuple[slice, np.ndarray]]:
    try:
        ngc = np.asarray(spectral.ngc, dtype=np.int32)
        ngs = np.asarray(spectral.ngs, dtype=np.int32)
        ngn = np.asarray(spectral.ngn, dtype=np.int32)
        rwgt = np.asarray(spectral.rwgt, dtype=np.float32)
    except Exception:
        ngc = np.asarray(_LW_NGC_VALUES, dtype=np.int32)
        ngs = np.asarray(_LW_NGS_VALUES, dtype=np.int32)
        ngn = np.asarray(_LW_NGN_VALUES, dtype=np.int32)
        rwgt = np.asarray(LW_RWGT, dtype=np.float32)
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
        raise ValueError(f"RRTMG LW band {band_index + 1} does not cover 16 original g-points")
    return groups


def _combine_old_last(values: Any, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty(arr.shape[:-1] + (len(groups),), dtype=np.float32)
    for igc, (old_slice, weights) in enumerate(groups):
        chunk = arr[..., old_slice]
        terms = np.asarray(
            chunk * weights.reshape((1,) * (chunk.ndim - 1) + (weights.size,))
            if weighted
            else chunk,
            dtype=np.float32,
        )
        out[..., igc] = np.add.accumulate(
            terms,
            axis=-1,
            dtype=np.float32,
        )[..., -1]
    return out


def _combine_old_first(values: Any, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty((len(groups),) + arr.shape[1:], dtype=np.float32)
    for igc, (old_slice, weights) in enumerate(groups):
        chunk = arr[old_slice, ...]
        terms = np.asarray(
            chunk * weights.reshape((weights.size,) + (1,) * (chunk.ndim - 1))
            if weighted
            else chunk,
            dtype=np.float32,
        )
        out[igc, ...] = np.add.accumulate(
            terms,
            axis=0,
            dtype=np.float32,
        )[-1, ...]
    return out


def _combine_optional_old_last(values: Any | None, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray | None:
    if values is None:
        return None
    return _combine_old_last(values, groups, weighted=weighted)


def _combine_optional_old_first(values: Any | None, groups: list[tuple[slice, np.ndarray]], *, weighted: bool) -> np.ndarray | None:
    if values is None:
        return None
    return _combine_old_first(values, groups, weighted=weighted)


def _compress_rrtmg_lw_band(
    band: RrtmgLwBandData,
    *,
    spectral: RrtmgSpectralInit,
    band_index: int,
) -> RrtmgLwCompressedBandData:
    groups = _rrtmg_lw_groups(spectral, band_index)
    return RrtmgLwCompressedBandData(
        band=band.band,
        fracrefa=_combine_optional_old_first(band.fracrefao, groups, weighted=False),
        fracrefb=_combine_optional_old_first(band.fracrefbo, groups, weighted=False),
        ka=_combine_optional_old_last(band.kao, groups, weighted=True),
        kb=_combine_optional_old_last(band.kbo, groups, weighted=True),
        ka_mn2=_combine_optional_old_last(band.kao_mn2, groups, weighted=True),
        kb_mn2=_combine_optional_old_last(band.kbo_mn2, groups, weighted=True),
        ka_mn2o=_combine_optional_old_last(band.kao_mn2o, groups, weighted=True),
        kb_mn2o=_combine_optional_old_last(band.kbo_mn2o, groups, weighted=True),
        ka_mo3=_combine_optional_old_last(band.kao_mo3, groups, weighted=True),
        ccl4=_combine_optional_old_first(band.ccl4o, groups, weighted=True),
        ka_mco2=_combine_optional_old_last(band.kao_mco2, groups, weighted=True),
        kb_mco2=_combine_optional_old_last(band.kbo_mco2, groups, weighted=True),
        cfc11adj=_combine_optional_old_first(band.cfc11adjo, groups, weighted=True),
        cfc12=_combine_optional_old_first(band.cfc12o, groups, weighted=True),
        cfc22adj=_combine_optional_old_first(band.cfc22adjo, groups, weighted=True),
        ka_mo2=_combine_optional_old_last(band.kao_mo2, groups, weighted=True),
        kb_mo2=_combine_optional_old_last(band.kbo_mo2, groups, weighted=True),
        ka_mco=_combine_optional_old_last(band.kao_mco, groups, weighted=True),
        kb_mo3=_combine_optional_old_last(band.kbo_mo3, groups, weighted=True),
        selfref=_combine_optional_old_last(band.selfrefo, groups, weighted=True),
        forref=_combine_optional_old_last(band.forrefo, groups, weighted=True),
    )


def combine_rrtmg_lw_lookup_tables(
    raw: RrtmgLwLookupTables,
    *,
    spectral: RrtmgSpectralInit | None = None,
) -> RrtmgLwCompressedLookupTables:
    """Apply WRF ``cmbgb1`` through ``cmbgb16`` g-point reduction."""

    spec = lwdatinit() if spectral is None else spectral
    raw_bands = (
        raw.band1,
        raw.band2,
        raw.band3,
        raw.band4,
        raw.band5,
        raw.band6,
        raw.band7,
        raw.band8,
        raw.band9,
        raw.band10,
        raw.band11,
        raw.band12,
        raw.band13,
        raw.band14,
        raw.band15,
        raw.band16,
    )
    return RrtmgLwCompressedLookupTables(
        *(
            _compress_rrtmg_lw_band(band, spectral=spec, band_index=band_index)
            for band_index, band in enumerate(raw_bands)
        )
    )


@lru_cache(maxsize=4)
def load_rrtmg_lw_compressed_lookup_tables(path: str | Path | None = None) -> RrtmgLwCompressedLookupTables:
    """Load and compress WRF RRTMG longwave lookup tables once per path."""

    return combine_rrtmg_lw_lookup_tables(load_rrtmg_lw_data(path))


def inatm_lw_column(
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
    cfc11vmr: Any,
    cfc12vmr: Any,
    cfc22vmr: Any,
    ccl4vmr: Any,
    emis: Any,
    inflglw: int,
    iceflglw: int,
    liqflglw: int,
    cldfmcl: Any | None = None,
    taucmcl: Any | None = None,
    ciwpmcl: Any | None = None,
    clwpmcl: Any | None = None,
    cswpmcl: Any | None = None,
    reicmcl: Any | None = None,
    relqmcl: Any | None = None,
    resnmcl: Any | None = None,
    tauaer: Any | None = None,
    spectral: RrtmgSpectralInit | None = None,
) -> RrtmgLwInatmResult:
    """WRF longwave ``inatm`` for one atmospheric column."""

    n = int(nlay)
    spec = lwdatinit() if spectral is None else spectral
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

    pz = plev_arr
    tz = tlev_arr
    amd = jnp.asarray(28.9660, dtype=jnp.float32)
    amw = jnp.asarray(18.0160, dtype=jnp.float32)
    avogad = jnp.asarray(spec.avogad, dtype=jnp.float32)
    grav = jnp.asarray(spec.grav, dtype=jnp.float32)
    one = jnp.float32(1.0)
    amm = _wrf_real_add(
        _wrf_real_mul(_wrf_real_sub(one, h2o), amd),
        _wrf_real_mul(h2o, amw),
    )
    pressure_mass = _wrf_real_mul(
        _wrf_real_mul(_wrf_real_sub(pz[:-1], pz[1:]), jnp.float32(1.0e3)),
        avogad,
    )
    mass_denominator = _wrf_real_mul(jnp.float32(1.0e2), grav)
    mass_denominator = _wrf_real_mul(mass_denominator, amm)
    mass_denominator = _wrf_real_mul(
        mass_denominator,
        _wrf_real_add(one, h2o),
    )
    coldry = _wrf_real_div(pressure_mass, mass_denominator)

    wkl_vmr = jnp.zeros((38, n), dtype=jnp.float32)
    wkl_vmr = wkl_vmr.at[0].set(h2o)
    wkl_vmr = wkl_vmr.at[1].set(co2)
    wkl_vmr = wkl_vmr.at[2].set(o3)
    wkl_vmr = wkl_vmr.at[3].set(n2o)
    wkl_vmr = wkl_vmr.at[5].set(ch4)
    wkl_vmr = wkl_vmr.at[6].set(o2)
    summol = jnp.zeros((n,), dtype=jnp.float32)
    for imol in range(1, 7):
        summol = _wrf_real_add(summol, wkl_vmr[imol])
    wbrodl = _wrf_real_mul(coldry, _wrf_real_sub(one, summol))
    wkl = _wrf_real_mul(wkl_vmr, coldry[None, :])

    def accumulate_columns(
        totals: tuple[Any, Any],
        values: tuple[Any, Any],
    ) -> tuple[tuple[Any, Any], None]:
        amttl, wvttl = totals
        coldry_level, water_level = values
        amttl = _wrf_real_add(
            _wrf_real_add(amttl, coldry_level),
            water_level,
        )
        wvttl = _wrf_real_add(wvttl, water_level)
        return (amttl, wvttl), None

    (amttl, wvttl), _ = lax.scan(
        accumulate_columns,
        (jnp.float32(0.0), jnp.float32(0.0)),
        (coldry, wkl[0]),
    )
    wvsh = _wrf_real_div(
        _wrf_real_mul(amw, wvttl),
        _wrf_real_mul(amd, amttl),
    )
    pwvcm = _wrf_real_div(
        _wrf_real_mul(
            wvsh,
            _wrf_real_mul(jnp.float32(1.0e3), pz[0]),
        ),
        _wrf_real_mul(jnp.float32(1.0e2), grav),
    )

    cross_vmr = jnp.stack(
        (
            jnp.asarray(ccl4vmr, dtype=jnp.float32)[:n],
            jnp.asarray(cfc11vmr, dtype=jnp.float32)[:n],
            jnp.asarray(cfc12vmr, dtype=jnp.float32)[:n],
            jnp.asarray(cfc22vmr, dtype=jnp.float32)[:n],
        ),
        axis=0,
    )
    wx = _wrf_real_mul(
        _wrf_real_mul(cross_vmr, coldry[None, :]),
        jnp.float32(1.0e-20),
    )
    semiss = jnp.asarray(emis, dtype=jnp.float32)[:RRTMG_LW_NBND]

    taua = jnp.zeros((n, RRTMG_LW_NBND), dtype=jnp.float32)
    if int(iaer) >= 1 and tauaer is not None:
        taua = jnp.asarray(tauaer, dtype=jnp.float32)[:n, :RRTMG_LW_NBND]

    cldfmc = jnp.zeros((RRTMG_LW_NGPT, n), dtype=jnp.float32)
    taucmc = jnp.zeros((RRTMG_LW_NGPT, n), dtype=jnp.float32)
    ciwpmc = jnp.zeros((RRTMG_LW_NGPT, n), dtype=jnp.float32)
    clwpmc = jnp.zeros((RRTMG_LW_NGPT, n), dtype=jnp.float32)
    cswpmc = jnp.zeros((RRTMG_LW_NGPT, n), dtype=jnp.float32)
    reicmc = jnp.zeros((n,), dtype=jnp.float32)
    relqmc = jnp.zeros((n,), dtype=jnp.float32)
    resnmc = jnp.zeros((n,), dtype=jnp.float32)
    inflag = int(inflglw) if int(icld) >= 1 else 0
    iceflag = int(iceflglw) if int(icld) >= 1 else 0
    liqflag = int(liqflglw) if int(icld) >= 1 else 0

    if int(icld) >= 1:
        cldfmc = jnp.asarray(cldfmcl, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n] if cldfmcl is not None else cldfmc
        taucmc = jnp.asarray(taucmcl, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n] if taucmcl is not None else taucmc
        ciwpmc = jnp.asarray(ciwpmcl, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n] if ciwpmcl is not None else ciwpmc
        clwpmc = jnp.asarray(clwpmcl, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n] if clwpmcl is not None else clwpmc
        cswpmc = jnp.asarray(cswpmcl, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n] if cswpmcl is not None else cswpmc
        reicmc = jnp.asarray(reicmcl, dtype=jnp.float32)[:n] if reicmcl is not None else reicmc
        relqmc = jnp.asarray(relqmcl, dtype=jnp.float32)[:n] if relqmcl is not None else relqmc
        resnmc = jnp.asarray(resnmcl, dtype=jnp.float32)[:n] if resnmcl is not None else resnmc

    return RrtmgLwInatmResult(
        nlayers=n,
        pavel=play_arr,
        pz=pz,
        tavel=tlay_arr,
        tz=tz,
        tbound=jnp.asarray(tsfc, dtype=jnp.float32),
        semiss=semiss,
        coldry=coldry,
        wbrodl=wbrodl,
        wkl=wkl,
        wx=wx,
        pwvcm=pwvcm,
        inflag=inflag,
        iceflag=iceflag,
        liqflag=liqflag,
        cldfmc=cldfmc,
        taucmc=taucmc,
        ciwpmc=ciwpmc,
        clwpmc=clwpmc,
        cswpmc=cswpmc,
        reicmc=reicmc,
        relqmc=relqmc,
        resnmc=resnmc,
        taua=taua,
    )


def _fortran_int(value: Any) -> Any:
    return jnp.asarray(value).astype(jnp.int32)


def _interp_planck_rows(table: Any, fortran_index: Any, fraction: Any) -> Any:
    row0 = fortran_index.astype(jnp.int32) - 1
    lower = table[row0]
    upper = table[row0 + 1]
    return _wrf_real_add(
        lower,
        _wrf_real_mul(
            fraction[..., None],
            _wrf_real_sub(upper, lower),
        ),
    )


def _interp_planck_vector(table: Any, fortran_index: Any, fraction: Any) -> Any:
    row0 = fortran_index.astype(jnp.int32) - 1
    lower = table[row0]
    upper = table[row0 + 1]
    fraction_arr = jnp.asarray(fraction, dtype=jnp.float32)
    if jnp.ndim(table) > jnp.ndim(fortran_index):
        fraction_arr = fraction_arr[..., None]
    return _wrf_real_add(
        lower,
        _wrf_real_mul(fraction_arr, _wrf_real_sub(upper, lower)),
    )


def _interp_by_fortran_row(table: Any, fortran_index: Any, fraction: Any, band0: Any) -> Any:
    row0 = fortran_index.astype(jnp.int32) - 1
    lower = table[row0, band0]
    upper = table[row0 + 1, band0]
    return lower + fraction * (upper - lower)


def setcoef_lw(
    *,
    nlayers: int,
    istart: int,
    pavel: Any,
    tavel: Any,
    tz: Any,
    tbound: Any,
    semiss: Any,
    coldry: Any,
    wkl: Any,
    wbroad: Any,
    tables: RrtmgLwSetcoefTables | None = None,
) -> RrtmgLwSetcoefResult:
    """WRF longwave ``setcoef`` over one column or leading column batches."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    preflog = jnp.asarray(tab.preflog, dtype=jnp.float32)
    tref = jnp.asarray(tab.tref, dtype=jnp.float32)
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    totplnk = jnp.asarray(tab.totplnk, dtype=jnp.float32)
    totplk16 = jnp.asarray(tab.totplk16, dtype=jnp.float32)

    pavel_arr = jnp.asarray(pavel, dtype=jnp.float32)[..., :n]
    tavel_arr = jnp.asarray(tavel, dtype=jnp.float32)[..., :n]
    tz_arr = jnp.asarray(tz, dtype=jnp.float32)[..., : n + 1]
    semiss_arr = jnp.asarray(semiss, dtype=jnp.float32)[..., :RRTMG_LW_NBND]
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[..., :n]
    wkl_arr = jnp.asarray(wkl, dtype=jnp.float32)[..., :, :n]
    wbroad_arr = jnp.asarray(wbroad, dtype=jnp.float32)[..., :n]

    tbound_arr = jnp.asarray(tbound, dtype=jnp.float32)
    temperature_offset = jnp.float32(159.0)
    bound_temperature = _wrf_real_sub(tbound_arr, temperature_offset)
    indbound = jnp.clip(_fortran_int(bound_temperature), 1, 180)
    tbndfrac = _wrf_real_sub(bound_temperature, indbound.astype(jnp.float32))
    level0_temperature = _wrf_real_sub(tz_arr[..., 0], temperature_offset)
    indlev0 = jnp.clip(_fortran_int(level0_temperature), 1, 180)
    t0frac = _wrf_real_sub(level0_temperature, indlev0.astype(jnp.float32))

    layer_temperature = _wrf_real_sub(tavel_arr, temperature_offset)
    indlay = jnp.clip(_fortran_int(layer_temperature), 1, 180)
    tlayfrac = _wrf_real_sub(layer_temperature, indlay.astype(jnp.float32))
    level_temperature = _wrf_real_sub(tz_arr[..., 1 : n + 1], temperature_offset)
    indlev = jnp.clip(_fortran_int(level_temperature), 1, 180)
    tlevfrac = _wrf_real_sub(level_temperature, indlev.astype(jnp.float32))

    plankbnd_all = _wrf_real_mul(
        semiss_arr,
        _interp_planck_vector(totplnk, indbound, tbndfrac),
    )
    planklev0_all = _interp_planck_vector(totplnk, indlev0, t0frac)
    planklay = _interp_planck_rows(totplnk, indlay, tlayfrac)
    planklev_layers = _interp_planck_rows(totplnk, indlev, tlevfrac)
    if int(istart) == 16:
        band = RRTMG_LW_NBND - 1
        plankbnd_all = plankbnd_all.at[..., band].set(
            _wrf_real_mul(
                semiss_arr[..., band],
                _interp_planck_vector(totplk16, indbound, tbndfrac),
            )
        )
        planklev0_all = planklev0_all.at[..., band].set(
            _interp_planck_vector(totplk16, indlev0, t0frac)
        )
        planklay = planklay.at[..., :, band].set(
            _interp_planck_vector(totplk16, indlay, tlayfrac)
        )
        planklev_layers = planklev_layers.at[..., :, band].set(
            _interp_planck_vector(totplk16, indlev, tlevfrac)
        )
    planklev = jnp.concatenate((planklev0_all[..., None, :], planklev_layers), axis=-2)

    one = jnp.float32(1.0)
    stpfac = _wrf_real_div(jnp.float32(296.0), jnp.float32(1013.0))
    plog = wrf_glibc_logf_positive_normal(pavel_arr)
    jp = jnp.clip(
        _fortran_int(
            _wrf_real_sub(
                jnp.float32(36.0),
                _wrf_real_mul(
                    jnp.float32(5.0),
                    _wrf_real_add(plog, jnp.float32(0.04)),
                ),
            )
        ),
        1,
        58,
    )
    jp_idx = jp - 1
    jp1_idx = jp
    fp = _wrf_real_mul(
        jnp.float32(5.0),
        _wrf_real_sub(preflog[jp_idx], plog),
    )
    temperature_fraction = _wrf_real_div(
        _wrf_real_sub(tavel_arr, tref[jp_idx]),
        jnp.float32(15.0),
    )
    jt = jnp.clip(
        _fortran_int(_wrf_real_add(jnp.float32(3.0), temperature_fraction)),
        1,
        4,
    )
    ft = _wrf_real_sub(
        temperature_fraction,
        (jt - 3).astype(jnp.float32),
    )
    temperature_fraction1 = _wrf_real_div(
        _wrf_real_sub(tavel_arr, tref[jp1_idx]),
        jnp.float32(15.0),
    )
    jt1 = jnp.clip(
        _fortran_int(_wrf_real_add(jnp.float32(3.0), temperature_fraction1)),
        1,
        4,
    )
    ft1 = _wrf_real_sub(
        temperature_fraction1,
        (jt1 - 3).astype(jnp.float32),
    )

    water = _wrf_real_div(wkl_arr[..., 0, :], coldry_arr)
    scalefac = _wrf_real_div(
        _wrf_real_mul(pavel_arr, stpfac),
        tavel_arr,
    )
    troposphere = plog > 4.56
    laytrop = jnp.sum(troposphere.astype(jnp.int32), axis=-1)

    forfac_base = _wrf_real_div(scalefac, _wrf_real_add(one, water))
    factor_trop_for = _wrf_real_div(
        _wrf_real_sub(jnp.float32(332.0), tavel_arr),
        jnp.float32(36.0),
    )
    indfor_trop = jnp.clip(_fortran_int(factor_trop_for), 1, 2)
    forfrac_trop = _wrf_real_sub(
        factor_trop_for,
        indfor_trop.astype(jnp.float32),
    )
    factor_strat_for = _wrf_real_div(
        _wrf_real_sub(tavel_arr, jnp.float32(188.0)),
        jnp.float32(36.0),
    )
    indfor = jnp.where(troposphere, indfor_trop, jnp.full_like(jp, 3, dtype=jnp.int32))
    forfrac = jnp.where(
        troposphere,
        forfrac_trop,
        _wrf_real_sub(factor_strat_for, one),
    )

    factor_self = _wrf_real_div(
        _wrf_real_sub(tavel_arr, jnp.float32(188.0)),
        jnp.float32(7.2),
    )
    indself_trop = jnp.clip(_fortran_int(factor_self) - 7, 1, 9)
    selffrac_trop = _wrf_real_sub(
        factor_self,
        (indself_trop + 7).astype(jnp.float32),
    )
    indself = jnp.where(troposphere, indself_trop, jnp.zeros_like(jp, dtype=jnp.int32))
    selffrac = jnp.where(troposphere, selffrac_trop, jnp.float32(0.0))

    scaleminor = _wrf_real_div(pavel_arr, tavel_arr)
    scaleminorn2 = _wrf_real_mul(
        scaleminor,
        _wrf_real_div(
            wbroad_arr,
            _wrf_real_add(coldry_arr, wkl_arr[..., 0, :]),
        ),
    )
    factor_minor = _wrf_real_div(
        _wrf_real_sub(tavel_arr, jnp.float32(180.8)),
        jnp.float32(7.2),
    )
    indminor = jnp.clip(_fortran_int(factor_minor), 1, 18)
    minorfrac = _wrf_real_sub(factor_minor, indminor.astype(jnp.float32))

    tiny = jnp.asarray(1.0e-20, dtype=jnp.float32)
    colh2o = _wrf_real_mul(tiny, wkl_arr[..., 0, :])
    colco2 = _wrf_real_mul(tiny, wkl_arr[..., 1, :])
    colo3 = _wrf_real_mul(tiny, wkl_arr[..., 2, :])
    coln2o = _wrf_real_mul(tiny, wkl_arr[..., 3, :])
    colco = _wrf_real_mul(tiny, wkl_arr[..., 4, :])
    colch4 = _wrf_real_mul(tiny, wkl_arr[..., 5, :])
    colo2 = _wrf_real_mul(tiny, wkl_arr[..., 6, :])
    fallback = _wrf_real_mul(
        jnp.asarray(1.0e-32, dtype=jnp.float32),
        coldry_arr,
    )
    colco2 = jnp.where(colco2 == 0.0, fallback, colco2)
    colo3 = jnp.where(colo3 == 0.0, fallback, colo3)
    coln2o = jnp.where(coln2o == 0.0, fallback, coln2o)
    colco = jnp.where(colco == 0.0, fallback, colco)
    colch4 = jnp.where(colch4 == 0.0, fallback, colch4)
    colbrd = _wrf_real_mul(tiny, wbroad_arr)

    rat_h2oco2 = chi_mls[0, jp_idx] / chi_mls[1, jp_idx]
    rat_h2oco2_1 = chi_mls[0, jp1_idx] / chi_mls[1, jp1_idx]
    zeros = jnp.zeros_like(pavel_arr, dtype=jnp.float32)
    rat_h2oo3 = jnp.where(troposphere, chi_mls[0, jp_idx] / chi_mls[2, jp_idx], zeros)
    rat_h2oo3_1 = jnp.where(troposphere, chi_mls[0, jp1_idx] / chi_mls[2, jp1_idx], zeros)
    rat_h2on2o = jnp.where(troposphere, chi_mls[0, jp_idx] / chi_mls[3, jp_idx], zeros)
    rat_h2on2o_1 = jnp.where(troposphere, chi_mls[0, jp1_idx] / chi_mls[3, jp1_idx], zeros)
    rat_h2och4 = jnp.where(troposphere, chi_mls[0, jp_idx] / chi_mls[5, jp_idx], zeros)
    rat_h2och4_1 = jnp.where(troposphere, chi_mls[0, jp1_idx] / chi_mls[5, jp1_idx], zeros)
    rat_n2oco2 = jnp.where(troposphere, chi_mls[3, jp_idx] / chi_mls[1, jp_idx], zeros)
    rat_n2oco2_1 = jnp.where(troposphere, chi_mls[3, jp1_idx] / chi_mls[1, jp1_idx], zeros)
    rat_o3co2 = jnp.where(troposphere, zeros, chi_mls[2, jp_idx] / chi_mls[1, jp_idx])
    rat_o3co2_1 = jnp.where(troposphere, zeros, chi_mls[2, jp1_idx] / chi_mls[1, jp1_idx])

    compfp = _wrf_real_sub(one, fp)
    fac10 = _wrf_real_mul(compfp, ft)
    fac00 = _wrf_real_mul(compfp, _wrf_real_sub(one, ft))
    fac11 = _wrf_real_mul(fp, ft1)
    fac01 = _wrf_real_mul(fp, _wrf_real_sub(one, ft1))

    selffac = _wrf_real_mul(
        colh2o,
        _wrf_real_mul(water, forfac_base),
    )
    forfac = _wrf_real_mul(colh2o, forfac_base)

    return RrtmgLwSetcoefResult(
        laytrop=laytrop,
        jp=jp,
        jt=jt,
        jt1=jt1,
        planklay=planklay,
        planklev=planklev,
        plankbnd=plankbnd_all,
        colh2o=colh2o,
        colco2=colco2,
        colo3=colo3,
        coln2o=coln2o,
        colco=colco,
        colch4=colch4,
        colo2=colo2,
        colbrd=colbrd,
        fac00=fac00,
        fac01=fac01,
        fac10=fac10,
        fac11=fac11,
        rat_h2oco2=rat_h2oco2,
        rat_h2oco2_1=rat_h2oco2_1,
        rat_h2oo3=rat_h2oo3,
        rat_h2oo3_1=rat_h2oo3_1,
        rat_h2on2o=rat_h2on2o,
        rat_h2on2o_1=rat_h2on2o_1,
        rat_h2och4=rat_h2och4,
        rat_h2och4_1=rat_h2och4_1,
        rat_n2oco2=rat_n2oco2,
        rat_n2oco2_1=rat_n2oco2_1,
        rat_o3co2=rat_o3co2,
        rat_o3co2_1=rat_o3co2_1,
        selffac=selffac,
        selffrac=selffrac,
        indself=indself,
        forfac=forfac,
        forfrac=forfrac,
        indfor=indfor,
        minorfrac=minorfrac,
        scaleminor=scaleminor,
        scaleminorn2=scaleminorn2,
        indminor=indminor,
    )


def _flatten_ref_table_5x(values: Any) -> Any:
    arr = jnp.asarray(values, dtype=jnp.float32)
    return jnp.reshape(jnp.swapaxes(arr, 0, 1), (arr.shape[0] * arr.shape[1], arr.shape[2]))


def _flatten_ref_table_fortran(values: Any) -> Any:
    arr = jnp.asarray(values, dtype=jnp.float32)
    axes = tuple(range(arr.ndim - 2, -1, -1)) + (arr.ndim - 1,)
    return jnp.reshape(jnp.transpose(arr, axes), (int(np.prod(arr.shape[:-1])), arr.shape[-1]))


def _safe_rows(table: Any, indices: Any) -> Any:
    arr = jnp.asarray(table, dtype=jnp.float32)
    idx = jnp.clip(jnp.asarray(indices, dtype=jnp.int32), 0, arr.shape[0] - 1)
    return arr[idx]


def _edge_interp_factors(specparm: Any, fs: Any, fa: Any, fb: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    one = jnp.float32(1.0)
    two = jnp.float32(2.0)
    low = specparm < 0.125
    high = specparm > 0.875
    p_low = _wrf_real_sub(fs, one)
    p4_low = _wrf_real_store(p_low**4)
    fk0_low = p4_low
    fk1_low = _wrf_real_sub(
        _wrf_real_sub(one, p_low),
        _wrf_real_mul(two, p4_low),
    )
    fk2_low = _wrf_real_add(p_low, p4_low)
    p_high = _wrf_real_sub(jnp.float32(0.0), fs)
    p4_high = _wrf_real_store(p_high**4)
    fk0_high = p4_high
    fk1_high = _wrf_real_sub(
        _wrf_real_sub(one, p_high),
        _wrf_real_mul(two, p4_high),
    )
    fk2_high = _wrf_real_add(p_high, p4_high)

    fac000 = jnp.where(
        low,
        _wrf_real_mul(fk0_low, fa),
        jnp.where(
            high,
            _wrf_real_mul(fk0_high, fa),
            _wrf_real_mul(_wrf_real_sub(one, fs), fa),
        ),
    )
    fac100 = jnp.where(
        low,
        _wrf_real_mul(fk1_low, fa),
        jnp.where(
            high,
            _wrf_real_mul(fk1_high, fa),
            _wrf_real_mul(fs, fa),
        ),
    )
    fac200 = jnp.where(
        low,
        _wrf_real_mul(fk2_low, fa),
        jnp.where(
            high,
            _wrf_real_mul(fk2_high, fa),
            jnp.float32(0.0),
        ),
    )
    fac010 = jnp.where(
        low,
        _wrf_real_mul(fk0_low, fb),
        jnp.where(
            high,
            _wrf_real_mul(fk0_high, fb),
            _wrf_real_mul(_wrf_real_sub(one, fs), fb),
        ),
    )
    fac110 = jnp.where(
        low,
        _wrf_real_mul(fk1_low, fb),
        jnp.where(
            high,
            _wrf_real_mul(fk1_high, fb),
            _wrf_real_mul(fs, fb),
        ),
    )
    fac210 = jnp.where(
        low,
        _wrf_real_mul(fk2_low, fb),
        jnp.where(
            high,
            _wrf_real_mul(fk2_high, fb),
            jnp.float32(0.0),
        ),
    )
    return fac000, fac100, fac200, fac010, fac110, fac210


def _edge_mixed_major(table: Any, base: Any, specparm: Any, fac0: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
    fac000, fac100, fac200, fac010, fac110, fac210 = fac0
    low = specparm < 0.125
    high = specparm > 0.875
    low_val = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac200[:, None],
            fac010[:, None],
            fac110[:, None],
            fac210[:, None],
        ),
        (
            _safe_rows(table, base),
            _safe_rows(table, base + 1),
            _safe_rows(table, base + 2),
            _safe_rows(table, base + 9),
            _safe_rows(table, base + 10),
            _safe_rows(table, base + 11),
        ),
    )
    high_val = _wrf_real_linear_combination(
        (
            fac200[:, None],
            fac100[:, None],
            fac000[:, None],
            fac210[:, None],
            fac110[:, None],
            fac010[:, None],
        ),
        (
            _safe_rows(table, base - 1),
            _safe_rows(table, base),
            _safe_rows(table, base + 1),
            _safe_rows(table, base + 8),
            _safe_rows(table, base + 9),
            _safe_rows(table, base + 10),
        ),
    )
    mid_val = _wrf_real_linear_combination(
        (
            fac000[:, None],
            fac100[:, None],
            fac010[:, None],
            fac110[:, None],
        ),
        (
            _safe_rows(table, base),
            _safe_rows(table, base + 1),
            _safe_rows(table, base + 9),
            _safe_rows(table, base + 10),
        ),
    )
    return jnp.where(low[:, None], low_val, jnp.where(high[:, None], high_val, mid_val))


def taumol1_lw(
    *,
    nlayers: int,
    pavel: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colbrd: Any,
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
    minorfrac: Any,
    scaleminorn2: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb1`` for band 1."""

    n = int(nlayers)
    pavel_arr = jnp.asarray(pavel, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colbrd_arr = jnp.asarray(colbrd, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    scaleminorn2_arr = jnp.asarray(scaleminorn2, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ka_mn2 = jnp.asarray(lookup.ka_mn2, dtype=jnp.float32)
    kb_mn2 = jnp.asarray(lookup.kb_mn2, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mn2.shape[0] - 2)
    indm1 = indm0 + 1

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_add(
            selfref[inds0],
            _wrf_real_mul(
                selffrac_arr[:, None],
                _wrf_real_sub(selfref[inds1], selfref[inds0]),
            ),
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_add(
            forref[indf0],
            _wrf_real_mul(
                forfrac_arr[:, None],
                _wrf_real_sub(forref[indf1], forref[indf0]),
            ),
        ),
    )
    scalen2 = _wrf_real_mul(colbrd_arr, scaleminorn2_arr)
    taun2_t = _wrf_real_mul(
        scalen2[:, None],
        _wrf_real_add(
            ka_mn2[indm0],
            _wrf_real_mul(
                minorfrac_arr[:, None],
                _wrf_real_sub(ka_mn2[indm1], ka_mn2[indm0]),
            ),
        ),
    )
    taun2_s = _wrf_real_mul(
        scalen2[:, None],
        _wrf_real_add(
            kb_mn2[indm0],
            _wrf_real_mul(
                minorfrac_arr[:, None],
                _wrf_real_sub(kb_mn2[indm1], kb_mn2[indm0]),
            ),
        ),
    )

    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    corr_t = jnp.where(
        pavel_arr < jnp.float32(250.0),
        _wrf_real_sub(
            jnp.float32(1.0),
            _wrf_real_div(
                _wrf_real_mul(
                    jnp.float32(0.15),
                    _wrf_real_sub(jnp.float32(250.0), pavel_arr),
                ),
                jnp.float32(154.4),
            ),
        ),
        jnp.float32(1.0),
    )
    corr_s = _wrf_real_sub(
        jnp.float32(1.0),
        _wrf_real_mul(
            jnp.float32(0.15),
            _wrf_real_div(pavel_arr, jnp.float32(95.6)),
        ),
    )
    taug_t = _wrf_real_mul(
        corr_t[:, None],
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_mul(colh2o_arr[:, None], major_t),
                    tauself,
                ),
                taufor,
            ),
            taun2_t,
        ),
    )
    taug_s = _wrf_real_mul(
        corr_s[:, None],
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_mul(colh2o_arr[:, None], major_s),
                taufor,
            ),
            taun2_s,
        ),
    )
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], jnp.broadcast_to(fracrefa[None, :], (n, ng)), jnp.broadcast_to(fracrefb[None, :], (n, ng)))
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol2_lw(
    *,
    nlayers: int,
    pavel: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
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
    lookup: RrtmgLwCompressedBandData,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb2`` for band 2."""

    n = int(nlayers)
    pavel_arr = jnp.asarray(pavel, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_add(
            selfref[inds0],
            _wrf_real_mul(
                selffrac_arr[:, None],
                _wrf_real_sub(selfref[inds1], selfref[inds0]),
            ),
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_add(
            forref[indf0],
            _wrf_real_mul(
                forfrac_arr[:, None],
                _wrf_real_sub(forref[indf1], forref[indf0]),
            ),
        ),
    )
    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    corr_t = _wrf_real_sub(
        jnp.float32(1.0),
        _wrf_real_div(
            _wrf_real_mul(
                jnp.float32(0.05),
                _wrf_real_sub(pavel_arr, jnp.float32(100.0)),
            ),
            jnp.float32(900.0),
        ),
    )
    taug_t = _wrf_real_mul(
        corr_t[:, None],
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_mul(colh2o_arr[:, None], major_t),
                tauself,
            ),
            taufor,
        ),
    )
    taug_s = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], major_s),
        taufor,
    )
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(
        trop_mask[:, None],
        jnp.broadcast_to(fracrefa[None, :], (n, ng)),
        jnp.broadcast_to(fracrefb[None, :], (n, ng)),
    )
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol3_lw(
    *,
    nlayers: int,
    pavel: Any,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    coln2o: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oco2: Any,
    rat_h2oco2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb3`` for band 3."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    pavel_arr = jnp.asarray(pavel, dtype=jnp.float32)[:n]
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    coln2o_arr = jnp.asarray(coln2o, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2oco2_arr = jnp.asarray(rat_h2oco2, dtype=jnp.float32)[:n]
    rat_h2oco2_1_arr = jnp.asarray(rat_h2oco2_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_fortran(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ka_mn2o = jnp.asarray(lookup.ka_mn2o, dtype=jnp.float32)
    kb_mn2o = jnp.asarray(lookup.kb_mn2o, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 8], chi_mls[1, 8])
    refrat_planck_b = _wrf_real_div(chi_mls[0, 12], chi_mls[1, 12])
    refrat_m_a = _wrf_real_div(chi_mls[0, 2], chi_mls[1, 2])
    refrat_m_b = _wrf_real_div(chi_mls[0, 12], chi_mls[1, 12])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_1_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jmn2o_t, fmn2o_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    chi_n2o = _wrf_real_div(coln2o_arr, coldry_arr)
    chi_ref = chi_mls[3, jnp.clip(jp_arr, 0, chi_mls.shape[1] - 1)]
    ratn2o = _wrf_real_div(
        _wrf_real_mul(jnp.float32(1.0e20), chi_n2o),
        chi_ref,
    )
    adjfac = _wrf_real_add(
        jnp.float32(0.5),
        _wrf_real_active_pow(
            _wrf_real_sub(ratn2o, jnp.float32(0.5)),
            0.65,
            ratn2o > jnp.float32(1.5),
        ),
    )
    adjusted_n2o = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac, chi_ref),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcoln2o = jnp.where(ratn2o > jnp.float32(1.5), adjusted_n2o, coln2o_arr)

    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mn2o.shape[1] - 2)
    indm1 = indm0 + 1
    jn0_t = jnp.clip(jmn2o_t - 1, 0, ka_mn2o.shape[0] - 2)
    jn1_t = jn0_t + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    n2om1_t = _wrf_real_interp(
        ka_mn2o[jn0_t, indm0],
        ka_mn2o[jn1_t, indm0],
        fmn2o_t[:, None],
    )
    n2om2_t = _wrf_real_interp(
        ka_mn2o[jn0_t, indm1],
        ka_mn2o[jn1_t, indm1],
        fmn2o_t[:, None],
    )
    absn2o_t = _wrf_real_interp(
        n2om1_t,
        n2om2_t,
        minorfrac_arr[:, None],
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(tau_major_t, tau_major1_t),
                tauself,
            ),
            taufor,
        ),
        _wrf_real_mul(adjcoln2o[:, None], absn2o_t),
    )

    speccomb_s, _, js_s, fs_s = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    speccomb1_s, _, js1_s, fs1_s = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_1_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    _, _, jmn2o_s, fmn2o_s = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_b,
        colco2_arr,
        4.0,
        one_minus,
    )
    _, _, jpl_s, fpl_s = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_b,
        colco2_arr,
        4.0,
        one_minus,
    )

    s000, s100, s010, s110 = _wrf_real_linear_factors(
        fs_s,
        fac00_arr,
        fac10_arr,
    )
    s001, s101, s011, s111 = _wrf_real_linear_factors(
        fs1_s,
        fac01_arr,
        fac11_arr,
    )
    ind0_s = (jp_arr - 13) * 25 + (jt_arr - 1) * 5 + (js_s - 1)
    ind1_s = (jp_arr - 12) * 25 + (jt1_arr - 1) * 5 + (js1_s - 1)
    major_s = _wrf_real_mul(
        speccomb_s[:, None],
        _wrf_real_linear_combination(
            (
                s000[:, None],
                s100[:, None],
                s010[:, None],
                s110[:, None],
            ),
            (
                _safe_rows(absb, ind0_s),
                _safe_rows(absb, ind0_s + 1),
                _safe_rows(absb, ind0_s + 5),
                _safe_rows(absb, ind0_s + 6),
            ),
        ),
    )
    major1_s = _wrf_real_mul(
        speccomb1_s[:, None],
        _wrf_real_linear_combination(
            (
                s001[:, None],
                s101[:, None],
                s011[:, None],
                s111[:, None],
            ),
            (
                _safe_rows(absb, ind1_s),
                _safe_rows(absb, ind1_s + 1),
                _safe_rows(absb, ind1_s + 5),
                _safe_rows(absb, ind1_s + 6),
            ),
        ),
    )
    indm0_s = jnp.clip(indminor_arr - 1, 0, kb_mn2o.shape[1] - 2)
    indm1_s = indm0_s + 1
    jn0_s = jnp.clip(jmn2o_s - 1, 0, kb_mn2o.shape[0] - 2)
    jn1_s = jn0_s + 1
    n2om1_s = _wrf_real_interp(
        kb_mn2o[jn0_s, indm0_s],
        kb_mn2o[jn1_s, indm0_s],
        fmn2o_s[:, None],
    )
    n2om2_s = _wrf_real_interp(
        kb_mn2o[jn0_s, indm1_s],
        kb_mn2o[jn1_s, indm1_s],
        fmn2o_s[:, None],
    )
    absn2o_s = _wrf_real_interp(
        n2om1_s,
        n2om2_s,
        minorfrac_arr[:, None],
    )
    jpl0_s = jnp.clip(jpl_s - 1, 0, fracrefb.shape[1] - 2)
    fracs_s = _wrf_real_interp(
        fracrefb[:, jpl0_s].T,
        fracrefb[:, jpl0_s + 1].T,
        fpl_s[:, None],
    )
    taug_s = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(major_s, major1_s),
            taufor,
        ),
        _wrf_real_mul(adjcoln2o[:, None], absn2o_s),
    )

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol4_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oco2: Any,
    rat_h2oco2_1: Any,
    rat_o3co2: Any,
    rat_o3co2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb4`` for band 4."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2oco2_arr = jnp.asarray(rat_h2oco2, dtype=jnp.float32)[:n]
    rat_h2oco2_1_arr = jnp.asarray(rat_h2oco2_1, dtype=jnp.float32)[:n]
    rat_o3co2_arr = jnp.asarray(rat_o3co2, dtype=jnp.float32)[:n]
    rat_o3co2_1_arr = jnp.asarray(rat_o3co2_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_fortran(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 10], chi_mls[1, 10])
    refrat_planck_b = _wrf_real_div(chi_mls[2, 12], chi_mls[1, 12])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_1_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(tau_major_t, tau_major1_t),
            tauself,
        ),
        taufor,
    )

    speccomb_s, _, js_s, fs_s = _wrf_real_species_interpolation(
        colo3_arr,
        rat_o3co2_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    speccomb1_s, _, js1_s, fs1_s = _wrf_real_species_interpolation(
        colo3_arr,
        rat_o3co2_1_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    _, _, jpl_s, fpl_s = _wrf_real_species_interpolation(
        colo3_arr,
        refrat_planck_b,
        colco2_arr,
        4.0,
        one_minus,
    )

    s000, s100, s010, s110 = _wrf_real_linear_factors(
        fs_s,
        fac00_arr,
        fac10_arr,
    )
    s001, s101, s011, s111 = _wrf_real_linear_factors(
        fs1_s,
        fac01_arr,
        fac11_arr,
    )
    ind0_s = (jp_arr - 13) * 25 + (jt_arr - 1) * 5 + (js_s - 1)
    ind1_s = (jp_arr - 12) * 25 + (jt1_arr - 1) * 5 + (js1_s - 1)
    major_s = _wrf_real_mul(
        speccomb_s[:, None],
        _wrf_real_linear_combination(
            (
                s000[:, None],
                s100[:, None],
                s010[:, None],
                s110[:, None],
            ),
            (
                _safe_rows(absb, ind0_s),
                _safe_rows(absb, ind0_s + 1),
                _safe_rows(absb, ind0_s + 5),
                _safe_rows(absb, ind0_s + 6),
            ),
        ),
    )
    major1_s = _wrf_real_mul(
        speccomb1_s[:, None],
        _wrf_real_linear_combination(
            (
                s001[:, None],
                s101[:, None],
                s011[:, None],
                s111[:, None],
            ),
            (
                _safe_rows(absb, ind1_s),
                _safe_rows(absb, ind1_s + 1),
                _safe_rows(absb, ind1_s + 5),
                _safe_rows(absb, ind1_s + 6),
            ),
        ),
    )
    taug_s = _wrf_real_add(major_s, major1_s)
    correction = jnp.asarray(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.88, 1.07, 1.1, 0.99, 0.88, 0.943],
        dtype=jnp.float32,
    )
    taug_s = _wrf_real_mul(taug_s, correction[None, :])
    jpl0_s = jnp.clip(jpl_s - 1, 0, fracrefb.shape[1] - 2)
    fracs_s = _wrf_real_interp(
        fracrefb[:, jpl0_s].T,
        fracrefb[:, jpl0_s + 1].T,
        fpl_s[:, None],
    )

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol5_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    wx: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oco2: Any,
    rat_h2oco2_1: Any,
    rat_o3co2: Any,
    rat_o3co2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb5`` for band 5."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:n]
    wx_arr = jnp.asarray(wx, dtype=jnp.float32)
    wx_ccl4 = wx_arr[0, :n] if wx_arr.ndim == 2 else wx_arr[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2oco2_arr = jnp.asarray(rat_h2oco2, dtype=jnp.float32)[:n]
    rat_h2oco2_1_arr = jnp.asarray(rat_h2oco2_1, dtype=jnp.float32)[:n]
    rat_o3co2_arr = jnp.asarray(rat_o3co2, dtype=jnp.float32)[:n]
    rat_o3co2_1_arr = jnp.asarray(rat_o3co2_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_fortran(lookup.kb)
    ka_mo3 = jnp.asarray(lookup.ka_mo3, dtype=jnp.float32)
    ccl4 = jnp.asarray(lookup.ccl4, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 4], chi_mls[1, 4])
    refrat_planck_b = _wrf_real_div(chi_mls[2, 42], chi_mls[1, 42])
    refrat_m_a = _wrf_real_div(chi_mls[0, 6], chi_mls[1, 6])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_1_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jmo3, fmo3 = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mo3.shape[1] - 2)
    indm1 = indm0 + 1
    jmo0 = jnp.clip(jmo3 - 1, 0, ka_mo3.shape[0] - 2)
    jmo1 = jmo0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    o3m1 = _wrf_real_interp(
        ka_mo3[jmo0, indm0],
        ka_mo3[jmo1, indm0],
        fmo3[:, None],
    )
    o3m2 = _wrf_real_interp(
        ka_mo3[jmo0, indm1],
        ka_mo3[jmo1, indm1],
        fmo3[:, None],
    )
    abso3 = _wrf_real_interp(
        o3m1,
        o3m2,
        minorfrac_arr[:, None],
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_add(tau_major_t, tau_major1_t),
                    tauself,
                ),
                taufor,
            ),
            _wrf_real_mul(abso3, colo3_arr[:, None]),
        ),
        _wrf_real_mul(wx_ccl4[:, None], ccl4[None, :]),
    )

    speccomb_s, _, js_s, fs_s = _wrf_real_species_interpolation(
        colo3_arr,
        rat_o3co2_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    speccomb1_s, _, js1_s, fs1_s = _wrf_real_species_interpolation(
        colo3_arr,
        rat_o3co2_1_arr,
        colco2_arr,
        4.0,
        one_minus,
    )
    _, _, jpl_s, fpl_s = _wrf_real_species_interpolation(
        colo3_arr,
        refrat_planck_b,
        colco2_arr,
        4.0,
        one_minus,
    )

    s000, s100, s010, s110 = _wrf_real_linear_factors(
        fs_s,
        fac00_arr,
        fac10_arr,
    )
    s001, s101, s011, s111 = _wrf_real_linear_factors(
        fs1_s,
        fac01_arr,
        fac11_arr,
    )
    ind0_s = (jp_arr - 13) * 25 + (jt_arr - 1) * 5 + (js_s - 1)
    ind1_s = (jp_arr - 12) * 25 + (jt1_arr - 1) * 5 + (js1_s - 1)
    major_s = _wrf_real_mul(
        speccomb_s[:, None],
        _wrf_real_linear_combination(
            (
                s000[:, None],
                s100[:, None],
                s010[:, None],
                s110[:, None],
            ),
            (
                _safe_rows(absb, ind0_s),
                _safe_rows(absb, ind0_s + 1),
                _safe_rows(absb, ind0_s + 5),
                _safe_rows(absb, ind0_s + 6),
            ),
        ),
    )
    major1_s = _wrf_real_mul(
        speccomb1_s[:, None],
        _wrf_real_linear_combination(
            (
                s001[:, None],
                s101[:, None],
                s011[:, None],
                s111[:, None],
            ),
            (
                _safe_rows(absb, ind1_s),
                _safe_rows(absb, ind1_s + 1),
                _safe_rows(absb, ind1_s + 5),
                _safe_rows(absb, ind1_s + 6),
            ),
        ),
    )
    taug_s = _wrf_real_add(
        _wrf_real_add(major_s, major1_s),
        _wrf_real_mul(wx_ccl4[:, None], ccl4[None, :]),
    )
    jpl0_s = jnp.clip(jpl_s - 1, 0, fracrefb.shape[1] - 2)
    fracs_s = _wrf_real_interp(
        fracrefb[:, jpl0_s].T,
        fracrefb[:, jpl0_s + 1].T,
        fpl_s[:, None],
    )

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol6_lw(
    *,
    nlayers: int,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    wx: Any,
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
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb6`` for band 6."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    wx_arr = jnp.asarray(wx, dtype=jnp.float32)
    wx_cfc11 = wx_arr[1, :n]
    wx_cfc12 = wx_arr[2, :n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    ka_mco2 = jnp.asarray(lookup.ka_mco2, dtype=jnp.float32)
    cfc11adj = jnp.asarray(lookup.cfc11adj, dtype=jnp.float32)
    cfc12 = jnp.asarray(lookup.cfc12, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0 = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1 = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mco2.shape[0] - 2)
    indm1 = indm0 + 1

    chi_co2 = colco2_arr / coldry_arr
    chi_ref = chi_mls[1, jnp.clip(jp_arr, 0, chi_mls.shape[1] - 1)]
    ratco2 = 1.0e20 * chi_co2 / chi_ref
    adjfac = 2.0 + _wrf_real_active_pow(jnp.maximum(ratco2 - 2.0, 0.0), 0.77, ratco2 > 3.0)
    adjcolco2 = jnp.where(ratco2 > 3.0, adjfac * chi_ref * coldry_arr * 1.0e-20, colco2_arr)

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_add(
            selfref[inds0],
            _wrf_real_mul(
                selffrac_arr[:, None],
                _wrf_real_sub(selfref[inds1], selfref[inds0]),
            ),
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_add(
            forref[indf0],
            _wrf_real_mul(
                forfrac_arr[:, None],
                _wrf_real_sub(forref[indf1], forref[indf0]),
            ),
        ),
    )
    absco2 = _wrf_real_add(
        ka_mco2[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(ka_mco2[indm1], ka_mco2[indm0]),
        ),
    )
    major = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0],
            absa[ind0 + 1],
            absa[ind1],
            absa[ind1 + 1],
        ),
    )
    cfc11_tau = _wrf_real_mul(wx_cfc11[:, None], cfc11adj[None, :])
    cfc12_tau = _wrf_real_mul(wx_cfc12[:, None], cfc12[None, :])
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_add(
                        _wrf_real_mul(colh2o_arr[:, None], major),
                        tauself,
                    ),
                    taufor,
                ),
                _wrf_real_mul(adjcolco2[:, None], absco2),
            ),
            cfc11_tau,
        ),
        cfc12_tau,
    )
    cross = _wrf_real_add(cfc11_tau, cfc12_tau)
    taug = jnp.where(trop_mask[:, None], taug_t, cross)
    fracs = jnp.broadcast_to(fracrefa[None, :], (n, ng))
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol7_lw(
    *,
    nlayers: int,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oo3: Any,
    rat_h2oo3_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb7`` for band 7."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2oo3_arr = jnp.asarray(rat_h2oo3, dtype=jnp.float32)[:n]
    rat_h2oo3_1_arr = jnp.asarray(rat_h2oo3_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    ka_mco2 = jnp.asarray(lookup.ka_mco2, dtype=jnp.float32)
    kb_mco2 = jnp.asarray(lookup.kb_mco2, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 2], chi_mls[2, 2])
    refrat_m_a = _wrf_real_div(chi_mls[0, 2], chi_mls[2, 2])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oo3_arr,
        colo3_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oo3_1_arr,
        colo3_arr,
        8.0,
        one_minus,
    )
    _, _, jmco2_t, fmco2_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a,
        colo3_arr,
        8.0,
        one_minus,
    )

    chi_co2 = _wrf_real_div(colco2_arr, coldry_arr)
    chi_ref = chi_mls[1, jnp.clip(jp_arr, 0, chi_mls.shape[1] - 1)]
    ratco2 = _wrf_real_div(
        _wrf_real_mul(jnp.float32(1.0e20), chi_co2),
        chi_ref,
    )
    adjfac_t = _wrf_real_add(
        jnp.float32(3.0),
        _wrf_real_active_pow(
            _wrf_real_sub(ratco2, jnp.float32(3.0)),
            0.79,
            ratco2 > jnp.float32(3.0),
        ),
    )
    adjusted_co2_t = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac_t, chi_ref),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcolco2_t = jnp.where(ratco2 > jnp.float32(3.0), adjusted_co2_t, colco2_arr)

    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colo3_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mco2.shape[1] - 2)
    indm1 = indm0 + 1
    jm0 = jnp.clip(jmco2_t - 1, 0, ka_mco2.shape[0] - 2)
    jm1 = jm0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    co2m1 = _wrf_real_interp(
        ka_mco2[jm0, indm0],
        ka_mco2[jm1, indm0],
        fmco2_t[:, None],
    )
    co2m2 = _wrf_real_interp(
        ka_mco2[jm0, indm1],
        ka_mco2[jm1, indm1],
        fmco2_t[:, None],
    )
    absco2_t = _wrf_real_interp(
        co2m1,
        co2m2,
        minorfrac_arr[:, None],
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(tau_major_t, tau_major1_t),
                tauself,
            ),
            taufor,
        ),
        _wrf_real_mul(adjcolco2_t[:, None], absco2_t),
    )

    adjfac_s = _wrf_real_add(
        jnp.float32(2.0),
        _wrf_real_active_pow(
            _wrf_real_sub(ratco2, jnp.float32(2.0)),
            0.79,
            ratco2 > jnp.float32(3.0),
        ),
    )
    adjusted_co2_s = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac_s, chi_ref),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcolco2_s = jnp.where(ratco2 > jnp.float32(3.0), adjusted_co2_s, colco2_arr)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    indm0_s = jnp.clip(indminor_arr - 1, 0, kb_mco2.shape[0] - 2)
    indm1_s = indm0_s + 1
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    absco2_s = _wrf_real_interp(
        kb_mco2[indm0_s],
        kb_mco2[indm1_s],
        minorfrac_arr[:, None],
    )
    correction = jnp.asarray(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.88, 1.07, 1.1, 0.99, 0.855, 1.0],
        dtype=jnp.float32,
    )
    taug_s = _wrf_real_mul(
        _wrf_real_add(
            _wrf_real_mul(colo3_arr[:, None], major_s),
            _wrf_real_mul(adjcolco2_s[:, None], absco2_s),
        ),
        correction[None, :],
    )
    fracs_s = jnp.broadcast_to(fracrefb[None, :], (n, int(fracrefb.shape[0])))

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol8_lw(
    *,
    nlayers: int,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    coln2o: Any,
    wx: Any,
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
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb8`` for band 8."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:n]
    coln2o_arr = jnp.asarray(coln2o, dtype=jnp.float32)[:n]
    wx_arr = jnp.asarray(wx, dtype=jnp.float32)
    wx_cfc12 = wx_arr[2, :n]
    wx_cfc22 = wx_arr[3, :n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    ka_mco2 = jnp.asarray(lookup.ka_mco2, dtype=jnp.float32)
    kb_mco2 = jnp.asarray(lookup.kb_mco2, dtype=jnp.float32)
    ka_mo3 = jnp.asarray(lookup.ka_mo3, dtype=jnp.float32)
    ka_mn2o = jnp.asarray(lookup.ka_mn2o, dtype=jnp.float32)
    kb_mn2o = jnp.asarray(lookup.kb_mn2o, dtype=jnp.float32)
    cfc12 = jnp.asarray(lookup.cfc12, dtype=jnp.float32)
    cfc22adj = jnp.asarray(lookup.cfc22adj, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mco2.shape[0] - 2)
    indm1 = indm0 + 1

    chi_co2 = _wrf_real_div(colco2_arr, coldry_arr)
    chi_ref = chi_mls[1, jnp.clip(jp_arr, 0, chi_mls.shape[1] - 1)]
    ratco2 = _wrf_real_div(
        _wrf_real_mul(jnp.float32(1.0e20), chi_co2),
        chi_ref,
    )
    adjfac = _wrf_real_add(
        jnp.float32(2.0),
        _wrf_real_active_pow(
            jnp.maximum(
                _wrf_real_sub(ratco2, jnp.float32(2.0)),
                jnp.float32(0.0),
            ),
            0.65,
            ratco2 > jnp.float32(3.0),
        ),
    )
    adjusted_co2 = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac, chi_ref),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcolco2 = jnp.where(ratco2 > jnp.float32(3.0), adjusted_co2, colco2_arr)

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_add(
            selfref[inds0],
            _wrf_real_mul(
                selffrac_arr[:, None],
                _wrf_real_sub(selfref[inds1], selfref[inds0]),
            ),
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_add(
            forref[indf0],
            _wrf_real_mul(
                forfrac_arr[:, None],
                _wrf_real_sub(forref[indf1], forref[indf0]),
            ),
        ),
    )
    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    absco2_t = _wrf_real_add(
        ka_mco2[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(ka_mco2[indm1], ka_mco2[indm0]),
        ),
    )
    abso3_t = _wrf_real_add(
        ka_mo3[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(ka_mo3[indm1], ka_mo3[indm0]),
        ),
    )
    absn2o_t = _wrf_real_add(
        ka_mn2o[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(ka_mn2o[indm1], ka_mn2o[indm0]),
        ),
    )
    absco2_s = _wrf_real_add(
        kb_mco2[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(kb_mco2[indm1], kb_mco2[indm0]),
        ),
    )
    absn2o_s = _wrf_real_add(
        kb_mn2o[indm0],
        _wrf_real_mul(
            minorfrac_arr[:, None],
            _wrf_real_sub(kb_mn2o[indm1], kb_mn2o[indm0]),
        ),
    )
    cfc12_tau = _wrf_real_mul(wx_cfc12[:, None], cfc12[None, :])
    cfc22_tau = _wrf_real_mul(wx_cfc22[:, None], cfc22adj[None, :])

    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_add(
                        _wrf_real_add(
                            _wrf_real_add(
                                _wrf_real_mul(colh2o_arr[:, None], major_t),
                                tauself,
                            ),
                            taufor,
                        ),
                        _wrf_real_mul(adjcolco2[:, None], absco2_t),
                    ),
                    _wrf_real_mul(colo3_arr[:, None], abso3_t),
                ),
                _wrf_real_mul(coln2o_arr[:, None], absn2o_t),
            ),
            cfc12_tau,
        ),
        cfc22_tau,
    )
    taug_s = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_mul(colo3_arr[:, None], major_s),
                    _wrf_real_mul(adjcolco2[:, None], absco2_s),
                ),
                _wrf_real_mul(coln2o_arr[:, None], absn2o_s),
            ),
            cfc12_tau,
        ),
        cfc22_tau,
    )
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(
        trop_mask[:, None],
        jnp.broadcast_to(fracrefa[None, :], (n, int(fracrefa.shape[0]))),
        jnp.broadcast_to(fracrefb[None, :], (n, int(fracrefb.shape[0]))),
    )
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol9_lw(
    *,
    nlayers: int,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    coln2o: Any,
    colch4: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2och4: Any,
    rat_h2och4_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb9`` for band 9."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    coln2o_arr = jnp.asarray(coln2o, dtype=jnp.float32)[:n]
    colch4_arr = jnp.asarray(colch4, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2och4_arr = jnp.asarray(rat_h2och4, dtype=jnp.float32)[:n]
    rat_h2och4_1_arr = jnp.asarray(rat_h2och4_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    ka_mn2o = jnp.asarray(lookup.ka_mn2o, dtype=jnp.float32)
    kb_mn2o = jnp.asarray(lookup.kb_mn2o, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 8], chi_mls[5, 8])
    refrat_m_a = _wrf_real_div(chi_mls[0, 2], chi_mls[5, 2])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2och4_arr,
        colch4_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2och4_1_arr,
        colch4_arr,
        8.0,
        one_minus,
    )
    _, _, jmn2o_t, fmn2o_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a,
        colch4_arr,
        8.0,
        one_minus,
    )

    chi_n2o = _wrf_real_div(coln2o_arr, coldry_arr)
    chi_ref = chi_mls[3, jnp.clip(jp_arr, 0, chi_mls.shape[1] - 1)]
    ratn2o = _wrf_real_div(
        _wrf_real_mul(jnp.float32(1.0e20), chi_n2o),
        chi_ref,
    )
    adjfac = _wrf_real_add(
        jnp.float32(0.5),
        _wrf_real_active_pow(
            _wrf_real_sub(ratn2o, jnp.float32(0.5)),
            0.65,
            ratn2o > jnp.float32(1.5),
        ),
    )
    adjusted_n2o = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac, chi_ref),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcoln2o = jnp.where(ratn2o > jnp.float32(1.5), adjusted_n2o, coln2o_arr)

    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colch4_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mn2o.shape[1] - 2)
    indm1 = indm0 + 1
    jm0 = jnp.clip(jmn2o_t - 1, 0, ka_mn2o.shape[0] - 2)
    jm1 = jm0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    n2om1 = _wrf_real_interp(
        ka_mn2o[jm0, indm0],
        ka_mn2o[jm1, indm0],
        fmn2o_t[:, None],
    )
    n2om2 = _wrf_real_interp(
        ka_mn2o[jm0, indm1],
        ka_mn2o[jm1, indm1],
        fmn2o_t[:, None],
    )
    absn2o_t = _wrf_real_interp(
        n2om1,
        n2om2,
        minorfrac_arr[:, None],
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(tau_major_t, tau_major1_t),
                tauself,
            ),
            taufor,
        ),
        _wrf_real_mul(adjcoln2o[:, None], absn2o_t),
    )

    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    indm0_s = jnp.clip(indminor_arr - 1, 0, kb_mn2o.shape[0] - 2)
    indm1_s = indm0_s + 1
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    absn2o_s = _wrf_real_interp(
        kb_mn2o[indm0_s],
        kb_mn2o[indm1_s],
        minorfrac_arr[:, None],
    )
    taug_s = _wrf_real_add(
        _wrf_real_mul(colch4_arr[:, None], major_s),
        _wrf_real_mul(adjcoln2o[:, None], absn2o_s),
    )
    fracs_s = jnp.broadcast_to(fracrefb[None, :], (n, int(fracrefb.shape[0])))

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol10_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
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
    lookup: RrtmgLwCompressedBandData,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb10`` for band 10."""

    n = int(nlayers)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(colh2o_arr[:, None], major_t),
            tauself,
        ),
        taufor,
    )
    taug_s = _wrf_real_add(
        _wrf_real_mul(colh2o_arr[:, None], major_s),
        taufor,
    )
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(
        trop_mask[:, None],
        jnp.broadcast_to(fracrefa[None, :], (n, ng)),
        jnp.broadcast_to(fracrefb[None, :], (n, ng)),
    )
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol11_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colo2: Any,
    scaleminor: Any,
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
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb11`` for band 11."""

    n = int(nlayers)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colo2_arr = jnp.asarray(colo2, dtype=jnp.float32)[:n]
    scaleminor_arr = jnp.asarray(scaleminor, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    ka_mo2 = jnp.asarray(lookup.ka_mo2, dtype=jnp.float32)
    kb_mo2 = jnp.asarray(lookup.kb_mo2, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0_t = jnp.clip(indminor_arr - 1, 0, ka_mo2.shape[0] - 2)
    indm1_t = indm0_t + 1
    indm0_s = jnp.clip(indminor_arr - 1, 0, kb_mo2.shape[0] - 2)
    indm1_s = indm0_s + 1

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    scaleo2 = _wrf_real_mul(colo2_arr, scaleminor_arr)
    tauo2_t = _wrf_real_mul(
        scaleo2[:, None],
        _wrf_real_interp(
            ka_mo2[indm0_t],
            ka_mo2[indm1_t],
            minorfrac_arr[:, None],
        ),
    )
    tauo2_s = _wrf_real_mul(
        scaleo2[:, None],
        _wrf_real_interp(
            kb_mo2[indm0_s],
            kb_mo2[indm1_s],
            minorfrac_arr[:, None],
        ),
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_mul(colh2o_arr[:, None], major_t),
                tauself,
            ),
            taufor,
        ),
        tauo2_t,
    )
    taug_s = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(colh2o_arr[:, None], major_s),
            taufor,
        ),
        tauo2_s,
    )
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(
        trop_mask[:, None],
        jnp.broadcast_to(fracrefa[None, :], (n, ng)),
        jnp.broadcast_to(fracrefb[None, :], (n, ng)),
    )
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol12_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oco2: Any,
    rat_h2oco2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb12`` for band 12."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2oco2_arr = jnp.asarray(rat_h2oco2, dtype=jnp.float32)[:n]
    rat_h2oco2_1_arr = jnp.asarray(rat_h2oco2_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 9], chi_mls[1, 9])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2oco2_1_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(tau_major_t, tau_major1_t),
            tauself,
        ),
        taufor,
    )

    zeros = jnp.zeros((n, ng), dtype=jnp.float32)
    taug = jnp.where(trop_mask[:, None], taug_t, zeros)
    fracs = jnp.where(trop_mask[:, None], fracs_t, zeros)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol13_lw(
    *,
    nlayers: int,
    coldry: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    coln2o: Any,
    colco: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2on2o: Any,
    rat_h2on2o_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb13`` for band 13."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    coldry_arr = jnp.asarray(coldry, dtype=jnp.float32)[:n]
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colo3_arr = jnp.asarray(colo3, dtype=jnp.float32)[:n]
    coln2o_arr = jnp.asarray(coln2o, dtype=jnp.float32)[:n]
    colco_arr = jnp.asarray(colco, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2on2o_arr = jnp.asarray(rat_h2on2o, dtype=jnp.float32)[:n]
    rat_h2on2o_1_arr = jnp.asarray(rat_h2on2o_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    ka_mco2 = jnp.asarray(lookup.ka_mco2, dtype=jnp.float32)
    ka_mco = jnp.asarray(lookup.ka_mco, dtype=jnp.float32)
    kb_mo3 = jnp.asarray(lookup.kb_mo3, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 4], chi_mls[3, 4])
    refrat_m_a = _wrf_real_div(chi_mls[0, 0], chi_mls[3, 0])
    refrat_m_a3 = _wrf_real_div(chi_mls[0, 2], chi_mls[3, 2])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2on2o_arr,
        coln2o_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2on2o_1_arr,
        coln2o_arr,
        8.0,
        one_minus,
    )
    _, _, jmco2_t, fmco2_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a,
        coln2o_arr,
        8.0,
        one_minus,
    )

    chi_co2 = _wrf_real_div(colco2_arr, coldry_arr)
    co2_reference = jnp.float32(3.55e-4)
    ratco2 = _wrf_real_div(
        _wrf_real_mul(jnp.float32(1.0e20), chi_co2),
        co2_reference,
    )
    adjfac = _wrf_real_add(
        jnp.float32(2.0),
        _wrf_real_active_pow(
            _wrf_real_sub(ratco2, jnp.float32(2.0)),
            0.68,
            ratco2 > jnp.float32(3.0),
        ),
    )
    adjusted_co2 = _wrf_real_mul(
        _wrf_real_mul(
            _wrf_real_mul(adjfac, co2_reference),
            coldry_arr,
        ),
        jnp.float32(1.0e-20),
    )
    adjcolco2 = jnp.where(ratco2 > jnp.float32(3.0), adjusted_co2, colco2_arr)

    _, _, jmco_t, fmco_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_m_a3,
        coln2o_arr,
        8.0,
        one_minus,
    )

    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        coln2o_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mco2.shape[1] - 2)
    indm1 = indm0 + 1
    jmco20 = jnp.clip(jmco2_t - 1, 0, ka_mco2.shape[0] - 2)
    jmco21 = jmco20 + 1
    jmco0 = jnp.clip(jmco_t - 1, 0, ka_mco.shape[0] - 2)
    jmco1 = jmco0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    co2m1 = _wrf_real_interp(
        ka_mco2[jmco20, indm0],
        ka_mco2[jmco21, indm0],
        fmco2_t[:, None],
    )
    co2m2 = _wrf_real_interp(
        ka_mco2[jmco20, indm1],
        ka_mco2[jmco21, indm1],
        fmco2_t[:, None],
    )
    absco2_t = _wrf_real_interp(
        co2m1,
        co2m2,
        minorfrac_arr[:, None],
    )
    com1 = _wrf_real_interp(
        ka_mco[jmco0, indm0],
        ka_mco[jmco1, indm0],
        fmco_t[:, None],
    )
    com2 = _wrf_real_interp(
        ka_mco[jmco0, indm1],
        ka_mco[jmco1, indm1],
        fmco_t[:, None],
    )
    absco_t = _wrf_real_interp(
        com1,
        com2,
        minorfrac_arr[:, None],
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(
                    _wrf_real_add(tau_major_t, tau_major1_t),
                    tauself,
                ),
                taufor,
            ),
            _wrf_real_mul(adjcolco2[:, None], absco2_t),
        ),
        _wrf_real_mul(colco_arr[:, None], absco_t),
    )

    indm0_s = jnp.clip(indminor_arr - 1, 0, kb_mo3.shape[0] - 2)
    indm1_s = indm0_s + 1
    abso3_s = _wrf_real_interp(
        kb_mo3[indm0_s],
        kb_mo3[indm1_s],
        minorfrac_arr[:, None],
    )
    taug_s = _wrf_real_mul(colo3_arr[:, None], abso3_s)
    fracs_s = jnp.broadcast_to(fracrefb[None, :], (n, ng))

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol14_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colco2: Any,
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
    lookup: RrtmgLwCompressedBandData,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb14`` for band 14."""

    n = int(nlayers)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_5x(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    ind0_t = jnp.clip((jp_arr - 1) * 5 + (jt_arr - 1), 0, absa.shape[0] - 2)
    ind1_t = jnp.clip(jp_arr * 5 + (jt1_arr - 1), 0, absa.shape[0] - 2)
    ind0_s = jnp.clip((jp_arr - 13) * 5 + (jt_arr - 1), 0, absb.shape[0] - 2)
    ind1_s = jnp.clip((jp_arr - 12) * 5 + (jt1_arr - 1), 0, absb.shape[0] - 2)
    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1

    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    major_t = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absa[ind0_t],
            absa[ind0_t + 1],
            absa[ind1_t],
            absa[ind1_t + 1],
        ),
    )
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_mul(colco2_arr[:, None], major_t),
            tauself,
        ),
        taufor,
    )
    taug_s = _wrf_real_mul(colco2_arr[:, None], major_s)
    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(
        trop_mask[:, None],
        jnp.broadcast_to(fracrefa[None, :], (n, ng)),
        jnp.broadcast_to(fracrefb[None, :], (n, ng)),
    )
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol15_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    coln2o: Any,
    colco2: Any,
    colbrd: Any,
    scaleminor: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_n2oco2: Any,
    rat_n2oco2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    indminor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb15`` for band 15."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    coln2o_arr = jnp.asarray(coln2o, dtype=jnp.float32)[:n]
    colco2_arr = jnp.asarray(colco2, dtype=jnp.float32)[:n]
    colbrd_arr = jnp.asarray(colbrd, dtype=jnp.float32)[:n]
    scaleminor_arr = jnp.asarray(scaleminor, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_n2oco2_arr = jnp.asarray(rat_n2oco2, dtype=jnp.float32)[:n]
    rat_n2oco2_1_arr = jnp.asarray(rat_n2oco2_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]
    minorfrac_arr = jnp.asarray(minorfrac, dtype=jnp.float32)[:n]
    indminor_arr = jnp.asarray(indminor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    ka_mn2 = jnp.asarray(lookup.ka_mn2, dtype=jnp.float32)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[3, 0], chi_mls[1, 0])
    refrat_m_a = _wrf_real_div(chi_mls[3, 0], chi_mls[1, 0])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        coln2o_arr,
        rat_n2oco2_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        coln2o_arr,
        rat_n2oco2_1_arr,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jmn2_t, fmn2_t = _wrf_real_species_interpolation(
        coln2o_arr,
        refrat_m_a,
        colco2_arr,
        8.0,
        one_minus,
    )
    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        coln2o_arr,
        refrat_planck_a,
        colco2_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    indm0 = jnp.clip(indminor_arr - 1, 0, ka_mn2.shape[1] - 2)
    indm1 = indm0 + 1
    jm0 = jnp.clip(jmn2_t - 1, 0, ka_mn2.shape[0] - 2)
    jm1 = jm0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    n2m1 = _wrf_real_interp(
        ka_mn2[jm0, indm0],
        ka_mn2[jm1, indm0],
        fmn2_t[:, None],
    )
    n2m2 = _wrf_real_interp(
        ka_mn2[jm0, indm1],
        ka_mn2[jm1, indm1],
        fmn2_t[:, None],
    )
    scalen2 = _wrf_real_mul(colbrd_arr, scaleminor_arr)
    taun2 = _wrf_real_mul(
        scalen2[:, None],
        _wrf_real_interp(n2m1, n2m2, minorfrac_arr[:, None]),
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(
                _wrf_real_add(tau_major_t, tau_major1_t),
                tauself,
            ),
            taufor,
        ),
        taun2,
    )

    zeros = jnp.zeros((n, ng), dtype=jnp.float32)
    taug = jnp.where(trop_mask[:, None], taug_t, zeros)
    fracs = jnp.where(trop_mask[:, None], fracs_t, zeros)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol16_lw(
    *,
    nlayers: int,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colch4: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2och4: Any,
    rat_h2och4_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    lookup: RrtmgLwCompressedBandData,
    tables: RrtmgLwSetcoefTables | None = None,
    oneminus: float = 1.0 - 1.0e-6,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taugb16`` for band 16."""

    n = int(nlayers)
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    chi_mls = jnp.asarray(tab.chi_mls, dtype=jnp.float32)
    laytrop_arr = jnp.asarray(laytrop, dtype=jnp.int32)
    jp_arr = jnp.asarray(jp, dtype=jnp.int32)[:n]
    jt_arr = jnp.asarray(jt, dtype=jnp.int32)[:n]
    jt1_arr = jnp.asarray(jt1, dtype=jnp.int32)[:n]
    colh2o_arr = jnp.asarray(colh2o, dtype=jnp.float32)[:n]
    colch4_arr = jnp.asarray(colch4, dtype=jnp.float32)[:n]
    fac00_arr = jnp.asarray(fac00, dtype=jnp.float32)[:n]
    fac01_arr = jnp.asarray(fac01, dtype=jnp.float32)[:n]
    fac10_arr = jnp.asarray(fac10, dtype=jnp.float32)[:n]
    fac11_arr = jnp.asarray(fac11, dtype=jnp.float32)[:n]
    rat_h2och4_arr = jnp.asarray(rat_h2och4, dtype=jnp.float32)[:n]
    rat_h2och4_1_arr = jnp.asarray(rat_h2och4_1, dtype=jnp.float32)[:n]
    selffac_arr = jnp.asarray(selffac, dtype=jnp.float32)[:n]
    selffrac_arr = jnp.asarray(selffrac, dtype=jnp.float32)[:n]
    indself_arr = jnp.asarray(indself, dtype=jnp.int32)[:n]
    forfac_arr = jnp.asarray(forfac, dtype=jnp.float32)[:n]
    forfrac_arr = jnp.asarray(forfrac, dtype=jnp.float32)[:n]
    indfor_arr = jnp.asarray(indfor, dtype=jnp.int32)[:n]

    fracrefa = jnp.asarray(lookup.fracrefa, dtype=jnp.float32)
    fracrefb = jnp.asarray(lookup.fracrefb, dtype=jnp.float32)
    absa = _flatten_ref_table_fortran(lookup.ka)
    absb = _flatten_ref_table_5x(lookup.kb)
    selfref = jnp.asarray(lookup.selfref, dtype=jnp.float32)
    forref = jnp.asarray(lookup.forref, dtype=jnp.float32)
    ng = int(fracrefa.shape[0])
    one_minus = jnp.asarray(oneminus, dtype=jnp.float32)

    trop_mask = jnp.arange(n, dtype=jnp.int32) < laytrop_arr
    refrat_planck_a = _wrf_real_div(chi_mls[0, 5], chi_mls[5, 5])

    speccomb_t, specparm_t, js_t, fs_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2och4_arr,
        colch4_arr,
        8.0,
        one_minus,
    )
    speccomb1_t, specparm1_t, js1_t, fs1_t = _wrf_real_species_interpolation(
        colh2o_arr,
        rat_h2och4_1_arr,
        colch4_arr,
        8.0,
        one_minus,
    )
    _, _, jpl_t, fpl_t = _wrf_real_species_interpolation(
        colh2o_arr,
        refrat_planck_a,
        colch4_arr,
        8.0,
        one_minus,
    )

    ind0_t = (jp_arr - 1) * 45 + (jt_arr - 1) * 9 + (js_t - 1)
    ind1_t = jp_arr * 45 + (jt1_arr - 1) * 9 + (js1_t - 1)
    fac0_t = _edge_interp_factors(specparm_t, fs_t, fac00_arr, fac10_arr)
    fac1_t = _edge_interp_factors(specparm1_t, fs1_t, fac01_arr, fac11_arr)
    tau_major_t = _wrf_real_mul(
        speccomb_t[:, None],
        _edge_mixed_major(absa, ind0_t, specparm_t, fac0_t),
    )
    tau_major1_t = _wrf_real_mul(
        speccomb1_t[:, None],
        _edge_mixed_major(absa, ind1_t, specparm1_t, fac1_t),
    )

    inds0 = jnp.clip(indself_arr - 1, 0, selfref.shape[0] - 2)
    inds1 = inds0 + 1
    indf0 = jnp.clip(indfor_arr - 1, 0, forref.shape[0] - 2)
    indf1 = indf0 + 1
    tauself = _wrf_real_mul(
        selffac_arr[:, None],
        _wrf_real_interp(
            selfref[inds0],
            selfref[inds1],
            selffrac_arr[:, None],
        ),
    )
    taufor = _wrf_real_mul(
        forfac_arr[:, None],
        _wrf_real_interp(
            forref[indf0],
            forref[indf1],
            forfrac_arr[:, None],
        ),
    )
    jpl0_t = jnp.clip(jpl_t - 1, 0, fracrefa.shape[1] - 2)
    fracs_t = _wrf_real_interp(
        fracrefa[:, jpl0_t].T,
        fracrefa[:, jpl0_t + 1].T,
        fpl_t[:, None],
    )
    taug_t = _wrf_real_add(
        _wrf_real_add(
            _wrf_real_add(tau_major_t, tau_major1_t),
            tauself,
        ),
        taufor,
    )

    # WRF initializes nspb(16)=0, so both upper-atmosphere base indices
    # collapse to the first pair of band-16 absorption coefficients.
    ind0_s = jnp.zeros_like(jp_arr)
    ind1_s = jnp.zeros_like(jp_arr)
    major_s = _wrf_real_linear_combination(
        (
            fac00_arr[:, None],
            fac10_arr[:, None],
            fac01_arr[:, None],
            fac11_arr[:, None],
        ),
        (
            absb[ind0_s],
            absb[ind0_s + 1],
            absb[ind1_s],
            absb[ind1_s + 1],
        ),
    )
    taug_s = _wrf_real_mul(colch4_arr[:, None], major_s)
    fracs_s = jnp.broadcast_to(fracrefb[None, :], (n, ng))

    taug = jnp.where(trop_mask[:, None], taug_t, taug_s)
    fracs = jnp.where(trop_mask[:, None], fracs_t, fracs_s)
    return RrtmgLwTaumolBandResult(fracs=fracs, taug=taug)


def taumol_lw(
    *,
    nlayers: int,
    pavel: Any,
    coldry: Any,
    wx: Any,
    laytrop: Any,
    jp: Any,
    jt: Any,
    jt1: Any,
    colh2o: Any,
    colco2: Any,
    colo3: Any,
    coln2o: Any,
    colco: Any,
    colch4: Any,
    colo2: Any,
    colbrd: Any,
    fac00: Any,
    fac01: Any,
    fac10: Any,
    fac11: Any,
    rat_h2oco2: Any,
    rat_h2oco2_1: Any,
    rat_h2oo3: Any,
    rat_h2oo3_1: Any,
    rat_h2on2o: Any,
    rat_h2on2o_1: Any,
    rat_h2och4: Any,
    rat_h2och4_1: Any,
    rat_n2oco2: Any,
    rat_n2oco2_1: Any,
    rat_o3co2: Any,
    rat_o3co2_1: Any,
    selffac: Any,
    selffrac: Any,
    indself: Any,
    forfac: Any,
    forfrac: Any,
    indfor: Any,
    minorfrac: Any,
    scaleminor: Any,
    scaleminorn2: Any,
    indminor: Any,
    lookup_tables: RrtmgLwCompressedLookupTables | None = None,
    tables: RrtmgLwSetcoefTables | None = None,
) -> RrtmgLwTaumolBandResult:
    """WRF longwave ``taumol`` assembled over all 140 compressed g-points."""

    lookup = load_rrtmg_lw_compressed_lookup_tables() if lookup_tables is None else lookup_tables
    tab = load_rrtmg_lw_setcoef_tables() if tables is None else tables
    bands = (
        taumol1_lw(
            nlayers=nlayers,
            pavel=pavel,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colbrd=colbrd,
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
            minorfrac=minorfrac,
            scaleminorn2=scaleminorn2,
            indminor=indminor,
            lookup=lookup.band1,
        ),
        taumol2_lw(
            nlayers=nlayers,
            pavel=pavel,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
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
            lookup=lookup.band2,
        ),
        taumol3_lw(
            nlayers=nlayers,
            pavel=pavel,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            coln2o=coln2o,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2oco2=rat_h2oco2,
            rat_h2oco2_1=rat_h2oco2_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band3,
            tables=tab,
        ),
        taumol4_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            colo3=colo3,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2oco2=rat_h2oco2,
            rat_h2oco2_1=rat_h2oco2_1,
            rat_o3co2=rat_o3co2,
            rat_o3co2_1=rat_o3co2_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            lookup=lookup.band4,
            tables=tab,
        ),
        taumol5_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            colo3=colo3,
            wx=wx,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2oco2=rat_h2oco2,
            rat_h2oco2_1=rat_h2oco2_1,
            rat_o3co2=rat_o3co2,
            rat_o3co2_1=rat_o3co2_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band5,
            tables=tab,
        ),
        taumol6_lw(
            nlayers=nlayers,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            wx=wx,
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
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band6,
            tables=tab,
        ),
        taumol7_lw(
            nlayers=nlayers,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            colo3=colo3,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2oo3=rat_h2oo3,
            rat_h2oo3_1=rat_h2oo3_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band7,
            tables=tab,
        ),
        taumol8_lw(
            nlayers=nlayers,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            colo3=colo3,
            coln2o=coln2o,
            wx=wx,
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
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band8,
            tables=tab,
        ),
        taumol9_lw(
            nlayers=nlayers,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            coln2o=coln2o,
            colch4=colch4,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2och4=rat_h2och4,
            rat_h2och4_1=rat_h2och4_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band9,
            tables=tab,
        ),
        taumol10_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
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
            lookup=lookup.band10,
        ),
        taumol11_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colo2=colo2,
            scaleminor=scaleminor,
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
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band11,
        ),
        taumol12_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2oco2=rat_h2oco2,
            rat_h2oco2_1=rat_h2oco2_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            lookup=lookup.band12,
            tables=tab,
        ),
        taumol13_lw(
            nlayers=nlayers,
            coldry=coldry,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colco2=colco2,
            colo3=colo3,
            coln2o=coln2o,
            colco=colco,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2on2o=rat_h2on2o,
            rat_h2on2o_1=rat_h2on2o_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band13,
            tables=tab,
        ),
        taumol14_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colco2=colco2,
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
            lookup=lookup.band14,
        ),
        taumol15_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            coln2o=coln2o,
            colco2=colco2,
            colbrd=colbrd,
            scaleminor=scaleminor,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_n2oco2=rat_n2oco2,
            rat_n2oco2_1=rat_n2oco2_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            minorfrac=minorfrac,
            indminor=indminor,
            lookup=lookup.band15,
            tables=tab,
        ),
        taumol16_lw(
            nlayers=nlayers,
            laytrop=laytrop,
            jp=jp,
            jt=jt,
            jt1=jt1,
            colh2o=colh2o,
            colch4=colch4,
            fac00=fac00,
            fac01=fac01,
            fac10=fac10,
            fac11=fac11,
            rat_h2och4=rat_h2och4,
            rat_h2och4_1=rat_h2och4_1,
            selffac=selffac,
            selffrac=selffrac,
            indself=indself,
            forfac=forfac,
            forfrac=forfrac,
            indfor=indfor,
            lookup=lookup.band16,
            tables=tab,
        ),
    )
    return RrtmgLwTaumolBandResult(
        fracs=jnp.concatenate(tuple(band.fracs for band in bands), axis=1),
        taug=jnp.concatenate(tuple(band.taug for band in bands), axis=1),
    )


def _rtrnmc_table_index(odepth: Any, pade: RrtmgLwPadeTables) -> Any:
    tblint = jnp.asarray(pade.tau_tbl.shape[0] - 1, dtype=jnp.float32)
    tblind = _wrf_real_div(
        odepth,
        _wrf_real_add(pade.bpade, odepth),
    )
    table_position = _wrf_real_add(
        _wrf_real_mul(tblint, tblind),
        jnp.float32(0.5),
    )
    return jnp.clip(
        table_position.astype(jnp.int32),
        0,
        pade.tau_tbl.shape[0] - 1,
    )


def _rtrnmc_clear_terms(odepth: Any, plfrac: Any, blay: Any, dplankdn: Any, dplankup: Any, pade: RrtmgLwPadeTables) -> tuple[Any, Any, Any]:
    small = odepth <= jnp.float32(0.06)
    odepth_rec = _wrf_real_mul(jnp.float32(0.166667), odepth)
    atrans_small = _wrf_real_sub(
        odepth,
        _wrf_real_mul(
            _wrf_real_mul(jnp.float32(0.5), odepth),
            odepth,
        ),
    )
    bbd_small = _wrf_real_mul(
        plfrac,
        _wrf_real_add(
            blay,
            _wrf_real_mul(dplankdn, odepth_rec),
        ),
    )
    bbugas_small = _wrf_real_mul(
        plfrac,
        _wrf_real_add(
            blay,
            _wrf_real_mul(dplankup, odepth_rec),
        ),
    )

    itr = _rtrnmc_table_index(odepth, pade)
    transc = pade_transmittance(odepth, pade.exp_tbl[itr])
    atrans_table = _wrf_real_sub(jnp.float32(1.0), transc)
    tausfac = pade_source_factor(odepth, pade.tfn_tbl[itr])
    bbd_table = _wrf_real_mul(
        plfrac,
        _wrf_real_add(
            blay,
            _wrf_real_mul(tausfac, dplankdn),
        ),
    )
    bbugas_table = _wrf_real_mul(
        plfrac,
        _wrf_real_add(
            blay,
            _wrf_real_mul(tausfac, dplankup),
        ),
    )
    return (
        jnp.where(small, atrans_small, atrans_table),
        jnp.where(small, bbd_small, bbd_table),
        jnp.where(small, bbugas_small, bbugas_table),
    )


def _rtrnmc_cloudy_terms(
    odepth: Any,
    odcld: Any,
    plfrac: Any,
    blay: Any,
    dplankdn: Any,
    dplankup: Any,
    pade: RrtmgLwPadeTables,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    rec_6 = jnp.asarray(0.166667, dtype=jnp.float32)
    odtot_raw = odepth + odcld

    odepth_rec1 = rec_6 * odepth
    odtot_rec1 = rec_6 * odtot_raw
    atrans1 = odepth - 0.5 * odepth * odepth
    gassrc1 = plfrac * (blay + dplankdn * odepth_rec1) * atrans1
    atot1 = odtot_raw - 0.5 * odtot_raw * odtot_raw
    bbdtot1 = plfrac * (blay + dplankdn * odtot_rec1)
    bbdgas1 = plfrac * (blay + dplankdn * odepth_rec1)
    bbugas1 = plfrac * (blay + dplankup * odepth_rec1)
    bbutot1 = plfrac * (blay + dplankup * odtot_rec1)

    ittot2 = _rtrnmc_table_index(odtot_raw, pade)
    tfactot2 = pade_source_factor(odtot_raw, pade.tfn_tbl[ittot2])
    atrans2 = atrans1
    gassrc2 = gassrc1
    atot2 = 1.0 - pade_transmittance(odtot_raw, pade.exp_tbl[ittot2])
    bbdtot2 = plfrac * (blay + tfactot2 * dplankdn)
    bbdgas2 = bbdgas1
    bbugas2 = bbugas1
    bbutot2 = plfrac * (blay + tfactot2 * dplankup)

    itgas3 = _rtrnmc_table_index(odepth, pade)
    odepth_tbl3 = pade_optical_depth(odepth, pade.tau_tbl[itgas3])
    atrans3 = 1.0 - pade_transmittance(odepth, pade.exp_tbl[itgas3])
    tfacgas3 = pade_source_factor(odepth, pade.tfn_tbl[itgas3])
    gassrc3 = atrans3 * plfrac * (blay + tfacgas3 * dplankdn)
    odtot3 = odepth_tbl3 + odcld
    ittot3 = _rtrnmc_table_index(odtot3, pade)
    tfactot3 = pade_source_factor(odtot3, pade.tfn_tbl[ittot3])
    atot3 = 1.0 - pade_transmittance(odtot3, pade.exp_tbl[ittot3])
    bbdtot3 = plfrac * (blay + tfactot3 * dplankdn)
    bbdgas3 = plfrac * (blay + tfacgas3 * dplankdn)
    bbugas3 = plfrac * (blay + tfacgas3 * dplankup)
    bbutot3 = plfrac * (blay + tfactot3 * dplankup)

    first = odtot_raw < 0.06
    second = odepth <= 0.06
    atrans = jnp.where(first, atrans1, jnp.where(second, atrans2, atrans3))
    gassrc = jnp.where(first, gassrc1, jnp.where(second, gassrc2, gassrc3))
    atot = jnp.where(first, atot1, jnp.where(second, atot2, atot3))
    bbdtot = jnp.where(first, bbdtot1, jnp.where(second, bbdtot2, bbdtot3))
    bbdgas = jnp.where(first, bbdgas1, jnp.where(second, bbdgas2, bbdgas3))
    bbugas = jnp.where(first, bbugas1, jnp.where(second, bbugas2, bbugas3))
    bbutot = jnp.where(first, bbutot1, jnp.where(second, bbutot2, bbutot3))
    return atrans, gassrc, atot, bbdtot, bbdgas, bbugas, bbutot


def rtrnmc_lw(
    *,
    nlayers: int,
    istart: int,
    iend: int,
    iout: int,
    pz: Any,
    semiss: Any,
    ncbands: int,
    cldfmc: Any,
    taucmc: Any,
    planklay: Any,
    planklev: Any,
    plankbnd: Any,
    pwvcm: Any,
    fracs: Any,
    taut: Any,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgLwPadeTables | None = None,
) -> RrtmgLwRtrnmcResult:
    """WRF longwave ``rtrnmc`` for one column."""

    if int(iout) != 0:
        raise NotImplementedError("WRF rtrnmc_lw per-band iout path is inactive and not translated")
    del ncbands
    n = int(nlayers)
    spec = lwdatinit() if spectral is None else spectral
    pade = rrtmg_lw_pade_tables() if pade_tables is None else pade_tables
    pz_arr = jnp.asarray(pz, dtype=jnp.float32)[: n + 1]
    semiss_arr = jnp.asarray(semiss, dtype=jnp.float32)[:RRTMG_LW_NBND]
    cldf = jnp.asarray(cldfmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    taucm = jnp.asarray(taucmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    planklay_arr = jnp.asarray(planklay, dtype=jnp.float32)[:n, :RRTMG_LW_NBND]
    planklev_arr = jnp.asarray(planklev, dtype=jnp.float32)[: n + 1, :RRTMG_LW_NBND]
    plankbnd_arr = jnp.asarray(plankbnd, dtype=jnp.float32)[:RRTMG_LW_NBND]
    fracs_arr = jnp.asarray(fracs, dtype=jnp.float32)[:n, :RRTMG_LW_NGPT]
    taut_arr = jnp.asarray(taut, dtype=jnp.float32)[:n, :RRTMG_LW_NGPT]

    a0 = jnp.asarray(
        (1.66, 1.55, 1.58, 1.66, 1.54, 1.454, 1.89, 1.33, 1.668, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66),
        dtype=jnp.float32,
    )
    a1 = jnp.asarray(
        (0.0, 0.25, 0.22, 0.0, 0.13, 0.446, -0.10, 0.40, -0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dtype=jnp.float32,
    )
    a2 = jnp.asarray(
        (0.0, -12.0, -11.7, 0.0, -0.72, -0.243, 0.19, -0.062, 0.414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dtype=jnp.float32,
    )
    band_idx = jnp.arange(RRTMG_LW_NBND)
    variable_diffusivity = ~((band_idx == 0) | (band_idx == 3) | (band_idx >= 9))
    secdiff_var = jnp.clip(
        a0
        + a1
        * wrf_glibc_expf_finite(
            a2 * jnp.asarray(pwvcm, dtype=jnp.float32)
        ),
        1.50,
        1.80,
    )
    secdiff = jnp.where(variable_diffusivity, secdiff_var, 1.66)

    ngb0 = jnp.asarray(NGB_LW, dtype=jnp.int32) - 1
    odcld = jnp.zeros((n, RRTMG_LW_NGPT), dtype=jnp.float32)
    abscld = jnp.zeros((n, RRTMG_LW_NGPT), dtype=jnp.float32)
    efclfrac = jnp.zeros((n, RRTMG_LW_NGPT), dtype=jnp.float32)
    for lay in range(n):
        for ig in range(RRTMG_LW_NGPT):
            cloud_active = cldf[ig, lay] == 1.0
            ib = ngb0[ig]
            od = secdiff[ib] * taucm[ig, lay]
            transcld = jnp.exp(-od)
            absval = 1.0 - transcld
            odcld = odcld.at[lay, ig].set(jnp.where(cloud_active, od, 0.0))
            abscld = abscld.at[lay, ig].set(jnp.where(cloud_active, absval, 0.0))
            efclfrac = efclfrac.at[lay, ig].set(jnp.where(cloud_active, absval * cldf[ig, lay], 0.0))
    icldlyr = jnp.max((cldf == 1.0).astype(jnp.int32), axis=0)

    totuflux = jnp.zeros((n + 1,), dtype=jnp.float32)
    totdflux = jnp.zeros((n + 1,), dtype=jnp.float32)
    totuclfl = jnp.zeros((n + 1,), dtype=jnp.float32)
    totdclfl = jnp.zeros((n + 1,), dtype=jnp.float32)

    ngs = tuple(int(value) for value in np.asarray(spec.ngs))
    delwave = jnp.asarray(spec.delwave, dtype=jnp.float32)
    for iband in range(int(istart) - 1, int(iend)):
        start = 0 if iband == 0 else ngs[iband - 1]
        end = ngs[iband]
        urad = jnp.zeros((n + 1,), dtype=jnp.float32)
        drad = jnp.zeros((n + 1,), dtype=jnp.float32)
        clrurad = jnp.zeros((n + 1,), dtype=jnp.float32)
        clrdrad = jnp.zeros((n + 1,), dtype=jnp.float32)
        for igc in range(start, end):
            radld = jnp.asarray(0.0, dtype=jnp.float32)
            radclrd = jnp.asarray(0.0, dtype=jnp.float32)
            iclddn = jnp.asarray(0, dtype=jnp.int32)
            atrans = jnp.zeros((n,), dtype=jnp.float32)
            bbugas = jnp.zeros((n,), dtype=jnp.float32)
            bbutot = jnp.zeros((n,), dtype=jnp.float32)
            atot = jnp.zeros((n,), dtype=jnp.float32)

            for lev in range(n - 1, -1, -1):
                plfrac = fracs_arr[lev, igc]
                blay = planklay_arr[lev, iband]
                dplankup = planklev_arr[lev + 1, iband] - blay
                dplankdn = planklev_arr[lev, iband] - blay
                odepth = jnp.maximum(secdiff[iband] * taut_arr[lev, igc], 0.0)
                cloudy_layer = icldlyr[lev] == 1
                iclddn = jnp.maximum(iclddn, cloudy_layer.astype(jnp.int32))

                a_cloud, gas_cloud, atot_cloud, bbdtot_cloud, bbdgas_cloud, bbugas_cloud, bbutot_cloud = _rtrnmc_cloudy_terms(
                    odepth,
                    odcld[lev, igc],
                    plfrac,
                    blay,
                    dplankdn,
                    dplankup,
                    pade,
                )
                radld_cloud = (
                    radld
                    - radld * (a_cloud + efclfrac[lev, igc] * (1.0 - a_cloud))
                    + gas_cloud
                    + cldf[igc, lev] * (bbdtot_cloud * atot_cloud - gas_cloud)
                )

                a_clear, bbd_clear, bbugas_clear = _rtrnmc_clear_terms(odepth, plfrac, blay, dplankdn, dplankup, pade)
                radld_clear = radld + (bbd_clear - radld) * a_clear

                a_new = jnp.where(cloudy_layer, a_cloud, a_clear)
                bbugas_new = jnp.where(cloudy_layer, bbugas_cloud, bbugas_clear)
                bbutot_new = jnp.where(cloudy_layer, bbutot_cloud, 0.0)
                atot_new = jnp.where(cloudy_layer, atot_cloud, 0.0)
                bbd_for_clear = jnp.where(cloudy_layer, bbdgas_cloud, bbd_clear)
                radld_new = jnp.where(cloudy_layer, radld_cloud, radld_clear)

                atrans = atrans.at[lev].set(a_new)
                bbugas = bbugas.at[lev].set(bbugas_new)
                bbutot = bbutot.at[lev].set(bbutot_new)
                atot = atot.at[lev].set(atot_new)
                drad = drad.at[lev].add(radld_new)

                sep_clear = iclddn == 1
                radclrd_sep = radclrd + (bbd_for_clear - radclrd) * a_new
                clrdrad_sep = clrdrad[lev] + radclrd_sep
                clrdrad_same = drad[lev]
                radclrd = jnp.where(sep_clear, radclrd_sep, radld_new)
                clrdrad = clrdrad.at[lev].set(jnp.where(sep_clear, clrdrad_sep, clrdrad_same))
                radld = radld_new

            rad0 = fracs_arr[0, igc] * plankbnd_arr[iband]
            reflect = 1.0 - semiss_arr[iband]
            radlu = rad0 + reflect * radld
            radclru = rad0 + reflect * radclrd
            urad = urad.at[0].add(radlu)
            clrurad = clrurad.at[0].add(radclru)
            has_cloud = iclddn == 1
            for lev in range(n):
                cloudy_layer = icldlyr[lev] == 1
                gas_source = bbugas[lev] * atrans[lev]
                radlu_cloud = (
                    radlu
                    - radlu * (atrans[lev] + efclfrac[lev, igc] * (1.0 - atrans[lev]))
                    + gas_source
                    + cldf[igc, lev] * (bbutot[lev] * atot[lev] - gas_source)
                )
                radlu_clear = radlu + (bbugas[lev] - radlu) * atrans[lev]
                radlu = jnp.where(cloudy_layer, radlu_cloud, radlu_clear)
                urad = urad.at[lev + 1].add(radlu)

                radclru_sep = radclru + (bbugas[lev] - radclru) * atrans[lev]
                clrurad_sep = clrurad[lev + 1] + radclru_sep
                clrurad_same = urad[lev + 1]
                radclru = jnp.where(has_cloud, radclru_sep, radlu)
                clrurad = clrurad.at[lev + 1].set(jnp.where(has_cloud, clrurad_sep, clrurad_same))

        uflux = urad * 0.5
        dflux = drad * 0.5
        uclfl = clrurad * 0.5
        dclfl = clrdrad * 0.5
        totuflux = totuflux + uflux * delwave[iband]
        totdflux = totdflux + dflux * delwave[iband]
        totuclfl = totuclfl + uclfl * delwave[iband]
        totdclfl = totdclfl + dclfl * delwave[iband]

    fluxfac = jnp.asarray(spec.fluxfac, dtype=jnp.float32)
    heatfac = jnp.asarray(spec.heatfac, dtype=jnp.float32)
    totuflux = totuflux * fluxfac
    totdflux = totdflux * fluxfac
    totuclfl = totuclfl * fluxfac
    totdclfl = totdclfl * fluxfac
    fnet = totuflux - totdflux
    fnetc = totuclfl - totdclfl
    htr = jnp.zeros((n + 1,), dtype=jnp.float32)
    htrc = jnp.zeros((n + 1,), dtype=jnp.float32)
    for lev in range(1, n + 1):
        htr = htr.at[lev - 1].set(heatfac * (fnet[lev - 1] - fnet[lev]) / (pz_arr[lev - 1] - pz_arr[lev]))
        htrc = htrc.at[lev - 1].set(heatfac * (fnetc[lev - 1] - fnetc[lev]) / (pz_arr[lev - 1] - pz_arr[lev]))
    return RrtmgLwRtrnmcResult(
        totuflux=totuflux,
        totdflux=totdflux,
        fnet=fnet,
        htr=htr,
        totuclfl=totuclfl,
        totdclfl=totdclfl,
        fnetc=fnetc,
        htrc=htrc,
    )


def rtrnmc_lw_columns(
    *,
    nlayers: int,
    istart: int,
    iend: int,
    iout: int,
    pz: Any,
    semiss: Any,
    ncbands: int,
    cldfmc: Any,
    taucmc: Any,
    planklay: Any,
    planklev: Any,
    plankbnd: Any,
    pwvcm: Any,
    fracs: Any,
    taut: Any,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgLwPadeTables | None = None,
) -> RrtmgLwRtrnmcResult:
    """Batch WRF longwave ``rtrnmc`` over columns and g-points."""

    if int(iout) != 0:
        raise NotImplementedError("WRF rtrnmc_lw per-band iout path is inactive and not translated")
    if int(istart) != 1 or int(iend) != RRTMG_LW_NBND:
        raise NotImplementedError("Batched rtrnmc_lw_columns currently covers the active full-band WRF path")
    del ncbands

    n = int(nlayers)
    spec = lwdatinit() if spectral is None else spectral
    pade = rrtmg_lw_pade_tables() if pade_tables is None else pade_tables
    pade = pade._replace(bpade=jnp.asarray(pade.bpade).reshape(()))

    pz_arr = jnp.asarray(pz, dtype=jnp.float32)[:, : n + 1]
    semiss_arr = jnp.asarray(semiss, dtype=jnp.float32)[:, :RRTMG_LW_NBND]
    cldf = jnp.asarray(cldfmc, dtype=jnp.float32)[:, :RRTMG_LW_NGPT, :n]
    taucm = jnp.asarray(taucmc, dtype=jnp.float32)[:, :RRTMG_LW_NGPT, :n]
    planklay_arr = jnp.asarray(planklay, dtype=jnp.float32)[:, :n, :RRTMG_LW_NBND]
    planklev_arr = jnp.asarray(planklev, dtype=jnp.float32)[:, : n + 1, :RRTMG_LW_NBND]
    plankbnd_arr = jnp.asarray(plankbnd, dtype=jnp.float32)[:, :RRTMG_LW_NBND]
    fracs_arr = jnp.asarray(fracs, dtype=jnp.float32)[:, :n, :RRTMG_LW_NGPT]
    taut_arr = jnp.asarray(taut, dtype=jnp.float32)[:, :n, :RRTMG_LW_NGPT]

    a0 = jnp.asarray(
        (1.66, 1.55, 1.58, 1.66, 1.54, 1.454, 1.89, 1.33, 1.668, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66),
        dtype=jnp.float32,
    )
    a1 = jnp.asarray(
        (0.0, 0.25, 0.22, 0.0, 0.13, 0.446, -0.10, 0.40, -0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dtype=jnp.float32,
    )
    a2 = jnp.asarray(
        (0.0, -12.0, -11.7, 0.0, -0.72, -0.243, 0.19, -0.062, 0.414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dtype=jnp.float32,
    )
    band_idx = jnp.arange(RRTMG_LW_NBND)
    variable_diffusivity = ~((band_idx == 0) | (band_idx == 3) | (band_idx >= 9))
    secdiff_var = jnp.clip(
        _wrf_real_add(
            a0[None, :],
            _wrf_real_mul(
                a1[None, :],
                _wrf_real_store(
                    wrf_glibc_expf_finite(
                        _wrf_real_mul(
                            a2[None, :],
                            jnp.asarray(pwvcm, dtype=jnp.float32)[:, None],
                        ),
                    )
                ),
            ),
        ),
        jnp.float32(1.50),
        jnp.float32(1.80),
    )
    secdiff = jnp.where(variable_diffusivity[None, :], secdiff_var, jnp.asarray(1.66, dtype=jnp.float32))

    ngb0 = jnp.asarray(NGB_LW, dtype=jnp.int32)[:RRTMG_LW_NGPT] - 1
    secdiff_g = jnp.take(secdiff, ngb0, axis=1)
    semiss_g = jnp.take(semiss_arr, ngb0, axis=1)
    planklay_g = jnp.take(planklay_arr, ngb0, axis=2)
    planklev_g = jnp.take(planklev_arr, ngb0, axis=2)
    plankbnd_g = jnp.take(plankbnd_arr, ngb0, axis=1)

    cldf_lg = jnp.moveaxis(cldf, 2, 1)
    taucm_lg = jnp.moveaxis(taucm, 2, 1)
    cloud_active = cldf_lg == 1.0
    odcld_raw = secdiff_g[:, None, :] * taucm_lg
    abscld = 1.0 - jnp.exp(-odcld_raw)
    odcld = jnp.where(cloud_active, odcld_raw, 0.0)
    efclfrac = jnp.where(cloud_active, abscld * cldf_lg, 0.0)
    icldlyr = jnp.max((cldf_lg == 1.0).astype(jnp.int32), axis=2)

    dplankup = _wrf_real_sub(planklev_g[:, 1 : n + 1, :], planklay_g)
    dplankdn = _wrf_real_sub(planklev_g[:, :n, :], planklay_g)
    odepth = jnp.maximum(
        _wrf_real_mul(secdiff_g[:, None, :], taut_arr),
        jnp.float32(0.0),
    )

    cng = cldf_lg.shape[0], RRTMG_LW_NGPT
    zeros_g = jnp.zeros(cng, dtype=jnp.float32)
    cloud_layers = jnp.broadcast_to(icldlyr[:, :, None], cldf_lg.shape).astype(jnp.float32)

    def down_step(carry: tuple[Any, ...], inputs: tuple[Any, ...], pade: RrtmgLwPadeTables):
        radld, radclrd, iclddn, *_previous_optics = carry
        plfrac, blay, dplankdn_i, dplankup_i, odepth_i, cloudy_i, odcld_i, efclfrac_i, cldf_i = inputs
        cloudy_layer = cloudy_i == 1
        iclddn_new = jnp.maximum(iclddn, cloudy_layer.astype(jnp.float32))

        a_cloud, gas_cloud, atot_cloud, bbdtot_cloud, bbdgas_cloud, bbugas_cloud, bbutot_cloud = _rtrnmc_cloudy_terms(
            odepth_i,
            odcld_i,
            plfrac,
            blay,
            dplankdn_i,
            dplankup_i,
            pade,
        )
        radld_cloud = (
            radld
            - radld * (a_cloud + efclfrac_i * (1.0 - a_cloud))
            + gas_cloud
            + cldf_i * (bbdtot_cloud * atot_cloud - gas_cloud)
        )

        a_clear, bbd_clear, bbugas_clear = _rtrnmc_clear_terms(odepth_i, plfrac, blay, dplankdn_i, dplankup_i, pade)
        radld_clear = _wrf_real_add(
            radld,
            _wrf_real_mul(_wrf_real_sub(bbd_clear, radld), a_clear),
        )

        a_new = jnp.where(cloudy_layer, a_cloud, a_clear)
        bbugas_new = jnp.where(cloudy_layer, bbugas_cloud, bbugas_clear)
        bbutot_new = jnp.where(cloudy_layer, bbutot_cloud, 0.0)
        atot_new = jnp.where(cloudy_layer, atot_cloud, 0.0)
        bbd_for_clear = jnp.where(cloudy_layer, bbdgas_cloud, bbd_clear)
        radld_new = jnp.where(cloudy_layer, radld_cloud, radld_clear)

        sep_clear = iclddn_new == 1
        radclrd_sep = _wrf_real_add(
            radclrd,
            _wrf_real_mul(_wrf_real_sub(bbd_for_clear, radclrd), a_new),
        )
        radclrd_new = jnp.where(sep_clear, radclrd_sep, radld_new)
        values = (radld_new, radclrd_new, iclddn_new, a_new, bbugas_new, bbutot_new, atot_new)
        return values, values

    down_inputs = (
        jnp.moveaxis(fracs_arr, 1, 0),
        jnp.moveaxis(planklay_g, 1, 0),
        jnp.moveaxis(dplankdn, 1, 0),
        jnp.moveaxis(dplankup, 1, 0),
        jnp.moveaxis(odepth, 1, 0),
        jnp.moveaxis(cloud_layers, 1, 0),
        jnp.moveaxis(odcld, 1, 0),
        jnp.moveaxis(efclfrac, 1, 0),
        jnp.moveaxis(cldf_lg, 1, 0),
    )
    # Optical tables stay read-only; each column keeps its dependent radiance
    # sweep local while emitting coefficients needed by the upward sweep.
    (radld_surface, radclrd_surface, has_cloud, *_last_optics), down_outputs = scan_columns(
        down_step,
        (zeros_g,) * 7,
        down_inputs,
        pade,
        reverse=True,
    )
    (down_rad, down_clear, _cloud_history, atrans_by_level,
     bbugas_by_level, bbutot_by_level, atot_by_level) = down_outputs
    drad_lg = jnp.moveaxis(down_rad, 0, 1)
    clrdrad_lg = jnp.moveaxis(down_clear, 0, 1)
    atrans = jnp.moveaxis(atrans_by_level, 0, 1)
    bbugas = jnp.moveaxis(bbugas_by_level, 0, 1)
    bbutot = jnp.moveaxis(bbutot_by_level, 0, 1)
    atot = jnp.moveaxis(atot_by_level, 0, 1)

    rad0 = _wrf_real_mul(fracs_arr[:, 0, :], plankbnd_g)
    reflect = _wrf_real_sub(jnp.float32(1.0), semiss_g)
    radlu0 = _wrf_real_add(rad0, _wrf_real_mul(reflect, radld_surface))
    radclru0 = _wrf_real_add(rad0, _wrf_real_mul(reflect, radclrd_surface))
    has_cloud_bool = has_cloud == 1

    def up_step(carry: tuple[Any, Any], inputs: tuple[Any, ...], _parameter: Any):
        radlu, radclru = carry
        a_i, bbugas_i, bbutot_i, atot_i, efclfrac_i, cldf_i, cloudy_i, has_cloud_i = inputs
        cloudy_layer = cloudy_i == 1
        gas_source = bbugas_i * a_i
        radlu_cloud = (
            radlu
            - radlu * (a_i + efclfrac_i * (1.0 - a_i))
            + gas_source
            + cldf_i * (bbutot_i * atot_i - gas_source)
        )
        radlu_clear = _wrf_real_add(
            radlu,
            _wrf_real_mul(_wrf_real_sub(bbugas_i, radlu), a_i),
        )
        radlu_new = jnp.where(cloudy_layer, radlu_cloud, radlu_clear)

        radclru_sep = _wrf_real_add(
            radclru,
            _wrf_real_mul(_wrf_real_sub(bbugas_i, radclru), a_i),
        )
        radclru_new = jnp.where(has_cloud_i == 1, radclru_sep, radlu_new)
        return (radlu_new, radclru_new), (radlu_new, radclru_new)

    up_inputs = (
        jnp.moveaxis(atrans, 1, 0),
        jnp.moveaxis(bbugas, 1, 0),
        jnp.moveaxis(bbutot, 1, 0),
        jnp.moveaxis(atot, 1, 0),
        jnp.moveaxis(efclfrac, 1, 0),
        jnp.moveaxis(cldf_lg, 1, 0),
        jnp.moveaxis(cloud_layers, 1, 0),
        jnp.moveaxis(jnp.broadcast_to(has_cloud_bool[:, None, :], cldf_lg.shape).astype(jnp.float32), 1, 0),
    )
    _up_carry, up_outputs = scan_columns(up_step, (radlu0, radclru0), up_inputs)
    up_rad, up_clear = up_outputs
    urad_lg = jnp.concatenate((radlu0[:, None, :], jnp.moveaxis(up_rad, 0, 1)), axis=1)
    clrurad_lg = jnp.concatenate((radclru0[:, None, :], jnp.moveaxis(up_clear, 0, 1)), axis=1)
    drad_full = jnp.concatenate((drad_lg, jnp.zeros((drad_lg.shape[0], 1, drad_lg.shape[2]), dtype=jnp.float32)), axis=1)
    clrdrad_full = jnp.concatenate((clrdrad_lg, jnp.zeros((clrdrad_lg.shape[0], 1, clrdrad_lg.shape[2]), dtype=jnp.float32)), axis=1)

    fluxfac = jnp.asarray(spec.fluxfac, dtype=jnp.float32)
    heatfac = jnp.asarray(spec.heatfac, dtype=jnp.float32)
    totuflux = _wrf_lw_broadband_flux(urad_lg, spec.delwave, fluxfac)
    totdflux = _wrf_lw_broadband_flux(drad_full, spec.delwave, fluxfac)
    totuclfl = _wrf_lw_broadband_flux(clrurad_lg, spec.delwave, fluxfac)
    totdclfl = _wrf_lw_broadband_flux(clrdrad_full, spec.delwave, fluxfac)
    fnet = _wrf_ordered_difference(totuflux, totdflux)
    fnetc = _wrf_ordered_difference(totuclfl, totdclfl)

    htr_core = heatfac * (fnet[:, :n] - fnet[:, 1 : n + 1]) / (pz_arr[:, :n] - pz_arr[:, 1 : n + 1])
    htrc_core = heatfac * (fnetc[:, :n] - fnetc[:, 1 : n + 1]) / (pz_arr[:, :n] - pz_arr[:, 1 : n + 1])
    htr = jnp.concatenate((htr_core, jnp.zeros((htr_core.shape[0], 1), dtype=jnp.float32)), axis=1)
    htrc = jnp.concatenate((htrc_core, jnp.zeros((htrc_core.shape[0], 1), dtype=jnp.float32)), axis=1)
    return RrtmgLwRtrnmcResult(
        totuflux=totuflux,
        totdflux=totdflux,
        fnet=fnet,
        htr=htr,
        totuclfl=totuclfl,
        totdclfl=totdclfl,
        fnetc=fnetc,
        htrc=htrc,
    )


def cldprmc_lw(
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
    cloud_tables: RrtmgLwCloudTables | None = None,
) -> RrtmgLwCldprmcResult:
    """WRF ``rrtmg_lw_cldprmc:cldprmc`` for one column.

    Inputs and outputs use WRF's ``(ngptlw, nlayers)`` orientation.
    """

    n = int(nlayers)
    inflag_i = int(inflag)
    iceflag_i = int(iceflag)
    liqflag_i = int(liqflag)
    if inflag_i == 1:
        raise NotImplementedError("WRF cldprmc_lw stops for INFLAG = 1 with MCICA")
    if inflag_i >= 2 and liqflag_i not in (0, 1):
        raise NotImplementedError(f"WRF cldprmc_lw liquid option {liqflag_i} is not translated")

    tables = load_rrtmg_lw_cloud_tables() if cloud_tables is None else cloud_tables
    cldmin = jnp.asarray(1.0e-20, dtype=jnp.float32)

    cldf = jnp.asarray(cldfmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    ciwp = jnp.asarray(ciwpmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    clwp = jnp.asarray(clwpmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    cswp = jnp.asarray(cswpmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    tau_in = jnp.asarray(taucmc, dtype=jnp.float32)[:RRTMG_LW_NGPT, :n]
    radice = jnp.asarray(reicmc, dtype=jnp.float32)[:n][None, :]
    radliq = jnp.asarray(relqmc, dtype=jnp.float32)[:n][None, :]
    radsno = jnp.asarray(resnmc, dtype=jnp.float32)[:n][None, :]
    cwp = ciwp + clwp + cswp
    active = (cldf >= cldmin) & ((cwp >= cldmin) | (tau_in >= cldmin))

    if inflag_i == 0:
        return RrtmgLwCldprmcResult(ncbands=1, taucmc=tau_in)

    ngb = jnp.asarray(NGB_LW, dtype=jnp.int32)[:RRTMG_LW_NGPT]
    band0 = ngb[:, None] - 1
    zeros = jnp.zeros_like(tau_in)
    has_ice_or_snow = (ciwp + cswp) != 0.0

    if iceflag_i == 0:
        abscoice = tables.absice0[0] + tables.absice0[1] / radice
        abscosno = zeros
        ncbands = 1
    elif iceflag_i == 1:
        icb = jnp.asarray((1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5), dtype=jnp.int32)
        mapped_band0 = icb[band0] - 1
        abscoice = tables.absice1[0, mapped_band0] + tables.absice1[1, mapped_band0] / radice
        abscosno = zeros
        ncbands = 5
    elif iceflag_i == 2:
        factor = (radice - 2.0) / 3.0
        index = _fortran_int(factor)
        index = jnp.where(index == 43, 42, index)
        fint = factor - index.astype(jnp.float32)
        abscoice = _interp_by_fortran_row(tables.absice2, index, fint, band0)
        abscosno = zeros
        ncbands = 16
    elif iceflag_i >= 3:
        factor = (radice - 2.0) / 3.0
        index = _fortran_int(factor)
        index = jnp.where(index == 46, 45, index)
        fint = factor - index.astype(jnp.float32)
        abscoice = _interp_by_fortran_row(tables.absice3, index, fint, band0)
        abscosno = zeros
        ncbands = 16
    else:
        raise NotImplementedError(f"WRF cldprmc_lw ice option {iceflag_i} is not translated")

    abscoice = jnp.where(has_ice_or_snow, abscoice, 0.0)
    abscosno = jnp.where(has_ice_or_snow, abscosno, 0.0)

    if iceflag_i == 5:
        factor_snow = (radsno - 2.0) / 3.0
        index_snow = _fortran_int(factor_snow)
        index_snow = jnp.where(index_snow == 46, 45, index_snow)
        fint_snow = factor_snow - index_snow.astype(jnp.float32)
        abscosno_raw = _interp_by_fortran_row(tables.absice3, index_snow, fint_snow, band0)
        abscosno = jnp.where(cswp > 0.0, abscosno_raw, abscosno)

    if liqflag_i == 0:
        abscoliq = jnp.where(clwp == 0.0, 0.0, tables.absliq0)
    elif liqflag_i == 1:
        index_liq = _fortran_int(radliq - 1.5)
        index_liq = jnp.where(index_liq == 0, 1, index_liq)
        index_liq = jnp.where(index_liq == 58, 57, index_liq)
        fint_liq = radliq - 1.5 - index_liq.astype(jnp.float32)
        abscoliq_raw = _interp_by_fortran_row(tables.absliq1, index_liq, fint_liq, band0)
        abscoliq = jnp.where(clwp == 0.0, 0.0, abscoliq_raw)
    else:
        raise NotImplementedError(f"WRF cldprmc_lw liquid option {liqflag_i} is not translated")

    tau_new = ciwp * abscoice + clwp * abscoliq + cswp * abscosno
    return RrtmgLwCldprmcResult(ncbands=ncbands, taucmc=jnp.where(active, tau_new, tau_in))


def cldprmc_lw_columns(
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
    cloud_tables: RrtmgLwCloudTables | None = None,
) -> RrtmgLwCldprmcResult:
    """Batch WRF longwave ``cldprmc`` over independent horizontal columns."""

    def _one_column(cldf_i, ciwp_i, clwp_i, cswp_i, rei_i, rel_i, res_i, tau_i):
        return cldprmc_lw(
            nlayers=nlayers,
            inflag=inflag,
            iceflag=iceflag,
            liqflag=liqflag,
            cldfmc=cldf_i,
            ciwpmc=ciwp_i,
            clwpmc=clwp_i,
            cswpmc=cswp_i,
            reicmc=rei_i,
            relqmc=rel_i,
            resnmc=res_i,
            taucmc=tau_i,
            cloud_tables=cloud_tables,
        ).taucmc

    tau = jax.vmap(_one_column)(cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc, taucmc)
    ncbands = cldprmc_lw(
        nlayers=nlayers,
        inflag=inflag,
        iceflag=iceflag,
        liqflag=liqflag,
        cldfmc=jnp.asarray(cldfmc, dtype=jnp.float32)[0],
        ciwpmc=jnp.asarray(ciwpmc, dtype=jnp.float32)[0],
        clwpmc=jnp.asarray(clwpmc, dtype=jnp.float32)[0],
        cswpmc=jnp.asarray(cswpmc, dtype=jnp.float32)[0],
        reicmc=jnp.asarray(reicmc, dtype=jnp.float32)[0],
        relqmc=jnp.asarray(relqmc, dtype=jnp.float32)[0],
        resnmc=jnp.asarray(resnmc, dtype=jnp.float32)[0],
        taucmc=jnp.asarray(taucmc, dtype=jnp.float32)[0],
        cloud_tables=cloud_tables,
    ).ncbands
    return RrtmgLwCldprmcResult(ncbands=ncbands, taucmc=tau)


def rrtmg_lw_column(
    *,
    nlay: int,
    icld: int,
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
    cfc11vmr: Any,
    cfc12vmr: Any,
    cfc22vmr: Any,
    ccl4vmr: Any,
    emis: Any,
    inflglw: int,
    iceflglw: int,
    liqflglw: int,
    cldfmcl: Any | None = None,
    taucmcl: Any | None = None,
    ciwpmcl: Any | None = None,
    clwpmcl: Any | None = None,
    cswpmcl: Any | None = None,
    reicmcl: Any | None = None,
    relqmcl: Any | None = None,
    resnmcl: Any | None = None,
    tauaer: Any | None = None,
    iaer: int = 10,
    calc_clean_atm_diag: int = 0,
    spectral: RrtmgSpectralInit | None = None,
    setcoef_tables: RrtmgLwSetcoefTables | None = None,
    lookup_tables: RrtmgLwCompressedLookupTables | None = None,
    cloud_tables: RrtmgLwCloudTables | None = None,
    pade_tables: RrtmgLwPadeTables | None = None,
) -> RrtmgLwDriverResult:
    """WRF longwave ``rrtmg_lw`` for one atmospheric column."""

    del calc_clean_atm_diag
    n = int(nlay)
    spec = lwdatinit() if spectral is None else spectral
    inatm = inatm_lw_column(
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
        cfc11vmr=cfc11vmr,
        cfc12vmr=cfc12vmr,
        cfc22vmr=cfc22vmr,
        ccl4vmr=ccl4vmr,
        emis=emis,
        inflglw=inflglw,
        iceflglw=iceflglw,
        liqflglw=liqflglw,
        cldfmcl=cldfmcl,
        taucmcl=taucmcl,
        ciwpmcl=ciwpmcl,
        clwpmcl=clwpmcl,
        cswpmcl=cswpmcl,
        reicmcl=reicmcl,
        relqmcl=relqmcl,
        resnmcl=resnmcl,
        tauaer=tauaer,
        spectral=spec,
    )
    cloud = cldprmc_lw(
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
        cloud_tables=cloud_tables,
    )
    coef = setcoef_lw(
        nlayers=n,
        istart=1,
        pavel=inatm.pavel,
        tavel=inatm.tavel,
        tz=inatm.tz,
        tbound=inatm.tbound,
        semiss=inatm.semiss,
        coldry=inatm.coldry,
        wkl=inatm.wkl,
        wbroad=inatm.wbrodl,
        tables=setcoef_tables,
    )
    tau = taumol_lw(
        nlayers=n,
        pavel=inatm.pavel,
        wx=inatm.wx,
        coldry=inatm.coldry,
        laytrop=coef.laytrop,
        jp=coef.jp,
        jt=coef.jt,
        jt1=coef.jt1,
        colh2o=coef.colh2o,
        colco2=coef.colco2,
        colo3=coef.colo3,
        coln2o=coef.coln2o,
        colco=coef.colco,
        colch4=coef.colch4,
        colo2=coef.colo2,
        colbrd=coef.colbrd,
        fac00=coef.fac00,
        fac01=coef.fac01,
        fac10=coef.fac10,
        fac11=coef.fac11,
        rat_h2oco2=coef.rat_h2oco2,
        rat_h2oco2_1=coef.rat_h2oco2_1,
        rat_h2oo3=coef.rat_h2oo3,
        rat_h2oo3_1=coef.rat_h2oo3_1,
        rat_h2on2o=coef.rat_h2on2o,
        rat_h2on2o_1=coef.rat_h2on2o_1,
        rat_h2och4=coef.rat_h2och4,
        rat_h2och4_1=coef.rat_h2och4_1,
        rat_n2oco2=coef.rat_n2oco2,
        rat_n2oco2_1=coef.rat_n2oco2_1,
        rat_o3co2=coef.rat_o3co2,
        rat_o3co2_1=coef.rat_o3co2_1,
        selffac=coef.selffac,
        selffrac=coef.selffrac,
        indself=coef.indself,
        forfac=coef.forfac,
        forfrac=coef.forfrac,
        indfor=coef.indfor,
        minorfrac=coef.minorfrac,
        scaleminor=coef.scaleminor,
        scaleminorn2=coef.scaleminorn2,
        indminor=coef.indminor,
        lookup_tables=lookup_tables,
        tables=setcoef_tables,
    )
    if int(iaer) == 0:
        taut = tau.taug
    elif int(iaer) == 10:
        ngb0 = jnp.asarray(NGB_LW, dtype=jnp.int32)[:RRTMG_LW_NGPT] - 1
        taut = tau.taug + inatm.taua[:, ngb0]
    else:
        raise NotImplementedError(f"WRF rrtmg_lw aerosol option iaer={int(iaer)} is not implemented")
    rt = rtrnmc_lw(
        nlayers=n,
        istart=1,
        iend=RRTMG_LW_NBND,
        iout=0,
        pz=inatm.pz,
        semiss=inatm.semiss,
        ncbands=cloud.ncbands,
        cldfmc=inatm.cldfmc,
        taucmc=cloud.taucmc,
        planklay=coef.planklay,
        planklev=coef.planklev,
        plankbnd=coef.plankbnd,
        pwvcm=inatm.pwvcm,
        fracs=tau.fracs,
        taut=taut,
        spectral=spec,
        pade_tables=pade_tables,
    )
    zeros_clean = jnp.zeros((n + 1,), dtype=jnp.float32)
    return RrtmgLwDriverResult(
        uflx=rt.totuflux,
        dflx=rt.totdflux,
        hr=rt.htr[:n],
        uflxc=rt.totuclfl,
        dflxc=rt.totdclfl,
        hrc=rt.htrc[:n],
        uflxcln=zeros_clean,
        dflxcln=zeros_clean,
        fracs=tau.fracs,
        taug=tau.taug,
        taut=taut,
        taucmc=cloud.taucmc,
    )


def rrtmg_lw_columns(
    *,
    nlay: int,
    icld: int,
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
    cfc11vmr: Any,
    cfc12vmr: Any,
    cfc22vmr: Any,
    ccl4vmr: Any,
    emis: Any,
    inflglw: int,
    iceflglw: int,
    liqflglw: int,
    cldfmcl: Any,
    taucmcl: Any,
    ciwpmcl: Any,
    clwpmcl: Any,
    cswpmcl: Any,
    reicmcl: Any,
    relqmcl: Any,
    resnmcl: Any,
    tauaer: Any,
    iaer: int = 10,
    spectral: RrtmgSpectralInit | None = None,
    setcoef_tables: RrtmgLwSetcoefTables | None = None,
    lookup_tables: RrtmgLwCompressedLookupTables | None = None,
    cloud_tables: RrtmgLwCloudTables | None = None,
    pade_tables: RrtmgLwPadeTables | None = None,
) -> RrtmgLwDriverResult:
    """Batch WRF longwave ``rrtmg_lw`` over horizontal columns with ``vmap``."""

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0

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
        cfc11vmr_i,
        cfc12vmr_i,
        cfc22vmr_i,
        ccl4vmr_i,
        emis_i,
        cldfmcl_i,
        taucmcl_i,
        ciwpmcl_i,
        clwpmcl_i,
        cswpmcl_i,
        reicmcl_i,
        relqmcl_i,
        resnmcl_i,
        tauaer_i,
    ):
        return inatm_lw_column(
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
            cfc11vmr=cfc11vmr_i,
            cfc12vmr=cfc12vmr_i,
            cfc22vmr=cfc22vmr_i,
            ccl4vmr=ccl4vmr_i,
            emis=emis_i,
            inflglw=inflglw,
            iceflglw=iceflglw,
            liqflglw=liqflglw,
            cldfmcl=cldfmcl_i,
            taucmcl=taucmcl_i,
            ciwpmcl=ciwpmcl_i,
            clwpmcl=clwpmcl_i,
            cswpmcl=cswpmcl_i,
            reicmcl=reicmcl_i,
            relqmcl=relqmcl_i,
            resnmcl=resnmcl_i,
            tauaer=tauaer_i,
            spectral=spectral,
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
        cfc11vmr,
        cfc12vmr,
        cfc22vmr,
        ccl4vmr,
        emis,
        cldfmcl,
        taucmcl,
        ciwpmcl,
        clwpmcl,
        cswpmcl,
        reicmcl,
        relqmcl,
        resnmcl,
        tauaer,
    )
    _profile_rrtmg_lw_write("columns_inatm", profile_start, inatm)

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    cloud_inflag = int(inflglw) if int(icld) >= 1 else 0
    cloud_iceflag = int(iceflglw) if int(icld) >= 1 else 0
    cloud_liqflag = int(liqflglw) if int(icld) >= 1 else 0
    cloud = cldprmc_lw_columns(
        nlayers=nlay,
        inflag=cloud_inflag,
        iceflag=cloud_iceflag,
        liqflag=cloud_liqflag,
        cldfmc=inatm.cldfmc,
        ciwpmc=inatm.ciwpmc,
        clwpmc=inatm.clwpmc,
        cswpmc=inatm.cswpmc,
        reicmc=inatm.reicmc,
        relqmc=inatm.relqmc,
        resnmc=inatm.resnmc,
        taucmc=inatm.taucmc,
        cloud_tables=cloud_tables,
    )
    _profile_rrtmg_lw_write("columns_cldprmc", profile_start, cloud)

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0

    coef = setcoef_lw(
        nlayers=nlay,
        istart=1,
        pavel=inatm.pavel,
        tavel=inatm.tavel,
        tz=inatm.tz,
        tbound=inatm.tbound,
        semiss=inatm.semiss,
        coldry=inatm.coldry,
        wkl=inatm.wkl,
        wbroad=inatm.wbrodl,
        tables=setcoef_tables,
    )
    _profile_rrtmg_lw_write("columns_setcoef", profile_start, coef)

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0

    def _taumol_one(
        pavel_i,
        coldry_i,
        wx_i,
        laytrop_i,
        jp_i,
        jt_i,
        jt1_i,
        colh2o_i,
        colco2_i,
        colo3_i,
        coln2o_i,
        colco_i,
        colch4_i,
        colo2_i,
        colbrd_i,
        fac00_i,
        fac01_i,
        fac10_i,
        fac11_i,
        rat_h2oco2_i,
        rat_h2oco2_1_i,
        rat_h2oo3_i,
        rat_h2oo3_1_i,
        rat_h2on2o_i,
        rat_h2on2o_1_i,
        rat_h2och4_i,
        rat_h2och4_1_i,
        rat_n2oco2_i,
        rat_n2oco2_1_i,
        rat_o3co2_i,
        rat_o3co2_1_i,
        selffac_i,
        selffrac_i,
        indself_i,
        forfac_i,
        forfrac_i,
        indfor_i,
        minorfrac_i,
        scaleminor_i,
        scaleminorn2_i,
        indminor_i,
    ):
        return taumol_lw(
            nlayers=nlay,
            pavel=pavel_i,
            coldry=coldry_i,
            wx=wx_i,
            laytrop=laytrop_i,
            jp=jp_i,
            jt=jt_i,
            jt1=jt1_i,
            colh2o=colh2o_i,
            colco2=colco2_i,
            colo3=colo3_i,
            coln2o=coln2o_i,
            colco=colco_i,
            colch4=colch4_i,
            colo2=colo2_i,
            colbrd=colbrd_i,
            fac00=fac00_i,
            fac01=fac01_i,
            fac10=fac10_i,
            fac11=fac11_i,
            rat_h2oco2=rat_h2oco2_i,
            rat_h2oco2_1=rat_h2oco2_1_i,
            rat_h2oo3=rat_h2oo3_i,
            rat_h2oo3_1=rat_h2oo3_1_i,
            rat_h2on2o=rat_h2on2o_i,
            rat_h2on2o_1=rat_h2on2o_1_i,
            rat_h2och4=rat_h2och4_i,
            rat_h2och4_1=rat_h2och4_1_i,
            rat_n2oco2=rat_n2oco2_i,
            rat_n2oco2_1=rat_n2oco2_1_i,
            rat_o3co2=rat_o3co2_i,
            rat_o3co2_1=rat_o3co2_1_i,
            selffac=selffac_i,
            selffrac=selffrac_i,
            indself=indself_i,
            forfac=forfac_i,
            forfrac=forfrac_i,
            indfor=indfor_i,
            minorfrac=minorfrac_i,
            scaleminor=scaleminor_i,
            scaleminorn2=scaleminorn2_i,
            indminor=indminor_i,
            lookup_tables=lookup_tables,
            tables=setcoef_tables,
        )

    tau = jax.vmap(_taumol_one)(
        inatm.pavel,
        inatm.coldry,
        inatm.wx,
        coef.laytrop,
        coef.jp,
        coef.jt,
        coef.jt1,
        coef.colh2o,
        coef.colco2,
        coef.colo3,
        coef.coln2o,
        coef.colco,
        coef.colch4,
        coef.colo2,
        coef.colbrd,
        coef.fac00,
        coef.fac01,
        coef.fac10,
        coef.fac11,
        coef.rat_h2oco2,
        coef.rat_h2oco2_1,
        coef.rat_h2oo3,
        coef.rat_h2oo3_1,
        coef.rat_h2on2o,
        coef.rat_h2on2o_1,
        coef.rat_h2och4,
        coef.rat_h2och4_1,
        coef.rat_n2oco2,
        coef.rat_n2oco2_1,
        coef.rat_o3co2,
        coef.rat_o3co2_1,
        coef.selffac,
        coef.selffrac,
        coef.indself,
        coef.forfac,
        coef.forfrac,
        coef.indfor,
        coef.minorfrac,
        coef.scaleminor,
        coef.scaleminorn2,
        coef.indminor,
    )
    _profile_rrtmg_lw_write("columns_taumol", profile_start, tau)

    if int(iaer) == 0:
        taut = tau.taug
    elif int(iaer) == 10:
        ngb0 = jnp.asarray(NGB_LW, dtype=jnp.int32)[:RRTMG_LW_NGPT] - 1
        taut = tau.taug + jnp.take(inatm.taua, ngb0, axis=2)
    else:
        raise NotImplementedError(f"WRF rrtmg_lw aerosol option iaer={int(iaer)} is not implemented")

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0

    rt = rtrnmc_lw_columns(
        nlayers=nlay,
        istart=1,
        iend=RRTMG_LW_NBND,
        iout=0,
        pz=inatm.pz,
        semiss=inatm.semiss,
        ncbands=cloud.ncbands,
        cldfmc=inatm.cldfmc,
        taucmc=cloud.taucmc,
        planklay=coef.planklay,
        planklev=coef.planklev,
        plankbnd=coef.plankbnd,
        pwvcm=inatm.pwvcm,
        fracs=tau.fracs,
        taut=taut,
        spectral=spectral,
        pade_tables=pade_tables,
    )
    _profile_rrtmg_lw_write("columns_rtrnmc", profile_start, rt)

    zeros_clean = jnp.zeros_like(rt.totuflux)
    return RrtmgLwDriverResult(
        uflx=rt.totuflux,
        dflx=rt.totdflux,
        hr=rt.htr[:, :nlay],
        uflxc=rt.totuclfl,
        dflxc=rt.totdclfl,
        hrc=rt.htrc[:, :nlay],
        uflxcln=zeros_clean,
        dflxcln=zeros_clean,
        fracs=tau.fracs,
        taug=tau.taug,
        taut=taut,
        taucmc=cloud.taucmc,
    )


_LW_BUFFER_PPROF = jnp.asarray(
    (
        1000.00,
        855.47,
        731.82,
        626.05,
        535.57,
        458.16,
        391.94,
        335.29,
        286.83,
        245.38,
        209.91,
        179.57,
        153.62,
        131.41,
        112.42,
        96.17,
        82.27,
        70.38,
        60.21,
        51.51,
        44.06,
        37.69,
        32.25,
        27.59,
        23.60,
        20.19,
        17.27,
        14.77,
        12.64,
        10.81,
        9.25,
        7.91,
        6.77,
        5.79,
        4.95,
        4.24,
        3.63,
        3.10,
        2.65,
        2.27,
        1.94,
        1.66,
        1.42,
        1.22,
        1.04,
        0.89,
        0.76,
        0.65,
        0.56,
        0.48,
        0.41,
        0.35,
        0.30,
        0.26,
        0.22,
        0.19,
        0.16,
        0.14,
        0.12,
        0.10,
    ),
    dtype=jnp.float32,
)
_LW_BUFFER_TPROF = jnp.asarray(
    (
        286.96,
        281.07,
        275.16,
        268.11,
        260.56,
        253.02,
        245.62,
        238.41,
        231.57,
        225.91,
        221.72,
        217.79,
        215.06,
        212.74,
        210.25,
        210.16,
        210.69,
        212.14,
        213.74,
        215.37,
        216.82,
        217.94,
        219.03,
        220.18,
        221.37,
        222.64,
        224.16,
        225.88,
        227.63,
        229.51,
        231.50,
        233.73,
        236.18,
        238.78,
        241.60,
        244.44,
        247.35,
        250.33,
        253.32,
        256.30,
        259.22,
        262.12,
        264.80,
        266.50,
        267.59,
        268.44,
        268.69,
        267.76,
        266.13,
        263.96,
        261.54,
        258.93,
        256.15,
        253.23,
        249.89,
        246.67,
        243.48,
        240.25,
        236.66,
        233.86,
    ),
    dtype=jnp.float32,
)


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


def _lw_cloud_flags(*, has_reqc: int, has_reqi: int, has_reqs: int, icloud: int) -> tuple[int, int, int]:
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


def _lw_greenhouse_scalars(
    *,
    yr: int,
    julian: float,
    ghg_input: int,
    case_dir: str | Path | None,
    co2vmr: float | None,
    n2ovmr: float | None,
    ch4vmr: float | None,
    cfc11vmr: float | None,
    cfc12vmr: float | None,
) -> tuple[float, float, float, float, float]:
    if all(value is not None for value in (co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr)):
        return float(co2vmr), float(n2ovmr), float(ch4vmr), float(cfc11vmr), float(cfc12vmr)
    if int(ghg_input) == 1 and case_dir is not None:
        gas = read_camgases(int(yr), float(julian), True, "RRTMG", path=Path(case_dir) / "CAMtr_volume_mixing_ratio")
        return gas.co2vmr, gas.n2ovmr, gas.ch4vmr, gas.cfc11vmr, gas.cfc12vmr
    if int(ghg_input) == 0:
        co2 = (280.0 + 90.0 * np.exp(0.02 * (int(yr) - 2000))) * 1.0e-6
        return float(co2), 319.0e-9, 1774.0e-9, 0.251e-9, 0.538e-9
    return 379.0e-6, 319.0e-9, 1774.0e-9, 0.251e-9, 0.538e-9


def _lw_buffer_temperature_interfaces(plev: Any, tlev: Any, *, nk: int) -> Any:
    p = jnp.asarray(plev, dtype=jnp.float32)
    t = jnp.asarray(tlev, dtype=jnp.float32)
    count = jnp.sum(_LW_BUFFER_PPROF[None, None, :] >= p[:, :, None], axis=-1)
    idx = jnp.clip(count - 1, 0, _LW_BUFFER_PPROF.shape[0] - 1).astype(jnp.int32)
    idx_next = jnp.minimum(idx + 1, _LW_BUFFER_PPROF.shape[0] - 1)
    p0 = _LW_BUFFER_PPROF[idx]
    p1 = _LW_BUFFER_PPROF[idx_next]
    t0 = _LW_BUFFER_TPROF[idx]
    t1 = _LW_BUFFER_TPROF[idx_next]
    at_top = idx == _LW_BUFFER_PPROF.shape[0] - 1
    weight = jnp.where(at_top, 0.0, (p - p0) / jnp.where(at_top, 1.0, p1 - p0))
    varint = weight * (t1 - t0) + t0
    anchor = int(nk) - 1
    offset = t[:, anchor] - varint[:, anchor]
    extra = varint[:, int(nk) :] + offset[:, None]
    return t.at[:, int(nk) :].set(extra)


def _extend_lw_top_buffer(
    *,
    p3d_mb: Any,
    p8w: Any,
    t8w: Any,
    t3d: Any,
    h2ovmr: Any,
    co2vmr: Any,
    o2vmr: Any,
    ch4vmr: Any,
    n2ovmr: Any,
    cfc11vmr: Any,
    cfc12vmr: Any,
    cfc22vmr: Any,
    ccl4vmr: Any,
    o33d: Any,
    p_top: float,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    p3d_c = jnp.asarray(p3d_mb, dtype=jnp.float32)
    p8w_c = _wrf_real_pa_to_hpa(p8w)
    t8w_c = jnp.asarray(t8w, dtype=jnp.float32)
    t3d_c = jnp.asarray(t3d, dtype=jnp.float32)
    h2o_c = jnp.asarray(h2ovmr, dtype=jnp.float32)
    co2_c = jnp.asarray(co2vmr, dtype=jnp.float32)
    o2_c = jnp.asarray(o2vmr, dtype=jnp.float32)
    ch4_c = jnp.asarray(ch4vmr, dtype=jnp.float32)
    n2o_c = jnp.asarray(n2ovmr, dtype=jnp.float32)
    cfc11_c = jnp.asarray(cfc11vmr, dtype=jnp.float32)
    cfc12_c = jnp.asarray(cfc12vmr, dtype=jnp.float32)
    cfc22_c = jnp.asarray(cfc22vmr, dtype=jnp.float32)
    ccl4_c = jnp.asarray(ccl4vmr, dtype=jnp.float32)
    o3_c = jnp.asarray(o33d, dtype=jnp.float32)
    ncol = int(t3d_c.shape[0])
    nk = int(t3d_c.shape[1])
    nlay = rrtmg_lw_nlayers(p_top=float(p_top), kme=nk + 1)
    extra_layers = int(nlay) - nk

    plev = jnp.zeros((ncol, nlay + 1), dtype=jnp.float32).at[:, : nk + 1].set(p8w_c[:, : nk + 1])
    for lev in range(nk, nlay):
        plev = plev.at[:, lev + 1].set(plev[:, lev] - jnp.float32(4.0))
    plev = plev.at[:, nlay].set(jnp.float32(0.0))
    play = jnp.zeros((ncol, nlay), dtype=jnp.float32).at[:, :nk].set(p3d_c[:, :nk])
    for lay in range(nk, nlay):
        play = play.at[:, lay].set(jnp.float32(0.5) * (plev[:, lay] + plev[:, lay + 1]))
    return _finish_lw_top_buffer(
        play=play,
        plev=plev,
        t8w_c=t8w_c,
        t3d_c=t3d_c,
        h2o_c=h2o_c,
        co2_c=co2_c,
        o2_c=o2_c,
        ch4_c=ch4_c,
        n2o_c=n2o_c,
        cfc11_c=cfc11_c,
        cfc12_c=cfc12_c,
        cfc22_c=cfc22_c,
        ccl4_c=ccl4_c,
        o3_c=o3_c,
        nk=nk,
        nlay=nlay,
        extra_layers=extra_layers,
        rounding_token=rounding_token,
    )


def _finish_lw_top_buffer(
    *,
    play: Any,
    plev: Any,
    t8w_c: Any,
    t3d_c: Any,
    h2o_c: Any,
    co2_c: Any,
    o2_c: Any,
    ch4_c: Any,
    n2o_c: Any,
    cfc11_c: Any,
    cfc12_c: Any,
    cfc22_c: Any,
    ccl4_c: Any,
    o3_c: Any,
    nk: int,
    nlay: int,
    extra_layers: int,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    del extra_layers
    ncol = int(t3d_c.shape[0])
    play = jnp.asarray(play, dtype=jnp.float32)

    tlev = jnp.zeros((ncol, nlay + 1), dtype=jnp.float32).at[:, : nk + 1].set(t8w_c[:, : nk + 1])
    tlev = _lw_buffer_temperature_interfaces(plev, tlev, nk=nk)
    tlay = jnp.zeros((ncol, nlay), dtype=jnp.float32).at[:, :nk].set(t3d_c[:, :nk])
    for lev in range(nk, nlay + 1):
        tlay = tlay.at[:, lev - 1].set(0.5 * (tlev[:, lev] + tlev[:, lev - 1]))

    def extend_last(values: Any) -> Any:
        arr = jnp.asarray(values, dtype=jnp.float32)
        return jnp.concatenate((arr[:, :nk], jnp.repeat(arr[:, nk - 1 : nk], nlay - nk, axis=1)), axis=1)

    h2o = extend_last(h2o_c)
    co2 = extend_last(co2_c)
    o2 = extend_last(o2_c)
    ch4 = extend_last(ch4_c)
    n2o = extend_last(n2o_c)
    cfc11 = extend_last(cfc11_c)
    cfc12 = extend_last(cfc12_c)
    cfc22 = extend_last(cfc22_c)
    ccl4 = extend_last(ccl4_c)

    o3mmr = jax.vmap(lambda col_plev: inirad(col_plev, kte=nlay - 1))(plev)
    amdo = jnp.float32(0.603461)
    o3_top = _wrf_real_sub(
        o3_c[:, nk - 1 : nk],
        _wrf_real_mul(
            o3mmr[:, nk - 1 : nk],
            amdo,
            rounding_token,
        ),
        rounding_token,
    )
    o3_top = _wrf_real_add(
        o3_top,
        _wrf_real_mul(o3mmr[:, nk:], amdo, rounding_token),
        rounding_token,
    )
    o3_top = jnp.where(
        o3_top <= 0.0,
        _wrf_real_mul(o3mmr[:, nk:], amdo, rounding_token),
        o3_top,
    )
    o3 = jnp.concatenate((o3_c[:, :nk], o3_top), axis=1)
    return play, plev, tlay, tlev, h2o, o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4


def rrtmg_lwrad_wrf_columns(
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
    emiss: Any,
    xland: Any,
    xice: Any,
    snow: Any,
    re_cloud: Any,
    re_ice: Any,
    re_snow: Any,
    p_top: float,
    yr: int,
    julian: Any,
    ghg_input: int = 1,
    case_dir: str | Path | None = None,
    cldovrlp: int = 2,
    idcor: int = 0,
    icloud: int = 1,
    has_reqc: int = 1,
    has_reqi: int = 1,
    has_reqs: int = 1,
    co2vmr: float | None = None,
    n2ovmr: float | None = None,
    ch4vmr: float | None = None,
    cfc11vmr: float | None = None,
    cfc12vmr: float | None = None,
    lookup_tables: RrtmgLwCompressedLookupTables | None = None,
    spectral: RrtmgSpectralInit | None = None,
    pade_tables: RrtmgLwPadeTables | None = None,
    cloud_tables: RrtmgLwCloudTables | None = None,
    setcoef_tables: RrtmgLwSetcoefTables | None = None,
    rounding_token: Any | None = None,
) -> RrtmgLwWrfColumnResult:
    """Batched JAX translation of WRF ``RRTMG_LWRAD`` for one model domain."""

    _profile_rrtmg_lw_event("start")
    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    del idcor
    t3d_arr = jnp.asarray(t3d, dtype=jnp.float32)
    nx = int(t3d_arr.shape[0])
    nk = int(t3d_arr.shape[1])
    ny = int(t3d_arr.shape[2])
    ncol = nx * ny
    nlay = rrtmg_lw_nlayers(p_top=float(p_top), kme=nk + 1)

    lookup = load_rrtmg_lw_compressed_lookup_tables() if lookup_tables is None else lookup_tables
    clouds = load_rrtmg_lw_cloud_tables() if cloud_tables is None else cloud_tables
    setcoef_tab = load_rrtmg_lw_setcoef_tables() if setcoef_tables is None else setcoef_tables
    spec = lwdatinit() if spectral is None else spectral
    pade = rrtmg_lw_pade_tables() if pade_tables is None else pade_tables
    _profile_rrtmg_lw_write("tables", profile_start, ())

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    p3d_c = _wrf_real_pa_to_hpa(_flatten_wrf_mass_columns(p3d))
    p8w_c = _flatten_wrf_mass_columns(p8w)
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
    xland_c = _flatten_wrf_2d(xland)
    xice_c = _flatten_wrf_2d(xice)
    snow_c = _flatten_wrf_2d(snow)

    if all(value is not None for value in (co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr)):
        co2_scalar = jnp.asarray(co2vmr, dtype=jnp.float32)
        n2o_scalar = jnp.asarray(n2ovmr, dtype=jnp.float32)
        ch4_scalar = jnp.asarray(ch4vmr, dtype=jnp.float32)
        cfc11_scalar = jnp.asarray(cfc11vmr, dtype=jnp.float32)
        cfc12_scalar = jnp.asarray(cfc12vmr, dtype=jnp.float32)
    else:
        co2_scalar, n2o_scalar, ch4_scalar, cfc11_scalar, cfc12_scalar = _lw_greenhouse_scalars(
            yr=int(yr),
            julian=float(julian),
            ghg_input=int(ghg_input),
            case_dir=case_dir,
            co2vmr=co2vmr,
            n2ovmr=n2ovmr,
            ch4vmr=ch4vmr,
            cfc11vmr=cfc11vmr,
            cfc12vmr=cfc12vmr,
        )
    h2o_base = qv_c[:, :nk] * jnp.float32(1.607793)
    co2_base = jnp.full((ncol, nk), co2_scalar, dtype=jnp.float32)
    n2o_base = jnp.full((ncol, nk), n2o_scalar, dtype=jnp.float32)
    ch4_base = jnp.full((ncol, nk), ch4_scalar, dtype=jnp.float32)
    cfc11_base = jnp.full((ncol, nk), cfc11_scalar, dtype=jnp.float32)
    cfc12_base = jnp.full((ncol, nk), cfc12_scalar, dtype=jnp.float32)
    cfc22_base = jnp.full((ncol, nk), 0.169e-9, dtype=jnp.float32)
    ccl4_base = jnp.full((ncol, nk), 0.093e-9, dtype=jnp.float32)
    o2_base = jnp.full((ncol, nk), 0.209488, dtype=jnp.float32)
    play, plev, tlay, tlev, h2ovmr, o3vmr, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4 = _extend_lw_top_buffer(
        p3d_mb=p3d_c,
        p8w=p8w_c,
        t8w=t8w_c,
        t3d=t3d_c,
        h2ovmr=h2o_base,
        co2vmr=co2_base,
        o2vmr=o2_base,
        ch4vmr=ch4_base,
        n2ovmr=n2o_base,
        cfc11vmr=cfc11_base,
        cfc12vmr=cfc12_base,
        cfc22vmr=cfc22_base,
        ccl4vmr=ccl4_base,
        o33d=o33d_c[:, :nk],
        p_top=float(p_top),
        rounding_token=rounding_token,
    )
    _profile_rrtmg_lw_write("prepare_atmosphere", profile_start, (play, plev, tlay, tlev, h2ovmr, o3vmr))

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    pdel = plev[:, :nk] - plev[:, 1 : nk + 1]
    inflglw, iceflglw, liqflglw = _lw_cloud_flags(
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
            inflg=inflglw,
            iceflg=iceflglw,
            g=9.8066,
        )
    )(tlay[:, :nk], pdel, cldfra_c[:, :nk], radii.qc, radii.qi, radii.qs, xland_c, xice_c, snow_c, radii.recloud, radii.reice, radii.resnow)
    _profile_rrtmg_lw_write("cloud_inputs", profile_start, (radii, paths))

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    extra = nlay - nk
    zero_top = jnp.zeros((ncol, extra), dtype=jnp.float32)
    ten_top = jnp.full((ncol, extra), 10.0, dtype=jnp.float32)
    cldfrac = jnp.concatenate((paths.cldfrac, zero_top), axis=1)
    clwpth = jnp.concatenate((paths.clwpth, zero_top), axis=1)
    ciwpth = jnp.concatenate((paths.ciwpth, zero_top), axis=1)
    cswpth = jnp.concatenate((paths.cswpth, zero_top), axis=1)
    rel = jnp.concatenate((paths.rel, ten_top), axis=1)
    rei = jnp.concatenate((paths.rei, ten_top), axis=1)
    res = jnp.concatenate((paths.res, ten_top), axis=1)
    taucld = jnp.zeros((RRTMG_LW_NBND, ncol, nlay), dtype=jnp.float32)
    mcica = mcica_subcol_lw_overlap2(
        play=play,
        cldfrac=cldfrac,
        ciwp=ciwpth,
        clwp=clwpth,
        cswp=cswpth,
        rei=rei,
        rel=rel,
        res=res,
        tauc=taucld,
        change_seed=150,
        ngb=NGB_LW,
    )
    _profile_rrtmg_lw_write("mcica", profile_start, mcica)

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    tauaer = jnp.zeros((ncol, nlay, RRTMG_LW_NBND), dtype=jnp.float32)
    column = rrtmg_lw_columns(
        nlay=nlay,
        icld=int(cldovrlp),
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
        cfc11vmr=cfc11,
        cfc12vmr=cfc12,
        cfc22vmr=cfc22,
        ccl4vmr=ccl4,
        emis=jnp.repeat(
            _flatten_wrf_2d(emiss)[:, None],
            RRTMG_LW_NBND,
            axis=1,
        ),
        inflglw=inflglw,
        iceflglw=iceflglw,
        liqflglw=liqflglw,
        cldfmcl=jnp.swapaxes(mcica.cldfmcl, 0, 1),
        taucmcl=jnp.swapaxes(mcica.taucmcl, 0, 1),
        ciwpmcl=jnp.swapaxes(mcica.ciwpmcl, 0, 1),
        clwpmcl=jnp.swapaxes(mcica.clwpmcl, 0, 1),
        cswpmcl=jnp.swapaxes(mcica.cswpmcl, 0, 1),
        reicmcl=mcica.reicmcl,
        relqmcl=mcica.relqmcl,
        resnmcl=mcica.resnmcl,
        tauaer=tauaer,
        iaer=10,
        spectral=spec,
        setcoef_tables=setcoef_tab,
        lookup_tables=lookup,
        cloud_tables=clouds,
        pade_tables=pade,
    )
    _profile_rrtmg_lw_write("rrtmg_lw_columns", profile_start, column)

    profile_start = perf_counter() if _PROFILE_RRTMG_LW_PATH else 0.0
    hr = _wrf_real_store(column.hr, rounding_token)
    hrc = _wrf_real_store(column.hrc, rounding_token)
    uflx = _wrf_real_store(column.uflx, rounding_token)
    dflx = _wrf_real_store(column.dflx, rounding_token)
    uflxc = _wrf_real_store(column.uflxc, rounding_token)
    dflxc = _wrf_real_store(column.dflxc, rounding_token)
    rthratenlw_c = _wrf_ordered_divisions(
        hr[:, :nk],
        jnp.float32(86400.0),
        pi3d_c[:, :nk],
        rounding_token=rounding_token,
    )
    rthratenlwc_c = _wrf_ordered_divisions(
        hrc[:, :nk],
        jnp.float32(86400.0),
        pi3d_c[:, :nk],
        rounding_token=rounding_token,
    )
    zeros2 = jnp.zeros((ncol,), dtype=jnp.float32)
    result = RrtmgLwWrfColumnResult(
        rthratenlw=_unflatten_wrf_mass_columns(rthratenlw_c, nx=nx, ny=ny),
        rthratenlwc=_unflatten_wrf_mass_columns(rthratenlwc_c, nx=nx, ny=ny),
        lwupt=_unflatten_wrf_2d(uflx[:, -1], nx=nx, ny=ny),
        lwuptc=_unflatten_wrf_2d(uflxc[:, -1], nx=nx, ny=ny),
        lwuptcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        lwdnt=_unflatten_wrf_2d(dflx[:, -1], nx=nx, ny=ny),
        lwdntc=_unflatten_wrf_2d(dflxc[:, -1], nx=nx, ny=ny),
        lwdntcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        lwupb=_unflatten_wrf_2d(uflx[:, 0], nx=nx, ny=ny),
        lwupbc=_unflatten_wrf_2d(uflxc[:, 0], nx=nx, ny=ny),
        lwupbcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        lwdnb=_unflatten_wrf_2d(dflx[:, 0], nx=nx, ny=ny),
        lwdnbc=_unflatten_wrf_2d(dflxc[:, 0], nx=nx, ny=ny),
        lwdnbcln=_unflatten_wrf_2d(zeros2, nx=nx, ny=ny),
        glw=_unflatten_wrf_2d(dflx[:, 0], nx=nx, ny=ny),
        olr=_unflatten_wrf_2d(uflx[:, -1], nx=nx, ny=ny),
        lwcf=_unflatten_wrf_2d(uflxc[:, -1] - uflx[:, -1], nx=nx, ny=ny),
    )
    _profile_rrtmg_lw_write("outputs", profile_start, result)
    return result


__all__ = [
    "RrtmgLwBandData",
    "RrtmgLwCldprmcResult",
    "RrtmgLwCloudTables",
    "RrtmgLwCompressedBandData",
    "RrtmgLwCompressedLookupTables",
    "RrtmgLwDriverResult",
    "RrtmgLwInatmResult",
    "RrtmgLwLookupTables",
    "RrtmgLwRtrnmcResult",
    "RrtmgLwSetcoefResult",
    "RrtmgLwSetcoefTables",
    "RrtmgLwTaumolBandResult",
    "RrtmgLwWrfColumnResult",
    "combine_rrtmg_lw_lookup_tables",
    "cldprmc_lw",
    "cldprmc_lw_columns",
    "inatm_lw_column",
    "load_rrtmg_lw_compressed_lookup_tables",
    "load_rrtmg_lw_data",
    "load_rrtmg_lw_cloud_tables",
    "load_rrtmg_lw_setcoef_tables",
    "rrtmg_lw_column",
    "rrtmg_lw_columns",
    "rrtmg_lwrad_wrf_columns",
    "rtrnmc_lw",
    "setcoef_lw",
    "taumol1_lw",
    "taumol2_lw",
    "taumol3_lw",
    "taumol4_lw",
    "taumol5_lw",
    "taumol6_lw",
    "taumol7_lw",
    "taumol8_lw",
    "taumol9_lw",
    "taumol10_lw",
    "taumol11_lw",
    "taumol12_lw",
    "taumol13_lw",
    "taumol14_lw",
    "taumol15_lw",
    "taumol16_lw",
    "taumol_lw",
]
