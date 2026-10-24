"""Translations from WRF ``phys/module_physics_init`` initialization drivers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from wrf_jax.physics.cu_init import cu_init as _cu_init
from wrf_jax.physics.fdob_init import fdob_init as _fdob_init
from wrf_jax.physics.fg_init import fg_init as _fg_init
from wrf_jax.physics.module_sf_noahdrv import lsminit
from wrf_jax.physics.mp_init import mp_init as _mp_init
from wrf_jax.physics.myjpbl import myjpblinit
from wrf_jax.physics.myjsfc import myjsfcinit
from wrf_jax.physics.ra_init import ra_init as _ra_init
from wrf_jax.physics.shcu_init import shcu_init as _shcu_init
from wrf_jax.physics.z2sigma import z2sigma as _z2sigma


SFCLAYREVSCHEME = 1
MYJSFCSCHEME = 2
QNSESFCSCHEME = 4
SFCLAYSCHEME = 91
SLABSCHEME = 1
LSMSCHEME = 2
NOAHMPSCHEME = 4
RUCLSMSCHEME = 3
PXLSMSCHEME = 7
YSUSCHEME = 1
MYJPBLSCHEME = 2
GODDARDLWSCHEME = 5
GODDARDSWSCHEME = 5
NUWRF4ICESCHEME = 7
CAMMGMPSCHEME = 11
KFETASCHEME = 1
GFSCHEME = 3
G3SCHEME = 5
CAMZMSCHEME = 7
MSKFSCHEME = 11
GDSCHEME = 93
CAMUWSHCUSCHEME = 2
CAMUWPBLSCHEME = 9
RE_QC_BG = np.float32(2.49e-6)
RE_QI_BG = np.float32(4.99e-6)
RE_QS_BG = np.float32(9.99e-6)


@dataclass(frozen=True)
class BlInitResult:
    stepbl: int
    rublten: Any
    rvblten: Any
    rthblten: Any
    rqvblten: Any
    rqcblten: Any
    rqiblten: Any
    tke_pbl: Any
    exch_h: Any
    ust: Any
    lowlyr: Any
    znt: Any
    vegfra: Any
    snow: Any
    snowc: Any
    snowh: Any
    canwat: Any
    smstav: Any
    smstot: Any
    sfcrunoff: Any
    udrunoff: Any
    acsnow: Any
    acsnom: Any
    ivgtyp: Any
    isltyp: Any
    tslb: Any
    smois: Any
    sh2o: Any
    zs: Any
    dzs: Any
    snoalb: Any
    sfcevp: Any
    achfx: Any | None = None
    aclhf: Any | None = None
    acgrdflx: Any | None = None
    tsk_save: Any | None = None
    irr_rand_field: Any | None = None
    lsm_parameters: Any | None = None
    messages: tuple[str, ...] = ()
    external_calls: tuple[str, ...] = ()
    carried_arrays: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LanduseInitResult:
    """Result of WRF ``landuse_init`` with explicit output arrays and state."""

    albedo: Any
    albbck: Any
    snoalb: Any
    mavail: Any
    emiss: Any
    embck: Any
    znt: Any
    z0: Any
    thc: Any
    xland: Any
    xicem: Any
    isice: int
    lucats: int
    luseas: int
    isn: int
    fractional_seaice: int
    lu_state: Any
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhyInitResult:
    """Explicit state returned by WRF ``phy_init`` orchestration."""

    fields: Mapping[str, Any]
    stepbl: int
    stepra: int
    stepcu: int
    stepfg: int
    warm_rain: bool
    adv_moist_cond: bool
    is_cammgmp_used: bool
    icloud_cu: int
    landuse_isice: int
    landuse_lucats: int
    landuse_luseas: int
    landuse_isn: int
    sfull: Any
    shalf: Any
    pptop: float
    subresults: Mapping[str, Any]
    config_updates: Mapping[str, Any]
    messages: tuple[str, ...] = ()
    external_calls: tuple[str, ...] = ()


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_domain_value(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return int(default)
        return int(value[0])
    return int(value)


def _nint_positive(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _nint_fortran(value: float) -> int:
    value_f = float(value)
    if value_f >= 0.0:
        return int(np.floor(value_f + 0.5))
    return int(np.ceil(value_f - 0.5))


def _unpack_landuse_state(lu_state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(lu_state, dtype=np.float64)
    if state.size < 7300:
        raise ValueError("lu_state must contain at least 7300 values")
    scfx = np.zeros(100, dtype=np.float64)
    albd = np.zeros((100, 12), dtype=np.float64)
    slmo = np.zeros((100, 12), dtype=np.float64)
    sfem = np.zeros((100, 12), dtype=np.float64)
    sfz0 = np.zeros((100, 12), dtype=np.float64)
    sfhc = np.zeros((100, 12), dtype=np.float64)
    therin = np.zeros((100, 12), dtype=np.float64)
    cursor = 0
    for cat in range(100):
        scfx[cat] = state[cursor]
        cursor += 1
        for season in range(12):
            albd[cat, season] = state[cursor]
            cursor += 1
            slmo[cat, season] = state[cursor]
            cursor += 1
            sfem[cat, season] = state[cursor]
            cursor += 1
            sfz0[cat, season] = state[cursor]
            cursor += 1
            sfhc[cat, season] = state[cursor]
            cursor += 1
            therin[cat, season] = state[cursor]
            cursor += 1
    return scfx, albd, slmo, sfem, sfz0, sfhc, therin


def _pack_landuse_state(
    lu_state: Any,
    scfx: np.ndarray,
    albd: np.ndarray,
    slmo: np.ndarray,
    sfem: np.ndarray,
    sfz0: np.ndarray,
    sfhc: np.ndarray,
    therin: np.ndarray,
) -> np.ndarray:
    state = np.asarray(lu_state, dtype=np.float64).copy()
    if state.size < 7300:
        raise ValueError("lu_state must contain at least 7300 values")
    cursor = 0
    for cat in range(100):
        state[cursor] = scfx[cat]
        cursor += 1
        for season in range(12):
            state[cursor] = albd[cat, season]
            cursor += 1
            state[cursor] = slmo[cat, season]
            cursor += 1
            state[cursor] = sfem[cat, season]
            cursor += 1
            state[cursor] = sfz0[cat, season]
            cursor += 1
            state[cursor] = sfhc[cat, season]
            cursor += 1
            state[cursor] = therin[cat, season]
            cursor += 1
    return state


def _copy_table_into_landuse_arrays(
    table: Mapping[str, Any],
    albd: np.ndarray,
    slmo: np.ndarray,
    sfem: np.ndarray,
    sfz0: np.ndarray,
    sfhc: np.ndarray,
    therin: np.ndarray,
    scfx: np.ndarray,
) -> tuple[int, int]:
    lucats = int(table["lucats"])
    luseas = int(table["luseas"])
    albd[:lucats, :luseas] = np.asarray(table["albd"], dtype=np.float64)[:lucats, :luseas]
    slmo[:lucats, :luseas] = np.asarray(table["slmo"], dtype=np.float64)[:lucats, :luseas]
    sfem[:lucats, :luseas] = np.asarray(table["sfem"], dtype=np.float64)[:lucats, :luseas]
    sfz0[:lucats, :luseas] = np.asarray(table["sfz0"], dtype=np.float64)[:lucats, :luseas]
    sfhc[:lucats, :luseas] = np.asarray(table["sfhc"], dtype=np.float64)[:lucats, :luseas]
    therin[:lucats, :luseas] = np.asarray(table["therin"], dtype=np.float64)[:lucats, :luseas]
    scfx[:lucats] = np.asarray(table["scfx"], dtype=np.float64)[:lucats]
    return lucats, luseas


def read_landuse_tables(table_dir: str | Path = ".") -> dict[str, dict[str, Any]]:
    """Read WRF ``LANDUSE.TBL`` blocks into ``landuse_init`` table mappings."""

    path = Path(table_dir) / "LANDUSE.TBL"
    lines = path.read_text(encoding="ascii").splitlines()
    tables: dict[str, dict[str, Any]] = {}
    idx = 0
    while idx < len(lines):
        label = lines[idx].strip()
        idx += 1
        if not label:
            continue
        if idx >= len(lines):
            break
        header = lines[idx].split("'", 1)[0].replace(",", " ").split()
        idx += 1
        if len(header) < 2:
            continue
        lucats = int(header[0])
        luseas = int(header[1])
        albd = np.zeros((lucats, luseas), dtype=np.float64)
        slmo = np.zeros((lucats, luseas), dtype=np.float64)
        sfem = np.zeros((lucats, luseas), dtype=np.float64)
        sfz0 = np.zeros((lucats, luseas), dtype=np.float64)
        sfhc = np.zeros((lucats, luseas), dtype=np.float64)
        therin = np.zeros((lucats, luseas), dtype=np.float64)
        scfx = np.zeros(lucats, dtype=np.float64)
        for season in range(luseas):
            if idx >= len(lines):
                raise ValueError(f"LANDUSE.TBL block {label!r} ended before season {season + 1}")
            idx += 1
            for cat in range(lucats):
                if idx >= len(lines):
                    raise ValueError(f"LANDUSE.TBL block {label!r} ended before category {cat + 1}")
                tokens = lines[idx].split("'", 1)[0].replace(",", " ").split()
                idx += 1
                if len(tokens) < 8:
                    raise ValueError(f"invalid LANDUSE.TBL category row in block {label!r}: {lines[idx - 1]!r}")
                category = int(float(tokens[0]))
                if category != cat + 1:
                    raise ValueError(f"LANDUSE.TBL block {label!r} missing category {cat + 1}")
                albd[cat, season] = float(tokens[1])
                slmo[cat, season] = float(tokens[2])
                sfem[cat, season] = float(tokens[3])
                sfz0[cat, season] = float(tokens[4])
                therin[cat, season] = float(tokens[5])
                scfx[cat] = float(tokens[6])
                sfhc[cat, season] = float(tokens[7])
        tables[label] = {
            "lucats": lucats,
            "luseas": luseas,
            "albd": albd,
            "slmo": slmo,
            "sfem": sfem,
            "sfz0": sfz0,
            "sfhc": sfhc,
            "therin": therin,
            "scfx": scfx,
        }
    return tables


def _tile2d_direct(ims: int, jms: int, its: int, ite: int, jts: int, jte: int) -> tuple[slice, slice]:
    return (
        slice(int(its) - int(ims), int(ite) - int(ims) + 1),
        slice(int(jts) - int(jms), int(jte) - int(jms) + 1),
    )


def _zero_2d_tile(array: Any, slices: tuple[slice, slice]) -> Any:
    arr = jnp.asarray(array)
    return arr.at[slices].set(jnp.asarray(0.0, dtype=arr.dtype))


def _copy_2d_tile(source: Any, target: Any, slices: tuple[slice, slice]) -> Any:
    source_arr = jnp.asarray(source)
    target_arr = jnp.asarray(target)
    return target_arr.at[slices].set(source_arr[slices])


def _tile3d(
    ide: int,
    jde: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[slice, slice, slice]:
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    return (
        slice(int(its) - int(ims), itf - int(ims) + 1),
        slice(int(kts) - int(kms), ktf - int(kms) + 1),
        slice(int(jts) - int(jms), jtf - int(jms) + 1),
    )


def _tile2d_bounded(
    ide: int,
    jde: int,
    ims: int,
    jms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
) -> tuple[slice, slice]:
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    return (
        slice(int(its) - int(ims), itf - int(ims) + 1),
        slice(int(jts) - int(jms), jtf - int(jms) + 1),
    )


def _full_k_tile(
    ide: int,
    jde: int,
    ims: int,
    jms: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
) -> tuple[slice, slice, slice]:
    i_slice, j_slice = _tile2d_bounded(ide, jde, ims, jms, its, ite, jts, jte)
    return i_slice, slice(0, int(kme) - int(kms) + 1), j_slice


def _nint_jax(values: Any) -> Any:
    arr = jnp.asarray(values)
    return jnp.where(arr >= 0, jnp.floor(arr + 0.5), jnp.ceil(arr - 0.5))


def _set_2d_constant(fields: dict[str, Any], name: str, slices: tuple[slice, slice], value: float) -> None:
    if name in fields and fields[name] is not None:
        arr = jnp.asarray(fields[name])
        fields[name] = arr.at[slices].set(jnp.asarray(value, dtype=arr.dtype))


def _set_3d_constant(fields: dict[str, Any], name: str, slices: tuple[slice, slice, slice], value: float) -> None:
    if name in fields and fields[name] is not None:
        arr = jnp.asarray(fields[name])
        fields[name] = arr.at[slices].set(jnp.asarray(value, dtype=arr.dtype))


def _zero_2d_names(fields: dict[str, Any], names: Sequence[str], slices: tuple[slice, slice]) -> None:
    for name in names:
        _set_2d_constant(fields, name, slices, 0.0)


def _zero_3d_names(fields: dict[str, Any], names: Sequence[str], slices: tuple[slice, slice, slice]) -> None:
    for name in names:
        _set_3d_constant(fields, name, slices, 0.0)


def _copy_2d_name(fields: dict[str, Any], source: str, target: str, slices: tuple[slice, slice]) -> None:
    if source in fields and target in fields and fields[source] is not None and fields[target] is not None:
        fields[target] = _copy_2d_tile(fields[source], fields[target], slices)


def _ncdn40_noah_ivgtyp(ivgtyp: Any, ims: int, jms: int, its: int, ite: int, jts: int, jte: int) -> Any:
    ivgtyp_out = jnp.asarray(ivgtyp)
    slices = _tile2d_direct(ims, jms, its, ite, jts, jte)
    tile = ivgtyp_out[slices]
    tile = jnp.where((tile == 23) | (tile == 25), jnp.asarray(24, dtype=tile.dtype), tile)
    return ivgtyp_out.at[slices].set(tile)


def landuse_init(
    lu_index: Any,
    snowc: Any,
    albedo: Any,
    albbck: Any,
    snoalb: Any,
    mavail: Any,
    emiss: Any,
    embck: Any,
    znt: Any,
    z0: Any,
    thc: Any,
    xland: Any,
    xice: Any,
    xicem: Any,
    julday: int,
    cen_lat: float,
    iswater: int,
    mminlu: str,
    isice: int,
    lucats: int,
    luseas: int,
    isn: int,
    fractional_seaice: int,
    lu_state: Any,
    allowed_to_read: bool,
    usemonalb: bool,
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
    landuse_tables: Mapping[str, Mapping[str, Any]] | None = None,
    fatal: Any | None = None,
) -> LanduseInitResult:
    """Initialize land-use dependent surface fields from WRF ``lu_state``."""

    del ids, jds, kds, kde, kms, kme, kts, kte
    fatal_fn = (lambda message: (_ for _ in ()).throw(RuntimeError(message))) if fatal is None else fatal
    messages: list[str] = []
    xice_threshold = 0.5 if int(fractional_seaice) == 0 else 0.02

    scfx, albd, slmo, sfem, sfz0, sfhc, therin = _unpack_landuse_state(lu_state)
    isn_out = 1
    if int(julday) < 105 or int(julday) > 288:
        isn_out = 2
    if float(cen_lat) < 0.0:
        isn_out = 3 - isn_out

    found_lu = True
    lun = int(lucats)
    nsn = -1
    if bool(allowed_to_read):
        messages.append(f'INPUT LandUse = "{str(mminlu).strip()}"')
        found_lu = False
        table = None if landuse_tables is None else landuse_tables.get(str(mminlu).strip())
        if table is not None:
            lucats, luseas = _copy_table_into_landuse_arrays(
                table, albd, slmo, sfem, sfz0, sfhc, therin, scfx
            )
            lun = int(lucats)
            nsn = int(luseas)
            found_lu = True
            if nsn == 1:
                isn_out = 1
        else:
            messages.append("LANDUSE IN INPUT FILE DOES NOT MATCH LUTABLE: TABLE NOT USED")

    lu_index_arr = np.asarray(lu_index)
    snowc_arr = np.asarray(snowc)
    snoalb_arr = np.asarray(snoalb)
    xice_arr = np.asarray(xice)
    albedo_out = np.asarray(albedo).copy()
    albbck_out = np.asarray(albbck).copy()
    mavail_out = np.asarray(mavail).copy()
    emiss_out = np.asarray(emiss).copy()
    embck_out = np.asarray(embck).copy()
    znt_out = np.asarray(znt).copy()
    z0_out = np.asarray(z0).copy()
    thc_out = np.asarray(thc).copy()
    xland_out = np.asarray(xland).copy()
    xicem_out = np.asarray(xicem).copy()

    if found_lu:
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        season = int(isn_out) - 1
        for j in range(int(jts), jtf + 1):
            jj = j - int(jms)
            for i in range(int(its), itf + 1):
                ii = i - int(ims)
                landuse = _nint_fortran(float(lu_index_arr[ii, jj]))
                if bool(allowed_to_read) and (landuse < 0 or landuse > lun):
                    fatal_fn(f"ERROR: LANDUSE OUTSIDE RANGE ={landuse} AT {i},{j} LUN= {lun}")
                if landuse == 0:
                    landuse = int(iswater)
                cat = landuse - 1
                if not bool(usemonalb):
                    albbck_out[ii, jj] = albd[cat, season] / 100.0
                albedo_out[ii, jj] = albbck_out[ii, jj]
                if snowc_arr[ii, jj] > 0.5:
                    if bool(usemonalb):
                        albedo_out[ii, jj] = snoalb_arr[ii, jj]
                    else:
                        albedo_out[ii, jj] = albbck_out[ii, jj] * (1.0 + scfx[cat])
                thc_out[ii, jj] = therin[cat, season] / 100.0
                z0_out[ii, jj] = sfz0[cat, season] / 100.0
                znt_out[ii, jj] = z0_out[ii, jj]
                embck_out[ii, jj] = sfem[cat, season]
                emiss_out[ii, jj] = embck_out[ii, jj]
                mavail_out[ii, jj] = slmo[cat, season]
                xland_out[ii, jj] = 1.0 if landuse != int(iswater) else 2.0
                xicem_out[ii, jj] = xice_arr[ii, jj]
                if xice_arr[ii, jj] >= xice_threshold:
                    ice_cat = int(isice) - 1
                    xland_out[ii, jj] = 1.0
                    albbck_out[ii, jj] = albd[ice_cat, season] / 100.0
                    embck_out[ii, jj] = sfem[ice_cat, season]
                    if int(fractional_seaice) == 1:
                        albedo_out[ii, jj] = xice_arr[ii, jj] * albbck_out[ii, jj] + (1.0 - xice_arr[ii, jj]) * 0.08
                        emiss_out[ii, jj] = xice_arr[ii, jj] * embck_out[ii, jj] + (1.0 - xice_arr[ii, jj]) * 0.98
                    else:
                        albedo_out[ii, jj] = albbck_out[ii, jj]
                        emiss_out[ii, jj] = embck_out[ii, jj]
                    thc_out[ii, jj] = therin[ice_cat, season] / 100.0
                    z0_out[ii, jj] = sfz0[ice_cat, season] / 100.0
                    znt_out[ii, jj] = z0_out[ii, jj]
                    mavail_out[ii, jj] = slmo[ice_cat, season]

    packed_state = _pack_landuse_state(lu_state, scfx, albd, slmo, sfem, sfz0, sfhc, therin)
    return LanduseInitResult(
        albedo=jnp.asarray(albedo_out),
        albbck=jnp.asarray(albbck_out),
        snoalb=jnp.asarray(snoalb),
        mavail=jnp.asarray(mavail_out),
        emiss=jnp.asarray(emiss_out),
        embck=jnp.asarray(embck_out),
        znt=jnp.asarray(znt_out),
        z0=jnp.asarray(z0_out),
        thc=jnp.asarray(thc_out),
        xland=jnp.asarray(xland_out),
        xicem=jnp.asarray(xicem_out),
        isice=int(isice),
        lucats=int(lucats),
        luseas=int(luseas),
        isn=int(isn_out),
        fractional_seaice=int(fractional_seaice),
        lu_state=jnp.asarray(packed_state),
        messages=tuple(messages),
    )


def phy_init(
    *,
    id: int,
    config_flags: Any,
    dt: float,
    restart: bool,
    zfull: Any,
    zhalf: Any,
    p_top: float,
    radt: float,
    bldt: float,
    cudt: float,
    mpdt: float,
    fgdt: float = 0.0,
    stepbl: int = 0,
    stepra: int = 0,
    stepcu: int = 0,
    stepfg: int = 0,
    warm_rain: bool = False,
    adv_moist_cond: bool = True,
    is_cammgmp_used: bool = False,
    fields: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | Any | None = None,
    moved: bool = False,
    allowed_to_read: bool = True,
    start_of_simulation: bool = True,
    pptop: float = 0.0,
    swrad_scat: float = 0.0,
    cen_lat: float | None = None,
    julyr: int = 0,
    julday: int = 1,
    gmt: float = 0.0,
    levsiz: int = 0,
    n_ozmixm: int = 0,
    alevsiz: int = 0,
    no_src_types: int = 0,
    dx: float = 0.0,
    dy: float = 0.0,
    num_soil_layers: int = 0,
    itimestep: int = 0,
    iswater: int | None = None,
    isice: int | None = None,
    isurban: int | None = None,
    mminlu: str | None = None,
    usemonalb: bool = False,
    landuse_lucats: int = 0,
    landuse_luseas: int = 0,
    landuse_isn: int = 0,
    landuse_tables: Mapping[str, Mapping[str, Any]] | None = None,
    has_reqc: int = 0,
    has_reqi: int = 0,
    has_reqs: int = 0,
    ids: int = 1,
    ide: int = 1,
    jds: int = 1,
    jde: int = 1,
    kds: int = 1,
    kde: int = 1,
    ims: int = 1,
    ime: int = 1,
    jms: int = 1,
    jme: int = 1,
    kms: int = 1,
    kme: int = 1,
    its: int = 1,
    ite: int = 1,
    jts: int = 1,
    jte: int = 1,
    kts: int = 1,
    kte: int = 1,
) -> PhyInitResult:
    """Initialize WRF physics state and delegate to translated scheme initializers.

    Arrays are passed through ``fields`` by their lower-case WRF variable name.
    This keeps Fortran ``intent(inout)`` state explicit while avoiding hidden
    module mutation in the Python/JAX version.
    """

    state: dict[str, Any] = dict(fields or {})
    model = {} if model_config is None else model_config
    messages: list[str] = ["top of phy_init"]
    external_calls: list[str] = []
    subresults: dict[str, Any] = {}

    if cen_lat is None:
        cen_lat = float(_read(config_flags, "cen_lat", 0.0))
    iswater = int(_read(config_flags, "iswater", 16) if iswater is None else iswater)
    isice = int(_read(config_flags, "isice", 24) if isice is None else isice)
    isurban = int(_read(config_flags, "isurban", 1) if isurban is None else isurban)
    mminlu_loc = str(_read(config_flags, "mminlu", "    ") if mminlu is None else mminlu)

    landuse_isice = int(isice)
    icloud_cu = 0
    cu_physics = _as_domain_value(_read(config_flags, "cu_physics", 0))
    shcu_physics = _as_domain_value(_read(config_flags, "shcu_physics", 0))
    mp_physics = _as_domain_value(_read(config_flags, "mp_physics", 0))
    bl_pbl_physics = _as_domain_value(_read(config_flags, "bl_pbl_physics", 0))
    ra_lw_physics = _as_domain_value(_read(config_flags, "ra_lw_physics", 0))
    ra_sw_physics = _as_domain_value(_read(config_flags, "ra_sw_physics", 0))
    sf_sfclay_physics = _as_domain_value(_read(config_flags, "sf_sfclay_physics", 0))

    if bool(_read(config_flags, "cu_rad_feedback", False)):
        if cu_physics == KFETASCHEME:
            icloud_cu = 2
        elif cu_physics in (GFSCHEME, G3SCHEME, GDSCHEME):
            icloud_cu = 1
    if cu_physics == MSKFSCHEME:
        icloud_cu = 2

    tile2d = _tile2d_bounded(ide, jde, ims, jms, its, ite, jts, jte)
    tile3d = _tile3d(ide, jde, kde, ims, jms, kms, its, ite, jts, jte, kts, kte)
    tile3d_full_k = _full_k_tile(ide, jde, ims, jms, kms, kme, its, ite, jts, jte)

    if _as_domain_value(_read(config_flags, "aercu_opt", 0)) == 1:
        _zero_3d_names(
            state,
            ("qc_cu", "qi_cu", "qr_cu", "qs_cu", "nc_cu", "ni_cu", "nr_cu", "ns_cu", "ccn_cu"),
            tile3d,
        )
        for name, value in (
            ("efcg", 2.51),
            ("efcs", 2.51),
            ("efis", 5.01),
            ("efig", 5.01),
            ("efsg", 10.01),
            ("efss", 10.01),
        ):
            _set_3d_constant(state, name, tile3d, value)

    if not bool(restart):
        if (cu_physics == CAMZMSCHEME or shcu_physics == CAMUWSHCUSCHEME) and "rliq" in state:
            state["rliq"] = jnp.asarray(state["rliq"]) * jnp.asarray(0.0, dtype=jnp.asarray(state["rliq"]).dtype)

        if not bool(moved):
            _set_2d_constant(state, "xland", tile2d, float(_read(config_flags, "ideal_xland", 1)))
            _zero_2d_names(
                state,
                ("gsw", "glw", "mol", "pblh", "hfx", "qfx", "rainbl", "rainncv", "snowncv", "graupelncv", "acsnow"),
                tile2d,
            )
            _set_2d_constant(state, "ust", tile2d, 0.0001)
            _zero_3d_names(state, ("exch_h",), tile3d_full_k)

        if "ivgtyp" in state and "lu_index" in state:
            ivgtyp_arr = jnp.asarray(state["ivgtyp"])
            lu_tile = _nint_jax(jnp.asarray(state["lu_index"])[tile2d]).astype(ivgtyp_arr.dtype)
            state["ivgtyp"] = ivgtyp_arr.at[tile2d].set(lu_tile)

        if "tmn_update" in state and int(np.asarray(state["tmn_update"]).item()) == 1 and "tmn" in state:
            tmn_tile = jnp.asarray(state["tmn"])[tile2d]
            for name in ("tyr", "tlag"):
                if name in state:
                    arr = jnp.asarray(state[name])
                    if arr.ndim == 2:
                        state[name] = arr.at[tile2d].set(tmn_tile)
                    elif arr.ndim == 3:
                        state[name] = arr.at[tile2d[0], :, tile2d[1]].set(tmn_tile[:, None, :])
            _zero_2d_names(state, ("tyra", "tdly"), tile2d)

        if "xland" in state:
            xland_tile = jnp.asarray(state["xland"])[tile2d]
            land = xland_tile < 1.5
            if "albbck" in state and mminlu_loc.strip() == "":
                albbck = jnp.asarray(state["albbck"])
                state["albbck"] = albbck.at[tile2d].set(jnp.where(land, 0.2, 0.08).astype(albbck.dtype))
            if "embck" in state:
                embck = jnp.asarray(state["embck"])
                state["embck"] = embck.at[tile2d].set(jnp.where(land, 0.85, 0.98).astype(embck.dtype))
            if "albedo" in state and "albbck" in state:
                state["albedo"] = _copy_2d_tile(state["albbck"], state["albedo"], tile2d)
            if "emiss" in state and "embck" in state:
                state["emiss"] = _copy_2d_tile(state["embck"], state["emiss"], tile2d)
            _set_2d_constant(state, "thc", tile2d, 0.04)
            for name in ("znt", "z0"):
                if name in state:
                    arr = jnp.asarray(state[name])
                    state[name] = arr.at[tile2d].set(jnp.where(land, 0.1, 0.0001).astype(arr.dtype))
            if "mavail" in state:
                mavail = jnp.asarray(state["mavail"])
                state["mavail"] = mavail.at[tile2d].set(jnp.where(land, 0.3, 1.0).astype(mavail.dtype))

        if _as_domain_value(_read(config_flags, "cu_diag", 0)) == 1:
            _zero_3d_names(state, ("gd_cloud", "gd_cloud2", "gd_cloud_a", "gd_cloud2_a", "qc_cu", "qi_cu"), tile3d)

        _zero_2d_names(state, ("raincv_a", "raincv_b"), tile2d)

        if int(has_reqc) != 0:
            _set_3d_constant(state, "re_cloud", tile3d, float(RE_QC_BG))
        if int(has_reqi) != 0:
            _set_3d_constant(state, "re_ice", tile3d, float(RE_QI_BG))
        if int(has_reqs) != 0:
            _set_3d_constant(state, "re_snow", tile3d, float(RE_QS_BG))

        if ra_lw_physics == GODDARDLWSCHEME or ra_sw_physics == GODDARDSWSCHEME or mp_physics == NUWRF4ICESCHEME:
            _zero_3d_names(
                state,
                ("re_cloud_gsfc", "re_rain_gsfc", "re_snow_gsfc", "re_ice_gsfc", "re_graupel_gsfc", "re_hail_gsfc"),
                tile3d,
            )

        if sf_sfclay_physics in (MYJSFCSCHEME, QNSESFCSCHEME):
            _copy_2d_name(state, "u10", "u10e", tile2d)
            _copy_2d_name(state, "v10", "v10e", tile2d)

        if mminlu_loc.strip() != "":
            needed = (
                "lu_index",
                "snowc",
                "albedo",
                "albbck",
                "snoalb",
                "mavail",
                "emiss",
                "embck",
                "znt",
                "z0",
                "thc",
                "xland",
                "xice",
                "xicem",
                "lu_state",
            )
            missing = [name for name in needed if name not in state]
            if missing:
                raise ValueError(f"phy_init landuse_init missing fields: {', '.join(missing)}")
            landuse = landuse_init(
                state["lu_index"],
                state["snowc"],
                state["albedo"],
                state["albbck"],
                state["snoalb"],
                state["mavail"],
                state["emiss"],
                state["embck"],
                state["znt"],
                state["z0"],
                state["thc"],
                state["xland"],
                state["xice"],
                state["xicem"],
                int(julday),
                float(cen_lat),
                iswater,
                mminlu_loc.strip(),
                landuse_isice,
                int(landuse_lucats),
                int(landuse_luseas),
                int(landuse_isn),
                _as_domain_value(_read(config_flags, "fractional_seaice", 0)),
                state["lu_state"],
                bool(allowed_to_read),
                bool(usemonalb),
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
                landuse_tables=landuse_tables,
            )
            subresults["landuse_init"] = landuse
            external_calls.append("landuse_init")
            messages.extend(landuse.messages)
            for name in ("albedo", "albbck", "snoalb", "mavail", "emiss", "embck", "znt", "z0", "thc", "xland", "xicem", "lu_state"):
                state[name] = getattr(landuse, name)
            landuse_isice = landuse.isice
            landuse_lucats = landuse.lucats
            landuse_luseas = landuse.luseas
            landuse_isn = landuse.isn

    sfull, shalf, pptop_out = _z2sigma(
        zfull,
        zhalf,
        float(p_top),
        config_flags,
        bool(allowed_to_read),
        kds,
        kde,
        kms,
        kme,
        kts,
        kte,
    )
    del pptop

    is_cammgmp_used_out = mp_physics == CAMMGMPSCHEME
    if is_cammgmp_used_out:
        is_cammgmp_used = True
    if (
        bl_pbl_physics == CAMUWPBLSCHEME
        or cu_physics == CAMZMSCHEME
        or shcu_physics == CAMUWSHCUSCHEME
        or mp_physics == CAMMGMPSCHEME
    ):
        external_calls.append("cam_init")

    if _as_domain_value(_read(config_flags, "windfarm_opt", 0)) == 1:
        external_calls.append("init_module_wind_fitch")

    def require(names: Sequence[str], stage: str) -> None:
        missing = [name for name in names if name not in state]
        if missing:
            raise ValueError(f"phy_init {stage} missing fields: {', '.join(missing)}")

    require(("rthraten", "rthratenlw", "rthratensw", "cldfra", "emiss", "xlat", "xlong"), "ra_init")
    ra = _ra_init(
        id,
        stepra,
        radt,
        dt,
        state["rthraten"],
        state["rthratenlw"],
        state["rthratensw"],
        state["cldfra"],
        state["emiss"],
        float(cen_lat),
        int(julyr),
        int(julday),
        float(gmt),
        int(levsiz),
        state["xlat"],
        state["xlong"],
        int(n_ozmixm),
        int(alevsiz),
        int(no_src_types),
        sfull,
        shalf,
        float(pptop_out),
        float(swrad_scat),
        float(p_top),
        config_flags,
        bool(restart),
        bool(allowed_to_read),
        bool(start_of_simulation),
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
        cldfra_dp=state.get("cldfra_dp"),
        cldfra_sh=state.get("cldfra_sh"),
        cldfra_old=state.get("cldfra_old"),
        ghi_accum=state.get("ghi_accum"),
    )
    subresults["ra_init"] = ra
    external_calls.append("ra_init")
    stepra = ra.stepra
    for name in ("rthraten", "rthratenlw", "rthratensw", "cldfra", "emiss", "cldfra_dp", "cldfra_sh", "cldfra_old", "ghi_accum"):
        value = getattr(ra, name)
        if value is not None:
            state[name] = value

    require(
        (
            "rublten",
            "rvblten",
            "rthblten",
            "rqvblten",
            "rqcblten",
            "rqiblten",
            "tsk",
            "tmn",
            "ust",
            "lowlyr",
            "tslb",
            "zs",
            "dzs",
            "tke_pbl",
            "exch_h",
            "vegfra",
            "snow",
            "snowc",
            "canwat",
            "smstav",
            "smstot",
            "sfcrunoff",
            "udrunoff",
            "acsnow",
            "acsnom",
            "ivgtyp",
            "isltyp",
            "smois",
            "smfr3d",
            "mavail",
            "snowh",
            "sh2o",
            "snoalb",
            "znt",
            "xland",
            "xice",
        ),
        "bl_init",
    )
    bl = bl_init(
        stepbl,
        bldt,
        dt,
        state["rublten"],
        state["rvblten"],
        state["rthblten"],
        state["rqvblten"],
        state["rqcblten"],
        state["rqiblten"],
        state["tsk"],
        state["tmn"],
        config_flags,
        bool(restart),
        state["ust"],
        state["lowlyr"],
        state["tslb"],
        state["zs"],
        state["dzs"],
        int(num_soil_layers),
        state["tke_pbl"],
        state["exch_h"],
        state["vegfra"],
        state["snow"],
        state["snowc"],
        state["canwat"],
        state["smstav"],
        state["smstot"],
        state["sfcrunoff"],
        state["udrunoff"],
        state["acsnow"],
        state["acsnom"],
        state["ivgtyp"],
        state["isltyp"],
        isurban,
        state["smois"],
        state["smfr3d"],
        state["mavail"],
        state["snowh"],
        state["sh2o"],
        state["snoalb"],
        state["znt"],
        state["xland"],
        state["xice"],
        mminlu_loc.strip(),
        bool(allowed_to_read),
        bool(start_of_simulation),
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
        achfx=state.get("achfx"),
        aclhf=state.get("aclhf"),
        acgrdflx=state.get("acgrdflx"),
        tsk_save=state.get("tsk_save"),
        sf_urban_physics=_as_domain_value(_read(config_flags, "sf_urban_physics", 0)),
        irr_rand_field=state.get("irr_rand_field"),
        irr_ph=_as_domain_value(_read(config_flags, "irr_ph", 0)),
        irr_freq=_as_domain_value(_read(config_flags, "irr_freq", 1), 1),
        sfcevp=state.get("sfcevp"),
    )
    subresults["bl_init"] = bl
    external_calls.append("bl_init")
    external_calls.extend(bl.external_calls)
    messages.extend(bl.messages)
    stepbl = bl.stepbl
    for name in (
        "rublten",
        "rvblten",
        "rthblten",
        "rqvblten",
        "rqcblten",
        "rqiblten",
        "tke_pbl",
        "exch_h",
        "ust",
        "lowlyr",
        "znt",
        "vegfra",
        "snow",
        "snowc",
        "snowh",
        "canwat",
        "smstav",
        "smstot",
        "sfcrunoff",
        "udrunoff",
        "acsnow",
        "acsnom",
        "ivgtyp",
        "isltyp",
        "tslb",
        "smois",
        "sh2o",
        "zs",
        "dzs",
        "snoalb",
        "sfcevp",
        "achfx",
        "aclhf",
        "acgrdflx",
        "tsk_save",
        "irr_rand_field",
    ):
        value = getattr(bl, name)
        if value is not None:
            state[name] = value

    require(("rainc", "raincv"), "cu_init")
    cu = _cu_init(
        dx,
        stepcu,
        cudt,
        dt,
        state["rainc"],
        state["raincv"],
        config_flags,
        bool(restart),
        bool(allowed_to_read),
        bool(start_of_simulation),
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
    subresults["cu_init"] = cu
    external_calls.append("cu_init")
    external_calls.extend(cu.external_calls)
    stepcu = cu.stepcu
    state["rainc"] = cu.rainc
    state["raincv"] = cu.raincv

    shcu = _shcu_init(
        stepcu,
        cudt,
        dt,
        state["rainc"],
        state["raincv"],
        config_flags,
        bool(restart),
        bool(allowed_to_read),
        bool(start_of_simulation),
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
    subresults["shcu_init"] = shcu
    external_calls.append("shcu_init")
    external_calls.extend(shcu.external_calls)
    stepcu = shcu.stepcu
    state["rainc"] = shcu.rainc
    state["raincv"] = shcu.raincv

    require(("rainnc", "snownc", "graupelnc"), "mp_init")
    mp = _mp_init(
        state["rainnc"],
        state["snownc"],
        state["graupelnc"],
        config_flags,
        bool(restart),
        mpdt,
        dt,
        dx,
        dy,
        bool(allowed_to_read),
        bool(start_of_simulation),
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
    subresults["mp_init"] = mp
    external_calls.append("mp_init")
    external_calls.extend(mp.external_calls)
    warm_rain = mp.warm_rain
    adv_moist_cond = mp.adv_moist_cond
    state["rainnc"] = mp.rainnc
    state["snownc"] = mp.snownc
    state["graupelnc"] = mp.graupelnc

    fg = _fg_init(
        stepfg,
        fgdt,
        dt,
        id,
        config_flags,
        bool(restart),
        bool(allowed_to_read),
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
    subresults["fg_init"] = fg
    external_calls.append("fg_init")
    external_calls.extend(fg.external_calls)
    stepfg = fg.stepfg

    obs_nudge_opt = _read(model, "obs_nudge_opt", [0])
    max_dom = _as_domain_value(_read(model, "max_dom", 1), 1)
    fdob = _fdob_init(
        obs_nudge_opt,
        max_dom,
        id,
        _read(model, "parent_id", [1]),
        _as_domain_value(_read(model, "obs_idynin", 0)),
        float(_read(model, "obs_dtramp", 0.0)),
        _read(model, "fdda_end", [0.0]),
        bool(_read(model, "restart", restart)),
        float(_read(model, "obs_twindo_cg", 0.0)),
        float(_read(model, "obs_twindo", 0.0)),
        int(itimestep),
        int(_read(model, "start_year", [julyr])[int(id) - 1] if isinstance(_read(model, "start_year", [julyr]), Sequence) else _read(model, "start_year", julyr)),
        int(_read(model, "start_month", [1])[int(id) - 1] if isinstance(_read(model, "start_month", [1]), Sequence) else _read(model, "start_month", 1)),
        int(_read(model, "start_day", [julday])[int(id) - 1] if isinstance(_read(model, "start_day", [julday]), Sequence) else _read(model, "start_day", julday)),
        int(_read(model, "start_hour", [0])[int(id) - 1] if isinstance(_read(model, "start_hour", [0]), Sequence) else _read(model, "start_hour", 0)),
        int(_read(model, "start_minute", [0])[int(id) - 1] if isinstance(_read(model, "start_minute", [0]), Sequence) else _read(model, "start_minute", 0)),
        int(_read(model, "start_second", [0])[int(id) - 1] if isinstance(_read(model, "start_second", [0]), Sequence) else _read(model, "start_second", 0)),
        state.get("fdob"),
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
    subresults["fdob_init"] = fdob
    external_calls.append("fdob_init")
    external_calls.extend(fdob.external_calls)
    state["fdob"] = fdob.fdob

    return PhyInitResult(
        fields=state,
        stepbl=int(stepbl),
        stepra=int(stepra),
        stepcu=int(stepcu),
        stepfg=int(stepfg),
        warm_rain=bool(warm_rain),
        adv_moist_cond=bool(adv_moist_cond),
        is_cammgmp_used=bool(is_cammgmp_used),
        icloud_cu=int(icloud_cu),
        landuse_isice=int(landuse_isice),
        landuse_lucats=int(landuse_lucats),
        landuse_luseas=int(landuse_luseas),
        landuse_isn=int(landuse_isn),
        sfull=sfull,
        shalf=shalf,
        pptop=float(pptop_out),
        subresults=subresults,
        config_updates={"icloud_cu": int(icloud_cu)},
        messages=tuple(messages),
        external_calls=tuple(external_calls),
    )


def bl_init(
    stepbl: int,
    bldt: float,
    dt: float,
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    tsk: Any,
    tmn: Any,
    config_flags: Any,
    restart: bool,
    ust: Any,
    lowlyr: Any,
    tslb: Any,
    zs: Any,
    dzs: Any,
    num_soil_layers: int,
    tke_pbl: Any,
    exch_h: Any,
    vegfra: Any,
    snow: Any,
    snowc: Any,
    canwat: Any,
    smstav: Any,
    smstot: Any,
    sfcrunoff: Any,
    udrunoff: Any,
    acsnow: Any,
    acsnom: Any,
    ivgtyp: Any,
    isltyp: Any,
    isurban: int,
    smois: Any,
    smfr3d: Any,
    mavail: Any,
    snowh: Any,
    sh2o: Any,
    snoalb: Any,
    znt: Any,
    xland: Any,
    xice: Any,
    mminlu: str,
    allowed_to_read: bool,
    start_of_simulation: bool,
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
    achfx: Any | None = None,
    aclhf: Any | None = None,
    acgrdflx: Any | None = None,
    tsk_save: Any | None = None,
    sf_urban_physics: int = 0,
    irr_rand_field: Any | None = None,
    irr_ph: int = 0,
    irr_freq: int = 1,
    parameters: Any | None = None,
    table_dir: str | Path = ".",
    lsminit_fn: Any | None = None,
    myjsfcinit_fn: Any | None = None,
    myjpblinit_fn: Any | None = None,
    **carried_arrays: Any,
) -> BlInitResult:
    """Initialize surface-layer, land-surface, and PBL state for one WRF tile."""

    del stepbl, tmn, isurban, smfr3d, mavail, start_of_simulation

    result_stepbl = max(_nint_positive(float(bldt) * 60.0 / float(dt)), 1)
    messages: list[str] = []
    external_calls: list[str] = []

    rublten_out = jnp.asarray(rublten)
    rvblten_out = jnp.asarray(rvblten)
    rthblten_out = jnp.asarray(rthblten)
    rqvblten_out = jnp.asarray(rqvblten)
    rqcblten_out = jnp.asarray(rqcblten)
    rqiblten_out = jnp.asarray(rqiblten)
    tke_pbl_out = jnp.asarray(tke_pbl)
    exch_h_out = jnp.asarray(exch_h)
    ust_out = jnp.asarray(ust)
    lowlyr_out = jnp.asarray(lowlyr)
    znt_out = jnp.asarray(znt)
    vegfra_out = jnp.asarray(vegfra)
    snow_out = jnp.asarray(snow)
    snowc_out = jnp.asarray(snowc)
    snowh_out = jnp.asarray(snowh)
    canwat_out = jnp.asarray(canwat)
    smstav_out = jnp.asarray(smstav)
    smstot_out = jnp.asarray(smstot)
    sfcrunoff_out = jnp.asarray(sfcrunoff)
    udrunoff_out = jnp.asarray(udrunoff)
    acsnow_out = jnp.asarray(acsnow)
    acsnom_out = jnp.asarray(acsnom)
    ivgtyp_out = jnp.asarray(ivgtyp)
    isltyp_out = jnp.asarray(isltyp)
    tslb_out = jnp.asarray(tslb)
    smois_out = jnp.asarray(smois)
    sh2o_out = jnp.asarray(sh2o)
    zs_out = jnp.asarray(zs)
    dzs_out = jnp.asarray(dzs)
    snoalb_out = jnp.asarray(snoalb)
    sfcevp_out = jnp.asarray(carried_arrays.pop("sfcevp", None)) if "sfcevp" in carried_arrays else None
    if sfcevp_out is None:
        sfcevp_out = jnp.zeros_like(jnp.asarray(achfx)) if achfx is not None else jnp.asarray(snow) * 0.0
    achfx_out = None if achfx is None else jnp.asarray(achfx)
    aclhf_out = None if aclhf is None else jnp.asarray(aclhf)
    acgrdflx_out = None if acgrdflx is None else jnp.asarray(acgrdflx)
    tsk_save_out = None if tsk_save is None else jnp.asarray(tsk_save)
    irr_rand_out = None if irr_rand_field is None else jnp.asarray(irr_rand_field)
    lsm_parameters = parameters

    direct2d = _tile2d_direct(ims, jms, its, ite, jts, jte)
    if achfx is not None and not bool(restart):
        if aclhf is None or acgrdflx is None:
            raise ValueError("aclhf and acgrdflx are required when achfx is present")
        achfx_out = _zero_2d_tile(achfx_out, direct2d)
        aclhf_out = _zero_2d_tile(aclhf_out, direct2d)
        acgrdflx_out = _zero_2d_tile(acgrdflx_out, direct2d)
        sfcevp_out = _zero_2d_tile(sfcevp_out, direct2d)

    if tsk_save is not None and not bool(restart) and _as_domain_value(_read(config_flags, "fractional_seaice", 0)) == 1:
        tsk_save_out = _copy_2d_tile(tsk, tsk_save_out, direct2d)

    isfc = 0
    sf_sfclay_physics = _as_domain_value(_read(config_flags, "sf_sfclay_physics", 0))
    if sf_sfclay_physics == MYJSFCSCHEME:
        init_fn = myjsfcinit if myjsfcinit_fn is None else myjsfcinit_fn
        myj_sfc = init_fn(
            lowlyr_out,
            ust_out,
            znt_out,
            xland,
            xice,
            ivgtyp_out,
            restart,
            allowed_to_read,
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
        lowlyr_out = myj_sfc.lowlyr
        ust_out = myj_sfc.ustar
        znt_out = myj_sfc.z0
        isfc = 2
        external_calls.append("myjsfcinit")
    elif sf_sfclay_physics not in (0, -1):
        raise NotImplementedError(f"bl_init sf_sfclay_physics={sf_sfclay_physics} is not translated")

    sf_surface_physics = _as_domain_value(_read(config_flags, "sf_surface_physics", 0))
    if sf_surface_physics == LSMSCHEME:
        if str(mminlu).strip() == "NLCD40":
            ivgtyp_out = _ncdn40_noah_ivgtyp(ivgtyp_out, ims, jms, its, ite, jts, jte)
            messages.append("Using NLCD40 for Noah, redefine urban categories")
        init_fn = lsminit if lsminit_fn is None else lsminit_fn
        if irr_rand_out is None:
            irr_rand_out = jnp.zeros_like(ivgtyp_out)
        lsm = init_fn(
            vegfra_out,
            snow_out,
            snowc_out,
            snowh_out,
            canwat_out,
            smstav_out,
            smstot_out,
            sfcrunoff_out,
            udrunoff_out,
            acsnow_out,
            acsnom_out,
            ivgtyp_out,
            isltyp_out,
            tslb_out,
            smois_out,
            sh2o_out,
            zs_out,
            dzs_out,
            mminlu,
            snoalb_out,
            bool(carried_arrays.pop("fndsoilw", True)),
            bool(carried_arrays.pop("fndsnowh", True)),
            bool(carried_arrays.pop("rdmaxalb", False)),
            num_soil_layers,
            restart,
            allowed_to_read,
            irr_rand_out,
            irr_ph,
            irr_freq,
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
            parameters=lsm_parameters,
            table_dir=table_dir,
            soil_veg_gen_parm_fn=carried_arrays.pop("soil_veg_gen_parm_fn", None),
            random_values=carried_arrays.pop("random_values", None),
        )
        vegfra_out = lsm.vegfra
        snow_out = lsm.snow
        snowc_out = lsm.snowc
        snowh_out = lsm.snowh
        canwat_out = lsm.canwat
        smstav_out = lsm.smstav
        smstot_out = lsm.smstot
        sfcrunoff_out = lsm.sfcrunoff
        udrunoff_out = lsm.udrunoff
        acsnow_out = lsm.acsnow
        acsnom_out = lsm.acsnom
        snoalb_out = lsm.snoalb
        tslb_out = lsm.tslb
        smois_out = lsm.smois
        sh2o_out = lsm.sh2o
        zs_out = lsm.zs
        dzs_out = lsm.dzs
        irr_rand_out = lsm.irr_rand_field
        lsm_parameters = lsm.parameters
        messages.extend(lsm.messages)
        external_calls.append("lsminit")
    elif sf_surface_physics not in (0, -1):
        raise NotImplementedError(f"bl_init sf_surface_physics={sf_surface_physics} is not translated")

    bl_pbl_physics = _as_domain_value(_read(config_flags, "bl_pbl_physics", 0))
    if bl_pbl_physics == MYJPBLSCHEME:
        if isfc != 2:
            raise ValueError("module_physics_init: use myjsfc scheme for this pbl option")
        if int(sf_urban_physics) in (2, 3):
            raise NotImplementedError("bl_init MYJ urban initialization is not translated")
        init_fn = myjpblinit if myjpblinit_fn is None else myjpblinit_fn
        pbl = init_fn(
            rublten_out,
            rvblten_out,
            rthblten_out,
            rqvblten_out,
            tke_pbl_out,
            exch_h_out,
            restart,
            allowed_to_read,
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
        rublten_out = pbl.rublten
        rvblten_out = pbl.rvblten
        rthblten_out = pbl.rthblten
        rqvblten_out = pbl.rqvblten
        tke_pbl_out = pbl.tke_myj
        exch_h_out = pbl.exch_h
        external_calls.append("myjpblinit")
    elif bl_pbl_physics not in (0, -1):
        raise NotImplementedError(f"bl_init bl_pbl_physics={bl_pbl_physics} is not translated")

    return BlInitResult(
        stepbl=result_stepbl,
        rublten=rublten_out,
        rvblten=rvblten_out,
        rthblten=rthblten_out,
        rqvblten=rqvblten_out,
        rqcblten=rqcblten_out,
        rqiblten=rqiblten_out,
        tke_pbl=tke_pbl_out,
        exch_h=exch_h_out,
        ust=ust_out,
        lowlyr=lowlyr_out,
        znt=znt_out,
        vegfra=vegfra_out,
        snow=snow_out,
        snowc=snowc_out,
        snowh=snowh_out,
        canwat=canwat_out,
        smstav=smstav_out,
        smstot=smstot_out,
        sfcrunoff=sfcrunoff_out,
        udrunoff=udrunoff_out,
        acsnow=acsnow_out,
        acsnom=acsnom_out,
        ivgtyp=ivgtyp_out,
        isltyp=isltyp_out,
        tslb=tslb_out,
        smois=smois_out,
        sh2o=sh2o_out,
        zs=zs_out,
        dzs=dzs_out,
        snoalb=snoalb_out,
        sfcevp=sfcevp_out,
        achfx=achfx_out,
        aclhf=aclhf_out,
        acgrdflx=acgrdflx_out,
        tsk_save=tsk_save_out,
        irr_rand_field=irr_rand_out,
        lsm_parameters=lsm_parameters,
        messages=tuple(messages),
        external_calls=tuple(external_calls),
        carried_arrays=dict(carried_arrays),
    )


__all__ = [
    "BlInitResult",
    "PhyInitResult",
    "LanduseInitResult",
    "LSMSCHEME",
    "MYJPBLSCHEME",
    "MYJSFCSCHEME",
    "QNSESFCSCHEME",
    "SFCLAYSCHEME",
    "SFCLAYREVSCHEME",
    "bl_init",
    "landuse_init",
    "phy_init",
    "read_landuse_tables",
]
