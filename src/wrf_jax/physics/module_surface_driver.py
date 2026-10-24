"""Translations from WRF ``phys/module_surface_driver``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


from wrf_jax.core.wrf_math import sqrt_for_wrf
from wrf_jax.physics.myjsfc import A2S, A3S, A4S, PQ0SEA, MyjSfcResult, MyjSfcTables, myjsfc

SF_SFCLAY_NONE = 0
MYJSFCSCHEME = 2
LSMSCHEME = 2

_EXPLICIT_RESULT_FIELDS = frozenset(
    {
        "q_ref2m",
        "u_phytmp",
        "v_phytmp",
        "qgh",
        "chs",
        "cpm",
        "chs2",
        "psfc",
        "ust",
        "ustm",
        "wspd",
        "ch",
        "dtmin",
        "dtbl",
        "bldtacttime",
    }
)


@dataclass(frozen=True)
class SurfaceDriverResult:
    q_ref2m: float
    u_phytmp: Any | None = None
    v_phytmp: Any | None = None
    qgh: Any | None = None
    chs: Any | None = None
    cpm: Any | None = None
    chs2: Any | None = None
    psfc: Any | None = None
    ust: Any | None = None
    ustm: Any | None = None
    wspd: Any | None = None
    ch: Any | None = None
    dtmin: float = 0.0
    dtbl: float = 0.0
    bldtacttime: float | None = None
    carried_fields: Mapping[str, Any] = field(default_factory=dict)
    ran_parameterization: bool = False
    external_calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class MyjSfcSeaIceWrapperResult:
    tsk: Any
    qsfc: Any
    thz0: Any
    qz0: Any
    uz0: Any
    vz0: Any
    sst: Any
    br: Any
    chs_sea: Any
    chs2_sea: Any
    cqs2_sea: Any
    cpm_sea: Any
    flhc_sea: Any
    flqc_sea: Any
    qsfc_sea: Any
    qgh_sea: Any
    qz0_sea: Any
    hfx_sea: Any
    qfx_sea: Any
    flx_lh_sea: Any
    tsk_sea: Any
    ustar: Any
    znt: Any
    pblh: Any
    rmol: Any
    akhs: Any
    akms: Any
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


def _tile_slices(ims: int, jms: int, its: int, ite: int, jts: int, jte: int) -> tuple[slice, slice]:
    return (
        slice(_idx(its, ims), _idx(ite, ims) + 1),
        slice(_idx(jts, jms), _idx(jte, jms) + 1),
    )


def _tile_3d_slices(
    ims: int,
    kms: int,
    jms: int,
    its: int,
    ite: int,
    kts: int,
    kte: int,
    jts: int,
    jte: int,
) -> tuple[slice, slice, slice]:
    return (
        slice(_idx(its, ims), _idx(ite, ims) + 1),
        slice(_idx(kts, kms), _idx(kte, kms) + 1),
        slice(_idx(jts, jms), _idx(jte, jms) + 1),
    )


def _as_array_or_none(value: Any) -> Any | None:
    return None if value is None else jnp.asarray(value)


def _zero_2d_tiles(value: Any | None, ims: int, jms: int, i_start: Any, i_end: Any, j_start: Any, j_end: Any) -> Any | None:
    if value is None:
        return None
    out = jnp.asarray(value)
    for its, ite, jts, jte in zip(i_start, i_end, j_start, j_end, strict=True):
        i_slice, j_slice = _tile_slices(ims, jms, int(its), int(ite), int(jts), int(jte))
        out = out.at[i_slice, j_slice].set(0.0)
    return out


def _zero_3d_tiles(
    value: Any | None,
    ims: int,
    kms: int,
    jms: int,
    i_start: Any,
    i_end: Any,
    j_start: Any,
    j_end: Any,
    k_start: int,
    k_end: int,
) -> Any | None:
    if value is None:
        return None
    out = jnp.asarray(value)
    for its, ite, jts, jte in zip(i_start, i_end, j_start, j_end, strict=True):
        slices = _tile_3d_slices(ims, kms, jms, int(its), int(ite), k_start, k_end, int(jts), int(jte))
        out = out.at[slices].set(0.0)
    return out


def _mapping_from_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "__dataclass_fields__"):
        return {name: getattr(result, name) for name in result.__dataclass_fields__}
    raise TypeError("surface_driver callback must return a mapping, dataclass result, or None")


def _carried_without_explicit_results(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in fields.items() if name not in _EXPLICIT_RESULT_FIELDS}


def accumulate_surface_precipitation(rainbl: Any, raincv: Any,
                                     rainncv: Any = None, rainshv: Any = None) -> Any:
    """WRF precipitation in mm accumulated between surface-physics calls."""
    rain = jnp.asarray(rainbl) + jnp.asarray(raincv)
    if rainncv is not None:
        rain = rain + jnp.asarray(rainncv)
    if rainshv is not None:
        rain = rain + jnp.asarray(rainshv)
    return jnp.maximum(rain, 0.0)


def _surface_should_run(
    *,
    itimestep: int,
    stepbl: int,
    bldt: float | None,
    curr_secs: float | None,
    adapt_step_flag: bool | None,
    bldtacttime: float | None,
) -> tuple[bool, float | None]:
    doing_adapt_dt = bool(adapt_step_flag)
    if int(itimestep) == 1:
        return True, bldtacttime
    if bldt is not None:
        if float(bldt) == 0.0 or int(stepbl) == 1:
            return True, bldtacttime
    elif int(stepbl) == 1:
        return True, bldtacttime
    if (not doing_adapt_dt) and int(itimestep) % int(stepbl) == 0:
        return True, bldtacttime
    if doing_adapt_dt and curr_secs is not None and bldtacttime is not None and float(curr_secs) >= float(bldtacttime):
        return True, bldtacttime
    return False, bldtacttime


def surface_driver(
    *,
    sf_sfclay_physics: int,
    sf_surface_physics: int,
    bl_pbl_physics: int = 0,
    ra_lw_physics: int = 0,
    fractional_seaice: int = 1,
    itimestep: int = 1,
    stepbl: int = 1,
    bldt: float | None = None,
    dt: float = 0.0,
    curr_secs: float | None = None,
    adapt_step_flag: bool | None = None,
    bldtacttime: float | None = None,
    isfflx: int = 1,
    u_phy: Any | None = None,
    v_phy: Any | None = None,
    u_phytmp: Any | None = None,
    v_phytmp: Any | None = None,
    u_frame: float = 0.0,
    v_frame: float = 0.0,
    uoce: Any | None = None,
    voce: Any | None = None,
    p8w: Any | None = None,
    psfc: Any | None = None,
    qgh: Any | None = None,
    chs: Any | None = None,
    cpm: Any | None = None,
    chs2: Any | None = None,
    ust: Any | None = None,
    ustm: Any | None = None,
    wspd: Any | None = None,
    ch: Any | None = None,
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
    i_start: Any = (1,),
    i_end: Any = (1,),
    j_start: Any = (1,),
    j_end: Any = (1,),
    kts: int = 1,
    kte: int = 1,
    myjsfc_fn: Any | None = None,
    lsm_fn: Any | None = None,
    sfcdiags_fn: Any | None = None,
    **fields: Any,
) -> SurfaceDriverResult:
    """Run the supported WRF ``surface_driver`` control path.

    The configured case resolves to MYJ surface layer and Noah LSM through the
    CONUS physics suite. This function translates the driver orchestration and
    keeps scheme kernels explicit through callbacks or already translated lower
    functions. Unsupported schemes are rejected instead of approximated.
    """

    del bl_pbl_physics, ra_lw_physics, ids, ide, jds, jde, kds, kde

    q_ref2m = 0.0
    carried = dict(fields)

    if int(sf_sfclay_physics) == SF_SFCLAY_NONE:
        return SurfaceDriverResult(
            q_ref2m=q_ref2m,
            u_phytmp=u_phytmp,
            v_phytmp=v_phytmp,
            qgh=qgh,
            chs=chs,
            cpm=cpm,
            chs2=chs2,
            psfc=psfc,
            ust=ust,
            ustm=ustm,
            wspd=wspd,
            ch=ch,
            bldtacttime=bldtacttime,
            carried_fields=_carried_without_explicit_results(carried),
            ran_parameterization=False,
        )

    i_start_tuple = tuple(int(value) for value in i_start)
    i_end_tuple = tuple(int(value) for value in i_end)
    j_start_tuple = tuple(int(value) for value in j_start)
    j_end_tuple = tuple(int(value) for value in j_end)

    qgh_out = _zero_2d_tiles(qgh, ims, jms, i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple)
    chs_out = _zero_2d_tiles(chs, ims, jms, i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple)
    cpm_out = _zero_2d_tiles(cpm, ims, jms, i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple)
    chs2_out = _zero_2d_tiles(chs2, ims, jms, i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple)

    if carried.get("rainbl") is not None:
        rain = jnp.asarray(carried["rainbl"])
        accumulated = accumulate_surface_precipitation(
            rain, carried.get("raincv", 0.0), carried.get("rainncv"), carried.get("rainshv"),
        )
        for its, ite, jts, jte in zip(i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple, strict=True):
            tile = _tile_slices(ims, jms, its, ite, jts, jte)
            rain = rain.at[tile].set(accumulated[tile])
        carried["rainbl"] = rain

    run_param, bldtacttime_out = _surface_should_run(
        itimestep=itimestep,
        stepbl=stepbl,
        bldt=bldt,
        curr_secs=curr_secs,
        adapt_step_flag=adapt_step_flag,
        bldtacttime=bldtacttime,
    )
    if not run_param:
        u_phytmp_out = _zero_3d_tiles(
            u_phytmp if u_phytmp is not None else u_phy,
            ims,
            kms,
            jms,
            i_start_tuple,
            i_end_tuple,
            j_start_tuple,
            j_end_tuple,
            kms,
            kme,
        )
        v_phytmp_out = _zero_3d_tiles(
            v_phytmp if v_phytmp is not None else v_phy,
            ims,
            kms,
            jms,
            i_start_tuple,
            i_end_tuple,
            j_start_tuple,
            j_end_tuple,
            kms,
            kme,
        )
        return SurfaceDriverResult(
            q_ref2m=q_ref2m,
            u_phytmp=u_phytmp_out,
            v_phytmp=v_phytmp_out,
            qgh=qgh_out,
            chs=chs_out,
            cpm=cpm_out,
            chs2=chs2_out,
            psfc=psfc,
            ust=ust,
            ustm=ustm,
            wspd=wspd,
            ch=ch,
            bldtacttime=bldtacttime_out,
            carried_fields=_carried_without_explicit_results(carried),
            ran_parameterization=False,
        )

    dtmin = float(dt) / 60.0
    if bldt is not None and float(bldt) == 0.0:
        dtbl = float(dt)
    elif bldt is not None and bool(adapt_step_flag):
        dtbl = float(bldt) * 60.0
    else:
        dtbl = float(dt) * float(stepbl)

    psfc_out = _as_array_or_none(psfc)
    u_phy_arr = _as_array_or_none(u_phy)
    v_phy_arr = _as_array_or_none(v_phy)
    p8w_arr = _as_array_or_none(p8w)
    uoce_arr = _as_array_or_none(uoce)
    voce_arr = _as_array_or_none(voce)

    full_single_tile = (
        len(i_start_tuple) == 1
        and int(i_start_tuple[0]) == int(ims)
        and int(i_end_tuple[0]) == int(ime)
        and int(j_start_tuple[0]) == int(jms)
        and int(j_end_tuple[0]) == int(jme)
        and int(kts) == int(kms)
        and int(kte) == int(kme)
    )
    if full_single_tile and u_phy_arr is not None:
        u_phytmp_out = u_phy_arr + float(u_frame)
        if uoce_arr is not None:
            k0 = _idx(kts, kms)
            u_phytmp_out = u_phytmp_out.at[:, k0, :].add(-uoce_arr)
    else:
        u_phytmp_out = _zero_3d_tiles(
            u_phytmp if u_phytmp is not None else u_phy,
            ims,
            kms,
            jms,
            i_start_tuple,
            i_end_tuple,
            j_start_tuple,
            j_end_tuple,
            kms,
            kme,
        )
    if full_single_tile and v_phy_arr is not None:
        v_phytmp_out = v_phy_arr + float(v_frame)
        if voce_arr is not None:
            k0 = _idx(kts, kms)
            v_phytmp_out = v_phytmp_out.at[:, k0, :].add(-voce_arr)
    else:
        v_phytmp_out = _zero_3d_tiles(
            v_phytmp if v_phytmp is not None else v_phy,
            ims,
            kms,
            jms,
            i_start_tuple,
            i_end_tuple,
            j_start_tuple,
            j_end_tuple,
            kms,
            kme,
        )

    external_calls: list[str] = []
    for its, ite, jts, jte in zip(i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple, strict=True):
        i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
        k_slice = slice(_idx(kts, kms), _idx(kte, kms) + 1)
        k0 = _idx(kts, kms)
        if p8w_arr is not None:
            if psfc_out is None:
                psfc_out = jnp.zeros_like(p8w_arr[:, k0, :])
            psfc_out = psfc_out.at[i_slice, j_slice].set(p8w_arr[i_slice, k0, j_slice])
        if (not full_single_tile) and u_phy_arr is not None and u_phytmp_out is not None:
            u_tile = u_phy_arr[i_slice, k_slice, j_slice] + float(u_frame)
            u_phytmp_out = u_phytmp_out.at[i_slice, k_slice, j_slice].set(u_tile)
            if uoce_arr is not None:
                u_surface = u_phytmp_out[i_slice, k0, j_slice] - uoce_arr[i_slice, j_slice]
                u_phytmp_out = u_phytmp_out.at[i_slice, k0, j_slice].set(u_surface)
        if (not full_single_tile) and v_phy_arr is not None and v_phytmp_out is not None:
            v_tile = v_phy_arr[i_slice, k_slice, j_slice] + float(v_frame)
            v_phytmp_out = v_phytmp_out.at[i_slice, k_slice, j_slice].set(v_tile)
            if voce_arr is not None:
                v_surface = v_phytmp_out[i_slice, k0, j_slice] - voce_arr[i_slice, j_slice]
                v_phytmp_out = v_phytmp_out.at[i_slice, k0, j_slice].set(v_surface)

    ust_out = _as_array_or_none(ust)
    ustm_out = _as_array_or_none(ustm)
    wspd_out = _as_array_or_none(wspd)
    ch_out = _as_array_or_none(ch)

    if int(sf_sfclay_physics) == MYJSFCSCHEME:
        if myjsfc_fn is None:
            raise NotImplementedError("surface_driver MYJ branch requires an explicit myjsfc_fn callback")
        updates = _mapping_from_result(
            myjsfc_fn(
                u_phytmp=u_phytmp_out,
                v_phytmp=v_phytmp_out,
                psfc=psfc_out,
                qgh=qgh_out,
                chs=chs_out,
                cpm=cpm_out,
                chs2=chs2_out,
                ust=ust_out,
                fields=carried,
            )
        )
        qgh_out = updates.get("qgh", qgh_out)
        chs_out = updates.get("chs", chs_out)
        cpm_out = updates.get("cpm", cpm_out)
        chs2_out = updates.get("chs2", chs2_out)
        ust_out = updates.get("ust", ust_out)
        carried.update(updates)
        external_calls.append("MYJSFC")
    else:
        raise NotImplementedError(
            "surface_driver currently supports only sf_sfclay_physics=0 or MYJSFCSCHEME=2; "
            f"got {int(sf_sfclay_physics)}"
        )

    for its, ite, jts, jte in zip(i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple, strict=True):
        i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
        k0 = _idx(kts, kms)
        if ust_out is not None:
            if ustm_out is None:
                ustm_out = jnp.zeros_like(ust_out)
            ustm_out = ustm_out.at[i_slice, j_slice].set(ust_out[i_slice, j_slice])
        if u_phy_arr is not None and v_phy_arr is not None:
            if wspd_out is None:
                wspd_out = jnp.zeros_like(u_phy_arr[:, k0, :])
            speed = jnp.maximum(sqrt_for_wrf(u_phy_arr[i_slice, k0, j_slice] ** 2 + v_phy_arr[i_slice, k0, j_slice] ** 2), 0.001)
            wspd_out = wspd_out.at[i_slice, j_slice].set(speed)
        if chs_out is not None:
            if ch_out is None:
                ch_out = jnp.zeros_like(jnp.asarray(chs_out))
            ch_out = ch_out.at[i_slice, j_slice].set(jnp.asarray(chs_out)[i_slice, j_slice])

    if int(isfflx) != 0:
        if int(sf_surface_physics) == LSMSCHEME:
            if lsm_fn is None:
                raise NotImplementedError("surface_driver Noah LSM branch requires an explicit lsm_fn callback")
            lsm_updates = _mapping_from_result(
                lsm_fn(
                    u_phytmp=u_phytmp_out,
                    v_phytmp=v_phytmp_out,
                    psfc=psfc_out,
                    qgh=qgh_out,
                    chs=chs_out,
                    cpm=cpm_out,
                    chs2=chs2_out,
                    dtbl=dtbl,
                    fields=carried,
                )
            )
            carried.update(lsm_updates)
            external_calls.append("LSM")
            if sfcdiags_fn is not None:
                sfcdiags_updates = _mapping_from_result(
                    sfcdiags_fn(
                        psfc=psfc_out,
                        chs=chs_out,
                        chs2=chs2_out,
                        fields=carried,
                    )
                )
                carried.update(sfcdiags_updates)
                external_calls.append("SFCDIAGS")
        else:
            raise NotImplementedError(
                "surface_driver currently supports only sf_surface_physics=LSMSCHEME=2 when ISFFLX is enabled; "
                f"got {int(sf_surface_physics)}"
            )

    if carried.get("rainbl") is not None:
        carried["rainbl"] = _zero_2d_tiles(
            carried["rainbl"], ims, jms, i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple,
        )

    return SurfaceDriverResult(
        q_ref2m=q_ref2m,
        u_phytmp=u_phytmp_out,
        v_phytmp=v_phytmp_out,
        qgh=qgh_out,
        chs=chs_out,
        cpm=cpm_out,
        chs2=chs2_out,
        psfc=psfc_out,
        ust=ust_out,
        ustm=ustm_out,
        wspd=wspd_out,
        ch=ch_out,
        dtmin=dtmin,
        dtbl=dtbl,
        bldtacttime=bldtacttime_out,
        carried_fields=_carried_without_explicit_results(carried),
        ran_parameterization=True,
        external_calls=tuple(external_calls),
    )


def _tile_psfc(pint: Any, lowlyr: Any, ims: int, jms: int, kms: int, its: int, ite: int, jts: int, jte: int) -> Any:
    i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
    low_idx = jnp.asarray(lowlyr)[i_slice, j_slice].astype(jnp.int32) - int(kms)
    return jnp.take_along_axis(jnp.asarray(pint)[i_slice, :, j_slice], low_idx[:, None, :], axis=1)[:, 0, :]


def _blend_tile(target: Any, sea: Any, xice: Any, ice_mask: Any, ims: int, jms: int, its: int, ite: int, jts: int, jte: int) -> Any:
    i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
    target_arr = jnp.asarray(target)
    target_tile = target_arr[i_slice, j_slice]
    sea_tile = jnp.asarray(sea)[i_slice, j_slice]
    xice_tile = jnp.asarray(xice)[i_slice, j_slice]
    blended = target_tile * xice_tile + (1.0 - xice_tile) * sea_tile
    return target_arr.at[i_slice, j_slice].set(jnp.where(ice_mask, blended, target_tile))


def _call_myjsfc(fn: Any, *args: Any, tables: MyjSfcTables | None) -> MyjSfcResult:
    return fn(*args, tables=tables)


def get_local_ice_tsk(
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    itimestep: int,
    sfc_layer_values: bool,
    tice2tsk_if2cold: bool,
    xice: Any,
    xice_threshold: float,
    sst: Any,
    tsk: Any,
) -> tuple[Any, Any, Any]:
    """Return WRF ``get_local_ice_tsk`` updated ``SST``, ``TSK_SEA``, and ``TSK_ICE``."""

    del ime, jme
    xice_arr = jnp.asarray(xice)
    sst_out = jnp.asarray(sst).copy()
    tsk_arr = jnp.asarray(tsk)
    tsk_sea = jnp.zeros_like(tsk_arr)
    tsk_ice = jnp.zeros_like(tsk_arr)

    i0 = int(its) - int(ims)
    i1 = int(ite) - int(ims) + 1
    j0 = int(jts) - int(jms)
    j1 = int(jte) - int(jms) + 1

    xice_sub = xice_arr[i0:i1, j0:j1]
    sst_sub = sst_out[i0:i1, j0:j1]
    tsk_sub = tsk_arr[i0:i1, j0:j1]
    ice_mask = (xice_sub >= float(xice_threshold)) & (xice_sub <= 1.0)

    ice_sst = jnp.where(sst_sub < 271.4, 271.4, sst_sub)
    if bool(sfc_layer_values):
        warm = (ice_sst > 273.0) & (jnp.asarray(itimestep) <= 3)
        ice_sst = jnp.where(warm & (xice_sub >= 0.6), 271.4, ice_sst)
        ice_sst = jnp.where(warm & (xice_sub < 0.6) & (xice_sub >= 0.4), 273.0, ice_sst)
        ice_sst = jnp.where(
            warm & (xice_sub < 0.4) & (xice_sub >= 0.2) & (ice_sst > 275.0),
            275.0,
            ice_sst,
        )
        ice_sst = jnp.where(
            warm & (xice_sub < 0.2) & (ice_sst > 278.0),
            278.0,
            ice_sst,
        )

    safe_xice = jnp.where(ice_mask, xice_sub, 1.0)
    if bool(tice2tsk_if2cold):
        ice_tsk = jnp.minimum(tsk_sub, 273.15)
    else:
        ice_tsk = (tsk_sub - (1.0 - safe_xice) * ice_sst) / safe_xice
        ice_tsk = jnp.where(ice_tsk < 221.4, 221.4, ice_tsk)
    ice_tsk = jnp.where((xice_sub < 0.2) & (tsk_sub < 253.15), 253.15, ice_tsk)
    ice_tsk = jnp.where((xice_sub < 0.1) & (tsk_sub < 263.15), 263.15, ice_tsk)

    sst_region = jnp.where(ice_mask, ice_sst, sst_sub)
    sea_region = jnp.where(ice_mask, ice_sst, tsk_sub)
    ice_region = jnp.where(ice_mask, ice_tsk, tsk_sub)

    sst_out = sst_out.at[i0:i1, j0:j1].set(sst_region)
    tsk_sea = tsk_sea.at[i0:i1, j0:j1].set(sea_region)
    tsk_ice = tsk_ice.at[i0:i1, j0:j1].set(ice_region)
    return sst_out, tsk_sea, tsk_ice


def myjsfc_seaice_wrapper(
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
    tice2tsk_if2cold: bool,
    xice_threshold: float,
    xice: Any,
    sst: Any,
    chs_sea: Any,
    chs2_sea: Any,
    cqs2_sea: Any,
    cpm_sea: Any,
    flhc_sea: Any,
    flqc_sea: Any,
    qsfc_sea: Any,
    qgh_sea: Any,
    qz0_sea: Any,
    hfx_sea: Any,
    qfx_sea: Any,
    flx_lh_sea: Any,
    tsk_sea: Any,
    ustar: Any,
    znt: Any,
    z0base: Any,
    pblh: Any,
    mavail: Any,
    rmol: Any,
    akhs: Any,
    akms: Any,
    br: Any,
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
    p1000: float,
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
    myjsfc_fn: Any = myjsfc,
) -> MyjSfcSeaIceWrapperResult:
    """Run WRF ``myjsfc_seaice_wrapper`` with explicit returned mutable state."""

    i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
    xice_arr = jnp.asarray(xice)
    xice_tile = xice_arr[i_slice, j_slice]
    ice_mask = (xice_tile >= float(xice_threshold)) & (xice_tile <= 1.0)

    sst_out, tsk_sea_out, tsk_local = get_local_ice_tsk(
        ims,
        ime,
        jms,
        jme,
        its,
        ite,
        jts,
        jte,
        itimestep,
        True,
        tice2tsk_if2cold,
        xice,
        xice_threshold,
        sst,
        tsk,
    )

    tsk_out = jnp.asarray(tsk).copy()
    tsk_out = tsk_out.at[i_slice, j_slice].set(tsk_local[i_slice, j_slice])
    qsfc_out = jnp.asarray(qsfc).copy()
    qsfc_sea_out = jnp.asarray(qsfc_sea).copy()
    hfx_sea_out = jnp.asarray(hfx_sea).copy()
    qfx_sea_out = jnp.asarray(qfx_sea).copy()
    flx_lh_sea_out = jnp.asarray(flx_lh_sea).copy()

    psfc = _tile_psfc(pint, lowlyr, ims, jms, kms, its, ite, jts, jte)
    tsk_tile = tsk_out[i_slice, j_slice]
    qsfc_sea_tile = PQ0SEA / psfc * jnp.exp(A2S * (tsk_tile - A3S) / (tsk_tile - A4S))
    qsfc_sea_out = qsfc_sea_out.at[i_slice, j_slice].set(
        jnp.where(ice_mask, qsfc_sea_tile, qsfc_sea_out[i_slice, j_slice])
    )
    qsfc_ice_tile = (qsfc_out[i_slice, j_slice] - (1.0 - xice_tile) * qsfc_sea_tile) / xice_tile
    qsfc_out = qsfc_out.at[i_slice, j_slice].set(jnp.where(ice_mask, qsfc_ice_tile, qsfc_out[i_slice, j_slice]))
    hfx_sea_out = hfx_sea_out.at[i_slice, j_slice].set(
        jnp.where(ice_mask, jnp.asarray(hfx)[i_slice, j_slice], hfx_sea_out[i_slice, j_slice])
    )
    qfx_sea_out = qfx_sea_out.at[i_slice, j_slice].set(
        jnp.where(ice_mask, jnp.asarray(qfx)[i_slice, j_slice], qfx_sea_out[i_slice, j_slice])
    )
    flx_lh_sea_out = flx_lh_sea_out.at[i_slice, j_slice].set(
        jnp.where(ice_mask, jnp.asarray(flx_lh)[i_slice, j_slice], flx_lh_sea_out[i_slice, j_slice])
    )

    qsfc_hold = qsfc_out
    qz0_hold = jnp.asarray(qz0)
    thz0_hold = jnp.asarray(thz0)
    uz0_hold = jnp.asarray(uz0)
    vz0_hold = jnp.asarray(vz0)
    ustar_hold = jnp.asarray(ustar)
    znt_hold = jnp.asarray(znt)
    pblh_hold = jnp.asarray(pblh)
    rmol_hold = jnp.asarray(rmol)
    akhs_hold = jnp.asarray(akhs)
    akms_hold = jnp.asarray(akms)

    local = _call_myjsfc(
        myjsfc_fn,
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
        tsk_out,
        qsfc_out,
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
        br,
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
        p1000,
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

    xland_sea = jnp.asarray(xland).copy()
    mavail_sea = jnp.asarray(mavail).copy()
    znt_sea = znt_hold.copy()
    z0base_sea = jnp.asarray(z0base).copy()
    tsk_sea_call = tsk_sea_out.copy()
    qsfc_sea_call = qsfc_sea_out.copy()

    sst_clamped = jnp.where(sst_out[i_slice, j_slice] < 271.4, 271.4, sst_out[i_slice, j_slice])
    sst_out = sst_out.at[i_slice, j_slice].set(jnp.where(ice_mask, sst_clamped, sst_out[i_slice, j_slice]))
    sea_qsfc = PQ0SEA / psfc * jnp.exp(A2S * (sst_out[i_slice, j_slice] - A3S) / (sst_out[i_slice, j_slice] - A4S))
    xland_sea = xland_sea.at[i_slice, j_slice].set(jnp.where(ice_mask, 2.0, xland_sea[i_slice, j_slice]))
    mavail_sea = mavail_sea.at[i_slice, j_slice].set(jnp.where(ice_mask, 1.0, mavail_sea[i_slice, j_slice]))
    znt_sea = znt_sea.at[i_slice, j_slice].set(jnp.where(ice_mask, 0.0001, znt_hold[i_slice, j_slice]))
    z0base_sea = z0base_sea.at[i_slice, j_slice].set(jnp.where(ice_mask, 0.0001, z0base_sea[i_slice, j_slice]))
    tsk_sea_call = tsk_sea_call.at[i_slice, j_slice].set(
        jnp.where(ice_mask, sst_out[i_slice, j_slice], tsk_out[i_slice, j_slice])
    )
    qsfc_sea_call = qsfc_sea_call.at[i_slice, j_slice].set(jnp.where(ice_mask, sea_qsfc, qsfc_hold[i_slice, j_slice]))

    sea = _call_myjsfc(
        myjsfc_fn,
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
        tsk_sea_call,
        qsfc_sea_call,
        thz0_hold,
        qz0_hold,
        uz0_hold,
        vz0_hold,
        lowlyr,
        xland_sea,
        ivgtyp,
        isurban,
        iz0tlnd,
        ustar_hold,
        znt_sea,
        z0base_sea,
        pblh_hold,
        mavail_sea,
        rmol_hold,
        akhs_hold,
        akms_hold,
        jnp.asarray(br).copy(),
        chs_sea,
        chs2_sea,
        cqs2_sea,
        hfx_sea_out,
        qfx_sea_out,
        flx_lh_sea_out,
        flhc_sea,
        flqc_sea,
        qgh_sea,
        cpm_sea,
        jnp.zeros_like(jnp.asarray(ct)),
        jnp.zeros_like(jnp.asarray(u10)),
        jnp.zeros_like(jnp.asarray(v10)),
        jnp.zeros_like(jnp.asarray(t02)),
        jnp.zeros_like(jnp.asarray(th02)),
        jnp.zeros_like(jnp.asarray(tshltr)),
        jnp.zeros_like(jnp.asarray(th10)),
        jnp.zeros_like(jnp.asarray(q02)),
        jnp.zeros_like(jnp.asarray(qshltr)),
        jnp.zeros_like(jnp.asarray(q10)),
        jnp.zeros_like(jnp.asarray(pshltr)),
        p1000,
        jnp.zeros_like(jnp.asarray(u10e)),
        jnp.zeros_like(jnp.asarray(v10e)),
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

    ct_out = _blend_tile(local.ct, sea.ct, xice, ice_mask, ims, jms, its, ite, jts, jte)
    pshltr_out = _blend_tile(local.pshltr, sea.pshltr, xice, ice_mask, ims, jms, its, ite, jts, jte)
    qshltr_out = _blend_tile(local.qshltr, sea.qshltr, xice, ice_mask, ims, jms, its, ite, jts, jte)
    q02_out = _blend_tile(local.q02, sea.q02, xice, ice_mask, ims, jms, its, ite, jts, jte)
    q10_out = _blend_tile(local.q10, sea.q10, xice, ice_mask, ims, jms, its, ite, jts, jte)
    th02_out = _blend_tile(local.th02, sea.th02, xice, ice_mask, ims, jms, its, ite, jts, jte)
    th10_out = _blend_tile(local.th10, sea.th10, xice, ice_mask, ims, jms, its, ite, jts, jte)
    tshltr_out = _blend_tile(local.tshltr, sea.tshltr, xice, ice_mask, ims, jms, its, ite, jts, jte)
    t02_out = _blend_tile(local.t02, sea.t02, xice, ice_mask, ims, jms, its, ite, jts, jte)
    u10_out = _blend_tile(local.u10, sea.u10, xice, ice_mask, ims, jms, its, ite, jts, jte)
    v10_out = _blend_tile(local.v10, sea.v10, xice, ice_mask, ims, jms, its, ite, jts, jte)
    thz0_out = _blend_tile(local.thz0, sea.thz0, xice, ice_mask, ims, jms, its, ite, jts, jte)
    uz0_out = _blend_tile(local.uz0, sea.uz0, xice, ice_mask, ims, jms, its, ite, jts, jte)
    vz0_out = _blend_tile(local.vz0, sea.vz0, xice, ice_mask, ims, jms, its, ite, jts, jte)
    ustar_out = _blend_tile(local.ustar, sea.ustar, xice, ice_mask, ims, jms, its, ite, jts, jte)
    pblh_out = _blend_tile(local.pblh, sea.pblh, xice, ice_mask, ims, jms, its, ite, jts, jte)
    rmol_out = _blend_tile(local.rmol, sea.rmol, xice, ice_mask, ims, jms, its, ite, jts, jte)
    akhs_out = _blend_tile(local.akhs, sea.akhs, xice, ice_mask, ims, jms, its, ite, jts, jte)
    akms_out = _blend_tile(local.akms, sea.akms, xice, ice_mask, ims, jms, its, ite, jts, jte)
    u10e_out = jnp.asarray(local.u10e).at[i_slice, j_slice].set(
        jnp.where(ice_mask, u10_out[i_slice, j_slice], jnp.asarray(local.u10e)[i_slice, j_slice])
    )
    v10e_out = jnp.asarray(local.v10e).at[i_slice, j_slice].set(
        jnp.where(ice_mask, v10_out[i_slice, j_slice], jnp.asarray(local.v10e)[i_slice, j_slice])
    )

    return MyjSfcSeaIceWrapperResult(
        tsk=tsk_out,
        qsfc=local.qsfc,
        thz0=thz0_out,
        qz0=local.qz0,
        uz0=uz0_out,
        vz0=vz0_out,
        sst=sst_out,
        br=local.rib,
        chs_sea=sea.chs,
        chs2_sea=sea.chs2,
        cqs2_sea=sea.cqs2,
        cpm_sea=sea.cpm,
        flhc_sea=sea.flhc,
        flqc_sea=sea.flqc,
        qsfc_sea=sea.qsfc,
        qgh_sea=sea.qgh,
        qz0_sea=sea.qz0,
        hfx_sea=sea.hfx,
        qfx_sea=sea.qfx,
        flx_lh_sea=sea.flx_lh,
        tsk_sea=tsk_sea_call,
        ustar=ustar_out,
        znt=local.znt,
        pblh=pblh_out,
        rmol=rmol_out,
        akhs=akhs_out,
        akms=akms_out,
        chs=local.chs,
        chs2=local.chs2,
        cqs2=local.cqs2,
        hfx=local.hfx,
        qfx=local.qfx,
        flx_lh=local.flx_lh,
        flhc=local.flhc,
        flqc=local.flqc,
        qgh=local.qgh,
        cpm=local.cpm,
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


__all__ = [
    "LSMSCHEME",
    "MYJSFCSCHEME",
    "SF_SFCLAY_NONE",
    "MyjSfcSeaIceWrapperResult",
    "SurfaceDriverResult",
    "get_local_ice_tsk",
    "myjsfc_seaice_wrapper",
    "surface_driver",
]
