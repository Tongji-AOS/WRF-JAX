"""Selected translations from WRF ``module_radiation_driver``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


RADIATION_PHYSICS_NONE = 0


@dataclass(frozen=True)
class RadiationDriverResult:
    """Explicit outputs carried by WRF ``radiation_driver``."""

    rthraten: Any | None = None
    rthratenlw: Any | None = None
    rthratenlwc: Any | None = None
    rthratensw: Any | None = None
    rthratenswc: Any | None = None
    gsw: Any | None = None
    glw: Any | None = None
    swdown: Any | None = None
    swddir: Any | None = None
    swddni: Any | None = None
    swddif: Any | None = None
    radtacttime: float | None = None
    carried_fields: Mapping[str, Any] = field(default_factory=dict)
    ran_parameterization: bool = False
    external_calls: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreRadiationDriverResult:
    """Explicit outputs mutated by WRF ``pre_radiation_driver``."""

    min_ptchsz: int
    ht_shad: Any
    ht_loc: Any
    shadowmask: Any
    niter: int | None = None


def _copy_array(value: Any) -> np.ndarray:
    return np.array(value, copy=True)


def _idx(index: int, lower: int) -> int:
    return int(index) - int(lower)


def _tile_bounds(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def radiation_driver(
    *,
    lw_physics: int,
    sw_physics: int,
    rthraten: Any | None = None,
    rthratenlw: Any | None = None,
    rthratenlwc: Any | None = None,
    rthratensw: Any | None = None,
    rthratenswc: Any | None = None,
    gsw: Any | None = None,
    glw: Any | None = None,
    swdown: Any | None = None,
    swddir: Any | None = None,
    swddni: Any | None = None,
    swddif: Any | None = None,
    radtacttime: float | None = None,
    **fields: Any,
) -> RadiationDriverResult:
    """Run the translated WRF radiation driver for supported configurations.

    The configured migration case has both longwave and shortwave radiation
    disabled. WRF returns immediately in that branch, before the normal driver
    clears heating rates or surface radiation fluxes. Active radiation schemes
    require their scheme-specific column kernels and are rejected here until
    those functions are translated.
    """

    if int(lw_physics) == RADIATION_PHYSICS_NONE and int(sw_physics) == RADIATION_PHYSICS_NONE:
        return RadiationDriverResult(
            rthraten=rthraten,
            rthratenlw=rthratenlw,
            rthratenlwc=rthratenlwc,
            rthratensw=rthratensw,
            rthratenswc=rthratenswc,
            gsw=gsw,
            glw=glw,
            swdown=swdown,
            swddir=swddir,
            swddni=swddni,
            swddif=swddif,
            radtacttime=radtacttime,
            carried_fields=dict(fields),
            ran_parameterization=False,
        )

    raise NotImplementedError(
        "radiation_driver currently implements only the WRF "
        "lw_physics=0 and sw_physics=0 branch; "
        f"got lw_physics={int(lw_physics)} sw_physics={int(sw_physics)}"
    )


def pre_radiation_driver(
    *,
    grid: Any = None,
    config_flags: Any = None,
    itimestep: int,
    ra_call_offset: int,
    xlat: Any,
    xlong: Any,
    gmt: float,
    julian: float,
    xtime: float,
    radt: float,
    stepra: int,
    ht: Any,
    dx: float,
    dy: float,
    sina: Any,
    cosa: Any,
    shadowmask: Any,
    slope_rad: int,
    topo_shading: int,
    shadlen: float,
    ht_shad: Any,
    ht_loc: Any,
    ht_shad_bxs: Any | None = None,
    ht_shad_bxe: Any | None = None,
    ht_shad_bys: Any | None = None,
    ht_shad_bye: Any | None = None,
    nested: bool = False,
    min_ptchsz: int = 0,
    spec_bdy_width: int = 0,
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
    ips: int = 1,
    ipe: int = 1,
    jps: int = 1,
    jpe: int = 1,
    kps: int = 1,
    kpe: int = 1,
    i_start: Sequence[int] = (),
    i_end: Sequence[int] = (),
    j_start: Sequence[int] = (),
    j_end: Sequence[int] = (),
    kts: int = 1,
    kte: int = 1,
    num_tiles: int = 1,
    dm_minval_integer: Callable[[int], int] | None = None,
    radconst_fn: Callable[..., tuple[float, float]] | None = None,
    spec_bdyfield_fn: Callable[..., Any] | None = None,
    toposhad_init_fn: Callable[..., tuple[Any, Any, Any]] | None = None,
    toposhad_fn: Callable[..., tuple[Any, Any, Any]] | None = None,
    halo_toposhad_fn: Callable[..., tuple[Any, Any, Any]] | None = None,
) -> PreRadiationDriverResult:
    """Run WRF ``pre_radiation_driver`` control flow.

    The configured target case does not enter the topographic-shading branch.
    When that branch is enabled, the untranslated shading/halo kernels are
    supplied as explicit callbacks rather than hidden global calls.
    """

    del slope_rad, kds, kde, kms, kme

    min_value = int(min_ptchsz)
    if int(itimestep) == 1:
        psx = int(ipe) - int(ips) + 1
        psy = int(jpe) - int(jps) + 1
        if dm_minval_integer is not None:
            psx = int(dm_minval_integer(psx))
            psy = int(dm_minval_integer(psy))
        min_value = min(psx, psy)

    ht_shad_out = _copy_array(ht_shad)
    ht_loc_out = _copy_array(ht_loc)
    shadowmask_out = _copy_array(shadowmask)

    shading_due = int(topo_shading) == 1 and (
        int(itimestep) == 1 or int(itimestep) % int(stepra) == 1 + int(ra_call_offset)
    )
    if not shading_due:
        return PreRadiationDriverResult(
            min_ptchsz=min_value,
            ht_shad=ht_shad_out,
            ht_loc=ht_loc_out,
            shadowmask=shadowmask_out,
            niter=None,
        )

    if radconst_fn is None or toposhad_init_fn is None or toposhad_fn is None:
        raise NotImplementedError(
            "topographic shading requires radconst, toposhad_init, and toposhad callbacks"
        )

    declin, _solcon = radconst_fn(xtime=xtime, julian=julian)

    ht_arr = np.asarray(ht)
    for j in range(int(jms), int(jme) + 1):
        for i in range(int(ims), int(ime) + 1):
            ht_loc_out[_idx(i, ims), _idx(j, jms)] = ht_arr[_idx(i, ims), _idx(j, jms)]

    if int(ids) == int(ips) and int(ide) == int(ipe) and int(jds) == int(jps) and int(jde) == int(jpe):
        niter = 1
    else:
        niter = int(float(shadlen) / (float(dx) * float(min_value)) + 3.0)

    starts_i = _tile_bounds(i_start) or (int(ips),) * int(num_tiles)
    ends_i = _tile_bounds(i_end) or (int(ipe),) * int(num_tiles)
    starts_j = _tile_bounds(j_start) or (int(jps),) * int(num_tiles)
    ends_j = _tile_bounds(j_end) or (int(jpe),) * int(num_tiles)

    if nested:
        if spec_bdyfield_fn is None:
            raise NotImplementedError("nested topographic shading requires spec_bdyfield callback")
        for tile in range(int(num_tiles)):
            ht_shad_out = spec_bdyfield_fn(
                ht_shad_out,
                ht_shad_bxs,
                ht_shad_bxe,
                ht_shad_bys,
                ht_shad_bye,
                "m",
                config_flags,
                spec_bdy_width,
                2,
                ids,
                ide,
                jds,
                jde,
                ims,
                ime,
                jms,
                jme,
                ips,
                ipe,
                jps,
                jpe,
                starts_i[tile],
                ends_i[tile],
                starts_j[tile],
                ends_j[tile],
            )

    for ni in range(1, niter + 1):
        for tile in range(int(num_tiles)):
            ht_shad_out, ht_loc_out, shadowmask_out = toposhad_init_fn(
                ht_shad_out,
                ht_loc_out,
                shadowmask_out,
                nested,
                ni,
                ids,
                ide,
                jds,
                jde,
                ims,
                ime,
                jms,
                jme,
                ips,
                min(ipe, ide - 1),
                jps,
                min(jpe, jde - 1),
                starts_i[tile],
                min(ends_i[tile], ide - 1),
                starts_j[tile],
                min(ends_j[tile], jde - 1),
                kts,
                kte,
            )
        for tile in range(int(num_tiles)):
            ht_shad_out, ht_loc_out, shadowmask_out = toposhad_fn(
                xlat,
                xlong,
                sina,
                cosa,
                xtime,
                gmt,
                radt,
                declin,
                dx,
                dy,
                ht_shad_out,
                ht_loc_out,
                ni,
                shadowmask_out,
                shadlen,
                ids,
                ide,
                jds,
                jde,
                ims,
                ime,
                jms,
                jme,
                ips,
                min(ipe, ide - 1),
                jps,
                min(jpe, jde - 1),
                starts_i[tile],
                min(ends_i[tile], ide - 1),
                starts_j[tile],
                min(ends_j[tile], jde - 1),
                kts,
                kte,
            )
        if halo_toposhad_fn is not None:
            ht_shad_out, ht_loc_out, shadowmask_out = halo_toposhad_fn(
                grid,
                ht_shad_out,
                ht_loc_out,
                shadowmask_out,
            )

    return PreRadiationDriverResult(
        min_ptchsz=min_value,
        ht_shad=ht_shad_out,
        ht_loc=ht_loc_out,
        shadowmask=shadowmask_out,
        niter=niter,
    )


__all__ = [
    "PreRadiationDriverResult",
    "RADIATION_PHYSICS_NONE",
    "RadiationDriverResult",
    "pre_radiation_driver",
    "radiation_driver",
]
