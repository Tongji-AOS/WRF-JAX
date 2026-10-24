"""Translation support for ``module_physics_init::ra_init``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


RRTMSCHEME = 1
CAMLWSCHEME = 3
RRTMG_LWSCHEME = 4
GODDARDLWSCHEME = 5
FLGLWSCHEME = 7
HELDSUAREZ = 31
GFDLLWSCHEME = 99

SWRADSCHEME = 1
GSFCSWSCHEME = 2
CAMSWSCHEME = 3
RRTMG_SWSCHEME = 4
GODDARDSWSCHEME = 5
FLGSWSCHEME = 7
GFDLSWSCHEME = 99


@dataclass(frozen=True)
class RaInitResult:
    stepra: int
    rthraten: Any
    rthratenlw: Any
    rthratensw: Any
    cldfra: Any
    emiss: Any
    cldfra_dp: Any | None = None
    cldfra_sh: Any | None = None
    cldfra_old: Any | None = None
    ghi_accum: Any | None = None
    external_calls: tuple[str, ...] = ()


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _tile_slices(
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
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
    del ids, jds, kds
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    return (
        slice(int(its) - int(ims), itf - int(ims) + 1),
        slice(int(kts) - int(kms), ktf - int(kms) + 1),
        slice(int(jts) - int(jms), jtf - int(jms) + 1),
    )


def _zero_3d_tile(array: Any, slices: tuple[slice, slice, slice]) -> Any:
    arr = jnp.asarray(array)
    return arr.at[slices].set(jnp.asarray(0.0, dtype=arr.dtype))


def _zero_2d_tile(
    array: Any,
    i_slice: slice,
    j_slice: slice,
) -> Any:
    arr = jnp.asarray(array)
    return arr.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=arr.dtype))


def _nint_positive(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def ra_init(
    id: int,
    stepra: int,
    radt: float,
    dt: float,
    rthraten: Any,
    rthratenlw: Any,
    rthratensw: Any,
    cldfra: Any,
    emiss: Any,
    cen_lat: float,
    julyr: int,
    julday: int,
    gmt: float,
    levsiz: int,
    xlat: Any,
    xlong: Any,
    n_ozmixm: int,
    alevsiz: int,
    no_src_types: int,
    sfull: Any,
    shalf: Any,
    pptop: float,
    swrad_scat: float,
    p_top: float,
    config_flags: Any,
    restart: bool,
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
    cldfra_dp: Any | None = None,
    cldfra_sh: Any | None = None,
    cldfra_old: Any | None = None,
    ghi_accum: Any | None = None,
    ozmixm: Any | None = None,
    pin: Any | None = None,
    aerodm: Any | None = None,
    pina: Any | None = None,
    m_ps_1: Any | None = None,
    m_ps_2: Any | None = None,
    m_hybi: Any | None = None,
    aerosolc_1: Any | None = None,
    aerosolc_2: Any | None = None,
    paerlev: int | None = None,
    n_aerosolc: int | None = None,
) -> RaInitResult:
    """Initialize radiation driver state for the active WRF tile."""

    del (
        stepra,
        cen_lat,
        julyr,
        julday,
        gmt,
        levsiz,
        xlat,
        xlong,
        n_ozmixm,
        alevsiz,
        no_src_types,
        sfull,
        shalf,
        pptop,
        swrad_scat,
        p_top,
        restart,
        allowed_to_read,
        ime,
        jme,
        kme,
        ozmixm,
        pin,
        aerodm,
        pina,
        m_ps_1,
        m_ps_2,
        m_hybi,
        aerosolc_1,
        aerosolc_2,
        paerlev,
        n_aerosolc,
    )

    result_stepra = max(_nint_positive(float(radt) * 60.0 / float(dt)), 1)
    rthraten_out = jnp.asarray(rthraten)
    rthratenlw_out = jnp.asarray(rthratenlw)
    rthratensw_out = jnp.asarray(rthratensw)
    cldfra_out = jnp.asarray(cldfra)
    emiss_out = jnp.asarray(emiss)
    cldfra_dp_out = None if cldfra_dp is None else jnp.asarray(cldfra_dp)
    cldfra_sh_out = None if cldfra_sh is None else jnp.asarray(cldfra_sh)
    cldfra_old_out = None if cldfra_old is None else jnp.asarray(cldfra_old)
    ghi_accum_out = None if ghi_accum is None else jnp.asarray(ghi_accum)

    slices = _tile_slices(ids, ide, jds, jde, kds, kde, ims, jms, kms, its, ite, jts, jte, kts, kte)
    if start_of_simulation:
        rthraten_out = _zero_3d_tile(rthraten_out, slices)
        rthratenlw_out = _zero_3d_tile(rthratenlw_out, slices)
        rthratensw_out = _zero_3d_tile(rthratensw_out, slices)
        cldfra_out = _zero_3d_tile(cldfra_out, slices)

        if cldfra_dp is not None:
            if cldfra_sh is None:
                raise ValueError("cldfra_sh is required when cldfra_dp is present")
            cldfra_dp_out = _zero_3d_tile(cldfra_dp_out, slices)
            cldfra_sh_out = _zero_3d_tile(cldfra_sh_out, slices)

        if cldfra_old is not None:
            cldfra_old_out = _zero_3d_tile(cldfra_old_out, slices)

        if ghi_accum is not None:
            i_slice, _, j_slice = slices
            ghi_accum_out = _zero_2d_tile(ghi_accum_out, i_slice, j_slice)

    external_calls: list[str] = []
    if int(_read(config_flags, "o3input", 0)) == 2 and int(id) == 1:
        external_calls.append("oznini")
    if int(_read(config_flags, "aer_opt", 0)) == 1 and int(id) == 1:
        external_calls.append("aerosol_in")

    acswalloc = False
    aclwalloc = False
    camlw = False
    etalw = False

    lw_scheme = int(_read(config_flags, "ra_lw_physics", 0))
    if lw_scheme == GODDARDLWSCHEME:
        external_calls.append("init_goddardrad_lw")
    elif lw_scheme == RRTMSCHEME:
        external_calls.append("rrtminit")
    elif lw_scheme == CAMLWSCHEME:
        external_calls.append("camradinit_lw")
        camlw = True
        aclwalloc = True
    elif lw_scheme == RRTMG_LWSCHEME:
        external_calls.append("rrtmg_lwinit")
        aclwalloc = True
    elif lw_scheme == GFDLLWSCHEME:
        external_calls.append("gfdletainit_lw")
        etalw = True
    elif lw_scheme == HELDSUAREZ:
        external_calls.append("hsinit")
    elif lw_scheme == FLGLWSCHEME:
        pass

    sw_scheme = int(_read(config_flags, "ra_sw_physics", 0))
    if sw_scheme == GODDARDSWSCHEME:
        external_calls.append("init_goddardrad_sw")
    elif sw_scheme == SWRADSCHEME:
        external_calls.append("swinit")
    elif sw_scheme == CAMSWSCHEME:
        if not camlw:
            external_calls.append("camradinit_sw")
        acswalloc = True
    elif sw_scheme == GSFCSWSCHEME:
        external_calls.append("gsfc_swinit")
    elif sw_scheme == RRTMG_SWSCHEME:
        external_calls.append("rrtmg_swinit")
        acswalloc = True
    elif sw_scheme == GFDLSWSCHEME:
        if not etalw:
            external_calls.append("gfdletainit_sw")
    elif sw_scheme == FLGSWSCHEME:
        pass

    if float(_read(config_flags, "bucket_J", 0.0)) > 0.0 and not (acswalloc and aclwalloc):
        raise RuntimeError("Need CAM or RRTMG radiation for bucket_J option")

    return RaInitResult(
        stepra=result_stepra,
        rthraten=rthraten_out,
        rthratenlw=rthratenlw_out,
        rthratensw=rthratensw_out,
        cldfra=cldfra_out,
        emiss=emiss_out,
        cldfra_dp=cldfra_dp_out,
        cldfra_sh=cldfra_sh_out,
        cldfra_old=cldfra_old_out,
        ghi_accum=ghi_accum_out,
        external_calls=tuple(external_calls),
    )


__all__ = ["RaInitResult", "ra_init"]
