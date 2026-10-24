"""Translation support for ``module_physics_init::mp_init``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


MP_SCHEME_CALLS = {
    1: "kessler_warm_rain",
    3: "wsm3init",
    4: "wsm5init",
    6: "wsm6init",
    11: "cammgmp_init",
    14: "wdm5init",
    16: "wdm6init",
    24: "wsm7init",
    26: "wdm7init",
    28: "thompson_init",
    50: "p3_init_1category",
    51: "p3_init_1category_nc",
    52: "p3_init_2category",
    53: "p3_init_1cat_3mom",
    55: "jensen_ishmael_init",
    56: "ntu_init",
    95: "etanewinit",
    96: "morr_two_moment_init",
    97: "nssl_2mom_init",
    98: "nssl_2mom_init",
    99: "nssl_2mom_init",
    100: "nssl_2mom_init",
    108: "thompson_init",
}


@dataclass(frozen=True)
class MpInitResult:
    rainnc: Any
    snownc: Any
    graupelnc: Any
    warm_rain: bool
    adv_moist_cond: bool
    ccn_conc: Any | None = None
    carried_arrays: Mapping[str, Any] | None = None
    external_calls: tuple[str, ...] = ()


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def mp_init(
    rainnc: Any,
    snownc: Any,
    graupelnc: Any,
    config_flags: Any,
    restart: bool,
    mpdt: float,
    dt: float,
    dx: float,
    dy: float,
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
    ccn_conc: Any | None = None,
    **arrays: Any,
) -> MpInitResult:
    """Initialize microphysics-driver state for the active WRF tile."""

    del restart, mpdt, dt, dx, dy, allowed_to_read, ids, jds, kds, kde, kms, kme, kts, kte, ime, jme
    warm_rain = False
    adv_moist_cond = True
    rainnc_out = jnp.asarray(rainnc)
    snownc_out = jnp.asarray(snownc)
    graupelnc_out = jnp.asarray(graupelnc)

    if start_of_simulation:
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        i_slice = slice(int(its) - int(ims), itf - int(ims) + 1)
        j_slice = slice(int(jts) - int(jms), jtf - int(jms) + 1)
        rainnc_out = rainnc_out.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=rainnc_out.dtype))
        snownc_out = snownc_out.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=snownc_out.dtype))
        graupelnc_out = graupelnc_out.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=graupelnc_out.dtype))

    mp_scheme = int(_read(config_flags, "mp_physics", 0))
    external_calls: tuple[str, ...] = ()
    if mp_scheme == 1:
        warm_rain = True
    elif mp_scheme == 95:
        adv_moist_cond = False

    if mp_scheme in MP_SCHEME_CALLS and mp_scheme != 1:
        external_calls = (MP_SCHEME_CALLS[mp_scheme],)

    if int(_read(config_flags, "madwrf_opt", 0)) == 2:
        has_required = bool(_read(config_flags, "f_qc", False)) and bool(_read(config_flags, "f_qi", False))
        has_required = has_required and bool(_read(config_flags, "f_qs", False))
        if not has_required:
            raise RuntimeError("madwrf_opt = 2 requires qc, qi, and qs")

    return MpInitResult(
        rainnc=rainnc_out,
        snownc=snownc_out,
        graupelnc=graupelnc_out,
        warm_rain=warm_rain,
        adv_moist_cond=adv_moist_cond,
        ccn_conc=ccn_conc,
        carried_arrays=dict(arrays),
        external_calls=external_calls,
    )


__all__ = ["MP_SCHEME_CALLS", "MpInitResult", "mp_init"]
