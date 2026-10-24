"""Translation support for ``module_physics_init::shcu_init``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


SHCU_SCHEME_CALLS = {
    2: "init_uwshcu",
    3: "grimsinit",
    4: "nscvinit",
    5: "deng_shcu_init",
}


@dataclass(frozen=True)
class ShcuInitResult:
    stepcu: int
    rainc: Any
    raincv: Any
    carried_arrays: Mapping[str, Any]
    external_calls: tuple[str, ...] = ()


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _nint_positive(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def shcu_init(
    stepcu: int,
    cudt: float,
    dt: float,
    rainc: Any,
    raincv: Any,
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
    **arrays: Any,
) -> ShcuInitResult:
    """Initialize shallow-cumulus driver state for the active WRF tile."""

    del stepcu, restart, allowed_to_read, ids, jds, kds, kde, kms, kme, kts, kte, ime, jme
    result_stepcu = max(_nint_positive(float(cudt) * 60.0 / float(dt)), 1)
    rainc_out = jnp.asarray(rainc)
    raincv_out = jnp.asarray(raincv)

    if start_of_simulation:
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        i_slice = slice(int(its) - int(ims), itf - int(ims) + 1)
        j_slice = slice(int(jts) - int(jms), jtf - int(jms) + 1)
        rainc_out = rainc_out.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=rainc_out.dtype))
        raincv_out = raincv_out.at[i_slice, j_slice].set(jnp.asarray(0.0, dtype=raincv_out.dtype))

    shcu_scheme = int(_read(config_flags, "shcu_physics", 0))
    external_calls = ()
    if shcu_scheme in SHCU_SCHEME_CALLS:
        external_calls = (SHCU_SCHEME_CALLS[shcu_scheme],)

    return ShcuInitResult(
        stepcu=result_stepcu,
        rainc=rainc_out,
        raincv=raincv_out,
        carried_arrays=dict(arrays),
        external_calls=external_calls,
    )


__all__ = ["SHCU_SCHEME_CALLS", "ShcuInitResult", "shcu_init"]
