"""Translation support for ``module_physics_init::fg_init``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


FG_SCHEME_CALLS = {
    1: "fddagdinit",
    2: "fddaspnudginginit",
}


@dataclass(frozen=True)
class FgInitResult:
    stepfg: int
    carried_arrays: Mapping[str, Any]
    external_calls: tuple[str, ...] = ()


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _nint_positive(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def fg_init(
    stepfg: int,
    fgdt: float,
    dt: float,
    id: int,
    config_flags: Any,
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
    **arrays: Any,
) -> FgInitResult:
    """Initialize analysis nudging driver state."""

    del stepfg, id, restart, allowed_to_read, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
    del its, ite, jts, jte, kts, kte
    result_stepfg = max(_nint_positive(float(fgdt) * 60.0 / float(dt)), 1)
    scheme = int(_read(config_flags, "grid_fdda", 0))
    external_calls = ()
    if scheme in FG_SCHEME_CALLS:
        external_calls = (FG_SCHEME_CALLS[scheme],)
    return FgInitResult(stepfg=result_stepfg, carried_arrays=dict(arrays), external_calls=external_calls)


__all__ = ["FG_SCHEME_CALLS", "FgInitResult", "fg_init"]
