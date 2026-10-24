"""Outer-shell translation of WRF ``phys/module_fddagd_driver``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _read(config: Any, name: str, default: Any = 0) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def fddagd_driver(config_flags: Any, **fields: Any) -> dict[str, Any]:
    """Return FDDA driver fields unchanged for the disabled WRF FDDA path."""

    if int(_read(config_flags, "grid_fdda", 0)) == 0 and int(_read(config_flags, "grid_sfdda", 0)) == 0:
        return dict(fields)
    raise NotImplementedError(
        "fddagd_driver currently implements the configured disabled FDDA path only"
    )


__all__ = ["fddagd_driver"]
