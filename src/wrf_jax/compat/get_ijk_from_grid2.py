"""Compatibility helper for WRF ``module_domain::get_ijk_from_grid2``."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass
from typing import Any

DATA_ORDER_XYZ = 1
DATA_ORDER_YXZ = 2
DATA_ORDER_ZXY = 3
DATA_ORDER_ZYX = 4
DATA_ORDER_XZY = 5
DATA_ORDER_YZX = 6
MODEL_DATA_ORDER = DATA_ORDER_XZY

_ORDER_DIMS = {
    DATA_ORDER_XYZ: ("31", "32", "33"),
    DATA_ORDER_YXZ: ("32", "31", "33"),
    DATA_ORDER_ZXY: ("32", "33", "31"),
    DATA_ORDER_ZYX: ("33", "32", "31"),
    DATA_ORDER_XZY: ("31", "33", "32"),
    DATA_ORDER_YZX: ("33", "31", "32"),
}


@dataclass(frozen=True)
class IJKBounds:
    ids: int
    ide: int
    jds: int
    jde: int
    kds: int
    kde: int
    ims: int
    ime: int
    jms: int
    jme: int
    kms: int
    kme: int
    ips: int
    ipe: int
    jps: int
    jpe: int
    kps: int
    kpe: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.ids,
            self.ide,
            self.jds,
            self.jde,
            self.kds,
            self.kde,
            self.ims,
            self.ime,
            self.jms,
            self.jme,
            self.kms,
            self.kme,
            self.ips,
            self.ipe,
            self.jps,
            self.jpe,
            self.kps,
            self.kpe,
        )


def _read(grid: Any, name: str) -> int:
    if isinstance(grid, Mapping):
        try:
            return int(grid[name])
        except KeyError as exc:
            raise ValueError(f"grid is missing required field '{name}'") from exc
    if not hasattr(grid, name):
        raise ValueError(f"grid is missing required field '{name}'")
    return int(getattr(grid, name))


def _pair(grid: Any, prefix_start: str, prefix_end: str, dim: str) -> tuple[int, int]:
    return _read(grid, f"{prefix_start}{dim}"), _read(grid, f"{prefix_end}{dim}")


def get_ijk_from_grid2(
    grid: Any,
    *,
    model_data_order: int = MODEL_DATA_ORDER,
    output: MutableSequence[int] | None = None,
) -> IJKBounds:
    """Return WRF's domain, memory, and patch bounds for the configured order."""

    try:
        i_dim, j_dim, k_dim = _ORDER_DIMS[int(model_data_order)]
    except KeyError as exc:
        raise ValueError(f"unsupported model_data_order {model_data_order!r}") from exc

    ids, ide = _pair(grid, "sd", "ed", i_dim)
    jds, jde = _pair(grid, "sd", "ed", j_dim)
    kds, kde = _pair(grid, "sd", "ed", k_dim)
    ims, ime = _pair(grid, "sm", "em", i_dim)
    jms, jme = _pair(grid, "sm", "em", j_dim)
    kms, kme = _pair(grid, "sm", "em", k_dim)
    ips, ipe = _pair(grid, "sp", "ep", i_dim)
    jps, jpe = _pair(grid, "sp", "ep", j_dim)
    kps, kpe = _pair(grid, "sp", "ep", k_dim)

    result = IJKBounds(
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
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
    )
    if output is not None:
        if len(output) < 18:
            raise ValueError("output holder must contain at least 18 elements")
        output[:18] = result.as_tuple()
    return result


__all__ = [
    "DATA_ORDER_XYZ",
    "DATA_ORDER_XZY",
    "DATA_ORDER_YXZ",
    "DATA_ORDER_YZX",
    "DATA_ORDER_ZXY",
    "DATA_ORDER_ZYX",
    "IJKBounds",
    "MODEL_DATA_ORDER",
    "get_ijk_from_grid2",
]
