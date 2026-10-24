"""Compatibility helper for WRF ``module_domain::get_ijk_from_subgrid``."""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Any

from .get_ijk_from_grid2 import MODEL_DATA_ORDER, _read, get_ijk_from_grid2


@dataclass(frozen=True)
class IJKSubgridBounds:
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


def _scale_lower(bound: int, ratio: int) -> int:
    return (bound - 1) * ratio + 1


def get_ijk_from_subgrid(
    grid: Any,
    *,
    model_data_order: int = MODEL_DATA_ORDER,
    output: MutableSequence[int] | None = None,
) -> IJKSubgridBounds:
    """Return WRF subgrid bounds after applying ``sr_x`` and ``sr_y``."""

    bounds = get_ijk_from_grid2(grid, model_data_order=model_data_order)
    sr_x = _read(grid, "sr_x")
    sr_y = _read(grid, "sr_y")

    result = IJKSubgridBounds(
        bounds.ids,
        bounds.ide * sr_x,
        bounds.jds,
        bounds.jde * sr_y,
        bounds.kds,
        bounds.kde,
        _scale_lower(bounds.ims, sr_x),
        bounds.ime * sr_x,
        _scale_lower(bounds.jms, sr_y),
        bounds.jme * sr_y,
        bounds.kms,
        bounds.kme,
        _scale_lower(bounds.ips, sr_x),
        bounds.ipe * sr_x,
        _scale_lower(bounds.jps, sr_y),
        bounds.jpe * sr_y,
        bounds.kps,
        bounds.kpe,
    )
    if output is not None:
        if len(output) < 18:
            raise ValueError("output holder must contain at least 18 elements")
        output[:18] = result.as_tuple()
    return result


__all__ = ["IJKSubgridBounds", "get_ijk_from_subgrid"]
