"""Compatibility helper for WRF ``module_domain::get_ijk_from_grid1``."""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Any

from .get_ijk_from_grid2 import MODEL_DATA_ORDER, _ORDER_DIMS, _pair, get_ijk_from_grid2


@dataclass(frozen=True)
class IJKGrid1Bounds:
    values: tuple[int, ...]

    def as_tuple(self) -> tuple[int, ...]:
        return self.values


def _suffix_bounds(grid: Any, dim_order: tuple[str, str, str], suffix: str) -> tuple[int, ...]:
    i_dim, j_dim, k_dim = dim_order
    im, jm, km = (
        _pair(grid, f"sm", f"em", f"{i_dim}{suffix}"),
        _pair(grid, f"sm", f"em", f"{j_dim}{suffix}"),
        _pair(grid, f"sm", f"em", f"{k_dim}{suffix}"),
    )
    ip, jp, kp = (
        _pair(grid, f"sp", f"ep", f"{i_dim}{suffix}"),
        _pair(grid, f"sp", f"ep", f"{j_dim}{suffix}"),
        _pair(grid, f"sp", f"ep", f"{k_dim}{suffix}"),
    )
    return (*im, *jm, *km, *ip, *jp, *kp)


def get_ijk_from_grid1(
    grid: Any,
    *,
    model_data_order: int = MODEL_DATA_ORDER,
    output: MutableSequence[int] | None = None,
) -> IJKGrid1Bounds:
    """Return WRF grid bounds including staggered x/y memory and patch bounds."""

    try:
        dim_order = _ORDER_DIMS[int(model_data_order)]
    except KeyError as exc:
        raise ValueError(f"unsupported model_data_order {model_data_order!r}") from exc

    base = get_ijk_from_grid2(grid, model_data_order=model_data_order).as_tuple()
    result = IJKGrid1Bounds(
        (*base, *_suffix_bounds(grid, dim_order, "x"), *_suffix_bounds(grid, dim_order, "y"))
    )
    if output is not None:
        if len(output) < 42:
            raise ValueError("output holder must contain at least 42 elements")
        output[:42] = result.as_tuple()
    return result


__all__ = ["IJKGrid1Bounds", "get_ijk_from_grid1"]
