"""Explicit forcing containers for WRF boundary data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoundaryForcing:
    """Time-dependent lateral boundary arrays loaded from ``wrfbdy``."""

    times: tuple[str, ...]
    names: tuple[str, ...]
    arrays: tuple[Any, ...]

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], tuple[str, ...]]]:
        return self.arrays, (self.times, self.names)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[tuple[str, ...], tuple[str, ...]], children: tuple[Any, ...]) -> "BoundaryForcing":
        times, names = aux_data
        return cls(times=times, names=names, arrays=children)

    def as_dict(self) -> dict[str, Any]:
        return dict(zip(self.names, self.arrays, strict=True))


__all__ = ["BoundaryForcing"]
