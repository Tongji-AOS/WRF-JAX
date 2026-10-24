"""Core model state containers for the WRF-JAX runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WRFState:
    """Differentiable prognostic subset required by the configured case."""

    u: Any
    v: Any
    t: Any
    qvapor: Any
    time_index: int = 0
    time_seconds: float = 0.0

    def tree_flatten(self) -> tuple[tuple[Any, Any, Any, Any], tuple[int, float]]:
        return (self.u, self.v, self.t, self.qvapor), (int(self.time_index), float(self.time_seconds))

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[int, float], children: tuple[Any, Any, Any, Any]) -> "WRFState":
        time_index, time_seconds = aux_data
        u, v, t, qvapor = children
        return cls(u=u, v=v, t=t, qvapor=qvapor, time_index=time_index, time_seconds=time_seconds)

    def advance_metadata(self, *, time_index: int, time_seconds: float) -> "WRFState":
        return replace(self, time_index=int(time_index), time_seconds=float(time_seconds))


__all__ = ["WRFState"]
