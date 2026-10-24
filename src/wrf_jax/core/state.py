"""PyTree state containers for the production WRF-JAX core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TimeState:
    """Dynamic model-clock values that must not hide on Python grid objects."""

    itimestep: Any
    elapsed_seconds: Any
    dtbc: Any

    def tree_flatten(self) -> tuple[tuple[Any, Any, Any], tuple[()]]:
        return (self.itimestep, self.elapsed_seconds, self.dtbc), ()

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[()], children: tuple[Any, Any, Any]) -> "TimeState":
        del aux_data
        itimestep, elapsed_seconds, dtbc = children
        return cls(itimestep=itimestep, elapsed_seconds=elapsed_seconds, dtbc=dtbc)

    @classmethod
    def from_grid(cls, grid: Any) -> "TimeState":
        return cls(
            itimestep=jnp.asarray(int(getattr(grid, "itimestep", 0)), dtype=jnp.int32),
            elapsed_seconds=jnp.asarray(float(getattr(grid, "current_time", 0.0)), dtype=jnp.float32),
            dtbc=jnp.asarray(float(getattr(grid, "dtbc", 0.0)), dtype=jnp.float32),
        )

    def advance(self, dt_seconds: float, boundary_interval_seconds: int) -> "TimeState":
        elapsed = self.elapsed_seconds + jnp.asarray(float(dt_seconds), dtype=jnp.asarray(self.elapsed_seconds).dtype)
        dtbc = self.dtbc + jnp.asarray(float(dt_seconds), dtype=jnp.asarray(self.dtbc).dtype)
        interval = int(boundary_interval_seconds)
        if interval > 0:
            at_boundary_time = (elapsed > 0.0) & (jnp.mod(elapsed, float(interval)) == 0.0)
            dtbc = jnp.where(at_boundary_time, jnp.asarray(0.0, dtype=dtbc.dtype), dtbc)
        return replace(self, itimestep=self.itimestep + jnp.asarray(1, dtype=jnp.asarray(self.itimestep).dtype), elapsed_seconds=elapsed, dtbc=dtbc)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelState:
    """Fixed-name PyTree of WRF dynamic arrays plus explicit model time."""

    array_names: tuple[str, ...]
    arrays: tuple[Any, ...]
    time: TimeState

    def __post_init__(self) -> None:
        if len(self.array_names) != len(self.arrays):
            raise ValueError("ModelState array_names and arrays must have the same length")
        if tuple(sorted(self.array_names)) != tuple(self.array_names):
            raise ValueError("ModelState array_names must be sorted for stable PyTree structure")
        if len(set(self.array_names)) != len(self.array_names):
            raise ValueError("ModelState array_names must be unique")

    def tree_flatten(self) -> tuple[tuple[tuple[Any, ...], TimeState], tuple[tuple[str, ...]]]:
        return (self.arrays, self.time), (self.array_names,)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[tuple[str, ...]], children: tuple[tuple[Any, ...], TimeState]) -> "ModelState":
        (array_names,) = aux_data
        arrays, time = children
        return cls(array_names=array_names, arrays=tuple(arrays), time=time)

    def as_dict(self) -> dict[str, Any]:
        return dict(zip(self.array_names, self.arrays, strict=True))

    def select_arrays(self, names: Iterable[str]) -> "ModelState":
        """Project a stage's updates without copying untouched model buffers."""
        selected = tuple(sorted(names))
        values = self.as_dict()
        return ModelState(selected, tuple(values[name] for name in selected), self.time)

    def with_arrays(self, updates: dict[str, Any]) -> "ModelState":
        if not updates:
            return self
        values = self.as_dict()
        unknown = set(updates) - set(values)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"cannot update unknown ModelState arrays: {names}")
        values.update(updates)
        return replace(self, arrays=tuple(values[name] for name in self.array_names))

    def require(self, *names: str) -> tuple[Any, ...]:
        values = self.as_dict()
        return tuple(values[name] for name in names)


__all__ = ["ModelState", "TimeState"]
