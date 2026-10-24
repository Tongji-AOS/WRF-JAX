"""Resident recurrent-state storage for the production WRF-JAX core.

The strict translation keeps one named array per WRF field because that is the
most useful representation at the Python/runtime boundary.  That shape is
not a good long-lived executor carry: a configured case can expose hundreds
of leaves even though many have the same shape and dtype.

``ResidentStateLayout`` provides the next boundary in the architecture.  It
groups compatible fields into fixed buffers once, and ``ResidentState`` keeps
those buffers resident across calls.  A stage can obtain named views for
reading, while a stage result can be applied by updating only the buffers that
contain changed fields.  No host round-trip is involved and the mapping is
static during tracing.

This module intentionally does not decide which WRF fields are prognostic.
The caller supplies a normal ``ModelState`` assembled by the strict/runtime
layer, so the storage remains valid for other cases and configured layouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.state import ModelState, TimeState


def _array_signature(value: Any) -> tuple[tuple[int, ...], str]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError(
            "resident state fields must be array-like, "
            f"got {type(value).__name__}"
        )
    return tuple(int(size) for size in shape), np.dtype(dtype).str


@dataclass(frozen=True)
class ResidentBufferGroup:
    """Static membership and layout of one resident buffer."""

    names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str

    @property
    def buffer_shape(self) -> tuple[int, ...]:
        return (len(self.names), *self.shape)


@dataclass(frozen=True)
class ResidentStateLayout:
    """A reversible, static mapping from named fields to resident buffers."""

    array_names: tuple[str, ...]
    groups: tuple[ResidentBufferGroup, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.array_names)) != self.array_names:
            raise ValueError("resident array_names must be sorted")
        if len(set(self.array_names)) != len(self.array_names):
            raise ValueError("resident array_names must be unique")
        grouped = tuple(name for group in self.groups for name in group.names)
        if tuple(sorted(grouped)) != self.array_names:
            raise ValueError("resident groups must contain every field exactly once")

    @classmethod
    def from_state(cls, state: ModelState) -> "ResidentStateLayout":
        members: dict[tuple[tuple[int, ...], str], list[str]] = {}
        for name, value in zip(state.array_names, state.arrays, strict=True):
            members.setdefault(_array_signature(value), []).append(name)
        groups = tuple(
            ResidentBufferGroup(
                names=tuple(members[key]),
                shape=key[0],
                dtype=key[1],
            )
            for key in sorted(members, key=lambda item: (item[1], item[0]))
        )
        return cls(array_names=state.array_names, groups=groups)

    @property
    def array_count(self) -> int:
        return len(self.array_names)

    @property
    def buffer_count(self) -> int:
        return len(self.groups)

    @property
    def field_to_group_slot(self) -> dict[str, tuple[int, int]]:
        """Return the static field-to-buffer lookup used by update paths."""

        return {
            name: (group_index, slot)
            for group_index, group in enumerate(self.groups)
            for slot, name in enumerate(group.names)
        }

    def _values(self, state: ModelState) -> dict[str, Any]:
        if state.array_names != self.array_names:
            raise ValueError("ModelState names do not match resident layout")
        return state.as_dict()

    def pack(self, state: ModelState) -> "ResidentState":
        """Create resident buffers from a named state at an outer boundary."""

        values = self._values(state)
        buffers: list[Any] = []
        for group in self.groups:
            arrays = tuple(values[name] for name in group.names)
            for name, value in zip(group.names, arrays, strict=True):
                if _array_signature(value) != (group.shape, group.dtype):
                    raise ValueError(
                        f"field {name!r} no longer matches its resident layout"
                    )
            buffers.append(jnp.stack(arrays, axis=0))
        return ResidentState(layout=self, buffers=tuple(buffers), time=state.time)

    def unpack(self, state: "ResidentState") -> ModelState:
        """Expose named zero-copy views of resident buffers inside a JAX graph."""

        if state.layout != self:
            raise ValueError("resident state layout does not match this layout")
        if len(state.buffers) != len(self.groups):
            raise ValueError("resident buffer count does not match layout")
        values: dict[str, Any] = {}
        for group, buffer in zip(self.groups, state.buffers, strict=True):
            if tuple(int(size) for size in buffer.shape) != group.buffer_shape:
                raise ValueError(
                    f"resident buffer shape {tuple(buffer.shape)} does not match "
                    f"{group.buffer_shape}"
                )
            if np.dtype(buffer.dtype).str != group.dtype:
                raise ValueError(
                    f"resident buffer dtype {buffer.dtype} does not match {group.dtype}"
                )
            for slot, name in enumerate(group.names):
                values[name] = buffer[slot]
        return ModelState(
            array_names=self.array_names,
            arrays=tuple(values[name] for name in self.array_names),
            time=state.time,
        )

    def views(self, state: "ResidentState") -> dict[str, Any]:
        """Return the named buffer views without constructing a second state."""

        if state.layout != self:
            raise ValueError("resident state layout does not match this layout")
        values: dict[str, Any] = {}
        for group, buffer in zip(self.groups, state.buffers, strict=True):
            for slot, name in enumerate(group.names):
                values[name] = buffer[slot]
        return values

    def apply_updates(
        self,
        state: "ResidentState",
        updates: Mapping[str, Any] | ModelState,
        *,
        time: TimeState | None = None,
    ) -> "ResidentState":
        """Apply only changed fields, preserving untouched resident buffers.

        ``ModelState`` stage results commonly contain only an update subset;
        accepting both mappings and states makes this method usable while the
        typed stage contracts are being lifted out of the strict harness.
        Each affected group is rebuilt once, rather than stacking all fields
        again at every stage boundary.
        """

        if state.layout != self:
            raise ValueError("resident state layout does not match this layout")
        if isinstance(updates, ModelState):
            if not updates.array_names:
                return replace(state, time=updates.time if time is None else time)
            update_values = updates.as_dict()
            update_time = updates.time
        else:
            update_values = dict(updates)
            update_time = state.time
        unknown = set(update_values) - set(self.array_names)
        if unknown:
            raise KeyError(
                "cannot update unknown resident fields: "
                + ", ".join(sorted(unknown))
            )
        slots = self.field_to_group_slot
        changed_groups: dict[int, dict[int, Any]] = {}
        for name, value in update_values.items():
            group_index, slot = slots[name]
            group = self.groups[group_index]
            if _array_signature(value) != (group.shape, group.dtype):
                raise ValueError(
                    f"field {name!r} update does not match resident layout"
                )
            changed_groups.setdefault(group_index, {})[slot] = value
        buffers = list(state.buffers)
        for group_index, changed in changed_groups.items():
            buffer = buffers[group_index]
            for slot, value in changed.items():
                buffer = buffer.at[slot].set(value)
            buffers[group_index] = buffer
        return ResidentState(
            layout=self,
            buffers=tuple(buffers),
            time=update_time if time is None else time,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ResidentState:
    """A compact PyTree carry whose buffers remain device resident."""

    layout: ResidentStateLayout
    buffers: tuple[Any, ...]
    time: TimeState

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[ResidentStateLayout]]:
        return (*self.buffers, self.time), (self.layout,)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[ResidentStateLayout],
        children: tuple[Any, ...],
    ) -> "ResidentState":
        (layout,) = aux_data
        buffers = tuple(children[: len(layout.groups)])
        (time,) = children[len(layout.groups) :]
        return cls(layout=layout, buffers=buffers, time=time)

    def unpack(self) -> ModelState:
        return self.layout.unpack(self)

    def views(self) -> dict[str, Any]:
        return self.layout.views(self)

    def apply_updates(
        self,
        updates: Mapping[str, Any] | ModelState,
        *,
        time: TimeState | None = None,
    ) -> "ResidentState":
        return self.layout.apply_updates(self, updates, time=time)


__all__ = ["ResidentBufferGroup", "ResidentState", "ResidentStateLayout"]
