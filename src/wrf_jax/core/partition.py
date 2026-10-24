"""Runtime-field partition helpers for the production core boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.state import ModelState, TimeState


def is_dynamic_array_value(value: Any) -> bool:
    """Return whether ``value`` should be treated as an array leaf."""

    return isinstance(value, (jax.Array, np.ndarray)) or (hasattr(value, "shape") and hasattr(value, "dtype"))


def array_nbytes(value: Any) -> int:
    """Return the byte size of arrays in a field or parameter PyTree."""

    return sum(int(leaf.size) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree_util.tree_leaves(value))


def is_array_parameter(value: Any) -> bool:
    """Return whether a field is an array or a parameter PyTree of arrays."""
    leaves = jax.tree_util.tree_leaves(value)
    return bool(leaves) and all(is_dynamic_array_value(leaf) for leaf in leaves)


def split_static_dynamic_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split runtime fields into scalar/static values and array-like leaves."""

    dynamic = {name: value for name, value in fields.items() if is_dynamic_array_value(value)}
    dynamic.setdefault("dtbc", jnp.asarray(fields.get("dtbc", 0.0), dtype=jnp.float32))
    static = {name: value for name, value in fields.items() if name not in dynamic}
    return static, dynamic


@dataclass(frozen=True)
class FieldPartition:
    """Explicit runtime-field split used at compiled core boundaries."""

    dynamic_fields: dict[str, Any]
    static_array_fields: dict[str, Any]
    static_scalar_fields: dict[str, Any]
    input_buffer_copies: int = 0

    @property
    def dynamic_names(self) -> tuple[str, ...]:
        return tuple(self.dynamic_fields)

    @property
    def static_array_bytes(self) -> int:
        return sum(array_nbytes(value) for value in self.static_array_fields.values())

    @property
    def state_array_names(self) -> tuple[str, ...]:
        """Sorted dynamic array names carried by ``ModelState``.

        Model time fields are kept in ``TimeState`` instead of duplicating them
        as ordinary arrays.  This keeps the eventual scan carry shaped like the
        production state rather than a strict-runtime dictionary.
        """

        return tuple(sorted(name for name in self.dynamic_fields if name != "dtbc"))

    def model_state(self, grid: Any) -> ModelState:
        """Return the dynamic carry as a stable typed PyTree state."""

        dtbc = self.dynamic_fields.get("dtbc", getattr(grid, "dtbc", 0.0))
        itimestep = self.static_scalar_fields.get("itimestep", getattr(grid, "itimestep", 0))
        elapsed_seconds = self.static_scalar_fields.get("elapsed_seconds", getattr(grid, "current_time", 0.0))
        names = self.state_array_names
        return ModelState(
            array_names=names,
            arrays=tuple(self.dynamic_fields[name] for name in names),
            time=TimeState(
                itimestep=jnp.asarray(_scalar_value(itimestep, 0), dtype=jnp.int32),
                elapsed_seconds=jnp.asarray(_scalar_value(elapsed_seconds, 0.0), dtype=jnp.float32),
                dtbc=jnp.asarray(dtbc, dtype=jnp.float32),
            ),
        )

    def dynamic_fields_from_state(self, state: ModelState) -> dict[str, Any]:
        """Convert a typed state back to the partition's dynamic field order."""

        return model_state_to_dynamic_fields(state, self.dynamic_names)


def model_state_to_dynamic_fields(state: ModelState, names: Iterable[str]) -> dict[str, Any]:
    """Rebuild fields from a name-only layout, without retaining old arrays."""
    names = tuple(names)
    values = state.as_dict()
    if "dtbc" in names:
        values["dtbc"] = state.time.dtbc
    missing = [name for name in names if name not in values]
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise KeyError(f"cannot rebuild dynamic fields from ModelState; missing fields: {missing_names}")
    return {name: values[name] for name in names}


def _scalar_value(value: Any, default: Any) -> Any:
    try:
        arr = np.asarray(jax.device_get(value)).reshape(-1)
    except Exception:
        return default
    if arr.size == 0:
        return default
    item = arr[0]
    try:
        return item.item()
    except AttributeError:
        return item


def _array_buffer_key(value: Any) -> tuple[int, tuple[int, ...], np.dtype[Any]] | None:
    try:
        array = value if isinstance(value, jax.Array) else jnp.asarray(value)
        if not isinstance(array, jax.Array):
            return None
        return (int(array.unsafe_buffer_pointer()), array.shape, array.dtype)
    except Exception:
        return None


def _deduplicate_buffers_against(
    fields: Mapping[str, Any],
    seen: set[tuple[int, tuple[int, ...], np.dtype[Any]]],
    *,
    remember_unique: bool,
) -> tuple[dict[str, Any], int]:
    """Copy leaves that alias buffers already present in ``seen``."""

    deduplicated: dict[str, Any] = {}
    copies = 0
    def separate(value):
        nonlocal copies
        key = _array_buffer_key(value)
        if key is not None and key in seen:
            copies += 1
            return value.copy() if isinstance(value, jax.Array) else jnp.asarray(value).copy()
        else:
            if key is not None and remember_unique:
                seen.add(key)
            return value

    for name, value in fields.items():
        deduplicated[name] = separate(value) if isinstance(value, jax.Array) else jax.tree_util.tree_map(separate, value)
    return deduplicated, copies


def deduplicate_input_buffers(dynamic_fields: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Copy later aliases so a donated PyTree does not repeat a JAX buffer."""

    return _deduplicate_buffers_against(dynamic_fields, set(), remember_unique=True)


def partition_fields_for_compiled_step(
    fields: Mapping[str, Any],
    *,
    dynamic_names: Iterable[str],
    prune_static_fields: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    deduplicate_buffers: bool = True,
) -> FieldPartition:
    """Build the explicit field split consumed by compiled step builders."""

    dynamic_name_tuple = tuple(dynamic_names)
    dynamic_key_set = set(dynamic_name_tuple)
    missing = sorted(name for name in dynamic_key_set if name not in fields)
    if missing:
        raise KeyError(f"cannot partition compiled step fields; missing dynamic fields: {', '.join(missing)}")
    dynamic_fields = {name: fields[name] for name in dynamic_name_tuple}
    static_fields = {name: value for name, value in fields.items() if name not in dynamic_fields}
    if prune_static_fields is not None:
        static_fields = prune_static_fields(static_fields)
    static_array_fields = {
        name: value for name, value in static_fields.items() if is_array_parameter(value)
    }
    static_scalar_fields = {
        name: value for name, value in static_fields.items() if name not in static_array_fields
    }
    copy_count = 0
    if deduplicate_buffers:
        dynamic_fields, copy_count = deduplicate_input_buffers(dynamic_fields)
        donated_buffer_keys = {
            key
            for value in dynamic_fields.values()
            if (key := _array_buffer_key(value)) is not None
        }
        # ``ModelState`` is donated to the compiled step, whereas Grid,
        # Config, and Forcing arrays are retained across calls.  A retained
        # array may still be a Python alias of an evolving state field after
        # strict-runtime assembly.  Separate those aliases once at the core
        # boundary so XLA never receives the same buffer as both donated and
        # non-donated input.  Static-to-static aliases are safe and remain
        # shared.
        static_array_fields, static_copy_count = _deduplicate_buffers_against(
            static_array_fields,
            donated_buffer_keys,
            remember_unique=False,
        )
        copy_count += static_copy_count
    return FieldPartition(
        dynamic_fields=dynamic_fields,
        static_array_fields=static_array_fields,
        static_scalar_fields=static_scalar_fields,
        input_buffer_copies=copy_count,
    )


__all__ = [
    "array_nbytes",
    "deduplicate_input_buffers",
    "FieldPartition",
    "is_array_parameter",
    "is_dynamic_array_value",
    "model_state_to_dynamic_fields",
    "partition_fields_for_compiled_step",
    "split_static_dynamic_fields",
]
