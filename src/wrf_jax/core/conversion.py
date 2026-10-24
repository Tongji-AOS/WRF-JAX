"""Conversion between the strict runtime dictionary and core PyTree state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from wrf_jax.core.field_sets import CONFIGURED_DYNAMIC_ARRAY_NAMES
from wrf_jax.core.state import ModelState, TimeState


def is_state_array(value: Any) -> bool:
    """Return whether a runtime field is a numeric array carried by ModelState."""

    if not hasattr(value, "shape"):
        return False
    try:
        dtype = np.dtype(getattr(value, "dtype"))
    except TypeError:
        return False
    return dtype.kind in "biufc"


def state_array_names(fields: Mapping[str, Any], *, dynamic_only: bool = False) -> tuple[str, ...]:
    """Stable sorted names of numeric runtime arrays."""

    names = tuple(sorted(name for name, value in fields.items() if is_state_array(value)))
    if not dynamic_only:
        return names
    dynamic = set(CONFIGURED_DYNAMIC_ARRAY_NAMES)
    return tuple(name for name in names if name in dynamic)


def fields_to_model_state(
    fields: Mapping[str, Any],
    grid: Any,
    *,
    dynamic_only: bool = False,
    array_names: Iterable[str] | None = None,
) -> ModelState:
    """Build a fixed-name PyTree from strict runtime fields and grid time."""

    if array_names is None:
        names = state_array_names(fields, dynamic_only=dynamic_only)
    else:
        names = tuple(sorted(dict.fromkeys(str(name) for name in array_names)))
        missing = [name for name in names if name not in fields]
        if missing:
            raise KeyError(f"cannot build ModelState; missing fields: {', '.join(missing)}")
        non_arrays = [name for name in names if not is_state_array(fields[name])]
        if non_arrays:
            raise TypeError(f"ModelState fields must be numeric arrays: {', '.join(non_arrays)}")
    return ModelState(
        array_names=names,
        arrays=tuple(fields[name] for name in names),
        time=TimeState.from_grid(grid),
    )


def model_state_to_fields(state: ModelState, fields_template: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay ModelState arrays onto a strict runtime field template."""

    fields = dict(fields_template)
    fields.update(state.as_dict())
    return fields


def apply_time_state_to_grid(time: TimeState, grid: Any) -> None:
    """Write explicit model time back to the strict grid object."""

    setattr(grid, "itimestep", int(np.asarray(time.itimestep)))
    setattr(grid, "current_time", float(np.asarray(time.elapsed_seconds)))
    setattr(grid, "dtbc", float(np.asarray(time.dtbc)))
    clock = getattr(grid, "domain_clock", None)
    clockint = getattr(clock, "clockint", None)
    curr_time = getattr(clockint, "curr_time", None)
    if curr_time is not None and hasattr(curr_time, "base"):
        curr_time.base.s = int(np.asarray(time.elapsed_seconds))


__all__ = [
    "apply_time_state_to_grid",
    "fields_to_model_state",
    "is_state_array",
    "model_state_to_fields",
    "state_array_names",
]
