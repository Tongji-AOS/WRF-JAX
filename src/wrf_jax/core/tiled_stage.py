"""Persistent row-tiled execution for coarse WRF-JAX stage families.

The stage callable receives a normal field dictionary and may either mutate it
in place or return a mapping of updates.  Horizontal arrays are placed once on
a row mesh, receive explicit neighbor halos inside ``shard_map``, and leave
the stage partitioned, with padded storage for uneven row counts. This is the execution
boundary needed to compose several scientific kernels without a host or
device-0 gather between them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any
import weakref

import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src.interpreters import partial_eval
import numpy as np

from wrf_jax.core.devices import configured_devices
from wrf_jax.core.executable_cache import reusable_terminal_jit
from wrf_jax.core.halo_exchange import crop_halo, exchange_halos
from wrf_jax.core.partition import is_array_parameter


StageFn = Callable[[dict[str, Any]], Mapping[str, Any] | None]


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _row_axis(value: Any, *, global_x: int, global_y: int, padded_y: int | None = None) -> int | None:
    shape = getattr(value, "shape", ())
    if len(shape) < 2 or int(shape[0]) != int(global_x):
        return None
    extents = {int(global_y), int(padded_y if padded_y is not None else global_y)}
    matches = tuple(index for index in range(1, len(shape)) if int(shape[index]) in extents)
    # WRF fields use (i,j) or (i,k,j[,species]); k can equal the row count.
    preferred = 1 if len(shape) == 2 else 2
    if preferred in matches:
        return preferred
    return matches[0] if len(matches) == 1 else None


def _spec(value: Any, axis: int | None) -> jax.sharding.PartitionSpec:
    if axis is None:
        return jax.P()
    parts: list[str | None] = [None] * len(value.shape)
    parts[int(axis)] = "row"
    return jax.P(*parts)


@partial(jax.jit, static_argnames=("axes", "rows", "padded_rows"))
def _pad_physical_rows(values, *, axes, rows, padded_rows):
    result = []
    for value, axis in zip(values, axes, strict=True):
        if axis is not None and value.shape[axis] == rows and padded_rows > rows:
            widths = [(0, 0)] * value.ndim
            widths[axis] = (0, padded_rows - rows)
            value = jnp.pad(value, widths, mode="edge")
        result.append(value)
    return tuple(result)


@partial(jax.jit, static_argnames=("axes", "rows"))
def _join_physical_rows(shard_values, *, axes, rows):
    result = []
    for pieces, axis in zip(shard_values, axes, strict=True):
        value = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=axis)
        if axis is not None and value.shape[axis] != rows:
            selection = [slice(None)] * value.ndim
            selection[axis] = slice(0, rows)
            value = value[tuple(selection)]
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class TiledStateLayout:
    """One reusable placement contract shared by consecutive coarse stages."""

    global_x: int
    global_y: int
    row_shards: int
    mesh: jax.sharding.Mesh

    @classmethod
    def build(cls, *, global_x: int, global_y: int, row_shards: int) -> TiledStateLayout:
        shards = int(row_shards)
        if shards <= 0:
            raise ValueError(f"row_shards must be positive, got {shards}")
        if min(int(global_x), int(global_y)) < 1 or shards > int(global_y):
            raise ValueError("positive grid dimensions and at least one physical row per shard required")
        if (shards - 1) * ((int(global_y) + shards - 1) // shards) >= int(global_y):
            raise ValueError("row_shards would create an empty final tile")
        devices = tuple(jax.local_devices())
        if devices[0].platform == "gpu":
            devices = configured_devices()
        if len(devices) < shards:
            raise ValueError(
                f"tiled state needs {shards} devices but only {len(devices)} are selected"
            )
        mesh = jax.sharding.Mesh(np.asarray(devices[:shards], dtype=object), ("row",))
        return cls(
            global_x=int(global_x),
            global_y=int(global_y),
            row_shards=shards,
            mesh=mesh,
        )

    @property
    def local_rows(self) -> int:
        return (self.global_y + self.row_shards - 1) // self.row_shards

    @property
    def padded_y(self) -> int:
        return self.local_rows * self.row_shards

    def row_bounds(self, shard: int, halo: int = 0) -> dict[str, int]:
        """Global WRF row coordinates for a shard's physical and memory cells."""
        if not 0 <= shard < self.row_shards:
            raise ValueError(f"invalid row shard {shard}")
        start = shard * self.local_rows + 1
        end = min(start + self.local_rows - 1, self.global_y)
        return {
            "jds": 1, "jde": self.global_y,
            "jms": start - halo, "jme": start + self.local_rows - 1 + halo,
            "jps": start, "jpe": end, "jts": start, "jte": end,
        }

    def row_axis(self, value: Any) -> int | None:
        return _row_axis(value, global_x=self.global_x, global_y=self.global_y,
                         padded_y=self.padded_y)

    def place(self, value: Any, axis: int | None) -> Any:
        return self.place_many((value,), (axis,))[0]

    def place_many(self, values, axes):
        """Submit all changed fields together, including replicated parameters."""
        values = _pad_physical_rows(tuple(values), axes=tuple(axes), rows=self.global_y,
                                    padded_rows=self.padded_y)
        shardings = tuple(jax.sharding.NamedSharding(self.mesh, _spec(value, axis))
                          for value, axis in zip(values, axes, strict=True))
        return jax.device_put(values, shardings)

    def gather(self, value: Any, axis: int | None = None) -> Any:
        """Materialize physical rows only at an explicit outer boundary."""
        if axis is None:
            axis = self.row_axis(value)
        return self.gather_many((value,), (axis,))[0]

    def gather_many(self, values, axes=None):
        """Copy local buffers directly and join a stage's fields in one call."""
        if axes is None:
            axes = tuple(self.row_axis(value) for value in values)
        destination = self.mesh.devices.flat[0]
        buffers = []
        for value, axis in zip(values, axes, strict=True):
            if isinstance(value, jax.Array) and not value.sharding.is_fully_replicated:
                if axis is None:
                    raise ValueError("a partitioned field requires a row axis to gather")
                # Resharding a complete array can fall back to NumPy in JAX.
                shards = sorted(value.addressable_shards,
                                key=lambda shard: shard.index[axis].start or 0)
                pieces = tuple(shard.data for shard in shards)
            elif isinstance(value, jax.Array):
                local = next((shard.data for shard in value.addressable_shards
                              if shard.device == destination), value.addressable_shards[0].data)
                pieces = (local,)
            else:
                pieces = (value,)
            buffers.append(pieces)
        buffers = jax.device_put(tuple(buffers), destination)
        return _join_physical_rows(buffers, axes=tuple(axes), rows=self.global_y)

    def sharding(self, value: Any) -> jax.sharding.NamedSharding:
        return jax.sharding.NamedSharding(
            self.mesh,
            _spec(value, self.row_axis(value)),
        )

    def place_fields(
        self,
        fields: Mapping[str, Any],
        *,
        names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Place selected array fields once while preserving outer-shell values."""

        selected = frozenset(fields if names is None else names)
        placed = dict(fields)
        for name in selected:
            value = fields[name]
            if is_array_parameter(value):
                placed[name] = self.place(value, self.row_axis(value))
        return placed


@dataclass(frozen=True)
class TiledStageProgram:
    """A coarse tiled executable and its explicit placement contract."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_specs: tuple[jax.sharding.PartitionSpec, ...]
    output_specs: tuple[jax.sharding.PartitionSpec, ...]
    output_row_axes: tuple[int | None, ...]
    mesh: jax.sharding.Mesh
    row_shards: int
    local_rows: int
    halo_width: int
    execute: Any
    state_layout: TiledStateLayout

    def place(self, fields: Mapping[str, Any]) -> tuple[Any, ...]:
        values = []
        for name, spec in zip(self.input_names, self.input_specs, strict=True):
            axis = spec.index("row") if "row" in spec else None
            values.append(self.state_layout.place(fields[name], axis))
        return tuple(values)

    def outputs(self, values: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(self.output_names, values, strict=True))

    def gather_outputs(self, values: tuple[Any, ...]) -> dict[str, Any]:
        return {
            name: self.state_layout.gather(value, axis)
            for name, value, axis in zip(self.output_names, values, self.output_row_axes, strict=True)
        }

    def apply(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the stage and retain partitioned outputs for its successor."""

        updated = dict(fields)
        updated.update(self.outputs(self.execute(*self.place(fields))))
        return updated

    def apply_placed(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Execute from an already placed state without another placement pass."""

        values = tuple(fields[name] for name in self.input_names)
        updated = dict(fields)
        updated.update(self.outputs(self.execute(*values)))
        return updated


def build_tiled_stage(
    fields: Mapping[str, Any],
    *,
    stage_fn: StageFn,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    output_prototypes: Mapping[str, Any],
    global_x: int,
    global_y: int,
    row_shards: int,
    halo_width: int,
    local_static_updates: Mapping[str, Any] | None = None,
    shard_static_updates: tuple[Mapping[str, Any] | None, ...] | None = None,
    lower_edge_static_updates: Mapping[str, Any] | None = None,
    upper_edge_static_updates: Mapping[str, Any] | None = None,
    copy_upper_x_edge: bool = False,
    copy_upper_row_edge: bool = False,
    restore_upper_x_from_input: tuple[str, ...] = (),
    restore_upper_row_from_input: tuple[str, ...] = (),
    state_layout: TiledStateLayout | None = None,
) -> TiledStageProgram:
    """Build one persistent row-tiled stage-family executable.

    ``input_names`` is deliberately explicit: a coarse stage must own its
    scientific data contract instead of carrying the entire compatibility
    dictionary through every compiler call.
    """

    names = tuple(dict.fromkeys(input_names))
    outputs = tuple(dict.fromkeys(output_names))
    if any(name not in fields for name in names):
        missing = tuple(name for name in names if name not in fields)
        raise KeyError(f"tiled stage input fields are missing: {missing}")
    if any(not is_array_parameter(fields[name]) for name in names):
        invalid = tuple(name for name in names if not is_array_parameter(fields[name]))
        raise TypeError(f"tiled stage inputs must be arrays or array parameter trees: {invalid}")
    if any(name not in output_prototypes for name in outputs):
        missing = tuple(name for name in outputs if name not in output_prototypes)
        raise KeyError(f"tiled stage output prototypes are missing: {missing}")

    shards = int(row_shards)
    halo = int(halo_width)
    if shards <= 0:
        raise ValueError(f"row_shards must be positive, got {shards}")
    local_rows = (int(global_y) + shards - 1) // shards
    if halo < 0:
        raise ValueError(f"halo_width must not be negative, got {halo}")
    layout = state_layout or TiledStateLayout.build(
        global_x=int(global_x),
        global_y=int(global_y),
        row_shards=shards,
    )
    if (
        layout.global_x != int(global_x)
        or layout.global_y != int(global_y)
        or layout.row_shards != shards
    ):
        raise ValueError("tiled stage dimensions do not match the shared state layout")
    mesh = layout.mesh

    if shard_static_updates is not None:
        if len(shard_static_updates) != shards:
            raise ValueError(
                "shard_static_updates must contain one mapping per row shard "
                f"({shards}), got {len(shard_static_updates)}"
            )

    input_values = tuple(fields[name] for name in names)
    input_leaf_counts = tuple(len(jax.tree_util.tree_leaves(value)) for value in input_values)
    input_axes = tuple(
        _row_axis(value, global_x=int(global_x), global_y=int(global_y))
        for value in input_values
    )
    input_specs = tuple(_spec(value, axis) for value, axis in zip(input_values, input_axes, strict=True))
    output_values = tuple(output_prototypes[name] for name in outputs)
    output_axes = tuple(
        _row_axis(value, global_x=int(global_x), global_y=int(global_y))
        for value in output_values
    )
    output_specs = tuple(_spec(value, axis) for value, axis in zip(output_values, output_axes, strict=True))

    dynamic_names = frozenset(names)
    static_fields = {
        name: value
        for name, value in fields.items()
        if name not in dynamic_names and not is_array_parameter(value)
    }
    if local_static_updates:
        static_fields.update(local_static_updates)

    # Build the complete per-shard static contract once, outside the traced
    # function.  WRF bounds are Python integers consumed as static arguments by
    # many translated kernels, so they cannot be selected with an array value
    # inside the kernel.  ``lax.switch`` below selects among these compile-time
    # variants while keeping the dynamic field buffers resident on their
    # devices.
    shard_static_variants: tuple[Mapping[str, Any], ...] | None = None
    if shard_static_updates is not None:
        variants: list[Mapping[str, Any]] = []
        for shard_index, shard_updates in enumerate(shard_static_updates):
            merged = dict(shard_updates or {})
            if shard_index == 0 and lower_edge_static_updates:
                merged.update(lower_edge_static_updates)
            if shard_index == shards - 1 and upper_edge_static_updates:
                merged.update(upper_edge_static_updates)
            variants.append(merged)
        shard_static_variants = tuple(variants)

    original_names = names

    def run_stage(values, static_updates):
        local_fields = dict(static_fields)
        local_fields.update(zip(original_names, values, strict=True))
        if static_updates:
            local_fields.update(static_updates)
        updates = stage_fn(local_fields)
        if updates:
            local_fields.update(updates)
        return tuple(local_fields[name] for name in outputs)

    abstracts = []
    for value, axis in zip(input_values, input_axes, strict=True):
        def abstract(leaf):
            shape = list(leaf.shape)
            if axis is not None:
                shape[axis] = local_rows + 2 * halo
            return jax.ShapeDtypeStruct(tuple(shape), leaf.dtype)
        abstracts.append(jax.tree_util.tree_map(abstract, value))
    variants = shard_static_variants or (None, lower_edge_static_updates, upper_edge_static_updates)
    traced = []
    used = [False] * sum(input_leaf_counts)
    for variant in variants:
        closed = jax.make_jaxpr(lambda *xs: run_stage(xs, variant))(*abstracts)
        pruned, mask = partial_eval.dce_jaxpr(closed.jaxpr, [True] * len(outputs))
        traced.append((pruned, closed.consts, mask))
        used = [left or right for left, right in zip(used, mask, strict=True)]
    offset = 0
    used_fields = []
    for count in input_leaf_counts:
        used_fields.append(any(used[offset:offset + count]))
        offset += count
    for name in (*restore_upper_x_from_input, *restore_upper_row_from_input):
        if name in names:
            used_fields[names.index(name)] = True
    # Parameter trees keep their explicit argument structure, while each
    # branch consumes only the live leaves of that structure.
    used = [keep for keep, count in zip(used_fields, input_leaf_counts, strict=True)
            for _ in range(count)]
    # Prune before packing faces: otherwise one used field makes every field's
    # communication live, even when XLA would discard its computation.
    names = tuple(name for name, keep in zip(names, used_fields, strict=True) if keep)
    input_axes = tuple(axis for axis, keep in zip(input_axes, used_fields, strict=True) if keep)
    input_specs = tuple(spec for spec, keep in zip(input_specs, used_fields, strict=True) if keep)
    kept_values = tuple(value for value, keep in zip(input_values, used_fields, strict=True) if keep)
    input_tree = jax.tree_util.tree_structure(kept_values)
    leaf_axes = tuple(axis for axis, value in zip(input_axes, kept_values, strict=True)
                      for _ in jax.tree_util.tree_leaves(value))
    branches = []
    for jaxpr, consts, mask in traced:
        indices = tuple(index for index, keep in enumerate(
            required for required, live in zip(mask, used, strict=True) if live
        ) if keep)

        def evaluate(values, jaxpr=jaxpr, consts=consts, indices=indices):
            return tuple(jax_core.eval_jaxpr(jaxpr, consts, *(values[index] for index in indices)))

        branches.append(evaluate)

    def local_stage(*local_values: Any) -> tuple[Any, ...]:
        extended_values = exchange_halos(
            tuple(jax.tree_util.tree_leaves(local_values)), axes=leaf_axes, width=halo,
            axis_name="row", shard_count=shards, periodic=False,
        )

        if shard_static_variants is not None:
            shard_index = jax.lax.axis_index("row")
            raw_outputs = jax.lax.switch(shard_index, tuple(branches), extended_values)
        elif lower_edge_static_updates is not None or upper_edge_static_updates is not None:
            shard_index = jax.lax.axis_index("row")

            def run_nonlower(values: tuple[Any, ...]) -> tuple[Any, ...]:
                return jax.lax.cond(
                    shard_index == shards - 1,
                    branches[2],
                    branches[0],
                    values,
                )

            raw_outputs = jax.lax.cond(
                shard_index == 0,
                branches[1],
                run_nonlower,
                extended_values,
            )
        else:
            raw_outputs = branches[0](extended_values)

        input_by_name = dict(zip(names, input_tree.unflatten(extended_values), strict=True))
        restore_x = frozenset(restore_upper_x_from_input)
        restore_row = frozenset(restore_upper_row_from_input)
        result: list[Any] = []
        for name, value, axis in zip(outputs, raw_outputs, output_axes, strict=True):
            if axis is not None:
                target_rows = local_rows + 2 * halo
                actual_rows = int(value.shape[int(axis)])
                if actual_rows < target_rows:
                    raise ValueError(
                        f"tiled stage output row extent {actual_rows} is smaller than {target_rows}"
                    )
                if actual_rows > target_rows:
                    selection = [slice(None)] * int(value.ndim)
                    selection[int(axis)] = slice(0, target_rows)
                    value = value[tuple(selection)]
            source = input_by_name.get(name)
            if name in restore_x and source is not None:
                value = value.at[-1].set(source[-1])
            elif copy_upper_x_edge and value.ndim >= 1 and int(value.shape[0]) == int(global_x):
                value = value.at[-1].set(value[-2])
            if name in restore_row and source is not None and axis is not None:
                upper = [slice(None)] * int(value.ndim)
                upper[int(axis)] = halo + layout.global_y - (shards - 1) * local_rows - 1
                restored = value.at[tuple(upper)].set(source[tuple(upper)])
                value = jax.lax.cond(
                    jax.lax.axis_index("row") == shards - 1,
                    lambda _: restored,
                    lambda _: value,
                    operand=None,
                )
            elif copy_upper_row_edge and axis is not None:
                upper = [slice(None)] * int(value.ndim)
                source = [slice(None)] * int(value.ndim)
                upper[int(axis)] = halo + layout.global_y - (shards - 1) * local_rows - 1
                source[int(axis)] = upper[int(axis)] - 1
                replaced = value.at[tuple(upper)].set(value[tuple(source)])
                value = jax.lax.cond(
                    jax.lax.axis_index("row") == shards - 1,
                    lambda _: replaced,
                    lambda _: value,
                    operand=None,
                )
            if axis is not None:
                value = crop_halo(value, axis=int(axis), width=halo)
                if layout.padded_y != layout.global_y:
                    # Refresh padding from the new physical edge, so a later
                    # stage never consumes stale values beyond the domain.
                    last = layout.global_y - (shards - 1) * local_rows - 1
                    indices = jnp.minimum(jnp.arange(local_rows), last)
                    value = jax.lax.cond(
                        jax.lax.axis_index("row") == shards - 1,
                        lambda x: jnp.take(x, indices, axis=int(axis)),
                        lambda x: x,
                        value,
                    )
            result.append(value)
        return tuple(result)

    execute = jax.jit(
        jax.shard_map(
            local_stage,
            mesh=mesh,
            in_specs=input_specs,
            out_specs=output_specs,
            check_vma=False,
        ),
        inline=False,
    )
    return TiledStageProgram(
        input_names=names,
        output_names=outputs,
        input_specs=input_specs,
        output_specs=output_specs,
        output_row_axes=output_axes,
        mesh=mesh,
        row_shards=shards,
        local_rows=local_rows,
        halo_width=halo,
        execute=execute,
        state_layout=layout,
    )


class TiledFieldSequence:
    """Prepare consecutive field stages once and retain their distributed buffers."""

    def __init__(self, stage_functions, *, global_x, global_y, row_shards, halo_width,
                 tile_metadata=None, output_names=None):
        self.stage_functions = tuple(stage_functions)
        self.layout = TiledStateLayout.build(
            global_x=global_x, global_y=global_y, row_shards=row_shards,
        )
        self.halo_widths = ((halo_width,) * len(self.stage_functions)
                            if isinstance(halo_width, int) else tuple(halo_width))
        if len(self.halo_widths) != len(self.stage_functions):
            raise ValueError("each tiled stage must have a halo width")
        self.tile_metadata = tile_metadata
        self.retained_outputs = None if output_names is None else frozenset(output_names)
        self.programs = None
        self._placed_inputs = {}

    def _prepare(self, fields):
        prototypes = dict(fields)
        stages = []
        produced = set()
        tile_metadata = {}
        for stage_fn, halo in zip(self.stage_functions, self.halo_widths, strict=True):
            array_names = tuple(name for name, value in prototypes.items() if is_array_parameter(value))
            metadata = {name: value for name, value in prototypes.items() if name not in array_names}

            def abstract_stage(*arrays):
                state = dict(metadata)
                state.update(zip(array_names, arrays, strict=True))
                before = dict(state)
                updates = stage_fn(state)
                if updates:
                    state.update(updates)
                return {name: value for name, value in state.items()
                        if _is_array(value) and value is not before.get(name)}

            abstracts = tuple(jax.tree_util.tree_map(
                lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), prototypes[name],
            ) for name in array_names)
            outputs = jax.eval_shape(abstract_stage, *abstracts)
            variants = []
            for shard in range(self.layout.row_shards):
                bounds = self.layout.row_bounds(shard, halo)
                for suffix in ("x", "y"):
                    for key in ("jms", "jme", "jps", "jpe"):
                        bounds[key + suffix] = bounds[key]
                if self.tile_metadata is not None and halo <= 1:
                    key = (shard, halo)
                    if key not in tile_metadata:
                        tile_metadata[key] = self.tile_metadata(fields, bounds)
                    bounds.update(tile_metadata[key])
                variants.append(bounds)
            stages.append((stage_fn, halo, dict(prototypes), array_names,
                           outputs, tuple(variants)))
            produced.update(outputs)
            prototypes.update(outputs)

        outputs = produced if self.retained_outputs is None else produced & self.retained_outputs
        required = set(outputs)
        programs = []
        # Walk backwards so overwritten scratch fields never cross a stage
        # boundary unless a later computation actually consumes their value.
        for stage_fn, halo, inputs, array_names, prototypes, variants in reversed(stages):
            names = tuple(name for name in prototypes if name in required)
            if not names:
                continue
            program = build_tiled_stage(
                inputs, stage_fn=stage_fn, input_names=array_names,
                output_names=names, output_prototypes=prototypes,
                global_x=self.layout.global_x, global_y=self.layout.global_y,
                row_shards=self.layout.row_shards, halo_width=halo,
                shard_static_updates=variants, state_layout=self.layout,
            )
            required.difference_update(program.output_names)
            required.update(program.input_names)
            programs.append(program)
        self.input_names = tuple(sorted(required))
        self.output_names = tuple(sorted(outputs))
        self.programs = tuple(reversed(programs))

        def execute(*arrays):
            state = dict(zip(self.input_names, arrays, strict=True))
            for index, program in enumerate(self.programs):
                state = program.apply_placed(state)
                if index + 1 < len(self.programs):
                    # Preserve the rounding boundary of separately submitted stages.
                    updates = jax.lax.optimization_barrier(tuple(state[name] for name in program.output_names))
                    state.update(zip(program.output_names, updates, strict=True))
            return tuple(state[name] for name in self.output_names)

        # The composed program can reuse unchanged kernels from its stages.
        kernel_family = (build_tiled_stage.__module__,
                         f"{build_tiled_stage.__qualname__}.<locals>.local_stage")
        self.execute = reusable_terminal_jit(execute, preserve_call=True, kernel_family=kernel_family, context={
            "program": "tiled_field_sequence", "stages": self.stage_functions,
            "metadata": {name: value for name, value in fields.items()
                         if not is_array_parameter(value)},
            "inputs": self.input_names, "outputs": self.output_names,
            "grid": (self.layout.global_x, self.layout.global_y),
            "row_shards": self.layout.row_shards, "halo_widths": self.halo_widths,
            "tile_metadata": tuple(tile_metadata.values()),
        })

    def __call__(self, fields):
        if self.programs is None:
            self._prepare(fields)
        inputs = tuple(fields[name] for name in self.input_names)
        placed_by_id = {}
        missing = {}
        for value in inputs:
            key = id(value)
            cached = self._placed_inputs.get(key)
            if cached is not None and cached[0]() is value:
                placed_by_id[key] = cached[1]
            else:
                missing[key] = value
        if missing:
            values = tuple(missing.values())
            placed_values = self.layout.place_many(values, tuple(self.layout.row_axis(value) for value in values))
        else:
            placed_values = ()
        for (key, value), placed in zip(missing.items(), placed_values, strict=True):
            placed_by_id[key] = placed
            if placed is not value:
                # Reuse invariant buffers without retaining obsolete time steps.
                cache = self._placed_inputs
                try:
                    reference = weakref.ref(value, lambda _ref, key=key: cache.pop(key, None))
                except TypeError:
                    pass
                else:
                    cache[key] = (reference, placed)
        values = tuple(placed_by_id[id(value)] for value in inputs)
        outputs = self.execute(*values)
        return dict(zip(self.output_names, self.layout.gather_many(outputs), strict=True))


__all__ = ["TiledFieldSequence", "TiledStageProgram", "TiledStateLayout", "build_tiled_stage"]
