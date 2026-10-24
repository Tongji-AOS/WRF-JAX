"""Device-local halo exchange primitives for horizontally tiled WRF fields."""

from __future__ import annotations

from typing import Any, Hashable

import jax
import jax.numpy as jnp


def _axis_slice(value: Any, axis: int, start: int | None, stop: int | None) -> Any:
    index = [slice(None)] * int(jnp.ndim(value))
    index[int(axis)] = slice(start, stop)
    return value[tuple(index)]


def exchange_halo(
    local_value: Any,
    *,
    axis: int,
    width: int,
    axis_name: Hashable,
    shard_count: int,
    periodic: bool = False,
) -> Any:
    """Add neighbor-owned halos to one local shard.

    The function is intended for ``shard_map`` or a named ``vmap``.  Interior
    halos come from the adjacent shard through ``lax.ppermute``.  Non-periodic
    outer edges extend their nearest physical cell; physical/specification boundary
    kernels remain responsible for the scientific boundary values.
    """

    return exchange_halos(
        (jnp.asarray(local_value),), axes=(axis,), width=width, axis_name=axis_name,
        shard_count=shard_count, periodic=periodic,
    )[0]


def exchange_halos(
    local_values: tuple[Any, ...],
    *,
    axes: tuple[int | None, ...],
    width: int,
    axis_name: Hashable,
    shard_count: int,
    periodic: bool = False,
) -> tuple[Any, ...]:
    """Exchange a stage's faces in one message per direction and dtype.

    Packing amortizes collective launch overhead across fields. Replicated
    leaves have axis=None and are returned unchanged. Every value and its
    physical boundary convention match exchange_halo.
    """
    if len(local_values) != len(axes):
        raise ValueError("one halo axis per local value required")
    width_i = int(width)
    count = int(shard_count)
    if width_i <= 0:
        return tuple(local_values)
    if count <= 0:
        raise ValueError(f"shard_count must be positive, got {count}")
    grouped = {}
    result = list(local_values)
    for index, (value, axis) in enumerate(zip(local_values, axes, strict=True)):
        if axis is None:
            continue
        array = jnp.asarray(value)
        axis_i = int(axis)
        if axis_i < 0:
            axis_i += array.ndim
        if axis_i < 0 or axis_i >= array.ndim:
            raise ValueError(f"halo axis {axis} is invalid for rank-{array.ndim} value")
        chunk_width = min(width_i, int(array.shape[axis_i]))
        if chunk_width < 1:
            raise ValueError("cannot exchange a halo for an empty local axis")
        lower = _axis_slice(array, axis_i, 0, chunk_width)
        upper = _axis_slice(array, axis_i, -chunk_width, None)
        grouped.setdefault((array.dtype, chunk_width), []).append((index, axis_i, array, lower, upper))
    send_to_upper = tuple((source, (source + 1) % count) for source in range(count))
    send_to_lower = tuple((source, (source - 1) % count) for source in range(count))
    shard_index = jax.lax.axis_index(axis_name)
    for (_, chunk_width), fields in grouped.items():
        own_lower = jnp.concatenate([lower.reshape(-1) for _, _, _, lower, _ in fields])
        own_upper = jnp.concatenate([upper.reshape(-1) for _, _, _, _, upper in fields])
        if not periodic:
            boundary_lower = jnp.concatenate([
                jnp.broadcast_to(_axis_slice(array, axis_i, 0, 1), lower.shape).reshape(-1)
                for _, axis_i, array, lower, _ in fields
            ])
            boundary_upper = jnp.concatenate([
                jnp.broadcast_to(_axis_slice(array, axis_i, -1, None), upper.shape).reshape(-1)
                for _, axis_i, array, _, upper in fields
            ])
        lower_halo, upper_halo = own_upper, own_lower
        lower_chunks, upper_chunks = [], []
        # Large device counts can make a tile narrower than the stencil. Relay
        # whole neighboring tiles until every requested row is available.
        for _ in range((width_i + chunk_width - 1) // chunk_width):
            lower_halo = jax.lax.ppermute(lower_halo, axis_name, send_to_upper)
            upper_halo = jax.lax.ppermute(upper_halo, axis_name, send_to_lower)
            if not periodic:
                lower_halo = jnp.where(shard_index == 0, boundary_lower, lower_halo)
                upper_halo = jnp.where(shard_index == count - 1, boundary_upper, upper_halo)
            lower_chunks.append(lower_halo)
            upper_chunks.append(upper_halo)
        offset = 0
        for index, axis_i, array, lower, _ in fields:
            end = offset + lower.size
            lower_rows = jnp.concatenate(tuple(chunk[offset:end].reshape(lower.shape)
                                                for chunk in reversed(lower_chunks)), axis=axis_i)
            upper_rows = jnp.concatenate(tuple(chunk[offset:end].reshape(lower.shape)
                                                for chunk in upper_chunks), axis=axis_i)
            result[index] = jnp.concatenate((
                _axis_slice(lower_rows, axis_i, -width_i, None), array,
                _axis_slice(upper_rows, axis_i, 0, width_i),
            ), axis=axis_i)
            offset = end
    return tuple(result)


def crop_halo(value: Any, *, axis: int, width: int) -> Any:
    """Remove equal lower and upper halo slabs."""

    array = jnp.asarray(value)
    width_i = int(width)
    if width_i <= 0:
        return array
    return _axis_slice(array, int(axis), width_i, -width_i)


__all__ = ["crop_halo", "exchange_halo", "exchange_halos"]
