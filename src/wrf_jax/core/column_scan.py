"""Execute the same pointwise vertical recurrence on CPU and CUDA."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
from jax import lax
import jax.numpy as jnp


def _ordinary(body, initial, inputs, parameter, *, reverse=False):
    return lax.scan(
        lambda carry, xs: body(carry, xs, parameter), initial, inputs, reverse=reverse
    )


def _result_shapes(body, initial, inputs, parameter):
    layer = tuple(jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in inputs)
    return jax.eval_shape(body, initial, layer, parameter)


def _column_kernel(body, initial, inputs, parameter, *, interpret=False, reverse=False):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    from wrf_jax.core.real_arithmetic import register_column_lowerings

    register_column_lowerings()

    block = 128
    leaves = jax.tree.leaves(initial)
    shape = leaves[0].shape
    levels, columns = inputs[0].shape[0], leaves[0].size
    width = shape[-1] if shape else 1
    rows = columns // width
    # Keep the model's row/level/inner-column layout across the custom call.
    # Flattening all columns behind the level axis forces whole-field copies.
    flat_inputs = tuple(
        x if x.ndim == 1 else jnp.swapaxes(x.reshape(levels, rows, width), 0, 1) for x in inputs
    )

    def kernel(initial_refs, input_refs, parameter_ref, last_refs, values_refs):
        indices = pl.program_id(0) * block + jnp.arange(block)
        mask = indices < columns
        row, inner = indices // width, indices % width
        carry = jax.tree.map(lambda ref: plt.load(ref.at[indices], mask=mask, other=0), initial_refs)
        value_parameter = jax.tree.map(lambda ref: ref[()] if ref.ndim == 0 else ref, parameter_ref)

        def layer(iteration, previous):
            k = levels - 1 - iteration if reverse else iteration
            xs = tuple(
                ref[k] if ref.ndim == 1 else plt.load(ref.at[row, k, inner], mask=mask, other=0)
                for ref in input_refs
            )
            current, value = body(previous, xs, value_parameter)
            jax.tree.map(lambda ref, x: plt.store(ref.at[row, k, inner], x, mask=mask), values_refs, value)
            return current

        last = lax.fori_loop(0, levels, layer, carry)
        jax.tree.map(lambda ref, x: plt.store(ref.at[indices], x, mask=mask), last_refs, last)

    last_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct((columns,), x.dtype), initial)
    _, history = _result_shapes(body, initial, inputs, parameter)
    history_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct((rows, levels, width), x.dtype), history)
    last, values = pl.pallas_call(
        kernel,
        out_shape=(last_shape, history_shape),
        grid=((columns + block - 1) // block,),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4),
    )(jax.tree.map(lambda x: x.reshape(-1), initial), flat_inputs, parameter)
    return (
        jax.tree.map(lambda x: x.reshape(shape), last),
        jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).reshape((levels, *shape)), values),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 4))
def _dispatch(body, initial, inputs, parameter, reverse):
    return lax.platform_dependent(
        initial,
        inputs,
        parameter,
        cuda=partial(_column_kernel, body, reverse=reverse),
        default=partial(_ordinary, body, reverse=reverse),
    )


@_dispatch.defjvp
def _dispatch_jvp(body, reverse, primals, tangents):
    # Differentiate the shared recurrence, not a separate scientific formula.
    primal = _dispatch(body, *primals, reverse)
    _, tangent = jax.jvp(partial(_ordinary, body, reverse=reverse), primals, tangents)
    return primal, tangent


def scan_columns(
    body: Callable,
    initial: Any,
    inputs: tuple[Any, ...],
    parameter: Any = None,
    *,
    reverse: bool = False,
) -> tuple[Any, Any]:
    """Scan levels of independent columns with optional read-only parameters.

    ``body(carry, layer_inputs, parameter)`` must have column-local numerical
    outputs: no cross-column arithmetic, stencils, or captured dynamic arrays.
    A tile-wide predicate may skip fully inactive work only if every column's
    result is independent of the other columns present in that tile. Parameter
    leaves may be scalars or one-dimensional lookup tables, indexed explicitly
    in the body; the tables must never be mutated.
    Carry keeps the structure, shapes and dtypes of ``initial``. Per-level
    output may have a different PyTree structure. All carry and output leaves
    share a column shape but may mix floating, integer and boolean dtypes.
    Inputs have shape ``(levels, *column_shape)`` or ``(levels,)`` for a
    per-level scalar shared by columns. CUDA keeps the recurrence
    inside one kernel; other devices and derivatives use the same body in scan.
    With reverse=True, visit levels from top to bottom while storing each
    history value at its input level, matching lax.scan(reverse=True).
    """
    # Under disable_jit, tracing a Pallas fori_loop unrolls every layer before
    # compiling the kernel. Debug the ordinary shared recurrence instead.
    if jax.config.jax_disable_jit:
        return _ordinary(body, initial, inputs, parameter, reverse=reverse)
    initial = jax.tree.map(jnp.asarray, initial)
    leaves = jax.tree.leaves(initial)
    if not leaves or any(x.shape != leaves[0].shape for x in leaves):
        raise ValueError("column scan requires nonempty carry with matching leaf shapes")
    representative = leaves[0]
    inputs = tuple(jnp.asarray(value) for value in inputs)
    if not inputs:
        raise ValueError("column scan requires nonempty inputs")
    if any(
        x.ndim == 0 or x.shape[0] != inputs[0].shape[0] or x.shape[1:] not in ((), representative.shape)
        for x in inputs
    ):
        raise ValueError("column scan input layers must match the initial carry")
    if parameter is not None:
        parameter = jax.tree.map(jnp.asarray, parameter)
        if any(value.ndim > 1 for value in jax.tree.leaves(parameter)):
            raise ValueError("column scan parameters must be scalar or one-dimensional tables")
    if not representative.size or inputs[0].shape[0] == 0:
        return _ordinary(body, initial, inputs, parameter, reverse=reverse)
    current, history = _result_shapes(body, initial, inputs, parameter)
    if jax.tree.structure(current) != jax.tree.structure(initial):
        raise ValueError("column scan carry structure must remain unchanged")
    for before, after in zip(leaves, jax.tree.leaves(current), strict=True):
        if after.shape != before.shape or after.dtype != before.dtype:
            raise ValueError("column scan carry shape and dtype must remain unchanged")
    if any(x.shape != representative.shape for x in jax.tree.leaves(history)):
        raise ValueError("column scan history leaves must match the column shape")
    supported = (jnp.bool_, jnp.int32, jnp.uint32, jnp.int64, jnp.uint64, jnp.float32, jnp.float64)
    if any(x.dtype not in supported for x in (*inputs, *leaves, *jax.tree.leaves(history))):
        return _ordinary(body, initial, inputs, parameter, reverse=reverse)
    return _dispatch(body, initial, inputs, parameter, reverse)


def scan_vmapped_columns(body, initial, inputs, parameter=None):
    """Pack independent outer-vmap columns into the shared column scan.

    The body has the same contract as scan_columns. Mapped parameters use
    ordinary scan, since a batch of lookup tables is not a shared lookup table.
    This wrapper is useful inside existing scalar-column physics functions.
    """
    from jax.custom_batching import custom_vmap

    @custom_vmap
    def compute(initial, inputs, parameters):
        return scan_columns(body, initial, inputs, parameters)

    @compute.def_vmap
    def batch(size, mapped, initial, inputs, parameters):
        initial_mapped, inputs_mapped, parameters_mapped = mapped
        if any(jax.tree.leaves(parameters_mapped)):
            axes = jax.tree.map(lambda flag: 0 if flag else None, mapped)
            result = jax.vmap(partial(_ordinary, body), in_axes=axes)(
                initial, inputs, parameters,
            )
        else:
            def broadcast(value, is_mapped):
                return value if is_mapped else jnp.broadcast_to(value, (size, *value.shape))

            initial = jax.tree.map(broadcast, initial, initial_mapped)
            inputs = tuple(
                jnp.swapaxes(broadcast(value, flag), 0, 1)
                for value, flag in zip(inputs, inputs_mapped, strict=True)
            )
            current, history = scan_columns(body, initial, inputs, parameters)
            result = current, jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), history)
        return result, jax.tree.map(lambda _: True, result)

    @jax.custom_jvp
    def differentiable(initial, inputs, parameters):
        return compute(initial, inputs, parameters)

    @differentiable.defjvp
    def derivative(primals, tangents):
        primal = differentiable(*primals)
        _, tangent = jax.jvp(partial(_ordinary, body), primals, tangents)
        return primal, tangent

    initial = jax.tree.map(jnp.asarray, initial)
    inputs = tuple(jnp.asarray(value) for value in inputs)
    parameter = jax.tree.map(jnp.asarray, parameter)
    return differentiable(initial, inputs, parameter)
