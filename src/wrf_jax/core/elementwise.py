"""Keep independent scientific expressions in one CPU/GPU-compatible body."""

from functools import partial
from math import prod

import jax
from jax import lax
import jax.numpy as jnp


def _element_kernel(body, inputs, parameter, *, interpret=False):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    from wrf_jax.core.real_arithmetic import register_column_lowerings

    register_column_lowerings()
    shape = jnp.broadcast_shapes(*(value.shape for value in inputs))
    size, block = prod(shape), 128
    result = jax.eval_shape(body, inputs, parameter)
    output_shapes = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct((size,), value.dtype), result
    )

    def kernel(input_refs, parameter_refs, output_refs):
        indices = pl.program_id(0) * block + jnp.arange(block)
        mask = indices < size
        values = []
        for value, ref in zip(inputs, input_refs, strict=True):
            dimensions = (1,) * (len(shape) - value.ndim) + value.shape
            offsets = jnp.zeros_like(indices)
            for axis, length in enumerate(dimensions):
                if length != 1:
                    coordinate = (indices // prod(shape[axis + 1 :])) % shape[axis]
                    offsets += coordinate * prod(dimensions[axis + 1 :])
            values.append(plt.load(ref.at[offsets], mask=mask, other=0))
        params = jax.tree.map(
            lambda ref: ref[()] if ref.ndim == 0 else ref, parameter_refs
        )
        outputs = body(tuple(values), params)
        jax.tree.map(
            lambda ref, value: plt.store(ref.at[indices], value, mask=mask),
            output_refs,
            outputs,
        )

    (result,) = pl.pallas_call(
        kernel,
        out_shape=(output_shapes,),
        grid=((size + block - 1) // block,),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4),
    )(tuple(value.reshape(-1) for value in inputs), parameter)
    return jax.tree.map(lambda value: value.reshape(shape), result)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _dispatch(body, inputs, parameter):
    return lax.platform_dependent(
        inputs,
        parameter,
        cuda=partial(_element_kernel, body),
        default=body,
    )


@_dispatch.defjvp
def _dispatch_jvp(body, primals, tangents):
    value = _dispatch(body, *primals)
    _, tangent = jax.jvp(body, primals, tangents)
    return value, tangent


def map_elements(body, inputs, parameter=None):
    """Evaluate column-local expressions without materializing intermediates.

    Inputs use standard broadcasting. Every output leaf must have the complete
    broadcast shape. Parameters are read-only scalars or 1D arrays, including
    lookup tables and flattened source fields for explicitly indexed gathers.
    The body cannot reduce across elements or capture dynamic
    arrays. CPU, debug execution and differentiation use that same JAX body.
    CUDA only changes memory access and scheduling, not the scientific formula.
    """
    inputs = tuple(jnp.asarray(value) for value in inputs)
    if not inputs:
        raise ValueError("elementwise mapping requires inputs")
    shape = jnp.broadcast_shapes(*(value.shape for value in inputs))
    parameter = jax.tree.map(jnp.asarray, parameter)
    if any(value.ndim > 1 for value in jax.tree.leaves(parameter)):
        raise ValueError("elementwise parameters must be scalar or 1D tables")
    result = jax.eval_shape(body, inputs, parameter)
    if not jax.tree.leaves(result) or any(
        value.shape != shape for value in jax.tree.leaves(result)
    ):
        raise ValueError("elementwise outputs must have the broadcast input shape")
    if jax.config.jax_disable_jit or not prod(shape):
        return body(inputs, parameter)
    return _dispatch(body, inputs, parameter)
