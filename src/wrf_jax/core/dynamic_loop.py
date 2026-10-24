"""Differentiate runtime-count recurrences without changing their forward loop."""

from functools import partial

import jax
from jax import lax
from jax._src import core, dispatch
from jax._src.interpreters import ad, batching, mlir, partial_eval
import jax.numpy as jnp
import numpy as np


_TANGENT = core.Primitive("wrf_dynamic_loop_tangent")
_TANGENT.multiple_results = True
_TANGENT.def_impl(partial(dispatch.apply_primitive, _TANGENT))


def _zeros(value):
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.zeros_like(value)
    return np.zeros(value.shape, dtype=jax.dtypes.float0)


def _body(jaxpr, index, carry, parameters):
    return tuple(core.jaxpr_as_fun(jaxpr)(index, *carry, *parameters))


def _primal(lower, upper, values, *, jaxpr, carry_count):
    carry, parameters = values[:carry_count], values[carry_count:]
    return lax.fori_loop(lower, upper, lambda i, y: _body(jaxpr, i, y, parameters), carry)


def _tangent(*arguments, jaxpr, carry_count, input_indices, output_indices, batches=0):
    if batches:
        return jax.vmap(partial(
            _tangent, jaxpr=jaxpr, carry_count=carry_count, input_indices=input_indices,
            output_indices=output_indices, batches=batches - 1,
        ))(*arguments)
    lower, upper, *arguments = arguments
    count = len(jaxpr.in_avals) - 1
    values, directions = tuple(arguments[:count]), arguments[count:]
    ds = list(map(_zeros, values))
    for index, value in zip(input_indices, directions, strict=True):
        ds[index] = value
    parameters, dp = values[carry_count:], tuple(ds[carry_count:])
    carry = values[:carry_count]
    carry_directions = tuple(ds[index] for index in output_indices)

    def advance(index, pair):
        carry, active_ds = pair
        dc = list(map(_zeros, carry))
        for position, value in zip(output_indices, active_ds, strict=True):
            dc[position] = value
        value, derivative = jax.jvp(
            lambda y, p: _body(jaxpr, index, y, p),
            (carry, parameters), (tuple(dc), dp),
        )
        return value, tuple(derivative[position] for position in output_indices)

    return lax.fori_loop(lower, upper, advance, (carry, carry_directions))[1]


def _pullback(*arguments, jaxpr, carry_count, input_indices, output_indices, batches=0):
    if batches:
        return jax.vmap(partial(
            _pullback, jaxpr=jaxpr, carry_count=carry_count, input_indices=input_indices,
            output_indices=output_indices, batches=batches - 1,
        ))(*arguments)
    lower, upper, *arguments = arguments
    count = len(jaxpr.in_avals) - 1
    values, cotangents = tuple(arguments[:count]), tuple(arguments[count:])
    parameters = values[carry_count:]
    parameter_indices = tuple(i - carry_count for i in input_indices if i >= carry_count)
    parameter_gradient = tuple(jnp.zeros_like(parameters[i]) for i in parameter_indices)

    def reverse(iteration, pair):
        ct, accumulated = pair
        index = upper - 1 - iteration
        # Recompute only this short recurrence's prefix; no fixed step cap or
        # large per-column history is introduced into ordinary forecasts.
        carry = _primal(lower, index, values, jaxpr=jaxpr, carry_count=carry_count)
        seed = list(map(_zeros, carry))
        for position, value in zip(output_indices, ct, strict=True):
            seed[position] = value
        _, transpose = jax.vjp(lambda y, p: _body(jaxpr, index, y, p), carry, parameters)
        dc, dp = transpose(tuple(seed))
        return tuple(dc[i] for i in output_indices), tuple(
            previous + dp[i] for previous, i in zip(accumulated, parameter_indices, strict=True)
        )

    dc, dp = lax.fori_loop(0, jnp.maximum(upper - lower, 0), reverse,
                          (cotangents, parameter_gradient))
    gradient = dict(zip(output_indices, dc, strict=True))
    gradient.update((i + carry_count, value) for i, value in zip(parameter_indices, dp, strict=True))
    return tuple(gradient[i] for i in input_indices)


@_TANGENT.def_abstract_eval
def _abstract(lower, upper, *values, output_indices, **options):
    return tuple(values[index] for index in output_indices)


def _lowering(ctx, *values, **options):
    return mlir.lower_fun(partial(_tangent, **options), multiple_results=True)(ctx, *values)


mlir.register_lowering(_TANGENT, _lowering)


def _transpose(cotangents, *arguments, **options):
    count = len(options["jaxpr"].in_avals) + 1
    residuals = arguments[:count]
    if any(ad.is_undefined_primal(value) for value in residuals):
        raise TypeError("dynamic-loop transpose requires its primal recurrence inputs")
    cotangents = tuple(ad.instantiate_zeros(value) for value in cotangents)
    gradients = _pullback(*residuals, *cotangents, **options)
    return (None,) * count + gradients


ad.primitive_transposes[_TANGENT] = _transpose


def _batch(arguments, dimensions, **options):
    size = next(value.shape[dim] for value, dim in zip(arguments, dimensions, strict=True)
                if dim is not batching.not_mapped)
    values = tuple(batching.bdim_at_front(value, dim, size)
                   for value, dim in zip(arguments, dimensions, strict=True))
    result = _TANGENT.bind(*values, **dict(options, batches=options.get("batches", 0) + 1))
    return result, (0,) * len(result)


batching.primitive_batchers[_TANGENT] = _batch


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _loop(body, lower, upper, initial, parameters):
    return lax.fori_loop(lower, upper, lambda i, y: body(i, y, *parameters), initial)


@_loop.defjvp
def _jvp(body, primals, tangents):
    lower, upper, initial, parameters = primals
    result = _loop(body, *primals)
    values, structure = jax.tree.flatten((initial, parameters))
    directions = jax.tree.leaves(tangents[2:])
    carry, output_tree = jax.tree.flatten(initial)
    input_indices = tuple(i for i, value in enumerate(values)
                          if jnp.issubdtype(value.dtype, jnp.inexact))
    output_indices = tuple(i for i, value in enumerate(carry)
                           if jnp.issubdtype(value.dtype, jnp.inexact))

    def flat_body(index, *arrays):
        y, p = jax.tree.unflatten(structure, arrays)
        return body(index, y, *p)

    jaxpr = jax.make_jaxpr(flat_body)(lower, *values)
    active = _TANGENT.bind(lower, upper, *values, *(directions[i] for i in input_indices),
                          jaxpr=jaxpr, carry_count=len(carry), input_indices=input_indices,
                          output_indices=output_indices)
    derivative = list(map(_zeros, carry))
    for index, value in zip(output_indices, active, strict=True):
        derivative[index] = value
    return result, jax.tree.unflatten(output_tree, derivative)


def dynamic_fori_loop(lower, upper, body, initial):
    """Like lax.fori_loop, with first-order JVP/VJP for runtime step counts.

    The integer count is held fixed when differentiating the executed branch.
    Reverse mode replays the same body rather than truncating its iteration
    count. Captured array parameters receive gradients as well as the carry.
    """
    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    if lower.dtype != upper.dtype:
        # Match fori_loop's weak signed-integer promotion before tracing the
        # body, so its index aval matches the index used by the runtime loop.
        signed = all(jnp.issubdtype(v.dtype, jnp.signedinteger) for v in (lower, upper))
        lower_weak, upper_weak = core.get_aval(lower).weak_type, core.get_aval(upper).weak_type
        if signed and lower_weak and not upper_weak:
            lower = lax.convert_element_type(lower, upper.dtype)
        elif signed and upper_weak and not lower_weak:
            upper = lax.convert_element_type(upper, lower.dtype)
        else:
            raise TypeError("lower and upper arguments to fori_loop must have equal "
                            f"types, got {lower.dtype.name} and {upper.dtype.name}")
    initial = jax.tree.map(jnp.asarray, initial)
    # closure_convert hoists differentiable captures only. Integer/boolean
    # branch markers must also be explicit when this loop is inside vmap.
    structure = jax.tree.structure(initial)
    closed = jax.make_jaxpr(body)(lower, initial)
    program = partial_eval.convert_constvars_jaxpr(closed.jaxpr)

    def converted(index, carry, *parameters):
        values = core.eval_jaxpr(program, (), *parameters, index, *jax.tree.leaves(carry))
        return jax.tree.unflatten(structure, values)

    return _loop(converted, lower, upper, initial, tuple(closed.consts))
