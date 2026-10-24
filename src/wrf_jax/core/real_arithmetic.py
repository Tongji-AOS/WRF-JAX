"""WRF arithmetic for XLA and persistent column kernels."""

from functools import lru_cache, partial

import jax
from jax import lax
from jax._src import core, dispatch
from jax._src.interpreters import ad, batching, mlir
import jax.numpy as jnp


def _assignment(value, token):
    bits = lax.bitcast_convert_type(value, jnp.uint32)
    return lax.bitcast_convert_type(lax.bitwise_xor(bits, token), jnp.float32)


_ASSIGNMENT = core.Primitive("wrf_real_assignment")
_DIVIDE = core.Primitive("wrf_real_divide")
_REDUCED_ASSIGNMENT = core.Primitive("wrf_reduced_real_assignment")
_COLUMN_BINARY = core.Primitive("wrf_column_binary")
_COLUMN_BARRIER = core.Primitive("wrf_column_barrier")
_COLUMN_RINT = core.Primitive("wrf_column_rint")
_BINARY_OPERATIONS = dict(add=lax.add, sub=lax.sub, mul=lax.mul, div=lax.div)
for primitive in (_ASSIGNMENT, _DIVIDE):
    primitive.def_impl(partial(dispatch.apply_primitive, primitive))
    batching.defbroadcasting(primitive)
_REDUCED_ASSIGNMENT.def_impl(partial(dispatch.apply_primitive, _REDUCED_ASSIGNMENT))
batching.defvectorized(_REDUCED_ASSIGNMENT)
for primitive in (_COLUMN_BINARY, _COLUMN_BARRIER):
    primitive.def_impl(partial(dispatch.apply_primitive, primitive))
batching.defbroadcasting(_COLUMN_BINARY)
batching.defvectorized(_COLUMN_BARRIER)
_COLUMN_RINT.def_impl(partial(dispatch.apply_primitive, _COLUMN_RINT))
batching.defvectorized(_COLUMN_RINT)


@_COLUMN_RINT.def_abstract_eval
def _column_rint_abstract(value):
    if value.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("column nearest-even rounding requires float32 or float64")
    return value


mlir.register_lowering(_COLUMN_RINT, mlir.lower_fun(jnp.rint, multiple_results=False))
ad.defjvp_zero(_COLUMN_RINT)


def column_rint(value):
    """Nearest-even integer rounding with a floating result on both devices."""
    return _COLUMN_RINT.bind(jnp.asarray(value))


@_COLUMN_BINARY.def_abstract_eval
def _column_binary_abstract(left, right, *, operation, separate=False):
    if (left.dtype != right.dtype
            or left.dtype not in (jnp.float32, jnp.float64)):
        raise TypeError("column arithmetic requires matching floating operands")
    return left.update(shape=jnp.broadcast_shapes(left.shape, right.shape))


def _column_binary_lowering(ctx, left, right, *, operation, separate=False):
    function = _BINARY_OPERATIONS[operation]
    if separate:
        from wrf_jax.core.optimization import semantic_call
        calculate = lambda a, b: semantic_call(function, a, b)
    else:
        calculate = function
    return mlir.lower_fun(calculate, multiple_results=False)(ctx, left, right)


_COLUMN_BARRIER.def_abstract_eval(lambda value: value)
mlir.register_lowering(_COLUMN_BINARY, _column_binary_lowering)
mlir.register_lowering(_COLUMN_BARRIER, mlir.lower_fun(lax.optimization_barrier, multiple_results=False))
ad.deflinear(_COLUMN_BARRIER, lambda value: [value])


def _column_binary_left_jvp(tangent, left, right, *, operation, separate=False):
    if operation in ("add", "sub"):
        # vmap can leave a shared operand with a singleton batch dimension.
        return jnp.broadcast_to(tangent, jnp.broadcast_shapes(left.shape, right.shape))
    return tangent * right if operation == "mul" else tangent / right


def _column_binary_right_jvp(tangent, left, right, *, operation, separate=False):
    if operation in ("add", "sub"):
        tangent = tangent if operation == "add" else -tangent
        return jnp.broadcast_to(tangent, jnp.broadcast_shapes(left.shape, right.shape))
    return tangent * left if operation == "mul" else -(left / right) * (tangent / right)


ad.defjvp(_COLUMN_BINARY, _column_binary_left_jvp, _column_binary_right_jvp)


def column_binary(operation, left, right, *, separate=False):
    """Use ordinary JAX operators and per-operation IEEE rounding in columns.

    Pallas otherwise contracts multiply/add and approximates division. The
    column lowering preserves the separate instructions of unfused XLA calls.
    Integer arithmetic is unchanged; normal JAX transformations remain valid.
    ``separate`` also pins this operation behind an XLA call boundary, for
    recurrences whose CPU result requires preventing multiply-add contraction.
    """
    if operation not in _BINARY_OPERATIONS:
        raise ValueError(f"unsupported column operation: {operation}")
    left, right = jnp.asarray(left), jnp.asarray(right)
    dtype = jnp.result_type(left, right)
    left, right = jnp.broadcast_arrays(left.astype(dtype), right.astype(dtype))
    if dtype not in (jnp.float32, jnp.float64):
        return _BINARY_OPERATIONS[operation](left, right)
    options = dict(operation=operation)
    if separate:
        options["separate"] = True
    return _COLUMN_BINARY.bind(left, right, **options)


def column_barrier(value):
    """Retain an XLA boundary; column operations already round individually."""
    return jax.tree.map(lambda x: _COLUMN_BARRIER.bind(x), value)


@_ASSIGNMENT.def_abstract_eval
def _assignment_abstract(value, token):
    if value.dtype != jnp.float32 or token.dtype != jnp.uint32:
        raise TypeError("WRF REAL assignment needs float32 values and uint32 tokens")
    return value.update(shape=jnp.broadcast_shapes(value.shape, token.shape))


@_DIVIDE.def_abstract_eval
def _divide_abstract(left, right):
    if left.dtype != jnp.float32 or right.dtype != jnp.float32:
        raise TypeError("WRF REAL division needs float32 operands")
    return left.update(shape=jnp.broadcast_shapes(left.shape, right.shape))


@_REDUCED_ASSIGNMENT.def_abstract_eval
def _reduced_assignment_abstract(value):
    if value.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("reduced REAL assignment needs floating operands")
    return value.update(dtype=jnp.dtype(jnp.float32))


def _reduced_assignment_value(value):
    if value.dtype == jnp.float32:
        return value
    rounded = value.astype(jnp.float32)
    signed_zero = lax.bitcast_convert_type(
        lax.bitcast_convert_type(rounded, jnp.uint32) & jnp.uint32(0x80000000),
        jnp.float32,
    )
    # ReducePrecision rounds the significand before flushing small exponents.
    # Its even tie below binary32 tiny is a quarter subnormal ULP below tiny,
    # not the half-ULP threshold of a plain float32 cast followed by flushing.
    threshold = jnp.float64(float.fromhex("0x1.ffffffp-127"))
    return jnp.where(jnp.abs(value) < threshold, signed_zero, rounded)


mlir.register_lowering(_REDUCED_ASSIGNMENT,
                       mlir.lower_fun(_reduced_assignment_value, multiple_results=False))


def _reduced_assignment_transpose(tangent, value):
    dtype = value.aval.dtype if isinstance(value, ad.UndefinedPrimal) else value.dtype
    return [lax.reduce_precision(tangent.astype(dtype), exponent_bits=8, mantissa_bits=23)]


ad.deflinear2(_REDUCED_ASSIGNMENT, _reduced_assignment_transpose)


# The dynamic integer XOR already preserves each binary32 assignment. An
# additional whole-array barrier needlessly prevents adjacent stores fusing.
mlir.register_lowering(_ASSIGNMENT, mlir.lower_fun(_assignment, multiple_results=False))
mlir.register_lowering(_DIVIDE, mlir.lower_fun(lax.div, multiple_results=False))
# A zero token is a REAL store, not a physical discontinuity. Differentiate
# its real-valued input as for a float cast; keep diagnostic bit edits opaque.
ad.defjvp(_ASSIGNMENT, lambda tangent, value, token: jnp.where(token == 0, tangent, 0), None)
ad.defjvp(_DIVIDE, lambda tangent, left, right: tangent / right,
          lambda tangent, left, right: -(left / right) * (tangent / right))


def real_assignment(value, rounding_token):
    """Keep binary32 assignment and its dynamic integer rounding token."""
    value, token = jnp.broadcast_arrays(jnp.asarray(value, jnp.float32),
                                       jnp.asarray(rounding_token, jnp.uint32))
    return _ASSIGNMENT.bind(value, token)


def real_divide(left, right):
    """Divide binary32 values without Triton's approximate reciprocal."""
    left, right = jnp.broadcast_arrays(jnp.asarray(left, jnp.float32),
                                      jnp.asarray(right, jnp.float32))
    return _DIVIDE.bind(left, right)


def real_reduced_assignment(value):
    """Preserve longwave's ReducePrecision-and-cast assignment on each device."""
    return _REDUCED_ASSIGNMENT.bind(jnp.asarray(value))


def real_add(left, right, token):
    return real_assignment(jnp.asarray(left, jnp.float64) + jnp.asarray(right, jnp.float64), token)


def real_subtract(left, right, token):
    return real_assignment(jnp.asarray(left, jnp.float64) - jnp.asarray(right, jnp.float64), token)


def real_multiply(left, right, token):
    return real_assignment(jnp.asarray(left, jnp.float64) * jnp.asarray(right, jnp.float64), token)


def real_divide_assigned(left, right, token):
    return real_assignment(real_divide(real_assignment(left, token), real_assignment(right, token)), token)


def _column_lower_fun(function):
    """Trace fixed arithmetic once per abstract signature, emitting fresh IR."""
    from jax._src import api_util, config, core, linear_util as lu
    from jax._src.interpreters import partial_eval as pe
    from jax._src.pallas.triton import lowering

    def outputs(*args, **kwargs):
        return (function(*args, **kwargs),)

    @lru_cache(maxsize=128)
    def traced(avals, parameters, trace_context):
        parameters = dict(parameters)
        wrapped = lu.wrap_init(
            outputs, parameters,
            debug_info=api_util.debug_info("wrf arithmetic lowering", function, avals, parameters),
        )
        jaxpr, _, constants = pe.trace_to_jaxpr_dynamic(wrapped, avals)
        return core.ClosedJaxpr(jaxpr, constants)

    def lower(ctx, *values, **parameters):
        jaxpr = traced(tuple(ctx.avals_in), tuple(sorted(parameters.items())), config.trace_context())
        return lowering._closed_call_lowering_rule(ctx, *values, call_jaxpr=jaxpr)[0]

    return lower


@lru_cache(maxsize=1)
def register_column_lowerings():
    """Register only our own primitives, without changing JAX's operations."""
    from jax._src.pallas.triton import lowering
    from jax.experimental.pallas import triton as plt

    def binary_instruction(left, right, *, operation, separate=False):
        left, right = jnp.broadcast_arrays(left, right)
        scalar = left.ndim == 0
        if scalar:
            left, right = left.reshape((1,)), right.reshape((1,))
        bits = left.dtype.itemsize * 8
        constraint = "f" if bits == 32 else "d"
        value, = plt.elementwise_inline_asm(
            f"{operation}.rn.f{bits} $0, $1, $2;", args=[left, right],
            constraints=f"={constraint},{constraint},{constraint}", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, left.dtype)])
        return value.reshape(()) if scalar else value

    def divide(left, right):
        left, right = jnp.broadcast_arrays(left, right)
        result, = plt.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;", args=[left, right],
            constraints="=f,f,f", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, left.dtype)],
        )
        return result

    def round_integer(value):
        scalar = value.ndim == 0
        if scalar:
            value = value.reshape((1,))
        bits = value.dtype.itemsize * 8
        constraint = "f" if bits == 32 else "d"
        rounded, = plt.elementwise_inline_asm(
            f"cvt.rni.f{bits}.f{bits} $0, $1;", args=[value],
            constraints=f"={constraint},{constraint}", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(value.shape, value.dtype)],
        )
        return rounded.reshape(()) if scalar else rounded

    assignment_rule = _column_lower_fun(_assignment)

    def assignment_lowering(ctx, value, token):
        # Pallas passes literal operands as host values. Its bitcast lowering
        # expects IR values, including for a scalar constant such as 1.0.
        values = [lowering._ensure_ir_value(x, aval)
                  for x, aval in zip((value, token), ctx.avals_in, strict=True)]
        return assignment_rule(ctx, *values)

    # The integer XOR retains the assignment in the column kernel. A separate
    # XLA optimization barrier has no meaning inside a Pallas-generated kernel.
    lowering.register_lowering(_ASSIGNMENT)(assignment_lowering)
    lowering.register_lowering(_DIVIDE)(_column_lower_fun(divide))
    lowering.register_lowering(_REDUCED_ASSIGNMENT)(
        _column_lower_fun(_reduced_assignment_value))
    lowering.register_lowering(_COLUMN_BINARY)(_column_lower_fun(binary_instruction))
    lowering.register_lowering(_COLUMN_BARRIER)(_column_lower_fun(lambda x: x))
    lowering.register_lowering(_COLUMN_RINT)(_column_lower_fun(round_integer))
