"""Floating-point math kernels matching the WRF reference build."""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.real_arithmetic import column_rint


def _float64_from_bits(bits: int) -> np.float64:
    return np.asarray(bits, dtype=np.uint64).view(np.float64)


@jax.custom_jvp
def sqrt_for_wrf(value: Any) -> Any:
    """Preserve WRF REAL square-root rounding on CPU and GPU."""
    array = jnp.asarray(value)
    if array.dtype == jnp.float32 and jax.config.x64_enabled:
        root = jnp.sqrt(array.astype(jnp.float64))
        # The bit-level use prevents LLVM from narrowing the operation to
        # CUDA's sqrt.approx.f32 before the final REAL conversion.
        return jnp.nextafter(root, root).astype(jnp.float32)
    return jnp.sqrt(array)


@sqrt_for_wrf.defjvp
def _sqrt_for_wrf_jvp(
    primals: tuple[Any], tangents: tuple[Any]
) -> tuple[Any, Any]:
    result = sqrt_for_wrf(primals[0])
    return result, (tangents[0] * jnp.asarray(0.5, dtype=result.dtype)) / result


@jax.custom_jvp
def clipped_sqrt_for_wrf(value: Any) -> Any:
    """Preserve the clipped WRF root with a finite closed-branch derivative."""
    value = jnp.asarray(value)
    return sqrt_for_wrf(jnp.maximum(value, jnp.asarray(0, value.dtype)))


@clipped_sqrt_for_wrf.defjvp
def _clipped_sqrt_for_wrf_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    result = clipped_sqrt_for_wrf(value)
    active = value > 0
    denominator = jnp.where(active, result, jnp.ones_like(result))
    derivative = tangent * jnp.asarray(0.5, result.dtype) / denominator
    return result, jnp.where(active, derivative, jnp.zeros_like(derivative))


# GNU Fortran vectorizes the WRF REAL pressure power in ``phy_prep`` to
# glibc's four-lane ``powf``.  Reproduce that positive-normal path with JAX
# primitives so the Python model keeps the compiler math semantics without a
# host callback or a compiled Fortran/C bridge.
_RECIPROCALS = np.arange(512, 1025, dtype=np.float64)
_POWF_LOG_LOOKUP_NP = np.where(
    _RECIPROCALS <= np.float64(724.0),
    np.log(np.float64(512.0) / _RECIPROCALS),
    np.log(np.float64(1024.0) / _RECIPROCALS),
)
_POWF_EXP_LOOKUP_NP = np.exp2(np.arange(2049, dtype=np.float64) / np.float64(2048.0))
_POWF_EXP_LOOKUP_NP *= _float64_from_bits(0x3FEFFFFFFC27DD9E)
_POWF_EXP_LOOKUP_NP[1242] = np.nextafter(_POWF_EXP_LOOKUP_NP[1242], np.float64(np.inf))

_POWF_LOG_LOOKUP = jnp.asarray(_POWF_LOG_LOOKUP_NP, dtype=jnp.float64)
_POWF_EXP_LOOKUP = jnp.asarray(_POWF_EXP_LOOKUP_NP, dtype=jnp.float64)
_POWF_LOG_CUBIC = jnp.float64(_float64_from_bits(0x3FD55555555543C5))
_POWF_LOG_QUADRATIC = jnp.float64(_float64_from_bits(0xBFDFFFFFFFFFF81F))
_POWF_LN2 = jnp.float64(_float64_from_bits(0x3FE62E42FEFA39EF))
_POWF_EXP_SCALE = jnp.float64(_float64_from_bits(0x40A71547652B82FE))
_POWF_EXP_STEP = jnp.float64(_float64_from_bits(0x3F362F3904051FA1))


# glibc 2.39 scalar/FMA ``logf`` table used by the WRF reference build.
# The implementation below follows the positive-normal path selected by the
# x86-64 IFUNC on the reference Zen2 CPU.  Keeping the hexadecimal constants
# in binary64 is necessary because glibc evaluates the approximation in
# double precision and rounds only its return value to float32.
_GLIBC_LOGF_TABLE_NP = np.asarray(
    [
        (float.fromhex("0x1.661ec79f8f3bep+0"), float.fromhex("-0x1.57bf7808caadep-2")),
        (float.fromhex("0x1.571ed4aaf883dp+0"), float.fromhex("-0x1.2bef0a7c06ddbp-2")),
        (float.fromhex("0x1.49539f0f010bp+0"), float.fromhex("-0x1.01eae7f513a67p-2")),
        (float.fromhex("0x1.3c995b0b80385p+0"), float.fromhex("-0x1.b31d8a68224e9p-3")),
        (float.fromhex("0x1.30d190c8864a5p+0"), float.fromhex("-0x1.6574f0ac07758p-3")),
        (float.fromhex("0x1.25e227b0b8eap+0"), float.fromhex("-0x1.1aa2bc79c81p-3")),
        (float.fromhex("0x1.1bb4a4a1a343fp+0"), float.fromhex("-0x1.a4e76ce8c0e5ep-4")),
        (float.fromhex("0x1.12358f08ae5bap+0"), float.fromhex("-0x1.1973c5a611cccp-4")),
        (float.fromhex("0x1.0953f419900a7p+0"), float.fromhex("-0x1.252f438e10c1ep-5")),
        (float.fromhex("0x1p+0"), float.fromhex("0x0p+0")),
        (float.fromhex("0x1.e608cfd9a47acp-1"), float.fromhex("0x1.aa5aa5df25984p-5")),
        (float.fromhex("0x1.ca4b31f026aap-1"), float.fromhex("0x1.c5e53aa362eb4p-4")),
        (float.fromhex("0x1.b2036576afce6p-1"), float.fromhex("0x1.526e57720db08p-3")),
        (float.fromhex("0x1.9c2d163a1aa2dp-1"), float.fromhex("0x1.bc2860d22477p-3")),
        (float.fromhex("0x1.886e6037841edp-1"), float.fromhex("0x1.1058bc8a07ee1p-2")),
        (float.fromhex("0x1.767dcf5534862p-1"), float.fromhex("0x1.4043057b6ee09p-2")),
    ],
    dtype=np.float64,
)
_GLIBC_LOGF_TABLE = jnp.asarray(_GLIBC_LOGF_TABLE_NP, dtype=jnp.float64)
_GLIBC_LOGF_LN2 = jnp.float64(float.fromhex("0x1.62e42fefa39efp-1"))
_GLIBC_LOGF_POLY = jnp.asarray(
    np.asarray(
        [
            float.fromhex("-0x1.00ea348b88334p-2"),
            float.fromhex("0x1.5575b0be00b6ap-2"),
            float.fromhex("-0x1.ffffef20a4123p-2"),
        ],
        dtype=np.float64,
    ),
    dtype=jnp.float64,
)


# glibc 2.39 scalar/FMA ``expf`` data.  RRTMG SW calls scalar ``exp`` with
# REAL arguments while forming ``co2mult``; GNU Fortran consequently resolves
# that operation to this binary32 libm entry point on the reference host.
# glibc evaluates the normal finite path in binary64 and rounds only the
# return value to binary32.
_GLIBC_EXPF_TABLE = jnp.asarray(
    np.asarray(
        [
            0x3FF0000000000000,
            0x3FEFD9B0D3158574,
            0x3FEFB5586CF9890F,
            0x3FEF9301D0125B51,
            0x3FEF72B83C7D517B,
            0x3FEF54873168B9AA,
            0x3FEF387A6E756238,
            0x3FEF1E9DF51FDEE1,
            0x3FEF06FE0A31B715,
            0x3FEEF1A7373AA9CB,
            0x3FEEDEA64C123422,
            0x3FEECE086061892D,
            0x3FEEBFDAD5362A27,
            0x3FEEB42B569D4F82,
            0x3FEEAB07DD485429,
            0x3FEEA47EB03A5585,
            0x3FEEA09E667F3BCD,
            0x3FEE9F75E8EC5F74,
            0x3FEEA11473EB0187,
            0x3FEEA589994CCE13,
            0x3FEEACE5422AA0DB,
            0x3FEEB737B0CDC5E5,
            0x3FEEC49182A3F090,
            0x3FEED503B23E255D,
            0x3FEEE89F995AD3AD,
            0x3FEEFF76F2FB5E47,
            0x3FEF199BDD85529C,
            0x3FEF3720DCEF9069,
            0x3FEF5818DCFBA487,
            0x3FEF7C97337B9B5F,
            0x3FEFA4AFA2A490DA,
            0x3FEFD0765B6E4540,
        ],
        dtype=np.uint64,
    ),
    dtype=jnp.uint64,
)
_GLIBC_EXPF_INV_LN2_SCALED = jnp.float64(float.fromhex("0x1.71547652b82fep+5"))
_GLIBC_EXPF_POLY = jnp.asarray(
    np.asarray(
        [
            float.fromhex("0x1.c6af84b912394p-20"),
            float.fromhex("0x1.ebfce50fac4f3p-13"),
            float.fromhex("0x1.62e42ff0c52d6p-6"),
        ],
        dtype=np.float64,
    ),
    dtype=jnp.float64,
)


def _glibc_logf_positive_normal_primal(value: Any) -> Any:
    x = jnp.asarray(value, dtype=jnp.float32)
    if not jax.config.x64_enabled:
        return jnp.log(x).astype(jnp.float32)

    bits = lax.bitcast_convert_type(x, jnp.uint32)
    reduced_bits = bits - jnp.uint32(0x3F330000)
    table_index = jnp.bitwise_and(
        jnp.right_shift(reduced_bits, jnp.uint32(19)),
        jnp.uint32(15),
    ).astype(jnp.int32)
    exponent = lax.shift_right_arithmetic(
        lax.bitcast_convert_type(reduced_bits, jnp.int32),
        jnp.int32(23),
    )
    significand_bits = bits - jnp.bitwise_and(reduced_bits, jnp.uint32(0xFF800000))
    significand = lax.bitcast_convert_type(significand_bits, jnp.float32).astype(jnp.float64)

    reciprocal = _GLIBC_LOGF_TABLE[table_index, 0]
    log_center = _GLIBC_LOGF_TABLE[table_index, 1]
    residual = significand * reciprocal - jnp.float64(1.0)
    base = log_center + exponent.astype(jnp.float64) * _GLIBC_LOGF_LN2
    residual_squared = residual * residual
    polynomial = _GLIBC_LOGF_POLY[1] * residual + _GLIBC_LOGF_POLY[2]
    polynomial = _GLIBC_LOGF_POLY[0] * residual_squared + polynomial
    result = polynomial * residual_squared + (base + residual)
    return result.astype(jnp.float32)


def glibc_expf_tables() -> tuple[Any, Any, Any]:
    """Read-only lookup inputs for ordinary and resident evaluation."""
    return _GLIBC_EXPF_TABLE, _GLIBC_EXPF_POLY, _GLIBC_EXPF_INV_LN2_SCALED


def _glibc_expf_finite_primal(value: Any, tables: Any = None) -> Any:
    x = jnp.asarray(value, dtype=jnp.float32)
    if not jax.config.x64_enabled:
        return jnp.exp(x).astype(jnp.float32)

    exp_table, exp_poly, inverse_ln2 = glibc_expf_tables() if tables is None else tables
    x64 = x.astype(jnp.float64)
    scaled = inverse_ln2 * x64
    # glibc's add/subtract-2**52 idiom implements nearest-even rounding here.
    # Expressing that idiom literally lets XLA cancel the subtraction across
    # the bitcast used for the table index.  ``rint`` has the same contract in
    # this finite range and keeps the rounded integer explicit in the graph.
    rounded_integer = column_rint(scaled).astype(jnp.int64)
    rounded = rounded_integer.astype(jnp.float64)
    residual = scaled - rounded

    rounded_bits = lax.bitcast_convert_type(rounded_integer, jnp.uint64)
    table_index = jnp.bitwise_and(rounded_bits, jnp.uint64(31)).astype(jnp.int32)
    scale_bits = exp_table[table_index] + lax.shift_left(
        rounded_bits,
        jnp.uint64(47),
    )
    scale = lax.bitcast_convert_type(scale_bits, jnp.float64)

    polynomial = exp_poly[0] * residual + exp_poly[1]
    residual_squared = residual * residual
    result = exp_poly[2] * residual + jnp.float64(1.0)
    result = polynomial * residual_squared + result
    return (result * scale).astype(jnp.float32)


@jax.custom_jvp
def wrf_glibc_expf_finite(value: Any, tables: Any = None) -> Any:
    """Match glibc 2.39 scalar/FMA ``expf`` for finite RRTMG inputs."""

    return _glibc_expf_finite_primal(value, tables)


@wrf_glibc_expf_finite.defjvp
def _wrf_glibc_expf_finite_jvp(
    primals: tuple[Any, Any],
    tangents: tuple[Any, Any],
) -> tuple[Any, Any]:
    value, tables = primals
    value_tangent, _ = tangents
    result = _glibc_expf_finite_primal(value, tables)
    return result, value_tangent * result


@jax.custom_jvp
def wrf_glibc_logf_positive_normal(value: Any) -> Any:
    """Match the reference host's glibc scalar ``logf`` for positive normals."""

    return _glibc_logf_positive_normal_primal(value)


@wrf_glibc_logf_positive_normal.defjvp
def _wrf_glibc_logf_positive_normal_jvp(
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Any, Any]:
    (value,) = primals
    (value_tangent,) = tangents
    result = _glibc_logf_positive_normal_primal(value)
    return result, value_tangent / jnp.asarray(value, dtype=jnp.float32)


def _glibc_log10f_positive_normal_primal(value: Any) -> Any:
    """Translate glibc 2.39 ``e_log10f`` for positive normal inputs."""

    x = jnp.asarray(value, dtype=jnp.float32)
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    exponent = (
        jnp.bitwise_and(
            jnp.right_shift(bits, jnp.uint32(23)),
            jnp.uint32(255),
        ).astype(jnp.int32)
        - jnp.int32(127)
    )
    negative_exponent = (exponent < jnp.int32(0)).astype(jnp.int32)
    reduced_bits = jnp.bitwise_or(
        jnp.bitwise_and(bits, jnp.uint32(0x007FFFFF)),
        (jnp.int32(127) - negative_exponent).astype(jnp.uint32)
        << jnp.uint32(23),
    )
    reduced = lax.bitcast_convert_type(reduced_bits, jnp.float32)
    exponent_value = (exponent + negative_exponent).astype(jnp.float32)

    inverse_ln10 = lax.bitcast_convert_type(
        jnp.uint32(0x3EDE5BD9),
        jnp.float32,
    )
    log10_2_high = lax.bitcast_convert_type(
        jnp.uint32(0x3E9A2080),
        jnp.float32,
    )
    log10_2_low = lax.bitcast_convert_type(
        jnp.uint32(0x355427DB),
        jnp.float32,
    )
    low_term = lax.optimization_barrier(exponent_value * log10_2_low)
    log_term = lax.optimization_barrier(
        inverse_ln10 * wrf_glibc_logf_positive_normal(reduced)
    )
    result = lax.optimization_barrier(low_term + log_term)
    high_term = lax.optimization_barrier(exponent_value * log10_2_high)
    return lax.optimization_barrier(result + high_term)


@jax.custom_jvp
def wrf_glibc_log10f_positive_normal(value: Any) -> Any:
    """Match glibc 2.39 scalar ``log10f`` for positive normal inputs."""

    return _glibc_log10f_positive_normal_primal(value)


@wrf_glibc_log10f_positive_normal.defjvp
def _wrf_glibc_log10f_positive_normal_jvp(
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Any, Any]:
    (value,) = primals
    (value_tangent,) = tangents
    result = _glibc_log10f_positive_normal_primal(value)
    derivative = (
        jnp.asarray(0.4342944819032518, dtype=jnp.float32)
        / jnp.asarray(value, dtype=jnp.float32)
    )
    return result, value_tangent * derivative


# Scalar powf uses a different approximation from the four-lane vector powf.
# Coefficients and range reduction: glibc 2.39 e_powf.c/e_powf_log2_data.c
# (GNU LGPL 2.1 or later). Reuse the identical reciprocal and exp2 tables above.
_SCALAR_POWF_LOG2_CENTERS = jnp.asarray(
    np.asarray([
        float.fromhex(value) for value in (
            "-0x1.efec65b963019p-2", "-0x1.b0b6832d4fca4p-2",
            "-0x1.7418b0a1fb77bp-2", "-0x1.39de91a6dcf7bp-2",
            "-0x1.01d9bf3f2b631p-2", "-0x1.97c1d1b3b7afp-3",
            "-0x1.2f9e393af3c9fp-3", "-0x1.960cbbf788d5cp-4",
            "-0x1.a6f9db6475fcep-5", "0x0p+0",
            "0x1.338ca9f24f53dp-4", "0x1.476a9543891bap-3",
            "0x1.e840b4ac4e4d2p-3", "0x1.40645f0c6651cp-2",
            "0x1.88e9c2c1b9ff8p-2", "0x1.ce0a44eb17bccp-2",
        )
    ], dtype=np.float64), dtype=jnp.float64,
)
_SCALAR_POWF_POLY = tuple(np.float64(float.fromhex(value)) for value in (
    "0x1.27616c9496e0bp-2", "-0x1.71969a075c67ap-2",
    "0x1.ec70a6ca7baddp-2", "-0x1.7154748bef6c8p-1", "0x1.71547652ab82bp0",
))
_SCALAR_EXP2F_POLY = tuple(np.float64(float.fromhex(value)) for value in (
    "0x1.c6af84b912394p-5", "0x1.ebfce50fac4f3p-3", "0x1.62e42ff0c52d6p-1",
))


def scalar_powf_tables() -> tuple[Any, Any, Any]:
    """Read-only reciprocal, logarithm-center and exponential lookup inputs."""
    return _GLIBC_LOGF_TABLE[:, 0], _SCALAR_POWF_LOG2_CENTERS, _GLIBC_EXPF_TABLE


def _scalar_powf_primal(base: Any, exponent: Any, tables: Any = None) -> Any:
    base32 = jnp.asarray(base, dtype=jnp.float32)
    exponent32 = jnp.asarray(exponent, dtype=jnp.float32)
    if not jax.config.x64_enabled:
        return jnp.power(base32, exponent32)

    reciprocal, log_centers, exp_table = scalar_powf_tables() if tables is None else tables

    bits = lax.bitcast_convert_type(base32, jnp.uint32)
    reduced = bits - jnp.uint32(0x3F330000)
    interval = ((reduced >> jnp.uint32(19)) & jnp.uint32(15)).astype(jnp.int32)
    top = reduced & jnp.uint32(0xFF800000)
    exponent2 = lax.shift_right_arithmetic(
        lax.bitcast_convert_type(top, jnp.int32), jnp.int32(23)
    )
    mantissa = lax.bitcast_convert_type(bits - top, jnp.float32).astype(jnp.float64)
    residual = mantissa * reciprocal[interval] - jnp.float64(1.0)
    square = residual * residual
    fourth = square * square
    a0, a1, a2, a3, a4 = _SCALAR_POWF_POLY
    center = log_centers[interval] + exponent2.astype(jnp.float64)
    logarithm2 = (a0 * residual + a1) * fourth + (
        (a2 * residual + a3) * square + (a4 * residual + center)
    )
    power2 = exponent32.astype(jnp.float64) * logarithm2
    # Explicit nearest-even rounding prevents XLA from cancelling glibc's
    # add/subtract-large-constant rounding idiom (as for expf above).
    rounded = column_rint(power2 * jnp.float64(32.0)).astype(jnp.int64)
    coordinate = lax.bitcast_convert_type(rounded, jnp.uint64)
    remainder = power2 - rounded.astype(jnp.float64) * jnp.float64(1.0 / 32.0)
    scale_bits = exp_table[(coordinate & jnp.uint64(31)).astype(jnp.int32)]
    scale = lax.bitcast_convert_type(scale_bits + (coordinate << jnp.uint64(47)), jnp.float64)
    c0, c1, c2 = _SCALAR_EXP2F_POLY
    polynomial = (c0 * remainder + c1) * (remainder * remainder)
    polynomial = polynomial + (c2 * remainder + jnp.float64(1.0))
    result = (polynomial * scale).astype(jnp.float32)
    result = jnp.where(power2 <= jnp.float64(-150.0), jnp.float32(0.0), result)
    overflow_threshold = jnp.float64(float.fromhex("0x1.fffffffd1d571p+6"))
    result = jnp.where(power2 > overflow_threshold, jnp.float32(jnp.inf), result)
    normal = (
        (bits >= jnp.uint32(0x00800000))
        & (bits < jnp.uint32(0x7F800000))
        & jnp.isfinite(exponent32)
    )
    def with_fallback():
        fallback = jnp.power(
            base32.astype(jnp.float64), exponent32.astype(jnp.float64)
        ).astype(jnp.float32)
        return jnp.where(normal, result, fallback)

    # The fallback is only needed when this batch contains special values.
    return lax.cond(jnp.max((~normal).astype(jnp.int32), initial=0) == 0,
                    lambda: result, with_fallback)


@jax.custom_jvp
def scalar_powf_for_wrf(base: Any, exponent: Any, tables: Any = None) -> Any:
    """Reproduce WRF's scalar positive-normal powf on CPU and GPU."""
    return _scalar_powf_primal(base, exponent, tables)


@scalar_powf_for_wrf.defjvp
def _scalar_powf_for_wrf_jvp(
    primals: tuple[Any, Any, Any], tangents: tuple[Any, Any, Any]
) -> tuple[Any, Any]:
    base, exponent, tables = primals
    base_dot, exponent_dot, _ = tangents
    result = _scalar_powf_primal(base, exponent, tables)
    base32 = jnp.asarray(base, dtype=jnp.float32)
    exponent32 = jnp.asarray(exponent, dtype=jnp.float32)
    return result, result * (
        jnp.asarray(base_dot, dtype=jnp.float32) * exponent32 / base32
        + jnp.asarray(exponent_dot, dtype=jnp.float32) * jnp.log(base32)
    )


@jax.jit
def _vector_powf_positive_normal_primal(base: Any, exponent: Any) -> Any:
    base32 = jnp.asarray(base, dtype=jnp.float32)
    exponent32 = jnp.asarray(exponent, dtype=jnp.float32)

    if not jax.config.x64_enabled:
        return jnp.power(base32, exponent32).astype(jnp.float32)

    base_bits = lax.bitcast_convert_type(base32, jnp.uint32)
    base_exponent = (
        jnp.bitwise_and(jnp.right_shift(base_bits, jnp.uint32(23)), jnp.uint32(255)).astype(jnp.int32)
        - jnp.int32(127)
    )
    significand = jnp.ldexp(base32.astype(jnp.float64), -base_exponent)
    reciprocal = jnp.rint(jnp.float64(1024.0) / significand).astype(jnp.int32)
    residual = significand * reciprocal.astype(jnp.float64) / jnp.float64(1024.0) - jnp.float64(1.0)

    adjusted_exponent = base_exponent + (reciprocal <= jnp.int32(724)).astype(jnp.int32)
    residual_squared = residual * residual
    log_polynomial = residual + residual_squared * (
        _POWF_LOG_QUADRATIC + _POWF_LOG_CUBIC * residual
    )
    log_base = (
        adjusted_exponent.astype(jnp.float64) * _POWF_LN2
        + jnp.take(_POWF_LOG_LOOKUP, reciprocal - jnp.int32(512))
        + log_polynomial
    )

    scaled_power = exponent32.astype(jnp.float64) * log_base * _POWF_EXP_SCALE
    table_coordinate = jnp.rint(scaled_power - jnp.float64(0.5)).astype(jnp.int32)
    exp_residual = (scaled_power - table_coordinate.astype(jnp.float64)) * _POWF_EXP_STEP
    table_value = jnp.take(
        _POWF_EXP_LOOKUP,
        jnp.bitwise_and(table_coordinate, jnp.int32(2047)),
    )
    result64 = jnp.ldexp(
        table_value * (jnp.float64(1.0) + exp_residual),
        jnp.right_shift(table_coordinate, jnp.int32(11)),
    )
    vector_result = result64.astype(jnp.float32)

    is_vector_domain = jnp.logical_and(
        base_bits >= jnp.uint32(0x00800000),
        base_bits <= jnp.uint32(0x4059FE36),
    )
    # Retain WRF's scalar fallback without selecting the device's native libm.
    def with_fallback():
        scalar_result = _scalar_powf_primal(base32, exponent32)
        return jnp.where(is_vector_domain, vector_result, scalar_result)

    return lax.cond(jnp.max((~is_vector_domain).astype(jnp.int32), initial=0) == 0,
                    lambda: vector_result, with_fallback)


@jax.custom_jvp
def vector_powf_for_wrf(base: Any, exponent: Any) -> Any:
    """Match WRF's vectorized REAL ``base**exponent`` on positive normals."""

    return _vector_powf_positive_normal_primal(base, exponent)


@vector_powf_for_wrf.defjvp
def _vector_powf_for_wrf_jvp(
    primals: tuple[Any, Any],
    tangents: tuple[Any, Any],
) -> tuple[Any, Any]:
    base, exponent = primals
    base_dot, exponent_dot = tangents
    result = _vector_powf_positive_normal_primal(base, exponent)
    base32 = jnp.asarray(base, dtype=jnp.float32)
    exponent32 = jnp.asarray(exponent, dtype=jnp.float32)
    tangent = result * (
        jnp.asarray(base_dot, dtype=jnp.float32) * exponent32 / base32
        + jnp.asarray(exponent_dot, dtype=jnp.float32) * jnp.log(base32)
    )
    return result, tangent


def _store_real(value: Any, rounding_token: Any) -> Any:
    """Materialize one WRF ``REAL`` instruction result in float32."""

    bits = jax.lax.bitcast_convert_type(jnp.asarray(value, dtype=jnp.float32), jnp.uint32)
    bits = jax.lax.bitwise_xor(bits, jnp.asarray(rounding_token, dtype=jnp.uint32))
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


@jax.custom_jvp
def wrf_sse4_logf_positive_normal(value: Any, rounding_token: Any) -> Any:
    """Match the WRF/GCC SSE4 vector ``logf`` for positive normal inputs.

    WRF 4.5 built with GCC ``-ftree-vectorize`` lowers the inner
    ``calc_p_rho_phi`` logarithm to glibc ``_ZGVbN4v_logf``. That routine is
    a degree-9 single-precision polynomial and can differ from scalar
    ``logf``/XLA by up to several ulps. Pressure ratios in the caller are
    finite, positive, normal values, so the libmvec special-value fallback is
    intentionally outside this compact hot-path translation.
    """

    x = jnp.asarray(value, dtype=jnp.float32)
    token = jnp.asarray(rounding_token, dtype=jnp.uint32)

    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    reduced_bits = bits - jnp.uint32(0x3F2AAAAB)
    signed_reduced = jax.lax.bitcast_convert_type(reduced_bits, jnp.int32)
    exponent = jax.lax.shift_right_arithmetic(signed_reduced, jnp.int32(23))
    mantissa_bits = (reduced_bits & jnp.uint32(0x007FFFFF)) + jnp.uint32(0x3F2AAAAB)

    reduced = jax.lax.bitcast_convert_type(mantissa_bits, jnp.float32)
    reduced = _store_real(reduced - jnp.float32(1.0), token)
    ln2 = jax.lax.bitcast_convert_type(jnp.uint32(0x3F317218), jnp.float32)
    exponent_ln2 = _store_real(exponent.astype(jnp.float32) * ln2, token)

    polynomial = jax.lax.bitcast_convert_type(jnp.uint32(0xBE1B6A22), jnp.float32)
    for coefficient_bits in (
        0x3E2DB86B,
        0xBE289358,
        0x3E4AFB81,
        0xBE80061D,
        0x3EAAAEE7,
        0xBF000000,
    ):
        polynomial = _store_real(polynomial * reduced, token)
        coefficient = jax.lax.bitcast_convert_type(jnp.uint32(coefficient_bits), jnp.float32)
        polynomial = _store_real(polynomial + coefficient, token)
    polynomial = _store_real(polynomial * reduced, token)
    polynomial = _store_real(polynomial * reduced, token)
    result = _store_real(reduced + polynomial, token)
    return _store_real(result + exponent_ln2, token)


@wrf_sse4_logf_positive_normal.defjvp
def _wrf_sse4_logf_positive_normal_jvp(
    primals: tuple[Any, Any],
    tangents: tuple[Any, Any],
) -> tuple[Any, Any]:
    value, rounding_token = primals
    value_tangent, _ = tangents
    result = wrf_sse4_logf_positive_normal(value, rounding_token)
    return result, value_tangent / jnp.asarray(value, dtype=jnp.float32)
