"""Pointwise microphysics coupling with explicit arithmetic order."""

from functools import lru_cache, partial

import jax.numpy as jnp

from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.real_arithmetic import column_binary


def _precipitation(inputs, parameters):
    rainc, rainsh, cuppt, pratec, pratesh = inputs
    dt, scale = parameters
    add = partial(column_binary, "add")
    mul = partial(column_binary, "mul")
    div = partial(column_binary, "div")
    return (
        add(rainc, mul(pratec, dt)),
        add(rainsh, mul(pratesh, dt)),
        add(cuppt, div(mul(add(pratec, pratesh), dt), scale)),
    )


def advance_precipitation(inputs, dt, *, preserve_eager_rounding=False):
    parameters = (dt, jnp.asarray(1000.0, inputs[0].dtype))
    if preserve_eager_rounding:
        return map_elements(_precipitation, inputs, parameters)
    return _precipitation(inputs, parameters)


def _finish(inputs, parameters, *, use_theta_m):
    t_new, th_phy, h_before, qv, qc, qv_before, qc_before = inputs
    dt, limit, ratio, t0 = parameters
    add = partial(column_binary, "add")
    sub = partial(column_binary, "sub")
    mul = partial(column_binary, "mul")
    div = partial(column_binary, "div")
    mpten = jnp.clip(sub(th_phy, h_before), -limit, limit)
    qvten = sub(qv, qv_before)
    qcten = sub(qc, qc_before)
    if use_theta_m:
        moist_factor = add(1.0, mul(ratio, qv))
        old_factor = add(1.0, mul(ratio, qv_before))
        latent = mul(mul(ratio, qvten), th_phy)
        heating = mul(mpten, moist_factor)
        t_out = sub(add(add(mul(h_before, old_factor), heating), latent), t0)
        th_phy_m_t0 = sub(div(add(t_out, t0), moist_factor), t0)
        h_diabatic = div(add(heating, latent), dt)
    else:
        t_out = add(t_new, mpten)
        th_phy_m_t0 = t_out
        h_diabatic = div(mpten, dt)
    return t_out, th_phy_m_t0, h_diabatic, div(qvten, dt), div(qcten, dt)


@lru_cache(maxsize=2)
def _finish_body(use_theta_m):
    return partial(_finish, use_theta_m=use_theta_m)


def finish_microphysics(
    inputs, *, dt, limit, ratio, t0, use_theta_m, preserve_eager_rounding=True
):
    """Keep the calling stage's arithmetic while sharing one scientific body.

    An independently submitted stage rounds each array operation. Ordinary
    compiled timesteps already compose these expressions; retain their native
    compiler behavior instead of imposing new boundaries during optimization.
    """
    dtype = inputs[0].dtype
    parameters = tuple(jnp.asarray(value, dtype) for value in (dt, limit, ratio, t0))
    body = _finish_body(use_theta_m)
    if preserve_eager_rounding:
        return map_elements(body, inputs, parameters)
    return body(inputs, parameters)
