"""Shared pointwise water-physics preparation with explicit arithmetic order."""

from functools import partial

import jax.numpy as jnp
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.real_arithmetic import column_barrier, column_binary


def binary(operation, left, right):
    return column_barrier(column_binary(operation, left, right))


add = partial(binary, "add")

sub = partial(binary, "sub")

mul = partial(binary, "mul")

div = partial(binary, "div")


def height(inputs, parameters):
    ph, phb = inputs
    return div(add(ph, phb), parameters)


def mass_geometry(inputs, parameters):
    low, high = inputs
    return mul(0.5, add(low, high)), sub(high, low)


def thermodynamics(inputs, parameters, *, use_theta_m):
    t, qv, al, alb, p, pb = inputs
    t0, rv, rd, p0, exponent = parameters
    theta = add(t, t0)
    if use_theta_m:
        theta = div(theta, add(1.0, mul(div(rv, rd), qv)))
    rho = div(1.0, add(al, alb))
    pressure = add(p, pb)
    exner = div(pressure, p0) ** exponent
    return theta, rho, pressure, exner


def inner_pressure(inputs, parameters):
    fzm, fzp, high, low = inputs
    return add(mul(fzm, high), mul(fzp, low))


def edge_pressure(inputs, parameters, *, logarithmic):
    edge, first, second, p_first, p_second = inputs
    weight = div(sub(edge, second), sub(first, second))
    if logarithmic:
        p_first = jnp.log(jnp.maximum(p_first, 1.0))
        p_second = jnp.log(jnp.maximum(p_second, 1.0))
    result = add(mul(weight, p_first), mul(sub(1.0, weight), p_second))
    return jnp.exp(result) if logarithmic else result


def prepare_microphysics(
    t,
    qv,
    qc,
    ph,
    phb,
    al,
    alb,
    p,
    pb,
    fzm,
    fzp,
    *,
    gravity,
    t0,
    rv,
    rd,
    p0,
    exponent,
    use_theta_m,
    preserve_eager_rounding,
):
    def evaluate(body, inputs, parameters=None):
        if preserve_eager_rounding:
            return map_elements(body, inputs, parameters)
        return body(inputs, parameters)

    z_at_w = evaluate(height, (ph, phb), jnp.asarray(gravity, ph.dtype))
    z, dz = evaluate(mass_geometry, (z_at_w[:, :-1, :], z_at_w[:, 1:, :]))
    dz8w = jnp.zeros_like(z_at_w).at[:, :-1, :].set(dz)
    parameters = (
        *(jnp.asarray(value, t.dtype) for value in (t0, rv, rd)),
        jnp.asarray(p0, jnp.result_type(p, pb)),
        jnp.asarray(exponent, jnp.result_type(p, pb)),
    )
    theta, rho, pressure, exner = evaluate(
        partial(thermodynamics, use_theta_m=use_theta_m),
        (t, qv, al, alb, p, pb),
        parameters,
    )
    nk = int(t.shape[1])
    p8w = jnp.zeros_like(z_at_w)
    if nk > 1:
        inner = evaluate(
            inner_pressure,
            (
                fzm[1:nk][None, :, None],
                fzp[1:nk][None, :, None],
                pressure[:, 1:nk, :],
                pressure[:, : nk - 1, :],
            ),
        )
        bottom = evaluate(
            partial(edge_pressure, logarithmic=False),
            (
                z_at_w[:, 0, :],
                z[:, 0, :],
                z[:, 1, :],
                pressure[:, 0, :],
                pressure[:, 1, :],
            ),
        )
        top = evaluate(
            partial(edge_pressure, logarithmic=True),
            (
                z_at_w[:, nk, :],
                z[:, nk - 1, :],
                z[:, nk - 2, :],
                pressure[:, nk - 1, :],
                pressure[:, nk - 2, :],
            ),
        )
        p8w = (
            p8w.at[:, 1:nk, :].set(inner).at[:, 0, :].set(bottom).at[:, nk, :].set(top)
        )
    return dict(
        rho=rho,
        th_phy=theta,
        h_diabatic=theta,
        qv_diabatic=qv,
        qc_diabatic=qc,
        pi_phy=exner,
        p_phy=pressure,
        p8w=p8w,
        z_at_w=z_at_w,
        z=z,
        dz8w=dz8w,
    )
