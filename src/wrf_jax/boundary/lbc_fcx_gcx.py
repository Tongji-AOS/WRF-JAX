"""JAX-friendly translation of ``module_bc_em::lbc_fcx_gcx``."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def lbc_fcx_gcx(
    fcx: Any,
    gcx: Any,
    spec_bdy_width: int,
    spec_zone: int,
    relax_zone: int,
    dt: float,
    spec_exp: float,
    specified: bool,
    nested: bool,
) -> tuple[Any, Any]:
    """Return updated lateral-boundary relaxation coefficients.

    WRF loops over Fortran indices ``spec_zone + 1`` through
    ``spec_zone + relax_zone``. The translated function returns immutable
    JAX arrays with the corresponding zero-based positions updated.
    """

    width = int(spec_bdy_width)
    zone = int(spec_zone)
    relax = int(relax_zone)
    if width < 0:
        raise ValueError("spec_bdy_width must be non-negative")
    if relax <= 1:
        raise ValueError("relax_zone must be greater than 1")

    fcx_arr = jnp.asarray(fcx)
    gcx_arr = jnp.asarray(gcx)
    dtype = fcx_arr.dtype

    if not (specified or nested):
        return fcx_arr, gcx_arr

    fortran_indices = jnp.arange(zone + 1, zone + relax + 1, dtype=jnp.int32)
    zero_indices = fortran_indices - 1
    distance = jnp.asarray(zone + relax, dtype=dtype) - fortran_indices.astype(dtype)
    relax_den = jnp.asarray(relax - 1, dtype=dtype)
    dt_value = jnp.asarray(dt, dtype=dtype)
    fcx_values = (jnp.asarray(0.1, dtype=dtype) / dt_value) * distance / relax_den
    gcx_values = (jnp.asarray(1.0, dtype=dtype) / dt_value / jnp.asarray(50.0, dtype=dtype)) * distance / relax_den

    if specified:
        spongeweight = jnp.exp(-(fortran_indices - (zone + 1)).astype(dtype) * jnp.asarray(spec_exp, dtype=dtype))
        fcx_values = fcx_values * spongeweight
        gcx_values = gcx_values * spongeweight

    return fcx_arr.at[zero_indices].set(fcx_values), gcx_arr.at[zero_indices].set(gcx_values)


__all__ = ["lbc_fcx_gcx"]
