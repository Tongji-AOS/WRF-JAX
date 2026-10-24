"""Snow properties shared by the WRF Noah land and sea-ice schemes."""

from typing import Any

import jax.numpy as jnp


def csnow(dsnow: Any) -> Any:
    dsnow = jnp.asarray(dsnow, dtype=jnp.float32)
    return jnp.asarray(2.0 * 0.11631 * 0.328, dtype=dsnow.dtype) * (
        jnp.asarray(10.0, dtype=dsnow.dtype) ** (jnp.asarray(2.25, dtype=dsnow.dtype) * dsnow)
    )


def snow_new(temp: Any, newsn: Any, snowh: Any, sndens: Any) -> tuple[Any, Any]:
    dtype = jnp.float32
    temp = jnp.asarray(temp, dtype=dtype)
    newsn = jnp.asarray(newsn, dtype=dtype)
    snowhc = jnp.asarray(snowh, dtype=dtype) * jnp.asarray(100.0, dtype=dtype)
    newsnc = newsn * jnp.asarray(100.0, dtype=dtype)
    tempc = temp - jnp.asarray(273.15, dtype=dtype)
    warm_offset = jnp.maximum(tempc + jnp.asarray(15.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype))
    dsnew = jnp.where(
        tempc <= jnp.asarray(-15.0, dtype=dtype),
        jnp.asarray(0.05, dtype=dtype),
        jnp.asarray(0.05, dtype=dtype)
        + jnp.asarray(0.0017, dtype=dtype) * warm_offset ** jnp.asarray(1.5, dtype=dtype),
    )
    hnewc = newsnc / dsnew
    total_depth = snowhc + hnewc
    weighted_density = (snowhc * sndens + hnewc * dsnew) / jnp.where(
        total_depth == jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype), total_depth,
    )
    sndens_out = jnp.where(
        total_depth < jnp.asarray(1.0e-3, dtype=dtype),
        jnp.maximum(dsnew, sndens), weighted_density,
    )
    return total_depth * jnp.asarray(0.01, dtype=dtype), sndens_out


def snowpack(esd: Any, dtsec: Any, snowh: Any, sndens: Any, tsnow: Any, tsoil: Any) -> tuple[Any, Any]:
    dtype = jnp.float32
    esdc = jnp.asarray(esd, dtype=dtype) * jnp.asarray(100.0, dtype=dtype)
    dthr = jnp.asarray(dtsec, dtype=dtype) / jnp.asarray(3600.0, dtype=dtype)
    tsnowc = jnp.asarray(tsnow, dtype=dtype) - jnp.asarray(273.15, dtype=dtype)
    tsoilc = jnp.asarray(tsoil, dtype=dtype) - jnp.asarray(273.15, dtype=dtype)
    tavgc = jnp.asarray(0.5, dtype=dtype) * (tsnowc + tsoilc)
    esdcx = jnp.maximum(esdc, jnp.asarray(1.0e-2, dtype=dtype))
    bfac = dthr * jnp.asarray(0.01, dtype=dtype) * jnp.exp(
        jnp.asarray(0.08, dtype=dtype) * tavgc - jnp.asarray(21.0, dtype=dtype) * sndens
    )
    pexp = jnp.asarray(0.0, dtype=dtype)
    for order in range(4, 0, -1):
        pexp = (jnp.asarray(1.0, dtype=dtype) + pexp) * bfac * esdcx / jnp.asarray(order + 1, dtype=dtype)
    pexp = pexp + jnp.asarray(1.0, dtype=dtype)
    density = jnp.clip(jnp.asarray(sndens, dtype=dtype) * pexp, 0.05, 0.40)
    dw = jnp.asarray(0.13, dtype=dtype) * dthr / jnp.asarray(24.0, dtype=dtype)
    melt_density = jnp.minimum(density * (jnp.asarray(1.0, dtype=dtype) - dw) + dw, jnp.asarray(0.40, dtype=dtype))
    density = jnp.where(tsnowc >= jnp.asarray(0.0, dtype=dtype), melt_density, density)
    return esdc / density * jnp.asarray(0.01, dtype=dtype), density


def snowz0(sncovr: Any, z0brd: Any, snowh: Any) -> Any:
    dtype = jnp.float32
    burial = jnp.asarray(7.0, dtype=dtype) * z0brd - snowh
    z0eff = jnp.where(burial <= jnp.asarray(0.0007, dtype=dtype), jnp.asarray(0.001, dtype=dtype), burial / jnp.asarray(7.0, dtype=dtype))
    return (jnp.asarray(1.0, dtype=dtype) - sncovr) * z0brd + sncovr * z0eff
