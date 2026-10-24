"""WRF 4.5 Noah permanent land-ice surface physics."""

from typing import Any

import jax
import jax.numpy as jnp

from wrf_jax.physics.module_sf_noah_seaice import _snopac, penman
from wrf_jax.physics.noah_snow import csnow, snow_new, snowz0


def sflx_glacial(
    *, dt: Any, sldpth: Any, ffrozp: Any, lwdn: Any, solnet: Any,
    sfcprs: Any, prcp: Any, sfctmp: Any, q2: Any, th2: Any, q2sat: Any,
    dqsdt2: Any, alb: Any, snoalb: Any, tbot: Any, z0brd: Any,
    t1: Any, stc: Any, snowh: Any, sneqv: Any, ch: Any, sncovr: Any,
    snotime1: Any, ribb: Any, lvcoef: Any,
) -> dict[str, Any]:
    """Advance one glacial column, following ``SFLX_GLACIAL``."""
    dtype = jnp.float32
    one = jnp.asarray(1.0, dtype=dtype)
    zsoil = -jnp.cumsum(jnp.asarray(sldpth, dtype=dtype))
    reset = sneqv < 0.10
    sneqv = jnp.where(reset, jnp.asarray(0.10, dtype=dtype), sneqv)
    snowh = jnp.where(reset, jnp.asarray(0.50, dtype=dtype), snowh)
    sndens = sneqv / snowh
    snowng = (prcp > 0.0) & (ffrozp > 0.5)
    frzgra = (prcp > 0.0) & ~snowng & (t1 <= 273.15)
    frozen_precip = snowng | frzgra
    new_snow = prcp * dt * jnp.asarray(0.001, dtype=dtype)
    depth, density = snow_new(sfctmp, new_snow, snowh, sndens)
    cold_firn = (sncovr > 0.99) & (
        (stc[0] < 268.15) | (snowng & (t1 < 273.0) & (sfctmp < 273.0))
    )
    sndens = jnp.where(frozen_precip, jnp.where(cold_firn, 0.2, density), sndens)
    sneqv = jnp.where(frozen_precip, sneqv + new_snow, sneqv)
    snowh = jnp.where(frozen_precip, depth, snowh)
    sncond = csnow(sndens)
    age = jnp.where(snowng, 0.0, snotime1 + dt)
    fresh_albedo = snoalb + lvcoef * (0.85 - snoalb)
    aged_albedo = fresh_albedo * (0.94 ** ((age / 86400.0) ** 0.58))
    albedo = jnp.maximum(jnp.where(snowng, fresh_albedo, aged_albedo), alb)
    emissi = jnp.asarray(0.95, dtype=dtype)
    dsoil = -0.5 * zsoil[0]
    dtot = snowh + dsoil
    df1 = (snowh / dtot) * sncond + (dsoil / dtot) * sncond
    ssoil = df1 * (t1 - stc[0]) / jnp.minimum(dtot, 2.0 * dsoil)
    z0 = snowz0(one, z0brd, snowh)
    fdown = solnet + lwdn
    t2v = sfctmp * (one + 0.61 * q2)
    rho = sfcprs / (jnp.asarray(287.04, dtype=dtype) * t2v)
    rch = rho * jnp.asarray(1004.6, dtype=dtype) * ch
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    etp, flx2, rr = penman(sfctmp, sfcprs, ch, th2, prcp, fdown, t24,
                          ssoil, q2, q2sat, rch, snowng, frzgra, dqsdt2, emissi, t1)
    (etp, t1, stc, snowh, sneqv, _, sncovr, dew, esnow,
     flx1, flx2, flx3, ssoil, snomlt) = _snopac(
        etp, prcp, snowng, dt, df1, q2, t1, sfctmp, t24, th2, fdown,
        stc, rch, rr, one, sneqv, sndens, snowh, zsoil, tbot, flx2,
        emissi, ribb, seaice_albedo_opt=0, glacial=True,
    )
    q1 = q2 + etp * jnp.asarray(1004.5, dtype=dtype) / rch
    sheat = -(ch * jnp.asarray(1004.5, dtype=dtype) * sfcprs) / (
        jnp.asarray(287.04, dtype=dtype) * t2v
    ) * (th2 - t1)
    latent = etp * jnp.asarray(2.83e6, dtype=dtype)
    esnow = esnow * jnp.asarray(2.83e6, dtype=dtype)
    return dict(t1=t1, stc=stc, snowh=snowh, sneqv=sneqv, sncovr=sncovr,
                snotime1=age, albedo=albedo, emissi=emissi, z0=z0,
                eta_kinematic=etp, eta=jnp.where(latent > 0.0, esnow, latent),
                etp=latent, esnow=esnow, sheat=sheat, q1=q1, ssoil=-ssoil,
                fdown=fdown, dew=dew, flx1=flx1, flx2=flx2, flx3=flx3,
                snomlt=snomlt, runoff1=snomlt / dt)


glacial_columns = jax.jit(jax.vmap(lambda inputs: sflx_glacial(**inputs)))
