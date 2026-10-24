"""JAX translation of WRF Noah sea-ice surface physics.

The scalar kernel follows ``module_sf_noah_seaice.F``.  ``seaice_noah``
gathers only active sea-ice columns, batches the scalar kernel with
``jax.vmap``, and scatters the updated state back to the global arrays.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from wrf_jax.physics.module_sf_noahlsm import hstep
from wrf_jax.physics.noah_snow import csnow, snow_new, snowpack, snowz0


CP = 1004.5
CP_PENMAN = 1004.6
CPH2O = 4.218e3
CPICE = 2.106e3
LSUBF = 3.335e5
LSUBC = 2.501e6
LSUBS = 2.83e6
LVH2O = 2.501e6
R_D = 287.0
RD_NOAH = 287.04
RHOWATER = 1000.0
SIGMA = 5.67e-8
STBOLT = 5.67051e-8
TFREEZ = 273.15
XLF = 3.50e5
XLV = 2.5e6


class SflxSeaiceResult(NamedTuple):
    z0brd: Any
    z0: Any
    emissi: Any
    t1: Any
    stc: Any
    snowh: Any
    sneqv: Any
    albedo: Any
    ch: Any
    eta: Any
    sheat: Any
    eta_kinematic: Any
    fdown: Any
    esnow: Any
    dew: Any
    etp: Any
    ssoil: Any
    flx1: Any
    flx2: Any
    flx3: Any
    snomlt: Any
    sncovr: Any
    runoff1: Any
    q1: Any
    ribb: Any


def hrtice(stc: Any, tbot: Any, zsoil: Any, yy: Any, zz1: Any, df1: Any,
           *, glacial: bool = False) -> tuple[Any, Any, Any, Any]:
    """Build WRF's ice heat-diffusion system, including firn depth dependence."""

    dtype = jnp.float32
    stc = jnp.asarray(stc, dtype=dtype)
    zsoil = jnp.asarray(zsoil, dtype=dtype)
    nsoil = int(stc.shape[0])
    hcpct = jnp.asarray(1.72396e6, dtype=dtype)
    if glacial:
        hcpct = jnp.asarray(1.e6, dtype=dtype) * (0.8194 - 0.1309 * 0.5 * zsoil[0])
    ai = jnp.zeros_like(stc)
    bi = jnp.zeros_like(stc)
    ci = jnp.zeros_like(stc)
    rhsts = jnp.zeros_like(stc)

    zbot = zsoil[nsoil - 1]
    if glacial:
        zbot = jnp.asarray(-25.0, dtype=dtype)
    ddz = jnp.asarray(1.0, dtype=dtype) / (-jnp.asarray(0.5, dtype=dtype) * zsoil[1])
    ci = ci.at[0].set((df1 * ddz) / (zsoil[0] * hcpct))
    bi = bi.at[0].set(
        -ci[0]
        + df1
        / (
            jnp.asarray(0.5, dtype=dtype)
            * zsoil[0]
            * zsoil[0]
            * hcpct
            * zz1
        )
    )
    dtsdz = (stc[0] - stc[1]) / (-jnp.asarray(0.5, dtype=dtype) * zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (jnp.asarray(0.5, dtype=dtype) * zsoil[0] * zz1)
    rhsts = rhsts.at[0].set((df1 * dtsdz - ssoil) / (zsoil[0] * hcpct))

    ddz2 = jnp.asarray(0.0, dtype=dtype)
    df1k = df1
    df1n = df1
    for idx in range(1, nsoil):
        if glacial:
            zmd = 0.5 * (zsoil[idx] + zsoil[idx - 1])
            hcpct = jnp.asarray(1.e6, dtype=dtype) * (0.8194 - 0.1309 * zmd)
            df1n = 0.32333 - 0.10073 * zmd
        if idx != nsoil - 1:
            layer_span = zsoil[idx - 1] - zsoil[idx + 1]
            denom_gradient = jnp.asarray(0.5, dtype=dtype) * layer_span
            dtsdz2 = (stc[idx] - stc[idx + 1]) / denom_gradient
            ddz2 = jnp.asarray(2.0, dtype=dtype) / layer_span
            ci = ci.at[idx].set(
                -df1n * ddz2 / ((zsoil[idx - 1] - zsoil[idx]) * hcpct)
            )
        else:
            denom_gradient = (
                jnp.asarray(0.5, dtype=dtype) * (zsoil[idx - 1] + zsoil[idx]) - zbot
            )
            dtsdz2 = (stc[idx] - tbot) / denom_gradient
            ci = ci.at[idx].set(jnp.asarray(0.0, dtype=dtype))
        denom = (zsoil[idx] - zsoil[idx - 1]) * hcpct
        rhsts = rhsts.at[idx].set((df1n * dtsdz2 - df1k * dtsdz) / denom)
        ai = ai.at[idx].set(
            -df1k * ddz / ((zsoil[idx - 1] - zsoil[idx]) * hcpct)
        )
        bi = bi.at[idx].set(-(ai[idx] + ci[idx]))
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2
    return rhsts, ai, bi, ci


def shflx_seaice(stc: Any, dt: Any, yy: Any, zz1: Any, zsoil: Any, tbot: Any, df1: Any,
                 *, glacial: bool = False) -> Any:
    """Advance ice layer temperatures with WRF's implicit solver."""

    stc = jnp.asarray(stc, dtype=jnp.float32)
    rhsts, ai, bi, ci = hrtice(stc, tbot, zsoil, yy, zz1, df1, glacial=glacial)
    result = hstep(jnp.zeros_like(stc), stc, rhsts, dt, int(stc.shape[0]), ai, bi, ci)
    return result.stcout


def penman(
    sfctmp: Any,
    sfcprs: Any,
    ch: Any,
    th2: Any,
    prcp: Any,
    fdown: Any,
    t24: Any,
    ssoil: Any,
    q2: Any,
    q2sat: Any,
    rch: Any,
    snowng: Any,
    frzgra: Any,
    dqsdt2: Any,
    emissi: Any,
    t1: Any,
) -> tuple[Any, Any, Any]:
    """Return WRF sea-ice Penman potential evaporation terms."""

    dtype = jnp.float32
    elcp = jnp.asarray(2.4888e3, dtype=dtype)
    cold = t1 <= jnp.asarray(TFREEZ, dtype=dtype)
    elcp1 = jnp.where(cold, elcp * jnp.asarray(LSUBS / LSUBC, dtype=dtype), elcp)
    lvs = jnp.where(cold, jnp.asarray(LSUBS, dtype=dtype), jnp.asarray(LSUBC, dtype=dtype))
    delta = elcp1 * dqsdt2
    rr = emissi * t24 * jnp.asarray(6.48e-8, dtype=dtype) / (sfcprs * ch) + jnp.asarray(1.0, dtype=dtype)
    precip_heat = jnp.where(snowng, jnp.asarray(CPICE, dtype=dtype), jnp.asarray(CPH2O, dtype=dtype))
    rr = jnp.where(prcp > jnp.asarray(0.0, dtype=dtype), rr + precip_heat * prcp / rch, rr)
    flx2 = jnp.where(frzgra, -jnp.asarray(LSUBF, dtype=dtype) * prcp, jnp.asarray(0.0, dtype=dtype))
    fnet = fdown - emissi * jnp.asarray(SIGMA, dtype=dtype) * t24 - ssoil - flx2
    rad = fnet / rch + th2 - sfctmp
    vapor = elcp1 * (q2sat - q2)
    epsca = (vapor * rr + rad * delta) / (delta + rr)
    return epsca * rch / lvs, flx2, rr


def _snopac(
    etp: Any,
    prcp: Any,
    snowng: Any,
    dt: Any,
    df1: Any,
    q2: Any,
    t1: Any,
    sfctmp: Any,
    t24: Any,
    th2: Any,
    fdown: Any,
    stc: Any,
    rch: Any,
    rr: Any,
    sncovr: Any,
    esd: Any,
    sndens: Any,
    snowh: Any,
    zsoil: Any,
    tbot: Any,
    flx2: Any,
    emissi: Any,
    ribb: Any,
    seaice_albedo_opt: int,
    glacial: bool = False,
) -> tuple[Any, ...]:
    dtype = jnp.float32
    zero = jnp.asarray(0.0, dtype=dtype)

    stable_deposition = (ribb >= jnp.asarray(0.1, dtype=dtype)) & (
        fdown > jnp.asarray(150.0, dtype=dtype)
    )
    corrected_etp = (
        jnp.minimum(etp * (jnp.asarray(1.0, dtype=dtype) - ribb), zero)
        * sncovr
        / jnp.asarray(0.980, dtype=dtype)
        + etp * (jnp.asarray(0.980, dtype=dtype) - sncovr)
    ) / jnp.asarray(0.980, dtype=dtype)
    etp = jnp.where((etp <= zero) & stable_deposition, corrected_etp, etp)
    downward = etp <= zero
    dew = jnp.where(downward, -etp * jnp.asarray(0.001, dtype=dtype), zero)
    esnow = jnp.where(downward, zero, etp * sncovr)
    esnow2 = jnp.where(
        downward,
        etp * jnp.asarray(0.001, dtype=dtype) * dt,
        esnow * jnp.asarray(0.001, dtype=dtype) * dt,
    )
    etanrg = jnp.where(
        downward,
        etp
        * (
            (jnp.asarray(1.0, dtype=dtype) - sncovr) * jnp.asarray(LSUBC, dtype=dtype)
            + sncovr * jnp.asarray(LSUBS, dtype=dtype)
        ),
        esnow * jnp.asarray(LSUBS, dtype=dtype),
    )
    flx1 = jnp.where(
        snowng,
        jnp.asarray(CPICE, dtype=dtype) * prcp * (t1 - sfctmp),
        jnp.where(
            prcp > zero,
            jnp.asarray(CPH2O, dtype=dtype) * prcp * (t1 - sfctmp),
            zero,
        ),
    )
    dsoil = -jnp.asarray(0.5, dtype=dtype) * zsoil[0]
    dtot = snowh + dsoil
    denom = jnp.asarray(1.0, dtype=dtype) + df1 / (dtot * rr * rch)
    t12a = (
        (fdown - flx1 - flx2 - emissi * jnp.asarray(SIGMA, dtype=dtype) * t24) / rch
        + th2
        - sfctmp
        - etanrg / rch
    ) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom

    def frozen_branch(_: None) -> tuple[Any, Any, Any, Any]:
        t1_out = t12
        ssoil_out = df1 * (t1_out - stc[0]) / dtot
        return t1_out, ssoil_out, jnp.maximum(zero, esd - esnow2), zero

    def melt_branch(_: None) -> tuple[Any, Any, Any, Any]:
        t1_out = jnp.asarray(TFREEZ, dtype=dtype)
        melt_depth = jnp.minimum(dtot, 2.0 * dsoil) if glacial else dtot
        ssoil_out = df1 * (t1_out - stc[0]) / melt_depth

        def exhausted(_: None) -> tuple[Any, Any]:
            return zero, zero

        def available(_: None) -> tuple[Any, Any]:
            esd_after_sublimation = esd - esnow2
            seh = rch * (t1_out - th2)
            t14 = (t1_out * t1_out) * (t1_out * t1_out)
            melt_energy = jnp.maximum(
                zero,
                fdown
                - flx1
                - flx2
                - emissi * jnp.asarray(SIGMA, dtype=dtype) * t14
                - ssoil_out
                - seh
                - etanrg,
            )
            proposed_melt = melt_energy * jnp.asarray(0.001 / LSUBF, dtype=dtype) * dt
            retain = esd_after_sublimation - proposed_melt >= jnp.asarray(1.0e-6, dtype=dtype)
            final_melt = jnp.where(retain, proposed_melt, esd_after_sublimation)
            return jnp.where(retain, esd_after_sublimation - proposed_melt, zero), final_melt

        esd_out, snomlt_out = jax.lax.cond(
            esd - esnow2 <= jnp.asarray(1.0e-6, dtype=dtype), exhausted, available, operand=None
        )
        return t1_out, ssoil_out, esd_out, snomlt_out

    t1, ssoil, esd, snomlt = jax.lax.cond(t12 <= jnp.asarray(TFREEZ, dtype=dtype), frozen_branch, melt_branch, operand=None)
    flx3 = jnp.where(
        snomlt > zero,
        snomlt / dt * jnp.asarray(1000.0 * LSUBF, dtype=dtype),
        zero,
    )
    zz1 = jnp.asarray(1.0, dtype=dtype)
    yy = stc[0] - jnp.asarray(0.5, dtype=dtype) * ssoil * zsoil[0] * zz1 / df1
    stc = shflx_seaice(stc, dt, yy, zz1, zsoil, tbot, df1, glacial=glacial)

    compacted_h, compacted_density = snowpack(esd, dt, snowh, sndens, t1, yy)
    if glacial:
        keep = esd >= jnp.asarray(0.10, dtype=dtype)
        snowh = jnp.where(keep, compacted_h, jnp.asarray(0.50, dtype=dtype))
        sndens = jnp.where(keep, compacted_density, sndens)
        esd = jnp.maximum(esd, jnp.asarray(0.10, dtype=dtype))
    elif int(seaice_albedo_opt) == 1:
        keep = esd >= jnp.asarray(0.0001, dtype=dtype)
        snowh = jnp.where(keep, compacted_h, jnp.asarray(0.0005, dtype=dtype))
        sndens = jnp.where(keep, compacted_density, sndens)
        esd = jnp.where(keep, esd, jnp.asarray(0.0001, dtype=dtype))
        sncovr = jnp.where(keep, sncovr, jnp.asarray(0.005, dtype=dtype))
    else:
        keep = esd >= jnp.asarray(0.01, dtype=dtype)
        snowh = jnp.where(keep, compacted_h, jnp.asarray(0.05, dtype=dtype))
        sndens = jnp.where(keep, compacted_density, sndens)
        esd = jnp.where(keep, esd, jnp.asarray(0.01, dtype=dtype))
        sncovr = jnp.where(keep, sncovr, jnp.asarray(1.0, dtype=dtype))
    return etp, t1, stc, snowh, esd, sndens, sncovr, dew, esnow, flx1, flx2, flx3, ssoil, snomlt


def sflx_seaice(
    *,
    seaice_albedo_opt: int,
    seaice_albedo_default: Any,
    seaice_snowdepth_opt: int,
    seaice_snowdepth_max: Any,
    seaice_snowdepth_min: Any,
    ffrozp: Any,
    dt: Any,
    sithick: Any,
    lwdn: Any,
    solnet: Any,
    sfcprs: Any,
    prcp: Any,
    sfctmp: Any,
    q2: Any,
    th2: Any,
    q2sat: Any,
    dqsdt2: Any,
    tbot: Any,
    z0brd: Any,
    emissi: Any,
    t1: Any,
    stc: Any,
    snowh: Any,
    sneqv: Any,
    ch: Any,
    albedosi: Any,
    snowonsi: Any,
    sncovr: Any,
    ribb: Any,
) -> SflxSeaiceResult:
    """Run one WRF Noah sea-ice column."""

    dtype = jnp.float32
    one = jnp.asarray(1.0, dtype=dtype)
    zsoil = -jnp.asarray(sithick, dtype=dtype) * (
        jnp.arange(1, int(stc.shape[0]) + 1, dtype=dtype) / jnp.asarray(int(stc.shape[0]), dtype=dtype)
    )
    z0brd = jnp.asarray(0.001, dtype=dtype)
    if int(seaice_albedo_opt) == 1:
        below_minimum = sneqv < jnp.asarray(0.0001, dtype=dtype)
        sneqv = jnp.where(below_minimum, jnp.asarray(0.0001, dtype=dtype), sneqv)
        snowh = jnp.where(below_minimum, jnp.asarray(0.0005, dtype=dtype), snowh)
    else:
        below_minimum = sneqv < jnp.asarray(0.01, dtype=dtype)
        sneqv = jnp.where(below_minimum, jnp.asarray(0.01, dtype=dtype), sneqv)
        snowh = jnp.where(below_minimum, jnp.asarray(0.05, dtype=dtype), snowh)

    if int(seaice_snowdepth_opt) == 0:
        sndens = sneqv / snowh
        snowh = jnp.maximum(
            jnp.asarray(seaice_snowdepth_min, dtype=dtype),
            jnp.minimum(snowh, jnp.asarray(seaice_snowdepth_max, dtype=dtype)),
        )
        sneqv = snowh * sndens
    else:
        sndens = jnp.asarray(0.3, dtype=dtype)
        snowh = snowonsi
        sneqv = snowh * sndens
    sndens = sneqv / snowh
    sncond = csnow(sndens)

    snowng = (prcp > jnp.asarray(0.0, dtype=dtype)) & (ffrozp > jnp.asarray(0.5, dtype=dtype))
    frzgra = (
        (prcp > jnp.asarray(0.0, dtype=dtype))
        & ~snowng
        & (t1 <= jnp.asarray(TFREEZ, dtype=dtype))
    )
    new_precip = snowng | frzgra
    sn_new = prcp * dt * jnp.asarray(0.001, dtype=dtype)
    new_snowh, new_sndens = snow_new(sfctmp, sn_new, snowh, sndens)
    cold_firn = (sncovr > jnp.asarray(0.99, dtype=dtype)) & (
        (stc[0] < jnp.asarray(TFREEZ - 5.0, dtype=dtype))
        | (snowng & (t1 < jnp.asarray(273.0, dtype=dtype)) & (sfctmp < jnp.asarray(273.0, dtype=dtype)))
    )
    sneqv = jnp.where(new_precip, sneqv + sn_new, sneqv)
    snowh = jnp.where(new_precip, new_snowh, snowh)
    sndens = jnp.where(new_precip, jnp.where(cold_firn, jnp.asarray(0.2, dtype=dtype), new_sndens), sndens)
    sncond = jnp.where(new_precip, csnow(sndens), sncond)

    if int(seaice_albedo_opt) == 0:
        sncovr = one
        emissi = jnp.asarray(0.98, dtype=dtype)
        albedo = jnp.asarray(seaice_albedo_default, dtype=dtype)
    elif int(seaice_albedo_opt) == 1:
        alb_snow = jnp.where(
            t1 < jnp.asarray(268.15, dtype=dtype),
            jnp.asarray(0.8, dtype=dtype),
            jnp.where(
                t1 < jnp.asarray(273.15, dtype=dtype),
                jnp.asarray(0.65, dtype=dtype) - jnp.asarray(0.03, dtype=dtype) * (t1 - jnp.asarray(273.15, dtype=dtype)),
                jnp.asarray(0.65, dtype=dtype),
            ),
        )
        alb_ice = jnp.where(
            sfctmp <= jnp.asarray(273.15, dtype=dtype),
            jnp.asarray(0.65, dtype=dtype),
            jnp.where(
                sfctmp < jnp.asarray(278.15, dtype=dtype),
                jnp.asarray(0.65, dtype=dtype) - jnp.asarray(0.04, dtype=dtype) * (sfctmp - jnp.asarray(273.15, dtype=dtype)),
                jnp.asarray(0.45, dtype=dtype),
            ),
        )
        snow_cover_for_albedo = snowh / (snowh + jnp.asarray(0.10, dtype=dtype))
        albedo = snow_cover_for_albedo * alb_snow + (one - snow_cover_for_albedo) * alb_ice
    else:
        sncovr = one
        emissi = jnp.asarray(0.98, dtype=dtype)
        albedo = albedosi

    df1 = jnp.asarray(2.2, dtype=dtype)
    dsoil = -jnp.asarray(0.5, dtype=dtype) * zsoil[0]
    dtot = snowh + dsoil
    frcsno = snowh / dtot
    frcsoi = dsoil / dtot
    df1a = frcsno * sncond + frcsoi * df1
    df1 = df1a * sncovr + df1 * (one - sncovr)
    ssoil = df1 * (t1 - stc[0]) / dtot
    z0 = snowz0(sncovr, z0brd, snowh)
    fdown = solnet + lwdn
    t2v = sfctmp * (one + jnp.asarray(0.61, dtype=dtype) * q2)
    t24 = (sfctmp * sfctmp) * (sfctmp * sfctmp)
    rho = sfcprs / (jnp.asarray(RD_NOAH, dtype=dtype) * t2v)
    rch = rho * jnp.asarray(CP_PENMAN, dtype=dtype) * ch
    etp, flx2, rr = penman(
        sfctmp, sfcprs, ch, th2, prcp, fdown, t24, ssoil, q2, q2sat,
        rch, snowng, frzgra, dqsdt2, emissi, t1,
    )
    (
        etp,
        t1,
        stc,
        snowh,
        sneqv,
        sndens,
        sncovr,
        dew,
        esnow,
        flx1,
        flx2,
        flx3,
        ssoil,
        snomlt,
    ) = _snopac(
        etp, prcp, snowng, dt, df1, q2, t1, sfctmp, t24, th2, fdown,
        stc, rch, rr, sncovr, sneqv, sndens, snowh, zsoil, tbot, flx2,
        emissi, ribb, seaice_albedo_opt,
    )
    eta_kinematic = etp

    if int(seaice_snowdepth_opt) == 0:
        sndens = sneqv / snowh
        snowh = jnp.maximum(
            jnp.asarray(seaice_snowdepth_min, dtype=dtype),
            jnp.minimum(snowh, jnp.asarray(seaice_snowdepth_max, dtype=dtype)),
        )
        sneqv = snowh * sndens
    else:
        snowh = snowonsi
        sneqv = snowh * jnp.asarray(0.3, dtype=dtype)

    q1 = q2 + eta_kinematic * jnp.asarray(CP, dtype=dtype) / rch
    sheat = -(
        ch * jnp.asarray(CP, dtype=dtype) * sfcprs
        / (jnp.asarray(R_D, dtype=dtype) * t2v)
        * (th2 - t1)
    )
    esnow_energy = esnow * jnp.asarray(LSUBS, dtype=dtype)
    etp_energy = etp * (
        (one - sncovr) * jnp.asarray(LVH2O, dtype=dtype)
        + sncovr * jnp.asarray(LSUBS, dtype=dtype)
    )
    eta = jnp.where(etp_energy > jnp.asarray(0.0, dtype=dtype), esnow_energy, etp_energy)
    ssoil = -ssoil
    runoff1 = snomlt / dt
    return SflxSeaiceResult(
        z0brd=z0brd,
        z0=z0,
        emissi=emissi,
        t1=t1,
        stc=stc,
        snowh=snowh,
        sneqv=sneqv,
        albedo=albedo,
        ch=ch,
        eta=eta,
        sheat=sheat,
        eta_kinematic=eta_kinematic,
        fdown=fdown,
        esnow=esnow_energy,
        dew=dew,
        etp=etp_energy,
        ssoil=ssoil,
        flx1=flx1,
        flx2=flx2,
        flx3=flx3,
        snomlt=snomlt,
        sncovr=sncovr,
        runoff1=runoff1,
        q1=q1,
        ribb=ribb,
    )


def seaice_noah(
    *,
    seaice_albedo_opt: int,
    seaice_albedo_default: float,
    seaice_thickness_opt: int,
    seaice_thickness_default: float,
    seaice_snowdepth_opt: int,
    seaice_snowdepth_max: float,
    seaice_snowdepth_min: float,
    t3d: Any,
    qv3d: Any,
    p8w3d: Any,
    dz8w: Any,
    dt: float,
    frpcpn: bool,
    sr: Any,
    glw: Any,
    swdown: Any,
    rainbl: Any,
    qgh: Any,
    xice: Any,
    albsi: Any,
    icedepth: Any,
    snowsi: Any,
    tslb: Any,
    emiss: Any,
    albedo: Any,
    z02d: Any,
    tsk: Any,
    snow: Any,
    snowc: Any,
    snowh: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    rib: Any,
    znt: Any,
    lh: Any,
    hfx: Any,
    qfx: Any,
    potevp: Any,
    grdflx: Any,
    qsfc: Any,
    acsnow: Any,
    acsnom: Any,
    snopcx: Any,
    sfcrunoff: Any,
    noahres: Any,
    active_i: Any,
    active_j: Any,
    k0: int = 0,
    update_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run WRF ``seaice_noah`` on packed active columns."""

    if int(seaice_albedo_opt) not in (0, 1, 2):
        raise ValueError(f"unsupported seaice_albedo_opt={seaice_albedo_opt}")
    if int(seaice_thickness_opt) not in (0, 1):
        raise ValueError(f"unsupported seaice_thickness_opt={seaice_thickness_opt}")
    if int(seaice_snowdepth_opt) not in (0, 1):
        raise ValueError(f"unsupported seaice_snowdepth_opt={seaice_snowdepth_opt}")

    dtype = jnp.float32
    ii = jnp.asarray(active_i, dtype=jnp.int32)
    jj = jnp.asarray(active_j, dtype=jnp.int32)
    if int(ii.shape[0]) == 0:
        return {
            "tslb": tslb,
            "emiss": emiss,
            "albedo": albedo,
            "z0": z02d,
            "tsk": tsk,
            "snow": snow,
            "snowc": snowc,
            "snowh": snowh,
            "chs": chs,
            "chs2": chs2,
            "cqs2": cqs2,
            "znt": znt,
            "lh": lh,
            "hfx": hfx,
            "qfx": qfx,
            "potevp": potevp,
            "grdflx": grdflx,
            "qsfc": qsfc,
            "acsnow": acsnow,
            "acsnom": acsnom,
            "snopcx": snopcx,
            "sfcrunoff": sfcrunoff,
            "noahres": noahres,
            "albsi": albsi,
            "icedepth": icedepth,
            "snowsi": snowsi,
        }

    arrays2 = {
        name: jnp.asarray(value, dtype=dtype)
        for name, value in {
            "sr": sr,
            "glw": glw,
            "swdown": swdown,
            "rainbl": rainbl,
            "qgh": qgh,
            "xice": xice,
            "albsi": albsi,
            "icedepth": icedepth,
            "snowsi": snowsi,
            "emiss": emiss,
            "albedo": albedo,
            "z02d": z02d,
            "tsk": tsk,
            "snow": snow,
            "snowc": snowc,
            "snowh": snowh,
            "chs": chs,
            "chs2": chs2,
            "cqs2": cqs2,
            "rib": rib,
            "znt": znt,
            "lh": lh,
            "hfx": hfx,
            "qfx": qfx,
            "potevp": potevp,
            "grdflx": grdflx,
            "qsfc": qsfc,
            "acsnow": acsnow,
            "acsnom": acsnom,
            "snopcx": snopcx,
            "sfcrunoff": sfcrunoff,
            "noahres": noahres,
        }.items()
    }
    t3d = jnp.asarray(t3d, dtype=dtype)
    qv3d = jnp.asarray(qv3d, dtype=dtype)
    p8w3d = jnp.asarray(p8w3d, dtype=dtype)
    dz8w = jnp.asarray(dz8w, dtype=dtype)
    tslb = jnp.asarray(tslb, dtype=dtype)
    dt_arr = jnp.asarray(dt, dtype=dtype)

    sfctmp = t3d[ii, k0, jj]
    t1 = arrays2["tsk"][ii, jj]
    emissi = arrays2["emiss"][ii, jj]
    lwdn = arrays2["glw"][ii, jj] * emissi
    sneqv = arrays2["snow"][ii, jj] * jnp.asarray(0.001, dtype=dtype)
    snowh_col = arrays2["snowh"][ii, jj]
    sncovr = arrays2["snowc"][ii, jj]
    solnet = arrays2["swdown"][ii, jj] * (
        jnp.asarray(1.0, dtype=dtype) - arrays2["albedo"][ii, jj]
    )
    sfcprs = jnp.asarray(0.5, dtype=dtype) * (
        p8w3d[ii, k0 + 1, jj] + p8w3d[ii, k0, jj]
    )
    psfc = p8w3d[ii, k0, jj]
    q2 = qv3d[ii, k0, jj] / (jnp.asarray(1.0, dtype=dtype) + qv3d[ii, k0, jj])
    capa = jnp.asarray(R_D / CP, dtype=dtype)
    apes = (jnp.asarray(1.0e5, dtype=dtype) / psfc) ** capa
    apelm = (jnp.asarray(1.0e5, dtype=dtype) / sfcprs) ** capa
    th2 = sfctmp * apelm / apes
    q2sat = arrays2["qgh"][ii, jj] / (
        jnp.asarray(1.0, dtype=dtype) + arrays2["qgh"][ii, jj]
    )
    dqsdt2 = q2sat * jnp.asarray(17.67 * (273.15 - 29.65), dtype=dtype) / (
        sfctmp - jnp.asarray(29.65, dtype=dtype)
    ) ** jnp.asarray(2.0, dtype=dtype)
    e2sat = jnp.asarray(611.2, dtype=dtype) * jnp.exp(
        jnp.asarray(6174.0, dtype=dtype)
        * (jnp.asarray(1.0 / 273.15, dtype=dtype) - jnp.asarray(1.0, dtype=dtype) / sfctmp)
    )
    q2sati = jnp.asarray(0.622, dtype=dtype) * e2sat / (sfcprs - e2sat)
    q2sati = q2sati / (jnp.asarray(1.0, dtype=dtype) + q2sati)
    snow_present = arrays2["snow"][ii, jj] > jnp.asarray(0.0, dtype=dtype)
    warm_skin = t1 > jnp.asarray(273.14, dtype=dtype)
    snow_fraction = arrays2["snowc"][ii, jj]
    ice_qsat = jnp.where(
        warm_skin,
        q2sat * (jnp.asarray(1.0, dtype=dtype) - snow_fraction) + q2sati * snow_fraction,
        q2sati,
    )
    ice_dqsdt = jnp.where(
        warm_skin,
        dqsdt2 * (jnp.asarray(1.0, dtype=dtype) - snow_fraction)
        + q2sati * jnp.asarray(6174.0, dtype=dtype) / (sfctmp**jnp.asarray(2.0, dtype=dtype)) * snow_fraction,
        q2sati * jnp.asarray(6174.0, dtype=dtype) / (sfctmp**jnp.asarray(2.0, dtype=dtype)),
    )
    ice_dqsdt = jnp.where(
        (t1 > jnp.asarray(273.0, dtype=dtype)) & (snow_fraction > jnp.asarray(0.0, dtype=dtype)),
        ice_dqsdt * (jnp.asarray(1.0, dtype=dtype) - snow_fraction),
        ice_dqsdt,
    )
    q2sat = jnp.where(snow_present, ice_qsat, q2sat)
    dqsdt2 = jnp.where(snow_present, ice_dqsdt, dqsdt2)
    prcp = arrays2["rainbl"][ii, jj] / dt_arr
    ffrozp = arrays2["sr"][ii, jj] if bool(frpcpn) else jnp.where(
        sfctmp <= jnp.asarray(TFREEZ, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    if int(seaice_thickness_opt) == 0:
        sithick = jnp.full_like(sfctmp, jnp.asarray(seaice_thickness_default, dtype=dtype))
    else:
        sithick = jnp.clip(arrays2["icedepth"][ii, jj], 0.10, 10.0)
    if int(seaice_snowdepth_opt) == 0:
        snowonsi = jnp.zeros_like(sfctmp)
    else:
        snowonsi = jnp.clip(
            arrays2["snowsi"][ii, jj], seaice_snowdepth_min, seaice_snowdepth_max
        )
        sneqv = snowonsi * jnp.asarray(0.3, dtype=dtype)
        snowh_col = snowonsi
    albedosi = arrays2["albsi"][ii, jj]

    kernel = partial(
        sflx_seaice,
        seaice_albedo_opt=int(seaice_albedo_opt),
        seaice_albedo_default=jnp.asarray(seaice_albedo_default, dtype=dtype),
        seaice_snowdepth_opt=int(seaice_snowdepth_opt),
        seaice_snowdepth_max=jnp.asarray(seaice_snowdepth_max, dtype=dtype),
        seaice_snowdepth_min=jnp.asarray(seaice_snowdepth_min, dtype=dtype),
        dt=dt_arr,
        tbot=jnp.asarray(271.36, dtype=dtype),
    )
    result = jax.vmap(
        lambda ff, thick, lw, sw, pressure, precip, temp, humidity, theta, qsat, dqdt, zbase,
        emissivity, skin, soil_temp, snow_depth, snow_equiv, exchange, albice, prescribed_snow,
        cover, richardson: kernel(
            ffrozp=ff,
            sithick=thick,
            lwdn=lw,
            solnet=sw,
            sfcprs=pressure,
            prcp=precip,
            sfctmp=temp,
            q2=humidity,
            th2=theta,
            q2sat=qsat,
            dqsdt2=dqdt,
            z0brd=zbase,
            emissi=emissivity,
            t1=skin,
            stc=soil_temp,
            snowh=snow_depth,
            sneqv=snow_equiv,
            ch=exchange,
            albedosi=albice,
            snowonsi=prescribed_snow,
            sncovr=cover,
            ribb=richardson,
        )
    )(
        ffrozp,
        sithick,
        lwdn,
        solnet,
        sfcprs,
        prcp,
        sfctmp,
        q2,
        th2,
        q2sat,
        dqsdt2,
        arrays2["z02d"][ii, jj],
        emissi,
        t1,
        tslb[ii, :, jj],
        snowh_col,
        sneqv,
        arrays2["chs"][ii, jj],
        albedosi,
        snowonsi,
        sncovr,
        arrays2["rib"][ii, jj],
    )

    def scatter2(name: str, values: Any) -> Any:
        return arrays2[name].at[ii, jj].set(values)

    tslb_out = tslb.at[ii, :, jj].set(result.stc)
    emiss_out = scatter2("emiss", result.emissi)
    albedo_out = scatter2("albedo", result.albedo)
    z02d_out = scatter2("z02d", result.z0brd)
    tsk_out = scatter2("tsk", result.t1)
    snow_out = scatter2("snow", result.sneqv * jnp.asarray(1000.0, dtype=dtype))
    snowc_out = scatter2("snowc", result.sncovr)
    snowh_out = scatter2("snowh", result.snowh)
    chs_out = scatter2("chs", result.ch)
    chs2_out = scatter2("chs2", arrays2["cqs2"][ii, jj])
    cqs2_values = jnp.where(
        result.q1 > arrays2["qsfc"][ii, jj], result.ch, arrays2["cqs2"][ii, jj]
    )
    cqs2_out = scatter2("cqs2", cqs2_values)
    znt_out = scatter2("znt", result.z0)
    lh_out = scatter2("lh", result.eta)
    hfx_out = scatter2("hfx", result.sheat)
    qfx_out = scatter2("qfx", result.eta_kinematic)
    grdflx_out = scatter2("grdflx", result.ssoil)
    qsfc_out = scatter2("qsfc", result.q1 / (jnp.asarray(1.0, dtype=dtype) - result.q1))

    albsi_out = arrays2["albsi"]
    icedepth_out = arrays2["icedepth"]
    snowsi_out = arrays2["snowsi"]
    if int(seaice_thickness_opt) == 1:
        icedepth_out = icedepth_out.at[ii, jj].set(sithick)
    if int(seaice_snowdepth_opt) == 1:
        snowsi_out = snowsi_out.at[ii, jj].set(snowonsi)

    potevp_out = arrays2["potevp"]
    acsnow_out = arrays2["acsnow"]
    acsnom_out = arrays2["acsnom"]
    snopcx_out = arrays2["snopcx"]
    sfcrunoff_out = arrays2["sfcrunoff"]
    noahres_out = arrays2["noahres"]
    if bool(update_diagnostics):
        fdtw = dt_arr / jnp.asarray(XLV * RHOWATER, dtype=dtype)
        fdtliw = dt_arr / jnp.asarray(RHOWATER * XLF, dtype=dtype)
        potevp_out = potevp_out.at[ii, jj].add(result.etp * fdtw)
        acsnow_out = acsnow_out.at[ii, jj].add(
            jnp.where(ffrozp > jnp.asarray(0.5, dtype=dtype), prcp * dt_arr, jnp.asarray(0.0, dtype=dtype))
        )
        acsnom_out = acsnom_out.at[ii, jj].add(result.snomlt * jnp.asarray(1000.0, dtype=dtype))
        snopcx_out = snopcx_out.at[ii, jj].add(-result.snomlt / fdtliw)
        sfcrunoff_out = sfcrunoff_out.at[ii, jj].add(
            result.runoff1 * dt_arr * jnp.asarray(1000.0, dtype=dtype)
        )
        residual = (
            solnet
            + lwdn
            - result.sheat
            + result.ssoil
            - result.eta
            - result.emissi * jnp.asarray(STBOLT, dtype=dtype) * result.t1**jnp.asarray(4.0, dtype=dtype)
            - result.flx1
            - result.flx2
            - result.flx3
        )
        noahres_out = noahres_out.at[ii, jj].set(residual)

    return {
        "tslb": tslb_out,
        "emiss": emiss_out,
        "albedo": albedo_out,
        "z0": z02d_out,
        "tsk": tsk_out,
        "snow": snow_out,
        "snowc": snowc_out,
        "snowh": snowh_out,
        "chs": chs_out,
        "chs2": chs2_out,
        "cqs2": cqs2_out,
        "znt": znt_out,
        "lh": lh_out,
        "hfx": hfx_out,
        "qfx": qfx_out,
        "potevp": potevp_out,
        "grdflx": grdflx_out,
        "qsfc": qsfc_out,
        "acsnow": acsnow_out,
        "acsnom": acsnom_out,
        "snopcx": snopcx_out,
        "sfcrunoff": sfcrunoff_out,
        "noahres": noahres_out,
        "albsi": albsi_out,
        "icedepth": icedepth_out,
        "snowsi": snowsi_out,
    }


__all__ = [
    "SflxSeaiceResult",
    "csnow",
    "hrtice",
    "penman",
    "seaice_noah",
    "sflx_seaice",
    "shflx_seaice",
    "snow_new",
    "snowpack",
    "snowz0",
]
