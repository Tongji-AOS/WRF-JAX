"""Translations from WRF ``phys/module_sf_noahlsm``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import jax.numpy as jnp

from wrf_jax.core.optimization import optimization_barrier
from wrf_jax.core.real_arithmetic import real_assignment
from wrf_jax.core.wrf_math import wrf_glibc_log10f_positive_normal
from wrf_jax.physics.noah_snow import csnow, snow_new, snowpack, snowz0


RD = 287.04
SIGMA = 5.67e-8
CPH2O = 4.218e3
CPICE = 2.106e3
LSUBF = 3.335e5
CP_MODEL = 1004.5


def _store_wrf_real(
    value: Any,
    dtype: Any,
    rounding_token: Any | None = None,
) -> Any:
    """Materialize a WRF REAL assignment before the next statement."""

    array = jnp.asarray(value, dtype=dtype)
    if rounding_token is None:
        return optimization_barrier(array)
    token = jnp.asarray(rounding_token, dtype=jnp.uint32)
    return jax.lax.select(token == jnp.asarray(0, dtype=jnp.uint32), array, -array)


def _store_wrf_real_operation(
    value: Any,
    rounding_token: Any | None,
) -> Any:
    """Materialize one float32 arithmetic operation like a WRF REAL temporary."""

    array = jnp.asarray(value, dtype=jnp.float32)
    if rounding_token is None:
        return optimization_barrier(array)
    return optimization_barrier(real_assignment(array, rounding_token))


@dataclass(frozen=True)
class NoahLsmParameters:
    lucats: int
    slcats: int
    slpcats: int
    bare: int
    nrotbl: Any
    snuptbl: Any
    rstbl: Any
    rgltbl: Any
    hstbl: Any
    emissmintbl: Any
    emissmaxtbl: Any
    laimintbl: Any
    laimaxtbl: Any
    z0mintbl: Any
    z0maxtbl: Any
    albedomintbl: Any
    albedomaxtbl: Any
    ztopvtbl: Any
    zbotvtbl: Any
    topt_data: Any
    cmcmax_data: Any
    cfactr_data: Any
    rsmax_data: Any
    slop_data: Any
    sbeta_data: Any
    fxexp_data: Any
    csoil_data: Any
    salp_data: Any
    refdk_data: Any
    refkdt_data: Any
    frzk_data: Any
    zbot_data: Any
    czil_data: Any
    lvcoef_data: Any
    bb: Any
    drysmc: Any
    f11: Any
    maxsmc: Any
    refsmc: Any
    satpsi: Any
    satdk: Any
    satdw: Any
    wltsmc: Any
    qtz: Any


@dataclass(frozen=True)
class RedprmResult:
    cfactr: Any
    cmcmax: Any
    rsmax: Any
    topt: Any
    refkdt: Any
    kdt: Any
    sbeta: Any
    shdfac: Any
    rsmin: Any
    rgl: Any
    hs: Any
    zbot: Any
    frzx: Any
    psisat: Any
    slope: Any
    snup: Any
    salp: Any
    bexp: Any
    dksat: Any
    dwsat: Any
    smcmax: Any
    smcwlt: Any
    smcref: Any
    smcdry: Any
    f1: Any
    quartz: Any
    fxexp: Any
    rtdis: Any
    nroot: Any
    czil: Any
    laimin: Any
    laimax: Any
    emissmin: Any
    emissmax: Any
    albedomin: Any
    albedomax: Any
    z0min: Any
    z0max: Any
    csoil: Any
    ptu: Any
    lvcoef: Any
    ztopv: Any
    zbotv: Any


@dataclass(frozen=True)
class PenmanResult:
    epsca: Any
    etp: Any
    flx2: Any
    rch: Any
    rr: Any
    t24: Any
    flx4: Any
    etpn: Any


@dataclass(frozen=True)
class CanresResult:
    pc: Any
    rc: Any
    rcq: Any
    rcs: Any
    rcsoil: Any
    rct: Any


@dataclass(frozen=True)
class WdfcndResult:
    wdf: Any
    wcnd: Any


@dataclass(frozen=True)
class SrtResult:
    rhstt: Any
    ai: Any
    bi: Any
    ci: Any
    runoff1: Any
    runoff2: Any
    sfhead1rt: Any
    infxs1rt: Any


@dataclass(frozen=True)
class SstepResult:
    sh2oout: Any
    cmc: Any
    rhstt: Any
    smc: Any
    ai: Any
    bi: Any
    ci: Any
    runoff3: Any
    infxs1rt: Any


@dataclass(frozen=True)
class SmflxResult:
    smc: Any
    sh2o: Any
    cmc: Any
    runoff1: Any
    runoff2: Any
    runoff3: Any
    drip: Any
    sfhead1rt: Any
    infxs1rt: Any


@dataclass(frozen=True)
class HstepResult:
    stcout: Any
    rhsts: Any
    ai: Any
    bi: Any
    ci: Any


@dataclass(frozen=True)
class HrtResult:
    rhsts: Any
    sh2o: Any
    ai: Any
    bi: Any
    ci: Any
    hcpct_fasdas: Any


@dataclass(frozen=True)
class ShflxResult:
    ssoil: Any
    stc: Any
    sh2o: Any
    t1: Any
    hcpct_fasdas: Any


@dataclass(frozen=True)
class SoilSurfaceResult:
    eta: Any
    smc: Any
    cmc: Any
    t1: Any
    ssoil: Any
    stc: Any
    sh2o: Any
    beta: Any
    drip: Any
    dew: Any
    flx1: Any
    flx3: Any
    runoff1: Any
    runoff2: Any
    runoff3: Any
    edir: Any
    ec: Any
    et: Any
    ett: Any
    sfhead1rt: Any
    infxs1rt: Any
    etpnd1: Any
    qfx_phy: Any
    hcpct_fasdas: Any

    etp: Any
    esnow: Any
    sneqv: Any
    snowh: Any
    sncovr: Any
    snomlt: Any


@dataclass(frozen=True)
class SflxResult:
    eta: Any
    sheat: Any
    eta_kinematic: Any
    fdown: Any
    ec: Any
    edir: Any
    et: Any
    ett: Any
    esnow: Any
    drip: Any
    dew: Any
    beta: Any
    etp: Any
    ssoil: Any
    flx1: Any
    flx2: Any
    flx3: Any
    flx4: Any
    fvb: Any
    fbur: Any
    fgsn: Any
    snomlt: Any
    sncovr: Any
    runoff1: Any
    runoff2: Any
    runoff3: Any
    rc: Any
    pc: Any
    rsmin: Any
    xlai: Any
    rcs: Any
    rct: Any
    rcq: Any
    rcsoil: Any
    soilw: Any
    soilm: Any
    q1: Any
    smav: Any
    emb_rd: Any
    albedo: Any
    z0brd: Any
    z0: Any
    emissi: Any
    alb: Any
    cmc: Any
    t1: Any
    stc: Any
    smc: Any
    sh2o: Any
    snowh: Any
    sneqv: Any
    nroot: Any
    sfhead1rt: Any
    infxs1rt: Any
    etpnd1: Any
    xsda_qfx: Any
    hfx_phy: Any
    qfx_phy: Any
    xqnorm: Any
    hcpct_fasdas: Any
    snotime1: Any
    ribb: Any
    smcwlt: Any
    smcdry: Any
    smcref: Any
    smcmax: Any


@dataclass(frozen=True)
class SnksrcResult:
    tsnsr: Any
    sh2o: Any


@dataclass(frozen=True)
class EvapoResult:
    eta1: Any
    edir: Any
    ec: Any
    et: Any
    ett: Any
    sfhead1rt: Any
    etpnd1: Any


@dataclass(frozen=True)
class Rosr12Result:
    p: Any
    c: Any
    delta: Any


def rosr12(p: Any, a: Any, b: Any, c: Any, d: Any, delta: Any, nsoil: int) -> Rosr12Result:
    """Solve WRF ``ROSR12`` tridiagonal system, returning updated ``P``, ``C`` and ``DELTA``."""

    dtype = jnp.result_type(p, a, b, c, d, delta, jnp.float32)
    p_out = jnp.asarray(p, dtype=dtype)
    a_arr = jnp.asarray(a, dtype=dtype)
    b_arr = jnp.asarray(b, dtype=dtype)
    c_out = jnp.asarray(c, dtype=dtype)
    d_arr = jnp.asarray(d, dtype=dtype)
    delta_out = jnp.asarray(delta, dtype=dtype)
    nsoil_i = int(nsoil)

    c_out = c_out.at[nsoil_i - 1].set(0.0)
    p_out = p_out.at[0].set(-c_out[0] / b_arr[0])
    delta_out = delta_out.at[0].set(d_arr[0] / b_arr[0])
    for idx in range(1, nsoil_i):
        denominator = b_arr[idx] + a_arr[idx] * p_out[idx - 1]
        p_out = p_out.at[idx].set(-c_out[idx] / denominator)
        delta_out = delta_out.at[idx].set((d_arr[idx] - a_arr[idx] * delta_out[idx - 1]) / denominator)

    p_out = p_out.at[nsoil_i - 1].set(delta_out[nsoil_i - 1])
    for k in range(2, nsoil_i + 1):
        kk = nsoil_i - k
        p_out = p_out.at[kk].set(p_out[kk] * p_out[kk + 1] + delta_out[kk])

    return Rosr12Result(p=p_out, c=c_out, delta=delta_out)


def wdfcnd(smc: Any, smcmax: Any, bexp: Any, dksat: Any, dwsat: Any, sicemax: Any) -> WdfcndResult:
    """Return WRF ``WDFCND`` soil water diffusivity and hydraulic conductivity."""

    dtype = jnp.result_type(smc, smcmax, bexp, dksat, dwsat, sicemax, jnp.float32)
    smc = jnp.asarray(smc, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    dksat = jnp.asarray(dksat, dtype=dtype)
    dwsat = jnp.asarray(dwsat, dtype=dtype)
    sicemax = jnp.asarray(sicemax, dtype=dtype)

    factr1 = 0.05 / smcmax
    factr2 = smc / smcmax
    factr1 = jnp.minimum(factr1, factr2)
    expon = bexp + 2.0
    unfrozen_wdf = dwsat * factr2**expon
    vkwgt = 1.0 / (1.0 + (500.0 * sicemax) ** 3.0)
    frozen_wdf = vkwgt * unfrozen_wdf + (1.0 - vkwgt) * dwsat * factr1**expon
    wdf = jnp.where(sicemax > 0.0, frozen_wdf, unfrozen_wdf)
    wcnd = dksat * factr2 ** ((2.0 * bexp) + 3.0)
    return WdfcndResult(wdf=wdf, wcnd=wcnd)


def srt(
    rhstt: Any,
    edir: Any,
    et: Any,
    sh2o: Any,
    sh2oa: Any,
    nsoil: int,
    pcpdrp: Any,
    zsoil: Any,
    dwsat: Any,
    dksat: Any,
    smcmax: Any,
    bexp: Any,
    runoff1: Any,
    runoff2: Any,
    dt: Any,
    smcwlt: Any,
    slope: Any,
    kdt: Any,
    frzx: Any,
    sice: Any,
    ai: Any,
    bi: Any,
    ci: Any,
    sfhead1rt: Any,
    infxs1rt: Any,
) -> SrtResult:
    """Return WRF ``SRT`` soil-water right-hand side and tridiagonal terms."""

    del runoff1, runoff2
    dtype = jnp.result_type(
        rhstt,
        edir,
        et,
        sh2o,
        sh2oa,
        pcpdrp,
        zsoil,
        dwsat,
        dksat,
        smcmax,
        bexp,
        dt,
        smcwlt,
        slope,
        kdt,
        frzx,
        sice,
        ai,
        bi,
        ci,
        sfhead1rt,
        infxs1rt,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    if nsoil_i < 2:
        raise ValueError("SRT requires at least two soil layers")

    rhstt_out = jnp.asarray(rhstt, dtype=dtype)
    et_arr = jnp.asarray(et, dtype=dtype)
    sh2o_arr = jnp.asarray(sh2o, dtype=dtype)
    sh2oa_arr = jnp.asarray(sh2oa, dtype=dtype)
    sice_arr = jnp.asarray(sice, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    ai_out = jnp.asarray(ai, dtype=dtype)
    bi_out = jnp.asarray(bi, dtype=dtype)
    ci_out = jnp.asarray(ci, dtype=dtype)
    edir = jnp.asarray(edir, dtype=dtype)
    pcpdrp = jnp.asarray(pcpdrp, dtype=dtype)
    dwsat = jnp.asarray(dwsat, dtype=dtype)
    dksat = jnp.asarray(dksat, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    slope = jnp.asarray(slope, dtype=dtype)
    kdt = jnp.asarray(kdt, dtype=dtype)
    frzx = jnp.asarray(frzx, dtype=dtype)
    sfhead1rt_out = jnp.asarray(sfhead1rt, dtype=dtype)
    infxs1rt_out = jnp.asarray(infxs1rt, dtype=dtype)

    sicemax = jnp.asarray(0.0, dtype=dtype)
    for idx in range(nsoil_i):
        sicemax = jnp.maximum(sicemax, sice_arr[idx])

    pddum = pcpdrp
    runoff1_out = jnp.asarray(0.0, dtype=dtype)

    has_precip = pcpdrp != 0.0
    dt1 = dt / 86400.0
    smcav = smcmax - smcwlt
    dmax = [jnp.asarray(0.0, dtype=dtype)] * nsoil_i
    dmax[0] = -zsoil_arr[0] * smcav
    dice = -zsoil_arr[0] * sice_arr[0]
    dmax[0] = dmax[0] * (1.0 - (sh2oa_arr[0] + sice_arr[0] - smcwlt) / smcav)
    dd = dmax[0]

    for ks in range(1, nsoil_i):
        dice = dice + (zsoil_arr[ks - 1] - zsoil_arr[ks]) * sice_arr[ks]
        dmax[ks] = (zsoil_arr[ks - 1] - zsoil_arr[ks]) * smcav
        dmax[ks] = dmax[ks] * (1.0 - (sh2oa_arr[ks] + sice_arr[ks] - smcwlt) / smcav)
        dd = dd + dmax[ks]

    val = 1.0 - jnp.exp(-kdt * dt1)
    ddt = dd * val
    px = pcpdrp * dt
    px = jnp.where(px < 0.0, 0.0, px)
    infmax = (px * (ddt / jnp.where((px + ddt) != 0.0, px + ddt, 1.0))) / dt
    cvfrz = 3
    acrt = cvfrz * frzx / jnp.where(dice != 0.0, dice, 1.0)
    total = jnp.asarray(1.0, dtype=dtype)
    for j in range(1, cvfrz):
        kval = 1
        for jj in range(j + 1, cvfrz):
            kval *= jj
        total = total + (acrt ** (cvfrz - j)) / float(kval)
    fcr_frozen = 1.0 - jnp.exp(-acrt) * total
    fcr = jnp.where(dice > 1.0e-2, fcr_frozen, 1.0)

    infmax = infmax * fcr
    first_cond = wdfcnd(sh2oa_arr[0], smcmax, bexp, dksat, dwsat, sicemax)
    infmax = jnp.maximum(infmax, first_cond.wcnd)
    infmax = jnp.minimum(infmax, px / dt)
    runoff_cond = has_precip & (pcpdrp > infmax)
    runoff1_candidate = pcpdrp - infmax
    runoff1_out = jnp.where(runoff_cond, runoff1_candidate, runoff1_out)
    infxs1rt_out = jnp.where(runoff_cond, runoff1_candidate * dt * 1000.0, infxs1rt_out)
    pddum = jnp.where(runoff_cond, infmax, pddum)

    first = wdfcnd(sh2oa_arr[0], smcmax, bexp, dksat, dwsat, sicemax)
    wdf = first.wdf
    wcnd = first.wcnd
    ddz = 1.0 / (-0.5 * zsoil_arr[1])
    ai_out = ai_out.at[0].set(0.0)
    bi_out = bi_out.at[0].set(wdf * ddz / (-zsoil_arr[0]))
    ci_out = ci_out.at[0].set(-bi_out[0])
    dsmdz = (sh2o_arr[0] - sh2o_arr[1]) / (-0.5 * zsoil_arr[1])
    rhstt_out = rhstt_out.at[0].set((wdf * dsmdz + wcnd - pddum + edir + et_arr[0]) / zsoil_arr[0])

    ddz2 = jnp.asarray(0.0, dtype=dtype)
    runoff2_out = jnp.asarray(0.0, dtype=dtype)
    for k in range(1, nsoil_i):
        denom2 = zsoil_arr[k - 1] - zsoil_arr[k]
        if k != nsoil_i - 1:
            slopx = jnp.asarray(1.0, dtype=dtype)
            second = wdfcnd(sh2oa_arr[k], smcmax, bexp, dksat, dwsat, sicemax)
            wdf2 = second.wdf
            wcnd2 = second.wcnd
            denom = zsoil_arr[k - 1] - zsoil_arr[k + 1]
            dsmdz2 = (sh2o_arr[k] - sh2o_arr[k + 1]) / (denom * 0.5)
            ddz2 = 2.0 / denom
            ci_out = ci_out.at[k].set(-wdf2 * ddz2 / denom2)
        else:
            slopx = slope
            second = wdfcnd(sh2oa_arr[nsoil_i - 1], smcmax, bexp, dksat, dwsat, sicemax)
            wdf2 = second.wdf
            wcnd2 = second.wcnd
            dsmdz2 = jnp.asarray(0.0, dtype=dtype)
            ci_out = ci_out.at[k].set(0.0)

        numer = (wdf2 * dsmdz2) + slopx * wcnd2 - (wdf * dsmdz) - wcnd + et_arr[k]
        rhstt_out = rhstt_out.at[k].set(numer / (-denom2))
        ai_out = ai_out.at[k].set(-wdf * ddz / denom2)
        bi_out = bi_out.at[k].set(-(ai_out[k] + ci_out[k]))
        if k == nsoil_i - 1:
            runoff2_out = slopx * wcnd2
        else:
            wdf = wdf2
            wcnd = wcnd2
            dsmdz = dsmdz2
            ddz = ddz2

    return SrtResult(
        rhstt=rhstt_out,
        ai=ai_out,
        bi=bi_out,
        ci=ci_out,
        runoff1=runoff1_out,
        runoff2=runoff2_out,
        sfhead1rt=sfhead1rt_out,
        infxs1rt=infxs1rt_out,
    )


def sstep(
    sh2oout: Any,
    sh2oin: Any,
    cmc: Any,
    rhstt: Any,
    rhsct: Any,
    dt: Any,
    nsoil: int,
    smcmax: Any,
    cmcmax: Any,
    runoff3: Any,
    zsoil: Any,
    smc: Any,
    sice: Any,
    ai: Any,
    bi: Any,
    ci: Any,
    infxs1rt: Any,
) -> SstepResult:
    """Return WRF ``SSTEP`` updated soil liquid water and implicit coefficients."""

    del runoff3
    dtype = jnp.result_type(
        sh2oout,
        sh2oin,
        cmc,
        rhstt,
        rhsct,
        dt,
        smcmax,
        cmcmax,
        zsoil,
        smc,
        sice,
        ai,
        bi,
        ci,
        infxs1rt,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    sh2oout_arr = jnp.asarray(sh2oout, dtype=dtype)
    sh2oin_arr = jnp.asarray(sh2oin, dtype=dtype)
    cmc_out = jnp.asarray(cmc, dtype=dtype)
    rhstt_out = jnp.asarray(rhstt, dtype=dtype)
    rhsct = jnp.asarray(rhsct, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    cmcmax = jnp.asarray(cmcmax, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    smc_out = jnp.asarray(smc, dtype=dtype)
    sice_arr = jnp.asarray(sice, dtype=dtype)
    ai_out = jnp.asarray(ai, dtype=dtype)
    bi_out = jnp.asarray(bi, dtype=dtype)
    ci_out = jnp.asarray(ci, dtype=dtype)
    infxs1rt_out = jnp.asarray(infxs1rt, dtype=dtype)

    for k in range(nsoil_i):
        rhstt_out = rhstt_out.at[k].set(rhstt_out[k] * dt)
        ai_out = ai_out.at[k].set(ai_out[k] * dt)
        bi_out = bi_out.at[k].set(1.0 + bi_out[k] * dt)
        ci_out = ci_out.at[k].set(ci_out[k] * dt)

    rhsttin = rhstt_out
    ciin = ci_out
    solved = rosr12(ci_out, ai_out, bi_out, ciin, rhsttin, rhstt_out, nsoil_i)
    ci_out = solved.p
    rhstt_out = solved.delta

    wplus = jnp.asarray(0.0, dtype=dtype)
    ddz = -zsoil_arr[0]
    for k in range(nsoil_i):
        if k != 0:
            ddz = zsoil_arr[k - 1] - zsoil_arr[k]
        sh2o_value = sh2oin_arr[k] + ci_out[k] + wplus / ddz
        stot = sh2o_value + sice_arr[k]
        if k == 0:
            layer_dz = -zsoil_arr[0]
        else:
            layer_dz = -zsoil_arr[k] + zsoil_arr[k - 1]
        wplus = jnp.where(stot > smcmax, (stot - smcmax) * layer_dz, 0.0)
        smc_value = jnp.maximum(jnp.minimum(stot, smcmax), 0.02)
        smc_out = smc_out.at[k].set(smc_value)
        sh2oout_arr = sh2oout_arr.at[k].set(jnp.maximum(smc_value - sice_arr[k], 0.0))

    runoff3_out = wplus
    cmc_out = cmc_out + dt * rhsct
    cmc_out = jnp.where(cmc_out < 1.0e-20, 0.0, cmc_out)
    cmc_out = jnp.minimum(cmc_out, cmcmax)

    return SstepResult(
        sh2oout=sh2oout_arr,
        cmc=cmc_out,
        rhstt=rhstt_out,
        smc=smc_out,
        ai=ai_out,
        bi=bi_out,
        ci=ci_out,
        runoff3=runoff3_out,
        infxs1rt=infxs1rt_out,
    )


def smflx(
    smc: Any,
    nsoil: int,
    cmc: Any,
    dt: Any,
    prcp1: Any,
    zsoil: Any,
    sh2o: Any,
    slope: Any,
    kdt: Any,
    frzfact: Any,
    smcmax: Any,
    bexp: Any,
    smcwlt: Any,
    dksat: Any,
    dwsat: Any,
    shdfac: Any,
    cmcmax: Any,
    runoff1: Any,
    runoff2: Any,
    runoff3: Any,
    edir: Any,
    ec: Any,
    et: Any,
    drip: Any,
    sfhead1rt: Any,
    infxs1rt: Any,
    irrigation_channel: Any | None = None,
) -> SmflxResult:
    """Return WRF ``SMFLX`` canopy and soil moisture updates."""

    del runoff1, runoff2, runoff3, drip
    dtype = jnp.result_type(
        smc,
        cmc,
        dt,
        prcp1,
        zsoil,
        sh2o,
        slope,
        kdt,
        frzfact,
        smcmax,
        bexp,
        smcwlt,
        dksat,
        dwsat,
        shdfac,
        cmcmax,
        edir,
        ec,
        et,
        sfhead1rt,
        infxs1rt,
        0.0 if irrigation_channel is None else irrigation_channel,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    smc_out = jnp.asarray(smc, dtype=dtype)
    sh2o_out = jnp.asarray(sh2o, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    et_arr = jnp.asarray(et, dtype=dtype)
    cmc_out = jnp.asarray(cmc, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    prcp1 = jnp.asarray(prcp1, dtype=dtype)
    slope = jnp.asarray(slope, dtype=dtype)
    kdt = jnp.asarray(kdt, dtype=dtype)
    frzfact = jnp.asarray(frzfact, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    dksat = jnp.asarray(dksat, dtype=dtype)
    dwsat = jnp.asarray(dwsat, dtype=dtype)
    shdfac = jnp.asarray(shdfac, dtype=dtype)
    cmcmax = jnp.asarray(cmcmax, dtype=dtype)
    edir = jnp.asarray(edir, dtype=dtype)
    ec = jnp.asarray(ec, dtype=dtype)
    sfhead1rt_out = jnp.asarray(sfhead1rt, dtype=dtype)
    infxs1rt_out = jnp.asarray(infxs1rt, dtype=dtype)

    rhsct = shdfac * prcp1 - ec
    drip_out = jnp.asarray(0.0, dtype=dtype)
    trhsct = dt * rhsct
    excess = cmc_out + trhsct
    drip_out = jnp.where(excess > cmcmax, excess - cmcmax, drip_out)
    pcpdrp = (1.0 - shdfac) * prcp1 + drip_out / dt
    if irrigation_channel is not None:
        irrigation_value = jnp.asarray(irrigation_channel, dtype=dtype)
        pcpdrp = pcpdrp + jnp.where(irrigation_value != 0.0, 0.001 * irrigation_value, 0.0)

    sice = smc_out - sh2o_out
    fac2 = jnp.asarray(0.0, dtype=dtype)
    for i in range(nsoil_i):
        fac2 = jnp.maximum(fac2, sh2o_out[i] / smcmax)
    flimit = fac2mit(smcmax)
    use_predictor_corrector = ((pcpdrp * dt) > (0.0001 * 1000.0 * (-zsoil_arr[0]) * smcmax)) | (fac2 > flimit)

    zeros = jnp.zeros((nsoil_i,), dtype=dtype)
    first_srt = srt(
        zeros,
        edir,
        et_arr,
        sh2o_out,
        sh2o_out,
        nsoil_i,
        pcpdrp,
        zsoil_arr,
        dwsat,
        dksat,
        smcmax,
        bexp,
        0.0,
        0.0,
        dt,
        smcwlt,
        slope,
        kdt,
        frzfact,
        sice,
        zeros,
        zeros,
        zeros,
        sfhead1rt_out,
        infxs1rt_out,
    )
    first_step = sstep(
        zeros,
        sh2o_out,
        jnp.asarray(0.0, dtype=dtype),
        first_srt.rhstt,
        rhsct,
        dt,
        nsoil_i,
        smcmax,
        cmcmax,
        0.0,
        zsoil_arr,
        smc_out,
        sice,
        first_srt.ai,
        first_srt.bi,
        first_srt.ci,
        first_srt.infxs1rt,
    )
    sh2oa = (sh2o_out + first_step.sh2oout) * 0.5
    second_srt = srt(
        zeros,
        edir,
        et_arr,
        sh2o_out,
        sh2oa,
        nsoil_i,
        pcpdrp,
        zsoil_arr,
        dwsat,
        dksat,
        smcmax,
        bexp,
        first_srt.runoff1,
        first_srt.runoff2,
        dt,
        smcwlt,
        slope,
        kdt,
        frzfact,
        sice,
        zeros,
        zeros,
        zeros,
        first_srt.sfhead1rt,
        first_step.infxs1rt,
    )
    predictor_step = sstep(
        sh2o_out,
        sh2o_out,
        cmc_out,
        second_srt.rhstt,
        rhsct,
        dt,
        nsoil_i,
        smcmax,
        cmcmax,
        0.0,
        zsoil_arr,
        first_step.smc,
        sice,
        second_srt.ai,
        second_srt.bi,
        second_srt.ci,
        second_srt.infxs1rt,
    )
    only_srt = srt(
        zeros,
        edir,
        et_arr,
        sh2o_out,
        sh2o_out,
        nsoil_i,
        pcpdrp,
        zsoil_arr,
        dwsat,
        dksat,
        smcmax,
        bexp,
        0.0,
        0.0,
        dt,
        smcwlt,
        slope,
        kdt,
        frzfact,
        sice,
        zeros,
        zeros,
        zeros,
        sfhead1rt_out,
        infxs1rt_out,
    )
    direct_step = sstep(
        sh2o_out,
        sh2o_out,
        cmc_out,
        only_srt.rhstt,
        rhsct,
        dt,
        nsoil_i,
        smcmax,
        cmcmax,
        0.0,
        zsoil_arr,
        smc_out,
        sice,
        only_srt.ai,
        only_srt.bi,
        only_srt.ci,
        only_srt.infxs1rt,
    )
    final_step = SstepResult(
        sh2oout=jnp.where(use_predictor_corrector, predictor_step.sh2oout, direct_step.sh2oout),
        cmc=jnp.where(use_predictor_corrector, predictor_step.cmc, direct_step.cmc),
        rhstt=jnp.where(use_predictor_corrector, predictor_step.rhstt, direct_step.rhstt),
        smc=jnp.where(use_predictor_corrector, predictor_step.smc, direct_step.smc),
        ai=jnp.where(use_predictor_corrector, predictor_step.ai, direct_step.ai),
        bi=jnp.where(use_predictor_corrector, predictor_step.bi, direct_step.bi),
        ci=jnp.where(use_predictor_corrector, predictor_step.ci, direct_step.ci),
        runoff3=jnp.where(use_predictor_corrector, predictor_step.runoff3, direct_step.runoff3),
        infxs1rt=jnp.where(use_predictor_corrector, predictor_step.infxs1rt, direct_step.infxs1rt),
    )
    runoff1_out = jnp.where(use_predictor_corrector, second_srt.runoff1, only_srt.runoff1)
    runoff2_out = jnp.where(use_predictor_corrector, second_srt.runoff2, only_srt.runoff2)
    sfhead1rt_out = jnp.where(use_predictor_corrector, second_srt.sfhead1rt, only_srt.sfhead1rt)

    return SmflxResult(
        smc=final_step.smc,
        sh2o=final_step.sh2oout,
        cmc=final_step.cmc,
        runoff1=runoff1_out,
        runoff2=runoff2_out,
        runoff3=final_step.runoff3,
        drip=drip_out,
        sfhead1rt=sfhead1rt_out,
        infxs1rt=final_step.infxs1rt,
    )


def hrt(
    rhsts: Any,
    stc: Any,
    smc: Any,
    smcmax: Any,
    nsoil: int,
    zsoil: Any,
    yy: Any,
    zz1: Any,
    tbot: Any,
    zbot: Any,
    psisat: Any,
    sh2o: Any,
    dt: Any,
    bexp: Any,
    soiltyp: int,
    opt_thcnd: int,
    f1: Any,
    df1: Any,
    quartz: Any,
    csoil: Any,
    ai: Any,
    bi: Any,
    ci: Any,
    vegtyp: int,
    isurban: int,
    hcpct_fasdas: Any,
    rounding_token: Any | None = None,
) -> HrtResult:
    """Return WRF ``HRT`` soil heat right-hand side and tridiagonal terms."""

    del f1, hcpct_fasdas
    dtype = jnp.result_type(
        rhsts,
        stc,
        smc,
        smcmax,
        zsoil,
        yy,
        zz1,
        tbot,
        zbot,
        psisat,
        sh2o,
        dt,
        bexp,
        df1,
        quartz,
        csoil,
        ai,
        bi,
        ci,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    stc_arr = jnp.asarray(stc, dtype=dtype)
    smc_arr = jnp.asarray(smc, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    sh2o_out = jnp.asarray(sh2o, dtype=dtype)
    rhsts_out = jnp.asarray(rhsts, dtype=dtype)
    ai_out = jnp.asarray(ai, dtype=dtype)
    bi_out = jnp.asarray(bi, dtype=dtype)
    ci_out = jnp.asarray(ci, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    yy = jnp.asarray(yy, dtype=dtype)
    zz1 = jnp.asarray(zz1, dtype=dtype)
    tbot = jnp.asarray(tbot, dtype=dtype)
    zbot = jnp.asarray(zbot, dtype=dtype)
    psisat = jnp.asarray(psisat, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    df1 = jnp.asarray(df1, dtype=dtype)
    quartz = jnp.asarray(quartz, dtype=dtype)
    csoil = jnp.asarray(csoil, dtype=dtype)

    t0 = jnp.asarray(273.15, dtype=dtype)
    cair = jnp.asarray(1004.0, dtype=dtype)
    cice = jnp.asarray(2.106e6, dtype=dtype)
    ch2o = jnp.asarray(4.2e6, dtype=dtype)
    csoil_loc = jnp.where(jnp.asarray(vegtyp, dtype=jnp.int32) == int(isurban), jnp.asarray(3.0e6, dtype=dtype), csoil)

    hcpct = sh2o_out[0] * ch2o + (1.0 - smcmax) * csoil_loc + (smcmax - smc_arr[0]) * cair + (smc_arr[0] - sh2o_out[0]) * cice
    hcpct_fasdas_out = hcpct
    ddz = 1.0 / (-0.5 * zsoil_arr[1])
    ai_out = ai_out.at[0].set(0.0)
    ci_out = ci_out.at[0].set((df1 * ddz) / (zsoil_arr[0] * hcpct))
    bi_out = bi_out.at[0].set(-ci_out[0] + df1 / (0.5 * zsoil_arr[0] * zsoil_arr[0] * hcpct * zz1))
    dtsdz = (stc_arr[0] - stc_arr[1]) / (-0.5 * zsoil_arr[1])
    ssoil = df1 * (stc_arr[0] - yy) / (0.5 * zsoil_arr[0] * zz1)
    denom = zsoil_arr[0] * hcpct
    rhsts_out = rhsts_out.at[0].set((df1 * dtsdz - ssoil) / denom)
    qtot = -rhsts_out[0] * denom

    sice = smc_arr[0] - sh2o_out[0]
    tsurf = (yy + (zz1 - 1.0) * stc_arr[0]) / zz1
    tbk = tbnd(stc_arr[0], stc_arr[1], zsoil_arr, zbot, 1, nsoil_i)
    sink_cond = (sice > 0.0) | (stc_arr[0] < t0) | (tsurf < t0) | (tbk < t0)
    tavg = tmpavg(tsurf, stc_arr[0], tbk, zsoil_arr, nsoil_i, 1, rounding_token)
    sink = snksrc(0.0, tavg, smc_arr[0], sh2o_out[0], zsoil_arr, nsoil_i, smcmax, psisat, bexp, dt, 1, qtot)
    sh2o_out = sh2o_out.at[0].set(jnp.where(sink_cond, sink.sh2o, sh2o_out[0]))
    rhsts_out = rhsts_out.at[0].set(jnp.where(sink_cond, rhsts_out[0] - sink.tsnsr / denom, rhsts_out[0]))

    ddz2 = jnp.asarray(0.0, dtype=dtype)
    df1k = df1
    for k in range(2, nsoil_i + 1):
        idx = k - 1
        hcpct = (
            sh2o_out[idx] * ch2o
            + (1.0 - smcmax) * csoil_loc
            + (smcmax - smc_arr[idx]) * cair
            + (smc_arr[idx] - sh2o_out[idx]) * cice
        )
        df1n = tdfcnd(smc_arr[idx], quartz, smcmax, sh2o_out[idx], bexp, psisat, soiltyp, int(opt_thcnd))
        df1n = jnp.where(jnp.asarray(vegtyp, dtype=jnp.int32) == int(isurban), jnp.asarray(3.24, dtype=dtype), df1n)

        if k != nsoil_i:
            denom_grad = 0.5 * (zsoil_arr[idx - 1] - zsoil_arr[idx + 1])
            dtsdz2 = (stc_arr[idx] - stc_arr[idx + 1]) / denom_grad
            ddz2 = 2.0 / (zsoil_arr[idx - 1] - zsoil_arr[idx + 1])
            ci_out = ci_out.at[idx].set(-df1n * ddz2 / ((zsoil_arr[idx - 1] - zsoil_arr[idx]) * hcpct))
            tbk1 = tbnd(stc_arr[idx], stc_arr[idx + 1], zsoil_arr, zbot, k, nsoil_i)
        else:
            denom_grad = 0.5 * (zsoil_arr[idx - 1] + zsoil_arr[idx]) - zbot
            dtsdz2 = (stc_arr[idx] - tbot) / denom_grad
            ci_out = ci_out.at[idx].set(0.0)
            tbk1 = tbnd(stc_arr[idx], tbot, zsoil_arr, zbot, k, nsoil_i)

        denom = (zsoil_arr[idx] - zsoil_arr[idx - 1]) * hcpct
        rhsts_out = rhsts_out.at[idx].set((df1n * dtsdz2 - df1k * dtsdz) / denom)
        qtot = -denom * rhsts_out[idx]
        sice = smc_arr[idx] - sh2o_out[idx]
        tavg = tmpavg(tbk, stc_arr[idx], tbk1, zsoil_arr, nsoil_i, k, rounding_token)
        sink_cond = (sice > 0.0) | (stc_arr[idx] < t0) | (tbk < t0) | (tbk1 < t0)
        sink = snksrc(0.0, tavg, smc_arr[idx], sh2o_out[idx], zsoil_arr, nsoil_i, smcmax, psisat, bexp, dt, k, qtot)
        sh2o_out = sh2o_out.at[idx].set(jnp.where(sink_cond, sink.sh2o, sh2o_out[idx]))
        rhsts_out = rhsts_out.at[idx].set(jnp.where(sink_cond, rhsts_out[idx] - sink.tsnsr / denom, rhsts_out[idx]))

        ai_out = ai_out.at[idx].set(-df1k * ddz / ((zsoil_arr[idx - 1] - zsoil_arr[idx]) * hcpct))
        bi_out = bi_out.at[idx].set(-(ai_out[idx] + ci_out[idx]))
        tbk = tbk1
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2

    return HrtResult(rhsts=rhsts_out, sh2o=sh2o_out, ai=ai_out, bi=bi_out, ci=ci_out, hcpct_fasdas=hcpct_fasdas_out)


def hstep(stcout: Any, stcin: Any, rhsts: Any, dt: Any, nsoil: int, ai: Any, bi: Any, ci: Any) -> HstepResult:
    """Return WRF ``HSTEP`` updated soil temperatures and implicit coefficients."""

    dtype = jnp.result_type(stcout, stcin, rhsts, dt, ai, bi, ci, jnp.float32)
    nsoil_i = int(nsoil)
    stcout_arr = jnp.asarray(stcout, dtype=dtype)
    stcin_arr = jnp.asarray(stcin, dtype=dtype)
    rhsts_out = jnp.asarray(rhsts, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    ai_out = jnp.asarray(ai, dtype=dtype)
    bi_out = jnp.asarray(bi, dtype=dtype)
    ci_out = jnp.asarray(ci, dtype=dtype)

    for k in range(nsoil_i):
        rhsts_out = rhsts_out.at[k].set(rhsts_out[k] * dt)
        ai_out = ai_out.at[k].set(ai_out[k] * dt)
        bi_out = bi_out.at[k].set(1.0 + bi_out[k] * dt)
        ci_out = ci_out.at[k].set(ci_out[k] * dt)

    rhstsin = rhsts_out
    ciin = ci_out
    solved = rosr12(ci_out, ai_out, bi_out, ciin, rhstsin, rhsts_out, nsoil_i)
    ci_out = solved.p
    rhsts_out = solved.delta

    for k in range(nsoil_i):
        stcout_arr = stcout_arr.at[k].set(stcin_arr[k] + ci_out[k])

    return HstepResult(stcout=stcout_arr, rhsts=rhsts_out, ai=ai_out, bi=bi_out, ci=ci_out)


def shflx(
    ssoil: Any,
    stc: Any,
    smc: Any,
    smcmax: Any,
    nsoil: int,
    t1: Any,
    dt: Any,
    yy: Any,
    zz1: Any,
    zsoil: Any,
    tbot: Any,
    zbot: Any,
    smcwlt: Any,
    psisat: Any,
    sh2o: Any,
    bexp: Any,
    f1: Any,
    df1: Any,
    quartz: Any,
    csoil: Any,
    vegtyp: int,
    isurban: int,
    soiltyp: int,
    opt_thcnd: int,
    hcpct_fasdas: Any,
    rounding_token: Any | None = None,
) -> ShflxResult:
    """Return WRF ``SHFLX`` updated soil temperatures, liquid water and heat flux."""

    del ssoil, t1, smcwlt
    dtype = jnp.result_type(
        stc,
        smc,
        smcmax,
        dt,
        yy,
        zz1,
        zsoil,
        tbot,
        zbot,
        psisat,
        sh2o,
        bexp,
        f1,
        df1,
        quartz,
        csoil,
        hcpct_fasdas,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    stc_arr = jnp.asarray(stc, dtype=dtype)
    smc_arr = jnp.asarray(smc, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    sh2o_arr = jnp.asarray(sh2o, dtype=dtype)
    zeros = jnp.zeros((nsoil_i,), dtype=dtype)

    heat = hrt(
        zeros,
        stc_arr,
        smc_arr,
        jnp.asarray(smcmax, dtype=dtype),
        nsoil_i,
        zsoil_arr,
        jnp.asarray(yy, dtype=dtype),
        jnp.asarray(zz1, dtype=dtype),
        jnp.asarray(tbot, dtype=dtype),
        jnp.asarray(zbot, dtype=dtype),
        jnp.asarray(psisat, dtype=dtype),
        sh2o_arr,
        jnp.asarray(dt, dtype=dtype),
        jnp.asarray(bexp, dtype=dtype),
        soiltyp,
        int(opt_thcnd),
        jnp.asarray(f1, dtype=dtype),
        jnp.asarray(df1, dtype=dtype),
        jnp.asarray(quartz, dtype=dtype),
        jnp.asarray(csoil, dtype=dtype),
        zeros,
        zeros,
        zeros,
        vegtyp,
        int(isurban),
        jnp.asarray(hcpct_fasdas, dtype=dtype),
        rounding_token,
    )
    stepped = hstep(
        zeros,
        stc_arr,
        heat.rhsts,
        jnp.asarray(dt, dtype=dtype),
        nsoil_i,
        heat.ai,
        heat.bi,
        heat.ci,
    )
    stc_out = stepped.stcout
    zz1_arr = jnp.asarray(zz1, dtype=dtype)
    zz1_offset = _store_wrf_real_operation(
        zz1_arr - jnp.asarray(1.0, dtype=dtype),
        rounding_token,
    )
    weighted_stc = _store_wrf_real_operation(
        zz1_offset * stc_out[0],
        rounding_token,
    )
    t1_numerator = _store_wrf_real_operation(
        jnp.asarray(yy, dtype=dtype) + weighted_stc,
        rounding_token,
    )
    t1_out = _store_wrf_real_operation(
        t1_numerator / zz1_arr,
        rounding_token,
    )
    soil_temperature_delta = _store_wrf_real_operation(
        stc_out[0] - t1_out,
        rounding_token,
    )
    ssoil_numerator = _store_wrf_real_operation(
        jnp.asarray(df1, dtype=dtype) * soil_temperature_delta,
        rounding_token,
    )
    ssoil_denominator = _store_wrf_real_operation(
        jnp.asarray(0.5, dtype=dtype) * zsoil_arr[0],
        rounding_token,
    )
    ssoil_out = _store_wrf_real_operation(
        ssoil_numerator / ssoil_denominator,
        rounding_token,
    )
    return ShflxResult(
        ssoil=ssoil_out,
        stc=stc_out,
        sh2o=heat.sh2o,
        t1=t1_out,
        hcpct_fasdas=heat.hcpct_fasdas,
    )


def soil_surface_fluxes(
    etp: Any,
    eta: Any,
    prcp: Any,
    smc: Any,
    smcmax: Any,
    smcwlt: Any,
    smcref: Any,
    smcdry: Any,
    cmc: Any,
    cmcmax: Any,
    nsoil: int,
    dt: Any,
    shdfac: Any,
    sbeta: Any,
    q2: Any,
    t1: Any,
    sfctmp: Any,
    t24: Any,
    th2: Any,
    fdown: Any,
    f1: Any,
    emissi: Any,
    ssoil: Any,
    stc: Any,
    epsca: Any,
    bexp: Any,
    pc: Any,
    rch: Any,
    rr: Any,
    cfactr: Any,
    sh2o: Any,
    slope: Any,
    kdt: Any,
    frzfact: Any,
    psisat: Any,
    zsoil: Any,
    dksat: Any,
    dwsat: Any,
    tbot: Any,
    zbot: Any,
    runoff1: Any,
    runoff2: Any,
    runoff3: Any,
    edir: Any,
    ec: Any,
    et: Any,
    ett: Any,
    nroot: int,
    rtdis: Any,
    quartz: Any,
    fxexp: Any,
    csoil: Any,
    beta: Any,
    drip: Any,
    dew: Any,
    flx1: Any,
    flx3: Any,
    vegtyp: int,
    isurban: int,
    sfhead1rt: Any,
    infxs1rt: Any,
    etpnd1: Any,
    soiltyp: int,
    opt_thcnd: int,
    xsda_qfx: Any,
    qfx_phy: Any,
    xqnorm: Any,
    fasdas: int,
    hcpct_fasdas: Any,
    irrigation_channel: Any | None = None,
    rounding_token: Any | None = None,
    *,
    snowng: Any,
    prcpf: Any,
    sneqv: Any,
    snowh: Any,
    sndens: Any,
    sncovr: Any,
    snow_df1: Any,
    flx2: Any,
    ribb: Any,
) -> SoilSurfaceResult:
    """WRF NOPAC/SNOPAC with one shared evaporation and soil water/heat solve."""

    del eta, ssoil, runoff1, runoff2, runoff3, edir, ec, ett
    del beta, drip, dew, flx1, flx3, qfx_phy
    dtype = jnp.result_type(
        etp,
        prcp,
        smc,
        smcmax,
        smcwlt,
        smcref,
        smcdry,
        cmc,
        cmcmax,
        dt,
        shdfac,
        sbeta,
        q2,
        t1,
        sfctmp,
        t24,
        th2,
        fdown,
        f1,
        emissi,
        stc,
        epsca,
        bexp,
        pc,
        rch,
        rr,
        cfactr,
        sh2o,
        slope,
        kdt,
        frzfact,
        psisat,
        zsoil,
        dksat,
        dwsat,
        tbot,
        zbot,
        et,
        rtdis,
        quartz,
        fxexp,
        csoil,
        sfhead1rt,
        infxs1rt,
        etpnd1,
        xsda_qfx,
        xqnorm,
        hcpct_fasdas,
        0.0 if irrigation_channel is None else irrigation_channel,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    etp = jnp.asarray(etp, dtype=dtype)
    prcp = jnp.asarray(prcp, dtype=dtype)
    smc_out = jnp.asarray(smc, dtype=dtype)
    sh2o_out = jnp.asarray(sh2o, dtype=dtype)
    stc_out = jnp.asarray(stc, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    rtdis_arr = jnp.asarray(rtdis, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    smcref = jnp.asarray(smcref, dtype=dtype)
    smcdry = jnp.asarray(smcdry, dtype=dtype)
    cmc_out = jnp.asarray(cmc, dtype=dtype)
    cmcmax = jnp.asarray(cmcmax, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    shdfac = jnp.asarray(shdfac, dtype=dtype)
    sbeta = jnp.asarray(sbeta, dtype=dtype)
    q2 = jnp.asarray(q2, dtype=dtype)
    t1_out = jnp.asarray(t1, dtype=dtype)
    sfctmp = jnp.asarray(sfctmp, dtype=dtype)
    t24 = jnp.asarray(t24, dtype=dtype)
    th2 = jnp.asarray(th2, dtype=dtype)
    fdown = jnp.asarray(fdown, dtype=dtype)
    f1 = jnp.asarray(f1, dtype=dtype)
    emissi = jnp.asarray(emissi, dtype=dtype)
    epsca = jnp.asarray(epsca, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    pc = jnp.asarray(pc, dtype=dtype)
    rch = jnp.asarray(rch, dtype=dtype)
    rr = jnp.asarray(rr, dtype=dtype)
    cfactr = jnp.asarray(cfactr, dtype=dtype)
    slope = jnp.asarray(slope, dtype=dtype)
    kdt = jnp.asarray(kdt, dtype=dtype)
    frzfact = jnp.asarray(frzfact, dtype=dtype)
    psisat = jnp.asarray(psisat, dtype=dtype)
    dksat = jnp.asarray(dksat, dtype=dtype)
    dwsat = jnp.asarray(dwsat, dtype=dtype)
    tbot = jnp.asarray(tbot, dtype=dtype)
    zbot = jnp.asarray(zbot, dtype=dtype)
    quartz = jnp.asarray(quartz, dtype=dtype)
    fxexp = jnp.asarray(fxexp, dtype=dtype)
    csoil = jnp.asarray(csoil, dtype=dtype)
    xsda_qfx = jnp.asarray(xsda_qfx, dtype=dtype)
    xqnorm = jnp.asarray(xqnorm, dtype=dtype)
    sfhead1rt_out = jnp.asarray(sfhead1rt, dtype=dtype)
    infxs1rt_out = jnp.asarray(infxs1rt, dtype=dtype)
    hcpct_fasdas_out = jnp.asarray(hcpct_fasdas, dtype=dtype)

    has_snow = sneqv > 0.0
    store = lambda value: _store_wrf_real(value, dtype, rounding_token)
    frost_etp = store((jnp.minimum(etp * (1.0 - ribb), 0.0) * sncovr / 0.980
                       + etp * (0.980 - sncovr)) / 0.980)
    etp = jnp.where(has_snow & (etp <= 0.0) & (ribb >= 0.1) & (fdown > 150.0), frost_etp, etp)

    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    dew_out = jnp.asarray(0.0, dtype=dtype)
    qfx_phy_out = jnp.asarray(0.0, dtype=dtype)
    edir1 = jnp.asarray(0.0, dtype=dtype)
    ec1 = jnp.asarray(0.0, dtype=dtype)
    et1 = jnp.zeros((nsoil_i,), dtype=dtype)
    ett1 = jnp.asarray(0.0, dtype=dtype)
    etpnd1_out = jnp.asarray(0.0, dtype=dtype)

    etp_positive = etp > 0.0
    evap = evapo(
        0.0,
        smc_out,
        nsoil_i,
        cmc_out,
        jnp.where(etp_positive, etp1, 0.0),
        dt,
        zsoil_arr,
        sh2o_out,
        smcmax,
        bexp,
        pc,
        smcwlt,
        dksat,
        dwsat,
        smcref,
        shdfac,
        cmcmax,
        smcdry,
        cfactr,
        0.0,
        0.0,
        et1,
        0.0,
        sfctmp,
        q2,
        nroot,
        rtdis_arr,
        fxexp,
        sfhead1rt_out,
        etpnd1_out,
        rounding_token,
    )
    eta1 = evap.eta1
    edir1 = evap.edir
    ec1 = evap.ec
    et1 = evap.et
    ett1 = evap.ett
    sfhead1rt_out = evap.sfhead1rt
    etpnd1_out = evap.etpnd1

    # Snow-covered fractions sublimate; only the exposed fraction evaporates
    # soil/canopy water. Both fractions share the implicit soil solvers below.
    exposed = jnp.where(has_snow, 1.0 - sncovr, 1.0)
    eta1 = store(eta1 * exposed)
    edir1 = store(edir1 * exposed)
    ec1 = store(ec1 * exposed)
    et1 = store(et1 * exposed)
    ett1 = store(ett1 * exposed)
    etpnd1_out = jnp.where(has_snow, etpnd1_out * 1000.0, etpnd1_out)

    if int(fasdas) == 1:
        wetty = jnp.ones((nsoil_i,), dtype=dtype)
        for k in range(nsoil_i):
            qfx_phy_out = qfx_phy_out + et1[k]
            wetty = wetty.at[k].set(jnp.where((smc_out[k] >= smcref) & (xsda_qfx > 0.0), 0.0, wetty[k]))
        qfx_phy_total = edir1 + ec1 + qfx_phy_out
        eall_now = qfx_phy_total
        qfx_phy_out = qfx_phy_total * 1000.0
        has_eall = eall_now != 0.0
        safe_eall = jnp.where(has_eall, eall_now, 1.0)
        efdir = (edir1 / safe_eall) * xsda_qfx * 1.0e-03 * xqnorm
        edir1 = jnp.where(has_eall, edir1 + efdir * wetty[0], edir1)
        efc = (ec1 / safe_eall) * xsda_qfx * 1.0e-03 * xqnorm
        ec1 = jnp.where(has_eall, ec1 + efc, ec1)
        for k in range(nsoil_i):
            eft = (et1[k] / safe_eall) * xsda_qfx * 1.0e-03 * xqnorm
            et1 = et1.at[k].set(jnp.where(has_eall, et1[k] + eft * wetty[k], et1[k]))
    else:
        qfx_phy_out = jnp.asarray(0.0, dtype=dtype)

    dew_out = jnp.where(etp_positive, 0.0, -etp1)
    prcp1_moisture = jnp.where(etp_positive, prcp1, prcp1 + dew_out)

    esnow = store(jnp.where(has_snow & etp_positive, etp * sncovr, 0.0))
    esnow2 = store(jnp.where(etp_positive, esnow * 0.001, etp1) * dt)
    etns = store(eta1 * 1000.0)
    etanrg = store(jnp.where(etp_positive, esnow * 2.83e6 + etns * 2.501e6,
                            etp * ((1.0 - sncovr) * 2.501e6 + sncovr * 2.83e6)))
    snow_flx1 = store(jnp.where(snowng, CPICE * prcp * (t1_out - sfctmp),
                               jnp.where(prcp > 0.0, CPH2O * prcp * (t1_out - sfctmp), 0.0)))
    dtot = store(snowh - 0.5 * zsoil_arr[0])
    denom = store(1.0 + snow_df1 / (dtot * rr * rch))
    t12a = store(((fdown - snow_flx1 - flx2 - emissi * SIGMA * t24) / rch
                  + th2 - sfctmp - etanrg / rch) / rr)
    t12b = store(snow_df1 * stc_out[0] / (dtot * rr * rch))
    t12 = store((sfctmp + t12a + t12b) / denom)
    thaw = t12 > 273.15
    snow_weight = jnp.maximum(0.01, sncovr ** 2.0)
    snow_t1 = store(jnp.where(thaw, 273.15 * snow_weight + t12 * (1.0 - snow_weight), t12))
    snow_ssoil = store(snow_df1 * (snow_t1 - stc_out[0]) / dtot)
    remaining = store(sneqv - esnow2)
    melt_active = has_snow & thaw & (remaining > 1.e-6)
    seh = store(rch * (snow_t1 - th2))
    t14 = store(snow_t1 * snow_t1)
    t14 = store(t14 * t14)
    snow_flx3 = store(jnp.maximum(0.0, fdown - snow_flx1 - flx2
                                 - emissi * SIGMA * t14 - snow_ssoil - seh - etanrg))
    ex = store(snow_flx3 * 0.001 / LSUBF)
    snomlt = store(ex * dt)
    complete_melt = remaining - snomlt < 1.e-6
    ex = jnp.where(complete_melt, store(remaining / dt), ex)
    snow_flx3 = jnp.where(complete_melt, store(ex * 1000.0 * LSUBF), snow_flx3)
    snomlt = jnp.where(complete_melt, remaining, snomlt)
    ex = jnp.where(melt_active, ex, 0.0)
    snomlt = jnp.where(melt_active, snomlt, 0.0)
    snow_flx3 = jnp.where(melt_active, snow_flx3, 0.0)
    snow_water = jnp.where(thaw,
                           jnp.where(melt_active & ~complete_melt, store(remaining - snomlt), 0.0),
                           jnp.maximum(0.0, remaining))
    snow_water = jnp.where(has_snow, snow_water, 0.0)
    prcp1_moisture = jnp.where(has_snow, store(prcpf * 0.001 + ex), prcp1_moisture)
    if irrigation_channel is not None:
        irrigation_channel = jnp.where(has_snow, 0.0, irrigation_channel)
    moisture = smflx(
        smc_out,
        nsoil_i,
        cmc_out,
        dt,
        prcp1_moisture,
        zsoil_arr,
        sh2o_out,
        slope,
        kdt,
        frzfact,
        smcmax,
        bexp,
        smcwlt,
        dksat,
        dwsat,
        shdfac,
        cmcmax,
        0.0,
        0.0,
        0.0,
        edir1,
        ec1,
        et1,
        0.0,
        sfhead1rt_out,
        infxs1rt_out,
        irrigation_channel,
    )
    eta_positive = _store_wrf_real_operation(
        eta1 * 1000.0,
        rounding_token,
    )
    eta_out = jnp.where(etp_positive, eta_positive, etp)
    eta_out = jnp.where(has_snow, store(esnow + etns - 1000.0 * dew_out), eta_out)

    smc_out = moisture.smc
    sh2o_out = moisture.sh2o
    cmc_out = moisture.cmc
    runoff1_out = moisture.runoff1
    runoff2_out = moisture.runoff2
    runoff3_out = moisture.runoff3
    drip_out = moisture.drip
    sfhead1rt_out = moisture.sfhead1rt
    infxs1rt_out = moisture.infxs1rt

    beta_positive = _store_wrf_real_operation(
        eta_out / etp,
        rounding_token,
    )
    beta_out = jnp.where(
        etp <= 0.0,
        jnp.where(etp < 0.0, 1.0, 0.0),
        beta_positive,
    )

    edir_out = _store_wrf_real_operation(edir1 * 1000.0, rounding_token)
    ec_out = _store_wrf_real_operation(ec1 * 1000.0, rounding_token)
    et_out = _store_wrf_real_operation(et1 * 1000.0, rounding_token)
    ett_out = _store_wrf_real_operation(ett1 * 1000.0, rounding_token)

    df1 = _tdfcnd_wrf_real(
            smc_out[0],
            quartz,
            smcmax,
            sh2o_out[0],
            bexp,
            psisat,
            soiltyp,
            int(opt_thcnd),
        rounding_token,
    )
    df1 = _store_wrf_real(
        jnp.where(
            jnp.asarray(vegtyp, dtype=jnp.int32) == int(isurban),
            jnp.asarray(3.24, dtype=dtype),
            df1,
        ),
        dtype,
        rounding_token,
    )
    df1 = _store_wrf_real_operation(
        df1 * jnp.exp(sbeta * shdfac),
        rounding_token,
    )
    df1 = jnp.where(has_snow, snow_df1, df1)

    mul = lambda left, right: _store_wrf_real_operation(
        left * right,
        rounding_token,
    )
    add = lambda left, right: _store_wrf_real_operation(
        left + right,
        rounding_token,
    )
    sub = lambda left, right: _store_wrf_real_operation(
        left - right,
        rounding_token,
    )
    div = lambda left, right: _store_wrf_real_operation(
        left / right,
        rounding_token,
    )
    longwave = mul(mul(emissi, SIGMA), t24)
    yynum = sub(fdown, longwave)
    yy_correction = add(div(yynum, rch), th2)
    yy_correction = sub(yy_correction, sfctmp)
    yy_correction = sub(yy_correction, mul(beta_out, epsca))
    yy = add(sfctmp, div(yy_correction, rr))
    zz1_denominator = mul(mul(mul(-0.5, zsoil_arr[0]), rch), rr)
    zz1 = add(div(df1, zz1_denominator), 1.0)
    snow_yy = store(stc_out[0] - 0.5 * snow_ssoil * zsoil_arr[0] / snow_df1)
    yy = jnp.where(has_snow, snow_yy, yy)
    zz1 = jnp.where(has_snow, 1.0, zz1)

    heat = shflx(
        0.0,
        stc_out,
        smc_out,
        smcmax,
        nsoil_i,
        t1_out,
        dt,
        yy,
        zz1,
        zsoil_arr,
        tbot,
        zbot,
        smcwlt,
        psisat,
        sh2o_out,
        bexp,
        f1,
        df1,
        quartz,
        csoil,
        vegtyp,
        int(isurban),
        soiltyp,
        int(opt_thcnd),
        hcpct_fasdas_out,
        rounding_token,
    )
    ssoil_out = jnp.where(has_snow, snow_ssoil, heat.ssoil)
    stc_out = heat.stc
    sh2o_out = heat.sh2o
    t1_out = jnp.where(has_snow, snow_t1, heat.t1)
    hcpct_fasdas_out = heat.hcpct_fasdas

    flx1_out = jnp.where(has_snow, snow_flx1, CPH2O * prcp * (t1_out - sfctmp))
    flx3_out = snow_flx3
    snowh_new, _ = snowpack(snow_water, dt, snowh, sndens, t1_out, yy)
    snowh_new = jnp.where(snow_water > 0.0, snowh_new, 0.0)
    sncovr = jnp.where(snow_water > 0.0, sncovr, 0.0)

    return SoilSurfaceResult(
        eta=eta_out,
        smc=smc_out,
        cmc=cmc_out,
        t1=t1_out,
        ssoil=ssoil_out,
        stc=stc_out,
        sh2o=sh2o_out,
        beta=beta_out,
        drip=drip_out,
        dew=dew_out,
        flx1=flx1_out,
        flx3=flx3_out,
        runoff1=runoff1_out,
        runoff2=runoff2_out,
        runoff3=runoff3_out,
        edir=edir_out,
        ec=ec_out,
        et=et_out,
        ett=ett_out,
        sfhead1rt=sfhead1rt_out,
        infxs1rt=infxs1rt_out,
        etpnd1=etpnd1_out,
        qfx_phy=qfx_phy_out,
        hcpct_fasdas=hcpct_fasdas_out,
        etp=etp,
        esnow=esnow,
        sneqv=snow_water,
        snowh=snowh_new,
        sncovr=sncovr,
        snomlt=snomlt,
    )


def snksrc(
    tsnsr: Any,
    tavg: Any,
    smc: Any,
    sh2o: Any,
    zsoil: Any,
    nsoil: int,
    smcmax: Any,
    psisat: Any,
    bexp: Any,
    dt: Any,
    k: int,
    qtot: Any,
) -> SnksrcResult:
    """Return WRF ``SNKSRC`` latent heat source and updated liquid water."""

    del tsnsr, nsoil
    dtype = jnp.result_type(tavg, smc, sh2o, zsoil, smcmax, psisat, bexp, dt, qtot, jnp.float32)
    tavg = jnp.asarray(tavg, dtype=dtype)
    smc = jnp.asarray(smc, dtype=dtype)
    sh2o_old = jnp.asarray(sh2o, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    psisat = jnp.asarray(psisat, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    qtot = jnp.asarray(qtot, dtype=dtype)
    layer = int(k)

    if layer == 1:
        dz = -zsoil_arr[0]
    else:
        dz = zsoil_arr[layer - 2] - zsoil_arr[layer - 1]

    free = frh2o(tavg, smc, sh2o_old, smcmax, bexp, psisat)
    dh2o = jnp.asarray(1.0e3, dtype=dtype)
    hlice = jnp.asarray(3.335e5, dtype=dtype)
    xh2o = sh2o_old + qtot * dt / (dh2o * hlice * dz)

    cond_low = (xh2o < sh2o_old) & (xh2o < free)
    xh2o = jnp.where(cond_low, jnp.where(free > sh2o_old, sh2o_old, free), xh2o)
    cond_high = (xh2o > sh2o_old) & (xh2o > free)
    xh2o = jnp.where(cond_high, jnp.where(free < sh2o_old, sh2o_old, free), xh2o)

    xh2o = jnp.maximum(xh2o, 0.0)
    xh2o = jnp.minimum(xh2o, smc)
    tsnsr_out = -dh2o * hlice * dz * (xh2o - sh2o_old) / dt
    return SnksrcResult(tsnsr=tsnsr_out, sh2o=xh2o)


def evapo(
    eta1: Any,
    smc: Any,
    nsoil: int,
    cmc: Any,
    etp1: Any,
    dt: Any,
    zsoil: Any,
    sh2o: Any,
    smcmax: Any,
    bexp: Any,
    pc: Any,
    smcwlt: Any,
    dksat: Any,
    dwsat: Any,
    smcref: Any,
    shdfac: Any,
    cmcmax: Any,
    smcdry: Any,
    cfactr: Any,
    edir: Any,
    ec: Any,
    et: Any,
    ett: Any,
    sfctmp: Any,
    q2: Any,
    nroot: int,
    rtdis: Any,
    fxexp: Any,
    sfhead1rt: Any,
    etpnd1: Any,
    rounding_token: Any | None = None,
) -> EvapoResult:
    """Return WRF ``EVAPO`` evapotranspiration partitioning."""

    del eta1, edir, ec, ett
    dtype = jnp.result_type(
        smc,
        cmc,
        etp1,
        dt,
        zsoil,
        sh2o,
        smcmax,
        bexp,
        pc,
        smcwlt,
        dksat,
        dwsat,
        smcref,
        shdfac,
        cmcmax,
        smcdry,
        cfactr,
        et,
        sfctmp,
        q2,
        rtdis,
        fxexp,
        sfhead1rt,
        etpnd1,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    smc_arr = jnp.asarray(smc, dtype=dtype)
    sh2o_arr = jnp.asarray(sh2o, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    rtdis_arr = jnp.asarray(rtdis, dtype=dtype)
    cmc = jnp.asarray(cmc, dtype=dtype)
    etp1 = jnp.asarray(etp1, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    pc = jnp.asarray(pc, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    dksat = jnp.asarray(dksat, dtype=dtype)
    dwsat = jnp.asarray(dwsat, dtype=dtype)
    smcref = jnp.asarray(smcref, dtype=dtype)
    shdfac = jnp.asarray(shdfac, dtype=dtype)
    cmcmax = jnp.asarray(cmcmax, dtype=dtype)
    smcdry = jnp.asarray(smcdry, dtype=dtype)
    cfactr = jnp.asarray(cfactr, dtype=dtype)
    sfctmp = jnp.asarray(sfctmp, dtype=dtype)
    q2 = jnp.asarray(q2, dtype=dtype)
    fxexp = jnp.asarray(fxexp, dtype=dtype)
    sfhead1rt_out = jnp.asarray(sfhead1rt, dtype=dtype)
    etpnd1_out = jnp.asarray(etpnd1, dtype=dtype)

    edir_out = jnp.asarray(0.0, dtype=dtype)
    ec_out = jnp.asarray(0.0, dtype=dtype)
    ett_out = jnp.asarray(0.0, dtype=dtype)
    et_out = jnp.zeros((nsoil_i,), dtype=dtype)

    etp_active = etp1 > 0.0
    edir_candidate = devap(etp1, smc_arr[0], zsoil_arr[0], shdfac, smcmax, bexp, dksat, dwsat, smcdry, smcref, smcwlt, fxexp)
    edir_out = jnp.where(etp_active & (shdfac < 1.0), edir_candidate, 0.0)

    et_candidate = transp(
        nsoil_i,
        etp1,
        sh2o_arr,
        cmc,
        zsoil_arr,
        shdfac,
        smcwlt,
        cmcmax,
        pc,
        cfactr,
        smcref,
        sfctmp,
        q2,
        nroot,
        rtdis_arr,
        rounding_token,
    )
    et_out = jnp.where(etp_active & (shdfac > 0.0), et_candidate, et_out)
    for k in range(nsoil_i):
        ett_out = _store_wrf_real_operation(
            ett_out + et_out[k],
            rounding_token,
        )

    ec_water = jnp.where(cmc > 0.0, cmc, cmcmax)
    ec_candidate = shdfac * ((ec_water / cmcmax) ** cfactr) * etp1
    ec_out = jnp.where(etp_active & (shdfac > 0.0) & (cmc > 0.0), ec_candidate, 0.0)
    ec_out = jnp.minimum(cmc / dt, ec_out)

    eta1_out = _store_wrf_real_operation(
        _store_wrf_real_operation(edir_out + ett_out, rounding_token) + ec_out,
        rounding_token,
    )
    return EvapoResult(eta1=eta1_out, edir=edir_out, ec=ec_out, et=et_out, ett=ett_out, sfhead1rt=sfhead1rt_out, etpnd1=etpnd1_out)


def fac2mit(smcmax: Any) -> Any:
    """Return WRF ``FAC2MIT`` limiting factor for selected soil porosities."""

    dtype = jnp.result_type(smcmax, jnp.float32)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    flimit = jnp.asarray(0.90, dtype=dtype)
    flimit = jnp.where(smcmax == jnp.asarray(0.395, dtype=dtype), 0.59, flimit)
    flimit = jnp.where((smcmax == jnp.asarray(0.434, dtype=dtype)) | (smcmax == jnp.asarray(0.404, dtype=dtype)), 0.85, flimit)
    flimit = jnp.where((smcmax == jnp.asarray(0.465, dtype=dtype)) | (smcmax == jnp.asarray(0.406, dtype=dtype)), 0.86, flimit)
    flimit = jnp.where((smcmax == jnp.asarray(0.476, dtype=dtype)) | (smcmax == jnp.asarray(0.439, dtype=dtype)), 0.74, flimit)
    flimit = jnp.where((smcmax == jnp.asarray(0.200, dtype=dtype)) | (smcmax == jnp.asarray(0.464, dtype=dtype)), 0.80, flimit)
    return flimit


def _frh2o_impl(tkelv: Any, smc: Any, sh2o: Any, smcmax: Any, bexp: Any, psis: Any) -> Any:

    dtype = jnp.result_type(tkelv, smc, sh2o, smcmax, bexp, psis, jnp.float32)
    tkelv = jnp.asarray(tkelv, dtype=dtype)
    smc = jnp.asarray(smc, dtype=dtype)
    sh2o = jnp.asarray(sh2o, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    psis = jnp.asarray(psis, dtype=dtype)
    ck = jnp.asarray(8.0, dtype=dtype)
    bx = jnp.minimum(bexp, 5.5)
    t0 = jnp.asarray(273.15, dtype=dtype)

    warm = tkelv > (t0 - 1.0e-3)
    swl = smc - sh2o
    swl = jnp.minimum(swl, smc - 0.02)
    swl = jnp.maximum(swl, 0.0)
    converged = jnp.asarray(False)
    for _ in range(10):
        df = jnp.log((psis * 9.81 / 3.335e5) * ((1.0 + ck * swl) ** 2.0) * (smcmax / (smc - swl)) ** bx) - jnp.log(
            -(tkelv - t0) / tkelv
        )
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        swlk = jnp.minimum(swlk, smc - 0.02)
        swlk = jnp.maximum(swlk, 0.0)
        dswl = jnp.abs(swlk - swl)
        update = ~converged
        swl = jnp.where(update, swlk, swl)
        converged = converged | (dswl <= 0.005)

    iter_free = smc - swl
    fk = (((3.335e5 / (9.81 * (-psis))) * ((tkelv - t0) / tkelv)) ** (-1.0 / bx)) * smcmax
    fk = jnp.maximum(fk, 0.02)
    fallback_free = jnp.minimum(fk, smc)
    cold_free = jnp.where(converged, iter_free, fallback_free)
    return jnp.where(warm, smc, cold_free)


@jax.custom_jvp
def frh2o(tkelv: Any, smc: Any, sh2o: Any, smcmax: Any, bexp: Any, psis: Any) -> Any:
    """Return WRF ``FRH2O`` without differentiating absent frozen soil."""

    return _frh2o_impl(tkelv, smc, sh2o, smcmax, bexp, psis)


@frh2o.defjvp
def _frh2o_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
    value = frh2o(*primals)

    def active_formula(tkelv: Any, smc: Any, sh2o: Any, smcmax: Any, bexp: Any, psis: Any) -> Any:
        warm = tkelv > (273.15 - 1.0e-3)
        temperature = jnp.where(warm, jnp.asarray(272.15, dtype=jnp.asarray(tkelv).dtype), tkelv)
        cold = _frh2o_impl(temperature, smc, sh2o, smcmax, bexp, psis)
        return jnp.where(warm, smc, cold)

    _, derivative = jax.jvp(active_formula, primals, tangents)
    return value, derivative


def devap(
    etp1: Any,
    smc: Any,
    zsoil: Any,
    shdfac: Any,
    smcmax: Any,
    bexp: Any,
    dksat: Any,
    dwsat: Any,
    smcdry: Any,
    smcref: Any,
    smcwlt: Any,
    fxexp: Any,
) -> Any:
    """Return WRF ``DEVAP`` direct soil evaporation."""

    del zsoil, bexp, dksat, dwsat, smcref, smcwlt
    dtype = jnp.result_type(etp1, smc, shdfac, smcmax, smcdry, fxexp, jnp.float32)
    etp1 = jnp.asarray(etp1, dtype=dtype)
    smc = jnp.asarray(smc, dtype=dtype)
    shdfac = jnp.asarray(shdfac, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    smcdry = jnp.asarray(smcdry, dtype=dtype)
    fxexp = jnp.asarray(fxexp, dtype=dtype)

    sratio = (smc - smcdry) / (smcmax - smcdry)
    fx = jnp.where(sratio > 0.0, jnp.clip(sratio**fxexp, 0.0, 1.0), 0.0)
    return fx * (1.0 - shdfac) * etp1


def transp(
    nsoil: int,
    etp1: Any,
    smc: Any,
    cmc: Any,
    zsoil: Any,
    shdfac: Any,
    smcwlt: Any,
    cmcmax: Any,
    pc: Any,
    cfactr: Any,
    smcref: Any,
    sfctmp: Any,
    q2: Any,
    nroot: int,
    rtdis: Any,
    rounding_token: Any | None = None,
) -> Any:
    """Return WRF ``TRANSP`` layer transpiration array ``ET``."""

    del zsoil, sfctmp, q2
    dtype = jnp.result_type(etp1, smc, cmc, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, rtdis, jnp.float32)
    smc_arr = jnp.asarray(smc, dtype=dtype)
    rtdis_arr = jnp.asarray(rtdis, dtype=dtype)
    etp1 = jnp.asarray(etp1, dtype=dtype)
    cmc = jnp.asarray(cmc, dtype=dtype)
    shdfac = jnp.asarray(shdfac, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    cmcmax = jnp.asarray(cmcmax, dtype=dtype)
    pc = jnp.asarray(pc, dtype=dtype)
    cfactr = jnp.asarray(cfactr, dtype=dtype)
    smcref = jnp.asarray(smcref, dtype=dtype)
    nsoil_i = int(nsoil)
    nroot_i = jnp.asarray(nroot, dtype=jnp.int32)

    et = jnp.zeros((nsoil_i,), dtype=dtype)
    add = lambda left, right: _store_wrf_real_operation(
        left + right,
        rounding_token,
    )
    sub = lambda left, right: _store_wrf_real_operation(
        left - right,
        rounding_token,
    )
    mul = lambda left, right: _store_wrf_real_operation(
        left * right,
        rounding_token,
    )
    div = lambda left, right: _store_wrf_real_operation(
        left / right,
        rounding_token,
    )
    uncovered_etp = mul(mul(shdfac, pc), etp1)
    covered_fraction = sub(
        1.0,
        _store_wrf_real_operation(
            (jnp.where(cmc != 0.0, cmc, cmcmax) / cmcmax) ** cfactr,
            rounding_token,
        ),
    )
    covered_etp = mul(uncovered_etp, covered_fraction)
    etp1a = jnp.where(cmc != 0.0, covered_etp, uncovered_etp)

    gx_values = []
    sgx = jnp.asarray(0.0, dtype=dtype)
    for idx in range(nsoil_i):
        gx = div(
            sub(smc_arr[idx], smcwlt),
            sub(smcref, smcwlt),
        )
        gx = _store_wrf_real_operation(
            jnp.maximum(jnp.minimum(gx, 1.0), 0.0),
            rounding_token,
        )
        gx = jnp.where(idx < nroot_i, gx, 0.0)
        gx_values.append(gx)
        sgx = add(sgx, gx)
    root_count = jnp.asarray(jnp.maximum(nroot_i, 1), dtype=dtype)
    sgx = div(sgx, root_count)

    adjusted = []
    denom = jnp.asarray(0.0, dtype=dtype)
    for idx, gx in enumerate(gx_values):
        rtx = sub(add(rtdis_arr[idx], gx), sgx)
        value = mul(gx, jnp.maximum(rtx, 0.0))
        value = jnp.where(idx < nroot_i, value, 0.0)
        adjusted.append(value)
        denom = add(denom, value)
    denom = jnp.where(denom <= 0.0, 1.0, denom)

    for idx, value in enumerate(adjusted):
        et_value = div(mul(etp1a, value), denom)
        et = et.at[idx].set(jnp.where(idx < nroot_i, et_value, 0.0))
    return et


def tbnd(tu: Any, tb: Any, zsoil: Any, zbot: Any, k: int, nsoil: int) -> Any:
    """Return WRF ``TBND`` boundary temperature interpolation for layer ``k``."""

    dtype = jnp.result_type(tu, tb, zsoil, zbot, jnp.float32)
    tu = jnp.asarray(tu, dtype=dtype)
    tb = jnp.asarray(tb, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    zbot = jnp.asarray(zbot, dtype=dtype)
    k_i = int(k)
    nsoil_i = int(nsoil)
    zup = jnp.asarray(0.0, dtype=dtype) if k_i == 1 else zsoil_arr[k_i - 2]
    zb = 2.0 * zbot - zsoil_arr[k_i - 1] if k_i == nsoil_i else zsoil_arr[k_i]
    return tu + (tb - tu) * (zup - zsoil_arr[k_i - 1]) / (zup - zb)


def tmpavg(tup: Any, tm: Any, tdn: Any, zsoil: Any, nsoil: int, k: int,
           rounding_token: Any | None = None) -> Any:
    """Return WRF ``TMPAVG`` average layer temperature across freeze/thaw segments."""

    del nsoil
    dtype = jnp.result_type(tup, tm, tdn, zsoil, jnp.float32)
    tup = jnp.asarray(tup, dtype=dtype)
    tm = jnp.asarray(tm, dtype=dtype)
    tdn = jnp.asarray(tdn, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    k_i = int(k)
    t0 = jnp.asarray(273.15, dtype=dtype)
    store = lambda value: _store_wrf_real(value, dtype, rounding_token)
    add = lambda left, right: store(left + right)
    sub = lambda left, right: store(left - right)
    mul = lambda left, right: store(left * right)
    div = lambda left, right: store(left / right)
    dz = -zsoil_arr[0] if k_i == 1 else zsoil_arr[k_i - 2] - zsoil_arr[k_i - 1]
    dzh = mul(dz, 0.5)

    # A one-ULP change in this average changes the subsequent frozen-water solve.
    def area_average(*areas):
        total = areas[0]
        for area in areas[1:]:
            total = add(total, area)
        return div(mul(0.5, total), dz)

    avg_all_below = div(add(add(tup, mul(2.0, tm)), tdn), 4.0)
    bottom_crossing = (tm < t0) != (tdn < t0)
    x0 = div(mul(sub(t0, tm), dzh), jnp.where(bottom_crossing, sub(tdn, tm), 1.0))
    avg_up_mid_below = area_average(
        mul(tup, dzh), mul(tm, add(dzh, x0)), mul(t0, sub(mul(2.0, dzh), x0)))

    top_crossing = (tup < t0) != (tm < t0)
    xup = div(mul(sub(t0, tup), dzh), jnp.where(top_crossing, sub(tm, tup), 1.0))
    xdn = sub(dzh, x0)
    avg_up_down_below = area_average(
        mul(tup, xup), mul(t0, sub(sub(mul(2.0, dz), xup), xdn)), mul(tdn, xdn))

    avg_up_below = area_average(mul(tup, xup), mul(t0, sub(mul(2.0, dz), xup)))

    xup_warm = sub(dzh, xup)
    avg_mid_down_below = area_average(
        mul(t0, sub(dz, xup_warm)), mul(tm, add(dzh, xup_warm)), mul(tdn, dzh))

    xdn_warm = x0
    avg_mid_below = area_average(
        mul(t0, sub(sub(mul(2.0, dz), xup_warm), xdn_warm)), mul(tm, add(xup_warm, xdn_warm)))

    xdn_cold = xdn
    avg_down_below = div(add(mul(t0, sub(dz, xdn_cold)),
                            mul(mul(0.5, add(t0, tdn)), xdn_cold)), dz)
    avg_all_above = avg_all_below

    tup_cold = tup < t0
    tm_cold = tm < t0
    tdn_cold = tdn < t0
    cold_up_branch = jnp.where(
        tm_cold,
        jnp.where(tdn_cold, avg_all_below, avg_up_mid_below),
        jnp.where(tdn_cold, avg_up_down_below, avg_up_below),
    )
    warm_up_branch = jnp.where(
        tm_cold,
        jnp.where(tdn_cold, avg_mid_down_below, avg_mid_below),
        jnp.where(tdn_cold, avg_down_below, avg_all_above),
    )
    return jnp.where(tup_cold, cold_up_branch, warm_up_branch)


def canres(
    solar: Any,
    ch: Any,
    sfctmp: Any,
    q2: Any,
    sfcprs: Any,
    smc: Any,
    zsoil: Any,
    nsoil: int,
    smcwlt: Any,
    smcref: Any,
    rsmin: Any,
    nroot: int,
    q2sat: Any,
    dqsdt2: Any,
    topt: Any,
    rsmax: Any,
    rgl: Any,
    hs: Any,
    xlai: Any,
    emissi: Any,
    rounding_token: Any | None = None,
) -> CanresResult:
    """Return WRF ``CANRES`` canopy resistance and plant coefficient terms."""

    dtype = jnp.result_type(
        solar,
        ch,
        sfctmp,
        q2,
        sfcprs,
        smc,
        zsoil,
        smcwlt,
        smcref,
        rsmin,
        q2sat,
        dqsdt2,
        topt,
        rsmax,
        rgl,
        hs,
        xlai,
        emissi,
        jnp.float32,
    )
    solar = jnp.asarray(solar, dtype=dtype)
    ch = jnp.asarray(ch, dtype=dtype)
    sfctmp = jnp.asarray(sfctmp, dtype=dtype)
    q2 = jnp.asarray(q2, dtype=dtype)
    sfcprs = jnp.asarray(sfcprs, dtype=dtype)
    smc_arr = jnp.asarray(smc, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)
    smcwlt = jnp.asarray(smcwlt, dtype=dtype)
    smcref = jnp.asarray(smcref, dtype=dtype)
    rsmin = jnp.asarray(rsmin, dtype=dtype)
    q2sat = jnp.asarray(q2sat, dtype=dtype)
    dqsdt2 = jnp.asarray(dqsdt2, dtype=dtype)
    topt = jnp.asarray(topt, dtype=dtype)
    rsmax = jnp.asarray(rsmax, dtype=dtype)
    rgl = jnp.asarray(rgl, dtype=dtype)
    hs = jnp.asarray(hs, dtype=dtype)
    xlai = jnp.asarray(xlai, dtype=dtype)
    emissi = jnp.asarray(emissi, dtype=dtype)

    nsoil_i = int(nsoil)
    nroot_i = jnp.asarray(nroot, dtype=jnp.int32)
    ff = 0.55 * 2.0 * solar / (rgl * xlai)
    rcs = (ff + rsmin / rsmax) / (1.0 + ff)
    rcs = jnp.maximum(rcs, 0.0001)
    rct = 1.0 - 0.0016 * ((topt - sfctmp) ** 2.0)
    rct = jnp.maximum(rct, 0.0001)
    rcq = 1.0 / (1.0 + hs * (q2sat - q2))
    rcq = jnp.maximum(rcq, 0.01)

    add = lambda left, right: _store_wrf_real_operation(
        left + right,
        rounding_token,
    )
    sub = lambda left, right: _store_wrf_real_operation(
        left - right,
        rounding_token,
    )
    mul = lambda left, right: _store_wrf_real_operation(
        left * right,
        rounding_token,
    )
    div = lambda left, right: _store_wrf_real_operation(
        left / right,
        rounding_token,
    )

    gx = div(sub(smc_arr[0], smcwlt), sub(smcref, smcwlt))
    gx = jnp.clip(gx, 0.0, 1.0)
    root_index = jnp.clip(nroot_i - 1, 0, nsoil_i - 1)
    root_depth = jnp.take(zsoil_arr, root_index, axis=0)
    rcsoil = mul(
        div(zsoil_arr[0], root_depth),
        jnp.where(0 < nroot_i, gx, 0.0),
    )
    for idx in range(1, nsoil_i):
        gx = div(sub(smc_arr[idx], smcwlt), sub(smcref, smcwlt))
        gx = jnp.clip(gx, 0.0, 1.0)
        part = mul(
            div(sub(zsoil_arr[idx], zsoil_arr[idx - 1]), root_depth),
            gx,
        )
        rcsoil = add(
            rcsoil,
            jnp.where(
            idx < nroot_i,
                part,
                0.0,
            ),
        )
    rcsoil = jnp.maximum(rcsoil, 0.0001)
    rc_denominator = mul(xlai, rcs)
    rc_denominator = mul(rc_denominator, rct)
    rc_denominator = mul(rc_denominator, rcq)
    rc_denominator = mul(rc_denominator, rcsoil)
    rc = div(rsmin, rc_denominator)
    radiative_factor = mul(mul(mul(4.0, emissi), SIGMA), RD)
    radiative_factor = div(radiative_factor, CP_MODEL)
    rr = mul(
        radiative_factor,
        _store_wrf_real_operation(sfctmp**4.0, rounding_token),
    )
    rr = div(rr, mul(sfcprs, ch))
    rr = add(rr, 1.0)
    delta = mul(div(2.501000e6, CP_MODEL), dqsdt2)
    pc_numerator = add(rr, delta)
    pc_factor = add(1.0, mul(rc, ch))
    pc_denominator = add(mul(rr, pc_factor), delta)
    pc = div(pc_numerator, pc_denominator)
    return CanresResult(pc=pc, rc=rc, rcq=rcq, rcs=rcs, rcsoil=rcsoil, rct=rct)


def penman(
    sfctmp: Any,
    sfcprs: Any,
    ch: Any,
    t2v: Any,
    th2: Any,
    prcp: Any,
    fdown: Any,
    ssoil: Any,
    q2: Any,
    q2sat: Any,
    snowng: bool,
    frzgra: bool,
    dqsdt2: Any,
    emissi_in: Any,
    sneqv: Any,
    t1: Any,
    sncovr: Any,
    aoasis: Any,
    albedo: Any,
    soldn: Any,
    fvb: Any,
    gama: Any,
    stc1: Any,
    ua_phys: bool,
    rounding_token: Any | None = None,
) -> PenmanResult:
    """Return WRF ``PENMAN`` potential evaporation and helper flux terms."""

    del t1
    dtype = jnp.result_type(
        sfctmp,
        sfcprs,
        ch,
        t2v,
        th2,
        prcp,
        fdown,
        ssoil,
        q2,
        q2sat,
        dqsdt2,
        emissi_in,
        sneqv,
        sncovr,
        aoasis,
        albedo,
        soldn,
        fvb,
        gama,
        stc1,
        jnp.float32,
    )
    sfctmp = jnp.asarray(sfctmp, dtype=dtype)
    sfcprs = jnp.asarray(sfcprs, dtype=dtype)
    ch = jnp.asarray(ch, dtype=dtype)
    t2v = jnp.asarray(t2v, dtype=dtype)
    th2 = jnp.asarray(th2, dtype=dtype)
    prcp = jnp.asarray(prcp, dtype=dtype)
    fdown = jnp.asarray(fdown, dtype=dtype)
    ssoil = jnp.asarray(ssoil, dtype=dtype)
    q2 = jnp.asarray(q2, dtype=dtype)
    q2sat = jnp.asarray(q2sat, dtype=dtype)
    dqsdt2 = jnp.asarray(dqsdt2, dtype=dtype)
    emissi = jnp.asarray(emissi_in, dtype=dtype)
    sneqv = jnp.asarray(sneqv, dtype=dtype)
    sncovr = jnp.asarray(sncovr, dtype=dtype)
    aoasis = jnp.asarray(aoasis, dtype=dtype)
    albedo = jnp.asarray(albedo, dtype=dtype)
    soldn = jnp.asarray(soldn, dtype=dtype)
    fvb = jnp.asarray(fvb, dtype=dtype)
    gama = jnp.asarray(gama, dtype=dtype)
    stc1 = jnp.asarray(stc1, dtype=dtype)

    elcp = jnp.asarray(2.4888e3, dtype=dtype)
    lsubc = jnp.asarray(2.501000e6, dtype=dtype)
    cp_local = jnp.asarray(1004.6, dtype=dtype)
    lsubs = jnp.asarray(2.83e6, dtype=dtype)
    algdsn = jnp.asarray(0.5, dtype=dtype)
    alvgsn = jnp.asarray(0.13, dtype=dtype)

    elcp1 = _store_wrf_real(
        (1.0 - sncovr) * elcp + sncovr * elcp * lsubs / lsubc,
        dtype,
        rounding_token,
    )
    lvs = _store_wrf_real(
        (1.0 - sncovr) * lsubc + sncovr * lsubs,
        dtype,
        rounding_token,
    )
    flx2 = jnp.asarray(0.0, dtype=dtype)
    delta = _store_wrf_real(elcp1 * dqsdt2, dtype, rounding_token)
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (RD * t2v)
    rch = rho * cp_local * ch

    rr = jnp.where(snowng, rr + CPICE * prcp / rch,
                   jnp.where(prcp > 0.0, rr + CPH2O * prcp / rch, rr))

    add = lambda left, right: _store_wrf_real_operation(
        left + right,
        rounding_token,
    )
    sub = lambda left, right: _store_wrf_real_operation(
        left - right,
        rounding_token,
    )
    mul = lambda left, right: _store_wrf_real_operation(
        left * right,
        rounding_token,
    )
    div = lambda left, right: _store_wrf_real_operation(
        left / right,
        rounding_token,
    )
    emitted_lw = mul(mul(emissi, SIGMA), t24)
    fnet = sub(sub(fdown, emitted_lw), ssoil)
    flx4 = jnp.asarray(0.0, dtype=dtype)
    fnetn = jnp.asarray(0.0, dtype=dtype)
    etpn = jnp.asarray(0.0, dtype=dtype)
    if bool(ua_phys):
        snow_absorbs = (sneqv > 0.0) & (fnet > 0.0) & (soldn > 0.0)
        totabs = (1.0 - albedo) * soldn * fvb
        ucabs = jnp.minimum(totabs, ((1.0 - algdsn) * (1.0 - alvgsn) * soldn * gama) * fvb)
        flx4_candidate = jnp.minimum(totabs - ucabs, jnp.minimum(250.0, 0.5 * (1.0 - albedo) * soldn))
        flx4 = jnp.where(snow_absorbs, flx4_candidate, 0.0)
        signck = (stc1 - 273.15) * (sfctmp - 273.15)
        use_flx4 = (flx4 > 0.0) & ((signck <= 0.0) | (stc1 < 273.15))
        fnetn_when_enough = fnet - flx4
        flx4 = jnp.where(use_flx4, jnp.where(fnet >= flx4, flx4, fnet), 0.0)
        fnetn = jnp.where(use_flx4, jnp.where(fnet >= flx4, fnetn_when_enough, 0.0), 0.0)

    flx2 = jnp.where(frzgra, -LSUBF * prcp, 0.0)
    fnet = jnp.where(frzgra, fnet - flx2, fnet)
    if bool(ua_phys):
        fnetn = jnp.where(frzgra, fnetn - flx2, fnetn)

    rad = sub(add(div(fnet, rch), th2), sfctmp)
    vapor_term = mul(elcp1, sub(q2sat, q2))
    epsca = div(
        add(mul(vapor_term, rr), mul(rad, delta)),
        add(delta, rr),
    )
    epsca_scaled = mul(epsca, aoasis)
    epsca = jnp.where(epsca > 0.0, epsca_scaled, epsca)
    etp = div(mul(epsca, rch), lvs)

    if bool(ua_phys):
        radn = fnetn / rch + th2 - sfctmp
        epscan = (vapor_term * rr + radn * delta) / (delta + rr)
        etpn = epscan * rch / lvs

    return PenmanResult(epsca=epsca, etp=etp, flx2=flx2, rch=rch, rr=rr, t24=t24, flx4=flx4, etpn=etpn)


def tdfcnd(
    smc: Any,
    qz: Any,
    smcmax: Any,
    sh2o: Any,
    bexp: Any,
    psisat: Any,
    soiltyp: int,
    opt_thcnd: int,
) -> Any:
    """Return WRF ``TDFCND`` soil thermal diffusivity/conductivity."""

    dtype = jnp.result_type(smc, qz, smcmax, sh2o, bexp, psisat, jnp.float32)
    smc = jnp.asarray(smc, dtype=dtype)
    qz = jnp.asarray(qz, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    sh2o = jnp.asarray(sh2o, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    psisat = jnp.asarray(psisat, dtype=dtype)

    use_peters_lidard = int(opt_thcnd) == 1
    if int(opt_thcnd) == 2:
        soiltyp_arr = jnp.asarray(soiltyp, dtype=jnp.int32)
        use_peters_lidard = (soiltyp_arr != 3) & (soiltyp_arr != 4)
    if int(opt_thcnd) == 1 or int(opt_thcnd) == 2:
        satratio = smc / smcmax
        thkice = jnp.asarray(2.2, dtype=dtype)
        thkw = jnp.asarray(0.57, dtype=dtype)
        thko = jnp.asarray(2.0, dtype=dtype)
        thkqtz = jnp.asarray(7.7, dtype=dtype)
        thks = thkqtz**qz * thko ** (1.0 - qz)
        xunfroz = sh2o / smc
        xu = xunfroz * smcmax
        thksat = thks ** (1.0 - smcmax) * thkice ** (smcmax - xu) * thkw**xu
        gammd = (1.0 - smcmax) * 2700.0
        thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
        akei = satratio
        akel = jnp.where(satratio > 0.1, jnp.log10(satratio) + 1.0, 0.0)
        ake = ((smc - sh2o) * akei + sh2o * akel) / smc
        peters_lidard = ake * (thksat - thkdry) + thkdry
        if int(opt_thcnd) == 1:
            return peters_lidard
    psif = psisat * 100.0 * (smcmax / smc) ** bexp
    pf = jnp.log10(jnp.abs(psif))
    other = jnp.where(pf <= 5.1, 420.0 * jnp.exp(-(pf + 2.7)), 0.1744)
    if int(opt_thcnd) not in (1, 2):
        return other
    return jnp.where(use_peters_lidard, peters_lidard, other)


def _tdfcnd_wrf_real(
    smc: Any,
    qz: Any,
    smcmax: Any,
    sh2o: Any,
    bexp: Any,
    psisat: Any,
    soiltyp: int,
    opt_thcnd: int,
    rounding_token: Any,
) -> Any:
    """Evaluate the initial SFLX TDFCND call in WRF REAL order."""

    dtype = jnp.result_type(smc, qz, smcmax, sh2o, bexp, psisat, jnp.float32)
    smc = jnp.asarray(smc, dtype=dtype)
    qz = jnp.asarray(qz, dtype=dtype)
    smcmax = jnp.asarray(smcmax, dtype=dtype)
    sh2o = jnp.asarray(sh2o, dtype=dtype)
    bexp = jnp.asarray(bexp, dtype=dtype)
    psisat = jnp.asarray(psisat, dtype=dtype)

    use_peters_lidard = int(opt_thcnd) == 1
    if int(opt_thcnd) == 2:
        soiltyp_arr = jnp.asarray(soiltyp, dtype=jnp.int32)
        use_peters_lidard = (soiltyp_arr != 3) & (soiltyp_arr != 4)

    if int(opt_thcnd) in (1, 2):
        satratio = _store_wrf_real(smc / smcmax, dtype, rounding_token)
        thkice = jnp.asarray(2.2, dtype=dtype)
        thkw = jnp.asarray(0.57, dtype=dtype)
        thko = jnp.asarray(2.0, dtype=dtype)
        thkqtz = jnp.asarray(7.7, dtype=dtype)

        other_fraction = _store_wrf_real(1.0 - qz, dtype, rounding_token)
        quartz_term = _store_wrf_real(thkqtz**qz, dtype, rounding_token)
        other_term = _store_wrf_real(
            thko**other_fraction,
            dtype,
            rounding_token,
        )
        thks = _store_wrf_real(
            quartz_term * other_term,
            dtype,
            rounding_token,
        )

        xunfroz = _store_wrf_real(sh2o / smc, dtype, rounding_token)
        xu = _store_wrf_real(xunfroz * smcmax, dtype, rounding_token)
        solid_fraction = _store_wrf_real(
            1.0 - smcmax,
            dtype,
            rounding_token,
        )
        ice_fraction = _store_wrf_real(
            smcmax - xu,
            dtype,
            rounding_token,
        )
        thksat = _store_wrf_real(
            thks**solid_fraction,
            dtype,
            rounding_token,
        )
        thksat = _store_wrf_real(
            thksat * _store_wrf_real(
                thkice**ice_fraction,
                dtype,
                rounding_token,
            ),
            dtype,
            rounding_token,
        )
        thksat = _store_wrf_real(
            thksat * _store_wrf_real(thkw**xu, dtype, rounding_token),
            dtype,
            rounding_token,
        )

        gammd = _store_wrf_real(
            solid_fraction * 2700.0,
            dtype,
            rounding_token,
        )
        dry_numerator = _store_wrf_real(
            0.135 * gammd,
            dtype,
            rounding_token,
        )
        dry_numerator = _store_wrf_real(
            dry_numerator + 64.7,
            dtype,
            rounding_token,
        )
        dry_denominator = _store_wrf_real(
            0.947 * gammd,
            dtype,
            rounding_token,
        )
        dry_denominator = _store_wrf_real(
            2700.0 - dry_denominator,
            dtype,
            rounding_token,
        )
        thkdry = _store_wrf_real(
            dry_numerator / dry_denominator,
            dtype,
            rounding_token,
        )

        akel = _store_wrf_real(
            wrf_glibc_log10f_positive_normal(satratio) + 1.0,
            dtype,
            rounding_token,
        )
        akel = jnp.where(satratio > 0.1, akel, 0.0)
        frozen_water = _store_wrf_real(smc - sh2o, dtype, rounding_token)
        frozen_term = _store_wrf_real(
            frozen_water * satratio,
            dtype,
            rounding_token,
        )
        liquid_term = _store_wrf_real(
            sh2o * akel,
            dtype,
            rounding_token,
        )
        ake = _store_wrf_real(
            frozen_term + liquid_term,
            dtype,
            rounding_token,
        )
        ake = _store_wrf_real(ake / smc, dtype, rounding_token)
        conductivity_range = _store_wrf_real(
            thksat - thkdry,
            dtype,
            rounding_token,
        )
        peters_lidard = _store_wrf_real(
            ake * conductivity_range,
            dtype,
            rounding_token,
        )
        peters_lidard = _store_wrf_real(
            peters_lidard + thkdry,
            dtype,
            rounding_token,
        )
        if int(opt_thcnd) == 1:
            return peters_lidard

    psif = _store_wrf_real(psisat * 100.0, dtype, rounding_token)
    moisture_ratio = _store_wrf_real(smcmax / smc, dtype, rounding_token)
    moisture_power = _store_wrf_real(
        moisture_ratio**bexp,
        dtype,
        rounding_token,
    )
    psif = _store_wrf_real(psif * moisture_power, dtype, rounding_token)
    pf = _store_wrf_real(
        jnp.log10(jnp.abs(psif)),
        dtype,
        rounding_token,
    )
    decay_offset = _store_wrf_real(pf + 2.7, dtype, rounding_token)
    decay = _store_wrf_real(
        jnp.exp(-decay_offset),
        dtype,
        rounding_token,
    )
    other = _store_wrf_real(420.0 * decay, dtype, rounding_token)
    other = jnp.where(pf <= 5.1, other, 0.1744)
    if int(opt_thcnd) not in (1, 2):
        return other
    return jnp.where(use_peters_lidard, peters_lidard, other)


def _table_value(values: Any, wrf_index: Any, dtype: Any) -> Any:
    index = jnp.asarray(wrf_index, dtype=jnp.int32) - 1
    return jnp.take(jnp.asarray(values, dtype=dtype), index, axis=0)


def _table_int_value(values: Any, wrf_index: int) -> int:
    return int(np.asarray(values)[int(wrf_index) - 1])


def redprm(
    vegtyp: int,
    soiltyp: int,
    slopetyp: int,
    shdfac: Any,
    sldpth: Any,
    zsoil: Any,
    nsoil: int,
    params: NoahLsmParameters,
    llanduse: str = "",
    lsoil: str = "",
    local: bool = False,
) -> RedprmResult:
    """Expand Noah LSM soil, vegetation, and slope parameters for one column."""

    del llanduse, lsoil, local
    nsoil_i = int(nsoil)
    vegtyp_i = jnp.asarray(vegtyp, dtype=jnp.int32)
    soiltyp_i = jnp.asarray(soiltyp, dtype=jnp.int32)
    slopetyp_i = jnp.asarray(slopetyp, dtype=jnp.int32)

    dtype = jnp.result_type(shdfac, sldpth, zsoil, jnp.float32)
    shdfac_out = jnp.asarray(shdfac, dtype=dtype)
    sldpth_arr = jnp.asarray(sldpth, dtype=dtype)
    zsoil_arr = jnp.asarray(zsoil, dtype=dtype)

    csoil = jnp.asarray(params.csoil_data, dtype=dtype)
    bexp = _table_value(params.bb, soiltyp_i, dtype)
    dksat = _table_value(params.satdk, soiltyp_i, dtype)
    dwsat = _table_value(params.satdw, soiltyp_i, dtype)
    f1 = _table_value(params.f11, soiltyp_i, dtype)
    psisat = _table_value(params.satpsi, soiltyp_i, dtype)
    quartz = _table_value(params.qtz, soiltyp_i, dtype)
    smcdry = _table_value(params.drysmc, soiltyp_i, dtype)
    smcmax = _table_value(params.maxsmc, soiltyp_i, dtype)
    smcref = _table_value(params.refsmc, soiltyp_i, dtype)
    smcwlt = _table_value(params.wltsmc, soiltyp_i, dtype)

    zbot = jnp.asarray(params.zbot_data, dtype=dtype)
    salp = jnp.asarray(params.salp_data, dtype=dtype)
    sbeta = jnp.asarray(params.sbeta_data, dtype=dtype)
    refdk = jnp.asarray(params.refdk_data, dtype=dtype)
    frzk = jnp.asarray(params.frzk_data, dtype=dtype)
    fxexp = jnp.asarray(params.fxexp_data, dtype=dtype)
    refkdt = jnp.asarray(params.refkdt_data, dtype=dtype)
    ptu = jnp.asarray(0.0, dtype=dtype)
    kdt = refkdt * dksat / refdk
    czil = jnp.asarray(params.czil_data, dtype=dtype)
    slope = _table_value(params.slop_data, slopetyp_i, dtype)
    lvcoef = jnp.asarray(params.lvcoef_data, dtype=dtype)

    frzfact = (smcmax / smcref) * (0.412 / 0.468)
    frzx = frzk * frzfact

    topt = jnp.asarray(params.topt_data, dtype=dtype)
    cmcmax = jnp.asarray(params.cmcmax_data, dtype=dtype)
    cfactr = jnp.asarray(params.cfactr_data, dtype=dtype)
    rsmax = jnp.asarray(params.rsmax_data, dtype=dtype)
    nroot = jnp.asarray(_table_value(params.nrotbl, vegtyp_i, jnp.int32), dtype=jnp.int32)
    snup = _table_value(params.snuptbl, vegtyp_i, dtype)
    rsmin = _table_value(params.rstbl, vegtyp_i, dtype)
    rgl = _table_value(params.rgltbl, vegtyp_i, dtype)
    hs = _table_value(params.hstbl, vegtyp_i, dtype)
    emissmin = _table_value(params.emissmintbl, vegtyp_i, dtype)
    emissmax = _table_value(params.emissmaxtbl, vegtyp_i, dtype)
    laimin = _table_value(params.laimintbl, vegtyp_i, dtype)
    laimax = _table_value(params.laimaxtbl, vegtyp_i, dtype)
    z0min = _table_value(params.z0mintbl, vegtyp_i, dtype)
    z0max = _table_value(params.z0maxtbl, vegtyp_i, dtype)
    albedomin = _table_value(params.albedomintbl, vegtyp_i, dtype)
    albedomax = _table_value(params.albedomaxtbl, vegtyp_i, dtype)
    ztopv = _table_value(params.ztopvtbl, vegtyp_i, dtype)
    zbotv = _table_value(params.zbotvtbl, vegtyp_i, dtype)

    shdfac_out = jnp.where(vegtyp_i == int(params.bare), jnp.asarray(0.0, dtype=dtype), shdfac_out)

    rtdis = jnp.zeros((nsoil_i,), dtype=dtype)
    root_index = jnp.clip(nroot - 1, 0, nsoil_i - 1)
    root_depth = jnp.take(zsoil_arr, root_index, axis=0)
    for idx in range(nsoil_i):
        active_root = idx < nroot
        rtdis = rtdis.at[idx].set(jnp.where(active_root, -sldpth_arr[idx] / root_depth, 0.0))

    return RedprmResult(
        cfactr=cfactr,
        cmcmax=cmcmax,
        rsmax=rsmax,
        topt=topt,
        refkdt=refkdt,
        kdt=kdt,
        sbeta=sbeta,
        shdfac=shdfac_out,
        rsmin=rsmin,
        rgl=rgl,
        hs=hs,
        zbot=zbot,
        frzx=frzx,
        psisat=psisat,
        slope=slope,
        snup=snup,
        salp=salp,
        bexp=bexp,
        dksat=dksat,
        dwsat=dwsat,
        smcmax=smcmax,
        smcwlt=smcwlt,
        smcref=smcref,
        smcdry=smcdry,
        f1=f1,
        quartz=quartz,
        fxexp=fxexp,
        rtdis=rtdis,
        nroot=nroot,
        czil=czil,
        laimin=laimin,
        laimax=laimax,
        emissmin=emissmin,
        emissmax=emissmax,
        albedomin=albedomin,
        albedomax=albedomax,
        z0min=z0min,
        z0max=z0max,
        csoil=csoil,
        ptu=ptu,
        lvcoef=lvcoef,
        ztopv=ztopv,
        zbotv=zbotv,
    )


def sflx(
    *,
    f_frozp: Any,
    isurban: int,
    dt: Any,
    zlvl: Any,
    nsoil: int,
    sldpth: Any,
    local: bool,
    llanduse: str,
    lsoil: str,
    lwdn: Any,
    soldn: Any,
    solnet: Any,
    sfcprs: Any,
    prcp: Any,
    sfctmp: Any,
    q2: Any,
    sfcspd: Any,
    cosz: Any,
    prcprain: Any,
    solardirect: Any,
    th2: Any,
    q2sat: Any,
    dqsdt2: Any,
    vegtyp: int,
    soiltyp: int,
    slopetyp: int,
    shdfac: Any,
    shdmin: Any,
    shdmax: Any,
    alb: Any,
    snoalb: Any,
    tbot: Any,
    z0brd: Any,
    z0: Any,
    emissi: Any,
    cmc: Any,
    t1: Any,
    stc: Any,
    smc: Any,
    sh2o: Any,
    snowh: Any,
    sneqv: Any,
    albedo: Any,
    ch: Any,
    cm: Any,
    eta: Any,
    sheat: Any,
    eta_kinematic: Any,
    fdown: Any,
    ec: Any,
    edir: Any,
    et: Any,
    ett: Any,
    esnow: Any,
    drip: Any,
    dew: Any,
    beta: Any,
    etp: Any,
    ssoil: Any,
    flx1: Any,
    flx2: Any,
    flx3: Any,
    flx4: Any,
    fvb: Any,
    fbur: Any,
    fgsn: Any,
    ua_phys: bool,
    snomlt: Any,
    sncovr: Any,
    runoff1: Any,
    runoff2: Any,
    runoff3: Any,
    rc: Any,
    pc: Any,
    rsmin: Any,
    xlai: Any,
    rcs: Any,
    rct: Any,
    rcq: Any,
    rcsoil: Any,
    soilw: Any,
    soilm: Any,
    q1: Any,
    smav: Any,
    rdlai2d: bool,
    usemonalb: bool,
    snotime1: Any,
    ribb: Any,
    smcwlt: Any,
    smcdry: Any,
    smcref: Any,
    smcmax: Any,
    nroot: int,
    sfhead1rt: Any,
    infxs1rt: Any,
    etpnd1: Any,
    opt_thcnd: int,
    aoasis: Any,
    xsda_qfx: Any,
    hfx_phy: Any,
    qfx_phy: Any,
    xqnorm: Any,
    fasdas: int,
    hcpct_fasdas: Any,
    params: NoahLsmParameters,
    irrigation_channel: Any | None = None,
    rounding_token: Any | None = None,
) -> SflxResult:
    """Return WRF ``SFLX`` for snow-free and snow-covered Noah land columns."""

    if ua_phys and (float(sneqv) > 1.e-7 or
                    (float(prcp) > 0.0 and
                     (float(f_frozp) > 0.5 or float(t1) <= 273.15))):
        raise NotImplementedError("Noah UA snow-canopy physics is not implemented")

    del zlvl, sfcspd, cosz, prcprain, solardirect, cm, albedo
    del eta, sheat, eta_kinematic, fdown, ec, edir, ett, esnow
    del drip, dew, beta, etp, ssoil, flx1, flx2, flx3
    del snomlt, sncovr, runoff1, runoff2, runoff3, rc, pc
    del rsmin, rcs, rct, rcq, rcsoil, soilw, soilm, q1, smav
    del smcwlt, smcdry, smcref, smcmax, nroot

    dtype = jnp.result_type(
        dt,
        sldpth,
        lwdn,
        soldn,
        solnet,
        sfcprs,
        prcp,
        sfctmp,
        q2,
        th2,
        q2sat,
        dqsdt2,
        shdfac,
        shdmin,
        shdmax,
        alb,
        snoalb,
        tbot,
        z0brd,
        z0,
        emissi,
        cmc,
        t1,
        stc,
        smc,
        sh2o,
        snowh,
        sneqv,
        ch,
        et,
        xlai,
        snotime1,
        ribb,
        sfhead1rt,
        infxs1rt,
        etpnd1,
        aoasis,
        xsda_qfx,
        hfx_phy,
        qfx_phy,
        xqnorm,
        hcpct_fasdas,
        jnp.float32,
    )
    nsoil_i = int(nsoil)
    dt = jnp.asarray(dt, dtype=dtype)
    sldpth_arr = jnp.asarray(sldpth, dtype=dtype)
    stc_out = jnp.asarray(stc, dtype=dtype)
    smc_out = jnp.asarray(smc, dtype=dtype)
    sh2o_out = jnp.asarray(sh2o, dtype=dtype)
    et_out = jnp.asarray(et, dtype=dtype)
    shdfac_out = jnp.asarray(shdfac, dtype=dtype)
    alb_out = jnp.asarray(alb, dtype=dtype)
    z0brd_out = jnp.asarray(z0brd, dtype=dtype)
    z0_out = jnp.asarray(z0, dtype=dtype)
    emissi_out = jnp.asarray(emissi, dtype=dtype)
    cmc_out = jnp.asarray(cmc, dtype=dtype)
    t1_out = jnp.asarray(t1, dtype=dtype)
    snowh_out = jnp.asarray(snowh, dtype=dtype)
    sneqv_out = jnp.asarray(sneqv, dtype=dtype)
    xlai_out = jnp.asarray(xlai, dtype=dtype)
    ch = jnp.asarray(ch, dtype=dtype)
    q2 = jnp.asarray(q2, dtype=dtype)
    q2sat = jnp.asarray(q2sat, dtype=dtype)
    sfctmp = jnp.asarray(sfctmp, dtype=dtype)
    sfcprs = jnp.asarray(sfcprs, dtype=dtype)
    th2 = jnp.asarray(th2, dtype=dtype)
    soldn = jnp.asarray(soldn, dtype=dtype)
    solnet = jnp.asarray(solnet, dtype=dtype)
    lwdn = jnp.asarray(lwdn, dtype=dtype)
    prcp = jnp.asarray(prcp, dtype=dtype)
    dqsdt2 = jnp.asarray(dqsdt2, dtype=dtype)
    shdmin = jnp.asarray(shdmin, dtype=dtype)
    shdmax = jnp.asarray(shdmax, dtype=dtype)
    snoalb = jnp.asarray(snoalb, dtype=dtype)
    tbot = jnp.asarray(tbot, dtype=dtype)
    aoasis = jnp.asarray(aoasis, dtype=dtype)
    sfhead1rt_out = jnp.asarray(sfhead1rt, dtype=dtype)
    infxs1rt_out = jnp.asarray(infxs1rt, dtype=dtype)
    etpnd1_out = jnp.asarray(etpnd1, dtype=dtype)
    xsda_qfx_out = jnp.asarray(xsda_qfx, dtype=dtype)
    hfx_phy_out = jnp.asarray(hfx_phy, dtype=dtype)
    qfx_phy_out = jnp.asarray(qfx_phy, dtype=dtype)
    xqnorm_out = jnp.asarray(xqnorm, dtype=dtype)
    hcpct_fasdas_out = jnp.asarray(hcpct_fasdas, dtype=dtype)
    snotime1_out = jnp.asarray(snotime1, dtype=dtype)
    ribb_out = jnp.asarray(ribb, dtype=dtype)

    flx4_out = jnp.asarray(flx4, dtype=dtype)
    fvb_out = jnp.asarray(fvb, dtype=dtype)
    fbur_out = jnp.asarray(fbur, dtype=dtype)
    fgsn_out = jnp.asarray(fgsn, dtype=dtype)
    if not bool(ua_phys):
        flx4_out = jnp.asarray(0.0, dtype=dtype)
        fvb_out = jnp.asarray(0.0, dtype=dtype)
        fbur_out = jnp.asarray(0.0, dtype=dtype)
        fgsn_out = jnp.asarray(0.0, dtype=dtype)

    zsoil = []
    running = -sldpth_arr[0]
    zsoil.append(running)
    for idx in range(1, nsoil_i):
        running = -sldpth_arr[idx] + running
        zsoil.append(running)
    zsoil_arr = jnp.stack(zsoil)

    prm = redprm(
        vegtyp,
        soiltyp,
        slopetyp,
        shdfac_out,
        sldpth_arr,
        zsoil_arr,
        nsoil_i,
        params,
        llanduse=llanduse,
        lsoil=lsoil,
        local=local,
    )
    shdfac_out = prm.shdfac
    rsmin_out = prm.rsmin
    smcmax_out = prm.smcmax
    smcref_out = prm.smcref
    smcwlt_out = prm.smcwlt
    smcdry_out = prm.smcdry
    urban = jnp.asarray(vegtyp, dtype=jnp.int32) == int(isurban)
    shdfac_out = jnp.where(urban, jnp.asarray(0.05, dtype=dtype), shdfac_out)
    rsmin_out = jnp.where(urban, jnp.asarray(400.0, dtype=dtype), rsmin_out)
    smcmax_out = jnp.where(urban, jnp.asarray(0.45, dtype=dtype), smcmax_out)
    smcref_out = jnp.where(urban, jnp.asarray(0.42, dtype=dtype), smcref_out)
    smcwlt_out = jnp.where(urban, jnp.asarray(0.40, dtype=dtype), smcwlt_out)
    smcdry_out = jnp.where(urban, jnp.asarray(0.40, dtype=dtype), smcdry_out)

    interp_numerator = _store_wrf_real_operation(
        shdfac_out - shdmin,
        rounding_token,
    )
    interp_denominator = _store_wrf_real_operation(
        shdmax - shdmin,
        rounding_token,
    )
    interp_candidate = _store_wrf_real_operation(
        interp_numerator / jnp.where(shdmax > shdmin, interp_denominator, 1.0),
        rounding_token,
    )
    interp = jnp.where(
        shdmax > shdmin,
        jnp.clip(interp_candidate, 0.0, 1.0),
        0.5,
    )
    ge_max = shdfac_out >= shdmax
    le_min = shdfac_out <= shdmin
    mid = shdmax > shdmin
    emiss_min_weight = _store_wrf_real_operation(
        1.0 - interp,
        rounding_token,
    )
    emiss_min_term = _store_wrf_real_operation(
        emiss_min_weight * prm.emissmin,
        rounding_token,
    )
    emiss_max_term = _store_wrf_real_operation(
        interp * prm.emissmax,
        rounding_token,
    )
    embrd_mid = _store_wrf_real_operation(
        emiss_min_term + emiss_max_term,
        rounding_token,
    )
    xlai_mid = (1.0 - interp) * prm.laimin + interp * prm.laimax
    alb_mid = (1.0 - interp) * prm.albedomax + interp * prm.albedomin
    z0_mid = (1.0 - interp) * prm.z0min + interp * prm.z0max
    embrd_default = 0.5 * prm.emissmin + 0.5 * prm.emissmax
    xlai_default = 0.5 * prm.laimin + 0.5 * prm.laimax
    alb_default = 0.5 * prm.albedomin + 0.5 * prm.albedomax
    z0_default = 0.5 * prm.z0min + 0.5 * prm.z0max
    embrd = jnp.where(ge_max, prm.emissmax, jnp.where(le_min, prm.emissmin, jnp.where(mid, embrd_mid, embrd_default)))
    xlai_candidate = jnp.where(ge_max, prm.laimax, jnp.where(le_min, prm.laimin, jnp.where(mid, xlai_mid, xlai_default)))
    alb_candidate = jnp.where(ge_max, prm.albedomin, jnp.where(le_min, prm.albedomax, jnp.where(mid, alb_mid, alb_default)))
    z0brd_out = jnp.where(ge_max, prm.z0max, jnp.where(le_min, prm.z0min, jnp.where(mid, z0_mid, z0_default)))
    if not bool(rdlai2d):
        xlai_out = xlai_candidate
    if not bool(usemonalb):
        alb_out = alb_candidate

    store = lambda value: _store_wrf_real(value, dtype, rounding_token)
    old_snow = sneqv_out > 1.0e-7
    sneqv_out = jnp.where(old_snow, sneqv_out, 0.0)
    snowh_out = jnp.where(old_snow, snowh_out, 0.0)
    sndens = store(jnp.where(old_snow, sneqv_out / jnp.where(old_snow, snowh_out, 1.0), 0.0))
    snowng = (prcp > 0.0) & (jnp.asarray(f_frozp) > 0.5)
    frzgra = (prcp > 0.0) & ~snowng & (t1_out <= 273.15)
    frozen_precip = snowng | frzgra
    new_snow = store(prcp * dt * 0.001)
    new_depth, new_density = snow_new(sfctmp, new_snow, snowh_out, sndens)
    sneqv_out = jnp.where(frozen_precip, store(sneqv_out + new_snow), sneqv_out)
    snowh_out = jnp.where(frozen_precip, new_depth, snowh_out)
    sndens = jnp.where(frozen_precip, new_density, sndens)
    prcpf = jnp.where(frozen_precip, 0.0, prcp)
    has_snow = sneqv_out > 0.0
    sncond = jnp.where(has_snow, csnow(sndens), 1.0)

    # SNFRAC and ALCALC use the updated snowpack, before any melt/sublimation.
    rsnow = store(sneqv_out / prm.snup)
    cover = store(1.0 - (jnp.exp(-prm.salp * rsnow) - rsnow * jnp.exp(-prm.salp)))
    sncovr_out = jnp.where(has_snow, jnp.minimum(jnp.where(sneqv_out < prm.snup, cover, 1.0), 0.98), 0.0)
    snow_age = jnp.where(snowng, 0.0, store(snotime1_out + dt))
    snotime1_out = jnp.where(has_snow, snow_age, snotime1_out)
    snoalb1 = store(snoalb + prm.lvcoef * (0.85 - snoalb))
    snoalb2 = store(snoalb1 * (0.94 ** ((snow_age / 86400.0) ** 0.58)))
    snoalb2 = jnp.maximum(jnp.where(snowng, snoalb1, snoalb2), alb_out)
    albedo_out = jnp.where(has_snow, jnp.minimum(store(alb_out + sncovr_out * (snoalb2 - alb_out)), snoalb2), alb_out)
    emissi_out = jnp.where(has_snow, store(embrd + sncovr_out * (0.95 - embrd)), embrd)
    z0_out = jnp.where(sncovr_out > 0.0, snowz0(sncovr_out, z0brd_out, snowh_out), z0brd_out)

    df1 = _store_wrf_real(
        _tdfcnd_wrf_real(
            smc_out[0],
            prm.quartz,
            smcmax_out,
            sh2o_out[0],
            prm.bexp,
            prm.psisat,
            soiltyp,
            opt_thcnd,
            rounding_token,
        ),
        dtype,
        rounding_token,
    )
    df1 = _store_wrf_real(
        jnp.where(urban, jnp.asarray(3.24, dtype=dtype), df1),
        dtype,
        rounding_token,
    )
    df1 = _store_wrf_real(
        df1 * jnp.exp(prm.sbeta * shdfac_out),
        dtype,
        rounding_token,
    )
    df1 = jnp.where(sncovr_out > 0.97, sncond, df1)

    dsoil = _store_wrf_real(
        -(jnp.asarray(0.5, dtype=dtype) * zsoil_arr[0]),
        dtype,
        rounding_token,
    )
    dtot = store(snowh_out + dsoil)
    frcsno = store(snowh_out / dtot)
    frcsoi = store(dsoil / dtot)
    df1a = store(frcsno * sncond + frcsoi * df1)
    df1 = jnp.where(has_snow, store(df1a * sncovr_out + df1 * (1.0 - sncovr_out)), df1)
    ssoil_out = _store_wrf_real(
        df1 * (t1_out - stc_out[0]) / jnp.where(has_snow, dtot, dsoil),
        dtype,
        rounding_token,
    )
    fdown_out = solnet + lwdn
    t2v = sfctmp * (1.0 + 0.61 * q2)
    pen = penman(
        sfctmp,
        sfcprs,
        ch,
        t2v,
        th2,
        prcp,
        fdown_out,
        ssoil_out,
        q2,
        q2sat,
        snowng,
        frzgra,
        dqsdt2,
        emissi_out,
        sneqv_out,
        t1_out,
        sncovr_out,
        aoasis,
        albedo_out,
        soldn,
        fvb_out,
        jnp.asarray(0.0, dtype=dtype),
        stc_out[0],
        bool(ua_phys),
        rounding_token,
    )
    flx2_out = pen.flx2
    flx4_out = pen.flx4

    canopy_active = (shdfac_out > 0.0) & (xlai_out > 0.0)
    canopy = canres(
        soldn,
        ch,
        sfctmp,
        q2,
        sfcprs,
        sh2o_out,
        zsoil_arr,
        nsoil_i,
        smcwlt_out,
        smcref_out,
        rsmin_out,
        prm.nroot,
        q2sat,
        dqsdt2,
        prm.topt,
        prm.rsmax,
        prm.rgl,
        prm.hs,
        jnp.where(canopy_active, xlai_out, 1.0),
        emissi_out,
        rounding_token,
    )
    pc_out = jnp.where(canopy_active, canopy.pc, 0.0)
    rc_out = jnp.where(canopy_active, canopy.rc, 0.0)
    rcq_out = jnp.where(canopy_active, canopy.rcq, 0.0)
    rcs_out = jnp.where(canopy_active, canopy.rcs, 0.0)
    rcsoil_out = jnp.where(canopy_active, canopy.rcsoil, 0.0)
    rct_out = jnp.where(canopy_active, canopy.rct, 0.0)

    surface = soil_surface_fluxes(
        pen.etp,
        jnp.asarray(0.0, dtype=dtype),
        prcp,
        smc_out,
        smcmax_out,
        smcwlt_out,
        smcref_out,
        smcdry_out,
        cmc_out,
        prm.cmcmax,
        nsoil_i,
        dt,
        shdfac_out,
        prm.sbeta,
        q2,
        t1_out,
        sfctmp,
        pen.t24,
        th2,
        fdown_out,
        prm.f1,
        emissi_out,
        ssoil_out,
        stc_out,
        pen.epsca,
        prm.bexp,
        pc_out,
        pen.rch,
        pen.rr,
        prm.cfactr,
        sh2o_out,
        prm.slope,
        prm.kdt,
        prm.frzx,
        prm.psisat,
        zsoil_arr,
        prm.dksat,
        prm.dwsat,
        tbot,
        prm.zbot,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        et_out,
        jnp.asarray(0.0, dtype=dtype),
        prm.nroot,
        prm.rtdis,
        prm.quartz,
        prm.fxexp,
        prm.csoil,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        vegtyp,
        isurban,
        sfhead1rt_out,
        infxs1rt_out,
        etpnd1_out,
        soiltyp,
        opt_thcnd,
        xsda_qfx_out,
        qfx_phy_out,
        xqnorm_out,
        fasdas,
        hcpct_fasdas_out,
        irrigation_channel,
        rounding_token,
        snowng=snowng,
        prcpf=prcpf,
        sneqv=sneqv_out,
        snowh=snowh_out,
        sndens=sndens,
        sncovr=sncovr_out,
        snow_df1=df1,
        flx2=flx2_out,
        ribb=ribb_out,
    )

    sneqv_out = surface.sneqv
    snowh_out = surface.snowh
    sncovr_out = surface.sncovr

    eta_kinematic_out = surface.eta
    q1_out = q2 + eta_kinematic_out * CP_MODEL / pen.rch
    sheat_out = -((ch * CP_MODEL * sfcprs) / (RD * t2v)) * (th2 - surface.t1)
    if bool(ua_phys):
        sheat_out = sheat_out + flx4_out
    if int(fasdas) == 1:
        hfx_phy_out = sheat_out

    lvh2o = jnp.asarray(2.501e6, dtype=dtype)
    lsubs = jnp.asarray(2.83e6, dtype=dtype)
    edir_out = _store_wrf_real_operation(
        surface.edir * lvh2o,
        rounding_token,
    )
    ec_out = _store_wrf_real_operation(
        surface.ec * lvh2o,
        rounding_token,
    )
    et_scaled = _store_wrf_real_operation(
        surface.et * lvh2o,
        rounding_token,
    )
    ett_out = _store_wrf_real_operation(
        surface.ett * lvh2o,
        rounding_token,
    )
    etpnd1_out = _store_wrf_real_operation(
        surface.etpnd1 * lvh2o,
        rounding_token,
    )
    esnow_out = _store_wrf_real_operation(
        surface.esnow * lsubs,
        rounding_token,
    )
    latent_heat = _store_wrf_real_operation(
        _store_wrf_real_operation(
            (1.0 - sncovr_out) * lvh2o,
            rounding_token,
        )
        + _store_wrf_real_operation(
            sncovr_out * lsubs,
            rounding_token,
        ),
        rounding_token,
    )
    etp_out = _store_wrf_real_operation(
        surface.etp * latent_heat,
        rounding_token,
    )
    eta_positive = _store_wrf_real_operation(
        edir_out + ec_out,
        rounding_token,
    )
    eta_positive = _store_wrf_real_operation(
        eta_positive + ett_out,
        rounding_token,
    )
    eta_positive = _store_wrf_real_operation(
        eta_positive + esnow_out,
        rounding_token,
    )
    eta_out = jnp.where(etp_out > 0.0, eta_positive, etp_out)
    beta_positive = _store_wrf_real_operation(
        eta_out / etp_out,
        rounding_token,
    )
    beta_out = jnp.where(
        etp_out == 0.0,
        jnp.asarray(0.0, dtype=dtype),
        beta_positive,
    )
    ssoil_final = -surface.ssoil

    runoff3_out = surface.runoff3 / dt
    runoff2_out = surface.runoff2 + runoff3_out
    soilm_smc = surface.smc
    soilm_out = -soilm_smc[0] * zsoil_arr[0]
    for idx in range(1, nsoil_i):
        soilm_out = soilm_out + soilm_smc[idx] * (zsoil_arr[idx - 1] - zsoil_arr[idx])
    soilwm = -(smcmax_out - smcwlt_out) * zsoil_arr[0]
    soilww = -(soilm_smc[0] - smcwlt_out) * zsoil_arr[0]
    smav_out = (surface.smc - smcwlt_out) / (smcmax_out - smcwlt_out)
    for idx in range(1, nsoil_i):
        active_root = idx < prm.nroot
        soilwm = soilwm + jnp.where(active_root, (smcmax_out - smcwlt_out) * (zsoil_arr[idx - 1] - zsoil_arr[idx]), 0.0)
        soilww = soilww + jnp.where(active_root, (surface.smc[idx] - smcwlt_out) * (zsoil_arr[idx - 1] - zsoil_arr[idx]), 0.0)
    soilw_out = jnp.where(soilwm < 1.0e-6, jnp.asarray(0.0, dtype=dtype), soilww / soilwm)
    soilm_out = jnp.where(soilwm < 1.0e-6, jnp.asarray(0.0, dtype=dtype), soilm_out)

    return SflxResult(
        eta=eta_out,
        sheat=sheat_out,
        eta_kinematic=eta_kinematic_out,
        fdown=fdown_out,
        ec=ec_out,
        edir=edir_out,
        et=et_scaled,
        ett=ett_out,
        esnow=esnow_out,
        drip=surface.drip,
        dew=surface.dew,
        beta=beta_out,
        etp=etp_out,
        ssoil=ssoil_final,
        flx1=surface.flx1,
        flx2=flx2_out,
        flx3=surface.flx3,
        flx4=flx4_out,
        fvb=fvb_out,
        fbur=fbur_out,
        fgsn=fgsn_out,
        snomlt=surface.snomlt,
        sncovr=sncovr_out,
        runoff1=surface.runoff1,
        runoff2=runoff2_out,
        runoff3=runoff3_out,
        rc=rc_out,
        pc=pc_out,
        rsmin=rsmin_out,
        xlai=xlai_out,
        rcs=rcs_out,
        rct=rct_out,
        rcq=rcq_out,
        rcsoil=rcsoil_out,
        soilw=soilw_out,
        soilm=soilm_out,
        q1=q1_out,
        smav=smav_out,
        emb_rd=embrd,
        albedo=albedo_out,
        z0brd=z0brd_out,
        z0=z0_out,
        emissi=emissi_out,
        alb=alb_out,
        cmc=surface.cmc,
        t1=surface.t1,
        stc=surface.stc,
        smc=surface.smc,
        sh2o=surface.sh2o,
        snowh=snowh_out,
        sneqv=sneqv_out,
        nroot=prm.nroot,
        sfhead1rt=surface.sfhead1rt,
        infxs1rt=surface.infxs1rt,
        etpnd1=etpnd1_out,
        xsda_qfx=xsda_qfx_out,
        hfx_phy=hfx_phy_out,
        qfx_phy=surface.qfx_phy,
        xqnorm=xqnorm_out,
        hcpct_fasdas=surface.hcpct_fasdas,
        snotime1=snotime1_out,
        ribb=ribb_out,
        smcwlt=smcwlt_out,
        smcdry=smcdry_out,
        smcref=smcref_out,
        smcmax=smcmax_out,
    )


__all__ = [
    "CanresResult",
    "EvapoResult",
    "HrtResult",
    "HstepResult",
    "SoilSurfaceResult",
    "NoahLsmParameters",
    "PenmanResult",
    "RedprmResult",
    "Rosr12Result",
    "SnksrcResult",
    "SmflxResult",
    "SrtResult",
    "SstepResult",
    "ShflxResult",
    "SflxResult",
    "WdfcndResult",
    "canres",
    "devap",
    "evapo",
    "fac2mit",
    "frh2o",
    "hrt",
    "hstep",
    "soil_surface_fluxes",
    "penman",
    "redprm",
    "rosr12",
    "snksrc",
    "shflx",
    "sflx",
    "smflx",
    "srt",
    "sstep",
    "tbnd",
    "tdfcnd",
    "tmpavg",
    "transp",
    "wdfcnd",
]
