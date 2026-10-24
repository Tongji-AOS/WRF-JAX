"""Kernels translated from WRF ``dyn_em/module_em``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jax import lax
import jax.numpy as jnp
from wrf_jax.dyn_em.module_advect_em import advect_scalar, advect_scalar_pd
from wrf_jax.dyn_em.module_big_step_utilities_em import (
    _idx,
    _read,
    _store_wrf_real,
    calc_alt,
    calc_cq,
    calc_mu_uv,
    calc_php,
    calc_ww_cp,
    couple_momentum,
    zero_tend,
    zero_tend2d,
)
from wrf_jax.dyn_em.module_ieva_em import chk_ieva, ww_split

PARAM_FIRST_SCALAR = 2
P_QV = 2
P_QC = 3
P_QR = 4
P_QI = 5
P_QS = 6
P_QG = 7
POSITIVEDEF = 1
MONOTONIC = 2
WENO_SCALAR = 3
WENOPD_SCALAR = 4
GDSCHEME = 93
G3SCHEME = 5
GFSCHEME = 3
KFETASCHEME = 1
MSKFSCHEME = 11
TIEDTKESCHEME = 6
NTIEDTKESCHEME = 16
WENO_MOM = 3
HELDSUAREZ = 31


@dataclass(frozen=True)
class RkTendencyResult:
    """Explicit state and call trace for WRF ``rk_tendency`` orchestration."""

    fields: Mapping[str, Any]
    events: tuple[str, ...]
    dt_step: float
    ieva: bool
    max_vert_cfl: float
    max_horiz_cfl: float


class RkStepPrepResult:
    def __init__(
        self,
        *,
        ru: Any,
        rv: Any,
        rw: Any,
        ww: Any,
        php: Any,
        alt: Any,
        muu: Any,
        muv: Any,
        mut: Any,
        cqu: Any,
        cqv: Any,
        cqw: Any,
    ):
        self.ru = ru
        self.rv = rv
        self.rw = rw
        self.ww = ww
        self.php = php
        self.alt = alt
        self.muu = muu
        self.muv = muv
        self.mut = mut
        self.cqu = cqu
        self.cqv = cqv
        self.cqw = cqw


class InitZeroTendencyResult:
    def __init__(
        self,
        *,
        ru_tendf: Any,
        rv_tendf: Any,
        rw_tendf: Any,
        ph_tendf: Any,
        t_tendf: Any,
        tke_tendf: Any,
        mu_tendf: Any,
        moist_tendf: Any,
        chem_tendf: Any,
        scalar_tendf: Any,
        tracer_tendf: Any,
    ):
        self.ru_tendf = ru_tendf
        self.rv_tendf = rv_tendf
        self.rw_tendf = rw_tendf
        self.ph_tendf = ph_tendf
        self.t_tendf = t_tendf
        self.tke_tendf = tke_tendf
        self.mu_tendf = mu_tendf
        self.moist_tendf = moist_tendf
        self.chem_tendf = chem_tendf
        self.scalar_tendf = scalar_tendf
        self.tracer_tendf = tracer_tendf


class RkScalarTendResult:
    def __init__(
        self,
        *,
        scalar_tends: Any,
        advect_tend: Any,
        h_tendency: Any,
        z_tendency: Any,
        rqvften: Any,
        ww_e: Any,
        ww_i: Any,
    ):
        self.scalar_tends = scalar_tends
        self.advect_tend = advect_tend
        self.h_tendency = h_tendency
        self.z_tendency = z_tendency
        self.rqvften = rqvften
        self.ww_e = ww_e
        self.ww_i = ww_i


def init_zero_tendency(
    ru_tendf: Any,
    rv_tendf: Any,
    rw_tendf: Any,
    ph_tendf: Any,
    t_tendf: Any,
    tke_tendf: Any,
    mu_tendf: Any,
    moist_tendf: Any,
    chem_tendf: Any,
    scalar_tendf: Any,
    tracer_tendf: Any,
    n_tracer: int,
    n_moist: int,
    n_chem: int,
    n_scalar: int,
    rk_step: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    *,
    zero_tend_fn=zero_tend,
    zero_tend2d_fn=zero_tend2d,
) -> InitZeroTendencyResult:
    """Return WRF ``init_zero_tendency`` fields with active tendency tiles zeroed."""

    del rk_step

    def zero3(field: Any) -> Any:
        return zero_tend_fn(field, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

    ru_out = zero3(ru_tendf)
    rv_out = zero3(rv_tendf)
    rw_out = zero3(rw_tendf)
    ph_out = zero3(ph_tendf)
    t_out = zero3(t_tendf)
    tke_out = zero3(tke_tendf)
    mu_out = zero_tend2d_fn(mu_tendf, ids, ide, jds, jde, kds, kds, ims, ime, jms, jme, kms, kms, its, ite, jts, jte, kts, kts)

    moist_out = jnp.asarray(moist_tendf)
    for im in range(1, int(n_moist) + 1):
        zeroed = zero3(moist_out[:, :, :, im - 1])
        moist_out = moist_out.at[:, :, :, im - 1].set(zeroed)

    chem_out = jnp.asarray(chem_tendf)
    for ic in range(1, int(n_chem) + 1):
        zeroed = zero3(chem_out[:, :, :, ic - 1])
        chem_out = chem_out.at[:, :, :, ic - 1].set(zeroed)

    tracer_out = jnp.asarray(tracer_tendf)
    for ic in range(1, int(n_tracer) + 1):
        zeroed = zero3(tracer_out[:, :, :, ic - 1])
        tracer_out = tracer_out.at[:, :, :, ic - 1].set(zeroed)

    scalar_out = jnp.asarray(scalar_tendf)
    for ic in range(1, int(n_scalar) + 1):
        zeroed = zero3(scalar_out[:, :, :, ic - 1])
        scalar_out = scalar_out.at[:, :, :, ic - 1].set(zeroed)

    return InitZeroTendencyResult(
        ru_tendf=ru_out,
        rv_tendf=rv_out,
        rw_tendf=rw_out,
        ph_tendf=ph_out,
        t_tendf=t_out,
        tke_tendf=tke_out,
        mu_tendf=mu_out,
        moist_tendf=moist_out,
        chem_tendf=chem_out,
        scalar_tendf=scalar_out,
        tracer_tendf=tracer_out,
    )


def rk_step_prep(
    config_flags: Any,
    rk_step: int,
    u: Any,
    v: Any,
    w: Any,
    t: Any,
    ph: Any,
    mu: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    moist: Any,
    mub: Any,
    phb: Any,
    pb: Any,
    p: Any,
    al: Any,
    alb: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fnm: Any,
    fnp: Any,
    dnw: Any,
    rdx: float,
    rdy: float,
    n_moist: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
    *,
    calculate_full_2d_fn=None,
    calc_mu_uv_fn=calc_mu_uv,
    couple_momentum_fn=couple_momentum,
    calc_ww_cp_fn=calc_ww_cp,
    calc_cq_fn=calc_cq,
    calc_alt_fn=calc_alt,
    calc_php_fn=calc_php,
) -> RkStepPrepResult:
    """Run WRF ``rk_step_prep`` orchestration and return its output fields."""

    del rk_step, t, pb, p, fnm, fnp

    if calculate_full_2d_fn is None:
        mut = jnp.asarray(mub) + jnp.asarray(mu)
    else:
        mut = calculate_full_2d_fn(
            mub,
            mu,
            ids,
            ide,
            jds,
            jde,
            1,
            2,
            ims,
            ime,
            jms,
            jme,
            1,
            1,
            its,
            ite,
            jts,
            jte,
            1,
            1,
        )

    muu, muv = calc_mu_uv_fn(
        config_flags,
        mu,
        mub,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )

    ru, rv, rw = couple_momentum_fn(
        muu,
        u,
        msfuy,
        muv,
        v,
        msfvx,
        msfvx_inv,
        mut,
        w,
        msfty,
        c1h,
        c2h,
        c1f,
        c2f,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )

    ww = calc_ww_cp_fn(
        u,
        v,
        mu,
        mub,
        c1h,
        c2h,
        rdx,
        rdy,
        msftx,
        msfty,
        msfux,
        msfuy,
        msfvx,
        msfvx_inv,
        msfvy,
        dnw,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )

    cqu, cqv, cqw = calc_cq_fn(
        moist,
        n_moist,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token=rounding_token,
    )

    alt = calc_alt_fn(al, alb, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    php = calc_php_fn(ph, phb, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

    return RkStepPrepResult(
        ru=ru,
        rv=rv,
        rw=rw,
        ww=ww,
        php=php,
        alt=alt,
        muu=muu,
        muv=muv,
        mut=mut,
        cqu=cqu,
        cqv=cqv,
        cqw=cqw,
    )


def calculate_phy_tend(
    config_flags: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    muu: Any,
    muv: Any,
    pi3d: Any,
    rthraten: Any,
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rucuten: Any,
    rvcuten: Any,
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqrcuten: Any,
    rqicuten: Any,
    rqscuten: Any,
    rushten: Any,
    rvshten: Any,
    rthshten: Any,
    rqvshten: Any,
    rqcshten: Any,
    rqrshten: Any,
    rqishten: Any,
    rqsshten: Any,
    rqgshten: Any,
    rundgdten: Any,
    rvndgdten: Any,
    rthndgdten: Any,
    rqvndgdten: Any,
    rmundgdten: Any,
    scalar: Any,
    scalar_tend: Any,
    num_scalar: int,
    tracer: Any,
    tracer_tend: Any,
    num_tracer: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
    p_qv: int = P_QV,
    p_qc: int = P_QC,
    p_qr: int = P_QR,
    p_qi: int = P_QI,
    p_qs: int = P_QS,
    p_qg: int = P_QG,
    param_first_scalar: int = PARAM_FIRST_SCALAR,
) -> tuple[Any, ...]:
    """Apply WRF ``calculate_phy_tend`` mass factors to physics tendencies."""

    del pi3d, scalar, tracer, ime, jme, kme

    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mut_arr = jnp.asarray(mut)
    muu_arr = jnp.asarray(muu)
    muv_arr = jnp.asarray(muv)

    rthraten_out = jnp.asarray(rthraten)
    rublten_out = jnp.asarray(rublten)
    rvblten_out = jnp.asarray(rvblten)
    rthblten_out = jnp.asarray(rthblten)
    rqvblten_out = jnp.asarray(rqvblten)
    rqcblten_out = jnp.asarray(rqcblten)
    rqiblten_out = jnp.asarray(rqiblten)
    rucuten_out = jnp.asarray(rucuten)
    rvcuten_out = jnp.asarray(rvcuten)
    rthcuten_out = jnp.asarray(rthcuten)
    rqvcuten_out = jnp.asarray(rqvcuten)
    rqccuten_out = jnp.asarray(rqccuten)
    rqrcuten_out = jnp.asarray(rqrcuten)
    rqicuten_out = jnp.asarray(rqicuten)
    rqscuten_out = jnp.asarray(rqscuten)
    rushten_out = jnp.asarray(rushten)
    rvshten_out = jnp.asarray(rvshten)
    rthshten_out = jnp.asarray(rthshten)
    rqvshten_out = jnp.asarray(rqvshten)
    rqcshten_out = jnp.asarray(rqcshten)
    rqrshten_out = jnp.asarray(rqrshten)
    rqishten_out = jnp.asarray(rqishten)
    rqsshten_out = jnp.asarray(rqsshten)
    rqgshten_out = jnp.asarray(rqgshten)
    rundgdten_out = jnp.asarray(rundgdten)
    rvndgdten_out = jnp.asarray(rvndgdten)
    rthndgdten_out = jnp.asarray(rthndgdten)
    rqvndgdten_out = jnp.asarray(rqvndgdten)
    rmundgdten_out = jnp.asarray(rmundgdten)
    scalar_tend_out = jnp.asarray(scalar_tend)
    tracer_tend_out = jnp.asarray(tracer_tend)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    itsu = max(int(its), int(ids) + 1)
    jtsv = max(int(jts), int(jds) + 1)

    def store_real(value: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(value, dtype=jnp.float32)
        return _store_wrf_real(value, rounding_token)

    def multiply_3d(field: Any, mass: Any, i_start: int = int(its), j_start: int = int(jts)) -> Any:
        if i_start > itf or j_start > jtf or int(kts) > ktf:
            return field
        region = (
            slice(i_start - ims, itf - ims + 1),
            slice(kts - kms, ktf - kms + 1),
            slice(j_start - jms, jtf - jms + 1),
        )
        i, k, j = region
        # WRF's REAL expression is evaluated as a rounded multiply followed by
        # a rounded add.  Dynamic bit-level stores prevent XLA from contracting
        # ``c1 * mass + c2`` into an FMA.
        factor_product = store_real(
            c1_arr[k][None, :, None] * mass[i, j][:, None, :]
        )
        factor = store_real(factor_product + c2_arr[k][None, :, None])
        result = store_real(field[region] * factor).astype(field.dtype)
        return lax.dynamic_update_slice(field, result, tuple(part.start for part in region))

    def multiply_4d_species(field: Any, num_species: int) -> Any:
        if int(its) > itf or int(jts) > jtf or int(kts) > ktf or num_species < param_first_scalar:
            return field
        region = (
            slice(its - ims, itf - ims + 1),
            slice(kts - kms, ktf - kms + 1),
            slice(jts - jms, jtf - jms + 1),
            slice(param_first_scalar - 1, num_species),
        )
        i, k, j, _ = region
        factor_product = store_real(
            c1_arr[k][None, :, None] * mut_arr[i, j][:, None, :]
        )
        factor = store_real(factor_product + c2_arr[k][None, :, None])
        result = store_real(field[region] * factor[..., None]).astype(field.dtype)
        return lax.dynamic_update_slice(field, result, tuple(part.start for part in region))

    if int(_read(config_flags, "ra_lw_physics", 0)) > 0 or int(_read(config_flags, "ra_sw_physics", 0)) > 0:
        rthraten_out = multiply_3d(rthraten_out, mut_arr)

    if int(_read(config_flags, "cu_physics", 0)) > 0:
        rucuten_out = multiply_3d(rucuten_out, mut_arr)
        rvcuten_out = multiply_3d(rvcuten_out, mut_arr)
        rthcuten_out = multiply_3d(rthcuten_out, mut_arr)
        rqvcuten_out = multiply_3d(rqvcuten_out, mut_arr)
        if int(p_qc) >= int(param_first_scalar):
            rqccuten_out = multiply_3d(rqccuten_out, mut_arr)
        if int(p_qr) >= int(param_first_scalar):
            rqrcuten_out = multiply_3d(rqrcuten_out, mut_arr)
        if int(p_qi) >= int(param_first_scalar):
            rqicuten_out = multiply_3d(rqicuten_out, mut_arr)
        if int(p_qs) >= int(param_first_scalar):
            rqscuten_out = multiply_3d(rqscuten_out, mut_arr)

    if int(_read(config_flags, "shcu_physics", 0)) > 0:
        rushten_out = multiply_3d(rushten_out, mut_arr)
        rvshten_out = multiply_3d(rvshten_out, mut_arr)
        rthshten_out = multiply_3d(rthshten_out, mut_arr)
        rqvshten_out = multiply_3d(rqvshten_out, mut_arr)
        if int(p_qc) >= int(param_first_scalar):
            rqcshten_out = multiply_3d(rqcshten_out, mut_arr)
        if int(p_qr) >= int(param_first_scalar):
            rqrshten_out = multiply_3d(rqrshten_out, mut_arr)
        if int(p_qi) >= int(param_first_scalar):
            rqishten_out = multiply_3d(rqishten_out, mut_arr)
        if int(p_qs) >= int(param_first_scalar):
            rqsshten_out = multiply_3d(rqsshten_out, mut_arr)
        if int(p_qg) >= int(param_first_scalar):
            rqgshten_out = multiply_3d(rqgshten_out, mut_arr)

    if int(_read(config_flags, "bl_pbl_physics", 0)) > 0:
        rublten_out = multiply_3d(rublten_out, mut_arr)
        rvblten_out = multiply_3d(rvblten_out, mut_arr)
        rthblten_out = multiply_3d(rthblten_out, mut_arr)
        if int(p_qv) >= int(param_first_scalar):
            rqvblten_out = multiply_3d(rqvblten_out, mut_arr)
        if int(p_qc) >= int(param_first_scalar):
            rqcblten_out = multiply_3d(rqcblten_out, mut_arr)
        if int(p_qi) >= int(param_first_scalar):
            rqiblten_out = multiply_3d(rqiblten_out, mut_arr)

    if int(_read(config_flags, "grid_fdda", 0)) > 0:
        rundgdten_out = multiply_3d(rundgdten_out, muu_arr, i_start=itsu)
        rvndgdten_out = multiply_3d(rvndgdten_out, muv_arr, j_start=jtsv)
        rthndgdten_out = multiply_3d(rthndgdten_out, mut_arr)
        if int(p_qv) >= int(param_first_scalar):
            rqvndgdten_out = multiply_3d(rqvndgdten_out, mut_arr)

    scalar_tend_out = multiply_4d_species(scalar_tend_out, num_scalar)
    tracer_tend_out = multiply_4d_species(tracer_tend_out, num_tracer)

    return (
        rthraten_out,
        rublten_out,
        rvblten_out,
        rthblten_out,
        rqvblten_out,
        rqcblten_out,
        rqiblten_out,
        rucuten_out,
        rvcuten_out,
        rthcuten_out,
        rqvcuten_out,
        rqccuten_out,
        rqrcuten_out,
        rqicuten_out,
        rqscuten_out,
        rushten_out,
        rvshten_out,
        rthshten_out,
        rqvshten_out,
        rqcshten_out,
        rqrshten_out,
        rqishten_out,
        rqsshten_out,
        rqgshten_out,
        rundgdten_out,
        rvndgdten_out,
        rthndgdten_out,
        rqvndgdten_out,
        rmundgdten_out,
        scalar_tend_out,
        tracer_tend_out,
    )


def rk_tendency(
    *,
    config_flags: Any,
    rk_step: int,
    fields: Mapping[str, Any],
    n_moist: int,
    non_hydrostatic: bool,
    top_lid: bool,
    dt: float,
    rdx: float,
    rdy: float,
    khdif: float = 0.0,
    kvdif: float = 0.0,
    diff_6th_opt: int = 0,
    diff_6th_factor: float = 0.0,
    adv_opt: int = 0,
    dampcoef: float = 0.0,
    zdamp: float = 0.0,
    damp_opt: int = 0,
    rad_nudge: int = 0,
    ids: int = 1,
    ide: int = 1,
    jds: int = 1,
    jde: int = 1,
    kds: int = 1,
    kde: int = 1,
    ims: int = 1,
    ime: int = 1,
    jms: int = 1,
    jme: int = 1,
    kms: int = 1,
    kme: int = 1,
    its: int = 1,
    ite: int = 1,
    jts: int = 1,
    jte: int = 1,
    kts: int = 1,
    kte: int = 1,
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]] | None = None,
    record_events: bool = True,
) -> RkTendencyResult:
    """Run WRF ``rk_tendency`` as explicit dycore orchestration.

    This top-level routine wires together already translated kernels and exposes
    not-yet-inlined kernels through named callbacks. It preserves WRF's call
    order and explicit ``intent(out)``/``intent(inout)`` state while keeping the
    translated child kernels independently testable.
    """

    del n_moist, top_lid, khdif, kvdif, diff_6th_factor, dampcoef, zdamp, ids, jds, kds

    state: dict[str, Any] = {name: value for name, value in fields.items()}
    events: list[str] = []
    events_enabled = bool(record_events)

    def require(names: tuple[str, ...], stage: str) -> None:
        missing = [name for name in names if name not in state]
        if missing:
            raise ValueError(f"rk_tendency {stage} missing fields: {', '.join(missing)}")

    def call_stage(name: str) -> None:
        if events_enabled:
            events.append(f"call:{name}")
        if callbacks is None or name not in callbacks:
            return
        debug_fn = state.get("rk_tendency_inner_debug_fn")
        before = dict(state) if debug_fn is not None else None
        updates = callbacks[name](state)
        if debug_fn is not None:
            after = dict(state)
            if updates:
                after.update(updates)
            debug_fn(name, before, after, updates)
        if updates:
            state.update(updates)

    for name in ("ru_tend", "rv_tend", "rw_tend", "t_tend", "ph_tend", "u_save", "v_save", "w_save", "ph_save", "t_save"):
        require((name,), f"zero_tend:{name}")
        if events_enabled:
            events.append(f"call:zero_tend:{name}")
        state[name] = zero_tend(state[name], 1, int(ide), 1, int(jde), 1, int(kde), int(ims), int(ime), int(jms), int(jme), int(kms), int(kme), int(its), int(ite), int(jts), int(jte), int(kts), int(kte))

    for name in ("mu_tend", "mu_save"):
        require((name,), f"zero_tend2d:{name}")
        if events_enabled:
            events.append(f"call:zero_tend2d:{name}")
        state[name] = zero_tend2d(state[name], 1, int(ide), 1, int(jde), 1, 1, int(ims), int(ime), int(jms), int(jme), 1, 1, int(its), int(ite), int(jts), int(jte), 1, 1)

    if events_enabled:
        events.append("call:nl_get_time_step")
    time_step = int(_read(config_flags, "time_step", 1))
    rk_order = int(_read(config_flags, "rk_ord", int(rk_step)))
    dt_step = float(dt) / float(rk_order - int(rk_step) + 1)

    require(
        (
            "ww_e",
            "ww_i",
            "u",
            "v",
            "ww",
            "mut",
            "rdnw",
            "msfty",
            "c1f",
            "c2f",
            "msfux",
            "msfuy",
            "msfvx",
            "msfvy",
        ),
        "ww_split",
    )
    if events_enabled:
        events.append("call:ww_split")
    split = ww_split(
        state["ww_e"],
        state["ww_i"],
        state["u"],
        state["v"],
        state["ww"],
        state["mut"],
        state["rdnw"],
        state["msfty"],
        state["c1f"],
        state["c2f"],
        rdx,
        rdy,
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvy"],
        dt,
        config_flags,
        rk_step,
        1,
        int(ide),
        1,
        int(jde),
        1,
        int(kde),
        int(ims),
        int(ime),
        int(jms),
        int(jme),
        int(kms),
        int(kme),
        int(its),
        int(ite),
        int(jts),
        int(jte),
        int(kts),
        int(kte),
    )
    state["ww_e"] = split.ww_e
    state["ww_i"] = split.ww_i

    ieva = chk_ieva(config_flags, rk_step)
    if events_enabled:
        events.append("call:chk_ieva")
    if ieva:
        for name in ("calculate_full", "calc_mut_new", "calc_mu_uv_1:old", "calc_mu_uv_1:new"):
            call_stage(name)

    final_weno_mom = int(rk_step) == rk_order and int(adv_opt) == WENO_MOM
    call_stage("advect_weno_u" if final_weno_mom else "advect_u")
    if ieva:
        call_stage("advect_u_implicit")

    call_stage("advect_weno_v" if final_weno_mom else "advect_v")
    if ieva:
        call_stage("advect_v_implicit")

    if bool(non_hydrostatic):
        call_stage("advect_weno_w" if final_weno_mom else "advect_w")

    scalar_weno = (
        int(_read(config_flags, "scalar_adv_opt", 0)) in {WENO_SCALAR, WENOPD_SCALAR}
        or int(_read(config_flags, "moist_adv_opt", 0)) in {WENO_SCALAR, WENOPD_SCALAR}
    ) and int(rk_step) == rk_order
    call_stage("advect_scalar_weno" if scalar_weno else "advect_scalar")
    if ieva:
        call_stage("advect_s_implicit")

    if int(_read(config_flags, "cu_physics", 0)) in {GDSCHEME, GFSCHEME, G3SCHEME, NTIEDTKESCHEME}:
        call_stage("set_tend")

    call_stage("rhs_ph")
    if ieva:
        call_stage("advect_ph_implicit")
        if bool(non_hydrostatic):
            call_stage("advect_w_implicit")

    call_stage("horizontal_pressure_gradient")
    if bool(non_hydrostatic):
        call_stage("pg_buoy_w")

    call_stage("w_damp")
    if bool(_read(config_flags, "pert_coriolis", False)):
        call_stage("perturbation_coriolis")
    else:
        call_stage("coriolis")
    call_stage("curvature")

    if int(_read(config_flags, "ra_lw_physics", 0)) == HELDSUAREZ:
        call_stage("held_suarez_damp")

    if int(rk_step) == 1:
        if int(_read(config_flags, "diff_opt", 0)) == 1:
            for name in ("horizontal_diffusion:u", "horizontal_diffusion:v"):
                call_stage(name)
            if bool(non_hydrostatic):
                call_stage("horizontal_diffusion:w")
            call_stage("horizontal_diffusion_3dmp:m")
            if int(_read(config_flags, "bl_pbl_physics", 0)) == 0:
                for name in ("vertical_diffusion_u", "vertical_diffusion_v"):
                    call_stage(name)
                if bool(non_hydrostatic):
                    call_stage("vertical_diffusion:w")
                call_stage("vertical_diffusion_3dmp")
        if int(diff_6th_opt) != 0:
            for name in ("sixth_order_diffusion:u", "sixth_order_diffusion:v"):
                call_stage(name)
            if bool(non_hydrostatic):
                call_stage("sixth_order_diffusion:w")
            call_stage("sixth_order_diffusion:m")
        if int(damp_opt) == 2:
            call_stage("rk_rayleigh_damp")
        if int(rad_nudge) == 1:
            call_stage("theta_relaxation")

    if bool(_read(config_flags, "use_adaptive_time_step", False)):
        max_vert_cfl = float(state.get("max_vert_cfl", 0.0))
        max_horiz_cfl = float(state.get("max_horiz_cfl", 0.0))
    else:
        max_vert_cfl = 0.0
        max_horiz_cfl = 0.0
    state.setdefault("time_step", time_step)

    return RkTendencyResult(
        fields=state,
        events=tuple(events) if events_enabled else (),
        dt_step=float(dt_step),
        ieva=bool(ieva),
        max_vert_cfl=max_vert_cfl,
        max_horiz_cfl=max_horiz_cfl,
    )


def rk_addtend_dry(
    ru_tend: Any,
    rv_tend: Any,
    rw_tend: Any,
    ph_tend: Any,
    t_tend: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    rw_tendf: Any,
    ph_tendf: Any,
    t_tendf: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    ph_save: Any,
    t_save: Any,
    mu_tend: Any,
    mu_tendf: Any,
    rk_step: int,
    c1: Any,
    c2: Any,
    h_diabatic: Any,
    mut: Any,
    msftx: Any,
    msfty: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    kps: int,
    kpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> tuple[Any, ...]:
    """Return WRF ``rk_addtend_dry`` updated dry RK tendency arrays."""

    del kds, kde, msftx, msfux, msfvy, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe

    ru_tend_out = jnp.asarray(ru_tend)
    rv_tend_out = jnp.asarray(rv_tend)
    rw_tend_out = jnp.asarray(rw_tend)
    ph_tend_out = jnp.asarray(ph_tend)
    t_tend_out = jnp.asarray(t_tend)
    ru_tendf_out = jnp.asarray(ru_tendf)
    rv_tendf_out = jnp.asarray(rv_tendf)
    rw_tendf_out = jnp.asarray(rw_tendf)
    ph_tendf_out = jnp.asarray(ph_tendf)
    t_tendf_out = jnp.asarray(t_tendf)
    mu_tend_out = jnp.asarray(mu_tend)
    mu_tendf_out = jnp.asarray(mu_tendf)

    u_save_arr = jnp.asarray(u_save)
    v_save_arr = jnp.asarray(v_save)
    w_save_arr = jnp.asarray(w_save)
    ph_save_arr = jnp.asarray(ph_save)
    t_save_arr = jnp.asarray(t_save)
    h_diabatic_arr = jnp.asarray(h_diabatic)
    mut_arr = jnp.asarray(mut)
    msfty_arr = jnp.asarray(msfty)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvx_inv_arr = jnp.asarray(msfvx_inv)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)

    def store_real(value: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(value, dtype=jnp.float32)
        return _store_wrf_real(value, rounding_token)

    def add(left: Any, right: Any) -> Any:
        return store_real(jnp.asarray(left, dtype=jnp.float32) + jnp.asarray(right, dtype=jnp.float32))

    def multiply(left: Any, right: Any) -> Any:
        return store_real(jnp.asarray(left, dtype=jnp.float32) * jnp.asarray(right, dtype=jnp.float32))

    def divide(left: Any, right: Any) -> Any:
        return store_real(jnp.asarray(left, dtype=jnp.float32) / jnp.asarray(right, dtype=jnp.float32))

    def slice3(i_start: int, i_end: int, k_start: int, k_end: int, j_start: int, j_end: int):
        if int(i_end) < int(i_start) or int(k_end) < int(k_start) or int(j_end) < int(j_start):
            return None
        i_slice = slice(int(i_start) - int(ims), int(i_end) - int(ims) + 1)
        k_slice = slice(int(k_start) - int(kms), int(k_end) - int(kms) + 1)
        j_slice = slice(int(j_start) - int(jms), int(j_end) - int(jms) + 1)
        return i_slice, k_slice, j_slice

    ru_idx = slice3(int(its), int(ite), int(kts), int(kte) - 1, int(jts), min(int(jte), int(jde) - 1))
    if ru_idx is not None:
        i_slice, k_slice, j_slice = ru_idx
        idx = (i_slice, k_slice, j_slice)
        metric = msfuy_arr[i_slice, j_slice][:, None, :]
        if int(rk_step) == 1:
            ru_tendf_out = ru_tendf_out.at[idx].set(
                add(ru_tendf_out[idx], multiply(u_save_arr[idx], metric))
            )
        ru_tend_out = ru_tend_out.at[idx].set(
            add(ru_tend_out[idx], divide(ru_tendf_out[idx], metric))
        )

    rv_idx = slice3(int(its), min(int(ite), int(ide) - 1), int(kts), int(kte) - 1, int(jts), int(jte))
    if rv_idx is not None:
        i_slice, k_slice, j_slice = rv_idx
        idx = (i_slice, k_slice, j_slice)
        metric_inv = msfvx_inv_arr[i_slice, j_slice][:, None, :]
        metric = msfvx_arr[i_slice, j_slice][:, None, :]
        if int(rk_step) == 1:
            rv_tendf_out = rv_tendf_out.at[idx].set(
                add(rv_tendf_out[idx], multiply(v_save_arr[idx], metric))
            )
        rv_tend_out = rv_tend_out.at[idx].set(
            add(rv_tend_out[idx], multiply(rv_tendf_out[idx], metric_inv))
        )

    rw_idx = slice3(int(its), min(int(ite), int(ide) - 1), int(kts), int(kte), int(jts), min(int(jte), int(jde) - 1))
    if rw_idx is not None:
        i_slice, k_slice, j_slice = rw_idx
        idx = (i_slice, k_slice, j_slice)
        metric = msfty_arr[i_slice, j_slice][:, None, :]
        if int(rk_step) == 1:
            rw_tendf_out = rw_tendf_out.at[idx].set(
                add(rw_tendf_out[idx], multiply(w_save_arr[idx], metric))
            )
        rw_tend_out = rw_tend_out.at[idx].set(
            add(rw_tend_out[idx], divide(rw_tendf_out[idx], metric))
        )
        if int(rk_step) == 1:
            ph_tendf_out = ph_tendf_out.at[idx].set(add(ph_tendf_out[idx], ph_save_arr[idx]))
        ph_tend_out = ph_tend_out.at[idx].set(
            add(ph_tend_out[idx], divide(ph_tendf_out[idx], metric))
        )

    t_idx = slice3(int(its), min(int(ite), int(ide) - 1), int(kts), int(kte) - 1, int(jts), min(int(jte), int(jde) - 1))
    if t_idx is not None:
        i_slice, k_slice, j_slice = t_idx
        idx = (i_slice, k_slice, j_slice)
        metric = msfty_arr[i_slice, j_slice][:, None, :]
        mass = add(
            multiply(c1_arr[k_slice][None, :, None], mut_arr[i_slice, j_slice][:, None, :]),
            c2_arr[k_slice][None, :, None],
        )
        if int(rk_step) == 1:
            t_tendf_out = t_tendf_out.at[idx].set(add(t_tendf_out[idx], t_save_arr[idx]))
        heating = divide(multiply(mass, h_diabatic_arr[idx]), metric)
        t_tend_out = t_tend_out.at[idx].set(
            add(add(t_tend_out[idx], divide(t_tendf_out[idx], metric)), heating)
        )

    i_end = min(int(ite), int(ide) - 1)
    j_end = min(int(jte), int(jde) - 1)
    if i_end >= int(its) and j_end >= int(jts):
        idx2 = (slice(int(its) - int(ims), i_end - int(ims) + 1), slice(int(jts) - int(jms), j_end - int(jms) + 1))
        mu_tend_out = mu_tend_out.at[idx2].set(add(mu_tend_out[idx2], mu_tendf_out[idx2]))

    return (
        ru_tend_out,
        rv_tend_out,
        rw_tend_out,
        ph_tend_out,
        t_tend_out,
        ru_tendf_out,
        rv_tendf_out,
        rw_tendf_out,
        ph_tendf_out,
        t_tendf_out,
        mu_tend_out,
        mu_tendf_out,
    )


def rk_scalar_tend(
    scs: int,
    sce: int,
    config_flags: Any,
    tenddec: bool,
    rk_step: int,
    dt_step: float,
    ru: Any,
    rv: Any,
    ww: Any,
    ww_e: Any,
    ww_i: Any,
    u_old: Any,
    v_old: Any,
    mut: Any,
    mub: Any,
    mu_old: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    alt: Any,
    scalar_old: Any,
    scalar: Any,
    scalar_tends: Any,
    advect_tend: Any,
    h_tendency: Any,
    z_tendency: Any,
    rqvften: Any,
    base: Any,
    moist_step: bool,
    fnm: Any,
    fnp: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    rdx: float,
    rdy: float,
    rdn: Any,
    rdnw: Any,
    khdif: float,
    kvdif: float,
    xkmhd: Any,
    diff_6th_opt: int,
    diff_6th_factor: float,
    adv_opt: int,
    phb: Any,
    ph: Any,
    mix2_off: bool,
    mix6_off: bool,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    *,
    time_step: int | None = None,
) -> RkScalarTendResult:
    """Run WRF ``rk_scalar_tend`` for translated scalar-advection paths."""

    del alt, base, rdn, khdif, kvdif, xkmhd, diff_6th_factor, phb, ph, msfvx_inv

    rk_order = int(_read(config_flags, "rk_ord", int(rk_step)))
    dt = float(dt_step) * (rk_order - int(rk_step) + 1)
    if time_step is None:
        time_step = int(_read(config_flags, "time_step", 1))

    split = ww_split(
        ww_e,
        ww_i,
        u_old,
        v_old,
        ww,
        mut,
        rdnw,
        msfty,
        c1f,
        c2f,
        rdx,
        rdy,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        dt,
        config_flags,
        rk_step,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    ww_e_out = split.ww_e
    ww_i_out = split.ww_i

    ieva = chk_ieva(config_flags, rk_step)
    if ieva:
        mut_old = jnp.asarray(mub) + jnp.asarray(mu_old)
    else:
        mut_old = None

    scalar_tends_out = jnp.asarray(scalar_tends)
    advect_tend_out = jnp.asarray(advect_tend)
    h_tendency_out = jnp.asarray(h_tendency)
    z_tendency_out = jnp.asarray(z_tendency)
    rqvften_out = jnp.asarray(rqvften)

    def zero3(field: Any) -> Any:
        return zero_tend(field, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

    cu_triggers = {GDSCHEME, G3SCHEME, GFSCHEME, KFETASCHEME, MSKFSCHEME, TIEDTKESCHEME, NTIEDTKESCHEME}

    def set_tend_like(field_adv_tend: Any) -> Any:
        out = rqvften_out
        i_fortran = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1)
        k_fortran = jnp.arange(int(kts), min(int(kte), int(kde) - 1) + 1)
        j_fortran = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1)
        if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
            return out
        ii = _idx(i_fortran, ims)
        kk = _idx(k_fortran, kms)
        jj = _idx(j_fortran, jms)
        idx = (ii[:, None, None], kk[None, :, None], jj[None, None, :])
        metric = jnp.asarray(msfty)[ii[:, None, None], jj[None, None, :]]
        return out.at[idx].set(jnp.asarray(field_adv_tend)[idx] * metric)

    for im in range(int(scs), int(sce) + 1):
        im_idx = im - int(scs)
        advect_tend_out = zero3(advect_tend_out)
        h_tendency_out = zero3(h_tendency_out)
        z_tendency_out = zero3(z_tendency_out)

        final_rk = int(rk_step) == rk_order
        if final_rk and int(adv_opt) == POSITIVEDEF:
            advect_tend_out, h_tendency_out, z_tendency_out = advect_scalar_pd(
                jnp.asarray(scalar)[:, :, :, im_idx],
                jnp.asarray(scalar_old)[:, :, :, im_idx],
                advect_tend_out,
                h_tendency_out,
                z_tendency_out,
                ru,
                rv,
                ww_e_out,
                c1h,
                c2h,
                mut,
                mub,
                mu_old,
                int(time_step),
                config_flags,
                tenddec,
                msfux,
                msfuy,
                msfvx,
                msfvy,
                msftx,
                msfty,
                fnm,
                fnp,
                rdx,
                rdy,
                rdnw,
                dt_step,
                ids,
                ide,
                jds,
                jde,
                kds,
                kde,
                ims,
                ime,
                jms,
                jme,
                kms,
                kme,
                its,
                ite,
                jts,
                jte,
                kts,
                kte,
            )
        elif final_rk and int(adv_opt) in {MONOTONIC, WENO_SCALAR, WENOPD_SCALAR}:
            raise NotImplementedError("rk_scalar_tend does not yet cover monotonic/WENO scalar advection")
        else:
            advect_tend_out = advect_scalar(
                jnp.asarray(scalar)[:, :, :, im_idx],
                jnp.asarray(scalar)[:, :, :, im_idx],
                advect_tend_out,
                ru,
                rv,
                ww_e_out,
                c1h,
                c2h,
                mut,
                int(time_step),
                config_flags,
                msfux,
                msfuy,
                msfvx,
                msfvy,
                msftx,
                msfty,
                fnm,
                fnp,
                rdx,
                rdy,
                rdnw,
                ids,
                ide,
                jds,
                jde,
                kds,
                kde,
                ims,
                ime,
                jms,
                jme,
                kms,
                kme,
                its,
                ite,
                jts,
                jte,
                kts,
                kte,
            )

        if ieva:
            del mut_old
            raise NotImplementedError("rk_scalar_tend does not yet cover implicit vertical scalar advection")

        if int(_read(config_flags, "cu_physics", 0)) in cu_triggers and bool(moist_step) and im == P_QV:
            rqvften_out = set_tend_like(advect_tend_out)

        if int(rk_step) == 1:
            if int(_read(config_flags, "diff_opt", 0)) == 1:
                if not bool(mix2_off):
                    raise NotImplementedError("rk_scalar_tend does not yet cover first-order horizontal diffusion")
                if int(_read(config_flags, "bl_pbl_physics", 0)) == 0:
                    raise NotImplementedError("rk_scalar_tend does not yet cover vertical diffusion")
            if int(diff_6th_opt) != 0 and not bool(mix6_off):
                raise NotImplementedError("rk_scalar_tend does not yet cover sixth-order diffusion")

    return RkScalarTendResult(
        scalar_tends=scalar_tends_out,
        advect_tend=advect_tend_out,
        h_tendency=h_tendency_out,
        z_tendency=z_tendency_out,
        rqvften=rqvften_out,
        ww_e=ww_e_out,
        ww_i=ww_i_out,
    )


def rk_update_scalar(
    scs: int,
    sce: int,
    scalar_1: Any,
    scalar_2: Any,
    sc_tend: Any,
    advh_t: Any | None,
    advz_t: Any | None,
    advect_tend: Any,
    h_tendency: Any,
    z_tendency: Any,
    msftx: Any,
    msfty: Any,
    c1: Any,
    c2: Any,
    mu_old: Any,
    mu_new: Any,
    mu_base: Any,
    rk_step: int,
    dt: float,
    spec_zone: int,
    config_flags: Any,
    tenddec: bool,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any | None, Any | None]:
    """Return WRF ``rk_update_scalar`` updated scalar and optional diagnostics."""

    del kds, msftx, ime, jme, kme

    scalar_1_out = jnp.asarray(scalar_1)
    scalar_2_out = jnp.asarray(scalar_2)
    sc_tend_arr = jnp.asarray(sc_tend)
    advect_tend_arr = jnp.asarray(advect_tend)
    h_tendency_arr = jnp.asarray(h_tendency)
    z_tendency_arr = jnp.asarray(z_tendency)
    msfty_arr = jnp.asarray(msfty)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mu_old_arr = jnp.asarray(mu_old)
    mu_new_arr = jnp.asarray(mu_new)
    mu_base_arr = jnp.asarray(mu_base)
    advh_t_out = None if advh_t is None else jnp.asarray(advh_t)
    advz_t_out = None if advz_t is None else jnp.asarray(advz_t)

    def store(value: Any) -> Any:
        array = jnp.asarray(value, dtype=scalar_2_out.dtype)
        if rounding_token is None:
            return array
        return _store_wrf_real(array, rounding_token)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = int(kte) - 1

    i_start_spc = i_start
    i_end_spc = i_end
    j_start_spc = j_start
    j_end_spc = j_end
    k_start_spc = k_start
    k_end_spc = k_end

    if bool(_read(config_flags, "nested", False)) or bool(_read(config_flags, "specified", False)):
        if not bool(_read(config_flags, "periodic_x", False)):
            i_start = max(int(its), int(ids) + int(spec_zone))
            i_end = min(int(ite), int(ide) - int(spec_zone) - 1)
        j_start = max(int(jts), int(jds) + int(spec_zone))
        j_end = min(int(jte), int(jde) - int(spec_zone) - 1)
        k_start = int(kts)
        k_end = min(int(kte), int(kde) - 1)

    full_i_end = min(int(ite), int(ide) - 1)
    full_j_end = min(int(jte), int(jde) - 1)
    full_k_end = min(int(kte), int(kde) - 1)
    if int(its) > full_i_end or int(jts) > full_j_end or int(kts) > full_k_end:
        return scalar_1_out, scalar_2_out, advh_t_out, advz_t_out

    fii = slice(int(its) - int(ims), full_i_end - int(ims) + 1)
    fkk = slice(int(kts) - int(kms), full_k_end - int(kms) + 1)
    fjj = slice(int(jts) - int(jms), full_j_end - int(jms) + 1)
    full_idx3 = (fii, fkk, fjj)
    muold = store(mu_old_arr[fii, fjj] + mu_base_arr[fii, fjj])
    munew = store(mu_new_arr[fii, fjj] + mu_base_arr[fii, fjj])

    def add_advect_tendency(tendency: Any) -> Any:
        if i_start > i_end or k_start > k_end or j_start > j_end:
            return tendency
        ii = slice(i_start - int(ims), i_end - int(ims) + 1)
        kk = slice(k_start - int(kms), k_end - int(kms) + 1)
        jj = slice(j_start - int(jms), j_end - int(jms) + 1)
        idx = (ii, kk, jj)
        values = store(advect_tend_arr[idx] * msfty_arr[ii, jj][:, None, :])
        return tendency.at[idx].set(values)

    def add_scalar_tendency(tendency: Any, scalar_index: int) -> Any:
        if i_start_spc > i_end_spc or k_start_spc > k_end_spc or j_start_spc > j_end_spc:
            return tendency
        ii = slice(i_start_spc - int(ims), i_end_spc - int(ims) + 1)
        kk = slice(k_start_spc - int(kms), k_end_spc - int(kms) + 1)
        jj = slice(j_start_spc - int(jms), j_end_spc - int(jms) + 1)
        idx = (ii, kk, jj)
        idx4 = idx + (int(scalar_index) - int(scs),)
        return tendency.at[idx].set(store(tendency[idx] + sc_tend_arr[idx4]))

    for im in range(int(scs), int(sce) + 1):
        im_idx = im - int(scs)
        tendency = jnp.zeros_like(advect_tend_arr)
        tendency = add_advect_tendency(tendency)
        tendency = add_scalar_tendency(tendency, im)
        idx4 = full_idx3 + (im_idx,)
        if int(rk_step) == 1:
            scalar_1_out = scalar_1_out.at[idx4].set(scalar_2_out[idx4])
            old_scalar = scalar_2_out[idx4]
        else:
            old_scalar = scalar_1_out[idx4]
        old_mass_product = store(c1_arr[fkk][None, :, None] * muold[:, None, :])
        old_mass = store(old_mass_product + c2_arr[fkk][None, :, None])
        new_mass_product = store(c1_arr[fkk][None, :, None] * munew[:, None, :])
        new_mass = store(new_mass_product + c2_arr[fkk][None, :, None])
        old_coupled = store(old_mass * old_scalar)
        tendency_dt = store(jnp.asarray(dt, dtype=scalar_2_out.dtype) * tendency[full_idx3])
        numerator = store(old_coupled + tendency_dt)
        updated = store(numerator / new_mass)
        scalar_2_out = scalar_2_out.at[idx4].set(updated)

    if (
        int(rk_step) != 1
        and advh_t_out is not None
        and advz_t_out is not None
        and bool(tenddec)
        and int(rk_step) == int(_read(config_flags, "rk_ord", int(rk_step)))
    ):
        species_count = max(0, int(sce) - int(scs) + 1)
        metric = msfty_arr[fii, fjj][:, None, :]
        advh_t_out = advh_t_out.at[full_idx3].set(
            advh_t_out[full_idx3] + species_count * float(dt) * h_tendency_arr[full_idx3] * metric / new_mass
        )
        advz_t_out = advz_t_out.at[full_idx3].set(
            advz_t_out[full_idx3] + species_count * float(dt) * z_tendency_arr[full_idx3] * metric / new_mass
        )

    return scalar_1_out, scalar_2_out, advh_t_out, advz_t_out


def rk_update_scalar_pd(
    scs: int,
    sce: int,
    scalar: Any,
    sc_tend: Any,
    c1: Any,
    c2: Any,
    mu_old: Any,
    mu_new: Any,
    mu_base: Any,
    rk_step: int,
    dt: float,
    spec_zone: int,
    config_flags: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[Any, Any]:
    """Return WRF ``rk_update_scalar_pd`` updated scalar and cleared tendencies."""

    del rk_step, spec_zone, config_flags, ids, jds, kds, ime, jme, kme

    scalar_out = jnp.asarray(scalar)
    sc_tend_out = jnp.asarray(sc_tend)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mu_old_arr = jnp.asarray(mu_old)
    mu_new_arr = jnp.asarray(mu_new)
    mu_base_arr = jnp.asarray(mu_base)

    i_fortran = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1)
    k_fortran = jnp.arange(int(kts), min(int(kte), int(kde) - 1) + 1)
    j_fortran = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return scalar_out, sc_tend_out

    ii = _idx(i_fortran, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)
    idx3 = (ii[:, None, None], kk[None, :, None], jj[None, None, :])
    muold = mu_old_arr[ii[:, None], jj[None, :]] + mu_base_arr[ii[:, None], jj[None, :]]
    munew = mu_new_arr[ii[:, None], jj[None, :]] + mu_base_arr[ii[:, None], jj[None, :]]
    old_mass = c1_arr[kk][None, :, None] * muold[:, None, :] + c2_arr[kk][None, :, None]
    new_mass = c1_arr[kk][None, :, None] * munew[:, None, :] + c2_arr[kk][None, :, None]

    for im in range(int(scs), int(sce) + 1):
        im_idx = im - int(scs)
        idx4 = idx3 + (im_idx,)
        tendency = sc_tend_out[idx4]
        sc_tend_out = sc_tend_out.at[idx4].set(0.0)
        updated = (old_mass * scalar_out[idx4] + float(dt) * tendency) / new_mass
        scalar_out = scalar_out.at[idx4].set(updated)

    return scalar_out, sc_tend_out


__all__ = [
    "PARAM_FIRST_SCALAR",
    "POSITIVEDEF",
    "P_QC",
    "P_QG",
    "P_QI",
    "P_QR",
    "P_QS",
    "P_QV",
    "InitZeroTendencyResult",
    "RkScalarTendResult",
    "RkStepPrepResult",
    "RkTendencyResult",
    "init_zero_tendency",
    "rk_step_prep",
    "calculate_phy_tend",
    "rk_tendency",
    "rk_addtend_dry",
    "rk_scalar_tend",
    "rk_update_scalar",
    "rk_update_scalar_pd",
]
