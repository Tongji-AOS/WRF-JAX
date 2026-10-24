"""Kernels translated from WRF ``dyn_em/module_bc_em``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jax import lax
import jax.numpy as jnp
from wrf_jax.core.real_arithmetic import real_assignment

from wrf_jax.boundary.physical_bc import set_physical_bc2d, set_physical_bc3d
from wrf_jax.boundary.relax_bdytend import relax_bdytend
from wrf_jax.boundary.relax_bdytend_tile import relax_bdytend_tile
from wrf_jax.boundary.spec_bdytend import spec_bdytend


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    """Preserve a WRF REAL array assignment inside a larger compiled graph."""

    return real_assignment(value, rounding_token)


class RkPhysBcDry1Result:
    def __init__(self, *, u: Any, v: Any, rw: Any, w: Any, muu: Any, muv: Any, mut: Any, php: Any, alt: Any, p: Any):
        self.u = u
        self.v = v
        self.rw = rw
        self.w = w
        self.muu = muu
        self.muv = muv
        self.mut = mut
        self.php = php
        self.alt = alt
        self.p = p


class RkPhysBcDry2Result:
    def __init__(self, *, u: Any, v: Any, w: Any, t: Any, ph: Any, mu: Any):
        self.u = u
        self.v = v
        self.w = w
        self.t = t
        self.ph = ph
        self.mu = mu


class SetPhysBcDry2Result:
    def __init__(
        self,
        *,
        u_1: Any,
        u_2: Any,
        v_1: Any,
        v_2: Any,
        w_1: Any,
        w_2: Any,
        t_1: Any,
        t_2: Any,
        ph_1: Any,
        ph_2: Any,
        mu_1: Any,
        mu_2: Any,
    ):
        self.u_1 = u_1
        self.u_2 = u_2
        self.v_1 = v_1
        self.v_2 = v_2
        self.w_1 = w_1
        self.w_2 = w_2
        self.t_1 = t_1
        self.t_2 = t_2
        self.ph_1 = ph_1
        self.ph_2 = ph_2
        self.mu_1 = mu_1
        self.mu_2 = mu_2


class SpecBdyDryResult:
    def __init__(self, *, ru_tend: Any, rv_tend: Any, ph_tend: Any, t_tend: Any, rw_tend: Any, mu_tend: Any):
        self.ru_tend = ru_tend
        self.rv_tend = rv_tend
        self.ph_tend = ph_tend
        self.t_tend = t_tend
        self.rw_tend = rw_tend
        self.mu_tend = mu_tend


class RelaxBdyDryResult:
    def __init__(self, *, ru_tendf: Any, rv_tendf: Any, ph_tendf: Any, t_tendf: Any, rw_tendf: Any, mu_tend: Any):
        self.ru_tendf = ru_tendf
        self.rv_tendf = rv_tendf
        self.ph_tendf = ph_tendf
        self.t_tendf = t_tendf
        self.rw_tendf = rw_tendf
        self.mu_tend = mu_tend


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(values: Any, lower_bound: int) -> Any:
    return values - int(lower_bound)


def rk_phys_bc_dry_1(
    config_flags: Any,
    u: Any,
    v: Any,
    rw: Any,
    w: Any,
    muu: Any,
    muv: Any,
    mut: Any,
    php: Any,
    alt: Any,
    p: Any,
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
    *,
    set_physical_bc3d_fn=set_physical_bc3d,
    set_physical_bc2d_fn=set_physical_bc2d,
) -> RkPhysBcDry1Result:
    """Apply WRF RK dry-state physical boundary conditions for state group 1."""

    def bc3(dat: Any, variable: str) -> Any:
        return set_physical_bc3d_fn(
            dat,
            variable,
            config_flags,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )

    def bc2(dat: Any, variable: str) -> Any:
        return set_physical_bc2d_fn(
            dat,
            variable,
            config_flags,
            ids,
            ide,
            jds,
            jde,
            ims,
            ime,
            jms,
            jme,
            ips,
            ipe,
            jps,
            jpe,
            its,
            ite,
            jts,
            jte,
        )

    return RkPhysBcDry1Result(
        u=bc3(u, "u"),
        v=bc3(v, "v"),
        rw=bc3(rw, "w"),
        w=bc3(w, "w"),
        php=bc3(php, "w"),
        alt=bc3(alt, "t"),
        p=bc3(p, "p"),
        muu=bc2(muu, "u"),
        muv=bc2(muv, "v"),
        mut=bc2(mut, "t"),
    )


def rk_phys_bc_dry_2(
    config_flags: Any,
    u: Any,
    v: Any,
    w: Any,
    t: Any,
    ph: Any,
    mu: Any,
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
    *,
    set_physical_bc3d_fn=set_physical_bc3d,
    set_physical_bc2d_fn=set_physical_bc2d,
) -> RkPhysBcDry2Result:
    """Apply WRF RK dry-state physical boundary conditions for state group 2."""

    def bc3(dat: Any, variable: str) -> Any:
        return set_physical_bc3d_fn(
            dat,
            variable,
            config_flags,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )

    def bc2(dat: Any, variable: str) -> Any:
        return set_physical_bc2d_fn(
            dat,
            variable,
            config_flags,
            ids,
            ide,
            jds,
            jde,
            ims,
            ime,
            jms,
            jme,
            ips,
            ipe,
            jps,
            jpe,
            its,
            ite,
            jts,
            jte,
        )

    return RkPhysBcDry2Result(
        u=bc3(u, "U"),
        v=bc3(v, "V"),
        w=bc3(w, "w"),
        t=bc3(t, "p"),
        ph=bc3(ph, "w"),
        mu=bc2(mu, "t"),
    )


def set_phys_bc_dry_2(
    config_flags: Any,
    u_1: Any,
    u_2: Any,
    v_1: Any,
    v_2: Any,
    w_1: Any,
    w_2: Any,
    t_1: Any,
    t_2: Any,
    ph_1: Any,
    ph_2: Any,
    mu_1: Any,
    mu_2: Any,
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
    *,
    set_physical_bc3d_fn=set_physical_bc3d,
    set_physical_bc2d_fn=set_physical_bc2d,
) -> SetPhysBcDry2Result:
    """Apply WRF dry physical boundary conditions for two time levels."""

    def bc3(dat: Any, variable: str) -> Any:
        return set_physical_bc3d_fn(
            dat,
            variable,
            config_flags,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )

    def bc2(dat: Any, variable: str) -> Any:
        return set_physical_bc2d_fn(
            dat,
            variable,
            config_flags,
            ids,
            ide,
            jds,
            jde,
            ims,
            ime,
            jms,
            jme,
            ips,
            ipe,
            jps,
            jpe,
            its,
            ite,
            jts,
            jte,
        )

    u_1_out = bc3(u_1, "U")
    u_2_out = bc3(u_2, "U")
    v_1_out = bc3(v_1, "V")
    v_2_out = bc3(v_2, "V")
    w_1_out = bc3(w_1, "w")
    w_2_out = bc3(w_2, "w")
    t_1_out = bc3(t_1, "p")
    t_2_out = bc3(t_2, "p")
    ph_1_out = bc3(ph_1, "w")
    ph_2_out = bc3(ph_2, "w")
    mu_1_out = bc2(mu_1, "t")
    mu_2_out = bc2(mu_2, "t")

    return SetPhysBcDry2Result(
        u_1=u_1_out,
        u_2=u_2_out,
        v_1=v_1_out,
        v_2=v_2_out,
        w_1=w_1_out,
        w_2=w_2_out,
        t_1=t_1_out,
        t_2=t_2_out,
        ph_1=ph_1_out,
        ph_2=ph_2_out,
        mu_1=mu_1_out,
        mu_2=mu_2_out,
    )


def set_w_surface(
    config_flags: Any,
    znw: Any,
    fill_w_flag: bool,
    w: Any,
    ht: Any,
    u: Any,
    v: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    rdx: float,
    rdy: float,
    msftx: Any,
    msfty: Any,
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
) -> Any:
    """Compute WRF's terrain-induced lower-boundary vertical velocity."""

    del kds, kde, ime, jme, kme
    w_out = jnp.asarray(w).copy()
    ht_arr = jnp.asarray(ht)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    znw_arr = jnp.asarray(znw)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    cf1_arr = jnp.asarray(cf1, dtype=w_out.dtype)
    cf2_arr = jnp.asarray(cf2, dtype=w_out.dtype)
    cf3_arr = jnp.asarray(cf3, dtype=w_out.dtype)
    rdx_arr = jnp.asarray(rdx, dtype=w_out.dtype)
    rdy_arr = jnp.asarray(rdy, dtype=w_out.dtype)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if i_end < i_start or j_end < j_start:
        return w_out

    im1_limit = int(ids) - 1 if bool(_read(config_flags, "periodic_x", False)) else int(ids)
    ip1_limit = int(ide) if bool(_read(config_flags, "periodic_x", False)) else int(ide) - 1
    jm1_limit = int(jds) - 1 if bool(_read(config_flags, "periodic_y", False)) else int(jds)
    jp1_limit = int(jde) if bool(_read(config_flags, "periodic_y", False)) else int(jde) - 1

    i_fortran = jnp.arange(i_start, i_end + 1)
    j_fortran = jnp.arange(j_start, j_end + 1)
    im1 = jnp.maximum(i_fortran - 1, im1_limit)
    ip1 = jnp.minimum(i_fortran + 1, ip1_limit)
    jm1 = jnp.maximum(j_fortran - 1, jm1_limit)
    jp1 = jnp.minimum(j_fortran + 1, jp1_limit)

    ii = _idx(i_fortran, ims)
    jj = _idx(j_fortran, jms)
    im1i = _idx(im1, ims)
    ip1i = _idx(ip1, ims)
    jm1j = _idx(jm1, jms)
    jp1j = _idx(jp1, jms)
    k1 = int(1) - int(kms)
    k2 = int(2) - int(kms)
    k3 = int(3) - int(kms)

    i2 = ii[:, None]
    j2 = jj[None, :]
    ip1_2 = ip1i[:, None]
    im1_2 = im1i[:, None]
    jp1_2 = jp1j[None, :]
    jm1_2 = jm1j[None, :]

    v_jp1 = cf1_arr * v_arr[i2, k1, jp1_2] + cf2_arr * v_arr[i2, k2, jp1_2] + cf3_arr * v_arr[i2, k3, jp1_2]
    v_j = cf1_arr * v_arr[i2, k1, j2] + cf2_arr * v_arr[i2, k2, j2] + cf3_arr * v_arr[i2, k3, j2]
    u_ip1 = cf1_arr * u_arr[ip1_2, k1, j2] + cf2_arr * u_arr[ip1_2, k2, j2] + cf3_arr * u_arr[ip1_2, k3, j2]
    u_i = cf1_arr * u_arr[i2, k1, j2] + cf2_arr * u_arr[i2, k2, j2] + cf3_arr * u_arr[i2, k3, j2]

    y_part = msfty_arr[i2, j2] * 0.5 * rdy_arr * (
        (ht_arr[i2, jp1_2] - ht_arr[i2, j2]) * v_jp1
        + (ht_arr[i2, j2] - ht_arr[i2, jm1_2]) * v_j
    )
    x_part = msftx_arr[i2, j2] * 0.5 * rdx_arr * (
        (ht_arr[ip1_2, j2] - ht_arr[i2, j2]) * u_ip1
        + (ht_arr[i2, j2] - ht_arr[im1_2, j2]) * u_i
    )
    surface = y_part + x_part

    w_out = w_out.at[i2, k1, j2].set(surface)

    if fill_w_flag:
        k_fortran = jnp.arange(int(kts) + 1, int(kte) + 1)
        if k_fortran.size:
            kk = _idx(k_fortran, kms)
            filled = surface[:, None, :] * znw_arr[kk][None, :, None] * znw_arr[kk][None, :, None]
            w_out = w_out.at[ii[:, None, None], kk[None, :, None], jj[None, None, :]].set(filled)

    return w_out


def mass_weight(
    field: Any,
    mut: Any,
    c1: Any,
    c2: Any,
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
    irs: int,
    ire: int,
    jrs: int,
    jre: int,
    krs: int,
    kre: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> Any:
    """Return WRF ``mass_weight`` output ``rfield``."""

    del ids, ide, jds, jde, kds, kde, ime, jme, kme

    field_arr = jnp.asarray(field)
    mut_arr = jnp.asarray(mut)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    out_shape = (int(ire) - int(irs) + 1, int(kre) - int(krs) + 1, int(jre) - int(jrs) + 1)
    i_fortran = jnp.arange(int(irs), int(ire) + 1)
    k_fortran = jnp.arange(int(krs), int(kre) + 1)
    j_fortran = jnp.arange(int(jrs), int(jre) + 1)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return jnp.zeros(out_shape, dtype=field_arr.dtype)

    valid_i = (i_fortran >= int(its)) & (i_fortran <= int(ite))
    valid_k = (k_fortran >= int(kts)) & (k_fortran <= int(kte))
    valid_j = (j_fortran >= int(jts)) & (j_fortran <= int(jte))
    fi = jnp.clip(_idx(i_fortran, ims), 0, int(field_arr.shape[0]) - 1)
    fk = jnp.clip(_idx(k_fortran, kms), 0, int(field_arr.shape[1]) - 1)
    fj = jnp.clip(_idx(j_fortran, jms), 0, int(field_arr.shape[2]) - 1)

    field_block = field_arr[fi[:, None, None], fk[None, :, None], fj[None, None, :]]
    mut_block = mut_arr[fi[:, None], fj[None, :]]
    if rounding_token is None:
        weights = c1_arr[fk][None, :, None] * mut_block[:, None, :] + c2_arr[fk][None, :, None]
        weighted_field = field_block * weights
    else:
        mass_product = _store_wrf_real(c1_arr[fk][None, :, None] * mut_block[:, None, :], rounding_token)
        weights = _store_wrf_real(mass_product + c2_arr[fk][None, :, None], rounding_token)
        weighted_field = _store_wrf_real(field_block * weights, rounding_token)
    valid = valid_i[:, None, None] & valid_k[None, :, None] & valid_j[None, None, :]
    return jnp.where(valid, weighted_field, jnp.asarray(0.0, dtype=field_arr.dtype))


def relax_bdy_dry(
    config_flags: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    ph_tendf: Any,
    t_tendf: Any,
    rw_tendf: Any,
    mu_tend: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    ru: Any,
    rv: Any,
    ph: Any,
    t: Any,
    w: Any,
    mu: Any,
    mut: Any,
    u_bxs: Any,
    u_bxe: Any,
    u_bys: Any,
    u_bye: Any,
    v_bxs: Any,
    v_bxe: Any,
    v_bys: Any,
    v_bye: Any,
    ph_bxs: Any,
    ph_bxe: Any,
    ph_bys: Any,
    ph_bye: Any,
    t_bxs: Any,
    t_bxe: Any,
    t_bys: Any,
    t_bye: Any,
    w_bxs: Any,
    w_bxe: Any,
    w_bys: Any,
    w_bye: Any,
    mu_bxs: Any,
    mu_bxe: Any,
    mu_bys: Any,
    mu_bye: Any,
    u_btxs: Any,
    u_btxe: Any,
    u_btys: Any,
    u_btye: Any,
    v_btxs: Any,
    v_btxe: Any,
    v_btys: Any,
    v_btye: Any,
    ph_btxs: Any,
    ph_btxe: Any,
    ph_btys: Any,
    ph_btye: Any,
    t_btxs: Any,
    t_btxe: Any,
    t_btys: Any,
    t_btye: Any,
    w_btxs: Any,
    w_btxe: Any,
    w_btys: Any,
    w_btye: Any,
    mu_btxs: Any,
    mu_btxe: Any,
    mu_btys: Any,
    mu_btye: Any,
    spec_bdy_width: int,
    spec_zone: int,
    relax_zone: int,
    dtbc: float,
    fcx: Any,
    gcx: Any,
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
) -> RelaxBdyDryResult:
    """Return WRF dry-state lateral-boundary relaxation tendencies."""

    ru_out = relax_bdytend(
        ru,
        ru_tendf,
        u_bxs,
        u_bxe,
        u_bys,
        u_bye,
        u_btxs,
        u_btxe,
        u_btys,
        u_btye,
        "u",
        config_flags,
        spec_bdy_width,
        spec_zone,
        relax_zone,
        dtbc,
        fcx,
        gcx,
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
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token=rounding_token,
    )
    rv_out = relax_bdytend(
        rv,
        rv_tendf,
        v_bxs,
        v_bxe,
        v_bys,
        v_bye,
        v_btxs,
        v_btxe,
        v_btys,
        v_btye,
        "v",
        config_flags,
        spec_bdy_width,
        spec_zone,
        relax_zone,
        dtbc,
        fcx,
        gcx,
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
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token=rounding_token,
    )

    i_start = max(int(its) - 1, int(ids))
    i_end = min(int(ite) + 1, int(ide) - 1)
    j_start = max(int(jts) - 1, int(jds))
    j_end = min(int(jte) + 1, int(jde) - 1)

    def tile_relax(weighted_field: Any, field_tend: Any, bxs: Any, bxe: Any, bys: Any, bye: Any, btxs: Any, btxe: Any, btys: Any, btye: Any, variable: str) -> Any:
        return relax_bdytend_tile(
            weighted_field,
            field_tend,
            bxs,
            bxe,
            bys,
            bye,
            btxs,
            btxe,
            btys,
            btye,
            variable,
            config_flags,
            spec_bdy_width,
            spec_zone,
            relax_zone,
            dtbc,
            fcx,
            gcx,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
            int(its) - 1,
            int(ite) + 1,
            int(jts) - 1,
            int(jte) + 1,
            kts,
            kte,
            rounding_token=rounding_token,
        )

    rfield_ph = mass_weight(
        ph,
        mut,
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
        int(its) - 1,
        int(ite) + 1,
        int(jts) - 1,
        int(jte) + 1,
        kts,
        kte,
        i_start,
        i_end,
        j_start,
        j_end,
        kts,
        kte,
        rounding_token=rounding_token,
    )
    ph_out = tile_relax(rfield_ph, ph_tendf, ph_bxs, ph_bxe, ph_bys, ph_bye, ph_btxs, ph_btxe, ph_btys, ph_btye, "h")

    rfield_t = mass_weight(
        t,
        mut,
        c1h,
        c2h,
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
        int(its) - 1,
        int(ite) + 1,
        int(jts) - 1,
        int(jte) + 1,
        kts,
        kte,
        i_start,
        i_end,
        j_start,
        j_end,
        kts,
        int(kte) - 1,
        rounding_token=rounding_token,
    )
    t_out = tile_relax(rfield_t, t_tendf, t_bxs, t_bxe, t_bys, t_bye, t_btxs, t_btxe, t_btys, t_btye, "t")

    mu_3d = jnp.expand_dims(jnp.asarray(mu), axis=1)
    mu_tend_3d = jnp.expand_dims(jnp.asarray(mu_tend), axis=1)
    mu_out_3d = relax_bdytend(
        mu_3d,
        mu_tend_3d,
        mu_bxs,
        mu_bxe,
        mu_bys,
        mu_bye,
        mu_btxs,
        mu_btxe,
        mu_btys,
        mu_btye,
        "m",
        config_flags,
        spec_bdy_width,
        spec_zone,
        relax_zone,
        dtbc,
        fcx,
        gcx,
        ids,
        ide,
        jds,
        jde,
        1,
        1,
        ims,
        ime,
        jms,
        jme,
        1,
        1,
        ips,
        ipe,
        jps,
        jpe,
        1,
        1,
        its,
        ite,
        jts,
        jte,
        1,
        1,
        rounding_token=rounding_token,
    )
    mu_out = jnp.squeeze(mu_out_3d, axis=1)

    if bool(_read(config_flags, "nested", False)):
        rfield_w = mass_weight(
            w,
            mut,
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
            int(its) - 1,
            int(ite) + 1,
            int(jts) - 1,
            int(jte) + 1,
            kts,
            kte,
            i_start,
            i_end,
            j_start,
            j_end,
            kts,
            kte,
            rounding_token=rounding_token,
        )
        rw_out = tile_relax(rfield_w, rw_tendf, w_bxs, w_bxe, w_bys, w_bye, w_btxs, w_btxe, w_btys, w_btye, "h")
    else:
        rw_out = jnp.asarray(rw_tendf)

    return RelaxBdyDryResult(
        ru_tendf=ru_out,
        rv_tendf=rv_out,
        ph_tendf=ph_out,
        t_tendf=t_out,
        rw_tendf=rw_out,
        mu_tend=mu_out,
    )


def relax_bdy_scalar(
    scalar_tend: Any,
    scalar: Any,
    mu: Any,
    c1h: Any,
    c2h: Any,
    scalar_bxs: Any,
    scalar_bxe: Any,
    scalar_bys: Any,
    scalar_bye: Any,
    scalar_btxs: Any,
    scalar_btxe: Any,
    scalar_btys: Any,
    scalar_btye: Any,
    spec_bdy_width: int,
    spec_zone: int,
    relax_zone: int,
    dtbc: float,
    fcx: Any,
    gcx: Any,
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
) -> Any:
    """Return WRF scalar lateral-boundary relaxation tendencies."""

    i_start = max(int(its) - 1, int(ids))
    i_end = min(int(ite) + 1, int(ide) - 1)
    j_start = max(int(jts) - 1, int(jds))
    j_end = min(int(jte) + 1, int(jde) - 1)

    rscalar = mass_weight(
        scalar,
        mu,
        c1h,
        c2h,
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
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        i_start,
        i_end,
        j_start,
        j_end,
        kts,
        int(kte) - 1,
        rounding_token=rounding_token,
    )
    result = relax_bdytend(
        rscalar,
        scalar_tend,
        scalar_bxs,
        scalar_bxe,
        scalar_bys,
        scalar_bye,
        scalar_btxs,
        scalar_btxe,
        scalar_btys,
        scalar_btye,
        "q",
        config_flags,
        spec_bdy_width,
        spec_zone,
        relax_zone,
        dtbc,
        fcx,
        gcx,
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
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token=rounding_token,
    )
    return result


def spec_bdy_dry(
    config_flags: Any,
    u_bxs: Any,
    u_bxe: Any,
    u_bys: Any,
    u_bye: Any,
    v_bxs: Any,
    v_bxe: Any,
    v_bys: Any,
    v_bye: Any,
    ph_bxs: Any,
    ph_bxe: Any,
    ph_bys: Any,
    ph_bye: Any,
    t_bxs: Any,
    t_bxe: Any,
    t_bys: Any,
    t_bye: Any,
    w_bxs: Any,
    w_bxe: Any,
    w_bys: Any,
    w_bye: Any,
    mu_bxs: Any,
    mu_bxe: Any,
    mu_bys: Any,
    mu_bye: Any,
    u_btxs: Any,
    u_btxe: Any,
    u_btys: Any,
    u_btye: Any,
    v_btxs: Any,
    v_btxe: Any,
    v_btys: Any,
    v_btye: Any,
    ph_btxs: Any,
    ph_btxe: Any,
    ph_btys: Any,
    ph_btye: Any,
    t_btxs: Any,
    t_btxe: Any,
    t_btys: Any,
    t_btye: Any,
    w_btxs: Any,
    w_btxe: Any,
    w_btys: Any,
    w_btye: Any,
    mu_btxs: Any,
    mu_btxe: Any,
    mu_btys: Any,
    mu_btye: Any,
    spec_bdy_width: int,
    spec_zone: int,
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
) -> SpecBdyDryResult:
    """Return WRF dry-state specified-boundary tendencies."""

    def bdy3(bxs: Any, bxe: Any, bys: Any, bye: Any, btxs: Any, btxe: Any, btys: Any, btye: Any, variable: str) -> Any:
        return spec_bdytend(
            bxs,
            bxe,
            bys,
            bye,
            btxs,
            btxe,
            btys,
            btye,
            variable,
            config_flags,
            spec_bdy_width,
            spec_zone,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )

    ru_tend = bdy3(u_bxs, u_bxe, u_bys, u_bye, u_btxs, u_btxe, u_btys, u_btye, "u")
    rv_tend = bdy3(v_bxs, v_bxe, v_bys, v_bye, v_btxs, v_btxe, v_btys, v_btye, "v")
    ph_tend = bdy3(ph_bxs, ph_bxe, ph_bys, ph_bye, ph_btxs, ph_btxe, ph_btys, ph_btye, "h")
    t_tend = bdy3(t_bxs, t_bxe, t_bys, t_bye, t_btxs, t_btxe, t_btys, t_btye, "t")

    mu_tend_3d = spec_bdytend(
        mu_bxs,
        mu_bxe,
        mu_bys,
        mu_bye,
        mu_btxs,
        mu_btxe,
        mu_btys,
        mu_btye,
        "m",
        config_flags,
        spec_bdy_width,
        spec_zone,
        ids,
        ide,
        jds,
        jde,
        1,
        1,
        ims,
        ime,
        jms,
        jme,
        1,
        1,
        ips,
        ipe,
        jps,
        jpe,
        1,
        1,
        its,
        ite,
        jts,
        jte,
        1,
        1,
    )
    mu_tend = jnp.squeeze(mu_tend_3d, axis=1)

    if bool(_read(config_flags, "nested", False)):
        rw_tend = bdy3(w_bxs, w_bxe, w_bys, w_bye, w_btxs, w_btxe, w_btys, w_btye, "h")
    else:
        rw_tend = jnp.zeros_like(ph_tend)

    return SpecBdyDryResult(
        ru_tend=ru_tend,
        rv_tend=rv_tend,
        ph_tend=ph_tend,
        t_tend=t_tend,
        rw_tend=rw_tend,
        mu_tend=mu_tend,
    )


def spec_bdy_scalar(
    scalar_bxs: Any,
    scalar_bxe: Any,
    scalar_bys: Any,
    scalar_bye: Any,
    scalar_btxs: Any,
    scalar_btxe: Any,
    scalar_btys: Any,
    scalar_btye: Any,
    spec_bdy_width: int,
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
) -> Any:
    """Return WRF scalar specified-boundary tendencies."""

    return spec_bdytend(
        scalar_bxs,
        scalar_bxe,
        scalar_bys,
        scalar_bye,
        scalar_btxs,
        scalar_btxe,
        scalar_btys,
        scalar_btye,
        "q",
        config_flags,
        spec_bdy_width,
        spec_zone,
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
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )


def spec_bdyupdate_ph(
    ph_save: Any,
    field: Any,
    field_tend: Any,
    mu_tend: Any,
    muts: Any,
    c1: Any,
    c2: Any,
    dt: float,
    variable_in: str,
    config_flags: Any,
    spec_zone: int,
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
) -> Any:
    """Return ``field`` after WRF ``spec_bdyupdate_ph`` boundary updates."""

    del kds, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe

    ph_save_arr = jnp.asarray(ph_save)
    out = jnp.asarray(field).copy()
    tend = jnp.asarray(field_tend)
    mu_tend_arr = jnp.asarray(mu_tend)
    muts_arr = jnp.asarray(muts)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)

    variable = str(variable_in)[0]
    if variable == "U":
        variable = "u"
    if variable == "V":
        variable = "v"
    if variable == "M":
        variable = "m"
    if variable == "H":
        variable = "h"

    ibs = int(ids)
    ibe = int(ide) - 1
    itf = min(int(ite), int(ide) - 1)
    jbs = int(jds)
    jbe = int(jde) - 1
    jtf = min(int(jte), int(jde) - 1)
    ktf = int(kde) - 1
    if variable == "u":
        ibe = int(ide)
        itf = min(int(ite), int(ide))
    if variable == "v":
        jbe = int(jde)
        jtf = min(int(jte), int(jde))
    if variable in {"m", "h"}:
        ktf = int(kte)

    periodic_x = bool(_read(config_flags, "periodic_x", False))
    dt_value = float(dt)

    k_slice = slice(int(kts) - int(kms), ktf - int(kms) + 1)

    def i_slice(i0: int, i1: int) -> slice:
        return slice(i0 - int(ims), i1 - int(ims) + 1)

    def j_slice(j0: int, j1: int) -> slice:
        return slice(j0 - int(jms), j1 - int(jms) + 1)

    def update_j_edge(j: int, i0: int, i1: int) -> None:
        nonlocal out
        if i0 <= i1:
            ii = i_slice(i0, i1)
            jj = j - int(jms)
            muts_block = muts_arr[ii, jj][:, None]
            mu_tend_block = mu_tend_arr[ii, jj][:, None]
            c1_block = c1_arr[k_slice][None, :]
            c2_block = c2_arr[k_slice][None, :]
            field_block = out[ii, k_slice, jj]
            tend_block = tend[ii, k_slice, jj]
            ph_save_block = ph_save_arr[ii, k_slice, jj]
            if rounding_token is not None:
                # A standalone WRF subroutine receives already stored REAL
                # arrays.  Preserve that call boundary when XLA inlines this
                # update into the acoustic small-step graph; otherwise it can
                # fuse the producers of MUTS/PH into this expression and move
                # the last float32 rounding point.
                muts_block = _store_wrf_real(muts_block, rounding_token)
                mu_tend_block = _store_wrf_real(mu_tend_block, rounding_token)
                field_block = _store_wrf_real(field_block, rounding_token)
                tend_block = _store_wrf_real(tend_block, rounding_token)
                ph_save_block = _store_wrf_real(ph_save_block, rounding_token)
            mu_old = muts_block - dt_value * mu_tend_block
            denom = c1_block * muts_block + c2_block
            ratio = (c1_block * mu_old + c2_block) / denom
            value = (
                field_block * ratio
                + dt_value * tend_block / denom
                + ph_save_block * (ratio - 1.0)
            )
            out = out.at[ii, k_slice, jj].set(value)

    def update_i_edge(i: int, j0: int, j1: int) -> None:
        nonlocal out
        if j0 <= j1:
            ii = i - int(ims)
            jj = j_slice(j0, j1)
            muts_block = muts_arr[ii, jj][None, :]
            mu_tend_block = mu_tend_arr[ii, jj][None, :]
            c1_block = c1_arr[k_slice][:, None]
            c2_block = c2_arr[k_slice][:, None]
            field_block = out[ii, k_slice, jj]
            tend_block = tend[ii, k_slice, jj]
            ph_save_block = ph_save_arr[ii, k_slice, jj]
            if rounding_token is not None:
                muts_block = _store_wrf_real(muts_block, rounding_token)
                mu_tend_block = _store_wrf_real(mu_tend_block, rounding_token)
                field_block = _store_wrf_real(field_block, rounding_token)
                tend_block = _store_wrf_real(tend_block, rounding_token)
                ph_save_block = _store_wrf_real(ph_save_block, rounding_token)
            mu_old = muts_block - dt_value * mu_tend_block
            denom = c1_block * muts_block + c2_block
            ratio = (c1_block * mu_old + c2_block) / denom
            value = (
                field_block * ratio
                + dt_value * tend_block / denom
                + ph_save_block * (ratio - 1.0)
            )
            out = out.at[ii, k_slice, jj].set(value)

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            b_dist = j - jbs
            b_limit = 0 if periodic_x else b_dist
            update_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit))

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            b_dist = jbe - j
            b_limit = 0 if periodic_x else b_dist
            update_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit))

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                b_dist = i - ibs
                update_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1))

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                b_dist = ibe - i
                update_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1))

    return out


__all__ = [
    "RkPhysBcDry1Result",
    "RkPhysBcDry2Result",
    "SetPhysBcDry2Result",
    "SpecBdyDryResult",
    "RelaxBdyDryResult",
    "rk_phys_bc_dry_1",
    "rk_phys_bc_dry_2",
    "set_phys_bc_dry_2",
    "set_w_surface",
    "mass_weight",
    "relax_bdy_dry",
    "relax_bdy_scalar",
    "spec_bdy_dry",
    "spec_bdy_scalar",
    "spec_bdyupdate_ph",
]
