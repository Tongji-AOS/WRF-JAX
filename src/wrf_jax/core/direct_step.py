"""Prepared direct WRF-JAX time-step core.

This module is the first split between the strict WRF translation harness and
the production core. It preserves the configured ``solve_em`` numerical order,
but moves per-step invariant decisions out of the hot loop and calls prepared
stage functions directly instead of routing every phase through a callback
dispatcher.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
import os
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.boundary.physical_bc import set_physical_bc2d, set_physical_bc3d
from wrf_jax.boundary.spec_bdy_final import spec_bdy_final
from wrf_jax.boundary.spec_bdyupdate import spec_bdyupdate
from wrf_jax.boundary.zero_grad_bdy import zero_grad_bdy
from wrf_jax.compat.get_ijk_from_grid1 import get_ijk_from_grid1
from wrf_jax.compat.get_ijk_from_subgrid import get_ijk_from_subgrid
from wrf_jax.compat.module_tiles import set_tiles_once
from wrf_jax.dyn_em.module_bc_em import spec_bdyupdate_ph
from wrf_jax.dyn_em.module_small_step_em import advance_mu_t, advance_uv, advance_w, calc_p_rho, sumflux
from wrf_jax.dyn_em.solve_em import (
    MYNNPBLSCHEME,
    ORIGINAL,
    PARAM_FIRST_SCALAR,
    P_QV,
    WENO_SCALAR,
    ZONE_SOLVE_EM,
    SolveEmResult,
    _as_tuple,
    _compute_num_sound_steps,
    _default_current_time,
    _default_real_time,
    _default_sim_start_time,
    _default_start_time,
    _final_calc_p_rho_phi_bounds,
    _read,
    _read_bool,
    _read_float,
    _read_int,
    _rk_step_sizes,
    _tile_bounds,
    _tile_count,
    _write,
)

StageFn = Callable[[dict[str, Any]], Mapping[str, Any] | None]
_COMPILED_SMALL_LOOP = os.environ.get("WRF_PY_COMPILED_SMALL_LOOP", "0").strip().lower() in {"1", "true", "yes", "on"}
_DIRECT_SMALL_STEP_LOOP = os.environ.get("WRF_PY_DIRECT_SMALL_STEP_LOOP", "0").strip().lower() in {"1", "true", "yes", "on"}
_DIRECT_FIRST_RK_PART1 = os.environ.get("WRF_PY_DIRECT_FIRST_RK_PART1", "1").strip().lower() in {"1", "true", "yes", "on"}
_FUSED_SMALL_STEP_BOUNDARIES = os.environ.get("WRF_PY_FUSED_SMALL_STEP_BOUNDARIES", "1").strip().lower() in {"1", "true", "yes", "on"}
_UNROLL_DIRECT_SMALL_LOOP = os.environ.get("WRF_PY_UNROLL_DIRECT_SMALL_LOOP", "1").strip().lower() in {"1", "true", "yes", "on"}
_COMPILED_STAGE_SEMANTIC_BARRIERS = os.environ.get("WRF_PY_COMPILED_STAGE_SEMANTIC_BARRIERS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_COMPILED_SMALL_LOOP_CACHE: dict[tuple[int, int, float, tuple[int, ...], tuple[str, ...]], Callable[[dict[str, Any]], tuple[Any, ...]]] = {}
_DIRECT_SMALL_LOOP_CACHE: dict[tuple[int, int, float, tuple[int, ...], "_BoundaryConfig", int, bool, bool, bool], Callable[[dict[str, Any]], tuple[Any, ...]]] = {}
_SMALL_LOOP_OUTPUT_NAMES = (
    "u_2",
    "v_2",
    "ww",
    "ww_1",
    "mu_2",
    "muave",
    "muts",
    "mudf",
    "ru_m",
    "rv_m",
    "ww_m",
    "t_2",
    "t_2ave",
    "w_2",
    "ph_2",
    "al",
    "p",
    "pm1",
)
_DIRECT_STEP_PROFILE_PATH = os.environ.get("WRF_PY_PROFILE_DIRECT_STEP", "").strip()


@dataclass(frozen=True)
class _BoundaryConfig:
    periodic_x: bool = False
    periodic_y: bool = False
    specified: bool = False
    nested: bool = False
    polar: bool = False
    open_xs: bool = False
    open_xe: bool = False
    open_ys: bool = False
    open_ye: bool = False
    symmetric_xs: bool = False
    symmetric_xe: bool = False
    symmetric_ys: bool = False
    symmetric_ye: bool = False
    damp_opt: int = 0
    dampcoef: float = 0.0
    zdamp: float = 1.0
    phi_adv_z: int = 1


_BOUNDARY_STATIC_ARGNAMES = (
    "dts_rk",
    "config_flags",
    "spec_zone",
    "specified",
    "ids",
    "ide",
    "jds",
    "jde",
    "kds",
    "kde",
    "ims",
    "ime",
    "jms",
    "jme",
    "kms",
    "kme",
    "ips",
    "ipe",
    "jps",
    "jpe",
    "kps",
    "kpe",
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)
_PHYSICAL_BC_STATIC_ARGNAMES = tuple(name for name in _BOUNDARY_STATIC_ARGNAMES if name not in {"dts_rk", "spec_zone", "specified"})
_FINAL_SPEC_BDY_STATIC_ARGNAMES = (
    "config_flags",
    "spec_bdy_width",
    "spec_zone",
    "ids",
    "ide",
    "jds",
    "jde",
    "kds",
    "kde",
    "ims",
    "ime",
    "jms",
    "jme",
    "kms",
    "kme",
    "ips",
    "ipe",
    "jps",
    "jpe",
    "kps",
    "kpe",
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)


def _boundary_config(config_flags: Any) -> _BoundaryConfig:
    return _BoundaryConfig(
        periodic_x=bool(_read(config_flags, "periodic_x", False)),
        periodic_y=bool(_read(config_flags, "periodic_y", False)),
        specified=bool(_read(config_flags, "specified", False)),
        nested=bool(_read(config_flags, "nested", False)),
        polar=bool(_read(config_flags, "polar", False)),
        open_xs=bool(_read(config_flags, "open_xs", False)),
        open_xe=bool(_read(config_flags, "open_xe", False)),
        open_ys=bool(_read(config_flags, "open_ys", False)),
        open_ye=bool(_read(config_flags, "open_ye", False)),
        symmetric_xs=bool(_read(config_flags, "symmetric_xs", False)),
        symmetric_xe=bool(_read(config_flags, "symmetric_xe", False)),
        symmetric_ys=bool(_read(config_flags, "symmetric_ys", False)),
        symmetric_ye=bool(_read(config_flags, "symmetric_ye", False)),
        damp_opt=int(_read(config_flags, "damp_opt", 0)),
        dampcoef=float(_read(config_flags, "dampcoef", 0.0)),
        zdamp=float(_read(config_flags, "zdamp", 1.0)),
        phi_adv_z=int(_read(config_flags, "phi_adv_z", 1)),
    )


@partial(jax.jit, static_argnames=_FINAL_SPEC_BDY_STATIC_ARGNAMES)
def _fused_spec_bdy_final_prognostics(
    u_2: Any,
    v_2: Any,
    t_2: Any,
    ph_2: Any,
    mu_2: Any,
    qvapor: Any,
    muus: Any,
    muvs: Any,
    muts: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    msfuy: Any,
    msfvx: Any,
    msfty: Any,
    u_bxs: Any,
    u_bxe: Any,
    u_bys: Any,
    u_bye: Any,
    u_btxs: Any,
    u_btxe: Any,
    u_btys: Any,
    u_btye: Any,
    v_bxs: Any,
    v_bxe: Any,
    v_bys: Any,
    v_bye: Any,
    v_btxs: Any,
    v_btxe: Any,
    v_btys: Any,
    v_btye: Any,
    t_bxs: Any,
    t_bxe: Any,
    t_bys: Any,
    t_bye: Any,
    t_btxs: Any,
    t_btxe: Any,
    t_btys: Any,
    t_btye: Any,
    ph_bxs: Any,
    ph_bxe: Any,
    ph_bys: Any,
    ph_bye: Any,
    ph_btxs: Any,
    ph_btxe: Any,
    ph_btys: Any,
    ph_btye: Any,
    mu_bxs: Any,
    mu_bxe: Any,
    mu_bys: Any,
    mu_bye: Any,
    mu_btxs: Any,
    mu_btxe: Any,
    mu_btys: Any,
    mu_btye: Any,
    qvapor_bxs: Any,
    qvapor_bxe: Any,
    qvapor_bys: Any,
    qvapor_bye: Any,
    qvapor_btxs: Any,
    qvapor_btxe: Any,
    qvapor_btys: Any,
    qvapor_btye: Any,
    dtbc: Any,
    *,
    config_flags: _BoundaryConfig,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    u_out = spec_bdy_final(
        u_2,
        muus,
        c1h,
        c2h,
        msfuy,
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
        dtbc,
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
    v_out = spec_bdy_final(
        v_2,
        muvs,
        c1h,
        c2h,
        msfvx,
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
        dtbc,
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
    t_out = spec_bdy_final(
        t_2,
        muts,
        c1h,
        c2h,
        msfty,
        t_bxs,
        t_bxe,
        t_bys,
        t_bye,
        t_btxs,
        t_btxe,
        t_btys,
        t_btye,
        "t",
        config_flags,
        spec_bdy_width,
        spec_zone,
        dtbc,
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
    ph_out = spec_bdy_final(
        ph_2,
        muts,
        c1f,
        c2f,
        msfty,
        ph_bxs,
        ph_bxe,
        ph_bys,
        ph_bye,
        ph_btxs,
        ph_btxe,
        ph_btys,
        ph_btye,
        "h",
        config_flags,
        spec_bdy_width,
        spec_zone,
        dtbc,
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
    mu_out = spec_bdy_final(
        jnp.asarray(mu_2)[:, None, :],
        muts,
        c1h,
        c2h,
        msfty,
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
        dtbc,
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
    )[:, 0, :]
    qvapor_out = spec_bdy_final(
        qvapor,
        muts,
        c1h,
        c2h,
        msfty,
        qvapor_bxs,
        qvapor_bxe,
        qvapor_bys,
        qvapor_bye,
        qvapor_btxs,
        qvapor_btxe,
        qvapor_btys,
        qvapor_btye,
        "t",
        config_flags,
        spec_bdy_width,
        spec_zone,
        dtbc,
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
    return u_out, v_out, t_out, ph_out, mu_out, qvapor_out


@partial(jax.jit, static_argnames=_BOUNDARY_STATIC_ARGNAMES)
def _fused_spec_bdyupdate_uv(
    u_2: Any,
    ru_tend: Any,
    v_2: Any,
    rv_tend: Any,
    *,
    dts_rk: float,
    config_flags: _BoundaryConfig,
    spec_zone: int,
    specified: bool,
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
) -> tuple[Any, Any]:
    del specified
    u_out = spec_bdyupdate(
        u_2,
        ru_tend,
        dts_rk,
        "u",
        config_flags,
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
    v_out = spec_bdyupdate(
        v_2,
        rv_tend,
        dts_rk,
        "v",
        config_flags,
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
    return u_out, v_out


@partial(jax.jit, static_argnames=_BOUNDARY_STATIC_ARGNAMES)
def _fused_spec_bdyupdate_t_mu(
    t_2: Any,
    t_tend: Any,
    mu_2: Any,
    muts: Any,
    mu_tend: Any,
    *,
    dts_rk: float,
    config_flags: _BoundaryConfig,
    spec_zone: int,
    specified: bool,
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
) -> tuple[Any, Any, Any]:
    del specified
    t_out = spec_bdyupdate(
        t_2,
        t_tend,
        dts_rk,
        "t",
        config_flags,
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
    mu_3d = jnp.asarray(mu_2)[:, None, :]
    muts_3d = jnp.asarray(muts)[:, None, :]
    mu_tend_3d = jnp.asarray(mu_tend)[:, None, :]
    mu_out = spec_bdyupdate(
        mu_3d,
        mu_tend_3d,
        dts_rk,
        "m",
        config_flags,
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
    )[:, 0, :]
    muts_out = spec_bdyupdate(
        muts_3d,
        mu_tend_3d,
        dts_rk,
        "m",
        config_flags,
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
    )[:, 0, :]
    return t_out, mu_out, muts_out


@partial(jax.jit, static_argnames=_BOUNDARY_STATIC_ARGNAMES)
def _fused_spec_bdyupdate_ph_w(
    ph_save: Any,
    ph_2: Any,
    ph_tend: Any,
    mu_tend: Any,
    muts: Any,
    c1f: Any,
    c2f: Any,
    w_2: Any,
    rw_tend: Any,
    rounding_token: Any,
    *,
    dts_rk: float,
    config_flags: _BoundaryConfig,
    spec_zone: int,
    specified: bool,
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
) -> tuple[Any, Any]:
    ph_out = spec_bdyupdate_ph(
        ph_save,
        ph_2,
        ph_tend,
        mu_tend,
        muts,
        c1f,
        c2f,
        dts_rk,
        "h",
        config_flags,
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
        rounding_token,
    )
    if specified:
        w_out = zero_grad_bdy(
            w_2,
            "w",
            config_flags,
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
    else:
        w_out = spec_bdyupdate(
            w_2,
            rw_tend,
            dts_rk,
            "h",
            config_flags,
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
    return ph_out, w_out


@partial(jax.jit, static_argnames=_PHYSICAL_BC_STATIC_ARGNAMES)
def _fused_initial_dry_physical_bc(
    rho: Any,
    al: Any,
    ph_2: Any,
    *,
    config_flags: _BoundaryConfig,
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
) -> tuple[Any, Any, Any]:
    rho_out = set_physical_bc3d(rho, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    al_out = set_physical_bc3d(al, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    ph_out = set_physical_bc3d(ph_2, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    return rho_out, al_out, ph_out


@partial(jax.jit, static_argnames=_PHYSICAL_BC_STATIC_ARGNAMES)
def _fused_pre_small_physical_bc(
    ru_tend: Any,
    rv_tend: Any,
    ph_2: Any,
    al: Any,
    p: Any,
    t_1: Any,
    t_save: Any,
    mu_1: Any,
    mu_2: Any,
    mudf: Any,
    *,
    config_flags: _BoundaryConfig,
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
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    ru_tend_out = set_physical_bc3d(ru_tend, "u", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    rv_tend_out = set_physical_bc3d(rv_tend, "v", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    ph_out = set_physical_bc3d(ph_2, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    al_out = set_physical_bc3d(al, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    p_out = set_physical_bc3d(p, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    t_1_out = set_physical_bc3d(t_1, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    t_save_out = set_physical_bc3d(t_save, "t", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    mu_1_out = set_physical_bc2d(mu_1, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    mu_2_out = set_physical_bc2d(mu_2, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    mudf_out = set_physical_bc2d(mudf, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    return ru_tend_out, rv_tend_out, ph_out, al_out, p_out, t_1_out, t_save_out, mu_1_out, mu_2_out, mudf_out


@partial(jax.jit, static_argnames=_PHYSICAL_BC_STATIC_ARGNAMES)
def _fused_small_physical_bc(
    ph_2: Any,
    al: Any,
    p: Any,
    muts: Any,
    mu_2: Any,
    mudf: Any,
    *,
    config_flags: _BoundaryConfig,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    ph_out = set_physical_bc3d(ph_2, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    al_out = set_physical_bc3d(al, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    p_out = set_physical_bc3d(p, "p", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    muts_out = set_physical_bc2d(muts, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    mu_out = set_physical_bc2d(mu_2, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    mudf_out = set_physical_bc2d(mudf, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    return ph_out, al_out, p_out, muts_out, mu_out, mudf_out


@partial(jax.jit, static_argnames=_PHYSICAL_BC_STATIC_ARGNAMES)
def _fused_post_small_final_physical_bc(
    ru_m: Any,
    rv_m: Any,
    ww_m: Any,
    mut: Any,
    muts: Any,
    *,
    config_flags: _BoundaryConfig,
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
) -> tuple[Any, Any, Any, Any, Any]:
    ru_m_out = set_physical_bc3d(ru_m, "u", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    rv_m_out = set_physical_bc3d(rv_m, "v", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    ww_m_out = set_physical_bc3d(ww_m, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte)
    mut_out = set_physical_bc2d(mut, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    muts_out = set_physical_bc2d(muts, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte)
    return ru_m_out, rv_m_out, ww_m_out, mut_out, muts_out


_SMALL_LOOP_DYNAMIC_ARRAY_NAMES = frozenset(
    (
        "a",
        "al",
        "alpha",
        "alt",
        "c2a",
        "cqu",
        "cqv",
        "cqw",
        "mudf",
        "mu_2",
        "mu_tend",
        "muave",
        "muts",
        "muu",
        "muv",
        "mut",
        "p",
        "ph_2",
        "ph_save",
        "ph_tend",
        "php",
        "pm1",
        "ru_m",
        "ru_tend",
        "rv_m",
        "rv_tend",
        "rw_tend",
        "t_2",
        "t_2ave",
        "t_save",
        "t_tend",
        "u_2",
        "u_save",
        "v_2",
        "v_save",
        "w_2",
        "w_save",
        "ww",
        "ww_1",
        "ww_m",
    )
)


@dataclass(frozen=True)
class DirectStepPlan:
    """Prepared static metadata and stage callables for one WRF step."""

    ijk: tuple[int, ...]
    subgrid_ijk: tuple[int, ...]
    tile_bounds: tuple[tuple[int, int, int, int], ...]
    rk_order: int
    num_sound_steps: int
    dts: float
    num_moist: int
    num_chem: int
    num_tracer: int
    num_scalar: int
    callbacks: Mapping[str, StageFn]

    @property
    def bounds(self) -> tuple[int, ...]:
        return self.ijk[:42]


@dataclass
class DirectStepStageContext:
    """Per-step static/control context shared by direct stage families."""

    bounds: tuple[int, ...]
    num_3d_m: int
    num_3d_c: int
    num_3d_s: int
    f_flux: bool
    curr_secs: float
    curr_secs2: float
    adapt_step_flag: bool
    restart_flag: bool
    diag_flag: bool
    specified: bool
    constant_bc: bool
    specified_or_nested: bool
    non_hydrostatic: bool
    h_sca_adv_order: int
    h_mom_adv_order: int
    fused_boundary_groups: bool
    fused_small_boundaries: bool
    boundary_config: _BoundaryConfig
    spec_zone: int
    spec_bdy_width: int
    max_vert_cfl_tmp: list[float]
    max_horiz_cfl_tmp: list[float]

    @property
    def ids(self) -> int:
        return self.bounds[0]

    @property
    def ide(self) -> int:
        return self.bounds[1]

    @property
    def jds(self) -> int:
        return self.bounds[2]

    @property
    def jde(self) -> int:
        return self.bounds[3]

    def boundary_kwargs(self, dts_rk_value: float) -> dict[str, Any]:
        (
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
            *_,
        ) = self.bounds
        return {
            "dts_rk": float(dts_rk_value),
            "config_flags": self.boundary_config,
            "spec_zone": int(self.spec_zone),
            "specified": bool(self.specified),
            "ids": ids,
            "ide": ide,
            "jds": jds,
            "jde": jde,
            "kds": kds,
            "kde": kde,
            "ims": ims,
            "ime": ime,
            "jms": jms,
            "jme": jme,
            "kms": kms,
            "kme": kme,
            "ips": ips,
            "ipe": ipe,
            "jps": jps,
            "jpe": jpe,
            "kps": kps,
            "kpe": kpe,
            "its": ips,
            "ite": ipe,
            "jts": jps,
            "jte": jpe,
            "kts": kps,
            "kte": kpe,
        }

    def physical_bc_kwargs(self) -> dict[str, Any]:
        kwargs = self.boundary_kwargs(0.0)
        kwargs.pop("dts_rk")
        kwargs.pop("spec_zone")
        kwargs.pop("specified")
        return kwargs


POST_RK_PHYSICS_STAGE_FAMILY = (
    "advance_ppt",
    "phy_prep_part2",
    "moist_physics_prep_em",
    "microphysics_driver",
    "moist_physics_finish_em",
    "set_physical_bc3d:h_diabatic",
    "calc_p_rho_phi:final",
    "set_phys_bc_dry_2",
    "set_physical_bc3d:tke_1",
    "set_physical_bc3d:tke_2",
    "set_physical_bc3d:moist",
    "set_physical_bc3d:scalar",
)


POST_RK_PHYSICS_STAGE_GROUPS = (
    "post_advance_ppt",
    "post_phy_prep_part2",
    "post_moist_prep",
    "post_microphysics",
    "post_moist_finish",
    "post_calc_p_rho_phi_final",
    "post_physical_bc",
)


FINAL_BOUNDARY_STAGE_FAMILY = (
    "spec_bdy_final:u",
    "spec_bdy_final:v",
    "spec_bdy_final:t",
    "spec_bdy_final:ph",
    "spec_bdy_final:mu",
    "spec_bdy_final:moist",
    "set_w_surface",
    "after_all_rk_steps",
)


RK_STEP_STAGE_FAMILY = (
    "initialize_moist_old",
    "rk_step_prep",
    "rk_phys_bc_dry_1",
    "first_rk_step_part1",
    "first_rk_step_part2",
    "rk_tendency",
    "dry_boundary_tendency",
    "small_step_prep",
    "small_step_loop",
    "small_step_finish",
    "moist_scalar_advance",
    "scalar_advance",
    "calc_p_rho_phi",
    "rk_phys_bc_dry_2",
)


RK_COMPILED_SUBSTAGES = (
    "rk_step_prep",
    "rk_phys_bc_dry_1",
    "rk_tendency",
    "dry_boundary_tendency",
    "small_step_prep",
    "small_step_loop",
    "small_step_finish",
    "moist_scalar_advance",
    "scalar_advance",
    "calc_p_rho_phi",
    "rk_phys_bc_dry_2",
)


RK1_PRE_FIRST_PART1_STAGE_FAMILY = (
    "initialize_moist_old",
    "set_physical_bc3d:moist_old",
    "rk_step_prep",
    "rk_phys_bc_dry_1",
    "set_physical_bc3d:rho",
    "set_physical_bc3d:al",
    "set_physical_bc3d:ph_2",
)


FIRST_RK_PART1_DIRECT_STAGE_FAMILY = (
    "init_zero_tendency",
    "phy_prep",
    "radiation_driver",
    "surface_pbl_driver",
    "cumulus_driver",
    "shallowcu_driver",
    "fddagd_driver",
)


def _callback(callbacks: Mapping[str, StageFn], name: str) -> StageFn | None:
    return callbacks.get(name)


def _profile_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _stage_profile_name(fn: StageFn, context: Mapping[str, Any]) -> str:
    name = getattr(fn, "__name__", type(fn).__name__)
    parts = [name]
    for key in ("rk_step", "iteration", "ij"):
        if key in context:
            parts.append(f"{key}={context[key]}")
    return " ".join(parts)


def _replace_last_axis_component(array: Any, index: int, value: Any) -> Any:
    arr = jnp.asarray(array)
    component = jnp.expand_dims(jnp.asarray(value, dtype=arr.dtype), axis=-1)
    return jnp.concatenate([arr[..., : int(index)], component, arr[..., int(index) + 1 :]], axis=-1)


def _write_stage_profile(name: str, seconds: float) -> None:
    if not _DIRECT_STEP_PROFILE_PATH:
        return
    with open(_DIRECT_STEP_PROFILE_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{seconds:.6f} {name}\n")


def _apply(fn: StageFn | None, state: dict[str, Any], **context: Any) -> None:
    if context:
        state.update(context)
    if fn is None:
        return
    profile_name = _stage_profile_name(fn, context) if _DIRECT_STEP_PROFILE_PATH else ""
    start = perf_counter() if _DIRECT_STEP_PROFILE_PATH else 0.0
    updates = fn(state)
    if updates:
        if _COMPILED_STAGE_SEMANTIC_BARRIERS:
            updates = jax.tree_util.tree_map(
                lambda value: jax.lax.optimization_barrier(value) if isinstance(value, jax.core.Tracer) else value,
                updates,
            )
        state.update(updates)
    if _DIRECT_STEP_PROFILE_PATH:
        _profile_ready(updates)
        _write_stage_profile(profile_name, perf_counter() - start)


def _is_numeric_array(value: Any) -> bool:
    if not hasattr(value, "shape"):
        return False
    try:
        dtype = np.dtype(getattr(value, "dtype"))
    except TypeError:
        return False
    return dtype.kind in "biufc" and len(tuple(getattr(value, "shape"))) > 0


def _dynamic_array_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in state.items()
        if name in _SMALL_LOOP_DYNAMIC_ARRAY_NAMES and _is_numeric_array(value)
    }


def _compiled_small_loop_fn(
    *,
    state: Mapping[str, Any],
    callbacks: Mapping[str, StageFn],
    rk_step: int,
    dts_rk: float,
    number_of_small_timesteps: int,
    bounds: tuple[int, ...],
    dynamic_names: tuple[str, ...],
) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    key = (int(rk_step), int(number_of_small_timesteps), float(dts_rk), tuple(bounds), dynamic_names)
    static_state = {name: value for name, value in state.items() if name not in dynamic_names}
    cacheable = not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(static_state))
    if cacheable:
        cached = _COMPILED_SMALL_LOOP_CACHE.get(key)
        if cached is not None:
            return cached

    specified = _read_bool(static_state["config_flags"], "specified")
    nested_cfg = _read_bool(static_state["config_flags"], "nested")
    specified_or_nested = specified or nested_cfg
    non_hydrostatic = _read_bool(static_state["config_flags"], "non_hydrostatic", True)

    advance_uv = callbacks.get("advance_uv")
    advance_mu_t = callbacks.get("advance_mu_t")
    advance_w = callbacks.get("advance_w")
    sumflux = callbacks.get("sumflux")
    calc_p_rho = callbacks.get("calc_p_rho")
    spec_bdyupdate_u = callbacks.get("spec_bdyupdate:u")
    spec_bdyupdate_v = callbacks.get("spec_bdyupdate:v")
    spec_bdyupdate_t = callbacks.get("spec_bdyupdate:t")
    spec_bdyupdate_mu_2 = callbacks.get("spec_bdyupdate:mu_2")
    spec_bdyupdate_muts = callbacks.get("spec_bdyupdate:muts")
    spec_bdyupdate_w = callbacks.get("spec_bdyupdate:w")
    spec_bdyupdate_ph = callbacks.get("spec_bdyupdate_ph")
    zero_grad_bdy = callbacks.get("zero_grad_bdy")
    small_ph_2 = callbacks.get("set_physical_bc3d:ph_2:small")
    small_al = callbacks.get("set_physical_bc3d:al:small")
    small_p = callbacks.get("set_physical_bc3d:p:small")
    small_muts = callbacks.get("set_physical_bc2d:muts:small")
    small_mu_2 = callbacks.get("set_physical_bc2d:mu_2:small")
    small_mudf = callbacks.get("set_physical_bc2d:mudf:small")

    def _loop(dynamic_state: dict[str, Any]) -> tuple[Any, ...]:
        loop_state = dict(static_state)
        loop_state.update(dynamic_state)
        loop_state.update({"rk_step": int(rk_step), "dts_rk": float(dts_rk), "number_of_small_timesteps": int(number_of_small_timesteps)})
        for iteration in range(1, int(number_of_small_timesteps) + 1):
            loop_state["iteration"] = iteration
            _apply(advance_uv, loop_state, ij=1)
            if specified_or_nested:
                _apply(spec_bdyupdate_u, loop_state, ij=1)
                _apply(spec_bdyupdate_v, loop_state, ij=1)
            _apply(advance_mu_t, loop_state, ij=1)
            if specified_or_nested:
                _apply(spec_bdyupdate_t, loop_state, ij=1)
                _apply(spec_bdyupdate_mu_2, loop_state, ij=1)
                _apply(spec_bdyupdate_muts, loop_state, ij=1)
            if non_hydrostatic:
                _apply(advance_w, loop_state, ij=1)
            _apply(sumflux, loop_state, ij=1)
            if specified_or_nested and non_hydrostatic:
                _apply(spec_bdyupdate_ph, loop_state, ij=1)
                if specified:
                    _apply(zero_grad_bdy, loop_state, ij=1)
                else:
                    _apply(spec_bdyupdate_w, loop_state, ij=1)
            _apply(calc_p_rho, loop_state, ij=1, iteration=iteration)
            _apply(small_ph_2, loop_state, ij=1)
            _apply(small_al, loop_state, ij=1)
            _apply(small_p, loop_state, ij=1)
            _apply(small_muts, loop_state, ij=1)
            _apply(small_mu_2, loop_state, ij=1)
            _apply(small_mudf, loop_state, ij=1)
        return tuple(loop_state[name] for name in dynamic_names)

    compiled = jax.jit(_loop)
    if cacheable:
        _COMPILED_SMALL_LOOP_CACHE[key] = compiled
    return compiled


def _run_small_step_loop(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    rk_step: int,
    dts_rk: float,
    number_of_small_timesteps: int,
    bounds: tuple[int, ...],
) -> bool:
    if not _COMPILED_SMALL_LOOP:
        return False
    dynamic_state = _dynamic_array_fields(state)
    dynamic_names = tuple(sorted(dynamic_state))
    compiled = _compiled_small_loop_fn(
        state=state,
        callbacks=callbacks,
        rk_step=rk_step,
        dts_rk=dts_rk,
        number_of_small_timesteps=number_of_small_timesteps,
        bounds=bounds,
        dynamic_names=dynamic_names,
    )
    outputs = compiled(dynamic_state)
    state.update(dict(zip(dynamic_names, outputs, strict=True)))
    state["iteration"] = int(number_of_small_timesteps)
    return True


_DIRECT_SMALL_LOOP_INPUT_NAMES = frozenset(
    (
        "a",
        "al",
        "alt",
        "alpha",
        "c1f",
        "c1h",
        "c2a",
        "c2f",
        "c2h",
        "c3h",
        "cf1",
        "cf2",
        "cf3",
        "cqu",
        "cqv",
        "cqw",
        "dnw",
        "fnm",
        "fnp",
        "gamma",
        "ht",
        "mudf",
        "mu_2",
        "mu_tend",
        "muave",
        "muts",
        "muu",
        "muv",
        "mut",
        "msftx",
        "msfty",
        "msfux",
        "msfuy",
        "msfvx",
        "msfvx_inv",
        "msfvy",
        "p",
        "pb",
        "ph_2",
        "ph_save",
        "ph_tend",
        "phb",
        "php",
        "pm1",
        "rdn",
        "rdnw",
        "ru_m",
        "ru_tend",
        "rv_m",
        "rv_tend",
        "rw_tend",
        "t_2",
        "t_2ave",
        "t_save",
        "t_tend",
        "u_2",
        "u_save",
        "v_2",
        "v_save",
        "w_2",
        "w_save",
        "ww",
        "ww_1",
        "ww_m",
    )
)

_DIRECT_SMALL_LOOP_OUTPUT_NAMES = (
    "u_2",
    "v_2",
    "ww",
    "ww_1",
    "mu_2",
    "muave",
    "muts",
    "mudf",
    "ru_m",
    "rv_m",
    "ww_m",
    "t_2",
    "t_2ave",
    "w_2",
    "ph_2",
    "al",
    "p",
    "pm1",
)


def _direct_small_loop_fn(
    *,
    rk_step: int,
    dts_rk: float,
    number_of_small_timesteps: int,
    bounds: tuple[int, ...],
    config_flags: _BoundaryConfig,
    spec_zone: int,
    specified: bool,
    non_hydrostatic: bool,
    top_lid: bool,
) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    key = (
        int(rk_step),
        int(number_of_small_timesteps),
        float(dts_rk),
        tuple(bounds),
        config_flags,
        int(spec_zone),
        bool(specified),
        bool(non_hydrostatic),
        bool(top_lid),
    )
    cached = _DIRECT_SMALL_LOOP_CACHE.get(key)
    if cached is not None:
        return cached

    (
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
        *_unused,
    ) = bounds
    its, ite, jts, jte, kts, kte = ips, ipe, jps, jpe, kps, kpe
    specified_or_nested = bool(config_flags.specified or config_flags.nested)

    def _loop(dynamic_state: dict[str, Any]) -> tuple[Any, ...]:
        def _body(iteration: Any, carry: dict[str, Any]) -> dict[str, Any]:
            s = carry
            uv = advance_uv(
                s["u_2"],
                s["ru_tend"],
                s["v_2"],
                s["rv_tend"],
                s["p"],
                s["pb"],
                s["ph_2"],
                s["php"],
                s["alt"],
                s["al"],
                s["mu_2"],
                s["muu"],
                s["cqu"],
                s["muv"],
                s["cqv"],
                s["mudf"],
                s["c1h"],
                s["c2h"],
                s["c1f"],
                s["c2f"],
                s["c3h"],
                None,
                None,
                None,
                s["msfux"],
                s["msfuy"],
                s["msfvx"],
                s["msfvx_inv"],
                s["msfvy"],
                s["_rdx"],
                s["_rdy"],
                dts_rk,
                s["cf1"],
                s["cf2"],
                s["cf3"],
                s["fnm"],
                s["fnp"],
                s["_emdiv"],
                s["rdnw"],
                config_flags,
                spec_zone,
                non_hydrostatic,
                top_lid,
                ids=ids,
                ide=ide,
                jds=jds,
                jde=jde,
                kds=kds,
                kde=kde,
                ims=ims,
                ime=ime,
                jms=jms,
                jme=jme,
                kms=kms,
                kme=kme,
                its=its,
                ite=ite,
                jts=jts,
                jte=jte,
                kts=kts,
                kte=kte,
            )
            s["u_2"], s["v_2"] = jax.lax.optimization_barrier((uv.u, uv.v))
            if specified_or_nested:
                s["u_2"], s["v_2"] = jax.lax.optimization_barrier(_fused_spec_bdyupdate_uv(
                    s["u_2"],
                    s["ru_tend"],
                    s["v_2"],
                    s["rv_tend"],
                    dts_rk=dts_rk,
                    config_flags=config_flags,
                    spec_zone=spec_zone,
                    specified=specified,
                    ids=ids,
                    ide=ide,
                    jds=jds,
                    jde=jde,
                    kds=kds,
                    kde=kde,
                    ims=ims,
                    ime=ime,
                    jms=jms,
                    jme=jme,
                    kms=kms,
                    kme=kme,
                    ips=ips,
                    ipe=ipe,
                    jps=jps,
                    jpe=jpe,
                    kps=kps,
                    kpe=kpe,
                    its=its,
                    ite=ite,
                    jts=jts,
                    jte=jte,
                    kts=kts,
                    kte=kte,
                ))

            mut = advance_mu_t(
                s["ww"],
                s["ww_1"],
                s["u_2"],
                s["u_save"],
                s["v_2"],
                s["v_save"],
                s["mu_2"],
                s["mut"],
                s["muave"],
                s["muts"],
                s["muu"],
                s["muv"],
                s["mudf"],
                s["c1h"],
                s["c2h"],
                s["c1f"],
                s["c2f"],
                s["c3h"],
                None,
                None,
                None,
                s["ru_m"],
                s["rv_m"],
                s["ww_m"],
                s["t_2"],
                s["t_save"],
                s["t_2ave"],
                s["t_tend"],
                s["mu_tend"],
                s["_rdx"],
                s["_rdy"],
                dts_rk,
                s["_epssm"],
                s["dnw"],
                s["fnm"],
                s["fnp"],
                s["rdnw"],
                s["msfux"],
                s["msfuy"],
                s["msfvx"],
                s["msfvx_inv"],
                s["msfvy"],
                s["msftx"],
                s["msfty"],
                iteration,
                config_flags,
                s["_wrf_real_rounding_token"],
                ids=ids,
                ide=ide,
                jds=jds,
                jde=jde,
                kds=kds,
                kde=kde,
                ims=ims,
                ime=ime,
                jms=jms,
                jme=jme,
                kms=kms,
                kme=kme,
                its=its,
                ite=ite,
                jts=jts,
                jte=jte,
                kts=kts,
                kte=kte,
            )
            (
                s["ww"],
                s["ww_1"],
                s["mu_2"],
                s["muave"],
                s["muts"],
                s["mudf"],
                s["ru_m"],
                s["rv_m"],
                s["ww_m"],
                s["t_2"],
                s["t_2ave"],
            ) = jax.lax.optimization_barrier(
                (mut.ww, mut.ww_1, mut.mu, mut.muave, mut.muts, mut.mudf, mut.uam, mut.vam, mut.wwam, mut.t, mut.t_ave)
            )
            if specified_or_nested:
                s["t_2"], s["mu_2"], s["muts"] = jax.lax.optimization_barrier(_fused_spec_bdyupdate_t_mu(
                    s["t_2"],
                    s["t_tend"],
                    s["mu_2"],
                    s["muts"],
                    s["mu_tend"],
                    dts_rk=dts_rk,
                    config_flags=config_flags,
                    spec_zone=spec_zone,
                    specified=specified,
                    ids=ids,
                    ide=ide,
                    jds=jds,
                    jde=jde,
                    kds=kds,
                    kde=kde,
                    ims=ims,
                    ime=ime,
                    jms=jms,
                    jme=jme,
                    kms=kms,
                    kme=kme,
                    ips=ips,
                    ipe=ipe,
                    jps=jps,
                    jpe=jpe,
                    kps=kps,
                    kpe=kpe,
                    its=its,
                    ite=ite,
                    jts=jts,
                    jte=jte,
                    kts=kts,
                    kte=kte,
                ))
            if non_hydrostatic:
                w_result = advance_w(
                    s["w_2"],
                    s["rw_tend"],
                    s["ww"],
                    s["w_save"],
                    s["u_2"],
                    s["v_2"],
                    s["mu_2"],
                    s["mut"],
                    s["muave"],
                    s["muts"],
                    s["c1h"],
                    s["c2h"],
                    s["c1f"],
                    s["c2f"],
                    s["c3h"],
                    None,
                    None,
                    None,
                    s["t_2ave"],
                    s["t_2"],
                    s["t_save"],
                    s["ph_2"],
                    s["ph_save"],
                    s["phb"],
                    s["ph_tend"],
                    s["ht"],
                    s["c2a"],
                    s["cqw"],
                    s["alt"],
                    None,
                    s["a"],
                    s["alpha"],
                    s["gamma"],
                    s["_rdx"],
                    s["_rdy"],
                    dts_rk,
                    s["_t0"],
                    s["_epssm"],
                    s["dnw"],
                    s["fnm"],
                    s["fnp"],
                    s["rdnw"],
                    s["rdn"],
                    s["cf1"],
                    s["cf2"],
                    s["cf3"],
                    s["msftx"],
                    s["msfty"],
                    config_flags,
                    top_lid,
                    s["_wrf_real_rounding_token"],
                    ids=ids,
                    ide=ide,
                    jds=jds,
                    jde=jde,
                    kds=kds,
                    kde=kde,
                    ims=ims,
                    ime=ime,
                    jms=jms,
                    jme=jme,
                    kms=kms,
                    kme=kme,
                    its=its,
                    ite=ite,
                    jts=jts,
                    jte=jte,
                    kts=kts,
                    kte=kte,
                )
                s["t_2ave"], s["w_2"], s["ph_2"] = jax.lax.optimization_barrier(
                    (w_result.t_2ave, w_result.w, w_result.ph)
                )

            flux = sumflux(
                s["u_2"],
                s["v_2"],
                s["ww"],
                s["u_save"],
                s["v_save"],
                s["ww_1"],
                s["muu"],
                s["muv"],
                s["c1h"],
                s["c2h"],
                s["c1f"],
                s["c2f"],
                s["c3h"],
                None,
                None,
                None,
                s["ru_m"],
                s["rv_m"],
                s["ww_m"],
                s["_epssm"],
                s["msfux"],
                s["msfuy"],
                s["msfvx"],
                s["msfvx_inv"],
                s["msfvy"],
                iteration,
                int(number_of_small_timesteps),
                ids=ids,
                ide=ide,
                jds=jds,
                jde=jde,
                kds=kds,
                kde=kde,
                ims=ims,
                ime=ime,
                jms=jms,
                jme=jme,
                kms=kms,
                kme=kme,
                its=its,
                ite=ite,
                jts=jts,
                jte=jte,
                kts=kts,
                kte=kte,
            )
            s["ru_m"], s["rv_m"], s["ww_m"] = jax.lax.optimization_barrier((flux.ru_m, flux.rv_m, flux.ww_m))
            if specified_or_nested and non_hydrostatic:
                s["ph_2"], s["w_2"] = jax.lax.optimization_barrier(_fused_spec_bdyupdate_ph_w(
                    s["ph_save"],
                    s["ph_2"],
                    s["ph_tend"],
                    s["mu_tend"],
                    s["muts"],
                    s["c1f"],
                    s["c2f"],
                    s["w_2"],
                    s["rw_tend"],
                    s["_wrf_real_rounding_token"],
                    dts_rk=dts_rk,
                    config_flags=config_flags,
                    spec_zone=spec_zone,
                    specified=specified,
                    ids=ids,
                    ide=ide,
                    jds=jds,
                    jde=jde,
                    kds=kds,
                    kde=kde,
                    ims=ims,
                    ime=ime,
                    jms=jms,
                    jme=jme,
                    kms=kms,
                    kme=kme,
                    ips=ips,
                    ipe=ipe,
                    jps=jps,
                    jpe=jpe,
                    kps=kps,
                    kpe=kpe,
                    its=its,
                    ite=ite,
                    jts=jts,
                    jte=jte,
                    kts=kts,
                    kte=kte,
                ))

            prho = calc_p_rho(
                s["al"],
                s["p"],
                s["ph_2"],
                s["alt"],
                s["t_2"],
                s["t_save"],
                s["c2a"],
                s["pm1"],
                s["mu_2"],
                s["muts"],
                s["c1h"],
                s["c2h"],
                s["c1f"],
                s["c2f"],
                s["c3h"],
                None,
                None,
                None,
                None,
                s["_t0"],
                s["rdnw"],
                s["dnw"],
                s["_smdiv"],
                non_hydrostatic,
                iteration,
                ids=ids,
                ide=ide,
                jds=jds,
                jde=jde,
                kds=kds,
                kde=kde,
                ims=ims,
                ime=ime,
                jms=jms,
                jme=jme,
                kms=kms,
                kme=kme,
                its=its,
                ite=ite,
                jts=jts,
                jte=jte,
                kts=kts,
                kte=kte,
            )
            s["al"], s["p"], s["ph_2"], s["pm1"] = jax.lax.optimization_barrier(
                (prho.al, prho.p, prho.ph, prho.pm1)
            )
            s["ph_2"], s["al"], s["p"], s["muts"], s["mu_2"], s["mudf"] = jax.lax.optimization_barrier(_fused_small_physical_bc(
                s["ph_2"],
                s["al"],
                s["p"],
                s["muts"],
                s["mu_2"],
                s["mudf"],
                config_flags=config_flags,
                ids=ids,
                ide=ide,
                jds=jds,
                jde=jde,
                kds=kds,
                kde=kde,
                ims=ims,
                ime=ime,
                jms=jms,
                jme=jme,
                kms=kms,
                kme=kme,
                ips=ips,
                ipe=ipe,
                jps=jps,
                jpe=jpe,
                kps=kps,
                kpe=kpe,
                its=its,
                ite=ite,
                jts=jts,
                jte=jte,
                kts=kts,
                kte=kte,
            ))
            return s

        if _UNROLL_DIRECT_SMALL_LOOP:
            s = dict(dynamic_state)
            for iteration in range(1, int(number_of_small_timesteps) + 1):
                s = _body(iteration, s)
        else:
            s = jax.lax.fori_loop(1, int(number_of_small_timesteps) + 1, _body, dict(dynamic_state))
        return tuple(s[name] for name in _DIRECT_SMALL_LOOP_OUTPUT_NAMES)

    compiled = jax.jit(
        _loop,
        inline=False,
        compiler_options={"xla_disable_hlo_passes": "call-inliner"},
    )
    _DIRECT_SMALL_LOOP_CACHE[key] = compiled
    return compiled


def _run_direct_small_step_loop(
    *,
    state: dict[str, Any],
    config_flags: _BoundaryConfig,
    rk_step: int,
    dts_rk: float,
    number_of_small_timesteps: int,
    bounds: tuple[int, ...],
    spec_zone: int,
    specified: bool,
    non_hydrostatic: bool,
) -> bool:
    if not _DIRECT_SMALL_STEP_LOOP:
        return False
    missing = [name for name in _DIRECT_SMALL_LOOP_INPUT_NAMES if name not in state]
    if missing:
        return False
    dynamic_state = {name: state[name] for name in _DIRECT_SMALL_LOOP_INPUT_NAMES}
    dynamic_state["_rdx"] = 1.0 / float(state["grid"].dx)
    dynamic_state["_rdy"] = 1.0 / float(state["grid"].dy)
    dynamic_state["_emdiv"] = _field_or_direct(state, "emdiv", _config_namelist_value(state["config_flags"], "emdiv", 1, 0.01))
    dynamic_state["_epssm"] = _field_or_direct(state, "epssm", _config_namelist_value(state["config_flags"], "epssm", 1, 0.1))
    dynamic_state["_smdiv"] = _field_or_direct(state, "smdiv", _config_namelist_value(state["config_flags"], "smdiv", 1, 0.1))
    dynamic_state["_t0"] = _field_or_direct(state, "t0", 300.0)
    compiled = _direct_small_loop_fn(
        rk_step=rk_step,
        dts_rk=dts_rk,
        number_of_small_timesteps=number_of_small_timesteps,
        bounds=bounds,
        config_flags=config_flags,
        spec_zone=spec_zone,
        specified=specified,
        non_hydrostatic=non_hydrostatic,
        top_lid=bool(_read_bool(state["config_flags"], "top_lid", False)),
    )
    outputs = compiled(dynamic_state)
    state.update(dict(zip(_DIRECT_SMALL_LOOP_OUTPUT_NAMES, outputs, strict=True)))
    state["iteration"] = int(number_of_small_timesteps)
    return True


def _field_or_direct(state: Mapping[str, Any], name: str, default: Any) -> Any:
    return state[name] if name in state else default


def _is_traced_value(value: Any) -> bool:
    return isinstance(value, jax.core.Tracer)


def _write_if_concrete(target: Any, name: str, value: Any) -> None:
    if _is_traced_value(value):
        return
    try:
        # A concrete device scalar can remain on the grid as a scalar. Reading
        # it back here serializes every host-submitted stage with the GPU.
        if not (isinstance(value, jax.Array) and value.ndim == 0) and hasattr(value, "item"):
            value = value.item()
        _write(target, name, value)
    except (TypeError, ValueError):
        return


def _seed_rk_step_size_fields(
    *,
    state: dict[str, Any],
    grid: Any,
    plan: DirectStepPlan,
    rk_step: int,
) -> None:
    dt_rk, dts_rk, number_of_small_timesteps = _rk_step_sizes(
        plan.rk_order,
        int(rk_step),
        _read_float(grid, "dt", 0.0),
        plan.dts,
        plan.num_sound_steps,
    )
    state.update(
        {
            "rk_step": int(rk_step),
            "dt_rk": dt_rk,
            "dts_rk": dts_rk,
            "number_of_small_timesteps": number_of_small_timesteps,
        }
    )


def _set_current_dry_aliases(state: dict[str, Any]) -> None:
    for alias, current in (
        ("u", "u_2"),
        ("v", "v_2"),
        ("w", "w_2"),
        ("t", "t_2"),
        ("ph", "ph_2"),
        ("mu", "mu_2"),
    ):
        if current in state:
            state[alias] = state[current]


def _config_namelist_value(config_flags: Any, name: str, grid_id: int, default: Any = 0) -> Any:
    value = _read(config_flags, name, default)
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, list | tuple):
        index = max(0, int(grid_id) - 1)
        if index < len(value):
            return value[index]
        return default
    return value


def _run_first_rk_step_part1_direct(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    f_flux: bool,
) -> None:
    """Run the configured first RK part1 sequence without the generic driver."""

    _run_first_rk_step_part1_stage_family(
        state=state,
        callbacks=callbacks,
        plan=plan,
        f_flux=f_flux,
        stage_names=FIRST_RK_PART1_DIRECT_STAGE_FAMILY,
    )


def _seed_first_rk_physics_surface_context(state: dict[str, Any]) -> None:
    grid_id = _read_int(state["grid"], "id", 1)
    state["iswater"] = int(_config_namelist_value(state["config_flags"], "iswater", grid_id, 0))
    state["isurban"] = int(_config_namelist_value(state["config_flags"], "isurban", grid_id, 0))
    state["mminlu"] = str(_config_namelist_value(state["config_flags"], "mminlu", grid_id, ""))
    state["hydro_dt"] = 0.0 if _read_int(state["grid"], "num_nests", 0) < 1 else -1.0
    state["do_bioe"] = False
    state["do_meganfile"] = False


def _run_first_rk_step_part1_stage_family(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    f_flux: bool,
    stage_names: tuple[str, ...],
) -> None:
    """Run selected first-RK physics substages.

    This is the production-path split point for the former monolithic
    ``first_rk_step_part1`` block.  Direct mode still calls the full sequence;
    compiled trajectory code can now lift the expensive column-physics stages
    independently instead of forcing one giant trace.
    """

    _set_current_dry_aliases(state)
    selected = set(stage_names)
    init_zero_tendency = callbacks.get("init_zero_tendency")
    phy_prep = callbacks.get("phy_prep")
    radiation_driver = callbacks.get("radiation_driver")
    surface_driver = callbacks.get("surface_driver")
    surface_pbl_driver = callbacks.get("surface_pbl_driver")
    pbl_driver = callbacks.get("pbl_driver")
    cumulus_driver = callbacks.get("cumulus_driver")
    shallowcu_driver = callbacks.get("shallowcu_driver")
    fddagd_driver = callbacks.get("fddagd_driver")

    if "init_zero_tendency" in selected:
        for ij, _bounds in enumerate(plan.tile_bounds, start=1):
            _apply(init_zero_tendency, state, ij=ij)

    if "phy_prep" in selected:
        for ij, _bounds in enumerate(plan.tile_bounds, start=1):
            _apply(phy_prep, state, ij=ij)

    if "radiation_driver" in selected:
        _apply(radiation_driver, state)

    if "surface_pbl_driver" in selected:
        _seed_first_rk_physics_surface_context(state)
        if surface_pbl_driver is not None:
            _apply(surface_pbl_driver, state)
        else:
            _apply(surface_driver, state)
            _apply(pbl_driver, state)

    if "cumulus_driver" in selected:
        state["l_flux"] = bool(f_flux)
        _apply(cumulus_driver, state)
    if "shallowcu_driver" in selected:
        _apply(shallowcu_driver, state)
    if "fddagd_driver" in selected:
        _apply(fddagd_driver, state)


def _run_first_rk_step_part2_stage(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    adapt_step_flag: bool,
    curr_secs: float,
) -> None:
    """Run the first-RK part2 update as an independently liftable stage."""

    _apply(callbacks.get("first_rk_step_part2"), state, adapt_step_flag=adapt_step_flag, curr_secs=curr_secs)


def build_direct_step_plan(
    *,
    grid: Any,
    config_flags: Any,
    callbacks: Mapping[str, StageFn],
    num_moist: int = 2,
    num_chem: int = 0,
    num_tracer: int = 0,
    num_scalar: int = 0,
) -> DirectStepPlan:
    """Prepare per-trajectory static metadata for the direct core."""

    ijk = _as_tuple(get_ijk_from_grid1(grid), 42, "get_ijk_from_grid")
    (
        ids,
        ide,
        jds,
        jde,
        _kds,
        _kde,
        _ims,
        _ime,
        _jms,
        _jme,
        _kms,
        _kme,
        ips,
        ipe,
        jps,
        jpe,
        *_rest,
    ) = ijk[:42]
    subgrid_ijk = _as_tuple(get_ijk_from_subgrid(grid), 18, "get_ijk_from_subgrid")
    set_tiles_once(ZONE_SOLVE_EM, grid, ids, ide, jds, jde, ips, ipe, jps, jpe, config=config_flags)
    tile_bounds = tuple(_tile_bounds(grid, ij, ips, ipe, jps, jpe) for ij in range(1, _tile_count(grid) + 1))
    num_sound_steps = _compute_num_sound_steps(grid, config_flags)
    return DirectStepPlan(
        ijk=ijk,
        subgrid_ijk=subgrid_ijk,
        tile_bounds=tile_bounds,
        rk_order=_read_int(config_flags, "rk_ord", 3),
        num_sound_steps=num_sound_steps,
        dts=_read_float(grid, "dt", 0.0) / float(num_sound_steps),
        num_moist=int(num_moist),
        num_chem=int(num_chem),
        num_tracer=int(num_tracer),
        num_scalar=int(num_scalar),
        callbacks=callbacks,
    )


def _seed_runtime_bounds(state: dict[str, Any], plan: DirectStepPlan) -> None:
    (
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
        imsx,
        imex,
        jmsx,
        jmex,
        kmsx,
        kmex,
        ipsx,
        ipex,
        jpsx,
        jpex,
        kpsx,
        kpex,
        imsy,
        imey,
        jmsy,
        jmey,
        kmsy,
        kmey,
        ipsy,
        ipey,
        jpsy,
        jpey,
        kpsy,
        kpey,
    ) = plan.bounds
    tile_its, tile_ite, tile_jts, tile_jte = plan.tile_bounds[0]
    state.update(
        {
            "ids": ids,
            "ide": ide,
            "jds": jds,
            "jde": jde,
            "kds": kds,
            "kde": kde,
            "ims": ims,
            "ime": ime,
            "jms": jms,
            "jme": jme,
            "kms": kms,
            "kme": kme,
            "ips": ips,
            "ipe": ipe,
            "jps": jps,
            "jpe": jpe,
            "kps": kps,
            "kpe": kpe,
            "its": tile_its,
            "ite": tile_ite,
            "jts": tile_jts,
            "jte": tile_jte,
            "kts": kps,
            "kte": kpe,
            "imsx": imsx,
            "imex": imex,
            "jmsx": jmsx,
            "jmex": jmex,
            "kmsx": kmsx,
            "kmex": kmex,
            "ipsx": ipsx,
            "ipex": ipex,
            "jpsx": jpsx,
            "jpex": jpex,
            "kpsx": kpsx,
            "kpex": kpex,
            "imsy": imsy,
            "imey": imey,
            "jmsy": jmsy,
            "jmey": jmey,
            "kmsy": kmsy,
            "kmey": kmey,
            "ipsy": ipsy,
            "ipey": ipey,
            "jpsy": jpsy,
            "jpey": jpey,
            "kpsy": kpsy,
            "kpey": kpey,
        }
    )


def _initialize_direct_step_stage_context(
    *,
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    fields: Mapping[str, Any],
    moist: Any = None,
    chem: Any = None,
    tracer: Any = None,
    scalar: Any = None,
    advance_clock: bool = True,
) -> tuple[dict[str, Any], DirectStepStageContext]:
    state: dict[str, Any] = dict(fields)
    state.update(
        {
            "grid": grid,
            "config_flags": config_flags,
            "moist": state.get("moist") if moist is None else moist,
            "chem": state.get("chem") if chem is None else chem,
            "tracer": state.get("tracer") if tracer is None else tracer,
            "scalar": state.get("scalar") if scalar is None else scalar,
        }
    )
    _seed_runtime_bounds(state, plan)

    num_3d_m = int(plan.num_moist)
    num_3d_c = int(plan.num_chem)
    num_3d_s = int(plan.num_scalar)
    if _read_int(grid, "dfi_stage", 0) == 2:
        num_3d_m = P_QV
        num_3d_s = PARAM_FIRST_SCALAR - 1
    state.update({"num_moist": num_3d_m, "num_chem": num_3d_c, "num_tracer": int(plan.num_tracer), "num_scalar": num_3d_s})

    if "curr_secs" in state:
        curr_secs = state["curr_secs"]
    else:
        current_time = _default_current_time(grid)
        sim_start_time = _default_sim_start_time(grid)
        curr_secs = _default_real_time(current_time - sim_start_time)
    if "curr_secs2" in state:
        curr_secs2 = state["curr_secs2"]
    else:
        current_time2 = _default_current_time(grid)
        start_time = _default_start_time(grid)
        curr_secs2 = _default_real_time(current_time2 - start_time)
    _write_if_concrete(grid, "current_time", curr_secs)

    nested = _read_bool(grid, "nested", _read_bool(config_flags, "nested", False))
    if _read_bool(config_flags, "use_adaptive_time_step") and ((not nested) or (nested and abs(_read_float(grid, "dtbc", 0.0)) < 0.0001)):
        raise NotImplementedError("direct core does not support adaptive time step yet")
    adapt_step_flag = False
    restart_flag = bool(_read(grid, "restart_alarm", False))
    diag_flag = bool(_read(grid, "history_alarm", False))
    if _read_int(config_flags, "nwp_diagnostics", 0) == 1:
        diag_flag = True
    state.update(
        {
            "adapt_step_flag": adapt_step_flag,
            "restart_flag": restart_flag,
            "diag_flag": diag_flag,
            "curr_secs": curr_secs,
            "curr_secs2": curr_secs2,
        }
    )

    if advance_clock:
        previous_itimestep = state.get("itimestep", _read_int(grid, "itimestep", 0))
        if _is_traced_value(previous_itimestep):
            previous_itimestep_array = jnp.asarray(previous_itimestep, dtype=jnp.int32)
            next_itimestep = previous_itimestep_array + jnp.asarray(1, dtype=previous_itimestep_array.dtype)
        else:
            next_itimestep = int(previous_itimestep) + 1
        state["itimestep"] = next_itimestep
        _write_if_concrete(grid, "itimestep", next_itimestep)
        previous_dtbc = state.get("dtbc", _read_float(grid, "dtbc", 0.0))
        next_dtbc = jnp.asarray(previous_dtbc, dtype=jnp.float32) + jnp.asarray(_read_float(grid, "dt", 0.0), dtype=jnp.float32)
        state["dtbc"] = next_dtbc
        _write_if_concrete(grid, "dtbc", next_dtbc)

    tile_count = len(plan.tile_bounds)
    max_vert_cfl_tmp = [0.0 for _ in range(tile_count)]
    max_horiz_cfl_tmp = [0.0 for _ in range(tile_count)]
    if _read_bool(config_flags, "use_adaptive_time_step"):
        _write(grid, "max_cfl_val", 0.0)
        _write(grid, "max_horiz_cfl", 0.0)
        _write(grid, "max_vert_cfl", 0.0)

    specified = _read_bool(config_flags, "specified")
    nested_cfg = _read_bool(config_flags, "nested")
    specified_or_nested = specified or nested_cfg
    fused_boundary_groups = _FUSED_SMALL_STEP_BOUNDARIES and len(plan.tile_bounds) == 1
    boundary_config = _boundary_config(config_flags)
    context = DirectStepStageContext(
        bounds=plan.bounds,
        num_3d_m=num_3d_m,
        num_3d_c=num_3d_c,
        num_3d_s=num_3d_s,
        f_flux=_read_int(config_flags, "do_avgflx_cugd", 0) == 1,
        curr_secs=curr_secs,
        curr_secs2=curr_secs2,
        adapt_step_flag=adapt_step_flag,
        restart_flag=restart_flag,
        diag_flag=diag_flag,
        specified=specified,
        constant_bc=_read_bool(config_flags, "constant_bc"),
        specified_or_nested=specified_or_nested,
        non_hydrostatic=_read_bool(config_flags, "non_hydrostatic", True),
        h_sca_adv_order=_read_int(config_flags, "h_sca_adv_order", 5),
        h_mom_adv_order=_read_int(config_flags, "h_mom_adv_order", 5),
        fused_boundary_groups=fused_boundary_groups,
        fused_small_boundaries=fused_boundary_groups,
        boundary_config=boundary_config,
        spec_zone=_read_int(config_flags, "spec_zone", _read_int(config_flags, "spec_bdy_width", 1)),
        spec_bdy_width=_read_int(config_flags, "spec_bdy_width", int(state.get("spec_bdy_width", 5))),
        max_vert_cfl_tmp=max_vert_cfl_tmp,
        max_horiz_cfl_tmp=max_horiz_cfl_tmp,
    )
    return state, context


def _run_post_rk_physics_stage_family(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    num_3d_m: int,
    num_3d_s: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
) -> None:
    """Run the post-RK physics/output-boundary family as one explicit stage.

    Keeping this family behind a stable function boundary is the first
    production-core split away from the monolithic direct step.  The sequence is
    still the exact direct-mode order, but it can now be lifted, audited, and
    eventually compiled independently from the RK/small-step dycore family.
    """

    _run_post_rk_physics_stage_group(
        state=state,
        callbacks=callbacks,
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        num_3d_m=num_3d_m,
        num_3d_s=num_3d_s,
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        stage_names=POST_RK_PHYSICS_STAGE_GROUPS,
    )


def _run_post_rk_physics_stage_group(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    num_3d_m: int,
    num_3d_s: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    stage_names: tuple[str, ...],
) -> None:
    """Run selected post-RK physics groups behind stable stage boundaries."""

    selected = set(stage_names)
    cb = lambda name: _callback(callbacks, name)
    advance_ppt = cb("advance_ppt")
    phy_prep_part2 = cb("phy_prep_part2")
    moist_physics_prep_em = cb("moist_physics_prep_em")
    microphysics_driver = cb("microphysics_driver")
    moist_physics_finish_em = cb("moist_physics_finish_em")
    calc_p_rho_phi_final = cb("calc_p_rho_phi:final")
    set_phys_bc_dry_2 = cb("set_phys_bc_dry_2")

    if "post_advance_ppt" in selected:
        for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
            _apply(advance_ppt, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if "post_phy_prep_part2" in selected:
        for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
            _apply(phy_prep_part2, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if _read_int(config_flags, "mp_physics", 0) != 0:
        if "post_moist_prep" in selected:
            for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
                _apply(moist_physics_prep_em, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

        if "post_microphysics" in selected:
            _apply(microphysics_driver, state)

        if "post_moist_finish" in selected:
            for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
                _apply(moist_physics_finish_em, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
                _apply(cb("set_physical_bc3d:h_diabatic"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

        if "post_calc_p_rho_phi_final" in selected:
            for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
                calc_its, calc_ite, calc_jts, calc_jte = _final_calc_p_rho_phi_bounds(
                    grid,
                    config_flags,
                    tile_its,
                    tile_ite,
                    tile_jts,
                    tile_jte,
                    ids,
                    ide,
                    jds,
                    jde,
                )
                _apply(calc_p_rho_phi_final, state, ij=ij, its=calc_its, ite=calc_ite, jts=calc_jts, jte=calc_jte)

    if "post_physical_bc" in selected:
        for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
            _apply(set_phys_bc_dry_2, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            _apply(cb("set_physical_bc3d:tke_1"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            _apply(cb("set_physical_bc3d:tke_2"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            if num_3d_m >= PARAM_FIRST_SCALAR:
                _apply(cb("set_physical_bc3d:moist"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            if num_3d_s >= PARAM_FIRST_SCALAR:
                _apply(cb("set_physical_bc3d:scalar"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)


def _run_final_boundary_stage_family(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    boundary_config: Any,
    specified_or_nested: bool,
    fused_boundary_groups: bool,
    spec_bdy_width: int,
    spec_zone: int,
    num_3d_m: int,
    curr_secs: float,
    curr_secs2: float,
    diag_flag: bool,
) -> None:
    """Run final lateral-boundary, surface-W, and post-RK finalize stages."""

    (
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
        imsx,
        imex,
        jmsx,
        jmex,
        kmsx,
        kmex,
        ipsx,
        ipex,
        jpsx,
        jpex,
        kpsx,
        kpex,
        imsy,
        imey,
        jmsy,
        jmey,
        kmsy,
        kmey,
        ipsy,
        ipey,
        jpsy,
        jpey,
        kpsy,
        kpey,
    ) = plan.bounds

    cb = lambda name: _callback(callbacks, name)
    set_w_surface = cb("set_w_surface")
    after_all_rk_steps = cb("after_all_rk_steps")

    if specified_or_nested and fused_boundary_groups and PARAM_FIRST_SCALAR <= num_3d_m <= P_QV:
        tile_its, tile_ite, tile_jts, tile_jte = plan.tile_bounds[0]
        qvapor = jnp.asarray(state["moist"])[..., 1]
        (
            state["u_2"],
            state["v_2"],
            state["t_2"],
            state["ph_2"],
            state["mu_2"],
            qvapor,
        ) = _fused_spec_bdy_final_prognostics(
            state["u_2"],
            state["v_2"],
            state["t_2"],
            state["ph_2"],
            state["mu_2"],
            qvapor,
            state["muus"],
            state["muvs"],
            state["muts"],
            state["c1h"],
            state["c2h"],
            state["c1f"],
            state["c2f"],
            state["msfuy"],
            state["msfvx"],
            state["msfty"],
            state["u_bxs"],
            state["u_bxe"],
            state["u_bys"],
            state["u_bye"],
            state["u_btxs"],
            state["u_btxe"],
            state["u_btys"],
            state["u_btye"],
            state["v_bxs"],
            state["v_bxe"],
            state["v_bys"],
            state["v_bye"],
            state["v_btxs"],
            state["v_btxe"],
            state["v_btys"],
            state["v_btye"],
            state["t_bxs"],
            state["t_bxe"],
            state["t_bys"],
            state["t_bye"],
            state["t_btxs"],
            state["t_btxe"],
            state["t_btys"],
            state["t_btye"],
            state["ph_bxs"],
            state["ph_bxe"],
            state["ph_bys"],
            state["ph_bye"],
            state["ph_btxs"],
            state["ph_btxe"],
            state["ph_btys"],
            state["ph_btye"],
            state["mu_bxs"],
            state["mu_bxe"],
            state["mu_bys"],
            state["mu_bye"],
            state["mu_btxs"],
            state["mu_btxe"],
            state["mu_btys"],
            state["mu_btye"],
            state["qvapor_bxs"],
            state["qvapor_bxe"],
            state["qvapor_bys"],
            state["qvapor_bye"],
            state["qvapor_btxs"],
            state["qvapor_btxe"],
            state["qvapor_btys"],
            state["qvapor_btye"],
            state["dtbc"],
            config_flags=boundary_config,
            spec_bdy_width=spec_bdy_width,
            spec_zone=spec_zone,
            ids=ids,
            ide=ide,
            jds=jds,
            jde=jde,
            kds=kds,
            kde=kde,
            ims=ims,
            ime=ime,
            jms=jms,
            jme=jme,
            kms=kms,
            kme=kme,
            ips=ips,
            ipe=ipe,
            jps=jps,
            jpe=jpe,
            kps=kps,
            kpe=kpe,
            its=tile_its,
            ite=tile_ite,
            jts=tile_jts,
            jte=tile_jte,
            kts=kps,
            kte=kpe,
        )
        state["moist"] = _replace_last_axis_component(state["moist"], 1, qvapor)
        state["qvapor"] = qvapor
    elif specified_or_nested:
        for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
            for stage_name in (
                "spec_bdy_final:u",
                "spec_bdy_final:v",
                "spec_bdy_final:t",
                "spec_bdy_final:ph",
                "spec_bdy_final:mu",
            ):
                _apply(cb(stage_name), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            if num_3d_m >= PARAM_FIRST_SCALAR:
                _apply(cb("spec_bdy_final:moist"), state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    for ij, _bounds in enumerate(plan.tile_bounds, start=1):
        _apply(set_w_surface, state, ij=ij)

    _apply(
        after_all_rk_steps,
        state,
        curr_secs=curr_secs,
        curr_secs2=curr_secs2,
        diag_flag=diag_flag,
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        ips=ips,
        ipe=ipe,
        jps=jps,
        jpe=jpe,
        kps=kps,
        kpe=kpe,
        imsx=imsx,
        imex=imex,
        jmsx=jmsx,
        jmex=jmex,
        kmsx=kmsx,
        kmex=kmex,
        ipsx=ipsx,
        ipex=ipex,
        jpsx=jpsx,
        jpex=jpex,
        kpsx=kpsx,
        kpex=kpex,
        imsy=imsy,
        imey=imey,
        jmsy=jmsy,
        jmey=jmey,
        kmsy=kmsy,
        kmey=kmey,
        ipsy=ipsy,
        ipey=ipey,
        jpsy=jpsy,
        jpey=jpey,
        kpsy=kpsy,
        kpey=kpey,
    )


def _run_rk_step_stage_family(
    *,
    state: dict[str, Any],
    callbacks: Mapping[str, StageFn],
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    boundary_config: Any,
    boundary_kwargs_fn: Callable[[float], dict[str, Any]],
    physical_bc_kwargs_fn: Callable[[], dict[str, Any]],
    num_3d_m: int,
    num_3d_s: int,
    specified: bool,
    constant_bc: bool,
    specified_or_nested: bool,
    non_hydrostatic: bool,
    fused_boundary_groups: bool,
    fused_small_boundaries: bool,
    spec_zone: int,
    f_flux: bool,
    restart_flag: bool,
    curr_secs: float,
    adapt_step_flag: bool,
    max_vert_cfl_tmp: list[float],
    max_horiz_cfl_tmp: list[float],
    rk_steps: tuple[int, ...] | None = None,
    initialize: bool = True,
    stop_before_first_rk_part1: bool = False,
    stop_after_first_rk_part1: bool = False,
    start_after_first_rk_part1: bool = False,
    stage_names: tuple[str, ...] | None = None,
) -> None:
    """Run the RK/small-step family behind one explicit orchestration boundary."""

    selected = set(RK_STEP_STAGE_FAMILY if stage_names is None else stage_names)
    cb = lambda name: _callback(callbacks, name)
    initialize_moist_old = cb("initialize_moist_old")
    zero_bdytend = cb("zero_bdytend")
    set_moist_old_bc = cb("set_physical_bc3d:moist_old")
    rk_step_prep = cb("rk_step_prep")
    rk_phys_bc_dry_1 = cb("rk_phys_bc_dry_1")
    set_rho_bc = cb("set_physical_bc3d:rho")
    set_al_bc = cb("set_physical_bc3d:al")
    set_ph_2_bc = cb("set_physical_bc3d:ph_2")
    first_rk_step_part1 = cb("first_rk_step_part1")
    first_rk_step_part2 = cb("first_rk_step_part2")
    rk_tendency = cb("rk_tendency")
    relax_bdy_dry = cb("relax_bdy_dry")
    dry_boundary_tendency = cb("dry_boundary_tendency")
    rk_addtend_dry = cb("rk_addtend_dry")
    spec_bdy_dry = cb("spec_bdy_dry")
    small_step_prep = cb("small_step_prep")
    calc_p_rho = cb("calc_p_rho")
    calc_coef_w = cb("calc_coef_w")
    advance_uv = cb("advance_uv")
    advance_mu_t = cb("advance_mu_t")
    advance_w = cb("advance_w")
    sumflux = cb("sumflux")
    spec_bdyupdate_u = cb("spec_bdyupdate:u")
    spec_bdyupdate_v = cb("spec_bdyupdate:v")
    spec_bdyupdate_t = cb("spec_bdyupdate:t")
    spec_bdyupdate_mu_2 = cb("spec_bdyupdate:mu_2")
    spec_bdyupdate_muts = cb("spec_bdyupdate:muts")
    spec_bdyupdate_w = cb("spec_bdyupdate:w")
    spec_bdyupdate_ph = cb("spec_bdyupdate_ph")
    zero_grad_bdy = cb("zero_grad_bdy")
    calc_mu_uv_1 = cb("calc_mu_uv_1")
    small_step_finish = cb("small_step_finish")
    moist_scalar_advance = cb("moist_scalar_advance")
    scalar_advance = cb("scalar_advance")
    calc_p_rho_phi = cb("calc_p_rho_phi")
    rk_phys_bc_dry_2 = cb("rk_phys_bc_dry_2")

    if initialize and "initialize_moist_old" in selected:
        for ij, (tile_its, tile_ite, tile_jts, tile_jte) in enumerate(plan.tile_bounds, start=1):
            if specified and constant_bc:
                _apply(zero_bdytend, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            _apply(initialize_moist_old, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if initialize and "initialize_moist_old" in selected and num_3d_m >= PARAM_FIRST_SCALAR:
        for ij, _bounds in enumerate(plan.tile_bounds, start=1):
            _apply(set_moist_old_bc, state, ij=ij)

    active_rk_steps = tuple(range(1, plan.rk_order + 1)) if rk_steps is None else rk_steps
    for rk_step in active_rk_steps:
        _seed_rk_step_size_fields(state=state, grid=grid, plan=plan, rk_step=rk_step)
        dts_rk = state["dts_rk"]
        number_of_small_timesteps = int(state["number_of_small_timesteps"])

        if not start_after_first_rk_part1:
            if "rk_step_prep" in selected:
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    _apply(rk_step_prep, state, ij=ij)

            if "rk_phys_bc_dry_1" in selected:
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    _apply(rk_phys_bc_dry_1, state, ij=ij)
                    if fused_boundary_groups:
                        state["rho"], state["al"], state["ph_2"] = _fused_initial_dry_physical_bc(
                            state["rho"],
                            state["al"],
                            state["ph_2"],
                            **physical_bc_kwargs_fn(),
                        )
                    else:
                        _apply(set_rho_bc, state, ij=ij)
                        _apply(set_al_bc, state, ij=ij)
                        _apply(set_ph_2_bc, state, ij=ij)

            if rk_step == 1 and stop_before_first_rk_part1:
                continue

            if rk_step == 1 and "first_rk_step_part1" in selected:
                if _DIRECT_FIRST_RK_PART1:
                    state.update({"f_flux": f_flux, "restart_flag": restart_flag, "curr_secs": curr_secs})
                    _run_first_rk_step_part1_direct(state=state, callbacks=callbacks, plan=plan, f_flux=f_flux)
                else:
                    _apply(first_rk_step_part1, state, f_flux=f_flux, restart_flag=restart_flag, curr_secs=curr_secs)
            if rk_step == 1 and "first_rk_step_part2" in selected:
                _run_first_rk_step_part2_stage(
                    state=state,
                    callbacks=callbacks,
                    adapt_step_flag=adapt_step_flag,
                    curr_secs=curr_secs,
                )
            if rk_step == 1 and stop_after_first_rk_part1:
                continue

        if "rk_tendency" in selected:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(rk_tendency, state, ij=ij)
                if _read_bool(config_flags, "use_adaptive_time_step"):
                    max_vert_cfl_tmp[ij - 1] = max(max_vert_cfl_tmp[ij - 1], float(state.get("max_vert_cfl", 0.0)))
                    max_horiz_cfl_tmp[ij - 1] = max(max_horiz_cfl_tmp[ij - 1], float(state.get("max_horiz_cfl", 0.0)))

        if "rk_tendency" in selected and _read_bool(config_flags, "use_adaptive_time_step"):
            max_horiz = max(max_horiz_cfl_tmp) if max_horiz_cfl_tmp else 0.0
            max_vert = max(max_vert_cfl_tmp) if max_vert_cfl_tmp else 0.0
            _write(grid, "max_horiz_cfl", max_horiz)
            _write(grid, "max_vert_cfl", max_vert)
            _write(grid, "max_cfl_val", max(max_horiz, max_vert, _read_float(grid, "max_cfl_val", 0.0)))

        if "dry_boundary_tendency" in selected:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                if specified_or_nested and dry_boundary_tendency is not None:
                    _apply(dry_boundary_tendency, state, ij=ij)
                else:
                    if specified_or_nested and rk_step == 1:
                        _apply(relax_bdy_dry, state, ij=ij)
                    _apply(rk_addtend_dry, state, ij=ij)
                    if specified_or_nested:
                        _apply(spec_bdy_dry, state, ij=ij)

        if "small_step_prep" in selected:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(small_step_prep, state, ij=ij)
                _apply(calc_p_rho, state, ij=ij, iteration=0)
                if non_hydrostatic:
                    _apply(calc_coef_w, state, ij=ij)

            if fused_boundary_groups:
                state["ij"] = 1
                (
                    state["ru_tend"],
                    state["rv_tend"],
                    state["ph_2"],
                    state["al"],
                    state["p"],
                    state["t_1"],
                    state["t_save"],
                    state["mu_1"],
                    state["mu_2"],
                    state["mudf"],
                ) = _fused_pre_small_physical_bc(
                    state["ru_tend"],
                    state["rv_tend"],
                    state["ph_2"],
                    state["al"],
                    state["p"],
                    state["t_1"],
                    state["t_save"],
                    state["mu_1"],
                    state["mu_2"],
                    state["mudf"],
                    **physical_bc_kwargs_fn(),
                )
            else:
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    for stage_name in (
                        "set_physical_bc3d:ru_tend",
                        "set_physical_bc3d:rv_tend",
                        "set_physical_bc3d:ph_2",
                        "set_physical_bc3d:al",
                        "set_physical_bc3d:p",
                        "set_physical_bc3d:t_1",
                        "set_physical_bc3d:t_save",
                        "set_physical_bc2d:mu_1",
                        "set_physical_bc2d:mu_2",
                        "set_physical_bc2d:mudf",
                    ):
                        _apply(cb(stage_name), state, ij=ij)

        compiled_small_loop_done = False
        if "small_step_loop" in selected and len(plan.tile_bounds) == 1 and fused_small_boundaries:
            compiled_small_loop_done = _run_direct_small_step_loop(
                state=state,
                config_flags=boundary_config,
                rk_step=rk_step,
                dts_rk=dts_rk,
                number_of_small_timesteps=number_of_small_timesteps,
                bounds=plan.bounds,
                spec_zone=spec_zone,
                specified=specified,
                non_hydrostatic=non_hydrostatic,
            )
        if "small_step_loop" in selected and not compiled_small_loop_done and len(plan.tile_bounds) == 1:
            compiled_small_loop_done = _run_small_step_loop(
                state=state,
                callbacks=callbacks,
                rk_step=rk_step,
                dts_rk=dts_rk,
                number_of_small_timesteps=number_of_small_timesteps,
                bounds=plan.bounds,
            )
        if "small_step_loop" in selected and not compiled_small_loop_done:
            boundary_kwargs = boundary_kwargs_fn(dts_rk)
            physical_bc_kwargs = physical_bc_kwargs_fn()
            for iteration in range(1, number_of_small_timesteps + 1):
                state["iteration"] = iteration
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    _apply(advance_uv, state, ij=ij)
                if specified_or_nested and fused_small_boundaries:
                    state["u_2"], state["v_2"] = _fused_spec_bdyupdate_uv(
                        state["u_2"],
                        state["ru_tend"],
                        state["v_2"],
                        state["rv_tend"],
                        **boundary_kwargs,
                    )
                else:
                    for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                        if specified_or_nested:
                            _apply(spec_bdyupdate_u, state, ij=ij)
                            _apply(spec_bdyupdate_v, state, ij=ij)
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    _apply(advance_mu_t, state, ij=ij)
                if specified_or_nested and fused_small_boundaries:
                    state["t_2"], state["mu_2"], state["muts"] = _fused_spec_bdyupdate_t_mu(
                        state["t_2"],
                        state["t_tend"],
                        state["mu_2"],
                        state["muts"],
                        state["mu_tend"],
                        **boundary_kwargs,
                    )
                    if non_hydrostatic:
                        for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                            _apply(advance_w, state, ij=ij)
                else:
                    for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                        if specified_or_nested:
                            _apply(spec_bdyupdate_t, state, ij=ij)
                            _apply(spec_bdyupdate_mu_2, state, ij=ij)
                            _apply(spec_bdyupdate_muts, state, ij=ij)
                        if non_hydrostatic:
                            _apply(advance_w, state, ij=ij)
                for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                    _apply(sumflux, state, ij=ij)
                    if specified_or_nested and non_hydrostatic:
                        if fused_small_boundaries:
                            state["ph_2"], state["w_2"] = _fused_spec_bdyupdate_ph_w(
                                state["ph_save"],
                                state["ph_2"],
                                state["ph_tend"],
                                state["mu_tend"],
                                state["muts"],
                                state["c1f"],
                                state["c2f"],
                                state["w_2"],
                                state["rw_tend"],
                                state["_wrf_real_rounding_token"],
                                **boundary_kwargs,
                            )
                        else:
                            _apply(spec_bdyupdate_ph, state, ij=ij)
                            if specified:
                                _apply(zero_grad_bdy, state, ij=ij)
                            else:
                                _apply(spec_bdyupdate_w, state, ij=ij)
                    _apply(calc_p_rho, state, ij=ij, iteration=iteration)
                if fused_small_boundaries:
                    (
                        state["ph_2"],
                        state["al"],
                        state["p"],
                        state["muts"],
                        state["mu_2"],
                        state["mudf"],
                    ) = _fused_small_physical_bc(
                        state["ph_2"],
                        state["al"],
                        state["p"],
                        state["muts"],
                        state["mu_2"],
                        state["mudf"],
                        **physical_bc_kwargs,
                    )
                else:
                    for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                        for stage_name in (
                            "set_physical_bc3d:ph_2:small",
                            "set_physical_bc3d:al:small",
                            "set_physical_bc3d:p:small",
                            "set_physical_bc2d:muts:small",
                            "set_physical_bc2d:mu_2:small",
                            "set_physical_bc2d:mudf:small",
                        ):
                            _apply(cb(stage_name), state, ij=ij)

        if "small_step_finish" in selected:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(calc_mu_uv_1, state, ij=ij)
                _apply(small_step_finish, state, ij=ij)
                if rk_step == plan.rk_order:
                    if fused_boundary_groups:
                        state["ru_m"], state["rv_m"], state["ww_m"], state["mut"], state["muts"] = _fused_post_small_final_physical_bc(
                            state["ru_m"],
                            state["rv_m"],
                            state["ww_m"],
                            state["mut"],
                            state["muts"],
                            **physical_bc_kwargs_fn(),
                        )
                    else:
                        for stage_name in (
                            "set_physical_bc3d:ru_m",
                            "set_physical_bc3d:rv_m",
                            "set_physical_bc3d:ww_m",
                            "set_physical_bc2d:mut",
                            "set_physical_bc2d:muts",
                        ):
                            _apply(cb(stage_name), state, ij=ij)

        if (
            "moist_scalar_advance" in selected
            and
            num_3d_m >= PARAM_FIRST_SCALAR
            and rk_step == plan.rk_order
            and _read_int(config_flags, "moist_adv_opt", ORIGINAL) not in {ORIGINAL, WENO_SCALAR}
        ):
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(cb("rk_update_scalar_pd:moist_old"), state, ij=ij)
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(set_moist_old_bc, state, ij=ij)

        if (
            "scalar_advance" in selected
            and
            num_3d_s >= PARAM_FIRST_SCALAR
            and rk_step == plan.rk_order
            and _read_int(config_flags, "scalar_adv_opt", ORIGINAL) not in {ORIGINAL, WENO_SCALAR}
        ):
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(cb("rk_update_scalar_pd:scalar_old"), state, ij=ij)
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(cb("set_physical_bc3d:scalar_old"), state, ij=ij)

        if "moist_scalar_advance" in selected and num_3d_m >= PARAM_FIRST_SCALAR:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(moist_scalar_advance, state, ij=ij)

        if "scalar_advance" in selected and num_3d_s >= PARAM_FIRST_SCALAR:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(scalar_advance, state, ij=ij)

        if "calc_p_rho_phi" in selected:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(calc_p_rho_phi, state, ij=ij)

        if "rk_phys_bc_dry_2" in selected and rk_step < plan.rk_order:
            for ij, _bounds in enumerate(plan.tile_bounds, start=1):
                _apply(rk_phys_bc_dry_2, state, ij=ij)
                if num_3d_m >= PARAM_FIRST_SCALAR:
                    _apply(cb("set_physical_bc3d:moist"), state, ij=ij)
                if num_3d_s >= PARAM_FIRST_SCALAR:
                    _apply(cb("set_physical_bc3d:scalar"), state, ij=ij)


def run_direct_step(
    *,
    plan: DirectStepPlan,
    grid: Any,
    config_flags: Any,
    fields: Mapping[str, Any],
    moist: Any = None,
    chem: Any = None,
    tracer: Any = None,
    scalar: Any = None,
) -> SolveEmResult:
    """Run one prepared direct WRF-JAX step for the configured single domain."""

    callbacks = plan.callbacks
    state, context = _initialize_direct_step_stage_context(
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        fields=fields,
        moist=moist,
        chem=chem,
        tracer=tracer,
        scalar=scalar,
    )

    _run_rk_step_stage_family(
        state=state,
        callbacks=callbacks,
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        boundary_config=context.boundary_config,
        boundary_kwargs_fn=context.boundary_kwargs,
        physical_bc_kwargs_fn=context.physical_bc_kwargs,
        num_3d_m=context.num_3d_m,
        num_3d_s=context.num_3d_s,
        specified=context.specified,
        constant_bc=context.constant_bc,
        specified_or_nested=context.specified_or_nested,
        non_hydrostatic=context.non_hydrostatic,
        fused_boundary_groups=context.fused_boundary_groups,
        fused_small_boundaries=context.fused_small_boundaries,
        spec_zone=context.spec_zone,
        f_flux=context.f_flux,
        restart_flag=context.restart_flag,
        curr_secs=context.curr_secs,
        adapt_step_flag=context.adapt_step_flag,
        max_vert_cfl_tmp=context.max_vert_cfl_tmp,
        max_horiz_cfl_tmp=context.max_horiz_cfl_tmp,
    )

    _run_post_rk_physics_stage_family(
        state=state,
        callbacks=callbacks,
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        num_3d_m=context.num_3d_m,
        num_3d_s=context.num_3d_s,
        ids=context.ids,
        ide=context.ide,
        jds=context.jds,
        jde=context.jde,
    )

    _run_final_boundary_stage_family(
        state=state,
        callbacks=callbacks,
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        boundary_config=context.boundary_config,
        specified_or_nested=context.specified_or_nested,
        fused_boundary_groups=context.fused_boundary_groups,
        spec_bdy_width=context.spec_bdy_width,
        spec_zone=context.spec_zone,
        num_3d_m=context.num_3d_m,
        curr_secs=context.curr_secs,
        curr_secs2=context.curr_secs2,
        diag_flag=context.diag_flag,
    )

    if context.h_mom_adv_order > 6 or context.h_sca_adv_order > 6:
        raise ValueError("direct core invalid horizontal advection order")
    if _read_bool(config_flags, "coupler_on", False):
        _write(grid, "just_read_auxinput4", bool(_read(grid, "auxinput4_alarm", False)))
    if _read_int(grid, "id", 1) == 1:
        _write(grid, "just_read_boundary", bool(_read(grid, "boundary_alarm", False)))

    return SolveEmResult(
        grid=grid,
        config_flags=config_flags,
        fields=state,
        ijk=plan.ijk,
        subgrid_ijk=plan.subgrid_ijk,
        rk_order=plan.rk_order,
        num_sound_steps=plan.num_sound_steps,
        dts=plan.dts,
        adapt_step_flag=context.adapt_step_flag,
        restart_flag=context.restart_flag,
        diag_flag=context.diag_flag,
        curr_secs=context.curr_secs,
        curr_secs2=context.curr_secs2,
        max_vert_cfl_tmp=tuple(context.max_vert_cfl_tmp),
        max_horiz_cfl_tmp=tuple(context.max_horiz_cfl_tmp),
        events=(),
    )


__all__ = [
    "DirectStepPlan",
    "FIRST_RK_PART1_DIRECT_STAGE_FAMILY",
    "FINAL_BOUNDARY_STAGE_FAMILY",
    "POST_RK_PHYSICS_STAGE_GROUPS",
    "POST_RK_PHYSICS_STAGE_FAMILY",
    "RK_COMPILED_SUBSTAGES",
    "RK1_PRE_FIRST_PART1_STAGE_FAMILY",
    "RK_STEP_STAGE_FAMILY",
    "build_direct_step_plan",
    "run_direct_step",
    "_seed_rk_step_size_fields",
]
