"""Strict runtime assembly for the configured WRF-JAX case."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache, partial
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.boundary.physical_bc import set_physical_bc2d, set_physical_bc3d
from wrf_jax.boundary.flow_dep_bdy import flow_dep_bdy
from wrf_jax.boundary.lbc_fcx_gcx import lbc_fcx_gcx
from wrf_jax.boundary.spec_bdy_final import spec_bdy_final
from wrf_jax.boundary.spec_bdytend import overlay_spec_bdytend, spec_bdytend
from wrf_jax.boundary.spec_bdyupdate import spec_bdyupdate
from wrf_jax.boundary.zero_grad_bdy import zero_grad_bdy
from wrf_jax.compat.get_ijk_from_grid1 import get_ijk_from_grid1
from wrf_jax.compat.module_date_time import wrf_atotime
from wrf_jax.core.executable_cache import reusable_keyword_jit, reusable_terminal_jit
from wrf_jax.core.devices import batch_partitions, configured_devices, submit_device_batches
from wrf_jax.core.optimization import semantic_call
from wrf_jax.core.real_arithmetic import real_assignment
from wrf_jax.core.wrf_math import (
    scalar_powf_for_wrf,
    sqrt_for_wrf,
    vector_powf_for_wrf,
    wrf_glibc_logf_positive_normal,
    wrf_sse4_logf_positive_normal,
)
from wrf_jax.data_io import CaseData, assert_state_finite, load_case
from wrf_jax.dyn_em.module_advect_em_fast import (
    advect_scalar_pd_vectorized,
    advect_scalar_vectorized,
    advect_u_vectorized,
    advect_v_vectorized,
    advect_w_vectorized,
    subtract_scalar_pd_prelimit_fluxes,
)
from wrf_jax.dyn_em.module_big_step_utilities_em import (
    _wrf_patch_scalar_tail_offsets,
    calc_mu_uv_1,
    calc_p_rho_phi,
    calc_ww_cp,
    conv_t_tendf_to_moist,
    initialize_moist_old,
)
from wrf_jax.dyn_em.module_big_step_utilities_em import pg_buoy_w, w_damp
from wrf_jax.dyn_em.module_big_step_utilities_em import zero_tend, zero_tend2d
from wrf_jax.dyn_em.module_big_step_utilities_em_fast import coriolis_vectorized, curvature_vectorized, horizontal_pressure_gradient_vectorized, rhs_ph_vectorized
from wrf_jax.dyn_em.module_bc_em import (
    relax_bdy_dry,
    relax_bdy_scalar,
    rk_phys_bc_dry_1,
    rk_phys_bc_dry_2,
    set_phys_bc_dry_2,
    set_w_surface,
    spec_bdy_dry,
    spec_bdy_scalar,
    spec_bdyupdate_ph,
)
from wrf_jax.dyn_em.module_big_step_utilities_em import phy_prep, phy_prep_part2
from wrf_jax.dyn_em.module_diffusion_em import (
    CalculateKmKhResult,
    DeformAndDivResult,
    DiffMetricsResult,
    HorizontalDiffusion2Result,
    compute_diff_metrics,
    phy_bc,
)
from wrf_jax.dyn_em.module_diffusion_em_fast import (
    cal_deform_and_div_vectorized,
    calculate_km_kh_vectorized,
    horizontal_diffusion_2_without_theta_vectorized,
    horizontal_diffusion_s_vectorized,
)
from wrf_jax.dyn_em.module_em import G3SCHEME, GDSCHEME, GFSCHEME, HELDSUAREZ, NTIEDTKESCHEME, WENO_MOM, calculate_phy_tend
from wrf_jax.dyn_em.module_em import rk_step_prep
from wrf_jax.dyn_em.module_em import init_zero_tendency
from wrf_jax.dyn_em.module_em import MONOTONIC, POSITIVEDEF, WENO_SCALAR, WENOPD_SCALAR
from wrf_jax.dyn_em.module_em import rk_addtend_dry
from wrf_jax.dyn_em.module_em import rk_tendency
from wrf_jax.dyn_em.module_em import rk_update_scalar
from wrf_jax.dyn_em.module_first_rk_step_part1 import first_rk_step_part1
from wrf_jax.dyn_em.module_first_rk_step_part2 import first_rk_step_part2
from wrf_jax.dyn_em.module_ieva_em import chk_ieva, ww_split
from wrf_jax.dyn_em.module_small_step_em import advance_mu_t, advance_uv, advance_w, calc_coef_w, calc_p_rho, small_step_finish, small_step_prep, sumflux
from wrf_jax.dyn_em.solve_em import SolveEmResult, solve_em
from wrf_jax.esmf_time.types import BaseTime, ESMFClock, ESMFClockInt, ESMFTimeInterval
from wrf_jax.physics.module_cumulus_driver import cumulus_driver
from wrf_jax.physics.moist_coupling import finish_microphysics
from wrf_jax.physics.moist_preparation import prepare_microphysics
from wrf_jax.physics.module_fddagd_driver import fddagd_driver
from wrf_jax.physics.module_bl_gwdo import gwdo
from wrf_jax.physics.module_physics_addtendc import advance_ppt, update_phy_ten
from wrf_jax.physics.module_physics_init import landuse_init, read_landuse_tables
from wrf_jax.physics.module_pbl_driver import pbl_driver
from wrf_jax.physics.module_ra_clwrf_support import load_camgases_table, read_camgases
from wrf_jax.physics.module_sf_sfcdiags import sfcdiags
from wrf_jax.physics.module_sf_noahdrv import lsm, lsminit, soil_veg_gen_parm
from wrf_jax.physics.module_sf_noah_seaice import seaice_noah
from wrf_jax.physics.module_mp_thompson import (
    calc_effect_rad,
    calc_refl10cm,
    load_thompson_tables,
    rsif,
    rsif_rounded,
    rslf,
    rslf_rounded,
    thompson_column_driver,
    thompson_grid_driver_defaults,
)
from wrf_jax.physics.module_radiation_driver import radiation_driver
from wrf_jax.physics.module_ra_rrtmg_lw import (
    load_rrtmg_lw_cloud_tables,
    load_rrtmg_lw_compressed_lookup_tables,
    load_rrtmg_lw_setcoef_tables,
    lwdatinit,
    rrtmg_lw_pade_tables,
    rrtmg_lwrad_wrf_columns,
)
from wrf_jax.physics.module_ra_rrtmg_sw import (
    load_rrtmg_sw_cloud_tables,
    load_rrtmg_sw_compressed_lookup_tables,
    rrtmg_sw_pade_tables,
    rrtmg_swrad_wrf_columns,
    swdatinit,
)
from wrf_jax.physics.module_shallowcu_driver import shallowcu_driver
from wrf_jax.physics.module_surface_driver import accumulate_surface_precipitation, myjsfc_seaice_wrapper, surface_driver
from wrf_jax.physics.myjpbl import EPSQ2, myjpbl, myjpblinit
from wrf_jax.physics.myjsfc import CP, G, P608, R_D, myjsfc_vectorized, myjsfcinit

R_V = 461.6
T0 = 300.0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


_INLINE_DIRECT_STAGE_JITS = _env_flag("WRF_PY_INLINE_DIRECT_STAGE_JITS", False)
_UPDATE_NOAH_NONFEEDBACK_DIAGNOSTICS = _env_flag("WRF_PY_UPDATE_NOAH_NONFEEDBACK_DIAGNOSTICS", False)
_USE_JAX_RRTMG_LW = _env_flag("WRF_PY_USE_JAX_RRTMG_LW", True)
_USE_JAX_RRTMG_SW = _env_flag("WRF_PY_USE_JAX_RRTMG_SW", True)
_JIT_RRTMG_LW = _env_flag("WRF_PY_JIT_RRTMG_LW", True)
_JIT_RRTMG_SW = _env_flag("WRF_PY_JIT_RRTMG_SW", True)
_RRTMG_CHUNK_COLUMNS = _env_int("WRF_PY_RRTMG_CHUNK_COLUMNS", 16384)
_RRTMG_CHUNK_WORKERS = max(1, _env_int("WRF_PY_RRTMG_CHUNK_WORKERS", 4))
_RRTMG_PARALLEL_LW_SW = _env_flag("WRF_PY_RRTMG_PARALLEL_LW_SW", True)
_RRTMG_SPARSE_CLOUDS = _env_flag("WRF_PY_RRTMG_SPARSE_CLOUDS", False)
_THOMPSON_CHUNK_COLUMNS = _env_int("WRF_PY_THOMPSON_CHUNK_COLUMNS", 65536)
_THOMPSON_CHUNK_WORKERS = max(1, _env_int("WRF_PY_THOMPSON_CHUNK_WORKERS", 1))

CalculateKmKhResult = jax.tree_util.register_dataclass(
    CalculateKmKhResult,
    data_fields=("xkmh", "xkmv", "xkhh", "xkhv", "bn2", "moist", "tke", "hpbl", "xkmv_meso"),
    meta_fields=(),
)
DeformAndDivResult = jax.tree_util.register_dataclass(
    DeformAndDivResult,
    data_fields=("div", "defor11", "defor22", "defor33", "defor12", "defor13", "defor23", "nba_rij"),
    meta_fields=(),
)
DiffMetricsResult = jax.tree_util.register_dataclass(
    DiffMetricsResult,
    data_fields=("z", "rdz", "rdzw", "zx", "zy"),
    meta_fields=(),
)
HorizontalDiffusion2Result = jax.tree_util.register_dataclass(
    HorizontalDiffusion2Result,
    data_fields=(
        "rt_tendf",
        "ru_tendf",
        "rv_tendf",
        "rw_tendf",
        "tke_tendf",
        "moist_tendf",
        "chem_tendf",
        "scalar_tendf",
        "tracer_tendf",
        "nba_mij",
    ),
    meta_fields=(),
)


class MissingKernelImplementationError(RuntimeError):
    """Raised when the strict runtime reaches an unwired numerical stage."""


@dataclass(frozen=True)
class _StaticRuntimeConfig:
    """Hashable subset of namelist flags used inside jitted hot kernels."""

    rk_ord: int = 3
    wrf_reference_nproc_x: int = 1
    nested: bool = False
    specified: bool = False
    periodic_x: bool = False
    periodic_y: bool = False
    polar: bool = False
    open_xs: bool = False
    open_xe: bool = False
    open_ys: bool = False
    open_ye: bool = False
    symmetric_xs: bool = False
    symmetric_xe: bool = False
    symmetric_ys: bool = False
    symmetric_ye: bool = False
    h_mom_adv_order: int = 5
    v_mom_adv_order: int = 3
    h_sca_adv_order: int = 5
    v_sca_adv_order: int = 3
    phi_adv_z: int = 1
    km_opt: int = 4
    diff_opt: int = 2
    sfs_opt: int = 0
    m_opt: int = 0
    damp_opt: int = 0
    c_s: float = 0.25
    dampcoef: float = 0.0
    zdamp: float = 1.0
    mix_full_fields: bool = False
    tke_mix2_off: bool = False
    moist_mix2_off: bool = False
    chem_mix2_off: bool = False
    scalar_mix2_off: bool = False
    tracer_mix2_off: bool = False
    p_m11: int = 1
    p_m12: int = 2
    p_m13: int = 3
    p_m22: int = 4
    p_m23: int = 5
    use_theta_m: int = 0
    fft_filter_lat: float = 0.0
    w_crit_cfl: float = 2.0
    zadvect_implicit: int = 0
    w_damping: int = 0
    use_adaptive_time_step: bool = False
    ra_lw_physics: int = 0
    ra_sw_physics: int = 0
    mp_physics: int = 0
    cu_physics: int = 0
    shcu_physics: int = 0
    bl_pbl_physics: int = 0
    grid_fdda: int = 0
    radt: float = 0.0
    stepra: int = 1
    ra_call_offset: int = 0
    icloud: int = 1
    cldovrlp: int = 2
    idcor: int = 0
    ghg_input: int = 1
    no_mp_heating: int = 0
    mp_tend_lim: float = 10.0
    do_radar_ref: int = 0
    calc_clean_atm_diag: int = 0


def _static_runtime_config(config_flags: Any) -> _StaticRuntimeConfig:
    return _StaticRuntimeConfig(
        rk_ord=int(getattr(config_flags, "rk_ord", 3)),
        wrf_reference_nproc_x=int(
            getattr(config_flags, "wrf_reference_nproc_x", 1)
        ),
        nested=bool(getattr(config_flags, "nested", False)),
        specified=bool(getattr(config_flags, "specified", False)),
        periodic_x=bool(getattr(config_flags, "periodic_x", False)),
        periodic_y=bool(getattr(config_flags, "periodic_y", False)),
        polar=bool(getattr(config_flags, "polar", False)),
        open_xs=bool(getattr(config_flags, "open_xs", False)),
        open_xe=bool(getattr(config_flags, "open_xe", False)),
        open_ys=bool(getattr(config_flags, "open_ys", False)),
        open_ye=bool(getattr(config_flags, "open_ye", False)),
        symmetric_xs=bool(getattr(config_flags, "symmetric_xs", False)),
        symmetric_xe=bool(getattr(config_flags, "symmetric_xe", False)),
        symmetric_ys=bool(getattr(config_flags, "symmetric_ys", False)),
        symmetric_ye=bool(getattr(config_flags, "symmetric_ye", False)),
        h_mom_adv_order=int(getattr(config_flags, "h_mom_adv_order", 5)),
        v_mom_adv_order=int(getattr(config_flags, "v_mom_adv_order", 3)),
        h_sca_adv_order=int(getattr(config_flags, "h_sca_adv_order", 5)),
        v_sca_adv_order=int(getattr(config_flags, "v_sca_adv_order", 3)),
        phi_adv_z=int(getattr(config_flags, "phi_adv_z", 1)),
        km_opt=int(getattr(config_flags, "km_opt", 4)),
        diff_opt=int(getattr(config_flags, "diff_opt", 2)),
        sfs_opt=int(getattr(config_flags, "sfs_opt", 0)),
        m_opt=int(getattr(config_flags, "m_opt", 0)),
        damp_opt=int(getattr(config_flags, "damp_opt", 0)),
        c_s=float(getattr(config_flags, "c_s", 0.25)),
        dampcoef=float(getattr(config_flags, "dampcoef", 0.0)),
        zdamp=float(getattr(config_flags, "zdamp", 1.0)),
        mix_full_fields=bool(getattr(config_flags, "mix_full_fields", False)),
        tke_mix2_off=bool(getattr(config_flags, "tke_mix2_off", False)),
        moist_mix2_off=bool(getattr(config_flags, "moist_mix2_off", False)),
        chem_mix2_off=bool(getattr(config_flags, "chem_mix2_off", False)),
        scalar_mix2_off=bool(getattr(config_flags, "scalar_mix2_off", False)),
        tracer_mix2_off=bool(getattr(config_flags, "tracer_mix2_off", False)),
        p_m11=int(getattr(config_flags, "p_m11", 1)),
        p_m12=int(getattr(config_flags, "p_m12", 2)),
        p_m13=int(getattr(config_flags, "p_m13", 3)),
        p_m22=int(getattr(config_flags, "p_m22", 4)),
        p_m23=int(getattr(config_flags, "p_m23", 5)),
        use_theta_m=int(getattr(config_flags, "use_theta_m", 0)),
        fft_filter_lat=float(getattr(config_flags, "fft_filter_lat", 0.0)),
        w_crit_cfl=float(getattr(config_flags, "w_crit_cfl", 2.0)),
        zadvect_implicit=int(getattr(config_flags, "zadvect_implicit", 0)),
        w_damping=int(getattr(config_flags, "w_damping", 0)),
        use_adaptive_time_step=bool(getattr(config_flags, "use_adaptive_time_step", False)),
        ra_lw_physics=int(getattr(config_flags, "ra_lw_physics", 0)),
        ra_sw_physics=int(getattr(config_flags, "ra_sw_physics", 0)),
        mp_physics=int(getattr(config_flags, "mp_physics", 0)),
        cu_physics=int(getattr(config_flags, "cu_physics", 0)),
        shcu_physics=int(getattr(config_flags, "shcu_physics", 0)),
        bl_pbl_physics=int(getattr(config_flags, "bl_pbl_physics", 0)),
        grid_fdda=int(getattr(config_flags, "grid_fdda", 0)),
        radt=float(getattr(config_flags, "radt", 0.0)),
        stepra=int(getattr(config_flags, "stepra", 1)),
        ra_call_offset=int(getattr(config_flags, "ra_call_offset", 0)),
        icloud=int(getattr(config_flags, "icloud", 1)),
        cldovrlp=int(getattr(config_flags, "cldovrlp", 2)),
        idcor=int(getattr(config_flags, "idcor", 0)),
        ghg_input=int(getattr(config_flags, "ghg_input", 1)),
        no_mp_heating=int(getattr(config_flags, "no_mp_heating", 0)),
        mp_tend_lim=float(getattr(config_flags, "mp_tend_lim", 10.0)),
        do_radar_ref=int(getattr(config_flags, "do_radar_ref", 0)),
        calc_clean_atm_diag=int(getattr(config_flags, "calc_clean_atm_diag", 0)),
    )


def _static_config(state: Mapping[str, Any]) -> _StaticRuntimeConfig:
    value = state.get("_static_config_flags")
    if isinstance(value, _StaticRuntimeConfig):
        return value
    return _static_runtime_config(state["config_flags"])


_STATIC_DOMAIN_ARGNAMES = (
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
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)

_RRTMG_LW_STATIC_ARGNAMES = (
    "p_top",
    "ghg_input",
    "case_dir",
    "cldovrlp",
    "idcor",
    "icloud",
    "has_reqc",
    "has_reqi",
    "has_reqs",
)

_RRTMG_SW_STATIC_ARGNAMES = (
    "ghg_input",
    "cldovrlp",
    "idcor",
    "icloud",
    "has_reqc",
    "has_reqi",
    "has_reqs",
    "sf_surface_physics",
)

_rrtmg_lwrad_wrf_columns_jit = reusable_keyword_jit(
    rrtmg_lwrad_wrf_columns,
    static_argnames=_RRTMG_LW_STATIC_ARGNAMES,
)

_rrtmg_swrad_wrf_columns_jit = reusable_keyword_jit(
    rrtmg_swrad_wrf_columns,
    static_argnames=_RRTMG_SW_STATIC_ARGNAMES,
)


@lru_cache(maxsize=8)
def _paired_rrtmg_program(lw_static: tuple, sw_static: tuple) -> Callable[..., Any]:
    """Submit the existing LW/SW kernels together, with tables as inputs."""
    lw_config, sw_config = dict(lw_static), dict(sw_static)

    def radiation_pair(lw_values, sw_values):
        return (
            _rrtmg_lwrad_wrf_columns_jit(**lw_config, **lw_values),
            _rrtmg_swrad_wrf_columns_jit(**sw_config, **sw_values),
        )

    return reusable_terminal_jit(radiation_pair, context={
        "program": "paired_rrtmg", "lw_static": lw_static, "sw_static": sw_static,
    })


def _run_paired_rrtmg(lw_kwargs: Mapping[str, Any], sw_kwargs: Mapping[str, Any]) -> Any:
    lw_static = tuple((name, lw_kwargs[name]) for name in _RRTMG_LW_STATIC_ARGNAMES if name in lw_kwargs)
    sw_static = tuple((name, sw_kwargs[name]) for name in _RRTMG_SW_STATIC_ARGNAMES if name in sw_kwargs)
    lw_values = {name: value for name, value in lw_kwargs.items() if name not in _RRTMG_LW_STATIC_ARGNAMES}
    sw_values = {name: value for name, value in sw_kwargs.items() if name not in _RRTMG_SW_STATIC_ARGNAMES}
    return _paired_rrtmg_program(lw_static, sw_static)(lw_values, sw_values)


@lru_cache(maxsize=1)
def _runtime_rrtmg_lw_tables() -> tuple[Any, Any, Any, Any, Any]:
    return (
        load_rrtmg_lw_compressed_lookup_tables(),
        load_rrtmg_lw_cloud_tables(),
        load_rrtmg_lw_setcoef_tables(),
        lwdatinit(),
        rrtmg_lw_pade_tables(),
    )


@lru_cache(maxsize=1)
def _runtime_rrtmg_sw_tables() -> tuple[Any, Any, Any, Any]:
    return (
        load_rrtmg_sw_compressed_lookup_tables(),
        load_rrtmg_sw_cloud_tables(),
        swdatinit(),
        rrtmg_sw_pade_tables(),
    )


def preload_runtime_jax_tables(case_dir: str | Path | None = None) -> None:
    if _USE_JAX_RRTMG_LW:
        _runtime_rrtmg_lw_tables()
    if _USE_JAX_RRTMG_SW:
        _runtime_rrtmg_sw_tables()
    if case_dir is not None:
        case_dir_str = str(case_dir)
        load_thompson_tables(case_dir_str)


_PHYSICAL_BC_DOMAIN_ARGNAMES = (
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

_RUNTIME_BOUND_NAMES = (
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
    "imsx",
    "imex",
    "jmsx",
    "jmex",
    "kmsx",
    "kmex",
    "ipsx",
    "ipex",
    "jpsx",
    "jpex",
    "kpsx",
    "kpex",
    "imsy",
    "imey",
    "jmsy",
    "jmey",
    "kmsy",
    "kmey",
    "ipsy",
    "ipey",
    "jpsy",
    "jpey",
    "kpsy",
    "kpey",
)

@partial(jax.jit, static_argnums=(19,), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _calc_coef_w_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = calc_coef_w(*args, **kwargs)
    return result.a, result.alpha, result.gamma


@partial(jax.jit, static_argnums=(19,), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=False)
def _calc_coef_w_rounded_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = calc_coef_w(*args, **kwargs)
    return result.a, result.alpha, result.gamma


def _materialize_wrf_real(value: Any, rounding_token: Any) -> Any:
    return real_assignment(value, rounding_token)


@jax.jit
def _wrf_swdown_from_gsw(
    gsw: Any,
    albedo: Any,
    rounding_token: Any | None = None,
) -> Any:
    """Reconstruct WRF ``SWDOWN = GSW / (1 - ALBEDO)`` in default REAL."""

    def store(value: Any) -> Any:
        value32 = jnp.asarray(value, dtype=jnp.float32)
        if rounding_token is not None:
            return _materialize_wrf_real(value32, rounding_token)
        return jax.lax.optimization_barrier(value32)

    denominator = store(store(jnp.float32(1.0)) - store(albedo))
    return store(store(gsw) / denominator)


_RK_STEP_PREP_STATIC_ARGNUMS = (0, 1, 29, 30, 31, *range(32, 50))


@partial(jax.jit, static_argnums=_RK_STEP_PREP_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _rk_step_prep_jit_tuple(*args: Any) -> tuple[Any, ...]:
    result = rk_step_prep(*args)
    return (
        result.ru,
        result.rv,
        result.rw,
        result.ww,
        result.php,
        result.alt,
        result.muu,
        result.muv,
        result.mut,
        result.cqu,
        result.cqv,
        result.cqw,
    )


_CALC_WW_CP_STATIC_ARGNUMS = (6, 7, *range(16, 34))


@partial(jax.jit, static_argnums=_CALC_WW_CP_STATIC_ARGNUMS, inline=False)
def _calc_ww_cp_jit(*args: Any) -> Any:
    return calc_ww_cp(*args)


@partial(jax.jit, static_argnums=(39, 40, 41, 42), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _advance_uv_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    result = advance_uv(*args, **kwargs)
    return result.u, result.v


@partial(jax.jit, static_argnums=(39, 40, 41, 42), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=False)
def _advance_uv_rounded_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    result = advance_uv(*args, **kwargs)
    return result.u, result.v


def _advance_uv_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    result = advance_uv(*args, **kwargs)
    return result.u, result.v


@partial(jax.jit, static_argnums=(49,), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=False)
def _small_step_prep_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = small_step_prep(*args, **kwargs)
    return (
        result.u_1,
        result.u_2,
        result.v_1,
        result.v_2,
        result.w_1,
        result.w_2,
        result.t_1,
        result.t_2,
        result.ph_1,
        result.ph_2,
        result.mu_1,
        result.mu_2,
        result.muus,
        result.muvs,
        result.muts,
        result.mudf,
        result.u_save,
        result.v_save,
        result.w_save,
        result.t_save,
        result.ph_save,
        result.mu_save,
        result.ww_save,
        result.c2a,
        result.u_term_new,
        result.u_term_old,
        result.v_term_new,
        result.v_term_old,
    )


def _small_step_prep_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = small_step_prep(*args, **kwargs)
    return (
        result.u_1,
        result.u_2,
        result.v_1,
        result.v_2,
        result.w_1,
        result.w_2,
        result.t_1,
        result.t_2,
        result.ph_1,
        result.ph_2,
        result.mu_1,
        result.mu_2,
        result.muus,
        result.muvs,
        result.muts,
        result.mudf,
        result.u_save,
        result.v_save,
        result.w_save,
        result.t_save,
        result.ph_save,
        result.mu_save,
        result.ww_save,
        result.c2a,
        result.u_term_new,
        result.u_term_old,
        result.v_term_new,
        result.v_term_old,
    )


@partial(jax.jit, static_argnums=(23, 24), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _calc_p_rho_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
    step_is_zero = bool(args[24])
    calc_args = (*args[:24], 0 if step_is_zero else 1)
    result = calc_p_rho(*calc_args, **kwargs)
    return result.al, result.p, result.ph, result.pm1


def _calc_p_rho_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
    step_is_zero = bool(args[24])
    calc_args = (*args[:24], 0 if step_is_zero else 1)
    result = calc_p_rho(*calc_args, **kwargs)
    return result.al, result.p, result.ph, result.pm1


@partial(jax.jit, static_argnums=(45,), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _advance_mu_t_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = advance_mu_t(*args, **kwargs)
    return (
        result.ww,
        result.ww_1,
        result.mu,
        result.muave,
        result.muts,
        result.mudf,
        result.uam,
        result.vam,
        result.wwam,
        result.t,
        result.t_ave,
    )


def _advance_mu_t_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = advance_mu_t(*args, **kwargs)
    return (
        result.ww,
        result.ww_1,
        result.mu,
        result.muave,
        result.muts,
        result.mudf,
        result.uam,
        result.vam,
        result.wwam,
        result.t,
        result.t_ave,
    )


@partial(jax.jit, static_argnums=(48, 49), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _advance_w_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = advance_w(*args, **kwargs)
    return result.t_2ave, result.w, result.ph


@partial(jax.jit, static_argnums=(48, 49), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=False)
def _advance_w_rounded_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = advance_w(*args, **kwargs)
    return result.t_2ave, result.w, result.ph, result.rhs_adv_local


@partial(jax.jit, static_argnums=(48, 49), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=False)
def _advance_w_rounded_diagnostic_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Return internal solve arrays for the standalone WRF comparison harness."""

    result = advance_w(*args, **kwargs)
    return (
        result.t_2ave,
        result.w,
        result.ph,
        result.raw_w_local,
        result.solved_w_local,
        result.pressure_upper_local,
        result.pressure_lower_local,
        result.pressure_part_local,
        result.pressure_numerator_local,
        result.pressure_denominator_local,
        result.pressure_ratio_local,
        result.pressure_rhs_delta_local,
        result.pressure_ph_delta_local,
        result.pressure_rhs_term_local,
        result.pressure_ph_term_local,
        result.pressure_bracket_local,
        result.rhs_base_local,
        result.wdwn_local,
        result.rhs_adv_local,
        result.rhs_dec_local,
        result.thermal_upper_local,
        result.thermal_lower_local,
        result.buoyancy_part_local,
        result.damping_htop_local,
        result.damping_hbot_local,
        result.damping_hk_local,
        result.damping_angle_local,
        result.damping_sine_local,
        result.damping_weight_local,
        result.damping_mass_local,
        result.damping_term_local,
        result.damping_numerator_local,
        result.damping_denominator_local,
        result.damped_w_local,
    )


def _advance_w_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = advance_w(*args, **kwargs)
    return result.t_2ave, result.w, result.ph


@partial(jax.jit, static_argnums=(25, 26), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _sumflux_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = sumflux(*args, **kwargs)
    return result.ru_m, result.rv_m, result.ww_m


def _sumflux_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    result = sumflux(*args, **kwargs)
    return result.ru_m, result.rv_m, result.ww_m


@partial(jax.jit, static_argnums=(43, 44), static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _small_step_finish_jit_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = small_step_finish(*args, **kwargs)
    return (
        result.u_2,
        result.v_2,
        result.w_2,
        result.t_2,
        result.ph_2,
        result.ww,
        result.mu_2,
        result.mu_1,
        result.mut,
        result.muts,
        result.muu,
        result.muus,
        result.muv,
        result.muvs,
        result.mu_save,
    )


def _small_step_finish_eager_tuple(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    result = small_step_finish(*args, **kwargs)
    return (
        result.u_2,
        result.v_2,
        result.w_2,
        result.t_2,
        result.ph_2,
        result.ww,
        result.mu_2,
        result.mu_1,
        result.mut,
        result.muts,
        result.muu,
        result.muus,
        result.muv,
        result.muvs,
        result.mu_save,
    )


_CALC_MU_SMALL_FINISH_STATIC_ARGNAMES = (
    "config_flags",
    "rk_step",
    "rk_order",
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
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)


@partial(jax.jit, static_argnames=_CALC_MU_SMALL_FINISH_STATIC_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _calc_mu_uv_small_step_finish_jit_tuple(
    *,
    u_2: Any,
    u_1: Any,
    v_2: Any,
    v_1: Any,
    w_2: Any,
    w_1: Any,
    t_2: Any,
    t_1: Any,
    ph_2: Any,
    ph_1: Any,
    ww: Any,
    ww1: Any,
    mu_2: Any,
    mu_1: Any,
    mut: Any,
    muts: Any,
    muu: Any,
    muv: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    t_save: Any,
    ph_save: Any,
    mu_save: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    h_diabatic: Any,
    number_of_small_timesteps: int,
    dts: Any,
    config_flags: Any,
    rk_step: int,
    rk_order: int,
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
) -> tuple[Any, ...]:
    muus, muvs = calc_mu_uv_1(
        config_flags,
        muts,
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
    result = small_step_finish(
        u_2,
        u_1,
        v_2,
        v_1,
        w_2,
        w_1,
        t_2,
        t_1,
        ph_2,
        ph_1,
        ww,
        ww1,
        mu_2,
        mu_1,
        mut,
        muts,
        muu,
        muus,
        muv,
        muvs,
        c1h,
        c2h,
        c1f,
        c2f,
        c3h,
        c4h,
        c3f,
        c4f,
        u_save,
        v_save,
        w_save,
        t_save,
        ph_save,
        mu_save,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        msftx,
        msfty,
        h_diabatic,
        number_of_small_timesteps,
        dts,
        rk_step,
        rk_order,
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
        rounding_token=rounding_token,
    )
    return (
        result.u_2,
        result.v_2,
        result.w_2,
        result.t_2,
        result.ph_2,
        result.ww,
        result.mu_2,
        result.muus,
        result.muvs,
    )


_PHY_PREP_STATIC_ARGNUMS = (0, 17, 22, *range(23, 41))
_CALCULATE_PHY_TEND_STATIC_ARGNUMS = (0, 38, 41, *range(42, 60))


@partial(
    jax.jit,
    static_argnums=_PHY_PREP_STATIC_ARGNUMS,
    inline=False,
)
def _phy_prep_jit_tuple(*args: Any) -> tuple[Any, ...]:
    return phy_prep(*args[:-1], rounding_token=args[-1])


@partial(jax.jit, static_argnums=_CALCULATE_PHY_TEND_STATIC_ARGNUMS, inline=True)
def _calculate_phy_tend_jit_tuple(*args: Any) -> tuple[Any, ...]:
    return calculate_phy_tend(*args)


_CALCULATE_PHY_TEND_PBL_ONLY_STATIC_ARGNAMES = (
    "ids",
    "ide",
    "jds",
    "jde",
    "kds",
    "kde",
    "ims",
    "jms",
    "kms",
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)


@partial(jax.jit, static_argnames=_CALCULATE_PHY_TEND_PBL_ONLY_STATIC_ARGNAMES, inline=True)
def _calculate_phy_tend_pbl_only_jit_tuple(
    c1: Any,
    c2: Any,
    mut: Any,
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rounding_token: Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Specialized physics-tendency mass scaling for the configured MYJ path."""

    del ids, jds, kds
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    if itf < int(its) or jtf < int(jts) or ktf < int(kts):
        return rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten

    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mut_arr = jnp.asarray(mut)
    i0 = int(its) - int(ims)
    i1 = itf - int(ims) + 1
    k0 = int(kts) - int(kms)
    k1 = ktf - int(kms) + 1
    j0 = int(jts) - int(jms)
    j1 = jtf - int(jms) + 1
    i_slice = slice(i0, i1)
    k_slice = slice(k0, k1)
    j_slice = slice(j0, j1)
    # Preserve the two WRF REAL operations.  XLA otherwise contracts this
    # multiply-add and changes the mass factor by one float32 ULP.
    factor_product = _materialize_wrf_real(
        c1_arr[k_slice][None, :, None] * mut_arr[i_slice, j_slice][:, None, :],
        rounding_token,
    )
    factor = _materialize_wrf_real(
        factor_product + c2_arr[k_slice][None, :, None], rounding_token
    )

    def multiply(field: Any) -> Any:
        arr = jnp.asarray(field)
        scaled = _materialize_wrf_real(
            arr[i_slice, k_slice, j_slice] * factor, rounding_token
        )
        inner = jnp.concatenate((arr[i_slice, :k0, j_slice], scaled, arr[i_slice, k1:, j_slice]), axis=1)
        slab = jnp.concatenate((arr[i_slice, :, :j0], inner, arr[i_slice, :, j1:]), axis=2)
        return jnp.concatenate((arr[:i0, :, :], slab, arr[i1:, :, :]), axis=0)

    return (
        multiply(rublten),
        multiply(rvblten),
        multiply(rthblten),
        multiply(rqvblten),
        multiply(rqcblten),
        multiply(rqiblten),
    )


_ADVECT_STATIC_ARGNUMS = (9, 10, 19, 20, *range(22, 40))
_ADVECT_PD_STATIC_ARGNUMS = (13, 14, 15, 24, 25, 27, *range(28, 46))
_RHS_PH_STATIC_ARGNUMS = (*range(16, 20), 27, 28, *range(29, 47))
_HPG_STATIC_ARGNUMS = (*range(18, 25), 31, 32, 33, *range(34, 52))
_CORIOLIS_STATIC_ARGNUMS = (6, *range(19, 37))
_CURVATURE_STATIC_ARGNUMS = (9, 19, 20, *range(21, 39))
_HDIFF2_STATIC_ARGNUMS = (6, 8, 10, 12, 16, 23, 38, 39, 40, 45, 46, 47, *range(53, 71))
_HDIFF_S_STATIC_ARGNUMS = (1, 10, 11, 14, 15, 16, 24, *range(25, 43))
_CAL_DEFORM_STATIC_ARGNUMS = (0, 12, 21, 22, 29, 30, 31, *range(34, 52))
_CALCULATE_KM_KH_STATIC_ARGNUMS = (0, 1, 2, 3, 4, 10, 11, 28, 29, 32, 33, 34, 35, 36, 37, 38, *range(46, 64))
_COMPUTE_DIFF_METRICS_STATIC_ARGNUMS = (0, 8, 9, *range(10, 28))
_RELAX_BDY_DRY_STATIC_ARGNUMS = (0, 66, 67, 68, *range(72, 96))
_SPEC_BDY_DRY_STATIC_ARGNUMS = (0, 49, 50, *range(51, 75))
_SPEC_BDY_DRY_OVERLAY_STATIC_ARGNUMS = (6, 55, 56, *range(57, 81))
_SPEC_BDY_DRY_OVERLAY_MASKS_STATIC_ARGNUMS = (12, 61, 62, *range(63, 87))
_RK_ADDTEND_DRY_STATIC_ARGNUMS = (17, *range(29, 53))
_RELAX_BDY_DRY_ARG_COUNT = 97
_RK_ADDTEND_DRY_ARG_COUNT = 54
_SPEC_BDY_DRY_ARG_COUNT = 75
_DRY_BOUNDARY_MASK_ARG_COUNT = 6
_FUSED_DRY_RK1_STATIC_ARGNUMS = tuple(
    sorted(
        {
            *(_DRY_BOUNDARY_MASK_ARG_COUNT + index for index in _RELAX_BDY_DRY_STATIC_ARGNUMS),
            *(_DRY_BOUNDARY_MASK_ARG_COUNT + _RELAX_BDY_DRY_ARG_COUNT + index for index in _RK_ADDTEND_DRY_STATIC_ARGNUMS),
            *(
                _DRY_BOUNDARY_MASK_ARG_COUNT
                + _RELAX_BDY_DRY_ARG_COUNT
                + _RK_ADDTEND_DRY_ARG_COUNT
                + index
                for index in _SPEC_BDY_DRY_STATIC_ARGNUMS
            ),
        }
    )
)
_FUSED_DRY_NONRK1_STATIC_ARGNUMS = tuple(
    sorted(
        {
            *(_DRY_BOUNDARY_MASK_ARG_COUNT + index for index in _RK_ADDTEND_DRY_STATIC_ARGNUMS),
            *(_DRY_BOUNDARY_MASK_ARG_COUNT + _RK_ADDTEND_DRY_ARG_COUNT + index for index in _SPEC_BDY_DRY_STATIC_ARGNUMS),
        }
    )
)
_CALC_MU_UV_1_STATIC_ARGNUMS = (0, *range(2, 22))
_RK_UPDATE_SCALAR_STATIC_ARGNUMS = (0, 1, 17, 18, 19, 20, 21, *range(22, 40))
_MOIST_SCALAR_ADVANCE_STATIC_ARGNAMES = (
    "config_flags",
    "rk_step",
    "rk_order",
    "adv_opt",
    "time_step",
    "do_final_pd",
    "do_cu_rqvften",
    "do_moist_boundary",
    "rdx",
    "rdy",
    "dt_split",
    "dt_rk",
    "spec_bdy_width",
    "spec_zone",
    "relax_zone",
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
_BATCHED_NONQV_MOIST_STATIC_ARGNAMES = (
    "config_flags",
    "rk_step",
    "do_final_pd",
    "do_flow_dep_boundary",
    "time_step",
    "rdx",
    "rdy",
    "dt_rk",
    "spec_zone",
    *_STATIC_DOMAIN_ARGNAMES,
    "ips",
    "ipe",
    "jps",
    "jpe",
    "kps",
    "kpe",
)
_RK_TENDENCY_DIRECT_STATIC_ARGNAMES = (
    "rk_step",
    "rk_order",
    "time_step",
    "config_flags",
    "non_hydrostatic",
    "top_lid",
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
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
    "rdx",
    "rdy",
    "dt",
    "cfn",
    "cfn1",
    "cf1",
    "cf2",
    "cf3",
)
_RK_TENDENCY_DIRECT_NO_IEVA_STATIC_ARGNAMES = tuple(
    name for name in _RK_TENDENCY_DIRECT_STATIC_ARGNAMES if name != "rk_step"
)


_JIT_COMPUTE_DIFF_METRICS = _env_flag("WRF_PY_JIT_COMPUTE_DIFF_METRICS", True)
_JIT_CAL_DEFORM_AND_DIV = _env_flag("WRF_PY_JIT_CAL_DEFORM_AND_DIV", True)
_JIT_CALCULATE_KM_KH = _env_flag("WRF_PY_JIT_CALCULATE_KM_KH", True)
_JIT_HORIZONTAL_DIFFUSION_2 = _env_flag("WRF_PY_JIT_HORIZONTAL_DIFFUSION_2", True)
_JIT_SMALL_STEP_PREP = _env_flag("WRF_PY_JIT_SMALL_STEP_PREP", True)
# XLA fusion changes WRF REAL statement rounding in phy_prep. Keep the
# statement-ordered Python path as the default until a JIT-safe fence exists.
_JIT_PHY_PREP = _env_flag("WRF_PY_JIT_PHY_PREP", False)
_JIT_CALC_P_RHO = _env_flag("WRF_PY_JIT_CALC_P_RHO", True)
_JIT_ADVANCE_UV = _env_flag("WRF_PY_JIT_ADVANCE_UV", True)
_JIT_ADVANCE_MU_T = _env_flag("WRF_PY_JIT_ADVANCE_MU_T", True)
_JIT_ADVANCE_W = _env_flag("WRF_PY_JIT_ADVANCE_W", True)
_JIT_SUMFLUX = _env_flag("WRF_PY_JIT_SUMFLUX", True)
_JIT_SMALL_STEP_FINISH = _env_flag("WRF_PY_JIT_SMALL_STEP_FINISH", True)
_JIT_CALCULATE_PHY_TEND = _env_flag("WRF_PY_JIT_CALCULATE_PHY_TEND", True)
_JIT_RK_UPDATE_SCALAR = _env_flag("WRF_PY_JIT_RK_UPDATE_SCALAR", True)
_DIRECT_FIRST_RK_PART2 = _env_flag("WRF_PY_DIRECT_FIRST_RK_PART2", True)
_DIRECT_RK_TENDENCY = _env_flag("WRF_PY_DIRECT_RK_TENDENCY", True)
_FUSED_RK_TENDENCY = _env_flag("WRF_PY_FUSED_RK_TENDENCY", True)
_INLINE_RK_TENDENCY_DIRECT = _env_flag("WRF_PY_INLINE_RK_TENDENCY_DIRECT", True)
_FUSED_SPEC_BDY_DRY_OVERLAY = _env_flag("WRF_PY_FUSED_SPEC_BDY_DRY_OVERLAY", True)
_FUSED_DRY_BOUNDARY_TENDENCY_STAGE = _env_flag("WRF_PY_FUSED_DRY_BOUNDARY_TENDENCY_STAGE", True)
_FUSED_DRY_BOUNDARY_SINGLE_KERNEL = _env_flag("WRF_PY_FUSED_DRY_BOUNDARY_SINGLE_KERNEL", False)
_STATIC_DRY_BOUNDARY_MASKS = _env_flag("WRF_PY_STATIC_DRY_BOUNDARY_MASKS", False)
_FUSED_MOIST_SCALAR_ADVANCE = _env_flag("WRF_PY_FUSED_MOIST_SCALAR_ADVANCE", True)
_BATCHED_NONQV_MOIST_ADVANCE = _env_flag("WRF_PY_BATCHED_NONQV_MOIST_ADVANCE", False)
_FUSED_CALC_MU_SMALL_FINISH = _env_flag("WRF_PY_FUSED_CALC_MU_SMALL_FINISH", True)
_FAST_PBL_ONLY_PHY_TEND = _env_flag("WRF_PY_FAST_PBL_ONLY_PHY_TEND", True)
_FAST_FIXED_MYJ_PBL_STAGE = _env_flag("WRF_PY_FAST_FIXED_MYJ_PBL_STAGE", True)
_FAST_FIXED_MYJ_NOAH_SURFACE_STAGE = _env_flag("WRF_PY_FAST_FIXED_MYJ_NOAH_SURFACE_STAGE", True)
_LOCAL_SURFACE_PBL_UPDATES = _env_flag("WRF_PY_LOCAL_SURFACE_PBL_UPDATES", False)
_EAGER_PHYSICS_LOCAL_CROP = _env_flag("WRF_PY_EAGER_PHYSICS_LOCAL_CROP", True)
_STATIC_NOAH_ACTIVE_COLUMNS = _env_flag("WRF_PY_STATIC_NOAH_ACTIVE_COLUMNS", False)
_STATIC_NOAH_ACTIVE_ROWS = _env_flag("WRF_PY_STATIC_NOAH_ACTIVE_ROWS", True)
_PROFILE_SURFACE_PBL_PATH = os.environ.get("WRF_PY_PROFILE_SURFACE_PBL", "").strip()
_PROFILE_RADIATION_PATH = os.environ.get("WRF_PY_PROFILE_RADIATION", "").strip()
_PBL_DRIVER_IMPL = pbl_driver


def _profile_surface_pbl_ready(value: Any) -> None:
    if not _PROFILE_SURFACE_PBL_PATH:
        return
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _profile_surface_pbl_write(name: str, start: float, value: Any) -> None:
    if not _PROFILE_SURFACE_PBL_PATH:
        return
    _profile_surface_pbl_ready(value)
    with open(_PROFILE_SURFACE_PBL_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{perf_counter() - start:.6f} {name}\n")


def _profile_radiation_ready(value: Any) -> None:
    if not _PROFILE_RADIATION_PATH:
        return
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _profile_radiation_event(name: str) -> None:
    if not _PROFILE_RADIATION_PATH:
        return
    with open(_PROFILE_RADIATION_PATH, "a", encoding="ascii") as handle:
        handle.write(f"0.000000 {name}\n")


def _profile_radiation_write(name: str, start: float, value: Any) -> None:
    if not _PROFILE_RADIATION_PATH:
        return
    _profile_radiation_ready(value)
    with open(_PROFILE_RADIATION_PATH, "a", encoding="ascii") as handle:
        handle.write(f"{perf_counter() - start:.6f} {name}\n")

_RELAX_BDY_DRY_IMPL = relax_bdy_dry
_SPEC_BDY_DRY_IMPL = spec_bdy_dry


@partial(jax.jit, static_argnames=(*_STATIC_DOMAIN_ARGNAMES, "rdx", "rdy", "dt", "config_flags", "rk_step"), inline=_INLINE_DIRECT_STAGE_JITS)
def _ww_split_jit_tuple(
    *,
    ww_e: Any,
    ww_i: Any,
    u: Any,
    v: Any,
    ww: Any,
    mut: Any,
    rdnw: Any,
    msfty: Any,
    c1f: Any,
    c2f: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    rdx: float,
    rdy: float,
    dt: float,
    config_flags: Any,
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
) -> tuple[Any, Any]:
    split = ww_split(
        ww_e,
        ww_i,
        u,
        v,
        ww,
        mut,
        rdnw,
        msfty,
        c1f,
        c2f,
        float(rdx),
        float(rdy),
        msfux,
        msfuy,
        msfvx,
        msfvy,
        float(dt),
        config_flags,
        int(rk_step),
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
    return split.ww_e, split.ww_i


@partial(jax.jit, static_argnames=_STATIC_DOMAIN_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _zero_rk_tendency_pack_jit_tuple(
    *,
    ru_tend: Any,
    rv_tend: Any,
    rw_tend: Any,
    t_tend: Any,
    ph_tend: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    ph_save: Any,
    t_save: Any,
    mu_tend: Any,
    mu_save: Any,
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
) -> tuple[Any, ...]:
    zero3_args = (ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    zero2_args = (ids, ide, jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, its, ite, jts, jte, 1, 1)
    return (
        zero_tend(ru_tend, *zero3_args),
        zero_tend(rv_tend, *zero3_args),
        zero_tend(rw_tend, *zero3_args),
        zero_tend(t_tend, *zero3_args),
        zero_tend(ph_tend, *zero3_args),
        zero_tend(u_save, *zero3_args),
        zero_tend(v_save, *zero3_args),
        zero_tend(w_save, *zero3_args),
        zero_tend(ph_save, *zero3_args),
        zero_tend(t_save, *zero3_args),
        zero_tend2d(mu_tend, *zero2_args),
        zero_tend2d(mu_save, *zero2_args),
    )


def _rk_tendency_direct_impl(
    *,
    ru_tend: Any,
    rv_tend: Any,
    rw_tend: Any,
    t_tend: Any,
    ph_tend: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    ph_save: Any,
    t_save: Any,
    mu_tend: Any,
    mu_save: Any,
    ww_e: Any,
    ww_i: Any,
    u: Any,
    v: Any,
    w: Any,
    t: Any,
    ph: Any,
    ww: Any,
    mut: Any,
    rdnw: Any,
    rdn: Any,
    msfty: Any,
    c1f: Any,
    c2f: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    ru: Any,
    rv: Any,
    rw: Any,
    c1h: Any,
    c2h: Any,
    fnm: Any,
    fnp: Any,
    phb: Any,
    muu: Any,
    muv: Any,
    alt: Any,
    p: Any,
    pb: Any,
    al: Any,
    php: Any,
    cqu: Any,
    cqv: Any,
    cqw: Any,
    mu: Any,
    mub: Any,
    f: Any,
    e: Any,
    sina: Any,
    cosa: Any,
    clat: Any,
    rk_step: int,
    rk_order: int,
    time_step: int,
    config_flags: Any,
    non_hydrostatic: bool,
    top_lid: bool,
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
    rdx: float,
    rdy: float,
    dt: float,
    cfn: float,
    cfn1: float,
    cf1: float,
    cf2: float,
    cf3: float,
    rounding_token: Any | None = None,
) -> tuple[Any, ...]:
    ru_tend = zero_tend(ru_tend, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    rv_tend = zero_tend(rv_tend, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    rw_tend = zero_tend(rw_tend, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    t_tend = zero_tend(t_tend, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    ph_tend = zero_tend(ph_tend, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    u_save = zero_tend(u_save, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    v_save = zero_tend(v_save, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    w_save = zero_tend(w_save, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    ph_save = zero_tend(ph_save, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    t_save = zero_tend(t_save, 1, ide, 1, jde, 1, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    mu_tend = zero_tend2d(mu_tend, 1, ide, 1, jde, 1, 1, ims, ime, jms, jme, 1, 1, its, ite, jts, jte, 1, 1)
    mu_save = zero_tend2d(mu_save, 1, ide, 1, jde, 1, 1, ims, ime, jms, jme, 1, 1, its, ite, jts, jte, 1, 1)

    split = ww_split(
        ww_e,
        ww_i,
        u,
        v,
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
        1,
        ide,
        1,
        jde,
        1,
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
    ww_e = split.ww_e
    ww_i = split.ww_i

    ru_tend = advect_u_vectorized(
        u,
        u,
        ru_tend,
        ru,
        rv,
        ww_e,
        c1h,
        c2h,
        mut,
        time_step,
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
        rounding_token,
    )
    rv_tend = advect_v_vectorized(
        v,
        v,
        rv_tend,
        ru,
        rv,
        ww_e,
        c1h,
        c2h,
        mut,
        time_step,
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
        rounding_token,
    )
    if non_hydrostatic:
        rw_tend = advect_w_vectorized(
            w,
            w,
            rw_tend,
            ru,
            rv,
            ww_e,
            c1h,
            c2h,
            mut,
            time_step,
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
            rdn,
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
    t_tend = advect_scalar_vectorized(
        t,
        t,
        t_tend,
        ru,
        rv,
        ww_e,
        c1h,
        c2h,
        mut,
        time_step,
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
        rounding_token,
    )
    ph_tend = rhs_ph_vectorized(
        ph_tend,
        u,
        v,
        ww_e,
        ph,
        ph,
        phb,
        w,
        mut,
        muu,
        muv,
        c1f,
        c2f,
        fnm,
        fnp,
        rdnw,
        cfn,
        cfn1,
        rdx,
        rdy,
        msfux,
        msfuy,
        msfvx,
        msfvx_inv,
        msfvy,
        msftx,
        msfty,
        non_hydrostatic,
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
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        rounding_token,
    )
    ru_tend, rv_tend = horizontal_pressure_gradient_vectorized(
        ru_tend,
        rv_tend,
        ph,
        alt,
        p,
        pb,
        al,
        php,
        cqu,
        cqv,
        muu,
        muv,
        mu,
        c1h,
        c2h,
        fnm,
        fnp,
        rdnw,
        cf1,
        cf2,
        cf3,
        cfn,
        cfn1,
        rdx,
        rdy,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        msftx,
        msfty,
        config_flags,
        non_hydrostatic,
        top_lid,
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
    if non_hydrostatic:
        rw_tend, cqw = pg_buoy_w(
            rw_tend,
            p,
            cqw,
            mu,
            mub,
            c1f,
            c2f,
            rdnw,
            rdn,
            9.81,
            msftx,
            msfty,
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
    if int(getattr(config_flags, "w_damping", 0)) == 1 or bool(getattr(config_flags, "use_adaptive_time_step", False)):
        rw_tend, max_vert_cfl, max_horiz_cfl = w_damp(
            rw_tend,
            u,
            v,
            ww,
            w,
            mut,
            c1f,
            c2f,
            rdnw,
            rdx,
            rdy,
            msfux,
            msfuy,
            msfvx,
            msfvy,
            dt,
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
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )
    else:
        max_vert_cfl = jnp.asarray(0.0, dtype=jnp.asarray(rw_tend).dtype)
        max_horiz_cfl = jnp.asarray(0.0, dtype=jnp.asarray(rw_tend).dtype)
    ru_tend, rv_tend, rw_tend = coriolis_vectorized(
        ru,
        rv,
        rw,
        ru_tend,
        rv_tend,
        rw_tend,
        config_flags,
        msftx,
        msfty,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        f,
        e,
        sina,
        cosa,
        fnm,
        fnp,
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
    ru_tend, rv_tend, rw_tend = curvature_vectorized(
        ru,
        rv,
        rw,
        u,
        v,
        w,
        ru_tend,
        rv_tend,
        rw_tend,
        config_flags,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        msftx,
        msfty,
        clat,
        fnm,
        fnp,
        rdx,
        rdy,
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
        1.0 / 6370.0e03,
        np.pi / 180.0,
        rounding_token,
    )
    return (
        ru_tend,
        rv_tend,
        rw_tend,
        t_tend,
        ph_tend,
        u_save,
        v_save,
        w_save,
        ph_save,
        t_save,
        mu_tend,
        mu_save,
        ww_e,
        ww_i,
        cqw,
        max_vert_cfl,
        max_horiz_cfl,
    )


_rk_tendency_direct_jit_tuple = jax.jit(
    _rk_tendency_direct_impl,
    static_argnames=_RK_TENDENCY_DIRECT_STATIC_ARGNAMES,
    inline=_INLINE_RK_TENDENCY_DIRECT,
)
_rk_tendency_direct_no_ieva_jit_tuple = jax.jit(
    _rk_tendency_direct_impl,
    static_argnames=_RK_TENDENCY_DIRECT_NO_IEVA_STATIC_ARGNAMES,
    inline=_INLINE_RK_TENDENCY_DIRECT,
)


@partial(jax.jit, static_argnums=_RK_ADDTEND_DRY_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _rk_addtend_dry_jit_tuple(*args: Any) -> tuple[Any, ...]:
    return rk_addtend_dry(*args)


@partial(jax.jit, static_argnums=_RELAX_BDY_DRY_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _relax_bdy_dry_jit_tuple(*args: Any) -> tuple[Any, ...]:
    result = _RELAX_BDY_DRY_IMPL(*args)
    return (
        result.ru_tendf,
        result.rv_tendf,
        result.ph_tendf,
        result.t_tendf,
        result.rw_tendf,
        result.mu_tend,
    )


@partial(jax.jit, static_argnums=_SPEC_BDY_DRY_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _spec_bdy_dry_jit_tuple(*args: Any) -> tuple[Any, ...]:
    result = _SPEC_BDY_DRY_IMPL(*args)
    return (
        result.ru_tend,
        result.rv_tend,
        result.ph_tend,
        result.t_tend,
        result.rw_tend,
        result.mu_tend,
    )


def _overlay_spec_bdy_dry_direct(
    ru_tend_prev: Any,
    rv_tend_prev: Any,
    ph_tend_prev: Any,
    t_tend_prev: Any,
    rw_tend_prev: Any,
    mu_tend_prev: Any,
    *args: Any,
) -> tuple[Any, ...]:
    config_flags = args[0]
    spec_bdy_width = args[49]
    spec_zone = args[50]
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte = args[51:75]

    def overlay3(previous: Any, tendency_offset: int, variable: str) -> Any:
        return overlay_spec_bdytend(
            previous,
            args[tendency_offset],
            args[tendency_offset + 1],
            args[tendency_offset + 2],
            args[tendency_offset + 3],
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

    mu_tend = overlay_spec_bdytend(
        mu_tend_prev,
        args[45],
        args[46],
        args[47],
        args[48],
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
    nested = bool(config_flags.get("nested", False) if isinstance(config_flags, Mapping) else getattr(config_flags, "nested", False))
    rw_tend = overlay3(rw_tend_prev, 41, "h") if nested else rw_tend_prev
    return (
        overlay3(ru_tend_prev, 25, "u"),
        overlay3(rv_tend_prev, 29, "v"),
        overlay3(ph_tend_prev, 33, "h"),
        overlay3(t_tend_prev, 37, "t"),
        rw_tend,
        mu_tend,
    )


@partial(jax.jit, static_argnums=_CALC_MU_UV_1_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _calc_mu_uv_1_jit_tuple(*args: Any) -> tuple[Any, Any]:
    return calc_mu_uv_1(*args)


@partial(jax.jit, static_argnums=_SPEC_BDY_DRY_OVERLAY_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _spec_bdy_dry_overlay_jit_tuple(
    ru_tend_prev: Any,
    rv_tend_prev: Any,
    ph_tend_prev: Any,
    t_tend_prev: Any,
    rw_tend_prev: Any,
    mu_tend_prev: Any,
    *args: Any,
) -> tuple[Any, ...]:
    return _overlay_spec_bdy_dry_direct(
        ru_tend_prev,
        rv_tend_prev,
        ph_tend_prev,
        t_tend_prev,
        rw_tend_prev,
        mu_tend_prev,
        *args,
    )


@partial(jax.jit, static_argnums=_SPEC_BDY_DRY_OVERLAY_MASKS_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _spec_bdy_dry_overlay_masks_jit_tuple(
    ru_tend_prev: Any,
    rv_tend_prev: Any,
    ph_tend_prev: Any,
    t_tend_prev: Any,
    rw_tend_prev: Any,
    mu_tend_prev: Any,
    ru_mask: Any,
    rv_mask: Any,
    ph_mask: Any,
    t_mask: Any,
    rw_mask: Any,
    mu_mask: Any,
    *args: Any,
) -> tuple[Any, ...]:
    del ru_mask, rv_mask, ph_mask, t_mask, rw_mask, mu_mask
    return _overlay_spec_bdy_dry_direct(
        ru_tend_prev,
        rv_tend_prev,
        ph_tend_prev,
        t_tend_prev,
        rw_tend_prev,
        mu_tend_prev,
        *args,
    )


@partial(jax.jit, static_argnums=_FUSED_DRY_RK1_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _dry_boundary_tendency_rk1_jit_tuple(*args: Any) -> tuple[Any, ...]:
    masks = args[:_DRY_BOUNDARY_MASK_ARG_COUNT]
    start = _DRY_BOUNDARY_MASK_ARG_COUNT
    relax_args = args[start : start + _RELAX_BDY_DRY_ARG_COUNT]
    start += _RELAX_BDY_DRY_ARG_COUNT
    rk_args = list(args[start : start + _RK_ADDTEND_DRY_ARG_COUNT])
    start += _RK_ADDTEND_DRY_ARG_COUNT
    spec_args = args[start : start + _SPEC_BDY_DRY_ARG_COUNT]

    relaxed = _RELAX_BDY_DRY_IMPL(*relax_args)
    u_save = relaxed.ru_tendf
    v_save = relaxed.rv_tendf
    ph_save = relaxed.ph_tendf
    t_save = relaxed.t_tendf
    w_save = relaxed.rw_tendf
    relaxed_mu_tend = relaxed.mu_tend

    rk_args[10] = u_save
    rk_args[11] = v_save
    rk_args[12] = w_save
    rk_args[13] = ph_save
    rk_args[14] = t_save
    rk_args[15] = relaxed_mu_tend
    (
        ru_tend,
        rv_tend,
        rw_tend,
        ph_tend,
        t_tend,
        ru_tendf,
        rv_tendf,
        rw_tendf,
        ph_tendf,
        t_tendf,
        mu_tend,
        mu_tendf,
    ) = rk_addtend_dry(*rk_args)

    del masks
    ru_tend, rv_tend, ph_tend, t_tend, rw_tend, mu_tend = _overlay_spec_bdy_dry_direct(
        ru_tend,
        rv_tend,
        ph_tend,
        t_tend,
        rw_tend,
        mu_tend,
        *spec_args,
    )

    return (
        u_save,
        v_save,
        ph_save,
        t_save,
        w_save,
        ru_tend,
        rv_tend,
        rw_tend,
        ph_tend,
        t_tend,
        ru_tendf,
        rv_tendf,
        rw_tendf,
        ph_tendf,
        t_tendf,
        mu_tend,
        mu_tendf,
    )


@partial(jax.jit, static_argnums=_FUSED_DRY_NONRK1_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _dry_boundary_tendency_nonrk1_jit_tuple(*args: Any) -> tuple[Any, ...]:
    masks = args[:_DRY_BOUNDARY_MASK_ARG_COUNT]
    start = _DRY_BOUNDARY_MASK_ARG_COUNT
    rk_args = args[start : start + _RK_ADDTEND_DRY_ARG_COUNT]
    start += _RK_ADDTEND_DRY_ARG_COUNT
    spec_args = args[start : start + _SPEC_BDY_DRY_ARG_COUNT]

    (
        ru_tend,
        rv_tend,
        rw_tend,
        ph_tend,
        t_tend,
        ru_tendf,
        rv_tendf,
        rw_tendf,
        ph_tendf,
        t_tendf,
        mu_tend,
        mu_tendf,
    ) = rk_addtend_dry(*rk_args)

    del masks
    ru_tend, rv_tend, ph_tend, t_tend, rw_tend, mu_tend = _overlay_spec_bdy_dry_direct(
        ru_tend,
        rv_tend,
        ph_tend,
        t_tend,
        rw_tend,
        mu_tend,
        *spec_args,
    )
    return (
        ru_tend,
        rv_tend,
        rw_tend,
        ph_tend,
        t_tend,
        ru_tendf,
        rv_tendf,
        rw_tendf,
        ph_tendf,
        t_tendf,
        mu_tend,
        mu_tendf,
    )


@partial(jax.jit, static_argnums=_RK_UPDATE_SCALAR_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _rk_update_scalar_jit_tuple(*args: Any) -> tuple[Any, Any, Any | None, Any | None]:
    return rk_update_scalar(*args)


@partial(jax.jit, static_argnums=_ADVECT_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _advect_u_jit(*args: Any) -> Any:
    return advect_u_vectorized(*args)


@partial(jax.jit, static_argnums=_ADVECT_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _advect_v_jit(*args: Any) -> Any:
    return advect_v_vectorized(*args)


@partial(jax.jit, static_argnums=_ADVECT_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _advect_w_jit(*args: Any) -> Any:
    return advect_w_vectorized(*args)


@partial(jax.jit, static_argnums=_ADVECT_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _advect_scalar_jit(*args: Any) -> Any:
    return advect_scalar_vectorized(*args)


@partial(jax.jit, static_argnums=_ADVECT_PD_STATIC_ARGNUMS, inline=False)
def _advect_scalar_pd_components_jit(*args: Any) -> tuple[Any, ...]:
    return advect_scalar_pd_vectorized(
        *args,
        return_directional_components=True,
    )


@partial(jax.jit, static_argnums=_ADVECT_PD_STATIC_ARGNUMS, inline=False)
def _advect_scalar_pd_prelimit_jit(*args: Any) -> tuple[Any, ...]:
    return advect_scalar_pd_vectorized(
        *args,
        return_prelimit_components=True,
    )


@partial(
    jax.jit,
    static_argnums=tuple(index + 1 for index in _ADVECT_PD_STATIC_ARGNUMS),
    inline=False,
)
def _advect_scalar_pd_from_prelimit_jit(
    prelimit_fluxes: tuple[Any, ...],
    *args: Any,
) -> tuple[Any, ...]:
    precomputed_fluxes = subtract_scalar_pd_prelimit_fluxes(
        *prelimit_fluxes,
        args[46],
    )
    return advect_scalar_pd_vectorized(
        *args,
        precomputed_fluxes=precomputed_fluxes,
    )


def _advect_scalar_pd_jit(*args: Any) -> tuple[Any, Any, Any]:
    prelimit_fluxes = _advect_scalar_pd_prelimit_jit(*args)
    tendency, h_tendency, z_tendency = _advect_scalar_pd_from_prelimit_jit(
        prelimit_fluxes,
        *args,
    )
    return tendency, h_tendency, z_tendency


def _rk_update_qv_single_array(
    *,
    qv: Any,
    moist_old_qv: Any,
    moist_tend_qv: Any,
    advect_tend: Any,
    msfty: Any,
    c1: Any,
    c2: Any,
    mu_old: Any,
    mu_new: Any,
    mu_base: Any,
    rk_step: int,
    dt: float,
    spec_zone: int,
    specified_or_nested: bool,
    periodic_x: bool,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> tuple[Any, Any]:
    """Specialized WRF ``rk_update_scalar`` for the configured QV-only path."""

    del kds
    qv_arr = jnp.asarray(qv)
    old_arr = jnp.asarray(moist_old_qv)
    tend_arr = jnp.asarray(moist_tend_qv)
    adv_arr = jnp.asarray(advect_tend)
    msfty_arr = jnp.asarray(msfty)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mu_old_arr = jnp.asarray(mu_old)
    mu_new_arr = jnp.asarray(mu_new)
    mu_base_arr = jnp.asarray(mu_base)

    def store(value: Any) -> Any:
        array = jnp.asarray(value, dtype=qv_arr.dtype)
        if rounding_token is None:
            return array
        return jax.lax.optimization_barrier(_materialize_wrf_real(array, rounding_token))

    i_fortran = int(ims) + jnp.arange(int(qv_arr.shape[0]))
    k_fortran = int(kms) + jnp.arange(int(qv_arr.shape[1]))
    j_fortran = int(jms) + jnp.arange(int(qv_arr.shape[2]))

    full_i_start = int(its)
    full_i_end = min(int(ite), int(ide) - 1)
    full_j_start = int(jts)
    full_j_end = min(int(jte), int(jde) - 1)
    full_k_start = int(kts)
    full_k_end = min(int(kte), int(kde) - 1)

    adv_i_start = full_i_start
    adv_i_end = full_i_end
    adv_j_start = full_j_start
    adv_j_end = full_j_end
    adv_k_start = full_k_start
    adv_k_end = full_k_end
    if bool(specified_or_nested):
        if not bool(periodic_x):
            adv_i_start = max(int(its), int(ids) + int(spec_zone))
            adv_i_end = min(int(ite), int(ide) - int(spec_zone) - 1)
        adv_j_start = max(int(jts), int(jds) + int(spec_zone))
        adv_j_end = min(int(jte), int(jde) - int(spec_zone) - 1)
        adv_k_start = int(kts)
        adv_k_end = min(int(kte), int(kde) - 1)

    full_i = (i_fortran >= full_i_start) & (i_fortran <= full_i_end)
    full_k = (k_fortran >= full_k_start) & (k_fortran <= full_k_end)
    full_j = (j_fortran >= full_j_start) & (j_fortran <= full_j_end)
    adv_i = (i_fortran >= adv_i_start) & (i_fortran <= adv_i_end)
    adv_k = (k_fortran >= adv_k_start) & (k_fortran <= adv_k_end)
    adv_j = (j_fortran >= adv_j_start) & (j_fortran <= adv_j_end)
    full_mask = full_i[:, None, None] & full_k[None, :, None] & full_j[None, None, :]
    adv_mask = adv_i[:, None, None] & adv_k[None, :, None] & adv_j[None, None, :]

    kk = jnp.clip(k_fortran - int(kms), 0, int(c1_arr.shape[0]) - 1)
    old_tile = old_arr[:, : int(qv_arr.shape[1]), :]
    old_scalar = qv_arr if int(rk_step) == 1 else old_tile
    if rounding_token is None:
        old_mass = c1_arr[kk][None, :, None] * (mu_old_arr + mu_base_arr)[:, None, :] + c2_arr[kk][None, :, None]
        new_mass = c1_arr[kk][None, :, None] * (mu_new_arr + mu_base_arr)[:, None, :] + c2_arr[kk][None, :, None]
        tendency = jnp.where(full_mask, tend_arr, jnp.asarray(0.0, dtype=tend_arr.dtype))
        tendency = tendency + jnp.where(
            adv_mask,
            adv_arr * msfty_arr[:, None, :],
            jnp.asarray(0.0, dtype=adv_arr.dtype),
        )
        updated = (old_mass * old_scalar + float(dt) * tendency) / new_mass
    else:
        muold = store(mu_old_arr + mu_base_arr)
        munew = store(mu_new_arr + mu_base_arr)
        old_mass_product = store(c1_arr[kk][None, :, None] * muold[:, None, :])
        old_mass = store(old_mass_product + c2_arr[kk][None, :, None])
        new_mass_product = store(c1_arr[kk][None, :, None] * munew[:, None, :])
        new_mass = store(new_mass_product + c2_arr[kk][None, :, None])

        zero = jnp.asarray(0.0, dtype=qv_arr.dtype)
        adv_tendency = store(adv_arr * msfty_arr[:, None, :])
        tendency = jnp.where(adv_mask, adv_tendency, zero)
        tendency_with_scalar = store(tendency + tend_arr)
        tendency = jnp.where(full_mask, tendency_with_scalar, tendency)

        old_coupled = store(old_mass * old_scalar)
        tendency_dt = store(jnp.asarray(dt, dtype=qv_arr.dtype) * tendency)
        numerator = store(old_coupled + tendency_dt)
        updated = store(numerator / new_mass)
    qv_out = jnp.where(full_mask, updated, qv_arr)
    if int(rk_step) == 1:
        old_tile_out = jnp.where(full_mask, qv_arr, old_tile)
        old_out = old_arr.at[:, : int(qv_arr.shape[1]), :].set(old_tile_out)
    else:
        old_out = old_arr
    return old_out, qv_out


def _rk_update_qv_pd_single_array(
    *,
    scalar: Any,
    sc_tend: Any,
    c1: Any,
    c2: Any,
    mu_old: Any,
    mu_new: Any,
    mu_base: Any,
    dt: float,
    ide: int,
    jde: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[Any, Any]:
    """Specialized QV-only ``rk_update_scalar_pd`` without 4D species scatter."""

    scalar_arr = jnp.asarray(scalar)
    tend_arr = jnp.asarray(sc_tend)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mu_old_arr = jnp.asarray(mu_old)
    mu_new_arr = jnp.asarray(mu_new)
    mu_base_arr = jnp.asarray(mu_base)
    active_k_count = min(int(scalar_arr.shape[1]), int(tend_arr.shape[1]), int(c1_arr.shape[0]))
    scalar_tile = scalar_arr[:, :active_k_count, :]
    tend_tile = tend_arr[:, :active_k_count, :]

    i_fortran = int(ims) + jnp.arange(int(scalar_arr.shape[0]))
    k_fortran = int(kms) + jnp.arange(active_k_count)
    j_fortran = int(jms) + jnp.arange(int(scalar_arr.shape[2]))
    i_valid = (i_fortran >= int(its)) & (i_fortran <= min(int(ite), int(ide) - 1))
    k_valid = (k_fortran >= int(kts)) & (k_fortran <= min(int(kte), int(kde) - 1))
    j_valid = (j_fortran >= int(jts)) & (j_fortran <= min(int(jte), int(jde) - 1))
    valid = i_valid[:, None, None] & k_valid[None, :, None] & j_valid[None, None, :]

    kk = jnp.clip(k_fortran - int(kms), 0, int(c1_arr.shape[0]) - 1)
    old_mass = c1_arr[kk][None, :, None] * (mu_old_arr + mu_base_arr)[:, None, :] + c2_arr[kk][None, :, None]
    new_mass = c1_arr[kk][None, :, None] * (mu_new_arr + mu_base_arr)[:, None, :] + c2_arr[kk][None, :, None]
    updated = (old_mass * scalar_tile + float(dt) * tend_tile) / new_mass
    zero = jnp.asarray(0.0, dtype=tend_arr.dtype)
    scalar_front = jnp.where(valid, updated, scalar_tile)
    tend_front = jnp.where(valid, zero, tend_tile)
    if active_k_count < int(scalar_arr.shape[1]):
        scalar_out = jnp.concatenate([scalar_front, scalar_arr[:, active_k_count:, :]], axis=1)
    else:
        scalar_out = scalar_front
    if active_k_count < int(tend_arr.shape[1]):
        tend_out = jnp.concatenate([tend_front, tend_arr[:, active_k_count:, :]], axis=1)
    else:
        tend_out = tend_front
    return scalar_out, tend_out


def _replace_last_axis_component(array: Any, index: int, value: Any) -> Any:
    arr = jnp.asarray(array)
    component = jnp.expand_dims(jnp.asarray(value, dtype=arr.dtype), axis=-1)
    return jnp.concatenate([arr[..., : int(index)], component, arr[..., int(index) + 1 :]], axis=-1)


@partial(jax.jit, static_argnames=_MOIST_SCALAR_ADVANCE_STATIC_ARGNAMES, inline=_INLINE_DIRECT_STAGE_JITS)
def _moist_scalar_advance_jit_tuple(
    *,
    qv: Any,
    moist: Any,
    moist_old: Any,
    moist_tend: Any,
    rqvften: Any,
    ww_e: Any,
    ww_i: Any,
    u: Any,
    v: Any,
    ww: Any,
    muts: Any,
    rdnw: Any,
    msfty: Any,
    c1f: Any,
    c2f: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    ru_m: Any,
    rv_m: Any,
    c1h: Any,
    c2h: Any,
    mub: Any,
    mu_1: Any,
    mut: Any,
    mu_2: Any,
    msftx: Any,
    fnm: Any,
    fnp: Any,
    qvapor_bxs: Any,
    qvapor_bxe: Any,
    qvapor_bys: Any,
    qvapor_bye: Any,
    qvapor_btxs: Any,
    qvapor_btxe: Any,
    qvapor_btys: Any,
    qvapor_btye: Any,
    fcx: Any,
    gcx: Any,
    rounding_token: Any,
    config_flags: Any,
    rk_step: int,
    rk_order: int,
    adv_opt: int,
    time_step: int,
    do_final_pd: bool,
    do_cu_rqvften: bool,
    do_moist_boundary: bool,
    rdx: float,
    rdy: float,
    dt_split: float,
    dt_rk: float,
    dtbc: float,
    spec_bdy_width: int,
    spec_zone: int,
    relax_zone: int,
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
) -> tuple[Any, ...]:
    qv_arr = jnp.asarray(qv)
    moist_arr = jnp.asarray(moist)
    moist_old_arr = jnp.asarray(moist_old)
    moist_tend_arr = jnp.asarray(moist_tend)
    moist_tend_qv = moist_tend_arr[..., 1]
    zero3 = jnp.zeros_like(qv_arr)

    split = ww_split(
        ww_e,
        ww_i,
        u,
        v,
        ww,
        muts,
        rdnw,
        msfty,
        c1f,
        c2f,
        float(rdx),
        float(rdy),
        msfux,
        msfuy,
        msfvx,
        msfvy,
        float(dt_split),
        config_flags,
        int(rk_step),
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

    if bool(do_final_pd):
        advect_tend, h_tendency, z_tendency = advect_scalar_pd_vectorized(
            qv_arr,
            moist_old_arr,
            zero3,
            zero3,
            zero3,
            ru_m,
            rv_m,
            ww_e_out,
            c1h,
            c2h,
            muts,
            mub,
            mu_1,
            int(time_step),
            config_flags,
            False,
            msfux,
            msfuy,
            msfvx,
            msfvy,
            msftx,
            msfty,
            fnm,
            fnp,
            float(rdx),
            float(rdy),
            rdnw,
            float(dt_rk),
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
            kps,
            kpe,
            rounding_token,
        )
    else:
        advect_tend = advect_scalar_vectorized(
            qv_arr,
            qv_arr,
            zero3,
            ru_m,
            rv_m,
            ww_e_out,
            c1h,
            c2h,
            muts,
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
            float(rdx),
            float(rdy),
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
            kps,
            kpe,
        )
        h_tendency = zero3
        z_tendency = zero3

    rqvften_out = advect_tend * jnp.asarray(msfty)[:, None, :] if bool(do_cu_rqvften) else jnp.asarray(rqvften)

    if bool(do_moist_boundary):
        relaxed = relax_bdy_scalar(
            moist_tend_qv,
            qv_arr,
            mut,
            c1h,
            c2h,
            qvapor_bxs,
            qvapor_bxe,
            qvapor_bys,
            qvapor_bye,
            qvapor_btxs,
            qvapor_btxe,
            qvapor_btys,
            qvapor_btye,
            int(spec_bdy_width),
            int(spec_zone),
            int(relax_zone),
            dtbc,
            fcx,
            gcx,
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
            kps,
            kpe,
        )
        specified = spec_bdy_scalar(
            qvapor_bxs,
            qvapor_bxe,
            qvapor_bys,
            qvapor_bye,
            qvapor_btxs,
            qvapor_btxe,
            qvapor_btys,
            qvapor_btye,
            int(spec_bdy_width),
            int(spec_zone),
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
            kps,
            kpe,
        )
        mask_values = spec_bdytend(
            qvapor_bxs,
            qvapor_bxe,
            qvapor_bys,
            qvapor_bye,
            jnp.ones_like(jnp.asarray(qvapor_btxs)),
            jnp.ones_like(jnp.asarray(qvapor_btxe)),
            jnp.ones_like(jnp.asarray(qvapor_btys)),
            jnp.ones_like(jnp.asarray(qvapor_btye)),
            "q",
            config_flags,
            int(spec_bdy_width),
            int(spec_zone),
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
            kps,
            kpe,
        )
        moist_tend_qv = _overlay_specified_boundary(relaxed, specified, jnp.asarray(mask_values) != 0)

    moist_old_out_qv, qv_out = _rk_update_qv_single_array(
        qv=qv_arr,
        moist_old_qv=moist_old_arr,
        moist_tend_qv=moist_tend_qv,
        advect_tend=advect_tend,
        msfty=msfty,
        c1=c1h,
        c2=c2h,
        mu_old=mu_1,
        mu_new=mu_2,
        mu_base=mub,
        rk_step=int(rk_step),
        dt=float(dt_rk),
        spec_zone=int(spec_zone),
        specified_or_nested=bool(getattr(config_flags, "nested", False)) or bool(getattr(config_flags, "specified", False)),
        periodic_x=bool(getattr(config_flags, "periodic_x", False)),
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        jms=jms,
        kms=kms,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kps,
        kte=kpe,
        rounding_token=rounding_token,
    )
    moist_out = _replace_last_axis_component(moist_arr, 1, qv_out)
    moist_tend_out = _replace_last_axis_component(moist_tend_arr, 1, moist_tend_qv)
    return (
        moist_old_out_qv,
        moist_out,
        qv_out,
        moist_tend_out,
        advect_tend,
        h_tendency,
        z_tendency,
        rqvften_out,
        ww_e_out,
        ww_i_out,
    )


@partial(jax.jit, static_argnames=_BATCHED_NONQV_MOIST_STATIC_ARGNAMES, inline=False)
def _batched_nonqv_moist_advance_jit_tuple(
    *,
    moist: Any,
    moist_old: Any,
    moist_tend: Any,
    ww_e: Any,
    ru_m: Any,
    rv_m: Any,
    c1h: Any,
    c2h: Any,
    muts: Any,
    mub: Any,
    mu_1: Any,
    mu_2: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fnm: Any,
    fnp: Any,
    rdnw: Any,
    rounding_token: Any,
    config_flags: Any,
    rk_step: int,
    do_final_pd: bool,
    do_flow_dep_boundary: bool,
    time_step: int,
    rdx: float,
    rdy: float,
    dt_rk: float,
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
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    kps: int,
    kpe: int,
) -> tuple[Any, Any, Any]:
    """Advance non-QV moisture species as one device execution unit."""

    moist_arr = jnp.asarray(moist)
    moist_old_arr = jnp.asarray(moist_old)
    moist_tend_arr = jnp.asarray(moist_tend)
    if int(moist_arr.shape[-1]) <= 2:
        return moist_old_arr, moist_arr, moist_tend_arr

    species = jnp.moveaxis(moist_arr[..., 2:], -1, 0)
    species_old = jnp.moveaxis(moist_old_arr[..., 2:], -1, 0)
    species_tend = jnp.moveaxis(moist_tend_arr[..., 2:], -1, 0)

    if bool(do_final_pd):

        def advect_one(field: Any, field_old: Any) -> tuple[Any, Any, Any]:
            zero = jnp.zeros_like(field)
            return _advect_scalar_pd_jit(
                field,
                field_old,
                zero,
                zero,
                zero,
                ru_m,
                rv_m,
                ww_e,
                c1h,
                c2h,
                muts,
                mub,
                mu_1,
                int(time_step),
                config_flags,
                False,
                msfux,
                msfuy,
                msfvx,
                msfvy,
                msftx,
                msfty,
                fnm,
                fnp,
                float(rdx),
                float(rdy),
                rdnw,
                float(dt_rk),
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

        advect_tend, h_tendency, z_tendency = jax.vmap(advect_one)(species, species_old)
    else:

        def advect_one(field: Any) -> Any:
            return _advect_scalar_jit(
                field,
                field,
                jnp.zeros_like(field),
                ru_m,
                rv_m,
                ww_e,
                c1h,
                c2h,
                muts,
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
                float(rdx),
                float(rdy),
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

        advect_tend = jax.vmap(advect_one)(species)
        h_tendency = jnp.zeros_like(advect_tend)
        z_tendency = jnp.zeros_like(advect_tend)

    def update_one(
        field_old: Any,
        field: Any,
        tendency: Any,
        advect: Any,
        horizontal: Any,
        vertical: Any,
    ) -> tuple[Any, Any]:
        scalar_1, scalar_2, _advh_t, _advz_t = _rk_update_scalar_jit_tuple(
            1,
            1,
            field_old[..., None],
            field[..., None],
            tendency[..., None],
            None,
            None,
            advect,
            horizontal,
            vertical,
            msftx,
            msfty,
            c1h,
            c2h,
            mu_1,
            mu_2,
            mub,
            int(rk_step),
            float(dt_rk),
            int(spec_zone),
            config_flags,
            False,
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
        return jnp.asarray(scalar_1)[..., 0], jnp.asarray(scalar_2)[..., 0]

    species_old_out, species_out = jax.vmap(update_one)(
        species_old,
        species,
        species_tend,
        advect_tend,
        h_tendency,
        z_tendency,
    )

    if bool(do_flow_dep_boundary):
        species_out = jax.lax.optimization_barrier(species_out)

        def apply_flow_boundary(field: Any) -> Any:
            return flow_dep_bdy(
                field,
                ru_m,
                rv_m,
                config_flags,
                int(spec_zone),
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

        species_out = jax.vmap(apply_flow_boundary)(species_out)

    moist_old_out = jnp.concatenate(
        [moist_old_arr[..., :2], jnp.moveaxis(species_old_out, 0, -1)],
        axis=-1,
    )
    moist_out = jnp.concatenate(
        [moist_arr[..., :2], jnp.moveaxis(species_out, 0, -1)],
        axis=-1,
    )
    return moist_old_out, moist_out, moist_tend_arr


@partial(jax.jit, static_argnums=_RHS_PH_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _rhs_ph_jit(*args: Any) -> Any:
    return rhs_ph_vectorized(*args)


@partial(jax.jit, static_argnums=_HPG_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _horizontal_pressure_gradient_jit(*args: Any) -> tuple[Any, Any]:
    return horizontal_pressure_gradient_vectorized(*args)


@partial(jax.jit, static_argnums=_CORIOLIS_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _coriolis_jit(*args: Any) -> tuple[Any, Any, Any]:
    return coriolis_vectorized(*args)


@partial(jax.jit, static_argnums=_CURVATURE_STATIC_ARGNUMS, inline=_INLINE_DIRECT_STAGE_JITS)
def _curvature_jit(*args: Any) -> tuple[Any, Any, Any]:
    return curvature_vectorized(*args)


@partial(jax.jit, static_argnums=_HDIFF2_STATIC_ARGNUMS, inline=True)
def _horizontal_diffusion_2_jit_tuple(*args: Any) -> tuple[Any, ...]:
    result = horizontal_diffusion_2_vectorized(*args)
    return (
        result.rt_tendf,
        result.ru_tendf,
        result.rv_tendf,
        result.rw_tendf,
        result.tke_tendf,
        result.moist_tendf,
        result.chem_tendf,
        result.scalar_tendf,
        result.tracer_tendf,
        result.nba_mij,
    )


@partial(jax.jit, static_argnums=_HDIFF2_STATIC_ARGNUMS, inline=True)
def _horizontal_diffusion_2_jit_result(*args: Any) -> HorizontalDiffusion2Result:
    return horizontal_diffusion_2_without_theta_vectorized(*args)


@partial(jax.jit, static_argnums=_HDIFF_S_STATIC_ARGNUMS, inline=False)
def _horizontal_diffusion_s_jit(*args: Any) -> Any:
    return horizontal_diffusion_s_vectorized(*args)


@partial(jax.jit, static_argnums=_CAL_DEFORM_STATIC_ARGNUMS, inline=True)
def _cal_deform_and_div_jit_result(*args: Any) -> DeformAndDivResult:
    return cal_deform_and_div_vectorized(*args)


@partial(jax.jit, static_argnums=_CALCULATE_KM_KH_STATIC_ARGNUMS, inline=True)
def _calculate_km_kh_jit_result(*args: Any) -> CalculateKmKhResult:
    return calculate_km_kh_vectorized(*args)


@partial(jax.jit, static_argnums=_COMPUTE_DIFF_METRICS_STATIC_ARGNUMS, inline=True)
def _compute_diff_metrics_jit_result(*args: Any) -> DiffMetricsResult:
    return compute_diff_metrics(*args)


@partial(jax.jit, static_argnames=("config_flags", *_PHYSICAL_BC_DOMAIN_ARGNAMES), inline=True)
def _first_rk_part2_diffusion_bc_jit_tuple(
    rdzw: Any,
    rdz: Any,
    z: Any,
    zx: Any,
    zy: Any,
    ustm: Any,
    ust: Any,
    *,
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
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    return (
        set_physical_bc3d(rdzw, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte),
        set_physical_bc3d(rdz, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte),
        set_physical_bc3d(z, "w", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte),
        set_physical_bc3d(zx, "e", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte),
        set_physical_bc3d(zy, "f", config_flags, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte),
        set_physical_bc2d(ustm, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte),
        set_physical_bc2d(ust, "t", config_flags, ids, ide, jds, jde, ims, ime, jms, jme, ips, ipe, jps, jpe, its, ite, jts, jte),
    )


COMMUNICATION_IDENTITY_STAGES = frozenset(
    {
        "halo_em_moist_old_e_5_sub",
        "halo_em_moist_old_e_7_sub",
        "period_bdy_em_moist_old_sub",
        "halo_em_a_sub",
        "period_bdy_em_a_sub",
        "halo_em_scalar_e_5_sub",
        "halo_em_b_sub",
        "period_bdy_em_b_sub",
        "period_bdy_em_b3_sub",
        "halo_em_d2_3_sub",
        "halo_em_d2_5_sub",
        "period_bdy_em_d_sub",
        "period_bdy_em_moist2_sub",
        "halo_em_c_sub",
        "halo_em_c2_sub",
        "period_bdy_em_e_sub",
        "halo_em_e_3_sub",
        "halo_em_e_5_sub",
        "halo_em_moist_e_3_sub",
        "halo_em_moist_e_5_sub",
        "halo_em_chem_e_3_sub",
        "halo_em_chem_e_5_sub",
        "halo_em_tracer_e_3_sub",
        "halo_em_tracer_e_5_sub",
        "halo_em_scalar_e_3_sub",
        "halo_em_scalar_e_5_sub",
    }
)


NUMERICAL_STAGE_NAMES = (
    "zero_bdytend",
    "set_physical_bc3d:moist_old",
    "rk_step_prep",
    "rk_phys_bc_dry_1",
    "set_physical_bc3d:rho",
    "set_physical_bc3d:al",
    "set_physical_bc3d:ph_2",
    "first_rk_step_part1",
    "first_rk_step_part2",
    "rk_tendency",
    "relax_bdy_dry",
    "rk_addtend_dry",
    "spec_bdy_dry",
    "small_step_prep",
    "calc_p_rho",
    "calc_coef_w",
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
    "advance_uv",
    "spec_bdyupdate:u",
    "spec_bdyupdate:v",
    "advance_mu_t",
    "spec_bdyupdate:t",
    "spec_bdyupdate:mu_2",
    "spec_bdyupdate:muts",
    "advance_w",
    "sumflux",
    "spec_bdyupdate_ph",
    "zero_grad_bdy",
    "spec_bdyupdate:w",
    "set_physical_bc3d:ph_2:small",
    "set_physical_bc3d:al:small",
    "set_physical_bc3d:p:small",
    "set_physical_bc2d:muts:small",
    "set_physical_bc2d:mu_2:small",
    "set_physical_bc2d:mudf:small",
    "calc_mu_uv_1",
    "small_step_finish",
    "moist_scalar_advance",
    "calc_p_rho_phi",
    "calc_p_rho_phi:final",
    "advance_ppt",
    "phy_prep_part2",
    "set_physical_bc3d:moist_qv",
    "rk_phys_bc_dry_2",
    "set_physical_bc3d:ru_m",
    "set_physical_bc3d:rv_m",
    "set_physical_bc3d:ww_m",
    "set_physical_bc2d:mut",
    "set_physical_bc2d:muts",
    "set_phys_bc_dry_2",
    "set_physical_bc3d:tke_1",
    "set_physical_bc3d:tke_2",
    "spec_bdy_final:u",
    "spec_bdy_final:v",
    "spec_bdy_final:t",
    "spec_bdy_final:ph",
    "spec_bdy_final:mu",
    "spec_bdy_final:moist",
    "spec_bdy_final:moist_qv",
    "set_w_surface",
    "after_all_rk_steps",
)


@dataclass(frozen=True)
class RuntimeAssembly:
    """Inputs and callbacks for one strict WRF-JAX solve step."""

    case: CaseData
    grid: Any
    config_flags: Any
    fields: dict[str, Any]
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]]


def _namelist_index(config: Mapping[str, Mapping[str, Any]], group: str, key: str, default: Any, index: int = 0) -> Any:
    value = config.get(group, {}).get(key, default)
    values = value if isinstance(value, list) else [value]
    if index < len(values):
        return values[index]
    return values[-1]


def _runtime_grid(case: CaseData) -> SimpleNamespace:
    e_we = case.config.e_we
    e_sn = case.config.e_sn
    e_vert = case.config.e_vert
    halo = _horizontal_halo_width(case)
    start_time = wrf_atotime(case.config.start_time)
    stop_time = wrf_atotime(case.config.end_time)
    time_step = ESMFTimeInterval(BaseTime(s=int(case.config.time_step)))
    attrs = {
        "id": 1,
        "active_this_task": True,
        "dx": case.config.dx,
        "dy": case.config.dy,
        "dt": case.config.time_step,
        "domain_clock": ESMFClock(
            ESMFClockInt(
                time_step=time_step,
                start_time=start_time,
                stop_time=stop_time,
                curr_time=start_time,
            )
        ),
        "time_step_sound": 0,
        "max_msftx": 1.0,
        "max_msfty": 1.0,
        "current_time": 0.0,
        "sim_start_time": 0.0,
        "start_time": 0.0,
        "restart_alarm": False,
        "history_alarm": False,
        "boundary_alarm": False,
        "sr_x": 1,
        "sr_y": 1,
        "num_tiles": 1,
        "itimestep": 0,
        "dtbc": 0.0,
        "khdif": float(case.attrs.get("KHDIF", _namelist_index(case.config.namelist, "dynamics", "khdif", 0.0))),
        "kvdif": float(case.attrs.get("KVDIF", _namelist_index(case.config.namelist, "dynamics", "kvdif", 0.0))),
        "dampcoef": float(case.attrs.get("DAMPCOEF", _namelist_index(case.config.namelist, "dynamics", "dampcoef", 0.0))),
        "zdamp": float(case.attrs.get("ZDAMP", _namelist_index(case.config.namelist, "dynamics", "zdamp", 0.0))),
        "diff_6th_opt": int(case.attrs.get("DIFF_6TH_OPT", _namelist_index(case.config.namelist, "dynamics", "diff_6th_opt", 0))),
        "diff_6th_factor": float(
            case.attrs.get("DIFF_6TH_FACTOR", _namelist_index(case.config.namelist, "dynamics", "diff_6th_factor", 0.0))
        ),
    }
    for suffix, end in (("31", e_we), ("32", e_vert), ("33", e_sn)):
        attrs[f"sd{suffix}"] = 1
        attrs[f"ed{suffix}"] = int(end)
        attrs[f"sp{suffix}"] = 1
        attrs[f"ep{suffix}"] = int(end)
    attrs["sm31"] = 1 - halo
    attrs["em31"] = int(e_we) + halo
    attrs["sm32"] = 1
    attrs["em32"] = int(e_vert)
    attrs["sm33"] = 1 - halo
    attrs["em33"] = int(e_sn) + halo
    for suffix, end in (("31x", e_we), ("32x", e_vert), ("33x", e_sn), ("31y", e_we), ("32y", e_vert), ("33y", e_sn)):
        if suffix.startswith("31"):
            attrs[f"sm{suffix}"] = 1 - halo
            attrs[f"em{suffix}"] = int(end) + halo
        elif suffix.startswith("33"):
            attrs[f"sm{suffix}"] = 1 - halo
            attrs[f"em{suffix}"] = int(end) + halo
        else:
            attrs[f"sm{suffix}"] = 1
            attrs[f"em{suffix}"] = int(end)
        attrs[f"sp{suffix}"] = 1
        attrs[f"ep{suffix}"] = int(end)
    return SimpleNamespace(**attrs)


def _config_flags(case: CaseData) -> SimpleNamespace:
    namelist = case.config.namelist
    attrs = case.attrs
    fractional_seaice_value = int(
        attrs.get("FRACTIONAL_SEAICE", _namelist_index(namelist, "physics", "fractional_seaice", 0))
    )
    xice_threshold_default = 0.02 if fractional_seaice_value == 1 else 0.5
    radt = float(attrs.get("RADT", _namelist_index(namelist, "physics", "radt", 0.0)))
    stepra = int(attrs.get("STEPRA", _namelist_index(namelist, "physics", "stepra", 1)))
    if stepra <= 1 and radt > 0.0:
        stepra = max(1, int(round((radt * 60.0) / float(case.config.time_step))))
    return SimpleNamespace(
        do_avgflx_cugd=0,
        wrf_reference_nproc_x=max(
            1, int(os.environ.get("WRF_PY_REFERENCE_NPROC_X", "8"))
        ),
        specified=bool(_namelist_index(namelist, "bdy_control", "specified", True)),
        constant_bc=False,
        nested=False,
        spec_bdy_width=int(_namelist_index(namelist, "bdy_control", "spec_bdy_width", 5)),
        spec_zone=int(_namelist_index(namelist, "bdy_control", "spec_zone", 1)),
        relax_zone=int(
            _namelist_index(
                namelist,
                "bdy_control",
                "relax_zone",
                int(_namelist_index(namelist, "bdy_control", "spec_bdy_width", 5))
                - int(_namelist_index(namelist, "bdy_control", "spec_zone", 1)),
            )
        ),
        spec_exp=float(_namelist_index(namelist, "bdy_control", "spec_exp", 0.0)),
        use_adaptive_time_step=False,
        map_proj=int(case.grid.map_proj or 0),
        nwp_diagnostics=0,
        time_step=int(case.config.time_step),
        rk_ord=3,
        mp_physics=int(attrs.get("MP_PHYSICS", _namelist_index(namelist, "physics", "mp_physics", 0))),
        cu_physics=int(attrs.get("CU_PHYSICS", _namelist_index(namelist, "physics", "cu_physics", 0))),
        ra_lw_physics=int(attrs.get("RA_LW_PHYSICS", _namelist_index(namelist, "physics", "ra_lw_physics", 0))),
        ra_sw_physics=int(attrs.get("RA_SW_PHYSICS", _namelist_index(namelist, "physics", "ra_sw_physics", 0))),
        shcu_physics=int(attrs.get("SHCU_PHYSICS", _namelist_index(namelist, "physics", "shcu_physics", 0))),
        radt=radt,
        stepra=stepra,
        ra_call_offset=int(attrs.get("RA_CALL_OFFSET", _namelist_index(namelist, "physics", "ra_call_offset", 0))),
        icloud=int(attrs.get("ICLOUD", _namelist_index(namelist, "physics", "icloud", 1))),
        cldovrlp=int(attrs.get("CLDOVRLP", _namelist_index(namelist, "physics", "cldovrlp", 2))),
        idcor=int(attrs.get("IDCOR", _namelist_index(namelist, "physics", "idcor", 0))),
        ghg_input=int(attrs.get("GHG_INPUT", _namelist_index(namelist, "physics", "ghg_input", 1))),
        no_mp_heating=int(attrs.get("NO_MP_HEATING", _namelist_index(namelist, "physics", "no_mp_heating", 0))),
        mp_tend_lim=float(attrs.get("MP_TEND_LIM", _namelist_index(namelist, "physics", "mp_tend_lim", 10.0))),
        do_radar_ref=int(attrs.get("DO_RADAR_REF", _namelist_index(namelist, "physics", "do_radar_ref", 0))),
        calc_clean_atm_diag=int(
            attrs.get("CALC_CLEAN_ATM_DIAG", _namelist_index(namelist, "physics", "calc_clean_atm_diag", 0))
        ),
        bl_pbl_physics=int(attrs.get("BL_PBL_PHYSICS", _namelist_index(namelist, "physics", "bl_pbl_physics", -1))),
        sf_sfclay_physics=int(attrs.get("SF_SFCLAY_PHYSICS", _namelist_index(namelist, "physics", "sf_sfclay_physics", -1))),
        sf_surface_physics=int(attrs.get("SF_SURFACE_PHYSICS", _namelist_index(namelist, "physics", "sf_surface_physics", -1))),
        sf_urban_physics=int(attrs.get("SF_URBAN_PHYSICS", _namelist_index(namelist, "physics", "sf_urban_physics", 0))),
        iswater=int(attrs.get("ISWATER", 17)),
        isice=int(attrs.get("ISICE", 15)),
        isurban=int(attrs.get("ISURBAN", 13)),
        mminlu=str(attrs.get("MMINLU", "MODIFIED_IGBP_MODIS_NOAH")),
        julday=int(attrs.get("JULDAY", 1)),
        cen_lat=float(attrs.get("CEN_LAT", 0.0)),
        num_soil_layers=int(attrs.get("NUM_SOIL_LAYERS", 4)),
        isfflx=int(attrs.get("ISFFLX", _namelist_index(namelist, "physics", "isfflx", 1))),
        stepbl=int(attrs.get("STEPBL", _namelist_index(namelist, "physics", "stepbl", 1))),
        bldt=float(attrs.get("BLDT", _namelist_index(namelist, "physics", "bldt", 0.0))),
        fractional_seaice=fractional_seaice_value,
        seaice_albedo_opt=int(
            attrs.get("SEAICE_ALBEDO_OPT", _namelist_index(namelist, "physics", "seaice_albedo_opt", 0))
        ),
        seaice_albedo_default=float(
            attrs.get("SEAICE_ALBEDO_DEFAULT", _namelist_index(namelist, "physics", "seaice_albedo_default", 0.65))
        ),
        seaice_snowdepth_opt=int(
            attrs.get("SEAICE_SNOWDEPTH_OPT", _namelist_index(namelist, "physics", "seaice_snowdepth_opt", 0))
        ),
        seaice_snowdepth_max=float(
            attrs.get("SEAICE_SNOWDEPTH_MAX", _namelist_index(namelist, "physics", "seaice_snowdepth_max", 1.0e10))
        ),
        seaice_snowdepth_min=float(
            attrs.get("SEAICE_SNOWDEPTH_MIN", _namelist_index(namelist, "physics", "seaice_snowdepth_min", 0.001))
        ),
        seaice_thickness_opt=int(
            attrs.get("SEAICE_THICKNESS_OPT", _namelist_index(namelist, "physics", "seaice_thickness_opt", 0))
        ),
        seaice_thickness_default=float(
            attrs.get("SEAICE_THICKNESS_DEFAULT", _namelist_index(namelist, "physics", "seaice_thickness_default", 3.0))
        ),
        tice2tsk_if2cold=bool(
            attrs.get("TICE2TSK_IF2COLD", _namelist_index(namelist, "physics", "tice2tsk_if2cold", False))
        ),
        xice_threshold=float(
            attrs.get("XICE_THRESHOLD", _namelist_index(namelist, "physics", "xice_threshold", xice_threshold_default))
        ),
        usemonalb=bool(attrs.get("USEMONALB", _namelist_index(namelist, "physics", "usemonalb", False))),
        rdlai2d=bool(attrs.get("RDLAI2D", _namelist_index(namelist, "physics", "rdlai2d", False))),
        rdmaxalb=bool(attrs.get("RDMAXALB", _namelist_index(namelist, "physics", "rdmaxalb", True))),
        opt_thcnd=int(attrs.get("OPT_THCND", _namelist_index(namelist, "physics", "opt_thcnd", 1))),
        iz0tlnd=int(attrs.get("IZ0TLND", _namelist_index(namelist, "physics", "iz0tlnd", 0))),
        gwd_opt=int(attrs.get("GWD_OPT", _namelist_index(namelist, "dynamics", "gwd_opt", 0))),
        grid_fdda=int(attrs.get("GRID_FDDA", _namelist_index(namelist, "fdda", "grid_fdda", 0))),
        grid_sfdda=int(attrs.get("GRID_SFDDA", _namelist_index(namelist, "fdda", "grid_sfdda", 0))),
        stepcu=int(attrs.get("STEPCU", _namelist_index(namelist, "physics", "stepcu", 1))),
        cudt=float(attrs.get("CUDT", _namelist_index(namelist, "physics", "cudt", 0.0))),
        non_hydrostatic=bool(_namelist_index(namelist, "dynamics", "non_hydrostatic", True)),
        diff_opt=int(attrs.get("DIFF_OPT", _namelist_index(namelist, "dynamics", "diff_opt", 0))),
        km_opt=int(attrs.get("KM_OPT", _namelist_index(namelist, "dynamics", "km_opt", 0))),
        damp_opt=int(attrs.get("DAMP_OPT", _namelist_index(namelist, "dynamics", "damp_opt", 0))),
        dampcoef=float(attrs.get("DAMPCOEF", _namelist_index(namelist, "dynamics", "dampcoef", 0.0))),
        zdamp=float(attrs.get("ZDAMP", _namelist_index(namelist, "dynamics", "zdamp", 0.0))),
        use_input_w=bool(attrs.get("USE_INPUT_W", _namelist_index(namelist, "dynamics", "use_input_w", False))),
        khdif=float(attrs.get("KHDIF", _namelist_index(namelist, "dynamics", "khdif", 0.0))),
        kvdif=float(attrs.get("KVDIF", _namelist_index(namelist, "dynamics", "kvdif", 0.0))),
        diff_6th_opt=int(attrs.get("DIFF_6TH_OPT", _namelist_index(namelist, "dynamics", "diff_6th_opt", 0))),
        diff_6th_factor=float(attrs.get("DIFF_6TH_FACTOR", _namelist_index(namelist, "dynamics", "diff_6th_factor", 0.0))),
        c_s=float(attrs.get("C_S", _namelist_index(namelist, "dynamics", "c_s", 0.25))),
        mix_full_fields=bool(attrs.get("MIX_FULL_FIELDS", _namelist_index(namelist, "dynamics", "mix_full_fields", False))),
        mix_upper_bound=float(attrs.get("MIX_UPPER_BOUND", _namelist_index(namelist, "dynamics", "mix_upper_bound", 0.1))),
        w_damping=int(attrs.get("W_DAMPING", _namelist_index(namelist, "dynamics", "w_damping", 0))),
        smdiv=float(attrs.get("SMDIV", _namelist_index(namelist, "dynamics", "smdiv", 0.1))),
        emdiv=float(attrs.get("EMDIV", _namelist_index(namelist, "dynamics", "emdiv", 0.01))),
        epssm=float(attrs.get("EPSSM", _namelist_index(namelist, "dynamics", "epssm", 0.1))),
        zadvect_implicit=int(attrs.get("ZADVECT_IMPLICIT", _namelist_index(namelist, "dynamics", "zadvect_implicit", 0))),
        hypsometric_opt=int(attrs.get("HYPSOMETRIC_OPT", _namelist_index(namelist, "dynamics", "hypsometric_opt", 2))),
        momentum_adv_opt=int(attrs.get("MOMENTUM_ADV_OPT", _namelist_index(namelist, "dynamics", "momentum_adv_opt", 1))),
        moist_adv_opt=int(attrs.get("MOIST_ADV_OPT", _namelist_index(namelist, "dynamics", "moist_adv_opt", 1))),
        chem_adv_opt=int(attrs.get("CHEM_ADV_OPT", _namelist_index(namelist, "dynamics", "chem_adv_opt", 1))),
        tracer_adv_opt=int(attrs.get("TRACER_ADV_OPT", _namelist_index(namelist, "dynamics", "tracer_adv_opt", 1))),
        scalar_adv_opt=int(attrs.get("SCALAR_ADV_OPT", _namelist_index(namelist, "dynamics", "scalar_adv_opt", 1))),
        tke_adv_opt=int(attrs.get("TKE_ADV_OPT", _namelist_index(namelist, "dynamics", "tke_adv_opt", 1))),
        moist_mix2_off=bool(attrs.get("MOIST_MIX2_OFF", _namelist_index(namelist, "dynamics", "moist_mix2_off", False))),
        chem_mix2_off=bool(attrs.get("CHEM_MIX2_OFF", _namelist_index(namelist, "dynamics", "chem_mix2_off", False))),
        tracer_mix2_off=bool(attrs.get("TRACER_MIX2_OFF", _namelist_index(namelist, "dynamics", "tracer_mix2_off", False))),
        scalar_mix2_off=bool(attrs.get("SCALAR_MIX2_OFF", _namelist_index(namelist, "dynamics", "scalar_mix2_off", False))),
        tke_mix2_off=bool(attrs.get("TKE_MIX2_OFF", _namelist_index(namelist, "dynamics", "tke_mix2_off", False))),
        moist_mix6_off=bool(attrs.get("MOIST_MIX6_OFF", _namelist_index(namelist, "dynamics", "moist_mix6_off", False))),
        chem_mix6_off=bool(attrs.get("CHEM_MIX6_OFF", _namelist_index(namelist, "dynamics", "chem_mix6_off", False))),
        tracer_mix6_off=bool(attrs.get("TRACER_MIX6_OFF", _namelist_index(namelist, "dynamics", "tracer_mix6_off", False))),
        scalar_mix6_off=bool(attrs.get("SCALAR_MIX6_OFF", _namelist_index(namelist, "dynamics", "scalar_mix6_off", False))),
        tke_mix6_off=bool(attrs.get("TKE_MIX6_OFF", _namelist_index(namelist, "dynamics", "tke_mix6_off", False))),
        use_theta_m=int(attrs.get("USE_THETA_M", _namelist_index(namelist, "dynamics", "use_theta_m", 0))),
        have_bcs_moist=bool(attrs.get("HAVE_BCS_MOIST", _namelist_index(namelist, "bdy_control", "have_bcs_moist", False))),
        have_bcs_scalar=bool(attrs.get("HAVE_BCS_SCALAR", _namelist_index(namelist, "bdy_control", "have_bcs_scalar", False))),
        use_q_diabatic=int(attrs.get("USE_Q_DIABATIC", _namelist_index(namelist, "physics", "use_q_diabatic", 0))),
        top_lid=bool(attrs.get("TOP_LID", _namelist_index(namelist, "dynamics", "top_lid", False))),
        h_mom_adv_order=5,
        v_mom_adv_order=3,
        h_sca_adv_order=5,
        v_sca_adv_order=3,
        num_moist=7 if int(attrs.get("MP_PHYSICS", _namelist_index(namelist, "physics", "mp_physics", 0))) == 8 else 2,
        num_chem=0,
        num_tracer=0,
        num_scalar=3 if int(attrs.get("MP_PHYSICS", _namelist_index(namelist, "physics", "mp_physics", 0))) == 8 else 1,
        coupler_on=False,
    )


def _xkj(array: Any) -> Any:
    arr = jnp.asarray(array)
    if arr.ndim != 3:
        return arr
    return jnp.transpose(arr, (2, 0, 1))


_BOUNDARY_SUFFIXES = ("_bxs", "_bxe", "_bys", "_bye", "_btxs", "_btxe", "_btys", "_btye")
_DEFAULT_HORIZONTAL_HALO = 0


def _is_boundary_forcing_name(name: str) -> bool:
    return str(name).endswith(_BOUNDARY_SUFFIXES)


def _horizontal_halo_width(case: CaseData) -> int:
    del case
    return _DEFAULT_HORIZONTAL_HALO


def _pad_to_shape_edge(array: Any, target_shape: tuple[int, ...]) -> Any:
    arr = jnp.asarray(array)
    if arr.shape == target_shape:
        return arr
    if arr.ndim != len(target_shape):
        return arr
    pad_width = []
    for current, target in zip(arr.shape, target_shape, strict=True):
        if int(current) > int(target):
            return arr
        pad_width.append((0, int(target) - int(current)))
    if not any(width for _low, width in pad_width):
        return arr
    return jnp.pad(arr, tuple(pad_width), mode="edge")


def _pad_horizontal_memory_edge(array: Any, target_shape: tuple[int, ...], halo: int) -> Any:
    arr = jnp.asarray(array)
    if arr.shape == target_shape:
        return arr
    if arr.ndim != len(target_shape):
        return arr
    if arr.ndim == 2:
        pad_width = (
            (int(halo), int(target_shape[0]) - int(arr.shape[0]) - int(halo)),
            (int(halo), int(target_shape[1]) - int(arr.shape[1]) - int(halo)),
        )
    elif arr.ndim == 3:
        pad_width = (
            (int(halo), int(target_shape[0]) - int(arr.shape[0]) - int(halo)),
            (0, int(target_shape[1]) - int(arr.shape[1])),
            (int(halo), int(target_shape[2]) - int(arr.shape[2]) - int(halo)),
        )
    elif arr.ndim == 4:
        pad_width = (
            (int(halo), int(target_shape[0]) - int(arr.shape[0]) - int(halo)),
            (0, int(target_shape[1]) - int(arr.shape[1])),
            (int(halo), int(target_shape[2]) - int(arr.shape[2]) - int(halo)),
            (0, int(target_shape[3]) - int(arr.shape[3])),
        )
    else:
        return arr
    if any(low < 0 or high < 0 for low, high in pad_width):
        return arr
    return jnp.pad(arr, pad_width, mode="edge")


def _runtime_memory_dims(case: CaseData) -> tuple[int, int, int]:
    return int(case.config.e_we), int(case.config.e_sn), int(case.config.e_vert)


def _state_memory_dims(state: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(state["ide"]), int(state["jde"]), int(state["kde"])


def _pad_runtime_memory_field_by_dims(name: str, value: Any, e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> Any:
    """Expand WRF history/input-shaped arrays to WRF runtime memory extents."""

    if _is_boundary_forcing_name(name):
        return value
    try:
        arr = jnp.asarray(value)
    except (TypeError, ValueError):
        return value
    if arr.ndim not in (2, 3, 4):
        return value

    nx_mass = e_we - 1
    ny_mass = e_sn - 1
    nz_mass = e_vert - 1
    nz_stag = e_vert
    target_x = e_we + 2 * int(halo)
    target_y = e_sn + 2 * int(halo)

    if arr.ndim == 2:
        shape = tuple(int(dim) for dim in arr.shape)
        if shape == (target_x, target_y):
            return arr
        if shape[0] in {nx_mass, e_we} and shape[1] in {ny_mass, e_sn}:
            return _pad_horizontal_memory_edge(arr, (target_x, target_y), halo)
        return value

    if arr.ndim == 4:
        shape4 = tuple(int(dim) for dim in arr.shape)
        if shape4[0] == target_x and shape4[2] == target_y:
            return arr
        if shape4[0] in {nx_mass, e_we} and shape4[2] in {ny_mass, e_sn}:
            return _pad_horizontal_memory_edge(arr, (target_x, shape4[1], target_y, shape4[3]), halo)
        return value

    shape3 = tuple(int(dim) for dim in arr.shape)
    del nz_mass, nz_stag
    if shape3[0] == target_x and shape3[2] == target_y:
        return arr
    if shape3[0] in {nx_mass, e_we} and shape3[2] in {ny_mass, e_sn}:
        return _pad_horizontal_memory_edge(arr, (target_x, shape3[1], target_y), halo)
    return value


_PAD_HORIZONTAL_PACK_CACHE: dict[tuple[tuple[int, ...], int, int], Any] = {}


def _pad_horizontal_memory_edge_pack(values: tuple[Any, ...], target_shape: tuple[int, ...], halo: int) -> tuple[Any, ...]:
    """Pad same-shaped local fields to runtime memory extents in one JAX call."""

    key = (tuple(int(dim) for dim in target_shape), int(halo), len(values))
    cached = _PAD_HORIZONTAL_PACK_CACHE.get(key)
    if cached is None:

        def _pad_pack(*arrays: Any) -> tuple[Any, ...]:
            return tuple(_pad_horizontal_memory_edge(array, target_shape, halo) for array in arrays)

        cached = jax.jit(_pad_pack)
        _PAD_HORIZONTAL_PACK_CACHE[key] = cached
    return tuple(cached(*values))


def _runtime_memory_pad_target_shape(name: str, value: Any, e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> tuple[int, ...] | None:
    """Return the runtime target shape when a local array needs horizontal padding."""

    if _is_boundary_forcing_name(name):
        return None
    try:
        arr = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.ndim not in (2, 3, 4):
        return None

    nx_mass = e_we - 1
    ny_mass = e_sn - 1
    target_x = e_we + 2 * int(halo)
    target_y = e_sn + 2 * int(halo)

    if arr.ndim == 2:
        shape = tuple(int(dim) for dim in arr.shape)
        if shape == (target_x, target_y):
            return None
        if shape[0] in {nx_mass, e_we} and shape[1] in {ny_mass, e_sn}:
            target = (target_x, target_y)
            return target if shape != target else None
        return None

    if arr.ndim == 4:
        shape4 = tuple(int(dim) for dim in arr.shape)
        if shape4[0] == target_x and shape4[2] == target_y:
            return None
        if shape4[0] in {nx_mass, e_we} and shape4[2] in {ny_mass, e_sn}:
            target = (target_x, shape4[1], target_y, shape4[3])
            return target if shape4 != target else None
        return None

    shape3 = tuple(int(dim) for dim in arr.shape)
    if shape3[0] == target_x and shape3[2] == target_y:
        return None
    if shape3[0] in {nx_mass, e_we} and shape3[2] in {ny_mass, e_sn}:
        target = (target_x, shape3[1], target_y)
        return target if shape3 != target else None
    return None


def _pad_runtime_memory_fields_by_dims_packed(fields: Mapping[str, Any], e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> dict[str, Any]:
    """Expand local runtime fields, grouping same-target pads into fewer JAX dispatches."""

    output: dict[str, Any] = {}
    grouped_names: dict[tuple[int, ...], list[str]] = {}
    grouped_values: dict[tuple[int, ...], list[Any]] = {}
    for name, value in fields.items():
        target_shape = _runtime_memory_pad_target_shape(name, value, e_we, e_sn, e_vert, halo)
        if target_shape is None:
            output[name] = _pad_runtime_memory_field_by_dims(name, value, e_we, e_sn, e_vert, halo)
            continue
        grouped_names.setdefault(target_shape, []).append(name)
        grouped_values.setdefault(target_shape, []).append(jnp.asarray(value))

    for target_shape, names in grouped_names.items():
        padded_values = _pad_horizontal_memory_edge_pack(tuple(grouped_values[target_shape]), target_shape, halo)
        output.update(dict(zip(names, padded_values, strict=True)))
    return output


def _pad_runtime_memory_field(name: str, value: Any, case: CaseData) -> Any:
    return _pad_runtime_memory_field_by_dims(name, value, *_runtime_memory_dims(case), _horizontal_halo_width(case))


def _pad_runtime_memory_fields(fields: dict[str, Any], case: CaseData) -> dict[str, Any]:
    return {name: _pad_runtime_memory_field(name, value, case) for name, value in fields.items()}


_U_STAGGER_3D_NAMES = {"u", "u_1", "u_2", "u_save", "ru", "ru_tend", "ru_tendf", "ru_m", "uam"}
_V_STAGGER_3D_NAMES = {"v", "v_1", "v_2", "v_save", "rv", "rv_tend", "rv_tendf", "rv_m", "vam"}
_U_STAGGER_2D_NAMES = {"muu", "muus", "msfux", "msfuy", "mapfac_ux", "mapfac_uy"}
_V_STAGGER_2D_NAMES = {"muv", "muvs", "msfvx", "msfvx_inv", "msfvy", "mapfac_vx", "mapfac_vy"}


def _crop_runtime_field_for_physics(name: str, value: Any, e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> Any:
    """Crop padded runtime memory arrays back to WRF history extents for physics drivers."""

    if _is_boundary_forcing_name(name):
        return value
    arr = jnp.asarray(value)
    i0 = int(halo)
    j0 = int(halo)
    if arr.ndim == 2 and arr.shape[0] >= e_we and arr.shape[1] >= e_sn:
        if name in _U_STAGGER_2D_NAMES:
            return arr[i0 : i0 + e_we, j0 : j0 + e_sn - 1]
        if name in _V_STAGGER_2D_NAMES:
            return arr[i0 : i0 + e_we - 1, j0 : j0 + e_sn]
        return arr[i0 : i0 + e_we - 1, j0 : j0 + e_sn - 1]
    if arr.ndim == 3 and arr.shape[0] >= e_we and arr.shape[2] >= e_sn and arr.shape[1] in (e_vert - 1, e_vert):
        if name in _U_STAGGER_3D_NAMES:
            return arr[i0 : i0 + e_we, :, j0 : j0 + e_sn - 1]
        if name in _V_STAGGER_3D_NAMES:
            return arr[i0 : i0 + e_we - 1, :, j0 : j0 + e_sn]
        return arr[i0 : i0 + e_we - 1, :, j0 : j0 + e_sn - 1]
    if arr.ndim == 4 and arr.shape[0] >= e_we and arr.shape[2] >= e_sn and arr.shape[1] in (e_vert - 1, e_vert):
        return arr[i0 : i0 + e_we - 1, :, j0 : j0 + e_sn - 1, :]
    return value


def _crop_runtime_fields_for_physics(fields: Mapping[str, Any], e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> dict[str, Any]:
    return {name: _crop_runtime_field_for_physics(name, value, e_we, e_sn, e_vert, halo) for name, value in fields.items()}


def _pad_runtime_memory_fields_by_dims(fields: Mapping[str, Any], e_we: int, e_sn: int, e_vert: int, halo: int = 0) -> dict[str, Any]:
    return {name: _pad_runtime_memory_field_by_dims(name, value, e_we, e_sn, e_vert, halo) for name, value in fields.items()}


def _with_runtime_bounds(fields: Mapping[str, Any], grid: Any) -> dict[str, Any]:
    output = dict(fields)
    for name, value in zip(_RUNTIME_BOUND_NAMES, get_ijk_from_grid1(grid).as_tuple(), strict=True):
        output[name] = int(value)
    return output


_SCALAR_FLOAT_FIELD_NAMES = ("cf1", "cf2", "cf3", "cfn", "cfn1", "p_top", "p00", "u_frame", "v_frame")

_MOIST_FIELD_COMPONENTS: tuple[tuple[int, str], ...] = (
    (1, "qvapor"),
    (2, "qcloud"),
    (3, "qrain"),
    (4, "qice"),
    (5, "qsnow"),
    (6, "qgraup"),
)
_MIN_CONUS_NUM_MOIST = max(index for index, _name in _MOIST_FIELD_COMPONENTS) + 1
_SCALAR_FIELD_COMPONENTS: tuple[tuple[int, str, str], ...] = (
    (1, "qni", "qnice"),
    (2, "qnr", "qnrain"),
)
_SCALAR_BOUNDARY_ALIASES: dict[int, str] = {
    1: "qnice",
    2: "qnrain",
}
_MOIST_BOUNDARY_ALIASES: dict[int, str] = {index: name for index, name in _MOIST_FIELD_COMPONENTS}


def _scalar_float(state: Mapping[str, Any], name: str) -> float:
    cache_name = f"_{name}_float"
    if cache_name in state:
        return float(state[cache_name])
    value = state[name]
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, jax.core.Tracer):
        raise TypeError(f"{name} must be a static scalar, got a JAX tracer")
    return float(np.asarray(value).reshape(-1)[0])


def _scalar_float_or(state: Mapping[str, Any], name: str, default: float) -> float:
    if name not in state:
        return float(default)
    return _scalar_float(state, name)


def _with_scalar_float_cache(fields: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(fields)
    for name in _SCALAR_FLOAT_FIELD_NAMES:
        if name in output:
            output[f"_{name}_float"] = _scalar_float(output, name)
    return output


def _num_moist_from_fields(fields: Mapping[str, Any]) -> int:
    moist = fields.get("moist")
    if moist is None:
        return 2
    return int(jnp.asarray(moist).shape[-1])


def _num_scalar_from_fields(fields: Mapping[str, Any]) -> int:
    scalar = fields.get("scalar")
    if scalar is None:
        return 0
    return int(jnp.asarray(scalar).shape[-1])


def _moist_component(moist: Any, index: int, fallback: Any) -> Any:
    arr = jnp.asarray(moist)
    if arr.ndim == 4 and arr.shape[-1] > int(index):
        return arr[..., int(index)]
    return fallback


def _replace_moist_components(moist: Any, updates: Mapping[int, Any]) -> Any:
    out = jnp.asarray(moist)
    if out.ndim != 4:
        return out
    replacements = {}
    size = out.shape[-1]
    for index, value in updates.items():
        index = int(index)
        index = index + size if index < 0 else index
        if 0 <= index < size:
            replacements[index] = jnp.broadcast_to(jnp.asarray(value, dtype=out.dtype), out.shape[:-1])
    if not replacements:
        return out
    # Assemble the species axis once, preserving untouched components.
    pieces = []
    start = 0
    for index in sorted(replacements):
        if start < index:
            pieces.append(out[..., start:index])
        pieces.append(replacements[index][..., None])
        start = index + 1
    if start < size:
        pieces.append(out[..., start:])
    return jnp.concatenate(pieces, axis=-1)


def _moist_fields_from_array(moist: Any) -> dict[str, Any]:
    arr = jnp.asarray(moist)
    if arr.ndim != 4:
        return {}
    output: dict[str, Any] = {}
    for index, name in _MOIST_FIELD_COMPONENTS:
        if arr.shape[-1] > index:
            output[name] = arr[..., index]
    return output


def _build_conus_moist_array(fields: Mapping[str, Any]) -> Any:
    qvapor = jnp.asarray(fields["qvapor"])
    components: list[Any] = [jnp.zeros_like(qvapor)]
    for index, name in _MOIST_FIELD_COMPONENTS:
        if index == 1:
            value = qvapor
        else:
            value = jnp.asarray(fields.get(name, jnp.zeros_like(qvapor)), dtype=qvapor.dtype)
        components.append(value)
    return jnp.stack(components, axis=3)


def _build_conus_scalar_array(fields: Mapping[str, Any]) -> Any:
    template = jnp.asarray(fields["qvapor"])
    components: list[Any] = [jnp.zeros_like(template)]
    for _index, canonical, alias in _SCALAR_FIELD_COMPONENTS:
        value = fields.get(canonical, fields.get(alias, jnp.zeros_like(template)))
        components.append(jnp.asarray(value, dtype=template.dtype))
    return jnp.stack(components, axis=3)


def _scalar_component(scalar: Any, index: int, fallback: Any) -> Any:
    arr = jnp.asarray(scalar)
    if arr.ndim == 4 and arr.shape[-1] > int(index):
        return arr[..., int(index)]
    return fallback


def _replace_scalar_components(scalar: Any, updates: Mapping[int, Any]) -> Any:
    return _replace_moist_components(scalar, updates)


def _scalar_fields_from_array(scalar: Any) -> dict[str, Any]:
    arr = jnp.asarray(scalar)
    if arr.ndim != 4:
        return {}
    output: dict[str, Any] = {}
    for index, canonical, alias in _SCALAR_FIELD_COMPONENTS:
        if arr.shape[-1] > index:
            output[canonical] = arr[..., index]
            output[alias] = arr[..., index]
    return output


def _scalar_boundary_component(state: Mapping[str, Any], species_index: int, suffix: str, fallback: Any) -> Any:
    alias = _SCALAR_BOUNDARY_ALIASES.get(int(species_index), "")
    if alias:
        name = f"{alias}{suffix}"
        if name in state:
            return state[name]
    for index, canonical, component_alias in _SCALAR_FIELD_COMPONENTS:
        if int(index) != int(species_index):
            continue
        for prefix in (canonical, component_alias):
            name = f"{prefix}{suffix}"
            if name in state:
                return state[name]
    return fallback


def _moist_boundary_component(state: Mapping[str, Any], species_index: int, suffix: str, fallback: Any) -> Any:
    alias = _MOIST_BOUNDARY_ALIASES.get(int(species_index), "")
    if alias:
        name = f"{alias}{suffix}"
        if name in state:
            return state[name]
    return fallback


def _use_theta_m_value(case: CaseData) -> int:
    return int(case.attrs.get("USE_THETA_M", _namelist_index(case.config.namelist, "dynamics", "use_theta_m", 0)))


def _dry_theta_to_moist_theta_perturb(theta_dry_perturb: Any, qvapor: Any) -> Any:
    theta = jnp.asarray(theta_dry_perturb) + jnp.asarray(T0, dtype=jnp.asarray(theta_dry_perturb).dtype)
    qv = jnp.asarray(qvapor)
    ratio = jnp.asarray(R_V, dtype=theta.dtype) / jnp.asarray(R_D, dtype=theta.dtype)
    return theta * (1.0 + ratio * qv) - jnp.asarray(T0, dtype=theta.dtype)


def _fields(case: CaseData) -> dict[str, Any]:
    fields = dict(case.fields)
    fields["_case_dir"] = str(case.input_path.parent)
    fields["u"] = _xkj(case.state.u)
    fields["v"] = _xkj(case.state.v)
    fields["t"] = _xkj(case.state.t)
    fields["qvapor"] = _xkj(case.state.qvapor)
    fields = _pad_runtime_memory_fields(fields, case)
    for target, source in (("qni", "qnice"), ("qnr", "qnrain")):
        if target not in fields and source in fields:
            fields[target] = fields[source]
    for target, source in (("qnice", "qni"), ("qnrain", "qnr")):
        if target not in fields and source in fields:
            fields[target] = fields[source]
    for _index, name in _MOIST_FIELD_COMPONENTS:
        if name not in fields:
            fields[name] = jnp.zeros_like(fields["qvapor"])
    fields["moist"] = _build_conus_moist_array(fields)
    fields["scalar"] = _build_conus_scalar_array(fields)
    fields.setdefault("scalar_old", fields["scalar"])
    if _use_theta_m_value(case) == 1:
        fields["t"] = _dry_theta_to_moist_theta_perturb(fields["t"], fields["qvapor"])
    aliases = {
        "msfux": "mapfac_ux",
        "msfuy": "mapfac_uy",
        "msfvx": "mapfac_vx",
        "msfvy": "mapfac_vy",
        "msftx": "mapfac_mx",
        "msfty": "mapfac_my",
        "sina": "sinalpha",
        "cosa": "cosalpha",
        "ht": "hgt",
        "oc12d": "con",
    }
    for target, source in aliases.items():
        if target not in fields and source in fields:
            fields[target] = fields[source]
    if "msfvx_inv" not in fields and "msfvx" in fields:
        fields["msfvx_inv"] = 1.0 / fields["msfvx"]
    if "rho" not in fields and "al" in fields and "alb" in fields:
        fields["rho"] = 1.0 / (fields["al"] + fields["alb"])
    if "fzm" not in fields and "fnm" in fields:
        fields["fzm"] = fields["fnm"]
    if "fzp" not in fields and "fnp" in fields:
        fields["fzp"] = fields["fnp"]
    for target, source in (
        ("u_1", "u"),
        ("u_2", "u"),
        ("v_1", "v"),
        ("v_2", "v"),
        ("w_1", "w"),
        ("w_2", "w"),
        ("ph_2", "ph"),
        ("ph_1", "ph"),
        ("t_1", "t"),
        ("t_2", "t"),
        ("mu_1", "mu"),
        ("mu_2", "mu"),
    ):
        if target not in fields and source in fields:
            fields[target] = fields[source]
    zero_like = {
        "ru_tend": "u",
        "rv_tend": "v",
        "rw_tend": "w",
        "ph_tend": "ph",
        "t_tend": "t",
        "ru_tendf": "u",
        "rv_tendf": "v",
        "rw_tendf": "w",
        "ph_tendf": "ph",
        "t_tendf": "t",
        "tke_tend": "t",
        "tke_1": "t",
        "tke_2": "t",
        "u_save": "u",
        "v_save": "v",
        "w_save": "w",
        "ph_save": "ph",
        "t_save": "t",
        "rthften": "t",
        "h_diabatic": "t",
        "qv_diabatic": "qvapor",
        "qc_diabatic": "qcloud",
        "ww_e": "w",
        "ww_i": "w",
        "mu_tend": "mu",
        "mu_save": "mu",
        "mu_tendf": "mu",
        "muts": "mu",
        "mudf": "mu",
        "muave": "mu",
        "c2a": "t",
        "pm1": "p",
        "a": "w",
        "alpha": "w",
        "gamma": "w",
        "ww_1": "w",
        "uam": "u",
        "vam": "v",
        "wwam": "w",
        "t_ave": "t",
        "t_2ave": "t",
        "ru_m": "u",
        "rv_m": "v",
        "ww_m": "w",
    }
    for target, source in zero_like.items():
        if target not in fields and source in fields:
            fields[target] = jnp.zeros_like(fields[source])
    if "mu" in fields:
        mu_dtype = jnp.asarray(fields["mu"]).dtype
        if "u" in fields:
            fields["muus"] = jnp.zeros((fields["u"].shape[0], fields["u"].shape[2]), dtype=mu_dtype)
        if "v" in fields:
            fields["muvs"] = jnp.zeros((fields["v"].shape[0], fields["v"].shape[2]), dtype=mu_dtype)
    for target in ("moist_tend", "chem", "chem_tend", "scalar_tend", "tracer", "tracer_tend"):
        if target not in fields:
            fields[target] = jnp.zeros_like(fields["scalar"] if target == "scalar_tend" else fields["moist"])
    return fields


def _start_em_initialize_w(fields: dict[str, Any], grid: Any, config_flags: Any) -> dict[str, Any]:
    """Mirror WRF ``start_em`` initialization of terrain-induced ``W``."""

    if "w_2" not in fields or "u_2" not in fields or "v_2" not in fields:
        return fields

    ids = int(grid.sd31)
    ide = int(grid.ed31)
    jds = int(grid.sd33)
    jde = int(grid.ed33)
    kds = int(grid.sd32)
    kde = int(grid.ed32)
    ims = int(grid.sm31)
    ime = int(grid.em31)
    jms = int(grid.sm33)
    jme = int(grid.em33)
    kms = int(grid.sm32)
    kme = int(grid.em32)
    its = int(grid.sp31)
    ite = int(grid.ep31)
    jts = int(grid.sp33)
    jte = int(grid.ep33)
    kts = int(grid.sp32)
    kte = int(grid.ep32)

    i_slice = slice(its - ims, min(ite, ide - 1) - ims + 1)
    j_slice = slice(jts - jms, min(jte, jde - 1) - jms + 1)
    k1 = 1 - kms
    bottom_w = jnp.asarray(fields["w_2"])[i_slice, k1, j_slice]
    w_is_zero = bool(jnp.all(jnp.abs(bottom_w) < 1.0e-6))
    use_input_w = bool(getattr(config_flags, "use_input_w", False))
    if not (w_is_zero or not use_input_w):
        return fields

    cf1 = _scalar_float(fields, "cf1")
    cf2 = _scalar_float(fields, "cf2")
    cf3 = _scalar_float(fields, "cf3")
    common_args = (
        config_flags,
        fields["znw"],
        True,
        fields["ht"],
        cf1,
        cf2,
        cf3,
        1.0 / float(grid.dx),
        1.0 / float(grid.dy),
        fields["msftx"],
        fields["msfty"],
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
    w_1 = set_w_surface(
        common_args[0],
        common_args[1],
        common_args[2],
        fields["w_1"],
        common_args[3],
        fields["u_1"],
        fields["v_1"],
        *common_args[4:],
    )
    w_2 = set_w_surface(
        common_args[0],
        common_args[1],
        common_args[2],
        fields["w_2"],
        common_args[3],
        fields["u_2"],
        fields["v_2"],
        *common_args[4:],
    )
    fields = dict(fields)
    fields["w_1"] = w_1
    fields["w_2"] = w_2
    fields["w"] = w_2
    return fields


def _start_em_recompute_pressure_from_alt(fields: dict[str, Any], grid: Any, config_flags: Any) -> dict[str, Any]:
    """Mirror WRF ``start_em`` pressure update using initialized inverse density."""

    required = ("p", "pb", "al", "alb", "t_1", "moist")
    if any(name not in fields for name in required):
        return fields
    if bool(getattr(config_flags, "restart", False)):
        return fields
    if int(getattr(config_flags, "rebalance", 0)) == 1 or bool(getattr(config_flags, "cycling", False)):
        return fields

    ids = int(grid.sd31)
    ide = int(grid.ed31)
    jds = int(grid.sd33)
    jde = int(grid.ed33)
    kds = int(grid.sd32)
    kde = int(grid.ed32)
    ims = int(grid.sm31)
    jms = int(grid.sm33)
    kms = int(grid.sm32)
    its = int(getattr(grid, "sp31", ids))
    ite = int(getattr(grid, "ep31", ide))
    jts = int(getattr(grid, "sp33", jds))
    jte = int(getattr(grid, "ep33", jde))
    kts = int(getattr(grid, "sp32", kds))
    kte = int(getattr(grid, "ep32", kde))

    itf = min(ite, ide - 1)
    jtf = min(jte, jde - 1)
    ktf = min(kte - 1, kde - 1)
    if itf < its or jtf < jts or ktf < kts:
        return fields

    p_out = np.asarray(jax.device_get(jnp.asarray(fields["p"])), dtype=np.float32).copy()
    pb = np.asarray(jax.device_get(jnp.asarray(fields["pb"])), dtype=np.float32)
    al = np.asarray(jax.device_get(jnp.asarray(fields["al"])), dtype=np.float32)
    alb = np.asarray(jax.device_get(jnp.asarray(fields["alb"])), dtype=np.float32)
    if "alt" in fields:
        alt = np.asarray(jax.device_get(jnp.asarray(fields["alt"])), dtype=np.float32)
    else:
        alt = np.asarray(al + alb, dtype=np.float32)
    t_1 = np.asarray(jax.device_get(jnp.asarray(fields["t_1"])), dtype=np.float32)
    moist = np.asarray(jax.device_get(jnp.asarray(fields["moist"])), dtype=np.float32)

    i_slice = slice(its - ims, itf - ims + 1)
    j_slice = slice(jts - jms, jtf - jms + 1)
    k_slice = slice(kts - kms, ktf - kms + 1)
    region = (i_slice, k_slice, j_slice)

    p1000 = np.float32(100000.0)
    numerator = np.float32(np.float32(R_D) * np.float32(np.float32(T0) + t_1[region]))
    if int(getattr(config_flags, "use_theta_m", 0)) == 0 and moist.shape[-1] > 1:
        ratio = np.float32(np.float32(R_V) / np.float32(R_D))
        qvf = np.float32(1.0 + ratio * moist[region + (1,)])
        numerator = np.float32(numerator * qvf)
    temp = np.float32(numerator / np.float32(p1000 * alt[region]))
    # start_em uses scalar powf, unlike calc_p_rho_phi's vector VPOW.
    powered = np.asarray(jax.device_get(scalar_powf_for_wrf(temp, np.float32(1.4))))
    p_out[region] = np.float32(np.float32(p1000 * powered) - pb[region])

    output = dict(fields)
    output["p"] = jnp.asarray(p_out)
    output["p_hyd"] = jnp.asarray(np.float32(p_out + pb))
    return output


def _initialize_landuse_fields(fields: dict[str, Any], grid: Any, config_flags: Any, table_dir: str | Path) -> dict[str, Any]:
    """Run WRF ``landuse_init`` for surface fields absent from ``wrfinput``."""

    fields = dict(fields)
    shape2 = jnp.asarray(fields["xland"]).shape
    zeros2 = jnp.zeros(shape2, dtype=jnp.float32)
    xice = jnp.asarray(fields.get("xice", fields.get("seaice", zeros2)))
    albbck = jnp.asarray(fields.get("albbck", zeros2))
    landuse = landuse_init(
        fields["lu_index"],
        fields["snowc"],
        fields.get("albedo", albbck),
        albbck,
        fields["snoalb"],
        fields.get("mavail", zeros2),
        fields.get("emiss", zeros2),
        fields.get("embck", zeros2),
        fields.get("znt", zeros2),
        fields.get("z0", zeros2),
        fields.get("thc", zeros2),
        fields["xland"],
        xice,
        fields.get("xicem", zeros2),
        int(getattr(config_flags, "julday", 1)),
        float(getattr(config_flags, "cen_lat", 0.0)),
        int(config_flags.iswater),
        str(config_flags.mminlu),
        int(config_flags.isice),
        int(getattr(config_flags, "landuse_lucats", 0)),
        int(getattr(config_flags, "landuse_luseas", 0)),
        int(getattr(config_flags, "landuse_isn", 1)),
        int(getattr(config_flags, "fractional_seaice", 0)),
        fields.get("lu_state", jnp.zeros(7300, dtype=jnp.float32)),
        True,
        bool(getattr(config_flags, "usemonalb", False)),
        int(grid.sd31),
        int(grid.ed31),
        int(grid.sd33),
        int(grid.ed33),
        int(grid.sd32),
        int(grid.ed32),
        int(grid.sm31),
        int(grid.em31),
        int(grid.sm33),
        int(grid.em33),
        int(grid.sm32),
        int(grid.em32),
        int(grid.sp31),
        int(grid.ep31),
        int(grid.sp33),
        int(grid.ep33),
        int(grid.sp32),
        int(grid.ep32),
        landuse_tables=read_landuse_tables(table_dir),
    )
    fields.update(
        {
            "albedo": landuse.albedo,
            "albbck": landuse.albbck,
            "snoalb": landuse.snoalb,
            "mavail": landuse.mavail,
            "emiss": landuse.emiss,
            "embck": landuse.embck,
            "znt": landuse.znt,
            "z0": landuse.z0,
            "z0base": landuse.z0,
            "thc": landuse.thc,
            "xland": landuse.xland,
            "xice": xice,
            "xicem": landuse.xicem,
            "lu_state": landuse.lu_state,
        }
    )
    config_flags.isice = landuse.isice
    config_flags.landuse_lucats = landuse.lucats
    config_flags.landuse_luseas = landuse.luseas
    config_flags.landuse_isn = landuse.isn
    config_flags.fractional_seaice = landuse.fractional_seaice
    return fields


def _initialize_noah_lsm_fields(fields: dict[str, Any], grid: Any, config_flags: Any, table_dir: str | Path, noah_tables: Any) -> dict[str, Any]:
    """Run WRF ``LSMINIT`` startup state for the configured Noah land-surface scheme."""

    if int(getattr(config_flags, "sf_surface_physics", 0)) != 2:
        return fields

    fields = dict(fields)
    shape2 = jnp.asarray(fields["xland"]).shape
    zeros2 = jnp.zeros(shape2, dtype=jnp.float32)
    irr_rand_field = fields.get("irr_rand_field", jnp.zeros_like(jnp.asarray(fields["ivgtyp"], dtype=jnp.int32)))

    lsm_init = lsminit(
        fields["vegfra"],
        fields["snow"],
        fields["snowc"],
        fields["snowh"],
        fields.get("canwat", zeros2),
        fields.get("smstav", zeros2),
        fields.get("smstot", zeros2),
        fields.get("sfcrunoff", zeros2),
        fields.get("udrunoff", zeros2),
        fields.get("acsnow", zeros2),
        fields.get("acsnom", zeros2),
        fields["ivgtyp"],
        fields["isltyp"],
        fields["tslb"],
        fields["smois"],
        fields["sh2o"],
        fields["zs"],
        fields["dzs"],
        str(config_flags.mminlu),
        fields["snoalb"],
        True,
        True,
        bool(getattr(config_flags, "rdmaxalb", True)),
        int(config_flags.num_soil_layers),
        False,
        False,
        irr_rand_field,
        int(getattr(config_flags, "irr_ph", 0)),
        int(getattr(config_flags, "irr_freq", 1)),
        int(grid.sd31),
        int(grid.ed31),
        int(grid.sd33),
        int(grid.ed33),
        int(grid.sd32),
        int(grid.ed32),
        int(grid.sm31),
        int(grid.em31),
        int(grid.sm33),
        int(grid.em33),
        int(grid.sm32),
        int(grid.em32),
        int(grid.sp31),
        int(grid.ep31),
        int(grid.sp33),
        int(grid.ep33),
        int(grid.sp32),
        int(grid.ep32),
        parameters=noah_tables,
        table_dir=table_dir,
    )
    fields.update(
        {
            "vegfra": lsm_init.vegfra,
            "snow": lsm_init.snow,
            "snowc": lsm_init.snowc,
            "snowh": lsm_init.snowh,
            "canwat": lsm_init.canwat,
            "smstav": lsm_init.smstav,
            "smstot": lsm_init.smstot,
            "sfcrunoff": lsm_init.sfcrunoff,
            "udrunoff": lsm_init.udrunoff,
            "acsnow": lsm_init.acsnow,
            "acsnom": lsm_init.acsnom,
            "snoalb": lsm_init.snoalb,
            "tslb": lsm_init.tslb,
            "smois": lsm_init.smois,
            "sh2o": lsm_init.sh2o,
            "zs": lsm_init.zs,
            "dzs": lsm_init.dzs,
            "irr_rand_field": lsm_init.irr_rand_field,
        }
    )
    return fields


def _initialize_physics_startup_fields(fields: dict[str, Any], grid: Any, config_flags: Any) -> dict[str, Any]:
    """Apply WRF physics startup side effects needed before the first RK step."""

    fields = dict(fields)
    shape2 = jnp.asarray(fields["xland"]).shape
    shape_mass = jnp.asarray(fields["t"]).shape
    shape_full = jnp.asarray(fields["w"]).shape
    zeros2 = jnp.zeros(shape2, dtype=jnp.float32)
    zeros_mass = jnp.zeros(shape_mass, dtype=jnp.float32)
    zeros_full = jnp.zeros(shape_full, dtype=jnp.float32)

    for name in (
        "gsw",
        "glw",
        "mol",
        "rmol",
        "pblh",
        "hfx",
        "qfx",
        "lh",
        "flx_lh",
        "rainbl",
        "rainnc",
        "rainncv",
        "snownc",
        "snowncv",
        "graupelnc",
        "graupelncv",
        "sr",
        "rainc",
        "raincv",
        "rainsh",
        "pratec",
        "pratesh",
        "nca",
        "htop",
        "hbot",
        "cutop",
        "cubot",
        "cuppt",
        "olr",
        "rlwtoa",
        "lwcf",
        "swcf",
        "lwupt",
        "lwuptc",
        "lwuptcln",
        "lwdnt",
        "lwdntc",
        "lwdntcln",
        "lwupb",
        "lwupbc",
        "lwupbcln",
        "lwdnb",
        "lwdnbc",
        "lwdnbcln",
        "swupt",
        "swuptc",
        "swuptcln",
        "swdnt",
        "swdntc",
        "swdntcln",
        "swupb",
        "swupbc",
        "swupbcln",
        "swdnb",
        "swdnbc",
        "swdnbcln",
        "swdownc",
        "swddnic",
        "swddirc",
        "xcoszen",
        "coszr",
        "hrang",
    ):
        fields.setdefault(name, zeros2)
    for name in (
        "akhs",
        "akms",
        "rib",
        "br",
        "chs",
        "chs2",
        "cqs2",
        "flhc",
        "flqc",
        "qgh",
        "ct",
        "swdown",
        "swddir",
        "swddif",
    ):
        fields.setdefault(name, zeros2)
    for name in ("qsfc", "thz0", "qz0", "uz0", "vz0"):
        fields.setdefault(name, zeros2)
    fields.setdefault("cpm", jnp.full(shape2, 1004.0, dtype=jnp.float32))
    fields.setdefault("lowlyr", jnp.zeros(shape2, dtype=jnp.int32))
    fields.setdefault("ust", zeros2)
    fields.setdefault("ustar", fields["ust"])

    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten"):
        fields.setdefault(name, zeros_mass)
    for name in (
        "rthraten",
        "rthratenlw",
        "rthratenlwc",
        "rthratensw",
        "rthratenswc",
        "rthcuten",
        "rqvcuten",
        "rqccuten",
        "rqrcuten",
        "rqicuten",
        "rqscuten",
        "rucuten",
        "rvcuten",
        "refl_10cm",
        "re_cloud",
        "re_ice",
        "re_snow",
        "f_ice_phy",
        "f_rain_phy",
        "cldfra",
        "cldfra_old",
        "qv_diabatic",
        "qc_diabatic",
        "th_phy_m_t0",
    ):
        fields.setdefault(name, zeros_mass)
    fields.setdefault("h_diabatic", zeros_mass)
    fields.setdefault("cu_act_flag", jnp.ones(shape2, dtype=jnp.int32))
    fields.setdefault("has_reqc", 1)
    fields.setdefault("has_reqi", 1)
    fields.setdefault("has_reqs", 1)
    fields.setdefault("tke_pbl", zeros_full)
    fields.setdefault("exch_h", zeros_full)

    if int(getattr(config_flags, "sf_sfclay_physics", 0)) == 2:
        sfc = myjsfcinit(
            fields["lowlyr"],
            fields["ust"],
            fields["znt"],
            fields["xland"],
            fields.get("xice", fields.get("seaice", zeros2)),
            fields["ivgtyp"],
            bool(getattr(config_flags, "restart", False)),
            True,
            int(grid.sd31),
            int(grid.ed31),
            int(grid.sd33),
            int(grid.ed33),
            int(grid.sd32),
            int(grid.ed32),
            int(grid.sm31),
            int(grid.em31),
            int(grid.sm33),
            int(grid.em33),
            int(grid.sm32),
            int(grid.em32),
            int(grid.sp31),
            int(grid.ep31),
            int(grid.sp33),
            int(grid.ep33),
            int(grid.sp32),
            int(grid.ep32),
        )
        fields["lowlyr"] = sfc.lowlyr
        fields["ust"] = sfc.ustar
        fields["ustar"] = sfc.ustar
        fields["znt"] = sfc.z0
        fields.setdefault("z0base", sfc.z0)

    if int(getattr(config_flags, "bl_pbl_physics", 0)) == 2:
        pbl = myjpblinit(
            fields["rublten"],
            fields["rvblten"],
            fields["rthblten"],
            fields["rqvblten"],
            fields["tke_pbl"],
            fields["exch_h"],
            bool(getattr(config_flags, "restart", False)),
            True,
            int(grid.sd31),
            int(grid.ed31),
            int(grid.sd33),
            int(grid.ed33),
            int(grid.sd32),
            int(grid.ed32),
            int(grid.sm31),
            int(grid.em31),
            int(grid.sm33),
            int(grid.em33),
            int(grid.sm32),
            int(grid.em32),
            int(grid.sp31),
            int(grid.ep31),
            int(grid.sp33),
            int(grid.ep33),
            int(grid.sp32),
            int(grid.ep32),
        )
        fields["rublten"] = pbl.rublten
        fields["rvblten"] = pbl.rvblten
        fields["rthblten"] = pbl.rthblten
        fields["rqvblten"] = pbl.rqvblten
        fields["tke_pbl"] = pbl.tke_myj
        fields["exch_h"] = pbl.exch_h

    return fields


def _with_current_dry_aliases(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a field view where WRF dry-state aliases point at current state."""

    fields = dict(state)
    for alias, current in (
        ("u", "u_2"),
        ("v", "v_2"),
        ("w", "w_2"),
        ("t", "t_2"),
        ("ph", "ph_2"),
        ("mu", "mu_2"),
    ):
        if current in state:
            fields[alias] = state[current]
    return fields


def _set_current_dry_aliases_inplace(state: dict[str, Any]) -> None:
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


def _shape2(state: Mapping[str, Any]) -> tuple[int, int]:
    for name in ("tsk", "psfc", "xland", "hgt"):
        value = state.get(name)
        if value is not None and getattr(value, "ndim", 0) == 2:
            return tuple(int(dim) for dim in value.shape)
    if all(name in state for name in ("ims", "ime", "jms", "jme")):
        return max(int(state["ime"]) - int(state["ims"]) + 1, 1), max(int(state["jme"]) - int(state["jms"]) + 1, 1)
    if "ide" in state and "jde" in state:
        return max(int(state["ide"]), 1), max(int(state["jde"]), 1)
    raise MissingKernelImplementationError("surface runtime requires at least one 2-D land/surface field")


def _zeros2(state: Mapping[str, Any]) -> Any:
    return jnp.zeros(_shape2(state), dtype=jnp.float32)


def _full2(state: Mapping[str, Any], value: float) -> Any:
    return jnp.full(_shape2(state), value, dtype=jnp.float32)


def _zeros3_like(state: Mapping[str, Any], name: str = "t_phy") -> Any:
    return jnp.zeros_like(jnp.asarray(state[name]))


def _field_or(state: Mapping[str, Any], name: str, default: Any) -> Any:
    value = state.get(name)
    return default if value is None else value


def _myjpbl_elflx_field(state: Mapping[str, Any], default: Any) -> Any:
    return _field_or(state, "lh", _field_or(state, "elflx", _field_or(state, "flx_lh", default)))


def _radiation_real(value: Any) -> Any:
    """Keep REAL radiation fields on their JAX device."""
    return jnp.asarray(value, dtype=jnp.float32)


def _fortran_int(value: Any) -> Any:
    arr = jnp.asarray(value, dtype=jnp.int32)
    if isinstance(arr, jax.core.Tracer):
        return arr
    return np.asfortranarray(np.asarray(arr, dtype=np.int32))


def _has_traced_array(*values: Any) -> bool:
    # Inspect PyTree leaves without converting, uploading or concatenating
    # them. Radiation lookup tables can contain heterogeneous array tuples.
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values))


def _has_nonzero_host_array(*values: Any) -> bool:
    for value in values:
        arr = jnp.asarray(value)
        if isinstance(arr, jax.core.Tracer):
            return True
        if np.any(np.asarray(arr) != 0):
            return True
    return False


def _to_mass_k(value: Any, mass_template: Any) -> Any:
    arr = jnp.asarray(value)
    mass = jnp.asarray(mass_template)
    if arr.ndim == 3 and arr.shape[1] == mass.shape[1] + 1:
        return arr[:, : mass.shape[1], :]
    if arr.ndim == 3 and arr.shape[1] == mass.shape[1] - 1:
        return jnp.zeros_like(mass).at[:, : arr.shape[1], :].set(arr)
    return arr


def _to_full_k(value: Any, full_template: Any) -> Any:
    arr = jnp.asarray(value)
    full = jnp.asarray(full_template)
    if arr.ndim == 3 and arr.shape[1] == full.shape[1] - 1:
        return jnp.zeros_like(full).at[:, : arr.shape[1], :].set(arr)
    if arr.ndim == 3 and arr.shape[1] == full.shape[1] + 1:
        return arr[:, : full.shape[1], :]
    return arr


def _radiation_thermodynamic_columns(state: Mapping[str, Any], *values: Any) -> tuple[Any, ...]:
    """Fill unused memory columns without changing physical columns or seeds."""
    its, ite, jts, jte, _, _ = _physics_bounds(state)
    i0, i1 = its - int(state["ims"]), ite - int(state["ims"])
    j0, j1 = jts - int(state["jms"]), jte - int(state["jms"])
    result = []
    for value in values:
        array = jnp.asarray(value)
        if i0 != 0 or i1 != array.shape[0] - 1:
            indices = jnp.clip(jnp.arange(array.shape[0]), i0, i1)
            array = jnp.take(array, indices, axis=0)
        if j0 != 0 or j1 != array.shape[2] - 1:
            indices = jnp.clip(jnp.arange(array.shape[2]), j0, j1)
            array = jnp.take(array, indices, axis=2)
        result.append(array)
    return tuple(result)


@jax.jit
def _radiation_mass_inputs(values: tuple[Any, ...], full: Any, mass: Any) -> tuple[tuple[Any, ...], Any]:
    """Pack mass inputs together, preserving the prior padding and REAL casts."""
    expanded = tuple(_to_full_k(value, full) for value in values)

    def nonzero(value):
        # Integer inspection retains subnormals when a CPU kernel enables FTZ.
        if jnp.issubdtype(value.dtype, jnp.floating):
            bits = value.dtype.itemsize * 8
            unsigned = jnp.dtype(f"uint{bits}")
            magnitude = jax.lax.bitcast_convert_type(value, unsigned) & jnp.asarray((1 << (bits - 1)) - 1, unsigned)
            return jnp.any(magnitude != 0)
        return jnp.any(value != 0)

    # Cloud presence historically included the stored upper level, even though
    # the radiation solvers consume only mass levels. Keep that decision intact.
    condensate_present = (
        nonzero(expanded[1]) | nonzero(expanded[2]) | nonzero(expanded[3])
    )
    return tuple(_to_mass_k(value, mass) for value in expanded), condensate_present


def _slice_radiation_arg(value: Any, *, nx: int, ny: int, j_start: int, j_stop: int) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None:
        return value
    if len(shape) == 3 and int(shape[0]) == nx and int(shape[2]) == ny:
        return value[:, :, j_start:j_stop]
    if len(shape) == 2 and int(shape[0]) == nx and int(shape[1]) == ny:
        return value[:, j_start:j_stop]
    return value


def _materialize_radiation_result(result: Any) -> Any:
    result_type = type(result)
    return result_type(
        **{
            name: np.asfortranarray(np.asarray(getattr(result, name), dtype=np.float32))
            for name in result._fields
        }
    )


def _concat_radiation_results(results: list[Any]) -> Any:
    if not results:
        raise ValueError("cannot concatenate empty radiation results")
    result_type = type(results[0])
    values: dict[str, Any] = {}
    for name in results[0]._fields:
        chunks = [jnp.asarray(getattr(result, name), dtype=jnp.float32) for result in results]
        ndim = chunks[0].ndim
        axis = 2 if ndim == 3 else 1 if ndim == 2 else 0
        values[name] = jnp.concatenate(chunks, axis=axis)
    return result_type(**values)


def _select_radiation_columns(
    value: Any,
    indices: np.ndarray,
    *,
    nx: int,
    ny: int,
    target_columns: int | None = None,
) -> Any:
    """Gather an arbitrary set of horizontal columns for a radiation call."""

    shape = getattr(value, "shape", None)
    if shape is None:
        return value
    if len(shape) == 3 and int(shape[0]) == nx and int(shape[2]) == ny:
        array = np.asarray(value)
        columns = np.swapaxes(array, 1, 2).reshape(nx * ny, array.shape[1])
        selected = columns[indices]
        if target_columns is not None and selected.shape[0] < int(target_columns):
            selected = np.concatenate(
                (
                    selected,
                    np.repeat(selected[:1], int(target_columns) - selected.shape[0], axis=0),
                ),
                axis=0,
            )
        return np.asfortranarray(selected[:, :, None])
    if len(shape) == 2 and int(shape[0]) == nx and int(shape[1]) == ny:
        columns = np.asarray(value).reshape(nx * ny)
        selected = columns[indices]
        if target_columns is not None and selected.shape[0] < int(target_columns):
            selected = np.concatenate(
                (
                    selected,
                    np.repeat(selected[:1], int(target_columns) - selected.shape[0], axis=0),
                ),
                axis=0,
            )
        return np.asfortranarray(selected[:, None])
    return value


def _trim_radiation_columns(result: Any, column_count: int) -> Any:
    """Discard padding columns from a fixed-capacity sparse-cloud call."""

    result_type = type(result)
    fields: dict[str, Any] = {}
    for name in result._fields:
        value = np.asarray(getattr(result, name), dtype=np.float32)
        if value.ndim == 3:
            fields[name] = np.asfortranarray(value[:column_count, :, :])
        elif value.ndim == 2:
            fields[name] = np.asfortranarray(value[:column_count, :])
        else:
            raise ValueError(
                f"unsupported padded radiation result field {name} shape={value.shape}"
            )
    return result_type(**fields)


def _scatter_radiation_columns(
    base: Any,
    update: Any,
    indices: np.ndarray,
    *,
    nx: int,
    ny: int,
) -> Any:
    """Put a gathered radiation result back into the original grid layout."""

    result_type = type(base)
    fields: dict[str, Any] = {}
    for name in base._fields:
        base_array = np.asarray(getattr(base, name), dtype=np.float32)
        update_array = np.asarray(getattr(update, name), dtype=np.float32)
        if base_array.ndim == 3:
            columns = np.swapaxes(base_array, 1, 2).reshape(
                nx * ny, base_array.shape[1]
            ).copy()
            update_columns = np.swapaxes(update_array, 1, 2).reshape(
                indices.size, update_array.shape[1]
            )
            columns[indices] = update_columns
            fields[name] = np.asfortranarray(
                np.swapaxes(columns.reshape(nx, ny, -1), 1, 2)
            )
        elif base_array.ndim == 2:
            columns = base_array.reshape(nx * ny).copy()
            columns[indices] = update_array.reshape(indices.size)
            fields[name] = np.asfortranarray(columns.reshape(nx, ny))
        else:
            raise ValueError(
                f"unsupported radiation result field {name} shape={base_array.shape}"
            )
    return result_type(**fields)


def _run_sparse_cloud_radiation_side(
    kernel: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    max_columns: int,
) -> Any:
    """Run clear and cloudy columns separately when clouds are sparse.

    RRTMG's clear-column branch is independent of the cloud optical-depth
    branch.  Keeping the two populations in separate compiled shapes lets
    XLA avoid paying the cloudy branch cost for the large clear-sky majority.
    The results are restored in the original WRF ``(west_east, bottom_top,
    south_north)`` layout, so this is an outer scheduling optimization only.
    """

    cldfra = np.asarray(kwargs["cldfra3d"])
    nx, _levels, ny = (int(cldfra.shape[0]), int(cldfra.shape[1]), int(cldfra.shape[2]))
    cloud_indices = np.flatnonzero(
        np.any(cldfra != 0, axis=1).reshape(nx * ny)
    )
    sparse_capacity = max(1, (nx * ny + 3) // 4)
    if cloud_indices.size > sparse_capacity:
        return _run_radiation_columns_chunked(
            kernel,
            kwargs,
            max_columns=max_columns,
        )
    clear_kwargs = dict(kwargs)
    clear_kwargs["icloud"] = 0
    clear_result = _run_radiation_columns_chunked(
        kernel,
        clear_kwargs,
        max_columns=max_columns,
    )
    if cloud_indices.size == 0:
        return clear_result
    cloud_kwargs = {
        name: _select_radiation_columns(
            value,
            cloud_indices,
            nx=nx,
            ny=ny,
            target_columns=sparse_capacity,
        )
        for name, value in kwargs.items()
    }
    cloud_result = _run_radiation_columns_chunked(
        kernel,
        cloud_kwargs,
        max_columns=max_columns,
    )
    return _scatter_radiation_columns(
        _materialize_radiation_result(clear_result),
        _materialize_radiation_result(
            _trim_radiation_columns(cloud_result, cloud_indices.size)
        ),
        cloud_indices,
        nx=nx,
        ny=ny,
    )


@lru_cache(maxsize=16)
def _radiation_batch_program(kernel: Callable[..., Any], static_values: tuple, rows: int,
                             device: Any = None) -> Callable[..., Any]:
    """Keep fixed-size radiation row batches and their outputs on the device."""
    static = dict(static_values)

    def radiation_batches(values):
        nx, _, ny = values["t3d"].shape
        if ny <= rows:
            return kernel(**static, **values)
        count = (ny + rows - 1) // rows
        batch_rows = (ny + count - 1) // count
        extent = count * batch_rows
        # Balance row batches so one compiled shape also handles the remainder.
        # Repeated edge columns are independent and discarded from the result.
        packed = {}
        for name, value in values.items():
            shape = getattr(value, "shape", ())
            if len(shape) == 3 and shape[0] == nx and shape[2] == ny:
                value = jnp.pad(value, ((0, 0), (0, 0), (0, extent - ny)), mode="edge")
                packed[name] = value.reshape(nx, shape[1], count, batch_rows).transpose(2, 0, 1, 3)
            elif tuple(shape) == (nx, ny):
                value = jnp.pad(value, ((0, 0), (0, extent - ny)), mode="edge")
                packed[name] = value.reshape(nx, count, batch_rows).transpose(1, 0, 2)

        batches = jax.lax.map(lambda batch: kernel(**static, **dict(values, **batch)), packed)
        fields = {}
        for name, value in batches._asdict().items():
            value = jnp.asarray(value, dtype=jnp.float32)
            if value.ndim == 4:
                fields[name] = value.transpose(1, 2, 0, 3).reshape(nx, value.shape[2], extent)[:, :, :ny]
            elif value.ndim == 3:
                fields[name] = value.transpose(1, 0, 2).reshape(nx, extent)[:, :ny]
            else:
                raise ValueError(f"unsupported radiation field {name}: {value.shape}")
        return type(batches)(**fields)

    return reusable_terminal_jit(radiation_batches, context={
        "program": "radiation_batches", "kernel": (kernel.__module__, kernel.__qualname__),
        "static": static_values, "rows": rows,
        "device": None if device is None else (device.platform, device.id),
    })


def _run_radiation_columns_chunked(kernel: Callable[..., Any], kwargs: Mapping[str, Any], *, max_columns: int) -> Any:
    t3d = jnp.asarray(kwargs["t3d"])
    nx = int(t3d.shape[0])
    ny = int(t3d.shape[2])
    total_columns = nx * ny
    # A traced radiation event is already one pure JAX batch. Python thread
    # scheduling must stay outside tracing. For ordinary calls the row batches
    # retain their device arrays, including while concatenating the outputs.
    if _has_traced_array(t3d, *kwargs.values()):
        return kernel(**kwargs)
    if _RRTMG_CHUNK_WORKERS > 1:
        parallel_rows = max(1, (ny + _RRTMG_CHUNK_WORKERS - 1) // _RRTMG_CHUNK_WORKERS)
        parallel_columns = parallel_rows * nx
        max_columns = parallel_columns if int(max_columns) <= 0 else min(int(max_columns), parallel_columns)
    if int(max_columns) <= 0 or total_columns <= int(max_columns):
        return kernel(**kwargs)

    rows_per_chunk = max(1, int(max_columns) // max(1, nx))
    if jax.default_backend() == "gpu" and _RRTMG_CHUNK_WORKERS == 1 and kernel in (
        _rrtmg_lwrad_wrf_columns_jit, _rrtmg_swrad_wrf_columns_jit,
    ):
        names = _RRTMG_LW_STATIC_ARGNAMES if kernel is _rrtmg_lwrad_wrf_columns_jit else _RRTMG_SW_STATIC_ARGNAMES
        static = tuple((name, kwargs[name]) for name in names if name in kwargs)
        values = {name: value for name, value in kwargs.items() if name not in names}
        devices = configured_devices()
        ranges = batch_partitions(ny, rows_per_chunk, len(devices))
        devices = devices[:len(ranges)]
        partitions = [values] if len(ranges) == 1 else [
            {name: _slice_radiation_arg(value, nx=nx, ny=ny, j_start=start, j_stop=stop)
             for name, value in values.items()} for start, stop in ranges
        ]

        def compute(local, device):
            return _radiation_batch_program(kernel, static, rows_per_chunk, device)(local)

        results = submit_device_batches(compute, partitions, devices)
        if len(results) == 1:
            return results[0]
        # Bring only outputs back to the state device; keep bounded workspaces
        # local and retain the exact row/batch ordering used with one GPU.
        target = next(iter(t3d.devices()))
        return _concat_radiation_results([jax.device_put(result, target) for result in results])
    chunk_ranges = [(j_start, min(ny, j_start + rows_per_chunk)) for j_start in range(0, ny, rows_per_chunk)]

    def _run_one_chunk(chunk_range: tuple[int, int]) -> Any:
        j_start, j_stop = chunk_range
        chunk_kwargs = {
            name: _slice_radiation_arg(value, nx=nx, ny=ny, j_start=j_start, j_stop=j_stop)
            for name, value in kwargs.items()
        }
        return kernel(**chunk_kwargs)

    if _RRTMG_CHUNK_WORKERS <= 1 or len(chunk_ranges) <= 2:
        chunks = [_run_one_chunk(chunk_range) for chunk_range in chunk_ranges]
        return _concat_radiation_results(chunks)

    with ThreadPoolExecutor(max_workers=min(_RRTMG_CHUNK_WORKERS, len(chunk_ranges))) as pool:
        chunks = list(pool.map(_run_one_chunk, chunk_ranges))
    return _concat_radiation_results(chunks)


def _run_radiation_sides(
    lw_kernel: Callable[..., Any],
    lw_kwargs: Mapping[str, Any],
    sw_kernel: Callable[..., Any],
    sw_kwargs: Mapping[str, Any],
    *,
    max_columns: int,
    parallel: bool,
) -> tuple[Any, float, Any, float]:
    """Run independent longwave and shortwave column batches.

    A paired submission reports its shared enqueue duration for both sides;
    these overlapping durations are not individual GPU times and cannot add.
    """

    if (
        parallel
        and jax.default_backend() == "gpu"
        and lw_kernel is _rrtmg_lwrad_wrf_columns_jit
        and sw_kernel is _rrtmg_swrad_wrf_columns_jit
        and not _RRTMG_SPARSE_CLOUDS
        and _RRTMG_CHUNK_WORKERS == 1
        and not _has_traced_array(*lw_kwargs.values(), *sw_kwargs.values())
        and all(
            int(max_columns) <= 0
            or int(values["t3d"].shape[0]) * int(values["t3d"].shape[2]) <= int(max_columns)
            for values in (lw_kwargs, sw_kwargs)
        )
    ):
        start = perf_counter()
        lw_result, sw_result = _run_paired_rrtmg(lw_kwargs, sw_kwargs)
        seconds = perf_counter() - start
        return lw_result, seconds, sw_result, seconds

    def run_one(kernel: Callable[..., Any], kwargs: Mapping[str, Any]) -> tuple[Any, float]:
        start = perf_counter()
        if _RRTMG_SPARSE_CLOUDS and not _has_traced_array(*kwargs.values()):
            result = _run_sparse_cloud_radiation_side(
                kernel,
                kwargs,
                max_columns=max_columns,
            )
        else:
            result = _run_radiation_columns_chunked(
                kernel,
                kwargs,
                max_columns=max_columns,
            )
        return result, perf_counter() - start

    if not parallel:
        lw_result, lw_seconds = run_one(lw_kernel, lw_kwargs)
        sw_result, sw_seconds = run_one(sw_kernel, sw_kwargs)
        return lw_result, lw_seconds, sw_result, sw_seconds

    with ThreadPoolExecutor(max_workers=2) as pool:
        lw_future = pool.submit(run_one, lw_kernel, lw_kwargs)
        sw_future = pool.submit(run_one, sw_kernel, sw_kwargs)
        lw_result, lw_seconds = lw_future.result()
        sw_result, sw_seconds = sw_future.result()
    return lw_result, lw_seconds, sw_result, sw_seconds


def _run_column_batch_chunked(
    batch_fn: Callable[..., tuple[Any, ...]],
    args: tuple[Any, ...],
    *,
    max_columns: int,
    workers: int = 1,
) -> tuple[Any, ...]:
    ncol = int(jnp.asarray(args[0]).shape[0])
    worker_count = max(1, int(workers))
    chunk_columns = int(max_columns)
    if worker_count > 1:
        parallel_columns = (ncol + worker_count - 1) // worker_count
        chunk_columns = parallel_columns if chunk_columns <= 0 else min(chunk_columns, parallel_columns)
    if chunk_columns <= 0 or ncol <= chunk_columns:
        return batch_fn(*args)

    column_arguments = tuple(
        getattr(arg, "ndim", 0) > 0 and arg.shape[0] == ncol for arg in args
    )
    if _has_traced_array(*args):
        # Keep bounded column work inside the device graph. In particular, an
        # enclosing timestep scan must not expand Thompson to the entire domain.
        complete = (ncol + chunk_columns - 1) // chunk_columns
        chunk_columns = (ncol + complete - 1) // complete
        extent = complete * chunk_columns
        mapped_arguments = tuple(
            jnp.pad(arg, ((0, extent - ncol), *((0, 0),) * (arg.ndim - 1)), mode="edge")
            .reshape((complete, chunk_columns, *arg.shape[1:]))
            for arg, is_column in zip(args, column_arguments, strict=True)
            if is_column
        )

        def mapped_batch(columns):
            iterator = iter(columns)
            return batch_fn(*(
                next(iterator) if is_column else arg
                for arg, is_column in zip(args, column_arguments, strict=True)
            ))

        # Equal shapes compile the column physics once; padded columns are
        # independent and disappear before returning to the model state.
        return tuple(
            value.reshape((extent, *value.shape[2:]))[:ncol]
            for value in jax.lax.map(mapped_batch, mapped_arguments)
        )

    device_resident = any(
        isinstance(arg, jax.Array) and any(device.platform != "cpu" for device in arg.devices())
        for arg in args
    )
    chunk_ranges = tuple(
        (start, min(ncol, start + chunk_columns))
        for start in range(0, ncol, chunk_columns)
    )

    def run_chunk(chunk_range: tuple[int, int]) -> tuple[Any, ...]:
        start, stop = chunk_range
        chunk_args = tuple(
            arg[start:stop] if getattr(arg, "ndim", 0) > 0 and int(getattr(arg, "shape", (0,))[0]) == ncol else arg
            for arg in args
        )
        chunk_result = batch_fn(*chunk_args)
        if device_resident:
            return tuple(jnp.asarray(value, dtype=jnp.float32) for value in chunk_result)
        return tuple(np.asarray(value, dtype=np.float32) for value in chunk_result)

    if worker_count > 1 and len(chunk_ranges) > 1 and not device_resident:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(chunk_ranges))) as pool:
            chunk_results = tuple(pool.map(run_chunk, chunk_ranges))
    else:
        chunk_results = tuple(run_chunk(chunk_range) for chunk_range in chunk_ranges)
    if not chunk_results:
        raise ValueError("cannot concatenate empty column batch")
    if device_resident:
        return tuple(
            jnp.concatenate(tuple(result[index] for result in chunk_results), axis=0)
            for index in range(len(chunk_results[0]))
        )
    return tuple(
        jnp.asarray(
            np.concatenate(tuple(result[index] for result in chunk_results), axis=0),
            dtype=jnp.float32,
        )
        for index in range(len(chunk_results[0]))
    )


def _physics_bounds(state: Mapping[str, Any], *, skip_spec_zone: bool = False) -> tuple[int, int, int, int, int, int]:
    ids = int(state["ids"])
    ide = int(state["ide"])
    jds = int(state["jds"])
    jde = int(state["jde"])
    kds = int(state["kds"])
    kde = int(state["kde"])
    its0 = int(state.get("its", state.get("ips", ids)))
    ite0 = int(state.get("ite", state.get("ipe", ide)))
    jts0 = int(state.get("jts", state.get("jps", jds)))
    jte0 = int(state.get("jte", state.get("jpe", jde)))
    sz = int(_cfg_value(state, "spec_zone", 1)) if skip_spec_zone and bool(_cfg_value(state, "specified", False)) else 0
    channel = bool(_cfg_value(state, "periodic_x", False)) and bool(_cfg_value(state, "specified", False))
    its = max(its0, ids if channel else ids + sz)
    ite = min(ite0, ide - 1 if channel else ide - 1 - sz)
    jts = max(jts0, jds + sz)
    jte = min(jte0, jde - 1 - sz)
    kts = int(state.get("kts", state.get("kps", kds)))
    kte = min(int(state.get("kte", state.get("kpe", kde))), kde - 1)
    return its, ite, jts, jte, kts, kte


def _with_fortran_case_dir(state: Mapping[str, Any]) -> Callable[[], None]:
    case_dir = state.get("_case_dir")
    old_dir = os.getcwd()
    if case_dir:
        os.chdir(str(case_dir))

    def _restore() -> None:
        os.chdir(old_dir)

    return _restore


def _start_datetime_from_state(state: Mapping[str, Any]) -> datetime:
    case_config = state.get("_case_config")
    if case_config is not None and getattr(case_config, "start_time", None):
        return datetime.strptime(str(case_config.start_time), "%Y-%m-%d_%H:%M:%S")
    start = getattr(state.get("config_flags", None), "start_time", None)
    if start is not None:
        return datetime.strptime(str(start), "%Y-%m-%d_%H:%M:%S")
    return datetime(2021, 1, 1)


def _radiation_time_values(state: Mapping[str, Any]) -> tuple[int, float, float, float]:
    start = _start_datetime_from_state(state)
    elapsed = float(state.get("curr_secs", getattr(state["grid"], "current_time", 0.0)))
    current = start + timedelta(seconds=elapsed)
    year_start = datetime(current.year, 1, 1)
    julian = (current - year_start).total_seconds() / 86400.0
    gmt = float(start.hour) + float(start.minute) / 60.0 + float(start.second) / 3600.0
    xtime = elapsed / 60.0
    return int(current.year), julian, gmt, xtime


def _radiation_greenhouse_scalars(
    case_dir: str | None,
    yr: int,
    julian: float,
    ghg_input: int,
) -> tuple[np.float32, np.float32, np.float32, np.float32, np.float32]:
    if int(ghg_input) == 1 and case_dir is not None:
        table = load_camgases_table(Path(case_dir) / "CAMtr_volume_mixing_ratio")
        gas = read_camgases(
            int(yr),
            float(julian),
            False,
            "RRTMG",
            state=table,
        )
        return (
            np.float32(gas.co2vmr),
            np.float32(gas.n2ovmr),
            np.float32(gas.ch4vmr),
            np.float32(gas.cfc11vmr),
            np.float32(gas.cfc12vmr),
        )
    if int(ghg_input) == 0:
        co2 = (280.0 + 90.0 * np.exp(0.02 * (int(yr) - 2000))) * 1.0e-6
        return (
            np.float32(co2),
            np.float32(319.0e-9),
            np.float32(1774.0e-9),
            np.float32(0.251e-9),
            np.float32(0.538e-9),
        )
    return (
        np.float32(379.0e-6),
        np.float32(319.0e-9),
        np.float32(1774.0e-9),
        np.float32(0.251e-9),
        np.float32(0.538e-9),
    )


def _radconst(julian: float) -> tuple[float, float]:
    real = np.float32
    degrad = real(np.pi / 180.0)
    dpd = real(real(360.0) / real(365.0))
    julian_real = real(julian)

    obecl = real(real(23.5) * degrad)
    sinob = real(np.sin(obecl))
    if julian_real >= real(80.0):
        sxlong = real(dpd * real(julian_real - real(80.0)))
    else:
        sxlong = real(dpd * real(julian_real + real(285.0)))
    sxlong = real(sxlong * degrad)
    declin = real(np.arcsin(real(sinob * real(np.sin(sxlong)))))

    djul = real(real(julian_real * real(360.0)) / real(365.0))
    rjul = real(djul * degrad)
    twice_rjul = real(real(2.0) * rjul)
    eccfac = real(1.000110)
    eccfac = real(eccfac + real(real(0.034221) * real(np.cos(rjul))))
    eccfac = real(eccfac + real(real(0.001280) * real(np.sin(rjul))))
    eccfac = real(eccfac + real(real(0.000719) * real(np.cos(twice_rjul))))
    eccfac = real(eccfac + real(real(0.000077) * real(np.sin(twice_rjul))))
    return float(declin), float(real(real(1370.0) * eccfac))


_CAM_OZONE_CACHE: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_cam_ozone_tables(case_dir: str | Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = Path(case_dir) if case_dir is not None else Path.cwd()
    key = base.resolve()
    cached = _CAM_OZONE_CACHE.get(key)
    if cached is not None:
        return cached

    plev = np.loadtxt(base / "ozone_plev.formatted", dtype=np.float32) * np.float32(100.0)
    lat = np.loadtxt(base / "ozone_lat.formatted", dtype=np.float32)
    raw = np.loadtxt(base / "ozone.formatted", dtype=np.float32)
    levsiz = int(plev.shape[0])
    latsiz = int(lat.shape[0])
    expected = 12 * latsiz * levsiz
    if int(raw.size) != expected:
        raise ValueError(f"ozone.formatted has {raw.size} values, expected {expected}")
    ozone = raw.reshape((12, latsiz, levsiz))
    cached = (plev, lat, ozone)
    _CAM_OZONE_CACHE[key] = cached
    return cached


def _cam_ozone_month_weights(julian: float) -> tuple[int, int, float, float]:
    date_oz = np.asarray([16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350], dtype=np.float32)
    daysperyear = np.float32(365.0)
    real = np.float32
    int_julian = real(real(julian) + real(1.0))
    ijul = int(int_julian)
    frac = real(int_julian - real(ijul))
    ijul = ijul % 365
    if ijul == 0:
        ijul = 365
    int_julian = real(frac + real(ijul))
    future = np.nonzero(date_oz > int_julian)[0]
    np_idx = int(future[0]) if future.size else 0
    if np_idx > 0:
        nm_idx = np_idx - 1
        cdayozm = date_oz[nm_idx]
        cdayozp = date_oz[np_idx]
        deltat = real(cdayozp - cdayozm)
        fact1 = real(real(cdayozp - int_julian) / deltat)
        fact2 = real(real(int_julian - cdayozm) / deltat)
    else:
        nm_idx = 11
        cdayozm = date_oz[nm_idx]
        cdayozp = date_oz[0]
        deltat = real(real(cdayozp + daysperyear) - cdayozm)
        if int_julian > cdayozp:
            fact1 = real(real(real(cdayozp + daysperyear) - int_julian) / deltat)
            fact2 = real(real(int_julian - cdayozm) / deltat)
        else:
            fact1 = real(real(cdayozp - int_julian) / deltat)
            fact2 = real(real(real(int_julian + daysperyear) - cdayozm) / deltat)
    return nm_idx, np_idx, float(fact1), float(fact2)


def _cam_o3rad_np(xlat: Any, p3d: Any, *, julian: float, case_dir: str | Path | None) -> np.ndarray:
    """Return WRF ``O3RAD`` for ``o3input=2`` using CAM ozone tables."""

    plev, lat_ozone, ozone = _load_cam_ozone_tables(case_dir)
    nm_idx, np_idx, fact1, fact2 = _cam_ozone_month_weights(julian)

    lat_grid = np.asarray(xlat, dtype=np.float32)
    p = np.asarray(p3d, dtype=np.float32)
    nx, nk, ny = p.shape
    levsiz = int(plev.shape[0])
    flat_lat = lat_grid.reshape(-1)
    lat_idx = np.searchsorted(lat_ozone, flat_lat, side="left") - 1
    lat_idx = np.clip(lat_idx, 0, int(lat_ozone.size) - 2)
    lat0 = lat_ozone[lat_idx]
    lat1 = lat_ozone[lat_idx + 1]
    lat_delta = (flat_lat - lat0).astype(np.float32)
    lat_span = (lat1 - lat0).astype(np.float32)

    # WRF's order is monthly latitude interpolation in ``oznini`` followed
    # by time interpolation in ``ozn_time_int``.  These linear operations do
    # not commute once every REAL statement is rounded to binary32.
    monthly_at_lat: list[np.ndarray] = []
    for month_idx in (nm_idx, np_idx):
        lower = ozone[month_idx, lat_idx, :]
        upper = ozone[month_idx, lat_idx + 1, :]
        slope = ((upper - lower).astype(np.float32) / lat_span[:, None]).astype(np.float32)
        offset = (slope * lat_delta[:, None]).astype(np.float32)
        monthly_at_lat.append((lower + offset).astype(np.float32))
    time_left = (monthly_at_lat[0] * np.float32(fact1)).astype(np.float32)
    time_right = (monthly_at_lat[1] * np.float32(fact2)).astype(np.float32)
    oz_flat = (time_left + time_right).astype(np.float32)
    oz_xy_lev = oz_flat.reshape(nx, ny, levsiz)

    idx = np.searchsorted(plev, p, side="left") - 1
    idx = np.clip(idx, 0, levsiz - 2)
    idx_xyk = np.transpose(idx, (0, 2, 1))
    oz0 = np.take_along_axis(oz_xy_lev, idx_xyk, axis=2)
    oz1 = np.take_along_axis(oz_xy_lev, idx_xyk + 1, axis=2)
    p_xyk = np.transpose(p, (0, 2, 1))
    pin0 = plev[idx_xyk]
    pin1 = plev[idx_xyk + 1]
    dpu = (p_xyk - pin0).astype(np.float32)
    dpl = (pin1 - p_xyk).astype(np.float32)
    lower_term = (oz0 * dpl).astype(np.float32)
    upper_term = (oz1 * dpu).astype(np.float32)
    numerator = (lower_term + upper_term).astype(np.float32)
    denominator = (dpl + dpu).astype(np.float32)
    out_xyk = (numerator / denominator).astype(np.float32)

    top_mask = p_xyk < plev[0]
    if np.any(top_mask):
        top_scaled = (oz_xy_lev[:, :, :1] * p_xyk).astype(np.float32)
        top_scaled = (top_scaled / plev[0]).astype(np.float32)
        out_xyk = np.where(top_mask, top_scaled, out_xyk)
    bottom_mask = p_xyk > plev[-1]
    if np.any(bottom_mask):
        out_xyk = np.where(bottom_mask, oz_xy_lev[:, :, -1:], out_xyk)
    return np.asfortranarray(np.transpose(out_xyk, (0, 2, 1)).astype(np.float32))


def _cam_o3rad_jax(
    xlat: Any,
    p3d: Any,
    *,
    julian: float,
    case_dir: str | Path | None,
    rounding_token: Any | None = None,
) -> Any:
    """Prepare calendar/table inputs for device-resident WRF ozone interpolation."""

    plev_np, lat_ozone_np, ozone_np = _load_cam_ozone_tables(case_dir)
    nm_idx, np_idx, fact1, fact2 = _cam_ozone_month_weights(julian)
    return _cam_o3rad_arrays(
        xlat, p3d, plev_np, lat_ozone_np, ozone_np,
        jnp.asarray((nm_idx, np_idx), dtype=jnp.int32),
        jnp.asarray((fact1, fact2), dtype=jnp.float32),
        jnp.asarray(0 if rounding_token is None else rounding_token, dtype=jnp.uint32),
    )


@jax.jit
def _cam_o3rad_arrays(
    xlat: Any, p3d: Any, plev: Any, lat_ozone: Any, ozone: Any,
    month_indices: Any, month_weights: Any, rounding_token: Any,
) -> Any:
    """Interpolate ozone without copying pressure/latitude through the host.

    Calendar weights and month indices are dynamic inputs, so hourly updates
    reuse one executable. All REAL stores retain the WRF operation order.
    """

    plev = jnp.asarray(plev, dtype=jnp.float32)
    lat_ozone = jnp.asarray(lat_ozone, dtype=jnp.float32)
    ozone = jnp.asarray(ozone, dtype=jnp.float32)
    lat_grid = jnp.asarray(xlat, dtype=jnp.float32)
    p = jnp.asarray(p3d, dtype=jnp.float32)
    nx, nk, ny = p.shape
    levsiz = int(plev.shape[0])

    def store(value: Any) -> Any:
        return _materialize_wrf_real(value, rounding_token)

    flat_lat = lat_grid.reshape((-1,))
    lat_idx = jnp.searchsorted(lat_ozone, flat_lat, side="left") - 1
    lat_idx = jnp.clip(lat_idx, 0, int(lat_ozone.size) - 2).astype(jnp.int32)
    lat0 = jnp.take(lat_ozone, lat_idx)
    lat1 = jnp.take(lat_ozone, lat_idx + 1)
    lat_delta = store(flat_lat - lat0)
    lat_span = store(lat1 - lat0)

    monthly_at_lat = []
    for month_idx in (month_indices[0], month_indices[1]):
        lower = ozone[month_idx, lat_idx, :]
        upper = ozone[month_idx, lat_idx + 1, :]
        slope = store(store(upper - lower) / lat_span[:, None])
        offset = store(slope * lat_delta[:, None])
        monthly_at_lat.append(store(lower + offset))
    time_left = store(monthly_at_lat[0] * month_weights[0])
    time_right = store(monthly_at_lat[1] * month_weights[1])
    oz_flat = store(time_left + time_right)
    oz_xy_lev = oz_flat.reshape((nx, ny, levsiz))

    idx = jnp.searchsorted(plev, p, side="left") - 1
    idx = jnp.clip(idx, 0, levsiz - 2).astype(jnp.int32)
    idx_xyk = jnp.transpose(idx, (0, 2, 1))
    oz0 = jnp.take_along_axis(oz_xy_lev, idx_xyk, axis=2)
    oz1 = jnp.take_along_axis(oz_xy_lev, idx_xyk + 1, axis=2)
    p_xyk = jnp.transpose(p, (0, 2, 1))
    pin0 = jnp.take(plev, idx_xyk)
    pin1 = jnp.take(plev, idx_xyk + 1)
    dpu = store(p_xyk - pin0)
    dpl = store(pin1 - p_xyk)
    lower_term = store(oz0 * dpl)
    upper_term = store(oz1 * dpu)
    numerator = store(lower_term + upper_term)
    denominator = store(dpl + dpu)
    out_xyk = store(numerator / denominator)

    top_scaled = store(oz_xy_lev[:, :, :1] * p_xyk)
    top_scaled = store(top_scaled / plev[0])
    out_xyk = jnp.where(p_xyk < plev[0], top_scaled, out_xyk)
    out_xyk = jnp.where(p_xyk > plev[-1], oz_xy_lev[:, :, -1:], out_xyk)
    return jnp.transpose(out_xyk, (0, 2, 1)).astype(jnp.float32)


def _cal_cldfra1_np(qv: Any, qc: Any, qi: Any, qs: Any, t_phy: Any, p_phy: Any) -> np.ndarray:
    """Vectorized WRF ``cal_cldfra1`` for active QV/QC/QI/QS fields."""

    qv_arr = np.asarray(qv, dtype=np.float32)
    qc_arr = np.asarray(qc, dtype=np.float32)
    qi_arr = np.asarray(qi, dtype=np.float32)
    qs_arr = np.asarray(qs, dtype=np.float32)
    t_arr = np.asarray(t_phy, dtype=np.float32)
    p_arr = np.asarray(p_phy, dtype=np.float32)

    arrays = np.broadcast_arrays(qv_arr, qc_arr, qi_arr, qs_arr, t_arr, p_arr)
    qv_arr, qc_arr, qi_arr, qs_arr, t_arr, p_arr = arrays
    active = ((t_arr > np.float32(100.0)) & (p_arr > np.float32(100.0))
              & ((qi_arr + qc_arr + qs_arr) >= np.float32(1.0e-12)))
    if not np.any(active):
        return np.asfortranarray(np.zeros(active.shape, dtype=np.float32))
    compact = not np.all(active)
    if compact:
        # The final WRF mask sets these other layers to exactly zero. Keep the
        # same NumPy formulas and rounding for every potentially nonzero layer.
        active_indices = np.flatnonzero(active)
        qv_arr, qc_arr, qi_arr, qs_arr, t_arr, p_arr = (value.flat[active_indices] for value in arrays)

    alpha0 = np.float32(100.0)
    gamma = np.float32(0.49)
    qcldmin = np.float32(1.0e-12)
    pexp = np.float32(0.25)
    rhgrid = np.float32(1.0)
    svp1 = np.float32(0.61078)
    svp2 = np.float32(17.2693882)
    svpi2 = np.float32(21.8745584)
    svp3 = np.float32(35.86)
    svpi3 = np.float32(7.66)
    svpt0 = np.float32(273.15)
    ep_2 = np.float32(287.0 / 461.6)

    valid = (t_arr > np.float32(100.0)) & (p_arr > np.float32(100.0))
    t_safe = np.where(valid, t_arr, svpt0)
    tc = t_safe - svpt0
    esw = np.float32(1000.0) * svp1 * np.exp(svp2 * tc / (t_safe - svp3)).astype(np.float32)
    esi = np.float32(1000.0) * svp1 * np.exp(svpi2 * tc / (t_safe - svpi3)).astype(np.float32)
    esw = np.where(valid, esw, np.float32(0.0))
    esi = np.where(valid, esi, np.float32(0.0))
    qvsw = np.zeros_like(p_arr, dtype=np.float32)
    qvsi = np.zeros_like(p_arr, dtype=np.float32)
    np.divide(ep_2 * esw, p_arr - esw, out=qvsw, where=valid & ((p_arr - esw) > np.float32(1.0)))
    np.divide(ep_2 * esi, p_arr - esi, out=qvsi, where=valid & ((p_arr - esi) > np.float32(1.0)))

    qcld = qi_arr + qc_arr + qs_arr
    weight = np.zeros_like(qcld, dtype=np.float32)
    np.divide(qi_arr + qs_arr, qcld, out=weight, where=qcld >= qcldmin)
    qvs_weight = (np.float32(1.0) - weight) * qvsw + weight * qvsi
    rhum = np.zeros_like(qv_arr, dtype=np.float32)
    np.divide(qv_arr, qvs_weight, out=rhum, where=valid & (qvs_weight > np.float32(0.0)))

    subsat = np.maximum(np.float32(1.0e-10), rhgrid * qvs_weight - qv_arr)
    denom = subsat**gamma
    arg = np.maximum(np.float32(-6.9), -alpha0 * qcld / denom)
    rhum_pos = np.maximum(np.float32(1.0e-10), rhum)
    partial = (rhum_pos / rhgrid) ** pexp * (np.float32(1.0) - np.exp(arg).astype(np.float32))
    partial = np.where(partial < np.float32(0.01), np.float32(0.0), partial)
    cldfra = np.where(valid & (qcld >= qcldmin), np.where(rhum >= rhgrid, np.float32(1.0), partial), np.float32(0.0))
    if compact:
        full = np.zeros(active.shape, dtype=np.float32, order="F")
        full.flat[active_indices] = cldfra
        return np.asfortranarray(full)
    return np.asfortranarray(cldfra.astype(np.float32))


def _cal_cldfra1_jax(qv: Any, qc: Any, qi: Any, qs: Any, t_phy: Any, p_phy: Any) -> Any:
    """JAX-compatible WRF ``cal_cldfra1`` for active QV/QC/QI/QS fields."""

    qv_arr = jnp.asarray(qv, dtype=jnp.float32)
    qc_arr = jnp.asarray(qc, dtype=jnp.float32)
    qi_arr = jnp.asarray(qi, dtype=jnp.float32)
    qs_arr = jnp.asarray(qs, dtype=jnp.float32)
    t_arr = jnp.asarray(t_phy, dtype=jnp.float32)
    p_arr = jnp.asarray(p_phy, dtype=jnp.float32)

    alpha0 = jnp.float32(100.0)
    gamma = jnp.float32(0.49)
    qcldmin = jnp.float32(1.0e-12)
    pexp = jnp.float32(0.25)
    rhgrid = jnp.float32(1.0)
    svp1 = jnp.float32(0.61078)
    svp2 = jnp.float32(17.2693882)
    svpi2 = jnp.float32(21.8745584)
    svp3 = jnp.float32(35.86)
    svpi3 = jnp.float32(7.66)
    svpt0 = jnp.float32(273.15)
    ep_2 = jnp.float32(287.0 / 461.6)

    valid = (t_arr > jnp.float32(100.0)) & (p_arr > jnp.float32(100.0))
    t_safe = jnp.where(valid, t_arr, svpt0)
    tc = t_safe - svpt0
    esw = jnp.float32(1000.0) * svp1 * jnp.exp(svp2 * tc / (t_safe - svp3))
    esi = jnp.float32(1000.0) * svp1 * jnp.exp(svpi2 * tc / (t_safe - svpi3))
    esw = jnp.where(valid, esw, jnp.float32(0.0))
    esi = jnp.where(valid, esi, jnp.float32(0.0))
    water_valid = valid & ((p_arr - esw) > jnp.float32(1.0))
    ice_valid = valid & ((p_arr - esi) > jnp.float32(1.0))
    qvsw = jnp.where(water_valid, ep_2 * esw / jnp.where(water_valid, p_arr - esw, 1.0), jnp.float32(0.0))
    qvsi = jnp.where(ice_valid, ep_2 * esi / jnp.where(ice_valid, p_arr - esi, 1.0), jnp.float32(0.0))

    qcld = qi_arr + qc_arr + qs_arr
    cloudy = qcld >= qcldmin
    weight = jnp.where(cloudy, (qi_arr + qs_arr) / jnp.where(cloudy, qcld, 1.0), jnp.float32(0.0))
    qvs_weight = (jnp.float32(1.0) - weight) * qvsw + weight * qvsi
    humidity_valid = valid & (qvs_weight > jnp.float32(0.0))
    rhum = jnp.where(humidity_valid, qv_arr / jnp.where(humidity_valid, qvs_weight, 1.0), jnp.float32(0.0))

    subsat = jnp.maximum(jnp.float32(1.0e-10), rhgrid * qvs_weight - qv_arr)
    denom = subsat**gamma
    arg = jnp.maximum(jnp.float32(-6.9), -alpha0 * qcld / denom)
    rhum_pos = jnp.maximum(jnp.float32(1.0e-10), rhum)
    partial = (rhum_pos / rhgrid) ** pexp * (jnp.float32(1.0) - jnp.exp(arg))
    partial = jnp.where(partial < jnp.float32(0.01), jnp.float32(0.0), partial)
    return jnp.where(valid & (qcld >= qcldmin), jnp.where(rhum >= rhgrid, jnp.float32(1.0), partial), jnp.float32(0.0)).astype(jnp.float32)


def _svml_float32_from_bits_np(bits: int) -> np.float32:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)[()]


def _svml_xor_float32_np(values: Any, bits: Any) -> np.ndarray:
    return np.bitwise_xor(
        np.asarray(values, dtype=np.float32).view(np.uint32),
        np.asarray(bits, dtype=np.uint32),
    ).view(np.float32)


def _svml_sinf_np(values: Any) -> np.ndarray:
    real = np.float32
    source = np.asarray(values, dtype=np.float32)
    source_bits = source.view(np.uint32)
    sign = np.bitwise_and(source_bits, np.uint32(0x80000000))
    reduced = np.bitwise_and(source_bits, np.uint32(0x7FFFFFFF)).view(np.float32)

    y = real(reduced * _svml_float32_from_bits_np(0x3EA2F983))
    y = real(y + _svml_float32_from_bits_np(0x4B400000))
    octant_sign = np.left_shift(y.view(np.uint32), np.uint32(31))
    y = real(y - _svml_float32_from_bits_np(0x4B400000))

    x = real(reduced - real(y * _svml_float32_from_bits_np(0x40490000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x3A7DA000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x34222000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x2CB4611A)))
    x2 = real(x * x)
    signed_x = _svml_xor_float32_np(x, octant_sign)

    polynomial = real(_svml_float32_from_bits_np(0x362EDEF8) * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0xB94FB7FF))
    polynomial = real(polynomial * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0x3C08876A))
    polynomial = real(polynomial * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0xBE2AAAA6))
    correction = real(x2 * polynomial)
    correction = real(signed_x * correction)
    result = real(signed_x + correction)
    result = _svml_xor_float32_np(result, sign)
    return np.where(
        np.abs(source) > _svml_float32_from_bits_np(0x461C4000),
        np.sin(source),
        result,
    ).astype(np.float32)


def _svml_cosf_np(values: Any) -> np.ndarray:
    real = np.float32
    source = np.asarray(values, dtype=np.float32)

    y = real(_svml_float32_from_bits_np(0x3FC90FDB) + source)
    y = real(y * _svml_float32_from_bits_np(0x3EA2F983))
    y = real(y + _svml_float32_from_bits_np(0x4B400000))
    octant_sign = np.left_shift(y.view(np.uint32), np.uint32(31))
    y = real(y - _svml_float32_from_bits_np(0x4B400000))
    y = real(y - _svml_float32_from_bits_np(0x3F000000))

    x = real(source - real(y * _svml_float32_from_bits_np(0x40490000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x3A7DA000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x34222000)))
    x = real(x - real(y * _svml_float32_from_bits_np(0x2CB4611A)))
    x2 = real(x * x)
    signed_x = _svml_xor_float32_np(x, octant_sign)

    polynomial = real(_svml_float32_from_bits_np(0x362EDEF8) * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0xB94FB7FF))
    polynomial = real(polynomial * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0x3C08876A))
    polynomial = real(polynomial * x2)
    polynomial = real(polynomial + _svml_float32_from_bits_np(0xBE2AAAA6))
    correction = real(x2 * polynomial)
    correction = real(signed_x * correction)
    result = real(signed_x + correction)
    return np.where(
        np.abs(source) > _svml_float32_from_bits_np(0x461C4000),
        np.cos(source),
        result,
    ).astype(np.float32)


@lru_cache(maxsize=1)
def _wrf_scalar_libm() -> tuple[Any, Any]:
    library = ctypes.CDLL(ctypes.util.find_library("m") or None)
    sinf = library.sinf
    sinf.argtypes = (ctypes.c_float,)
    sinf.restype = ctypes.c_float
    cosf = library.cosf
    cosf.argtypes = (ctypes.c_float,)
    cosf.restype = ctypes.c_float
    return sinf, cosf


def _wrf_scalar_trig_np(values: Any, *, cosine: bool) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    sinf, cosf = _wrf_scalar_libm()
    function = cosf if cosine else sinf
    result = np.fromiter(
        (function(float(value)) for value in source.flat),
        dtype=np.float32,
        count=source.size,
    )
    return result.reshape(source.shape)


def _svml_float32_from_bits_jax(bits: int) -> Any:
    return jax.lax.bitcast_convert_type(jnp.asarray(bits, dtype=jnp.uint32), jnp.float32)


def _svml_xor_float32_jax(values: Any, bits: Any) -> Any:
    value_bits = jax.lax.bitcast_convert_type(
        jnp.asarray(values, dtype=jnp.float32), jnp.uint32
    )
    result_bits = jnp.bitwise_xor(value_bits, jnp.asarray(bits, dtype=jnp.uint32))
    return jax.lax.bitcast_convert_type(result_bits, jnp.float32)


def _svml_sinf_jax(values: Any) -> Any:
    store = lambda value: jax.lax.optimization_barrier(
        jnp.asarray(value, dtype=jnp.float32)
    )
    source = store(values)
    source_bits = jax.lax.bitcast_convert_type(source, jnp.uint32)
    sign = jnp.bitwise_and(source_bits, jnp.uint32(0x80000000))
    reduced_bits = jnp.bitwise_and(source_bits, jnp.uint32(0x7FFFFFFF))
    reduced = jax.lax.bitcast_convert_type(reduced_bits, jnp.float32)

    y = store(reduced * _svml_float32_from_bits_jax(0x3EA2F983))
    y = store(y + _svml_float32_from_bits_jax(0x4B400000))
    y_bits = jax.lax.bitcast_convert_type(y, jnp.uint32)
    octant_sign = jnp.left_shift(y_bits, jnp.uint32(31))
    y = store(y - _svml_float32_from_bits_jax(0x4B400000))

    x = store(reduced - store(y * _svml_float32_from_bits_jax(0x40490000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x3A7DA000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x34222000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x2CB4611A)))
    x2 = store(x * x)
    signed_x = _svml_xor_float32_jax(x, octant_sign)

    polynomial = store(_svml_float32_from_bits_jax(0x362EDEF8) * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0xB94FB7FF))
    polynomial = store(polynomial * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0x3C08876A))
    polynomial = store(polynomial * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0xBE2AAAA6))
    correction = store(x2 * polynomial)
    correction = store(signed_x * correction)
    result = store(signed_x + correction)
    result = _svml_xor_float32_jax(result, sign)
    return jnp.where(
        jnp.abs(source) > _svml_float32_from_bits_jax(0x461C4000),
        jnp.sin(source),
        result,
    )


def _svml_cosf_jax(values: Any) -> Any:
    store = lambda value: jax.lax.optimization_barrier(
        jnp.asarray(value, dtype=jnp.float32)
    )
    source = store(values)

    y = store(_svml_float32_from_bits_jax(0x3FC90FDB) + source)
    y = store(y * _svml_float32_from_bits_jax(0x3EA2F983))
    y = store(y + _svml_float32_from_bits_jax(0x4B400000))
    y_bits = jax.lax.bitcast_convert_type(y, jnp.uint32)
    octant_sign = jnp.left_shift(y_bits, jnp.uint32(31))
    y = store(y - _svml_float32_from_bits_jax(0x4B400000))
    y = store(y - _svml_float32_from_bits_jax(0x3F000000))

    x = store(source - store(y * _svml_float32_from_bits_jax(0x40490000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x3A7DA000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x34222000)))
    x = store(x - store(y * _svml_float32_from_bits_jax(0x2CB4611A)))
    x2 = store(x * x)
    signed_x = _svml_xor_float32_jax(x, octant_sign)

    polynomial = store(_svml_float32_from_bits_jax(0x362EDEF8) * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0xB94FB7FF))
    polynomial = store(polynomial * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0x3C08876A))
    polynomial = store(polynomial * x2)
    polynomial = store(polynomial + _svml_float32_from_bits_jax(0xBE2AAAA6))
    correction = store(x2 * polynomial)
    correction = store(signed_x * correction)
    result = store(signed_x + correction)
    return jnp.where(
        jnp.abs(source) > _svml_float32_from_bits_jax(0x461C4000),
        jnp.cos(source),
        result,
    )


def _calc_coszen_np(xlat: Any, xlong: Any, *, julian: float, xtime: float, gmt: float, declin: float) -> np.ndarray:
    real = np.float32
    degrad = real(np.pi / 180.0)
    lat = np.asarray(xlat, dtype=np.float32)
    lon = np.asarray(xlong, dtype=np.float32)
    julian_real = real(julian)

    da = real(real(real(6.2831853071795862) * real(julian_real - real(1.0))) / real(365.0))
    twice_da = real(real(2.0) * da)
    eot = real(0.000075)
    eot = real(eot + real(real(0.001868) * real(np.cos(da))))
    eot = real(eot - real(real(0.032077) * real(np.sin(da))))
    eot = real(eot - real(real(0.014615) * real(np.cos(twice_da))))
    eot = real(eot - real(real(0.04089) * real(np.sin(twice_da))))
    eot = real(eot * real(229.18))
    xt24 = real(real(np.fmod(real(xtime), real(1440.0))) + eot)

    tloctm = real(real(gmt) + real(xt24 / real(60.0)))
    tloctm = real(tloctm + real(lon / real(15.0)))
    hrang = real(real(real(15.0) * real(tloctm - real(12.0))) * degrad)
    xxlat = real(lat * degrad)
    vector_stop = (xxlat.shape[0] // 4) * 4
    sin_xxlat = _svml_sinf_np(xxlat)
    cos_xxlat = _svml_cosf_np(xxlat)
    cos_hrang = _svml_cosf_np(hrang)
    if vector_stop < xxlat.shape[0]:
        tail = slice(vector_stop, None)
        sin_xxlat[tail] = _wrf_scalar_trig_np(xxlat[tail], cosine=False)
        cos_xxlat[tail] = _wrf_scalar_trig_np(xxlat[tail], cosine=True)
        cos_hrang[tail] = _wrf_scalar_trig_np(hrang[tail], cosine=True)
    sin_term = real(real(sin_xxlat) * real(np.sin(real(declin))))
    cos_term = real(real(cos_xxlat) * real(np.cos(real(declin))))
    cos_term = real(cos_term * real(cos_hrang))
    coszen = real(sin_term + cos_term)
    return np.asfortranarray(np.minimum(np.maximum(coszen, real(-1.0)), real(1.0)))


def _calc_coszen_jax(xlat: Any, xlong: Any, *, julian: float, xtime: float, gmt: float, declin: float) -> Any:
    store = lambda value: jax.lax.optimization_barrier(jnp.asarray(value, dtype=jnp.float32))
    degrad = store(np.pi / 180.0)
    lat = store(xlat)
    lon = store(xlong)
    julian_real = store(julian)

    da = store(store(store(6.2831853071795862) * store(julian_real - store(1.0))) / store(365.0))
    twice_da = store(store(2.0) * da)
    eot = store(0.000075)
    eot = store(eot + store(store(0.001868) * store(jnp.cos(da))))
    eot = store(eot - store(store(0.032077) * store(jnp.sin(da))))
    eot = store(eot - store(store(0.014615) * store(jnp.cos(twice_da))))
    eot = store(eot - store(store(0.04089) * store(jnp.sin(twice_da))))
    eot = store(eot * store(229.18))
    xt24 = store(store(jnp.mod(store(xtime), store(1440.0))) + eot)

    tloctm = store(store(gmt) + store(xt24 / store(60.0)))
    tloctm = store(tloctm + store(lon / store(15.0)))
    hrang = store(store(store(15.0) * store(tloctm - store(12.0))) * degrad)
    xxlat = store(lat * degrad)
    declin_arr = store(declin)
    vector_stop = (xxlat.shape[0] // 4) * 4
    vector_mask = jnp.arange(xxlat.shape[0]) < vector_stop
    vector_mask = jnp.reshape(
        vector_mask, (xxlat.shape[0],) + (1,) * (xxlat.ndim - 1)
    )
    sin_xxlat = store(
        jnp.where(vector_mask, _svml_sinf_jax(xxlat), store(jnp.sin(xxlat)))
    )
    cos_xxlat = store(
        jnp.where(vector_mask, _svml_cosf_jax(xxlat), store(jnp.cos(xxlat)))
    )
    cos_hrang = store(
        jnp.where(vector_mask, _svml_cosf_jax(hrang), store(jnp.cos(hrang)))
    )
    sin_term = store(sin_xxlat * store(jnp.sin(declin_arr)))
    cos_term = store(cos_xxlat * store(jnp.cos(declin_arr)))
    cos_term = store(cos_term * cos_hrang)
    coszen = store(sin_term + cos_term)
    return jnp.minimum(jnp.maximum(coszen, store(-1.0)), store(1.0))


def _cfg_value(state: Mapping[str, Any], name: str, default: Any) -> Any:
    config = state["config_flags"]
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _boundary_params(state: Mapping[str, Any]) -> tuple[int, int, int]:
    width = int(_cfg_value(state, "spec_bdy_width", state.get("spec_bdy_width", 5)))
    spec_zone = int(_cfg_value(state, "spec_zone", state.get("spec_zone", 1)))
    relax_zone = int(_cfg_value(state, "relax_zone", state.get("relax_zone", width - spec_zone)))
    return width, spec_zone, relax_zone


def _static_spec_bdytend_mask(
    fields: Mapping[str, Any],
    config_flags: Any,
    *,
    bxs: Any,
    bxe: Any,
    bys: Any,
    bye: Any,
    btxs: Any,
    btxe: Any,
    btys: Any,
    btye: Any,
    variable: str,
    kds: int | None = None,
    kde: int | None = None,
    kms: int | None = None,
    kme: int | None = None,
    kps: int | None = None,
    kpe: int | None = None,
    kts: int | None = None,
    kte: int | None = None,
) -> Any:
    width = int(_cfg_value({"config_flags": config_flags, **fields}, "spec_bdy_width", fields.get("spec_bdy_width", 5)))
    spec_zone = int(_cfg_value({"config_flags": config_flags, **fields}, "spec_zone", fields.get("spec_zone", 1)))
    values = spec_bdytend(
        bxs,
        bxe,
        bys,
        bye,
        jnp.ones_like(jnp.asarray(btxs)),
        jnp.ones_like(jnp.asarray(btxe)),
        jnp.ones_like(jnp.asarray(btys)),
        jnp.ones_like(jnp.asarray(btye)),
        variable,
        config_flags,
        width,
        spec_zone,
        fields["ids"],
        fields["ide"],
        fields["jds"],
        fields["jde"],
        fields["kds"] if kds is None else kds,
        fields["kde"] if kde is None else kde,
        fields["ims"],
        fields["ime"],
        fields["jms"],
        fields["jme"],
        fields["kms"] if kms is None else kms,
        fields["kme"] if kme is None else kme,
        fields["ips"],
        fields["ipe"],
        fields["jps"],
        fields["jpe"],
        fields["kps"] if kps is None else kps,
        fields["kpe"] if kpe is None else kpe,
        fields.get("its", fields["ips"]),
        fields.get("ite", fields["ipe"]),
        fields.get("jts", fields["jps"]),
        fields.get("jte", fields["jpe"]),
        fields.get("kts", fields["kps"]) if kts is None else kts,
        fields.get("kte", fields["kpe"]) if kte is None else kte,
    )
    return jnp.asarray(values) != 0


def _with_static_dry_boundary_masks(fields: Mapping[str, Any], config_flags: Any) -> dict[str, Any]:
    output = dict(fields)
    required = (
        "u_bxs",
        "u_bxe",
        "u_bys",
        "u_bye",
        "u_btxs",
        "u_btxe",
        "u_btys",
        "u_btye",
        "v_bxs",
        "v_bxe",
        "v_bys",
        "v_bye",
        "v_btxs",
        "v_btxe",
        "v_btys",
        "v_btye",
        "ph_bxs",
        "ph_bxe",
        "ph_bys",
        "ph_bye",
        "ph_btxs",
        "ph_btxe",
        "ph_btys",
        "ph_btye",
        "t_bxs",
        "t_bxe",
        "t_bys",
        "t_bye",
        "t_btxs",
        "t_btxe",
        "t_btys",
        "t_btye",
        "w_bxs",
        "w_bxe",
        "w_bys",
        "w_bye",
        "w_btxs",
        "w_btxe",
        "w_btys",
        "w_btye",
        "mu_bxs",
        "mu_bxe",
        "mu_bys",
        "mu_bye",
        "mu_btxs",
        "mu_btxe",
        "mu_btys",
        "mu_btye",
    )
    if any(name not in output for name in required):
        return output

    output["_dry_boundary_ru_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["u_bxs"],
        bxe=output["u_bxe"],
        bys=output["u_bys"],
        bye=output["u_bye"],
        btxs=output["u_btxs"],
        btxe=output["u_btxe"],
        btys=output["u_btys"],
        btye=output["u_btye"],
        variable="u",
    )
    output["_dry_boundary_rv_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["v_bxs"],
        bxe=output["v_bxe"],
        bys=output["v_bys"],
        bye=output["v_bye"],
        btxs=output["v_btxs"],
        btxe=output["v_btxe"],
        btys=output["v_btys"],
        btye=output["v_btye"],
        variable="v",
    )
    output["_dry_boundary_ph_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["ph_bxs"],
        bxe=output["ph_bxe"],
        bys=output["ph_bys"],
        bye=output["ph_bye"],
        btxs=output["ph_btxs"],
        btxe=output["ph_btxe"],
        btys=output["ph_btys"],
        btye=output["ph_btye"],
        variable="h",
    )
    output["_dry_boundary_t_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["t_bxs"],
        bxe=output["t_bxe"],
        bys=output["t_bys"],
        bye=output["t_bye"],
        btxs=output["t_btxs"],
        btxe=output["t_btxe"],
        btys=output["t_btys"],
        btye=output["t_btye"],
        variable="t",
    )
    output["_dry_boundary_rw_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["w_bxs"],
        bxe=output["w_bxe"],
        bys=output["w_bys"],
        bye=output["w_bye"],
        btxs=output["w_btxs"],
        btxe=output["w_btxe"],
        btys=output["w_btys"],
        btye=output["w_btye"],
        variable="h",
    )
    output["_dry_boundary_mu_mask"] = _static_spec_bdytend_mask(
        output,
        config_flags,
        bxs=output["mu_bxs"],
        bxe=output["mu_bxe"],
        bys=output["mu_bys"],
        bye=output["mu_bye"],
        btxs=output["mu_btxs"],
        btxe=output["mu_btxe"],
        btys=output["mu_btys"],
        btye=output["mu_btye"],
        variable="m",
        kds=1,
        kde=1,
        kms=1,
        kme=1,
        kps=1,
        kpe=1,
        kts=1,
        kte=1,
    )[:, 0, :]
    return output


def _boundary_weights(state: Mapping[str, Any]) -> tuple[Any, Any]:
    if bool(state.get("_boundary_weights_ready", False)) and "fcx" in state and "gcx" in state:
        return state["fcx"], state["gcx"]
    width, spec_zone, relax_zone = _boundary_params(state)
    fcx = _field_or(state, "fcx", getattr(state["grid"], "fcx", jnp.zeros((width,), dtype=jnp.float32)))
    gcx = _field_or(state, "gcx", getattr(state["grid"], "gcx", jnp.zeros((width,), dtype=jnp.float32)))
    return lbc_fcx_gcx(
        fcx,
        gcx,
        width,
        spec_zone,
        relax_zone,
        float(state["grid"].dt),
        float(_cfg_value(state, "spec_exp", 0.0)),
        bool(_cfg_value(state, "specified", False)),
        bool(_cfg_value(state, "nested", False)),
    )


_DRY_BOUNDARY_NAMES = (
    "u_bxs",
    "u_bxe",
    "u_bys",
    "u_bye",
    "v_bxs",
    "v_bxe",
    "v_bys",
    "v_bye",
    "ph_bxs",
    "ph_bxe",
    "ph_bys",
    "ph_bye",
    "t_bxs",
    "t_bxe",
    "t_bys",
    "t_bye",
    "w_bxs",
    "w_bxe",
    "w_bys",
    "w_bye",
    "mu_bxs",
    "mu_bxe",
    "mu_bys",
    "mu_bye",
    "u_btxs",
    "u_btxe",
    "u_btys",
    "u_btye",
    "v_btxs",
    "v_btxe",
    "v_btys",
    "v_btye",
    "ph_btxs",
    "ph_btxe",
    "ph_btys",
    "ph_btye",
    "t_btxs",
    "t_btxe",
    "t_btys",
    "t_btye",
    "w_btxs",
    "w_btxe",
    "w_btys",
    "w_btye",
    "mu_btxs",
    "mu_btxe",
    "mu_btys",
    "mu_btye",
)


def _dry_boundary_args(state: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [name for name in _DRY_BOUNDARY_NAMES if name not in state]
    if missing:
        raise MissingKernelImplementationError(f"dry boundary stage requires missing wrfbdy fields: {', '.join(missing)}")
    return tuple(state[name] for name in _DRY_BOUNDARY_NAMES)


def _spec_bdytend_mask(
    state: Mapping[str, Any],
    *,
    bxs: Any,
    bxe: Any,
    bys: Any,
    bye: Any,
    btxs: Any,
    btxe: Any,
    btys: Any,
    btye: Any,
    variable: str,
    kds: int | None = None,
    kde: int | None = None,
    kms: int | None = None,
    kme: int | None = None,
    kps: int | None = None,
    kpe: int | None = None,
    kts: int | None = None,
    kte: int | None = None,
) -> Any:
    width, spec_zone, _relax_zone = _boundary_params(state)
    ones_xs = jnp.ones_like(jnp.asarray(btxs))
    ones_xe = jnp.ones_like(jnp.asarray(btxe))
    ones_ys = jnp.ones_like(jnp.asarray(btys))
    ones_ye = jnp.ones_like(jnp.asarray(btye))
    mask_values = spec_bdytend(
        bxs,
        bxe,
        bys,
        bye,
        ones_xs,
        ones_xe,
        ones_ys,
        ones_ye,
        variable,
        _static_config(state),
        width,
        spec_zone,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"] if kds is None else kds,
        state["kde"] if kde is None else kde,
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"] if kms is None else kms,
        state["kme"] if kme is None else kme,
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"] if kps is None else kps,
        state["kpe"] if kpe is None else kpe,
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"] if kts is None else kts,
        state["kpe"] if kte is None else kte,
    )
    return jnp.asarray(mask_values) != 0


def _overlay_specified_boundary(previous: Any, specified: Any, mask: Any) -> Any:
    previous_arr = jnp.asarray(previous)
    slices = tuple(slice(0, int(dim)) for dim in previous_arr.shape)
    specified_arr = jnp.asarray(specified)[slices]
    mask_arr = jnp.asarray(mask)[slices]
    return jnp.where(mask_arr, specified_arr, previous_arr)


def _crop2_to_mass(state: Mapping[str, Any], value: Any) -> Any:
    ni, nj = _shape2(state)
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.full((ni, nj), arr, dtype=arr.dtype)
    if arr.ndim == 2 and arr.shape[0] <= ni + 1 and arr.shape[1] <= nj + 1:
        return arr[:ni, :nj]
    i0 = max(1 - int(state.get("ims", 1)), 0)
    j0 = max(1 - int(state.get("jms", 1)), 0)
    return arr[i0 : i0 + ni, j0 : j0 + nj]


def _crop3_to_mass(state: Mapping[str, Any], value: Any) -> Any:
    ni, nj = _shape2(state)
    arr = jnp.asarray(value)
    i0 = max(1 - int(state.get("ims", 1)), 0)
    j0 = max(1 - int(state.get("jms", 1)), 0)
    if arr.ndim == 0:
        template = jnp.asarray(state["t_phy"])[i0 : i0 + ni, :, j0 : j0 + nj]
        return jnp.full_like(template, arr)
    if arr.ndim == 3 and arr.shape[0] <= ni + 1 and arr.shape[2] <= nj + 1:
        return arr[:ni, :, :nj]
    return arr[i0 : i0 + ni, :, j0 : j0 + nj]


def _surface_bounds(state: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    its = int(state.get("its", state["ids"]))
    ite = min(int(state.get("ite", state["ide"])), int(state["ide"]) - 1)
    jts = int(state.get("jts", state["jds"]))
    jte = min(int(state.get("jte", state["jde"])), int(state["jde"]) - 1)
    kts = int(state.get("kps", state["kds"]))
    kte = min(int(state.get("kpe", state["kde"])), int(state["kde"]) - 1)
    return its, ite, jts, jte, kts, kte


def _with_static_gwdo_active_columns(fields: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(fields)
    var_value = output.get("var")
    if var_value is None:
        return output
    its, ite, jts, jte, _kts, _kte = _surface_bounds(output)
    i0 = int(its) - int(output["ims"])
    i1 = int(ite) - int(output["ims"]) + 1
    j0 = int(jts) - int(output["jms"])
    j1 = int(jte) - int(output["jms"]) + 1
    if i0 < 0 or j0 < 0:
        return output
    var_arr = np.asarray(jax.device_get(jnp.asarray(var_value, dtype=jnp.float32)))
    if var_arr.ndim != 2 or i1 > var_arr.shape[0] or j1 > var_arr.shape[1]:
        var_arr = np.asarray(jax.device_get(_crop2_to_mass(output, var_value)), dtype=np.float32)
        i0 = 0
        i1 = var_arr.shape[0]
        j0 = 0
        j1 = var_arr.shape[1]
    active_local_i, active_local_j = np.nonzero(var_arr[i0:i1, j0:j1] > np.float32(0.0))
    if active_local_i.size == 0:
        return output
    active_i = active_local_i.astype(np.int32, copy=False) + np.int32(i0)
    active_j = active_local_j.astype(np.int32, copy=False) + np.int32(j0)
    output["_static_gwdo_active_i"] = jnp.asarray(active_i, dtype=jnp.int32)
    output["_static_gwdo_active_j"] = jnp.asarray(active_j, dtype=jnp.int32)
    output["_static_gwdo_active_count"] = int(active_i.size)
    output["_static_gwdo_total_count"] = int((i1 - i0) * (j1 - j0))

    row_j: list[int] = []
    row_i_start: list[int] = []
    row_len: list[int] = []
    for local_j in range(j1 - j0):
        row_i = np.nonzero(var_arr[i0:i1, j0 + local_j] > np.float32(0.0))[0]
        if row_i.size == 0:
            continue
        row_j.append(j0 + local_j)
        row_i_start.append(i0 + int(row_i.min()))
        row_len.append(int(row_i.max() - row_i.min() + 1))
    if row_j:
        max_len = int(max(row_len))
        output["_static_gwdo_row_j"] = jnp.asarray(row_j, dtype=jnp.int32)
        output["_static_gwdo_row_i_start"] = jnp.asarray(row_i_start, dtype=jnp.int32)
        output["_static_gwdo_row_len"] = jnp.asarray(row_len, dtype=jnp.int32)
        output["_static_gwdo_row_max_len"] = max_len
        output["_static_gwdo_row_padded_count"] = int(len(row_j) * max_len)
    return output


def _with_static_noah_active_columns(fields: Mapping[str, Any], config_flags: Any,
                                     *, land_columns: bool = True) -> dict[str, Any]:
    """Precompute land columns that actually execute Noah LSM."""

    output = dict(fields)
    if "xland" not in fields or "ivgtyp" not in fields:
        return output
    xland = np.asarray(jax.device_get(jnp.asarray(fields["xland"], dtype=jnp.float32)), dtype=np.float32)
    xice_source = fields.get("xice", fields.get("seaice", np.zeros_like(xland, dtype=np.float32)))
    xice = np.asarray(jax.device_get(jnp.asarray(xice_source, dtype=jnp.float32)), dtype=np.float32)
    ivgtyp = np.asarray(jax.device_get(jnp.asarray(fields["ivgtyp"], dtype=jnp.int32)), dtype=np.int32)
    i0 = int(fields.get("ips", fields["ids"])) - int(fields["ims"])
    i1 = int(fields.get("ipe", fields["ide"] - 1)) - int(fields["ims"]) + 1
    j0 = int(fields.get("jps", fields["jds"])) - int(fields["jms"])
    j1 = int(fields.get("jpe", fields["jde"] - 1)) - int(fields["jms"]) + 1
    xland_tile = xland[i0:i1, j0:j1]
    xice_tile = xice[i0:i1, j0:j1]
    ivgtyp_tile = ivgtyp[i0:i1, j0:j1]
    glacial = (
        (xland_tile < np.float32(1.5))
        & (xice_tile < np.float32(getattr(config_flags, "xice_threshold", 0.02)))
        & (ivgtyp_tile == int(getattr(config_flags, "isice", 16)))
    )
    glacier_i, glacier_j = np.nonzero(glacial)
    output["_static_glacial_active_i"] = jnp.asarray(glacier_i + i0, dtype=jnp.int32)
    output["_static_glacial_active_j"] = jnp.asarray(glacier_j + j0, dtype=jnp.int32)
    output["_static_glacial_active_count"] = int(glacier_i.size)
    if not land_columns:
        return output
    active = (
        (xland_tile < np.float32(1.5))
        & (xice_tile < np.float32(getattr(config_flags, "xice_threshold", 0.02)))
        & (ivgtyp_tile != int(getattr(config_flags, "isice", 16)))
    )
    local_i, local_j = np.nonzero(active)
    active_i = local_i.astype(np.int32, copy=False) + np.int32(i0)
    active_j = local_j.astype(np.int32, copy=False) + np.int32(j0)
    output["_static_noah_active_i"] = jnp.asarray(active_i, dtype=jnp.int32)
    output["_static_noah_active_j"] = jnp.asarray(active_j, dtype=jnp.int32)
    output["_static_noah_active_count"] = int(active_i.size)
    output["_static_noah_total_count"] = int((i1 - i0) * (j1 - j0))
    active_rows = np.nonzero(active.any(axis=1))[0].astype(np.int32, copy=False) if _STATIC_NOAH_ACTIVE_ROWS else np.asarray([], dtype=np.int32)
    if active_rows.size:
        row_counts = active[active_rows].sum(axis=1).astype(np.int32, copy=False)
        row_max_len = int(row_counts.max())
        row_i = np.repeat((active_rows + np.int32(i0))[:, None], row_max_len, axis=1).astype(np.int32, copy=False)
        row_j = np.zeros((int(active_rows.size), row_max_len), dtype=np.int32)
        row_mask = np.zeros((int(active_rows.size), row_max_len), dtype=np.bool_)
        for row_index, local_row in enumerate(active_rows):
            active_cols = np.nonzero(active[int(local_row)])[0].astype(np.int32, copy=False)
            inactive_cols = np.nonzero(~active[int(local_row)])[0].astype(np.int32, copy=False)
            pad_col = int(inactive_cols[0]) if inactive_cols.size else int(active_cols[-1])
            row_j[row_index, :] = np.int32(pad_col + j0)
            row_j[row_index, : active_cols.size] = active_cols + np.int32(j0)
            row_mask[row_index, : active_cols.size] = True
        output["_static_noah_active_row_i"] = jnp.asarray(row_i, dtype=jnp.int32)
        output["_static_noah_active_row_j"] = jnp.asarray(row_j, dtype=jnp.int32)
        output["_static_noah_active_row_mask"] = jnp.asarray(row_mask)
        output["_static_noah_active_row_count"] = int(active_rows.size)
        output["_static_noah_active_row_max_len"] = int(row_max_len)
    return output


def _with_static_seaice_active_columns(fields: Mapping[str, Any], config_flags: Any) -> dict[str, Any]:
    """Precompute columns that execute WRF ``seaice_noah``."""

    output = dict(fields)
    xice_source = fields.get("xice", fields.get("seaice"))
    if xice_source is None:
        output["_static_seaice_active_i"] = jnp.zeros((0,), dtype=jnp.int32)
        output["_static_seaice_active_j"] = jnp.zeros((0,), dtype=jnp.int32)
        output["_static_seaice_active_count"] = 0
        return output
    xice = np.asarray(
        jax.device_get(jnp.asarray(xice_source, dtype=jnp.float32)), dtype=np.float32
    )
    i0 = int(fields.get("ips", fields["ids"])) - int(fields["ims"])
    i1 = int(fields.get("ipe", fields["ide"] - 1)) - int(fields["ims"]) + 1
    j0 = int(fields.get("jps", fields["jds"])) - int(fields["jms"])
    j1 = int(fields.get("jpe", fields["jde"] - 1)) - int(fields["jms"]) + 1
    threshold = np.float32(getattr(config_flags, "xice_threshold", 0.02))
    local_i, local_j = np.nonzero(xice[i0:i1, j0:j1] >= threshold)
    active_i = local_i.astype(np.int32, copy=False) + np.int32(i0)
    active_j = local_j.astype(np.int32, copy=False) + np.int32(j0)
    output["_static_seaice_active_i"] = jnp.asarray(active_i, dtype=jnp.int32)
    output["_static_seaice_active_j"] = jnp.asarray(active_j, dtype=jnp.int32)
    output["_static_seaice_active_count"] = int(active_i.size)
    return output


def _horizontal_halo_from_state(state: Mapping[str, Any]) -> int:
    return max(1 - int(state.get("ims", 1)), 0, 1 - int(state.get("jms", 1)), 0)


def _local_physics_bounds(e_we: int, e_sn: int, e_vert: int) -> dict[str, int]:
    ni = max(int(e_we) - 1, 1)
    nj = max(int(e_sn) - 1, 1)
    nk = max(int(e_vert) - 1, 1)
    bounds = {
        "ids": 1,
        "ide": int(e_we),
        "jds": 1,
        "jde": int(e_sn),
        "kds": 1,
        "kde": int(e_vert),
        "ims": 1,
        "ime": ni,
        "jms": 1,
        "jme": nj,
        "kms": 1,
        "kme": nk,
        "ips": 1,
        "ipe": ni,
        "jps": 1,
        "jpe": nj,
        "kps": 1,
        "kpe": nk,
        "its": 1,
        "ite": ni,
        "jts": 1,
        "jte": nj,
        "kts": 1,
        "kte": nk,
    }
    for suffix, end in (("31", e_we), ("32", e_vert), ("33", e_sn)):
        bounds[f"sd{suffix}"] = 1
        bounds[f"ed{suffix}"] = int(end)
        bounds[f"sm{suffix}"] = 1
        bounds[f"em{suffix}"] = int(end)
        bounds[f"sp{suffix}"] = 1
        bounds[f"ep{suffix}"] = int(end)
    return bounds


def _is_local_physics_bounds(state: Mapping[str, Any]) -> bool:
    try:
        ide = int(state["ide"])
        jde = int(state["jde"])
        kde = int(state["kde"])
        return (
            int(state["ids"]) == 1
            and int(state["jds"]) == 1
            and int(state["kds"]) == 1
            and int(state["ims"]) == 1
            and int(state["jms"]) == 1
            and int(state["kms"]) == 1
            and int(state["ips"]) == 1
            and int(state["jps"]) == 1
            and int(state["kps"]) == 1
            and int(state.get("its", 1)) == 1
            and int(state.get("jts", 1)) == 1
            and int(state.get("kts", 1)) == 1
            and int(state["ime"]) == ide - 1
            and int(state["jme"]) == jde - 1
            and int(state["kme"]) == kde - 1
            and int(state["ipe"]) == ide - 1
            and int(state["jpe"]) == jde - 1
            and int(state["kpe"]) == kde - 1
            and int(state.get("ite", ide - 1)) == ide - 1
            and int(state.get("jte", jde - 1)) == jde - 1
            and int(state.get("kte", kde - 1)) == kde - 1
        )
    except (KeyError, TypeError, ValueError):
        return False


class _FieldOverlay(Mapping[str, Any]):
    """Read-only field view that applies updates without copying the base mapping."""

    def __init__(self, base: Mapping[str, Any], updates: Mapping[str, Any]) -> None:
        self._base = base
        self._updates = updates

    def __getitem__(self, name: str) -> Any:
        if name in self._updates:
            return self._updates[name]
        return self._base[name]

    def __iter__(self):
        yield from self._updates
        for name in self._base:
            if name not in self._updates:
                yield name

    def __len__(self) -> int:
        return len(self._updates) + sum(1 for name in self._base if name not in self._updates)


def _overlay_fields(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Mapping[str, Any]:
    return base if not updates else _FieldOverlay(base, updates)


class _LazyPhysicsLocalFields(Mapping[str, Any]):
    """Crop physical-domain array fields only when a physics kernel reads them."""

    def __init__(
        self,
        base: Mapping[str, Any],
        *,
        e_we: int,
        e_sn: int,
        e_vert: int,
        halo: int,
    ) -> None:
        self._base = base
        self._bounds = _local_physics_bounds(e_we, e_sn, e_vert)
        self._e_we = int(e_we)
        self._e_sn = int(e_sn)
        self._e_vert = int(e_vert)
        self._halo = int(halo)
        self._cache: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        if name in self._bounds:
            return self._bounds[name]
        if name in self._cache:
            return self._cache[name]
        value = self._base[name]
        shape = getattr(value, "shape", None)
        if (
            name not in _RUNTIME_BOUND_NAMES
            and not _is_boundary_forcing_name(name)
            and shape is not None
            and len(shape) in (2, 3, 4)
        ):
            value = _crop_runtime_field_for_physics(
                name,
                jnp.asarray(value),
                self._e_we,
                self._e_sn,
                self._e_vert,
                self._halo,
            )
        self._cache[name] = value
        return value

    def __iter__(self):
        yield from self._bounds
        for name in self._base:
            if name not in self._bounds:
                yield name

    def __len__(self) -> int:
        return len(self._bounds) + sum(1 for name in self._base if name not in self._bounds)


def _physics_local_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a physical-domain view for column/surface physics kernels."""

    if "ide" not in state or "jde" not in state or "kde" not in state:
        return state
    if _is_local_physics_bounds(state):
        return state
    e_we = int(state["ide"])
    e_sn = int(state["jde"])
    e_vert = int(state["kde"])
    halo = _horizontal_halo_from_state(state)
    if not _EAGER_PHYSICS_LOCAL_CROP:
        return _overlay_fields(state, _local_physics_bounds(e_we, e_sn, e_vert))
    return _LazyPhysicsLocalFields(
        state,
        e_we=e_we,
        e_sn=e_sn,
        e_vert=e_vert,
        halo=halo,
    )


def _physics_callback_state(state: Mapping[str, Any], fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Merge driver callback fields without recropping an already-local state."""

    base = _overlay_fields(state, fields)
    if not _is_local_physics_bounds(base):
        base = _physics_local_state(base)
    defaults = _surface_default_updates(base)
    return _overlay_fields(base, defaults)


def _pad_physics_updates_to_runtime(state: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    if "ide" not in state or "jde" not in state or "kde" not in state:
        return dict(updates)
    halo = _horizontal_halo_from_state(state)
    return _pad_runtime_memory_fields_by_dims_packed(updates, int(state["ide"]), int(state["jde"]), int(state["kde"]), halo)


_SURFACE_DEFAULT_NAMES = (
    "ust",
    "znt",
    "z0",
    "z0base",
    "pblh",
    "mavail",
    "rmol",
    "akhs",
    "akms",
    "rib",
    "br",
    "chs",
    "chs2",
    "cqs2",
    "hfx",
    "qfx",
    "lh",
    "flx_lh",
    "elflx",
    "flhc",
    "flqc",
    "qgh",
    "cpm",
    "ct",
    "u10e",
    "v10e",
    "t02",
    "th02",
    "tshltr",
    "th10",
    "q02",
    "qshltr",
    "q10",
    "pshltr",
    "qsfc",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "grdflx",
    "smstav",
    "smstot",
    "sfcrunoff",
    "udrunoff",
    "rainbl",
    "snotime",
    "acsnom",
    "acsnow",
    "snopcx",
    "potevp",
    "noahres",
    "chklowq",
    "rc2",
    "xlai2",
    "flx4_2d",
    "fvb_2d",
    "fbur_2d",
    "fgsn_2d",
    "swdown",
    "swddir",
    "swddif",
    "glw",
    "gsw",
    "emiss",
    "embck",
    "xice",
    "lowlyr",
    "tke_pbl",
    "exch_h",
)


def _surface_defaults_complete(state: Mapping[str, Any]) -> bool:
    return all(state.get(name) is not None for name in _SURFACE_DEFAULT_NAMES)


def _surface_default_updates(state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("_surface_defaults_complete") is True or _surface_defaults_complete(state):
        return {}
    updates: dict[str, Any] = {}
    zeros2 = _zeros2(state)
    xland = _crop2_to_mass(state, _field_or(state, "xland", jnp.ones(_shape2(state), dtype=jnp.float32)))
    land = xland < 1.5
    physics_init_z0 = jnp.where(land, jnp.asarray(0.1, dtype=jnp.float32), jnp.asarray(0.0001, dtype=jnp.float32))
    physics_init_emiss = jnp.where(land, jnp.asarray(0.85, dtype=jnp.float32), jnp.asarray(0.98, dtype=jnp.float32))
    physics_init_mavail = jnp.where(land, jnp.asarray(0.3, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32))
    defaults = {
        "ust": _full2(state, 0.1),
        "znt": physics_init_z0,
        "z0": physics_init_z0,
        "z0base": physics_init_z0,
        "pblh": zeros2,
        "mavail": physics_init_mavail,
        "rmol": zeros2,
        "akhs": zeros2,
        "akms": zeros2,
        "rib": zeros2,
        "br": zeros2,
        "chs": zeros2,
        "chs2": zeros2,
        "cqs2": zeros2,
        "hfx": zeros2,
        "qfx": zeros2,
        "lh": zeros2,
        "flx_lh": zeros2,
        "elflx": _myjpbl_elflx_field(state, zeros2),
        "flhc": zeros2,
        "flqc": zeros2,
        "qgh": zeros2,
        "cpm": _full2(state, 1004.0),
        "ct": zeros2,
        "u10e": _field_or(state, "u10", zeros2),
        "v10e": _field_or(state, "v10", zeros2),
        "t02": _field_or(state, "t2", zeros2),
        "th02": _field_or(state, "th2", zeros2),
        "tshltr": _field_or(state, "t2", zeros2),
        "th10": _field_or(state, "th2", zeros2),
        "q02": _field_or(state, "q2", zeros2),
        "qshltr": _field_or(state, "q2", zeros2),
        "q10": _field_or(state, "q2", zeros2),
        "pshltr": _field_or(state, "psfc", zeros2),
        "qsfc": zeros2,
        "thz0": _field_or(state, "th2", zeros2),
        "qz0": _field_or(state, "q2", zeros2),
        "uz0": _field_or(state, "u10", zeros2),
        "vz0": _field_or(state, "v10", zeros2),
        "grdflx": zeros2,
        "smstav": zeros2,
        "smstot": zeros2,
        "sfcrunoff": zeros2,
        "udrunoff": zeros2,
        "rainbl": zeros2,
        "snotime": zeros2,
        "acsnom": zeros2,
        "acsnow": zeros2,
        "snopcx": zeros2,
        "potevp": zeros2,
        "noahres": zeros2,
        "chklowq": jnp.full(_shape2(state), -1.0, dtype=jnp.float32),
        "rc2": zeros2,
        "xlai2": zeros2,
        "flx4_2d": zeros2,
        "fvb_2d": zeros2,
        "fbur_2d": zeros2,
        "fgsn_2d": zeros2,
        "swdown": zeros2,
        "swddir": zeros2,
        "swddif": zeros2,
        "glw": zeros2,
        "gsw": zeros2,
        "emiss": physics_init_emiss,
        "embck": _field_or(state, "emiss", physics_init_emiss),
        "xice": _field_or(state, "seaice", zeros2),
    }
    for name, value in defaults.items():
        if state.get(name) is None:
            updates[name] = value
    if state.get("lowlyr") is None:
        updates["lowlyr"] = jnp.ones(_shape2(state), dtype=jnp.int32)
    if state.get("tke_pbl") is None:
        updates["tke_pbl"] = jnp.full_like(jnp.asarray(state["t_phy"]), float(EPSQ2))
    if state.get("exch_h") is None:
        updates["exch_h"] = _zeros3_like(state)
    updates["_surface_defaults_complete"] = True
    return updates


def _identity_stage(_state: dict[str, Any]) -> Mapping[str, Any] | None:
    return None


def _missing_stage(name: str) -> Callable[[dict[str, Any]], Mapping[str, Any] | None]:
    def _raise(_state: dict[str, Any]) -> Mapping[str, Any] | None:
        raise MissingKernelImplementationError(f"strict WRF-JAX runtime reached unwired stage {name!r}")

    return _raise


def _initialize_moist_old_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = initialize_moist_old(
        state["moist"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"moist_old": result}


def _rk_step_prep_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    (
        ru,
        rv,
        rw,
        ww,
        php,
        alt,
        muu,
        muv,
        mut,
        cqu,
        cqv,
        cqw,
    ) = _rk_step_prep_jit_tuple(
        _static_config(state),
        state["rk_step"],
        state["u_2"],
        state["v_2"],
        state["w_2"],
        state["t_2"],
        state["ph_2"],
        state["mu_2"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["moist"],
        state["mub"],
        state["phb"],
        state["pb"],
        state["p"],
        state["al"],
        state["alb"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["msftx"],
        state["msfty"],
        state["fnm"],
        state["fnp"],
        state["dnw"],
        1.0 / float(state["grid"].dx),
        1.0 / float(state["grid"].dy),
        int(state.get("num_moist", _num_moist_from_fields(state))),
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        state["_wrf_real_rounding_token"],
    )
    # The compiled kernel consumes a dynamic zero token that prevents XLA from
    # contracting across WRF REAL stores.  Concrete execution retains the
    # existing operation-by-operation path.
    traced = _has_traced_array(state["u_2"])
    calc_ww_cp_fn = _calc_ww_cp_jit if traced else calc_ww_cp
    rounding_token = state["_wrf_real_rounding_token"] if traced else None
    ww = calc_ww_cp_fn(
        state["u_2"],
        state["v_2"],
        state["mu_2"],
        state["mub"],
        state["c1h"],
        state["c2h"],
        1.0 / float(state["grid"].dx),
        1.0 / float(state["grid"].dy),
        state["msftx"],
        state["msfty"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["dnw"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        rounding_token,
    )
    return {
        "ru": ru,
        "rv": rv,
        "rw": rw,
        "ww": ww,
        "php": php,
        "alt": alt,
        "muu": muu,
        "muv": muv,
        "mut": mut,
        "cqu": cqu,
        "cqv": cqv,
        "cqw": cqw,
    }


def _rk_phys_bc_dry_1_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = rk_phys_bc_dry_1(
        state["config_flags"],
        state["ru"],
        state["rv"],
        state["rw"],
        state["ww"],
        state["muu"],
        state["muv"],
        state["mut"],
        state["php"],
        state["alt"],
        state["p"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {
        "ru": result.u,
        "rv": result.v,
        "rw": result.rw,
        "ww": result.w,
        "muu": result.muu,
        "muv": result.muv,
        "mut": result.mut,
        "php": result.php,
        "alt": result.alt,
        "p": result.p,
    }


def _bc3_variable(field_name: str) -> str:
    return {
        "rho": "p",
        "al": "p",
        "al:small": "p",
        "p": "p",
        "p:small": "p",
        "moist": "p",
        "moist_old": "p",
        "ph_2": "w",
        "ph_2:small": "w",
        "rdzw": "w",
        "rdz": "w",
        "z": "w",
        "zx": "e",
        "zy": "f",
        "t_1": "p",
        "t_save": "t",
        "ru_tend": "u",
        "rv_tend": "v",
        "ru_m": "u",
        "rv_m": "v",
        "ww_m": "w",
        "tke_1": "p",
        "tke_2": "p",
    }.get(field_name, field_name.split(":", 1)[0])


def _bc2_variable(field_name: str) -> str:
    return {
        "muts": "t",
        "muts:small": "t",
        "mudf": "t",
        "mudf:small": "t",
        "mu_1": "t",
        "mu_2": "t",
        "mu_2:small": "t",
        "mut": "t",
        "ustm": "t",
        "ust": "t",
    }.get(field_name, field_name.split(":", 1)[0])


def _set_physical_bc3d_stage(stage_name: str) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    field_name = stage_name.removeprefix("set_physical_bc3d:")

    def _apply(state: dict[str, Any]) -> Mapping[str, Any]:
        base_name = field_name.split(":", 1)[0]
        if base_name not in state:
            raise MissingKernelImplementationError(f"stage {stage_name!r} requires missing field {base_name!r}")
        source = jnp.asarray(state[base_name])

        def apply_one(value: Any) -> Any:
            return set_physical_bc3d(
                value,
                _bc3_variable(field_name),
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )

        if source.ndim == 4:
            updated = source
            for component in range(int(source.shape[-1])):
                updated = updated.at[..., component].set(apply_one(source[..., component]))
        else:
            updated = apply_one(source)
        return {base_name: updated}

    return _apply


def _set_physical_bc2d_stage(stage_name: str) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    field_name = stage_name.removeprefix("set_physical_bc2d:")

    def _apply(state: dict[str, Any]) -> Mapping[str, Any]:
        base_name = field_name.split(":", 1)[0]
        if base_name not in state:
            raise MissingKernelImplementationError(f"stage {stage_name!r} requires missing field {base_name!r}")
        updated = set_physical_bc2d(
            state[base_name],
            _bc2_variable(field_name),
            state["config_flags"],
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["ips"],
            state["ipe"],
            state["jps"],
            state["jpe"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
        )
        return {base_name: updated}

    return _apply


def _init_zero_tendency_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = init_zero_tendency(
        state["ru_tendf"],
        state["rv_tendf"],
        state["rw_tendf"],
        state["ph_tendf"],
        state["t_tendf"],
        state["tke_tend"],
        state["mu_tendf"],
        state["moist_tend"],
        state["chem_tend"],
        state["scalar_tend"],
        state["tracer_tend"],
        int(state.get("num_tracer", 0)),
        int(state.get("num_moist", _num_moist_from_fields(state))),
        int(state.get("num_chem", 0)),
        int(state.get("num_scalar", _num_scalar_from_fields(state))),
        state["rk_step"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {
        "ru_tendf": result.ru_tendf,
        "rv_tendf": result.rv_tendf,
        "rw_tendf": result.rw_tendf,
        "ph_tendf": result.ph_tendf,
        "t_tendf": result.t_tendf,
        "tke_tend": result.tke_tendf,
        "mu_tendf": result.mu_tendf,
        "moist_tend": result.moist_tendf,
        "chem_tend": result.chem_tendf,
        "scalar_tend": result.scalar_tendf,
        "tracer_tend": result.tracer_tendf,
    }


def _phy_prep_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    phy_prep_fn = _phy_prep_jit_tuple if _JIT_PHY_PREP else phy_prep
    phy_config = _static_config(state) if _JIT_PHY_PREP else state["config_flags"]
    n_moist = int(jnp.asarray(state["moist"]).shape[-1])
    rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w = phy_prep_fn(
        phy_config,
        state["mut"],
        state["muu"],
        state["muv"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["u_2"],
        state["v_2"],
        state["p"],
        state["pb"],
        state["alt"],
        state["ph_2"],
        state["phb"],
        state["t_2"],
        state["moist"],
        n_moist,
        state["dnw"],
        state["fzm"],
        state["fzp"],
        state["znw"],
        _scalar_float(state, "p_top"),
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        *(
            (state["_wrf_real_rounding_token"],)
            if _JIT_PHY_PREP
            else ()
        ),
    )
    return {
        "rho": rho,
        "th_phy": th_phy,
        "th_phy_m_t0": th_phy_m_t0,
        "p_phy": p_phy,
        "pi_phy": pi_phy,
        "u_phy": u_phy,
        "v_phy": v_phy,
        "p8w": p8w,
        "t_phy": t_phy,
        "t8w": t8w,
        "z": z,
        "z_at_w": z_at_w,
        "dz8w": dz8w,
        "p_hyd": p_hyd,
        "p_hyd_w": p_hyd_w,
    }


def _radiation_should_run(state: Mapping[str, Any]) -> Any:
    if bool(state.get("_skip_radiation", False)):
        return False
    if bool(state.get("_force_radiation", False)):
        return True
    lw = int(getattr(state["config_flags"], "ra_lw_physics", 0))
    sw = int(getattr(state["config_flags"], "ra_sw_physics", 0))
    if lw == 0 and sw == 0:
        return False
    itimestep = _runtime_itimestep(state)
    stepra = max(1, int(getattr(state["config_flags"], "stepra", 1)))
    radt = float(getattr(state["config_flags"], "radt", 0.0))
    offset = int(getattr(state["config_flags"], "ra_call_offset", 0))
    return itimestep == 1 or radt == 0.0 or stepra == 1 or itimestep % stepra == 1 + offset


def _radiation_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    debug_fn = state.get("radiation_driver_debug_fn")

    def _debug(label: str) -> None:
        if callable(debug_fn):
            debug_fn(label)

    lw_physics = int(state["config_flags"].ra_lw_physics)
    sw_physics = int(state["config_flags"].ra_sw_physics)
    if lw_physics == 0 and sw_physics == 0:
        result = radiation_driver(lw_physics=lw_physics, sw_physics=sw_physics)
        return dict(result.carried_fields)
    if not _radiation_should_run(state):
        return {}
    if lw_physics != 4 or sw_physics != 4:
        raise NotImplementedError(f"active radiation requires RRTMG lw/sw physics 4, got {lw_physics}/{sw_physics}")

    if not (_USE_JAX_RRTMG_LW and _USE_JAX_RRTMG_SW):
        raise RuntimeError("production radiation requires the Python/JAX RRTMG LW and SW implementations")
    yr, julian, gmt, xtime = _radiation_time_values(state)

    zero2 = _zeros2(state)
    zero_mass = jnp.zeros_like(jnp.asarray(state["t_2"]))
    zero_full = jnp.zeros_like(jnp.asarray(_field_or(state, "p8w", state["w_2"])))
    qv = jnp.asarray(state["moist"])[..., 1] if "moist" in state else _field_or(state, "qvapor", zero_mass)
    p8w = _to_full_k(_field_or(state, "p_hyd_w", _field_or(state, "p8w", zero_full)), zero_full)
    t8w = _to_full_k(_field_or(state, "t8w", zero_full), zero_full)
    mass_inputs, cloud_condensate_present = _radiation_mass_inputs(
        (
            qv,
            _field_or(state, "qcloud", zero_mass),
            _field_or(state, "qice", zero_mass),
            _field_or(state, "qsnow", zero_mass),
            _field_or(state, "p_hyd", _field_or(state, "p_phy", state["p"] + state["pb"])),
            _field_or(state, "pi_phy", zero_mass),
            _field_or(state, "dz8w", zero_full),
            _field_or(state, "t_phy", state["t_2"] + T0),
            _field_or(state, "cldfra", zero_mass),
            _field_or(state, "re_cloud", zero_mass),
            _field_or(state, "re_ice", zero_mass),
            _field_or(state, "re_snow", zero_mass),
        ), zero_full, zero_mass,
    )
    qv, qc, qi, qs, p3d, pi3d, dz8w, t3d, cldfra, re_cloud, re_ice, re_snow = mass_inputs
    # phy_prep leaves non-domain memory columns at zero. They still occupy
    # RRTMG batch slots, but zero pressure/temperature poisons their adjoints.
    p3d, pi3d, dz8w, t3d, p8w, t8w = _radiation_thermodynamic_columns(
        state, p3d, pi3d, dz8w, t3d, p8w, t8w,
    )
    icloud_effective = int(getattr(state["config_flags"], "icloud", 1))
    if icloud_effective == 1:
        if not _has_traced_array(cloud_condensate_present) and not bool(cloud_condensate_present):
            cldfra = jnp.zeros_like(cldfra, dtype=jnp.float32)
            icloud_effective = 0
        elif _has_traced_array(qv, qc, qi, qs, t3d, p3d):
            cldfra = _cal_cldfra1_jax(qv, qc, qi, qs, t3d, p3d)
        else:
            cldfra = _cal_cldfra1_np(qv, qc, qi, qs, t3d, p3d)

    declin, solcon = _radconst(julian)
    # WRF computes the RRTMG shortwave solar zenith angle at the center of the
    # radiation interval, not at the beginning of the interval.
    sw_xtime = float(xtime) + float(getattr(state["config_flags"], "radt", 0.0)) * 0.5
    xlat = _field_or(state, "xlat", zero2)
    xlong = _field_or(state, "xlong", zero2)
    if _has_traced_array(xlat, xlong):
        coszr = _calc_coszen_jax(xlat, xlong, julian=julian, xtime=sw_xtime, gmt=gmt, declin=declin)
    else:
        coszr = _calc_coszen_np(xlat, xlong, julian=julian, xtime=sw_xtime, gmt=gmt, declin=declin)
    coszr_f = _radiation_real(coszr)
    o33d = _radiation_real(
        _cam_o3rad_jax(
            _field_or(state, "xlat", zero2),
            p3d,
            julian=julian,
            case_dir=state.get("_case_dir"),
            rounding_token=state.get("_wrf_real_rounding_token"),
        )
    )
    its, ite, jts, jte, kts, kte = _physics_bounds(state)
    co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr = _radiation_greenhouse_scalars(
        str(state.get("_case_dir")) if state.get("_case_dir") is not None else None,
        int(yr),
        float(julian),
        int(getattr(state["config_flags"], "ghg_input", 1)),
    )

    if _USE_JAX_RRTMG_LW:
        lw_kernel = _rrtmg_lwrad_wrf_columns_jit if _JIT_RRTMG_LW else rrtmg_lwrad_wrf_columns
        lw_lookup, lw_clouds, lw_setcoef, lw_spectral, lw_pade = _runtime_rrtmg_lw_tables()
        lw_kwargs = {
            "p3d": _to_mass_k(p3d, zero_mass),
            "p8w": p8w,
            "pi3d": _to_mass_k(pi3d, zero_mass),
            "t3d": _to_mass_k(t3d, zero_mass),
            "t8w": t8w,
            "tsk": _field_or(state, "tsk", zero2),
            "dz8w": _to_mass_k(dz8w, zero_mass),
            "qv3d": _to_mass_k(qv, zero_mass),
            "qc3d": _to_mass_k(qc, zero_mass),
            "qi3d": _to_mass_k(qi, zero_mass),
            "qs3d": _to_mass_k(qs, zero_mass),
            "cldfra3d": _to_mass_k(cldfra, zero_mass),
            "o33d": _to_mass_k(o33d, zero_mass),
            "emiss": _field_or(state, "emiss", jnp.ones_like(zero2)),
            "xland": _field_or(state, "xland", zero2),
            "xice": _field_or(state, "xice", zero2),
            "snow": _field_or(state, "snow", zero2),
            "re_cloud": _to_mass_k(re_cloud, zero_mass),
            "re_ice": _to_mass_k(re_ice, zero_mass),
            "re_snow": _to_mass_k(re_snow, zero_mass),
            "p_top": float(_scalar_float(state, "p_top")),
            "yr": int(yr),
            "julian": np.float32(julian),
            "ghg_input": int(getattr(state["config_flags"], "ghg_input", 1)),
            "case_dir": None,
            "cldovrlp": int(getattr(state["config_flags"], "cldovrlp", 2)),
            "idcor": int(getattr(state["config_flags"], "idcor", 0)),
            "icloud": icloud_effective,
            "has_reqc": int(_field_or(state, "has_reqc", 1)),
            "has_reqi": int(_field_or(state, "has_reqi", 1)),
            "has_reqs": int(_field_or(state, "has_reqs", 1)),
            "co2vmr": co2vmr,
            "n2ovmr": n2ovmr,
            "ch4vmr": ch4vmr,
            "cfc11vmr": cfc11vmr,
            "cfc12vmr": cfc12vmr,
            "lookup_tables": lw_lookup,
            "cloud_tables": lw_clouds,
            "setcoef_tables": lw_setcoef,
            "spectral": lw_spectral,
            "pade_tables": lw_pade,
            "rounding_token": state.get("_wrf_real_rounding_token"),
        }
    else:
        raise RuntimeError("production radiation cannot disable the Python/JAX RRTMG LW implementation")
    if _USE_JAX_RRTMG_SW:
        sw_kernel = _rrtmg_swrad_wrf_columns_jit if _JIT_RRTMG_SW else rrtmg_swrad_wrf_columns
        sw_lookup, sw_clouds, sw_spectral, sw_pade = _runtime_rrtmg_sw_tables()
        sw_kwargs = {
            "p3d": _to_mass_k(p3d, zero_mass),
            "p8w": p8w,
            "pi3d": _to_mass_k(pi3d, zero_mass),
            "t3d": _to_mass_k(t3d, zero_mass),
            "t8w": t8w,
            "tsk": _field_or(state, "tsk", zero2),
            "dz8w": _to_mass_k(dz8w, zero_mass),
            "qv3d": _to_mass_k(qv, zero_mass),
            "qc3d": _to_mass_k(qc, zero_mass),
            "qi3d": _to_mass_k(qi, zero_mass),
            "qs3d": _to_mass_k(qs, zero_mass),
            "cldfra3d": _to_mass_k(cldfra, zero_mass),
            "o33d": _to_mass_k(o33d, zero_mass),
            "albedo": _field_or(state, "albedo", zero2),
            "coszr": coszr,
            "xland": _field_or(state, "xland", zero2),
            "xice": _field_or(state, "xice", zero2),
            "snow": _field_or(state, "snow", zero2),
            "re_cloud": _to_mass_k(re_cloud, zero_mass),
            "re_ice": _to_mass_k(re_ice, zero_mass),
            "re_snow": _to_mass_k(re_snow, zero_mass),
            "solcon": np.float32(solcon),
            "yr": int(yr),
            "julian": np.float32(julian),
            "ghg_input": int(getattr(state["config_flags"], "ghg_input", 1)),
            "cldovrlp": int(getattr(state["config_flags"], "cldovrlp", 2)),
            "idcor": int(getattr(state["config_flags"], "idcor", 0)),
            "icloud": icloud_effective,
            "has_reqc": int(_field_or(state, "has_reqc", 1)),
            "has_reqi": int(_field_or(state, "has_reqi", 1)),
            "has_reqs": int(_field_or(state, "has_reqs", 1)),
            "sf_surface_physics": int(getattr(state["config_flags"], "sf_surface_physics", 0)),
            "co2vmr": co2vmr,
            "n2ovmr": n2ovmr,
            "ch4vmr": ch4vmr,
            "lookup_tables": sw_lookup,
            "cloud_tables": sw_clouds,
            "spectral": sw_spectral,
            "pade_tables": sw_pade,
            "rounding_token": state.get("_wrf_real_rounding_token"),
        }
    else:
        raise RuntimeError("production radiation cannot disable the Python/JAX RRTMG SW implementation")

    _debug("before_lw")
    _profile_radiation_event("before_lw")
    _debug("before_sw")
    _profile_radiation_event("before_sw")
    lw_result, lw_seconds, sw_result, sw_seconds = _run_radiation_sides(
        lw_kernel,
        lw_kwargs,
        sw_kernel,
        sw_kwargs,
        max_columns=_RRTMG_CHUNK_COLUMNS,
        parallel=_RRTMG_PARALLEL_LW_SW and not _has_traced_array(t3d),
    )

    rthratenlw = _radiation_real(lw_result.rthratenlw)
    rthratenlwc = _radiation_real(lw_result.rthratenlwc)
    lwupt = _radiation_real(lw_result.lwupt)
    lwuptc = _radiation_real(lw_result.lwuptc)
    lwuptcln = _radiation_real(lw_result.lwuptcln)
    lwdnt = _radiation_real(lw_result.lwdnt)
    lwdntc = _radiation_real(lw_result.lwdntc)
    lwdntcln = _radiation_real(lw_result.lwdntcln)
    lwupb = _radiation_real(lw_result.lwupb)
    lwupbc = _radiation_real(lw_result.lwupbc)
    lwupbcln = _radiation_real(lw_result.lwupbcln)
    lwdnb = _radiation_real(lw_result.lwdnb)
    lwdnbc = _radiation_real(lw_result.lwdnbc)
    lwdnbcln = _radiation_real(lw_result.lwdnbcln)
    glw = _radiation_real(lw_result.glw)
    olr = _radiation_real(lw_result.olr)
    lwcf = _radiation_real(lw_result.lwcf)
    rthratensw = _radiation_real(sw_result.rthratensw)
    rthratenswc = _radiation_real(sw_result.rthratenswc)
    swupt = _radiation_real(sw_result.swupt)
    swuptc = _radiation_real(sw_result.swuptc)
    swuptcln = _radiation_real(sw_result.swuptcln)
    swdnt = _radiation_real(sw_result.swdnt)
    swdntc = _radiation_real(sw_result.swdntc)
    swdntcln = _radiation_real(sw_result.swdntcln)
    swupb = _radiation_real(sw_result.swupb)
    swupbc = _radiation_real(sw_result.swupbc)
    swupbcln = _radiation_real(sw_result.swupbcln)
    swdnb = _radiation_real(sw_result.swdnb)
    swdnbc = _radiation_real(sw_result.swdnbc)
    swdnbcln = _radiation_real(sw_result.swdnbcln)
    swcf = _radiation_real(sw_result.swcf)
    gsw = _radiation_real(sw_result.gsw)
    swdown = _radiation_real(
        _wrf_swdown_from_gsw(
            gsw,
            _field_or(state, "albedo", zero2),
            state.get("_wrf_real_rounding_token"),
        )
    )
    swddir = _radiation_real(sw_result.swddir)
    swddni = _radiation_real(sw_result.swddni)
    swddif = _radiation_real(sw_result.swddif)
    swdownc = _radiation_real(sw_result.swdownc)
    swddnic = _radiation_real(sw_result.swddnic)
    swddirc = _radiation_real(sw_result.swddirc)
    xcoszen = _radiation_real(sw_result.coszr)
    _profile_radiation_write(
        "lw",
        perf_counter() - lw_seconds,
        (rthratenlw, rthratenlwc, lwupt, lwdnt, lwupb, lwdnb, glw, olr, lwcf),
    )
    _profile_radiation_write(
        "sw",
        perf_counter() - sw_seconds,
        (rthratensw, rthratenswc, swupt, swdnt, swupb, swdnb, swcf, gsw, swddir, swddni, swddif),
    )
    _debug("after_lw")
    _profile_radiation_write(
        "lw_sw_wall",
        perf_counter() - max(lw_seconds, sw_seconds),
        (rthratenlw, rthratensw),
    )
    _debug("after_sw")
    rthratenlw_out = _to_mass_k(rthratenlw, zero_mass)
    rthratenlwc_out = _to_mass_k(rthratenlwc, zero_mass)
    rthratensw_out = _to_mass_k(rthratensw, zero_mass)
    rthratenswc_out = _to_mass_k(rthratenswc, zero_mass)
    rthraten = jnp.asarray(rthratenlw_out, dtype=jnp.float32) + jnp.asarray(rthratensw_out, dtype=jnp.float32)
    return {
        "rthratenlw": rthratenlw_out,
        "rthratenlwc": rthratenlwc_out,
        "rthratensw": rthratensw_out,
        "rthratenswc": rthratenswc_out,
        "rthraten": rthraten,
        "cldfra": _to_mass_k(cldfra, zero_mass),
        "lwupt": lwupt,
        "lwuptc": lwuptc,
        "lwuptcln": lwuptcln,
        "lwdnt": lwdnt,
        "lwdntc": lwdntc,
        "lwdntcln": lwdntcln,
        "lwupb": lwupb,
        "lwupbc": lwupbc,
        "lwupbcln": lwupbcln,
        "lwdnb": lwdnb,
        "lwdnbc": lwdnbc,
        "lwdnbcln": lwdnbcln,
        "glw": glw,
        "olr": olr,
        "rlwtoa": olr,
        "lwcf": lwcf,
        "swupt": swupt,
        "swuptc": swuptc,
        "swuptcln": swuptcln,
        "swdnt": swdnt,
        "swdntc": swdntc,
        "swdntcln": swdntcln,
        "swupb": swupb,
        "swupbc": swupbc,
        "swupbcln": swupbcln,
        "swdnb": swdnb,
        "swdnbc": swdnbc,
        "swdnbcln": swdnbcln,
        "swdown": swdown,
        "swcf": swcf,
        "gsw": gsw,
        "swddir": swddir,
        "swddni": swddni,
        "swddif": swddif,
        "swdownc": swdownc,
        "swddnic": swddnic,
        "swddirc": swddirc,
        "xcoszen": xcoszen,
        "coszr": coszr_f,
    }


def _moist_physics_prep_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    t_new = jnp.asarray(state["t_2"])
    moist_arr = jnp.asarray(state["moist"]) if "moist" in state else None
    qv = _to_mass_k(_moist_component(moist_arr, 1, _field_or(state, "qvapor", t_new)) if moist_arr is not None else _field_or(state, "qvapor", t_new), t_new)
    qc = _to_mass_k(_moist_component(moist_arr, 2, _field_or(state, "qcloud", jnp.zeros_like(t_new))) if moist_arr is not None else _field_or(state, "qcloud", jnp.zeros_like(t_new)), t_new)
    ph = jnp.asarray(state["ph_2"])
    phb = jnp.asarray(state["phb"])
    return prepare_microphysics(
        t_new, qv, qc, ph, phb,
        *(jnp.asarray(state[name]) for name in ("al", "alb", "p", "pb", "fzm", "fzp")),
        gravity=G, t0=T0, rv=R_V, rd=R_D,
        p0=_field_or(state, "p0", 100000.0), exponent=R_D / CP,
        use_theta_m=int(getattr(state["config_flags"], "use_theta_m", 0)) == 1,
        preserve_eager_rounding=bool(state.get("_preserve_water_statement_rounding", False)) or bool(state.get("_force_radiation", False)) or not _has_traced_array(t_new),
    )


@lru_cache(maxsize=4)
def _device_thompson_tables(case_dir: str) -> Any:
    """Place unchanged lookup data on the selected device once per case."""
    return jax.tree_util.tree_map(jnp.asarray, load_thompson_tables(case_dir))


def _compiled_thompson_batch(case_dir: str, *, tables: Any = None) -> Callable[..., tuple[Any, ...]]:
    """Bind explicit model parameters; diagnostics may load them by case name."""
    if tables is None:
        tables = _device_thompson_tables(case_dir)
    return partial(_thompson_batch, tables)


@jax.jit
def _thompson_batch(tables, qv, qc, qi, qr, qs, qg, qni, qnr, th, pii, p, w, dz, qvs, qvsi, delqvs, dt):

    def _run_column(qv1d, qc1d, qi1d, qr1d, qs1d, qg1d, ni1d, nr1d, th1d, pii1d, p1d, w1d, dz1d, qvs1d, qvsi1d, delqvs1d, dt):
        t1d = th1d * pii1d
        defaults = thompson_grid_driver_defaults(qv1d, qg1d, t1d, p1d)
        result = thompson_column_driver(
            qv1d,
            qc1d,
            qi1d,
            qr1d,
            qs1d,
            qg1d,
            defaults.qb1d,
            ni1d,
            nr1d,
            defaults.nc1d,
            defaults.ng1d,
            defaults.nwfa1d,
            defaults.nifa1d,
            defaults.nbca1d,
            t1d,
            p1d,
            dz1d,
            w1d,
            tables,
            dt,
            qvs1d=qvs1d,
            qvsi1d=qvsi1d,
            delqvs1d=delqvs1d,
        )
        final = result.final
        effect = calc_effect_rad(final.t1d, p1d, final.qv1d, final.qc1d, final.nc1d, final.qi1d, final.ni1d, final.qs1d)
        return (
            final.qv1d,
            final.qc1d,
            final.qr1d,
            final.qi1d,
            final.qs1d,
            final.qg1d,
            final.ni1d,
            final.nr1d,
            final.t1d / pii1d,
            result.pptrain,
            result.pptsnow,
            result.pptgraul,
            result.pptice,
            effect.re_qc1d,
            effect.re_qi1d,
            effect.re_qs1d,
            final.ng1d,
            final.qb1d,
        )

    return jax.vmap(_run_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))(qv, qc, qi, qr, qs, qg, qni, qnr, th, pii, p, w, dz, qvs, qvsi, delqvs, dt)


@jax.jit
def _wrf_driver_qv_negative_fix_kernel(raw: Any, old: Any, use_average: Any) -> Any:
    """Apply the vertical recurrence with one reusable compiled executable."""

    next_old = jnp.concatenate([old[:, 1:], old[:, -1:]], axis=1)

    def _step(prev: Any, inputs: tuple[Any, Any, Any]) -> tuple[Any, Any]:
        raw_level, next_old_level, use_average_level = inputs
        averaged = jnp.maximum(jnp.float32(1.0e-7), jnp.float32(0.5) * (prev + next_old_level))
        corrected = jnp.where(raw_level < jnp.float32(0.0), jnp.where(use_average_level, averaged, jnp.float32(1.0e-7)), raw_level)
        return corrected, corrected

    _, corrected_by_level = jax.lax.scan(_step, old[:, 0], (raw.T, next_old.T, use_average))
    return corrected_by_level.T


def _wrf_driver_qv_negative_fix(qv_raw_cols: Any, qv_input_cols: Any, *, kts: int, kte: int) -> Any:
    """Replicate ``mp_gt_driver`` post-pass correction for negative water vapor."""

    raw = jnp.asarray(qv_raw_cols, dtype=jnp.float32)
    old = jnp.asarray(qv_input_cols, dtype=jnp.float32)
    nk = int(raw.shape[1])
    levels = jnp.arange(nk, dtype=jnp.int32) + jnp.asarray(int(kts), dtype=jnp.int32)
    use_average = (levels < jnp.asarray(int(kte) - 2, dtype=jnp.int32)) & (levels > jnp.asarray(int(kts) + 1, dtype=jnp.int32))
    return _wrf_driver_qv_negative_fix_kernel(raw, old, use_average)


def _microphysics_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    mp_physics = int(getattr(state["config_flags"], "mp_physics", 0))
    if mp_physics == 0:
        return {}
    if mp_physics != 8:
        raise NotImplementedError(f"active microphysics requires Thompson mp_physics=8, got {mp_physics}")

    zero2 = _zeros2(state)
    mass = jnp.asarray(_field_or(state, "th_phy", state["t_2"] + T0))
    full = jnp.asarray(_field_or(state, "p8w", state["w_2"]))
    moist_arr = jnp.asarray(state["moist"]) if "moist" in state else None
    qv = jnp.asarray(_to_full_k(_moist_component(moist_arr, 1, _field_or(state, "qvapor", mass)) if moist_arr is not None else _field_or(state, "qvapor", mass), full), dtype=jnp.float32)
    qc = jnp.asarray(_to_full_k(_moist_component(moist_arr, 2, _field_or(state, "qcloud", jnp.zeros_like(mass))) if moist_arr is not None else _field_or(state, "qcloud", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    qr = jnp.asarray(_to_full_k(_moist_component(moist_arr, 3, _field_or(state, "qrain", jnp.zeros_like(mass))) if moist_arr is not None else _field_or(state, "qrain", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    qi = jnp.asarray(_to_full_k(_moist_component(moist_arr, 4, _field_or(state, "qice", jnp.zeros_like(mass))) if moist_arr is not None else _field_or(state, "qice", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    qs = jnp.asarray(_to_full_k(_moist_component(moist_arr, 5, _field_or(state, "qsnow", jnp.zeros_like(mass))) if moist_arr is not None else _field_or(state, "qsnow", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    qg = jnp.asarray(_to_full_k(_moist_component(moist_arr, 6, _field_or(state, "qgraup", jnp.zeros_like(mass))) if moist_arr is not None else _field_or(state, "qgraup", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    scalar_arr = jnp.asarray(state["scalar"]) if "scalar" in state else None
    qni = jnp.asarray(
        _to_full_k(
            _scalar_component(scalar_arr, 1, _field_or(state, "qni", _field_or(state, "qnice", jnp.zeros_like(mass))))
            if scalar_arr is not None
            else _field_or(state, "qni", _field_or(state, "qnice", jnp.zeros_like(mass))),
            full,
        ),
        dtype=jnp.float32,
    )
    qnr = jnp.asarray(
        _to_full_k(
            _scalar_component(scalar_arr, 2, _field_or(state, "qnr", _field_or(state, "qnrain", jnp.zeros_like(mass))))
            if scalar_arr is not None
            else _field_or(state, "qnr", _field_or(state, "qnrain", jnp.zeros_like(mass))),
            full,
        ),
        dtype=jnp.float32,
    )
    th = jnp.asarray(_to_full_k(mass, full), dtype=jnp.float32)
    qv_before_mp = qv
    qc_before_mp = qc
    qr_before_mp = qr
    qi_before_mp = qi
    qs_before_mp = qs
    qg_before_mp = qg
    th_before_mp = th
    pii = jnp.asarray(_to_full_k(_field_or(state, "pi_phy", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    pressure = state.get("p_phy")
    if pressure is None:
        pressure = state.get("p_hyd")
    if pressure is None:
        pressure = state["p"] + state["pb"]
    p = jnp.asarray(_to_full_k(pressure, full), dtype=jnp.float32)
    w = jnp.asarray(_to_full_k(_field_or(state, "w_2", full), full), dtype=jnp.float32)
    dz = jnp.asarray(_to_full_k(_field_or(state, "dz8w", jnp.zeros_like(full)), full), dtype=jnp.float32)
    rainnc = jnp.asarray(_field_or(state, "rainnc", zero2), dtype=jnp.float32)
    rainncv = jnp.asarray(_field_or(state, "rainncv", zero2), dtype=jnp.float32)
    snownc = jnp.asarray(_field_or(state, "snownc", zero2), dtype=jnp.float32)
    snowncv = jnp.asarray(_field_or(state, "snowncv", zero2), dtype=jnp.float32)
    graupelnc = jnp.asarray(_field_or(state, "graupelnc", zero2), dtype=jnp.float32)
    graupelncv = jnp.asarray(_field_or(state, "graupelncv", zero2), dtype=jnp.float32)
    sr = jnp.zeros_like(jnp.asarray(_field_or(state, "sr", zero2), dtype=jnp.float32))
    refl_10cm = jnp.asarray(_to_full_k(_field_or(state, "refl_10cm", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    re_cloud = jnp.asarray(_to_full_k(_field_or(state, "re_cloud", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    re_ice = jnp.asarray(_to_full_k(_field_or(state, "re_ice", jnp.zeros_like(mass)), full), dtype=jnp.float32)
    re_snow = jnp.asarray(_to_full_k(_field_or(state, "re_snow", jnp.zeros_like(mass)), full), dtype=jnp.float32)

    its, ite, jts, jte, kts, kte = _physics_bounds(state, skip_spec_zone=True)
    i_end = min(int(ite), int(state["ide"]) - 1)
    j_end = min(int(jte), int(state["jde"]) - 1)
    i0 = max(int(its) - int(state["ims"]), 0)
    i1 = max(i_end - int(state["ims"]) + 1, i0)
    j0 = max(int(jts) - int(state["jms"]), 0)
    j1 = max(j_end - int(state["jms"]) + 1, j0)
    k0 = max(int(kts) - int(state["kms"]), 0)
    k1 = max(int(kte) - int(state["kms"]) + 1, k0)
    rounding_token = None
    if i1 > i0 and j1 > j0 and k1 > k0:
        ni = i1 - i0
        nj = j1 - j0
        nk = k1 - k0

        def _active_columns(value: Any) -> Any:
            tile = jnp.asarray(value)[i0:i1, k0:k1, j0:j1]
            return jnp.reshape(jnp.transpose(tile, (0, 2, 1)), (ni * nj, nk))

        def _columns_to_active(value: Any) -> Any:
            return jnp.transpose(jnp.reshape(value, (ni, nj, nk)), (0, 2, 1))

        def _columns_to_2d(value: Any) -> Any:
            return jnp.reshape(value, (ni, nj))

        case_dir = str(state.get("_case_dir", Path.cwd()))
        thompson_batch = _compiled_thompson_batch(case_dir, tables=state["thompson_tables"])
        dt = jnp.asarray(float(state["grid"].dt), dtype=jnp.float32)
        # Retain the independently submitted Thompson arithmetic when the
        # caller is compiled, including its dynamic REAL timestep argument.
        if bool(state.get("_preserve_water_statement_rounding", False)) or bool(state.get("_force_radiation", False)):
            thompson_batch = partial(semantic_call, thompson_batch)
            dt = _materialize_wrf_real(
                dt, state.get("_wrf_real_rounding_token", jnp.uint32(0)),
            )
        qv_input_cols = _active_columns(qv)
        th_cols_in = _active_columns(th)
        pii_cols_in = _active_columns(pii)
        p_cols_in = _active_columns(p)
        t_cols_in = th_cols_in * pii_cols_in
        rounding_token = state["_wrf_real_rounding_token"] if _has_traced_array(t_cols_in) else None
        if rounding_token is None:
            qvs_cols_in = rslf(p_cols_in, t_cols_in)
            qvsi_cols_in = jnp.where(t_cols_in <= jnp.float32(273.15), rsif(p_cols_in, t_cols_in), qvs_cols_in)
            qvs_0c_cols_in = rslf(p_cols_in, jnp.ones_like(t_cols_in) * jnp.float32(273.15))
        else:
            qvs_cols_in = rslf_rounded(p_cols_in, t_cols_in, rounding_token)
            qvsi_cols_in = jnp.where(
                t_cols_in <= jnp.float32(273.15),
                rsif_rounded(p_cols_in, t_cols_in, rounding_token),
                qvs_cols_in,
            )
            qvs_0c_cols_in = rslf_rounded(
                p_cols_in,
                jnp.ones_like(t_cols_in) * jnp.float32(273.15),
                rounding_token,
            )
        delqvs_cols_in = jnp.maximum(jnp.float32(0.0), qvs_0c_cols_in - qv_input_cols)

        thompson_args = (
            qv_input_cols,
            _active_columns(qc),
            _active_columns(qi),
            _active_columns(qr),
            _active_columns(qs),
            _active_columns(qg),
            _active_columns(qni),
            _active_columns(qnr),
            th_cols_in,
            pii_cols_in,
            p_cols_in,
            _active_columns(w),
            _active_columns(dz),
            qvs_cols_in,
            qvsi_cols_in,
            delqvs_cols_in,
            dt,
        )
        (
            qv_raw_cols,
            qc_cols,
            qr_cols,
            qi_cols,
            qs_cols,
            qg_cols,
            qni_cols,
            qnr_cols,
            th_cols,
            pptrain_cols,
            pptsnow_cols,
            pptgraul_cols,
            pptice_cols,
            re_cloud_cols,
            re_ice_cols,
            re_snow_cols,
            ng_cols,
            qb_cols,
        ) = _run_column_batch_chunked(
            thompson_batch,
            thompson_args,
            max_columns=_THOMPSON_CHUNK_COLUMNS,
            workers=_THOMPSON_CHUNK_WORKERS,
        )
        qv_cols = _wrf_driver_qv_negative_fix(qv_raw_cols, qv_input_cols, kts=int(kts), kte=int(kte))

        qv = qv.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qv_cols))
        qc = qc.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qc_cols))
        qr = qr.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qr_cols))
        qi = qi.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qi_cols))
        qs = qs.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qs_cols))
        qg = qg.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qg_cols))
        qni = qni.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qni_cols))
        qnr = qnr.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(qnr_cols))
        th = th.at[i0:i1, k0:k1, j0:j1].set(_columns_to_active(th_cols))
        rain_step = pptrain_cols + pptsnow_cols + pptgraul_cols + pptice_cols
        snow_step = pptsnow_cols + pptice_cols
        graupel_step = pptgraul_cols
        rain_step_2d = _columns_to_2d(rain_step)
        snow_step_2d = _columns_to_2d(snow_step)
        graupel_step_2d = _columns_to_2d(graupel_step)
        rainncv = rainncv.at[i0:i1, j0:j1].set(rain_step_2d)
        rainnc = rainnc.at[i0:i1, j0:j1].add(rain_step_2d)
        snowncv = snowncv.at[i0:i1, j0:j1].set(snow_step_2d)
        snownc = snownc.at[i0:i1, j0:j1].add(snow_step_2d)
        graupelncv = graupelncv.at[i0:i1, j0:j1].set(graupel_step_2d)
        graupelnc = graupelnc.at[i0:i1, j0:j1].add(graupel_step_2d)
        sr = sr.at[i0:i1, j0:j1].set((snow_step_2d + graupel_step_2d) / (rain_step_2d + jnp.float32(1.0e-12)))
        if int(_field_or(state, "has_reqc", 1)) != 0 and int(_field_or(state, "has_reqi", 1)) != 0 and int(_field_or(state, "has_reqs", 1)) != 0:
            re_cloud = re_cloud.at[i0:i1, k0:k1, j0:j1].set(jnp.maximum(jnp.float32(2.49e-6), jnp.minimum(_columns_to_active(re_cloud_cols), jnp.float32(50.0e-6))))
            re_ice = re_ice.at[i0:i1, k0:k1, j0:j1].set(jnp.maximum(jnp.float32(4.99e-6), jnp.minimum(_columns_to_active(re_ice_cols), jnp.float32(125.0e-6))))
            re_snow = re_snow.at[i0:i1, k0:k1, j0:j1].set(jnp.maximum(jnp.float32(9.99e-6), jnp.minimum(_columns_to_active(re_snow_cols), jnp.float32(999.0e-6))))
        if int(getattr(state["config_flags"], "do_radar_ref", 0)) == 1:
            refl_np = np.asarray(refl_10cm, dtype=np.float32).copy()
            qv_np = np.asarray(qv)
            qc_np = np.asarray(qc)
            qr_np = np.asarray(qr)
            qs_np = np.asarray(qs)
            qg_np = np.asarray(qg)
            qnr_np = np.asarray(qnr)
            th_np = np.asarray(th)
            pii_np = np.asarray(pii)
            p_np = np.asarray(p)
            ng_np = np.asarray(_columns_to_active(ng_cols))
            qb_np = np.asarray(_columns_to_active(qb_cols))
            for ii in range(ni):
                for jj in range(nj):
                    refl_np[i0 + ii, k0:k1, j0 + jj] = np.maximum(
                        -35.0,
                        calc_refl10cm(
                            qv_np[i0 + ii, k0:k1, j0 + jj],
                            qc_np[i0 + ii, k0:k1, j0 + jj],
                            qr_np[i0 + ii, k0:k1, j0 + jj],
                            qnr_np[i0 + ii, k0:k1, j0 + jj],
                            qs_np[i0 + ii, k0:k1, j0 + jj],
                            qg_np[i0 + ii, k0:k1, j0 + jj],
                            ng_np[ii, :, jj],
                            qb_np[ii, :, jj],
                            th_np[i0 + ii, k0:k1, j0 + jj] * pii_np[i0 + ii, k0:k1, j0 + jj],
                            p_np[i0 + ii, k0:k1, j0 + jj],
                            ke_diag=int(kte),
                            radar_state=tables.radar,
                        ),
                    )
            refl_10cm = jnp.asarray(refl_np)

    qv_out = _to_mass_k(qv, mass)
    qc_out = _to_mass_k(qc, mass)
    qr_out = _to_mass_k(qr, mass)
    qi_out = _to_mass_k(qi, mass)
    qs_out = _to_mass_k(qs, mass)
    qg_out = _to_mass_k(qg, mass)
    qni_out = _to_mass_k(qni, mass)
    qnr_out = _to_mass_k(qnr, mass)
    th_out = _to_mass_k(th, mass)
    dt_mp = jnp.asarray(float(state["grid"].dt), dtype=jnp.float32)
    tendency_rounding_token = rounding_token
    tendency_dt = dt_mp
    if tendency_rounding_token is not None:
        tendency_dt = _materialize_wrf_real(tendency_dt, tendency_rounding_token)

    def _microphysics_tendency(after: Any, before: Any) -> Any:
        delta = after - before
        if tendency_rounding_token is not None:
            delta = _materialize_wrf_real(delta, tendency_rounding_token)
        tendency = delta / tendency_dt
        if tendency_rounding_token is not None:
            tendency = _materialize_wrf_real(tendency, tendency_rounding_token)
        return _to_mass_k(tendency, mass)

    rthmpten = _microphysics_tendency(th, th_before_mp)
    rqvmpten = _microphysics_tendency(qv, qv_before_mp)
    rqcmpten = _microphysics_tendency(qc, qc_before_mp)
    rqrmpten = _microphysics_tendency(qr, qr_before_mp)
    rqimpten = _microphysics_tendency(qi, qi_before_mp)
    rqsmpten = _microphysics_tendency(qs, qs_before_mp)
    rqgmpten = _microphysics_tendency(qg, qg_before_mp)
    refl_10cm_out = _to_mass_k(refl_10cm, mass)
    re_cloud_out = _to_mass_k(re_cloud, mass)
    re_ice_out = _to_mass_k(re_ice, mass)
    re_snow_out = _to_mass_k(re_snow, mass)
    moist = _replace_moist_components(
        state["moist"],
        {
            1: qv_out,
            2: qc_out,
            3: qr_out,
            4: qi_out,
            5: qs_out,
            6: qg_out,
        },
    )
    scalar = _replace_scalar_components(_field_or(state, "scalar", jnp.zeros(moist.shape[:3] + (3,), dtype=jnp.asarray(moist).dtype)), {1: qni_out, 2: qnr_out})
    return {
        "qvapor": qv_out,
        "moist": moist,
        "qcloud": qc_out,
        "qrain": qr_out,
        "qice": qi_out,
        "qsnow": qs_out,
        "qgraup": qg_out,
        "qni": qni_out,
        "qnice": qni_out,
        "qnr": qnr_out,
        "qnrain": qnr_out,
        "scalar": scalar,
        "th_phy": th_out,
        "rthmpten": rthmpten,
        "rqvmpten": rqvmpten,
        "rqcmpten": rqcmpten,
        "rqrmpten": rqrmpten,
        "rqimpten": rqimpten,
        "rqsmpten": rqsmpten,
        "rqgmpten": rqgmpten,
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
        "graupelnc": graupelnc,
        "graupelncv": graupelncv,
        "sr": sr,
        "refl_10cm": refl_10cm_out,
        "re_cloud": re_cloud_out,
        "re_ice": re_ice_out,
        "re_snow": re_snow_out,
    }


def _moist_physics_finish_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    t_new = jnp.asarray(state["t_2"])
    th_phy = jnp.asarray(_field_or(state, "th_phy", t_new + T0))
    h_before = jnp.asarray(_field_or(state, "h_diabatic", th_phy))
    qv = _to_mass_k(jnp.asarray(state["moist"])[..., 1] if "moist" in state else _field_or(state, "qvapor", t_new), t_new)
    qc = _to_mass_k(_field_or(state, "qcloud", jnp.zeros_like(t_new)), t_new)
    qv_before = _to_mass_k(_field_or(state, "qv_diabatic", qv), t_new)
    qc_before = _to_mass_k(_field_or(state, "qc_diabatic", qc), t_new)
    dt = float(state["grid"].dt)
    if int(getattr(state["config_flags"], "no_mp_heating", 0)) == 0:
        limit = float(getattr(state["config_flags"], "mp_tend_lim", 10.0)) * dt
        ratio = jnp.asarray(R_V, dtype=t_new.dtype) / jnp.asarray(R_D, dtype=t_new.dtype)
        t_out, th_phy_m_t0, h_diabatic, qv_diabatic, qc_diabatic = finish_microphysics(
            (t_new, th_phy, h_before, qv, qc, qv_before, qc_before),
            dt=dt, limit=limit, ratio=ratio, t0=T0,
            use_theta_m=int(getattr(state["config_flags"], "use_theta_m", 0)) == 1,
            preserve_eager_rounding=bool(state.get("_preserve_water_statement_rounding", False)) or bool(state.get("_force_radiation", False)) or not _has_traced_array(t_new),
        )
    else:
        t_out = t_new
        th_phy_m_t0 = t_new
        h_diabatic = jnp.zeros_like(t_new)
        qv_diabatic = jnp.zeros_like(t_new)
        qc_diabatic = jnp.zeros_like(t_new)
    return {
        "t_2": t_out,
        "t": t_out,
        "th_phy_m_t0": th_phy_m_t0,
        "h_diabatic": h_diabatic,
        "qv_diabatic": qv_diabatic,
        "qc_diabatic": qc_diabatic,
    }


def _as_result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "__dataclass_fields__"):
        return {name: getattr(result, name) for name in result.__dataclass_fields__}
    raise TypeError(f"cannot convert {type(result).__name__} to a runtime update mapping")


def _runtime_myjsfc_callback(state: dict[str, Any]) -> Callable[..., Mapping[str, Any]]:
    def _call(
        *,
        u_phytmp: Any,
        v_phytmp: Any,
        psfc: Any,
        qgh: Any,
        chs: Any,
        cpm: Any,
        chs2: Any,
        ust: Any,
        fields: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        base = _physics_callback_state(state, fields)
        its, ite, jts, jte, kts, kte = _surface_bounds(base)

        moist = jnp.asarray(base["moist"])
        qv = _crop3_to_mass(base, moist[..., 1])
        qc = _crop3_to_mass(base, _moist_component(moist, 2, jnp.zeros_like(moist[..., 1])))
        p1000mb = _scalar_float_or(base, "p00", 100000.0)

        zero2 = _zeros2(base)
        full01_2 = _full2(base, 0.1)
        ones2 = jnp.ones(_shape2(base), dtype=jnp.float32)
        ones2_i = jnp.ones(_shape2(base), dtype=jnp.int32)
        tke_pbl = _field_or(base, "tke_pbl", jnp.full_like(base["t_phy"], float(EPSQ2)))
        myjsfc_itimestep = _runtime_itimestep(base)
        rounding_token = base.get("_wrf_real_rounding_token")
        if rounding_token is not None:
            myjsfc_itimestep = jax.lax.bitwise_xor(
                jnp.asarray(myjsfc_itimestep, dtype=jnp.int32),
                jnp.asarray(rounding_token, dtype=jnp.uint32).astype(jnp.int32),
            )
        surface_prefix_args = (
            myjsfc_itimestep,
            _crop2_to_mass(base, base["hgt"]),
            _crop3_to_mass(base, base["dz8w"]),
            _crop3_to_mass(base, _field_or(base, "p_hyd", base["p_phy"])),
            _crop3_to_mass(base, _field_or(base, "p_hyd_w", base["p8w"])),
            _crop3_to_mass(base, base["th_phy"]),
            _crop3_to_mass(base, base["t_phy"]),
            qv,
            qc,
            _crop3_to_mass(base, u_phytmp),
            _crop3_to_mass(base, v_phytmp),
            _crop3_to_mass(base, tke_pbl),
            _crop2_to_mass(base, base["tsk"]),
            _crop2_to_mass(base, _field_or(base, "qsfc", zero2)),
            _crop2_to_mass(base, _field_or(base, "thz0", _field_or(base, "th2", zero2))),
            _crop2_to_mass(base, _field_or(base, "qz0", _field_or(base, "q2", zero2))),
            _crop2_to_mass(base, _field_or(base, "uz0", _field_or(base, "u10", zero2))),
            _crop2_to_mass(base, _field_or(base, "vz0", _field_or(base, "v10", zero2))),
            _crop2_to_mass(base, _field_or(base, "lowlyr", ones2_i)),
            _crop2_to_mass(base, base["xland"]),
            _crop2_to_mass(base, base["ivgtyp"]),
            int(base["config_flags"].isurban),
            int(base["config_flags"].iz0tlnd),
        )
        surface_state_args = (
            _crop2_to_mass(base, _field_or(base, "ust", _field_or(base, "ustar", full01_2))),
            _crop2_to_mass(base, _field_or(base, "znt", full01_2)),
            # WRF surface_driver passes its live Z0 field as MYJSFC's
            # Z0BASE argument.  Noah updates Z0 every physics step, whereas
            # the old Python-only z0base copy remained frozen at initialization
            # and incorrectly shadowed those updates from step 2 onward.
            _crop2_to_mass(base, _field_or(base, "z0", _field_or(base, "znt", full01_2))),
            _crop2_to_mass(base, _field_or(base, "pblh", zero2)),
            _crop2_to_mass(base, _field_or(base, "mavail", ones2)),
            _crop2_to_mass(base, _field_or(base, "rmol", zero2)),
            _crop2_to_mass(base, _field_or(base, "akhs", zero2)),
            _crop2_to_mass(base, _field_or(base, "akms", zero2)),
        )
        surface_flux_args = (
            _crop2_to_mass(base, _field_or(base, "br", _field_or(base, "rib", zero2))),
            _crop2_to_mass(base, chs),
            _crop2_to_mass(base, chs2),
            _crop2_to_mass(base, _field_or(base, "cqs2", zero2)),
            _crop2_to_mass(base, _field_or(base, "hfx", zero2)),
            _crop2_to_mass(base, _field_or(base, "qfx", zero2)),
            _crop2_to_mass(base, _field_or(base, "flx_lh", _field_or(base, "lh", zero2))),
            _crop2_to_mass(base, _field_or(base, "flhc", zero2)),
            _crop2_to_mass(base, _field_or(base, "flqc", zero2)),
            _crop2_to_mass(base, qgh),
            _crop2_to_mass(base, cpm),
            _crop2_to_mass(base, _field_or(base, "ct", zero2)),
            _crop2_to_mass(base, _field_or(base, "u10", zero2)),
            _crop2_to_mass(base, _field_or(base, "v10", zero2)),
            _crop2_to_mass(base, _field_or(base, "t02", _field_or(base, "t2", zero2))),
            _crop2_to_mass(base, _field_or(base, "th02", _field_or(base, "th2", zero2))),
            _crop2_to_mass(base, _field_or(base, "tshltr", _field_or(base, "t2", zero2))),
            _crop2_to_mass(base, _field_or(base, "th10", _field_or(base, "th2", zero2))),
            _crop2_to_mass(base, _field_or(base, "q02", _field_or(base, "q2", zero2))),
            _crop2_to_mass(base, _field_or(base, "qshltr", _field_or(base, "q2", zero2))),
            _crop2_to_mass(base, _field_or(base, "q10", _field_or(base, "q2", zero2))),
            _crop2_to_mass(base, _field_or(base, "pshltr", _field_or(base, "psfc", zero2))),
        )
        surface_tail_args = (
            int(base["ids"]),
            int(base["ide"]),
            int(base["jds"]),
            int(base["jde"]),
            int(base["kds"]),
            int(base["kde"]),
            int(base["ims"]),
            int(base["ime"]),
            int(base["jms"]),
            int(base["jme"]),
            int(base["kms"]),
            int(base["kme"]),
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )
        use_fractional_seaice = int(getattr(base["config_flags"], "fractional_seaice", 0)) == 1 and not bool(
            base.get("_static_fractional_seaice_inactive", False)
        )
        if use_fractional_seaice:
            surface = myjsfc_seaice_wrapper(
                *surface_prefix_args,
                bool(getattr(base["config_flags"], "tice2tsk_if2cold", False)),
                float(getattr(base["config_flags"], "xice_threshold", 0.02)),
                _crop2_to_mass(base, _field_or(base, "xice", _field_or(base, "seaice", zero2))),
                _crop2_to_mass(base, _field_or(base, "sst", base["tsk"])),
                _crop2_to_mass(base, _field_or(base, "chs_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "chs2_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "cqs2_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "cpm_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "flhc_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "flqc_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "qsfc_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "qgh_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "qz0_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "hfx_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "qfx_sea", zero2)),
                _crop2_to_mass(base, _field_or(base, "flx_lh_sea", _field_or(base, "lh_sea", zero2))),
                _crop2_to_mass(base, _field_or(base, "tsk_sea", zero2)),
                *surface_state_args,
                *surface_flux_args,
                p1000mb,
                _crop2_to_mass(base, _field_or(base, "u10e", _field_or(base, "u10", _zeros2(base)))),
                _crop2_to_mass(base, _field_or(base, "v10e", _field_or(base, "v10", _zeros2(base)))),
                *surface_tail_args,
                myjsfc_fn=myjsfc_vectorized,
            )
        else:
            surface = myjsfc_vectorized(
                *surface_prefix_args,
                *surface_state_args,
                *surface_flux_args,
                p1000mb,
                _crop2_to_mass(base, _field_or(base, "u10e", _field_or(base, "u10", _zeros2(base)))),
                _crop2_to_mass(base, _field_or(base, "v10e", _field_or(base, "v10", _zeros2(base)))),
                *surface_tail_args,
            )
        br_value = surface.rib if hasattr(surface, "rib") else surface.br
        updates = {
            "qsfc": surface.qsfc,
            "thz0": surface.thz0,
            "qz0": surface.qz0,
            "uz0": surface.uz0,
            "vz0": surface.vz0,
            "ustar": surface.ustar,
            "ust": surface.ustar,
            "znt": surface.znt,
            "pblh": surface.pblh,
            "rmol": surface.rmol,
            "akhs": surface.akhs,
            "akms": surface.akms,
            "rib": br_value,
            "br": br_value,
            "chs": surface.chs,
            "chs2": surface.chs2,
            "cqs2": surface.cqs2,
            "hfx": surface.hfx,
            "qfx": surface.qfx,
            "flx_lh": surface.flx_lh,
            "lh": surface.flx_lh,
            "flhc": surface.flhc,
            "flqc": surface.flqc,
            "qgh": surface.qgh,
            "cpm": surface.cpm,
            "ct": surface.ct,
            "u10": surface.u10,
            "v10": surface.v10,
            "t02": surface.t02,
            "th02": surface.th02,
            "tshltr": surface.tshltr,
            "th10": surface.th10,
            "q02": surface.q02,
            "qshltr": surface.qshltr,
            "q10": surface.q10,
            "pshltr": surface.pshltr,
            "u10e": surface.u10e,
            "v10e": surface.v10e,
        }
        if hasattr(surface, "tsk"):
            updates.update(
                {
                    "tsk": surface.tsk,
                    "sst": surface.sst,
                    "chs_sea": surface.chs_sea,
                    "chs2_sea": surface.chs2_sea,
                    "cqs2_sea": surface.cqs2_sea,
                    "cpm_sea": surface.cpm_sea,
                    "flhc_sea": surface.flhc_sea,
                    "flqc_sea": surface.flqc_sea,
                    "qsfc_sea": surface.qsfc_sea,
                    "qgh_sea": surface.qgh_sea,
                    "qz0_sea": surface.qz0_sea,
                    "hfx_sea": surface.hfx_sea,
                    "qfx_sea": surface.qfx_sea,
                    "flx_lh_sea": surface.flx_lh_sea,
                    "lh_sea": surface.flx_lh_sea,
                    "tsk_sea": surface.tsk_sea,
                }
            )
        return updates

    return _call


def _fractional_seaice_noah_inputs(
    base: Mapping[str, Any],
    albedo: Any,
    emiss: Any,
) -> tuple[Any, Any]:
    """Recover ice-only radiative properties before running Noah.

    WRF stores grid-cell means between physics steps.  Its surface driver
    removes the open-water contribution before ``seaice_noah`` and restores
    that contribution afterward.
    """

    xice = jnp.asarray(base["xice"])
    threshold = jnp.asarray(float(base["config_flags"].xice_threshold), dtype=xice.dtype)
    ice_mask = (xice >= threshold) & (xice <= jnp.asarray(1.0, dtype=xice.dtype))
    safe_xice = jnp.where(ice_mask, xice, jnp.asarray(1.0, dtype=xice.dtype))
    open_water_fraction = jnp.asarray(1.0, dtype=xice.dtype) - xice

    def recover(grid_mean: Any, open_water_value: float) -> Any:
        mean = jnp.asarray(grid_mean)
        fraction = safe_xice.astype(mean.dtype)
        open_fraction = open_water_fraction.astype(mean.dtype)
        water = jnp.asarray(open_water_value, dtype=mean.dtype)
        ice = (mean - open_fraction * water) / fraction
        return jnp.where(ice_mask, ice, mean)

    return recover(albedo, 0.08), recover(emiss, 0.98)


def _fractional_seaice_noah_updates(
    base: Mapping[str, Any],
    lsm_updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Blend Noah's ice-only outputs with MYJ open-water tile outputs.

    This is the whole-array translation of the ``FRACTIONAL_SEAICE`` block
    immediately following ``seaice_noah`` in WRF's ``surface_driver``.  The
    blend is required before the next surface-layer call: MYJ's sea-ice
    wrapper expects grid-cell means and backs out the ice-only component.
    """

    updates = dict(lsm_updates)
    xice = jnp.asarray(base["xice"])
    threshold = jnp.asarray(float(base["config_flags"].xice_threshold), dtype=xice.dtype)
    ice_mask = (xice >= threshold) & (xice <= jnp.asarray(1.0, dtype=xice.dtype))
    open_water_fraction = jnp.asarray(1.0, dtype=xice.dtype) - xice

    def blend(ice_value: Any, sea_value: Any) -> Any:
        ice = jnp.asarray(ice_value)
        fraction = xice.astype(ice.dtype)
        open_fraction = open_water_fraction.astype(ice.dtype)
        sea = jnp.asarray(sea_value, dtype=ice.dtype)
        blended = ice * fraction + open_fraction * sea
        return jnp.where(ice_mask, blended, ice)

    updates["albedo"] = blend(updates["albedo"], jnp.asarray(0.08, dtype=xice.dtype))
    updates["emiss"] = blend(updates["emiss"], jnp.asarray(0.98, dtype=xice.dtype))

    blend_pairs = (
        ("flhc", "flhc_sea"),
        ("flqc", "flqc_sea"),
        ("cpm", "cpm_sea"),
        ("cqs2", "cqs2_sea"),
        ("chs2", "chs2_sea"),
        ("chs", "chs_sea"),
        ("qsfc", "qsfc_sea"),
        ("qgh", "qgh_sea"),
        ("qz0", "qz0_sea"),
        ("hfx", "hfx_sea"),
        ("qfx", "qfx_sea"),
        ("lh", "flx_lh_sea"),
    )
    for field_name, sea_name in blend_pairs:
        ice_value = updates.get(field_name, base[field_name])
        updates[field_name] = blend(ice_value, base[sea_name])

    ice_tsk = jnp.asarray(updates["tsk"])
    previous_tsk_save = jnp.asarray(base.get("tsk_save", base["tsk"]), dtype=ice_tsk.dtype)
    updates["tsk_save"] = jnp.where(ice_mask, ice_tsk, previous_tsk_save)
    updates["tsk"] = blend(ice_tsk, base["tsk_sea"])
    updates["flx_lh"] = updates["lh"]
    updates["elflx"] = updates["lh"]
    return updates


def _runtime_lsm_callback(state: dict[str, Any]) -> Callable[..., Mapping[str, Any]]:
    static_dzs_value = state.get("_static_dzs_tuple")
    if static_dzs_value is None:
        static_dzs = tuple(float(value) for value in jax.device_get(jnp.asarray(state["dzs"], dtype=jnp.float32)).reshape(-1))
    else:
        static_dzs = tuple(float(value) for value in static_dzs_value)
    static_dt = float(state["grid"].dt)
    static_xice_threshold = float(state["config_flags"].xice_threshold)

    def _call(
        *,
        u_phytmp: Any,
        v_phytmp: Any,
        psfc: Any,
        qgh: Any,
        chs: Any,
        cpm: Any,
        chs2: Any,
        fields: Mapping[str, Any],
        dtbl: float | None = None,
    ) -> Mapping[str, Any]:
        del u_phytmp, v_phytmp, psfc
        base = _physics_callback_state(state, fields)
        surface_dt = static_dt if dtbl is None else float(dtbl)
        its, ite, jts, jte, kts, kte = _surface_bounds(base)
        qv = jnp.asarray(base["moist"])[..., 1]
        num_soil_layers = int(base["config_flags"].num_soil_layers)
        parameters = base.get("noah_parameters")
        if parameters is None:
            raise MissingKernelImplementationError("surface_driver Noah LSM requires loaded Noah table parameters")
        albedo = base["albbck"] if "albedo" not in base and "albbck" in base else _field_or(base, "albedo", _full2(base, 0.2))
        emiss = _field_or(base, "emiss", _full2(base, 0.95))
        use_fractional_seaice = int(getattr(base["config_flags"], "fractional_seaice", 0)) == 1 and not bool(
            base.get("_static_fractional_seaice_inactive", False)
        )
        if use_fractional_seaice:
            albedo, emiss = _fractional_seaice_noah_inputs(base, albedo, emiss)
        result = lsm(
            dz8w=base["dz8w"],
            qv3d=qv,
            p8w3d=_field_or(base, "p_hyd_w", base["p8w"]),
            t3d=base["t_phy"],
            tsk=base["tsk"],
            hfx=_field_or(base, "hfx", _zeros2(base)),
            qfx=_field_or(base, "qfx", _zeros2(base)),
            lh=_field_or(base, "lh", _zeros2(base)),
            grdflx=_field_or(base, "grdflx", _zeros2(base)),
            qgh=qgh,
            gsw=_field_or(base, "gsw", _zeros2(base)),
            swdown=_field_or(base, "swdown", _zeros2(base)),
            swddir=_field_or(base, "swddir", _zeros2(base)),
            swddif=_field_or(base, "swddif", _zeros2(base)),
            glw=_field_or(base, "glw", _zeros2(base)),
            smstav=_field_or(base, "smstav", _zeros2(base)),
            smstot=_field_or(base, "smstot", _zeros2(base)),
            sfcrunoff=_field_or(base, "sfcrunoff", _zeros2(base)),
            udrunoff=_field_or(base, "udrunoff", _zeros2(base)),
            ivgtyp=base["ivgtyp"],
            isltyp=base["isltyp"],
            isurban=int(base["config_flags"].isurban),
            isice=int(base["config_flags"].isice),
            vegfra=base["vegfra"],
            albedo=albedo,
            albbck=_field_or(base, "albbck", _full2(base, 0.2)),
            znt=_field_or(base, "znt", _full2(base, 0.1)),
            z0=_field_or(base, "z0", _field_or(base, "z0base", _full2(base, 0.1))),
            tmn=base["tmn"],
            xland=base["xland"],
            xice=_field_or(base, "xice", _zeros2(base)),
            emiss=emiss,
            embck=_field_or(base, "embck", _full2(base, 0.95)),
            snowc=_field_or(base, "snowc", _zeros2(base)),
            qsfc=_field_or(base, "qsfc", _zeros2(base)),
            rainbl=_field_or(base, "rainbl", _zeros2(base)),
            mminlu=str(base["config_flags"].mminlu),
            num_soil_layers=num_soil_layers,
            dt=surface_dt,
            dzs=static_dzs,
            itimestep=_runtime_itimestep(base),
            smois=base["smois"],
            tslb=base["tslb"],
            snow=_field_or(base, "snow", _zeros2(base)),
            canwat=_field_or(base, "canwat", _zeros2(base)),
            chs=chs,
            chs2=chs2,
            cqs2=_field_or(base, "cqs2", _zeros2(base)),
            cpm=cpm,
            rovcp=jnp.asarray(287.0 / 1004.0, dtype=jnp.float32),
            sr=_field_or(base, "sr", _zeros2(base)),
            chklowq=_field_or(base, "chklowq", jnp.full(_shape2(base), -1.0, dtype=jnp.float32)),
            lai=_field_or(base, "lai", _zeros2(base)),
            qz0=_field_or(base, "qz0", _field_or(base, "q2", _zeros2(base))),
            myj=True,
            frpcpn="sr" in base,
            sh2o=base["sh2o"],
            snowh=_field_or(base, "snowh", _zeros2(base)),
            snoalb=_field_or(base, "snoalb", _full2(base, 0.8)),
            shdmin=_field_or(base, "shdmin", _zeros2(base)),
            shdmax=_field_or(base, "shdmax", _full2(base, 100.0)),
            snotime=_field_or(base, "snotime", _zeros2(base)),
            acsnom=_field_or(base, "acsnom", _zeros2(base)),
            acsnow=_field_or(base, "acsnow", _zeros2(base)),
            snopcx=_field_or(base, "snopcx", _zeros2(base)),
            potevp=_field_or(base, "potevp", _zeros2(base)),
            smcrel=_field_or(base, "smcrel", jnp.zeros_like(base["smois"])),
            xice_threshold=static_xice_threshold,
            rdlai2d=bool(base["config_flags"].rdlai2d),
            usemonalb=bool(base["config_flags"].usemonalb),
            rib=_field_or(base, "rib", _zeros2(base)),
            noahres=_field_or(base, "noahres", _zeros2(base)),
            opt_thcnd=int(base["config_flags"].opt_thcnd),
            ua_phys=False,
            flx4_2d=_field_or(base, "flx4_2d", _zeros2(base)),
            fvb_2d=_field_or(base, "fvb_2d", _zeros2(base)),
            fbur_2d=_field_or(base, "fbur_2d", _zeros2(base)),
            fgsn_2d=_field_or(base, "fgsn_2d", _zeros2(base)),
            noah_active_row_i=base.get("_static_noah_active_row_i"),
            noah_active_row_j=base.get("_static_noah_active_row_j"),
            noah_active_row_mask=base.get("_static_noah_active_row_mask"),
            noah_active_i=base.get("_static_noah_active_i"),
            noah_active_j=base.get("_static_noah_active_j"),
            noah_glacial_i=base.get("_static_glacial_active_i"),
            noah_glacial_j=base.get("_static_glacial_active_j"),
            update_diagnostics=_UPDATE_NOAH_NONFEEDBACK_DIAGNOSTICS,
            rounding_token=base.get("_wrf_real_rounding_token"),
            ids=base["ids"],
            ide=base["ide"],
            jds=base["jds"],
            jde=base["jde"],
            kds=base["kds"],
            kde=base["kde"],
            ims=base["ims"],
            ime=base["ime"],
            jms=base["jms"],
            jme=base["jme"],
            kms=base["kms"],
            kme=base["kme"],
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
            kts=kts,
            kte=kte,
            sf_urban_physics=int(base["config_flags"].sf_urban_physics),
            rc2=_field_or(base, "rc2", _zeros2(base)),
            xlai2=_field_or(base, "xlai2", _zeros2(base)),
            parameters=parameters,
        )
        updates = _as_result_mapping(result)
        seaice_updates = seaice_noah(
            seaice_albedo_opt=int(getattr(base["config_flags"], "seaice_albedo_opt", 0)),
            seaice_albedo_default=float(getattr(base["config_flags"], "seaice_albedo_default", 0.65)),
            seaice_thickness_opt=int(getattr(base["config_flags"], "seaice_thickness_opt", 0)),
            seaice_thickness_default=float(getattr(base["config_flags"], "seaice_thickness_default", 3.0)),
            seaice_snowdepth_opt=int(getattr(base["config_flags"], "seaice_snowdepth_opt", 0)),
            seaice_snowdepth_max=float(getattr(base["config_flags"], "seaice_snowdepth_max", 1.0e10)),
            seaice_snowdepth_min=float(getattr(base["config_flags"], "seaice_snowdepth_min", 0.001)),
            t3d=base["t_phy"],
            qv3d=qv,
            p8w3d=_field_or(base, "p_hyd_w", base["p8w"]),
            dz8w=base["dz8w"],
            dt=surface_dt,
            frpcpn="sr" in base,
            sr=_field_or(base, "sr", _zeros2(base)),
            glw=_field_or(base, "glw", _zeros2(base)),
            swdown=_field_or(base, "swdown", _zeros2(base)),
            rainbl=_field_or(base, "rainbl", _zeros2(base)),
            qgh=qgh,
            xice=_field_or(base, "xice", _zeros2(base)),
            albsi=_field_or(base, "albsi", jnp.full(_shape2(base), -1.0e36, dtype=jnp.float32)),
            icedepth=_field_or(base, "icedepth", jnp.full(_shape2(base), -1.0e36, dtype=jnp.float32)),
            snowsi=_field_or(base, "snowsi", jnp.full(_shape2(base), -1.0e36, dtype=jnp.float32)),
            tslb=updates["tslb"],
            emiss=updates["emiss"],
            albedo=updates["albedo"],
            z02d=updates["z0"],
            tsk=updates["tsk"],
            snow=updates["snow"],
            snowc=updates["snowc"],
            snowh=updates["snowh"],
            chs=updates["chs"],
            chs2=updates["chs2"],
            cqs2=updates["cqs2"],
            rib=updates["rib"],
            znt=updates["znt"],
            lh=updates["lh"],
            hfx=updates["hfx"],
            qfx=updates["qfx"],
            potevp=updates["potevp"],
            grdflx=updates["grdflx"],
            qsfc=updates["qsfc"],
            acsnow=updates["acsnow"],
            acsnom=updates["acsnom"],
            snopcx=updates["snopcx"],
            sfcrunoff=updates["sfcrunoff"],
            noahres=updates["noahres"],
            active_i=base.get("_static_seaice_active_i", jnp.zeros((0,), dtype=jnp.int32)),
            active_j=base.get("_static_seaice_active_j", jnp.zeros((0,), dtype=jnp.int32)),
            k0=int(kts) - int(base["kms"]),
            update_diagnostics=_UPDATE_NOAH_NONFEEDBACK_DIAGNOSTICS,
        )
        for field_name in (
            "tslb",
            "emiss",
            "albedo",
            "z0",
            "tsk",
            "snow",
            "snowc",
            "snowh",
            "chs",
            "chs2",
            "cqs2",
            "znt",
            "lh",
            "hfx",
            "qfx",
            "potevp",
            "grdflx",
            "qsfc",
            "acsnow",
            "acsnom",
            "snopcx",
            "sfcrunoff",
            "noahres",
        ):
            updates[field_name] = seaice_updates[field_name]
        for optional_name in ("albsi", "icedepth", "snowsi"):
            if optional_name in base:
                updates[optional_name] = seaice_updates[optional_name]
        if use_fractional_seaice:
            updates = _fractional_seaice_noah_updates(base, updates)
        else:
            updates["flx_lh"] = updates["lh"]
            updates["elflx"] = updates["lh"]
        return updates

    return _call


def _runtime_sfcdiags_callback(state: dict[str, Any]) -> Callable[..., Mapping[str, Any]]:
    def _call(
        *,
        psfc: Any,
        chs: Any,
        chs2: Any,
        fields: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        base = _physics_callback_state(state, fields)
        its, ite, jts, jte, _kts, _kte = _surface_bounds(base)
        qv = _crop3_to_mass(base, jnp.asarray(base["moist"])[..., 1])
        zero2 = _zeros2(base)
        psfc_field = psfc if psfc is not None else _field_or(base, "psfc", zero2)
        chs_field = chs if chs is not None else _field_or(base, "chs", zero2)
        chs2_field = chs2 if chs2 is not None else _field_or(base, "chs2", zero2)
        result = sfcdiags(
            _crop2_to_mass(base, _field_or(base, "hfx", zero2)),
            _crop2_to_mass(base, _field_or(base, "qfx", zero2)),
            _crop2_to_mass(base, base["tsk"]),
            _crop2_to_mass(base, _field_or(base, "qsfc", zero2)),
            _crop2_to_mass(base, chs2_field),
            _crop2_to_mass(base, _field_or(base, "cqs2", zero2)),
            _crop2_to_mass(base, _field_or(base, "t2", _field_or(base, "t02", zero2))),
            _crop2_to_mass(base, _field_or(base, "th2", _field_or(base, "th02", zero2))),
            _crop2_to_mass(base, _field_or(base, "q2", _field_or(base, "q02", zero2))),
            _crop2_to_mass(base, psfc_field),
            jnp.asarray(CP, dtype=jnp.float32),
            jnp.asarray(R_D, dtype=jnp.float32),
            jnp.asarray(R_D / CP, dtype=jnp.float32),
            _crop2_to_mass(base, chs_field),
            _crop3_to_mass(base, base["t_phy"]),
            qv,
            False,
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
        )
        q2_limited = jnp.asarray(result.q2)
        i_slice = slice(int(its) - int(base["ims"]), int(ite) - int(base["ims"]) + 1)
        j_slice = slice(int(jts) - int(base["jms"]), int(jte) - int(base["jms"]) + 1)
        xland = _crop2_to_mass(base, _field_or(base, "xland", jnp.ones(_shape2(base), dtype=jnp.float32)))
        qv_low = qv[:, 0, :]
        q2_tile = q2_limited[i_slice, j_slice]
        q2_cap_tile = jnp.where(
            xland[i_slice, j_slice] < 1.5,
            jnp.minimum(q2_tile, 1.05 * qv_low[i_slice, j_slice]),
            q2_tile,
        )
        q2_limited = q2_limited.at[i_slice, j_slice].set(q2_cap_tile)
        return {
            "q2": q2_limited,
            "q02": q2_limited,
            "qshltr": q2_limited,
            "t2": result.t2,
            "t02": result.t2,
            "tshltr": result.t2,
            "th2": result.th2,
            "th02": result.th2,
        }

    return _call


def _surface_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    runtime_state = state
    state = _physics_local_state(state)
    defaults = _surface_default_updates(state)
    if defaults:
        state = dict(state)
        state.update(defaults)
    its, ite, jts, jte, kts, kte = _surface_bounds(state)
    result = surface_driver(
        sf_sfclay_physics=int(state["config_flags"].sf_sfclay_physics),
        sf_surface_physics=int(state["config_flags"].sf_surface_physics),
        bl_pbl_physics=int(state["config_flags"].bl_pbl_physics),
        ra_lw_physics=int(state["config_flags"].ra_lw_physics),
        fractional_seaice=int(getattr(state["config_flags"], "fractional_seaice", 0)),
        itimestep=_runtime_itimestep(state),
        stepbl=int(state["config_flags"].stepbl),
        bldt=float(state["config_flags"].bldt),
        dt=float(state["grid"].dt),
        curr_secs=state["curr_secs"],
        adapt_step_flag=state["adapt_step_flag"],
        bldtacttime=state.get("bldtacttime"),
        isfflx=int(state["config_flags"].isfflx),
        u_phy=state["u_phy"],
        v_phy=state["v_phy"],
        u_phytmp=state.get("u_phytmp"),
        v_phytmp=state.get("v_phytmp"),
        u_frame=_scalar_float_or(state, "u_frame", 0.0),
        v_frame=_scalar_float_or(state, "v_frame", 0.0),
        uoce=state.get("uoce"),
        voce=state.get("voce"),
        p8w=_field_or(state, "p_hyd_w", state["p8w"]),
        psfc=state.get("psfc"),
        qgh=state.get("qgh"),
        chs=state.get("chs"),
        cpm=state.get("cpm"),
        chs2=state.get("chs2"),
        ust=state.get("ust"),
        ustm=state.get("ustm"),
        wspd=state.get("wspd"),
        ch=state.get("ch"),
        rainbl=state.get("rainbl"),
        raincv=state.get("raincv", 0.0),
        rainncv=state.get("rainncv"),
        rainshv=state.get("rainshv"),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        i_start=(its,),
        i_end=(ite,),
        j_start=(jts,),
        j_end=(jte,),
        kts=kts,
        kte=kte,
        myjsfc_fn=_runtime_myjsfc_callback(state),
        lsm_fn=_runtime_lsm_callback(state),
        sfcdiags_fn=_runtime_sfcdiags_callback(state),
    )
    updates = dict(defaults)
    updates.update(dict(result.carried_fields))
    updates.update(
        {
            "q_ref2m": result.q_ref2m,
            "u_phytmp": result.u_phytmp,
            "v_phytmp": result.v_phytmp,
            "qgh": result.qgh,
            "chs": result.chs,
            "cpm": result.cpm,
            "chs2": result.chs2,
            "psfc": result.psfc,
            "ust": result.ust,
            "ustm": result.ustm,
            "wspd": result.wspd,
            "ch": result.ch,
            "bldtacttime": result.bldtacttime,
        }
    )
    local_after_surface = dict(state)
    local_after_surface.update(updates)
    padded = _pad_physics_updates_to_runtime(runtime_state, updates)
    padded["_physics_local_after_surface"] = local_after_surface
    return padded


def _runtime_myjpbl_callback(state: dict[str, Any]) -> Callable[..., Mapping[str, Any]]:
    def _call(
        *,
        dt: float,
        dtbl: float,
        stepbl: int,
        u_phytmp: Any,
        v_phytmp: Any,
        psfc: Any,
        rublten: Any,
        rvblten: Any,
        rthblten: Any,
        rqvblten: Any,
        rqcblten: Any,
        rqiblten: Any,
        pblh: Any,
        kpbl: Any,
        qv_curr: Any,
        qc_curr: Any,
        qi_curr: Any,
        fields: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del dtbl, psfc
        base = _physics_callback_state(state, fields)
        its, ite, jts, jte, kts, kte = _surface_bounds(base)
        qv_source = jnp.asarray(base["moist"])[..., 1] if qv_curr is None else qv_curr
        qc_source = jnp.zeros_like(qv_source) if qc_curr is None else qc_curr
        qi_source = None if qi_curr is None else qi_curr
        qv = _crop3_to_mass(base, qv_source)
        qcw = _crop3_to_mass(base, qc_source)
        qci = None if qi_source is None else _crop3_to_mass(base, qi_source)
        zero3 = jnp.zeros_like(qv)
        zero2 = _zeros2(base)
        full01_2 = _full2(base, 0.1)
        minus_one2 = jnp.full(_shape2(base), -1.0, dtype=jnp.float32)
        ones2_i = jnp.ones(_shape2(base), dtype=jnp.int32)
        zeros2_i = jnp.zeros(_shape2(base), dtype=jnp.int32)
        tke_default = jnp.full_like(base["t_phy"], float(EPSQ2))
        zero_t_phy = jnp.zeros_like(base["t_phy"])
        result = myjpbl(
            dt=float(dt),
            stepbl=int(stepbl),
            ht=_crop2_to_mass(base, base["hgt"]),
            dz=_crop3_to_mass(base, base["dz8w"]),
            pmid=_crop3_to_mass(base, _field_or(base, "p_hyd", base["p_phy"])),
            pint=_crop3_to_mass(base, _field_or(base, "p_hyd_w", base["p8w"])),
            th=_crop3_to_mass(base, base["th_phy"]),
            t=_crop3_to_mass(base, base["t_phy"]),
            exner=_crop3_to_mass(base, base["pi_phy"]),
            qv=qv,
            qcw=qcw,
            qci=qci,
            qcs=zero3,
            qcr=zero3,
            qcg=zero3,
            u=_crop3_to_mass(base, u_phytmp),
            v=_crop3_to_mass(base, v_phytmp),
            rho=_crop3_to_mass(base, base["rho"]),
            tsk=_crop2_to_mass(base, base["tsk"]),
            qsfc=_crop2_to_mass(base, _field_or(base, "qsfc", zero2)),
            chklowq=_crop2_to_mass(base, _field_or(base, "chklowq", minus_one2)),
            thz0=_crop2_to_mass(base, _field_or(base, "thz0", _field_or(base, "th2", zero2))),
            qz0=_crop2_to_mass(base, _field_or(base, "qz0", _field_or(base, "q2", zero2))),
            uz0=_crop2_to_mass(base, _field_or(base, "uz0", _field_or(base, "u10", zero2))),
            vz0=_crop2_to_mass(base, _field_or(base, "vz0", _field_or(base, "v10", zero2))),
            lowlyr=_crop2_to_mass(base, _field_or(base, "lowlyr", ones2_i)),
            xland=_crop2_to_mass(base, base["xland"]),
            sice=_crop2_to_mass(base, _field_or(base, "xice", zero2)),
            snow=_crop2_to_mass(base, _field_or(base, "snow", zero2)),
            tke_myj=_crop3_to_mass(base, _field_or(base, "tke_pbl", tke_default)),
            exch_h=_crop3_to_mass(base, _field_or(base, "exch_h", zero_t_phy)),
            ustar=_crop2_to_mass(base, _field_or(base, "ust", full01_2)),
            znt=_crop2_to_mass(base, _field_or(base, "znt", full01_2)),
            el_myj=_crop3_to_mass(base, _field_or(base, "el_pbl", zero_t_phy)),
            pblh=_crop2_to_mass(base, _field_or(base, "pblh", pblh)),
            kpbl=_crop2_to_mass(base, _field_or(base, "kpbl", zeros2_i)),
            ct=_crop2_to_mass(base, _field_or(base, "ct", zero2)),
            akhs=_crop2_to_mass(base, _field_or(base, "akhs", zero2)),
            akms=_crop2_to_mass(base, _field_or(base, "akms", zero2)),
            elflx=_crop2_to_mass(
                base,
                _myjpbl_elflx_field(base, zero2),
            ),
            mixht=_crop2_to_mass(base, _field_or(base, "mixht", zero2)),
            rublten=_crop3_to_mass(base, _field_or(base, "rublten", rublten)),
            rvblten=_crop3_to_mass(base, _field_or(base, "rvblten", rvblten)),
            rthblten=_crop3_to_mass(base, _field_or(base, "rthblten", rthblten)),
            rqvblten=_crop3_to_mass(base, _field_or(base, "rqvblten", rqvblten)),
            rqcblten=None if rqcblten is None else _crop3_to_mass(base, rqcblten),
            rqiblten=None if rqiblten is None else _crop3_to_mass(base, rqiblten),
            rqsblten=None,
            rqrblten=None,
            rqgblten=None,
            ids=base["ids"],
            ide=base["ide"],
            jds=base["jds"],
            jde=base["jde"],
            kds=base["kds"],
            kde=base["kde"],
            ims=base["ims"],
            ime=base["ime"],
            jms=base["jms"],
            jme=base["jme"],
            kms=base["kms"],
            kme=base["kme"],
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
            kts=kts,
            kte=kte,
            rounding_token=base.get("_wrf_real_rounding_token"),
        )
        result_updates = _as_result_mapping(result)
        updates = {
            name: result_updates[name]
            for name in (
                "rublten",
                "rvblten",
                "rthblten",
                "rqvblten",
                "rqcblten",
                "rqiblten",
                "pblh",
                "kpbl",
                "qsfc",
                "thz0",
                "qz0",
                "uz0",
                "vz0",
                "znt",
                "exch_h",
                "ct",
                "akhs",
                "akms",
                "mixht",
            )
            if name in result_updates
        }
        if "ustar" in result_updates:
            updates["ust"] = result_updates["ustar"]
        if "tke_myj" in result_updates:
            updates["tke_pbl"] = result_updates["tke_myj"]
        if "el_myj" in result_updates:
            updates["el_pbl"] = result_updates["el_myj"]
        return updates

    return _call


def _runtime_gwd_callback(state: dict[str, Any]) -> Callable[..., Mapping[str, Any]]:
    def _call(
        *,
        gwd_opt: int,
        dtbl: float,
        rublten: Any,
        rvblten: Any,
        rthblten: Any,
        pblh: Any,
        kpbl: Any,
        fields: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del rthblten, pblh
        if int(gwd_opt) != 1:
            raise NotImplementedError(f"runtime GWDO callback currently supports gwd_opt=1; got {int(gwd_opt)}")
        base = _physics_callback_state(state, fields)
        its, ite, jts, jte, kts, kte = _surface_bounds(base)
        zero2 = _zeros2(base)
        zero3 = jnp.zeros_like(_crop3_to_mass(base, rublten))
        qv = _crop3_to_mass(base, jnp.asarray(base["moist"])[..., 1])
        ones2 = _full2(base, 1.0)
        ep1 = float(P608)

        (
            rublten_out,
            rvblten_out,
            dtaux3d_out,
            dtauy3d_out,
            dusfcg_out,
            dvsfcg_out,
        ) = gwdo(
            u3d=_crop3_to_mass(base, base["u_phy"]),
            v3d=_crop3_to_mass(base, base["v_phy"]),
            t3d=_crop3_to_mass(base, base["t_phy"]),
            qv3d=qv,
            p3d=_crop3_to_mass(base, _field_or(base, "p_hyd", base["p_phy"])),
            p3di=_crop3_to_mass(base, _field_or(base, "p_hyd_w", base["p8w"])),
            pi3d=_crop3_to_mass(base, base["pi_phy"]),
            z=_crop3_to_mass(base, base["z"]),
            rublten=_crop3_to_mass(base, rublten),
            rvblten=_crop3_to_mass(base, rvblten),
            dtaux3d=_crop3_to_mass(base, _field_or(base, "dtaux3d", zero3)),
            dtauy3d=_crop3_to_mass(base, _field_or(base, "dtauy3d", zero3)),
            dusfcg=_crop2_to_mass(base, _field_or(base, "dusfcg", zero2)),
            dvsfcg=_crop2_to_mass(base, _field_or(base, "dvsfcg", zero2)),
            var2d=_crop2_to_mass(base, _field_or(base, "var", zero2)),
            oc12d=_crop2_to_mass(base, _field_or(base, "oc12d", _field_or(base, "con", zero2))),
            oa2d1=_crop2_to_mass(base, _field_or(base, "oa1", zero2)),
            oa2d2=_crop2_to_mass(base, _field_or(base, "oa2", zero2)),
            oa2d3=_crop2_to_mass(base, _field_or(base, "oa3", zero2)),
            oa2d4=_crop2_to_mass(base, _field_or(base, "oa4", zero2)),
            ol2d1=_crop2_to_mass(base, _field_or(base, "ol1", zero2)),
            ol2d2=_crop2_to_mass(base, _field_or(base, "ol2", zero2)),
            ol2d3=_crop2_to_mass(base, _field_or(base, "ol3", zero2)),
            ol2d4=_crop2_to_mass(base, _field_or(base, "ol4", zero2)),
            sina=_crop2_to_mass(base, _field_or(base, "sina", _field_or(base, "sinalpha", zero2))),
            cosa=_crop2_to_mass(base, _field_or(base, "cosa", _field_or(base, "cosalpha", ones2))),
            znu=base["znu"],
            znw=base["znw"],
            p_top=_scalar_float(base, "p_top"),
            cp=float(CP),
            g=float(G),
            rd=float(R_D),
            rv=461.6,
            ep1=ep1,
            pi=3.141592653,
            dt=float(dtbl),
            dx=float(base["grid"].dx),
            kpbl2d=_crop2_to_mass(base, kpbl),
            itimestep=_runtime_itimestep(base),
            kpblmax_static=base.get("_static_gwdo_kpblmax"),
            gwdo_active_i=base.get("_static_gwdo_active_i"),
            gwdo_active_j=base.get("_static_gwdo_active_j"),
            gwdo_row_j=base.get("_static_gwdo_row_j"),
            gwdo_row_i_start=base.get("_static_gwdo_row_i_start"),
            gwdo_row_len=base.get("_static_gwdo_row_len"),
            gwdo_row_max_len=base.get("_static_gwdo_row_max_len"),
            rounding_token=base.get("_wrf_real_rounding_token"),
            ids=base["ids"],
            ide=base["ide"],
            jds=base["jds"],
            jde=base["jde"],
            kds=base["kds"],
            kde=base["kde"],
            ims=base["ims"],
            ime=base["ime"],
            jms=base["jms"],
            jme=base["jme"],
            kms=base["kms"],
            kme=base["kme"],
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
            kts=kts,
            kte=kte,
        )
        return {
            "rublten": rublten_out,
            "rvblten": rvblten_out,
            "dtaux3d": dtaux3d_out,
            "dtauy3d": dtauy3d_out,
            "dusfcg": dusfcg_out,
            "dvsfcg": dvsfcg_out,
        }

    return _call


def _pbl_should_run_runtime(state: Mapping[str, Any]) -> tuple[bool, float | None]:
    stepbl = int(state["config_flags"].stepbl)
    bldt = float(state["config_flags"].bldt)
    bldtacttime = state.get("bldtacttime")
    curr_secs = state.get("curr_secs")
    adapt_step_flag = bool(state.get("adapt_step_flag", False))
    if adapt_step_flag and curr_secs is not None and bldtacttime == 0.0:
        bldtacttime = float(curr_secs) + bldt * 60.0
    if bldt == 0.0 or stepbl == 1:
        return True, bldtacttime
    itimestep = _runtime_itimestep(state)
    if itimestep == 1:
        return True, bldtacttime
    if (not adapt_step_flag) and itimestep % stepbl == 0:
        return True, bldtacttime
    if adapt_step_flag and curr_secs is not None and bldtacttime is not None and float(curr_secs) >= float(bldtacttime):
        return True, float(curr_secs) + bldt * 60.0
    return False, bldtacttime


def _pbl_dtbl_runtime(state: Mapping[str, Any]) -> float:
    dt = float(state["grid"].dt)
    bldt = float(state["config_flags"].bldt)
    if bldt == 0.0:
        return dt
    if bool(state.get("adapt_step_flag", False)):
        return bldt * 60.0
    return dt * float(state["config_flags"].stepbl)


def _state_dtbc(state: Mapping[str, Any]) -> Any:
    """Return the carried boundary-clock value instead of stale grid mutation."""

    if "dtbc" in state:
        return state["dtbc"]
    return getattr(state["grid"], "dtbc", state["grid"].dt)


def _runtime_itimestep(state: Mapping[str, Any]) -> Any:
    """Return the current model step from explicit state, falling back to grid."""

    if "_physics_itimestep" in state:
        return int(state["_physics_itimestep"])
    value = state.get("itimestep", getattr(state["grid"], "itimestep", 0))
    if isinstance(value, (int, np.integer)):
        return int(value)
    array = jnp.asarray(value, dtype=jnp.int32)
    if isinstance(array, jax.core.Tracer):
        return array
    return int(np.asarray(array))


_FIXED_MYJ_PACK_NAMES = (
    "u_phytmp",
    "v_phytmp",
    "psfc",
    "qv",
    "qcw",
    "qci",
    "zero3",
    "ht",
    "dz",
    "pmid",
    "pint",
    "th",
    "t",
    "exner",
    "u",
    "v",
    "tsk",
    "qsfc",
    "chklowq",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "lowlyr",
    "xland",
    "sice",
    "snow",
    "tke_myj",
    "exch_h",
    "ustar",
    "znt",
    "el_myj",
    "pblh",
    "kpbl",
    "ct",
    "akhs",
    "akms",
    "elflx",
    "mixht",
    "u_gwd",
    "v_gwd",
    "t_gwd",
    "p_gwd",
    "p8w_gwd",
    "pi_gwd",
    "z_gwd",
    "dtaux3d",
    "dtauy3d",
    "dusfcg",
    "dvsfcg",
    "var",
    "oc12d",
    "oa1",
    "oa2",
    "oa3",
    "oa4",
    "ol1",
    "ol2",
    "ol3",
    "ol4",
    "sina",
    "cosa",
)


_INLINE_COLUMN_PHYSICS_JITS = _env_flag("WRF_PY_INLINE_COLUMN_PHYSICS_JITS", False)


@partial(jax.jit, static_argnames=("ni", "nj"), inline=_INLINE_COLUMN_PHYSICS_JITS)
def _fixed_myj_pack_inputs(
    *,
    u_phy: Any,
    v_phy: Any,
    t_phy: Any,
    moist_qv: Any,
    moist_qc: Any,
    moist_qi: Any,
    hgt: Any,
    dz8w: Any,
    p_hyd: Any,
    p8w: Any,
    th_phy: Any,
    pi_phy: Any,
    z: Any,
    tsk: Any,
    qsfc: Any,
    chklowq: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    xice: Any,
    snow: Any,
    tke_pbl: Any,
    exch_h: Any,
    ust: Any,
    znt: Any,
    el_pbl: Any,
    pblh: Any,
    kpbl: Any,
    ct: Any,
    akhs: Any,
    akms: Any,
    elflx: Any,
    mixht: Any,
    dtaux3d: Any,
    dtauy3d: Any,
    dusfcg: Any,
    dvsfcg: Any,
    var: Any,
    oc12d: Any,
    oa1: Any,
    oa2: Any,
    oa3: Any,
    oa4: Any,
    ol1: Any,
    ol2: Any,
    ol3: Any,
    ol4: Any,
    sina: Any,
    cosa: Any,
    u_frame: Any,
    v_frame: Any,
    ni: int,
    nj: int,
) -> tuple[Any, ...]:
    """Pack active-domain MYJ/GWDO inputs in one compiled slice kernel."""

    def c2(value: Any) -> Any:
        return jnp.asarray(value)[:ni, :nj]

    def c3(value: Any) -> Any:
        return jnp.asarray(value)[:ni, :, :nj]

    u_phytmp = jnp.asarray(u_phy) + jnp.asarray(u_frame, dtype=jnp.asarray(u_phy).dtype)
    v_phytmp = jnp.asarray(v_phy) + jnp.asarray(v_frame, dtype=jnp.asarray(v_phy).dtype)
    qv = c3(moist_qv)
    qcw = c3(moist_qc)
    qci = c3(moist_qi)
    zero3 = jnp.zeros_like(qv)
    p8w_arr = jnp.asarray(p8w)
    psfc = p8w_arr[:, 0, :]

    return (
        u_phytmp,
        v_phytmp,
        psfc,
        qv,
        qcw,
        qci,
        zero3,
        c2(hgt),
        c3(dz8w),
        c3(p_hyd),
        c3(p8w_arr),
        c3(th_phy),
        c3(t_phy),
        c3(pi_phy),
        c3(u_phytmp),
        c3(v_phytmp),
        c2(tsk),
        c2(qsfc),
        c2(chklowq),
        c2(thz0),
        c2(qz0),
        c2(uz0),
        c2(vz0),
        c2(lowlyr),
        c2(xland),
        c2(xice),
        c2(snow),
        c3(tke_pbl),
        c3(exch_h),
        c2(ust),
        c2(znt),
        c3(el_pbl),
        c2(pblh),
        c2(kpbl),
        c2(ct),
        c2(akhs),
        c2(akms),
        c2(elflx),
        c2(mixht),
        c3(u_phy),
        c3(v_phy),
        c3(t_phy),
        c3(p_hyd),
        c3(p8w_arr),
        c3(pi_phy),
        c3(z),
        c3(dtaux3d),
        c3(dtauy3d),
        c2(dusfcg),
        c2(dvsfcg),
        c2(var),
        c2(oc12d),
        c2(oa1),
        c2(oa2),
        c2(oa3),
        c2(oa4),
        c2(ol1),
        c2(ol2),
        c2(ol3),
        c2(ol4),
        c2(sina),
        c2(cosa),
    )


def _fixed_myj_pbl_stage(state: dict[str, Any]) -> dict[str, Any] | None:
    if not _FAST_FIXED_MYJ_PBL_STAGE:
        return None
    if pbl_driver is not _PBL_DRIVER_IMPL:
        return None
    config = state["config_flags"]
    if int(config.bl_pbl_physics) != 2:
        return None
    if int(getattr(config, "windfarm_opt", 0)) != 0:
        return None
    gwd_opt = int(getattr(config, "gwd_opt", 0))
    if gwd_opt not in (0, 1):
        return None

    run_param, bldtacttime = _pbl_should_run_runtime(state)
    if not run_param:
        return None

    dt = float(state["grid"].dt)
    stepbl = int(config.stepbl)
    dtbl = _pbl_dtbl_runtime(state)
    its, ite, jts, jte, kts, kte = _surface_bounds(state)

    zero2 = _zeros2(state)
    full01_2 = _full2(state, 0.1)
    minus_one2 = jnp.full(_shape2(state), -1.0, dtype=jnp.float32)
    ones2_i = jnp.ones(_shape2(state), dtype=jnp.int32)
    zeros2_i = jnp.zeros(_shape2(state), dtype=jnp.int32)
    tke_default = jnp.full_like(state["t_phy"], float(EPSQ2))
    zero_t_phy = jnp.zeros_like(state["t_phy"])
    moist_arr = jnp.asarray(state["moist"])
    qv_full = moist_arr[..., 1]
    zero_qv_full = jnp.zeros_like(qv_full)
    qcw_full = _moist_component(moist_arr, 2, zero_qv_full)
    qci_full = _moist_component(moist_arr, 4, zero_qv_full)
    p8w = _field_or(state, "p_hyd_w", state["p8w"])
    pblh_in = _field_or(state, "pblh", zero2)
    kpbl_in = _field_or(state, "kpbl", zeros2_i)
    profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
    packed_values = _fixed_myj_pack_inputs(
                u_phy=state["u_phy"],
                v_phy=state["v_phy"],
                t_phy=state["t_phy"],
                moist_qv=qv_full,
                moist_qc=qcw_full,
                moist_qi=qci_full,
                hgt=state["hgt"],
                dz8w=state["dz8w"],
                p_hyd=_field_or(state, "p_hyd", state["p_phy"]),
                p8w=p8w,
                th_phy=state["th_phy"],
                pi_phy=state["pi_phy"],
                z=state["z"],
                tsk=state["tsk"],
                qsfc=_field_or(state, "qsfc", zero2),
                chklowq=_field_or(state, "chklowq", minus_one2),
                thz0=_field_or(state, "thz0", _field_or(state, "th2", zero2)),
                qz0=_field_or(state, "qz0", _field_or(state, "q2", zero2)),
                uz0=_field_or(state, "uz0", _field_or(state, "u10", zero2)),
                vz0=_field_or(state, "vz0", _field_or(state, "v10", zero2)),
                lowlyr=_field_or(state, "lowlyr", ones2_i),
                xland=state["xland"],
                xice=_field_or(state, "xice", zero2),
                snow=_field_or(state, "snow", zero2),
                tke_pbl=_field_or(state, "tke_pbl", tke_default),
                exch_h=_field_or(state, "exch_h", zero_t_phy),
                ust=_field_or(state, "ust", full01_2),
                znt=_field_or(state, "znt", full01_2),
                el_pbl=_field_or(state, "el_pbl", zero_t_phy),
                pblh=pblh_in,
                kpbl=kpbl_in,
                ct=_field_or(state, "ct", zero2),
                akhs=_field_or(state, "akhs", zero2),
                akms=_field_or(state, "akms", zero2),
                elflx=_myjpbl_elflx_field(state, zero2),
                mixht=_field_or(state, "mixht", zero2),
                dtaux3d=_field_or(state, "dtaux3d", zero_qv_full),
                dtauy3d=_field_or(state, "dtauy3d", zero_qv_full),
                dusfcg=_field_or(state, "dusfcg", zero2),
                dvsfcg=_field_or(state, "dvsfcg", zero2),
                var=_field_or(state, "var", zero2),
                oc12d=_field_or(state, "oc12d", _field_or(state, "con", zero2)),
                oa1=_field_or(state, "oa1", zero2),
                oa2=_field_or(state, "oa2", zero2),
                oa3=_field_or(state, "oa3", zero2),
                oa4=_field_or(state, "oa4", zero2),
                ol1=_field_or(state, "ol1", zero2),
                ol2=_field_or(state, "ol2", zero2),
                ol3=_field_or(state, "ol3", zero2),
                ol4=_field_or(state, "ol4", zero2),
                sina=_field_or(state, "sina", _field_or(state, "sinalpha", zero2)),
                cosa=_field_or(state, "cosa", _field_or(state, "cosalpha", _full2(state, 1.0))),
                u_frame=_scalar_float_or(state, "u_frame", 0.0),
                v_frame=_scalar_float_or(state, "v_frame", 0.0),
                ni=int(state["ide"]) - 1,
                nj=int(state["jde"]) - 1,
            )
    _profile_surface_pbl_write("pack_myj_pbl_inputs", profile_start, packed_values)
    packed = dict(
        zip(
            _FIXED_MYJ_PACK_NAMES,
            packed_values,
            strict=True,
        )
    )
    qv = packed["qv"]
    qcw = packed["qcw"]
    qci = packed["qci"]
    zero3 = packed["zero3"]
    u_phytmp = packed["u_phytmp"]
    v_phytmp = packed["v_phytmp"]
    psfc = packed["psfc"]

    profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
    myj = myjpbl(
        dt=dt,
        stepbl=stepbl,
        ht=packed["ht"],
        dz=packed["dz"],
        pmid=packed["pmid"],
        pint=packed["pint"],
        th=packed["th"],
        t=packed["t"],
        exner=packed["exner"],
        qv=qv,
        qcw=qcw,
        qci=qci,
        qcs=zero3,
        qcr=zero3,
        qcg=zero3,
        u=packed["u"],
        v=packed["v"],
        rho=state["rho"],
        tsk=packed["tsk"],
        qsfc=packed["qsfc"],
        chklowq=packed["chklowq"],
        thz0=packed["thz0"],
        qz0=packed["qz0"],
        uz0=packed["uz0"],
        vz0=packed["vz0"],
        lowlyr=packed["lowlyr"],
        xland=packed["xland"],
        sice=packed["sice"],
        snow=packed["snow"],
        tke_myj=packed["tke_myj"],
        exch_h=packed["exch_h"],
        ustar=packed["ustar"],
        znt=packed["znt"],
        el_myj=packed["el_myj"],
        pblh=packed["pblh"],
        kpbl=packed["kpbl"],
        ct=packed["ct"],
        akhs=packed["akhs"],
        akms=packed["akms"],
        elflx=packed["elflx"],
        mixht=packed["mixht"],
        rublten=zero3,
        rvblten=zero3,
        rthblten=zero3,
        rqvblten=zero3,
        rqcblten=zero3,
        rqiblten=zero3,
        rqsblten=None,
        rqrblten=None,
        rqgblten=None,
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
        rounding_token=state.get("_wrf_real_rounding_token"),
    )
    _profile_surface_pbl_write("myjpbl", profile_start, myj)

    updates: dict[str, Any] = {
        "u_phytmp": u_phytmp,
        "v_phytmp": v_phytmp,
        "tskold": state.get("tsk"),
        "ustold": state.get("ust"),
        "zntold": state.get("znt"),
        "hpbl_hold": pblh_in,
        "psfc": psfc,
        "rublten": myj.rublten,
        "rvblten": myj.rvblten,
        "rthblten": myj.rthblten,
        "rqvblten": myj.rqvblten,
        "rqcblten": myj.rqcblten,
        "rqiblten": myj.rqiblten,
        "pblh": myj.pblh,
        "kpbl": myj.kpbl,
        "bldtacttime": bldtacttime,
        "qsfc": myj.qsfc,
        "thz0": myj.thz0,
        "qz0": myj.qz0,
        "uz0": myj.uz0,
        "vz0": myj.vz0,
        "znt": myj.znt,
        "exch_h": myj.exch_h,
        "ct": myj.ct,
        "akhs": myj.akhs,
        "akms": myj.akms,
        "mixht": myj.mixht,
        "ust": myj.ustar,
        "tke_pbl": myj.tke_myj,
        "el_pbl": myj.el_myj,
    }

    if gwd_opt == 1:
        ep1 = float(P608)
        profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
        (
            rublten_out,
            rvblten_out,
            dtaux3d_out,
            dtauy3d_out,
            dusfcg_out,
            dvsfcg_out,
        ) = gwdo(
            u3d=packed["u_gwd"],
            v3d=packed["v_gwd"],
            t3d=packed["t_gwd"],
            qv3d=qv,
            p3d=packed["p_gwd"],
            p3di=packed["p8w_gwd"],
            pi3d=packed["pi_gwd"],
            z=packed["z_gwd"],
            rublten=myj.rublten,
            rvblten=myj.rvblten,
            dtaux3d=packed["dtaux3d"],
            dtauy3d=packed["dtauy3d"],
            dusfcg=packed["dusfcg"],
            dvsfcg=packed["dvsfcg"],
            var2d=packed["var"],
            oc12d=packed["oc12d"],
            oa2d1=packed["oa1"],
            oa2d2=packed["oa2"],
            oa2d3=packed["oa3"],
            oa2d4=packed["oa4"],
            ol2d1=packed["ol1"],
            ol2d2=packed["ol2"],
            ol2d3=packed["ol3"],
            ol2d4=packed["ol4"],
            sina=packed["sina"],
            cosa=packed["cosa"],
            znu=state["znu"],
            znw=state["znw"],
            p_top=_scalar_float(state, "p_top"),
            cp=float(CP),
            g=float(G),
            rd=float(R_D),
            rv=461.6,
            ep1=ep1,
            pi=3.141592653,
            dt=float(dtbl),
            dx=float(state["grid"].dx),
            kpbl2d=myj.kpbl,
            itimestep=_runtime_itimestep(state),
            kpblmax_static=state.get("_static_gwdo_kpblmax"),
            gwdo_active_i=state.get("_static_gwdo_active_i"),
            gwdo_active_j=state.get("_static_gwdo_active_j"),
            gwdo_row_j=state.get("_static_gwdo_row_j"),
            gwdo_row_i_start=state.get("_static_gwdo_row_i_start"),
            gwdo_row_len=state.get("_static_gwdo_row_len"),
            gwdo_row_max_len=state.get("_static_gwdo_row_max_len"),
            rounding_token=state.get("_wrf_real_rounding_token"),
            ids=state["ids"],
            ide=state["ide"],
            jds=state["jds"],
            jde=state["jde"],
            kds=state["kds"],
            kde=state["kde"],
            ims=state["ims"],
            ime=state["ime"],
            jms=state["jms"],
            jme=state["jme"],
            kms=state["kms"],
            kme=state["kme"],
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
            kts=kts,
            kte=kte,
        )
        _profile_surface_pbl_write("gwdo", profile_start, (rublten_out, rvblten_out, dtaux3d_out, dtauy3d_out, dusfcg_out, dvsfcg_out))
        updates.update(
            {
                "rublten": rublten_out,
                "rvblten": rvblten_out,
                "dtaux3d": dtaux3d_out,
                "dtauy3d": dtauy3d_out,
                "dusfcg": dusfcg_out,
                "dvsfcg": dvsfcg_out,
            }
        )
    return updates


def _pbl_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    runtime_state = state
    cached_local = state.get("_physics_local_after_surface")
    state = dict(cached_local) if isinstance(cached_local, dict) else _physics_local_state(state)
    defaults = _surface_default_updates(state)
    if defaults:
        state = dict(state)
        state.update(defaults)
    fast_updates = _fixed_myj_pbl_stage(state)
    if fast_updates is not None:
        updates = dict(defaults)
        updates.update(fast_updates)
        padded = _pad_physics_updates_to_runtime(runtime_state, updates)
        padded["_physics_local_after_surface"] = None
        return padded
    zero3 = jnp.zeros_like(_crop3_to_mass(state, jnp.asarray(state["moist"])[..., 0]))
    its, ite, jts, jte, kts, kte = _surface_bounds(state)
    result = pbl_driver(
        bl_pbl_physics=int(state["config_flags"].bl_pbl_physics),
        itimestep=_runtime_itimestep(state),
        dt=float(state["grid"].dt),
        u_frame=_scalar_float_or(state, "u_frame", 0.0),
        v_frame=_scalar_float_or(state, "v_frame", 0.0),
        bldt=float(state["config_flags"].bldt),
        curr_secs=state["curr_secs"],
        adapt_step_flag=state["adapt_step_flag"],
        bldtacttime=state.get("bldtacttime"),
        stepbl=int(state["config_flags"].stepbl),
        ra_lw_physics=int(state["config_flags"].ra_lw_physics),
        sf_sfclay_physics=int(state["config_flags"].sf_sfclay_physics),
        sf_urban_physics=int(state["config_flags"].sf_urban_physics),
        gwd_opt=int(state["config_flags"].gwd_opt),
        tsk=state.get("tsk"),
        znt=state.get("znt"),
        ust=state.get("ust"),
        pblh=state.get("pblh"),
        hpbl_hold=state.get("hpbl_hold"),
        u_phy=state["u_phy"],
        v_phy=state["v_phy"],
        u_phytmp=state.get("u_phytmp"),
        v_phytmp=state.get("v_phytmp"),
        p8w=_field_or(state, "p_hyd_w", state["p8w"]),
        psfc=state.get("psfc"),
        rublten=state.get("rublten", zero3),
        rvblten=state.get("rvblten", zero3),
        rthblten=state.get("rthblten", zero3),
        rqvblten=state.get("rqvblten", zero3),
        rqcblten=state.get("rqcblten", zero3),
        rqiblten=state.get("rqiblten"),
        qv_curr=jnp.asarray(state["moist"])[..., 1],
        qc_curr=_moist_component(state["moist"], 2, zero3),
        qi_curr=_moist_component(state["moist"], 4, zero3),
        f_qi=jnp.asarray(state["moist"]).shape[-1] > 4,
        kpbl=state.get("kpbl"),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        i_start=(its,),
        i_end=(ite,),
        j_start=(jts,),
        j_end=(jte,),
        kts=kts,
        kte=kte,
        myjpbl_fn=_runtime_myjpbl_callback(state),
        gwd_fn=_runtime_gwd_callback(state),
    )
    updates = dict(defaults)
    updates.update(dict(result.carried_fields))
    updates.update(
        {
            "u_phytmp": result.u_phytmp,
            "v_phytmp": result.v_phytmp,
            "tskold": result.tskold,
            "ustold": result.ustold,
            "zntold": result.zntold,
            "hpbl_hold": result.hpbl_hold,
            "psfc": result.psfc,
            "rublten": result.rublten,
            "rvblten": result.rvblten,
            "rthblten": result.rthblten,
            "rqvblten": result.rqvblten,
            "rqcblten": result.rqcblten,
            "rqiblten": result.rqiblten,
            "pblh": result.pblh,
            "kpbl": result.kpbl,
            "bldtacttime": result.bldtacttime,
        }
    )
    padded = _pad_physics_updates_to_runtime(runtime_state, updates)
    padded["_physics_local_after_surface"] = None
    return padded


def _fixed_myj_surface_layer_local_updates(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    """Run the fixed MYJ surface-layer phase without fusing Noah or PBL."""

    if not _FAST_FIXED_MYJ_NOAH_SURFACE_STAGE:
        return None
    config = state["config_flags"]
    if int(getattr(config, "sf_sfclay_physics", 0)) != 2:
        return None
    if int(getattr(config, "sf_surface_physics", 0)) != 2:
        return None
    if int(getattr(config, "isfflx", 1)) == 0:
        return None

    run_param, bldtacttime = _pbl_should_run_runtime(state)
    if not run_param:
        return None

    local_state = _physics_local_state(state)
    defaults = _surface_default_updates(local_state)
    local_state = _overlay_fields(local_state, defaults)

    k0 = int(local_state.get("kts", local_state["kms"])) - int(local_state["kms"])
    zero2 = _zeros2(local_state)
    u_phy = jnp.asarray(local_state["u_phy"])
    v_phy = jnp.asarray(local_state["v_phy"])
    u_phytmp = u_phy + jnp.asarray(_scalar_float_or(local_state, "u_frame", 0.0), dtype=u_phy.dtype)
    v_phytmp = v_phy + jnp.asarray(_scalar_float_or(local_state, "v_frame", 0.0), dtype=v_phy.dtype)
    if local_state.get("uoce") is not None:
        u_phytmp = u_phytmp.at[:, k0, :].add(-jnp.asarray(local_state["uoce"], dtype=u_phytmp.dtype))
    if local_state.get("voce") is not None:
        v_phytmp = v_phytmp.at[:, k0, :].add(-jnp.asarray(local_state["voce"], dtype=v_phytmp.dtype))
    psfc = jnp.asarray(_field_or(local_state, "p_hyd_w", local_state["p8w"]))[:, k0, :]

    qgh_out = zero2
    chs_out = zero2
    cpm_out = zero2
    chs2_out = zero2

    profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
    myjsfc_updates = dict(
        _runtime_myjsfc_callback(local_state)(
            u_phytmp=u_phytmp,
            v_phytmp=v_phytmp,
            psfc=psfc,
            qgh=qgh_out,
            chs=chs_out,
            cpm=cpm_out,
            chs2=chs2_out,
            ust=local_state.get("ust"),
            fields={},
        )
    )
    _profile_surface_pbl_write("myjsfc", profile_start, myjsfc_updates)
    qgh_out = myjsfc_updates.get("qgh", qgh_out)
    chs_out = myjsfc_updates.get("chs", chs_out)
    cpm_out = myjsfc_updates.get("cpm", cpm_out)
    chs2_out = myjsfc_updates.get("chs2", chs2_out)
    ust_out = myjsfc_updates.get("ust", local_state.get("ust"))

    ustm_out = ust_out
    rounding_token = local_state.get("_wrf_real_rounding_token")
    def store_wrf_real(value: Any) -> Any:
        value32 = jnp.asarray(value, dtype=jnp.float32)
        if rounding_token is None:
            return jax.lax.optimization_barrier(value32)
        return _materialize_wrf_real(value32, rounding_token)

    u_surface_squared = store_wrf_real(u_phy[:, k0, :] ** 2)
    v_surface_squared = store_wrf_real(v_phy[:, k0, :] ** 2)
    wind_squared = store_wrf_real(
        u_surface_squared + v_surface_squared,
    )
    wspd_out = jnp.maximum(
        store_wrf_real(sqrt_for_wrf(wind_squared)),
        jnp.asarray(0.001, dtype=u_phy.dtype),
    )
    ch_out = chs_out

    updates = dict(defaults)
    updates.update(myjsfc_updates)
    updates.update(
        {
            "q_ref2m": 0.0,
            "u_phytmp": u_phytmp,
            "v_phytmp": v_phytmp,
            "qgh": qgh_out,
            "chs": chs_out,
            "cpm": cpm_out,
            "chs2": chs2_out,
            "psfc": psfc,
            "ust": ust_out,
            "ustm": ustm_out,
            "wspd": wspd_out,
            "ch": ch_out,
            "bldtacttime": bldtacttime,
        }
    )
    local_after_surface_layer = _overlay_fields(local_state, updates)
    return updates, local_after_surface_layer


def _fixed_noah_land_surface_local_updates(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    """Run Noah LSM and surface diagnostics after the MYJ layer phase."""

    if not _FAST_FIXED_MYJ_NOAH_SURFACE_STAGE:
        return None
    config = state["config_flags"]
    if int(getattr(config, "sf_sfclay_physics", 0)) != 2:
        return None
    if int(getattr(config, "sf_surface_physics", 0)) != 2:
        return None
    if int(getattr(config, "isfflx", 1)) == 0:
        return None

    run_param, _bldtacttime = _pbl_should_run_runtime(state)
    if not run_param:
        return None

    local_state = _physics_local_state(state)
    defaults = _surface_default_updates(local_state)
    local_state = _overlay_fields(local_state, defaults)
    rain = accumulate_surface_precipitation(
        _field_or(local_state, "rainbl", _zeros2(local_state)),
        local_state.get("raincv", 0.0), local_state.get("rainncv"), local_state.get("rainshv"),
    )
    local_state = _overlay_fields(local_state, {"rainbl": rain})
    k0 = int(local_state.get("kts", local_state["kms"])) - int(local_state["kms"])
    u_phytmp = jnp.asarray(local_state["u_phytmp"])
    v_phytmp = jnp.asarray(local_state["v_phytmp"])
    psfc = _field_or(
        local_state,
        "psfc",
        jnp.asarray(_field_or(local_state, "p_hyd_w", local_state["p8w"]))[:, k0, :],
    )
    qgh = local_state["qgh"]
    chs = local_state["chs"]
    cpm = local_state["cpm"]
    chs2 = local_state["chs2"]

    profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
    lsm_updates = dict(
        _runtime_lsm_callback(local_state)(
            u_phytmp=u_phytmp,
            v_phytmp=v_phytmp,
            psfc=psfc,
            qgh=qgh,
            chs=chs,
            cpm=cpm,
            chs2=chs2,
            dtbl=_pbl_dtbl_runtime(local_state),
            fields={},
        )
    )
    _profile_surface_pbl_write("lsm", profile_start, lsm_updates)

    profile_start = perf_counter() if _PROFILE_SURFACE_PBL_PATH else 0.0
    sfcdiags_updates = dict(
        _runtime_sfcdiags_callback(local_state)(
            psfc=psfc,
            chs=chs,
            chs2=chs2,
            fields=lsm_updates,
        )
    )
    _profile_surface_pbl_write("sfcdiags", profile_start, sfcdiags_updates)

    updates = dict(defaults)
    updates.update(lsm_updates)
    updates.update(sfcdiags_updates)
    updates["rainbl"] = jnp.zeros_like(rain)
    local_after_surface = _overlay_fields(local_state, updates)
    return updates, local_after_surface


def _fixed_myj_noah_surface_local_updates(state: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    """Specialized MYJ + Noah path composed from explicit science phases."""

    surface_layer = _fixed_myj_surface_layer_local_updates(state)
    if surface_layer is None:
        return None
    layer_updates, local_after_layer = surface_layer
    land_surface = _fixed_noah_land_surface_local_updates(local_after_layer)
    if land_surface is None:
        return None
    land_updates, local_after_surface = land_surface
    updates = dict(layer_updates)
    updates.update(land_updates)
    return updates, local_after_surface


def _surface_driver_local_updates(state: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Run surface physics in local physics bounds and return unpadded updates."""

    fixed_updates = _fixed_myj_noah_surface_local_updates(state)
    if fixed_updates is not None:
        return fixed_updates

    local_state = _physics_local_state(state)
    defaults = _surface_default_updates(local_state)
    local_state = _overlay_fields(local_state, defaults)
    its, ite, jts, jte, kts, kte = _surface_bounds(local_state)
    result = surface_driver(
        sf_sfclay_physics=int(local_state["config_flags"].sf_sfclay_physics),
        sf_surface_physics=int(local_state["config_flags"].sf_surface_physics),
        bl_pbl_physics=int(local_state["config_flags"].bl_pbl_physics),
        ra_lw_physics=int(local_state["config_flags"].ra_lw_physics),
        fractional_seaice=int(getattr(local_state["config_flags"], "fractional_seaice", 0)),
        itimestep=_runtime_itimestep(local_state),
        stepbl=int(local_state["config_flags"].stepbl),
        bldt=float(local_state["config_flags"].bldt),
        dt=float(local_state["grid"].dt),
        curr_secs=local_state["curr_secs"],
        adapt_step_flag=local_state["adapt_step_flag"],
        bldtacttime=local_state.get("bldtacttime"),
        isfflx=int(local_state["config_flags"].isfflx),
        u_phy=local_state["u_phy"],
        v_phy=local_state["v_phy"],
        u_phytmp=local_state.get("u_phytmp"),
        v_phytmp=local_state.get("v_phytmp"),
        u_frame=_scalar_float_or(local_state, "u_frame", 0.0),
        v_frame=_scalar_float_or(local_state, "v_frame", 0.0),
        uoce=local_state.get("uoce"),
        voce=local_state.get("voce"),
        p8w=_field_or(local_state, "p_hyd_w", local_state["p8w"]),
        psfc=local_state.get("psfc"),
        qgh=local_state.get("qgh"),
        chs=local_state.get("chs"),
        cpm=local_state.get("cpm"),
        chs2=local_state.get("chs2"),
        ust=local_state.get("ust"),
        ustm=local_state.get("ustm"),
        wspd=local_state.get("wspd"),
        ch=local_state.get("ch"),
        rainbl=local_state.get("rainbl"),
        raincv=local_state.get("raincv", 0.0),
        rainncv=local_state.get("rainncv"),
        rainshv=local_state.get("rainshv"),
        ids=local_state["ids"],
        ide=local_state["ide"],
        jds=local_state["jds"],
        jde=local_state["jde"],
        kds=local_state["kds"],
        kde=local_state["kde"],
        ims=local_state["ims"],
        ime=local_state["ime"],
        jms=local_state["jms"],
        jme=local_state["jme"],
        kms=local_state["kms"],
        kme=local_state["kme"],
        i_start=(its,),
        i_end=(ite,),
        j_start=(jts,),
        j_end=(jte,),
        kts=kts,
        kte=kte,
        myjsfc_fn=_runtime_myjsfc_callback(local_state),
        lsm_fn=_runtime_lsm_callback(local_state),
        sfcdiags_fn=_runtime_sfcdiags_callback(local_state),
    )
    updates = dict(defaults)
    updates.update(dict(result.carried_fields))
    updates.update(
        {
            "q_ref2m": result.q_ref2m,
            "u_phytmp": result.u_phytmp,
            "v_phytmp": result.v_phytmp,
            "qgh": result.qgh,
            "chs": result.chs,
            "cpm": result.cpm,
            "chs2": result.chs2,
            "psfc": result.psfc,
            "ust": result.ust,
            "ustm": result.ustm,
            "wspd": result.wspd,
            "ch": result.ch,
            "bldtacttime": result.bldtacttime,
        }
    )
    local_after_surface = _overlay_fields(local_state, updates)
    return updates, local_after_surface


def _pbl_driver_local_updates(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run PBL physics in local physics bounds and return unpadded updates."""

    local_state = _physics_local_state(state)
    defaults = _surface_default_updates(local_state)
    local_state = _overlay_fields(local_state, defaults)
    fast_updates = _fixed_myj_pbl_stage(local_state)
    if fast_updates is not None:
        updates = dict(defaults)
        updates.update(fast_updates)
        return updates
    zero3 = jnp.zeros_like(_crop3_to_mass(local_state, jnp.asarray(local_state["moist"])[..., 0]))
    its, ite, jts, jte, kts, kte = _surface_bounds(local_state)
    result = pbl_driver(
        bl_pbl_physics=int(local_state["config_flags"].bl_pbl_physics),
        itimestep=_runtime_itimestep(local_state),
        dt=float(local_state["grid"].dt),
        u_frame=_scalar_float_or(local_state, "u_frame", 0.0),
        v_frame=_scalar_float_or(local_state, "v_frame", 0.0),
        bldt=float(local_state["config_flags"].bldt),
        curr_secs=local_state["curr_secs"],
        adapt_step_flag=local_state["adapt_step_flag"],
        bldtacttime=local_state.get("bldtacttime"),
        stepbl=int(local_state["config_flags"].stepbl),
        ra_lw_physics=int(local_state["config_flags"].ra_lw_physics),
        sf_sfclay_physics=int(local_state["config_flags"].sf_sfclay_physics),
        sf_urban_physics=int(local_state["config_flags"].sf_urban_physics),
        gwd_opt=int(local_state["config_flags"].gwd_opt),
        tsk=local_state.get("tsk"),
        znt=local_state.get("znt"),
        ust=local_state.get("ust"),
        pblh=local_state.get("pblh"),
        hpbl_hold=local_state.get("hpbl_hold"),
        u_phy=local_state["u_phy"],
        v_phy=local_state["v_phy"],
        u_phytmp=local_state.get("u_phytmp"),
        v_phytmp=local_state.get("v_phytmp"),
        p8w=_field_or(local_state, "p_hyd_w", local_state["p8w"]),
        psfc=local_state.get("psfc"),
        rublten=local_state.get("rublten", zero3),
        rvblten=local_state.get("rvblten", zero3),
        rthblten=local_state.get("rthblten", zero3),
        rqvblten=local_state.get("rqvblten", zero3),
        rqcblten=local_state.get("rqcblten", zero3),
        rqiblten=local_state.get("rqiblten"),
        qv_curr=jnp.asarray(local_state["moist"])[..., 1],
        qc_curr=_moist_component(local_state["moist"], 2, zero3),
        qi_curr=_moist_component(local_state["moist"], 4, zero3),
        f_qi=jnp.asarray(local_state["moist"]).shape[-1] > 4,
        kpbl=local_state.get("kpbl"),
        ids=local_state["ids"],
        ide=local_state["ide"],
        jds=local_state["jds"],
        jde=local_state["jde"],
        kds=local_state["kds"],
        kde=local_state["kde"],
        ims=local_state["ims"],
        ime=local_state["ime"],
        jms=local_state["jms"],
        jme=local_state["jme"],
        kms=local_state["kms"],
        kme=local_state["kme"],
        i_start=(its,),
        i_end=(ite,),
        j_start=(jts,),
        j_end=(jte,),
        kts=kts,
        kte=kte,
        myjpbl_fn=_runtime_myjpbl_callback(local_state),
        gwd_fn=_runtime_gwd_callback(local_state),
    )
    updates = dict(defaults)
    updates.update(dict(result.carried_fields))
    updates.update(
        {
            "u_phytmp": result.u_phytmp,
            "v_phytmp": result.v_phytmp,
            "tskold": result.tskold,
            "ustold": result.ustold,
            "zntold": result.zntold,
            "hpbl_hold": result.hpbl_hold,
            "psfc": result.psfc,
            "rublten": result.rublten,
            "rvblten": result.rvblten,
            "rthblten": result.rthblten,
            "rqvblten": result.rqvblten,
            "rqcblten": result.rqcblten,
            "rqiblten": result.rqiblten,
            "pblh": result.pblh,
            "kpbl": result.kpbl,
            "bldtacttime": result.bldtacttime,
        }
    )
    return updates


def surface_row_tile_metadata(fields: Mapping[str, Any], bounds: Mapping[str, int]) -> dict[str, Any]:
    """Reindex static sparse physics columns once when preparing a row tile."""

    updates = {}
    offset = int(bounds["jms"]) - 1
    stop = min(int(bounds["jme"]), int(bounds["jde"]) - 1)
    for scheme in ("gwdo", "noah", "seaice", "glacial"):
        prefix = f"_static_{scheme}_"
        for name in fields:
            if name.startswith(prefix) and name != "_static_gwdo_kpblmax":
                updates[name] = None
        iname, jname = prefix + "active_i", prefix + "active_j"
        if fields.get(iname) is None or fields.get(jname) is None:
            continue
        ii, jj = (np.asarray(jax.device_get(fields[name]), dtype=np.int32)
                  for name in (iname, jname))
        selected = (jj >= offset) & (jj < stop)
        updates[iname] = jnp.asarray(ii[selected])
        updates[jname] = jnp.asarray(jj[selected] - offset)
        updates[prefix + "active_count"] = int(selected.sum())
        updates[prefix + "total_count"] = (int(fields["ide"]) - 1) * (stop - offset)
    return updates


def _surface_pbl_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Direct-runtime fast path for adjacent surface and PBL stages."""

    parent = state
    nx, ny = int(state["ide"]), int(state["jde"])
    row_start = int(state["jms"])
    rows = int(state["jme"]) - row_start + 1
    row_tile = (row_start, int(state["jme"])) != (1, ny) and not _is_local_physics_bounds(state)
    if row_tile:
        state = dict(state)
        state.update(_local_physics_bounds(nx, rows + 1, int(state["kde"])))
    surface_updates, local_after_surface = _surface_driver_local_updates(state)
    pbl_updates = _pbl_driver_local_updates(local_after_surface)
    updates = dict(surface_updates)
    updates.update(pbl_updates)
    if _LOCAL_SURFACE_PBL_UPDATES:
        padded = dict(updates)
    else:
        padded = _pad_physics_updates_to_runtime(state, updates)
    if row_tile:
        for name, value in tuple(padded.items()):
            shape = getattr(value, "shape", ())
            if len(shape) < 2 or int(shape[0]) != nx:
                continue
            axis = 1 if len(shape) == 2 else 2
            if int(shape[axis]) not in (rows, rows + 1):
                continue
            selection = [slice(None)] * len(shape)
            selection[axis] = slice(0, rows)
            value = value[tuple(selection)]
            soil = name in ("sh2o", "smcrel", "smois", "tslb")
            value = value.at[-1].set(parent[name][-1] if soil else value[-2])
            edge = ny - row_start
            if 0 <= edge < rows:
                selection[axis] = edge
                previous = list(selection)
                previous[axis] = edge - 1
                source = parent[name][tuple(selection)] if soil else value[tuple(previous)]
                value = value.at[tuple(selection)].set(source)
            padded[name] = value
    padded["_physics_local_after_surface"] = None
    return padded


def _cumulus_driver_call_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    zero2 = _zeros2(state)
    zero3 = jnp.zeros_like(jnp.asarray(_field_or(state, "t_phy", state["t_2"])))
    zero3_memory = jnp.zeros_like(jnp.asarray(_field_or(state, "p_hyd_w", _field_or(state, "p8w", state["w_2"]))))

    def _to_memory_k(value: Any) -> Any:
        arr = jnp.asarray(value)
        if (
            arr.ndim == 3
            and arr.shape[0] == zero3_memory.shape[0]
            and arr.shape[2] == zero3_memory.shape[2]
            and arr.shape[1] == zero3_memory.shape[1] - 1
        ):
            return zero3_memory.at[:, : arr.shape[1], :].set(arr)
        if (
            arr.ndim == 3
            and arr.shape[0] == zero3_memory.shape[0]
            and arr.shape[2] == zero3_memory.shape[2]
            and arr.shape[1] == zero3_memory.shape[1] + 1
        ):
            return arr[:, : zero3_memory.shape[1], :]
        return arr

    xland = _field_or(state, "xland", zero2)
    zero2_mem = jnp.zeros_like(jnp.asarray(xland))
    znu = jnp.asarray(state["znu"])
    if znu.ndim == 1 and znu.shape[0] == zero3_memory.shape[1] - 1:
        znu = jnp.concatenate([znu, jnp.zeros((1,), dtype=znu.dtype)], axis=0)
    moist_arr = jnp.asarray(state["moist"]) if "moist" in state and state["moist"] is not None else None
    qv = _moist_component(moist_arr, 1, _field_or(state, "qvapor", zero3)) if moist_arr is not None else _field_or(state, "qvapor", zero3)
    if qv.shape != zero3.shape:
        qv = _field_or(state, "qvapor", qv)
    qc = _moist_component(moist_arr, 2, _field_or(state, "qcloud", zero3)) if moist_arr is not None else _field_or(state, "qcloud", zero3)
    qi = _moist_component(moist_arr, 4, _field_or(state, "qice", zero3)) if moist_arr is not None else _field_or(state, "qice", zero3)
    fields = {
        "dt": float(state["grid"].dt),
        "znu": znu,
        "qfx": _field_or(state, "qfx", zero2_mem),
        "u3d": _to_memory_k(_field_or(state, "u_phy", state["u_2"])),
        "v3d": _to_memory_k(_field_or(state, "v_phy", state["v_2"])),
        "w": _to_memory_k(state["w_2"]),
        "t3d": _to_memory_k(_field_or(state, "t_phy", state["t_2"])),
        "qv3d": _to_memory_k(qv),
        "qc3d": _to_memory_k(qc),
        "qi3d": _to_memory_k(qi),
        "pi3d": _to_memory_k(_field_or(state, "pi_phy", zero3)),
        "rho3d": _to_memory_k(_field_or(state, "rho", zero3)),
        "qvften": _to_memory_k(_field_or(state, "rqvften", zero3)),
        "qvpblten": _to_memory_k(_field_or(state, "rqvblten", zero3)),
        "dz8w": _to_memory_k(_field_or(state, "dz8w", zero3_memory)),
        "pcps": _to_memory_k(_field_or(state, "p_hyd", _field_or(state, "p_phy", zero3))),
        "p8w": _to_memory_k(_field_or(state, "p_hyd_w", _field_or(state, "p8w", zero3_memory))),
        "xland": xland,
        "cu_act_flag": _field_or(state, "cu_act_flag", jnp.ones_like(zero2_mem, dtype=jnp.int32)),
        "raincv": _field_or(state, "raincv", zero2_mem),
        "rthcuten": _to_memory_k(_field_or(state, "rthcuten", zero3)),
        "rqvcuten": _to_memory_k(_field_or(state, "rqvcuten", zero3)),
        "rqccuten": _to_memory_k(_field_or(state, "rqccuten", zero3)),
        "rqicuten": _to_memory_k(_field_or(state, "rqicuten", zero3)),
        "rucuten": _to_memory_k(_field_or(state, "rucuten", zero3)),
        "rvcuten": _to_memory_k(_field_or(state, "rvcuten", zero3)),
        "rounding_token": state.get("_wrf_real_rounding_token"),
        # A spatially resident shard is already one complete Tiedtke row
        # block.  The driver consumes this marker to prevent re-entering the
        # host-oriented worker chunker inside the shard.
        "_tiedtke_resident_chunk": bool(state.get("_tiedtke_resident_chunk", False)),
        "f_qv": True,
        "f_qc": True,
        "f_qr": True,
        "f_qi": True,
        "f_qs": True,
        "ids": state["ids"],
        "ide": state["ide"],
        "jds": state["jds"],
        "jde": state["jde"],
        "kds": state["kds"],
        "kde": state["kde"],
        "ims": state["ims"],
        "ime": state["ime"],
        "jms": state["jms"],
        "jme": state["jme"],
        "kms": state["kms"],
        "kme": state["kme"],
        "its": state.get("its", state["ids"]),
        "ite": min(int(state.get("ite", state["ide"] - 1)), int(state["ide"]) - 1),
        "jts": state.get("jts", state["jds"]),
        "jte": min(int(state.get("jte", state["jde"] - 1)), int(state["jde"]) - 1),
        "kts": state["kps"],
        "kte": min(int(state["kpe"]), int(state["kde"]) - 1),
    }
    return fields


def _cumulus_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    fields = _cumulus_driver_call_fields(state)
    result = cumulus_driver(
        cu_physics=int(state["config_flags"].cu_physics),
        itimestep=_runtime_itimestep(state),
        stepcu=int(state["config_flags"].stepcu),
        cudt=float(state["config_flags"].cudt),
        curr_secs=state["curr_secs"],
        adapt_step_flag=state["adapt_step_flag"],
        cudtacttime=state.get("cudtacttime"),
        pratec=state.get("pratec"),
        **fields,
    )
    updates = dict(result.carried_fields)
    updates["pratec"] = result.pratec
    updates["cudtacttime"] = result.cudtacttime
    return updates


def _shallowcu_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = shallowcu_driver(
        shcu_physics=int(state["config_flags"].shcu_physics),
        pratesh=state.get("pratesh"),
        rainsh=state.get("rainsh"),
        nca=state.get("nca"),
        htop=state.get("htop"),
        hbot=state.get("hbot"),
    )
    updates = dict(result.carried_fields)
    updates.update(
        {
            "pratesh": result.pratesh,
            "rainsh": result.rainsh,
            "nca": result.nca,
            "htop": result.htop,
            "hbot": result.hbot,
        }
    )
    return updates


def _fddagd_driver_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    return fddagd_driver(state["config_flags"])


def _rk_tendency_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    def _flt(name: str) -> float:
        return _scalar_float(state, name)

    def _time_step(inner: Mapping[str, Any]) -> int:
        return int(getattr(inner["config_flags"], "time_step", 1))

    def _advect_u_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        return {
            "ru_tend": _advect_u_jit(
                inner["u"],
                inner["u"],
                inner["ru_tend"],
                inner["ru"],
                inner["rv"],
                inner["ww_e"],
                inner["c1h"],
                inner["c2h"],
                inner["mut"],
                _time_step(inner),
                _static_config(inner),
                inner["msfux"],
                inner["msfuy"],
                inner["msfvx"],
                inner["msfvy"],
                inner["msftx"],
                inner["msfty"],
                inner["fnm"],
                inner["fnp"],
                1.0 / float(inner["grid"].dx),
                1.0 / float(inner["grid"].dy),
                inner["rdnw"],
                inner["ids"],
                inner["ide"],
                inner["jds"],
                inner["jde"],
                inner["kds"],
                inner["kde"],
                inner["ims"],
                inner["ime"],
                inner["jms"],
                inner["jme"],
                inner["kms"],
                inner["kme"],
                inner["its"],
                inner["ite"],
                inner["jts"],
                inner["jte"],
                inner["kps"],
                inner["kpe"],
                inner["_wrf_real_rounding_token"],
            )
        }

    def _advect_v_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        return {
            "rv_tend": _advect_v_jit(
                inner["v"],
                inner["v"],
                inner["rv_tend"],
                inner["ru"],
                inner["rv"],
                inner["ww_e"],
                inner["c1h"],
                inner["c2h"],
                inner["mut"],
                _time_step(inner),
                _static_config(inner),
                inner["msfux"],
                inner["msfuy"],
                inner["msfvx"],
                inner["msfvy"],
                inner["msftx"],
                inner["msfty"],
                inner["fnm"],
                inner["fnp"],
                1.0 / float(inner["grid"].dx),
                1.0 / float(inner["grid"].dy),
                inner["rdnw"],
                inner["ids"],
                inner["ide"],
                inner["jds"],
                inner["jde"],
                inner["kds"],
                inner["kde"],
                inner["ims"],
                inner["ime"],
                inner["jms"],
                inner["jme"],
                inner["kms"],
                inner["kme"],
                inner["its"],
                inner["ite"],
                inner["jts"],
                inner["jte"],
                inner["kps"],
                inner["kpe"],
                inner["_wrf_real_rounding_token"],
            )
        }

    def _advect_w_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        return {
            "rw_tend": _advect_w_jit(
                inner["w"],
                inner["w"],
                inner["rw_tend"],
                inner["ru"],
                inner["rv"],
                inner["ww_e"],
                inner["c1h"],
                inner["c2h"],
                inner["mut"],
                _time_step(inner),
                _static_config(inner),
                inner["msfux"],
                inner["msfuy"],
                inner["msfvx"],
                inner["msfvy"],
                inner["msftx"],
                inner["msfty"],
                inner["fnm"],
                inner["fnp"],
                1.0 / float(inner["grid"].dx),
                1.0 / float(inner["grid"].dy),
                inner["rdn"],
                inner["ids"],
                inner["ide"],
                inner["jds"],
                inner["jde"],
                inner["kds"],
                inner["kde"],
                inner["ims"],
                inner["ime"],
                inner["jms"],
                inner["jme"],
                inner["kms"],
                inner["kme"],
                inner["its"],
                inner["ite"],
                inner["jts"],
                inner["jte"],
                inner["kps"],
                inner["kpe"],
                inner["_wrf_real_rounding_token"],
            )
        }

    def _advect_scalar_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        return {
            "t_tend": _advect_scalar_jit(
                inner["t"],
                inner["t"],
                inner["t_tend"],
                inner["ru"],
                inner["rv"],
                inner["ww_e"],
                inner["c1h"],
                inner["c2h"],
                inner["mut"],
                _time_step(inner),
                _static_config(inner),
                inner["msfux"],
                inner["msfuy"],
                inner["msfvx"],
                inner["msfvy"],
                inner["msftx"],
                inner["msfty"],
                inner["fnm"],
                inner["fnp"],
                1.0 / float(inner["grid"].dx),
                1.0 / float(inner["grid"].dy),
                inner["rdnw"],
                inner["ids"],
                inner["ide"],
                inner["jds"],
                inner["jde"],
                inner["kds"],
                inner["kde"],
                inner["ims"],
                inner["ime"],
                inner["jms"],
                inner["jme"],
                inner["kms"],
                inner["kme"],
                inner["its"],
                inner["ite"],
                inner["jts"],
                inner["jte"],
                inner["kps"],
                inner["kpe"],
                inner["_wrf_real_rounding_token"],
            )
        }

    def _rhs_ph_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        return {
            "ph_tend": _rhs_ph_jit(
                inner["ph_tend"],
                inner["u"],
                inner["v"],
                inner["ww_e"],
                inner["ph"],
                inner["ph"],
                inner["phb"],
                inner["w"],
                inner["mut"],
                inner["muu"],
                inner["muv"],
                inner["c1f"],
                inner["c2f"],
                inner["fnm"],
                inner["fnp"],
                inner["rdnw"],
                _flt("cfn"),
                _flt("cfn1"),
                1.0 / float(inner["grid"].dx),
                1.0 / float(inner["grid"].dy),
                inner["msfux"],
                inner["msfuy"],
                inner["msfvx"],
                inner["msfvx_inv"],
                inner["msfvy"],
                inner["msftx"],
                inner["msfty"],
                bool(getattr(inner["config_flags"], "non_hydrostatic", True)),
                _static_config(inner),
                inner["ids"],
                inner["ide"],
                inner["jds"],
                inner["jde"],
                inner["kds"],
                inner["kde"],
                inner["ims"],
                inner["ime"],
                inner["jms"],
                inner["jme"],
                inner["kms"],
                inner["kme"],
                inner["its"],
                inner["ite"],
                inner["jts"],
                inner["jte"],
                inner["kps"],
                inner["kpe"],
                inner["_wrf_real_rounding_token"],
            )
        }

    def _horizontal_pressure_gradient_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        ru_out, rv_out = _horizontal_pressure_gradient_jit(
            inner["ru_tend"],
            inner["rv_tend"],
            inner["ph"],
            inner["alt"],
            inner["p"],
            inner["pb"],
            inner["al"],
            inner["php"],
            inner["cqu"],
            inner["cqv"],
            inner["muu"],
            inner["muv"],
            inner["mu"],
            inner["c1h"],
            inner["c2h"],
            inner["fnm"],
            inner["fnp"],
            inner["rdnw"],
            _flt("cf1"),
            _flt("cf2"),
            _flt("cf3"),
            _flt("cfn"),
            _flt("cfn1"),
            1.0 / float(inner["grid"].dx),
            1.0 / float(inner["grid"].dy),
            inner["msfux"],
            inner["msfuy"],
            inner["msfvx"],
            inner["msfvy"],
            inner["msftx"],
            inner["msfty"],
            _static_config(inner),
            bool(getattr(inner["config_flags"], "non_hydrostatic", True)),
            bool(getattr(inner["config_flags"], "top_lid", False)),
            inner["ids"],
            inner["ide"],
            inner["jds"],
            inner["jde"],
            inner["kds"],
            inner["kde"],
            inner["ims"],
            inner["ime"],
            inner["jms"],
            inner["jme"],
            inner["kms"],
            inner["kme"],
            inner["its"],
            inner["ite"],
            inner["jts"],
            inner["jte"],
            inner["kps"],
            inner["kpe"],
            inner["_wrf_real_rounding_token"],
        )
        return {"ru_tend": ru_out, "rv_tend": rv_out}

    def _pg_buoy_w_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        rw_out, cqw_out = pg_buoy_w(
            inner["rw_tend"],
            inner["p"],
            inner["cqw"],
            inner["mu"],
            inner["mub"],
            inner["c1f"],
            inner["c2f"],
            inner["rdnw"],
            inner["rdn"],
            9.81,
            inner["msftx"],
            inner["msfty"],
            inner["ids"],
            inner["ide"],
            inner["jds"],
            inner["jde"],
            inner["kds"],
            inner["kde"],
            inner["ims"],
            inner["ime"],
            inner["jms"],
            inner["jme"],
            inner["kms"],
            inner["kme"],
            inner["its"],
            inner["ite"],
            inner["jts"],
            inner["jte"],
            inner["kps"],
            inner["kpe"],
        )
        return {"rw_tend": rw_out, "cqw": cqw_out}

    def _w_damp_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        if int(getattr(inner["config_flags"], "w_damping", 0)) != 1 and not bool(getattr(inner["config_flags"], "use_adaptive_time_step", False)):
            return {"rw_tend": inner["rw_tend"]}
        rw_out, max_vert, max_horiz = w_damp(
            inner["rw_tend"],
            inner["u"],
            inner["v"],
            inner["ww"],
            inner["w"],
            inner["mut"],
            inner["c1f"],
            inner["c2f"],
            inner["rdnw"],
            1.0 / float(inner["grid"].dx),
            1.0 / float(inner["grid"].dy),
            inner["msfux"],
            inner["msfuy"],
            inner["msfvx"],
            inner["msfvy"],
            float(inner["grid"].dt),
            inner["config_flags"],
            inner["ids"],
            inner["ide"],
            inner["jds"],
            inner["jde"],
            inner["kds"],
            inner["kde"],
            inner["ims"],
            inner["ime"],
            inner["jms"],
            inner["jme"],
            inner["kms"],
            inner["kme"],
            inner["its"],
            inner["ite"],
            inner["jts"],
            inner["jte"],
            inner["kps"],
            inner["kpe"],
        )
        if bool(getattr(inner["config_flags"], "use_adaptive_time_step", False)):
            return {"rw_tend": rw_out, "max_vert_cfl": float(max_vert), "max_horiz_cfl": float(max_horiz)}
        return {"rw_tend": rw_out}

    def _coriolis_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        ru_out, rv_out, rw_out = _coriolis_jit(
            inner["ru"],
            inner["rv"],
            inner["rw"],
            inner["ru_tend"],
            inner["rv_tend"],
            inner["rw_tend"],
            _static_config(inner),
            inner["msftx"],
            inner["msfty"],
            inner["msfux"],
            inner["msfuy"],
            inner["msfvx"],
            inner["msfvy"],
            inner["f"],
            inner["e"],
            inner["sina"],
            inner["cosa"],
            inner["fnm"],
            inner["fnp"],
            inner["ids"],
            inner["ide"],
            inner["jds"],
            inner["jde"],
            inner["kds"],
            inner["kde"],
            inner["ims"],
            inner["ime"],
            inner["jms"],
            inner["jme"],
            inner["kms"],
            inner["kme"],
            inner["its"],
            inner["ite"],
            inner["jts"],
            inner["jte"],
            inner["kps"],
            inner["kpe"],
            inner["_wrf_real_rounding_token"],
        )
        return {"ru_tend": ru_out, "rv_tend": rv_out, "rw_tend": rw_out}

    def _curvature_stage(inner: dict[str, Any]) -> Mapping[str, Any]:
        ru_out, rv_out, rw_out = _curvature_jit(
            inner["ru"],
            inner["rv"],
            inner["rw"],
            inner["u"],
            inner["v"],
            inner["w"],
            inner["ru_tend"],
            inner["rv_tend"],
            inner["rw_tend"],
            _static_config(inner),
            inner["msfux"],
            inner["msfuy"],
            inner["msfvx"],
            inner["msfvy"],
            inner["msftx"],
            inner["msfty"],
            inner["clat"],
            inner["fnm"],
            inner["fnp"],
            1.0 / float(inner["grid"].dx),
            1.0 / float(inner["grid"].dy),
            inner["ids"],
            inner["ide"],
            inner["jds"],
            inner["jde"],
            inner["kds"],
            inner["kde"],
            inner["ims"],
            inner["ime"],
            inner["jms"],
            inner["jme"],
            inner["kms"],
            inner["kme"],
            inner["its"],
            inner["ite"],
            inner["jts"],
            inner["jte"],
            inner["kps"],
            inner["kpe"],
            1.0 / 6370.0e03,
            np.pi / 180.0,
            inner["_wrf_real_rounding_token"],
        )
        return {"ru_tend": ru_out, "rv_tend": rv_out, "rw_tend": rw_out}

    inner_stage_names = (
        "calculate_full",
        "calc_mut_new",
        "calc_mu_uv_1:old",
        "calc_mu_uv_1:new",
        "advect_u",
        "advect_u_implicit",
        "advect_v",
        "advect_v_implicit",
        "advect_w",
        "advect_scalar",
        "advect_scalar_weno",
        "advect_s_implicit",
        "set_tend",
        "rhs_ph",
        "advect_ph_implicit",
        "advect_w_implicit",
        "horizontal_pressure_gradient",
        "pg_buoy_w",
        "w_damp",
        "perturbation_coriolis",
        "coriolis",
        "curvature",
        "held_suarez_damp",
        "horizontal_diffusion:u",
        "horizontal_diffusion:v",
        "horizontal_diffusion:w",
        "horizontal_diffusion_3dmp:m",
        "vertical_diffusion_u",
        "vertical_diffusion_v",
        "vertical_diffusion:w",
        "vertical_diffusion_3dmp",
        "sixth_order_diffusion:u",
        "sixth_order_diffusion:v",
        "sixth_order_diffusion:w",
        "sixth_order_diffusion:m",
        "rk_rayleigh_damp",
        "theta_relaxation",
    )
    inner_callbacks = {name: _missing_stage(f"rk_tendency:{name}") for name in inner_stage_names}
    inner_callbacks.update(
        {
            "advect_u": _advect_u_stage,
            "advect_v": _advect_v_stage,
            "advect_w": _advect_w_stage,
            "advect_scalar": _advect_scalar_stage,
            "rhs_ph": _rhs_ph_stage,
            "horizontal_pressure_gradient": _horizontal_pressure_gradient_stage,
            "pg_buoy_w": _pg_buoy_w_stage,
            "w_damp": _w_damp_stage,
            "coriolis": _coriolis_stage,
            "curvature": _curvature_stage,
        }
    )
    inner_debug_fn = state.get("rk_tendency_inner_debug_fn")
    if inner_debug_fn is not None:
        wrapped_inner_callbacks = {}

        def _debug_wrapper(name: str, callback: Callable[[dict[str, Any]], Mapping[str, Any] | None]) -> Callable[[dict[str, Any]], Mapping[str, Any] | None]:
            def _wrapped(inner: dict[str, Any]) -> Mapping[str, Any] | None:
                updates = callback(inner)
                after = dict(inner)
                if updates:
                    after.update(updates)
                inner_debug_fn(name, inner, after, updates)
                return updates

            return _wrapped

        for inner_name, inner_callback in inner_callbacks.items():
            wrapped_inner_callbacks[inner_name] = _debug_wrapper(inner_name, inner_callback)
        inner_callbacks = wrapped_inner_callbacks
    if _DIRECT_RK_TENDENCY:
        config_flags = state["config_flags"]
        rk_step = int(state["rk_step"])
        rk_order = int(getattr(config_flags, "rk_ord", rk_step))
        adv_opt = int(getattr(config_flags, "momentum_adv_opt", getattr(config_flags, "adv_opt", 0)))
        scalar_weno = (
            int(getattr(config_flags, "scalar_adv_opt", 0)) in {WENO_SCALAR, WENOPD_SCALAR}
            or int(getattr(config_flags, "moist_adv_opt", 0)) in {WENO_SCALAR, WENOPD_SCALAR}
        ) and rk_step == rk_order
        ieva = bool(chk_ieva(config_flags, rk_step))
        unsupported = (
            ieva
            or (rk_step == rk_order and adv_opt == WENO_MOM)
            or scalar_weno
            or int(getattr(config_flags, "cu_physics", 0)) in {GDSCHEME, GFSCHEME, G3SCHEME, NTIEDTKESCHEME}
            or bool(getattr(config_flags, "pert_coriolis", False))
            or int(getattr(config_flags, "ra_lw_physics", 0)) == HELDSUAREZ
            or (rk_step == 1 and int(getattr(config_flags, "diff_opt", 0)) == 1)
            or int(getattr(config_flags, "diff_6th_opt", getattr(state["grid"], "diff_6th_opt", 0))) != 0
            or int(getattr(config_flags, "damp_opt", 0)) == 2
            or int(getattr(config_flags, "rad_nudge", 0)) == 1
        )
        if not unsupported:
            _set_current_dry_aliases_inplace(state)
            ide = int(state["ide"])
            jde = int(state["jde"])
            kde = int(state["kde"])
            ims = int(state["ims"])
            ime = int(state["ime"])
            jms = int(state["jms"])
            jme = int(state["jme"])
            kms = int(state["kms"])
            kme = int(state["kme"])
            its = int(state["its"])
            ite = int(state["ite"])
            jts = int(state["jts"])
            jte = int(state["jte"])
            kts = int(state["kps"])
            kte = int(state["kpe"])
            time_step = int(getattr(config_flags, "time_step", 1))
            dt_step = float(state["grid"].dt) / float(rk_order - rk_step + 1)
            if _FUSED_RK_TENDENCY and inner_debug_fn is None:
                rk_tendency_direct_fn = (
                    _rk_tendency_direct_no_ieva_jit_tuple
                    if int(getattr(config_flags, "zadvect_implicit", 0)) <= 0
                    else _rk_tendency_direct_jit_tuple
                )
                (
                    state["ru_tend"],
                    state["rv_tend"],
                    state["rw_tend"],
                    state["t_tend"],
                    state["ph_tend"],
                    state["u_save"],
                    state["v_save"],
                    state["w_save"],
                    state["ph_save"],
                    state["t_save"],
                    state["mu_tend"],
                    state["mu_save"],
                    state["ww_e"],
                    state["ww_i"],
                    state["cqw"],
                    max_vert_cfl,
                    max_horiz_cfl,
                ) = rk_tendency_direct_fn(
                    ru_tend=state["ru_tend"],
                    rv_tend=state["rv_tend"],
                    rw_tend=state["rw_tend"],
                    t_tend=state["t_tend"],
                    ph_tend=state["ph_tend"],
                    u_save=state["u_save"],
                    v_save=state["v_save"],
                    w_save=state["w_save"],
                    ph_save=state["ph_save"],
                    t_save=state["t_save"],
                    mu_tend=state["mu_tend"],
                    mu_save=state["mu_save"],
                    ww_e=state["ww_e"],
                    ww_i=state["ww_i"],
                    u=state["u"],
                    v=state["v"],
                    w=state["w"],
                    t=state["t"],
                    ph=state["ph"],
                    ww=state["ww"],
                    mut=state["mut"],
                    rdnw=state["rdnw"],
                    rdn=state["rdn"],
                    msfty=state["msfty"],
                    c1f=state["c1f"],
                    c2f=state["c2f"],
                    msfux=state["msfux"],
                    msfuy=state["msfuy"],
                    msfvx=state["msfvx"],
                    msfvx_inv=state["msfvx_inv"],
                    msfvy=state["msfvy"],
                    msftx=state["msftx"],
                    ru=state["ru"],
                    rv=state["rv"],
                    rw=state["rw"],
                    c1h=state["c1h"],
                    c2h=state["c2h"],
                    fnm=state["fnm"],
                    fnp=state["fnp"],
                    phb=state["phb"],
                    muu=state["muu"],
                    muv=state["muv"],
                    alt=state["alt"],
                    p=state["p"],
                    pb=state["pb"],
                    al=state["al"],
                    php=state["php"],
                    cqu=state["cqu"],
                    cqv=state["cqv"],
                    cqw=state["cqw"],
                    mu=state["mu"],
                    mub=state["mub"],
                    f=state["f"],
                    e=state["e"],
                    sina=state["sina"],
                    cosa=state["cosa"],
                    clat=state["clat"],
                    rk_step=rk_step,
                    rk_order=rk_order,
                    time_step=time_step,
                    config_flags=_static_config(state),
                    non_hydrostatic=bool(getattr(config_flags, "non_hydrostatic", True)),
                    top_lid=bool(getattr(config_flags, "top_lid", False)),
                    ids=state["ids"],
                    ide=state["ide"],
                    jds=state["jds"],
                    jde=state["jde"],
                    kds=state["kds"],
                    kde=state["kde"],
                    ims=state["ims"],
                    ime=state["ime"],
                    jms=state["jms"],
                    jme=state["jme"],
                    kms=state["kms"],
                    kme=state["kme"],
                    its=state["its"],
                    ite=state["ite"],
                    jts=state["jts"],
                    jte=state["jte"],
                    kts=state["kps"],
                    kte=state["kpe"],
                    rdx=1.0 / float(state["grid"].dx),
                    rdy=1.0 / float(state["grid"].dy),
                    dt=float(state["grid"].dt),
                    cfn=_flt("cfn"),
                    cfn1=_flt("cfn1"),
                    cf1=_flt("cf1"),
                    cf2=_flt("cf2"),
                    cf3=_flt("cf3"),
                    rounding_token=state["_wrf_real_rounding_token"],
                )
                state.setdefault("time_step", time_step)
                max_vert = float(max_vert_cfl) if bool(getattr(config_flags, "use_adaptive_time_step", False)) else 0.0
                max_horiz = float(max_horiz_cfl) if bool(getattr(config_flags, "use_adaptive_time_step", False)) else 0.0
                return {"dt_step": dt_step, "ieva": ieva, "max_vert_cfl": max_vert, "max_horiz_cfl": max_horiz}
            (
                state["ru_tend"],
                state["rv_tend"],
                state["rw_tend"],
                state["t_tend"],
                state["ph_tend"],
                state["u_save"],
                state["v_save"],
                state["w_save"],
                state["ph_save"],
                state["t_save"],
                state["mu_tend"],
                state["mu_save"],
            ) = _zero_rk_tendency_pack_jit_tuple(
                ru_tend=state["ru_tend"],
                rv_tend=state["rv_tend"],
                rw_tend=state["rw_tend"],
                t_tend=state["t_tend"],
                ph_tend=state["ph_tend"],
                u_save=state["u_save"],
                v_save=state["v_save"],
                w_save=state["w_save"],
                ph_save=state["ph_save"],
                t_save=state["t_save"],
                mu_tend=state["mu_tend"],
                mu_save=state["mu_save"],
                ids=1,
                ide=ide,
                jds=1,
                jde=jde,
                kds=1,
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

            state["ww_e"], state["ww_i"] = _ww_split_jit_tuple(
                ww_e=state["ww_e"],
                ww_i=state["ww_i"],
                u=state["u"],
                v=state["v"],
                ww=state["ww"],
                mut=state["mut"],
                rdnw=state["rdnw"],
                msfty=state["msfty"],
                c1f=state["c1f"],
                c2f=state["c2f"],
                msfux=state["msfux"],
                msfuy=state["msfuy"],
                msfvx=state["msfvx"],
                msfvy=state["msfvy"],
                rdx=1.0 / float(state["grid"].dx),
                rdy=1.0 / float(state["grid"].dy),
                dt=float(state["grid"].dt),
                config_flags=_static_config(state),
                rk_step=rk_step,
                ids=1,
                ide=ide,
                jds=1,
                jde=jde,
                kds=1,
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

            def _run(name: str) -> None:
                callback = inner_callbacks.get(name)
                if callback is None:
                    return
                updates = callback(state)
                if updates:
                    state.update(updates)

            _run("advect_u")
            _run("advect_v")
            if bool(getattr(config_flags, "non_hydrostatic", True)):
                _run("advect_w")
            _run("advect_scalar")
            _run("rhs_ph")
            _run("horizontal_pressure_gradient")
            if bool(getattr(config_flags, "non_hydrostatic", True)):
                _run("pg_buoy_w")
            _run("w_damp")
            _run("coriolis")
            _run("curvature")
            state.setdefault("time_step", time_step)
            return {"dt_step": dt_step, "ieva": ieva, "max_vert_cfl": 0.0, "max_horiz_cfl": 0.0}
    result = rk_tendency(
        config_flags=state["config_flags"],
        rk_step=int(state["rk_step"]),
        fields=_with_current_dry_aliases(state),
        n_moist=int(state.get("num_moist", state["moist"].shape[-1])),
        non_hydrostatic=bool(getattr(state["config_flags"], "non_hydrostatic", True)),
        top_lid=bool(getattr(state["config_flags"], "top_lid", False)),
        dt=float(state["grid"].dt),
        rdx=1.0 / float(state["grid"].dx),
        rdy=1.0 / float(state["grid"].dy),
        khdif=float(getattr(state["grid"], "khdif", 0.0)),
        kvdif=float(getattr(state["grid"], "kvdif", 0.0)),
        diff_6th_opt=int(getattr(state["config_flags"], "diff_6th_opt", getattr(state["grid"], "diff_6th_opt", 0))),
        diff_6th_factor=float(getattr(state["grid"], "diff_6th_factor", 0.0)),
        adv_opt=int(getattr(state["config_flags"], "momentum_adv_opt", getattr(state["config_flags"], "adv_opt", 0))),
        dampcoef=float(getattr(state["grid"], "dampcoef", 0.0)),
        zdamp=float(getattr(state["grid"], "zdamp", 0.0)),
        damp_opt=int(getattr(state["config_flags"], "damp_opt", 0)),
        rad_nudge=int(getattr(state["config_flags"], "rad_nudge", 0)),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
        callbacks=inner_callbacks,
        record_events=False,
    )
    updates = dict(result.fields)
    updates.update(
        {
            "dt_step": result.dt_step,
            "ieva": result.ieva,
            "max_vert_cfl": result.max_vert_cfl,
            "max_horiz_cfl": result.max_horiz_cfl,
        }
    )
    return updates


def _relax_bdy_dry_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    width, spec_zone, relax_zone = _boundary_params(state)
    fcx, gcx = _boundary_weights(state)
    args = (
        _static_config(state),
        state["u_save"],
        state["v_save"],
        state["ph_save"],
        state["t_save"],
        state["w_save"],
        state["mu_tend"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["ru"],
        state["rv"],
        state["ph_2"],
        state["t_2"],
        state["w_2"],
        state["mu_2"],
        state["mut"],
        *_dry_boundary_args(state),
        width,
        spec_zone,
        relax_zone,
        _state_dtbc(state),
        fcx,
        gcx,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        state.get("_wrf_real_rounding_token"),
    )
    if relax_bdy_dry is _RELAX_BDY_DRY_IMPL:
        ru_tendf, rv_tendf, ph_tendf, t_tendf, rw_tendf, mu_tend = _relax_bdy_dry_jit_tuple(*args)
    else:
        result = relax_bdy_dry(*args)
        ru_tendf, rv_tendf, ph_tendf, t_tendf, rw_tendf, mu_tend = (
            result.ru_tendf,
            result.rv_tendf,
            result.ph_tendf,
            result.t_tendf,
            result.rw_tendf,
            result.mu_tend,
        )
    return {
        "u_save": ru_tendf,
        "v_save": rv_tendf,
        "ph_save": ph_tendf,
        "t_save": t_tendf,
        "w_save": rw_tendf,
        "mu_tend": mu_tend,
        "fcx": fcx,
        "gcx": gcx,
        "_boundary_weights_ready": True,
    }


def _rk_addtend_dry_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = _rk_addtend_dry_jit_tuple(
        state["ru_tend"],
        state["rv_tend"],
        state["rw_tend"],
        state["ph_tend"],
        state["t_tend"],
        state["ru_tendf"],
        state["rv_tendf"],
        state["rw_tendf"],
        state["ph_tendf"],
        state["t_tendf"],
        state["u_save"],
        state["v_save"],
        state["w_save"],
        state["ph_save"],
        state["t_save"],
        state["mu_tend"],
        state["mu_tendf"],
        state["rk_step"],
        state["c1h"],
        state["c2h"],
        state["h_diabatic"],
        state["mut"],
        state["msftx"],
        state["msfty"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        state["_wrf_real_rounding_token"],
    )
    names = (
        "ru_tend",
        "rv_tend",
        "rw_tend",
        "ph_tend",
        "t_tend",
        "ru_tendf",
        "rv_tendf",
        "rw_tendf",
        "ph_tendf",
        "t_tendf",
        "mu_tend",
        "mu_tendf",
    )
    return dict(zip(names, result, strict=True))


def _spec_bdy_dry_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    width, spec_zone, _relax_zone = _boundary_params(state)
    args = (
        _static_config(state),
        *_dry_boundary_args(state),
        width,
        spec_zone,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    if (
        _FUSED_SPEC_BDY_DRY_OVERLAY
        and spec_bdy_dry is _SPEC_BDY_DRY_IMPL
        and "_dry_boundary_ru_mask" in state
        and "_dry_boundary_rv_mask" in state
        and "_dry_boundary_ph_mask" in state
        and "_dry_boundary_t_mask" in state
        and "_dry_boundary_rw_mask" in state
        and "_dry_boundary_mu_mask" in state
    ):
        return dict(
            zip(
                ("ru_tend", "rv_tend", "ph_tend", "t_tend", "rw_tend", "mu_tend"),
                _spec_bdy_dry_overlay_masks_jit_tuple(
                    state["ru_tend"],
                    state["rv_tend"],
                    state["ph_tend"],
                    state["t_tend"],
                    state["rw_tend"],
                    state["mu_tend"],
                    state["_dry_boundary_ru_mask"],
                    state["_dry_boundary_rv_mask"],
                    state["_dry_boundary_ph_mask"],
                    state["_dry_boundary_t_mask"],
                    state["_dry_boundary_rw_mask"],
                    state["_dry_boundary_mu_mask"],
                    *args,
                ),
                strict=True,
            )
        )
    if _FUSED_SPEC_BDY_DRY_OVERLAY and spec_bdy_dry is _SPEC_BDY_DRY_IMPL:
        return dict(
            zip(
                ("ru_tend", "rv_tend", "ph_tend", "t_tend", "rw_tend", "mu_tend"),
                _spec_bdy_dry_overlay_jit_tuple(
                    state["ru_tend"],
                    state["rv_tend"],
                    state["ph_tend"],
                    state["t_tend"],
                    state["rw_tend"],
                    state["mu_tend"],
                    *args,
                ),
                strict=True,
            )
        )
    else:
        result = spec_bdy_dry(*args)
        ru_tend_spec, rv_tend_spec, ph_tend_spec, t_tend_spec, rw_tend_spec, mu_tend_spec = (
            result.ru_tend,
            result.rv_tend,
            result.ph_tend,
            result.t_tend,
            result.rw_tend,
            result.mu_tend,
        )
    ru_mask = _spec_bdytend_mask(
        state,
        bxs=state["u_bxs"],
        bxe=state["u_bxe"],
        bys=state["u_bys"],
        bye=state["u_bye"],
        btxs=state["u_btxs"],
        btxe=state["u_btxe"],
        btys=state["u_btys"],
        btye=state["u_btye"],
        variable="u",
    )
    rv_mask = _spec_bdytend_mask(
        state,
        bxs=state["v_bxs"],
        bxe=state["v_bxe"],
        bys=state["v_bys"],
        bye=state["v_bye"],
        btxs=state["v_btxs"],
        btxe=state["v_btxe"],
        btys=state["v_btys"],
        btye=state["v_btye"],
        variable="v",
    )
    ph_mask = _spec_bdytend_mask(
        state,
        bxs=state["ph_bxs"],
        bxe=state["ph_bxe"],
        bys=state["ph_bys"],
        bye=state["ph_bye"],
        btxs=state["ph_btxs"],
        btxe=state["ph_btxe"],
        btys=state["ph_btys"],
        btye=state["ph_btye"],
        variable="h",
    )
    t_mask = _spec_bdytend_mask(
        state,
        bxs=state["t_bxs"],
        bxe=state["t_bxe"],
        bys=state["t_bys"],
        bye=state["t_bye"],
        btxs=state["t_btxs"],
        btxe=state["t_btxe"],
        btys=state["t_btys"],
        btye=state["t_btye"],
        variable="t",
    )
    mu_mask = _spec_bdytend_mask(
        state,
        bxs=state["mu_bxs"],
        bxe=state["mu_bxe"],
        bys=state["mu_bys"],
        bye=state["mu_bye"],
        btxs=state["mu_btxs"],
        btxe=state["mu_btxe"],
        btys=state["mu_btys"],
        btye=state["mu_btye"],
        variable="m",
        kds=1,
        kde=1,
        kms=1,
        kme=1,
        kps=1,
        kpe=1,
        kts=1,
        kte=1,
    )[:, 0, :]
    if bool(getattr(state["config_flags"], "nested", False)):
        rw_mask = _spec_bdytend_mask(
            state,
            bxs=state["w_bxs"],
            bxe=state["w_bxe"],
            bys=state["w_bys"],
            bye=state["w_bye"],
            btxs=state["w_btxs"],
            btxe=state["w_btxe"],
            btys=state["w_btys"],
            btye=state["w_btye"],
            variable="h",
        )
        rw_tend = _overlay_specified_boundary(state["rw_tend"], rw_tend_spec, rw_mask)
    else:
        rw_tend = state["rw_tend"]
    return {
        "ru_tend": _overlay_specified_boundary(state["ru_tend"], ru_tend_spec, ru_mask),
        "rv_tend": _overlay_specified_boundary(state["rv_tend"], rv_tend_spec, rv_mask),
        "ph_tend": _overlay_specified_boundary(state["ph_tend"], ph_tend_spec, ph_mask),
        "t_tend": _overlay_specified_boundary(state["t_tend"], t_tend_spec, t_mask),
        "rw_tend": rw_tend,
        "mu_tend": _overlay_specified_boundary(state["mu_tend"], mu_tend_spec, mu_mask),
    }


def _dry_boundary_mask_args(state: Mapping[str, Any]) -> tuple[Any, ...] | None:
    names = (
        "_dry_boundary_ru_mask",
        "_dry_boundary_rv_mask",
        "_dry_boundary_ph_mask",
        "_dry_boundary_t_mask",
        "_dry_boundary_rw_mask",
        "_dry_boundary_mu_mask",
    )
    if any(name not in state for name in names):
        return None
    return tuple(state[name] for name in names)


def _rk_addtend_dry_args(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        state["ru_tend"],
        state["rv_tend"],
        state["rw_tend"],
        state["ph_tend"],
        state["t_tend"],
        state["ru_tendf"],
        state["rv_tendf"],
        state["rw_tendf"],
        state["ph_tendf"],
        state["t_tendf"],
        state["u_save"],
        state["v_save"],
        state["w_save"],
        state["ph_save"],
        state["t_save"],
        state["mu_tend"],
        state["mu_tendf"],
        state["rk_step"],
        state["c1h"],
        state["c2h"],
        state["h_diabatic"],
        state["mut"],
        state["msftx"],
        state["msfty"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
        state["_wrf_real_rounding_token"],
    )


def _spec_bdy_dry_args(state: Mapping[str, Any], width: int, spec_zone: int) -> tuple[Any, ...]:
    return (
        _static_config(state),
        *_dry_boundary_args(state),
        width,
        spec_zone,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )


def _dry_boundary_tendency_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Run adjacent dry lateral-boundary tendency stages through one callback."""

    masks = _dry_boundary_mask_args(state)
    if (
        _FUSED_DRY_BOUNDARY_SINGLE_KERNEL
        and masks is not None
        and relax_bdy_dry is _RELAX_BDY_DRY_IMPL
        and spec_bdy_dry is _SPEC_BDY_DRY_IMPL
    ):
        width, spec_zone, relax_zone = _boundary_params(state)
        fcx, gcx = _boundary_weights(state)
        rk_args = _rk_addtend_dry_args(state)
        spec_args = _spec_bdy_dry_args(state, width, spec_zone)
        if int(state["rk_step"]) == 1:
            relax_args = (
                _static_config(state),
                state["u_save"],
                state["v_save"],
                state["ph_save"],
                state["t_save"],
                state["w_save"],
                state["mu_tend"],
                state["c1h"],
                state["c2h"],
                state["c1f"],
                state["c2f"],
                state["ru"],
                state["rv"],
                state["ph_2"],
                state["t_2"],
                state["w_2"],
                state["mu_2"],
                state["mut"],
                *_dry_boundary_args(state),
                width,
                spec_zone,
                relax_zone,
                _state_dtbc(state),
                fcx,
                gcx,
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
                state.get("_wrf_real_rounding_token"),
            )
            names = (
                "u_save",
                "v_save",
                "ph_save",
                "t_save",
                "w_save",
                "ru_tend",
                "rv_tend",
                "rw_tend",
                "ph_tend",
                "t_tend",
                "ru_tendf",
                "rv_tendf",
                "rw_tendf",
                "ph_tendf",
                "t_tendf",
                "mu_tend",
                "mu_tendf",
            )
            values = _dry_boundary_tendency_rk1_jit_tuple(*masks, *relax_args, *rk_args, *spec_args)
            updates = dict(zip(names, values, strict=True))
            updates.update({"fcx": fcx, "gcx": gcx, "_boundary_weights_ready": True})
            return updates

        names = (
            "ru_tend",
            "rv_tend",
            "rw_tend",
            "ph_tend",
            "t_tend",
            "ru_tendf",
            "rv_tendf",
            "rw_tendf",
            "ph_tendf",
            "t_tendf",
            "mu_tend",
            "mu_tendf",
        )
        values = _dry_boundary_tendency_nonrk1_jit_tuple(*masks, *rk_args, *spec_args)
        return dict(zip(names, values, strict=True))

    updates: dict[str, Any] = {}
    local = dict(state)
    if int(local["rk_step"]) == 1:
        stage_updates = dict(_relax_bdy_dry_stage(local))
        local.update(stage_updates)
        updates.update(stage_updates)
    stage_updates = dict(_rk_addtend_dry_stage(local))
    local.update(stage_updates)
    updates.update(stage_updates)
    stage_updates = dict(_spec_bdy_dry_stage(local))
    updates.update(stage_updates)
    return updates


def _small_step_prep_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    small_step_prep_fn = _small_step_prep_jit_tuple if _JIT_SMALL_STEP_PREP else _small_step_prep_eager_tuple
    (
        u_1,
        u_2,
        v_1,
        v_2,
        w_1,
        w_2,
        t_1,
        t_2,
        ph_1,
        ph_2,
        mu_1,
        mu_2,
        muus,
        muvs,
        muts,
        mudf,
        u_save,
        v_save,
        w_save,
        t_save,
        ph_save,
        mu_save,
        ww_save,
        c2a,
        _u_term_new,
        _u_term_old,
        _v_term_new,
        _v_term_old,
    ) = small_step_prep_fn(
        state["u_1"],
        state["u_2"],
        state["v_1"],
        state["v_2"],
        state["w_1"],
        state["w_2"],
        state["t_1"],
        state["t_2"],
        state["ph_1"],
        state["ph_2"],
        state["mub"],
        state["mu_1"],
        state["mu_2"],
        state["muu"],
        state["muus"],
        state["muv"],
        state["muvs"],
        state["mut"],
        state["muts"],
        state["mudf"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["u_save"],
        state["v_save"],
        state["w_save"],
        state["t_save"],
        state["ph_save"],
        state["mu_save"],
        state["ww"],
        state["ww_1"],
        state["c2a"],
        state["pb"],
        state["p"],
        state["alt"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["msftx"],
        state["msfty"],
        1.0 / float(state["grid"].dx),
        1.0 / float(state["grid"].dy),
        int(state["rk_step"]),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
        rounding_token=state.get("_wrf_real_rounding_token"),
    )
    return {
        "u_1": u_1,
        "u_2": u_2,
        "v_1": v_1,
        "v_2": v_2,
        "w_1": w_1,
        "w_2": w_2,
        "t_1": t_1,
        "t_2": t_2,
        "ph_1": ph_1,
        "ph_2": ph_2,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "muus": muus,
        "muvs": muvs,
        "muts": muts,
        "mudf": mudf,
        "u_save": u_save,
        "v_save": v_save,
        "w_save": w_save,
        "t_save": t_save,
        "ph_save": ph_save,
        "mu_save": mu_save,
        "ww_1": ww_save,
        "c2a": c2a,
    }


def _calc_p_rho_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    calc_p_rho_fn = _calc_p_rho_jit_tuple if _JIT_CALC_P_RHO else _calc_p_rho_eager_tuple
    calc_p_rho_args = (
        state["al"],
        state["p"],
        state["ph_2"],
        state["alt"],
        state["t_2"],
        state["t_save"],
        state["c2a"],
        state["pm1"],
        state["mu_2"],
        state["muts"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["znu"],
        _field_or(state, "t0", 300.0),
        state["rdnw"],
        state["dnw"],
        _field_or(state, "smdiv", _cfg_value(state, "smdiv", 0.1)),
    )
    non_hydrostatic = bool(getattr(state["config_flags"], "non_hydrostatic", True))
    step_is_zero = int(state.get("iteration", 0)) == 0
    domain_kwargs = dict(
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    rounding_token = state["_wrf_real_rounding_token"]
    if _has_traced_array(state["p"]):

        def call_calc_p_rho(
            dynamic_args: tuple[Any, ...], dynamic_token: Any
        ) -> tuple[Any, Any, Any, Any]:
            return calc_p_rho_fn(
                *dynamic_args,
                non_hydrostatic,
                step_is_zero,
                rounding_token=dynamic_token,
                **domain_kwargs,
            )

        al, p, ph, pm1 = semantic_call(
            call_calc_p_rho,
            calc_p_rho_args,
            rounding_token,
        )
    else:
        al, p, ph, pm1 = calc_p_rho_fn(
            *calc_p_rho_args,
            non_hydrostatic,
            step_is_zero,
            rounding_token=rounding_token,
            **domain_kwargs,
        )
    return {"al": al, "p": p, "ph_2": ph, "pm1": pm1}


def _calc_p_rho_phi_pressure_log(
    pressure_ratio: Any,
    *,
    ids: int,
    ide: int,
    i_start: int,
    i_end: int,
    nproc_x: int,
    rounding_token: Any,
) -> Any:
    """Match WRF's vector-body and rank-local scalar-tail ``logf``."""

    ratio = jnp.asarray(pressure_ratio, dtype=jnp.float32)
    pressure_log = wrf_sse4_logf_positive_normal(ratio, rounding_token)

    scalar_offsets = _wrf_patch_scalar_tail_offsets(
        ids=int(ids),
        ide=int(ide),
        i_start=int(i_start),
        i_end=int(i_end),
        nproc_x=int(nproc_x),
    )
    if not scalar_offsets:
        return pressure_log

    offsets = jnp.asarray(scalar_offsets, dtype=jnp.int32)
    scalar_log = _materialize_wrf_real(
        wrf_glibc_logf_positive_normal(ratio[offsets]),
        rounding_token,
    )
    return pressure_log.at[offsets].set(scalar_log)


def _calc_p_rho_phi_nonhydro_opt2(
    *,
    moist: Any,
    n_moist: int,
    alb: Any,
    muts: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    ph: Any,
    phb: Any,
    al: Any | None = None,
    p: Any | None = None,
    pb: Any,
    t: Any,
    p0: float,
    t0: float,
    ptop: float,
    use_theta_m: int,
    ids: int,
    ide: int,
    jde: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    param_first_scalar: int = 2,
    p_qv: int = 2,
    rounding_token: Any | None = None,
    wrf_reference_nproc_x: int = 1,
) -> tuple[Any, Any, Any, Any]:
    """Vectorized WRF ``calc_p_rho_phi`` path for nonhydrostatic hypsometric option 2."""

    moist_arr = jnp.asarray(moist)
    alb_arr = jnp.asarray(alb)
    muts_arr = jnp.asarray(muts)
    c3h_arr = jnp.asarray(c3h)
    c4h_arr = jnp.asarray(c4h)
    c3f_arr = jnp.asarray(c3f)
    c4f_arr = jnp.asarray(c4f)
    ph_arr = jnp.asarray(ph)
    phb_arr = jnp.asarray(phb)
    pb_arr = jnp.asarray(pb)
    t_arr = jnp.asarray(t)
    dtype = t_arr.dtype

    def store(value: Any) -> Any:
        array = jnp.asarray(value, dtype=dtype)
        if rounding_token is None:
            return array
        return _materialize_wrf_real(array, rounding_token)

    def add(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype))

    def sub(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype))

    def mul(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype))

    def div(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) / jnp.asarray(right, dtype=dtype))

    al_arr = jnp.zeros_like(t_arr) if al is None else jnp.asarray(al)
    p_arr = jnp.zeros_like(t_arr) if p is None else jnp.asarray(p)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    if itf < int(its) or jtf < int(jts) or ktf < int(kts):
        return al, p, ph_arr, phb_arr

    i_slice = slice(int(its) - int(ims), itf - int(ims) + 1)
    j_slice = slice(int(jts) - int(jms), jtf - int(jms) + 1)
    k_slice = slice(int(kts) - int(kms), ktf - int(kms) + 1)
    kp1_slice = slice(int(kts) - int(kms) + 1, ktf - int(kms) + 2)

    muts_ij = muts_arr[i_slice, j_slice]
    c3h_k = c3h_arr[k_slice]
    c4h_k = c4h_arr[k_slice]
    c3f_k = c3f_arr[k_slice]
    c4f_k = c4f_arr[k_slice]
    c3f_kp1 = c3f_arr[kp1_slice]
    c4f_kp1 = c4f_arr[kp1_slice]

    muts_ikj = muts_ij[:, None, :]
    if rounding_token is None:
        pfu = c3f_kp1[None, :, None] * muts_ikj + c4f_kp1[None, :, None] + float(ptop)
        pfd = c3f_k[None, :, None] * muts_ikj + c4f_k[None, :, None] + float(ptop)
        phm = c3h_k[None, :, None] * muts_ikj + c4h_k[None, :, None] + float(ptop)
    else:
        ptop_real = store(ptop)
        pfu = add(add(mul(c3f_kp1[None, :, None], muts_ikj), c4f_kp1[None, :, None]), ptop_real)
        pfd = add(add(mul(c3f_k[None, :, None], muts_ikj), c4f_k[None, :, None]), ptop_real)
        phm = add(add(mul(c3h_k[None, :, None], muts_ikj), c4h_k[None, :, None]), ptop_real)

    region = (i_slice, k_slice, j_slice)
    region_kp1 = (i_slice, kp1_slice, j_slice)
    if rounding_token is None:
        al_region = (ph_arr[region_kp1] - ph_arr[region] + phb_arr[region_kp1] - phb_arr[region]) / phm / jnp.log(pfd / pfu) - alb_arr[region]
    else:
        geopotential_depth = sub(ph_arr[region_kp1], ph_arr[region])
        geopotential_depth = add(geopotential_depth, phb_arr[region_kp1])
        geopotential_depth = sub(geopotential_depth, phb_arr[region])
        pressure_log = _calc_p_rho_phi_pressure_log(
            div(pfd, pfu),
            ids=int(ids),
            ide=int(ide),
            i_start=int(its),
            i_end=itf,
            nproc_x=int(wrf_reference_nproc_x),
            rounding_token=rounding_token,
        )
        al_region = sub(div(div(geopotential_depth, phm), pressure_log), alb_arr[region])

    if int(use_theta_m) == 1 or int(n_moist) < int(param_first_scalar):
        qvf = 1.0
    else:
        qv_index = int(p_qv) - 1
        ratio = jnp.asarray(R_V, dtype=t_arr.dtype) / jnp.asarray(R_D, dtype=t_arr.dtype)
        if rounding_token is None:
            qvf = jnp.asarray(1.0, dtype=t_arr.dtype) + ratio * moist_arr[region + (qv_index,)]
        else:
            qvf = add(store(1.0), mul(ratio, moist_arr[region + (qv_index,)]))

    if rounding_token is None:
        temp = (R_D * (float(t0) + t_arr[region]) * qvf) / (float(p0) * (al_region + alb_arr[region]))
        p_region = (temp**1.4) * float(p0) - pb_arr[region]
    else:
        numerator = mul(store(R_D), add(store(t0), t_arr[region]))
        numerator = mul(numerator, qvf)
        denominator = mul(store(p0), add(al_region, alb_arr[region]))
        temp = div(numerator, denominator)
        # WRF's VPOW is a float32 vector powf operation, not a generic
        # XLA power expression. Preserve that compiler-level math path.
        powered = store(vector_powf_for_wrf(temp, store(1.4)))
        p_region = sub(mul(powered, store(p0)), pb_arr[region])

    region_is_full = (
        int(i_slice.start or 0) == 0
        and int(k_slice.start or 0) == 0
        and int(j_slice.start or 0) == 0
        and int(i_slice.stop or 0) == int(al_arr.shape[0])
        and int(k_slice.stop or 0) == int(al_arr.shape[1])
        and int(j_slice.stop or 0) == int(al_arr.shape[2])
        and tuple(al_region.shape) == tuple(al_arr.shape)
        and tuple(p_region.shape) == tuple(p_arr.shape)
    )
    if region_is_full:
        return al_region, p_region, ph_arr, phb_arr
    start_indices = (
        int(i_slice.start or 0),
        int(k_slice.start or 0),
        int(j_slice.start or 0),
    )
    return (
        jax.lax.dynamic_update_slice(al_arr, al_region, start_indices),
        jax.lax.dynamic_update_slice(p_arr, p_region, start_indices),
        ph_arr,
        phb_arr,
    )


def _calc_p_rho_phi_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    nonhydro = bool(getattr(state["config_flags"], "non_hydrostatic", True))
    hypsometric_opt = int(getattr(state["config_flags"], "hypsometric_opt", 2))
    if nonhydro and hypsometric_opt == 2:
        al, p, ph, phb = _calc_p_rho_phi_nonhydro_opt2(
            moist=state["moist"],
            n_moist=2,
            alb=state["alb"],
            muts=state["muts"],
            c3h=state["c3h"],
            c4h=state["c4h"],
            c3f=state["c3f"],
            c4f=state["c4f"],
            ph=state["ph_2"],
            phb=state["phb"],
            al=state["al"],
            p=state["p"],
            pb=state["pb"],
            t=state["t_2"],
            p0=_field_or(state, "p0", 100000.0),
            t0=_field_or(state, "t0", 300.0),
            ptop=_scalar_float(state, "p_top"),
            use_theta_m=int(getattr(state["config_flags"], "use_theta_m", 0)),
            ids=state["ids"],
            ide=state["ide"],
            jde=state["jde"],
            kde=state["kde"],
            ims=state["ims"],
            jms=state["jms"],
            kms=state["kms"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
            rounding_token=state["_wrf_real_rounding_token"],
            wrf_reference_nproc_x=int(
                getattr(state["config_flags"], "wrf_reference_nproc_x", 1)
            ),
        )
        return {"al": al, "p": p, "ph_2": ph, "phb": phb}

    al, p, ph, phb = calc_p_rho_phi(
        state["moist"],
        2,
        hypsometric_opt,
        state["alb"],
        state["mu_2"],
        state["muts"],
        state["c1h"],
        state["c2h"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["ph_2"],
        state["phb"],
        state["pb"],
        state["t_2"],
        _field_or(state, "p0", 100000.0),
        _field_or(state, "t0", 300.0),
        _scalar_float(state, "p_top"),
        state["znu"],
        state["znw"],
        state["dnw"],
        state["rdnw"],
        state["rdn"],
        nonhydro,
        int(getattr(state["config_flags"], "use_theta_m", 0)),
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"al": al, "p": p, "ph_2": ph, "phb": phb}


def _advance_ppt_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    zero2 = _zeros2(state)
    zero3 = jnp.zeros_like(jnp.asarray(state["t_2"]))
    result = advance_ppt(
        _field_or(state, "rthcuten", zero3),
        _field_or(state, "rqvcuten", zero3),
        _field_or(state, "rqccuten", zero3),
        _field_or(state, "rqrcuten", zero3),
        _field_or(state, "cldfra_cup", zero3),
        _field_or(state, "rqicuten", zero3),
        _field_or(state, "rqscuten", zero3),
        _field_or(state, "rainc", zero2),
        _field_or(state, "raincv", zero2),
        _field_or(state, "rainsh", zero2),
        _field_or(state, "pratec", zero2),
        _field_or(state, "pratesh", zero2),
        _field_or(state, "nca", zero2),
        _field_or(state, "htop", zero2),
        _field_or(state, "hbot", zero2),
        _field_or(state, "cutop", zero2),
        _field_or(state, "cubot", zero2),
        _field_or(state, "cuppt", zero2),
        float(state["grid"].dt),
        state["config_flags"],
        preserve_eager_rounding=bool(state.get("_preserve_water_statement_rounding", False)) or bool(state.get("_force_radiation", False)) or not _has_traced_array(zero3),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state.get("its", state["ids"]),
        ite=state.get("ite", state["ide"] - 1),
        jts=state.get("jts", state["jds"]),
        jte=state.get("jte", state["jde"] - 1),
        kts=state["kps"],
        kte=state["kpe"],
    )
    return {
        "rthcuten": result.rthcuten,
        "rqvcuten": result.rqvcuten,
        "rqccuten": result.rqccuten,
        "rqrcuten": result.rqrcuten,
        "cldfra_cup": result.cldfra_cup,
        "rqicuten": result.rqicuten,
        "rqscuten": result.rqscuten,
        "rainc": result.rainc,
        "raincv": result.raincv,
        "rainsh": result.rainsh,
        "pratec": result.pratec,
        "pratesh": result.pratesh,
        "nca": result.nca,
        "htop": result.htop,
        "hbot": result.hbot,
        "cutop": result.cutop,
        "cubot": result.cubot,
        "cuppt": result.cuppt,
    }


def _phy_prep_part2_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    zero3 = jnp.zeros_like(jnp.asarray(state["t_2"]))
    zero_ph = jnp.zeros_like(jnp.asarray(state["ph_2"]))
    zero2 = jnp.zeros_like(jnp.asarray(state["mut"]))
    th_phy = _field_or(state, "th_phy", jnp.asarray(state["t_2"]) + jnp.asarray(T0, dtype=jnp.asarray(state["t_2"]).dtype))
    outputs = phy_prep_part2(
        state["config_flags"],
        _field_or(state, "muts", state["mut"]),
        _field_or(state, "muus", state["muu"]),
        _field_or(state, "muvs", state["muv"]),
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        _field_or(state, "rthraten", zero3),
        _field_or(state, "rthblten", zero3),
        _field_or(state, "rublten", zero3),
        _field_or(state, "rvblten", zero3),
        _field_or(state, "rqvblten", zero3),
        _field_or(state, "rqcblten", zero3),
        _field_or(state, "rqiblten", zero3),
        _field_or(state, "rucuten", zero3),
        _field_or(state, "rvcuten", zero3),
        _field_or(state, "rthcuten", zero3),
        _field_or(state, "rqvcuten", zero3),
        _field_or(state, "rqccuten", zero3),
        _field_or(state, "rqrcuten", zero3),
        _field_or(state, "rqicuten", zero3),
        _field_or(state, "rqscuten", zero3),
        _field_or(state, "rushten", zero3),
        _field_or(state, "rvshten", zero3),
        _field_or(state, "rthshten", zero3),
        _field_or(state, "rqvshten", zero3),
        _field_or(state, "rqcshten", zero3),
        _field_or(state, "rqrshten", zero3),
        _field_or(state, "rqishten", zero3),
        _field_or(state, "rqsshten", zero3),
        _field_or(state, "rqgshten", zero3),
        _field_or(state, "rthften", zero3),
        _field_or(state, "rqvften", zero3),
        _field_or(state, "rundgdten", zero3),
        _field_or(state, "rvndgdten", zero3),
        _field_or(state, "rthndgdten", zero3),
        _field_or(state, "rphndgdten", zero_ph),
        _field_or(state, "rqvndgdten", zero3),
        _field_or(state, "rmundgdten", zero2),
        state["t_2"],
        th_phy,
        jnp.asarray(state["moist"])[..., 1],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state.get("its", state["ids"]),
        state.get("ite", state["ide"] - 1),
        state.get("jts", state["jds"]),
        state.get("jte", state["jde"] - 1),
        state["kps"],
        state["kpe"],
    )
    names = (
        "rthraten",
        "rthblten",
        "rublten",
        "rvblten",
        "rqvblten",
        "rqcblten",
        "rqiblten",
        "rucuten",
        "rvcuten",
        "rthcuten",
        "rqvcuten",
        "rqccuten",
        "rqrcuten",
        "rqicuten",
        "rqscuten",
        "rushten",
        "rvshten",
        "rthshten",
        "rqvshten",
        "rqcshten",
        "rqrshten",
        "rqishten",
        "rqsshten",
        "rqgshten",
        "rthften",
        "rqvften",
        "rundgdten",
        "rvndgdten",
        "rthndgdten",
        "rphndgdten",
        "rqvndgdten",
        "rmundgdten",
        "th_phy",
    )
    return dict(zip(names, outputs, strict=True))


def _calc_coef_w_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    use_wrf_real_rounding = "_wrf_real_rounding_token" in state
    calc_coef_w_fn = _calc_coef_w_rounded_jit_tuple if use_wrf_real_rounding else _calc_coef_w_jit_tuple
    rounding_args = (
        (state["_wrf_real_rounding_token"],)
        if use_wrf_real_rounding
        else ()
    )
    a_out, alpha_out, gamma_out = calc_coef_w_fn(
        state["a"],
        state["alpha"],
        state["gamma"],
        state["mut"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["cqw"],
        state["rdn"],
        state["rdnw"],
        state["c2a"],
        state["dts_rk"],
        9.81,
        _field_or(state, "epssm", _cfg_value(state, "epssm", 0.1)),
        bool(getattr(state["config_flags"], "top_lid", False)),
        *rounding_args,
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    return {"a": a_out, "alpha": alpha_out, "gamma": gamma_out}


def _advance_uv_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    _width, spec_zone, _relax_zone = _boundary_params(state)
    traced = _has_traced_array(state["u_2"])
    rounding_token = state.get("_wrf_real_rounding_token")
    use_wrf_real_rounding = rounding_token is not None
    if traced or (_JIT_ADVANCE_UV and use_wrf_real_rounding):
        advance_uv_fn = _advance_uv_rounded_jit_tuple
    elif _JIT_ADVANCE_UV:
        advance_uv_fn = _advance_uv_jit_tuple
    else:
        advance_uv_fn = _advance_uv_eager_tuple
    rounding_args = (rounding_token,) if use_wrf_real_rounding else ()
    rdx = 1.0 / float(state["grid"].dx)
    rdy = 1.0 / float(state["grid"].dy)
    dts_rk = state["dts_rk"]
    emdiv = _field_or(state, "emdiv", _cfg_value(state, "emdiv", 0.01))
    if use_wrf_real_rounding:
        rdx = state.get("_rdx", rdx)
        rdy = state.get("_rdy", rdy)
        dts_rk = _materialize_wrf_real(dts_rk, rounding_token)
        emdiv = _materialize_wrf_real(emdiv, rounding_token)
    u_out, v_out = advance_uv_fn(
        state["u_2"],
        state["ru_tend"],
        state["v_2"],
        state["rv_tend"],
        state["p"],
        state["pb"],
        state["ph_2"],
        state["php"],
        state["alt"],
        state["al"],
        state["mu_2"],
        state["muu"],
        state["cqu"],
        state["muv"],
        state["cqv"],
        state["mudf"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        rdx,
        rdy,
        dts_rk,
        state["cf1"],
        state["cf2"],
        state["cf3"],
        state["fnm"],
        state["fnp"],
        emdiv,
        state["rdnw"],
        _static_config(state),
        spec_zone,
        bool(getattr(state["config_flags"], "non_hydrostatic", True)),
        bool(getattr(state["config_flags"], "top_lid", False)),
        *rounding_args,
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    return {"u_2": u_out, "v_2": v_out}


def _spec_bdyupdate_stage(stage_name: str) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    target = stage_name.removeprefix("spec_bdyupdate:")
    field_name, tend_name, variable = {
        "u": ("u_2", "ru_tend", "u"),
        "v": ("v_2", "rv_tend", "v"),
        "t": ("t_2", "t_tend", "t"),
        "mu_2": ("mu_2", "mu_tend", "m"),
        "muts": ("muts", "mu_tend", "m"),
        "w": ("w_2", "rw_tend", "h"),
    }[target]

    def _apply(state: dict[str, Any]) -> Mapping[str, Any]:
        _width, spec_zone, _relax_zone = _boundary_params(state)
        field = state[field_name]
        field_tend = state[tend_name]
        is_2d_mass = target in {"mu_2", "muts"}
        if is_2d_mass:
            field = jnp.asarray(field)[:, None, :]
            field_tend = jnp.asarray(field_tend)[:, None, :]
            kds = kde = kms = kme = kps = kpe = kts = kte = 1
        else:
            kds = state["kds"]
            kde = state["kde"]
            kms = state["kms"]
            kme = state["kme"]
            kps = state["kps"]
            kpe = state["kpe"]
            kts = state["kps"]
            kte = state["kpe"]
        updated = spec_bdyupdate(
            field,
            field_tend,
            state["dts_rk"],
            variable,
            state["config_flags"],
            spec_zone,
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            kds,
            kde,
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            kms,
            kme,
            state["ips"],
            state["ipe"],
            state["jps"],
            state["jpe"],
            kps,
            kpe,
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            kts,
            kte,
        )
        if is_2d_mass:
            updated = jnp.asarray(updated)[:, 0, :]
        return {field_name: updated}

    return _apply


def _advance_mu_t_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    advance_mu_t_fn = _advance_mu_t_jit_tuple if _JIT_ADVANCE_MU_T else _advance_mu_t_eager_tuple
    (
        ww,
        ww_1,
        mu,
        muave,
        muts,
        mudf,
        uam,
        vam,
        wwam,
        t,
        t_ave,
    ) = advance_mu_t_fn(
        state["ww"],
        state["ww_1"],
        state["u_2"],
        state["u_save"],
        state["v_2"],
        state["v_save"],
        state["mu_2"],
        state["mut"],
        state["muave"],
        state["muts"],
        state["muu"],
        state["muv"],
        state["mudf"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["ru_m"],
        state["rv_m"],
        state["ww_m"],
        state["t_2"],
        state["t_save"],
        state["t_2ave"],
        state["t_tend"],
        state["mu_tend"],
        1.0 / float(state["grid"].dx),
        1.0 / float(state["grid"].dy),
        state["dts_rk"],
        _field_or(state, "epssm", _cfg_value(state, "epssm", 0.1)),
        state["dnw"],
        state["fnm"],
        state["fnp"],
        state["rdnw"],
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        state["msftx"],
        state["msfty"],
        int(state.get("iteration", 1)),
        _static_config(state),
        state["_wrf_real_rounding_token"],
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    return {
        "ww": ww,
        "ww_1": ww_1,
        "mu_2": mu,
        "muave": muave,
        "muts": muts,
        "mudf": mudf,
        "ru_m": uam,
        "rv_m": vam,
        "ww_m": wwam,
        "t_2": t,
        "t_2ave": t_ave,
    }


def _advance_w_stage(
    state: dict[str, Any], *, _include_diagnostics: bool = False
) -> Mapping[str, Any]:
    traced = _has_traced_array(state["w_2"])
    use_wrf_real_rounding = traced or _JIT_ADVANCE_W
    if _include_diagnostics:
        if not use_wrf_real_rounding:
            raise ValueError("advance_w internal diagnostics require rounded JIT execution")
        advance_w_fn = _advance_w_rounded_diagnostic_jit_tuple
    else:
        advance_w_fn = _advance_w_rounded_jit_tuple if use_wrf_real_rounding else _advance_w_eager_tuple
    rounding_args = (state["_wrf_real_rounding_token"],) if use_wrf_real_rounding else ()
    rdx = 1.0 / float(state["grid"].dx)
    rdy = 1.0 / float(state["grid"].dy)
    dts_rk = state["dts_rk"]
    t0 = _field_or(state, "t0", 300.0)
    epssm = _field_or(state, "epssm", _cfg_value(state, "epssm", 0.1))
    if use_wrf_real_rounding:
        token = state["_wrf_real_rounding_token"]
        rdx = state["_rdx"]
        rdy = state["_rdy"]
        dts_rk = _materialize_wrf_real(dts_rk, token)
        t0 = _materialize_wrf_real(t0, token)
        epssm = _materialize_wrf_real(epssm, token)
    advance_w_args = (
        state["w_2"],
        state["rw_tend"],
        state["ww"],
        state["w_save"],
        state["u_2"],
        state["v_2"],
        state["mu_2"],
        state["mut"],
        state["muave"],
        state["muts"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["t_2ave"],
        state["t_2"],
        state["t_save"],
        state["ph_2"],
        state["ph_save"],
        state["phb"],
        state["ph_tend"],
        state["ht"],
        state["c2a"],
        state["cqw"],
        state["alt"],
        state["alb"],
        state["a"],
        state["alpha"],
        state["gamma"],
        rdx,
        rdy,
        dts_rk,
        t0,
        epssm,
        state["dnw"],
        state["fnm"],
        state["fnp"],
        state["rdnw"],
        state["rdn"],
        state["cf1"],
        state["cf2"],
        state["cf3"],
        state["msftx"],
        state["msfty"],
    )
    static_config = _static_config(state)
    top_lid = bool(getattr(state["config_flags"], "top_lid", False))
    domain_kwargs = dict(
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    if traced and use_wrf_real_rounding:
        rounding_token = state["_wrf_real_rounding_token"]

        def call_advance_w(
            dynamic_args: tuple[Any, ...], dynamic_token: Any
        ) -> tuple[Any, ...]:
            return advance_w_fn(
                *dynamic_args,
                static_config,
                top_lid,
                dynamic_token,
                **domain_kwargs,
            )

        outputs = semantic_call(call_advance_w, advance_w_args, rounding_token)
    else:
        outputs = advance_w_fn(
            *advance_w_args,
            static_config,
            top_lid,
            *rounding_args,
            **domain_kwargs,
        )
    t_2ave, w, ph = outputs[:3]
    result = {"t_2ave": t_2ave, "w_2": w, "ph_2": ph}
    if _include_diagnostics:
        result["_advance_w_raw_local"] = outputs[3]
        result["_advance_w_solved_local"] = outputs[4]
        result["_advance_w_pressure_upper_local"] = outputs[5]
        result["_advance_w_pressure_lower_local"] = outputs[6]
        result["_advance_w_pressure_part_local"] = outputs[7]
        result["_advance_w_pressure_numerator_local"] = outputs[8]
        result["_advance_w_pressure_denominator_local"] = outputs[9]
        result["_advance_w_pressure_ratio_local"] = outputs[10]
        result["_advance_w_pressure_rhs_delta_local"] = outputs[11]
        result["_advance_w_pressure_ph_delta_local"] = outputs[12]
        result["_advance_w_pressure_rhs_term_local"] = outputs[13]
        result["_advance_w_pressure_ph_term_local"] = outputs[14]
        result["_advance_w_pressure_bracket_local"] = outputs[15]
        result["_advance_w_rhs_base_local"] = outputs[16]
        result["_advance_w_wdwn_local"] = outputs[17]
        result["_advance_w_rhs_adv_local"] = outputs[18]
        result["_advance_w_rhs_dec_local"] = outputs[19]
        result["_advance_w_thermal_upper_local"] = outputs[20]
        result["_advance_w_thermal_lower_local"] = outputs[21]
        result["_advance_w_buoyancy_part_local"] = outputs[22]
        result["_advance_w_damping_htop_local"] = outputs[23]
        result["_advance_w_damping_hbot_local"] = outputs[24]
        result["_advance_w_damping_hk_local"] = outputs[25]
        result["_advance_w_damping_angle_local"] = outputs[26]
        result["_advance_w_damping_sine_local"] = outputs[27]
        result["_advance_w_damping_weight_local"] = outputs[28]
        result["_advance_w_damping_mass_local"] = outputs[29]
        result["_advance_w_damping_term_local"] = outputs[30]
        result["_advance_w_damping_numerator_local"] = outputs[31]
        result["_advance_w_damping_denominator_local"] = outputs[32]
        result["_advance_w_damped_w_local"] = outputs[33]
    return result


def _sumflux_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    sumflux_fn = _sumflux_jit_tuple if _JIT_SUMFLUX else _sumflux_eager_tuple
    iteration = int(state.get("iteration", 1))
    number_of_small_timesteps = int(state.get("number_of_small_timesteps", 1))
    if iteration <= 1:
        sumflux_iteration = 1
    elif iteration >= number_of_small_timesteps:
        sumflux_iteration = number_of_small_timesteps
    else:
        sumflux_iteration = 2
    ru_m, rv_m, ww_m = sumflux_fn(
        state["u_2"],
        state["v_2"],
        state["ww"],
        state["u_save"],
        state["v_save"],
        state["ww_1"],
        state["muu"],
        state["muv"],
        state["c1h"],
        state["c2h"],
        state["c1f"],
        state["c2f"],
        state["c3h"],
        state["c4h"],
        state["c3f"],
        state["c4f"],
        state["ru_m"],
        state["rv_m"],
        state["ww_m"],
        _field_or(state, "epssm", _cfg_value(state, "epssm", 0.1)),
        state["msfux"],
        state["msfuy"],
        state["msfvx"],
        state["msfvx_inv"],
        state["msfvy"],
        sumflux_iteration,
        number_of_small_timesteps,
        state["_wrf_real_rounding_token"],
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    return {"ru_m": ru_m, "rv_m": rv_m, "ww_m": ww_m}


def _spec_bdyupdate_ph_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    _width, spec_zone, _relax_zone = _boundary_params(state)
    updated = spec_bdyupdate_ph(
        state["ph_save"],
        state["ph_2"],
        state["ph_tend"],
        state["mu_tend"],
        state["muts"],
        state["c1f"],
        state["c2f"],
        state["dts_rk"],
        "h",
        state["config_flags"],
        spec_zone,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"ph_2": updated}


def _zero_grad_bdy_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    _width, spec_zone, _relax_zone = _boundary_params(state)
    updated = zero_grad_bdy(
        state["w_2"],
        "w",
        state["config_flags"],
        spec_zone,
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"w_2": updated}


def _calc_mu_uv_1_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    if _FUSED_CALC_MU_SMALL_FINISH:
        return {}
    muu, muv = _calc_mu_uv_1_jit_tuple(
        _static_config(state),
        state["muts"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"muus": muu, "muvs": muv}


def _small_step_finish_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    if _FUSED_CALC_MU_SMALL_FINISH and _JIT_SMALL_STEP_FINISH:
        (
            u_2,
            v_2,
            w_2,
            t_2,
            ph_2,
            ww,
            mu_2,
            muus,
            muvs,
        ) = _calc_mu_uv_small_step_finish_jit_tuple(
            u_2=state["u_2"],
            u_1=state["u_1"],
            v_2=state["v_2"],
            v_1=state["v_1"],
            w_2=state["w_2"],
            w_1=state["w_1"],
            t_2=state["t_2"],
            t_1=state["t_1"],
            ph_2=state["ph_2"],
            ph_1=state["ph_1"],
            ww=state["ww"],
            ww1=state["ww_1"],
            mu_2=state["mu_2"],
            mu_1=state["mu_1"],
            mut=state["mut"],
            muts=state["muts"],
            muu=state["muu"],
            muv=state["muv"],
            c1h=state["c1h"],
            c2h=state["c2h"],
            c1f=state["c1f"],
            c2f=state["c2f"],
            c3h=state["c3h"],
            c4h=state["c4h"],
            c3f=state["c3f"],
            c4f=state["c4f"],
            u_save=state["u_save"],
            v_save=state["v_save"],
            w_save=state["w_save"],
            t_save=state["t_save"],
            ph_save=state["ph_save"],
            mu_save=state["mu_save"],
            msfux=state["msfux"],
            msfuy=state["msfuy"],
            msfvx=state["msfvx"],
            msfvy=state["msfvy"],
            msftx=state["msftx"],
            msfty=state["msfty"],
            h_diabatic=state["h_diabatic"],
            number_of_small_timesteps=int(state.get("number_of_small_timesteps", 1)),
            dts=state["dts_rk"],
            config_flags=_static_config(state),
            rk_step=int(state["rk_step"]),
            rk_order=int(state.get("rk_order", 3)),
            ids=state["ids"],
            ide=state["ide"],
            jds=state["jds"],
            jde=state["jde"],
            kds=state["kds"],
            kde=state["kde"],
            ims=state["ims"],
            ime=state["ime"],
            jms=state["jms"],
            jme=state["jme"],
            kms=state["kms"],
            kme=state["kme"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
            rounding_token=state.get("_wrf_real_rounding_token"),
        )
    else:
        small_step_finish_fn = _small_step_finish_jit_tuple if _JIT_SMALL_STEP_FINISH else _small_step_finish_eager_tuple
        (
            u_2,
            v_2,
            w_2,
            t_2,
            ph_2,
            ww,
            mu_2,
            mu_1,
            mut,
            muts,
            muu,
            muus,
            muv,
            muvs,
            mu_save,
        ) = small_step_finish_fn(
            state["u_2"],
            state["u_1"],
            state["v_2"],
            state["v_1"],
            state["w_2"],
            state["w_1"],
            state["t_2"],
            state["t_1"],
            state["ph_2"],
            state["ph_1"],
            state["ww"],
            state["ww_1"],
            state["mu_2"],
            state["mu_1"],
            state["mut"],
            state["muts"],
            state["muu"],
            state["muus"],
            state["muv"],
            state["muvs"],
            state["c1h"],
            state["c2h"],
            state["c1f"],
            state["c2f"],
            state["c3h"],
            state["c4h"],
            state["c3f"],
            state["c4f"],
            state["u_save"],
            state["v_save"],
            state["w_save"],
            state["t_save"],
            state["ph_save"],
            state["mu_save"],
            state["msfux"],
            state["msfuy"],
            state["msfvx"],
            state["msfvy"],
            state["msftx"],
            state["msfty"],
            state["h_diabatic"],
            int(state.get("number_of_small_timesteps", 1)),
            state["dts_rk"],
            int(state["rk_step"]),
            int(state.get("rk_order", 3)),
            ids=state["ids"],
            ide=state["ide"],
            jds=state["jds"],
            jde=state["jde"],
            kds=state["kds"],
            kde=state["kde"],
            ims=state["ims"],
            ime=state["ime"],
            jms=state["jms"],
            jme=state["jme"],
            kms=state["kms"],
            kme=state["kme"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
            rounding_token=state.get("_wrf_real_rounding_token"),
        )
    return {
        "u_2": u_2,
        "v_2": v_2,
        "w_2": w_2,
        "t_2": t_2,
        "ph_2": ph_2,
        "ww": ww,
        "mu_2": mu_2,
        "muus": muus,
        "muvs": muvs,
    }


def _set_physical_bc3d_moist_qv_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    qv = set_physical_bc3d(
        jnp.asarray(state["moist"])[..., 1],
        "p",
        state["config_flags"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    moist = _replace_last_axis_component(state["moist"], 1, qv)
    return {"moist": moist, "qvapor": qv}


def _set_physical_bc3d_moist_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    updates = dict(_set_physical_bc3d_stage("set_physical_bc3d:moist")(state))
    moist = updates["moist"]
    synced = _moist_fields_from_array(moist)
    return {"moist": moist, **synced}


def _set_physical_bc3d_scalar_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    updates = dict(_set_physical_bc3d_stage("set_physical_bc3d:scalar")(state))
    scalar = updates["scalar"]
    synced = _scalar_fields_from_array(scalar)
    return {"scalar": scalar, **synced}


def _moist_scalar_advance_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    moist_arr = jnp.asarray(state["moist"])
    moist_old_arr = jnp.asarray(state["moist_old"])
    moist_tend_arr = jnp.asarray(state["moist_tend"])
    qv = moist_arr[..., 1]
    rounding_token = state["_wrf_real_rounding_token"] if _has_traced_array(qv) else None
    moist_old_qv = moist_old_arr[..., 1] if moist_old_arr.ndim == 4 else moist_old_arr
    moist_tend_qv = moist_tend_arr[..., 1]
    zero3 = jnp.zeros_like(qv)

    rk_step = int(state["rk_step"])
    rk_order = int(getattr(state["config_flags"], "rk_ord", rk_step))
    adv_opt = int(getattr(state["config_flags"], "moist_adv_opt", 1))
    if rk_step == rk_order and adv_opt in {MONOTONIC, WENO_SCALAR, WENOPD_SCALAR}:
        raise MissingKernelImplementationError("moisture scalar advance does not yet cover final monotonic/WENO advection")
    if _FUSED_MOIST_SCALAR_ADVANCE and int(moist_arr.shape[-1]) <= 2:
        width, spec_zone, relax_zone = _boundary_params(state)
        fcx, gcx = _boundary_weights(state)
        do_moist_boundary = (
            (bool(getattr(state["config_flags"], "specified", False)) or bool(getattr(state["config_flags"], "nested", False)))
            and rk_step == 1
        )
        (
            moist_old_out,
            moist_out,
            qv_out,
            moist_tend_out,
            advect_tend,
            h_tendency,
            z_tendency,
            rqvften,
            ww_e,
            ww_i,
        ) = _moist_scalar_advance_jit_tuple(
            qv=qv,
            moist=state["moist"],
            moist_old=moist_old_qv,
            moist_tend=state["moist_tend"],
            rqvften=_field_or(state, "rqvften", zero3),
            ww_e=state["ww_e"],
            ww_i=state["ww_i"],
            u=state["u_1"],
            v=state["v_1"],
            ww=state["ww_m"],
            muts=state["muts"],
            rdnw=state["rdnw"],
            msfty=state["msfty"],
            c1f=state["c1f"],
            c2f=state["c2f"],
            msfux=state["msfux"],
            msfuy=state["msfuy"],
            msfvx=state["msfvx"],
            msfvy=state["msfvy"],
            ru_m=state["ru_m"],
            rv_m=state["rv_m"],
            c1h=state["c1h"],
            c2h=state["c2h"],
            mub=state["mub"],
            mu_1=state["mu_1"],
            mut=state["mut"],
            mu_2=state["mu_2"],
            msftx=state["msftx"],
            fnm=state["fnm"],
            fnp=state["fnp"],
            qvapor_bxs=state.get("qvapor_bxs", zero3),
            qvapor_bxe=state.get("qvapor_bxe", zero3),
            qvapor_bys=state.get("qvapor_bys", zero3),
            qvapor_bye=state.get("qvapor_bye", zero3),
            qvapor_btxs=state.get("qvapor_btxs", zero3),
            qvapor_btxe=state.get("qvapor_btxe", zero3),
            qvapor_btys=state.get("qvapor_btys", zero3),
            qvapor_btye=state.get("qvapor_btye", zero3),
            fcx=fcx,
            gcx=gcx,
            rounding_token=state["_wrf_real_rounding_token"],
            config_flags=_static_config(state),
            rk_step=rk_step,
            rk_order=rk_order,
            adv_opt=adv_opt,
            time_step=int(getattr(state["config_flags"], "time_step", 1)),
            do_final_pd=rk_step == rk_order and adv_opt == POSITIVEDEF,
            do_cu_rqvften=int(getattr(state["config_flags"], "cu_physics", 0)) in {93, 5, 3, 1, 11, 6, 16},
            do_moist_boundary=do_moist_boundary,
            rdx=1.0 / float(state["grid"].dx),
            rdy=1.0 / float(state["grid"].dy),
            dt_split=float(state["dt_rk"]) * float(rk_order - rk_step + 1),
            dt_rk=float(state["dt_rk"]),
            dtbc=_state_dtbc(state),
            spec_bdy_width=width,
            spec_zone=spec_zone,
            relax_zone=relax_zone,
            ids=state["ids"],
            ide=state["ide"],
            jds=state["jds"],
            jde=state["jde"],
            kds=state["kds"],
            kde=state["kde"],
            ims=state["ims"],
            ime=state["ime"],
            jms=state["jms"],
            jme=state["jme"],
            kms=state["kms"],
            kme=state["kme"],
            ips=state["ips"],
            ipe=state["ipe"],
            jps=state["jps"],
            jpe=state["jpe"],
            kps=state["kps"],
            kpe=state["kpe"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
        )
        return {
            "moist_old": moist_old_out,
            "moist": moist_out,
            "qvapor": qv_out,
            "moist_tend": moist_tend_out,
            "advect_tend": advect_tend,
            "h_tendency": h_tendency,
            "z_tendency": z_tendency,
            "rqvften": rqvften,
            "ww_e": ww_e,
            "ww_i": ww_i,
        }

    ww_e, ww_i = _ww_split_jit_tuple(
        ww_e=state["ww_e"],
        ww_i=state["ww_i"],
        u=state["u_1"],
        v=state["v_1"],
        ww=state["ww_m"],
        mut=state["muts"],
        rdnw=state["rdnw"],
        msfty=state["msfty"],
        c1f=state["c1f"],
        c2f=state["c2f"],
        msfux=state["msfux"],
        msfuy=state["msfuy"],
        msfvx=state["msfvx"],
        msfvy=state["msfvy"],
        rdx=1.0 / float(state["grid"].dx),
        rdy=1.0 / float(state["grid"].dy),
        dt=float(state["dt_rk"]) * float(rk_order - rk_step + 1),
        config_flags=_static_config(state),
        rk_step=rk_step,
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )
    if rk_step == rk_order and adv_opt == POSITIVEDEF:
        advect_tend, h_tendency, z_tendency = _advect_scalar_pd_jit(
            qv,
            moist_old_qv,
            zero3,
            zero3,
            zero3,
            state["ru_m"],
            state["rv_m"],
            ww_e,
            state["c1h"],
            state["c2h"],
            state["muts"],
            state["mub"],
            state["mu_1"],
            int(getattr(state["config_flags"], "time_step", 1)),
            _static_config(state),
            False,
            state["msfux"],
            state["msfuy"],
            state["msfvx"],
            state["msfvy"],
            state["msftx"],
            state["msfty"],
            state["fnm"],
            state["fnp"],
            1.0 / float(state["grid"].dx),
            1.0 / float(state["grid"].dy),
            state["rdnw"],
            float(state["dt_rk"]),
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["kds"],
            state["kde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["kms"],
            state["kme"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            state["kps"],
            state["kpe"],
            state["_wrf_real_rounding_token"],
        )
    else:
        advect_tend = _advect_scalar_jit(
            qv,
            qv,
            zero3,
            state["ru_m"],
            state["rv_m"],
            ww_e,
            state["c1h"],
            state["c2h"],
            state["muts"],
            int(getattr(state["config_flags"], "time_step", 1)),
            _static_config(state),
            state["msfux"],
            state["msfuy"],
            state["msfvx"],
            state["msfvy"],
            state["msftx"],
            state["msfty"],
            state["fnm"],
            state["fnp"],
            1.0 / float(state["grid"].dx),
            1.0 / float(state["grid"].dy),
            state["rdnw"],
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["kds"],
            state["kde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["kms"],
            state["kme"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            state["kps"],
            state["kpe"],
            state["_wrf_real_rounding_token"],
        )
        h_tendency = zero3
        z_tendency = zero3
    rqvften = _field_or(state, "rqvften", zero3)
    if int(getattr(state["config_flags"], "cu_physics", 0)) in {93, 5, 3, 1, 11, 6, 16}:
        rqvften = advect_tend * state["msfty"][:, None, :]
    do_moist_boundary = (
        (bool(getattr(state["config_flags"], "specified", False)) or bool(getattr(state["config_flags"], "nested", False)))
        and int(state["rk_step"]) == 1
        and (
            bool(getattr(state["config_flags"], "nested", False))
            or bool(getattr(state["config_flags"], "have_bcs_moist", False))
        )
    )
    if (bool(getattr(state["config_flags"], "specified", False)) or bool(getattr(state["config_flags"], "nested", False))) and int(state["rk_step"]) == 1:
        is_qvapor = True
        if is_qvapor or bool(getattr(state["config_flags"], "nested", False)) or (
            bool(getattr(state["config_flags"], "specified", False)) and bool(getattr(state["config_flags"], "have_bcs_moist", False))
        ):
            width, spec_zone, relax_zone = _boundary_params(state)
            fcx, gcx = _boundary_weights(state)
            relaxed = relax_bdy_scalar(
                moist_tend_qv,
                qv,
                state["mut"],
                state["c1h"],
                state["c2h"],
                state["qvapor_bxs"],
                state["qvapor_bxe"],
                state["qvapor_bys"],
                state["qvapor_bye"],
                state["qvapor_btxs"],
                state["qvapor_btxe"],
                state["qvapor_btys"],
                state["qvapor_btye"],
                width,
                spec_zone,
                relax_zone,
                _state_dtbc(state),
                fcx,
                gcx,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
                rounding_token=rounding_token,
            )
            specified = spec_bdy_scalar(
                state["qvapor_bxs"],
                state["qvapor_bxe"],
                state["qvapor_bys"],
                state["qvapor_bye"],
                state["qvapor_btxs"],
                state["qvapor_btxe"],
                state["qvapor_btys"],
                state["qvapor_btye"],
                width,
                spec_zone,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            mask = _spec_bdytend_mask(
                state,
                bxs=state["qvapor_bxs"],
                bxe=state["qvapor_bxe"],
                bys=state["qvapor_bys"],
                bye=state["qvapor_bye"],
                btxs=state["qvapor_btxs"],
                btxe=state["qvapor_btxe"],
                btys=state["qvapor_btys"],
                btye=state["qvapor_btye"],
                variable="q",
            )
            moist_tend_qv = _overlay_specified_boundary(relaxed, specified, mask)

    rk_update_scalar_fn = _rk_update_scalar_jit_tuple if _JIT_RK_UPDATE_SCALAR else rk_update_scalar
    rk_update_scalar_config = _static_config(state) if _JIT_RK_UPDATE_SCALAR else state["config_flags"]
    moist_old_work = moist_old_arr if moist_old_arr.ndim == 4 else _replace_moist_components(moist_arr, {1: moist_old_arr})
    moist_old_out = moist_old_work
    moist_out = moist_arr
    moist_tend_out = moist_tend_arr.at[..., 1].set(moist_tend_qv)
    qv_out = qv

    use_batched_nonqv = (
        _BATCHED_NONQV_MOIST_ADVANCE
        and _JIT_RK_UPDATE_SCALAR
        and int(moist_arr.shape[-1]) > 2
        and not bool(getattr(state["config_flags"], "nested", False))
        and not bool(getattr(state["config_flags"], "have_bcs_moist", False))
    )
    species_stop = 2 if use_batched_nonqv else int(moist_arr.shape[-1])
    for species_index in range(1, species_stop):
        scalar = moist_arr[..., species_index]
        scalar_old = moist_old_work[..., species_index]
        species_tend = moist_tend_qv if species_index == 1 else moist_tend_arr[..., species_index]
        if species_index == 1:
            species_advect_tend = advect_tend
            species_h_tendency = h_tendency
            species_z_tendency = z_tendency
        elif rk_step == rk_order and adv_opt == POSITIVEDEF:
            species_advect_tend, species_h_tendency, species_z_tendency = _advect_scalar_pd_jit(
                scalar,
                scalar_old,
                zero3,
                zero3,
                zero3,
                state["ru_m"],
                state["rv_m"],
                ww_e,
                state["c1h"],
                state["c2h"],
                state["muts"],
                state["mub"],
                state["mu_1"],
                int(getattr(state["config_flags"], "time_step", 1)),
                _static_config(state),
                False,
                state["msfux"],
                state["msfuy"],
                state["msfvx"],
                state["msfvy"],
                state["msftx"],
                state["msfty"],
                state["fnm"],
                state["fnp"],
                1.0 / float(state["grid"].dx),
                1.0 / float(state["grid"].dy),
                state["rdnw"],
                float(state["dt_rk"]),
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
                state["_wrf_real_rounding_token"],
            )
        else:
            species_advect_tend = _advect_scalar_jit(
                scalar,
                scalar,
                zero3,
                state["ru_m"],
                state["rv_m"],
                ww_e,
                state["c1h"],
                state["c2h"],
                state["muts"],
                int(getattr(state["config_flags"], "time_step", 1)),
                _static_config(state),
                state["msfux"],
                state["msfuy"],
                state["msfvx"],
                state["msfvy"],
                state["msftx"],
                state["msfty"],
                state["fnm"],
                state["fnp"],
                1.0 / float(state["grid"].dx),
                1.0 / float(state["grid"].dy),
                state["rdnw"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            species_h_tendency = zero3
            species_z_tendency = zero3

        if species_index != 1 and do_moist_boundary:
            width, spec_zone, relax_zone = _boundary_params(state)
            fcx, gcx = _boundary_weights(state)
            bxs = _moist_boundary_component(state, species_index, "_bxs", jnp.zeros_like(state["qvapor_bxs"]))
            bxe = _moist_boundary_component(state, species_index, "_bxe", jnp.zeros_like(state["qvapor_bxe"]))
            bys = _moist_boundary_component(state, species_index, "_bys", jnp.zeros_like(state["qvapor_bys"]))
            bye = _moist_boundary_component(state, species_index, "_bye", jnp.zeros_like(state["qvapor_bye"]))
            btxs = _moist_boundary_component(state, species_index, "_btxs", jnp.zeros_like(state["qvapor_btxs"]))
            btxe = _moist_boundary_component(state, species_index, "_btxe", jnp.zeros_like(state["qvapor_btxe"]))
            btys = _moist_boundary_component(state, species_index, "_btys", jnp.zeros_like(state["qvapor_btys"]))
            btye = _moist_boundary_component(state, species_index, "_btye", jnp.zeros_like(state["qvapor_btye"]))
            relaxed = relax_bdy_scalar(
                species_tend,
                scalar,
                state["mut"],
                state["c1h"],
                state["c2h"],
                bxs,
                bxe,
                bys,
                bye,
                btxs,
                btxe,
                btys,
                btye,
                width,
                spec_zone,
                relax_zone,
                _state_dtbc(state),
                fcx,
                gcx,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
                rounding_token=rounding_token,
            )
            specified = spec_bdy_scalar(
                bxs,
                bxe,
                bys,
                bye,
                btxs,
                btxe,
                btys,
                btye,
                width,
                spec_zone,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            mask = _spec_bdytend_mask(
                state,
                bxs=bxs,
                bxe=bxe,
                bys=bys,
                bye=bye,
                btxs=btxs,
                btxe=btxe,
                btys=btys,
                btye=btye,
                variable="q",
            )
            species_tend = _overlay_specified_boundary(relaxed, specified, mask)

        scalar_1, scalar_2, advh_t, advz_t = rk_update_scalar_fn(
            1,
            1,
            scalar_old[..., None],
            scalar[..., None],
            species_tend[..., None],
            None,
            None,
            species_advect_tend,
            species_h_tendency,
            species_z_tendency,
            state["msftx"],
            state["msfty"],
            state["c1h"],
            state["c2h"],
            state["mu_1"],
            state["mu_2"],
            state["mub"],
            int(state["rk_step"]),
            float(state["dt_rk"]),
            int(getattr(state["config_flags"], "spec_zone", state.get("spec_zone", 1))),
            rk_update_scalar_config,
            False,
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["kds"],
            state["kde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["kms"],
            state["kme"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            state["kps"],
            state["kpe"],
            state["_wrf_real_rounding_token"],
        )
        del advh_t, advz_t
        species_out = jnp.asarray(scalar_2)[..., 0]
        if species_index != 1 and bool(getattr(state["config_flags"], "specified", False)) and not bool(
            getattr(state["config_flags"], "have_bcs_moist", False)
        ):
            _width, spec_zone, _relax_zone = _boundary_params(state)
            species_out = flow_dep_bdy(
                species_out,
                state["ru_m"],
                state["rv_m"],
                state["config_flags"],
                spec_zone,
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
        moist_old_out = moist_old_out.at[..., species_index].set(jnp.asarray(scalar_1)[..., 0])
        moist_out = moist_out.at[..., species_index].set(species_out)
        moist_tend_out = moist_tend_out.at[..., species_index].set(species_tend)
        if species_index == 1:
            qv_out = species_out

    if use_batched_nonqv:
        moist_old_out, moist_out, moist_tend_out = _batched_nonqv_moist_advance_jit_tuple(
            moist=moist_out,
            moist_old=moist_old_out,
            moist_tend=moist_tend_out,
            ww_e=ww_e,
            ru_m=state["ru_m"],
            rv_m=state["rv_m"],
            c1h=state["c1h"],
            c2h=state["c2h"],
            muts=state["muts"],
            mub=state["mub"],
            mu_1=state["mu_1"],
            mu_2=state["mu_2"],
            msfux=state["msfux"],
            msfuy=state["msfuy"],
            msfvx=state["msfvx"],
            msfvy=state["msfvy"],
            msftx=state["msftx"],
            msfty=state["msfty"],
            fnm=state["fnm"],
            fnp=state["fnp"],
            rdnw=state["rdnw"],
            rounding_token=state["_wrf_real_rounding_token"],
            config_flags=_static_config(state),
            rk_step=rk_step,
            do_final_pd=rk_step == rk_order and adv_opt == POSITIVEDEF,
            do_flow_dep_boundary=bool(getattr(state["config_flags"], "specified", False)),
            time_step=int(getattr(state["config_flags"], "time_step", 1)),
            rdx=1.0 / float(state["grid"].dx),
            rdy=1.0 / float(state["grid"].dy),
            dt_rk=float(state["dt_rk"]),
            spec_zone=int(getattr(state["config_flags"], "spec_zone", state.get("spec_zone", 1))),
            ids=state["ids"],
            ide=state["ide"],
            jds=state["jds"],
            jde=state["jde"],
            kds=state["kds"],
            kde=state["kde"],
            ims=state["ims"],
            ime=state["ime"],
            jms=state["jms"],
            jme=state["jme"],
            kms=state["kms"],
            kme=state["kme"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
            ips=state["ips"],
            ipe=state["ipe"],
            jps=state["jps"],
            jpe=state["jpe"],
            kps=state["kps"],
            kpe=state["kpe"],
        )

    synced = _moist_fields_from_array(moist_out)
    return {
        "moist_old": moist_old_out,
        "moist": moist_out,
        "qvapor": qv_out,
        **synced,
        "moist_tend": moist_tend_out,
        "advect_tend": advect_tend,
        "h_tendency": h_tendency,
        "z_tendency": z_tendency,
        "rqvften": rqvften,
        "ww_e": ww_e,
        "ww_i": ww_i,
    }


def _rk_update_scalar_pd_moist_old_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Apply WRF's final-RK positive-definite physics update to ``moist_old``."""

    moist_old = jnp.asarray(state["moist_old"])
    moist_tend = jnp.asarray(state["moist_tend"])
    size = 2 if moist_old.ndim == 3 else moist_old.shape[-1]
    if size <= 1:
        return {"moist_old": moist_old, "moist_tend": moist_tend}
    species = moist_old[..., None] if moist_old.ndim == 3 else moist_old[..., 1:]
    updated_old, updated_tend = _rk_update_species_pd_arrays(
        species, moist_tend[..., 1:size], state
    )
    if moist_old.ndim == 3:
        moist_old_out = updated_old[..., 0]
    else:
        moist_old_out = jnp.concatenate([moist_old[..., :1], updated_old], axis=-1)
    moist_tend_out = jnp.concatenate(
        [moist_tend[..., :1], updated_tend, moist_tend[..., size:]], axis=-1
    )
    return {"moist_old": moist_old_out, "moist_tend": moist_tend_out}


def _rk_update_scalar_pd_scalar_old_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Apply WRF's final-RK positive-definite update to ``scalar_old``."""

    scalar_old = jnp.asarray(_field_or(state, "scalar_old", state["scalar"]))
    scalar_tend = jnp.asarray(state["scalar_tend"])
    if scalar_old.ndim != 4 or scalar_old.shape[-1] <= 1:
        return {"scalar_old": scalar_old, "scalar_tend": scalar_tend}
    size = scalar_old.shape[-1]
    updated_old, updated_tend = _rk_update_species_pd_arrays(
        scalar_old[..., 1:], scalar_tend[..., 1:size], state
    )
    scalar_old_out = jnp.concatenate([scalar_old[..., :1], updated_old], axis=-1)
    scalar_tend_out = jnp.concatenate(
        [scalar_tend[..., :1], updated_tend, scalar_tend[..., size:]], axis=-1
    )
    return {"scalar_old": scalar_old_out, "scalar_tend": scalar_tend_out}


def _rk_update_species_pd_arrays(scalar: Any, tendency: Any, state: Mapping[str, Any]) -> tuple[Any, Any]:
    """Batch independent species without changing their arithmetic or halos."""

    def update_one(field: Any, field_tendency: Any) -> tuple[Any, Any]:
        return _rk_update_qv_pd_single_array(
            scalar=field,
            sc_tend=field_tendency,
            c1=state["c1h"],
            c2=state["c2h"],
            mu_old=state["mu_1"],
            mu_new=state["mu_1"],
            mu_base=state["mub"],
            dt=float(state["dt_rk"]),
            ide=state["ide"],
            jde=state["jde"],
            kde=state["kde"],
            ims=state["ims"],
            jms=state["jms"],
            kms=state["kms"],
            its=state["its"],
            ite=state["ite"],
            jts=state["jts"],
            jte=state["jte"],
            kts=state["kps"],
            kte=state["kpe"],
        )
    return jax.vmap(update_one, in_axes=-1, out_axes=-1)(scalar, tendency)


def _scalar_advance_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Advance WRF 4D ``scalar`` species such as Thompson QNICE/QNRAIN."""

    scalar_arr = jnp.asarray(state["scalar"])
    if scalar_arr.ndim != 4 or scalar_arr.shape[-1] <= 1:
        return {}
    scalar_old_arr = jnp.asarray(_field_or(state, "scalar_old", scalar_arr))
    scalar_tend_arr = jnp.asarray(state["scalar_tend"])
    zero3 = jnp.zeros_like(scalar_arr[..., 1])

    rk_step = int(state["rk_step"])
    rk_order = int(getattr(state["config_flags"], "rk_ord", rk_step))
    adv_opt = int(getattr(state["config_flags"], "scalar_adv_opt", 1))
    if rk_step == rk_order and adv_opt in {MONOTONIC, WENO_SCALAR, WENOPD_SCALAR}:
        raise MissingKernelImplementationError("4D scalar advance does not yet cover final monotonic/WENO advection")

    ww_e, ww_i = _ww_split_jit_tuple(
        ww_e=state["ww_e"],
        ww_i=state["ww_i"],
        u=state["u_1"],
        v=state["v_1"],
        ww=state["ww_m"],
        mut=state["muts"],
        rdnw=state["rdnw"],
        msfty=state["msfty"],
        c1f=state["c1f"],
        c2f=state["c2f"],
        msfux=state["msfux"],
        msfuy=state["msfuy"],
        msfvx=state["msfvx"],
        msfvy=state["msfvy"],
        rdx=1.0 / float(state["grid"].dx),
        rdy=1.0 / float(state["grid"].dy),
        dt=float(state["dt_rk"]) * float(rk_order - rk_step + 1),
        config_flags=_static_config(state),
        rk_step=rk_step,
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        its=state["its"],
        ite=state["ite"],
        jts=state["jts"],
        jte=state["jte"],
        kts=state["kps"],
        kte=state["kpe"],
    )

    rk_update_scalar_fn = _rk_update_scalar_jit_tuple if _JIT_RK_UPDATE_SCALAR else rk_update_scalar
    rk_update_scalar_config = _static_config(state) if _JIT_RK_UPDATE_SCALAR else state["config_flags"]
    scalar_old_out = scalar_old_arr
    scalar_out = scalar_arr
    scalar_tend_out = scalar_tend_arr
    last_advect_tend = zero3
    last_h_tendency = zero3
    last_z_tendency = zero3
    do_scalar_boundary = (
        (bool(getattr(state["config_flags"], "nested", False)) or bool(getattr(state["config_flags"], "specified", False)))
        and rk_step == 1
        and (
            bool(getattr(state["config_flags"], "nested", False))
            or bool(getattr(state["config_flags"], "have_bcs_scalar", False))
        )
    )

    width, spec_zone, relax_zone = _boundary_params(state)
    fcx, gcx = _boundary_weights(state)
    for species_index in range(1, int(scalar_arr.shape[-1])):
        species = scalar_arr[..., species_index]
        species_old = scalar_old_arr[..., species_index]
        species_tend = scalar_tend_arr[..., species_index]
        if rk_step == rk_order and adv_opt == POSITIVEDEF:
            advect_tend, h_tendency, z_tendency = _advect_scalar_pd_jit(
                species,
                species_old,
                zero3,
                zero3,
                zero3,
                state["ru_m"],
                state["rv_m"],
                ww_e,
                state["c1h"],
                state["c2h"],
                state["muts"],
                state["mub"],
                state["mu_1"],
                int(getattr(state["config_flags"], "time_step", 1)),
                _static_config(state),
                False,
                state["msfux"],
                state["msfuy"],
                state["msfvx"],
                state["msfvy"],
                state["msftx"],
                state["msfty"],
                state["fnm"],
                state["fnp"],
                1.0 / float(state["grid"].dx),
                1.0 / float(state["grid"].dy),
                state["rdnw"],
                float(state["dt_rk"]),
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
                state["_wrf_real_rounding_token"],
            )
        else:
            advect_tend = _advect_scalar_jit(
                species,
                species,
                zero3,
                state["ru_m"],
                state["rv_m"],
                ww_e,
                state["c1h"],
                state["c2h"],
                state["muts"],
                int(getattr(state["config_flags"], "time_step", 1)),
                _static_config(state),
                state["msfux"],
                state["msfuy"],
                state["msfvx"],
                state["msfvy"],
                state["msftx"],
                state["msfty"],
                state["fnm"],
                state["fnp"],
                1.0 / float(state["grid"].dx),
                1.0 / float(state["grid"].dy),
                state["rdnw"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            h_tendency = zero3
            z_tendency = zero3

        if do_scalar_boundary:
            bxs = _scalar_boundary_component(state, species_index, "_bxs", jnp.zeros_like(state["qvapor_bxs"]))
            bxe = _scalar_boundary_component(state, species_index, "_bxe", jnp.zeros_like(state["qvapor_bxe"]))
            bys = _scalar_boundary_component(state, species_index, "_bys", jnp.zeros_like(state["qvapor_bys"]))
            bye = _scalar_boundary_component(state, species_index, "_bye", jnp.zeros_like(state["qvapor_bye"]))
            btxs = _scalar_boundary_component(state, species_index, "_btxs", jnp.zeros_like(state["qvapor_btxs"]))
            btxe = _scalar_boundary_component(state, species_index, "_btxe", jnp.zeros_like(state["qvapor_btxe"]))
            btys = _scalar_boundary_component(state, species_index, "_btys", jnp.zeros_like(state["qvapor_btys"]))
            btye = _scalar_boundary_component(state, species_index, "_btye", jnp.zeros_like(state["qvapor_btye"]))
            relaxed = relax_bdy_scalar(
                species_tend,
                species,
                state["mut"],
                state["c1h"],
                state["c2h"],
                bxs,
                bxe,
                bys,
                bye,
                btxs,
                btxe,
                btys,
                btye,
                width,
                spec_zone,
                relax_zone,
                _state_dtbc(state),
                fcx,
                gcx,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            specified = spec_bdy_scalar(
                bxs,
                bxe,
                bys,
                bye,
                btxs,
                btxe,
                btys,
                btye,
                width,
                spec_zone,
                state["config_flags"],
                state["ids"],
                state["ide"],
                state["jds"],
                state["jde"],
                state["kds"],
                state["kde"],
                state["ims"],
                state["ime"],
                state["jms"],
                state["jme"],
                state["kms"],
                state["kme"],
                state["ips"],
                state["ipe"],
                state["jps"],
                state["jpe"],
                state["kps"],
                state["kpe"],
                state["its"],
                state["ite"],
                state["jts"],
                state["jte"],
                state["kps"],
                state["kpe"],
            )
            mask = _spec_bdytend_mask(
                state,
                bxs=bxs,
                bxe=bxe,
                bys=bys,
                bye=bye,
                btxs=btxs,
                btxe=btxe,
                btys=btys,
                btye=btye,
                variable="q",
            )
            species_tend = _overlay_specified_boundary(relaxed, specified, mask)

        scalar_1, scalar_2, advh_t, advz_t = rk_update_scalar_fn(
            1,
            1,
            species_old[..., None],
            species[..., None],
            species_tend[..., None],
            None,
            None,
            advect_tend,
            h_tendency,
            z_tendency,
            state["msftx"],
            state["msfty"],
            state["c1h"],
            state["c2h"],
            state["mu_1"],
            state["mu_2"],
            state["mub"],
            int(state["rk_step"]),
            float(state["dt_rk"]),
            int(getattr(state["config_flags"], "spec_zone", state.get("spec_zone", 1))),
            rk_update_scalar_config,
            False,
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["kds"],
            state["kde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["kms"],
            state["kme"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            state["kps"],
            state["kpe"],
            state["_wrf_real_rounding_token"],
        )
        del advh_t, advz_t
        species_out = jnp.asarray(scalar_2)[..., 0]
        scalar_old_out = scalar_old_out.at[..., species_index].set(jnp.asarray(scalar_1)[..., 0])
        scalar_out = scalar_out.at[..., species_index].set(species_out)
        scalar_tend_out = scalar_tend_out.at[..., species_index].set(species_tend)
        last_advect_tend = advect_tend
        last_h_tendency = h_tendency
        last_z_tendency = z_tendency

    synced = _scalar_fields_from_array(scalar_out)
    return {
        "scalar_old": scalar_old_out,
        "scalar": scalar_out,
        **synced,
        "scalar_tend": scalar_tend_out,
        "advect_tend": last_advect_tend,
        "h_tendency": last_h_tendency,
        "z_tendency": last_z_tendency,
        "ww_e": ww_e,
        "ww_i": ww_i,
    }


def _set_phys_bc_dry_2_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = set_phys_bc_dry_2(
        state["config_flags"],
        state["u_1"],
        state["u_2"],
        state["v_1"],
        state["v_2"],
        state["w_1"],
        state["w_2"],
        state["t_1"],
        state["t_2"],
        state["ph_1"],
        state["ph_2"],
        state["mu_1"],
        state["mu_2"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {
        "u_1": result.u_1,
        "u_2": result.u_2,
        "v_1": result.v_1,
        "v_2": result.v_2,
        "w_1": result.w_1,
        "w_2": result.w_2,
        "t_1": result.t_1,
        "t_2": result.t_2,
        "ph_1": result.ph_1,
        "ph_2": result.ph_2,
        "mu_1": result.mu_1,
        "mu_2": result.mu_2,
    }


def _rk_phys_bc_dry_2_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    result = rk_phys_bc_dry_2(
        state["config_flags"],
        state["u_2"],
        state["v_2"],
        state["w_2"],
        state["t_2"],
        state["ph_2"],
        state["mu_2"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["ips"],
        state["ipe"],
        state["jps"],
        state["jpe"],
        state["kps"],
        state["kpe"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {
        "u_2": result.u,
        "v_2": result.v,
        "w_2": result.w,
        "t_2": result.t,
        "ph_2": result.ph,
        "mu_2": result.mu,
    }


def _spec_bdy_final_stage(stage_name: str) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    target = stage_name.removeprefix("spec_bdy_final:")
    mapping = {
        "u": ("u_2", "muus", "c1h", "c2h", "msfuy", "u", False),
        "v": ("v_2", "muvs", "c1h", "c2h", "msfvx", "v", False),
        "t": ("t_2", "muts", "c1h", "c2h", "msfty", "t", False),
        "ph": ("ph_2", "muts", "c1f", "c2f", "msfty", "h", False),
        "mu": ("mu_2", "muts", "c1h", "c2h", "msfty", "m", True),
    }
    field_name, mu_name, c1_name, c2_name, msf_name, variable, is_2d_mass = mapping[target]

    def _apply(state: dict[str, Any]) -> Mapping[str, Any]:
        width, spec_zone, _relax_zone = _boundary_params(state)
        field = state[field_name]
        if is_2d_mass:
            field = jnp.asarray(field)[:, None, :]
            kds = kde = kms = kme = kps = kpe = kts = kte = 1
        else:
            kds = state["kds"]
            kde = state["kde"]
            kms = state["kms"]
            kme = state["kme"]
            kps = state["kps"]
            kpe = state["kpe"]
            kts = state["kps"]
            kte = state["kpe"]
        updated = spec_bdy_final(
            field,
            state[mu_name],
            state[c1_name],
            state[c2_name],
            state[msf_name],
            state[f"{target}_bxs"],
            state[f"{target}_bxe"],
            state[f"{target}_bys"],
            state[f"{target}_bye"],
            state[f"{target}_btxs"],
            state[f"{target}_btxe"],
            state[f"{target}_btys"],
            state[f"{target}_btye"],
            variable,
            state["config_flags"],
            width,
            spec_zone,
            _state_dtbc(state),
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            kds,
            kde,
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            kms,
            kme,
            state["ips"],
            state["ipe"],
            state["jps"],
            state["jpe"],
            kps,
            kpe,
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            kts,
            kte,
        )
        if is_2d_mass:
            updated = jnp.asarray(updated)[:, 0, :]
        return {field_name: updated}

    return _apply


def _spec_bdy_final_moist_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Apply WRF's final specified-boundary update to active moist species."""

    moist = jnp.asarray(state["moist"])
    if moist.ndim != 4 or moist.shape[-1] <= 1:
        return {}
    width, spec_zone, _relax_zone = _boundary_params(state)
    config_flags = state["config_flags"]
    update_non_qv = bool(getattr(config_flags, "nested", False)) or (
        bool(getattr(config_flags, "specified", False)) and bool(getattr(config_flags, "have_bcs_moist", False))
    )
    max_species = min(int(state.get("num_moist", moist.shape[-1])) - 1, moist.shape[-1] - 1)
    updates: dict[int, Any] = {}
    fallback = {
        "_bxs": jnp.zeros_like(state["qvapor_bxs"]),
        "_bxe": jnp.zeros_like(state["qvapor_bxe"]),
        "_bys": jnp.zeros_like(state["qvapor_bys"]),
        "_bye": jnp.zeros_like(state["qvapor_bye"]),
        "_btxs": jnp.zeros_like(state["qvapor_btxs"]),
        "_btxe": jnp.zeros_like(state["qvapor_btxe"]),
        "_btys": jnp.zeros_like(state["qvapor_btys"]),
        "_btye": jnp.zeros_like(state["qvapor_btye"]),
    }
    for species_index in range(1, max_species + 1):
        if species_index != 1 and not update_non_qv:
            continue
        updates[species_index] = spec_bdy_final(
            moist[..., species_index],
            state["muts"],
            state["c1h"],
            state["c2h"],
            state["msfty"],
            _moist_boundary_component(state, species_index, "_bxs", fallback["_bxs"]),
            _moist_boundary_component(state, species_index, "_bxe", fallback["_bxe"]),
            _moist_boundary_component(state, species_index, "_bys", fallback["_bys"]),
            _moist_boundary_component(state, species_index, "_bye", fallback["_bye"]),
            _moist_boundary_component(state, species_index, "_btxs", fallback["_btxs"]),
            _moist_boundary_component(state, species_index, "_btxe", fallback["_btxe"]),
            _moist_boundary_component(state, species_index, "_btys", fallback["_btys"]),
            _moist_boundary_component(state, species_index, "_btye", fallback["_btye"]),
            "t",
            config_flags,
            width,
            spec_zone,
            _state_dtbc(state),
            state["ids"],
            state["ide"],
            state["jds"],
            state["jde"],
            state["kds"],
            state["kde"],
            state["ims"],
            state["ime"],
            state["jms"],
            state["jme"],
            state["kms"],
            state["kme"],
            state["ips"],
            state["ipe"],
            state["jps"],
            state["jpe"],
            state["kps"],
            state["kpe"],
            state["its"],
            state["ite"],
            state["jts"],
            state["jte"],
            state["kps"],
            state["kpe"],
        )
    if not updates:
        return {}
    moist = _replace_moist_components(moist, updates)
    return {"moist": moist, **_moist_fields_from_array(moist)}


def _spec_bdy_final_moist_qv_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    """Backward-compatible alias for the full WRF moist final boundary loop."""

    return _spec_bdy_final_moist_stage(state)


def _set_w_surface_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    updated = set_w_surface(
        state["config_flags"],
        state["znw"],
        False,
        state["w_2"],
        state["ht"],
        state["u_2"],
        state["v_2"],
        state["cf1"],
        state["cf2"],
        state["cf3"],
        1.0 / float(state["grid"].dx),
        1.0 / float(state["grid"].dy),
        state["msftx"],
        state["msfty"],
        state["ids"],
        state["ide"],
        state["jds"],
        state["jde"],
        state["kds"],
        state["kde"],
        state["ims"],
        state["ime"],
        state["jms"],
        state["jme"],
        state["kms"],
        state["kme"],
        state["its"],
        state["ite"],
        state["jts"],
        state["jte"],
        state["kps"],
        state["kpe"],
    )
    return {"w_2": updated}


def _first_rk_step_part2_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    def _conv_t_tendf_to_moist_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        traced = _has_traced_array(inner_state["t_1"])
        t_tendf = conv_t_tendf_to_moist(
            inner_state["t_1"],
            jnp.asarray(inner_state["moist_old"])[..., 1]
            if jnp.asarray(inner_state["moist_old"]).ndim == 4
            else jnp.asarray(inner_state["moist_old"]),
            inner_state["t_tendf"],
            jnp.asarray(inner_state["moist_tend"])[..., 1],
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            rounding_token=inner_state["_wrf_real_rounding_token"] if traced else None,
        )
        return {"t_tendf": t_tendf}

    def _horizontal_diffusion_2_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        base = _field_or(inner_state, "z", jnp.zeros_like(inner_state["ph"]))
        zeros = jnp.zeros_like(base)
        nba_mij = _field_or(inner_state, "nba_mij", jnp.zeros(base.shape + (6,), dtype=base.dtype))
        horizontal_diffusion_2_fn = (
            _horizontal_diffusion_2_jit_result
            if _JIT_HORIZONTAL_DIFFUSION_2
            else horizontal_diffusion_2_without_theta_vectorized
        )
        horizontal_diffusion_s_fn = (
            _horizontal_diffusion_s_jit
            if _JIT_HORIZONTAL_DIFFUSION_2
            else horizontal_diffusion_s_vectorized
        )
        hdiff_config = _static_config(inner_state) if _JIT_HORIZONTAL_DIFFUSION_2 else inner_state["config_flags"]
        theta_for_diffusion = _field_or(inner_state, "t_2", inner_state["t"])
        rho_for_diffusion = _field_or(inner_state, "rho", zeros)
        call_args = (
            inner_state["t_tendf"],
            inner_state["ru_tendf"],
            inner_state["rv_tendf"],
            inner_state["rw_tendf"],
            inner_state["tke_tend"],
            inner_state["moist_tend"],
            int(inner_state.get("num_moist", inner_state["moist"].shape[-1])),
            inner_state["chem_tend"],
            int(inner_state.get("num_chem", 0)),
            inner_state["scalar_tend"],
            int(inner_state.get("num_scalar", 0)),
            inner_state["tracer_tend"],
            int(inner_state.get("num_tracer", 0)),
            theta_for_diffusion,
            inner_state["th_phy"],
            _field_or(inner_state, "tke_2", _field_or(inner_state, "tke", zeros)),
            hdiff_config,
            inner_state["defor11"],
            inner_state["defor22"],
            inner_state["defor12"],
            inner_state["defor13"],
            inner_state["defor23"],
            nba_mij,
            int(inner_state.get("n_nba_mij", nba_mij.shape[-1])),
            inner_state["div"],
            inner_state["moist"],
            inner_state["chem"],
            inner_state["scalar"],
            inner_state["tracer"],
            inner_state["msfux"],
            inner_state["msfuy"],
            inner_state["msfvx"],
            inner_state["msfvy"],
            inner_state["msftx"],
            inner_state["msfty"],
            inner_state["xkmh"],
            inner_state["xkmv"],
            inner_state["xkhh"],
            int(inner_state["config_flags"].km_opt),
            1.0 / float(inner_state["grid"].dx),
            1.0 / float(inner_state["grid"].dy),
            inner_state["rdz"],
            inner_state["rdzw"],
            inner_state["fnm"],
            inner_state["fnp"],
            _scalar_float(inner_state, "cf1"),
            _scalar_float(inner_state, "cf2"),
            _scalar_float(inner_state, "cf3"),
            inner_state["zx"],
            inner_state["zy"],
            inner_state["dn"],
            inner_state["dnw"],
            rho_for_diffusion,
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state["_wrf_real_rounding_token"],
        )
        result = horizontal_diffusion_2_fn(*call_args)
        rt_tendf = horizontal_diffusion_s_fn(
            inner_state["t_tendf"],
            hdiff_config,
            theta_for_diffusion,
            inner_state["msftx"],
            inner_state["msfty"],
            inner_state["msfux"],
            inner_state["msfuy"],
            inner_state["msfvx"],
            inner_state["msfvy"],
            inner_state["xkhh"],
            1.0 / float(inner_state["grid"].dx),
            1.0 / float(inner_state["grid"].dy),
            inner_state["fnm"],
            inner_state["fnp"],
            _scalar_float(inner_state, "cf1"),
            _scalar_float(inner_state, "cf2"),
            _scalar_float(inner_state, "cf3"),
            inner_state["zx"],
            inner_state["zy"],
            inner_state["rdz"],
            inner_state["rdzw"],
            inner_state["dnw"],
            inner_state["dn"],
            rho_for_diffusion,
            False,
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state["_wrf_real_rounding_token"],
        )
        updates = {
            "t_tendf": rt_tendf,
            "ru_tendf": result.ru_tendf,
            "rv_tendf": result.rv_tendf,
            "rw_tendf": result.rw_tendf,
            "tke_tend": result.tke_tendf,
            "moist_tend": result.moist_tendf,
            "chem_tend": result.chem_tendf,
            "scalar_tend": result.scalar_tendf,
            "tracer_tend": result.tracer_tendf,
            "nba_mij": result.nba_mij,
        }
        return updates

    def _update_phy_ten_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        zero_mass = jnp.zeros_like(inner_state["t_tendf"])
        zero_ph = jnp.zeros_like(inner_state["ph_tendf"])
        zero_mu = jnp.zeros_like(inner_state["mu_tendf"])
        result = update_phy_ten(
            inner_state["ph_tendf"],
            inner_state["t_tendf"],
            inner_state["ru_tendf"],
            inner_state["rv_tendf"],
            inner_state["moist_tend"],
            inner_state["scalar_tend"],
            inner_state["mu_tendf"],
            _field_or(inner_state, "rthraten", zero_mass),
            _field_or(inner_state, "rthblten", zero_mass),
            _field_or(inner_state, "rthcuten", zero_mass),
            _field_or(inner_state, "rthshten", zero_mass),
            _field_or(inner_state, "rublten", zero_mass),
            _field_or(inner_state, "rucuten", zero_mass),
            _field_or(inner_state, "rushten", zero_mass),
            _field_or(inner_state, "rvblten", zero_mass),
            _field_or(inner_state, "rvcuten", zero_mass),
            _field_or(inner_state, "rvshten", zero_mass),
            _field_or(inner_state, "rqvblten", zero_mass),
            _field_or(inner_state, "rqcblten", zero_mass),
            _field_or(inner_state, "rqiblten", zero_mass),
            _field_or(inner_state, "rqniblten", zero_mass),
            _field_or(inner_state, "rqvcuten", zero_mass),
            _field_or(inner_state, "rqccuten", zero_mass),
            _field_or(inner_state, "rqrcuten", zero_mass),
            _field_or(inner_state, "rqicuten", zero_mass),
            _field_or(inner_state, "rqscuten", zero_mass),
            _field_or(inner_state, "rqcncuten", zero_mass),
            _field_or(inner_state, "rqincuten", zero_mass),
            _field_or(inner_state, "rqvshten", zero_mass),
            _field_or(inner_state, "rqcshten", zero_mass),
            _field_or(inner_state, "rqrshten", zero_mass),
            _field_or(inner_state, "rqishten", zero_mass),
            _field_or(inner_state, "rqsshten", zero_mass),
            _field_or(inner_state, "rqgshten", zero_mass),
            _field_or(inner_state, "rqcnshten", zero_mass),
            _field_or(inner_state, "rqinshten", zero_mass),
            _field_or(inner_state, "rundgdten", zero_mass),
            _field_or(inner_state, "rvndgdten", zero_mass),
            _field_or(inner_state, "rthndgdten", zero_mass),
            _field_or(inner_state, "rphndgdten", zero_ph),
            _field_or(inner_state, "rqvndgdten", zero_mass),
            _field_or(inner_state, "rmundgdten", zero_mu),
            _field_or(inner_state, "rthfrten", zero_mass),
            _field_or(inner_state, "rqvfrten", zero_mass),
            int(inner_state.get("num_moist", inner_state["moist_tend"].shape[-1])),
            int(inner_state.get("num_scalar", 0)),
            inner_state["config_flags"],
            int(inner_state.get("rk_step", 1)),
            bool(inner_state.get("adv_moist_cond", False)),
            ids=inner_state["ids"],
            ide=inner_state["ide"],
            jds=inner_state["jds"],
            jde=inner_state["jde"],
            kds=inner_state["kds"],
            kde=inner_state["kde"],
            ims=inner_state["ims"],
            ime=inner_state["ime"],
            jms=inner_state["jms"],
            jme=inner_state["jme"],
            kms=inner_state["kms"],
            kme=inner_state["kme"],
            its=inner_state.get("its", inner_state["ids"]),
            ite=inner_state.get("ite", inner_state["ide"] - 1),
            jts=inner_state.get("jts", inner_state["jds"]),
            jte=inner_state.get("jte", inner_state["jde"] - 1),
            kts=inner_state["kps"],
            kte=inner_state["kpe"],
        )
        return {
            "ph_tendf": result.rph_tendf,
            "t_tendf": result.rt_tendf,
            "ru_tendf": result.ru_tendf,
            "rv_tendf": result.rv_tendf,
            "moist_tend": result.moist_tendf,
            "scalar_tend": result.scalar_tendf,
            "mu_tendf": result.mu_tendf,
        }

    def _phy_bc_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        base = _field_or(inner_state, "z", jnp.zeros_like(inner_state["ph"]))
        zeros = jnp.zeros_like(base)
        result = phy_bc(
            inner_state["config_flags"],
            inner_state["div"],
            inner_state["defor11"],
            inner_state["defor22"],
            inner_state["defor33"],
            inner_state["defor12"],
            inner_state["defor13"],
            inner_state["defor23"],
            inner_state["xkmh"],
            inner_state["xkmv"],
            inner_state["xkhh"],
            inner_state["xkhv"],
            _field_or(inner_state, "tke_2", _field_or(inner_state, "tke", zeros)),
            _field_or(inner_state, "rho", zeros),
            _field_or(inner_state, "rublten", zeros),
            _field_or(inner_state, "rvblten", zeros),
            _field_or(inner_state, "rucuten", zeros),
            _field_or(inner_state, "rvcuten", zeros),
            _field_or(inner_state, "rushten", zeros),
            _field_or(inner_state, "rvshten", zeros),
            _field_or(inner_state, "gamu", jnp.zeros_like(inner_state["mut"])),
            _field_or(inner_state, "gamv", jnp.zeros_like(inner_state["mut"])),
            _field_or(inner_state, "xkmv_meso", zeros),
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state["ips"],
            inner_state["ipe"],
            inner_state["jps"],
            inner_state["jpe"],
            inner_state["kps"],
            inner_state["kpe"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
        )
        return {
            "div": result.div,
            "defor11": result.defor11,
            "defor22": result.defor22,
            "defor33": result.defor33,
            "defor12": result.defor12,
            "defor13": result.defor13,
            "defor23": result.defor23,
            "xkmh": result.xkmh,
            "xkmv": result.xkmv,
            "xkhh": result.xkhh,
            "xkhv": result.xkhv,
            "tke_2": result.tke,
            "rho": result.rho,
            "rublten": result.rublten,
            "rvblten": result.rvblten,
            "rucuten": result.rucuten,
            "rvcuten": result.rvcuten,
            "rushten": result.rushten,
            "rvshten": result.rvshten,
            "gamu": result.gamu,
            "gamv": result.gamv,
            "xkmv_meso": result.xkmv_meso,
        }

    def _calculate_km_kh_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        base = _field_or(inner_state, "z", jnp.zeros_like(inner_state["ph"]))
        zeros = jnp.zeros_like(base)
        zeros2 = jnp.zeros_like(inner_state["mut"])
        calculate_km_kh_fn = _calculate_km_kh_jit_result if _JIT_CALCULATE_KM_KH else calculate_km_kh_vectorized
        km_kh_config = _static_config(inner_state) if _JIT_CALCULATE_KM_KH else inner_state["config_flags"]
        result = calculate_km_kh_fn(
            km_kh_config,
            float(inner_state["grid"].dt),
            float(_field_or(inner_state, "dampcoef", getattr(inner_state["config_flags"], "dampcoef", 0.0))),
            float(_field_or(inner_state, "zdamp", getattr(inner_state["config_flags"], "zdamp", 0.0))),
            int(getattr(inner_state["config_flags"], "damp_opt", 0)),
            _field_or(inner_state, "xkmh", zeros),
            _field_or(inner_state, "xkmv", zeros),
            _field_or(inner_state, "xkhh", zeros),
            _field_or(inner_state, "xkhv", zeros),
            _field_or(inner_state, "bn2", zeros),
            float(_field_or(inner_state, "khdif", getattr(inner_state["config_flags"], "khdif", 0.0))),
            float(_field_or(inner_state, "kvdif", getattr(inner_state["config_flags"], "kvdif", 0.0))),
            inner_state["div"],
            inner_state["defor11"],
            inner_state["defor22"],
            inner_state["defor33"],
            inner_state["defor12"],
            inner_state["defor13"],
            inner_state["defor23"],
            _field_or(inner_state, "tke_2", _field_or(inner_state, "tke", zeros)),
            inner_state["p8w"],
            inner_state["t8w"],
            inner_state["th_phy"],
            inner_state["t_phy"],
            inner_state["p_phy"],
            inner_state["moist"],
            inner_state["dn"],
            inner_state["dnw"],
            float(inner_state["grid"].dx),
            float(inner_state["grid"].dy),
            inner_state["rdz"],
            inner_state["rdzw"],
            int(getattr(inner_state["config_flags"], "mix_isotropic", 0)),
            int(inner_state.get("num_moist", 2)),
            _scalar_float(inner_state, "cf1"),
            _scalar_float(inner_state, "cf2"),
            _scalar_float(inner_state, "cf3"),
            bool(_field_or(inner_state, "warm_rain", getattr(inner_state["config_flags"], "warm_rain", False))),
            float(_field_or(inner_state, "mix_upper_bound", getattr(inner_state["config_flags"], "mix_upper_bound", 100.0))),
            inner_state["msftx"],
            inner_state["msfty"],
            inner_state["zx"],
            inner_state["zy"],
            _field_or(inner_state, "pblh", zeros2),
            _field_or(inner_state, "dlk", zeros),
            _field_or(inner_state, "xkmv_meso", zeros),
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state.get("_wrf_real_rounding_token"),
        )
        return {
            "xkmh": result.xkmh,
            "xkmv": result.xkmv,
            "xkhh": result.xkhh,
            "xkhv": result.xkhv,
            "bn2": result.bn2,
            "moist": result.moist,
            "tke_2": result.tke,
            "pblh": result.hpbl,
            "xkmv_meso": result.xkmv_meso,
        }

    def _cal_deform_and_div_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        base = _field_or(inner_state, "z", jnp.zeros_like(inner_state["ph"]))
        zeros = jnp.zeros_like(base)
        cal_deform_and_div_fn = _cal_deform_and_div_jit_result if _JIT_CAL_DEFORM_AND_DIV else cal_deform_and_div_vectorized
        deform_config = _static_config(inner_state) if _JIT_CAL_DEFORM_AND_DIV else inner_state["config_flags"]
        result = cal_deform_and_div_fn(
            deform_config,
            _field_or(inner_state, "u_2", inner_state["u"]),
            _field_or(inner_state, "v_2", inner_state["v"]),
            _field_or(inner_state, "w_2", inner_state["w"]),
            _field_or(inner_state, "div", zeros),
            _field_or(inner_state, "defor11", zeros),
            _field_or(inner_state, "defor22", zeros),
            _field_or(inner_state, "defor33", zeros),
            _field_or(inner_state, "defor12", zeros),
            _field_or(inner_state, "defor13", zeros),
            _field_or(inner_state, "defor23", zeros),
            inner_state.get("nba_rij"),
            int(inner_state.get("n_nba_rij", 0)),
            inner_state["u_base"],
            inner_state["v_base"],
            inner_state["msfux"],
            inner_state["msfuy"],
            inner_state["msfvx"],
            inner_state["msfvy"],
            inner_state["msftx"],
            inner_state["msfty"],
            1.0 / float(inner_state["grid"].dx),
            1.0 / float(inner_state["grid"].dy),
            inner_state["dn"],
            inner_state["dnw"],
            inner_state["rdz"],
            inner_state["rdzw"],
            inner_state["fnm"],
            inner_state["fnp"],
            _scalar_float(inner_state, "cf1"),
            _scalar_float(inner_state, "cf2"),
            _scalar_float(inner_state, "cf3"),
            inner_state["zx"],
            inner_state["zy"],
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state.get("_wrf_real_rounding_token"),
        )
        return {
            "div": result.div,
            "defor11": result.defor11,
            "defor22": result.defor22,
            "defor33": result.defor33,
            "defor12": result.defor12,
            "defor13": result.defor13,
            "defor23": result.defor23,
            "nba_rij": result.nba_rij,
        }

    def _compute_diff_metrics_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        base = _field_or(inner_state, "z", jnp.zeros_like(inner_state["ph"]))
        zeros = jnp.zeros_like(base)
        compute_diff_metrics_fn = _compute_diff_metrics_jit_result if _JIT_COMPUTE_DIFF_METRICS else compute_diff_metrics
        diff_metrics_config = _static_config(inner_state) if _JIT_COMPUTE_DIFF_METRICS else inner_state["config_flags"]
        result = compute_diff_metrics_fn(
            diff_metrics_config,
            inner_state["ph"],
            inner_state["phb"],
            base,
            _field_or(inner_state, "rdz", zeros),
            _field_or(inner_state, "rdzw", zeros),
            _field_or(inner_state, "zx", zeros),
            _field_or(inner_state, "zy", zeros),
            1.0 / float(inner_state["grid"].dx),
            1.0 / float(inner_state["grid"].dy),
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state["_wrf_real_rounding_token"] if _JIT_COMPUTE_DIFF_METRICS else None,
        )
        return {
            "z": result.z,
            "rdz": result.rdz,
            "rdzw": result.rdzw,
            "zx": result.zx,
            "zy": result.zy,
        }

    def _calculate_phy_tend_stage(inner_state: dict[str, Any]) -> Mapping[str, Any]:
        zero3 = jnp.zeros_like(inner_state["t_tendf"])
        cfg = inner_state["config_flags"]
        num_scalar = int(inner_state.get("num_scalar", 0))
        num_tracer = int(inner_state.get("num_tracer", 0))
        if (
            _FAST_PBL_ONLY_PHY_TEND
            and _JIT_CALCULATE_PHY_TEND
            and int(getattr(cfg, "ra_lw_physics", 0)) == 0
            and int(getattr(cfg, "ra_sw_physics", 0)) == 0
            and int(getattr(cfg, "cu_physics", 0)) == 0
            and int(getattr(cfg, "shcu_physics", 0)) == 0
            and int(getattr(cfg, "grid_fdda", 0)) == 0
            and int(getattr(cfg, "bl_pbl_physics", 0)) > 0
            and num_scalar <= 1
            and num_tracer == 0
        ):
            (
                rublten,
                rvblten,
                rthblten,
                rqvblten,
                rqcblten,
                rqiblten,
            ) = _calculate_phy_tend_pbl_only_jit_tuple(
                inner_state["c1h"],
                inner_state["c2h"],
                inner_state["mut"],
                _field_or(inner_state, "rublten", zero3),
                _field_or(inner_state, "rvblten", zero3),
                _field_or(inner_state, "rthblten", zero3),
                _field_or(inner_state, "rqvblten", zero3),
                _field_or(inner_state, "rqcblten", zero3),
                _field_or(inner_state, "rqiblten", zero3),
                inner_state["_wrf_real_rounding_token"],
                ids=inner_state["ids"],
                ide=inner_state["ide"],
                jds=inner_state["jds"],
                jde=inner_state["jde"],
                kds=inner_state["kds"],
                kde=inner_state["kde"],
                ims=inner_state["ims"],
                jms=inner_state["jms"],
                kms=inner_state["kms"],
                its=inner_state.get("its", inner_state["ids"]),
                ite=inner_state.get("ite", inner_state["ide"] - 1),
                jts=inner_state.get("jts", inner_state["jds"]),
                jte=inner_state.get("jte", inner_state["jde"] - 1),
                kts=inner_state["kps"],
                kte=inner_state["kpe"],
            )
            return {
                "rublten": rublten,
                "rvblten": rvblten,
                "rthblten": rthblten,
                "rqvblten": rqvblten,
                "rqcblten": rqcblten,
                "rqiblten": rqiblten,
            }
        calculate_phy_tend_fn = _calculate_phy_tend_jit_tuple if _JIT_CALCULATE_PHY_TEND else calculate_phy_tend
        phy_tend_config = _static_config(inner_state) if _JIT_CALCULATE_PHY_TEND else inner_state["config_flags"]
        result = calculate_phy_tend_fn(
            phy_tend_config,
            inner_state["c1h"],
            inner_state["c2h"],
            inner_state["mut"],
            inner_state["muu"],
            inner_state["muv"],
            inner_state.get("pi_phy", inner_state.get("pi")),
            _field_or(inner_state, "rthraten", zero3),
            _field_or(inner_state, "rublten", zero3),
            _field_or(inner_state, "rvblten", zero3),
            _field_or(inner_state, "rthblten", zero3),
            _field_or(inner_state, "rqvblten", zero3),
            _field_or(inner_state, "rqcblten", zero3),
            _field_or(inner_state, "rqiblten", zero3),
            _field_or(inner_state, "rucuten", zero3),
            _field_or(inner_state, "rvcuten", zero3),
            _field_or(inner_state, "rthcuten", zero3),
            _field_or(inner_state, "rqvcuten", zero3),
            _field_or(inner_state, "rqccuten", zero3),
            _field_or(inner_state, "rqrcuten", zero3),
            _field_or(inner_state, "rqicuten", zero3),
            _field_or(inner_state, "rqscuten", zero3),
            _field_or(inner_state, "rushten", zero3),
            _field_or(inner_state, "rvshten", zero3),
            _field_or(inner_state, "rthshten", zero3),
            _field_or(inner_state, "rqvshten", zero3),
            _field_or(inner_state, "rqcshten", zero3),
            _field_or(inner_state, "rqrshten", zero3),
            _field_or(inner_state, "rqishten", zero3),
            _field_or(inner_state, "rqsshten", zero3),
            _field_or(inner_state, "rqgshten", zero3),
            _field_or(inner_state, "rundgdten", zero3),
            _field_or(inner_state, "rvndgdten", zero3),
            _field_or(inner_state, "rthndgdten", zero3),
            _field_or(inner_state, "rqvndgdten", zero3),
            _field_or(inner_state, "rmundgdten", jnp.zeros_like(inner_state["mut"])),
            inner_state["scalar"],
            inner_state["scalar_tend"],
            num_scalar,
            inner_state["tracer"],
            inner_state["tracer_tend"],
            num_tracer,
            inner_state["ids"],
            inner_state["ide"],
            inner_state["jds"],
            inner_state["jde"],
            inner_state["kds"],
            inner_state["kde"],
            inner_state["ims"],
            inner_state["ime"],
            inner_state["jms"],
            inner_state["jme"],
            inner_state["kms"],
            inner_state["kme"],
            inner_state.get("its", inner_state["ids"]),
            inner_state.get("ite", inner_state["ide"] - 1),
            inner_state.get("jts", inner_state["jds"]),
            inner_state.get("jte", inner_state["jde"] - 1),
            inner_state["kps"],
            inner_state["kpe"],
            inner_state["_wrf_real_rounding_token"],
        )
        names = (
            "rthraten",
            "rublten",
            "rvblten",
            "rthblten",
            "rqvblten",
            "rqcblten",
            "rqiblten",
            "rucuten",
            "rvcuten",
            "rthcuten",
            "rqvcuten",
            "rqccuten",
            "rqrcuten",
            "rqicuten",
            "rqscuten",
            "rushten",
            "rvshten",
            "rthshten",
            "rqvshten",
            "rqcshten",
            "rqrshten",
            "rqishten",
            "rqsshten",
            "rqgshten",
            "rundgdten",
            "rvndgdten",
            "rthndgdten",
            "rqvndgdten",
            "rmundgdten",
            "scalar_tend",
            "tracer_tend",
        )
        return dict(zip(names, result, strict=True))

    inner_callbacks = {
        "calculate_phy_tend": _calculate_phy_tend_stage,
        "compute_diff_metrics": _compute_diff_metrics_stage,
        "cal_deform_and_div": _cal_deform_and_div_stage,
        "calculate_km_kh": _calculate_km_kh_stage,
        "phy_bc": _phy_bc_stage,
        "update_phy_ten": _update_phy_ten_stage,
        "conv_t_tendf_to_moist": _conv_t_tendf_to_moist_stage,
        "horizontal_diffusion_2": _horizontal_diffusion_2_stage,
    }
    for stage_name in (
        "set_physical_bc3d:rdzw",
        "set_physical_bc3d:rdz",
        "set_physical_bc3d:z",
        "set_physical_bc3d:zx",
        "set_physical_bc3d:zy",
    ):
        inner_callbacks[stage_name] = _set_physical_bc3d_stage(stage_name)
    for stage_name in ("set_physical_bc2d:ustm", "set_physical_bc2d:ust"):
        inner_callbacks[stage_name] = _set_physical_bc2d_stage(stage_name)
    inner_debug_fn = state.get("first_rk_step_part2_inner_debug_fn")
    if inner_debug_fn is not None:
        wrapped_inner_callbacks = {}

        def _debug_wrapper(name: str, callback: Callable[[dict[str, Any]], Mapping[str, Any] | None]) -> Callable[[dict[str, Any]], Mapping[str, Any] | None]:
            def _wrapped(inner: dict[str, Any]) -> Mapping[str, Any] | None:
                updates = callback(inner)
                after = dict(inner)
                if updates:
                    after.update(updates)
                inner_debug_fn(name, inner, after, updates)
                return updates

            return _wrapped

        for inner_name, inner_callback in inner_callbacks.items():
            wrapped_inner_callbacks[inner_name] = _debug_wrapper(inner_name, inner_callback)
        inner_callbacks = wrapped_inner_callbacks
    if _DIRECT_FIRST_RK_PART2:
        grid = state["grid"]
        config_flags = state["config_flags"]
        start_at = state.get("first_rk_step_part2_start_at")
        stop_after = state.get("first_rk_step_part2_stop_after")

        # A spatially resident executor may need to exchange halos between
        # dependent diffusion kernels.  Keep this split opt-in and static: the
        # ordinary production path still enters at the first stage and runs the
        # same sequence in one call.  The names describe scientific data
        # boundaries, not implementation callbacks.
        _part2_order = (
            "calculate_phy_tend",
            "compute_diff_metrics",
            "diffusion_bc",
            "cal_deform_and_div",
            "calculate_km_kh",
            "phy_bc",
            "update_phy_ten",
            "conv_t_tendf_to_moist",
            "horizontal_diffusion_2",
        )
        if start_at is None:
            start_index = 0
        else:
            try:
                start_index = _part2_order.index(str(start_at))
            except ValueError as error:
                raise ValueError(
                    f"first_rk_step_part2_start_at must be one of {_part2_order}, got {start_at!r}"
                ) from error

        def _enabled(name: str) -> bool:
            return _part2_order.index(name) >= start_index

        def _stop(name: str) -> bool:
            return stop_after is not None and str(stop_after) == name

        def _ri(source: Any, name: str, default: int = 0) -> int:
            return int(getattr(source, name, default))

        unsupported = (
            _ri(grid, "skebs_on", 0) == 1
            or _ri(grid, "sppt_on", 0) == 1
            or _ri(grid, "rand_perturb_on", 0) == 1
            or _ri(config_flags, "multi_perturb", 0) == 1
            or _ri(grid, "spp_conv", 0) == 1
            or _ri(grid, "spp_pbl", 0) == 1
            or _ri(grid, "spp_lsm", 0) == 1
            or _ri(config_flags, "grid_fdda", 0) == 1
            or _ri(grid, "obs_nudge_opt", 0) == 1
            or _ri(config_flags, "sfs_opt", 0) > 0
            or (_ri(config_flags, "diff_opt", 0) == 2 and _ri(config_flags, "bl_pbl_physics", 0) == 0)
            or _ri(config_flags, "nwp_diagnostics", 0) == 1
            or (_ri(config_flags, "afwa_diag_opt", 0) == 1 and _ri(config_flags, "afwa_severe_opt", 0) == 1)
            or (_ri(config_flags, "diff_opt", 0) == 2 and _ri(config_flags, "km_opt", 0) == 5)
        )
        if not unsupported:
            _set_current_dry_aliases_inplace(state)

            def _run(name: str) -> None:
                callback = inner_callbacks.get(name)
                if callback is None:
                    return
                updates = callback(state)
                if updates:
                    state.update(updates)

            if _enabled("calculate_phy_tend"):
                _run("calculate_phy_tend")
                if _stop("calculate_phy_tend"):
                    return None

            diff_opt = _ri(config_flags, "diff_opt", 0)
            km_opt = _ri(config_flags, "km_opt", 0)
            if diff_opt in {1, 2}:
                if _enabled("compute_diff_metrics"):
                    _run("compute_diff_metrics")
                    if _stop("compute_diff_metrics"):
                        return None
                if _enabled("diffusion_bc"):
                    if all(name in state for name in ("rdzw", "rdz", "z", "zx", "zy", "ustm", "ust")):
                        (
                            state["rdzw"],
                            state["rdz"],
                            state["z"],
                            state["zx"],
                            state["zy"],
                            state["ustm"],
                            state["ust"],
                        ) = _first_rk_part2_diffusion_bc_jit_tuple(
                            state["rdzw"],
                            state["rdz"],
                            state["z"],
                            state["zx"],
                            state["zy"],
                            state["ustm"],
                            state["ust"],
                            config_flags=_static_config(state),
                            ids=state["ids"],
                            ide=state["ide"],
                            jds=state["jds"],
                            jde=state["jde"],
                            kds=state["kds"],
                            kde=state["kde"],
                            ims=state["ims"],
                            ime=state["ime"],
                            jms=state["jms"],
                            jme=state["jme"],
                            kms=state["kms"],
                            kme=state["kme"],
                            ips=state["ips"],
                            ipe=state["ipe"],
                            jps=state["jps"],
                            jpe=state["jpe"],
                            kps=state["kps"],
                            kpe=state["kpe"],
                            its=state["its"],
                            ite=state["ite"],
                            jts=state["jts"],
                            jte=state["jte"],
                            kts=state["kps"],
                            kte=state["kpe"],
                        )
                    else:
                        for stage_name in (
                            "set_physical_bc3d:rdzw",
                            "set_physical_bc3d:rdz",
                            "set_physical_bc3d:z",
                            "set_physical_bc3d:zx",
                            "set_physical_bc3d:zy",
                            "set_physical_bc2d:ustm",
                            "set_physical_bc2d:ust",
                        ):
                            _run(stage_name)
                    if _stop("diffusion_bc"):
                        return None
                if _enabled("cal_deform_and_div"):
                    _run("cal_deform_and_div")
                    if _stop("cal_deform_and_div"):
                        return None
                if _enabled("calculate_km_kh"):
                    _run("calculate_km_kh")
                    if _stop("calculate_km_kh"):
                        return None

            if _enabled("phy_bc"):
                _run("phy_bc")
                if _stop("phy_bc"):
                    return None
            h_sca_adv_order = _ri(config_flags, "h_sca_adv_order", 0)
            if h_sca_adv_order > 6:
                raise ValueError(f"first_rk_step_part2 invalid h_sca_adv_order = {h_sca_adv_order}")
            if _enabled("update_phy_ten"):
                _run("update_phy_ten")
                if _stop("update_phy_ten"):
                    return None

            if _ri(config_flags, "use_theta_m", 0) == 1 and _enabled("conv_t_tendf_to_moist"):
                _run("conv_t_tendf_to_moist")
                if _stop("conv_t_tendf_to_moist"):
                    return None

            if diff_opt == 2 and _enabled("horizontal_diffusion_2"):
                _run("horizontal_diffusion_2")
                if _stop("horizontal_diffusion_2"):
                    return None

            return None
    result = first_rk_step_part2(
        grid=state["grid"],
        config_flags=state["config_flags"],
        fields=_with_current_dry_aliases(state),
        moist=state["moist"],
        moist_old=state.get("moist_old"),
        moist_tend=state["moist_tend"],
        chem=state["chem"],
        chem_tend=state["chem_tend"],
        scalar=state["scalar"],
        scalar_tend=state["scalar_tend"],
        tracer=state["tracer"],
        tracer_tend=state["tracer_tend"],
        ru_tendf=state["ru_tendf"],
        rv_tendf=state["rv_tendf"],
        rw_tendf=state["rw_tendf"],
        t_tendf=state["t_tendf"],
        ph_tendf=state["ph_tendf"],
        mu_tendf=state["mu_tendf"],
        tke_tend=state["tke_tend"],
        adapt_step_flag=state["adapt_step_flag"],
        curr_secs=state["curr_secs"],
        chklowq=state.get("chklowq"),
        th_phy=state.get("th_phy"),
        pi_phy=state.get("pi_phy"),
        p_phy=state.get("p_phy"),
        t_phy=state.get("t_phy"),
        dz8w=state.get("dz8w"),
        p8w=state.get("p8w"),
        t8w=state.get("t8w"),
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        ips=state["ips"],
        ipe=state["ipe"],
        jps=state["jps"],
        jpe=state["jpe"],
        kps=state["kps"],
        kpe=state["kpe"],
        imsx=state["imsx"],
        imex=state["imex"],
        jmsx=state["jmsx"],
        jmex=state["jmex"],
        kmsx=state["kmsx"],
        kmex=state["kmex"],
        ipsx=state["ipsx"],
        ipex=state["ipex"],
        jpsx=state["jpsx"],
        jpex=state["jpex"],
        kpsx=state["kpsx"],
        kpex=state["kpex"],
        imsy=state["imsy"],
        imey=state["imey"],
        jmsy=state["jmsy"],
        jmey=state["jmey"],
        kmsy=state["kmsy"],
        kmey=state["kmey"],
        ipsy=state["ipsy"],
        ipey=state["ipey"],
        jpsy=state["jpsy"],
        jpey=state["jpey"],
        kpsy=state["kpsy"],
        kpey=state["kpey"],
        k_start=state["kps"],
        k_end=state["kpe"],
        callbacks=inner_callbacks,
        record_events=False,
    )
    updates = result.fields
    if isinstance(updates, dict):
        updates["grid"] = result.grid
        return updates
    return dict(updates) | {"grid": result.grid}


def _first_rk_step_part1_stage(state: dict[str, Any]) -> Mapping[str, Any]:
    inner_callbacks = {
        "init_zero_tendency": _init_zero_tendency_stage,
        "phy_prep": _phy_prep_stage,
        "radiation_driver": _radiation_driver_stage,
        "surface_driver": _surface_driver_stage,
        "pbl_driver": _pbl_driver_stage,
        "cumulus_driver": _cumulus_driver_stage,
        "shallowcu_driver": _shallowcu_driver_stage,
        "fddagd_driver": _fddagd_driver_stage,
    }
    inner_debug_fn = state.get("first_rk_step_part1_inner_debug_fn")
    if inner_debug_fn is not None:
        wrapped_inner_callbacks = {}

        def _debug_wrapper(name: str, callback: Callable[[dict[str, Any]], Mapping[str, Any] | None]) -> Callable[[dict[str, Any]], Mapping[str, Any] | None]:
            def _wrapped(inner: dict[str, Any]) -> Mapping[str, Any] | None:
                updates = callback(inner)
                after = dict(inner)
                if updates:
                    after.update(updates)
                inner_debug_fn(name, inner, after, updates)
                return updates

            return _wrapped

        for inner_name, inner_callback in inner_callbacks.items():
            wrapped_inner_callbacks[inner_name] = _debug_wrapper(inner_name, inner_callback)
        inner_callbacks = wrapped_inner_callbacks
    result = first_rk_step_part1(
        grid=state["grid"],
        config_flags=state["config_flags"],
        fields=_with_current_dry_aliases(state),
        moist=state["moist"],
        moist_tend=state["moist_tend"],
        chem=state["chem"],
        chem_tend=state["chem_tend"],
        scalar=state["scalar"],
        scalar_tend=state["scalar_tend"],
        tracer=state["tracer"],
        tracer_tend=state["tracer_tend"],
        ru_tendf=state["ru_tendf"],
        rv_tendf=state["rv_tendf"],
        rw_tendf=state["rw_tendf"],
        t_tendf=state["t_tendf"],
        ph_tendf=state["ph_tendf"],
        mu_tendf=state["mu_tendf"],
        tke_tend=state["tke_tend"],
        adapt_step_flag=state["adapt_step_flag"],
        curr_secs=state["curr_secs"],
        ids=state["ids"],
        ide=state["ide"],
        jds=state["jds"],
        jde=state["jde"],
        kds=state["kds"],
        kde=state["kde"],
        ims=state["ims"],
        ime=state["ime"],
        jms=state["jms"],
        jme=state["jme"],
        kms=state["kms"],
        kme=state["kme"],
        ips=state["ips"],
        ipe=state["ipe"],
        jps=state["jps"],
        jpe=state["jpe"],
        kps=state["kps"],
        kpe=state["kpe"],
        imsx=state["imsx"],
        imex=state["imex"],
        jmsx=state["jmsx"],
        jmex=state["jmex"],
        kmsx=state["kmsx"],
        kmex=state["kmex"],
        ipsx=state["ipsx"],
        ipex=state["ipex"],
        jpsx=state["jpsx"],
        jpex=state["jpex"],
        kpsx=state["kpsx"],
        kpex=state["kpex"],
        imsy=state["imsy"],
        imey=state["imey"],
        jmsy=state["jmsy"],
        jmey=state["jmey"],
        kmsy=state["kmsy"],
        kmey=state["kmey"],
        ipsy=state["ipsy"],
        ipey=state["ipey"],
        jpsy=state["jpsy"],
        jpey=state["jpey"],
        kpsy=state["kpsy"],
        kpey=state["kpey"],
        k_start=state["kps"],
        k_end=state["kpe"],
        f_flux=state["f_flux"],
        restart_flag=state["restart_flag"],
        callbacks=inner_callbacks,
        record_events=False,
    )
    updates = result.fields
    if isinstance(updates, dict):
        updates["grid"] = result.grid
        return updates
    return dict(updates) | {"grid": result.grid}


def strict_solve_callbacks() -> dict[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]]:
    callbacks: dict[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]] = {}
    callbacks.update({name: _missing_stage(name) for name in NUMERICAL_STAGE_NAMES})
    callbacks["initialize_moist_old"] = _initialize_moist_old_stage
    callbacks["rk_step_prep"] = _rk_step_prep_stage
    callbacks["rk_phys_bc_dry_1"] = _rk_phys_bc_dry_1_stage
    callbacks["first_rk_step_part1"] = _first_rk_step_part1_stage
    callbacks["first_rk_step_part2"] = _first_rk_step_part2_stage
    callbacks["init_zero_tendency"] = _init_zero_tendency_stage
    callbacks["phy_prep"] = _phy_prep_stage
    callbacks["radiation_driver"] = _radiation_driver_stage
    callbacks["surface_driver"] = _surface_driver_stage
    callbacks["surface_pbl_driver"] = _surface_pbl_driver_stage
    callbacks["pbl_driver"] = _pbl_driver_stage
    callbacks["cumulus_driver"] = _cumulus_driver_stage
    callbacks["shallowcu_driver"] = _shallowcu_driver_stage
    callbacks["fddagd_driver"] = _fddagd_driver_stage
    callbacks["rk_tendency"] = _rk_tendency_stage
    callbacks["relax_bdy_dry"] = _relax_bdy_dry_stage
    if _FUSED_DRY_BOUNDARY_TENDENCY_STAGE:
        callbacks["dry_boundary_tendency"] = _dry_boundary_tendency_stage
    callbacks["rk_addtend_dry"] = _rk_addtend_dry_stage
    callbacks["spec_bdy_dry"] = _spec_bdy_dry_stage
    callbacks["small_step_prep"] = _small_step_prep_stage
    callbacks["calc_p_rho"] = _calc_p_rho_stage
    callbacks["calc_coef_w"] = _calc_coef_w_stage
    callbacks["advance_uv"] = _advance_uv_stage
    callbacks["advance_mu_t"] = _advance_mu_t_stage
    callbacks["advance_w"] = _advance_w_stage
    callbacks["sumflux"] = _sumflux_stage
    callbacks["spec_bdyupdate_ph"] = _spec_bdyupdate_ph_stage
    callbacks["zero_grad_bdy"] = _zero_grad_bdy_stage
    callbacks["calc_mu_uv_1"] = _calc_mu_uv_1_stage
    callbacks["small_step_finish"] = _small_step_finish_stage
    callbacks["rk_update_scalar_pd:moist_old"] = _rk_update_scalar_pd_moist_old_stage
    callbacks["rk_update_scalar_pd:scalar_old"] = _rk_update_scalar_pd_scalar_old_stage
    callbacks["moist_scalar_advance"] = _moist_scalar_advance_stage
    callbacks["scalar_advance"] = _scalar_advance_stage
    callbacks["calc_p_rho_phi"] = _calc_p_rho_phi_stage
    callbacks["calc_p_rho_phi:final"] = _calc_p_rho_phi_stage
    callbacks["advance_ppt"] = _advance_ppt_stage
    callbacks["phy_prep_part2"] = _phy_prep_part2_stage
    callbacks["moist_physics_prep_em"] = _moist_physics_prep_stage
    callbacks["microphysics_driver"] = _microphysics_driver_stage
    callbacks["moist_physics_finish_em"] = _moist_physics_finish_stage
    callbacks["set_physical_bc3d:h_diabatic"] = _set_physical_bc3d_stage("set_physical_bc3d:h_diabatic")
    callbacks["rk_phys_bc_dry_2"] = _rk_phys_bc_dry_2_stage
    callbacks["set_phys_bc_dry_2"] = _set_phys_bc_dry_2_stage
    callbacks["set_physical_bc3d:scalar"] = _set_physical_bc3d_stage("set_physical_bc3d:scalar")
    callbacks["set_physical_bc3d:scalar_old"] = _set_physical_bc3d_stage("set_physical_bc3d:scalar_old")
    callbacks["set_w_surface"] = _set_w_surface_stage
    callbacks["after_all_rk_steps"] = _identity_stage
    for stage_name in ("spec_bdy_final:u", "spec_bdy_final:v", "spec_bdy_final:t", "spec_bdy_final:ph", "spec_bdy_final:mu"):
        callbacks[stage_name] = _spec_bdy_final_stage(stage_name)
    callbacks["spec_bdy_final:moist"] = _spec_bdy_final_moist_stage
    callbacks["spec_bdy_final:moist_qv"] = _spec_bdy_final_moist_qv_stage
    for stage_name in ("spec_bdyupdate:u", "spec_bdyupdate:v", "spec_bdyupdate:t", "spec_bdyupdate:mu_2", "spec_bdyupdate:muts", "spec_bdyupdate:w"):
        callbacks[stage_name] = _spec_bdyupdate_stage(stage_name)
    for name in tuple(callbacks):
        if name.startswith("set_physical_bc3d:"):
            callbacks[name] = _set_physical_bc3d_stage(name)
        elif name.startswith("set_physical_bc2d:"):
            callbacks[name] = _set_physical_bc2d_stage(name)
    callbacks["set_physical_bc3d:moist"] = _set_physical_bc3d_moist_stage
    callbacks["set_physical_bc3d:moist_qv"] = _set_physical_bc3d_moist_qv_stage
    callbacks["set_physical_bc3d:scalar"] = _set_physical_bc3d_scalar_stage
    return callbacks


def _fractional_seaice_inactive(fields: Mapping[str, Any], config_flags: Any) -> bool:
    if int(getattr(config_flags, "fractional_seaice", 0)) != 1:
        return False
    xice = fields.get("xice", fields.get("seaice"))
    if xice is None:
        return False
    xice_host = np.asarray(jax.device_get(jnp.asarray(xice, dtype=jnp.float32)), dtype=np.float32)
    if xice_host.size == 0:
        return False
    threshold = np.float32(getattr(config_flags, "xice_threshold", 0.02))
    return bool(np.nanmax(xice_host) < threshold)


def assemble_runtime(case_dir: str) -> RuntimeAssembly:
    case = load_case(case_dir)
    assert_state_finite(case.state)
    fields = _fields(case)
    fields["_case_config"] = case.config
    grid = _runtime_grid(case)
    fields = _with_runtime_bounds(fields, grid)
    fields["_rdx"] = jnp.asarray(1.0 / float(grid.dx), dtype=jnp.float32)
    fields["_rdy"] = jnp.asarray(1.0 / float(grid.dy), dtype=jnp.float32)
    fields = _with_scalar_float_cache(fields)
    config_flags = _config_flags(case)
    fields["_static_config_flags"] = _static_runtime_config(config_flags)
    fields["_wrf_real_rounding_token"] = jnp.asarray(0, dtype=jnp.uint32)
    fields["_static_fractional_seaice_inactive"] = _fractional_seaice_inactive(fields, config_flags)
    if _STATIC_DRY_BOUNDARY_MASKS:
        fields = _with_static_dry_boundary_masks(fields, config_flags)
    fields = _initialize_landuse_fields(fields, grid, config_flags, case_dir)
    noah_tables = soil_veg_gen_parm(
        str(case.attrs.get("MMINLU", "MODIFIED_IGBP_MODIS_NOAH")),
        "STAS",
        table_dir=case_dir,
    )
    fields = _initialize_noah_lsm_fields(fields, grid, config_flags, case_dir, noah_tables)
    fields["_static_dzs_tuple"] = tuple(float(value) for value in jax.device_get(jnp.asarray(fields["dzs"], dtype=jnp.float32)).reshape(-1))
    znu_host = np.asarray(jax.device_get(jnp.asarray(fields["znu"], dtype=jnp.float32))).reshape(-1)
    kts = int(fields.get("kps", fields["kds"]))
    kte = min(int(fields.get("kpe", fields["kde"])), int(fields["kde"]) - 1)
    kms = int(fields["kms"])
    kpblmax = None
    for k in range(kts, kte + 1):
        if znu_host[k - kms] > np.float32(0.6):
            kpblmax = k + 1
    if kpblmax is not None:
        fields["_static_gwdo_kpblmax"] = int(kpblmax)
    fields = _with_static_gwdo_active_columns(fields)
    fields = _with_static_noah_active_columns(fields, config_flags, land_columns=_STATIC_NOAH_ACTIVE_COLUMNS)
    fields = _with_static_seaice_active_columns(fields, config_flags)
    fields = _initialize_physics_startup_fields(fields, grid, config_flags)
    fields = _start_em_recompute_pressure_from_alt(fields, grid, config_flags)
    fields = _start_em_initialize_w(fields, grid, config_flags)
    if _surface_defaults_complete(fields):
        fields["_surface_defaults_complete"] = True
    fields["noah_parameters"] = noah_tables.to_noah_lsm_parameters()
    if int(getattr(config_flags, "mp_physics", 0)) == 8:
        # Keep lookup data as device-resident parameters, outside model state
        # and compiled program constants. CPU and GPU consume the same values.
        fields["thompson_tables"] = _device_thompson_tables(str(case_dir))
    return RuntimeAssembly(
        case=case,
        grid=grid,
        config_flags=config_flags,
        fields=fields,
        callbacks=strict_solve_callbacks(),
    )


def run_one_strict_step(case_dir: str) -> SolveEmResult:
    assembly = assemble_runtime(case_dir)
    return solve_em(
        grid=assembly.grid,
        config_flags=assembly.config_flags,
        fields=assembly.fields,
        moist=assembly.fields["moist"],
        scalar=assembly.fields.get("scalar"),
        num_moist=_num_moist_from_fields(assembly.fields),
        num_scalar=_num_scalar_from_fields(assembly.fields),
        callbacks=assembly.callbacks,
    )


__all__ = [
    "MissingKernelImplementationError",
    "RuntimeAssembly",
    "assemble_runtime",
    "run_one_strict_step",
    "strict_solve_callbacks",
]
