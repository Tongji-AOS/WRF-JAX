"""Translation of WRF ``dyn_em/module_first_rk_step_part2`` orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

SCALESASSCHEME = 4
TIEDTKESCHEME = 6
CAMZMSCHEME = 7
MSKFSCHEME = 11
KSASSCHEME = 14
NTIEDTKESCHEME = 16
SASSCHEME = 94
NSASSCHEME = 96

_CU_HALO_SCHEMES = {
    SASSCHEME,
    TIEDTKESCHEME,
    NTIEDTKESCHEME,
    MSKFSCHEME,
    CAMZMSCHEME,
    SCALESASSCHEME,
    NSASSCHEME,
    KSASSCHEME,
}


class _EventLog:
    __slots__ = ("_enabled", "_items")

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._items: list[str] = []

    def append(self, item: str) -> None:
        if self._enabled:
            self._items.append(item)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def as_tuple(self) -> tuple[str, ...]:
        return tuple(self._items)


@dataclass(frozen=True)
class FirstRkStepPart2Result:
    """Explicit state and call trace for WRF ``first_rk_step_part2``."""

    grid: Any
    fields: Mapping[str, Any]
    events: tuple[str, ...]
    rk_step: int


def _read(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _read_int(source: Any, name: str, default: int = 0) -> int:
    return int(_read(source, name, default))


def _read_float(source: Any, name: str, default: float = 0.0) -> float:
    return float(_read(source, name, default))


def _num_tiles(grid: Any) -> int:
    return max(1, _read_int(grid, "num_tiles", 1))


def _run_tiles(
    tile_count: int,
    debug_event: str | None,
    call_name: str,
    events: _EventLog,
    call_stage: Callable[[str], None],
) -> None:
    for _ij in range(1, tile_count + 1):
        if debug_event is not None and events.enabled:
            events.append(debug_event)
        call_stage(call_name)


def first_rk_step_part2(
    *,
    grid: Any,
    config_flags: Any,
    fields: Mapping[str, Any] | None = None,
    moist: Any = None,
    moist_old: Any = None,
    moist_tend: Any = None,
    chem: Any = None,
    chem_tend: Any = None,
    tracer: Any = None,
    tracer_tend: Any = None,
    scalar: Any = None,
    scalar_tend: Any = None,
    fdda3d: Any = None,
    fdda2d: Any = None,
    ru_tendf: Any = None,
    rv_tendf: Any = None,
    rw_tendf: Any = None,
    t_tendf: Any = None,
    ph_tendf: Any = None,
    mu_tendf: Any = None,
    tke_tend: Any = None,
    adapt_step_flag: bool = False,
    curr_secs: float = 0.0,
    psim: Any = None,
    psih: Any = None,
    gz1oz0: Any = None,
    chklowq: Any = None,
    cu_act_flag: Any = None,
    hol: Any = None,
    th_phy: Any = None,
    pi_phy: Any = None,
    p_phy: Any = None,
    t_phy: Any = None,
    dz8w: Any = None,
    p8w: Any = None,
    t8w: Any = None,
    nba_mij: Any = None,
    n_nba_mij: int = 0,
    nba_rij: Any = None,
    n_nba_rij: int = 0,
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
    ips: int = 1,
    ipe: int = 1,
    jps: int = 1,
    jpe: int = 1,
    kps: int = 1,
    kpe: int = 1,
    imsx: int = 1,
    imex: int = 1,
    jmsx: int = 1,
    jmex: int = 1,
    kmsx: int = 1,
    kmex: int = 1,
    ipsx: int = 1,
    ipex: int = 1,
    jpsx: int = 1,
    jpex: int = 1,
    kpsx: int = 1,
    kpex: int = 1,
    imsy: int = 1,
    imey: int = 1,
    jmsy: int = 1,
    jmey: int = 1,
    kmsy: int = 1,
    kmey: int = 1,
    ipsy: int = 1,
    ipey: int = 1,
    jpsy: int = 1,
    jpey: int = 1,
    kpsy: int = 1,
    kpey: int = 1,
    k_start: int = 1,
    k_end: int = 1,
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]] | None = None,
    record_events: bool = True,
) -> FirstRkStepPart2Result:
    """Run WRF ``first_rk_step_part2`` as explicit step orchestration.

    This routine keeps the Fortran branch structure and call order explicit.
    Heavy child kernels are reached through named callbacks so each translated
    kernel remains independently testable and can be wired into this wrapper
    without hidden global state.
    """

    del (
        adapt_step_flag,
        curr_secs,
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
        k_start,
        k_end,
        n_nba_mij,
        n_nba_rij,
    )

    state: dict[str, Any] = dict(fields or {})
    for name, value in {
        "moist": moist,
        "moist_old": moist_old,
        "moist_tend": moist_tend,
        "chem": chem,
        "chem_tend": chem_tend,
        "tracer": tracer,
        "tracer_tend": tracer_tend,
        "scalar": scalar,
        "scalar_tend": scalar_tend,
        "fdda3d": fdda3d,
        "fdda2d": fdda2d,
        "ru_tendf": ru_tendf,
        "rv_tendf": rv_tendf,
        "rw_tendf": rw_tendf,
        "t_tendf": t_tendf,
        "ph_tendf": ph_tendf,
        "mu_tendf": mu_tendf,
        "tke_tend": tke_tend,
        "psim": psim,
        "psih": psih,
        "gz1oz0": gz1oz0,
        "chklowq": chklowq,
        "cu_act_flag": cu_act_flag,
        "hol": hol,
        "th_phy": th_phy,
        "pi_phy": pi_phy,
        "p_phy": p_phy,
        "t_phy": t_phy,
        "dz8w": dz8w,
        "p8w": p8w,
        "t8w": t8w,
        "nba_mij": nba_mij,
        "nba_rij": nba_rij,
    }.items():
        if value is not None:
            state[name] = value

    grid_out = grid
    state["grid"] = grid_out
    state["config_flags"] = config_flags
    events = _EventLog(record_events)
    rk_step = 1

    def call_stage(name: str) -> None:
        nonlocal grid_out
        if events.enabled:
            events.append(f"call:{name}")
        if callbacks is None or name not in callbacks:
            return
        updates = callbacks[name](state)
        if not updates:
            return
        state.update(updates)
        if "grid" in updates:
            grid_out = updates["grid"]

    grid_id_is_root = _read_int(grid_out, "id", 1) == 1
    if _read_int(grid_out, "skebs_on", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:skebs_t")
        call_stage("rand_pert_update:skebs_u")
        call_stage("rand_pert_update:skebs_v")

    if _read_int(grid_out, "sppt_on", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:sppt")

    if _read_int(grid_out, "rand_perturb_on", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:rand_perturb")

    stepsp = _read_int(grid_out, "stepsp", 0)
    stepsp_test = stepsp > 0 and _read_int(grid_out, "itimestep", 0) % stepsp == 0
    if _read_int(config_flags, "multi_perturb", 0) == 1 and grid_id_is_root and (
        stepsp_test or _read_int(grid_out, "itimestep", 0) == 1
    ):
        for _n in range(2, _read_int(config_flags, "num_pert_3d", 0) + 1):
            call_stage("contiguize_2d:spforcs3d:load")
            call_stage("contiguize_2d:spforcc3d:load")
            call_stage("contiguize_2d:sp_amp3d:load")
            call_stage("rand_pert_update:multi_perturb")
            call_stage("contiguize_2d:spforcs3d:store")
            call_stage("contiguize_2d:spforcc3d:store")
            call_stage("contiguize_2d:sp_amp3d:store")

    if _read_int(grid_out, "spp_conv", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:spp_conv")
    if _read_int(grid_out, "spp_pbl", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:spp_pbl")
    if _read_int(grid_out, "spp_lsm", 0) == 1 and grid_id_is_root:
        call_stage("rand_pert_update:spp_lsm")

    tile_count = _num_tiles(grid_out)
    _run_tiles(tile_count, "debug:call calculate_phy_tend", "calculate_phy_tend", events, call_stage)

    diff_opt = _read_int(config_flags, "diff_opt", 0)
    km_opt = _read_int(config_flags, "km_opt", 0)
    if diff_opt in {1, 2}:
        _run_tiles(tile_count, "debug:call compute_diff_metrics", "compute_diff_metrics", events, call_stage)
        call_stage("halo_em_tke_c_sub")
        call_stage("period_bdy_em_a1_sub")

        for _ij in range(1, tile_count + 1):
            if events.enabled:
                events.append("debug:call bc for diffusion_metrics")
            for name in (
                "set_physical_bc3d:rdzw",
                "set_physical_bc3d:rdz",
                "set_physical_bc3d:z",
                "set_physical_bc3d:zx",
                "set_physical_bc3d:zy",
                "set_physical_bc2d:ustm",
                "set_physical_bc2d:ust",
            ):
                call_stage(name)

        _run_tiles(tile_count, "debug:call cal_deform_and_div", "cal_deform_and_div", events, call_stage)
        call_stage("halo_em_helicity_sub")

        if _read_int(config_flags, "nwp_diagnostics", 0) == 1 or (
            _read_int(config_flags, "afwa_diag_opt", 0) == 1 and _read_int(config_flags, "afwa_severe_opt", 0) == 1
        ):
            _run_tiles(tile_count, "debug:call cal_helicity", "cal_helicity", events, call_stage)

        call_stage("halo_em_tke_d_sub")
        for _ij in range(1, tile_count + 1):
            if diff_opt == 2 and km_opt == 5:
                call_stage("nonlocal_flux")
                call_stage("free_atmos_length")
                call_stage("meso_length_scale")
            if events.enabled:
                events.append("debug:call calculate_km_kh")
            call_stage("calculate_km_kh")

        call_stage("halo_em_tke_e_sub")

    call_stage("period_bdy_em_phy_bc_sub")

    if _read_int(config_flags, "grid_fdda", 0) == 1:
        call_stage("period_bdy_em_fdda_bc_sub")

    call_stage("period_bdy_em_chem_sub")
    _run_tiles(tile_count, "debug:call phy_bc", "phy_bc", events, call_stage)

    if _read_int(config_flags, "sfs_opt", 0) > 0 and diff_opt == 2:
        call_stage("sfs_driver")

    if _read_int(config_flags, "bl_pbl_physics", 0) >= 1:
        call_stage("halo_em_phys_pbl_sub")
    if _read_int(config_flags, "shcu_physics", 0) > 1:
        call_stage("halo_em_phys_shcu_sub")
    if _read_int(config_flags, "cu_physics", 0) in _CU_HALO_SCHEMES:
        call_stage("halo_em_phys_cu_sub")
    if _read_int(config_flags, "grid_fdda", 0) >= 1:
        call_stage("halo_em_fdda_sub")
    if diff_opt >= 1:
        call_stage("halo_em_phys_diffusion_sub")

    h_sca_adv_order = _read_int(config_flags, "h_sca_adv_order", 0)
    if h_sca_adv_order <= 4:
        call_stage("halo_em_tke_3_sub")
    elif h_sca_adv_order <= 6:
        call_stage("halo_em_tke_5_sub")
    else:
        raise ValueError(f"first_rk_step_part2 invalid h_sca_adv_order = {h_sca_adv_order}")

    _run_tiles(tile_count, "debug:call update_phy_ten", "update_phy_ten", events, call_stage)

    if _read_int(grid_out, "skebs_on", 0) == 1:
        _run_tiles(tile_count, "debug:call update_stoch_ten", "update_stoch_ten", events, call_stage)

    if _read_int(grid_out, "sppt_on", 0) == 1:
        _run_tiles(tile_count, None, "perturb_physics_tend", events, call_stage)

    if diff_opt == 2 and km_opt in {2, 5}:
        _run_tiles(tile_count, None, "tke_rhs", events, call_stage)

    if _read_int(grid_out, "obs_nudge_opt", 0) == 1 and _read_float(grid_out, "xtime", 0.0) <= _read_float(grid_out, "fdda_end", 0.0):
        call_stage("halo_obs_nudge_sub")
        _run_tiles(tile_count, None, "fddaobs_driver", events, call_stage)

    if _read_int(config_flags, "use_theta_m", 0) == 1:
        _run_tiles(tile_count, None, "conv_t_tendf_to_moist", events, call_stage)

    if diff_opt == 2:
        if _read_int(config_flags, "bl_pbl_physics", 0) == 0:
            if km_opt == 5:
                _run_tiles(tile_count, None, "vertical_diffusion_implicit", events, call_stage)
            else:
                _run_tiles(tile_count, "debug:call vertical_diffusion_2", "vertical_diffusion_2", events, call_stage)
        _run_tiles(tile_count, "debug:call horizontal_diffusion_2", "horizontal_diffusion_2", events, call_stage)

    return FirstRkStepPart2Result(
        grid=grid_out,
        fields=state,
        events=events.as_tuple(),
        rk_step=rk_step,
    )


__all__ = [
    "CAMZMSCHEME",
    "FirstRkStepPart2Result",
    "KSASSCHEME",
    "MSKFSCHEME",
    "NSASSCHEME",
    "NTIEDTKESCHEME",
    "SASSCHEME",
    "SCALESASSCHEME",
    "TIEDTKESCHEME",
    "first_rk_step_part2",
]
