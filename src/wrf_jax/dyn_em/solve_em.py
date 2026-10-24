"""Control-flow translation of WRF ``dyn_em/solve_em``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from wrf_jax.compat.get_ijk_from_grid1 import get_ijk_from_grid1
from wrf_jax.compat.get_ijk_from_subgrid import get_ijk_from_subgrid
from wrf_jax.compat.module_tiles import set_tiles_once

DFI_BCK = 2
MYNNPBLSCHEME = 5
ORIGINAL = 0
PARAM_FIRST_SCALAR = 2
P_QV = 2
PROJ_CASSINI = 6
WENO_SCALAR = 3
ZONE_SOLVE_EM = 1


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
class SolveEmResult:
    """Explicit result for WRF ``solve_em`` time-step orchestration."""

    grid: Any
    config_flags: Any
    fields: Mapping[str, Any]
    ijk: tuple[int, ...]
    subgrid_ijk: tuple[int, ...]
    rk_order: int
    num_sound_steps: int
    dts: float
    adapt_step_flag: bool
    restart_flag: bool
    diag_flag: bool
    curr_secs: float
    curr_secs2: float
    max_vert_cfl_tmp: tuple[float, ...]
    max_horiz_cfl_tmp: tuple[float, ...]
    events: tuple[str, ...]


def _read(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _write(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[name] = value
        return
    setattr(obj, name, value)


def _read_bool(obj: Any, name: str, default: bool = False) -> bool:
    return bool(_read(obj, name, default))


def _read_int(obj: Any, name: str, default: int = 0) -> int:
    return int(_read(obj, name, default))


def _read_float(obj: Any, name: str, default: float = 0.0) -> float:
    return float(_read(obj, name, default))


def _as_tuple(result: Any, expected: int, name: str) -> tuple[int, ...]:
    values = result.as_tuple() if hasattr(result, "as_tuple") else result
    output = tuple(int(value) for value in values)
    if len(output) < expected:
        raise ValueError(f"{name} must return at least {expected} bounds")
    return output


def _default_current_time(grid: Any) -> float:
    return _read_float(grid, "current_time", 0.0)


def _default_sim_start_time(grid: Any) -> float:
    return _read_float(grid, "sim_start_time", 0.0)


def _default_start_time(grid: Any) -> float:
    return _read_float(grid, "start_time", 0.0)


def _default_real_time(interval: Any) -> float:
    return float(interval)


def _default_is_alarm_tstep(_domain_clock: Any, alarm: Any) -> bool:
    return bool(alarm)


def _noop_debug(_level: int, _message: str) -> None:
    return None


def _tile_count(grid: Any) -> int:
    return max(1, _read_int(grid, "num_tiles", 1))


def _tile_bounds(grid: Any, tile: int, its: int, ite: int, jts: int, jte: int) -> tuple[int, int, int, int]:
    index = int(tile) - 1
    i_start = _read(grid, "i_start", None)
    i_end = _read(grid, "i_end", None)
    j_start = _read(grid, "j_start", None)
    j_end = _read(grid, "j_end", None)
    if i_start is None or i_end is None or j_start is None or j_end is None:
        return int(its), int(ite), int(jts), int(jte)
    return int(i_start[index]), int(i_end[index]), int(j_start[index]), int(j_end[index])


def _final_calc_p_rho_phi_bounds(
    grid: Any,
    config_flags: Any,
    tile_its: int,
    tile_ite: int,
    tile_jts: int,
    tile_jte: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
) -> tuple[int, int, int, int]:
    sz = _read_int(grid, "spec_zone", _read_int(config_flags, "spec_zone", 0)) if (
        _read_bool(config_flags, "specified") or _read_bool(config_flags, "nested")
    ) else 0
    if _read_bool(config_flags, "periodic_x"):
        its = max(int(tile_its), int(ids))
        ite = min(int(tile_ite), int(ide) - 1)
    else:
        its = max(int(tile_its), int(ids) + sz)
        ite = min(int(tile_ite), int(ide) - 1 - sz)
    jts = max(int(tile_jts), int(jds) + sz)
    jte = min(int(tile_jte), int(jde) - 1 - sz)
    return its, ite, jts, jte


def _debug(events: _EventLog, wrf_debug_fn: Callable[[int, str], Any], stage: str, level: int = 200) -> None:
    if events.enabled:
        events.append(f"debug:{stage}")
    if wrf_debug_fn is not _noop_debug:
        wrf_debug_fn(level, stage)


def _stage(
    events: _EventLog,
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]],
    name: str,
    state: dict[str, Any],
    event_name: str | None = None,
    **context: Any,
) -> None:
    if events.enabled:
        events.append(f"call:{event_name or name}")
    if context:
        state.update(context)
    callback = callbacks.get(name)
    if callback is None:
        return
    updates = callback(state)
    if updates:
        state.update(updates)


def _call_domain_time(
    events: _EventLog,
    name: str,
    fn: Callable[[Any], Any],
    grid: Any,
) -> Any:
    if events.enabled:
        events.append(f"call:{name}")
    return fn(grid)


def _call_real_time(events: _EventLog, real_time_fn: Callable[[Any], float], interval: Any) -> float:
    if events.enabled:
        events.append("call:real_time")
    return float(real_time_fn(interval))


def _call_alarm(
    events: _EventLog,
    label: str,
    is_alarm_tstep_fn: Callable[[Any, Any], bool],
    grid: Any,
    alarm: Any,
) -> bool:
    if events.enabled:
        events.append(f"call:is_alarm_tstep:{label}")
    return bool(is_alarm_tstep_fn(_read(grid, "domain_clock", None), alarm))


def _compute_num_sound_steps(grid: Any, config_flags: Any) -> int:
    time_step_sound = _read_int(grid, "time_step_sound", 0)
    if time_step_sound != 0:
        return time_step_sound

    dt = _read_float(grid, "dt", 0.0)
    spacing = min(_read_float(grid, "dx", 1.0), _read_float(grid, "dy", 1.0))
    if spacing == 0.0:
        raise ValueError("solve_em requires non-zero grid spacing")

    if _read_bool(config_flags, "use_adaptive_time_step") and _read_int(config_flags, "map_proj", 0) == PROJ_CASSINI:
        fft_filter_lat = _read_float(config_flags, "fft_filter_lat", 0.0)
        max_msft = min(
            max(_read_float(grid, "max_msftx", 1.0), _read_float(grid, "max_msfty", 1.0)),
            1.0 / np.cos(np.deg2rad(fft_filter_lat)),
        )
        return max(2 * (int(300.0 * dt / (spacing / max_msft) - 0.01) + 1), 4)
    if _read_bool(config_flags, "use_adaptive_time_step"):
        max_msft = max(_read_float(grid, "max_msftx", 1.0), _read_float(grid, "max_msfty", 1.0))
        return max(2 * (int(300.0 * dt / (spacing / max_msft) - 0.01) + 1), 4)
    return max(2 * (int(300.0 * dt / spacing - 0.01) + 1), 4)


def _rk_step_sizes(rk_order: int, rk_step: int, dt: float, dts: float, num_sound_steps: int) -> tuple[float, float, int]:
    if rk_order == 1:
        raise ValueError("solve_em does not support removed leapfrog rk_order=1")
    if rk_order == 2:
        if rk_step == 1:
            return 0.5 * dt, dts, max(1, num_sound_steps // 2)
        return dt, dts, num_sound_steps
    if rk_order == 3:
        if rk_step == 1:
            dt_rk = dt / 3.0
            return dt_rk, dt_rk, 1
        if rk_step == 2:
            return 0.5 * dt, dts, max(1, num_sound_steps // 2)
        return dt, dts, num_sound_steps
    raise ValueError(f"solve_em unknown rk_order={rk_order}")


def solve_em(
    *,
    grid: Any,
    config_flags: Any,
    fields: Mapping[str, Any] | None = None,
    moist: Any = None,
    chem: Any = None,
    tracer: Any = None,
    scalar: Any = None,
    num_moist: int = 0,
    num_chem: int = 0,
    num_tracer: int = 0,
    num_scalar: int = 0,
    get_ijk_from_grid_fn: Callable[..., Any] = get_ijk_from_grid1,
    get_ijk_from_subgrid_fn: Callable[..., Any] = get_ijk_from_subgrid,
    set_tiles_fn: Callable[..., Any] = set_tiles_once,
    domain_get_current_time_fn: Callable[[Any], Any] = _default_current_time,
    domain_get_sim_start_time_fn: Callable[[Any], Any] = _default_sim_start_time,
    domain_get_start_time_fn: Callable[[Any], Any] = _default_start_time,
    real_time_fn: Callable[[Any], float] = _default_real_time,
    is_alarm_tstep_fn: Callable[[Any, Any], bool] = _default_is_alarm_tstep,
    adapt_timestep_fn: Callable[[Any, Any], Any] | None = None,
    wrf_debug_fn: Callable[[int, str], Any] = _noop_debug,
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]] | None = None,
    record_events: bool = True,
) -> SolveEmResult:
    """Run WRF ``solve_em`` as explicit time-step orchestration.

    The translated numerical child kernels are reached through named callbacks.
    This preserves the Fortran order of the top-level driver while keeping the
    differentiable JAX kernels separate from I/O, communication, and driver
    mutation.
    """

    events = _EventLog(record_events)
    stage_callbacks = callbacks or {}
    state: dict[str, Any] = dict(fields or {})
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

    if events.enabled:
        events.append("call:get_ijk_from_grid")
    ijk = _as_tuple(get_ijk_from_grid_fn(grid), 42, "get_ijk_from_grid")
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
    ) = ijk[:42]
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

    if events.enabled:
        events.append("call:get_ijk_from_subgrid")
    subgrid_ijk = _as_tuple(get_ijk_from_subgrid_fn(grid), 18, "get_ijk_from_subgrid")
    k_start = kps
    k_end = kpe
    its, ite, jts, jte = ips, ipe, jps, jpe

    num_3d_m = int(num_moist)
    num_3d_c = int(num_chem)
    num_3d_s = int(num_scalar)
    if _read_int(grid, "dfi_stage", 0) == DFI_BCK:
        num_3d_m = P_QV
        num_3d_s = PARAM_FIRST_SCALAR - 1
    state.update({"num_moist": num_3d_m, "num_chem": num_3d_c, "num_tracer": num_tracer, "num_scalar": num_3d_s})
    f_flux = _read_int(config_flags, "do_avgflx_cugd", 0) == 1

    if events.enabled:
        events.append("call:set_tiles")
    set_tiles_fn(ZONE_SOLVE_EM, grid, ids, ide, jds, jde, ips, ipe, jps, jpe, config=config_flags)

    current_time = _call_domain_time(events, "domain_get_current_time", domain_get_current_time_fn, grid)
    sim_start_time = _call_domain_time(events, "domain_get_sim_start_time", domain_get_sim_start_time_fn, grid)
    curr_secs = _call_real_time(events, real_time_fn, current_time - sim_start_time)
    current_time2 = _call_domain_time(events, "domain_get_current_time", domain_get_current_time_fn, grid)
    start_time = _call_domain_time(events, "domain_get_start_time", domain_get_start_time_fn, grid)
    curr_secs2 = _call_real_time(events, real_time_fn, current_time2 - start_time)

    old_dt = _read_float(grid, "dt", 0.0)
    del old_dt
    nested = _read_bool(grid, "nested", _read_bool(config_flags, "nested", False))
    if _read_bool(config_flags, "use_adaptive_time_step") and ((not nested) or (nested and abs(_read_float(grid, "dtbc", 0.0)) < 0.0001)):
        if events.enabled:
            events.append("call:adapt_timestep")
        if adapt_timestep_fn is not None:
            adapt_timestep_fn(grid, config_flags)
        adapt_step_flag = True
    else:
        adapt_step_flag = False

    restart_flag = _call_alarm(events, "restart", is_alarm_tstep_fn, grid, _read(grid, "restart_alarm", False))
    diag_flag = _call_alarm(events, "history", is_alarm_tstep_fn, grid, _read(grid, "history_alarm", False))
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

    _write(grid, "itimestep", _read_int(grid, "itimestep", 0) + 1)
    _write(grid, "dtbc", _read_float(grid, "dtbc", 0.0) + _read_float(grid, "dt", 0.0))

    rk_order = _read_int(config_flags, "rk_ord", 3)
    num_sound_steps = _compute_num_sound_steps(grid, config_flags)
    dts = _read_float(grid, "dt", 0.0) / float(num_sound_steps)
    tile_count = _tile_count(grid)
    tile_range = range(1, tile_count + 1)

    max_vert_cfl_tmp = [0.0 for _ in range(tile_count)]
    max_horiz_cfl_tmp = [0.0 for _ in range(tile_count)]
    if _read_bool(config_flags, "use_adaptive_time_step"):
        _write(grid, "max_cfl_val", 0.0)
        _write(grid, "max_horiz_cfl", 0.0)
        _write(grid, "max_vert_cfl", 0.0)

    for ij in tile_range:
        tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
        if _read_bool(config_flags, "specified") and _read_bool(config_flags, "constant_bc"):
            _stage(events, stage_callbacks, "zero_bdytend", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
        _stage(events, stage_callbacks, "initialize_moist_old", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if num_3d_m >= PARAM_FIRST_SCALAR:
        _stage(events, stage_callbacks, "halo_em_moist_old_e_7_sub", state)
        _stage(events, stage_callbacks, "period_bdy_em_moist_old_sub", state)
        for ij in tile_range:
            _stage(events, stage_callbacks, "set_physical_bc3d:moist_old", state, ij=ij)

    for rk_step in range(1, rk_order + 1):
        dt_rk, dts_rk, number_of_small_timesteps = _rk_step_sizes(
            rk_order,
            rk_step,
            _read_float(grid, "dt", 0.0),
            dts,
            num_sound_steps,
        )
        state.update({"rk_step": rk_step, "dt_rk": dt_rk, "dts_rk": dts_rk, "number_of_small_timesteps": number_of_small_timesteps})

        _debug(events, wrf_debug_fn, "call rk_step_prep")
        for ij in tile_range:
            _stage(events, stage_callbacks, "rk_step_prep", state, ij=ij)

        _stage(events, stage_callbacks, "halo_em_a_sub", state)
        _stage(events, stage_callbacks, "period_bdy_em_a_sub", state)

        for ij in tile_range:
            _debug(events, wrf_debug_fn, "call rk_phys_bc_dry_1")
            _stage(events, stage_callbacks, "rk_phys_bc_dry_1", state, ij=ij)
            _stage(events, stage_callbacks, "set_physical_bc3d:rho", state, ij=ij)
            _stage(events, stage_callbacks, "set_physical_bc3d:al", state, ij=ij)
            _stage(events, stage_callbacks, "set_physical_bc3d:ph_2", state, ij=ij)

        if rk_step == 1:
            _stage(events, stage_callbacks, "first_rk_step_part1", state, f_flux=f_flux, restart_flag=restart_flag, curr_secs=curr_secs)
            if _read_int(config_flags, "bl_pbl_physics", 0) == MYNNPBLSCHEME:
                _stage(events, stage_callbacks, "halo_em_scalar_e_5_sub", state)
            _stage(events, stage_callbacks, "first_rk_step_part2", state, adapt_step_flag=adapt_step_flag, curr_secs=curr_secs)

        for ij in tile_range:
            _debug(events, wrf_debug_fn, "call rk_tendency")
            _stage(events, stage_callbacks, "rk_tendency", state, ij=ij)
            if _read_bool(config_flags, "use_adaptive_time_step"):
                max_vert_cfl_tmp[ij - 1] = max(max_vert_cfl_tmp[ij - 1], float(state.get("max_vert_cfl", 0.0)))
                max_horiz_cfl_tmp[ij - 1] = max(max_horiz_cfl_tmp[ij - 1], float(state.get("max_horiz_cfl", 0.0)))

        if _read_bool(config_flags, "use_adaptive_time_step"):
            max_horiz = max(max_horiz_cfl_tmp) if max_horiz_cfl_tmp else 0.0
            max_vert = max(max_vert_cfl_tmp) if max_vert_cfl_tmp else 0.0
            _write(grid, "max_horiz_cfl", max_horiz)
            _write(grid, "max_vert_cfl", max_vert)
            _write(grid, "max_cfl_val", max(max_horiz, max_vert, _read_float(grid, "max_cfl_val", 0.0)))

        for ij in tile_range:
            if (_read_bool(config_flags, "specified") or _read_bool(config_flags, "nested")) and rk_step == 1:
                _stage(events, stage_callbacks, "relax_bdy_dry", state, ij=ij)
            _stage(events, stage_callbacks, "rk_addtend_dry", state, ij=ij)
            if _read_bool(config_flags, "specified") or _read_bool(config_flags, "nested"):
                _stage(events, stage_callbacks, "spec_bdy_dry", state, ij=ij)

        for ij in tile_range:
            _debug(events, wrf_debug_fn, "call small_step_prep")
            _stage(events, stage_callbacks, "small_step_prep", state, ij=ij)
            _stage(events, stage_callbacks, "calc_p_rho", state, ij=ij, iteration=0)
            if _read_bool(config_flags, "non_hydrostatic", True):
                _stage(events, stage_callbacks, "calc_coef_w", state, ij=ij)

        _stage(events, stage_callbacks, "halo_em_b_sub", state)
        _stage(events, stage_callbacks, "period_bdy_em_b_sub", state)

        for ij in tile_range:
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
                _stage(events, stage_callbacks, stage_name, state, ij=ij)

        for iteration in range(1, number_of_small_timesteps + 1):
            state["iteration"] = iteration
            _stage(events, stage_callbacks, "period_bdy_em_b_sub", state)
            for ij in tile_range:
                _stage(events, stage_callbacks, "advance_uv", state, ij=ij)
            for ij in tile_range:
                if _read_bool(config_flags, "specified") or _read_bool(config_flags, "nested"):
                    _stage(events, stage_callbacks, "spec_bdyupdate:u", state, ij=ij)
                    _stage(events, stage_callbacks, "spec_bdyupdate:v", state, ij=ij)
            _stage(events, stage_callbacks, "halo_em_c_sub", state)
            for ij in tile_range:
                _stage(events, stage_callbacks, "advance_mu_t", state, ij=ij)
            for ij in tile_range:
                if _read_bool(config_flags, "specified") or _read_bool(config_flags, "nested"):
                    _stage(events, stage_callbacks, "spec_bdyupdate:t", state, ij=ij)
                    _stage(events, stage_callbacks, "spec_bdyupdate:mu_2", state, ij=ij)
                    _stage(events, stage_callbacks, "spec_bdyupdate:muts", state, ij=ij)
                if _read_bool(config_flags, "non_hydrostatic", True):
                    _stage(events, stage_callbacks, "advance_w", state, ij=ij)
            for ij in tile_range:
                _stage(events, stage_callbacks, "sumflux", state, ij=ij)
                if (_read_bool(config_flags, "specified") or _read_bool(config_flags, "nested")) and _read_bool(config_flags, "non_hydrostatic", True):
                    _stage(events, stage_callbacks, "spec_bdyupdate_ph", state, ij=ij)
                    if _read_bool(config_flags, "specified"):
                        _stage(events, stage_callbacks, "zero_grad_bdy", state, ij=ij)
                    else:
                        _stage(events, stage_callbacks, "spec_bdyupdate:w", state, ij=ij)
                _stage(events, stage_callbacks, "calc_p_rho", state, ij=ij, iteration=iteration)
            _stage(events, stage_callbacks, "halo_em_c2_sub", state)
            _stage(events, stage_callbacks, "period_bdy_em_b3_sub", state)
            for ij in tile_range:
                for stage_name in (
                    "set_physical_bc3d:ph_2:small",
                    "set_physical_bc3d:al:small",
                    "set_physical_bc3d:p:small",
                    "set_physical_bc2d:muts:small",
                    "set_physical_bc2d:mu_2:small",
                    "set_physical_bc2d:mudf:small",
                ):
                    _stage(events, stage_callbacks, stage_name, state, ij=ij)

        for ij in tile_range:
            _debug(events, wrf_debug_fn, "call rk_small_finish")
            _stage(events, stage_callbacks, "calc_mu_uv_1", state, ij=ij)
            _stage(events, stage_callbacks, "small_step_finish", state, ij=ij)
            if rk_step == rk_order:
                for stage_name in (
                    "set_physical_bc3d:ru_m",
                    "set_physical_bc3d:rv_m",
                    "set_physical_bc3d:ww_m",
                    "set_physical_bc2d:mut",
                    "set_physical_bc2d:muts",
                ):
                    _stage(events, stage_callbacks, stage_name, state, ij=ij)

        if (
            num_3d_m >= PARAM_FIRST_SCALAR
            and rk_step == rk_order
            and _read_int(config_flags, "moist_adv_opt", ORIGINAL) not in {ORIGINAL, WENO_SCALAR}
        ):
            for ij in tile_range:
                _debug(events, wrf_debug_fn, "call rk_update_scalar_pd")
                _stage(events, stage_callbacks, "rk_update_scalar_pd:moist_old", state, ij=ij)
            _stage(
                events,
                stage_callbacks,
                "halo_em_moist_old_e_7_sub" if _read_int(config_flags, "h_sca_adv_order", 5) > 4 else "halo_em_moist_old_e_5_sub",
                state,
            )
            _stage(events, stage_callbacks, "period_bdy_em_moist_old_sub", state)
            for ij in tile_range:
                _stage(events, stage_callbacks, "set_physical_bc3d:moist_old", state, ij=ij)

        if (
            num_3d_s >= PARAM_FIRST_SCALAR
            and rk_step == rk_order
            and _read_int(config_flags, "scalar_adv_opt", ORIGINAL) not in {ORIGINAL, WENO_SCALAR}
        ):
            for ij in tile_range:
                _debug(events, wrf_debug_fn, "call rk_update_scalar_pd for scalar")
                _stage(events, stage_callbacks, "rk_update_scalar_pd:scalar_old", state, ij=ij)
            _stage(
                events,
                stage_callbacks,
                "halo_em_scalar_old_e_7_sub" if _read_int(config_flags, "h_sca_adv_order", 5) > 4 else "halo_em_scalar_old_e_5_sub",
                state,
            )
            _stage(events, stage_callbacks, "period_bdy_em_scalar_old_sub", state)
            for ij in tile_range:
                _stage(events, stage_callbacks, "set_physical_bc3d:scalar_old", state, ij=ij)

        if num_3d_m >= PARAM_FIRST_SCALAR:
            for ij in tile_range:
                _debug(events, wrf_debug_fn, "call rk_scalar_tend")
                _stage(events, stage_callbacks, "moist_scalar_advance", state, ij=ij)

        if num_3d_s >= PARAM_FIRST_SCALAR:
            for ij in tile_range:
                _debug(events, wrf_debug_fn, "call rk_scalar_tend for scalar")
                _stage(events, stage_callbacks, "scalar_advance", state, ij=ij)

        for ij in tile_range:
            _stage(events, stage_callbacks, "calc_p_rho_phi", state, ij=ij)

        if rk_step < rk_order:
            _stage(events, stage_callbacks, "halo_em_d2_5_sub" if _read_int(config_flags, "h_sca_adv_order", 5) > 4 else "halo_em_d2_3_sub", state)
            _stage(events, stage_callbacks, "period_bdy_em_d_sub", state)
            _stage(events, stage_callbacks, "period_bdy_em_moist2_sub", state)
            for ij in tile_range:
                _debug(events, wrf_debug_fn, "call rk_phys_bc_dry_2")
                _stage(events, stage_callbacks, "rk_phys_bc_dry_2", state, event_name="set_phys_bc_dry_2", ij=ij)
                if num_3d_m >= PARAM_FIRST_SCALAR:
                    _stage(events, stage_callbacks, "set_physical_bc3d:moist", state, ij=ij)
                if num_3d_s >= PARAM_FIRST_SCALAR:
                    _stage(events, stage_callbacks, "set_physical_bc3d:scalar", state, ij=ij)
            _stage(events, stage_callbacks, "halo_em_moist_e_5_sub" if _read_int(config_flags, "h_sca_adv_order", 5) > 4 else "halo_em_moist_e_3_sub", state)
            if num_3d_s >= PARAM_FIRST_SCALAR:
                _stage(events, stage_callbacks, "halo_em_scalar_e_5_sub" if _read_int(config_flags, "h_sca_adv_order", 5) > 4 else "halo_em_scalar_e_3_sub", state)

    for ij in tile_range:
        tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
        _debug(events, wrf_debug_fn, "call advance_ppt")
        _stage(events, stage_callbacks, "advance_ppt", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    for ij in tile_range:
        tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
        _debug(events, wrf_debug_fn, "call phy_prep_part2")
        _stage(events, stage_callbacks, "phy_prep_part2", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if _read_int(config_flags, "mp_physics", 0) != 0:
        for ij in tile_range:
            tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
            _debug(events, wrf_debug_fn, "call moist_physics_prep_em")
            _stage(events, stage_callbacks, "moist_physics_prep_em", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

        _debug(events, wrf_debug_fn, "call microphysics_driver")
        _stage(events, stage_callbacks, "microphysics_driver", state)

        for ij in tile_range:
            tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
            _debug(events, wrf_debug_fn, "call moist_physics_finish_em")
            _stage(events, stage_callbacks, "moist_physics_finish_em", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            _stage(events, stage_callbacks, "set_physical_bc3d:h_diabatic", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

        for ij in tile_range:
            tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
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
            _stage(events, stage_callbacks, "calc_p_rho_phi:final", state, ij=ij, its=calc_its, ite=calc_ite, jts=calc_jts, jte=calc_jte)

    for ij in tile_range:
        tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
        _debug(events, wrf_debug_fn, "call set_phys_bc_dry_2")
        _stage(events, stage_callbacks, "set_phys_bc_dry_2", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
        _stage(events, stage_callbacks, "set_physical_bc3d:tke_1", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
        _stage(events, stage_callbacks, "set_physical_bc3d:tke_2", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
        if num_3d_m >= PARAM_FIRST_SCALAR:
            _stage(events, stage_callbacks, "set_physical_bc3d:moist_qv", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    if _read_bool(config_flags, "specified") or _read_bool(config_flags, "nested"):
        for ij in tile_range:
            tile_its, tile_ite, tile_jts, tile_jte = _tile_bounds(grid, ij, its, ite, jts, jte)
            _debug(events, wrf_debug_fn, "call spec_bdy_final")
            for stage_name in (
                "spec_bdy_final:u",
                "spec_bdy_final:v",
                "spec_bdy_final:t",
                "spec_bdy_final:ph",
                "spec_bdy_final:mu",
            ):
                _stage(events, stage_callbacks, stage_name, state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)
            if num_3d_m >= PARAM_FIRST_SCALAR:
                _stage(events, stage_callbacks, "spec_bdy_final:moist", state, ij=ij, its=tile_its, ite=tile_ite, jts=tile_jts, jte=tile_jte)

    _stage(events, stage_callbacks, "halo_em_c_sub", state)
    _stage(events, stage_callbacks, "period_bdy_em_e_sub", state)
    _debug(events, wrf_debug_fn, "call set_w_surface", level=10)
    for ij in tile_range:
        _stage(events, stage_callbacks, "set_w_surface", state, ij=ij)

    _stage(
        events,
        stage_callbacks,
        "after_all_rk_steps",
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

    _debug(events, wrf_debug_fn, "call HALO_RK_E")
    if _read_int(config_flags, "h_mom_adv_order", 5) <= 4 and _read_int(config_flags, "h_sca_adv_order", 5) <= 4:
        _stage(events, stage_callbacks, "halo_em_e_3_sub", state)
    elif _read_int(config_flags, "h_mom_adv_order", 5) <= 6 and _read_int(config_flags, "h_sca_adv_order", 5) <= 6:
        _stage(events, stage_callbacks, "halo_em_e_5_sub", state)
    else:
        raise ValueError("solve_em invalid horizontal advection order")

    if num_3d_m >= PARAM_FIRST_SCALAR:
        _debug(events, wrf_debug_fn, "call HALO_RK_MOIST")
        _stage(events, stage_callbacks, "halo_em_moist_e_3_sub" if _read_int(config_flags, "h_sca_adv_order", 5) <= 4 else "halo_em_moist_e_5_sub", state)
    if num_3d_c >= PARAM_FIRST_SCALAR:
        _debug(events, wrf_debug_fn, "call HALO_RK_CHEM")
        _stage(events, stage_callbacks, "halo_em_chem_e_3_sub" if _read_int(config_flags, "h_sca_adv_order", 5) <= 4 else "halo_em_chem_e_5_sub", state)
    if int(num_tracer) >= PARAM_FIRST_SCALAR:
        _debug(events, wrf_debug_fn, "call HALO_RK_TRACER")
        _stage(events, stage_callbacks, "halo_em_tracer_e_3_sub" if _read_int(config_flags, "h_sca_adv_order", 5) <= 4 else "halo_em_tracer_e_5_sub", state)
    if num_3d_s >= PARAM_FIRST_SCALAR:
        _debug(events, wrf_debug_fn, "call HALO_RK_SCALAR")
        _stage(events, stage_callbacks, "halo_em_scalar_e_3_sub" if _read_int(config_flags, "h_sca_adv_order", 5) <= 4 else "halo_em_scalar_e_5_sub", state)

    _debug(events, wrf_debug_fn, "call end of solve_em")
    if _read_bool(config_flags, "coupler_on", False):
        just_read_auxinput4 = _call_alarm(events, "auxinput4", is_alarm_tstep_fn, grid, _read(grid, "auxinput4_alarm", False))
        _write(grid, "just_read_auxinput4", just_read_auxinput4)
    if _read_int(grid, "id", 1) == 1:
        just_read_boundary = _call_alarm(events, "boundary", is_alarm_tstep_fn, grid, _read(grid, "boundary_alarm", False))
        _write(grid, "just_read_boundary", just_read_boundary)

    return SolveEmResult(
        grid=grid,
        config_flags=config_flags,
        fields=state,
        ijk=ijk,
        subgrid_ijk=subgrid_ijk,
        rk_order=rk_order,
        num_sound_steps=num_sound_steps,
        dts=dts,
        adapt_step_flag=adapt_step_flag,
        restart_flag=restart_flag,
        diag_flag=diag_flag,
        curr_secs=curr_secs,
        curr_secs2=curr_secs2,
        max_vert_cfl_tmp=tuple(max_vert_cfl_tmp),
        max_horiz_cfl_tmp=tuple(max_horiz_cfl_tmp),
        events=events.as_tuple(),
    )


__all__ = ["SolveEmResult", "solve_em"]
