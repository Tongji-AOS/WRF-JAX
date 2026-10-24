"""Translation of WRF ``dyn_em/module_first_rk_step_part1`` orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from wrf_jax.compat.module_domain import domain_clock_get
from wrf_jax.esmf_time.esmf_timeget import esmf_timeget

NOAHMPSCHEME = 4


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
class FirstRkStepPart1Result:
    """Explicit state and call trace for WRF ``first_rk_step_part1``."""

    grid: Any
    fields: Mapping[str, Any]
    events: tuple[str, ...]
    rk_step: int
    l_flux: bool
    hydro_dt: float
    current_time_parts: tuple[int, int, int, int, int] | None
    iswater: int
    isurban: int
    mminlu: str


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


def _time_parts_from_clock(grid: Any) -> tuple[int, int, int, int, int] | None:
    try:
        clock_result = domain_clock_get(grid, current_time=True, current_timestr=True)
    except Exception:
        current_time = _read(grid, "current_time", None)
    else:
        current_time = clock_result.current_time
    if current_time is None:
        return None
    time_result = esmf_timeget(current_time, yy=[0], day_of_year=[0], h=[0], m=[0], s=[0])
    return (
        int(time_result.yy),
        int(time_result.day_of_year),
        int(time_result.h),
        int(time_result.m),
        int(time_result.s),
    )


def first_rk_step_part1(
    *,
    grid: Any,
    config_flags: Any,
    fields: Mapping[str, Any] | None = None,
    moist: Any = None,
    moist_tend: Any = None,
    chem: Any = None,
    chem_tend: Any = None,
    tracer: Any = None,
    tracer_tend: Any = None,
    scalar: Any = None,
    scalar_tend: Any = None,
    fdda3d: Any = None,
    fdda2d: Any = None,
    aerod: Any = None,
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
    f_flux: bool | None = None,
    aerocu: Any = None,
    restart_flag: bool | None = None,
    feedback_is_ready: bool | None = None,
    callbacks: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any] | None]] | None = None,
    record_events: bool = True,
) -> FirstRkStepPart1Result:
    """Run WRF ``first_rk_step_part1`` as explicit step orchestration.

    The heavy physics, radiation, fire, FDDA, and halo routines are invoked
    through named callbacks. This keeps the Fortran call order and branch logic
    testable while translated child kernels remain independent.
    """

    del (
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
    )

    state: dict[str, Any] = dict(fields or {})
    for name, value in {
        "moist": moist,
        "moist_tend": moist_tend,
        "chem": chem,
        "chem_tend": chem_tend,
        "tracer": tracer,
        "tracer_tend": tracer_tend,
        "scalar": scalar,
        "scalar_tend": scalar_tend,
        "fdda3d": fdda3d,
        "fdda2d": fdda2d,
        "aerod": aerod,
        "aerocu": aerocu,
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
        "adapt_step_flag": adapt_step_flag,
        "curr_secs": curr_secs,
        "restart_flag": restart_flag,
        "feedback_is_ready": feedback_is_ready,
    }.items():
        if value is not None:
            state[name] = value

    grid_out = grid
    state["grid"] = grid_out
    state["config_flags"] = config_flags
    events = _EventLog(record_events)
    rk_step = 1
    l_flux = bool(f_flux) if f_flux is not None else False
    hydro_dt = 0.0
    iswater = 0
    isurban = 0
    mminlu = ""

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
            state["grid"] = grid_out

    call_stage("get_ijk_from_subgrid")

    tile_count = _num_tiles(grid_out)
    _run_tiles(tile_count, "debug:call init_zero_tendency", "init_zero_tendency", events, call_stage)

    call_stage("halo_em_phys_a_sub")

    _run_tiles(tile_count, "debug:call phy_prep", "phy_prep", events, call_stage)

    if events.enabled:
        events.append("call:domain_clock_get")
    current_time_parts = _time_parts_from_clock(grid_out)
    if events.enabled:
        events.append("call:wrfu_timeget")

    call_stage("pre_radiation_driver")
    if events.enabled:
        events.append("debug:call radiation_driver")
    call_stage("radiation_driver")

    if (
        _read_int(config_flags, "sf_surface_physics", 0) == NOAHMPSCHEME
        and _read_int(config_flags, "opt_run", 0) == 5
        and _read_int(grid_out, "stepwtd", 1) != 0
        and _read_int(grid_out, "itimestep", 0) % _read_int(grid_out, "stepwtd", 1) == 0
    ):
        call_stage("halo_em_hydro_noahmp_sub")

    grid_id = _read_int(grid_out, "id", 1)
    call_stage("nl_get_iswater")
    iswater = int(_config_namelist_value(config_flags, "iswater", grid_id, 0))
    state["iswater"] = iswater

    call_stage("nl_get_isurban")
    isurban = int(_config_namelist_value(config_flags, "isurban", grid_id, 0))
    state["isurban"] = isurban

    call_stage("nl_get_mminlu")
    mminlu = str(_config_namelist_value(config_flags, "mminlu", grid_id, ""))
    state["mminlu"] = mminlu

    call_stage("halo_pwp_sub")

    if events.enabled:
        events.append("debug:call surface_driver")
    hydro_dt = 0.0 if _read_int(grid_out, "num_nests", 0) < 1 else -1.0
    state["hydro_dt"] = hydro_dt
    state["do_bioe"] = False
    state["do_meganfile"] = False
    call_stage("surface_driver")

    if events.enabled:
        events.append("debug:call pbl_driver")
    call_stage("pbl_driver")

    if (_read_int(grid_out, "sr_x", 0) > 0 or _read_int(grid_out, "sr_y", 0) > 0) and _read_int(
        config_flags, "ifire", 0
    ) == 2:
        call_stage("fire_driver_em_step")

    if events.enabled:
        events.append("debug:call cumulus_driver")
    state["l_flux"] = l_flux
    call_stage("cumulus_driver")

    if _read_int(config_flags, "cu_diag", 0) == 1:
        _run_tiles(tile_count, None, "convtrans_prep", events, call_stage)

    if events.enabled:
        events.append("debug:call shallow_cumulus_driver")
    call_stage("shallowcu_driver")

    call_stage("force_scm")

    call_stage("halo_em_fdda_sfc_sub")

    if events.enabled:
        events.append("debug:call fddagd_driver")
    call_stage("fddagd_driver")

    result_fields = {key: value for key, value in state.items() if key not in {"grid", "config_flags"}}
    return FirstRkStepPart1Result(
        grid=grid_out,
        fields=result_fields,
        events=events.as_tuple(),
        rk_step=rk_step,
        l_flux=l_flux,
        hydro_dt=hydro_dt,
        current_time_parts=current_time_parts,
        iswater=iswater,
        isurban=isurban,
        mminlu=mminlu,
    )


__all__ = ["FirstRkStepPart1Result", "NOAHMPSCHEME", "first_rk_step_part1"]
