"""Minimal production clock helper from WRF ``module_domain``."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass
from typing import Any

from wrf_jax.esmf_time.esmf_clockgetcurrtime import esmf_clockgetcurrtime
from wrf_jax.esmf_time.esmf_clockgetstarttime import esmf_clockgetstarttime
from wrf_jax.esmf_time.esmf_clockgetstoptime import esmf_clockgetstoptime
from wrf_jax.esmf_time.esmf_clockgettimestep import esmf_clockgettimestep
from wrf_jax.esmf_time.esmf_timeget import esmf_timeget
from wrf_jax.esmf_time.esmf_timeintervalget import esmf_timeintervalget
from wrf_jax.esmf_time.fraction_to_string import fraction_to_string
from wrf_jax.compat.module_date_time import wrf_timetoa


@dataclass(frozen=True)
class DomainClockGetResult:
    """Explicit return object for the subset of WRF ``domain_clock_get`` in use."""

    current_time: Any | None = None
    current_timestr: str | None = None
    current_timestr_frac: str | None = None
    start_time: Any | None = None
    start_timestr: str | None = None
    stop_time: Any | None = None
    stop_timestr: str | None = None
    time_step: Any | None = None
    time_stepstr: str | None = None
    time_stepstr_frac: str | None = None
    advance_count: int | None = None
    current_day_of_year_real: float | None = None
    minutes_since_simulation_start: float | None = None
    time_since_simulation_start: Any | None = None
    simulation_start_time: Any | None = None
    simulation_start_time_str: str | None = None


def _read_field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise ValueError(f"grid is missing required field {name!r}")
        return source[name]
    if not hasattr(source, name):
        raise ValueError(f"grid is missing required field {name!r}")
    return getattr(source, name)


def _requested(value: Any) -> bool:
    return value is not None and value is not False


def _write_requested(holder: Any, value: Any) -> None:
    if isinstance(holder, MutableSequence):
        if len(holder) < 1:
            raise ValueError("output holder must contain at least one element")
        holder[0] = value


def _time_to_string(value: Any) -> str:
    return wrf_timetoa(value).rstrip()


def _clock_value(grid: Any, getter: Any, label: str) -> Any:
    rc = [-1]
    result = getter(_read_field(grid, "domain_clock"), rc=rc)
    if rc[0] != 0:
        raise RuntimeError(f"{label}: WRFU_ClockGet failed")
    return result


def domain_get_current_time(grid: Any) -> Any:
    return _clock_value(grid, esmf_clockgetcurrtime, "domain_get_current_time")


def domain_get_start_time(grid: Any) -> Any:
    return _clock_value(grid, esmf_clockgetstarttime, "domain_get_start_time")


def domain_get_stop_time(grid: Any) -> Any:
    return _clock_value(grid, esmf_clockgetstoptime, "domain_get_stop_time")


def domain_get_time_step(grid: Any) -> Any:
    return _clock_value(grid, esmf_clockgettimestep, "domain_get_time_step")


def domain_get_advance_count(grid: Any) -> int:
    clock = _read_field(grid, "domain_clock")
    if isinstance(clock, Mapping):
        if "advance_count" in clock:
            return int(clock["advance_count"])
        if "clockint" in clock:
            clock = clock["clockint"]
    clockint = getattr(clock, "clockint", clock)
    if isinstance(clockint, Mapping):
        return int(clockint["advance_count"])
    return int(clockint.advance_count)


def domain_clock_get(
    grid: Any,
    *,
    current_time: Any = None,
    current_timestr: Any = None,
    current_timestr_frac: Any = None,
    start_time: Any = None,
    start_timestr: Any = None,
    stop_time: Any = None,
    stop_timestr: Any = None,
    time_step: Any = None,
    time_stepstr: Any = None,
    time_stepstr_frac: Any = None,
    advance_count: Any = None,
    current_day_of_year_real: Any = None,
    **unused_context: Any,
) -> DomainClockGetResult:
    """Return the WRF clock fields used by the production direct trajectory."""

    del unused_context

    current_time_value = None
    current_timestr_value = None
    current_timestr_frac_value = None
    current_day_of_year_real_value = None
    if (
        _requested(current_time)
        or _requested(current_timestr)
        or _requested(current_timestr_frac)
        or _requested(current_day_of_year_real)
    ):
        current_time_value = domain_get_current_time(grid)
        if _requested(current_time):
            _write_requested(current_time, current_time_value)
        if _requested(current_timestr):
            current_timestr_value = _time_to_string(current_time_value)
            _write_requested(current_timestr, current_timestr_value)
        if _requested(current_timestr_frac):
            sn = [0]
            sd = [0]
            esmf_timeget(current_time_value, sn=sn, sd=sd, rc=[-1])
            current_timestr_frac_value = (
                _time_to_string(current_time_value) + fraction_to_string(sn[0], sd[0]).rstrip()
            )
            _write_requested(current_timestr_frac, current_timestr_frac_value)
        if _requested(current_day_of_year_real):
            day_holder = [0.0]
            esmf_timeget(current_time_value, day_of_year_r8=day_holder, rc=[-1])
            current_day_of_year_real_value = float(day_holder[0]) - 1.0
            _write_requested(current_day_of_year_real, current_day_of_year_real_value)

    start_time_value = None
    start_timestr_value = None
    if _requested(start_time) or _requested(start_timestr):
        start_time_value = domain_get_start_time(grid)
        if _requested(start_time):
            _write_requested(start_time, start_time_value)
        if _requested(start_timestr):
            start_timestr_value = _time_to_string(start_time_value)
            _write_requested(start_timestr, start_timestr_value)

    stop_time_value = None
    stop_timestr_value = None
    if _requested(stop_time) or _requested(stop_timestr):
        stop_time_value = domain_get_stop_time(grid)
        if _requested(stop_time):
            _write_requested(stop_time, stop_time_value)
        if _requested(stop_timestr):
            stop_timestr_value = _time_to_string(stop_time_value)
            _write_requested(stop_timestr, stop_timestr_value)

    time_step_value = None
    time_stepstr_value = None
    time_stepstr_frac_value = None
    if _requested(time_step) or _requested(time_stepstr) or _requested(time_stepstr_frac):
        time_step_value = domain_get_time_step(grid)
        if _requested(time_step):
            _write_requested(time_step, time_step_value)
        if _requested(time_stepstr):
            holder = [""]
            esmf_timeintervalget(time_step_value, time_string=holder, rc=[-1])
            time_stepstr_value = holder[0].rstrip()
            _write_requested(time_stepstr, time_stepstr_value)
        if _requested(time_stepstr_frac):
            tmp_str = [""]
            sn = [0]
            sd = [0]
            esmf_timeintervalget(time_step_value, time_string=tmp_str, sn=sn, sd=sd, rc=[-1])
            time_stepstr_frac_value = tmp_str[0].rstrip() + fraction_to_string(sn[0], sd[0]).rstrip()
            _write_requested(time_stepstr_frac, time_stepstr_frac_value)

    advance_count_value = None
    if _requested(advance_count):
        advance_count_value = domain_get_advance_count(grid)
        _write_requested(advance_count, advance_count_value)

    return DomainClockGetResult(
        current_time=current_time_value,
        current_timestr=current_timestr_value,
        current_timestr_frac=current_timestr_frac_value,
        start_time=start_time_value,
        start_timestr=start_timestr_value,
        stop_time=stop_time_value,
        stop_timestr=stop_timestr_value,
        time_step=time_step_value,
        time_stepstr=time_stepstr_value,
        time_stepstr_frac=time_stepstr_frac_value,
        advance_count=advance_count_value,
        current_day_of_year_real=current_day_of_year_real_value,
    )


__all__ = [
    "DomainClockGetResult",
    "domain_clock_get",
    "domain_get_current_time",
    "domain_get_start_time",
    "domain_get_stop_time",
    "domain_get_time_step",
]
