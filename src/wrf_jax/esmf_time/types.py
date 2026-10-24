"""Small immutable data types for WRF ESMF time translation."""

from __future__ import annotations

from dataclasses import dataclass, field
@dataclass(frozen=True)
class BaseTime:
    """WRF ``ESMF_BaseTime`` subset used by the migrated functions."""

    s: int = 0
    sn: int = 0
    sd: int = 0


@dataclass(frozen=True)
class ESMFCalendarType:
    """WRF ``ESMF_CalendarType`` integer enum wrapper."""

    caltype: int


ESMF_CAL_GREGORIAN = ESMFCalendarType(1)
ESMF_CAL_JULIAN = ESMFCalendarType(2)
ESMF_CAL_NOLEAP = ESMFCalendarType(3)
ESMF_CAL_360DAY = ESMFCalendarType(4)
ESMF_CAL_GENERIC = ESMFCalendarType(5)
ESMF_CAL_NOCALENDAR = ESMFCalendarType(6)


@dataclass(frozen=True)
class ESMFDaysPerYear:
    """WRF ``ESMF_DaysPerYear`` subset used by calendar creation."""

    d: int = 0


@dataclass(frozen=True)
class ESMFCalendar:
    """WRF ``ESMF_Calendar`` subset used by the migrated ESMF routines."""

    calendar_type: ESMFCalendarType = field(default_factory=lambda: ESMFCalendarType(0))
    set: bool = False
    days_per_month: tuple[int, ...] = field(default_factory=lambda: (0,) * 12)
    seconds_per_day: int = 0
    seconds_per_year: int = 0
    days_per_year: ESMFDaysPerYear = field(default_factory=ESMFDaysPerYear)


@dataclass(frozen=True)
class ESMFTime:
    """WRF ``ESMF_Time`` subset used by the migrated functions."""

    basetime: BaseTime
    yr: int = 0
    calendar: ESMFCalendar | None = None


@dataclass(frozen=True)
class ESMFTimeInterval:
    """WRF ``ESMF_TimeInterval`` subset used by migrated functions."""

    basetime: BaseTime
    yr: int = 0


@dataclass(frozen=True)
class ESMFClockInt:
    """WRF ``ESMF_ClockInt`` subset used by clock getter translations."""

    time_step: ESMFTimeInterval
    start_time: ESMFTime
    stop_time: ESMFTime
    curr_time: ESMFTime
    ref_time: ESMFTime = field(default_factory=lambda: ESMFTime(BaseTime()))
    prev_time: ESMFTime = field(default_factory=lambda: ESMFTime(BaseTime()))
    advance_count: int = 0
    num_alarms: int = 0
    alarm_list: tuple[ESMFAlarm | None, ...] = ()


@dataclass(frozen=True)
class ESMFClock:
    """WRF ``ESMF_Clock`` wrapper around the clock internals."""

    clockint: ESMFClockInt


@dataclass(frozen=True)
class ESMFAlarmInt:
    """WRF ``ESMF_AlarmInt`` subset needed by alarm getter translations."""

    ring_interval: ESMFTimeInterval = field(default_factory=lambda: ESMFTimeInterval(BaseTime()))
    ring_time: ESMFTime = field(default_factory=lambda: ESMFTime(BaseTime()))
    prev_ring_time: ESMFTime = field(default_factory=lambda: ESMFTime(BaseTime()))
    stop_time: ESMFTime = field(default_factory=lambda: ESMFTime(BaseTime()))
    ringing: bool = False
    enabled: bool = False
    ring_time_set: bool = False
    ring_interval_set: bool = False
    stop_time_set: bool = False


@dataclass(frozen=True)
class ESMFAlarm:
    """WRF ``ESMF_Alarm`` wrapper around optional alarm internals."""

    alarmint: ESMFAlarmInt | None = None
