"""WRF ESMF calendar day/month table initialization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

from wrf_jax.esmf_time.types import BaseTime

MONTHS_PER_YEAR = 12
SECONDS_PER_DAY = 86400
MDAY = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MDAYLEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class CalendarDayTables:
    """Explicit replacement for WRF ESMF calendar module day tables."""

    mday: tuple[int, ...]
    mdayleap: tuple[int, ...]
    daym: tuple[int, ...]
    daymleap: tuple[int, ...]
    mdaycum: tuple[int, ...]
    mdayleapcum: tuple[int, ...]
    monthbdys: tuple[BaseTime, ...]
    monthbdysleap: tuple[BaseTime, ...]


def _day_to_month_table(month_days: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(month for month, days in enumerate(month_days, start=1) for _ in range(days))


def _cumulative_month_days(month_days: tuple[int, ...]) -> tuple[int, ...]:
    return (0, *tuple(accumulate(month_days)))


def _month_boundaries(cumulative_days: tuple[int, ...]) -> tuple[BaseTime, ...]:
    return tuple(BaseTime(s=SECONDS_PER_DAY * days, sn=0, sd=0) for days in cumulative_days)


def initdaym() -> CalendarDayTables:
    """Build WRF normal-year and leap-year day/month lookup tables."""

    mdaycum = _cumulative_month_days(MDAY)
    mdayleapcum = _cumulative_month_days(MDAYLEAP)
    return CalendarDayTables(
        mday=MDAY,
        mdayleap=MDAYLEAP,
        daym=_day_to_month_table(MDAY),
        daymleap=_day_to_month_table(MDAYLEAP),
        mdaycum=mdaycum,
        mdayleapcum=mdayleapcum,
        monthbdys=_month_boundaries(mdaycum),
        monthbdysleap=_month_boundaries(mdayleapcum),
    )


__all__ = [
    "CalendarDayTables",
    "MDAY",
    "MDAYLEAP",
    "MONTHS_PER_YEAR",
    "SECONDS_PER_DAY",
    "initdaym",
]
