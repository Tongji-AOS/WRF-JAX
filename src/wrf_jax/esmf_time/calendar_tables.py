"""Calendar tables used by WRF ESMF time helpers."""

from __future__ import annotations

from wrf_jax.esmf_time.types import BaseTime

SECONDS_PER_DAY = 86400
MONTHS_PER_YEAR = 12
MDAY = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MDAY_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _month_boundaries(days_per_month: tuple[int, ...]) -> tuple[BaseTime, ...]:
    boundaries = [BaseTime()]
    cumulative_days = 0
    for days in days_per_month:
        cumulative_days += days
        boundaries.append(BaseTime(s=cumulative_days * SECONDS_PER_DAY))
    return tuple(boundaries)


MONTHBDYS = _month_boundaries(MDAY)
MONTHBDYS_LEAP = _month_boundaries(MDAY_LEAP)


__all__ = [
    "MDAY",
    "MDAY_LEAP",
    "MONTHBDYS",
    "MONTHBDYS_LEAP",
    "MONTHS_PER_YEAR",
    "SECONDS_PER_DAY",
]
