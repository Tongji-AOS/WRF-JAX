"""Translation of ``wrf_esmf_calendarmod::esmf_calendarcreate``."""

from __future__ import annotations

from collections.abc import MutableSequence

from wrf_jax.esmf_time.initdaym import MDAY, SECONDS_PER_DAY, initdaym
from wrf_jax.esmf_time.types import ESMFCalendar, ESMFCalendarType, ESMFDaysPerYear, ESMF_CAL_GREGORIAN

ESMF_SUCCESS = 0
ESMF_FAILURE = -1


def _set_rc(rc: MutableSequence[int] | None, value: int) -> None:
    if rc is None:
        return
    if len(rc) == 0:
        raise ValueError("rc output holder must contain at least one element")
    rc[0] = value


def esmf_calendarcreate(
    calendartype: ESMFCalendarType,
    name: str | None = None,
    rc: MutableSequence[int] | None = None,
) -> ESMFCalendar:
    """Create WRF's hard-coded Gregorian ESMF calendar.

    The WRF 4.5 implementation ignores ``name`` and rejects every calendar type
    except ``ESMF_CAL_GREGORIAN``.
    """

    del name
    _set_rc(rc, ESMF_FAILURE)

    if calendartype.caltype != ESMF_CAL_GREGORIAN.caltype:
        return ESMFCalendar()

    tables = initdaym()
    days_per_year = ESMFDaysPerYear(d=len(tables.daym))
    calendar = ESMFCalendar(
        calendar_type=ESMF_CAL_GREGORIAN,
        set=True,
        days_per_month=MDAY,
        seconds_per_day=SECONDS_PER_DAY,
        seconds_per_year=SECONDS_PER_DAY * days_per_year.d,
        days_per_year=days_per_year,
    )
    _set_rc(rc, ESMF_SUCCESS)
    return calendar


__all__ = ["ESMF_FAILURE", "ESMF_SUCCESS", "esmf_calendarcreate"]
