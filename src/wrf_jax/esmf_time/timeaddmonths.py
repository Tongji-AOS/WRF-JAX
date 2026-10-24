"""WRF ``timeaddmonths`` translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence

from wrf_jax.esmf_time.calendar_tables import MONTHBDYS, MONTHBDYS_LEAP, MONTHS_PER_YEAR
from wrf_jax.esmf_time.esmf_basetimesum import esmf_basetimesum
from wrf_jax.esmf_time.esmf_calendarcreate import ESMF_FAILURE, ESMF_SUCCESS
from wrf_jax.esmf_time.nfeb import nfeb
from wrf_jax.esmf_time.types import ESMFTime


@dataclass(frozen=True)
class TimeAddMonthsResult:
    """Result of WRF ``timeaddmonths``."""

    time: ESMFTime
    ierr: int


def _set_holder(holder: MutableSequence[int], value: int, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def timeaddmonths(
    time: ESMFTime,
    mm: int,
    ierr: MutableSequence[int] | None = None,
) -> TimeAddMonthsResult:
    """Increment ``time`` by seconds from start of year to month ``mm``.

    The WRF routine does not normalize the containing ``ESMF_Time`` after the
    base-time addition, so year rollover is intentionally not applied here.
    """

    month = int(mm)
    if month < 1 or month > MONTHS_PER_YEAR:
        if ierr is not None:
            _set_holder(ierr, ESMF_FAILURE, "ierr")
        return TimeAddMonthsResult(time=time, ierr=ESMF_FAILURE)

    boundaries = MONTHBDYS_LEAP if nfeb(time.yr) == 29 else MONTHBDYS
    new_basetime = esmf_basetimesum(time.basetime, boundaries[month - 1])
    new_time = ESMFTime(basetime=new_basetime, yr=time.yr, calendar=time.calendar)

    if ierr is not None:
        _set_holder(ierr, ESMF_SUCCESS, "ierr")
    return TimeAddMonthsResult(time=new_time, ierr=ESMF_SUCCESS)


__all__ = ["TimeAddMonthsResult", "timeaddmonths"]
