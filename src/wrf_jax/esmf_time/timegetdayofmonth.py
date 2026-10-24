"""WRF ``timegetdayofmonth`` translation."""

from __future__ import annotations

from typing import MutableSequence

from wrf_jax.esmf_time.calendar_tables import MONTHBDYS, MONTHBDYS_LEAP, SECONDS_PER_DAY
from wrf_jax.esmf_time.esmf_basetimedifference import esmf_basetimedifference
from wrf_jax.esmf_time.nfeb import nfeb
from wrf_jax.esmf_time.timegetmonth import timegetmonth
from wrf_jax.esmf_time.types import ESMFTime


def _set_holder(holder: MutableSequence[int], value: int, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def timegetdayofmonth(time: ESMFTime, dd: MutableSequence[int] | None = None) -> int:
    """Extract WRF day-of-month from a normalized-in-year ``ESMF_Time``."""

    month = timegetmonth(time)
    boundaries = MONTHBDYS_LEAP if nfeb(time.yr) == 29 else MONTHBDYS
    tmpbasetime = esmf_basetimedifference(time.basetime, boundaries[month - 1])
    day = int(tmpbasetime.s) // SECONDS_PER_DAY + 1

    if dd is not None:
        _set_holder(dd, day, "DD")
    return day


__all__ = ["timegetdayofmonth"]
