"""WRF ``ESMF_TimeSet`` translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence

from wrf_jax.esmf_time.esmf_calendarcreate import ESMF_FAILURE, ESMF_SUCCESS
from wrf_jax.esmf_time.esmf_basetimesum import esmf_basetimesum
from wrf_jax.esmf_time.normalize_time import normalize_time
from wrf_jax.esmf_time.timeaddmonths import timeaddmonths
from wrf_jax.esmf_time.types import BaseTime, ESMFCalendar, ESMFTime

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class TimeSetResult:
    time: ESMFTime
    rc: int = ESMF_SUCCESS


def _set_holder(holder: MutableSequence[int], value: int, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def esmf_timeset(
    time: ESMFTime | None = None,
    *,
    yy: int | None = None,
    yyl: int | None = None,
    mm: int | None = None,
    dd: int | None = None,
    d: int | None = None,
    dl: int | None = None,
    h: int | None = None,
    m: int | None = None,
    s: int | None = None,
    sl: int | None = None,
    ms: int | None = None,
    us: int | None = None,
    ns: int | None = None,
    d_: float | None = None,
    h_: float | None = None,
    m_: float | None = None,
    s_: float | None = None,
    ms_: float | None = None,
    us_: float | None = None,
    ns_: float | None = None,
    sn: int | None = None,
    sd: int | None = None,
    calendar: ESMFCalendar | None = None,
    rc: MutableSequence[int] | None = None,
) -> TimeSetResult:
    """Initialize a WRF ``ESMF_Time`` from selected optional units."""

    _ = (yyl, d, dl, sl, us, ns, d_, h_, m_, s_, ms_, us_, ns_)
    if rc is not None:
        _set_holder(rc, ESMF_FAILURE, "rc")

    source_calendar = calendar if calendar is not None else (time.calendar if time is not None else None)
    result = ESMFTime(BaseTime(s=0, sn=0, sd=0), yr=0, calendar=source_calendar)
    if yy is not None:
        result = ESMFTime(result.basetime, yr=int(yy), calendar=result.calendar)

    if mm is not None:
        month_result = timeaddmonths(result, int(mm))
        result = month_result.time
        if month_result.ierr == ESMF_FAILURE and rc is not None:
            _set_holder(rc, ESMF_FAILURE, "rc")
            return TimeSetResult(time=result, rc=ESMF_FAILURE)

    basetime = result.basetime
    if dd is not None:
        basetime = esmf_basetimesum(basetime, BaseTime(s=SECONDS_PER_DAY * (int(dd) - 1)))
    if h is not None:
        basetime = esmf_basetimesum(basetime, BaseTime(s=3600 * int(h)))
    if m is not None:
        basetime = esmf_basetimesum(basetime, BaseTime(s=60 * int(m)))
    if s is not None:
        basetime = esmf_basetimesum(basetime, BaseTime(s=int(s)))

    if sn is not None and sd is None:
        raise ValueError("ESMF_TimeSet: Must specify Sd if Sn is specified")
    if sd is not None and ms is not None:
        raise ValueError("ESMF_TimeSet: Must not specify both Sd and MS")

    if ms is not None:
        basetime = BaseTime(s=basetime.s, sn=int(ms), sd=1000)
    elif sd is not None:
        basetime = BaseTime(s=basetime.s, sn=int(sn) if sn is not None else 0, sd=int(sd))
    else:
        basetime = BaseTime(s=basetime.s, sn=0, sd=0)

    result = normalize_time(ESMFTime(basetime, yr=result.yr, calendar=result.calendar))
    if rc is not None:
        _set_holder(rc, ESMF_SUCCESS, "rc")
    return TimeSetResult(time=result, rc=ESMF_SUCCESS)


__all__ = ["SECONDS_PER_DAY", "TimeSetResult", "esmf_timeset"]
