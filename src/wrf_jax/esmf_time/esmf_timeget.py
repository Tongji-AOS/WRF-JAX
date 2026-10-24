"""WRF ``ESMF_TimeGet`` translation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import MutableSequence

from wrf_jax.esmf_time.esmf_calendarcreate import ESMF_SUCCESS
from wrf_jax.esmf_time.esmf_timediff import esmf_timediff
from wrf_jax.esmf_time.esmf_timegetdayofyearinteger import esmf_timegetdayofyearinteger
from wrf_jax.esmf_time.esmf_timeintervalset import SECONDS_PER_DAY, esmf_timeintervalset
from wrf_jax.esmf_time.esmf_timeintervalsum import esmf_timeintervalsum
from wrf_jax.esmf_time.esmfold_timegetstring import esmfold_timegetstring
from wrf_jax.esmf_time.esmfold_timeintervalgetstring import _fortran_mod, _trunc_div
from wrf_jax.esmf_time.timegetdayofmonth import timegetdayofmonth
from wrf_jax.esmf_time.timegetmonth import timegetmonth
from wrf_jax.esmf_time.types import BaseTime, ESMFTime, ESMFTimeInterval


@dataclass(frozen=True)
class TimeGetResult:
    yy: int | None = None
    mm: int | None = None
    dd: int | None = None
    h: int | None = None
    m: int | None = None
    s: int | None = None
    ms: int | None = None
    sn: int | None = None
    sd: int | None = None
    day_of_year: int | None = None
    day_of_year_r8: float | None = None
    day_of_year_intvl: ESMFTimeInterval | None = None
    time_string: str | None = None
    rc: int = ESMF_SUCCESS


def _set_holder(holder: MutableSequence, value, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def _nint(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def esmf_timeget(
    time: ESMFTime,
    *,
    yy: MutableSequence[int] | None = None,
    mm: MutableSequence[int] | None = None,
    dd: MutableSequence[int] | None = None,
    h: MutableSequence[int] | None = None,
    m: MutableSequence[int] | None = None,
    s: MutableSequence[int] | None = None,
    ms: MutableSequence[int] | None = None,
    sn: MutableSequence[int] | None = None,
    sd: MutableSequence[int] | None = None,
    day_of_year: MutableSequence[int] | None = None,
    day_of_year_r8: MutableSequence[float] | None = None,
    day_of_year_intvl: MutableSequence[ESMFTimeInterval] | None = None,
    time_string: MutableSequence[str] | None = None,
    time_string_len: int | None = None,
    rc: MutableSequence[int] | None = None,
) -> TimeGetResult:
    """Get selected components from a WRF ``ESMF_Time``."""

    seconds = int(time.basetime.s)
    frac_num = int(time.basetime.sn)
    frac_den = int(time.basetime.sd)

    yy_value = int(time.yr) if yy is not None else None
    mm_value = timegetmonth(time) if mm is not None else None
    dd_value = timegetdayofmonth(time) if dd is not None else None

    h_value = None
    if h is not None:
        h_value = _trunc_div(_fortran_mod(seconds, SECONDS_PER_DAY), 3600)

    m_value = None
    if m is not None:
        m_value = _trunc_div(_fortran_mod(seconds, 3600), 60)

    s_value = None
    if s is not None:
        s_value = _fortran_mod(seconds, 60)
        if dd is not None and h is None and m is None:
            s_value = _fortran_mod(seconds, SECONDS_PER_DAY)

    ms_value = None
    if ms is not None:
        if frac_den != 0:
            ms_value = _nint((float(frac_num) / float(frac_den)) * 1000.0)
        else:
            ms_value = 0

    sn_value = None
    sd_value = None
    if sn is not None and sd is not None:
        sn_value = frac_num
        sd_value = frac_den

    day_of_year_value = esmf_timegetdayofyearinteger(time) if day_of_year is not None else None

    day_of_year_r8_value = None
    if day_of_year_r8 is not None:
        rsec = float(seconds)
        if frac_den != 0:
            rsec += float(frac_num) / float(frac_den)
        day_of_year_r8_value = (rsec / float(SECONDS_PER_DAY)) + 1.0

    time_string_value = None
    if time_string is not None:
        year = int(time.yr)
        month = timegetmonth(time)
        dayofmonth = timegetdayofmonth(time)
        hour = _trunc_div(_fortran_mod(seconds, SECONDS_PER_DAY), 3600)
        minute = _trunc_div(_fortran_mod(seconds, 3600), 60)
        second = _fortran_mod(seconds, 60)
        time_string_value = esmfold_timegetstring(
            year,
            month,
            dayofmonth,
            hour,
            minute,
            second,
            timestring_len=time_string_len,
        )

    day_of_year_intvl_value = None
    if day_of_year_intvl is not None:
        begofyear = ESMFTime(BaseTime(s=0, sn=0, sd=0), yr=int(time.yr), calendar=time.calendar)
        day_step = esmf_timeintervalset(d=1, s=0).timeinterval
        day_of_year_intvl_value = esmf_timeintervalsum(esmf_timediff(time, begofyear), day_step)

    if yy is not None:
        _set_holder(yy, yy_value, "YY")
    if mm is not None:
        _set_holder(mm, mm_value, "MM")
    if dd is not None:
        _set_holder(dd, dd_value, "DD")
    if h is not None:
        _set_holder(h, h_value, "H")
    if m is not None:
        _set_holder(m, m_value, "M")
    if s is not None:
        _set_holder(s, s_value, "S")
    if ms is not None:
        _set_holder(ms, ms_value, "MS")
    if sn is not None and sd is not None:
        _set_holder(sn, sn_value, "Sn")
        _set_holder(sd, sd_value, "Sd")
    if day_of_year is not None:
        _set_holder(day_of_year, day_of_year_value, "dayOfYear")
    if day_of_year_r8 is not None:
        _set_holder(day_of_year_r8, day_of_year_r8_value, "dayOfYear_r8")
    if day_of_year_intvl is not None:
        _set_holder(day_of_year_intvl, day_of_year_intvl_value, "dayOfYear_intvl")
    if time_string is not None:
        _set_holder(time_string, time_string_value, "timeString")
    if rc is not None:
        _set_holder(rc, ESMF_SUCCESS, "rc")

    return TimeGetResult(
        yy=yy_value,
        mm=mm_value,
        dd=dd_value,
        h=h_value,
        m=m_value,
        s=s_value,
        ms=ms_value,
        sn=sn_value,
        sd=sd_value,
        day_of_year=day_of_year_value,
        day_of_year_r8=day_of_year_r8_value,
        day_of_year_intvl=day_of_year_intvl_value,
        time_string=time_string_value,
        rc=ESMF_SUCCESS,
    )


__all__ = ["TimeGetResult", "esmf_timeget"]
