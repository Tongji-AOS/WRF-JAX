"""WRF ``c_esmc_basetimediff`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.esmf_basetimedifference import esmf_basetimedifference
from wrf_jax.esmf_time.normalize_timeint import normalize_timeint
from wrf_jax.esmf_time.nsecondsinyear import nsecondsinyear
from wrf_jax.esmf_time.types import BaseTime, ESMFTime, ESMFTimeInterval


def c_esmc_basetimediff(time1: ESMFTime, time2: ESMFTime) -> ESMFTimeInterval:
    """Return WRF ``time1 - time2`` as an ``ESMF_TimeInterval``."""

    basetime = esmf_basetimedifference(time1.basetime, time2.basetime)
    seconds = int(basetime.s)
    if int(time1.yr) > int(time2.yr):
        for year in range(int(time2.yr), int(time1.yr)):
            seconds += nsecondsinyear(year)
    elif int(time2.yr) > int(time1.yr):
        for year in range(int(time1.yr), int(time2.yr)):
            seconds -= nsecondsinyear(year)
    return normalize_timeint(ESMFTimeInterval(BaseTime(s=seconds, sn=basetime.sn, sd=basetime.sd), yr=0))


__all__ = ["c_esmc_basetimediff"]
