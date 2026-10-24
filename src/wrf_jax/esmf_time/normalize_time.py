"""WRF ``normalize_time`` translation."""

from __future__ import annotations

from dataclasses import replace

from wrf_jax.esmf_time.esmf_basetimedifference import esmf_basetimedifference
from wrf_jax.esmf_time.esmf_basetimege import esmf_basetimege
from wrf_jax.esmf_time.esmf_basetimelt import esmf_basetimelt
from wrf_jax.esmf_time.esmf_basetimesum import esmf_basetimesum
from wrf_jax.esmf_time.normalize_basetime import normalize_basetime
from wrf_jax.esmf_time.nsecondsinyear import nsecondsinyear
from wrf_jax.esmf_time.types import BaseTime, ESMFTime


def normalize_time(time: ESMFTime) -> ESMFTime:
    """Normalize WRF ``ESMF_Time`` into a non-negative year-local basetime."""

    basetime = normalize_basetime(time.basetime)
    year = int(time.yr)
    zerotime = BaseTime()

    while esmf_basetimelt(basetime, zerotime):
        year -= 1
        basetime = esmf_basetimesum(basetime, BaseTime(s=nsecondsinyear(year)))

    while True:
        cmptime = BaseTime(s=nsecondsinyear(year))
        if not esmf_basetimege(basetime, cmptime):
            break
        basetime = esmf_basetimedifference(basetime, cmptime)
        year += 1

    return replace(time, basetime=basetime, yr=year)


__all__ = ["normalize_time"]
