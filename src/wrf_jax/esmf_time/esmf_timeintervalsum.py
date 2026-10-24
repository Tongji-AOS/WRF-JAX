"""WRF ``ESMF_TimeIntervalSum`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.esmf_basetimesum import esmf_basetimesum
from wrf_jax.esmf_time.normalize_timeint import normalize_timeint
from wrf_jax.esmf_time.timeintchecknormalized import timeintchecknormalized
from wrf_jax.esmf_time.types import ESMFTimeInterval


def esmf_timeintervalsum(
    timeinterval1: ESMFTimeInterval,
    timeinterval2: ESMFTimeInterval,
) -> ESMFTimeInterval:
    """Return WRF ``timeinterval1 + timeinterval2`` for normalized intervals."""

    timeintchecknormalized(timeinterval1, "ESMF_TimeIntervalSum arg1")
    timeintchecknormalized(timeinterval2, "ESMF_TimeIntervalSum arg2")

    result = ESMFTimeInterval(
        basetime=esmf_basetimesum(timeinterval1.basetime, timeinterval2.basetime),
        yr=timeinterval1.yr,
    )
    return normalize_timeint(result)


__all__ = ["esmf_timeintervalsum"]
