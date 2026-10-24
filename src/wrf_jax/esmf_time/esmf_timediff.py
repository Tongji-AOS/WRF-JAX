"""WRF ``ESMF_TimeDiff`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.c_esmc_basetimediff import c_esmc_basetimediff
from wrf_jax.esmf_time.types import ESMFTime, ESMFTimeInterval


def esmf_timediff(time1: ESMFTime, time2: ESMFTime) -> ESMFTimeInterval:
    """Return WRF ``ESMF_TimeDiff(time1, time2)``."""

    return c_esmc_basetimediff(time1, time2)


__all__ = ["esmf_timediff"]
