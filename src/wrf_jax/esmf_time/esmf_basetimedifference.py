"""WRF ``ESMF_BaseTimeDifference`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.esmf_basetimesum import esmf_basetimesum
from wrf_jax.esmf_time.types import BaseTime


def esmf_basetimedifference(basetime1: BaseTime, basetime2: BaseTime) -> BaseTime:
    """Return the normalized WRF difference ``basetime1 - basetime2``."""

    neg2 = BaseTime(s=-int(basetime2.s), sn=-int(basetime2.sn), sd=int(basetime2.sd))
    return esmf_basetimesum(basetime1, neg2)


__all__ = ["esmf_basetimedifference"]
