"""WRF ``ESMF_BaseTimeSum`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.compute_lcd import compute_lcd
from wrf_jax.esmf_time.normalize_basetime import normalize_basetime
from wrf_jax.esmf_time.types import BaseTime


def esmf_basetimesum(basetime1: BaseTime, basetime2: BaseTime) -> BaseTime:
    """Return the normalized WRF sum of two ``ESMF_BaseTime`` values."""

    s = int(basetime1.s) + int(basetime2.s)
    sn1 = int(basetime1.sn)
    sd1 = int(basetime1.sd)
    sn2 = int(basetime2.sn)
    sd2 = int(basetime2.sd)

    if sd1 == 0 and sd2 == 0:
        sn = 0
        sd = 0
    elif sd1 != 0 and sd2 == 0:
        sn = sn1
        sd = sd1
    elif sd1 == 0 and sd2 != 0:
        sn = sn2
        sd = sd2
    else:
        sd = compute_lcd(sd1, sd2)
        sn = (sn1 * sd // sd1) + (sn2 * sd // sd2)

    return normalize_basetime(BaseTime(s=s, sn=sn, sd=sd))


__all__ = ["esmf_basetimesum"]
