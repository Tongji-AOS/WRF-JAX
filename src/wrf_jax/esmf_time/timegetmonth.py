"""WRF ``timegetmonth`` translation."""

from __future__ import annotations

from typing import MutableSequence

from wrf_jax.esmf_time.calendar_tables import MONTHBDYS, MONTHBDYS_LEAP, MONTHS_PER_YEAR
from wrf_jax.esmf_time.esmf_basetimege import esmf_basetimege
from wrf_jax.esmf_time.esmf_basetimelt import esmf_basetimelt
from wrf_jax.esmf_time.nfeb import nfeb
from wrf_jax.esmf_time.types import ESMFTime


def _set_holder(holder: MutableSequence[int], value: int, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def timegetmonth(time: ESMFTime, mm: MutableSequence[int] | None = None) -> int:
    """Extract WRF month-of-year from a normalized-in-year ``ESMF_Time``."""

    boundaries = MONTHBDYS_LEAP if nfeb(time.yr) == 29 else MONTHBDYS
    month = -1
    for index in range(1, MONTHS_PER_YEAR + 1):
        if esmf_basetimege(time.basetime, boundaries[index - 1]) and esmf_basetimelt(
            time.basetime,
            boundaries[index],
        ):
            month = index
            break

    if month == -1:
        raise ValueError("timegetmonth: could not extract month of year from time")

    if mm is not None:
        _set_holder(mm, month, "MM")
    return month


__all__ = ["timegetmonth"]
