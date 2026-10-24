"""WRF ``ESMF_TimeIntervalSet`` translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence

from wrf_jax.esmf_time.esmf_calendarcreate import ESMF_FAILURE, ESMF_SUCCESS
from wrf_jax.esmf_time.nfeb import nfeb
from wrf_jax.esmf_time.normalize_timeint import normalize_timeint
from wrf_jax.esmf_time.types import BaseTime, ESMFTimeInterval

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class TimeIntervalSetResult:
    """Result of WRF ``ESMF_TimeIntervalSet``."""

    timeinterval: ESMFTimeInterval
    rc: int = ESMF_SUCCESS


def _set_holder(holder: MutableSequence[int], value: int, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def esmf_timeintervalset(
    *,
    yy: int | None = None,
    yyl: int | None = None,
    mm: int | None = None,
    mol: int | None = None,
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
    rc: MutableSequence[int] | None = None,
) -> TimeIntervalSetResult:
    """Initialize a WRF ``ESMF_TimeInterval`` from the optional unit set.

    This follows the WRF 4.5 implementation literally. The Fortran interface
    declares large-integer, month, microsecond, nanosecond, and floating-point
    optional arguments, but this WRF routine does not use them.
    """

    if rc is not None:
        _set_holder(rc, ESMF_FAILURE, "rc")

    # Mark intentionally unused source-interface arguments so accidental
    # behavior is not inferred from their names.
    _ = (yyl, mm, mol, dl, sl, us, ns, d_, h_, m_, s_, ms_, us_, ns_)

    yr = int(yy) if yy is not None else 0
    seconds = 0

    if nfeb(2004) == 28:
        seconds += 365 * yr * SECONDS_PER_DAY
        yr = 0

    if d is not None:
        seconds += SECONDS_PER_DAY * int(d)
    if h is not None:
        seconds += 3600 * int(h)
    if m is not None:
        seconds += 60 * int(m)
    if s is not None:
        seconds += int(s)

    if sn is not None and sd is None:
        raise ValueError("ESMF_TimeIntervalSet: Must specify Sd if Sn is specified")
    if sd is not None and ms is not None:
        raise ValueError("ESMF_TimeIntervalSet: Must not specify both Sd and MS")

    frac_numerator = 0
    frac_denominator = 0
    if ms is not None:
        frac_numerator = int(ms)
        frac_denominator = 1000
    elif sd is not None:
        frac_denominator = int(sd)
        if sn is not None:
            frac_numerator = int(sn)

    timeinterval = normalize_timeint(
        ESMFTimeInterval(
            basetime=BaseTime(s=seconds, sn=frac_numerator, sd=frac_denominator),
            yr=yr,
        )
    )

    if rc is not None:
        _set_holder(rc, ESMF_SUCCESS, "rc")

    return TimeIntervalSetResult(timeinterval=timeinterval, rc=ESMF_SUCCESS)


__all__ = [
    "SECONDS_PER_DAY",
    "TimeIntervalSetResult",
    "esmf_timeintervalset",
]
