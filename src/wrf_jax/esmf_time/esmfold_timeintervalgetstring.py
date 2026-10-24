"""WRF ``ESMFold_TimeIntervalGetString`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.types import ESMFTimeInterval


ESMF_SUCCESS = 0
SECONDS_PER_DAY = 86400


def _trunc_div(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        return -quotient
    return quotient


def _fortran_mod(numerator: int, denominator: int) -> int:
    return numerator - _trunc_div(numerator, denominator) * denominator


def _signnormtimeint(timeinterval: ESMFTimeInterval) -> int:
    seconds = int(timeinterval.basetime.s)
    numerator = int(timeinterval.basetime.sn)
    positive = seconds > 0 or numerator > 0
    negative = seconds < 0 or numerator < 0
    if positive and negative:
        raise RuntimeError("signnormtimeint: signs of fields cannot be mixed")
    if positive:
        return 1
    if negative:
        return -1
    return 0


def _iw(value: int, width: int, min_digits: int) -> str:
    sign = "-" if value < 0 else ""
    digits = str(abs(int(value))).rjust(min_digits, "0")
    result = sign + digits
    if len(result) > width:
        return "*" * width
    return result.rjust(width)


def esmfold_timeintervalgetstring(
    timeinterval: ESMFTimeInterval,
    *,
    timestring_len: int | None = None,
) -> str:
    """Return WRF's string form for an ``ESMF_TimeInterval``."""

    negative = _signnormtimeint(timeinterval) == -1
    seconds = int(timeinterval.basetime.s)
    numerator = int(timeinterval.basetime.sn)
    if negative:
        seconds = -seconds
        numerator = -numerator
    _ = numerator

    sign = "-" if negative else ""
    hours = _trunc_div(_fortran_mod(seconds, SECONDS_PER_DAY), 3600)
    minutes = _trunc_div(_fortran_mod(seconds, 3600), 60)
    secs = _fortran_mod(seconds, 60)
    value = (
        f"{sign}"
        f"{_iw(_trunc_div(seconds, SECONDS_PER_DAY), 10, 10)}_"
        f"{_iw(hours, 3, 3)}:"
        f"{_iw(minutes, 3, 3)}:"
        f"{_iw(secs, 3, 3)}"
    )

    if timestring_len is None:
        return value

    timestring_len = int(timestring_len)
    if timestring_len < 0:
        raise ValueError("timestring_len must be non-negative")
    return value[:timestring_len].ljust(timestring_len)


__all__ = ["ESMF_SUCCESS", "SECONDS_PER_DAY", "esmfold_timeintervalgetstring"]
