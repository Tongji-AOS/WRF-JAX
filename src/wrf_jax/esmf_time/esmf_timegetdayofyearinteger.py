"""Translation of ``wrf_esmf_timemod::esmf_timegetdayofyearinteger``."""

from __future__ import annotations

from collections.abc import MutableSequence

from wrf_jax.esmf_time.types import ESMFTime

ESMF_SUCCESS = 0
SECONDS_PER_DAY = 86400


def _div_trunc_toward_zero(numerator: int, denominator: int) -> int:
    """Return integer division with Fortran's truncation semantics."""

    if numerator < 0:
        return -((-numerator) // denominator)
    return numerator // denominator


def esmf_timegetdayofyearinteger(
    time: ESMFTime,
    day_of_year: MutableSequence[int] | None = None,
    rc: MutableSequence[int] | None = None,
) -> int:
    """Return the one-based day of year for ``time`` and update holders."""

    result = _div_trunc_toward_zero(int(time.basetime.s), SECONDS_PER_DAY) + 1

    if day_of_year is not None:
        if len(day_of_year) < 1:
            raise ValueError("day_of_year output holder must contain at least one element")
        day_of_year[0] = result

    if rc is not None:
        if len(rc) < 1:
            raise ValueError("rc holder must contain at least one element")
        rc[0] = ESMF_SUCCESS

    return result
