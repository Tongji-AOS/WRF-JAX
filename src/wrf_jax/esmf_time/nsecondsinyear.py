"""WRF ``nsecondsinyear`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.calendar_tables import SECONDS_PER_DAY
from wrf_jax.esmf_time.ndaysinyear import ndaysinyear


def nsecondsinyear(year: int) -> int:
    """Return the number of seconds in ``year`` using WRF's calendar rule."""

    return SECONDS_PER_DAY * int(ndaysinyear(int(year)))


__all__ = ["nsecondsinyear"]
