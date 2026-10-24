"""WRF ``ndaysinyear`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.nfeb import nfeb


def ndaysinyear(year: int) -> int:
    """Return the number of days in ``year`` using WRF's Gregorian rule."""

    return 366 if nfeb(int(year)) == 29 else 365


__all__ = ["ndaysinyear"]
