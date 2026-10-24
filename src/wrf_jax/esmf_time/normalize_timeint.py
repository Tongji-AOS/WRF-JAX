"""WRF ``normalize_timeint`` translation."""

from __future__ import annotations

from dataclasses import replace

from wrf_jax.esmf_time.normalize_basetime import normalize_basetime
from wrf_jax.esmf_time.types import ESMFTimeInterval


def normalize_timeint(timeint: ESMFTimeInterval) -> ESMFTimeInterval:
    """Normalize the base-time part of a WRF ``ESMF_TimeInterval``."""

    return replace(timeint, basetime=normalize_basetime(timeint.basetime))


__all__ = ["normalize_timeint"]
