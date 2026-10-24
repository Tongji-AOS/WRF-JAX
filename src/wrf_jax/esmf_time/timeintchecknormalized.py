"""WRF ``timeintchecknormalized`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.types import ESMFTimeInterval


def timeintchecknormalized(timeint: ESMFTimeInterval, msgstr: str) -> None:
    """Raise if a WRF ``ESMF_TimeInterval`` has a nonzero year component."""

    if timeint.yr != 0:
        raise ValueError(f"un-normalized TimeInterval not allowed:  {msgstr.rstrip()}")
