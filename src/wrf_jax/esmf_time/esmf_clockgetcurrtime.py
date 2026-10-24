"""Translation of ``wrf_esmf_clockmod::esmf_clockgetcurrtime``."""

from __future__ import annotations

from collections.abc import MutableSequence

from wrf_jax.esmf_time.types import ESMFClock, ESMFTime

ESMF_SUCCESS = 0


def esmf_clockgetcurrtime(
    clock: ESMFClock, rc: MutableSequence[int] | None = None
) -> ESMFTime:
    """Return the current time from ``clock`` and optionally set ``rc``."""

    curr_time = clock.clockint.curr_time
    if rc is not None:
        if len(rc) == 0:
            raise ValueError("rc output holder must contain at least one element")
        rc[0] = ESMF_SUCCESS
    return curr_time
