"""Translation of ``wrf_esmf_clockmod::esmf_clockgettimestep``."""

from __future__ import annotations

from typing import MutableSequence

from wrf_jax.esmf_time.types import ESMFClock, ESMFTimeInterval

ESMF_SUCCESS = 0


def esmf_clockgettimestep(
    clock: ESMFClock,
    rc: MutableSequence[int] | None = None,
) -> ESMFTimeInterval:
    """Return ``clock``'s time step, optionally setting WRF's success code."""

    if rc is not None:
        if len(rc) < 1:
            raise ValueError("rc holder must contain at least one element")
        rc[0] = ESMF_SUCCESS

    return clock.clockint.time_step
