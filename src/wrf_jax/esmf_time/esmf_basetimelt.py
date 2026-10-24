"""WRF ``ESMF_BaseTimeLT`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.seccmp import seccmp
from wrf_jax.esmf_time.types import BaseTime


def esmf_basetimelt(basetime1: BaseTime, basetime2: BaseTime) -> bool:
    """Return WRF ``basetime1 .LT. basetime2``."""

    return (
        seccmp(
            basetime1.s,
            basetime1.sn,
            basetime1.sd,
            basetime2.s,
            basetime2.sn,
            basetime2.sd,
        )
        < 0
    )


__all__ = ["esmf_basetimelt"]
