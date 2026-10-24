"""WRF ``seccmp`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.compute_lcd import compute_lcd


def seccmp(s1: int, sn1: int, sd1: int, s2: int, sn2: int, sd2: int) -> int:
    """Compare WRF seconds plus fractional seconds.

    Returns ``1`` if the first value is greater, ``-1`` if it is smaller, and
    ``0`` if they compare equal. Inputs are assumed to be normalized the same
    way WRF expects before calling ``seccmp``.
    """

    n1 = int(sn1)
    n2 = int(sn2)
    if n1 != 0 or n2 != 0:
        lcd = compute_lcd(int(sd1), int(sd2))
        if sd1 != 0:
            n1 *= lcd // int(sd1)
        if sd2 != 0:
            n2 *= lcd // int(sd2)

    if s1 > s2:
        return 1
    if s1 < s2:
        return -1
    if n1 > n2:
        return 1
    if n1 < n2:
        return -1
    return 0
