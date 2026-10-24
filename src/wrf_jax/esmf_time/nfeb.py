"""WRF ``nfeb`` translation."""

from __future__ import annotations


def nfeb(year: int) -> int:
    """Return the number of February days for WRF's Gregorian leap-year rule."""

    year = int(year)
    num_days = 28
    if year % 4 == 0:
        num_days = 29
        if year % 100 == 0:
            num_days = 28
            if year % 400 == 0:
                num_days = 29
    return num_days
