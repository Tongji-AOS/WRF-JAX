"""WRF ``compute_lcd`` translation."""

from __future__ import annotations

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)


def compute_lcd(e1: int, e2: int) -> int:
    """Return WRF's limited-prime common denominator for ``e1`` and ``e2``."""

    d1 = int(e1)
    d2 = int(e2)
    if d1 == 0 and d2 == 0:
        return 1
    if d1 == 0:
        d1 = d2
    if d2 == 0:
        d2 = d1
    if d1 == d2:
        return d1

    lcd = d1 * d2
    for p in _PRIMES:
        while lcd // p != 0 and (lcd // p) % d1 == 0 and (lcd // p) % d2 == 0:
            lcd //= p
    return lcd
