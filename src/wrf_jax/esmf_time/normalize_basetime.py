"""WRF ``normalize_basetime`` translation."""

from __future__ import annotations

from wrf_jax.esmf_time.types import BaseTime


_SIMPLIFY_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
_EXTENDED_SIMPLIFY_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
)


def _trunc_div(numerator: int, denominator: int) -> int:
    """Integer division with Fortran semantics: truncate toward zero."""

    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        return -quotient
    return quotient


def _fortran_mod(numerator: int, denominator: int) -> int:
    """Fortran ``MOD`` semantics: remainder has the dividend's sign."""

    return numerator - _trunc_div(numerator, denominator) * denominator


def _simplify_with_primes(numerator: int, denominator: int, primes: tuple[int, ...]) -> tuple[int, int]:
    if numerator == 0:
        return 0, 1
    if _fortran_mod(denominator, numerator) == 0:
        return 1, _trunc_div(denominator, numerator)

    n = numerator
    d = denominator
    for prime in primes:
        keepgoing = True
        while keepgoing:
            keepgoing = False
            if (
                _trunc_div(d, prime) != 0
                and _trunc_div(n, prime) != 0
                and _fortran_mod(d, prime) == 0
                and _fortran_mod(n, prime) == 0
            ):
                d = _trunc_div(d, prime)
                n = _trunc_div(n, prime)
                keepgoing = True
    return n, d


def normalize_basetime(basetime: BaseTime) -> BaseTime:
    """Normalize WRF ``ESMF_BaseTime`` seconds and fractional seconds.

    This preserves the Fortran routine's integer division, ``MOD`` sign
    behavior, and prime-list based fraction simplification.
    """

    s = basetime.s
    sn = basetime.sn
    sd = basetime.sd

    if sd < 0:
        raise ValueError("normalize_basetime: denominator of seconds cannot be negative")
    if sd == 0 and sn != 0:
        raise ValueError(
            "normalize_basetime: denominator of seconds cannot be zero when numerator is non-zero"
        )

    if sd > 0:
        if abs(sn) >= sd:
            s += _trunc_div(sn, sd)
            sn = _fortran_mod(sn, sd)

        if s > 0 and sn < 0:
            s -= 1
            sn += sd
        if s < 0 and sn > 0:
            s += 1
            sn -= sd

    if sd > 0 and sn > 0:
        sn, sd = _simplify_with_primes(sn, sd, _SIMPLIFY_PRIMES)
        if sd > 10000 and sn > 10000:
            sn, sd = _simplify_with_primes(sn, sd, _EXTENDED_SIMPLIFY_PRIMES)

    return BaseTime(s=s, sn=sn, sd=sd)
