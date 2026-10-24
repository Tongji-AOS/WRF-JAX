"""WRF ``fraction_to_string`` integer wrapper translation."""

from __future__ import annotations

from .fraction_to_stringi8 import fraction_to_stringi8


def fraction_to_string(
    numerator: int,
    denominator: int,
    frac_str_len: int | None = None,
) -> str:
    """Return WRF's signed fraction string for default integers."""

    return fraction_to_stringi8(int(numerator), int(denominator), frac_str_len=frac_str_len)


__all__ = ["fraction_to_string"]
