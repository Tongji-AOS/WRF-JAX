"""WRF ``fraction_to_stringi8`` translation."""

from __future__ import annotations


def _i2_2(value: int) -> str:
    text = str(abs(int(value))).rjust(2, "0")
    if len(text) > 2:
        return "**"
    return text


def fraction_to_stringi8(
    numerator: int,
    denominator: int,
    frac_str_len: int | None = None,
) -> str:
    """Return WRF's signed fraction string for integer*8 values."""

    numerator = int(numerator)
    denominator = int(denominator)
    if denominator > 0 and numerator % denominator != 0:
        sign = "+" if numerator > 0 else "-"
        value = f"{sign}{_i2_2(numerator)}/{_i2_2(denominator)}"
    else:
        value = ""

    if frac_str_len is None:
        return value

    frac_str_len = int(frac_str_len)
    if frac_str_len < 0:
        raise ValueError("frac_str_len must be non-negative")
    return value[:frac_str_len].ljust(frac_str_len)


__all__ = ["fraction_to_stringi8"]
