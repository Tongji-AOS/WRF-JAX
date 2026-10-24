"""Translation of ``wrf_esmf_timemod::esmfold_timegetstring``."""

from __future__ import annotations


def _fortran_i_zero(value: int, width: int, minimum_digits: int) -> str:
    """Format an integer like Fortran ``Iw.m`` for the ESMF time string."""

    value = int(value)
    sign = "-" if value < 0 else ""
    digits = str(abs(value)).zfill(minimum_digits)
    field = f"{sign}{digits}"
    if len(field) > width:
        return "*" * width
    return field.rjust(width)


def esmfold_timegetstring(
    year: int,
    month: int,
    dayofmonth: int,
    hour: int,
    minute: int,
    second: int,
    timestring_len: int | None = None,
) -> str:
    """Return WRF ESMF old time string ``YYYY-MM-DD_hh:mm:ss``.

    ``timestring_len`` mirrors a Fortran ``CHARACTER`` output buffer: the
    formatted record is truncated when the buffer is short and blank-padded
    when it is long.
    """

    time_string = (
        f"{_fortran_i_zero(year, 4, 4)}-"
        f"{_fortran_i_zero(month, 2, 2)}-"
        f"{_fortran_i_zero(dayofmonth, 2, 2)}_"
        f"{_fortran_i_zero(hour, 2, 2)}:"
        f"{_fortran_i_zero(minute, 2, 2)}:"
        f"{_fortran_i_zero(second, 2, 2)}"
    )

    if timestring_len is None:
        return time_string

    timestring_len = int(timestring_len)
    if timestring_len < 0:
        raise ValueError("timestring_len must be non-negative")
    return time_string[:timestring_len].ljust(timestring_len)
