"""Compatibility helpers from WRF ``share/module_date_time``."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from wrf_jax.esmf_time.esmf_calendarcreate import ESMF_SUCCESS
from wrf_jax.esmf_time.esmf_timeget import esmf_timeget
from wrf_jax.esmf_time.esmf_timeset import esmf_timeset
from wrf_jax.esmf_time.types import ESMFTime


def _fixed_int(value: str, start: int, end: int, name: str) -> int:
    try:
        return int(value[start:end])
    except ValueError as exc:
        raise ValueError(f"wrf_atotime could not parse {name} from {value!r}") from exc


def _fortran_nint_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def wrf_atotime(str_value: str) -> ESMFTime:
    """Parse WRF date text into ``WRFU_Time``/``ESMFTime`` semantics."""

    text = str(str_value)
    year = _fixed_int(text, 0, 4, "year")
    month = _fixed_int(text, 5, 7, "month")
    day = _fixed_int(text, 8, 10, "day")
    hour = _fixed_int(text, 11, 13, "hour")
    minute = _fixed_int(text, 14, 16, "minute")
    second = _fixed_int(text, 17, 19, "second")
    millisecond = 0

    if len(text) >= 20 and text[19] == ".":
        ten_thousandth = _fixed_int(text, 20, 24, "ten-thousandths")
        millisecond = _fortran_nint_positive(float(ten_thousandth) / 10.0)

    rc = [-1]
    result = esmf_timeset(
        yy=year,
        mm=month,
        dd=day,
        h=hour,
        m=minute,
        s=second,
        ms=millisecond,
        rc=rc,
    )
    if rc[0] != ESMF_SUCCESS:
        raise RuntimeError("WRFU_TimeSet() in wrf_atotime() FAILED")
    return result.time


def wrf_timetoa(
    time: ESMFTime,
    *,
    str_len: int | None = None,
    wrf_debug_fn: Callable[[int, str], Any] | None = None,
) -> str:
    """Format WRF ``WRFU_Time`` as ``YYYY-MM-DD_hh:mm:ss`` text."""

    if str_len is not None and int(str_len) < 19:
        raise ValueError("wrf_timetoa:  str is too short")

    tmp_holder = [""]
    rc = [-1]
    esmf_timeget(time, time_string=tmp_holder, time_string_len=256, rc=rc)
    if rc[0] != ESMF_SUCCESS:
        raise RuntimeError(
            f"WRFU_TimeGet() returns {rc[0]} in wrf_timetoa() FAILED: "
            f"timeString >{tmp_holder[0].rstrip()}<"
        )

    tmpstr = tmp_holder[0].rstrip()
    out_len = len(tmpstr) if str_len is None else min(int(str_len), len(tmpstr))
    out = tmpstr[:out_len]
    if len(out) >= 11:
        out = out[:10] + "_" + out[11:]
    if str_len is not None:
        out = out.ljust(int(str_len))

    if wrf_debug_fn is not None:
        wrf_debug_fn(150, f"DEBUG wrf_timetoa():  returning with str = [{out.rstrip()}]")
    return out


__all__ = ["wrf_atotime", "wrf_timetoa"]
