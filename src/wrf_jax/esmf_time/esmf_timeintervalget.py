"""WRF ``ESMF_TimeIntervalGet`` translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence

from wrf_jax.esmf_time.esmfold_timeintervalgetstring import (
    ESMF_SUCCESS,
    SECONDS_PER_DAY,
    _fortran_mod,
    _trunc_div,
    esmfold_timeintervalgetstring,
)
from wrf_jax.esmf_time.types import ESMFTimeInterval


@dataclass(frozen=True)
class TimeIntervalGetResult:
    """Values requested from ``ESMF_TimeIntervalGet``."""

    d: int | None = None
    d_r8: float | None = None
    s: int | None = None
    s_i8: int | None = None
    sn: int | None = None
    sd: int | None = None
    time_string: str | None = None
    rc: int = ESMF_SUCCESS


def _set_holder(holder: MutableSequence, value, name: str) -> None:
    if len(holder) < 1:
        raise ValueError(f"{name} output holder must contain at least one element")
    holder[0] = value


def esmf_timeintervalget(
    timeinterval: ESMFTimeInterval,
    *,
    d: MutableSequence[int] | None = None,
    d_r8: MutableSequence[float] | None = None,
    s: MutableSequence[int] | None = None,
    s_i8: MutableSequence[int] | None = None,
    sn: MutableSequence[int] | None = None,
    sd: MutableSequence[int] | None = None,
    time_string: MutableSequence[str] | None = None,
    time_string_len: int | None = None,
    rc: MutableSequence[int] | None = None,
) -> TimeIntervalGetResult:
    """Get selected components from a WRF ``ESMF_TimeInterval``.

    The Fortran routine writes only optional arguments that are present. If
    ``D`` or ``d_r8`` is present, ``S``/``S_i8`` holders are overwritten with
    the remainder seconds inside the day.
    """

    seconds = int(timeinterval.basetime.s)
    sn_value = int(timeinterval.basetime.sn)
    sd_value = int(timeinterval.basetime.sd)

    result_s = seconds if s is not None else None
    result_s_i8 = seconds if s_i8 is not None else None

    result_d = None
    if d is not None:
        result_d = _trunc_div(seconds, SECONDS_PER_DAY)
        if s is not None:
            result_s = _fortran_mod(seconds, SECONDS_PER_DAY)
        if s_i8 is not None:
            result_s_i8 = _fortran_mod(seconds, SECONDS_PER_DAY)

    result_d_r8 = None
    if d_r8 is not None:
        result_d_r8 = float(seconds) / float(SECONDS_PER_DAY)
        if s is not None:
            result_s = _fortran_mod(seconds, SECONDS_PER_DAY)
        if s_i8 is not None:
            result_s_i8 = _fortran_mod(seconds, SECONDS_PER_DAY)

    result_sn = sn_value if sn is not None else None
    result_sd = sd_value if sd is not None else None
    result_time_string = (
        esmfold_timeintervalgetstring(
            timeinterval, timestring_len=time_string_len
        )
        if time_string is not None
        else None
    )

    if s is not None:
        _set_holder(s, result_s, "S")
    if s_i8 is not None:
        _set_holder(s_i8, result_s_i8, "S_i8")
    if d is not None:
        _set_holder(d, result_d, "D")
    if d_r8 is not None:
        _set_holder(d_r8, result_d_r8, "d_r8")
    if sn is not None:
        _set_holder(sn, result_sn, "Sn")
    if sd is not None:
        _set_holder(sd, result_sd, "Sd")
    if time_string is not None:
        _set_holder(time_string, result_time_string, "TimeString")
    if rc is not None:
        _set_holder(rc, ESMF_SUCCESS, "rc")

    return TimeIntervalGetResult(
        d=result_d,
        d_r8=result_d_r8,
        s=result_s,
        s_i8=result_s_i8,
        sn=result_sn,
        sd=result_sd,
        time_string=result_time_string,
        rc=ESMF_SUCCESS,
    )


__all__ = ["TimeIntervalGetResult", "esmf_timeintervalget"]
