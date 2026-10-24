"""Translations from WRF ``module_ra_clWRF_support``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path

CYR = 1800
SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class CamGasesState:
    """Explicit state for WRF ``read_CAMgases`` SAVE arrays."""

    yrdata: tuple[int, ...]
    juldata: tuple[float, ...]
    co2r: tuple[float, ...]
    n2or: tuple[float, ...]
    ch4r: tuple[float, ...]
    cfc11r: tuple[float, ...]
    cfc12r: tuple[float, ...]
    max_years: int


@dataclass(frozen=True)
class CamGasesResult:
    """Trace-gas volume mixing ratios returned by ``read_CAMgases``."""

    co2vmr: float
    n2ovmr: float
    ch4vmr: float
    cfc11vmr: float
    cfc12vmr: float
    state: CamGasesState


def _is_leap_year(year: int) -> bool:
    my1 = year % 4
    my2 = year % 100
    my3 = year % 400
    return (my1 == 0 and my2 != 0) or my3 == 0


def _midmonth_julian(year: int, month: int) -> float:
    monday = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if _is_leap_year(year):
        monday[2] = 29
    # Fortran uses one-based monday and mondata=6 for file data.
    return float(sum(monday[:month]) + float(monday[month]) / 2.0 - 0.5)


def load_camgases_table(path: str | Path) -> CamGasesState:
    """Reuse parsed SAVE arrays without hiding edits to the input table."""

    return _parse_camgases_table(Path(path).read_text(encoding="ascii"))


@lru_cache(maxsize=32)
def _parse_camgases_table(contents: str) -> CamGasesState:
    # Content keys also work on filesystems with coarse or virtual timestamps.
    rows: list[tuple[int, float, float, float, float, float]] = []
    with StringIO(contents) as handle:
        next(handle, None)
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            )

    if len(rows) < 2:
        raise ValueError("CAMtr_volume_mixing_ratio must contain at least two data rows")
    if len(rows) > CYR:
        raise ValueError("CAMtr_volume_mixing_ratio has more rows than WRF cyr")

    yrdata = tuple(row[0] for row in rows)
    juldata = tuple(_midmonth_julian(year, 6) for year in yrdata)
    return CamGasesState(
        yrdata=yrdata,
        juldata=juldata,
        co2r=tuple(row[1] for row in rows),
        n2or=tuple(row[2] for row in rows),
        ch4r=tuple(row[3] for row in rows),
        cfc11r=tuple(row[4] for row in rows),
        cfc12r=tuple(row[5] for row in rows),
        max_years=len(rows),
    )


def _orig_val(tracer: str, model: str) -> float:
    if model == "CAM":
        values = {
            "CO2": 3.55e-4,
            "N2O": 0.311e-6,
            "CH4": 1.714e-6,
            "CFC11": 0.280e-9,
            "CFC12": 0.503e-9,
        }
    elif model in {"RRTM", "RRTMG"}:
        values = {
            "CO2": 379.0e-6,
            "N2O": 319.0e-9,
            "CH4": 1774.0e-9,
            "CFC11": 0.251e-9,
            "CFC12": 0.538e-9,
        }
    else:
        raise ValueError(f"CLWRF not implemented for the {model} radiative scheme.")
    try:
        return values[tracer]
    except KeyError as exc:
        raise ValueError(f"CLWRF: Trace gas {tracer} not valid for scheme {model}") from exc


def _find_year_in(yr: int, julian: float, state: CamGasesState) -> tuple[int, bool]:
    for index, (data_year, data_julian) in enumerate(
        zip(state.yrdata, state.juldata),
        start=1,
    ):
        if data_year > yr:
            if index + 1 >= state.max_years:
                return state.max_years - 1, True
            return index, True
        if data_year == yr and data_julian > julian:
            if index + 1 >= state.max_years:
                return state.max_years - 1, True
            return index, True

    if state.max_years >= 2:
        return state.max_years - 1, True
    return 0, False


def _valid_years(year_in: int, gas: tuple[float, ...], tot_years: int) -> tuple[int, int]:
    yr_loc = year_in
    yr2 = yr_loc
    yr1 = yr_loc - 1

    if sum(1 for value in gas[: yr_loc - 1] if value > 0.0) == 0:
        for idata in range(yr_loc, tot_years):
            if gas[idata - 1] > 0.0:
                yr1 = idata
                break
        for idata in range(yr1 + 1, tot_years + 1):
            if gas[idata - 1] > 0.0:
                yr2 = idata
                break
    else:
        if gas[yr_loc - 1] < 0.0:
            if any(value > 0.0 for value in gas[yr_loc - 1 : tot_years]):
                for idata in range(yr_loc + 1, tot_years + 1):
                    if gas[idata - 1] > 0.0:
                        yr2 = idata
                        break
            else:
                for idata in range(yr_loc - 1, 1, -1):
                    if gas[idata - 1] > 0.0:
                        yr2 = idata
                        break

        yr_loc = min(yr_loc - 1, yr2 - 1)
        if gas[yr_loc - 1] < 0.0:
            if any(value > 0.0 for value in gas[: yr_loc - 1]):
                for idata in range(yr_loc - 1, 0, -1):
                    if gas[idata - 1] > 0.0:
                        yr1 = idata
                        break
        else:
            yr1 = yr_loc

    return yr1, yr2


def _interpolate_camgases(
    yr: int,
    julian: float,
    yeari: int,
    juli: int,
    yr1: int,
    yr2: int,
    yearf: int,
    julf: int,
    gas: tuple[float, ...],
) -> float:
    minyear = min(yr, yeari)
    maxyear = max(yr, yearf)

    fact2 = float(juli)
    fact1 = float(julf)
    x = float(julian)

    for ny in range(minyear, maxyear):
        nday = 366 if _is_leap_year(ny) else 365
        if ny < yeari:
            fact2 += nday
        if ny < yr:
            x += nday
        if ny < yearf:
            fact1 += nday

    deltat = fact1 - fact2
    fact1_weight = (fact1 - x) / deltat
    fact2_weight = (x - fact2) / deltat
    interp_gas = gas[yr1 - 1] * fact1_weight + gas[yr2 - 1] * fact2_weight
    if interp_gas < 0.0:
        return -99999.0
    return float(interp_gas)


def read_camgases(
    yr: int,
    julian: float,
    readtrfile: bool,
    model: str,
    *,
    path: str | Path = "CAMtr_volume_mixing_ratio",
    state: CamGasesState | None = None,
) -> CamGasesResult:
    """Read/interpolate WRF CAM greenhouse gas values.

    The Fortran routine stores its table in ``SAVE`` arrays. The Python
    translation returns the table state explicitly so repeated calls can pass
    it back with ``readtrfile=False``.
    """

    if readtrfile:
        table_path = Path(path)
        if table_path.exists():
            state = load_camgases_table(table_path)
        elif model == "CAM":
            raise FileNotFoundError(
                "CAM built-in CO2-only fallback is not sufficient for all gases; "
                "provide CAMtr_volume_mixing_ratio"
            )
        else:
            raise FileNotFoundError("CLWRF: 'CAMtr_volume_mixing_ratio' does not exist")
    elif state is None:
        raise ValueError("read_camgases requires explicit state when readtrfile is false")

    assert state is not None
    year_in, found_year = _find_year_in(int(yr), float(julian), state)
    if year_in == 1:
        year_in += 1

    co2vmr = n2ovmr = ch4vmr = cfc11vmr = cfc12vmr = -9999.999

    if found_year:
        total_valid = sum(1 for value in state.co2r[: state.max_years] if value > 0.0)
        if total_valid >= 2:
            outputs = []
            for gas in (state.co2r, state.n2or, state.ch4r, state.cfc11r, state.cfc12r):
                yr1, yr2 = _valid_years(year_in, gas, state.max_years)
                nyrm = state.yrdata[yr1 - 1]
                njulm = int(state.juldata[yr1 - 1])
                nyrp = state.yrdata[yr2 - 1]
                njulp = int(state.juldata[yr2 - 1])
                outputs.append(
                    _interpolate_camgases(
                        int(yr),
                        float(julian),
                        nyrm,
                        njulm,
                        yr1,
                        yr2,
                        nyrp,
                        njulp,
                        gas,
                    )
                )
            co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr = outputs

    if (
        co2vmr < 0.0
        or n2ovmr < 0.0
        or ch4vmr < 0.0
        or cfc11vmr < 0.0
        or cfc12vmr < 0.0
        or not found_year
    ):
        co2vmr = _orig_val("CO2", model)
        n2ovmr = _orig_val("N2O", model)
        ch4vmr = _orig_val("CH4", model)
        cfc11vmr = _orig_val("CFC11", model)
        cfc12vmr = _orig_val("CFC12", model)
    else:
        if co2vmr < 270.0:
            co2vmr = 270.0
        if n2ovmr < 270.0:
            n2ovmr = 270.0
        if ch4vmr < 700.0:
            ch4vmr = 700.0
        co2vmr *= 1.0e-6
        n2ovmr *= 1.0e-9
        ch4vmr *= 1.0e-9
        cfc11vmr *= 1.0e-12
        cfc12vmr *= 1.0e-12

    return CamGasesResult(
        co2vmr=co2vmr,
        n2ovmr=n2ovmr,
        ch4vmr=ch4vmr,
        cfc11vmr=cfc11vmr,
        cfc12vmr=cfc12vmr,
        state=state,
    )


__all__ = ["CamGasesResult", "CamGasesState", "load_camgases_table", "read_camgases"]
