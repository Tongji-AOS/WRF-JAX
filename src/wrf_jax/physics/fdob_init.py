"""Translation support for ``module_physics_init::fdob_init``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class FdobInitResult:
    start_year: int
    start_month: int
    start_day: int
    start_hour: int
    start_minute: int
    start_second: int
    fdob: Any
    external_calls: tuple[str, ...] = ()


def fdob_init(
    obs_nudge_opt: Sequence[int],
    maxdom: int,
    inest: int,
    parid: Sequence[int],
    idynin: int,
    dtramp: float,
    fdaend: Sequence[float],
    restart: bool,
    obs_twindo_cg: float,
    obs_twindo: float,
    itimestep: int,
    start_year: int,
    start_month: int,
    start_day: int,
    start_hour: int,
    start_minute: int,
    start_second: int,
    fdob: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    **kwargs: Any,
) -> FdobInitResult:
    """Initialize observation nudging state when enabled."""

    del (
        maxdom,
        parid,
        idynin,
        dtramp,
        fdaend,
        restart,
        obs_twindo_cg,
        obs_twindo,
        itimestep,
        ids,
        ide,
        jds,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
        kwargs,
    )
    if int(inest) < 1 or int(inest) > len(obs_nudge_opt):
        raise IndexError("inest uses Fortran 1-based indexing and must be within obs_nudge_opt")

    external_calls: tuple[str, ...] = ()
    if int(obs_nudge_opt[int(inest) - 1]) != 0:
        del jde
        external_calls = ("fddaobs_init",)

    return FdobInitResult(
        start_year=int(start_year),
        start_month=int(start_month),
        start_day=int(start_day),
        start_hour=int(start_hour),
        start_minute=int(start_minute),
        start_second=int(start_second),
        fdob=fdob,
        external_calls=external_calls,
    )


__all__ = ["FdobInitResult", "fdob_init"]
