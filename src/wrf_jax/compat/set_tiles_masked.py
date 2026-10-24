"""Translation of WRF ``module_tiles::set_tiles_masked``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TilesMaskedResult:
    num_tiles: int
    istarts: tuple[int, ...]
    iends: tuple[int, ...]
    jstarts: tuple[int, ...]
    jends: tuple[int, ...]


def set_tiles_masked(
    imask: Any,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
) -> TilesMaskedResult:
    """Find rectangular one-valued tiles in a Fortran-indexed mask."""

    del jme
    mask = np.asarray(imask).copy()
    ims = int(ims)
    ime = int(ime)
    jms = int(jms)
    ips = int(ips)
    ipe = int(ipe)
    jps = int(jps)
    jpe = int(jpe)
    if mask.shape[0] < ime - ims + 1:
        raise ValueError("imask first dimension is smaller than ims:ime")

    def get(i: int, j: int) -> int:
        return int(mask[i - ims, j - jms])

    def set_zero(i: int, j: int) -> None:
        mask[i - ims, j - jms] = 0

    istarts: list[int] = []
    iends: list[int] = []
    jstarts: list[int] = []
    jends: list[int] = []

    def any_active() -> bool:
        return bool(np.any(mask[ips - ims : ipe - ims + 1, jps - jms : jpe - jms + 1] == 1))

    while any_active():
        for j in range(jps, jpe + 1):
            for i in range(ips, ipe + 1):
                if get(i, j) != 1:
                    continue

                istart = i
                iend = i
                jstart = j
                jend = j
                set_zero(i, j)

                for ir in range(istart + 1, ipe + 1):
                    if get(ir, j) == 1:
                        iend = ir
                        set_zero(ir, j)
                    else:
                        break

                for jr in range(jstart + 1, jpe + 1):
                    if all(get(ir, jr) == 1 for ir in range(istart, iend + 1)):
                        jend = jr
                        for ir in range(istart, iend + 1):
                            set_zero(ir, jr)
                    else:
                        break

                istarts.append(istart)
                iends.append(iend)
                jstarts.append(jstart)
                jends.append(jend)

    return TilesMaskedResult(
        num_tiles=len(istarts),
        istarts=tuple(istarts),
        iends=tuple(iends),
        jstarts=tuple(jstarts),
        jends=tuple(jends),
    )


__all__ = ["TilesMaskedResult", "set_tiles_masked"]
