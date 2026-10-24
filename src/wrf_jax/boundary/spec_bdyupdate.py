"""Specified-boundary tendency application from WRF ``module_bc``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


@lru_cache(maxsize=128)
def _boundary_update_mask(
    shape: tuple[int, int, int],
    variable: str,
    periodic_x: bool,
    spec_zone: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    mask = np.zeros(shape, dtype=np.bool_)

    ibs = int(ids)
    ibe = int(ide) - 1
    itf = min(int(ite), int(ide) - 1)
    jbs = int(jds)
    jbe = int(jde) - 1
    jtf = min(int(jte), int(jde) - 1)
    ktf = int(kde) - 1
    if variable == "u":
        ibe = int(ide)
        itf = min(int(ite), int(ide))
    if variable == "v":
        jbe = int(jde)
        jtf = min(int(jte), int(jde))
    if variable in {"m", "h"}:
        ktf = int(kte)

    k_slice = slice(_idx(int(kts), kms), _idx(ktf, kms) + 1)

    def i_slice(i0: int, i1: int) -> slice:
        return slice(_idx(i0, ims), _idx(i1, ims) + 1)

    def j_slice(j0: int, j1: int) -> slice:
        return slice(_idx(j0, jms), _idx(j1, jms) + 1)

    def mark_j_edge(j: int, i0: int, i1: int) -> None:
        if i0 <= i1:
            mask[i_slice(i0, i1), k_slice, _idx(j, jms)] = True

    def mark_i_edge(i: int, j0: int, j1: int) -> None:
        if j0 <= j1:
            mask[_idx(i, ims), k_slice, j_slice(j0, j1)] = True

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            b_dist = j - jbs
            b_limit = 0 if periodic_x else b_dist
            mark_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit))

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            b_dist = jbe - j
            b_limit = 0 if periodic_x else b_dist
            mark_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit))

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                b_dist = i - ibs
                mark_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1))

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                b_dist = ibe - i
                mark_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1))

    return mask


def spec_bdyupdate(
    field: Any,
    field_tend: Any,
    dt: float,
    variable_in: str,
    config_flags: Any,
    spec_zone: int,
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
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    kps: int,
    kpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Return ``field`` with WRF specified-boundary tendencies applied."""

    del kds, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe

    out = jnp.asarray(field)
    tend = jnp.asarray(field_tend)

    variable = str(variable_in)[0]
    if variable == "U":
        variable = "u"
    if variable == "V":
        variable = "v"
    if variable == "M":
        variable = "m"
    if variable == "H":
        variable = "h"

    periodic_x = bool(_read(config_flags, "periodic_x", False))
    dt_value = float(dt)
    mask = jnp.asarray(
        _boundary_update_mask(
            tuple(int(dim) for dim in out.shape),
            variable,
            periodic_x,
            int(spec_zone),
            int(ids),
            int(ide),
            int(jds),
            int(jde),
            int(kde),
            int(ims),
            int(jms),
            int(kms),
            int(its),
            int(ite),
            int(jts),
            int(jte),
            int(kts),
            int(kte),
        )
    )
    return jnp.where(mask, out + dt_value * tend, out)


__all__ = ["spec_bdyupdate"]
