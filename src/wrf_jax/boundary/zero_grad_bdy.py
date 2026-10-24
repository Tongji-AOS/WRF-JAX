"""Zero-gradient boundary condition from WRF ``module_bc``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


def zero_grad_bdy(
    field: Any,
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
    """Return ``field`` after WRF zero-gradient boundary copying."""

    del kds, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe, kte

    out = jnp.asarray(field)
    variable = str(variable_in)[0]
    if variable == "U":
        variable = "u"
    if variable == "V":
        variable = "v"

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
    if variable == "w":
        ktf = int(kde)

    periodic_x = bool(_read(config_flags, "periodic_x", False))

    k_idx = jnp.arange(_idx(int(kts), kms), _idx(ktf, kms) + 1)

    def copy_j_edge(j: int, i0: int, i1: int, j_inner: int) -> None:
        nonlocal out
        if i0 <= i1:
            i_wrf = jnp.arange(i0, i1 + 1)
            i_inner_wrf = i_wrf if periodic_x else jnp.clip(i_wrf, ibs + int(spec_zone), ibe - int(spec_zone))
            i_idx = i_wrf - int(ims)
            i_inner_idx = i_inner_wrf - int(ims)
            source = out[i_inner_idx[:, None], k_idx[None, :], _idx(j_inner, jms)]
            out = out.at[i_idx[:, None], k_idx[None, :], _idx(j, jms)].set(source)

    def copy_i_edge(i: int, j0: int, j1: int, i_inner: int) -> None:
        nonlocal out
        if j0 <= j1:
            j_wrf = jnp.arange(j0, j1 + 1)
            j_inner_wrf = jnp.clip(j_wrf, jbs + int(spec_zone), jbe - int(spec_zone))
            j_idx = j_wrf - int(jms)
            j_inner_idx = j_inner_wrf - int(jms)
            source = out[_idx(i_inner, ims), k_idx[:, None], j_inner_idx[None, :]]
            out = out.at[_idx(i, ims), k_idx[:, None], j_idx[None, :]].set(source)

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            b_dist = j - jbs
            b_limit = 0 if periodic_x else b_dist
            copy_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit), jbs + int(spec_zone))

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            b_dist = jbe - j
            b_limit = 0 if periodic_x else b_dist
            copy_j_edge(j, max(int(its), b_limit + ibs), min(itf, ibe - b_limit), jbe - int(spec_zone))

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                b_dist = i - ibs
                copy_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1), ibs + int(spec_zone))

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                b_dist = ibe - i
                copy_i_edge(i, max(int(jts), b_dist + jbs + 1), min(jtf, jbe - b_dist - 1), ibe - int(spec_zone))

    return out


__all__ = ["zero_grad_bdy"]
