"""Final specified-boundary field update from WRF ``module_bc``."""

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


def _range_values(start: int, stop: int) -> Any:
    if int(start) > int(stop):
        return None
    return jnp.arange(int(start), int(stop) + 1)


def spec_bdy_final(
    field: Any,
    mu: Any,
    c1: Any,
    c2: Any,
    msf: Any,
    field_bdy_xs: Any,
    field_bdy_xe: Any,
    field_bdy_ys: Any,
    field_bdy_ye: Any,
    field_bdy_tend_xs: Any,
    field_bdy_tend_xe: Any,
    field_bdy_tend_ys: Any,
    field_bdy_tend_ye: Any,
    variable_in: str,
    config_flags: Any,
    spec_bdy_width: int,
    spec_zone: int,
    dtbc: float,
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
    """Return ``field`` after applying WRF's final specified-boundary values."""

    del spec_bdy_width, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe, kte

    out = jnp.asarray(field)
    mu_arr = jnp.asarray(mu)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    msf_arr = jnp.asarray(msf)
    bdy_xs = jnp.asarray(field_bdy_xs)
    bdy_xe = jnp.asarray(field_bdy_xe)
    bdy_ys = jnp.asarray(field_bdy_ys)
    bdy_ye = jnp.asarray(field_bdy_ye)
    bdy_tend_xs = jnp.asarray(field_bdy_tend_xs)
    bdy_tend_xe = jnp.asarray(field_bdy_tend_xe)
    bdy_tend_ys = jnp.asarray(field_bdy_tend_ys)
    bdy_tend_ye = jnp.asarray(field_bdy_tend_ye)

    variable = str(variable_in)[0]
    if variable == "U":
        variable = "u"
    if variable == "V":
        variable = "v"
    if variable == "W":
        variable = "w"
    if variable == "M":
        variable = "m"
    if variable == "T":
        variable = "t"
    if variable == "H":
        variable = "h"

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
    if variable in {"m", "h", "w"}:
        ktf = int(kde)

    msfcouple = variable in {"u", "v", "w"}
    mucouple = variable != "m"
    periodic_x = bool(_read(config_flags, "periodic_x", False))
    dtbc_value = jnp.asarray(dtbc, dtype=out.dtype)

    k_vals = _range_values(int(kts), ktf)
    if k_vals is None:
        return out
    kk_field = k_vals - int(kms)
    kk_bdy = k_vals - int(kds)
    c1_k = c1_arr[kk_field]
    c2_k = c2_arr[kk_field]

    def apply_y_boundary(j: int, b_dist: int, bdy: Any, tend: Any) -> None:
        nonlocal out
        b_limit = 0 if periodic_x else int(b_dist)
        i_vals = _range_values(max(int(its), b_limit + ibs), min(itf, ibe - b_limit))
        if i_vals is None:
            return
        ii = i_vals - int(ims)
        jj = _idx(j, jms)
        b_index = int(b_dist)
        bfield = bdy[ii[:, None], kk_bdy[None, :], b_index] + dtbc_value * tend[ii[:, None], kk_bdy[None, :], b_index]
        values = bfield
        if msfcouple:
            values = values * msf_arr[ii, jj][:, None]
        if mucouple:
            xmu = c1_k[None, :] * mu_arr[ii, jj][:, None] + c2_k[None, :]
            values = values / xmu
        out = out.at[ii[:, None], kk_field[None, :], jj].set(values)

    def apply_x_boundary(i: int, b_dist: int, bdy: Any, tend: Any) -> None:
        nonlocal out
        j_vals = _range_values(max(int(jts), int(b_dist) + jbs + 1), min(jtf, jbe - int(b_dist) - 1))
        if j_vals is None:
            return
        ii = _idx(i, ims)
        jj = j_vals - int(jms)
        b_index = int(b_dist)
        bfield = bdy[jj[:, None], kk_bdy[None, :], b_index] + dtbc_value * tend[jj[:, None], kk_bdy[None, :], b_index]
        values = jnp.transpose(bfield, (1, 0))
        if msfcouple:
            values = values * msf_arr[ii, jj][None, :]
        if mucouple:
            xmu = c1_k[:, None] * mu_arr[ii, jj][None, :] + c2_k[:, None]
            values = values / xmu
        out = out.at[ii, kk_field[:, None], jj[None, :]].set(values)

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            b_dist = j - jbs
            apply_y_boundary(j, b_dist, bdy_ys, bdy_tend_ys)

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            b_dist = jbe - j
            apply_y_boundary(j, b_dist, bdy_ye, bdy_tend_ye)

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                b_dist = i - ibs
                apply_x_boundary(i, b_dist, bdy_xs, bdy_tend_xs)

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                b_dist = ibe - i
                apply_x_boundary(i, b_dist, bdy_xe, bdy_tend_xe)

    return out


__all__ = ["spec_bdy_final"]
