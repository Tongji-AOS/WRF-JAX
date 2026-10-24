"""Flow-dependent specified boundary update from WRF ``module_bc``."""

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


def flow_dep_bdy(
    field: Any,
    u: Any,
    v: Any,
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
    """Return ``field`` after WRF's inflow-zero/outflow-copy boundary rule.

    ``field`` is unstaggered. ``u`` and ``v`` are used only for their sign,
    exactly as in WRF's ``flow_dep_bdy``.
    """

    del kds, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe, kte

    out = jnp.asarray(field)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)

    ibs = int(ids)
    ibe = int(ide) - 1
    itf = min(int(ite), int(ide) - 1)
    jbs = int(jds)
    jbe = int(jde) - 1
    jtf = min(int(jte), int(jde) - 1)
    ktf = int(kde) - 1
    periodic_x = bool(_read(config_flags, "periodic_x", False))

    k_vals = _range_values(int(kts), ktf)
    if k_vals is None:
        return out
    kk = k_vals - int(kms)

    def apply_y_boundary(j: int, i0: int, i1: int, src_j: int, use_next_v: bool, sign_is_outflow: str) -> None:
        nonlocal out
        i_vals = _range_values(i0, i1)
        if i_vals is None:
            return
        i_inner_wrf = i_vals if periodic_x else jnp.clip(i_vals, ibs + int(spec_zone), ibe - int(spec_zone))
        ii = i_vals - int(ims)
        ii_inner = i_inner_wrf - int(ims)
        jj = _idx(j, jms)
        src_jj = _idx(src_j, jms)
        wind_jj = _idx(j + 1 if use_next_v else j, jms)
        source = out[ii_inner[:, None], kk[None, :], src_jj]
        wind = v_arr[ii[:, None], kk[None, :], wind_jj]
        if sign_is_outflow == "negative":
            values = jnp.where(wind < 0.0, source, jnp.zeros_like(source))
        else:
            values = jnp.where(wind > 0.0, source, jnp.zeros_like(source))
        out = out.at[ii[:, None], kk[None, :], jj].set(values)

    def apply_x_boundary(i: int, j0: int, j1: int, src_i: int, use_next_u: bool, sign_is_outflow: str) -> None:
        nonlocal out
        j_vals = _range_values(j0, j1)
        if j_vals is None:
            return
        j_inner_wrf = jnp.clip(j_vals, jbs + int(spec_zone), jbe - int(spec_zone))
        jj = j_vals - int(jms)
        jj_inner = j_inner_wrf - int(jms)
        ii = _idx(i, ims)
        src_ii = _idx(src_i, ims)
        wind_ii = _idx(i + 1 if use_next_u else i, ims)
        source = out[src_ii, kk[:, None], jj_inner[None, :]]
        wind = u_arr[wind_ii, kk[:, None], jj[None, :]]
        if sign_is_outflow == "negative":
            values = jnp.where(wind < 0.0, source, jnp.zeros_like(source))
        else:
            values = jnp.where(wind > 0.0, source, jnp.zeros_like(source))
        out = out.at[ii, kk[:, None], jj[None, :]].set(values)

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            b_dist = j - jbs
            b_limit = 0 if periodic_x else b_dist
            apply_y_boundary(
                j,
                max(int(its), b_limit + ibs),
                min(itf, ibe - b_limit),
                jbs + int(spec_zone),
                False,
                "negative",
            )

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            b_dist = jbe - j
            b_limit = 0 if periodic_x else b_dist
            apply_y_boundary(
                j,
                max(int(its), b_limit + ibs),
                min(itf, ibe - b_limit),
                jbe - int(spec_zone),
                True,
                "positive",
            )

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                b_dist = i - ibs
                apply_x_boundary(
                    i,
                    max(int(jts), b_dist + jbs + 1),
                    min(jtf, jbe - b_dist - 1),
                    ibs + int(spec_zone),
                    False,
                    "negative",
                )

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                b_dist = ibe - i
                apply_x_boundary(
                    i,
                    max(int(jts), b_dist + jbs + 1),
                    min(jtf, jbe - b_dist - 1),
                    ibe - int(spec_zone),
                    True,
                    "positive",
                )

    return out


__all__ = ["flow_dep_bdy"]
