"""Specified-boundary tendency extraction from WRF ``module_bc``."""

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
def _bdytend_index_groups(
    variable: str,
    periodic_x: bool,
    spec_zone: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
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
) -> tuple[tuple[Any, Any, Any, Any, Any, Any], ...]:
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

    groups: list[list[np.ndarray]] = [[np.asarray([], dtype=np.int32) for _ in range(6)] for _ in range(4)]

    def append(group_index: int, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        for n, values in enumerate(arrays):
            groups[group_index][n] = np.concatenate((groups[group_index][n], values.astype(np.int32, copy=False)))

    def append_y_edge(group_index: int, j: int, b_dist: int) -> None:
        b_limit = 0 if periodic_x else int(b_dist)
        i_start = max(int(its), b_limit + ibs)
        i_end = min(itf, ibe - b_limit)
        k_start = int(kts)
        k_end = ktf
        if i_end < i_start or k_end < k_start:
            return
        i_fortran = np.arange(i_start, i_end + 1, dtype=np.int32)
        k_fortran = np.arange(k_start, k_end + 1, dtype=np.int32)
        ii, kk_t = np.meshgrid(i_fortran - int(ims), k_fortran - int(kms), indexing="ij")
        _, kk_b = np.meshgrid(i_fortran - int(ims), k_fortran - int(kds), indexing="ij")
        jj = np.full(ii.shape, int(j) - int(jms), dtype=np.int32)
        bidx = np.full(ii.shape, int(b_dist), dtype=np.int32)
        append(group_index, (ii.ravel(), kk_t.ravel(), jj.ravel(), ii.ravel(), kk_b.ravel(), bidx.ravel()))

    def append_x_edge(group_index: int, i: int, b_dist: int) -> None:
        j_start = max(int(jts), int(b_dist) + jbs + 1)
        j_end = min(jtf, jbe - int(b_dist) - 1)
        k_start = int(kts)
        k_end = ktf
        if j_end < j_start or k_end < k_start:
            return
        j_fortran = np.arange(j_start, j_end + 1, dtype=np.int32)
        k_fortran = np.arange(k_start, k_end + 1, dtype=np.int32)
        kk_t, jj_t = np.meshgrid(k_fortran - int(kms), j_fortran - int(jms), indexing="ij")
        kk_b, jj_b = np.meshgrid(k_fortran - int(kds), j_fortran - int(jms), indexing="ij")
        ii = np.full(kk_t.shape, int(i) - int(ims), dtype=np.int32)
        bidx = np.full(kk_t.shape, int(b_dist), dtype=np.int32)
        append(group_index, (ii.ravel(), kk_t.ravel(), jj_t.ravel(), jj_b.ravel(), kk_b.ravel(), bidx.ravel()))

    if int(jts) - jbs < int(spec_zone):
        for j in range(int(jts), min(jtf, jbs + int(spec_zone) - 1) + 1):
            append_y_edge(0, j, j - jbs)

    if jbe - jtf < int(spec_zone):
        for j in range(max(int(jts), jbe - int(spec_zone) + 1), jtf + 1):
            append_y_edge(1, j, jbe - j)

    if not periodic_x:
        if int(its) - ibs < int(spec_zone):
            for i in range(int(its), min(itf, ibs + int(spec_zone) - 1) + 1):
                append_x_edge(2, i, i - ibs)

        if ibe - itf < int(spec_zone):
            for i in range(max(int(its), ibe - int(spec_zone) + 1), itf + 1):
                append_x_edge(3, i, ibe - i)

    return tuple(tuple(values.astype(np.int32, copy=False) for values in group) for group in groups)


def spec_bdytend(
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
    """Return WRF's ``field_tend`` boundary array for specified boundaries."""

    del field_bdy_xs, field_bdy_xe, field_bdy_ys, field_bdy_ye
    del spec_bdy_width, ips, ipe, jps, jpe, kps, kpe

    bdy_tend_xs = jnp.asarray(field_bdy_tend_xs)
    bdy_tend_xe = jnp.asarray(field_bdy_tend_xe)
    bdy_tend_ys = jnp.asarray(field_bdy_tend_ys)
    bdy_tend_ye = jnp.asarray(field_bdy_tend_ye)
    out = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1), dtype=bdy_tend_ys.dtype)

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

    index_groups = _bdytend_index_groups(
        variable,
        periodic_x,
        int(spec_zone),
        int(ids),
        int(ide),
        int(jds),
        int(jde),
        int(kds),
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

    def apply_group(current: Any, bdy_tend: Any, group: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        target_i, target_k, target_j, source_a, source_k, source_b = group
        if int(target_i.size) == 0:
            return current
        values = bdy_tend[source_a, source_k, source_b]
        return current.at[target_i, target_k, target_j].set(values)

    out = apply_group(out, bdy_tend_ys, index_groups[0])
    out = apply_group(out, bdy_tend_ye, index_groups[1])
    out = apply_group(out, bdy_tend_xs, index_groups[2])
    out = apply_group(out, bdy_tend_xe, index_groups[3])

    return out


def overlay_spec_bdytend(
    current: Any,
    field_bdy_tend_xs: Any,
    field_bdy_tend_xe: Any,
    field_bdy_tend_ys: Any,
    field_bdy_tend_ye: Any,
    variable_in: str,
    config_flags: Any,
    spec_bdy_width: int,
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
    """Overlay specified-boundary tendencies directly onto an existing field."""

    del spec_bdy_width, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe

    out = jnp.asarray(current)
    bdy_tend_xs = jnp.asarray(field_bdy_tend_xs)
    bdy_tend_xe = jnp.asarray(field_bdy_tend_xe)
    bdy_tend_ys = jnp.asarray(field_bdy_tend_ys)
    bdy_tend_ye = jnp.asarray(field_bdy_tend_ye)

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
    index_groups = _bdytend_index_groups(
        variable,
        periodic_x,
        int(spec_zone),
        int(ids),
        int(ide),
        int(jds),
        int(jde),
        int(kds),
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

    def apply_group_3d(current_out: Any, bdy_tend: Any, group: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        target_i, target_k, target_j, source_a, source_k, source_b = group
        if int(target_i.size) == 0:
            return current_out
        values = bdy_tend[source_a, source_k, source_b]
        return current_out.at[target_i, target_k, target_j].set(values)

    def apply_group_2d(current_out: Any, bdy_tend: Any, group: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        target_i, _target_k, target_j, source_a, source_k, source_b = group
        if int(target_i.size) == 0:
            return current_out
        values = bdy_tend[source_a, source_k, source_b]
        return current_out.at[target_i, target_j].set(values)

    apply_group = apply_group_2d if out.ndim == 2 else apply_group_3d
    out = apply_group(out, bdy_tend_ys, index_groups[0])
    out = apply_group(out, bdy_tend_ye, index_groups[1])
    out = apply_group(out, bdy_tend_xs, index_groups[2])
    out = apply_group(out, bdy_tend_xe, index_groups[3])
    return out


__all__ = ["overlay_spec_bdytend", "spec_bdytend"]
