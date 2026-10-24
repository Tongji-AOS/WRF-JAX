"""Boundary tendency relaxation from WRF ``module_bc::relax_bdytend_core``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jax import lax
import jax.numpy as jnp
from wrf_jax.core.real_arithmetic import real_assignment


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    return real_assignment(value, rounding_token)


def relax_bdytend_core(
    field: Any,
    field_tend: Any,
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
    relax_zone: int,
    dtbc: float,
    fcx: Any,
    gcx: Any,
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
    iXs: int,
    iXe: int,
    jXs: int,
    jXe: int,
    kXs: int,
    kXe: int,
    rounding_token: Any | None = None,
) -> Any:
    """Return ``field_tend`` after WRF lateral-boundary relaxation updates.

    The interface keeps WRF's explicit lower and upper bounds so callers can
    pass arrays laid out like the original Fortran dummy arguments. Bounds and
    branch choices are static Python values; array state is returned instead of
    being mutated in place.
    """

    del spec_bdy_width, ime, jme, kme, ips, ipe, jps, jpe, kps, kpe, iXe, jXe, kXe

    fld = jnp.asarray(field)
    out = jnp.asarray(field_tend)
    bdy_xs = jnp.asarray(field_bdy_xs)
    bdy_xe = jnp.asarray(field_bdy_xe)
    bdy_ys = jnp.asarray(field_bdy_ys)
    bdy_ye = jnp.asarray(field_bdy_ye)
    bdy_tend_xs = jnp.asarray(field_bdy_tend_xs)
    bdy_tend_xe = jnp.asarray(field_bdy_tend_xe)
    bdy_tend_ys = jnp.asarray(field_bdy_tend_ys)
    bdy_tend_ye = jnp.asarray(field_bdy_tend_ye)
    fcx_arr = jnp.asarray(fcx)
    gcx_arr = jnp.asarray(gcx)

    variable = str(variable_in)[0]
    if variable == "U":
        variable = "u"
    if variable == "V":
        variable = "v"
    if variable == "M":
        variable = "m"
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
    if variable in {"m", "h"}:
        ktf = int(kte)

    periodic_x = bool(_read(config_flags, "periodic_x", False))
    dtbc_value = jnp.asarray(dtbc, dtype=fld.dtype)

    def real(value: Any) -> Any:
        if rounding_token is None:
            return value
        return _store_wrf_real(value, rounding_token)

    def add(left: Any, right: Any) -> Any:
        return real(left + right)

    def sub(left: Any, right: Any) -> Any:
        return real(left - right)

    def mul(left: Any, right: Any) -> Any:
        return real(left * right)

    def boundary_difference(boundary: Any, boundary_tend: Any, interior: Any) -> Any:
        return sub(add(boundary, mul(dtbc_value, boundary_tend)), interior)

    def add_update(idx: tuple[Any, Any, Any], b_dist: int, fls0: Any, fls1: Any, fls2: Any, fls3: Any, fls4: Any) -> None:
        nonlocal out
        b = b_dist + 1
        old = out[idx]
        if rounding_token is None:
            value = old + fcx_arr[b - 1] * fls0 - gcx_arr[b - 1] * (fls1 + fls2 + fls3 + fls4 - 4.0 * fls0)
        else:
            neighbor_sum = add(add(add(fls1, fls2), fls3), fls4)
            laplacian = sub(neighbor_sum, mul(jnp.asarray(4.0, dtype=fld.dtype), fls0))
            forcing = mul(fcx_arr[b - 1], fls0)
            relaxation = mul(gcx_arr[b - 1], laplacian)
            value = sub(add(old, forcing), relaxation)
        out = out.at[idx].set(value)

    def update_y_edge(j: int, b_dist: int, bdy: Any, bdy_tend: Any, south: bool) -> None:
        b_limit = 0 if periodic_x else int(b_dist)
        i_start = max(int(its), b_limit + ibs)
        i_end = min(itf, ibe - b_limit)
        k_start = int(kts)
        k_end = ktf
        if i_end < i_start or k_end < k_start:
            return

        jj_t = int(j) - int(jms)
        jj_f = int(j) - int(jXs)
        jm1_f = int(j) - 1 - int(jXs)
        jp1_f = int(j) + 1 - int(jXs)
        b0 = int(b_dist) + 1
        bm = int(b_dist)
        bp = int(b_dist) + 2
        if i_start > ibs and i_end < ibe:
            ii_t = slice(i_start - int(ims), i_end - int(ims) + 1)
            kk_t = slice(k_start - int(kms), k_end - int(kms) + 1)
            idx = (ii_t, kk_t, jj_t)

            ii_f = slice(i_start - int(iXs), i_end - int(iXs) + 1)
            im1_f = slice(i_start - 1 - int(iXs), i_end - 1 - int(iXs) + 1)
            ip1_f = slice(i_start + 1 - int(iXs), i_end + 1 - int(iXs) + 1)
            kk_f = slice(k_start - int(kXs), k_end - int(kXs) + 1)

            ii_b = slice(i_start - int(ims), i_end - int(ims) + 1)
            im1_b = slice(i_start - 1 - int(ims), i_end - 1 - int(ims) + 1)
            ip1_b = slice(i_start + 1 - int(ims), i_end + 1 - int(ims) + 1)
            kk_b = slice(k_start - int(kds), k_end - int(kds) + 1)

            fls0 = boundary_difference(bdy[ii_b, kk_b, b0 - 1], bdy_tend[ii_b, kk_b, b0 - 1], fld[ii_f, kk_f, jj_f])
            fls1 = boundary_difference(bdy[im1_b, kk_b, b0 - 1], bdy_tend[im1_b, kk_b, b0 - 1], fld[im1_f, kk_f, jj_f])
            fls2 = boundary_difference(bdy[ip1_b, kk_b, b0 - 1], bdy_tend[ip1_b, kk_b, b0 - 1], fld[ip1_f, kk_f, jj_f])
            if south:
                fls3 = boundary_difference(bdy[ii_b, kk_b, bm - 1], bdy_tend[ii_b, kk_b, bm - 1], fld[ii_f, kk_f, jm1_f])
                fls4 = boundary_difference(bdy[ii_b, kk_b, bp - 1], bdy_tend[ii_b, kk_b, bp - 1], fld[ii_f, kk_f, jp1_f])
            else:
                fls3 = boundary_difference(bdy[ii_b, kk_b, bm - 1], bdy_tend[ii_b, kk_b, bm - 1], fld[ii_f, kk_f, jp1_f])
                fls4 = boundary_difference(bdy[ii_b, kk_b, bp - 1], bdy_tend[ii_b, kk_b, bp - 1], fld[ii_f, kk_f, jm1_f])
            add_update(idx, int(b_dist), fls0, fls1, fls2, fls3, fls4)
            return

        i_fortran = jnp.arange(i_start, i_end + 1)
        k_fortran = jnp.arange(k_start, k_end + 1)
        im1 = jnp.maximum(i_fortran - 1, ibs)
        ip1 = jnp.minimum(i_fortran + 1, ibe)

        ii_t = i_fortran - int(ims)
        kk_t = k_fortran - int(kms)
        idx = (ii_t[:, None], kk_t[None, :], jj_t)

        ii_f = i_fortran - int(iXs)
        im1_f = im1 - int(iXs)
        ip1_f = ip1 - int(iXs)
        kk_f = k_fortran - int(kXs)

        ii_b = i_fortran - int(ims)
        im1_b = im1 - int(ims)
        ip1_b = ip1 - int(ims)
        kk_b = k_fortran - int(kds)

        fls0 = boundary_difference(bdy[ii_b[:, None], kk_b[None, :], b0 - 1], bdy_tend[ii_b[:, None], kk_b[None, :], b0 - 1], fld[ii_f[:, None], kk_f[None, :], jj_f])
        fls1 = boundary_difference(bdy[im1_b[:, None], kk_b[None, :], b0 - 1], bdy_tend[im1_b[:, None], kk_b[None, :], b0 - 1], fld[im1_f[:, None], kk_f[None, :], jj_f])
        fls2 = boundary_difference(bdy[ip1_b[:, None], kk_b[None, :], b0 - 1], bdy_tend[ip1_b[:, None], kk_b[None, :], b0 - 1], fld[ip1_f[:, None], kk_f[None, :], jj_f])
        if south:
            fls3 = boundary_difference(bdy[ii_b[:, None], kk_b[None, :], bm - 1], bdy_tend[ii_b[:, None], kk_b[None, :], bm - 1], fld[ii_f[:, None], kk_f[None, :], jm1_f])
            fls4 = boundary_difference(bdy[ii_b[:, None], kk_b[None, :], bp - 1], bdy_tend[ii_b[:, None], kk_b[None, :], bp - 1], fld[ii_f[:, None], kk_f[None, :], jp1_f])
        else:
            fls3 = boundary_difference(bdy[ii_b[:, None], kk_b[None, :], bm - 1], bdy_tend[ii_b[:, None], kk_b[None, :], bm - 1], fld[ii_f[:, None], kk_f[None, :], jp1_f])
            fls4 = boundary_difference(bdy[ii_b[:, None], kk_b[None, :], bp - 1], bdy_tend[ii_b[:, None], kk_b[None, :], bp - 1], fld[ii_f[:, None], kk_f[None, :], jm1_f])
        add_update(idx, int(b_dist), fls0, fls1, fls2, fls3, fls4)

    def update_x_edge(i: int, b_dist: int, bdy: Any, bdy_tend: Any, west: bool) -> None:
        j_start = max(int(jts), int(b_dist) + jbs + 1)
        j_end = min(jtf, jbe - int(b_dist) - 1)
        k_start = int(kts)
        k_end = ktf
        if j_end < j_start or k_end < k_start:
            return

        ii_t = int(i) - int(ims)
        kk_t = slice(k_start - int(kms), k_end - int(kms) + 1)
        jj_t = slice(j_start - int(jms), j_end - int(jms) + 1)
        idx = (ii_t, kk_t, jj_t)

        ii_f = int(i) - int(iXs)
        im1_f = int(i) - 1 - int(iXs)
        ip1_f = int(i) + 1 - int(iXs)
        kk_f = slice(k_start - int(kXs), k_end - int(kXs) + 1)
        jj_f = slice(j_start - int(jXs), j_end - int(jXs) + 1)
        jm1_f = slice(j_start - 1 - int(jXs), j_end - 1 - int(jXs) + 1)
        jp1_f = slice(j_start + 1 - int(jXs), j_end + 1 - int(jXs) + 1)

        jj_b = slice(j_start - int(jms), j_end - int(jms) + 1)
        jm1_b = slice(j_start - 1 - int(jms), j_end - 1 - int(jms) + 1)
        jp1_b = slice(j_start + 1 - int(jms), j_end + 1 - int(jms) + 1)
        kk_b = slice(k_start - int(kds), k_end - int(kds) + 1)
        b0 = int(b_dist) + 1
        bm = int(b_dist)
        bp = int(b_dist) + 2

        fls0 = boundary_difference(bdy[jj_b, kk_b, b0 - 1].T, bdy_tend[jj_b, kk_b, b0 - 1].T, fld[ii_f, kk_f, jj_f])
        fls1 = boundary_difference(bdy[jm1_b, kk_b, b0 - 1].T, bdy_tend[jm1_b, kk_b, b0 - 1].T, fld[ii_f, kk_f, jm1_f])
        fls2 = boundary_difference(bdy[jp1_b, kk_b, b0 - 1].T, bdy_tend[jp1_b, kk_b, b0 - 1].T, fld[ii_f, kk_f, jp1_f])
        if west:
            fls3 = boundary_difference(bdy[jj_b, kk_b, bm - 1].T, bdy_tend[jj_b, kk_b, bm - 1].T, fld[im1_f, kk_f, jj_f])
            fls4 = boundary_difference(bdy[jj_b, kk_b, bp - 1].T, bdy_tend[jj_b, kk_b, bp - 1].T, fld[ip1_f, kk_f, jj_f])
        else:
            fls3 = boundary_difference(bdy[jj_b, kk_b, bm - 1].T, bdy_tend[jj_b, kk_b, bm - 1].T, fld[ip1_f, kk_f, jj_f])
            fls4 = boundary_difference(bdy[jj_b, kk_b, bp - 1].T, bdy_tend[jj_b, kk_b, bp - 1].T, fld[im1_f, kk_f, jj_f])
        add_update(idx, int(b_dist), fls0, fls1, fls2, fls3, fls4)

    if int(jts) - jbs < int(relax_zone):
        for j in range(max(int(jts), jbs + int(spec_zone)), min(jtf, jbs + int(relax_zone) - 1) + 1):
            update_y_edge(j, j - jbs, bdy_ys, bdy_tend_ys, True)

    if jbe - jtf < int(relax_zone):
        for j in range(max(int(jts), jbe - int(relax_zone) + 1), min(jtf, jbe - int(spec_zone)) + 1):
            update_y_edge(j, jbe - j, bdy_ye, bdy_tend_ye, False)

    if not periodic_x:
        if int(its) - ibs < int(relax_zone):
            for i in range(max(int(its), ibs + int(spec_zone)), min(itf, ibs + int(relax_zone) - 1) + 1):
                update_x_edge(i, i - ibs, bdy_xs, bdy_tend_xs, True)

        if ibe - itf < int(relax_zone):
            for i in range(max(int(its), ibe - int(relax_zone) + 1), min(itf, ibe - int(spec_zone)) + 1):
                update_x_edge(i, ibe - i, bdy_xe, bdy_tend_xe, False)

    return out


__all__ = ["relax_bdytend_core"]
