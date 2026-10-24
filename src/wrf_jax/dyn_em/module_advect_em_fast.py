"""Vectorized advection kernels for runtime hot paths."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
from wrf_jax.core.real_arithmetic import real_assignment, real_divide
import numpy as np


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(vals: Any, lower: int) -> Any:
    return vals - int(lower)


def _start_and_size(vals: Any, lower: int) -> tuple[Any, int]:
    arr = jnp.asarray(vals, dtype=jnp.int32)
    return arr[0] - int(lower), int(arr.size)


def _clip_start(start: Any, size: int, extent: int) -> Any:
    return jnp.clip(start, 0, max(int(extent) - int(size), 0))


def _take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, ims: int, kms: int, jms: int) -> Any:
    arr = jnp.asarray(arr)
    sizes = (int(i_vals.size), int(k_vals.size), int(j_vals.size))
    start_i = jnp.clip(_idx(i_vals[0], ims), 0, arr.shape[0] - sizes[0])
    start_k = jnp.clip(_idx(k_vals[0], kms), 0, arr.shape[1] - sizes[1])
    start_j = jnp.clip(_idx(j_vals[0], jms), 0, arr.shape[2] - sizes[2])
    return jax.lax.dynamic_slice(arr, (start_i, start_k, start_j), sizes)


def _take2(arr: Any, i_vals: Any, j_vals: Any, ims: int, jms: int) -> Any:
    arr = jnp.asarray(arr)
    sizes = (int(i_vals.size), int(j_vals.size))
    start_i = jnp.clip(_idx(i_vals[0], ims), 0, arr.shape[0] - sizes[0])
    start_j = jnp.clip(_idx(j_vals[0], jms), 0, arr.shape[1] - sizes[1])
    return jax.lax.dynamic_slice(arr, (start_i, start_j), sizes)


def _set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any, ims: int, kms: int, jms: int) -> Any:
    if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
        return arr
    i0, ni = _start_and_size(i_vals, ims)
    k0, nk = _start_and_size(k_vals, kms)
    j0, nj = _start_and_size(j_vals, jms)
    if ni > arr.shape[0] or nk > arr.shape[1] or nj > arr.shape[2]:
        return arr.at[_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :]].set(values)
    return jax.lax.dynamic_update_slice(
        arr,
        values,
        (_clip_start(i0, ni, arr.shape[0]), _clip_start(k0, nk, arr.shape[1]), _clip_start(j0, nj, arr.shape[2])),
    )


def _minimum_tile(arr: Any, values: Any, starts: tuple[int, ...]) -> Any:
    """Apply rectangular minima with scatter's negative-index and drop rules."""

    values = jnp.asarray(values, dtype=arr.dtype)
    axes = []
    for start, size, extent in zip(starts, values.shape, arr.shape, strict=True):
        segments = []
        for lower, upper, shift in ((-extent, 0, extent), (0, extent, 0)):
            left, right = max(start, lower), min(start + size, upper)
            if right > left:
                segments.append((left + shift, right - left, slice(left - start, right - start)))
        axes.append(segments)
    for tile in product(*axes):
        origin, sizes, source = zip(*tile, strict=True)
        limits = tuple(start + size for start, size in zip(origin, sizes, strict=True))
        updated = jnp.minimum(jax.lax.slice(arr, origin, limits), values[source])
        arr = jax.lax.dynamic_update_slice(arr, updated, origin)
    return arr


def _write_tile(arr: Any, values: Any, starts: tuple[int, ...]) -> Any:
    """Store a rectangular tile, retaining scatter semantics outside storage."""
    values = jnp.asarray(values, dtype=arr.dtype)
    if all(0 <= start and start + size <= extent
           for start, size, extent in zip(starts, values.shape, arr.shape, strict=True)):
        return jax.lax.dynamic_update_slice(arr, values, starts)
    indices = jnp.ix_(*(jnp.arange(start, start + size)
                        for start, size in zip(starts, values.shape, strict=True)))
    return arr.at[indices].set(values)


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    """Retain REAL rounding through the explicit dynamic integer token."""

    return real_assignment(value, rounding_token)


def _real_add(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) + jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _real_sub(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) - jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _real_sub_strict(left: Any, right: Any, rounding_token: Any) -> Any:
    """Perform one correctly rounded REAL subtraction without reassociation."""

    with jax.enable_x64(True):
        result = (
            jnp.asarray(left, dtype=jnp.float64)
            - jnp.asarray(right, dtype=jnp.float64)
        ).astype(jnp.float32)
    return _store_wrf_real(result, rounding_token)


def _real_add_strict(left: Any, right: Any, rounding_token: Any) -> Any:
    """Perform one correctly rounded REAL addition without reassociation."""

    with jax.enable_x64(True):
        result = (
            jnp.asarray(left, dtype=jnp.float64)
            + jnp.asarray(right, dtype=jnp.float64)
        ).astype(jnp.float32)
    return _store_wrf_real(result, rounding_token)


def _real_mul_strict(left: Any, right: Any, rounding_token: Any) -> Any:
    """Multiply two stored REAL operands without contraction into producers."""

    with jax.enable_x64(True):
        result = (
            jnp.asarray(left, dtype=jnp.float64)
            * jnp.asarray(right, dtype=jnp.float64)
        ).astype(jnp.float32)
    return _store_wrf_real(result, rounding_token)


def _real_mul(left: Any, right: Any, rounding_token: Any) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) * jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _real_div(left: Any, right: Any, rounding_token: Any) -> Any:
    denominator = real_assignment(right, rounding_token)
    return _store_wrf_real(
        real_divide(jnp.asarray(left, dtype=jnp.float32), denominator),
        rounding_token,
    )


def _flux3(
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    ua: Any,
    time_step: int,
    rounding_token: Any | None = None,
) -> Any:
    if rounding_token is not None:
        flux4 = _real_div(
            _real_sub(
                _real_mul(7.0, _real_add(q_i, q_im1, rounding_token), rounding_token),
                _real_add(q_ip1, q_im2, rounding_token),
                rounding_token,
            ),
            12.0,
            rounding_token,
        )
        correction = _real_div(
            _real_sub(
                _real_sub(q_ip1, q_im2, rounding_token),
                _real_mul(3.0, _real_sub(q_i, q_im1, rounding_token), rounding_token),
                rounding_token,
            ),
            12.0,
            rounding_token,
        )
        sign = jnp.where(ua >= 0.0, 1.0, -1.0)
        sign = sign if int(time_step) >= 0 else -sign
        return _real_add(
            flux4,
            _real_mul(sign, correction, rounding_token),
            rounding_token,
        )
    flux4 = (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0
    sign = jnp.where(ua >= 0.0, 1.0, -1.0)
    sign = sign if int(time_step) >= 0 else -sign
    return flux4 + sign * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1)) / 12.0


def _flux5(
    q_im3: Any,
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    q_ip2: Any,
    ua: Any,
    time_step: int,
    rounding_token: Any | None = None,
) -> Any:
    if rounding_token is not None:
        flux6 = _real_div(
            _real_add(
                _real_sub(
                    _real_mul(37.0, _real_add(q_i, q_im1, rounding_token), rounding_token),
                    _real_mul(8.0, _real_add(q_ip1, q_im2, rounding_token), rounding_token),
                    rounding_token,
                ),
                _real_add(q_ip2, q_im3, rounding_token),
                rounding_token,
            ),
            60.0,
            rounding_token,
        )
        correction = _real_add(
            _real_sub(
                _real_sub(q_ip2, q_im3, rounding_token),
                _real_mul(5.0, _real_sub(q_ip1, q_im2, rounding_token), rounding_token),
                rounding_token,
            ),
            _real_mul(10.0, _real_sub(q_i, q_im1, rounding_token), rounding_token),
            rounding_token,
        )
        sign = jnp.where(ua >= 0.0, 1.0, -1.0)
        sign = sign if int(time_step) >= 0 else -sign
        return _real_sub(
            flux6,
            _real_mul(
                sign,
                _real_div(correction, 60.0, rounding_token),
                rounding_token,
            ),
            rounding_token,
        )
    flux6 = (37.0 * (q_i + q_im1) - 8.0 * (q_ip1 + q_im2) + (q_ip2 + q_im3)) / 60.0
    sign = jnp.where(ua >= 0.0, 1.0, -1.0)
    sign = sign if int(time_step) >= 0 else -sign
    return flux6 - sign * ((q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)) / 60.0


def _flux3_pd(
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    ua: Any,
    time_step: int,
    rounding_token: Any | None = None,
) -> Any:
    if rounding_token is not None:
        flux4 = _real_sub(
            _real_mul(jnp.float32(7.0 / 12.0), _real_add(q_i, q_im1, rounding_token), rounding_token),
            _real_mul(jnp.float32(1.0 / 12.0), _real_add(q_ip1, q_im2, rounding_token), rounding_token),
            rounding_token,
        )
        correction = _real_mul(
            jnp.float32(1.0 / 12.0),
            _real_sub(
                _real_sub(q_ip1, q_im2, rounding_token),
                _real_mul(3.0, _real_sub(q_i, q_im1, rounding_token), rounding_token),
                rounding_token,
            ),
            rounding_token,
        )
        sign = jnp.where(ua >= 0.0, 1.0, -1.0)
        sign = sign if int(time_step) >= 0 else -sign
        return _real_add(flux4, _real_mul(sign, correction, rounding_token), rounding_token)
    flux4 = (7.0 / 12.0) * (q_i + q_im1) - (1.0 / 12.0) * (q_ip1 + q_im2)
    sign = jnp.where(ua >= 0.0, 1.0, -1.0)
    sign = sign if int(time_step) >= 0 else -sign
    return flux4 + sign * (1.0 / 12.0) * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1))


def _flux5_pd(
    q_im3: Any,
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    q_ip2: Any,
    ua: Any,
    time_step: int,
    rounding_token: Any | None = None,
) -> Any:
    if rounding_token is not None:
        flux6 = _real_add(
            _real_sub(
                _real_mul(jnp.float32(37.0 / 60.0), _real_add(q_i, q_im1, rounding_token), rounding_token),
                _real_mul(jnp.float32(2.0 / 15.0), _real_add(q_ip1, q_im2, rounding_token), rounding_token),
                rounding_token,
            ),
            _real_mul(jnp.float32(1.0 / 60.0), _real_add(q_ip2, q_im3, rounding_token), rounding_token),
            rounding_token,
        )
        correction = _real_add(
            _real_sub(
                _real_sub(q_ip2, q_im3, rounding_token),
                _real_mul(5.0, _real_sub(q_ip1, q_im2, rounding_token), rounding_token),
                rounding_token,
            ),
            _real_mul(10.0, _real_sub(q_i, q_im1, rounding_token), rounding_token),
            rounding_token,
        )
        sign = jnp.where(ua >= 0.0, 1.0, -1.0)
        sign = sign if int(time_step) >= 0 else -sign
        return _real_sub(
            flux6,
            _real_mul(_real_mul(sign, jnp.float32(1.0 / 60.0), rounding_token), correction, rounding_token),
            rounding_token,
        )
    flux6 = (
        (37.0 / 60.0) * (q_i + q_im1)
        - (2.0 / 15.0) * (q_ip1 + q_im2)
        + (1.0 / 60.0) * (q_ip2 + q_im3)
    )
    sign = jnp.where(ua >= 0.0, 1.0, -1.0)
    sign = sign if int(time_step) >= 0 else -sign
    return flux6 - sign * (1.0 / 60.0) * (
        (q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)
    )


def _empty_like_flux(i_vals: Any, k_vals: Any, j_vals: Any, dtype: Any) -> Any:
    return jnp.zeros((int(i_vals.size), int(k_vals.size), int(j_vals.size)), dtype=dtype)


def _assemble_flux_patches(shape, patches, *, axis, dtype):
    """Pack flux rectangles once, retaining the last write at overlaps."""
    if not patches or not all(shape):
        return jnp.zeros(shape, dtype)
    h_edges, k_edges = {0, shape[axis]}, {0, shape[1]}
    for h, k, value in patches:
        if h < 0 or k < 0 or h + value.shape[axis] > shape[axis] or k + value.shape[1] > shape[1]:
            raise ValueError("flux patch outside its local array")
        h_edges.update((h, h + value.shape[axis]))
        k_edges.update((k, k + value.shape[1]))
    hs, ks = sorted(h_edges), sorted(k_edges)
    layers = []
    for k0, k1 in zip(ks, ks[1:]):
        faces = []
        for h0, h1 in zip(hs, hs[1:]):
            match = next(((h, k, value) for h, k, value in reversed(patches)
                          if h <= h0 and h1 <= h + value.shape[axis]
                          and k <= k0 and k1 <= k + value.shape[1]), None)
            if match is None:
                size = list(shape)
                size[axis], size[1] = h1 - h0, k1 - k0
                part = jnp.zeros(size, dtype)
            else:
                h, k, value = match
                region = [slice(None)] * 3
                region[axis], region[1] = slice(h0 - h, h1 - h), slice(k0 - k, k1 - k)
                part = value[tuple(region)]
            faces.append(part)
        layers.append(jnp.concatenate(faces, axis=axis))
    return jnp.concatenate(layers, axis=1)


def _flux_upwind(
    q_im1: Any,
    q_i: Any,
    cr: Any,
    rounding_token: Any | None = None,
) -> Any:
    if rounding_token is not None:
        magnitude = jnp.abs(cr)
        positive = jnp.minimum(
            jnp.float32(1.0),
            _real_add(cr, magnitude, rounding_token),
        )
        negative = jnp.maximum(
            jnp.float32(-1.0),
            _real_sub(cr, magnitude, rounding_token),
        )
        return _real_add(
            _real_mul(
                _real_mul(0.5, positive, rounding_token),
                q_im1,
                rounding_token,
            ),
            _real_mul(
                _real_mul(0.5, negative, rounding_token),
                q_i,
                rounding_token,
            ),
            rounding_token,
        )
    return 0.5 * jnp.minimum(1.0, cr + jnp.abs(cr)) * q_im1 + 0.5 * jnp.maximum(-1.0, cr - jnp.abs(cr)) * q_i


def advect_u_vectorized(
    u: Any,
    u_old: Any,
    tendency: Any,
    ru: Any,
    rv: Any,
    rom: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    time_step: int,
    config_flags: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
    rdzw: Any,
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
    rounding_token: Any | None = None,
) -> Any:
    """Vectorized WRF ``advect_u`` for the configured h5/v3 non-periodic branch."""

    del msfuy, msfvx, msfvy, msftx, msfty, kds, ime, jme, kme
    if int(_read(config_flags, "h_mom_adv_order", 5)) != 5 or int(_read(config_flags, "v_mom_adv_order", 3)) != 3:
        raise NotImplementedError("advect_u_vectorized covers h_mom_adv_order=5 and v_mom_adv_order=3")
    if _read(config_flags, "periodic_x", False) or _read(config_flags, "periodic_y", False) or _read(config_flags, "polar", False):
        raise NotImplementedError("advect_u_vectorized currently targets the configured non-periodic real case")

    del c1, c2, mut
    u_arr = jnp.asarray(u)
    u_old_arr = jnp.asarray(u_old)
    tend = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msfux_arr = jnp.asarray(msfux)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)
    if rounding_token is None:
        add = lambda left, right: left + right
        sub = lambda left, right: left - right
        mul = lambda left, right: left * right
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    degrade_xs = not (bool(_read(config_flags, "symmetric_xs", False)) or int(its) > int(ids) + 3)
    degrade_xe = not (bool(_read(config_flags, "symmetric_xe", False)) or int(ite) < int(ide) - 2)
    degrade_ys = not (bool(_read(config_flags, "symmetric_ys", False)) or int(jts) > int(jds) + 3)
    degrade_ye = not (bool(_read(config_flags, "symmetric_ye", False)) or int(jte) < int(jde) - 4)

    # Y-direction flux divergence.
    i0 = int(its)
    i1 = int(ite)
    if bool(_read(config_flags, "open_xs", False)) or specified:
        i0 = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified:
        i1 = min(int(ide) - 1, int(ite))
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    jf0 = j0
    jf1 = j1 + 1
    if degrade_ys:
        j0 = max(int(jts), int(jds) + 1)
        jf0 = int(jds) + 3
    if degrade_ye:
        j1 = min(int(jte), int(jde) - 2)
        jf1 = int(jde) - 3

    i_vals = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j_iface = jnp.arange(j0, j1 + 2, dtype=jnp.int32)
    fy = _empty_like_flux(i_vals, k_vals, j_iface, tend.dtype)

    def set_fy(j_values: Any, values: Any) -> None:
        nonlocal fy
        if int(j_values.size):
            fy = jax.lax.dynamic_update_slice(fy, values, (jnp.int32(0), jnp.int32(0), j_values[0] - j0))

    j5 = jnp.arange(max(j0, jf0), min(j1 + 1, jf1) + 1, dtype=jnp.int32)
    if int(j5.size):
        vel = mul(0.5, add(
            _take3(rv_arr, i_vals, k_vals, j5, ims, kms, jms),
            _take3(rv_arr, i_vals - 1, k_vals, j5, ims, kms, jms),
        ))
        set_fy(j5, mul(vel, _flux5(
            _take3(u_arr, i_vals, k_vals, j5 - 3, ims, kms, jms),
            _take3(u_arr, i_vals, k_vals, j5 - 2, ims, kms, jms),
            _take3(u_arr, i_vals, k_vals, j5 - 1, ims, kms, jms),
            _take3(u_arr, i_vals, k_vals, j5, ims, kms, jms),
            _take3(u_arr, i_vals, k_vals, j5 + 1, ims, kms, jms),
            _take3(u_arr, i_vals, k_vals, j5 + 2, ims, kms, jms),
            vel,
            time_step,
            rounding_token,
        )))
    for jb in (int(jds) + 1, int(jde) - 1):
        if j0 <= jb <= j1 + 1:
            ja = jnp.asarray([jb], dtype=jnp.int32)
            set_fy(ja, mul(
                mul(0.25, add(_take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(rv_arr, i_vals - 1, k_vals, ja, ims, kms, jms))),
                add(_take3(u_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(u_arr, i_vals, k_vals, ja - 1, ims, kms, jms)),
            ))
    for jb in (int(jds) + 2, int(jde) - 2):
        if j0 <= jb <= j1 + 1:
            ja = jnp.asarray([jb], dtype=jnp.int32)
            vel = mul(0.5, add(
                _take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms),
                _take3(rv_arr, i_vals - 1, k_vals, ja, ims, kms, jms),
            ))
            set_fy(ja, mul(vel, _flux3(
                _take3(u_arr, i_vals, k_vals, ja - 2, ims, kms, jms),
                _take3(u_arr, i_vals, k_vals, ja - 1, ims, kms, jms),
                _take3(u_arr, i_vals, k_vals, ja, ims, kms, jms),
                _take3(u_arr, i_vals, k_vals, ja + 1, ims, kms, jms),
                vel,
                time_step,
                rounding_token,
            )))
    j_cell = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    if int(j_cell.size):
        delta = sub(fy[:, :, j_cell + 1 - j0], fy[:, :, j_cell - j0])
        mrdy = mul(_take2(msfux_arr, i_vals, j_cell, ims, jms), float(rdy))
        values = sub(
            _take3(tend, i_vals, k_vals, j_cell, ims, kms, jms),
            mul(mrdy[:, None, :], delta),
        )
        tend = _set3(tend, i_vals, k_vals, j_cell, values, ims, kms, jms)

    # X-direction flux divergence.
    i0 = int(its)
    i1 = int(ite)
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    if0 = i0
    if1 = i1 + 1
    if degrade_xs:
        i0 = max(int(ids) + 1, int(its))
        if0 = int(ids) + 3
    if degrade_xe:
        i1 = min(int(ide) - 1, int(ite))
        if1 = int(ide) - 2

    i_cell = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    i_iface = jnp.arange(i0, i1 + 2, dtype=jnp.int32)
    j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    fx = _empty_like_flux(i_iface, k_vals, j_vals, tend.dtype)

    def set_fx(i_values: Any, values: Any) -> None:
        nonlocal fx
        if int(i_values.size):
            fx = jax.lax.dynamic_update_slice(fx, values, (i_values[0] - i0, jnp.int32(0), jnp.int32(0)))

    i5 = jnp.arange(max(i0, if0), min(i1 + 1, if1) + 1, dtype=jnp.int32)
    if int(i5.size):
        vel = mul(0.5, add(
            _take3(ru_arr, i5, k_vals, j_vals, ims, kms, jms),
            _take3(ru_arr, i5 - 1, k_vals, j_vals, ims, kms, jms),
        ))
        set_fx(i5, mul(vel, _flux5(
            _take3(u_arr, i5 - 3, k_vals, j_vals, ims, kms, jms),
            _take3(u_arr, i5 - 2, k_vals, j_vals, ims, kms, jms),
            _take3(u_arr, i5 - 1, k_vals, j_vals, ims, kms, jms),
            _take3(u_arr, i5, k_vals, j_vals, ims, kms, jms),
            _take3(u_arr, i5 + 1, k_vals, j_vals, ims, kms, jms),
            _take3(u_arr, i5 + 2, k_vals, j_vals, ims, kms, jms),
            vel,
            time_step,
            rounding_token,
        )))
    if degrade_xs:
        if i0 == int(ids) + 1:
            ia = jnp.asarray([int(ids) + 1], dtype=jnp.int32)
            ub = jnp.where(specified & (_take3(u_arr, ia, k_vals, j_vals, ims, kms, jms) < 0.0), _take3(u_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms))
            set_fx(ia, mul(
                mul(0.25, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia - 1, k_vals, j_vals, ims, kms, jms))),
                add(_take3(u_arr, ia, k_vals, j_vals, ims, kms, jms), ub),
            ))
        ia_value = int(ids) + 2
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = mul(0.5, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia - 1, k_vals, j_vals, ims, kms, jms)))
            set_fx(ia, mul(vel, _flux3(_take3(u_arr, ia - 2, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia + 1, k_vals, j_vals, ims, kms, jms), vel, time_step, rounding_token)))
    if degrade_xe:
        if i1 == int(ide) - 1:
            ia_value = int(ide)
            if i0 <= ia_value <= i1 + 1:
                ia = jnp.asarray([ia_value], dtype=jnp.int32)
                ub = jnp.where(specified & (_take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms) > 0.0), _take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia, k_vals, j_vals, ims, kms, jms))
                set_fx(ia, mul(
                    mul(0.25, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia - 1, k_vals, j_vals, ims, kms, jms))),
                    add(_take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms), ub),
                ))
        ia_value = int(ide) - 1
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = mul(0.5, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia - 1, k_vals, j_vals, ims, kms, jms)))
            set_fx(ia, mul(vel, _flux3(_take3(u_arr, ia - 2, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia - 1, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(u_arr, ia + 1, k_vals, j_vals, ims, kms, jms), vel, time_step, rounding_token)))
    if int(i_cell.size) and int(j_vals.size):
        delta = sub(fx[i_cell + 1 - i0, :, :], fx[i_cell - i0, :, :])
        mrdx = mul(_take2(msfux_arr, i_cell, j_vals, ims, jms), float(rdx))
        values = sub(
            _take3(tend, i_cell, k_vals, j_vals, ims, kms, jms),
            mul(mrdx[:, None, :], delta),
        )
        tend = _set3(tend, i_cell, k_vals, j_vals, values, ims, kms, jms)

    # Vertical flux divergence.
    i0 = int(its)
    i1 = min(int(ite), int(ide))
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    if bool(_read(config_flags, "open_ys", False)) or specified:
        i0 = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_ye", False)) or specified:
        i1 = min(int(ide) - 1, int(ite))
    i_vals = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    k_iface = jnp.arange(int(kts), int(kte) + 1, dtype=jnp.int32)
    vf = _empty_like_flux(i_vals, k_iface, j_vals, tend.dtype)
    k3 = jnp.arange(int(kts) + 2, ktf, dtype=jnp.int32)
    if int(k3.size):
        vel = mul(0.5, add(
            _take3(rom_arr, i_vals - 1, k3, j_vals, ims, kms, jms),
            _take3(rom_arr, i_vals, k3, j_vals, ims, kms, jms),
        ))
        values = mul(vel, _flux3(
            _take3(u_arr, i_vals, k3 - 2, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals, k3 - 1, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals, k3, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals, k3 + 1, j_vals, ims, kms, jms),
            -vel,
            time_step,
            rounding_token,
        ))
        vf = jax.lax.dynamic_update_slice(vf, values, (0, 2, 0))
    for kb in (int(kts) + 1, ktf):
        if int(kts) <= kb <= int(kte):
            ka = jnp.asarray([kb], dtype=jnp.int32)
            values = mul(
                mul(0.5, add(_take3(rom_arr, i_vals, ka, j_vals, ims, kms, jms), _take3(rom_arr, i_vals - 1, ka, j_vals, ims, kms, jms))),
                add(
                    mul(fzm_arr[_idx(ka, kms)][None, :, None], _take3(u_arr, i_vals, ka, j_vals, ims, kms, jms)),
                    mul(fzp_arr[_idx(ka, kms)][None, :, None], _take3(u_arr, i_vals, ka - 1, j_vals, ims, kms, jms)),
                ),
            )
            vf = jax.lax.dynamic_update_slice(vf, values, (0, kb - int(kts), 0))
    k_cell = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(j_vals.size) and int(k_cell.size):
        delta = sub(vf[:, k_cell + 1 - int(kts), :], vf[:, k_cell - int(kts), :])
        values = sub(
            _take3(tend, i_vals, k_cell, j_vals, ims, kms, jms),
            mul(rdzw_arr[_idx(k_cell, kms)][None, :, None], delta),
        )
        tend = _set3(tend, i_vals, k_cell, j_vals, values, ims, kms, jms)

    # Open boundary branches are not active for the configured specified real case.
    del u_old_arr
    return tend


def advect_v_vectorized(
    v: Any,
    v_old: Any,
    tendency: Any,
    ru: Any,
    rv: Any,
    rom: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    time_step: int,
    config_flags: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
    rdzw: Any,
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
    rounding_token: Any | None = None,
) -> Any:
    """Vectorized WRF ``advect_v`` for the configured h5/v3 non-periodic branch."""

    del c1, c2, mut, msfux, msfuy, msftx, msfty, kds, ime, jme, kme, v_old
    if int(_read(config_flags, "h_mom_adv_order", 5)) != 5 or int(_read(config_flags, "v_mom_adv_order", 3)) != 3:
        raise NotImplementedError("advect_v_vectorized covers h_mom_adv_order=5 and v_mom_adv_order=3")
    if _read(config_flags, "periodic_x", False) or _read(config_flags, "periodic_y", False) or _read(config_flags, "polar", False):
        raise NotImplementedError("advect_v_vectorized currently targets the configured non-periodic real case")

    v_arr = jnp.asarray(v)
    tend = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)
    if rounding_token is None:
        add = lambda left, right: left + right
        sub = lambda left, right: left - right
        mul = lambda left, right: left * right
        div = lambda left, right: left / right
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div = lambda left, right: _real_div(left, right, rounding_token)
    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    degrade_xs = not (bool(_read(config_flags, "symmetric_xs", False)) or int(its) > int(ids) + 3)
    degrade_xe = not (bool(_read(config_flags, "symmetric_xe", False)) or int(ite) < int(ide) - 3)
    degrade_ys = not (bool(_read(config_flags, "symmetric_ys", False)) or int(jts) > int(jds) + 3)
    degrade_ye = not (bool(_read(config_flags, "symmetric_ye", False)) or int(jte) < int(jde) - 3)

    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j0 = int(jts)
    j1 = int(jte)
    jf0 = j0
    jf1 = j1 + 1
    if degrade_ys:
        j0 = max(int(jts), int(jds) + 1)
        jf0 = int(jds) + 3
    if degrade_ye:
        j1 = min(int(jte), int(jde) - 1)
        jf1 = int(jde) - 2
    j_iface = jnp.arange(j0, j1 + 2, dtype=jnp.int32)
    fy = _empty_like_flux(i_vals, k_vals, j_iface, tend.dtype)

    def set_fy(j_values: Any, values: Any) -> None:
        nonlocal fy
        if int(j_values.size):
            fy = jax.lax.dynamic_update_slice(fy, values, (jnp.int32(0), jnp.int32(0), j_values[0] - j0))

    j5 = jnp.arange(max(j0, jf0), min(j1 + 1, jf1) + 1, dtype=jnp.int32)
    if int(j5.size):
        vel = mul(0.5, add(
            _take3(rv_arr, i_vals, k_vals, j5, ims, kms, jms),
            _take3(rv_arr, i_vals, k_vals, j5 - 1, ims, kms, jms),
        ))
        set_fy(j5, mul(vel, _flux5(
            _take3(v_arr, i_vals, k_vals, j5 - 3, ims, kms, jms),
            _take3(v_arr, i_vals, k_vals, j5 - 2, ims, kms, jms),
            _take3(v_arr, i_vals, k_vals, j5 - 1, ims, kms, jms),
            _take3(v_arr, i_vals, k_vals, j5, ims, kms, jms),
            _take3(v_arr, i_vals, k_vals, j5 + 1, ims, kms, jms),
            _take3(v_arr, i_vals, k_vals, j5 + 2, ims, kms, jms),
            vel,
            time_step,
            rounding_token,
        )))
    jb = int(jds) + 1
    if j0 <= jb <= j1 + 1:
        ja = jnp.asarray([jb], dtype=jnp.int32)
        vb = jnp.where(specified & (_take3(v_arr, i_vals, k_vals, ja, ims, kms, jms) < 0.0), _take3(v_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(v_arr, i_vals, k_vals, ja - 1, ims, kms, jms))
        set_fy(ja, mul(
            mul(0.25, add(_take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(rv_arr, i_vals, k_vals, ja - 1, ims, kms, jms))),
            add(_take3(v_arr, i_vals, k_vals, ja, ims, kms, jms), vb),
        ))
    jb = int(jde)
    if j0 <= jb <= j1 + 1:
        ja = jnp.asarray([jb], dtype=jnp.int32)
        vb = jnp.where(specified & (_take3(v_arr, i_vals, k_vals, ja - 1, ims, kms, jms) > 0.0), _take3(v_arr, i_vals, k_vals, ja - 1, ims, kms, jms), _take3(v_arr, i_vals, k_vals, ja, ims, kms, jms))
        set_fy(ja, mul(
            mul(0.25, add(_take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(rv_arr, i_vals, k_vals, ja - 1, ims, kms, jms))),
            add(vb, _take3(v_arr, i_vals, k_vals, ja - 1, ims, kms, jms)),
        ))
    for jb in (int(jds) + 2, int(jde) - 1):
        if j0 <= jb <= j1 + 1:
            ja = jnp.asarray([jb], dtype=jnp.int32)
            vel = mul(0.5, add(_take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(rv_arr, i_vals, k_vals, ja - 1, ims, kms, jms)))
            set_fy(ja, mul(vel, _flux3(_take3(v_arr, i_vals, k_vals, ja - 2, ims, kms, jms), _take3(v_arr, i_vals, k_vals, ja - 1, ims, kms, jms), _take3(v_arr, i_vals, k_vals, ja, ims, kms, jms), _take3(v_arr, i_vals, k_vals, ja + 1, ims, kms, jms), vel, time_step, rounding_token)))
    j_cell = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    if int(j_cell.size):
        delta = sub(fy[:, :, j_cell + 1 - j0], fy[:, :, j_cell - j0])
        mrdy = mul(_take2(msfvy_arr, i_vals, j_cell, ims, jms), float(rdy))
        values = sub(
            _take3(tend, i_vals, k_vals, j_cell, ims, kms, jms),
            mul(mrdy[:, None, :], delta),
        )
        tend = _set3(tend, i_vals, k_vals, j_cell, values, ims, kms, jms)

    # X-direction flux divergence.
    i0 = int(its)
    i1 = min(int(ite), int(ide) - 1)
    j0 = int(jts)
    j1 = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified:
        j0 = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified:
        j1 = min(int(jde) - 1, int(jte))
    if0 = i0
    if1 = i1 + 1
    if degrade_xs:
        i0 = max(int(ids) + 1, int(its))
        if0 = min(i0 + 2, int(ids) + 3)
    if degrade_xe:
        i1 = min(int(ide) - 2, int(ite))
        if1 = int(ide) - 3
    i_cell = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    i_iface = jnp.arange(i0, i1 + 2, dtype=jnp.int32)
    j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    fx = _empty_like_flux(i_iface, k_vals, j_vals, tend.dtype)

    def set_fx(i_values: Any, values: Any) -> None:
        nonlocal fx
        if int(i_values.size):
            fx = jax.lax.dynamic_update_slice(fx, values, (i_values[0] - i0, jnp.int32(0), jnp.int32(0)))

    i5 = jnp.arange(max(i0, if0), min(i1 + 1, if1) + 1, dtype=jnp.int32)
    if int(i5.size):
        vel = mul(0.5, add(_take3(ru_arr, i5, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, i5, k_vals, j_vals - 1, ims, kms, jms)))
        set_fx(i5, mul(vel, _flux5(_take3(v_arr, i5 - 3, k_vals, j_vals, ims, kms, jms), _take3(v_arr, i5 - 2, k_vals, j_vals, ims, kms, jms), _take3(v_arr, i5 - 1, k_vals, j_vals, ims, kms, jms), _take3(v_arr, i5, k_vals, j_vals, ims, kms, jms), _take3(v_arr, i5 + 1, k_vals, j_vals, ims, kms, jms), _take3(v_arr, i5 + 2, k_vals, j_vals, ims, kms, jms), vel, time_step, rounding_token)))
    if degrade_xs:
        ia_value = int(ids) + 1
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            set_fx(ia, mul(
                mul(0.25, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia, k_vals, j_vals - 1, ims, kms, jms))),
                add(_take3(v_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia - 1, k_vals, j_vals, ims, kms, jms)),
            ))
        ia_value = int(ids) + 2
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = mul(0.5, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia, k_vals, j_vals - 1, ims, kms, jms)))
            set_fx(ia, mul(vel, _flux3(_take3(v_arr, ia - 2, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia - 1, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia + 1, k_vals, j_vals, ims, kms, jms), vel, time_step, rounding_token)))
    if degrade_xe:
        ia_value = int(ide) - 1
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            set_fx(ia, mul(
                mul(0.25, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia, k_vals, j_vals - 1, ims, kms, jms))),
                add(_take3(v_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia - 1, k_vals, j_vals, ims, kms, jms)),
            ))
        ia_value = int(ide) - 2
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = mul(0.5, add(_take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(ru_arr, ia, k_vals, j_vals - 1, ims, kms, jms)))
            set_fx(ia, mul(vel, _flux3(_take3(v_arr, ia - 2, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia - 1, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia, k_vals, j_vals, ims, kms, jms), _take3(v_arr, ia + 1, k_vals, j_vals, ims, kms, jms), vel, time_step, rounding_token)))
    if int(i_cell.size) and int(j_vals.size):
        delta = sub(fx[i_cell + 1 - i0, :, :], fx[i_cell - i0, :, :])
        mrdx = mul(_take2(msfvy_arr, i_cell, j_vals, ims, jms), float(rdx))
        values = sub(
            _take3(tend, i_cell, k_vals, j_vals, ims, kms, jms),
            mul(mrdx[:, None, :], delta),
        )
        tend = _set3(tend, i_cell, k_vals, j_vals, values, ims, kms, jms)

    # Vertical flux divergence.
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j0 = int(jts)
    j1 = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified:
        j0 = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified:
        j1 = min(int(jde) - 1, int(jte))
    j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    k_iface = jnp.arange(int(kts), int(kte) + 1, dtype=jnp.int32)
    vf = _empty_like_flux(i_vals, k_iface, j_vals, tend.dtype)
    k3 = jnp.arange(int(kts) + 2, ktf, dtype=jnp.int32)
    if int(k3.size):
        vel = mul(0.5, add(_take3(rom_arr, i_vals, k3, j_vals, ims, kms, jms), _take3(rom_arr, i_vals, k3, j_vals - 1, ims, kms, jms)))
        values = mul(vel, _flux3(_take3(v_arr, i_vals, k3 - 2, j_vals, ims, kms, jms), _take3(v_arr, i_vals, k3 - 1, j_vals, ims, kms, jms), _take3(v_arr, i_vals, k3, j_vals, ims, kms, jms), _take3(v_arr, i_vals, k3 + 1, j_vals, ims, kms, jms), -vel, time_step, rounding_token))
        vf = jax.lax.dynamic_update_slice(vf, values, (0, 2, 0))
    for kb in (int(kts) + 1, ktf):
        if int(kts) <= kb <= int(kte):
            ka = jnp.asarray([kb], dtype=jnp.int32)
            values = mul(
                mul(0.5, add(_take3(rom_arr, i_vals, ka, j_vals, ims, kms, jms), _take3(rom_arr, i_vals, ka, j_vals - 1, ims, kms, jms))),
                add(
                    mul(fzm_arr[_idx(ka, kms)][None, :, None], _take3(v_arr, i_vals, ka, j_vals, ims, kms, jms)),
                    mul(fzp_arr[_idx(ka, kms)][None, :, None], _take3(v_arr, i_vals, ka - 1, j_vals, ims, kms, jms)),
                ),
            )
            vf = jax.lax.dynamic_update_slice(vf, values, (0, kb - int(kts), 0))
    k_cell = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(j_vals.size) and int(k_cell.size):
        delta = sub(vf[:, k_cell + 1 - int(kts), :], vf[:, k_cell - int(kts), :])
        metric = div(
            _take2(msfvy_arr, i_vals, j_vals, ims, jms),
            _take2(msfvx_arr, i_vals, j_vals, ims, jms),
        )
        values = sub(
            _take3(tend, i_vals, k_cell, j_vals, ims, kms, jms),
            mul(
                mul(metric[:, None, :], rdzw_arr[_idx(k_cell, kms)][None, :, None]),
                delta,
            ),
        )
        tend = _set3(tend, i_vals, k_cell, j_vals, values, ims, kms, jms)

    return tend


def advect_w_vectorized(
    w: Any,
    w_old: Any,
    tendency: Any,
    ru: Any,
    rv: Any,
    rom: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    time_step: int,
    config_flags: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
    rdzu: Any,
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
    rounding_token: Any | None = None,
) -> Any:
    """Vectorized WRF ``advect_w`` for the configured h5/v3 non-periodic branch."""

    del c1, c2, mut, msfux, msfuy, msfvx, msfvy, msfty, kds, ime, jme, kme, w_old
    if int(_read(config_flags, "h_sca_adv_order", 5)) != 5 or int(_read(config_flags, "v_sca_adv_order", 3)) != 3:
        raise NotImplementedError("advect_w_vectorized covers h_sca_adv_order=5 and v_sca_adv_order=3")
    if _read(config_flags, "periodic_x", False) or _read(config_flags, "periodic_y", False) or _read(config_flags, "polar", False):
        raise NotImplementedError("advect_w_vectorized currently targets the configured non-periodic real case")

    w_arr = jnp.asarray(w)
    tend = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msftx_arr = jnp.asarray(msftx)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzu_arr = jnp.asarray(rdzu)
    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        store = jnp.asarray
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        store = lambda value: _store_wrf_real(value, rounding_token)
    ktf = min(int(kte), int(kde) - 1)

    degrade_xs = not (bool(_read(config_flags, "symmetric_xs", False)) or int(its) > int(ids) + 3)
    degrade_xe = not (bool(_read(config_flags, "symmetric_xe", False)) or int(ite) < int(ide) - 3)
    degrade_ys = not (bool(_read(config_flags, "symmetric_ys", False)) or int(jts) > int(jds) + 3)
    degrade_ye = not (bool(_read(config_flags, "symmetric_ye", False)) or int(jte) < int(jde) - 4)

    def ruw(i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        return add(
            mul(
                fzm_arr[_idx(k_vals, kms)][None, :, None],
                _take3(ru_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
            mul(
                fzp_arr[_idx(k_vals, kms)][None, :, None],
                _take3(ru_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
            ),
        )

    def rvw(i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        return add(
            mul(
                fzm_arr[_idx(k_vals, kms)][None, :, None],
                _take3(rv_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
            mul(
                fzp_arr[_idx(k_vals, kms)][None, :, None],
                _take3(rv_arr, i_vals, k_vals - 1, j_vals, ims, kms, jms),
            ),
        )

    def ruw_top(i_vals: Any, j_vals: Any) -> Any:
        km1 = np.asarray([ktf], dtype=np.int32)
        km2 = np.asarray([ktf - 1], dtype=np.int32)
        return sub(
            mul(
                sub(2.0, fzm_arr[ktf - int(kms)]),
                _take3(ru_arr, i_vals, km1, j_vals, ims, kms, jms),
            ),
            mul(
                fzp_arr[ktf - int(kms)],
                _take3(ru_arr, i_vals, km2, j_vals, ims, kms, jms),
            ),
        )

    def rvw_top(i_vals: Any, j_vals: Any) -> Any:
        km1 = np.asarray([ktf], dtype=np.int32)
        km2 = np.asarray([ktf - 1], dtype=np.int32)
        return sub(
            mul(
                sub(2.0, fzm_arr[ktf - int(kms)]),
                _take3(rv_arr, i_vals, km1, j_vals, ims, kms, jms),
            ),
            mul(
                fzp_arr[ktf - int(kms)],
                _take3(rv_arr, i_vals, km2, j_vals, ims, kms, jms),
            ),
        )

    i_vals = np.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_all = np.arange(int(kts) + 1, ktf + 2, dtype=jnp.int32)
    k_norm = np.arange(int(kts) + 1, ktf + 1, dtype=jnp.int32)
    k_top = np.asarray([ktf + 1], dtype=np.int32)
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    jf0 = j0
    jf1 = j1 + 1
    if degrade_ys:
        j0 = max(int(jts), int(jds) + 1)
        jf0 = int(jds) + 3
    if degrade_ye:
        j1 = min(int(jte), int(jde) - 2)
        jf1 = int(jde) - 3
    j_iface = np.arange(j0, j1 + 2, dtype=jnp.int32)
    fy_shape = (i_vals.size, k_all.size, j_iface.size)
    fy_patches = []

    def assign_fy(j_values: Any, k_values: Any, values: Any) -> None:
        if j_values.size and k_values.size:
            fy_patches.append((int(j_values[0]) - j0,
                               int(k_values[0]) - (int(kts) + 1), store(values)))

    j5 = np.arange(max(j0, jf0), min(j1 + 1, jf1) + 1, dtype=jnp.int32)
    if int(j5.size):
        if int(k_norm.size):
            vel = rvw(i_vals, k_norm, j5)
            assign_fy(
                j5,
                k_norm,
                mul(
                    vel,
                    _flux5(
                        _take3(w_arr, i_vals, k_norm, j5 - 3, ims, kms, jms),
                        _take3(w_arr, i_vals, k_norm, j5 - 2, ims, kms, jms),
                        _take3(w_arr, i_vals, k_norm, j5 - 1, ims, kms, jms),
                        _take3(w_arr, i_vals, k_norm, j5, ims, kms, jms),
                        _take3(w_arr, i_vals, k_norm, j5 + 1, ims, kms, jms),
                        _take3(w_arr, i_vals, k_norm, j5 + 2, ims, kms, jms),
                        vel,
                        time_step,
                        rounding_token,
                    ),
                ),
            )
        vel = rvw_top(i_vals, j5)
        assign_fy(
            j5,
            k_top,
            mul(
                vel,
                _flux5(
                    _take3(w_arr, i_vals, k_top, j5 - 3, ims, kms, jms),
                    _take3(w_arr, i_vals, k_top, j5 - 2, ims, kms, jms),
                    _take3(w_arr, i_vals, k_top, j5 - 1, ims, kms, jms),
                    _take3(w_arr, i_vals, k_top, j5, ims, kms, jms),
                    _take3(w_arr, i_vals, k_top, j5 + 1, ims, kms, jms),
                    _take3(w_arr, i_vals, k_top, j5 + 2, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                ),
            ),
        )
    for jb in (int(jds) + 1, int(jde) - 1):
        if j0 <= jb <= j1 + 1:
            ja = np.asarray([jb], dtype=np.int32)
            if int(k_norm.size):
                vel = rvw(i_vals, k_norm, ja)
                assign_fy(
                    ja,
                    k_norm,
                    mul(
                        mul(0.5, vel),
                        add(
                            _take3(w_arr, i_vals, k_norm, ja, ims, kms, jms),
                            _take3(w_arr, i_vals, k_norm, ja - 1, ims, kms, jms),
                        ),
                    ),
                )
            vel = rvw_top(i_vals, ja)
            assign_fy(
                ja,
                k_top,
                mul(
                    mul(0.5, vel),
                    add(
                        _take3(w_arr, i_vals, k_top, ja, ims, kms, jms),
                        _take3(w_arr, i_vals, k_top, ja - 1, ims, kms, jms),
                    ),
                ),
            )
    for jb in (int(jds) + 2, int(jde) - 2):
        if j0 <= jb <= j1 + 1:
            ja = np.asarray([jb], dtype=np.int32)
            if int(k_norm.size):
                vel = rvw(i_vals, k_norm, ja)
                assign_fy(
                    ja,
                    k_norm,
                    mul(
                        vel,
                        _flux3(
                            _take3(w_arr, i_vals, k_norm, ja - 2, ims, kms, jms),
                            _take3(w_arr, i_vals, k_norm, ja - 1, ims, kms, jms),
                            _take3(w_arr, i_vals, k_norm, ja, ims, kms, jms),
                            _take3(w_arr, i_vals, k_norm, ja + 1, ims, kms, jms),
                            vel,
                            time_step,
                            rounding_token,
                        ),
                    ),
                )
            vel = rvw_top(i_vals, ja)
            assign_fy(
                ja,
                k_top,
                mul(
                    vel,
                    _flux3(
                        _take3(w_arr, i_vals, k_top, ja - 2, ims, kms, jms),
                        _take3(w_arr, i_vals, k_top, ja - 1, ims, kms, jms),
                        _take3(w_arr, i_vals, k_top, ja, ims, kms, jms),
                        _take3(w_arr, i_vals, k_top, ja + 1, ims, kms, jms),
                        vel,
                        time_step,
                        rounding_token,
                    ),
                ),
            )
    j_cell = np.arange(j0, j1 + 1, dtype=jnp.int32)
    fy = _assemble_flux_patches(fy_shape, fy_patches, axis=2, dtype=tend.dtype)
    if int(j_cell.size):
        delta = sub(fy[:, :, j_cell + 1 - j0], fy[:, :, j_cell - j0])
        mrdy = mul(_take2(msftx_arr, i_vals, j_cell, ims, jms)[:, None, :], float(rdy))
        values = sub(
            _take3(tend, i_vals, k_all, j_cell, ims, kms, jms),
            mul(mrdy, delta),
        )
        tend = _set3(tend, i_vals, k_all, j_cell, values, ims, kms, jms)

    # X-direction flux divergence.
    i0 = int(its)
    i1 = min(int(ite), int(ide) - 1)
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    if0 = i0
    if1 = i1 + 1
    if degrade_xs:
        i0 = max(int(ids) + 1, int(its))
        if0 = min(i0 + 2, int(ids) + 3)
    if degrade_xe:
        i1 = min(int(ide) - 2, int(ite))
        if1 = int(ide) - 3
    i_cell = np.arange(i0, i1 + 1, dtype=jnp.int32)
    i_iface = np.arange(i0, i1 + 2, dtype=jnp.int32)
    j_vals = np.arange(j0, j1 + 1, dtype=jnp.int32)
    fx_shape = (i_iface.size, k_all.size, j_vals.size)
    fx_patches = []

    def assign_fx(i_values: Any, k_values: Any, values: Any) -> None:
        if i_values.size and k_values.size:
            fx_patches.append((int(i_values[0]) - i0,
                               int(k_values[0]) - (int(kts) + 1), store(values)))

    i5 = np.arange(max(i0, if0), min(i1 + 1, if1) + 1, dtype=jnp.int32)
    if int(i5.size):
        if int(k_norm.size):
            vel = ruw(i5, k_norm, j_vals)
            assign_fx(
                i5,
                k_norm,
                mul(
                    vel,
                    _flux5(
                        _take3(w_arr, i5 - 3, k_norm, j_vals, ims, kms, jms),
                        _take3(w_arr, i5 - 2, k_norm, j_vals, ims, kms, jms),
                        _take3(w_arr, i5 - 1, k_norm, j_vals, ims, kms, jms),
                        _take3(w_arr, i5, k_norm, j_vals, ims, kms, jms),
                        _take3(w_arr, i5 + 1, k_norm, j_vals, ims, kms, jms),
                        _take3(w_arr, i5 + 2, k_norm, j_vals, ims, kms, jms),
                        vel,
                        time_step,
                        rounding_token,
                    ),
                ),
            )
        vel = ruw_top(i5, j_vals)
        assign_fx(
            i5,
            k_top,
            mul(
                vel,
                _flux5(
                    _take3(w_arr, i5 - 3, k_top, j_vals, ims, kms, jms),
                    _take3(w_arr, i5 - 2, k_top, j_vals, ims, kms, jms),
                    _take3(w_arr, i5 - 1, k_top, j_vals, ims, kms, jms),
                    _take3(w_arr, i5, k_top, j_vals, ims, kms, jms),
                    _take3(w_arr, i5 + 1, k_top, j_vals, ims, kms, jms),
                    _take3(w_arr, i5 + 2, k_top, j_vals, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                ),
            ),
        )
    for ib in (int(ids) + 1, int(ide) - 1):
        if i0 <= ib <= i1 + 1:
            ia = np.asarray([ib], dtype=np.int32)
            if int(k_norm.size):
                vel = ruw(ia, k_norm, j_vals)
                assign_fx(
                    ia,
                    k_norm,
                    mul(
                        mul(0.5, vel),
                        add(
                            _take3(w_arr, ia, k_norm, j_vals, ims, kms, jms),
                            _take3(w_arr, ia - 1, k_norm, j_vals, ims, kms, jms),
                        ),
                    ),
                )
            vel = ruw_top(ia, j_vals)
            assign_fx(
                ia,
                k_top,
                mul(
                    mul(0.5, vel),
                    add(
                        _take3(w_arr, ia, k_top, j_vals, ims, kms, jms),
                        _take3(w_arr, ia - 1, k_top, j_vals, ims, kms, jms),
                    ),
                ),
            )
    for ib in (int(ids) + 2, int(ide) - 2):
        if i0 <= ib <= i1 + 1:
            ia = np.asarray([ib], dtype=np.int32)
            if int(k_norm.size):
                vel = ruw(ia, k_norm, j_vals)
                assign_fx(
                    ia,
                    k_norm,
                    mul(
                        vel,
                        _flux3(
                            _take3(w_arr, ia - 2, k_norm, j_vals, ims, kms, jms),
                            _take3(w_arr, ia - 1, k_norm, j_vals, ims, kms, jms),
                            _take3(w_arr, ia, k_norm, j_vals, ims, kms, jms),
                            _take3(w_arr, ia + 1, k_norm, j_vals, ims, kms, jms),
                            vel,
                            time_step,
                            rounding_token,
                        ),
                    ),
                )
            vel = ruw_top(ia, j_vals)
            assign_fx(
                ia,
                k_top,
                mul(
                    vel,
                    _flux3(
                        _take3(w_arr, ia - 2, k_top, j_vals, ims, kms, jms),
                        _take3(w_arr, ia - 1, k_top, j_vals, ims, kms, jms),
                        _take3(w_arr, ia, k_top, j_vals, ims, kms, jms),
                        _take3(w_arr, ia + 1, k_top, j_vals, ims, kms, jms),
                        vel,
                        time_step,
                        rounding_token,
                    ),
                ),
            )
    fx = _assemble_flux_patches(fx_shape, fx_patches, axis=0, dtype=tend.dtype)
    if int(i_cell.size):
        delta = sub(fx[i_cell + 1 - i0, :, :], fx[i_cell - i0, :, :])
        mrdx = mul(_take2(msftx_arr, i_cell, j_vals, ims, jms)[:, None, :], float(rdx))
        values = sub(
            _take3(tend, i_cell, k_all, j_vals, ims, kms, jms),
            mul(mrdx, delta),
        )
        tend = _set3(tend, i_cell, k_all, j_vals, values, ims, kms, jms)

    # Vertical flux divergence.
    i_vals = np.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j_vals = np.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    k_iface = np.arange(int(kts), int(kte) + 1, dtype=jnp.int32)
    vf = _empty_like_flux(i_vals, k_iface, j_vals, tend.dtype)
    k3 = np.arange(int(kts) + 2, ktf + 1, dtype=jnp.int32)
    if int(k3.size):
        vel = mul(
            0.5,
            add(
                _take3(rom_arr, i_vals, k3, j_vals, ims, kms, jms),
                _take3(rom_arr, i_vals, k3 - 1, j_vals, ims, kms, jms),
            ),
        )
        values = store(
            mul(
                vel,
                _flux3(
                    _take3(w_arr, i_vals, k3 - 2, j_vals, ims, kms, jms),
                    _take3(w_arr, i_vals, k3 - 1, j_vals, ims, kms, jms),
                    _take3(w_arr, i_vals, k3, j_vals, ims, kms, jms),
                    _take3(w_arr, i_vals, k3 + 1, j_vals, ims, kms, jms),
                    -vel,
                    time_step,
                    rounding_token,
                ),
            )
        )
        vf = jax.lax.dynamic_update_slice(
            vf, values.astype(vf.dtype), (0, int(k3[0]) - int(kts), 0)
        )
    for kb in (int(kts) + 1, ktf + 1):
        ka = np.asarray([kb], dtype=np.int32)
        if int(kts) <= kb <= int(kte):
            values = mul(
                mul(
                    0.25,
                    add(
                        _take3(rom_arr, i_vals, ka, j_vals, ims, kms, jms),
                        _take3(rom_arr, i_vals, ka - 1, j_vals, ims, kms, jms),
                    ),
                ),
                add(
                    _take3(w_arr, i_vals, ka, j_vals, ims, kms, jms),
                    _take3(w_arr, i_vals, ka - 1, j_vals, ims, kms, jms),
                ),
            )
            vf = jax.lax.dynamic_update_slice(
                vf, store(values).astype(vf.dtype), (0, kb - int(kts), 0)
            )
    k_cell = np.arange(int(kts) + 1, ktf + 1, dtype=jnp.int32)
    if int(k_cell.size):
        delta = sub(vf[:, k_cell + 1 - int(kts), :], vf[:, k_cell - int(kts), :])
        values = sub(
            _take3(tend, i_vals, k_cell, j_vals, ims, kms, jms),
            mul(rdzu_arr[_idx(k_cell, kms)][None, :, None], delta),
        )
        tend = _set3(tend, i_vals, k_cell, j_vals, values, ims, kms, jms)
    k_top_update = np.asarray([ktf + 1], dtype=np.int32)
    values = add(
        _take3(tend, i_vals, k_top_update, j_vals, ims, kms, jms),
        mul(
            mul(2.0, rdzu_arr[ktf - int(kms)]),
            vf[:, ktf + 1 - int(kts), :][:, None, :],
        ),
    )
    tend = _set3(tend, i_vals, k_top_update, j_vals, values, ims, kms, jms)
    if rounding_token is not None:
        tend = _store_wrf_real(tend, rounding_token)
    return tend


def advect_scalar_vectorized(
    field: Any,
    field_old: Any,
    tendency: Any,
    ru: Any,
    rv: Any,
    rom: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    time_step: int,
    config_flags: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
    rdzw: Any,
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
    rounding_token: Any | None = None,
) -> Any:
    """Vectorized WRF ``advect_scalar`` for the configured h5/v3 non-periodic branch."""

    del field_old, c1, c2, mut, msfux, msfuy, msfvx, msfvy, msfty, kds, ime, jme, kme
    if int(_read(config_flags, "h_sca_adv_order", 5)) != 5 or int(_read(config_flags, "v_sca_adv_order", 3)) != 3:
        raise NotImplementedError("advect_scalar_vectorized covers h_sca_adv_order=5 and v_sca_adv_order=3")
    if (
        _read(config_flags, "periodic_x", False)
        or _read(config_flags, "periodic_y", False)
        or _read(config_flags, "polar", False)
        or _read(config_flags, "open_xs", False)
        or _read(config_flags, "open_xe", False)
        or _read(config_flags, "open_ys", False)
        or _read(config_flags, "open_ye", False)
    ):
        raise NotImplementedError("advect_scalar_vectorized currently targets the configured closed non-periodic real case")

    field_arr = jnp.asarray(field)
    tend = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msftx_arr = jnp.asarray(msftx)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)
    ktf = min(int(kte), int(kde) - 1)

    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left) + jnp.asarray(right)
        sub = lambda left, right: jnp.asarray(left) - jnp.asarray(right)
        mul = lambda left, right: jnp.asarray(left) * jnp.asarray(right)
        store = jnp.asarray
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        store = lambda value: _store_wrf_real(value, rounding_token)

    degrade_xs = not (bool(_read(config_flags, "symmetric_xs", False)) or int(its) > int(ids) + 3)
    degrade_xe = not (bool(_read(config_flags, "symmetric_xe", False)) or int(ite) < int(ide) - 3)
    degrade_ys = not (bool(_read(config_flags, "symmetric_ys", False)) or int(jts) > int(jds) + 3)
    degrade_ye = not (bool(_read(config_flags, "symmetric_ye", False)) or int(jte) < int(jde) - 4)

    # Y-direction scalar flux divergence.
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j0 = int(jts)
    j1 = min(int(jte), int(jde) - 1)
    jf0 = j0
    jf1 = j1 + 1
    if degrade_ys:
        j0 = max(int(jts), int(jds) + 1)
        jf0 = int(jds) + 3
    if degrade_ye:
        j1 = min(int(jte), int(jde) - 2)
        jf1 = int(jde) - 3
    j_iface = jnp.arange(j0, j1 + 2, dtype=jnp.int32)
    fy = _empty_like_flux(i_vals, k_vals, j_iface, tend.dtype)

    def set_fy(j_values: Any, values: Any) -> None:
        nonlocal fy
        if int(j_values.size):
            fy = jax.lax.dynamic_update_slice(fy, store(values), (jnp.int32(0), jnp.int32(0), j_values[0] - j0))

    j5 = jnp.arange(max(j0, jf0), min(j1 + 1, jf1) + 1, dtype=jnp.int32)
    if int(j5.size):
        vel = _take3(rv_arr, i_vals, k_vals, j5, ims, kms, jms)
        set_fy(
            j5,
            mul(
                vel,
                _flux5(
                _take3(field_arr, i_vals, k_vals, j5 - 3, ims, kms, jms),
                _take3(field_arr, i_vals, k_vals, j5 - 2, ims, kms, jms),
                _take3(field_arr, i_vals, k_vals, j5 - 1, ims, kms, jms),
                _take3(field_arr, i_vals, k_vals, j5, ims, kms, jms),
                _take3(field_arr, i_vals, k_vals, j5 + 1, ims, kms, jms),
                _take3(field_arr, i_vals, k_vals, j5 + 2, ims, kms, jms),
                vel,
                time_step,
                rounding_token,
                ),
            ),
        )
    for jb in (int(jds) + 1, int(jde) - 1):
        if j0 <= jb <= j1 + 1:
            ja = jnp.asarray([jb], dtype=jnp.int32)
            set_fy(
                ja,
                mul(
                    mul(0.5, _take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms)),
                    add(
                        _take3(field_arr, i_vals, k_vals, ja, ims, kms, jms),
                        _take3(field_arr, i_vals, k_vals, ja - 1, ims, kms, jms),
                    ),
                ),
            )
    for jb in (int(jds) + 2, int(jde) - 2):
        if j0 <= jb <= j1 + 1:
            ja = jnp.asarray([jb], dtype=jnp.int32)
            vel = _take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms)
            set_fy(
                ja,
                mul(
                    vel,
                    _flux3(
                    _take3(field_arr, i_vals, k_vals, ja - 2, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, ja - 1, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, ja, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, ja + 1, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                    ),
                ),
            )
    j_cell = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    if int(j_cell.size):
        delta = sub(fy[:, :, j_cell + 1 - j0], fy[:, :, j_cell - j0])
        mrdy = mul(_take2(msftx_arr, i_vals, j_cell, ims, jms)[:, None, :], float(rdy))
        values = sub(
            _take3(tend, i_vals, k_vals, j_cell, ims, kms, jms),
            mul(mrdy, delta),
        )
        tend = _set3(tend, i_vals, k_vals, j_cell, values, ims, kms, jms)

    # X-direction scalar flux divergence.
    i0 = int(its)
    i1 = min(int(ite), int(ide) - 1)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    if0 = i0
    if1 = i1 + 1
    if degrade_xs:
        i0 = max(int(ids) + 1, int(its))
        if0 = min(i0 + 2, int(ids) + 3)
    if degrade_xe:
        i1 = min(int(ide) - 2, int(ite))
        if1 = int(ide) - 3
    i_cell = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    i_iface = jnp.arange(i0, i1 + 2, dtype=jnp.int32)
    fx = _empty_like_flux(i_iface, k_vals, j_vals, tend.dtype)

    def set_fx(i_values: Any, values: Any) -> None:
        nonlocal fx
        if int(i_values.size):
            fx = jax.lax.dynamic_update_slice(fx, store(values), (i_values[0] - i0, jnp.int32(0), jnp.int32(0)))

    i5 = jnp.arange(max(i0, if0), min(i1 + 1, if1) + 1, dtype=jnp.int32)
    if int(i5.size):
        vel = _take3(ru_arr, i5, k_vals, j_vals, ims, kms, jms)
        set_fx(
            i5,
            mul(
                vel,
                _flux5(
                _take3(field_arr, i5 - 3, k_vals, j_vals, ims, kms, jms),
                _take3(field_arr, i5 - 2, k_vals, j_vals, ims, kms, jms),
                _take3(field_arr, i5 - 1, k_vals, j_vals, ims, kms, jms),
                _take3(field_arr, i5, k_vals, j_vals, ims, kms, jms),
                _take3(field_arr, i5 + 1, k_vals, j_vals, ims, kms, jms),
                _take3(field_arr, i5 + 2, k_vals, j_vals, ims, kms, jms),
                vel,
                time_step,
                rounding_token,
                ),
            ),
        )
    if degrade_xs:
        ia_value = int(ids) + 1
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            set_fx(
                ia,
                mul(
                    mul(0.5, _take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms)),
                    add(
                        _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                        _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                    ),
                ),
            )
        ia_value = int(ids) + 2
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = _take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms)
            set_fx(
                ia,
                mul(
                    vel,
                    _flux3(
                    _take3(field_arr, ia - 2, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia + 1, k_vals, j_vals, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                    ),
                ),
            )
    if degrade_xe:
        ia_value = int(ide) - 1
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            set_fx(
                ia,
                mul(
                    mul(0.5, _take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms)),
                    add(
                        _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                        _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                    ),
                ),
            )
        ia_value = int(ide) - 2
        if i0 <= ia_value <= i1 + 1:
            ia = jnp.asarray([ia_value], dtype=jnp.int32)
            vel = _take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms)
            set_fx(
                ia,
                mul(
                    vel,
                    _flux3(
                    _take3(field_arr, ia - 2, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, ia + 1, k_vals, j_vals, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                    ),
                ),
            )
    if int(i_cell.size) and int(j_vals.size):
        delta = sub(fx[i_cell + 1 - i0, :, :], fx[i_cell - i0, :, :])
        mrdx = mul(_take2(msftx_arr, i_cell, j_vals, ims, jms)[:, None, :], float(rdx))
        values = sub(
            _take3(tend, i_cell, k_vals, j_vals, ims, kms, jms),
            mul(mrdx, delta),
        )
        tend = _set3(tend, i_cell, k_vals, j_vals, values, ims, kms, jms)

    # Vertical scalar flux divergence.
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    k_iface = jnp.arange(int(kts), int(kte) + 1, dtype=jnp.int32)
    vf = _empty_like_flux(i_vals, k_iface, j_vals, tend.dtype)
    k3 = np.arange(int(kts) + 2, ktf, dtype=np.int32)
    if int(k3.size):
        vel = _take3(rom_arr, i_vals, k3, j_vals, ims, kms, jms)
        values = mul(
            vel,
            _flux3(
                _take3(field_arr, i_vals, k3 - 2, j_vals, ims, kms, jms),
                _take3(field_arr, i_vals, k3 - 1, j_vals, ims, kms, jms),
                _take3(field_arr, i_vals, k3, j_vals, ims, kms, jms),
                _take3(field_arr, i_vals, k3 + 1, j_vals, ims, kms, jms),
                -vel,
                time_step,
                rounding_token,
            ),
        )
        vf = jax.lax.dynamic_update_slice(
            vf, values.astype(vf.dtype), (0, int(k3[0]) - int(kts), 0)
        )
    for kb in (int(kts) + 1, ktf):
        if int(kts) <= kb <= int(kte):
            ka = np.asarray([kb], dtype=np.int32)
            values = mul(
                _take3(rom_arr, i_vals, ka, j_vals, ims, kms, jms),
                add(
                    mul(fzm_arr[_idx(ka, kms)][None, :, None], _take3(field_arr, i_vals, ka, j_vals, ims, kms, jms)),
                    mul(fzp_arr[_idx(ka, kms)][None, :, None], _take3(field_arr, i_vals, ka - 1, j_vals, ims, kms, jms)),
                ),
            )
            vf = jax.lax.dynamic_update_slice(
                vf, store(values).astype(vf.dtype), (0, kb - int(kts), 0)
            )
    k_cell = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    if int(k_cell.size):
        delta = sub(vf[:, k_cell + 1 - int(kts), :], vf[:, k_cell - int(kts), :])
        values = sub(
            _take3(tend, i_vals, k_cell, j_vals, ims, kms, jms),
            mul(rdzw_arr[_idx(k_cell, kms)][None, :, None], delta),
        )
        tend = _set3(tend, i_vals, k_cell, j_vals, values, ims, kms, jms)

    if rounding_token is not None:
        tend = _store_wrf_real(tend, rounding_token)
    return tend


def advect_scalar_pd_vectorized(
    field: Any,
    field_old: Any,
    tendency: Any,
    h_tendency: Any,
    z_tendency: Any,
    ru: Any,
    rv: Any,
    rom: Any,
    c1: Any,
    c2: Any,
    mut: Any,
    mub: Any,
    mu_old: Any,
    time_step: int,
    config_flags: Any,
    tenddec: bool,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
    rdzw: Any,
    dt: float,
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
    rounding_token: Any | None = None,
    return_directional_components: bool = False,
    return_flux_components: bool = False,
    return_prelimit_components: bool = False,
    precomputed_fluxes: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> tuple[Any, ...]:
    """Vectorized WRF ``advect_scalar_pd`` for the configured h5/v3 real-case branch."""

    del msfux, msfuy, msfvx, msfvy, ime, jme, kme
    if int(_read(config_flags, "h_sca_adv_order", 5)) != 5 or int(_read(config_flags, "v_sca_adv_order", 3)) != 3:
        raise NotImplementedError("advect_scalar_pd_vectorized covers h_sca_adv_order=5 and v_sca_adv_order=3")
    if (
        _read(config_flags, "periodic_x", False)
        or _read(config_flags, "periodic_y", False)
        or _read(config_flags, "polar", False)
        or _read(config_flags, "open_xs", False)
        or _read(config_flags, "open_xe", False)
        or _read(config_flags, "open_ys", False)
        or _read(config_flags, "open_ye", False)
    ):
        raise NotImplementedError("advect_scalar_pd_vectorized currently targets the configured closed non-periodic real case")

    field_arr = jnp.asarray(field)
    field_old_arr = jnp.asarray(field_old)
    tend = jnp.asarray(tendency).copy()
    h_out = jnp.asarray(h_tendency).copy()
    z_out = jnp.asarray(z_tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mut_arr = jnp.asarray(mut)
    mub_arr = jnp.asarray(mub)
    mu_old_arr = jnp.asarray(mu_old)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)

    def store(value: Any) -> Any:
        array = jnp.asarray(value, dtype=tend.dtype)
        if rounding_token is None:
            return array
        return _store_wrf_real(array, rounding_token)

    if rounding_token is None:
        add = lambda left, right: left + right
        sub = lambda left, right: left - right
        mul = lambda left, right: left * right
        div = lambda left, right: left / right
    else:
        add = lambda left, right: _real_add(left, right, rounding_token)
        sub = lambda left, right: _real_sub(left, right, rounding_token)
        mul = lambda left, right: _real_mul(left, right, rounding_token)
        div = lambda left, right: _real_div(left, right, rounding_token)
    ktf = min(int(kte), int(kde) - 1)
    eps = 1.0e-20
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    degrade_xs = not (bool(_read(config_flags, "symmetric_xs", False)) or int(its) > int(ids) + 3)
    degrade_xe = not (bool(_read(config_flags, "symmetric_xe", False)) or int(ite) < int(ide) - 4)
    degrade_ys = not (bool(_read(config_flags, "symmetric_ys", False)) or int(jts) > int(jds) + 3)
    degrade_ye = not (bool(_read(config_flags, "symmetric_ye", False)) or int(jte) < int(jde) - 4)

    i_flux = jnp.arange(int(ids), int(ide), dtype=jnp.int32)
    k_flux = jnp.arange(int(kds), int(kde), dtype=jnp.int32)
    kz_flux = jnp.arange(int(kds), int(kde) + 1, dtype=jnp.int32)
    j_flux = jnp.arange(int(jds), int(jde), dtype=jnp.int32)
    fqx = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqxl = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqxh = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqy = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqyl = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqyh = _empty_like_flux(i_flux, k_flux, j_flux, tend.dtype)
    fqz = _empty_like_flux(i_flux, kz_flux, j_flux, tend.dtype)
    fqzl = _empty_like_flux(i_flux, kz_flux, j_flux, tend.dtype)
    fqzh = _empty_like_flux(i_flux, kz_flux, j_flux, tend.dtype)

    def get_flux(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, *, k_lower: int = kds) -> Any:
        return _take3(arr, i_vals, k_vals, j_vals, ids, k_lower, jds)

    def set_flux(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any, *, k_lower: int = kds) -> Any:
        return _set3(arr, i_vals, k_vals, j_vals, values, ids, k_lower, jds)

    # Supplied fluxes already contain this stage. Avoid tracing it again.
    if precomputed_fluxes is None or return_prelimit_components or return_flux_components:
        # Y-direction high-order minus low-order anti-diffusive flux.
        i0 = int(its) - 1
        i1 = min(int(ite), int(ide) - 1) + 1
        j0 = int(jts) - 1
        j1 = min(int(jte), int(jde) - 1) + 1
        jf0 = j0
        jf1 = j1 + 1
        if degrade_xs:
            i0 = max(int(its) - 1, int(ids))
        if degrade_xe:
            i1 = min(int(ite) + 1, int(ide) - 1)
        if degrade_ys:
            j0 = max(int(jts) - 1, int(jds) + 1)
            jf0 = int(jds) + 3
        if degrade_ye:
            j1 = min(int(jte) + 1, int(jde) - 2)
            jf1 = int(jde) - 3

        i_vals = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
        k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)

        def y_flux(j_values: Any, high_values: Any) -> tuple[Any, Any, Any]:
            metric_here = _take2(msftx_arr, i_vals, j_values, ims, jms)
            metric_before = _take2(msftx_arr, i_vals, j_values - 1, ims, jms)
            c1k = c1_arr[_idx(k_vals, kms)][None, :, None]
            c2k = c2_arr[_idx(k_vals, kms)][None, :, None]
            mut_here = _take2(mut_arr, i_vals, j_values, ims, jms)[:, None, :]
            mut_before = _take2(mut_arr, i_vals, j_values - 1, ims, jms)[:, None, :]
            if rounding_token is None:
                dy = store(2.0 / (metric_here + metric_before) / float(rdy))
                mu = store(0.5 * (c1k * mut_here + c2k + c1k * mut_before + c2k))
            else:
                dy = div(div(2.0, add(metric_here, metric_before)), float(rdy))
                mu = mul(
                    0.5,
                    add(add(mul(c1k, mut_here), c2k), add(mul(c1k, mut_before), c2k)),
                )
            vel = _take3(rv_arr, i_vals, k_vals, j_values, ims, kms, jms)
            if rounding_token is None:
                cr = store(vel * float(dt) / dy[:, None, :] / mu)
                low = store(
                    mu
                    * (dy[:, None, :] / float(dt))
                    * _flux_upwind(
                        _take3(field_old_arr, i_vals, k_vals, j_values - 1, ims, kms, jms),
                        _take3(field_old_arr, i_vals, k_vals, j_values, ims, kms, jms),
                        cr,
                    )
                )
            else:
                cr = div(div(mul(vel, float(dt)), dy[:, None, :]), mu)
                low = mul(
                    mul(mu, div(dy[:, None, :], float(dt))),
                    _flux_upwind(
                        _take3(field_old_arr, i_vals, k_vals, j_values - 1, ims, kms, jms),
                        _take3(field_old_arr, i_vals, k_vals, j_values, ims, kms, jms),
                        cr,
                        rounding_token,
                    ),
                )
            high_stored = store(high_values)
            if rounding_token is None:
                anti = store(high_stored - low)
            else:
                anti = _real_sub_strict(high_stored, low, rounding_token)
            return anti, low, high_stored

        j5 = jnp.arange(max(j0, jf0), min(j1 + 1, jf1) + 1, dtype=jnp.int32)
        if int(j5.size):
            vel = _take3(rv_arr, i_vals, k_vals, j5, ims, kms, jms)
            high = mul(
                vel,
                _flux5_pd(
                    _take3(field_arr, i_vals, k_vals, j5 - 3, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, j5 - 2, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, j5 - 1, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, j5, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, j5 + 1, ims, kms, jms),
                    _take3(field_arr, i_vals, k_vals, j5 + 2, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                ),
            )
            anti, low, high_stored = y_flux(j5, high)
            fqy = set_flux(fqy, i_vals, k_vals, j5, anti)
            fqyl = set_flux(fqyl, i_vals, k_vals, j5, low)
            fqyh = set_flux(fqyh, i_vals, k_vals, j5, high_stored)
        for jb in (int(jds) + 1, int(jds) + 2, int(jde) - 2, int(jde) - 1):
            if j0 <= jb <= j1 + 1:
                ja = jnp.asarray([jb], dtype=jnp.int32)
                vel = _take3(rv_arr, i_vals, k_vals, ja, ims, kms, jms)
                if jb in (int(jds) + 1, int(jde) - 1):
                    high = mul(
                        mul(0.5, vel),
                        add(
                            _take3(field_arr, i_vals, k_vals, ja, ims, kms, jms),
                            _take3(field_arr, i_vals, k_vals, ja - 1, ims, kms, jms),
                        ),
                    )
                else:
                    high = mul(
                        vel,
                        _flux3_pd(
                            _take3(field_arr, i_vals, k_vals, ja - 2, ims, kms, jms),
                            _take3(field_arr, i_vals, k_vals, ja - 1, ims, kms, jms),
                            _take3(field_arr, i_vals, k_vals, ja, ims, kms, jms),
                            _take3(field_arr, i_vals, k_vals, ja + 1, ims, kms, jms),
                            vel,
                            time_step,
                            rounding_token,
                        ),
                    )
                anti, low, high_stored = y_flux(ja, high)
                fqy = set_flux(fqy, i_vals, k_vals, ja, anti)
                fqyl = set_flux(fqyl, i_vals, k_vals, ja, low)
                fqyh = set_flux(fqyh, i_vals, k_vals, ja, high_stored)

        # X-direction high-order minus low-order anti-diffusive flux.
        i0 = int(its) - 1
        i1 = min(int(ite), int(ide) - 1) + 1
        if0 = i0
        if1 = i1 + 1
        j0 = int(jts) - 1
        j1 = min(int(jte), int(jde) - 1) + 1
        if degrade_ys:
            j0 = max(int(jts) - 1, int(jds))
        if degrade_ye:
            j1 = min(int(jte) + 1, int(jde) - 1)
        if degrade_xs:
            i0 = max(int(ids) + 1, int(its) - 1)
            if0 = int(ids) + 3
        if degrade_xe:
            i1 = min(int(ide) - 2, int(ite) + 1)
            if1 = int(ide) - 3

        j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)

        def x_flux(i_values: Any, high_values: Any) -> tuple[Any, Any, Any]:
            metric_here = _take2(msfty_arr, i_values, j_vals, ims, jms)
            metric_before = _take2(msfty_arr, i_values - 1, j_vals, ims, jms)
            c1k = c1_arr[_idx(k_vals, kms)][None, :, None]
            c2k = c2_arr[_idx(k_vals, kms)][None, :, None]
            mut_here = _take2(mut_arr, i_values, j_vals, ims, jms)[:, None, :]
            mut_before = _take2(mut_arr, i_values - 1, j_vals, ims, jms)[:, None, :]
            if rounding_token is None:
                dx = store(2.0 / (metric_here + metric_before) / float(rdx))
                mu = store(0.5 * (c1k * mut_here + c2k + c1k * mut_before + c2k))
            else:
                dx = div(div(2.0, add(metric_here, metric_before)), float(rdx))
                mu = mul(
                    0.5,
                    add(add(mul(c1k, mut_here), c2k), add(mul(c1k, mut_before), c2k)),
                )
            vel = _take3(ru_arr, i_values, k_vals, j_vals, ims, kms, jms)
            if rounding_token is None:
                cr = store(vel * float(dt) / dx[:, None, :] / mu)
                low = store(
                    mu
                    * (dx[:, None, :] / float(dt))
                    * _flux_upwind(
                        _take3(field_old_arr, i_values - 1, k_vals, j_vals, ims, kms, jms),
                        _take3(field_old_arr, i_values, k_vals, j_vals, ims, kms, jms),
                        cr,
                    )
                )
            else:
                cr = div(div(mul(vel, float(dt)), dx[:, None, :]), mu)
                low = mul(
                    mul(mu, div(dx[:, None, :], float(dt))),
                    _flux_upwind(
                        _take3(field_old_arr, i_values - 1, k_vals, j_vals, ims, kms, jms),
                        _take3(field_old_arr, i_values, k_vals, j_vals, ims, kms, jms),
                        cr,
                        rounding_token,
                    ),
                )
            high_stored = store(high_values)
            if rounding_token is None:
                anti = store(high_stored - low)
            else:
                anti = _real_sub_strict(high_stored, low, rounding_token)
            return anti, low, high_stored

        i5 = jnp.arange(max(i0, if0), min(i1, if1) + 1, dtype=jnp.int32)
        if int(i5.size):
            vel = _take3(ru_arr, i5, k_vals, j_vals, ims, kms, jms)
            high = mul(
                vel,
                _flux5_pd(
                    _take3(field_arr, i5 - 3, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, i5 - 2, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, i5 - 1, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, i5, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, i5 + 1, k_vals, j_vals, ims, kms, jms),
                    _take3(field_arr, i5 + 2, k_vals, j_vals, ims, kms, jms),
                    vel,
                    time_step,
                    rounding_token,
                ),
            )
            anti, low, high_stored = x_flux(i5, high)
            fqx = set_flux(fqx, i5, k_vals, j_vals, anti)
            fqxl = set_flux(fqxl, i5, k_vals, j_vals, low)
            fqxh = set_flux(fqxh, i5, k_vals, j_vals, high_stored)
        for ib in (int(ids) + 1, int(ids) + 2, int(ide) - 2, int(ide) - 1):
            if i0 <= ib <= i1 + 1:
                ia = jnp.asarray([ib], dtype=jnp.int32)
                vel = _take3(ru_arr, ia, k_vals, j_vals, ims, kms, jms)
                if ib in (int(ids) + 1, int(ide) - 1):
                    high = mul(
                        mul(0.5, vel),
                        add(
                            _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                            _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                        ),
                    )
                else:
                    high = mul(
                        vel,
                        _flux3_pd(
                            _take3(field_arr, ia - 2, k_vals, j_vals, ims, kms, jms),
                            _take3(field_arr, ia - 1, k_vals, j_vals, ims, kms, jms),
                            _take3(field_arr, ia, k_vals, j_vals, ims, kms, jms),
                            _take3(field_arr, ia + 1, k_vals, j_vals, ims, kms, jms),
                            vel,
                            time_step,
                            rounding_token,
                        ),
                    )
                anti, low, high_stored = x_flux(ia, high)
                fqx = set_flux(fqx, ia, k_vals, j_vals, anti)
                fqxl = set_flux(fqxl, ia, k_vals, j_vals, low)
                fqxh = set_flux(fqxh, ia, k_vals, j_vals, high_stored)

        # Vertical high-order minus low-order anti-diffusive flux.
        i0 = int(its) - 1
        i1 = min(int(ite), int(ide) - 1) + 1
        j0 = int(jts) - 1
        j1 = min(int(jte), int(jde) - 1) + 1
        if degrade_xs:
            i0 = max(int(its) - 1, int(ids))
        if degrade_xe:
            i1 = min(int(ite) + 1, int(ide) - 1)
        if degrade_ys:
            j0 = max(int(jts) - 1, int(jds))
        if degrade_ye:
            j1 = min(int(jte) + 1, int(jde) - 1)
        i_vals = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
        j_vals = jnp.arange(j0, j1 + 1, dtype=jnp.int32)

        def z_flux(k_values: Any, high_values: Any) -> tuple[Any, Any, Any]:
            dz = div(
                2.0,
                add(
                    rdzw_arr[_idx(k_values, kms)],
                    rdzw_arr[_idx(k_values - 1, kms)],
                ),
            )
            mu = add(
                mul(
                    c1_arr[_idx(k_values, kms)][None, :, None],
                    _take2(mut_arr, i_vals, j_vals, ims, jms)[:, None, :],
                ),
                c2_arr[_idx(k_values, kms)][None, :, None],
            )
            vel = _take3(rom_arr, i_vals, k_values, j_vals, ims, kms, jms)
            cr = div(div(mul(vel, float(dt)), dz[None, :, None]), mu)
            low = mul(
                mul(mu, div(dz[None, :, None], float(dt))),
                _flux_upwind(
                    _take3(field_old_arr, i_vals, k_values - 1, j_vals, ims, kms, jms),
                    _take3(field_old_arr, i_vals, k_values, j_vals, ims, kms, jms),
                    cr,
                    rounding_token,
                ),
            )
            high_stored = store(high_values)
            if rounding_token is None:
                anti = store(high_stored - low)
            else:
                high_stored = jax.lax.optimization_barrier(high_stored)
                low = jax.lax.optimization_barrier(low)
                anti = _real_sub_strict(high_stored, low, rounding_token)
            return anti, low, high_stored

        k3 = jnp.arange(int(kts) + 2, ktf, dtype=jnp.int32)
        if int(k3.size):
            vel = _take3(rom_arr, i_vals, k3, j_vals, ims, kms, jms)
            high = mul(
                vel,
                _flux3_pd(
                    _take3(field_arr, i_vals, k3 - 2, j_vals, ims, kms, jms),
                    _take3(field_arr, i_vals, k3 - 1, j_vals, ims, kms, jms),
                    _take3(field_arr, i_vals, k3, j_vals, ims, kms, jms),
                    _take3(field_arr, i_vals, k3 + 1, j_vals, ims, kms, jms),
                    -vel,
                    time_step,
                    rounding_token,
                ),
            )
            anti, low, high_stored = z_flux(k3, high)
            fqz = set_flux(fqz, i_vals, k3, j_vals, anti, k_lower=kds)
            fqzl = set_flux(fqzl, i_vals, k3, j_vals, low, k_lower=kds)
            fqzh = set_flux(fqzh, i_vals, k3, j_vals, high_stored, k_lower=kds)
        for kb in (int(kts) + 1, ktf):
            if int(kts) <= kb <= int(kte):
                ka = jnp.asarray([kb], dtype=jnp.int32)
                vel = _take3(rom_arr, i_vals, ka, j_vals, ims, kms, jms)
                high = mul(
                    vel,
                    add(
                        mul(
                            fzm_arr[_idx(ka, kms)][None, :, None],
                            _take3(field_arr, i_vals, ka, j_vals, ims, kms, jms),
                        ),
                        mul(
                            fzp_arr[_idx(ka, kms)][None, :, None],
                            _take3(field_arr, i_vals, ka - 1, j_vals, ims, kms, jms),
                        ),
                    ),
                )
                anti, low, high_stored = z_flux(ka, high)
                fqz = set_flux(fqz, i_vals, ka, j_vals, anti, k_lower=kds)
                fqzl = set_flux(fqzl, i_vals, ka, j_vals, low, k_lower=kds)
                fqzh = set_flux(fqzh, i_vals, ka, j_vals, high_stored, k_lower=kds)

    if return_prelimit_components:
        return fqxh, fqxl, fqyh, fqyl, fqzh, fqzl

    if precomputed_fluxes is not None:
        fqx, fqxl, fqy, fqyl, fqz, fqzl = precomputed_fluxes

    # Positive-definite limiter.  Each anti-diffusive face is owned by one
    # upwind cell, so face scale factors can be built independently.
    pi0 = int(its) - 1
    pi1 = min(int(ite), int(ide) - 1) + 1
    pj0 = int(jts) - 1
    pj1 = min(int(jte), int(jde) - 1) + 1
    if degrade_xs:
        pi0 = max(int(its) - 1, int(ids))
    if degrade_xe:
        pi1 = min(int(ite) + 1, int(ide) - 1)
    if degrade_ys:
        pj0 = max(int(jts) - 1, int(jds))
    if degrade_ye:
        pj1 = min(int(jte) + 1, int(jde) - 1)
    if specified:
        if degrade_xs:
            pi0 = max(int(its) - 1, int(ids) + 1)
        if degrade_xe:
            pi1 = min(int(ite) + 1, int(ide) - 2)
        if degrade_ys:
            pj0 = max(int(jts) - 1, int(jds) + 1)
        if degrade_ye:
            pj1 = min(int(jte) + 1, int(jde) - 2)

    pi = jnp.arange(pi0, pi1 + 1, dtype=jnp.int32)
    pk = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    pj = jnp.arange(pj0, pj1 + 1, dtype=jnp.int32)
    if int(pi.size) and int(pk.size) and int(pj.size):
        c1k = c1_arr[_idx(pk, kms)][None, :, None]
        c2k = c2_arr[_idx(pk, kms)][None, :, None]
        msftx_pk = _take2(msftx_arr, pi, pj, ims, jms)[:, None, :]
        msfty_pk = _take2(msfty_arr, pi, pj, ims, jms)[:, None, :]
        if rounding_token is None:
            mass_old = (
                (
                    c1k * _take2(mub_arr, pi, pj, ims, jms)[:, None, :]
                    + c2k
                )
                + c1k
                * _take2(mu_old_arr, pi, pj, ims, jms)[:, None, :]
            )
            ph_low = store(mass_old * _take3(field_old_arr, pi, pk, pj, ims, kms, jms) - float(dt) * (
                msftx_pk
                * msfty_pk
                * (
                    float(rdx) * (get_flux(fqxl, pi + 1, pk, pj) - get_flux(fqxl, pi, pk, pj))
                    + float(rdy) * (get_flux(fqyl, pi, pk, pj + 1) - get_flux(fqyl, pi, pk, pj))
                )
                + msfty_pk * rdzw_arr[_idx(pk, kms)][None, :, None] * (get_flux(fqzl, pi, pk + 1, pj) - get_flux(fqzl, pi, pk, pj))
            ))
        else:
            mass_old = _real_add(
                _real_add(
                    _real_mul(
                        c1k,
                        _take2(mub_arr, pi, pj, ims, jms)[:, None, :],
                        rounding_token,
                    ),
                    c2k,
                    rounding_token,
                ),
                _real_mul(
                    c1k,
                    _take2(mu_old_arr, pi, pj, ims, jms)[:, None, :],
                    rounding_token,
                ),
                rounding_token,
            )
            low_horizontal = _real_mul(
                _real_mul(msftx_pk, msfty_pk, rounding_token),
                _real_add(
                    _real_mul(
                        float(rdx),
                        _real_sub(
                            get_flux(fqxl, pi + 1, pk, pj),
                            get_flux(fqxl, pi, pk, pj),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    _real_mul(
                        float(rdy),
                        _real_sub(
                            get_flux(fqyl, pi, pk, pj + 1),
                            get_flux(fqyl, pi, pk, pj),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    rounding_token,
                ),
                rounding_token,
            )
            low_vertical = _real_mul(
                _real_mul(
                    msfty_pk,
                    rdzw_arr[_idx(pk, kms)][None, :, None],
                    rounding_token,
                ),
                _real_sub(
                    get_flux(fqzl, pi, pk + 1, pj),
                    get_flux(fqzl, pi, pk, pj),
                    rounding_token,
                ),
                rounding_token,
            )
            ph_low = _real_sub(
                _real_mul(
                    mass_old,
                    _take3(field_old_arr, pi, pk, pj, ims, kms, jms),
                    rounding_token,
                ),
                _real_mul(
                    float(dt),
                    _real_add(low_horizontal, low_vertical, rounding_token),
                    rounding_token,
                ),
                rounding_token,
            )
        east = get_flux(fqx, pi + 1, pk, pj)
        west = get_flux(fqx, pi, pk, pj)
        north = get_flux(fqy, pi, pk, pj + 1)
        south = get_flux(fqy, pi, pk, pj)
        top = get_flux(fqz, pi, pk + 1, pj)
        bottom = get_flux(fqz, pi, pk, pj)
        if rounding_token is None:
            flux_out = store(float(dt) * (
                msftx_pk
                * msfty_pk
                * (
                    float(rdx) * (jnp.maximum(0.0, east) - jnp.minimum(0.0, west))
                    + float(rdy) * (jnp.maximum(0.0, north) - jnp.minimum(0.0, south))
                )
                + msfty_pk
                * rdzw_arr[_idx(pk, kms)][None, :, None]
                * (jnp.minimum(0.0, top) - jnp.maximum(0.0, bottom))
            ))
            limited_scale = store(jnp.maximum(0.0, real_divide(ph_low, flux_out + eps)))
        else:
            horizontal_out = add(
                mul(
                    float(rdx),
                    sub(jnp.maximum(0.0, east), jnp.minimum(0.0, west)),
                ),
                mul(
                    float(rdy),
                    sub(jnp.maximum(0.0, north), jnp.minimum(0.0, south)),
                ),
            )
            horizontal_flux = mul(mul(msftx_pk, msfty_pk), horizontal_out)
            vertical_out = sub(
                jnp.minimum(0.0, top),
                jnp.maximum(0.0, bottom),
            )
            vertical_flux = mul(
                mul(msfty_pk, rdzw_arr[_idx(pk, kms)][None, :, None]),
                vertical_out,
            )
            flux_out = mul(float(dt), add(horizontal_flux, vertical_flux))
            limited_scale = jnp.maximum(
                jnp.float32(0.0),
                div(ph_low, add(flux_out, eps)),
            )
        cell_scale = jnp.where(flux_out > ph_low, limited_scale, jnp.float32(1.0))

        fqx_scale = jnp.ones_like(fqx)
        fqy_scale = jnp.ones_like(fqy)
        fqz_scale = jnp.ones_like(fqz)
        one = jnp.ones_like(cell_scale)
        fqx_scale = _minimum_tile(
            fqx_scale, jnp.where(east > 0.0, cell_scale, one),
            (pi0 + 1 - ids, kts - kds, pj0 - jds),
        )
        fqx_scale = _minimum_tile(
            fqx_scale, jnp.where(west < 0.0, cell_scale, one),
            (pi0 - ids, kts - kds, pj0 - jds),
        )
        fqy_scale = _minimum_tile(
            fqy_scale, jnp.where(north > 0.0, cell_scale, one),
            (pi0 - ids, kts - kds, pj0 + 1 - jds),
        )
        fqy_scale = _minimum_tile(
            fqy_scale, jnp.where(south < 0.0, cell_scale, one),
            (pi0 - ids, kts - kds, pj0 - jds),
        )
        fqz_scale = _minimum_tile(
            fqz_scale, jnp.where(top < 0.0, cell_scale, one),
            (pi0 - ids, kts + 1 - kds, pj0 - jds),
        )
        fqz_scale = _minimum_tile(
            fqz_scale, jnp.where(bottom > 0.0, cell_scale, one),
            (pi0 - ids, kts - kds, pj0 - jds),
        )
        pre_limit_fluxes = (fqx, fqxl, fqy, fqyl, fqz, fqzl)
        if rounding_token is None:
            fqx = store(fqx * fqx_scale)
            fqy = store(fqy * fqy_scale)
            fqz = store(fqz * fqz_scale)
        else:
            fqx = _real_mul_strict(fqx, fqx_scale, rounding_token)
            fqy = _real_mul_strict(fqy, fqy_scale, rounding_token)
            fqz = _real_mul_strict(fqz, fqz_scale, rounding_token)
    # Apply divergence from limited anti-diffusive plus low-order fluxes.
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    z_divergence_full = jnp.zeros_like(tend)
    z_update_mask = jnp.zeros(tend.shape, dtype=jnp.bool_)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        starts = (int(its) - ims, int(kts) - kms, int(jts) - jms)
        idx = (_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :])
        fqz_top = get_flux(fqz, i_vals, k_vals + 1, j_vals)
        fqz_bottom = get_flux(fqz, i_vals, k_vals, j_vals)
        fqzl_top = get_flux(fqzl, i_vals, k_vals + 1, j_vals)
        fqzl_bottom = get_flux(fqzl, i_vals, k_vals, j_vals)
        if rounding_token is None:
            z_divergence = rdzw_arr[_idx(k_vals, kms)][None, :, None] * (
                fqz_top - fqz_bottom + fqzl_top - fqzl_bottom
            )
            z_values = store(jnp.float32(0.0) - z_divergence)
            z_tendency = store(tend[idx] - z_divergence)
        else:
            z_flux_delta = sub(add(sub(fqz_top, fqz_bottom), fqzl_top), fqzl_bottom)
            z_divergence = mul(
                rdzw_arr[_idx(k_vals, kms)][None, :, None],
                z_flux_delta,
            )
            z_values = sub(jnp.float32(0.0), z_divergence)
            z_divergence_full = _write_tile(z_divergence_full, z_divergence, starts)
            z_update_mask = _write_tile(z_update_mask, jnp.ones(z_divergence.shape, dtype=jnp.bool_), starts)
        if rounding_token is None:
            tend = _write_tile(tend, z_tendency, starts)
        if bool(tenddec):
            z_out = _write_tile(z_out, z_values, starts)

    xi0 = int(its)
    xi1 = min(int(ite), int(ide) - 1)
    if degrade_xs:
        xi0 = max(int(its), int(ids) + 1)
    if degrade_xe:
        xi1 = min(int(ite), int(ide) - 2)
    i_vals = jnp.arange(xi0, xi1 + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    x_divergence_full = jnp.zeros_like(tend)
    x_update_mask = jnp.zeros(tend.shape, dtype=jnp.bool_)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        starts = (xi0 - ims, int(kts) - kms, int(jts) - jms)
        idx = (_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :])
        fqx_east = get_flux(fqx, i_vals + 1, k_vals, j_vals)
        fqx_west = get_flux(fqx, i_vals, k_vals, j_vals)
        fqxl_east = get_flux(fqxl, i_vals + 1, k_vals, j_vals)
        fqxl_west = get_flux(fqxl, i_vals, k_vals, j_vals)
        if rounding_token is None:
            x_divergence = _take2(msftx_arr, i_vals, j_vals, ims, jms)[:, None, :] * float(rdx) * (
                fqx_east - fqx_west + fqxl_east - fqxl_west
            )
            x_values = store(jnp.float32(0.0) - x_divergence)
            x_tendency = store(tend[idx] - x_divergence)
        else:
            x_flux_delta = sub(add(sub(fqx_east, fqx_west), fqxl_east), fqxl_west)
            x_divergence = mul(
                _take2(msftx_arr, i_vals, j_vals, ims, jms)[:, None, :],
                mul(float(rdx), x_flux_delta),
            )
            x_values = sub(jnp.float32(0.0), x_divergence)
            x_divergence_full = _write_tile(x_divergence_full, x_divergence, starts)
            x_update_mask = _write_tile(x_update_mask, jnp.ones(x_divergence.shape, dtype=jnp.bool_), starts)
        if rounding_token is None:
            tend = _write_tile(tend, x_tendency, starts)
        if bool(tenddec):
            h_out = _write_tile(h_out, x_values, starts)

    yi0 = int(jts)
    yi1 = min(int(jte), int(jde) - 1)
    if degrade_ys:
        yi0 = max(int(jts), int(jds) + 1)
    if degrade_ye:
        yi1 = min(int(jte), int(jde) - 2)
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j_vals = jnp.arange(yi0, yi1 + 1, dtype=jnp.int32)
    y_divergence_full = jnp.zeros_like(tend)
    y_update_mask = jnp.zeros(tend.shape, dtype=jnp.bool_)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        starts = (int(its) - ims, int(kts) - kms, yi0 - jms)
        idx = (_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :])
        fqy_north = get_flux(fqy, i_vals, k_vals, j_vals + 1)
        fqy_south = get_flux(fqy, i_vals, k_vals, j_vals)
        fqyl_north = get_flux(fqyl, i_vals, k_vals, j_vals + 1)
        fqyl_south = get_flux(fqyl, i_vals, k_vals, j_vals)
        if rounding_token is None:
            y_divergence = _take2(msftx_arr, i_vals, j_vals, ims, jms)[:, None, :] * float(rdy) * (
                fqy_north - fqy_south + fqyl_north - fqyl_south
            )
            y_values = store(jnp.float32(0.0) - y_divergence)
            y_tendency = store(tend[idx] - y_divergence)
        else:
            y_flux_delta = sub(add(sub(fqy_north, fqy_south), fqyl_north), fqyl_south)
            y_divergence = mul(
                _take2(msftx_arr, i_vals, j_vals, ims, jms)[:, None, :],
                mul(float(rdy), y_flux_delta),
            )
            y_values = sub(jnp.float32(0.0), y_divergence)
            y_divergence_full = _write_tile(y_divergence_full, y_divergence, starts)
            y_update_mask = _write_tile(y_update_mask, jnp.ones(y_divergence.shape, dtype=jnp.bool_), starts)
        if rounding_token is None:
            tend = _write_tile(tend, y_tendency, starts)
        if bool(tenddec):
            h_values = (
                _real_sub(h_out[idx], y_divergence, rounding_token)
                if rounding_token is not None
                else store(h_out[idx] - y_divergence)
            )
            h_out = _write_tile(h_out, h_values, starts)

    zero = jnp.zeros_like(tend)
    if rounding_token is None:
        z_direction = jnp.where(z_update_mask, store(-z_divergence_full), zero)
        x_direction = jnp.where(x_update_mask, store(-x_divergence_full), zero)
        y_direction = jnp.where(y_update_mask, store(-y_divergence_full), zero)
        tend = jnp.where(z_update_mask, store(tend + z_direction), tend)
        tend = jnp.where(x_update_mask, store(tend + x_direction), tend)
        tend = jnp.where(y_update_mask, store(tend + y_direction), tend)
    else:
        z_direction = jnp.where(
            z_update_mask,
            _real_sub_strict(zero, z_divergence_full, rounding_token),
            zero,
        )
        x_direction = jnp.where(
            x_update_mask,
            _real_sub_strict(zero, x_divergence_full, rounding_token),
            zero,
        )
        y_direction = jnp.where(
            y_update_mask,
            _real_sub_strict(zero, y_divergence_full, rounding_token),
            zero,
        )
        tend = jnp.where(
            z_update_mask,
            _real_add_strict(tend, z_direction, rounding_token),
            tend,
        )
        tend = jnp.where(
            x_update_mask,
            _real_add_strict(tend, x_direction, rounding_token),
            tend,
        )
        tend = jnp.where(
            y_update_mask,
            _real_add_strict(tend, y_direction, rounding_token),
            tend,
        )

    if return_directional_components or return_flux_components:
        components = (
            tend,
            h_out,
            z_out,
            z_direction,
            x_direction,
            y_direction,
            z_update_mask,
            x_update_mask,
            y_update_mask,
        )
        if return_flux_components:
            return components + (
                fqx,
                fqxl,
                fqy,
                fqyl,
                fqz,
                fqzl,
                ph_low,
                flux_out,
                cell_scale,
                pi,
                pk,
                pj,
                *pre_limit_fluxes,
                horizontal_out,
                horizontal_flux,
                vertical_out,
                vertical_flux,
                fqzh,
            )
        return components

    return tend, h_out, z_out


def subtract_scalar_pd_prelimit_fluxes(
    fqxh: Any,
    fqxl: Any,
    fqyh: Any,
    fqyl: Any,
    fqzh: Any,
    fqzl: Any,
    rounding_token: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Form anti-diffusive fluxes from materialized WRF REAL operands."""

    return (
        _real_sub_strict(fqxh, fqxl, rounding_token),
        fqxl,
        _real_sub_strict(fqyh, fqyl, rounding_token),
        fqyl,
        _real_sub_strict(fqzh, fqzl, rounding_token),
        fqzl,
    )


__all__ = [
    "advect_u_vectorized",
    "advect_v_vectorized",
    "advect_w_vectorized",
    "advect_scalar_vectorized",
    "advect_scalar_pd_vectorized",
    "subtract_scalar_pd_prelimit_fluxes",
]
