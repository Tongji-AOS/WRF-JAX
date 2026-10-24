"""Kernels translated from WRF ``phys/module_bl_gwdo``."""

from __future__ import annotations

import os
from typing import Any

import jax
import numpy as np
import jax.numpy as jnp

from wrf_jax.core.wrf_math import clipped_sqrt_for_wrf, sqrt_for_wrf
from wrf_jax.core.optimization import semantic_call
from wrf_jax.core.column_scan import scan_vmapped_columns
from wrf_jax.core.real_arithmetic import column_binary


_BATCH_GWDO_COLUMNS = os.environ.get("WRF_PY_BATCH_GWDO_COLUMNS", "0").strip().lower() in {"1", "true", "yes", "on"}
_JAX_GWDO = os.environ.get("WRF_PY_JAX_GWDO", "1").strip().lower() in {"1", "true", "yes", "on"}
_JAX_GWDO_ACTIVE_COLUMNS = os.environ.get("WRF_PY_JAX_GWDO_ACTIVE_COLUMNS", "1").strip().lower() in {"1", "true", "yes", "on"}
_JAX_GWDO_ACTIVE_ZERO_INACTIVE = os.environ.get("WRF_PY_JAX_GWDO_ACTIVE_ZERO_INACTIVE", "1").strip().lower() in {"1", "true", "yes", "on"}
_JAX_GWDO_VMAP_COLUMNS = os.environ.get("WRF_PY_JAX_GWDO_VMAP_COLUMNS", "1").strip().lower() in {"1", "true", "yes", "on"}
_JAX_GWDO_ROW_RANGES = os.environ.get("WRF_PY_JAX_GWDO_ROW_RANGES", "0").strip().lower() in {"1", "true", "yes", "on"}
_INLINE_COLUMN_PHYSICS_JITS = os.environ.get("WRF_PY_INLINE_COLUMN_PHYSICS_JITS", "0").strip().lower() in {"1", "true", "yes", "on"}


@jax.custom_jvp
def _materialize_real_operation(value: Any) -> Any:
    """Expose a WRF REAL result so XLA cannot contract across the store."""

    return jnp.nextafter(value, value)


@_materialize_real_operation.defjvp
def _materialize_real_operation_jvp(
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Any, Any]:
    return _materialize_real_operation(primals[0]), tangents[0]


@jax.custom_jvp
def _materialize_real_with_token(value: Any, rounding_token: Any) -> Any:
    array = jnp.asarray(value, dtype=jnp.float32)
    bits = jax.lax.bitcast_convert_type(array, jnp.uint32)
    bits = jax.lax.bitwise_xor(
        bits,
        jnp.asarray(rounding_token, dtype=jnp.uint32),
    )
    restored = jax.lax.bitcast_convert_type(bits, jnp.float32)
    return jnp.nextafter(restored, restored)


@_materialize_real_with_token.defjvp
def _materialize_real_with_token_jvp(
    primals: tuple[Any, Any],
    tangents: tuple[Any, Any],
) -> tuple[Any, Any]:
    return _materialize_real_with_token(*primals), tangents[0]


def _real_store(value: Any, rounding_token: Any | None = None) -> Any:
    array = jnp.asarray(value, dtype=jnp.float32)
    if rounding_token is None:
        return _materialize_real_operation(array)
    return _materialize_real_with_token(array, rounding_token)


def _real_add(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _real_store(
        column_binary("add", _real_store(left, rounding_token), _real_store(right, rounding_token)),
        rounding_token,
    )


def _real_sub(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _real_store(
        column_binary("sub", _real_store(left, rounding_token), _real_store(right, rounding_token)),
        rounding_token,
    )


def _real_mul(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _real_store(
        column_binary("mul", _real_store(left, rounding_token), _real_store(right, rounding_token)),
        rounding_token,
    )


def _real_div(left: Any, right: Any, rounding_token: Any | None = None) -> Any:
    return _real_store(
        column_binary("div", _real_store(left, rounding_token), _real_store(right, rounding_token)),
        rounding_token,
    )


def _blocked_stress_profile(first, has_block, block_level, decrement, *, levels, rounding_token):
    def step(carry, layer, token):
        previous, active, top, delta = carry
        (idx,) = layer
        current = jnp.where(
            active & ((idx + 1) <= top),
            _real_sub(previous, delta, token),
            jnp.float32(0.0),
        )
        return (current, active, top, delta), current

    _, history = scan_vmapped_columns(
        step, (first, has_block, block_level, decrement),
        (jnp.arange(1, levels + 1, dtype=jnp.int32),), rounding_token,
    )
    return jnp.concatenate((jnp.asarray(first)[None], history))


def _surface_stress_integral(stress_u, stress_v, pressure_width, rounding_token):
    def step(carry, layer, token):
        previous_u, previous_v = carry
        du, dv, width = layer
        current_u = _real_add(previous_u, _real_mul(du, width, token), token)
        current_v = _real_add(previous_v, _real_mul(dv, width, token), token)
        return (current_u, current_v), ()

    current, _ = scan_vmapped_columns(
        step, (jnp.float32(0.0), jnp.float32(0.0)),
        (stress_u, stress_v, pressure_width), rounding_token,
    )
    return current


def _low_level_average(u, v, rho, pressure_width, inverse_depth, top, *, levels):
    def step(carry, layer, _parameter):
        mean_u, mean_v, mean_rho, inverse, limit = carry
        index, wind_u, wind_v, density, width = layer
        include = (index + 1) < limit
        weight = width * inverse
        mean_u = mean_u + jnp.where(include, weight * wind_u, jnp.float32(0))
        mean_v = mean_v + jnp.where(include, weight * wind_v, jnp.float32(0))
        mean_rho = mean_rho + jnp.where(include, weight * density, jnp.float32(0))
        return (mean_u, mean_v, mean_rho, inverse, limit), ()

    indices = jnp.arange(levels, dtype=jnp.int32)
    current, _ = scan_vmapped_columns(
        step,
        (jnp.float32(0), jnp.float32(0), jnp.float32(0), inverse_depth, top),
        (
            indices,
            *(jnp.take(x, indices, mode="clip") for x in (u, v, rho, pressure_width)),
        ),
    )
    return current[:3]


def _stability_average(
    bnv2_first,
    usqj_first,
    pressure,
    bnv2,
    usqj,
    inverse_depth,
    top,
    *,
    levels,
    rounding_token,
):
    def step(carry, layer, token):
        mean_bnv2, mean_usqj, inverse, limit = carry
        level, lower, upper, stability, richardson = layer
        include = level < limit
        weight = _real_mul(_real_sub(lower, upper, token), inverse, token)
        mean_bnv2 = _real_add(
            mean_bnv2,
            jnp.where(include, _real_mul(stability, weight, token), jnp.float32(0)),
            token,
        )
        mean_usqj = _real_add(
            mean_usqj,
            jnp.where(include, _real_mul(richardson, weight, token), jnp.float32(0)),
            token,
        )
        return (mean_bnv2, mean_usqj, inverse, limit), ()

    layer_levels = jnp.arange(2, levels + 1, dtype=jnp.int32)
    current, _ = scan_vmapped_columns(
        step,
        (bnv2_first, usqj_first, inverse_depth, top),
        (
            layer_levels,
            jnp.take(pressure, layer_levels - 1, mode="clip"),
            jnp.take(pressure, layer_levels, mode="clip"),
            jnp.take(bnv2, layer_levels - 1, mode="clip"),
            jnp.take(usqj, layer_levels - 1, mode="clip"),
        ),
        rounding_token,
    )
    return current[:2]


def _critical_stress_profile(taup, kbl, usqj, velco, bnv2, rho, active, coefm, dxy):
    """Keep the preceding stress and critical-level decision column-local."""
    nk = usqj.shape[0]
    if nk <= 2:
        return taup

    def step(carry, layer, _parameter):
        previous, critical, brvf, top, enabled, coefficient, width = carry
        k, richardson, velocity, stability, rho_lower, rho_upper, existing = layer
        ge_top = k >= top
        critical = jnp.where(
            ge_top,
            critical | (richardson < jnp.float32(0.25)) | (velocity <= jnp.float32(0)),
            critical,
        )
        brvf = jnp.where(
            ge_top, sqrt_for_wrf(jnp.maximum(stability, jnp.float32(1.0e-5))), brvf
        )
        update = ge_top & enabled & (~critical) & (previous > jnp.float32(0))
        safe_velocity = jnp.where(
            jnp.abs(velocity) > jnp.float32(0), velocity, jnp.float32(1)
        )
        temv = column_binary("div", jnp.float32(1), safe_velocity)
        tem1 = (
            column_binary("div", coefficient, width)
            * (rho_upper + rho_lower)
            * brvf
            * velocity
            * jnp.float32(0.5)
        )
        safe_tem1 = jnp.where(tem1 > jnp.float32(0), tem1, jnp.float32(1))
        hd = clipped_sqrt_for_wrf(column_binary("div", previous, safe_tem1))
        fro = brvf * hd * temv
        tem2 = clipped_sqrt_for_wrf(richardson)
        safe_tem2 = jnp.where(tem2 > jnp.float32(0), tem2, jnp.float32(1))
        tem_local = jnp.float32(1) + safe_tem2 * fro
        rim = column_binary(
            "div", richardson * (jnp.float32(1) - fro), tem_local * tem_local
        )
        temc = jnp.float32(2) + column_binary("div", jnp.float32(1), safe_tem2)
        hd_break = column_binary(
            "div",
            velocity
            * (jnp.float32(2) * clipped_sqrt_for_wrf(temc) - temc),
            jnp.where(brvf > jnp.float32(0), brvf, jnp.float32(1)),
        )
        tau_break = tem1 * hd_break * hd_break
        next_stress = jnp.where(rim <= jnp.float32(0.25), tau_break, previous)
        current = jnp.where(update, next_stress, existing)
        return (current, critical, brvf, top, enabled, coefficient, width), current

    _, history = scan_vmapped_columns(
        step,
        (taup[1], jnp.asarray(False), jnp.float32(0), kbl, active, coefm, dxy),
        (
            jnp.arange(2, nk, dtype=jnp.int32),
            usqj[1:-1],
            velco[1:-1],
            bnv2[1:-1],
            rho[1:-1],
            rho[2:],
            taup[2:nk],
        ),
    )
    return taup.at[2:nk].set(history)


def _blocking_level(kbl, bnv2, zl, pressure_width, rho, u, v, active, gravity):
    """Search downward without carrying or rewriting whole profiles."""
    nk = u.shape[0]
    if nk < 2:
        return jnp.int32(0), jnp.float32(0)
    top_height = _take_level_1d(zl, kbl)
    levels = jnp.arange(nk, 1, -1, dtype=jnp.int32)
    block_levels = jnp.minimum(levels, kbl)
    block_heights = _take_level_1d(zl, block_levels) - zl[0]

    def step(carry, layer, _parameter):
        block_level, potential, block_height, top, enabled, top_z, g = carry
        k, stability, z, width, density, wind_u, wind_v, new_level, new_height = layer
        update = enabled & (block_level == 0) & (k <= top)
        increment = column_binary(
            "div", column_binary("div", stability * (top_z - z) * width, g), density
        )
        potential = potential + jnp.where(update, increment, jnp.float32(0))
        kinetic = jnp.float32(0.5) * (wind_u * wind_u + wind_v * wind_v)
        blocked = update & (potential >= kinetic)
        block_level = jnp.where(blocked, new_level, block_level)
        block_height = jnp.where(blocked, new_height, block_height)
        return (block_level, potential, block_height, top, enabled, top_z, g), ()

    current, _ = scan_vmapped_columns(
        step,
        (
            jnp.int32(0),
            jnp.float32(0),
            jnp.float32(0),
            kbl,
            active,
            top_height,
            gravity,
        ),
        (
            levels,
            bnv2[:0:-1],
            zl[:0:-1],
            pressure_width[:0:-1],
            rho[:0:-1],
            u[:0:-1],
            v[:0:-1],
            block_levels,
            block_heights,
        ),
    )
    return current[0], current[2]


def _uses_jax_array(value: Any) -> bool:
    return isinstance(value, jax.Array) or isinstance(value, jax.core.Tracer)


def _call_gwdo_kernel(function: Any, args: tuple[Any, ...], static_kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Preserve the standalone GWDO kernel inside a larger device graph."""

    if any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(args)
    ):
        def call(*values: Any) -> tuple[Any, ...]:
            return function(*values, **static_kwargs)

        return semantic_call(call, *args)
    return function(*args, **static_kwargs)


def gwdo(
    u3d: Any,
    v3d: Any,
    t3d: Any,
    qv3d: Any,
    p3d: Any,
    p3di: Any,
    pi3d: Any,
    z: Any,
    rublten: Any,
    rvblten: Any,
    dtaux3d: Any,
    dtauy3d: Any,
    dusfcg: Any,
    dvsfcg: Any,
    var2d: Any,
    oc12d: Any,
    oa2d1: Any,
    oa2d2: Any,
    oa2d3: Any,
    oa2d4: Any,
    ol2d1: Any,
    ol2d2: Any,
    ol2d3: Any,
    ol2d4: Any,
    sina: Any,
    cosa: Any,
    znu: Any,
    znw: Any,
    p_top: float,
    cp: float,
    g: float,
    rd: float,
    rv: float,
    ep1: float,
    pi: float,
    dt: float,
    dx: float,
    kpbl2d: Any,
    itimestep: int,
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
    *,
    gwdo2d_fn=None,
    kpblmax_static: int | None = None,
    gwdo_active_i: Any | None = None,
    gwdo_active_j: Any | None = None,
    gwdo_row_j: Any | None = None,
    gwdo_row_i_start: Any | None = None,
    gwdo_row_len: Any | None = None,
    gwdo_row_max_len: int | None = None,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return WRF ``gwdo`` 3-D gravity-wave drag tendency updates."""

    del znw, p_top, itimestep

    if gwdo2d_fn is None:
        gwdo2d_fn = gwdo2d

    if _JAX_GWDO and gwdo2d_fn is gwdo2d and _uses_jax_array(u3d):
        if kpblmax_static is None:
            znu_host = np.asarray(jax.device_get(jnp.asarray(znu, dtype=jnp.float32)))
            kpblmax_static = None
            for k in range(int(kts), int(kte) + 1):
                if znu_host[int(k) - int(kms)] > np.float32(0.6):
                    kpblmax_static = k + 1
            if kpblmax_static is None:
                raise ValueError("WRF gwdo expects at least one znu(k) > 0.6 over kts:kte")
        common_args = (
            u3d,
            v3d,
            t3d,
            qv3d,
            p3d,
            p3di,
            pi3d,
            z,
            rublten,
            rvblten,
            dtaux3d,
            dtauy3d,
            dusfcg,
            dvsfcg,
            var2d,
            oc12d,
            oa2d1,
            oa2d2,
            oa2d3,
            oa2d4,
            ol2d1,
            ol2d2,
            ol2d3,
            ol2d4,
            sina,
            cosa,
            cp,
            g,
            rd,
            rv,
            ep1,
            pi,
            dt,
            dx,
            kpbl2d,
            rounding_token,
        )
        static_kwargs = dict(
            kpblmax=int(kpblmax_static),
            ids=ids,
            ide=ide,
            jds=jds,
            jde=jde,
            kds=kds,
            kde=kde,
            ims=ims,
            ime=ime,
            jms=jms,
            jme=jme,
            kms=kms,
            kme=kme,
            its=its,
            ite=ite,
            jts=jts,
            jte=jte,
            kts=kts,
            kte=kte,
        )
        if (
            _JAX_GWDO_ROW_RANGES
            and gwdo_row_j is not None
            and gwdo_row_i_start is not None
            and gwdo_row_len is not None
            and gwdo_row_max_len is not None
        ):
            row_kwargs = dict(static_kwargs)
            row_kwargs["row_max_len"] = int(gwdo_row_max_len)
            return _call_gwdo_kernel(
                _gwdo_jax_row_range_jit,
                (*common_args, gwdo_row_j, gwdo_row_i_start, gwdo_row_len),
                row_kwargs,
            )
        if _JAX_GWDO_ACTIVE_COLUMNS and gwdo_active_i is not None and gwdo_active_j is not None:
            return _call_gwdo_kernel(
                _gwdo_jax_active_jit,
                (*common_args, gwdo_active_i, gwdo_active_j),
                static_kwargs,
            )
        return _call_gwdo_kernel(_gwdo_jax_jit, common_args, static_kwargs)

    u3d_arr = np.asarray(u3d, dtype=np.float32)
    v3d_arr = np.asarray(v3d, dtype=np.float32)
    t3d_arr = np.asarray(t3d, dtype=np.float32)
    qv3d_arr = np.asarray(qv3d, dtype=np.float32)
    p3d_arr = np.asarray(p3d, dtype=np.float32)
    p3di_arr = np.asarray(p3di, dtype=np.float32)
    pi3d_arr = np.asarray(pi3d, dtype=np.float32)
    z_arr = np.asarray(z, dtype=np.float32)
    var2d_arr = np.asarray(var2d, dtype=np.float32)
    oc12d_arr = np.asarray(oc12d, dtype=np.float32)
    oa2d1_arr = np.asarray(oa2d1, dtype=np.float32)
    oa2d2_arr = np.asarray(oa2d2, dtype=np.float32)
    oa2d3_arr = np.asarray(oa2d3, dtype=np.float32)
    oa2d4_arr = np.asarray(oa2d4, dtype=np.float32)
    ol2d1_arr = np.asarray(ol2d1, dtype=np.float32)
    ol2d2_arr = np.asarray(ol2d2, dtype=np.float32)
    ol2d3_arr = np.asarray(ol2d3, dtype=np.float32)
    ol2d4_arr = np.asarray(ol2d4, dtype=np.float32)
    sina_arr = np.asarray(sina, dtype=np.float32)
    cosa_arr = np.asarray(cosa, dtype=np.float32)
    znu_arr = np.asarray(znu, dtype=np.float32)
    kpbl2d_arr = np.asarray(kpbl2d, dtype=np.int32)

    rublten_out = np.array(rublten, dtype=np.float32, copy=True, order="F")
    rvblten_out = np.array(rvblten, dtype=np.float32, copy=True, order="F")
    dtaux3d_out = np.array(dtaux3d, dtype=np.float32, copy=True, order="F")
    dtauy3d_out = np.array(dtauy3d, dtype=np.float32, copy=True, order="F")
    dusfcg_out = np.array(dusfcg, dtype=np.float32, copy=True, order="F")
    dvsfcg_out = np.array(dvsfcg, dtype=np.float32, copy=True, order="F")

    def fi(i: int) -> int:
        return int(i) - int(ims)

    def fj(j: int) -> int:
        return int(j) - int(jms)

    def fk(k: int) -> int:
        return int(k) - int(kms)

    def ti(i: int) -> int:
        return int(i) - int(its)

    def tk(k: int) -> int:
        return int(k) - int(kts)

    kpblmax = None
    for k in range(int(kts), int(kte) + 1):
        if znu_arr[fk(k)] > np.float32(0.6):
            kpblmax = k + 1
    if kpblmax is None:
        raise ValueError("WRF gwdo expects at least one znu(k) > 0.6 over kts:kte")

    nk = int(kte) - int(kts) + 1
    k0_full = fk(kts)
    k1_full = fk(kte) + 1
    ki1_full = fk(kte + 1) + 1
    full_i = np.arange(fi(its), fi(ite) + 1, dtype=np.intp)

    if _BATCH_GWDO_COLUMNS:
        full_j = np.arange(fj(jts), fj(jte) + 1, dtype=np.intp)
        ii_grid, jj_grid = np.meshgrid(full_i, full_j, indexing="ij")
        active_mask = var2d_arr[ii_grid, jj_grid] > np.float32(0.0)

        dtaux3d_out[full_i[:, None], k0_full:k1_full, full_j[None, :]] = np.float32(0.0)
        dtauy3d_out[full_i[:, None], k0_full:k1_full, full_j[None, :]] = np.float32(0.0)
        dusfcg_out[full_i[:, None], full_j[None, :]] = np.float32(0.0)
        dvsfcg_out[full_i[:, None], full_j[None, :]] = np.float32(0.0)

        if np.any(active_mask):
            active_i = ii_grid[active_mask]
            active_j = jj_grid[active_mask]
            n_active = int(active_i.size)
            level_idx = np.arange(k0_full, k1_full, dtype=np.intp)
            ilevel_idx = np.arange(k0_full, ki1_full, dtype=np.intp)

            pdh = np.asfortranarray(p3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]], dtype=np.float32)
            pdhi = np.asfortranarray(p3di_arr[active_i[:, None], ilevel_idx[None, :], active_j[:, None]], dtype=np.float32)
            delprsi = pdhi[:, :nk] - pdhi[:, 1 : nk + 1]

            sinv_col = sina_arr[active_i, active_j][:, None]
            cosv_col = cosa_arr[active_i, active_j][:, None]
            u_local = u3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]]
            v_local = v3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]]
            ugeo = np.asfortranarray(u_local * cosv_col - v_local * sinv_col, dtype=np.float32)
            vgeo = np.asfortranarray(u_local * sinv_col + v_local * cosv_col, dtype=np.float32)
            dudt = np.zeros((n_active, nk), dtype=np.float32, order="F")
            dvdt = np.zeros((n_active, nk), dtype=np.float32, order="F")

            oa4 = np.asfortranarray(
                np.stack(
                    (
                        oa2d1_arr[active_i, active_j],
                        oa2d2_arr[active_i, active_j],
                        oa2d3_arr[active_i, active_j],
                        oa2d4_arr[active_i, active_j],
                    ),
                    axis=1,
                ),
                dtype=np.float32,
            )
            ol4 = np.asfortranarray(
                np.stack(
                    (
                        ol2d1_arr[active_i, active_j],
                        ol2d2_arr[active_i, active_j],
                        ol2d3_arr[active_i, active_j],
                        ol2d4_arr[active_i, active_j],
                    ),
                    axis=1,
                ),
                dtype=np.float32,
            )

            dudt_out, dvdt_out, dtaux, dtauy, dusfc, dvsfc = gwdo2d_fn(
                dudt,
                dvdt,
                ugeo,
                vgeo,
                np.asfortranarray(t3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]], dtype=np.float32),
                np.asfortranarray(qv3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]], dtype=np.float32),
                delprsi,
                pdhi,
                pdh,
                np.asfortranarray(pi3d_arr[active_i[:, None], level_idx[None, :], active_j[:, None]], dtype=np.float32),
                np.asfortranarray(z_arr[active_i[:, None], level_idx[None, :], active_j[:, None]], dtype=np.float32),
                int(kpblmax),
                var2d_arr[active_i, active_j],
                oc12d_arr[active_i, active_j],
                oa4,
                ol4,
                g,
                cp,
                rd,
                rv,
                ep1,
                pi,
                dx,
                dt,
                kpbl2d_arr[active_i, active_j],
                1,
                1,
                n_active,
                1,
                1,
                1,
                nk,
                1,
                n_active,
                1,
                1,
                1,
                nk,
                1,
                n_active,
                1,
                1,
                1,
                nk,
            )
            dudt_arr = np.asarray(dudt_out, dtype=np.float32)
            dvdt_arr = np.asarray(dvdt_out, dtype=np.float32)
            dtaux_arr = np.asarray(dtaux, dtype=np.float32)
            dtauy_arr = np.asarray(dtauy, dtype=np.float32)
            dusfc_arr = np.asarray(dusfc, dtype=np.float32)
            dvsfc_arr = np.asarray(dvsfc, dtype=np.float32)

            rublten_out[active_i[:, None], level_idx[None, :], active_j[:, None]] = (
                rublten_out[active_i[:, None], level_idx[None, :], active_j[:, None]] + dudt_arr * cosv_col + dvdt_arr * sinv_col
            )
            rvblten_out[active_i[:, None], level_idx[None, :], active_j[:, None]] = (
                rvblten_out[active_i[:, None], level_idx[None, :], active_j[:, None]] - dudt_arr * sinv_col + dvdt_arr * cosv_col
            )
            dtaux3d_out[active_i[:, None], level_idx[None, :], active_j[:, None]] = dtaux_arr * cosv_col + dtauy_arr * sinv_col
            dtauy3d_out[active_i[:, None], level_idx[None, :], active_j[:, None]] = -dtaux_arr * sinv_col + dtauy_arr * cosv_col
            sinv = sina_arr[active_i, active_j]
            cosv = cosa_arr[active_i, active_j]
            dusfcg_out[active_i, active_j] = dusfc_arr * cosv + dvsfc_arr * sinv
            dvsfcg_out[active_i, active_j] = -dusfc_arr * sinv + dvsfc_arr * cosv

        return (
            jnp.asarray(rublten_out),
            jnp.asarray(rvblten_out),
            jnp.asarray(dtaux3d_out),
            jnp.asarray(dtauy3d_out),
            jnp.asarray(dusfcg_out),
            jnp.asarray(dvsfcg_out),
        )

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        dtaux3d_out[full_i, k0_full:k1_full, jj] = np.float32(0.0)
        dtauy3d_out[full_i, k0_full:k1_full, jj] = np.float32(0.0)
        dusfcg_out[full_i, jj] = np.float32(0.0)
        dvsfcg_out[full_i, jj] = np.float32(0.0)

        active_i = full_i[var2d_arr[full_i, jj] > np.float32(0.0)]
        if active_i.size == 0:
            continue

        n_active = int(active_i.size)
        pdh = np.asfortranarray(p3d_arr[active_i, k0_full:k1_full, jj], dtype=np.float32)
        pdhi = np.asfortranarray(p3di_arr[active_i, k0_full:ki1_full, jj], dtype=np.float32)
        delprsi = pdhi[:, :nk] - pdhi[:, 1 : nk + 1]

        sinv_col = sina_arr[active_i, jj][:, None]
        cosv_col = cosa_arr[active_i, jj][:, None]
        u_local = u3d_arr[active_i, k0_full:k1_full, jj]
        v_local = v3d_arr[active_i, k0_full:k1_full, jj]
        ugeo = np.asfortranarray(u_local * cosv_col - v_local * sinv_col, dtype=np.float32)
        vgeo = np.asfortranarray(u_local * sinv_col + v_local * cosv_col, dtype=np.float32)
        dudt = np.zeros((n_active, nk), dtype=np.float32, order="F")
        dvdt = np.zeros((n_active, nk), dtype=np.float32, order="F")

        oa4 = np.asfortranarray(
            np.stack(
                (
                    oa2d1_arr[active_i, jj],
                    oa2d2_arr[active_i, jj],
                    oa2d3_arr[active_i, jj],
                    oa2d4_arr[active_i, jj],
                ),
                axis=1,
            ),
            dtype=np.float32,
        )
        ol4 = np.asfortranarray(
            np.stack(
                (
                    ol2d1_arr[active_i, jj],
                    ol2d2_arr[active_i, jj],
                    ol2d3_arr[active_i, jj],
                    ol2d4_arr[active_i, jj],
                ),
                axis=1,
            ),
            dtype=np.float32,
        )

        dudt_out, dvdt_out, dtaux, dtauy, dusfc, dvsfc = gwdo2d_fn(
            dudt,
            dvdt,
            ugeo,
            vgeo,
            np.asfortranarray(t3d_arr[active_i, k0_full:k1_full, jj], dtype=np.float32),
            np.asfortranarray(qv3d_arr[active_i, k0_full:k1_full, jj], dtype=np.float32),
            delprsi,
            pdhi,
            pdh,
            np.asfortranarray(pi3d_arr[active_i, k0_full:k1_full, jj], dtype=np.float32),
            np.asfortranarray(z_arr[active_i, k0_full:k1_full, jj], dtype=np.float32),
            int(kpblmax),
            var2d_arr[active_i, jj],
            oc12d_arr[active_i, jj],
            oa4,
            ol4,
            g,
            cp,
            rd,
            rv,
            ep1,
            pi,
            dx,
            dt,
            kpbl2d_arr[active_i, jj],
            j,
            1,
            n_active,
            1,
            1,
            1,
            nk,
            1,
            n_active,
            1,
            1,
            1,
            nk,
            1,
            n_active,
            1,
            1,
            1,
            nk,
        )
        dudt_arr = np.asarray(dudt_out, dtype=np.float32)
        dvdt_arr = np.asarray(dvdt_out, dtype=np.float32)
        dtaux_arr = np.asarray(dtaux, dtype=np.float32)
        dtauy_arr = np.asarray(dtauy, dtype=np.float32)
        dusfc_arr = np.asarray(dusfc, dtype=np.float32)
        dvsfc_arr = np.asarray(dvsfc, dtype=np.float32)

        sinv_col = sina_arr[active_i, jj][:, None]
        cosv_col = cosa_arr[active_i, jj][:, None]
        rublten_out[active_i, k0_full:k1_full, jj] = (
            rublten_out[active_i, k0_full:k1_full, jj] + dudt_arr * cosv_col + dvdt_arr * sinv_col
        )
        rvblten_out[active_i, k0_full:k1_full, jj] = (
            rvblten_out[active_i, k0_full:k1_full, jj] - dudt_arr * sinv_col + dvdt_arr * cosv_col
        )
        dtaux3d_out[active_i, k0_full:k1_full, jj] = dtaux_arr * cosv_col + dtauy_arr * sinv_col
        dtauy3d_out[active_i, k0_full:k1_full, jj] = -dtaux_arr * sinv_col + dtauy_arr * cosv_col
        sinv = sina_arr[active_i, jj]
        cosv = cosa_arr[active_i, jj]
        dusfcg_out[active_i, jj] = dusfc_arr * cosv + dvsfc_arr * sinv
        dvsfcg_out[active_i, jj] = -dusfc_arr * sinv + dvsfc_arr * cosv

    return (
        jnp.asarray(rublten_out),
        jnp.asarray(rvblten_out),
        jnp.asarray(dtaux3d_out),
        jnp.asarray(dtauy3d_out),
        jnp.asarray(dusfcg_out),
        jnp.asarray(dvsfcg_out),
    )


def _take_column_level(field: Any, level_fortran: Any) -> Any:
    idx = jnp.clip(jnp.asarray(level_fortran, dtype=jnp.int32) - 1, 0, int(field.shape[1]) - 1)
    return jnp.take_along_axis(field, idx[:, None], axis=1)[:, 0]


def _take_level_1d(field: Any, level_fortran: Any) -> Any:
    idx = jnp.clip(jnp.asarray(level_fortran, dtype=jnp.int32) - 1, 0, int(field.shape[0]) - 1)
    return field[idx]


def _gwdo1d_jax_core(
    dudt: Any,
    dvdt: Any,
    u1: Any,
    v1: Any,
    t1: Any,
    q1: Any,
    del_: Any,
    prsi: Any,
    prsl: Any,
    prslk: Any,
    zl: Any,
    *,
    kpblmax: int,
    var: Any,
    oc1: Any,
    oa4: Any,
    ol4: Any,
    g_: Any,
    cp_: Any,
    rd_: Any,
    fv_: Any,
    pi_: Any,
    dxmeter: Any,
    deltim: Any,
    kpbl: Any,
    return_diagnostics: bool = False,
    rounding_token: Any | None = None,
) -> tuple[Any, ...]:
    f32 = jnp.float32
    dudt_out = jnp.asarray(dudt, dtype=f32)
    dvdt_out = jnp.asarray(dvdt, dtype=f32)
    u1 = jnp.asarray(u1, dtype=f32)
    v1 = jnp.asarray(v1, dtype=f32)
    t1 = jnp.asarray(t1, dtype=f32)
    q1 = jnp.asarray(q1, dtype=f32)
    del_ = jnp.asarray(del_, dtype=f32)
    prsi = jnp.asarray(prsi, dtype=f32)
    prsl = jnp.asarray(prsl, dtype=f32)
    prslk = jnp.asarray(prslk, dtype=f32)
    zl = jnp.asarray(zl, dtype=f32)
    var = jnp.asarray(var, dtype=f32)
    oc1 = jnp.asarray(oc1, dtype=f32)
    oa4 = jnp.asarray(oa4, dtype=f32)
    ol4 = jnp.asarray(ol4, dtype=f32)

    nk = int(u1.shape[0])
    levels = jnp.arange(1, nk + 1, dtype=jnp.int32)
    levels_ip1 = jnp.arange(1, nk + 2, dtype=jnp.int32)

    ric = f32(0.25)
    dw2min = f32(1.0)
    rimin = f32(-100.0)
    bnv2min = f32(1.0e-5)
    efmin = f32(0.0)
    efmax = f32(10.0)
    gmax = f32(1.0)
    veleps = f32(1.0)
    frc = f32(1.0)
    ce = f32(0.8)
    cg = f32(0.5)
    kpblmin = 2
    mdir = 8
    nwdir = jnp.asarray([6, 7, 5, 8, 2, 3, 1, 4], dtype=jnp.int32)
    frmax = f32(10.0)
    olmin = f32(1.0e-5)
    odmin = f32(0.1)
    odmax = f32(10.0)

    g_ = f32(g_)
    cp_ = f32(cp_)
    rd_ = f32(rd_)
    fv_ = f32(fv_)
    pi_ = f32(pi_)
    dxmeter = f32(dxmeter)
    deltim = f32(deltim)

    fdir = f32(float(mdir)) / (f32(2.0) * pi_)
    sqrt2dx = sqrt_for_wrf(dxmeter * dxmeter + dxmeter * dxmeter)
    dxy4 = jnp.asarray([dxmeter, dxmeter, sqrt2dx, sqrt2dx], dtype=f32)
    dxy4p = jnp.asarray([dxmeter, dxmeter, sqrt2dx, sqrt2dx], dtype=f32)
    cleff = dxmeter

    virtual_factor = _real_add(
        f32(1.0),
        _real_mul(fv_, q1, rounding_token),
        rounding_token,
    )
    vtj = _real_mul(t1, virtual_factor, rounding_token)
    vtk = _real_div(vtj, prslk, rounding_token)
    rho = _real_div(
        _real_mul(
            _real_div(f32(1.0), rd_, rounding_token),
            prsl,
            rounding_token,
        ),
        vtj,
        rounding_token,
    )

    zlowtop = f32(2.0) * var
    scan_levels = jnp.arange(2, nk + 1, dtype=jnp.int32)
    scan_z = zl[1:nk] - zl[0]
    lowtop_mask = scan_z >= zlowtop
    first_lowtop_pos = jnp.argmax(lowtop_mask)
    first_lowtop_level = jnp.take(scan_levels, first_lowtop_pos)
    klowtop = jnp.where(jnp.any(lowtop_mask), first_lowtop_level + 1, 0).astype(jnp.int32)
    kbl = jnp.clip(klowtop, kpblmin, int(kpblmax)).astype(jnp.int32)

    prsi_kbl = _take_level_1d(prsi, kbl)
    prsl_kbl = _take_level_1d(prsl, kbl)
    delks = _real_store(f32(1.0) / (prsi[0] - prsi_kbl))
    delks1 = _real_store(f32(1.0) / (prsl[0] - prsl_kbl))

    ubar, vbar, rhobar = _low_level_average(
        u1, v1, rho, del_, delks, kbl, levels=int(kpblmax),
    )
    ubar, vbar, rhobar = jax.lax.optimization_barrier((ubar, vbar, rhobar))

    wdir = _real_store(jnp.arctan2(ubar, vbar) + pi_)
    idir = (jnp.floor(fdir * wdir + f32(0.5)).astype(jnp.int32) % mdir) + 1
    nwd = nwdir[idir - 1]
    quadrant = ((nwd - 1) % 4).astype(jnp.int32)
    sign = (1 - 2 * ((nwd - 1) // 4)).astype(f32)
    oa = sign * oa4[quadrant]
    ol = ol4[quadrant]
    ol4p = ol4[jnp.asarray([1, 0, 3, 2], dtype=jnp.int32)]
    olp = ol4p[quadrant]
    od = _real_store(jnp.clip(olp / jnp.maximum(ol, olmin), odmin, odmax))
    dxy = dxy4[quadrant]
    dxyp = dxy4p[quadrant]

    ti_local = _real_div(
        f32(2.0),
        _real_add(t1[:-1], t1[1:], rounding_token),
        rounding_token,
    )
    rdz = _real_div(
        f32(1.0),
        _real_sub(zl[1:], zl[:-1], rounding_token),
        rounding_token,
    )
    du = _real_sub(u1[:-1], u1[1:], rounding_token)
    dv = _real_sub(v1[:-1], v1[1:], rounding_token)
    dw2 = _real_add(
        _real_mul(du, du, rounding_token),
        _real_mul(dv, dv, rounding_token),
        rounding_token,
    )
    shr2 = _real_mul(
        _real_mul(jnp.maximum(dw2, dw2min), rdz, rounding_token),
        rdz,
        rounding_token,
    )
    vtj_delta = _real_sub(vtj[1:], vtj[:-1], rounding_token)
    buoyancy_term = _real_add(
        _real_div(g_, cp_, rounding_token),
        _real_mul(rdz, vtj_delta, rounding_token),
        rounding_token,
    )
    bvf2 = _real_mul(
        _real_mul(g_, buoyancy_term, rounding_token),
        ti_local,
        rounding_token,
    )
    usqj_values = jnp.maximum(_real_div(bvf2, shr2, rounding_token), rimin)
    usqj = jnp.zeros((nk,), dtype=f32).at[: nk - 1].set(usqj_values)
    bnv2_numerator = _real_mul(
        _real_mul(_real_mul(f32(2.0), g_), rdz),
        _real_sub(vtk[1:], vtk[:-1]),
    )
    bnv2_values = _real_div(bnv2_numerator, _real_add(vtk[1:], vtk[:-1]))
    bnv2 = jnp.zeros((nk,), dtype=f32).at[: nk - 1].set(bnv2_values)
    usqj, bnv2 = jax.lax.optimization_barrier((usqj, bnv2))

    low_wind_squared = _real_add(
        _real_mul(ubar, ubar, rounding_token),
        _real_mul(vbar, vbar, rounding_token),
        rounding_token,
    )
    ulow = jnp.maximum(
        _real_store(sqrt_for_wrf(low_wind_squared), rounding_token),
        f32(1.0),
    )
    rulow = _real_div(f32(1.0), ulow, rounding_token)
    velco_u = _real_mul(
        _real_add(u1[:-1], u1[1:], rounding_token),
        ubar,
        rounding_token,
    )
    velco_v = _real_mul(
        _real_add(v1[:-1], v1[1:], rounding_token),
        vbar,
        rounding_token,
    )
    velco_core = _real_mul(
        f32(0.5),
        _real_add(velco_u, velco_v, rounding_token),
        rounding_token,
    )
    velco_core = _real_mul(velco_core, rulow, rounding_token)
    velco_core = jnp.where((velco_core < veleps) & (velco_core > f32(0.0)), veleps, velco_core)
    velco_values = velco_core
    velco = jnp.zeros((nk,), dtype=f32).at[: nk - 1].set(velco_core)
    ulow, rulow, velco = jax.lax.optimization_barrier((ulow, rulow, velco))

    ldrag = velco[0] <= f32(0.0)
    drag_mask = (levels[: int(kpblmax)] >= kpblmin) & (levels[: int(kpblmax)] < kbl)
    ldrag = ldrag | jnp.any(drag_mask & (velco[: int(kpblmax)] <= f32(0.0)))

    wtkbj = _real_mul(
        _real_sub(prsl[0], prsl[1], rounding_token),
        delks1,
        rounding_token,
    )
    bnv2_first = _real_mul(wtkbj, bnv2[0], rounding_token)
    usqj_first = _real_mul(wtkbj, usqj[0], rounding_token)
    avg2_mask = (levels[: int(kpblmax)] >= kpblmin) & (levels[: int(kpblmax)] < kbl)

    bnv2_first, usqj_first = _stability_average(
        bnv2_first, usqj_first, prsl, bnv2, usqj, delks1, kbl,
        levels=int(kpblmax), rounding_token=rounding_token,
    )
    bnv2 = bnv2.at[0].set(bnv2_first)
    usqj = usqj.at[0].set(usqj_first)
    bnv2, usqj = jax.lax.optimization_barrier((bnv2, usqj))

    ldrag = ldrag | (bnv2[0] <= f32(0.0)) | (ulow == f32(1.0)) | (var <= f32(0.0))
    usqj = usqj.at[: int(kpblmax)].set(jnp.where(avg2_mask, usqj[0], usqj[: int(kpblmax)]))

    active = ~ldrag
    bnv = jnp.where(active, clipped_sqrt_for_wrf(bnv2[0]), f32(0.0))
    fr = jnp.minimum(bnv * rulow * var * od, frmax)
    xn = jnp.where(active, ubar * rulow, f32(0.0))
    yn = jnp.where(active, vbar * rulow, f32(0.0))
    bnv, fr, xn, yn = jax.lax.optimization_barrier((bnv, fr, xn, yn))

    efact_base = jnp.maximum(
        _real_add(oa, f32(2.0), rounding_token),
        f32(1.0e-20),
    )
    efact_exponent = _real_div(
        _real_mul(ce, fr, rounding_token),
        frc,
        rounding_token,
    )
    efact = _real_store(jnp.power(efact_base, efact_exponent), rounding_token)
    efact = jnp.clip(efact, efmin, efmax)
    coefm = _real_store(
        jnp.power(
            _real_add(f32(1.0), ol, rounding_token),
            _real_add(oa, f32(1.0), rounding_token),
        ),
        rounding_token,
    )
    xlinv = _real_div(coefm, cleff, rounding_token)
    tem = _real_mul(
        _real_mul(fr, fr, rounding_token),
        oc1,
        rounding_token,
    )
    safe_bnv = jnp.where(bnv > f32(0.0), bnv, f32(1.0))
    gfobnv = _real_div(
        _real_mul(gmax, tem, rounding_token),
        _real_mul(
            _real_add(tem, cg, rounding_token),
            safe_bnv,
            rounding_token,
        ),
        rounding_token,
    )
    taub_active = _real_mul(xlinv, rhobar, rounding_token)
    taub_active = _real_mul(taub_active, ulow, rounding_token)
    taub_active = _real_mul(taub_active, ulow, rounding_token)
    taub_active = _real_mul(taub_active, ulow, rounding_token)
    taub_active = _real_mul(taub_active, gfobnv, rounding_token)
    taub_active = _real_mul(taub_active, efact, rounding_token)
    taub = jnp.where(active, taub_active, f32(0.0))
    coefm = jnp.where(active, coefm, f32(0.0))
    efact, coefm, xlinv, gfobnv, taub = jax.lax.optimization_barrier(
        (efact, coefm, xlinv, gfobnv, taub)
    )

    taup = jnp.zeros((nk + 1,), dtype=f32)
    taup_mask = (levels_ip1 <= int(kpblmax)) & (levels_ip1 <= kbl)
    taup = jnp.where(taup_mask, taub, taup)

    taup = _critical_stress_profile(taup, kbl, usqj, velco, bnv2, rho, active, coefm, dxy)
    taup = jax.lax.optimization_barrier(taup)
    kblk, zblk = _blocking_level(kbl, bnv2, zl, del_, rho, u1, v1, active, g_)
    kblk, zblk = jax.lax.optimization_barrier((kblk, zblk))

    has_block = active & (kblk != 0)
    fbdcd = jnp.maximum(
        _real_sub(
            f32(2.0),
            _real_div(f32(1.0), od, rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    taufb0 = _real_mul(f32(0.5), rhobar, rounding_token)
    taufb0 = _real_mul(taufb0, coefm, rounding_token)
    taufb0 = _real_div(
        taufb0,
        _real_mul(dxmeter, dxmeter, rounding_token),
        rounding_token,
    )
    taufb0 = _real_mul(taufb0, fbdcd, rounding_token)
    taufb0 = _real_mul(taufb0, dxyp, rounding_token)
    taufb0 = _real_mul(taufb0, olp, rounding_token)
    taufb0 = _real_mul(taufb0, zblk, rounding_token)
    taufb0 = _real_mul(
        taufb0,
        _real_mul(ulow, ulow, rounding_token),
        rounding_token,
    )
    taufb0 = jnp.where(has_block, taufb0, f32(0.0))
    tautem = _real_div(
        taufb0,
        jnp.maximum((kblk - 1).astype(f32), f32(1.0)),
        rounding_token,
    )
    taufb = _blocked_stress_profile(
        taufb0, has_block, kblk, tautem, levels=nk, rounding_token=rounding_token,
    )
    taup = _real_add(taup, taufb, rounding_token)

    taud = _real_div(
        _real_mul(
            _real_sub(taup[1:], taup[:-1], rounding_token),
            g_,
            rounding_token,
        ),
        del_,
        rounding_token,
    )
    dtfac_width = max(int(kpblmax) - 1, 0)
    dtfac_mask = (levels[:dtfac_width] <= kbl) & (taud[:dtfac_width] != f32(0.0))
    safe_taud = jnp.where(taud[:dtfac_width] != f32(0.0), taud[:dtfac_width], f32(1.0))
    dtfac_candidates = jnp.where(dtfac_mask, jnp.abs(velco[:dtfac_width] / (deltim * safe_taud)), jnp.inf)
    dtfac_min = jnp.min(dtfac_candidates)
    dtfac = jnp.minimum(f32(1.0), dtfac_min)
    dtfac = jnp.where(jnp.isfinite(dtfac), dtfac, f32(1.0))

    taud = _real_mul(taud, dtfac, rounding_token)
    dtaux2d = _real_mul(taud, xn, rounding_token)
    dtauy2d = _real_mul(taud, yn, rounding_token)
    dudt_out = _real_add(dudt_out, dtaux2d, rounding_token)
    dvdt_out = _real_add(dvdt_out, dtauy2d, rounding_token)
    dusfc, dvsfc = _surface_stress_integral(dtaux2d, dtauy2d, del_, rounding_token)
    reciprocal_g = _real_div(f32(-1.0), g_, rounding_token)
    dusfc = _real_mul(reciprocal_g, dusfc, rounding_token)
    dvsfc = _real_mul(reciprocal_g, dvsfc, rounding_token)

    outputs = (dudt_out, dvdt_out, dtaux2d, dtauy2d, dusfc, dvsfc)
    if not return_diagnostics:
        return outputs
    diagnostics = (
        vtj,
        vtk,
        rho,
        kbl,
        delks,
        delks1,
        ubar,
        vbar,
        rhobar,
        idir,
        oa,
        ol,
        olp,
        od,
        ti_local,
        rdz,
        du,
        dv,
        dw2,
        shr2,
        vtj_delta,
        buoyancy_term,
        bvf2,
        usqj_values,
        bnv2_numerator,
        bnv2_values,
        usqj,
        bnv2,
        ulow,
        rulow,
        velco_u,
        velco_v,
        velco_values,
        velco,
        ldrag,
        bnv,
        fr,
        xn,
        yn,
        efact,
        coefm,
        xlinv,
        gfobnv,
        taub,
        taup,
        kblk,
        zblk,
        taufb0,
        dtfac,
    )
    return (*outputs, diagnostics)


def _gwdo2d_jax_vmap_core(
    dudt: Any,
    dvdt: Any,
    u1: Any,
    v1: Any,
    t1: Any,
    q1: Any,
    del_: Any,
    prsi: Any,
    prsl: Any,
    prslk: Any,
    zl: Any,
    *,
    kpblmax: int,
    var: Any,
    oc1: Any,
    oa4: Any,
    ol4: Any,
    g_: Any,
    cp_: Any,
    rd_: Any,
    fv_: Any,
    pi_: Any,
    dxmeter: Any,
    deltim: Any,
    kpbl: Any,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    return jax.vmap(
        lambda dudt_i, dvdt_i, u1_i, v1_i, t1_i, q1_i, del_i, prsi_i, prsl_i, prslk_i, zl_i, var_i, oc1_i, oa4_i, ol4_i, kpbl_i: _gwdo1d_jax_core(
            dudt_i,
            dvdt_i,
            u1_i,
            v1_i,
            t1_i,
            q1_i,
            del_i,
            prsi_i,
            prsl_i,
            prslk_i,
            zl_i,
            kpblmax=kpblmax,
            var=var_i,
            oc1=oc1_i,
            oa4=oa4_i,
            ol4=ol4_i,
            g_=g_,
            cp_=cp_,
            rd_=rd_,
            fv_=fv_,
            pi_=pi_,
            dxmeter=dxmeter,
            deltim=deltim,
            kpbl=kpbl_i,
            rounding_token=rounding_token,
        )
    )(dudt, dvdt, u1, v1, t1, q1, del_, prsi, prsl, prslk, zl, var, oc1, oa4, ol4, kpbl)


def _gwdo2d_jax_core(
    dudt: Any,
    dvdt: Any,
    u1: Any,
    v1: Any,
    t1: Any,
    q1: Any,
    del_: Any,
    prsi: Any,
    prsl: Any,
    prslk: Any,
    zl: Any,
    *,
    kpblmax: int,
    var: Any,
    oc1: Any,
    oa4: Any,
    ol4: Any,
    g_: Any,
    cp_: Any,
    rd_: Any,
    fv_: Any,
    pi_: Any,
    dxmeter: Any,
    deltim: Any,
    kpbl: Any,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    f32 = jnp.float32
    dudt_out = jnp.asarray(dudt, dtype=f32)
    dvdt_out = jnp.asarray(dvdt, dtype=f32)
    u1 = jnp.asarray(u1, dtype=f32)
    v1 = jnp.asarray(v1, dtype=f32)
    t1 = jnp.asarray(t1, dtype=f32)
    q1 = jnp.asarray(q1, dtype=f32)
    del_ = jnp.asarray(del_, dtype=f32)
    prsi = jnp.asarray(prsi, dtype=f32)
    prsl = jnp.asarray(prsl, dtype=f32)
    prslk = jnp.asarray(prslk, dtype=f32)
    zl = jnp.asarray(zl, dtype=f32)
    var = jnp.asarray(var, dtype=f32)
    oc1 = jnp.asarray(oc1, dtype=f32)
    oa4 = jnp.asarray(oa4, dtype=f32)
    ol4 = jnp.asarray(ol4, dtype=f32)

    ncol = int(u1.shape[0])
    nk = int(u1.shape[1])
    levels = jnp.arange(1, nk + 1, dtype=jnp.int32)
    levels_ip1 = jnp.arange(1, nk + 2, dtype=jnp.int32)
    level_mask = levels[None, :]

    ric = f32(0.25)
    dw2min = f32(1.0)
    rimin = f32(-100.0)
    bnv2min = f32(1.0e-5)
    efmin = f32(0.0)
    efmax = f32(10.0)
    xl = f32(4.0e4)
    gmax = f32(1.0)
    veleps = f32(1.0)
    frc = f32(1.0)
    ce = f32(0.8)
    cg = f32(0.5)
    kpblmin = 2
    mdir = 8
    nwdir = jnp.asarray([6, 7, 5, 8, 2, 3, 1, 4], dtype=jnp.int32)
    frmax = f32(10.0)
    olmin = f32(1.0e-5)
    odmin = f32(0.1)
    odmax = f32(10.0)

    g_ = f32(g_)
    cp_ = f32(cp_)
    rd_ = f32(rd_)
    fv_ = f32(fv_)
    pi_ = f32(pi_)
    dxmeter = f32(dxmeter)
    deltim = f32(deltim)

    if _JAX_GWDO_VMAP_COLUMNS:
        return _gwdo2d_jax_vmap_core(
            dudt_out,
            dvdt_out,
            u1,
            v1,
            t1,
            q1,
            del_,
            prsi,
            prsl,
            prslk,
            zl,
            kpblmax=int(kpblmax),
            var=var,
            oc1=oc1,
            oa4=oa4,
            ol4=ol4,
            g_=g_,
            cp_=cp_,
            rd_=rd_,
            fv_=fv_,
            pi_=pi_,
            dxmeter=dxmeter,
            deltim=deltim,
            kpbl=kpbl,
            rounding_token=rounding_token,
        )

    fdir = f32(float(mdir)) / (f32(2.0) * pi_)
    sqrt2dx = sqrt_for_wrf(dxmeter * dxmeter + dxmeter * dxmeter)
    dxy4 = jnp.broadcast_to(jnp.asarray([dxmeter, dxmeter, sqrt2dx, sqrt2dx], dtype=f32), (ncol, 4))
    dxy4p = jnp.broadcast_to(jnp.asarray([dxmeter, dxmeter, sqrt2dx, sqrt2dx], dtype=f32), (ncol, 4))
    cleff = jnp.full((ncol,), dxmeter, dtype=f32)

    vtj = t1 * (f32(1.0) + fv_ * q1)
    vtk = vtj / prslk
    rho = (f32(1.0) / rd_) * prsl / vtj

    zlowtop = f32(2.0) * var
    scan_levels = jnp.arange(2, nk + 1, dtype=jnp.int32)
    scan_z = zl[:, 1:nk] - zl[:, 0:1]
    lowtop_mask = scan_z >= zlowtop[:, None]
    first_lowtop_pos = jnp.argmax(lowtop_mask, axis=1)
    first_lowtop_level = jnp.take(scan_levels, first_lowtop_pos)
    klowtop = jnp.where(jnp.any(lowtop_mask, axis=1), first_lowtop_level + 1, 0).astype(jnp.int32)
    kbl = jnp.clip(klowtop, kpblmin, int(kpblmax)).astype(jnp.int32)

    prsi_kbl = _take_column_level(prsi, kbl)
    prsl_kbl = _take_column_level(prsl, kbl)
    delks = f32(1.0) / (prsi[:, 0] - prsi_kbl)
    delks1 = f32(1.0) / (prsl[:, 0] - prsl_kbl)

    pbl_avg_mask = (level_mask <= int(kpblmax)) & (level_mask < kbl[:, None])
    rcsks = del_ * delks[:, None]
    rdelks = del_ * delks[:, None]
    ubar = jnp.sum(jnp.where(pbl_avg_mask, rcsks * u1, f32(0.0)), axis=1)
    vbar = jnp.sum(jnp.where(pbl_avg_mask, rcsks * v1, f32(0.0)), axis=1)
    rhobar = jnp.sum(jnp.where(pbl_avg_mask, rdelks * rho, f32(0.0)), axis=1)

    wdir = jnp.arctan2(ubar, vbar) + pi_
    idir = (jnp.floor(fdir * wdir + f32(0.5)).astype(jnp.int32) % mdir) + 1
    nwd = nwdir[idir - 1]
    quadrant = ((nwd - 1) % 4).astype(jnp.int32)
    sign = (1 - 2 * ((nwd - 1) // 4)).astype(f32)
    oa = sign * jnp.take_along_axis(oa4, quadrant[:, None], axis=1)[:, 0]
    ol = jnp.take_along_axis(ol4, quadrant[:, None], axis=1)[:, 0]
    ol4p = ol4[:, jnp.asarray([1, 0, 3, 2], dtype=jnp.int32)]
    olp = jnp.take_along_axis(ol4p, quadrant[:, None], axis=1)[:, 0]
    od = jnp.clip(olp / jnp.maximum(ol, olmin), odmin, odmax)
    dxy = jnp.take_along_axis(dxy4, quadrant[:, None], axis=1)[:, 0]
    dxyp = jnp.take_along_axis(dxy4p, quadrant[:, None], axis=1)[:, 0]

    ti_local = f32(2.0) / (t1[:, :-1] + t1[:, 1:])
    rdz = f32(1.0) / (zl[:, 1:] - zl[:, :-1])
    du = u1[:, :-1] - u1[:, 1:]
    dv = v1[:, :-1] - v1[:, 1:]
    dw2 = du * du + dv * dv
    shr2 = jnp.maximum(dw2, dw2min) * rdz * rdz
    bvf2 = g_ * (g_ / cp_ + rdz * (vtj[:, 1:] - vtj[:, :-1])) * ti_local
    usqj = jnp.zeros((ncol, nk), dtype=f32).at[:, : nk - 1].set(jnp.maximum(bvf2 / shr2, rimin))
    bnv2 = jnp.zeros((ncol, nk), dtype=f32).at[:, : nk - 1].set(
        f32(2.0) * g_ * rdz * (vtk[:, 1:] - vtk[:, :-1]) / (vtk[:, 1:] + vtk[:, :-1])
    )

    ulow = jnp.maximum(sqrt_for_wrf(ubar * ubar + vbar * vbar), f32(1.0))
    rulow = f32(1.0) / ulow
    velco_core = f32(0.5) * ((u1[:, :-1] + u1[:, 1:]) * ubar[:, None] + (v1[:, :-1] + v1[:, 1:]) * vbar[:, None])
    velco_core = velco_core * rulow[:, None]
    velco_core = jnp.where((velco_core < veleps) & (velco_core > f32(0.0)), veleps, velco_core)
    velco = jnp.zeros((ncol, nk), dtype=f32).at[:, : nk - 1].set(velco_core)

    ldrag = velco[:, 0] <= f32(0.0)
    drag_mask = (level_mask[:, : int(kpblmax)] >= kpblmin) & (level_mask[:, : int(kpblmax)] < kbl[:, None])
    ldrag = ldrag | jnp.any(drag_mask & (velco[:, : int(kpblmax)] <= f32(0.0)), axis=1)

    wtkbj = (prsl[:, 0] - prsl[:, 1]) * delks1
    bnv2 = bnv2.at[:, 0].set(wtkbj * bnv2[:, 0])
    usqj = usqj.at[:, 0].set(wtkbj * usqj[:, 0])
    avg2_mask = (level_mask[:, : int(kpblmax)] >= kpblmin) & (level_mask[:, : int(kpblmax)] < kbl[:, None])
    rdelks2 = (prsl[:, : int(kpblmax)] - prsl[:, 1 : int(kpblmax) + 1]) * delks1[:, None]
    bnv2_first = bnv2[:, 0] + jnp.sum(jnp.where(avg2_mask, bnv2[:, : int(kpblmax)] * rdelks2, f32(0.0)), axis=1)
    usqj_first = usqj[:, 0] + jnp.sum(jnp.where(avg2_mask, usqj[:, : int(kpblmax)] * rdelks2, f32(0.0)), axis=1)
    bnv2 = bnv2.at[:, 0].set(bnv2_first)
    usqj = usqj.at[:, 0].set(usqj_first)

    ldrag = ldrag | (bnv2[:, 0] <= f32(0.0)) | (ulow == f32(1.0)) | (var <= f32(0.0))
    usqj = usqj.at[:, : int(kpblmax)].set(jnp.where(avg2_mask, usqj[:, 0:1], usqj[:, : int(kpblmax)]))

    active = ~ldrag
    bnv = jnp.where(active, clipped_sqrt_for_wrf(bnv2[:, 0]), f32(0.0))
    fr = jnp.minimum(bnv * rulow * var * od, frmax)
    xn = jnp.where(active, ubar * rulow, f32(0.0))
    yn = jnp.where(active, vbar * rulow, f32(0.0))

    efact = jnp.power(jnp.maximum(oa + f32(2.0), f32(1.0e-20)), ce * fr / frc)
    efact = jnp.clip(efact, efmin, efmax)
    coefm = jnp.power(f32(1.0) + ol, oa + f32(1.0))
    xlinv = coefm / cleff
    tem = fr * fr * oc1
    safe_bnv = jnp.where(bnv > f32(0.0), bnv, f32(1.0))
    gfobnv = gmax * tem / ((tem + cg) * safe_bnv)
    taub_active = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact
    taub = jnp.where(active, taub_active, f32(0.0))
    coefm = jnp.where(active, coefm, f32(0.0))

    taup = jnp.zeros((ncol, nk + 1), dtype=f32)
    taup_mask = (levels_ip1[None, :] <= int(kpblmax)) & (levels_ip1[None, :] <= kbl[:, None])
    taup = jnp.where(taup_mask, taub[:, None], taup)

    icrilv0 = jnp.zeros((ncol,), dtype=bool)
    brvf0 = jnp.zeros((ncol,), dtype=f32)

    def critical_body(k: int, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        taup_c, icrilv_c, brvf_c = carry
        idx = k - 1
        kp1_idx = k
        ge_kbl = k >= kbl
        icrilv_new = jnp.where(ge_kbl, icrilv_c | (usqj[:, idx] < ric) | (velco[:, idx] <= f32(0.0)), icrilv_c)
        brvf_new = jnp.where(ge_kbl, sqrt_for_wrf(jnp.maximum(bnv2[:, idx], bnv2min)), brvf_c)
        can_update = ge_kbl & active & (~icrilv_new) & (taup_c[:, idx] > f32(0.0))
        safe_vel = jnp.where(jnp.abs(velco[:, idx]) > f32(0.0), velco[:, idx], f32(1.0))
        temv = f32(1.0) / safe_vel
        tem1 = coefm / dxy * (rho[:, kp1_idx] + rho[:, idx]) * brvf_new * velco[:, idx] * f32(0.5)
        safe_tem1 = jnp.where(tem1 > f32(0.0), tem1, f32(1.0))
        hd = clipped_sqrt_for_wrf(taup_c[:, idx] / safe_tem1)
        fro = brvf_new * hd * temv
        tem2 = clipped_sqrt_for_wrf(usqj[:, idx])
        safe_tem2 = jnp.where(tem2 > f32(0.0), tem2, f32(1.0))
        tem_local = f32(1.0) + safe_tem2 * fro
        rim = usqj[:, idx] * (f32(1.0) - fro) / (tem_local * tem_local)
        temc = f32(2.0) + f32(1.0) / safe_tem2
        hd_break = velco[:, idx] * (f32(2.0) * clipped_sqrt_for_wrf(temc) - temc) / jnp.where(
            brvf_new > f32(0.0), brvf_new, f32(1.0)
        )
        tau_break = tem1 * hd_break * hd_break
        tau_next = jnp.where(rim <= ric, tau_break, taup_c[:, idx])
        taup_c = taup_c.at[:, kp1_idx].set(jnp.where(can_update, tau_next, taup_c[:, kp1_idx]))
        return taup_c, icrilv_new, brvf_new

    taup, icrilv0, brvf0 = jax.lax.fori_loop(kpblmin, nk, critical_body, (taup, icrilv0, brvf0))

    del icrilv0, brvf0

    kblk = jnp.zeros((ncol,), dtype=jnp.int32)
    fbdpe = jnp.zeros((ncol,), dtype=f32)
    zblk = jnp.zeros((ncol,), dtype=f32)
    zl_kbl = _take_column_level(zl, kbl)

    def blocking_body(offset: int, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        kblk_c, fbdpe_c, zblk_c = carry
        k = nk - offset
        idx = k - 1
        update = active & (kblk_c == 0) & (k <= kbl)
        fbdpe_next = fbdpe_c + jnp.where(update, bnv2[:, idx] * (zl_kbl - zl[:, idx]) * del_[:, idx] / g_ / rho[:, idx], f32(0.0))
        fbdke = f32(0.5) * (u1[:, idx] * u1[:, idx] + v1[:, idx] * v1[:, idx])
        block = update & (fbdpe_next >= fbdke)
        kblk_new = jnp.minimum(jnp.full((ncol,), k, dtype=jnp.int32), kbl)
        kblk_c = jnp.where(block, kblk_new, kblk_c)
        zblk_c = jnp.where(block, _take_column_level(zl, kblk_new) - zl[:, 0], zblk_c)
        return kblk_c, fbdpe_next, zblk_c

    kblk, fbdpe, zblk = jax.lax.fori_loop(0, nk - kpblmin + 1, blocking_body, (kblk, fbdpe, zblk))

    has_block = active & (kblk != 0)
    fbdcd = jnp.maximum(f32(2.0) - f32(1.0) / od, f32(0.0))
    taufb0 = f32(0.5) * rhobar * coefm / (dxmeter * dxmeter) * fbdcd * dxyp * olp * zblk * ulow * ulow
    tautem = taufb0 / jnp.maximum((kblk - 1).astype(f32), f32(1.0))
    taufb = jnp.where(
        (levels_ip1[None, :] <= kblk[:, None]) & has_block[:, None],
        taufb0[:, None] - (levels_ip1[None, :].astype(f32) - f32(1.0)) * tautem[:, None],
        f32(0.0),
    )
    taup = taup + taufb

    taud = (taup[:, 1:] - taup[:, :-1]) * g_ / del_
    dtfac_mask = (level_mask[:, : max(int(kpblmax) - 1, 0)] <= kbl[:, None]) & (
        taud[:, : max(int(kpblmax) - 1, 0)] != f32(0.0)
    )
    safe_taud = jnp.where(taud[:, : max(int(kpblmax) - 1, 0)] != f32(0.0), taud[:, : max(int(kpblmax) - 1, 0)], f32(1.0))
    dtfac_candidates = jnp.where(
        dtfac_mask,
        jnp.abs(velco[:, : max(int(kpblmax) - 1, 0)] / (deltim * safe_taud)),
        jnp.inf,
    )
    dtfac_min = jnp.min(dtfac_candidates, axis=1)
    dtfac = jnp.minimum(jnp.ones((ncol,), dtype=f32), dtfac_min)
    dtfac = jnp.where(jnp.isfinite(dtfac), dtfac, f32(1.0))

    taud = taud * dtfac[:, None]
    dtaux2d = taud * xn[:, None]
    dtauy2d = taud * yn[:, None]
    dudt_out = dudt_out + dtaux2d
    dvdt_out = dvdt_out + dtauy2d
    dusfc = -jnp.sum(dtaux2d * del_, axis=1) / g_
    dvsfc = -jnp.sum(dtauy2d * del_, axis=1) / g_

    return dudt_out, dvdt_out, dtaux2d, dtauy2d, dusfc, dvsfc


def gwdo2d_jax(
    dudt: Any,
    dvdt: Any,
    u1: Any,
    v1: Any,
    t1: Any,
    q1: Any,
    del_: Any,
    prsi: Any,
    prsl: Any,
    prslk: Any,
    zl: Any,
    kpblmax: int,
    var: Any,
    oc1: Any,
    oa4: Any,
    ol4: Any,
    g_: float,
    cp_: float,
    rd_: float,
    rv_: float,
    fv_: float,
    pi_: float,
    dxmeter: float,
    deltim: float,
    kpbl: Any,
    lat: int,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    del rv_, lat, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
    return _gwdo2d_jax_core(
        dudt,
        dvdt,
        u1,
        v1,
        t1,
        q1,
        del_,
        prsi,
        prsl,
        prslk,
        zl,
        kpblmax=int(kpblmax),
        var=var,
        oc1=oc1,
        oa4=oa4,
        ol4=ol4,
        g_=g_,
        cp_=cp_,
        rd_=rd_,
        fv_=fv_,
        pi_=pi_,
        dxmeter=dxmeter,
        deltim=deltim,
        kpbl=kpbl,
        rounding_token=rounding_token,
    )


def _gwdo_jax_impl(
    u3d: Any,
    v3d: Any,
    t3d: Any,
    qv3d: Any,
    p3d: Any,
    p3di: Any,
    pi3d: Any,
    z: Any,
    rublten: Any,
    rvblten: Any,
    dtaux3d: Any,
    dtauy3d: Any,
    dusfcg: Any,
    dvsfcg: Any,
    var2d: Any,
    oc12d: Any,
    oa2d1: Any,
    oa2d2: Any,
    oa2d3: Any,
    oa2d4: Any,
    ol2d1: Any,
    ol2d2: Any,
    ol2d3: Any,
    ol2d4: Any,
    sina: Any,
    cosa: Any,
    cp: Any,
    g: Any,
    rd: Any,
    rv: Any,
    ep1: Any,
    pi: Any,
    dt: Any,
    dx: Any,
    kpbl2d: Any,
    rounding_token: Any,
    *,
    kpblmax: int,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    del rv, ids, ide, jds, jde, kds, kde, ime, jme, kme

    f32 = jnp.float32
    u3d = jnp.asarray(u3d, dtype=f32)
    v3d = jnp.asarray(v3d, dtype=f32)
    t3d = jnp.asarray(t3d, dtype=f32)
    qv3d = jnp.asarray(qv3d, dtype=f32)
    p3d = jnp.asarray(p3d, dtype=f32)
    p3di = jnp.asarray(p3di, dtype=f32)
    pi3d = jnp.asarray(pi3d, dtype=f32)
    z = jnp.asarray(z, dtype=f32)
    rublten_out = jnp.asarray(rublten, dtype=f32)
    rvblten_out = jnp.asarray(rvblten, dtype=f32)
    dtaux3d_out = jnp.asarray(dtaux3d, dtype=f32)
    dtauy3d_out = jnp.asarray(dtauy3d, dtype=f32)
    dusfcg_out = jnp.asarray(dusfcg, dtype=f32)
    dvsfcg_out = jnp.asarray(dvsfcg, dtype=f32)
    var2d = jnp.asarray(var2d, dtype=f32)
    oc12d = jnp.asarray(oc12d, dtype=f32)
    oa2d1 = jnp.asarray(oa2d1, dtype=f32)
    oa2d2 = jnp.asarray(oa2d2, dtype=f32)
    oa2d3 = jnp.asarray(oa2d3, dtype=f32)
    oa2d4 = jnp.asarray(oa2d4, dtype=f32)
    ol2d1 = jnp.asarray(ol2d1, dtype=f32)
    ol2d2 = jnp.asarray(ol2d2, dtype=f32)
    ol2d3 = jnp.asarray(ol2d3, dtype=f32)
    ol2d4 = jnp.asarray(ol2d4, dtype=f32)
    sina = jnp.asarray(sina, dtype=f32)
    cosa = jnp.asarray(cosa, dtype=f32)
    kpbl2d = jnp.asarray(kpbl2d, dtype=jnp.int32)

    i0 = int(its) - int(ims)
    i1 = int(ite) - int(ims) + 1
    j0 = int(jts) - int(jms)
    j1 = int(jte) - int(jms) + 1
    k0 = int(kts) - int(kms)
    k1 = int(kte) - int(kms) + 1
    nk = k1 - k0
    ni = i1 - i0
    nj = j1 - j0

    def col3(field: Any) -> Any:
        return jnp.transpose(field[i0:i1, k0:k1, j0:j1], (0, 2, 1)).reshape((ni * nj, nk))

    def col3i(field: Any) -> Any:
        return jnp.transpose(field[i0:i1, k0 : k1 + 1, j0:j1], (0, 2, 1)).reshape((ni * nj, nk + 1))

    def col2(field: Any) -> Any:
        return field[i0:i1, j0:j1].reshape((ni * nj,))

    def uncol3(cols: Any) -> Any:
        return jnp.transpose(jnp.reshape(cols, (ni, nj, nk)), (0, 2, 1))

    def uncol2(cols: Any) -> Any:
        return jnp.reshape(cols, (ni, nj))

    sinv = col2(sina)
    cosv = col2(cosa)
    u_local = col3(u3d)
    v_local = col3(v3d)
    ugeo = _real_sub(
        _real_mul(u_local, cosv[:, None], rounding_token),
        _real_mul(v_local, sinv[:, None], rounding_token),
        rounding_token,
    )
    vgeo = _real_add(
        _real_mul(u_local, sinv[:, None], rounding_token),
        _real_mul(v_local, cosv[:, None], rounding_token),
        rounding_token,
    )
    pdhi = col3i(p3di)
    delprsi = _real_sub(pdhi[:, :nk], pdhi[:, 1 : nk + 1], rounding_token)
    dudt = jnp.zeros_like(ugeo)
    dvdt = jnp.zeros_like(vgeo)
    oa4 = jnp.stack((col2(oa2d1), col2(oa2d2), col2(oa2d3), col2(oa2d4)), axis=1)
    ol4 = jnp.stack((col2(ol2d1), col2(ol2d2), col2(ol2d3), col2(ol2d4)), axis=1)

    dudt_out, dvdt_out, dtaux, dtauy, dusfc, dvsfc = _gwdo2d_jax_core(
        dudt,
        dvdt,
        ugeo,
        vgeo,
        col3(t3d),
        col3(qv3d),
        delprsi,
        pdhi,
        col3(p3d),
        col3(pi3d),
        col3(z),
        kpblmax=int(kpblmax),
        var=col2(var2d),
        oc1=col2(oc12d),
        oa4=oa4,
        ol4=ol4,
        g_=g,
        cp_=cp,
        rd_=rd,
        fv_=ep1,
        pi_=pi,
        dxmeter=dx,
        deltim=dt,
        kpbl=col2(kpbl2d),
        rounding_token=rounding_token,
    )

    rubl_add = uncol3(
        _real_add(
            _real_mul(dudt_out, cosv[:, None], rounding_token),
            _real_mul(dvdt_out, sinv[:, None], rounding_token),
            rounding_token,
        )
    )
    rvbl_add = uncol3(
        _real_add(
            _real_mul(-dudt_out, sinv[:, None], rounding_token),
            _real_mul(dvdt_out, cosv[:, None], rounding_token),
            rounding_token,
        )
    )
    dtaux_out = uncol3(
        _real_add(
            _real_mul(dtaux, cosv[:, None], rounding_token),
            _real_mul(dtauy, sinv[:, None], rounding_token),
            rounding_token,
        )
    )
    dtauy_out = uncol3(
        _real_add(
            _real_mul(-dtaux, sinv[:, None], rounding_token),
            _real_mul(dtauy, cosv[:, None], rounding_token),
            rounding_token,
        )
    )
    dusfcg_tile = uncol2(
        _real_add(
            _real_mul(dusfc, cosv, rounding_token),
            _real_mul(dvsfc, sinv, rounding_token),
            rounding_token,
        )
    )
    dvsfcg_tile = uncol2(
        _real_add(
            _real_mul(-dusfc, sinv, rounding_token),
            _real_mul(dvsfc, cosv, rounding_token),
            rounding_token,
        )
    )

    rublten_out = rublten_out.at[i0:i1, k0:k1, j0:j1].add(rubl_add)
    rvblten_out = rvblten_out.at[i0:i1, k0:k1, j0:j1].add(rvbl_add)
    dtaux3d_out = dtaux3d_out.at[i0:i1, k0:k1, j0:j1].set(dtaux_out)
    dtauy3d_out = dtauy3d_out.at[i0:i1, k0:k1, j0:j1].set(dtauy_out)
    dusfcg_out = dusfcg_out.at[i0:i1, j0:j1].set(dusfcg_tile)
    dvsfcg_out = dvsfcg_out.at[i0:i1, j0:j1].set(dvsfcg_tile)
    return rublten_out, rvblten_out, dtaux3d_out, dtauy3d_out, dusfcg_out, dvsfcg_out


def _gwdo_jax_active_impl(
    u3d: Any,
    v3d: Any,
    t3d: Any,
    qv3d: Any,
    p3d: Any,
    p3di: Any,
    pi3d: Any,
    z: Any,
    rublten: Any,
    rvblten: Any,
    dtaux3d: Any,
    dtauy3d: Any,
    dusfcg: Any,
    dvsfcg: Any,
    var2d: Any,
    oc12d: Any,
    oa2d1: Any,
    oa2d2: Any,
    oa2d3: Any,
    oa2d4: Any,
    ol2d1: Any,
    ol2d2: Any,
    ol2d3: Any,
    ol2d4: Any,
    sina: Any,
    cosa: Any,
    cp: Any,
    g: Any,
    rd: Any,
    rv: Any,
    ep1: Any,
    pi: Any,
    dt: Any,
    dx: Any,
    kpbl2d: Any,
    rounding_token: Any,
    gwdo_active_i: Any,
    gwdo_active_j: Any,
    *,
    kpblmax: int,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    del rv, ids, ide, jds, jde, kds, kde, ime, jme, kme

    f32 = jnp.float32
    u3d = jnp.asarray(u3d, dtype=f32)
    v3d = jnp.asarray(v3d, dtype=f32)
    t3d = jnp.asarray(t3d, dtype=f32)
    qv3d = jnp.asarray(qv3d, dtype=f32)
    p3d = jnp.asarray(p3d, dtype=f32)
    p3di = jnp.asarray(p3di, dtype=f32)
    pi3d = jnp.asarray(pi3d, dtype=f32)
    z = jnp.asarray(z, dtype=f32)
    rublten_out = jnp.asarray(rublten, dtype=f32)
    rvblten_out = jnp.asarray(rvblten, dtype=f32)
    dtaux3d_out = jnp.asarray(dtaux3d, dtype=f32)
    dtauy3d_out = jnp.asarray(dtauy3d, dtype=f32)
    dusfcg_out = jnp.asarray(dusfcg, dtype=f32)
    dvsfcg_out = jnp.asarray(dvsfcg, dtype=f32)
    var2d = jnp.asarray(var2d, dtype=f32)
    oc12d = jnp.asarray(oc12d, dtype=f32)
    oa2d1 = jnp.asarray(oa2d1, dtype=f32)
    oa2d2 = jnp.asarray(oa2d2, dtype=f32)
    oa2d3 = jnp.asarray(oa2d3, dtype=f32)
    oa2d4 = jnp.asarray(oa2d4, dtype=f32)
    ol2d1 = jnp.asarray(ol2d1, dtype=f32)
    ol2d2 = jnp.asarray(ol2d2, dtype=f32)
    ol2d3 = jnp.asarray(ol2d3, dtype=f32)
    ol2d4 = jnp.asarray(ol2d4, dtype=f32)
    sina = jnp.asarray(sina, dtype=f32)
    cosa = jnp.asarray(cosa, dtype=f32)
    kpbl2d = jnp.asarray(kpbl2d, dtype=jnp.int32)
    active_i = jnp.asarray(gwdo_active_i, dtype=jnp.int32)
    active_j = jnp.asarray(gwdo_active_j, dtype=jnp.int32)

    i0 = int(its) - int(ims)
    i1 = int(ite) - int(ims) + 1
    j0 = int(jts) - int(jms)
    j1 = int(jte) - int(jms) + 1
    k0 = int(kts) - int(kms)
    k1 = int(kte) - int(kms) + 1
    nk = k1 - k0

    levels = jnp.arange(k0, k1, dtype=jnp.int32)
    ilevels = jnp.arange(k0, k1 + 1, dtype=jnp.int32)
    idx3 = (active_i[:, None], levels[None, :], active_j[:, None])
    idx2 = (active_i, active_j)

    def col3(field: Any) -> Any:
        return field[idx3]

    def col3i(field: Any) -> Any:
        return field[active_i[:, None], ilevels[None, :], active_j[:, None]]

    def col2(field: Any) -> Any:
        return field[idx2]

    sinv = col2(sina)
    cosv = col2(cosa)
    u_local = col3(u3d)
    v_local = col3(v3d)
    ugeo = _real_sub(
        _real_mul(u_local, cosv[:, None], rounding_token),
        _real_mul(v_local, sinv[:, None], rounding_token),
        rounding_token,
    )
    vgeo = _real_add(
        _real_mul(u_local, sinv[:, None], rounding_token),
        _real_mul(v_local, cosv[:, None], rounding_token),
        rounding_token,
    )
    pdhi = col3i(p3di)
    delprsi = _real_sub(pdhi[:, :nk], pdhi[:, 1 : nk + 1], rounding_token)
    dudt = jnp.zeros_like(ugeo)
    dvdt = jnp.zeros_like(vgeo)
    oa4 = jnp.stack((col2(oa2d1), col2(oa2d2), col2(oa2d3), col2(oa2d4)), axis=1)
    ol4 = jnp.stack((col2(ol2d1), col2(ol2d2), col2(ol2d3), col2(ol2d4)), axis=1)

    dudt_out, dvdt_out, dtaux, dtauy, dusfc, dvsfc = _gwdo2d_jax_core(
        dudt,
        dvdt,
        ugeo,
        vgeo,
        col3(t3d),
        col3(qv3d),
        delprsi,
        pdhi,
        col3(p3d),
        col3(pi3d),
        col3(z),
        kpblmax=int(kpblmax),
        var=col2(var2d),
        oc1=col2(oc12d),
        oa4=oa4,
        ol4=ol4,
        g_=g,
        cp_=cp,
        rd_=rd,
        fv_=ep1,
        pi_=pi,
        dxmeter=dx,
        deltim=dt,
        kpbl=col2(kpbl2d),
        rounding_token=rounding_token,
    )

    rubl_add = _real_add(
        _real_mul(dudt_out, cosv[:, None], rounding_token),
        _real_mul(dvdt_out, sinv[:, None], rounding_token),
        rounding_token,
    )
    rvbl_add = _real_add(
        _real_mul(-dudt_out, sinv[:, None], rounding_token),
        _real_mul(dvdt_out, cosv[:, None], rounding_token),
        rounding_token,
    )
    dtaux_out = _real_add(
        _real_mul(dtaux, cosv[:, None], rounding_token),
        _real_mul(dtauy, sinv[:, None], rounding_token),
        rounding_token,
    )
    dtauy_out = _real_add(
        _real_mul(-dtaux, sinv[:, None], rounding_token),
        _real_mul(dtauy, cosv[:, None], rounding_token),
        rounding_token,
    )
    dusfcg_active = _real_add(
        _real_mul(dusfc, cosv, rounding_token),
        _real_mul(dvsfc, sinv, rounding_token),
        rounding_token,
    )
    dvsfcg_active = _real_add(
        _real_mul(-dusfc, sinv, rounding_token),
        _real_mul(dvsfc, cosv, rounding_token),
        rounding_token,
    )

    rublten_out = rublten_out.at[idx3].add(rubl_add)
    rvblten_out = rvblten_out.at[idx3].add(rvbl_add)

    if _JAX_GWDO_ACTIVE_ZERO_INACTIVE:
        dtaux3d_out = dtaux3d_out.at[i0:i1, k0:k1, j0:j1].set(jnp.zeros((i1 - i0, nk, j1 - j0), dtype=f32))
        dtauy3d_out = dtauy3d_out.at[i0:i1, k0:k1, j0:j1].set(jnp.zeros((i1 - i0, nk, j1 - j0), dtype=f32))
        dusfcg_out = dusfcg_out.at[i0:i1, j0:j1].set(jnp.zeros((i1 - i0, j1 - j0), dtype=f32))
        dvsfcg_out = dvsfcg_out.at[i0:i1, j0:j1].set(jnp.zeros((i1 - i0, j1 - j0), dtype=f32))

    dtaux3d_out = dtaux3d_out.at[idx3].set(dtaux_out)
    dtauy3d_out = dtauy3d_out.at[idx3].set(dtauy_out)
    dusfcg_out = dusfcg_out.at[idx2].set(dusfcg_active)
    dvsfcg_out = dvsfcg_out.at[idx2].set(dvsfcg_active)
    return rublten_out, rvblten_out, dtaux3d_out, dtauy3d_out, dusfcg_out, dvsfcg_out


def _gwdo_jax_row_range_impl(
    u3d: Any,
    v3d: Any,
    t3d: Any,
    qv3d: Any,
    p3d: Any,
    p3di: Any,
    pi3d: Any,
    z: Any,
    rublten: Any,
    rvblten: Any,
    dtaux3d: Any,
    dtauy3d: Any,
    dusfcg: Any,
    dvsfcg: Any,
    var2d: Any,
    oc12d: Any,
    oa2d1: Any,
    oa2d2: Any,
    oa2d3: Any,
    oa2d4: Any,
    ol2d1: Any,
    ol2d2: Any,
    ol2d3: Any,
    ol2d4: Any,
    sina: Any,
    cosa: Any,
    cp: Any,
    g: Any,
    rd: Any,
    rv: Any,
    ep1: Any,
    pi: Any,
    dt: Any,
    dx: Any,
    kpbl2d: Any,
    rounding_token: Any,
    gwdo_row_j: Any,
    gwdo_row_i_start: Any,
    gwdo_row_len: Any,
    *,
    row_max_len: int,
    kpblmax: int,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    del rv, ids, ide, jds, jde, kds, kde, ime, jme, kme

    f32 = jnp.float32
    u3d = jnp.asarray(u3d, dtype=f32)
    v3d = jnp.asarray(v3d, dtype=f32)
    t3d = jnp.asarray(t3d, dtype=f32)
    qv3d = jnp.asarray(qv3d, dtype=f32)
    p3d = jnp.asarray(p3d, dtype=f32)
    p3di = jnp.asarray(p3di, dtype=f32)
    pi3d = jnp.asarray(pi3d, dtype=f32)
    z = jnp.asarray(z, dtype=f32)
    rublten_out = jnp.asarray(rublten, dtype=f32)
    rvblten_out = jnp.asarray(rvblten, dtype=f32)
    dtaux3d_out = jnp.asarray(dtaux3d, dtype=f32)
    dtauy3d_out = jnp.asarray(dtauy3d, dtype=f32)
    dusfcg_out = jnp.asarray(dusfcg, dtype=f32)
    dvsfcg_out = jnp.asarray(dvsfcg, dtype=f32)
    var2d = jnp.asarray(var2d, dtype=f32)
    oc12d = jnp.asarray(oc12d, dtype=f32)
    oa2d1 = jnp.asarray(oa2d1, dtype=f32)
    oa2d2 = jnp.asarray(oa2d2, dtype=f32)
    oa2d3 = jnp.asarray(oa2d3, dtype=f32)
    oa2d4 = jnp.asarray(oa2d4, dtype=f32)
    ol2d1 = jnp.asarray(ol2d1, dtype=f32)
    ol2d2 = jnp.asarray(ol2d2, dtype=f32)
    ol2d3 = jnp.asarray(ol2d3, dtype=f32)
    ol2d4 = jnp.asarray(ol2d4, dtype=f32)
    sina = jnp.asarray(sina, dtype=f32)
    cosa = jnp.asarray(cosa, dtype=f32)
    kpbl2d = jnp.asarray(kpbl2d, dtype=jnp.int32)

    row_j = jnp.asarray(gwdo_row_j, dtype=jnp.int32)
    row_i_start = jnp.asarray(gwdo_row_i_start, dtype=jnp.int32)
    row_len = jnp.asarray(gwdo_row_len, dtype=jnp.int32)
    nrow = int(row_j.shape[0])
    width = int(row_max_len)
    nk = int(kte) - int(kts) + 1
    k0 = int(kts) - int(kms)
    k1 = k0 + nk

    offsets = jnp.arange(width, dtype=jnp.int32)
    i_matrix = row_i_start[:, None] + offsets[None, :]
    j_matrix = jnp.broadcast_to(row_j[:, None], (nrow, width))
    valid_matrix = offsets[None, :] < row_len[:, None]
    i_flat = jnp.reshape(i_matrix, (nrow * width,))
    j_flat = jnp.reshape(j_matrix, (nrow * width,))
    valid_flat = jnp.reshape(valid_matrix, (nrow * width,))
    safe_i = jnp.where(valid_flat, i_flat, jnp.zeros_like(i_flat))
    safe_j = jnp.where(valid_flat, j_flat, jnp.zeros_like(j_flat))

    levels = jnp.arange(k0, k1, dtype=jnp.int32)
    ilevels = jnp.arange(k0, k1 + 1, dtype=jnp.int32)
    idx3 = (safe_i[:, None], levels[None, :], safe_j[:, None])
    idx2 = (safe_i, safe_j)

    def col3(field: Any) -> Any:
        return field[idx3]

    def col3i(field: Any) -> Any:
        return field[safe_i[:, None], ilevels[None, :], safe_j[:, None]]

    def col2(field: Any) -> Any:
        return field[idx2]

    sinv = col2(sina)
    cosv = col2(cosa)
    u_local = col3(u3d)
    v_local = col3(v3d)
    ugeo = _real_sub(
        _real_mul(u_local, cosv[:, None], rounding_token),
        _real_mul(v_local, sinv[:, None], rounding_token),
        rounding_token,
    )
    vgeo = _real_add(
        _real_mul(u_local, sinv[:, None], rounding_token),
        _real_mul(v_local, cosv[:, None], rounding_token),
        rounding_token,
    )
    pdhi = col3i(p3di)
    delprsi = _real_sub(pdhi[:, :nk], pdhi[:, 1 : nk + 1], rounding_token)
    dudt = jnp.zeros_like(ugeo)
    dvdt = jnp.zeros_like(vgeo)
    oa4 = jnp.stack((col2(oa2d1), col2(oa2d2), col2(oa2d3), col2(oa2d4)), axis=1)
    ol4 = jnp.stack((col2(ol2d1), col2(ol2d2), col2(ol2d3), col2(ol2d4)), axis=1)

    dudt_out, dvdt_out, dtaux, dtauy, dusfc, dvsfc = _gwdo2d_jax_core(
        dudt,
        dvdt,
        ugeo,
        vgeo,
        col3(t3d),
        col3(qv3d),
        delprsi,
        pdhi,
        col3(p3d),
        col3(pi3d),
        col3(z),
        kpblmax=int(kpblmax),
        var=jnp.where(valid_flat, col2(var2d), f32(0.0)),
        oc1=col2(oc12d),
        oa4=oa4,
        ol4=ol4,
        g_=g,
        cp_=cp,
        rd_=rd,
        fv_=ep1,
        pi_=pi,
        dxmeter=dx,
        deltim=dt,
        kpbl=col2(kpbl2d),
        rounding_token=rounding_token,
    )

    valid3 = valid_flat[:, None]
    rubl_add = jnp.where(
        valid3,
        _real_add(
            _real_mul(dudt_out, cosv[:, None], rounding_token),
            _real_mul(dvdt_out, sinv[:, None], rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    rvbl_add = jnp.where(
        valid3,
        _real_add(
            _real_mul(-dudt_out, sinv[:, None], rounding_token),
            _real_mul(dvdt_out, cosv[:, None], rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    dtaux_out = jnp.where(
        valid3,
        _real_add(
            _real_mul(dtaux, cosv[:, None], rounding_token),
            _real_mul(dtauy, sinv[:, None], rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    dtauy_out = jnp.where(
        valid3,
        _real_add(
            _real_mul(-dtaux, sinv[:, None], rounding_token),
            _real_mul(dtauy, cosv[:, None], rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    dusfcg_active = jnp.where(
        valid_flat,
        _real_add(
            _real_mul(dusfc, cosv, rounding_token),
            _real_mul(dvsfc, sinv, rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )
    dvsfcg_active = jnp.where(
        valid_flat,
        _real_add(
            _real_mul(-dusfc, sinv, rounding_token),
            _real_mul(dvsfc, cosv, rounding_token),
            rounding_token,
        ),
        f32(0.0),
    )

    rublten_out = rublten_out.at[idx3].add(rubl_add)
    rvblten_out = rvblten_out.at[idx3].add(rvbl_add)
    if _JAX_GWDO_ACTIVE_ZERO_INACTIVE:
        i0 = int(its) - int(ims)
        i1 = int(ite) - int(ims) + 1
        j0 = int(jts) - int(jms)
        j1 = int(jte) - int(jms) + 1
        dtaux3d_out = dtaux3d_out.at[i0:i1, k0:k1, j0:j1].set(jnp.zeros((i1 - i0, nk, j1 - j0), dtype=f32))
        dtauy3d_out = dtauy3d_out.at[i0:i1, k0:k1, j0:j1].set(jnp.zeros((i1 - i0, nk, j1 - j0), dtype=f32))
        dusfcg_out = dusfcg_out.at[i0:i1, j0:j1].set(jnp.zeros((i1 - i0, j1 - j0), dtype=f32))
        dvsfcg_out = dvsfcg_out.at[i0:i1, j0:j1].set(jnp.zeros((i1 - i0, j1 - j0), dtype=f32))
    dtaux3d_out = dtaux3d_out.at[idx3].set(dtaux_out)
    dtauy3d_out = dtauy3d_out.at[idx3].set(dtauy_out)
    dusfcg_out = dusfcg_out.at[idx2].set(dusfcg_active)
    dvsfcg_out = dvsfcg_out.at[idx2].set(dvsfcg_active)
    return rublten_out, rvblten_out, dtaux3d_out, dtauy3d_out, dusfcg_out, dvsfcg_out


_GWDO_JAX_STATIC_ARGNAMES = (
    "kpblmax",
    "ids",
    "ide",
    "jds",
    "jde",
    "kds",
    "kde",
    "ims",
    "ime",
    "jms",
    "jme",
    "kms",
    "kme",
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
)
_gwdo_jax_jit = jax.jit(_gwdo_jax_impl, static_argnames=_GWDO_JAX_STATIC_ARGNAMES, inline=_INLINE_COLUMN_PHYSICS_JITS)
_gwdo_jax_active_jit = jax.jit(_gwdo_jax_active_impl, static_argnames=_GWDO_JAX_STATIC_ARGNAMES, inline=_INLINE_COLUMN_PHYSICS_JITS)
_gwdo_jax_row_range_jit = jax.jit(
    _gwdo_jax_row_range_impl,
    static_argnames=(*_GWDO_JAX_STATIC_ARGNAMES, "row_max_len"),
    inline=_INLINE_COLUMN_PHYSICS_JITS,
)


def gwdo2d(
    dudt: Any,
    dvdt: Any,
    u1: Any,
    v1: Any,
    t1: Any,
    q1: Any,
    del_: Any,
    prsi: Any,
    prsl: Any,
    prslk: Any,
    zl: Any,
    kpblmax: int,
    var: Any,
    oc1: Any,
    oa4: Any,
    ol4: Any,
    g_: float,
    cp_: float,
    rd_: float,
    rv_: float,
    fv_: float,
    pi_: float,
    dxmeter: float,
    deltim: float,
    kpbl: Any,
    lat: int,
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
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return WRF ``gwdo2d`` updates for tendencies, stress, and surface drag."""

    dudt_out = dudt.copy()
    dvdt_out = dvdt.copy()
    u1_arr = u1
    v1_arr = v1
    t1_arr = t1
    q1_arr = q1
    del_arr = del_
    prsi_arr = prsi
    prsl_arr = prsl
    prslk_arr = prslk
    zl_arr = zl
    var_arr = var
    oc1_arr = oc1
    oa4_arr = oa4
    ol4_arr = ol4

    ni = int(ite) - int(its) + 1
    nk = int(kte) - int(kts) + 1

    def ti(i: int) -> int:
        return int(i) - int(its)

    def tk(k: int) -> int:
        return int(k) - int(kts)

    def fi(i: int) -> int:
        return int(i) - int(ims)

    def fk(k: int) -> int:
        return int(k) - int(kms)

    ric = np.float32(0.25)
    dw2min = np.float32(1.0)
    rimin = np.float32(-100.0)
    bnv2min = np.float32(1.0e-5)
    efmin = np.float32(0.0)
    efmax = np.float32(10.0)
    xl = np.float32(4.0e4)
    gmax = np.float32(1.0)
    veleps = np.float32(1.0)
    frc = np.float32(1.0)
    ce = np.float32(0.8)
    cg = np.float32(0.5)
    kpblmin = 2
    mdir = 8
    nwdir = np.array([6, 7, 5, 8, 2, 3, 1, 4], dtype=np.int32)
    frmax = np.float32(10.0)
    olmin = np.float32(1.0e-5)
    odmin = np.float32(0.1)
    odmax = np.float32(10.0)

    lcap = int(kte)
    lcapp1 = lcap + 1
    fdir = np.float32(mdir / (2.0 * float(pi_)))

    delx = np.full(ni, np.float32(dxmeter), dtype=np.float32)
    dely = np.full(ni, np.float32(dxmeter), dtype=np.float32)
    dxy4 = np.zeros((ni, 4), dtype=np.float32)
    dxy4p = np.zeros((ni, 4), dtype=np.float32)
    dxy4[:, 0] = delx
    dxy4[:, 1] = dely
    dxy4[:, 2] = np.sqrt(delx * delx + dely * dely)
    dxy4[:, 3] = dxy4[:, 2]
    dxy4p[:, 0] = dxy4[:, 1]
    dxy4p[:, 1] = dxy4[:, 0]
    dxy4p[:, 2] = dxy4[:, 3]
    dxy4p[:, 3] = dxy4[:, 2]
    cleff = np.full(ni, np.float32(dxmeter), dtype=np.float32)

    ldrag = np.zeros(ni, dtype=np.bool_)
    icrilv = np.zeros(ni, dtype=np.bool_)
    kloop1 = np.ones(ni, dtype=np.bool_)
    klowtop = np.zeros(ni, dtype=np.int32)
    kbl = np.zeros(ni, dtype=np.int32)

    dtaux2d = np.zeros((ni, nk), dtype=np.float32)
    dtauy2d = np.zeros((ni, nk), dtype=np.float32)
    xn = np.zeros(ni, dtype=np.float32)
    yn = np.zeros(ni, dtype=np.float32)
    ubar = np.zeros(ni, dtype=np.float32)
    vbar = np.zeros(ni, dtype=np.float32)
    rhobar = np.zeros(ni, dtype=np.float32)
    ulow = np.zeros(ni, dtype=np.float32)
    rulow = np.zeros(ni, dtype=np.float32)
    oa = np.zeros(ni, dtype=np.float32)
    ol = np.zeros(ni, dtype=np.float32)
    taub = np.zeros(ni, dtype=np.float32)
    usqj = np.zeros((ni, nk), dtype=np.float32)
    bnv2 = np.zeros((ni, nk), dtype=np.float32)
    vtj = np.zeros((ni, nk), dtype=np.float32)
    vtk = np.zeros((ni, nk), dtype=np.float32)
    taup = np.zeros((ni, nk + 1), dtype=np.float32)
    taud = np.zeros((ni, nk), dtype=np.float32)
    rho = np.zeros((ni, nk), dtype=np.float32)
    velco = np.zeros((ni, nk), dtype=np.float32)
    dtfac = np.ones(ni, dtype=np.float32)
    xlinv = np.full(ni, np.float32(1.0) / xl, dtype=np.float32)
    komax = np.zeros(ni, dtype=np.int32)
    taufb = np.zeros((ni, nk + 1), dtype=np.float32)
    delks = np.zeros(ni, dtype=np.float32)
    delks1 = np.zeros(ni, dtype=np.float32)
    zlowtop = np.zeros(ni, dtype=np.float32)
    coefm = np.zeros(ni, dtype=np.float32)
    bnv = np.zeros(ni, dtype=np.float32)
    fr = np.zeros(ni, dtype=np.float32)
    brvf = np.zeros(ni, dtype=np.float32)
    dxy = np.zeros(ni, dtype=np.float32)
    dxyp = np.zeros(ni, dtype=np.float32)
    olp = np.zeros(ni, dtype=np.float32)
    od = np.zeros(ni, dtype=np.float32)
    dusfc = np.zeros(ni, dtype=np.float32)
    dvsfc = np.zeros(ni, dtype=np.float32)

    for k in range(int(kts), int(kte) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            kk = tk(k)
            vtj[ii, kk] = t1_arr[fi(i), fk(k)] * (1.0 + np.float32(fv_) * q1_arr[fi(i), fk(k)])
            vtk[ii, kk] = vtj[ii, kk] / prslk_arr[fi(i), fk(k)]
            rho[ii, kk] = (1.0 / np.float32(rd_)) * prsl_arr[ii, kk] / vtj[ii, kk]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        zlowtop[ii] = 2.0 * var_arr[fi(i)]

    for k in range(int(kts) + 1, int(kte) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if kloop1[ii] and zl_arr[fi(i), fk(k)] - zl_arr[fi(i), fk(1)] >= zlowtop[ii]:
                klowtop[ii] = k + 1
                kloop1[ii] = False

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        kbl[ii] = max(min(int(klowtop[ii]), int(kpblmax)), kpblmin)
    komax[:] = kbl[:]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        delks[ii] = 1.0 / (prsi_arr[ii, tk(1)] - prsi_arr[ii, tk(int(kbl[ii]))])
        delks1[ii] = 1.0 / (prsl_arr[ii, tk(1)] - prsl_arr[ii, tk(int(kbl[ii]))])

    for k in range(int(kts), int(kpblmax) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k < int(kbl[ii]):
                rcsks = del_arr[ii, tk(k)] * delks[ii]
                rdelks = del_arr[ii, tk(k)] * delks[ii]
                ubar[ii] += rcsks * u1_arr[ii, tk(k)]
                vbar[ii] += rcsks * v1_arr[ii, tk(k)]
                rhobar[ii] += rdelks * rho[ii, tk(k)]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        wdir = np.arctan2(ubar[ii], vbar[ii]) + np.float32(pi_)
        idir = int(np.floor(float(fdir * wdir) + 0.5)) % mdir + 1
        nwd = int(nwdir[idir - 1])
        quadrant = (nwd - 1) % 4
        oa[ii] = (1 - 2 * int((nwd - 1) / 4)) * oa4_arr[ii, quadrant]
        ol[ii] = ol4_arr[ii, quadrant]
        ol4p = np.array([ol4_arr[ii, 1], ol4_arr[ii, 0], ol4_arr[ii, 3], ol4_arr[ii, 2]], dtype=np.float32)
        olp[ii] = ol4p[quadrant]
        od[ii] = olp[ii] / max(ol[ii], olmin)
        od[ii] = min(od[ii], odmax)
        od[ii] = max(od[ii], odmin)
        dxy[ii] = dxy4[ii, quadrant]
        dxyp[ii] = dxy4p[ii, quadrant]

    for k in range(int(kts), int(kte)):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            kk = tk(k)
            ti_local = 2.0 / (t1_arr[fi(i), fk(k)] + t1_arr[fi(i), fk(k + 1)])
            rdz = 1.0 / (zl_arr[fi(i), fk(k + 1)] - zl_arr[fi(i), fk(k)])
            tem1 = u1_arr[ii, kk] - u1_arr[ii, tk(k + 1)]
            tem2 = v1_arr[ii, kk] - v1_arr[ii, tk(k + 1)]
            dw2 = tem1 * tem1 + tem2 * tem2
            shr2 = max(dw2, dw2min) * rdz * rdz
            bvf2 = np.float32(g_) * (np.float32(g_) / np.float32(cp_) + rdz * (vtj[ii, tk(k + 1)] - vtj[ii, kk])) * ti_local
            usqj[ii, kk] = max(bvf2 / shr2, rimin)
            bnv2[ii, kk] = 2.0 * np.float32(g_) * rdz * (vtk[ii, tk(k + 1)] - vtk[ii, kk]) / (vtk[ii, tk(k + 1)] + vtk[ii, kk])

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        ulow[ii] = max(np.sqrt(ubar[ii] * ubar[ii] + vbar[ii] * vbar[ii]), np.float32(1.0))
        rulow[ii] = 1.0 / ulow[ii]

    for k in range(int(kts), int(kte)):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            kk = tk(k)
            velco[ii, kk] = 0.5 * ((u1_arr[ii, kk] + u1_arr[ii, tk(k + 1)]) * ubar[ii] + (v1_arr[ii, kk] + v1_arr[ii, tk(k + 1)]) * vbar[ii])
            velco[ii, kk] *= rulow[ii]
            if velco[ii, kk] < veleps and velco[ii, kk] > 0.0:
                velco[ii, kk] = veleps

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        ldrag[ii] = velco[ii, tk(1)] <= 0.0

    for k in range(kpblmin, int(kpblmax) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k < int(kbl[ii]):
                ldrag[ii] = ldrag[ii] or velco[ii, tk(k)] <= 0.0

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        wtkbj = (prsl_arr[ii, tk(1)] - prsl_arr[ii, tk(2)]) * delks1[ii]
        bnv2[ii, tk(1)] *= wtkbj
        usqj[ii, tk(1)] *= wtkbj

    for k in range(kpblmin, int(kpblmax) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k < int(kbl[ii]):
                rdelks = (prsl_arr[ii, tk(k)] - prsl_arr[ii, tk(k + 1)]) * delks1[ii]
                bnv2[ii, tk(1)] += bnv2[ii, tk(k)] * rdelks
                usqj[ii, tk(1)] += usqj[ii, tk(k)] * rdelks

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        ldrag[ii] = ldrag[ii] or bnv2[ii, tk(1)] <= 0.0
        ldrag[ii] = ldrag[ii] or ulow[ii] == 1.0
        ldrag[ii] = ldrag[ii] or var_arr[fi(i)] <= 0.0

    for k in range(kpblmin, int(kpblmax) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k < int(kbl[ii]):
                usqj[ii, tk(k)] = usqj[ii, tk(1)]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        if not ldrag[ii]:
            bnv[ii] = np.sqrt(bnv2[ii, tk(1)])
            fr[ii] = bnv[ii] * rulow[ii] * var_arr[fi(i)] * od[ii]
            fr[ii] = min(fr[ii], frmax)
            xn[ii] = ubar[ii] * rulow[ii]
            yn[ii] = vbar[ii] * rulow[ii]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        if not ldrag[ii]:
            efact = (oa[ii] + 2.0) ** (ce * fr[ii] / frc)
            efact = min(max(efact, efmin), efmax)
            coefm[ii] = (1.0 + ol[ii]) ** (oa[ii] + 1.0)
            xlinv[ii] = coefm[ii] / cleff[ii]
            tem = fr[ii] * fr[ii] * oc1_arr[fi(i)]
            gfobnv = gmax * tem / ((tem + cg) * bnv[ii])
            taub[ii] = xlinv[ii] * rhobar[ii] * ulow[ii] * ulow[ii] * ulow[ii] * gfobnv * efact
        else:
            taub[ii] = 0.0
            xn[ii] = 0.0
            yn[ii] = 0.0

    for k in range(int(kts), int(kpblmax) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k <= int(kbl[ii]):
                taup[ii, tk(k)] = taub[ii]

    for k in range(kpblmin, int(kte)):
        kp1 = k + 1
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k >= int(kbl[ii]):
                icrilv[ii] = icrilv[ii] or (usqj[ii, tk(k)] < ric) or (velco[ii, tk(k)] <= 0.0)
                brvf[ii] = np.sqrt(max(bnv2[ii, tk(k)], bnv2min))
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k >= int(kbl[ii]) and not ldrag[ii]:
                if (not icrilv[ii]) and taup[ii, tk(k)] > 0.0:
                    temv = 1.0 / velco[ii, tk(k)]
                    tem1 = coefm[ii] / dxy[ii] * (rho[ii, tk(kp1)] + rho[ii, tk(k)]) * brvf[ii] * velco[ii, tk(k)] * 0.5
                    hd = np.sqrt(taup[ii, tk(k)] / tem1)
                    fro = brvf[ii] * hd * temv
                    tem2 = np.sqrt(usqj[ii, tk(k)])
                    tem = 1.0 + tem2 * fro
                    rim = usqj[ii, tk(k)] * (1.0 - fro) / (tem * tem)
                    if rim <= ric:
                        if oa[ii] <= 0.0 or kp1 >= kpblmin:
                            temc = 2.0 + 1.0 / tem2
                            hd = velco[ii, tk(k)] * (2.0 * np.sqrt(temc) - temc) / brvf[ii]
                            taup[ii, tk(kp1)] = tem1 * hd * hd
                    else:
                        taup[ii, tk(kp1)] = taup[ii, tk(k)]

    if lcap < int(kte):
        for klcap in range(lcapp1, int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                ii = ti(i)
                taup[ii, tk(klcap)] = prsi_arr[ii, tk(klcap)] / prsi_arr[ii, tk(lcap)] * taup[ii, tk(lcap)]

    for i in range(int(its), int(ite) + 1):
        ii = ti(i)
        if not ldrag[ii]:
            kblk = 0
            fbdpe = np.float32(0.0)
            for k in range(int(kte), kpblmin - 1, -1):
                if kblk == 0 and k <= int(kbl[ii]):
                    fbdpe += bnv2[ii, tk(k)] * (zl_arr[fi(i), fk(int(kbl[ii]))] - zl_arr[fi(i), fk(k)]) * del_arr[ii, tk(k)] / np.float32(g_) / rho[ii, tk(k)]
                    fbdke = 0.5 * (u1_arr[ii, tk(k)] ** 2.0 + v1_arr[ii, tk(k)] ** 2.0)
                    if fbdpe >= fbdke:
                        kblk = min(k, int(kbl[ii]))
                        zblk = zl_arr[fi(i), fk(kblk)] - zl_arr[fi(i), fk(int(kts))]
            if kblk != 0:
                fbdcd = max(2.0 - 1.0 / od[ii], 0.0)
                taufb[ii, tk(int(kts))] = 0.5 * rhobar[ii] * coefm[ii] / np.float32(dxmeter) ** 2.0 * fbdcd * dxyp[ii] * olp[ii] * zblk * ulow[ii] ** 2.0
                tautem = taufb[ii, tk(int(kts))] / float(kblk - int(kts))
                for k in range(int(kts) + 1, kblk + 1):
                    taufb[ii, tk(k)] = taufb[ii, tk(k - 1)] - tautem
                taup[ii, :] = taup[ii, :] + taufb[ii, :]

    for k in range(int(kts), int(kte) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            taud[ii, tk(k)] = (taup[ii, tk(k + 1)] - taup[ii, tk(k)]) * np.float32(g_) / del_arr[ii, tk(k)]

    for k in range(int(kts), int(kpblmax)):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            if k <= int(kbl[ii]) and taud[ii, tk(k)] != 0.0:
                dtfac[ii] = min(dtfac[ii], abs(velco[ii, tk(k)] / (np.float32(deltim) * taud[ii, tk(k)])))

    for k in range(int(kts), int(kte) + 1):
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            kk = tk(k)
            taud[ii, kk] *= dtfac[ii]
            dtaux = taud[ii, kk] * xn[ii]
            dtauy = taud[ii, kk] * yn[ii]
            dtaux2d[ii, kk] = dtaux
            dtauy2d[ii, kk] = dtauy
            dudt_out[ii, kk] = dtaux + dudt_out[ii, kk]
            dvdt_out[ii, kk] = dtauy + dvdt_out[ii, kk]
            dusfc[ii] += dtaux * del_arr[ii, kk]
            dvsfc[ii] += dtauy * del_arr[ii, kk]

    dusfc *= -1.0 / np.float32(g_)
    dvsfc *= -1.0 / np.float32(g_)

    return dudt_out, dvdt_out, dtaux2d, dtauy2d, dusfc, dvsfc


try:
    from numba import njit

    gwdo2d = njit(cache=True)(gwdo2d)
except Exception:
    pass


__all__ = ["gwdo", "gwdo2d"]
