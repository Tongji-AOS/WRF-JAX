"""Kernels translated from WRF ``dyn_em/module_ieva_em``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp

from wrf_jax.core.wrf_math import sqrt_for_wrf



ALPHA_MAX = 1.1
ALPHA_MIN = 0.8


@dataclass(frozen=True)
class GridConfigRec:
    """Explicit subset of WRF ``grid_config_rec_type`` needed by ``chk_ieva``."""

    rk_ord: int
    zadvect_implicit: int


@dataclass(frozen=True)
class WWSplitResult:
    ww_e: Any
    ww_i: Any


class HasChkIevaConfig(Protocol):
    rk_ord: int
    zadvect_implicit: int


def _read(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"config_flags is missing required field '{name}'") from exc
    if not hasattr(config, name):
        raise ValueError(f"config_flags is missing required field '{name}'")
    return getattr(config, name)


def chk_ieva(config_flags: HasChkIevaConfig | Mapping[str, Any], rk_step: int) -> bool:
    """Return WRF ``CHK_IEVA`` implicit vertical advection activation flag."""

    rk_order = int(_read(config_flags, "rk_ord"))
    zadvect_implicit = int(_read(config_flags, "zadvect_implicit"))
    return zadvect_implicit > 0 and int(rk_step) == rk_order


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


def ww_split(
    ww_e: Any,
    ww_i: Any,
    u: Any,
    v: Any,
    ww: Any,
    mut: Any,
    rdnw: Any,
    msfty: Any,
    c1f: Any,
    c2f: Any,
    rdx: float,
    rdy: float,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    dt: float,
    config_flags: HasChkIevaConfig | Mapping[str, Any],
    rk_step: int,
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
) -> WWSplitResult:
    """Split WRF ``ww`` into explicit and implicit vertical-advection parts."""

    del kds, kde, jds, kme, msfty, msfux, msfuy, msfvx, msfvy
    ww_e_out = jnp.asarray(ww_e).copy()
    ww_i_out = jnp.asarray(ww_i).copy()
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    ww_arr = jnp.asarray(ww)
    mut_arr = jnp.asarray(mut)
    rdnw_arr = jnp.asarray(rdnw)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)

    def idx(vals: Any, lower: int) -> Any:
        return vals - int(lower)

    def take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any) -> Any:
        ii = jnp.clip(idx(i_vals, ims), 0, arr.shape[0] - 1)
        kk = jnp.clip(idx(k_vals, kms), 0, arr.shape[1] - 1)
        jj = jnp.clip(idx(j_vals, jms), 0, arr.shape[2] - 1)
        return arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]

    def take2(arr: Any, i_vals: Any, j_vals: Any) -> Any:
        ii = jnp.clip(idx(i_vals, ims), 0, arr.shape[0] - 1)
        jj = jnp.clip(idx(j_vals, jms), 0, arr.shape[1] - 1)
        return arr[ii[:, None], jj[None, :]]

    def set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any) -> Any:
        if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
            return arr
        return arr.at[idx(i_vals, ims)[:, None, None], idx(k_vals, kms)[None, :, None], idx(j_vals, jms)[None, None, :]].set(values)

    ieva = chk_ieva(config_flags, rk_step)
    ktf = int(kte)
    i_start = max(int(its) - 1, int(ims))
    i_end = min(min(int(ite), int(ide) - 1) + 1, int(ime))
    j_start = max(int(jts) - 1, int(jms))
    j_end = min(min(int(jte), int(jde) - 1) + 1, int(jme))
    i_vals = jnp.arange(i_start, i_end + 1, dtype=jnp.int32)
    j_vals = jnp.arange(j_start, j_end + 1, dtype=jnp.int32)

    if ieva:
        k_mid = jnp.arange(int(kts) + 1, ktf, dtype=jnp.int32)
        ww_e_out = set3(ww_e_out, i_vals, k_mid, j_vals, jnp.zeros((int(i_vals.size), int(k_mid.size), int(j_vals.size)), dtype=ww_e_out.dtype))
        ww_i_out = set3(ww_i_out, i_vals, k_mid, j_vals, jnp.zeros((int(i_vals.size), int(k_mid.size), int(j_vals.size)), dtype=ww_i_out.dtype))

        k_bottom = jnp.asarray([int(kts)], dtype=jnp.int32)
        k_top = jnp.asarray([ktf], dtype=jnp.int32)
        ww_e_out = set3(ww_e_out, i_vals, k_bottom, j_vals, take3(ww_arr, i_vals, k_bottom, j_vals))
        ww_i_out = set3(ww_i_out, i_vals, k_bottom, j_vals, jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=ww_i_out.dtype))
        ww_e_out = set3(ww_e_out, i_vals, k_top, j_vals, take3(ww_arr, i_vals, k_top, j_vals))
        ww_i_out = set3(ww_i_out, i_vals, k_top, j_vals, jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=ww_i_out.dtype))

        cmnx_ratio = ALPHA_MIN / ALPHA_MAX
        cutoff = 2.0 - cmnx_ratio
        r4cmx = 1.0 / (4.0 - 4.0 * cmnx_ratio)
        ceps = 0.9

        if int(k_mid.size):
            cx = 0.25 * float(rdx) * (
                take3(u_arr, i_vals + 1, k_mid, j_vals)
                + take3(u_arr, i_vals, k_mid, j_vals)
                + take3(u_arr, i_vals + 1, k_mid - 1, j_vals)
                + take3(u_arr, i_vals, k_mid - 1, j_vals)
            )
            cy = 0.25 * float(rdy) * (
                take3(v_arr, i_vals, k_mid, j_vals + 1)
                + take3(v_arr, i_vals, k_mid, j_vals)
                + take3(v_arr, i_vals, k_mid - 1, j_vals + 1)
                + take3(v_arr, i_vals, k_mid - 1, j_vals)
            )
            cw_max = jnp.maximum(ALPHA_MAX - float(dt) * ceps * sqrt_for_wrf(cx**2 + cy**2), 0.0)
            cr = (
                take3(ww_arr, i_vals, k_mid, j_vals)
                * float(dt)
                * rdnw_arr[idx(k_mid, kms)][None, :, None]
                / (c1f_arr[idx(k_mid, kms)][None, :, None] * take2(mut_arr, i_vals, j_vals)[:, None, :] + c2f_arr[idx(k_mid, kms)][None, :, None])
            )
            cw = jnp.abs(cr)
            cw_max2 = cw_max**2
            cw_min = cw_max * cmnx_ratio
            cff = jnp.where(cw < cw_min, cw_max2, jnp.where(cw < cutoff * cw_min, cw_max2 + r4cmx * (cw - cw_min) ** 2, cw_max * cw))
            cff_safe = jnp.where(cff == 0.0, 1.0, cff)
            wfrac = jnp.clip(cw_max2 / cff_safe, 0.0, 1.0)
            ww_value = take3(ww_arr, i_vals, k_mid, j_vals)
            ww_e_out = set3(ww_e_out, i_vals, k_mid, j_vals, jnp.where(cw_max > 0.0, ww_value * wfrac, 0.0))
            ww_i_out = set3(ww_i_out, i_vals, k_mid, j_vals, jnp.where(cw_max > 0.0, ww_value * (1.0 - wfrac), ww_value))
    else:
        k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
        ww_e_out = set3(ww_e_out, i_vals, k_vals, j_vals, take3(ww_arr, i_vals, k_vals, j_vals))
        ww_i_out = set3(ww_i_out, i_vals, k_vals, j_vals, jnp.zeros((int(i_vals.size), int(k_vals.size), int(j_vals.size)), dtype=ww_i_out.dtype))

    return WWSplitResult(ww_e=ww_e_out, ww_i=ww_i_out)


__all__ = ["ALPHA_MAX", "ALPHA_MIN", "GridConfigRec", "WWSplitResult", "chk_ieva", "ww_split"]
