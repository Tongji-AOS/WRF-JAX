"""Selected translations from WRF ``dyn_em/module_small_step_em``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from jax import custom_jvp, lax
import jax.numpy as jnp

from wrf_jax.core.column_scan import scan_columns
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.real_arithmetic import column_barrier, column_binary


R_D = 287.0
CP = 7.0 * R_D / 2.0
CPOVCV = CP / (CP - R_D)
G = 9.81


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in jnp.asarray(value).shape)


def _update3(arr: Any, values: Any, i_slice: slice, k_slice: slice, j_slice: slice) -> Any:
    arr_shape = _shape_tuple(arr)
    values_shape = _shape_tuple(values)
    start = (int(i_slice.start), int(k_slice.start), int(j_slice.start))
    if start == (0, 0, 0) and values_shape == arr_shape:
        return jnp.asarray(values, dtype=jnp.asarray(arr).dtype)
    return lax.dynamic_update_slice(
        arr,
        values,
        start,
    )


def _update2(arr: Any, values: Any, i_slice: slice, j_slice: slice) -> Any:
    arr_shape = _shape_tuple(arr)
    values_shape = _shape_tuple(values)
    start = (int(i_slice.start), int(j_slice.start))
    if start == (0, 0) and values_shape == arr_shape:
        return jnp.asarray(values, dtype=jnp.asarray(arr).dtype)
    return lax.dynamic_update_slice(
        arr,
        values,
        start,
    )


def _add3(arr: Any, values: Any, i_slice: slice, k_slice: slice, j_slice: slice) -> Any:
    start = (int(i_slice.start), int(k_slice.start), int(j_slice.start))
    if start == (0, 0, 0) and _shape_tuple(values) == _shape_tuple(arr):
        return jnp.asarray(arr) + jnp.asarray(values, dtype=jnp.asarray(arr).dtype)
    current = lax.dynamic_slice(arr, start, _shape_tuple(values))
    return lax.dynamic_update_slice(arr, current + values, start)


def _add2(arr: Any, values: Any, i_slice: slice, j_slice: slice) -> Any:
    start = (int(i_slice.start), int(j_slice.start))
    if start == (0, 0) and _shape_tuple(values) == _shape_tuple(arr):
        return jnp.asarray(arr) + jnp.asarray(values, dtype=jnp.asarray(arr).dtype)
    current = lax.dynamic_slice(arr, start, _shape_tuple(values))
    return lax.dynamic_update_slice(arr, current + values, start)


def _wrf_real_mul(a: Any, b: Any) -> Any:
    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) * jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_add(a: Any, b: Any) -> Any:
    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) + jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_sub(a: Any, b: Any) -> Any:
    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) - jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_div(a: Any, b: Any) -> Any:
    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) / jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


@custom_jvp
def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    array = jnp.asarray(value)
    if array.dtype == jnp.float64:
        bits = lax.bitcast_convert_type(array, jnp.uint64)
        bits = lax.bitwise_xor(bits, jnp.asarray(rounding_token, dtype=jnp.uint64))
        return lax.bitcast_convert_type(bits, jnp.float64)
    bits = lax.bitcast_convert_type(jnp.asarray(array, dtype=jnp.float32), jnp.uint32)
    bits = lax.bitwise_xor(bits, jnp.asarray(rounding_token, dtype=jnp.uint32))
    return lax.bitcast_convert_type(bits, jnp.float32)


@_store_wrf_real.defjvp
def _store_wrf_real_jvp(primals, tangents):
    value = _store_wrf_real(*primals)
    tangent = jnp.broadcast_to(jnp.asarray(tangents[0], value.dtype), value.shape)
    return value, jnp.where(jnp.asarray(primals[1]) == 0, tangent, 0)


def _divide_wrf_real(value: Any, denominator: Any, rounding_token: Any) -> Any:
    """Force IEEE division instead of XLA constant reciprocal multiplication."""

    dynamic_denominator = _store_wrf_real(denominator, rounding_token)
    return _store_wrf_real(jnp.asarray(value) / dynamic_denominator, rounding_token)


@dataclass(frozen=True)
class SmallStepPrepResult:
    u_1: Any
    u_2: Any
    v_1: Any
    v_2: Any
    w_1: Any
    w_2: Any
    t_1: Any
    t_2: Any
    ph_1: Any
    ph_2: Any
    mu_1: Any
    mu_2: Any
    muus: Any
    muvs: Any
    muts: Any
    mudf: Any
    u_save: Any
    v_save: Any
    w_save: Any
    t_save: Any
    ph_save: Any
    mu_save: Any
    ww_save: Any
    c2a: Any
    u_term_new: Any | None = None
    u_term_old: Any | None = None
    v_term_new: Any | None = None
    v_term_old: Any | None = None


@dataclass(frozen=True)
class CalcPRhoResult:
    al: Any
    p: Any
    ph: Any
    pm1: Any


@dataclass(frozen=True)
class CalcCoefWResult:
    a: Any
    alpha: Any
    gamma: Any


@dataclass(frozen=True)
class AdvanceUVResult:
    u: Any
    v: Any


@dataclass(frozen=True)
class AdvanceMuTResult:
    ww: Any
    ww_1: Any
    mu: Any
    muave: Any
    muts: Any
    mudf: Any
    uam: Any
    vam: Any
    wwam: Any
    t: Any
    t_ave: Any


@dataclass(frozen=True)
class AdvanceWResult:
    t_2ave: Any
    w: Any
    ph: Any
    raw_w_local: Any | None = None
    forward_w_local: Any | None = None
    solved_w_local: Any | None = None
    pressure_upper_local: Any | None = None
    pressure_lower_local: Any | None = None
    pressure_part_local: Any | None = None
    pressure_numerator_local: Any | None = None
    pressure_denominator_local: Any | None = None
    pressure_ratio_local: Any | None = None
    pressure_rhs_delta_local: Any | None = None
    pressure_ph_delta_local: Any | None = None
    pressure_rhs_term_local: Any | None = None
    pressure_ph_term_local: Any | None = None
    pressure_bracket_local: Any | None = None
    rhs_base_local: Any | None = None
    wdwn_local: Any | None = None
    rhs_adv_local: Any | None = None
    rhs_dec_local: Any | None = None
    thermal_upper_local: Any | None = None
    thermal_lower_local: Any | None = None
    buoyancy_part_local: Any | None = None
    damping_htop_local: Any | None = None
    damping_hbot_local: Any | None = None
    damping_hk_local: Any | None = None
    damping_angle_local: Any | None = None
    damping_sine_local: Any | None = None
    damping_weight_local: Any | None = None
    damping_mass_local: Any | None = None
    damping_term_local: Any | None = None
    damping_numerator_local: Any | None = None
    damping_denominator_local: Any | None = None
    damped_w_local: Any | None = None


@dataclass(frozen=True)
class SumFluxResult:
    ru_m: Any
    rv_m: Any
    ww_m: Any


@dataclass(frozen=True)
class SmallStepFinishResult:
    u_2: Any
    v_2: Any
    w_2: Any
    t_2: Any
    ph_2: Any
    ww: Any
    mu_2: Any
    mu_1: Any
    mut: Any
    muts: Any
    muu: Any
    muus: Any
    muv: Any
    muvs: Any
    mu_save: Any


def _i(wrf_i: int, ims: int) -> int:
    return int(wrf_i) - int(ims)


def _k(wrf_k: int, kms: int) -> int:
    return int(wrf_k) - int(kms)


def _j(wrf_j: int, jms: int) -> int:
    return int(wrf_j) - int(jms)


def _flag(config_flags: Any, name: str) -> bool:
    if config_flags is None:
        return False
    if hasattr(config_flags, "get"):
        return bool(config_flags.get(name, False))
    return bool(getattr(config_flags, name, False))


def _cfg_value(config_flags: Any, name: str, default: Any) -> Any:
    if config_flags is None:
        return default
    if hasattr(config_flags, "get"):
        return config_flags.get(name, default)
    return getattr(config_flags, name, default)


def small_step_prep(
    u_1: Any,
    u_2: Any,
    v_1: Any,
    v_2: Any,
    w_1: Any,
    w_2: Any,
    t_1: Any,
    t_2: Any,
    ph_1: Any,
    ph_2: Any,
    mub: Any,
    mu_1: Any,
    mu_2: Any,
    muu: Any,
    muus: Any,
    muv: Any,
    muvs: Any,
    mut: Any,
    muts: Any,
    mudf: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    t_save: Any,
    ph_save: Any,
    mu_save: Any,
    ww: Any,
    ww_save: Any,
    c2a: Any,
    pb: Any,
    p: Any,
    alt: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    rdx: Any,
    rdy: Any,
    rk_step: int,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> SmallStepPrepResult:
    """Run WRF ``small_step_prep`` with explicit array state."""

    del kds, ime, jme, kme, c3h, c4h, c3f, c4f, msfux, msfvx, msfvy, msftx, rdx, rdy
    dtype = jnp.result_type(
        u_1,
        u_2,
        v_1,
        v_2,
        w_1,
        w_2,
        t_1,
        t_2,
        ph_1,
        ph_2,
        mub,
        mu_1,
        mu_2,
        muu,
        muus,
        muv,
        muvs,
        mut,
        muts,
        mudf,
        c1h,
        c2h,
        c1f,
        c2f,
        u_save,
        v_save,
        w_save,
        t_save,
        ph_save,
        mu_save,
        ww,
        ww_save,
        c2a,
        pb,
        p,
        alt,
        msfuy,
        msfvx_inv,
        msfty,
        jnp.float32,
    )
    u_1_out = jnp.asarray(u_1, dtype=dtype)
    u_2_out = jnp.asarray(u_2, dtype=dtype)
    v_1_out = jnp.asarray(v_1, dtype=dtype)
    v_2_out = jnp.asarray(v_2, dtype=dtype)
    w_1_out = jnp.asarray(w_1, dtype=dtype)
    w_2_out = jnp.asarray(w_2, dtype=dtype)
    t_1_out = jnp.asarray(t_1, dtype=dtype)
    t_2_out = jnp.asarray(t_2, dtype=dtype)
    ph_1_out = jnp.asarray(ph_1, dtype=dtype)
    ph_2_out = jnp.asarray(ph_2, dtype=dtype)
    mub_arr = jnp.asarray(mub, dtype=dtype)
    mu_1_out = jnp.asarray(mu_1, dtype=dtype)
    mu_2_out = jnp.asarray(mu_2, dtype=dtype)
    muu_arr = jnp.asarray(muu, dtype=dtype)
    muus_out = jnp.asarray(muus, dtype=dtype)
    muv_arr = jnp.asarray(muv, dtype=dtype)
    muvs_out = jnp.asarray(muvs, dtype=dtype)
    mut_arr = jnp.asarray(mut, dtype=dtype)
    muts_out = jnp.asarray(muts, dtype=dtype)
    mudf_out = jnp.asarray(mudf, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    c1f_arr = jnp.asarray(c1f, dtype=dtype)
    c2f_arr = jnp.asarray(c2f, dtype=dtype)
    u_save_out = jnp.asarray(u_save, dtype=dtype)
    v_save_out = jnp.asarray(v_save, dtype=dtype)
    w_save_out = jnp.asarray(w_save, dtype=dtype)
    t_save_out = jnp.asarray(t_save, dtype=dtype)
    ph_save_out = jnp.asarray(ph_save, dtype=dtype)
    mu_save_out = jnp.asarray(mu_save, dtype=dtype)
    ww_arr = jnp.asarray(ww, dtype=dtype)
    ww_save_out = jnp.asarray(ww_save, dtype=dtype)
    c2a_out = jnp.asarray(c2a, dtype=dtype)
    pb_arr = jnp.asarray(pb, dtype=dtype)
    p_arr = jnp.asarray(p, dtype=dtype)
    alt_arr = jnp.asarray(alt, dtype=dtype)
    msfuy_arr = jnp.asarray(msfuy, dtype=dtype)
    msfvx_inv_arr = jnp.asarray(msfvx_inv, dtype=dtype)
    msfty_arr = jnp.asarray(msfty, dtype=dtype)

    def store(value: Any) -> Any:
        value32 = jnp.asarray(value, dtype=jnp.float32)
        if rounding_token is None:
            return lax.optimization_barrier(value32)
        return _store_wrf_real(value32, rounding_token)

    def add_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=jnp.float32)
            + jnp.asarray(right, dtype=jnp.float32)
        )

    def sub_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=jnp.float32)
            - jnp.asarray(right, dtype=jnp.float32)
        )

    def mul_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=jnp.float32)
            * jnp.asarray(right, dtype=jnp.float32)
        )

    def div_real(left: Any, right: Any) -> Any:
        if rounding_token is None:
            return store(
                jnp.asarray(left, dtype=jnp.float32)
                / jnp.asarray(right, dtype=jnp.float32)
            )
        return _divide_wrf_real(left, right, rounding_token)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = min(int(kte), int(kde) - 1)
    i_endu = int(ite)
    j_endv = int(jte)

    i_m_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    i_u_slice = slice(i_start - int(ims), i_endu - int(ims) + 1)
    j_m_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    j_v_slice = slice(j_start - int(jms), j_endv - int(jms) + 1)
    k_m_slice = slice(k_start - int(kms), k_end - int(kms) + 1)
    k_w_copy_slice = slice(k_start - int(kms), min(int(kde), int(kte)) - int(kms) + 1)
    k_w_full_slice = slice(k_start - int(kms), int(kde) - int(kms) + 1)

    def field2(arr: Any, i_slice: slice, j_slice: slice) -> Any:
        return arr[i_slice, j_slice]

    def field3(arr: Any, i_slice: slice, k_slice: slice, j_slice: slice) -> Any:
        return arr[i_slice, k_slice, j_slice]

    def west_neighbor2(arr: Any) -> Any:
        if i_start > int(ims):
            return arr[slice(i_u_slice.start - 1, i_u_slice.stop - 1), j_m_slice]
        current = arr[i_u_slice, j_m_slice]
        if i_u_slice.stop - i_u_slice.start <= 1:
            return current
        shifted = arr[slice(i_u_slice.start, i_u_slice.stop - 1), j_m_slice]
        return jnp.concatenate([current[:1, :], shifted], axis=0)

    def south_neighbor2(arr: Any) -> Any:
        if j_start > int(jms):
            return arr[i_m_slice, slice(j_v_slice.start - 1, j_v_slice.stop - 1)]
        current = arr[i_m_slice, j_v_slice]
        if j_v_slice.stop - j_v_slice.start <= 1:
            return current
        shifted = arr[i_m_slice, slice(j_v_slice.start, j_v_slice.stop - 1)]
        return jnp.concatenate([current[:, :1], shifted], axis=1)

    if int(rk_step) == 1:
        mu_2_mass = field2(mu_2_out, i_m_slice, j_m_slice)
        mu_1_out = _update2(mu_1_out, mu_2_mass, i_m_slice, j_m_slice)
        mudf_out = _update2(mudf_out, jnp.zeros_like(field2(mudf_out, i_m_slice, j_m_slice)), i_m_slice, j_m_slice)
        u_1_out = _update3(u_1_out, field3(u_2_out, i_u_slice, k_m_slice, j_m_slice), i_u_slice, k_m_slice, j_m_slice)
        v_1_out = _update3(v_1_out, field3(v_2_out, i_m_slice, k_m_slice, j_v_slice), i_m_slice, k_m_slice, j_v_slice)
        t_1_out = _update3(t_1_out, field3(t_2_out, i_m_slice, k_m_slice, j_m_slice), i_m_slice, k_m_slice, j_m_slice)
        w_1_out = _update3(w_1_out, field3(w_2_out, i_m_slice, k_w_copy_slice, j_m_slice), i_m_slice, k_w_copy_slice, j_m_slice)
        ph_1_out = _update3(ph_1_out, field3(ph_2_out, i_m_slice, k_w_copy_slice, j_m_slice), i_m_slice, k_w_copy_slice, j_m_slice)
        muts_out = _update2(
            muts_out,
            add_real(field2(mub_arr, i_m_slice, j_m_slice), mu_2_mass),
            i_m_slice,
            j_m_slice,
        )
        muus_out = _update2(muus_out, field2(muu_arr, i_u_slice, j_m_slice), i_u_slice, j_m_slice)
        muvs_out = _update2(muvs_out, field2(muv_arr, i_m_slice, j_v_slice), i_m_slice, j_v_slice)
        mu_save_out = _update2(mu_save_out, mu_2_mass, i_m_slice, j_m_slice)
        mu_2_out = _update2(mu_2_out, jnp.zeros_like(mu_2_mass), i_m_slice, j_m_slice)
    else:
        mu_1_mass = field2(mu_1_out, i_m_slice, j_m_slice)
        muts_out = _update2(
            muts_out,
            add_real(field2(mub_arr, i_m_slice, j_m_slice), mu_1_mass),
            i_m_slice,
            j_m_slice,
        )
        muus_sum = add_real(
            field2(mub_arr, i_u_slice, j_m_slice),
            field2(mu_1_out, i_u_slice, j_m_slice),
        )
        muus_sum = add_real(muus_sum, west_neighbor2(mub_arr))
        muus_sum = add_real(muus_sum, west_neighbor2(mu_1_out))
        muus_value = mul_real(store(0.5), muus_sum)
        muvs_sum = add_real(
            field2(mub_arr, i_m_slice, j_v_slice),
            field2(mu_1_out, i_m_slice, j_v_slice),
        )
        muvs_sum = add_real(muvs_sum, south_neighbor2(mub_arr))
        muvs_sum = add_real(muvs_sum, south_neighbor2(mu_1_out))
        muvs_value = mul_real(store(0.5), muvs_sum)
        mu_2_mass = field2(mu_2_out, i_m_slice, j_m_slice)
        muus_out = _update2(muus_out, muus_value, i_u_slice, j_m_slice)
        muvs_out = _update2(muvs_out, muvs_value, i_m_slice, j_v_slice)
        mu_save_out = _update2(mu_save_out, mu_2_mass, i_m_slice, j_m_slice)
        mu_2_out = _update2(
            mu_2_out,
            sub_real(mu_1_mass, mu_2_mass),
            i_m_slice,
            j_m_slice,
        )

    k_top_slice = slice(_k(kde, kms), _k(kde, kms) + 1)
    k_bottom_slice = slice(_k(1, kms), _k(1, kms) + 1)
    ww_save_out = _update3(
        ww_save_out,
        jnp.zeros_like(field3(ww_save_out, i_m_slice, k_top_slice, j_m_slice)),
        i_m_slice,
        k_top_slice,
        j_m_slice,
    )
    ww_save_out = _update3(
        ww_save_out,
        jnp.zeros_like(field3(ww_save_out, i_m_slice, k_bottom_slice, j_m_slice)),
        i_m_slice,
        k_bottom_slice,
        j_m_slice,
    )

    c2a_numerator = mul_real(
        store(CPOVCV),
        add_real(
            field3(pb_arr, i_m_slice, k_m_slice, j_m_slice),
            field3(p_arr, i_m_slice, k_m_slice, j_m_slice),
        ),
    )
    c2a_value = div_real(
        c2a_numerator,
        field3(alt_arr, i_m_slice, k_m_slice, j_m_slice),
    )
    c2a_out = _update3(c2a_out, c2a_value, i_m_slice, k_m_slice, j_m_slice)

    u_old = field3(u_2_out, i_u_slice, k_m_slice, j_m_slice)
    u_save_out = _update3(u_save_out, u_old, i_u_slice, k_m_slice, j_m_slice)
    u_muus = field2(muus_out, i_u_slice, j_m_slice)[:, None, :]
    u_muu = field2(muu_arr, i_u_slice, j_m_slice)[:, None, :]
    u_metric = field2(msfuy_arr, i_u_slice, j_m_slice)[:, None, :]
    u_c1h = c1h_arr[k_m_slice][None, :, None]
    u_c2h = c2h_arr[k_m_slice][None, :, None]
    u_mass_new = add_real(mul_real(u_c1h, u_muus), u_c2h)
    u_mass_old = add_real(mul_real(u_c1h, u_muu), u_c2h)
    if int(rk_step) == 1:
        u_term_new = jnp.zeros_like(u_old)
        u_term_old = jnp.zeros_like(u_old)
        u_value = jnp.zeros_like(u_old)
    else:
        u_term_new = mul_real(
            u_mass_new,
            field3(u_1_out, i_u_slice, k_m_slice, j_m_slice),
        )
        u_term_old = mul_real(u_mass_old, u_old)
        u_value = div_real(
            sub_real(u_term_new, u_term_old),
            u_metric,
        )
    u_2_out = _update3(u_2_out, u_value, i_u_slice, k_m_slice, j_m_slice)

    v_old = field3(v_2_out, i_m_slice, k_m_slice, j_v_slice)
    v_save_out = _update3(v_save_out, v_old, i_m_slice, k_m_slice, j_v_slice)
    v_muvs = field2(muvs_out, i_m_slice, j_v_slice)[:, None, :]
    v_muv = field2(muv_arr, i_m_slice, j_v_slice)[:, None, :]
    v_metric_inv = field2(msfvx_inv_arr, i_m_slice, j_v_slice)[:, None, :]
    v_c1h = c1h_arr[k_m_slice][None, :, None]
    v_c2h = c2h_arr[k_m_slice][None, :, None]
    v_mass_new = add_real(mul_real(v_c1h, v_muvs), v_c2h)
    v_mass_old = add_real(mul_real(v_c1h, v_muv), v_c2h)
    if int(rk_step) == 1:
        v_term_new = jnp.zeros_like(v_old)
        v_term_old = jnp.zeros_like(v_old)
        v_value = jnp.zeros_like(v_old)
    else:
        v_term_new = mul_real(
            v_mass_new,
            field3(v_1_out, i_m_slice, k_m_slice, j_v_slice),
        )
        v_term_old = mul_real(v_mass_old, v_old)
        v_value = mul_real(
            sub_real(v_term_new, v_term_old),
            v_metric_inv,
        )
    v_2_out = _update3(v_2_out, v_value, i_m_slice, k_m_slice, j_v_slice)

    t_old = field3(t_2_out, i_m_slice, k_m_slice, j_m_slice)
    t_save_out = _update3(t_save_out, t_old, i_m_slice, k_m_slice, j_m_slice)
    t_muts = field2(muts_out, i_m_slice, j_m_slice)[:, None, :]
    t_mut = field2(mut_arr, i_m_slice, j_m_slice)[:, None, :]
    t_c1h = c1h_arr[k_m_slice][None, :, None]
    t_c2h = c2h_arr[k_m_slice][None, :, None]
    t_mass_new = add_real(mul_real(t_c1h, t_muts), t_c2h)
    t_mass_old = add_real(mul_real(t_c1h, t_mut), t_c2h)
    if int(rk_step) == 1:
        t_value = jnp.zeros_like(t_old)
    else:
        t_value = sub_real(
            mul_real(
                t_mass_new,
                field3(t_1_out, i_m_slice, k_m_slice, j_m_slice),
            ),
            mul_real(t_mass_old, t_old),
        )
    t_2_out = _update3(t_2_out, t_value, i_m_slice, k_m_slice, j_m_slice)

    w_old = field3(w_2_out, i_m_slice, k_w_full_slice, j_m_slice)
    w_save_out = _update3(w_save_out, w_old, i_m_slice, k_w_full_slice, j_m_slice)
    w_muts = field2(muts_out, i_m_slice, j_m_slice)[:, None, :]
    w_mut = field2(mut_arr, i_m_slice, j_m_slice)[:, None, :]
    w_metric = field2(msfty_arr, i_m_slice, j_m_slice)[:, None, :]
    w_c1f = c1f_arr[k_w_full_slice][None, :, None]
    w_c2f = c2f_arr[k_w_full_slice][None, :, None]
    w_mass_new = add_real(mul_real(w_c1f, w_muts), w_c2f)
    w_mass_old = add_real(mul_real(w_c1f, w_mut), w_c2f)
    if int(rk_step) == 1:
        w_value = jnp.zeros_like(w_old)
    else:
        w_value = div_real(
            sub_real(
                mul_real(
                    w_mass_new,
                    field3(w_1_out, i_m_slice, k_w_full_slice, j_m_slice),
                ),
                mul_real(w_mass_old, w_old),
            ),
            w_metric,
        )
    w_2_out = _update3(w_2_out, w_value, i_m_slice, k_w_full_slice, j_m_slice)
    ph_old = field3(ph_2_out, i_m_slice, k_w_full_slice, j_m_slice)
    ph_save_out = _update3(ph_save_out, ph_old, i_m_slice, k_w_full_slice, j_m_slice)
    ph_2_out = _update3(
        ph_2_out,
        sub_real(field3(ph_1_out, i_m_slice, k_w_full_slice, j_m_slice), ph_old),
        i_m_slice,
        k_w_full_slice,
        j_m_slice,
    )
    ww_save_out = _update3(ww_save_out, field3(ww_arr, i_m_slice, k_w_full_slice, j_m_slice), i_m_slice, k_w_full_slice, j_m_slice)

    return SmallStepPrepResult(
        u_1=u_1_out,
        u_2=u_2_out,
        v_1=v_1_out,
        v_2=v_2_out,
        w_1=w_1_out,
        w_2=w_2_out,
        t_1=t_1_out,
        t_2=t_2_out,
        ph_1=ph_1_out,
        ph_2=ph_2_out,
        mu_1=mu_1_out,
        mu_2=mu_2_out,
        muus=muus_out,
        muvs=muvs_out,
        muts=muts_out,
        mudf=mudf_out,
        u_save=u_save_out,
        v_save=v_save_out,
        w_save=w_save_out,
        t_save=t_save_out,
        ph_save=ph_save_out,
        mu_save=mu_save_out,
        ww_save=ww_save_out,
        c2a=c2a_out,
        u_term_new=u_term_new,
        u_term_old=u_term_old,
        v_term_new=v_term_new,
        v_term_old=v_term_old,
    )


def _nonhydro_pressure_elements(values, parameters, *, first_step):
    mut, mu, c1h, c2h, alt, theta1, ph_upper, ph_lower, rdnw, theta2, c2a, pm1 = values
    token, t0, smdiv = parameters
    dtype = alt.dtype

    def store(value):
        return value if token is None else _store_wrf_real(value, token)

    def binary(operation, left, right):
        return store(column_binary(operation, jnp.asarray(left, dtype), jnp.asarray(right, dtype)))

    def divide(left, right):
        return binary("div", left, store(jnp.asarray(right, dtype)))

    mass = binary("add", binary("mul", c1h, mut), c2h)
    dry_mass = binary("mul", c1h, mu)
    ph_delta = binary("sub", ph_upper, ph_lower)
    bracket = binary("add", binary("mul", alt, dry_mass), binary("mul", rdnw, ph_delta))
    al_value = binary("mul", divide(-1.0, mass), bracket)
    temperature_delta = binary("sub", theta2, binary("mul", dry_mass, theta1))
    pressure_fraction = divide(
        binary("mul", alt, temperature_delta),
        binary("mul", mass, binary("add", t0, theta1)),
    )
    pressure = binary("mul", c2a, binary("sub", pressure_fraction, al_value))
    filtered = pressure if first_step else binary(
        "add", pressure, binary("mul", smdiv, binary("sub", pressure, pm1))
    )
    return al_value, filtered, pressure


def _nonhydro_pressure_gather(indices, parameters, *, shapes, first_step):
    i, k, j = indices
    fields, arithmetic = parameters
    values = []
    for position, (field, shape) in enumerate(zip(fields, shapes, strict=True)):
        if len(shape) == 1:
            offset = k
        elif len(shape) == 2:
            offset = i * shape[1] + j
        else:
            # The upper geopotential interface is one level above the cell.
            level = k + 1 if position == 6 else k
            offset = (i * shape[1] + level) * shape[2] + j
        values.append(field[offset])
    return _nonhydro_pressure_elements(tuple(values), arithmetic, first_step=first_step)


def _nonhydro_pressure_block(arrays, arithmetic, *, region, first_step):
    i_slice, k_slice, j_slice = region
    values = []
    for position, field in enumerate(arrays):
        if field.ndim == 1:
            value = field[k_slice][None, :, None]
        elif field.ndim == 2:
            value = field[i_slice, j_slice][:, None, :]
        else:
            level = slice(k_slice.start + 1, k_slice.stop + 1) if position == 6 else k_slice
            value = field[i_slice, level, j_slice]
        values.append(value)
    return _nonhydro_pressure_elements(tuple(values), arithmetic, first_step=first_step)


def _nonhydro_pressure_full(indices, parameters, *, shapes, region, first_step, output_shape):
    fields, arithmetic, previous = parameters
    i, k, j = indices
    offset = (i * output_shape[1] + k) * output_shape[2] + j
    active = jnp.asarray(True)
    clipped = []
    for index, (start, stop) in zip(indices, region, strict=True):
        active = active & (index >= start) & (index < stop)
        clipped.append(jnp.clip(index, start, stop - 1))
    values = _nonhydro_pressure_gather(
        tuple(clipped), (fields, arithmetic), shapes=shapes, first_step=first_step,
    )
    return tuple(jnp.where(active, value, old[offset])
                 for value, old in zip(values, previous, strict=True))


def calc_p_rho(
    al: Any,
    p: Any,
    ph: Any,
    alt: Any,
    t_2: Any,
    t_1: Any,
    c2a: Any,
    pm1: Any,
    mu: Any,
    mut: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    znu: Any,
    t0: Any,
    rdnw: Any,
    dnw: Any,
    smdiv: Any,
    non_hydrostatic: bool,
    step: int,
    *,
    rounding_token: Any | None = None,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> CalcPRhoResult:
    """Run WRF ``calc_p_rho`` with explicit array state."""

    del ids, jds, kds, ime, jme, kme, c1f, c2f, c4h, c3f, c4f, znu
    dtype = jnp.result_type(
        al,
        p,
        ph,
        alt,
        t_2,
        t_1,
        c2a,
        pm1,
        mu,
        mut,
        c1h,
        c2h,
        c3h,
        t0,
        rdnw,
        dnw,
        smdiv,
        jnp.float32,
    )
    al_out = jnp.asarray(al, dtype=dtype)
    p_out = jnp.asarray(p, dtype=dtype)
    ph_out = jnp.asarray(ph, dtype=dtype)
    alt_arr = jnp.asarray(alt, dtype=dtype)
    t_2_arr = jnp.asarray(t_2, dtype=dtype)
    t_1_arr = jnp.asarray(t_1, dtype=dtype)
    c2a_arr = jnp.asarray(c2a, dtype=dtype)
    pm1_out = jnp.asarray(pm1, dtype=dtype)
    mu_arr = jnp.asarray(mu, dtype=dtype)
    mut_arr = jnp.asarray(mut, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    c3h_arr = jnp.asarray(c3h, dtype=dtype)
    rdnw_arr = jnp.asarray(rdnw, dtype=dtype)
    dnw_arr = jnp.asarray(dnw, dtype=dtype)
    t0_val = jnp.asarray(t0, dtype=dtype)
    smdiv_val = jnp.asarray(smdiv, dtype=dtype)

    store = (
        (lambda value: value)
        if rounding_token is None
        else (lambda value: _store_wrf_real(value, rounding_token))
    )

    def add_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype)
        )

    def sub_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype)
        )

    def mul_real(left: Any, right: Any) -> Any:
        return store(
            jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype)
        )

    def div_real(left: Any, right: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(left, dtype=dtype) / jnp.asarray(right, dtype=dtype)
        return _divide_wrf_real(left, right, rounding_token)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = min(int(kte), int(kde) - 1)

    i_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    j_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    k_slice = slice(k_start - int(kms), k_end - int(kms) + 1)
    i_fortran = jnp.arange(i_start, i_end + 1)
    j_fortran = jnp.arange(j_start, j_end + 1)
    k_fortran = jnp.arange(k_start, k_end + 1)
    ii = i_fortran - int(ims)
    jj = j_fortran - int(jms)
    kk = k_fortran - int(kms)
    if bool(non_hydrostatic):
        arrays = (
            mut_arr, mu_arr, c1h_arr, c2h_arr, alt_arr, t_1_arr,
            ph_out, ph_out, rdnw_arr, t_2_arr, c2a_arr, pm1_out,
        )
        arithmetic = (rounding_token, t0_val, smdiv_val)
        region = (i_slice, k_slice, j_slice)
        block = partial(
            _nonhydro_pressure_block, region=region, first_step=int(step) == 0,
        )

        def sliced(fields, parameters, previous):
            values = block(fields, parameters)
            return tuple(_update3(old, value, *region)
                         for old, value in zip(previous, values, strict=True))

        def mapped(fields, parameters, previous):
            shape = previous[0].shape
            indices = (jnp.arange(shape[0])[:, None, None],
                       jnp.arange(shape[1])[None, :, None],
                       jnp.arange(shape[2])[None, None, :])
            body = partial(
                _nonhydro_pressure_full, shapes=tuple(value.shape for value in fields),
                region=tuple((axis.start, axis.stop) for axis in region),
                first_step=int(step) == 0, output_shape=shape,
            )
            return map_elements(body, indices, (
                tuple(field.reshape(-1) for field in fields), parameters,
                tuple(field.reshape(-1) for field in previous),
            ))

        previous = (al_out, p_out, pm1_out)
        if any(axis.stop <= axis.start for axis in region):
            return CalcPRhoResult(al=al_out, p=p_out, ph=ph_out, pm1=pm1_out)
        # CUDA emits complete fields so a tile update needs no separate copy.
        if (rounding_token is None or jnp.ndim(rounding_token) != 0
                or any(value.shape != al_out.shape for value in previous)):
            al_result, p_result, pm1_result = sliced(arrays, arithmetic, previous)
        else:
            al_result, p_result, pm1_result = lax.platform_dependent(
                arrays, arithmetic, previous, cuda=mapped, default=sliced,
            )
        return CalcPRhoResult(
            al=al_result, p=p_result, ph=ph_out, pm1=pm1_result,
        )
    else:
        mut_block = mut_arr[ii[:, None], jj[None, :]][:, None, :]
        mu_block = mu_arr[ii[:, None], jj[None, :]][:, None, :]
        for kk_scalar in range(int(k_start), int(k_end) + 1):
            kk_idx = kk_scalar - int(kms)
            kkp1_idx = kk_scalar + 1 - int(kms)
            idx_k = (ii[:, None], kk_idx, jj[None, :])
            idx_kp1 = (ii[:, None], kkp1_idx, jj[None, :])
            mass_k = add_real(
                mul_real(c1h_arr[kk_idx], mut_block[:, 0, :]),
                c2h_arr[kk_idx],
            )
            dry_mass_k = mul_real(c1h_arr[kk_idx], mu_block[:, 0, :])
            p_value = mul_real(mu_block[:, 0, :], c3h_arr[kk_idx])
            al_value = sub_real(
                div_real(
                    mul_real(
                        alt_arr[idx_k],
                        sub_real(
                            t_2_arr[idx_k],
                            mul_real(dry_mass_k, t_1_arr[idx_k]),
                        ),
                    ),
                    mul_real(mass_k, add_real(t0_val, t_1_arr[idx_k])),
                ),
                div_real(p_value, c2a_arr[idx_k]),
            )
            ph_value = sub_real(
                ph_out[idx_k],
                mul_real(
                    dnw_arr[kk_idx],
                    add_real(
                        mul_real(mass_k, al_value),
                        mul_real(dry_mass_k, alt_arr[idx_k]),
                    ),
                ),
            )
            p_out = p_out.at[idx_k].set(p_value)
            al_out = al_out.at[idx_k].set(al_value)
            ph_out = ph_out.at[idx_kp1].set(ph_value)

    if int(step) == 0:
        pm1_out = _update3(pm1_out, p_out[i_slice, k_slice, j_slice], i_slice, k_slice, j_slice)
    else:
        ptmp = p_out[i_slice, k_slice, j_slice]
        p_out = _update3(
            p_out,
            add_real(
                ptmp,
                mul_real(
                    smdiv_val,
                    sub_real(ptmp, pm1_out[i_slice, k_slice, j_slice]),
                ),
            ),
            i_slice,
            k_slice,
            j_slice,
        )
        pm1_out = _update3(pm1_out, ptmp, i_slice, k_slice, j_slice)

    return CalcPRhoResult(al=al_out, p=p_out, ph=ph_out, pm1=pm1_out)


def _small_finish_real(operation, left, right, token):
    left, right = jnp.asarray(left, jnp.float32), jnp.asarray(right, jnp.float32)
    if operation == "div" and token is not None:
        right = _store_wrf_real(right, token)
    value = column_binary(operation, left, right)
    return column_barrier(value) if token is None else _store_wrf_real(value, token)


def _small_finish_velocity(values, token):
    c1, c2, mass_old, mass_new, map_factor, velocity, saved = values
    operation = lambda name, a, b: _small_finish_real(name, a, b, token)
    old = operation("add", operation("mul", c1, mass_old), c2)
    new = operation("add", operation("mul", c1, mass_new), c2)
    numerator = operation("add", operation("mul", map_factor, velocity),
                          operation("mul", saved, old))
    return operation("div", numerator, new)


def _small_finish_temperature(values, parameters, *, final_step):
    c1, c2, mass_old, mass_new, temperature, saved, diabatic = values
    token, dts, nsmall = parameters
    operation = lambda name, a, b: _small_finish_real(name, a, b, token)
    old = operation("add", operation("mul", c1, mass_old), c2)
    new = operation("add", operation("mul", c1, mass_new), c2)
    if final_step:
        heating = operation("mul", operation("mul", operation("mul", dts, nsmall), old), diabatic)
        temperature = operation("sub", temperature, heating)
    numerator = operation("add", temperature, operation("mul", saved, old))
    return operation("div", numerator, new)


def _small_finish_gather(indices, parameters, *, body, shapes, region, output_index):
    i, k, j = indices
    fields, arithmetic = parameters
    target_shape = shapes[output_index]
    original = fields[output_index][(i * target_shape[1] + k) * target_shape[2] + j]
    active = jnp.asarray(True)
    clipped = []
    for coordinate, part in zip(indices, region, strict=True):
        active = active & (coordinate >= part.start) & (coordinate < part.stop)
        clipped.append(jnp.clip(coordinate, part.start, part.stop - 1))
    i, k, j = clipped
    values = []
    for field, shape in zip(fields, shapes, strict=True):
        offset = k if len(shape) == 1 else i * shape[-1] + j
        if len(shape) == 3:
            offset = (i * shape[1] + k) * shape[2] + j
        values.append(field[offset])
    return jnp.where(active, body(tuple(values), arithmetic), original)


def _small_finish_update(body, fields, arithmetic, region, output_index):
    if any(part.stop <= part.start for part in region):
        return fields[output_index]

    def block(arrays, parameters):
        i, k, j = region
        values = tuple(
            value[k][None, :, None] if value.ndim == 1
            else value[i, j][:, None, :] if value.ndim == 2
            else value[i, k, j]
            for value in arrays
        )
        return _update3(arrays[output_index], body(values, parameters), i, k, j)

    def mapped(arrays, parameters):
        # Emit the full result in one kernel, preserving cells outside the tile
        # without a separate whole-field copy and dynamic-update operation.
        i, k, j = (jnp.arange(size, dtype=jnp.int32) for size in arrays[output_index].shape)
        return map_elements(
            partial(_small_finish_gather, body=body, shapes=tuple(value.shape for value in arrays),
                    region=region, output_index=output_index),
            (i[:, None, None], k[None, :, None], j[None, None, :]),
            (tuple(value.reshape(-1) for value in arrays), parameters),
        )

    token = arithmetic[0] if isinstance(arithmetic, tuple) else arithmetic
    if token is not None and jnp.ndim(token) != 0:
        return block(fields, arithmetic)
    return lax.platform_dependent(fields, arithmetic, cuda=mapped, default=block)


def small_step_finish(
    u_2: Any,
    u_1: Any,
    v_2: Any,
    v_1: Any,
    w_2: Any,
    w_1: Any,
    t_2: Any,
    t_1: Any,
    ph_2: Any,
    ph_1: Any,
    ww: Any,
    ww1: Any,
    mu_2: Any,
    mu_1: Any,
    mut: Any,
    muts: Any,
    muu: Any,
    muus: Any,
    muv: Any,
    muvs: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    u_save: Any,
    v_save: Any,
    w_save: Any,
    t_save: Any,
    ph_save: Any,
    mu_save: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    h_diabatic: Any,
    number_of_small_timesteps: int,
    dts: Any,
    rk_step: int,
    rk_order: int,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    rounding_token: Any | None = None,
) -> SmallStepFinishResult:
    """Run WRF ``small_step_finish`` with explicit array state."""

    del ids, jds, ime, jme, kme, kts, kte, u_1, v_1, w_1, t_1, ph_1, c3h, c4h, c3f, c4f, msfux, msfvy, msftx
    dtype = jnp.result_type(
        u_2,
        v_2,
        w_2,
        t_2,
        ph_2,
        ww,
        ww1,
        mu_2,
        mu_1,
        mut,
        muts,
        muu,
        muus,
        muv,
        muvs,
        c1h,
        c2h,
        c1f,
        c2f,
        u_save,
        v_save,
        w_save,
        t_save,
        ph_save,
        mu_save,
        msfuy,
        msfvx,
        msfty,
        h_diabatic,
        dts,
        jnp.float32,
    )
    u_2_out = jnp.asarray(u_2, dtype=dtype)
    v_2_out = jnp.asarray(v_2, dtype=dtype)
    w_2_out = jnp.asarray(w_2, dtype=dtype)
    t_2_out = jnp.asarray(t_2, dtype=dtype)
    ph_2_out = jnp.asarray(ph_2, dtype=dtype)
    ww_out = jnp.asarray(ww, dtype=dtype)
    ww1_arr = jnp.asarray(ww1, dtype=dtype)
    mu_2_out = jnp.asarray(mu_2, dtype=dtype)
    mu_1_out = jnp.asarray(mu_1, dtype=dtype)
    mut_out = jnp.asarray(mut, dtype=dtype)
    muts_out = jnp.asarray(muts, dtype=dtype)
    muu_out = jnp.asarray(muu, dtype=dtype)
    muus_out = jnp.asarray(muus, dtype=dtype)
    muv_out = jnp.asarray(muv, dtype=dtype)
    muvs_out = jnp.asarray(muvs, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    c1f_arr = jnp.asarray(c1f, dtype=dtype)
    c2f_arr = jnp.asarray(c2f, dtype=dtype)
    u_save_arr = jnp.asarray(u_save, dtype=dtype)
    v_save_arr = jnp.asarray(v_save, dtype=dtype)
    w_save_arr = jnp.asarray(w_save, dtype=dtype)
    t_save_arr = jnp.asarray(t_save, dtype=dtype)
    ph_save_arr = jnp.asarray(ph_save, dtype=dtype)
    mu_save_out = jnp.asarray(mu_save, dtype=dtype)
    msfuy_arr = jnp.asarray(msfuy, dtype=dtype)
    msfvx_arr = jnp.asarray(msfvx, dtype=dtype)
    msfty_arr = jnp.asarray(msfty, dtype=dtype)
    h_diabatic_arr = jnp.asarray(h_diabatic, dtype=dtype)
    dts_val = jnp.asarray(dts, dtype=dtype)
    nsmall = jnp.asarray(number_of_small_timesteps, dtype=dtype)

    def add_real(left: Any, right: Any) -> Any:
        return _small_finish_real("add", left, right, rounding_token)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    i_endu = int(ite)
    j_endv = int(jte)

    i_mass = slice(_i(i_start, ims), _i(i_end, ims) + 1)
    i_u = slice(_i(i_start, ims), _i(i_endu, ims) + 1)
    j_mass = slice(_j(j_start, jms), _j(j_end, jms) + 1)
    j_v = slice(_j(j_start, jms), _j(j_endv, jms) + 1)
    k_mass = slice(_k(int(kds), kms), _k(int(kde) - 1, kms) + 1)
    k_w = slice(_k(int(kds), kms), _k(int(kde), kms) + 1)

    v_2_out = _small_finish_update(
        _small_finish_velocity,
        (c1h_arr, c2h_arr, muv_out, muvs_out, msfvx_arr, v_2_out, v_save_arr),
        rounding_token, (i_mass, k_mass, j_v), 5,
    )

    u_2_out = _small_finish_update(
        _small_finish_velocity,
        (c1h_arr, c2h_arr, muu_out, muus_out, msfuy_arr, u_2_out, u_save_arr),
        rounding_token, (i_u, k_mass, j_mass), 5,
    )

    w_2_out = _small_finish_update(
        _small_finish_velocity,
        (c1f_arr, c2f_arr, mut_out, muts_out, msfty_arr, w_2_out, w_save_arr),
        rounding_token, (i_mass, k_w, j_mass), 5,
    )
    ph_2_out = _update3(
        ph_2_out,
        add_real(ph_2_out[i_mass, k_w, j_mass], ph_save_arr[i_mass, k_w, j_mass]),
        i_mass,
        k_w,
        j_mass,
    )
    ww_out = _update3(
        ww_out,
        add_real(ww_out[i_mass, k_w, j_mass], ww1_arr[i_mass, k_w, j_mass]),
        i_mass,
        k_w,
        j_mass,
    )

    t_2_out = _small_finish_update(
        partial(_small_finish_temperature, final_step=int(rk_step) >= int(rk_order)),
        (c1h_arr, c2h_arr, mut_out, muts_out, t_2_out, t_save_arr, h_diabatic_arr),
        (rounding_token, dts_val, nsmall), (i_mass, k_mass, j_mass), 4,
    )

    mu_2_out = _update2(
        mu_2_out,
        add_real(mu_2_out[i_mass, j_mass], mu_save_out[i_mass, j_mass]),
        i_mass,
        j_mass,
    )

    return SmallStepFinishResult(
        u_2=u_2_out,
        v_2=v_2_out,
        w_2=w_2_out,
        t_2=t_2_out,
        ph_2=ph_2_out,
        ww=ww_out,
        mu_2=mu_2_out,
        mu_1=mu_1_out,
        mut=mut_out,
        muts=muts_out,
        muu=muu_out,
        muus=muus_out,
        muv=muv_out,
        muvs=muvs_out,
        mu_save=mu_save_out,
    )


def _acoustic_coefficient_step(previous_gamma, inputs, parameter):
    """Eliminate one tridiagonal level with the original REAL stores."""
    a, b, c = inputs
    token, one = parameter
    dtype = previous_gamma.dtype

    def operation(name, left, right):
        left, right = jnp.asarray(left, dtype), jnp.asarray(right, dtype)
        if token is None:
            if name == "div":
                return column_binary(name, left, right)
            return {"mul": lax.mul, "sub": lax.sub}[name](left, right)
        if name == "div":
            left = _store_wrf_real(left, token)
            right = _store_wrf_real(_store_wrf_real(right, token), token)
        return _store_wrf_real(column_binary(name, left, right), token)

    alpha = operation(
        "div", one, operation("sub", b, operation("mul", a, previous_gamma))
    )
    gamma = operation("mul", c, alpha)
    return gamma, (alpha, gamma)


def calc_coef_w(
    a: Any,
    alpha: Any,
    gamma: Any,
    mut: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    cqw: Any,
    rdn: Any,
    rdnw: Any,
    c2a: Any,
    dts: Any,
    g: Any,
    epssm: Any,
    top_lid: bool,
    rounding_token: Any | None = None,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> CalcCoefWResult:
    """Run WRF ``calc_coef_w`` with explicit array state."""

    del ids, jds, kds, ime, jme, kme, c3h, c4h, c3f, c4f, kts, kte
    dtype = jnp.result_type(
        a,
        alpha,
        gamma,
        mut,
        c1h,
        c2h,
        c1f,
        c2f,
        cqw,
        rdn,
        rdnw,
        c2a,
        dts,
        g,
        epssm,
        jnp.float32,
    )
    a_out = jnp.asarray(a, dtype=dtype)
    alpha_out = jnp.asarray(alpha, dtype=dtype)
    gamma_out = jnp.asarray(gamma, dtype=dtype)
    mut_arr = jnp.asarray(mut, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    c1f_arr = jnp.asarray(c1f, dtype=dtype)
    c2f_arr = jnp.asarray(c2f, dtype=dtype)
    cqw_arr = jnp.asarray(cqw, dtype=dtype)
    rdn_arr = jnp.asarray(rdn, dtype=dtype)
    rdnw_arr = jnp.asarray(rdnw, dtype=dtype)
    c2a_arr = jnp.asarray(c2a, dtype=dtype)
    store = (lambda value: value) if rounding_token is None else (lambda value: _store_wrf_real(value, rounding_token))

    def add_real(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype))

    def sub_real(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype))

    def mul_real(left: Any, right: Any) -> Any:
        return store(jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype))

    def div_real(left: Any, right: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(left, dtype=dtype) / jnp.asarray(right, dtype=dtype)
        return _divide_wrf_real(
            store(jnp.asarray(left, dtype=dtype)),
            store(jnp.asarray(right, dtype=dtype)),
            rounding_token,
        )

    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    one_plus_eps = add_real(one, epssm)
    cof_base = mul_real(half, dts)
    cof_base = mul_real(cof_base, g)
    cof_base = mul_real(cof_base, one_plus_eps)
    cof = mul_real(cof_base, cof_base)
    lid_flag = jnp.asarray(0.0 if bool(top_lid) else 1.0, dtype=dtype)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    i_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    j_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    mut_block = mut_arr[i_slice, j_slice]
    a_block = a_out[i_slice, :, j_slice]
    alpha_block = alpha_out[i_slice, :, j_slice]
    gamma_block = gamma_out[i_slice, :, j_slice]
    c2a_block = c2a_arr[i_slice, :, j_slice]
    cqw_block = cqw_arr[i_slice, :, j_slice]

    kk_kde = _k(kde, kms)
    kk_kdem1 = _k(kde - 1, kms)
    k_top_h = _k(int(kde) - 1, kms)
    a_block = a_block.at[:, _k(2, kms), :].set(0.0)
    top_mass_h = add_real(mul_real(c1h_arr[k_top_h], mut_block), c2h_arr[k_top_h])
    top_mass_f = add_real(mul_real(c1f_arr[k_top_h], mut_block), c2f_arr[k_top_h])
    top_numerator = mul_real(-two, cof)
    top_numerator = mul_real(top_numerator, mul_real(rdnw_arr[kk_kdem1], rdnw_arr[kk_kdem1]))
    top_numerator = mul_real(top_numerator, c2a_block[:, kk_kdem1, :])
    top_numerator = mul_real(top_numerator, lid_flag)
    a_top = div_real(
        top_numerator,
        mul_real(top_mass_h, top_mass_f),
    )
    a_block = a_block.at[:, kk_kde, :].set(a_top)
    gamma_block = gamma_block.at[:, _k(1, kms), :].set(0.0)

    if int(kde) > 3:
        mid = slice(_k(3, kms), _k(int(kde) - 1, kms) + 1)
        below = slice(_k(2, kms), _k(int(kde) - 2, kms) + 1)
        denom = mul_real(
            add_real(
                mul_real(c1h_arr[below][None, :, None], mut_block[:, None, :]),
                c2h_arr[below][None, :, None],
            ),
            add_real(
                mul_real(c1f_arr[below][None, :, None], mut_block[:, None, :]),
                c2f_arr[below][None, :, None],
            ),
        )
        value = mul_real(-cqw_block[:, mid, :], cof)
        value = mul_real(value, rdn_arr[mid][None, :, None])
        value = mul_real(value, rdnw_arr[below][None, :, None])
        value = mul_real(value, c2a_block[:, below, :])
        value = div_real(value, denom)
        a_block = a_block.at[:, mid, :].set(value)

    recur = slice(_k(2, kms), _k(int(kde) - 1, kms) + 1)
    recur_below = slice(_k(1, kms), _k(int(kde) - 2, kms) + 1)
    recur_above = slice(_k(3, kms), _k(int(kde), kms) + 1)
    mass_h_here = add_real(
        mul_real(c1h_arr[recur][None, :, None], mut_block[:, None, :]),
        c2h_arr[recur][None, :, None],
    )
    mass_f_here = add_real(
        mul_real(c1f_arr[recur][None, :, None], mut_block[:, None, :]),
        c2f_arr[recur][None, :, None],
    )
    mass_h_below = add_real(
        mul_real(c1h_arr[recur_below][None, :, None], mut_block[:, None, :]),
        c2h_arr[recur_below][None, :, None],
    )
    denom_here = mul_real(mass_h_here, mass_f_here)
    denom_below = mul_real(mass_h_below, mass_f_here)
    b_upper = div_real(
        mul_real(rdnw_arr[recur][None, :, None], c2a_block[:, recur, :]),
        denom_here,
    )
    b_lower = div_real(
        mul_real(rdnw_arr[recur_below][None, :, None], c2a_block[:, recur_below, :]),
        denom_below,
    )
    b_term = mul_real(cqw_block[:, recur, :], cof)
    b_term = mul_real(b_term, rdn_arr[recur][None, :, None])
    b_term = mul_real(b_term, add_real(b_upper, b_lower))
    b_stack = add_real(one, b_term)

    mass_f_above = add_real(
        mul_real(c1f_arr[recur_above][None, :, None], mut_block[:, None, :]),
        c2f_arr[recur_above][None, :, None],
    )
    c_term = mul_real(-cqw_block[:, recur, :], cof)
    c_term = mul_real(c_term, rdn_arr[recur][None, :, None])
    c_term = mul_real(c_term, rdnw_arr[recur][None, :, None])
    c_term = mul_real(c_term, c2a_block[:, recur, :])
    c_stack = div_real(c_term, mul_real(mass_h_here, mass_f_above))

    last_gamma, (alpha_scan, gamma_scan) = scan_columns(
        _acoustic_coefficient_step,
        gamma_block[:, _k(1, kms), :],
        (
            jnp.moveaxis(a_block[:, recur, :], 1, 0),
            jnp.moveaxis(b_stack, 1, 0),
            jnp.moveaxis(c_stack, 1, 0),
        ),
        (rounding_token, one),
    )
    alpha_block = alpha_block.at[:, recur, :].set(jnp.moveaxis(alpha_scan, 0, 1))
    gamma_block = gamma_block.at[:, recur, :].set(jnp.moveaxis(gamma_scan, 0, 1))

    kk = _k(kde, kms)
    km1 = _k(kde - 1, kms)
    top_b_numerator = mul_real(two, cof)
    top_b_numerator = mul_real(top_b_numerator, mul_real(rdnw_arr[km1], rdnw_arr[km1]))
    top_b_numerator = mul_real(top_b_numerator, c2a_block[:, km1, :])
    top_b_denominator = mul_real(
        add_real(mul_real(c1h_arr[km1], mut_block), c2h_arr[km1]),
        add_real(mul_real(c1f_arr[kk], mut_block), c2f_arr[kk]),
    )
    b = add_real(one, div_real(top_b_numerator, top_b_denominator))
    alpha_value = div_real(one, sub_real(b, mul_real(a_block[:, kk, :], last_gamma)))
    alpha_block = alpha_block.at[:, kk, :].set(alpha_value)
    gamma_block = gamma_block.at[:, kk, :].set(0.0)

    a_out = a_out.at[i_slice, :, j_slice].set(a_block)
    alpha_out = alpha_out.at[i_slice, :, j_slice].set(alpha_block)
    gamma_out = gamma_out.at[i_slice, :, j_slice].set(gamma_block)

    return CalcCoefWResult(a=a_out, alpha=alpha_out, gamma=gamma_out)


def _acoustic_velocity_expression(get, k, constants, *, axis, k_start, k_end, kde, kms, top_lid, nonhydro):
    """WRF pressure-gradient and divergence damping, with REAL statement order."""
    token, dts, spacing, emdiv, cf1, cf2, cf3, one, half, minus_one = constants
    dtype = dts.dtype

    def store(value):
        value = jnp.asarray(value, dtype)
        return value if token is None else _store_wrf_real(value, token)

    def binary(op, left, right):
        return store(column_binary(op, jnp.asarray(left, dtype), jnp.asarray(right, dtype)))

    def add(left, right):
        return binary("add", left, right)

    def sub(left, right):
        return binary("sub", left, right)

    def mul(left, right):
        return binary("mul", left, right)

    def div(left, right):
        return binary("div", left, store(right))

    grid_length = div(store(one), spacing)
    damping = mul(mul(mul(store(minus_one), emdiv), grid_length), sub(get("mudf"), get("mudf", neighbor=True)))
    damping = div(damping, get("map_den")) if axis == 0 else mul(damping, get("map_inv"))
    phi = add(sub(get("ph", delta=1), get("ph", delta=1, neighbor=True)),
              sub(get("ph"), get("ph", neighbor=True)))
    pressure = mul(add(get("alt"), get("alt", neighbor=True)), sub(get("p"), get("p", neighbor=True)))
    base = mul(add(get("al"), get("al", neighbor=True)), sub(get("pb"), get("pb", neighbor=True)))
    term = add(add(phi, pressure), base)
    ratio = div(get("map_num"), get("map_den"))
    mass = add(mul(get("c1h"), get("mass")), get("c2h"))
    prefactor = mul(mul(mul(ratio, store(half)), spacing), mass)
    gradient = mul(prefactor, term)

    if nonhydro:
        def pair(*, delta=0, fixed=None, reverse=False):
            left = get("p", delta=delta, fixed=fixed)
            right = get("p", delta=delta, fixed=fixed, neighbor=True)
            return add(right, left) if reverse else add(left, right)

        def interface(delta):
            level = k + delta
            bottom = mul(store(half), add(add(mul(cf1, pair(fixed=1 - kms)), mul(cf2, pair(fixed=2 - kms))), mul(cf3, pair(fixed=3 - kms))))
            interior = mul(store(half), add(mul(get("fnm", delta=delta), pair(delta=delta)),
                                         mul(get("fnp", delta=delta), pair(delta=delta - 1))))
            active = (level >= k_start + 1 - kms) & (level <= k_end - kms)
            value = jnp.where(active, interior, jnp.asarray(0., dtype))
            value = jnp.where(level == 1 - kms, bottom, value)
            if top_lid:
                top = mul(store(half), add(add(mul(cf1, pair(fixed=kde - 1 - kms, reverse=True)),
                                           mul(cf2, pair(fixed=kde - 2 - kms, reverse=True))),
                                       mul(cf3, pair(fixed=kde - 3 - kms, reverse=True))))
                value = jnp.where(level == kde - kms, top, value)
            return value

        vertical = mul(get("rdnw"), sub(interface(1), interface(0)))
        mu_vertical = mul(store(half), add(mul(get("c1h"), get("mu", neighbor=True)),
                                         mul(get("c1h"), get("mu"))))
        correction_prefactor = mul(mul(ratio, spacing), sub(get("php"), get("php", neighbor=True)))
        gradient = add(gradient, mul(correction_prefactor, sub(vertical, mu_vertical)))
    pressure_update = mul(mul(dts, get("cq")), gradient)
    return add(sub(get("velocity"), pressure_update), mul(get("c1h"), damping))


def _acoustic_velocity_cell(indices, parameters, *, shapes, axis, zones, polar_rows=(), **options):
    i, k, j = indices
    fields, constants = parameters

    def get(name, *, delta=0, fixed=None, neighbor=False):
        shape = shapes[name]
        level = k + delta if fixed is None else fixed
        x = i - int(neighbor and axis == 0)
        y = j - int(neighbor and axis == 2)
        if len(shape) == 1:
            offset = jnp.clip(level, 0, shape[0] - 1)
        elif len(shape) == 2:
            offset = jnp.clip(x, 0, shape[0] - 1) * shape[1] + jnp.clip(y, 0, shape[1] - 1)
        else:
            offset = ((jnp.clip(x, 0, shape[0] - 1) * shape[1] + jnp.clip(level, 0, shape[1] - 1))
                      * shape[2] + jnp.clip(y, 0, shape[2] - 1))
        return fields[name][offset]

    def inside(region):
        active = jnp.asarray(True)
        for position, (start, stop) in zip(indices, region, strict=True):
            active = active & (position >= start) & (position < stop)
        return active

    def rounded(operation, left, right):
        return _store_wrf_real(column_binary(operation, left, right), constants[0])

    tendency = rounded("mul", constants[1], get("tendency"))
    original = get("velocity")
    advanced = jnp.where(inside(zones[0]), rounded("add", original, tendency), original)
    read = lambda name, **offsets: advanced if name == "velocity" else get(name, **offsets)
    pressure = _acoustic_velocity_expression(read, k, constants, axis=axis, **options)
    result = jnp.where(inside(zones[1]), pressure, advanced)
    for row in polar_rows:
        region = (zones[0][0], zones[0][1], (row, row + 1))
        result = jnp.where(inside(region), jnp.asarray(0., result.dtype), result)
    return _store_wrf_real(result, constants[0])


def _acoustic_velocity_block(fields, constants, *, axis, region, **options):
    i, k, j = region

    def get(name, *, delta=0, fixed=None, neighbor=False):
        value = fields[name]
        if value.ndim > 1:
            x = slice(i.start - int(neighbor and axis == 0), i.stop - int(neighbor and axis == 0))
            y = slice(j.start - int(neighbor and axis == 2), j.stop - int(neighbor and axis == 2))
            value = value[x, ..., y]
            if value.ndim == 2:
                return value[:, None, :]
        length = value.shape[0] if value.ndim == 1 else value.shape[1]
        start, stop = (k.start + delta, k.stop + delta) if fixed is None else (fixed, fixed + 1)
        low, high = max(0, -start), max(0, stop - length)
        levels = slice(max(0, start), min(length, stop))
        value = value[levels][None, :, None] if value.ndim == 1 else value[:, levels, :]
        return jnp.pad(value, ((0, 0), (low, high), (0, 0)), mode="edge") if low or high else value

    return _acoustic_velocity_expression(get, jnp.arange(k.start, k.stop)[None, :, None],
                                        constants, axis=axis, **options)


def _advance_acoustic_component(fields, arithmetic, *, axis, tendency_region,
                                pressure_region, polar_rows=(), **options):
    """Share arithmetic; CPU slices blocks and CUDA updates each cell once."""
    constants = (*arithmetic, *(jnp.asarray(x, arithmetic[1].dtype) for x in (1., .5, -1.)))

    def block(values, constants):
        token, dts = constants[:2]

        def store(value):
            return value if token is None else _store_wrf_real(value, token)

        velocity = values["velocity"]
        tendency = store(column_binary("mul", dts, values["tendency"][tendency_region]))
        updated = store(column_binary("add", velocity[tendency_region], tendency))
        velocity = _update3(velocity, updated, *tendency_region)
        if all(region.start < region.stop for region in pressure_region):
            result = _acoustic_velocity_block(
                dict(values, velocity=velocity), constants,
                axis=axis, region=pressure_region, **options,
            )
            velocity = _update3(velocity, result, *pressure_region)
        for row in polar_rows:
            velocity = velocity.at[tendency_region[0], tendency_region[1], row].set(0.)
        return store(velocity)

    def gpu(values, constants):
        ni, nk, nj = values["velocity"].shape
        indices = (jnp.arange(ni)[:, None, None], jnp.arange(nk)[None, :, None],
                   jnp.arange(nj)[None, None, :])
        shapes = {name: value.shape for name, value in values.items()}
        zones = tuple(tuple((r.start, r.stop) for r in region)
                      for region in (tendency_region, pressure_region))
        body = partial(_acoustic_velocity_cell, shapes=shapes, axis=axis, zones=zones,
                       polar_rows=polar_rows, **options)
        parameters = ({name: value.reshape(-1) for name, value in values.items()}, constants)
        return map_elements(body, indices, parameters)

    if arithmetic[0] is None or jnp.ndim(arithmetic[0]):
        return block(fields, constants)
    return lax.platform_dependent(fields, constants, cuda=gpu, default=block)


def advance_uv(
    u: Any,
    ru_tend: Any,
    v: Any,
    rv_tend: Any,
    p: Any,
    pb: Any,
    ph: Any,
    php: Any,
    alt: Any,
    al: Any,
    mu: Any,
    muu: Any,
    cqu: Any,
    muv: Any,
    cqv: Any,
    mudf: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    rdx: Any,
    rdy: Any,
    dts: Any,
    cf1: Any,
    cf2: Any,
    cf3: Any,
    fnm: Any,
    fnp: Any,
    emdiv: Any,
    rdnw: Any,
    config_flags: Any,
    spec_zone: int,
    non_hydrostatic: bool,
    top_lid: bool,
    rounding_token: Any | None = None,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> AdvanceUVResult:
    """Run WRF ``advance_uv`` with explicit array state."""

    del kds, ime, jme, kme, c1f, c2f, c3h, c4h, c3f, c4f
    dtype = jnp.result_type(
        u,
        ru_tend,
        v,
        rv_tend,
        p,
        pb,
        ph,
        php,
        alt,
        al,
        mu,
        muu,
        cqu,
        muv,
        cqv,
        mudf,
        c1h,
        c2h,
        msfux,
        msfuy,
        msfvx,
        msfvx_inv,
        msfvy,
        rdx,
        rdy,
        dts,
        cf1,
        cf2,
        cf3,
        fnm,
        fnp,
        emdiv,
        rdnw,
        jnp.float32,
    )
    u_out = jnp.asarray(u, dtype=dtype)
    v_out = jnp.asarray(v, dtype=dtype)
    ru_tend_arr = jnp.asarray(ru_tend, dtype=dtype)
    rv_tend_arr = jnp.asarray(rv_tend, dtype=dtype)
    p_arr = jnp.asarray(p, dtype=dtype)
    pb_arr = jnp.asarray(pb, dtype=dtype)
    ph_arr = jnp.asarray(ph, dtype=dtype)
    php_arr = jnp.asarray(php, dtype=dtype)
    alt_arr = jnp.asarray(alt, dtype=dtype)
    al_arr = jnp.asarray(al, dtype=dtype)
    mu_arr = jnp.asarray(mu, dtype=dtype)
    muu_arr = jnp.asarray(muu, dtype=dtype)
    cqu_arr = jnp.asarray(cqu, dtype=dtype)
    muv_arr = jnp.asarray(muv, dtype=dtype)
    cqv_arr = jnp.asarray(cqv, dtype=dtype)
    mudf_arr = jnp.asarray(mudf, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    msfux_arr = jnp.asarray(msfux, dtype=dtype)
    msfuy_arr = jnp.asarray(msfuy, dtype=dtype)
    msfvx_arr = jnp.asarray(msfvx, dtype=dtype)
    msfvx_inv_arr = jnp.asarray(msfvx_inv, dtype=dtype)
    msfvy_arr = jnp.asarray(msfvy, dtype=dtype)
    fnm_arr = jnp.asarray(fnm, dtype=dtype)
    fnp_arr = jnp.asarray(fnp, dtype=dtype)
    rdnw_arr = jnp.asarray(rdnw, dtype=dtype)
    rdx_val = jnp.asarray(rdx, dtype=dtype)
    rdy_val = jnp.asarray(rdy, dtype=dtype)
    dts_val = jnp.asarray(dts, dtype=dtype)
    cf1_val = jnp.asarray(cf1, dtype=dtype)
    cf2_val = jnp.asarray(cf2, dtype=dtype)
    cf3_val = jnp.asarray(cf3, dtype=dtype)
    emdiv_val = jnp.asarray(emdiv, dtype=dtype)

    if _flag(config_flags, "nested") or _flag(config_flags, "specified"):
        i_start = max(int(its), int(ids) + int(spec_zone))
        i_end = min(int(ite), int(ide) - int(spec_zone) - 1)
        j_start = max(int(jts), int(jds) + int(spec_zone))
        j_end = min(int(jte), int(jde) - int(spec_zone) - 1)
        k_start = int(kts)
        k_end = min(int(kte), int(kde) - 1)
        i_endu = min(int(ite), int(ide) - int(spec_zone))
        j_endv = min(int(jte), int(jde) - int(spec_zone))
        if _flag(config_flags, "periodic_x"):
            i_start = int(its)
            i_end = min(int(ite), int(ide) - 1)
            i_endu = int(ite)
    else:
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)
        j_start = int(jts)
        j_end = min(int(jte), int(jde) - 1)
        k_start = int(kts)
        k_end = int(kte) - 1
        i_endu = int(ite)
        j_endv = int(jte)

    i_start_up = i_start
    i_end_up = i_endu
    j_start_vp = j_start
    j_end_vp = j_endv
    if (_flag(config_flags, "open_xs") or _flag(config_flags, "symmetric_xs")) and int(its) == int(ids):
        i_start_up += 1
    if (_flag(config_flags, "open_xe") or _flag(config_flags, "symmetric_xe")) and int(ite) == int(ide):
        i_end_up -= 1
    if (
        _flag(config_flags, "open_ys")
        or _flag(config_flags, "symmetric_ys")
        or _flag(config_flags, "polar")
    ) and int(jts) == int(jds):
        j_start_vp += 1
    if (
        _flag(config_flags, "open_ye")
        or _flag(config_flags, "symmetric_ye")
        or _flag(config_flags, "polar")
    ) and int(jte) == int(jde):
        j_end_vp -= 1

    i_start_u_tend = i_start
    i_end_u_tend = i_endu
    j_start_v_tend = j_start
    j_end_v_tend = j_endv
    if _flag(config_flags, "symmetric_xs") and int(its) == int(ids):
        i_start_u_tend += 1
    if _flag(config_flags, "symmetric_xe") and int(ite) == int(ide):
        i_end_u_tend -= 1
    if _flag(config_flags, "symmetric_ys") and int(jts) == int(jds):
        j_start_v_tend += 1
    if _flag(config_flags, "symmetric_ye") and int(jte) == int(jde):
        j_end_v_tend -= 1

    k_slice = slice(k_start - kms, k_end - kms + 1)
    i_u_tend_slice = slice(i_start_u_tend - ims, i_end_u_tend - ims + 1)
    j_u_slice = slice(j_start - jms, j_end - jms + 1)
    i_m_slice = slice(i_start - ims, i_end - ims + 1)
    j_v_tend_slice = slice(j_start_v_tend - jms, j_end_v_tend - jms + 1)
    i_up_slice = slice(i_start_up - ims, i_end_up - ims + 1)
    j_vp_slice = slice(j_start_vp - jms, j_end_vp - jms + 1)
    common = dict(p=p_arr, pb=pb_arr, ph=ph_arr, php=php_arr, alt=alt_arr, al=al_arr,
                  mu=mu_arr, mudf=mudf_arr, c1h=c1h_arr, c2h=c2h_arr, fnm=fnm_arr,
                  fnp=fnp_arr, rdnw=rdnw_arr)
    options = dict(k_start=k_start, k_end=k_end, kde=kde, kms=kms,
                   top_lid=bool(top_lid), nonhydro=bool(non_hydrostatic))
    u_out = _advance_acoustic_component(
        dict(common, velocity=u_out, tendency=ru_tend_arr, mass=muu_arr,
             cq=cqu_arr, map_num=msfux_arr, map_den=msfuy_arr),
        (rounding_token, dts_val, rdx_val, emdiv_val, cf1_val, cf2_val, cf3_val),
        axis=0, tendency_region=(i_u_tend_slice, k_slice, j_u_slice),
        pressure_region=(i_up_slice, k_slice, j_u_slice), **options,
    )
    polar_rows = []
    if _flag(config_flags, "polar"):
        if int(jts) == int(jds):
            polar_rows.append(jds - jms)
        if int(jte) == int(jde):
            polar_rows.append(jde - jms)
    v_out = _advance_acoustic_component(
        dict(common, velocity=v_out, tendency=rv_tend_arr, mass=muv_arr,
             cq=cqv_arr, map_num=msfvy_arr, map_den=msfvx_arr, map_inv=msfvx_inv_arr),
        (rounding_token, dts_val, rdy_val, emdiv_val, cf1_val, cf2_val, cf3_val),
        axis=2, tendency_region=(i_m_slice, k_slice, j_v_tend_slice),
        pressure_region=(i_m_slice, k_slice, j_vp_slice),
        polar_rows=tuple(polar_rows), **options,
    )
    return AdvanceUVResult(u=u_out, v=v_out)


def _mass_flux_binary(name: str, left: Any, right: Any, token: Any) -> Any:
    if token is None:
        if name == "div":
            return column_binary(name, left, right)
        return {"add": lax.add, "sub": lax.sub, "mul": lax.mul}[name](left, right)
    if name == "div":
        right = _store_wrf_real(right, token)
    return _store_wrf_real(column_binary(name, left, right), token)


def _acoustic_mass_step(previous: Any, inputs: Any, token: Any) -> Any:
    depth, divergence = inputs
    result = _mass_flux_binary(
        "add", previous, _mass_flux_binary("mul", depth, divergence, token), token
    )
    return result, None


def _acoustic_vertical_flux_step(previous: Any, inputs: Any, token: Any) -> Any:
    ww, mass_tendency, forcing, map_factor = previous
    depth, c1, divergence = inputs
    integrated = _mass_flux_binary(
        "add", _mass_flux_binary("mul", c1, mass_tendency, token), divergence, token
    )
    integrated = _mass_flux_binary(
        "add", integrated, _mass_flux_binary("mul", c1, forcing, token), token
    )
    increment = _mass_flux_binary(
        "div", _mass_flux_binary("mul", depth, integrated, token), map_factor, token
    )
    ww = _mass_flux_binary("sub", ww, increment, token)
    return (ww, mass_tendency, forcing, map_factor), ww


def advance_mu_t(
    ww: Any,
    ww_1: Any,
    u: Any,
    u_1: Any,
    v: Any,
    v_1: Any,
    mu: Any,
    mut: Any,
    muave: Any,
    muts: Any,
    muu: Any,
    muv: Any,
    mudf: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    uam: Any,
    vam: Any,
    wwam: Any,
    t: Any,
    t_1: Any,
    t_ave: Any,
    ft: Any,
    mu_tend: Any,
    rdx: Any,
    rdy: Any,
    dts: Any,
    epssm: Any,
    dnw: Any,
    fnm: Any,
    fnp: Any,
    rdnw: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    step: int,
    config_flags: Any,
    rounding_token: Any | None = None,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> AdvanceMuTResult:
    """Run WRF ``advance_mu_t`` with explicit array state."""

    del kds, ime, jme, kme, c1f, c2f, c3h, c4h, c3f, c4f, msfux, msfvy, step
    dtype = jnp.result_type(
        ww,
        ww_1,
        u,
        u_1,
        v,
        v_1,
        mu,
        mut,
        muave,
        muts,
        muu,
        muv,
        mudf,
        c1h,
        c2h,
        uam,
        vam,
        wwam,
        t,
        t_1,
        t_ave,
        ft,
        mu_tend,
        rdx,
        rdy,
        dts,
        epssm,
        dnw,
        fnm,
        fnp,
        rdnw,
        msfuy,
        msfvx,
        msfvx_inv,
        msftx,
        msfty,
        jnp.float32,
    )
    ww_out = jnp.asarray(ww, dtype=dtype)
    ww_1_out = jnp.asarray(ww_1, dtype=dtype)
    u_arr = jnp.asarray(u, dtype=dtype)
    u_1_arr = jnp.asarray(u_1, dtype=dtype)
    v_arr = jnp.asarray(v, dtype=dtype)
    v_1_arr = jnp.asarray(v_1, dtype=dtype)
    mu_out = jnp.asarray(mu, dtype=dtype)
    mut_arr = jnp.asarray(mut, dtype=dtype)
    muave_out = jnp.asarray(muave, dtype=dtype)
    muts_out = jnp.asarray(muts, dtype=dtype)
    muu_arr = jnp.asarray(muu, dtype=dtype)
    muv_arr = jnp.asarray(muv, dtype=dtype)
    mudf_out = jnp.asarray(mudf, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    uam_out = jnp.asarray(uam, dtype=dtype)
    vam_out = jnp.asarray(vam, dtype=dtype)
    wwam_out = jnp.asarray(wwam, dtype=dtype)
    t_out = jnp.asarray(t, dtype=dtype)
    t_1_arr = jnp.asarray(t_1, dtype=dtype)
    t_ave_out = jnp.asarray(t_ave, dtype=dtype)
    ft_arr = jnp.asarray(ft, dtype=dtype)
    mu_tend_arr = jnp.asarray(mu_tend, dtype=dtype)
    dnw_arr = jnp.asarray(dnw, dtype=dtype)
    fnm_arr = jnp.asarray(fnm, dtype=dtype)
    fnp_arr = jnp.asarray(fnp, dtype=dtype)
    rdnw_arr = jnp.asarray(rdnw, dtype=dtype)
    msfuy_arr = jnp.asarray(msfuy, dtype=dtype)
    msfvx_inv_arr = jnp.asarray(msfvx_inv, dtype=dtype)
    msftx_arr = jnp.asarray(msftx, dtype=dtype)
    msfty_arr = jnp.asarray(msfty, dtype=dtype)
    rdx_val = jnp.asarray(rdx, dtype=dtype)
    rdy_val = jnp.asarray(rdy, dtype=dtype)
    dts_val = jnp.asarray(dts, dtype=dtype)
    epssm_val = jnp.asarray(epssm, dtype=dtype)
    if rounding_token is None:
        add = lambda left, right: jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype)
        sub = lambda left, right: jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype)
        mul = lambda left, right: jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype)
        div = lambda left, right: jnp.asarray(left, dtype=dtype) / jnp.asarray(right, dtype=dtype)
    else:
        # The dynamic integer XOR already preserves each REAL rounding point.
        # An additional optimization barrier splits these pointwise operations
        # into separate GPU kernels, including inside every vertical iteration.
        store = lambda value: _store_wrf_real(value, rounding_token)
        add = lambda left, right: store(
            jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype)
        )
        sub = lambda left, right: store(
            jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype)
        )
        mul = lambda left, right: store(
            jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype)
        )
        div = lambda left, right: store(
            jnp.asarray(left, dtype=dtype) / store(jnp.asarray(right, dtype=dtype))
        )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = int(kte) - 1
    if not _flag(config_flags, "periodic_x"):
        if _flag(config_flags, "specified") or _flag(config_flags, "nested"):
            i_start = max(int(its), int(ids) + 1)
            i_end = min(int(ite), int(ide) - 2)
    if _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jts), int(jds) + 1)
        j_end = min(int(jte), int(jde) - 2)

    i_slice = slice(_i(i_start, ims), _i(i_end, ims) + 1)
    im1_slice = slice(_i(i_start - 1, ims), _i(i_end - 1, ims) + 1)
    ip1_slice = slice(_i(i_start + 1, ims), _i(i_end + 1, ims) + 1)
    j_slice = slice(_j(j_start, jms), _j(j_end, jms) + 1)
    jm1_slice = slice(_j(j_start - 1, jms), _j(j_end - 1, jms) + 1)
    jp1_slice = slice(_j(j_start + 1, jms), _j(j_end + 1, jms) + 1)
    k_flux_slice = slice(_k(k_start, kms), _k(k_end, kms) + 1)
    k_all_slice = slice(_k(1, kms), _k(k_end, kms) + 1)
    k_flux0 = _k(k_start, kms)

    c1h_flux = c1h_arr[k_flux_slice][None, :, None]
    c2h_flux = c2h_arr[k_flux_slice][None, :, None]
    msftx_ij = msftx_arr[i_slice, j_slice][:, None, :]
    msfty_ij = msfty_arr[i_slice, j_slice][:, None, :]

    v_jp1_mass = add(
        mul(c1h_flux, muv_arr[i_slice, jp1_slice][:, None, :]),
        c2h_flux,
    )
    v_j_mass = add(
        mul(c1h_flux, muv_arr[i_slice, j_slice][:, None, :]),
        c2h_flux,
    )
    v_jp1_coupled = mul(
        mul(v_jp1_mass, v_1_arr[i_slice, k_flux_slice, jp1_slice]),
        msfvx_inv_arr[i_slice, jp1_slice][:, None, :],
    )
    v_j_coupled = mul(
        mul(v_j_mass, v_1_arr[i_slice, k_flux_slice, j_slice]),
        msfvx_inv_arr[i_slice, j_slice][:, None, :],
    )
    v_flux = sub(
        add(v_arr[i_slice, k_flux_slice, jp1_slice], v_jp1_coupled),
        add(v_arr[i_slice, k_flux_slice, j_slice], v_j_coupled),
    )

    u_ip1_mass = add(
        mul(c1h_flux, muu_arr[ip1_slice, j_slice][:, None, :]),
        c2h_flux,
    )
    u_i_mass = add(
        mul(c1h_flux, muu_arr[i_slice, j_slice][:, None, :]),
        c2h_flux,
    )
    u_ip1_coupled = div(
        mul(u_ip1_mass, u_1_arr[ip1_slice, k_flux_slice, j_slice]),
        msfuy_arr[ip1_slice, j_slice][:, None, :],
    )
    u_i_coupled = div(
        mul(u_i_mass, u_1_arr[i_slice, k_flux_slice, j_slice]),
        msfuy_arr[i_slice, j_slice][:, None, :],
    )
    u_flux = sub(
        add(u_arr[ip1_slice, k_flux_slice, j_slice], u_ip1_coupled),
        add(u_arr[i_slice, k_flux_slice, j_slice], u_i_coupled),
    )
    horizontal_flux = add(mul(rdy_val, v_flux), mul(rdx_val, u_flux))
    map_factor = mul(msftx_ij, msfty_ij)
    dvdxi_block = mul(map_factor, horizontal_flux)
    dmdt_block, _ = scan_columns(
        _acoustic_mass_step,
        jnp.zeros_like(dvdxi_block[:, 0, :]),
        (dnw_arr[k_flux_slice], jnp.swapaxes(dvdxi_block, 0, 1)),
        rounding_token,
    )

    old_mu = mu_out[i_slice, j_slice]
    tendency = add(dmdt_block, mu_tend_arr[i_slice, j_slice])
    new_mu = add(old_mu, mul(dts_val, tendency))
    mu_out = _update2(mu_out, new_mu, i_slice, j_slice)
    mudf_out = _update2(mudf_out, tendency, i_slice, j_slice)
    muts_out = _update2(muts_out, add(mut_arr[i_slice, j_slice], new_mu), i_slice, j_slice)
    weighted_new_mu = mul(add(jnp.float32(1.0), epssm_val), new_mu)
    weighted_old_mu = mul(sub(jnp.float32(1.0), epssm_val), old_mu)
    muave_out = _update2(
        muave_out,
        mul(jnp.float32(0.5), add(weighted_new_mu, weighted_old_mu)),
        i_slice,
        j_slice,
    )

    ww_local = ww_out[i_slice, k_all_slice, j_slice]
    if k_end >= 2:
        _, ww_values = scan_columns(
            _acoustic_vertical_flux_step,
            (
                ww_out[i_slice, _k(1, kms), j_slice],
                dmdt_block,
                mu_tend_arr[i_slice, j_slice],
                msfty_arr[i_slice, j_slice],
            ),
            (
                dnw_arr[k_flux0:k_flux0 + k_end - 1],
                c1h_arr[k_flux0:k_flux0 + k_end - 1],
                jnp.swapaxes(dvdxi_block[:, :k_end - 1, :], 0, 1),
            ),
            rounding_token,
        )
        ww_local = jnp.concatenate((ww_local[:, :1, :], jnp.swapaxes(ww_values, 0, 1)), axis=1)
    ww_local = sub(ww_local, ww_1_out[i_slice, k_all_slice, j_slice])
    ww_out = _update3(ww_out, ww_local, i_slice, k_all_slice, j_slice)

    t_ave_out = _update3(t_ave_out, t_out[i_slice, k_all_slice, j_slice], i_slice, k_all_slice, j_slice)
    t_after_tendency = add(
        t_out[i_slice, k_all_slice, j_slice],
        mul(
            mul(msfty_ij, dts_val),
            ft_arr[i_slice, k_all_slice, j_slice],
        ),
    )
    i_len = i_slice.stop - i_slice.start
    j_len = j_slice.stop - j_slice.start
    wdtn_base_k = _k(1, kms)
    wdtn_block = jnp.zeros((i_len, _k(k_end + 1, kms) - wdtn_base_k + 1, j_len), dtype=dtype)
    if k_end >= 2:
        k_wdtn_slice = slice(_k(2, kms), _k(k_end, kms) + 1)
        k_wdtn_m1_slice = slice(_k(1, kms), _k(k_end - 1, kms) + 1)
        local_k_wdtn_slice = slice(k_wdtn_slice.start - wdtn_base_k, k_wdtn_slice.stop - wdtn_base_k)
        wdtn_value = mul(
            ww_local[:, 1:, :],
            add(
                mul(
                    fnm_arr[k_wdtn_slice][None, :, None],
                    t_1_arr[i_slice, k_wdtn_slice, j_slice],
                ),
                mul(
                    fnp_arr[k_wdtn_slice][None, :, None],
                    t_1_arr[i_slice, k_wdtn_m1_slice, j_slice],
                ),
            ),
        )
        wdtn_block = _update3(wdtn_block, wdtn_value, slice(0, i_len), local_k_wdtn_slice, slice(0, j_len))

    msftx_ij = msftx_arr[i_slice, j_slice][:, None, :]
    msfty_ij = msfty_arr[i_slice, j_slice][:, None, :]
    v_north = mul(
        v_arr[i_slice, k_all_slice, jp1_slice],
        add(
            t_1_arr[i_slice, k_all_slice, jp1_slice],
            t_1_arr[i_slice, k_all_slice, j_slice],
        ),
    )
    v_south = mul(
        v_arr[i_slice, k_all_slice, j_slice],
        add(
            t_1_arr[i_slice, k_all_slice, j_slice],
            t_1_arr[i_slice, k_all_slice, jm1_slice],
        ),
    )
    v_flux_theta = mul(mul(jnp.float32(0.5), rdy_val), sub(v_north, v_south))
    u_east = mul(
        u_arr[ip1_slice, k_all_slice, j_slice],
        add(
            t_1_arr[ip1_slice, k_all_slice, j_slice],
            t_1_arr[i_slice, k_all_slice, j_slice],
        ),
    )
    u_west = mul(
        u_arr[i_slice, k_all_slice, j_slice],
        add(
            t_1_arr[i_slice, k_all_slice, j_slice],
            t_1_arr[im1_slice, k_all_slice, j_slice],
        ),
    )
    u_flux_theta = mul(mul(jnp.float32(0.5), rdx_val), sub(u_east, u_west))
    horiz = mul(
        msftx_ij,
        add(v_flux_theta, u_flux_theta),
    )
    k_all_plus_slice = slice(_k(2, kms), _k(k_end + 1, kms) + 1)
    local_k_all = slice(k_all_slice.start - wdtn_base_k, k_all_slice.stop - wdtn_base_k)
    local_k_all_plus = slice(k_all_plus_slice.start - wdtn_base_k, k_all_plus_slice.stop - wdtn_base_k)
    vertical = mul(
        rdnw_arr[k_all_slice][None, :, None],
        sub(wdtn_block[:, local_k_all_plus, :], wdtn_block[:, local_k_all, :]),
    )
    decrement = mul(mul(dts_val, msfty_ij), add(horiz, vertical))
    t_out = _update3(
        t_out,
        sub(t_after_tendency, decrement),
        i_slice,
        k_all_slice,
        j_slice,
    )

    return AdvanceMuTResult(
        ww=ww_out,
        ww_1=ww_1_out,
        mu=mu_out,
        muave=muave_out,
        muts=muts_out,
        mudf=mudf_out,
        uam=uam_out,
        vam=vam_out,
        wwam=wwam_out,
        t=t_out,
        t_ave=t_ave_out,
    )


def advance_w(
    w: Any,
    rw_tend: Any,
    ww: Any,
    w_save: Any,
    u: Any,
    v: Any,
    mu1: Any,
    mut: Any,
    muave: Any,
    muts: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    t_2ave: Any,
    t_2: Any,
    t_1: Any,
    ph: Any,
    ph_1: Any,
    phb: Any,
    ph_tend: Any,
    ht: Any,
    c2a: Any,
    cqw: Any,
    alt: Any,
    alb: Any,
    a: Any,
    alpha: Any,
    gamma: Any,
    rdx: Any,
    rdy: Any,
    dts: Any,
    t0: Any,
    epssm: Any,
    dnw: Any,
    fnm: Any,
    fnp: Any,
    rdnw: Any,
    rdn: Any,
    cf1: Any,
    cf2: Any,
    cf3: Any,
    msftx: Any,
    msfty: Any,
    config_flags: Any,
    top_lid: bool,
    rounding_token: Any | None = None,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> AdvanceWResult:
    """Run WRF ``advance_w`` with explicit array state."""

    del kds, ime, jme, kme, mu1, c3h, c4h, c3f, c4f, dnw, alb
    dtype = jnp.result_type(
        w,
        rw_tend,
        ww,
        w_save,
        u,
        v,
        mut,
        muave,
        muts,
        c1h,
        c2h,
        c1f,
        c2f,
        t_2ave,
        t_2,
        t_1,
        ph,
        ph_1,
        phb,
        ph_tend,
        ht,
        c2a,
        cqw,
        alt,
        a,
        alpha,
        gamma,
        rdx,
        rdy,
        dts,
        t0,
        epssm,
        fnm,
        fnp,
        rdnw,
        rdn,
        cf1,
        cf2,
        cf3,
        msftx,
        msfty,
        jnp.float32,
    )
    w_out = jnp.asarray(w, dtype=dtype)
    rw_tend_arr = jnp.asarray(rw_tend, dtype=dtype)
    ww_arr = jnp.asarray(ww, dtype=dtype)
    w_save_arr = jnp.asarray(w_save, dtype=dtype)
    u_arr = jnp.asarray(u, dtype=dtype)
    v_arr = jnp.asarray(v, dtype=dtype)
    mut_arr = jnp.asarray(mut, dtype=dtype)
    muave_arr = jnp.asarray(muave, dtype=dtype)
    muts_arr = jnp.asarray(muts, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    c1f_arr = jnp.asarray(c1f, dtype=dtype)
    c2f_arr = jnp.asarray(c2f, dtype=dtype)
    t_2ave_out = jnp.asarray(t_2ave, dtype=dtype)
    t_2_arr = jnp.asarray(t_2, dtype=dtype)
    t_1_arr = jnp.asarray(t_1, dtype=dtype)
    ph_out = jnp.asarray(ph, dtype=dtype)
    ph_1_arr = jnp.asarray(ph_1, dtype=dtype)
    phb_arr = jnp.asarray(phb, dtype=dtype)
    ph_tend_arr = jnp.asarray(ph_tend, dtype=dtype)
    ht_arr = jnp.asarray(ht, dtype=dtype)
    c2a_arr = jnp.asarray(c2a, dtype=dtype)
    cqw_arr = jnp.asarray(cqw, dtype=dtype)
    alt_arr = jnp.asarray(alt, dtype=dtype)
    a_arr = jnp.asarray(a, dtype=dtype)
    alpha_arr = jnp.asarray(alpha, dtype=dtype)
    gamma_arr = jnp.asarray(gamma, dtype=dtype)
    fnm_arr = jnp.asarray(fnm, dtype=dtype)
    fnp_arr = jnp.asarray(fnp, dtype=dtype)
    rdnw_arr = jnp.asarray(rdnw, dtype=dtype)
    rdn_arr = jnp.asarray(rdn, dtype=dtype)
    msftx_arr = jnp.asarray(msftx, dtype=dtype)
    msfty_arr = jnp.asarray(msfty, dtype=dtype)
    rdx_val = jnp.asarray(rdx, dtype=dtype)
    rdy_val = jnp.asarray(rdy, dtype=dtype)
    dts_val = jnp.asarray(dts, dtype=dtype)
    t0_val = jnp.asarray(t0, dtype=dtype)
    epssm_val = jnp.asarray(epssm, dtype=dtype)
    cf1_val = jnp.asarray(cf1, dtype=dtype)
    cf2_val = jnp.asarray(cf2, dtype=dtype)
    cf3_val = jnp.asarray(cf3, dtype=dtype)
    g_val = jnp.asarray(G, dtype=dtype)
    store = (lambda value: value) if rounding_token is None else (lambda value: _store_wrf_real(value, rounding_token))

    def mul_real(left: Any, right: Any) -> Any:
        """Multiply as WRF REAL without LLVM contracting a later add/sub."""

        value = jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype)
        return value if rounding_token is None else _store_wrf_real(value, rounding_token)

    def add_real(left: Any, right: Any) -> Any:
        value = jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype)
        return value if rounding_token is None else _store_wrf_real(value, rounding_token)

    def sub_real(left: Any, right: Any) -> Any:
        value = jnp.asarray(left, dtype=dtype) - jnp.asarray(right, dtype=dtype)
        return value if rounding_token is None else _store_wrf_real(value, rounding_token)

    def div_real(left: Any, right: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(left, dtype=dtype) / jnp.asarray(right, dtype=dtype)
        return _divide_wrf_real(left, right, rounding_token)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_end = int(kte) - 1
    if not _flag(config_flags, "periodic_x"):
        if _flag(config_flags, "specified") or _flag(config_flags, "nested"):
            i_start = max(int(its), int(ids) + 1)
            i_end = min(int(ite), int(ide) - 2)
    if _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jts), int(jds) + 1)
        j_end = min(int(jte), int(jde) - 2)

    dampmag = mul_real(
        dts_val,
        jnp.asarray(_cfg_value(config_flags, "dampcoef", 0.0), dtype=dtype),
    )
    hdepth = jnp.asarray(_cfg_value(config_flags, "zdamp", 1.0), dtype=dtype)
    phi_adv_z = int(_cfg_value(config_flags, "phi_adv_z", 1))
    damp_opt = int(_cfg_value(config_flags, "damp_opt", 0))
    pi_val = jnp.asarray(jnp.pi, dtype=dtype)

    i_slice = slice(_i(i_start, ims), _i(i_end, ims) + 1)
    im1_slice = slice(_i(i_start - 1, ims), _i(i_end - 1, ims) + 1)
    ip1_slice = slice(_i(i_start + 1, ims), _i(i_end + 1, ims) + 1)
    j_slice = slice(_j(j_start, jms), _j(j_end, jms) + 1)
    jm1_slice = slice(_j(j_start - 1, jms), _j(j_end - 1, jms) + 1)
    jp1_slice = slice(_j(j_start + 1, jms), _j(j_end + 1, jms) + 1)
    k_mass_slice = slice(_k(1, kms), _k(k_end, kms) + 1)
    k_w_slice = slice(_k(2, kms), _k(k_end + 1, kms) + 1)
    k_top = _k(k_end + 1, kms)
    mut_ij = mut_arr[i_slice, j_slice][:, None, :]
    muave_ij = muave_arr[i_slice, j_slice][:, None, :]
    muts_ij = muts_arr[i_slice, j_slice][:, None, :]
    msfty_ij = msfty_arr[i_slice, j_slice][:, None, :]
    msft_inv_ij = 1.0 / msfty_ij

    c1h_mass = c1h_arr[k_mass_slice][None, :, None]
    c2h_mass = c2h_arr[k_mass_slice][None, :, None]
    one_plus_eps = jnp.asarray(1.0, dtype=dtype) + epssm_val
    one_minus_eps = jnp.asarray(1.0, dtype=dtype) - epssm_val
    t_2ave_value = mul_real(
        jnp.asarray(0.5, dtype=dtype),
        mul_real(one_plus_eps, t_2_arr[i_slice, k_mass_slice, j_slice])
        + mul_real(one_minus_eps, t_2ave_out[i_slice, k_mass_slice, j_slice]),
    )
    # WRF writes this first averaging statement to the REAL ``t_2ave`` array
    # before evaluating the normalization statement below.
    t_2ave_value = store(t_2ave_value)
    t_2ave_value = (
        t_2ave_value + mul_real(mul_real(c1h_mass, muave_ij), t0_val)
    ) / mul_real(
        mul_real(c1h_mass, muts_ij) + c2h_mass,
        t0_val + t_1_arr[i_slice, k_mass_slice, j_slice],
    )
    # The normalized value is likewise stored before it feeds the w equation.
    t_2ave_value = store(t_2ave_value)
    t_2ave_out = t_2ave_out.at[i_slice, k_mass_slice, j_slice].set(t_2ave_value)
    rhs_w = mul_real(
        dts_val,
        ph_tend_arr[i_slice, k_w_slice, j_slice]
        + mul_real(
            mul_real(mul_real(jnp.asarray(0.5, dtype=dtype), g_val), one_minus_eps),
            w_out[i_slice, k_w_slice, j_slice],
        ),
    )
    # WRF stores this loop result in the REAL ``rhs`` array before the
    # vertical-advection loop reads it.  Preserve that statement boundary so
    # XLA cannot contract the two loops into one expression.
    rhs_w = store(rhs_w)
    rhs_base_local = rhs_w

    if phi_adv_z == 2:
        wdwn_value = mul_real(rdnw_arr[k_mass_slice][None, :, None], (
            ph_1_arr[i_slice, k_w_slice, j_slice]
            - ph_1_arr[i_slice, k_mass_slice, j_slice]
            + phb_arr[i_slice, k_w_slice, j_slice]
            - phb_arr[i_slice, k_mass_slice, j_slice]
        ))
    else:
        wdwn_value = mul_real(
            mul_real(
                mul_real(
                    jnp.asarray(0.5, dtype=dtype),
                    ww_arr[i_slice, k_w_slice, j_slice] + ww_arr[i_slice, k_mass_slice, j_slice],
                ),
                rdnw_arr[k_mass_slice][None, :, None],
            ),
            (
                ph_1_arr[i_slice, k_w_slice, j_slice]
                - ph_1_arr[i_slice, k_mass_slice, j_slice]
                + phb_arr[i_slice, k_w_slice, j_slice]
                - phb_arr[i_slice, k_mass_slice, j_slice]
            ),
        )
    # ``wdwn`` is likewise populated by a complete Fortran loop before it is
    # consumed while updating ``rhs``.
    wdwn_w = store(wdwn_value)
    wdwn_local = wdwn_w

    if k_end >= 2:
        k_mid_slice = slice(_k(2, kms), _k(k_end, kms) + 1)
        rhs_delta = mul_real(fnm_arr[k_mid_slice][None, :, None], wdwn_w[:, 1:, :]) + mul_real(
            fnp_arr[k_mid_slice][None, :, None], wdwn_w[:, :-1, :]
        )
        if phi_adv_z == 2:
            rhs_delta = mul_real(ww_arr[i_slice, k_mid_slice, j_slice], rhs_delta)
        rhs_w = rhs_w.at[:, : k_end - 1, :].add(-mul_real(dts_val, rhs_delta))
        rhs_w = store(rhs_w)
    rhs_adv_local = rhs_w

    rhs_w = ph_out[i_slice, k_w_slice, j_slice] + mul_real(msfty_ij, rhs_w) / (
        mul_real(c1f_arr[k_w_slice][None, :, None], mut_ij) + c2f_arr[k_w_slice][None, :, None]
    )
    if bool(top_lid):
        rhs_w = rhs_w.at[:, -1, :].set(0.0)
    # The decoupled geopotential RHS is an array result used by later loops.
    rhs_w = store(rhs_w)
    rhs_dec_local = rhs_w
    rhs_local = jnp.concatenate((jnp.zeros_like(rhs_w[:, :1, :]), rhs_w), axis=1)

    k1 = _k(1, kms)
    k2 = _k(2, kms)
    k3 = _k(3, kms)
    lower_v = (
        mul_real(
            ht_arr[i_slice, jp1_slice] - ht_arr[i_slice, j_slice],
            mul_real(cf1_val, v_arr[i_slice, k1, jp1_slice])
            + mul_real(cf2_val, v_arr[i_slice, k2, jp1_slice])
            + mul_real(cf3_val, v_arr[i_slice, k3, jp1_slice]),
        )
        + mul_real(
            ht_arr[i_slice, j_slice] - ht_arr[i_slice, jm1_slice],
            mul_real(cf1_val, v_arr[i_slice, k1, j_slice])
            + mul_real(cf2_val, v_arr[i_slice, k2, j_slice])
            + mul_real(cf3_val, v_arr[i_slice, k3, j_slice]),
        )
    )
    lower_u = (
        mul_real(
            ht_arr[ip1_slice, j_slice] - ht_arr[i_slice, j_slice],
            mul_real(cf1_val, u_arr[ip1_slice, k1, j_slice])
            + mul_real(cf2_val, u_arr[ip1_slice, k2, j_slice])
            + mul_real(cf3_val, u_arr[ip1_slice, k3, j_slice]),
        )
        + mul_real(
            ht_arr[i_slice, j_slice] - ht_arr[im1_slice, j_slice],
            mul_real(cf1_val, u_arr[i_slice, k1, j_slice])
            + mul_real(cf2_val, u_arr[i_slice, k2, j_slice])
            + mul_real(cf3_val, u_arr[i_slice, k3, j_slice]),
        )
    )
    w_lower = mul_real(
        mul_real(mul_real(msfty_arr[i_slice, j_slice], jnp.asarray(0.5, dtype=dtype)), rdy_val),
        lower_v,
    ) + mul_real(
        mul_real(mul_real(msftx_arr[i_slice, j_slice], jnp.asarray(0.5, dtype=dtype)), rdx_val),
        lower_u,
    )
    w_lower = store(w_lower)
    w_out = w_out.at[i_slice, k1, j_slice].set(w_lower)

    empty_mid = jnp.empty(
        (i_end - i_start + 1, 0, j_end - j_start + 1),
        dtype=dtype,
    )
    pressure_upper_local = empty_mid
    pressure_lower_local = empty_mid
    pressure_part_local = empty_mid
    pressure_numerator_local = empty_mid
    pressure_denominator_local = empty_mid
    pressure_ratio_local = empty_mid
    pressure_rhs_delta_local = empty_mid
    pressure_ph_delta_local = empty_mid
    pressure_rhs_term_local = empty_mid
    pressure_ph_term_local = empty_mid
    pressure_bracket_local = empty_mid
    thermal_upper_local = empty_mid
    thermal_lower_local = empty_mid
    buoyancy_part_local = empty_mid
    if k_end >= 2:
        k_mid_slice = slice(_k(2, kms), _k(k_end, kms) + 1)
        k_mid_m1_slice = slice(_k(1, kms), _k(k_end - 1, kms) + 1)
        k_mid_plus_slice = slice(_k(3, kms), _k(k_end + 1, kms) + 1)
        c1h_mid = c1h_arr[k_mid_slice][None, :, None]
        c2h_mid = c2h_arr[k_mid_slice][None, :, None]
        c1h_m1 = c1h_arr[k_mid_m1_slice][None, :, None]
        c2h_m1 = c2h_arr[k_mid_m1_slice][None, :, None]
        pressure_one_plus = add_real(jnp.asarray(1.0, dtype=dtype), epssm_val)
        pressure_one_minus = sub_real(jnp.asarray(1.0, dtype=dtype), epssm_val)
        pressure_numerator = mul_real(
            c2a_arr[i_slice, k_mid_slice, j_slice],
            rdnw_arr[k_mid_slice][None, :, None],
        )
        pressure_denominator = add_real(mul_real(c1h_mid, mut_ij), c2h_mid)
        pressure_ratio = div_real(pressure_numerator, pressure_denominator)
        pressure_rhs_delta = sub_real(
            rhs_local[:, 2 : k_end + 1, :],
            rhs_local[:, 1:k_end, :],
        )
        pressure_ph_delta = sub_real(
            ph_out[i_slice, k_mid_plus_slice, j_slice],
            ph_out[i_slice, k_mid_slice, j_slice],
        )
        pressure_rhs_term = mul_real(pressure_one_plus, pressure_rhs_delta)
        pressure_ph_term = mul_real(pressure_one_minus, pressure_ph_delta)
        pressure_bracket = add_real(pressure_rhs_term, pressure_ph_term)
        pressure_upper = mul_real(
            pressure_ratio,
            pressure_bracket,
        )
        pressure_lower_numerator = mul_real(
            c2a_arr[i_slice, k_mid_m1_slice, j_slice],
            rdnw_arr[k_mid_m1_slice][None, :, None],
        )
        pressure_lower_denominator = add_real(mul_real(c1h_m1, mut_ij), c2h_m1)
        pressure_lower_ratio = div_real(
            pressure_lower_numerator,
            pressure_lower_denominator,
        )
        pressure_lower_rhs_delta = sub_real(
            rhs_local[:, 1:k_end, :],
            rhs_local[:, : k_end - 1, :],
        )
        pressure_lower_ph_delta = sub_real(
            ph_out[i_slice, k_mid_slice, j_slice],
            ph_out[i_slice, k_mid_m1_slice, j_slice],
        )
        pressure_lower_bracket = add_real(
            mul_real(pressure_one_plus, pressure_lower_rhs_delta),
            mul_real(pressure_one_minus, pressure_lower_ph_delta),
        )
        pressure_lower = mul_real(
            pressure_lower_ratio,
            pressure_lower_bracket,
        )
        pressure_part = mul_real(
            mul_real(
                mul_real(
                    mul_real(jnp.asarray(0.5, dtype=dtype), dts_val),
                    g_val,
                ),
                rdn_arr[k_mid_slice][None, :, None],
            ),
            sub_real(pressure_upper, pressure_lower),
        )
        thermal_upper = mul_real(
            mul_real(
                c2a_arr[i_slice, k_mid_slice, j_slice],
                alt_arr[i_slice, k_mid_slice, j_slice],
            ),
            t_2ave_value[:, 1:, :],
        )
        thermal_lower = mul_real(
            mul_real(
                c2a_arr[i_slice, k_mid_m1_slice, j_slice],
                alt_arr[i_slice, k_mid_m1_slice, j_slice],
            ),
            t_2ave_value[:, :-1, :],
        )
        buoyancy_part = sub_real(
            mul_real(
                rdn_arr[k_mid_slice][None, :, None],
                sub_real(thermal_upper, thermal_lower),
            ),
            mul_real(c1f_arr[k_mid_slice][None, :, None], muave_ij),
        )
        pressure_upper_local = pressure_upper
        pressure_lower_local = pressure_lower
        pressure_part_local = pressure_part
        pressure_numerator_local = pressure_numerator
        pressure_denominator_local = pressure_denominator
        pressure_ratio_local = pressure_ratio
        pressure_rhs_delta_local = pressure_rhs_delta
        pressure_ph_delta_local = pressure_ph_delta
        pressure_rhs_term_local = pressure_rhs_term
        pressure_ph_term_local = pressure_ph_term
        pressure_bracket_local = pressure_bracket
        thermal_upper_local = thermal_upper
        thermal_lower_local = thermal_lower
        buoyancy_part_local = buoyancy_part
        tendency_increment = mul_real(
            dts_val,
            rw_tend_arr[i_slice, k_mid_slice, j_slice],
        )
        pressure_increment = mul_real(
            mul_real(msft_inv_ij, cqw_arr[i_slice, k_mid_slice, j_slice]),
            pressure_part,
        )
        buoyancy_increment = mul_real(
            mul_real(mul_real(dts_val, g_val), msft_inv_ij),
            buoyancy_part,
        )
        w_mid = add_real(
            add_real(
                add_real(
                    w_out[i_slice, k_mid_slice, j_slice],
                    tendency_increment,
                ),
                pressure_increment,
            ),
            buoyancy_increment,
        )
        # WRF completes the full raw-w loop and writes REAL ``w`` before the
        # tridiagonal forward sweep starts.
        w_mid = store(w_mid)

    k_top_wrf = k_end + 1
    k_top = _k(k_top_wrf, kms)
    k_top_m1 = _k(k_top_wrf - 1, kms)
    pressure_top_factor = mul_real(
        mul_real(jnp.asarray(-0.5, dtype=dtype), dts_val),
        g_val,
    ) / (mul_real(c1h_arr[k_top_m1], mut_arr[i_slice, j_slice]) + c2h_arr[k_top_m1])
    pressure_top_factor = mul_real(
        pressure_top_factor,
        mul_real(rdnw_arr[k_top_m1], rdnw_arr[k_top_m1]),
    )
    pressure_top_factor = mul_real(pressure_top_factor, jnp.asarray(2.0, dtype=dtype))
    pressure_top_factor = mul_real(
        pressure_top_factor,
        c2a_arr[i_slice, k_top_m1, j_slice],
    )
    pressure_top = mul_real(
        pressure_top_factor,
        mul_real(one_plus_eps, rhs_local[:, -1, :] - rhs_local[:, -2, :])
        + mul_real(
            one_minus_eps,
            ph_out[i_slice, k_top, j_slice] - ph_out[i_slice, k_top_m1, j_slice],
        ),
    )
    buoyancy_top = mul_real(
        mul_real(-dts_val, g_val),
        mul_real(
            mul_real(
                mul_real(
                    mul_real(jnp.asarray(2.0, dtype=dtype), rdnw_arr[k_top_m1]),
                    c2a_arr[i_slice, k_top_m1, j_slice],
                ),
                alt_arr[i_slice, k_top_m1, j_slice],
            ),
            t_2ave_value[:, -1, :],
        )
        + mul_real(c1f_arr[k_top], muave_arr[i_slice, j_slice]),
    )
    w_top = (
        w_out[i_slice, k_top, j_slice]
        + mul_real(dts_val, rw_tend_arr[i_slice, k_top, j_slice])
        + mul_real(jnp.asarray(1.0, dtype=dtype) / msfty_arr[i_slice, j_slice], pressure_top + buoyancy_top)
    )
    if bool(top_lid):
        w_top = jnp.zeros_like(w_top)
    w_top = store(w_top)
    if k_end >= 2:
        w_local = jnp.concatenate([w_lower[:, None, :], w_mid, w_top[:, None, :]], axis=1)
    else:
        w_local = jnp.concatenate([w_lower[:, None, :], w_top[:, None, :]], axis=1)
    raw_w_local = w_local

    forward_slice = slice(_k(2, kms), _k(k_end + 1, kms) + 1)

    def forward_body(previous_w: Any, inputs: tuple[Any, Any, Any], token: Any) -> tuple[Any, Any]:
        w_k, a_k, alpha_k = inputs
        # One WRF REAL assignment:
        #   w(k) = (w(k) - a(k) * w(k-1)) * alpha(k)
        # Round only after the complete statement.  Splitting the expression
        # into stored products changes the vertical recurrence.
        next_w = (w_k - a_k * previous_w) * alpha_k
        next_w = next_w if token is None else _store_wrf_real(next_w, token)
        return next_w, next_w

    _, w_forward_scan = scan_columns(
        forward_body,
        w_local[:, 0, :],
        (
            jnp.moveaxis(w_local[:, 1:, :], 1, 0),
            jnp.moveaxis(a_arr[i_slice, forward_slice, j_slice], 1, 0),
            jnp.moveaxis(alpha_arr[i_slice, forward_slice, j_slice], 1, 0),
        ),
        rounding_token,
    )
    w_local = jnp.concatenate([w_local[:, :1, :], jnp.moveaxis(w_forward_scan, 0, 1)], axis=1)
    forward_w_local = w_local

    if k_end >= 2:
        backward_slice = slice(_k(2, kms), _k(k_end, kms) + 1)

        def backward_body(next_w: Any, inputs: tuple[Any, Any], token: Any) -> tuple[Any, Any]:
            w_k, gamma_k = inputs
            # One WRF REAL assignment; preserve contraction within its RHS.
            current_w = w_k - gamma_k * next_w
            current_w = current_w if token is None else _store_wrf_real(current_w, token)
            return current_w, current_w

        _, w_backward_scan = scan_columns(
            backward_body,
            w_local[:, -1, :],
            (
                jnp.moveaxis(w_local[:, 1:-1, :], 1, 0),
                jnp.moveaxis(gamma_arr[i_slice, backward_slice, j_slice], 1, 0),
            ),
            rounding_token,
            reverse=True,
        )
        w_local = jnp.concatenate(
            [
                w_local[:, :1, :],
                jnp.moveaxis(w_backward_scan, 0, 1),
                w_local[:, -1:, :],
            ],
            axis=1,
        )
    solved_w_local = w_local

    damping_htop_local = None
    damping_hbot_local = None
    damping_hk_local = None
    damping_angle_local = None
    damping_sine_local = None
    damping_weight_local = None
    damping_mass_local = None
    damping_term_local = None
    damping_numerator_local = None
    damping_denominator_local = None
    damped_w_local = None
    if damp_opt == 3:
        # These are scalar REAL assignments inside WRF's k/i damping loop.
        # Keep every assignment boundary explicit so XLA cannot reassociate
        # the height, sine, mass, and final quotient expressions.
        htop = div_real(
            add_real(
                ph_1_arr[i_slice, k_top, j_slice],
                phb_arr[i_slice, k_top, j_slice],
            ),
            g_val,
        )
        hbot = sub_real(htop, hdepth)
        k_damp_slice = slice(_k(2, kms), _k(k_end + 1, kms) + 1)
        hk = div_real(
            add_real(
                ph_1_arr[i_slice, k_damp_slice, j_slice],
                phb_arr[i_slice, k_damp_slice, j_slice],
            ),
            g_val,
        )
        angle = div_real(
            mul_real(
                mul_real(jnp.asarray(0.5, dtype=dtype), pi_val),
                sub_real(hk, hbot[:, None, :]),
            ),
            hdepth,
        )
        sine = store(jnp.sin(angle))
        dampwt_active = mul_real(mul_real(dampmag, sine), sine)
        dampwt = store(
            jnp.where(
                hk >= hbot[:, None, :],
                dampwt_active,
                jnp.asarray(0.0, dtype=dtype),
            )
        )
        mass = add_real(
            mul_real(
                c1f_arr[k_damp_slice][None, :, None],
                mut_arr[i_slice, j_slice][:, None, :],
            ),
            c2f_arr[k_damp_slice][None, :, None],
        )
        damping_term = mul_real(
            mul_real(dampwt, mass),
            w_save_arr[i_slice, k_damp_slice, j_slice],
        )
        damping_numerator = sub_real(w_local[:, 1:, :], damping_term)
        damping_denominator = add_real(jnp.asarray(1.0, dtype=dtype), dampwt)
        w_damped = div_real(damping_numerator, damping_denominator)
        w_damped = store(w_damped)
        damping_htop_local = htop
        damping_hbot_local = hbot
        damping_hk_local = hk
        damping_angle_local = angle
        damping_sine_local = sine
        damping_weight_local = dampwt
        damping_mass_local = mass
        damping_term_local = damping_term
        damping_numerator_local = damping_numerator
        damping_denominator_local = damping_denominator
        damped_w_local = w_damped
        w_local = jnp.concatenate([w_local[:, :1, :], w_damped], axis=1)

    w_full_slice = slice(_k(1, kms), _k(k_end + 1, kms) + 1)
    w_out = w_out.at[i_slice, w_full_slice, j_slice].set(w_local)

    ph_increment = mul_real(msfty_ij, jnp.asarray(0.5, dtype=dtype))
    ph_increment = mul_real(ph_increment, dts_val)
    ph_increment = mul_real(ph_increment, g_val)
    ph_increment = mul_real(ph_increment, one_plus_eps)
    ph_increment = mul_real(ph_increment, w_local[:, 1:, :])
    ph_increment = ph_increment / (
        mul_real(c1f_arr[k_w_slice][None, :, None], muts_ij) + c2f_arr[k_w_slice][None, :, None]
    )
    ph_value = rhs_w + ph_increment
    ph_out = ph_out.at[i_slice, k_w_slice, j_slice].set(ph_value)

    if rounding_token is not None:
        t_2ave_out = store(t_2ave_out)
        w_out = store(w_out)
        ph_out = store(ph_out)
    return AdvanceWResult(
        t_2ave=t_2ave_out,
        w=w_out,
        ph=ph_out,
        raw_w_local=raw_w_local,
        forward_w_local=forward_w_local,
        solved_w_local=solved_w_local,
        pressure_upper_local=pressure_upper_local,
        pressure_lower_local=pressure_lower_local,
        pressure_part_local=pressure_part_local,
        pressure_numerator_local=pressure_numerator_local,
        pressure_denominator_local=pressure_denominator_local,
        pressure_ratio_local=pressure_ratio_local,
        pressure_rhs_delta_local=pressure_rhs_delta_local,
        pressure_ph_delta_local=pressure_ph_delta_local,
        pressure_rhs_term_local=pressure_rhs_term_local,
        pressure_ph_term_local=pressure_ph_term_local,
        pressure_bracket_local=pressure_bracket_local,
        rhs_base_local=rhs_base_local,
        wdwn_local=wdwn_local,
        rhs_adv_local=rhs_adv_local,
        rhs_dec_local=rhs_dec_local,
        thermal_upper_local=thermal_upper_local,
        thermal_lower_local=thermal_lower_local,
        buoyancy_part_local=buoyancy_part_local,
        damping_htop_local=damping_htop_local,
        damping_hbot_local=damping_hbot_local,
        damping_hk_local=damping_hk_local,
        damping_angle_local=damping_angle_local,
        damping_sine_local=damping_sine_local,
        damping_weight_local=damping_weight_local,
        damping_mass_local=damping_mass_local,
        damping_term_local=damping_term_local,
        damping_numerator_local=damping_numerator_local,
        damping_denominator_local=damping_denominator_local,
        damped_w_local=damped_w_local,
    )


def sumflux(
    ru: Any,
    rv: Any,
    ww: Any,
    u_lin: Any,
    v_lin: Any,
    ww_lin: Any,
    muu: Any,
    muv: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    ru_m: Any,
    rv_m: Any,
    ww_m: Any,
    epssm: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    iteration: int,
    number_of_small_timesteps: int,
    rounding_token: Any | None = None,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> SumFluxResult:
    """Run WRF ``sumflux`` with explicit array state."""

    del ids, jds, kds, ime, jme, kme, c1f, c2f, c3h, c4h, c3f, c4f, epssm, msfux, msfvx, msfvy
    dtype = jnp.result_type(
        ru,
        rv,
        ww,
        u_lin,
        v_lin,
        ww_lin,
        muu,
        muv,
        c1h,
        c2h,
        ru_m,
        rv_m,
        ww_m,
        msfuy,
        msfvx_inv,
        jnp.float32,
    )
    ru_arr = jnp.asarray(ru, dtype=dtype)
    rv_arr = jnp.asarray(rv, dtype=dtype)
    ww_arr = jnp.asarray(ww, dtype=dtype)
    u_lin_arr = jnp.asarray(u_lin, dtype=dtype)
    v_lin_arr = jnp.asarray(v_lin, dtype=dtype)
    ww_lin_arr = jnp.asarray(ww_lin, dtype=dtype)
    muu_arr = jnp.asarray(muu, dtype=dtype)
    muv_arr = jnp.asarray(muv, dtype=dtype)
    c1h_arr = jnp.asarray(c1h, dtype=dtype)
    c2h_arr = jnp.asarray(c2h, dtype=dtype)
    ru_m_out = jnp.asarray(ru_m, dtype=dtype)
    rv_m_out = jnp.asarray(rv_m, dtype=dtype)
    ww_m_out = jnp.asarray(ww_m, dtype=dtype)
    msfuy_arr = jnp.asarray(msfuy, dtype=dtype)
    msfvx_inv_arr = jnp.asarray(msfvx_inv, dtype=dtype)
    if rounding_token is None:
        add = lambda left, right: left + right
        mul = lambda left, right: left * right
        div = lambda left, right: left / right
    else:
        add = lambda left, right: _store_wrf_real(
            jnp.asarray(left, dtype=dtype) + jnp.asarray(right, dtype=dtype),
            rounding_token,
        )
        mul = lambda left, right: _store_wrf_real(
            jnp.asarray(left, dtype=dtype) * jnp.asarray(right, dtype=dtype),
            rounding_token,
        )
        div = lambda left, right: _divide_wrf_real(
            jnp.asarray(left, dtype=dtype),
            jnp.asarray(right, dtype=dtype),
            rounding_token,
        )
    nsteps = jnp.asarray(number_of_small_timesteps, dtype=dtype)
    iteration_value = jnp.asarray(iteration)
    is_first_iteration = iteration_value == jnp.asarray(1, dtype=iteration_value.dtype)
    is_final_iteration = iteration_value == jnp.asarray(number_of_small_timesteps, dtype=iteration_value.dtype)

    mini = min(int(ide) - 1, int(ite))
    minj = min(int(jde) - 1, int(jte))
    mink = min(int(kde) - 1, int(kte))

    def i_slice(start: int, end: int) -> slice:
        return slice(_i(start, ims), _i(end, ims) + 1)

    def k_slice(start: int, end: int) -> slice:
        return slice(_k(start, kms), _k(end, kms) + 1)

    def j_slice(start: int, end: int) -> slice:
        return slice(_j(start, jms), _j(end, jms) + 1)

    all_i = i_slice(int(its), int(ite))
    all_k = k_slice(int(kts), int(kte))
    all_j = j_slice(int(jts), int(jte))
    common_i = i_slice(int(its), mini)
    common_k = k_slice(int(kts), mink)
    common_j = j_slice(int(jts), minj)

    # Each staggered field has one rectangular active region. Keep the
    # accumulated values local until the final store, including its outer face.
    u_region = (all_i, common_k, common_j)
    v_region = (common_i, common_k, all_j)
    w_region = (common_i, all_k, common_j)

    def accumulate(previous, flux, region):
        values = previous[region]
        return add(jnp.where(is_first_iteration, jnp.zeros_like(values), values), flux[region])

    def finish(previous, accumulated, final):
        values = jnp.where(is_final_iteration, final, accumulated)
        tile = previous[all_i, all_k, all_j]
        tile = jnp.where(is_first_iteration, jnp.zeros_like(tile), tile)
        local = tuple(slice(0, size) for size in values.shape)
        tile = _update3(tile, values, *local)
        return _update3(previous, tile, all_i, all_k, all_j)

    ru_accumulated = accumulate(ru_m_out, ru_arr, u_region)
    rv_accumulated = accumulate(rv_m_out, rv_arr, v_region)
    ww_accumulated = accumulate(ww_m_out, ww_arr, w_region)
    c1h_common = c1h_arr[common_k][None, :, None]
    c2h_common = c2h_arr[common_k][None, :, None]
    ru_final = add(
        div(ru_accumulated, nsteps),
        div(
            mul(
                add(mul(c1h_common, muu_arr[all_i, common_j][:, None, :]), c2h_common),
                u_lin_arr[u_region],
            ),
            msfuy_arr[all_i, common_j][:, None, :],
        ),
    )
    rv_final = add(
        div(rv_accumulated, nsteps),
        mul(
            mul(
                add(mul(c1h_common, muv_arr[common_i, all_j][:, None, :]), c2h_common),
                v_lin_arr[v_region],
            ),
            msfvx_inv_arr[common_i, all_j][:, None, :],
        ),
    )
    ww_final = add(div(ww_accumulated, nsteps), ww_lin_arr[w_region])
    ru_m_out = finish(ru_m_out, ru_accumulated, ru_final)
    rv_m_out = finish(rv_m_out, rv_accumulated, rv_final)
    ww_m_out = finish(ww_m_out, ww_accumulated, ww_final)

    return SumFluxResult(ru_m=ru_m_out, rv_m=rv_m_out, ww_m=ww_m_out)


__all__ = [
    "AdvanceMuTResult",
    "AdvanceUVResult",
    "AdvanceWResult",
    "CalcCoefWResult",
    "CalcPRhoResult",
    "SmallStepPrepResult",
    "SmallStepFinishResult",
    "SumFluxResult",
    "advance_mu_t",
    "advance_uv",
    "advance_w",
    "calc_coef_w",
    "calc_p_rho",
    "small_step_finish",
    "small_step_prep",
    "sumflux",
]
