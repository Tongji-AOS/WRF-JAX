"""Vectorized big-step utility kernels for runtime hot paths."""

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


def _store_wrf_real(value: Any, rounding_token: Any | None) -> Any:
    """Materialize a WRF default-REAL statement result inside XLA."""

    array = jnp.asarray(value, dtype=jnp.float32)
    if rounding_token is None:
        return array
    return real_assignment(array, rounding_token)


def _wrf_real_add(left: Any, right: Any, rounding_token: Any | None) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) + jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _wrf_real_sub(left: Any, right: Any, rounding_token: Any | None) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) - jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _wrf_real_mul(left: Any, right: Any, rounding_token: Any | None) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) * jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _wrf_real_div(left: Any, right: Any, rounding_token: Any | None) -> Any:
    return _store_wrf_real(
        jnp.asarray(left, dtype=jnp.float32) / jnp.asarray(right, dtype=jnp.float32),
        rounding_token,
    )


def _wrf_real_mul_chain(*values: Any, rounding_token: Any | None) -> Any:
    result = jnp.asarray(values[0], dtype=jnp.float32)
    for value in values[1:]:
        result = _wrf_real_mul(result, value, rounding_token)
    return result


def _idx(vals: Any, lower: int) -> Any:
    return vals - int(lower)


def _start_and_size(vals: Any, lower: int) -> tuple[Any, int]:
    arr = jnp.asarray(vals, dtype=jnp.int32)
    return arr[0] - int(lower), int(arr.size)


def _clip_start(start: Any, size: int, extent: int) -> Any:
    return jnp.clip(start, 0, max(int(extent) - int(size), 0))


def _take3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, ims: int, kms: int, jms: int) -> Any:
    arr = jnp.asarray(arr)
    i0, ni = _start_and_size(i_vals, ims)
    k0, nk = _start_and_size(k_vals, kms)
    j0, nj = _start_and_size(j_vals, jms)

    def slow(_: Any) -> Any:
        ii = jnp.clip(_idx(i_vals, ims), 0, arr.shape[0] - 1)
        kk = jnp.clip(_idx(k_vals, kms), 0, arr.shape[1] - 1)
        jj = jnp.clip(_idx(j_vals, jms), 0, arr.shape[2] - 1)
        return arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]

    if ni > arr.shape[0] or nk > arr.shape[1] or nj > arr.shape[2]:
        return slow(None)
    return lax.dynamic_slice(
        arr,
        (_clip_start(i0, ni, arr.shape[0]), _clip_start(k0, nk, arr.shape[1]), _clip_start(j0, nj, arr.shape[2])),
        (ni, nk, nj),
    )


def _take2(arr: Any, i_vals: Any, j_vals: Any, ims: int, jms: int) -> Any:
    arr = jnp.asarray(arr)
    i0, ni = _start_and_size(i_vals, ims)
    j0, nj = _start_and_size(j_vals, jms)

    def slow(_: Any) -> Any:
        ii = jnp.clip(_idx(i_vals, ims), 0, arr.shape[0] - 1)
        jj = jnp.clip(_idx(j_vals, jms), 0, arr.shape[1] - 1)
        return arr[ii[:, None], jj[None, :]]

    if ni > arr.shape[0] or nj > arr.shape[1]:
        return slow(None)
    return lax.dynamic_slice(arr, (_clip_start(i0, ni, arr.shape[0]), _clip_start(j0, nj, arr.shape[1])), (ni, nj))


def _add3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any, ims: int, kms: int, jms: int) -> Any:
    if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
        return arr
    i0, ni = _start_and_size(i_vals, ims)
    k0, nk = _start_and_size(k_vals, kms)
    j0, nj = _start_and_size(j_vals, jms)

    def slow(_: Any) -> Any:
        return arr.at[_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :]].add(values)

    if ni > arr.shape[0] or nk > arr.shape[1] or nj > arr.shape[2]:
        return slow(None)
    start = (_clip_start(i0, ni, arr.shape[0]), _clip_start(k0, nk, arr.shape[1]), _clip_start(j0, nj, arr.shape[2]))
    current = lax.dynamic_slice(arr, start, (ni, nk, nj))
    return lax.dynamic_update_slice(arr, current + values, start)


def _set3(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any, ims: int, kms: int, jms: int) -> Any:
    if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
        return arr
    i0, ni = _start_and_size(i_vals, ims)
    k0, nk = _start_and_size(k_vals, kms)
    j0, nj = _start_and_size(j_vals, jms)

    def slow(_: Any) -> Any:
        return arr.at[_idx(i_vals, ims)[:, None, None], _idx(k_vals, kms)[None, :, None], _idx(j_vals, jms)[None, None, :]].set(values)

    if ni > arr.shape[0] or nk > arr.shape[1] or nj > arr.shape[2]:
        return slow(None)
    return lax.dynamic_update_slice(
        arr,
        values,
        (_clip_start(i0, ni, arr.shape[0]), _clip_start(k0, nk, arr.shape[1]), _clip_start(j0, nj, arr.shape[2])),
    )


def rhs_ph_vectorized(
    ph_tend: Any,
    u: Any,
    v: Any,
    ww: Any,
    ph: Any,
    ph_old: Any,
    phb: Any,
    w: Any,
    mut: Any,
    muuf: Any,
    muvf: Any,
    c1f: Any,
    c2f: Any,
    fnm: Any,
    fnp: Any,
    rdnw: Any,
    cfn: float,
    cfn1: float,
    rdx: float,
    rdy: float,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    non_hydrostatic: bool,
    config_flags: Any,
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
    grav: float = 9.81,
) -> Any:
    """Vectorized WRF ``rhs_ph`` for the configured sixth-order horizontal branch."""

    del ph_old, kds, ime, jme, kme, msfuy, msfvx, msfvx_inv, msftx, kts
    advective_order = int(_read(config_flags, "h_sca_adv_order", 2))
    phi_adv_z = int(_read(config_flags, "phi_adv_z", 1))
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    open_xs = bool(_read(config_flags, "open_xs", False))
    open_xe = bool(_read(config_flags, "open_xe", False))
    open_ys = bool(_read(config_flags, "open_ys", False))
    open_ye = bool(_read(config_flags, "open_ye", False))
    if advective_order > 6 or advective_order <= 4:
        raise NotImplementedError("rhs_ph_vectorized currently covers the configured 5th/6th-order horizontal branch")
    if phi_adv_z != 1:
        raise NotImplementedError("rhs_ph_vectorized currently covers phi_adv_z=1")
    if open_xs or open_xe or open_ys or open_ye:
        raise NotImplementedError("rhs_ph_vectorized currently targets the configured non-open real case")

    out = jnp.asarray(ph_tend).copy()
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    ww_arr = jnp.asarray(ww)
    ph_arr = jnp.asarray(ph)
    phb_arr = jnp.asarray(phb)
    w_arr = jnp.asarray(w)
    mut_arr = jnp.asarray(mut)
    muuf_arr = jnp.asarray(muuf)
    muvf_arr = jnp.asarray(muvf)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rdnw_arr = jnp.asarray(rdnw)
    msfux_arr = jnp.asarray(msfux)
    msfvy_arr = jnp.asarray(msfvy)
    msfty_arr = jnp.asarray(msfty)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    i_vals = jnp.arange(int(its), itf + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), jtf + 1, dtype=jnp.int32)

    def mass(k_vals: Any, i_values: Any, j_values: Any, mu_arr: Any = mut_arr) -> Any:
        return _wrf_real_add(
            _wrf_real_mul(
                c1f_arr[_idx(k_vals, kms)][None, :, None],
                _take2(mu_arr, i_values, j_values, ims, jms)[:, None, :],
                rounding_token,
            ),
            c2f_arr[_idx(k_vals, kms)][None, :, None],
            rounding_token,
        )

    def phi_delta(i_values: Any, k_values: Any, j_values: Any) -> Any:
        value = _wrf_real_sub(
            _take3(ph_arr, i_values, k_values, j_values, ims, kms, jms),
            _take3(ph_arr, i_values, k_values - 1, j_values, ims, kms, jms),
            rounding_token,
        )
        value = _wrf_real_add(
            value,
            _take3(phb_arr, i_values, k_values, j_values, ims, kms, jms),
            rounding_token,
        )
        return _wrf_real_sub(
            value,
            _take3(phb_arr, i_values, k_values - 1, j_values, ims, kms, jms),
            rounding_token,
        )

    def strict_update(i_values: Any, k_values: Any, j_values: Any, term: Any, *, operation: str) -> None:
        nonlocal out
        current = _take3(out, i_values, k_values, j_values, ims, kms, jms)
        if operation == "add":
            values = _wrf_real_add(current, term, rounding_token)
        elif operation == "sub":
            values = _wrf_real_sub(current, term, rounding_token)
        else:
            raise ValueError(operation)
        out = _set3(out, i_values, k_values, j_values, values, ims, kms, jms)

    # Vertical geopotential advection.
    k_mid = jnp.arange(2, int(kte), dtype=jnp.int32)
    if int(k_mid.size):
        wd_next = _wrf_real_mul_chain(
            _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _take3(ww_arr, i_vals, k_mid + 1, j_vals, ims, kms, jms),
                    _take3(ww_arr, i_vals, k_mid, j_vals, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            ),
            rdnw_arr[_idx(k_mid, kms)][None, :, None],
            phi_delta(i_vals, k_mid + 1, j_vals),
            rounding_token=rounding_token,
        )
        wd_cur = _wrf_real_mul_chain(
            _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _take3(ww_arr, i_vals, k_mid, j_vals, ims, kms, jms),
                    _take3(ww_arr, i_vals, k_mid - 1, j_vals, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            ),
            rdnw_arr[_idx(k_mid - 1, kms)][None, :, None],
            phi_delta(i_vals, k_mid, j_vals),
            rounding_token=rounding_token,
        )
        vertical = _wrf_real_add(
            _wrf_real_mul(fnm_arr[_idx(k_mid, kms)][None, :, None], wd_next, rounding_token),
            _wrf_real_mul(fnp_arr[_idx(k_mid, kms)][None, :, None], wd_cur, rounding_token),
            rounding_token,
        )
        strict_update(i_vals, k_mid, j_vals, vertical, operation="sub")

    if bool(non_hydrostatic):
        k_top = jnp.asarray([int(kde)], dtype=jnp.int32)
        out = _set3(out, i_vals, k_top, j_vals, jnp.zeros((int(i_vals.size), 1, int(j_vals.size)), dtype=out.dtype), ims, kms, jms)
        k_mass = jnp.arange(2, int(kte) + 1, dtype=jnp.int32)
        values = _wrf_real_div(
            _wrf_real_mul_chain(
                mass(k_mass, i_vals, j_vals),
                grav,
                _take3(w_arr, i_vals, k_mass, j_vals, ims, kms, jms),
                rounding_token=rounding_token,
            ),
            _take2(msfty_arr, i_vals, j_vals, ims, jms)[:, None, :],
            rounding_token,
        )
        strict_update(i_vals, k_mass, j_vals, values, operation="add")

    def horizontal_phi_delta(
        high_i: Any,
        k_values: Any,
        high_j: Any,
        low_i: Any,
        low_j: Any,
    ) -> Any:
        """Evaluate WRF's ``phb_hi-phb_lo+ph_hi-ph_lo`` left to right."""

        value = _wrf_real_sub(
            _take3(phb_arr, high_i, k_values, high_j, ims, kms, jms),
            _take3(phb_arr, low_i, k_values, low_j, ims, kms, jms),
            rounding_token,
        )
        value = _wrf_real_add(
            value,
            _take3(ph_arr, high_i, k_values, high_j, ims, kms, jms),
            rounding_token,
        )
        return _wrf_real_sub(
            value,
            _take3(ph_arr, low_i, k_values, low_j, ims, kms, jms),
            rounding_token,
        )

    def horizontal_face_factor(
        velocity_arr: Any,
        mu_arr: Any,
        map_factor_arr: Any,
        i_values: Any,
        k_values: Any,
        j_values: Any,
        *,
        top: bool = False,
    ) -> Any:
        if top:
            velocity = _wrf_real_add(
                _wrf_real_mul(
                    cfn,
                    _take3(velocity_arr, i_values, k_values - 1, j_values, ims, kms, jms),
                    rounding_token,
                ),
                _wrf_real_mul(
                    cfn1,
                    _take3(velocity_arr, i_values, k_values - 2, j_values, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
        else:
            velocity = _wrf_real_add(
                _take3(velocity_arr, i_values, k_values, j_values, ims, kms, jms),
                _take3(velocity_arr, i_values, k_values - 1, j_values, ims, kms, jms),
                rounding_token,
            )
        return _wrf_real_mul_chain(
            mass(k_values, i_values, j_values, mu_arr),
            velocity,
            _take2(map_factor_arr, i_values, j_values, ims, jms)[:, None, :],
            rounding_token=rounding_token,
        )

    def axis_difference(
        field_arr: Any,
        i_values: Any,
        k_values: Any,
        j_values: Any,
        *,
        axis: str,
        offset: int,
    ) -> Any:
        if axis == "y":
            high_i, low_i = i_values, i_values
            high_j, low_j = j_values + int(offset), j_values - int(offset)
        elif axis == "x":
            high_i, low_i = i_values + int(offset), i_values - int(offset)
            high_j, low_j = j_values, j_values
        else:
            raise ValueError(axis)
        return _wrf_real_sub(
            _take3(field_arr, high_i, k_values, high_j, ims, kms, jms),
            _take3(field_arr, low_i, k_values, low_j, ims, kms, jms),
            rounding_token,
        )

    def high_order_gradient(
        i_values: Any,
        k_values: Any,
        j_values: Any,
        *,
        axis: str,
        coeff: tuple[float, float, float],
    ) -> Any:
        """Evaluate the WRF stencil in its textual left-to-right REAL order."""

        c_a, c_b, c_c = coeff
        value = _wrf_real_mul(
            c_a,
            axis_difference(ph_arr, i_values, k_values, j_values, axis=axis, offset=1),
            rounding_token,
        )
        value = _wrf_real_sub(
            value,
            _wrf_real_mul(
                -c_b,
                axis_difference(ph_arr, i_values, k_values, j_values, axis=axis, offset=2),
                rounding_token,
            ),
            rounding_token,
        )
        if c_c != 0.0:
            third = axis_difference(ph_arr, i_values, k_values, j_values, axis=axis, offset=3)
            if c_c != 1.0:
                third = _wrf_real_mul(c_c, third, rounding_token)
            value = _wrf_real_add(value, third, rounding_token)
        value = _wrf_real_add(
            value,
            _wrf_real_mul(
                c_a,
                axis_difference(phb_arr, i_values, k_values, j_values, axis=axis, offset=1),
                rounding_token,
            ),
            rounding_token,
        )
        value = _wrf_real_sub(
            value,
            _wrf_real_mul(
                -c_b,
                axis_difference(phb_arr, i_values, k_values, j_values, axis=axis, offset=2),
                rounding_token,
            ),
            rounding_token,
        )
        if c_c != 0.0:
            third = axis_difference(phb_arr, i_values, k_values, j_values, axis=axis, offset=3)
            if c_c != 1.0:
                third = _wrf_real_mul(c_c, third, rounding_token)
            value = _wrf_real_add(value, third, rounding_token)
        return value

    def horizontal_prefactor(
        i_values: Any,
        j_values: Any,
        leading_coefficient: float,
        inverse_spacing: float,
    ) -> Any:
        return _wrf_real_div(
            _wrf_real_mul(leading_coefficient, inverse_spacing, rounding_token),
            _take2(msfty_arr, i_values, j_values, ims, jms)[:, None, :],
            rounding_token,
        )

    def apply_second_order(
        i_values: Any,
        k_values: Any,
        j_values: Any,
        first: Any,
        second: Any,
        leading_coefficient: float,
        inverse_spacing: float,
    ) -> None:
        flux = _wrf_real_add(first, second, rounding_token)
        term = _wrf_real_mul(
            horizontal_prefactor(i_values, j_values, leading_coefficient, inverse_spacing),
            flux,
            rounding_token,
        )
        strict_update(i_values, k_values, j_values, term, operation="sub")

    def apply_high_order(
        i_values: Any,
        k_values: Any,
        j_values: Any,
        velocity_factor: Any,
        gradient: Any,
        leading_coefficient: float,
        inverse_spacing: float,
        denominator: float,
    ) -> None:
        inner = _wrf_real_mul_chain(
            velocity_factor,
            _wrf_real_div(1.0, denominator, rounding_token),
            gradient,
            rounding_token=rounding_token,
        )
        term = _wrf_real_mul(
            horizontal_prefactor(i_values, j_values, leading_coefficient, inverse_spacing),
            inner,
            rounding_token,
        )
        strict_update(i_values, k_values, j_values, term, operation="sub")

    def y_second(j_values: Any, i_start: int, i_end: int) -> None:
        i_local = jnp.arange(int(i_start), int(i_end) + 1, dtype=jnp.int32)
        if int(i_local.size) == 0 or int(j_values.size) == 0:
            return
        k_local = jnp.arange(2, int(kte), dtype=jnp.int32)
        if int(k_local.size):
            upper = _wrf_real_mul(
                horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_local, j_values + 1),
                horizontal_phi_delta(i_local, k_local, j_values + 1, i_local, j_values),
                rounding_token,
            )
            lower = _wrf_real_mul(
                horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_local, j_values),
                horizontal_phi_delta(i_local, k_local, j_values, i_local, j_values - 1),
                rounding_token,
            )
            apply_second_order(i_local, k_local, j_values, upper, lower, 0.25, rdy)
        k_top_local = jnp.asarray([int(kte)], dtype=jnp.int32)
        upper = _wrf_real_mul(
            horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_top_local, j_values + 1, top=True),
            horizontal_phi_delta(i_local, k_top_local, j_values + 1, i_local, j_values),
            rounding_token,
        )
        lower = _wrf_real_mul(
            horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_top_local, j_values, top=True),
            horizontal_phi_delta(i_local, k_top_local, j_values, i_local, j_values - 1),
            rounding_token,
        )
        apply_second_order(i_local, k_top_local, j_values, upper, lower, 0.5, rdy)

    def y_high(j_values: Any, i_start: int, i_end: int, coeff: tuple[float, float, float], denominator: float) -> None:
        i_local = jnp.arange(int(i_start), int(i_end) + 1, dtype=jnp.int32)
        if int(i_local.size) == 0 or int(j_values.size) == 0:
            return
        k_local = jnp.arange(2, int(kte), dtype=jnp.int32)
        if int(k_local.size):
            velocity_factor = _wrf_real_add(
                horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_local, j_values + 1),
                horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_local, j_values),
                rounding_token,
            )
            gradient = high_order_gradient(i_local, k_local, j_values, axis="y", coeff=coeff)
            apply_high_order(i_local, k_local, j_values, velocity_factor, gradient, 0.25, rdy, denominator)
        k_top_local = jnp.asarray([int(kte)], dtype=jnp.int32)
        velocity_factor = _wrf_real_add(
            horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_top_local, j_values + 1, top=True),
            horizontal_face_factor(v_arr, muvf_arr, msfvy_arr, i_local, k_top_local, j_values, top=True),
            rounding_token,
        )
        gradient = high_order_gradient(i_local, k_top_local, j_values, axis="y", coeff=coeff)
        apply_high_order(i_local, k_top_local, j_values, velocity_factor, gradient, 0.5, rdy, denominator)

    def x_second(i_values: Any, j_start: int, j_end: int) -> None:
        j_local = jnp.arange(int(j_start), int(j_end) + 1, dtype=jnp.int32)
        if int(i_values.size) == 0 or int(j_local.size) == 0:
            return
        k_local = jnp.arange(2, int(kte), dtype=jnp.int32)
        if int(k_local.size):
            right = _wrf_real_mul(
                horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values + 1, k_local, j_local),
                horizontal_phi_delta(i_values + 1, k_local, j_local, i_values, j_local),
                rounding_token,
            )
            left = _wrf_real_mul(
                horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values, k_local, j_local),
                horizontal_phi_delta(i_values, k_local, j_local, i_values - 1, j_local),
                rounding_token,
            )
            apply_second_order(i_values, k_local, j_local, right, left, 0.25, rdx)
        k_top_local = jnp.asarray([int(kte)], dtype=jnp.int32)
        right = _wrf_real_mul(
            horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values + 1, k_top_local, j_local, top=True),
            horizontal_phi_delta(i_values + 1, k_top_local, j_local, i_values, j_local),
            rounding_token,
        )
        left = _wrf_real_mul(
            horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values, k_top_local, j_local, top=True),
            horizontal_phi_delta(i_values, k_top_local, j_local, i_values - 1, j_local),
            rounding_token,
        )
        apply_second_order(i_values, k_top_local, j_local, right, left, 0.5, rdx)

    def x_high(i_values: Any, j_start: int, j_end: int, coeff: tuple[float, float, float], denominator: float) -> None:
        j_local = jnp.arange(int(j_start), int(j_end) + 1, dtype=jnp.int32)
        if int(i_values.size) == 0 or int(j_local.size) == 0:
            return
        k_local = jnp.arange(2, int(kte), dtype=jnp.int32)
        if int(k_local.size):
            velocity_factor = _wrf_real_add(
                horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values + 1, k_local, j_local),
                horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values, k_local, j_local),
                rounding_token,
            )
            gradient = high_order_gradient(i_values, k_local, j_local, axis="x", coeff=coeff)
            apply_high_order(i_values, k_local, j_local, velocity_factor, gradient, 0.25, rdx, denominator)
        k_top_local = jnp.asarray([int(kte)], dtype=jnp.int32)
        velocity_factor = _wrf_real_add(
            horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values + 1, k_top_local, j_local, top=True),
            horizontal_face_factor(u_arr, muuf_arr, msfux_arr, i_values, k_top_local, j_local, top=True),
            rounding_token,
        )
        gradient = high_order_gradient(i_values, k_top_local, j_local, axis="x", coeff=coeff)
        apply_high_order(i_values, k_top_local, j_local, velocity_factor, gradient, 0.5, rdx, denominator)

    y_i_start = int(its)
    y_itf = min(int(ite), int(ide) - 1)
    y_j_start = int(jts)
    y_jtf = min(int(jte), int(jde) - 1)
    if open_ys or specified:
        y_j_start = max(int(jts), int(jds) + 3)
    if open_ye or specified:
        y_jtf = min(y_jtf, int(jde) - 4)
    y_high(jnp.arange(y_j_start, y_jtf + 1, dtype=jnp.int32), y_i_start, y_itf, (45.0, -9.0, 1.0), 60.0)
    if (open_ys or specified) and int(jts) <= int(jds) + 2 <= int(jte):
        y_high(jnp.asarray([int(jds) + 2], dtype=jnp.int32), y_i_start, y_itf, (8.0, -1.0, 0.0), 12.0)
    if (open_ye or specified) and int(jts) <= int(jde) - 3 <= int(jte):
        y_high(jnp.asarray([int(jde) - 3], dtype=jnp.int32), y_i_start, y_itf, (8.0, -1.0, 0.0), 12.0)
    if (open_ys or specified) and int(jts) <= int(jds) + 1 <= int(jte):
        y_second(jnp.asarray([int(jds) + 1], dtype=jnp.int32), y_i_start, y_itf)
    if (open_ye or specified) and int(jts) <= int(jde) - 2 <= int(jte):
        y_second(jnp.asarray([int(jde) - 2], dtype=jnp.int32), y_i_start, y_itf)

    x_i_start = int(its)
    x_itf = min(int(ite), int(ide) - 1)
    x_j_start = int(jts)
    x_jtf = min(int(jte), int(jde) - 1)
    if open_xs or specified:
        x_i_start = max(int(its), int(ids) + 3)
    if open_xe or specified:
        x_itf = min(x_itf, int(ide) - 4)
    x_high(jnp.arange(x_i_start, x_itf + 1, dtype=jnp.int32), x_j_start, x_jtf, (45.0, -9.0, 1.0), 60.0)
    if open_xs and int(its) <= int(ids) + 2 <= int(ite):
        x_high(jnp.asarray([int(ids) + 2], dtype=jnp.int32), x_j_start, x_jtf, (8.0, -1.0, 0.0), 12.0)
    if open_xe and int(its) <= int(ide) - 3 <= int(ite):
        x_high(jnp.asarray([int(ide) - 3], dtype=jnp.int32), x_j_start, x_jtf, (8.0, -1.0, 0.0), 12.0)
    if (open_xs or specified) and int(its) <= int(ids) + 1 <= int(ite):
        x_second(jnp.asarray([int(ids) + 1], dtype=jnp.int32), x_j_start, x_jtf)
    if (open_xe or specified) and int(its) <= int(ide) - 2 <= int(ite):
        x_second(jnp.asarray([int(ide) - 2], dtype=jnp.int32), x_j_start, x_jtf)

    return out


def horizontal_pressure_gradient_vectorized(
    ru_tend: Any,
    rv_tend: Any,
    ph: Any,
    alt: Any,
    p: Any,
    pb: Any,
    al: Any,
    php: Any,
    cqu: Any,
    cqv: Any,
    muu: Any,
    muv: Any,
    mu: Any,
    c1h: Any,
    c2h: Any,
    fnm: Any,
    fnp: Any,
    rdnw: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    cfn: float,
    cfn1: float,
    rdx: float,
    rdy: float,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    config_flags: Any,
    non_hydrostatic: bool,
    top_lid: bool,
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
    return_diagnostics: bool = False,
) -> tuple[Any, ...]:
    """Vectorized WRF ``horizontal_pressure_gradient`` for the real-case branch."""

    del kds, ime, jme, kme, msftx, msfty, kts
    if bool(_read(config_flags, "open_xs", False)) or bool(_read(config_flags, "open_xe", False)):
        raise NotImplementedError("horizontal_pressure_gradient_vectorized does not cover open x boundaries")
    if bool(_read(config_flags, "open_ys", False)) or bool(_read(config_flags, "open_ye", False)):
        raise NotImplementedError("horizontal_pressure_gradient_vectorized does not cover open y boundaries")
    if bool(_read(config_flags, "polar", False)) or bool(_read(config_flags, "periodic_x", False)):
        raise NotImplementedError("horizontal_pressure_gradient_vectorized targets non-polar, non-periodic-x grids")

    ru_out = jnp.asarray(ru_tend).copy()
    rv_out = jnp.asarray(rv_tend).copy()
    ph_arr = jnp.asarray(ph)
    alt_arr = jnp.asarray(alt)
    p_arr = jnp.asarray(p)
    pb_arr = jnp.asarray(pb)
    al_arr = jnp.asarray(al)
    php_arr = jnp.asarray(php)
    cqu_arr = jnp.asarray(cqu)
    cqv_arr = jnp.asarray(cqv)
    muu_arr = jnp.asarray(muu)
    muv_arr = jnp.asarray(muv)
    mu_arr = jnp.asarray(mu)
    c1h_arr = jnp.asarray(c1h)
    c2h_arr = jnp.asarray(c2h)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rdnw_arr = jnp.asarray(rdnw)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    if return_diagnostics:
        dpx_primary_out = jnp.zeros_like(ru_out)
        dpx_total_out = jnp.zeros_like(ru_out)
        dpy_primary_out = jnp.zeros_like(rv_out)
        dpy_total_out = jnp.zeros_like(rv_out)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    ktf = min(int(kte), int(kde) - 1)

    def set_dpn(dpn: Any, k_start: int, values: Any) -> Any:
        if values.shape[1] == 0:
            return dpn
        return lax.dynamic_update_slice(dpn, values, (0, k_start - int(kms), 0))

    # V tendency: y-direction pressure gradient.
    i_v = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j_start = int(jts)
    j_end = int(jte)
    if specified and int(jts) == int(jds):
        j_start = int(jts) + 1
    if specified and int(jte) == int(jde):
        j_end -= 1
    j_v = jnp.arange(j_start, j_end + 1, dtype=jnp.int32)
    k_cell = jnp.arange(1, ktf + 1, dtype=jnp.int32)
    if int(i_v.size) and int(j_v.size) and int(k_cell.size):
        dpn = jnp.zeros((int(i_v.size), int(kde) - int(kms) + 1, int(j_v.size)), dtype=p_arr.dtype)
        if bool(non_hydrostatic):
            k1 = jnp.asarray([1], dtype=jnp.int32)
            first = _wrf_real_mul(
                cf1,
                _wrf_real_add(
                    _take3(p_arr, i_v, k1, j_v - 1, ims, kms, jms),
                    _take3(p_arr, i_v, k1, j_v, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            second = _wrf_real_mul(
                cf2,
                _wrf_real_add(
                    _take3(p_arr, i_v, k1 + 1, j_v - 1, ims, kms, jms),
                    _take3(p_arr, i_v, k1 + 1, j_v, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            third = _wrf_real_mul(
                cf3,
                _wrf_real_add(
                    _take3(p_arr, i_v, k1 + 2, j_v - 1, ims, kms, jms),
                    _take3(p_arr, i_v, k1 + 2, j_v, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            values = _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _wrf_real_add(first, second, rounding_token),
                    third,
                    rounding_token,
                ),
                rounding_token,
            )
            dpn = set_dpn(dpn, 1, values)
            k_top = jnp.asarray([int(kde)], dtype=jnp.int32)
            if bool(top_lid):
                values = _wrf_real_mul(
                    0.5,
                    _wrf_real_add(
                        _wrf_real_mul(
                            cfn,
                            _wrf_real_add(
                                _take3(p_arr, i_v, k_top - 1, j_v - 1, ims, kms, jms),
                                _take3(p_arr, i_v, k_top - 1, j_v, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        _wrf_real_mul(
                            cfn1,
                            _wrf_real_add(
                                _take3(p_arr, i_v, k_top - 2, j_v - 1, ims, kms, jms),
                                _take3(p_arr, i_v, k_top - 2, j_v, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    rounding_token,
                )
                dpn = set_dpn(dpn, int(kde), values)
            else:
                dpn = set_dpn(dpn, int(kde), jnp.zeros((int(i_v.size), 1, int(j_v.size)), dtype=p_arr.dtype))
            k_mid = jnp.arange(2, ktf + 1, dtype=jnp.int32)
            if int(k_mid.size):
                values = _wrf_real_mul(
                    0.5,
                    _wrf_real_add(
                        _wrf_real_mul(
                            fnm_arr[_idx(k_mid, kms)][None, :, None],
                            _wrf_real_add(
                                _take3(p_arr, i_v, k_mid, j_v - 1, ims, kms, jms),
                                _take3(p_arr, i_v, k_mid, j_v, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        _wrf_real_mul(
                            fnp_arr[_idx(k_mid, kms)][None, :, None],
                            _wrf_real_add(
                                _take3(p_arr, i_v, k_mid - 1, j_v - 1, ims, kms, jms),
                                _take3(p_arr, i_v, k_mid - 1, j_v, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    rounding_token,
                )
                dpn = set_dpn(dpn, 2, values)

        ratio = _wrf_real_div(
            _take2(msfvy_arr, i_v, j_v, ims, jms),
            _take2(msfvx_arr, i_v, j_v, ims, jms),
            rounding_token,
        )
        mass = _wrf_real_add(
            _wrf_real_mul(
                c1h_arr[_idx(k_cell, kms)][None, :, None],
                _take2(muv_arr, i_v, j_v, ims, jms)[:, None, :],
                rounding_token,
            ),
            c2h_arr[_idx(k_cell, kms)][None, :, None],
            rounding_token,
        )
        prefactor = _wrf_real_mul(
            _wrf_real_mul(
                _wrf_real_mul(ratio[:, None, :], 0.5, rounding_token),
                rdy,
                rounding_token,
            ),
            mass,
            rounding_token,
        )
        phi = _wrf_real_sub(
            _wrf_real_add(
                _wrf_real_sub(
                    _take3(ph_arr, i_v, k_cell + 1, j_v, ims, kms, jms),
                    _take3(ph_arr, i_v, k_cell + 1, j_v - 1, ims, kms, jms),
                    rounding_token,
                ),
                _take3(ph_arr, i_v, k_cell, j_v, ims, kms, jms),
                rounding_token,
            ),
            _take3(ph_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
            rounding_token,
        )
        p_term = _wrf_real_mul(
            _wrf_real_add(
                _take3(alt_arr, i_v, k_cell, j_v, ims, kms, jms),
                _take3(alt_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take3(p_arr, i_v, k_cell, j_v, ims, kms, jms),
                _take3(p_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
                rounding_token,
            ),
            rounding_token,
        )
        pb_term = _wrf_real_mul(
            _wrf_real_add(
                _take3(al_arr, i_v, k_cell, j_v, ims, kms, jms),
                _take3(al_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take3(pb_arr, i_v, k_cell, j_v, ims, kms, jms),
                _take3(pb_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
                rounding_token,
            ),
            rounding_token,
        )
        dpy = _wrf_real_mul(
            prefactor,
            _wrf_real_add(
                _wrf_real_add(phi, p_term, rounding_token),
                pb_term,
                rounding_token,
            ),
            rounding_token,
        )
        if return_diagnostics:
            dpy_primary_out = _set3(dpy_primary_out, i_v, k_cell, j_v, dpy, ims, kms, jms)
        if bool(non_hydrostatic):
            dpn_diff = _wrf_real_sub(
                dpn[:, 2 - int(kms):ktf + 2 - int(kms), :],
                dpn[:, 1 - int(kms):ktf + 1 - int(kms), :],
                rounding_token,
            )
            php_delta = _wrf_real_sub(
                _take3(php_arr, i_v, k_cell, j_v, ims, kms, jms),
                _take3(php_arr, i_v, k_cell, j_v - 1, ims, kms, jms),
                rounding_token,
            )
            dpn_term = _wrf_real_mul(
                rdnw_arr[_idx(k_cell, kms)][None, :, None],
                dpn_diff,
                rounding_token,
            )
            mu_term = _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _wrf_real_mul(
                        c1h_arr[_idx(k_cell, kms)][None, :, None],
                        _take2(mu_arr, i_v, j_v - 1, ims, jms)[:, None, :],
                        rounding_token,
                    ),
                    _wrf_real_mul(
                        c1h_arr[_idx(k_cell, kms)][None, :, None],
                        _take2(mu_arr, i_v, j_v, ims, jms)[:, None, :],
                        rounding_token,
                    ),
                    rounding_token,
                ),
                rounding_token,
            )
            term4 = _wrf_real_mul(
                _wrf_real_mul(
                    _wrf_real_mul(ratio[:, None, :], rdy, rounding_token),
                    php_delta,
                    rounding_token,
                ),
                _wrf_real_sub(dpn_term, mu_term, rounding_token),
                rounding_token,
            )
            dpy = _wrf_real_add(dpy, term4, rounding_token)
            if return_diagnostics:
                dpy_total_out = _set3(dpy_total_out, i_v, k_cell, j_v, dpy, ims, kms, jms)
            scaled = _wrf_real_mul(
                _take3(cqv_arr, i_v, k_cell, j_v, ims, kms, jms),
                dpy,
                rounding_token,
            )
            update = _wrf_real_sub(
                _take3(rv_out, i_v, k_cell, j_v, ims, kms, jms),
                scaled,
                rounding_token,
            )
        else:
            update = _wrf_real_sub(
                _take3(rv_out, i_v, k_cell, j_v, ims, kms, jms),
                dpy,
                rounding_token,
            )
        rv_out = _set3(rv_out, i_v, k_cell, j_v, update, ims, kms, jms)

    # U tendency: x-direction pressure gradient.
    i_start = int(its)
    i_end = int(ite)
    if specified and int(its) == int(ids):
        i_start = int(its) + 1
    if specified and int(ite) == int(ide):
        i_end -= 1
    i_u = jnp.arange(i_start, i_end + 1, dtype=jnp.int32)
    j_u = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    if int(i_u.size) and int(j_u.size) and int(k_cell.size):
        dpn = jnp.zeros((int(i_u.size), int(kde) - int(kms) + 1, int(j_u.size)), dtype=p_arr.dtype)
        if bool(non_hydrostatic):
            k1 = jnp.asarray([1], dtype=jnp.int32)
            first = _wrf_real_mul(
                cf1,
                _wrf_real_add(
                    _take3(p_arr, i_u - 1, k1, j_u, ims, kms, jms),
                    _take3(p_arr, i_u, k1, j_u, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            second = _wrf_real_mul(
                cf2,
                _wrf_real_add(
                    _take3(p_arr, i_u - 1, k1 + 1, j_u, ims, kms, jms),
                    _take3(p_arr, i_u, k1 + 1, j_u, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            third = _wrf_real_mul(
                cf3,
                _wrf_real_add(
                    _take3(p_arr, i_u - 1, k1 + 2, j_u, ims, kms, jms),
                    _take3(p_arr, i_u, k1 + 2, j_u, ims, kms, jms),
                    rounding_token,
                ),
                rounding_token,
            )
            values = _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _wrf_real_add(first, second, rounding_token),
                    third,
                    rounding_token,
                ),
                rounding_token,
            )
            dpn = set_dpn(dpn, 1, values)
            k_top = jnp.asarray([int(kde)], dtype=jnp.int32)
            if bool(top_lid):
                values = _wrf_real_mul(
                    0.5,
                    _wrf_real_add(
                        _wrf_real_mul(
                            cfn,
                            _wrf_real_add(
                                _take3(p_arr, i_u - 1, k_top - 1, j_u, ims, kms, jms),
                                _take3(p_arr, i_u, k_top - 1, j_u, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        _wrf_real_mul(
                            cfn1,
                            _wrf_real_add(
                                _take3(p_arr, i_u - 1, k_top - 2, j_u, ims, kms, jms),
                                _take3(p_arr, i_u, k_top - 2, j_u, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    rounding_token,
                )
                dpn = set_dpn(dpn, int(kde), values)
            else:
                dpn = set_dpn(dpn, int(kde), jnp.zeros((int(i_u.size), 1, int(j_u.size)), dtype=p_arr.dtype))
            k_mid = jnp.arange(2, ktf + 1, dtype=jnp.int32)
            if int(k_mid.size):
                values = _wrf_real_mul(
                    0.5,
                    _wrf_real_add(
                        _wrf_real_mul(
                            fnm_arr[_idx(k_mid, kms)][None, :, None],
                            _wrf_real_add(
                                _take3(p_arr, i_u - 1, k_mid, j_u, ims, kms, jms),
                                _take3(p_arr, i_u, k_mid, j_u, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        _wrf_real_mul(
                            fnp_arr[_idx(k_mid, kms)][None, :, None],
                            _wrf_real_add(
                                _take3(p_arr, i_u - 1, k_mid - 1, j_u, ims, kms, jms),
                                _take3(p_arr, i_u, k_mid - 1, j_u, ims, kms, jms),
                                rounding_token,
                            ),
                            rounding_token,
                        ),
                        rounding_token,
                    ),
                    rounding_token,
                )
                dpn = set_dpn(dpn, 2, values)

        ratio = _wrf_real_div(
            _take2(msfux_arr, i_u, j_u, ims, jms),
            _take2(msfuy_arr, i_u, j_u, ims, jms),
            rounding_token,
        )
        mass = _wrf_real_add(
            _wrf_real_mul(
                c1h_arr[_idx(k_cell, kms)][None, :, None],
                _take2(muu_arr, i_u, j_u, ims, jms)[:, None, :],
                rounding_token,
            ),
            c2h_arr[_idx(k_cell, kms)][None, :, None],
            rounding_token,
        )
        prefactor = _wrf_real_mul(
            _wrf_real_mul(
                _wrf_real_mul(ratio[:, None, :], 0.5, rounding_token),
                rdx,
                rounding_token,
            ),
            mass,
            rounding_token,
        )
        phi = _wrf_real_sub(
            _wrf_real_add(
                _wrf_real_sub(
                    _take3(ph_arr, i_u, k_cell + 1, j_u, ims, kms, jms),
                    _take3(ph_arr, i_u - 1, k_cell + 1, j_u, ims, kms, jms),
                    rounding_token,
                ),
                _take3(ph_arr, i_u, k_cell, j_u, ims, kms, jms),
                rounding_token,
            ),
            _take3(ph_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
            rounding_token,
        )
        p_term = _wrf_real_mul(
            _wrf_real_add(
                _take3(alt_arr, i_u, k_cell, j_u, ims, kms, jms),
                _take3(alt_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take3(p_arr, i_u, k_cell, j_u, ims, kms, jms),
                _take3(p_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
                rounding_token,
            ),
            rounding_token,
        )
        pb_term = _wrf_real_mul(
            _wrf_real_add(
                _take3(al_arr, i_u, k_cell, j_u, ims, kms, jms),
                _take3(al_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take3(pb_arr, i_u, k_cell, j_u, ims, kms, jms),
                _take3(pb_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
                rounding_token,
            ),
            rounding_token,
        )
        dpx = _wrf_real_mul(
            prefactor,
            _wrf_real_add(
                _wrf_real_add(phi, p_term, rounding_token),
                pb_term,
                rounding_token,
            ),
            rounding_token,
        )
        if return_diagnostics:
            dpx_primary_out = _set3(dpx_primary_out, i_u, k_cell, j_u, dpx, ims, kms, jms)
        if bool(non_hydrostatic):
            dpn_diff = _wrf_real_sub(
                dpn[:, 2 - int(kms):ktf + 2 - int(kms), :],
                dpn[:, 1 - int(kms):ktf + 1 - int(kms), :],
                rounding_token,
            )
            php_delta = _wrf_real_sub(
                _take3(php_arr, i_u, k_cell, j_u, ims, kms, jms),
                _take3(php_arr, i_u - 1, k_cell, j_u, ims, kms, jms),
                rounding_token,
            )
            dpn_term = _wrf_real_mul(
                rdnw_arr[_idx(k_cell, kms)][None, :, None],
                dpn_diff,
                rounding_token,
            )
            mu_term = _wrf_real_mul(
                0.5,
                _wrf_real_add(
                    _wrf_real_mul(
                        c1h_arr[_idx(k_cell, kms)][None, :, None],
                        _take2(mu_arr, i_u - 1, j_u, ims, jms)[:, None, :],
                        rounding_token,
                    ),
                    _wrf_real_mul(
                        c1h_arr[_idx(k_cell, kms)][None, :, None],
                        _take2(mu_arr, i_u, j_u, ims, jms)[:, None, :],
                        rounding_token,
                    ),
                    rounding_token,
                ),
                rounding_token,
            )
            term4 = _wrf_real_mul(
                _wrf_real_mul(
                    _wrf_real_mul(ratio[:, None, :], rdx, rounding_token),
                    php_delta,
                    rounding_token,
                ),
                _wrf_real_sub(dpn_term, mu_term, rounding_token),
                rounding_token,
            )
            dpx = _wrf_real_add(dpx, term4, rounding_token)
            if return_diagnostics:
                dpx_total_out = _set3(dpx_total_out, i_u, k_cell, j_u, dpx, ims, kms, jms)
            scaled = _wrf_real_mul(
                _take3(cqu_arr, i_u, k_cell, j_u, ims, kms, jms),
                dpx,
                rounding_token,
            )
            update = _wrf_real_sub(
                _take3(ru_out, i_u, k_cell, j_u, ims, kms, jms),
                scaled,
                rounding_token,
            )
        else:
            update = _wrf_real_sub(
                _take3(ru_out, i_u, k_cell, j_u, ims, kms, jms),
                dpx,
                rounding_token,
            )
        ru_out = _set3(ru_out, i_u, k_cell, j_u, update, ims, kms, jms)

    if return_diagnostics:
        return ru_out, rv_out, dpx_primary_out, dpx_total_out, dpy_primary_out, dpy_total_out
    return ru_out, rv_out


def coriolis_vectorized(
    ru: Any,
    rv: Any,
    rw: Any,
    ru_tend: Any,
    rv_tend: Any,
    rw_tend: Any,
    config_flags: Any,
    msftx: Any,
    msfty: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    f: Any,
    e: Any,
    sina: Any,
    cosa: Any,
    fzm: Any,
    fzp: Any,
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
) -> tuple[Any, Any, Any]:
    """Vectorized WRF ``coriolis`` tendency updates."""

    del kds, ime, jme, kme
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rw_arr = jnp.asarray(rw)
    ru_out = jnp.asarray(ru_tend).copy()
    rv_out = jnp.asarray(rv_tend).copy()
    rw_out = jnp.asarray(rw_tend).copy()
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    f_arr = jnp.asarray(f)
    e_arr = jnp.asarray(e)
    sina_arr = jnp.asarray(sina)
    cosa_arr = jnp.asarray(cosa)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    ktf = min(int(kte), int(kde) - 1)

    def sum4(first: Any, second: Any, third: Any, fourth: Any) -> Any:
        value = _wrf_real_add(first, second, rounding_token)
        value = _wrf_real_add(value, third, rounding_token)
        return _wrf_real_add(value, fourth, rounding_token)

    # U tendency.
    i_start = int(its)
    i_end = int(ite)
    if bool(_read(config_flags, "open_xs", False)) or specified:
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified:
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)
    i_vals = jnp.arange(i_start, i_end + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        ratio = _wrf_real_div(
            _take2(msfux_arr, i_vals, j_vals, ims, jms),
            _take2(msfuy_arr, i_vals, j_vals, ims, jms),
            rounding_token,
        )
        f_sum = _wrf_real_add(
            _take2(f_arr, i_vals, j_vals, ims, jms),
            _take2(f_arr, i_vals - 1, j_vals, ims, jms),
            rounding_token,
        )
        e_sum = _wrf_real_add(
            _take2(e_arr, i_vals, j_vals, ims, jms),
            _take2(e_arr, i_vals - 1, j_vals, ims, jms),
            rounding_token,
        )
        cosa_sum = _wrf_real_add(
            _take2(cosa_arr, i_vals, j_vals, ims, jms),
            _take2(cosa_arr, i_vals - 1, j_vals, ims, jms),
            rounding_token,
        )
        rv_sum = sum4(
            _take3(rv_arr, i_vals - 1, k_vals, j_vals + 1, ims, kms, jms),
            _take3(rv_arr, i_vals, k_vals, j_vals + 1, ims, kms, jms),
            _take3(rv_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
            _take3(rv_arr, i_vals, k_vals, j_vals, ims, kms, jms),
        )
        rw_sum = sum4(
            _take3(rw_arr, i_vals - 1, k_vals + 1, j_vals, ims, kms, jms),
            _take3(rw_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
            _take3(rw_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
            _take3(rw_arr, i_vals, k_vals, j_vals, ims, kms, jms),
        )
        f_term = _wrf_real_mul_chain(
            ratio[:, None, :],
            0.5,
            f_sum[:, None, :],
            0.25,
            rv_sum,
            rounding_token=rounding_token,
        )
        e_term = _wrf_real_mul_chain(
            0.5,
            e_sum[:, None, :],
            0.5,
            cosa_sum[:, None, :],
            0.25,
            rw_sum,
            rounding_token=rounding_token,
        )
        values = _wrf_real_sub(
            _wrf_real_add(
                _take3(ru_out, i_vals, k_vals, j_vals, ims, kms, jms),
                f_term,
                rounding_token,
            ),
            e_term,
            rounding_token,
        )
        ru_out = _set3(ru_out, i_vals, k_vals, j_vals, values, ims, kms, jms)

    # V tendency.
    j_start = int(jts)
    j_end = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_start = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_end = min(int(jde) - 1, int(jte))
    i_vals = jnp.arange(int(its), min(int(ide) - 1, int(ite)) + 1, dtype=jnp.int32)
    j_vals = jnp.arange(j_start, j_end + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        ratio = _wrf_real_div(
            _take2(msfvy_arr, i_vals, j_vals, ims, jms),
            _take2(msfvx_arr, i_vals, j_vals, ims, jms),
            rounding_token,
        )
        f_sum = _wrf_real_add(
            _take2(f_arr, i_vals, j_vals, ims, jms),
            _take2(f_arr, i_vals, j_vals - 1, ims, jms),
            rounding_token,
        )
        e_sum = _wrf_real_add(
            _take2(e_arr, i_vals, j_vals, ims, jms),
            _take2(e_arr, i_vals, j_vals - 1, ims, jms),
            rounding_token,
        )
        sina_sum = _wrf_real_add(
            _take2(sina_arr, i_vals, j_vals, ims, jms),
            _take2(sina_arr, i_vals, j_vals - 1, ims, jms),
            rounding_token,
        )
        ru_sum = sum4(
            _take3(ru_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            _take3(ru_arr, i_vals + 1, k_vals, j_vals, ims, kms, jms),
            _take3(ru_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
            _take3(ru_arr, i_vals + 1, k_vals, j_vals - 1, ims, kms, jms),
        )
        rw_sum = sum4(
            _take3(rw_arr, i_vals, k_vals + 1, j_vals - 1, ims, kms, jms),
            _take3(rw_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
            _take3(rw_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
            _take3(rw_arr, i_vals, k_vals, j_vals, ims, kms, jms),
        )
        f_term = _wrf_real_mul_chain(
            ratio[:, None, :],
            0.5,
            f_sum[:, None, :],
            0.25,
            ru_sum,
            rounding_token=rounding_token,
        )
        e_term = _wrf_real_mul_chain(
            ratio[:, None, :],
            0.5,
            e_sum[:, None, :],
            0.5,
            sina_sum[:, None, :],
            0.25,
            rw_sum,
            rounding_token=rounding_token,
        )
        values = _wrf_real_add(
            _wrf_real_sub(
                _take3(rv_out, i_vals, k_vals, j_vals, ims, kms, jms),
                f_term,
                rounding_token,
            ),
            e_term,
            rounding_token,
        )
        rv_out = _set3(rv_out, i_vals, k_vals, j_vals, values, ims, kms, jms)

    # W tendency.
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_w = jnp.arange(int(kts) + 1, ktf + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_w.size) and int(j_vals.size):
        ru_vert = _wrf_real_mul(
            0.5,
            _wrf_real_add(
                _wrf_real_mul(
                    fzm_arr[_idx(k_w, kms)][None, :, None],
                    _wrf_real_add(
                        _take3(ru_arr, i_vals, k_w, j_vals, ims, kms, jms),
                        _take3(ru_arr, i_vals + 1, k_w, j_vals, ims, kms, jms),
                        rounding_token,
                    ),
                    rounding_token,
                ),
                _wrf_real_mul(
                    fzp_arr[_idx(k_w, kms)][None, :, None],
                    _wrf_real_add(
                        _take3(ru_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
                        _take3(ru_arr, i_vals + 1, k_w - 1, j_vals, ims, kms, jms),
                        rounding_token,
                    ),
                    rounding_token,
                ),
                rounding_token,
            ),
            rounding_token,
        )
        rv_vert = _wrf_real_mul(
            0.5,
            _wrf_real_add(
                _wrf_real_mul(
                    fzm_arr[_idx(k_w, kms)][None, :, None],
                    _wrf_real_add(
                        _take3(rv_arr, i_vals, k_w, j_vals, ims, kms, jms),
                        _take3(rv_arr, i_vals, k_w, j_vals + 1, ims, kms, jms),
                        rounding_token,
                    ),
                    rounding_token,
                ),
                _wrf_real_mul(
                    fzp_arr[_idx(k_w, kms)][None, :, None],
                    _wrf_real_add(
                        _take3(rv_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
                        _take3(rv_arr, i_vals, k_w - 1, j_vals + 1, ims, kms, jms),
                        rounding_token,
                    ),
                    rounding_token,
                ),
                rounding_token,
            ),
            rounding_token,
        )
        vertical = _wrf_real_sub(
            _wrf_real_mul(
                _take2(cosa_arr, i_vals, j_vals, ims, jms)[:, None, :],
                ru_vert,
                rounding_token,
            ),
            _wrf_real_mul_chain(
                _wrf_real_div(
                    _take2(msftx_arr, i_vals, j_vals, ims, jms),
                    _take2(msfty_arr, i_vals, j_vals, ims, jms),
                    rounding_token,
                )[:, None, :],
                _take2(sina_arr, i_vals, j_vals, ims, jms)[:, None, :],
                rv_vert,
                rounding_token=rounding_token,
            ),
            rounding_token,
        )
        values = _wrf_real_add(
            _take3(rw_out, i_vals, k_w, j_vals, ims, kms, jms),
            _wrf_real_mul(
                _take2(e_arr, i_vals, j_vals, ims, jms)[:, None, :],
                vertical,
                rounding_token,
            ),
            rounding_token,
        )
        rw_out = _set3(rw_out, i_vals, k_w, j_vals, values, ims, kms, jms)

    return ru_out, rv_out, rw_out


def curvature_vectorized(
    ru: Any,
    rv: Any,
    rw: Any,
    u: Any,
    v: Any,
    w: Any,
    ru_tend: Any,
    rv_tend: Any,
    rw_tend: Any,
    config_flags: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    xlat: Any,
    fzm: Any,
    fzp: Any,
    rdx: float,
    rdy: float,
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
    reradius: float = 1.0 / 6370.0e03,
    degrad: float = 3.1415926535897932384626433 / 180.0,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Vectorized WRF ``curvature`` tendency updates."""

    del kds, ime, jme, kme, w
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rw_arr = jnp.asarray(rw)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    ru_out = jnp.asarray(ru_tend).copy()
    rv_out = jnp.asarray(rv_tend).copy()
    rw_out = jnp.asarray(rw_tend).copy()
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    xlat_arr = jnp.asarray(xlat)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    vxgm = jnp.zeros_like(ru_arr)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    ktf = min(int(kte), int(kde) - 1)

    def sum4(first: Any, second: Any, third: Any, fourth: Any) -> Any:
        value = _wrf_real_add(first, second, rounding_token)
        value = _wrf_real_add(value, third, rounding_token)
        return _wrf_real_add(value, fourth, rounding_token)

    i0 = int(its) - 1
    i1 = int(ite)
    j0 = int(jts) - 1
    j1 = int(jte)
    if (bool(_read(config_flags, "open_xs", False)) or specified) and int(its) == int(ids):
        i0 = int(its)
    if (bool(_read(config_flags, "open_xe", False)) or specified) and int(ite) == int(ide):
        i1 = int(ite) - 1
    if (bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False))) and int(jts) == int(jds):
        j0 = int(jts)
    if (bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False))) and int(jte) == int(jde):
        j1 = int(jte) - 1
    if bool(_read(config_flags, "periodic_x", False)):
        i0 = int(its) - 1
        i1 = int(ite)
    i_vx = jnp.arange(i0, i1 + 1, dtype=jnp.int32)
    k_vals = jnp.arange(int(kts), ktf + 1, dtype=jnp.int32)
    j_vx = jnp.arange(j0, j1 + 1, dtype=jnp.int32)
    if int(i_vx.size) and int(k_vals.size) and int(j_vx.size):
        u_term = _wrf_real_mul_chain(
            0.5,
            _wrf_real_add(
                _take3(u_arr, i_vx, k_vals, j_vx, ims, kms, jms),
                _take3(u_arr, i_vx + 1, k_vals, j_vx, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take2(msfvx_arr, i_vx, j_vx + 1, ims, jms),
                _take2(msfvx_arr, i_vx, j_vx, ims, jms),
                rounding_token,
            )[:, None, :],
            rdy,
            rounding_token=rounding_token,
        )
        v_term = _wrf_real_mul_chain(
            0.5,
            _wrf_real_add(
                _take3(v_arr, i_vx, k_vals, j_vx, ims, kms, jms),
                _take3(v_arr, i_vx, k_vals, j_vx + 1, ims, kms, jms),
                rounding_token,
            ),
            _wrf_real_sub(
                _take2(msfuy_arr, i_vx + 1, j_vx, ims, jms),
                _take2(msfuy_arr, i_vx, j_vx, ims, jms),
                rounding_token,
            )[:, None, :],
            rdx,
            rounding_token=rounding_token,
        )
        values = _wrf_real_sub(u_term, v_term, rounding_token)
        vxgm = _set3(vxgm, i_vx, k_vals, j_vx, values, ims, kms, jms)

    if (bool(_read(config_flags, "open_xs", False)) or (specified and not bool(_read(config_flags, "periodic_x", False)))) and int(its) == int(ids):
        i_dst = jnp.asarray([int(its) - 1], dtype=jnp.int32)
        i_src = jnp.asarray([int(its)], dtype=jnp.int32)
        j_copy = jnp.arange(int(jts), int(jte), dtype=jnp.int32)
        vxgm = _set3(vxgm, i_dst, k_vals, j_copy, _take3(vxgm, i_src, k_vals, j_copy, ims, kms, jms), ims, kms, jms)
    if (bool(_read(config_flags, "open_xe", False)) or (specified and not bool(_read(config_flags, "periodic_x", False)))) and int(ite) == int(ide):
        i_dst = jnp.asarray([int(ite)], dtype=jnp.int32)
        i_src = jnp.asarray([int(ite) - 1], dtype=jnp.int32)
        j_copy = jnp.arange(int(jts), int(jte), dtype=jnp.int32)
        vxgm = _set3(vxgm, i_dst, k_vals, j_copy, _take3(vxgm, i_src, k_vals, j_copy, ims, kms, jms), ims, kms, jms)
    if (bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False))) and int(jts) == int(jds):
        j_dst = jnp.asarray([int(jts) - 1], dtype=jnp.int32)
        j_src = jnp.asarray([int(jts)], dtype=jnp.int32)
        i_copy = jnp.arange(int(its) - 1, int(ite) + 1, dtype=jnp.int32)
        vxgm = _set3(vxgm, i_copy, k_vals, j_dst, _take3(vxgm, i_copy, k_vals, j_src, ims, kms, jms), ims, kms, jms)
    if (bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False))) and int(jte) == int(jde):
        j_dst = jnp.asarray([int(jte)], dtype=jnp.int32)
        j_src = jnp.asarray([int(jte) - 1], dtype=jnp.int32)
        i_copy = jnp.arange(int(its) - 1, int(ite) + 1, dtype=jnp.int32)
        vxgm = _set3(vxgm, i_copy, k_vals, j_dst, _take3(vxgm, i_copy, k_vals, j_src, ims, kms, jms), ims, kms, jms)

    i_start = int(its)
    i_end = int(ite)
    if bool(_read(config_flags, "open_xs", False)) or specified:
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified:
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)
    map_or_polar = int(_read(config_flags, "map_proj", 0)) == 6 or bool(_read(config_flags, "polar", False))

    i_vals = jnp.arange(i_start, i_end + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jde) - 1, int(jte)) + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        rv_avg = _wrf_real_mul(
            0.25,
            sum4(
                _take3(rv_arr, i_vals - 1, k_vals, j_vals + 1, ims, kms, jms),
                _take3(rv_arr, i_vals, k_vals, j_vals + 1, ims, kms, jms),
                _take3(rv_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
                _take3(rv_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
            rounding_token,
        )
        rw_avg = _wrf_real_mul(
            0.25,
            sum4(
                _take3(rw_arr, i_vals - 1, k_vals + 1, j_vals, ims, kms, jms),
                _take3(rw_arr, i_vals - 1, k_vals, j_vals, ims, kms, jms),
                _take3(rw_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
                _take3(rw_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
            rounding_token,
        )
        if map_or_polar:
            ratio = _wrf_real_div(
                _take2(msfux_arr, i_vals, j_vals, ims, jms),
                _take2(msfuy_arr, i_vals, j_vals, ims, jms),
                rounding_token,
            )
            tan_lat = _store_wrf_real(
                jnp.tan(
                    _wrf_real_mul(
                        _take2(xlat_arr, i_vals, j_vals, ims, jms),
                        degrad,
                        rounding_token,
                    )
                ),
                rounding_token,
            )
            horizontal = _wrf_real_mul(
                _wrf_real_mul(ratio[:, None, :], rv_avg, rounding_token),
                tan_lat[:, None, :],
                rounding_token,
            )
            delta = _wrf_real_mul_chain(
                _take3(u_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                reradius,
                _wrf_real_sub(horizontal, rw_avg, rounding_token),
                rounding_token=rounding_token,
            )
            values = _wrf_real_add(
                _take3(ru_out, i_vals, k_vals, j_vals, ims, kms, jms),
                delta,
                rounding_token,
            )
        else:
            horizontal = _wrf_real_mul_chain(
                0.5,
                _wrf_real_add(
                    _take3(vxgm, i_vals, k_vals, j_vals, ims, kms, jms),
                    _take3(vxgm, i_vals - 1, k_vals, j_vals, ims, kms, jms),
                    rounding_token,
                ),
                rv_avg,
                rounding_token=rounding_token,
            )
            vertical = _wrf_real_mul_chain(
                _take3(u_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                reradius,
                rw_avg,
                rounding_token=rounding_token,
            )
            values = _wrf_real_sub(
                _wrf_real_add(
                    _take3(ru_out, i_vals, k_vals, j_vals, ims, kms, jms),
                    horizontal,
                    rounding_token,
                ),
                vertical,
                rounding_token,
            )
        ru_out = _set3(ru_out, i_vals, k_vals, j_vals, values, ims, kms, jms)

    j_start = int(jts)
    j_end = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_start = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_end = min(int(jde) - 1, int(jte))
    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    j_vals = jnp.arange(j_start, j_end + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_vals.size) and int(j_vals.size):
        ru_avg = _wrf_real_mul(
            0.25,
            sum4(
                _take3(ru_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                _take3(ru_arr, i_vals + 1, k_vals, j_vals, ims, kms, jms),
                _take3(ru_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
                _take3(ru_arr, i_vals + 1, k_vals, j_vals - 1, ims, kms, jms),
            ),
            rounding_token,
        )
        rw_avg = _wrf_real_mul(
            0.25,
            sum4(
                _take3(rw_arr, i_vals, k_vals + 1, j_vals - 1, ims, kms, jms),
                _take3(rw_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
                _take3(rw_arr, i_vals, k_vals + 1, j_vals, ims, kms, jms),
                _take3(rw_arr, i_vals, k_vals, j_vals, ims, kms, jms),
            ),
            rounding_token,
        )
        ratio = _wrf_real_div(
            _take2(msfvy_arr, i_vals, j_vals, ims, jms),
            _take2(msfvx_arr, i_vals, j_vals, ims, jms),
            rounding_token,
        )
        if map_or_polar:
            u_avg = _wrf_real_mul(
                0.25,
                sum4(
                    _take3(u_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                    _take3(u_arr, i_vals + 1, k_vals, j_vals, ims, kms, jms),
                    _take3(u_arr, i_vals, k_vals, j_vals - 1, ims, kms, jms),
                    _take3(u_arr, i_vals + 1, k_vals, j_vals - 1, ims, kms, jms),
                ),
                rounding_token,
            )
            tan_lat = _store_wrf_real(
                jnp.tan(
                    _wrf_real_mul_chain(
                        _wrf_real_add(
                            _take2(xlat_arr, i_vals, j_vals, ims, jms),
                            _take2(xlat_arr, i_vals, j_vals - 1, ims, jms),
                            rounding_token,
                        ),
                        0.5,
                        degrad,
                        rounding_token=rounding_token,
                    )
                ),
                rounding_token,
            )
            horizontal = _wrf_real_mul_chain(
                u_avg,
                tan_lat[:, None, :],
                ru_avg,
                rounding_token=rounding_token,
            )
            vertical = _wrf_real_mul(
                _take3(v_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                rw_avg,
                rounding_token,
            )
            delta = _wrf_real_mul_chain(
                ratio[:, None, :],
                reradius,
                _wrf_real_add(horizontal, vertical, rounding_token),
                rounding_token=rounding_token,
            )
            values = _wrf_real_sub(
                _take3(rv_out, i_vals, k_vals, j_vals, ims, kms, jms),
                delta,
                rounding_token,
            )
        else:
            horizontal = _wrf_real_mul_chain(
                0.5,
                _wrf_real_add(
                    _take3(vxgm, i_vals, k_vals, j_vals, ims, kms, jms),
                    _take3(vxgm, i_vals, k_vals, j_vals - 1, ims, kms, jms),
                    rounding_token,
                ),
                ru_avg,
                rounding_token=rounding_token,
            )
            vertical = _wrf_real_mul_chain(
                ratio[:, None, :],
                _take3(v_arr, i_vals, k_vals, j_vals, ims, kms, jms),
                reradius,
                rw_avg,
                rounding_token=rounding_token,
            )
            values = _wrf_real_sub(
                _wrf_real_sub(
                    _take3(rv_out, i_vals, k_vals, j_vals, ims, kms, jms),
                    horizontal,
                    rounding_token,
                ),
                vertical,
                rounding_token,
            )
        rv_out = _set3(rv_out, i_vals, k_vals, j_vals, values, ims, kms, jms)

    i_vals = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1, dtype=jnp.int32)
    k_w = jnp.arange(max(2, int(kts)), ktf + 1, dtype=jnp.int32)
    j_vals = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1, dtype=jnp.int32)
    if int(i_vals.size) and int(k_w.size) and int(j_vals.size):
        def vertical_pair(
            current_left: Any,
            current_right: Any,
            previous_left: Any,
            previous_right: Any,
        ) -> Any:
            current = _wrf_real_mul(
                fzm_arr[_idx(k_w, kms)][None, :, None],
                _wrf_real_add(current_left, current_right, rounding_token),
                rounding_token,
            )
            previous = _wrf_real_mul(
                fzp_arr[_idx(k_w, kms)][None, :, None],
                _wrf_real_add(previous_left, previous_right, rounding_token),
                rounding_token,
            )
            return _wrf_real_mul(
                0.5,
                _wrf_real_add(current, previous, rounding_token),
                rounding_token,
            )

        ru_vert = vertical_pair(
            _take3(ru_arr, i_vals, k_w, j_vals, ims, kms, jms),
            _take3(ru_arr, i_vals + 1, k_w, j_vals, ims, kms, jms),
            _take3(ru_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
            _take3(ru_arr, i_vals + 1, k_w - 1, j_vals, ims, kms, jms),
        )
        u_vert = vertical_pair(
            _take3(u_arr, i_vals, k_w, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals + 1, k_w, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
            _take3(u_arr, i_vals + 1, k_w - 1, j_vals, ims, kms, jms),
        )
        rv_vert = vertical_pair(
            _take3(rv_arr, i_vals, k_w, j_vals, ims, kms, jms),
            _take3(rv_arr, i_vals, k_w, j_vals + 1, ims, kms, jms),
            _take3(rv_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
            _take3(rv_arr, i_vals, k_w - 1, j_vals + 1, ims, kms, jms),
        )
        v_vert = vertical_pair(
            _take3(v_arr, i_vals, k_w, j_vals, ims, kms, jms),
            _take3(v_arr, i_vals, k_w, j_vals + 1, ims, kms, jms),
            _take3(v_arr, i_vals, k_w - 1, j_vals, ims, kms, jms),
            _take3(v_arr, i_vals, k_w - 1, j_vals + 1, ims, kms, jms),
        )
        ratio = _wrf_real_div(
            _take2(msftx_arr, i_vals, j_vals, ims, jms),
            _take2(msfty_arr, i_vals, j_vals, ims, jms),
            rounding_token,
        )
        kinetic = _wrf_real_add(
            _wrf_real_mul(ru_vert, u_vert, rounding_token),
            _wrf_real_mul_chain(
                ratio[:, None, :],
                rv_vert,
                v_vert,
                rounding_token=rounding_token,
            ),
            rounding_token,
        )
        values = _wrf_real_add(
            _take3(rw_out, i_vals, k_w, j_vals, ims, kms, jms),
            _wrf_real_mul(reradius, kinetic, rounding_token),
            rounding_token,
        )
        rw_out = _set3(rw_out, i_vals, k_w, j_vals, values, ims, kms, jms)

    return ru_out, rv_out, rw_out


__all__ = ["rhs_ph_vectorized", "horizontal_pressure_gradient_vectorized", "coriolis_vectorized", "curvature_vectorized"]
