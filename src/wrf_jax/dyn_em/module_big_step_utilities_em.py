"""Kernels translated from WRF ``dyn_em/module_big_step_utilities_em``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from jax import lax
import jax.numpy as jnp

from wrf_jax.core.column_scan import scan_columns
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.optimization import semantic_call
from wrf_jax.core.real_arithmetic import column_barrier, column_binary, real_assignment
from wrf_jax.core.wrf_math import scalar_powf_for_wrf, vector_powf_for_wrf

W_ALPHA = 0.3
W_BETA = 1.0
DEGRAD = jnp.pi / 180.0


def _idx(values: Any, lower_bound: int) -> Any:
    return values - int(lower_bound)


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _wrf_real(value: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    array = jnp.asarray(value, dtype=dtype)
    if rounding_token is not None:
        return _store_wrf_real(array, rounding_token)
    return lax.optimization_barrier(array)


def _wrf_real_add(a: Any, b: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    if rounding_token is not None:
        return _store_wrf_real(
            jnp.asarray(a, dtype=dtype) + jnp.asarray(b, dtype=dtype),
            rounding_token,
        )
    return _wrf_real(
        _wrf_real(a, dtype) + _wrf_real(b, dtype),
        dtype,
    )


def _wrf_real_sub(a: Any, b: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    if rounding_token is not None:
        return _store_wrf_real(
            jnp.asarray(a, dtype=dtype) - jnp.asarray(b, dtype=dtype),
            rounding_token,
        )
    return _wrf_real(
        _wrf_real(a, dtype) - _wrf_real(b, dtype),
        dtype,
    )


def _wrf_real_mul(a: Any, b: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    if rounding_token is not None:
        return _store_wrf_real(
            jnp.asarray(a, dtype=dtype) * jnp.asarray(b, dtype=dtype),
            rounding_token,
        )
    return _wrf_real(
        _wrf_real(a, dtype) * _wrf_real(b, dtype),
        dtype,
    )


def _wrf_real_div(a: Any, b: Any, dtype: Any = jnp.float32, rounding_token: Any | None = None) -> Any:
    if rounding_token is not None:
        return _store_wrf_real(
            jnp.asarray(a, dtype=dtype) / jnp.asarray(b, dtype=dtype),
            rounding_token,
        )
    return _wrf_real(
        _wrf_real(a, dtype) / _wrf_real(b, dtype),
        dtype,
    )


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    """Force one WRF REAL store inside a larger XLA program."""

    return real_assignment(value, rounding_token)


def _wrf_patch_scalar_tail_offsets(
    *,
    ids: int,
    ide: int,
    i_start: int,
    i_end: int,
    nproc_x: int,
    vector_width: int = 4,
) -> tuple[int, ...]:
    """Return global-kernel offsets executed by WRF's scalar loop tails.

    RSL_LITE distributes remainder points equally toward both ends of the
    processor mesh.  GCC then handles each rank-local ``i`` loop as a
    four-wide vector body followed by a scalar tail.  The returned offsets
    reproduce those executable statement semantics without decomposing the
    JAX state or communicating between artificial ranks.
    """

    domain_length = int(ide) - int(ids) + 1
    if domain_length <= 0 or int(i_end) < int(i_start):
        return ()
    processor_count = min(max(1, int(nproc_x)), domain_length)
    width = domain_length // processor_count
    remainder = domain_length % processor_count
    left_extra = remainder // 2
    patch_widths = (
        (width + 1,) * left_extra
        + (width,) * (processor_count - remainder)
        + (width + 1,) * (remainder - left_extra)
    )

    offsets: list[int] = []
    patch_start = int(ids)
    for patch_width in patch_widths:
        patch_end = patch_start + patch_width - 1
        loop_start = max(patch_start, int(i_start))
        loop_end = min(patch_end, int(i_end))
        loop_length = max(0, loop_end - loop_start + 1)
        scalar_length = loop_length % int(vector_width)
        if scalar_length:
            scalar_start = loop_end - scalar_length + 1
            offsets.extend(
                global_i - int(i_start)
                for global_i in range(scalar_start, loop_end + 1)
            )
        patch_start = patch_end + 1
    return tuple(offsets)


def _powf_for_wrf_patch_i_loops(
    base: Any,
    exponent: Any,
    *,
    ids: int,
    ide: int,
    i_start: int,
    i_end: int,
    nproc_x: int,
) -> Any:
    """Match WRF vector bodies and scalar tails for a reference MPI mesh."""

    base_arr = jnp.asarray(base, dtype=jnp.float32)
    result = vector_powf_for_wrf(base_arr, exponent)
    scalar_offsets = _wrf_patch_scalar_tail_offsets(
        ids=ids,
        ide=ide,
        i_start=i_start,
        i_end=i_end,
        nproc_x=nproc_x,
    )
    if not scalar_offsets:
        return result
    offset_array = jnp.asarray(scalar_offsets, dtype=jnp.int32)
    scalar_values = scalar_powf_for_wrf(
        base_arr[offset_array, ...],
        jnp.asarray(exponent, dtype=jnp.float32),
    )
    return result.at[offset_array, ...].set(scalar_values)


def calc_mu_uv(
    config_flags: Any,
    mu: Any,
    mub: Any,
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
) -> tuple[Any, Any]:
    """Return WRF ``calc_mu_uv`` outputs ``muu`` and ``muv``."""

    del kds, kde, kms, kme, kts, kte

    mu_arr = jnp.asarray(mu)
    mub_arr = jnp.asarray(mub)
    shape = (int(ime) - int(ims) + 1, int(jme) - int(jms) + 1)
    muu = jnp.zeros(shape, dtype=mu_arr.dtype)
    muv = jnp.zeros(shape, dtype=mu_arr.dtype)

    periodic_x = bool(_read(config_flags, "periodic_x", False))
    periodic_y = bool(_read(config_flags, "periodic_y", False))

    itf = int(ite)
    jtf = min(int(jte), int(jde) - 1)
    if int(jts) <= jtf and int(its) <= itf:
        i_vals = jnp.arange(int(its), itf + 1)
        j_vals = jnp.arange(int(jts), jtf + 1)
        left_neighbor = i_vals - 1
        if int(its) == int(ids):
            left_edge_neighbor = int(its) - 1 if periodic_x else int(its)
            left_neighbor = jnp.where(i_vals == int(its), left_edge_neighbor, left_neighbor)
        right_first = i_vals
        right_second = left_neighbor
        if int(ite) == int(ide):
            right_edge_neighbor = int(ite) if periodic_x else int(ite) - 1
            right_first = jnp.where(i_vals == int(ite), int(ite) - 1, right_first)
            right_second = jnp.where(i_vals == int(ite), right_edge_neighbor, right_second)

        ii = i_vals - int(ims)
        jj = j_vals - int(jms)
        ia = right_first - int(ims)
        ib = right_second - int(ims)
        values = 0.5 * (
            mu_arr[ia[:, None], jj[None, :]]
            + mu_arr[ib[:, None], jj[None, :]]
            + mub_arr[ia[:, None], jj[None, :]]
            + mub_arr[ib[:, None], jj[None, :]]
        )
        muu = muu.at[ii[:, None], jj[None, :]].set(values)

    itf = min(int(ite), int(ide) - 1)
    jtf = int(jte)
    if int(jts) <= jtf and int(its) <= itf:
        i_vals = jnp.arange(int(its), itf + 1)
        j_vals = jnp.arange(int(jts), jtf + 1)
        lower_neighbor = j_vals - 1
        if int(jts) == int(jds):
            lower_edge_neighbor = int(jts) - 1 if periodic_y else int(jts)
            lower_neighbor = jnp.where(j_vals == int(jts), lower_edge_neighbor, lower_neighbor)
        upper_first = j_vals
        upper_second = lower_neighbor
        if int(jte) == int(jde):
            upper_edge_neighbor = int(jte) if periodic_y else int(jte) - 1
            upper_first = jnp.where(j_vals == int(jte), int(jte) - 1, upper_first)
            upper_second = jnp.where(j_vals == int(jte), upper_edge_neighbor, upper_second)

        ii = i_vals - int(ims)
        jj = j_vals - int(jms)
        ja = upper_first - int(jms)
        jb = upper_second - int(jms)
        values = 0.5 * (
            mu_arr[ii[:, None], ja[None, :]]
            + mu_arr[ii[:, None], jb[None, :]]
            + mub_arr[ii[:, None], ja[None, :]]
            + mub_arr[ii[:, None], jb[None, :]]
        )
        muv = muv.at[ii[:, None], jj[None, :]].set(values)

    return muu, muv


def calc_mu_uv_1(
    config_flags: Any,
    mu: Any,
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
) -> tuple[Any, Any]:
    """Return WRF ``calc_mu_uv_1`` outputs without base-state mass."""

    mu_arr = jnp.asarray(mu)
    return calc_mu_uv(
        config_flags,
        mu_arr,
        jnp.zeros_like(mu_arr),
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )


def calc_alt(
    al: Any,
    alb: Any,
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
) -> Any:
    """Return WRF ``calc_alt`` output ``alt = al + alb`` over the active tile."""

    del ids, jds, kds

    al_arr = jnp.asarray(al)
    alb_arr = jnp.asarray(alb)
    out = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1), dtype=al_arr.dtype)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)

    if itf < int(its) or ktf < int(kts) or jtf < int(jts):
        return out

    start = (int(its) - int(ims), int(kts) - int(kms), int(jts) - int(jms))
    tile = (slice(start[0], itf - int(ims) + 1),
            slice(start[1], ktf - int(kms) + 1),
            slice(start[2], jtf - int(jms) + 1))
    values = al_arr[tile] + alb_arr[tile]
    return lax.dynamic_update_slice(out, values.astype(out.dtype), start)


def calc_p_rho_phi(
    moist: Any,
    n_moist: int,
    hypsometric_opt: int,
    alb: Any,
    mu: Any,
    muts: Any,
    c1: Any,
    c2: Any,
    c3h: Any,
    c4h: Any,
    c3f: Any,
    c4f: Any,
    ph: Any,
    phb: Any,
    pb: Any,
    t: Any,
    p0: float,
    t0: float,
    ptop: float,
    znu: Any,
    znw: Any,
    dnw: Any,
    rdnw: Any,
    rdn: Any,
    non_hydrostatic: bool,
    use_theta_m: int,
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
    param_first_scalar: int = 2,
    p_qv: int = 2,
    r_d: float = 287.0,
    r_v: float = 461.6,
    p1000mb: float = 100000.0,
    cpovcv: float = 1.4,
    cvpm: float = -1.0 / 1.4,
) -> tuple[Any, Any, Any, Any]:
    """Return WRF ``calc_p_rho_phi`` outputs ``al``, ``p``, ``ph``, and ``phb``."""

    del ids, jds, kds, znu, znw

    moist_arr = jnp.asarray(moist)
    alb_arr = jnp.asarray(alb)
    mu_arr = jnp.asarray(mu)
    muts_arr = jnp.asarray(muts)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    c3h_arr = jnp.asarray(c3h)
    c4h_arr = jnp.asarray(c4h)
    c3f_arr = jnp.asarray(c3f)
    c4f_arr = jnp.asarray(c4f)
    ph_out = jnp.asarray(ph).copy()
    phb_out = jnp.asarray(phb).copy()
    pb_arr = jnp.asarray(pb)
    t_arr = jnp.asarray(t)
    dnw_arr = jnp.asarray(dnw)
    rdnw_arr = jnp.asarray(rdnw)
    rdn_arr = jnp.asarray(rdn)

    shape = (int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1)
    dtype = t_arr.dtype
    al = jnp.zeros(shape, dtype=dtype)
    p = jnp.zeros(shape, dtype=dtype)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    rvovrd = float(r_v) / float(r_d)
    qv_index = int(p_qv) - 1
    moist_start = int(param_first_scalar) - 1

    def i3(i: int, k: int, j: int) -> tuple[int, int, int]:
        return int(i) - int(ims), int(k) - int(kms), int(j) - int(jms)

    def i2(i: int, j: int) -> tuple[int, int]:
        return int(i) - int(ims), int(j) - int(jms)

    def kidx(k: int) -> int:
        return int(k) - int(kms)

    def qtot_at(i: int, k: int, j: int) -> Any:
        if int(n_moist) < int(param_first_scalar):
            return 0.0
        return jnp.sum(moist_arr[i3(i, k, j) + (slice(moist_start, int(n_moist)),)])

    def qtot_half_at(i: int, k: int, j: int) -> Any:
        if int(n_moist) < int(param_first_scalar):
            return 0.0
        lower = moist_arr[i3(i, k, j) + (slice(moist_start, int(n_moist)),)]
        upper = moist_arr[i3(i, k + 1, j) + (slice(moist_start, int(n_moist)),)]
        return 0.5 * jnp.sum(lower + upper)

    def qvf_at(i: int, k: int, j: int) -> Any:
        return 1.0 + rvovrd * moist_arr[i3(i, k, j) + (qv_index,)]

    if bool(non_hydrostatic):
        if int(hypsometric_opt) == 1:
            for j in range(int(jts), jtf + 1):
                for k in range(int(kts), ktf + 1):
                    for i in range(int(its), itf + 1):
                        idx = i3(i, k, j)
                        value = -(
                            alb_arr[idx] * (c1_arr[kidx(k)] * mu_arr[i2(i, j)])
                            + rdnw_arr[kidx(k)] * (ph_out[i3(i, k + 1, j)] - ph_out[idx])
                        ) / (c1_arr[kidx(k)] * muts_arr[i2(i, j)] + c2_arr[kidx(k)])
                        al = al.at[idx].set(value)
        elif int(hypsometric_opt) == 2:
            for j in range(int(jts), jtf + 1):
                for k in range(int(kts), ktf + 1):
                    for i in range(int(its), itf + 1):
                        idx = i3(i, k, j)
                        pfu = c3f_arr[kidx(k + 1)] * muts_arr[i2(i, j)] + c4f_arr[kidx(k + 1)] + float(ptop)
                        pfd = c3f_arr[kidx(k)] * muts_arr[i2(i, j)] + c4f_arr[kidx(k)] + float(ptop)
                        phm = c3h_arr[kidx(k)] * muts_arr[i2(i, j)] + c4h_arr[kidx(k)] + float(ptop)
                        value = (
                            ph_out[i3(i, k + 1, j)]
                            - ph_out[idx]
                            + phb_out[i3(i, k + 1, j)]
                            - phb_out[idx]
                        ) / phm / jnp.log(pfd / pfu) - alb_arr[idx]
                        al = al.at[idx].set(value)
        else:
            raise ValueError("calc_p_rho_phi: hypsometric_opt should be 1 or 2")

        for j in range(int(jts), jtf + 1):
            for k in range(int(kts), ktf + 1):
                for i in range(int(its), itf + 1):
                    idx = i3(i, k, j)
                    qvf = 1.0 if int(use_theta_m) == 1 or int(n_moist) < int(param_first_scalar) else qvf_at(i, k, j)
                    temp = (float(r_d) * (float(t0) + t_arr[idx]) * qvf) / (float(p0) * (al[idx] + alb_arr[idx]))
                    p = p.at[idx].set((temp**float(cpovcv)) * float(p0) - pb_arr[idx])

    else:
        if int(n_moist) >= int(param_first_scalar):
            for j in range(int(jts), jtf + 1):
                k = ktf
                for i in range(int(its), itf + 1):
                    idx = i3(i, k, j)
                    qtot = qtot_at(i, k, j)
                    value_p = -0.5 * ((c1_arr[kidx(k)] * mu_arr[i2(i, j)]) + qtot * (c1_arr[kidx(k)] * muts_arr[i2(i, j)] + c2_arr[kidx(k)])) / rdnw_arr[kidx(k)]
                    p = p.at[idx].set(value_p)
                    qvf = 1.0 if int(use_theta_m) == 1 else qvf_at(i, k, j)
                    value_al = (float(r_d) / float(p1000mb)) * (t_arr[idx] + float(t0)) * qvf * (((value_p + pb_arr[idx]) / float(p1000mb)) ** float(cvpm)) - alb_arr[idx]
                    al = al.at[idx].set(value_al)

                for k in range(ktf - 1, int(kts) - 1, -1):
                    for i in range(int(its), itf + 1):
                        idx = i3(i, k, j)
                        qtot = qtot_half_at(i, k, j)
                        value_p = p[i3(i, k + 1, j)] - ((c1_arr[kidx(k)] * mu_arr[i2(i, j)]) + qtot * (c1_arr[kidx(k)] * muts_arr[i2(i, j)] + c2_arr[kidx(k)])) / rdn_arr[kidx(k + 1)]
                        p = p.at[idx].set(value_p)
                        qvf = 1.0 if int(use_theta_m) == 1 else qvf_at(i, k, j)
                        value_al = (float(r_d) / float(p1000mb)) * (t_arr[idx] + float(t0)) * qvf * (((value_p + pb_arr[idx]) / float(p1000mb)) ** float(cvpm)) - alb_arr[idx]
                        al = al.at[idx].set(value_al)
        else:
            for j in range(int(jts), jtf + 1):
                k = ktf
                for i in range(int(its), itf + 1):
                    idx = i3(i, k, j)
                    value_p = -0.5 * (c1_arr[kidx(k)] * mu_arr[i2(i, j)]) / rdnw_arr[kidx(k)]
                    p = p.at[idx].set(value_p)
                    value_al = (float(r_d) / float(p1000mb)) * (t_arr[idx] + float(t0)) * (((value_p + pb_arr[idx]) / float(p1000mb)) ** float(cvpm)) - alb_arr[idx]
                    al = al.at[idx].set(value_al)

                for k in range(ktf - 1, int(kts) - 1, -1):
                    for i in range(int(its), itf + 1):
                        idx = i3(i, k, j)
                        value_p = p[i3(i, k + 1, j)] - (c1_arr[kidx(k)] * mu_arr[i2(i, j)]) / rdn_arr[kidx(k + 1)]
                        p = p.at[idx].set(value_p)
                        value_al = (float(r_d) / float(p1000mb)) * (t_arr[idx] + float(t0)) * (((value_p + pb_arr[idx]) / float(p1000mb)) ** float(cvpm)) - alb_arr[idx]
                        al = al.at[idx].set(value_al)

        if int(hypsometric_opt) == 1:
            for j in range(int(jts), jtf + 1):
                for kk in range(2, ktf + 2):
                    k = kk - 1
                    for i in range(int(its), itf + 1):
                        value = ph_out[i3(i, k, j)] - dnw_arr[kidx(k)] * (
                            (c1_arr[kidx(k)] * muts_arr[i2(i, j)] + c2_arr[kidx(k)]) * al[i3(i, k, j)]
                            + (c1_arr[kidx(k)] * mu_arr[i2(i, j)]) * alb_arr[i3(i, k, j)]
                        )
                        ph_out = ph_out.at[i3(i, k + 1, j)].set(value)
        else:
            for j in range(int(jts), jtf + 1):
                for i in range(int(its), itf + 1):
                    ph_out = ph_out.at[i3(i, int(kts), j)].set(phb_out[i3(i, int(kts), j)])
                for k in range(int(kts) + 1, ktf + 2):
                    for i in range(int(its), itf + 1):
                        pfu = c3f_arr[kidx(k)] * muts_arr[i2(i, j)] + c4f_arr[kidx(k)] + float(ptop)
                        pfd = c3f_arr[kidx(k - 1)] * muts_arr[i2(i, j)] + c4f_arr[kidx(k - 1)] + float(ptop)
                        phm = c3h_arr[kidx(k - 1)] * muts_arr[i2(i, j)] + c4h_arr[kidx(k - 1)] + float(ptop)
                        value = ph_out[i3(i, k - 1, j)] + (al[i3(i, k - 1, j)] + alb_arr[i3(i, k - 1, j)]) * phm * jnp.log(pfd / pfu)
                        ph_out = ph_out.at[i3(i, k, j)].set(value)
                for k in range(int(kts), ktf + 2):
                    for i in range(int(its), itf + 1):
                        ph_out = ph_out.at[i3(i, k, j)].set(ph_out[i3(i, k, j)] - phb_out[i3(i, k, j)])

    return al, p, ph_out, phb_out


def calc_php(
    ph: Any,
    phb: Any,
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
) -> Any:
    """Return WRF ``calc_php`` half-level geopotential average."""

    del ids, jds, kds

    ph_arr = jnp.asarray(ph)
    phb_arr = jnp.asarray(phb)
    out = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1), dtype=ph_arr.dtype)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)

    if itf < int(its) or ktf < int(kts) or jtf < int(jts):
        return out

    start = (int(its) - int(ims), int(kts) - int(kms), int(jts) - int(jms))
    tile = (slice(start[0], itf - int(ims) + 1),
            slice(start[1], ktf - int(kms) + 1),
            slice(start[2], jtf - int(jms) + 1))
    above = (tile[0], slice(start[1] + 1, ktf - int(kms) + 2), tile[2])
    values = 0.5 * (
        phb_arr[tile] + phb_arr[above] + ph_arr[tile] + ph_arr[above]
    )
    return lax.dynamic_update_slice(out, values.astype(out.dtype), start)


def _vertical_mass_flux_step(carry: Any, inputs: Any, token: Any) -> Any:
    previous, mass_rate = carry
    delta_eta, mass_coefficient, divergence = inputs
    if token is None:
        current = previous - delta_eta * mass_coefficient * mass_rate - divergence
    else:
        weight = real_assignment(delta_eta * mass_coefficient, token)
        tendency = real_assignment(weight * mass_rate, token)
        current = real_assignment(previous - tendency, token)
        current = real_assignment(current - divergence, token)
    return (jnp.asarray(current, previous.dtype), mass_rate), current


def _mass_divergence_layer(previous: Any, inputs: Any, _parameter: Any) -> Any:
    return column_binary("add", previous, inputs[0]), None


def _column_mass_divergence(divergence: Any) -> Any:
    # WRF accumulates bottom-to-top; a parallel reduction changes omega.
    initial = jnp.zeros(divergence.shape[::2], dtype=divergence.dtype)
    total, _ = scan_columns(
        _mass_divergence_layer, initial, (jnp.moveaxis(divergence, 1, 0),)
    )
    return total


def calc_ww_cp(
    u: Any,
    v: Any,
    mup: Any,
    mub: Any,
    c1h: Any,
    c2h: Any,
    rdx: float,
    rdy: float,
    msftx: Any,
    msfty: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvx_inv: Any,
    msfvy: Any,
    dnw: Any,
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
    """Return WRF ``calc_ww_cp`` vertical mass flux diagnostic."""

    del ids, jds, kds, msfty, msfux, msfvx, msfvy

    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    mup_arr = jnp.asarray(mup)
    mub_arr = jnp.asarray(mub)
    c1h_arr = jnp.asarray(c1h)
    c2h_arr = jnp.asarray(c2h)
    msftx_arr = jnp.asarray(msftx)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_inv_arr = jnp.asarray(msfvx_inv)
    dnw_arr = jnp.asarray(dnw)
    out = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1), dtype=u_arr.dtype)

    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    itf = min(int(ite), int(ide) - 1)
    if itf < int(its) or jtf < int(jts) or ktf < int(kts):
        return out

    i_fortran = jnp.arange(int(its), itf + 1)
    k_fortran = jnp.arange(int(kts), ktf + 1)
    j_fortran = jnp.arange(int(jts), jtf + 1)
    ii = _idx(i_fortran, ims)
    ip1 = _idx(i_fortran + 1, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)
    jp1 = _idx(j_fortran + 1, jms)
    im1 = _idx(i_fortran - 1, ims)
    jm1 = _idx(j_fortran - 1, jms)
    # The global-array runtime does not carry WRF's low-side halo cells.  Avoid
    # Python/JAX negative-index wraparound at physical boundaries.
    im1 = jnp.where(im1 < 0, ii, im1)
    jm1 = jnp.where(jm1 < 0, jj, jm1)

    mup_i_j = mup_arr[ii[:, None], jj[None, :]]
    mub_i_j = mub_arr[ii[:, None], jj[None, :]]
    muu_i = 0.5 * (mup_i_j + mub_i_j + mup_arr[im1[:, None], jj[None, :]] + mub_arr[im1[:, None], jj[None, :]])
    muu_ip1 = 0.5 * (mup_arr[ip1[:, None], jj[None, :]] + mub_arr[ip1[:, None], jj[None, :]] + mup_i_j + mub_i_j)
    muv_j = 0.5 * (mup_i_j + mub_i_j + mup_arr[ii[:, None], jm1[None, :]] + mub_arr[ii[:, None], jm1[None, :]])
    muv_jp1 = 0.5 * (mup_arr[ii[:, None], jp1[None, :]] + mub_arr[ii[:, None], jp1[None, :]] + mup_i_j + mub_i_j)

    c1 = c1h_arr[kk][None, :, None]
    c2 = c2h_arr[kk][None, :, None]
    u_ip1 = u_arr[ip1[:, None, None], kk[None, :, None], jj[None, None, :]]
    u_i = u_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    v_jp1 = v_arr[ii[:, None, None], kk[None, :, None], jp1[None, None, :]]
    v_j = v_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]

    if rounding_token is None:
        x_flux = (
            (c1 * muu_ip1[:, None, :] + c2) * u_ip1 / msfuy_arr[ip1[:, None], jj[None, :]][:, None, :]
            - (c1 * muu_i[:, None, :] + c2) * u_i / msfuy_arr[ii[:, None], jj[None, :]][:, None, :]
        )
        y_flux = (
            (c1 * muv_jp1[:, None, :] + c2) * v_jp1 * msfvx_inv_arr[ii[:, None], jp1[None, :]][:, None, :]
            - (c1 * muv_j[:, None, :] + c2) * v_j * msfvx_inv_arr[ii[:, None], jj[None, :]][:, None, :]
        )
        divv = msftx_arr[ii[:, None], jj[None, :]][:, None, :] * dnw_arr[kk][None, :, None] * (float(rdx) * x_flux + float(rdy) * y_flux)
    else:
        store = lambda value: _store_wrf_real(value, rounding_token)
        x_weighted_ip1 = store(c1 * muu_ip1[:, None, :])
        x_weighted_i = store(c1 * muu_i[:, None, :])
        x_mass_ip1 = store(x_weighted_ip1 + c2)
        x_mass_i = store(x_weighted_i + c2)
        x_momentum_ip1 = store(x_mass_ip1 * u_ip1)
        x_momentum_i = store(x_mass_i * u_i)
        x_scaled_ip1 = store(x_momentum_ip1 / msfuy_arr[ip1[:, None], jj[None, :]][:, None, :])
        x_scaled_i = store(x_momentum_i / msfuy_arr[ii[:, None], jj[None, :]][:, None, :])
        x_flux = store(x_scaled_ip1 - x_scaled_i)
        y_weighted_jp1 = store(c1 * muv_jp1[:, None, :])
        y_weighted_j = store(c1 * muv_j[:, None, :])
        y_mass_jp1 = store(y_weighted_jp1 + c2)
        y_mass_j = store(y_weighted_j + c2)
        y_momentum_jp1 = store(y_mass_jp1 * v_jp1)
        y_momentum_j = store(y_mass_j * v_j)
        y_scaled_jp1 = store(y_momentum_jp1 * msfvx_inv_arr[ii[:, None], jp1[None, :]][:, None, :])
        y_scaled_j = store(y_momentum_j * msfvx_inv_arr[ii[:, None], jj[None, :]][:, None, :])
        y_flux = store(y_scaled_jp1 - y_scaled_j)
        x_divergence = store(float(rdx) * x_flux)
        y_divergence = store(float(rdy) * y_flux)
        horizontal_divergence = store(x_divergence + y_divergence)
        mapped_dnw = store(msftx_arr[ii[:, None], jj[None, :]][:, None, :] * dnw_arr[kk][None, :, None])
        divv = store(mapped_dnw * horizontal_divergence)
    # Keep the ordered column sum independent of the enclosing stage layout.
    dmdt = semantic_call(_column_mass_divergence, divv)

    prev = jnp.zeros((i_fortran.size, j_fortran.size), dtype=u_arr.dtype)
    if ktf >= 2:
        levels = jnp.arange(2, ktf + 1)
        previous_levels = levels - 1 - int(kms)
        if rounding_token is not None:
            prev = prev.astype(jnp.float32)
        local = levels - 1 - int(kts)
        _, history = scan_columns(
            _vertical_mass_flux_step,
            (prev, dmdt),
            (
                dnw_arr[previous_levels],
                c1h_arr[previous_levels],
                jnp.moveaxis(divv[:, local, :], 1, 0),
            ),
            rounding_token,
        )
        out = lax.dynamic_update_slice(
            out, jnp.moveaxis(history, 0, 1).astype(out.dtype),
            (int(its) - int(ims), 2 - int(kms), int(jts) - int(jms)),
        )

    return out


def calculate_full(
    rfieldb: Any,
    rfieldp: Any,
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
) -> Any:
    """Return WRF ``calculate_full`` output ``rfieldb + rfieldp``."""

    del ids, jds, kds

    base = jnp.asarray(rfieldb)
    pert = jnp.asarray(rfieldp)
    out = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1), dtype=base.dtype)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    i_fortran = jnp.arange(int(its) - 1, itf + 1)
    k_fortran = jnp.arange(int(kts), ktf + 1)
    j_fortran = jnp.arange(int(jts) - 1, jtf + 1)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return out

    ii = _idx(i_fortran, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)
    values = base[ii[:, None, None], kk[None, :, None], jj[None, None, :]] + pert[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    return out.at[ii[:, None, None], kk[None, :, None], jj[None, None, :]].set(values)


def conv_t_tendf_to_moist(
    t_1: Any,
    moist_old: Any,
    t_tendf: Any,
    moist_tend: Any,
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
    r_v: float = 461.6,
    r_d: float = 287.0,
    t0: float = 300.0,
    rounding_token: Any | None = None,
) -> Any:
    """Return WRF ``conv_t_tendf_to_moist`` updated ``t_tendf``."""

    del ids, jds, kds, kde, ime, jme, kme

    t_arr = jnp.asarray(t_1)
    moist_old_arr = jnp.asarray(moist_old)
    out = jnp.asarray(t_tendf).copy()
    moist_tend_arr = jnp.asarray(moist_tend)

    i_fortran = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1)
    k_fortran = jnp.arange(int(kts), int(kte))
    j_fortran = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return out

    ii = _idx(i_fortran, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)
    dtype = t_arr.dtype
    ratio_real = _wrf_real_div(
        _wrf_real(r_v, dtype, rounding_token),
        _wrf_real(r_d, dtype, rounding_token),
        dtype,
        rounding_token,
    )
    moist = moist_old_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    old_tend = out[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    theta = t_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    mtend = moist_tend_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    if rounding_token is None:
        values = (
            (1.0 + ratio_real * moist) * old_tend
            + ratio_real
            * (theta + jnp.asarray(t0, dtype=dtype))
            / (1.0 + ratio_real * moist)
            * mtend
        )
    else:
        one = _wrf_real(1.0, dtype, rounding_token)
        t0_real = _wrf_real(t0, dtype, rounding_token)

        first_vapor = _wrf_real_mul(ratio_real, moist, dtype, rounding_token)
        first_factor = _wrf_real_add(one, first_vapor, dtype, rounding_token)
        first_term = _wrf_real_mul(first_factor, old_tend, dtype, rounding_token)

        theta_plus_t0 = _wrf_real_add(theta, t0_real, dtype, rounding_token)
        second_numerator = _wrf_real_mul(ratio_real, theta_plus_t0, dtype, rounding_token)
        second_vapor = _wrf_real_mul(ratio_real, moist, dtype, rounding_token)
        second_denominator = _wrf_real_add(one, second_vapor, dtype, rounding_token)
        second_quotient = _wrf_real_div(second_numerator, second_denominator, dtype, rounding_token)
        second_term = _wrf_real_mul(second_quotient, mtend, dtype, rounding_token)
        values = _wrf_real_add(first_term, second_term, dtype, rounding_token)
    return out.at[ii[:, None, None], kk[None, :, None], jj[None, None, :]].set(values)


def initialize_moist_old(
    moist: Any,
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
) -> Any:
    """Return WRF ``initialize_moist_old`` output copied from ``moist``."""

    del ids, jds, kds, kde

    moist_arr = jnp.asarray(moist)
    out_shape = (int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1) + tuple(moist_arr.shape[3:])
    out = jnp.zeros(out_shape, dtype=moist_arr.dtype)

    i_fortran = jnp.arange(int(its), min(int(ite), int(ide) - 1) + 1)
    k_fortran = jnp.arange(int(kts), int(kte))
    j_fortran = jnp.arange(int(jts), min(int(jte), int(jde) - 1) + 1)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return out

    ii = _idx(i_fortran, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)
    idx = (ii[:, None, None], kk[None, :, None], jj[None, None, :])
    values = moist_arr[idx + (Ellipsis,)]
    return out.at[idx + (Ellipsis,)].set(values)


def coriolis(
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
) -> tuple[Any, Any, Any]:
    """Return WRF ``coriolis`` updated momentum tendencies."""

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

    def i2(i: int, j: int) -> tuple[int, int]:
        return int(i) - int(ims), int(j) - int(jms)

    def i3(i: int, k: int, j: int) -> tuple[int, int, int]:
        return int(i) - int(ims), int(k) - int(kms), int(j) - int(jms)

    i_start = int(its)
    i_end = int(ite)
    if bool(_read(config_flags, "open_xs", False)) or specified or bool(_read(config_flags, "nested", False)):
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified or bool(_read(config_flags, "nested", False)):
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)

    for j in range(int(jts), min(int(jte), int(jde) - 1) + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = ru_out[i3(i, k, j)]
                value = value + (msfux_arr[i2(i, j)] / msfuy_arr[i2(i, j)]) * 0.5 * (f_arr[i2(i, j)] + f_arr[i2(i - 1, j)]) * 0.25 * (
                    rv_arr[i3(i - 1, k, j + 1)] + rv_arr[i3(i, k, j + 1)] + rv_arr[i3(i - 1, k, j)] + rv_arr[i3(i, k, j)]
                )
                value = value - 0.5 * (e_arr[i2(i, j)] + e_arr[i2(i - 1, j)]) * 0.5 * (cosa_arr[i2(i, j)] + cosa_arr[i2(i - 1, j)]) * 0.25 * (
                    rw_arr[i3(i - 1, k + 1, j)] + rw_arr[i3(i - 1, k, j)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)]
                )
                ru_out = ru_out.at[i3(i, k, j)].set(value)

    j_start = int(jts)
    j_end = int(jte)
    if (
        bool(_read(config_flags, "open_ys", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ):
        j_start = max(int(jds) + 1, int(jts))
    if (
        bool(_read(config_flags, "open_ye", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ):
        j_end = min(int(jde) - 1, int(jte))

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(int(its), min(int(ide) - 1, int(ite)) + 1):
                value = rv_out[i3(i, k, j)]
                value = value - (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * 0.5 * (f_arr[i2(i, j)] + f_arr[i2(i, j - 1)]) * 0.25 * (
                    ru_arr[i3(i, k, j)] + ru_arr[i3(i + 1, k, j)] + ru_arr[i3(i, k, j - 1)] + ru_arr[i3(i + 1, k, j - 1)]
                )
                value = value + (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * 0.5 * (e_arr[i2(i, j)] + e_arr[i2(i, j - 1)]) * 0.5 * (
                    sina_arr[i2(i, j)] + sina_arr[i2(i, j - 1)]
                ) * 0.25 * (rw_arr[i3(i, k + 1, j - 1)] + rw_arr[i3(i, k, j - 1)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)])
                rv_out = rv_out.at[i3(i, k, j)].set(value)

    for j in range(int(jts), min(int(jte), int(jde) - 1) + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(int(its), min(int(ite), int(ide) - 1) + 1):
                value = rw_out[i3(i, k, j)] + e_arr[i2(i, j)] * (
                    cosa_arr[i2(i, j)]
                    * 0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (ru_arr[i3(i, k, j)] + ru_arr[i3(i + 1, k, j)])
                        + fzp_arr[int(k) - int(kms)] * (ru_arr[i3(i, k - 1, j)] + ru_arr[i3(i + 1, k - 1, j)])
                    )
                    - (msftx_arr[i2(i, j)] / msfty_arr[i2(i, j)])
                    * sina_arr[i2(i, j)]
                    * 0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (rv_arr[i3(i, k, j)] + rv_arr[i3(i, k, j + 1)])
                        + fzp_arr[int(k) - int(kms)] * (rv_arr[i3(i, k - 1, j)] + rv_arr[i3(i, k - 1, j + 1)])
                    )
                )
                rw_out = rw_out.at[i3(i, k, j)].set(value)

    return ru_out, rv_out, rw_out


def couple_momentum(
    muu: Any,
    u: Any,
    msfu: Any,
    muv: Any,
    v: Any,
    msfv: Any,
    msfv_inv: Any,
    mut: Any,
    w: Any,
    msft: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
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
    """Return WRF ``couple_momentum`` outputs ``ru``, ``rv``, and ``rw``."""

    del ids, jds, kds, msfv

    muu_arr = jnp.asarray(muu)
    muv_arr = jnp.asarray(muv)
    mut_arr = jnp.asarray(mut)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    w_arr = jnp.asarray(w)
    msfu_arr = jnp.asarray(msfu)
    msfv_inv_arr = jnp.asarray(msfv_inv)
    msft_arr = jnp.asarray(msft)
    c1h_arr = jnp.asarray(c1h)
    c2h_arr = jnp.asarray(c2h)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)
    shape = (int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1)
    def component(wind, mass, mapping, c1, c2, last_i, last_k, last_j, *, inverse_map=False):
        output = jnp.zeros(shape, dtype=u_arr.dtype)
        region = (
            slice(int(its) - int(ims), last_i - int(ims) + 1),
            slice(int(kts) - int(kms), last_k - int(kms) + 1),
            slice(int(jts) - int(jms), last_j - int(jms) + 1),
        )
        if any(part.stop <= part.start for part in region):
            return output
        ii, kk, jj = region
        wind_values = wind[region]
        c1_values = c1[kk][None, :, None]
        c2_values = c2[kk][None, :, None]
        mass_values = mass[ii, jj][:, None, :]
        map_values = mapping[ii, jj][:, None, :]
        if rounding_token is None:
            coupled = wind_values * (c1_values * mass_values + c2_values)
            values = coupled * map_values if inverse_map else coupled / map_values
        else:
            mass_product = _store_wrf_real(c1_values * mass_values, rounding_token)
            mass_weight = _store_wrf_real(mass_product + c2_values, rounding_token)
            coupled = _store_wrf_real(wind_values * mass_weight, rounding_token)
            mapped = coupled * map_values if inverse_map else coupled / map_values
            values = _store_wrf_real(mapped, rounding_token)
        return lax.dynamic_update_slice(
            output, values.astype(output.dtype), tuple(part.start for part in region)
        )

    ktf = min(int(kte), int(kde) - 1)
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    return (
        component(u_arr, muu_arr, msfu_arr, c1h_arr, c2h_arr, int(ite), ktf, jtf),
        component(v_arr, muv_arr, msfv_inv_arr, c1h_arr, c2h_arr, itf, ktf, int(jte), inverse_map=True),
        component(w_arr, mut_arr, msft_arr, c1f_arr, c2f_arr, itf, int(kte), jtf),
    )


def curvature(
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
) -> tuple[Any, Any, Any]:
    """Return WRF ``curvature`` updated momentum tendencies."""

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
    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)

    def i2(i: int, j: int) -> tuple[int, int]:
        return int(i) - int(ims), int(j) - int(jms)

    def i3(i: int, k: int, j: int) -> tuple[int, int, int]:
        return int(i) - int(ims), int(k) - int(kms), int(j) - int(jms)

    i_start = int(its) - 1
    i_end = int(ite)
    j_start = int(jts) - 1
    j_end = int(jte)
    if (bool(_read(config_flags, "open_xs", False)) or specified or bool(_read(config_flags, "nested", False))) and int(its) == int(ids):
        i_start = int(its)
    if (bool(_read(config_flags, "open_xe", False)) or specified or bool(_read(config_flags, "nested", False))) and int(ite) == int(ide):
        i_end = int(ite) - 1
    if (
        bool(_read(config_flags, "open_ys", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jts) == int(jds):
        j_start = int(jts)
    if (
        bool(_read(config_flags, "open_ye", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jte) == int(jde):
        j_end = int(jte) - 1
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its) - 1
        i_end = int(ite)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = 0.5 * (u_arr[i3(i, k, j)] + u_arr[i3(i + 1, k, j)]) * (msfvx_arr[i2(i, j + 1)] - msfvx_arr[i2(i, j)]) * float(rdy)
                value = value - 0.5 * (v_arr[i3(i, k, j)] + v_arr[i3(i, k, j + 1)]) * (msfuy_arr[i2(i + 1, j)] - msfuy_arr[i2(i, j)]) * float(rdx)
                vxgm = vxgm.at[i3(i, k, j)].set(value)

    if (bool(_read(config_flags, "open_xs", False)) or (specified and not bool(_read(config_flags, "periodic_x", False))) or bool(_read(config_flags, "nested", False))) and int(its) == int(ids):
        for j in range(int(jts), int(jte)):
            for k in range(int(kts), ktf + 1):
                vxgm = vxgm.at[i3(int(its) - 1, k, j)].set(vxgm[i3(int(its), k, j)])

    if (bool(_read(config_flags, "open_xe", False)) or (specified and not bool(_read(config_flags, "periodic_x", False))) or bool(_read(config_flags, "nested", False))) and int(ite) == int(ide):
        for j in range(int(jts), int(jte)):
            for k in range(int(kts), ktf + 1):
                vxgm = vxgm.at[i3(int(ite), k, j)].set(vxgm[i3(int(ite) - 1, k, j)])

    if (
        bool(_read(config_flags, "open_ys", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jts) == int(jds):
        for k in range(int(kts), ktf + 1):
            for i in range(int(its) - 1, int(ite) + 1):
                vxgm = vxgm.at[i3(i, k, int(jts) - 1)].set(vxgm[i3(i, k, int(jts))])

    if (
        bool(_read(config_flags, "open_ye", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jte) == int(jde):
        for k in range(int(kts), ktf + 1):
            for i in range(int(its) - 1, int(ite) + 1):
                vxgm = vxgm.at[i3(i, k, int(jte))].set(vxgm[i3(i, k, int(jte) - 1)])

    i_start = int(its)
    if bool(_read(config_flags, "open_xs", False)) or specified or bool(_read(config_flags, "nested", False)):
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified or bool(_read(config_flags, "nested", False)):
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)

    map_or_polar = int(_read(config_flags, "map_proj", 0)) == 6 or bool(_read(config_flags, "polar", False))
    for j in range(int(jts), min(int(jde) - 1, int(jte)) + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                if map_or_polar:
                    value = ru_out[i3(i, k, j)] + u_arr[i3(i, k, j)] * float(reradius) * (
                        (msfux_arr[i2(i, j)] / msfuy_arr[i2(i, j)])
                        * 0.25
                        * (rv_arr[i3(i - 1, k, j + 1)] + rv_arr[i3(i, k, j + 1)] + rv_arr[i3(i - 1, k, j)] + rv_arr[i3(i, k, j)])
                        * jnp.tan(xlat_arr[i2(i, j)] * float(degrad))
                        - 0.25 * (rw_arr[i3(i - 1, k + 1, j)] + rw_arr[i3(i - 1, k, j)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)])
                    )
                else:
                    value = ru_out[i3(i, k, j)] + 0.5 * (vxgm[i3(i, k, j)] + vxgm[i3(i - 1, k, j)]) * 0.25 * (
                        rv_arr[i3(i - 1, k, j + 1)] + rv_arr[i3(i, k, j + 1)] + rv_arr[i3(i - 1, k, j)] + rv_arr[i3(i, k, j)]
                    ) - u_arr[i3(i, k, j)] * float(reradius) * 0.25 * (
                        rw_arr[i3(i - 1, k + 1, j)] + rw_arr[i3(i - 1, k, j)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)]
                    )
                ru_out = ru_out.at[i3(i, k, j)].set(value)

    j_start = int(jts)
    if (
        bool(_read(config_flags, "open_ys", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ):
        j_start = max(int(jds) + 1, int(jts))
    if (
        bool(_read(config_flags, "open_ye", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ):
        j_end = min(int(jde) - 1, int(jte))

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(int(its), min(int(ite), int(ide) - 1) + 1):
                if map_or_polar:
                    value = rv_out[i3(i, k, j)] - (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * float(reradius) * (
                        0.25
                        * (u_arr[i3(i, k, j)] + u_arr[i3(i + 1, k, j)] + u_arr[i3(i, k, j - 1)] + u_arr[i3(i + 1, k, j - 1)])
                        * jnp.tan((xlat_arr[i2(i, j)] + xlat_arr[i2(i, j - 1)]) * 0.5 * float(degrad))
                        * 0.25
                        * (ru_arr[i3(i, k, j)] + ru_arr[i3(i + 1, k, j)] + ru_arr[i3(i, k, j - 1)] + ru_arr[i3(i + 1, k, j - 1)])
                        + v_arr[i3(i, k, j)]
                        * 0.25
                        * (rw_arr[i3(i, k + 1, j - 1)] + rw_arr[i3(i, k, j - 1)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)])
                    )
                else:
                    value = rv_out[i3(i, k, j)] - 0.5 * (vxgm[i3(i, k, j)] + vxgm[i3(i, k, j - 1)]) * 0.25 * (
                        ru_arr[i3(i, k, j)] + ru_arr[i3(i + 1, k, j)] + ru_arr[i3(i, k, j - 1)] + ru_arr[i3(i + 1, k, j - 1)]
                    ) - (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * v_arr[i3(i, k, j)] * float(reradius) * 0.25 * (
                        rw_arr[i3(i, k + 1, j - 1)] + rw_arr[i3(i, k, j - 1)] + rw_arr[i3(i, k + 1, j)] + rw_arr[i3(i, k, j)]
                    )
                rv_out = rv_out.at[i3(i, k, j)].set(value)

    for j in range(int(jts), min(int(jte), int(jde) - 1) + 1):
        for k in range(max(2, int(kts)), ktf + 1):
            for i in range(int(its), min(int(ite), int(ide) - 1) + 1):
                value = rw_out[i3(i, k, j)] + float(reradius) * (
                    0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (ru_arr[i3(i, k, j)] + ru_arr[i3(i + 1, k, j)])
                        + fzp_arr[int(k) - int(kms)] * (ru_arr[i3(i, k - 1, j)] + ru_arr[i3(i + 1, k - 1, j)])
                    )
                    * 0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (u_arr[i3(i, k, j)] + u_arr[i3(i + 1, k, j)])
                        + fzp_arr[int(k) - int(kms)] * (u_arr[i3(i, k - 1, j)] + u_arr[i3(i + 1, k - 1, j)])
                    )
                    + (msftx_arr[i2(i, j)] / msfty_arr[i2(i, j)])
                    * 0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (rv_arr[i3(i, k, j)] + rv_arr[i3(i, k, j + 1)])
                        + fzp_arr[int(k) - int(kms)] * (rv_arr[i3(i, k - 1, j)] + rv_arr[i3(i, k - 1, j + 1)])
                    )
                    * 0.5
                    * (
                        fzm_arr[int(k) - int(kms)] * (v_arr[i3(i, k, j)] + v_arr[i3(i, k, j + 1)])
                        + fzp_arr[int(k) - int(kms)] * (v_arr[i3(i, k - 1, j)] + v_arr[i3(i, k - 1, j + 1)])
                    )
                )
                rw_out = rw_out.at[i3(i, k, j)].set(value)

    return ru_out, rv_out, rw_out


def horizontal_pressure_gradient(
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
) -> tuple[Any, Any]:
    """Return WRF ``horizontal_pressure_gradient`` updated U/V tendencies."""

    del kds, jme, msftx, msfty

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

    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    c1 = c1h_arr

    def i2(i: int, j: int) -> tuple[int, int]:
        return int(i) - int(ims), int(j) - int(jms)

    def i3(i: int, k: int, j: int) -> tuple[int, int, int]:
        return int(i) - int(ims), int(k) - int(kms), int(j) - int(jms)

    itf = min(int(ite), int(ide) - 1)
    jtf = int(jte)
    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    j_start = int(jts)
    if (
        bool(_read(config_flags, "open_ys", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jts) == int(jds):
        j_start = int(jts) + 1
    if (
        bool(_read(config_flags, "open_ye", False))
        or specified
        or bool(_read(config_flags, "nested", False))
        or bool(_read(config_flags, "polar", False))
    ) and int(jte) == int(jde):
        jtf -= 1

    for j in range(j_start, jtf + 1):
        dpn = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1), dtype=p_arr.dtype)
        if non_hydrostatic:
            k = 1
            for i in range(i_start, itf + 1):
                dpn = dpn.at[int(i) - int(ims), int(k) - int(kms)].set(
                    0.5
                    * (
                        float(cf1) * (p_arr[i3(i, k, j - 1)] + p_arr[i3(i, k, j)])
                        + float(cf2) * (p_arr[i3(i, k + 1, j - 1)] + p_arr[i3(i, k + 1, j)])
                        + float(cf3) * (p_arr[i3(i, k + 2, j - 1)] + p_arr[i3(i, k + 2, j)])
                    )
                )
                dpn = dpn.at[int(i) - int(ims), int(kde) - int(kms)].set(0.0)
            if top_lid:
                for i in range(i_start, itf + 1):
                    dpn = dpn.at[int(i) - int(ims), int(kde) - int(kms)].set(
                        0.5
                        * (
                            float(cfn) * (p_arr[i3(i, int(kde) - 1, j - 1)] + p_arr[i3(i, int(kde) - 1, j)])
                            + float(cfn1) * (p_arr[i3(i, int(kde) - 2, j - 1)] + p_arr[i3(i, int(kde) - 2, j)])
                        )
                    )
            for k in range(2, ktf + 1):
                for i in range(i_start, itf + 1):
                    dpn = dpn.at[int(i) - int(ims), int(k) - int(kms)].set(
                        0.5
                        * (
                            fnm_arr[int(k) - int(kms)] * (p_arr[i3(i, k, j - 1)] + p_arr[i3(i, k, j)])
                            + fnp_arr[int(k) - int(kms)] * (p_arr[i3(i, k - 1, j - 1)] + p_arr[i3(i, k - 1, j)])
                        )
                    )
        for k in range(1, ktf + 1):
            for i in range(i_start, itf + 1):
                dpy = (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * 0.5 * float(rdy) * (c1h_arr[int(k) - int(kms)] * muv_arr[i2(i, j)] + c2h_arr[int(k) - int(kms)]) * (
                    (ph_arr[i3(i, k + 1, j)] - ph_arr[i3(i, k + 1, j - 1)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i, k, j - 1)])
                    + (alt_arr[i3(i, k, j)] + alt_arr[i3(i, k, j - 1)]) * (p_arr[i3(i, k, j)] - p_arr[i3(i, k, j - 1)])
                    + (al_arr[i3(i, k, j)] + al_arr[i3(i, k, j - 1)]) * (pb_arr[i3(i, k, j)] - pb_arr[i3(i, k, j - 1)])
                )
                if non_hydrostatic:
                    dpy = dpy + (msfvy_arr[i2(i, j)] / msfvx_arr[i2(i, j)]) * float(rdy) * (php_arr[i3(i, k, j)] - php_arr[i3(i, k, j - 1)]) * (
                        rdnw_arr[int(k) - int(kms)] * (dpn[int(i) - int(ims), int(k + 1) - int(kms)] - dpn[int(i) - int(ims), int(k) - int(kms)])
                        - 0.5 * ((c1[int(k) - int(kms)] * mu_arr[i2(i, j - 1)]) + (c1[int(k) - int(kms)] * mu_arr[i2(i, j)]))
                    )
                    rv_out = rv_out.at[i3(i, k, j)].set(rv_out[i3(i, k, j)] - cqv_arr[i3(i, k, j)] * dpy)
                else:
                    rv_out = rv_out.at[i3(i, k, j)].set(rv_out[i3(i, k, j)] - dpy)

    itf = int(ite)
    jtf = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    j_start = int(jts)
    if (bool(_read(config_flags, "open_xs", False)) or specified or bool(_read(config_flags, "nested", False))) and int(its) == int(ids):
        i_start = int(its) + 1
    if (bool(_read(config_flags, "open_xe", False)) or specified or bool(_read(config_flags, "nested", False))) and int(ite) == int(ide):
        itf -= 1
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        itf = int(ite)

    for j in range(j_start, jtf + 1):
        dpn = jnp.zeros((int(ime) - int(ims) + 1, int(kme) - int(kms) + 1), dtype=p_arr.dtype)
        if non_hydrostatic:
            k = 1
            for i in range(i_start, itf + 1):
                dpn = dpn.at[int(i) - int(ims), int(k) - int(kms)].set(
                    0.5
                    * (
                        float(cf1) * (p_arr[i3(i - 1, k, j)] + p_arr[i3(i, k, j)])
                        + float(cf2) * (p_arr[i3(i - 1, k + 1, j)] + p_arr[i3(i, k + 1, j)])
                        + float(cf3) * (p_arr[i3(i - 1, k + 2, j)] + p_arr[i3(i, k + 2, j)])
                    )
                )
                dpn = dpn.at[int(i) - int(ims), int(kde) - int(kms)].set(0.0)
            if top_lid:
                for i in range(i_start, itf + 1):
                    dpn = dpn.at[int(i) - int(ims), int(kde) - int(kms)].set(
                        0.5
                        * (
                            float(cfn) * (p_arr[i3(i - 1, int(kde) - 1, j)] + p_arr[i3(i, int(kde) - 1, j)])
                            + float(cfn1) * (p_arr[i3(i - 1, int(kde) - 2, j)] + p_arr[i3(i, int(kde) - 2, j)])
                        )
                    )
            for k in range(2, ktf + 1):
                for i in range(i_start, itf + 1):
                    dpn = dpn.at[int(i) - int(ims), int(k) - int(kms)].set(
                        0.5
                        * (
                            fnm_arr[int(k) - int(kms)] * (p_arr[i3(i - 1, k, j)] + p_arr[i3(i, k, j)])
                            + fnp_arr[int(k) - int(kms)] * (p_arr[i3(i - 1, k - 1, j)] + p_arr[i3(i, k - 1, j)])
                        )
                    )
        for k in range(1, ktf + 1):
            for i in range(i_start, itf + 1):
                dpx = (msfux_arr[i2(i, j)] / msfuy_arr[i2(i, j)]) * 0.5 * float(rdx) * (c1h_arr[int(k) - int(kms)] * muu_arr[i2(i, j)] + c2h_arr[int(k) - int(kms)]) * (
                    (ph_arr[i3(i, k + 1, j)] - ph_arr[i3(i - 1, k + 1, j)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i - 1, k, j)])
                    + (alt_arr[i3(i, k, j)] + alt_arr[i3(i - 1, k, j)]) * (p_arr[i3(i, k, j)] - p_arr[i3(i - 1, k, j)])
                    + (al_arr[i3(i, k, j)] + al_arr[i3(i - 1, k, j)]) * (pb_arr[i3(i, k, j)] - pb_arr[i3(i - 1, k, j)])
                )
                if non_hydrostatic:
                    dpx = dpx + (msfux_arr[i2(i, j)] / msfuy_arr[i2(i, j)]) * float(rdx) * (php_arr[i3(i, k, j)] - php_arr[i3(i - 1, k, j)]) * (
                        rdnw_arr[int(k) - int(kms)] * (dpn[int(i) - int(ims), int(k + 1) - int(kms)] - dpn[int(i) - int(ims), int(k) - int(kms)])
                        - 0.5 * ((c1[int(k) - int(kms)] * mu_arr[i2(i - 1, j)]) + (c1[int(k) - int(kms)] * mu_arr[i2(i, j)]))
                    )
                    ru_out = ru_out.at[i3(i, k, j)].set(ru_out[i3(i, k, j)] - cqu_arr[i3(i, k, j)] * dpx)
                else:
                    ru_out = ru_out.at[i3(i, k, j)].set(ru_out[i3(i, k, j)] - dpx)

    return ru_out, rv_out


def pg_buoy_w(
    rw_tend: Any,
    p: Any,
    cqw: Any,
    muf: Any,
    mubf: Any,
    c1f: Any,
    c2f: Any,
    rdnw: Any,
    rdn: Any,
    g: float,
    msftx: Any,
    msfty: Any,
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
) -> tuple[Any, Any]:
    """Return WRF ``pg_buoy_w`` updated ``rw_tend`` and ``cqw``."""

    del ids, jds, kds, kts, kte, msftx

    rw_out = jnp.asarray(rw_tend).copy()
    cqw_out = jnp.asarray(cqw).copy()
    p_arr = jnp.asarray(p)
    muf_arr = jnp.asarray(muf)
    mubf_arr = jnp.asarray(mubf)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)
    rdnw_arr = jnp.asarray(rdnw)
    rdn_arr = jnp.asarray(rdn)
    msfty_arr = jnp.asarray(msfty)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    i_fortran = jnp.arange(int(its), itf + 1)
    j_fortran = jnp.arange(int(jts), jtf + 1)
    if i_fortran.size == 0 or j_fortran.size == 0:
        return rw_out, cqw_out

    ii = _idx(i_fortran, ims)
    jj = _idx(j_fortran, jms)
    grid = (ii[:, None], jj[None, :])

    top_k = int(kde)
    top_kk = top_k - int(kms)
    lower_kk = top_k - 1 - int(kms)
    cq = cqw_out[ii[:, None], lower_kk, jj[None, :]]
    if rounding_token is None:
        cq1 = 1.0 / (1.0 + cq)
        cq2 = cq * cq1
        top_update = (1.0 / msfty_arr[grid]) * float(g) * (
            cq1 * 2.0 * rdnw_arr[lower_kk] * (-p_arr[ii[:, None], lower_kk, jj[None, :]])
            - (c1f_arr[top_kk] * muf_arr[grid])
            - cq2 * (c1f_arr[top_kk] * mubf_arr[grid] + c2f_arr[top_kk])
        )
        rw_out = rw_out.at[ii[:, None], top_kk, jj[None, :]].add(top_update)
    else:
        add = lambda left, right: _wrf_real_add(left, right, rounding_token=rounding_token)
        sub = lambda left, right: _wrf_real_sub(left, right, rounding_token=rounding_token)
        mul = lambda left, right: _wrf_real_mul(left, right, rounding_token=rounding_token)
        div = lambda left, right: _wrf_real_div(left, right, rounding_token=rounding_token)
        one = _wrf_real(1.0, rounding_token=rounding_token)
        cq1 = div(one, add(one, cq))
        cq2 = mul(cq, cq1)
        pressure_term = mul(mul(mul(cq1, 2.0), rdnw_arr[lower_kk]), -p_arr[ii[:, None], lower_kk, jj[None, :]])
        dry_mass = mul(c1f_arr[top_kk], muf_arr[grid])
        base_mass = add(mul(c1f_arr[top_kk], mubf_arr[grid]), c2f_arr[top_kk])
        moist_mass = mul(cq2, base_mass)
        bracket = sub(sub(pressure_term, dry_mass), moist_mass)
        scale = mul(div(one, msfty_arr[grid]), g)
        top_update = mul(scale, bracket)
        top_values = add(rw_out[ii[:, None], top_kk, jj[None, :]], top_update)
        rw_out = rw_out.at[ii[:, None], top_kk, jj[None, :]].set(top_values)

    mid_kk = jnp.arange(2 - int(kms), top_k - int(kms))
    if mid_kk.size:
        idx3 = (ii[:, None, None], mid_kk[None, :, None], jj[None, None, :])
        cq = cqw_out[idx3]
        if rounding_token is None:
            cq1 = 1.0 / (1.0 + cq)
            cq2 = cq * cq1
            update = (1.0 / msfty_arr[grid][:, None, :]) * float(g) * (
                cq1
                * rdn_arr[mid_kk][None, :, None]
                * (p_arr[idx3] - p_arr[ii[:, None, None], mid_kk[None, :, None] - 1, jj[None, None, :]])
                - (c1f_arr[mid_kk][None, :, None] * muf_arr[grid][:, None, :])
                - cq2 * (c1f_arr[mid_kk][None, :, None] * mubf_arr[grid][:, None, :] + c2f_arr[mid_kk][None, :, None])
            )
            rw_out = rw_out.at[idx3].add(update)
        else:
            cq1 = div(one, add(one, cq))
            cq2 = mul(cq, cq1)
            pressure_delta = sub(
                p_arr[idx3],
                p_arr[ii[:, None, None], mid_kk[None, :, None] - 1, jj[None, None, :]],
            )
            pressure_term = mul(mul(cq1, rdn_arr[mid_kk][None, :, None]), pressure_delta)
            dry_mass = mul(c1f_arr[mid_kk][None, :, None], muf_arr[grid][:, None, :])
            base_mass = add(
                mul(c1f_arr[mid_kk][None, :, None], mubf_arr[grid][:, None, :]),
                c2f_arr[mid_kk][None, :, None],
            )
            moist_mass = mul(cq2, base_mass)
            bracket = sub(sub(pressure_term, dry_mass), moist_mass)
            scale = mul(div(one, msfty_arr[grid][:, None, :]), g)
            update = mul(scale, bracket)
            values = add(rw_out[idx3], update)
            rw_out = rw_out.at[idx3].set(values)
        cqw_out = cqw_out.at[idx3].set(cq1)

    return rw_out, cqw_out


def _hydrostatic_pressure_column(carry, layer, parameter):
    pressure, mass = carry
    qtot, c1, c2, dnw = layer
    one, token = parameter
    dtype = pressure.dtype

    def binary(operation, left, right):
        left, right = jnp.asarray(left, dtype), jnp.asarray(right, dtype)
        if token is None:
            left, right = column_barrier(left), column_barrier(right)
        value = column_binary(operation, left, right)
        return column_barrier(value) if token is None else _store_wrf_real(value, token)

    wet = binary("add", one, qtot)
    dry = binary("add", binary("mul", c1, mass), c2)
    step = binary("mul", binary("mul", wet, dry), dnw)
    current = binary("sub", pressure, step).astype(dtype)
    return (current, mass), current


def phy_prep(
    config_flags: Any,
    mut: Any,
    muu: Any,
    muv: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    u: Any,
    v: Any,
    p: Any,
    pb: Any,
    alt: Any,
    ph: Any,
    phb: Any,
    t: Any,
    moist: Any,
    n_moist: int,
    dnw: Any,
    fzm: Any,
    fzp: Any,
    znw: Any,
    p_top: float,
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
    p_qv: int = 2,
    param_first_scalar: int = 2,
    r_v: float = 461.6,
    r_d: float = 287.0,
    t0: float = 300.0,
    p1000mb: float = 100000.0,
    rcp: float = 2.0 / 7.0,
    grav: float = 9.81,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Return WRF ``phy_prep`` diagnostic fields for physics preparation."""

    del jds, kds, c1f, c2f, muu, muv, znw

    mut_arr = jnp.asarray(mut)
    c1 = jnp.asarray(c1h)
    c2 = jnp.asarray(c2h)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    p_arr = jnp.asarray(p)
    pb_arr = jnp.asarray(pb)
    alt_arr = jnp.asarray(alt)
    ph_arr = jnp.asarray(ph)
    phb_arr = jnp.asarray(phb)
    t_arr = jnp.asarray(t)
    moist_arr = jnp.asarray(moist)
    dnw_arr = jnp.asarray(dnw)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)

    shape = (int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1)
    dtype = t_arr.dtype
    rho = jnp.zeros(shape, dtype=dtype)
    th_phy = jnp.zeros(shape, dtype=dtype)
    th_phy_m_t0 = jnp.zeros(shape, dtype=dtype)
    p_phy = jnp.zeros(shape, dtype=dtype)
    pi_phy = jnp.zeros(shape, dtype=dtype)
    u_phy = jnp.zeros(shape, dtype=dtype)
    v_phy = jnp.zeros(shape, dtype=dtype)
    p8w = jnp.zeros(shape, dtype=dtype)
    t_phy = jnp.zeros(shape, dtype=dtype)
    t8w = jnp.zeros(shape, dtype=dtype)
    z = jnp.zeros(shape, dtype=dtype)
    z_at_w = jnp.zeros(shape, dtype=dtype)
    dz8w = jnp.zeros(shape, dtype=dtype)
    p_hyd = jnp.zeros(shape, dtype=dtype)
    p_hyd_w = jnp.zeros(shape, dtype=dtype)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = min(int(kte), int(kde) - 1)

    if i_start > i_end or k_start > k_end or j_start > j_end:
        return rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w

    i_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    j_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    k_slice = slice(k_start - int(kms), k_end - int(kms) + 1)
    k_above = slice(k_start - int(kms) + 1, k_end - int(kms) + 2)
    idx3 = (i_slice, k_slice, j_slice)
    qv_idx = int(p_qv) - 1
    real = lambda value: _wrf_real(value, dtype, rounding_token)
    add = lambda left, right: _wrf_real_add(left, right, dtype, rounding_token)
    sub = lambda left, right: _wrf_real_sub(left, right, dtype, rounding_token)
    mul = lambda left, right: _wrf_real_mul(left, right, dtype, rounding_token)
    div = lambda left, right: _wrf_real_div(left, right, dtype, rounding_token)
    one = real(1.0)
    half = real(0.5)
    t0_real = real(t0)
    ratio = div(real(r_v), real(r_d))
    qv_values = moist_arr[idx3 + (qv_idx,)]
    if int(_read(config_flags, "use_theta_m", 0)) == 1 and int(p_qv) >= int(param_first_scalar):
        th_values = div(
            add(t_arr[idx3], t0_real),
            add(one, mul(ratio, qv_values)),
        )
    else:
        th_values = add(t_arr[idx3], t0_real)
    th_phy = th_phy.at[idx3].set(th_values)
    th_phy_m_t0 = th_phy_m_t0.at[idx3].set(sub(th_values, t0_real))
    p_values = add(p_arr[idx3], pb_arr[idx3])
    p_phy = p_phy.at[idx3].set(p_values)
    pressure_ratio = div(p_values, real(p1000mb))
    pi_values = real(
        _powf_for_wrf_patch_i_loops(
            pressure_ratio,
            real(rcp),
            ids=int(ids),
            ide=int(ide),
            i_start=i_start,
            i_end=i_end,
            nproc_x=int(_read(config_flags, "wrf_reference_nproc_x", 1)),
        )
    )
    pi_phy = pi_phy.at[idx3].set(pi_values)
    t_values = mul(th_values, pi_values)
    t_phy = t_phy.at[idx3].set(t_values)
    rho_values = mul(div(one, alt_arr[idx3]), add(one, qv_values))
    rho = rho.at[idx3].set(rho_values)
    u_phy = u_phy.at[idx3].set(
        mul(half, add(u_arr[idx3], u_arr[i_slice.start + 1:i_slice.stop + 1, k_slice, j_slice]))
    )
    v_phy = v_phy.at[idx3].set(
        mul(half, add(v_arr[idx3], v_arr[i_slice, k_slice, j_slice.start + 1:j_slice.stop + 1]))
    )

    idx3w = (i_slice, slice(k_start - int(kms), int(kte) - int(kms) + 1), j_slice)
    grav_real = real(grav)
    z_at_w_values = div(add(phb_arr[idx3w], ph_arr[idx3w]), grav_real)
    z_at_w = z_at_w.at[idx3w].set(z_at_w_values)

    if int(kte) > k_start:
        dz_slice = slice(k_start - int(kms), int(kte) - int(kms))
        idx3dz = (i_slice, dz_slice, j_slice)
        dz_values = sub(z_at_w[i_slice, dz_slice.start + 1:dz_slice.stop + 1, j_slice], z_at_w[idx3dz])
        dz8w = dz8w.at[idx3dz].set(dz_values)
    dz8w = dz8w.at[i_slice, int(kte) - int(kms), j_slice].set(0.0)

    z_values = mul(half, add(z_at_w[idx3], z_at_w[i_slice, k_above, j_slice]))
    z = z.at[idx3].set(z_values)

    if k_end >= 2:
        k_mid_fortran = jnp.arange(2, k_end + 1)
        kk_mid = _idx(k_mid_fortran, kms)
        idx3mid = (i_slice, slice(2 - int(kms), k_end - int(kms) + 1), j_slice)
        mid_below = (i_slice, slice(1 - int(kms), k_end - int(kms)), j_slice)
        p8w_values = add(
            mul(fzm_arr[kk_mid][None, :, None], p_phy[idx3mid]),
            mul(
                fzp_arr[kk_mid][None, :, None],
                p_phy[mid_below],
            ),
        )
        t8w_values = add(
            mul(fzm_arr[kk_mid][None, :, None], t_phy[idx3mid]),
            mul(
                fzp_arr[kk_mid][None, :, None],
                t_phy[mid_below],
            ),
        )
        p8w = p8w.at[idx3mid].set(p8w_values)
        t8w = t8w.at[idx3mid].set(t8w_values)

    kk_bottom = int(k_start) - int(kms)
    kk_second = kk_bottom + 1
    z0 = z_at_w[i_slice, kk_bottom, j_slice]
    z1 = z[i_slice, kk_bottom, j_slice]
    z2 = z[i_slice, kk_second, j_slice]
    w1 = div(sub(z0, z2), sub(z1, z2))
    w2 = sub(one, w1)
    p8w = p8w.at[i_slice, kk_bottom, j_slice].set(
        add(
            mul(w1, p_phy[i_slice, kk_bottom, j_slice]),
            mul(w2, p_phy[i_slice, kk_second, j_slice]),
        )
    )
    t8w = t8w.at[i_slice, kk_bottom, j_slice].set(
        add(
            mul(w1, t_phy[i_slice, kk_bottom, j_slice]),
            mul(w2, t_phy[i_slice, kk_second, j_slice]),
        )
    )

    kk_top = int(kde) - int(kms)
    kk_kend = int(k_end) - int(kms)
    kk_kendm1 = kk_kend - 1
    z0 = z_at_w[i_slice, int(kte) - int(kms), j_slice]
    z1 = z[i_slice, kk_kend, j_slice]
    z2 = z[i_slice, kk_kendm1, j_slice]
    w1 = div(sub(z0, z2), sub(z1, z2))
    w2 = sub(one, w1)
    p8w = p8w.at[i_slice, kk_top, j_slice].set(
        real(
            jnp.exp(
                add(
                    mul(w1, real(jnp.log(p_phy[i_slice, int(kde) - 1 - int(kms), j_slice]))),
                    mul(w2, real(jnp.log(p_phy[i_slice, int(kde) - 2 - int(kms), j_slice]))),
                )
            )
        )
    )
    t8w = t8w.at[i_slice, kk_top, j_slice].set(
        add(
            mul(w1, t_phy[i_slice, int(kde) - 1 - int(kms), j_slice]),
            mul(w2, t_phy[i_slice, int(kde) - 2 - int(kms), j_slice]),
        )
    )

    p_hyd_w = p_hyd_w.at[i_slice, int(kte) - int(kms), j_slice].set(real(p_top))
    scalar_start = int(param_first_scalar) - 1
    scalar_stop = int(n_moist)
    hydro_slice = slice(k_start - int(kms), int(kte) - int(kms))
    hydro_index = (i_slice, hydro_slice, j_slice)
    column_shape = (i_end - i_start + 1, j_end - j_start + 1)
    qtot = jnp.zeros((column_shape[0], int(kte) - k_start, column_shape[1]), dtype=dtype)
    for scalar_n in range(scalar_start, scalar_stop):
        qtot = add(qtot, moist_arr[hydro_index + (scalar_n,)])
    # Keep the descending WRF recurrence inside one shared CPU/GPU column loop.
    _, history = scan_columns(
        _hydrostatic_pressure_column,
        (jnp.full(column_shape, real(p_top), dtype=dtype), mut_arr[i_slice, j_slice]),
        (jnp.moveaxis(qtot, 1, 0), c1[hydro_slice], c2[hydro_slice], dnw_arr[hydro_slice]),
        (one, rounding_token),
        reverse=True,
    )
    p_hyd_w = p_hyd_w.at[hydro_index].set(jnp.moveaxis(history, 0, 1))

    p_hyd_values = mul(
        half,
        add(p_hyd_w[idx3], p_hyd_w[i_slice, k_above, j_slice]),
    )
    p_hyd = p_hyd.at[idx3].set(p_hyd_values)

    return rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w


def phy_prep_part2(
    config_flags: Any,
    mut: Any,
    muu: Any,
    muv: Any,
    c1h: Any,
    c2h: Any,
    c1f: Any,
    c2f: Any,
    rthraten: Any,
    rthblten: Any,
    rublten: Any,
    rvblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rucuten: Any,
    rvcuten: Any,
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqrcuten: Any,
    rqicuten: Any,
    rqscuten: Any,
    rushten: Any,
    rvshten: Any,
    rthshten: Any,
    rqvshten: Any,
    rqcshten: Any,
    rqrshten: Any,
    rqishten: Any,
    rqsshten: Any,
    rqgshten: Any,
    rthften: Any,
    rqvften: Any,
    rundgdten: Any,
    rvndgdten: Any,
    rthndgdten: Any,
    rphndgdten: Any,
    rqvndgdten: Any,
    rmundgdten: Any,
    t_new: Any,
    th_phy: Any,
    qv: Any,
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
    p_qv: int = 2,
    p_qc: int = 3,
    p_qr: int = 4,
    p_qi: int = 5,
    p_qs: int = 6,
    p_qg: int = 7,
    param_first_scalar: int = 2,
    gdscheme: int = 93,
    gfscheme: int = 3,
    g3scheme: int = 5,
    kfetascheme: int = 1,
    mskfscheme: int = 11,
    tiedtkescheme: int = 6,
    ntiedtkescheme: int = 16,
    r_v: float = 461.6,
    r_d: float = 287.0,
    t0: float = 300.0,
) -> tuple[Any, ...]:
    """Return WRF ``phy_prep_part2`` updated physics tendency arrays."""

    del kds, ime, jme, kme

    mut_arr = jnp.asarray(mut)
    muu_arr = jnp.asarray(muu)
    muv_arr = jnp.asarray(muv)
    c1h_arr = jnp.asarray(c1h)
    c2h_arr = jnp.asarray(c2h)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)
    t_new_arr = jnp.asarray(t_new)
    qv_arr = jnp.asarray(qv)

    rthraten_out = jnp.asarray(rthraten).copy()
    rthblten_out = jnp.asarray(rthblten).copy()
    rublten_out = jnp.asarray(rublten).copy()
    rvblten_out = jnp.asarray(rvblten).copy()
    rqvblten_out = jnp.asarray(rqvblten).copy()
    rqcblten_out = jnp.asarray(rqcblten).copy()
    rqiblten_out = jnp.asarray(rqiblten).copy()
    rucuten_out = jnp.asarray(rucuten).copy()
    rvcuten_out = jnp.asarray(rvcuten).copy()
    rthcuten_out = jnp.asarray(rthcuten).copy()
    rqvcuten_out = jnp.asarray(rqvcuten).copy()
    rqccuten_out = jnp.asarray(rqccuten).copy()
    rqrcuten_out = jnp.asarray(rqrcuten).copy()
    rqicuten_out = jnp.asarray(rqicuten).copy()
    rqscuten_out = jnp.asarray(rqscuten).copy()
    rushten_out = jnp.asarray(rushten).copy()
    rvshten_out = jnp.asarray(rvshten).copy()
    rthshten_out = jnp.asarray(rthshten).copy()
    rqvshten_out = jnp.asarray(rqvshten).copy()
    rqcshten_out = jnp.asarray(rqcshten).copy()
    rqrshten_out = jnp.asarray(rqrshten).copy()
    rqishten_out = jnp.asarray(rqishten).copy()
    rqsshten_out = jnp.asarray(rqsshten).copy()
    rqgshten_out = jnp.asarray(rqgshten).copy()
    rthften_out = jnp.asarray(rthften).copy()
    rqvften_out = jnp.asarray(rqvften).copy()
    rundgdten_out = jnp.asarray(rundgdten).copy()
    rvndgdten_out = jnp.asarray(rvndgdten).copy()
    rthndgdten_out = jnp.asarray(rthndgdten).copy()
    rphndgdten_out = jnp.asarray(rphndgdten).copy()
    rqvndgdten_out = jnp.asarray(rqvndgdten).copy()
    rmundgdten_out = jnp.asarray(rmundgdten).copy()
    th_phy_out = jnp.asarray(th_phy).copy()

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = min(int(kte), int(kde) - 1)

    if i_start > i_end or k_start > k_end or j_start > j_end:
        return (
            rthraten_out,
            rthblten_out,
            rublten_out,
            rvblten_out,
            rqvblten_out,
            rqcblten_out,
            rqiblten_out,
            rucuten_out,
            rvcuten_out,
            rthcuten_out,
            rqvcuten_out,
            rqccuten_out,
            rqrcuten_out,
            rqicuten_out,
            rqscuten_out,
            rushten_out,
            rvshten_out,
            rthshten_out,
            rqvshten_out,
            rqcshten_out,
            rqrshten_out,
            rqishten_out,
            rqsshten_out,
            rqgshten_out,
            rthften_out,
            rqvften_out,
            rundgdten_out,
            rvndgdten_out,
            rthndgdten_out,
            rphndgdten_out,
            rqvndgdten_out,
            rmundgdten_out,
            th_phy_out,
        )

    ii = slice(i_start - int(ims), i_end + 1 - int(ims))
    kk = slice(k_start - int(kms), k_end + 1 - int(kms))
    jj = slice(j_start - int(jms), j_end + 1 - int(jms))
    idx3 = (ii, kk, jj)
    denom = c1h_arr[kk][None, :, None] * mut_arr[ii, jj][:, None, :] + c2h_arr[kk][None, :, None]

    def divide_mass(field: Any) -> Any:
        return field.at[idx3].set(field[idx3] / denom)

    if int(_read(config_flags, "ra_lw_physics", 0)) > 0 or int(_read(config_flags, "ra_sw_physics", 0)) > 0:
        rthraten_out = divide_mass(rthraten_out)

    if int(_read(config_flags, "cu_physics", 0)) > 0:
        rucuten_out = divide_mass(rucuten_out)
        rvcuten_out = divide_mass(rvcuten_out)
        rthcuten_out = divide_mass(rthcuten_out)
        if int(p_qv) >= int(param_first_scalar):
            rqvcuten_out = divide_mass(rqvcuten_out)
        if int(p_qc) >= int(param_first_scalar):
            rqccuten_out = divide_mass(rqccuten_out)
        if int(p_qr) >= int(param_first_scalar):
            rqrcuten_out = divide_mass(rqrcuten_out)
        if int(p_qi) >= int(param_first_scalar):
            rqicuten_out = divide_mass(rqicuten_out)
        if int(p_qs) >= int(param_first_scalar):
            rqscuten_out = divide_mass(rqscuten_out)

    if int(_read(config_flags, "shcu_physics", 0)) > 0:
        rushten_out = divide_mass(rushten_out)
        rvshten_out = divide_mass(rvshten_out)
        rthshten_out = divide_mass(rthshten_out)
        if int(p_qv) >= int(param_first_scalar):
            rqvshten_out = divide_mass(rqvshten_out)
        if int(p_qc) >= int(param_first_scalar):
            rqcshten_out = divide_mass(rqcshten_out)
        if int(p_qr) >= int(param_first_scalar):
            rqrshten_out = divide_mass(rqrshten_out)
        if int(p_qi) >= int(param_first_scalar):
            rqishten_out = divide_mass(rqishten_out)
        if int(p_qs) >= int(param_first_scalar):
            rqsshten_out = divide_mass(rqsshten_out)
        if int(p_qg) >= int(param_first_scalar):
            rqgshten_out = divide_mass(rqgshten_out)

    if int(_read(config_flags, "bl_pbl_physics", 0)) > 0:
        rublten_out = divide_mass(rublten_out)
        rvblten_out = divide_mass(rvblten_out)
        rthblten_out = divide_mass(rthblten_out)
        if int(p_qv) >= int(param_first_scalar):
            rqvblten_out = divide_mass(rqvblten_out)
        if int(p_qc) >= int(param_first_scalar):
            rqcblten_out = divide_mass(rqcblten_out)
        if int(p_qi) >= int(param_first_scalar):
            rqiblten_out = divide_mass(rqiblten_out)

    cu_physics = int(_read(config_flags, "cu_physics", 0))
    cu_qv_schemes = {int(gdscheme), int(gfscheme), int(g3scheme), int(kfetascheme), int(mskfscheme), int(tiedtkescheme), int(ntiedtkescheme)}
    cu_theta_schemes = {int(gdscheme), int(gfscheme), int(g3scheme), int(ntiedtkescheme)}
    if cu_physics in cu_qv_schemes and int(p_qv) >= int(param_first_scalar):
        rqvften_out = divide_mass(rqvften_out)

    if cu_physics in cu_theta_schemes:
        rthften_out = divide_mass(rthften_out)
        if int(_read(config_flags, "use_theta_m", 0)) == 1:
            ratio = float(r_v) / float(r_d)
            theta = (t_new_arr[idx3] + float(t0)) / (1.0 + ratio * qv_arr[idx3])
            th_phy_out = th_phy_out.at[idx3].set(theta)
            corrected = theta / (t_new_arr[idx3] + float(t0)) * (rthften_out[idx3] - ratio * theta * rqvften_out[idx3])
            rthften_out = rthften_out.at[idx3].set(corrected)

    if int(_read(config_flags, "grid_fdda", 0)) > 0:
        i_startu = max(int(its), int(ids) + 1)
        j_startv = max(int(jts), int(jds) + 1)
        if i_startu <= i_end:
            iiu = slice(i_startu - int(ims), i_end + 1 - int(ims))
            idx3u = (iiu, kk, jj)
            denom_u = c1h_arr[kk][None, :, None] * muu_arr[iiu, jj][:, None, :] + c2h_arr[kk][None, :, None]
            rundgdten_out = rundgdten_out.at[idx3u].set(rundgdten_out[idx3u] / denom_u)
        if j_startv <= j_end:
            jjv = slice(j_startv - int(jms), j_end + 1 - int(jms))
            idx3v = (ii, kk, jjv)
            denom_v = c1h_arr[kk][None, :, None] * muv_arr[ii, jjv][:, None, :] + c2h_arr[kk][None, :, None]
            rvndgdten_out = rvndgdten_out.at[idx3v].set(rvndgdten_out[idx3v] / denom_v)
        rthndgdten_out = divide_mass(rthndgdten_out)

        if int(_read(config_flags, "grid_fdda", 0)) == 2:
            kkf = slice(k_start - int(kms), int(kte) + 1 - int(kms))
            idx3f = (ii, kkf, jj)
            denom_f = c1f_arr[kkf][None, :, None] * mut_arr[ii, jj][:, None, :] + c2f_arr[kkf][None, :, None]
            rphndgdten_out = rphndgdten_out.at[idx3f].set(rphndgdten_out[idx3f] / denom_f)
        elif int(_read(config_flags, "grid_fdda", 0)) >= 1 and int(p_qv) >= int(param_first_scalar):
            rqvndgdten_out = divide_mass(rqvndgdten_out)

    return (
        rthraten_out,
        rthblten_out,
        rublten_out,
        rvblten_out,
        rqvblten_out,
        rqcblten_out,
        rqiblten_out,
        rucuten_out,
        rvcuten_out,
        rthcuten_out,
        rqvcuten_out,
        rqccuten_out,
        rqrcuten_out,
        rqicuten_out,
        rqscuten_out,
        rushten_out,
        rvshten_out,
        rthshten_out,
        rqvshten_out,
        rqcshten_out,
        rqrshten_out,
        rqishten_out,
        rqsshten_out,
        rqgshten_out,
        rthften_out,
        rqvften_out,
        rundgdten_out,
        rvndgdten_out,
        rthndgdten_out,
        rphndgdten_out,
        rqvndgdten_out,
        rmundgdten_out,
        th_phy_out,
    )


def rhs_ph(
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
    grav: float = 9.81,
) -> Any:
    """Return WRF ``rhs_ph`` updated geopotential tendency."""

    del kds, ime, jme, kme, msfuy, msfvx, msfvx_inv, msftx

    out = jnp.asarray(ph_tend).copy()
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    ww_arr = jnp.asarray(ww)
    ph_arr = jnp.asarray(ph)
    ph_old_arr = jnp.asarray(ph_old)
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

    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
    advective_order = int(_read(config_flags, "h_sca_adv_order", 2))
    phi_adv_z = int(_read(config_flags, "phi_adv_z", 1))
    open_xs = bool(_read(config_flags, "open_xs", False))
    open_xe = bool(_read(config_flags, "open_xe", False))
    open_ys = bool(_read(config_flags, "open_ys", False))
    open_ye = bool(_read(config_flags, "open_ye", False))

    def i2(i: int, j: int) -> tuple[int, int]:
        return int(i) - int(ims), int(j) - int(jms)

    def i3(i: int, k: int, j: int) -> tuple[int, int, int]:
        return int(i) - int(ims), int(k) - int(kms), int(j) - int(jms)

    def k1(k: int) -> int:
        return int(k) - int(kms)

    def mass(k: int, i: int, j: int) -> Any:
        return c1f_arr[k1(k)] * mut_arr[i2(i, j)] + c2f_arr[k1(k)]

    def y_second(j: int, i_start: int, itf_local: int) -> None:
        nonlocal out
        for k in range(2, int(kte)):
            for i in range(int(i_start), int(itf_local) + 1):
                term = (
                    (c1f_arr[k1(k)] * muvf_arr[i2(i, j + 1)] + c2f_arr[k1(k)])
                    * (v_arr[i3(i, k, j + 1)] + v_arr[i3(i, k - 1, j + 1)])
                    * msfvy_arr[i2(i, j + 1)]
                    * (phb_arr[i3(i, k, j + 1)] - phb_arr[i3(i, k, j)] + ph_arr[i3(i, k, j + 1)] - ph_arr[i3(i, k, j)])
                    + (c1f_arr[k1(k)] * muvf_arr[i2(i, j)] + c2f_arr[k1(k)])
                    * (v_arr[i3(i, k, j)] + v_arr[i3(i, k - 1, j)])
                    * msfvy_arr[i2(i, j)]
                    * (phb_arr[i3(i, k, j)] - phb_arr[i3(i, k, j - 1)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i, k, j - 1)])
                )
                out = out.at[i3(i, k, j)].add(-(0.25 * float(rdy) / msfty_arr[i2(i, j)]) * term)
        k = int(kte)
        for i in range(int(i_start), int(itf_local) + 1):
            term = (
                (c1f_arr[k1(k)] * muvf_arr[i2(i, j + 1)] + c2f_arr[k1(k)])
                * (float(cfn) * v_arr[i3(i, k - 1, j + 1)] + float(cfn1) * v_arr[i3(i, k - 2, j + 1)])
                * msfvy_arr[i2(i, j + 1)]
                * (phb_arr[i3(i, k, j + 1)] - phb_arr[i3(i, k, j)] + ph_arr[i3(i, k, j + 1)] - ph_arr[i3(i, k, j)])
                + (c1f_arr[k1(k)] * muvf_arr[i2(i, j)] + c2f_arr[k1(k)])
                * (float(cfn) * v_arr[i3(i, k - 1, j)] + float(cfn1) * v_arr[i3(i, k - 2, j)])
                * msfvy_arr[i2(i, j)]
                * (phb_arr[i3(i, k, j)] - phb_arr[i3(i, k, j - 1)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i, k, j - 1)])
            )
            out = out.at[i3(i, k, j)].add(-(0.5 * float(rdy) / msfty_arr[i2(i, j)]) * term)

    def x_second(i: int, j_start_local: int, jtf_local: int) -> None:
        nonlocal out
        for j in range(int(j_start_local), int(jtf_local) + 1):
            for k in range(2, int(kte)):
                term = (
                    (c1f_arr[k1(k)] * muuf_arr[i2(i + 1, j)] + c2f_arr[k1(k)])
                    * (u_arr[i3(i + 1, k, j)] + u_arr[i3(i + 1, k - 1, j)])
                    * msfux_arr[i2(i + 1, j)]
                    * (phb_arr[i3(i + 1, k, j)] - phb_arr[i3(i, k, j)] + ph_arr[i3(i + 1, k, j)] - ph_arr[i3(i, k, j)])
                    + (c1f_arr[k1(k)] * muuf_arr[i2(i, j)] + c2f_arr[k1(k)])
                    * (u_arr[i3(i, k, j)] + u_arr[i3(i, k - 1, j)])
                    * msfux_arr[i2(i, j)]
                    * (phb_arr[i3(i, k, j)] - phb_arr[i3(i - 1, k, j)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i - 1, k, j)])
                )
                out = out.at[i3(i, k, j)].add(-(0.25 * float(rdx) / msfty_arr[i2(i, j)]) * term)
            k = int(kte)
            term = (
                (c1f_arr[k1(k)] * muuf_arr[i2(i + 1, j)] + c2f_arr[k1(k)])
                * (float(cfn) * u_arr[i3(i + 1, k - 1, j)] + float(cfn1) * u_arr[i3(i + 1, k - 2, j)])
                * msfux_arr[i2(i + 1, j)]
                * (phb_arr[i3(i + 1, k, j)] - phb_arr[i3(i, k, j)] + ph_arr[i3(i + 1, k, j)] - ph_arr[i3(i, k, j)])
                + (c1f_arr[k1(k)] * muuf_arr[i2(i, j)] + c2f_arr[k1(k)])
                * (float(cfn) * u_arr[i3(i, k - 1, j)] + float(cfn1) * u_arr[i3(i, k - 2, j)])
                * msfux_arr[i2(i, j)]
                * (phb_arr[i3(i, k, j)] - phb_arr[i3(i - 1, k, j)] + ph_arr[i3(i, k, j)] - ph_arr[i3(i - 1, k, j)])
            )
            out = out.at[i3(i, k, j)].add(-(0.5 * float(rdx) / msfty_arr[i2(i, j)]) * term)

    def y_high(j: int, i_start: int, itf_local: int, coeff: tuple[float, float, float], scale: float) -> None:
        nonlocal out
        c_a, c_b, c_c = coeff
        max_offset = 3 if c_c != 0.0 else 2
        for k in range(2, int(kte)):
            for i in range(int(i_start), int(itf_local) + 1):
                grad = (
                    c_a * (ph_arr[i3(i, k, j + 1)] - ph_arr[i3(i, k, j - 1)])
                    + c_b * (ph_arr[i3(i, k, j + 2)] - ph_arr[i3(i, k, j - 2)])
                    + c_c * (ph_arr[i3(i, k, j + max_offset)] - ph_arr[i3(i, k, j - max_offset)])
                    + c_a * (phb_arr[i3(i, k, j + 1)] - phb_arr[i3(i, k, j - 1)])
                    + c_b * (phb_arr[i3(i, k, j + 2)] - phb_arr[i3(i, k, j - 2)])
                    + c_c * (phb_arr[i3(i, k, j + max_offset)] - phb_arr[i3(i, k, j - max_offset)])
                )
                vel = (
                    (c1f_arr[k1(k)] * muvf_arr[i2(i, j + 1)] + c2f_arr[k1(k)])
                    * (v_arr[i3(i, k, j + 1)] + v_arr[i3(i, k - 1, j + 1)])
                    * msfvy_arr[i2(i, j + 1)]
                    + (c1f_arr[k1(k)] * muvf_arr[i2(i, j)] + c2f_arr[k1(k)])
                    * (v_arr[i3(i, k, j)] + v_arr[i3(i, k - 1, j)])
                    * msfvy_arr[i2(i, j)]
                )
                out = out.at[i3(i, k, j)].add(-(0.25 * float(rdy) / msfty_arr[i2(i, j)]) * vel * scale * grad)
        k = int(kte)
        for i in range(int(i_start), int(itf_local) + 1):
            grad = (
                c_a * (ph_arr[i3(i, k, j + 1)] - ph_arr[i3(i, k, j - 1)])
                + c_b * (ph_arr[i3(i, k, j + 2)] - ph_arr[i3(i, k, j - 2)])
                + c_c * (ph_arr[i3(i, k, j + max_offset)] - ph_arr[i3(i, k, j - max_offset)])
                + c_a * (phb_arr[i3(i, k, j + 1)] - phb_arr[i3(i, k, j - 1)])
                + c_b * (phb_arr[i3(i, k, j + 2)] - phb_arr[i3(i, k, j - 2)])
                + c_c * (phb_arr[i3(i, k, j + max_offset)] - phb_arr[i3(i, k, j - max_offset)])
            )
            vel = (
                (c1f_arr[k1(k)] * muvf_arr[i2(i, j + 1)] + c2f_arr[k1(k)])
                * (float(cfn) * v_arr[i3(i, k - 1, j + 1)] + float(cfn1) * v_arr[i3(i, k - 2, j + 1)])
                * msfvy_arr[i2(i, j + 1)]
                + (c1f_arr[k1(k)] * muvf_arr[i2(i, j)] + c2f_arr[k1(k)])
                * (float(cfn) * v_arr[i3(i, k - 1, j)] + float(cfn1) * v_arr[i3(i, k - 2, j)])
                * msfvy_arr[i2(i, j)]
            )
            out = out.at[i3(i, k, j)].add(-(0.5 * float(rdy) / msfty_arr[i2(i, j)]) * vel * scale * grad)

    def x_high(i: int, j_start_local: int, jtf_local: int, coeff: tuple[float, float, float], scale: float) -> None:
        nonlocal out
        c_a, c_b, c_c = coeff
        max_offset = 3 if c_c != 0.0 else 2
        for j in range(int(j_start_local), int(jtf_local) + 1):
            for k in range(2, int(kte)):
                grad = (
                    c_a * (ph_arr[i3(i + 1, k, j)] - ph_arr[i3(i - 1, k, j)])
                    + c_b * (ph_arr[i3(i + 2, k, j)] - ph_arr[i3(i - 2, k, j)])
                    + c_c * (ph_arr[i3(i + max_offset, k, j)] - ph_arr[i3(i - max_offset, k, j)])
                    + c_a * (phb_arr[i3(i + 1, k, j)] - phb_arr[i3(i - 1, k, j)])
                    + c_b * (phb_arr[i3(i + 2, k, j)] - phb_arr[i3(i - 2, k, j)])
                    + c_c * (phb_arr[i3(i + max_offset, k, j)] - phb_arr[i3(i - max_offset, k, j)])
                )
                vel = (
                    (c1f_arr[k1(k)] * muuf_arr[i2(i + 1, j)] + c2f_arr[k1(k)])
                    * (u_arr[i3(i + 1, k, j)] + u_arr[i3(i + 1, k - 1, j)])
                    * msfux_arr[i2(i + 1, j)]
                    + (c1f_arr[k1(k)] * muuf_arr[i2(i, j)] + c2f_arr[k1(k)])
                    * (u_arr[i3(i, k, j)] + u_arr[i3(i, k - 1, j)])
                    * msfux_arr[i2(i, j)]
                )
                out = out.at[i3(i, k, j)].add(-(0.25 * float(rdx) / msfty_arr[i2(i, j)]) * vel * scale * grad)
            k = int(kte)
            grad = (
                c_a * (ph_arr[i3(i + 1, k, j)] - ph_arr[i3(i - 1, k, j)])
                + c_b * (ph_arr[i3(i + 2, k, j)] - ph_arr[i3(i - 2, k, j)])
                + c_c * (ph_arr[i3(i + max_offset, k, j)] - ph_arr[i3(i - max_offset, k, j)])
                + c_a * (phb_arr[i3(i + 1, k, j)] - phb_arr[i3(i - 1, k, j)])
                + c_b * (phb_arr[i3(i + 2, k, j)] - phb_arr[i3(i - 2, k, j)])
                + c_c * (phb_arr[i3(i + max_offset, k, j)] - phb_arr[i3(i - max_offset, k, j)])
            )
            vel = (
                (c1f_arr[k1(k)] * muuf_arr[i2(i + 1, j)] + c2f_arr[k1(k)])
                * (float(cfn) * u_arr[i3(i + 1, k - 1, j)] + float(cfn1) * u_arr[i3(i + 1, k - 2, j)])
                * msfux_arr[i2(i + 1, j)]
                + (c1f_arr[k1(k)] * muuf_arr[i2(i, j)] + c2f_arr[k1(k)])
                * (float(cfn) * u_arr[i3(i, k - 1, j)] + float(cfn1) * u_arr[i3(i, k - 2, j)])
                * msfux_arr[i2(i, j)]
            )
            out = out.at[i3(i, k, j)].add(-(0.5 * float(rdx) / msfty_arr[i2(i, j)]) * vel * scale * grad)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    for j in range(int(jts), jtf + 1):
        wdwn: dict[tuple[int, int], Any] = {}
        if phi_adv_z == 2:
            for k in range(2, int(kte) + 1):
                for i in range(int(its), itf + 1):
                    wdwn[(i, k)] = rdnw_arr[k1(k - 1)] * (ph_arr[i3(i, k, j)] - ph_arr[i3(i, k - 1, j)] + phb_arr[i3(i, k, j)] - phb_arr[i3(i, k - 1, j)])
            for k in range(2, int(kte)):
                for i in range(int(its), itf + 1):
                    out = out.at[i3(i, k, j)].add(-ww_arr[i3(i, k, j)] * (fnm_arr[k1(k)] * wdwn[(i, k + 1)] + fnp_arr[k1(k)] * wdwn[(i, k)]))
        else:
            for k in range(2, int(kte) + 1):
                for i in range(int(its), itf + 1):
                    wdwn[(i, k)] = (
                        0.5
                        * (ww_arr[i3(i, k, j)] + ww_arr[i3(i, k - 1, j)])
                        * rdnw_arr[k1(k - 1)]
                        * (ph_arr[i3(i, k, j)] - ph_arr[i3(i, k - 1, j)] + phb_arr[i3(i, k, j)] - phb_arr[i3(i, k - 1, j)])
                    )
            for k in range(2, int(kte)):
                for i in range(int(its), itf + 1):
                    out = out.at[i3(i, k, j)].add(-(fnm_arr[k1(k)] * wdwn[(i, k + 1)] + fnp_arr[k1(k)] * wdwn[(i, k)]))

    if non_hydrostatic:
        for j in range(int(jts), jtf + 1):
            for i in range(int(its), itf + 1):
                out = out.at[i3(i, int(kde), j)].set(0.0)
            for k in range(2, int(kte) + 1):
                for i in range(int(its), itf + 1):
                    out = out.at[i3(i, k, j)].add(mass(k, i, j) * float(grav) * w_arr[i3(i, k, j)] / msfty_arr[i2(i, j)])

    if advective_order <= 2:
        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if (open_ys or specified) and int(jts) == int(jds):
            j_start = int(jts) + 1
        if (open_ye or specified) and int(jte) == int(jde):
            jtf = int(jte) - 2
        for j in range(j_start, jtf + 1):
            y_second(j, i_start, itf)

        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if (open_xs or specified) and int(its) == int(ids):
            i_start = int(its) + 1
        if (open_xe or specified) and int(ite) == int(ide):
            itf = int(ite) - 2
        for i in range(i_start, itf + 1):
            x_second(i, j_start, jtf)

    elif advective_order <= 4:
        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if (open_ys or specified) and int(jts) == int(jds):
            j_start = int(jts) + 2
        if (open_ye or specified) and int(jte) == int(jde):
            jtf = int(jte) - 3
        for j in range(j_start, jtf + 1):
            y_high(j, i_start, itf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if (open_ys or specified) and int(jts) <= int(jds) + 1:
            y_second(int(jds) + 1, i_start, itf)
        if (open_ye or specified) and int(jte) >= int(jde) - 2:
            y_second(int(jde) - 2, i_start, itf)

        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if open_xs and int(its) == int(ids):
            i_start = int(its) + 2
        if open_xe and int(ite) == int(ide):
            itf = int(ite) - 3
        for i in range(i_start, itf + 1):
            x_high(i, j_start, jtf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if (open_xs or specified) and int(its) <= int(ids) + 1:
            x_second(int(ids) + 1, j_start, jtf)
        if (open_xe or specified) and int(ite) >= int(ide) - 2:
            x_second(int(ide) - 2, j_start, jtf)

    elif advective_order <= 6:
        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if open_ys or specified:
            j_start = max(int(jts), int(jds) + 3)
        if open_ye or specified:
            jtf = min(jtf, int(jde) - 4)
        for j in range(j_start, jtf + 1):
            y_high(j, i_start, itf, (45.0, -9.0, 1.0), 1.0 / 60.0)
        if (open_ys or specified) and int(jts) <= int(jds) + 2 <= int(jte):
            y_high(int(jds) + 2, i_start, itf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if (open_ye or specified) and int(jts) <= int(jde) - 3 <= int(jte):
            y_high(int(jde) - 3, i_start, itf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if (open_ys or specified) and int(jts) <= int(jds) + 1 <= int(jte):
            y_second(int(jds) + 1, i_start, itf)
        if (open_ye or specified) and int(jts) <= int(jde) - 2 <= int(jte):
            y_second(int(jde) - 2, i_start, itf)

        i_start = int(its)
        j_start = int(jts)
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        if open_xs or specified:
            i_start = max(int(its), int(ids) + 3)
        if open_xe or specified:
            itf = min(itf, int(ide) - 4)
        for i in range(i_start, itf + 1):
            x_high(i, j_start, jtf, (45.0, -9.0, 1.0), 1.0 / 60.0)
        if open_xs and int(its) <= int(ids) + 2 <= int(ite):
            x_high(int(ids) + 2, j_start, jtf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if open_xe and int(its) <= int(ide) - 3 <= int(ite):
            x_high(int(ide) - 3, j_start, jtf, (8.0, -1.0, 0.0), 1.0 / 12.0)
        if (open_xs or specified) and int(its) <= int(ids) + 1 <= int(ite):
            x_second(int(ids) + 1, j_start, jtf)
        if (open_xe or specified) and int(its) <= int(ide) - 2 <= int(ite):
            x_second(int(ide) - 2, j_start, jtf)

    i_start = int(its)
    itf = min(int(ite), int(ide) - 1)
    if open_ys and int(jts) == int(jds):
        j = int(jts)
        for k in range(2, int(kde) + 1):
            kz = min(int(k), int(kde) - 1)
            for i in range(int(its), itf + 1):
                vb = 0.5 * (fnm_arr[k1(kz)] * (v_arr[i3(i, kz, j + 1)] + v_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (v_arr[i3(i, kz - 1, j + 1)] + v_arr[i3(i, kz - 1, j)]))
                vl = jnp.minimum(vb, 0.0)
                out = out.at[i3(i, k, j)].add(-float(rdy) * mass(k, i, j) * (vl * (ph_old_arr[i3(i, k, j + 1)] - ph_old_arr[i3(i, k, j)])))
    if open_ye and int(jte) == int(jde):
        j = int(jte) - 1
        for k in range(2, int(kde) + 1):
            kz = min(int(k), int(kde) - 1)
            for i in range(int(its), itf + 1):
                vb = 0.5 * (fnm_arr[k1(kz)] * (v_arr[i3(i, kz, j + 1)] + v_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (v_arr[i3(i, kz - 1, j + 1)] + v_arr[i3(i, kz - 1, j)]))
                vr = jnp.maximum(vb, 0.0)
                out = out.at[i3(i, k, j)].add(-float(rdy) * mass(k, i, j) * (vr * (ph_old_arr[i3(i, k, j)] - ph_old_arr[i3(i, k, j - 1)])))

    jtf = min(int(jte), int(jde) - 1)
    if open_xs and int(its) == int(ids):
        i = int(its)
        for j in range(int(jts), jtf + 1):
            for k in range(2, int(kde)):
                kz = int(k)
                ub = 0.5 * (fnm_arr[k1(kz)] * (u_arr[i3(i + 1, kz, j)] + u_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (u_arr[i3(i + 1, kz - 1, j)] + u_arr[i3(i, kz - 1, j)]))
                ul = jnp.minimum(ub, 0.0)
                out = out.at[i3(i, k, j)].add(-(msfux_arr[i2(i, j)] / msfty_arr[i2(i, j)]) * float(rdx) * mass(k, i, j) * (ul * (ph_old_arr[i3(i + 1, k, j)] - ph_old_arr[i3(i, k, j)])))
            k = int(kde)
            kz = int(k)
            ub = 0.5 * (fnm_arr[k1(kz)] * (u_arr[i3(i + 1, kz, j)] + u_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (u_arr[i3(i + 1, kz - 1, j)] + u_arr[i3(i, kz - 1, j)]))
            ul = jnp.minimum(ub, 0.0)
            out = out.at[i3(i, k, j)].add(-(msfux_arr[i2(i, j)] / msfty_arr[i2(i, j)]) * float(rdx) * mass(k, i, j) * (ul * (ph_old_arr[i3(i + 1, k, j)] - ph_old_arr[i3(i, k, j)])))
    if open_xe and int(ite) == int(ide):
        i = int(ite) - 1
        for j in range(int(jts), jtf + 1):
            for k in range(2, int(kde)):
                kz = int(k)
                ub = 0.5 * (fnm_arr[k1(kz)] * (u_arr[i3(i + 1, kz, j)] + u_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (u_arr[i3(i + 1, kz - 1, j)] + u_arr[i3(i, kz - 1, j)]))
                ur = jnp.maximum(ub, 0.0)
                out = out.at[i3(i, k, j)].add(-(msfux_arr[i2(i, j)] / msfty_arr[i2(i, j)]) * float(rdx) * mass(k, i, j) * (ur * (ph_old_arr[i3(i, k, j)] - ph_old_arr[i3(i - 1, k, j)])))
            k = int(kde)
            kz = int(k) - 1
            ub = 0.5 * (fnm_arr[k1(kz)] * (u_arr[i3(i + 1, kz, j)] + u_arr[i3(i, kz, j)]) + fnp_arr[k1(kz)] * (u_arr[i3(i + 1, kz - 1, j)] + u_arr[i3(i, kz - 1, j)]))
            ur = jnp.maximum(ub, 0.0)
            out = out.at[i3(i, k, j)].add(-(msfux_arr[i2(i, j)] / msfty_arr[i2(i, j)]) * float(rdx) * mass(k, i, j) * (ur * (ph_old_arr[i3(i, k, j)] - ph_old_arr[i3(i - 1, k, j)])))

    return out


def zero_tend(
    tendency: Any,
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
) -> Any:
    """Return WRF ``zero_tend`` with the active 3-D tile set to zero."""

    del ids, ide, jds, jde, kds, kde

    out = jnp.asarray(tendency).copy()
    i0 = _idx(int(its), ims)
    i1 = _idx(int(ite), ims)
    k0 = _idx(int(kts), kms)
    k1 = _idx(int(kte), kms)
    j0 = _idx(int(jts), jms)
    j1 = _idx(int(jte), jms)
    if i1 < i0 or k1 < k0 or j1 < j0:
        return out
    if i0 == 0 and k0 == 0 and j0 == 0 and i1 == out.shape[0] - 1 and k1 == out.shape[1] - 1 and j1 == out.shape[2] - 1:
        return jnp.zeros_like(out)
    return out.at[i0 : i1 + 1, k0 : k1 + 1, j0 : j1 + 1].set(0.0)


def zero_tend2d(
    tendency: Any,
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
) -> Any:
    """Return WRF ``zero_tend2d`` with the active 2-D tile set to zero."""

    del ids, ide, jds, jde, kds, kde, kms, kme, kts, kte

    out = jnp.asarray(tendency).copy()
    i0 = _idx(int(its), ims)
    i1 = _idx(int(ite), ims)
    j0 = _idx(int(jts), jms)
    j1 = _idx(int(jte), jms)
    if i1 < i0 or j1 < j0:
        return out
    if i0 == 0 and j0 == 0 and i1 == out.shape[0] - 1 and j1 == out.shape[1] - 1:
        return jnp.zeros_like(out)
    return out.at[i0 : i1 + 1, j0 : j1 + 1].set(0.0)


def _calc_cq_points(indices, parameters, *, shape, bounds, dtype, species):
    """Evaluate all staggered moisture coefficients with ordered REAL stores."""
    i, k, j = indices
    moist, token = parameters
    nx, nk, ny, ns = shape
    its, ite, jts, jte, kts, ktf, ide, jde, ims, jms, kms = bounds
    fi, fk, fj = i + ims, k + kms, j + jms
    active = (fi >= its) & (fi <= ite) & (fj >= jts) & (fj <= jte)
    active &= (fk >= kts) & (fk <= ktf)
    masks = (
        active & (fj < jde),
        active & (fi < ide),
        active & (fi < ide) & (fj < jde) & (fk > kts),
    )

    def canonical(value, extent):
        return jnp.clip(jnp.where(value < 0, value + extent, value), 0, extent - 1)

    def read(di, dk, dj, species_index):
        # Preserve JAX's existing per-axis negative-index and clipping rules.
        s = species_index if species_index >= 0 else species_index + ns
        offset = (
            (canonical(i + di, nx) * nk + canonical(k + dk, nk)) * ny
            + canonical(j + dj, ny)
        ) * ns + min(max(s, 0), ns - 1)
        return moist[offset]

    def binary(operation, left, right):
        left, right = jnp.asarray(left, dtype), jnp.asarray(right, dtype)
        if token is None:
            left, right = column_barrier(left), column_barrier(right)
        value = column_binary(operation, left, right)
        return column_barrier(value) if token is None else real_assignment(value, token)

    output_shape = jnp.broadcast_shapes(i.shape, k.shape, j.shape)
    sums = tuple(jnp.zeros(output_shape, dtype) for _ in range(3))
    if species:
        for s in species:
            here = read(0, 0, 0, s)
            sums = tuple(
                binary("add", binary("add", value, here), read(*offset, s))
                for value, offset in zip(
                    sums, ((-1, 0, 0), (0, 0, -1), (0, -1, 0)), strict=True
                )
            )
        values = tuple(binary("mul", 0.5, value) for value in sums)
        values = tuple(
            binary("div", 1.0, binary("add", 1.0, value)) for value in values[:2]
        ) + (values[2],)
    else:
        values = (jnp.ones(output_shape, dtype), jnp.ones(output_shape, dtype), sums[2])
    return tuple(
        jnp.where(mask, value, jnp.asarray(0, dtype))
        for mask, value in zip(masks, values, strict=True)
    )


def calc_cq(
    moist: Any,
    n_moist: int,
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
    param_first_scalar: int = 2,
    rounding_token: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Return WRF ``calc_cq`` outputs using one shared cell expression."""

    del ids, jds, kds
    moist = jnp.asarray(moist)
    shape = (int(ime) - int(ims) + 1, int(kme) - int(kms) + 1, int(jme) - int(jms) + 1)
    if not all(shape):
        return tuple(jnp.zeros(shape, moist.dtype) for _ in range(3))
    bounds = tuple(
        int(value)
        for value in (
            its,
            ite,
            jts,
            jte,
            kts,
            min(int(kte), int(kde) - 1),
            ide,
            jde,
            ims,
            jms,
            kms,
        )
    )
    return map_elements(
        partial(
            _calc_cq_points,
            shape=moist.shape,
            bounds=bounds,
            dtype=moist.dtype,
            species=tuple(range(int(param_first_scalar) - 1, int(n_moist))),
        ),
        (
            jnp.arange(shape[0], dtype=jnp.int32)[:, None, None],
            jnp.arange(shape[1], dtype=jnp.int32)[None, :, None],
            jnp.arange(shape[2], dtype=jnp.int32)[None, None, :],
        ),
        (moist.reshape(-1), rounding_token),
    )


def w_damp(
    rw_tend: Any,
    u: Any,
    v: Any,
    ww: Any,
    w: Any,
    mut: Any,
    c1f: Any,
    c2f: Any,
    rdnw: Any,
    rdx: float,
    rdy: float,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    dt: float,
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
) -> tuple[Any, Any, Any]:
    """Return WRF ``w_damp`` outputs ``rw_tend`` and max CFL diagnostics."""

    del ids, jds, kds, ime, jme, kme, kts, kte, msfuy, msfvx

    rw_arr = jnp.asarray(rw_tend)
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    ww_arr = jnp.asarray(ww)
    w_arr = jnp.asarray(w)
    mut_arr = jnp.asarray(mut)
    c1f_arr = jnp.asarray(c1f)
    c2f_arr = jnp.asarray(c2f)
    rdnw_arr = jnp.asarray(rdnw)
    msfux_arr = jnp.asarray(msfux)
    msfvy_arr = jnp.asarray(msfvy)

    itf = min(int(ite), int(ide) - 1)
    jtf = min(int(jte), int(jde) - 1)
    i_fortran = jnp.arange(int(its), itf + 1)
    k_fortran = jnp.arange(2, int(kde))
    j_fortran = jnp.arange(int(jts), jtf + 1)

    zero = jnp.asarray(0.0, dtype=rw_arr.dtype)
    if i_fortran.size == 0 or k_fortran.size == 0 or j_fortran.size == 0:
        return rw_arr, zero, zero

    ii = _idx(i_fortran, ims)
    kk = _idx(k_fortran, kms)
    jj = _idx(j_fortran, jms)

    c1 = c1f_arr[kk][None, :, None]
    c2 = c2f_arr[kk][None, :, None]
    rdnw_blk = rdnw_arr[kk][None, :, None]
    mut_blk = mut_arr[ii[:, None], jj[None, :]][:, None, :]
    denom = c1 * mut_blk + c2

    ww_blk = ww_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    vert_cfl = jnp.abs(ww_blk / denom * rdnw_blk * jnp.asarray(dt, dtype=rw_arr.dtype))

    msfux_blk = msfux_arr[ii[:, None], jj[None, :]]
    if bool(_read(config_flags, "polar", False)):
        msfxffl = 1.0 / jnp.cos(jnp.asarray(float(_read(config_flags, "fft_filter_lat", 0.0)), dtype=rw_arr.dtype) * DEGRAD)
        msfuxt = jnp.minimum(msfux_blk, msfxffl)
    else:
        msfuxt = msfux_blk

    msfvy_blk = msfvy_arr[ii[:, None], jj[None, :]]
    u_blk = u_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    v_blk = v_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    horiz_cfl = jnp.maximum(
        jnp.abs(u_blk * jnp.asarray(rdx, dtype=rw_arr.dtype) * msfuxt[:, None, :] * jnp.asarray(dt, dtype=rw_arr.dtype)),
        jnp.abs(v_blk * jnp.asarray(rdy, dtype=rw_arr.dtype) * msfvy_blk[:, None, :] * jnp.asarray(dt, dtype=rw_arr.dtype)),
    )

    w_crit_cfl = jnp.asarray(float(_read(config_flags, "w_crit_cfl", 2.0)), dtype=rw_arr.dtype)
    w_damp_on = w_crit_cfl if int(_read(config_flags, "zadvect_implicit", 0)) > 0 else jnp.asarray(W_BETA, dtype=rw_arr.dtype)
    damp_enabled = int(_read(config_flags, "w_damping", 0)) == 1

    w_blk = w_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    sign_w = jnp.where(w_blk < 0.0, -1.0, 1.0).astype(rw_arr.dtype)
    delta = sign_w * jnp.asarray(W_ALPHA, dtype=rw_arr.dtype) * (vert_cfl - w_crit_cfl) * denom
    rw_blk = rw_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
    rw_new = rw_blk - jnp.where((vert_cfl > w_damp_on) & damp_enabled, delta, zero)
    rw_out = rw_arr.at[ii[:, None, None], kk[None, :, None], jj[None, None, :]].set(rw_new)

    return rw_out, jnp.max(vert_cfl), jnp.max(horiz_cfl)


__all__ = [
    "calc_mu_uv",
    "calc_mu_uv_1",
    "calc_alt",
    "calc_p_rho_phi",
    "calc_php",
    "calc_ww_cp",
    "calculate_full",
    "conv_t_tendf_to_moist",
    "initialize_moist_old",
    "coriolis",
    "couple_momentum",
    "curvature",
    "horizontal_pressure_gradient",
    "pg_buoy_w",
    "phy_prep",
    "phy_prep_part2",
    "rhs_ph",
    "w_damp",
    "zero_tend",
    "zero_tend2d",
    "calc_cq",
]
