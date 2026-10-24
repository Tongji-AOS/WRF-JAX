"""Advection kernels translated from WRF ``dyn_em/module_advect_em``."""

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


def _sign_int(a: int, b: int) -> int:
    return abs(int(a)) if int(b) >= 0 else -abs(int(a))


def _sign_real(a: float, b: float) -> float:
    return abs(float(a)) if float(b) >= 0.0 else -abs(float(a))


def _flux4(q_im2: Any, q_im1: Any, q_i: Any, q_ip1: Any, _ua: Any) -> Any:
    return (7.0 * (q_i + q_im1) - (q_ip1 + q_im2)) / 12.0


def _flux3(q_im2: Any, q_im1: Any, q_i: Any, q_ip1: Any, ua: Any, time_step: int) -> Any:
    return _flux4(q_im2, q_im1, q_i, q_ip1, ua) + _sign_int(1, time_step) * _sign_real(1.0, float(ua)) * (
        (q_ip1 - q_im2) - 3.0 * (q_i - q_im1)
    ) / 12.0


def _flux6(q_im3: Any, q_im2: Any, q_im1: Any, q_i: Any, q_ip1: Any, q_ip2: Any, _ua: Any) -> Any:
    return (37.0 * (q_i + q_im1) - 8.0 * (q_ip1 + q_im2) + (q_ip2 + q_im3)) / 60.0


def _flux5(
    q_im3: Any,
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    q_ip2: Any,
    ua: Any,
    time_step: int,
) -> Any:
    return _flux6(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, ua) - _sign_int(1, time_step) * _sign_real(
        1.0, float(ua)
    ) * ((q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)) / 60.0


def _flux3_pd(q_im2: Any, q_im1: Any, q_i: Any, q_ip1: Any, ua: Any, time_step: int) -> Any:
    flux4 = (7.0 / 12.0) * (q_i + q_im1) - (1.0 / 12.0) * (q_ip1 + q_im2)
    return flux4 + _sign_int(1, time_step) * _sign_real(1.0, float(ua)) * (1.0 / 12.0) * (
        (q_ip1 - q_im2) - 3.0 * (q_i - q_im1)
    )


def _flux5_pd(
    q_im3: Any,
    q_im2: Any,
    q_im1: Any,
    q_i: Any,
    q_ip1: Any,
    q_ip2: Any,
    ua: Any,
    time_step: int,
) -> Any:
    flux6 = (
        (37.0 / 60.0) * (q_i + q_im1)
        - (2.0 / 15.0) * (q_ip1 + q_im2)
        + (1.0 / 60.0) * (q_ip2 + q_im3)
    )
    return flux6 - _sign_int(1, time_step) * _sign_real(1.0, float(ua)) * (1.0 / 60.0) * (
        (q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)
    )


def _flux_upwind(q_im1: Any, q_i: Any, cr: Any) -> Any:
    cr_value = float(cr)
    return 0.5 * min(1.0, cr_value + abs(cr_value)) * q_im1 + 0.5 * max(-1.0, cr_value - abs(cr_value)) * q_i


def advect_scalar(
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
) -> Any:
    """Update scalar tendency for the configured WRF fifth/third order path."""

    del c1, c2, mut, msfux, msfuy, msfvx, msfvy, msfty, kds, ime, jme, kme
    horz_order = int(_read(config_flags, "h_sca_adv_order", 5))
    vert_order = int(_read(config_flags, "v_sca_adv_order", 3))
    if horz_order != 5 or vert_order != 3:
        raise NotImplementedError("advect_scalar currently covers WRF h_sca_adv_order=5 and v_sca_adv_order=3")

    field_arr = jnp.asarray(field)
    field_old_arr = jnp.asarray(field_old)
    tendency_out = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msftx_arr = jnp.asarray(msftx)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)

    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    def f(i: int, k: int, j: int) -> Any:
        return field_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def fold(i: int, k: int, j: int) -> Any:
        return field_old_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def ru_v(i: int, k: int, j: int) -> Any:
        return ru_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rv_v(i: int, k: int, j: int) -> Any:
        return rv_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rom_v(i: int, k: int, j: int) -> Any:
        return rom_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def msftx_v(i: int, j: int) -> Any:
        return msftx_arr[_idx(i, ims), _idx(j, jms)]

    def set_t(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    def t(i: int, k: int, j: int) -> Any:
        return tendency_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    degrade_xs = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xs", False))
        or int(its) > int(ids) + 3
    )
    degrade_xe = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xe", False))
        or int(ite) < int(ide) - 3
    )
    degrade_ys = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ys", False))
        or int(jts) > int(jds) + 3
    )
    degrade_ye = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ye", False))
        or int(jte) < int(jde) - 4
    )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    j_start_f = j_start
    j_end_f = j_end + 1
    if degrade_ys:
        j_start = max(int(jts), int(jds) + 1)
        j_start_f = int(jds) + 3
    if degrade_ye:
        j_end = min(int(jte), int(jde) - 2)
        j_end_f = int(jde) - 3
    if bool(_read(config_flags, "polar", False)):
        j_end = min(int(jte), int(jde) - 1)

    fqy_prev: dict[tuple[int, int], Any] = {}
    fqy_next: dict[tuple[int, int], Any] = {}
    for j in range(j_start, j_end + 2):
        current: dict[tuple[int, int], Any] = {}
        if j_start_f <= j <= j_end_f:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = rv_v(i, k, j)
                    current[(i, k)] = vel * _flux5(
                        f(i, k, j - 3),
                        f(i, k, j - 2),
                        f(i, k, j - 1),
                        f(i, k, j),
                        f(i, k, j + 1),
                        f(i, k, j + 2),
                        vel,
                        time_step,
                    )
        elif j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.5 * rv_v(i, k, j) * (f(i, k, j) + f(i, k, j - 1))
        elif j == int(jds) + 2:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = rv_v(i, k, j)
                    current[(i, k)] = vel * _flux3(f(i, k, j - 2), f(i, k, j - 1), f(i, k, j), f(i, k, j + 1), vel, time_step)
        elif j == int(jde) - 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.5 * rv_v(i, k, j) * (f(i, k, j) + f(i, k, j - 1))
        elif j == int(jde) - 2:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = rv_v(i, k, j)
                    current[(i, k)] = vel * _flux3(f(i, k, j - 2), f(i, k, j - 1), f(i, k, j), f(i, k, j + 1), vel, time_step)

        if bool(_read(config_flags, "polar", False)) and j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msftx_v(i, j - 1) * float(rdy) * current[(i, k)])
        elif bool(_read(config_flags, "polar", False)) and j == int(jde):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) + msftx_v(i, j - 1) * float(rdy) * fqy_prev[(i, k)])
        elif j > j_start:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msftx_v(i, j - 1) * float(rdy) * (current[(i, k)] - fqy_prev[(i, k)]))
        fqy_next = fqy_prev
        fqy_prev = current

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    i_start_f = i_start
    i_end_f = i_end + 1
    if degrade_xs:
        i_start = max(int(ids) + 1, int(its))
        i_start_f = min(i_start + 2, int(ids) + 3)
    if degrade_xe:
        i_end = min(int(ide) - 2, int(ite))
        i_end_f = int(ide) - 3

    for j in range(j_start, j_end + 1):
        fqx: dict[tuple[int, int], Any] = {}
        for k in range(int(kts), ktf + 1):
            for i in range(i_start_f, i_end_f + 1):
                vel = ru_v(i, k, j)
                fqx[(i, k)] = vel * _flux5(
                    f(i - 3, k, j),
                    f(i - 2, k, j),
                    f(i - 1, k, j),
                    f(i, k, j),
                    f(i + 1, k, j),
                    f(i + 2, k, j),
                    vel,
                    time_step,
                )
        if degrade_xs:
            for i in range(i_start, i_start_f):
                if i == int(ids) + 1:
                    for k in range(int(kts), ktf + 1):
                        fqx[(i, k)] = 0.5 * ru_v(i, k, j) * (f(i, k, j) + f(i - 1, k, j))
                if i == int(ids) + 2:
                    for k in range(int(kts), ktf + 1):
                        vel = ru_v(i, k, j)
                        fqx[(i, k)] = vel * _flux3(f(i - 2, k, j), f(i - 1, k, j), f(i, k, j), f(i + 1, k, j), vel, time_step)
        if degrade_xe:
            for i in range(i_end_f + 1, i_end + 2):
                if i == int(ide) - 1:
                    for k in range(int(kts), ktf + 1):
                        fqx[(i, k)] = 0.5 * ru_v(i, k, j) * (f(i, k, j) + f(i - 1, k, j))
                if i == int(ide) - 2:
                    for k in range(int(kts), ktf + 1):
                        vel = ru_v(i, k, j)
                        fqx[(i, k)] = vel * _flux3(f(i - 2, k, j), f(i - 1, k, j), f(i, k, j), f(i + 1, k, j), vel, time_step)
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - msftx_v(i, j) * float(rdx) * (fqx[(i + 1, k)] - fqx[(i, k)]))

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    if bool(_read(config_flags, "open_xs", False)) and int(its) == int(ids):
        for j in range(j_start, j_end + 1):
            for k in range(int(kts), ktf + 1):
                ub = min(0.5 * float(ru_v(its, k, j) + ru_v(int(its) + 1, k, j)), 0.0)
                set_t(its, k, j, t(its, k, j) - float(rdx) * (ub * (fold(int(its) + 1, k, j) - fold(its, k, j)) + f(its, k, j) * (ru_v(int(its) + 1, k, j) - ru_v(its, k, j))))

    if bool(_read(config_flags, "open_xe", False)) and int(ite) == int(ide):
        for j in range(j_start, j_end + 1):
            for k in range(int(kts), ktf + 1):
                ub = max(0.5 * float(ru_v(int(ite) - 1, k, j) + ru_v(ite, k, j)), 0.0)
                set_t(i_end, k, j, t(i_end, k, j) - float(rdx) * (ub * (fold(i_end, k, j) - fold(i_end - 1, k, j)) + f(i_end, k, j) * (ru_v(ite, k, j) - ru_v(int(ite) - 1, k, j))))

    if bool(_read(config_flags, "open_ys", False)) and int(jts) == int(jds):
        for i in range(i_start, i_end + 1):
            for k in range(int(kts), ktf + 1):
                vb = min(0.5 * float(rv_v(i, k, jts) + rv_v(i, k, int(jts) + 1)), 0.0)
                set_t(i, k, jts, t(i, k, jts) - float(rdy) * (vb * (fold(i, k, int(jts) + 1) - fold(i, k, jts)) + f(i, k, jts) * (rv_v(i, k, int(jts) + 1) - rv_v(i, k, jts))))

    if bool(_read(config_flags, "open_ye", False)) and int(jte) == int(jde):
        for i in range(i_start, i_end + 1):
            for k in range(int(kts), ktf + 1):
                vb = max(0.5 * float(rv_v(i, k, int(jte) - 1) + rv_v(i, k, jte)), 0.0)
                set_t(i, k, j_end, t(i, k, j_end) - float(rdy) * (vb * (fold(i, k, j_end) - fold(i, k, j_end - 1)) + f(i, k, j_end) * (rv_v(i, k, jte) - rv_v(i, k, int(jte) - 1))))

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    for j in range(j_start, j_end + 1):
        vflux: dict[tuple[int, int], Any] = {}
        for i in range(i_start, i_end + 1):
            vflux[(i, int(kts))] = 0.0
            vflux[(i, int(kte))] = 0.0
        for k in range(int(kts) + 2, ktf):
            for i in range(i_start, i_end + 1):
                vel = rom_v(i, k, j)
                vflux[(i, k)] = vel * _flux3(f(i, k - 2, j), f(i, k - 1, j), f(i, k, j), f(i, k + 1, j), -vel, time_step)
        for i in range(i_start, i_end + 1):
            k = int(kts) + 1
            vflux[(i, k)] = rom_v(i, k, j) * (fzm_arr[_idx(k, kms)] * f(i, k, j) + fzp_arr[_idx(k, kms)] * f(i, k - 1, j))
            k = ktf
            vflux[(i, k)] = rom_v(i, k, j) * (fzm_arr[_idx(k, kms)] * f(i, k, j) + fzp_arr[_idx(k, kms)] * f(i, k - 1, j))
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - rdzw_arr[_idx(k, kms)] * (vflux[(i, k + 1)] - vflux[(i, k)]))

    return tendency_out


def advect_scalar_pd(
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
) -> tuple[Any, Any, Any]:
    """Positive-definite scalar advection for the configured WRF h5/v3 path."""

    del msfux, msfuy, msfvx, msfvy, kds, ime, jme, kme
    horz_order = int(_read(config_flags, "h_sca_adv_order", 5))
    vert_order = int(_read(config_flags, "v_sca_adv_order", 3))
    if horz_order != 5 or vert_order != 3:
        raise NotImplementedError("advect_scalar_pd currently covers WRF h_sca_adv_order=5 and v_sca_adv_order=3")

    field_arr = jnp.asarray(field)
    field_old_arr = jnp.asarray(field_old)
    tendency_out = jnp.asarray(tendency).copy()
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

    ktf = min(int(kte), int(kde) - 1)
    eps = 1.0e-20

    def f(i: int, k: int, j: int) -> Any:
        return field_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def fold(i: int, k: int, j: int) -> Any:
        return field_old_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def ru_v(i: int, k: int, j: int) -> Any:
        return ru_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rv_v(i: int, k: int, j: int) -> Any:
        return rv_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rom_v(i: int, k: int, j: int) -> Any:
        return rom_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def mut_v(i: int, j: int) -> Any:
        return mut_arr[_idx(i, ims), _idx(j, jms)]

    def mub_v(i: int, j: int) -> Any:
        return mub_arr[_idx(i, ims), _idx(j, jms)]

    def mu_old_v(i: int, j: int) -> Any:
        return mu_old_arr[_idx(i, ims), _idx(j, jms)]

    def msftx_v(i: int, j: int) -> Any:
        return msftx_arr[_idx(i, ims), _idx(j, jms)]

    def msfty_v(i: int, j: int) -> Any:
        return msfty_arr[_idx(i, ims), _idx(j, jms)]

    def c1_v(k: int) -> Any:
        return c1_arr[_idx(k, kms)]

    def c2_v(k: int) -> Any:
        return c2_arr[_idx(k, kms)]

    def rdzw_v(k: int) -> Any:
        return rdzw_arr[_idx(k, kms)]

    def set_t(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    def set_h(i: int, k: int, j: int, value: Any) -> None:
        nonlocal h_out
        h_out = h_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    def set_z(i: int, k: int, j: int, value: Any) -> None:
        nonlocal z_out
        z_out = z_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    def t(i: int, k: int, j: int) -> Any:
        return tendency_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    degrade_xs = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xs", False))
        or int(its) > int(ids) + 3
    )
    degrade_xe = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xe", False))
        or int(ite) < int(ide) - 4
    )
    degrade_ys = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ys", False))
        or int(jts) > int(jds) + 3
    )
    degrade_ye = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ye", False))
        or int(jte) < int(jde) - 4
    )

    fqx: dict[tuple[int, int, int], Any] = {}
    fqy: dict[tuple[int, int, int], Any] = {}
    fqz: dict[tuple[int, int, int], Any] = {}
    fqxl: dict[tuple[int, int, int], Any] = {}
    fqyl: dict[tuple[int, int, int], Any] = {}
    fqzl: dict[tuple[int, int, int], Any] = {}

    def dget(data: dict[tuple[int, int, int], Any], i: int, k: int, j: int) -> Any:
        return data.get((i, k, j), 0.0)

    i_start = int(its) - 1
    i_end = min(int(ite), int(ide) - 1) + 1
    j_start = int(jts) - 1
    j_end = min(int(jte), int(jde) - 1) + 1
    j_start_f = j_start
    j_end_f = j_end + 1
    if degrade_xs:
        i_start = max(int(its) - 1, int(ids))
    if degrade_xe:
        i_end = min(int(ite) + 1, int(ide) - 1)
    if degrade_ys:
        j_start = max(int(jts) - 1, int(jds) + 1)
        j_start_f = int(jds) + 3
    if degrade_ye:
        j_end = min(int(jte) + 1, int(jde) - 2)
        j_end_f = int(jde) - 3

    for j in range(j_start, j_end + 2):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                if not (j_start_f <= j <= j_end_f or j in (int(jds) + 1, int(jds) + 2, int(jde) - 1, int(jde) - 2)):
                    continue
                dy = 2.0 / (msftx_v(i, j) + msftx_v(i, j - 1)) / float(rdy)
                mu = 0.5 * ((c1_v(k) * mut_v(i, j) + c2_v(k)) + (c1_v(k) * mut_v(i, j - 1) + c2_v(k)))
                vel = rv_v(i, k, j)
                cr = vel * float(dt) / dy / mu
                fqyl[(i, k, j)] = mu * (dy / float(dt)) * _flux_upwind(fold(i, k, j - 1), fold(i, k, j), cr)
                if j_start_f <= j <= j_end_f:
                    high = vel * _flux5_pd(f(i, k, j - 3), f(i, k, j - 2), f(i, k, j - 1), f(i, k, j), f(i, k, j + 1), f(i, k, j + 2), vel, time_step)
                elif j in (int(jds) + 1, int(jde) - 1):
                    high = 0.5 * vel * (f(i, k, j) + f(i, k, j - 1))
                else:
                    high = vel * _flux3_pd(f(i, k, j - 2), f(i, k, j - 1), f(i, k, j), f(i, k, j + 1), vel, time_step)
                fqy[(i, k, j)] = high - fqyl[(i, k, j)]

    i_start = int(its) - 1
    i_end = min(int(ite), int(ide) - 1) + 1
    i_start_f = i_start
    i_end_f = i_end + 1
    j_start = int(jts) - 1
    j_end = min(int(jte), int(jde) - 1) + 1
    if degrade_ys:
        j_start = max(int(jts) - 1, int(jds))
    if degrade_ye:
        j_end = min(int(jte) + 1, int(jde) - 1)
    if degrade_xs:
        i_start = max(int(ids) + 1, int(its) - 1)
        i_start_f = int(ids) + 3
    if degrade_xe:
        i_end = min(int(ide) - 2, int(ite) + 1)
        i_end_f = int(ide) - 3

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start_f, i_end_f + 1):
                dx = 2.0 / (msfty_v(i, j) + msfty_v(i - 1, j)) / float(rdx)
                mu = 0.5 * ((c1_v(k) * mut_v(i, j) + c2_v(k)) + (c1_v(k) * mut_v(i - 1, j) + c2_v(k)))
                vel = ru_v(i, k, j)
                cr = vel * float(dt) / dx / mu
                fqxl[(i, k, j)] = mu * (dx / float(dt)) * _flux_upwind(fold(i - 1, k, j), fold(i, k, j), cr)
                high = vel * _flux5_pd(f(i - 3, k, j), f(i - 2, k, j), f(i - 1, k, j), f(i, k, j), f(i + 1, k, j), f(i + 2, k, j), vel, time_step)
                fqx[(i, k, j)] = high - fqxl[(i, k, j)]
            if degrade_xs:
                for i in range(i_start, i_start_f):
                    dx = 2.0 / (msfty_v(i, j) + msfty_v(i - 1, j)) / float(rdx)
                    mu = 0.5 * ((c1_v(k) * mut_v(i, j) + c2_v(k)) + (c1_v(k) * mut_v(i - 1, j) + c2_v(k)))
                    vel = ru_v(i, k, j)
                    cr = (vel / mu if i == int(ids) + 1 else vel / mu) * float(dt) / dx
                    fqxl[(i, k, j)] = mu * (dx / float(dt)) * _flux_upwind(fold(i - 1, k, j), fold(i, k, j), cr)
                    if i == int(ids) + 1:
                        high = 0.5 * vel * (f(i, k, j) + f(i - 1, k, j))
                    elif i == int(ids) + 2:
                        high = vel * _flux3_pd(f(i - 2, k, j), f(i - 1, k, j), f(i, k, j), f(i + 1, k, j), vel, time_step)
                    else:
                        continue
                    fqx[(i, k, j)] = high - fqxl[(i, k, j)]
            if degrade_xe:
                for i in range(i_end_f + 1, i_end + 2):
                    dx = 2.0 / (msfty_v(i, j) + msfty_v(i - 1, j)) / float(rdx)
                    mu = 0.5 * ((c1_v(k) * mut_v(i, j) + c2_v(k)) + (c1_v(k) * mut_v(i - 1, j) + c2_v(k)))
                    vel = ru_v(i, k, j)
                    cr = vel * float(dt) / dx / mu
                    fqxl[(i, k, j)] = mu * (dx / float(dt)) * _flux_upwind(fold(i - 1, k, j), fold(i, k, j), cr)
                    if i == int(ide) - 1:
                        high = 0.5 * vel * (f(i, k, j) + f(i - 1, k, j))
                    elif i == int(ide) - 2:
                        high = vel * _flux3_pd(f(i - 2, k, j), f(i - 1, k, j), f(i, k, j), f(i + 1, k, j), vel, time_step)
                    else:
                        continue
                    fqx[(i, k, j)] = high - fqxl[(i, k, j)]

    i_start = int(its) - 1
    i_end = min(int(ite), int(ide) - 1) + 1
    j_start = int(jts) - 1
    j_end = min(int(jte), int(jde) - 1) + 1
    if degrade_xs:
        i_start = max(int(its) - 1, int(ids))
    if degrade_xe:
        i_end = min(int(ite) + 1, int(ide) - 1)
    if degrade_ys:
        j_start = max(int(jts) - 1, int(jds))
    if degrade_ye:
        j_end = min(int(jte) + 1, int(jde) - 1)

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            fqz[(i, 1, j)] = 0.0
            fqzl[(i, 1, j)] = 0.0
            fqz[(i, int(kde), j)] = 0.0
            fqzl[(i, int(kde), j)] = 0.0
        for k in range(int(kts) + 2, ktf):
            for i in range(i_start, i_end + 1):
                dz = 2.0 / (rdzw_v(k) + rdzw_v(k - 1))
                mu = c1_v(k) * mut_v(i, j) + c2_v(k)
                vel = rom_v(i, k, j)
                cr = vel * float(dt) / dz / mu
                fqzl[(i, k, j)] = mu * (dz / float(dt)) * _flux_upwind(fold(i, k - 1, j), fold(i, k, j), cr)
                high = vel * _flux3_pd(f(i, k - 2, j), f(i, k - 1, j), f(i, k, j), f(i, k + 1, j), -vel, time_step)
                fqz[(i, k, j)] = high - fqzl[(i, k, j)]
        for i in range(i_start, i_end + 1):
            for k in (int(kts) + 1, ktf):
                dz = 2.0 / (rdzw_v(k) + rdzw_v(k - 1))
                mu = c1_v(k) * mut_v(i, j) + c2_v(k)
                vel = rom_v(i, k, j)
                cr = vel * float(dt) / dz / mu
                fqzl[(i, k, j)] = mu * (dz / float(dt)) * _flux_upwind(fold(i, k - 1, j), fold(i, k, j), cr)
                high = vel * (fzm_arr[_idx(k, kms)] * f(i, k, j) + fzp_arr[_idx(k, kms)] * f(i, k - 1, j))
                fqz[(i, k, j)] = high - fqzl[(i, k, j)]

    pd_i_start = int(its) - 1
    pd_i_end = min(int(ite), int(ide) - 1) + 1
    pd_j_start = int(jts) - 1
    pd_j_end = min(int(jte), int(jde) - 1) + 1
    if degrade_xs:
        pd_i_start = max(int(its) - 1, int(ids))
    if degrade_xe:
        pd_i_end = min(int(ite) + 1, int(ide) - 1)
    if degrade_ys:
        pd_j_start = max(int(jts) - 1, int(jds))
    if degrade_ye:
        pd_j_end = min(int(jte) + 1, int(jde) - 1)
    if bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False)):
        if degrade_xs:
            pd_i_start = max(int(its) - 1, int(ids) + 1)
        if degrade_xe:
            pd_i_end = min(int(ite) + 1, int(ide) - 2)
        if degrade_ys:
            pd_j_start = max(int(jts) - 1, int(jds) + 1)
        if degrade_ye:
            pd_j_end = min(int(jte) + 1, int(jde) - 2)

    for j in range(pd_j_start, pd_j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(pd_i_start, pd_i_end + 1):
                ph_low = ((c1_v(k) * mub_v(i, j) + c2_v(k)) + (c1_v(k) * mu_old_v(i, j))) * fold(i, k, j) - float(dt) * (
                    msftx_v(i, j)
                    * msfty_v(i, j)
                    * (float(rdx) * (dget(fqxl, i + 1, k, j) - dget(fqxl, i, k, j)) + float(rdy) * (dget(fqyl, i, k, j + 1) - dget(fqyl, i, k, j)))
                    + msfty_v(i, j) * rdzw_v(k) * (dget(fqzl, i, k + 1, j) - dget(fqzl, i, k, j))
                )
                flux_out = float(dt) * (
                    msftx_v(i, j)
                    * msfty_v(i, j)
                    * (
                        float(rdx) * (max(0.0, float(dget(fqx, i + 1, k, j))) - min(0.0, float(dget(fqx, i, k, j))))
                        + float(rdy) * (max(0.0, float(dget(fqy, i, k, j + 1))) - min(0.0, float(dget(fqy, i, k, j))))
                    )
                    + msfty_v(i, j)
                    * rdzw_v(k)
                    * (min(0.0, float(dget(fqz, i, k + 1, j))) - max(0.0, float(dget(fqz, i, k, j))))
                )
                if float(flux_out) > float(ph_low):
                    scale = max(0.0, float(ph_low) / (float(flux_out) + eps))
                    if float(dget(fqx, i + 1, k, j)) > 0.0:
                        fqx[(i + 1, k, j)] = scale * dget(fqx, i + 1, k, j)
                    if float(dget(fqx, i, k, j)) < 0.0:
                        fqx[(i, k, j)] = scale * dget(fqx, i, k, j)
                    if float(dget(fqy, i, k, j + 1)) > 0.0:
                        fqy[(i, k, j + 1)] = scale * dget(fqy, i, k, j + 1)
                    if float(dget(fqy, i, k, j)) < 0.0:
                        fqy[(i, k, j)] = scale * dget(fqy, i, k, j)
                    if float(dget(fqz, i, k + 1, j)) < 0.0:
                        fqz[(i, k + 1, j)] = scale * dget(fqz, i, k + 1, j)
                    if float(dget(fqz, i, k, j)) > 0.0:
                        fqz[(i, k, j)] = scale * dget(fqz, i, k, j)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = -rdzw_v(k) * (dget(fqz, i, k + 1, j) - dget(fqz, i, k, j) + dget(fqzl, i, k + 1, j) - dget(fqzl, i, k, j))
                set_t(i, k, j, t(i, k, j) + value)
                if tenddec:
                    set_z(i, k, j, value)

    if degrade_xs:
        i_start = max(int(its), int(ids) + 1)
    if degrade_xe:
        i_end = min(int(ite), int(ide) - 2)
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = -msftx_v(i, j) * (float(rdx) * (dget(fqx, i + 1, k, j) - dget(fqx, i, k, j) + dget(fqxl, i + 1, k, j) - dget(fqxl, i, k, j)))
                set_t(i, k, j, t(i, k, j) + value)
                if tenddec:
                    set_h(i, k, j, value)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    if degrade_ys:
        j_start = max(int(jts), int(jds) + 1)
    if degrade_ye:
        j_end = min(int(jte), int(jde) - 2)
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = -msftx_v(i, j) * (float(rdy) * (dget(fqy, i, k, j + 1) - dget(fqy, i, k, j) + dget(fqyl, i, k, j + 1) - dget(fqyl, i, k, j)))
                set_t(i, k, j, t(i, k, j) + value)
                if tenddec:
                    set_h(i, k, j, h_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)] + value)

    return tendency_out, h_out, z_out


def advect_u(
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
) -> Any:
    """Update U-momentum tendency for the configured WRF fifth/third order path."""

    del msfuy, msfvx, msfvy, msftx, msfty, kds, ime, jme, kme
    horz_order = int(_read(config_flags, "h_mom_adv_order", 5))
    vert_order = int(_read(config_flags, "v_mom_adv_order", 3))
    if horz_order != 5 or vert_order != 3:
        raise NotImplementedError("advect_u currently covers WRF h_mom_adv_order=5 and v_mom_adv_order=3")

    u_arr = jnp.asarray(u)
    u_old_arr = jnp.asarray(u_old)
    tendency_out = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mut_arr = jnp.asarray(mut)
    msfux_arr = jnp.asarray(msfux)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)

    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    def q(i: int, k: int, j: int) -> Any:
        return u_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def qold(i: int, k: int, j: int) -> Any:
        return u_old_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def ru_v(i: int, k: int, j: int) -> Any:
        return ru_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rv_v(i: int, k: int, j: int) -> Any:
        return rv_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rom_v(i: int, k: int, j: int) -> Any:
        return rom_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def msfux_v(i: int, j: int) -> Any:
        return msfux_arr[_idx(i, ims), _idx(j, jms)]

    def mut_v(i: int, j: int) -> Any:
        return mut_arr[_idx(i, ims), _idx(j, jms)]

    def c1_v(k: int) -> Any:
        return c1_arr[_idx(k, kms)]

    def c2_v(k: int) -> Any:
        return c2_arr[_idx(k, kms)]

    def rdzw_v(k: int) -> Any:
        return rdzw_arr[_idx(k, kms)]

    def t(i: int, k: int, j: int) -> Any:
        return tendency_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set_t(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    degrade_xs = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xs", False))
        or int(its) > int(ids) + 3
    )
    degrade_xe = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xe", False))
        or int(ite) < int(ide) - 2
    )
    degrade_ys = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ys", False))
        or int(jts) > int(jds) + 3
    )
    degrade_ye = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ye", False))
        or int(jte) < int(jde) - 4
    )

    i_start = int(its)
    i_end = int(ite)
    if bool(_read(config_flags, "open_xs", False)) or specified:
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_xe", False)) or specified:
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    j_start_f = j_start
    j_end_f = j_end + 1
    if degrade_ys:
        j_start = max(int(jts), int(jds) + 1)
        j_start_f = int(jds) + 3
    if degrade_ye:
        j_end = min(int(jte), int(jde) - 2)
        j_end_f = int(jde) - 3
    if bool(_read(config_flags, "polar", False)):
        j_end = min(int(jte), int(jde) - 1)

    fqy_prev: dict[tuple[int, int], Any] = {}
    for j in range(j_start, j_end + 2):
        current: dict[tuple[int, int], Any] = {}
        if j_start_f <= j <= j_end_f:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i - 1, k, j))
                    current[(i, k)] = vel * _flux5(q(i, k, j - 3), q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), q(i, k, j + 2), vel, time_step)
        elif j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.25 * (rv_v(i, k, j) + rv_v(i - 1, k, j)) * (q(i, k, j) + q(i, k, j - 1))
        elif j == int(jds) + 2:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i - 1, k, j))
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)
        elif j == int(jde) - 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.25 * (rv_v(i, k, j) + rv_v(i - 1, k, j)) * (q(i, k, j) + q(i, k, j - 1))
        elif j == int(jde) - 2:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i - 1, k, j))
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)
        if bool(_read(config_flags, "polar", False)) and j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msfux_v(i, j - 1) * float(rdy) * current[(i, k)])
        elif bool(_read(config_flags, "polar", False)) and j == int(jde):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) + msfux_v(i, j - 1) * float(rdy) * fqy_prev[(i, k)])
        elif j > j_start:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msfux_v(i, j - 1) * float(rdy) * (current[(i, k)] - fqy_prev[(i, k)]))
        fqy_prev = current

    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    i_start_f = i_start
    i_end_f = i_end + 1
    if degrade_xs:
        i_start = max(int(ids) + 1, int(its))
        i_start_f = int(ids) + 3
    if degrade_xe:
        i_end = min(int(ide) - 1, int(ite))
        i_end_f = int(ide) - 2

    for j in range(j_start, j_end + 1):
        fqx: dict[tuple[int, int], Any] = {}
        for k in range(int(kts), ktf + 1):
            for i in range(i_start_f, i_end_f + 1):
                vel = 0.5 * (ru_v(i, k, j) + ru_v(i - 1, k, j))
                fqx[(i, k)] = vel * _flux5(q(i - 3, k, j), q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), q(i + 2, k, j), vel, time_step)
            if degrade_xs:
                if i_start == int(ids) + 1:
                    i = int(ids) + 1
                    ub = q(i - 1, k, j)
                    if specified and float(q(i, k, j)) < 0.0:
                        ub = q(i, k, j)
                    fqx[(i, k)] = 0.25 * (ru_v(i, k, j) + ru_v(i - 1, k, j)) * (q(i, k, j) + ub)
                i = int(ids) + 2
                vel = 0.5 * (ru_v(i, k, j) + ru_v(i - 1, k, j))
                fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
            if degrade_xe:
                if i_end == int(ide) - 1:
                    i = int(ide)
                    ub = q(i, k, j)
                    if specified and float(q(i - 1, k, j)) > 0.0:
                        ub = q(i - 1, k, j)
                    fqx[(i, k)] = 0.25 * (ru_v(i, k, j) + ru_v(i - 1, k, j)) * (q(i - 1, k, j) + ub)
                i = int(ide) - 1
                vel = 0.5 * (ru_v(i, k, j) + ru_v(i - 1, k, j))
                fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - msfux_v(i, j) * float(rdx) * (fqx[(i + 1, k)] - fqx[(i, k)]))

    if bool(_read(config_flags, "open_xs", False)) and int(its) == int(ids):
        for j in range(int(jts), min(int(jte), int(jde) - 1) + 1):
            for k in range(int(kts), ktf + 1):
                ub = min(float(ru_v(its, k, j) - 25.0 * (c1_v(k) * mut_v(its, j) + c2_v(k))), 0.0)
                set_t(its, k, j, t(its, k, j) - float(rdx) * ub * (qold(int(its) + 1, k, j) - qold(its, k, j)))

    if bool(_read(config_flags, "open_xe", False)) and int(ite) == int(ide):
        for j in range(int(jts), min(int(jte), int(jde) - 1) + 1):
            for k in range(int(kts), ktf + 1):
                ub = max(float(ru_v(ite, k, j) + 25.0 * (c1_v(k) * mut_v(int(ite) - 1, j) + c2_v(k))), 0.0)
                set_t(ite, k, j, t(ite, k, j) - float(rdx) * ub * (qold(ite, k, j) - qold(int(ite) - 1, k, j)))

    i_start = int(its)
    i_end = min(int(ite), int(ide))
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if bool(_read(config_flags, "open_ys", False)) or specified:
        i_start = max(int(ids) + 1, int(its))
    if bool(_read(config_flags, "open_ye", False)) or specified:
        i_end = min(int(ide) - 1, int(ite))
    if bool(_read(config_flags, "periodic_x", False)):
        i_start = int(its)
        i_end = int(ite)

    for j in range(j_start, j_end + 1):
        vflux: dict[tuple[int, int], Any] = {}
        for i in range(i_start, i_end + 1):
            vflux[(i, int(kts))] = 0.0
            vflux[(i, int(kte))] = 0.0
        for k in range(int(kts) + 2, ktf):
            for i in range(i_start, i_end + 1):
                vel = 0.5 * (rom_v(i - 1, k, j) + rom_v(i, k, j))
                vflux[(i, k)] = vel * _flux3(q(i, k - 2, j), q(i, k - 1, j), q(i, k, j), q(i, k + 1, j), -vel, time_step)
        for i in range(i_start, i_end + 1):
            for k in (int(kts) + 1, ktf):
                vflux[(i, k)] = 0.5 * (rom_v(i, k, j) + rom_v(i - 1, k, j)) * (
                    fzm_arr[_idx(k, kms)] * q(i, k, j) + fzp_arr[_idx(k, kms)] * q(i, k - 1, j)
                )
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - rdzw_v(k) * (vflux[(i, k + 1)] - vflux[(i, k)]))

    return tendency_out


def advect_v(
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
) -> Any:
    """Update V-momentum tendency for the configured WRF fifth/third order path."""

    del msfux, msfuy, msftx, msfty, kds, ime, jme, kme
    horz_order = int(_read(config_flags, "h_mom_adv_order", 5))
    vert_order = int(_read(config_flags, "v_mom_adv_order", 3))
    if horz_order != 5 or vert_order != 3:
        raise NotImplementedError("advect_v currently covers WRF h_mom_adv_order=5 and v_mom_adv_order=3")

    v_arr = jnp.asarray(v)
    v_old_arr = jnp.asarray(v_old)
    tendency_out = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    c1_arr = jnp.asarray(c1)
    c2_arr = jnp.asarray(c2)
    mut_arr = jnp.asarray(mut)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzw_arr = jnp.asarray(rdzw)

    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    def q(i: int, k: int, j: int) -> Any:
        return v_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def qold(i: int, k: int, j: int) -> Any:
        return v_old_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def ru_v(i: int, k: int, j: int) -> Any:
        return ru_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rv_v(i: int, k: int, j: int) -> Any:
        return rv_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rom_v(i: int, k: int, j: int) -> Any:
        return rom_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def msfvx_v(i: int, j: int) -> Any:
        return msfvx_arr[_idx(i, ims), _idx(j, jms)]

    def msfvy_v(i: int, j: int) -> Any:
        return msfvy_arr[_idx(i, ims), _idx(j, jms)]

    def mut_v(i: int, j: int) -> Any:
        return mut_arr[_idx(i, ims), _idx(j, jms)]

    def c1_v(k: int) -> Any:
        return c1_arr[_idx(k, kms)]

    def c2_v(k: int) -> Any:
        return c2_arr[_idx(k, kms)]

    def rdzw_v(k: int) -> Any:
        return rdzw_arr[_idx(k, kms)]

    def t(i: int, k: int, j: int) -> Any:
        return tendency_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set_t(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    degrade_xs = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xs", False))
        or int(its) > int(ids) + 3
    )
    degrade_xe = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xe", False))
        or int(ite) < int(ide) - 3
    )
    degrade_ys = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ys", False))
        or int(jts) > int(jds) + 3
    )
    degrade_ye = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ye", False))
        or int(jte) < int(jde) - 3
    )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    j_start_f = j_start
    j_end_f = j_end + 1
    if degrade_ys:
        j_start = max(int(jts), int(jds) + 1)
        j_start_f = int(jds) + 3
    if degrade_ye:
        j_end = min(int(jte), int(jde) - 1)
        j_end_f = int(jde) - 2

    fqy_prev: dict[tuple[int, int], Any] = {}
    for j in range(j_start, j_end + 2):
        current: dict[tuple[int, int], Any] = {}
        if j_start_f <= j <= j_end_f:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i, k, j - 1))
                    current[(i, k)] = vel * _flux5(q(i, k, j - 3), q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), q(i, k, j + 2), vel, time_step)
        elif j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vb = q(i, k, j - 1)
                    if specified and float(q(i, k, j)) < 0.0:
                        vb = q(i, k, j)
                    current[(i, k)] = 0.25 * (rv_v(i, k, j) + rv_v(i, k, j - 1)) * (q(i, k, j) + vb)
        elif j == int(jds) + 2:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i, k, j - 1))
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)
        elif j == int(jde):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vb = q(i, k, j)
                    if specified and float(q(i, k, j - 1)) > 0.0:
                        vb = q(i, k, j - 1)
                    current[(i, k)] = 0.25 * (rv_v(i, k, j) + rv_v(i, k, j - 1)) * (vb + q(i, k, j - 1))
        elif j == int(jde) - 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    vel = 0.5 * (rv_v(i, k, j) + rv_v(i, k, j - 1))
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)

        if bool(_read(config_flags, "polar", False)) and j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, 0.0)
        elif bool(_read(config_flags, "polar", False)) and j == int(jde) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, 0.0)
        elif j > j_start:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msfvy_v(i, j - 1) * float(rdy) * (current[(i, k)] - fqy_prev[(i, k)]))
        fqy_prev = current

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_start = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_end = min(int(jde) - 1, int(jte))
    i_start_f = i_start
    i_end_f = i_end + 1
    if degrade_xs:
        i_start = max(int(ids) + 1, int(its))
        i_start_f = min(i_start + 2, int(ids) + 3)
    if degrade_xe:
        i_end = min(int(ide) - 2, int(ite))
        i_end_f = int(ide) - 3

    for j in range(j_start, j_end + 1):
        fqx: dict[tuple[int, int], Any] = {}
        for k in range(int(kts), ktf + 1):
            for i in range(i_start_f, i_end_f + 1):
                vel = 0.5 * (ru_v(i, k, j) + ru_v(i, k, j - 1))
                fqx[(i, k)] = vel * _flux5(q(i - 3, k, j), q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), q(i + 2, k, j), vel, time_step)
            if degrade_xs:
                for i in range(i_start, i_start_f):
                    if i == int(ids) + 1:
                        fqx[(i, k)] = 0.25 * (ru_v(i, k, j) + ru_v(i, k, j - 1)) * (q(i, k, j) + q(i - 1, k, j))
                    if i == int(ids) + 2:
                        vel = 0.5 * (ru_v(i, k, j) + ru_v(i, k, j - 1))
                        fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
            if degrade_xe:
                for i in range(i_end_f + 1, i_end + 2):
                    if i == int(ide) - 1:
                        fqx[(i, k)] = 0.25 * (ru_v(i_end + 1, k, j) + ru_v(i_end + 1, k, j - 1)) * (q(i_end + 1, k, j) + q(i_end, k, j))
                    if i == int(ide) - 2:
                        vel = 0.5 * (ru_v(i, k, j) + ru_v(i, k, j - 1))
                        fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - msfvy_v(i, j) * float(rdx) * (fqx[(i + 1, k)] - fqx[(i, k)]))

    if bool(_read(config_flags, "polar", False)) and int(jts) == int(jds):
        for i in range(int(its), int(ite) + 1):
            for k in range(int(kts), ktf + 1):
                set_t(i, k, int(jts), 0.0)
    if bool(_read(config_flags, "polar", False)) and int(jte) == int(jde):
        for i in range(int(its), int(ite) + 1):
            for k in range(int(kts), ktf + 1):
                set_t(i, k, int(jte), 0.0)

    if bool(_read(config_flags, "open_ys", False)) and int(jts) == int(jds):
        for i in range(int(its), min(int(ite), int(ide) - 1) + 1):
            for k in range(int(kts), ktf + 1):
                vb = min(float(rv_v(i, k, jts) - 25.0 * (c1_v(k) * mut_v(i, jts) + c2_v(k))), 0.0)
                set_t(i, k, jts, t(i, k, jts) - float(rdy) * vb * (qold(i, k, int(jts) + 1) - qold(i, k, jts)))

    if bool(_read(config_flags, "open_ye", False)) and int(jte) == int(jde):
        for i in range(int(its), min(int(ite), int(ide) - 1) + 1):
            for k in range(int(kts), ktf + 1):
                vb = max(float(rv_v(i, k, jte) + 25.0 * (c1_v(k) * mut_v(i, int(jte) - 1) + c2_v(k))), 0.0)
                set_t(i, k, jte, t(i, k, jte) - float(rdy) * vb * (qold(i, k, jte) - qold(i, k, int(jte) - 1)))

    j_start = int(jts)
    j_end = min(int(jte), int(jde))
    jmin = int(jds)
    jmax = int(jde) - 1
    if bool(_read(config_flags, "open_ys", False)):
        j_start = max(int(jds) + 1, int(jts))
        jmin = int(jds)
    if bool(_read(config_flags, "open_ye", False)):
        j_end = min(int(jte), int(jde) - 1)
        jmax = int(jde) - 1

    if bool(_read(config_flags, "open_xs", False)) and int(its) == int(ids):
        for j in range(j_start, j_end + 1):
            jp = min(jmax, j)
            jm = max(jmin, j - 1)
            for k in range(int(kts), ktf + 1):
                uw = 0.5 * (ru_v(its, k, jp) + ru_v(its, k, jm))
                ub = min(float(uw), 0.0)
                dup = ru_v(int(its) + 1, k, jp) - ru_v(its, k, jp)
                dum = ru_v(int(its) + 1, k, jm) - ru_v(its, k, jm)
                value = t(its, k, j) - msfvy_v(its, j) * float(rdx) * (
                    ub * (qold(int(its) + 1, k, j) - qold(its, k, j)) + 0.5 * q(its, k, j) * (dup + dum)
                )
                set_t(its, k, j, value)

    if bool(_read(config_flags, "open_xe", False)) and int(ite) == int(ide):
        for j in range(j_start, j_end + 1):
            jp = min(jmax, j)
            jm = max(jmin, j - 1)
            for k in range(int(kts), ktf + 1):
                uw = 0.5 * (ru_v(ite, k, jp) + ru_v(ite, k, jm))
                ub = max(float(uw), 0.0)
                dup = ru_v(ite, k, jp) - ru_v(int(ite) - 1, k, jp)
                dum = ru_v(ite, k, jm) - ru_v(int(ite) - 1, k, jm)
                value = t(int(ite) - 1, k, j) - msfvy_v(int(ite) - 1, j) * float(rdx) * (
                    ub * (qold(int(ite) - 1, k, j) - qold(int(ite) - 2, k, j)) + 0.5 * q(int(ite) - 1, k, j) * (dup + dum)
                )
                set_t(int(ite) - 1, k, j, value)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    if bool(_read(config_flags, "open_ys", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_start = max(int(jds) + 1, int(jts))
    if bool(_read(config_flags, "open_ye", False)) or specified or bool(_read(config_flags, "polar", False)):
        j_end = min(int(jde) - 1, int(jte))

    for j in range(j_start, j_end + 1):
        vflux: dict[tuple[int, int], Any] = {}
        for i in range(i_start, i_end + 1):
            vflux[(i, int(kts))] = 0.0
            vflux[(i, int(kte))] = 0.0
        for k in range(int(kts) + 2, ktf):
            for i in range(i_start, i_end + 1):
                vel = 0.5 * (rom_v(i, k, j) + rom_v(i, k, j - 1))
                vflux[(i, k)] = vel * _flux3(q(i, k - 2, j), q(i, k - 1, j), q(i, k, j), q(i, k + 1, j), -vel, time_step)
        for i in range(i_start, i_end + 1):
            for k in (int(kts) + 1, ktf):
                vflux[(i, k)] = 0.5 * (rom_v(i, k, j) + rom_v(i, k, j - 1)) * (
                    fzm_arr[_idx(k, kms)] * q(i, k, j) + fzp_arr[_idx(k, kms)] * q(i, k - 1, j)
                )
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - (msfvy_v(i, j) / msfvx_v(i, j)) * rdzw_v(k) * (vflux[(i, k + 1)] - vflux[(i, k)]))

    return tendency_out


def advect_w(
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
) -> Any:
    """Update W-momentum tendency for the configured WRF fifth/third order path."""

    del c1, c2, mut, msfux, msfuy, msfvx, msfvy, msfty, kds, ime, jme, kme
    horz_order = int(_read(config_flags, "h_sca_adv_order", 5))
    vert_order = int(_read(config_flags, "v_sca_adv_order", 3))
    if horz_order != 5 or vert_order != 3:
        raise NotImplementedError("advect_w currently covers WRF h_sca_adv_order=5 and v_sca_adv_order=3")

    w_arr = jnp.asarray(w)
    w_old_arr = jnp.asarray(w_old)
    tendency_out = jnp.asarray(tendency).copy()
    ru_arr = jnp.asarray(ru)
    rv_arr = jnp.asarray(rv)
    rom_arr = jnp.asarray(rom)
    msftx_arr = jnp.asarray(msftx)
    fzm_arr = jnp.asarray(fzm)
    fzp_arr = jnp.asarray(fzp)
    rdzu_arr = jnp.asarray(rdzu)

    ktf = min(int(kte), int(kde) - 1)
    specified = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))

    def q(i: int, k: int, j: int) -> Any:
        return w_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def qold(i: int, k: int, j: int) -> Any:
        return w_old_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def ru_v(i: int, k: int, j: int) -> Any:
        return ru_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rv_v(i: int, k: int, j: int) -> Any:
        return rv_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def rom_v(i: int, k: int, j: int) -> Any:
        return rom_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def msftx_v(i: int, j: int) -> Any:
        return msftx_arr[_idx(i, ims), _idx(j, jms)]

    def fzm_v(k: int) -> Any:
        return fzm_arr[_idx(k, kms)]

    def fzp_v(k: int) -> Any:
        return fzp_arr[_idx(k, kms)]

    def rdzu_v(k: int) -> Any:
        return rdzu_arr[_idx(k, kms)]

    def ru_w(i: int, k: int, j: int) -> Any:
        if k == ktf + 1:
            return (2.0 - fzm_v(k - 1)) * ru_v(i, k - 1, j) - fzp_v(k - 1) * ru_v(i, k - 2, j)
        return fzm_v(k) * ru_v(i, k, j) + fzp_v(k) * ru_v(i, k - 1, j)

    def rv_w(i: int, k: int, j: int) -> Any:
        if k == ktf + 1:
            return (2.0 - fzm_v(k - 1)) * rv_v(i, k - 1, j) - fzp_v(k - 1) * rv_v(i, k - 2, j)
        return fzm_v(k) * rv_v(i, k, j) + fzp_v(k) * rv_v(i, k - 1, j)

    def t(i: int, k: int, j: int) -> Any:
        return tendency_out[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set_t(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    degrade_xs = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xs", False))
        or int(its) > int(ids) + 3
    )
    degrade_xe = not (
        bool(_read(config_flags, "periodic_x", False))
        or bool(_read(config_flags, "symmetric_xe", False))
        or int(ite) < int(ide) - 3
    )
    degrade_ys = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ys", False))
        or int(jts) > int(jds) + 3
    )
    degrade_ye = not (
        bool(_read(config_flags, "periodic_y", False))
        or bool(_read(config_flags, "symmetric_ye", False))
        or int(jte) < int(jde) - 4
    )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    j_start_f = j_start
    j_end_f = j_end + 1
    if degrade_ys:
        j_start = max(int(jts), int(jds) + 1)
        j_start_f = int(jds) + 3
    if degrade_ye:
        j_end = min(int(jte), int(jde) - 2)
        j_end_f = int(jde) - 3
    if bool(_read(config_flags, "polar", False)):
        j_end = min(int(jte), int(jde) - 1)

    fqy_prev: dict[tuple[int, int], Any] = {}
    for j in range(j_start, j_end + 2):
        current: dict[tuple[int, int], Any] = {}
        if j_start_f <= j <= j_end_f:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    vel = rv_w(i, k, j)
                    current[(i, k)] = vel * _flux5(q(i, k, j - 3), q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), q(i, k, j + 2), vel, time_step)
        elif j == int(jds) + 1:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.5 * rv_w(i, k, j) * (q(i, k, j) + q(i, k, j - 1))
        elif j == int(jds) + 2:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    vel = rv_w(i, k, j)
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)
        elif j == int(jde) - 1:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    current[(i, k)] = 0.5 * rv_w(i, k, j) * (q(i, k, j) + q(i, k, j - 1))
        elif j == int(jde) - 2:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    vel = rv_w(i, k, j)
                    current[(i, k)] = vel * _flux3(q(i, k, j - 2), q(i, k, j - 1), q(i, k, j), q(i, k, j + 1), vel, time_step)

        if bool(_read(config_flags, "polar", False)) and j == int(jds) + 1:
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msftx_v(i, j - 1) * float(rdy) * current[(i, k)])
        elif bool(_read(config_flags, "polar", False)) and j == int(jde):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) + msftx_v(i, j - 1) * float(rdy) * fqy_prev[(i, k)])
        elif j > j_start:
            for k in range(int(kts) + 1, ktf + 2):
                for i in range(i_start, i_end + 1):
                    set_t(i, k, j - 1, t(i, k, j - 1) - msftx_v(i, j - 1) * float(rdy) * (current[(i, k)] - fqy_prev[(i, k)]))
        fqy_prev = current

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    i_start_f = i_start
    i_end_f = i_end + 1
    if degrade_xs:
        i_start = max(int(ids) + 1, int(its))
        i_start_f = min(i_start + 2, int(ids) + 3)
    if degrade_xe:
        i_end = min(int(ide) - 2, int(ite))
        i_end_f = int(ide) - 3

    for j in range(j_start, j_end + 1):
        fqx: dict[tuple[int, int], Any] = {}
        for k in range(int(kts) + 1, ktf + 2):
            for i in range(i_start_f, i_end_f + 1):
                vel = ru_w(i, k, j)
                fqx[(i, k)] = vel * _flux5(q(i - 3, k, j), q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), q(i + 2, k, j), vel, time_step)
            if degrade_xs:
                for i in range(i_start, i_start_f):
                    if i == int(ids) + 1:
                        fqx[(i, k)] = 0.5 * ru_w(i, k, j) * (q(i, k, j) + q(i - 1, k, j))
                    if i == int(ids) + 2:
                        vel = ru_w(i, k, j)
                        fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
            if degrade_xe:
                for i in range(i_end_f + 1, i_end + 2):
                    if i == int(ide) - 1:
                        fqx[(i, k)] = 0.5 * ru_w(i, k, j) * (q(i, k, j) + q(i - 1, k, j))
                    if i == int(ide) - 2:
                        vel = ru_w(i, k, j)
                        fqx[(i, k)] = vel * _flux3(q(i - 2, k, j), q(i - 1, k, j), q(i, k, j), q(i + 1, k, j), vel, time_step)
        for k in range(int(kts) + 1, ktf + 2):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - msftx_v(i, j) * float(rdx) * (fqx[(i + 1, k)] - fqx[(i, k)]))

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    if bool(_read(config_flags, "open_xs", False)) and int(its) == int(ids):
        for j in range(j_start, j_end + 1):
            for k in range(int(kts) + 1, ktf + 2):
                uw = 0.5 * (ru_w(its, k, j) + ru_w(int(its) + 1, k, j))
                ub = min(float(uw), 0.0)
                div = ru_w(int(its) + 1, k, j) - ru_w(its, k, j)
                set_t(its, k, j, t(its, k, j) - float(rdx) * (ub * (qold(int(its) + 1, k, j) - qold(its, k, j)) + q(its, k, j) * div))

    if bool(_read(config_flags, "open_xe", False)) and int(ite) == int(ide):
        for j in range(j_start, j_end + 1):
            for k in range(int(kts) + 1, ktf + 2):
                uw = 0.5 * (ru_w(int(ite) - 1, k, j) + ru_w(ite, k, j))
                ub = max(float(uw), 0.0)
                div = ru_w(ite, k, j) - ru_w(int(ite) - 1, k, j)
                set_t(i_end, k, j, t(i_end, k, j) - float(rdx) * (ub * (qold(i_end, k, j) - qold(i_end - 1, k, j)) + q(i_end, k, j) * div))

    if bool(_read(config_flags, "open_ys", False)) and int(jts) == int(jds):
        for i in range(i_start, i_end + 1):
            for k in range(int(kts) + 1, ktf + 2):
                vw = 0.5 * (rv_w(i, k, jts) + rv_w(i, k, int(jts) + 1))
                vb = min(float(vw), 0.0)
                div = rv_w(i, k, int(jts) + 1) - rv_w(i, k, jts)
                set_t(i, k, jts, t(i, k, jts) - float(rdy) * (vb * (qold(i, k, int(jts) + 1) - qold(i, k, jts)) + q(i, k, jts) * div))

    if bool(_read(config_flags, "open_ye", False)) and int(jte) == int(jde):
        for i in range(i_start, i_end + 1):
            for k in range(int(kts) + 1, ktf + 2):
                vw = 0.5 * (rv_w(i, k, int(jte) - 1) + rv_w(i, k, jte))
                vb = max(float(vw), 0.0)
                div = rv_w(i, k, jte) - rv_w(i, k, int(jte) - 1)
                set_t(i, k, j_end, t(i, k, j_end) - float(rdy) * (vb * (qold(i, k, j_end) - qold(i, k, j_end - 1)) + q(i, k, j_end) * div))

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    for j in range(j_start, j_end + 1):
        vflux: dict[tuple[int, int], Any] = {}
        for i in range(i_start, i_end + 1):
            vflux[(i, int(kts))] = 0.0
            vflux[(i, int(kte))] = 0.0
        for k in range(int(kts) + 2, ktf + 1):
            for i in range(i_start, i_end + 1):
                vel = 0.5 * (rom_v(i, k, j) + rom_v(i, k - 1, j))
                vflux[(i, k)] = vel * _flux3(q(i, k - 2, j), q(i, k - 1, j), q(i, k, j), q(i, k + 1, j), -vel, time_step)
        for i in range(i_start, i_end + 1):
            for k in (int(kts) + 1, ktf + 1):
                vflux[(i, k)] = 0.25 * (rom_v(i, k, j) + rom_v(i, k - 1, j)) * (q(i, k, j) + q(i, k - 1, j))
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                set_t(i, k, j, t(i, k, j) - rdzu_v(k) * (vflux[(i, k + 1)] - vflux[(i, k)]))
        k = ktf + 1
        for i in range(i_start, i_end + 1):
            set_t(i, k, j, t(i, k, j) + 2.0 * rdzu_v(k - 1) * vflux[(i, k)])

    return tendency_out


__all__ = ["advect_scalar", "advect_scalar_pd", "advect_u", "advect_v", "advect_w"]
