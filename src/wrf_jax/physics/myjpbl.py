"""Translations from ``module_bl_myjpbl``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.column_scan import scan_columns
from wrf_jax.core.elementwise import map_elements
from wrf_jax.core.optimization import semantic_call
from wrf_jax.core.real_arithmetic import column_binary, real_assignment, real_divide_assigned
from wrf_jax.core.wrf_math import (
    clipped_sqrt_for_wrf as _clipped_sqrt,
    sqrt_for_wrf,
    wrf_glibc_expf_finite,
)


EPSQ2 = np.float32(0.20000000298023224)
EPS1 = np.float32(9.999999960041972e-13)
EPS2 = np.float32(0.0)
EPSTRB = np.float32(1.0000000195414814e-24)
G = np.float32(9.8100004196167)
R_D = np.float32(287.0)
CP = np.float32(1004.5)
R_V = np.float32(461.6000061035156)
XLS = np.float32(2850000.0)
XLV = np.float32(2500000.0)
RVOVRD = np.float32(1.6083624362945557)
P608 = np.float32(0.6083624362945557)
CAPA = np.float32(0.2857142984867096)
PQ0 = np.float32(379.9051513671875)
A2 = np.float32(17.26938819885254)
A3 = np.float32(273.1600036621094)
A4 = np.float32(35.86000061035156)
ELOCP = np.float32(2707.81494140625)
BETA = np.float32(0.0036630036775022745)
BTG = np.float32(0.03593406826257706)
VKARMAN = np.float32(0.4)
EPSL = np.float32(0.3199999928474426)
EPSRU = np.float32(1.0000000116860974e-7)
FH = np.float32(1.0099999904632568)
ALPH = np.float32(0.30000001192092896)
EL0MAX = np.float32(1000.0)
EL0MIN = np.float32(1.0)
ELFC = np.float32(0.11500000208616257)
ESQ = np.float32(5.0)
ELZ0 = np.float32(0.0)
A1 = np.float32(0.659888505935669)
A2X = np.float32(0.6574209928512573)
B1 = np.float32(11.8779935836792)
B2 = np.float32(7.226971626281738)
C1 = np.float32(0.0008309559780173004)
ADNH = np.float32(0.09810666739940643)
ADNM = np.float32(0.9729986786842346)
ANMH = np.float32(-0.0033144615590572357)
ANMM = np.float32(0.2444441020488739)
BDNH = np.float32(0.8395565152168274)
BDNM = np.float32(2.6127171516418457)
BEQH = np.float32(1.1201599836349487)
BEQM = np.float32(-5.205894947052002)
BNMH = np.float32(-0.023623811081051826)
BNMM = np.float32(0.6582434773445129)
BSHH = np.float32(0.09223730117082596)
BSHM = np.float32(0.0042818873189389706)
BSMH = np.float32(0.244597926735878)
CESH = np.float32(0.6574209928512573)
CESM = np.float32(0.6582434773445129)
AEQH = np.float32(0.13747581839561462)
AEQM = np.float32(-1.930506944656372)
REQU = np.float32(0.0712122917175293)
EPSGH = np.float32(9.999999717180685e-10)
EPSGM = np.float32(7.12122930068837e-11)
UBRYL = np.float32(0.14356787502765656)
UBRY = np.float32(0.14356788992881775)
UBRY3 = np.float32(0.43070366978645325)
AUBH = np.float32(0.02960565686225891)
AUBM = np.float32(-0.41573813557624817)
BUBH = np.float32(0.2908875048160553)
BUBM = np.float32(-1.1187937259674072)
CUBR = np.float32(0.5692963600158691)
RCUBR = np.float32(1.756554365158081)
RB1 = np.float32(0.08418930321931839)
RLIVWV = np.float32(1.1399999856948853)
SEAFC = np.float32(0.9800000190734863)
PQ0SEA = np.float32(372.30706787109375)


def _contains_jax_tracer(values: Any) -> bool:
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(values)
    )


def _call_tuple_kernel(function: Any, *args: Any) -> tuple[Any, ...]:
    """Keep a validated MYJ subroutine distinct inside an outer trace."""

    if _contains_jax_tracer(args):
        return semantic_call(function, *args)
    return tuple(function(*args))


def _call_single_kernel(function: Any, *args: Any) -> Any:
    """Single-result counterpart of :func:`_call_tuple_kernel`."""

    if _contains_jax_tracer(args):
        return semantic_call(function, *args)
    return function(*args)


def _myj_real(value: Any, rounding_token: Any) -> Any:
    """Materialize one WRF default-REAL arithmetic result inside XLA."""

    return real_assignment(value, rounding_token)


def _myj_double(value: Any) -> Any:
    """Evaluate one uncontracted REAL operation before rounding to binary32."""

    return jnp.asarray(value, dtype=jnp.float64)


def _myj_real_add(left: Any, right: Any, rounding_token: Any) -> Any:
    return _myj_real(_myj_double(left) + _myj_double(right), rounding_token)


def _myj_real_sub(left: Any, right: Any, rounding_token: Any) -> Any:
    return _myj_real(_myj_double(left) - _myj_double(right), rounding_token)


def _myj_real_mul(left: Any, right: Any, rounding_token: Any) -> Any:
    return _myj_real(_myj_double(left) * _myj_double(right), rounding_token)


def _myj_real_div(left: Any, right: Any, rounding_token: Any) -> Any:
    # Division cannot contract with a neighboring operation. Keep the divide
    # itself binary32 while retaining explicit stores around it.
    left, right = jnp.broadcast_arrays(left, right)
    return real_divide_assigned(left, right, rounding_token)


def _myj_real_neg(value: Any, rounding_token: Any) -> Any:
    return _myj_real(-_myj_double(value), rounding_token)


def _myj_real_exp(value: Any, rounding_token: Any) -> Any:
    """Match the scalar ``expf`` called by the WRF reference build."""

    return _myj_real(wrf_glibc_expf_finite(value), rounding_token)


@dataclass(frozen=True)
class MyjPblInitResult:
    rublten: Any
    rvblten: Any
    rthblten: Any
    rqvblten: Any
    tke_myj: Any
    exch_h: Any


@dataclass(frozen=True)
class MyjPblResult:
    qsfc: Any
    thz0: Any
    qz0: Any
    uz0: Any
    vz0: Any
    tke_myj: Any
    exch_h: Any
    ustar: Any
    znt: Any
    el_myj: Any
    pblh: Any
    kpbl: Any
    ct: Any
    akhs: Any
    akms: Any
    mixht: Any
    rublten: Any
    rvblten: Any
    rthblten: Any
    rqvblten: Any
    rqcblten: Any | None = None
    rqiblten: Any | None = None
    rqsblten: Any | None = None
    rqrblten: Any | None = None
    rqgblten: Any | None = None


@jax.jit
def _batched_mixlen_full_lmh(
    u: Any,
    v: Any,
    t: Any,
    the: Any,
    q: Any,
    cwm: Any,
    q2: Any,
    z: Any,
    ct: Any,
    rounding_token: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Vectorized WRF ``MIXLEN`` for columns where ``lmh`` spans all levels."""

    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    t_arr = jnp.asarray(t)
    the_arr = jnp.asarray(the)
    q_arr = jnp.asarray(q)
    cwm_arr = jnp.asarray(cwm)
    q2_arr = jnp.asarray(q2)
    z_arr = jnp.asarray(z)
    ct_arr = jnp.asarray(ct, dtype=q2_arr.dtype)
    ncol, nk = q2_arr.shape
    nkm1 = nk - 1
    store = lambda value: _myj_real(value, rounding_token)
    real_add = lambda left, right: _myj_real_add(left, right, rounding_token)
    real_sub = lambda left, right: _myj_real_sub(left, right, rounding_token)
    real_mul = lambda left, right: _myj_real_mul(left, right, rounding_token)
    real_div = lambda left, right: _myj_real_div(left, right, rounding_token)

    k_values = jnp.arange(1, nk, dtype=jnp.int32)
    weak_tke = q2_arr[:, :nkm1] <= jnp.asarray(EPSQ2 * FH, dtype=q2_arr.dtype)
    any_weak = jnp.any(weak_tke, axis=1)
    lpbl = jnp.where(any_weak, jnp.max(jnp.where(weak_tke, k_values[None, :], 0), axis=1), 1)

    z_lpbl_plus_1 = jnp.take_along_axis(z_arr, lpbl[:, None], axis=1)[:, 0]
    pblh = z_lpbl_plus_1 - z_arr[:, nk]

    dth = real_sub(the_arr[:, :nkm1], the_arr[:, 1:nk])
    if nk > 2:
        dth_turn = (dth[:, : nk - 2] > 0.0) & (dth[:, 1:nkm1] <= 0.0)
        turn_values = jnp.arange(1, nk - 1, dtype=jnp.int32)
        chosen_turn = jnp.max(jnp.where(dth_turn, turn_values[None, :], 0), axis=1)
        dth = real_add(
            dth,
            jnp.where(k_values[None, :] == chosen_turn[:, None], ct_arr[:, None], np.float32(0.0)),
        )
    ct_out = jnp.zeros_like(ct_arr)

    rdz = real_div(np.float32(2.0), real_sub(z_arr[:, :nkm1], z_arr[:, 2 : nk + 1]))
    du = real_sub(u_arr[:, :nkm1], u_arr[:, 1:nk])
    dv = real_sub(v_arr[:, :nkm1], v_arr[:, 1:nk])
    gml = real_add(real_mul(du, du), real_mul(dv, dv))
    gml = real_mul(real_mul(gml, rdz), rdz)
    gm = store(jnp.maximum(gml, EPSGM))
    tem = real_mul(real_add(t_arr[:, :nkm1], t_arr[:, 1:nk]), np.float32(0.5))
    thm = real_mul(real_add(the_arr[:, :nkm1], the_arr[:, 1:nk]), np.float32(0.5))
    a = real_mul(thm, P608)
    b = real_mul(
        real_sub(real_sub(real_div(ELOCP, tem), np.float32(1.0)), P608),
        thm,
    )
    qsum = real_add(
        real_add(real_add(q_arr[:, :nkm1], q_arr[:, 1:nk]), cwm_arr[:, :nkm1]),
        cwm_arr[:, 1:nk],
    )
    qfactor = real_add(
        real_mul(qsum, real_mul(np.float32(0.5), P608)),
        np.float32(1.0),
    )
    qgradient = real_sub(
        real_add(real_sub(q_arr[:, :nkm1], q_arr[:, 1:nk]), cwm_arr[:, :nkm1]),
        cwm_arr[:, 1:nk],
    )
    cwm_gradient = real_sub(cwm_arr[:, :nkm1], cwm_arr[:, 1:nk])
    ghl = real_add(
        real_add(real_mul(dth, qfactor), real_mul(qgradient, a)),
        real_mul(cwm_gradient, b),
    )
    ghl = real_mul(ghl, rdz)
    gh = store(jnp.where(jnp.abs(ghl) <= EPSGH, jnp.asarray(EPSGH, dtype=q2_arr.dtype), ghl))

    stable = (gh >= EPSGH) & (gm / gh <= REQU)

    aubr = real_mul(real_add(real_mul(AUBM, gm), real_mul(AUBH, gh)), gh)
    bubr = real_add(real_mul(BUBM, gm), real_mul(BUBH, gh))
    disc_st = real_sub(real_mul(real_mul(bubr, bubr), np.float32(0.25)), real_mul(aubr, CUBR))
    sqrt_st = store(_clipped_sqrt(disc_st))
    qol2st = real_mul(real_add(real_mul(np.float32(-0.5), bubr), sqrt_st), RCUBR)
    eloq2x_st = real_div(np.float32(1.0), jnp.where(qol2st != 0.0, qol2st, np.float32(1.0)))
    elm_st = store(jnp.maximum(store(_clipped_sqrt(real_mul(eloq2x_st, q2_arr[:, :nkm1]))), EPSL))

    aden = real_mul(real_add(real_mul(ADNM, gm), real_mul(ADNH, gh)), gh)
    bden = real_add(real_mul(BDNM, gm), real_mul(BDNH, gh))
    disc_un = real_sub(real_mul(real_mul(bden, bden), np.float32(0.25)), aden)
    sqrt_un = store(_clipped_sqrt(disc_un))
    qol2un = real_add(real_mul(np.float32(-0.5), bden), sqrt_un)
    eloq2x_un = real_div(np.float32(1.0), real_add(qol2un, EPSRU))
    elm_un = store(jnp.maximum(store(_clipped_sqrt(real_mul(eloq2x_un, q2_arr[:, :nkm1]))), EPSL))

    elm = jnp.where(stable, EPSL, jnp.where(gh >= EPSGH, elm_st, elm_un))
    last_stable = jnp.max(jnp.where(stable, k_values[None, :], 0), axis=1)
    lmxl = jnp.where(last_stable > 0, last_stable, nk)

    lmxl = jnp.where(elm[:, nk - 2] == jnp.asarray(EPSL, dtype=elm.dtype), nk, lmxl)
    z_lmxl = jnp.take_along_axis(z_arr, (lmxl - 1)[:, None], axis=1)[:, 0]
    mixht = real_sub(z_lmxl, z_arr[:, nk])

    q1_active = store(jnp.where(
        jnp.arange(1, nk + 1, dtype=jnp.int32)[None, :] >= lpbl[:, None],
        store(sqrt_for_wrf(q2_arr)),
        np.float32(0.0),
    ))
    q1 = jnp.concatenate([q1_active, jnp.zeros((ncol, 1), dtype=q2_arr.dtype)], axis=1)

    qdzl = real_mul(
        real_add(q1[:, :nkm1], q1[:, 1:nk]),
        real_sub(z_arr[:, 1:nk], z_arr[:, 2 : nk + 1]),
    )
    zsum = real_sub(
        real_sub(real_add(z_arr[:, 1:nk], z_arr[:, 2 : nk + 1]), z_arr[:, nk : nk + 1]),
        z_arr[:, nk : nk + 1],
    )
    szq_terms = real_mul(zsum, qdzl)

    def sum_step(
        carry: tuple[Any, Any], terms: tuple[Any, Any], token: Any,
    ) -> tuple[tuple[Any, Any], None]:
        szq_acc, sq_acc = carry
        szq_term, sq_term = terms
        return (
            _myj_real_add(szq_term, szq_acc, token),
            _myj_real_add(sq_term, sq_acc, token),
        ), None

    (szq, sq), _ = scan_columns(
        sum_step,
        (jnp.zeros((ncol,), dtype=q2_arr.dtype), jnp.zeros((ncol,), dtype=q2_arr.dtype)),
        (jnp.swapaxes(szq_terms, 0, 1), jnp.swapaxes(qdzl, 0, 1)),
        rounding_token,
    )

    sq_safe = jnp.where(sq != 0.0, sq, 1.0)
    el0 = real_div(real_mul(real_mul(ALPH, szq), np.float32(0.5)), sq_safe)
    el0 = store(jnp.minimum(el0, EL0MAX))
    el0 = store(jnp.maximum(el0, EL0MIN))

    lpblm = jnp.maximum(lpbl - 1, 1)
    value_low = store(jnp.minimum(real_mul(real_sub(z_arr[:, :nkm1], z_arr[:, 2 : nk + 1]), ELFC), elm))
    mask_low = k_values[None, :] <= lpblm[:, None]
    el = jnp.where(mask_low, value_low, jnp.zeros((ncol, nkm1), dtype=q2_arr.dtype))
    rel = jnp.where(mask_low, real_div(value_low, elm), jnp.zeros((ncol, nkm1), dtype=q2_arr.dtype))

    vkrmz = real_mul(real_sub(z_arr[:, 1:nk], z_arr[:, nk : nk + 1]), VKARMAN)
    value_high = store(jnp.minimum(
        real_div(vkrmz, real_add(real_div(vkrmz, el0[:, None]), np.float32(1.0))),
        elm,
    ))
    mask_high = (lpbl[:, None] < nk) & (k_values[None, :] >= lpbl[:, None])
    el = jnp.where(mask_high, value_high, el)
    rel = jnp.where(mask_high, real_div(value_high, elm), rel)

    if nk > 3:
        srel_mid = real_mul(real_add(
            real_mul(real_add(rel[:, :-2], rel[:, 2:]), np.float32(0.5)),
            rel[:, 1:-1],
        ), np.float32(0.5))
        srel_mid = store(jnp.minimum(srel_mid, rel[:, 1:-1]))
        mid_k_values = jnp.arange(2, nk - 1, dtype=jnp.int32)
        mask_mid = (mid_k_values[None, :] >= lpbl[:, None] + 1) & (mid_k_values[None, :] <= nk - 2)
        el_mid = jnp.where(mask_mid, jnp.maximum(real_mul(srel_mid, elm[:, 1:-1]), EPSL), el[:, 1:-1])
        el = jnp.concatenate([el[:, :1], el_mid, el[:, -1:]], axis=1)

    return gm, gh, el, pblh, lpbl, lmxl, ct_out, mixht


def _batched_prodq2_full_lmh_impl(
    dtturbl: Any,
    ustar: Any,
    gm: Any,
    gh: Any,
    el: Any,
    q2: Any,
    rounding_token: Any,
) -> tuple[Any, Any]:
    """Vectorized WRF ``PRODQ2`` for columns where ``lmh`` spans all levels."""

    store = lambda value: _myj_real(value, rounding_token)
    real_add = lambda left, right: _myj_real_add(left, right, rounding_token)
    real_sub = lambda left, right: _myj_real_sub(left, right, rounding_token)
    real_mul = lambda left, right: _myj_real_mul(left, right, rounding_token)
    real_div = lambda left, right: _myj_real_div(left, right, rounding_token)
    real_neg = lambda value: _myj_real_neg(value, rounding_token)
    real_exp = lambda value: _myj_real_exp(value, rounding_token)

    gm_arr = store(gm)
    gh_arr = store(gh)
    el_arr = store(el)
    q2_arr = store(q2)
    ncol, nk = q2_arr.shape
    del ncol

    gml = store(gm_arr)
    ghl = store(gh_arr)
    aequ = real_mul(
        real_add(real_mul(AEQM, gml), real_mul(AEQH, ghl)),
        ghl,
    )
    bequ = real_add(real_mul(BEQM, gml), real_mul(BEQH, ghl))
    eq_disc = real_sub(real_mul(real_mul(bequ, bequ), np.float32(0.25)), aequ)
    eqol2 = real_add(
        real_mul(np.float32(-0.5), bequ),
        store(sqrt_for_wrf(eq_disc)),
    )
    bad = (
        real_add(gml, real_mul(ghl, ghl)) <= EPSTRB
    ) | (
        (ghl >= EPSGH) & (real_div(gml, ghl) <= REQU)
    ) | (eqol2 <= EPS2)

    eqol2_safe = store(jnp.where(bad, np.float32(1.0), eqol2))
    anum = real_mul(
        real_add(real_mul(ANMM, gml), real_mul(ANMH, ghl)),
        ghl,
    )
    bnum = real_add(real_mul(BNMM, gml), real_mul(BNMH, ghl))
    aden = real_mul(
        real_add(real_mul(ADNM, gml), real_mul(ADNH, ghl)),
        ghl,
    )
    bden = real_add(real_mul(BDNM, gml), real_mul(BDNH, ghl))
    cden = store(np.float32(1.0))
    arhs = real_mul(
        real_neg(real_sub(real_mul(anum, bden), real_mul(bnum, aden))),
        np.float32(2.0),
    )
    brhs = real_mul(real_neg(anum), np.float32(4.0))
    crhs = real_mul(real_neg(bnum), np.float32(2.0))
    dloq1 = real_div(el_arr, store(sqrt_for_wrf(q2_arr[:, : nk - 1])))
    eloq21 = real_div(np.float32(1.0), eqol2_safe)
    eloq11 = store(sqrt_for_wrf(eloq21))
    eloq31 = real_mul(eloq21, eloq11)
    eloq41 = real_mul(eloq21, eloq21)
    eloq51 = real_mul(eloq21, eloq31)
    rden1 = real_div(
        np.float32(1.0),
        real_add(
            real_add(real_mul(aden, eloq41), real_mul(bden, eloq21)),
            cden,
        ),
    )
    rhsp1 = real_mul(
        real_mul(
            real_add(
                real_add(real_mul(arhs, eloq51), real_mul(brhs, eloq31)),
                real_mul(crhs, eloq11),
            ),
            rden1,
        ),
        rden1,
    )
    eloq12 = real_add(
        eloq11,
        real_mul(
            real_sub(dloq1, eloq11),
            real_exp(real_mul(rhsp1, dtturbl)),
        ),
    )
    eloq12 = store(jnp.maximum(eloq12, EPS1))
    eloq22 = real_mul(eloq12, eloq12)
    eloq32 = real_mul(eloq22, eloq12)
    eloq42 = real_mul(eloq22, eloq22)
    eloq52 = real_mul(eloq22, eloq32)
    rden2 = real_div(
        np.float32(1.0),
        real_add(
            real_add(real_mul(aden, eloq42), real_mul(bden, eloq22)),
            cden,
        ),
    )
    rhs2 = real_add(
        real_mul(
            real_neg(
                real_add(real_mul(anum, eloq42), real_mul(bnum, eloq22))
            ),
            rden2,
        ),
        RB1,
    )
    rhsp2 = real_mul(
        real_mul(
            real_add(
                real_add(real_mul(arhs, eloq52), real_mul(brhs, eloq32)),
                real_mul(crhs, eloq12),
            ),
            rden2,
        ),
        rden2,
    )
    rhst2 = real_div(rhs2, rhsp2)
    eloq13 = real_add(
        real_sub(eloq12, rhst2),
        real_mul(
            real_sub(real_add(rhst2, dloq1), eloq12),
            real_exp(real_mul(rhsp2, dtturbl)),
        ),
    )
    eloq13 = store(jnp.maximum(eloq13, EPS1))
    eloqn = store(eloq13)
    good_eloq = eloqn > EPS1
    q2_new = store(
        jnp.maximum(
            real_div(real_mul(el_arr, el_arr), real_mul(eloqn, eloqn)),
            EPSQ2,
        )
    )
    q2_value = store(jnp.where(bad | (~good_eloq), EPSQ2, q2_new))
    el_out = store(jnp.where(
        bad | (~good_eloq) | (q2_value == jnp.asarray(EPSQ2, dtype=q2_value.dtype)),
        EPSL,
        el_arr,
    ))

    q2_surface = store(
        jnp.maximum(
            real_mul(
                real_mul(
                    store(jnp.power(B1, np.float32(2.0) / np.float32(3.0))),
                    ustar,
                ),
                ustar,
            ),
            EPSQ2,
        )
    )
    q2_out = jnp.concatenate([q2_value, q2_surface[:, None]], axis=1)
    return el_out, q2_out


_batched_prodq2_full_lmh = jax.jit(_batched_prodq2_full_lmh_impl)


def _batched_difcof_full_lmh_impl(
    gm: Any,
    gh: Any,
    el: Any,
    q2: Any,
    z: Any,
    rounding_token: Any,
) -> tuple[Any, Any]:
    """Vectorized WRF ``DIFCOF`` for columns where ``lmh`` spans all levels."""

    store = lambda value: _myj_real(value, rounding_token)
    real_add = lambda left, right: _myj_real_add(left, right, rounding_token)
    real_sub = lambda left, right: _myj_real_sub(left, right, rounding_token)
    real_mul = lambda left, right: _myj_real_mul(left, right, rounding_token)
    real_div = lambda left, right: _myj_real_div(left, right, rounding_token)

    gm_arr = store(gm)
    gh_arr = store(gh)
    el_arr = store(el)
    q2_arr = store(q2)
    z_arr = store(z)
    _, nkm1 = gm_arr.shape
    ell = store(el_arr)
    eloq2 = real_div(real_mul(ell, ell), q2_arr[:, :nkm1])
    eloq4 = real_mul(eloq2, eloq2)
    gml = store(gm_arr)
    ghl = store(gh_arr)
    aden = real_mul(
        real_add(real_mul(ADNM, gml), real_mul(ADNH, ghl)),
        ghl,
    )
    bden = real_add(real_mul(BDNM, gml), real_mul(BDNH, ghl))
    cden = store(np.float32(1.0))
    besm = real_mul(BSMH, ghl)
    besh = real_add(real_mul(BSHM, gml), real_mul(BSHH, ghl))
    rden = real_div(
        np.float32(1.0),
        real_add(
            real_add(real_mul(aden, eloq4), real_mul(bden, eloq2)),
            cden,
        ),
    )
    esm = real_mul(real_add(real_mul(besm, eloq2), CESM), rden)
    esh = real_mul(real_add(real_mul(besh, eloq2), CESH), rden)
    rdz = real_div(
        np.float32(2.0),
        real_sub(z_arr[:, :nkm1], z_arr[:, 2 : nkm1 + 2]),
    )
    q1l = store(sqrt_for_wrf(q2_arr[:, :nkm1]))
    elqdz = real_mul(real_mul(ell, q1l), rdz)
    akm = real_mul(elqdz, esm)
    akh = real_mul(elqdz, esh)

    return akh, akm


_batched_difcof_full_lmh = jax.jit(_batched_difcof_full_lmh_impl)


def _turbulence_lower_boundary(inputs, parameter):
    del parameter
    dtozs, akqs, akq, cm, cr, rsq2, bottom, surface = inputs
    cf = column_binary("div", -dtozs * akq, cm)
    numerator = dtozs * akqs * surface - rsq2 * cf + bottom
    denominator = (akq + akqs) * dtozs - cr * cf + 1.0
    return column_binary("div", numerator, denominator)


@jax.jit
def _batched_vdifq_full_lmh(dtdif: Any, q2: Any, el: Any, z: Any) -> Any:
    """Vectorized WRF ``VDIFQ`` for columns where ``lmh`` spans all levels."""

    q2_out = jnp.asarray(q2)
    el_arr = jnp.asarray(el)
    z_arr = jnp.asarray(z)
    ncol, nk = q2_out.shape
    del ncol
    esqhf = 0.5 * ESQ

    dtoz = (dtdif + dtdif) / (z_arr[:, : nk - 2] - z_arr[:, 2:nk])
    akq = (
        sqrt_for_wrf((q2_out[:, : nk - 2] + q2_out[:, 1 : nk - 1]) * 0.5)
        * (el_arr[:, : nk - 2] + el_arr[:, 1 : nk - 1])
        * esqhf
        / (z_arr[:, 1 : nk - 1] - z_arr[:, 2:nk])
    )
    cr = -dtoz * akq

    cm0 = dtoz[:, 0] * akq[:, 0] + 1.0
    rsq20 = q2_out[:, 0]

    def forward(carry: tuple[Any, Any], inputs: tuple[Any, ...], parameter: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        del parameter
        cm_prev, rsq2_prev = carry
        dtozl, akq_prev, akq_curr, cr_prev, q2_curr = inputs
        cf = column_binary("div", -dtozl * akq_prev, cm_prev)
        cm_curr = -cr_prev * cf + (akq_prev + akq_curr) * dtozl + 1.0
        rsq2_curr = -rsq2_prev * cf + q2_curr
        return (cm_curr, rsq2_curr), (cm_curr, rsq2_curr)

    _, (cm_tail, rsq2_tail) = scan_columns(
        forward,
        (cm0, rsq20),
        (
            jnp.swapaxes(dtoz[:, 1 : nk - 2], 0, 1),
            jnp.swapaxes(akq[:, : nk - 3], 0, 1),
            jnp.swapaxes(akq[:, 1 : nk - 2], 0, 1),
            jnp.swapaxes(cr[:, : nk - 3], 0, 1),
            jnp.swapaxes(q2_out[:, 1 : nk - 2], 0, 1),
        ),
        None,
    )
    cm = jnp.concatenate([cm0[:, None], jnp.swapaxes(cm_tail, 0, 1)], axis=1)
    rsq2 = jnp.concatenate([rsq20[:, None], jnp.swapaxes(rsq2_tail, 0, 1)], axis=1)

    dtozs = (dtdif + dtdif) / (z_arr[:, nk - 2] - z_arr[:, nk])
    akqs = (
        sqrt_for_wrf((q2_out[:, nk - 2] + q2_out[:, nk - 1]) * 0.5)
        * (el_arr[:, nk - 2] + ELZ0)
        * esqhf
        / (z_arr[:, nk - 1] - z_arr[:, nk])
    )
    # The boundary must not change fusion when either adjacent sweep is packed.
    q2_lmh_m1 = map_elements(_turbulence_lower_boundary, (
        dtozs, akqs, akq[:, nk - 3], cm[:, nk - 3], cr[:, nk - 3],
        rsq2[:, nk - 3], q2_out[:, nk - 2], q2_out[:, nk - 1],
    ))

    def backward(carry: Any, inputs: tuple[Any, Any, Any], parameter: Any) -> tuple[Any, Any]:
        del parameter
        q2_next = carry
        cm_curr, cr_curr, rsq2_curr = inputs
        q2_curr = column_binary("div", -cr_curr * q2_next + rsq2_curr, cm_curr)
        return q2_curr, q2_curr

    _, q2_rev = scan_columns(
        backward,
        q2_lmh_m1,
        (
            jnp.swapaxes(cm[:, ::-1], 0, 1),
            jnp.swapaxes(cr[:, ::-1], 0, 1),
            jnp.swapaxes(rsq2[:, ::-1], 0, 1),
        ),
        None,
    )
    q2_lower = jnp.swapaxes(q2_rev[::-1], 0, 1)
    q2_out = jnp.concatenate([q2_lower, q2_lmh_m1[:, None], q2_out[:, nk - 1 : nk]], axis=1)

    return q2_out


@jax.jit
def _batched_vdifh_active_full_lmh(
    dtdif: Any,
    lpbl: Any,
    sz0: Any,
    rkhs: Any,
    clow: Any,
    cts: Any,
    active_species: Any,
    rkh: Any,
    zhk: Any,
    rho: Any,
    rounding_token: Any,
) -> Any:
    """Vectorized WRF ``VDIFH`` for columns where ``lmh`` spans all levels."""

    sz0_arr = jnp.asarray(sz0)
    clow_arr = jnp.asarray(clow)
    cts_arr = jnp.asarray(cts)
    active_species = jnp.asarray(active_species)
    rkh_arr = jnp.asarray(rkh)
    zhk_arr = jnp.asarray(zhk)
    rho_arr = jnp.asarray(rho)
    _, nspec, nk = active_species.shape

    store = lambda value: _myj_real(value, rounding_token)
    real_add = lambda left, right: _myj_real_add(left, right, rounding_token)
    real_sub = lambda left, right: _myj_real_sub(left, right, rounding_token)
    real_mul = lambda left, right: _myj_real_mul(left, right, rounding_token)
    real_div = lambda left, right: _myj_real_div(left, right, rounding_token)
    real_neg = lambda value: _myj_real_neg(value, rounding_token)

    dtoz = real_div(
        dtdif,
        real_sub(zhk_arr[:, : nk - 1], zhk_arr[:, 1:nk]),
    )
    cr = real_mul(real_neg(dtoz), rkh_arr[:, : nk - 1])
    rkhz = real_mul(
        rkh_arr[:, : nk - 1],
        real_sub(zhk_arr[:, : nk - 1], zhk_arr[:, 2 : nk + 1]),
    )
    active = (jnp.arange(1, nk, dtype=jnp.int32)[None, :] >= jnp.asarray(lpbl)[:, None])
    rkct = store(
        jnp.where(
            active[:, None, :],
            real_mul(
                real_mul(rkhz[:, None, :], cts_arr[:, :nspec, None]),
                np.float32(0.5),
            ),
            np.float32(0.0),
        )
    )

    rhok = store(rho_arr[:, 0])
    cm0 = real_add(real_mul(dtoz[:, 0], rkh_arr[:, 0]), rhok)
    rss0 = real_add(
        real_mul(real_neg(rkct[:, :, 0]), dtoz[:, 0, None]),
        real_mul(active_species[:, :, 0], rhok[:, None]),
    )

    def forward(carry: tuple[Any, Any], inputs: tuple[Any, ...], token: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        store = lambda value: _myj_real(value, token)
        real_add = lambda left, right: _myj_real_add(left, right, token)
        real_sub = lambda left, right: _myj_real_sub(left, right, token)
        real_mul = lambda left, right: _myj_real_mul(left, right, token)
        real_div = lambda left, right: _myj_real_div(left, right, token)
        real_neg = lambda value: _myj_real_neg(value, token)
        cm_prev, rss_prev = carry
        dtozl, rkh_prev, rkh_curr, cr_prev, rho_curr, rkct_prev, rkct_curr, species_curr = inputs
        dtozl = store(dtozl)
        cf = real_div(real_mul(real_neg(dtozl), rkh_prev), cm_prev)
        cm_curr = real_add(
            real_add(
                real_mul(real_neg(cr_prev), cf),
                real_mul(real_add(rkh_prev, rkh_curr), dtozl),
            ),
            rho_curr,
        )
        rss_curr = real_add(
            real_add(
                real_mul(real_neg(rss_prev), cf),
                real_mul(real_sub(rkct_prev, rkct_curr), dtozl),
            ),
            real_mul(species_curr, rho_curr),
        )
        return (cm_curr, rss_curr), (cm_curr, rss_curr)

    # Each species is an independent column with identical matrix coefficients.
    def species_layers(value: Any) -> Any:
        layers = jnp.swapaxes(value, 0, 1)
        return jnp.broadcast_to(layers[:, :, None], (*layers.shape, nspec))

    _, (cm_tail, rss_tail) = scan_columns(
        forward,
        (jnp.broadcast_to(cm0[:, None], rss0.shape), rss0),
        (
            species_layers(dtoz[:, 1 : nk - 1]),
            species_layers(rkh_arr[:, : nk - 2]),
            species_layers(rkh_arr[:, 1 : nk - 1]),
            species_layers(cr[:, : nk - 2]),
            species_layers(rho_arr[:, 1 : nk - 1]),
            jnp.moveaxis(rkct[:, :, : nk - 2], 2, 0),
            jnp.moveaxis(rkct[:, :, 1 : nk - 1], 2, 0),
            jnp.moveaxis(active_species[:, :, 1 : nk - 1], 2, 0),
        ),
        rounding_token,
    )
    cm_tail = cm_tail[:, :, 0]
    cm = jnp.concatenate([cm0[:, None], jnp.swapaxes(cm_tail, 0, 1)], axis=1)
    rss = jnp.concatenate([rss0[:, :, None], jnp.moveaxis(rss_tail, 0, 2)], axis=2)

    dtozs = real_div(
        dtdif,
        real_sub(zhk_arr[:, nk - 1], zhk_arr[:, nk]),
    )
    rkhh = store(rkh_arr[:, nk - 2])
    cf = real_div(real_mul(real_neg(dtozs), rkhh), cm[:, nk - 2])
    cmb = real_mul(cr[:, nk - 2], cf)
    rhok = store(rho_arr[:, nk - 1])
    rkss = real_mul(jnp.asarray(rkhs)[:, None], clow_arr[:, :nspec])
    cmsb = real_add(
        real_add(
            real_neg(cmb[:, None]),
            real_mul(real_add(rkhh[:, None], rkss), dtozs[:, None]),
        ),
        rhok[:, None],
    )
    rssb = real_add(
        real_add(
            real_mul(real_neg(rss[:, :, nk - 2]), cf[:, None]),
            real_mul(rkct[:, :, nk - 2], dtozs[:, None]),
        ),
        real_mul(active_species[:, :, nk - 1], rhok[:, None]),
    )
    species_bottom = real_div(
        real_add(
            real_mul(real_mul(dtozs[:, None], rkss), sz0_arr[:, :nspec]),
            rssb,
        ),
        cmsb,
    )

    def backward(carry: Any, inputs: tuple[Any, Any, Any], token: Any) -> tuple[Any, Any]:
        real_add = lambda left, right: _myj_real_add(left, right, token)
        real_mul = lambda left, right: _myj_real_mul(left, right, token)
        real_div = lambda left, right: _myj_real_div(left, right, token)
        real_neg = lambda value: _myj_real_neg(value, token)
        species_next = carry
        cm_curr, cr_curr, rss_curr = inputs
        rcml = real_div(np.float32(1.0), cm_curr)
        species_curr = real_mul(
            real_add(
                real_mul(real_neg(cr_curr), species_next),
                rss_curr,
            ),
            rcml,
        )
        return species_curr, species_curr

    _, species_rev = scan_columns(
        backward,
        species_bottom,
        (
            species_layers(cm[:, ::-1]),
            species_layers(cr[:, ::-1]),
            jnp.moveaxis(rss[:, :, ::-1], 2, 0),
        ),
        rounding_token,
    )
    species_lower = jnp.moveaxis(species_rev[::-1], 0, 2)
    return jnp.concatenate([species_lower, species_bottom[:, :, None]], axis=2)


@jax.jit
def _batched_vdifv_full_lmh(
    dtdif: Any,
    uz0: Any,
    vz0: Any,
    rkms: Any,
    u: Any,
    v: Any,
    rkm: Any,
    z: Any,
    rho: Any,
    rounding_token: Any,
) -> tuple[Any, Any]:
    """Vectorized WRF ``VDIFV`` for columns where ``lmh`` spans all levels."""

    store = lambda value: _myj_real(value, rounding_token)
    real_add = lambda left, right: _myj_real_add(left, right, rounding_token)
    real_sub = lambda left, right: _myj_real_sub(left, right, rounding_token)
    real_mul = lambda left, right: _myj_real_mul(left, right, rounding_token)
    real_div = lambda left, right: _myj_real_div(left, right, rounding_token)
    real_neg = lambda value: _myj_real_neg(value, rounding_token)

    u_out = store(u)
    v_out = store(v)
    rkm_arr = store(rkm)
    z_arr = store(z)
    rho_arr = store(rho)
    ncol, nk = u_out.shape
    del ncol

    dtoz = real_div(
        dtdif,
        real_sub(z_arr[:, : nk - 1], z_arr[:, 1:nk]),
    )
    cr = real_mul(real_neg(dtoz), rkm_arr[:, : nk - 1])

    rhok = store(rho_arr[:, 0])
    cm0 = real_add(real_mul(dtoz[:, 0], rkm_arr[:, 0]), rhok)
    rsu0 = real_mul(u_out[:, 0], rhok)
    rsv0 = real_mul(v_out[:, 0], rhok)

    def forward(carry: tuple[Any, Any, Any], inputs: tuple[Any, ...], token: Any) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
        store = lambda value: _myj_real(value, token)
        real_add = lambda left, right: _myj_real_add(left, right, token)
        real_mul = lambda left, right: _myj_real_mul(left, right, token)
        real_div = lambda left, right: _myj_real_div(left, right, token)
        real_neg = lambda value: _myj_real_neg(value, token)
        cm_prev, rsu_prev, rsv_prev = carry
        dtozl, rkm_prev, rkm_curr, cr_prev, rho_curr, u_curr, v_curr = inputs
        dtozl = store(dtozl)
        cf = real_div(real_mul(real_neg(dtozl), rkm_prev), cm_prev)
        rhok_curr = store(rho_curr)
        cm_curr = real_add(
            real_add(
                real_mul(real_neg(cr_prev), cf),
                real_mul(real_add(rkm_prev, rkm_curr), dtozl),
            ),
            rhok_curr,
        )
        rsu_curr = real_add(
            real_mul(real_neg(rsu_prev), cf),
            real_mul(u_curr, rhok_curr),
        )
        rsv_curr = real_add(
            real_mul(real_neg(rsv_prev), cf),
            real_mul(v_curr, rhok_curr),
        )
        return (cm_curr, rsu_curr, rsv_curr), (cm_curr, rsu_curr, rsv_curr)

    _, (cm_tail, rsu_tail, rsv_tail) = scan_columns(
        forward,
        (cm0, rsu0, rsv0),
        (
            jnp.swapaxes(dtoz[:, 1 : nk - 1], 0, 1),
            jnp.swapaxes(rkm_arr[:, : nk - 2], 0, 1),
            jnp.swapaxes(rkm_arr[:, 1 : nk - 1], 0, 1),
            jnp.swapaxes(cr[:, : nk - 2], 0, 1),
            jnp.swapaxes(rho_arr[:, 1 : nk - 1], 0, 1),
            jnp.swapaxes(u_out[:, 1 : nk - 1], 0, 1),
            jnp.swapaxes(v_out[:, 1 : nk - 1], 0, 1),
        ),
        rounding_token,
    )
    cm = jnp.concatenate([cm0[:, None], jnp.swapaxes(cm_tail, 0, 1)], axis=1)
    rsu = jnp.concatenate([rsu0[:, None], jnp.swapaxes(rsu_tail, 0, 1)], axis=1)
    rsv = jnp.concatenate([rsv0[:, None], jnp.swapaxes(rsv_tail, 0, 1)], axis=1)

    dtozs = real_div(
        dtdif,
        real_sub(z_arr[:, nk - 1], z_arr[:, nk]),
    )
    rkmh = store(rkm_arr[:, nk - 2])
    cf = real_div(real_mul(real_neg(dtozs), rkmh), cm[:, nk - 2])
    rhok = store(rho_arr[:, nk - 1])
    rcmvb = real_div(
        np.float32(1.0),
        real_add(
            real_sub(
                real_mul(real_add(rkmh, rkms), dtozs),
                real_mul(cr[:, nk - 2], cf),
            ),
            rhok,
        ),
    )
    dtozak = real_mul(dtozs, rkms)
    u_bottom = real_mul(
        real_add(
            real_sub(
                real_mul(dtozak, uz0),
                real_mul(rsu[:, nk - 2], cf),
            ),
            real_mul(u_out[:, nk - 1], rhok),
        ),
        rcmvb,
    )
    v_bottom = real_mul(
        real_add(
            real_sub(
                real_mul(dtozak, vz0),
                real_mul(rsv[:, nk - 2], cf),
            ),
            real_mul(v_out[:, nk - 1], rhok),
        ),
        rcmvb,
    )

    def backward(carry: tuple[Any, Any], inputs: tuple[Any, Any, Any, Any], token: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        real_add = lambda left, right: _myj_real_add(left, right, token)
        real_mul = lambda left, right: _myj_real_mul(left, right, token)
        real_div = lambda left, right: _myj_real_div(left, right, token)
        real_neg = lambda value: _myj_real_neg(value, token)
        u_next, v_next = carry
        cm_curr, cr_curr, rsu_curr, rsv_curr = inputs
        rcmkl = real_div(np.float32(1.0), cm_curr)
        u_curr = real_mul(
            real_add(real_mul(real_neg(cr_curr), u_next), rsu_curr),
            rcmkl,
        )
        v_curr = real_mul(
            real_add(real_mul(real_neg(cr_curr), v_next), rsv_curr),
            rcmkl,
        )
        return (u_curr, v_curr), (u_curr, v_curr)

    _, (u_rev, v_rev) = scan_columns(
        backward,
        (u_bottom, v_bottom),
        (
            jnp.swapaxes(cm[:, ::-1], 0, 1),
            jnp.swapaxes(cr[:, ::-1], 0, 1),
            jnp.swapaxes(rsu[:, ::-1], 0, 1),
            jnp.swapaxes(rsv[:, ::-1], 0, 1),
        ),
        rounding_token,
    )
    u_lower = jnp.swapaxes(u_rev[::-1], 0, 1)
    v_lower = jnp.swapaxes(v_rev[::-1], 0, 1)
    u_out = jnp.concatenate([u_lower, u_bottom[:, None]], axis=1)
    v_out = jnp.concatenate([v_lower, v_bottom[:, None]], axis=1)

    return u_out, v_out


@jax.jit
def _batched_zint_bottom_up(ht: Any, dz: Any) -> Any:
    """Accumulate interface heights in their original bottom-up order."""

    def zint_step(carry: Any, inputs: tuple[Any], parameter: Any) -> tuple[Any, Any]:
        dz_level, = inputs
        z_next = column_binary("add", carry, dz_level)
        return z_next, z_next

    _, zint_bottom_up = scan_columns(zint_step, ht, (jnp.swapaxes(dz, 0, 1),))
    return zint_bottom_up


def _myjpbl_full_lmh_batch_impl(
    dt: Any,
    stepbl: Any,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    exner: Any,
    qv: Any,
    qcw: Any,
    qci: Any,
    u: Any,
    v: Any,
    tsk: Any,
    qsfc: Any,
    chklowq: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    xland: Any,
    sice: Any,
    snow: Any,
    tke_myj: Any,
    exch_h: Any,
    ustar: Any,
    znt: Any,
    el_myj: Any,
    pblh: Any,
    kpbl: Any,
    ct: Any,
    akhs: Any,
    akms: Any,
    elflx: Any,
    mixht: Any,
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rounding_token: Any,
    flqi: bool,
) -> tuple[Any, ...]:
    """Run MYJ over all horizontal columns as one fixed-depth batched kernel."""

    del pblh, kpbl, mixht

    ht_arr = jnp.asarray(ht, dtype=jnp.float32)
    dz_arr = jnp.asarray(dz, dtype=jnp.float32)
    pmid_arr = jnp.asarray(pmid, dtype=jnp.float32)
    pint_arr = jnp.asarray(pint, dtype=jnp.float32)
    th_arr = jnp.asarray(th, dtype=jnp.float32)
    t_arr = jnp.asarray(t, dtype=jnp.float32)
    exner_arr = jnp.asarray(exner, dtype=jnp.float32)
    qv_arr = jnp.asarray(qv, dtype=jnp.float32)
    qcw_arr = jnp.asarray(qcw, dtype=jnp.float32)
    qci_arr = jnp.asarray(qci, dtype=jnp.float32)
    u_arr = jnp.asarray(u, dtype=jnp.float32)
    v_arr = jnp.asarray(v, dtype=jnp.float32)

    ni, nk, nj = t_arr.shape
    ncol = ni * nj

    def col2(x: Any) -> Any:
        return jnp.asarray(x, dtype=jnp.float32).reshape((ncol,))

    def col3(x: Any) -> Any:
        return jnp.asarray(x, dtype=jnp.float32).transpose(0, 2, 1).reshape((ncol, nk))

    def uncol2(x: Any) -> Any:
        return jnp.asarray(x).reshape((ni, nj))

    def uncol3(x: Any) -> Any:
        return jnp.asarray(x).reshape((ni, nj, nk)).transpose(0, 2, 1)

    ht_c = col2(ht_arr)
    dz_c = col3(dz_arr)
    pmid_c = col3(pmid_arr)
    pint_c = col3(pint_arr)
    th_c = col3(th_arr)
    t_c = col3(t_arr)
    exner_c = col3(exner_arr)
    qv_c = col3(qv_arr)
    qcw_c = col3(qcw_arr)
    qci_c = col3(qci_arr)
    u_c = col3(u_arr)
    v_c = col3(v_arr)

    dtturbl = jnp.asarray(dt, dtype=jnp.float32) * jnp.asarray(stepbl, dtype=jnp.float32)
    rdtturbl = jnp.asarray(1.0, dtype=jnp.float32) / dtturbl
    dtdif = dtturbl

    cwm_orig = qcw_c + jnp.where(flqi, qci_c, 0.0)
    the_orig = (cwm_orig * (-jnp.asarray(ELOCP, dtype=jnp.float32) / t_c) + 1.0) * th_c
    ape_orig = 1.0 / exner_c

    zint_bottom_up = _batched_zint_bottom_up(ht_c, dz_c)
    zint_layers = jnp.swapaxes(zint_bottom_up, 0, 1)[:, ::-1]
    zhk = jnp.concatenate([zint_layers, ht_c[:, None]], axis=1)

    tk_col = jnp.flip(t_c, axis=1)
    thek = jnp.flip(the_orig, axis=1)
    qv_helper = jnp.flip(qv_c, axis=1)
    qk = qv_helper / (1.0 + qv_helper)
    cwmk = jnp.flip(cwm_orig, axis=1)
    uk = jnp.flip(u_c, axis=1)
    vk = jnp.flip(v_c, axis=1)
    q2k = 2.0 * jnp.flip(jnp.asarray(tke_myj, dtype=jnp.float32).transpose(0, 2, 1).reshape((ncol, nk)), axis=1)

    ct_c = col2(ct)
    gm, gh, el, pblh_c, lpbl, lmxl, ct_c, mixht_c = _call_tuple_kernel(
        _batched_mixlen_full_lmh,
        uk,
        vk,
        tk_col,
        thek,
        qk,
        cwmk,
        q2k,
        zhk,
        ct_c,
        rounding_token,
    )
    el, q2k = _call_tuple_kernel(
        _batched_prodq2_full_lmh,
        dtturbl,
        col2(ustar),
        gm,
        gh,
        el,
        q2k,
        rounding_token,
    )
    kpbl_c = nk - lpbl + 1
    akh, akm = _call_tuple_kernel(
        _batched_difcof_full_lmh,
        gm,
        gh,
        el,
        q2k,
        zhk,
        rounding_token,
    )

    exch_c = col3(exch_h)
    hidx = jnp.arange(nk - 2, -1, -1)
    deltaz = 0.5 * (zhk[:, hidx] - zhk[:, hidx + 2])
    exch_update = akh[:, hidx] * deltaz
    exch_c = jnp.concatenate([exch_update, exch_c[:, nk - 1 : nk]], axis=1)

    q2k = _call_single_kernel(_batched_vdifq_full_lmh, dtdif, q2k, el, zhk)
    q2k = jnp.maximum(q2k, EPSQ2)

    tke_c = 0.5 * jnp.flip(q2k, axis=1)
    el_myj_c = jnp.concatenate([jnp.zeros((ncol, 1), dtype=jnp.float32), jnp.flip(el, axis=1)], axis=1)

    tsk_c = col2(tsk)
    qsfc_c = col2(qsfc)
    thz0_c = col2(thz0)
    qz0_c = col2(qz0)
    xland_c = col2(xland)
    sice_c = col2(sice)
    snow_c = col2(snow)
    chklowq_c = col2(chklowq)
    akhs_c = col2(akhs)
    akms_c = col2(akms)
    elflx_c = col2(elflx)

    psfc = pint_c[:, 0]
    thsk = _call_single_kernel(
        lambda surface_temperature, surface_pressure: surface_temperature
        * (1.0e5 / surface_pressure) ** CAPA,
        tsk_c,
        psfc,
    )
    pressure_bottom_up = _call_single_kernel(
        lambda pressure: jnp.flip(pressure, axis=1),
        pmid_c,
    )
    virtual_vapor = _call_single_kernel(
        lambda vapor: jnp.asarray(P608, dtype=jnp.float32) * vapor,
        qk,
    )
    virtual_factor = _call_single_kernel(lambda vapor: 1.0 + vapor, virtual_vapor)
    virtual_factor = _call_single_kernel(
        lambda factor, condensate: factor - condensate,
        virtual_factor,
        cwmk,
    )
    dry_temperature = _call_single_kernel(
        lambda temperature: jnp.asarray(R_D, dtype=jnp.float32) * temperature,
        tk_col,
    )
    virtual_temperature = _call_single_kernel(
        lambda temperature, factor: temperature * factor,
        dry_temperature,
        virtual_factor,
    )
    rho_helper = _call_single_kernel(
        lambda pressure, temperature: pressure / temperature,
        pressure_bottom_up,
        virtual_temperature,
    )
    akh_dens = _call_single_kernel(
        lambda diffusivity, density: diffusivity
        * 0.5
        * (density[:, : nk - 1] + density[:, 1:nk]),
        akh,
        rho_helper,
    )

    seamask = _call_single_kernel(lambda land: land - 1.0, xland_c)
    thz0_c = _call_single_kernel(
        lambda sea, skin_theta, reference_theta: (1.0 - sea) * skin_theta
        + sea * reference_theta,
        seamask,
        thsk,
        thz0_c,
    )
    akhs_dens = _call_single_kernel(
        lambda exchange, density: exchange * density[:, nk - 1],
        akhs_c,
        rho_helper,
    )
    qfc1 = _call_single_kernel(
        lambda low_q, exchange: jnp.asarray(XLV, dtype=jnp.float32)
        * low_q
        * exchange,
        chklowq_c,
        akhs_dens,
    )
    qfc1 = _call_single_kernel(
        lambda snow_amount, sea_ice, flux: jnp.where(
            (snow_amount > 0.0) | (sea_ice > 0.5),
            flux * jnp.asarray(RLIVWV, dtype=jnp.float32),
            flux,
        ),
        snow_c,
        sice_c,
        qfc1,
    )
    qfc1_safe = _call_single_kernel(
        lambda flux: jnp.where(flux > 0.0, flux, 1.0),
        qfc1,
    )
    land_qsfc = _call_single_kernel(
        lambda flux, vapor, surface_flux, safe_flux, previous: jnp.where(
            flux > 0.0,
            vapor[:, nk - 1] + surface_flux / safe_flux,
            previous,
        ),
        qfc1,
        qk,
        elflx_c,
        qfc1_safe,
        qsfc_c,
    )
    exnsfc_base = _call_single_kernel(
        lambda surface_pressure: 1.0e5 / surface_pressure,
        psfc,
    )
    exnsfc = _call_single_kernel(lambda base: base**CAPA, exnsfc_base)
    sea_qsfc_prefactor = _call_single_kernel(
        lambda surface_pressure: jnp.asarray(PQ0SEA, dtype=jnp.float32)
        / surface_pressure,
        psfc,
    )
    sea_qsfc_a3 = _call_single_kernel(
        lambda exner_surface: jnp.asarray(A3, dtype=jnp.float32) * exner_surface,
        exnsfc,
    )
    sea_qsfc_numerator = _call_single_kernel(
        lambda skin_theta, offset: skin_theta - offset,
        thsk,
        sea_qsfc_a3,
    )
    sea_qsfc_numerator = _call_single_kernel(
        lambda numerator: jnp.asarray(A2, dtype=jnp.float32) * numerator,
        sea_qsfc_numerator,
    )
    sea_qsfc_a4 = _call_single_kernel(
        lambda exner_surface: jnp.asarray(A4, dtype=jnp.float32) * exner_surface,
        exnsfc,
    )
    sea_qsfc_denominator = _call_single_kernel(
        lambda skin_theta, offset: skin_theta - offset,
        thsk,
        sea_qsfc_a4,
    )
    sea_qsfc_exponent = _call_single_kernel(
        lambda numerator, denominator: numerator / denominator,
        sea_qsfc_numerator,
        sea_qsfc_denominator,
    )
    sea_qsfc_exponential = _call_single_kernel(jnp.exp, sea_qsfc_exponent)
    sea_qsfc = _call_single_kernel(
        lambda prefactor, exponential: prefactor * exponential,
        sea_qsfc_prefactor,
        sea_qsfc_exponential,
    )
    qsfc_c = _call_single_kernel(
        lambda sea, land_value, sea_value: jnp.where(
            sea < 0.5,
            land_value,
            sea_value,
        ),
        seamask,
        land_qsfc,
        sea_qsfc,
    )
    qz0_c = _call_single_kernel(
        lambda sea, surface_vapor, reference_vapor: (1.0 - sea) * surface_vapor
        + sea * reference_vapor,
        seamask,
        qsfc_c,
        qz0_c,
    )

    qcw_helper = jnp.flip(qcw_c, axis=1)
    qci_helper = jnp.flip(qci_c, axis=1)
    if flqi:
        species = jnp.stack([thek, qk, qcw_helper, qci_helper], axis=1)
        sz0 = jnp.stack([thz0_c, qz0_c, jnp.zeros_like(thz0_c), jnp.zeros_like(thz0_c)], axis=1)
        clow = jnp.stack([jnp.ones_like(chklowq_c), chklowq_c, jnp.zeros_like(chklowq_c), jnp.zeros_like(chklowq_c)], axis=1)
        cts = jnp.stack([ct_c, jnp.zeros_like(ct_c), jnp.zeros_like(ct_c), jnp.zeros_like(ct_c)], axis=1)
    else:
        species = jnp.stack([thek, qk, qcw_helper], axis=1)
        sz0 = jnp.stack([thz0_c, qz0_c, jnp.zeros_like(thz0_c)], axis=1)
        clow = jnp.stack([jnp.ones_like(chklowq_c), chklowq_c, jnp.zeros_like(chklowq_c)], axis=1)
        cts = jnp.stack([ct_c, jnp.zeros_like(ct_c), jnp.zeros_like(ct_c)], axis=1)

    species = _call_single_kernel(
        _batched_vdifh_active_full_lmh,
        dtdif,
        lpbl,
        sz0,
        akhs_dens,
        clow,
        cts,
        species,
        akh_dens,
        zhk,
        rho_helper,
        rounding_token,
    )

    thek_new = species[:, 0, :]
    qk_new = species[:, 1, :]
    qcwk_new = species[:, 2, :]
    qcik_new = species[:, 3, :]
    cwm_new = _call_single_kernel(
        lambda cloud_water, cloud_ice: cloud_water
        + jnp.where(flqi, cloud_ice, 0.0),
        qcwk_new,
        qcik_new,
    )

    thnew_orig = _call_single_kernel(
        lambda theta_energy, condensate, inverse_exner: jnp.flip(
            theta_energy
            + condensate
            * jnp.asarray(ELOCP, dtype=jnp.float32)
            * jnp.flip(inverse_exner, axis=1),
            axis=1,
        ),
        thek_new,
        cwm_new,
        ape_orig,
    )
    q_new_orig = _call_single_kernel(lambda value: jnp.flip(value, axis=1), qk_new)
    qcw_new_orig = _call_single_kernel(lambda value: jnp.flip(value, axis=1), qcwk_new)
    qci_new_orig = _call_single_kernel(lambda value: jnp.flip(value, axis=1), qcik_new)

    rth_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt,
        thnew_orig,
        th_c,
        rdtturbl,
    )
    qold = _call_single_kernel(lambda vapor: vapor / (1.0 + vapor), qv_c)
    rqv_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt
        / (1.0 - updated) ** 2,
        q_new_orig,
        qold,
        rdtturbl,
    )
    rqc_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt,
        qcw_new_orig,
        qcw_c,
        rdtturbl,
    )
    rqi_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt,
        qci_new_orig,
        qci_c,
        rdtturbl,
    )

    akm_dens = _call_single_kernel(
        lambda diffusivity, density: diffusivity
        * 0.5
        * (density[:, : nk - 1] + density[:, 1:nk]),
        akm,
        rho_helper,
    )
    akms_dens = _call_single_kernel(
        lambda exchange, density: exchange * density[:, nk - 1],
        akms_c,
        rho_helper,
    )
    uk_new, vk_new = _call_tuple_kernel(
        _batched_vdifv_full_lmh,
        dtdif,
        col2(uz0),
        col2(vz0),
        akms_dens,
        uk,
        vk,
        akm_dens,
        zhk,
        rho_helper,
        rounding_token,
    )
    u_new_orig = _call_single_kernel(lambda value: jnp.flip(value, axis=1), uk_new)
    v_new_orig = _call_single_kernel(lambda value: jnp.flip(value, axis=1), vk_new)
    rubl_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt,
        u_new_orig,
        u_c,
        rdtturbl,
    )
    rvbl_c = _call_single_kernel(
        lambda updated, original, reciprocal_dt: (updated - original)
        * reciprocal_dt,
        v_new_orig,
        v_c,
        rdtturbl,
    )

    return (
        uncol2(qsfc_c),
        uncol2(thz0_c),
        uncol2(qz0_c),
        jnp.asarray(uz0, dtype=jnp.float32),
        jnp.asarray(vz0, dtype=jnp.float32),
        uncol3(tke_c),
        uncol3(exch_c),
        jnp.asarray(ustar, dtype=jnp.float32),
        jnp.asarray(znt, dtype=jnp.float32),
        uncol3(el_myj_c),
        uncol2(pblh_c),
        uncol2(kpbl_c.astype(jnp.int32)),
        uncol2(ct_c),
        jnp.asarray(akhs, dtype=jnp.float32),
        jnp.asarray(akms, dtype=jnp.float32),
        uncol2(mixht_c),
        uncol3(rubl_c),
        uncol3(rvbl_c),
        uncol3(rth_c),
        uncol3(rqv_c),
        uncol3(rqc_c) if rqcblten is not None else None,
        uncol3(rqi_c) if rqiblten is not None else None,
    )


_JIT_MYJPBL_FULL_LMH_BATCH = os.environ.get("WRF_PY_JIT_MYJPBL_FULL_LMH_BATCH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DISABLE_MYJPBL_FULL_LMH_BATCH_JIT_CONTEXT = os.environ.get(
    "WRF_PY_DISABLE_MYJPBL_FULL_LMH_BATCH_JIT_CONTEXT", "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_USE_MYJPBL_FULL_LMH_BATCH = os.environ.get("WRF_PY_USE_MYJPBL_FULL_LMH_BATCH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_ASSUME_LOWLYR_KTS_FOR_BATCH = os.environ.get("WRF_PY_ASSUME_MYJPBL_LOWLYR_KTS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_INLINE_COLUMN_PHYSICS_JITS = os.environ.get("WRF_PY_INLINE_COLUMN_PHYSICS_JITS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_myjpbl_full_lmh_batch_jit = jax.jit(_myjpbl_full_lmh_batch_impl, static_argnames=("flqi",), inline=_INLINE_COLUMN_PHYSICS_JITS)


def myjpbl(
    dt: float,
    stepbl: int,
    ht: Any,
    dz: Any,
    pmid: Any,
    pint: Any,
    th: Any,
    t: Any,
    exner: Any,
    qv: Any,
    qcw: Any,
    qci: Any,
    qcs: Any,
    qcr: Any,
    qcg: Any,
    u: Any,
    v: Any,
    rho: Any,
    tsk: Any,
    qsfc: Any,
    chklowq: Any,
    thz0: Any,
    qz0: Any,
    uz0: Any,
    vz0: Any,
    lowlyr: Any,
    xland: Any,
    sice: Any,
    snow: Any,
    tke_myj: Any,
    exch_h: Any,
    ustar: Any,
    znt: Any,
    el_myj: Any,
    pblh: Any,
    kpbl: Any,
    ct: Any,
    akhs: Any,
    akms: Any,
    elflx: Any,
    mixht: Any,
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rqsblten: Any,
    rqrblten: Any,
    rqgblten: Any,
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
) -> MyjPblResult:
    """Run WRF ``MYJPBL`` column physics over the active tile."""

    del rho, qcs, qcr, qcg
    if qv is None or qcw is None:
        raise ValueError("WRF MYJPBL requires qv and qcw arrays")

    flqi = qci is not None
    nspec = 4 if flqi else 3
    nspecs = 1
    nspece = 7
    dtturbl = np.float32(float(dt) * int(stepbl))
    rdtturbl = np.float32(1.0) / dtturbl
    dtdif = dtturbl

    ht_arr = jnp.asarray(ht, dtype=jnp.float32)
    dz_arr = jnp.asarray(dz, dtype=jnp.float32)
    pmid_arr = jnp.asarray(pmid, dtype=jnp.float32)
    pint_arr = jnp.asarray(pint, dtype=jnp.float32)
    th_arr = jnp.asarray(th, dtype=jnp.float32)
    t_arr = jnp.asarray(t, dtype=jnp.float32)
    exner_arr = jnp.asarray(exner, dtype=jnp.float32)
    qv_arr = jnp.asarray(qv, dtype=jnp.float32)
    qcw_arr = jnp.asarray(qcw, dtype=jnp.float32)
    qci_arr = None if qci is None else jnp.asarray(qci, dtype=jnp.float32)
    u_arr = jnp.asarray(u, dtype=jnp.float32)
    v_arr = jnp.asarray(v, dtype=jnp.float32)
    tsk_arr = jnp.asarray(tsk, dtype=jnp.float32)
    chklowq_arr = jnp.asarray(chklowq, dtype=jnp.float32)
    lowlyr_arr = jnp.asarray(lowlyr, dtype=jnp.int32)
    xland_arr = jnp.asarray(xland, dtype=jnp.float32)
    sice_arr = jnp.asarray(sice, dtype=jnp.float32)
    snow_arr = jnp.asarray(snow, dtype=jnp.float32)
    elflx_arr = jnp.asarray(elflx, dtype=jnp.float32)

    qsfc_out = jnp.asarray(qsfc, dtype=jnp.float32)
    thz0_out = jnp.asarray(thz0, dtype=jnp.float32)
    qz0_out = jnp.asarray(qz0, dtype=jnp.float32)
    uz0_out = jnp.asarray(uz0, dtype=jnp.float32)
    vz0_out = jnp.asarray(vz0, dtype=jnp.float32)
    tke_out = jnp.asarray(tke_myj, dtype=jnp.float32)
    exch_h_out = jnp.asarray(exch_h, dtype=jnp.float32)
    ustar_out = jnp.asarray(ustar, dtype=jnp.float32)
    znt_out = jnp.asarray(znt, dtype=jnp.float32)
    el_myj_out = jnp.asarray(el_myj, dtype=jnp.float32)
    pblh_out = jnp.asarray(pblh, dtype=jnp.float32)
    kpbl_out = jnp.asarray(kpbl, dtype=jnp.int32)
    ct_out = jnp.asarray(ct, dtype=jnp.float32)
    akhs_out = jnp.asarray(akhs, dtype=jnp.float32)
    akms_out = jnp.asarray(akms, dtype=jnp.float32)
    mixht_out = jnp.asarray(mixht, dtype=jnp.float32)
    rublten_out = jnp.asarray(rublten, dtype=jnp.float32)
    rvblten_out = jnp.asarray(rvblten, dtype=jnp.float32)
    rthblten_out = jnp.asarray(rthblten, dtype=jnp.float32)
    rqvblten_out = jnp.asarray(rqvblten, dtype=jnp.float32)
    rqcblten_out = None if rqcblten is None else jnp.asarray(rqcblten, dtype=jnp.float32)
    rqiblten_out = None if rqiblten is None else jnp.asarray(rqiblten, dtype=jnp.float32)
    rqsblten_out = None if rqsblten is None else jnp.asarray(rqsblten, dtype=jnp.float32)
    rqrblten_out = None if rqrblten is None else jnp.asarray(rqrblten, dtype=jnp.float32)
    rqgblten_out = None if rqgblten is None else jnp.asarray(rqgblten, dtype=jnp.float32)

    ni = int(ite) - int(its) + 1
    nj = int(jte) - int(jts) + 1
    nk = int(kte) - int(kts) + 1

    def has_active_3d_shape(arr: Any) -> bool:
        arr_shape = jnp.asarray(arr).shape
        return len(arr_shape) == 3 and arr_shape[0] == ni and arr_shape[1] >= nk and arr_shape[2] == nj

    def active_3d(arr: Any) -> Any:
        return jnp.asarray(arr, dtype=jnp.float32)[:, :nk, :]

    def restore_3d(base: Any, active: Any) -> Any:
        base_arr = jnp.asarray(base)
        if base_arr.shape[1] == nk:
            return active
        return base_arr.at[:, :nk, :].set(active)

    def restore_zeroed_3d(base: Any, active: Any) -> Any:
        base_arr = jnp.zeros_like(jnp.asarray(base))
        if base_arr.shape[1] == nk:
            return active
        return base_arr.at[:, :nk, :].set(active)

    kts_value = int(kts)
    kte_value = int(kte)
    full_depth_bounds = (kts_value == 1 and kte_value == nk) or (kts_value == 0 and kte_value == nk - 1)

    if (
        _USE_MYJPBL_FULL_LMH_BATCH
        and
        full_depth_bounds
        and ht_arr.shape == (ni, nj)
        and has_active_3d_shape(dz_arr)
        and has_active_3d_shape(pmid_arr)
        and has_active_3d_shape(pint_arr)
        and has_active_3d_shape(th_arr)
        and has_active_3d_shape(t_arr)
        and has_active_3d_shape(exner_arr)
        and has_active_3d_shape(qv_arr)
        and has_active_3d_shape(qcw_arr)
        and has_active_3d_shape(u_arr)
        and has_active_3d_shape(v_arr)
        and lowlyr_arr.shape == (ni, nj)
        and (_ASSUME_LOWLYR_KTS_FOR_BATCH or bool(jnp.all(lowlyr_arr == int(kts))))
    ):
        batch_rounding_token = (
            jnp.asarray(0, dtype=jnp.uint32)
            if rounding_token is None
            else jnp.asarray(rounding_token, dtype=jnp.uint32)
        )
        batch_qci = jnp.zeros_like(qv_arr) if qci_arr is None else qci_arr
        # Keep the full MYJPBL batch out of one fused XLA graph by default;
        # full-batch JIT changed one-ulp THNEW values that later amplify in
        # Tiedtke.  The Python batch implementation still calls the smaller
        # leaf kernels above, so their existing JIT boundaries remain usable.
        batch_impl = (
            _myjpbl_full_lmh_batch_impl
            if _DISABLE_MYJPBL_FULL_LMH_BATCH_JIT_CONTEXT or not _JIT_MYJPBL_FULL_LMH_BATCH
            else _myjpbl_full_lmh_batch_jit
        )
        batch_args = (
            float(dt),
            int(stepbl),
            ht_arr,
            active_3d(dz_arr),
            active_3d(pmid_arr),
            active_3d(pint_arr),
            active_3d(th_arr),
            active_3d(t_arr),
            active_3d(exner_arr),
            active_3d(qv_arr),
            active_3d(qcw_arr),
            active_3d(batch_qci),
            active_3d(u_arr),
            active_3d(v_arr),
            tsk_arr,
            qsfc_out,
            chklowq_arr,
            thz0_out,
            qz0_out,
            uz0_out,
            vz0_out,
            xland_arr,
            sice_arr,
            snow_arr,
            active_3d(tke_out),
            active_3d(exch_h_out),
            ustar_out,
            znt_out,
            active_3d(el_myj_out),
            pblh_out,
            kpbl_out,
            ct_out,
            akhs_out,
            akms_out,
            elflx_arr,
            mixht_out,
            active_3d(rublten_out),
            active_3d(rvblten_out),
            active_3d(rthblten_out),
            active_3d(rqvblten_out),
            None if rqcblten_out is None else active_3d(rqcblten_out),
            None if rqiblten_out is None else active_3d(rqiblten_out),
            batch_rounding_token,
        )
        if _contains_jax_tracer(batch_args):
            def traced_batch_impl(*values: Any) -> tuple[Any, ...]:
                return batch_impl(*values, flqi=flqi)

            batch_outputs = semantic_call(traced_batch_impl, *batch_args)
        else:
            batch_outputs = batch_impl(*batch_args, flqi=flqi)
        (
            qsfc_b,
            thz0_b,
            qz0_b,
            uz0_b,
            vz0_b,
            tke_b,
            exch_h_b,
            ustar_b,
            znt_b,
            el_myj_b,
            pblh_b,
            kpbl_b,
            ct_b,
            akhs_b,
            akms_b,
            mixht_b,
            rublten_b,
            rvblten_b,
            rthblten_b,
            rqvblten_b,
            rqcblten_b,
            rqiblten_b,
        ) = batch_outputs
        return MyjPblResult(
            qsfc=qsfc_b,
            thz0=thz0_b,
            qz0=qz0_b,
            uz0=uz0_b,
            vz0=vz0_b,
            tke_myj=restore_3d(tke_out, tke_b),
            exch_h=restore_3d(exch_h_out, exch_h_b),
            ustar=ustar_b,
            znt=znt_b,
            el_myj=restore_zeroed_3d(el_myj_out, el_myj_b),
            pblh=pblh_b,
            kpbl=kpbl_b,
            ct=ct_b,
            akhs=akhs_b,
            akms=akms_b,
            mixht=mixht_b,
            rublten=restore_3d(rublten_out, rublten_b),
            rvblten=restore_3d(rvblten_out, rvblten_b),
            rthblten=restore_3d(rthblten_out, rthblten_b),
            rqvblten=restore_3d(rqvblten_out, rqvblten_b),
            rqcblten=None if rqcblten_out is None else restore_3d(rqcblten_out, rqcblten_b),
            rqiblten=None if rqiblten_out is None else restore_3d(rqiblten_out, rqiblten_b),
            rqsblten=None if rqsblten_out is None else jnp.asarray(rqsblten_out),
            rqrblten=None if rqrblten_out is None else jnp.asarray(rqrblten_out),
            rqgblten=None if rqgblten_out is None else jnp.asarray(rqgblten_out),
        )

    ht_arr = np.asarray(ht_arr, dtype=np.float32)
    dz_arr = np.asarray(dz_arr, dtype=np.float32)
    pmid_arr = np.asarray(pmid_arr, dtype=np.float32)
    pint_arr = np.asarray(pint_arr, dtype=np.float32)
    th_arr = np.asarray(th_arr, dtype=np.float32)
    t_arr = np.asarray(t_arr, dtype=np.float32)
    exner_arr = np.asarray(exner_arr, dtype=np.float32)
    qv_arr = np.asarray(qv_arr, dtype=np.float32)
    qcw_arr = np.asarray(qcw_arr, dtype=np.float32)
    qci_arr = None if qci_arr is None else np.asarray(qci_arr, dtype=np.float32)
    u_arr = np.asarray(u_arr, dtype=np.float32)
    v_arr = np.asarray(v_arr, dtype=np.float32)
    tsk_arr = np.asarray(tsk_arr, dtype=np.float32)
    chklowq_arr = np.asarray(chklowq_arr, dtype=np.float32)
    lowlyr_arr = np.asarray(lowlyr_arr, dtype=np.int32)
    xland_arr = np.asarray(xland_arr, dtype=np.float32)
    sice_arr = np.asarray(sice_arr, dtype=np.float32)
    snow_arr = np.asarray(snow_arr, dtype=np.float32)
    elflx_arr = np.asarray(elflx_arr, dtype=np.float32)

    qsfc_out = np.array(qsfc_out, dtype=np.float32, copy=True, order="F")
    thz0_out = np.array(thz0_out, dtype=np.float32, copy=True, order="F")
    qz0_out = np.array(qz0_out, dtype=np.float32, copy=True, order="F")
    uz0_out = np.array(uz0_out, dtype=np.float32, copy=True, order="F")
    vz0_out = np.array(vz0_out, dtype=np.float32, copy=True, order="F")
    tke_out = np.array(tke_out, dtype=np.float32, copy=True, order="F")
    exch_h_out = np.array(exch_h_out, dtype=np.float32, copy=True, order="F")
    ustar_out = np.array(ustar_out, dtype=np.float32, copy=True, order="F")
    znt_out = np.array(znt_out, dtype=np.float32, copy=True, order="F")
    el_myj_out = np.array(el_myj_out, dtype=np.float32, copy=True, order="F")
    pblh_out = np.array(pblh_out, dtype=np.float32, copy=True, order="F")
    kpbl_out = np.array(kpbl_out, dtype=np.int32, copy=True, order="F")
    ct_out = np.array(ct_out, dtype=np.float32, copy=True, order="F")
    akhs_out = np.array(akhs_out, dtype=np.float32, copy=True, order="F")
    akms_out = np.array(akms_out, dtype=np.float32, copy=True, order="F")
    mixht_out = np.array(mixht_out, dtype=np.float32, copy=True, order="F")
    rublten_out = np.array(rublten_out, dtype=np.float32, copy=True, order="F")
    rvblten_out = np.array(rvblten_out, dtype=np.float32, copy=True, order="F")
    rthblten_out = np.array(rthblten_out, dtype=np.float32, copy=True, order="F")
    rqvblten_out = np.array(rqvblten_out, dtype=np.float32, copy=True, order="F")
    rqcblten_out = None if rqcblten_out is None else np.array(rqcblten_out, dtype=np.float32, copy=True, order="F")
    rqiblten_out = None if rqiblten_out is None else np.array(rqiblten_out, dtype=np.float32, copy=True, order="F")
    rqsblten_out = None if rqsblten_out is None else np.array(rqsblten_out, dtype=np.float32, copy=True, order="F")
    rqrblten_out = None if rqrblten_out is None else np.array(rqrblten_out, dtype=np.float32, copy=True, order="F")
    rqgblten_out = None if rqgblten_out is None else np.array(rqgblten_out, dtype=np.float32, copy=True, order="F")

    def fi(i: int) -> int:
        return int(i) - int(ims)

    def fj(j: int) -> int:
        return int(j) - int(jms)

    def fk(k: int) -> int:
        return int(k) - int(kms)

    def ti(i: int) -> int:
        return int(i) - int(its)

    def tj(j: int) -> int:
        return int(j) - int(jts)

    def tk(k: int) -> int:
        return int(k) - int(kts)

    akh = np.zeros((ni, max(nk - 1, 1), nj), dtype=np.float32, order="F")
    akm = np.zeros((ni, max(nk - 1, 1), nj), dtype=np.float32, order="F")
    zint = np.zeros((ni, nk + 1, nj), dtype=np.float32, order="F")
    ape = np.zeros((ni, nk, nj), dtype=np.float32, order="F")
    the = np.zeros((ni, nk, nj), dtype=np.float32, order="F")
    cwm = np.zeros((ni, nk, nj), dtype=np.float32, order="F")
    rhok = np.zeros((nk, ni), dtype=np.float32, order="F")
    lpbl = np.zeros((ni, nj), dtype=np.int32, order="F")

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        for i in range(int(its), int(ite) + 1):
            zint[ti(i), tk(int(kte) + 1), jt] = ht_arr[fi(i), jj]

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        for k in range(int(kte), int(kts) - 1, -1):
            kflip = int(kte) + 1 - k
            for i in range(int(its), int(ite) + 1):
                ii = ti(i)
                zint[ii, tk(k), jt] = zint[ii, tk(k + 1), jt] + dz_arr[fi(i), fk(kflip), jj]
                ape[ii, tk(k), jt] = np.float32(1.0) / exner_arr[fi(i), fk(k), jj]
                tx = t_arr[fi(i), fk(k), jj]
                cwm_val = qcw_arr[fi(i), fk(k), jj]
                if flqi:
                    cwm_val = cwm_val + qci_arr[fi(i), fk(k), jj]
                cwm[ii, tk(k), jt] = cwm_val
                the[ii, tk(k), jt] = (cwm_val * (-np.float32(ELOCP) / tx) + np.float32(1.0)) * th_arr[fi(i), fk(k), jj]

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        for i in range(int(its), int(ite) + 1):
            el_myj_out[fi(i), :, jj] = np.float32(0.0)

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            lmh = int(kte) - int(lowlyr_arr[fi(i), jj]) + 1
            tk_col = np.zeros((nk,), dtype=np.float32, order="F")
            thek = np.zeros((nk,), dtype=np.float32, order="F")
            qk = np.zeros((nk,), dtype=np.float32, order="F")
            cwmk = np.zeros((nk,), dtype=np.float32, order="F")
            pk = np.zeros((nk,), dtype=np.float32, order="F")
            uk = np.zeros((nk,), dtype=np.float32, order="F")
            vk = np.zeros((nk,), dtype=np.float32, order="F")
            q2k = np.zeros((nk,), dtype=np.float32, order="F")
            zhk = np.zeros((nk + 1,), dtype=np.float32, order="F")

            for k in range(int(kte), int(kts) - 1, -1):
                kflip = int(kte) + 1 - k
                kk = tk(k)
                tk_col[kk] = t_arr[fi(i), fk(kflip), jj]
                thek[kk] = the[ii, tk(kflip), jt]
                ratiomx = qv_arr[fi(i), fk(kflip), jj]
                qk[kk] = ratiomx / (np.float32(1.0) + ratiomx)
                cwmk[kk] = cwm[ii, tk(kflip), jt]
                pk[kk] = pmid_arr[fi(i), fk(kflip), jj]
                uk[kk] = u_arr[fi(i), fk(kflip), jj]
                vk[kk] = v_arr[fi(i), fk(kflip), jj]
                q2k[kk] = np.float32(2.0) * tke_out[fi(i), fk(kflip), jj]
                zhk[kk] = zint[ii, tk(k), jt]
            zhk[tk(int(kte) + 1)] = ht_arr[fi(i), jj]

            gm, gh, el, pblh_value, lpbl_value, lmxl, ct_value, mixht_value = mixlen(
                lmh,
                uk,
                vk,
                tk_col,
                thek,
                qk,
                cwmk,
                q2k,
                zhk,
                float(ct_out[fi(i), jj]),
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
            gm_arr = np.asarray(gm, dtype=np.float32)
            gh_arr = np.asarray(gh, dtype=np.float32)
            el_arr = np.asarray(el, dtype=np.float32)
            pblh_out[fi(i), jj] = np.float32(pblh_value)
            lpbl[ii, jt] = int(lpbl_value)
            ct_out[fi(i), jj] = np.float32(ct_value)
            mixht_out[fi(i), jj] = np.float32(mixht_value)

            el_arr, q2k = prodq2(
                lmh,
                float(dtturbl),
                float(ustar_out[fi(i), jj]),
                gm_arr,
                gh_arr,
                el_arr,
                q2k,
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
            el_arr = np.asarray(el_arr, dtype=np.float32)
            q2k = np.asarray(q2k, dtype=np.float32)

            kpbl_out[fi(i), jj] = int(kte) - lpbl[ii, jt] + 1

            akhk, akmk = difcof(
                lmh,
                lmxl,
                gm_arr,
                gh_arr,
                el_arr,
                tk_col,
                q2k,
                zhk,
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
                0,
            )
            akhk = np.asarray(akhk, dtype=np.float32)
            akmk = np.asarray(akmk, dtype=np.float32)

            for k in range(int(kts), int(kte)):
                kflip = int(kte) - k
                akh[ii, tk(k), jt] = akhk[tk(k)]
                akm[ii, tk(k), jt] = akmk[tk(k)]
                deltaz = np.float32(0.5) * (zhk[tk(kflip)] - zhk[tk(kflip + 2)])
                exch_h_out[fi(i), fk(k), jj] = akhk[tk(kflip)] * deltaz

            q2k = np.asarray(
                vdifq(
                    lmh,
                    float(dtdif),
                    q2k,
                    el_arr,
                    zhk,
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
                ),
                dtype=np.float32,
            ).copy()

            for k in range(int(kts), int(kte) + 1):
                kflip = int(kte) + 1 - k
                q2k[tk(kflip)] = max(q2k[tk(kflip)], np.float32(EPSQ2))
                tke_out[fi(i), fk(k), jj] = np.float32(0.5) * q2k[tk(kflip)]
                if kflip < int(kte):
                    el_myj_out[fi(i), fk(k), jj] = el_arr[tk(kflip)]

    thsk = np.zeros((ni, nj), dtype=np.float32, order="F")
    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        for i in range(int(its), int(ite) + 1):
            psfc = pint_arr[fi(i), fk(int(lowlyr_arr[fi(i), jj])), jj]
            thsk[ti(i), jt] = tsk_arr[fi(i), jj] * (np.float32(1.0e5) / psfc) ** np.float32(CAPA)

    for j in range(int(jts), int(jte) + 1):
        jj = fj(j)
        jt = tj(j)
        last_lmh = int(kte) - int(lowlyr_arr[fi(int(ite)), jj]) + 1
        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            thek = np.zeros((nk,), dtype=np.float32, order="F")
            qk = np.zeros((nk,), dtype=np.float32, order="F")
            cwmk = np.zeros((nk,), dtype=np.float32, order="F")
            qcwk = np.zeros((nk,), dtype=np.float32, order="F")
            qcik = np.zeros((nk,), dtype=np.float32, order="F")
            zhk = np.zeros((nk + 1,), dtype=np.float32, order="F")
            species = np.zeros((nspece - nspecs + 1, nk), dtype=np.float32, order="F")

            for k in range(int(kte), int(kts) - 1, -1):
                kflip = int(kte) + 1 - k
                kk = tk(k)
                thek[kk] = the[ii, tk(kflip), jt]
                ratiomx = qv_arr[fi(i), fk(kflip), jj]
                qk[kk] = ratiomx / (np.float32(1.0) + ratiomx)
                cwmk[kk] = cwm[ii, tk(kflip), jt]
                qcwk[kk] = qcw_arr[fi(i), fk(kflip), jj]
                species[0, kk] = thek[kk]
                species[1, kk] = qk[kk]
                species[2, kk] = qcwk[kk]
                if flqi:
                    qcik[kk] = qci_arr[fi(i), fk(kflip), jj]
                    species[3, kk] = qcik[kk]
                zhk[kk] = zint[ii, tk(k), jt]
                rhok[kk, ii] = pmid_arr[fi(i), fk(kflip), jj] / (
                    np.float32(R_D)
                    * t_arr[fi(i), fk(kflip), jj]
                    * (np.float32(1.0) + np.float32(P608) * qk[kk] - cwmk[kk])
                )

            akhk = np.zeros((max(nk - 1, 1),), dtype=np.float32, order="F")
            for k in range(int(kts), int(kte)):
                akhk[tk(k)] = akh[ii, tk(k), jt] * np.float32(0.5) * (rhok[tk(k), ii] + rhok[tk(k + 1), ii])
            zhk[tk(int(kte) + 1)] = zint[ii, tk(int(kte) + 1), jt]

            seamask = xland_arr[fi(i), jj] - np.float32(1.0)
            thz0_out[fi(i), jj] = (np.float32(1.0) - seamask) * thsk[ii, jt] + seamask * thz0_out[fi(i), jj]

            llow = int(lowlyr_arr[fi(i), jj])
            akhs_dens = akhs_out[fi(i), jj] * rhok[tk(int(kte) + 1 - llow), ii]
            if seamask < np.float32(0.5):
                qfc1 = np.float32(XLV) * chklowq_arr[fi(i), jj] * akhs_dens
                if snow_arr[fi(i), jj] > np.float32(0.0) or sice_arr[fi(i), jj] > np.float32(0.5):
                    qfc1 = qfc1 * np.float32(RLIVWV)
                if qfc1 > np.float32(0.0):
                    qlow = qk[tk(int(kte) + 1 - llow)]
                    qsfc_out[fi(i), jj] = qlow + elflx_arr[fi(i), jj] / qfc1
            else:
                psfc = pint_arr[fi(i), fk(llow), jj]
                exnsfc = (np.float32(1.0e5) / psfc) ** np.float32(CAPA)
                qsfc_out[fi(i), jj] = (
                    np.float32(PQ0SEA)
                    / psfc
                    * np.exp(np.float32(A2) * (thsk[ii, jt] - np.float32(A3) * exnsfc) / (thsk[ii, jt] - np.float32(A4) * exnsfc))
                )

            qz0_out[fi(i), jj] = (np.float32(1.0) - seamask) * qsfc_out[fi(i), jj] + seamask * qz0_out[fi(i), jj]

            sz0 = np.zeros((nspece,), dtype=np.float32, order="F")
            clow = np.zeros((nspece,), dtype=np.float32, order="F")
            cts = np.zeros((nspece,), dtype=np.float32, order="F")
            sz0[0] = thz0_out[fi(i), jj]
            sz0[1] = qz0_out[fi(i), jj]
            clow[0] = np.float32(1.0)
            clow[1] = chklowq_arr[fi(i), jj]
            cts[0] = ct_out[fi(i), jj]

            lmh = int(kte) - int(lowlyr_arr[fi(i), jj]) + 1
            last_lmh = lmh
            species = np.asarray(
                vdifh(
                    float(dtdif),
                    lmh,
                    nspecs,
                    nspece,
                    int(lpbl[ii, jt]),
                    sz0,
                    float(akhs_dens),
                    clow,
                    cts,
                    species,
                    nspec,
                    akhk,
                    zhk,
                    rhok[:, ii],
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
                    i,
                    j,
                ),
                dtype=np.float32,
            )

            for k in range(int(kts), int(kte) + 1):
                kk = tk(k)
                thek[kk] = species[0, kk]
                qk[kk] = species[1, kk]
                qcwk[kk] = species[2, kk]
                cwmk[kk] = qcwk[kk]
                if flqi:
                    qcik[kk] = species[3, kk]
                    cwmk[kk] = cwmk[kk] + qcik[kk]

            for k in range(int(kts), int(kte) + 1):
                kflip = int(kte) + 1 - k
                thold = th_arr[fi(i), fk(k), jj]
                thnew = thek[tk(kflip)] + cwmk[tk(kflip)] * np.float32(ELOCP) * ape[ii, tk(k), jt]
                dtdt = (thnew - thold) * rdtturbl
                qold = qv_arr[fi(i), fk(k), jj] / (np.float32(1.0) + qv_arr[fi(i), fk(k), jj])
                dqdt = (qk[tk(kflip)] - qold) * rdtturbl
                rthblten_out[fi(i), fk(k), jj] = dtdt
                rqvblten_out[fi(i), fk(k), jj] = dqdt / (np.float32(1.0) - qk[tk(kflip)]) ** np.float32(2.0)
                if rqcblten_out is not None:
                    rqcblten_out[fi(i), fk(k), jj] = (qcwk[tk(kflip)] - qcw_arr[fi(i), fk(k), jj]) * rdtturbl
                if flqi and rqiblten_out is not None:
                    rqiblten_out[fi(i), fk(k), jj] = (qcik[tk(kflip)] - qci_arr[fi(i), fk(k), jj]) * rdtturbl

        for i in range(int(its), int(ite) + 1):
            ii = ti(i)
            akmk = np.zeros((max(nk - 1, 1),), dtype=np.float32, order="F")
            for k in range(int(kts), int(kte)):
                akmk[tk(k)] = akm[ii, tk(k), jt] * (rhok[tk(k), ii] + rhok[tk(k + 1), ii]) * np.float32(0.5)

            llow = int(lowlyr_arr[fi(i), jj])
            akms_dens = akms_out[fi(i), jj] * rhok[tk(int(kte) + 1 - llow), ii]
            uk = np.zeros((nk,), dtype=np.float32, order="F")
            vk = np.zeros((nk,), dtype=np.float32, order="F")
            zhk = np.zeros((nk + 1,), dtype=np.float32, order="F")
            for k in range(int(kte), int(kts) - 1, -1):
                kflip = int(kte) + 1 - k
                uk[tk(k)] = u_arr[fi(i), fk(kflip), jj]
                vk[tk(k)] = v_arr[fi(i), fk(kflip), jj]
                zhk[tk(k)] = zint[ii, tk(k), jt]
            zhk[tk(int(kte) + 1)] = zint[ii, tk(int(kte) + 1), jt]

            uk, vk = vdifv(
                last_lmh,
                float(dtdif),
                float(uz0_out[fi(i), jj]),
                float(vz0_out[fi(i), jj]),
                float(akms_dens),
                uk,
                vk,
                akmk,
                zhk,
                rhok[:, ii],
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
                i,
                j,
            )
            uk = np.asarray(uk, dtype=np.float32)
            vk = np.asarray(vk, dtype=np.float32)
            for k in range(int(kts), int(kte) + 1):
                kflip = int(kte) + 1 - k
                rublten_out[fi(i), fk(k), jj] = (uk[tk(kflip)] - u_arr[fi(i), fk(k), jj]) * rdtturbl
                rvblten_out[fi(i), fk(k), jj] = (vk[tk(kflip)] - v_arr[fi(i), fk(k), jj]) * rdtturbl

    return MyjPblResult(
        qsfc=jnp.asarray(qsfc_out),
        thz0=jnp.asarray(thz0_out),
        qz0=jnp.asarray(qz0_out),
        uz0=jnp.asarray(uz0_out),
        vz0=jnp.asarray(vz0_out),
        tke_myj=jnp.asarray(tke_out),
        exch_h=jnp.asarray(exch_h_out),
        ustar=jnp.asarray(ustar_out),
        znt=jnp.asarray(znt_out),
        el_myj=jnp.asarray(el_myj_out),
        pblh=jnp.asarray(pblh_out),
        kpbl=jnp.asarray(kpbl_out),
        ct=jnp.asarray(ct_out),
        akhs=jnp.asarray(akhs_out),
        akms=jnp.asarray(akms_out),
        mixht=jnp.asarray(mixht_out),
        rublten=jnp.asarray(rublten_out),
        rvblten=jnp.asarray(rvblten_out),
        rthblten=jnp.asarray(rthblten_out),
        rqvblten=jnp.asarray(rqvblten_out),
        rqcblten=None if rqcblten_out is None else jnp.asarray(rqcblten_out),
        rqiblten=None if rqiblten_out is None else jnp.asarray(rqiblten_out),
        rqsblten=None if rqsblten_out is None else jnp.asarray(rqsblten_out),
        rqrblten=None if rqrblten_out is None else jnp.asarray(rqrblten_out),
        rqgblten=None if rqgblten_out is None else jnp.asarray(rqgblten_out),
    )


def myjpblinit(
    rublten: Any,
    rvblten: Any,
    rthblten: Any,
    rqvblten: Any,
    tke_myj: Any,
    exch_h: Any,
    restart: bool,
    allowed_to_read: bool,
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
) -> MyjPblInitResult:
    """Initialize MYJ PBL tendency and exchange arrays."""

    del allowed_to_read, ids, jds, kds, ime, jme, kme
    rublten_out = jnp.asarray(rublten)
    rvblten_out = jnp.asarray(rvblten)
    rthblten_out = jnp.asarray(rthblten)
    rqvblten_out = jnp.asarray(rqvblten)
    tke_myj_out = jnp.asarray(tke_myj)
    exch_h_out = jnp.asarray(exch_h)

    if not bool(restart):
        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        ktf = min(int(kte), int(kde) - 1)
        slices = (
            slice(int(its) - int(ims), itf - int(ims) + 1),
            slice(int(kts) - int(kms), ktf - int(kms) + 1),
            slice(int(jts) - int(jms), jtf - int(jms) + 1),
        )

        tke_myj_out = tke_myj_out.at[slices].set(jnp.asarray(EPSQ2, dtype=tke_myj_out.dtype))
        rublten_out = rublten_out.at[slices].set(jnp.asarray(0.0, dtype=rublten_out.dtype))
        rvblten_out = rvblten_out.at[slices].set(jnp.asarray(0.0, dtype=rvblten_out.dtype))
        rthblten_out = rthblten_out.at[slices].set(jnp.asarray(0.0, dtype=rthblten_out.dtype))
        rqvblten_out = rqvblten_out.at[slices].set(jnp.asarray(0.0, dtype=rqvblten_out.dtype))
        exch_h_out = exch_h_out.at[slices].set(jnp.asarray(0.0, dtype=exch_h_out.dtype))

    return MyjPblInitResult(
        rublten=rublten_out,
        rvblten=rvblten_out,
        rthblten=rthblten_out,
        rqvblten=rqvblten_out,
        tke_myj=tke_myj_out,
        exch_h=exch_h_out,
    )


def difcof(
    lmh: int,
    lmxl: int,
    gm: Any,
    gh: Any,
    el: Any,
    t: Any,
    q2: Any,
    z: Any,
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
    print_diag: int = 0,
) -> tuple[Any, Any]:
    """Return WRF ``DIFCOF`` outputs ``AKH`` and ``AKM`` for one column."""

    del lmxl, t, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, print_diag

    gm_arr = jnp.asarray(gm)
    gh_arr = jnp.asarray(gh)
    el_arr = jnp.asarray(el)
    q2_arr = jnp.asarray(q2)
    z_arr = jnp.asarray(z)
    akh = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    akm = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)

    for k in range(1, int(lmh)):
        kk = int(k) - int(kts)
        ell = el_arr[kk]
        eloq2 = ell * ell / q2_arr[kk]
        eloq4 = eloq2 * eloq2
        gml = gm_arr[kk]
        ghl = gh_arr[kk]
        aden = (ADNM * gml + ADNH * ghl) * ghl
        bden = BDNM * gml + BDNH * ghl
        cden = 1.0
        besm = BSMH * ghl
        besh = BSHM * gml + BSHH * ghl
        rden = 1.0 / (aden * eloq4 + bden * eloq2 + cden)
        esm = (besm * eloq2 + CESM) * rden
        esh = (besh * eloq2 + CESH) * rden
        rdz = 2.0 / (z_arr[int(k) - int(kts)] - z_arr[int(k + 2) - int(kts)])
        q1l = sqrt_for_wrf(q2_arr[kk])
        elqdz = ell * q1l * rdz
        akm = akm.at[kk].set(elqdz * esm)
        akh = akh.at[kk].set(elqdz * esh)

    return akh, akm


def mixlen(
    lmh: int,
    u: Any,
    v: Any,
    t: Any,
    the: Any,
    q: Any,
    cwm: Any,
    q2: Any,
    z: Any,
    ct: float,
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
) -> tuple[Any, Any, Any, float, int, int, float, float]:
    """Return WRF ``MIXLEN`` outputs ``GM``, ``GH``, ``EL`` and scalars."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte

    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    t_arr = jnp.asarray(t)
    the_arr = jnp.asarray(the)
    q_arr = jnp.asarray(q)
    cwm_arr = jnp.asarray(cwm)
    q2_arr = jnp.asarray(q2)
    z_arr = jnp.asarray(z)
    gm = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    gh = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    el = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    q1 = jnp.zeros((int(kte) - int(kts) + 1,), dtype=q2_arr.dtype)
    dth = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    elm = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)
    rel = jnp.zeros((int(kte) - int(kts),), dtype=q2_arr.dtype)

    def kk(k: int) -> int:
        return int(k) - int(kts)

    lpbl = int(lmh)
    for k in range(int(lmh) - 1, 0, -1):
        if float(q2_arr[kk(k)]) <= EPSQ2 * FH:
            lpbl = int(k)
            break
    else:
        lpbl = 1

    pblh = float(z_arr[kk(lpbl + 1)] - z_arr[kk(int(lmh) + 1)])

    for k in range(int(kts), int(lmh) + 1):
        q1 = q1.at[kk(k)].set(0.0)
    for k in range(1, int(lmh)):
        dth = dth.at[kk(k)].set(the_arr[kk(k)] - the_arr[kk(k + 1)])
    for k in range(int(lmh) - 2, 0, -1):
        if float(dth[kk(k)]) > 0.0 and float(dth[kk(k + 1)]) <= 0.0:
            dth = dth.at[kk(k)].set(dth[kk(k)] + float(ct))
            break
    ct_out = 0.0

    for k in range(int(kts), int(lmh)):
        rdz = 2.0 / (z_arr[kk(k)] - z_arr[kk(k + 2)])
        gml = ((u_arr[kk(k)] - u_arr[kk(k + 1)]) ** 2 + (v_arr[kk(k)] - v_arr[kk(k + 1)]) ** 2) * rdz * rdz
        gm = gm.at[kk(k)].set(jnp.maximum(gml, EPSGM))
        tem = (t_arr[kk(k)] + t_arr[kk(k + 1)]) * 0.5
        thm = (the_arr[kk(k)] + the_arr[kk(k + 1)]) * 0.5
        a = thm * P608
        b = (ELOCP / tem - 1.0 - P608) * thm
        ghl = (
            dth[kk(k)] * ((q_arr[kk(k)] + q_arr[kk(k + 1)] + cwm_arr[kk(k)] + cwm_arr[kk(k + 1)]) * (0.5 * P608) + 1.0)
            + (q_arr[kk(k)] - q_arr[kk(k + 1)] + cwm_arr[kk(k)] - cwm_arr[kk(k + 1)]) * a
            + (cwm_arr[kk(k)] - cwm_arr[kk(k + 1)]) * b
        ) * rdz
        if abs(float(ghl)) <= EPSGH:
            ghl = jnp.asarray(EPSGH, dtype=q2_arr.dtype)
        gh = gh.at[kk(k)].set(ghl)

    lmxl = int(lmh)
    for k in range(int(kts), int(lmh)):
        gml = gm[kk(k)]
        ghl = gh[kk(k)]
        if float(ghl) >= EPSGH:
            if float(gml / ghl) <= REQU:
                elm = elm.at[kk(k)].set(EPSL)
                lmxl = int(k)
            else:
                aubr = (AUBM * gml + AUBH * ghl) * ghl
                bubr = BUBM * gml + BUBH * ghl
                qol2st = (-0.5 * bubr + sqrt_for_wrf(bubr * bubr * 0.25 - aubr * CUBR)) * RCUBR
                eloq2x = 1.0 / qol2st
                elm = elm.at[kk(k)].set(jnp.maximum(sqrt_for_wrf(eloq2x * q2_arr[kk(k)]), EPSL))
        else:
            aden = (ADNM * gml + ADNH * ghl) * ghl
            bden = BDNM * gml + BDNH * ghl
            qol2un = -0.5 * bden + sqrt_for_wrf(bden * bden * 0.25 - aden)
            eloq2x = 1.0 / (qol2un + EPSRU)
            elm = elm.at[kk(k)].set(jnp.maximum(sqrt_for_wrf(eloq2x * q2_arr[kk(k)]), EPSL))

    if float(elm[kk(int(lmh) - 1)]) == EPSL:
        lmxl = int(lmh)

    blmx = float(z_arr[kk(lmxl)] - z_arr[kk(int(lmh) + 1)])
    mixht = blmx

    for k in range(lpbl, int(lmh) + 1):
        q1 = q1.at[kk(k)].set(sqrt_for_wrf(q2_arr[kk(k)]))

    szq = jnp.asarray(0.0, dtype=q2_arr.dtype)
    sq = jnp.asarray(0.0, dtype=q2_arr.dtype)
    for k in range(int(kts), int(lmh)):
        qdzl = (q1[kk(k)] + q1[kk(k + 1)]) * (z_arr[kk(k + 1)] - z_arr[kk(k + 2)])
        szq = (z_arr[kk(k + 1)] + z_arr[kk(k + 2)] - z_arr[kk(int(lmh) + 1)] - z_arr[kk(int(lmh) + 1)]) * qdzl + szq
        sq = qdzl + sq

    el0 = jnp.minimum(ALPH * szq * 0.5 / sq, EL0MAX)
    el0 = jnp.maximum(el0, EL0MIN)

    lpblm = max(lpbl - 1, 1)
    for k in range(int(kts), lpblm + 1):
        value = jnp.minimum((z_arr[kk(k)] - z_arr[kk(k + 2)]) * ELFC, elm[kk(k)])
        el = el.at[kk(k)].set(value)
        rel = rel.at[kk(k)].set(value / elm[kk(k)])

    if lpbl < int(lmh):
        for k in range(lpbl, int(lmh)):
            vkrmz = (z_arr[kk(k + 1)] - z_arr[kk(int(lmh) + 1)]) * VKARMAN
            value = jnp.minimum(vkrmz / (vkrmz / el0 + 1.0), elm[kk(k)])
            el = el.at[kk(k)].set(value)
            rel = rel.at[kk(k)].set(value / elm[kk(k)])

    for k in range(lpbl + 1, int(lmh) - 1):
        srel = jnp.minimum(((rel[kk(k - 1)] + rel[kk(k + 1)]) * 0.5 + rel[kk(k)]) * 0.5, rel[kk(k)])
        el = el.at[kk(k)].set(jnp.maximum(srel * elm[kk(k)], EPSL))

    return gm, gh, el, pblh, lpbl, lmxl, ct_out, mixht


def prodq2(
    lmh: int,
    dtturbl: float,
    ustar: float,
    gm: Any,
    gh: Any,
    el: Any,
    q2: Any,
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
    """Return WRF ``PRODQ2`` updated ``EL`` and ``Q2``."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte

    gm_arr = jnp.asarray(gm)
    gh_arr = jnp.asarray(gh)
    el_out = jnp.asarray(el).copy()
    q2_out = jnp.asarray(q2).copy()

    def kk(k: int) -> int:
        return int(k) - int(kts)

    for k in range(1, int(lmh)):
        gml = gm_arr[kk(k)]
        ghl = gh_arr[kk(k)]
        aequ = (AEQM * gml + AEQH * ghl) * ghl
        bequ = BEQM * gml + BEQH * ghl
        eqol2 = -0.5 * bequ + sqrt_for_wrf(bequ * bequ * 0.25 - aequ)

        if (
            float(gml + ghl * ghl) <= EPSTRB
            or (float(ghl) >= EPSGH and float(gml / ghl) <= REQU)
            or float(eqol2) <= EPS2
        ):
            q2_out = q2_out.at[kk(k)].set(EPSQ2)
            el_out = el_out.at[kk(k)].set(EPSL)
        else:
            anum = (ANMM * gml + ANMH * ghl) * ghl
            bnum = BNMM * gml + BNMH * ghl
            aden = (ADNM * gml + ADNH * ghl) * ghl
            bden = BDNM * gml + BDNH * ghl
            cden = 1.0
            arhs = -(anum * bden - bnum * aden) * 2.0
            brhs = -anum * 4.0
            crhs = -bnum * 2.0
            dloq1 = el_out[kk(k)] / sqrt_for_wrf(q2_out[kk(k)])
            eloq21 = 1.0 / eqol2
            eloq11 = sqrt_for_wrf(eloq21)
            eloq31 = eloq21 * eloq11
            eloq41 = eloq21 * eloq21
            eloq51 = eloq21 * eloq31
            rden1 = 1.0 / (aden * eloq41 + bden * eloq21 + cden)
            rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1
            eloq12 = eloq11 + (dloq1 - eloq11) * wrf_glibc_expf_finite(
                rhsp1 * float(dtturbl)
            )
            eloq12 = jnp.maximum(eloq12, EPS1)
            eloq22 = eloq12 * eloq12
            eloq32 = eloq22 * eloq12
            eloq42 = eloq22 * eloq22
            eloq52 = eloq22 * eloq32
            rden2 = 1.0 / (aden * eloq42 + bden * eloq22 + cden)
            rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + RB1
            rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2
            rhst2 = rhs2 / rhsp2
            eloq13 = (
                eloq12
                - rhst2
                + (rhst2 + dloq1 - eloq12)
                * wrf_glibc_expf_finite(rhsp2 * float(dtturbl))
            )
            eloq13 = jnp.maximum(eloq13, EPS1)
            eloqn = eloq13
            if float(eloqn) > EPS1:
                q2_new = el_out[kk(k)] * el_out[kk(k)] / (eloqn * eloqn)
                q2_new = jnp.maximum(q2_new, EPSQ2)
                q2_out = q2_out.at[kk(k)].set(q2_new)
                if float(q2_new) == EPSQ2:
                    el_out = el_out.at[kk(k)].set(EPSL)
            else:
                q2_out = q2_out.at[kk(k)].set(EPSQ2)
                el_out = el_out.at[kk(k)].set(EPSL)

    q2_surface = jnp.maximum((B1 ** (2.0 / 3.0)) * float(ustar) * float(ustar), EPSQ2)
    q2_out = q2_out.at[kk(int(lmh))].set(q2_surface)
    return el_out, q2_out


def vdifq(
    lmh: int,
    dtdif: float,
    q2: Any,
    el: Any,
    z: Any,
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
    """Return WRF ``VDIFQ`` updated ``Q2``."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kte

    q2_out = jnp.asarray(q2).copy()
    el_arr = jnp.asarray(el)
    z_arr = jnp.asarray(z)
    work_len = max(int(lmh) - 1 - int(kts), 0)
    akq = jnp.zeros((work_len + 1,), dtype=q2_out.dtype)
    cm = jnp.zeros((work_len + 1,), dtype=q2_out.dtype)
    cr = jnp.zeros((work_len + 1,), dtype=q2_out.dtype)
    dtoz = jnp.zeros((work_len + 1,), dtype=q2_out.dtype)
    rsq2 = jnp.zeros((work_len + 1,), dtype=q2_out.dtype)

    def kk(k: int) -> int:
        return int(k) - int(kts)

    esqhf = 0.5 * ESQ
    for k in range(int(kts), int(lmh) - 1):
        dtoz = dtoz.at[kk(k)].set((float(dtdif) + float(dtdif)) / (z_arr[kk(k)] - z_arr[kk(k + 2)]))
        akq = akq.at[kk(k)].set(sqrt_for_wrf((q2_out[kk(k)] + q2_out[kk(k + 1)]) * 0.5) * (el_arr[kk(k)] + el_arr[kk(k + 1)]) * esqhf / (z_arr[kk(k + 1)] - z_arr[kk(k + 2)]))
        cr = cr.at[kk(k)].set(-dtoz[kk(k)] * akq[kk(k)])

    cm = cm.at[kk(1)].set(dtoz[kk(1)] * akq[kk(1)] + 1.0)
    rsq2 = rsq2.at[kk(1)].set(q2_out[kk(1)])
    for k in range(int(kts) + 1, int(lmh) - 1):
        cf = -dtoz[kk(k)] * akq[kk(k - 1)] / cm[kk(k - 1)]
        cm = cm.at[kk(k)].set(-cr[kk(k - 1)] * cf + (akq[kk(k - 1)] + akq[kk(k)]) * dtoz[kk(k)] + 1.0)
        rsq2 = rsq2.at[kk(k)].set(-rsq2[kk(k - 1)] * cf + q2_out[kk(k)])

    dtozs = (float(dtdif) + float(dtdif)) / (z_arr[kk(int(lmh) - 1)] - z_arr[kk(int(lmh) + 1)])
    akqs = sqrt_for_wrf((q2_out[kk(int(lmh) - 1)] + q2_out[kk(int(lmh))]) * 0.5) * (el_arr[kk(int(lmh) - 1)] + ELZ0) * esqhf / (
        z_arr[kk(int(lmh))] - z_arr[kk(int(lmh) + 1)]
    )
    cf = -dtozs * akq[kk(int(lmh) - 2)] / cm[kk(int(lmh) - 2)]
    q2_lmh_m1 = (
        dtozs * akqs * q2_out[kk(int(lmh))]
        - rsq2[kk(int(lmh) - 2)] * cf
        + q2_out[kk(int(lmh) - 1)]
    ) / ((akq[kk(int(lmh) - 2)] + akqs) * dtozs - cr[kk(int(lmh) - 2)] * cf + 1.0)
    q2_out = q2_out.at[kk(int(lmh) - 1)].set(q2_lmh_m1)

    for k in range(int(lmh) - 2, int(kts) - 1, -1):
        q2_out = q2_out.at[kk(k)].set((-cr[kk(k)] * q2_out[kk(k + 1)] + rsq2[kk(k)]) / cm[kk(k)])

    return q2_out


def vdifh(
    dtdif: float,
    lmh: int,
    mss: int,
    mse: int,
    lpbl: int,
    sz0: Any,
    rkhs: float,
    clow: Any,
    cts: Any,
    species: Any,
    nspec: int,
    rkh: Any,
    zhk: Any,
    rho: Any,
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
    i: int,
    j: int,
) -> Any:
    """Return WRF ``VDIFH`` updated species columns."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, i, j

    out = jnp.asarray(species).copy()
    sz0_arr = jnp.asarray(sz0)
    clow_arr = jnp.asarray(clow)
    cts_arr = jnp.asarray(cts)
    rkh_arr = jnp.asarray(rkh)
    zhk_arr = jnp.asarray(zhk)
    rho_arr = jnp.asarray(rho)
    kwork = max(int(lmh) - int(kts), 1)
    mwork = int(mse) - int(mss) + 1
    cm = jnp.zeros((kwork + 1,), dtype=out.dtype)
    cr = jnp.zeros((kwork + 1,), dtype=out.dtype)
    dtoz = jnp.zeros((kwork + 1,), dtype=out.dtype)
    rkct = jnp.zeros((mwork, kwork + 1), dtype=out.dtype)
    rss = jnp.zeros((mwork, kwork + 1), dtype=out.dtype)

    def kk(k: int) -> int:
        return int(k) - int(kts)

    def mm(m: int) -> int:
        return int(m) - int(mss)

    for k in range(int(kts), int(lmh)):
        dtoz = dtoz.at[kk(k)].set(float(dtdif) / (zhk_arr[kk(k)] - zhk_arr[kk(k + 1)]))
        cr = cr.at[kk(k)].set(-dtoz[kk(k)] * rkh_arr[kk(k)])
        if k < int(lpbl):
            for m in range(int(mss), int(nspec) + 1):
                rkct = rkct.at[mm(m), kk(k)].set(0.0)
        else:
            rkhz = rkh_arr[kk(k)] * (zhk_arr[kk(k)] - zhk_arr[kk(k + 2)])
            for m in range(int(mss), int(nspec) + 1):
                rkct = rkct.at[mm(m), kk(k)].set(rkhz * cts_arr[mm(m)] * 0.5)

    rhok = rho_arr[kk(int(kts))]
    cm = cm.at[kk(int(kts))].set(dtoz[kk(int(kts))] * rkh_arr[kk(int(kts))] + rhok)
    for m in range(int(mss), int(nspec) + 1):
        rss = rss.at[mm(m), kk(int(kts))].set(-rkct[mm(m), kk(int(kts))] * dtoz[kk(int(kts))] + out[mm(m), kk(int(kts))] * rhok)

    for k in range(int(kts) + 1, int(lmh)):
        dtozl = dtoz[kk(k)]
        cf = -dtozl * rkh_arr[kk(k - 1)] / cm[kk(k - 1)]
        rhok = rho_arr[kk(k)]
        cm = cm.at[kk(k)].set(-cr[kk(k - 1)] * cf + (rkh_arr[kk(k - 1)] + rkh_arr[kk(k)]) * dtozl + rhok)
        for m in range(int(mss), int(nspec) + 1):
            rss = rss.at[mm(m), kk(k)].set(-rss[mm(m), kk(k - 1)] * cf + (rkct[mm(m), kk(k - 1)] - rkct[mm(m), kk(k)]) * dtozl + out[mm(m), kk(k)] * rhok)

    dtozs = float(dtdif) / (zhk_arr[kk(int(lmh))] - zhk_arr[kk(int(lmh) + 1)])
    rkhh = rkh_arr[kk(int(lmh) - 1)]
    cf = -dtozs * rkhh / cm[kk(int(lmh) - 1)]
    cmb = cr[kk(int(lmh) - 1)] * cf
    rhok = rho_arr[kk(int(lmh))]
    for m in range(int(mss), int(nspec) + 1):
        rkss = float(rkhs) * clow_arr[mm(m)]
        cmsb = -cmb + (rkhh + rkss) * dtozs + rhok
        rssb = -rss[mm(m), kk(int(lmh) - 1)] * cf + rkct[mm(m), kk(int(lmh) - 1)] * dtozs + out[mm(m), kk(int(lmh))] * rhok
        out = out.at[mm(m), kk(int(lmh))].set((dtozs * rkss * sz0_arr[mm(m)] + rssb) / cmsb)

    for k in range(int(lmh) - 1, int(kts) - 1, -1):
        rcml = 1.0 / cm[kk(k)]
        for m in range(int(mss), int(nspec) + 1):
            out = out.at[mm(m), kk(k)].set((-cr[kk(k)] * out[mm(m), kk(k + 1)] + rss[mm(m), kk(k)]) * rcml)

    return out


def vdifv(
    lmh: int,
    dtdif: float,
    uz0: float,
    vz0: float,
    rkms: float,
    u: Any,
    v: Any,
    rkm: Any,
    z: Any,
    rho: Any,
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
    i: int,
    j: int,
) -> tuple[Any, Any]:
    """Return WRF ``VDIFV`` updated ``U`` and ``V`` columns."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kte, i, j

    u_out = jnp.asarray(u).copy()
    v_out = jnp.asarray(v).copy()
    rkm_arr = jnp.asarray(rkm)
    z_arr = jnp.asarray(z)
    rho_arr = jnp.asarray(rho)
    work_len = max(int(lmh) - int(kts), 1)
    cm = jnp.zeros((work_len + 1,), dtype=u_out.dtype)
    cr = jnp.zeros((work_len + 1,), dtype=u_out.dtype)
    dtoz = jnp.zeros((work_len + 1,), dtype=u_out.dtype)
    rsu = jnp.zeros((work_len + 1,), dtype=u_out.dtype)
    rsv = jnp.zeros((work_len + 1,), dtype=u_out.dtype)

    def kk(k: int) -> int:
        return int(k) - int(kts)

    for k in range(1, int(lmh)):
        dtoz = dtoz.at[kk(k)].set(float(dtdif) / (z_arr[kk(k)] - z_arr[kk(k + 1)]))
        cr = cr.at[kk(k)].set(-dtoz[kk(k)] * rkm_arr[kk(k)])

    rhok = rho_arr[kk(1)]
    cm = cm.at[kk(1)].set(dtoz[kk(1)] * rkm_arr[kk(1)] + rhok)
    rsu = rsu.at[kk(1)].set(u_out[kk(1)] * rhok)
    rsv = rsv.at[kk(1)].set(v_out[kk(1)] * rhok)

    for k in range(2, int(lmh)):
        dtozl = dtoz[kk(k)]
        cf = -dtozl * rkm_arr[kk(k - 1)] / cm[kk(k - 1)]
        rhok = rho_arr[kk(k)]
        cm = cm.at[kk(k)].set(-cr[kk(k - 1)] * cf + (rkm_arr[kk(k - 1)] + rkm_arr[kk(k)]) * dtozl + rhok)
        rsu = rsu.at[kk(k)].set(-rsu[kk(k - 1)] * cf + u_out[kk(k)] * rhok)
        rsv = rsv.at[kk(k)].set(-rsv[kk(k - 1)] * cf + v_out[kk(k)] * rhok)

    dtozs = float(dtdif) / (z_arr[kk(int(lmh))] - z_arr[kk(int(lmh) + 1)])
    rkmh = rkm_arr[kk(int(lmh) - 1)]
    cf = -dtozs * rkmh / cm[kk(int(lmh) - 1)]
    rhok = rho_arr[kk(int(lmh))]
    rcmvb = 1.0 / ((rkmh + float(rkms)) * dtozs - cr[kk(int(lmh) - 1)] * cf + rhok)
    dtozak = dtozs * float(rkms)
    u_out = u_out.at[kk(int(lmh))].set((dtozak * float(uz0) - rsu[kk(int(lmh) - 1)] * cf + u_out[kk(int(lmh))] * rhok) * rcmvb)
    v_out = v_out.at[kk(int(lmh))].set((dtozak * float(vz0) - rsv[kk(int(lmh) - 1)] * cf + v_out[kk(int(lmh))] * rhok) * rcmvb)

    for k in range(int(lmh) - 1, 0, -1):
        rcml = 1.0 / cm[kk(k)]
        u_out = u_out.at[kk(k)].set((-cr[kk(k)] * u_out[kk(k + 1)] + rsu[kk(k)]) * rcml)
        v_out = v_out.at[kk(k)].set((-cr[kk(k)] * v_out[kk(k + 1)] + rsv[kk(k)]) * rcml)

    return u_out, v_out


__all__ = [
    "EPSQ2",
    "MyjPblInitResult",
    "MyjPblResult",
    "myjpbl",
    "myjpblinit",
    "difcof",
    "mixlen",
    "prodq2",
    "vdifq",
    "vdifh",
    "vdifv",
]
