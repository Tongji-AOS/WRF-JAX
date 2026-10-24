"""Translation of WRF ``phys/module_sf_sfcdiags``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class SfcdiagsResult:
    q2: Any
    th2: Any
    t2: Any


def sfcdiags(
    hfx: Any,
    qfx: Any,
    tsk: Any,
    qsfc: Any,
    chs2: Any,
    cqs2: Any,
    t2: Any,
    th2: Any,
    q2: Any,
    psfc: Any,
    cp: Any,
    r_d: Any,
    rovcp: Any,
    chs: Any,
    t3d: Any,
    qv3d: Any,
    ua_phys: bool,
    its: int = 1,
    ite: int | None = None,
    jts: int = 1,
    jte: int | None = None,
) -> SfcdiagsResult:
    """Update WRF two-meter surface diagnostics ``Q2``, ``T2`` and ``TH2``."""

    dtype = jnp.result_type(hfx, qfx, tsk, qsfc, chs2, cqs2, t2, th2, q2, psfc, cp, r_d, rovcp, chs, t3d, qv3d, jnp.float32)
    hfx_arr = jnp.asarray(hfx, dtype=dtype)
    qfx_arr = jnp.asarray(qfx, dtype=dtype)
    tsk_arr = jnp.asarray(tsk, dtype=dtype)
    qsfc_arr = jnp.asarray(qsfc, dtype=dtype)
    chs2_arr = jnp.asarray(chs2, dtype=dtype)
    cqs2_arr = jnp.asarray(cqs2, dtype=dtype)
    t2_out = jnp.asarray(t2, dtype=dtype)
    th2_out = jnp.asarray(th2, dtype=dtype)
    q2_out = jnp.asarray(q2, dtype=dtype)
    psfc_arr = jnp.asarray(psfc, dtype=dtype)
    cp_arr = jnp.asarray(cp, dtype=dtype)
    r_d_arr = jnp.asarray(r_d, dtype=dtype)
    rovcp_arr = jnp.asarray(rovcp, dtype=dtype)
    chs_arr = jnp.asarray(chs, dtype=dtype)
    t3d_arr = jnp.asarray(t3d, dtype=dtype)
    qv3d_arr = jnp.asarray(qv3d, dtype=dtype)

    i_start = int(its) - 1
    i_stop = int(ite) if ite is not None else q2_out.shape[0]
    j_start = int(jts) - 1
    j_stop = int(jte) if jte is not None else q2_out.shape[1]
    region = (slice(i_start, i_stop), slice(j_start, j_stop))

    rho = psfc_arr / (r_d_arr * tsk_arr)
    if bool(ua_phys):
        q2_calc = qsfc_arr - chs_arr / cqs2_arr * (qsfc_arr - qv3d_arr[:, 0, :])
        t2_calc = tsk_arr - chs_arr / chs2_arr * (tsk_arr - t3d_arr[:, 0, :])
    else:
        q2_calc = qsfc_arr - qfx_arr / (rho * cqs2_arr)
        t2_calc = tsk_arr - hfx_arr / (rho * cp_arr * chs2_arr)
    q2_calc = jnp.where(cqs2_arr < 1.0e-5, qsfc_arr, q2_calc)
    t2_calc = jnp.where(chs2_arr < 1.0e-5, tsk_arr, t2_calc)
    th2_calc = t2_calc * (1.0e5 / psfc_arr) ** rovcp_arr

    q2_out = q2_out.at[region].set(q2_calc[region])
    t2_out = t2_out.at[region].set(t2_calc[region])
    th2_out = th2_out.at[region].set(th2_calc[region])
    return SfcdiagsResult(q2=q2_out, th2=th2_out, t2=t2_out)


__all__ = ["SfcdiagsResult", "sfcdiags"]
