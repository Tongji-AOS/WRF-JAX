"""Diffusion/deformation kernels translated from WRF ``module_diffusion_em``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from wrf_jax.core.real_arithmetic import real_assignment
from wrf_jax.core.optimization import optimization_barrier


from wrf_jax.core.wrf_math import sqrt_for_wrf
from wrf_jax.boundary.physical_bc import set_physical_bc2d, set_physical_bc3d

P_R12 = 1
P_R13 = 2
P_R23 = 3
P_M11 = 0
P_M22 = 1
P_M33 = 2
P_M12 = 3
P_M13 = 4
P_M23 = 5
G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
P1000MB = 100000.0
XLV = 2.5e6
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
EP_2 = R_D / R_V
PARAM_FIRST_SCALAR = 2
P_QV = 2
P_QC = 3
P_QI = 4
PRANDTL = 1.0 / 3.0


@dataclass(frozen=True)
class DeformAndDivResult:
    div: Any
    defor11: Any
    defor22: Any
    defor33: Any
    defor12: Any
    defor13: Any
    defor23: Any
    nba_rij: Any | None = None


@dataclass(frozen=True)
class TiTauResult:
    titau: Any
    mtau: Any


@dataclass(frozen=True)
class CalculateN2Result:
    bn2: Any
    moist: Any


@dataclass(frozen=True)
class DiffusionCoefficientsResult:
    xkmh: Any
    xkmv: Any
    xkhh: Any
    xkhv: Any


@dataclass(frozen=True)
class CalculateKmKhResult:
    xkmh: Any
    xkmv: Any
    xkhh: Any
    xkhv: Any
    bn2: Any
    moist: Any
    tke: Any
    hpbl: Any
    xkmv_meso: Any


@dataclass(frozen=True)
class DiffMetricsResult:
    z: Any
    rdz: Any
    rdzw: Any
    zx: Any
    zy: Any


@dataclass(frozen=True)
class HorizontalDiffusionSResult:
    tendency: Any


@dataclass(frozen=True)
class HorizontalDiffusionU2Result:
    tendency: Any
    nba_mij: Any


@dataclass(frozen=True)
class HorizontalDiffusionV2Result:
    tendency: Any
    nba_mij: Any


@dataclass(frozen=True)
class HorizontalDiffusionW2Result:
    tendency: Any
    nba_mij: Any


@dataclass(frozen=True)
class HorizontalDiffusion2Result:
    rt_tendf: Any
    ru_tendf: Any
    rv_tendf: Any
    rw_tendf: Any
    tke_tendf: Any
    moist_tendf: Any
    chem_tendf: Any
    scalar_tendf: Any
    tracer_tendf: Any
    nba_mij: Any


@dataclass(frozen=True)
class PhyBcResult:
    div: Any
    defor11: Any
    defor22: Any
    defor33: Any
    defor12: Any
    defor13: Any
    defor23: Any
    xkmh: Any
    xkmv: Any
    xkhh: Any
    xkhv: Any
    tke: Any
    rho: Any
    rublten: Any
    rvblten: Any
    rucuten: Any
    rvcuten: Any
    rushten: Any
    rvshten: Any
    gamu: Any
    gamv: Any
    xkmv_meso: Any


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


def _flag(config: Any, name: str) -> bool:
    return bool(_read(config, name, False))


def _int_value(config: Any, name: str, default: int = 0) -> int:
    return int(_read(config, name, default))


def phy_bc(
    config_flags: Any,
    div: Any,
    defor11: Any,
    defor22: Any,
    defor33: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    xkhv: Any,
    tke: Any,
    rho: Any,
    rublten: Any,
    rvblten: Any,
    rucuten: Any,
    rvcuten: Any,
    rushten: Any,
    rvshten: Any,
    gamu: Any,
    gamv: Any,
    xkmv_meso: Any,
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
    *,
    set_physical_bc3d_fn=set_physical_bc3d,
    set_physical_bc2d_fn=set_physical_bc2d,
) -> PhyBcResult:
    """Apply WRF diffusion physical boundary conditions to active arrays."""

    def bc3(dat: Any, variable: str) -> Any:
        return set_physical_bc3d_fn(
            dat,
            variable,
            config_flags,
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
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )

    def bc2(dat: Any, variable: str) -> Any:
        return set_physical_bc2d_fn(
            dat,
            variable,
            config_flags,
            ids,
            ide,
            jds,
            jde,
            ims,
            ime,
            jms,
            jme,
            ips,
            ipe,
            jps,
            jpe,
            its,
            ite,
            jts,
            jte,
        )

    rublten_out = rublten
    rvblten_out = rvblten
    if _int_value(config_flags, "bl_pbl_physics") > 0:
        rublten_out = bc3(rublten_out, "t")
        rvblten_out = bc3(rvblten_out, "t")

    rucuten_out = rucuten
    rvcuten_out = rvcuten
    if _int_value(config_flags, "cu_physics") > 0:
        rucuten_out = bc3(rucuten_out, "t")
        rvcuten_out = bc3(rvcuten_out, "t")

    rushten_out = rushten
    rvshten_out = rvshten
    if _int_value(config_flags, "shcu_physics") > 0:
        rushten_out = bc3(rushten_out, "t")
        rvshten_out = bc3(rvshten_out, "t")

    xkmh_out = bc3(xkmh, "t")
    xkhh_out = bc3(xkhh, "t")

    xkmv_out = xkmv
    xkhv_out = xkhv
    div_out = div
    defor11_out = defor11
    defor22_out = defor22
    defor33_out = defor33
    defor12_out = defor12
    defor13_out = defor13
    defor23_out = defor23
    rho_out = rho

    if _int_value(config_flags, "diff_opt") == 2:
        xkmv_out = bc3(xkmv_out, "t")
        xkhv_out = bc3(xkhv_out, "t")
        div_out = bc3(div_out, "t")
        defor11_out = bc3(defor11_out, "t")
        defor22_out = bc3(defor22_out, "t")
        defor33_out = bc3(defor33_out, "t")
        defor12_out = bc3(defor12_out, "d")
        defor13_out = bc3(defor13_out, "e")
        defor23_out = bc3(defor23_out, "f")
        rho_out = bc3(rho_out, "t")

    gamu_out = gamu
    gamv_out = gamv
    xkmv_meso_out = xkmv_meso
    if _int_value(config_flags, "diff_opt") == 2 and _int_value(config_flags, "km_opt") == 5:
        gamu_out = bc2(gamu_out, "t")
        gamv_out = bc2(gamv_out, "t")
        xkmv_meso_out = bc3(xkmv_meso_out, "t")

    return PhyBcResult(
        div=div_out,
        defor11=defor11_out,
        defor22=defor22_out,
        defor33=defor33_out,
        defor12=defor12_out,
        defor13=defor13_out,
        defor23=defor23_out,
        xkmh=xkmh_out,
        xkmv=xkmv_out,
        xkhh=xkhh_out,
        xkhv=xkhv_out,
        tke=tke,
        rho=rho_out,
        rublten=rublten_out,
        rvblten=rvblten_out,
        rucuten=rucuten_out,
        rvcuten=rvcuten_out,
        rushten=rushten_out,
        rvshten=rvshten_out,
        gamu=gamu_out,
        gamv=gamv_out,
        xkmv_meso=xkmv_meso_out,
    )


def cal_deform_and_div(
    config_flags: Any,
    u: Any,
    v: Any,
    w: Any,
    div: Any,
    defor11: Any,
    defor22: Any,
    defor33: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    nba_rij: Any | None,
    n_nba_rij: int,
    u_base: Any,
    v_base: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    rdx: float,
    rdy: float,
    dn: Any,
    dnw: Any,
    rdz: Any,
    rdzw: Any,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
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
) -> DeformAndDivResult:
    """Compute WRF deformation tensor components and divergence."""

    del kds, ime, jme, kme, n_nba_rij
    u_arr = jnp.asarray(u)
    v_arr = jnp.asarray(v)
    w_arr = jnp.asarray(w)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdz_arr = jnp.asarray(rdz)
    rdzw_arr = jnp.asarray(rdzw)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    dn_arr = jnp.asarray(dn)
    dnw_arr = jnp.asarray(dnw)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    u_base_arr = jnp.asarray(u_base)
    v_base_arr = jnp.asarray(v_base)

    div_out = jnp.asarray(div).copy()
    defor11_out = jnp.asarray(defor11).copy()
    defor22_out = jnp.asarray(defor22).copy()
    defor33_out = jnp.asarray(defor33).copy()
    defor12_out = jnp.asarray(defor12).copy()
    defor13_out = jnp.asarray(defor13).copy()
    defor23_out = jnp.asarray(defor23).copy()
    nba_out = None if nba_rij is None else jnp.asarray(nba_rij).copy()

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set3(arr: Any, i: int, k: int, j: int, value: Any) -> Any:
        return arr.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    def set_nba(i: int, k: int, j: int, p: int, value: Any) -> None:
        nonlocal nba_out
        if nba_out is None:
            raise ValueError("nba_rij is required when sfs_opt > 0")
        nba_out = nba_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms), int(p) - 1].set(value)

    ktf = min(int(kte), int(kde) - 1)
    ktes1 = int(kte) - 1
    ktes2 = int(kte) - 2
    cft2 = -0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)
    cft1 = 1.0 - cft2

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)

    mm: dict[tuple[int, int], Any] = {}
    hat: dict[tuple[int, int, int], Any] = {}
    hatavg: dict[tuple[int, int, int], Any] = {}
    tmp1: dict[tuple[int, int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            mm[(i, j)] = a2(msftx_arr, i, j) * a2(msfty_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 2):
                hat[(i, k, j)] = a3(u_arr, i, k, j) / a2(msfuy_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (hat[(i, k, j)] + hat[(i + 1, k, j)])
                    + av(fnp_arr, k) * (hat[(i, k - 1, j)] + hat[(i + 1, k - 1, j)])
                )

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            hatavg[(i, 1, j)] = 0.5 * (
                float(cf1) * hat[(i, 1, j)]
                + float(cf2) * hat[(i, 2, j)]
                + float(cf3) * hat[(i, 3, j)]
                + float(cf1) * hat[(i + 1, 1, j)]
                + float(cf2) * hat[(i + 1, 2, j)]
                + float(cf3) * hat[(i + 1, 3, j)]
            )
            hatavg[(i, int(kte), j)] = 0.5 * (
                cft1 * (hat[(i, ktes1, j)] + hat[(i + 1, ktes1, j)])
                + cft2 * (hat[(i, ktes2, j)] + hat[(i + 1, ktes2, j)])
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmpzx = 0.25 * (
                    a3(zx_arr, i, k, j)
                    + a3(zx_arr, i + 1, k, j)
                    + a3(zx_arr, i, k + 1, j)
                    + a3(zx_arr, i + 1, k + 1, j)
                )
                tmp1[(i, k, j)] = (hatavg[(i, k + 1, j)] - hatavg[(i, k, j)]) * tmpzx * a3(rdzw_arr, i, k, j)
                tmp1[(i, k, j)] = mm[(i, j)] * (float(rdx) * (hat[(i + 1, k, j)] - hat[(i, k, j)]) - tmp1[(i, k, j)])
                defor11_out = set3(defor11_out, i, k, j, 2.0 * tmp1[(i, k, j)])
                div_out = set3(div_out, i, k, j, tmp1[(i, k, j)])

    for j in range(j_start, j_end + 2):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                if _flag(config_flags, "polar") and (j == int(jds) or j == int(jde)):
                    hat[(i, k, j)] = jnp.asarray(0.0, dtype=v_arr.dtype)
                else:
                    hat[(i, k, j)] = a3(v_arr, i, k, j) / a2(msfvx_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (hat[(i, k, j)] + hat[(i, k, j + 1)])
                    + av(fnp_arr, k) * (hat[(i, k - 1, j)] + hat[(i, k - 1, j + 1)])
                )

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            hatavg[(i, 1, j)] = 0.5 * (
                float(cf1) * hat[(i, 1, j)]
                + float(cf2) * hat[(i, 2, j)]
                + float(cf3) * hat[(i, 3, j)]
                + float(cf1) * hat[(i, 1, j + 1)]
                + float(cf2) * hat[(i, 2, j + 1)]
                + float(cf3) * hat[(i, 3, j + 1)]
            )
            hatavg[(i, int(kte), j)] = 0.5 * (
                cft1 * (hat[(i, ktes1, j)] + hat[(i, ktes1, j + 1)])
                + cft2 * (hat[(i, ktes2, j)] + hat[(i, ktes2, j + 1)])
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmpzy = 0.25 * (
                    a3(zy_arr, i, k, j)
                    + a3(zy_arr, i, k, j + 1)
                    + a3(zy_arr, i, k + 1, j)
                    + a3(zy_arr, i, k + 1, j + 1)
                )
                tmp1[(i, k, j)] = (hatavg[(i, k + 1, j)] - hatavg[(i, k, j)]) * tmpzy * a3(rdzw_arr, i, k, j)
                tmp1[(i, k, j)] = mm[(i, j)] * (float(rdy) * (hat[(i, k, j + 1)] - hat[(i, k, j)]) - tmp1[(i, k, j)])
                defor22_out = set3(defor22_out, i, k, j, 2.0 * tmp1[(i, k, j)])
                div_out = set3(div_out, i, k, j, a3(div_out, i, k, j) + tmp1[(i, k, j)])

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmp1[(i, k, j)] = (a3(w_arr, i, k + 1, j) - a3(w_arr, i, k, j)) * a3(rdzw_arr, i, k, j)
                defor33_out = set3(defor33_out, i, k, j, 2.0 * tmp1[(i, k, j)])
                div_out = set3(div_out, i, k, j, a3(div_out, i, k, j) + tmp1[(i, k, j)])

    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            mm[(i, j)] = 0.25 * (a2(msfux_arr, i, j - 1) + a2(msfux_arr, i, j)) * (
                a2(msfvy_arr, i - 1, j) + a2(msfvy_arr, i, j)
            )

    for j in range(j_start - 1, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                hat[(i, k, j)] = a3(u_arr, i, k, j) / a2(msfux_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (hat[(i, k, j - 1)] + hat[(i, k, j)])
                    + av(fnp_arr, k) * (hat[(i, k - 1, j - 1)] + hat[(i, k - 1, j)])
                )

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            hatavg[(i, 1, j)] = 0.5 * (
                float(cf1) * hat[(i, 1, j - 1)]
                + float(cf2) * hat[(i, 2, j - 1)]
                + float(cf3) * hat[(i, 3, j - 1)]
                + float(cf1) * hat[(i, 1, j)]
                + float(cf2) * hat[(i, 2, j)]
                + float(cf3) * hat[(i, 3, j)]
            )
            hatavg[(i, int(kte), j)] = 0.5 * (
                cft1 * (hat[(i, ktes1, j - 1)] + hat[(i, ktes1, j)])
                + cft2 * (hat[(i, ktes2, j - 1)] + hat[(i, ktes2, j)])
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmpzy = 0.25 * (
                    a3(zy_arr, i - 1, k, j)
                    + a3(zy_arr, i, k, j)
                    + a3(zy_arr, i - 1, k + 1, j)
                    + a3(zy_arr, i, k + 1, j)
                )
                tmp1[(i, k, j)] = (hatavg[(i, k + 1, j)] - hatavg[(i, k, j)]) * 0.25 * tmpzy * (
                    a3(rdzw_arr, i, k, j)
                    + a3(rdzw_arr, i - 1, k, j)
                    + a3(rdzw_arr, i - 1, k, j - 1)
                    + a3(rdzw_arr, i, k, j - 1)
                )
                defor12_out = set3(
                    defor12_out,
                    i,
                    k,
                    j,
                    mm[(i, j)] * (float(rdy) * (hat[(i, k, j)] - hat[(i, k, j - 1)]) - tmp1[(i, k, j)]),
                )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start - 1, i_end + 1):
                hat[(i, k, j)] = a3(v_arr, i, k, j) / a2(msfvy_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (hat[(i - 1, k, j)] + hat[(i, k, j)])
                    + av(fnp_arr, k) * (hat[(i - 1, k - 1, j)] + hat[(i, k - 1, j)])
                )

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            hatavg[(i, 1, j)] = 0.5 * (
                float(cf1) * hat[(i - 1, 1, j)]
                + float(cf2) * hat[(i - 1, 2, j)]
                + float(cf3) * hat[(i - 1, 3, j)]
                + float(cf1) * hat[(i, 1, j)]
                + float(cf2) * hat[(i, 2, j)]
                + float(cf3) * hat[(i, 3, j)]
            )
            hatavg[(i, int(kte), j)] = 0.5 * (
                cft1 * (hat[(i, ktes1, j)] + hat[(i - 1, ktes1, j)])
                + cft2 * (hat[(i, ktes2, j)] + hat[(i - 1, ktes2, j)])
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmpzx = 0.25 * (
                    a3(zx_arr, i, k, j - 1)
                    + a3(zx_arr, i, k, j)
                    + a3(zx_arr, i, k + 1, j - 1)
                    + a3(zx_arr, i, k + 1, j)
                )
                tmp1[(i, k, j)] = (hatavg[(i, k + 1, j)] - hatavg[(i, k, j)]) * 0.25 * tmpzx * (
                    a3(rdzw_arr, i, k, j)
                    + a3(rdzw_arr, i, k, j - 1)
                    + a3(rdzw_arr, i - 1, k, j - 1)
                    + a3(rdzw_arr, i - 1, k, j)
                )
                second = mm[(i, j)] * (float(rdx) * (hat[(i, k, j)] - hat[(i - 1, k, j)]) - tmp1[(i, k, j)])
                if _int_value(config_flags, "sfs_opt", 0) > 0:
                    set_nba(i, k, j, P_R12, a3(defor12_out, i, k, j) - second)
                defor12_out = set3(defor12_out, i, k, j, a3(defor12_out, i, k, j) + second)

    if _int_value(config_flags, "sfs_opt", 0) > 0:
        raise NotImplementedError("cal_deform_and_div NBA boundary updates are not yet implemented")

    if not _flag(config_flags, "periodic_x") and i_start == int(ids) + 1:
        for j in range(int(jts), int(jte) + 1):
            for k in range(int(kts), int(kte) + 1):
                defor12_out = set3(defor12_out, int(ids), k, j, a3(defor12_out, int(ids) + 1, k, j))
    if not _flag(config_flags, "periodic_y") and j_start == int(jds) + 1:
        for k in range(int(kts), int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                defor12_out = set3(defor12_out, i, k, int(jds), a3(defor12_out, i, k, int(jds) + 1))
    if not _flag(config_flags, "periodic_x") and i_end == int(ide) - 1:
        for j in range(int(jts), int(jte) + 1):
            for k in range(int(kts), int(kte) + 1):
                defor12_out = set3(defor12_out, int(ide), k, j, a3(defor12_out, int(ide) - 1, k, j))
    if not _flag(config_flags, "periodic_y") and j_end == int(jde) - 1:
        for k in range(int(kts), int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                defor12_out = set3(defor12_out, i, k, int(jde), a3(defor12_out, i, k, int(jde) - 1))

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide))
    if _flag(config_flags, "periodic_y"):
        j_end = min(int(jte), int(jde))

    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            mm[(i, j)] = a2(msfux_arr, i, j) * a2(msfuy_arr, i, j)
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), int(kte) + 1):
            for i in range(i_start, i_end + 1):
                hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msfty_arr, i, j)
    i = i_start - 1
    for j in range(j_start, min(int(jte), int(jde) - 1) + 1):
        for k in range(int(kts), int(kte) + 1):
            hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msfty_arr, i, j)
    j = j_start - 1
    for k in range(int(kts), int(kte) + 1):
        for i in range(i_start, min(int(ite), int(ide) - 1) + 1):
            hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msfty_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.25 * (
                    hat[(i, k, j)] + hat[(i, k + 1, j)] + hat[(i - 1, k, j)] + hat[(i - 1, k + 1, j)]
                )
    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                tmp1[(i, k, j)] = (hatavg[(i, k, j)] - hatavg[(i, k - 1, j)]) * a3(zx_arr, i, k, j) * 0.5 * (
                    a3(rdz_arr, i, k, j) + a3(rdz_arr, i - 1, k, j)
                )
                defor13_out = set3(
                    defor13_out,
                    i,
                    k,
                    j,
                    mm[(i, j)] * (float(rdx) * (hat[(i, k, j)] - hat[(i - 1, k, j)]) - tmp1[(i, k, j)]),
                )
    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            defor13_out = set3(defor13_out, i, int(kts), j, 0.0)
            defor13_out = set3(defor13_out, i, ktf + 1, j, 0.0)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                if _flag(config_flags, "mix_full_fields"):
                    value = (a3(u_arr, i, k, j) - a3(u_arr, i, k - 1, j)) * 0.5 * (
                        a3(rdz_arr, i, k, j) + a3(rdz_arr, i - 1, k, j)
                    )
                else:
                    value = (a3(u_arr, i, k, j) - av(u_base_arr, k) - a3(u_arr, i, k - 1, j) + av(u_base_arr, k - 1)) * 0.5 * (
                        a3(rdz_arr, i, k, j) + a3(rdz_arr, i - 1, k, j)
                    )
                tmp1[(i, k, j)] = value
                defor13_out = set3(defor13_out, i, k, j, a3(defor13_out, i, k, j) + value)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "periodic_y"):
        j_end = min(int(jte), int(jde))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            mm[(i, j)] = a2(msfvx_arr, i, j) * a2(msfvy_arr, i, j)
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), int(kte) + 1):
            for i in range(i_start, i_end + 1):
                hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msftx_arr, i, j)
    i = i_start - 1
    for j in range(j_start, min(int(jte), int(jde) - 1) + 1):
        for k in range(int(kts), int(kte) + 1):
            hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msftx_arr, i, j)
    j = j_start - 1
    for k in range(int(kts), int(kte) + 1):
        for i in range(i_start, min(int(ite), int(ide) - 1) + 1):
            hat[(i, k, j)] = a3(w_arr, i, k, j) / a2(msftx_arr, i, j)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                hatavg[(i, k, j)] = 0.25 * (
                    hat[(i, k, j)] + hat[(i, k + 1, j)] + hat[(i, k, j - 1)] + hat[(i, k + 1, j - 1)]
                )
    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                tmp1[(i, k, j)] = (hatavg[(i, k, j)] - hatavg[(i, k - 1, j)]) * a3(zy_arr, i, k, j) * 0.5 * (
                    a3(rdz_arr, i, k, j) + a3(rdz_arr, i, k, j - 1)
                )
                defor23_out = set3(
                    defor23_out,
                    i,
                    k,
                    j,
                    mm[(i, j)] * (float(rdy) * (hat[(i, k, j)] - hat[(i, k, j - 1)]) - tmp1[(i, k, j)]),
                )
    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            defor23_out = set3(defor23_out, i, int(kts), j, 0.0)
            defor23_out = set3(defor23_out, i, ktf + 1, j, 0.0)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                if _flag(config_flags, "mix_full_fields"):
                    value = (a3(v_arr, i, k, j) - a3(v_arr, i, k - 1, j)) * 0.5 * (
                        a3(rdz_arr, i, k, j) + a3(rdz_arr, i, k, j - 1)
                    )
                else:
                    value = (a3(v_arr, i, k, j) - av(v_base_arr, k) - a3(v_arr, i, k - 1, j) + av(v_base_arr, k - 1)) * 0.5 * (
                        a3(rdz_arr, i, k, j) + a3(rdz_arr, i, k, j - 1)
                    )
                tmp1[(i, k, j)] = value
                defor23_out = set3(defor23_out, i, k, j, a3(defor23_out, i, k, j) + value)

    if not _flag(config_flags, "periodic_x") and i_start == int(ids) + 1:
        for j in range(int(jts), int(jte) + 1):
            for k in range(int(kts), int(kte) + 1):
                defor13_out = set3(defor13_out, int(ids), k, j, a3(defor13_out, int(ids) + 1, k, j))
                defor23_out = set3(defor23_out, int(ids), k, j, a3(defor23_out, int(ids) + 1, k, j))
    if not _flag(config_flags, "periodic_y") and j_start == int(jds) + 1:
        for k in range(int(kts), int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                defor13_out = set3(defor13_out, i, k, int(jds), a3(defor13_out, i, k, int(jds) + 1))
                defor23_out = set3(defor23_out, i, k, int(jds), a3(defor23_out, i, k, int(jds) + 1))
    if not _flag(config_flags, "periodic_x") and i_end == int(ide) - 1:
        for j in range(int(jts), int(jte) + 1):
            for k in range(int(kts), int(kte) + 1):
                defor13_out = set3(defor13_out, int(ide), k, j, a3(defor13_out, int(ide) - 1, k, j))
                defor23_out = set3(defor23_out, int(ide), k, j, a3(defor23_out, int(ide) - 1, k, j))
    if not _flag(config_flags, "periodic_y") and j_end == int(jde) - 1:
        for k in range(int(kts), int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                defor13_out = set3(defor13_out, i, k, int(jde), a3(defor13_out, i, k, int(jde) - 1))
                defor23_out = set3(defor23_out, i, k, int(jde), a3(defor23_out, i, k, int(jde) - 1))

    return DeformAndDivResult(
        div=div_out,
        defor11=defor11_out,
        defor22=defor22_out,
        defor33=defor33_out,
        defor12=defor12_out,
        defor13=defor13_out,
        defor23=defor23_out,
        nba_rij=nba_out,
    )


def cal_titau_11_22_33(
    config_flags: Any,
    titau: Any,
    tke: Any,
    xkx: Any,
    defor: Any,
    mtau: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
) -> TiTauResult:
    """Compute WRF ``titau`` for the 11/22/33 normal components."""

    del tke, kds, ime, jme, kme
    titau_out = jnp.asarray(titau).copy()
    mtau_out = jnp.asarray(mtau).copy()
    xkx_arr = jnp.asarray(xkx)
    defor_arr = jnp.asarray(defor)
    rho_arr = jnp.asarray(rho)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def t_idx(i: int, k: int, j: int) -> tuple[int, int, int]:
        return _idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set_titau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal titau_out
        titau_out = titau_out.at[t_idx(i, k, j)].set(value)

    def set_mtau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal mtau_out
        mtau_out = mtau_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)

    i_start -= int(is_ext)
    i_end += int(ie_ext)
    j_start -= int(js_ext)
    j_end += int(je_ext)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                if _int_value(config_flags, "sfs_opt", 0) > 0:
                    set_titau(i, k, j, a3(rho_arr, i, k, j) * a3(mtau_out, i, k, j))
                elif _int_value(config_flags, "m_opt", 0) == 1:
                    mtau_value = -a3(xkx_arr, i, k, j) * a3(defor_arr, i, k, j)
                    set_titau(i, k, j, a3(rho_arr, i, k, j) * mtau_value)
                    set_mtau(i, k, j, mtau_value)
                else:
                    set_titau(i, k, j, -a3(rho_arr, i, k, j) * a3(xkx_arr, i, k, j) * a3(defor_arr, i, k, j))

    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_12_21(
    config_flags: Any,
    titau: Any,
    xkx: Any,
    defor: Any,
    mtau: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
) -> TiTauResult:
    """Compute WRF ``titau`` for the 12/21 shear components."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau).copy()
    mtau_out = jnp.asarray(mtau).copy()
    xkx_arr = jnp.asarray(xkx)
    defor_arr = jnp.asarray(defor)
    rho_arr = jnp.asarray(rho)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def t_idx(i: int, k: int, j: int) -> tuple[int, int, int]:
        return _idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def set_titau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal titau_out
        titau_out = titau_out.at[t_idx(i, k, j)].set(value)

    def set_mtau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal mtau_out
        mtau_out = mtau_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)

    i_start -= int(is_ext)
    i_end += int(ie_ext)
    j_start -= int(js_ext)
    j_end += int(je_ext)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                rhoavg = 0.25 * (
                    a3(rho_arr, i - 1, k, j)
                    + a3(rho_arr, i, k, j)
                    + a3(rho_arr, i - 1, k, j - 1)
                    + a3(rho_arr, i, k, j - 1)
                )
                xkxavg = rhoavg * 0.25 * (
                    a3(xkx_arr, i - 1, k, j)
                    + a3(xkx_arr, i, k, j)
                    + a3(xkx_arr, i - 1, k, j - 1)
                    + a3(xkx_arr, i, k, j - 1)
                )
                if _int_value(config_flags, "sfs_opt", 0) > 0:
                    set_titau(i, k, j, rhoavg * a3(mtau_out, i, k, j))
                elif _int_value(config_flags, "m_opt", 0) == 1:
                    titau_value = -xkxavg * a3(defor_arr, i, k, j)
                    set_titau(i, k, j, titau_value)
                    set_mtau(i, k, j, titau_value / rhoavg)
                else:
                    set_titau(i, k, j, -xkxavg * a3(defor_arr, i, k, j))

    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_13_31(
    config_flags: Any,
    titau: Any,
    defor: Any,
    mtau: Any,
    xkx: Any,
    fnm: Any,
    fnp: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
) -> TiTauResult:
    """Compute WRF ``titau`` for the 13/31 shear components."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau).copy()
    mtau_out = jnp.asarray(mtau).copy()
    defor_arr = jnp.asarray(defor)
    xkx_arr = jnp.asarray(xkx)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rho_arr = jnp.asarray(rho)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def t_idx(i: int, k: int, j: int) -> tuple[int, int, int]:
        return _idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_titau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal titau_out
        titau_out = titau_out.at[t_idx(i, k, j)].set(value)

    def set_mtau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal mtau_out
        mtau_out = mtau_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)

    i_start -= int(is_ext)
    i_end += int(ie_ext)
    j_start -= int(js_ext)
    j_end += int(je_ext)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                rhoavg = 0.5 * (
                    av(fnm_arr, k) * (a3(rho_arr, i - 1, k, j) + a3(rho_arr, i, k, j))
                    + av(fnp_arr, k) * (a3(rho_arr, i - 1, k - 1, j) + a3(rho_arr, i, k - 1, j))
                )
                xkxavg = rhoavg * 0.5 * (
                    av(fnm_arr, k) * (a3(xkx_arr, i, k, j) + a3(xkx_arr, i - 1, k, j))
                    + av(fnp_arr, k) * (a3(xkx_arr, i, k - 1, j) + a3(xkx_arr, i - 1, k - 1, j))
                )
                if _int_value(config_flags, "sfs_opt", 0) > 0:
                    set_titau(i, k, j, rhoavg * a3(mtau_out, i, k, j))
                elif _int_value(config_flags, "m_opt", 0) == 1:
                    titau_value = -xkxavg * a3(defor_arr, i, k, j)
                    set_titau(i, k, j, titau_value)
                    set_mtau(i, k, j, titau_value / rhoavg)
                else:
                    set_titau(i, k, j, -xkxavg * a3(defor_arr, i, k, j))

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            set_titau(i, int(kts), j, 0.0)
            set_titau(i, ktf + 1, j, 0.0)

    return TiTauResult(titau=titau_out, mtau=mtau_out)


def cal_titau_23_32(
    config_flags: Any,
    titau: Any,
    defor: Any,
    mtau: Any,
    xkx: Any,
    fnm: Any,
    fnp: Any,
    rho: Any,
    is_ext: int,
    ie_ext: int,
    js_ext: int,
    je_ext: int,
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
) -> TiTauResult:
    """Compute WRF ``titau`` for the 23/32 shear components."""

    del kds, ime, jme, kme
    titau_out = jnp.asarray(titau).copy()
    mtau_out = jnp.asarray(mtau).copy()
    defor_arr = jnp.asarray(defor)
    xkx_arr = jnp.asarray(xkx)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    rho_arr = jnp.asarray(rho)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def t_idx(i: int, k: int, j: int) -> tuple[int, int, int]:
        return _idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_titau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal titau_out
        titau_out = titau_out.at[t_idx(i, k, j)].set(value)

    def set_mtau(i: int, k: int, j: int, value: Any) -> None:
        nonlocal mtau_out
        mtau_out = mtau_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    i_start -= int(is_ext)
    i_end += int(ie_ext)
    j_start -= int(js_ext)
    j_end += int(je_ext)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                rhoavg = 0.5 * (
                    av(fnm_arr, k) * (a3(rho_arr, i, k, j) + a3(rho_arr, i, k, j - 1))
                    + av(fnp_arr, k) * (a3(rho_arr, i, k - 1, j) + a3(rho_arr, i, k - 1, j - 1))
                )
                xkxavg = rhoavg * 0.5 * (
                    av(fnm_arr, k) * (a3(xkx_arr, i, k, j) + a3(xkx_arr, i, k, j - 1))
                    + av(fnp_arr, k) * (a3(xkx_arr, i, k - 1, j) + a3(xkx_arr, i, k - 1, j - 1))
                )
                if _int_value(config_flags, "sfs_opt", 0) > 0:
                    set_titau(i, k, j, rhoavg * a3(mtau_out, i, k, j))
                elif _int_value(config_flags, "m_opt", 0) == 1:
                    titau_value = -xkxavg * a3(defor_arr, i, k, j)
                    set_titau(i, k, j, titau_value)
                    set_mtau(i, k, j, titau_value / rhoavg)
                else:
                    set_titau(i, k, j, -xkxavg * a3(defor_arr, i, k, j))

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            set_titau(i, int(kts), j, 0.0)
            set_titau(i, ktf + 1, j, 0.0)

    return TiTauResult(titau=titau_out, mtau=mtau_out)


def calculate_n2(
    config_flags: Any,
    bn2: Any,
    moist: Any,
    theta: Any,
    t: Any,
    p: Any,
    p8w: Any,
    t8w: Any,
    dnw: Any,
    dn: Any,
    rdz: Any,
    rdzw: Any,
    n_moist: int,
    cf1: float,
    cf2: float,
    cf3: float,
    warm_rain: bool,
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
    param_first_scalar: int = PARAM_FIRST_SCALAR,
    p_qv: int = P_QV,
    p_qc: int = P_QC,
    p_qi: int = P_QI,
) -> CalculateN2Result:
    """Compute WRF moist Brunt-Vaisala frequency helper ``BN2``."""

    del warm_rain, kds, ime, jme, kme
    bn2_out = jnp.asarray(bn2).copy()
    moist_arr = jnp.asarray(moist)
    theta_arr = jnp.asarray(theta)
    t_arr = jnp.asarray(t)
    p_arr = jnp.asarray(p)
    p8w_arr = jnp.asarray(p8w)
    t8w_arr = jnp.asarray(t8w)
    dnw_arr = jnp.asarray(dnw)
    dn_arr = jnp.asarray(dn)
    rdz_arr = jnp.asarray(rdz)
    rdzw_arr = jnp.asarray(rdzw)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def m4(i: int, k: int, j: int, species: int) -> Any:
        return moist_arr[_idx(i, ims), _idx(k, kms), _idx(j, jms), int(species) - 1]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_bn2(i: int, k: int, j: int, value: Any) -> None:
        nonlocal bn2_out
        bn2_out = bn2_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    qc_cr = 0.00001
    ktf = min(int(kte), int(kde) - 1)
    ktes1 = int(kte) - 1
    ktes2 = int(kte) - 2

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    tmp1: dict[tuple[int, int, int], Any] = {}
    qvs: dict[tuple[int, int, int], Any] = {}
    qctmp: dict[tuple[int, int, int], Any] = {}
    tmp1sfc: dict[tuple[int, int], Any] = {}
    tmp1top: dict[tuple[int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                qctmp[(i, k, j)] = m4(i, k, j, p_qc) if int(p_qc) > int(param_first_scalar) else jnp.asarray(0.0, dtype=moist_arr.dtype)

    for j in range(int(jts), int(jte) + 1):
        for k in range(int(kts), int(kte) + 1):
            for i in range(int(its), int(ite) + 1):
                tmp1[(i, k, j)] = jnp.asarray(0.0, dtype=moist_arr.dtype)
    for j in range(int(jts), int(jte) + 1):
        for i in range(int(its), int(ite) + 1):
            tmp1sfc[(i, j)] = jnp.asarray(0.0, dtype=moist_arr.dtype)
            tmp1top[(i, j)] = jnp.asarray(0.0, dtype=moist_arr.dtype)

    for ispe in range(int(param_first_scalar), int(n_moist) + 1):
        if ispe in (int(p_qv), int(p_qc), int(p_qi)):
            for j in range(j_start, j_end + 1):
                for k in range(int(kts), ktf + 1):
                    for i in range(i_start, i_end + 1):
                        tmp1[(i, k, j)] = tmp1[(i, k, j)] + m4(i, k, j, ispe)
            for j in range(j_start, j_end + 1):
                for i in range(i_start, i_end + 1):
                    tmp1sfc[(i, j)] = tmp1sfc[(i, j)] + float(cf1) * m4(i, 1, j, ispe) + float(cf2) * m4(i, 2, j, ispe) + float(
                        cf3
                    ) * m4(i, 3, j, ispe)
                    tmp1top[(i, j)] = tmp1top[(i, j)] + m4(i, ktes1, j, ispe) + (
                        m4(i, ktes1, j, ispe) - m4(i, ktes2, j, ispe)
                    ) * 0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tc = a3(t_arr, i, k, j) - SVPT0
                es = 1000.0 * SVP1 * jnp.exp(SVP2 * tc / (a3(t_arr, i, k, j) - SVP3))
                qvs[(i, k, j)] = EP_2 * es / (a3(p_arr, i, k, j) - es)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf):
            for i in range(i_start, i_end + 1):
                tmpdz = 1.0 / a3(rdz_arr, i, k, j) + 1.0 / a3(rdz_arr, i, k + 1, j)
                saturated = (m4(i, k, j, p_qv) >= qvs[(i, k, j)]) | (qctmp[(i, k, j)] >= qc_cr)
                xlvqv = XLV * m4(i, k, j, p_qv)
                coefa = (1.0 + xlvqv / R_D / a3(t_arr, i, k, j)) / (
                    1.0 + XLV * xlvqv / CP / R_V / a3(t_arr, i, k, j) / a3(t_arr, i, k, j)
                ) / a3(theta_arr, i, k, j)
                thetaep1 = a3(theta_arr, i, k + 1, j) * (1.0 + XLV * qvs[(i, k + 1, j)] / CP / a3(t_arr, i, k + 1, j))
                thetaem1 = a3(theta_arr, i, k - 1, j) * (1.0 + XLV * qvs[(i, k - 1, j)] / CP / a3(t_arr, i, k - 1, j))
                saturated_bn2 = G * (coefa * (thetaep1 - thetaem1) / tmpdz - (tmp1[(i, k + 1, j)] - tmp1[(i, k - 1, j)]) / tmpdz)
                dry_bn2 = G * (
                    (a3(theta_arr, i, k + 1, j) - a3(theta_arr, i, k - 1, j)) / a3(theta_arr, i, k, j) / tmpdz
                    + 1.61 * (m4(i, k + 1, j, p_qv) - m4(i, k - 1, j, p_qv)) / tmpdz
                    - (tmp1[(i, k + 1, j)] - tmp1[(i, k - 1, j)]) / tmpdz
                )
                set_bn2(i, k, j, jnp.where(saturated, saturated_bn2, dry_bn2))

    k = int(kts)
    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            tmpdz_moist = 1.0 / a3(rdz_arr, i, k + 1, j) + 0.5 / a3(rdzw_arr, i, k, j)
            thetasfc = a3(t8w_arr, i, int(kts), j) / (a3(p8w_arr, i, k, j) / P1000MB) ** (R_D / CP)
            saturated = (m4(i, k, j, p_qv) >= qvs[(i, k, j)]) | (qctmp[(i, k, j)] >= qc_cr)
            qvsfc_sat = float(cf1) * qvs[(i, 1, j)] + float(cf2) * qvs[(i, 2, j)] + float(cf3) * qvs[(i, 3, j)]
            xlvqv = XLV * m4(i, k, j, p_qv)
            coefa = (1.0 + xlvqv / R_D / a3(t_arr, i, k, j)) / (
                1.0 + XLV * xlvqv / CP / R_V / a3(t_arr, i, k, j) / a3(t_arr, i, k, j)
            ) / a3(theta_arr, i, k, j)
            thetaep1 = a3(theta_arr, i, k + 1, j) * (1.0 + XLV * qvs[(i, k + 1, j)] / CP / a3(t_arr, i, k + 1, j))
            thetaesfc = thetasfc * (1.0 + XLV * qvsfc_sat / CP / a3(t8w_arr, i, int(kts), j))
            saturated_bn2 = G * (coefa * (thetaep1 - thetaesfc) / tmpdz_moist - (tmp1[(i, k + 1, j)] - tmp1sfc[(i, j)]) / tmpdz_moist)

            qvsfc_dry = float(cf1) * m4(i, 1, j, p_qv) + float(cf2) * m4(i, 2, j, p_qv) + float(cf3) * m4(i, 3, j, p_qv)
            tmpdz_dry = 1.0 / a3(rdzw_arr, i, k, j)
            dry_bn2 = G * (
                (a3(theta_arr, i, k + 1, j) - a3(theta_arr, i, k, j)) / a3(theta_arr, i, k, j) / tmpdz_dry
                + 1.61 * (m4(i, k + 1, j, p_qv) - qvsfc_dry) / tmpdz_dry
                - (tmp1[(i, k + 1, j)] - tmp1sfc[(i, j)]) / tmpdz_dry
            )
            set_bn2(i, k, j, jnp.where(saturated, saturated_bn2, dry_bn2))

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            set_bn2(i, ktf, j, a3(bn2_out, i, ktf - 1, j))

    return CalculateN2Result(bn2=bn2_out, moist=moist_arr)


def smag2d_km(
    config_flags: Any,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    xkhv: Any,
    defor11: Any,
    defor22: Any,
    defor12: Any,
    rdzw: Any,
    dx: float,
    dy: float,
    msftx: Any,
    msfty: Any,
    zx: Any,
    zy: Any,
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
) -> DiffusionCoefficientsResult:
    """Compute WRF 2-D Smagorinsky diffusion coefficients."""

    del kds, ime, jme, kme
    xkmh_out = jnp.asarray(xkmh).copy()
    xkmv_out = jnp.asarray(xkmv).copy()
    xkhh_out = jnp.asarray(xkhh).copy()
    xkhv_out = jnp.asarray(xkhv).copy()
    defor11_arr = jnp.asarray(defor11)
    defor22_arr = jnp.asarray(defor22)
    defor12_arr = jnp.asarray(defor12)
    rdzw_arr = jnp.asarray(rdzw)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def set_coeffs(i: int, k: int, j: int, xkmh_value: Any, xkmv_value: Any, xkhh_value: Any, xkhv_value: Any) -> None:
        nonlocal xkmh_out, xkmv_out, xkhh_out, xkhv_out
        idx = (_idx(i, ims), _idx(k, kms), _idx(j, jms))
        xkmh_out = xkmh_out.at[idx].set(xkmh_value)
        xkmv_out = xkmv_out.at[idx].set(xkmv_value)
        xkhh_out = xkhh_out.at[idx].set(xkhh_value)
        xkhv_out = xkhv_out.at[idx].set(xkhv_value)

    ktf = min(int(kte), int(kde) - 1)
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    pr = PRANDTL
    c_s = float(_read(config_flags, "c_s", 0.25))
    diff_opt = _int_value(config_flags, "diff_opt", 0)

    def2: dict[tuple[int, int, int], Any] = {}
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                value = 0.25 * (a3(defor11_arr, i, k, j) - a3(defor22_arr, i, k, j)) * (
                    a3(defor11_arr, i, k, j) - a3(defor22_arr, i, k, j)
                )
                tmp = 0.25 * (
                    a3(defor12_arr, i, k, j)
                    + a3(defor12_arr, i, k, j + 1)
                    + a3(defor12_arr, i + 1, k, j)
                    + a3(defor12_arr, i + 1, k, j + 1)
                )
                def2[(i, k, j)] = value + tmp * tmp

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                mlen_h = sqrt_for_wrf(float(dx) / a2(msftx_arr, i, j) * float(dy) / a2(msfty_arr, i, j))
                tmp = sqrt_for_wrf(def2[(i, k, j)])
                xkmh_value = c_s * c_s * mlen_h * mlen_h * tmp
                xkmh_value = jnp.minimum(xkmh_value, 10.0 * mlen_h)
                xkmv_value = xkmh_value
                xkhh_value = xkmh_value / pr
                xkhv_value = jnp.asarray(0.0, dtype=xkmh_out.dtype)
                if diff_opt == 2:
                    dxm = float(dx) / a2(msftx_arr, i, j)
                    dym = float(dy) / a2(msfty_arr, i, j)
                    tmpzx = (
                        0.25
                        * (
                            jnp.abs(a3(zx_arr, i, k, j))
                            + jnp.abs(a3(zx_arr, i + 1, k, j))
                            + jnp.abs(a3(zx_arr, i, k + 1, j))
                            + jnp.abs(a3(zx_arr, i + 1, k + 1, j))
                        )
                        * a3(rdzw_arr, i, k, j)
                        * dxm
                    )
                    tmpzy = (
                        0.25
                        * (
                            jnp.abs(a3(zy_arr, i, k, j))
                            + jnp.abs(a3(zy_arr, i, k, j + 1))
                            + jnp.abs(a3(zy_arr, i, k + 1, j))
                            + jnp.abs(a3(zy_arr, i, k + 1, j + 1))
                        )
                        * a3(rdzw_arr, i, k, j)
                        * dym
                    )
                    alpha = jnp.maximum(sqrt_for_wrf(tmpzx * tmpzx + tmpzy * tmpzy), 1.0)
                    def_limit = max(10.0 / float(mlen_h), 1.0e-3)
                    divisor = jnp.where(tmp > def_limit, alpha * alpha, alpha)
                    xkmh_value = xkmh_value / divisor
                    xkhh_value = xkmh_value / pr
                    xkmv_value = xkmh_value
                set_coeffs(i, k, j, xkmh_value, xkmv_value, xkhh_value, xkhv_value)

    return DiffusionCoefficientsResult(xkmh=xkmh_out, xkmv=xkmv_out, xkhh=xkhh_out, xkhv=xkhv_out)


def calculate_km_kh(
    config_flags: Any,
    dt: float,
    dampcoef: float,
    zdamp: float,
    damp_opt: int,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    xkhv: Any,
    bn2: Any,
    khdif: float,
    kvdif: float,
    div: Any,
    defor11: Any,
    defor22: Any,
    defor33: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    tke: Any,
    p8w: Any,
    t8w: Any,
    theta: Any,
    t: Any,
    p: Any,
    moist: Any,
    dn: Any,
    dnw: Any,
    dx: float,
    dy: float,
    rdz: Any,
    rdzw: Any,
    isotropic: int,
    n_moist: int,
    cf1: float,
    cf2: float,
    cf3: float,
    warm_rain: bool,
    mix_upper_bound: float,
    msftx: Any,
    msfty: Any,
    zx: Any,
    zy: Any,
    hpbl: Any,
    dlk: Any,
    xkmv_meso: Any,
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
    param_first_scalar: int = PARAM_FIRST_SCALAR,
    p_qv: int = P_QV,
    p_qc: int = P_QC,
    p_qi: int = P_QI,
) -> CalculateKmKhResult:
    """Compute WRF diffusion coefficients for the call-graph-covered branch."""

    del dt, dampcoef, zdamp, khdif, kvdif, div, defor33, defor13, defor23, isotropic, mix_upper_bound, dlk
    tke_out = jnp.asarray(tke).copy()
    hpbl_out = jnp.asarray(hpbl).copy()
    xkmv_meso_out = jnp.asarray(xkmv_meso).copy()

    n2_result = calculate_n2(
        config_flags,
        bn2,
        moist,
        theta,
        t,
        p,
        p8w,
        t8w,
        dnw,
        dn,
        rdz,
        rdzw,
        n_moist,
        cf1,
        cf2,
        cf3,
        warm_rain,
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
        param_first_scalar=param_first_scalar,
        p_qv=p_qv,
        p_qc=p_qc,
        p_qi=p_qi,
    )

    km_opt = _int_value(config_flags, "km_opt", 0)
    if km_opt != 4:
        raise NotImplementedError("calculate_km_kh currently implements the call-graph-covered km_opt=4 branch")

    coeffs = smag2d_km(
        config_flags,
        xkmh,
        xkmv,
        xkhh,
        xkhv,
        defor11,
        defor22,
        defor12,
        rdzw,
        dx,
        dy,
        msftx,
        msfty,
        zx,
        zy,
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

    if int(damp_opt) == 1:
        raise NotImplementedError("calculate_km_kh damping branch requires cal_dampkm, which is outside the current call graph")

    return CalculateKmKhResult(
        xkmh=coeffs.xkmh,
        xkmv=coeffs.xkmv,
        xkhh=coeffs.xkhh,
        xkhv=coeffs.xkhv,
        bn2=n2_result.bn2,
        moist=n2_result.moist,
        tke=tke_out,
        hpbl=hpbl_out,
        xkmv_meso=xkmv_meso_out,
    )


def compute_diff_metrics(
    config_flags: Any,
    ph: Any,
    phb: Any,
    z: Any,
    rdz: Any,
    rdzw: Any,
    zx: Any,
    zy: Any,
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
    rounding_token: Any | None = None,
) -> DiffMetricsResult:
    """Compute WRF diffusion metric arrays from geopotential."""

    del kds, ime, jme, kme
    ph_arr = jnp.asarray(ph)
    phb_arr = jnp.asarray(phb)
    z_out = jnp.asarray(z).copy()
    rdz_out = jnp.asarray(rdz).copy()
    rdzw_out = jnp.asarray(rdzw).copy()
    zx_out = jnp.asarray(zx).copy()
    zy_out = jnp.asarray(zy).copy()

    def _store_real(value: Any) -> Any:
        value32 = jnp.asarray(value, dtype=jnp.float32)
        if rounding_token is None:
            return value32
        return optimization_barrier(real_assignment(value32, rounding_token))

    def _divide_real(left: Any, right: Any) -> Any:
        if rounding_token is None:
            return jnp.asarray(left, dtype=jnp.float32) / jnp.asarray(
                right, dtype=jnp.float32
            )
        dynamic_denominator = _store_real(right)
        return _store_real(
            jnp.asarray(left, dtype=jnp.float32) / dynamic_denominator
        )

    one = jnp.asarray(1.0, dtype=jnp.float32)
    two = jnp.asarray(2.0, dtype=jnp.float32)
    half = jnp.asarray(0.5, dtype=jnp.float32)
    gravity = jnp.asarray(G, dtype=jnp.float32)
    rdx_real = jnp.asarray(rdx, dtype=jnp.float32)
    rdy_real = jnp.asarray(rdy, dtype=jnp.float32)

    def _max_fortran_index(arr: Any, axis: int, lower: int) -> int:
        return int(lower) + int(arr.shape[axis]) - 1

    ph_i_max = _max_fortran_index(ph_arr, 0, ims)
    ph_k_max = _max_fortran_index(ph_arr, 1, kms)
    ph_j_max = _max_fortran_index(ph_arr, 2, jms)
    out_i_max = _max_fortran_index(z_out, 0, ims)
    out_k_max = _max_fortran_index(z_out, 1, kms)
    out_j_max = _max_fortran_index(z_out, 2, jms)

    def _frange(start: int, end: int, low: int, high: int) -> Any:
        first = max(int(start), int(low))
        last = min(int(end), int(high))
        return jnp.arange(first, last + 1, dtype=jnp.int32) if first <= last else jnp.arange(0, 0, dtype=jnp.int32)

    def _idxs(values: Any, lower: int) -> Any:
        return values - int(lower)

    def _set_block(arr: Any, i_vals: Any, k_vals: Any, j_vals: Any, values: Any) -> Any:
        if int(i_vals.size) == 0 or int(k_vals.size) == 0 or int(j_vals.size) == 0:
            return arr
        ii = _idxs(i_vals, ims)
        kk = _idxs(k_vals, kms)
        jj = _idxs(j_vals, jms)
        return arr.at[ii[:, None, None], kk[None, :, None], jj[None, None, :]].set(values)

    z_at_w = _store_real(ph_arr + phb_arr)
    z_at_w = _divide_real(z_at_w, gravity)
    ktf = min(int(kte), int(kde) - 1, ph_k_max - int(kms), out_k_max)

    i_metrics = _frange(int(its), min(int(ite), int(ide) - 1), int(ims), min(ph_i_max, out_i_max))
    j_metrics = _frange(int(jts) - 1, int(jte), int(jms), min(ph_j_max, out_j_max))
    k_full = _frange(1, int(kte), int(kms), ph_k_max)
    k_metric = _frange(1, ktf, int(kms), min(ph_k_max - 1, out_k_max))

    if int(i_metrics.size) and int(j_metrics.size) and int(k_metric.size):
        ii = _idxs(i_metrics, ims)
        kk = _idxs(k_metric, kms)
        jj = _idxs(j_metrics, jms)
        dz_w = _store_real(
            z_at_w[ii[:, None, None], kk[None, :, None] + 1, jj[None, None, :]]
            - z_at_w[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
        )
        rdzw_values = _divide_real(one, dz_w)
        rdzw_out = _set_block(rdzw_out, i_metrics, k_metric, j_metrics, rdzw_values)

        k_rdz_mid = _frange(2, ktf, int(kms), min(ph_k_max - 1, out_k_max))
        if int(k_rdz_mid.size):
            kk_mid = _idxs(k_rdz_mid, kms)
            dz_mass = _store_real(
                z_at_w[ii[:, None, None], kk_mid[None, :, None] + 1, jj[None, None, :]]
                - z_at_w[ii[:, None, None], kk_mid[None, :, None] - 1, jj[None, None, :]]
            )
            rdz_mid = _divide_real(two, dz_mass)
            rdz_out = _set_block(rdz_out, i_metrics, k_rdz_mid, j_metrics, rdz_mid)

        k_bottom = _frange(1, 1, int(kms), min(ph_k_max - 1, out_k_max))
        if int(k_bottom.size):
            dz_bottom = _store_real(
                z_at_w[ii[:, None], 1 - int(kms) + 1, jj[None, :]]
                - z_at_w[ii[:, None], 1 - int(kms), jj[None, :]]
            )
            rdz_bottom = _divide_real(two, dz_bottom)
            rdz_out = rdz_out.at[_idxs(i_metrics, ims)[:, None], 1 - int(kms), _idxs(j_metrics, jms)[None, :]].set(
                rdz_bottom
            )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    i_mass = _frange(i_start, i_end, int(ims), min(ph_i_max, out_i_max))
    j_mass = _frange(j_start, j_end, int(jms), min(ph_j_max, out_j_max))
    k_zxzy = _frange(1, int(kte), int(kms), min(ph_k_max, out_k_max))

    i_x = _frange(max(int(ids) + 1, int(its)), i_end, max(int(ims) + 1, int(ims)), min(ph_i_max, out_i_max))
    if int(i_x.size) and int(j_mass.size) and int(k_zxzy.size):
        ii = _idxs(i_x, ims)
        kk = _idxs(k_zxzy, kms)
        jj = _idxs(j_mass, jms)
        zx_base = _store_real(
            phb_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
            - phb_arr[ii[:, None, None] - 1, kk[None, :, None], jj[None, None, :]]
        )
        zx_base = _store_real(rdx_real * zx_base)
        zx_base = _divide_real(zx_base, gravity)
        zx_perturbation = _store_real(
            ph_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
            - ph_arr[ii[:, None, None] - 1, kk[None, :, None], jj[None, None, :]]
        )
        zx_perturbation = _store_real(rdx_real * zx_perturbation)
        zx_perturbation = _divide_real(zx_perturbation, gravity)
        zx_values = _store_real(zx_base + zx_perturbation)
        zx_out = _set_block(zx_out, i_x, k_zxzy, j_mass, zx_values)

    j_y = _frange(max(int(jds) + 1, int(jts)), j_end, max(int(jms) + 1, int(jms)), min(ph_j_max, out_j_max))
    if int(i_mass.size) and int(j_y.size) and int(k_zxzy.size):
        ii = _idxs(i_mass, ims)
        kk = _idxs(k_zxzy, kms)
        jj = _idxs(j_y, jms)
        zy_base = _store_real(
            phb_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
            - phb_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :] - 1]
        )
        zy_base = _store_real(rdy_real * zy_base)
        zy_base = _divide_real(zy_base, gravity)
        zy_perturbation = _store_real(
            ph_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
            - ph_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :] - 1]
        )
        zy_perturbation = _store_real(rdy_real * zy_perturbation)
        zy_perturbation = _divide_real(zy_perturbation, gravity)
        zy_values = _store_real(zy_base + zy_perturbation)
        zy_out = _set_block(zy_out, i_mass, k_zxzy, j_y, zy_values)

    k_boundary = _frange(1, ktf, int(kms), out_k_max)
    if not _flag(config_flags, "periodic_x"):
        if int(ite) == int(ide):
            i_bdy = _frange(int(ide), int(ide), int(ims), out_i_max)
            zx_out = _set_block(zx_out, i_bdy, k_boundary, j_mass, jnp.zeros((int(i_bdy.size), int(k_boundary.size), int(j_mass.size)), dtype=zx_out.dtype))
        if int(its) == int(ids):
            i_bdy = _frange(int(ids), int(ids), int(ims), out_i_max)
            zx_out = _set_block(zx_out, i_bdy, k_boundary, j_mass, jnp.zeros((int(i_bdy.size), int(k_boundary.size), int(j_mass.size)), dtype=zx_out.dtype))

    if not _flag(config_flags, "periodic_y"):
        if int(jte) == int(jde):
            j_bdy = _frange(int(jde), int(jde), int(jms), out_j_max)
            zy_out = _set_block(zy_out, i_mass, k_boundary, j_bdy, jnp.zeros((int(i_mass.size), int(k_boundary.size), int(j_bdy.size)), dtype=zy_out.dtype))
        if int(jts) == int(jds):
            j_bdy = _frange(int(jds), int(jds), int(jms), out_j_max)
            zy_out = _set_block(zy_out, i_mass, k_boundary, j_bdy, jnp.zeros((int(i_mass.size), int(k_boundary.size), int(j_bdy.size)), dtype=zy_out.dtype))

    k_z = _frange(1, ktf, int(kms), min(ph_k_max - 1, out_k_max))
    if int(i_mass.size) and int(j_mass.size) and int(k_z.size):
        ii = _idxs(i_mass, ims)
        kk = _idxs(k_z, kms)
        jj = _idxs(j_mass, jms)
        z_values = _store_real(
            ph_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
            + phb_arr[ii[:, None, None], kk[None, :, None], jj[None, None, :]]
        )
        z_values = _store_real(
            z_values
            + ph_arr[ii[:, None, None], kk[None, :, None] + 1, jj[None, None, :]]
        )
        z_values = _store_real(
            z_values
            + phb_arr[ii[:, None, None], kk[None, :, None] + 1, jj[None, None, :]]
        )
        z_values = _store_real(half * z_values)
        z_values = _divide_real(z_values, gravity)
        z_out = _set_block(z_out, i_mass, k_z, j_mass, z_values)

    return DiffMetricsResult(z=z_out, rdz=rdz_out, rdzw=rdzw_out, zx=zx_out, zy=zy_out)


def horizontal_diffusion_s(
    tendency: Any,
    config_flags: Any,
    var: Any,
    msftx: Any,
    msfty: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    xkhh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
    rdz: Any,
    rdzw: Any,
    dnw: Any,
    dn: Any,
    rho: Any,
    doing_tke: bool,
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
) -> HorizontalDiffusionSResult:
    """Apply WRF scalar horizontal diffusion tendency update."""

    del msfuy, msfvx, rdz, kds, ime, jme, kme
    tendency_out = jnp.asarray(tendency).copy()
    var_arr = jnp.asarray(var)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    msfux_arr = jnp.asarray(msfux)
    msfvy_arr = jnp.asarray(msfvy)
    xkhh_arr = jnp.asarray(xkhh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    dnw_arr = jnp.asarray(dnw)
    dn_arr = jnp.asarray(dn)
    rho_arr = jnp.asarray(rho)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_tendency(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)
    ktes1 = int(kte) - 1
    ktes2 = int(kte) - 2

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    tmptendf: dict[tuple[int, int, int], Any] = {}
    if doing_tke:
        for j in range(j_start, j_end + 1):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    tmptendf[(i, k, j)] = a3(tendency_out, i, k, j)

    h1avg: dict[tuple[int, int, int], Any] = {}
    h2avg: dict[tuple[int, int, int], Any] = {}
    h1: dict[tuple[int, int, int], Any] = {}
    h2: dict[tuple[int, int, int], Any] = {}
    xkxavg: dict[tuple[int, int, int], Any] = {}
    zx_at_m: dict[tuple[int, int, int], Any] = {}
    zy_at_m: dict[tuple[int, int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 2):
                xkxavg[(i, k, j)] = 0.5 * (a3(xkhh_arr, i - 1, k, j) + a3(xkhh_arr, i, k, j)) * 0.5 * (
                    a3(rho_arr, i - 1, k, j) + a3(rho_arr, i, k, j)
                )
    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 2):
                h1avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (a3(var_arr, i - 1, k, j) + a3(var_arr, i, k, j))
                    + av(fnp_arr, k) * (a3(var_arr, i - 1, k - 1, j) + a3(var_arr, i, k - 1, j))
                )
    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 2):
            h1avg[(i, int(kts), j)] = 0.5 * (
                float(cf1) * a3(var_arr, i, 1, j)
                + float(cf2) * a3(var_arr, i, 2, j)
                + float(cf3) * a3(var_arr, i, 3, j)
                + float(cf1) * a3(var_arr, i - 1, 1, j)
                + float(cf2) * a3(var_arr, i - 1, 2, j)
                + float(cf3) * a3(var_arr, i - 1, 3, j)
            )
            h1avg[(i, ktf + 1, j)] = 0.5 * (
                a3(var_arr, i, ktes1, j)
                + (a3(var_arr, i, ktes1, j) - a3(var_arr, i, ktes2, j)) * 0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)
                + a3(var_arr, i - 1, ktes1, j)
                + (a3(var_arr, i - 1, ktes1, j) - a3(var_arr, i - 1, ktes2, j)) * 0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)
            )
    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 2):
                tmpzx = 0.5 * (a3(zx_arr, i, k, j) + a3(zx_arr, i, k + 1, j))
                rdzu = 2.0 / (1.0 / a3(rdzw_arr, i, k, j) + 1.0 / a3(rdzw_arr, i - 1, k, j))
                h1[(i, k, j)] = -a2(msfux_arr, i, j) * xkxavg[(i, k, j)] * (
                    float(rdx) * (a3(var_arr, i, k, j) - a3(var_arr, i - 1, k, j))
                    - tmpzx * (h1avg[(i, k + 1, j)] - h1avg[(i, k, j)]) * rdzu
                )

    for j in range(j_start, j_end + 2):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                xkxavg[(i, k, j)] = 0.5 * (a3(xkhh_arr, i, k, j - 1) + a3(xkhh_arr, i, k, j)) * 0.5 * (
                    a3(rho_arr, i, k, j - 1) + a3(rho_arr, i, k, j)
                )
    for j in range(j_start, j_end + 2):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                h2avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (a3(var_arr, i, k, j - 1) + a3(var_arr, i, k, j))
                    + av(fnp_arr, k) * (a3(var_arr, i, k - 1, j - 1) + a3(var_arr, i, k - 1, j))
                )
    for j in range(j_start, j_end + 2):
        for i in range(i_start, i_end + 1):
            h2avg[(i, int(kts), j)] = 0.5 * (
                float(cf1) * a3(var_arr, i, 1, j)
                + float(cf2) * a3(var_arr, i, 2, j)
                + float(cf3) * a3(var_arr, i, 3, j)
                + float(cf1) * a3(var_arr, i, 1, j - 1)
                + float(cf2) * a3(var_arr, i, 2, j - 1)
                + float(cf3) * a3(var_arr, i, 3, j - 1)
            )
            h2avg[(i, ktf + 1, j)] = 0.5 * (
                a3(var_arr, i, ktes1, j)
                + (a3(var_arr, i, ktes1, j) - a3(var_arr, i, ktes2, j)) * 0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)
                + a3(var_arr, i, ktes1, j - 1)
                + (a3(var_arr, i, ktes1, j - 1) - a3(var_arr, i, ktes2, j - 1)) * 0.5 * av(dnw_arr, ktes1) / av(dn_arr, ktes1)
            )
    for j in range(j_start, j_end + 2):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                tmpzy = 0.5 * (a3(zy_arr, i, k, j) + a3(zy_arr, i, k + 1, j))
                rdzv = 2.0 / (1.0 / a3(rdzw_arr, i, k, j) + 1.0 / a3(rdzw_arr, i, k, j - 1))
                h2[(i, k, j)] = -a2(msfvy_arr, i, j) * xkxavg[(i, k, j)] * (
                    float(rdy) * (a3(var_arr, i, k, j) - a3(var_arr, i, k, j - 1))
                    - tmpzy * (h2avg[(i, k + 1, j)] - h2avg[(i, k, j)]) * rdzv
                )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                h1avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (h1[(i + 1, k, j)] + h1[(i, k, j)])
                    + av(fnp_arr, k) * (h1[(i + 1, k - 1, j)] + h1[(i, k - 1, j)])
                )
                h2avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (h2[(i, k, j + 1)] + h2[(i, k, j)])
                    + av(fnp_arr, k) * (h2[(i, k - 1, j + 1)] + h2[(i, k - 1, j)])
                )
                zx_at_m[(i, k, j)] = 0.25 * (
                    a3(zx_arr, i, k, j) + a3(zx_arr, i + 1, k, j) + a3(zx_arr, i, k + 1, j) + a3(zx_arr, i + 1, k + 1, j)
                )
                zy_at_m[(i, k, j)] = 0.25 * (
                    a3(zy_arr, i, k, j) + a3(zy_arr, i, k, j + 1) + a3(zy_arr, i, k + 1, j) + a3(zy_arr, i, k + 1, j + 1)
                )
    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            h1avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            h1avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            h2avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            h2avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            zx_at_m[(i, int(kts), j)] = 0.25 * (
                a3(zx_arr, i, int(kts), j)
                + a3(zx_arr, i + 1, int(kts), j)
                + a3(zx_arr, i, int(kts) + 1, j)
                + a3(zx_arr, i + 1, int(kts) + 1, j)
            )
            zy_at_m[(i, int(kts), j)] = 0.25 * (
                a3(zy_arr, i, int(kts), j)
                + a3(zy_arr, i, int(kts), j + 1)
                + a3(zy_arr, i, int(kts) + 1, j)
                + a3(zy_arr, i, int(kts) + 1, j + 1)
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                mrdx = a2(msftx_arr, i, j) * float(rdx)
                mrdy = a2(msfty_arr, i, j) * float(rdy)
                increment = G / (av(dnw_arr, k) * a3(rdzw_arr, i, k, j)) * (
                    mrdx * (h1[(i + 1, k, j)] - h1[(i, k, j)])
                    + mrdy * (h2[(i, k, j + 1)] - h2[(i, k, j)])
                    - a2(msftx_arr, i, j) * zx_at_m[(i, k, j)] * (h1avg[(i, k + 1, j)] - h1avg[(i, k, j)]) * a3(rdzw_arr, i, k, j)
                    - a2(msfty_arr, i, j) * zy_at_m[(i, k, j)] * (h2avg[(i, k + 1, j)] - h2avg[(i, k, j)]) * a3(rdzw_arr, i, k, j)
                )
                set_tendency(i, k, j, a3(tendency_out, i, k, j) + increment)

    if doing_tke:
        for j in range(j_start, j_end + 1):
            for k in range(int(kts), ktf + 1):
                for i in range(i_start, i_end + 1):
                    set_tendency(i, k, j, tmptendf[(i, k, j)] + 2.0 * (a3(tendency_out, i, k, j) - tmptendf[(i, k, j)]))

    return HorizontalDiffusionSResult(tendency=tendency_out)


def _nba_component_index(config_flags: Any, name: str, default_zero_based: int, n_nba_mij: int) -> int:
    fortran_index = int(_read(config_flags, name, default_zero_based + 1))
    index = fortran_index - 1
    if index < 0 or index >= int(n_nba_mij):
        raise ValueError(f"{name}={fortran_index} is outside nba_mij component bounds")
    return index


def horizontal_diffusion_u_2(
    tendency: Any,
    config_flags: Any,
    defor11: Any,
    defor12: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msfux: Any,
    msfuy: Any,
    xkmh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dnw: Any,
    zx: Any,
    zy: Any,
    rdzw: Any,
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
) -> HorizontalDiffusionU2Result:
    """Apply WRF second-order horizontal diffusion to the U tendency."""

    del div, kds, ime, jme, kme
    tendency_out = jnp.asarray(tendency).copy()
    nba_out = jnp.asarray(nba_mij).copy()
    defor11_arr = jnp.asarray(defor11)
    defor12_arr = jnp.asarray(defor12)
    tke_arr = jnp.asarray(tke)
    msfux_arr = jnp.asarray(msfux)
    msfuy_arr = jnp.asarray(msfuy)
    xkmh_arr = jnp.asarray(xkmh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dnw_arr = jnp.asarray(dnw)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    rho_arr = jnp.asarray(rho)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_tendency(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)

    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 1, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = int(ite)

    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)

    p_m11 = _nba_component_index(config_flags, "p_m11", P_M11, int(n_nba_mij))
    p_m12 = _nba_component_index(config_flags, "p_m12", P_M12 if int(n_nba_mij) > P_M12 else P_M11, int(n_nba_mij))

    tau11 = cal_titau_11_22_33(
        config_flags,
        titau1_init,
        tke_arr,
        xkmh_arr,
        defor11_arr,
        nba_out[:, :, :, p_m11],
        rho_arr,
        1,
        0,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau1 = tau11.titau
    nba_out = nba_out.at[:, :, :, p_m11].set(tau11.mtau)

    tau12 = cal_titau_12_21(
        config_flags,
        titau2_init,
        xkmh_arr,
        defor12_arr,
        nba_out[:, :, :, p_m12],
        rho_arr,
        0,
        0,
        0,
        1,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau2 = tau12.titau
    nba_out = nba_out.at[:, :, :, p_m12].set(tau12.mtau)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def tt(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)]

    titau1avg: dict[tuple[int, int, int], Any] = {}
    titau2avg: dict[tuple[int, int, int], Any] = {}
    zx_at_u: dict[tuple[int, int, int], Any] = {}
    zy_at_u: dict[tuple[int, int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                titau1avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (tt(titau1, i - 1, k, j) + tt(titau1, i, k, j))
                    + av(fnp_arr, k) * (tt(titau1, i - 1, k - 1, j) + tt(titau1, i, k - 1, j))
                )
                titau2avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (tt(titau2, i, k, j + 1) + tt(titau2, i, k, j))
                    + av(fnp_arr, k) * (tt(titau2, i, k - 1, j + 1) + tt(titau2, i, k - 1, j))
                )
                zx_at_u[(i, k, j)] = 0.5 * (a3(zx_arr, i, k, j) + a3(zx_arr, i, k + 1, j))
                zy_at_u[(i, k, j)] = 0.125 * (
                    a3(zy_arr, i - 1, k, j)
                    + a3(zy_arr, i, k, j)
                    + a3(zy_arr, i - 1, k, j + 1)
                    + a3(zy_arr, i, k, j + 1)
                    + a3(zy_arr, i - 1, k + 1, j)
                    + a3(zy_arr, i, k + 1, j)
                    + a3(zy_arr, i - 1, k + 1, j + 1)
                    + a3(zy_arr, i, k + 1, j + 1)
                )

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            titau1avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau1avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau2avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau2avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            zx_at_u[(i, int(kts), j)] = 0.5 * (a3(zx_arr, i, int(kts), j) + a3(zx_arr, i, int(kts) + 1, j))
            zy_at_u[(i, int(kts), j)] = 0.125 * (
                a3(zy_arr, i - 1, int(kts), j)
                + a3(zy_arr, i, int(kts), j)
                + a3(zy_arr, i - 1, int(kts), j + 1)
                + a3(zy_arr, i, int(kts), j + 1)
                + a3(zy_arr, i - 1, int(kts) + 1, j)
                + a3(zy_arr, i, int(kts) + 1, j)
                + a3(zy_arr, i - 1, int(kts) + 1, j + 1)
                + a3(zy_arr, i, int(kts) + 1, j + 1)
            )

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                mrdx = a2(msfux_arr, i, j) * float(rdx)
                mrdy = a2(msfuy_arr, i, j) * float(rdy)
                tmpdz = (1.0 / a3(rdzw_arr, i, k, j) + 1.0 / a3(rdzw_arr, i - 1, k, j)) / 2.0
                increment = G * tmpdz / av(dnw_arr, k) * (
                    mrdx * (tt(titau1, i, k, j) - tt(titau1, i - 1, k, j))
                    + mrdy * (tt(titau2, i, k, j + 1) - tt(titau2, i, k, j))
                    - a2(msfux_arr, i, j) * zx_at_u[(i, k, j)] * (titau1avg[(i, k + 1, j)] - titau1avg[(i, k, j)]) / tmpdz
                    - a2(msfuy_arr, i, j) * zy_at_u[(i, k, j)] * (titau2avg[(i, k + 1, j)] - titau2avg[(i, k, j)]) / tmpdz
                )
                set_tendency(i, k, j, a3(tendency_out, i, k, j) + increment)

    return HorizontalDiffusionU2Result(tendency=tendency_out, nba_mij=nba_out)


def horizontal_diffusion_v_2(
    tendency: Any,
    config_flags: Any,
    defor12: Any,
    defor22: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msfvx: Any,
    msfvy: Any,
    xkmh: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dnw: Any,
    zx: Any,
    zy: Any,
    rdzw: Any,
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
) -> HorizontalDiffusionV2Result:
    """Apply WRF second-order horizontal diffusion to the V tendency."""

    del div, kds, ime, jme, kme
    tendency_out = jnp.asarray(tendency).copy()
    nba_out = jnp.asarray(nba_mij).copy()
    defor12_arr = jnp.asarray(defor12)
    defor22_arr = jnp.asarray(defor22)
    tke_arr = jnp.asarray(tke)
    msfvx_arr = jnp.asarray(msfvx)
    msfvy_arr = jnp.asarray(msfvy)
    xkmh_arr = jnp.asarray(xkmh)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dnw_arr = jnp.asarray(dnw)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdzw_arr = jnp.asarray(rdzw)
    rho_arr = jnp.asarray(rho)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_tendency(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 1, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)

    p_m12 = _nba_component_index(config_flags, "p_m12", P_M12 if int(n_nba_mij) > P_M12 else P_M11, int(n_nba_mij))
    p_m22 = _nba_component_index(config_flags, "p_m22", P_M22 if int(n_nba_mij) > P_M22 else P_M11, int(n_nba_mij))

    tau12 = cal_titau_12_21(
        config_flags,
        titau1_init,
        xkmh_arr,
        defor12_arr,
        nba_out[:, :, :, p_m12],
        rho_arr,
        0,
        1,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau1 = tau12.titau
    nba_out = nba_out.at[:, :, :, p_m12].set(tau12.mtau)

    tau22 = cal_titau_11_22_33(
        config_flags,
        titau2_init,
        tke_arr,
        xkmh_arr,
        defor22_arr,
        nba_out[:, :, :, p_m22],
        rho_arr,
        0,
        0,
        1,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau2 = tau22.titau
    nba_out = nba_out.at[:, :, :, p_m22].set(tau22.mtau)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def tt(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)]

    titau1avg: dict[tuple[int, int, int], Any] = {}
    titau2avg: dict[tuple[int, int, int], Any] = {}
    zx_at_v: dict[tuple[int, int, int], Any] = {}
    zy_at_v: dict[tuple[int, int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                titau1avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (tt(titau1, i + 1, k, j) + tt(titau1, i, k, j))
                    + av(fnp_arr, k) * (tt(titau1, i + 1, k - 1, j) + tt(titau1, i, k - 1, j))
                )
                titau2avg[(i, k, j)] = 0.5 * (
                    av(fnm_arr, k) * (tt(titau2, i, k, j - 1) + tt(titau2, i, k, j))
                    + av(fnp_arr, k) * (tt(titau2, i, k - 1, j - 1) + tt(titau2, i, k - 1, j))
                )
                zx_at_v[(i, k, j)] = 0.125 * (
                    a3(zx_arr, i, k, j)
                    + a3(zx_arr, i + 1, k, j)
                    + a3(zx_arr, i, k, j - 1)
                    + a3(zx_arr, i + 1, k, j - 1)
                    + a3(zx_arr, i, k + 1, j)
                    + a3(zx_arr, i + 1, k + 1, j)
                    + a3(zx_arr, i, k + 1, j - 1)
                    + a3(zx_arr, i + 1, k + 1, j - 1)
                )
                zy_at_v[(i, k, j)] = 0.5 * (a3(zy_arr, i, k, j) + a3(zy_arr, i, k + 1, j))

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            titau1avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau1avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau2avg[(i, int(kts), j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau2avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            zx_at_v[(i, int(kts), j)] = 0.125 * (
                a3(zx_arr, i, int(kts), j)
                + a3(zx_arr, i + 1, int(kts), j)
                + a3(zx_arr, i, int(kts), j - 1)
                + a3(zx_arr, i + 1, int(kts), j - 1)
                + a3(zx_arr, i, int(kts) + 1, j)
                + a3(zx_arr, i + 1, int(kts) + 1, j)
                + a3(zx_arr, i, int(kts) + 1, j - 1)
                + a3(zx_arr, i + 1, int(kts) + 1, j - 1)
            )
            zy_at_v[(i, int(kts), j)] = 0.5 * (a3(zy_arr, i, int(kts), j) + a3(zy_arr, i, int(kts) + 1, j))

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                mrdx = a2(msfvx_arr, i, j) * float(rdx)
                mrdy = a2(msfvy_arr, i, j) * float(rdy)
                tmpdz = (1.0 / a3(rdzw_arr, i, k, j) + 1.0 / a3(rdzw_arr, i, k, j - 1)) / 2.0
                increment = G * tmpdz / av(dnw_arr, k) * (
                    mrdy * (tt(titau2, i, k, j) - tt(titau2, i, k, j - 1))
                    + mrdx * (tt(titau1, i + 1, k, j) - tt(titau1, i, k, j))
                    - a2(msfvx_arr, i, j) * zx_at_v[(i, k, j)] * (titau1avg[(i, k + 1, j)] - titau1avg[(i, k, j)]) / tmpdz
                    - a2(msfvy_arr, i, j) * zy_at_v[(i, k, j)] * (titau2avg[(i, k + 1, j)] - titau2avg[(i, k, j)]) / tmpdz
                )
                set_tendency(i, k, j, a3(tendency_out, i, k, j) + increment)

    return HorizontalDiffusionV2Result(tendency=tendency_out, nba_mij=nba_out)


def horizontal_diffusion_w_2(
    tendency: Any,
    config_flags: Any,
    defor13: Any,
    defor23: Any,
    div: Any,
    nba_mij: Any,
    n_nba_mij: int,
    tke: Any,
    msftx: Any,
    msfty: Any,
    xkmv: Any,
    rdx: float,
    rdy: float,
    fnm: Any,
    fnp: Any,
    dn: Any,
    zx: Any,
    zy: Any,
    rdz: Any,
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
) -> HorizontalDiffusionW2Result:
    """Apply WRF second-order horizontal diffusion to the W tendency."""

    del div, tke, kds, ime, jme, kme
    tendency_out = jnp.asarray(tendency).copy()
    nba_out = jnp.asarray(nba_mij).copy()
    defor13_arr = jnp.asarray(defor13)
    defor23_arr = jnp.asarray(defor23)
    msftx_arr = jnp.asarray(msftx)
    msfty_arr = jnp.asarray(msfty)
    xkmv_arr = jnp.asarray(xkmv)
    fnm_arr = jnp.asarray(fnm)
    fnp_arr = jnp.asarray(fnp)
    dn_arr = jnp.asarray(dn)
    zx_arr = jnp.asarray(zx)
    zy_arr = jnp.asarray(zy)
    rdz_arr = jnp.asarray(rdz)
    rho_arr = jnp.asarray(rho)

    def a3(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(k, kms), _idx(j, jms)]

    def a2(arr: Any, i: int, j: int) -> Any:
        return arr[_idx(i, ims), _idx(j, jms)]

    def av(arr: Any, k: int) -> Any:
        return arr[_idx(k, kms)]

    def set_tendency(i: int, k: int, j: int, value: Any) -> None:
        nonlocal tendency_out
        tendency_out = tendency_out.at[_idx(i, ims), _idx(k, kms), _idx(j, jms)].set(value)

    ktf = min(int(kte), int(kde) - 1)

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    if _flag(config_flags, "open_xs") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_start = max(int(ids) + 1, int(its))
    if _flag(config_flags, "open_xe") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        i_end = min(int(ide) - 2, int(ite))
    if _flag(config_flags, "open_ys") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_start = max(int(jds) + 1, int(jts))
    if _flag(config_flags, "open_ye") or _flag(config_flags, "specified") or _flag(config_flags, "nested"):
        j_end = min(int(jde) - 2, int(jte))
    if _flag(config_flags, "periodic_x"):
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    titau_shape = (int(ite) - int(its) + 3, int(kte) - int(kts) + 1, int(jte) - int(jts) + 3)
    titau1_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)
    titau2_init = jnp.zeros(titau_shape, dtype=tendency_out.dtype)

    p_m13 = _nba_component_index(config_flags, "p_m13", P_M13 if int(n_nba_mij) > P_M13 else P_M11, int(n_nba_mij))
    p_m23 = _nba_component_index(config_flags, "p_m23", P_M23 if int(n_nba_mij) > P_M23 else P_M11, int(n_nba_mij))

    tau13 = cal_titau_13_31(
        config_flags,
        titau1_init,
        defor13_arr,
        nba_out[:, :, :, p_m13],
        xkmv_arr,
        fnm_arr,
        fnp_arr,
        rho_arr,
        0,
        1,
        0,
        0,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau1 = tau13.titau
    nba_out = nba_out.at[:, :, :, p_m13].set(tau13.mtau)

    tau23 = cal_titau_23_32(
        config_flags,
        titau2_init,
        defor23_arr,
        nba_out[:, :, :, p_m23],
        xkmv_arr,
        fnm_arr,
        fnp_arr,
        rho_arr,
        0,
        0,
        0,
        1,
        ids,
        ide,
        jds,
        jde,
        int(kms),
        kde,
        ims,
        int(ims) + nba_out.shape[0] - 1,
        jms,
        int(jms) + nba_out.shape[2] - 1,
        kms,
        int(kms) + nba_out.shape[1] - 1,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )
    titau2 = tau23.titau
    nba_out = nba_out.at[:, :, :, p_m23].set(tau23.mtau)

    titau_i_lb = int(its) - 1
    titau_j_lb = int(jts) - 1

    def tt(arr: Any, i: int, k: int, j: int) -> Any:
        return arr[_idx(i, titau_i_lb), _idx(k, kts), _idx(j, titau_j_lb)]

    titau1avg: dict[tuple[int, int, int], Any] = {}
    titau2avg: dict[tuple[int, int, int], Any] = {}
    zx_at_w: dict[tuple[int, int, int], Any] = {}
    zy_at_w: dict[tuple[int, int, int], Any] = {}

    for j in range(j_start, j_end + 1):
        for k in range(int(kts), ktf + 1):
            for i in range(i_start, i_end + 1):
                titau1avg[(i, k, j)] = 0.25 * (
                    tt(titau1, i + 1, k + 1, j)
                    + tt(titau1, i, k + 1, j)
                    + tt(titau1, i + 1, k, j)
                    + tt(titau1, i, k, j)
                )
                titau2avg[(i, k, j)] = 0.25 * (
                    tt(titau2, i, k + 1, j + 1)
                    + tt(titau2, i, k + 1, j)
                    + tt(titau2, i, k, j + 1)
                    + tt(titau2, i, k, j)
                )
                zx_at_w[(i, k, j)] = 0.5 * (a3(zx_arr, i, k, j) + a3(zx_arr, i + 1, k, j))
                zy_at_w[(i, k, j)] = 0.5 * (a3(zy_arr, i, k, j) + a3(zy_arr, i, k, j + 1))

    for j in range(j_start, j_end + 1):
        for i in range(i_start, i_end + 1):
            titau1avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)
            titau2avg[(i, ktf + 1, j)] = jnp.asarray(0.0, dtype=tendency_out.dtype)

    for j in range(j_start, j_end + 1):
        for k in range(int(kts) + 1, ktf + 1):
            for i in range(i_start, i_end + 1):
                mrdx = a2(msftx_arr, i, j) * float(rdx)
                mrdy = a2(msfty_arr, i, j) * float(rdy)
                increment = G / (av(dn_arr, k) * a3(rdz_arr, i, k, j)) * (
                    mrdx * (tt(titau1, i + 1, k, j) - tt(titau1, i, k, j))
                    + mrdy * (tt(titau2, i, k, j + 1) - tt(titau2, i, k, j))
                    - a2(msfty_arr, i, j)
                    * a3(rdz_arr, i, k, j)
                    * (
                        zx_at_w[(i, k, j)] * (titau1avg[(i, k, j)] - titau1avg[(i, k - 1, j)])
                        + zy_at_w[(i, k, j)] * (titau2avg[(i, k, j)] - titau2avg[(i, k - 1, j)])
                    )
                )
                set_tendency(i, k, j, a3(tendency_out, i, k, j) + increment)

    return HorizontalDiffusionW2Result(tendency=tendency_out, nba_mij=nba_out)


def horizontal_diffusion_2(
    rt_tendf: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    rw_tendf: Any,
    tke_tendf: Any,
    moist_tendf: Any,
    n_moist: int,
    chem_tendf: Any,
    n_chem: int,
    scalar_tendf: Any,
    n_scalar: int,
    tracer_tendf: Any,
    n_tracer: int,
    thp: Any,
    theta: Any,
    tke: Any,
    config_flags: Any,
    defor11: Any,
    defor22: Any,
    defor12: Any,
    defor13: Any,
    defor23: Any,
    nba_mij: Any,
    n_nba_mij: int,
    div: Any,
    moist: Any,
    chem: Any,
    scalar: Any,
    tracer: Any,
    msfux: Any,
    msfuy: Any,
    msfvx: Any,
    msfvy: Any,
    msftx: Any,
    msfty: Any,
    xkmh: Any,
    xkmv: Any,
    xkhh: Any,
    km_opt: int,
    rdx: float,
    rdy: float,
    rdz: Any,
    rdzw: Any,
    fnm: Any,
    fnp: Any,
    cf1: float,
    cf2: float,
    cf3: float,
    zx: Any,
    zy: Any,
    dn: Any,
    dnw: Any,
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
) -> HorizontalDiffusion2Result:
    """Apply WRF's second-order horizontal diffusion driver."""

    del theta

    u_result = horizontal_diffusion_u_2(
        ru_tendf,
        config_flags,
        defor11,
        defor12,
        div,
        nba_mij,
        n_nba_mij,
        tke,
        msfux,
        msfuy,
        xkmh,
        rdx,
        rdy,
        fnm,
        fnp,
        dnw,
        zx,
        zy,
        rdzw,
        rho,
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
    ru_out = u_result.tendency
    nba_out = u_result.nba_mij

    v_result = horizontal_diffusion_v_2(
        rv_tendf,
        config_flags,
        defor12,
        defor22,
        div,
        nba_out,
        n_nba_mij,
        tke,
        msfvx,
        msfvy,
        xkmh,
        rdx,
        rdy,
        fnm,
        fnp,
        dnw,
        zx,
        zy,
        rdzw,
        rho,
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
    rv_out = v_result.tendency
    nba_out = v_result.nba_mij

    w_result = horizontal_diffusion_w_2(
        rw_tendf,
        config_flags,
        defor13,
        defor23,
        div,
        nba_out,
        n_nba_mij,
        tke,
        msftx,
        msfty,
        xkmv,
        rdx,
        rdy,
        fnm,
        fnp,
        dn,
        zx,
        zy,
        rdz,
        rho,
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
    rw_out = w_result.tendency
    nba_out = w_result.nba_mij

    rt_out = horizontal_diffusion_s(
        rt_tendf,
        config_flags,
        thp,
        msftx,
        msfty,
        msfux,
        msfuy,
        msfvx,
        msfvy,
        xkhh,
        rdx,
        rdy,
        fnm,
        fnp,
        cf1,
        cf2,
        cf3,
        zx,
        zy,
        rdz,
        rdzw,
        dnw,
        dn,
        rho,
        False,
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
    ).tendency

    if ((int(km_opt) == 2 and not _flag(config_flags, "tke_mix2_off")) or int(km_opt) == 5):
        tke_out = horizontal_diffusion_s(
            tke_tendf,
            config_flags,
            tke,
            msftx,
            msfty,
            msfux,
            msfuy,
            msfvx,
            msfvy,
            xkhh,
            rdx,
            rdy,
            fnm,
            fnp,
            cf1,
            cf2,
            cf3,
            zx,
            zy,
            rdz,
            rdzw,
            dnw,
            dn,
            rho,
            True,
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
        ).tendency
    else:
        tke_out = jnp.asarray(tke_tendf)

    def diffuse_tracers(tend: Any, count: int, values: Any, disabled_flag: str) -> Any:
        out = jnp.asarray(tend).copy()
        if int(count) >= PARAM_FIRST_SCALAR and not _flag(config_flags, disabled_flag):
            vals = jnp.asarray(values)
            for index in range(PARAM_FIRST_SCALAR - 1, int(count)):
                updated = horizontal_diffusion_s(
                    out[:, :, :, index],
                    config_flags,
                    vals[:, :, :, index],
                    msftx,
                    msfty,
                    msfux,
                    msfuy,
                    msfvx,
                    msfvy,
                    xkhh,
                    rdx,
                    rdy,
                    fnm,
                    fnp,
                    cf1,
                    cf2,
                    cf3,
                    zx,
                    zy,
                    rdz,
                    rdzw,
                    dnw,
                    dn,
                    rho,
                    False,
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
                ).tendency
                out = out.at[:, :, :, index].set(updated)
        return out

    moist_out = diffuse_tracers(moist_tendf, n_moist, moist, "moist_mix2_off")
    chem_out = diffuse_tracers(chem_tendf, n_chem, chem, "chem_mix2_off")
    tracer_out = diffuse_tracers(tracer_tendf, n_tracer, tracer, "tracer_mix2_off")
    scalar_out = diffuse_tracers(scalar_tendf, n_scalar, scalar, "scalar_mix2_off")

    return HorizontalDiffusion2Result(
        rt_tendf=rt_out,
        ru_tendf=ru_out,
        rv_tendf=rv_out,
        rw_tendf=rw_out,
        tke_tendf=tke_out,
        moist_tendf=moist_out,
        chem_tendf=chem_out,
        scalar_tendf=scalar_out,
        tracer_tendf=tracer_out,
        nba_mij=nba_out,
    )


__all__ = [
    "CalculateN2Result",
    "CalculateKmKhResult",
    "DeformAndDivResult",
    "DiffusionCoefficientsResult",
    "DiffMetricsResult",
    "HorizontalDiffusionSResult",
    "HorizontalDiffusion2Result",
    "HorizontalDiffusionU2Result",
    "HorizontalDiffusionV2Result",
    "HorizontalDiffusionW2Result",
    "PhyBcResult",
    "EP_2",
    "G",
    "P1000MB",
    "P_M11",
    "P_M12",
    "P_M13",
    "P_M22",
    "P_M23",
    "P_M33",
    "PARAM_FIRST_SCALAR",
    "P_QC",
    "P_QI",
    "P_QV",
    "PRANDTL",
    "P_R12",
    "P_R13",
    "P_R23",
    "R_D",
    "R_V",
    "SVP1",
    "SVP2",
    "SVP3",
    "SVPT0",
    "TiTauResult",
    "XLV",
    "cal_deform_and_div",
    "cal_titau_11_22_33",
    "cal_titau_12_21",
    "cal_titau_13_31",
    "cal_titau_23_32",
    "calculate_n2",
    "calculate_km_kh",
    "compute_diff_metrics",
    "horizontal_diffusion_2",
    "horizontal_diffusion_s",
    "horizontal_diffusion_u_2",
    "horizontal_diffusion_v_2",
    "horizontal_diffusion_w_2",
    "phy_bc",
    "smag2d_km",
]
