"""Translations from WRF ``phys/module_mp_thompson``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math
from pathlib import Path
from typing import Any

from jax import custom_jvp, jvp, lax, tree_util
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.dynamic_loop import dynamic_fori_loop
from wrf_jax.core.optimization import optimization_barrier
from wrf_jax.core.real_arithmetic import real_assignment, real_divide
from wrf_jax.core.wrf_math import sqrt_for_wrf
from wrf_jax.physics.module_mp_radar import (
    MELT_OUTSIDE_G,
    MELT_OUTSIDE_S,
    RadarState,
    radar_init,
    rayleigh_soak_wetgraupel,
)


NTB_ARC = 7
NTB_ARW = 9
NTB_ART = 7
NTB_ARR = 5
NTB_ARK = 4
CCN_ACTIVATE_SHAPE = (NTB_ARC, NTB_ARW, NTB_ART, NTB_ARR, NTB_ARK)
NBINS = 100
NBC = NBINS
NBR = NBINS
NBS = NBINS
NBG = NBINS
NTB_C = 37
NTB_I = 64
NTB_R = 37
NTB_S = 37
NTB_G = 37
NTB_G1 = 37
NTB_R1 = 37
NTB_I1 = 55
NTB_T = 9
NTB_IN = 55

TA_NA = jnp.asarray([10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0], dtype=jnp.float32)
TA_WW = jnp.asarray([0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0], dtype=jnp.float32)
TA_TK = jnp.asarray([243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15], dtype=jnp.float32)

RE_QC_BG = jnp.float32(2.49e-6)
RE_QI_BG = jnp.float32(4.99e-6)
RE_QS_BG = jnp.float32(9.99e-6)

R = jnp.float32(287.04)
CP = jnp.float32(1004.0)
LVAP0 = jnp.float32(2.5e6)
R1 = jnp.float32(1.0e-12)
R2 = jnp.float32(1.0e-6)
EPS = jnp.float32(1.0e-15)
NA_IN1 = jnp.float32(0.5e6)
NT_C = jnp.float32(100.0e6)
NT_C_MAX = jnp.float32(1999.0e6)
TNO = jnp.float32(5.0)
ATO = jnp.float32(0.304)
RHO_W = jnp.float32(1000.0)
RHO_I = jnp.float32(890.0)
RHO_NOT = jnp.float32(101325.0) / (jnp.float32(287.05) * jnp.float32(298.0))
C_CUBE = jnp.float32(0.5)
C_SQRD = jnp.float32(0.15)
LSUB = jnp.float32(2.834e6)
ORV = jnp.float32(1.0) / jnp.float32(461.5)
SC3 = jnp.float32(0.632) ** (jnp.float32(1.0) / jnp.float32(3.0))
LFUS = LSUB - LVAP0
OLFUS = jnp.float32(1.0) / (LSUB - LVAP0)
EF_SI = jnp.float32(0.05)
EF_RI = jnp.float32(0.95)
RHO_G = np.asarray([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
IDX_BG1 = 4
AV_G = np.asarray([45.9173813, 67.0867386, 98.0158463, 122.353378, 143.204224, 161.794724, 178.762115, 194.488785, 209.225876], dtype=np.float32)
BV_G = np.asarray([0.640961647] * 9, dtype=np.float32)
AV_G_OLD = jnp.float32(442.0)
BV_G_OLD = jnp.float32(0.89)
A_COEFF = jnp.float32(0.47244157)
B_COEFF = jnp.float32(0.54698726)
D0S = jnp.float32(300.0e-6)
XM0I = jnp.float32(1.0e-12)
HGFR = jnp.float32(235.16)
AV_I = jnp.float32(1493.9)
BV_I = jnp.float32(1.0)
AM_R = jnp.float32(3.1415926536) * RHO_W / jnp.float32(6.0)
BM_R = jnp.float32(3.0)
AM_S = jnp.float32(0.069)
BM_S = jnp.float32(2.0)
AM_G = (np.float32(3.1415926536) * RHO_G / np.float32(6.0)).astype(np.float32)
BM_G = jnp.float32(3.0)
MU_G = jnp.float32(0.0)
AM_I = jnp.float32(3.1415926536) * RHO_I / jnp.float32(6.0)
BM_I = jnp.float32(3.0)
D0I = (XM0I / AM_I) ** (jnp.float32(1.0) / BM_I)
D0C = jnp.float32(1.0e-6)
D0R = jnp.float32(50.0e-6)
D0S = jnp.float32(300.0e-6)
AV_S = jnp.float32(40.0)
BV_S = jnp.float32(0.55)
FV_S = jnp.float32(100.0)
AV_R = jnp.float32(4854.0)
BV_R = jnp.float32(1.0)
FV_R = jnp.float32(195.0)
AV_C = jnp.float32(0.316946e8)
BV_C = jnp.float32(2.0)
AV_G_IDX_BG1_DEFAULT = jnp.float32(442.0)
BV_G_IDX_BG1_DEFAULT = jnp.float32(0.89)
MU_I = jnp.float32(0.0)
MU_R = jnp.float32(0.0)
MU_S = jnp.float32(0.6357)
KAP0 = jnp.float32(490.6)
KAP1 = jnp.float32(17.46)
LAM0 = jnp.float32(20.78)
LAM1 = jnp.float32(3.29)
OBMR = jnp.float32(1.0) / BM_R
OBMG = jnp.float32(1.0) / BM_G
OBMI = jnp.float32(1.0) / BM_I
OAMS = jnp.float32(1.0) / AM_S
OBMS = jnp.float32(1.0) / BM_S
OCMS = OAMS**OBMS
CIG1 = jnp.float32(1.0)
CIG2 = jnp.float32(6.0)
OIG1 = jnp.float32(1.0) / CIG1
OIG2 = jnp.float32(1.0) / CIG2
CSE1 = BM_S + jnp.float32(1.0)
CSE3 = BM_S * jnp.float32(2.0)
G_RATIO = jnp.asarray(
    [24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184, 2730, 3360, 4080, 4896],
    dtype=jnp.float32,
)
SA = jnp.asarray([5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204, 0.003199, 0.0, -0.015952], dtype=jnp.float32)
SB = jnp.asarray([0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.0, -0.003577], dtype=jnp.float32)
R_C = np.asarray(
    [
        1.0e-6,
        2.0e-6,
        3.0e-6,
        4.0e-6,
        5.0e-6,
        6.0e-6,
        7.0e-6,
        8.0e-6,
        9.0e-6,
        1.0e-5,
        2.0e-5,
        3.0e-5,
        4.0e-5,
        5.0e-5,
        6.0e-5,
        7.0e-5,
        8.0e-5,
        9.0e-5,
        1.0e-4,
        2.0e-4,
        3.0e-4,
        4.0e-4,
        5.0e-4,
        6.0e-4,
        7.0e-4,
        8.0e-4,
        9.0e-4,
        1.0e-3,
        2.0e-3,
        3.0e-3,
        4.0e-3,
        5.0e-3,
        6.0e-3,
        7.0e-3,
        8.0e-3,
        9.0e-3,
        1.0e-2,
    ],
    dtype=np.float32,
).astype(np.float64)

_GAMMLN_STP = jnp.float32(2.5066282746310005)
_GAMMLN_COF = jnp.asarray(
    [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ],
    dtype=jnp.float32,
)
_GAMMA_ITMAX = 100
_GAMMA_EPS = jnp.float32(3.0e-7)
_GAMMA_FPMIN = jnp.float32(1.0e-30)
_PI = jnp.float32(3.1415926536)
_R_UNI = jnp.float32(8.314)
_AR_VOLUME = jnp.float32(4.0) / jnp.float32(3.0) * _PI * jnp.float32(2.5e-6) ** jnp.float32(3.0)

_RSLF_C0 = jnp.float32(0.611583699e03)
_RSLF_C1 = jnp.float32(0.444606896e02)
_RSLF_C2 = jnp.float32(0.143177157e01)
_RSLF_C3 = jnp.float32(0.264224321e-1)
_RSLF_C4 = jnp.float32(0.299291081e-3)
_RSLF_C5 = jnp.float32(0.203154182e-5)
_RSLF_C6 = jnp.float32(0.702620698e-8)
_RSLF_C7 = jnp.float32(0.379534310e-11)
_RSLF_C8 = jnp.float32(-0.321582393e-13)

_RSIF_C0 = jnp.float32(0.609868993e03)
_RSIF_C1 = jnp.float32(0.499320233e02)
_RSIF_C2 = jnp.float32(0.184672631e01)
_RSIF_C3 = jnp.float32(0.402737184e-1)
_RSIF_C4 = jnp.float32(0.565392987e-3)
_RSIF_C5 = jnp.float32(0.521693933e-5)
_RSIF_C6 = jnp.float32(0.307839583e-7)
_RSIF_C7 = jnp.float32(0.105785160e-9)
_RSIF_C8 = jnp.float32(0.161444444e-12)

_NU_VALUES = np.arange(1, 16, dtype=np.float32)
CCE = jnp.asarray(
    np.stack(
        [
            _NU_VALUES + np.float32(1.0),
            np.float32(BM_R) + _NU_VALUES + np.float32(1.0),
            np.float32(BM_R) + _NU_VALUES + np.float32(4.0),
            _NU_VALUES + np.float32(2.0) + np.float32(1.0),
            np.float32(BM_R) + _NU_VALUES + np.float32(2.0) + np.float32(1.0),
        ],
        axis=0,
    ),
    dtype=jnp.float32,
)
CCG = jnp.asarray([[np.float32(math.gamma(float(value))) for value in row] for row in np.asarray(CCE)], dtype=jnp.float32)
OCG1 = jnp.float32(1.0) / CCG[0]
OCG2 = jnp.float32(1.0) / CCG[1]

CIE = jnp.asarray(
    [
        MU_I + jnp.float32(1.0),
        BM_I + MU_I + jnp.float32(1.0),
        BM_I + MU_I + jnp.float32(1.0) + jnp.float32(1.0),
        MU_I + jnp.float32(1.0) + jnp.float32(1.0),
        MU_I + jnp.float32(2.0),
        BM_I * jnp.float32(0.5) + MU_I + jnp.float32(1.0) + jnp.float32(1.0),
        BM_I * jnp.float32(0.5) + MU_I + jnp.float32(1.0),
    ],
    dtype=jnp.float32,
)
CIG = jnp.asarray([np.float32(math.gamma(float(value))) for value in np.asarray(CIE)], dtype=jnp.float32)


def _f64(value: Any) -> Any:
    return jnp.asarray(value, dtype=jnp.float64)


def _ice_lami_xdi(number_density: Any, mass_density: Any) -> tuple[Any, Any]:
    """WRF Thompson ice slope: ``lami`` is double precision, ``xDi`` is real."""

    lami = (_f64(AM_I) * _f64(CIG[1]) * _f64(OIG1) * _f64(number_density) / _f64(mass_density)) ** (_f64(1.0) / _f64(BM_I))
    xdi = jnp.asarray((_f64(BM_I) + _f64(MU_I) + _f64(1.0)) / lami, dtype=jnp.float32)
    return lami, xdi


def _ice_lami_for_diameter(diameter: float) -> Any:
    return _f64(CIE[1]) / _f64(diameter)


def _ice_number_from_lami(mass_density: Any, lami: Any, *, cap: float | None = None) -> Any:
    number = _f64(CIG[0]) * _f64(OIG2) * _f64(mass_density) / _f64(AM_I) * _f64(lami) ** _f64(BM_I)
    if cap is not None:
        number = jnp.minimum(_f64(cap), number)
    return jnp.asarray(number, dtype=jnp.float32)


def _wrf_real_mul_add(a: Any, b: Any, c: Any) -> Any:
    """Fortran REAL ``c + a*b`` with a rounded product before the addition."""

    product = lax.optimization_barrier(jnp.asarray(_f64(a) * _f64(b), dtype=jnp.float32))
    return jnp.asarray(jnp.asarray(c, dtype=jnp.float32) + product, dtype=jnp.float32)


def _wrf_real_mul(a: Any, b: Any) -> Any:
    """Fortran REAL multiply with an explicit rounded intermediate."""

    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) * jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_add(a: Any, b: Any) -> Any:
    """Fortran REAL addition with an explicit rounded intermediate."""

    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) + jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_sub(a: Any, b: Any) -> Any:
    """Fortran REAL subtraction with an explicit rounded intermediate."""

    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) - jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


def _wrf_real_div(a: Any, b: Any) -> Any:
    """Fortran REAL division with an explicit rounded intermediate."""

    return lax.optimization_barrier(jnp.asarray(jnp.asarray(a, dtype=jnp.float32) / jnp.asarray(b, dtype=jnp.float32), dtype=jnp.float32))


CRE = jnp.asarray(
    [
        BM_R + jnp.float32(1.0),
        MU_R + jnp.float32(1.0),
        BM_R + MU_R + jnp.float32(1.0),
        BM_R * jnp.float32(2.0) + MU_R + jnp.float32(1.0),
        MU_R + BV_R + jnp.float32(1.0),
        BM_R + MU_R + BV_R + jnp.float32(1.0),
        BM_R * jnp.float32(0.5) + MU_R + BV_R + jnp.float32(1.0),
        BM_R + MU_R + BV_R + jnp.float32(3.0),
        MU_R + BV_R + jnp.float32(3.0),
        MU_R + jnp.float32(2.0),
        jnp.float32(0.5) * (BV_R + jnp.float32(5.0) + jnp.float32(2.0) * MU_R),
        BM_R * jnp.float32(0.5) + MU_R + jnp.float32(1.0),
        BM_R * jnp.float32(2.0) + MU_R + BV_R + jnp.float32(1.0),
    ],
    dtype=jnp.float32,
)
CRG = jnp.asarray([np.float32(math.gamma(float(value))) for value in np.asarray(CRE)], dtype=jnp.float32)
ORG1 = jnp.float32(1.0) / CRG[0]
ORG2 = jnp.float32(1.0) / CRG[1]
ORG3 = jnp.float32(1.0) / CRG[2]

_CGE = np.zeros((12, 9), dtype=np.float32)
_CGE[0, :] = np.float32(BM_G) + np.float32(1.0)
_CGE[1, :] = np.float32(MU_G) + np.float32(1.0)
_CGE[2, :] = np.float32(BM_G) + np.float32(MU_G) + np.float32(1.0)
_CGE[3, :] = np.float32(BM_G) * np.float32(2.0) + np.float32(MU_G) + np.float32(1.0)
_CGE[9, :] = np.float32(MU_G) + np.float32(2.0)
_CGE[11, :] = np.float32(BM_G) * np.float32(0.5) + np.float32(MU_G) + np.float32(1.0)
for _m, _bv_g in enumerate(BV_G):
    _CGE[4, _m] = np.float32(BM_G) * np.float32(2.0) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE[5, _m] = np.float32(BM_G) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE[6, _m] = np.float32(BM_G) * np.float32(0.5) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE[7, _m] = np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE[8, _m] = np.float32(MU_G) + _bv_g + np.float32(3.0)
    _CGE[10, _m] = np.float32(0.5) * (_bv_g + np.float32(5.0) + np.float32(2.0) * np.float32(MU_G))
CGE = jnp.asarray(_CGE, dtype=jnp.float32)
CGG = jnp.asarray([[np.float32(math.gamma(float(value))) for value in row] for row in _CGE], dtype=jnp.float32)
_BV_G_NONHAIL = BV_G.copy()
_BV_G_NONHAIL[IDX_BG1] = np.float32(0.89)
_CGE_NONHAIL = _CGE.copy()
for _m, _bv_g in enumerate(_BV_G_NONHAIL):
    _CGE_NONHAIL[4, _m] = np.float32(BM_G) * np.float32(2.0) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE_NONHAIL[5, _m] = np.float32(BM_G) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE_NONHAIL[6, _m] = np.float32(BM_G) * np.float32(0.5) + np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE_NONHAIL[7, _m] = np.float32(MU_G) + _bv_g + np.float32(1.0)
    _CGE_NONHAIL[8, _m] = np.float32(MU_G) + _bv_g + np.float32(3.0)
    _CGE_NONHAIL[10, _m] = np.float32(0.5) * (_bv_g + np.float32(5.0) + np.float32(2.0) * np.float32(MU_G))
CGG_NONHAIL = jnp.asarray([[np.float32(math.gamma(float(value))) for value in row] for row in _CGE_NONHAIL], dtype=jnp.float32)
OGG1 = jnp.float32(1.0) / CGG[0, 0]
OGG2 = jnp.float32(1.0) / CGG[1, 0]
OGG3 = jnp.float32(1.0) / CGG[2, 0]
OGE1 = jnp.float32(1.0) / CGE[0, 0]

CSE = jnp.asarray(
    [
        BM_S + jnp.float32(1.0),
        BM_S + jnp.float32(2.0),
        BM_S * jnp.float32(2.0),
        BM_S + BV_S + jnp.float32(1.0),
        BM_S * jnp.float32(2.0) + BV_S + jnp.float32(1.0),
        BM_S * jnp.float32(2.0) + jnp.float32(1.0),
        BM_S + MU_S + jnp.float32(1.0),
        BM_S + MU_S + jnp.float32(2.0),
        BM_S + MU_S + jnp.float32(3.0),
        BM_S + MU_S + BV_S + jnp.float32(1.0),
        BM_S * jnp.float32(2.0) + MU_S + BV_S + jnp.float32(1.0),
        BM_S * jnp.float32(2.0) + MU_S + jnp.float32(1.0),
        BV_S + jnp.float32(2.0),
        BM_S + BV_S,
        MU_S + jnp.float32(1.0),
        jnp.float32(1.0) + (jnp.float32(1.0) + BV_S) * jnp.float32(0.5),
        BM_S + BV_S + jnp.float32(2.0),
    ],
    dtype=jnp.float32,
)
CSG = jnp.asarray([np.float32(math.gamma(float(value))) for value in np.asarray(CSE)], dtype=jnp.float32)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CalcEffectRadResult:
    re_qc1d: Any
    re_qi1d: Any
    re_qs1d: Any

    def tree_flatten(self) -> tuple[tuple[Any, Any, Any], None]:
        return (self.re_qc1d, self.re_qi1d, self.re_qs1d), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, Any, Any]) -> "CalcEffectRadResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonPreprocessResult:
    """Column state derived at the start of WRF ``mp_thompson``."""

    qv1d: Any
    qc1d: Any
    qi1d: Any
    qr1d: Any
    qs1d: Any
    qg1d: Any
    qb1d: Any
    ni1d: Any
    nr1d: Any
    nc1d: Any
    ng1d: Any
    temp: Any
    pres: Any
    qv: Any
    rc: Any
    ri: Any
    rr: Any
    rs: Any
    rg: Any
    rb: Any
    ni: Any
    nr: Any
    nc: Any
    ns: Any
    ng: Any
    nwfa: Any
    nifa: Any
    nbca: Any
    rho: Any
    rhof: Any
    rhof2: Any
    qvs: Any
    qvsi: Any
    delqvs: Any
    satw: Any
    sati: Any
    ssatw: Any
    ssati: Any
    diffu: Any
    visco: Any
    vsc2: Any
    tcond: Any
    lvap: Any
    ocp: Any
    twet: Any
    ilamr: Any
    ilamg: Any
    n0_r: Any
    n0_g: Any
    mvd_r: Any
    mvd_c: Any
    mvd_g: Any
    smob: Any
    smo2: Any
    smo1: Any
    smo0: Any
    smoc: Any
    smod: Any
    smoe: Any
    smof: Any
    smog: Any
    l_qc: Any
    l_qi: Any
    l_qr: Any
    l_qs: Any
    l_qg: Any
    idx_bg: Any
    no_micro: Any
    k_melting: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonPreprocessResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonWarmRainResult:
    """Warm-rain source/sink rates from WRF ``mp_thompson``."""

    mvd_c: Any
    prr_wau: Any
    pnr_wau: Any
    pnc_wau: Any
    prr_rcw: Any
    pnc_rcw: Any
    pnr_rcr: Any
    pna_rca: Any
    pnd_rcd: Any
    pnb_rcb: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonWarmRainResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonFrozenLookupIndices:
    """Lookup indexes used by WRF Thompson frozen hydrometeor process terms."""

    xds: Any
    rho_s: Any
    idx_tc: Any
    idx_t: Any
    it: Any
    idx_c: Any
    idx_n: Any
    idx_i: Any
    idx_i1: Any
    idx_r: Any
    idx_r1: Any
    idx_s: Any
    idx_g: Any
    idx_g1: Any
    n0_exp_r: Any
    n0_exp_g: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonFrozenLookupIndices":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonFrozenCollectionResult:
    """Frozen hydrometeor collection and wet-scavenging rates."""

    prs_scw: Any
    pnc_scw: Any
    prg_gcw: Any
    pnc_gcw: Any
    pbg_gcw: Any
    pna_sca: Any
    pnd_scd: Any
    pnb_scb: Any
    pna_gca: Any
    pnd_gcd: Any
    pnb_gcb: Any
    ef_sw: Any
    ef_gw: Any
    rime_dens: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonFrozenCollectionResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonRainFrozenCollectionResult:
    """Rain-snow and rain-graupel lookup-table collection rates."""

    prr_rcs: Any
    prs_rcs: Any
    prg_rcs: Any
    pnr_rcs: Any
    png_rcs: Any
    pbg_rcs: Any
    prr_rcg: Any
    prg_rcg: Any
    pnr_rcg: Any
    png_rcg: Any
    pbg_rcg: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonRainFrozenCollectionResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonFreezingResult:
    """Bigg rain/cloud-water freezing rates from WRF Thompson."""

    idx_in: Any
    prg_rfz: Any
    pri_rfz: Any
    pni_rfz: Any
    pnr_rfz: Any
    pbg_rfz: Any
    pri_wfz: Any
    pni_wfz: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonFreezingResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonIceNucleationResult:
    """Primary ice nucleation rates from WRF Thompson."""

    pri_inu: Any
    pni_inu: Any
    pri_iha: Any
    pni_iha: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonIceNucleationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonIceDepositionResult:
    """Cloud-ice deposition/sublimation and ice-to-snow conversion rates."""

    pri_ide: Any
    pni_ide: Any
    prs_ide: Any
    prs_iau: Any
    pni_iau: Any
    xdi: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonIceDepositionResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonSnowGraupelDepositionResult:
    """Snow and graupel deposition/sublimation rates from WRF Thompson."""

    prs_sde: Any
    prg_gde: Any
    png_gde: Any
    c_snow: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonSnowGraupelDepositionResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonIceCollectionResult:
    """Snow/rain collection of cloud ice from WRF Thompson."""

    prs_sci: Any
    pni_sci: Any
    pri_rci: Any
    pni_rci: Any
    pnr_rci: Any
    prr_rci: Any
    prg_rci: Any
    pbg_rci: Any
    xdi: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonIceCollectionResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonRimingResult:
    """Hallet-Mossop ice multiplication and rimed-snow conversion rates."""

    pri_ihm: Any
    pni_ihm: Any
    prs_ihm: Any
    prg_ihm: Any
    prs_scw: Any
    prg_scw: Any
    png_scw: Any
    pbg_scw: Any
    vts_boost: Any
    g_frac: Any
    rime_dens: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonRimingResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonMeltingResult:
    """Warm-layer snow/graupel melting and cloud-water collection adjustment."""

    prr_sml: Any
    pnr_sml: Any
    prs_sde: Any
    prr_gml: Any
    pnr_gml: Any
    pbg_gml: Any
    prg_gde: Any
    png_gde: Any
    prr_rcw: Any
    prs_scw: Any
    prg_gcw: Any
    n0_melt: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonMeltingResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonDepletionRescaleResult:
    """WRF Thompson hydrometeor/vapor depletion-limited process rates."""

    pri_inu: Any
    pri_ide: Any
    pni_ide: Any
    prs_ide: Any
    prs_sde: Any
    prg_gde: Any
    pri_iha: Any
    prr_wau: Any
    pri_wfz: Any
    prr_rcw: Any
    prs_scw: Any
    prg_scw: Any
    prg_gcw: Any
    prs_iau: Any
    prs_sci: Any
    pri_rci: Any
    prg_rfz: Any
    pri_rfz: Any
    prr_rci: Any
    prr_rcs: Any
    prr_rcg: Any
    prs_ihm: Any
    prr_sml: Any
    prs_rcs: Any
    prg_ihm: Any
    prr_gml: Any
    prg_rcg: Any
    pri_ihm: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonDepletionRescaleResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonTendencyResult:
    """WRF Thompson species tendencies after process-rate accumulation."""

    nwfaten: Any
    nifaten: Any
    nbcaten: Any
    qvten: Any
    qcten: Any
    ncten: Any
    qiten: Any
    niten: Any
    qrten: Any
    nrten: Any
    qsten: Any
    qgten: Any
    ngten: Any
    qbten: Any
    tten: Any
    mvd_r: Any
    mvd_g: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonTendencyResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonPostTendencyState:
    """WRF Thompson column state refreshed from accumulated tendencies."""

    temp: Any
    qv: Any
    rho: Any
    rhof: Any
    rhof2: Any
    qvs: Any
    ssatw: Any
    diffu: Any
    visco: Any
    vsc2: Any
    lvap: Any
    tcond: Any
    ocp: Any
    lvt2: Any
    nwfa: Any
    rc: Any
    nc: Any
    l_qc: Any
    ri: Any
    ni: Any
    l_qi: Any
    rr: Any
    nr: Any
    l_qr: Any
    mvd_r: Any
    rs: Any
    l_qs: Any
    rg: Any
    ng: Any
    rb: Any
    l_qg: Any
    idx_bg: Any
    smob: Any
    smo2: Any
    smoc: Any
    smod: Any
    ilamg: Any
    n0_g: Any
    ilamr: Any
    n0_r: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonPostTendencyState":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonCloudWaterCondensationResult:
    """WRF Thompson cloud-water condensation/evaporation update."""

    qvten: Any
    qcten: Any
    ncten: Any
    nwfaten: Any
    tten: Any
    rc: Any
    nc: Any
    l_qc: Any
    qv: Any
    temp: Any
    rho: Any
    qvs: Any
    ssatw: Any
    prw_vcd: Any
    pnc_wcd: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonCloudWaterCondensationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonRainEvaporationResult:
    """WRF Thompson rain evaporation update."""

    qvten: Any
    qrten: Any
    nrten: Any
    nwfaten: Any
    tten: Any
    rr: Any
    qv: Any
    nr: Any
    temp: Any
    rho: Any
    rhof: Any
    rhof2: Any
    diffu: Any
    visco: Any
    vsc2: Any
    lvap: Any
    tcond: Any
    ocp: Any
    prv_rev: Any
    pnr_rev: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonRainEvaporationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonRainSedimentationSetupResult:
    """Rain fall speeds and substep controls before Thompson sedimentation."""

    vtrk: Any
    vtnrk: Any
    rhof: Any
    ksed1: Any
    nstep: Any
    onstep: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonRainSedimentationSetupResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonRainSedimentationResult:
    """Rain sedimentation tendency and state update from WRF Thompson."""

    qrten: Any
    nrten: Any
    rr: Any
    nr: Any
    pptrain: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonRainSedimentationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonCloudSedimentationSetupResult:
    """Cloud-water fall speeds before Thompson sedimentation."""

    vtck: Any
    vtnck: Any
    ksed1: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonCloudSedimentationSetupResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonCloudSedimentationResult:
    """Cloud-water sedimentation tendency and state update."""

    qcten: Any
    ncten: Any
    rc: Any
    nc: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonCloudSedimentationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonIceSedimentationSetupResult:
    """Cloud-ice fall speeds and substep controls before sedimentation."""

    vtik: Any
    vtnik: Any
    ksed1: Any
    nstep: Any
    onstep: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonIceSedimentationSetupResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonIceSedimentationResult:
    """Cloud-ice sedimentation tendency and state update."""

    qiten: Any
    niten: Any
    ri: Any
    ni: Any
    pptice: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonIceSedimentationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonSnowSedimentationSetupResult:
    """Snow fall speeds and substep controls before sedimentation."""

    vtsk: Any
    ksed1: Any
    nstep: Any
    onstep: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonSnowSedimentationSetupResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonSnowSedimentationResult:
    """Snow sedimentation tendency and state update."""

    qsten: Any
    rs: Any
    pptsnow: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonSnowSedimentationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonGraupelSedimentationSetupResult:
    """Graupel/hail fall speeds and substep controls before sedimentation."""

    vtgk: Any
    vtngk: Any
    ksed1: Any
    nstep: Any
    onstep: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonGraupelSedimentationSetupResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonGraupelSedimentationResult:
    """Graupel/hail sedimentation tendency and state update."""

    qgten: Any
    ngten: Any
    qbten: Any
    rg: Any
    ng: Any
    rb: Any
    pptgraul: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonGraupelSedimentationResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonInstantPhaseChangeResult:
    """Instant cloud ice melt/cloud water freeze tendency update."""

    qcten: Any
    ncten: Any
    qiten: Any
    niten: Any
    tten: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonInstantPhaseChangeResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonFinalPassbackResult:
    """Final WRF Thompson parent-column values after all tendencies are applied."""

    t1d: Any
    qv1d: Any
    qc1d: Any
    nc1d: Any
    nwfa1d: Any
    nifa1d: Any
    nbca1d: Any
    qi1d: Any
    ni1d: Any
    qr1d: Any
    nr1d: Any
    mvd_r: Any
    qs1d: Any
    qg1d: Any
    ng1d: Any
    qb1d: Any
    idx_bg: Any
    mvd_g: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonFinalPassbackResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonColumnDriverResult:
    """Pure JAX Thompson single-column driver result."""

    final: ThompsonFinalPassbackResult
    tendencies: ThompsonTendencyResult
    pptrain: Any
    pptsnow: Any
    pptice: Any
    pptgraul: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonColumnDriverResult":
        del aux_data
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThompsonGridDriverDefaults:
    """WRF ``mp_gt_driver`` default one-column optional Thompson fields."""

    nc1d: Any
    nwfa1d: Any
    nifa1d: Any
    nbca1d: Any
    ng1d: Any
    qb1d: Any

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[Any, ...]) -> "ThompsonGridDriverDefaults":
        del aux_data
        return cls(*children)


@tree_util.register_dataclass
@dataclass(frozen=True)
class ThompsonQrAcrQgTables:
    """Lookup tables from WRF ``qr_acr_qg``."""

    tcg_racg: np.ndarray
    tmr_racg: np.ndarray
    tcr_gacr: np.ndarray
    tnr_racg: np.ndarray
    tnr_gacr: np.ndarray


@tree_util.register_dataclass
@dataclass(frozen=True)
class ThompsonQrAcrQsTables:
    """Lookup tables from WRF ``qr_acr_qs``."""

    tcs_racs1: np.ndarray
    tmr_racs1: np.ndarray
    tcs_racs2: np.ndarray
    tmr_racs2: np.ndarray
    tcr_sacr1: np.ndarray
    tms_sacr1: np.ndarray
    tcr_sacr2: np.ndarray
    tms_sacr2: np.ndarray
    tnr_racs1: np.ndarray
    tnr_racs2: np.ndarray
    tnr_sacr1: np.ndarray
    tnr_sacr2: np.ndarray


@tree_util.register_dataclass
@dataclass(frozen=True)
class ThompsonFreezeH2OTables:
    """Lookup tables from WRF ``freezeH2O``."""

    tpi_qrfz: np.ndarray
    tpg_qrfz: np.ndarray
    tni_qrfz: np.ndarray
    tnr_qrfz: np.ndarray
    tpi_qcfz: np.ndarray
    tni_qcfz: np.ndarray


@tree_util.register_dataclass
@dataclass(frozen=True)
class ThompsonQiAutQsTables:
    """Lookup tables from WRF ``qi_aut_qs``."""

    tps_iaus: np.ndarray
    tni_iaus: np.ndarray
    tpi_ide: np.ndarray


@tree_util.register_dataclass
@dataclass(frozen=True)
class ThompsonTables:
    """Explicit Thompson module lookup state needed by the Python driver."""

    qr_acr_qg: ThompsonQrAcrQgTables
    qr_acr_qs: ThompsonQrAcrQsTables
    freeze_h2o: ThompsonFreezeH2OTables
    qi_aut_qs: ThompsonQiAutQsTables
    efrw: np.ndarray
    efsw: np.ndarray
    tpc_wev: np.ndarray
    tnc_wev: np.ndarray
    tnccn_act: np.ndarray
    radar: RadarState


def _read_fortran_unformatted_records(path: str | Path, shapes: tuple[tuple[int, ...], ...], *, dtype: str = "f8") -> tuple[np.ndarray, ...]:
    raw = Path(path).read_bytes()
    offset = 0
    out: list[np.ndarray] = []
    dtype_be = np.dtype(f">{dtype}")
    for shape in shapes:
        if offset + 8 > len(raw):
            raise ValueError(f"Unexpected end of Fortran records in {path}")
        leading = int(np.frombuffer(raw, dtype=">i4", count=1, offset=offset)[0])
        offset += 4
        expected = int(np.prod(shape)) * dtype_be.itemsize
        if leading != expected:
            raise ValueError(f"Unexpected Fortran record size {leading}, expected {expected}, in {path}")
        payload = np.frombuffer(raw, dtype=dtype_be, count=int(np.prod(shape)), offset=offset)
        out.append(np.array(payload.reshape(shape, order="F"), dtype=np.float64 if dtype == "f8" else np.float32, copy=True))
        offset += expected
        trailing = int(np.frombuffer(raw, dtype=">i4", count=1, offset=offset)[0])
        offset += 4
        if trailing != expected:
            raise ValueError(f"Mismatched Fortran trailing record size {trailing}, expected {expected}, in {path}")
    if offset != len(raw):
        raise ValueError(f"Unexpected extra bytes in Fortran record file {path}: {len(raw) - offset}")
    return tuple(out)


def load_qr_acr_qg_table(path: str | Path, *, nrhg_table: int = 1) -> ThompsonQrAcrQgTables:
    """Read WRF Thompson ``qr_acr_qg`` lookup table file."""

    shape = (NTB_G1, NTB_G, int(nrhg_table), NTB_R1, NTB_R)
    records = _read_fortran_unformatted_records(path, (shape, shape, shape, shape, shape))
    return ThompsonQrAcrQgTables(*records)


def load_qr_acr_qs_table(path: str | Path) -> ThompsonQrAcrQsTables:
    """Read WRF Thompson ``qr_acr_qs`` lookup table file."""

    shape = (NTB_S, NTB_T, NTB_R1, NTB_R)
    records = _read_fortran_unformatted_records(path, (shape,) * 12)
    return ThompsonQrAcrQsTables(*records)


def load_freeze_h2o_table(path: str | Path) -> ThompsonFreezeH2OTables:
    """Read WRF Thompson ``freezeH2O`` lookup table file."""

    rain_shape = (NTB_R, NTB_R1, 45, NTB_IN)
    cloud_shape = (NTB_C, NBC, 45, NTB_IN)
    records = _read_fortran_unformatted_records(path, (rain_shape, rain_shape, rain_shape, rain_shape, cloud_shape, cloud_shape))
    tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz = records
    return ThompsonFreezeH2OTables(
        tpi_qrfz=tpi_qrfz,
        tpg_qrfz=tpg_qrfz,
        tni_qrfz=tni_qrfz,
        tnr_qrfz=tnr_qrfz,
        tpi_qcfz=tpi_qcfz,
        tni_qcfz=tni_qcfz,
    )


@lru_cache(maxsize=4)
def load_thompson_tables(case_dir: str) -> ThompsonTables:
    """Load/generate the full Thompson lookup state for a WRF case directory."""

    base = Path(case_dir)
    tpc_wev, tnc_wev = table_drop_evap()
    return ThompsonTables(
        qr_acr_qg=load_qr_acr_qg_table(base / "qr_acr_qg_V4.dat"),
        qr_acr_qs=load_qr_acr_qs_table(base / "qr_acr_qsV2.dat"),
        freeze_h2o=load_freeze_h2o_table(base / "freezeH2O.dat"),
        qi_aut_qs=table_qi_aut_qs(),
        efrw=table_efrw(),
        efsw=table_efsw(),
        tpc_wev=tpc_wev,
        tnc_wev=tnc_wev,
        tnccn_act=load_ccn_activate_table(base / "CCN_ACTIVATE.BIN"),
        radar=radar_init(),
    )


def _one_to_nine_decade_values(first_power: int, last_power: int, *, include_final_power: bool = True) -> np.ndarray:
    values: list[float] = []
    for power in range(first_power, last_power):
        scale = 10.0**power
        values.extend(float(multiplier * scale) for multiplier in range(1, 10))
    if include_final_power:
        values.append(float(10.0**last_power))
    return np.asarray(values, dtype=np.float32)


def thompson_ice_table_mixing_ratios() -> np.ndarray:
    """Cloud ice content lookup values ``r_i``."""

    return _one_to_nine_decade_values(-10, -3)


def thompson_ice_number_concentrations() -> np.ndarray:
    """Ice number concentration lookup values ``Nt_i``."""

    return _one_to_nine_decade_values(0, 6)


def thompson_ice_nuclei_concentrations() -> np.ndarray:
    """Ice nuclei concentration lookup values ``Nt_IN``."""

    return _one_to_nine_decade_values(0, 6)


def thompson_ice_diameters() -> np.ndarray:
    """Cloud ice diameter bins ``Di`` from Thompson initialization."""

    return _log_diameter_bins(float(np.float32(D0I)), 5.0 * float(np.float32(D0S)), NBINS)


def thompson_ice_diameter_widths() -> np.ndarray:
    """Cloud ice diameter bin widths ``dti`` from Thompson initialization."""

    edges = np.empty(NBINS + 1, dtype=np.float64)
    edges[0] = np.float64(np.float32(D0I))
    edges[-1] = np.float64(5.0 * float(np.float32(D0S)))
    for idx in range(1, NBINS):
        edges[idx] = np.exp((idx / NBINS) * np.log(edges[-1] / edges[0]) + np.log(edges[0]))
    return edges[1:] - edges[:-1]


def table_qi_aut_qs() -> ThompsonQiAutQsTables:
    """Generate Thompson ice-to-snow autoconversion lookup tables."""

    r_i = thompson_ice_table_mixing_ratios().astype(np.float64)[:, None]
    nt_i = thompson_ice_number_concentrations().astype(np.float64)[None, :]
    di = thompson_ice_diameters()
    dti = thompson_ice_diameter_widths()
    am_i = np.float64(np.float32(np.float32(3.1415926536) * np.float32(890.0) / np.float32(6.0)))
    bm_i = np.float64(np.float32(3.0))
    mu_i = np.float64(np.float32(0.0))
    cig2 = np.float64(np.float32(6.0))
    oig1 = np.float64(np.float32(1.0))
    obmi = np.float64(np.float32(1.0 / 3.0))
    d0s = np.float64(np.float32(D0S))

    lami = (am_i * cig2 * oig1 * nt_i / r_i) ** obmi
    di_mean = (bm_i + mu_i + np.float64(1.0)) / lami
    n0_i = nt_i * oig1 * lami ** np.float64(mu_i + 1.0)
    high = di_mean.astype(np.float32) > np.float32(5.0) * np.float32(D0S)
    low = di_mean.astype(np.float32) < np.float32(D0I)
    middle = ~(high | low)

    xlimit_intg = (lami * d0s).astype(np.float32)
    tpi_mid = np.asarray(gammp(np.float32(mu_i + 2.0), jnp.asarray(xlimit_intg, dtype=jnp.float32)), dtype=np.float64)
    tpi_ide = np.where(high, np.float64(0.0), np.where(low, np.float64(1.0), tpi_mid)).astype(np.float64)

    active = di >= d0s
    n_i = n0_i[:, :, None] * di[None, None, :] ** mu_i * np.exp(-lami[:, :, None] * di[None, None, :]) * dti[None, None, :]
    tps_mid = np.sum(n_i[:, :, active] * am_i * di[active][None, None, :] ** bm_i, axis=2, dtype=np.float64)
    tni_mid = np.sum(n_i[:, :, active], axis=2, dtype=np.float64)
    tps_iaus = np.where(high, r_i, np.where(low, np.float64(0.0), tps_mid)).astype(np.float64)
    tni_iaus = np.where(high, nt_i, np.where(low, np.float64(0.0), tni_mid)).astype(np.float64)
    return ThompsonQiAutQsTables(tps_iaus=tps_iaus, tni_iaus=tni_iaus, tpi_ide=tpi_ide)


def thompson_cloud_diameters() -> np.ndarray:
    """Cloud diameter bins ``Dc`` from Thompson initialization."""

    d0c = np.float64(np.float32(1.0e-6))
    return (np.arange(1, NBC + 1, dtype=np.float64) * d0c).astype(np.float64)


def _log_diameter_bins(start: float, stop: float, count: int) -> np.ndarray:
    edges = np.empty(count + 1, dtype=np.float64)
    edges[0] = float(start)
    edges[-1] = float(stop)
    for idx in range(1, count):
        edges[idx] = np.exp((idx / count) * np.log(edges[-1] / edges[0]) + np.log(edges[0]))
    return np.sqrt(edges[:-1] * edges[1:])


def thompson_rain_diameters() -> np.ndarray:
    """Rain diameter bins ``Dr`` from Thompson initialization."""

    return _log_diameter_bins(float(np.float32(50.0e-6)), 0.005, NBR)


def thompson_rain_table_mixing_ratios() -> np.ndarray:
    """Rain content lookup values ``r_r``."""

    return _one_to_nine_decade_values(-6, -2)


def thompson_snow_table_mixing_ratios() -> np.ndarray:
    """Snow content lookup values ``r_s``."""

    return _one_to_nine_decade_values(-6, -2)


def thompson_graupel_table_mixing_ratios() -> np.ndarray:
    """Graupel content lookup values ``r_g``."""

    return _one_to_nine_decade_values(-6, -2)


def thompson_rain_intercepts() -> np.ndarray:
    """Rain y-intercept lookup values ``N0r_exp``."""

    return _one_to_nine_decade_values(6, 10)


def thompson_graupel_intercepts() -> np.ndarray:
    """Graupel y-intercept lookup values ``N0g_exp``."""

    return _one_to_nine_decade_values(2, 6)


def thompson_snow_diameters() -> np.ndarray:
    """Snow diameter bins ``Ds`` from Thompson initialization."""

    return _log_diameter_bins(float(np.float32(300.0e-6)), 0.02, NBS)


def thompson_cloud_diameter_widths() -> np.ndarray:
    """Cloud diameter bin widths ``dtc`` from Thompson initialization."""

    return np.ones(NBC, dtype=np.float64) * np.float64(np.float32(1.0e-6))


def thompson_cloud_number_bins() -> np.ndarray:
    """Cloud droplet number concentration bins ``t_Nc``."""

    edges = np.empty(NBC + 1, dtype=np.float64)
    edges[0] = 1.0
    edges[-1] = 3000.0
    for idx in range(1, NBC):
        edges[idx] = np.exp((idx / NBC) * np.log(edges[-1] / edges[0]) + np.log(edges[0]))
    return np.sqrt(edges[:-1] * edges[1:]) * 1.0e6


def table_efrw() -> np.ndarray:
    """Generate Thompson rain-cloud-water collision efficiency table."""

    dc = thompson_cloud_diameters()[None, :]
    dr = thompson_rain_diameters()[:, None]
    p = dc / dr
    x = dc * 1.0e6
    ef_poly = np.where(
        dr < 75.0e-6,
        0.026794 * x - 0.20604,
        np.where(
            dr < 125.0e-6,
            -0.00066842 * x * x + 0.061542 * x - 0.37089,
            np.where(
                dr < 175.0e-6,
                4.091e-06 * x**4 - 0.00030908 * x**3 + 0.0066237 * x * x - 0.0013687 * x - 0.073022,
                np.where(
                    dr < 250.0e-6,
                    9.6719e-5 * x**3 - 0.0068901 * x * x + 0.17305 * x - 0.65988,
                    np.where(
                        dr < 350.0e-6,
                        9.0488e-5 * x**3 - 0.006585 * x * x + 0.16606 * x - 0.56125,
                        0.00010721 * x**3 - 0.0072962 * x * x + 0.1704 * x - 0.46929,
                    ),
                ),
            ),
        ),
    )
    vtr = -0.1021 + 4.932e3 * dr - 0.9551e6 * dr**2 + 0.07934e9 * dr**3 - 0.002362e12 * dr**4
    stokes = dc * dc * vtr * 1000.0 / (9.0 * 1.718e-5 * dr)
    reynolds = 9.0 * stokes / (p * p * 1000.0)
    f_val = np.log(np.maximum(reynolds, 1.0e-300))
    g_val = -0.1007 - 0.358 * f_val + 0.0261 * f_val * f_val
    k0 = np.exp(g_val)
    z_val = np.log(np.maximum(stokes / (k0 + 1.0e-15), 1.0e-300))
    h_val = 0.1465 + 1.302 * z_val - 0.607 * z_val * z_val + 0.293 * z_val * z_val * z_val
    yc0 = 2.0 / np.pi * np.arctan(h_val)
    ef_bg = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p))
    ef = np.where((dr < 50.0e-6) | (dc < 3.0e-6), 0.0, np.where(p > 0.25, ef_poly, ef_bg))
    return np.clip(ef.astype(np.float32), np.float32(0.0), np.float32(0.95)).astype(np.float32)


def table_efsw() -> np.ndarray:
    """Generate Thompson snow-cloud-water collision efficiency table."""

    dc = thompson_cloud_diameters()[None, :]
    ds = thompson_snow_diameters()[:, None]
    vtc = 1.19e4 * (1.0e4 * dc * dc * 0.25)
    vts = 40.0 * ds**0.55 * np.exp(-100.0 * ds) - vtc
    am_r = np.float64(np.float32(np.float32(3.1415926536) * np.float32(1000.0) / np.float32(6.0)))
    ds_m = (np.float64(np.float32(0.069)) * ds**2.0 / am_r) ** (1.0 / 3.0)
    p = dc / ds_m
    stokes = dc * dc * vts * 1000.0 / (9.0 * 1.718e-5 * ds_m)
    reynolds = 9.0 * stokes / (p * p * 1000.0)
    f_val = np.log(np.maximum(reynolds, 1.0e-300))
    g_val = -0.1007 - 0.358 * f_val + 0.0261 * f_val * f_val
    k0 = np.exp(g_val)
    z_val = np.log(np.maximum(stokes / (k0 + 1.0e-15), 1.0e-300))
    h_val = 0.1465 + 1.302 * z_val - 0.607 * z_val * z_val + 0.293 * z_val * z_val * z_val
    yc0 = 2.0 / np.pi * np.arctan(h_val)
    ef_sw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p))
    ef = np.where((p > 0.25) | (ds < 300.0e-6) | (dc < 6.0e-6) | (vts < 1.0e-3), 0.0, ef_sw)
    return np.clip(ef.astype(np.float32), np.float32(0.0), np.float32(0.95)).astype(np.float32)


def table_drop_evap() -> tuple[np.ndarray, np.ndarray]:
    """Generate Thompson cloud-drop evaporation mass and number tables."""

    dc = thompson_cloud_diameters()
    dtc = thompson_cloud_diameter_widths()
    t_nc = thompson_cloud_number_bins()
    am_r = np.float64(np.float32(np.float32(3.1415926536) * np.float32(1000.0) / np.float32(6.0)))
    obmr = np.float64(np.float32(1.0 / 3.0))
    massc = am_r * dc**3.0
    ccg2 = np.asarray([np.float32(math.gamma(n + 4.0)) for n in range(1, 16)], dtype=np.float32).astype(np.float64)
    ocg1 = np.asarray([np.float32(1.0 / math.gamma(n + 1.0)) for n in range(1, 16)], dtype=np.float32).astype(np.float64)
    cce1 = np.asarray([np.float32(n + 1.0) for n in range(1, 16)], dtype=np.float32).astype(np.float64)

    tpc_wev = np.empty((NBC, NTB_C, NBC), dtype=np.float64)
    tnc_wev = np.empty((NBC, NTB_C, NBC), dtype=np.float64)
    for k in range(NBC):
        nu_c = min(15, int(np.floor(np.float32(1000.0e6) / np.float32(t_nc[k]) + np.float32(0.5))) + 2)
        nu_idx = nu_c - 1
        for j in range(NTB_C):
            lamc = (t_nc[k] * am_r * ccg2[nu_idx] * ocg1[nu_idx] / R_C[j]) ** obmr
            n0_c = t_nc[k] * ocg1[nu_idx] * lamc ** cce1[nu_idx]
            n_c = n0_c * dc**nu_c * np.exp(-lamc * dc) * dtc
            tpc_wev[:, j, k] = np.cumsum(massc * n_c)
            tnc_wev[:, j, k] = np.cumsum(n_c)
    return tpc_wev, tnc_wev


def load_ccn_activate_table(path: str | Path) -> np.ndarray:
    """Read WRF Thompson ``CCN_ACTIVATE.BIN`` as a Fortran-order table."""

    raw = Path(path).read_bytes()
    expected_payload = int(np.prod(CCN_ACTIVATE_SHAPE)) * np.dtype("<f4").itemsize
    if len(raw) != expected_payload + 8:
        raise ValueError(f"Unexpected CCN_ACTIVATE.BIN size {len(raw)} bytes")
    endian = None
    markers = {}
    for candidate in ("<", ">"):
        leading = int(np.frombuffer(raw[:4], dtype=f"{candidate}i4", count=1)[0])
        trailing = int(np.frombuffer(raw[-4:], dtype=f"{candidate}i4", count=1)[0])
        markers[candidate] = (leading, trailing)
        if leading == expected_payload and trailing == expected_payload:
            endian = candidate
            break
    if endian is None:
        raise ValueError(f"Unexpected Fortran record markers {markers}")
    payload = np.frombuffer(raw[4:-4], dtype=f"{endian}f4")
    return np.array(payload.reshape(CCN_ACTIVATE_SHAPE, order="F"), dtype=np.float32, copy=True)


def _fortran_nint(x: Any) -> Any:
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    return jnp.where(x_arr >= jnp.float32(0.0), jnp.floor(x_arr + jnp.float32(0.5)), jnp.ceil(x_arr - jnp.float32(0.5))).astype(jnp.int32)


def activ_ncloud(tt: Any, ww: Any, nccn: Any, tnccn_act: Any) -> Any:
    """Activated cloud droplet concentration from Thompson ``activ_ncloud``."""

    tt_arr = jnp.asarray(tt, dtype=jnp.float32)
    ww_arr = jnp.asarray(ww, dtype=jnp.float32)
    nccn_arr = jnp.asarray(nccn, dtype=jnp.float32)
    table = jnp.asarray(tnccn_act, dtype=jnp.float32)

    n_local = nccn_arr * jnp.float32(1.0e-6)
    n_local = jnp.where(n_local >= TA_NA[-1], TA_NA[-1] - jnp.float32(1.0), n_local)
    n_local = jnp.where(n_local <= TA_NA[0], TA_NA[0] + jnp.float32(1.0), n_local)
    i_hi = jnp.searchsorted(TA_NA, n_local, side="right").astype(jnp.int32)
    i_hi = jnp.clip(i_hi, 1, NTB_ARC - 1)
    i_lo = i_hi - 1
    x1 = jnp.log(TA_NA[i_lo])
    x2 = jnp.log(TA_NA[i_hi])

    w_local = ww_arr
    w_local = jnp.where(w_local >= TA_WW[-1], TA_WW[-1] - jnp.float32(1.0), w_local)
    w_local = jnp.where(w_local <= TA_WW[0], TA_WW[0] + jnp.float32(0.001), w_local)
    j_hi = jnp.searchsorted(TA_WW, w_local, side="right").astype(jnp.int32)
    j_hi = jnp.clip(j_hi, 1, NTB_ARW - 1)
    j_lo = j_hi - 1
    y1 = jnp.log(TA_WW[j_lo])
    y2 = jnp.log(TA_WW[j_hi])

    k_idx = _fortran_nint((tt_arr - TA_TK[0]) * jnp.float32(0.1))
    k_idx = jnp.clip(k_idx, 0, NTB_ART - 1)
    l_idx = jnp.asarray(2, dtype=jnp.int32)
    m_idx = jnp.asarray(1, dtype=jnp.int32)

    a_val = table[i_lo, j_lo, k_idx, l_idx, m_idx]
    b_val = table[i_hi, j_lo, k_idx, l_idx, m_idx]
    c_val = table[i_hi, j_hi, k_idx, l_idx, m_idx]
    d_val = table[i_lo, j_hi, k_idx, l_idx, m_idx]
    nx = jnp.log(n_local)
    wy = jnp.log(w_local)
    t = (nx - x1) / (x2 - x1)
    u = (wy - y1) / (y2 - y1)
    fraction = (
        (jnp.float32(1.0) - t) * (jnp.float32(1.0) - u) * a_val
        + t * (jnp.float32(1.0) - u) * b_val
        + t * u * c_val
        + (jnp.float32(1.0) - t) * u * d_val
    )
    return nccn_arr * fraction


def _snow_moment_coeff(tc0: Any, moment: Any) -> tuple[Any, Any]:
    tc = jnp.asarray(tc0, dtype=jnp.float32)
    mm = jnp.asarray(moment, dtype=jnp.float32)
    tc2 = tc * tc
    tc3 = tc2 * tc
    mm2 = mm * mm
    mm3 = mm2 * mm
    loga = (
        SA[0]
        + SA[1] * tc
        + SA[2] * mm
        + SA[3] * tc * mm
        + SA[4] * tc2
        + SA[5] * mm2
        + SA[6] * tc2 * mm
        + SA[7] * tc * mm2
        + SA[8] * tc3
        + SA[9] * mm3
    )
    b_coef = (
        SB[0]
        + SB[1] * tc
        + SB[2] * mm
        + SB[3] * tc * mm
        + SB[4] * tc2
        + SB[5] * mm2
        + SB[6] * tc2 * mm
        + SB[7] * tc * mm2
        + SB[8] * tc3
        + SB[9] * mm3
    )
    return jnp.float32(10.0) ** loga, b_coef


def thompson_preprocess_column(
    qv1d: Any,
    qc1d: Any,
    qi1d: Any,
    qr1d: Any,
    qs1d: Any,
    qg1d: Any,
    qb1d: Any,
    ni1d: Any,
    nr1d: Any,
    nc1d: Any,
    ng1d: Any,
    nwfa1d: Any,
    nifa1d: Any,
    nbca1d: Any,
    t1d: Any,
    p1d: Any,
    *,
    qvs1d: Any | None = None,
    qvsi1d: Any | None = None,
    delqvs1d: Any | None = None,
    aer_init_opt: int = 1,
    wif_input_opt: int = 1,
    is_aerosol_aware: bool = False,
    is_hail_aware: bool = False,
) -> ThompsonPreprocessResult:
    """Translate the initial column preprocessing block of WRF ``mp_thompson``."""

    qv_input = jnp.asarray(qv1d, dtype=jnp.float32)
    qc_input = jnp.asarray(qc1d, dtype=jnp.float32)
    qi_input = jnp.asarray(qi1d, dtype=jnp.float32)
    qr_input = jnp.asarray(qr1d, dtype=jnp.float32)
    qs_input = jnp.asarray(qs1d, dtype=jnp.float32)
    qg_input = jnp.asarray(qg1d, dtype=jnp.float32)
    qb_input = jnp.asarray(qb1d, dtype=jnp.float32)
    ni_input = jnp.asarray(ni1d, dtype=jnp.float32)
    nr_input = jnp.asarray(nr1d, dtype=jnp.float32)
    nc_input = jnp.asarray(nc1d, dtype=jnp.float32)
    ng_input = jnp.asarray(ng1d, dtype=jnp.float32)
    nwfa_input = jnp.asarray(nwfa1d, dtype=jnp.float32)
    nifa_input = jnp.asarray(nifa1d, dtype=jnp.float32)
    nbca_input = jnp.asarray(nbca1d, dtype=jnp.float32)
    temp = jnp.asarray(t1d, dtype=jnp.float32)
    pres = jnp.asarray(p1d, dtype=jnp.float32)

    rho_g = jnp.asarray(RHO_G, dtype=jnp.float32)
    am_g = jnp.asarray(AM_G, dtype=jnp.float32)
    qv = jnp.maximum(jnp.float32(1.0e-10), qv_input)
    rho = jnp.float32(0.622) * pres / (R * temp * (qv + jnp.float32(0.622)))

    if is_aerosol_aware:
        if int(aer_init_opt) < 2:
            nwfa = jnp.maximum(jnp.float32(11.1e6), jnp.minimum(jnp.float32(9999.0e6), nwfa_input * rho))
            nifa = jnp.maximum(NA_IN1 * jnp.float32(0.01), jnp.minimum(jnp.float32(9999.0e6), nifa_input * rho))
            if int(wif_input_opt) == 2:
                nbca = jnp.maximum(jnp.float32(5.55e6), jnp.minimum(jnp.float32(9999.0e6), nbca_input * rho))
            else:
                nbca = jnp.zeros_like(rho)
        else:
            nwfa = jnp.maximum(jnp.float32(0.0), nwfa_input * rho)
            nifa = jnp.maximum(jnp.float32(0.0), nifa_input * rho)
            if int(wif_input_opt) == 2:
                nbca = jnp.maximum(jnp.float32(0.0), nbca_input * rho)
            else:
                nbca = jnp.zeros_like(rho)
    else:
        nwfa = jnp.maximum(jnp.float32(11.1e6), jnp.minimum(jnp.float32(9999.0e6), nwfa_input * rho))
        nifa = jnp.maximum(NA_IN1 * jnp.float32(0.01), jnp.minimum(jnp.float32(9999.0e6), nifa_input * rho))
        nbca = jnp.maximum(jnp.float32(5.55e6), jnp.minimum(jnp.float32(9999.0e6), nbca_input * rho))

    l_qc = qc_input > R1
    rc_active = jnp.where(l_qc, qc_input * rho, jnp.ones_like(rho))
    nc_seed = jnp.maximum(jnp.float32(2.0), jnp.minimum(nc_input * rho, NT_C_MAX))
    nu_c = jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / nc_seed) + jnp.asarray(2, dtype=jnp.int32))
    nu_idx = nu_c - jnp.asarray(1, dtype=jnp.int32)
    lamc = (nc_seed * AM_R * CCG[1, nu_idx] * OCG1[nu_idx] / rc_active) ** OBMR
    xdc = (BM_R + nu_c.astype(jnp.float32) + jnp.float32(1.0)) / lamc
    lamc = jnp.where(xdc < D0C, CCE[1, nu_idx] / D0C, jnp.where(xdc > D0R * jnp.float32(2.0), CCE[1, nu_idx] / (D0R * jnp.float32(2.0)), lamc))
    nc_active = jnp.minimum(NT_C_MAX, CCG[0, nu_idx] * OCG2[nu_idx] * rc_active / AM_R * lamc**BM_R)
    if not is_aerosol_aware:
        nc_active = jnp.ones_like(nc_active) * NT_C
    qc_clean = jnp.where(l_qc, qc_input, jnp.float32(0.0))
    nc1d_clean = jnp.where(l_qc, nc_input, jnp.float32(0.0))
    rc = jnp.where(l_qc, rc_active, R1)
    nc = jnp.where(l_qc, nc_active, jnp.float32(2.0))

    l_qi = qi_input > R1
    ri_active = jnp.where(l_qi, qi_input * rho, jnp.ones_like(rho))
    ni_seed = jnp.maximum(R2, ni_input * rho)
    lami_default = _ice_lami_for_diameter(5.0e-6)
    ni_missing = _ice_number_from_lami(ri_active, lami_default, cap=999.0e3)
    ni_active = jnp.where(ni_seed <= R2, ni_missing, ni_seed)
    _, xdi = _ice_lami_xdi(ni_active, ri_active)
    ni_small = _ice_number_from_lami(ri_active, lami_default, cap=999.0e3)
    lami_large = _ice_lami_for_diameter(300.0e-6)
    ni_large = _ice_number_from_lami(ri_active, lami_large)
    ni_active = jnp.where(xdi < jnp.float32(5.0e-6), ni_small, jnp.where(xdi > jnp.float32(300.0e-6), ni_large, ni_active))
    qi_clean = jnp.where(l_qi, qi_input, jnp.float32(0.0))
    ni1d_clean = jnp.where(l_qi, ni_input, jnp.float32(0.0))
    ri = jnp.where(l_qi, ri_active, R1)
    ni = jnp.where(l_qi, ni_active, R2)

    l_qr = qr_input > R1
    rr_active = jnp.where(l_qr, qr_input * rho, jnp.ones_like(rho))
    nr_seed = jnp.maximum(R2, nr_input * rho)
    lamr_missing = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / jnp.float32(1.0e-3)
    nr_missing = CRG[1] * ORG3 * rr_active * lamr_missing**BM_R / AM_R
    nr_active = jnp.where(nr_seed <= R2, nr_missing, nr_seed)
    lamr = (AM_R * CRG[2] * ORG2 * nr_active / rr_active) ** OBMR
    mvd_r_active = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr
    mvd_r_high = jnp.float32(2.5e-3)
    lamr_high = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / mvd_r_high
    nr_high = CRG[1] * ORG3 * rr_active * lamr_high**BM_R / AM_R
    mvd_r_low = D0R * jnp.float32(0.75)
    lamr_low = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / mvd_r_low
    nr_low = CRG[1] * ORG3 * rr_active * lamr_low**BM_R / AM_R
    nr_active = jnp.where(mvd_r_active > mvd_r_high, nr_high, jnp.where(mvd_r_active < mvd_r_low, nr_low, nr_active))
    qr_clean = jnp.where(l_qr, qr_input, jnp.float32(0.0))
    nr1d_clean = jnp.where(l_qr, nr_input, jnp.float32(0.0))
    rr = jnp.where(l_qr, rr_active, R1)
    nr = jnp.where(l_qr, nr_active, R2)

    l_qs = qs_input > R1
    qs_clean = jnp.where(l_qs, qs_input, jnp.float32(0.0))
    rs = jnp.where(l_qs, qs_input * rho, R1)

    l_qg = qg_input > R1
    rg_active = jnp.where(l_qg, qg_input * rho, jnp.ones_like(rho))
    ng_seed = jnp.maximum(R2, ng_input * rho)
    rb_qg = jnp.maximum(qg_input / rho_g[-1], qb_input)
    rb_qg = jnp.minimum(qg_input / rho_g[0], rb_qg)
    rb_default = R1 / rho / rho_g[-1]
    rb_work = jnp.where(l_qg, rb_qg, rb_default)
    idx_bg_work = jnp.maximum(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.minimum(_fortran_nint(qg_input / rb_work * jnp.float32(0.01)) + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(RHO_G.size, dtype=jnp.int32)) - 1,
    )
    am_g_work = jnp.take(am_g, idx_bg_work)
    lamg_missing = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / jnp.float32(1.5e-3)
    ogg3 = jnp.float32(1.0) / CIG2
    ng_missing = ogg3 * rg_active * lamg_missing**BM_G / am_g_work
    ng_active = jnp.where(ng_seed <= R2, ng_missing, ng_seed)
    lamg_initial = (am_g_work * CIG2 * jnp.float32(1.0) * ng_active / rg_active) ** OBMG
    mvd_g_active = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / lamg_initial
    mvd_g_high = jnp.float32(25.4e-3)
    lamg_high = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / mvd_g_high
    ng_high = ogg3 * rg_active * lamg_high**BM_G / am_g_work
    mvd_g_low = D0R
    lamg_low = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / mvd_g_low
    ng_low = ogg3 * rg_active * lamg_low**BM_G / am_g_work
    ng_active = jnp.where(mvd_g_active > mvd_g_high, ng_high, jnp.where(mvd_g_active < mvd_g_low, ng_low, ng_active))
    mvd_g_active = jnp.where(mvd_g_active > mvd_g_high, mvd_g_high, jnp.where(mvd_g_active < mvd_g_low, mvd_g_low, mvd_g_active))
    idx_bg = jnp.where(l_qg, idx_bg_work, jnp.asarray(IDX_BG1, dtype=jnp.int32))
    if not is_hail_aware:
        idx_bg = jnp.ones_like(idx_bg) * jnp.asarray(IDX_BG1, dtype=jnp.int32)
    qg_clean = jnp.where(l_qg, qg_input, jnp.float32(0.0))
    ng1d_clean = jnp.where(l_qg, ng_input, jnp.float32(0.0))
    qb_clean = jnp.where(l_qg, rb_qg, jnp.float32(0.0))
    rg = jnp.where(l_qg, rg_active, R1)
    ng = jnp.where(l_qg, ng_active, R2)
    rb = jnp.where(l_qg, rb_qg, rb_default)
    mvd_g = jnp.where(l_qg, mvd_g_active, jnp.float32(0.0))

    tempc = temp - jnp.float32(273.15)
    rhof = sqrt_for_wrf(RHO_NOT / rho)
    rhof2 = sqrt_for_wrf(rhof)
    qvs = jnp.asarray(qvs1d, dtype=jnp.float32) if qvs1d is not None else rslf(pres, temp)
    delqvs = (
        jnp.asarray(delqvs1d, dtype=jnp.float32)
        if delqvs1d is not None
        else jnp.maximum(jnp.float32(0.0), rslf(pres, jnp.ones_like(temp) * jnp.float32(273.15)) - qv)
    )
    qvsi = jnp.asarray(qvsi1d, dtype=jnp.float32) if qvsi1d is not None else jnp.where(tempc <= jnp.float32(0.0), rsif(pres, temp), qvs)
    levels = jnp.arange(temp.shape[0], dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    k_melting = jnp.max(jnp.where(tempc > jnp.float32(0.0), levels, jnp.asarray(0, dtype=jnp.int32)))
    satw = qv / qvs
    sati = qv / qvsi
    ssatw = satw - jnp.float32(1.0)
    ssati = sati - jnp.float32(1.0)
    ssatw = jnp.where(jnp.abs(ssatw) < EPS, jnp.float32(0.0), ssatw)
    ssati = jnp.where(jnp.abs(ssati) < EPS, jnp.float32(0.0), ssati)
    diffu = jnp.float32(2.11e-5) * (temp / jnp.float32(273.15)) ** jnp.float32(1.94) * (jnp.float32(101325.0) / pres)
    visco_warm = (jnp.float32(1.718) + jnp.float32(0.0049) * tempc) * jnp.float32(1.0e-5)
    visco_cold = (jnp.float32(1.718) + jnp.float32(0.0049) * tempc - jnp.float32(1.2e-5) * tempc * tempc) * jnp.float32(1.0e-5)
    visco = jnp.where(tempc >= jnp.float32(0.0), visco_warm, visco_cold)
    ocp = jnp.float32(1.0) / (CP * (jnp.float32(1.0) + jnp.float32(0.887) * qv))
    vsc2 = sqrt_for_wrf(rho / visco)
    lvap = LVAP0 + (jnp.float32(2106.0) - jnp.float32(4218.0)) * tempc
    tcond = (jnp.float32(5.69) + jnp.float32(0.0168) * tempc) * jnp.float32(1.0e-5) * jnp.float32(418.936)
    dew_t = jnp.minimum(temp - jnp.float32(0.001), t_dew(pres, qv))
    tlcl = t_lcl(temp, dew_t)
    the = theta_e(pres, temp, qv, tlcl)
    twet_calc = jnp.minimum(temp, comp_t_fr_the(the, pres))
    twet_mask = (levels <= k_melting) & (satw < jnp.float32(0.999))
    twet = jnp.where(twet_mask, twet_calc, temp)

    hydrometeor_present = jnp.any(l_qc | l_qi | l_qr | l_qs | l_qg)
    no_micro = (~hydrometeor_present) & (~jnp.any(ssati > jnp.float32(0.0)))

    tc0 = jnp.minimum(jnp.float32(-0.1), tempc)
    smob_active = rs * OAMS
    a_bm, b_bm = _snow_moment_coeff(tc0, BM_S)
    smo2_generic = (smob_active / a_bm) ** (jnp.float32(1.0) / b_bm)
    if float(BM_S) > 2.0 - 1.0e-3 and float(BM_S) < 2.0 + 1.0e-3:
        smo2_active = smob_active
    else:
        smo2_active = smo2_generic
    a0, b0 = _snow_moment_coeff(tc0, jnp.float32(0.0))
    a1, b1 = _snow_moment_coeff(tc0, jnp.float32(1.0))
    ac, bc = _snow_moment_coeff(tc0, CSE[0])
    ae, be = _snow_moment_coeff(tc0, CSE[12])
    af, bf = _snow_moment_coeff(tc0, CSE[15])
    ag, bg = _snow_moment_coeff(tc0, CSE[16])
    smo0_active = a0 * smo2_active**b0
    smo1_active = a1 * smo2_active**b1
    smoc_active = ac * smo2_active**bc
    smoe_active = ae * smo2_active**be
    smof_active = af * smo2_active**bf
    smog_active = ag * smo2_active**bg
    safe_smoc = jnp.where(l_qs, smoc_active, jnp.float32(1.0))
    m0 = smob_active / safe_smoc
    mrat = smob_active * m0 * m0 * m0
    slam1 = m0 * LAM0
    slam2 = m0 * LAM1
    ns_active = mrat * KAP0 / slam1 + mrat * KAP1 * m0**MU_S * CSG[14] / slam2 ** CSE[14]
    zeros = jnp.zeros_like(temp)
    smob = jnp.where(l_qs, smob_active, zeros)
    smo2 = jnp.where(l_qs, smo2_active, zeros)
    smo0 = jnp.where(l_qs, smo0_active, zeros)
    smo1 = jnp.where(l_qs, smo1_active, zeros)
    smoc = jnp.where(l_qs, smoc_active, zeros)
    smod = zeros
    smoe = jnp.where(l_qs, smoe_active, zeros)
    smof = jnp.where(l_qs, smof_active, zeros)
    smog = jnp.where(l_qs, smog_active, zeros)
    ns = jnp.where(l_qs, ns_active, zeros)

    am_g_final = jnp.take(am_g, idx_bg)
    lamg = (am_g_final * CIG2 * ng / rg) ** OBMG
    ilamg = jnp.float32(1.0) / lamg
    n0_g = ng * lamg
    lamr_final = (AM_R * CRG[2] * ORG2 * nr / rr) ** OBMR
    ilamr = jnp.float32(1.0) / lamr_final
    mvd_r = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr_final
    n0_r = nr * ORG2 * lamr_final ** CRE[1]
    mvd_c = zeros

    return ThompsonPreprocessResult(
        # WRF keeps the parent ``qv1d`` argument unchanged during
        # preprocessing.  ``qv`` is only the floor-protected local work
        # array.  This distinction matters when ``no_micro`` returns early:
        # ``mp_gt_driver`` must still see the original negative value so its
        # vertical-neighbour repair can run.
        qv1d=qv_input,
        qc1d=qc_clean,
        qi1d=qi_clean,
        qr1d=qr_clean,
        qs1d=qs_clean,
        qg1d=qg_clean,
        qb1d=qb_clean,
        ni1d=ni1d_clean,
        nr1d=nr1d_clean,
        nc1d=nc1d_clean,
        ng1d=ng1d_clean,
        temp=temp,
        pres=pres,
        qv=qv,
        rc=rc,
        ri=ri,
        rr=rr,
        rs=rs,
        rg=rg,
        rb=rb,
        ni=ni,
        nr=nr,
        nc=nc,
        ns=ns,
        ng=ng,
        nwfa=nwfa,
        nifa=nifa,
        nbca=nbca,
        rho=rho,
        rhof=rhof,
        rhof2=rhof2,
        qvs=qvs,
        qvsi=qvsi,
        delqvs=delqvs,
        satw=satw,
        sati=sati,
        ssatw=ssatw,
        ssati=ssati,
        diffu=diffu,
        visco=visco,
        vsc2=vsc2,
        tcond=tcond,
        lvap=lvap,
        ocp=ocp,
        twet=twet,
        ilamr=ilamr,
        ilamg=ilamg,
        n0_r=n0_r,
        n0_g=n0_g,
        mvd_r=mvd_r,
        mvd_c=mvd_c,
        mvd_g=mvd_g,
        smob=smob,
        smo2=smo2,
        smo1=smo1,
        smo0=smo0,
        smoc=smoc,
        smod=smod,
        smoe=smoe,
        smof=smof,
        smog=smog,
        l_qc=l_qc,
        l_qi=l_qi,
        l_qr=l_qr,
        l_qs=l_qs,
        l_qg=l_qg,
        idx_bg=idx_bg,
        no_micro=no_micro,
        k_melting=k_melting,
    )


@custom_jvp
def _clipped_fractional_power(value: Any, exponent: Any) -> Any:
    return jnp.maximum(value, jnp.float32(0.0)) ** exponent


@_clipped_fractional_power.defjvp
def _clipped_fractional_power_jvp(primals, tangents):
    value, exponent = primals
    result = _clipped_fractional_power(value, exponent)
    # These clipped branches have no cloud/snow process at zero. Avoid an
    # infinite root derivative contaminating their zero reverse seeds.
    positive = value > 0
    safe = jnp.where(positive, value, jnp.ones_like(value))
    slope = jnp.where(positive, exponent * result / safe, 0)
    return result, slope * tangents[0] + result * jnp.log(safe) * tangents[1]


def thompson_warm_rain_terms(
    pre: ThompsonPreprocessResult,
    efrw_table: Any,
    dt: Any,
    *,
    wif_input_opt: int = 1,
) -> ThompsonWarmRainResult:
    """Compute WRF Thompson warm-rain process rates before evaporation."""

    efrw = jnp.asarray(efrw_table, dtype=jnp.float32)
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)

    nu_c = jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / pre.nc) + jnp.asarray(2, dtype=jnp.int32))
    nu_idx = nu_c - jnp.asarray(1, dtype=jnp.int32)
    lamc = (pre.nc * AM_R * CCG[1, nu_idx] * OCG1[nu_idx] / pre.rc) ** OBMR
    xdc = jnp.maximum(D0C * jnp.float32(1.0e6), (pre.rc / (AM_R * pre.nc)) ** OBMR * jnp.float32(1.0e6))
    mvd_c_active = (jnp.float32(3.0) + nu_c.astype(jnp.float32) + jnp.float32(0.672)) / lamc
    mvd_c_active = jnp.maximum(D0C, jnp.minimum(mvd_c_active, D0R))
    mvd_c = jnp.where(pre.l_qc, mvd_c_active, jnp.ones_like(pre.temp) * D0C)

    rain_collecting = pre.l_qr & (pre.mvd_r > D0R)
    ef_rr = jnp.float32(1.0) - jnp.exp(jnp.float32(2300.0) * (pre.mvd_r - jnp.float32(1950.0e-6)))
    pnr_rcr = jnp.where(rain_collecting, ef_rr * jnp.float32(2.0) * pre.nr * pre.rr, zeros)

    dc_g = ((CCG[2, nu_idx] * OCG2[nu_idx]) ** OBMR / lamc) * jnp.float32(1.0e6)
    dc_b_arg = xdc * xdc * xdc * dc_g * dc_g * dc_g - xdc * xdc * xdc * xdc * xdc * xdc
    dc_b = _clipped_fractional_power(dc_b_arg, jnp.float32(1.0) / jnp.float32(6.0))
    zeta1_raw = jnp.float32(6.25e-6) * xdc * dc_b * dc_b * dc_b - jnp.float32(0.4)
    zeta1 = jnp.float32(0.5) * (zeta1_raw + jnp.abs(zeta1_raw))
    zeta = jnp.float32(0.027) * pre.rc * zeta1
    taud_raw = jnp.float32(0.5) * dc_b - jnp.float32(7.5)
    taud = jnp.float32(0.5) * (taud_raw + jnp.abs(taud_raw)) + R1
    tau = real_divide(jnp.float32(3.72), pre.rc * taud)
    autoconv = pre.rc > jnp.float32(0.01e-3)
    prr_wau_raw = zeta / tau
    prr_wau = jnp.where(autoconv, jnp.minimum(pre.rc * odts, prr_wau_raw), zeros)
    pnr_wau = jnp.where(autoconv, prr_wau / (AM_R * nu_c.astype(jnp.float32) * jnp.float32(10.0) * D0R * D0R * D0R), zeros)
    pnc_wau = jnp.where(autoconv, jnp.minimum(pre.nc * odts, prr_wau / (AM_R * mvd_c * mvd_c * mvd_c)), zeros)

    lamr = jnp.float32(1.0) / pre.ilamr
    rain_diameters = jnp.asarray(thompson_rain_diameters(), dtype=jnp.float32)
    idx_r = jnp.asarray(1, dtype=jnp.int32) + jnp.floor(
        jnp.float32(NBR) * jnp.log(pre.mvd_r / rain_diameters[0]) / jnp.log(rain_diameters[-1] / rain_diameters[0])
    ).astype(jnp.int32)
    idx_r = jnp.clip(idx_r, 1, NBR) - 1
    idx_c = jnp.clip(jnp.floor(mvd_c * jnp.float32(1.0e6)).astype(jnp.int32), 1, NBC) - 1
    ef_rw = efrw[idx_r, idx_c]
    collect_cloud = rain_collecting & (mvd_c > D0C)
    t1_qr_qc = _PI * jnp.float32(0.25) * AV_R * CRG[8]
    rain_kernel = pre.rhof * t1_qr_qc * ef_rw * pre.n0_r * ((lamr + FV_R) ** (-CRE[8]))
    prr_rcw = jnp.where(collect_cloud, jnp.minimum(pre.rc * odts, rain_kernel * pre.rc), zeros)
    pnc_rcw = jnp.where(collect_cloud, jnp.minimum(pre.nc * odts, rain_kernel * pre.nc), zeros)

    pna_rca = jnp.where(
        rain_collecting,
        jnp.minimum(
            pre.nwfa * odts,
            pre.rhof
            * t1_qr_qc
            * eff_aero(pre.mvd_r, jnp.float32(0.04e-6), pre.visco, pre.rho, pre.temp, "r")
            * pre.nwfa
            * pre.n0_r
            * ((lamr + FV_R) ** (-CRE[8])),
        ),
        zeros,
    )
    pnd_rcd = jnp.where(
        rain_collecting,
        jnp.minimum(
            pre.nifa * odts,
            pre.rhof
            * t1_qr_qc
            * eff_aero(pre.mvd_r, jnp.float32(0.8e-6), pre.visco, pre.rho, pre.temp, "r")
            * pre.nifa
            * pre.n0_r
            * ((lamr + FV_R) ** (-CRE[8])),
        ),
        zeros,
    )
    if int(wif_input_opt) == 2:
        pnb_rcb = jnp.where(
            rain_collecting,
            jnp.minimum(
                pre.nbca * odts,
                pre.rhof
                * t1_qr_qc
                * eff_aero(pre.mvd_r, jnp.float32(0.0236e-6), pre.visco, pre.rho, pre.temp, "r")
                * pre.nbca
                * pre.n0_r
                * ((lamr + FV_R) ** (-CRE[8])),
            ),
            zeros,
        )
    else:
        pnb_rcb = zeros

    return ThompsonWarmRainResult(
        mvd_c=mvd_c,
        prr_wau=prr_wau,
        pnr_wau=pnr_wau,
        pnc_wau=pnc_wau,
        prr_rcw=prr_rcw,
        pnc_rcw=pnc_rcw,
        pnr_rcr=pnr_rcr,
        pna_rca=pna_rca,
        pnd_rcd=pnd_rcd,
        pnb_rcb=pnb_rcb,
    )


def _decade_lookup_index(value: Any, first_value: Any, base_power: int, size: int, *, else_one_based: int = 1) -> Any:
    val = jnp.asarray(value, dtype=jnp.float32)
    active = val > jnp.asarray(first_value, dtype=jnp.float32)
    safe_val = jnp.maximum(val, jnp.float32(1.0e-30))
    exponent = jnp.floor(jnp.log10(safe_val)).astype(jnp.int32)
    normalized = val / (jnp.float32(10.0) ** exponent.astype(jnp.float32))
    idx_one = normalized.astype(jnp.int32) + jnp.asarray(10, dtype=jnp.int32) * (exponent - jnp.asarray(base_power, dtype=jnp.int32)) - (
        exponent - jnp.asarray(base_power, dtype=jnp.int32)
    )
    idx_one = jnp.where(active, idx_one, jnp.asarray(else_one_based, dtype=jnp.int32))
    idx_one = jnp.clip(idx_one, 1, size)
    return idx_one - jnp.asarray(1, dtype=jnp.int32)


def _cloud_number_lookup_index(number: Any) -> Any:
    """Return WRF's zero-based freezing/evaporation table index."""
    bins = thompson_cloud_number_bins()
    # WRF declares nic1 INTEGER, so assigning DLOG truncates 7.926... to 7.
    nic1 = int(np.log(bins[-1] / bins[0]))
    scaled = 1.0 + NBC * jnp.log(jnp.asarray(number, dtype=jnp.float64) / bins[0]) / nic1
    one_based = jnp.floor(jnp.maximum(scaled, 1.0) + 0.5).astype(jnp.int32)
    return jnp.clip(one_based, 1, NBC) - 1


def thompson_frozen_lookup_indices(pre: ThompsonPreprocessResult) -> ThompsonFrozenLookupIndices:
    """Compute Thompson lookup-table indexes used by frozen hydrometeor terms."""

    tempc = pre.temp - jnp.float32(273.15)
    snow_mass = jnp.where(pre.l_qs, pre.smob, jnp.ones_like(pre.smob))
    xds = jnp.where(pre.l_qs, pre.smoc / snow_mass, jnp.zeros_like(pre.temp))
    rho_g = jnp.asarray(RHO_G, dtype=jnp.float32)
    am_g = jnp.asarray(AM_G, dtype=jnp.float32)
    snow_size = jnp.where(pre.l_qs, xds, jnp.ones_like(xds))
    rho_s = jnp.where(pre.l_qs, jnp.maximum(rho_g[0], jnp.minimum(jnp.float32(0.13) / snow_size, RHO_I - jnp.float32(100.0))), jnp.float32(100.0))

    idx_tc = jnp.clip(_fortran_nint(-tempc), 1, 45) - jnp.asarray(1, dtype=jnp.int32)
    idx_t_work = ((tempc - jnp.float32(2.5)) / jnp.float32(5.0)).astype(jnp.int32) - jnp.asarray(1, dtype=jnp.int32)
    idx_t = jnp.minimum(jnp.maximum(jnp.asarray(1, dtype=jnp.int32), -idx_t_work), jnp.asarray(NTB_T, dtype=jnp.int32)) - jnp.asarray(1, dtype=jnp.int32)
    it = jnp.clip(_fortran_nint(-tempc), 1, 31) - jnp.asarray(1, dtype=jnp.int32)

    idx_c = _decade_lookup_index(pre.rc, R_C[0], -6, NTB_C)
    idx_n = _cloud_number_lookup_index(pre.nc)

    r_i = jnp.asarray(thompson_ice_table_mixing_ratios(), dtype=jnp.float32)
    nt_i = jnp.asarray(thompson_ice_number_concentrations(), dtype=jnp.float32)
    r_r = jnp.asarray(thompson_rain_table_mixing_ratios(), dtype=jnp.float32)
    r_s = jnp.asarray(thompson_snow_table_mixing_ratios(), dtype=jnp.float32)
    r_g = jnp.asarray(thompson_graupel_table_mixing_ratios(), dtype=jnp.float32)
    n0r_exp = jnp.asarray(thompson_rain_intercepts(), dtype=jnp.float32)
    n0g_exp = jnp.asarray(thompson_graupel_intercepts(), dtype=jnp.float32)

    idx_i = _decade_lookup_index(pre.ri, r_i[0], -10, NTB_I)
    idx_i1 = _decade_lookup_index(pre.ni, nt_i[0], 0, NTB_I1)
    idx_r = _decade_lookup_index(pre.rr, r_r[0], -6, NTB_R)
    idx_s = _decade_lookup_index(pre.rs, r_s[0], -6, NTB_S)
    idx_g = _decade_lookup_index(pre.rg, r_g[0], -6, NTB_G)

    lamr = jnp.float32(1.0) / pre.ilamr
    lam_exp_r = lamr * (CRG[2] * ORG2 * ORG1) ** BM_R
    n0_exp_r = ORG1 * pre.rr / AM_R * lam_exp_r**CRE[0]
    idx_r1_active = _decade_lookup_index(n0_exp_r, n0r_exp[0], 6, NTB_R1)
    idx_r1 = jnp.where(pre.rr > r_r[0], idx_r1_active, jnp.asarray(NTB_R1 - 1, dtype=jnp.int32))

    idx_bg = pre.idx_bg
    am_g_idx = jnp.take(am_g, idx_bg)
    lamg = jnp.float32(1.0) / pre.ilamg
    lam_exp_g = lamg * (CGG[2, 0] * OGG2 * OGG1) ** BM_G
    n0_exp_g = OGG1 * pre.rg / am_g_idx * lam_exp_g**CGE[0, 0]
    idx_g1_active = _decade_lookup_index(n0_exp_g, n0g_exp[0], 2, NTB_G1)
    idx_g1 = jnp.where(pre.rg > r_g[0], idx_g1_active, jnp.asarray(NTB_G1 - 1, dtype=jnp.int32))

    return ThompsonFrozenLookupIndices(
        xds=xds,
        rho_s=rho_s,
        idx_tc=idx_tc,
        idx_t=idx_t,
        it=it,
        idx_c=idx_c,
        idx_n=idx_n,
        idx_i=idx_i,
        idx_i1=idx_i1,
        idx_r=idx_r,
        idx_r1=idx_r1,
        idx_s=idx_s,
        idx_g=idx_g,
        idx_g1=idx_g1,
        n0_exp_r=n0_exp_r,
        n0_exp_g=n0_exp_g,
    )


def thompson_frozen_collection_terms(
    pre: ThompsonPreprocessResult,
    warm: ThompsonWarmRainResult,
    lookup: ThompsonFrozenLookupIndices,
    efsw_table: Any,
    dt: Any,
    *,
    wif_input_opt: int = 1,
) -> ThompsonFrozenCollectionResult:
    """Compute Thompson snow/graupel cloud-water collection and wet scavenging."""

    efsw = jnp.asarray(efsw_table, dtype=jnp.float32)
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    snow_diameters = jnp.asarray(thompson_snow_diameters(), dtype=jnp.float32)
    r_s = jnp.asarray(thompson_snow_table_mixing_ratios(), dtype=jnp.float32)
    r_g = jnp.asarray(thompson_graupel_table_mixing_ratios(), dtype=jnp.float32)
    bg = pre.idx_bg

    idx_ds = jnp.asarray(1, dtype=jnp.int32) + jnp.floor(
        jnp.float32(NBS) * jnp.log(lookup.xds / snow_diameters[0]) / jnp.log(snow_diameters[-1] / snow_diameters[0])
    ).astype(jnp.int32)
    idx_ds = jnp.clip(idx_ds, 1, NBS) - jnp.asarray(1, dtype=jnp.int32)
    idx_mvd_c = jnp.clip(jnp.floor(warm.mvd_c * jnp.float32(1.0e6)).astype(jnp.int32), 1, NBC) - jnp.asarray(1, dtype=jnp.int32)
    ef_sw = efsw[idx_ds, idx_mvd_c]

    t1_qs_qc = _PI * jnp.float32(0.25) * AV_S
    cloud_collect = pre.l_qc & (warm.mvd_c > D0C)
    snow_collect_cloud = cloud_collect & (lookup.xds > D0S)
    prs_scw_raw = pre.rhof * t1_qs_qc * ef_sw * pre.rc * pre.smoe
    pnc_scw_raw = pre.rhof * t1_qs_qc * ef_sw * pre.nc * pre.smoe
    prs_scw = jnp.where(snow_collect_cloud, jnp.minimum(pre.rc * odts, prs_scw_raw), zeros)
    pnc_scw = jnp.where(snow_collect_cloud, jnp.minimum(pre.nc * odts, pnc_scw_raw), zeros)

    av_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), AV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(AV_G, dtype=jnp.float32), bg))
    bv_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), BV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(BV_G, dtype=jnp.float32), bg))
    cge6 = BM_G + MU_G + bv_g + jnp.float32(1.0)
    cgg6 = wgamma(cge6)
    cge9 = MU_G + bv_g + jnp.float32(3.0)
    cgg9 = wgamma(cge9)
    xdg = (BM_G + MU_G + jnp.float32(1.0)) * pre.ilamg
    vtg = pre.rhof * av_g * cgg6 * OGG3 * pre.ilamg**bv_g
    stoke_g = warm.mvd_c * warm.mvd_c * vtg * RHO_W / (jnp.float32(9.0) * pre.visco * xdg)
    const_ri = -warm.mvd_c * jnp.float32(0.5e6) * vtg / jnp.minimum(jnp.float32(-0.1), pre.temp - jnp.float32(273.15))
    const_ri = jnp.maximum(jnp.float32(0.1), jnp.minimum(const_ri, jnp.float32(10.0)))
    rime_dens = (jnp.float32(0.051) + jnp.float32(0.114) * const_ri - jnp.float32(0.0055) * const_ri * const_ri) * jnp.float32(1000.0)
    ef_gw_mid = jnp.float32(0.55) * jnp.log10(jnp.float32(2.51) * stoke_g)
    ef_gw = jnp.where(stoke_g < jnp.float32(0.4), zeros, jnp.where(stoke_g > jnp.float32(10.0), jnp.float32(0.77), ef_gw_mid))
    ef_gw = jnp.where(pre.twet > jnp.float32(273.15), ef_gw * jnp.float32(0.1), ef_gw)
    t1_qg_qc = _PI * jnp.float32(0.25) * av_g * cgg9
    graupel_collect_cloud = cloud_collect & (pre.rg >= r_g[0])
    graupel_kernel = pre.rhof * t1_qg_qc * ef_gw * pre.n0_g * pre.ilamg**cge9
    prg_gcw = jnp.where(graupel_collect_cloud, graupel_kernel * pre.rc, zeros)
    pnc_gcw = jnp.where(graupel_collect_cloud, jnp.minimum(pre.nc * odts, graupel_kernel * pre.nc), zeros)
    pbg_gcw = jnp.where(graupel_collect_cloud & (pre.twet < jnp.float32(273.15)), prg_gcw / rime_dens, zeros)

    snow_active = pre.rs > r_s[0]
    pna_sca = jnp.where(
        snow_active,
        jnp.minimum(
            pre.nwfa * odts,
            pre.rhof * t1_qs_qc * eff_aero(lookup.xds, jnp.float32(0.04e-6), pre.visco, pre.rho, pre.temp, "s") * pre.nwfa * pre.smoe,
        ),
        zeros,
    )
    pnd_scd = jnp.where(
        snow_active,
        jnp.minimum(
            pre.nifa * odts,
            pre.rhof * t1_qs_qc * eff_aero(lookup.xds, jnp.float32(0.8e-6), pre.visco, pre.rho, pre.temp, "s") * pre.nifa * pre.smoe,
        ),
        zeros,
    )
    if int(wif_input_opt) == 2:
        pnb_scb = jnp.where(
            snow_active,
            jnp.minimum(
                pre.nbca * odts,
                pre.rhof * t1_qs_qc * eff_aero(lookup.xds, jnp.float32(0.0236e-6), pre.visco, pre.rho, pre.temp, "s") * pre.nbca * pre.smoe,
            ),
            zeros,
        )
    else:
        pnb_scb = zeros

    graupel_active = pre.rg > r_g[0]
    pna_gca = jnp.where(
        graupel_active,
        jnp.minimum(
            pre.nwfa * odts,
            pre.rhof
            * t1_qg_qc
            * eff_aero(xdg, jnp.float32(0.04e-6), pre.visco, pre.rho, pre.temp, "g")
            * pre.nwfa
            * pre.n0_g
            * pre.ilamg**cge9,
        ),
        zeros,
    )
    pnd_gcd = jnp.where(
        graupel_active,
        jnp.minimum(
            pre.nifa * odts,
            pre.rhof
            * t1_qg_qc
            * eff_aero(xdg, jnp.float32(0.8e-6), pre.visco, pre.rho, pre.temp, "g")
            * pre.nifa
            * pre.n0_g
            * pre.ilamg**cge9,
        ),
        zeros,
    )
    if int(wif_input_opt) == 2:
        pnb_gcb = jnp.where(
            graupel_active,
            jnp.minimum(
                pre.nbca * odts,
                pre.rhof
                * t1_qg_qc
                * eff_aero(xdg, jnp.float32(0.0236e-6), pre.visco, pre.rho, pre.temp, "g")
                * pre.nbca
                * pre.n0_g
                * pre.ilamg**cge9,
            ),
            zeros,
        )
    else:
        pnb_gcb = zeros

    return ThompsonFrozenCollectionResult(
        prs_scw=prs_scw,
        pnc_scw=pnc_scw,
        prg_gcw=prg_gcw,
        pnc_gcw=pnc_gcw,
        pbg_gcw=pbg_gcw,
        pna_sca=pna_sca,
        pnd_scd=pnd_scd,
        pnb_scb=pnb_scb,
        pna_gca=pna_gca,
        pnd_gcd=pnd_gcd,
        pnb_gcb=pnb_gcb,
        ef_sw=jnp.where(snow_collect_cloud, ef_sw, zeros),
        ef_gw=jnp.where(graupel_collect_cloud, ef_gw, zeros),
        rime_dens=jnp.where(graupel_collect_cloud, rime_dens, zeros),
    )


def thompson_rain_frozen_collection_terms(
    pre: ThompsonPreprocessResult,
    lookup: ThompsonFrozenLookupIndices,
    qr_acr_qs: ThompsonQrAcrQsTables,
    qr_acr_qg: ThompsonQrAcrQgTables,
    dt: Any,
) -> ThompsonRainFrozenCollectionResult:
    """Compute WRF Thompson rain collecting snow/graupel lookup-table rates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    r_r = jnp.asarray(thompson_rain_table_mixing_ratios(), dtype=jnp.float32)
    r_s = jnp.asarray(thompson_snow_table_mixing_ratios(), dtype=jnp.float32)
    r_g = jnp.asarray(thompson_graupel_table_mixing_ratios(), dtype=jnp.float32)

    idx_s = lookup.idx_s
    idx_t = lookup.idx_t
    idx_r1 = lookup.idx_r1
    idx_r = lookup.idx_r
    idx_g1 = lookup.idx_g1
    idx_g = lookup.idx_g
    bg = pre.idx_bg
    qg_table_bg = jnp.zeros_like(bg) if int(qr_acr_qg.tcg_racg.shape[2]) == 1 else bg

    tcs_racs1 = jnp.asarray(qr_acr_qs.tcs_racs1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tmr_racs1 = jnp.asarray(qr_acr_qs.tmr_racs1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tcr_sacr1 = jnp.asarray(qr_acr_qs.tcr_sacr1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tms_sacr1 = jnp.asarray(qr_acr_qs.tms_sacr1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tmr_racs2 = jnp.asarray(qr_acr_qs.tmr_racs2, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tcr_sacr2 = jnp.asarray(qr_acr_qs.tcr_sacr2, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tnr_racs1 = jnp.asarray(qr_acr_qs.tnr_racs1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tnr_racs2 = jnp.asarray(qr_acr_qs.tnr_racs2, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tnr_sacr1 = jnp.asarray(qr_acr_qs.tnr_sacr1, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]
    tnr_sacr2 = jnp.asarray(qr_acr_qs.tnr_sacr2, dtype=jnp.float32)[idx_s, idx_t, idx_r1, idx_r]

    rain_snow = (pre.rr >= r_r[0]) & (pre.rs >= r_s[0])
    cold = pre.twet < jnp.float32(273.15)

    prr_rcs_cold = -(tmr_racs2 + tcr_sacr2 + tmr_racs1 + tcr_sacr1)
    prs_rcs_cold = tmr_racs2 + tcr_sacr2 - tcs_racs1 - tms_sacr1
    prg_rcs_cold = tmr_racs1 + tcr_sacr1 + tcs_racs1 + tms_sacr1
    prr_rcs_cold = jnp.maximum(-pre.rr * odts, prr_rcs_cold)
    prs_rcs_cold = jnp.maximum(-pre.rs * odts, prs_rcs_cold)
    prg_rcs_cold = jnp.minimum((pre.rr + pre.rs) * odts, prg_rcs_cold)
    pnr_rcs_cold = tnr_racs1 + tnr_racs2 + tnr_sacr1 + tnr_sacr2
    pnr_rcs_cold = jnp.minimum(pre.nr * odts, pnr_rcs_cold)
    png_rcs_cold = pnr_rcs_cold
    pbg_rcs_cold = prg_rcs_cold / RHO_I

    prs_rcs_warm = -tcs_racs1 - tms_sacr1 + tmr_racs2 + tcr_sacr2
    prs_rcs_warm = jnp.maximum(-pre.rs * odts, prs_rcs_warm)
    prr_rcs_warm = -prs_rcs_warm

    prr_rcs = jnp.where(rain_snow, jnp.where(cold, prr_rcs_cold, prr_rcs_warm), zeros)
    prs_rcs = jnp.where(rain_snow, jnp.where(cold, prs_rcs_cold, prs_rcs_warm), zeros)
    prg_rcs = jnp.where(rain_snow & cold, prg_rcs_cold, zeros)
    pnr_rcs = jnp.where(rain_snow & cold, pnr_rcs_cold, zeros)
    png_rcs = jnp.where(rain_snow & cold, png_rcs_cold, zeros)
    pbg_rcs = jnp.where(rain_snow & cold, pbg_rcs_cold, zeros)

    tcg_racg = jnp.asarray(qr_acr_qg.tcg_racg, dtype=jnp.float32)[idx_g1, idx_g, qg_table_bg, idx_r1, idx_r]
    tmr_racg = jnp.asarray(qr_acr_qg.tmr_racg, dtype=jnp.float32)[idx_g1, idx_g, qg_table_bg, idx_r1, idx_r]
    tcr_gacr = jnp.asarray(qr_acr_qg.tcr_gacr, dtype=jnp.float32)[idx_g1, idx_g, qg_table_bg, idx_r1, idx_r]
    tnr_racg = jnp.asarray(qr_acr_qg.tnr_racg, dtype=jnp.float32)[idx_g1, idx_g, qg_table_bg, idx_r1, idx_r]
    tnr_gacr = jnp.asarray(qr_acr_qg.tnr_gacr, dtype=jnp.float32)[idx_g1, idx_g, qg_table_bg, idx_r1, idx_r]

    rain_graupel = (pre.rr >= r_r[0]) & (pre.rg >= r_g[0])
    prg_rcg_cold = jnp.minimum(pre.rr * odts, tmr_racg + tcr_gacr)
    prr_rcg_cold = -prg_rcg_cold
    pnr_rcg_cold = jnp.minimum(pre.nr * odts, tnr_racg + tnr_gacr)
    pbg_rcg_cold = prg_rcg_cold / RHO_I

    prr_rcg_warm = jnp.minimum(pre.rg * odts, tcg_racg)
    prg_rcg_warm = -prr_rcg_warm
    png_rcg_warm = jnp.minimum(pre.ng * odts, tnr_racg)
    rho_g = jnp.take(jnp.asarray(RHO_G, dtype=jnp.float32), bg)
    pbg_rcg_warm = prg_rcg_warm / rho_g
    pnr_rcg_warm = -jnp.float32(5.0) * tnr_gacr

    prr_rcg = jnp.where(rain_graupel, jnp.where(cold, prr_rcg_cold, prr_rcg_warm), zeros)
    prg_rcg = jnp.where(rain_graupel, jnp.where(cold, prg_rcg_cold, prg_rcg_warm), zeros)
    pnr_rcg = jnp.where(rain_graupel, jnp.where(cold, pnr_rcg_cold, pnr_rcg_warm), zeros)
    png_rcg = jnp.where(rain_graupel & (~cold), png_rcg_warm, zeros)
    pbg_rcg = jnp.where(rain_graupel, jnp.where(cold, pbg_rcg_cold, pbg_rcg_warm), zeros)

    return ThompsonRainFrozenCollectionResult(
        prr_rcs=prr_rcs,
        prs_rcs=prs_rcs,
        prg_rcs=prg_rcs,
        pnr_rcs=pnr_rcs,
        png_rcs=png_rcs,
        pbg_rcs=pbg_rcs,
        prr_rcg=prr_rcg,
        prg_rcg=prg_rcg,
        pnr_rcg=pnr_rcg,
        png_rcg=png_rcg,
        pbg_rcg=pbg_rcg,
    )


def thompson_freezing_terms(
    pre: ThompsonPreprocessResult,
    lookup: ThompsonFrozenLookupIndices,
    freeze_h2o: ThompsonFreezeH2OTables,
    dt: Any,
) -> ThompsonFreezingResult:
    """Compute WRF Thompson rain-drop and cloud-water freezing rates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    cold = pre.temp < jnp.float32(273.15)

    r_r = jnp.asarray(thompson_rain_table_mixing_ratios(), dtype=jnp.float32)
    nt_in = jnp.asarray(thompson_ice_nuclei_concentrations(), dtype=jnp.float32)
    r_c = jnp.asarray(R_C, dtype=jnp.float32)

    xni = jnp.full_like(pre.temp, jnp.float32(1000.0))
    idx_in = _decade_lookup_index(xni, nt_in[0], 0, NTB_IN)

    idx_r = lookup.idx_r
    idx_r1 = lookup.idx_r1
    idx_tc = lookup.idx_tc
    idx_c = lookup.idx_c
    idx_n = lookup.idx_n

    tpi_qrfz = jnp.asarray(freeze_h2o.tpi_qrfz, dtype=jnp.float32)[idx_r, idx_r1, idx_tc, idx_in]
    tpg_qrfz = jnp.asarray(freeze_h2o.tpg_qrfz, dtype=jnp.float32)[idx_r, idx_r1, idx_tc, idx_in]
    tni_qrfz = jnp.asarray(freeze_h2o.tni_qrfz, dtype=jnp.float32)[idx_r, idx_r1, idx_tc, idx_in]
    tnr_qrfz = jnp.asarray(freeze_h2o.tnr_qrfz, dtype=jnp.float32)[idx_r, idx_r1, idx_tc, idx_in]

    rain_lookup = pre.rr > r_r[0]
    rain_homogeneous = (pre.rr > R1) & (~rain_lookup) & (pre.temp < HGFR)
    prg_rfz_lookup = tpg_qrfz * odts
    pri_rfz_lookup = tpi_qrfz * odts
    pni_rfz_lookup = tni_qrfz * odts
    pnr_rfz_lookup = jnp.minimum(pre.nr * odts, tnr_qrfz * odts)
    prg_rfz = jnp.where(cold & rain_lookup, prg_rfz_lookup, zeros)
    pri_rfz = jnp.where(cold & rain_lookup, pri_rfz_lookup, jnp.where(cold & rain_homogeneous, pre.rr * odts, zeros))
    pni_rfz = jnp.where(cold & rain_lookup, pni_rfz_lookup, jnp.where(cold & rain_homogeneous, pre.nr * odts, zeros))
    pnr_rfz = jnp.where(cold & rain_lookup, pnr_rfz_lookup, zeros)
    pbg_rfz = prg_rfz / RHO_I

    tpi_qcfz = jnp.asarray(freeze_h2o.tpi_qcfz, dtype=jnp.float32)[idx_c, idx_n, idx_tc, idx_in]
    tni_qcfz = jnp.asarray(freeze_h2o.tni_qcfz, dtype=jnp.float32)[idx_c, idx_n, idx_tc, idx_in]
    cloud_lookup = pre.rc > r_c[0]
    cloud_homogeneous = (pre.rc > R1) & (~cloud_lookup) & (pre.temp < HGFR)
    pri_wfz_lookup = jnp.minimum(pre.rc * odts, tpi_qcfz * odts)
    pni_wfz_lookup = jnp.minimum(jnp.minimum(pre.nc * odts, pri_wfz_lookup / (jnp.float32(2.0) * XM0I)), tni_qcfz * odts)
    pri_wfz = jnp.where(cold & cloud_lookup, pri_wfz_lookup, jnp.where(cold & cloud_homogeneous, pre.rc * odts, zeros))
    pni_wfz = jnp.where(cold & cloud_lookup, pni_wfz_lookup, jnp.where(cold & cloud_homogeneous, pre.nc * odts, zeros))

    return ThompsonFreezingResult(
        idx_in=idx_in,
        prg_rfz=prg_rfz,
        pri_rfz=pri_rfz,
        pni_rfz=pni_rfz,
        pnr_rfz=pnr_rfz,
        pbg_rfz=pbg_rfz,
        pri_wfz=pri_wfz,
        pni_wfz=pni_wfz,
    )


def thompson_ice_nucleation_terms(
    pre: ThompsonPreprocessResult,
    freezing: ThompsonFreezingResult,
    dt: Any,
    *,
    is_aerosol_aware: bool = False,
    dusty_ice: bool = True,
    homog_ice: bool = True,
) -> ThompsonIceNucleationResult:
    """Compute WRF Thompson deposition and homogeneous ice nucleation rates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    tempc = pre.temp - jnp.float32(273.15)
    rate_max = (pre.qv - pre.qvsi) * pre.rho * odts * jnp.float32(0.999)

    # These processes are inside mp_thompson's parent IF (temp < T_0).
    cold = pre.temp < jnp.float32(273.15)
    nucleate = cold & (
        (pre.ssati >= jnp.float32(0.25))
        | ((pre.ssatw > EPS) & (pre.temp < jnp.float32(253.15)))
    )
    if bool(dusty_ice) and bool(is_aerosol_aware):
        xnc_inu = ice_demott(tempc, pre.qv, pre.qvs, pre.qvsi, pre.rho, pre.nifa)
    else:
        xnc_inu = jnp.minimum(jnp.float32(250.0e3), TNO * jnp.exp(ATO * (jnp.float32(273.15) - pre.temp)))

    xni_inu = pre.ni + (freezing.pni_rfz + freezing.pni_wfz) * dt_arr
    pni_inu_raw = jnp.float32(0.5) * (xnc_inu - xni_inu + jnp.abs(xnc_inu - xni_inu)) * odts
    pri_inu_raw = jnp.minimum(rate_max, XM0I * pni_inu_raw)
    pni_inu_limited = pri_inu_raw / XM0I
    pri_inu = jnp.where(nucleate, pri_inu_raw, zeros)
    pni_inu = jnp.where(nucleate, pni_inu_limited, zeros)

    xni_iha = pre.ns + pre.ni + (freezing.pni_rfz + freezing.pni_wfz + pni_inu) * dt_arr
    homog = (
        bool(is_aerosol_aware)
        and bool(homog_ice)
    )
    if homog:
        homog_active = (xni_iha <= jnp.float32(999.0e3)) & (pre.temp < jnp.float32(238.0)) & (pre.ssati >= jnp.float32(0.4))
        xnc_iha = ice_koop(pre.temp, pre.qv, pre.qvs, pre.nwfa, dt_arr)
        pni_iha_raw = xnc_iha * odts
        pri_iha_raw = jnp.minimum(rate_max, XM0I * jnp.float32(0.1) * pni_iha_raw)
        pni_iha_limited = pri_iha_raw / (XM0I * jnp.float32(0.1))
        pri_iha = jnp.where(homog_active, pri_iha_raw, zeros)
        pni_iha = jnp.where(homog_active, pni_iha_limited, zeros)
    else:
        pri_iha = zeros
        pni_iha = zeros

    return ThompsonIceNucleationResult(
        pri_inu=pri_inu,
        pni_inu=pni_inu,
        pri_iha=pri_iha,
        pni_iha=pni_iha,
    )


def _thompson_sublimation_prefactor(pre: ThompsonPreprocessResult) -> tuple[Any, Any]:
    otemp = _wrf_real_div(jnp.float32(1.0), pre.temp)
    rvs = _wrf_real_mul(pre.rho, pre.qvsi)
    lsub_otemp_orv = _wrf_real_mul(_wrf_real_mul(LSUB, otemp), ORV)
    lsub_term = _wrf_real_sub(lsub_otemp_orv, jnp.float32(1.0))
    rvs_p = _wrf_real_mul(_wrf_real_mul(rvs, otemp), lsub_term)
    rvs_pp_quad = _wrf_real_mul(_wrf_real_mul(_wrf_real_mul(otemp, lsub_term), otemp), lsub_term)
    rvs_pp_cubic = _wrf_real_mul(jnp.float32(-2.0), LSUB)
    rvs_pp_cubic = _wrf_real_mul(rvs_pp_cubic, otemp)
    rvs_pp_cubic = _wrf_real_mul(rvs_pp_cubic, otemp)
    rvs_pp_cubic = _wrf_real_mul(rvs_pp_cubic, otemp)
    rvs_pp_cubic = _wrf_real_mul(rvs_pp_cubic, ORV)
    rvs_pp_temp = _wrf_real_mul(otemp, otemp)
    rvs_pp = _wrf_real_mul(rvs, _wrf_real_add(_wrf_real_add(rvs_pp_quad, rvs_pp_cubic), rvs_pp_temp))
    gamsc = _wrf_real_mul(_wrf_real_div(_wrf_real_mul(LSUB, pre.diffu), pre.tcond), rvs_p)
    one_plus_gamsc = _wrf_real_add(jnp.float32(1.0), gamsc)
    gamsc_frac = _wrf_real_div(gamsc, one_plus_gamsc)
    alphsc = _wrf_real_mul(jnp.float32(0.5), gamsc_frac)
    alphsc = _wrf_real_mul(alphsc, gamsc_frac)
    alphsc = _wrf_real_mul(alphsc, _wrf_real_div(rvs_pp, rvs_p))
    alphsc = _wrf_real_mul(alphsc, _wrf_real_div(rvs, rvs_p))
    alphsc = jnp.maximum(jnp.float32(1.0e-9), alphsc)
    xsat = jnp.where(jnp.abs(pre.ssati) < jnp.float32(1.0e-9), jnp.zeros_like(pre.ssati), pre.ssati)
    xsat2 = _wrf_real_mul(xsat, xsat)
    xsat3 = _wrf_real_mul(xsat2, xsat)
    alphsc2 = _wrf_real_mul(alphsc, alphsc)
    alphsc3 = _wrf_real_mul(alphsc2, alphsc)
    poly = _wrf_real_sub(jnp.float32(1.0), _wrf_real_mul(alphsc, xsat))
    poly = _wrf_real_add(poly, _wrf_real_mul(_wrf_real_mul(jnp.float32(2.0), alphsc2), xsat2))
    poly = _wrf_real_sub(poly, _wrf_real_mul(_wrf_real_mul(jnp.float32(5.0), alphsc3), xsat3))
    t1_subl = _wrf_real_div(_wrf_real_mul(_wrf_real_mul(jnp.float32(4.0), _PI), poly), one_plus_gamsc)
    return rvs, t1_subl


def thompson_ice_deposition_terms(
    pre: ThompsonPreprocessResult,
    lookup: ThompsonFrozenLookupIndices,
    qi_aut_qs: ThompsonQiAutQsTables,
    dt: Any,
) -> ThompsonIceDepositionResult:
    """Compute WRF Thompson cloud-ice deposition and ice-to-snow conversion."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    rate_max = (pre.qv - pre.qvsi) * pre.rho * odts * jnp.float32(0.999)
    rvs, t1_subl = _thompson_sublimation_prefactor(pre)

    lami, xdi_raw = _ice_lami_xdi(pre.ni, pre.ri)
    ilami = _f64(1.0) / lami
    xdi = jnp.maximum(D0I, xdi_raw)
    xmi = AM_I * xdi**BM_I
    oxmi = jnp.float32(1.0) / xmi

    pri_prefix = _wrf_real_mul(C_CUBE, t1_subl)
    pri_prefix = _wrf_real_mul(pri_prefix, pre.diffu)
    pri_prefix = _wrf_real_mul(pri_prefix, pre.ssati)
    pri_prefix = _wrf_real_mul(pri_prefix, rvs)
    pri_prefix = _wrf_real_mul(pri_prefix, OIG1)
    pri_prefix = _wrf_real_mul(pri_prefix, CIG[4])
    pri_prefix = _wrf_real_mul(pri_prefix, pre.ni)
    pri_raw = _f64(pri_prefix) * ilami
    sublimating = pri_raw < _f64(0.0)
    pri_sub = jnp.maximum(jnp.maximum(_f64(-pre.ri * odts), pri_raw), _f64(rate_max))
    pni_sub = jnp.maximum(_f64(-pre.ni * odts), pri_sub * _f64(oxmi))

    tpi_ide = jnp.asarray(qi_aut_qs.tpi_ide, dtype=jnp.float32)[lookup.idx_i, lookup.idx_i1]
    pri_dep_limited = jnp.minimum(pri_raw, _f64(rate_max))
    prs_dep = (_f64(1.0) - _f64(tpi_ide)) * pri_dep_limited
    pri_dep = _f64(tpi_ide) * pri_dep_limited

    ice_active = pre.l_qi & (pre.temp < jnp.float32(273.15))
    pri_ide = jnp.where(ice_active, jnp.where(sublimating, pri_sub, pri_dep), zeros)
    pni_ide = jnp.where(ice_active & sublimating, pni_sub, zeros)
    prs_ide = jnp.where(ice_active & (~sublimating), prs_dep, zeros)

    tps_iaus = jnp.asarray(qi_aut_qs.tps_iaus, dtype=jnp.float32)[lookup.idx_i, lookup.idx_i1]
    tni_iaus = jnp.asarray(qi_aut_qs.tni_iaus, dtype=jnp.float32)[lookup.idx_i, lookup.idx_i1]
    large_ice = (lookup.idx_i == jnp.asarray(NTB_I - 1, dtype=jnp.int32)) | (xdi > jnp.float32(5.0) * D0S)
    small_ice = xdi < jnp.float32(0.1) * D0S
    prs_iau_table = jnp.minimum(pre.ri * jnp.float32(0.99) * odts, tps_iaus * odts)
    pni_iau_table = jnp.minimum(pre.ni * jnp.float32(0.95) * odts, tni_iaus * odts)
    prs_iau_active = jnp.where(large_ice, pre.ri * jnp.float32(0.99) * odts, jnp.where(small_ice, zeros, prs_iau_table))
    pni_iau_active = jnp.where(large_ice, pre.ni * jnp.float32(0.95) * odts, jnp.where(small_ice, zeros, pni_iau_table))
    prs_iau = jnp.where(ice_active, prs_iau_active, zeros)
    pni_iau = jnp.where(ice_active, pni_iau_active, zeros)

    return ThompsonIceDepositionResult(
        pri_ide=pri_ide,
        pni_ide=pni_ide,
        prs_ide=prs_ide,
        prs_iau=prs_iau,
        pni_iau=pni_iau,
        xdi=jnp.where(ice_active, xdi, zeros),
    )


def thompson_snow_graupel_deposition_terms(
    pre: ThompsonPreprocessResult,
    dt: Any,
) -> ThompsonSnowGraupelDepositionResult:
    """Compute WRF Thompson snow/graupel deposition and sublimation rates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    rate_max = (pre.qv - pre.qvsi) * pre.rho * odts * jnp.float32(0.999)
    rvs, t1_subl = _thompson_sublimation_prefactor(pre)

    tempc = pre.temp - jnp.float32(273.15)
    cold = pre.temp < jnp.float32(273.15)
    c_snow = C_SQRD + (tempc + jnp.float32(1.5)) * (C_CUBE - C_SQRD) / (jnp.float32(-30.0) + jnp.float32(1.5))
    c_snow = jnp.maximum(C_SQRD, jnp.minimum(c_snow, C_CUBE))
    t1_qs_sd = jnp.float32(0.86)
    t2_qs_sd = jnp.float32(0.28) * SC3 * sqrt_for_wrf(AV_S)
    prs_raw = c_snow * t1_subl * pre.diffu * pre.ssati * rvs * (t1_qs_sd * pre.smo1 + t2_qs_sd * pre.rhof2 * pre.vsc2 * pre.smof)
    prs_limited = jnp.where(
        prs_raw < jnp.float32(0.0),
        jnp.maximum(jnp.maximum(-pre.rs * odts, prs_raw), rate_max),
        jnp.minimum(prs_raw, rate_max),
    )
    prs_sde = jnp.where(pre.l_qs & cold, prs_limited, zeros)

    bg = pre.idx_bg
    av_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), AV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(AV_G, dtype=jnp.float32), bg))
    bv_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), BV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(BV_G, dtype=jnp.float32), bg))
    cge11 = jnp.float32(0.5) * (bv_g + jnp.float32(5.0) + jnp.float32(2.0) * MU_G)
    cgg11 = wgamma(cge11)
    t1_qg_sd = jnp.float32(0.86) * CGG[9, 0]
    t2_qg_sd = jnp.float32(0.28) * SC3 * sqrt_for_wrf(av_g) * cgg11
    prg_raw = (
        C_CUBE
        * t1_subl
        * pre.diffu
        * pre.ssati
        * rvs
        * pre.n0_g
        * (t1_qg_sd * pre.ilamg ** CGE[9, 0] + t2_qg_sd * pre.vsc2 * pre.rhof2 * pre.ilamg**cge11)
    )
    prg_limited = jnp.where(
        prg_raw < jnp.float32(0.0),
        jnp.maximum(jnp.maximum(-pre.rg * odts, prg_raw), rate_max),
        jnp.minimum(prg_raw, rate_max),
    )
    graupel_active = pre.l_qg & cold & (pre.ssati < -EPS)
    prg_gde = jnp.where(graupel_active, prg_limited, zeros)
    png_gde = jnp.where(graupel_active & (prg_raw < jnp.float32(0.0)), prg_gde * pre.ng / pre.rg, zeros)

    return ThompsonSnowGraupelDepositionResult(
        prs_sde=prs_sde,
        prg_gde=prg_gde,
        png_gde=png_gde,
        c_snow=jnp.where(pre.l_qs & cold, c_snow, zeros),
    )


def thompson_ice_collection_terms(
    pre: ThompsonPreprocessResult,
    dt: Any,
) -> ThompsonIceCollectionResult:
    """Compute WRF Thompson snow/rain collection of cloud ice."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(pre.temp)
    r_s = jnp.asarray(thompson_snow_table_mixing_ratios(), dtype=jnp.float32)
    r_r = jnp.asarray(thompson_rain_table_mixing_ratios(), dtype=jnp.float32)

    lami = (AM_I * CIG[1] * OIG1 * pre.ni / pre.ri) ** OBMI
    ilami = jnp.float32(1.0) / lami
    xdi = jnp.maximum(D0I, (BM_I + MU_I + jnp.float32(1.0)) * ilami)
    xmi = AM_I * xdi**BM_I
    oxmi = jnp.float32(1.0) / xmi

    ice_active = pre.l_qi & (pre.temp < jnp.float32(273.15))
    snow_active = ice_active & (pre.rs >= r_s[0])
    t1_qs_qi = _PI * jnp.float32(0.25) * AV_S
    prs_sci_raw = t1_qs_qi * pre.rhof * EF_SI * pre.ri * pre.smoe
    prs_sci = jnp.where(snow_active, prs_sci_raw, zeros)
    pni_sci = jnp.where(snow_active, prs_sci_raw * oxmi, zeros)

    rain_active = ice_active & (pre.rr >= r_r[0]) & (pre.mvd_r > jnp.float32(4.0) * xdi)
    lamr = jnp.float32(1.0) / pre.ilamr
    t1_qr_qi = _PI * jnp.float32(0.25) * AV_R * CRG[8]
    t2_qr_qi = _PI * jnp.float32(0.25) * AM_R * AV_R * CRG[7]
    rain_kernel = pre.rhof * t1_qr_qi * EF_RI * pre.n0_r * ((lamr + FV_R) ** (-CRE[8]))
    pri_rci_raw = rain_kernel * pre.ri
    pnr_rci_raw = jnp.minimum(pre.nr * odts, rain_kernel * pre.ni)
    pni_rci_raw = pri_rci_raw * oxmi
    prr_rci_raw = pre.rhof * t2_qr_qi * EF_RI * pre.ni * pre.n0_r * ((lamr + FV_R) ** (-CRE[7]))
    prr_rci_raw = jnp.minimum(pre.rr * odts, prr_rci_raw)
    prg_rci_raw = pri_rci_raw + prr_rci_raw
    pbg_rci_raw = prg_rci_raw / RHO_I

    pri_rci = jnp.where(rain_active, pri_rci_raw, zeros)
    pni_rci = jnp.where(rain_active, pni_rci_raw, zeros)
    pnr_rci = jnp.where(rain_active, pnr_rci_raw, zeros)
    prr_rci = jnp.where(rain_active, prr_rci_raw, zeros)
    prg_rci = jnp.where(rain_active, prg_rci_raw, zeros)
    pbg_rci = jnp.where(rain_active, pbg_rci_raw, zeros)

    return ThompsonIceCollectionResult(
        prs_sci=prs_sci,
        pni_sci=pni_sci,
        pri_rci=pri_rci,
        pni_rci=pni_rci,
        pnr_rci=pnr_rci,
        prr_rci=prr_rci,
        prg_rci=prg_rci,
        pbg_rci=pbg_rci,
        xdi=jnp.where(ice_active, xdi, zeros),
    )


def thompson_riming_terms(
    pre: ThompsonPreprocessResult,
    lookup: ThompsonFrozenLookupIndices,
    prs_scw: Any,
    prg_gcw: Any,
    prs_sde: Any,
    mvd_c: Any,
) -> ThompsonRimingResult:
    """Compute WRF Thompson rime-splinter and rimed-snow conversion terms."""

    prs_scw_arr = jnp.asarray(prs_scw, dtype=jnp.float32)
    prg_gcw_arr = jnp.asarray(prg_gcw, dtype=jnp.float32)
    prs_sde_arr = jnp.asarray(prs_sde, dtype=jnp.float32)
    mvd_c_arr = jnp.asarray(mvd_c, dtype=jnp.float32)
    zeros = jnp.zeros_like(pre.temp)
    ones = jnp.ones_like(pre.temp)
    tempc = pre.temp - jnp.float32(273.15)

    tf_mid = jnp.where(
        (tempc >= jnp.float32(-5.0)) & (tempc < jnp.float32(-3.0)),
        jnp.float32(0.5) * (jnp.float32(-3.0) - tempc),
        jnp.where((tempc > jnp.float32(-8.0)) & (tempc < jnp.float32(-5.0)), jnp.float32(0.33333333) * (jnp.float32(8.0) + tempc), zeros),
    )
    ihm_active = (prg_gcw_arr > EPS) & (tempc > jnp.float32(-8.0))
    pni_ihm = jnp.where(ihm_active, jnp.float32(3.5e8) * tf_mid * prg_gcw_arr, zeros)
    pri_ihm = XM0I * pni_ihm
    ihm_denom = jnp.where(ihm_active, prs_scw_arr + prg_gcw_arr, ones)
    prs_ihm = jnp.where(ihm_active, prs_scw_arr / ihm_denom * pri_ihm, zeros)
    prg_ihm = jnp.where(ihm_active, prg_gcw_arr / ihm_denom * pri_ihm, zeros)

    convert_active = (prs_scw_arr > jnp.float32(2.0) * prs_sde_arr) & (prs_sde_arr > EPS)
    r_frac = jnp.minimum(jnp.float32(30.0), prs_scw_arr / jnp.where(convert_active, prs_sde_arr, ones))
    g_frac_raw = jnp.minimum(jnp.float32(0.95), jnp.float32(0.15) + (r_frac - jnp.float32(2.0)) * jnp.float32(0.028))
    vts_boost_raw = jnp.minimum(jnp.float32(1.5), jnp.float32(1.1) + (r_frac - jnp.float32(2.0)) * jnp.float32(0.014))
    prg_scw_raw = g_frac_raw * prs_scw_arr
    png_scw_raw = prg_scw_raw * pre.smo0 / pre.rs
    vts = AV_S * _clipped_fractional_power(lookup.xds, BV_S) * jnp.exp(-FV_S * lookup.xds)
    const_ri = -jnp.float32(1.0) * (mvd_c_arr * jnp.float32(0.5e6)) * vts / jnp.minimum(jnp.float32(-0.1), tempc)
    const_ri = jnp.maximum(jnp.float32(0.1), jnp.minimum(const_ri, jnp.float32(10.0)))
    rime_dens_raw = (jnp.float32(0.051) + jnp.float32(0.114) * const_ri - jnp.float32(0.0055) * const_ri * const_ri) * jnp.float32(1000.0)
    dense_enough = rime_dens_raw >= jnp.float32(150.0)
    g_frac = jnp.where(convert_active & dense_enough, g_frac_raw, zeros)
    prg_scw = jnp.where(convert_active & dense_enough, prg_scw_raw, zeros)
    png_scw = jnp.where(convert_active & dense_enough, png_scw_raw, zeros)
    pbg_scw = jnp.where(convert_active & dense_enough, prg_scw / (jnp.float32(0.5) * (rime_dens_raw + lookup.rho_s)), zeros)
    prs_scw_out = jnp.where(convert_active, (jnp.float32(1.0) - g_frac) * prs_scw_arr, prs_scw_arr)
    vts_boost = jnp.where(convert_active, vts_boost_raw, ones)
    rime_dens = jnp.where(convert_active, rime_dens_raw, zeros)

    return ThompsonRimingResult(
        pri_ihm=pri_ihm,
        pni_ihm=pni_ihm,
        prs_ihm=prs_ihm,
        prg_ihm=prg_ihm,
        prs_scw=prs_scw_out,
        prg_scw=prg_scw,
        png_scw=png_scw,
        pbg_scw=pbg_scw,
        vts_boost=vts_boost,
        g_frac=g_frac,
        rime_dens=rime_dens,
    )


def thompson_melting_terms(
    pre: ThompsonPreprocessResult,
    dt: Any,
    prr_rcs: Any,
    prs_scw: Any,
    prg_gcw: Any,
    prr_rcw: Any,
) -> ThompsonMeltingResult:
    """Compute WRF Thompson warm-layer snow/graupel melting terms."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    prr_rcs_arr = jnp.asarray(prr_rcs, dtype=jnp.float32)
    prs_scw_arr = jnp.asarray(prs_scw, dtype=jnp.float32)
    prg_gcw_arr = jnp.asarray(prg_gcw, dtype=jnp.float32)
    prr_rcw_arr = jnp.asarray(prr_rcw, dtype=jnp.float32)
    zeros = jnp.zeros_like(pre.temp)
    warm = pre.temp >= jnp.float32(273.15)
    tempc = pre.temp - jnp.float32(273.15)
    rvs, t1_subl = _thompson_sublimation_prefactor(pre)

    t1_qs_me = _PI * jnp.float32(4.0) * C_SQRD * OLFUS * jnp.float32(0.86)
    t2_qs_me = _PI * jnp.float32(4.0) * C_SQRD * OLFUS * jnp.float32(0.28) * SC3 * sqrt_for_wrf(AV_S)
    t1_qs_sd = jnp.float32(0.86)
    t2_qs_sd = jnp.float32(0.28) * SC3 * sqrt_for_wrf(AV_S)
    melt_base = tempc * pre.tcond - LVAP0 * pre.diffu * pre.delqvs

    prr_sml_raw = melt_base * (t1_qs_me * pre.smo1 + t2_qs_me * pre.rhof2 * pre.vsc2 * pre.smof)
    prr_sml_enhanced = jnp.where(
        prr_sml_raw > jnp.float32(0.0),
        prr_sml_raw + jnp.float32(4218.0) * OLFUS * (pre.twet - jnp.float32(273.15)) * (prr_rcs_arr + prs_scw_arr),
        prr_sml_raw,
    )
    prr_sml_limited = jnp.minimum(pre.rs * odts, jnp.maximum(zeros, prr_sml_enhanced))
    snow_active = warm & pre.l_qs
    prr_sml = jnp.where(snow_active, prr_sml_limited, zeros)
    pnr_sml = jnp.where(
        snow_active & (prr_sml_limited > jnp.float32(0.0)),
        pre.smo0 / pre.rs * prr_sml_limited * (jnp.float32(10.0) ** (-jnp.float32(0.25) * (pre.twet - jnp.float32(273.15)))),
        zeros,
    )
    prs_sde_raw = C_SQRD * t1_subl * pre.diffu * pre.ssati * rvs * (t1_qs_sd * pre.smo1 + t2_qs_sd * pre.rhof2 * pre.vsc2 * pre.smof)
    prs_sde = jnp.where(
        snow_active & (prr_sml_limited <= jnp.float32(0.0)) & (pre.ssati < jnp.float32(0.0)),
        jnp.maximum(-pre.rs * odts, prs_sde_raw),
        zeros,
    )

    bg = pre.idx_bg
    rho_g = jnp.take(jnp.asarray(RHO_G, dtype=jnp.float32), bg)
    av_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), AV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(AV_G, dtype=jnp.float32), bg))
    bv_g = jnp.where(bg == jnp.asarray(IDX_BG1, dtype=jnp.int32), BV_G_IDX_BG1_DEFAULT, jnp.take(jnp.asarray(BV_G, dtype=jnp.float32), bg))
    cge11 = jnp.float32(0.5) * (bv_g + jnp.float32(5.0) + jnp.float32(2.0) * MU_G)
    cgg11 = wgamma(cge11)
    lamg = jnp.float32(1.0) / pre.ilamg
    n0_melt_low = (jnp.float32(1.0e-4) / pre.rg) * OGG2 * lamg ** CGE[1, 0]
    n0_melt = jnp.where(pre.rg * pre.ng < jnp.float32(1.0e-4), n0_melt_low, pre.n0_g)
    t1_qg_me = _PI * jnp.float32(4.0) * C_CUBE * OLFUS * jnp.float32(0.86) * CGG[9, 0]
    t2_qg_me = _PI * jnp.float32(4.0) * C_CUBE * OLFUS * jnp.float32(0.28) * SC3 * sqrt_for_wrf(av_g) * cgg11
    t1_qg_sd = jnp.float32(0.86) * CGG[9, 0]
    t2_qg_sd = jnp.float32(0.28) * SC3 * sqrt_for_wrf(av_g) * cgg11

    prr_gml_raw = melt_base * n0_melt * (t1_qg_me * pre.ilamg ** CGE[9, 0] + t2_qg_me * pre.rhof2 * pre.vsc2 * pre.ilamg**cge11)
    prr_gml_limited = jnp.minimum(pre.rg * odts, jnp.maximum(zeros, prr_gml_raw))
    graupel_active = warm & pre.l_qg
    prr_gml = jnp.where(graupel_active, prr_gml_limited, zeros)
    melt_f = jnp.maximum(jnp.float32(0.05), jnp.minimum(prr_gml_limited * dt_arr / pre.rg, jnp.float32(1.0)))
    pbg_gml_raw = prr_gml_limited / jnp.maximum(jnp.minimum(melt_f * rho_g, jnp.float32(1000.0)), jnp.float32(50.0))
    pnr_gml_raw = prr_gml_limited * pre.ng / pre.rg * (jnp.float32(10.0) ** (-jnp.float32(0.33) * (pre.twet - jnp.float32(273.15))))
    melting_graupel = graupel_active & (prr_gml_limited > jnp.float32(0.0))
    pbg_gml = jnp.where(melting_graupel, pbg_gml_raw, zeros)
    pnr_gml = jnp.where(melting_graupel, pnr_gml_raw, zeros)
    prg_gde_raw = C_CUBE * t1_subl * pre.diffu * pre.ssati * rvs * pre.n0_g * (
        t1_qg_sd * pre.ilamg ** CGE[9, 0] + t2_qg_sd * pre.vsc2 * pre.rhof2 * pre.ilamg**cge11
    )
    prg_gde = jnp.where(
        graupel_active & (prr_gml_limited <= jnp.float32(0.0)) & (pre.ssati < jnp.float32(0.0)),
        jnp.maximum(-pre.rg * odts, prg_gde_raw),
        zeros,
    )
    png_gde = jnp.where(prg_gde != jnp.float32(0.0), prg_gde * pre.ng / pre.rg, zeros)

    adjust_cloud_collection = warm & (dt_arr > jnp.float32(120.0))
    prr_rcw_out = jnp.where(adjust_cloud_collection, prr_rcw_arr + prs_scw_arr + prg_gcw_arr, prr_rcw_arr)
    prs_scw_out = jnp.where(adjust_cloud_collection, zeros, prs_scw_arr)
    prg_gcw_out = jnp.where(adjust_cloud_collection, zeros, prg_gcw_arr)

    return ThompsonMeltingResult(
        prr_sml=prr_sml,
        pnr_sml=pnr_sml,
        prs_sde=prs_sde,
        prr_gml=prr_gml,
        pnr_gml=pnr_gml,
        pbg_gml=pbg_gml,
        prg_gde=prg_gde,
        png_gde=png_gde,
        prr_rcw=prr_rcw_out,
        prs_scw=prs_scw_out,
        prg_gcw=prg_gcw_out,
        n0_melt=jnp.where(graupel_active, n0_melt, zeros),
    )


def _fortran_sign_unit(value: Any) -> Any:
    """Return WRF/Fortran ``SIGN(1.0, value)`` for array values."""

    arr = jnp.asarray(value, dtype=jnp.float32)
    return jnp.where(arr < jnp.float32(0.0), jnp.float32(-1.0), jnp.float32(1.0))


def _depletion_ratio(condition: Any, rate_max: Any, sump: Any) -> Any:
    cond = jnp.asarray(condition, dtype=bool)
    safe_sump = jnp.where(cond, sump, jnp.ones_like(sump))
    return jnp.where(cond, rate_max / safe_sump, jnp.ones_like(sump))


def thompson_depletion_rescale(
    pre: ThompsonPreprocessResult,
    dt: Any,
    *,
    pri_inu: Any,
    pri_ide: Any,
    pni_ide: Any,
    prs_ide: Any,
    prs_sde: Any,
    prg_gde: Any,
    pri_iha: Any,
    prr_wau: Any,
    pri_wfz: Any,
    prr_rcw: Any,
    prs_scw: Any,
    prg_scw: Any,
    prg_gcw: Any,
    prs_iau: Any,
    prs_sci: Any,
    pri_rci: Any,
    prg_rfz: Any,
    pri_rfz: Any,
    prr_rci: Any,
    prr_rcs: Any,
    prr_rcg: Any,
    prs_ihm: Any,
    prr_sml: Any,
    prs_rcs: Any,
    prg_ihm: Any,
    prr_gml: Any,
    prg_rcg: Any,
) -> ThompsonDepletionRescaleResult:
    """Apply WRF Thompson depletion limits to process rates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    pri_inu = jnp.asarray(pri_inu, dtype=jnp.float32)
    pri_ide = jnp.asarray(pri_ide, dtype=jnp.float32)
    pni_ide = jnp.asarray(pni_ide, dtype=jnp.float32)
    prs_ide = jnp.asarray(prs_ide, dtype=jnp.float32)
    prs_sde = jnp.asarray(prs_sde, dtype=jnp.float32)
    prg_gde = jnp.asarray(prg_gde, dtype=jnp.float32)
    pri_iha = jnp.asarray(pri_iha, dtype=jnp.float32)
    prr_wau = jnp.asarray(prr_wau, dtype=jnp.float32)
    pri_wfz = jnp.asarray(pri_wfz, dtype=jnp.float32)
    prr_rcw = jnp.asarray(prr_rcw, dtype=jnp.float32)
    prs_scw = jnp.asarray(prs_scw, dtype=jnp.float32)
    prg_scw = jnp.asarray(prg_scw, dtype=jnp.float32)
    prg_gcw = jnp.asarray(prg_gcw, dtype=jnp.float32)
    prs_iau = jnp.asarray(prs_iau, dtype=jnp.float32)
    prs_sci = jnp.asarray(prs_sci, dtype=jnp.float32)
    pri_rci = jnp.asarray(pri_rci, dtype=jnp.float32)
    prg_rfz = jnp.asarray(prg_rfz, dtype=jnp.float32)
    pri_rfz = jnp.asarray(pri_rfz, dtype=jnp.float32)
    prr_rci = jnp.asarray(prr_rci, dtype=jnp.float32)
    prr_rcs = jnp.asarray(prr_rcs, dtype=jnp.float32)
    prr_rcg = jnp.asarray(prr_rcg, dtype=jnp.float32)
    prs_ihm = jnp.asarray(prs_ihm, dtype=jnp.float32)
    prr_sml = jnp.asarray(prr_sml, dtype=jnp.float32)
    prs_rcs = jnp.asarray(prs_rcs, dtype=jnp.float32)
    prg_ihm = jnp.asarray(prg_ihm, dtype=jnp.float32)
    prr_gml = jnp.asarray(prr_gml, dtype=jnp.float32)
    prg_rcg = jnp.asarray(prg_rcg, dtype=jnp.float32)

    sump = pri_inu + pri_ide + prs_ide + prs_sde + prg_gde + pri_iha
    rate_max = (pre.qv - pre.qvsi) * pre.rho * odts * jnp.float32(0.999)
    ratio = _depletion_ratio(((sump > EPS) & (sump > rate_max)) | ((sump < -EPS) & (sump < rate_max)), rate_max, sump)
    pri_inu = pri_inu * ratio
    pri_ide = pri_ide * ratio
    pni_ide = pni_ide * ratio
    prs_ide = prs_ide * ratio
    prs_sde = prs_sde * ratio
    prg_gde = prg_gde * ratio
    pri_iha = pri_iha * ratio

    sump = -prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw - prg_gcw
    rate_max = -pre.rc * odts
    ratio = _depletion_ratio((sump < rate_max) & pre.l_qc, rate_max, sump)
    prr_wau = prr_wau * ratio
    pri_wfz = pri_wfz * ratio
    prr_rcw = prr_rcw * ratio
    prs_scw = prs_scw * ratio
    prg_scw = prg_scw * ratio
    prg_gcw = prg_gcw * ratio

    sump = pri_ide - prs_iau - prs_sci - pri_rci
    rate_max = -pre.ri * odts
    ratio = _depletion_ratio((sump < rate_max) & pre.l_qi, rate_max, sump)
    pri_ide = pri_ide * ratio
    prs_iau = prs_iau * ratio
    prs_sci = prs_sci * ratio
    pri_rci = pri_rci * ratio

    sump = -prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg
    rate_max = -pre.rr * odts
    ratio = _depletion_ratio((sump < rate_max) & pre.l_qr, rate_max, sump)
    prg_rfz = prg_rfz * ratio
    pri_rfz = pri_rfz * ratio
    prr_rci = prr_rci * ratio
    prr_rcs = prr_rcs * ratio
    prr_rcg = prr_rcg * ratio

    sump = prs_sde - prs_ihm - prr_sml + prs_rcs
    rate_max = -pre.rs * odts
    ratio = _depletion_ratio((sump < rate_max) & pre.l_qs, rate_max, sump)
    prs_sde = prs_sde * ratio
    prs_ihm = prs_ihm * ratio
    prr_sml = prr_sml * ratio
    prs_rcs = prs_rcs * ratio

    sump = prg_gde - prg_ihm - prr_gml + prg_rcg
    rate_max = -pre.rg * odts
    ratio = _depletion_ratio((sump < rate_max) & pre.l_qg, rate_max, sump)
    prg_gde = prg_gde * ratio
    prg_ihm = prg_ihm * ratio
    prr_gml = prr_gml * ratio
    prg_rcg = prg_rcg * ratio

    pri_ihm = prs_ihm + prg_ihm
    paired = jnp.minimum(jnp.abs(prr_rcg), jnp.abs(prg_rcg))
    prr_rcg = paired * _fortran_sign_unit(prr_rcg)
    prg_rcg = -prr_rcg
    warm_twet = pre.twet > jnp.float32(273.15)
    paired = jnp.minimum(jnp.abs(prr_rcs), jnp.abs(prs_rcs))
    prr_rcs_balanced = paired * _fortran_sign_unit(prr_rcs)
    prs_rcs_balanced = -prr_rcs_balanced
    prr_rcs = jnp.where(warm_twet, prr_rcs_balanced, prr_rcs)
    prs_rcs = jnp.where(warm_twet, prs_rcs_balanced, prs_rcs)

    return ThompsonDepletionRescaleResult(
        pri_inu=pri_inu,
        pri_ide=pri_ide,
        pni_ide=pni_ide,
        prs_ide=prs_ide,
        prs_sde=prs_sde,
        prg_gde=prg_gde,
        pri_iha=pri_iha,
        prr_wau=prr_wau,
        pri_wfz=pri_wfz,
        prr_rcw=prr_rcw,
        prs_scw=prs_scw,
        prg_scw=prg_scw,
        prg_gcw=prg_gcw,
        prs_iau=prs_iau,
        prs_sci=prs_sci,
        pri_rci=pri_rci,
        prg_rfz=prg_rfz,
        pri_rfz=pri_rfz,
        prr_rci=prr_rci,
        prr_rcs=prr_rcs,
        prr_rcg=prr_rcg,
        prs_ihm=prs_ihm,
        prr_sml=prr_sml,
        prs_rcs=prs_rcs,
        prg_ihm=prg_ihm,
        prr_gml=prr_gml,
        prg_rcg=prg_rcg,
        pri_ihm=pri_ihm,
    )


def _optional_rate(value: Any, reference: Any) -> Any:
    if value is None:
        return jnp.zeros_like(reference)
    return jnp.asarray(value, dtype=jnp.float32)


def thompson_tendency_accumulation(
    pre: ThompsonPreprocessResult,
    dt: Any,
    depleted: ThompsonDepletionRescaleResult,
    *,
    nwfaten: Any | None = None,
    nifaten: Any | None = None,
    nbcaten: Any | None = None,
    qvten: Any | None = None,
    qcten: Any | None = None,
    ncten: Any | None = None,
    qiten: Any | None = None,
    niten: Any | None = None,
    qrten: Any | None = None,
    nrten: Any | None = None,
    qsten: Any | None = None,
    qgten: Any | None = None,
    ngten: Any | None = None,
    qbten: Any | None = None,
    tten: Any | None = None,
    pna_rca: Any | None = None,
    pna_sca: Any | None = None,
    pna_gca: Any | None = None,
    pnd_rcd: Any | None = None,
    pnd_scd: Any | None = None,
    pnd_gcd: Any | None = None,
    pnb_rcb: Any | None = None,
    pnb_scb: Any | None = None,
    pnb_gcb: Any | None = None,
    pni_inu: Any | None = None,
    pni_iha: Any | None = None,
    pni_ihm: Any | None = None,
    pnc_wau: Any | None = None,
    pnc_rcw: Any | None = None,
    pni_wfz: Any | None = None,
    pnc_scw: Any | None = None,
    pnc_gcw: Any | None = None,
    pni_rfz: Any | None = None,
    pni_iau: Any | None = None,
    pni_sci: Any | None = None,
    pni_rci: Any | None = None,
    pnr_wau: Any | None = None,
    pnr_sml: Any | None = None,
    pnr_gml: Any | None = None,
    pnr_rfz: Any | None = None,
    pnr_rcr: Any | None = None,
    pnr_rcg: Any | None = None,
    pnr_rcs: Any | None = None,
    pnr_rci: Any | None = None,
    prg_rci: Any | None = None,
    prg_rcs: Any | None = None,
    png_scw: Any | None = None,
    png_rcg: Any | None = None,
    png_rcs: Any | None = None,
    png_gde: Any | None = None,
    pbg_scw: Any | None = None,
    pbg_rfz: Any | None = None,
    pbg_gcw: Any | None = None,
    pbg_rci: Any | None = None,
    pbg_rcs: Any | None = None,
    pbg_rcg: Any | None = None,
    pbg_sml: Any | None = None,
    pbg_gml: Any | None = None,
    is_aerosol_aware: bool = False,
    wif_input_opt: int = 1,
    dusty_ice: bool = False,
    ifdry: int = 0,
) -> ThompsonTendencyResult:
    """Accumulate WRF Thompson process rates into prognostic tendencies."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    ref = jnp.asarray(pre.temp, dtype=jnp.float32)
    orho = jnp.float32(1.0) / pre.rho

    nwfaten = _optional_rate(nwfaten, ref)
    nifaten = _optional_rate(nifaten, ref)
    nbcaten = _optional_rate(nbcaten, ref)
    qvten = _optional_rate(qvten, ref)
    qcten = _optional_rate(qcten, ref)
    ncten = _optional_rate(ncten, ref)
    qiten = _optional_rate(qiten, ref)
    niten = _optional_rate(niten, ref)
    qrten = _optional_rate(qrten, ref)
    nrten = _optional_rate(nrten, ref)
    qsten = _optional_rate(qsten, ref)
    qgten = _optional_rate(qgten, ref)
    ngten = _optional_rate(ngten, ref)
    qbten = _optional_rate(qbten, ref)
    tten = _optional_rate(tten, ref)

    pna_rca = _optional_rate(pna_rca, ref)
    pna_sca = _optional_rate(pna_sca, ref)
    pna_gca = _optional_rate(pna_gca, ref)
    pnd_rcd = _optional_rate(pnd_rcd, ref)
    pnd_scd = _optional_rate(pnd_scd, ref)
    pnd_gcd = _optional_rate(pnd_gcd, ref)
    pnb_rcb = _optional_rate(pnb_rcb, ref)
    pnb_scb = _optional_rate(pnb_scb, ref)
    pnb_gcb = _optional_rate(pnb_gcb, ref)
    pni_inu = _optional_rate(pni_inu, ref)
    pni_iha = _optional_rate(pni_iha, ref)
    pni_ihm = _optional_rate(pni_ihm, ref)
    pnc_wau = _optional_rate(pnc_wau, ref)
    pnc_rcw = _optional_rate(pnc_rcw, ref)
    pni_wfz = _optional_rate(pni_wfz, ref)
    pnc_scw = _optional_rate(pnc_scw, ref)
    pnc_gcw = _optional_rate(pnc_gcw, ref)
    pni_rfz = _optional_rate(pni_rfz, ref)
    pni_iau = _optional_rate(pni_iau, ref)
    pni_sci = _optional_rate(pni_sci, ref)
    pni_rci = _optional_rate(pni_rci, ref)
    pnr_wau = _optional_rate(pnr_wau, ref)
    pnr_sml = _optional_rate(pnr_sml, ref)
    pnr_gml = _optional_rate(pnr_gml, ref)
    pnr_rfz = _optional_rate(pnr_rfz, ref)
    pnr_rcr = _optional_rate(pnr_rcr, ref)
    pnr_rcg = _optional_rate(pnr_rcg, ref)
    pnr_rcs = _optional_rate(pnr_rcs, ref)
    pnr_rci = _optional_rate(pnr_rci, ref)
    prg_rci = _optional_rate(prg_rci, ref)
    prg_rcs = _optional_rate(prg_rcs, ref)
    png_scw = _optional_rate(png_scw, ref)
    png_rcg = _optional_rate(png_rcg, ref)
    png_rcs = _optional_rate(png_rcs, ref)
    png_gde = _optional_rate(png_gde, ref)
    pbg_scw = _optional_rate(pbg_scw, ref)
    pbg_rfz = _optional_rate(pbg_rfz, ref)
    pbg_gcw = _optional_rate(pbg_gcw, ref)
    pbg_rci = _optional_rate(pbg_rci, ref)
    pbg_rcs = _optional_rate(pbg_rcs, ref)
    pbg_rcg = _optional_rate(pbg_rcg, ref)
    pbg_sml = _optional_rate(pbg_sml, ref)
    pbg_gml = _optional_rate(pbg_gml, ref)

    if is_aerosol_aware:
        nwfaten = nwfaten - (pna_rca + pna_sca + pna_gca + pni_iha) * orho
        nifaten = nifaten - (pnd_rcd + pnd_scd + pnd_gcd) * orho
        if int(wif_input_opt) == 2:
            nbcaten = nbcaten - (pnb_rcb + pnb_scb + pnb_gcb) * orho
        else:
            nbcaten = jnp.zeros_like(nbcaten)
        if dusty_ice:
            nifaten = nifaten - pni_inu * orho
        else:
            nifaten = jnp.zeros_like(nifaten)

    qvten = qvten + (-depleted.pri_inu - depleted.pri_iha - depleted.pri_ide - depleted.prs_ide - depleted.prs_sde - depleted.prg_gde) * orho

    qcten = qcten + (-depleted.prr_wau - depleted.pri_wfz - depleted.prr_rcw - depleted.prs_scw - depleted.prg_scw - depleted.prg_gcw) * orho
    ncten = ncten + (-pnc_wau - pnc_rcw - pni_wfz - pnc_scw - pnc_gcw) * orho

    xrc = jnp.maximum(R1, (pre.qc1d + qcten * dt_arr) * pre.rho)
    xnc = jnp.maximum(jnp.float32(2.0), (pre.nc1d + ncten * dt_arr) * pre.rho)
    nu_c = jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / xnc) + jnp.asarray(2, dtype=jnp.int32))
    nu_idx = nu_c - jnp.asarray(1, dtype=jnp.int32)
    lamc = (xnc * AM_R * CCG[1, nu_idx] * OCG1[nu_idx] / pre.rc) ** OBMR
    xdc = (BM_R + nu_c.astype(jnp.float32) + jnp.float32(1.0)) / lamc
    ncten_low = (CCG[0, nu_idx] * OCG2[nu_idx] * xrc / AM_R * (CCE[1, nu_idx] / D0C) ** BM_R - pre.nc1d * pre.rho) * odts * orho
    ncten_high = (
        CCG[0, nu_idx] * OCG2[nu_idx] * xrc / AM_R * (CCE[1, nu_idx] / (D0R * jnp.float32(2.0))) ** BM_R - pre.nc1d * pre.rho
    ) * odts * orho
    ncten = jnp.where(xrc > R1, jnp.where(xdc < D0C, ncten_low, jnp.where(xdc > D0R * jnp.float32(2.0), ncten_high, ncten)), -pre.nc1d * odts)
    xnc_after = jnp.maximum(jnp.float32(0.0), (pre.nc1d + ncten * dt_arr) * pre.rho)
    ncten = jnp.where(xnc_after > NT_C_MAX, (NT_C_MAX - pre.nc1d * pre.rho) * odts * orho, ncten)

    qiten = qiten + (
        depleted.pri_inu
        + depleted.pri_iha
        + depleted.pri_ihm
        + depleted.pri_wfz
        + depleted.pri_rfz
        + depleted.pri_ide
        - depleted.prs_iau
        - depleted.prs_sci
        - depleted.pri_rci
    ) * orho
    niten = niten + (pni_inu + pni_iha + pni_ihm + pni_wfz + pni_rfz + depleted.pni_ide - pni_iau - pni_sci - pni_rci) * orho

    xri = jnp.maximum(R1, (pre.qi1d + qiten * dt_arr) * pre.rho)
    xni = jnp.maximum(R2, (pre.ni1d + niten * dt_arr) * pre.rho)
    _, xdi = _ice_lami_xdi(xni, xri)
    niten_small = (_ice_number_from_lami(xri, _ice_lami_for_diameter(5.0e-6), cap=999.0e3) - pre.ni1d * pre.rho) * odts * orho
    niten_large = (_ice_number_from_lami(xri, _ice_lami_for_diameter(300.0e-6)) - pre.ni1d * pre.rho) * odts * orho
    niten = jnp.where(
        xri > R1,
        jnp.where(xdi < jnp.float32(5.0e-6), niten_small, jnp.where(xdi > jnp.float32(300.0e-6), niten_large, niten)),
        -pre.ni1d * odts,
    )
    xni_after = jnp.maximum(jnp.float32(0.0), (pre.ni1d + niten * dt_arr) * pre.rho)
    niten = jnp.where(xni_after > jnp.float32(999.0e3), (jnp.float32(999.0e3) - pre.ni1d * pre.rho) * odts * orho, niten)

    qrten = qrten + (
        depleted.prr_wau
        + depleted.prr_rcw
        + depleted.prr_sml
        + depleted.prr_gml
        + depleted.prr_rcs
        + depleted.prr_rcg
        - depleted.prg_rfz
        - depleted.pri_rfz
        - depleted.prr_rci
    ) * orho
    nrten = nrten + (pnr_wau + pnr_sml + pnr_gml - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs + pnr_rci + pni_rfz)) * orho

    xrr = jnp.maximum(R1, (pre.qr1d + qrten * dt_arr) * pre.rho)
    xnr = jnp.maximum(R2, (pre.nr1d + nrten * dt_arr) * pre.rho)
    lamr = (AM_R * CRG[2] * ORG2 * xnr / xrr) ** OBMR
    mvd_r = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr
    nrten_high = (CRG[1] * ORG3 * xrr * ((jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / jnp.float32(2.5e-3)) ** BM_R / AM_R - pre.nr1d * pre.rho) * odts * orho
    nrten_low = (
        CRG[1] * ORG3 * xrr * ((jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / (D0R * jnp.float32(0.75))) ** BM_R / AM_R - pre.nr1d * pre.rho
    ) * odts * orho
    nrten = jnp.where(xrr > R1, jnp.where(mvd_r > jnp.float32(2.5e-3), nrten_high, jnp.where(mvd_r < D0R * jnp.float32(0.75), nrten_low, nrten)), -pre.nr1d * odts)
    qrten = jnp.where(xrr > R1, qrten, -pre.qr1d * odts)
    mvd_r = jnp.where(xrr > R1, jnp.minimum(jnp.maximum(mvd_r, D0R * jnp.float32(0.75)), jnp.float32(2.5e-3)), pre.mvd_r)

    qsten = qsten + (
        depleted.prs_iau + depleted.prs_sde + depleted.prs_sci + depleted.prs_scw + depleted.prs_rcs + depleted.prs_ide - depleted.prs_ihm - depleted.prr_sml
    ) * orho

    qgten = qgten + (
        depleted.prg_scw
        + depleted.prg_rfz
        + depleted.prg_gde
        + depleted.prg_rcg
        + depleted.prg_gcw
        + prg_rci
        + prg_rcs
        - depleted.prg_ihm
        - depleted.prr_gml
    ) * orho
    ngten = ngten + (png_scw + pnr_rfz - png_rcg + pnr_rci + png_rcs + png_gde - pnr_gml) * orho
    rho_g = jnp.take(jnp.asarray(RHO_G, dtype=jnp.float32), pre.idx_bg)
    qbten = qbten + (pbg_scw + pbg_rfz + pbg_gcw + pbg_rci + pbg_rcs + pbg_rcg + pbg_sml - pbg_gml + (depleted.prg_gde - depleted.prg_ihm) / rho_g) * orho

    xrg = jnp.maximum(R1, (pre.qg1d + qgten * dt_arr) * pre.rho)
    xng = jnp.maximum(R2, (pre.ng1d + ngten * dt_arr) * pre.rho)
    xrb = jnp.maximum(xrg / pre.rho / jnp.asarray(RHO_G[-1], dtype=jnp.float32), pre.qb1d + qbten * dt_arr)
    xrb = jnp.minimum(xrg / pre.rho / jnp.asarray(RHO_G[0], dtype=jnp.float32), xrb)
    del xrb
    am_g = jnp.take(jnp.asarray(AM_G, dtype=jnp.float32), pre.idx_bg)
    lamg = (am_g * CGG[2, 0] * OGG2 * xng / xrg) ** OBMG
    mvd_g = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / lamg
    ngten_high = (CGG[1, 0] * OGG3 * xrg * ((jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / jnp.float32(25.4e-3)) ** BM_G / am_g - pre.ng1d * pre.rho) * odts * orho
    ngten_low = (CGG[1, 0] * OGG3 * xrg * ((jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / D0R) ** BM_G / am_g - pre.ng1d * pre.rho) * odts * orho
    ngten = jnp.where(xrg > R1, jnp.where(mvd_g > jnp.float32(25.4e-3), ngten_high, jnp.where(mvd_g < D0R, ngten_low, ngten)), -pre.ng1d * odts)
    qgten = jnp.where(xrg > R1, qgten, -pre.qg1d * odts)
    qbten = jnp.where(xrg > R1, qbten, -pre.qb1d * odts)
    mvd_g = jnp.where(xrg > R1, jnp.minimum(jnp.maximum(mvd_g, D0R), jnp.float32(25.4e-3)), pre.mvd_g)

    dry_factor = jnp.float32(1 - int(ifdry))
    cold_tendency = (
        LSUB * pre.ocp * (depleted.pri_inu + depleted.pri_ide + depleted.prs_ide + depleted.prs_sde + depleted.prg_gde + depleted.pri_iha)
        + (LSUB - pre.lvap)
        * pre.ocp
        * (
            depleted.pri_wfz
            + depleted.pri_rfz
            + depleted.prg_rfz
            + depleted.prs_scw
            + depleted.prg_scw
            + depleted.prg_gcw
            + prg_rcs
            + depleted.prs_rcs
            + depleted.prr_rci
            + depleted.prg_rcg
        )
    ) * orho * dry_factor
    warm_tendency = (
        LFUS * pre.ocp * (-depleted.prr_sml - depleted.prr_gml - depleted.prr_rcg - depleted.prr_rcs)
        + LSUB * pre.ocp * (depleted.prs_sde + depleted.prg_gde)
    ) * orho * dry_factor
    tten = tten + jnp.where(pre.temp < jnp.float32(273.15), cold_tendency, warm_tendency)

    return ThompsonTendencyResult(
        nwfaten=nwfaten,
        nifaten=nifaten,
        nbcaten=nbcaten,
        qvten=qvten,
        qcten=qcten,
        ncten=ncten,
        qiten=qiten,
        niten=niten,
        qrten=qrten,
        nrten=nrten,
        qsten=qsten,
        qgten=qgten,
        ngten=ngten,
        qbten=qbten,
        tten=tten,
        mvd_r=mvd_r,
        mvd_g=mvd_g,
    )


def thompson_post_tendency_refresh(
    pre: ThompsonPreprocessResult,
    tendencies: ThompsonTendencyResult,
    dt: Any,
    *,
    is_aerosol_aware: bool = False,
    is_hail_aware: bool = False,
    iiwarm: bool = False,
) -> ThompsonPostTendencyState:
    """Refresh WRF Thompson internal column state after tendency accumulation."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    temp = pre.temp + dt_arr * jnp.asarray(tendencies.tten, dtype=jnp.float32)
    otemp = jnp.float32(1.0) / temp
    tempc = temp - jnp.float32(273.15)
    qv = jnp.maximum(jnp.float32(1.0e-10), pre.qv1d + dt_arr * jnp.asarray(tendencies.qvten, dtype=jnp.float32))
    rho = jnp.float32(0.622) * pre.pres / (R * temp * (qv + jnp.float32(0.622)))
    rhof = sqrt_for_wrf(RHO_NOT / rho)
    rhof2 = sqrt_for_wrf(rhof)
    qvs = rslf(pre.pres, temp)
    ssatw = qv / qvs - jnp.float32(1.0)
    ssatw = jnp.where(jnp.abs(ssatw) < EPS, jnp.zeros_like(ssatw), ssatw)
    diffu = jnp.float32(2.11e-5) * (temp / jnp.float32(273.15)) ** jnp.float32(1.94) * (jnp.float32(101325.0) / pre.pres)
    visco = jnp.where(
        tempc >= jnp.float32(0.0),
        (jnp.float32(1.718) + jnp.float32(0.0049) * tempc) * jnp.float32(1.0e-5),
        (jnp.float32(1.718) + jnp.float32(0.0049) * tempc - jnp.float32(1.2e-5) * tempc * tempc) * jnp.float32(1.0e-5),
    )
    vsc2 = sqrt_for_wrf(rho / visco)
    lvap = LVAP0 + (jnp.float32(2106.0) - jnp.float32(4218.0)) * tempc
    tcond = (jnp.float32(5.69) + jnp.float32(0.0168) * tempc) * jnp.float32(1.0e-5) * jnp.float32(418.936)
    ocp = jnp.float32(1.0) / (CP * (jnp.float32(1.0) + jnp.float32(0.887) * qv))
    lvt2 = lvap * lvap * ocp * ORV * otemp * otemp
    nwfa = jnp.maximum(jnp.float32(11.1e6), (pre.nwfa / pre.rho + jnp.asarray(tendencies.nwfaten, dtype=jnp.float32) * dt_arr) * rho)

    qc_new = pre.qc1d + jnp.asarray(tendencies.qcten, dtype=jnp.float32) * dt_arr
    l_qc = qc_new > R1
    rc = jnp.where(l_qc, qc_new * rho, R1)
    nc_active = jnp.maximum(jnp.float32(2.0), jnp.minimum((pre.nc1d + jnp.asarray(tendencies.ncten, dtype=jnp.float32) * dt_arr) * rho, NT_C_MAX))
    if not is_aerosol_aware:
        nc_active = jnp.ones_like(nc_active) * NT_C
    nc = jnp.where(l_qc, nc_active, jnp.float32(2.0))

    qi_new = pre.qi1d + jnp.asarray(tendencies.qiten, dtype=jnp.float32) * dt_arr
    l_qi = qi_new > R1
    ri = jnp.where(l_qi, qi_new * rho, R1)
    ni = jnp.where(l_qi, jnp.maximum(R2, (pre.ni1d + jnp.asarray(tendencies.niten, dtype=jnp.float32) * dt_arr) * rho), R2)

    qr_new = pre.qr1d + jnp.asarray(tendencies.qrten, dtype=jnp.float32) * dt_arr
    l_qr = qr_new > R1
    rr_active = jnp.where(l_qr, qr_new * rho, jnp.ones_like(rho))
    nr_active = jnp.maximum(R2, (pre.nr1d + jnp.asarray(tendencies.nrten, dtype=jnp.float32) * dt_arr) * rho)
    lamr_active = (AM_R * CRG[2] * ORG2 * nr_active / rr_active) ** OBMR
    mvd_r_active = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr_active
    lamr_high = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / jnp.float32(2.5e-3)
    nr_high = CRG[1] * ORG3 * rr_active * lamr_high**BM_R / AM_R
    lamr_low = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / (D0R * jnp.float32(0.75))
    nr_low = CRG[1] * ORG3 * rr_active * lamr_low**BM_R / AM_R
    nr_active = jnp.where(mvd_r_active > jnp.float32(2.5e-3), nr_high, jnp.where(mvd_r_active < D0R * jnp.float32(0.75), nr_low, nr_active))
    rr = jnp.where(l_qr, rr_active, R1)
    nr = jnp.where(l_qr, nr_active, R2)
    l_qr = jnp.asarray(l_qr, dtype=bool)

    qs_new = pre.qs1d + jnp.asarray(tendencies.qsten, dtype=jnp.float32) * dt_arr
    l_qs = qs_new > R1
    rs = jnp.where(l_qs, qs_new * rho, R1)

    qg_new = pre.qg1d + jnp.asarray(tendencies.qgten, dtype=jnp.float32) * dt_arr
    l_qg_active = qg_new > R1
    rho_g_table = jnp.asarray(RHO_G, dtype=jnp.float32)
    am_g_table = jnp.asarray(AM_G, dtype=jnp.float32)
    rg_active = jnp.where(l_qg_active, qg_new * rho, jnp.ones_like(rho))
    ng_seed = jnp.maximum(R2, (pre.ng1d + jnp.asarray(tendencies.ngten, dtype=jnp.float32) * dt_arr) * rho)
    rb_hail = jnp.maximum(rg_active / rho / rho_g_table[-1], pre.qb1d + jnp.asarray(tendencies.qbten, dtype=jnp.float32) * dt_arr)
    rb_hail = jnp.minimum(rg_active / rho / rho_g_table[0], rb_hail)
    idx_hail = jnp.clip(_fortran_nint(rg_active / rho / rb_hail * jnp.float32(0.01)) + jnp.asarray(1, dtype=jnp.int32), 1, RHO_G.size) - 1

    idx_nohail = jnp.ones_like(pre.idx_bg, dtype=jnp.int32) * jnp.asarray(IDX_BG1, dtype=jnp.int32)
    ygra1 = jnp.log10(jnp.maximum(jnp.float32(1.0e-9), rg_active))
    zans1 = jnp.float32(3.0) + (jnp.float32(2.0) / jnp.float32(7.0)) * (ygra1 + jnp.float32(8.0))
    zans1 = jnp.maximum(jnp.float32(2.0), jnp.minimum(zans1, jnp.float32(6.0)))
    n0_exp = jnp.float32(10.0) ** zans1
    am_g_bg1 = am_g_table[IDX_BG1]
    lam_exp = (n0_exp * am_g_bg1 * CGG[0, 0] / rg_active) ** OGE1
    lamg_nohail = lam_exp * (CGG[2, 0] * OGG2 * OGG1) ** OBMG
    ng_nohail = CGG[1, 0] * OGG3 * rg_active * lamg_nohail**BM_G / am_g_bg1
    rb_nohail = rg_active / rho / rho_g_table[IDX_BG1]

    idx_bg = jnp.where(is_hail_aware, idx_hail, idx_nohail)
    ng_active = jnp.where(is_hail_aware, ng_seed, ng_nohail)
    rb_active = jnp.where(is_hail_aware, rb_hail, rb_nohail)
    rb_default = R1 / rho / rho_g_table[-1]
    rg = jnp.where(l_qg_active, rg_active, R1)
    ng = jnp.where(l_qg_active, ng_active, R2)
    rb = jnp.where(l_qg_active, rb_active, rb_default)
    idx_bg = jnp.where(l_qg_active, idx_bg, jnp.asarray(IDX_BG1, dtype=jnp.int32))
    l_qg = jnp.asarray(l_qg_active, dtype=bool)

    zeros = jnp.zeros_like(temp)
    tc0 = jnp.minimum(jnp.float32(-0.1), temp - jnp.float32(273.15))
    smob_active = rs * OAMS
    a_bm, b_bm = _snow_moment_coeff(tc0, BM_S)
    smo2_generic = (smob_active / a_bm) ** (jnp.float32(1.0) / b_bm)
    if float(BM_S) > 2.0 - 1.0e-3 and float(BM_S) < 2.0 + 1.0e-3:
        smo2_active = smob_active
    else:
        smo2_active = smo2_generic
    a_c, b_c = _snow_moment_coeff(tc0, CSE[0])
    a_d, b_d = _snow_moment_coeff(tc0, CSE[13])
    smoc_active = a_c * smo2_active**b_c
    smod_active = a_d * smo2_active**b_d
    snow_moments_active = (~bool(iiwarm)) & l_qs
    smob = jnp.where(snow_moments_active, smob_active, zeros)
    smo2 = jnp.where(snow_moments_active, smo2_active, zeros)
    smoc = jnp.where(snow_moments_active, smoc_active, zeros)
    smod = jnp.where(snow_moments_active, smod_active, zeros)

    am_g_idx = jnp.take(am_g_table, idx_bg)
    lamg = (am_g_idx * CGG[2, 0] * OGG2 * ng / rg) ** OBMG
    ilamg_calc = jnp.float32(1.0) / lamg
    n0_g_calc = ng * OGG2 * lamg ** CGE[1, 0]
    ilamg = jnp.where(bool(iiwarm), zeros, ilamg_calc)
    n0_g = jnp.where(bool(iiwarm), zeros, n0_g_calc)

    lamr = (AM_R * CRG[2] * ORG2 * nr / rr) ** OBMR
    ilamr = jnp.float32(1.0) / lamr
    mvd_r = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr
    n0_r = nr * ORG2 * lamr ** CRE[1]

    return ThompsonPostTendencyState(
        temp=temp,
        qv=qv,
        rho=rho,
        rhof=rhof,
        rhof2=rhof2,
        qvs=qvs,
        ssatw=ssatw,
        diffu=diffu,
        visco=visco,
        vsc2=vsc2,
        lvap=lvap,
        tcond=tcond,
        ocp=ocp,
        lvt2=lvt2,
        nwfa=nwfa,
        rc=rc,
        nc=nc,
        l_qc=l_qc,
        ri=ri,
        ni=ni,
        l_qi=l_qi,
        rr=rr,
        nr=nr,
        l_qr=l_qr,
        mvd_r=mvd_r,
        rs=rs,
        l_qs=l_qs,
        rg=rg,
        ng=ng,
        rb=rb,
        l_qg=l_qg,
        idx_bg=idx_bg,
        smob=smob,
        smo2=smo2,
        smoc=smoc,
        smod=smod,
        ilamg=ilamg,
        n0_g=n0_g,
        ilamr=ilamr,
        n0_r=n0_r,
    )


def thompson_cloud_water_condensation(
    pre: ThompsonPreprocessResult,
    state: ThompsonPostTendencyState,
    tendencies: ThompsonTendencyResult,
    dt: Any,
    *,
    w1d: Any,
    tnc_wev: Any,
    tnccn_act: Any,
    is_aerosol_aware: bool = False,
    ifdry: int = 0,
) -> ThompsonCloudWaterCondensationResult:
    """Apply WRF Thompson cloud-water condensation and evaporation update."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odt = jnp.float32(1.0) / dt_arr
    odts = odt
    zeros = jnp.zeros_like(state.temp)
    qvten = jnp.asarray(tendencies.qvten, dtype=jnp.float32)
    qcten = jnp.asarray(tendencies.qcten, dtype=jnp.float32)
    ncten = jnp.asarray(tendencies.ncten, dtype=jnp.float32)
    nwfaten = jnp.asarray(tendencies.nwfaten, dtype=jnp.float32)
    tten = jnp.asarray(tendencies.tten, dtype=jnp.float32)

    active = (state.ssatw > EPS) | ((state.ssatw < -EPS) & state.l_qc)
    clap0 = (state.qv - state.qvs) / (jnp.float32(1.0) + state.lvt2 * state.qvs)

    def _newton(_, clap):
        exp_arg = jnp.exp(state.lvt2 * clap)
        fcd = state.qvs * exp_arg - state.qv + clap
        dfcd = state.qvs * state.lvt2 * exp_arg + jnp.float32(1.0)
        return clap - fcd / dfcd

    clap = lax.fori_loop(0, 3, _newton, clap0)
    xrc = state.rc + clap * state.rho
    prw_vcd_base = clap * odt
    condensing = active & (xrc > R1) & (clap > EPS)
    evaporating = active & (xrc > R1) & (clap < -EPS) & (state.ssatw < jnp.float32(-1.0e-6)) & bool(is_aerosol_aware)
    depleted = active & (xrc <= R1)

    xnc = jnp.where(bool(is_aerosol_aware), jnp.maximum(jnp.float32(2.0), activ_ncloud(state.temp, w1d, state.nwfa, tnccn_act)), jnp.ones_like(state.temp) * NT_C)
    pnc_cond = jnp.float32(0.5) * (xnc - state.nc + jnp.abs(xnc - state.nc)) * odts / state.rho

    otemp = jnp.float32(1.0) / state.temp
    rvs = state.rho * state.qvs
    lvap_otemp_orv = state.lvap * otemp * ORV
    rvs_p = rvs * otemp * (lvap_otemp_orv - jnp.float32(1.0))
    rvs_pp = rvs * (
        otemp * (lvap_otemp_orv - jnp.float32(1.0)) * otemp * (lvap_otemp_orv - jnp.float32(1.0))
        + (-jnp.float32(2.0) * state.lvap * otemp * otemp * otemp * ORV)
        + otemp * otemp
    )
    gamsc = state.lvap * state.diffu / state.tcond * rvs_p
    gamsc_frac = gamsc / (jnp.float32(1.0) + gamsc)
    alphsc = jnp.float32(0.5) * gamsc_frac * gamsc_frac * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(jnp.float32(1.0e-9), alphsc)
    xsat = jnp.where(jnp.abs(state.ssatw) < jnp.float32(1.0e-9), zeros, state.ssatw)
    t1_evap = (
        jnp.float32(2.0)
        * _PI
        * (
            jnp.float32(1.0)
            - alphsc * xsat
            + jnp.float32(2.0) * alphsc * alphsc * xsat * xsat
            - jnp.float32(5.0) * alphsc * alphsc * alphsc * xsat * xsat * xsat
        )
        / (jnp.float32(1.0) + gamsc)
    )
    dc_star = sqrt_for_wrf(-jnp.float32(2.0) * dt_arr * t1_evap / (jnp.float32(2.0) * _PI) * jnp.float32(4.0) * state.diffu * state.ssatw * rvs / RHO_W)
    idx_d = jnp.clip((jnp.float32(1.0e6) * dc_star).astype(jnp.int32), 1, NBC) - 1

    idx_n = _cloud_number_lookup_index(state.nc)
    idx_c = _decade_lookup_index(state.rc, R_C[0], -6, NTB_C)
    tnc = jnp.asarray(tnc_wev, dtype=jnp.float32)
    pnc_evap = jnp.maximum(-state.nc * jnp.float32(0.99) / state.rho * odt, -tnc[idx_d, idx_c, idx_n] / state.rho * odt)

    prw_active = jnp.where(depleted, -state.rc / state.rho * odt, prw_vcd_base)
    prw_active = jnp.where(evaporating, jnp.maximum(-state.rc * jnp.float32(0.99) / state.rho * odt, prw_active), prw_active)
    pnc_active = jnp.where(depleted, -state.nc / state.rho * odt, zeros)
    pnc_active = jnp.where(condensing, pnc_cond, pnc_active)
    pnc_active = jnp.where(evaporating, pnc_evap, pnc_active)
    prw_vcd = jnp.where(active, prw_active, zeros)
    pnc_wcd = jnp.where(active, pnc_active, zeros)

    qvten_out = qvten - prw_vcd
    qcten_out = qcten + prw_vcd
    ncten_out = ncten + pnc_wcd
    nwfaten_out = nwfaten - pnc_wcd
    tten_out = tten + state.lvap * state.ocp * prw_vcd * jnp.float32(1 - int(ifdry))

    rc = jnp.maximum(R1, (pre.qc1d + dt_arr * qcten_out) * state.rho)
    l_qc = rc != R1
    nc = jnp.maximum(jnp.float32(2.0), jnp.minimum((pre.nc1d + ncten_out * dt_arr) * state.rho, NT_C_MAX))
    if not is_aerosol_aware:
        nc = jnp.ones_like(nc) * NT_C
    qv = jnp.maximum(jnp.float32(1.0e-10), pre.qv1d + dt_arr * qvten_out)
    temp = pre.temp + dt_arr * tten_out
    rho = jnp.float32(0.622) * pre.pres / (R * temp * (qv + jnp.float32(0.622)))
    qvs = rslf(pre.pres, temp)
    ssatw = qv / qvs - jnp.float32(1.0)

    return ThompsonCloudWaterCondensationResult(
        qvten=jnp.where(active, qvten_out, qvten),
        qcten=jnp.where(active, qcten_out, qcten),
        ncten=jnp.where(active, ncten_out, ncten),
        nwfaten=jnp.where(active, nwfaten_out, nwfaten),
        tten=jnp.where(active, tten_out, tten),
        rc=jnp.where(active, rc, state.rc),
        nc=jnp.where(active, nc, state.nc),
        l_qc=jnp.where(active, l_qc, state.l_qc),
        qv=jnp.where(active, qv, state.qv),
        temp=jnp.where(active, temp, state.temp),
        rho=jnp.where(active, rho, state.rho),
        qvs=jnp.where(active, qvs, state.qvs),
        ssatw=jnp.where(active, ssatw, state.ssatw),
        prw_vcd=prw_vcd,
        pnc_wcd=pnc_wcd,
    )


def thompson_rain_evaporation(
    pre: ThompsonPreprocessResult,
    state: ThompsonPostTendencyState,
    cloud: ThompsonCloudWaterCondensationResult,
    tendencies: ThompsonTendencyResult,
    dt: Any,
    *,
    prr_gml: Any,
    ifdry: int = 0,
) -> ThompsonRainEvaporationResult:
    """Apply WRF Thompson rain evaporation after cloud-water adjustment."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odts = jnp.float32(1.0) / dt_arr
    zeros = jnp.zeros_like(state.temp)
    qvten = jnp.asarray(cloud.qvten, dtype=jnp.float32)
    qrten = jnp.asarray(tendencies.qrten, dtype=jnp.float32)
    nrten = jnp.asarray(tendencies.nrten, dtype=jnp.float32)
    nwfaten = jnp.asarray(cloud.nwfaten, dtype=jnp.float32)
    tten = jnp.asarray(cloud.tten, dtype=jnp.float32)
    prr_gml_arr = jnp.asarray(prr_gml, dtype=jnp.float32)

    active = (cloud.ssatw < -EPS) & state.l_qr & ~(cloud.prw_vcd > jnp.float32(0.0))
    otemp = jnp.float32(1.0) / cloud.temp
    orho = jnp.float32(1.0) / cloud.rho
    tempc = cloud.temp - jnp.float32(273.15)
    rhof = sqrt_for_wrf(RHO_NOT * orho)
    rhof2 = sqrt_for_wrf(rhof)
    diffu = jnp.float32(2.11e-5) * (cloud.temp / jnp.float32(273.15)) ** jnp.float32(1.94) * (jnp.float32(101325.0) / pre.pres)
    visco = jnp.where(
        tempc >= jnp.float32(0.0),
        (jnp.float32(1.718) + jnp.float32(0.0049) * tempc) * jnp.float32(1.0e-5),
        (jnp.float32(1.718) + jnp.float32(0.0049) * tempc - jnp.float32(1.2e-5) * tempc * tempc) * jnp.float32(1.0e-5),
    )
    vsc2 = sqrt_for_wrf(cloud.rho / visco)
    lvap = LVAP0 + (jnp.float32(2106.0) - jnp.float32(4218.0)) * tempc
    tcond = (jnp.float32(5.69) + jnp.float32(0.0168) * tempc) * jnp.float32(1.0e-5) * jnp.float32(418.936)
    ocp = jnp.float32(1.0) / (CP * (jnp.float32(1.0) + jnp.float32(0.887) * cloud.qv))

    rvs = cloud.rho * cloud.qvs
    lvap_otemp_orv = lvap * otemp * ORV
    rvs_p = rvs * otemp * (lvap_otemp_orv - jnp.float32(1.0))
    rvs_pp = rvs * (
        otemp * (lvap_otemp_orv - jnp.float32(1.0)) * otemp * (lvap_otemp_orv - jnp.float32(1.0))
        + (-jnp.float32(2.0) * lvap * otemp * otemp * otemp * ORV)
        + otemp * otemp
    )
    gamsc = lvap * diffu / tcond * rvs_p
    gamsc_frac = gamsc / (jnp.float32(1.0) + gamsc)
    alphsc = jnp.float32(0.5) * gamsc_frac * gamsc_frac * rvs_pp / rvs_p * rvs / rvs_p
    alphsc = jnp.maximum(jnp.float32(1.0e-9), alphsc)
    xsat = jnp.minimum(jnp.float32(-1.0e-9), cloud.ssatw)
    t1_evap = (
        jnp.float32(2.0)
        * _PI
        * (
            jnp.float32(1.0)
            - alphsc * xsat
            + jnp.float32(2.0) * alphsc * alphsc * xsat * xsat
            - jnp.float32(5.0) * alphsc * alphsc * alphsc * xsat * xsat * xsat
        )
        / (jnp.float32(1.0) + gamsc)
    )

    lamr = jnp.float32(1.0) / state.ilamr
    rapid = (cloud.qv / cloud.qvs < jnp.float32(0.95)) & (state.rr * orho <= jnp.float32(1.0e-8))
    t1_qr_ev = jnp.float32(0.78) * CRG[9]
    t2_qr_ev = jnp.float32(0.308) * SC3 * sqrt_for_wrf(AV_R) * CRG[10]
    prv_raw = (
        t1_evap
        * diffu
        * (-cloud.ssatw)
        * state.n0_r
        * rvs
        * (t1_qr_ev * state.ilamr ** CRE[9] + t2_qr_ev * vsc2 * rhof2 * ((lamr + jnp.float32(0.5) * FV_R) ** (-CRE[10])))
    )
    rate_max = jnp.minimum(state.rr * orho * odts, (cloud.qvs - cloud.qv) * odts)
    prv_general = jnp.minimum(rate_max, prv_raw * orho)
    prv_base = jnp.where(rapid, state.rr * orho * odts, prv_general)
    eva_factor = jnp.minimum(jnp.float32(1.0), jnp.float32(0.01) + (jnp.float32(0.99) - jnp.float32(0.01)) * (tempc / jnp.float32(20.0)))
    # WRF applies this reduction only in the general evaporation branch;
    # trace rain below 95% RH must still evaporate completely in one step.
    prv_reduced = jnp.where(~rapid & (prr_gml_arr > jnp.float32(0.0)), prv_base * eva_factor, prv_base)
    prv_rev = jnp.where(active, prv_reduced, zeros)
    pnr_rev = jnp.where(
        active,
        jnp.minimum(state.nr * jnp.float32(0.99) * orho * odts, prv_rev * state.nr / state.rr),
        zeros,
    )

    qrten_out = qrten - prv_rev
    qvten_out = qvten + prv_rev
    nrten_out = nrten - pnr_rev
    nwfaten_out = nwfaten + pnr_rev
    tten_out = tten - lvap * ocp * prv_rev * jnp.float32(1 - int(ifdry))
    rr = jnp.maximum(R1, (pre.qr1d + dt_arr * qrten_out) * cloud.rho)
    qv = jnp.maximum(jnp.float32(1.0e-10), pre.qv1d + dt_arr * qvten_out)
    nr = jnp.maximum(R2, (pre.nr1d + dt_arr * nrten_out) * cloud.rho)
    temp = pre.temp + dt_arr * tten_out
    rho = jnp.float32(0.622) * pre.pres / (R * temp * (qv + jnp.float32(0.622)))

    return ThompsonRainEvaporationResult(
        qvten=jnp.where(active, qvten_out, qvten),
        qrten=jnp.where(active, qrten_out, qrten),
        nrten=jnp.where(active, nrten_out, nrten),
        nwfaten=jnp.where(active, nwfaten_out, nwfaten),
        tten=jnp.where(active, tten_out, tten),
        rr=jnp.where(active, rr, state.rr),
        qv=jnp.where(active, qv, cloud.qv),
        nr=jnp.where(active, nr, state.nr),
        temp=jnp.where(active, temp, cloud.temp),
        rho=jnp.where(active, rho, cloud.rho),
        rhof=jnp.where(active, rhof, state.rhof),
        rhof2=jnp.where(active, rhof2, state.rhof2),
        diffu=jnp.where(active, diffu, state.diffu),
        visco=jnp.where(active, visco, state.visco),
        vsc2=jnp.where(active, vsc2, state.vsc2),
        lvap=jnp.where(active, lvap, state.lvap),
        tcond=jnp.where(active, tcond, state.tcond),
        ocp=jnp.where(active, ocp, state.ocp),
        prv_rev=prv_rev,
        pnr_rev=pnr_rev,
    )


def _sedimentation_profile_setup(
    speeds: tuple[Any, ...], present: Any, dz: Any, dt: Any, *, number_limits: bool = False,
) -> tuple[tuple[Any, ...], Any, Any]:
    """Propagate the nearest present level above, retaining WRF's step limits."""
    count = present.shape[0]
    if count == 0:
        return speeds, jnp.int32(0), jnp.int32(0)
    levels = jnp.arange(count, dtype=jnp.int32)
    source = jnp.where(present, levels, jnp.int32(count))
    # Only integer indices are reduced. Floating-point speeds are copied, not
    # summed or recomputed, through dry gaps in the original top-down order.
    nearest = lax.associative_scan(jnp.minimum, source[::-1])[::-1]
    filled = tuple(
        jnp.where(nearest < count, speed[jnp.minimum(nearest, count - 1)], jnp.float32(0))
        for speed in speeds
    )
    limit = jnp.maximum(*filled) if number_limits else filled[0]
    sedimenting = limit > jnp.float32(1.0e-3)
    delta = dz / jnp.maximum(limit, jnp.float32(1.0e-30))
    steps = (dt / delta + jnp.float32(1.0)).astype(jnp.int32)
    nstep = jnp.max(jnp.where(sedimenting, steps, jnp.int32(0)), initial=0)
    ksed = jnp.max(jnp.where(sedimenting, levels, jnp.int32(0)), initial=0)
    return filled, nstep, ksed


def thompson_rain_sedimentation_setup(
    state: ThompsonPostTendencyState,
    rain: ThompsonRainEvaporationResult,
    dt: Any,
    dzq: Any,
) -> ThompsonRainSedimentationSetupResult:
    """Compute WRF Thompson rain fall speeds and sedimentation substep size."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    rr = jnp.asarray(rain.rr, dtype=jnp.float32)
    nr = jnp.asarray(rain.nr, dtype=jnp.float32)
    rho = jnp.asarray(rain.rho, dtype=jnp.float32)
    nlev = rr.shape[0]
    zeros = jnp.zeros_like(rr)
    active_any = jnp.any(state.l_qr)
    rhof = sqrt_for_wrf(RHO_NOT / rho)
    lamr = (AM_R * CRG[2] * ORG2 * nr / rr) ** OBMR
    vtr_mass = rhof * AV_R * CRG[5] * ORG3 * lamr ** CRE[2] * ((lamr + FV_R) ** (-CRE[5]))
    vtr_number = rhof * AV_R * CRG[6] / CRG[11] * lamr ** CRE[11] * ((lamr + FV_R) ** (-CRE[6]))
    (vtrk, vtnrk), nstep_scan, ksed_scan = _sedimentation_profile_setup(
        (vtr_mass, vtr_number), rr > R1, dzq_arr, dt_arr, number_limits=True,
    )
    ksed1 = jnp.where((ksed_scan == nlev - 1) & (nlev > 1), jnp.asarray(nlev - 2, dtype=jnp.int32), ksed_scan)
    onstep = jnp.where(nstep_scan > 0, jnp.float32(1.0) / nstep_scan.astype(jnp.float32), jnp.float32(1.0))

    return ThompsonRainSedimentationSetupResult(
        vtrk=jnp.where(active_any, vtrk, zeros),
        vtnrk=jnp.where(active_any, vtnrk, zeros),
        rhof=jnp.where(active_any, rhof, rain.rhof),
        ksed1=jnp.where(active_any, ksed1, jnp.asarray(0, dtype=jnp.int32)),
        nstep=jnp.where(active_any, nstep_scan, jnp.asarray(0, dtype=jnp.int32)),
        onstep=jnp.where(active_any, onstep, jnp.float32(1.0)),
    )


def thompson_rain_sedimentation(
    rain: ThompsonRainEvaporationResult,
    setup: ThompsonRainSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    pptrain0: Any = 0.0,
) -> ThompsonRainSedimentationResult:
    """Apply WRF Thompson rain sedimentation flux updates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    odzq = jnp.float32(1.0) / dzq_arr
    orho = jnp.float32(1.0) / jnp.asarray(rain.rho, dtype=jnp.float32)
    indices = jnp.arange(jnp.asarray(rain.rr).shape[0], dtype=jnp.int32)
    top = jnp.asarray(jnp.asarray(rain.rr).shape[0] - 1, dtype=jnp.int32)
    bottom = jnp.asarray(0, dtype=jnp.int32)
    onstep = jnp.asarray(setup.onstep, dtype=jnp.float32)
    nsteps = _fortran_nint(jnp.float32(1.0) / onstep)

    def _substep(_, carry: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
        qrten, nrten, rr, nr, pptrain = carry
        sed_r = setup.vtrk * rr
        sed_n = setup.vtnrk * nr
        sed_r_next = jnp.concatenate([sed_r[1:], jnp.zeros((1,), dtype=sed_r.dtype)])
        sed_n_next = jnp.concatenate([sed_n[1:], jnp.zeros((1,), dtype=sed_n.dtype)])
        column_mask = indices <= setup.ksed1
        top_mask = indices == top
        mass_tendency = jnp.where(column_mask, (sed_r_next - sed_r) * odzq * onstep * orho, jnp.zeros_like(rr))
        number_tendency = jnp.where(column_mask, (sed_n_next - sed_n) * odzq * onstep * orho, jnp.zeros_like(nr))
        mass_tendency = mass_tendency + jnp.where(top_mask, -sed_r * odzq * onstep * orho, jnp.zeros_like(rr))
        number_tendency = number_tendency + jnp.where(top_mask, -sed_n * odzq * onstep * orho, jnp.zeros_like(nr))
        mass_state = jnp.where(column_mask, (sed_r_next - sed_r) * odzq * dt_arr * onstep, jnp.zeros_like(rr))
        number_state = jnp.where(column_mask, (sed_n_next - sed_n) * odzq * dt_arr * onstep, jnp.zeros_like(nr))
        mass_state = mass_state + jnp.where(top_mask, -sed_r * odzq * dt_arr * onstep, jnp.zeros_like(rr))
        number_state = number_state + jnp.where(top_mask, -sed_n * odzq * dt_arr * onstep, jnp.zeros_like(nr))
        qrten = qrten + mass_tendency
        nrten = nrten + number_tendency
        rr = jnp.maximum(R1, rr + mass_state)
        nr = jnp.maximum(R2, nr + number_state)
        pptrain = jnp.where(rr[bottom] > R1 * jnp.float32(1000.0), pptrain + sed_r[bottom] * dt_arr * onstep, pptrain)
        return qrten, nrten, rr, nr, pptrain

    qrten0 = jnp.asarray(rain.qrten, dtype=jnp.float32)
    nrten0 = jnp.asarray(rain.nrten, dtype=jnp.float32)
    rr0 = jnp.asarray(rain.rr, dtype=jnp.float32)
    nr0 = jnp.asarray(rain.nr, dtype=jnp.float32)
    pptrain_init = jnp.asarray(pptrain0, dtype=jnp.float32)
    qrten, nrten, rr, nr, pptrain = dynamic_fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        nsteps,
        _substep,
        (qrten0, nrten0, rr0, nr0, pptrain_init),
    )
    return ThompsonRainSedimentationResult(qrten=qrten, nrten=nrten, rr=rr, nr=nr, pptrain=pptrain)


def thompson_cloud_sedimentation_setup(
    cloud: ThompsonCloudWaterCondensationResult,
    rain_setup: ThompsonRainSedimentationSetupResult,
    dzq: Any,
    w1d: Any,
) -> ThompsonCloudSedimentationSetupResult:
    """Compute WRF Thompson cloud-water fall speeds and sedimentation start level."""

    rc = jnp.asarray(cloud.rc, dtype=jnp.float32)
    nc = jnp.asarray(cloud.nc, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    w_arr = jnp.asarray(w1d, dtype=jnp.float32)
    nlev = rc.shape[0]
    indices = jnp.arange(nlev, dtype=jnp.int32)
    setup_indices = jnp.arange(max(nlev - 1, 1), dtype=jnp.int32)
    active_any = jnp.any(cloud.l_qc)
    previous_height = jnp.concatenate([jnp.zeros((1,), dtype=dzq_arr.dtype), jnp.cumsum(dzq_arr[:-2])]) if nlev > 1 else jnp.zeros((1,), dtype=dzq_arr.dtype)
    processed = previous_height <= jnp.float32(500.0)
    ksed_candidates = jnp.where((rc[: max(nlev - 1, 1)] > R2) & processed, setup_indices, jnp.asarray(0, dtype=jnp.int32))
    ksed1 = jnp.max(ksed_candidates)

    nu_c = jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / nc) + jnp.asarray(2, dtype=jnp.int32))
    nu_idx = nu_c - jnp.asarray(1, dtype=jnp.int32)
    lamc = (nc * AM_R * CCG[1, nu_idx] * OCG1[nu_idx] / rc) ** OBMR
    ilamc = jnp.float32(1.0) / lamc
    vtck_calc = rain_setup.rhof * AV_C * CCG[4, nu_idx] * OCG2[nu_idx] * ilamc**BV_C
    vtnck_calc = rain_setup.rhof * AV_C * CCG[3, nu_idx] * OCG1[nu_idx] * ilamc**BV_C
    speed_active = active_any & (indices <= ksed1) & (rc > R1) & (w_arr < jnp.float32(1.0e-1))
    zeros = jnp.zeros_like(rc)
    return ThompsonCloudSedimentationSetupResult(
        vtck=jnp.where(speed_active, vtck_calc, zeros),
        vtnck=jnp.where(speed_active, vtnck_calc, zeros),
        ksed1=jnp.where(active_any, ksed1, jnp.asarray(0, dtype=jnp.int32)),
    )


def thompson_cloud_sedimentation(
    cloud: ThompsonCloudWaterCondensationResult,
    rain: ThompsonRainEvaporationResult,
    setup: ThompsonCloudSedimentationSetupResult,
    dt: Any,
    dzq: Any,
) -> ThompsonCloudSedimentationResult:
    """Apply WRF Thompson cloud-water sedimentation flux updates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    odzq = jnp.float32(1.0) / dzq_arr
    orho = jnp.float32(1.0) / jnp.asarray(rain.rho, dtype=jnp.float32)
    rc = jnp.asarray(cloud.rc, dtype=jnp.float32)
    nc = jnp.asarray(cloud.nc, dtype=jnp.float32)
    qcten = jnp.asarray(cloud.qcten, dtype=jnp.float32)
    ncten = jnp.asarray(cloud.ncten, dtype=jnp.float32)
    sed_c = setup.vtck * rc
    sed_n = setup.vtnck * nc
    sed_c_next = jnp.concatenate([sed_c[1:], jnp.zeros((1,), dtype=sed_c.dtype)])
    sed_n_next = jnp.concatenate([sed_n[1:], jnp.zeros((1,), dtype=sed_n.dtype)])
    indices = jnp.arange(rc.shape[0], dtype=jnp.int32)
    active_any = jnp.any(cloud.l_qc)
    mask = active_any & (indices <= setup.ksed1)
    mass_flux = (sed_c_next - sed_c) * odzq
    number_flux = (sed_n_next - sed_n) * odzq
    qcten = qcten + jnp.where(mask, mass_flux * orho, jnp.zeros_like(qcten))
    ncten = ncten + jnp.where(mask, number_flux * orho, jnp.zeros_like(ncten))
    rc = jnp.maximum(R1, rc + jnp.where(mask, mass_flux * dt_arr, jnp.zeros_like(rc)))
    nc = jnp.maximum(jnp.float32(10.0), nc + jnp.where(mask, number_flux * dt_arr, jnp.zeros_like(nc)))
    return ThompsonCloudSedimentationResult(qcten=qcten, ncten=ncten, rc=rc, nc=nc)


def thompson_ice_sedimentation_setup(
    state: ThompsonPostTendencyState,
    rain_setup: ThompsonRainSedimentationSetupResult,
    dt: Any,
    dzq: Any,
) -> ThompsonIceSedimentationSetupResult:
    """Compute WRF Thompson cloud-ice fall speeds and sedimentation substeps."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    ri = jnp.asarray(state.ri, dtype=jnp.float32)
    ni = jnp.asarray(state.ni, dtype=jnp.float32)
    nlev = ri.shape[0]
    zeros = jnp.zeros_like(ri)
    active_any = jnp.any(state.l_qi)
    lami = (AM_I * CIG[1] * OIG1 * ni / ri) ** OBMI
    ilami = jnp.float32(1.0) / lami
    vti_mass = rain_setup.rhof * AV_I * CIG[2] * OIG2 * ilami**BV_I
    vti_number = rain_setup.rhof * AV_I * CIG[5] / CIG[6] * ilami**BV_I
    (vtik, vtnik), nstep_scan, ksed_scan = _sedimentation_profile_setup(
        (vti_mass, vti_number), ri > R1, dzq_arr, dt_arr,
    )
    ksed1 = jnp.where((ksed_scan == nlev - 1) & (nlev > 1), jnp.asarray(nlev - 2, dtype=jnp.int32), ksed_scan)
    onstep = jnp.where(nstep_scan > 0, jnp.float32(1.0) / nstep_scan.astype(jnp.float32), jnp.float32(1.0))
    return ThompsonIceSedimentationSetupResult(
        vtik=jnp.where(active_any, vtik, zeros),
        vtnik=jnp.where(active_any, vtnik, zeros),
        ksed1=jnp.where(active_any, ksed1, jnp.asarray(0, dtype=jnp.int32)),
        nstep=jnp.where(active_any, nstep_scan, jnp.asarray(0, dtype=jnp.int32)),
        onstep=jnp.where(active_any, onstep, jnp.float32(1.0)),
    )


def thompson_ice_sedimentation(
    state: ThompsonPostTendencyState,
    tendencies: ThompsonTendencyResult,
    rain: ThompsonRainEvaporationResult,
    setup: ThompsonIceSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    pptice0: Any = 0.0,
) -> ThompsonIceSedimentationResult:
    """Apply WRF Thompson cloud-ice sedimentation flux updates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    odzq = jnp.float32(1.0) / dzq_arr
    orho = jnp.float32(1.0) / jnp.asarray(rain.rho, dtype=jnp.float32)
    indices = jnp.arange(jnp.asarray(state.ri).shape[0], dtype=jnp.int32)
    top = jnp.asarray(jnp.asarray(state.ri).shape[0] - 1, dtype=jnp.int32)
    bottom = jnp.asarray(0, dtype=jnp.int32)
    onstep = jnp.asarray(setup.onstep, dtype=jnp.float32)
    nsteps = _fortran_nint(jnp.float32(1.0) / onstep)

    def _substep(_, carry: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
        qiten, niten, ri, ni, pptice = carry
        sed_i = setup.vtik * ri
        sed_n = setup.vtnik * ni
        sed_i_next = jnp.concatenate([sed_i[1:], jnp.zeros((1,), dtype=sed_i.dtype)])
        sed_n_next = jnp.concatenate([sed_n[1:], jnp.zeros((1,), dtype=sed_n.dtype)])
        column_mask = indices <= setup.ksed1
        top_mask = indices == top
        mass_tendency = jnp.where(column_mask, (sed_i_next - sed_i) * odzq * onstep * orho, jnp.zeros_like(ri))
        number_tendency = jnp.where(column_mask, (sed_n_next - sed_n) * odzq * onstep * orho, jnp.zeros_like(ni))
        mass_tendency = mass_tendency + jnp.where(top_mask, -sed_i * odzq * onstep * orho, jnp.zeros_like(ri))
        number_tendency = number_tendency + jnp.where(top_mask, -sed_n * odzq * onstep * orho, jnp.zeros_like(ni))
        mass_state = jnp.where(column_mask, (sed_i_next - sed_i) * odzq * dt_arr * onstep, jnp.zeros_like(ri))
        number_state = jnp.where(column_mask, (sed_n_next - sed_n) * odzq * dt_arr * onstep, jnp.zeros_like(ni))
        mass_state = mass_state + jnp.where(top_mask, -sed_i * odzq * dt_arr * onstep, jnp.zeros_like(ri))
        number_state = number_state + jnp.where(top_mask, -sed_n * odzq * dt_arr * onstep, jnp.zeros_like(ni))
        qiten = qiten + mass_tendency
        niten = niten + number_tendency
        ri = jnp.maximum(R1, ri + mass_state)
        ni = jnp.maximum(R2, ni + number_state)
        pptice = jnp.where(ri[bottom] > R1 * jnp.float32(1000.0), pptice + sed_i[bottom] * dt_arr * onstep, pptice)
        return qiten, niten, ri, ni, pptice

    qiten0 = jnp.asarray(tendencies.qiten, dtype=jnp.float32)
    niten0 = jnp.asarray(tendencies.niten, dtype=jnp.float32)
    ri0 = jnp.asarray(state.ri, dtype=jnp.float32)
    ni0 = jnp.asarray(state.ni, dtype=jnp.float32)
    pptice_init = jnp.asarray(pptice0, dtype=jnp.float32)
    qiten, niten, ri, ni, pptice = dynamic_fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        nsteps,
        _substep,
        (qiten0, niten0, ri0, ni0, pptice_init),
    )
    return ThompsonIceSedimentationResult(qiten=qiten, niten=niten, ri=ri, ni=ni, pptice=pptice)


def _snow_fall_speed_formula(smob, smoc, rhof, rs, rr, vtr, boost, melting):
    xds = smoc / jnp.maximum(smob, jnp.float32(1.0e-30))
    mrat = jnp.float32(1.0) / xds
    ils1 = jnp.float32(1.0) / (mrat * LAM0 + FV_S)
    ils2 = jnp.float32(1.0) / (mrat * LAM1 + FV_S)
    t1_vts = KAP0 * CSG[3] * ils1 ** CSE[3]
    t2_vts = KAP1 * mrat**MU_S * CSG[9] * ils2 ** CSE[9]
    ils1_clear = jnp.float32(1.0) / (mrat * LAM0)
    ils2_clear = jnp.float32(1.0) / (mrat * LAM1)
    t3_vts = KAP0 * CSG[0] * ils1_clear ** CSE[0]
    t4_vts = KAP1 * mrat**MU_S * CSG[6] * ils2_clear ** CSE[6]
    vts = rhof * AV_S * (t1_vts + t2_vts) / (t3_vts + t4_vts)
    sr = rs / (rs + rr)
    vts_melt = vts * sr + (jnp.float32(1.0) - sr) * vtr
    return jnp.where(melting, vts_melt, vts * boost)


@custom_jvp
def _snow_fall_speed(smob, smoc, rhof, rs, rr, vtr, boost, melting):
    return _snow_fall_speed_formula(smob, smoc, rhof, rs, rr, vtr, boost, melting)


@_snow_fall_speed.defjvp
def _snow_fall_speed_jvp(primals, tangents):
    present = primals[3] > R1
    # Only the enabled branch has a snow size distribution. Keeping this
    # protection in the derivative retains the forecast's arithmetic order.
    safe = tuple(jnp.where(present, value, jnp.ones_like(value)) if i < 2 else value
                 for i, value in enumerate(primals))
    ds = tuple(jnp.where(present, value, jnp.zeros_like(value)) if i < 2 else value
               for i, value in enumerate(tangents))
    _, derivative = jvp(_snow_fall_speed_formula, safe, ds)
    return _snow_fall_speed(*primals), derivative


def thompson_snow_sedimentation_setup(
    state: ThompsonPostTendencyState,
    rain: ThompsonRainEvaporationResult,
    rain_setup: ThompsonRainSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    prr_sml: Any,
    vts_boost: Any,
    iiwarm: bool = False,
) -> ThompsonSnowSedimentationSetupResult:
    """Compute WRF Thompson snow fall speeds and sedimentation substeps."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    rs = jnp.asarray(state.rs, dtype=jnp.float32)
    prr_sml_arr = jnp.asarray(prr_sml, dtype=jnp.float32)
    vts_boost_arr = jnp.asarray(vts_boost, dtype=jnp.float32)
    nlev = rs.shape[0]
    indices = jnp.arange(nlev - 1, -1, -1, dtype=jnp.int32)
    zeros = jnp.zeros_like(rs)
    active_any = (~bool(iiwarm)) & jnp.any(state.l_qs)

    # Keep snow's original evaluation order; parallel evaluation changed CPU rounding.
    def _step(carry: tuple[Any, Any, Any], idx: Any) -> tuple[tuple[Any, Any, Any], Any]:
        next_vts, nstep_carry, ksed_carry = carry
        rs_k = rs[idx]
        vtsk_active = _snow_fall_speed(
            state.smob[idx], state.smoc[idx], rain_setup.rhof[idx], rs_k,
            rain.rr[idx], rain_setup.vtrk[idx], vts_boost_arr[idx],
            prr_sml_arr[idx] > jnp.float32(0),
        )
        vtsk_k = jnp.where(rs_k > R1, vtsk_active, next_vts)
        sedimenting = vtsk_k > jnp.float32(1.0e-3)
        delta_tp = dzq_arr[idx] / jnp.maximum(vtsk_k, jnp.float32(1.0e-30))
        nstep_here = (dt_arr / delta_tp + jnp.float32(1.0)).astype(jnp.int32)
        nstep_next = jnp.where(sedimenting, jnp.maximum(nstep_carry, nstep_here), nstep_carry)
        ksed_next = jnp.where(sedimenting, jnp.maximum(ksed_carry, idx), ksed_carry)
        return (vtsk_k, nstep_next, ksed_next), vtsk_k

    (_, nstep_scan, ksed_scan), vtsk_rev = lax.scan(
        _step,
        (jnp.float32(0.0), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        indices,
        unroll=4,
    )
    vtsk = vtsk_rev[::-1]
    ksed1 = jnp.where((ksed_scan == nlev - 1) & (nlev > 1), jnp.asarray(nlev - 2, dtype=jnp.int32), ksed_scan)
    onstep = jnp.where(nstep_scan > 0, jnp.float32(1.0) / nstep_scan.astype(jnp.float32), jnp.float32(1.0))
    return ThompsonSnowSedimentationSetupResult(
        vtsk=jnp.where(active_any, vtsk, zeros),
        ksed1=jnp.where(active_any, ksed1, jnp.asarray(0, dtype=jnp.int32)),
        nstep=jnp.where(active_any, nstep_scan, jnp.asarray(0, dtype=jnp.int32)),
        onstep=jnp.where(active_any, onstep, jnp.float32(1.0)),
    )


def thompson_snow_sedimentation(
    state: ThompsonPostTendencyState,
    tendencies: ThompsonTendencyResult,
    rain: ThompsonRainEvaporationResult,
    setup: ThompsonSnowSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    pptsnow0: Any = 0.0,
) -> ThompsonSnowSedimentationResult:
    """Apply WRF Thompson snow sedimentation flux updates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    odzq = jnp.float32(1.0) / dzq_arr
    orho = jnp.float32(1.0) / jnp.asarray(rain.rho, dtype=jnp.float32)
    indices = jnp.arange(jnp.asarray(state.rs).shape[0], dtype=jnp.int32)
    top = jnp.asarray(jnp.asarray(state.rs).shape[0] - 1, dtype=jnp.int32)
    bottom = jnp.asarray(0, dtype=jnp.int32)
    onstep = jnp.asarray(setup.onstep, dtype=jnp.float32)
    nsteps = _fortran_nint(jnp.float32(1.0) / onstep)

    def _substep(_, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        qsten, rs, pptsnow = carry
        sed_s = setup.vtsk * rs
        sed_s_next = jnp.concatenate([sed_s[1:], jnp.zeros((1,), dtype=sed_s.dtype)])
        column_mask = indices <= setup.ksed1
        top_mask = indices == top
        mass_tendency = jnp.where(column_mask, (sed_s_next - sed_s) * odzq * onstep * orho, jnp.zeros_like(rs))
        mass_tendency = mass_tendency + jnp.where(top_mask, -sed_s * odzq * onstep * orho, jnp.zeros_like(rs))
        mass_state = jnp.where(column_mask, (sed_s_next - sed_s) * odzq * dt_arr * onstep, jnp.zeros_like(rs))
        mass_state = mass_state + jnp.where(top_mask, -sed_s * odzq * dt_arr * onstep, jnp.zeros_like(rs))
        qsten = qsten + mass_tendency
        rs = jnp.maximum(R1, rs + mass_state)
        pptsnow = jnp.where(rs[bottom] > R1 * jnp.float32(1000.0), pptsnow + sed_s[bottom] * dt_arr * onstep, pptsnow)
        return qsten, rs, pptsnow

    qsten0 = jnp.asarray(tendencies.qsten, dtype=jnp.float32)
    rs0 = jnp.asarray(state.rs, dtype=jnp.float32)
    pptsnow_init = jnp.asarray(pptsnow0, dtype=jnp.float32)
    qsten, rs, pptsnow = dynamic_fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        nsteps,
        _substep,
        (qsten0, rs0, pptsnow_init),
    )
    return ThompsonSnowSedimentationResult(qsten=qsten, rs=rs, pptsnow=pptsnow)


def thompson_graupel_sedimentation_setup(
    state: ThompsonPostTendencyState,
    rain: ThompsonRainEvaporationResult,
    rain_setup: ThompsonRainSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    is_hail_aware: bool = False,
) -> ThompsonGraupelSedimentationSetupResult:
    """Compute WRF Thompson graupel/hail fall speeds and sedimentation substeps."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    rg = jnp.asarray(state.rg, dtype=jnp.float32)
    rb = jnp.asarray(state.rb, dtype=jnp.float32)
    rho = jnp.asarray(rain.rho, dtype=jnp.float32)
    visco = jnp.asarray(rain.visco, dtype=jnp.float32)
    rhof = jnp.asarray(rain_setup.rhof, dtype=jnp.float32)
    ilamg = jnp.asarray(state.ilamg, dtype=jnp.float32)
    idx_bg = jnp.asarray(state.idx_bg, dtype=jnp.int32)
    nlev = rg.shape[0]
    zeros = jnp.zeros_like(rg)
    active_any = jnp.any(state.l_qg)
    rho_g_table = jnp.asarray(RHO_G, dtype=jnp.float32)
    cgg_table = jnp.where(bool(is_hail_aware), CGG, CGG_NONHAIL)
    bfall = jnp.where(bool(is_hail_aware), jnp.float32(3.0) * B_COEFF - jnp.float32(1.0), BV_G_OLD)

    xrho_g = jnp.maximum(rho_g_table[0], jnp.minimum(rg / rho / rb, rho_g_table[-1]))
    afall_hail = A_COEFF * ((jnp.float32(4.0) * xrho_g * jnp.float32(9.8)) / (jnp.float32(3.0) * rho)) ** B_COEFF
    afall_hail = afall_hail * visco ** (jnp.float32(1.0) - jnp.float32(2.0) * B_COEFF)
    afall = jnp.where(bool(is_hail_aware), afall_hail, AV_G_OLD)
    vtg_mass = rhof * afall * cgg_table[5, idx_bg] * OGG3 * ilamg ** bfall
    vtg_number_mu0 = rhof * afall * cgg_table[6, idx_bg] / cgg_table[11, idx_bg] * ilamg ** bfall
    vtg_number_mug = rhof * afall * cgg_table[7, idx_bg] * OGG2 * ilamg ** bfall
    vtg_number = jnp.where(MU_G == jnp.float32(0.0), vtg_number_mu0, vtg_number_mug)
    (vtgk, vtngk), nstep_scan, ksed_scan = _sedimentation_profile_setup(
        (vtg_mass, vtg_number), rg > R1, dzq_arr, dt_arr,
    )
    ksed1 = jnp.where((ksed_scan == nlev - 1) & (nlev > 1), jnp.asarray(nlev - 2, dtype=jnp.int32), ksed_scan)
    onstep = jnp.where(nstep_scan > 0, jnp.float32(1.0) / nstep_scan.astype(jnp.float32), jnp.float32(1.0))
    return ThompsonGraupelSedimentationSetupResult(
        vtgk=jnp.where(active_any, vtgk, zeros),
        vtngk=jnp.where(active_any, vtngk, zeros),
        ksed1=jnp.where(active_any, ksed1, jnp.asarray(0, dtype=jnp.int32)),
        nstep=jnp.where(active_any, nstep_scan, jnp.asarray(0, dtype=jnp.int32)),
        onstep=jnp.where(active_any, onstep, jnp.float32(1.0)),
    )


def thompson_graupel_sedimentation(
    state: ThompsonPostTendencyState,
    tendencies: ThompsonTendencyResult,
    rain: ThompsonRainEvaporationResult,
    setup: ThompsonGraupelSedimentationSetupResult,
    dt: Any,
    dzq: Any,
    *,
    pptgraul0: Any = 0.0,
) -> ThompsonGraupelSedimentationResult:
    """Apply WRF Thompson graupel/hail sedimentation flux updates."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    dzq_arr = jnp.asarray(dzq, dtype=jnp.float32)
    odzq = jnp.float32(1.0) / dzq_arr
    rho = jnp.asarray(rain.rho, dtype=jnp.float32)
    orho = jnp.float32(1.0) / rho
    active_any = jnp.any(state.l_qg)
    indices = jnp.arange(jnp.asarray(state.rg).shape[0], dtype=jnp.int32)
    top = jnp.asarray(jnp.asarray(state.rg).shape[0] - 1, dtype=jnp.int32)
    bottom = jnp.asarray(0, dtype=jnp.int32)
    onstep = jnp.asarray(setup.onstep, dtype=jnp.float32)
    nsteps = _fortran_nint(jnp.float32(1.0) / onstep)
    rb_min_top = R1 / rho / jnp.asarray(RHO_G[-1], dtype=jnp.float32)

    def _substep(_, carry: tuple[Any, Any, Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        qgten, ngten, qbten, rg, ng, rb, pptgraul = carry
        sed_g = setup.vtgk * rg
        sed_n = setup.vtngk * ng
        sed_b = setup.vtgk * rb
        sed_g_next = jnp.concatenate([sed_g[1:], jnp.zeros((1,), dtype=sed_g.dtype)])
        sed_n_next = jnp.concatenate([sed_n[1:], jnp.zeros((1,), dtype=sed_n.dtype)])
        sed_b_next = jnp.concatenate([sed_b[1:], jnp.zeros((1,), dtype=sed_b.dtype)])
        column_mask = indices <= setup.ksed1
        top_mask = indices == top
        mass_flux = (sed_g_next - sed_g) * odzq
        number_flux = (sed_n_next - sed_n) * odzq
        bulk_flux = (sed_b_next - sed_b) * odzq
        qgten = qgten + jnp.where(column_mask, mass_flux * onstep * orho, jnp.zeros_like(qgten))
        ngten = ngten + jnp.where(column_mask, number_flux * onstep * orho, jnp.zeros_like(ngten))
        qbten = qbten + jnp.where(column_mask, bulk_flux * onstep, jnp.zeros_like(qbten))
        qgten = qgten + jnp.where(top_mask, -sed_g * odzq * onstep * orho, jnp.zeros_like(qgten))
        ngten = ngten + jnp.where(top_mask, -sed_n * odzq * onstep * orho, jnp.zeros_like(ngten))
        qbten = qbten + jnp.where(top_mask, -sed_b * odzq * onstep, jnp.zeros_like(qbten))
        rg = jnp.maximum(R1, rg + jnp.where(column_mask, mass_flux * dt_arr * onstep, jnp.zeros_like(rg)))
        ng = jnp.maximum(R2, ng + jnp.where(column_mask, number_flux * dt_arr * onstep, jnp.zeros_like(ng)))
        rb = jnp.maximum(rg / rho / jnp.asarray(RHO_G[-1], dtype=jnp.float32), rb + jnp.where(column_mask, bulk_flux * dt_arr * onstep, jnp.zeros_like(rb)))
        rg = jnp.maximum(R1, rg + jnp.where(top_mask, -sed_g * odzq * dt_arr * onstep, jnp.zeros_like(rg)))
        ng = jnp.maximum(R2, ng + jnp.where(top_mask, -sed_n * odzq * dt_arr * onstep, jnp.zeros_like(ng)))
        rb = jnp.maximum(rb_min_top, rb + jnp.where(top_mask, -sed_b * odzq * dt_arr * onstep, jnp.zeros_like(rb)))
        pptgraul = jnp.where(rg[bottom] > R1 * jnp.float32(1000.0), pptgraul + sed_g[bottom] * dt_arr * onstep, pptgraul)
        return qgten, ngten, qbten, rg, ng, rb, pptgraul

    qgten0 = jnp.asarray(tendencies.qgten, dtype=jnp.float32)
    ngten0 = jnp.asarray(tendencies.ngten, dtype=jnp.float32)
    qbten0 = jnp.asarray(tendencies.qbten, dtype=jnp.float32)
    rg0 = jnp.asarray(state.rg, dtype=jnp.float32)
    ng0 = jnp.asarray(state.ng, dtype=jnp.float32)
    rb0 = jnp.asarray(state.rb, dtype=jnp.float32)
    pptgraul_init = jnp.asarray(pptgraul0, dtype=jnp.float32)
    qgten, ngten, qbten, rg, ng, rb, pptgraul = dynamic_fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        nsteps,
        _substep,
        (qgten0, ngten0, qbten0, rg0, ng0, rb0, pptgraul_init),
    )
    return ThompsonGraupelSedimentationResult(
        qgten=jnp.where(active_any, qgten, qgten0),
        ngten=jnp.where(active_any, ngten, ngten0),
        qbten=jnp.where(active_any, qbten, qbten0),
        rg=jnp.where(active_any, rg, rg0),
        ng=jnp.where(active_any, ng, ng0),
        rb=jnp.where(active_any, rb, rb0),
        pptgraul=jnp.where(active_any, pptgraul, pptgraul_init),
    )


def thompson_instant_phase_change(
    pre: ThompsonPreprocessResult,
    state: ThompsonPostTendencyState | ThompsonRainEvaporationResult,
    qcten: Any,
    ncten: Any,
    qiten: Any,
    niten: Any,
    tten: Any,
    dt: Any,
    *,
    iiwarm: bool = False,
    ifdry: int = 0,
) -> ThompsonInstantPhaseChangeResult:
    """Apply WRF Thompson instant cloud-ice melt and cloud-water freeze update."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    odt = jnp.float32(1.0) / dt_arr
    qcten_arr = jnp.asarray(qcten, dtype=jnp.float32)
    ncten_arr = jnp.asarray(ncten, dtype=jnp.float32)
    qiten_arr = jnp.asarray(qiten, dtype=jnp.float32)
    niten_arr = jnp.asarray(niten, dtype=jnp.float32)
    tten_arr = jnp.asarray(tten, dtype=jnp.float32)
    active = ~bool(iiwarm)
    heat_factor = jnp.float32(1 - int(ifdry))

    xri = jnp.maximum(jnp.float32(0.0), _wrf_real_mul_add(qiten_arr, dt_arr, pre.qi1d))
    melt_ice = active & (jnp.asarray(state.temp, dtype=jnp.float32) > jnp.float32(273.15)) & (xri > jnp.float32(0.0))
    qcten_melt = qcten_arr + jnp.where(melt_ice, xri * odt, jnp.zeros_like(qcten_arr))
    ncten_melt = ncten_arr + jnp.where(melt_ice, jnp.asarray(pre.ni1d, dtype=jnp.float32) * odt, jnp.zeros_like(ncten_arr))
    qiten_melt = qiten_arr - jnp.where(melt_ice, xri * odt, jnp.zeros_like(qiten_arr))
    niten_melt = jnp.where(melt_ice, -jnp.asarray(pre.ni1d, dtype=jnp.float32) * odt, niten_arr)
    tten_melt = tten_arr - jnp.where(melt_ice, LFUS * jnp.asarray(state.ocp, dtype=jnp.float32) * xri * odt * heat_factor, jnp.zeros_like(tten_arr))

    xrc = jnp.maximum(jnp.float32(0.0), _wrf_real_mul_add(qcten_melt, dt_arr, pre.qc1d))
    xrc = jnp.where(xrc > R1, xrc, jnp.zeros_like(xrc))
    freeze_cloud = active & (jnp.asarray(state.temp, dtype=jnp.float32) < HGFR) & (xrc > jnp.float32(0.0))
    lfus2 = LSUB - jnp.asarray(state.lvap, dtype=jnp.float32)
    xnc = _wrf_real_mul_add(ncten_melt, dt_arr, pre.nc1d)
    qiten_out = qiten_melt + jnp.where(freeze_cloud, xrc * odt, jnp.zeros_like(qiten_melt))
    niten_out = niten_melt + jnp.where(freeze_cloud, xnc * odt, jnp.zeros_like(niten_melt))
    qcten_out = qcten_melt - jnp.where(freeze_cloud, xrc * odt, jnp.zeros_like(qcten_melt))
    ncten_out = ncten_melt - jnp.where(freeze_cloud, xnc * odt, jnp.zeros_like(ncten_melt))
    tten_out = tten_melt + jnp.where(freeze_cloud, lfus2 * jnp.asarray(state.ocp, dtype=jnp.float32) * xrc * odt * heat_factor, jnp.zeros_like(tten_melt))

    return ThompsonInstantPhaseChangeResult(qcten=qcten_out, ncten=ncten_out, qiten=qiten_out, niten=niten_out, tten=tten_out)


def thompson_final_passback(
    pre: ThompsonPreprocessResult,
    tendencies: ThompsonTendencyResult,
    rho: Any,
    dt: Any,
    *,
    phase: ThompsonInstantPhaseChangeResult | None = None,
    is_aerosol_aware: bool = False,
    is_hail_aware: bool = False,
    aer_init_opt: int = 0,
    wif_input_opt: int = 1,
) -> ThompsonFinalPassbackResult:
    """Apply WRF Thompson final passback corrections to parent 1D fields."""

    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    rho_arr = jnp.asarray(rho, dtype=jnp.float32)
    qcten = jnp.asarray(tendencies.qcten if phase is None else phase.qcten, dtype=jnp.float32)
    ncten = jnp.asarray(tendencies.ncten if phase is None else phase.ncten, dtype=jnp.float32)
    qiten = jnp.asarray(tendencies.qiten if phase is None else phase.qiten, dtype=jnp.float32)
    niten = jnp.asarray(tendencies.niten if phase is None else phase.niten, dtype=jnp.float32)
    tten = jnp.asarray(tendencies.tten if phase is None else phase.tten, dtype=jnp.float32)

    t1d = jnp.asarray(pre.temp, dtype=jnp.float32) + tten * dt_arr
    qv1d = jnp.maximum(jnp.float32(1.0e-10), jnp.asarray(pre.qv1d, dtype=jnp.float32) + jnp.asarray(tendencies.qvten, dtype=jnp.float32) * dt_arr)
    qc1d_raw = jnp.asarray(pre.qc1d, dtype=jnp.float32) + qcten * dt_arr
    nc1d_raw = jnp.maximum(jnp.float32(2.0) / rho_arr, jnp.minimum(jnp.asarray(pre.nc1d, dtype=jnp.float32) + ncten * dt_arr, NT_C_MAX))

    nwfa_base = jnp.asarray(pre.nwfa, dtype=jnp.float32) / jnp.asarray(pre.rho, dtype=jnp.float32)
    nifa_base = jnp.asarray(pre.nifa, dtype=jnp.float32) / jnp.asarray(pre.rho, dtype=jnp.float32)
    nbca_base = jnp.asarray(pre.nbca, dtype=jnp.float32) / jnp.asarray(pre.rho, dtype=jnp.float32)
    nwfa_update = nwfa_base + jnp.asarray(tendencies.nwfaten, dtype=jnp.float32) * dt_arr
    nifa_update = nifa_base + jnp.asarray(tendencies.nifaten, dtype=jnp.float32) * dt_arr
    nbca_update = nbca_base + jnp.asarray(tendencies.nbcaten, dtype=jnp.float32) * dt_arr
    if is_aerosol_aware and aer_init_opt >= 2:
        nwfa1d = jnp.maximum(jnp.float32(0.0), nwfa_update)
        nifa1d = jnp.maximum(jnp.float32(0.0), nifa_update)
        nbca1d = jnp.where(wif_input_opt == 2, jnp.maximum(jnp.float32(0.0), nbca_update), jnp.zeros_like(nbca_update))
    else:
        nwfa1d = jnp.maximum(jnp.float32(11.1e6), jnp.minimum(jnp.float32(9999.0e6), nwfa_update))
        nifa1d = jnp.maximum(NA_IN1 * jnp.float32(0.01), jnp.minimum(jnp.float32(9999.0e6), nifa_update))
        nbca_bounded = jnp.maximum(jnp.float32(5.55e6), jnp.minimum(jnp.float32(9999.0e6), nbca_update))
        nbca1d = jnp.where((not is_aerosol_aware) | (wif_input_opt == 2), nbca_bounded, jnp.zeros_like(nbca_bounded))

    qc_active = qc1d_raw > R1
    nu_c = jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / (nc1d_raw * rho_arr)) + jnp.asarray(2, dtype=jnp.int32))
    nu_idx = nu_c - jnp.asarray(1, dtype=jnp.int32)
    qc_for_size = jnp.where(qc_active, qc1d_raw, jnp.ones_like(qc1d_raw))
    lamc = (AM_R * CCG[1, nu_idx] * OCG1[nu_idx] * nc1d_raw / qc_for_size) ** OBMR
    xdc = (BM_R + nu_c.astype(jnp.float32) + jnp.float32(1.0)) / lamc
    lamc = jnp.where(xdc < D0C, CCE[1, nu_idx] / D0C, jnp.where(xdc > D0R * jnp.float32(2.0), CCE[1, nu_idx] / (D0R * jnp.float32(2.0)), lamc))
    nc_corrected = jnp.minimum(CCG[0, nu_idx] * OCG2[nu_idx] * qc1d_raw / AM_R * lamc**BM_R, NT_C_MAX / rho_arr)
    qc1d = jnp.where(qc_active, qc1d_raw, jnp.zeros_like(qc1d_raw))
    nc1d = jnp.where(qc_active, nc_corrected, jnp.zeros_like(nc1d_raw))

    qi1d_raw = _wrf_real_mul_add(qiten, dt_arr, pre.qi1d)
    ni1d_raw = jnp.maximum(R2 / rho_arr, _wrf_real_mul_add(niten, dt_arr, pre.ni1d))
    qi_active = qi1d_raw > R1
    qi_for_size = jnp.where(qi_active, qi1d_raw, jnp.ones_like(qi1d_raw))
    lami, xdi = _ice_lami_xdi(ni1d_raw, qi_for_size)
    lami = jnp.where(xdi < jnp.float32(5.0e-6), _ice_lami_for_diameter(5.0e-6), jnp.where(xdi > jnp.float32(300.0e-6), _ice_lami_for_diameter(300.0e-6), lami))
    ni_corrected = jnp.minimum(_ice_number_from_lami(qi1d_raw, lami), jnp.asarray(_f64(999.0e3) / _f64(rho_arr), dtype=jnp.float32))
    qi1d = jnp.where(qi_active, qi1d_raw, jnp.zeros_like(qi1d_raw))
    ni1d = jnp.where(qi_active, ni_corrected, jnp.zeros_like(ni1d_raw))

    qr1d_raw = jnp.asarray(pre.qr1d, dtype=jnp.float32) + jnp.asarray(tendencies.qrten, dtype=jnp.float32) * dt_arr
    nr1d_raw = jnp.maximum(R2 / rho_arr, jnp.asarray(pre.nr1d, dtype=jnp.float32) + jnp.asarray(tendencies.nrten, dtype=jnp.float32) * dt_arr)
    qr_active = qr1d_raw > R1
    qr_for_size = jnp.where(qr_active, qr1d_raw, jnp.ones_like(qr1d_raw))
    lamr = (AM_R * CRG[2] * ORG2 * nr1d_raw / qr_for_size) ** OBMR
    mvd_r = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / lamr
    mvd_r = jnp.where(mvd_r > jnp.float32(2.5e-3), jnp.float32(2.5e-3), jnp.where(mvd_r < D0R * jnp.float32(0.75), D0R * jnp.float32(0.75), mvd_r))
    lamr = (jnp.float32(3.0) + MU_R + jnp.float32(0.672)) / mvd_r
    nr_corrected = CRG[1] * ORG3 * qr1d_raw * lamr**BM_R / AM_R
    qr1d = jnp.where(qr_active, qr1d_raw, jnp.zeros_like(qr1d_raw))
    nr1d = jnp.where(qr_active, nr_corrected, jnp.zeros_like(nr1d_raw))
    mvd_r = jnp.where(qr_active, mvd_r, jnp.asarray(tendencies.mvd_r, dtype=jnp.float32))

    qs1d_raw = jnp.asarray(pre.qs1d, dtype=jnp.float32) + jnp.asarray(tendencies.qsten, dtype=jnp.float32) * dt_arr
    qs1d = jnp.where(qs1d_raw > R1, qs1d_raw, jnp.zeros_like(qs1d_raw))

    rho_g_table = jnp.asarray(RHO_G, dtype=jnp.float32)
    am_g_table = jnp.asarray(AM_G, dtype=jnp.float32)
    qg1d_raw = jnp.asarray(pre.qg1d, dtype=jnp.float32) + jnp.asarray(tendencies.qgten, dtype=jnp.float32) * dt_arr
    ng1d_raw = jnp.maximum(R2 / rho_arr, jnp.asarray(pre.ng1d, dtype=jnp.float32) + jnp.asarray(tendencies.ngten, dtype=jnp.float32) * dt_arr)
    qb1d_raw = jnp.maximum(qg1d_raw / rho_g_table[-1], jnp.asarray(pre.qb1d, dtype=jnp.float32) + jnp.asarray(tendencies.qbten, dtype=jnp.float32) * dt_arr)
    qb1d_raw = jnp.minimum(qg1d_raw / rho_g_table[0], qb1d_raw)
    idx_bg = jnp.clip(_fortran_nint(qg1d_raw / qb1d_raw * jnp.float32(0.01)) + jnp.asarray(1, dtype=jnp.int32), 1, RHO_G.size) - 1
    idx_bg = jnp.where(is_hail_aware, idx_bg, jnp.asarray(IDX_BG1, dtype=jnp.int32))
    qg_active = qg1d_raw > R1
    am_g_idx = jnp.take(am_g_table, idx_bg)
    qg_for_size = jnp.where(qg_active, qg1d_raw, jnp.ones_like(qg1d_raw))
    lamg = (am_g_idx * CGG[2, 0] * OGG2 * ng1d_raw / qg_for_size) ** OBMG
    mvd_g = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / lamg
    mvd_g = jnp.where(mvd_g > jnp.float32(25.4e-3), jnp.float32(25.4e-3), jnp.where(mvd_g < D0R, D0R, mvd_g))
    lamg = (jnp.float32(3.0) + MU_G + jnp.float32(0.672)) / mvd_g
    ng_corrected = CGG[1, 0] * OGG3 * qg1d_raw * lamg**BM_G / am_g_idx
    qg1d = jnp.where(qg_active, qg1d_raw, jnp.zeros_like(qg1d_raw))
    ng1d = jnp.where(qg_active, ng_corrected, jnp.zeros_like(ng1d_raw))
    qb1d = jnp.where(qg_active, qb1d_raw, jnp.zeros_like(qb1d_raw))
    idx_bg = jnp.where(qg_active, idx_bg, jnp.asarray(IDX_BG1, dtype=jnp.int32))
    mvd_g = jnp.where(qg_active, mvd_g, jnp.asarray(tendencies.mvd_g, dtype=jnp.float32))

    return ThompsonFinalPassbackResult(
        t1d=t1d,
        qv1d=qv1d,
        qc1d=qc1d,
        nc1d=nc1d,
        nwfa1d=nwfa1d,
        nifa1d=nifa1d,
        nbca1d=nbca1d,
        qi1d=qi1d,
        ni1d=ni1d,
        qr1d=qr1d,
        nr1d=nr1d,
        mvd_r=mvd_r,
        qs1d=qs1d,
        qg1d=qg1d,
        ng1d=ng1d,
        qb1d=qb1d,
        idx_bg=idx_bg,
        mvd_g=mvd_g,
    )


def thompson_grid_driver_defaults(
    qv1d: Any,
    qg1d: Any,
    t1d: Any,
    p1d: Any,
) -> ThompsonGridDriverDefaults:
    """Replicate WRF ``mp_gt_driver`` defaults before ``mp_thompson``.

    This is the non-aerosol-aware, non-hail-aware CONUS path where WRF does not
    pass optional ``nc/nwfa/nifa/nbca/ng/qb`` state arrays into the driver.
    """

    qv = jnp.maximum(jnp.float32(1.0e-10), jnp.asarray(qv1d, dtype=jnp.float32))
    qg = jnp.asarray(qg1d, dtype=jnp.float32)
    temp = jnp.asarray(t1d, dtype=jnp.float32)
    pres = jnp.asarray(p1d, dtype=jnp.float32)
    rho = jnp.float32(0.622) * pres / (R * temp * (qv + jnp.float32(0.622)))
    nc1d = NT_C / rho
    nwfa1d = jnp.float32(11.1e6) / rho
    nifa1d = NA_IN1 * jnp.float32(0.01) / rho
    nbca1d = jnp.float32(5.55e6) / rho

    qg_active = qg > R1
    am_g_bg1 = jnp.asarray(AM_G, dtype=jnp.float32)[IDX_BG1]
    qg_safe = jnp.where(qg_active, qg, jnp.ones_like(qg))
    ygra1 = jnp.log10(jnp.maximum(jnp.float32(1.0e-9), qg_safe * rho))
    zans1 = jnp.maximum(
        jnp.float32(2.0),
        jnp.minimum(jnp.float32(3.0) + jnp.float32(2.0) / jnp.float32(7.0) * (ygra1 + jnp.float32(8.0)), jnp.float32(6.0)),
    )
    n0_exp = jnp.float32(10.0) ** zans1
    lam_exp = (n0_exp * am_g_bg1 * CGG[0, 0] / (rho * qg_safe)) ** OGE1
    lamg = lam_exp * (CGG[2, 0] * OGG2 * OGG1) ** OBMG
    ng_active = CGG[1, 0] * OGG3 * rho * qg_safe * lamg**BM_G / am_g_bg1
    ng1d = jnp.where(qg_active, jnp.maximum(R2, ng_active / rho), jnp.zeros_like(qg))
    qb1d = jnp.where(qg_active, qg / jnp.asarray(RHO_G, dtype=jnp.float32)[IDX_BG1], jnp.zeros_like(qg))
    return ThompsonGridDriverDefaults(
        nc1d=nc1d,
        nwfa1d=nwfa1d,
        nifa1d=nifa1d,
        nbca1d=nbca1d,
        ng1d=ng1d,
        qb1d=qb1d,
    )


def _thompson_no_micro_return(
    pre: ThompsonPreprocessResult,
    *,
    qv1d: Any,
    nwfa1d: Any,
    nifa1d: Any,
    nbca1d: Any,
    t1d: Any,
) -> ThompsonColumnDriverResult:
    """Return the parent arrays preserved by WRF's early ``no_micro`` exit."""

    zeros = jnp.zeros_like(jnp.asarray(pre.temp, dtype=jnp.float32))
    final = ThompsonFinalPassbackResult(
        t1d=jnp.asarray(t1d, dtype=jnp.float32),
        qv1d=jnp.asarray(qv1d, dtype=jnp.float32),
        qc1d=jnp.asarray(pre.qc1d, dtype=jnp.float32),
        nc1d=jnp.asarray(pre.nc1d, dtype=jnp.float32),
        nwfa1d=jnp.asarray(nwfa1d, dtype=jnp.float32),
        nifa1d=jnp.asarray(nifa1d, dtype=jnp.float32),
        nbca1d=jnp.asarray(nbca1d, dtype=jnp.float32),
        qi1d=jnp.asarray(pre.qi1d, dtype=jnp.float32),
        ni1d=jnp.asarray(pre.ni1d, dtype=jnp.float32),
        qr1d=jnp.asarray(pre.qr1d, dtype=jnp.float32),
        nr1d=jnp.asarray(pre.nr1d, dtype=jnp.float32),
        mvd_r=jnp.asarray(pre.mvd_r, dtype=jnp.float32),
        qs1d=jnp.asarray(pre.qs1d, dtype=jnp.float32),
        qg1d=jnp.asarray(pre.qg1d, dtype=jnp.float32),
        ng1d=jnp.asarray(pre.ng1d, dtype=jnp.float32),
        qb1d=jnp.asarray(pre.qb1d, dtype=jnp.float32),
        idx_bg=jnp.asarray(pre.idx_bg),
        mvd_g=jnp.asarray(pre.mvd_g, dtype=jnp.float32),
    )
    tendencies = ThompsonTendencyResult(
        nwfaten=zeros,
        nifaten=zeros,
        nbcaten=zeros,
        qvten=zeros,
        qcten=zeros,
        ncten=zeros,
        qiten=zeros,
        niten=zeros,
        qrten=zeros,
        nrten=zeros,
        qsten=zeros,
        qgten=zeros,
        ngten=zeros,
        qbten=zeros,
        tten=zeros,
        mvd_r=jnp.asarray(pre.mvd_r, dtype=jnp.float32),
        mvd_g=jnp.asarray(pre.mvd_g, dtype=jnp.float32),
    )
    zero_precip = jnp.asarray(0.0, dtype=jnp.float32)
    return ThompsonColumnDriverResult(
        final=final,
        tendencies=tendencies,
        pptrain=zero_precip,
        pptsnow=zero_precip,
        pptice=zero_precip,
        pptgraul=zero_precip,
    )


def thompson_column_driver(
    qv1d: Any,
    qc1d: Any,
    qi1d: Any,
    qr1d: Any,
    qs1d: Any,
    qg1d: Any,
    qb1d: Any,
    ni1d: Any,
    nr1d: Any,
    nc1d: Any,
    ng1d: Any,
    nwfa1d: Any,
    nifa1d: Any,
    nbca1d: Any,
    t1d: Any,
    p1d: Any,
    dzq: Any,
    w1d: Any,
    tables: ThompsonTables,
    dt: Any,
    *,
    aer_init_opt: int = 1,
    wif_input_opt: int = 1,
    is_aerosol_aware: bool = False,
    is_hail_aware: bool = False,
    dusty_ice: bool = True,
    homog_ice: bool = True,
    ifdry: int = 0,
    iiwarm: bool = False,
    qvs1d: Any | None = None,
    qvsi1d: Any | None = None,
    delqvs1d: Any | None = None,
) -> ThompsonColumnDriverResult:
    """Run the translated WRF Thompson microphysics for one vertical column."""

    pre = thompson_preprocess_column(
        qv1d,
        qc1d,
        qi1d,
        qr1d,
        qs1d,
        qg1d,
        qb1d,
        ni1d,
        nr1d,
        nc1d,
        ng1d,
        nwfa1d,
        nifa1d,
        nbca1d,
        t1d,
        p1d,
        qvs1d=qvs1d,
        qvsi1d=qvsi1d,
        delqvs1d=delqvs1d,
        aer_init_opt=aer_init_opt,
        wif_input_opt=wif_input_opt,
        is_aerosol_aware=is_aerosol_aware,
        is_hail_aware=is_hail_aware,
    )
    warm = thompson_warm_rain_terms(pre, table_efrw(), dt, wif_input_opt=wif_input_opt)
    lookup = thompson_frozen_lookup_indices(pre)
    frozen = thompson_frozen_collection_terms(pre, warm, lookup, table_efsw(), dt, wif_input_opt=wif_input_opt)
    rain_frozen = thompson_rain_frozen_collection_terms(pre, lookup, tables.qr_acr_qs, tables.qr_acr_qg, dt)
    freezing = thompson_freezing_terms(pre, lookup, tables.freeze_h2o, dt)
    ice_nuc = thompson_ice_nucleation_terms(pre, freezing, dt, is_aerosol_aware=is_aerosol_aware, dusty_ice=dusty_ice, homog_ice=homog_ice)
    ice_dep = thompson_ice_deposition_terms(pre, lookup, tables.qi_aut_qs, dt)
    snow_graupel_dep = thompson_snow_graupel_deposition_terms(pre, dt)
    ice_coll = thompson_ice_collection_terms(pre, dt)
    riming = thompson_riming_terms(pre, lookup, frozen.prs_scw, frozen.prg_gcw, snow_graupel_dep.prs_sde, warm.mvd_c)
    melting = thompson_melting_terms(pre, dt, rain_frozen.prr_rcs, riming.prs_scw, frozen.prg_gcw, warm.prr_rcw)

    prs_sde = snow_graupel_dep.prs_sde + melting.prs_sde
    prg_gde = snow_graupel_dep.prg_gde + melting.prg_gde
    png_gde = snow_graupel_dep.png_gde + melting.png_gde
    zero = jnp.zeros_like(pre.temp)
    depleted = thompson_depletion_rescale(
        pre,
        dt,
        pri_inu=ice_nuc.pri_inu,
        pri_ide=ice_dep.pri_ide,
        pni_ide=ice_dep.pni_ide,
        prs_ide=ice_dep.prs_ide,
        prs_sde=prs_sde,
        prg_gde=prg_gde,
        pri_iha=ice_nuc.pri_iha,
        prr_wau=warm.prr_wau,
        pri_wfz=freezing.pri_wfz,
        prr_rcw=melting.prr_rcw,
        prs_scw=melting.prs_scw,
        prg_scw=riming.prg_scw,
        prg_gcw=melting.prg_gcw,
        prs_iau=ice_dep.prs_iau,
        prs_sci=ice_coll.prs_sci,
        pri_rci=ice_coll.pri_rci,
        prg_rfz=freezing.prg_rfz,
        pri_rfz=freezing.pri_rfz,
        prr_rci=ice_coll.prr_rci,
        prr_rcs=rain_frozen.prr_rcs,
        prr_rcg=rain_frozen.prr_rcg,
        prs_ihm=riming.prs_ihm,
        prr_sml=melting.prr_sml,
        prs_rcs=rain_frozen.prs_rcs,
        prg_ihm=riming.prg_ihm,
        prr_gml=melting.prr_gml,
        prg_rcg=rain_frozen.prg_rcg,
    )
    tendencies = thompson_tendency_accumulation(
        pre,
        dt,
        depleted,
        pna_rca=warm.pna_rca,
        pna_sca=frozen.pna_sca,
        pna_gca=frozen.pna_gca,
        pnd_rcd=warm.pnd_rcd,
        pnd_scd=frozen.pnd_scd,
        pnd_gcd=frozen.pnd_gcd,
        pnb_rcb=warm.pnb_rcb,
        pnb_scb=frozen.pnb_scb,
        pnb_gcb=frozen.pnb_gcb,
        pni_inu=ice_nuc.pni_inu,
        pni_iha=ice_nuc.pni_iha,
        pni_ihm=riming.pni_ihm,
        pnc_wau=warm.pnc_wau,
        pnc_rcw=warm.pnc_rcw,
        pni_wfz=freezing.pni_wfz,
        pnc_scw=frozen.pnc_scw,
        pnc_gcw=frozen.pnc_gcw,
        pni_rfz=freezing.pni_rfz,
        pni_iau=ice_dep.pni_iau,
        pni_sci=ice_coll.pni_sci,
        pni_rci=ice_coll.pni_rci,
        pnr_wau=warm.pnr_wau,
        pnr_sml=melting.pnr_sml,
        pnr_gml=melting.pnr_gml,
        pnr_rfz=freezing.pnr_rfz,
        pnr_rcr=warm.pnr_rcr,
        pnr_rcg=rain_frozen.pnr_rcg,
        pnr_rcs=rain_frozen.pnr_rcs,
        pnr_rci=ice_coll.pnr_rci,
        prg_rci=ice_coll.prg_rci,
        prg_rcs=rain_frozen.prg_rcs,
        png_scw=riming.png_scw,
        png_rcg=rain_frozen.png_rcg,
        png_rcs=rain_frozen.png_rcs,
        png_gde=png_gde,
        pbg_scw=riming.pbg_scw,
        pbg_rfz=freezing.pbg_rfz,
        pbg_gcw=frozen.pbg_gcw,
        pbg_rci=ice_coll.pbg_rci,
        pbg_rcs=rain_frozen.pbg_rcs,
        pbg_rcg=rain_frozen.pbg_rcg,
        pbg_sml=zero,
        pbg_gml=melting.pbg_gml,
        is_aerosol_aware=is_aerosol_aware,
        wif_input_opt=wif_input_opt,
        dusty_ice=dusty_ice,
        ifdry=ifdry,
    )
    state = thompson_post_tendency_refresh(pre, tendencies, dt, is_aerosol_aware=is_aerosol_aware, is_hail_aware=is_hail_aware, iiwarm=iiwarm)
    cloud = thompson_cloud_water_condensation(
        pre,
        state,
        tendencies,
        dt,
        w1d=w1d,
        tnc_wev=tables.tnc_wev,
        tnccn_act=tables.tnccn_act,
        is_aerosol_aware=is_aerosol_aware,
        ifdry=ifdry,
    )
    rain = thompson_rain_evaporation(pre, state, cloud, tendencies, dt, prr_gml=melting.prr_gml, ifdry=ifdry)
    # Keep the input materialization previously enforced by the level scans.
    rain_setup = thompson_rain_sedimentation_setup(*optimization_barrier((state, rain, dt, dzq)))
    rain_sed = thompson_rain_sedimentation(rain, rain_setup, dt, dzq)
    cloud_setup = thompson_cloud_sedimentation_setup(cloud, rain_setup, dzq, w1d)
    cloud_sed = thompson_cloud_sedimentation(cloud, rain, cloud_setup, dt, dzq)
    ice_setup = thompson_ice_sedimentation_setup(*optimization_barrier((state, rain_setup, dt, dzq)))
    ice_sed = thompson_ice_sedimentation(state, tendencies, rain, ice_setup, dt, dzq)
    snow_args, snow_kwargs = optimization_barrier((
        (state, rain, rain_setup, dt, dzq),
        dict(prr_sml=melting.prr_sml, vts_boost=riming.vts_boost),
    ))
    snow_setup = thompson_snow_sedimentation_setup(*snow_args, **snow_kwargs, iiwarm=iiwarm)
    snow_sed = thompson_snow_sedimentation(state, tendencies, rain, snow_setup, dt, dzq)
    graupel_setup = thompson_graupel_sedimentation_setup(*optimization_barrier((state, rain, rain_setup, dt, dzq)), is_hail_aware=is_hail_aware)
    graupel_sed = thompson_graupel_sedimentation(state, tendencies, rain, graupel_setup, dt, dzq)
    sedimented = ThompsonTendencyResult(
        nwfaten=cloud.nwfaten,
        nifaten=tendencies.nifaten,
        nbcaten=tendencies.nbcaten,
        qvten=rain.qvten,
        qcten=cloud_sed.qcten,
        ncten=cloud_sed.ncten,
        qiten=ice_sed.qiten,
        niten=ice_sed.niten,
        qrten=rain_sed.qrten,
        nrten=rain_sed.nrten,
        qsten=snow_sed.qsten,
        qgten=graupel_sed.qgten,
        ngten=graupel_sed.ngten,
        qbten=graupel_sed.qbten,
        tten=rain.tten,
        mvd_r=tendencies.mvd_r,
        mvd_g=tendencies.mvd_g,
    )
    phase = thompson_instant_phase_change(pre, rain, cloud_sed.qcten, cloud_sed.ncten, ice_sed.qiten, ice_sed.niten, rain.tten, dt, iiwarm=iiwarm, ifdry=ifdry)
    final = thompson_final_passback(
        pre,
        sedimented,
        rain.rho,
        dt,
        phase=phase,
        is_aerosol_aware=is_aerosol_aware,
        is_hail_aware=is_hail_aware,
        aer_init_opt=aer_init_opt,
        wif_input_opt=wif_input_opt,
    )
    final_tendencies = ThompsonTendencyResult(
        nwfaten=sedimented.nwfaten,
        nifaten=sedimented.nifaten,
        nbcaten=sedimented.nbcaten,
        qvten=sedimented.qvten,
        qcten=phase.qcten,
        ncten=phase.ncten,
        qiten=phase.qiten,
        niten=phase.niten,
        qrten=sedimented.qrten,
        nrten=sedimented.nrten,
        qsten=sedimented.qsten,
        qgten=sedimented.qgten,
        ngten=sedimented.ngten,
        qbten=sedimented.qbten,
        tten=phase.tten,
        mvd_r=sedimented.mvd_r,
        mvd_g=sedimented.mvd_g,
    )
    active_result = ThompsonColumnDriverResult(
        final=final,
        tendencies=final_tendencies,
        pptrain=rain_sed.pptrain,
        pptsnow=snow_sed.pptsnow,
        pptice=ice_sed.pptice,
        pptgraul=graupel_sed.pptgraul,
    )
    no_micro_result = _thompson_no_micro_return(
        pre,
        qv1d=qv1d,
        nwfa1d=nwfa1d,
        nifa1d=nifa1d,
        nbca1d=nbca1d,
        t1d=t1d,
    )
    return tree_util.tree_map(
        lambda active, early: jnp.where(pre.no_micro, early, active),
        active_result,
        no_micro_result,
    )


def calc_effect_rad(
    t1d: Any,
    p1d: Any,
    qv1d: Any,
    qc1d: Any,
    nc1d: Any,
    qi1d: Any,
    ni1d: Any,
    qs1d: Any,
    *,
    is_aerosol_aware: bool = False,
) -> CalcEffectRadResult:
    """Compute Thompson cloud/ice/snow effective radii for one column."""

    t = jnp.asarray(t1d, dtype=jnp.float32)
    p = jnp.asarray(p1d, dtype=jnp.float32)
    qv = jnp.asarray(qv1d, dtype=jnp.float32)
    qc = jnp.asarray(qc1d, dtype=jnp.float32)
    nc_input = jnp.asarray(nc1d, dtype=jnp.float32)
    qi = jnp.asarray(qi1d, dtype=jnp.float32)
    ni_input = jnp.asarray(ni1d, dtype=jnp.float32)
    qs = jnp.asarray(qs1d, dtype=jnp.float32)

    rho = jnp.float32(0.622) * p / (R * t * (qv + jnp.float32(0.622)))
    rc = jnp.maximum(R1, qc * rho)
    nc = jnp.maximum(jnp.float32(2.0), jnp.minimum(nc_input * rho, NT_C_MAX))
    if not is_aerosol_aware:
        nc = jnp.ones_like(nc) * NT_C
    ri = jnp.maximum(R1, qi * rho)
    ni = jnp.maximum(R2, ni_input * rho)
    rs = jnp.maximum(R1, qs * rho)

    inu_c = jnp.where(
        nc < jnp.float32(100.0),
        jnp.asarray(15, dtype=jnp.int32),
        jnp.where(
            nc > jnp.float32(1.0e10),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.minimum(jnp.asarray(15, dtype=jnp.int32), _fortran_nint(jnp.float32(1000.0e6) / nc) + jnp.asarray(2, dtype=jnp.int32)),
        ),
    )
    lamc = (nc * AM_R * G_RATIO[inu_c - 1] / rc) ** OBMR
    re_qc = jnp.maximum(
        jnp.float32(2.51e-6),
        jnp.minimum(jnp.float32(0.5) * (jnp.float32(3.0) + inu_c.astype(jnp.float32)) / lamc, jnp.float32(50.0e-6)),
    )
    re_qc = jnp.where((rc > R1) & (nc > R2), re_qc, RE_QC_BG)

    lami = (AM_I * CIG2 * OIG1 * ni / ri) ** OBMI
    re_qi = jnp.maximum(
        jnp.float32(2.51e-6),
        jnp.minimum(jnp.float32(0.5) * (jnp.float32(3.0) + MU_I) / lami, jnp.float32(125.0e-6)),
    )
    re_qi = jnp.where((ri > R1) & (ni > R2), re_qi, RE_QI_BG)

    tc0 = jnp.minimum(jnp.float32(-0.1), t - jnp.float32(273.15))
    smob = rs * OAMS
    smo2 = smob
    a_coeff, b_coeff = _snow_moment_coeff(tc0, CSE1)
    smoc = a_coeff * smo2**b_coeff
    re_qs = jnp.maximum(jnp.float32(5.01e-6), jnp.minimum(jnp.float32(0.5) * (smoc / smob), jnp.float32(999.0e-6)))
    re_qs = jnp.where(rs > R1, re_qs, RE_QS_BG)
    return CalcEffectRadResult(re_qc1d=re_qc, re_qi1d=re_qi, re_qs1d=re_qs)


def _fortran_nint_np(x: Any) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return np.where(values >= np.float32(0.0), np.floor(values + np.float32(0.5)), np.ceil(values - np.float32(0.5))).astype(np.int32)


def _snow_moment_coeff_np(tc0: Any, moment: Any) -> tuple[np.float32, np.float32]:
    tc = np.float32(tc0)
    mm = np.float32(moment)
    sa = np.asarray(SA, dtype=np.float32)
    sb = np.asarray(SB, dtype=np.float32)
    tc2 = np.float32(tc * tc)
    tc3 = np.float32(tc2 * tc)
    mm2 = np.float32(mm * mm)
    mm3 = np.float32(mm2 * mm)
    loga = np.float32(
        sa[0]
        + sa[1] * tc
        + sa[2] * mm
        + sa[3] * tc * mm
        + sa[4] * tc2
        + sa[5] * mm2
        + sa[6] * tc2 * mm
        + sa[7] * tc * mm2
        + sa[8] * tc3
        + sa[9] * mm3
    )
    b_coef = np.float32(
        sb[0]
        + sb[1] * tc
        + sb[2] * mm
        + sb[3] * tc * mm
        + sb[4] * tc2
        + sb[5] * mm2
        + sb[6] * tc2 * mm
        + sb[7] * tc * mm2
        + sb[8] * tc3
        + sb[9] * mm3
    )
    return np.float32(np.float32(10.0) ** loga), b_coef


def calc_refl10cm(
    qv1d: Any,
    qc1d: Any,
    qr1d: Any,
    nr1d: Any,
    qs1d: Any,
    qg1d: Any,
    ng1d: Any,
    qb1d: Any,
    t1d: Any,
    p1d: Any,
    dbz: Any | None = None,
    *,
    ke_diag: int | None = None,
    is_hail_aware: bool = False,
    radar_state: RadarState | None = None,
) -> np.ndarray:
    """Compute Thompson 10 cm radar reflectivity for one vertical column.

    The input arrays are the Python equivalent of WRF ``kts:kte`` slices.  When
    ``ke_diag`` is provided, it is the WRF-style one-based diagnostic top index
    within the supplied slice.
    """

    qv_input = np.asarray(qv1d, dtype=np.float32)
    nlev = int(qv_input.shape[0])
    if nlev == 0:
        return np.asarray([], dtype=np.float32)
    qc_input = np.asarray(qc1d, dtype=np.float32)
    qr_input = np.asarray(qr1d, dtype=np.float32)
    nr_input = np.asarray(nr1d, dtype=np.float32)
    qs_input = np.asarray(qs1d, dtype=np.float32)
    qg_input = np.asarray(qg1d, dtype=np.float32)
    ng_input = np.asarray(ng1d, dtype=np.float32)
    qb_input = np.asarray(qb1d, dtype=np.float32)
    temp = np.asarray(t1d, dtype=np.float32).copy()
    pres = np.asarray(p1d, dtype=np.float32).copy()
    out = np.full(nlev, np.float32(-35.0), dtype=np.float32) if dbz is None else np.asarray(dbz, dtype=np.float32).copy()
    out[:] = np.float32(-35.0)
    top_count = nlev if ke_diag is None else max(1, min(int(ke_diag), nlev))
    radar = radar_init() if radar_state is None else radar_state

    qv = np.maximum(np.float32(1.0e-10), qv_input).astype(np.float32)
    rho = (np.float32(0.622) * pres / (np.float32(R) * temp * (qv + np.float32(0.622)))).astype(np.float32)
    rc = np.maximum(np.float32(R1), qc_input * rho).astype(np.float32)
    del rc  # WRF computes rc here, but reflectivity does not use it later.

    rr = np.full(nlev, np.float32(R1), dtype=np.float32)
    nr = np.full(nlev, np.float32(R1), dtype=np.float32)
    rs = np.full(nlev, np.float32(R1), dtype=np.float32)
    rg = np.full(nlev, np.float32(R1), dtype=np.float32)
    ng = np.full(nlev, np.float32(R2), dtype=np.float32)
    rb = np.zeros(nlev, dtype=np.float32)
    idx_bg = np.full(nlev, IDX_BG1, dtype=np.int32)
    ilamr = np.zeros(nlev, dtype=np.float64)
    ilamg = np.zeros(nlev, dtype=np.float64)
    n0_r = np.zeros(nlev, dtype=np.float64)
    n0_g = np.zeros(nlev, dtype=np.float64)
    smob = np.zeros(nlev, dtype=np.float32)
    smo2 = np.zeros(nlev, dtype=np.float32)
    smoc = np.zeros(nlev, dtype=np.float32)
    smoz = np.zeros(nlev, dtype=np.float32)
    l_qr = np.zeros(nlev, dtype=bool)
    l_qs = np.zeros(nlev, dtype=bool)
    l_qg = np.zeros(nlev, dtype=bool)

    for k in range(nlev):
        if qr_input[k] > np.float32(R1):
            rr[k] = np.float32(qr_input[k] * rho[k])
            nr[k] = np.maximum(np.float32(R2), np.float32(nr_input[k] * rho[k]))
            lamr = np.float64((np.float32(AM_R) * np.float32(radar.xcrg[2]) * np.float32(radar.xorg2) * nr[k] / rr[k]) ** np.float32(OBMR))
            ilamr[k] = np.float64(1.0) / lamr
            n0_r[k] = np.float64(np.float32(nr[k] * np.float32(radar.xorg2) * np.float32(lamr ** np.float64(radar.xcre[1]))))
            l_qr[k] = True
        if qs_input[k] > np.float32(R2):
            rs[k] = np.float32(qs_input[k] * rho[k])
            l_qs[k] = True
        if qg_input[k] > np.float32(R2):
            rg[k] = np.float32(qg_input[k] * rho[k])
            ng[k] = np.maximum(np.float32(R2), np.float32(ng_input[k] * rho[k]))
            rb[k] = np.maximum(np.float32(qg_input[k] / RHO_G[-1]), qb_input[k])
            rb[k] = np.minimum(np.float32(qg_input[k] / RHO_G[0]), rb[k])
            idx_bg[k] = int(np.clip(_fortran_nint_np(qg_input[k] / rb[k] * np.float32(0.01))[()], 1, RHO_G.size) - 1)
            if not is_hail_aware:
                idx_bg[k] = IDX_BG1
            l_qg[k] = True

    if (top_count > 1 and np.any(l_qs)) or (top_count == 1 and l_qs[0]):
        for k in range(top_count):
            if not l_qs[k]:
                continue
            tc0 = np.minimum(np.float32(-0.1), np.float32(temp[k] - np.float32(273.15)))
            smob[k] = np.float32(rs[k] * np.float32(OAMS))
            if np.float32(BM_S) > np.float32(2.0 - 1.0e-3) and np.float32(BM_S) < np.float32(2.0 + 1.0e-3):
                smo2[k] = smob[k]
            else:
                a_coeff, b_coeff = _snow_moment_coeff_np(tc0, np.float32(BM_S))
                smo2[k] = np.float32((smob[k] / a_coeff) ** (np.float32(1.0) / b_coeff))
            a_coeff, b_coeff = _snow_moment_coeff_np(tc0, np.float32(CSE1))
            smoc[k] = np.float32(a_coeff * smo2[k] ** b_coeff)
            a_coeff, b_coeff = _snow_moment_coeff_np(tc0, np.float32(CSE3))
            smoz[k] = np.float32(a_coeff * smo2[k] ** b_coeff)

    if np.any(l_qg):
        for k in range(nlev - 1, -1, -1):
            if not l_qg[k]:
                continue
            bg = idx_bg[k]
            lamg = np.float64((AM_G[bg] * np.float32(radar.xcgg[2]) * np.float32(radar.xogg2) * ng[k] / rg[k]) ** np.float32(OBMG))
            ilamg[k] = np.float64(1.0) / lamg
            n0_g[k] = np.float64(np.float32(ng[k] * np.float32(radar.xogg2) * np.float32(lamg ** np.float64(radar.xcge[1]))))

    melti = False
    k_0 = 1
    for idx in range(nlev - 2, -1, -1):
        if (temp[idx] > np.float32(273.15)) and l_qr[idx] and (l_qs[idx + 1] or l_qg[idx + 1]):
            k_0 = max(idx + 2, k_0)
            melti = True
            break
    k_0loop = min(k_0, top_count + 1)

    ze_rain = np.full(nlev, np.float32(1.0e-22), dtype=np.float32)
    ze_snow = np.full(nlev, np.float32(1.0e-22), dtype=np.float32)
    ze_graupel = np.full(nlev, np.float32(1.0e-22), dtype=np.float32)
    for k in range(top_count):
        if l_qr[k]:
            ze_rain[k] = np.float32(n0_r[k] * np.float64(radar.xcrg[3]) * ilamr[k] ** np.float64(radar.xcre[3]))
        if l_qs[k]:
            ze_snow[k] = np.float32(
                np.float32(0.176 / 0.93)
                * np.float32(6.0 / 3.1415926536)
                * np.float32(6.0 / 3.1415926536)
                * np.float32(AM_S / 900.0)
                * np.float32(AM_S / 900.0)
                * smoz[k]
            )
        if l_qg[k]:
            bg = idx_bg[k]
            ze_graupel[k] = np.float32(
                np.float32(0.176 / 0.93)
                * np.float32(6.0 / 3.1415926536)
                * np.float32(6.0 / 3.1415926536)
                * np.float32(AM_G[bg] / 900.0)
                * np.float32(AM_G[bg] / 900.0)
                * np.float32(n0_g[k] * np.float64(radar.xcgg[3]) * ilamg[k] ** np.float64(radar.xcge[3]))
            )

    if melti and k_0 >= 2:
        for fortran_k in range(k_0loop - 1, 0, -1):
            k = fortran_k - 1
            k0_idx = k_0 - 1
            if l_qs[k] and l_qs[k0_idx]:
                fmelt_s = np.maximum(np.float64(0.05), np.minimum(np.float64(1.0) - np.float64(rs[k] / rs[k0_idx]), np.float64(0.99)))
                eta = np.float64(0.0)
                om3 = np.float32(1.0) / smoc[k]
                m0 = np.float32(smob[k] * om3)
                mrat = np.float32(smob[k] * m0 * m0 * m0)
                slam1 = np.float32(m0 * np.float32(LAM0))
                slam2 = np.float32(m0 * np.float32(LAM1))
                for n in range(radar.xx_ds.shape[0]):
                    x = np.float64(np.float32(AM_S) * np.float32(radar.xx_ds[n] ** np.float64(BM_S)))
                    cback = rayleigh_soak_wetgraupel(
                        x,
                        np.float64(OCMS),
                        np.float64(OBMS),
                        fmelt_s,
                        MELT_OUTSIDE_S,
                        radar.m_w_0,
                        radar.m_i_0,
                        radar.lamda_radar,
                        radar.mixingrulestring_s,
                        radar.matrixstring_s,
                        radar.inclusionstring_s,
                        radar.hoststring_s,
                        radar.hostmatrixstring_s,
                        radar.hostinclusionstring_s,
                        state=radar,
                    )
                    f_d = np.float64(
                        np.float32(
                            mrat
                            * (
                                np.float32(KAP0) * np.float32(math.exp(-float(slam1) * radar.xx_ds[n]))
                                + np.float32(KAP1)
                                * np.float32((np.float32(m0) * np.float32(radar.xx_ds[n])) ** np.float32(MU_S))
                                * np.float32(math.exp(-float(slam2) * radar.xx_ds[n]))
                            )
                        )
                    )
                    eta += f_d * cback * radar.simpson[n] * radar.xdt_s[n]
                ze_snow[k] = np.float32(radar.lamda4 / (radar.pi5 * radar.k_w) * eta)

    for k in range(top_count - 1, -1, -1):
        out[k] = np.float32(np.float32(10.0) * np.float32(math.log10(float((ze_rain[k] + ze_snow[k] + ze_graupel[k]) * np.float32(1.0e18)))))
    return out


def gammln(xx: Any) -> Any:
    """Natural log gamma approximation from WRF Thompson ``GAMMLN``."""

    x = jnp.asarray(xx, dtype=jnp.float32)
    y0 = x
    tmp0 = x + jnp.float32(5.5)
    tmp = (x + jnp.float32(0.5)) * jnp.log(tmp0) - tmp0
    ser0 = jnp.ones_like(x) * jnp.float32(1.000000000190015)

    def body(carry: tuple[Any, Any], cof: Any) -> tuple[tuple[Any, Any], None]:
        y, ser = carry
        y = y + jnp.float32(1.0)
        ser = ser + cof / y
        return (y, ser), None

    (_, ser), _ = lax.scan(body, (y0, ser0), _GAMMLN_COF)
    return tmp + jnp.log(_GAMMLN_STP * ser / x)


def _gser(a: Any, x: Any) -> tuple[Any, Any]:
    a_arr = jnp.asarray(a, dtype=jnp.float32)
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    x_safe = jnp.maximum(x_arr, jnp.float32(1.0e-30))
    gln = gammln(a_arr)
    ap0 = jnp.ones_like(x_arr) * a_arr
    sum0 = jnp.ones_like(x_arr) * (jnp.float32(1.0) / a_arr)
    del0 = sum0
    done0 = x_arr <= jnp.float32(0.0)

    def body(carry: tuple[Any, Any, Any, Any], _: Any) -> tuple[tuple[Any, Any, Any, Any], None]:
        ap, delta, total, done = carry
        ap_next = ap + jnp.float32(1.0)
        delta_next = delta * x_safe / ap_next
        total_next = total + delta_next
        converged = jnp.abs(delta_next) < jnp.abs(total_next) * _GAMMA_EPS
        update = ~done
        ap = jnp.where(update, ap_next, ap)
        delta = jnp.where(update, delta_next, delta)
        total = jnp.where(update, total_next, total)
        done = done | converged
        return (ap, delta, total, done), None

    (_, _, total, _), _ = lax.scan(body, (ap0, del0, sum0, done0), jnp.arange(_GAMMA_ITMAX))
    gams = total * jnp.exp(-x_safe + a_arr * jnp.log(x_safe) - gln)
    return jnp.where(x_arr <= jnp.float32(0.0), jnp.float32(0.0), gams), gln


def _gcf(a: Any, x: Any) -> tuple[Any, Any]:
    a_arr = jnp.asarray(a, dtype=jnp.float32)
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    x_safe = jnp.maximum(x_arr, jnp.float32(1.0e-30))
    gln = gammln(a_arr)
    b0 = x_safe + jnp.float32(1.0) - a_arr
    c0 = jnp.ones_like(x_arr) / _GAMMA_FPMIN
    d0 = jnp.float32(1.0) / b0
    h0 = d0
    done0 = jnp.zeros_like(x_arr, dtype=bool)

    def body(carry: tuple[Any, Any, Any, Any, Any], i: Any) -> tuple[tuple[Any, Any, Any, Any, Any], None]:
        b, c, d, h, done = carry
        ii = i.astype(jnp.float32)
        an = -ii * (ii - a_arr)
        b_next = b + jnp.float32(2.0)
        d_next = an * d + b_next
        d_next = jnp.where(jnp.abs(d_next) < _GAMMA_FPMIN, _GAMMA_FPMIN, d_next)
        c_next = b_next + an / c
        c_next = jnp.where(jnp.abs(c_next) < _GAMMA_FPMIN, _GAMMA_FPMIN, c_next)
        d_next = jnp.float32(1.0) / d_next
        del_next = d_next * c_next
        h_next = h * del_next
        converged = jnp.abs(del_next - jnp.float32(1.0)) < _GAMMA_EPS
        update = ~done
        b = jnp.where(update, b_next, b)
        c = jnp.where(update, c_next, c)
        d = jnp.where(update, d_next, d)
        h = jnp.where(update, h_next, h)
        done = done | converged
        return (b, c, d, h, done), None

    (_, _, _, h, _), _ = lax.scan(body, (b0, c0, d0, h0, done0), jnp.arange(1, _GAMMA_ITMAX + 1))
    gammcf = jnp.exp(-x_safe + a_arr * jnp.log(x_safe) - gln) * h
    return gammcf, gln


def gammp(a: Any, x: Any) -> Any:
    """Incomplete gamma function ``P(a,x)`` from WRF Thompson ``GAMMP``."""

    a_arr = jnp.asarray(a, dtype=jnp.float32)
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    gams, _ = _gser(a_arr, x_arr)
    gammcf, _ = _gcf(a_arr, x_arr)
    value = jnp.where(x_arr < a_arr + jnp.float32(1.0), gams, jnp.float32(1.0) - gammcf)
    return jnp.where((x_arr < jnp.float32(0.0)) | (a_arr <= jnp.float32(0.0)), jnp.float32(0.0), value)


def wgamma(y: Any) -> Any:
    """Gamma function from WRF Thompson ``WGAMMA``."""

    return jnp.exp(gammln(y))


def ice_demott(tempc: Any, qv: Any, qvs: Any, qvsi: Any, rho: Any, nifa: Any) -> Any:
    """Ice nucleation concentration from WRF Thompson ``iceDeMott``."""

    del qv, qvs, qvsi
    tempc_arr = jnp.asarray(tempc, dtype=jnp.float32)
    rho_arr = jnp.asarray(rho, dtype=jnp.float32)
    nifa_arr = jnp.asarray(nifa, dtype=jnp.float32)
    rho_not0 = jnp.float32(101325.0) / (jnp.float32(287.05) * jnp.float32(273.15))
    nifa_cc = jnp.maximum(jnp.float32(0.5), nifa_arr * rho_not0 * jnp.float32(1.0e-6) / rho_arr)
    xni = (jnp.float32(5.94e-5) * (-tempc_arr) ** jnp.float32(3.33)) * (
        nifa_cc ** ((jnp.float32(-0.0264) * tempc_arr) + jnp.float32(0.0033))
    )
    xni = xni * rho_arr / rho_not0 * jnp.float32(1000.0)
    return jnp.maximum(jnp.float32(0.0), xni)


def ice_koop(temp: Any, qv: Any, qvs: Any, naero: Any, dt: Any) -> Any:
    """Homogeneous freezing concentration from WRF Thompson ``iceKoop``."""

    temp_arr = jnp.asarray(temp, dtype=jnp.float32)
    qv_arr = jnp.asarray(qv, dtype=jnp.float32)
    qvs_arr = jnp.asarray(qvs, dtype=jnp.float32)
    naero_arr = jnp.asarray(naero, dtype=jnp.float32)
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    satw = qv_arr / qvs_arr
    mu_diff = (
        jnp.float32(210368.0)
        + jnp.float32(131.438) * temp_arr
        - jnp.float32(3.32373e6) / temp_arr
        - jnp.float32(41729.1) * jnp.log(temp_arr)
    )
    a_w_i = jnp.exp(mu_diff / (_R_UNI * temp_arr))
    delta_aw = satw - a_w_i
    log_j_rate = (
        jnp.float32(-906.7)
        + jnp.float32(8502.0) * delta_aw
        - jnp.float32(26924.0) * delta_aw * delta_aw
        + jnp.float32(29180.0) * delta_aw * delta_aw * delta_aw
    )
    log_j_rate = jnp.minimum(jnp.float32(20.0), log_j_rate)
    j_rate = jnp.float32(10.0) ** log_j_rate
    prob_h = jnp.minimum(jnp.float32(1.0) - jnp.exp(-j_rate * _AR_VOLUME * dt_arr), jnp.float32(1.0))
    xni = jnp.where(prob_h > jnp.float32(0.0), jnp.minimum(prob_h * naero_arr, jnp.float32(1000.0e3)), jnp.float32(0.0))
    return jnp.maximum(jnp.float32(0.0), xni)


def delta_p(yy: Any, y1: Any, y2: Any, aa: Any, bb: Any) -> Any:
    """Phillips ice-nucleation helper from WRF Thompson ``delta_p``."""

    yy_arr = jnp.asarray(yy, dtype=jnp.float32)
    y1_arr = jnp.asarray(y1, dtype=jnp.float32)
    y2_arr = jnp.asarray(y2, dtype=jnp.float32)
    aa_arr = jnp.asarray(aa, dtype=jnp.float32)
    bb_arr = jnp.asarray(bb, dtype=jnp.float32)
    dy = lax.optimization_barrier(y2_arr - y1_arr)
    den = lax.optimization_barrier(lax.optimization_barrier(dy * dy) * dy)
    a_coef = lax.optimization_barrier(lax.optimization_barrier(jnp.float32(6.0) * (aa_arr - bb_arr)) / den)
    y1_sq = lax.optimization_barrier(y1_arr * y1_arr)
    y1_cu = lax.optimization_barrier(y1_sq * y1_arr)
    b_left = lax.optimization_barrier(lax.optimization_barrier(a_coef * y1_cu) / jnp.float32(6.0))
    b_right = lax.optimization_barrier(lax.optimization_barrier(lax.optimization_barrier(a_coef * y1_sq) * y2_arr) * jnp.float32(0.5))
    b_coef = lax.optimization_barrier(lax.optimization_barrier(aa_arr + b_left) - b_right)
    a0 = b_coef
    a1 = lax.optimization_barrier(lax.optimization_barrier(a_coef * y1_arr) * y2_arr)
    a2 = lax.optimization_barrier(lax.optimization_barrier(-a_coef * lax.optimization_barrier(y1_arr + y2_arr)) * jnp.float32(0.5))
    a3 = lax.optimization_barrier(a_coef / jnp.float32(3.0))
    term1 = lax.optimization_barrier(a1 * yy_arr)
    term2 = lax.optimization_barrier(lax.optimization_barrier(a2 * yy_arr) * yy_arr)
    term3 = lax.optimization_barrier(lax.optimization_barrier(lax.optimization_barrier(a3 * yy_arr) * yy_arr) * yy_arr)
    dab_poly = lax.optimization_barrier(lax.optimization_barrier(lax.optimization_barrier(a0 + term1) + term2) + term3)
    dab = jnp.where(yy_arr <= y1_arr, aa_arr, jnp.where(yy_arr >= y2_arr, bb_arr, dab_poly))
    dab = jnp.where(dab < aa_arr, aa_arr, dab)
    dab = jnp.where(dab > bb_arr, bb_arr, dab)
    return dab


def eff_aero(
    d: Any,
    da: Any,
    visc: Any,
    rhoa: Any,
    temp: Any,
    species: str,
    *,
    av_g_idx_bg1: Any = AV_G_IDX_BG1_DEFAULT,
    bv_g_idx_bg1: Any = BV_G_IDX_BG1_DEFAULT,
) -> Any:
    """Collision efficiency of aerosols with rain/snow/graupel collectors."""

    d_arr = jnp.asarray(d, dtype=jnp.float32)
    da_arr = jnp.asarray(da, dtype=jnp.float32)
    visc_arr = jnp.asarray(visc, dtype=jnp.float32)
    rhoa_arr = jnp.asarray(rhoa, dtype=jnp.float32)
    temp_arr = jnp.asarray(temp, dtype=jnp.float32)
    species_code = species.lower()

    vt = jnp.float32(1.0)
    if species_code == "r":
        vt = (
            jnp.float32(-0.1021)
            + jnp.float32(4.932e3) * d_arr
            - jnp.float32(0.9551e6) * d_arr * d_arr
            + jnp.float32(0.07934e9) * d_arr * d_arr * d_arr
            - jnp.float32(0.002362e12) * d_arr * d_arr * d_arr * d_arr
        )
    elif species_code == "s":
        vt = AV_S * d_arr**BV_S
    elif species_code == "g":
        vt = jnp.asarray(av_g_idx_bg1, dtype=jnp.float32) * d_arr**jnp.asarray(bv_g_idx_bg1, dtype=jnp.float32)

    cc = jnp.float32(1.0) + jnp.float32(2.0) * jnp.float32(0.0256e-6) / da_arr * (
        jnp.float32(1.257) + jnp.float32(0.4) * jnp.exp(jnp.float32(-0.55) * da_arr / jnp.float32(0.0256e-6))
    )
    diff = jnp.float32(1.3806503e-23) * temp_arr * cc / (jnp.float32(3.0) * _PI * visc_arr * da_arr)
    re = jnp.float32(0.5) * rhoa_arr * d_arr * vt / visc_arr
    sc = visc_arr / (rhoa_arr * diff)
    st = da_arr * da_arr * vt * jnp.float32(1000.0) / (jnp.float32(9.0) * visc_arr * d_arr)
    aval = jnp.float32(1.0) + jnp.log(jnp.float32(1.0) + re)
    st2 = (jnp.float32(1.2) + jnp.float32(1.0) / jnp.float32(12.0) * aval) / (jnp.float32(1.0) + aval)
    eff = (
        jnp.float32(4.0) / (re * sc)
        * (
            jnp.float32(1.0)
            + jnp.float32(0.4) * sqrt_for_wrf(re) * sc ** jnp.float32(0.3333)
            + jnp.float32(0.16) * sqrt_for_wrf(re) * sqrt_for_wrf(sc)
        )
        + jnp.float32(4.0) * da_arr / d_arr * (jnp.float32(0.02) + da_arr / d_arr * (jnp.float32(1.0) + jnp.float32(2.0) * sqrt_for_wrf(re)))
    )
    eff = jnp.where(st > st2, eff + ((st - st2) / (st - st2 + jnp.float32(0.666667))) ** jnp.float32(1.5), eff)
    return jnp.maximum(jnp.float32(1.0e-5), jnp.minimum(eff, jnp.float32(1.0)))


def rslf(p: Any, t: Any) -> Any:
    """Liquid saturation vapor mixing ratio from WRF Thompson ``RSLF``."""

    return _saturation_mixing_ratio(p, t, ice=False)


def _store_wrf_real(value: Any, rounding_token: Any) -> Any:
    return real_assignment(value, rounding_token)


def _saturation_mixing_ratio(
    p: Any,
    t: Any,
    *,
    ice: bool,
    rounding_token: Any | None = None,
) -> Any:
    """Evaluate Thompson RSLF/RSIF with optional WRF REAL operation stores."""

    p_arr = jnp.asarray(p, dtype=jnp.float32)
    t_arr = jnp.asarray(t, dtype=jnp.float32)
    coefficients = (
        (_RSIF_C0, _RSIF_C1, _RSIF_C2, _RSIF_C3, _RSIF_C4, _RSIF_C5, _RSIF_C6, _RSIF_C7, _RSIF_C8)
        if ice
        else (_RSLF_C0, _RSLF_C1, _RSLF_C2, _RSLF_C3, _RSLF_C4, _RSLF_C5, _RSLF_C6, _RSLF_C7, _RSLF_C8)
    )
    if rounding_token is not None:
        from wrf_jax.core.elementwise import map_elements

        return map_elements(
            _rounded_saturation_body(ice),
            (p_arr, t_arr),
            (jnp.stack(coefficients), rounding_token),
        )
    else:
        x = jnp.maximum(jnp.float32(-80.0), t_arr - jnp.float32(273.16))
        vapor_pressure = coefficients[0] + x * (
            coefficients[1]
            + x
            * (
                coefficients[2]
                + x
                * (
                    coefficients[3]
                    + x
                    * (
                        coefficients[4]
                        + x * (coefficients[5] + x * (coefficients[6] + x * (coefficients[7] + x * coefficients[8])))
                    )
                )
            )
        )
        vapor_pressure = jnp.minimum(vapor_pressure, p_arr * jnp.float32(0.15))
        denominator = p_arr - vapor_pressure
    if ice:
        denominator = jnp.maximum(jnp.float32(1.0e-4), denominator)
    numerator = jnp.float32(0.622) * vapor_pressure
    return numerator / denominator


@lru_cache(maxsize=2)
def _rounded_saturation_body(ice):
    return partial(_rounded_saturation_values, ice=ice)


def _rounded_saturation_values(inputs, parameters, *, ice):
    from wrf_jax.core.real_arithmetic import real_divide

    p_arr, t_arr = inputs
    coefficients, rounding_token = parameters
    store = lambda value: _store_wrf_real(value, rounding_token)
    x = store(jnp.maximum(jnp.float32(-80.0), store(t_arr - jnp.float32(273.16))))
    polynomial = coefficients[8]
    for index in range(7, 0, -1):
        polynomial = store(coefficients[index] + store(x * polynomial))
    vapor_pressure = store(coefficients[0] + store(x * polynomial))
    vapor_pressure = store(jnp.minimum(vapor_pressure, store(p_arr * jnp.float32(0.15))))
    denominator = store(p_arr - vapor_pressure)
    if ice:
        denominator = store(jnp.maximum(jnp.float32(1.0e-4), denominator))
    numerator = store(jnp.float32(0.622) * vapor_pressure)
    return store(real_divide(numerator, denominator))


def rslf_rounded(p: Any, t: Any, rounding_token: Any) -> Any:
    """Token-rounded Thompson RSLF for a larger compiled graph."""

    return _saturation_mixing_ratio(p, t, ice=False, rounding_token=rounding_token)


def rsif(p: Any, t: Any) -> Any:
    """Ice saturation vapor mixing ratio from WRF Thompson ``RSIF``."""

    return _saturation_mixing_ratio(p, t, ice=True)


def rsif_rounded(p: Any, t: Any, rounding_token: Any) -> Any:
    """Token-rounded Thompson RSIF for a larger compiled graph."""

    return _saturation_mixing_ratio(p, t, ice=True, rounding_token=rounding_token)


def theta_e(pres_pa: Any, temp_k: Any, w_non: Any, tlcl_k: Any) -> Any:
    """Equivalent potential temperature from Thompson ``theta_e``."""

    pp = jnp.asarray(pres_pa, dtype=jnp.float32)
    tt = jnp.asarray(temp_k, dtype=jnp.float32)
    rr = jnp.asarray(w_non, dtype=jnp.float32) + jnp.float32(1.0e-8)
    tlc = jnp.asarray(tlcl_k, dtype=jnp.float32)
    power = jnp.float32(0.2854) * (jnp.float32(1.0) - jnp.float32(0.28) * rr)
    xx = tt * (jnp.float32(100000.0) / pp) ** power
    p1 = jnp.float32(3.376) / tlc - jnp.float32(0.00254)
    p2 = (rr * jnp.float32(1000.0)) * (jnp.float32(1.0) + jnp.float32(0.81) * rr)
    return xx * jnp.exp(p1 * p2)


def t_lcl(temp_k: Any, tdew_k: Any) -> Any:
    """Lifting condensation level temperature from Thompson ``t_lcl``."""

    tt = jnp.asarray(temp_k, dtype=jnp.float32)
    tttd = jnp.asarray(tdew_k, dtype=jnp.float32)
    denom = (jnp.float32(1.0) / (tttd - jnp.float32(56.0))) + (jnp.log(tt / tttd) / jnp.float32(800.0))
    return (jnp.float32(1.0) / denom) + jnp.float32(56.0)


def t_dew(pres_pa: Any, w_non: Any) -> Any:
    """Dewpoint temperature from Thompson ``t_dew``."""

    pp = jnp.asarray(pres_pa, dtype=jnp.float32)
    rr = jnp.asarray(w_non, dtype=jnp.float32) + jnp.float32(1.0e-8)
    es = pp * rr / (jnp.float32(0.622) + rr)
    esln = jnp.log(es)
    return (jnp.float32(35.86) * esln - jnp.float32(4947.2325)) / (esln - jnp.float32(23.6837))


def theta_wetb(thetae_k: Any) -> Any:
    """Wet-bulb potential temperature fit from Thompson ``theta_wetb``."""

    x = jnp.minimum(jnp.float32(475.0), jnp.asarray(thetae_k, dtype=jnp.float32))
    c = jnp.asarray(
        [-1.00922292e-10, -1.47945344e-8, -1.7303757e-6, -0.00012709, 1.15849867e-6, -3.518296861e-9, 3.5741522e-12],
        dtype=jnp.float32,
    )
    d = jnp.asarray(
        [0.0, -3.5223513e-10, -5.7250807e-8, -5.83975422e-6, 4.72445163e-8, -1.13402845e-10, 8.729580402e-14],
        dtype=jnp.float32,
    )
    answer_c = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * c[6])))))
    answer_d = d[0] + x * (d[1] + x * (d[2] + x * (d[3] + x * (d[4] + x * (d[5] + x * d[6])))))
    return jnp.where(x <= jnp.float32(335.5), answer_c, answer_d) + jnp.float32(273.15)


def _comp_t_fr_the_impl(thelcl_k: Any, pres_pa: Any, *, stop_on_convergence: bool) -> Any:
    """Shared WRF recurrence for forecasting and automatic differentiation."""

    thelcl = jnp.asarray(thelcl_k, dtype=jnp.float32)
    pp = jnp.asarray(pres_pa, dtype=jnp.float32)
    guess0 = (thelcl - jnp.float32(0.5) * jnp.maximum(thelcl - jnp.float32(270.0), jnp.float32(0.0)) ** jnp.float32(1.05)) * (
        pp / jnp.float32(100000.0)
    ) ** jnp.float32(0.2)
    done0 = jnp.zeros_like(guess0, dtype=bool)
    result0 = guess0

    def body(carry: tuple[Any, Any, Any], _: Any) -> tuple[tuple[Any, Any, Any], None]:
        guess, done, result = carry
        w1 = rslf(pp, guess)
        w2 = rslf(pp, guess + jnp.float32(1.0))
        tenu = theta_e(pp, guess, w1, guess)
        tenup = theta_e(pp, guess + jnp.float32(1.0), w2, guess + jnp.float32(1.0))
        cor = (thelcl - tenu) / (tenup - tenu)
        next_guess = guess + cor
        converged = (cor < jnp.float32(0.01)) & (-cor < jnp.float32(0.01))
        newly_done = (~done) & converged
        result = jnp.where(newly_done, next_guess, result)
        guess = jnp.where(done, guess, next_guess)
        done = done | converged
        return (guess, done, result), None

    initial = (guess0, done0, result0)
    if stop_on_convergence:
        def condition(carry):
            iteration, values = carry
            return (iteration < 100) & jnp.any(~values[1])

        def advance(carry):
            iteration, values = carry
            return iteration + 1, body(values, None)[0]

        _, (_, done, result) = lax.while_loop(condition, advance, (jnp.int32(0), initial))
    else:
        # Reverse-mode AD needs a static bound; the recurrence is identical.
        _, done, result = lax.scan(body, initial, jnp.arange(100))[0]
    fallback = theta_wetb(thelcl) * (pp / jnp.float32(100000.0)) ** jnp.float32(0.286)
    return jnp.where(done, result, fallback)


@custom_jvp
def comp_t_fr_the(thelcl_k: Any, pres_pa: Any) -> Any:
    """Temperature on a moist adiabat, stopping at WRF's convergence criterion."""

    return _comp_t_fr_the_impl(thelcl_k, pres_pa, stop_on_convergence=True)


@comp_t_fr_the.defjvp
def _comp_t_fr_the_jvp(primals, tangents):
    value = comp_t_fr_the(*primals)
    _, tangent = jvp(partial(_comp_t_fr_the_impl, stop_on_convergence=False), primals, tangents)
    return value, tangent


compT_fr_The = comp_t_fr_the
