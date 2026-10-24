"""Translations from WRF ``phys/module_mp_radar``.

This module is diagnostic support for Thompson radar reflectivity.  The
Fortran code keeps most values as module globals; the Python translation makes
the initialized radar state explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from jax import tree_util


NRBINS = 50
SLEN = 20
LAMDA_RADAR = np.float64(0.10)
MELT_OUTSIDE_S = np.float64(0.9)
MELT_OUTSIDE_G = np.float64(0.9)
BASIS = np.asarray((1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0), dtype=np.float64)

_PI_DOUBLE = np.float64(3.1415926535897932384626434)
_PI_RADAR = np.float64(3.14159)
_GAMMLN_STP = np.float64(2.5066282746310005)
_GAMMLN_COF = np.asarray(
    (
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class RadarState:
    """Explicit state computed by WRF ``radar_init``."""

    xx_dx: np.ndarray
    xx_ds: np.ndarray
    xdt_s: np.ndarray
    xx_dg: np.ndarray
    xdt_g: np.ndarray
    lamda_radar: np.float64
    k_w: np.float64
    pi5: np.float64
    lamda4: np.float64
    m_w_0: np.complex128
    m_i_0: np.complex128
    simpson: np.ndarray
    basis: np.ndarray
    xcre: np.ndarray
    xcse: np.ndarray
    xcge: np.ndarray
    xcrg: np.ndarray
    xcsg: np.ndarray
    xcgg: np.ndarray
    xorg2: np.float32
    xosg2: np.float32
    xogg2: np.float32
    xam_r: np.float32
    xbm_r: np.float32
    xmu_r: np.float32
    xobmr: np.float32
    xam_s: np.float32
    xbm_s: np.float32
    xmu_s: np.float32
    xoams: np.float32
    xobms: np.float32
    xocms: np.float32
    xam_g: np.float32
    xbm_g: np.float32
    xmu_g: np.float32
    xoamg: np.float32
    xobmg: np.float32
    xocmg: np.float32
    mixingrulestring_s: str = "maxwellgarnett"
    matrixstring_s: str = "water"
    inclusionstring_s: str = "spheroidal"
    hoststring_s: str = "air"
    hostmatrixstring_s: str = "icewater"
    hostinclusionstring_s: str = "spheroidal"
    mixingrulestring_g: str = "maxwellgarnett"
    matrixstring_g: str = "water"
    inclusionstring_g: str = "spheroidal"
    hoststring_g: str = "air"
    hostmatrixstring_g: str = "icewater"
    hostinclusionstring_g: str = "spheroidal"


_RADAR_ARRAY_FIELDS = (
    "xx_dx", "xx_ds", "xdt_s", "xx_dg", "xdt_g", "simpson", "basis",
    "xcre", "xcse", "xcge", "xcrg", "xcsg", "xcgg",
)
tree_util.register_dataclass(
    RadarState,
    data_fields=_RADAR_ARRAY_FIELDS,
    meta_fields=tuple(name for name in RadarState.__dataclass_fields__ if name not in _RADAR_ARRAY_FIELDS),
)


@dataclass(frozen=True)
class ComplexMixResult:
    """Complex refractive index result plus WRF-style error count."""

    value: np.complex128
    error: int


def _fortran_string(value: str) -> str:
    return value.split("\0", 1)[0].strip()


def gammln(xx: Any) -> np.float32:
    """Numerical Recipes ``GAMMLN`` used by WRF radar support."""

    x = np.float64(np.float32(xx))
    y = x
    tmp = x + np.float64(5.5)
    tmp = (x + np.float64(0.5)) * math.log(tmp) - tmp
    ser = np.float64(1.000000000190015)
    for cof in _GAMMLN_COF:
        y += np.float64(1.0)
        ser += cof / y
    return np.float32(tmp + math.log(_GAMMLN_STP * ser / x))


def wgamma(y: Any) -> np.float32:
    """Gamma function ``WGAMMA`` used by WRF radar support."""

    return np.float32(math.exp(float(gammln(y))))


def m_complex_water_ray(lambda_: Any, t: Any) -> np.complex128:
    """Complex refractive index of water from WRF ``m_complex_water_ray``."""

    lam = np.float64(lambda_)
    temp = np.float64(t)
    epsinf = np.float64(5.27137) + np.float64(0.02164740) * temp - np.float64(0.00131198) * temp * temp
    epss = np.float64(78.54) * (
        np.float64(1.0)
        - np.float64(4.579e-3) * (temp - np.float64(25.0))
        + np.float64(1.190e-5) * (temp - np.float64(25.0)) * (temp - np.float64(25.0))
        - np.float64(2.800e-8) * (temp - np.float64(25.0)) * (temp - np.float64(25.0)) * (temp - np.float64(25.0))
    )
    alpha = -np.float64(16.8129) / (temp + np.float64(273.16)) + np.float64(0.0609265)
    lambdas = np.float64(0.00033836) * math.exp(np.float64(2513.98) / (temp + np.float64(273.16))) * np.float64(1.0e-2)
    ratio = lambdas / lam
    nenner = np.float64(1.0) + np.float64(2.0) * ratio ** (np.float64(1.0) - alpha) * math.sin(alpha * _PI_DOUBLE * np.float64(0.5)) + ratio ** (
        np.float64(2.0) - np.float64(2.0) * alpha
    )
    epsr = epsinf + ((epss - epsinf) * (ratio ** (np.float64(1.0) - alpha) * math.sin(alpha * _PI_DOUBLE * np.float64(0.5)) + np.float64(1.0))) / nenner
    epsi = ((epss - epsinf) * (ratio ** (np.float64(1.0) - alpha) * math.cos(alpha * _PI_DOUBLE * np.float64(0.5)))) / nenner + lam * np.float64(1.25664) / np.float64(1.88496)
    return np.complex128(np.sqrt(np.complex128(epsr - 1j * epsi)))


def m_complex_ice_maetzler(lambda_: Any, t: Any) -> np.complex128:
    """Complex refractive index of ice from WRF ``m_complex_ice_maetzler``."""

    lam = np.float64(lambda_)
    temp = np.float64(t)
    c = np.float64(2.99e8)
    tk = temp + np.float64(273.16)
    freq = c / lam * np.float64(1.0e-9)
    b1 = np.float64(0.0207)
    b2 = np.float64(1.16e-11)
    b = np.float64(335.0)
    deltabeta = math.exp(np.float64(-10.02) + np.float64(0.0364) * (tk - np.float64(273.16)))
    exp_term = math.exp(b / tk)
    betam = (b1 / tk) * (exp_term / ((exp_term - np.float64(1.0)) ** 2)) + b2 * freq * freq
    beta = betam + deltabeta
    theta = np.float64(300.0) / tk - np.float64(1.0)
    alfa = (np.float64(0.00504) + np.float64(0.0062) * theta) * math.exp(np.float64(-22.1) * theta)
    permittivity = np.complex128(np.float64(3.1884) + np.float64(9.1e-4) * (tk - np.float64(273.16)) + 1j * (alfa / freq + beta * freq))
    return np.complex128(np.sqrt(np.conjugate(permittivity)))


def m_complex_maxwellgarnett(
    vol1: Any,
    vol2: Any,
    vol3: Any,
    m1: Any,
    m2: Any,
    m3: Any,
    inclusion: str,
) -> ComplexMixResult:
    """WRF ``m_complex_maxwellgarnett`` mixing rule."""

    v1 = np.float64(vol1)
    v2 = np.float64(vol2)
    v3 = np.float64(vol3)
    inc = _fortran_string(inclusion)
    if abs(v1 + v2 + v3 - np.float64(1.0)) > np.float64(1.0e-6):
        return ComplexMixResult(np.complex128(-999.99 - 999.99j), 1)

    m1t = np.complex128(m1) ** 2
    m2t = np.complex128(m2) ** 2
    m3t = np.complex128(m3) ** 2
    if inc == "spherical":
        beta2 = np.float64(3.0) * m1t / (m2t + np.float64(2.0) * m1t)
        beta3 = np.float64(3.0) * m1t / (m3t + np.float64(2.0) * m1t)
    elif inc == "spheroidal":
        beta2 = np.float64(2.0) * m1t / (m2t - m1t) * (m2t / (m2t - m1t) * np.log(m2t / m1t) - np.float64(1.0))
        beta3 = np.float64(2.0) * m1t / (m3t - m1t) * (m3t / (m3t - m1t) * np.log(m3t / m1t) - np.float64(1.0))
    else:
        return ComplexMixResult(np.complex128(-999.99 - 999.99j), 1)

    mixed = np.sqrt(
        ((np.float64(1.0) - v2 - v3) * m1t + v2 * beta2 * m2t + v3 * beta3 * m3t)
        / (np.float64(1.0) - v2 - v3 + v2 * beta2 + v3 * beta3)
    )
    return ComplexMixResult(np.complex128(mixed), 0)


def get_m_mix(
    m_a: Any,
    m_i: Any,
    m_w: Any,
    volair: Any,
    volice: Any,
    volwater: Any,
    mixingrule: str,
    matrix: str,
    inclusion: str,
) -> ComplexMixResult:
    """WRF ``get_m_mix`` refractive-index mixing dispatcher."""

    rule = _fortran_string(mixingrule)
    mat = _fortran_string(matrix)
    if rule != "maxwellgarnett":
        return ComplexMixResult(np.complex128(1.0 + 0.0j), 2)
    if mat == "ice":
        return m_complex_maxwellgarnett(volice, volair, volwater, m_i, m_a, m_w, inclusion)
    if mat == "water":
        return m_complex_maxwellgarnett(volwater, volair, volice, m_w, m_a, m_i, inclusion)
    if mat == "air":
        return m_complex_maxwellgarnett(volair, volwater, volice, m_a, m_w, m_i, inclusion)
    return ComplexMixResult(np.complex128(1.0 + 0.0j), 1)


def get_m_mix_nested(
    m_a: Any,
    m_i: Any,
    m_w: Any,
    volair: Any,
    volice: Any,
    volwater: Any,
    mixingrule: str,
    host: str,
    matrix: str,
    inclusion: str,
    hostmatrix: str,
    hostinclusion: str,
) -> ComplexMixResult:
    """WRF ``get_m_mix_nested`` nested refractive-index mixture."""

    host_code = _fortran_string(host)
    matrix_code = _fortran_string(matrix)
    hostmatrix_code = _fortran_string(hostmatrix)
    cumulerror = 0
    value = np.complex128(1.0 + 0.0j)
    va = np.float64(volair)
    vi = np.float64(volice)
    vw = np.float64(volwater)
    ma = np.complex128(m_a)
    mi = np.complex128(m_i)
    mw = np.complex128(m_w)

    if host_code == "air":
        if matrix_code == "air":
            cumulerror += 1
        else:
            vol1 = vi / max(vi + vw, np.float64(1.0e-10))
            vol2 = np.float64(1.0) - vol1
            tmp = get_m_mix(ma, mi, mw, np.float64(0.0), vol1, vol2, mixingrule, matrix, inclusion)
            cumulerror += tmp.error
            if hostmatrix_code == "air":
                nested = get_m_mix(ma, tmp.value, np.float64(2.0) * ma, va, np.float64(1.0) - va, np.float64(0.0), mixingrule, hostmatrix, hostinclusion)
                value = nested.value
                cumulerror += nested.error
            elif hostmatrix_code == "icewater":
                nested = get_m_mix(ma, tmp.value, np.float64(2.0) * ma, va, np.float64(1.0) - va, np.float64(0.0), mixingrule, "ice", hostinclusion)
                value = nested.value
                cumulerror += nested.error
            else:
                cumulerror += 1
    elif host_code == "ice":
        if matrix_code == "ice":
            cumulerror += 1
        else:
            vol1 = va / max(va + vw, np.float64(1.0e-10))
            vol2 = np.float64(1.0) - vol1
            tmp = get_m_mix(ma, mi, mw, vol1, np.float64(0.0), vol2, mixingrule, matrix, inclusion)
            cumulerror += tmp.error
            if hostmatrix_code == "ice":
                nested = get_m_mix(tmp.value, mi, np.float64(2.0) * ma, np.float64(1.0) - vi, vi, np.float64(0.0), mixingrule, hostmatrix, hostinclusion)
                value = nested.value
                cumulerror += nested.error
            elif hostmatrix_code == "airwater":
                nested = get_m_mix(tmp.value, mi, np.float64(2.0) * ma, np.float64(1.0) - vi, vi, np.float64(0.0), mixingrule, "air", hostinclusion)
                value = nested.value
                cumulerror += nested.error
            else:
                cumulerror += 1
    elif host_code == "water":
        if matrix_code == "water":
            cumulerror += 1
        else:
            vol1 = va / max(vi + va, np.float64(1.0e-10))
            vol2 = np.float64(1.0) - vol1
            tmp = get_m_mix(ma, mi, mw, vol1, vol2, np.float64(0.0), mixingrule, matrix, inclusion)
            cumulerror += tmp.error
            if hostmatrix_code == "water":
                nested = get_m_mix(np.float64(2.0) * ma, tmp.value, mw, np.float64(0.0), np.float64(1.0) - vw, vw, mixingrule, hostmatrix, hostinclusion)
                value = nested.value
                cumulerror += nested.error
            elif hostmatrix_code == "airice":
                nested = get_m_mix(np.float64(2.0) * ma, tmp.value, mw, np.float64(0.0), np.float64(1.0) - vw, vw, mixingrule, "ice", hostinclusion)
                value = nested.value
                cumulerror += nested.error
            else:
                cumulerror += 1
    elif host_code == "none":
        nested = get_m_mix(ma, mi, mw, va, vi, vw, mixingrule, matrix, inclusion)
        value = nested.value
        cumulerror += nested.error
    else:
        cumulerror += 1

    if cumulerror != 0:
        value = np.complex128(1.0 + 0.0j)
    return ComplexMixResult(np.complex128(value), int(cumulerror))


def _radar_bins(stop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.empty(NRBINS + 1, dtype=np.float64)
    edges[0] = np.float64(100.0e-6)
    edges[-1] = np.float64(stop)
    for n in range(2, NRBINS + 1):
        edges[n - 1] = math.exp((float(n - 1) / float(NRBINS)) * math.log(edges[-1] / edges[0]) + math.log(edges[0]))
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    return edges, centers, widths


def radar_init(
    *,
    xam_r: Any = np.float32(np.float32(3.1415926536) * np.float32(1000.0) / np.float32(6.0)),
    xbm_r: Any = np.float32(3.0),
    xmu_r: Any = np.float32(0.0),
    xam_s: Any = np.float32(0.069),
    xbm_s: Any = np.float32(2.0),
    xmu_s: Any = np.float32(0.6357),
    xam_g: Any = np.float32(np.float32(3.1415926536) * np.float32(400.0) / np.float32(6.0)),
    xbm_g: Any = np.float32(3.0),
    xmu_g: Any = np.float32(0.0),
    lamda_radar: Any = LAMDA_RADAR,
) -> RadarState:
    """Create the explicit state computed by WRF ``radar_init``."""

    lam = np.float64(lamda_radar)
    pi5 = _PI_RADAR**5
    lamda4 = lam**4
    m_w_0 = m_complex_water_ray(lam, np.float64(0.0))
    m_i_0 = m_complex_ice_maetzler(lam, np.float64(0.0))
    k_w = np.float64(abs((m_w_0 * m_w_0 - np.float64(1.0)) / (m_w_0 * m_w_0 + np.float64(2.0))) ** 2)

    simpson = np.zeros(NRBINS + 1, dtype=np.float64)
    for idx in range(0, NRBINS - 1, 2):
        simpson[idx : idx + 3] += BASIS

    snow_edges, snow_centers, snow_widths = _radar_bins(0.02)
    _, graupel_centers, graupel_widths = _radar_bins(0.05)

    xr = np.float32(xam_r)
    br = np.float32(xbm_r)
    mur = np.float32(xmu_r)
    xs = np.float32(xam_s)
    bs = np.float32(xbm_s)
    mus = np.float32(xmu_s)
    xg = np.float32(xam_g)
    bg = np.float32(xbm_g)
    mug = np.float32(xmu_g)

    xcre = np.asarray((1.0 + br, 1.0 + mur, 1.0 + br + mur, 1.0 + 2.0 * br + mur), dtype=np.float32)
    xcrg = np.asarray([wgamma(value) for value in xcre], dtype=np.float32)
    xorg2 = np.float32(1.0) / xcrg[1]

    xcse = np.asarray((1.0 + bs, 1.0 + mus, 1.0 + bs + mus, 1.0 + 2.0 * bs + mus), dtype=np.float32)
    xcsg = np.asarray([wgamma(value) for value in xcse], dtype=np.float32)
    xosg2 = np.float32(1.0) / xcsg[1]

    xcge = np.asarray((1.0 + bg, 1.0 + mug, 1.0 + bg + mug, 1.0 + 2.0 * bg + mug), dtype=np.float32)
    xcgg = np.asarray([wgamma(value) for value in xcge], dtype=np.float32)
    xogg2 = np.float32(1.0) / xcgg[1]

    xobmr = np.float32(1.0) / br
    xoams = np.float32(1.0) / xs
    xobms = np.float32(1.0) / bs
    xocms = np.float32(xoams**xobms)
    xoamg = np.float32(1.0) / xg
    xobmg = np.float32(1.0) / bg
    xocmg = np.float32(xoamg**xobmg)

    return RadarState(
        xx_dx=snow_edges,
        xx_ds=snow_centers,
        xdt_s=snow_widths,
        xx_dg=graupel_centers,
        xdt_g=graupel_widths,
        lamda_radar=lam,
        k_w=k_w,
        pi5=np.float64(pi5),
        lamda4=np.float64(lamda4),
        m_w_0=np.complex128(m_w_0),
        m_i_0=np.complex128(m_i_0),
        simpson=simpson,
        basis=BASIS.copy(),
        xcre=xcre,
        xcse=xcse,
        xcge=xcge,
        xcrg=xcrg,
        xcsg=xcsg,
        xcgg=xcgg,
        xorg2=xorg2,
        xosg2=xosg2,
        xogg2=xogg2,
        xam_r=xr,
        xbm_r=br,
        xmu_r=mur,
        xobmr=xobmr,
        xam_s=xs,
        xbm_s=bs,
        xmu_s=mus,
        xoams=xoams,
        xobms=xobms,
        xocms=xocms,
        xam_g=xg,
        xbm_g=bg,
        xmu_g=mug,
        xoamg=xoamg,
        xobmg=xobmg,
        xocmg=xocmg,
    )


def rayleigh_soak_wetgraupel(
    x_g: Any,
    a_geo: Any,
    b_geo: Any,
    fmelt: Any,
    meltratio_outside: Any,
    m_w: Any,
    m_i: Any,
    lambda_: Any,
    mixingrule: str,
    matrix: str,
    inclusion: str,
    host: str,
    hostmatrix: str,
    hostinclusion: str,
    *,
    state: RadarState | None = None,
) -> np.float64:
    """Rayleigh backscattering coefficient for wet snow/graupel particles."""

    radar = radar_init(lamda_radar=lambda_) if state is None else state
    xg = np.float64(x_g)
    ageo = np.float64(a_geo)
    bgeo = np.float64(b_geo)
    fm = max(min(np.float64(fmelt), np.float64(1.0)), np.float64(0.0))
    mra = max(min(np.float64(meltratio_outside), np.float64(1.0)), np.float64(0.0))
    mra = mra + (np.float64(1.0) - mra) * fm
    x_w = xg * fm
    d_g = ageo * xg**bgeo
    if d_g < np.float64(1.0e-12):
        return np.float64(0.0)

    vg = _PI_DOUBLE / np.float64(6.0) * d_g**3
    rhog = max(min(xg / vg, np.float64(900.0)), np.float64(10.0))
    vg = xg / rhog
    outside_limit = np.float64(1.0) - rhog / np.float64(1000.0)
    if mra <= outside_limit:
        volg = vg * (np.float64(1.0) - mra * fm)
    else:
        fm_limit = (np.float64(900.0) - rhog) / (mra * np.float64(900.0) - rhog + np.float64(900.0) * rhog / np.float64(1000.0))
        if fm <= fm_limit:
            volg = (np.float64(1.0) - mra * fm) * vg
        else:
            volg = (xg - x_w) / np.float64(900.0) + x_w / np.float64(1000.0)

    d_large = (np.float64(6.0) / _PI_DOUBLE * volg) ** (np.float64(1.0) / np.float64(3.0))
    volice = (xg - x_w) / (volg * np.float64(900.0))
    volwater = x_w / (np.float64(1000.0) * volg)
    volair = np.float64(1.0) - volice - volwater
    mixed = get_m_mix_nested(
        np.complex128(1.0 + 0.0j),
        np.complex128(m_i),
        np.complex128(m_w),
        volair,
        volice,
        volwater,
        mixingrule,
        host,
        matrix,
        inclusion,
        hostmatrix,
        hostinclusion,
    )
    if mixed.error != 0:
        return np.float64(0.0)
    m_core = mixed.value
    return np.float64(abs((m_core**2 - np.float64(1.0)) / (m_core**2 + np.float64(2.0))) ** 2 * radar.pi5 * d_large**6 / radar.lamda4)


__all__ = [
    "BASIS",
    "LAMDA_RADAR",
    "MELT_OUTSIDE_G",
    "MELT_OUTSIDE_S",
    "NRBINS",
    "ComplexMixResult",
    "RadarState",
    "gammln",
    "get_m_mix",
    "get_m_mix_nested",
    "m_complex_ice_maetzler",
    "m_complex_maxwellgarnett",
    "m_complex_water_ray",
    "radar_init",
    "rayleigh_soak_wetgraupel",
    "wgamma",
]
