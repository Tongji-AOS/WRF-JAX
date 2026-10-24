"""Translation of ``module_physics_init::z2sigma``."""

from __future__ import annotations

from typing import Any

import numpy as np


def z2sigma(
    zf: Any,
    zh: Any,
    p_top: float,
    config_flags: Any,
    allowed_to_read: bool,
    kds: int,
    kde: int,
    kms: int,
    kme: int,
    kts: int,
    kte: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert WRF vertical coordinate arrays to sigma arrays.

    ``kds/kde/kms/kme`` retain their WRF 1-based index meaning. ``config_flags``,
    ``allowed_to_read``, ``kts``, and ``kte`` are part of the Fortran interface
    but are unused by the WRF 4.5 implementation.
    """

    del config_flags, allowed_to_read, kts, kte
    zf_values = np.asarray(zf, dtype=np.float32)
    zh_values = np.asarray(zh, dtype=np.float32)
    expected_size = int(kme) - int(kms) + 1
    if zf_values.ndim != 1 or zh_values.ndim != 1:
        raise ValueError("zf and zh must be one-dimensional arrays")
    if zf_values.shape[0] < expected_size or zh_values.shape[0] < expected_size:
        raise ValueError("zf and zh must cover kms:kme")

    sf = np.zeros_like(zf_values, dtype=np.float32)
    sh = np.zeros_like(zh_values, dtype=np.float32)

    def offset(k: int) -> int:
        return int(k) - int(kms)

    if zf_values[offset(int(kde) // 2)] > np.float32(1.0):
        r = np.float32(287.05)
        g = np.float32(9.80665)
        ts = np.float32(288.15)
        gamma = np.float32(-6.5 / 1000.0)
        ps = np.float32(1013.25)
        ztrop = np.float32(11000.0)
        tstrat = np.float32(ts + gamma * ztrop)
        ptrop = np.float32(ps * (tstrat / ts) ** np.float32(-g / (gamma * r)))
        ptop = np.float32(0.0)

        for k in range(int(kde), int(kds) - 1, -1):
            z = np.float32(zf_values[offset(k)])
            if z <= ztrop:
                t = np.float32(ts + gamma * z)
                p = np.float32(ps * (t / ts) ** np.float32(-g / (gamma * r)))
            else:
                p = np.float32(ptrop * np.exp(np.float32(-g * (z - ztrop) / (r * tstrat))))

            if k == int(kde):
                ptop = p

            sf[offset(k)] = np.float32((p - ptop) / (ps - ptop))

            if k != int(kds):
                z = np.float32(0.5) * np.float32(zf_values[offset(k)] + zf_values[offset(k - 1)])
                if z <= ztrop:
                    t = np.float32(ts + gamma * z)
                    p = np.float32(ps * (t / ts) ** np.float32(-g / (gamma * r)))
                else:
                    p = np.float32(ptrop * np.exp(np.float32(-g * (z - ztrop) / (r * tstrat))))
                sh[offset(k - 1)] = np.float32((p - ptop) / (ps - ptop))

        pptop = float(np.float32(ptop / np.float32(10.0)))
    else:
        for k in range(int(kde), int(kds) - 1, -1):
            sf[offset(k)] = zf_values[offset(k)]
            if k != int(kde):
                sh[offset(k)] = zh_values[offset(k)]

        pptop = float(np.float32(np.float32(p_top) / np.float32(1000.0)))

    return sf, sh, pptop


__all__ = ["z2sigma"]
