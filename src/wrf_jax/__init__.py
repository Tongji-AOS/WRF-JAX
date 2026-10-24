"""WRF-JAX implementation of the configured WRF 4.5 model."""

from __future__ import annotations

import os
from pathlib import Path


def _configure_default_cpu_threads() -> None:
    threads = os.environ.get("WRF_PY_CPU_THREADS", "64").strip() or "64"
    inter_op_threads = os.environ.get("WRF_PY_CPU_INTER_OP_THREADS", "1").strip() or "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, threads)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", threads)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", inter_op_threads)
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true --xla_allow_excess_precision=false",
    )


_configure_default_cpu_threads()

import jax


def _configure_jax_precision() -> None:
    enable_x64 = os.environ.get("WRF_PY_ENABLE_JAX_X64", "1").strip().lower() in {"1", "true", "yes", "on"}
    jax.config.update("jax_enable_x64", enable_x64)


_configure_jax_precision()


def _configure_jax_compilation_cache() -> None:
    cache_enabled = os.environ.get("WRF_PY_ENABLE_JAX_COMPILATION_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not cache_enabled:
        jax.config.update("jax_enable_compilation_cache", False)
        return
    cache_dir = os.environ.get("WRF_PY_JAX_COMPILATION_CACHE_DIR") or os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parents[2] / ".jax_compilation_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    min_compile_time = float(os.environ.get("WRF_PY_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.0"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time)


_configure_jax_compilation_cache()
