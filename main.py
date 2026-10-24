"""Run the configured WRF-JAX em_real trajectory."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
import json
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_RUN_CONFIG = Path(
    os.environ.get("WRF_PY_RUN_CONFIG", str(ROOT / "config" / "default_run.json"))
).expanduser()
if not DEFAULT_RUN_CONFIG.is_absolute():
    DEFAULT_RUN_CONFIG = ROOT / DEFAULT_RUN_CONFIG
if "WRF_PY_RUN_CONFIG" in os.environ and not DEFAULT_RUN_CONFIG.is_file():
    raise FileNotFoundError(DEFAULT_RUN_CONFIG)


def _load_bootstrap_run_config() -> dict[str, object]:
    try:
        with DEFAULT_RUN_CONFIG.open("r", encoding="ascii") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _bootstrap_project_python() -> None:
    target = os.environ.get("WRF_PY_BOOTSTRAP_PYTHON")
    if target is None:
        target = str(_load_bootstrap_run_config().get("bootstrap_python", ""))
    target = target.strip()
    if target.lower() in {"", "0", "false", "no", "off"}:
        return
    target_path = Path(target).expanduser()
    if not target_path.exists():
        return
    try:
        if target_path.resolve() == Path(sys.executable).resolve():
            return
    except OSError:
        return
    os.execv(str(target_path), [str(target_path), *sys.argv])


_bootstrap_project_python()


def _configure_run_environment() -> None:
    run_config = _load_bootstrap_run_config()
    gpu_count = run_config.get("gpu_count", 1)
    if type(gpu_count) is not int or gpu_count < 1:
        raise ValueError(f"{DEFAULT_RUN_CONFIG} gpu_count must be a positive integer")
    os.environ.setdefault("WRF_PY_GPU_COUNT", str(gpu_count))
    configured = run_config.get("environment", {})
    if not isinstance(configured, dict):
        raise ValueError(f"{DEFAULT_RUN_CONFIG} environment must be a JSON object")
    for name, value in configured.items():
        if value is None:
            continue
        if isinstance(value, bool):
            text = "1" if value else "0"
        else:
            text = str(value)
        os.environ.setdefault(str(name), text)


_configure_run_environment()

import psutil


def _configure_default_cpu_parallelism() -> None:
    threads = os.environ.get("WRF_PY_CPU_THREADS", "64").strip()
    if not threads:
        return
    inter_op_threads = os.environ.get("WRF_PY_CPU_INTER_OP_THREADS", "1").strip() or "1"
    codegen_splits = os.environ.get("WRF_PY_CPU_CODEGEN_SPLIT_COUNT", "16").strip() or "16"
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", threads)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", inter_op_threads)
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_allow_excess_precision=false "
        f"--xla_cpu_parallel_codegen_split_count={codegen_splits}",
    )
    flags = os.environ.get("XLA_FLAGS", "").strip()
    if flags.startswith("-"):
        for argument in shlex.split(flags):
            if not argument.startswith("--"):
                raise ValueError(
                    f"Invalid XLA_FLAGS argument {argument!r}: XLA silently stops "
                    "reading flags at a token without '--'. Put thread settings "
                    "in OMP_NUM_THREADS/TF_NUM_INTRAOP_THREADS/TF_NUM_INTEROP_THREADS."
                )


def _parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(item))
    return sorted(dict.fromkeys(cpus))


def _physical_cpu_candidates(allowed: set[int]) -> list[int]:
    topology = Path("/sys/devices/system/cpu")
    groups: set[tuple[int, ...]] = set()
    for path in topology.glob("cpu[0-9]*/topology/thread_siblings_list"):
        try:
            siblings = tuple(cpu for cpu in _parse_cpu_list(path.read_text(encoding="ascii").strip()) if cpu in allowed)
        except (OSError, ValueError):
            continue
        if siblings:
            groups.add(tuple(sorted(siblings)))
    if not groups:
        return sorted(allowed)
    selected = [group[0] for group in sorted(groups, key=lambda item: item[0])]
    remaining = [cpu for cpu in sorted(allowed) if cpu not in selected]
    return selected + remaining


def _numa_local_cpu_candidates(allowed: set[int], threads: int) -> list[int]:
    topology = Path("/sys/devices/system/node")
    candidates: list[list[int]] = []
    for path in sorted(topology.glob("node[0-9]*/cpulist")):
        try:
            cpus = [cpu for cpu in _parse_cpu_list(path.read_text(encoding="ascii").strip()) if cpu in allowed]
        except (OSError, ValueError):
            continue
        if cpus:
            candidates.append(sorted(cpus))
    if not candidates:
        return []
    full = [cpus for cpus in candidates if len(cpus) >= int(threads)]
    if full:
        return full[0][: int(threads)]
    return max(candidates, key=len)


def _configure_default_cpu_affinity() -> None:
    enabled = os.environ.get("WRF_PY_SET_CPU_AFFINITY", "1").strip().lower() not in {"", "0", "false", "no", "off"}
    if not enabled:
        return
    try:
        process = psutil.Process(os.getpid())
        current = process.cpu_affinity()
    except (AttributeError, OSError, psutil.Error):
        return
    if not current:
        return
    explicit = os.environ.get("WRF_PY_CPU_AFFINITY")
    if explicit:
        try:
            requested = [cpu for cpu in _parse_cpu_list(explicit) if cpu in set(current)]
        except ValueError:
            return
    else:
        raw_threads = os.environ.get("WRF_PY_CPU_THREADS", "64").strip()
        try:
            threads = max(1, int(raw_threads))
        except ValueError:
            return
        if len(current) <= threads:
            return
        policy = os.environ.get("WRF_PY_CPU_AFFINITY_POLICY", "numa").strip().lower()
        if policy == "physical":
            requested = _physical_cpu_candidates(set(current))[:threads]
        else:
            requested = _numa_local_cpu_candidates(set(current), threads)
            if len(requested) < threads:
                requested = _physical_cpu_candidates(set(current))[:threads]
    if requested and requested != current:
        try:
            process.cpu_affinity(requested)
        except (OSError, psutil.Error):
            return


_configure_default_cpu_parallelism()
_configure_default_cpu_affinity()


def _load_default_run_config() -> dict[str, object]:
    if not DEFAULT_RUN_CONFIG.exists():
        return {}
    with DEFAULT_RUN_CONFIG.open("r", encoding="ascii") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{DEFAULT_RUN_CONFIG} must contain a JSON object")
    return data


def _default_path(config: dict[str, object], name: str, fallback: Path) -> Path:
    value = config.get(name)
    return Path(str(value)).expanduser() if value not in (None, "") else fallback


def _default_int(config: dict[str, object], name: str, fallback: int) -> int:
    value = config.get(name)
    return int(value) if value is not None else fallback


def _default_optional_int(config: dict[str, object], name: str) -> int | None:
    value = config.get(name)
    return int(value) if value is not None else None


def _configure_default_jax_compilation_cache() -> None:
    cache_dir = Path(os.environ.get("WRF_PY_JAX_CACHE_DIR", str(ROOT / ".jax_cache"))).expanduser()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")


_configure_default_jax_compilation_cache()


def _configure_default_wrfjax_runtime() -> None:
    os.environ.setdefault("WRF_PY_EXTRA_OUTPUT_MODE", "key")
    os.environ.setdefault("WRF_PY_MATERIALIZE_SNAPSHOTS_DURING_TRAJECTORY", "1")
    os.environ.setdefault("WRF_PY_STREAM_SNAPSHOT_OUTPUTS", "1")
    os.environ.setdefault("WRF_PY_KEEP_STREAMED_SNAPSHOTS", "0")
    os.environ.setdefault("WRF_PY_TRAJECTORY_PROGRESS_INTERVAL_SECONDS", "600")
    os.environ.setdefault("WRF_PY_RRTMG_CHUNK_COLUMNS", "16384")
    os.environ.setdefault("WRF_PY_RRTMG_CHUNK_WORKERS", "4")
    os.environ.setdefault("WRF_PY_RRTMG_SPARSE_CLOUDS", "0")
    os.environ.setdefault("WRF_PY_THOMPSON_CHUNK_COLUMNS", "65536")
    os.environ.setdefault("WRF_PY_THOMPSON_CHUNK_WORKERS", "1")
    os.environ.setdefault("WRF_PY_TIEDTKE_CHUNK_COLUMNS", "0")
    os.environ.setdefault("WRF_PY_TIEDTKE_CHUNK_WORKERS", "4")
    os.environ.setdefault("WRF_PY_TIEDTKE_BLOCK_CHUNKS", "0")
    os.environ.setdefault("WRF_PY_TIEDTKE_ROW_REDUCTION", "domain")
    os.environ.setdefault("WRF_PY_JIT_TIEDTKE", "0")
    os.environ.setdefault("WRF_PY_JIT_CUASC", "0")
    os.environ.setdefault("WRF_PY_COMPILED_DIRECT_STAGE_FAMILIES", "1")
    os.environ.setdefault("WRF_PY_STEP_LOCAL_WORKSPACE", "1")
    os.environ.setdefault("WRF_PY_COMPILED_DIRECT_STARTUP_STEPS", "0")
    os.environ.setdefault("WRF_PY_COMPILED_DIRECT_COMPILE_STARTUP", "0")
    os.environ.setdefault("WRF_PY_COMPILED_DIRECT_EAGER_RADIATION_FALLBACK", "0")
    os.environ.setdefault("WRF_PY_COMPILED_DIRECT_SYNC_STAGES", "0")
    os.environ.setdefault(
        "WRF_PY_COMPILED_DIRECT_EAGER_STAGES",
        "rk1_phy_prep,rk1_radiation,rk1_surface_pbl,rk1_cumulus,rk1_misc,post_microphysics",
    )
    os.environ.setdefault("WRF_PY_INCLUDE_CONFIGURED_DYNAMIC_CARRY", "1")


_configure_default_wrfjax_runtime()


def _configure_configured_resource_paths(config: dict[str, object]) -> None:
    for name, env_name in (
        ("rrtmg_lw_data", "WRF_PY_RRTMG_LW_DATA"),
        ("rrtmg_sw_data", "WRF_PY_RRTMG_SW_DATA"),
    ):
        value = config.get(name)
        if value not in (None, ""):
            os.environ.setdefault(env_name, str(Path(str(value)).expanduser()))


_DEFAULT_RUN = _load_default_run_config()
_configure_configured_resource_paths(_DEFAULT_RUN)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax

from wrf_jax.run_timing import write_timing_summary
from wrf_jax.core.devices import configured_devices
from wrf_jax.progress_logging import progress_logging
from wrf_jax.trajectory import (
    DEFAULT_OUTPUT_ROOT as TRAJECTORY_DEFAULT_OUTPUT_ROOT,
    PrognosticSnapshot,
    compare_snapshots_to_wrfout,
    materialize_snapshots,
    new_wrfjax_output_dir,
    run_compiled_direct_trajectory,
    run_direct_trajectory,
    run_strict_trajectory,
    snapshot_nonfinite_summaries,
    write_wrfjax_milestone_outputs,
)
from wrf_jax.runtime import preload_runtime_jax_tables

DEFAULT_OUTPUT_ROOT = _default_path(_DEFAULT_RUN, "output_root", TRAJECTORY_DEFAULT_OUTPUT_ROOT)
_FALLBACK_CASE_DIR = ROOT / "case"
DEFAULT_CASE_DIR = _default_path(_DEFAULT_RUN, "case_dir", _FALLBACK_CASE_DIR)
DEFAULT_WRFOUT = _default_path(
    _DEFAULT_RUN,
    "wrfout",
    DEFAULT_CASE_DIR / "wrfout_d01_2021-06-27_12:00:00",
)
DEFAULT_HOURS = _default_int(_DEFAULT_RUN, "hours", 72)
DEFAULT_DURATION_SECONDS = _default_optional_int(_DEFAULT_RUN, "duration_seconds")
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = _default_int(_DEFAULT_RUN, "snapshot_interval_seconds", 3600)
DEFAULT_MODE = str(_DEFAULT_RUN.get("mode", "direct"))


@dataclass(frozen=True)
class CpuStats:
    mean_percent: float
    max_percent: float
    sample_count: int


@dataclass(frozen=True)
class PhaseTimings:
    integrate_seconds: float
    materialize_seconds: float
    compare_seconds: float
    write_outputs_seconds: float


RUN_ENVIRONMENT_KEYS = (
    "WRF_PY_GPU_COUNT",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "WRF_PY_ENABLE_JAX_X64",
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "WRF_PY_CPU_THREADS",
    "WRF_PY_CPU_INTER_OP_THREADS",
    "WRF_PY_CPU_CODEGEN_SPLIT_COUNT",
    "WRF_PY_EXTRA_OUTPUT_MODE",
    "WRF_PY_MATERIALIZE_SNAPSHOTS_DURING_TRAJECTORY",
    "WRF_PY_STREAM_SNAPSHOT_OUTPUTS",
    "WRF_PY_KEEP_STREAMED_SNAPSHOTS",
    "WRF_PY_COMPARE_STREAMED_SNAPSHOTS",
    "WRF_PY_TRAJECTORY_PROGRESS_INTERVAL_SECONDS",
    "WRF_PY_RRTMG_CHUNK_COLUMNS",
    "WRF_PY_RRTMG_CHUNK_WORKERS",
    "WRF_PY_RRTMG_SPARSE_CLOUDS",
    "WRF_PY_THOMPSON_CHUNK_COLUMNS",
    "WRF_PY_THOMPSON_CHUNK_WORKERS",
    "WRF_PY_TIEDTKE_CHUNK_COLUMNS",
    "WRF_PY_TIEDTKE_CHUNK_WORKERS",
    "WRF_PY_TIEDTKE_BLOCK_CHUNKS",
    "WRF_PY_PROFILE_DIRECT_STEP",
    "WRF_PY_JIT_TIEDTKE",
    "WRF_PY_JIT_CUASC",
    "WRF_PY_DIRECT_SMALL_STEP_LOOP",
    "WRF_PY_COMPILED_DIRECT_BLOCK_STEPS",
    "WRF_PY_COMPILED_DIRECT_SCAN_UNROLL",
    "WRF_PY_COMPILED_DIRECT_STARTUP_STEPS",
    "WRF_PY_COMPILED_DIRECT_COMPILE_STARTUP",
    "WRF_PY_COMPILED_DIRECT_STAGE_FAMILIES",
    "WRF_PY_COMPILED_DIRECT_STAGE_SCAN",
    "WRF_PY_COMPILED_DIRECT_SEGMENT_COLUMN_PHYSICS",
    "WRF_PY_COMPILED_DIRECT_SCAN_COMPILED_CALLS",
    "WRF_PY_STEP_LOCAL_WORKSPACE",
    "WRF_PY_COMPILED_DIRECT_EAGER_STAGES",
    "WRF_PY_COMPILED_DIRECT_SYNC_STAGES",
    "WRF_PY_COMPILED_DIRECT_EAGER_RK1_PHYSICS",
    "WRF_PY_COMPILED_DIRECT_EAGER_RADIATION_FALLBACK",
    "WRF_PY_COMPILED_DIRECT_PROBE_STEP",
    "WRF_PY_COMPILED_DIRECT_SNAPSHOT_SCAN",
    "WRF_PY_COMPILED_STARTUP_STEADY_ONLY",
    "WRF_PY_DONATE_COMPILED_STARTUP",
    "WRF_PY_INCLUDE_CONFIGURED_DYNAMIC_CARRY",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "XLA_FLAGS",
)


def _device_details() -> dict[str, str]:
    import jax
    import jaxlib

    devices = jax.local_devices()
    details = {
        "python_executable": sys.executable,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": "; ".join(f"{device}: {device.device_kind}" for device in devices),
        "integration_devices": "; ".join(str(device) for device in configured_devices()),
        "jax_enable_x64": str(jax.config.x64_enabled),
        "cpu_affinity": ",".join(str(cpu) for cpu in psutil.Process().cpu_affinity()),
    }
    for index, device in enumerate(devices):
        stats = device.memory_stats() or {}
        for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
            if name in stats:
                details[f"device_{index}_{name}"] = str(stats[name])
    return details


class CpuSampler:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "CpuSampler":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="ascii")
        self._file.write("seconds_since_start,cpu_percent\n")
        self._file.flush()
        self._start = perf_counter()
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._file.close()

    def _sample_loop(self) -> None:
        while not self._stop.wait(1.0):
            value = float(self._process.cpu_percent(interval=None))
            self._samples.append(value)
            self._file.write(f"{perf_counter() - self._start:.3f},{value:.3f}\n")
            self._file.flush()

    @property
    def stats(self) -> CpuStats:
        if not self._samples:
            return CpuStats(mean_percent=0.0, max_percent=0.0, sample_count=0)
        return CpuStats(
            mean_percent=sum(self._samples) / len(self._samples),
            max_percent=max(self._samples),
            sample_count=len(self._samples),
        )


class GpuSampler:
    """Stream NVIDIA device-wide samples; never synchronize model arrays."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.process = None
        self.file = None
        self.error_file = None
        self.status = "not_applicable"

    def __enter__(self) -> "GpuSampler":
        import jax

        if jax.default_backend() != "gpu":
            return self
        self.file = self.path.open("w", encoding="ascii")
        self.error_file = self.path.with_suffix(".log").open("w", encoding="ascii")
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
                 "--format=csv,nounits", "--loop=1"],
                stdout=self.file, stderr=self.error_file,
            )
            self.status = "device_wide_nvidia_smi"
        except OSError as error:
            self.status = "unavailable"
            self.error_file.write(str(error) + "\n")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            elif self.process.returncode:
                self.status = "unavailable"
        for handle in (self.file, self.error_file):
            if handle is not None:
                handle.close()

    @property
    def stats(self) -> dict[str, str]:
        result = {"gpu_monitor": self.status}
        if self.status != "device_wide_nvidia_smi":
            return result
        devices: dict[str, list[tuple[float, float, float, float]]] = {}
        with self.path.open(encoding="ascii") as stream:
            for row in csv.reader(stream):
                if len(row) != 6:
                    continue
                try:
                    values = tuple(float(x.strip()) for x in row[2:])
                except ValueError:
                    continue
                devices.setdefault(row[1].strip(), []).append(values)
        for index, (uuid, samples) in enumerate(devices.items()):
            prefix = f"gpu_sampled_{index}"
            result[prefix + "_uuid"] = uuid
            result[prefix + "_samples"] = str(len(samples))
            for column, name in enumerate(("utilization_percent", "memory_activity_percent", "memory_used_mib", "power_w")):
                values = [row[column] for row in samples]
                result[f"{prefix}_{name}_mean"] = f"{sum(values) / len(values):.3f}"
                result[f"{prefix}_{name}_max"] = f"{max(values):.3f}"
        return result


@contextmanager
def _python_profile(output_dir: Path):
    """Optional host profile configured in the same file as the model run."""
    if not _DEFAULT_RUN.get("profile_python", False):
        yield
        return
    import cProfile
    import pstats

    profile = cProfile.Profile()
    profile.enable()
    try:
        yield
    finally:
        profile.disable()
        with (output_dir / "python_profile.txt").open("w", encoding="utf-8") as stream:
            stream.write("Host profile; profiling overhead is included in this run.\n")
            stream.write("Cumulative times overlap across nested calls.\n\n")
            statistics = pstats.Stats(profile, stream=stream)
            statistics.sort_stats("cumulative").print_stats(70)
            statistics.sort_stats("tottime").print_stats(40)


def _write_run_summary(
    path: Path,
    *,
    output_dir: Path,
    return_code: int,
    wall_seconds: float,
    phase_timings: PhaseTimings,
    cpu: CpuStats,
    case_dir: Path,
    wrfout: Path,
    duration_seconds: int,
    snapshot_interval_seconds: int,
    mode: str,
    runtime_metrics: dict[str, float] | None = None,
    gpu_stats: dict[str, str] | None = None,
) -> dict[str, float]:
    warmup_seconds = _DEFAULT_RUN.get("sustained_warmup_seconds", 3600)
    profile_window = _DEFAULT_RUN.get("profile_window_seconds")
    if profile_window is not None:
        warmup_seconds = max(warmup_seconds, profile_window[1])
    sustained_timings = write_timing_summary(output_dir, wall_seconds, warmup_seconds=warmup_seconds)
    lines = [
        f"output_dir={output_dir}",
        f"return_code={return_code}",
        f"wall_seconds={wall_seconds:.3f}",
        f"integrate_seconds={phase_timings.integrate_seconds:.3f}",
        f"materialize_seconds={phase_timings.materialize_seconds:.3f}",
        f"compare_seconds={phase_timings.compare_seconds:.3f}",
        f"write_outputs_seconds={phase_timings.write_outputs_seconds:.3f}",
        f"cpu_mean_percent={cpu.mean_percent:.3f}",
        f"cpu_max_percent={cpu.max_percent:.3f}",
        f"cpu_samples={cpu.sample_count}",
        f"case_dir={case_dir}",
        f"wrfout={wrfout}",
        f"default_run_config={DEFAULT_RUN_CONFIG if DEFAULT_RUN_CONFIG.exists() else 'none'}",
        f"duration_seconds={duration_seconds}",
        f"snapshot_interval_seconds={snapshot_interval_seconds}",
        f"mode={mode}",
    ]
    lines.extend(f"{name}={value:.3f}" for name, value in sustained_timings.items())
    if not sustained_timings:
        lines.append("sustained_integration_seconds=not_measured")
    lines.extend(f"{name}={value}" for name, value in _device_details().items())
    lines.extend(f"{name}={value}" for name, value in (gpu_stats or {}).items())
    for name in RUN_ENVIRONMENT_KEYS:
        if name in os.environ:
            lines.append(f"env_{name}={os.environ[name]}")
    if runtime_metrics:
        for name in sorted(runtime_metrics):
            lines.append(f"{name}={runtime_metrics[name]:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return sustained_timings


def _write_run_log(path: Path, args: argparse.Namespace, output_dir: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"case_dir={args.case_dir}",
        f"wrfout={args.wrfout}",
        f"hours={args.hours}",
        f"duration_seconds={args.duration_seconds}",
        f"snapshot_interval_seconds={args.snapshot_interval_seconds}",
        f"mode={args.mode}",
        f"output_dir={output_dir}",
        f"default_run_config={DEFAULT_RUN_CONFIG if DEFAULT_RUN_CONFIG.exists() else 'none'}",
    ]
    for name in RUN_ENVIRONMENT_KEYS:
        if name in os.environ:
            lines.append(f"env_{name}={os.environ[name]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.open("a", encoding="utf-8")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _stream_snapshot_outputs(snapshot_interval_seconds: int) -> bool:
    return _env_flag("WRF_PY_STREAM_SNAPSHOT_OUTPUTS", snapshot_interval_seconds >= 3600)


def _keep_streamed_snapshots() -> bool:
    if "keep_streamed_snapshots" in _DEFAULT_RUN:
        return bool(_DEFAULT_RUN["keep_streamed_snapshots"])
    return _env_flag("WRF_PY_KEEP_STREAMED_SNAPSHOTS", False)


def _compare_streamed_snapshots(snapshot_interval_seconds: int) -> bool:
    if "compare_streamed_snapshots" in _DEFAULT_RUN:
        return bool(_DEFAULT_RUN["compare_streamed_snapshots"])
    return _env_flag("WRF_PY_COMPARE_STREAMED_SNAPSHOTS", snapshot_interval_seconds >= 3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS, help="model hours to integrate when --duration-seconds is omitted")
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="exact model duration; overrides --hours",
    )
    parser.add_argument(
        "--snapshot-interval-seconds",
        type=int,
        default=DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        help="model seconds between output snapshots",
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR, help="WRF em_real case directory")
    parser.add_argument("--wrfout", type=Path, default=DEFAULT_WRFOUT, help="reference WRF output for comparison")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"directory that receives wrfjax-* outputs; default is {DEFAULT_OUTPUT_ROOT}/<git-branch>",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(_DEFAULT_RUN["out_dir"]) if _DEFAULT_RUN.get("out_dir") else None,
        help="explicit output directory",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "direct", "compiled-direct"),
        default=DEFAULT_MODE,
        help="trajectory implementation",
    )
    return parser.parse_args()


def _profile_window_seconds(config, duration_seconds, snapshot_interval_seconds):
    window = config.get("profile_window_seconds")
    if window is None:
        return None
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(value) is not int for value in window)
        or snapshot_interval_seconds <= 0
    ):
        raise ValueError("profile_window_seconds must be two integer snapshot times")
    start, end = window
    if start <= 0 or start % snapshot_interval_seconds or end != start + snapshot_interval_seconds or end > duration_seconds:
        raise ValueError("profile_window_seconds must select adjacent snapshots within the forecast")
    return start, end


def main() -> None:
    args = parse_args()
    configured_devices()  # Validate the requested count before loading a case.
    duration_seconds = int(args.duration_seconds) if args.duration_seconds is not None else int(args.hours) * 3600
    if duration_seconds <= 0:
        raise ValueError("duration must be positive")
    profile_window = _profile_window_seconds(_DEFAULT_RUN, duration_seconds, int(args.snapshot_interval_seconds))
    if profile_window is not None and (
        args.mode not in {"direct", "compiled-direct"}
        or not _stream_snapshot_outputs(int(args.snapshot_interval_seconds))
    ):
        raise ValueError("profiling requires streamed trajectory snapshots")
    if (
        args.out_dir is None
        and duration_seconds < 3600
        and not _stream_snapshot_outputs(int(args.snapshot_interval_seconds))
    ):
        raise ValueError("default wrfjax-* outputs require streamed snapshots or at least 3600s of model time")

    output_dir = args.out_dir if args.out_dir is not None else new_wrfjax_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(_DEFAULT_RUN, indent=2) + "\n", encoding="ascii"
    )
    run_log_path = output_dir / "run.log"
    cpu_path = output_dir / "cpu_usage.txt"
    summary_path = output_dir / "run_summary.txt"
    snapshot_index_path = output_dir / "snapshot_index.txt"

    print(f"output_dir={output_dir}", flush=True)
    start = perf_counter()
    memory_profile = nullcontext()
    if _DEFAULT_RUN.get("profile_device_memory", False):
        from tools.diagnostics.profile_device_memory import device_memory_profile
        memory_profile = device_memory_profile(output_dir)
    with _write_run_log(run_log_path, args, output_dir) as log, CpuSampler(cpu_path) as sampler, GpuSampler(output_dir / "gpu_usage.txt") as gpu_sampler, _python_profile(output_dir), memory_profile:
        for name, value in _device_details().items():
            line = f"{name}={value}"
            print(line, flush=True)
            log.write(line + "\n")
        log.flush()
        os.environ["WRF_PY_TRAJECTORY_PROGRESS_FILE"] = str(run_log_path)
        os.environ["WRF_PY_TRAJECTORY_TIMING_FILE"] = str(output_dir / "timing_events.txt")
        phase_start = perf_counter()
        print("preload_runtime_jax_tables=start", flush=True)
        log.write("preload_runtime_jax_tables=start\n")
        log.flush()
        preload_runtime_jax_tables(args.case_dir)
        preload_seconds = perf_counter() - phase_start
        print(f"preload_runtime_jax_tables_seconds={preload_seconds:.3f}", flush=True)
        log.write(f"preload_runtime_jax_tables_seconds={preload_seconds:.3f}\n")
        log.flush()
        if args.mode == "compiled-direct":
            runner = run_compiled_direct_trajectory
        else:
            runner = run_direct_trajectory if args.mode == "direct" else run_strict_trajectory
        phase_start = perf_counter()
        stream_outputs = _stream_snapshot_outputs(int(args.snapshot_interval_seconds))
        keep_streamed_snapshots = _keep_streamed_snapshots()
        compare_streamed_snapshots = _compare_streamed_snapshots(int(args.snapshot_interval_seconds))
        # Integration-only runs must not open a history file still being written.
        snapshot_metadata_path = args.wrfout if compare_streamed_snapshots else args.case_dir / "wrfinput_d01"
        if stream_outputs:
            snapshot_index_path.write_text(
                "elapsed_seconds,timestamp,output_dir,"
                "integrate_wall_since_previous_snapshot_seconds,write_seconds,fields_compared,nonfinite_fields\n",
                encoding="ascii",
            )
        last_stream_checkpoint = {"time": perf_counter()}
        profile_active = False

        def write_streamed_snapshot(snapshot: PrognosticSnapshot) -> None:
            nonlocal profile_active
            # Pending device computation belongs to integration, not writing.
            jax.block_until_ready((snapshot.u, snapshot.v, snapshot.t, snapshot.qvapor, snapshot.extra))
            integrate_wall_seconds = perf_counter() - last_stream_checkpoint["time"]
            if profile_active:
                jax.profiler.stop_trace()
                profile_active = False
                print(f"integration_profile_stop elapsed_seconds={snapshot.elapsed_seconds}", flush=True)
            started = perf_counter()
            streamed_snapshot = materialize_snapshots((snapshot,))[0]
            streamed_comparisons = (
                compare_snapshots_to_wrfout((streamed_snapshot,), args.wrfout) if compare_streamed_snapshots else None
            )
            elapsed_seconds = int(streamed_snapshot.elapsed_seconds)
            snapshot_dir = output_dir / "snapshots" / f"elapsed-{elapsed_seconds:06d}"
            streamed_milestone = write_wrfjax_milestone_outputs(
                (streamed_snapshot,),
                snapshot_dir,
                streamed_comparisons,
                reference_path=snapshot_metadata_path,
            )
            write_seconds = perf_counter() - started
            fields_compared = len(streamed_comparisons) if streamed_comparisons is not None else 0
            nonfinite_fields = {
                detail.split(":", 1)[0]
                for detail in snapshot_nonfinite_summaries(streamed_snapshot)
            }
            if streamed_comparisons is not None:
                nonfinite_fields.update(item.field for item in streamed_comparisons if not item.finite)
            nonfinite_fields = sorted(nonfinite_fields)
            line = (
                f"streamed_snapshot elapsed_seconds={streamed_snapshot.elapsed_seconds} "
                f"time={streamed_snapshot.timestamp} output_dir={streamed_milestone.output_dir} "
                f"integrate_wall_since_previous_snapshot_seconds={integrate_wall_seconds:.3f} "
                f"write_seconds={write_seconds:.3f} fields_compared={fields_compared} "
                f"nonfinite_fields={','.join(nonfinite_fields) if nonfinite_fields else 'none'}"
            )
            print(line, flush=True)
            log.write(line + "\n")
            with snapshot_index_path.open("a", encoding="ascii") as index:
                index.write(
                    f"{streamed_snapshot.elapsed_seconds},{streamed_snapshot.timestamp},"
                    f"{streamed_milestone.output_dir},{integrate_wall_seconds:.3f},"
                    f"{write_seconds:.3f},{fields_compared},"
                    f"{';'.join(nonfinite_fields) if nonfinite_fields else 'none'}\n"
                )
            log.flush()
            if nonfinite_fields and _DEFAULT_RUN.get("fail_on_nonfinite", False):
                raise FloatingPointError(
                    f"nonfinite snapshot at {elapsed_seconds}s: {','.join(nonfinite_fields)}; "
                    f"output preserved in {snapshot_dir}"
                )
            if profile_window is not None and elapsed_seconds == profile_window[0]:
                options = jax.profiler.ProfileOptions()
                options.python_tracer_level = 0
                options.host_tracer_level = 1
                jax.profiler.start_trace(str(output_dir / "jax_trace"), profiler_options=options)
                profile_active = True
                print(f"integration_profile_start elapsed_seconds={elapsed_seconds}; instrumented interval is not a speed benchmark", flush=True)
            last_stream_checkpoint["time"] = perf_counter()

        runner_kwargs = {
            "duration_seconds": duration_seconds,
            "snapshot_interval_seconds": int(args.snapshot_interval_seconds),
        }
        if args.mode in {"direct", "compiled-direct"}:
            runner_kwargs["snapshot_callback"] = write_streamed_snapshot if stream_outputs else None
            runner_kwargs["keep_snapshots"] = keep_streamed_snapshots if stream_outputs else True
        try:
            with progress_logging():
                result = runner(args.case_dir, **runner_kwargs)
        finally:
            if profile_active:
                jax.profiler.stop_trace()
        integrate_seconds = perf_counter() - phase_start
        phase_start = perf_counter()
        snapshots = materialize_snapshots(result.snapshots)
        materialize_seconds = perf_counter() - phase_start
        phase_start = perf_counter()
        comparisons = compare_snapshots_to_wrfout(snapshots, args.wrfout) if snapshots else tuple()
        compare_seconds = perf_counter() - phase_start
        phase_start = perf_counter()
        milestone = (
            write_wrfjax_milestone_outputs(
                snapshots,
                output_dir,
                comparisons,
                reference_path=args.wrfout,
            )
            if snapshots
            else None
        )
        write_outputs_seconds = perf_counter() - phase_start
        phase_timings = PhaseTimings(
            integrate_seconds=integrate_seconds,
            materialize_seconds=materialize_seconds,
            compare_seconds=compare_seconds,
            write_outputs_seconds=write_outputs_seconds,
        )
        runtime_metrics = result.runtime_metrics or {}
        for name, value in (
            ("integrate_seconds", integrate_seconds),
            ("materialize_seconds", materialize_seconds),
            ("compare_seconds", compare_seconds),
            ("write_outputs_seconds", write_outputs_seconds),
        ):
            line = f"{name}={value:.3f}"
            print(line, flush=True)
            log.write(line + "\n")
        for name in sorted(runtime_metrics):
            line = f"{name}={runtime_metrics[name]:.3f}"
            print(line, flush=True)
            log.write(line + "\n")
        for item in comparisons:
            line = (
                f"elapsed={item.elapsed_seconds}s time={item.timestamp} field={item.field} "
                f"nRMSE%={item.nrmse_percent:.8g} RMSE={item.rmse:.8g} units={item.units} "
                f"max_abs={item.max_abs_error:.8g} finite={item.finite}"
            )
            print(line, flush=True)
            log.write(line + "\n")
        if milestone is not None:
            log.write(f"netcdf={milestone.netcdf_path}\n")
            if milestone.comparison_path is not None:
                log.write(f"comparison={milestone.comparison_path}\n")
            if milestone.nrmse_summary_path is not None:
                log.write(f"nrmse_summary={milestone.nrmse_summary_path}\n")
            if milestone.goal_check_path is not None:
                log.write(f"goal_check={milestone.goal_check_path}\n")
            if milestone.diagnostic_fields_path is not None:
                log.write(f"diagnostic_fields={milestone.diagnostic_fields_path}\n")
        else:
            log.write("netcdf=streamed_snapshots_only\n")
            log.write(f"snapshot_index={snapshot_index_path}\n")
        log.flush()
    wall = perf_counter() - start
    cpu = sampler.stats
    sustained_timings = _write_run_summary(
        summary_path,
        output_dir=output_dir,
        return_code=0,
        wall_seconds=wall,
        phase_timings=phase_timings,
        cpu=cpu,
        case_dir=args.case_dir,
        wrfout=args.wrfout,
        duration_seconds=duration_seconds,
        snapshot_interval_seconds=int(args.snapshot_interval_seconds),
        mode=args.mode,
        runtime_metrics=runtime_metrics,
        gpu_stats=gpu_sampler.stats,
    )
    print(f"wall_seconds={wall:.3f}", flush=True)
    for name, value in sustained_timings.items():
        print(f"{name}={value:.3f}", flush=True)
    if not sustained_timings:
        print("sustained_integration_seconds=not_measured", flush=True)
    print(f"timing_summary={output_dir / 'timing_summary.md'}", flush=True)
    print(f"cpu_mean_percent={cpu.mean_percent:.3f}", flush=True)
    print(f"cpu_max_percent={cpu.max_percent:.3f}", flush=True)


if __name__ == "__main__":
    main()
