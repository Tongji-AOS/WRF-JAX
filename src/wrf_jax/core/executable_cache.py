"""Reuse compiled inference programs without retracing their Python builders.

This cache contains executables, never forecast states. Callers must supply all
closed-over configuration in ``context``; array arguments remain runtime inputs.
Entries are local to the source, JAX/compiler settings, device and input schema.
Like JAX's compilation cache, the directory must be writable only by trusted users.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timedelta
import fcntl
from functools import lru_cache, partial, wraps
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from time import perf_counter
from types import CodeType, FunctionType, SimpleNamespace
from typing import Any, Callable

import jax
import jaxlib
import numpy as np


def _context_value(value: Any) -> Any:
    """Encode supported static values by content, never by object address."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return (type(value).__name__, value)
    if value is Ellipsis:
        return ("ellipsis",)
    if isinstance(value, CodeType):
        # marshal's reference flags change when a constant tuple is kept alive.
        # Encode bytecode and constants by value, excluding debug locations.
        return ("code", value.co_argcount, value.co_posonlyargcount,
                value.co_kwonlyargcount, value.co_flags, value.co_code,
                _context_value(value.co_consts), value.co_names, value.co_varnames,
                value.co_freevars, value.co_cellvars, value.co_exceptiontable)
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot key an executable")
        return ("array", array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest())
    if isinstance(value, dict):
        return ("dict", tuple((key, _context_value(value[key])) for key in sorted(value)))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_context_value(item) for item in value))
    if isinstance(value, (frozenset, set)):
        return (type(value).__name__, tuple(sorted(_context_value(item) for item in value)))
    if isinstance(value, (Path, date, datetime, timedelta, np.dtype)):
        return (type(value).__name__, str(value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, FunctionType):
        # The package source digest covers the function body and module globals.
        # Generated callbacks also capture field names and other static choices.
        return (
            "function", value.__module__, value.__qualname__,
            _context_value(value.__defaults__), _context_value(value.__kwdefaults__),
            tuple(_context_value(cell.cell_contents) for cell in value.__closure__ or ()),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__module__, type(value).__qualname__,
                tuple((field.name, _context_value(getattr(value, field.name))) for field in fields(value)))
    if isinstance(value, SimpleNamespace):
        return ("namespace", _context_value(vars(value)))
    raise TypeError(f"unsupported executable context: {type(value).__module__}.{type(value).__qualname__}")


@lru_cache(maxsize=1)
def _source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py")) + sorted((root / "physics" / "data").rglob("*"))
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _program_key(args: tuple[Any, ...], context: Any) -> str:
    # Pallas registers compiler options on first import. Register them before
    # hashing config so a cache load and a fresh trace produce identical keys.
    from jax.experimental import pallas as _pallas  # noqa: F401

    leaves, tree = jax.tree_util.tree_flatten(args)
    devices = jax.devices()
    signature = []
    for leaf in leaves:
        aval = jax.typeof(leaf)
        signature.append((tuple(aval.shape), str(aval.dtype), bool(aval.weak_type), str(getattr(leaf, "sharding", None))))
    environment = {name: value for name, value in sorted(os.environ.items())
                   if name.startswith(("WRF_PY_", "JAX_", "XLA_"))
                   # Device count selects host batches; shape and sharding above
                   # already distinguish the compiled work on each device.
                   and name != "WRF_PY_GPU_COUNT"
                   and not name.endswith(("_FILE", "_DIR", "_PYTHON"))}
    def tree_schema(node):
        data = node.node_data()
        return None if data is None else (_context_value(data), tuple(tree_schema(child) for child in node.children()))

    def json_value(value):
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, complex):
            return {"complex": [value.real, value.imag]}
        raise TypeError(f"cannot encode cache key value {type(value)}")

    payload = (
        2, _source_digest(), sys.version_info[:3], jax.__version__, jaxlib.__version__, np.__version__,
        tuple((device.platform, device.device_kind, device.id, device.client.platform_version) for device in devices),
        environment, _context_value(jax.config.values), tree_schema(tree), signature, _context_value(context),
    )
    # Pickle memoizes object identity: equal strings/tuples with different
    # sharing gave different hashes across process starts. Cache keys encode
    # only values and types; the program artifact itself remains serialized.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_value)
    return hashlib.sha256(encoded.encode()).hexdigest()


def compile_model_program(
    function: Any,
    *args: Any,
    context: Any,
    cache_dir: Path | None = None,
    report: Callable[[str], None] = partial(print, flush=True),
    call_kwargs: dict[str, Any] | None = None,
    static_argnames: tuple[str, ...] = (),
    kernel_family: tuple[str, str] | None = None,
) -> Any:
    """Load the exact compiled program or compile and atomically save it.

    Unsupported contexts/backends and unreadable entries use normal compilation.
    CPU uses JAX's standard compilation cache: serialized CPU executables have
    failed symbol resolution even with a single code generation partition.
    Compilation/execution failures themselves are not hidden by this helper.
    """
    from jax.experimental import serialize_executable

    kwargs = {} if call_kwargs is None else call_kwargs
    compiler_options = None
    compile_now = lambda: function.lower(*args, **kwargs).compile(compiler_options=compiler_options)
    enabled = os.environ.get("WRF_PY_EXECUTABLE_CACHE", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled and cache_dir is None:
        return compile_now()
    if jax.default_backend() == "cpu":
        report("executable cache bypass: CPU uses the standard JAX compilation cache")
        return compile_now()
    directory = cache_dir
    if directory is None:
        parent = jax.config.jax_compilation_cache_dir
        if not parent:
            report("executable cache bypass: no compilation cache directory configured")
            return compile_now()
        directory = Path(parent) / "model_programs"
    try:
        builder = getattr(function, "__wrapped__", function)
        static_values = {name: kwargs[name] for name in static_argnames if name in kwargs}
        dynamic_kwargs = {name: value for name, value in kwargs.items() if name not in static_values}
        key = _program_key(args if call_kwargs is None else (args, dynamic_kwargs), {
            "builder": (builder.__module__, builder.__qualname__, builder.__code__),
            "context": context,
            "static_keywords": static_values,
        })
        directory.mkdir(parents=True, exist_ok=True)
    except (TypeError, ValueError, OSError, RecursionError) as error:
        report(f"executable cache bypass: {error}")
        return compile_now()
    path = directory / f"{key}.pkl"
    if path.exists():
        try:
            started = perf_counter()
            with path.open("rb") as stream:
                saved_key, executable, input_tree, output_tree, device_ids, kernel_name = pickle.load(stream)
            if saved_key != key:
                raise ValueError("executable cache key mismatch")
            guard = _kernel_cache_lock(directory / kernel_name) if kernel_name else nullcontext()
            with guard:
                devices_by_id = {device.id: device for device in jax.local_devices()}
                execution_devices = [devices_by_id[index] for index in device_ids]
                compiled = serialize_executable.deserialize_and_load(
                    executable, input_tree, output_tree, execution_devices=execution_devices,
                )
        except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, EOFError) as error:
            report(f"executable cache rebuild: {error}")
        else:
            report(f"executable cache loaded key={key} seconds={perf_counter() - started:.3f}")
            return compiled
    started = perf_counter()
    lowered = function.lower(*args, **kwargs)
    lower_seconds = perf_counter() - started
    kernel_name = None
    if "cuda" in jax.devices()[0].client.platform_version.lower():
        # A source/context change still invalidates the executable above, but
        # unchanged lowered computations can reuse CUDA kernels and JAX cache.
        try:
            kernel_key = _kernel_key(lowered, kernel_family or (builder.__module__, builder.__qualname__))
        except (AttributeError, TypeError, ValueError) as error:
            report(f"kernel cache uses source key: {error}")
            kernel_key = key
        kernel_name = f"{kernel_key}.kernels"
        compiler_options = {
            "xla_gpu_kernel_cache_file": str(directory.resolve() / kernel_name),
            "xla_gpu_enable_llvm_module_compilation_parallelism": True,
        }
    # XLA 0.9 writes its kernel file in place. Related model programs may
    # compile in different threads/processes, so serialize access to that file.
    guard = (
        _kernel_cache_lock(Path(compiler_options["xla_gpu_kernel_cache_file"]))
        if compiler_options is not None else nullcontext()
    )
    started = perf_counter()
    with guard:
        compiled = lowered.compile(compiler_options=compiler_options)
    report(f"executable compilation builder={builder.__qualname__} "
           f"lower_seconds={lower_seconds:.3f} backend_seconds={perf_counter() - started:.3f}")
    temporary = None
    try:
        executable, input_tree, output_tree = serialize_executable.serialize(compiled)
        device_ids = tuple(device.id for device in compiled.runtime_executable().local_devices())
        with tempfile.NamedTemporaryFile(dir=directory, prefix=key, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump((key, executable, input_tree, output_tree, device_ids, kernel_name), stream, protocol=5)
        temporary.replace(path)
        report(f"executable cache saved key={key} bytes={path.stat().st_size}")
    except (OSError, ValueError, RuntimeError, NotImplementedError, pickle.PicklingError) as error:
        report(f"executable cache not saved: {error}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return compiled


@contextmanager
def _kernel_cache_lock(path: Path):
    with path.with_suffix(".lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _kernel_key(lowered: Any, family: tuple[str, str]) -> str:
    """Share binaries within a builder family, retaining compiler compatibility.

    XLA fingerprints each fused computation, constants and buffer arguments.
    Its kernel cache therefore permits differing graphs and shapes; the model
    executable cache above still requires its complete source/context/schema.
    """
    digest = hashlib.sha256()
    settings = (
        1, family,
        jax.__version__, jaxlib.__version__,
        tuple((device.platform, device.device_kind, device.id, device.client.platform_version)
              for device in jax.devices()),
        os.environ.get("XLA_FLAGS", ""), _context_value(jax.config.values),
        _context_value(lowered._lowering._compiler_options_kvs),
    )
    digest.update(repr(settings).encode())
    return digest.hexdigest()


def reusable_terminal_jit(
    function: Callable[..., Any], *, context: Any,
    cache_dir: Path | None = None,
    preserve_call: bool = False,
    kernel_family: tuple[str, str] | None = None,
) -> Callable[..., Any]:
    """Compile a host-submitted stage while preserving its enclosing JAX graph.

    Array values, including model parameters, are always call arguments. During
    tracing the original function is expanded: an enclosing scan keeps exactly
    its existing scientific call boundaries. Only terminal host submissions
    reuse a separately compiled stage, avoiding many small dispatches.
    Set preserve_call for an existing jit boundary: enclosing transformations
    must keep that same boundary rather than expanding the Python function.
    """
    jitted = jax.jit(function, inline=False)
    programs: dict[Any, Any] = {}

    @wraps(function)
    def call(*args):
        leaves, tree = jax.tree_util.tree_flatten(args)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            return jitted(*args) if preserve_call else function(*args)
        # This process-local key needs no serialized strings. Query each abstract
        # value once; sharding objects already provide value equality and hashing.
        schema = []
        for leaf in leaves:
            # Device arrays already carry these scalar properties. typeof()
            # also rebuilds abstract sharding, while we key actual sharding below.
            aval = leaf.aval if isinstance(leaf, jax.Array) else jax.typeof(leaf)
            schema.append((aval.shape, aval.dtype, aval.weak_type, getattr(leaf, "sharding", None)))
        signature = (tree, tuple(schema))
        if signature not in programs:
            compiled = compile_model_program(
                jitted, *args, context=context, cache_dir=cache_dir, kernel_family=kernel_family,
            )
            if len(programs) >= 4:
                programs.pop(next(iter(programs)))
            programs[signature] = compiled
        return programs[signature](*args)

    return call


def reusable_keyword_jit(
    function: Callable[..., Any], *, static_argnames: tuple[str, ...],
    cache_dir: Path | None = None,
) -> Callable[..., Any]:
    """Reuse a keyword-only kernel without rebuilding its JAX graph per process.

    Transformations still see the original jit and scientific function. Terminal
    calls load that same compiled function, with all evolving arrays supplied as
    inputs. Static keywords are included in both memory and disk cache keys.
    """
    jitted = jax.jit(function, static_argnames=static_argnames)
    programs: dict[Any, Any] = {}

    @wraps(function)
    def call(**kwargs):
        enabled = os.environ.get("WRF_PY_EXECUTABLE_CACHE", "0").lower() in {"1", "true", "yes", "on"}
        if jax.default_backend() == "cpu" or (not enabled and cache_dir is None) or any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(kwargs)
        ):
            return jitted(**kwargs)
        static = {name: kwargs[name] for name in static_argnames if name in kwargs}
        dynamic = {name: value for name, value in kwargs.items() if name not in static}
        context = {"program": function.__qualname__, "static_keywords": static}
        try:
            leaves, tree = jax.tree_util.tree_flatten(dynamic)
            schema = []
            for leaf in leaves:
                aval = jax.typeof(leaf)
                schema.append((aval.shape, aval.dtype, aval.weak_type, getattr(leaf, "sharding", None)))
            key = (tree, tuple(schema), _context_value(static))
        except (TypeError, ValueError, RecursionError):
            return jitted(**kwargs)
        if key not in programs:
            compiled = compile_model_program(
                jitted, context=context, call_kwargs=kwargs,
                static_argnames=static_argnames, cache_dir=cache_dir,
            )
            if len(programs) >= 8:
                programs.pop(next(iter(programs)))
            programs[key] = compiled
        return programs[key](**dynamic)

    call.lower = jitted.lower
    return call
