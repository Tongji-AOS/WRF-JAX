"""Translation support for WRF ``module_cumulus_driver``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.optimization import semantic_call
from wrf_jax.physics.module_cu_tiedtke import cu_tiedtke


CU_PHYSICS_NONE = 0
TIEDTKE_SCHEME = 6
NEW_TIEDTKE_SCHEME = 16
_JIT_TIEDTKE = os.environ.get("WRF_PY_JIT_TIEDTKE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


_TIEDTKE_CHUNK_COLUMNS = _env_int("WRF_PY_TIEDTKE_CHUNK_COLUMNS", 0)
_TIEDTKE_CHUNK_WORKERS = max(1, _env_int("WRF_PY_TIEDTKE_CHUNK_WORKERS", 4))
_TIEDTKE_BLOCK_CHUNKS = os.environ.get("WRF_PY_TIEDTKE_BLOCK_CHUNKS", "1").strip().lower() not in {"", "0", "false", "no", "off"}
_TIEDTKE_DRIVER_PROFILE_PATH = os.environ.get("WRF_PY_PROFILE_TIEDTKE_DRIVER", "").strip()


def _profile_tiedtke_driver(message: str) -> None:
    if not _TIEDTKE_DRIVER_PROFILE_PATH:
        return
    with open(_TIEDTKE_DRIVER_PROFILE_PATH, "a", encoding="ascii") as handle:
        handle.write(message + "\n")


@dataclass(frozen=True)
class CumulusDriverResult:
    """Outputs carried by the translated cumulus driver."""

    pratec: Any | None = None
    cudtacttime: float | None = None
    carried_fields: Mapping[str, Any] = field(default_factory=dict)
    ran_parameterization: bool = False
    external_calls: tuple[str, ...] = ()


def _cumulus_should_run(
    *,
    itimestep: int,
    stepcu: int,
    cudt: float | None,
    curr_secs: float | None,
    adapt_step_flag: bool | None,
    cudtacttime: float | None,
) -> tuple[bool, float | None]:
    """Mirror WRF's cumulus scheduling gate."""

    adapt_step_flag_pass = False if adapt_step_flag is None else bool(adapt_step_flag)
    if not adapt_step_flag_pass and (
        (cudt is not None and float(cudt) == 0.0)
        or int(stepcu) == 1
    ):
        return True, None if cudtacttime is None else float(cudtacttime)

    if cudt is None:
        cudt_pass = -1.0
        cudtacttime_pass = -1.0
    else:
        cudt_pass = float(cudt)
        cudtacttime_pass = -1.0 if cudtacttime is None else float(cudtacttime)

    curr_secs_pass = -1.0 if curr_secs is None else float(curr_secs)
    if adapt_step_flag_pass:
        if cudtacttime_pass == 0.0:
            cudtacttime_pass = curr_secs_pass + cudt_pass * 60.0
    else:
        cudtacttime_pass = None if cudtacttime is None else float(cudtacttime)

    if int(itimestep) == 1:
        return True, cudtacttime_pass
    if cudt_pass == 0.0 or int(stepcu) == 1:
        return True, cudtacttime_pass
    if not adapt_step_flag_pass and int(itimestep) % int(stepcu) == 0:
        return True, cudtacttime_pass
    if adapt_step_flag_pass and curr_secs_pass >= float(cudtacttime_pass):
        return True, curr_secs_pass + cudt_pass * 60.0
    return False, cudtacttime_pass


def cumulus_driver(
    *,
    cu_physics: int,
    itimestep: int = 0,
    stepcu: int = 1,
    cudt: float | None = None,
    curr_secs: float | None = None,
    adapt_step_flag: bool | None = None,
    cudtacttime: float | None = None,
    pratec: Any | None = None,
    **fields: Any,
) -> CumulusDriverResult:
    """Run the translated WRF cumulus driver for supported configurations.

    ``cu_physics = 0`` returns without modifying output.  ``cu_physics = 6``
    uses the translated Tiedtke implementation.  Other active schemes are
    stopped instead of approximated.
    """

    if int(cu_physics) == CU_PHYSICS_NONE:
        return CumulusDriverResult(
            pratec=pratec,
            cudtacttime=cudtacttime,
            carried_fields=dict(fields),
            ran_parameterization=False,
        )

    run_parameterization, cudtacttime_out = _cumulus_should_run(
        itimestep=itimestep,
        stepcu=stepcu,
        cudt=cudt,
        curr_secs=curr_secs,
        adapt_step_flag=adapt_step_flag,
        cudtacttime=cudtacttime,
    )
    if not run_parameterization:
        return CumulusDriverResult(
            pratec=pratec,
            cudtacttime=cudtacttime,
            carried_fields=dict(fields),
            ran_parameterization=False,
        )

    if int(cu_physics) == TIEDTKE_SCHEME:
        carried = _run_tiedtke_scheme(
            itimestep=itimestep,
            stepcu=stepcu,
            pratec=pratec,
            fields=fields,
        )
        return CumulusDriverResult(
            pratec=carried["pratec"],
            cudtacttime=cudtacttime_out,
            carried_fields=carried,
            ran_parameterization=True,
            external_calls=("cu_tiedtke",),
        )

    if int(cu_physics) == NEW_TIEDTKE_SCHEME:
        raise NotImplementedError("cu_physics=16 requires module_cu_ntiedtke; CONUS maps to cu_physics=6 in this case")

    raise NotImplementedError(
        "cumulus_driver has no translated active branch for "
        f"cu_physics={int(cu_physics)} reached cudtacttime={cudtacttime_out!r}"
    )


def _array3(fields: Mapping[str, Any], name: str) -> Any:
    if name not in fields or fields[name] is None:
        raise KeyError(f"cumulus Tiedtke requires {name}")
    return jnp.asarray(fields[name], dtype=jnp.float32)


def _array2(fields: Mapping[str, Any], name: str) -> Any:
    if name not in fields or fields[name] is None:
        raise KeyError(f"cumulus Tiedtke requires {name}")
    return jnp.asarray(fields[name], dtype=jnp.float32)


def _optional3(fields: Mapping[str, Any], name: str, template: Any) -> Any:
    value = fields.get(name)
    if value is None:
        return jnp.zeros_like(template, dtype=jnp.float32)
    return jnp.asarray(value, dtype=jnp.float32)


def _optional2(value: Any, template: Any) -> Any:
    if value is None:
        return jnp.zeros_like(template, dtype=jnp.float32)
    return jnp.asarray(value, dtype=jnp.float32)


def _logical2(fields: Mapping[str, Any], name: str, template: Any) -> Any:
    value = fields.get(name)
    if value is None:
        return jnp.ones(template.shape, dtype=jnp.int32)
    return jnp.asarray(value, dtype=jnp.int32)


def _cu_tiedtke_tuple_impl(
    *,
    dt: Any,
    itimestep: Any,
    stepcu: int,
    raincv: Any,
    pratec: Any,
    qfx: Any,
    znu: Any,
    u3d: Any,
    v3d: Any,
    w: Any,
    t3d: Any,
    qv3d: Any,
    qc3d: Any,
    qi3d: Any,
    pi3d: Any,
    rho3d: Any,
    qvften: Any,
    qvpblten: Any,
    dz8w: Any,
    pcps: Any,
    p8w: Any,
    xland: Any,
    cu_act_flag: Any,
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
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqicuten: Any,
    rucuten: Any,
    rvcuten: Any,
    f_qv: bool,
    f_qc: bool,
    f_qr: bool,
    f_qi: bool,
    f_qs: bool,
    rounding_token: Any | None = None,
) -> tuple[Any, ...]:
    result = cu_tiedtke(
        dt,
        itimestep,
        stepcu,
        raincv,
        pratec,
        qfx,
        znu,
        u3d,
        v3d,
        w,
        t3d,
        qv3d,
        qc3d,
        qi3d,
        pi3d,
        rho3d,
        qvften,
        qvpblten,
        dz8w,
        pcps,
        p8w,
        xland,
        cu_act_flag,
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
        rthcuten,
        rqvcuten,
        rqccuten,
        rqicuten,
        rucuten,
        rvcuten,
        f_qv,
        f_qc,
        f_qr,
        f_qi,
        f_qs,
        rounding_token=rounding_token,
    )
    return (
        result.raincv,
        result.pratec,
        result.cu_act_flag,
        result.rthcuten,
        result.rqvcuten,
        result.rqccuten,
        result.rqicuten,
        result.rucuten,
        result.rvcuten,
    )


_TIEDTKE_STATIC_ARGNAMES = (
    "stepcu",
    "ids",
    "ide",
    "jds",
    "jde",
    "kds",
    "kde",
    "ims",
    "ime",
    "jms",
    "jme",
    "kms",
    "kme",
    "its",
    "ite",
    "jts",
    "jte",
    "kts",
    "kte",
    "f_qv",
    "f_qc",
    "f_qr",
    "f_qi",
    "f_qs",
)


_cu_tiedtke_tuple_jit = jax.jit(
    _cu_tiedtke_tuple_impl,
    static_argnames=_TIEDTKE_STATIC_ARGNAMES,
)


def _cu_tiedtke_tuple(**kwargs: Any) -> tuple[Any, ...]:
    if _JIT_TIEDTKE:
        return _cu_tiedtke_tuple_jit(**kwargs)
    return _cu_tiedtke_tuple_impl(**kwargs)


def _slice_tiedtke_arg(value: Any, *, j_start: int, j_stop: int) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None:
        return value
    if len(shape) == 3:
        return value[:, :, j_start:j_stop]
    if len(shape) == 2:
        return value[:, j_start:j_stop]
    return value


def _concat_tiedtke_chunks(chunks: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    if not chunks:
        raise ValueError("cannot concatenate empty Tiedtke chunks")
    outputs: list[Any] = []
    for values in zip(*chunks, strict=True):
        arrays = [jnp.asarray(value) for value in values]
        ndim = arrays[0].ndim
        axis = 2 if ndim == 3 else 1 if ndim == 2 else 0
        outputs.append(jnp.concatenate(arrays, axis=axis))
    return tuple(outputs)


_TIEDTKE_OUTPUT_ARG_NAMES = (
    "raincv",
    "pratec",
    "cu_act_flag",
    "rthcuten",
    "rqvcuten",
    "rqccuten",
    "rqicuten",
    "rucuten",
    "rvcuten",
)


def _restore_tiedtke_memory_rows(
    active_values: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    j_start: int,
    j_stop: int,
) -> tuple[Any, ...]:
    """Restore untouched memory/halo rows around chunked active rows."""

    outputs: list[Any] = []
    for name, active_value in zip(_TIEDTKE_OUTPUT_ARG_NAMES, active_values, strict=True):
        source = jnp.asarray(kwargs[name])
        active = jnp.asarray(active_value)
        axis = 2 if source.ndim == 3 else 1 if source.ndim == 2 else 0
        if source.ndim not in {2, 3}:
            raise ValueError(f"unsupported Tiedtke output rank for {name}: {source.ndim}")
        expected_rows = j_stop - j_start
        if active.shape[axis] != expected_rows:
            raise ValueError(
                f"chunked Tiedtke output {name} has {active.shape[axis]} active rows; "
                f"expected {expected_rows}"
            )
        if source.shape[axis] == expected_rows and j_start == 0:
            outputs.append(active)
            continue
        if source.ndim == 3:
            outputs.append(
                jnp.concatenate(
                    (source[:, :, :j_start], active, source[:, :, j_stop:]),
                    axis=2,
                )
            )
        else:
            outputs.append(
                jnp.concatenate(
                    (source[:, :j_start], active, source[:, j_stop:]),
                    axis=1,
                )
            )
    return tuple(outputs)


def _block_tiedtke_tuple(values: tuple[Any, ...]) -> None:
    for value in values:
        block = getattr(value, "block_until_ready", None)
        if block is not None:
            block()


def _contains_jax_tracer(values: Mapping[str, Any]) -> bool:
    """Return whether a Tiedtke call is staged by an outer transform."""

    return any(
        isinstance(value, jax.core.Tracer)
        for value in jax.tree_util.tree_leaves(values)
    )


def _semantic_tiedtke_tuple(kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keep one production row-chunk signature as a device subroutine."""

    static_names = frozenset(_TIEDTKE_STATIC_ARGNAMES)
    dynamic_names = tuple(
        name
        for name, value in kwargs.items()
        if name not in static_names and value is not None
    )
    static_kwargs = {
        name: value for name, value in kwargs.items() if name not in dynamic_names
    }

    def call(*dynamic_values: Any) -> tuple[Any, ...]:
        call_kwargs = dict(static_kwargs)
        call_kwargs.update(zip(dynamic_names, dynamic_values, strict=True))
        return _cu_tiedtke_tuple(**call_kwargs)

    return semantic_call(call, *(kwargs[name] for name in dynamic_names))


def _run_tiedtke_tuple_chunked(kwargs: Mapping[str, Any], *, max_columns: int) -> tuple[Any, ...]:
    traced = _contains_jax_tracer(kwargs)
    its = int(kwargs["its"])
    ite = int(kwargs["ite"])
    jts = int(kwargs["jts"])
    jte = int(kwargs["jte"])
    jms = int(kwargs["jms"])
    im = ite - its + 1
    jm = jte - jts + 1
    total_columns = im * jm
    if int(max_columns) <= 0:
        target_chunks = max(1, min(_TIEDTKE_CHUNK_WORKERS, jm))
        rows_per_chunk = max(1, (jm + target_chunks - 1) // target_chunks)
        max_columns = rows_per_chunk * im
    if total_columns <= int(max_columns):
        _profile_tiedtke_driver(f"single_chunk_start columns={total_columns}")
        start = perf_counter()
        result = (
            _semantic_tiedtke_tuple(kwargs)
            if traced
            else _cu_tiedtke_tuple(**dict(kwargs))
        )
        if not traced:
            _block_tiedtke_tuple(result)
        _profile_tiedtke_driver(f"single_chunk_done seconds={perf_counter() - start:.6f} columns={total_columns}")
        return result

    rows_per_chunk = max(1, int(max_columns) // max(1, im))
    j0 = jts - jms
    chunk_ranges = [(local_start, min(j0 + jm, local_start + rows_per_chunk)) for local_start in range(j0, j0 + jm, rows_per_chunk)]

    def _run_one_chunk(chunk_range: tuple[int, int]) -> tuple[Any, ...]:
        local_start, local_stop = chunk_range
        chunk_ny = local_stop - local_start
        columns = im * chunk_ny
        _profile_tiedtke_driver(f"chunk_start j_start={local_start} j_stop={local_stop} rows={chunk_ny} columns={columns}")
        start = perf_counter()
        chunk_kwargs = {
            name: _slice_tiedtke_arg(value, j_start=local_start, j_stop=local_stop)
            for name, value in kwargs.items()
        }
        chunk_kwargs.update(
            {
                "jms": 1,
                "jme": chunk_ny,
                "jts": 1,
                "jte": chunk_ny,
            }
        )
        result = (
            _semantic_tiedtke_tuple(chunk_kwargs)
            if traced
            else _cu_tiedtke_tuple(**chunk_kwargs)
        )
        if _TIEDTKE_BLOCK_CHUNKS and not traced:
            _block_tiedtke_tuple(result)
        _profile_tiedtke_driver(
            f"chunk_done seconds={perf_counter() - start:.6f} j_start={local_start} j_stop={local_stop} rows={chunk_ny} columns={columns}"
        )
        return result

    # Row dimensions are part of the scientific compilation boundary: retain
    # the same chunks on both devices. Submit GPU batches from one host thread
    # (device execution remains asynchronous). Concurrent calls of this kernel
    # on the A800 changed identical-input results intermittently; sequential
    # submission is also faster and avoids recreating a host pool each event.
    if traced or jax.default_backend() != "cpu" or _TIEDTKE_CHUNK_WORKERS <= 1 or len(chunk_ranges) <= 2:
        chunks = [_run_one_chunk(chunk_range) for chunk_range in chunk_ranges]
        concat_start = perf_counter()
        result = _restore_tiedtke_memory_rows(
            _concat_tiedtke_chunks(chunks),
            kwargs,
            j_start=j0,
            j_stop=j0 + jm,
        )
        _profile_tiedtke_driver(f"concat_done seconds={perf_counter() - concat_start:.6f} chunks={len(chunks)}")
        return result

    if _JIT_TIEDTKE:
        first = _run_one_chunk(chunk_ranges[0])
        with ThreadPoolExecutor(max_workers=min(_TIEDTKE_CHUNK_WORKERS, len(chunk_ranges) - 1)) as pool:
            rest = list(pool.map(_run_one_chunk, chunk_ranges[1:]))
        chunks = [first, *rest]
    else:
        with ThreadPoolExecutor(max_workers=min(_TIEDTKE_CHUNK_WORKERS, len(chunk_ranges))) as pool:
            chunks = list(pool.map(_run_one_chunk, chunk_ranges))
    concat_start = perf_counter()
    result = _restore_tiedtke_memory_rows(
        _concat_tiedtke_chunks(chunks),
        kwargs,
        j_start=j0,
        j_stop=j0 + jm,
    )
    _profile_tiedtke_driver(f"concat_done seconds={perf_counter() - concat_start:.6f} chunks={len(chunks)}")
    return result


def _run_tiedtke_scheme(
    *,
    itimestep: Any,
    stepcu: int,
    pratec: Any | None,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    t3d = _array3(fields, "t3d")
    u3d = _array3(fields, "u3d")
    v3d = _array3(fields, "v3d")
    w = _array3(fields, "w")
    qv3d = _array3(fields, "qv3d")
    qc3d = _array3(fields, "qc3d")
    qi3d = _array3(fields, "qi3d")
    pi3d = _array3(fields, "pi3d")
    rho3d = _array3(fields, "rho3d")
    qvften = _array3(fields, "qvften")
    qvpblten = _array3(fields, "qvpblten")
    dz8w = _array3(fields, "dz8w")
    pcps = _array3(fields, "pcps")
    p8w = _array3(fields, "p8w")
    qfx = _array2(fields, "qfx")
    xland = _array2(fields, "xland")
    raincv = _optional2(fields.get("raincv"), qfx)
    pratec_arr = _optional2(pratec, qfx)
    cu_act_flag = _logical2(fields, "cu_act_flag", qfx)

    ids = int(fields["ids"])
    ide = int(fields["ide"])
    jds = int(fields["jds"])
    jde = int(fields["jde"])
    kds = int(fields["kds"])
    kde = int(fields["kde"])
    ims = int(fields["ims"])
    ime = int(fields["ime"])
    jms = int(fields["jms"])
    jme = int(fields["jme"])
    kms = int(fields["kms"])
    kme = int(fields["kme"])
    its = int(fields["its"])
    ite = int(fields["ite"])
    jts = int(fields["jts"])
    jte = int(fields["jte"])
    kts = int(fields["kts"])
    kte = int(fields["kte"])

    rthcuten = _optional3(fields, "rthcuten", t3d)
    rqvcuten = _optional3(fields, "rqvcuten", t3d)
    rqccuten = _optional3(fields, "rqccuten", t3d)
    rqicuten = _optional3(fields, "rqicuten", t3d)
    rucuten = _optional3(fields, "rucuten", t3d)
    rvcuten = _optional3(fields, "rvcuten", t3d)

    # In a resident spatial executor these values are produced in the same
    # traced graph as the Tiedtke wrapper.  The reference host-chunk path
    # materializes them before entering the JIT kernel.  Recreate that
    # semantic boundary only for the opt-in diagnostic experiment.
    # The dynamic bitcast/XOR token is useful in several dycore kernels, but
    # it changes XLA's lowering of the complete Tiedtke graph enough to flip
    # marginal convective columns.  Tiedtke's ``_wrf_real`` already uses
    # ``optimization_barrier`` when no token is supplied; that path preserves
    # the WRF statement boundaries while retaining the full compiled kernel.
    rounding_token = None if _JIT_TIEDTKE else fields.get("rounding_token")

    (
        raincv_out,
        pratec_out,
        cu_act_flag_out,
        rthcuten_out,
        rqvcuten_out,
        rqccuten_out,
        rqicuten_out,
        rucuten_out,
        rvcuten_out,
    ) = _run_tiedtke_tuple_chunked(
        {
            "dt": jnp.asarray(fields["dt"], dtype=jnp.float32),
            "itimestep": jnp.asarray(itimestep, dtype=jnp.int32),
            "stepcu": int(stepcu),
            "raincv": raincv,
            "pratec": pratec_arr,
            "qfx": qfx,
            "znu": jnp.asarray(fields["znu"], dtype=jnp.float32),
            "u3d": u3d,
            "v3d": v3d,
            "w": w,
            "t3d": t3d,
            "qv3d": qv3d,
            "qc3d": qc3d,
            "qi3d": qi3d,
            "pi3d": pi3d,
            "rho3d": rho3d,
            "qvften": qvften,
            "qvpblten": qvpblten,
            "dz8w": dz8w,
            "pcps": pcps,
            "p8w": p8w,
            "xland": xland,
            "cu_act_flag": cu_act_flag,
            "ids": ids,
            "ide": ide,
            "jds": jds,
            "jde": jde,
            "kds": kds,
            "kde": kde,
            "ims": ims,
            "ime": ime,
            "jms": jms,
            "jme": jme,
            "kms": kms,
            "kme": kme,
            "its": its,
            "ite": ite,
            "jts": jts,
            "jte": jte,
            "kts": kts,
            "kte": kte,
            "rthcuten": rthcuten,
            "rqvcuten": rqvcuten,
            "rqccuten": rqccuten,
            "rqicuten": rqicuten,
            "rucuten": rucuten,
            "rvcuten": rvcuten,
            "rounding_token": rounding_token,
            "f_qv": bool(fields.get("f_qv", True)),
            "f_qc": bool(fields.get("f_qc", True)),
            "f_qr": bool(fields.get("f_qr", True)),
            "f_qi": bool(fields.get("f_qi", True)),
            "f_qs": bool(fields.get("f_qs", True)),
        },
        max_columns=(
            max(1, (ite - its + 1) * (jte - jts + 1))
            if bool(fields.get("_tiedtke_resident_chunk", False))
            else _TIEDTKE_CHUNK_COLUMNS
        ),
    )
    return {
        "raincv": raincv_out,
        "pratec": pratec_out,
        "cu_act_flag": cu_act_flag_out,
        "rthcuten": rthcuten_out,
        "rqvcuten": rqvcuten_out,
        "rqccuten": rqccuten_out,
        "rqicuten": rqicuten_out,
        "rucuten": rucuten_out,
        "rvcuten": rvcuten_out,
    }


__all__ = ["CU_PHYSICS_NONE", "TIEDTKE_SCHEME", "NEW_TIEDTKE_SCHEME", "CumulusDriverResult", "cumulus_driver"]
