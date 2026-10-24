"""Compiler boundaries that preserve WRF semantics and JAX transforms."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import numpy as np
from jax._src import api_util, core, linear_util as lu
from jax._src.interpreters import ad, mlir, partial_eval as pe


_SEMANTIC_CALL = core.CallPrimitive("wrf_semantic_call")
_SEMANTIC_CALL.def_impl(core.call_impl)
# Retain scientific call boundaries, but allow unused workspace results and
# their private inputs to die just as they do in an ordinary JAX call.
pe.dce_rules[_SEMANTIC_CALL] = pe.dce_jaxpr_call_rule
# Checkpointing must split primal results from tangent computations inside
# the boundary, just as it does for an ordinary JAX call.
pe.partial_eval_jaxpr_custom_rules[_SEMANTIC_CALL] = pe.partial_eval_jaxpr_custom_rules[core.call_p]

# GPU dynamic-copy rewrites in jaxlib 0.9.2 lose the enclosing loop context
# inside non-inlineable calls. These passes do not occur on CPU. Use on the
# outer jit when a semantic call can contain scan/while; leaves remain intact.
SEMANTIC_CALL_COMPILER_OPTIONS = {
    "xla_disable_hlo_passes":
    "fusion-dynamic-memcpy-rewriter,dynamic-slice-fusion-rewriter"
}


def _contains_semantic_call(jaxpr: core.Jaxpr) -> bool:
    return any(
        equation.primitive is _SEMANTIC_CALL
        or any(_contains_semantic_call(nested) for nested in core.jaxprs_in_params(equation.params))
        for equation in jaxpr.eqns
    )


def _semantic_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: Any,
    name: str = "wrf_semantic_call",
    backend: str | None = None,
    call_jaxpr: core.ClosedJaxpr | core.Jaxpr,
    gpu_stream: str | None = None,
    fuse_gpu: bool = True,
) -> Any:
    """Lower a call that cannot be fused before numerical optimization."""

    if isinstance(call_jaxpr, core.Jaxpr):
        call_jaxpr = pe.close_jaxpr(call_jaxpr)
    # XLA lowers a non-inlineable GPU call as a command buffer, but cannot
    # nest one such buffer inside another. Keep leaf scientific kernels
    # non-inlineable and expand composition wrappers on both CPU and GPU.
    # Explicit input/output barriers retain the wrapper's array boundaries.
    is_composition = _contains_semantic_call(call_jaxpr.jaxpr)
    attributes = {"inlineable": mlir.ir.StringAttr.get("true" if is_composition else "false")}
    if gpu_stream is not None and fuse_gpu and not is_composition:
        # XLA skips GPU fusion inside ordinary non-inlineable call bodies.
        # Like JAX compute_on, annotate stream zero to permit internal fusion
        # without allowing optimization across this scientific call boundary.
        attributes["_xla_stream_annotation"] = mlir.ir.StringAttr.get(gpu_stream)
    if is_composition and args:
        args = tuple(mlir.hlo.OptimizationBarrierOp(args).results)
    outputs, tokens = mlir.call_lowering(
        name,
        call_jaxpr,
        backend,
        ctx.module_context,
        ctx.avals_in,
        ctx.avals_out,
        ctx.tokens_in,
        *args,
        dim_var_values=ctx.dim_var_values,
        const_lowering=ctx.const_lowering,
        attributes=attributes,
    )
    ctx.set_tokens_out(tokens)
    if is_composition and outputs:
        return tuple(mlir.hlo.OptimizationBarrierOp(outputs).results)
    return outputs


mlir.register_lowering(_SEMANTIC_CALL, _semantic_call_lowering)
mlir.register_lowering(
    _SEMANTIC_CALL, partial(_semantic_call_lowering, gpu_stream="0"), platform="cuda",
)
def _semantic_call_transpose(cotangents, *arguments, **params):
    # Keep inactive floating outputs in the transpose signature. Symbolic
    # zeros otherwise change pruning and rematerialization of branch inputs.
    cotangents = tuple(
        ad.instantiate_zeros(value)
        if type(value) is ad.Zero and np.issubdtype(value.aval.dtype, np.inexact)
        else value
        for value in cotangents
    )
    return ad.call_transpose_fancy(_SEMANTIC_CALL, cotangents, *arguments, **params)


ad.fancy_transposes[_SEMANTIC_CALL] = _semantic_call_transpose


def _reverse_call_options(params, *_):
    # Large transpose fusions can spend hours simplifying GPU index maps.
    # This changes only derivative scheduling, not forward scientific kernels.
    return dict(params, fuse_gpu=False)


ad.call_transpose_param_updaters[_SEMANTIC_CALL] = _reverse_call_options


def semantic_call(function: Any, *args: Any, fuse_gpu: bool = True) -> Any:
    """Call a pure JAX function through a device-only semantic boundary.

    Leaf scientific calls remain distinct during numerical optimization.
    Nested composition wrappers expand with explicit array barriers, avoiding
    unsupported nested GPU command buffers. CUDA stream annotations permit
    optimization inside leaf calls; both devices use the same scientific
    functions. Standard JAX transforms never transfer model state through the
    host. ``fuse_gpu`` controls internal GPU fusion, not scientific behavior;
    some recurrent column kernels run faster without stream-annotated fusion.
    """

    debug_info = api_util.debug_info("wrf_semantic_call", function, args, {})
    wrapped = lu.wrap_init(function, debug_info=debug_info)
    flat_args, input_tree = jax.tree_util.tree_flatten(args)
    flat_function, output_tree = api_util.flatten_fun_nokwargs(wrapped, input_tree)
    flat_outputs = _SEMANTIC_CALL.bind(
        *flat_args, subfuns=(flat_function,), fuse_gpu=fuse_gpu,
    )
    return jax.tree_util.tree_unflatten(output_tree(), flat_outputs)


@jax.custom_jvp
def optimization_barrier(value: Any) -> Any:
    """Prevent cross-boundary fusion while preserving identity derivatives."""

    return jax.lax.optimization_barrier(value)


@optimization_barrier.defjvp
def _optimization_barrier_jvp(
    primals: tuple[Any],
    tangents: tuple[Any],
) -> tuple[Any, Any]:
    return (
        jax.lax.optimization_barrier(primals[0]),
        jax.lax.optimization_barrier(tangents[0]),
    )


__all__ = ["SEMANTIC_CALL_COMPILER_OPTIONS", "optimization_barrier", "semantic_call"]
