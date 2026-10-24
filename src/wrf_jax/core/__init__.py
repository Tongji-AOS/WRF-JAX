"""Core exports without importing the model when a kernel imports a utility."""

from importlib import import_module

_MODULE_EXPORTS = {
    "conversion": (
        "apply_time_state_to_grid",
        "fields_to_model_state",
        "model_state_to_fields",
        "state_array_names",
    ),
    "direct_step": ("DirectStepPlan", "build_direct_step_plan", "run_direct_step"),
    "partition": (
        "FieldPartition",
        "array_nbytes",
        "deduplicate_input_buffers",
        "is_dynamic_array_value",
        "partition_fields_for_compiled_step",
        "split_static_dynamic_fields",
    ),
    "step_em": (
        "StepEmCoreContext",
        "StepEmCoreResult",
        "build_step_em_core_context",
        "initial_step_em_state",
        "step_em_core_once",
    ),
    "state": ("ModelState", "TimeState"),
    "resident_state": ("ResidentBufferGroup", "ResidentState", "ResidentStateLayout"),
    "halo_exchange": ("crop_halo", "exchange_halo"),
    "tiled_stage": ("TiledStageProgram", "TiledStateLayout", "build_tiled_stage"),
}
_EXPORT_MODULES = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}
__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name):
    # Eager stepper exports cause a cycle when dycore imports column_scan.
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
