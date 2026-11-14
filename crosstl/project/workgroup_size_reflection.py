"""The reflected workgroup size of compute, mesh and task entry points whose pipeline or host may set it.

SPIR-V sizes a workgroup entry point with its ``LocalSize`` execution mode, with a ``LocalSizeId`` execution mode
(SPIR-V 1.2) whose operands may be specialization constants, or, for every entry point of the module and in
preference to both, with a constant decorated ``BuiltIn WorkgroupSize`` (SPIR-V 1.6 section 3.21). Vulkan accepts
``LocalSizeId`` only when VK_KHR_maintenance4, core in Vulkan 1.3, is enabled. GLSL names the specialization id of a
dimension with ``local_size_{x,y,z}_id`` (GLSL 4.60 section 4.4.1.4).

A reflected ``executionConfig`` keeps the size a pipeline gets without specialization under its usual keys and adds
``local_size_id``, the specialization id of each dimension, ``None`` for a fixed one.

A kernel whose threadgroup size the source's host chooses at dispatch time has no size of its own. CrossTL gives it
the specialization constants named in :data:`HOST_WORKGROUP_SIZE_CONSTANT_NAMES`, with a placeholder default of one,
and its GLSL source reads the size from the :data:`HOST_WORKGROUP_SIZE_MACROS` when no specialization step runs.
HLSL has no specialization constants, so its ``[numthreads]`` always names the macros, which the host defines when
it compiles the shader. Such an entry reports ``host_workgroup_size`` with those ids and macros and no size, so that no consumer dispatches
the placeholder: :func:`crosstl.translator.host_workgroup_size.require_static_workgroup_size` refuses it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_CONSTANT_NAMES,
    HOST_WORKGROUP_SIZE_MACROS,
)
from crosstl.translator.host_workgroup_size import HOST_WORKGROUP_SIZE_KEY

__all__ = [
    "LOCAL_SIZE_ID_EXTENSION",
    "WorkgroupSizeReflectionError",
    "glsl_execution_config",
    "hlsl_host_chosen_execution_config",
    "spirv_entry_execution_config",
    "spirv_workgroup_size_builtin",
]

LOCAL_SIZE_ID_EXTENSION = "VK_KHR_maintenance4"
HOST_CHOSEN_REASON = "has a workgroup size that the pipeline or host sets"

_AXES = ("x", "y", "z")
_WORKGROUP_MODELS = frozenset({"GLCompute", "MeshEXT", "TaskEXT", "MeshNV", "TaskNV"})


class WorkgroupSizeReflectionError(ValueError):
    """A workgroup size the reflector cannot resolve to constants and specialization constants."""


def _host_chosen_config(entry: str, ids: Sequence[Any], macros: Sequence[str] = ()):
    record: dict[str, Any] = {
        "entry": entry,
        "reason": HOST_CHOSEN_REASON,
        "specializationIds": list(ids),
    }
    if macros:
        record["macros"] = list(macros)
    return {"local_size_id": list(ids), HOST_WORKGROUP_SIZE_KEY: record}


def spirv_workgroup_size_builtin(decorations: Mapping[str, Mapping[str, Any]]):
    """The result id of the module's constant decorated ``BuiltIn WorkgroupSize``, if it declares one."""
    for result_id, decoration in decorations.items():
        if decoration.get("BuiltIn") == "WorkgroupSize":
            return result_id
    return None


def _resolve_component(component, constants, decorations):
    constant = constants.get(component)
    if constant is None or isinstance(constant.get("value"), bool):
        raise WorkgroupSizeReflectionError(
            f"workgroup size component {component} is not an integer constant"
        )
    value = constant.get("value")
    if not isinstance(value, int):
        raise WorkgroupSizeReflectionError(
            f"workgroup size component {component} is not an integer constant"
        )
    spec_id = None
    if constant.get("kind") == "specialization-constant":
        spec_id = decorations.get(component, {}).get("SpecId")
        if not isinstance(spec_id, int):
            raise WorkgroupSizeReflectionError(
                f"specialization constant {component} of the workgroup size has no SpecId"
            )
    return value, spec_id


def spirv_entry_execution_config(
    entry_point: Mapping[str, Any],
    *,
    mode: Mapping[str, Any] | None,
    builtin: str | None,
    composites: Mapping[str, Sequence[str]],
    constants: Mapping[str, Mapping[str, Any]],
    decorations: Mapping[str, Mapping[str, Any]],
    names: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``executionConfig`` of one SPIR-V entry point and the metadata its size adds.

    Raises :class:`WorkgroupSizeReflectionError` for a size that is not made of constants.
    """
    mode = mode or {}
    model = entry_point.get("metadata", {}).get("executionModel")
    if model not in _WORKGROUP_MODELS:
        if "local_size" in mode:
            return {"local_size": list(mode["local_size"])}, {}
        return {}, {}
    if builtin is not None:
        components = composites.get(builtin)
        if components is None or len(components) != 3:
            raise WorkgroupSizeReflectionError(
                f"the WorkgroupSize built-in {builtin} is not a three-component constant composite"
            )
        source, metadata = "WorkgroupSize", {}
    elif "local_size_id" in mode:
        components = mode["local_size_id"]
        source = "LocalSizeId"
        metadata = {"requiredExtensions": [LOCAL_SIZE_ID_EXTENSION]}
    elif "local_size" in mode:
        return {"local_size": list(mode["local_size"])}, {}
    else:
        return {}, {}
    resolved = [
        _resolve_component(component, constants, decorations)
        for component in components
    ]
    sizes = [value for value, _ in resolved]
    ids = [spec_id for _, spec_id in resolved]
    metadata = {"workgroupSizeSource": source, **metadata}
    if all(spec_id is None for spec_id in ids):
        return {"local_size": sizes}, metadata
    component_names = tuple(names.get(component) for component in components)
    if component_names == HOST_WORKGROUP_SIZE_CONSTANT_NAMES:
        return _host_chosen_config(str(entry_point.get("name")), ids), metadata
    return {"local_size": sizes, "local_size_id": ids}, metadata


def _layout_values(layout: Mapping[str, Any], suffix: str) -> list[Any]:
    return [layout.get(f"local_size_{axis}{suffix}") for axis in _AXES]


def hlsl_host_chosen_execution_config(
    operands: Sequence[Any], *, entry: str
) -> dict[str, Any] | None:
    """The ``executionConfig`` of an HLSL entry whose ``[numthreads]`` names the host's size macros, else ``None``.

    The entry has neither a size nor specialization ids, only the :data:`HOST_WORKGROUP_SIZE_MACROS` the host
    defines when it compiles the shader.
    """
    if (
        tuple(str(operand).strip() for operand in operands)
        != HOST_WORKGROUP_SIZE_MACROS
    ):
        return None
    record = {
        "entry": entry,
        "reason": HOST_CHOSEN_REASON,
        "macros": list(HOST_WORKGROUP_SIZE_MACROS),
    }
    return {HOST_WORKGROUP_SIZE_KEY: record}


def glsl_execution_config(
    layouts: Sequence[Mapping[str, Any]],
    *,
    entry: str,
) -> dict[str, Any]:
    """The ``executionConfig`` of a GLSL entry from each parsed ``layout(local_size_...) in;`` of its source.

    GLSL lets a source split the input layout over several declarations, and CrossTL's output for a size the host
    chooses declares it once per preprocessor branch: a specialization layout for SPIR-V compilation and a layout of
    :data:`HOST_WORKGROUP_SIZE_MACROS` otherwise. The first declaration of each qualifier wins.
    """
    if not layouts:
        return {}
    merged: dict[str, Any] = {}
    host_macros = False
    for layout in layouts:
        if tuple(_layout_values(layout, "")) == HOST_WORKGROUP_SIZE_MACROS:
            host_macros = True
            layout = {
                key: value
                for key, value in layout.items()
                if not key.startswith("local_size_") or key.endswith("_id")
            }
        for key, value in layout.items():
            merged.setdefault(key, value)
    ids = _layout_values(merged, "_id")
    if host_macros:
        return _host_chosen_config(entry, ids, HOST_WORKGROUP_SIZE_MACROS)
    config = {
        f"local_size_{axis}": merged.get(f"local_size_{axis}", 1) for axis in _AXES
    }
    if any(spec_id is not None for spec_id in ids):
        config["local_size_id"] = ids
    return config
