"""The workgroup size of each compute, mesh and task entry point of a SPIR-V module.

A constant decorated ``BuiltIn WorkgroupSize`` sizes every entry point of the module and takes precedence over
their ``LocalSize`` and ``LocalSizeId`` execution modes (SPIR-V 1.6 section 3.21, BuiltIn WorkgroupSize). The
generator declares one only when all workgroup entries of the module have the same size. In a module whose entries
have different sizes, each entry keeps its own execution mode, and ``gl_WorkGroupSize`` read in an entry or in a
function of its stage is an undecorated composite of that entry's size. Code shared by entries of different sizes
has no single size to read, so the generator refuses it. An entry whose size the pipeline specializes takes it
through ``LocalSizeId`` (SPIR-V 1.2), which Vulkan accepts when VK_KHR_maintenance4, core in Vulkan 1.3, is
enabled.

A kernel whose threadgroup size the source's host chooses at dispatch time (:mod:`crosstl.translator.
host_workgroup_size`) becomes such an entry. The host sets the size when it creates the pipeline, through the
three specialization constants that :func:`host_workgroup_size_ids` allocates: the smallest ids that no other
specialization constant or workgroup dimension of the program uses. Every host-sized entry of the module shares
them, because each Vulkan pipeline specializes its entry independently. Their default of one is the placeholder the
encoding requires; a pipeline created without them runs one-invocation workgroups.
"""

from __future__ import annotations

import re

from ..host_workgroup_size import HOST_WORKGROUP_SIZE_KEY, has_explicit_workgroup_size
from .specialized_workgroup_size import (
    SPECIALIZED_WORKGROUP_SIZE_KEYS,
    has_specialized_workgroup_size,
    workgroup_size_specialization_ids,
)

__all__ = [
    "effective_workgroup_config",
    "emit_entry_local_size_id",
    "entry_workgroup_size_value",
    "host_workgroup_size_ids",
    "plan_entry_workgroup_sizes",
]

_WORKGROUP_MODELS = frozenset({"GLCompute", "MeshEXT", "TaskEXT"})
_SPEC_ID = re.compile(r"\bSpecId (\d+)\b")
# No entry point is being generated: a function shared by the module's entry points.
_SHARED = object()


def _raw_config(generator, metadata) -> dict:
    if metadata is None:
        return generator.module_compute_entry_config or {}
    return getattr(metadata, "execution_config", None) or {}


def _is_host_sized(config) -> bool:
    return bool(config.get(HOST_WORKGROUP_SIZE_KEY)) and not (
        has_explicit_workgroup_size(config) or has_specialized_workgroup_size(config)
    )


def effective_workgroup_config(generator, config):
    """``config``, with the specialization ids of the host's size when the source's host chooses it."""
    config = config or {}
    if not _is_host_sized(config):
        return config
    ids = host_workgroup_size_ids(generator)
    return {**config, **dict(zip(SPECIALIZED_WORKGROUP_SIZE_KEYS, ids))}


def _workgroup_entries(generator, ast, top_level_entries) -> list:
    """The metadata that sizes each workgroup entry, as the execution modes read it (``None``: the module's)."""
    stages = getattr(ast, "stages", None) or {}
    if stages:
        return [
            stage
            for stage_type, stage in stages.items()
            if generator.spirv_execution_model(generator.stage_key(stage_type))
            in _WORKGROUP_MODELS
        ]
    entries = []
    for function, qualifier in top_level_entries:
        model = generator.spirv_execution_model(qualifier)
        if model == "GLCompute":
            entries.append(function)
        elif model in _WORKGROUP_MODELS:
            entries.append(None)
    return entries


def _declared_specialization_ids(generator, ast, entries) -> set:
    nodes = [
        *(getattr(ast, "constants", None) or []),
        *(getattr(ast, "global_variables", None) or []),
    ]
    for stage in (getattr(ast, "stages", None) or {}).values():
        nodes += [
            *(getattr(stage, "constants", None) or []),
            *(getattr(stage, "local_variables", None) or []),
        ]
    used = set()
    for node in nodes:
        declaration = getattr(node, "left", node)
        if not generator.spirv_specialization_constant_attributes(declaration):
            continue
        try:
            used.add(generator.spirv_specialization_constant_id(declaration))
        except ValueError:
            continue  # The declaration's own generation reports a malformed id.
    for metadata in entries:
        try:
            ids = workgroup_size_specialization_ids(_raw_config(generator, metadata))
        except ValueError:
            continue
        used.update(spec_id for spec_id in ids if spec_id is not None)
    return used


def _signature(generator, metadata):
    config = effective_workgroup_config(generator, _raw_config(generator, metadata))
    try:
        size = tuple(generator.compute_local_size(metadata))
        ids = workgroup_size_specialization_ids(config, "SPIR-V")
    except ValueError:
        # The entry's execution modes report it; until then it shares its size with no other entry.
        return ("unsized", id(metadata))
    return size, ids


def plan_entry_workgroup_sizes(generator, ast, top_level_entries) -> None:
    """Record whether the module's workgroup entries share one size, before any function reads it."""
    entries = _workgroup_entries(generator, ast, top_level_entries)
    generator.host_workgroup_size_spec_ids = None
    generator.used_workgroup_specialization_ids = _declared_specialization_ids(
        generator, ast, entries
    )
    generator.entry_workgroup_size_metadata = {
        id(metadata): metadata for metadata in entries if metadata is not None
    }
    generator.entry_workgroup_size_values = {}
    generator.module_has_one_workgroup_size = (
        len({_signature(generator, metadata) for metadata in entries}) <= 1
    )


def host_workgroup_size_ids(generator) -> tuple[int, int, int]:
    """The specialization ids of the x, y and z dimensions the host chooses, allocated once per module."""
    ids = getattr(generator, "host_workgroup_size_spec_ids", None)
    if ids is not None:
        return ids
    used = set(getattr(generator, "used_workgroup_specialization_ids", None) or ())
    for decoration in getattr(generator, "decorations", ()) or ():
        used.update(int(match) for match in _SPEC_ID.findall(decoration))
    allocated = []
    candidate = 0
    while len(allocated) < 3:
        if candidate not in used:
            allocated.append(candidate)
        candidate += 1
    generator.host_workgroup_size_spec_ids = tuple(allocated)
    return generator.host_workgroup_size_spec_ids


def _current_entry(generator):
    if generator.current_stage is not None:
        return generator.current_stage
    function = generator.current_function_node
    metadata = generator.entry_workgroup_size_metadata
    if function is not None and id(function) in metadata:
        return metadata[id(function)]
    return _SHARED


def _components(generator, metadata):
    from .spirv_specialized_workgroup_size import workgroup_size_components

    uint_type = generator.register_primitive_type("uint")
    return workgroup_size_components(generator, metadata, uint_type)


def entry_workgroup_size_value(generator, name: str, type_id):
    """``gl_WorkGroupSize`` of the entry being generated, in a module whose entries have different sizes."""
    from .SPIRV_codegen import SpirvId

    metadata = _current_entry(generator)
    if metadata is _SHARED:
        generator.refuse_lowering(
            f"SPIR-V {name} read outside an entry point has no single value: the module's compute, mesh and "
            "task entry points have different workgroup sizes"
        )
        metadata = None
    key = _signature(generator, metadata)
    cached = generator.entry_workgroup_size_values.get(key)
    if cached is not None:
        return cached
    components, specialized = _components(generator, metadata)
    id_value = generator.get_id()
    opcode = "OpSpecConstantComposite" if specialized else "OpConstantComposite"
    component_list = " ".join(f"%{component.id}" for component in components)
    generator.emit(f"%{id_value} = {opcode} %{type_id.id} {component_list}")
    generator.emit(f'OpName %{id_value} "{name}"')
    value = SpirvId(id_value, type_id.type, name)
    generator.value_types[id_value] = type_id
    generator.entry_workgroup_size_values[key] = value
    return value


def emit_entry_local_size_id(generator, function_id, metadata) -> bool:
    """Size a specialized entry of a module of several sizes with ``LocalSizeId``; whether it did."""
    if generator.module_has_one_workgroup_size:
        return False
    config = effective_workgroup_config(generator, _raw_config(generator, metadata))
    if not has_specialized_workgroup_size(config):
        return False
    if generator.requires_compute_derivatives:
        generator.refuse_lowering(
            "SPIR-V compute derivatives need even workgroup x and y dimensions, which a workgroup size "
            "the pipeline specializes does not guarantee"
        )
        return True
    components, _ = _components(generator, metadata)
    operands = " ".join(f"%{component.id}" for component in components)
    generator.emit(f"OpExecutionModeId %{function_id.id} LocalSizeId {operands}")
    generator.uses_local_size_id = True
    return True
