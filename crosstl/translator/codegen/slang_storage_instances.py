"""CrossGL storage-buffer structure instances, ``buffer S name``, as Slang byte-address buffers.

CrossGL declares a storage buffer that holds one structure as ``buffer S name`` and reads its members as ``name.m``.
The GLSL generator writes it as the block ``layout(std430) buffer S { ... } name`` (GLSL 4.60 section 4.3.9) and
the SPIR-V generator as a ``Block``-decorated ``StorageBuffer`` variable laid out by std430. The Slang generator wrote
``S name;``, which Slang reads as a uniform shader parameter (Slang user's guide, "Shader Parameters"): stores had no
buffer to reach, and ``atomicAdd(name.m, v)`` reached no storage either.

:func:`lower_slang_storage_instances` rewrites each such declaration into the form the GLSL importer gives that GLSL
block, ``S name @glsl_buffer_block(std430)``, and the generator lowers it as it lowers an imported block: to a
``RWByteAddressBuffer`` (``ByteAddressBuffer`` when it is ``readonly``) whose members are read, written and updated
atomically at their std430 offsets. The buffer then holds the bytes the GLSL and SPIR-V outputs hold, on SPIR-V and
DXIL alike. A ``RWStructuredBuffer<S>`` would not: D3D limits a structured-buffer element to 2048 bytes, so a block
with a large or runtime-sized array member has no structured-buffer form there.

The declaration keeps its other qualifiers and its attributes, so its binding and access are emitted as before. A
layout the declaration names itself (``std140``, ``std430`` or ``scalar``) is kept. A member the byte-address lowering
cannot lay out is refused by that lowering. The rewrite leaves alone a declaration that already is a block, and an
array ``buffer S name[N]``, which names the buffer's elements and is bound as ``RWStructuredBuffer<S>``.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A

__all__ = ["lower_slang_storage_instances"]

_LAYOUTS = ("std430", "std140", "scalar")


def _program_variables(program):
    """Every module-scope and stage-scope variable declaration of ``program``."""
    variables = list(getattr(program, "global_variables", None) or [])
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if hasattr(stages, "values") else stages:
        variables.extend(getattr(stage, "local_variables", None) or [])
    return variables


def _struct_names(program):
    names = {struct.name for struct in getattr(program, "structs", None) or []}
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if hasattr(stages, "values") else stages:
        names.update(
            struct.name for struct in getattr(stage, "local_structs", None) or []
        )
    return names


def _attribute_names(node):
    return [
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(node, "attributes", None) or []
    ]


def _is_storage_instance(node, struct_names):
    """Whether ``node`` declares ``buffer S name`` for a structure ``S``."""
    if not isinstance(node, A.VariableNode) or isinstance(node, A.ArrayNode):
        return False
    qualifiers = {
        str(value).lower() for value in getattr(node, "qualifiers", None) or []
    }
    if "buffer" not in qualifiers or "glsl_buffer_block" in _attribute_names(node):
        return False
    declared = getattr(node, "var_type", None)
    return (
        isinstance(declared, A.NamedType)
        and not declared.generic_args
        and declared.name in struct_names
    )


def _as_buffer_block(node):
    names = _attribute_names(node)
    layout = next((name for name in _LAYOUTS if name in names), "std430")
    node.qualifiers = [
        qualifier for qualifier in node.qualifiers if str(qualifier).lower() != "buffer"
    ]
    node.attributes = [
        attribute
        for attribute in node.attributes or []
        if str(getattr(attribute, "name", attribute)).lower() not in _LAYOUTS
    ]
    node.attributes.insert(
        0, A.AttributeNode("glsl_buffer_block", [A.IdentifierNode(layout)])
    )


def lower_slang_storage_instances(program):
    """Return ``program`` with every ``buffer S name`` declared as a std430 storage block.

    ``program`` itself is not changed: the rewrite works on a copy, made only when there is a declaration to rewrite.
    """
    if not isinstance(program, A.ShaderNode):
        return program
    struct_names = _struct_names(program)
    if not any(
        _is_storage_instance(node, struct_names) for node in _program_variables(program)
    ):
        return program
    program = deepcopy(program)
    for node in _program_variables(program):
        if _is_storage_instance(node, struct_names):
            _as_buffer_block(node)
    return program
