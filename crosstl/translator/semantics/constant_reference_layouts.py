"""Constant-buffer references with a declared layout, as constant buffers for targets that place their bytes.

An MSL stage entry binds a constant buffer as a reference parameter, ``constant Params& params [[buffer(2)]]``, and
the buffer holds one ``Params`` in the natural layout of MSL: a ``float3`` takes 16 bytes and a ``packed_float3``
12 at 4-byte alignment (MSL 3.2 section 2.2 and table 2.4). CrossGL keeps the parameter, and marks it
``@ buffer_layout(metal)`` where that layout differs from the ones the other targets give a reference parameter:
OpenGL and SPIR-V declare it as a std140 uniform block, which puts the ``float`` after a ``vec3`` at byte 12, and
HLSL as a ``ConstantBuffer<Params>``, which packs it the same way and moves a vector that would straddle a 16-byte
register.

The generators of those targets reproduce a declared layout for module-level constant buffers
(:mod:`..codegen.glsl_hlsl_cbuffer`, :mod:`..codegen.hlsl_glsl_block_registers`,
:mod:`..codegen.spirv_placed_blocks`). :func:`hoist_constant_reference_layouts` therefore moves each such parameter of
a stage entry into one: ``constant Params& params @buffer(2)`` becomes

    cbuffer params_block @buffer(2) @buffer_layout(metal) { Params params; }

with the parameter's attributes, and the parameter is removed. The block holds exactly one ``Params`` at byte 0, so
its bytes are the buffer's, and every ``params.member`` read in the entry and its callees reads the block member of
the same name. Two entries may share one block when they declare the same parameter; parameters that share a name but
not a type or binding, a marked reference parameter of a function that is not a stage entry, and a marked parameter
the function writes through, raise :class:`ConstantReferenceLayoutError`.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.glsl_block_layout import declared_buffer_layout
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions

__all__ = [
    "TARGETS",
    "ConstantReferenceLayoutError",
    "hoist_constant_reference_layouts",
]

HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
TARGETS = frozenset({"opengl", "glsl", "webgl", "vulkan", "spirv"}) | HLSL_TARGETS


class ConstantReferenceLayoutError(TranslationDiagnostic):
    """A constant-buffer reference with a declared layout that the target cannot bind as a constant buffer."""

    code = f"{CODE_PREFIX}constant-reference-layout-unsupported"


def _is_constant_reference(parameter):
    param_type = getattr(parameter, "param_type", None)
    return (
        isinstance(param_type, A.ReferenceType)
        and str(getattr(param_type, "address_space", "") or "").lower() == "constant"
    )


def _is_stage_entry(function, entries):
    if id(function) in entries:
        return True
    return any(
        str(getattr(attribute, "name", "")).lower() == "stage_entry"
        for attribute in getattr(function, "attributes", None) or []
    )


def _stage_entries(program):
    stages = getattr(program, "stages", None) or {}
    values = stages.values() if isinstance(stages, dict) else stages
    return {
        id(stage.entry_point)
        for stage in values
        if getattr(stage, "entry_point", None) is not None
    }


def _attribute_key(attribute):
    arguments = tuple(
        str(getattr(argument, "name", getattr(argument, "value", argument)))
        for argument in getattr(attribute, "arguments", None) or []
    )
    return str(getattr(attribute, "name", "")), arguments


def _type_key(type_node):
    return str(type_node)


def _module_names(program):
    names = set()
    for collection in ("structs", "cbuffers", "global_variables", "constants"):
        for node in getattr(program, collection, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    names.update(
        function.name
        for function in iter_functions(program)
        if isinstance(getattr(function, "name", None), str)
    )
    return names


def _block_name(parameter_name, taken):
    name = f"{parameter_name}_block"
    index = 1
    while name in taken:
        index += 1
        name = f"{parameter_name}_block{index}"
    taken.add(name)
    return name


def _hoisted_parameters(program, target):
    """``[(function, parameter, layout)]`` for every marked constant reference, refusing the ones no block holds."""
    entries = _stage_entries(program)
    found = []
    for function in iter_functions(program):
        for parameter in getattr(function, "parameters", None) or []:
            layout = declared_buffer_layout(getattr(parameter, "attributes", None))
            if layout is None or not _is_constant_reference(parameter):
                continue
            if not _is_stage_entry(function, entries):
                raise ConstantReferenceLayoutError(
                    f"constant reference parameter '{parameter.name}' of '{function.name}' declares the {layout} "
                    "layout, but only a stage entry binds a constant buffer",
                    target=target,
                )
            if getattr(parameter.param_type, "is_mutable", False):
                raise ConstantReferenceLayoutError(
                    f"constant reference parameter '{parameter.name}' of '{function.name}' is mutable, but a "
                    "constant buffer is read-only",
                    target=target,
                )
            found.append((function, parameter, layout))
    return found


def hoist_constant_reference_layouts(program, target) -> int:
    """Move marked constant-reference parameters of stage entries into module constant buffers; returns how many
    blocks were declared."""
    target = str(target or "")
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    hoisted = _hoisted_parameters(program, target)
    if not hoisted:
        return 0
    cbuffers = getattr(program, "cbuffers", None)
    if not isinstance(cbuffers, list):
        raise ConstantReferenceLayoutError(
            "the program has no constant-buffer list to receive the constant references",
            target=target,
        )
    taken = _module_names(program)
    blocks = {}
    for function, parameter, _layout in hoisted:
        referenced = parameter.param_type.referenced_type
        key = (
            _type_key(referenced),
            tuple(_attribute_key(attribute) for attribute in parameter.attributes),
        )
        previous = blocks.get(parameter.name)
        if previous is not None:
            if previous[0] != key:
                raise ConstantReferenceLayoutError(
                    f"constant reference parameter '{parameter.name}' of '{function.name}' differs in type or "
                    "binding from a parameter of the same name in another stage entry; the target declares both "
                    "as one module constant buffer",
                    target=target,
                )
        else:
            member = A.StructMemberNode(parameter.name, referenced)
            block = A.StructNode(
                _block_name(parameter.name, taken),
                [member],
                attributes=list(parameter.attributes),
            )
            block.is_cbuffer = True
            blocks[parameter.name] = (key, block)
            cbuffers.append(block)
        function.parameters = [
            other for other in function.parameters if other is not parameter
        ]
    return len(blocks)
