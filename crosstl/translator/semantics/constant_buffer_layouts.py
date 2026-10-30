"""Constant buffers declared as ``ConstantBuffer<T>`` with a layout, as constant blocks that hold one ``T``.

``ConstantBuffer<T> name`` is CrossGL's spelling of an HLSL ``ConstantBuffer<T>`` and of a uniform block with an
instance name (GLSL's ``uniform Block { ... } name;``, CrossGL's ``cbuffer Block { ... } name;``). A
``@ buffer_layout(L)`` attribute on it places the members of ``T`` by the rules of ``L``. The generators declared
such a buffer through the target's own constant-buffer form, a std140 ``uniform`` block in OpenGL and SPIR-V
(GLSL 4.60 section 7.6.2.2), an HLSL ``ConstantBuffer<T>`` under its packing rules and an MSL ``constant T&`` under
MSL's natural layout (MSL 3.2 section 2.2), and dropped the attribute. With ``struct T { float a; vec3 b; }`` a std140
buffer then read ``b`` at byte 4 in HLSL and MSL where the host wrote it at byte 16.

The generators reproduce a declared layout for constant blocks, members of struct type included
(:mod:`..codegen.glsl_hlsl_cbuffer`, :mod:`..codegen.hlsl_glsl_block_registers`, :mod:`..codegen.spirv_placed_blocks`
and the Metal generator's padded structs). :func:`hoist_constant_buffer_layouts` therefore declares each such buffer
whose layout differs from the target's own constant-buffer rules as

    cbuffer name_block @attrs { T name; }

with the declaration's attributes. The block holds one ``T`` at byte 0, so its bytes are the buffer's, and every
``name.member`` read reads the block member of the same name, as :mod:`.constant_reference_layouts` does for Metal's
constant reference parameters. A declaration in a stage becomes a block of that stage. An array of constant buffers
binds one descriptor per element, which one block cannot hold, and is left to the generator.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.glsl_block_layout import declared_buffer_layout
from .constant_buffers import constant_buffer_element
from .hub_types import iter_functions

__all__ = ["TARGET_LAYOUTS", "hoist_constant_buffer_layouts"]

# Each target's own constant-buffer layout: a buffer declared with it needs no block.
TARGET_LAYOUTS = {
    "opengl": "std140",
    "glsl": "std140",
    "webgl": "std140",
    "vulkan": "std140",
    "spirv": "std140",
    "directx": "hlsl_cbuffer",
    "hlsl": "hlsl_cbuffer",
    "dx11": "hlsl_cbuffer",
    "dx12": "hlsl_cbuffer",
    "d3d11": "hlsl_cbuffer",
    "d3d12": "hlsl_cbuffer",
    "metal": "metal",
}


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _module_names(program):
    names = set()
    for collection in ("structs", "cbuffers", "global_variables", "constants"):
        for node in getattr(program, collection, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    for stage in _stages(program):
        for collection in ("local_structs", "local_cbuffers", "local_variables"):
            for node in getattr(stage, collection, None) or []:
                name = getattr(node, "name", None)
                if isinstance(name, str):
                    names.add(name)
    names.update(
        function.name
        for function in iter_functions(program)
        if isinstance(getattr(function, "name", None), str)
    )
    return names


def _block_name(variable_name, taken):
    name = f"{variable_name}_block"
    index = 1
    while name in taken:
        index += 1
        name = f"{variable_name}_block{index}"
    taken.add(name)
    return name


def _hoisted(variables, native):
    """The ``ConstantBuffer<T>`` declarations of ``variables`` whose declared layout is not ``native``."""
    found = []
    for variable in variables or []:
        if not isinstance(variable, A.VariableNode):
            continue
        element = constant_buffer_element(getattr(variable, "var_type", None))
        if element is None:
            continue
        layout = declared_buffer_layout(getattr(variable, "attributes", None))
        if layout is None or layout == native:
            continue
        found.append((variable, element))
    return found


def _hoist(variables, blocks, native, taken):
    hoisted = _hoisted(variables, native)
    for variable, element in hoisted:
        block = A.StructNode(
            _block_name(variable.name, taken),
            [A.StructMemberNode(variable.name, element)],
            attributes=list(getattr(variable, "attributes", None) or []),
        )
        block.is_cbuffer = True
        blocks.append(block)
    removed = {id(variable) for variable, _element in hoisted}
    return [variable for variable in variables if id(variable) not in removed], len(
        hoisted
    )


def hoist_constant_buffer_layouts(program, target) -> int:
    """Declare the ``ConstantBuffer<T>`` buffers whose declared layout ``target`` does not give as constant blocks
    holding one ``T``; returns how many were declared."""
    native = TARGET_LAYOUTS.get(str(target or "").lower())
    if native is None or not isinstance(program, A.ShaderNode):
        return 0
    taken = _module_names(program)
    count = 0
    globals_ = getattr(program, "global_variables", None)
    if isinstance(globals_, list) and _hoisted(globals_, native):
        if not isinstance(getattr(program, "cbuffers", None), list):
            program.cbuffers = []
        program.global_variables, hoisted = _hoist(
            globals_, program.cbuffers, native, taken
        )
        count += hoisted
    for stage in _stages(program):
        variables = getattr(stage, "local_variables", None)
        if not isinstance(variables, list) or not _hoisted(variables, native):
            continue
        if not isinstance(getattr(stage, "local_cbuffers", None), list):
            stage.local_cbuffers = []
        stage.local_variables, hoisted = _hoist(
            variables, stage.local_cbuffers, native, taken
        )
        count += hoisted
    return count
