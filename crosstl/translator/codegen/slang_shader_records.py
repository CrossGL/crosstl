"""Shader record buffers in Slang output: a ``[[vk::shader_record]]`` constant buffer in the record's own layout.

A ray tracing stage reads the data of its shader binding table record through a CrossGL shader record block
(:mod:`..semantics.shader_records`), ``Record record @glsl_buffer_block(shaderRecordEXT, std430);``. Slang declares it
as a constant buffer with the ``[[vk::shader_record]]`` attribute, which slangc places in the
``ShaderRecordBufferKHR`` storage class without a descriptor binding (slangc 2026.20). A constant buffer takes its
layout from its second type argument, so the declaration keeps the record's layout exactly::

    [[vk::shader_record]] ConstantBuffer<Record, Std430DataLayout> record;   // std430, the default
    [[vk::shader_record]] ConstantBuffer<Record> record;                     // std140
    [[vk::shader_record]] ConstantBuffer<Record, ScalarDataLayout> record;   // scalar

The record keeps its struct declaration and member accesses (``record.color``), because the Slang generator does not
lower a shader record as it lowers the other GLSL buffer blocks (to a byte-address buffer). A stage has one record,
so an array of records is refused; :func:`..semantics.shader_records.refuse_shader_record_buffers` has already
refused a record that a stage without a shader binding table reads.
"""

from __future__ import annotations

from ..ast import ArrayType
from ..semantics.shader_records import (
    is_shader_record_buffer,
    shader_record_memory_layout,
)
from .glsl_buffer_layout import glsl_buffer_block_node_type

__all__ = ["slang_shader_record_declaration"]

# The second type argument of ConstantBuffer for each record layout; std140 is a constant buffer's default.
_LAYOUT_ARGUMENTS = {
    "std430": ", Std430DataLayout",
    "std140": "",
    "scalar": ", ScalarDataLayout",
}


def slang_shader_record_declaration(codegen, node):
    """The Slang declaration of the shader record block ``node``, or ``None`` when ``node`` is not one."""
    if not is_shader_record_buffer(node):
        return None
    name = getattr(node, "name", "?")
    node_type = glsl_buffer_block_node_type(node)
    if isinstance(node_type, ArrayType):
        return codegen.refuse_lowering(
            f"shader record buffer '{name}': a stage has one shader record, not an array of them"
        )
    record_type = str(codegen.resource_base_type(node_type))
    if record_type not in codegen.structs_by_name:
        return codegen.refuse_lowering(
            f"shader record buffer '{name}': its type '{record_type}' is not a structure"
        )
    codegen.register_variable_type(name, record_type, node)
    layout = _LAYOUT_ARGUMENTS[shader_record_memory_layout(node)]
    return f"[[vk::shader_record]] ConstantBuffer<{record_type}{layout}> {name};\n"
