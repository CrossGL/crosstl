"""DXC shader record buffers imported as CrossGL shader record blocks.

DXC reads ``[[vk::shader_record_ext]]`` (or ``[[vk::shader_record_nv]]``) on a ``ConstantBuffer<T>`` or ``cbuffer`` as
the shader binding table record of a ray tracing stage: SPIR-V declares it in the ``ShaderRecordBufferKHR`` storage
class with no descriptor set or binding, and any ``register`` is ignored (DXC SPIR-V codegen reference, "Vulkan
specific attributes"). Direct3D has no counterpart in HLSL, so DXC drops the attribute when it emits DXIL.

The importer used to pass the attribute through as ``@ vk_shader_record_ext``, which no CrossGL consumer knows: SPIR-V
bound the buffer as a uniform buffer at set 0, binding 0, GLSL redeclared it as ``uniform`` with ``binding = 0``, and
HLSL printed the attribute as an array size. The stage then read a descriptor the host never binds instead of its
record. A ``ConstantBuffer<T>`` record now becomes CrossGL's shader record block (see
:mod:`crosstl.translator.semantics.shader_records`)::

    T name @glsl_buffer_block(shaderRecordEXT, std430);

DXC lays a shader record out with its storage-buffer rule, the relaxed std430 rule: a vector member of 32-bit
components is aligned to its component size unless it would straddle a 16-byte boundary, while arrays and structs
keep their std430 alignment. CrossGL records take ``std430``, ``std140`` or ``scalar`` layouts, so the importer names
the one whose offsets equal DXC's (``std430`` first, then ``scalar``) and refuses a record that matches neither, one
whose members have no fixed 32- or 64-bit layout here (matrices, 16-bit and minimum-precision types, ``half``,
whose width depends on ``-enable-16bit-types``), and the ``cbuffer`` form, whose members CrossGL would reach as
globals of a block it cannot name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SHADER_RECORD_ATTRIBUTES",
    "HLSLShaderRecordError",
    "is_hlsl_shader_record",
    "shader_record_layout",
]

# DXC attribute -> the GLSL layout qualifier CrossGL records carry. The NV spelling selects the
# ShaderRecordBufferNV storage class, which SPV_NV_ray_tracing gives the same value as ShaderRecordBufferKHR.
SHADER_RECORD_ATTRIBUTES = {
    "vk::shader_record_ext": "shaderRecordEXT",
    "vk::shader_record_nv": "shaderRecordEXT",
}

_SCALAR_SIZES = {
    "bool": 4,
    "int": 4,
    "uint": 4,
    "dword": 4,
    "float": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "float32_t": 4,
    "double": 8,
    "float64_t": 8,
    "int64_t": 8,
    "uint64_t": 8,
}
_VECTOR = re.compile(r"([a-z_0-9]+?)([1-4])")
_TEMPLATE_VECTOR = re.compile(r"vector\s*<\s*([a-z_0-9]+)\s*,\s*([1-4])\s*>")
_CONSTANT_BUFFER = re.compile(r"ConstantBuffer\s*<\s*(\w+)\s*>")
_LAYOUTS = ("std430", "scalar")


class HLSLShaderRecordError(ValueError):
    """A DXC shader record buffer that CrossGL cannot express with DXC's layout."""

    project_diagnostic_code = "project.translate.hlsl-shader-record-unsupported"

    def __init__(self, name, reason):
        self.reason = reason
        super().__init__(
            f"Cannot import the DXC shader record buffer '{name}': {reason}"
        )


@dataclass(frozen=True)
class _Shape:
    kind: str  # "scalar", "vector", "array" or "struct"
    size: int  # component size for scalars and vectors
    count: int = 1  # components of a vector, elements of an array
    element: _Shape | None = None
    members: tuple = ()  # ((name, shape), ...) of a struct


def is_hlsl_shader_record(node):
    """The CrossGL record qualifier of a declaration DXC places in its shader record, or ``None``."""
    for attribute in getattr(node, "attributes", None) or []:
        qualifier = SHADER_RECORD_ATTRIBUTES.get(str(getattr(attribute, "name", "")))
        if qualifier is not None:
            return qualifier
    return None


def shader_record_layout(node, structs_by_name, *, is_cbuffer=False):
    """``(struct name, CrossGL layout)`` for a ``ConstantBuffer<T>`` shader record; raises when there is none."""
    name = str(getattr(node, "name", "") or "")
    if is_cbuffer:
        raise HLSLShaderRecordError(
            name,
            "only the ConstantBuffer<T> form is supported; declare the record as "
            "'[[vk::shader_record_ext]] ConstantBuffer<T> name;'",
        )
    match = _CONSTANT_BUFFER.fullmatch(str(getattr(node, "vtype", "") or "").strip())
    if match is None or match.group(1) not in structs_by_name:
        raise HLSLShaderRecordError(
            name, "the record must be a ConstantBuffer<T> of a structure T"
        )
    if getattr(node, "array_sizes", None) or getattr(node, "array_size", None):
        raise HLSLShaderRecordError(name, "a stage has one shader record, not an array")
    struct_name = match.group(1)
    shape = _struct_shape(struct_name, structs_by_name, name, ())
    relaxed = _offsets(shape, "relaxed")
    for layout in _LAYOUTS:
        if _offsets(shape, layout) == relaxed:
            return struct_name, layout
    raise HLSLShaderRecordError(
        name,
        f"DXC's relaxed std430 offsets of {struct_name} match neither the std430 nor "
        "the scalar layout",
    )


def _struct_shape(struct_name, structs_by_name, record, seen):
    if struct_name in seen:
        raise HLSLShaderRecordError(record, f"structure {struct_name} contains itself")
    struct = structs_by_name[struct_name]
    members = []
    for member in getattr(struct, "members", None) or []:
        shape = _member_shape(
            str(getattr(member, "vtype", "") or ""),
            structs_by_name,
            record,
            seen + (struct_name,),
        )
        for extent in reversed(list(getattr(member, "array_sizes", None) or [])):
            if not isinstance(extent, int) or extent <= 0:
                raise HLSLShaderRecordError(
                    record,
                    f"member {member.name} of {struct_name} has no fixed array extent",
                )
            shape = _Shape("array", 0, extent, shape)
        members.append((str(member.name), shape))
    if not members:
        raise HLSLShaderRecordError(record, f"structure {struct_name} has no members")
    return _Shape("struct", 0, members=tuple(members))


def _member_shape(type_name, structs_by_name, record, seen):
    text = re.sub(r"\s+", " ", type_name).strip()
    if text in structs_by_name:
        return _struct_shape(text, structs_by_name, record, seen)
    if text in _SCALAR_SIZES:
        return _Shape("scalar", _SCALAR_SIZES[text])
    vector = _TEMPLATE_VECTOR.fullmatch(text) or _VECTOR.fullmatch(text)
    if vector is not None and vector.group(1) in _SCALAR_SIZES:
        count = int(vector.group(2))
        size = _SCALAR_SIZES[vector.group(1)]
        return _Shape("scalar", size) if count == 1 else _Shape("vector", size, count)
    raise HLSLShaderRecordError(
        record, f"member type '{type_name}' has no fixed record layout in CrossGL"
    )


def _round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _alignment(shape, layout):
    """Base alignment; DXC's relaxed rule relaxes only vectors placed as struct members, so it uses std430's."""
    if shape.kind == "scalar":
        return shape.size
    if shape.kind == "vector":
        if layout == "scalar":
            return shape.size
        return (4 if shape.count == 3 else shape.count) * shape.size
    if shape.kind == "array":
        return _alignment(shape.element, layout)
    return max(_alignment(member, layout) for _, member in shape.members)


def _size(shape, layout):
    if shape.kind == "scalar":
        return shape.size
    if shape.kind == "vector":
        return shape.size * shape.count
    if shape.kind == "array":
        return _stride(shape, layout) * shape.count
    end = 0
    for _, member in shape.members:
        end = _place(member, end, layout) + _size(member, layout)
    return _round_up(end, _alignment(shape, layout))


def _stride(array, layout):
    return _round_up(_size(array.element, layout), _alignment(array.element, layout))


def _place(member, offset, layout):
    """The offset of a struct member placed at or after ``offset``."""
    if layout != "relaxed" or member.kind != "vector" or member.size > 4:
        return _round_up(offset, _alignment(member, layout))
    offset = _round_up(offset, member.size)
    size = _size(member, layout)
    if offset // 16 != (offset + size - 1) // 16:
        offset = _round_up(offset, 16)
    return offset


def _offsets(shape, layout, base=0, path=""):
    """Every member offset and array stride of ``shape`` placed at ``base``.

    Sizes are left out: a struct's size shows only through the offsets and strides that follow it, and the record
    block's own tail padding is never read.
    """
    if shape.kind == "array":
        return ((path, "stride", _stride(shape, layout)),) + _offsets(
            shape.element, layout, base, path + "[]"
        )
    if shape.kind != "struct":
        return ()
    found = ()
    offset = 0
    for name, member in shape.members:
        offset = _place(member, offset, layout)
        found += ((f"{path}.{name}", "offset", base + offset),)
        found += _offsets(member, layout, base + offset, f"{path}.{name}")
        offset += _size(member, layout)
    return found
