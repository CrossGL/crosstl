"""Constant buffers larger than Direct3D allows.

Direct3D limits a constant buffer view to 4096 16-byte registers, 64 KiB (D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT,
D3D12 likewise), and DXC's validator rejects a larger ``cbuffer`` ("CBuffer size is ... exceeding maximum of 65536
bytes"). Metal's constant address space and GLSL uniform blocks have no such shader-side limit (Vulkan's
``maxUniformBufferRange`` is a device limit the host checks). Nor can HLSL hold the data in a structured buffer
element, which is limited to 2048 bytes. So for HLSL targets, :func:`refuse_large_constant_buffers` raises
:class:`ConstantBufferTooLargeError` for a ``constant T&`` entry argument or a constant block whose contents exceed
64 KiB, instead of emitting a ``cbuffer`` DXC rejects.

:func:`hlsl_constant_buffer_size` packs a type by the HLSL packing rules for constant variables: a vector does not
straddle a 16-byte register; arrays, structures and matrix columns start a register; array elements take whole
registers except the last. Where a rule depends on what follows, it takes the larger size, so the result is never
below what DXC computes. A type the bound cannot size (an unsized array, an unknown type) gives ``None``, and is left
to the generator.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_structs, type_from_node

__all__ = [
    "CONSTANT_BUFFER_LIMIT",
    "ConstantBufferTooLargeError",
    "hlsl_constant_buffer_size",
    "refuse_large_constant_buffers",
]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})

CONSTANT_BUFFER_LIMIT = 65536
_REGISTER = 16
_MATRIX = re.compile(r"(?:d|f16)?mat([234])(?:x([234]))?")


def _round(value, alignment=_REGISTER):
    return -(-value // alignment) * alignment


def _matrix_shape(type_node):
    if isinstance(type_node, A.MatrixType):
        if str(type_node.cols).isdigit() and str(type_node.rows).isdigit():
            element = type_from_node(type_node.element_type)
            bits = element.bits if element is not None else 32
            return int(type_node.cols), int(type_node.rows), max(4, bits // 8)
        return None
    name = str(getattr(type_node, "name", "") or "")
    match = _MATRIX.fullmatch(name)
    if match is None:
        return None
    columns = int(match.group(1))
    rows = int(match.group(2) or columns)
    width = 8 if name.startswith("d") else 4
    return columns, rows, width


class _Packer:
    def __init__(self, structs):
        self.structs = structs
        self.active = set()

    def size(self, type_node):
        """The bytes ``type_node`` occupies from a register boundary, or ``None``."""
        if isinstance(type_node, A.ArrayType):
            size = getattr(type_node, "size", None)
            count = getattr(size, "value", size)
            if isinstance(count, str) and count.isdigit():
                count = int(count)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                return None
            element = self.size(type_node.element_type)
            if element is None:
                return None
            return (count - 1) * _round(element) + element
        matrix = _matrix_shape(type_node)
        if matrix is not None:
            columns, rows, width = matrix
            # Column-major: one register per column; a row-major matrix takes at most as many registers.
            return (max(columns, rows) - 1) * _REGISTER + max(columns, rows) * width
        value = type_from_node(type_node)
        if value is not None:
            return max(4, value.bits // 8) * value.lanes
        name = getattr(type_node, "name", None)
        struct = self.structs.get(name) if isinstance(name, str) else None
        if struct is None or name in self.active:
            return None
        self.active.add(name)
        try:
            return self.members(struct)
        finally:
            self.active.discard(name)

    def members(self, struct):
        offset = 0
        for member in getattr(struct, "members", None) or []:
            member_type = getattr(member, "member_type", None)
            size = self.size(member_type)
            if size is None:
                return None
            starts_register = (
                isinstance(member_type, A.ArrayType)
                or _matrix_shape(member_type) is not None
                or type_from_node(member_type) is None
            )
            if starts_register or offset % _REGISTER + min(size, _REGISTER) > _REGISTER:
                offset = _round(offset)
            offset += size
            if starts_register:
                # What follows an array, matrix or structure starts a register.
                offset = _round(offset)
        return offset


def hlsl_constant_buffer_size(type_node, structs):
    """An upper bound on the bytes a ``cbuffer`` holding ``type_node`` takes, or ``None``."""
    size = _Packer(structs).size(type_node)
    return None if size is None else _round(size)


class ConstantBufferTooLargeError(TranslationDiagnostic):
    """Constant data larger than a Direct3D constant buffer holds."""

    def __init__(self, target, name, size, *, source_location=None):
        self.buffer_name = name
        self.size = size
        self.source_location = source_location
        super().__init__(
            f"the constant buffer '{name}' takes at least {size} bytes, but a Direct3D constant buffer holds at "
            f"most {CONSTANT_BUFFER_LIMIT} and a structured buffer element at most 2048",
            target=target,
            code=f"{CODE_PREFIX}{target}-constant-buffer-too-large",
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def refuse_large_constant_buffers(program, target):
    """For HLSL targets, refuse constant entry arguments and constant blocks larger than 64 KiB."""
    target = str(target or "").lower()
    if target not in _HLSL_TARGETS or not isinstance(program, A.ShaderNode):
        return
    structs = {
        node.name: node
        for node in iter_structs(program)
        if isinstance(getattr(node, "name", None), str)
    }
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        for parameter in getattr(entry, "parameters", None) or []:
            declared = getattr(parameter, "param_type", None)
            if not isinstance(declared, A.ReferenceType):
                continue
            if str(getattr(declared, "address_space", "") or "").lower() != "constant":
                continue
            size = hlsl_constant_buffer_size(declared.referenced_type, structs)
            if size is not None and size > CONSTANT_BUFFER_LIMIT:
                raise ConstantBufferTooLargeError(
                    target,
                    parameter.name,
                    size,
                    source_location=getattr(parameter, "source_location", None),
                )
    blocks = list(getattr(program, "cbuffers", None) or [])
    for stage in _stages(program):
        blocks.extend(getattr(stage, "local_cbuffers", None) or [])
    for block in blocks:
        size = _Packer(structs).members(block)
        if size is not None and _round(size) > CONSTANT_BUFFER_LIMIT:
            raise ConstantBufferTooLargeError(
                target,
                getattr(block, "name", "?"),
                _round(size),
                source_location=getattr(block, "source_location", None),
            )
