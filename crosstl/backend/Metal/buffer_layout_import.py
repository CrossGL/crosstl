"""The CrossGL buffer layout that reproduces the bytes of a Metal buffer element.

A ``device`` or ``constant`` buffer of MSL holds its elements in their natural layout: ``float2`` is 8-byte and
``float3`` and ``float4`` 16-byte sized and aligned, a matrix is an array of such columns, a struct aligns each
member to its own alignment and rounds its size to the widest, and ``bool`` takes one byte (MSL 3.2 section 2.2 and
table 2.4). The packed vector types are storage forms of the vector types with scalar alignment and no padding
(MSL 3.2 section 2.2.3), and CrossGL spells ``packed_float3`` as ``vec3``.

The HLSL structured-buffer rules lay out every 32-bit scalar and vector at four-byte alignment with no padding
(D3D11.3 functional specification section 5.3.7), and GLSL's std430 rules size ``vec3`` 12 bytes with 16-byte
alignment (GLSL 4.60 section 7.6.2.2), so ``device float3*`` has a 16-byte stride against HLSL's 12 and
``struct { float3 a; float b; }`` takes 32 bytes against 16 for both. An importer that kept the CrossGL type alone
had the other targets read and write the buffer at other offsets than the host that filled it.
:meth:`MetalBufferLayoutImport.metal_buffer_element_layout` computes the Metal layout of an element type from its
source spelling and compares it with the HLSL structured, std430 and natural MSL layouts of its CrossGL type:

* an element laid out alike by all three keeps CrossGL's default (``None``), and so does a lone ``bool`` scalar or
  vector, whose one-byte storage the host binding contract reports;
* a packed element without matrices whose Metal layout is the HLSL structured layout is marked
  ``hlsl_structured``, which every target reproduces (the Metal generator with packed storage, OpenGL and SPIR-V
  with a buffer of 32-bit words);
* an element whose Metal layout is the natural MSL layout of its CrossGL type is marked ``metal``, which the Metal
  generator keeps and HLSL, OpenGL and SPIR-V reproduce with a buffer of 32-bit words where their rules differ;
* any other packed element is refused, because no CrossGL buffer layout reproduces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ...translator.codegen.array_utils import evaluate_literal_int_expression
from .MetalAst import VariableNode
from .type_layout import metal_type_layout

__all__ = ["HLSL_STRUCTURED", "METAL", "MetalBufferLayoutImport"]

HLSL_STRUCTURED = "hlsl_structured"
METAL = "metal"

_NAMESPACE = r"(?:(?:metal|simd)::)?"
_PACKED = re.compile(rf"^{_NAMESPACE}packed_([a-z0-9_]+?)([234])$")
_MATRIX = re.compile(rf"^{_NAMESPACE}([a-z]+)([234])x([234])$")
_VECTOR = re.compile(rf"^{_NAMESPACE}([a-z_]+?)([234])$")
_GENERIC_VECTOR = re.compile(
    r"^(?:metal::)?vec\s*<\s*([A-Za-z_:][A-Za-z0-9_:]*)\s*,\s*([234])\s*>$"
)
_ARRAY = re.compile(r"^(?:metal::)?array\s*<\s*(.+)\s*,\s*([^,<>]+?)\s*>$")


class _Unknown(Exception):
    """The element type has a part this module does not model."""


@dataclass(frozen=True)
class _Type:
    form: str  # scalar, vector, matrix, array or struct
    scalar: str = ""
    size: int = 0  # bytes per scalar
    lanes: int = 1  # components of a vector, rows of a matrix
    columns: int = 0
    packed: bool = False
    element: _Type | None = None
    count: int = 0
    members: tuple = ()


def _round(value, alignment):
    return -(-value // alignment) * alignment


def _vector(size, lanes, packed, rules):
    offsets = tuple(size * lane for lane in range(lanes))
    if rules == "hlsl" or lanes == 1 or packed and rules == "metal":
        return size * lanes, size, offsets
    if rules == "metal":
        extent = size * (4 if lanes == 3 else lanes)
        return extent, extent, offsets
    return size * lanes, size * (2 if lanes == 2 else 4), offsets


def _place(node, rules):
    """``(size, alignment, offsets)`` of ``node`` under ``rules``; ``offsets`` lists every scalar component.

    Matrices are columns of ``lanes`` rows, the column-major storage CrossGL gives every target.
    """
    size = 4 if node.scalar == "bool" and rules != "metal" else node.size
    if node.form == "scalar":
        return size, size, (0,)
    if node.form == "vector":
        return _vector(size, node.lanes, node.packed, rules)
    if node.form == "matrix":
        column, alignment, offsets = _vector(size, node.lanes, False, rules)
        stride = _round(column, alignment)
        return (
            stride * node.columns,
            alignment,
            tuple(
                index * stride + offset
                for index in range(node.columns)
                for offset in offsets
            ),
        )
    if node.form == "array":
        element_size, alignment, element_offsets = _place(node.element, rules)
        stride = _round(element_size, alignment)
        offsets = tuple(
            index * stride + offset
            for index in range(node.count)
            for offset in element_offsets
        )
        return stride * node.count, alignment, offsets
    if node.form == "struct":
        offset, alignment, offsets = 0, 1, []
        for member in node.members:
            member_size, member_alignment, member_offsets = _place(member, rules)
            offset = _round(offset, member_alignment)
            offsets.extend(offset + inner for inner in member_offsets)
            offset += member_size
            alignment = max(alignment, member_alignment)
        return _round(offset, alignment), alignment, tuple(offsets)
    raise _Unknown(node.form)


def _layout(node, rules):
    size, alignment, offsets = _place(node, rules)
    return _round(size, alignment), offsets


def _unpacked(node):
    """``node`` with every packed vector replaced by its vector type: the natural MSL layout of its CrossGL type."""
    return replace(
        node,
        packed=False,
        element=None if node.element is None else _unpacked(node.element),
        members=tuple(_unpacked(member) for member in node.members),
    )


def _walk(node):
    yield node
    if node.element is not None:
        yield from _walk(node.element)
    for member in node.members:
        yield from _walk(member)


class MetalBufferLayoutImport:
    """Mixin for :class:`MetalToCrossGLConverter`: the ``@buffer_layout`` a Metal buffer element needs."""

    def metal_buffer_element_layout(self, element_type, context):
        """``"hlsl_structured"``, ``"metal"`` or ``None`` for ``element_type``; records a refusal when no layout
        fits."""
        try:
            node = self.metal_layout_type(str(element_type), ())
        except _Unknown:
            if "packed_" in str(element_type):
                self.refuse_import(
                    f"{context}: the layout of the packed element type {element_type} cannot be determined"
                )
            return None
        if node.form in ("scalar", "vector") and node.scalar == "bool":
            return None
        parts = list(_walk(node))
        packed = any(part.packed for part in parts)
        matrix = any(part.form == "matrix" for part in parts)
        metal = _layout(node, "metal")
        if metal == _layout(node, "hlsl") == _layout(node, "std430"):
            return None
        if packed and not matrix and metal == _layout(node, "hlsl"):
            return HLSL_STRUCTURED
        if metal == _layout(_unpacked(node), "metal"):
            return METAL
        self.refuse_import(
            f"{context}: the Metal layout of {element_type} matches neither the HLSL structured-buffer layout nor "
            "the natural MSL layout of its CrossGL type"
        )
        return None

    def metal_device_pointer_element_type(self, parameter):
        """The element type of a ``device`` or ``constant`` pointer parameter of a helper function, or ``None``.

        A helper that takes a buffer pointer reads and writes the caller's buffer, so its element layout is the
        buffer's (MSL 3.2 section 4.2): the parameter gets the same ``@buffer_layout`` as a buffer parameter of a
        kernel.
        """
        qualifiers = set(self.effective_declaration_qualifiers(parameter))
        if not qualifiers & {"device", "constant"}:
            return None
        raw = str(self.resolve_type_alias(getattr(parameter, "vtype", None)) or "")
        raw = raw.strip()
        if raw.count("*") != 1 or "&" in raw:
            return None
        element = self.pointer_element_type(raw)
        return self.resolve_type_alias(element) if element else None

    def metal_layout_type(self, type_text, resolving):
        text = self.resolve_type_alias(type_text.strip())
        text = re.sub(r"^(?:(?:const|volatile)\s+)+", "", str(text)).strip()
        generic = _GENERIC_VECTOR.match(text)
        if generic:
            text = f"{generic.group(1).split('::')[-1]}{generic.group(2)}"
        packed = _PACKED.match(text)
        if packed:
            return self.metal_layout_vector(packed.group(1), int(packed.group(2)), True)
        matrix = _MATRIX.match(text)
        if matrix and self.metal_layout_scalar_size(matrix.group(1)):
            return _Type(
                "matrix",
                scalar=matrix.group(1),
                size=self.metal_layout_scalar_size(matrix.group(1)),
                lanes=int(matrix.group(3)),
                columns=int(matrix.group(2)),
            )
        scalar_size = self.metal_layout_scalar_size(text)
        if scalar_size:
            return _Type("scalar", scalar=text.split("::")[-1], size=scalar_size)
        vector = _VECTOR.match(text)
        if vector and self.metal_layout_scalar_size(vector.group(1)):
            return self.metal_layout_vector(
                vector.group(1), int(vector.group(2)), False
            )
        array = _ARRAY.match(text)
        if array:
            count = self.metal_layout_extent(array.group(2))
            element = self.metal_layout_type(array.group(1), resolving)
            return _Type("array", element=element, count=count)
        return self.metal_layout_struct(text, resolving)

    def metal_layout_vector(self, scalar, lanes, packed):
        size = self.metal_layout_scalar_size(scalar)
        if not size:
            raise _Unknown(scalar)
        return _Type("vector", scalar=scalar, size=size, lanes=lanes, packed=packed)

    @staticmethod
    def metal_layout_scalar_size(name):
        if not re.fullmatch(
            r"(?:metal::)?[a-z][a-z_]*(?:8|16|32|64)?(?:_t)?", str(name)
        ):
            return 0
        layout = metal_type_layout(str(name).split("::")[-1])
        return layout[0] if layout is not None and layout[0] == layout[1] else 0

    def metal_layout_extent(self, expression):
        if isinstance(expression, str):
            extent = self.evaluate_concrete_array_extent(expression)
        else:
            extent = evaluate_literal_int_expression(
                expression, self.visible_global_integral_constant_bindings()
            )
        if not isinstance(extent, int) or extent <= 0:
            raise _Unknown(expression)
        return extent

    def metal_layout_struct(self, text, resolving):
        name = self.normalized_metal_type(text)
        struct = self.struct_declarations.get(name)
        if (
            struct is None
            or name in resolving
            or name in self.ambiguous_struct_names
            or getattr(struct, "aggregate_kind", None) == "union"
            or any(
                getattr(struct, attribute, None)
                for attribute in (
                    "alignas",
                    "attributes",
                    "template_parameters",
                    "generics",
                    "bases",
                    "base_classes",
                    "base_types",
                )
            )
        ):
            raise _Unknown(text)
        members = []
        for member in getattr(struct, "members", None) or []:
            if not isinstance(member, VariableNode):
                continue
            qualifiers = {
                str(q).lower() for q in getattr(member, "qualifiers", None) or []
            }
            if "static" in qualifiers:
                continue
            member_type = str(getattr(member, "vtype", "") or "")
            suffix = str(getattr(member, "declarator_type_suffix", "") or "")
            if (
                not member_type
                or getattr(member, "alignas", None)
                or getattr(member, "bitfield_width", None) is not None
                or any(mark in member_type + suffix for mark in "*&")
            ):
                raise _Unknown(member_type)
            node = self.metal_layout_type(member_type, (*resolving, name))
            for extent in reversed(getattr(member, "array_sizes", None) or []):
                node = _Type(
                    "array", element=node, count=self.metal_layout_extent(extent)
                )
            members.append(node)
        if not members:
            raise _Unknown(text)
        return _Type("struct", members=tuple(members))
