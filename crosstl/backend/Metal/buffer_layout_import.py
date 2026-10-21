"""The CrossGL buffer layout that reproduces the bytes of a Metal buffer element holding packed vectors.

CrossGL spells MSL's ``packed_float3`` as ``vec3``: the packed types are storage forms of the vector types, with the
same components and operations but scalar alignment and no padding (MSL 3.2 section 2.2.3). A ``device
packed_float3*`` element is 12 bytes; a ``device float3*`` element, and a GLSL std430 ``vec3[]`` element, take 16.
An importer that only renamed the type therefore had every target read and write the buffer at other offsets than
the host that filled it.

HLSL structured buffers lay out every 32-bit scalar and vector at four-byte alignment with no padding (D3D11.3
functional specification section 5.3.7), which is exactly the layout of an element built from scalars and packed
vectors. :meth:`MetalBufferLayoutImport.metal_buffer_element_layout` computes the Metal layout of an element type
from its source spelling and compares it with the HLSL structured and std430 layouts of its CrossGL type:

* an element with no packed vector keeps CrossGL's default (``None``);
* an element whose Metal layout is the HLSL structured layout is marked ``hlsl_structured``, which every target
  reproduces (the Metal generator with packed storage, OpenGL and SPIR-V with a buffer of 32-bit words);
* an element whose Metal layout is the std430 layout needs no mark (``None``);
* any other packed element, or one holding a matrix, is refused, because no CrossGL buffer layout reproduces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...translator.codegen.array_utils import evaluate_literal_int_expression
from .MetalAst import VariableNode
from .type_layout import metal_type_layout

__all__ = ["HLSL_STRUCTURED", "MetalBufferLayoutImport"]

HLSL_STRUCTURED = "hlsl_structured"

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
    lanes: int = 1
    columns: int = 0
    packed: bool = False
    element: _Type | None = None
    count: int = 0
    members: tuple = ()


def _round(value, alignment):
    return -(-value // alignment) * alignment


def _place(node, rules):
    """``(size, alignment, offsets)`` of ``node`` under ``rules``; ``offsets`` lists every scalar component."""
    size = 4 if node.scalar == "bool" and rules != "metal" else node.size
    if node.form == "scalar":
        return size, size, (0,)
    if node.form == "vector":
        offsets = tuple(size * lane for lane in range(node.lanes))
        if rules == "hlsl" or node.packed and rules == "metal":
            return size * node.lanes, size, offsets
        if rules == "metal":
            extent = size * (4 if node.lanes == 3 else node.lanes)
            return extent, extent, offsets
        alignment = size * (2 if node.lanes == 2 else 4)
        return size * node.lanes, alignment, offsets
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


def _walk(node):
    yield node
    if node.element is not None:
        yield from _walk(node.element)
    for member in node.members:
        yield from _walk(member)


class MetalBufferLayoutImport:
    """Mixin for :class:`MetalToCrossGLConverter`: the ``@buffer_layout`` a Metal buffer element needs."""

    def metal_buffer_element_layout(self, element_type, context):
        """``"hlsl_structured"`` or ``None`` for ``element_type``; records a refusal when no layout fits."""
        try:
            node = self.metal_layout_type(str(element_type), ())
        except _Unknown:
            if "packed_" in str(element_type):
                self.refuse_import(
                    f"{context}: the layout of the packed element type {element_type} cannot be determined"
                )
            return None
        parts = list(_walk(node))
        if not any(part.packed for part in parts):
            return None
        if any(part.form == "matrix" for part in parts):
            self.refuse_import(
                f"{context}: a buffer element that holds packed vectors and matrices has no CrossGL buffer layout"
            )
            return None
        metal = _layout(node, "metal")
        if metal == _layout(node, "hlsl"):
            return HLSL_STRUCTURED
        if metal != _layout(node, "std430"):
            self.refuse_import(
                f"{context}: the Metal layout of {element_type} matches neither the HLSL structured-buffer nor "
                "the std430 layout of its CrossGL type"
            )
        return None

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
            return _Type("matrix", scalar=matrix.group(1))
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
