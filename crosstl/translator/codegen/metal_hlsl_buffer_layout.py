"""Metal declarations that reproduce HLSL constant and structured buffer layouts.

Metal aligns vectors to their full size, stores ``bool`` in one byte and pads
arrays of scalars to the scalar size, so a struct copied from HLSL does not
match the bytes the host writes. For every constant buffer and struct that an
HLSL buffer reaches, this planner emits a Metal struct whose members sit at the
exact HLSL offsets:

* explicit ``char`` padding moves a member to its HLSL offset;
* vectors use ``packed_*`` storage, which has scalar alignment;
* ``bool`` uses four-byte ``uint`` storage; reads convert back to ``bool``;
* constant-buffer arrays of scalars or vectors use four-component storage so
  each element occupies one 16-byte register; element reads select the
  original components;
* constant-buffer matrices keep the native type when its columns match the
  HLSL registers (``floatCx4``). Every other float, integer or boolean matrix
  is stored as its register words, ``struct { T v[...]; }`` with four words
  per register and none after the last vector, so the next member may pack
  into that register as in HLSL; a load helper builds the matrix on reads and
  a store helper splits it on writes.

A layout that Metal cannot express raises :class:`HlslLayoutError` instead of
silently reading other bytes.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HLSL_STRUCTURED_LAYOUT,
    HlslLayoutError,
    align_to,
    array_type,
    cbuffer_array_stride,
    matrix_type,
    scalar_type,
    struct_placements,
    struct_type,
    vector_type,
)

METAL_SCALAR_STORAGE = {
    "bool": ("uint", 4),
    "int": ("int", 4),
    "uint": ("uint", 4),
    "float": ("float", 4),
    "double": ("double", 8),
    "long": ("long", 8),
    "ulong": ("ulong", 8),
}

METAL_ELEMENT_SUFFIXES = {1: ".x", 2: ".xy", 3: ".xyz", 4: ""}

# The Metal type map spells 64-bit integer scalars with their fixed-width
# names; the layout tables use the Metal ``long``/``ulong`` keywords, which
# name the same types.
METAL_SCALAR_ALIASES = {"int64_t": "long", "uint64_t": "ulong"}


@dataclass
class MetalLayoutMember:
    """One member of a Metal struct that reproduces an HLSL layout."""

    name: str
    offset: int
    storage_type: str
    array_count: Optional[int] = None
    size: int = 0
    alignment: int = 1
    # Conversion applied when the member (or, for arrays, an element) is read.
    read_conversion: Optional[str] = None
    # Conversion applied to a logical value before it is stored.
    write_conversion: Optional[str] = None
    # Components selected after indexing an array whose elements were widened.
    element_suffix: str = ""
    nested_struct: Optional[str] = None

    @property
    def converted(self):
        return bool(
            self.read_conversion
            or self.write_conversion
            or self.element_suffix
            or self.storage_type.startswith("packed_")
        )


@dataclass
class MetalLayoutPlan:
    """A Metal struct definition that matches one HLSL layout."""

    name: str
    layout: str
    members: List[MetalLayoutMember] = field(default_factory=list)
    padding: Dict[str, int] = field(default_factory=dict)
    size: int = 0
    alignment: int = 1

    def member(self, name):
        for member in self.members:
            if member.name == name:
                return member
        return None

    def declaration_lines(self, padding_name):
        lines = []
        for index, member in enumerate(self.members):
            padding = self.padding.get(member.name, 0)
            if padding:
                lines.append(f"char {padding_name(index)}[{padding}];")
            suffix = f"[{member.array_count}]" if member.array_count is not None else ""
            lines.append(f"{member.storage_type} {member.name}{suffix};")
        return lines


class MetalHlslBufferLayouts:
    """Plan Metal storage for the HLSL buffers of one program."""

    def __init__(self, type_resolver):
        # ``type_resolver(type_name)`` returns ``("struct", name, members)`` for a
        # struct, where members are ``(member_name, raw_type)`` pairs, or the
        # Metal spelling of a value type.
        self.type_resolver = type_resolver
        self.struct_plans: Dict[str, MetalLayoutPlan] = {}
        self.struct_layouts: Dict[str, str] = {}
        # Definitions of the matrix storage structs and helpers, by struct name.
        self.matrix_storages: Dict[str, str] = {}

    def storage_definitions(self, plan):
        """Definitions of the matrix storage that the members of ``plan`` use.

        Each definition has an include guard, so every struct that needs one
        may emit it.
        """
        names = []
        for member in plan.members:
            if member.storage_type in self.matrix_storages:
                if member.storage_type not in names:
                    names.append(member.storage_type)
        return "".join(self.matrix_storages[name] for name in names)

    # Type descriptions -------------------------------------------------
    def layout_type(self, raw_type, stack=()):
        raw_type = str(raw_type).strip()
        array_match = re.fullmatch(r"(.+?)\s*\[\s*(\d+)\s*\]", raw_type)
        if array_match is not None:
            element = self.layout_type(array_match.group(1), stack)
            if element.kind == "array":
                raise HlslLayoutError(
                    "multidimensional arrays in HLSL buffers are not supported"
                )
            return array_type(element, int(array_match.group(2)))
        if raw_type.endswith("[]"):
            raise HlslLayoutError("unsized arrays in HLSL buffers are not supported")

        resolved = self.type_resolver(raw_type)
        if isinstance(resolved, tuple) and resolved and resolved[0] == "struct":
            _kind, name, members = resolved
            if name in stack:
                raise HlslLayoutError(f"recursive struct '{name}' in an HLSL buffer")
            return struct_type(
                name,
                [
                    (member_name, self.layout_type(member_type, (*stack, name)))
                    for member_name, member_type in members
                ],
            )

        metal_type = str(resolved)
        if metal_type in METAL_SCALAR_ALIASES:
            return scalar_type(METAL_SCALAR_ALIASES[metal_type])
        scalar_match = re.fullmatch(
            r"(bool|int|uint|float|double|long|ulong)([234])?", metal_type
        )
        if scalar_match is not None:
            scalar, width = scalar_match.groups()
            if width is None:
                return scalar_type(scalar)
            return vector_type(scalar, int(width))
        # Metal ``floatCxR`` and ``crosstl_intCxR`` hold C column vectors of R
        # components, which are the in-memory vectors of the HLSL matrix.
        matrix_match = re.fullmatch(
            r"(?:crosstl_)?(float|double|int|uint|bool)([234])x([234])", metal_type
        )
        if matrix_match is not None and (
            metal_type.startswith("crosstl_")
            == (matrix_match.group(1) in ("int", "uint", "bool"))
        ):
            scalar, columns, rows = matrix_match.groups()
            return matrix_type(scalar, int(columns), int(rows))
        raise HlslLayoutError(f"type '{raw_type}' has no HLSL buffer layout")

    # Struct plans -------------------------------------------------------
    def require_struct(self, layout_type, layout):
        """Plan the struct ``layout_type`` for ``layout`` (and nested structs)."""
        name = layout_type.struct_name
        existing = self.struct_layouts.get(name)
        if existing is not None and existing != layout:
            existing_plan = self.struct_plans[name]
            candidate = self.plan_members(name, layout_type, layout)
            if [
                (m.name, m.offset, m.storage_type, m.array_count, m.element_suffix)
                for m in existing_plan.members
            ] != [
                (m.name, m.offset, m.storage_type, m.array_count, m.element_suffix)
                for m in candidate.members
            ]:
                raise HlslLayoutError(
                    f"struct '{name}' is used in HLSL constant and structured "
                    "buffers with different layouts"
                )
            return existing_plan
        if existing is not None:
            return self.struct_plans[name]
        self.struct_layouts[name] = layout
        plan = self.plan_members(name, layout_type, layout)
        self.struct_plans[name] = plan
        return plan

    def plan_members(self, name, layout_type, layout, root=False):
        placements, _hlsl_size = struct_placements(layout_type, layout)
        plan = MetalLayoutPlan(name=name, layout=layout)
        cursor = 0
        max_alignment = 1
        for placement in placements:
            member = self.member_storage(placement, layout)
            if member.offset % member.alignment:
                raise HlslLayoutError(
                    f"member '{placement.name}' of '{name}' at HLSL offset "
                    f"{member.offset} cannot use Metal storage '{member.storage_type}'"
                )
            if member.offset < cursor:
                raise HlslLayoutError(
                    f"member '{placement.name}' of '{name}' at HLSL offset "
                    f"{member.offset} overlaps the Metal storage of the previous member"
                )
            if member.offset > cursor:
                plan.padding[member.name] = member.offset - cursor
            plan.members.append(member)
            cursor = member.offset + member.size
            max_alignment = max(max_alignment, member.alignment)
        plan.alignment = max_alignment
        plan.size = align_to(cursor, max_alignment)
        self.use_natural_vector_storage(plan, cursor, root)
        return plan

    @staticmethod
    def use_natural_vector_storage(plan, end, root):
        """Replace ``packed_*`` two- and four-component vectors by natural types.

        Those vectors have the same size either way, so a naturally aligned
        offset keeps every member, and the explicit padding before it, in place.
        A nested struct keeps its alignment, which determines its offset and
        array stride in the enclosing layout; the root of a constant buffer may
        raise its alignment while its size stays the same.
        """
        for member in plan.members:
            match = re.fullmatch(r"packed_([a-z]+)([24])", member.storage_type)
            if match is None:
                continue
            scalar, width = match.group(1), int(match.group(2))
            scalar_size = {"double": 8, "long": 8, "ulong": 8}.get(scalar, 4)
            alignment = scalar_size * width
            if member.offset % alignment:
                continue
            if alignment > plan.alignment and (
                not root or align_to(end, alignment) != plan.size
            ):
                continue
            member.storage_type = f"{scalar}{width}"
            member.alignment = alignment
            plan.alignment = max(plan.alignment, alignment)

    def member_storage(self, placement, layout):
        layout_type = placement.layout_type
        offset = placement.offset
        if layout_type.kind == "array":
            return self.array_member_storage(placement, layout)
        storage_type, size, alignment, read, write, nested = self.value_storage(
            layout_type, layout
        )
        return MetalLayoutMember(
            name=placement.name,
            offset=offset,
            storage_type=storage_type,
            size=size,
            alignment=alignment,
            read_conversion=read,
            write_conversion=write,
            nested_struct=nested,
        )

    def value_storage(self, layout_type, layout):
        """``(type, size, alignment, read, write, nested_struct)`` for one value."""
        kind = layout_type.kind
        if kind == "scalar":
            storage, size = METAL_SCALAR_STORAGE[layout_type.scalar]
            read = "bool" if layout_type.scalar == "bool" else None
            return storage, size, size, read, None, None
        if kind == "vector":
            storage, size = METAL_SCALAR_STORAGE[layout_type.scalar]
            width = layout_type.components
            read = write = None
            if layout_type.scalar == "bool":
                read = f"bool{width}"
                write = f"uint{width}"
            return f"packed_{storage}{width}", size * width, size, read, write, None
        if kind == "matrix":
            if layout != HLSL_CBUFFER_LAYOUT:
                raise HlslLayoutError(
                    "matrices in HLSL structured buffers have no Metal layout"
                )
            return self.matrix_storage(layout_type, padded=False)
        if kind == "struct":
            plan = self.require_struct(layout_type, layout)
            return plan.name, plan.size, plan.alignment, None, None, plan.name
        raise HlslLayoutError(f"unsupported HLSL buffer value kind '{kind}'")

    def array_member_storage(self, placement, layout):
        layout_type = placement.layout_type
        element = layout_type.element
        count = layout_type.count
        if layout == HLSL_CBUFFER_LAYOUT and element.kind in ("scalar", "vector"):
            storage, size = METAL_SCALAR_STORAGE[element.scalar]
            if size != 4:
                raise HlslLayoutError(
                    "64-bit scalar arrays in HLSL constant buffers are not supported"
                )
            width = element.components if element.kind == "vector" else 1
            read = None
            if element.scalar == "bool":
                read = "bool" if width == 1 else f"bool{width}"
            return MetalLayoutMember(
                name=placement.name,
                offset=placement.offset,
                storage_type=f"{storage}4",
                array_count=count,
                size=16 * count,
                alignment=16,
                read_conversion=read,
                element_suffix=METAL_ELEMENT_SUFFIXES[width],
            )
        if layout == HLSL_CBUFFER_LAYOUT and element.kind == "matrix":
            storage = self.matrix_storage(element, padded=True)
        else:
            storage = self.value_storage(element, layout)
        storage_type, size, alignment, read, write, nested = storage
        if layout == HLSL_CBUFFER_LAYOUT:
            stride = cbuffer_array_stride(layout_type)
            if size != stride:
                raise HlslLayoutError(
                    f"array '{placement.name}' has a {stride}-byte HLSL element "
                    f"stride but Metal element storage of {size} bytes"
                )
        return MetalLayoutMember(
            name=placement.name,
            offset=placement.offset,
            storage_type=storage_type,
            array_count=count,
            size=size * count,
            alignment=alignment,
            read_conversion=read,
            write_conversion=write,
            nested_struct=nested,
        )

    def matrix_storage(self, layout_type, padded):
        """Storage of a constant-buffer matrix, as returned by :meth:`value_storage`.

        ``padded`` storage fills the last register too, as array elements do.
        """
        scalar = layout_type.scalar
        columns, rows = layout_type.registers, layout_type.components
        if scalar == "double":
            raise HlslLayoutError("Metal has no double-precision matrices")
        if scalar == "float" and rows == 4:
            return f"float{columns}x4", 16 * columns, 16, None, None, None
        name = f"crosstl_hlsl_cbuffer_{scalar}{columns}x{rows}"
        if padded:
            name += "_padded"
        words = 4 * columns if padded else 4 * (columns - 1) + rows
        if name not in self.matrix_storages:
            self.matrix_storages[name] = metal_matrix_storage_definition(
                name, scalar, columns, rows, words
            )
        return name, 4 * words, 4, f"{name}_load", f"{name}_store", None

    def plan_cbuffer(self, name, members):
        """Plan a constant buffer declared with HLSL packing."""
        layout_type = struct_type(
            name,
            [
                (member_name, self.layout_type(member_type, (name,)))
                for member_name, member_type in members
            ],
        )
        return self.plan_members(name, layout_type, HLSL_CBUFFER_LAYOUT, root=True)

    def plan_structured_element(self, raw_type):
        """Storage of one structured-buffer element.

        Returns ``(storage_type, read_conversion, write_conversion)``.
        """
        layout_type = self.layout_type(raw_type)
        if layout_type.kind == "array":
            raise HlslLayoutError("structured buffers of arrays are not supported")
        storage_type, _size, _alignment, read, write, _nested = self.value_storage(
            layout_type, HLSL_STRUCTURED_LAYOUT
        )
        if layout_type.kind == "vector" and layout_type.components in (2, 4):
            # The element stride equals the vector size either way, and buffer
            # bases are aligned, so the natural vector type has the same layout.
            storage_type = storage_type[len("packed_") :]
        return storage_type, read, write


def metal_matrix_storage_definition(name, scalar, columns, rows, words):
    """A register-word struct for one matrix shape with its load and store helpers."""
    logical = (
        f"float{columns}x{rows}"
        if scalar == "float"
        else f"crosstl_{scalar}{columns}x{rows}"
    )
    word = "uint" if scalar == "bool" else scalar
    guard = name.upper()
    loads = []
    stores = []
    for column in range(columns):
        indices = [4 * column + row for row in range(rows)]
        values = ", ".join(f"m.v[{index}]" for index in indices)
        if scalar == "bool":
            vector = f"bool{rows}(uint{rows}({values}))"
        else:
            vector = f"{scalar}{rows}({values})"
        loads.append(f"    result[{column}] = {vector};\n")
        for row, index in enumerate(indices):
            value = f"value[{column}][{row}]"
            if scalar == "bool":
                value = f"uint({value})"
            stores.append(f"    m.v[{index}] = {value};\n")
    return (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        f"struct {name} {{\n    {word} v[{words}];\n}};\n"
        f"inline {logical} {name}_load({name} m) {{\n"
        f"    {logical} result;\n"
        f"{''.join(loads)}"
        "    return result;\n"
        "}\n"
        f"inline {name} {name}_store({logical} value) {{\n"
        f"    {name} m = {{}};\n"
        f"{''.join(stores)}"
        "    return m;\n"
        "}\n"
        "#endif\n"
    )
