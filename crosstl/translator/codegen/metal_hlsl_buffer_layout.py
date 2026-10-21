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
  original components. The last element of an HLSL array is not padded, so
  when the next member packs into its register the array is stored as its
  register words instead, ``struct { T v[...]; }`` with none after the last
  element, and an element helper reads ``v[4 * i]`` onwards;
* constant-buffer matrices keep the native type when its columns match the
  HLSL registers (``floatCx4``). Every other float, integer or boolean matrix
  is stored as its register words, ``struct { T v[...]; }`` with four words
  per register and none after the last vector, so the next member may pack
  into that register as in HLSL; a load helper builds the matrix on reads and
  a store helper splits it on writes;
* a ``row_major`` matrix holds one row per register. When its rows have four
  float components they are the columns of a native ``floatRx4``, which is
  read through ``transpose``; other shapes use register words whose load
  helper reads the rows;
* members with ``packoffset`` sit at the bytes it names, in offset order.

GLSL uniform blocks marked ``@ buffer_layout(std140)`` are planned the same way
at their std140 offsets (:mod:`std140_buffer_layout`). Their arrays and
matrices pad every element and column to 16 bytes, so a ``floatCx3`` matrix
keeps its native type too, and a nested struct ends with explicit padding up
to its std140 size, which is its array stride.

A layout that Metal cannot express raises :class:`HlslLayoutError` instead of
silently reading other bytes.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .array_utils import fold_array_type_counts
from .glsl_block_layout import (
    GLSL_BLOCK_LAYOUTS,
    SCALAR_BLOCK_LAYOUT,
    STD140_LAYOUT,
    STD430_LAYOUT,
    block_matrix_vector_stride,
    block_size,
    layout_array_stride,
    layout_struct_placements,
)
from .hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HLSL_STRUCTURED_LAYOUT,
    HlslLayoutError,
    align_to,
    array_type,
    oriented_matrix_type,
    scalar_type,
    struct_type,
    vector_type,
)

# Layouts whose arrays put every element in its own 16-byte register and whose
# matrices put every column (or row) in one.
REGISTER_LAYOUTS = (HLSL_CBUFFER_LAYOUT, STD140_LAYOUT)
LAYOUT_LABELS = {
    HLSL_CBUFFER_LAYOUT: "HLSL",
    HLSL_STRUCTURED_LAYOUT: "HLSL",
    STD140_LAYOUT: "std140",
    STD430_LAYOUT: "std430",
    SCALAR_BLOCK_LAYOUT: "scalar",
}

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
    # Helper that reads element ``i`` of an array held as register words: ``helper(storage, i)``.
    element_load: Optional[str] = None

    @property
    def converted(self):
        return bool(
            self.read_conversion
            or self.write_conversion
            or self.element_suffix
            or self.element_load
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
    # Bytes after the last member that belong to the struct (std140 rounds a
    # struct's size up to its alignment).
    tail_padding: int = 0

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
        if self.tail_padding:
            lines.append(
                f"char {padding_name(len(self.members))}[{self.tail_padding}];"
            )
        return lines


class MetalHlslBufferLayouts:
    """Plan Metal storage for the HLSL buffers of one program."""

    def __init__(self, type_resolver, int_constants=None):
        # ``type_resolver(type_name)`` returns ``("struct", name, members)`` for a
        # struct, where members are ``(member_name, raw_type)`` pairs, or the
        # Metal spelling of a value type.
        self.type_resolver = type_resolver
        # Named integer constants that array counts such as ``float4[N * 3]`` use.
        self.int_constants: Dict[str, int] = dict(int_constants or {})
        self.struct_plans: Dict[str, MetalLayoutPlan] = {}
        self.struct_layouts: Dict[str, str] = {}
        # Definitions of the register-word storage structs of matrices and
        # trimmed arrays, with their helpers, by struct name.
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
    def layout_type(self, raw_type, stack=(), row_major=False):
        """The layout type of ``raw_type``; ``row_major`` orients a matrix (or the matrices of an array)."""
        raw_type = fold_array_type_counts(str(raw_type).strip(), self.int_constants)
        array_match = re.fullmatch(r"(.+?)\s*\[\s*(\d+)\s*\]", raw_type)
        if array_match is not None:
            element = self.layout_type(array_match.group(1), stack, row_major)
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
            # Members are ``(name, raw_type)`` or ``(name, raw_type, row_major)``.
            return struct_type(
                name,
                [
                    (
                        member[0],
                        self.layout_type(
                            member[1], (*stack, name), len(member) > 2 and member[2]
                        ),
                    )
                    for member in members
                ],
            )

        metal_type = str(resolved)
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
            return oriented_matrix_type(scalar, int(columns), int(rows), row_major)
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
        placements, _source_size = layout_struct_placements(layout_type, layout, root)
        label = LAYOUT_LABELS[layout]
        # Metal lays a struct out in member order; members placed by packoffset
        # are declared in the order of their offsets.
        placements = sorted(placements, key=lambda placement: placement.offset)
        plan = MetalLayoutPlan(name=name, layout=layout)
        cursor = 0
        max_alignment = 1
        for index, placement in enumerate(placements):
            following = (
                placements[index + 1].offset if index + 1 < len(placements) else None
            )
            member = self.member_storage(placement, layout, following)
            if member.offset % member.alignment:
                raise HlslLayoutError(
                    f"member '{placement.name}' of '{name}' at {label} offset "
                    f"{member.offset} cannot use Metal storage '{member.storage_type}'"
                )
            if member.offset < cursor:
                raise HlslLayoutError(
                    f"member '{placement.name}' of '{name}' at {label} offset "
                    f"{member.offset} overlaps the Metal storage of the previous member"
                )
            if member.offset > cursor:
                plan.padding[member.name] = member.offset - cursor
            plan.members.append(member)
            cursor = member.offset + member.size
            max_alignment = max(max_alignment, member.alignment)
        if layout in GLSL_BLOCK_LAYOUTS and not root:
            # The struct's size is where the next member may start (and, but for
            # the scalar layout, its array stride); it is a multiple of the
            # struct's alignment, which no Metal member alignment exceeds.
            source_size = block_size(layout_type, layout)
            if source_size < cursor:
                raise HlslLayoutError(
                    f"struct '{name}' needs {cursor} bytes of Metal storage but has a "
                    f"{label} size of {source_size}"
                )
            plan.tail_padding = source_size - cursor
            cursor = source_size
        plan.alignment = max_alignment
        plan.size = align_to(cursor, max_alignment)
        # std140 and std430 align every vector at least as Metal does, so a
        # natural vector type never moves a member or changes a struct's size.
        self.use_natural_vector_storage(
            plan, cursor, root or layout in (STD140_LAYOUT, STD430_LAYOUT)
        )
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
            if member.read_conversion == member.storage_type:
                member.read_conversion = None
            member.alignment = alignment
            plan.alignment = max(plan.alignment, alignment)

    def member_storage(self, placement, layout, following=None):
        """Metal storage of one member; ``following`` is the offset of the member after it, if any."""
        layout_type = placement.layout_type
        offset = placement.offset
        if layout_type.kind == "array":
            return self.array_member_storage(placement, layout, following)
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
            # A packed vector converts to its natural type when read: matrix
            # products and the vector built-ins take no packed operands.
            read = f"{storage}{width}"
            write = None
            if layout_type.scalar == "bool":
                read = f"bool{width}"
                write = f"uint{width}"
            return f"packed_{storage}{width}", size * width, size, read, write, None
        if kind == "matrix":
            if layout in (STD430_LAYOUT, SCALAR_BLOCK_LAYOUT):
                return self.block_matrix_storage(layout_type, layout)
            if layout not in REGISTER_LAYOUTS:
                raise HlslLayoutError(
                    "matrices in HLSL structured buffers have no Metal layout"
                )
            # A std140 matrix is an array of its columns (rows): the last one is
            # padded like the others.
            return self.matrix_storage(
                layout_type, padded=layout == STD140_LAYOUT, layout=layout
            )
        if kind == "struct":
            plan = self.require_struct(layout_type, layout)
            return plan.name, plan.size, plan.alignment, None, None, plan.name
        raise HlslLayoutError(f"unsupported HLSL buffer value kind '{kind}'")

    def array_member_storage(self, placement, layout, following=None):
        layout_type = placement.layout_type
        element = layout_type.element
        count = layout_type.count
        label = LAYOUT_LABELS[layout]
        if layout in REGISTER_LAYOUTS and element.kind in ("scalar", "vector"):
            storage, size = METAL_SCALAR_STORAGE[element.scalar]
            if size != 4:
                raise HlslLayoutError(
                    f"64-bit scalar arrays in {label} buffers are not supported"
                )
            width = element.components if element.kind == "vector" else 1
            if (
                layout == HLSL_CBUFFER_LAYOUT
                and following is not None
                and following < placement.offset + 16 * count
            ):
                return self.trimmed_array_storage(placement, element.scalar, width)
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
        if layout in REGISTER_LAYOUTS and element.kind == "matrix":
            storage = self.matrix_storage(element, padded=True, layout=layout)
        elif layout in (STD430_LAYOUT, SCALAR_BLOCK_LAYOUT) and element.kind in (
            "scalar",
            "vector",
        ):
            storage = self.block_array_element_storage(layout_type, layout)
        else:
            storage = self.value_storage(element, layout)
        storage_type, size, alignment, read, write, nested = storage
        if layout in REGISTER_LAYOUTS or layout in GLSL_BLOCK_LAYOUTS:
            stride = layout_array_stride(layout_type, layout)
            if size != stride:
                raise HlslLayoutError(
                    f"array '{placement.name}' has a {stride}-byte {label} element "
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

    def trimmed_array_storage(self, placement, scalar, width):
        """Register words of a constant-buffer array whose last register the next member shares.

        Element ``i`` starts at word ``4 * i`` and the storage ends after the
        last element's ``width`` words, where HLSL packs the next member.
        """
        count = placement.layout_type.count
        logical = scalar if width == 1 else f"{scalar}{width}"
        name = f"crosstl_hlsl_cbuffer_{logical}_array{count}"
        words = 4 * (count - 1) + width
        if name not in self.matrix_storages:
            self.matrix_storages[name] = metal_trimmed_array_storage_definition(
                name, scalar, width, words
            )
        return MetalLayoutMember(
            name=placement.name,
            offset=placement.offset,
            storage_type=name,
            size=4 * words,
            alignment=4,
            element_load=f"{name}_element",
        )

    def matrix_storage(self, layout_type, padded, layout=HLSL_CBUFFER_LAYOUT):
        """Storage of a constant-buffer matrix, as returned by :meth:`value_storage`.

        ``padded`` storage fills the last register too, as array elements do.
        """
        scalar = layout_type.scalar
        registers, components = layout_type.registers, layout_type.components
        row_major = layout_type.row_major
        columns, rows = (
            (components, registers) if row_major else (registers, components)
        )
        if scalar == "double":
            raise HlslLayoutError("Metal has no double-precision matrices")
        if scalar == "float" and components == 4:
            # Each register holds a whole float4: a column, or a row of the transpose.
            native = f"float{registers}x4"
            conversion = "transpose" if row_major else None
            return native, 16 * registers, 16, conversion, conversion, None
        if scalar == "float" and components == 3 and layout == STD140_LAYOUT:
            # Metal's float3 columns take 16 bytes, as std140 columns (rows) do.
            native = f"float{registers}x3"
            conversion = "transpose" if row_major else None
            return native, 16 * registers, 16, conversion, conversion, None
        name = f"crosstl_hlsl_cbuffer_{scalar}{columns}x{rows}"
        if row_major:
            name += "_row_major"
        if padded:
            name += "_padded"
        words = 4 * registers if padded else 4 * (registers - 1) + components
        if name not in self.matrix_storages:
            self.matrix_storages[name] = metal_matrix_storage_definition(
                name, scalar, columns, rows, words, row_major
            )
        return name, 4 * words, 4, f"{name}_load", f"{name}_store", None

    def block_array_element_storage(self, layout_type, layout):
        """Storage of one element of a std430 or scalar array of scalars or vectors.

        The elements are packed, so the stride is the element size, except for
        the three-component vectors of std430, whose 16-byte stride is that of
        Metal's natural ``float3``.
        """
        element = layout_type.element
        storage = self.value_storage(element, layout)
        stride = layout_array_stride(layout_type, layout)
        if storage[1] == stride:
            return storage
        if element.kind == "vector" and element.components == 3:
            scalar, size = METAL_SCALAR_STORAGE[element.scalar]
            if 4 * size == stride and size == 4:
                read = "bool3" if element.scalar == "bool" else None
                write = "uint3" if element.scalar == "bool" else None
                return f"{scalar}3", stride, stride, read, write, None
        raise HlslLayoutError(
            f"{LAYOUT_LABELS[layout]} arrays of {element.scalar} values with a "
            f"{stride}-byte stride have no Metal layout"
        )

    def block_matrix_storage(self, layout_type, layout):
        """Storage of a std430 or scalar matrix, as returned by :meth:`value_storage`.

        A std430 float matrix stores its columns (rows, for ``row_major``) at
        the stride of Metal's native matrix of that column type; the scalar
        layout packs them, so its words are loaded and stored by helpers.
        """
        scalar = layout_type.scalar
        if scalar != "float":
            raise HlslLayoutError(f"Metal has no {scalar} matrices")
        registers, components = layout_type.registers, layout_type.components
        row_major = layout_type.row_major
        stride = block_matrix_vector_stride(layout_type, layout)
        if layout == STD430_LAYOUT:
            native = f"float{registers}x{components}"
            alignment = 8 if components == 2 else 16
            if stride != alignment:
                raise HlslLayoutError(
                    f"a {stride}-byte std430 matrix stride has no Metal layout"
                )
            conversion = "transpose" if row_major else None
            return native, stride * registers, alignment, conversion, conversion, None
        columns, rows = (
            (components, registers) if row_major else (registers, components)
        )
        name = f"crosstl_scalar_{scalar}{columns}x{rows}"
        if row_major:
            name += "_row_major"
        words = components * registers
        if name not in self.matrix_storages:
            self.matrix_storages[name] = metal_matrix_storage_definition(
                name, scalar, columns, rows, words, row_major, vector_words=components
            )
        return name, 4 * words, 4, f"{name}_load", f"{name}_store", None

    def plan_cbuffer(self, name, members, layout=HLSL_CBUFFER_LAYOUT):
        """Plan a constant buffer declared with HLSL packing or as a GLSL std140, std430 or scalar block.

        ``members`` are ``(name, raw_type, row_major, offset[, alignment])``,
        where ``offset`` is the byte offset of a ``packoffset`` or ``offset``
        qualifier and ``alignment`` that of an ``align`` qualifier, or ``None``.
        """
        layout_type = struct_type(
            name,
            [
                (member[0], self.layout_type(member[1], (name,), member[2]))
                for member in members
            ],
            [member[3] for member in members],
            [member[4] if len(member) > 4 else None for member in members],
        )
        return self.plan_members(name, layout_type, layout, root=True)

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
            if read == storage_type:
                read = None
        return storage_type, read, write


def metal_matrix_storage_definition(
    name, scalar, columns, rows, words, row_major=False, vector_words=4
):
    """A register-word struct for one matrix shape with its load and store helpers.

    The registers hold the columns of the matrix, or its rows when ``row_major``;
    consecutive registers start ``vector_words`` words apart.
    """
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
        if row_major:
            indices = [vector_words * row + column for row in range(rows)]
        else:
            indices = [vector_words * column + row for row in range(rows)]
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


def metal_trimmed_array_storage_definition(name, scalar, width, words):
    """A register-word struct for a constant-buffer array of ``width``-component values and its element helper."""
    logical = scalar if width == 1 else f"{scalar}{width}"
    word = "uint" if scalar == "bool" else scalar
    values = ", ".join(
        "a.v[4 * index]" if component == 0 else f"a.v[4 * index + {component}]"
        for component in range(width)
    )
    if scalar == "bool":
        value = (
            f"bool({values})" if width == 1 else f"bool{width}(uint{width}({values}))"
        )
    else:
        value = values if width == 1 else f"{logical}({values})"
    guard = name.upper()
    return (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        f"struct {name} {{\n    {word} v[{words}];\n}};\n"
        f"inline {logical} {name}_element({name} a, int index) {{\n"
        f"    return {value};\n"
        "}\n"
        "#endif\n"
    )
