"""OpenGL uniform blocks that reproduce HLSL constant-buffer and GLSL std430 or scalar layouts.

HLSL packs a constant buffer into 16-byte registers (see
``hlsl_buffer_layout``): a scalar or vector only moves to the next register
when it would straddle one, and an array, matrix or struct ends after its last
element, so the next member may use the rest of that register. GLSL std140
aligns two-component vectors to 8 bytes and wider ones to 16, and pads arrays,
matrices and structs to 16 bytes: ``float s[2]; float t;`` puts ``t`` at byte
36 in HLSL and at byte 48 in std140, and ``int c; int2 v;`` puts ``v`` at byte
4 in HLSL and at byte 8 in std140.

A uniform block marked ``@ buffer_layout(hlsl_cbuffer)`` keeps its std140
declaration when every scalar it holds has the same offset under both rules.
Otherwise the block declares its registers as one ``uvec4`` array, every
member becomes a private global variable of its own type and name, and a
loader that each stage entry calls before anything else assigns the members
from the bits at their HLSL offsets (``uintBitsToFloat``, ``packDouble2x32``
and so on). Member reads are unchanged, and every value is read from the bytes
the host wrote for it. A member type without an HLSL constant-buffer layout
raises :class:`OpenGLHlslConstantBufferError`.

A GLSL block marked ``@ buffer_layout(std430)`` or ``scalar`` (a Vulkan push
constant, or a block written for GL_EXT_scalar_block_layout) is declared the
same way: OpenGL uniform blocks have only the std140 layout, which moves every
``float[]`` element to its own 16 bytes. So is a std140 block whose ``offset``
or ``align`` qualifiers move its members. Arrays whose stride is not a whole
number of registers are read word by word.

An MSL constant buffer marked ``@ buffer_layout(metal)`` is declared the same
way: its ``float3`` members take 16 bytes and its ``@ packed`` vectors 12, so
``float3 a; float b;`` puts ``b`` at byte 16 where std140 puts it at 12.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ..ast import ArrayNode, FunctionNode
from .array_utils import (
    evaluate_literal_int_expression,
    fold_array_type_counts,
    format_c_style_array_declaration,
)
from .buffer_member_layout import (
    member_explicit_offset,
    member_is_packed,
    member_is_row_major,
)
from .glsl_block_layout import (
    BLOCK_LAYOUTS,
    block_array_stride,
    block_matrix_vector_stride,
    block_struct_placements,
    declared_buffer_layout,
    member_explicit_alignment,
    packed_layout_type,
)
from .hlsl_buffer_layout import (
    CBUFFER_REGISTER_SIZE,
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    align_to,
    array_type,
    cbuffer_array_stride,
    cbuffer_matrix_vector_stride,
    cbuffer_struct_placements,
    contains_row_major_matrix,
    oriented_matrix_type,
    scalar_type,
    struct_type,
    vector_type,
)

COMPONENTS = "xyzw"
WORD_SIZE = 4
# GLSL spellings of the value types (the result of ``map_type``).
GLSL_SCALARS = {
    "bool": "bool",
    "int": "int",
    "uint": "uint",
    "float": "float",
    "double": "double",
    "int64_t": "long",
    "uint64_t": "ulong",
}
VECTOR_PATTERN = re.compile(r"(?P<prefix>i64|u64|[iubd])?vec(?P<width>[234])")
VECTOR_SCALARS = {
    None: "float",
    "i": "int",
    "u": "uint",
    "b": "bool",
    "d": "double",
    "i64": "long",
    "u64": "ulong",
}
# GLSL ``matCxR`` holds C column vectors of R components, and so do the
# structs that emulate integer and boolean matrices (see ``glsl_hlsl_matrix``).
MATRIX_PATTERN = re.compile(
    r"(?:crossgl_(?P<emulated>[iub]))?(?P<double>d?)mat(?P<columns>[234])"
    r"(?:x(?P<rows>[234]))?"
)
MATRIX_SCALARS = {"i": "int", "u": "uint", "b": "bool"}
ARRAY_PATTERN = re.compile(r"(?P<element>.+?)\s*\[\s*(?P<count>\d+)\s*\]")


class OpenGLHlslConstantBufferError(ValueError):
    """An HLSL constant buffer whose layout OpenGL storage cannot reproduce."""

    def __init__(self, message, *, cbuffer=None, reason=None):
        super().__init__(message)
        self.cbuffer = cbuffer
        self.reason = reason


def std140_size_align(layout_type):
    """``(size, alignment)`` of the GLSL declaration of a value under std140."""
    kind = layout_type.kind
    if kind == "scalar":
        return layout_type.scalar_size, layout_type.scalar_size
    if kind == "vector":
        size = layout_type.scalar_size * layout_type.components
        factor = 2 if layout_type.components == 2 else 4
        return size, layout_type.scalar_size * factor
    if kind == "matrix":
        stride = std140_matrix_column_stride(layout_type)
        return stride * layout_type.registers, stride
    if kind == "array":
        stride, alignment = std140_array_stride_align(layout_type)
        return stride * layout_type.count, alignment
    if kind == "struct":
        _offsets, size, alignment = std140_struct_offsets(layout_type)
        return size, alignment
    raise HlslLayoutError(f"unsupported HLSL buffer value kind '{kind}'")


def std140_matrix_column_stride(layout_type):
    # A matrix is an array of its column vectors; emulated integer matrices
    # are structs holding exactly that array.
    _size, alignment = std140_size_align(
        vector_type(layout_type.scalar, layout_type.components)
    )
    return align_to(alignment, CBUFFER_REGISTER_SIZE)


def std140_array_stride_align(layout_type):
    size, alignment = std140_size_align(layout_type.element)
    alignment = align_to(alignment, CBUFFER_REGISTER_SIZE)
    return align_to(size, alignment), alignment


def std140_struct_offsets(layout_type):
    offsets = []
    cursor, max_alignment = 0, CBUFFER_REGISTER_SIZE
    for _name, member_type in layout_type.members:
        size, alignment = std140_size_align(member_type)
        cursor = align_to(cursor, alignment)
        offsets.append(cursor)
        cursor += size
        max_alignment = max(max_alignment, alignment)
    alignment = align_to(max_alignment, CBUFFER_REGISTER_SIZE)
    return offsets, align_to(cursor, alignment), alignment


def std140_scalar_offsets(layout_type, offset=0):
    """Byte offsets of every scalar of a value under std140, in order."""
    kind = layout_type.kind
    if kind == "scalar":
        return [offset]
    if kind == "vector":
        size = layout_type.scalar_size
        return [offset + k * size for k in range(layout_type.components)]
    if kind == "matrix":
        stride = std140_matrix_column_stride(layout_type)
        column = vector_type(layout_type.scalar, layout_type.components)
        return [
            item
            for index in range(layout_type.registers)
            for item in std140_scalar_offsets(column, offset + index * stride)
        ]
    if kind == "array":
        stride, _alignment = std140_array_stride_align(layout_type)
        return [
            item
            for index in range(layout_type.count)
            for item in std140_scalar_offsets(
                layout_type.element, offset + index * stride
            )
        ]
    offsets, _size, _alignment = std140_struct_offsets(layout_type)
    return [
        item
        for (_name, member_type), start in zip(layout_type.members, offsets)
        for item in std140_scalar_offsets(member_type, offset + start)
    ]


def layout_matrix_vector_stride(layout_type, layout):
    if layout in BLOCK_LAYOUTS:
        return block_matrix_vector_stride(layout_type, layout)
    return cbuffer_matrix_vector_stride(layout_type)


def layout_array_stride(layout_type, layout):
    if layout in BLOCK_LAYOUTS:
        return block_array_stride(layout_type, layout)
    return cbuffer_array_stride(layout_type)


def layout_member_placements(layout_type, layout):
    if layout in BLOCK_LAYOUTS:
        placements, _size = block_struct_placements(layout_type, layout)
    else:
        placements, _size = cbuffer_struct_placements(layout_type)
    return placements


def layout_scalar_offsets(layout_type, layout, offset=0):
    """Byte offsets of every scalar of a value under HLSL constant packing or a GLSL block layout."""
    kind = layout_type.kind
    if kind == "scalar":
        return [offset]
    if kind == "vector":
        size = layout_type.scalar_size
        return [offset + k * size for k in range(layout_type.components)]
    if kind == "matrix":
        stride = layout_matrix_vector_stride(layout_type, layout)
        column = vector_type(layout_type.scalar, layout_type.components)
        return [
            item
            for index in range(layout_type.registers)
            for item in layout_scalar_offsets(column, layout, offset + index * stride)
        ]
    if kind == "array":
        stride = layout_array_stride(layout_type, layout)
        return [
            item
            for index in range(layout_type.count)
            for item in layout_scalar_offsets(
                layout_type.element, layout, offset + index * stride
            )
        ]
    return [
        item
        for placement in layout_member_placements(layout_type, layout)
        for item in layout_scalar_offsets(
            placement.layout_type, layout, offset + placement.offset
        )
    ]


@dataclass(frozen=True)
class RegisterLocation:
    """A byte address in the register array: ``16 * sum(terms) + 4 * sum(word_terms) + byte``.

    ``terms`` are loop-dependent register counts, for arrays whose stride is a
    whole number of registers, so they never change the byte within one.
    ``word_terms`` are loop-dependent word counts, for the other arrays.
    """

    terms: Tuple[str, ...] = ()
    byte: int = 0
    word_terms: Tuple[str, ...] = ()

    def shifted(self, delta):
        return RegisterLocation(self.terms, self.byte + delta, self.word_terms)

    def indexed(self, term):
        return RegisterLocation((*self.terms, term), self.byte, self.word_terms)

    def word_indexed(self, term):
        return RegisterLocation(self.terms, self.byte, (*self.word_terms, term))

    def word_index(self):
        """The index of the location's word, counted from the first register."""
        parts = [f"({term}) * 4" for term in self.terms]
        parts.extend(f"({term})" for term in self.word_terms)
        parts.append(str(self.byte // WORD_SIZE))
        return " + ".join(parts)

    def register_and_component(self, registers):
        index_terms = list(self.terms)
        register = self.byte // CBUFFER_REGISTER_SIZE
        if register or not index_terms:
            index_terms.append(str(register))
        component = (self.byte % CBUFFER_REGISTER_SIZE) // WORD_SIZE
        return f"{registers}[{' + '.join(index_terms)}]", component


@dataclass
class RegisterLoader:
    """Statements that assign every member of one block from its registers."""

    registers: str
    index_names: List[str]
    lines: List[str] = field(default_factory=list)
    # The layout the host wrote: HLSL constant packing or a GLSL block layout.
    layout: str = HLSL_CBUFFER_LAYOUT
    # ``uint name(int word)`` reads one word at a computed index; set when used.
    word_reader: str = ""
    uses_word_reader: bool = False


class GLSLHlslConstantBuffers:
    """GLSL code generator support for HLSL constant-buffer layouts."""

    def reset_glsl_hlsl_cbuffers(self):
        self.glsl_hlsl_cbuffer_loaders: List[str] = []

    # Layout types ---------------------------------------------------------

    def glsl_hlsl_cbuffer_member_raw_type(self, member):
        if isinstance(member, ArrayNode):
            element = self.type_name_string(
                getattr(member, "element_type", getattr(member, "vtype", None))
            )
            size = getattr(member, "size", None)
            if not isinstance(size, int) and not str(size).isdigit():
                size = evaluate_literal_int_expression(
                    size, getattr(self, "literal_int_constants", None)
                )
            if size is None or int(str(size)) < 0:
                raise HlslLayoutError(
                    f"array '{member.name}' needs a constant size in an HLSL buffer"
                )
            return f"{element}[{int(str(size))}]"
        member_type = getattr(member, "member_type", None)
        if member_type is None:
            member_type = getattr(member, "vtype", None)
        return self.type_name_string(member_type)

    def glsl_hlsl_cbuffer_layout_type(self, raw_type, stack=(), row_major=False):
        """The layout type of ``raw_type``; ``row_major`` orients a matrix (or the matrices of an array)."""
        raw_type = fold_array_type_counts(
            str(raw_type or "").strip(), getattr(self, "literal_int_constants", None)
        )
        array_match = ARRAY_PATTERN.fullmatch(raw_type)
        if array_match is not None:
            element = self.glsl_hlsl_cbuffer_layout_type(
                array_match.group("element"), stack, row_major
            )
            if element.kind == "array":
                raise HlslLayoutError(
                    "multidimensional arrays in HLSL buffers are not supported"
                )
            return array_type(element, int(array_match.group("count")))
        if raw_type.endswith("]"):
            raise HlslLayoutError(
                f"array type '{raw_type}' needs a constant size in an HLSL buffer"
            )
        struct = self.glsl_struct_node(raw_type)
        if struct is not None:
            return self.glsl_hlsl_cbuffer_struct_layout_type(raw_type, struct, stack)
        mapped = str(self.map_type(raw_type) or "").strip()
        if mapped in GLSL_SCALARS:
            return scalar_type(GLSL_SCALARS[mapped])
        vector = VECTOR_PATTERN.fullmatch(mapped)
        if vector is not None:
            return vector_type(
                VECTOR_SCALARS[vector.group("prefix")], int(vector.group("width"))
            )
        matrix = MATRIX_PATTERN.fullmatch(mapped)
        if matrix is not None and not (
            matrix.group("emulated") and matrix.group("double")
        ):
            columns = int(matrix.group("columns"))
            rows = int(matrix.group("rows") or columns)
            if matrix.group("emulated"):
                scalar = MATRIX_SCALARS[matrix.group("emulated")]
            else:
                scalar = "double" if matrix.group("double") else "float"
            return oriented_matrix_type(scalar, columns, rows, row_major)
        struct = self.glsl_struct_node(mapped)
        if struct is not None:
            return self.glsl_hlsl_cbuffer_struct_layout_type(mapped, struct, stack)
        raise HlslLayoutError(f"type '{raw_type}' has no HLSL constant-buffer layout")

    def glsl_hlsl_cbuffer_member_layout_type(self, node, member):
        layout_type = self.glsl_hlsl_cbuffer_layout_type(
            self.glsl_hlsl_cbuffer_member_raw_type(member),
            (node.name,),
            member_is_row_major(member),
        )
        if member_is_packed(member):
            return packed_layout_type(layout_type, member.name)
        return layout_type

    def glsl_hlsl_cbuffer_struct_layout_type(self, type_name, struct, stack):
        name = self.glsl_source_type_identifier_name(type_name)
        if name in stack:
            raise HlslLayoutError(f"recursive struct '{name}' in an HLSL buffer")
        member_types = self.glsl_struct_member_types(name)
        members = []
        for member in getattr(struct, "members", []) or []:
            member_name = getattr(member, "name", None)
            if isinstance(member, FunctionNode) or member_name not in member_types:
                continue
            member_type = self.glsl_hlsl_cbuffer_layout_type(
                self.type_name_string(member_types[member_name]),
                (*stack, name),
                member_is_row_major(member),
            )
            if member_is_packed(member):
                member_type = packed_layout_type(member_type, member_name)
            members.append((member_name, member_type))
        return struct_type(name, members)

    # Declarations -----------------------------------------------------------

    def glsl_hlsl_cbuffer_block(self, node, layout, block_name):
        """Register storage for ``node`` when std140 differs from its layout, else ``None``.

        ``node`` is an HLSL constant buffer (``@ buffer_layout(hlsl_cbuffer)``)
        or a GLSL block (``std140``, ``std430`` or ``scalar``). Returns the
        uniform block, the private member variables and their loader; stage
        entries call the loader through :meth:`generate_glsl_hlsl_cbuffer_loads`.
        """
        if not hasattr(self, "glsl_hlsl_cbuffer_loaders"):
            self.reset_glsl_hlsl_cbuffers()
        source_layout = declared_buffer_layout(getattr(node, "attributes", None))
        members = [
            member
            for member in getattr(node, "members", []) or []
            if getattr(member, "name", None) and not isinstance(member, FunctionNode)
        ]
        if source_layout is None and any(
            member_explicit_offset(member) is not None for member in members
        ):
            # A block without a declared layout is declared std140 here; its ``@ offset`` members keep their bytes.
            source_layout = "std140"
        if source_layout != HLSL_CBUFFER_LAYOUT and source_layout not in BLOCK_LAYOUTS:
            return None
        # The GLSL block layouts (std140, std430, scalar) are their own labels.
        if source_layout == HLSL_CBUFFER_LAYOUT:
            label = "HLSL"
        elif source_layout == "metal":
            label = "MSL"
        else:
            label = source_layout
        try:
            try:
                placed = self.glsl_hlsl_cbuffer_placements(node, members, source_layout)
            except HlslLayoutError:
                # An array sized by a specialization constant has no literal size. A std140 block without offset
                # or align qualifiers is placed by the same rules as the std140 declaration written here at every
                # array count, so checking it at the constants' default values (for packed members and nested
                # orientation) settles it, and the declaration keeps the constant. Any other block would need its
                # count fixed, so it is refused.
                defaults = getattr(
                    self, "glsl_specialization_default_int_constants", None
                )
                if (
                    source_layout != "std140"
                    or not defaults
                    or defaults == getattr(self, "literal_int_constants", None)
                    or any(
                        member_explicit_offset(member) is not None
                        or member_explicit_alignment(member) is not None
                        for member in members
                    )
                ):
                    raise
                literal_int_constants = self.literal_int_constants
                self.literal_int_constants = defaults
                try:
                    placed = self.glsl_hlsl_cbuffer_placements(
                        node, members, source_layout
                    )
                finally:
                    self.literal_int_constants = literal_int_constants
                if placed is not None:
                    raise
            if placed is None:
                return None
            placements, size = placed
            return self.glsl_hlsl_cbuffer_register_block(
                node, layout, block_name, members, placements, size, source_layout
            )
        except HlslLayoutError as exc:
            raise OpenGLHlslConstantBufferError(
                f"Cannot reproduce the {label} layout of cbuffer '{node.name}' in "
                f"OpenGL: {exc}",
                cbuffer=node.name,
                reason=(
                    "hlsl-cbuffer-layout"
                    if source_layout == HLSL_CBUFFER_LAYOUT
                    else (
                        "metal-constant-layout"
                        if source_layout == "metal"
                        else "glsl-block-layout"
                    )
                ),
            ) from exc

    def glsl_hlsl_cbuffer_placements(self, node, members, source_layout):
        """The ``(placements, size)`` of ``node``'s members under ``source_layout``, or ``None`` when the
        std140 declaration this generator writes already places every scalar there."""
        layout_type = struct_type(
            node.name,
            [
                (
                    member.name,
                    self.glsl_hlsl_cbuffer_member_layout_type(node, member),
                )
                for member in members
            ],
            [member_explicit_offset(member) for member in members],
            [member_explicit_alignment(member) for member in members],
        )
        if source_layout == HLSL_CBUFFER_LAYOUT:
            placements, size = cbuffer_struct_placements(layout_type)
        else:
            placements, size = block_struct_placements(
                layout_type, source_layout, root=True
            )
        # The declaration this generator writes has no offset or align qualifiers.
        std140_offsets, _std140_size, _alignment = std140_struct_offsets(layout_type)
        # GLSL orients a whole block member, not the members of its struct.
        nested_orientation = any(
            member_type.kind != "matrix" and contains_row_major_matrix(member_type)
            for _name, member_type in layout_type.members
        )
        if not nested_orientation and all(
            layout_scalar_offsets(
                placement.layout_type, source_layout, placement.offset
            )
            == std140_scalar_offsets(placement.layout_type, std140_offset)
            for placement, std140_offset in zip(placements, std140_offsets)
        ):
            return None
        return placements, size

    def glsl_hlsl_cbuffer_register_block(
        self,
        node,
        layout,
        block_name,
        members,
        placements,
        size,
        source_layout=HLSL_CBUFFER_LAYOUT,
    ):
        identity = ("hlsl-cbuffer", id(node))
        registers = self.glsl_generated_module_identifier(
            (*identity, "registers"), "crossgl_cbuffer_registers"
        )
        loader_name = self.glsl_generated_module_identifier(
            (*identity, "loader"), f"crossgl_load_{block_name}"
        )
        register_count = max(1, align_to(size, CBUFFER_REGISTER_SIZE) // 16)
        loader = RegisterLoader(registers, [], layout=source_layout)
        loader.word_reader = self.glsl_generated_module_identifier(
            (*identity, "word"), f"crossgl_{block_name}_word"
        )
        declarations = ""
        for member, placement in zip(members, placements):
            name = self.glsl_global_identifier_name(member.name)
            declaration = format_c_style_array_declaration(
                self.map_type(self.glsl_hlsl_cbuffer_member_raw_type(member)), name
            )
            declarations += f"{declaration};\n"
            self.glsl_hlsl_cbuffer_load_value(
                loader,
                placement.layout_type,
                name,
                RegisterLocation(byte=placement.offset),
                depth=1,
            )
        self.glsl_hlsl_cbuffer_loaders.append(loader_name)
        body = "".join(f"{line}\n" for line in loader.lines)
        word_reader = ""
        if loader.uses_word_reader:
            word_reader = (
                f"uint {loader.word_reader}(int word) {{\n"
                f"    return {registers}[word >> 2][word & 3];\n"
                "}\n"
            )
        return (
            f"{layout} uniform {block_name} {{\n"
            f"    uvec4 {registers}[{register_count}];\n"
            "};\n"
            f"{declarations}"
            f"{word_reader}"
            f"void {loader_name}() {{\n{body}}}\n"
        )

    def generate_glsl_hlsl_cbuffer_loads(self, indent):
        """Calls of every register loader, first in a stage entry."""
        prefix = "    " * indent
        return "".join(
            f"{prefix}{name}();\n"
            for name in getattr(self, "glsl_hlsl_cbuffer_loaders", ()) or ()
        )

    # Loaders ----------------------------------------------------------------

    def glsl_hlsl_cbuffer_load_value(
        self, loader, layout_type, target, location, depth
    ):
        indent = "    " * depth
        kind = layout_type.kind
        if kind in ("scalar", "vector"):
            value = self.glsl_hlsl_cbuffer_vector_value(loader, layout_type, location)
            loader.lines.append(f"{indent}{target} = {value};")
            return
        if kind == "matrix":
            stride = layout_matrix_vector_stride(layout_type, loader.layout)
            column_type = vector_type(layout_type.scalar, layout_type.components)
            emulated = layout_type.scalar in ("int", "uint", "bool")
            columns = f"{target}.c" if emulated else target
            if layout_type.row_major:
                # Register ``r`` holds row ``r``: its component ``c`` is element ``[c][r]``.
                element = scalar_type(layout_type.scalar)
                for row in range(layout_type.registers):
                    for column in range(layout_type.components):
                        value = self.glsl_hlsl_cbuffer_vector_value(
                            loader,
                            element,
                            location.shifted(
                                row * stride + column * layout_type.scalar_size
                            ),
                        )
                        loader.lines.append(
                            f"{indent}{columns}[{column}][{row}] = {value};"
                        )
                return
            for index in range(layout_type.registers):
                value = self.glsl_hlsl_cbuffer_vector_value(
                    loader, column_type, location.shifted(index * stride)
                )
                loader.lines.append(f"{indent}{columns}[{index}] = {value};")
            return
        if kind == "struct":
            for placement in layout_member_placements(layout_type, loader.layout):
                member = self.glsl_struct_member_name(
                    layout_type.struct_name, placement.name
                )
                self.glsl_hlsl_cbuffer_load_value(
                    loader,
                    placement.layout_type,
                    f"{target}.{member}",
                    location.shifted(placement.offset),
                    depth,
                )
            return
        level = depth - 1
        while len(loader.index_names) <= level:
            loader.index_names.append(
                self.glsl_generated_module_identifier(
                    ("hlsl-cbuffer-index", len(loader.index_names)), "crossgl_index"
                )
            )
        index = loader.index_names[level]
        stride = layout_array_stride(layout_type, loader.layout)
        if stride % CBUFFER_REGISTER_SIZE == 0:
            # The elements start registers: the index counts whole registers.
            registers = stride // CBUFFER_REGISTER_SIZE
            term = index if registers == 1 else f"{index} * {registers}"
            element_location = location.indexed(term)
        else:
            words = stride // WORD_SIZE
            term = index if words == 1 else f"{index} * {words}"
            element_location = location.word_indexed(term)
        loader.lines.append(
            f"{indent}for (int {index} = 0; {index} < {layout_type.count}; ++{index}) {{"
        )
        self.glsl_hlsl_cbuffer_load_value(
            loader,
            layout_type.element,
            f"{target}[{index}]",
            element_location,
            depth + 1,
        )
        loader.lines.append(f"{indent}}}")

    @staticmethod
    def glsl_hlsl_cbuffer_word(loader, location):
        """The ``uint`` holding the word at ``location``."""
        if location.word_terms:
            loader.uses_word_reader = True
            return f"{loader.word_reader}({location.word_index()})"
        register, component = location.register_and_component(loader.registers)
        return f"{register}.{COMPONENTS[component]}"

    def glsl_hlsl_cbuffer_vector_value(self, loader, layout_type, location):
        """The value of a scalar or vector whose first byte is at ``location``."""
        scalar = layout_type.scalar
        width = layout_type.components if layout_type.kind == "vector" else 1
        component = (location.byte % CBUFFER_REGISTER_SIZE) // WORD_SIZE
        if (
            location.word_terms
            or (layout_type.scalar_size == 4 and component + width > len(COMPONENTS))
            or location.byte % layout_type.scalar_size
        ):
            # The value is not within one register at a known component: read it word by word.
            return self.glsl_hlsl_cbuffer_word_value(loader, layout_type, location)
        if layout_type.scalar_size == 8:
            parts = []
            for index in range(width):
                register, component = location.shifted(
                    8 * index
                ).register_and_component(loader.registers)
                words = f"{register}.{COMPONENTS[component:component + 2]}"
                parts.append(
                    {
                        "double": f"packDouble2x32({words})",
                        "long": f"packInt2x32(ivec2({words}))",
                        "ulong": f"packUint2x32({words})",
                    }[scalar]
                )
            if width == 1:
                return parts[0]
            prefix = {"double": "d", "long": "i64", "ulong": "u64"}[scalar]
            return f"{prefix}vec{width}({', '.join(parts)})"
        register, component = location.register_and_component(loader.registers)
        words = f"{register}.{COMPONENTS[component:component + width]}"
        return glsl_value_from_words(scalar, width, words)

    def glsl_hlsl_cbuffer_word_value(self, loader, layout_type, location):
        """A scalar or vector assembled from its words, wherever they lie."""
        scalar = layout_type.scalar
        width = layout_type.components if layout_type.kind == "vector" else 1
        if layout_type.scalar_size == 8:
            parts = []
            for index in range(width):
                low = self.glsl_hlsl_cbuffer_word(loader, location.shifted(8 * index))
                high = self.glsl_hlsl_cbuffer_word(
                    loader, location.shifted(8 * index + WORD_SIZE)
                )
                words = f"uvec2({low}, {high})"
                parts.append(
                    {
                        "double": f"packDouble2x32({words})",
                        "long": f"packInt2x32(ivec2({words}))",
                        "ulong": f"packUint2x32({words})",
                    }[scalar]
                )
            if width == 1:
                return parts[0]
            prefix = {"double": "d", "long": "i64", "ulong": "u64"}[scalar]
            return f"{prefix}vec{width}({', '.join(parts)})"
        words = [
            self.glsl_hlsl_cbuffer_word(loader, location.shifted(WORD_SIZE * index))
            for index in range(width)
        ]
        packed = words[0] if width == 1 else f"uvec{width}({', '.join(words)})"
        return glsl_value_from_words(scalar, width, packed)


def glsl_value_from_words(scalar, width, words):
    """Convert ``words``, a ``uint`` or ``uvecN`` expression, to a value of ``scalar`` components."""
    if width == 1:
        return {
            "uint": words,
            "int": f"int({words})",
            "float": f"uintBitsToFloat({words})",
            "bool": f"({words} != 0u)",
        }[scalar]
    return {
        "uint": words,
        "int": f"ivec{width}({words})",
        "float": f"uintBitsToFloat({words})",
        "bool": f"notEqual({words}, uvec{width}(0u))",
    }[scalar]
