"""OpenGL uniform blocks that reproduce HLSL constant-buffer layouts.

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
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ..ast import ArrayNode, FunctionNode
from .array_utils import format_c_style_array_declaration
from .buffer_member_layout import member_explicit_offset, member_is_row_major
from .hlsl_buffer_layout import (
    CBUFFER_REGISTER_SIZE,
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    align_to,
    array_type,
    buffer_layout_from_attributes,
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


def hlsl_scalar_offsets(layout_type, offset=0):
    """Byte offsets of every scalar of a value under HLSL constant packing."""
    kind = layout_type.kind
    if kind == "scalar":
        return [offset]
    if kind == "vector":
        size = layout_type.scalar_size
        return [offset + k * size for k in range(layout_type.components)]
    if kind == "matrix":
        stride = cbuffer_matrix_vector_stride(layout_type)
        column = vector_type(layout_type.scalar, layout_type.components)
        return [
            item
            for index in range(layout_type.registers)
            for item in hlsl_scalar_offsets(column, offset + index * stride)
        ]
    if kind == "array":
        stride = cbuffer_array_stride(layout_type)
        return [
            item
            for index in range(layout_type.count)
            for item in hlsl_scalar_offsets(
                layout_type.element, offset + index * stride
            )
        ]
    placements, _size = cbuffer_struct_placements(layout_type)
    return [
        item
        for placement in placements
        for item in hlsl_scalar_offsets(
            placement.layout_type, offset + placement.offset
        )
    ]


@dataclass(frozen=True)
class RegisterLocation:
    """A byte address in the register array: ``16 * sum(terms) + byte``.

    ``terms`` are loop-dependent register counts; every array element and
    struct starts a register, so they never change the byte within one.
    """

    terms: Tuple[str, ...] = ()
    byte: int = 0

    def shifted(self, delta):
        return RegisterLocation(self.terms, self.byte + delta)

    def indexed(self, term):
        return RegisterLocation((*self.terms, term), self.byte)

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
        raw_type = str(raw_type or "").strip()
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
            members.append((member_name, member_type))
        return struct_type(name, members)

    # Declarations -----------------------------------------------------------

    def glsl_hlsl_cbuffer_block(self, node, layout, block_name):
        """Register storage for ``node`` when std140 differs from HLSL, else ``None``.

        Returns the uniform block, the private member variables and their
        loader; stage entries call the loader through
        :meth:`generate_glsl_hlsl_cbuffer_loads`.
        """
        if not hasattr(self, "glsl_hlsl_cbuffer_loaders"):
            self.reset_glsl_hlsl_cbuffers()
        attributes = getattr(node, "attributes", None)
        if buffer_layout_from_attributes(attributes) != HLSL_CBUFFER_LAYOUT:
            return None
        members = [
            member
            for member in getattr(node, "members", []) or []
            if getattr(member, "name", None) and not isinstance(member, FunctionNode)
        ]
        try:
            layout_type = struct_type(
                node.name,
                [
                    (
                        member.name,
                        self.glsl_hlsl_cbuffer_layout_type(
                            self.glsl_hlsl_cbuffer_member_raw_type(member),
                            (node.name,),
                            member_is_row_major(member),
                        ),
                    )
                    for member in members
                ],
                [member_explicit_offset(member) for member in members],
            )
            placements, size = cbuffer_struct_placements(layout_type)
            std140_offsets, _std140_size, _alignment = std140_struct_offsets(
                layout_type
            )
            # GLSL orients a whole block member, not the members of its struct.
            nested_orientation = any(
                member_type.kind != "matrix" and contains_row_major_matrix(member_type)
                for _name, member_type in layout_type.members
            )
            if not nested_orientation and all(
                hlsl_scalar_offsets(placement.layout_type, placement.offset)
                == std140_scalar_offsets(placement.layout_type, std140_offset)
                for placement, std140_offset in zip(placements, std140_offsets)
            ):
                return None
            return self.glsl_hlsl_cbuffer_register_block(
                node, layout, block_name, members, placements, size
            )
        except HlslLayoutError as exc:
            raise OpenGLHlslConstantBufferError(
                f"Cannot reproduce the HLSL layout of cbuffer '{node.name}' in "
                f"OpenGL: {exc}",
                cbuffer=node.name,
                reason="hlsl-cbuffer-layout",
            ) from exc

    def glsl_hlsl_cbuffer_register_block(
        self, node, layout, block_name, members, placements, size
    ):
        identity = ("hlsl-cbuffer", id(node))
        registers = self.glsl_generated_module_identifier(
            (*identity, "registers"), "crossgl_cbuffer_registers"
        )
        loader_name = self.glsl_generated_module_identifier(
            (*identity, "loader"), f"crossgl_load_{block_name}"
        )
        register_count = max(1, align_to(size, CBUFFER_REGISTER_SIZE) // 16)
        loader = RegisterLoader(registers, [])
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
        return (
            f"{layout} uniform {block_name} {{\n"
            f"    uvec4 {registers}[{register_count}];\n"
            "};\n"
            f"{declarations}"
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
            stride = cbuffer_matrix_vector_stride(layout_type)
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
            placements, _size = cbuffer_struct_placements(layout_type)
            for placement in placements:
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
        # Arrays start a register and have a stride of whole registers.
        level = depth - 1
        while len(loader.index_names) <= level:
            loader.index_names.append(
                self.glsl_generated_module_identifier(
                    ("hlsl-cbuffer-index", len(loader.index_names)), "crossgl_index"
                )
            )
        index = loader.index_names[level]
        registers = cbuffer_array_stride(layout_type) // CBUFFER_REGISTER_SIZE
        term = index if registers == 1 else f"{index} * {registers}"
        loader.lines.append(
            f"{indent}for (int {index} = 0; {index} < {layout_type.count}; ++{index}) {{"
        )
        self.glsl_hlsl_cbuffer_load_value(
            loader,
            layout_type.element,
            f"{target}[{index}]",
            location.indexed(term),
            depth + 1,
        )
        loader.lines.append(f"{indent}}}")

    def glsl_hlsl_cbuffer_vector_value(self, loader, layout_type, location):
        """The value of a scalar or vector whose first byte is at ``location``."""
        scalar = layout_type.scalar
        width = layout_type.components if layout_type.kind == "vector" else 1
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
        if component + width > len(COMPONENTS):
            raise HlslLayoutError("a value straddles two constant-buffer registers")
        words = f"{register}.{COMPONENTS[component:component + width]}"
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
