"""HLSL constant buffers that read GLSL blocks HLSL packing cannot place.

``packoffset`` moves the members of a constant buffer, but not the elements
of an array, the vectors of a matrix or the members of a nested struct: an
HLSL constant-buffer array always starts each element in its own 16-byte
register. A GLSL std430 or scalar block (a Vulkan push constant, or a block
written for GL_EXT_scalar_block_layout) packs ``float t[4]`` into 16 bytes,
a ``mat2`` into 16 and, under the scalar layout, a ``vec3[]`` with a 12-byte
stride; a std140 block with ``offset`` qualifiers can place a nested struct
at offsets HLSL never uses. An MSL constant buffer (``@ buffer_layout(metal)``)
gives ``float3`` 16 bytes and a packed ``float3`` 12 at 4-byte alignment.

Such a block declares its bytes as one ``uint4`` array, every member becomes
a ``static`` global of its own type and name, and a loader that each stage
entry (and patch-constant function) calls before anything else assigns the
members from the words at their GLSL offsets (``asfloat``, ``asdouble`` and
so on). Member reads are unchanged, and every value comes from the bytes the
host wrote for it. HLSL ``static`` globals are private to each invocation, so
the copy is never shared between threads.
"""

import re
from dataclasses import dataclass, field
from typing import List

from ..ast import ArrayType, FunctionNode, NamedType
from ..refusals import DirectXRefusal
from .glsl_block_layout import block_struct_placements
from .glsl_hlsl_cbuffer import (
    COMPONENTS,
    WORD_SIZE,
    RegisterLocation,
    layout_array_stride,
    layout_matrix_vector_stride,
    layout_member_placements,
    layout_scalar_offsets,
)
from .hlsl_buffer_layout import (
    CBUFFER_REGISTER_SIZE,
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutType,
    align_to,
    cbuffer_held_struct_placements,
    vector_type,
)

# HLSL spellings of the layout scalars.
HLSL_SCALAR_TYPES = {
    "bool": "bool",
    "int": "int",
    "uint": "uint",
    "float": "float",
    "double": "double",
    "long": "int64_t",
    "ulong": "uint64_t",
}
_PACKOFFSET = re.compile(r"\s*:\s*packoffset\s*\([^)]*\)\s*$")
_ORIENTATION = re.compile(r"^(?:row_major|column_major)\s+")


@dataclass(frozen=True)
class HlslRegisterBlockPlan:
    """A GLSL block whose layout ``packoffset`` cannot reproduce, with its layout shape."""

    layout: str
    layout_type: object


def _holds_struct_placements(layout_type):
    """Whether a value holds a struct, at any depth, whose members carry ``@ offset(N)`` or ``@ align(N)``."""
    if layout_type.kind == "array":
        return _holds_struct_placements(layout_type.element)
    if layout_type.kind != "struct":
        return False
    return bool(layout_type.offsets or layout_type.alignments) or any(
        _holds_struct_placements(member) for _name, member in layout_type.members
    )


def _without_placements(layout_type):
    """``layout_type`` with the placements of every struct it holds removed."""
    if layout_type.kind == "array":
        return HlslLayoutType(
            "array",
            element=_without_placements(layout_type.element),
            count=layout_type.count,
        )
    if layout_type.kind != "struct":
        return layout_type
    return HlslLayoutType(
        "struct",
        struct_name=layout_type.struct_name,
        members=tuple(
            (name, _without_placements(member)) for name, member in layout_type.members
        ),
    )


def _type_struct_name(type_node):
    """The struct name a member type names, through any array dimensions, or ``None``."""
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, NamedType):
        return None if type_node.generic_args else type_node.name
    if isinstance(type_node, str):
        return type_node.split("[", 1)[0].strip() or None
    return None


def hlsl_members_hold_struct_placements(structs, members, member_tuple, stack=()):
    """Whether ``members`` hold a struct of ``structs``, at any depth, with an ``@ offset`` or ``@ align`` member.

    ``member_tuple(member)`` is ``(name, type, row_major, offset, alignment, packed)``.
    """
    for member in members:
        name = _type_struct_name(member_tuple(member)[1])
        struct = structs.get(name) if name is not None else None
        if struct is None or name in stack:
            continue
        inner = [
            item
            for item in getattr(struct, "members", []) or []
            if getattr(item, "name", None) and not isinstance(item, FunctionNode)
        ]
        if any(
            member_tuple(item)[3] is not None or member_tuple(item)[4] is not None
            for item in inner
        ) or hlsl_members_hold_struct_placements(
            structs, inner, member_tuple, (*stack, name)
        ):
            return True
    return False


def hlsl_cbuffer_register_plan(layout_type):
    """The plan of a constant buffer packed by HLSL rules whose held structs place a scalar elsewhere, else ``None``.

    ``packoffset`` places only the members of a constant buffer, and HLSL declares a struct once, packed by its
    rules alone, so a struct member's ``@ offset(N)`` or ``@ align(N)`` (DXC's ``[[vk::offset(N)]]`` of a struct
    member) that moves a scalar from where packing puts it is read from the buffer's registers.
    """
    members = layout_type.members
    if not any(_holds_struct_placements(member) for _name, member in members):
        return None
    plain = HlslLayoutType(
        "struct",
        struct_name=layout_type.struct_name,
        members=tuple((name, _without_placements(member)) for name, member in members),
        offsets=layout_type.offsets,
        alignments=layout_type.alignments,
    )
    if layout_scalar_offsets(layout_type, HLSL_CBUFFER_LAYOUT) == layout_scalar_offsets(
        plain, HLSL_CBUFFER_LAYOUT
    ):
        return None
    return HlslRegisterBlockPlan(HLSL_CBUFFER_LAYOUT, layout_type)


@dataclass(frozen=True)
class HlslRegisterBlockCode:
    """The HLSL that reads a GLSL block from ``uint4`` registers, apart from the registers' declaration.

    ``size`` is the block's size in bytes and ``first_offset`` the offset of its first member.
    """

    registers: str
    register_count: int
    size: int
    first_offset: int
    statics: str
    word_reader: str
    loader_name: str
    loader_body: str


@dataclass
class HlslRegisterLoader:
    """Statements that assign every member of one block from its ``uint4`` registers."""

    block: str
    registers: str
    word_reader: str
    layout: str
    lines: List[str] = field(default_factory=list)
    uses_word_reader: bool = False


def hlsl_vector_type_name(scalar, width):
    name = HLSL_SCALAR_TYPES[scalar]
    return name if width == 1 else f"{name}{width}"


def hlsl_value_from_words(scalar, width, words):
    """Convert ``words``, a ``uint`` or ``uintN`` expression, to a value of ``width`` ``scalar`` components."""
    if scalar == "uint":
        return words
    if scalar == "int":
        return f"asint({words})"
    if scalar == "float":
        return f"asfloat({words})"
    if scalar == "bool":
        return f"({words} != 0u)"
    raise DirectXRefusal(f"no 32-bit HLSL reinterpretation for '{scalar}'")


def hlsl_value_from_word_pair(scalar, low, high):
    """An 8-byte scalar from its low and high ``uint`` words."""
    if scalar == "double":
        return f"asdouble({low}, {high})"
    bits = f"((uint64_t({high}) << 32) | uint64_t({low}))"
    if scalar == "ulong":
        return bits
    if scalar == "long":
        return f"int64_t({bits})"
    raise DirectXRefusal(f"no 64-bit HLSL reinterpretation for '{scalar}'")


class HlslGlslBlockRegisters:
    """HLSL code generator support for GLSL blocks read through registers."""

    def reset_hlsl_glsl_block_registers(self):
        self.hlsl_glsl_block_loaders: List[str] = []
        self.hlsl_glsl_block_patch_functions = set()

    def hlsl_glsl_block_register_code(self, node, layout, layout_type):
        """The ``static`` members, word reader and loader of a GLSL block read from ``uint4`` registers.

        The registers are named by :attr:`HlslRegisterBlockCode.registers`; the caller declares them.
        """
        members = [
            member
            for member in getattr(node, "members", []) or []
            if getattr(member, "name", None) and not isinstance(member, FunctionNode)
        ]
        if layout == HLSL_CBUFFER_LAYOUT:
            placements, size = cbuffer_held_struct_placements(layout_type)
        else:
            placements, size = block_struct_placements(layout_type, layout, root=True)
        prefix = f"__crossgl_{node.name}"
        loader = HlslRegisterLoader(
            block=node.name,
            registers=f"{prefix}_registers",
            word_reader=f"{prefix}_word",
            layout=layout,
        )
        declarations = ""
        for member, placement in zip(members, placements):
            declaration = self.generate_hlsl_cbuffer_member_declaration(member)
            declaration = _ORIENTATION.sub("", _PACKOFFSET.sub("", declaration))
            declarations += f"static {declaration};\n"
            self.hlsl_glsl_block_load_value(
                loader,
                placement.layout_type,
                member.name,
                RegisterLocation(byte=placement.offset),
                depth=1,
            )
        word_reader = ""
        if loader.uses_word_reader:
            word_reader = (
                f"uint {loader.word_reader}(int word)\n{{\n"
                f"    return {loader.registers}[word >> 2][word & 3];\n"
                "}\n"
            )
        return HlslRegisterBlockCode(
            registers=loader.registers,
            register_count=max(1, align_to(size, CBUFFER_REGISTER_SIZE) // 16),
            size=size,
            first_offset=min((placement.offset for placement in placements), default=0),
            statics=declarations,
            word_reader=word_reader,
            loader_name=f"__crossgl_load_{node.name}",
            loader_body="".join(f"{line}\n" for line in loader.lines),
        )

    def hlsl_glsl_block_register_buffer(
        self, ast, node, layout, layout_type, buffer_keyword, register
    ):
        """The constant buffer, ``static`` members and loader of a GLSL block HLSL packing cannot place."""
        if not hasattr(self, "hlsl_glsl_block_loaders"):
            self.reset_hlsl_glsl_block_registers()
        code = self.hlsl_glsl_block_register_code(node, layout, layout_type)
        self.hlsl_glsl_block_loaders.append(code.loader_name)
        self.hlsl_glsl_block_patch_functions.update(
            self.hlsl_patch_constant_function_names(ast)
        )
        return (
            f"{buffer_keyword} {node.name}{register}\n{{\n"
            f"    uint4 {code.registers}[{code.register_count}];\n"
            "};\n"
            f"{code.statics}"
            f"{code.word_reader}"
            f"void {code.loader_name}()\n{{\n{code.loader_body}}}\n"
        )

    def hlsl_patch_constant_function_names(self, ast):
        """Names of the functions a hull shader runs as its patch-constant function."""
        functions = list(getattr(ast, "functions", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            functions.extend(getattr(stage, "local_functions", []) or [])
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                functions.append(entry)
        names = set()
        for func in functions:
            name = self.hlsl_stage_attribute_argument(func, "patchconstantfunc")
            if name:
                names.add(str(name).strip().strip('"'))
        return names

    def generate_hlsl_glsl_block_loads(self, function_name, is_entry, indent):
        """Calls of every register loader, first in a stage entry or patch-constant function."""
        loaders = getattr(self, "hlsl_glsl_block_loaders", None) or ()
        if not loaders:
            return ""
        patch_functions = getattr(self, "hlsl_glsl_block_patch_functions", set())
        if not is_entry and function_name not in patch_functions:
            return ""
        prefix = "    " * indent
        return "".join(f"{prefix}{name}();\n" for name in loaders)

    # Loaders ----------------------------------------------------------------

    def hlsl_glsl_block_load_value(self, loader, layout_type, target, location, depth):
        indent = "    " * depth
        kind = layout_type.kind
        if kind in ("scalar", "vector"):
            value = self.hlsl_glsl_block_vector_value(loader, layout_type, location)
            loader.lines.append(f"{indent}{target} = {value};")
            return
        if kind == "matrix":
            self.hlsl_glsl_block_load_matrix(
                loader, layout_type, target, location, indent
            )
            return
        if kind == "struct":
            for placement in layout_member_placements(layout_type, loader.layout):
                self.hlsl_glsl_block_load_value(
                    loader,
                    placement.layout_type,
                    f"{target}.{placement.name}",
                    location.shifted(placement.offset),
                    depth,
                )
            return
        index = f"__crossgl_{loader.block}_i{depth - 1}"
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
            f"{indent}for (int {index} = 0; {index} < {layout_type.count}; ++{index})"
        )
        loader.lines.append(f"{indent}{{")
        self.hlsl_glsl_block_load_value(
            loader,
            layout_type.element,
            f"{target}[{index}]",
            element_location,
            depth + 1,
        )
        loader.lines.append(f"{indent}}}")

    def hlsl_glsl_block_load_matrix(
        self, loader, layout_type, target, location, indent
    ):
        """Assign a matrix vector by vector: rows for row-major storage, columns otherwise.

        HLSL ``floatRxC`` is indexed by row, so a stored row is ``m[r]`` and a
        stored column ``c`` is the write mask ``m._m0c_m1c...``.
        """
        stride = layout_matrix_vector_stride(layout_type, loader.layout)
        vector = vector_type(layout_type.scalar, layout_type.components)
        for index in range(layout_type.registers):
            value = self.hlsl_glsl_block_vector_value(
                loader, vector, location.shifted(index * stride)
            )
            if layout_type.row_major:
                destination = f"{target}[{index}]"
            else:
                mask = "".join(
                    f"_m{row}{index}" for row in range(layout_type.components)
                )
                destination = f"{target}.{mask}"
            loader.lines.append(f"{indent}{destination} = {value};")

    @staticmethod
    def hlsl_glsl_block_word(loader, location):
        """The ``uint`` holding the word at ``location``."""
        if location.word_terms:
            loader.uses_word_reader = True
            return f"{loader.word_reader}({location.word_index()})"
        register, component = location.register_and_component(loader.registers)
        return f"{register}.{COMPONENTS[component]}"

    def hlsl_glsl_block_vector_value(self, loader, layout_type, location):
        """The value of a scalar or vector whose first byte is at ``location``."""
        scalar = layout_type.scalar
        width = layout_type.components if layout_type.kind == "vector" else 1
        if layout_type.scalar_size == 8:
            parts = [
                hlsl_value_from_word_pair(
                    scalar,
                    self.hlsl_glsl_block_word(loader, location.shifted(8 * index)),
                    self.hlsl_glsl_block_word(
                        loader, location.shifted(8 * index + WORD_SIZE)
                    ),
                )
                for index in range(width)
            ]
            if width == 1:
                return parts[0]
            return f"{hlsl_vector_type_name(scalar, width)}({', '.join(parts)})"
        component = (location.byte % CBUFFER_REGISTER_SIZE) // WORD_SIZE
        if location.word_terms or component + width > len(COMPONENTS):
            # The value is not within one register at a known component: read it word by word.
            words = [
                self.hlsl_glsl_block_word(loader, location.shifted(WORD_SIZE * index))
                for index in range(width)
            ]
            packed = words[0] if width == 1 else f"uint{width}({', '.join(words)})"
            return hlsl_value_from_words(scalar, width, packed)
        register, component = location.register_and_component(loader.registers)
        words = f"{register}.{COMPONENTS[component:component + width]}"
        return hlsl_value_from_words(scalar, width, words)


__all__ = [
    "HlslGlslBlockRegisters",
    "HlslRegisterBlockPlan",
    "hlsl_cbuffer_register_plan",
    "hlsl_members_hold_struct_placements",
    "hlsl_value_from_words",
    "hlsl_value_from_word_pair",
]
