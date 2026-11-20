"""WGSL uniform blocks read from 16-byte registers where WGSL's own layout would move their members.

CrossGL lays out a constant block by the rules it declares: std140 when it names none (std430 for a push-constant
block), or the ``@ buffer_layout(...)`` it names, with the ``@ offset`` and ``@ align`` of its members
(:mod:`glsl_block_layout`, :mod:`hlsl_buffer_layout`). WGSL lays out a ``var<uniform>`` struct by its own rules
(WGSL sections 14.4.1, "Alignment and Size", and 14.4.4, "Address Space Layout Constraints"), which differ from each
of them somewhere:

* a ``matCx2<f32>`` keeps its columns 8 bytes apart, where std140 and HLSL packing give each column 16 bytes;
* WGSL has no row-major matrices, so a ``@ row_major`` member declared ``matCxR`` reads its rows as columns;
* HLSL packing puts ``vec3 b`` after ``float a`` at byte 4, and the member after ``float f[2]`` at byte 20; WGSL
  aligns the ``vec3<f32>`` to 16 bytes and pads the array to 32;
* std430, scalar and Metal blocks keep ``float`` and ``vec2`` array elements 4 and 8 bytes apart, and the uniform
  address space rejects any array stride that is not a multiple of 16;
* the generator writes no WGSL attribute for ``@ offset`` or ``@ align``.

The generator keeps its typed declaration of a block when WGSL places every scalar of it at the byte the block's
layout gives it and the uniform address space accepts the struct (:func:`typed_layout_matches`). Any other block is
declared as ``registers: array<vec4<T>, N>``, the block's bytes in 16-byte registers, and each read of a member, or
of an element, column or struct member inside one, is built from the words at the offsets the layout gives them::

    mat2 m;            ->  mat2x2<f32>(_Params.registers[0].xy, _Params.registers[1].xy)
    @row_major mat3 r  ->  transpose(mat3x3<f32>(_Params.registers[2].xyz, ...))

``T`` is ``f32``, ``i32`` or ``u32`` when every scalar of the block has that type, so most reads need no ``bitcast``;
a block with mixed scalar types takes ``u32``, bit-casts the other scalars and reads a ``bool`` as ``word != 0u``. A
read whose index is not a constant and that would name its register more than once, and every read that builds an
array, calls a helper function that takes the register (or, when an array stride is not a whole number of registers,
the 4-byte word) where the value starts, so every index is evaluated once. A block whose scalars are not all 4-byte
``float``, ``int``, ``uint`` or ``bool`` values keeps the typed declaration; when it declares a layout or member
placements, a layout the planners cannot reproduce raises :class:`..refusals.WGSLRefusal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    FunctionNode,
    IdentifierNode,
    MemberAccessNode,
)
from ..refusals import WGSLRefusal
from .buffer_member_layout import (
    member_explicit_offset,
    member_is_packed,
    member_is_row_major,
)
from .glsl_block_layout import (
    STD140_LAYOUT,
    STD430_LAYOUT,
    block_matrix_vector_stride,
    block_struct_placements,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    layout_array_stride,
    layout_struct_placements,
    member_explicit_alignment,
)
from .glsl_hlsl_cbuffer import layout_scalar_offsets
from .hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    align_to,
    cbuffer_matrix_vector_stride,
    cbuffer_struct_placements,
    vector_type,
)
from .push_constant_blocks import is_push_constant

__all__ = [
    "plan_uniform_registers",
    "typed_layout_matches",
    "wgsl_register_block_lines",
    "wgsl_uniform_register_helpers",
    "wgsl_uniform_register_load",
]

REGISTER_SIZE = 16
WORD_SIZE = 4
COMPONENTS = "xyzw"
REGISTERS_MEMBER = "registers"
WGSL_SCALARS = {"float": "f32", "int": "i32", "uint": "u32", "bool": "bool"}
REGISTER_SCALARS = {"float": "f32", "int": "i32", "uint": "u32"}


@dataclass
class RegisterBlock:
    """A constant block declared as registers: its WGSL names, layout rules, member placements and register type."""

    name: str
    struct_name: str
    instance: str
    layout: str
    placements: dict
    register_count: int
    register_scalar: str


@dataclass
class UniformRegisterPlan:
    """The register blocks of a program, the members they declare, and the load helpers emitted so far."""

    blocks: dict
    members: dict
    helpers: dict = field(default_factory=dict)
    helper_names: set = field(default_factory=set)


@dataclass(frozen=True)
class _Place:
    """A value inside a register block: its layout type, its static byte offset and its index terms.

    ``registers`` and ``words`` hold ``(index expression, count)`` pairs: the value moves ``count`` registers (or
    4-byte words) for each step of the index. ``component_stride`` is the distance between the components of a
    vector that is a column of a row-major matrix, ``0`` for a vector whose components are adjacent.
    """

    block: RegisterBlock
    layout_type: object
    byte: int = 0
    registers: tuple = ()
    words: tuple = ()
    component_stride: int = 0


# Planning ------------------------------------------------------------------------------------------------------
def _members(node):
    return [
        member
        for member in getattr(node, "members", None) or []
        if getattr(member, "name", None) and not isinstance(member, FunctionNode)
    ]


def _member_tuple(member):
    """``(name, type, row_major, offset, alignment, packed)`` of a block or struct member."""
    if isinstance(member, ArrayNode):
        element_type = getattr(member, "element_type", getattr(member, "vtype", None))
        member_type = ArrayType(element_type, getattr(member, "size", None))
    else:
        member_type = (
            getattr(member, "member_type", None)
            or getattr(member, "var_type", None)
            or getattr(member, "vtype", None)
        )
    return (
        member.name,
        member_type,
        member_is_row_major(member),
        member_explicit_offset(member),
        member_explicit_alignment(member),
        member_is_packed(member),
    )


def _has_placements(members):
    return any(
        member_explicit_offset(member) is not None
        or member_explicit_alignment(member) is not None
        for member in members
    )


def block_layout(cbuffer):
    """The rules a constant block's bytes follow: its declared layout, std430 for a push-constant block, else std140."""
    declared = declared_buffer_layout(getattr(cbuffer, "attributes", None))
    if declared is not None:
        return declared
    return STD430_LAYOUT if is_push_constant(cbuffer) else STD140_LAYOUT


def _root_placements(layout_type, layout):
    if layout == HLSL_CBUFFER_LAYOUT:
        placements, _size = cbuffer_struct_placements(layout_type)
    else:
        placements, _size = block_struct_placements(layout_type, layout, root=True)
    return list(placements)


def _matrix_stride(layout_type, layout):
    if layout == HLSL_CBUFFER_LAYOUT:
        return cbuffer_matrix_vector_stride(layout_type)
    return block_matrix_vector_stride(layout_type, layout)


def _struct_placements(layout_type, layout):
    placements, _size = layout_struct_placements(layout_type, layout)
    return list(placements)


def _scalars(layout_type):
    """The scalar type names a value holds."""
    kind = layout_type.kind
    if kind in ("scalar", "vector", "matrix"):
        return {layout_type.scalar}
    if kind == "array":
        return _scalars(layout_type.element)
    names = set()
    for _name, member in layout_type.members:
        names |= _scalars(member)
    return names


def _wgsl_align_size(layout_type, nested=False):
    """WGSL's alignment and size of the type the generator declares for a uniform value (WGSL section 14.4.1).

    A block member that is a scalar array is declared as an array of 16-byte wrappers; ``nested`` marks a value
    inside a struct, which keeps its own types (see :func:`_wgsl_scalar_offsets`).
    """
    kind = layout_type.kind
    if kind == "scalar":
        return 4, 4
    if kind == "vector":
        components = layout_type.components
        return (8 if components == 2 else 16), 4 * components
    if kind == "matrix":
        # matCxR<f32>: C columns of vecR, each column aligned and strided like vecR (8 bytes for R = 2, else 16).
        column_align = 8 if layout_type.components == 2 else 16
        return column_align, layout_type.registers * column_align
    if kind == "array":
        if layout_type.element.kind == "scalar" and not nested:
            return 16, 16 * layout_type.count
        align, size = _wgsl_align_size(layout_type.element, True)
        return align, layout_type.count * align_to(size, align)
    align, cursor = 1, 0
    for _name, member in layout_type.members:
        member_align, member_size = _wgsl_align_size(member, True)
        cursor = align_to(cursor, member_align) + member_size
        align = max(align, member_align)
    return align, align_to(cursor, align)


def _wgsl_scalar_offsets(layout_type, base, offsets, nested=False, root=False):
    """Append WGSL's byte offset of every scalar of a typed uniform value; ``False`` when WGSL rejects the value.

    The generator declares a block member of type ``bool`` or ``vecN<bool>`` as ``u32`` or ``vecN<u32>`` and a
    block member that is a scalar array as an array of ``@size(16)`` wrappers. A struct the block holds keeps its
    own member types, so inside it (``nested``) a scalar array has 4-byte strides and a ``bool`` vector is not
    host-shareable (WGSL section 14.4.4). ``bool`` arrays are refused everywhere, so a block holding one is read
    from registers.
    """
    kind = layout_type.kind
    if kind == "scalar":
        offsets.append(base)
        return not (nested and layout_type.scalar == "bool")
    if kind == "vector":
        offsets.extend(base + 4 * index for index in range(layout_type.components))
        return not (nested and layout_type.scalar == "bool")
    if kind == "matrix":
        if layout_type.row_major:
            return False
        column_align, _size = _wgsl_align_size(layout_type)
        for column in range(layout_type.registers):
            offsets.extend(
                base + column * column_align + 4 * row
                for row in range(layout_type.components)
            )
        return True
    if kind == "array":
        element = layout_type.element
        if element.kind == "scalar" and not nested:
            stride = 16
        else:
            align, size = _wgsl_align_size(element, True)
            stride = align_to(size, align)
        valid = stride % REGISTER_SIZE == 0 and element.scalar != "bool"
        for index in range(layout_type.count):
            element_base = base + index * stride
            valid = _wgsl_scalar_offsets(element, element_base, offsets, True) and valid
        return valid
    valid = True
    cursor = minimum = 0
    member_nested = not root
    for _name, member in layout_type.members:
        align, size = _wgsl_align_size(member, member_nested)
        offset = align_to(cursor, align)
        # A struct or array member of a uniform struct is aligned to a multiple of 16, and the member after a
        # struct starts at least roundUp(16, SizeOf(S)) bytes after it (WGSL section 14.4.4).
        if member.kind in ("struct", "array") and offset % align_to(align, 16):
            valid = False
        if offset < minimum:
            valid = False
        valid = (
            _wgsl_scalar_offsets(member, base + offset, offsets, member_nested)
            and valid
        )
        cursor = offset + size
        minimum = offset + align_to(size, 16) if member.kind == "struct" else 0
    return valid


def typed_layout_matches(layout_type, placements, layout):
    """Whether the typed WGSL struct of a block puts every scalar at the byte its declared layout gives it."""
    declared = [
        offset
        for placement in placements
        for offset in layout_scalar_offsets(
            placement.layout_type, layout, placement.offset
        )
    ]
    # The typed declaration ignores ``@ offset`` and ``@ align``, as the model does.
    typed = []
    if not _wgsl_scalar_offsets(layout_type, 0, typed, root=True):
        return False
    return typed == declared


def _block_layout_type(codegen, cbuffer, structs):
    def struct_members(name):
        struct = structs.get(name)
        if struct is None:
            return None
        return [_member_tuple(member) for member in _members(struct)]

    return crossgl_struct_layout_type(
        cbuffer.name,
        [_member_tuple(member) for member in _members(cbuffer)],
        struct_members,
        (cbuffer.name,),
        codegen.resolve_integer_expression,
    )


def _holds_placements(structs, members, stack=()):
    if _has_placements(members):
        return True
    for member in members:
        member_type = _member_tuple(member)[1]
        while isinstance(member_type, ArrayType):
            member_type = member_type.element_type
        name = getattr(member_type, "name", None)
        struct = structs.get(name) if isinstance(name, str) else None
        if struct is None or name in stack:
            continue
        if _holds_placements(structs, _members(struct), (*stack, name)):
            return True
    return False


def _register_block(codegen, cbuffer, structs):
    """The register block a constant block needs, or ``None`` when its typed declaration is exact."""
    for member in getattr(cbuffer, "members", None) or []:
        codegen.validate_cbuffer_member(cbuffer, member)
    layout = block_layout(cbuffer)
    try:
        layout_type = _block_layout_type(codegen, cbuffer, structs)
        placements = _root_placements(layout_type, layout)
        scalars = _scalars(layout_type)
        if not scalars or not scalars <= set(WGSL_SCALARS):
            raise HlslLayoutError(
                "only float, int, uint and bool members have 4-byte words"
            )
        if typed_layout_matches(layout_type, placements, layout):
            return None
        extent = max(
            offset
            for placement in placements
            for offset in layout_scalar_offsets(
                placement.layout_type, layout, placement.offset
            )
        )
    except HlslLayoutError as error:
        declared = declared_buffer_layout(getattr(cbuffer, "attributes", None))
        if declared is None and not is_push_constant(cbuffer):
            if not _holds_placements(structs, _members(cbuffer)):
                return None
        raise WGSLRefusal(
            f"WGSL target cannot reproduce the {layout} layout of uniform block "
            f"{cbuffer.name}: {error}"
        ) from error
    register_scalar = (
        REGISTER_SCALARS[next(iter(scalars))]
        if len(scalars) == 1 and next(iter(scalars)) in REGISTER_SCALARS
        else "u32"
    )
    return RegisterBlock(
        name=cbuffer.name,
        struct_name=codegen.type_identifier_name(cbuffer.name),
        instance=codegen.cbuffer_instance_name(cbuffer),
        layout=layout,
        placements={placement.name: placement for placement in placements},
        register_count=extent // REGISTER_SIZE + 1,
        register_scalar=register_scalar,
    )


def plan_uniform_registers(codegen, cbuffers, structs):
    """The register plan of a program's constant blocks, or ``None`` when every typed declaration is exact."""
    struct_nodes = {
        getattr(struct, "name", None): struct
        for struct in structs
        if isinstance(getattr(struct, "name", None), str)
    }
    blocks = {}
    for cbuffer in cbuffers:
        block = _register_block(codegen, cbuffer, struct_nodes)
        if block is not None:
            blocks[cbuffer.name] = block
    if not blocks:
        return None
    members = {name: block for block in blocks.values() for name in block.placements}
    return UniformRegisterPlan(blocks=blocks, members=members)


def wgsl_register_block_lines(codegen, node):
    """The struct lines of a register block, or ``None`` when the block keeps its typed declaration."""
    plan = codegen._uniform_registers
    block = plan.blocks.get(node.name) if plan is not None else None
    if block is None:
        return None
    return [
        f"struct {block.struct_name} {{",
        f"    {REGISTERS_MEMBER}: array<vec4<{block.register_scalar}>, "
        f"{block.register_count}>,",
        "};",
    ]


# Places --------------------------------------------------------------------------------------------------------
def _with_index(codegen, place, layout_type, index_expr, stride, count):
    """The place ``stride`` bytes per index step past ``place``, holding a ``layout_type`` value."""
    value = codegen.resolve_integer_expression(index_expr)
    if value is not None:
        if not 0 <= value < count:
            raise WGSLRefusal(
                f"WGSL target cannot read element {value} of a uniform value of {count}"
            )
        return replace(place, layout_type=layout_type, byte=place.byte + value * stride)
    place = replace(place, layout_type=layout_type)
    if stride % REGISTER_SIZE == 0 and not place.words:
        return replace(
            place, registers=place.registers + ((index_expr, stride // REGISTER_SIZE),)
        )
    return replace(place, words=place.words + ((index_expr, stride // WORD_SIZE),))


def _column(codegen, place, index_expr):
    """The place of column ``index_expr`` of the matrix at ``place``."""
    matrix = place.layout_type
    layout = place.block.layout
    stride = _matrix_stride(matrix, layout)
    if not matrix.row_major:
        column = vector_type(matrix.scalar, matrix.components)
        return _with_index(codegen, place, column, index_expr, stride, matrix.registers)
    # A row-major matrix stores rows: column j is component j of each row, ``stride`` bytes apart.
    column = vector_type(matrix.scalar, matrix.registers)
    place = replace(place, component_stride=stride)
    return _with_index(codegen, place, column, index_expr, WORD_SIZE, matrix.components)


def _resolve(codegen, expr):
    """The register place an expression reads, or ``None`` when it is not a value inside a register block."""
    plan = codegen._uniform_registers
    if isinstance(expr, IdentifierNode):
        name = expr.name
        block = plan.members.get(name)
        if block is None or codegen.is_local_identifier(name):
            return None
        if codegen.is_reference_parameter_name(name):
            return None
        placement = block.placements[name]
        return _Place(block, placement.layout_type, placement.offset)
    if isinstance(expr, ArrayAccessNode):
        base = _resolve(codegen, expr.array_expr)
        if base is None:
            return None
        layout_type = base.layout_type
        if layout_type.kind == "array":
            stride = layout_array_stride(layout_type, base.block.layout)
            return _with_index(
                codegen,
                base,
                layout_type.element,
                expr.index_expr,
                stride,
                layout_type.count,
            )
        if layout_type.kind == "matrix":
            return _column(codegen, base, expr.index_expr)
        return None
    if isinstance(expr, MemberAccessNode) and not getattr(expr, "is_pointer", False):
        base = _resolve(codegen, expr.object_expr)
        if base is None or base.layout_type.kind != "struct":
            return None
        for placement in _struct_placements(base.layout_type, base.block.layout):
            if placement.name == expr.member:
                return replace(
                    base,
                    layout_type=placement.layout_type,
                    byte=base.byte + placement.offset,
                )
        raise WGSLRefusal(
            f"WGSL target found no member {expr.member} in uniform struct "
            f"{base.layout_type.struct_name}"
        )
    return None


# Reading -------------------------------------------------------------------------------------------------------
def _type_name(codegen, layout_type, wrapper=None):
    kind = layout_type.kind
    if kind == "scalar":
        return WGSL_SCALARS[layout_type.scalar]
    scalar = WGSL_SCALARS.get(layout_type.scalar, "")
    if kind == "vector":
        return f"vec{layout_type.components}<{scalar}>"
    if kind == "matrix":
        columns, rows = (
            (layout_type.components, layout_type.registers)
            if layout_type.row_major
            else (layout_type.registers, layout_type.components)
        )
        return f"mat{columns}x{rows}<{scalar}>"
    if kind == "array":
        element = wrapper or _type_name(codegen, layout_type.element)
        return f"array<{element}, {layout_type.count}>"
    return codegen.type_identifier_name(layout_type.struct_name)


def _slug(layout_type):
    kind = layout_type.kind
    if kind == "scalar":
        return WGSL_SCALARS[layout_type.scalar]
    if kind == "struct":
        return str(layout_type.struct_name)
    if kind == "array":
        return f"array{layout_type.count}_{_slug(layout_type.element)}"
    scalar = WGSL_SCALARS[layout_type.scalar]
    if kind == "vector":
        return f"vec{layout_type.components}{scalar}"
    if kind == "matrix":
        columns, rows = (
            (layout_type.components, layout_type.registers)
            if layout_type.row_major
            else (layout_type.registers, layout_type.components)
        )
        return f"mat{columns}x{rows}{scalar}"


def _steps(index, count):
    return index if count == 1 else f"{index} * {count}u"


def _sum(*terms):
    terms = [term for term in terms if term and term != "0u"]
    return " + ".join(terms) if terms else "0u"


class _Reader:
    """Writes the WGSL that reads values of one register block.

    ``mode`` is ``register`` or ``word``: ``base`` (WGSL text, or ``None`` for register 0) names the register, or the
    4-byte word, that byte offset 0 of every read is relative to.
    """

    def __init__(self, codegen, block, mode="register", base=None):
        self.codegen = codegen
        self.block = block
        self.mode = mode
        self.base = base

    def register(self, byte):
        index = byte // REGISTER_SIZE
        if self.base is None:
            text = str(index)
        else:
            text = _sum(self.base, f"{index}u" if index else "")
        return f"{self.block.instance}.{REGISTERS_MEMBER}[{text}]"

    def word(self, byte):
        if self.mode == "register":
            component = COMPONENTS[(byte % REGISTER_SIZE) // WORD_SIZE]
            return f"{self.register(byte)}.{component}"
        index = byte // WORD_SIZE
        text = _sum(self.base, f"{index}u" if index else "")
        if " " in text:
            text = f"({text})"
        return f"{self.block.instance}.{REGISTERS_MEMBER}[{text} / 4u][{text} % 4u]"

    def convert(self, scalar, components, text):
        target = WGSL_SCALARS[scalar]
        if scalar == "bool":
            zero = "0u" if components == 1 else f"vec{components}<u32>()"
            return f"({text} != {zero})"
        if target == self.block.register_scalar:
            return text
        if components == 1:
            return f"bitcast<{target}>({text})"
        return f"bitcast<vec{components}<{target}>>({text})"

    def contiguous_register(self, components, byte, component_stride=0):
        """Whether a vector's components lie side by side in one register, so a swizzle reads them."""
        if self.mode != "register" or component_stride not in (0, WORD_SIZE):
            return False
        first = (byte % REGISTER_SIZE) // WORD_SIZE
        return first + components <= 4

    def vector(self, layout_type, byte, component_stride=0):
        components = layout_type.components
        scalar = layout_type.scalar
        if self.contiguous_register(components, byte, component_stride):
            first = (byte % REGISTER_SIZE) // WORD_SIZE
            register = self.register(byte)
            text = (
                register
                if components == 4
                else f"{register}.{COMPONENTS[first:first + components]}"
            )
            return self.convert(scalar, components, text)
        step = component_stride or WORD_SIZE
        values = [
            self.convert(scalar, 1, self.word(byte + index * step))
            for index in range(components)
        ]
        return f"vec{components}<{WGSL_SCALARS[scalar]}>({', '.join(values)})"

    def matrix(self, layout_type, byte):
        stride = _matrix_stride(layout_type, self.block.layout)
        vector = vector_type(layout_type.scalar, layout_type.components)
        vectors = ", ".join(
            self.vector(vector, byte + index * stride)
            for index in range(layout_type.registers)
        )
        stored = f"mat{layout_type.registers}x{layout_type.components}<f32>({vectors})"
        return f"transpose({stored})" if layout_type.row_major else stored

    def struct(self, layout_type, byte):
        codegen = self.codegen
        uniform_form = layout_type.struct_name in codegen._uniform_buffer_struct_names
        values = []
        for placement in _struct_placements(layout_type, self.block.layout):
            member = placement.layout_type
            offset = byte + placement.offset
            if uniform_form and member.kind == "scalar" and member.scalar == "bool":
                # A struct a typed uniform block also holds declares its bool members as u32 words.
                values.append(f"select(0u, 1u, {self.value(member, offset)})")
                continue
            wrapper = None
            if uniform_form and member.kind == "array":
                element = member.element
                if element.kind == "scalar" and element.scalar in REGISTER_SCALARS:
                    wrapper = codegen._uniform_scalar_array_wrappers.get(
                        REGISTER_SCALARS[element.scalar]
                    )
            values.append(self.value(member, offset, wrapper=wrapper))
        return f"{_type_name(codegen, layout_type)}({', '.join(values)})"

    def value(self, layout_type, byte, component_stride=0, wrapper=None):
        kind = layout_type.kind
        if kind == "scalar":
            return self.convert(layout_type.scalar, 1, self.word(byte))
        if kind == "vector":
            return self.vector(layout_type, byte, component_stride)
        if kind == "matrix":
            return self.matrix(layout_type, byte)
        if kind == "array":
            name, argument = _array_helper(self, layout_type, byte, wrapper)
            return f"{name}({argument})"
        return self.struct(layout_type, byte)


def _helper_name(codegen, plan, block, layout_type):
    used = set(codegen._type_identifier_names.values())
    used.update(codegen._function_identifier_names.values())
    used.update(codegen._module_identifier_names.values())
    used.update(plan.helper_names)
    name = codegen.unique_wgsl_identifier(
        f"load_{block.struct_name}_{_slug(layout_type)}", used
    )
    plan.helper_names.add(name)
    return name


def _argument(reader, byte):
    """The helper argument and parameter mode for a value at ``byte`` of ``reader``'s base."""
    if reader.mode == "register":
        index = byte // REGISTER_SIZE
        return "register", _sum(reader.base, f"{index}u" if index else ""), byte % 16
    index = byte // WORD_SIZE
    return "word", _sum(reader.base, f"{index}u" if index else ""), 0


def _array_helper(reader, layout_type, byte, wrapper=None):
    """A helper that copies the array at ``byte`` element by element, and its argument."""
    codegen = reader.codegen
    plan = codegen._uniform_registers
    block = reader.block
    mode, argument, offset = _argument(reader, byte)
    key = ("array", block.name, layout_type, mode, offset, wrapper)
    helper = plan.helpers.get(key)
    if helper is None:
        name = _helper_name(codegen, plan, block, layout_type)
        plan.helpers[key] = (name, None)
        parameter = "r" if mode == "register" else "w"
        stride = layout_array_stride(layout_type, block.layout)
        if mode == "register" and stride % REGISTER_SIZE == 0:
            element_reader = _Reader(
                codegen, block, "register", _sum(parameter, _steps("i", stride // 16))
            )
        else:
            # A stride that is not a whole number of registers steps through words; ``offset`` stays the
            # element's byte within the register ``r`` names.
            base = parameter if mode == "word" else "r * 4u"
            element_reader = _Reader(
                codegen, block, "word", _sum(base, _steps("i", stride // WORD_SIZE))
            )
        element = element_reader.value(layout_type.element, offset)
        if wrapper is not None:
            element = f"{wrapper}({element})"
        type_name = _type_name(codegen, layout_type, wrapper)
        text = "\n".join(
            (
                f"fn {name}({parameter}: u32) -> {type_name} {{",
                f"    var value: {type_name};",
                f"    for (var i = 0u; i < {layout_type.count}u; i = i + 1u) {{",
                f"        value[i] = {element};",
                "    }",
                "    return value;",
                "}",
            )
        )
        plan.helpers[key] = (name, text)
        helper = plan.helpers[key]
    return helper[0], argument


def _value_helper(reader, place_type, byte, component_stride):
    """A helper that reads a value naming its base more than once, and its argument."""
    codegen = reader.codegen
    plan = codegen._uniform_registers
    block = reader.block
    mode, argument, offset = _argument(reader, byte)
    key = ("value", block.name, place_type, mode, offset, component_stride)
    helper = plan.helpers.get(key)
    if helper is None:
        name = _helper_name(codegen, plan, block, place_type)
        plan.helpers[key] = (name, None)
        parameter = "r" if mode == "register" else "w"
        body = _Reader(codegen, block, mode, parameter).value(
            place_type, offset, component_stride
        )
        type_name = _type_name(codegen, place_type)
        text = f"fn {name}({parameter}: u32) -> {type_name} {{\n    return {body};\n}}"
        plan.helpers[key] = (name, text)
        helper = plan.helpers[key]
    return helper[0], argument


def _index_text(codegen, index_expr, count):
    text = codegen.generate_expression(index_expr)
    index_type = codegen.expression_type(index_expr)
    if index_type is None or codegen.scalar_type_name(index_type) != "u32":
        text = f"u32({text})"
    return _steps(text, count)


def _reads_base_once(reader, layout_type, byte, component_stride):
    kind = layout_type.kind
    if kind == "array":
        return True
    if reader.mode != "register":
        return False
    if kind == "scalar":
        return True
    return kind == "vector" and reader.contiguous_register(
        layout_type.components, byte, component_stride
    )


def wgsl_uniform_register_load(codegen, expr):
    """The WGSL that reads ``expr`` from a register block, or ``None`` when ``expr`` reads no such value."""
    if not isinstance(expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode)):
        return None
    place = _resolve(codegen, expr)
    if place is None:
        return None
    block = place.block
    if not place.registers and not place.words:
        return _Reader(codegen, block).value(
            place.layout_type, place.byte, place.component_stride
        )
    registers = [_index_text(codegen, index, count) for index, count in place.registers]
    words = [_index_text(codegen, index, count) for index, count in place.words]
    if words:
        register_words = [f"({term}) * 4u" for term in registers]
        reader = _Reader(codegen, block, "word", _sum(*register_words, *words))
    else:
        reader = _Reader(codegen, block, "register", _sum(*registers))
    if _reads_base_once(reader, place.layout_type, place.byte, place.component_stride):
        return reader.value(place.layout_type, place.byte, place.component_stride)
    name, argument = _value_helper(
        reader, place.layout_type, place.byte, place.component_stride
    )
    return f"{name}({argument})"


def wgsl_uniform_register_helpers(codegen):
    """The load helpers the program's reads called, in the order they were first needed."""
    plan = codegen._uniform_registers
    if plan is None:
        return ""
    return "\n\n".join(text for _name, text in plan.helpers.values() if text)
