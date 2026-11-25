"""CrossGL constant blocks (``cbuffer``) as rust-gpu uniform and push-constant structs.

A CrossGL ``cbuffer Params @binding(3) { float scale; vec3 tint; };`` is one block whose members a shader reads by
their bare names. Its bytes follow the rules the block declares: std140 when it names none, std430 for a
``@push_constant`` block, or the ``@buffer_layout(...)`` it names, with the ``@offset`` and ``@align`` of its members
(:mod:`.glsl_block_layout`, :mod:`.hlsl_buffer_layout`). rust-gpu binds such a block as an entry-point parameter
``#[spirv(uniform, descriptor_set = S, binding = B)] params: &Params`` (or ``#[spirv(push_constant)]``) whose type is
a ``#[repr(C)]`` struct, and rustc_codegen_spirv decorates each member with the offset of its Rust layout.

:func:`lower_rust_gpu_constant_blocks` declares that struct and a ``uniform Params params`` variable with the block's
binding attributes, which the generator binds like any other uniform value, and rewrites every read of a member into
a read through ``params``. The struct places each member at the byte the block's layout gives it:

* a member whose Rust type lays out every scalar where the block does (``float``, ``vec3``, ``vec4[2]``, ``mat4``,
  a struct whose ``#[repr(C)]`` layout is the block's) keeps its type; ``u32`` fields fill any gap before it, as
  std140 leaves after ``vec3 a; vec4 b;`` and HLSL packing does not;
* a ``bool`` or ``bvecN`` member is held as ``u32`` or ``UVecN``, which a buffer can hold, and read as
  ``x != 0u`` or ``notEqual(x, uvecN(0u))``;
* an array of scalars or vectors whose elements sit 16 bytes apart (std140 and HLSL packing) is held as an array
  of 16-byte vectors and an element read takes the components it needs (``w[i]`` reads ``params.w[i].x``, and a
  ``bool`` array's ``flags[i]`` reads ``params.flags[i].x != 0u``); a matrix
  whose columns, or rows for ``@row_major``, sit 16 bytes apart is held as one 16-byte vector per column (row), and
  a read builds the matrix again (``mat2(m[0].xy, m[1].xy)``, transposed for rows), an array of such matrices
  indexing ``i * R + r``.

glam's ``Vec4``, ``Mat2`` and ``Mat4`` are 16-byte aligned and its other vector and matrix types are aligned to their
component (:mod:`.rust_gpu_layouts`), and glam has no non-square matrix. A member none of these forms places at its
offset, a whole-array read of an array held as 16-byte vectors, a ``bool`` array whose elements are not 16 bytes
apart and a ``bool`` inside a struct are refused with the member named.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    AttributeNode,
    BinaryOpNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PrimitiveType,
    StructMemberNode,
    StructNode,
    TypeNode,
    VariableNode,
    VectorType,
)
from ..semantics.rewrite import child_slots, get_child, is_pure, set_child
from ..semantics.scoped_walk import ScopedWalker, program_stages
from .buffer_member_layout import (
    member_explicit_offset,
    member_is_packed,
    member_is_row_major,
)
from .glsl_block_layout import (
    STD140_LAYOUT,
    STD430_LAYOUT,
    block_struct_placements,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    member_explicit_alignment,
)
from .glsl_hlsl_cbuffer import (
    layout_array_stride,
    layout_matrix_vector_stride,
    layout_scalar_offsets,
)
from .hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    align_to,
    cbuffer_struct_placements,
)
from .push_constant_blocks import is_push_constant
from .rust_gpu_constants import rust_integer_constants

__all__ = ["CONSTANT_BLOCK_FLAG", "lower_rust_gpu_constant_blocks"]

# Python attribute of the ``uniform`` variable a block became: its struct is laid out here, not by std140.
CONSTANT_BLOCK_FLAG = "rust_gpu_constant_block"

_RUST_SCALARS = frozenset({"float", "int", "uint", "double"})
_REGISTER_VECTORS = {"float": "vec4", "int": "ivec4", "uint": "uvec4", "bool": "uvec4"}
_REGISTER = 16
_COMPONENTS = "xyzw"


class _Unplaced(Exception):
    """A member the struct cannot hold at its offset; the message is the refusal reason."""


@dataclass
class _Member:
    """One member of a block: its offset and the form the struct holds it in."""

    name: str
    offset: int
    storage: TypeNode
    form: str  # "direct", "bool" or "registers"
    size: int = 0
    components: int = 0
    registers: int = 1
    count: int | None = None
    matrix: tuple | None = (
        None  # (size, row_major) of a square matrix read from its registers
    )
    boolean: bool = False  # registers that hold bool values as u32 words


def _members(node):
    return [
        member
        for member in getattr(node, "members", None) or []
        if getattr(member, "name", None) and not isinstance(member, FunctionNode)
    ]


def _member_type(member):
    if isinstance(member, ArrayNode):
        element = getattr(member, "element_type", getattr(member, "vtype", None))
        return ArrayType(element, getattr(member, "size", None))
    return (
        getattr(member, "member_type", None)
        or getattr(member, "var_type", None)
        or getattr(member, "vtype", None)
    )


def _member_tuple(member):
    """``(name, type, row_major, offset, alignment, packed)``, as :func:`.crossgl_struct_layout_type` reads them."""
    return (
        member.name,
        _member_type(member),
        member_is_row_major(member),
        member_explicit_offset(member),
        member_explicit_alignment(member),
        member_is_packed(member),
    )


def _block_layout(cbuffer):
    declared = declared_buffer_layout(getattr(cbuffer, "attributes", None))
    if declared is not None:
        return declared
    return STD430_LAYOUT if is_push_constant(cbuffer) else STD140_LAYOUT


def _structs(ast):
    structs = {}
    for struct in getattr(ast, "structs", None) or []:
        if isinstance(struct, StructNode):
            structs[struct.name] = struct
    for stage in program_stages(ast):
        for struct in getattr(stage, "local_structs", None) or []:
            if isinstance(struct, StructNode):
                structs[struct.name] = struct
    return structs


# Layouts ----------------------------------------------------------------------------------------------------------
def _holds_bool(layout_type):
    if layout_type.kind in ("scalar", "vector", "matrix"):
        return layout_type.scalar == "bool"
    if layout_type.kind == "array":
        return _holds_bool(layout_type.element)
    return any(_holds_bool(member) for _name, member in layout_type.members)


def _rust_shape(layout_type, path):
    """``(size, alignment, scalar offsets)`` of the ``#[repr(C)]`` Rust value the generator writes for a type."""
    kind = layout_type.kind
    if kind in ("scalar", "vector", "matrix"):
        if layout_type.scalar not in _RUST_SCALARS:
            raise _Unplaced(
                f"'{path}' holds {layout_type.scalar} values, which a rust-gpu buffer struct does not hold"
            )
        width = layout_type.scalar_size
    if kind == "scalar":
        return width, width, [0]
    if kind == "vector":
        count = layout_type.components
        align = 16 if (layout_type.scalar, count) == ("float", 4) else width
        return width * count, align, [k * width for k in range(count)]
    if kind == "matrix":
        size = layout_type.registers
        if layout_type.row_major:
            raise _Unplaced(f"'{path}' is a row-major matrix")
        if layout_type.components != size:
            raise _Unplaced(
                f"'{path}' is a non-square matrix, which glam does not have"
            )
        align = (
            16 if (layout_type.scalar, size) in (("float", 2), ("float", 4)) else width
        )
        column = size * width
        offsets = [c * column + r * width for c in range(size) for r in range(size)]
        return size * column, align, offsets
    if kind == "array":
        size, align, offsets = _rust_shape(layout_type.element, f"{path}[]")
        return (
            size * layout_type.count,
            align,
            [i * size + o for i in range(layout_type.count) for o in offsets],
        )
    cursor = 0
    align = 1
    offsets = []
    for name, member in layout_type.members:
        size, member_align, member_offsets = _rust_shape(member, f"{path}.{name}")
        cursor = align_to(cursor, member_align)
        offsets += [cursor + o for o in member_offsets]
        cursor += size
        align = max(align, member_align)
    return align_to(cursor, align), align, offsets


def _primitive(name):
    return PrimitiveType(name)


def _storage_scalar(layout_type):
    """The CrossGL type of a ``bool`` or ``bvecN`` member's ``uint`` storage."""
    if layout_type.kind == "scalar":
        return _primitive("uint")
    return VectorType(_primitive("uint"), layout_type.components)


def _uint_layout(layout_type):
    return replace(layout_type, scalar="uint")


def _direct(name, layout_type, offset, layout, member_type):
    """The member at ``offset`` with its own (or, for a ``bool``, its ``uint``) type, else ``_Unplaced``."""
    form = "direct"
    storage = member_type
    if _holds_bool(layout_type):
        if layout_type.kind not in ("scalar", "vector"):
            raise _Unplaced(
                f"'{name}' holds bool values inside an array or struct; the rust-gpu output holds a "
                "constant block's bool and bvec members only as u32 words"
            )
        layout_type = _uint_layout(layout_type)
        form = "bool"
        storage = _storage_scalar(layout_type)
    size, align, offsets = _rust_shape(layout_type, name)
    expected = layout_scalar_offsets(layout_type, layout, 0)
    if offset % align:
        raise _Unplaced(
            f"'{name}' is at {layout} offset {offset}, which its rust-gpu type, aligned to {align} bytes, "
            "cannot start at"
        )
    if offsets != expected:
        raise _Unplaced(
            f"'{name}' lays out its values at {layout} offsets {expected[:6]}, rust-gpu at {offsets[:6]}"
        )
    return _Member(name, offset, storage, form, size=size)


def _registers(name, layout_type, offset, layout):
    """The member as an array of 16-byte vectors, for std140 and HLSL arrays and matrices, else ``None``."""
    count = None
    element = layout_type
    if layout_type.kind == "array":
        count = layout_type.count
        element = layout_type.element
    if (
        element.kind not in ("scalar", "vector", "matrix")
        or element.scalar not in _REGISTER_VECTORS
    ):
        return None
    matrix = None
    registers = 1
    components = element.components
    if element.kind == "matrix":
        if element.registers != element.components or element.scalar != "float":
            return None
        if layout_matrix_vector_stride(element, layout) != _REGISTER:
            return None
        registers = element.registers
        matrix = (element.registers, element.row_major)
    elif count is None:
        return None
    if (
        count is not None
        and layout_array_stride(layout_type, layout) != _REGISTER * registers
    ):
        return None
    if offset % _REGISTER:
        return None
    width = element.scalar_size
    held = [
        offset + (i * registers + r) * _REGISTER + k * width
        for i in range(count or 1)
        for r in range(registers)
        for k in range(components)
    ]
    if held != layout_scalar_offsets(layout_type, layout, offset):
        return None
    total = (count or 1) * registers
    storage = ArrayType(
        NamedType(_REGISTER_VECTORS[element.scalar]),
        LiteralNode(total, _primitive("int")),
    )
    return _Member(
        name,
        offset,
        storage,
        "registers",
        size=_REGISTER * total,
        components=components,
        registers=registers,
        count=count,
        matrix=matrix,
        boolean=element.scalar == "bool",
    )


def _place(name, layout_type, offset, layout, member_type):
    try:
        return _direct(name, layout_type, offset, layout, member_type)
    except _Unplaced as reason:
        held = _registers(name, layout_type, offset, layout)
        if held is None:
            raise reason
        return held


def _plan(ast, cbuffer, constants):
    """The members of the struct a block becomes, in offset order, with ``u32`` fields filling the gaps."""
    structs = _structs(ast)
    layout = _block_layout(cbuffer)

    def struct_members(name):
        struct = structs.get(name)
        if struct is None:
            return None
        return [_member_tuple(member) for member in _members(struct)]

    def constant_value(expression):
        if isinstance(expression, IdentifierNode) and expression.name in constants:
            return constants[expression.name].value
        return None

    members = _members(cbuffer)
    try:
        layout_type = crossgl_struct_layout_type(
            cbuffer.name,
            [_member_tuple(member) for member in members],
            struct_members,
            (cbuffer.name,),
            constant_value,
        )
        if layout == HLSL_CBUFFER_LAYOUT:
            placements, _size = cbuffer_struct_placements(layout_type)
        else:
            placements, _size = block_struct_placements(layout_type, layout, root=True)
    except HlslLayoutError as error:
        raise _Unplaced(str(error)) from None
    planned = []
    for placement, member in zip(placements, members):
        planned.append(
            _place(
                placement.name,
                placement.layout_type,
                placement.offset,
                layout,
                _member_type(member),
            )
        )
    return planned


def _fields(planned, taken):
    """``(name, CrossGL type)`` of each struct field: the members, after ``u32`` fields that fill each gap."""
    fields = []
    cursor = 0
    pad = 0
    for member in sorted(planned, key=lambda m: m.offset):
        if member.offset < cursor:
            raise _Unplaced(f"'{member.name}' overlaps the member before it")
        gap = member.offset - cursor
        if gap % 4:
            raise _Unplaced(
                f"'{member.name}' starts {gap} bytes after the member before it"
            )
        for _word in range(gap // 4):
            while f"_pad{pad}" in taken:
                pad += 1
            fields.append((f"_pad{pad}", _primitive("uint")))
            pad += 1
        fields.append((member.name, member.storage))
        cursor = member.offset + member.size
    return fields


# Reads ------------------------------------------------------------------------------------------------------------
class _Sentinel(TypeNode):
    """Declares a block member in the walker's scope, so a read of it can be told from a shadowing local."""

    def __init__(self, block, member):
        super().__init__()
        self.block = block
        self.member = member


class _MemberReads(ScopedWalker):
    """Finds every identifier that reads a member of one of the lowered blocks."""

    def __init__(self, program, sentinels):
        self.sentinels = sentinels
        self.found = {}
        super().__init__(program)
        scope = self.scope
        scope.declare_cbuffer = self._declare_block
        for cbuffer in getattr(program, "cbuffers", None) or []:
            self._declare_block(cbuffer)

    def _declare_block(self, cbuffer):
        for member in _members(cbuffer):
            sentinel = self.sentinels.get((id(cbuffer), member.name))
            if sentinel is None:
                self.scope.declare(member.name, _member_type(member))
            else:
                self.scope.declare(member.name, sentinel)

    def visit(self, node):
        if isinstance(node, IdentifierNode):
            declared = self.scope.lookup(node.name)
            if isinstance(declared, _Sentinel):
                self.found[id(node)] = declared


def _literal(value, type_name="int"):
    return LiteralNode(value, _primitive(type_name))


def _read_field(sentinel):
    return MemberAccessNode(IdentifierNode(sentinel.block), sentinel.member.name)


def _components(expr, count):
    if count == 4:
        return expr
    return MemberAccessNode(expr, _COMPONENTS[:count])


def _register(sentinel, index):
    return ArrayAccessNode(_read_field(sentinel), index)


def _offset_index(index, scale, offset):
    if isinstance(index, LiteralNode) and isinstance(index.value, int):
        return _literal(index.value * scale + offset)
    scaled = index if scale == 1 else BinaryOpNode(index, "*", _literal(scale))
    return scaled if offset == 0 else BinaryOpNode(scaled, "+", _literal(offset))


def _matrix_read(sentinel, index):
    member = sentinel.member
    size, row_major = member.matrix
    vectors = [
        _components(
            _register(sentinel, _offset_index(index, member.registers, r)),
            member.components,
        )
        for r in range(member.registers)
    ]
    matrix = FunctionCallNode(IdentifierNode(f"mat{size}"), vectors)
    if row_major:
        return FunctionCallNode(IdentifierNode("transpose"), [matrix])
    return matrix


def _bool_value(words, components):
    """The bool (``components`` 1) or bvec value of ``u32`` words: ``x != 0u`` or ``notEqual(x, uvecN(0u))``."""
    if components > 1:
        zero = FunctionCallNode(
            IdentifierNode(f"uvec{components}"), [_literal(0, "uint")]
        )
        return FunctionCallNode(IdentifierNode("notEqual"), [words, zero])
    return BinaryOpNode(words, "!=", _literal(0, "uint"))


def _bool_read(sentinel):
    storage = sentinel.member.storage
    components = storage.size if isinstance(storage, VectorType) else 1
    return _bool_value(_read_field(sentinel), components)


class _Rewriter:
    def __init__(self, found, refuse):
        self.found = found
        self.refuse = refuse

    def identifier_read(self, node):
        sentinel = self.found[id(node)]
        member = sentinel.member
        if member.form == "direct":
            return _read_field(sentinel)
        if member.form == "bool":
            return _bool_read(sentinel)
        if member.count is None and member.matrix is not None:
            return _matrix_read(sentinel, _literal(0))
        self.refuse(
            f"resource: constant block member '{member.name}' is read as a whole array; the rust-gpu output "
            "holds it as 16-byte vectors and reads its elements only"
        )
        return _read_field(sentinel)

    def element_read(self, node):
        sentinel = self.found[id(node.array_expr)]
        member = sentinel.member
        index = self.rewrite(node.index_expr)
        if member.count is None:
            # A column of a column-major matrix is one register.
            return _components(_register(sentinel, index), member.components)
        if member.matrix is not None:
            if member.registers > 1 and not is_pure(index):
                self.refuse(
                    f"resource: constant block member '{member.name}' is indexed by an expression with side "
                    "effects, which the rust-gpu output would evaluate once per matrix register"
                )
            return _matrix_read(sentinel, index)
        element = _components(_register(sentinel, index), member.components)
        if member.boolean:
            return _bool_value(element, member.components)
        return element

    def is_member_read(self, node):
        return isinstance(node, IdentifierNode) and id(node) in self.found

    def rewrite(self, node):
        if node is None:
            return node
        if self.is_member_read(node):
            return self.identifier_read(node)
        if isinstance(node, ArrayAccessNode) and self.is_member_read(node.array_expr):
            member = self.found[id(node.array_expr)].member
            if member.form == "registers" and (
                member.count is not None or not member.matrix[1]
            ):
                return self.element_read(node)
        for field, key in list(child_slots(node)):
            if isinstance(node, FunctionCallNode) and field == "function":
                continue
            child = get_child(node, field, key)
            new = self.rewrite(child)
            if new is not child:
                set_child(node, field, key, new)
        return node


def _functions(ast):
    for function in getattr(ast, "functions", None) or []:
        if isinstance(function, FunctionNode):
            yield function
    for stage in program_stages(ast):
        for function in getattr(stage, "local_functions", None) or []:
            if isinstance(function, FunctionNode):
                yield function
        entry = getattr(stage, "entry_point", None)
        if isinstance(entry, FunctionNode):
            yield entry


def _taken_names(ast, referenced):
    names = set(referenced)
    for struct in _structs(ast).values():
        names.add(struct.name)
    for function in _functions(ast):
        names.add(function.name)
    for node in getattr(ast, "global_variables", None) or []:
        names.add(getattr(node, "name", None))
    for stage in program_stages(ast):
        for node in getattr(stage, "local_variables", None) or []:
            names.add(getattr(node, "name", None))
    # A block's own name names no value or type in CrossGL, so its struct may take it.
    for cbuffer in getattr(ast, "cbuffers", None) or []:
        names.update(member.name for member in _members(cbuffer))
    for stage in program_stages(ast):
        for cbuffer in getattr(stage, "local_cbuffers", None) or []:
            names.update(member.name for member in _members(cbuffer))
    names.discard(None)
    return names


def _fresh(base, taken):
    name = base
    suffix = 0
    while name in taken:
        suffix += 1
        name = f"{base}_{suffix}" if suffix > 1 else f"{base}_block"
    taken.add(name)
    return name


def _snake(name):
    out = []
    for index, char in enumerate(name):
        if (
            char.isupper()
            and index
            and (name[index - 1].islower() or name[index - 1].isdigit())
        ):
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _binding_attributes(cbuffer):
    return [
        attribute
        for attribute in getattr(cbuffer, "attributes", None) or []
        if str(getattr(attribute, "name", "")).lower() != BUFFER_LAYOUT_ATTRIBUTE
    ]


def lower_rust_gpu_constant_blocks(generator, ast) -> None:
    """Declare each constant block of ``ast`` as a struct and a ``uniform`` variable, and read members through it.

    A block the struct cannot hold is refused through ``generator.refuse_lowering`` and dropped from the program.
    """
    owners = [(ast, "cbuffers")] + [
        (stage, "local_cbuffers") for stage in program_stages(ast)
    ]
    blocks = [
        (owner, attribute, cbuffer)
        for owner, attribute in owners
        for cbuffer in list(getattr(owner, attribute, None) or [])
        if isinstance(cbuffer, StructNode)
    ]
    if not blocks:
        return
    from .rust_resource_bindings import referenced_identifier_names

    taken = _taken_names(ast, referenced_identifier_names(ast))
    constants = rust_integer_constants(ast)
    sentinels = {}
    declarations = []
    for owner, attribute, cbuffer in blocks:
        try:
            planned = _plan(ast, cbuffer, constants)
            fields = _fields(planned, {member.name for member in planned})
        except _Unplaced as reason:
            generator.refuse_lowering(
                f"resource: constant block '{cbuffer.name}': {reason}"
            )
            continue
        struct_name = _fresh(cbuffer.name, taken)
        variable_name = _fresh(_snake(cbuffer.name), taken)
        struct = StructNode(
            struct_name,
            [StructMemberNode(name, field_type) for name, field_type in fields],
        )
        variable = VariableNode(
            variable_name,
            NamedType(struct_name),
            attributes=[
                AttributeNode(a.name, list(a.arguments or []))
                for a in _binding_attributes(cbuffer)
            ],
            qualifiers=["uniform"],
        )
        setattr(variable, CONSTANT_BLOCK_FLAG, struct_name)
        for member in planned:
            sentinels[(id(cbuffer), member.name)] = _Sentinel(variable_name, member)
        declarations.append((owner, attribute, cbuffer, struct, variable))
    finder = _MemberReads(ast, sentinels)
    finder.run_program()
    rewriter = _Rewriter(finder.found, generator.refuse_lowering)
    for function in _functions(ast):
        rewriter.rewrite(function.body)
    for owner, attribute, cbuffer, struct, variable in declarations:
        ast.structs = list(getattr(ast, "structs", None) or []) + [struct]
        target = "global_variables" if owner is ast else "local_variables"
        setattr(owner, target, [variable] + list(getattr(owner, target, None) or []))
    for owner, attribute, _cbuffer in blocks:
        setattr(owner, attribute, [])
