"""Storage views of lane records: records of one 32-bit scalar type.

A lane record such as ``struct Pair { float re; float im; }`` has no padding:
it is its members' scalars, four bytes each, in declaration order. Two views
of storage whose elements are lane records are exact.

Byte-view copies. Tile loaders written for a generic element type ``T`` copy
storage in units of an over-aligned byte record::

    struct alignas(A) V { uchar v[N]; };
    *((threadgroup V*)(&dst[i])) = *((const device V*)(&src[i]));

When ``dst`` and ``src`` both point to elements of one lane record ``E`` and
``N`` is a whole number of ``E`` elements, the statement copies exactly the
bytes of the element copies ``dst[i + k] = src[i + k]`` for
``k < N / sizeof(E)``. Those copies are defined for every address, so they
need no alignment proof. The view's alignment must still divide ``sizeof(E)``
and must not exceed it, the rule the targets apply to byte views of scalar
storage. ``lower_record_byte_copies`` rewrites such statements on every
target. Statements that do not match exactly, including any whose address
expressions have side effects, are left for the target's own diagnostics.

Scalar views. A pointer to lane records viewed as a pointer to their scalar
(``(threadgroup const float*)pairs``) addresses lane ``j % n`` of record
``j / n``, where ``n`` is the lane count (``LaneView``, ``lane_load``).
"""

import copy
from dataclasses import dataclass

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerAccessNode,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .pointer_reinterpret import (
    _metal_aligned_byte_view_struct_contract,
    scalar_storage_layout,
)

_AMBIGUOUS = object()


def _attribute_names(node):
    names = set()
    for item in [
        *(getattr(node, "qualifiers", ()) or ()),
        *(getattr(node, "attributes", ()) or ()),
    ]:
        names.add(str(getattr(item, "name", item)).lower())
    return names


def lane_record_layout(struct_node):
    """Return ``(scalar_name, lane_count)`` for a padding-free lane record.

    A lane record has one or more members of one 32-bit, non-boolean scalar
    type and nothing that could change its layout: no attributes, generic
    parameters, bases, member attributes or member initializers. Its size is
    four bytes per member and it has no padding.
    """

    if not isinstance(struct_node, StructNode):
        return None
    if (
        getattr(struct_node, "attributes", None)
        or getattr(struct_node, "generic_params", None)
        or getattr(struct_node, "inheritance", None)
    ):
        return None
    members = [
        member
        for member in getattr(struct_node, "members", ()) or ()
        if "static" not in _attribute_names(member)
    ]
    if not members:
        return None
    scalars = set()
    for member in members:
        member_type = getattr(member, "member_type", None)
        if (
            not isinstance(member_type, PrimitiveType)
            or getattr(member, "attributes", None)
            or getattr(member, "default_value", None) is not None
        ):
            return None
        layout = scalar_storage_layout(member_type.name)
        if (
            layout is None
            or layout.bit_width != 32
            or getattr(member_type, "size_bits", None) not in {None, 32}
        ):
            return None
        scalars.add(layout.name)
    if len(scalars) != 1:
        return None
    return scalars.pop(), len(members)


def _structs(ast):
    structs = list(getattr(ast, "structs", ()) or ())
    stages = getattr(ast, "stages", None)
    stage_values = stages.values() if hasattr(stages, "values") else stages or ()
    for stage in stage_values:
        structs.extend(getattr(stage, "local_structs", ()) or ())
    by_name = {}
    for struct in structs:
        name = getattr(struct, "name", None)
        if isinstance(name, str):
            # Two records with one spelling cannot be resolved by name.
            by_name[name] = _AMBIGUOUS if name in by_name else struct
    return {
        name: struct for name, struct in by_name.items() if struct is not _AMBIGUOUS
    }


def _strip_reference(value):
    while isinstance(value, ReferenceType):
        value = value.referenced_type
    return value


class _Scope:
    """Declared types of one function's parameters, locals and the globals."""

    def __init__(self, function, globals_, structs):
        self.structs = structs
        self.types = dict(globals_)
        for parameter in getattr(function, "parameters", ()) or ():
            self.declare(parameter.name, getattr(parameter, "param_type", None))
        body = getattr(function, "body", None)
        if body is not None and hasattr(body, "walk"):
            for node in body.walk():
                if isinstance(node, VariableNode):
                    self.declare(node.name, getattr(node, "var_type", None))

    def declare(self, name, value):
        if not isinstance(name, str) or value is None:
            return
        previous = self.types.get(name)
        if previous is None:
            self.types[name] = value
        elif previous is not _AMBIGUOUS and repr(previous) != repr(value):
            self.types[name] = _AMBIGUOUS

    def lookup(self, name):
        value = self.types.get(name)
        return None if value is _AMBIGUOUS else value

    def member(self, record, name):
        record = _strip_reference(record)
        if not isinstance(record, NamedType) or record.generic_args:
            return None
        struct = self.structs.get(record.name)
        if struct is None:
            return None
        for member in getattr(struct, "members", ()) or ():
            if member.name == name and "static" not in _attribute_names(member):
                return member.member_type
        return None

    def value_type(self, expression):
        """The type of an lvalue expression, or ``None`` when unknown."""

        if isinstance(expression, IdentifierNode):
            return _strip_reference(self.lookup(expression.name))
        if isinstance(expression, ArrayAccessNode):
            owner = self.value_type(expression.array_expr)
            if isinstance(owner, PointerType):
                return _strip_reference(owner.pointee_type)
            if isinstance(owner, ArrayType):
                return _strip_reference(owner.element_type)
            return None
        if isinstance(expression, MemberAccessNode):
            return _strip_reference(
                self.member(self.value_type(expression.object_expr), expression.member)
            )
        if isinstance(expression, PointerAccessNode):
            owner = self.value_type(expression.pointer_expr)
            if not isinstance(owner, PointerType):
                return None
            return _strip_reference(self.member(owner.pointee_type, expression.member))
        if (
            isinstance(expression, UnaryOpNode)
            and expression.operator == "*"
            and not expression.is_postfix
        ):
            owner = self.value_type(expression.operand)
            if isinstance(owner, PointerType):
                return _strip_reference(owner.pointee_type)
        return None

    def pointee(self, expression):
        """The element type a pointer-valued expression addresses."""

        if (
            isinstance(expression, UnaryOpNode)
            and expression.operator == "&"
            and not expression.is_postfix
        ):
            return self.value_type(expression.operand)
        if isinstance(expression, BinaryOpNode) and expression.operator in {"+", "-"}:
            left = self.pointee(expression.left)
            if left is not None:
                return left
            return (
                self.pointee(expression.right) if expression.operator == "+" else None
            )
        owner = self.value_type(expression)
        if isinstance(owner, PointerType):
            return _strip_reference(owner.pointee_type)
        if isinstance(owner, ArrayType):
            return _strip_reference(owner.element_type)
        return None


def _has_side_effects(expression):
    for node in expression.walk():
        if isinstance(node, (AssignmentNode, FunctionCallNode)):
            return True
        if isinstance(node, UnaryOpNode) and node.operator in {"++", "--"}:
            return True
    return False


def _dereferenced_view(expression):
    if (
        isinstance(expression, UnaryOpNode)
        and expression.operator == "*"
        and not expression.is_postfix
        and isinstance(expression.operand, PointerReinterpretNode)
    ):
        return expression.operand
    return None


def _view_name(view):
    target = getattr(view, "target_type", None)
    pointee = getattr(target, "pointee_type", None)
    if not isinstance(target, PointerType) or not isinstance(pointee, NamedType):
        return None
    return None if pointee.generic_args else pointee.name


def _element(address, index):
    """``address[index]``, folding ``&base[i]`` into ``base[i + index]``."""

    offset = LiteralNode(index, PrimitiveType("int"))
    if (
        isinstance(address, UnaryOpNode)
        and address.operator == "&"
        and not address.is_postfix
        and isinstance(address.operand, ArrayAccessNode)
    ):
        base = copy.deepcopy(address.operand.array_expr)
        start = copy.deepcopy(address.operand.index_expr)
        return ArrayAccessNode(
            base, start if index == 0 else BinaryOpNode(start, "+", offset)
        )
    return ArrayAccessNode(copy.deepcopy(address), offset)


class _Lowering:
    def __init__(self, ast):
        self.structs = _structs(ast)
        self.lane_records = {
            name: layout
            for name, struct in self.structs.items()
            if (layout := lane_record_layout(struct)) is not None
        }
        self.byte_views = {
            name: contract
            for name, struct in self.structs.items()
            if (contract := _metal_aligned_byte_view_struct_contract(struct))
            is not None
        }
        self.globals = {}
        for variable in getattr(ast, "global_variables", ()) or ():
            name = getattr(variable, "name", None)
            if isinstance(name, str):
                self.globals[name] = getattr(variable, "var_type", None)

    def copies(self, statement, scope):
        assignment = (
            statement.expression
            if isinstance(statement, ExpressionStatementNode)
            else statement
        )
        if not isinstance(assignment, AssignmentNode) or assignment.operator != "=":
            return None
        destination = _dereferenced_view(assignment.target)
        source = _dereferenced_view(assignment.value)
        if destination is None or source is None:
            return None
        view = _view_name(destination)
        contract = self.byte_views.get(view)
        if contract is None or _view_name(source) != view:
            return None
        if not destination.target_type.is_mutable:
            return None
        addresses = (destination.expression, source.expression)
        if any(_has_side_effects(address) for address in addresses):
            return None
        elements = [scope.pointee(address) for address in addresses]
        if not all(
            isinstance(element, NamedType) and not element.generic_args
            for element in elements
        ):
            return None
        if elements[0].name != elements[1].name:
            return None
        layout = self.lane_records.get(elements[0].name)
        if layout is None:
            return None
        element_bytes = 4 * layout[1]
        alignment, byte_width = contract["alignment"], contract["byteWidth"]
        if (
            alignment > element_bytes
            or element_bytes % alignment
            or byte_width % element_bytes
        ):
            return None
        return [
            AssignmentNode(
                _element(destination.expression, index),
                _element(source.expression, index),
                "=",
            )
            for index in range(byte_width // element_bytes)
        ]

    def rewrite_block(self, block, scope):
        statements = getattr(block, "statements", None)
        if not isinstance(statements, list):
            return
        rewritten = []
        for statement in statements:
            replacement = self.copies(statement, scope)
            if replacement is None:
                rewritten.append(statement)
            else:
                rewritten.extend(
                    (
                        ExpressionStatementNode(item)
                        if isinstance(statement, ExpressionStatementNode)
                        else item
                    )
                    for item in replacement
                )
        block.statements = rewritten

    def run(self, ast):
        if not self.lane_records or not self.byte_views:
            return
        functions = [node for node in ast.walk() if isinstance(node, FunctionNode)]
        for function in functions:
            body = getattr(function, "body", None)
            if not isinstance(body, BlockNode):
                continue
            scope = _Scope(function, self.globals, self.structs)
            for node in list(body.walk()):
                if isinstance(node, BlockNode):
                    self.rewrite_block(node, scope)


def lower_record_byte_copies(ast):
    """Rewrite aligned byte-view copies of lane-record elements in place."""

    _Lowering(ast).run(ast)
    return ast


@dataclass(frozen=True)
class LaneView:
    """A ``target`` view of ``source`` lane records: ``ratio`` lanes each."""

    source: str
    target: str
    ratio: int
    members: tuple


def lane_view(struct_node, target_element):
    """The scalar view of ``struct_node`` lane records as ``target_element``.

    The view's element must be the record's own scalar type, so each lane is
    read as the member that holds it, without a bit conversion.
    """

    layout = lane_record_layout(struct_node)
    if layout is None or scalar_storage_layout(target_element) is None:
        return None
    if scalar_storage_layout(target_element).name != layout[0]:
        return None
    members = tuple(
        member.name
        for member in struct_node.members
        if "static" not in _attribute_names(member)
    )
    return LaneView(struct_node.name, target_element, layout[1], members)


def _position(value):
    # Positions are int64_t; typed operands keep every target's overloads exact.
    return FunctionCallNode(
        IdentifierNode("int64_t"), [LiteralNode(value, PrimitiveType("int"))]
    )


def _lane_member(view, storage, index, lane):
    record = ArrayAccessNode(
        copy.deepcopy(storage),
        BinaryOpNode(copy.deepcopy(index), "/", _position(view.ratio)),
    )
    return MemberAccessNode(record, view.members[lane])


def _is_lane(view, index, lane):
    # Positions in a view are never negative, so the remainder is the lane.
    remainder = (
        BinaryOpNode(copy.deepcopy(index), "&", _position(view.ratio - 1))
        if view.ratio & (view.ratio - 1) == 0
        else BinaryOpNode(copy.deepcopy(index), "%", _position(view.ratio))
    )
    return BinaryOpNode(remainder, "==", _position(lane))


def lane_load(view, storage, index):
    """Read lane ``index`` of ``storage``, an array of ``view.source`` records.

    ``index`` is evaluated once per lane, so callers pass a side-effect-free
    expression, as a helper's ``reference.offset + index`` is.
    """

    value = _lane_member(view, storage, index, view.ratio - 1)
    for lane in reversed(range(view.ratio - 1)):
        value = TernaryOpNode(
            _is_lane(view, index, lane), _lane_member(view, storage, index, lane), value
        )
    return value


def lane_store(view, storage, index, value, operator="="):
    """Apply ``operator`` with ``value`` to lane ``index`` of ``storage``.

    Each lane is a whole 32-bit member, so a store writes only that member
    and races with no store to another lane.
    """

    def assign(lane):
        return BlockNode(
            [
                AssignmentNode(
                    _lane_member(view, storage, index, lane),
                    copy.deepcopy(value),
                    operator,
                )
            ]
        )

    statement = assign(view.ratio - 1)
    for lane in reversed(range(view.ratio - 1)):
        statement = IfNode(_is_lane(view, index, lane), assign(lane), statement)
    return statement
