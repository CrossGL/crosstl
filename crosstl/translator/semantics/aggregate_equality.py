"""Whole-value ``==`` and ``!=`` on structures and arrays, written member by member for targets without them.

CrossGL keeps GLSL's meaning for equality on aggregates (ADR 0002): ``p == q`` on two structures, or on two arrays of
the same type, is one ``bool`` that is true when every member and every element compares equal, recursively, and
``p != q`` is its negation (GLSL 4.60 section 5.9). GLSL and GLSL ES have these operators, so their output keeps them.
No other target does: MSL and HLSL reject ``==`` on a structure or an array, WGSL and SPIR-V's ``OpFOrdEqual`` take
only scalars and vectors, Slang has no structure equality, and CUDA and HIP compare two arrays as pointers, which
compiles and gives a different answer.

:func:`lower_aggregate_equality` rewrites each such comparison into the conjunction of its leaves for every other
target: a scalar leaf is compared with the same operator, a vector leaf with ``all(a == b)`` or ``any(a != b)``
(CrossGL's ``==`` is componentwise on vectors), and a matrix leaf column by column. ``!=`` is the disjunction of the
leaves' ``!=``, which is the negation of ``==`` also for NaN components, because a float ``!=`` is the negation of its
``==``. ``&&`` and ``||`` keep the evaluation order of the members.

The operands are repeated once per leaf, so each must be free of side effects (:func:`.rewrite.is_pure`); a function
call or an assignment operand is refused with :class:`AggregateEqualityOperandError`. A leaf that has no equality
(samplers, textures, pointers, an array whose length is not a constant), an aggregate whose type the pass cannot
determine on one side when the other side is an aggregate, and an aggregate with more than
:data:`MAX_LEAF_COMPARISONS` leaves are refused with :class:`AggregateEqualityUnsupportedError`.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from .hub_types import call_name, type_from_node
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child
from .scoped_walk import ScopedWalker, parameter_type

__all__ = [
    "AggregateEqualityOperandError",
    "AggregateEqualityUnsupportedError",
    "MAX_LEAF_COMPARISONS",
    "lower_aggregate_equality",
]

# Targets whose ``==`` and ``!=`` already compare structures and arrays as whole values (GLSL 4.60 section 5.9,
# GLSL ES 3.20 section 5.9), and CrossGL itself.
NATIVE_AGGREGATE_EQUALITY = frozenset(
    {"opengl", "glsl", "webgl", "gles", "essl", "cgl", "crossgl"}
)

# The leaves one comparison may expand to. A larger aggregate would need a loop, which an expression cannot hold.
MAX_LEAF_COMPARISONS = 256

_EQUALITY = {"==": ("&&", "all"), "!=": ("||", "any")}
_GLSL_MATRIX = re.compile(r"[dbiu]?mat([234])(?:x[234])?")
# Parameter qualifiers that place an array parameter in an address space, which makes it a pointer.
_ADDRESS_SPACES = frozenset({"thread", "device", "threadgroup", "constant"})


class AggregateEqualityOperandError(TranslationDiagnostic):
    """A structure or array comparison whose operand would be evaluated once per member."""

    code = CODE_PREFIX + "aggregate-equality-impure-operand"


class AggregateEqualityUnsupportedError(TranslationDiagnostic):
    """A structure or array comparison that has no member-by-member form on the target."""

    code = CODE_PREFIX + "aggregate-equality-unsupported"


def lower_aggregate_equality(program, target) -> int:
    """Rewrite structure and array ``==``/``!=`` in ``program`` for ``target``, in place; return how many."""
    if str(target or "").lower() in NATIVE_AGGREGATE_EQUALITY:
        return 0
    if not isinstance(program, A.ShaderNode):
        return 0
    walker = _AggregateEquality(program, str(target or ""))
    walker.run_program()
    return walker.rewritten


class _AggregateEquality(ScopedWalker):
    def __init__(self, program, target):
        super().__init__(program)
        self.target = target
        self.rewritten = 0
        self.pointer_parameter_types = []
        self.constants = {
            constant.name: constant.value
            for constant in getattr(program, "constants", None) or []
            if getattr(constant, "name", None)
        }

    def _walk(self, node):
        if isinstance(node, A.VariableNode):
            # The walker declares a variable instead of visiting it; its initializer is rewritten first.
            for field, key in list(child_slots(node)):
                self._walk(get_child(node, field, key))
            self.visit(node)
            self.scope.declare(node.name, node.var_type)
            return
        super()._walk(node)

    def visit(self, node):
        # Children are visited first, so an inner comparison is rewritten before the one that contains it.
        for field, key in list(child_slots(node)):
            replacement = self._replacement(get_child(node, field, key))
            if replacement is not None:
                set_child(node, field, key, replacement)
                self.rewritten += 1

    def run(self, function, stage):
        # MSL follows C++, which adjusts an array parameter to a pointer; one in an address space (``thread float
        # v[4]``) arrives as such, so ``==`` on it compares addresses, not elements.
        self.pointer_parameter_types = [
            parameter_type(parameter)
            for parameter in getattr(function, "parameters", None) or []
            if isinstance(_strip_reference(parameter_type(parameter)), A.ArrayType)
            and _ADDRESS_SPACES.intersection(
                getattr(parameter, "qualifiers", None) or ()
            )
        ]
        super().run(function, stage)

    def _is_pointer_operand(self, expr):
        if isinstance(expr, A.BinaryOpNode) and expr.op in {"+", "-"}:
            return self._is_pointer_operand(expr.left) or self._is_pointer_operand(
                expr.right
            )
        if not isinstance(expr, A.IdentifierNode):
            return False
        declared = self.scope.node_type(expr)
        return any(declared is pointer for pointer in self.pointer_parameter_types)

    # Detection ------------------------------------------------------------------------------------------------
    def _replacement(self, node):
        if not isinstance(node, A.BinaryOpNode) or node.op not in _EQUALITY:
            return None
        if self._is_pointer_operand(node.left) or self._is_pointer_operand(node.right):
            # A pointer comparison is left to the generator, which emits or refuses pointer values itself.
            return None
        left_type, right_type = self._operand_type(node.left), self._operand_type(
            node.right
        )
        left_aggregate = self._is_aggregate(left_type)
        right_aggregate = self._is_aggregate(right_type)
        if not (left_aggregate or right_aggregate):
            return None
        location = getattr(node, "source_location", None)
        if not (left_aggregate and right_aggregate):
            self._refuse(
                f"'{node.op}' compares a {_describe(left_type or right_type)} with an operand whose type is not known",
                location,
            )
        if repr(_strip_reference(left_type)) != repr(_strip_reference(right_type)):
            self._refuse(
                f"'{node.op}' compares a {_describe(left_type)} with a {_describe(right_type)}",
                location,
            )
        if not (is_pure(node.left) and is_pure(node.right)):
            raise AggregateEqualityOperandError(
                f"'{node.op}' on a {_describe(left_type)} compares every member, so its operands are repeated; "
                "they must be free of side effects. Store the operand in a local first",
                span=_span(location),
                target=self.target,
            )
        terms = []
        self._leaves(node.left, node.right, left_type, node.op, terms, location)
        if not terms:
            self._refuse(
                f"'{node.op}' compares a {_describe(left_type)} that has no members",
                location,
            )
        combine = _EQUALITY[node.op][0]
        result = terms[0]
        for term in terms[1:]:
            result = A.BinaryOpNode(result, combine, term)
            _locate(result, location)
        return result

    def _operand_type(self, expr):
        declared = self.scope.node_type(expr)
        if declared is not None:
            return declared
        if isinstance(expr, A.ConstructorNode):
            return getattr(expr, "constructor_type", None)
        if isinstance(expr, A.FunctionCallNode):
            name = call_name(expr)
            if name in self.scope.structs:
                return A.NamedType(name)
            return self.scope.functions.get(name)
        return None

    def _is_aggregate(self, type_node):
        type_node = _strip_reference(type_node)
        if isinstance(type_node, A.ArrayType):
            return True
        return self._struct_members(type_node) is not None

    def _struct_members(self, type_node):
        if isinstance(type_node, (A.NamedType, A.PrimitiveType)):
            if getattr(type_node, "generic_args", None):
                return None
            return self.scope.structs.get(type_node.name)
        return None

    # Expansion ------------------------------------------------------------------------------------------------
    def _leaves(self, left, right, type_node, op, terms, location):
        """Append the comparisons of every leaf of ``left`` and ``right``, of type ``type_node``, to ``terms``."""
        type_node = _strip_reference(type_node)
        members = self._struct_members(type_node)
        if members is not None:
            for name, member_type in members.items():
                self._leaves(
                    _member(left, name, location),
                    _member(right, name, location),
                    member_type,
                    op,
                    terms,
                    location,
                )
            return
        if isinstance(type_node, A.ArrayType):
            length = self._array_length(type_node.size)
            if length is None:
                self._refuse(
                    f"'{op}' compares an array whose length is not a constant",
                    location,
                )
            for index in range(length):
                self._leaves(
                    _element(left, index, location),
                    _element(right, index, location),
                    type_node.element_type,
                    op,
                    terms,
                    location,
                )
            return
        columns = _matrix_columns(type_node)
        if columns is not None:
            for column in range(columns):
                self._append(
                    terms,
                    _reduce(
                        _EQUALITY[op][1],
                        _element(left, column, location),
                        op,
                        _element(right, column, location),
                        location,
                    ),
                    location,
                )
            return
        value = type_from_node(type_node)
        if value is None:
            self._refuse(
                f"'{op}' compares a member of type {_describe(type_node)}, which has no equality",
                location,
            )
        if value.lanes > 1:
            term = _reduce(_EQUALITY[op][1], left, op, right, location)
        else:
            term = _locate(A.BinaryOpNode(clone(left), op, clone(right)), location)
        self._append(terms, term, location)

    def _append(self, terms, term, location):
        if len(terms) >= MAX_LEAF_COMPARISONS:
            self._refuse(
                f"a structure or array comparison expands to more than {MAX_LEAF_COMPARISONS} member comparisons",
                location,
            )
        terms.append(term)

    def _array_length(self, size, seen=frozenset()):
        if isinstance(size, bool):
            return None
        if isinstance(size, int):
            return size if size >= 0 else None
        if isinstance(size, A.LiteralNode):
            return self._array_length(size.value, seen)
        if isinstance(size, str) and size.isdigit():
            return int(size)
        if (
            isinstance(size, A.IdentifierNode)
            and size.name in self.constants
            and size.name not in seen
        ):
            return self._array_length(self.constants[size.name], seen | {size.name})
        return None

    def _refuse(self, reason, location):
        raise AggregateEqualityUnsupportedError(
            reason, span=_span(location), target=self.target
        )


def _strip_reference(type_node):
    while isinstance(type_node, A.ReferenceType):
        type_node = type_node.referenced_type
    return type_node


def _matrix_columns(type_node):
    if isinstance(type_node, A.MatrixType):
        return int(type_node.cols) if str(type_node.cols).isdigit() else None
    name = getattr(type_node, "name", type_node if isinstance(type_node, str) else "")
    match = _GLSL_MATRIX.fullmatch(str(name or ""))
    return int(match.group(1)) if match else None


def _member(expr, name, location):
    return _locate(A.MemberAccessNode(clone(expr), name), location)


def _element(expr, index, location):
    literal = A.LiteralNode(index, A.PrimitiveType("int"))
    return _locate(A.ArrayAccessNode(clone(expr), literal), location)


def _reduce(reduction, left, op, right, location):
    comparison = _locate(A.BinaryOpNode(clone(left), op, clone(right)), location)
    return _locate(call(reduction, [comparison]), location)


def _locate(node, location):
    if location is not None:
        node.source_location = location
    return node


def _describe(type_node):
    type_node = _strip_reference(type_node)
    if isinstance(type_node, A.ArrayType):
        return f"array of {_describe(type_node.element_type)}"
    name = getattr(type_node, "name", None)
    if name:
        return f"'{name}'"
    return repr(type_node) if type_node is not None else "value of unknown type"


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        line, column, file = (location.get(key) for key in ("line", "column", "file"))
    else:
        line, column, file = (
            getattr(location, key, None) for key in ("line", "column", "file")
        )
    if not isinstance(line, int):
        return None
    return SourceSpan(
        file=file, line=line, column=column if isinstance(column, int) else None
    )
