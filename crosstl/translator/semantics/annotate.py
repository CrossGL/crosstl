"""Annotation: rewrite operations whose source rule is defined into rule-carrying hub intrinsics.

Expression operations become calls (``a + b`` on ``int`` from GLSL becomes ``__cgl_wrap_add(a, b)``, and ``a / b``
on ``uint`` from HLSL becomes ``__cgl_udiv_ones(a, b)``); the call keeps the original node in
``annotations["cgl_rule_origin"]`` so native lowering restores it exactly. Statement operations (``x += e``, ``++x``,
``x /= e``) keep their node and gain a :class:`RuleStatement` in ``annotations["cgl_rule_statement"]``. Operations
proved not to need the rule are left alone: literal-only arithmetic whose exact result fits, the update of a for-loop
counter bounded by a strict comparison, and division by a nonzero literal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import ast as A
from .conversions import resolve as resolve_conversion
from .helpers import is_helper
from .hub_types import (
    BOOL,
    INT32,
    UINT32,
    HubType,
    TypeScope,
    arithmetic_result,
    call_name,
    is_pure_callee,
    iter_functions,
)
from .rewrite import child_slots, get_child, set_child
from .rule_table import (
    BINARY_UDIV_ZERO_INTRINSIC,
    BINARY_WRAP_INTRINSIC,
    CONVERT_SAT,
    CONVERT_SAT_GENERIC,
    INTRINSIC_FAMILY,
    UDIV_ZERO,
    WRAP,
    source_families,
)

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*"}
_DIVISION_COMPOUND = {"/=": "/", "%=": "%"}


@dataclass(frozen=True)
class RuleStatement:
    """A statement-shaped operation that carries a rule: ``x op= e``, ``++x`` or ``--x``.

    ``conversions`` convert the target and the value to ``result_type`` before the polyfill; ``back_conversion``
    converts the result to the target's type.
    """

    intrinsic: str
    conversions: tuple[HubType | None, HubType | None] = (None, None)
    result_type: HubType | None = None
    back_conversion: HubType | None = None


def is_int32(value) -> bool:
    return value is not None and value.kind == "i" and value.bits == 32


def is_uint32(value) -> bool:
    return value is not None and value.kind == "u" and value.bits == 32


def _integer_operands(*operands) -> bool:
    """Every operand is an integer or a boolean of at most 32 bits."""
    return all(
        operand.kind in ("i", "u", "b") and operand.bits <= 32 for operand in operands
    )


def wrap32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value > INT32_MAX else value


class Annotator:
    def __init__(self, program, source, report):
        self.program = program
        self.families = source_families(source)
        self.scope = TypeScope(program)
        self.report = report
        self.constants: dict[str, int] = {}
        self._proven: set[int] = set()

    # Program structure ----------------------------------------------------------------------------------------
    def run(self):
        if not self.families and not _has_untyped_intrinsics(self.program):
            return
        program, scope = self.program, self.scope
        for constant in getattr(program, "constants", None) or []:
            scope.declare(constant.name, constant.const_type)
            value = self.constant_value(constant.value)
            value_type = self.scope.infer(constant.value) or INT32
            if value is not None and (is_int32(value_type) or is_uint32(value_type)):
                self.constants[constant.name] = value
        self._declare_variables(getattr(program, "global_variables", None) or [])
        for cbuffer in getattr(program, "cbuffers", None) or []:
            scope.declare_cbuffer(cbuffer)
        for function in getattr(program, "functions", None) or []:
            self.function(function)
        stages = getattr(program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            scope.push()
            self._declare_variables(getattr(stage, "local_variables", None) or [])
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
            for function in list(getattr(stage, "local_functions", None) or []) + [
                getattr(stage, "entry_point", None)
            ]:
                if function is not None:
                    self.function(function)
            scope.pop()

    def _declare_variables(self, variables):
        for variable in variables:
            self.scope.declare(variable.name, variable.var_type)
            self.scope.declare_block_members(variable)

    def function(self, function):
        if is_helper(function):
            return
        self.scope.push()
        for parameter in function.parameters or []:
            self.scope.declare(parameter.name, parameter.param_type)
        if function.body is not None:
            self.visit(function.body)
        self.scope.pop()

    # Traversal ------------------------------------------------------------------------------------------------
    def visit(self, node):
        """Annotate ``node``'s subtree depth-first and return its replacement (``node`` itself if unchanged)."""
        if isinstance(node, (A.FunctionNode, A.LambdaNode, A.StructNode)):
            return node
        if isinstance(node, A.BlockNode):
            self.scope.push()
            self.visit_children(node)
            self.scope.pop()
            return node
        if isinstance(node, A.ForNode):
            return self.visit_for(node)
        if isinstance(node, A.VariableNode):
            self.visit_children(node)
            self.scope.declare(node.name, node.var_type)
            return node
        self.visit_children(node)
        return self.rewrite(node)

    def visit_children(self, node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self.visit(child)
            if replacement is not child:
                set_child(node, field, key, replacement)

    def visit_for(self, node):
        self.scope.push()
        if node.init is not None:
            replacement = self.visit(node.init)
            if replacement is not node.init:
                set_child(node, "init", None, replacement)
        proven = self.counter_update_cannot_overflow(node)
        if proven is not None:
            self._proven.add(id(proven))
        for field in ("condition", "update", "body"):
            child = getattr(node, field)
            if child is not None:
                replacement = self.visit(child)
                if replacement is not child:
                    set_child(node, field, None, replacement)
        self.scope.pop()
        return node

    # Rewrites -------------------------------------------------------------------------------------------------
    def rewrite(self, node):
        # A legalization pass generated this operation and proved it exact.
        if node.annotations.get("cgl_rule_exempt"):
            return node
        wrap, udiv_zero = WRAP in self.families, UDIV_ZERO in self.families
        if isinstance(node, A.BinaryOpNode):
            if wrap and node.op in BINARY_WRAP_INTRINSIC:
                return self.rewrite_binary(node)
            if udiv_zero and node.op in BINARY_UDIV_ZERO_INTRINSIC:
                return self.rewrite_division(node)
        elif isinstance(node, A.UnaryOpNode):
            if wrap and node.op == "-":
                return self.rewrite_negate(node)
            if wrap and node.op in ("++", "--"):
                self.annotate_step(node)
        elif isinstance(node, A.AssignmentNode):
            if wrap and node.operator in _COMPOUND:
                self.annotate_compound(node)
            elif udiv_zero and node.operator in _DIVISION_COMPOUND:
                self.annotate_division_compound(node)
        elif isinstance(node, A.FunctionCallNode):
            return self.type_intrinsic(node)
        return node

    def type_intrinsic(self, node):
        """A rule intrinsic written in the source text gets its result type, which helper lowering needs.

        A generic saturated conversion is resolved by its operand's type instead (see :mod:`.conversions`).
        """
        name = call_name(node)
        if name in CONVERT_SAT_GENERIC:
            arguments = node.arguments or []
            operand = self.scope.infer(arguments[0]) if len(arguments) == 1 else None
            replacement = resolve_conversion(node, operand)
            if replacement is not node:
                self.report.count(CONVERT_SAT)
            return replacement
        if name in INTRINSIC_FAMILY and "cgl_rule_type" not in node.annotations:
            result = self.scope.infer(node)
            if result is not None:
                node.annotations["cgl_rule_type"] = result
        return node

    def rewrite_binary(self, node):
        left, right = self.scope.infer(node.left), self.scope.infer(node.right)
        result = arithmetic_result(left, right)
        if result is None:
            if left is None or right is None:
                self.report.unresolved += 1
            return node
        if not is_int32(result) or not self._wrap_operands(left, right):
            return node
        folded = self.fold(node)
        if folded is not None:
            return folded
        self.report.count(WRAP)
        replacement = A.FunctionCallNode(
            A.IdentifierNode(BINARY_WRAP_INTRINSIC[node.op]), [node.left, node.right]
        )
        replacement.annotations["cgl_rule_origin"] = node
        replacement.annotations["cgl_rule_type"] = result
        replacement.annotations["cgl_rule_convert"] = (
            _conversion(left),
            _conversion(right),
        )
        return replacement

    def rewrite_negate(self, node):
        operand = self.scope.infer(node.operand)
        if operand is None:
            self.report.unresolved += 1
            return node
        if not is_int32(operand):
            return node
        folded = self.fold(node)
        if folded is not None:
            return folded
        self.report.count(WRAP)
        replacement = A.FunctionCallNode(
            A.IdentifierNode("__cgl_wrap_neg"), [node.operand]
        )
        replacement.annotations["cgl_rule_origin"] = node
        replacement.annotations["cgl_rule_type"] = operand
        replacement.annotations["cgl_rule_convert"] = (None,)
        return replacement

    def annotate_step(self, node):
        if id(node) in self._proven:
            return
        operand = self.scope.infer(node.operand)
        if operand is None:
            self.report.unresolved += 1
        elif is_int32(operand):
            self.report.count(WRAP)
            intrinsic = "__cgl_wrap_add" if node.op == "++" else "__cgl_wrap_sub"
            node.annotations["cgl_rule_statement"] = RuleStatement(
                intrinsic, result_type=operand
            )

    def annotate_compound(self, node):
        if id(node) in self._proven:
            return
        target, value = self.scope.infer(node.target), self.scope.infer(node.value)
        if target is None or value is None:
            self.report.unresolved += 1
            return
        result = arithmetic_result(target, value)
        if not is_int32(target) or not is_int32(result):
            return
        if not self._wrap_operands(target, value) or value.lanes > target.lanes:
            return
        self.report.count(WRAP)
        intrinsic = BINARY_WRAP_INTRINSIC[_COMPOUND[node.operator]]
        node.annotations["cgl_rule_statement"] = RuleStatement(
            intrinsic, (None, _conversion(value)), target
        )

    def rewrite_division(self, node):
        """Unsigned 32-bit ``/`` and ``%`` whose divisor may be zero become ``__cgl_udiv_ones``/``__cgl_urem_ones``."""
        left, right = self.scope.infer(node.left), self.scope.infer(node.right)
        result = arithmetic_result(left, right)
        if result is None:
            if left is None or right is None:
                self.report.unresolved += 1
            return node
        if not is_uint32(result) or not _integer_operands(left, right):
            return node
        if self.nonzero(node.right):
            return node
        self.report.count(UDIV_ZERO)
        replacement = A.FunctionCallNode(
            A.IdentifierNode(BINARY_UDIV_ZERO_INTRINSIC[node.op]),
            [node.left, node.right],
        )
        replacement.annotations["cgl_rule_origin"] = node
        replacement.annotations["cgl_rule_type"] = result
        replacement.annotations["cgl_rule_convert"] = (
            _converted(left, result),
            _converted(right, result),
        )
        return replacement

    def annotate_division_compound(self, node):
        target, value = self.scope.infer(node.target), self.scope.infer(node.value)
        if target is None or value is None:
            self.report.unresolved += 1
            return
        result = arithmetic_result(target, value)
        if not is_uint32(result) or not _integer_operands(target, value):
            return
        if target.bits != 32 or target.lanes != result.lanes:
            return
        if self.nonzero(node.value):
            return
        self.report.count(UDIV_ZERO)
        intrinsic = BINARY_UDIV_ZERO_INTRINSIC[_DIVISION_COMPOUND[node.operator]]
        node.annotations["cgl_rule_statement"] = RuleStatement(
            intrinsic,
            (_converted(target, result), _converted(value, result)),
            result,
            None if target == result else target,
        )

    def nonzero(self, divisor) -> bool:
        """The divisor is a literal-only expression whose value is not zero modulo 2^32."""
        value = self.constant_value(divisor)
        return value is not None and value % (1 << 32) != 0

    @staticmethod
    def _wrap_operands(left, right) -> bool:
        """At least one operand is a 32-bit signed integer and the other converts to one exactly.

        Booleans and narrower integers are converted to ``int`` first (HLSL ``int + bool``). An operation on two
        such operands cannot overflow 32 bits, except a product of two 16-bit unsigned values, which belongs to
        the narrow-integer rules.
        """
        convertible = all(
            is_int32(operand)
            or operand.kind == "b"
            or (operand.kind in ("i", "u") and operand.bits < 32)
            for operand in (left, right)
        )
        return convertible and (is_int32(left) or is_int32(right))

    # Proofs ---------------------------------------------------------------------------------------------------
    def constant_value(self, expr):
        """The exact integer value of a literal-only expression, or ``None``."""
        if isinstance(expr, A.LiteralNode):
            value = expr.value
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value
        if isinstance(expr, A.IdentifierNode) and expr.name in self.constants:
            if self.scope.lookup(
                expr.name
            ) is not None and not self._is_global_constant(expr.name):
                return None
            return self.constants[expr.name]
        if isinstance(expr, A.UnaryOpNode) and expr.op in ("-", "+"):
            value = self.constant_value(expr.operand)
            return None if value is None else (-value if expr.op == "-" else value)
        if isinstance(expr, A.BinaryOpNode) and expr.op in ("+", "-", "*"):
            left, right = self.constant_value(expr.left), self.constant_value(
                expr.right
            )
            if left is None or right is None:
                return None
            exact = {"+": left + right, "-": left - right, "*": left * right}[expr.op]
            return wrap32(exact)
        if isinstance(expr, A.FunctionCallNode) and expr.annotations.get(
            "cgl_rule_origin"
        ):
            return self.constant_value(expr.annotations["cgl_rule_origin"])
        return None

    def _is_global_constant(self, name):
        declared = self.scope.lookup(name)
        return any(
            c.name == name and c.const_type is declared
            for c in getattr(self.program, "constants", None) or []
        )

    def fold(self, node):
        """Literal-only arithmetic: keep it when the exact result fits, else replace it with the wrapped value."""
        operands = (
            (node.left, node.right)
            if isinstance(node, A.BinaryOpNode)
            else (node.operand,)
        )
        values = [self.constant_value(operand) for operand in operands]
        if any(value is None for value in values):
            return None
        if isinstance(node, A.UnaryOpNode):
            exact = -values[0]
        else:
            exact = {"+": sum(values), "-": values[0] - values[1]}.get(node.op)
            if exact is None:
                exact = values[0] * values[1]
        if INT32_MIN <= exact <= INT32_MAX:
            return node
        self.report.count(WRAP)
        return int32_literal(wrap32(exact))

    def counter_update_cannot_overflow(self, loop):
        """The update node of ``for (...; i < e; ++i)`` when ``e`` is signed and nothing else writes ``i``.

        The condition held when the iteration started and the body did not change ``i``, so
        ``i + 1 <= e <= INT_MAX`` at the update (``i - 1 >= e >= INT_MIN`` for ``i > e; --i``).
        """
        counter, step, update = _counter_step(loop.update)
        if counter is None or self.scope.infer(A.IdentifierNode(counter)) != INT32:
            return None
        bound = _strict_bound(loop.condition, counter, step)
        if bound is None or self.scope.infer(bound) != INT32:
            return None
        if _writes(loop.condition, counter) or _writes(loop.body, counter):
            return None
        return update


def int32_literal(value: int):
    if value >= 0:
        return A.LiteralNode(value, A.PrimitiveType("int"))
    if value == INT32_MIN:
        return A.BinaryOpNode(
            A.UnaryOpNode("-", A.LiteralNode(INT32_MAX, A.PrimitiveType("int"))),
            "-",
            A.LiteralNode(1, A.PrimitiveType("int")),
        )
    return A.UnaryOpNode("-", A.LiteralNode(-value, A.PrimitiveType("int")))


def _conversion(operand):
    """Operands that are not ``int`` are converted before bit reinterpretation (HLSL ``int + bool``)."""
    if operand is not None and not is_int32(operand):
        return INT32.with_lanes(operand.lanes)
    return None


def _converted(operand, result):
    """The conversion of an operand to the operation's ``uint`` type: a scalar is also broadcast to its lanes."""
    if operand == result:
        return None
    return UINT32.with_lanes(result.lanes)


def _has_untyped_intrinsics(program) -> bool:
    return any(
        isinstance(node, A.FunctionCallNode)
        and call_name(node) in INTRINSIC_FAMILY
        and "cgl_rule_type" not in node.annotations
        for node in program.walk()
    )


def _counter_step(update):
    """``(name, +1 | -1, node)`` for ``++i``, ``i++``, ``--i``, ``i--``, ``i += 1`` and ``i -= 1``."""
    if isinstance(update, A.ExpressionStatementNode):
        update = update.expression
    if isinstance(update, A.UnaryOpNode) and update.op in ("++", "--"):
        if isinstance(update.operand, A.IdentifierNode):
            return update.operand.name, 1 if update.op == "++" else -1, update
    if (
        isinstance(update, A.AssignmentNode)
        and update.operator in ("+=", "-=")
        and isinstance(update.target, A.IdentifierNode)
        and isinstance(update.value, A.LiteralNode)
        and update.value.value == 1
        and not isinstance(update.value.value, bool)
    ):
        return update.target.name, 1 if update.operator == "+=" else -1, update
    return None, 0, None


def _strict_bound(condition, counter, step):
    """The bound ``e`` of ``i < e`` (step +1) or ``i > e`` (step -1), also as a conjunct of ``&&``."""
    if not isinstance(condition, A.BinaryOpNode):
        return None
    if condition.op == "&&":
        return _strict_bound(condition.left, counter, step) or _strict_bound(
            condition.right, counter, step
        )
    left_is_counter = (
        isinstance(condition.left, A.IdentifierNode) and condition.left.name == counter
    )
    right_is_counter = (
        isinstance(condition.right, A.IdentifierNode)
        and condition.right.name == counter
    )
    upper = {"<": "left", ">": "right"}
    lower = {">": "left", "<": "right"}
    table = upper if step > 0 else lower
    side = table.get(condition.op)
    if side == "left" and left_is_counter:
        return condition.right
    if side == "right" and right_is_counter:
        return condition.left
    return None


def _root_name(expr):
    while isinstance(expr, (A.ArrayAccessNode, A.MemberAccessNode, A.SwizzleNode)):
        expr = (
            getattr(expr, "array_expr", None)
            or getattr(expr, "object_expr", None)
            or getattr(expr, "vector_expr", None)
        )
    return expr.name if isinstance(expr, A.IdentifierNode) else None


def _writes(node, name) -> bool:
    """Whether ``node`` may write ``name``: assignment, increment, address-of, a redeclaration, or a call that
    receives it as an argument (it may be an ``out`` or ``inout`` parameter), unless the callee is a value
    constructor or a pure built-in."""
    if node is None:
        return False
    for item in node.walk():
        if isinstance(item, A.AssignmentNode) and _root_name(item.target) == name:
            return True
        if isinstance(item, A.UnaryOpNode) and item.op in ("++", "--", "&"):
            if _root_name(item.operand) == name:
                return True
        if isinstance(item, (A.VariableNode, A.ParameterNode)) and item.name == name:
            return True
        if isinstance(item, A.FunctionCallNode) and not is_pure_callee(call_name(item)):
            if any(_root_name(argument) == name for argument in item.arguments or []):
                return True
    return False


__all__ = ["Annotator", "BOOL", "INT32", "iter_functions"]
