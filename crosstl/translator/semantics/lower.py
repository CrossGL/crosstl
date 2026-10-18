"""Lowering: discharge each rule-carrying hub intrinsic as the target's rule table entry says.

``native`` restores the original operation, ``polyfill`` instantiates the CrossGL template from ``POLYFILLS`` with
the operands or calls the helper function from ``HELPERS`` (injected into the program, see :mod:`.helpers`), and
``refuse`` raises :class:`RuleInexpressibleError`. Calls of the intrinsics written in CrossGL
source text are lowered the same way, without an original node to restore.
"""

from __future__ import annotations

import functools

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .helpers import HelperSet
from .hub_types import call_name
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child
from .rule_table import (
    HELPERS,
    INTRINSIC_FAMILY,
    INTRINSICS,
    POLYFILLS,
    target_entry,
)


class RuleInexpressibleError(TranslationDiagnostic):
    """The target cannot express an operation rule that the source language defines."""

    code = CODE_PREFIX + "rule-inexpressible"


_STATEMENT_FIELDS = frozenset(
    {"statements", "body", "then_branch", "else_branch", "default_case"}
)


class Lowerer:
    def __init__(self, program, target, report):
        self.program = program
        self.target = target
        self.report = report
        self.helpers = HelperSet(program, target, RuleInexpressibleError)

    def run(self):
        self.visit(self.program, None, None)
        self.helpers.inject()

    def visit(self, node, parent, field):
        for slot_field, key in list(child_slots(node)):
            child = get_child(node, slot_field, key)
            replacement = self.visit(child, node, slot_field)
            if replacement is not child:
                set_child(node, slot_field, key, replacement)
        if isinstance(node, A.FunctionCallNode) and call_name(node) in INTRINSIC_FAMILY:
            return self.lower_call(node)
        statement = node.annotations.get("cgl_rule_statement")
        if statement is not None:
            return self.lower_statement(node, statement, parent, field)
        return node

    # Entries --------------------------------------------------------------------------------------------------
    def entry(self, intrinsic):
        family = INTRINSIC_FAMILY[intrinsic]
        entry = target_entry(family, self.target)
        if entry.kind == "refuse":
            raise RuleInexpressibleError(
                f"{self.report.source or 'the source'} defines the {family} rule for {intrinsic}, and "
                f"{self.target} cannot express it: {entry.reason}",
                target=self.target,
            )
        if entry.configuration:
            self.report.configurations.add(entry.configuration)
        self.report.discharged[entry.kind] = (
            self.report.discharged.get(entry.kind, 0) + 1
        )
        return entry

    def lower_call(self, node):
        intrinsic = call_name(node)
        operator, arity = INTRINSICS[intrinsic]
        arguments = list(node.arguments or [])
        if len(arguments) != arity:
            raise RuleInexpressibleError(
                f"{intrinsic} takes {arity} operand(s), got {len(arguments)}",
                target=self.target,
            )
        entry = self.entry(intrinsic)
        if entry.kind == "native":
            return _native(node, operator, arguments)
        conversions = node.annotations.get("cgl_rule_convert") or (None,) * arity
        operands = [_convert(a, c) for a, c in zip(arguments, conversions)]
        return self.polyfill(intrinsic, operands, node.annotations.get("cgl_rule_type"))

    def polyfill(self, intrinsic, operands, result_type):
        """The polyfill of ``intrinsic`` over ``operands``: a template instance or a call of an injected helper."""
        if intrinsic not in HELPERS:
            return instantiate(intrinsic, operands)
        if result_type is None:
            raise RuleInexpressibleError(
                f"the operand types of {intrinsic} could not be inferred, so its helper for {self.target} "
                "cannot be chosen",
                target=self.target,
            )
        return self.helpers.call(intrinsic, result_type.lanes, operands)

    def lower_statement(self, node, statement, parent, field):
        entry = self.entry(statement.intrinsic)
        del node.annotations["cgl_rule_statement"]
        if entry.kind == "native":
            return node
        if isinstance(node, A.AssignmentNode):
            return self.lower_compound(node, statement)
        return self.lower_step(node, statement, parent, field)

    def lower_compound(self, node, statement):
        if not is_pure(node.target):
            raise RuleInexpressibleError(
                f"the compound assignment {node.operator} to an lvalue with side effects cannot be rewritten "
                f"with the {INTRINSIC_FAMILY[statement.intrinsic]} polyfill for {self.target}",
                target=self.target,
            )
        target_conversion, value_conversion = statement.conversions
        value = self.polyfill(
            statement.intrinsic,
            [
                _convert(clone(node.target), target_conversion),
                _convert(node.value, value_conversion),
            ],
            statement.result_type,
        )
        value = _convert(value, statement.back_conversion)
        node.operator = "="
        node.value = value
        node.right = value
        return node

    def lower_step(self, node, statement, parent, field):
        in_statement = isinstance(parent, A.ExpressionStatementNode) or (
            field == "update" and isinstance(parent, A.ForNode)
        )
        in_block = field in _STATEMENT_FIELDS and isinstance(
            parent, (A.StatementNode, A.CaseNode)
        )
        if not (in_statement or in_block) or not is_pure(node.operand):
            raise RuleInexpressibleError(
                f"the signed {node.op} used as a value cannot be rewritten with wrapping arithmetic for "
                f"{self.target}; write it as a separate statement",
                target=self.target,
            )
        one = A.LiteralNode(1, A.PrimitiveType("int"))
        value = self.polyfill(
            statement.intrinsic, [clone(node.operand), one], statement.result_type
        )
        assignment = A.AssignmentNode(node.operand, value, "=")
        assignment.source_location = node.source_location
        return assignment if in_statement else A.ExpressionStatementNode(assignment)


def _native(node, operator, arguments):
    origin = node.annotations.get("cgl_rule_origin")
    if isinstance(origin, A.BinaryOpNode):
        origin.left, origin.right = arguments
        return origin
    if isinstance(origin, A.UnaryOpNode):
        origin.operand = arguments[0]
        return origin
    if len(arguments) == 1:
        return A.UnaryOpNode(operator, arguments[0])
    return A.BinaryOpNode(arguments[0], operator, arguments[1])


def _convert(operand, conversion):
    """``operand`` converted to the 32-bit integer type ``conversion`` (``None``: unchanged)."""
    if conversion is None:
        return operand
    scalar, vector = ("uint", "uvec") if conversion.kind == "u" else ("int", "ivec")
    name = scalar if conversion.lanes == 1 else f"{vector}{conversion.lanes}"
    return call(name, [operand])


@functools.lru_cache(maxsize=None)
def template(intrinsic):
    """The parsed expression of a polyfill template; parameters are the identifiers ``x0`` and ``x1``."""
    from ..source_registry import SOURCE_REGISTRY, register_default_sources

    register_default_sources()
    arity = INTRINSICS[intrinsic][1]
    parameters = ", ".join(f"int x{index}" for index in range(arity))
    text = (
        "shader __cgl_rule_template {\n"
        f"    int __cgl_template({parameters}) {{ return {POLYFILLS[intrinsic]}; }}\n"
        "}\n"
    )
    program = SOURCE_REGISTRY.get("cgl").parse(text)
    function = program.functions[0]
    (statement,) = function.body.statements
    return statement.value


def instantiate(intrinsic, operands):
    """A fresh copy of the template with ``x0``, ``x1`` replaced by the operands (each used exactly once)."""
    expression = clone(template(intrinsic))
    return _substitute(expression, {f"x{i}": o for i, o in enumerate(operands)})


def _substitute(node, bindings):
    if isinstance(node, A.IdentifierNode) and node.name in bindings:
        return bindings[node.name]
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        replacement = _substitute(child, bindings)
        if replacement is not child:
            set_child(node, field, key, replacement)
    return _fold_unsigned_literal(node)


def _fold_unsigned_literal(node):
    """``asuint(1)`` becomes ``1u``, so wrapped counters read ``asuint(i) + 1u``."""
    if isinstance(node, A.FunctionCallNode) and call_name(node) == "asuint":
        (argument,) = node.arguments
        value = getattr(argument, "value", None)
        if (
            isinstance(argument, A.LiteralNode)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            return A.LiteralNode(value, A.PrimitiveType("uint"))
    return node
