"""Refuse the result of a ``void`` function used as a value.

A call to a function declared ``void`` has no value. C++14 section 3.9.1/9, which MSL follows, allows an
expression of type ``void`` only as an expression statement, an operand of a comma expression, the second or third
operand of ``?:``, the operand of a conversion to ``void``, or the value of a ``return`` in a ``void`` function.
GLSL 4.60 section 4.1.1 allows ``void`` only as a function return type, and HLSL follows C++. Every target rejects
such a program, and SPIR-V, which receives no compiler between the translator and the driver, would carry an
``OpFAdd`` of a ``void`` id. The translator refuses it with :class:`VoidValueError` instead.

A callee counts as ``void`` only when every user function of that name returns ``void``, so an overload set with a
value-returning member is left alone. Built-in functions are not checked.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import child_slots, get_child

__all__ = ["VoidValueError", "refuse_void_values"]


class VoidValueError(TranslationDiagnostic):
    """A call to a ``void`` function in a position that needs its value."""

    def __init__(self, target, callee, where, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"'{callee}' returns void, so its call has no value {where} (C++14 section 3.9.1/9 and GLSL 4.60 "
            "section 4.1.1 allow a void call only as a statement, a comma operand, a '?:' branch, a cast to "
            "void or the value returned by a void function)",
            target=target,
            code=f"{CODE_PREFIX}{target or 'crossgl'}-void-value",
        )


def _is_void(type_node) -> bool:
    name = getattr(type_node, "name", type_node)
    return isinstance(name, str) and name == "void"


def _void_callees(program) -> frozenset[str]:
    """Names whose every user-function declaration returns ``void``."""
    void, valued = set(), set()
    for function in iter_functions(program):
        name = getattr(function, "name", None)
        if not isinstance(name, str):
            continue
        (void if _is_void(getattr(function, "return_type", None)) else valued).add(name)
    return frozenset(void - valued)


class _Checker:
    def __init__(self, target, void_callees):
        self.target = target
        self.void_callees = void_callees
        self.parents = {}
        self.returns_void = True

    def is_void_call(self, node) -> bool:
        return (
            isinstance(node, A.FunctionCallNode)
            and call_name(node) in self.void_callees
        )

    def is_void_expression(self, node) -> bool:
        if isinstance(node, A.TernaryOpNode):
            return self.is_void_expression(node.true_expr) and self.is_void_expression(
                node.false_expr
            )
        if isinstance(node, A.CommaExpressionNode):
            return bool(node.expressions) and self.is_void_expression(
                node.expressions[-1]
            )
        return self.is_void_call(node)

    def check_function(self, function):
        body = getattr(function, "body", None)
        if body is None:
            return
        self.returns_void = _is_void(getattr(function, "return_type", None))
        self.parents = {}
        stack = [body]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(reversed(node))
                continue
            if not isinstance(node, A.ASTNode):
                continue
            if self.is_void_call(node):
                self.check_use(node)
            for field, key in reversed(list(child_slots(node))):
                child = get_child(node, field, key)
                if isinstance(child, A.ASTNode):
                    self.parents[id(child)] = (node, field)
                stack.append(child)

    def check_use(self, call):
        """Raise unless ``call``'s value is discarded or is itself a ``void`` operand that is discarded."""
        node = call
        while True:
            parent, field = self.parents.get(id(node), (None, None))
            if parent is None:
                return
            if isinstance(parent, A.ExpressionStatementNode):
                if (
                    getattr(parent, "is_tail_expression", False)
                    and not self.returns_void
                ):
                    self.refuse(call, "as the value of a function that returns a value")
                return
            if isinstance(parent, A.ForNode) and field in ("init", "update"):
                return
            if isinstance(parent, A.ReturnNode):
                if not self.returns_void:
                    self.refuse(call, "as the value of a function that returns a value")
                return
            if isinstance(parent, A.CastNode) and _is_void(parent.target_type):
                return
            if isinstance(parent, A.FunctionCallNode) and call_name(parent) == "void":
                return
            if isinstance(parent, A.CommaExpressionNode):
                if parent.expressions and parent.expressions[-1] is not node:
                    return
                node = parent
                continue
            if isinstance(parent, A.TernaryOpNode) and field in (
                "true_expr",
                "false_expr",
            ):
                other = parent.false_expr if field == "true_expr" else parent.true_expr
                if not self.is_void_expression(other):
                    self.refuse(call, "beside a '?:' branch that has a value")
                node = parent
                continue
            self.refuse(call, "where its value is used")

    def refuse(self, call, where):
        raise VoidValueError(
            self.target,
            call_name(call),
            where,
            source_location=getattr(call, "source_location", None),
        )


def refuse_void_values(program, target) -> None:
    """Raise :class:`VoidValueError` for the first ``void`` call whose value the program uses."""
    void_callees = _void_callees(program)
    if not void_callees:
        return
    checker = _Checker(target, void_callees)
    for function in iter_functions(program):
        checker.check_function(function)
