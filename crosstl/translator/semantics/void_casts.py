"""Lower casts to ``void`` to the evaluations they stand for.

``(void)e`` evaluates ``e`` and discards its value (C++14 section 5.4 and 5.2.9/6, which MSL and HLSL follow); a
CrossGL program spells it ``void(e)``. GLSL (4.60 section 5.4.1 allows no constructor of type ``void``), WGSL and
SPIR-V have no such cast, and ``void(x);`` reads as a declaration in HLSL, so the cast is lowered before code
generation wherever its value is discarded:

* an expression statement ``void(e);`` becomes ``e;``, and is dropped when ``e`` provably has no effect (it only
  silences an unused-value warning);
* a statement ``a, void(b);`` becomes ``a; b;``;
* a statement ``c ? f() : g();`` whose branches are both ``void`` becomes ``if (c) { f(); } else { g(); }``, because
  WGSL has no conditional of type ``void``, and its branches are lowered the same way;
* a ``for`` header's init and update, and every comma operand but the last, drop the cast the same way.

A cast to ``void`` left anywhere else, such as a ``?:`` branch inside a larger expression, is refused with
:class:`VoidCastError` for every target but Metal, whose C++ expression grammar carries it, and CrossGL itself.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import SKIPPED_FIELDS, child_slots, get_child, is_pure, set_child

__all__ = ["VoidCastError", "lower_void_casts"]

_KEEPING = frozenset({"cgl", "crossgl", ""})
_CARRYING = frozenset({"metal"})


class VoidCastError(TranslationDiagnostic):
    """A cast to ``void`` in a position the target has no form for."""

    def __init__(self, target, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"{target} has no cast to void inside this expression (GLSL 4.60 section 5.4.1 and WGSL have no "
            "conversion to void); only a discarded statement, comma operand or 'for' header part can drop it",
            target=target,
            code=f"{CODE_PREFIX}{target}-void-cast",
        )


def _is_void_type(type_node) -> bool:
    name = getattr(type_node, "name", type_node)
    return isinstance(name, str) and name.strip() == "void"


def void_cast_operand(node):
    """The operand of ``void(e)`` or ``(void)e``, or ``None`` for any other node."""
    if isinstance(node, A.CastNode) and _is_void_type(node.target_type):
        return node.expression
    if (
        isinstance(node, A.FunctionCallNode)
        and call_name(node) == "void"
        and len(node.arguments or []) == 1
    ):
        return node.arguments[0]
    return None


def _void_callees(program) -> frozenset[str]:
    void, valued = set(), set()
    for function in iter_functions(program):
        name = getattr(function, "name", None)
        if isinstance(name, str):
            is_void = _is_void_type(getattr(function, "return_type", None))
            (void if is_void else valued).add(name)
    return frozenset(void - valued)


def _statement(expression, origin):
    return A.ExpressionStatementNode(
        expression, source_location=getattr(origin, "source_location", None)
    )


class _Lowering:
    def __init__(self, target, void_callees):
        self.target = target
        self.void_callees = void_callees

    def is_void(self, node) -> bool:
        if void_cast_operand(node) is not None:
            return True
        if isinstance(node, A.FunctionCallNode):
            return call_name(node) in self.void_callees
        if isinstance(node, A.CommaExpressionNode):
            return bool(node.expressions) and self.is_void(node.expressions[-1])
        if isinstance(node, A.TernaryOpNode):
            return self.is_void(node.true_expr) and self.is_void(node.false_expr)
        return False

    def statements(self, expression, origin):
        """Statements evaluating ``expression`` for its effects, or ``None`` to keep its statement as it is."""
        operand = void_cast_operand(expression)
        if operand is not None:
            lowered = self.statements(operand, origin)
            if lowered is not None:
                return lowered
            return [] if is_pure(operand) else [_statement(operand, origin)]
        if isinstance(expression, A.CommaExpressionNode):
            parts = [self.statements(part, origin) for part in expression.expressions]
            if all(part is None for part in parts):
                return None
            return [
                statement
                for part, lowered in zip(expression.expressions, parts)
                for statement in (
                    [_statement(part, origin)] if lowered is None else lowered
                )
            ]
        if (
            isinstance(expression, A.TernaryOpNode)
            and self.is_void(expression.true_expr)
            and self.is_void(expression.false_expr)
        ):
            return self.conditional(expression, origin)
        return None

    def conditional(self, expression, origin):
        branches = []
        for branch in (expression.true_expr, expression.false_expr):
            lowered = self.statements(branch, origin)
            branches.append(
                [_statement(branch, origin)] if lowered is None else lowered
            )
        then_body, else_body = branches
        if not then_body and not else_body:
            if is_pure(expression.condition):
                return []
            return [_statement(expression.condition, origin)]
        return [
            A.IfNode(
                expression.condition,
                A.BlockNode(then_body),
                A.BlockNode(else_body) if else_body else None,
                source_location=getattr(origin, "source_location", None),
            )
        ]

    def expressions(self, expression):
        """Expressions evaluating a discarded ``expression``, or ``None`` to keep it as it is."""
        operand = void_cast_operand(expression)
        if operand is not None:
            lowered = self.expressions(operand)
            if lowered is not None:
                return lowered
            return [] if is_pure(operand) else [operand]
        if isinstance(expression, A.CommaExpressionNode):
            parts = [self.expressions(part) for part in expression.expressions]
            if all(part is None for part in parts):
                return None
            return [
                item
                for part, lowered in zip(expression.expressions, parts)
                for item in ([part] if lowered is None else lowered)
            ]
        return None

    # Statement positions ----------------------------------------------------------------------------------------
    def statement_list(self, statements):
        result, changed = [], False
        for statement in statements:
            lowered = self.lowered_statement(statement)
            if lowered is None:
                result.append(statement)
            else:
                result.extend(lowered)
                changed = True
        if changed:
            statements[:] = result

    def lowered_statement(self, statement):
        if not isinstance(statement, A.ExpressionStatementNode) or getattr(
            statement, "is_tail_expression", False
        ):
            return None
        return self.statements(statement.expression, statement)

    def for_header(self, node):
        init = node.init
        if isinstance(init, A.ExpressionStatementNode):
            lowered = self.lowered_statement(init)
            if lowered is not None:
                init = self.header_statement(lowered, init)
        elif isinstance(init, A.ExpressionNode):
            lowered = self.expressions(init)
            if lowered is not None:
                init = self.header_expression(lowered)
        if init is not node.init:
            set_child(node, "init", None, init)
        update = node.update
        items = update if isinstance(update, list) else [update]
        lowered_items, changed = [], False
        for item in items:
            lowered = self.expressions(item) if item is not None else None
            changed = changed or lowered is not None
            lowered_items.extend([item] if lowered is None else lowered)
        if changed:
            if isinstance(update, list):
                update[:] = lowered_items
            else:
                set_child(node, "update", None, self.header_expression(lowered_items))

    def header_statement(self, statements, origin):
        """One ``for`` init statement for ``statements``; a lowered ``?:`` has no init form and is refused."""
        if not statements:
            return None
        if len(statements) == 1 and not isinstance(statements[0], A.IfNode):
            return statements[0]
        if any(not isinstance(item, A.ExpressionStatementNode) for item in statements):
            raise VoidCastError(
                self.target, source_location=getattr(origin, "source_location", None)
            )
        expressions = [statement.expression for statement in statements]
        return _statement(A.CommaExpressionNode(expressions), origin)

    @staticmethod
    def header_expression(expressions):
        if not expressions:
            return None
        if len(expressions) == 1:
            return expressions[0]
        return A.CommaExpressionNode(expressions)

    def comma_operands(self, node):
        """Drop the casts to ``void`` of a comma expression's discarded operands; ``node`` if it remains a comma."""
        *discarded, last = node.expressions
        kept, changed = [], False
        for part in discarded:
            lowered = self.expressions(part)
            changed = changed or lowered is not None
            kept.extend([part] if lowered is None else lowered)
        if not changed:
            return node
        if not kept:
            return last
        node.expressions[:] = kept + [last]
        return node

    def visit(self, root):
        stack, seen = [root], set()
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                self.statement_list(node)
                stack.extend(node)
                continue
            if not isinstance(node, A.ASTNode) or id(node) in seen:
                continue
            seen.add(id(node))
            for field, value in list(vars(node).items()):
                if field not in SKIPPED_FIELDS and isinstance(value, list):
                    self.statement_list(value)
            if isinstance(node, A.ForNode):
                self.for_header(node)
            for field, key in list(child_slots(node)):
                child = get_child(node, field, key)
                if isinstance(child, A.CommaExpressionNode) and child.expressions:
                    replacement = self.comma_operands(child)
                    if replacement is not child:
                        set_child(node, field, key, replacement)
                        child = replacement
                stack.append(child)


def _refuse_leftovers(root, target):
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, A.ASTNode):
            continue
        if void_cast_operand(node) is not None:
            raise VoidCastError(
                target, source_location=getattr(node, "source_location", None)
            )
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))


def lower_void_casts(program, target) -> None:
    """Lower discarded casts to ``void`` in place; raise :class:`VoidCastError` for one the target cannot carry."""
    target = str(target or "")
    if target in _KEEPING or not isinstance(program, A.ASTNode):
        return
    lowering = _Lowering(target, _void_callees(program))
    functions = list(iter_functions(program))
    for function in functions:
        if getattr(function, "body", None) is not None:
            lowering.visit(function.body)
    if target in _CARRYING:
        return
    for function in functions:
        if getattr(function, "body", None) is not None:
            _refuse_leftovers(function.body, target)
