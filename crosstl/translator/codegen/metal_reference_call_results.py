"""Calls whose Metal reference-argument copies the rest of their statement observes.

CrossGL follows GLSL for ``out`` and ``inout`` parameters (OpenGL Shading Language 4.60, section 6.1.1, "Function
Calling Conventions"): the value is copied back to the argument when the called function returns, so the rest of
the enclosing expression, and the store of an assignment, see the copied-back value. Arguments are evaluated once,
in order, from left to right (same section).

Metal lowers these parameters to ``thread T&``, so an argument outside the thread address space (a buffer element,
threadgroup data, a vector component) is copied into a temporary before the statement and stored back after it
(``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``). That is exact only when nothing evaluated
after the call touches the copied storage. Otherwise:

* ``tile[l] = bump(tile[l])`` stored the copy after the assignment, which overwrote the call's result;
* ``ivec2(bump(v.y), v.y)`` and ``bump(t[i]) + t[i]`` read the value from before the call;
* ``bump(t[i]) + bump(t[i])`` copied both arguments in before either call ran.

Such a call is evaluated first, into a local of its result type, in a declaration of its own that copies its
arguments in and out around the call. Everything evaluated before the call keeps its place in the order, as in
glslang: an earlier argument, or an operand to the left of the call, is evaluated into a local before the call
unless the call cannot change it (a literal, or a thread local the statement does not write); so is an index of an
lvalue the call is nested in, and an index of the assignment target, which glslang locates before it evaluates the
value. (For operands other than arguments GLSL does not fix the order, and Mesa reads a left operand after the
call; a source that depends on it is not portable.) The statement then reads the locals. A call evaluated conditionally (the right operand of ``&&``/``||``, a branch of
``?:``) is refused by the copy-in/copy-out lowering before this runs.
"""

from __future__ import annotations

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    CommaExpressionNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MetalRefusal
from ..semantics.lower import COMPOUND_TARGET_READ
from .metal_reference_argument_order import (
    _copied_lvalue_indices,
    _is_stable,
    _root_name,
    _statement_written_names,
    _storage_helpers_can_write,
)
from .metal_reference_sequenced_calls import (
    enclosing_call_reaches_copies,
    generate_sequenced_call_statement,
)

__all__ = ["generate_call_result_statement"]

_ASSIGNMENT_OPERATORS = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)


def _statement_shape(stmt):
    """``(value, assignment target)`` of a statement whose value is evaluated once, or ``None``."""
    if isinstance(stmt, (VariableNode, BackendVariableNode)):
        return getattr(stmt, "initial_value", None), None
    if isinstance(stmt, (AssignmentNode, BackendAssignmentNode)):
        return (
            getattr(stmt, "value", getattr(stmt, "right", None)),
            getattr(stmt, "target", getattr(stmt, "left", None)),
        )
    if isinstance(stmt, FunctionCallNode):
        return stmt, None
    if "ExpressionStatementNode" in type(stmt).__name__:
        expression = getattr(stmt, "expression", None)
        if isinstance(expression, (AssignmentNode, BackendAssignmentNode)):
            return _statement_shape(expression)
        return expression, None
    return None


def _ordered_children(node):
    """Sub-expressions of ``node`` in evaluation order, or ``None`` when the order is not modelled."""
    if isinstance(node, (LiteralNode, IdentifierNode, str, int, float, bool)):
        return []
    if isinstance(node, BinaryOpNode):
        return [node.left, node.right]
    if isinstance(node, FunctionCallNode):
        return list(getattr(node, "args", None) or [])
    if isinstance(node, ArrayAccessNode):
        return [node.array, node.index]
    if isinstance(node, MemberAccessNode):
        return [node.object]
    if isinstance(node, SwizzleNode):
        return [node.vector_expr]
    if isinstance(node, UnaryOpNode):
        return [node.operand]
    if isinstance(node, TernaryOpNode):
        return [node.condition, node.true_expr, node.false_expr]
    if isinstance(node, CommaExpressionNode):
        return list(node.expressions)
    if isinstance(node, ArrayLiteralNode):
        # Braced initializers are evaluated in order (C++14 [dcl.init.list]/4).
        return list(node.elements)
    if isinstance(node, CastNode):
        return [node.expression]
    if isinstance(node, ConstructorNode) and not node.named_arguments:
        return list(node.arguments or [])
    if isinstance(node, (AssignmentNode, BackendAssignmentNode)):
        # glslang locates the target before it evaluates the value, and stores last.
        return [
            getattr(node, "target", getattr(node, "left", None)),
            getattr(node, "value", getattr(node, "right", None)),
        ]
    return None


def _path_to(node, goal):
    """The nodes from ``node`` down to ``goal`` along modelled children; ``[]`` when an unmodelled node is crossed."""
    if node is goal:
        return [node]
    children = _ordered_children(node)
    if children is None:
        return []
    for child in children:
        path = _path_to(child, goal)
        if path is None:
            continue
        return [node, *path] if path else []
    return None


def _any_children(node):
    """Sub-expressions of ``node`` in evaluation order, or in attribute order when that is not modelled."""
    children = _ordered_children(node)
    if children is not None:
        return children
    found = []
    for key, child in getattr(node, "__dict__", {}).items():
        if key in {"parent", "annotations"}:
            continue
        for item in child if isinstance(child, (list, tuple)) else [child]:
            if hasattr(item, "__dict__") and not isinstance(item, type):
                found.append(item)
    return found


def _references(codegen, expr, roots, excluded=None):
    """Whether ``expr`` names one of ``roots`` outside ``excluded`` and the parts already evaluated into locals."""
    if (
        expr is None
        or expr is excluded
        or id(expr) in codegen.metal_argument_substitutions
    ):
        return False
    if isinstance(expr, str):
        return expr in roots
    if isinstance(expr, IdentifierNode):
        return expr.name in roots
    return any(
        _references(codegen, child, roots, excluded) for child in _any_children(expr)
    )


def _copied_calls(codegen, value, copied_ids):
    """Calls with copied reference arguments, innermost first: ``[(call, roots of the copied arguments)]``."""
    calls = []

    def visit(node):
        if node is None or id(node) in codegen.metal_argument_substitutions:
            return
        # Calls below an unmodelled node are found too, so that a read after them is refused.
        for child in _any_children(node):
            visit(child)
        if isinstance(node, FunctionCallNode):
            copied = [
                argument
                for argument in getattr(node, "args", None) or []
                if id(argument) in copied_ids
            ]
            roots = {_root_name(codegen, argument) for argument in copied}
            if copied and None not in roots:
                calls.append((node, roots))

    visit(value)
    return calls


def _reads_compound_target(node):
    """Whether ``node`` holds the target read of a compound assignment rewritten as ``t = f(t, v)``."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return False
    if (getattr(node, "annotations", None) or {}).get(COMPOUND_TARGET_READ):
        return True
    return any(_reads_compound_target(child) for child in _any_children(node))


def _split_siblings(path):
    """Operands evaluated before and after the last node of ``path``: ``(before, after)``.

    ``before`` holds ``(node, is_lvalue)``; the base of an array access the call indexes, and the target of an
    assignment the call is in the value of, are lvalues, whose indices, not their values, are evaluated first.
    """
    before, after = [], []
    for parent, child in zip(path, path[1:]):
        children = _ordered_children(parent)
        position = next(index for index, node in enumerate(children) if node is child)
        lvalue = isinstance(
            parent, (ArrayAccessNode, AssignmentNode, BackendAssignmentNode)
        )
        for node in children[:position]:
            if _reads_compound_target(node):
                # A rewritten compound assignment reads its target after its value.
                after.append(node)
            else:
                before.append((node, lvalue))
        after.extend(children[position + 1 :])
        if lvalue and position and not isinstance(parent, ArrayAccessNode):
            # A nested assignment stores to its target after its value.
            after.append(children[0])
    return before, after


def _stable(codegen, expr, written, shared_names):
    """Whether evaluating ``expr`` after the call gives the value it had before it."""
    if id(expr) in codegen.metal_argument_substitutions:
        return True
    if isinstance(expr, BinaryOpNode):
        operator = str(getattr(expr, "op", "") or "")
        return operator not in _ASSIGNMENT_OPERATORS and all(
            _stable(codegen, operand, written, shared_names)
            for operand in (expr.left, expr.right)
        )
    return _is_stable(codegen, expr, written, shared_names)


def _local_type(codegen, expr, what):
    """The CrossGL type of a local that holds ``expr``; refuses types a local cannot hold."""
    raw_type = codegen.expression_result_type(expr)
    type_name = codegen.type_name_string(raw_type) if raw_type is not None else None
    if (
        not type_name
        or type_name == "void"
        or isinstance(raw_type, PointerType)
        or "[" in type_name
        or "*" in type_name
        or codegen.is_resource_parameter_type(type_name)
    ):
        raise MetalRefusal(
            f"Metal cannot copy a reference argument back before the rest of the statement reads it: {what} "
            f"'{codegen.generate_expression(expr)}' has type '{type_name}', which cannot be evaluated into a "
            "local first; assign the call result to a local first"
        )
    return type_name


def _evaluate_into_local(codegen, expr, indent, prefix, what):
    """Declare a local initialized from ``expr``; returns ``(code, name)``."""
    type_name = _local_type(codegen, expr, what)
    codegen.metal_writeback_temporary_count += 1
    name = f"{prefix}{codegen.metal_writeback_temporary_count}"
    # A declaration of its own copies the reference arguments of calls in ``expr`` in and out around them.
    code = codegen.generate_statement(VariableNode(name, type_name, expr), indent)
    return code, name


def _hoist(codegen, stmt, indent, call, before, target):
    written = _statement_written_names(codegen, stmt)
    shared_names = _storage_helpers_can_write(codegen)
    early = []
    if target is not None:
        early.extend(_copied_lvalue_indices(target))
    for node, lvalue in before:
        early.extend(_copied_lvalue_indices(node) if lvalue else [node])
    prologue = ""
    substitutions = {}
    previous = codegen.metal_argument_substitutions
    try:
        for node in early:
            if _stable(codegen, node, written, shared_names):
                continue
            code, name = _evaluate_into_local(
                codegen, node, indent, "__crossgl_arg", "the operand"
            )
            prologue += code
            substitutions[id(node)] = name
            codegen.metal_argument_substitutions = {**previous, **substitutions}
        code, name = _evaluate_into_local(
            codegen, call, indent, "__crossgl_call", "the call"
        )
        prologue += code
        substitutions[id(call)] = name
        codegen.metal_argument_substitutions = {**previous, **substitutions}
        # The rest of the statement runs after the copy-back; copies of later calls are found again there.
        return prologue + codegen.generate_statement(stmt, indent)
    finally:
        codegen.metal_argument_substitutions = previous


def generate_call_result_statement(codegen, stmt, indent, writebacks):
    """Emit ``stmt`` when the rest of it observes a call's copied reference arguments, else return ``None``."""
    shape = _statement_shape(stmt)
    if shape is None:
        return None
    value, target = shape
    copied_ids = {id(argument) for argument, _qualifier, _parameter in writebacks}
    target_root = _root_name(codegen, target) if target is not None else None
    for call, roots in _copied_calls(codegen, value, copied_ids):
        path = _path_to(value, call)
        if path is None:
            continue
        stores_root = target_root in roots
        if not path:
            if stores_root or _references(codegen, value, roots, call):
                raise MetalRefusal(
                    f"Metal cannot copy the reference arguments of '{codegen.function_call_name(call)}' back "
                    "before the rest of the statement reads them: the call is nested in an expression whose "
                    "evaluation order is not modelled; assign the call result to a local first"
                )
            continue
        reach = enclosing_call_reaches_copies(codegen, path, roots)
        if reach == "in place":
            return generate_sequenced_call_statement(
                codegen, stmt, indent, call, writebacks
            )
        before, after = _split_siblings(path)
        if (
            reach is not None
            or stores_root
            or any(_references(codegen, node, roots) for node in after)
        ):
            return _hoist(codegen, stmt, indent, call, before, target)
    return None
