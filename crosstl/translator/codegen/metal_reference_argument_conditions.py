"""Copied Metal reference arguments in the headers of ``if``, loops and ``switch`` statements.

Metal lowers ``inout``/``out`` parameters to ``thread T&``, so an argument outside the thread address space (a
buffer element, threadgroup data, a vector component) is copied into a thread temporary before the call and
stored back after it (``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``). That needs a
statement around the call. A call in a statement header has none, and emitting it unchanged binds a ``thread``
reference to the other address space, which Metal rejects.

This module gives each header evaluation a statement of its own, keeping the GLSL rules for when the header runs
(OpenGL Shading Language 4.60, sections 6.2 "Selection" and 6.3 "Iteration"):

* an ``if`` condition is evaluated into a ``bool`` temporary just before the ``if``; a later ``else if``
  condition is evaluated in the ``else`` branch, so it still runs only after the earlier conditions failed;
* a ``while`` condition, and a ``for`` condition, become a guard at the top of the body of an endless loop
  (``if (!c) { break; }``). A ``continue`` in the body reaches the next test as before: for a ``for`` loop it runs
  the update first, which stays in the loop header;
* a ``for`` initializer is emitted as a statement in a block that encloses the loop;
* a ``switch`` selector of type ``int`` or ``uint`` is evaluated into a temporary in a block that encloses the
  ``switch``;
* a ``do``/``while`` condition and a ``for`` update are reached by a ``continue`` in the body, so no statement runs
  before every evaluation. Each call in them that copies arguments is evaluated in place as a comma expression
  (:mod:`.metal_reference_sequenced_calls`), whose temporaries are declared before the loop; the copies are made
  and stored back on every evaluation, and only then.

A selector of another type needs a conversion first, and is refused with :class:`.MetalLoweringRefused`.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    AssignmentNode,
    BlockNode,
    BreakNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    PrimitiveType,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .metal_reference_sequenced_calls import generate_conditional_call_statement
from .metal_refusals import MetalLoweringRefused

__all__ = ["generate_header_writeback_statement", "rewrite_metal_else_if_chain"]


def _probe(expression):
    """A declaration that evaluates ``expression`` once, for finding its copied reference arguments."""
    return VariableNode("__crossgl_probe", PrimitiveType("bool"), expression)


def _needs_copies(codegen, expression):
    if expression is None:
        return False
    expressions = expression if isinstance(expression, list) else [expression]
    return any(
        codegen.metal_reference_argument_writebacks(_probe(item), [])
        for item in expressions
        if item is not None
    )


def _statements(body):
    if hasattr(body, "statements"):
        return list(body.statements)
    if isinstance(body, list):
        return list(body)
    return [] if body is None else [body]


def _condition_temporary(codegen, condition):
    codegen.metal_writeback_temporary_count += 1
    name = f"__crossgl_cond{codegen.metal_writeback_temporary_count}"
    return name, VariableNode(name, PrimitiveType("bool"), condition)


def _guarded_body(codegen, condition, body):
    """The loop body with the condition evaluated first and leaving the loop when it fails."""
    name, declaration = _condition_temporary(codegen, condition)
    guard = IfNode(UnaryOpNode("!", IdentifierNode(name)), [BreakNode()])
    return [declaration, guard] + _statements(body)


def rewrite_metal_else_if_chain(codegen, node):
    """``node`` with each ``else if`` whose condition copies reference arguments moved into an ``else`` block.

    The ``else`` block emits the nested ``if`` as a statement, so its condition gets the statement it needs. The
    AST is not changed: the links of the chain that change are shallow copies.
    """
    else_branch = getattr(node, "else_branch", None)
    if not isinstance(else_branch, IfNode):
        return node
    rewritten = rewrite_metal_else_if_chain(codegen, else_branch)
    if _needs_copies(codegen, getattr(rewritten, "condition", None)):
        rewritten = [rewritten]
    if rewritten is else_branch:
        return node
    node = copy.copy(node)
    node.else_branch = rewritten
    node.else_body = rewritten
    return node


def _copying_calls(codegen, expression):
    """The calls in ``expression`` that copy reference arguments, and the copies of the whole expression."""
    writebacks = codegen.metal_reference_argument_writebacks(_probe(expression), [])
    calls = []
    for node in codegen.iter_ast_nodes(expression):
        if isinstance(node, FunctionCallNode) and any(
            argument is candidate
            for argument, _qualifier, _parameter in writebacks
            for candidate in list(getattr(node, "args", None) or [])
        ):
            if not any(node is call for call in calls):
                calls.append(node)
    return calls, writebacks


def _generate_in_place(codegen, stmt, indent, expression, discarded=False):
    """Emit ``stmt`` with the copying calls of ``expression``, one of its headers, evaluated in place.

    When the header's value is ``discarded`` (a ``for`` update), so are the values of the calls that make it up.
    """
    if expression is None:
        return None
    calls, writebacks = _copying_calls(codegen, expression)
    if not calls:
        return None
    items = expression if isinstance(expression, list) else [expression]
    unused = [
        item for item in items if discarded and isinstance(item, FunctionCallNode)
    ]
    return generate_conditional_call_statement(
        codegen, stmt, indent, calls, writebacks, unused
    )


_SELECTOR_TYPES = frozenset({"int", "uint"})


def _generate_switch(codegen, stmt, indent):
    selector = getattr(stmt, "expression", None)
    if not _needs_copies(codegen, selector):
        return None
    type_name = codegen.type_name_string(codegen.expression_result_type(selector))
    if type_name not in _SELECTOR_TYPES:
        raise MetalLoweringRefused(
            [
                "a reference argument copied in and out in a switch selector of type "
                f"'{type_name}': the selector must be evaluated into an int or uint local first"
            ]
        )
    codegen.metal_writeback_temporary_count += 1
    name = f"__crossgl_selector{codegen.metal_writeback_temporary_count}"
    declaration = VariableNode(name, PrimitiveType(type_name), selector)
    lowered = copy.copy(stmt)
    lowered.expression = IdentifierNode(name)
    # The block ends the temporary's scope with the switch.
    return codegen.generate_statement(BlockNode([declaration, lowered]), indent)


def _generate_if(codegen, stmt, indent):
    chain = rewrite_metal_else_if_chain(codegen, stmt)
    if not _needs_copies(codegen, chain.condition):
        if chain is stmt:
            return None
        return codegen.generate_if(chain, indent)
    name, declaration = _condition_temporary(codegen, chain.condition)
    lowered = IfNode(IdentifierNode(name), chain.then_branch, chain.else_branch)
    return codegen.generate_statement(declaration, indent) + codegen.generate_if(
        lowered, indent
    )


def _init_declarations(init):
    items = init if isinstance(init, list) else [init]
    return [
        item for item in items if isinstance(item, (VariableNode, BackendVariableNode))
    ]


@contextmanager
def _loop_scope(codegen, init):
    """The initializer's variables in scope, as the loop's condition and update see them."""
    previous = dict(codegen.local_variable_types)
    try:
        for declaration in _init_declarations(init):
            raw_type = getattr(
                declaration, "var_type", getattr(declaration, "vtype", None)
            )
            codegen.local_variable_types[declaration.name] = codegen.type_name_string(
                raw_type
            )
        yield
    finally:
        codegen.local_variable_types = previous


def _generate_for(codegen, stmt, indent):
    init = getattr(stmt, "init", None)
    condition = getattr(stmt, "condition", None)
    with _loop_scope(codegen, init):
        # The update sees the initializer's variables; the whole loop is generated again for the rest.
        update_code = _generate_in_place(
            codegen, stmt, indent, getattr(stmt, "update", None), discarded=True
        )
        if update_code is not None:
            return update_code
        condition_needs_copies = _needs_copies(codegen, condition)
    init_needs_copies = isinstance(
        init, (VariableNode, BackendVariableNode, AssignmentNode, BackendAssignmentNode)
    ) and bool(codegen.metal_reference_argument_writebacks(init, []))
    if not (init_needs_copies or condition_needs_copies):
        return None
    lowered = copy.copy(stmt)
    if condition_needs_copies:
        lowered.condition = None
        lowered.body = _guarded_body(codegen, condition, stmt.body)
    if not init_needs_copies:
        return codegen.generate_for(lowered, indent)
    lowered.init = None
    # The block keeps the initializer's variables scoped to the loop, as the header did.
    return codegen.generate_statement(BlockNode([init, lowered]), indent)


def generate_header_writeback_statement(codegen, stmt, indent):
    """Emit ``stmt`` when a header expression copies reference arguments in and out, else return ``None``."""
    if isinstance(stmt, IfNode):
        return _generate_if(codegen, stmt, indent)
    if isinstance(stmt, ForNode):
        return _generate_for(codegen, stmt, indent)
    if isinstance(stmt, WhileNode):
        condition = getattr(stmt, "condition", None)
        if not _needs_copies(codegen, condition):
            return None
        loop = ForNode(None, None, None, _guarded_body(codegen, condition, stmt.body))
        return codegen.generate_for(loop, indent)
    if isinstance(stmt, SwitchNode):
        return _generate_switch(codegen, stmt, indent)
    if isinstance(stmt, DoWhileNode):
        return _generate_in_place(
            codegen, stmt, indent, getattr(stmt, "condition", None)
        )
    return None
