"""Evaluation order for Metal reference arguments that are copied in and out.

CrossGL follows GLSL for calls (OpenGL Shading Language 4.60, section 6.1.1, "Function Calling Conventions"): the
arguments are evaluated once, in order, left to right. An ``inout`` argument contributes its value and its lvalue
at its own position, and the value is copied back to that lvalue when the function returns.

Metal lowers ``inout``/``out`` to ``thread T&``, so an argument outside the thread address space (a buffer element,
threadgroup data, a vector component) is copied into a temporary before the statement and stored back after it
(``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``). That moves the copy-in ahead of every
argument to its left, and re-evaluates the lvalue's indices after the call. This module keeps the source order:

* an argument to the left of a copied argument whose evaluation has side effects is evaluated first into a
  temporary of its parameter type, together with every argument before it whose value those side effects could
  change; the call then reads the temporaries;
* an index of a copied lvalue that the statement could change (anything but a literal or a thread local the
  statement does not write) is evaluated once into a temporary, before the copy-in, and both the copy-in and the
  copy-out use it.

Moving arguments out of the statement is only exact when the call is the statement's whole expression (a call
statement, a declaration initializer or the value of a plain assignment); any other placement is refused.
"""

from __future__ import annotations

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MetalRefusal

__all__ = ["generate_ordered_writeback_statement"]

_REFERENCE_QUALIFIERS = frozenset({"inout", "out"})


def _call_arguments(call):
    return list(getattr(call, "args", None) or getattr(call, "arguments", None) or [])


def _head_call(codegen, stmt):
    """The call that is the statement's whole evaluated expression, or ``None``."""
    if isinstance(stmt, FunctionCallNode):
        return stmt
    if "ExpressionStatementNode" in type(stmt).__name__:
        expression = getattr(stmt, "expression", None)
        return expression if isinstance(expression, FunctionCallNode) else None
    if isinstance(stmt, (VariableNode, BackendVariableNode)):
        value = getattr(stmt, "initial_value", None)
        return value if isinstance(value, FunctionCallNode) else None
    if isinstance(stmt, (AssignmentNode, BackendAssignmentNode)):
        if str(getattr(stmt, "operator", "=") or "=") != "=":
            return None
        target = getattr(stmt, "target", getattr(stmt, "left", None))
        value = getattr(stmt, "value", getattr(stmt, "right", None))
        if isinstance(
            value, FunctionCallNode
        ) and codegen.metal_expression_is_side_effect_free(target):
            return value
    return None


def _root_name(codegen, expr):
    node = expr
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        node = node.array if isinstance(node, ArrayAccessNode) else node.object
    name = codegen.expression_name(node)
    return name if isinstance(name, str) else None


def _statement_written_names(codegen, stmt):
    """Names whose storage the statement writes: assignment targets, increments and reference arguments."""
    written = set()
    for node in codegen.iter_ast_nodes(stmt):
        if isinstance(node, (AssignmentNode, BackendAssignmentNode)):
            name = _root_name(
                codegen, getattr(node, "target", getattr(node, "left", None))
            )
            if name:
                written.add(name)
        elif isinstance(node, UnaryOpNode) and str(
            getattr(node, "op", getattr(node, "operator", ""))
        ) in {"++", "--"}:
            name = _root_name(codegen, getattr(node, "operand", None))
            if name:
                written.add(name)
        elif isinstance(node, FunctionCallNode):
            for index, argument in enumerate(_call_arguments(node)):
                if (
                    codegen.metal_reference_parameter_qualifier(node, index)
                    in _REFERENCE_QUALIFIERS
                ):
                    name = _root_name(codegen, argument)
                    if name:
                        written.add(name)
    return written


def _storage_helpers_can_write(codegen):
    """Names in the current function whose storage a called helper can write without a reference argument."""
    names = {
        name
        for name, _declaration, _vtype in (
            codegen.metal_forwarded_thread_private_globals.get(
                codegen.current_function_name, []
            )
        )
    }
    for (
        nodes
    ) in codegen.metal_lowered_program_scope_groupshared_globals_by_function.values():
        names.update(codegen.metal_variable_name(node) for node in nodes)
    return names


def _is_stable(codegen, expr, written, shared_names):
    """Whether ``expr`` has the same value before and after the statement's calls run."""
    if isinstance(expr, LiteralNode) or isinstance(expr, (int, float, bool)):
        return True
    if isinstance(expr, IdentifierNode):
        name = expr.name
        return (
            name in codegen.local_variable_types
            and codegen.current_address_space_variables.get(name, "thread") == "thread"
            and name not in written
            and name not in shared_names
        )
    if isinstance(expr, MemberAccessNode):
        # A component or field of a stable thread-private value, such as ``id.x``.
        return _is_stable(codegen, expr.object, written, shared_names)
    if isinstance(expr, UnaryOpNode) and str(
        getattr(expr, "op", getattr(expr, "operator", ""))
    ) in {"-", "+", "~", "!"}:
        return _is_stable(
            codegen, getattr(expr, "operand", None), written, shared_names
        )
    return False


def _copied_lvalue_indices(argument):
    indices = []
    node = argument
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        if isinstance(node, ArrayAccessNode):
            indices.append(node.index)
            node = node.array
        else:
            node = node.object
    indices.reverse()
    return indices


def _hoisted_parameter_type(codegen, call, index):
    func_name = codegen.function_call_name(call)
    argument_count = len(_call_arguments(call))
    overloads = [
        overload
        for overload in codegen.function_overloads_by_name.get(func_name, [])
        if len(getattr(overload, "parameters", getattr(overload, "params", [])) or [])
        == argument_count
    ]
    if len(overloads) != 1:
        raise MetalRefusal(
            f"Metal cannot keep the left-to-right argument order of '{func_name}': the call has "
            f"{len(overloads)} overloads of {argument_count} parameters"
        )
    parameters = list(
        getattr(overloads[0], "parameters", getattr(overloads[0], "params", []))
    )
    raw_type = codegen.parameter_raw_type(parameters[index])
    type_name = codegen.type_name_string(raw_type)
    if (
        not type_name
        or isinstance(raw_type, PointerType)
        or codegen.is_array_type_node(raw_type)
        or "[" in type_name
        or codegen.is_resource_parameter_type(type_name)
    ):
        raise MetalRefusal(
            f"Metal cannot keep the left-to-right argument order of '{func_name}': argument {index + 1} "
            f"has type '{type_name}', which cannot be evaluated into a temporary first"
        )
    return raw_type


def _plan_hoists(codegen, stmt, writebacks, written, shared_names):
    """Arguments that must be evaluated, in order, before the copy-ins: ``[(argument, parameter type)]``."""
    copied_ids = {id(argument) for argument, _qualifier, _parameter in writebacks}
    hoists = []
    for call in codegen.iter_ast_nodes(stmt):
        if not isinstance(call, FunctionCallNode):
            continue
        arguments = _call_arguments(call)
        copied = [
            index
            for index, argument in enumerate(arguments)
            if id(argument) in copied_ids
        ]
        if not copied:
            continue
        effectful = [
            index
            for index in range(max(copied))
            if not codegen.metal_expression_is_side_effect_free(arguments[index])
        ]
        if not effectful:
            continue
        last_effect = max(effectful)
        func_name = codegen.function_call_name(call)
        if min(copied) < last_effect or _head_call(codegen, stmt) is not call:
            raise MetalRefusal(
                f"Metal cannot copy a reference argument of '{func_name}' in and out after an argument with "
                "side effects in this position; evaluate the earlier argument into a local first"
            )
        for index in range(last_effect + 1):
            if (
                codegen.metal_reference_parameter_qualifier(call, index)
                in _REFERENCE_QUALIFIERS
            ):
                continue
            argument = arguments[index]
            if index in effectful or not _is_stable(
                codegen, argument, written, shared_names
            ):
                hoists.append((argument, _hoisted_parameter_type(codegen, call, index)))
    return hoists


def _inside(codegen, roots, node):
    return any(
        node is candidate
        for root in roots
        for candidate in codegen.iter_ast_nodes(root)
    )


def _index_type(codegen, index):
    type_name = codegen.type_name_string(codegen.expression_result_type(index))
    mapped = codegen.map_type(type_name) if type_name else None
    if not mapped or codegen.vector_value_width(mapped) is not None:
        raise MetalRefusal(
            "Metal cannot copy a reference argument back: the type of its index "
            f"'{codegen.generate_expression(index)}' is unknown"
        )
    return mapped


def generate_ordered_writeback_statement(codegen, stmt, indent, writebacks):
    """Emit ``stmt`` with copied reference arguments, keeping left-to-right evaluation."""
    written = _statement_written_names(codegen, stmt)
    shared_names = _storage_helpers_can_write(codegen)
    hoists = _plan_hoists(codegen, stmt, writebacks, written, shared_names)
    hoisted_roots = [argument for argument, _raw_type in hoists]
    remaining = [
        writeback
        for writeback in writebacks
        if not _inside(codegen, hoisted_roots, writeback[0])
    ]
    indent_str = "    " * indent
    prologue = ""
    substitutions = {}
    for argument, raw_type in hoists:
        codegen.metal_writeback_temporary_count += 1
        name = f"__crossgl_arg{codegen.metal_writeback_temporary_count}"
        # The temporary is a declaration of its own, so its reference arguments are copied in and out there.
        prologue += codegen.generate_statement(
            VariableNode(name, raw_type, argument), indent
        )
        substitutions[id(argument)] = name
    for argument, _qualifier, _parameter in remaining:
        for index in _copied_lvalue_indices(argument):
            if _is_stable(codegen, index, written, shared_names):
                continue
            codegen.metal_writeback_temporary_count += 1
            name = f"__crossgl_index{codegen.metal_writeback_temporary_count}"
            index_type = _index_type(codegen, index)
            prologue += f"{indent_str}{index_type} {name} = {codegen.generate_expression(index)};\n"
            substitutions[id(index)] = name
    previous = codegen.metal_argument_substitutions
    codegen.metal_argument_substitutions = {**previous, **substitutions}
    try:
        if remaining:
            body = codegen.generate_statement_with_argument_writebacks(
                stmt, indent, remaining
            )
        else:
            codegen.metal_writeback_statement_ids.add(id(stmt))
            try:
                body = codegen.generate_statement(stmt, indent)
            finally:
                codegen.metal_writeback_statement_ids.discard(id(stmt))
    finally:
        codegen.metal_argument_substitutions = previous
    return prologue + body
