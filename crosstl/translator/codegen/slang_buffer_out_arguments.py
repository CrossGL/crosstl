"""Slang calls that pass an element of a GLSL buffer block to an ``out`` or ``inout`` parameter.

The Slang generator writes a GLSL buffer block whose layout a structured buffer cannot hold as a
``RWByteAddressBuffer``, so an element of it is a ``Load`` expression and a store is a ``Store`` call. Neither is an
l-value, and slangc refuses one as the argument of an ``out`` or ``inout`` parameter ("argument must be l-value",
E30047). GLSL passes such an argument by copy-in, copy-out (GLSL 4.60 section 6.1.1, "Function Calling
Conventions"), so a call statement is written that way: a local receives the element, the call writes the local, and
the local is stored back to the element after the call. A call in a larger expression, or one whose element index has
a side effect that the copy-out would repeat, is refused.
"""

from __future__ import annotations

from ..ast import (
    AssignmentNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = [
    "slang_buffer_out_argument_refusal",
    "slang_buffer_out_argument_statement",
]

_WRITING_QUALIFIERS = frozenset({"out", "inout"})


def _callee_name(call):
    function = getattr(call, "function", None)
    name = getattr(function, "name", function)
    return name if isinstance(name, str) else None


def _parameters(codegen, call):
    name = _callee_name(call)
    if name is None or name not in codegen.user_function_names:
        return None
    function = codegen.user_functions_by_name.get(name)
    if function is None:
        return None
    return list(
        getattr(function, "parameters", None) or getattr(function, "params", None) or []
    )


def _writes(parameter):
    return any(
        str(qualifier).lower() in _WRITING_QUALIFIERS
        for qualifier in getattr(parameter, "qualifiers", None) or []
    )


def _is_buffer_block_element(codegen, argument):
    if codegen.glsl_buffer_block_array_access(argument) is not None:
        return True
    access = codegen.glsl_buffer_block_member_access(argument)
    return access is not None and not access.get("runtime_array")


def _buffer_arguments(codegen, call):
    """``[(position, parameter)]`` of the buffer-block elements ``call`` passes to writing parameters."""
    parameters = _parameters(codegen, call)
    if not parameters:
        return []
    arguments = list(
        getattr(call, "arguments", None) or getattr(call, "args", None) or []
    )
    return [
        (position, parameters[position])
        for position, argument in enumerate(arguments)
        if position < len(parameters)
        and _writes(parameters[position])
        and _is_buffer_block_element(codegen, argument)
    ]


def _has_side_effect(codegen, expression):
    for node in codegen.walk_ast(expression):
        if isinstance(node, (FunctionCallNode, AssignmentNode)):
            return True
        if isinstance(node, UnaryOpNode) and str(getattr(node, "op", "")) in {
            "++",
            "--",
        }:
            return True
    return False


def slang_buffer_out_argument_refusal(codegen, call):
    """Refuse ``call`` in an expression when it passes a buffer-block element to a writing parameter."""
    found = _buffer_arguments(codegen, call)
    if not found:
        return None
    position, parameter = found[0]
    return codegen.refuse_lowering(
        f"Slang call to '{_callee_name(call)}': argument {position} ('{parameter.name}') is an element of a "
        "buffer block that Slang writes as a byte-address buffer, which is no l-value, and the call is part of "
        "a larger expression, so the copy back after the call has no place",
        "0",
    )


def slang_buffer_out_argument_statement(codegen, node):
    """Slang source of a call statement with buffer-block elements as writing arguments, or ``None``."""
    call = node.expression if isinstance(node, ExpressionStatementNode) else node
    if not isinstance(call, FunctionCallNode):
        return None
    found = _buffer_arguments(codegen, call)
    if not found:
        return None
    arguments = list(call.arguments)
    before, after = [], []
    for position, parameter in found:
        argument = arguments[position]
        if _has_side_effect(codegen, argument):
            return codegen.refuse_lowering(
                f"Slang call to '{_callee_name(call)}': argument {position} ('{parameter.name}') is a buffer "
                "element whose index has a side effect, which copying the result back would repeat",
                ";",
            )
        temp = codegen.unique_expression_temp_name(f"cgl_{parameter.name}")
        copy_in = (
            argument
            if "inout" in {str(q).lower() for q in parameter.qualifiers or []}
            else None
        )
        before.append(VariableNode(temp, parameter.param_type, initial_value=copy_in))
        after.append(AssignmentNode(argument, IdentifierNode(temp)))
        arguments[position] = IdentifierNode(temp)
    rewritten = FunctionCallNode(
        call.function,
        arguments,
        generic_args=call.generic_args if call.generic_suffix_present else None,
    )
    block = BlockNode(before + [ExpressionStatementNode(rewritten)] + after)
    return codegen.generate_block_statement(block)
