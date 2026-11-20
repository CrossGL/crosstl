"""Return values that copy Metal reference arguments in and out.

CrossGL follows GLSL for ``out`` and ``inout`` parameters (OpenGL Shading Language 4.60, section 6.1.1, "Function
Calling Conventions"): the value is copied back to the argument when the called function returns, which is before
the ``return`` that evaluated the call leaves the enclosing function (section 6.4, "Jumps").

Metal lowers ``inout``/``out`` to ``thread T&``, so an argument outside the thread address space (a buffer element,
threadgroup data, a vector component) is copied into a temporary before the statement and stored back after it
(``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``). A store after ``return`` never runs, so a
``return`` whose value needs such a copy, or a tail expression that becomes one, is emitted as a local of the
function's return type initialized from the value, followed by a ``return`` of that local. Initializing a local of
the return type applies the conversion the ``return`` would apply, so the returned value is unchanged.
"""

from __future__ import annotations

from ...backend.common_ast import ReturnNode as BackendReturnNode
from ..ast import (
    FunctionCallNode,
    IdentifierNode,
    PointerType,
    ReturnNode,
    VariableNode,
)
from ..refusals import MetalRefusal

__all__ = ["generate_return_writeback_statement"]


def _returned_value(codegen, stmt):
    """The value a ``return`` or a tail expression statement returns, or ``None``."""
    if isinstance(stmt, (ReturnNode, BackendReturnNode)):
        value = getattr(stmt, "value", None)
        return None if isinstance(value, list) else value
    if "ExpressionStatementNode" in type(stmt).__name__ and getattr(
        stmt, "is_tail_expression", False
    ):
        return_type = codegen.type_name_string(codegen.current_function_return_type)
        if return_type and return_type != "void":
            return getattr(stmt, "expression", None)
    return None


def _local_return_type(codegen):
    """The CrossGL type of a local that holds the returned value, or ``None`` when no local can hold it."""
    return_wrapper = codegen.current_function_return_wrapper
    if return_wrapper is not None:
        return_type = return_wrapper.get("source_type")
    else:
        return_type = codegen.current_function_return_type
    if return_type is None or isinstance(return_type, PointerType):
        return None
    if not isinstance(return_type, str):
        return_type = codegen.type_name_string(return_type)
    if not return_type or "[" in return_type or "*" in return_type:
        return None
    return return_type


def generate_return_writeback_statement(codegen, stmt, indent):
    """Emit ``stmt`` when its returned value copies reference arguments in and out, else return ``None``."""
    value = _returned_value(codegen, stmt)
    if value is None:
        return None
    return_type = _local_return_type(codegen)
    if return_type == "void":
        if not isinstance(value, FunctionCallNode):
            return None
        if not codegen.metal_reference_argument_writebacks(value, []):
            return None
        return codegen.generate_statement(value, indent) + codegen.generate_statement(
            ReturnNode(None), indent
        )
    probe = VariableNode("__crossgl_return", return_type or "void", initial_value=value)
    if not codegen.metal_reference_argument_writebacks(probe, []):
        return None
    if return_type is None:
        raise MetalRefusal(
            "Metal cannot copy a reference argument back before returning a value "
            f"of type '{codegen.type_name_string(codegen.current_function_return_type)}'; "
            "assign the call result to a local first"
        )
    codegen.metal_writeback_temporary_count += 1
    name = f"__crossgl_return{codegen.metal_writeback_temporary_count}"
    local = VariableNode(name, return_type, initial_value=value)
    return codegen.generate_statement(local, indent) + codegen.generate_statement(
        ReturnNode(IdentifierNode(name)), indent
    )
