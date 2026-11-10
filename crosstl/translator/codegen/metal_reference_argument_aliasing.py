"""Metal reference arguments that a direct ``thread T&`` binding would alias.

CrossGL follows GLSL for ``out`` and ``inout`` parameters (OpenGL Shading Language 4.60, section 6.1.1, "Function
Calling Conventions"): an ``inout`` argument is copied into the parameter when its turn comes, the parameter is a
variable of its own while the function runs, and it is copied back to the argument when the function returns.
Metal lowers these parameters to ``thread T&``. A thread local passed that way is the parameter, so another write
to the local, or another binding of it, while the call is evaluated reaches the parameter too, which copy-in and
copy-out semantics rule out:

* another argument of the call writes the local: ``scale(m, bump(m))`` hands ``scale`` the value ``bump`` left in
  ``m``, where GLSL copied ``m`` in before ``bump`` ran (C++ leaves the order of the two unspecified as well);
* another argument passes the local to a second reference or pointer parameter: ``h(m, m)`` makes both parameters
  one variable, so the first parameter sees the second one's writes;
* the callee reads or writes the variable itself: a module-scope variable that Metal forwards to the function as a
  reference parameter of its own (``f(g)`` where ``f`` assigns ``g``) is the same variable as the parameter.

Another argument that only reads the local is evaluated before the call runs, whichever order C++ takes, and sees
the value GLSL gives it. An argument that aliases is passed through a thread temporary instead, with the copy-in and
copy-out lowering that arguments outside the thread address space already use
(``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``, with the ordering in
:mod:`.metal_reference_argument_order` and :mod:`.metal_reference_call_results`).
"""

from __future__ import annotations

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerType,
    SwizzleNode,
    UnaryOpNode,
)
from .wgsl_conditional_effects import is_effectful_builtin

__all__ = [
    "metal_parameter_reaches_argument_storage",
    "metal_reference_argument_aliases",
]

_REFERENCE_QUALIFIERS = frozenset({"inout", "out"})


def _call_arguments(call):
    return list(getattr(call, "args", None) or getattr(call, "arguments", None) or [])


def _operator(node):
    return str(getattr(node, "op", getattr(node, "operator", "")) or "")


def _root_name(expr):
    """The variable an l-value (or ``&l-value``) names, through member, component and element accesses."""
    if (
        isinstance(expr, UnaryOpNode)
        and _operator(expr) == "&"
        and not getattr(expr, "is_postfix", False)
    ):
        expr = expr.operand
    while True:
        if isinstance(expr, MemberAccessNode):
            expr = expr.object
        elif isinstance(expr, SwizzleNode):
            expr = expr.vector_expr
        elif isinstance(expr, ArrayAccessNode):
            expr = expr.array
        else:
            break
    return expr.name if isinstance(expr, IdentifierNode) else None


def _forwarded_globals(codegen, name):
    """Module-scope variables Metal forwards to the program function ``name`` (those it reads or writes)."""
    if not isinstance(name, str):
        return set()
    return set(codegen.required_function_thread_private_global_argument_names(name))


def metal_parameter_reaches_argument_storage(codegen, call, index):
    """Whether parameter ``index`` of a program call reaches its argument's storage: a reference or a pointer."""
    if (
        codegen.metal_reference_parameter_qualifier(call, index)
        in _REFERENCE_QUALIFIERS
    ):
        return True
    for overload in codegen.function_overloads_by_name.get(
        codegen.function_call_name(call), []
    ):
        parameters = list(
            getattr(overload, "parameters", getattr(overload, "params", [])) or []
        )
        if index < len(parameters):
            raw_type = codegen.parameter_raw_type(parameters[index])
            if isinstance(raw_type, PointerType) or codegen.is_array_type_node(
                raw_type
            ):
                return True
    return False


def _call_writes(codegen, call, root):
    name = codegen.function_call_name(call)
    arguments = _call_arguments(call)
    if isinstance(name, str) and name in codegen.function_overloads_by_name:
        if root in _forwarded_globals(codegen, name):
            return True
        return any(
            _root_name(argument) == root
            and metal_parameter_reaches_argument_storage(codegen, call, index)
            for index, argument in enumerate(arguments)
        )
    if not isinstance(name, str) or is_effectful_builtin(name):
        return any(_root_name(argument) == root for argument in arguments)
    return False


def _writes(codegen, expr, root):
    """Whether evaluating ``expr`` can write the variable ``root``."""
    for node in codegen.iter_ast_nodes(expr):
        if isinstance(node, (AssignmentNode, BackendAssignmentNode)):
            if _root_name(getattr(node, "target", getattr(node, "left", None))) == root:
                return True
        elif isinstance(node, UnaryOpNode) and _operator(node) in {"++", "--"}:
            if _root_name(node.operand) == root:
                return True
        elif isinstance(node, FunctionCallNode) and _call_writes(codegen, node, root):
            return True
    return False


def metal_reference_argument_aliases(codegen, call, index, parameter):
    """Whether argument ``index`` of the program call ``call`` must not bind its ``thread T&`` parameter directly.

    The argument is a thread l-value. It must not bind directly when the callee reads or writes its variable as a
    module-scope variable, another argument binds the variable to a reference or pointer parameter, or another
    argument can write it. A parameter that names the ``thread`` address space (``inout thread T``, as the Metal
    front end imports ``thread T&``) is a reference in the source, whose aliasing the source intends.
    """
    qualifiers = {
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", None) or []
    }
    if "thread" in qualifiers:
        return False
    arguments = _call_arguments(call)
    root = _root_name(arguments[index])
    if root is None:
        return False
    if root in _forwarded_globals(codegen, codegen.function_call_name(call)):
        return True
    for other_index, other in enumerate(arguments):
        if other_index == index:
            continue
        if _root_name(other) == root and metal_parameter_reaches_argument_storage(
            codegen, call, other_index
        ):
            return True
        if _writes(codegen, other, root):
            return True
    return False
