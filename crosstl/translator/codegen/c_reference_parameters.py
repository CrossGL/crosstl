"""CrossGL ``out`` and ``inout`` parameters in CUDA and HIP output.

CrossGL follows GLSL 4.60 section 6.1.1: an ``out`` or ``inout`` argument is copied back to the caller when the
callee returns. C++ has no copy-out parameter, so the CUDA and HIP emitters declare such a parameter as an lvalue
reference (C++17 [dcl.init.ref]), which binds the caller's object. The two agree while nothing else reaches that
object during the call. Locals of the caller are reachable only through the parameter, so the one case left is a
call that passes overlapping storage to an ``out`` or ``inout`` parameter and to another parameter that also reaches
the caller's object: another ``out``/``inout`` parameter, or an array parameter, which C++ passes as a pointer.
Copy-in and copy-out keep those apart until the callee returns, whereas a reference writes the shared object as
the callee runs. :func:`overlapping_output_arguments` finds those calls so the emitters can refuse them. A scalar,
vector or structure passed by value is a copy in both languages.

Fixed-size array parameters already pass the caller's elements in C++ ([dcl.fct]/5), so an ``inout`` array is
declared as a reference to the array, ``T (&v)[N]``, which keeps its extent.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)

__all__ = [
    "OUTPUT_QUALIFIERS",
    "is_output_parameter",
    "overlapping_output_arguments",
    "reference_declarator",
]

OUTPUT_QUALIFIERS = frozenset({"out", "inout"})


def is_output_parameter(parameter) -> bool:
    """Whether CrossGL ``parameter`` is declared ``out`` or ``inout``."""
    return bool(
        {str(q).lower() for q in getattr(parameter, "qualifiers", []) or []}
        & OUTPUT_QUALIFIERS
    )


def reference_declarator(declarator: str, name: str) -> str:
    """``declarator`` of ``name`` as an lvalue reference: ``float x`` -> ``float& x``, ``float v[2]`` -> ``float (&v)[2]``.

    Returns ``declarator`` unchanged when it already declares a reference or a pointer to ``name``, or does not
    name it.
    """
    pattern = re.compile(rf"(?<![\w&*])({re.escape(name)})(?=\s*(\[|$))")
    matches = list(pattern.finditer(declarator))
    if not matches:
        return declarator
    match = matches[-1]
    head, tail = declarator[: match.start()], declarator[match.end() :]
    if head.rstrip().endswith(("&", "*")):
        return declarator
    if tail.lstrip().startswith("["):
        return f"{head.rstrip()} (&{name}){tail}"
    return f"{head.rstrip()}& {name}{tail}"


def _storage_root(expression):
    """The variable an lvalue argument names storage in, or ``None`` for an argument that is not an lvalue path."""
    current = expression
    while isinstance(current, (ArrayAccessNode, MemberAccessNode)):
        if isinstance(current, ArrayAccessNode):
            current = getattr(current, "array", getattr(current, "array_expr", None))
        else:
            current = current.object
    if isinstance(current, IdentifierNode):
        return current.name
    if isinstance(current, VariableNode):
        return current.name
    if isinstance(current, str):
        return current
    return None


def _reaches_caller(parameter):
    parameter_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    return is_output_parameter(parameter) or isinstance(parameter_type, ArrayType)


def overlapping_output_arguments(parameters, arguments):
    """The argument roots that a call passes to an ``out``/``inout`` parameter and to another caller-reaching one.

    ``parameters`` are the callee's CrossGL parameters and ``arguments`` the call's CrossGL arguments, in order.
    Two paths with one root, such as ``v.x`` and ``v.y`` or ``a[0]`` and ``a[1]``, count as overlapping: whether
    they name one element is not always known when the program is emitted.
    """
    roots = {}
    for parameter, argument in zip(parameters, arguments):
        root = _storage_root(argument)
        if root is None:
            continue
        roots.setdefault(root, []).append(
            (is_output_parameter(parameter), _reaches_caller(parameter))
        )
    return sorted(
        root
        for root, uses in roots.items()
        if any(output for output, _ in uses) and sum(reach for _, reach in uses) > 1
    )
