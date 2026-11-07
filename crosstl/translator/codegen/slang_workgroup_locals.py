"""Slang declarations of the workgroup variables a CrossGL function declares in its body.

MSL declares threadgroup memory as a variable of the kernel function's scope, ``threadgroup float tile[256];``
(MSL section 4.4, ``threadgroup`` address space), and the CrossGL importer keeps the declaration there with its
``threadgroup`` qualifier. Slang refuses ``groupshared`` on a function-scope variable with automatic storage
(slangc E31201, "modifier not allowed"). A ``static groupshared`` local is accepted: slangc gives it one instance per
workgroup, a SPIR-V ``Workgroup`` variable, a module-scope HLSL ``groupshared`` variable and a ``threadgroup``
variable of the Metal kernel, so :func:`slang_workgroup_local_prefix` adds ``static`` to the declaration.

A ``static`` local of a helper function is one variable for every call the workgroup makes, the storage that a
module-scope HLSL or GLSL hoist gives it as well.

What the declaration cannot keep is refused through the generator's lowering refusals:

- an initializer, which a groupshared variable cannot have (slangc E30623);
- a pointer into threadgroup memory, which the Slang generator does not lower.
"""

from __future__ import annotations

from ..ast import PointerType

__all__ = [
    "WORKGROUP_QUALIFIERS",
    "is_workgroup_declaration",
    "slang_workgroup_local_prefix",
]

WORKGROUP_QUALIFIERS = frozenset({"groupshared", "shared", "threadgroup", "workgroup"})


def is_workgroup_declaration(node) -> bool:
    """Whether ``node`` declares a variable with a workgroup-storage qualifier."""
    return any(
        str(qualifier).lower() in WORKGROUP_QUALIFIERS
        for qualifier in getattr(node, "qualifiers", None) or ()
    )


def _is_pointer_type(var_type) -> bool:
    if isinstance(var_type, PointerType):
        return True
    return isinstance(var_type, str) and var_type.strip().endswith("*")


def slang_workgroup_local_prefix(codegen, node, initial_value) -> str:
    """The storage prefix of the function-scope declaration ``node``: ``static`` for a workgroup variable.

    A workgroup variable with an initializer, or one of pointer type, is refused through
    ``codegen.refuse_lowering``; the prefix is still returned so that generation reaches the other refusals.
    """
    if not is_workgroup_declaration(node):
        return ""
    name = getattr(node, "name", None)
    if _is_pointer_type(getattr(node, "var_type", None)):
        codegen.refuse_lowering(
            f"Slang has no lowering for the threadgroup pointer '{name}'; the Slang generator declares "
            "workgroup memory as static groupshared variables and does not address it through pointers"
        )
    elif initial_value is not None:
        codegen.refuse_lowering(
            f"Slang groupshared variable '{name}' cannot have an initializer; workgroup memory is "
            "uninitialized until an invocation of the workgroup stores to it"
        )
    return "static "
