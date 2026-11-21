"""Workgroup variables that CUDA and HIP helper functions read or write.

CrossGL declares a workgroup variable (``shared float tile[64];``) at stage scope, where every function of the stage
sees it. The CUDA and HIP generators declare it ``__shared__`` inside the kernel, which only the kernel's own body
can name, so a stage function that used it failed to compile (nvcc: ``identifier "tile" is undefined``). CUDA and
HIP also accept ``__shared__`` at file scope (CUDA C++ Programming Guide, "__shared__"): the variable still lives in
the shared memory of each thread block and is visible to every function of the translation unit. A workgroup
variable that a stage function names is therefore declared at file scope, before the functions; one that only the
kernel names stays in the kernel.
"""

from __future__ import annotations

from ..ast import FunctionNode, IdentifierNode, ParameterNode, VariableNode
from .ast_walk import walk_ast_values

__all__ = ["helper_shared_variables"]


def _names_used(function, names):
    """The members of ``names`` that ``function`` refers to without declaring them itself."""
    used, bound = set(), set()
    for value in walk_ast_values(function):
        if isinstance(value, IdentifierNode) and value.name in names:
            used.add(value.name)
        elif isinstance(value, (VariableNode, ParameterNode)) and value is not function:
            if getattr(value, "name", None) in names:
                bound.add(value.name)
    return used - bound


def helper_shared_variables(stage, is_shared):
    """The workgroup variables of ``stage`` that one of its functions other than the entry point uses.

    ``is_shared`` tells a workgroup variable from the stage's other variables. The result keeps the stage's
    declaration order.
    """
    shared = [
        var
        for var in getattr(stage, "local_variables", None) or []
        if is_shared(var) and getattr(var, "name", None)
    ]
    if not shared:
        return []
    names = {var.name for var in shared}
    entry = getattr(stage, "entry_point", None)
    used = set()
    for function in getattr(stage, "local_functions", None) or []:
        if function is entry or not isinstance(function, FunctionNode):
            continue
        used |= _names_used(function, names)
    return [var for var in shared if var.name in used]
