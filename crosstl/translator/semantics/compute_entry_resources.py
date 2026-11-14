"""Bound resources of ``@ compute fn`` entries, declared as module globals for generators that bind only globals.

A CUDA, HIP, OpenCL or Metal kernel reaches CrossGL as a top-level ``@ compute fn`` whose buffers and textures are
parameters carrying ``@ group`` and ``@ binding``. The Slang generator binds resources only where they are declared
at module scope, as a stage block declares them, and used to emit such parameters as plain arrays of an ordinary
function. For those targets this pass declares each bound parameter as a module global of the same name, with its
attributes, and removes it from the entry's parameter list, so the body's references resolve to the global:

- ``var<storage, read> a: array<T>`` becomes ``StructuredBuffer<T> a``, and ``var<storage, read_write>`` (or
  ``var<storage>``) becomes ``RWStructuredBuffer<T>``, the types a stage block's ``buffer T a[]`` lowers to;
- a texture, sampler or buffer parameter with an ``@ binding`` keeps its type.

Two entries may share a resource only when they declare it identically. A resource whose name or binding another
declaration already uses, and a storage parameter that is not a runtime-sized array, are refused with
:class:`ComputeEntryResourceError` rather than rebound silently.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .compute_entry_values import (
    _binding_of,
    _global_names,
    _is_compute_entry,
)
from .opaque_structs import _is_opaque

__all__ = ["ComputeEntryResourceError", "TARGETS", "hoist_compute_entry_resources"]

TARGETS = frozenset({"slang"})


class ComputeEntryResourceError(TranslationDiagnostic):
    """A bound parameter of a compute entry that cannot become a module global without changing its meaning."""

    code = CODE_PREFIX + "compute-entry-resource"


def _qualifiers(parameter) -> list[str]:
    return [
        str(q).lower() for q in getattr(parameter, "resource_qualifiers", None) or []
    ]


def _global_type(function, parameter, target):
    """The module-global type of a bound entry parameter, or ``None`` when the parameter is not a resource."""
    qualifiers = _qualifiers(parameter)
    parameter_type = getattr(parameter, "param_type", None)
    if "storage" in qualifiers:
        if (
            not isinstance(parameter_type, A.ArrayType)
            or parameter_type.size is not None
        ):
            raise ComputeEntryResourceError(
                f"compute entry {function.name} binds storage parameter {parameter.name} of type "
                f"{parameter_type}, which is not a runtime-sized array; {target} declares bound storage as a "
                "structured buffer of elements",
                target=target,
            )
        read_only = "read" in qualifiers and "read_write" not in qualifiers
        kind = "StructuredBuffer" if read_only else "RWStructuredBuffer"
        return A.NamedType(kind, [parameter_type.element_type])
    if _binding_of(parameter)[1] is not None and _is_opaque(parameter_type):
        return parameter_type
    return None


def _signature(variable) -> tuple:
    """What makes two hoisted declarations the same resource: name, type and binding."""
    return (variable.name, repr(variable.var_type), _binding_of(variable))


def _hoist_entry(program, function, declared, global_names, taken, target) -> None:
    kept = []
    for parameter in getattr(function, "parameters", None) or []:
        resource_type = _global_type(function, parameter, target)
        if resource_type is None:
            kept.append(parameter)
            continue
        variable = A.VariableNode(
            parameter.name,
            resource_type,
            attributes=list(getattr(parameter, "attributes", None) or []),
        )
        variable.vtype = resource_type
        signature = _signature(variable)
        if parameter.name in declared:
            if declared[parameter.name] != signature:
                raise ComputeEntryResourceError(
                    f"compute entry {function.name} declares resource {parameter.name} differently from another "
                    "entry; one module global cannot be both",
                    target=target,
                )
            continue
        binding = _binding_of(variable)
        if parameter.name in global_names or binding in taken:
            raise ComputeEntryResourceError(
                f"compute entry {function.name} binds {parameter.name} at group {binding[0]}, binding "
                f"{binding[1]}, but a module declaration already uses that name or binding",
                target=target,
            )
        declared[parameter.name] = signature
        taken.add(binding)
        program.global_variables.append(variable)
    function.parameters = kept


def hoist_compute_entry_resources(program, target) -> None:
    """Declare the bound parameters of ``program``'s ``@ compute fn`` entries as module globals for ``target``."""
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return
    entries = [
        function
        for function in getattr(program, "functions", None) or []
        if isinstance(function, A.FunctionNode) and _is_compute_entry(function)
    ]
    if not entries:
        return
    if getattr(program, "global_variables", None) is None:
        program.global_variables = []
    global_names = _global_names(program)
    # Bindings of the module's own declarations; the entries' parameters are what this pass places.
    taken = set()
    for declaration in list(program.global_variables) + list(
        getattr(program, "cbuffers", None) or []
    ):
        group, binding = _binding_of(declaration)
        if binding is not None:
            taken.add((group, binding))
    declared: dict = {}
    for function in entries:
        _hoist_entry(program, function, declared, global_names, taken, target)
