"""Workgroup variables that a CrossGL entry point declares in its body, which WGSL declares at module scope.

MSL declares threadgroup memory as a variable of the kernel function's scope, ``threadgroup float tile[256];``
(MSL section 4.4, ``threadgroup`` address space), and the CrossGL importer keeps the declaration there with its
``threadgroup`` qualifier. WGSL declares ``var<workgroup>`` at module scope only (WGSL section 7.3, ``var``
declarations); a function-scope ``var`` holds one value per invocation, so a local emitted in place would give every
invocation a private array and lose each write that another invocation of the workgroup makes.

:func:`hoist_workgroup_locals` moves each workgroup local of an entry point to the program's module-scope
variables, keeping its name when no other declaration of the program uses it and taking a fresh one otherwise, and
renames the entry's references. :class:`WGSLWorkgroupLocalError` refuses what the move cannot keep:

- a workgroup local of a helper function, which one module-scope variable would share between calls;
- one with an initializer, which a workgroup variable cannot have (WGSL section 7.3);
- one whose name the entry declares more than once, or the program declares at module scope, whose references
  the move could not tell apart;
- one whose type names a value the entry declares, which module scope cannot see.
"""

from __future__ import annotations

from collections import Counter

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["WGSLWorkgroupLocalError", "hoist_workgroup_locals"]

WORKGROUP_QUALIFIERS = frozenset({"threadgroup", "workgroup", "shared", "groupshared"})
_DECLARATIONS = (
    A.VariableNode,
    A.ParameterNode,
    A.FunctionNode,
    A.StructNode,
    A.ConstantNode,
)


class WGSLWorkgroupLocalError(TranslationDiagnostic):
    """A function-scope workgroup variable that WGSL output cannot move to module scope."""

    def __init__(self, name, function, reason, detail):
        self.variable_name = name
        self.function_name = function
        self.refusal = reason
        self.missing_capabilities = ("wgsl.function-scope-workgroup-variable",)
        super().__init__(
            f"WGSL target cannot declare workgroup variable '{name}' of '{function}' at module scope, where WGSL "
            f"declares workgroup memory: {detail}",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-workgroup-local-{reason}",
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def _is_workgroup_local(node) -> bool:
    return isinstance(node, A.VariableNode) and any(
        str(qualifier).lower() in WORKGROUP_QUALIFIERS
        for qualifier in getattr(node, "qualifiers", None) or []
    )


def _walk(node):
    return node.walk() if hasattr(node, "walk") else ()


def _declared_names(node) -> Counter:
    return Counter(
        found.name
        for found in _walk(node)
        if isinstance(found, _DECLARATIONS) and isinstance(found.name, str)
    )


def _functions(program, entry_stage):
    """Every function of ``program``, each with whether it is a stage's entry point or a top-level entry."""
    found = [
        (function, bool(entry_stage(function)))
        for function in getattr(program, "functions", None) or []
    ]
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        for function in getattr(stage, "local_functions", None) or []:
            found.append((function, function is entry))
        if entry is not None and all(function is not entry for function, _ in found):
            found.append((entry, True))
    return [
        (function, is_entry)
        for function, is_entry in found
        if isinstance(function, A.FunctionNode)
    ]


def _remove_statement(body, statement) -> None:
    for block in _walk(body):
        statements = getattr(block, "statements", None)
        if isinstance(statements, list) and any(
            item is statement for item in statements
        ):
            statements[:] = [item for item in statements if item is not statement]
            return


def _fresh_name(base, taken) -> str:
    index = 1
    while f"{base}_{index}" in taken:
        index += 1
    name = f"{base}_{index}"
    taken.add(name)
    return name


def _check_hoistable(local, function, entry_names, module_names) -> None:
    name = local.name
    if local.initial_value is not None:
        raise WGSLWorkgroupLocalError(
            name,
            function.name,
            "initializer",
            "a workgroup variable has no initializer",
        )
    if entry_names[name] > 1 or name in module_names:
        raise WGSLWorkgroupLocalError(
            name,
            function.name,
            "ambiguous-name",
            "another declaration of the name would be renamed with it",
        )
    referenced = {
        found.name
        for found in _walk(local.var_type)
        if isinstance(found, A.IdentifierNode)
    }
    local_names = sorted(name for name in referenced if entry_names[name])
    if local_names:
        raise WGSLWorkgroupLocalError(
            name,
            function.name,
            "local-dimension",
            "its type names "
            + ", ".join(local_names)
            + ", which the entry point declares",
        )


def _rename(body, old, new) -> None:
    for found in _walk(body):
        if isinstance(found, A.IdentifierNode) and found.name == old:
            found.name = new


def hoist_workgroup_locals(program, entry_stage) -> int:
    """Move the workgroup locals of ``program``'s entry points to module scope, in place; return the count.

    ``entry_stage(function)`` names the stage a top-level function is the entry point of, or returns ``None``.
    """
    if program is None:
        return 0
    functions = _functions(program, entry_stage)
    hoisted = []
    for function, is_entry in functions:
        locals_ = [node for node in _walk(function.body) if _is_workgroup_local(node)]
        if locals_ and not is_entry:
            raise WGSLWorkgroupLocalError(
                locals_[0].name,
                function.name,
                "helper-function",
                "only an entry point's workgroup locals have one instance per workgroup",
            )
        if locals_:
            hoisted.append((function, locals_))
    if not hoisted:
        return 0
    declared = _declared_names(program)
    in_functions = Counter()
    for function, _ in functions:
        in_functions.update(_declared_names(function))
    module_names = {
        name for name, total in declared.items() if total > in_functions[name]
    }
    taken = set(declared)
    count = 0
    for function, locals_ in hoisted:
        entry_names = _declared_names(function)
        for local in locals_:
            _check_hoistable(local, function, entry_names, module_names)
        for local in locals_:
            name = local.name
            if declared[name] > 1:
                local.name = _fresh_name(name, taken)
                _rename(function.body, name, local.name)
            _remove_statement(function.body, local)
            program.global_variables.append(local)
            count += 1
    return count
