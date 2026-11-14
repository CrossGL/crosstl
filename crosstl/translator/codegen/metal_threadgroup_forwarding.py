"""Threadgroup storage that Metal helper functions reach through ``threadgroup`` references.

Metal declares threadgroup memory only inside a kernel (or object, mesh or tile) function, or receives it as a
``threadgroup`` pointer or reference argument (Metal Shading Language Specification 4.4, "Threadgroup Address
Space"). Program-scope variables are in the ``constant`` address space. Two CrossGL forms place threadgroup memory
where a helper function sees it by name:

* a program-scope ``groupshared``/``shared`` variable, as HLSL declares every ``groupshared`` array;
* a ``shared`` variable declared in a CrossGL stage block, next to the stage's helper functions.

Each entry that reaches such a variable, directly or through calls, declares it as a ``threadgroup`` local, and each
helper that reaches it receives the entry's storage as a ``threadgroup`` reference parameter appended to its
signature and to every call. A reference to the entry's variable is the same memory for every thread of the
threadgroup, so the source semantics are kept exactly.

A variable that cannot be forwarded this way (a helper declares a parameter or local of the same name, an array
dimension is unknown, or a stage outside compute, object and mesh reaches it) keeps the existing behaviour: a
program-scope variable stays refused, and a stage-block variable is refused here.
"""

from __future__ import annotations

from ..ast import ArrayType
from .stage_utils import normalize_stage_name, stage_matches

__all__ = ["collect_metal_forwarded_threadgroup_storage"]

_THREADGROUP_QUALIFIERS = frozenset(
    {"shared", "groupshared", "threadgroup", "workgroup"}
)


def _qualifier_names(node):
    names = {
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", []) or []
    }
    names |= {
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", []) or []
    }
    return names


def _is_stage_threadgroup_variable(node):
    qualifiers = _qualifier_names(node)
    return (
        bool(qualifiers & _THREADGROUP_QUALIFIERS)
        and "threadgroup_imageblock" not in qualifiers
    )


def threadgroup_reference_parameter(codegen, node, name):
    """Return the ``threadgroup`` reference declaration that forwards ``node``, or ``None``.

    Every array dimension must be known: Metal binds a reference to an array of a fixed size only.
    """
    vtype = getattr(node, "vtype", getattr(node, "var_type", None))
    sizes = []
    while isinstance(vtype, ArrayType):
        size = getattr(vtype, "size", None)
        if size is None:
            return None
        sizes.append(codegen.expression_to_string(size))
        vtype = vtype.element_type
    array_size = getattr(node, "array_size", None)
    if array_size is not None:
        sizes.append(codegen.expression_to_string(array_size))
    type_name = codegen.type_name_string(vtype)
    if not type_name or "[" in type_name:
        return None
    mapped_type = codegen.map_type(type_name)
    if not mapped_type:
        return None
    if sizes:
        suffix = "".join(f"[{size}]" for size in sizes)
        return f"threadgroup {mapped_type} (&{name}){suffix}"
    return f"threadgroup {mapped_type}& {name}"


def _reaching_helpers(direct_users, helper_calls):
    """Close ``direct_users`` over the helper call graph: a helper that calls a reaching helper reaches too."""
    reaching = set(direct_users)
    changed = True
    while changed:
        changed = False
        for helper_name in sorted(helper_calls):
            if helper_name not in reaching and helper_calls[helper_name] & reaching:
                reaching.add(helper_name)
                changed = True
    return reaching


def _direct_helper_users(codegen, helpers, name):
    return {
        func.name
        for func in helpers
        if codegen.node_uses_identifier(getattr(func, "body", []), name)
    }


def _shadowed(codegen, helpers, reaching, name):
    return any(
        name in codegen.metal_function_declared_names(func)
        for func in helpers
        if func.name in reaching
    )


def _record_forwarded(codegen, reaching, name, declaration, vtype):
    for helper_name in sorted(reaching):
        codegen.metal_forwarded_thread_private_globals.setdefault(
            helper_name, []
        ).append((name, declaration, vtype))


def _program_scope_entries(codegen, entries, reaching, name, entry_stages):
    """The entries that need the variable as a local, or ``None`` when one of them cannot declare it."""
    used_entries = []
    for stage_name, func, stage_locals in entries:
        if not (
            codegen.node_uses_identifier(getattr(func, "body", []), name)
            or codegen.called_user_function_names(func) & reaching
        ):
            continue
        if stage_name not in entry_stages or name in codegen.metal_entry_local_names(
            func, stage_locals
        ):
            return None
        used_entries.append(func)
    return used_entries


def _forward_program_scope_storage(
    codegen, global_vars, entries, helpers, helper_calls, entry_stages
):
    lowered_by_function = {}
    lowered_ids = set()
    for node in global_vars or []:
        if not codegen.is_program_scope_groupshared_global(node):
            continue
        if codegen.program_scope_groupshared_initializer(node) is not None:
            continue
        name = codegen.metal_variable_name(node)
        if not name:
            continue
        direct_users = _direct_helper_users(codegen, helpers, name)
        if not direct_users:
            # Only entries use it: the entry-local lowering already covers it.
            continue
        reaching = _reaching_helpers(direct_users, helper_calls)
        declaration = threadgroup_reference_parameter(codegen, node, name)
        if declaration is None or _shadowed(codegen, helpers, reaching, name):
            continue
        used_entries = _program_scope_entries(
            codegen, entries, reaching, name, entry_stages
        )
        if used_entries is None:
            continue
        _record_forwarded(
            codegen, reaching, name, declaration, getattr(node, "vtype", None)
        )
        for func in used_entries:
            lowered_by_function.setdefault(id(func), []).append(node)
        lowered_ids.add(id(node))
    return lowered_by_function, lowered_ids


def _forward_stage_block_storage(codegen, ast, target_stage, helpers, helper_calls):
    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not stage_matches(target_stage, stage_name):
            continue
        entry_point = getattr(stage, "entry_point", None)
        stage_helper_ids = {
            id(func)
            for func in getattr(stage, "local_functions", []) or []
            if func is not entry_point
        }
        stage_helpers = [func for func in helpers if id(func) in stage_helper_ids]
        for node in getattr(stage, "local_variables", []) or []:
            if not _is_stage_threadgroup_variable(node):
                continue
            name = codegen.metal_variable_name(node)
            if not name:
                continue
            direct_users = _direct_helper_users(codegen, stage_helpers, name)
            if not direct_users:
                continue
            reaching = _reaching_helpers(direct_users, helper_calls)
            declaration = threadgroup_reference_parameter(codegen, node, name)
            if declaration is None:
                codegen.refuse_lowering(
                    f"threadgroup variable '{name}' is used by helper function(s) "
                    f"{', '.join(sorted(reaching))}, but its type has no fixed size, so Metal cannot pass it "
                    "to them as a threadgroup reference"
                )
                continue
            if _shadowed(codegen, helpers, reaching, name):
                codegen.refuse_lowering(
                    f"threadgroup variable '{name}' is used by helper function(s) "
                    f"{', '.join(sorted(reaching))}, and one of them declares its own '{name}', so Metal "
                    "cannot pass the threadgroup storage to it under that name"
                )
                continue
            _record_forwarded(
                codegen, reaching, name, declaration, getattr(node, "vtype", None)
            )


def collect_metal_forwarded_threadgroup_storage(
    codegen, ast, target_stage, global_vars, all_functions, entry_stages
):
    """Forward threadgroup storage to the helper functions that reach it.

    Records each forwarded variable in ``codegen.metal_forwarded_thread_private_globals`` (the per-helper list of
    ``(name, reference declaration, type)`` that signatures and call sites read) and returns
    ``(lowered_by_function, lowered_ids)``: the program-scope variables each entry declares as a local, and the
    ids of those variables, which are no longer emitted at program scope.
    """
    entries = codegen.metal_program_scope_groupshared_stage_entries(ast, target_stage)
    entry_ids = {id(func) for _stage_name, func, _stage_locals in entries}
    helpers = [
        func
        for func in all_functions or []
        if id(func) not in entry_ids and getattr(func, "name", None)
    ]
    helper_calls = {}
    for func in helpers:
        helper_calls.setdefault(func.name, set()).update(
            codegen.called_user_function_names(func)
        )
    lowered = _forward_program_scope_storage(
        codegen, global_vars, entries, helpers, helper_calls, entry_stages
    )
    _forward_stage_block_storage(codegen, ast, target_stage, helpers, helper_calls)
    return lowered
