"""Shader-scope functions that name a stage's declarations become functions of that stage.

A function declared at shader scope can be called from every stage, so the only names it can resolve are the
shader's own. The SPIR-V importer writes every function but the entry point at shader scope, because a SPIR-V
module's functions are module-scope, while it declares the entry point's interface and private variables in the
stage block (``vec2 inUV @input @location(2);``). A helper that reads ``inUV`` then names a declaration it cannot
see: the HLSL and Metal generators wrote ``input.inUV`` in a function that has no ``input``, which DXC and the
Metal compiler reject, and the SPIR-V generator refused the unknown variable. GLSL output happened to work, because
GLSL declares a stage's interface at file scope.

:func:`bind_stage_functions` gives such a function the scope its names need. A shader-scope function *needs* a stage
when its body names a variable, constant buffer or constant-buffer member, structure or function the stage declares
and neither the shader nor the function itself declares, or when it calls a shader-scope function that needs the
stage. Every stage whose entry point or functions call it, directly or through other shader-scope functions, and
that it needs, gets its own copy of the function (each overload, in program order) ahead of the stage's functions.
The shader-scope declaration is removed when every stage that calls it took a copy; otherwise it stays for the stages
that resolve its names at shader scope, as it did before. A function no stage calls is left alone.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from .hub_types import call_name
from .rewrite import child_slots, get_child

__all__ = ["bind_stage_functions"]


def bind_stage_functions(program) -> int:
    """Move shader-scope functions that use a stage's declarations into that stage; returns how many copies."""
    if not isinstance(program, A.ShaderNode):
        return 0
    functions = [
        function
        for function in getattr(program, "functions", None) or []
        if isinstance(function, A.FunctionNode) and isinstance(function.name, str)
    ]
    stages = _stages(program)
    if not functions or not stages:
        return 0
    overloads = {}
    for function in functions:
        overloads.setdefault(function.name, []).append(function)
    shader_names = _shader_names(program)
    free = {name: _free_names(group) for name, group in overloads.items()}
    calls = {
        name: _called_names(group) & set(overloads) for name, group in overloads.items()
    }

    reached_by_stage = [(stage, _reached(stage, overloads)) for stage in stages]
    plans = []
    for stage, reached in reached_by_stage:
        stage_names = _stage_names(stage) - shader_names
        if not stage_names or not reached:
            continue
        needs = {name for name in reached if free[name] & stage_names}
        changed = bool(needs)
        while changed:
            changed = False
            for name in reached:
                if name not in needs and calls[name] & needs:
                    needs.add(name)
                    changed = True
        if needs:
            plans.append((stage, needs))
    if not plans:
        return 0

    # The program and its stages stay shared: a copy must not drag its parent links into the clone.
    memo = {id(program): program}
    memo.update((id(stage), stage) for stage in stages)
    copies = 0
    for stage, needs in plans:
        moved = []
        for function in functions:
            if function.name in needs:
                clone = deepcopy(function, dict(memo))
                clone.parent = stage
                moved.append(clone)
        stage.local_functions = moved + list(
            getattr(stage, "local_functions", None) or []
        )
        copies += len(moved)
    # A shader-scope function goes once every stage that calls it holds its own copy.
    needed = {id(stage): needs for stage, needs in plans}
    kept = set()
    for stage, reached in reached_by_stage:
        kept |= reached - needed.get(id(stage), set())
    moved_everywhere = set().union(*(needs for _stage, needs in plans)) - kept
    program.functions = [
        function
        for function in program.functions
        if not (
            isinstance(function, A.FunctionNode) and function.name in moved_everywhere
        )
    ]
    return copies


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _walk(node):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if id(current) in seen or not isinstance(current, A.ASTNode):
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )


def _declaration_names(nodes):
    names = set()
    for node in nodes or []:
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
        for member in getattr(node, "members", None) or []:
            member_name = getattr(member, "name", None)
            if isinstance(member_name, str):
                names.add(member_name)
    return names


def _shader_names(program):
    names = set()
    for field in (
        "functions",
        "global_variables",
        "constants",
        "cbuffers",
        "structs",
    ):
        names |= _declaration_names(getattr(program, field, None))
    return names


def _stage_names(stage):
    names = set()
    for field in (
        "local_variables",
        "local_functions",
        "local_cbuffers",
        "local_structs",
    ):
        names |= _declaration_names(getattr(stage, field, None))
    return names


def _function_names(function):
    """Names the function declares itself: its parameters, generic parameters and locals."""
    names = {
        parameter.name
        for parameter in getattr(function, "parameters", None) or []
        if isinstance(getattr(parameter, "name", None), str)
    }
    for generic in getattr(function, "generic_params", None) or []:
        name = getattr(generic, "name", generic)
        if isinstance(name, str):
            names.add(name)
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode) and isinstance(node.name, str)
    )
    return names


def _free_names(group):
    """Identifiers the overloads name that they do not declare themselves."""
    free = set()
    for function in group:
        named = {
            node.name
            for node in _walk(getattr(function, "body", None))
            if isinstance(node, A.IdentifierNode) and isinstance(node.name, str)
        }
        free |= named - _function_names(function)
    return free


def _called_names(group):
    return {
        call_name(node)
        for function in group
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.FunctionCallNode)
    }


def _reached(stage, overloads):
    """Shader-scope function names the stage's entry point and functions call, directly or not."""
    roots = [getattr(stage, "entry_point", None)]
    roots += list(getattr(stage, "local_functions", None) or [])
    pending = list(_called_names([root for root in roots if root is not None]))
    reached = set()
    while pending:
        name = pending.pop()
        if name in reached or name not in overloads:
            continue
        reached.add(name)
        pending.extend(_called_names(overloads[name]))
    return reached
