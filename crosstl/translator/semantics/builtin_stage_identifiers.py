"""``gl_`` built-in variables read or written by name are refused in a stage GLSL does not declare them in.

GLSL declares each built-in variable for particular stages (GLSL 4.60 section 7.1, "Built-In Language Variables").
:func:`~.builtin_stages.refuse_misplaced_builtins` checks the built-ins an entry point carries as attributes, but a
function body may also name one directly, as GLSL source does: a fragment stage reading ``gl_VertexID`` was written
unchanged by the GLSL, HLSL, Metal and Slang generators, and glslang rejects it as an undeclared identifier.

:func:`refuse_misplaced_builtin_identifiers` walks, for each stage, its entry point, its stage-local functions and the
program-scope functions the entry point reaches through calls. An identifier that names a built-in of
:data:`~.builtin_stages.BUILTIN_STAGE_RULES` and is not declared by the program, the stage or the function is refused
with :class:`~.builtin_values.BuiltinValueUnsupportedError` when GLSL declares that built-in in neither direction for
the stage. Whether a read or write fits the built-in's direction is left to the generators, which already check it.
"""

from __future__ import annotations

from collections import deque

from .. import ast as A
from .builtin_stages import _KNOWN_STAGES, _stage_name, _where, builtin_stage_rule
from .builtin_values import _INPUT, _OUTPUT, BuiltinValueUnsupportedError, _stages
from .hub_types import call_name
from .rewrite import child_slots, get_child

__all__ = ["refuse_misplaced_builtin_identifiers"]


def _walk(node, seen):
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _nodes(function):
    return list(_walk(getattr(function, "body", None), set()))


def _declared(function, nodes):
    names = {
        getattr(p, "name", None) for p in getattr(function, "parameters", None) or []
    }
    names.update(node.name for node in nodes if isinstance(node, A.VariableNode))
    return names


def _program_names(program, stages):
    names = {v.name for v in getattr(program, "global_variables", None) or []}
    for stage in stages:
        names.update(v.name for v in getattr(stage, "local_variables", None) or [])
    return names


def _stage_functions(stage, program_functions):
    """The stage's entry point and local functions, then the program functions they reach through calls."""
    entry = getattr(stage, "entry_point", None)
    local = [f for f in getattr(stage, "local_functions", None) or [] if f is not entry]
    functions = ([entry] if entry is not None else []) + local
    local_names = {getattr(f, "name", None) for f in functions}
    reached, queue = [], deque(functions)
    visited = {id(f) for f in functions}
    while queue:
        function = queue.popleft()
        reached.append(function)
        for node in _nodes(function):
            if not isinstance(node, A.FunctionCallNode):
                continue
            name = call_name(node)
            if not name or name in local_names:
                continue
            for callee in program_functions.get(name, ()):
                if id(callee) not in visited:
                    visited.add(id(callee))
                    queue.append(callee)
    return reached


def refuse_misplaced_builtin_identifiers(program, target) -> int:
    """Refuse built-in variables named in a stage GLSL does not declare them in; returns how many were checked.

    Raises :class:`~.builtin_values.BuiltinValueUnsupportedError` for the first misplaced built-in.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    stages = _stages(program)
    program_names = _program_names(program, stages)
    program_functions = {}
    for function in getattr(program, "functions", None) or []:
        program_functions.setdefault(getattr(function, "name", None), []).append(
            function
        )
    checked = 0
    for stage in stages:
        stage_name = _stage_name(getattr(stage, "stage", None))
        if stage_name not in _KNOWN_STAGES:
            continue
        for function in _stage_functions(stage, program_functions):
            nodes = _nodes(function)
            declared = program_names | _declared(function, nodes)
            for node in nodes:
                if not isinstance(node, A.IdentifierNode) or node.name in declared:
                    continue
                rule = builtin_stage_rule(node.name)
                if rule is None or not node.name.lower().startswith("gl_"):
                    continue
                checked += 1
                if stage_name in rule[_INPUT] or stage_name in rule[_OUTPUT]:
                    continue
                raise BuiltinValueUnsupportedError(
                    f"{node.name} in '{getattr(function, 'name', '')}' of the {stage_name} stage is not "
                    f"a {stage_name} built-in in GLSL; {_where(node.name, rule)} (GLSL 4.60 section 7.1)",
                    str(target or ""),
                )
    return checked
