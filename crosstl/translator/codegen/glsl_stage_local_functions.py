"""Stage-local functions whose name another stage calls as something else.

A CrossGL stage block may declare helper functions of its own (``StageNode.local_functions``). They are visible in
that stage only: a call with the same name in another stage reaches that stage's own helper, a module-scope function
or a built-in. The GLSL generator writes every stage of a program into one source, keeps one function table for the
whole program, and resolves calls by name. A vertex-stage helper ``void subgroupBarrier()`` therefore captured the
fragment stage's ``subgroupBarrier()`` built-in call: the helper is renamed because it shadows the built-in, the
fragment call was renamed with it, and the fragment section called a function it does not declare (issue #5026).

:func:`rename_stage_local_functions_called_elsewhere` renames, in a copy of the program, each stage-local function
whose name a call in another stage uses where that stage declares no function of the name, together with the calls
that do reach it. The calls in the other stages keep the name, so they reach what they named in the source.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..semantics.function_name_collisions import rename_functions
from ..semantics.hub_types import call_name

__all__ = ["rename_stage_local_functions_called_elsewhere"]


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _names(functions):
    return {
        function.name
        for function in functions or ()
        if isinstance(getattr(function, "name", None), str)
    }


def _calls(root, seen):
    """Every function call under ``root``, depth first, skipping nodes already in ``seen``."""
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, A.FunctionCallNode):
            yield node
        for field, value in reversed(list(vars(node).items())):
            if field == "parent":
                continue
            if isinstance(value, A.ASTNode):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(
                    item for item in reversed(value) if isinstance(item, A.ASTNode)
                )


def _foreign_stage_calls(program):
    """``(names, call ids)``: stage-local names that another stage calls where it declares no such function."""
    stages = _stages(program)
    module_names = _names(getattr(program, "functions", None))
    local_names = [_names(getattr(stage, "local_functions", None)) for stage in stages]
    candidates = set().union(*local_names) - module_names if local_names else set()
    if not candidates:
        return set(), set()
    names, call_ids = set(), set()
    for stage, own_names in zip(stages, local_names):
        seen = set()
        roots = list(getattr(stage, "local_functions", None) or [])
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            roots.append(entry)
        for root in roots:
            for call in _calls(root, seen):
                name = call_name(call).partition("<")[0].strip()
                if name in candidates and name not in own_names:
                    names.add(name)
                    call_ids.add(id(call))
    return names, call_ids


def rename_stage_local_functions_called_elsewhere(program):
    """``program``, or a copy whose stage-local functions that another stage's calls do not reach are renamed."""
    names, _call_ids = _foreign_stage_calls(program)
    if not names:
        return program
    program = deepcopy(program)
    names, call_ids = _foreign_stage_calls(program)
    rename_functions(program, names, keep_calls=frozenset(call_ids))
    return program
