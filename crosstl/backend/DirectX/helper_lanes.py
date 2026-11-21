"""``IsHelperLane()`` outside pixel shaders.

Shader Model 6.6 makes ``IsHelperLane()`` available in every stage, and outside pixel shaders, where no lane is a
helper, it returns ``false`` (DirectX-Specs, "HLSL Helper Lanes"; DXC compiles it in a compute shader to the
constant). CrossGL reads the value through ``gl_HelperInvocation``, which, as in
GLSL, only a fragment stage declares. The importer therefore writes ``false`` for a call in a function that only
non-pixel entry points reach, and ``gl_HelperInvocation`` everywhere else, where CrossGL checks the stage of every
entry point that reaches it.
"""

from __future__ import annotations

from .DirectxParser import FunctionCallNode

__all__ = ["helper_lanes_always_false"]

_FRAGMENT = "fragment"


def _called_names(function, children, known):
    """Names of the program functions ``function`` calls directly."""
    found = set()
    pending = [getattr(function, "body", None)]
    seen = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str):
            if node.name in known:
                found.add(node.name)
        pending.extend(children(node))
    return found


def _reaching_stages(source_functions, children):
    """Map each function name to the stages of the entry points that call it, directly or not."""
    calls = {
        name: (
            set().union(
                *(
                    _called_names(function, children, source_functions)
                    for function in overloads
                )
            )
        )
        for name, overloads in source_functions.items()
    }
    stages = {}
    for name in sorted(source_functions):
        for function in source_functions[name]:
            stage = getattr(function, "qualifier", None)
            if not stage:
                continue
            pending = [name]
            reached = set()
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                stages.setdefault(current, set()).add(stage)
                pending.extend(sorted(calls.get(current, ())))
    return stages


def helper_lanes_always_false(generator, function) -> bool:
    """Whether ``IsHelperLane()`` in ``function`` is ``false`` because no pixel shader reaches it."""
    if function is None:
        return False
    stage = getattr(function, "qualifier", None)
    if stage:
        return stage != _FRAGMENT
    cache = getattr(generator, "_helper_lane_stages", None)
    if cache is None or cache[0] is not generator.source_functions:
        cache = (
            generator.source_functions,
            _reaching_stages(generator.source_functions, generator.iter_ast_children),
        )
        generator._helper_lane_stages = cache
    stages = cache[1].get(function.name, set())
    return bool(stages) and _FRAGMENT not in stages
