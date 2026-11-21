"""Refuse CrossGL call cycles that pass through more than one function before HLSL is written.

DXC rejects every recursive call graph ("recursive functions are not allowed"). The HLSL generator lowers a
function's direct void self tail calls to a loop and refuses its other self calls itself (see
``hlsl_tail_recursive_void_call_ids``). A cycle through two or more functions, such as ``even -> odd -> even``,
cannot be lowered that way, so it fails closed here with :class:`DirectXRecursiveFunctionError` instead of reaching
DXC.

The cycles come from the shared static call graph (:class:`~.call_graph.CallGraph`), the same graph that the
Vulkan, WGSL and Metal generators refuse cycles from. Self edges are left out of the search so that the HLSL
generator's own handling of direct recursion still applies.
"""

from __future__ import annotations

from .. import ast as A
from .call_graph import CallGraph
from .directx_values import DirectXRecursiveFunctionError

__all__ = ["find_hlsl_indirect_call_cycle", "validate_hlsl_call_cycles"]


def find_hlsl_indirect_call_cycle(graph):
    """The first cycle of two or more functions in declaration order: ``(function indices, closing call)``.

    The indices start and end with the same function. Returns ``None`` when every cycle is a direct self call.
    """
    edges = [
        {
            target: call
            for target, call in graph.edges[source].items()
            if target != source
        }
        for source in range(len(graph.functions))
    ]
    state = [0] * len(graph.functions)  # 0 unvisited, 1 on the stack, 2 done
    for root in range(len(graph.functions)):
        if state[root]:
            continue
        state[root] = 1
        stack = [root]
        pending = [iter(sorted(edges[root]))]
        while pending:
            source = stack[-1]
            target = next(pending[-1], None)
            if target is None:
                pending.pop()
                state[stack.pop()] = 2
                continue
            if state[target] == 1:
                return stack[stack.index(target) :] + [target], edges[source][target]
            if state[target] == 0:
                state[target] = 1
                stack.append(target)
                pending.append(iter(sorted(edges[target])))
    return None


def validate_hlsl_call_cycles(program):
    """Raise :class:`DirectXRecursiveFunctionError` when ``program`` has a call cycle through several functions."""
    if not isinstance(program, A.ShaderNode):
        return
    graph = CallGraph(program)
    found = find_hlsl_indirect_call_cycle(graph)
    if found is None:
        return
    indices, call = found
    cycle = tuple(graph.functions[index].name for index in indices)
    raise DirectXRecursiveFunctionError(
        f"DirectX does not support the recursive call cycle '{' -> '.join(cycle)}': DXC rejects recursive "
        "functions, and only a function's own void tail calls can be lowered to a loop",
        function_name=cycle[0],
        cycle=cycle,
        reason="indirect-recursion-unsupported",
        source_location=getattr(call, "source_location", None),
    )
