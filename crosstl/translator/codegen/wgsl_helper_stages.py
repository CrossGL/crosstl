"""The stage each WGSL helper function runs in, so a helper reached only from compute entries may hold a barrier.

WGSL's ``workgroupBarrier``, ``storageBarrier`` and ``textureBarrier`` are valid in any function a compute entry point
calls (WGSL sections 13.3 and 17.11), as GLSL's ``barrier()`` is valid in any function a compute shader calls (GLSL
4.60 section 8.16). The WGSL emitter writes helpers once per module, outside any entry point, so it used to refuse a
barrier in every helper. :func:`helper_stage_names` finds, for each function the emitted entry points reach, the
stages that reach it; a helper reached by exactly one stage is emitted in that stage's context, so a barrier in a
helper only compute entries call is written, and an implicit-LOD lookup in such a helper uses the explicit level the
entry point itself would use. A helper reached by several stages, or by none, keeps no stage, and a barrier in it is
still refused, because WGSL rejects a barrier reached from a vertex or fragment entry point.

Calls are resolved by :class:`~.call_graph.CallGraph`, which also backs the recursion check, so a helper's callers are
those of the overload the call selects.
"""

from __future__ import annotations

from .call_graph import CallGraph
from .stage_utils import normalize_stage_name

__all__ = ["helper_stage_names"]


def helper_stage_names(program, stage_nodes):
    """Map ``id(function)`` to the one stage whose emitted entry points reach it, for every such function."""
    graph = CallGraph(program)
    index_of = {id(function): index for index, function in enumerate(graph.functions)}
    reached: dict[int, set[str]] = {}
    for stage_node in stage_nodes:
        entry = getattr(stage_node, "entry_point", None)
        start = index_of.get(id(entry))
        if start is None:
            continue
        stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
        seen = {start}
        pending = [start]
        while pending:
            index = pending.pop()
            reached.setdefault(index, set()).add(stage_name)
            for callee in graph.edges[index]:
                if callee not in seen:
                    seen.add(callee)
                    pending.append(callee)
    return {
        id(graph.functions[index]): next(iter(stages))
        for index, stages in reached.items()
        if len(stages) == 1
    }
