"""GLSL built-in functions that are only available in particular shader stages.

Unlike built-in variables, these restrictions apply to calls in function bodies. A
stage may call a program-scope helper, so the check follows the functions reachable
from each stage's entry point before refusing an unsupported call.
"""

from __future__ import annotations

from .. import ast as A
from .builtin_stage_identifiers import _nodes, _stage_functions
from .builtin_stages import _stage_name
from .builtin_values import BuiltinValueUnsupportedError, _stages
from .hub_types import call_name

__all__ = ["refuse_misplaced_builtin_functions"]

_GLSL_TARGETS = frozenset({"opengl"})

# GLSL's GL_KHR_shader_subgroup makes this barrier available to compute shaders;
# GL_EXT_mesh_shader extends that availability to task and mesh shaders.
_BUILTIN_FUNCTION_STAGES = {
    "subgroupMemoryBarrierShared": frozenset({"compute", "mesh", "task"}),
}


def refuse_misplaced_builtin_functions(program, target) -> int:
    """Refuse GLSL built-in calls made from stages where GLSL does not provide them.

    Only GLSL output needs this check: other backends have their own subgroup
    operations and stage rules. Returns the number of restricted calls checked,
    or raises :class:`BuiltinValueUnsupportedError` for the first misplaced call.
    """
    target_name = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target_name not in _GLSL_TARGETS:
        return 0

    stages = _stages(program)
    program_functions = {}
    defined_names = set()
    for function in getattr(program, "functions", None) or []:
        name = getattr(function, "name", None)
        if name:
            defined_names.add(name)
            program_functions.setdefault(name, []).append(function)

    for stage in stages:
        functions = [getattr(stage, "entry_point", None)]
        functions.extend(getattr(stage, "local_functions", None) or [])
        defined_names.update(
            name for name in (getattr(function, "name", None) for function in functions)
            if name
        )

    checked = 0
    for stage in stages:
        stage_name = _stage_name(getattr(stage, "stage", None))
        for function in _stage_functions(stage, program_functions):
            for node in _nodes(function):
                if not isinstance(node, A.FunctionCallNode):
                    continue
                name = call_name(node)
                allowed_stages = _BUILTIN_FUNCTION_STAGES.get(name)
                if allowed_stages is None or name in defined_names:
                    continue
                checked += 1
                if stage_name in allowed_stages:
                    continue
                allowed = ", ".join(sorted(allowed_stages))
                function_name = getattr(function, "name", "")
                raise BuiltinValueUnsupportedError(
                    f"{name} in '{function_name}' of the {stage_name} stage "
                    "is unavailable in GLSL; "
                    f"GLSL declares it only in the {allowed} stages",
                    target_name,
                )

    return checked
