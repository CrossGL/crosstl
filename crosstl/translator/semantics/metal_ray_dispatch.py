"""Ray generation dispatch values and ray flag constants spelled the way the Metal generator reads them.

A Metal ray tracing pipeline has no ray generation stage of its own: the host dispatches a compute kernel over the
launch grid, and the kernel traces rays with ``intersector<>::intersect`` (Metal Shading Language Specification,
section 6.18 "Ray-Tracing Functions"). CrossGL's ``ray_generation`` stage therefore becomes a ``kernel``, but the
generator printed the GLSL launch built-ins (``gl_LaunchIDEXT``, ``gl_LaunchSizeEXT``) and the DXR intrinsics
(``DispatchRaysIndex()``, ``DispatchRaysDimensions()``) verbatim, which MSL does not declare, and the GLSL ray flag
constants (``gl_RayFlagsOpaqueEXT``) likewise.

:func:`lower_metal_ray_dispatch` rewrites them for the Metal target:

* in a ray generation stage, the launch index becomes ``gl_GlobalInvocationID``, which the generator receives as
  the kernel's ``[[thread_position_in_grid]]``, and the launch size becomes ``gl_LaunchSizeEXT``, which it receives
  as ``[[threads_per_grid]]``: both ``uint3``, the type of the GLSL built-ins and the DXR intrinsics;
* a GLSL ray flag constant becomes the DXR constant of the same value (``GL_EXT_ray_tracing`` section 8.19 and the
  DXR ``RAY_FLAG`` enumeration agree bit for bit), which the generator prints as its value.

Refused with :class:`MetalRayDispatchUnsupportedError`:

* the launch index or size in any other ray stage: those stages become ``[[visible]]`` functions, which receive no
  grid position (MSL section 5.1.4 lists the attributes a kernel's parameters may take);
* a GLSL ``traceRayEXT`` whose payload location names a ``rayPayloadEXT`` variable. Only the closest-hit and miss
  shaders a trace invokes write that payload, and ``intersector<>::intersect`` invokes neither: Metal has no shader
  binding table and returns the ``intersection_result`` to the kernel instead. The generator declared the payload as
  a zero constant, so the kernel read zero where the traced value belonged.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hlsl_ray_interfaces import (
    _CONSTANTS,
    _attribute_names,
    _literal_int,
    _location,
    _stage_label,
    _stages,
    _transform,
    _walk,
)

__all__ = ["MetalRayDispatchUnsupportedError", "lower_metal_ray_dispatch"]

TARGETS = frozenset({"metal"})
S = A.ShaderStage
_RAY_STAGES = frozenset(
    {
        S.RAY_GENERATION,
        S.RAY_INTERSECTION,
        S.RAY_CLOSEST_HIT,
        S.RAY_MISS,
        S.RAY_ANY_HIT,
        S.RAY_CALLABLE,
    }
)
# Launch value spellings -> the name the Metal generator binds to a kernel attribute.
_LAUNCH_INDEX = "gl_GlobalInvocationID"
_LAUNCH_SIZE = "gl_LaunchSizeEXT"
_LAUNCH_VARIABLES = {
    "gl_LaunchIDEXT": _LAUNCH_INDEX,
    "gl_LaunchIDNV": _LAUNCH_INDEX,
    "gl_LaunchSizeEXT": _LAUNCH_SIZE,
    "gl_LaunchSizeNV": _LAUNCH_SIZE,
}
_LAUNCH_CALLS = {
    "DispatchRaysIndex": _LAUNCH_INDEX,
    "DispatchRaysDimensions": _LAUNCH_SIZE,
}
_LAUNCH_NAMES = {**_LAUNCH_VARIABLES, **_LAUNCH_CALLS}
_RAY_FLAGS = {
    name: value for name, value in _CONSTANTS.items() if name.startswith("gl_RayFlags")
}
_PAYLOAD_ATTRIBUTES = frozenset({"raypayloadext", "raypayloadnv"})


class MetalRayDispatchUnsupportedError(TranslationDiagnostic):
    """A ray tracing launch value or trace that a Metal kernel or visible function cannot express."""

    def __init__(self, stage, capability, detail):
        self.stage = stage
        self.missing_capabilities = (f"metal.{capability}",)
        super().__init__(
            f"Metal {stage} stage: {detail}",
            target="metal",
            code=f"{CODE_PREFIX}metal-ray-dispatch-unsupported",
        )


def _call_name(node):
    if not isinstance(node, A.FunctionCallNode) or node.arguments:
        return None
    function = node.function
    name = function.name if isinstance(function, A.IdentifierNode) else function
    return name if isinstance(name, str) and name in _LAUNCH_CALLS else None


def _launch_name(node):
    if isinstance(node, A.IdentifierNode) and node.name in _LAUNCH_VARIABLES:
        return node.name
    return _call_name(node)


def _payload_variables(program, stage):
    """Module- and stage-scope ``rayPayloadEXT`` variables, as (name, location) pairs."""
    variables = [
        *(getattr(program, "global_variables", None) or []),
        *(getattr(stage, "local_variables", None) or []),
    ]
    return [
        (variable.name, _location(variable))
        for variable in variables
        if any(
            name.lower() in _PAYLOAD_ATTRIBUTES for name in _attribute_names(variable)
        )
    ]


def _traced_payload(call, payloads):
    """The ``rayPayloadEXT`` variable a GLSL-form trace names by its location, or None."""
    if call.operation != "TraceRay" or len(call.arguments) != 11:
        return None
    location = _literal_int(call.arguments[10])
    for name, declared in payloads:
        if location is None or declared is None or declared == location:
            return name
    return None


def _stage_functions(program, stage):
    functions = list(getattr(stage, "local_functions", None) or [])
    if getattr(stage, "entry_point", None) is not None:
        functions.append(stage.entry_point)
    if len(_stages(program)) == 1:
        # Module-scope helpers belong to the program's only stage.
        functions.extend(getattr(program, "functions", None) or [])
    return functions


def _rewrite(node):
    if isinstance(node, A.IdentifierNode) and node.name in _RAY_FLAGS:
        return A.IdentifierNode(_RAY_FLAGS[node.name])
    name = _launch_name(node)
    if name is None or name == _LAUNCH_NAMES[name]:
        return None
    return A.IdentifierNode(_LAUNCH_NAMES[name])


def _refuse_launch_values(stage, functions, where):
    for function in functions:
        for node in _walk(function):
            name = _launch_name(node)
            if name is not None:
                raise MetalRayDispatchUnsupportedError(
                    _stage_label(stage),
                    "ray-launch-value",
                    f"{name} reads the ray launch grid, which only the ray generation kernel's entry point "
                    f"receives ([[thread_position_in_grid]], [[threads_per_grid]]); {where}, so pass the value "
                    "to it as an argument",
                )


def _refuse_traced_payloads(program, stage, functions):
    payloads = _payload_variables(program, stage)
    if not payloads:
        return
    for function in functions:
        for node in _walk(function):
            if not isinstance(node, A.RayTracingOpNode):
                continue
            name = _traced_payload(node, payloads)
            if name is not None:
                raise MetalRayDispatchUnsupportedError(
                    _stage_label(stage),
                    "ray-payload-shader-binding-table",
                    f"traceRayEXT writes rayPayloadEXT '{name}' through the closest-hit and miss shaders it "
                    "invokes, but Metal has no shader binding table: intersector<>::intersect invokes no "
                    "closest-hit or miss function and returns an intersection_result to the kernel instead",
                )


def lower_metal_ray_dispatch(program, target) -> int:
    """Rewrite ``program``'s ray launch values and ray flags for Metal; returns how many ray stages it visited."""
    if str(target or "").lower() not in TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    visited = 0
    for stage in _stages(program):
        if getattr(stage, "stage", None) not in _RAY_STAGES:
            continue
        visited += 1
        functions = _stage_functions(program, stage)
        if stage.stage is S.RAY_GENERATION:
            _refuse_traced_payloads(program, stage, functions)
            helpers = [f for f in functions if f is not stage.entry_point]
            for helper in helpers:
                _refuse_launch_values(
                    stage, [helper], f"helper '{helper.name}' is an ordinary function"
                )
        else:
            _refuse_launch_values(
                stage, functions, "this stage becomes a [[visible]] function"
            )
        for function in functions:
            _transform(function, _rewrite)
    return visited
