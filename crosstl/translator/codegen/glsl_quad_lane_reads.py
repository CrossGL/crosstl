"""GLSL output of ``QuadReadLaneAt`` with a lane index computed at run time.

GLSL's ``subgroupQuadBroadcast(value, id)`` takes ``id`` as an integral constant expression (GL_KHR_shader_subgroup,
"Quad Functions"), so glslang rejects a lane read from memory or computed by the program. HLSL's
``QuadReadLaneAt``, Metal's ``quad_broadcast`` and CrossGL's ``QuadReadLaneAt`` accept such a lane. A lane index
that is not an integer literal is read through ``crossglQuadReadLaneAt``: every invocation of the quad broadcasts
the value of each of the four quad lanes with a constant index, and the helper returns the one the lane selects.
That is the value ``subgroupQuadBroadcast`` reads for the lane, in every stage where quad operations run, because
the quad membership is the one ``subgroupQuadBroadcast`` itself uses. A lane above 3, which HLSL and Metal leave
undefined, reads quad lane ``lane & 3``, as CUDA and HIP output read it.
"""

from __future__ import annotations

from .array_utils import evaluate_literal_int_expression

__all__ = [
    "QUAD_LANE_READ_HELPER",
    "emit_glsl_quad_lane_read_helpers",
    "glsl_quad_lane_read_call",
    "quad_lane_read_needs_helper",
]

QUAD_LANE_READ_HELPER = "crossglQuadReadLaneAt"


def quad_lane_read_needs_helper(operation, arguments) -> bool:
    """Whether a ``QuadReadLaneAt`` call has a lane GLSL cannot pass to ``subgroupQuadBroadcast``."""
    if operation != "QuadReadLaneAt" or len(arguments or ()) != 2:
        return False
    return evaluate_literal_int_expression(arguments[1]) is None


def glsl_quad_lane_read_call(codegen, arguments) -> str:
    """``crossglQuadReadLaneAt(value, lane)``, declaring the overload for the value's GLSL type."""
    value_type = codegen.expression_result_type(arguments[0])
    mapped_type = codegen.map_type(value_type) if value_type is not None else None
    value = codegen.generate_expression(arguments[0])
    lane = codegen.generate_expression(arguments[1])
    lane_type = codegen.expression_result_type(arguments[1])
    if lane_type is None or codegen.map_type(lane_type) != "uint":
        lane = f"uint({lane})"
    helpers = codegen.__dict__.setdefault("glsl_quad_lane_read_helpers", {})
    value_types = (
        (mapped_type,)
        if mapped_type in codegen.GLSL_WAVE_HELPER_VALUE_TYPES
        else codegen.GLSL_WAVE_HELPER_VALUE_TYPES
    )
    for glsl_type in value_types:
        if glsl_type not in helpers:
            helpers[glsl_type] = _helper_definition(glsl_type)
    return f"{QUAD_LANE_READ_HELPER}({value}, {lane})"


def emit_glsl_quad_lane_read_helpers(codegen) -> str:
    """The ``crossglQuadReadLaneAt`` overloads the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("glsl_quad_lane_read_helpers", {}).values())


def _helper_definition(glsl_type: str) -> str:
    reads = "".join(
        f"    {glsl_type} lane{index} = subgroupQuadBroadcast(value, {index}u);\n"
        for index in range(4)
    )
    return (
        f"{glsl_type} {QUAD_LANE_READ_HELPER}({glsl_type} value, uint lane) {{\n"
        f"{reads}"
        "    uint index = lane & 3u;\n"
        "    return index == 0u ? lane0 : (index == 1u ? lane1 : "
        "(index == 2u ? lane2 : lane3));\n"
        "}\n\n"
    )
