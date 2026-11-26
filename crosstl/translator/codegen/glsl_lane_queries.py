"""Subgroup lane queries under OpenGL software-subgroup lowering.

A lane query (``WaveGetLaneIndex()``, Metal's ``simd_lane_id`` call form)
exchanges nothing between invocations. Under software lowering the lane is
``gl_LocalInvocationIndex`` modulo the subgroup width under any control
flow, so a function that only queries its lane needs no exchange record and
reads the lane from the entry's ``gl_SubgroupInvocationID`` alias.
"""

from ..ast import FunctionCallNode, WaveOpNode

LANE_QUERY = "WaveGetLaneIndex"


def is_lane_query(codegen, node):
    """Whether ``node`` is a lane query: a wave op or its call spelling."""
    if isinstance(node, WaveOpNode):
        return node.operation == LANE_QUERY
    if isinstance(node, FunctionCallNode):
        name = codegen.function_call_name(node)
        return codegen.glsl_wave_operation_name(name) == LANE_QUERY
    return False


def lane_query_expression(codegen, operation, arguments):
    """The software lane for a zero-argument lane query, or ``None``."""
    if codegen.software_subgroup_width is None:
        return None
    if operation != LANE_QUERY or arguments:
        return None
    return codegen.stage_input_builtin_alias("gl_SubgroupInvocationID", "compute")
