"""``OpSelect`` imported with the CrossGL form that matches its condition.

With a scalar ``bool`` condition, ``OpSelect`` picks one whole operand, which is CrossGL's ``?:``. With a vector
condition it selects per component (SPIR-V 1.6 section 3.52.15: "If Condition is a vector, Result Type must be a
vector with the same number of components ... the result is the component-wise selection"). CrossGL's ``?:`` needs
a scalar condition, as GLSL's does, so the component-wise form is ``mix(false_value, true_value, condition)`` with a
boolean selector (GLSL 4.60 section 8.3), which every emitter lowers to a selection and never to arithmetic.
"""

from __future__ import annotations

from .VulkanAst import FunctionCallNode, TernaryOpNode

__all__ = ["spirv_select_expression"]


def _is_boolean_vector(type_id, types):
    info = types.get(type_id) or {}
    if info.get("kind") != "vector":
        return False
    component = types.get(info.get("component_type")) or {}
    return component.get("kind") == "scalar" and component.get("name") == "bool"


def spirv_select_expression(
    condition, true_value, false_value, condition_type_id, types
):
    """The CrossGL expression of ``OpSelect`` for a condition of type ``condition_type_id``."""
    if _is_boolean_vector(condition_type_id, types):
        return FunctionCallNode("mix", [false_value, true_value, condition])
    return TernaryOpNode(condition, true_value, false_value)
