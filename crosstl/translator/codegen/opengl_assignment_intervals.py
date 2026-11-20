"""Interval updates for assignments to named GLSL integers and aggregates, for ``GLSLControlFlowIntervalAnalysis.visit_assignment``."""

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from .opengl_aggregate_assignment_intervals import (
    OpenGLAggregateAssignmentIntervalMixin,
)


class OpenGLAssignmentIntervalMixin(OpenGLAggregateAssignmentIntervalMixin):
    """Named-target members of ``GLSLControlFlowIntervalAnalysis.visit_assignment``."""

    def update_named_assignment_intervals(
        self,
        value,
        intervals,
        assigned,
        assigned_members,
        target,
        target_name,
    ):
        """Write the interval an assignment gives a named target, or clear the keys it makes unknown."""
        if target_name and isinstance(target, (str, IdentifierNode, VariableNode)):
            target_type = self.expression_source_type(target)
            assigned_interval = self.checked_interval(assigned, intervals, target_type)
            assigned_components = (
                self.checked_components(
                    assigned,
                    target_type,
                    intervals,
                )
                if isinstance(assigned, IdentifierNode)
                and target_type is not None
                and self.generator.type_name_string(target_type)
                == self.generator.type_name_string(
                    self.expression_source_type(assigned)
                )
                else None
            )
            operator = self.generator.map_operator(
                getattr(value, "operator", getattr(value, "op", "="))
            )
            current = intervals.get(target_name)
            if operator == "=":
                replacement = assigned_interval
            elif operator in {"+=", "-="} and current and assigned_interval:
                replacement = self.checked_interval(
                    BinaryOpNode(target, operator[0], assigned),
                    intervals,
                    target_type,
                )
            else:
                replacement = None
            replacement = self.generator.glsl_integer_storage_interval(
                replacement, target_type
            )
            self.clear_interval_targets(intervals, {target_name})
            if replacement is not None:
                intervals[target_name] = replacement
            if operator == "=":
                for component, bounds in enumerate(assigned_components or ()):
                    key = self.generator.glsl_index_component_interval_key(
                        target_name, component
                    )
                    if key is not None:
                        intervals[key] = bounds
                for member, interval in assigned_members.items():
                    intervals[f"{target_name}.{member}"] = interval
        elif isinstance(target, MemberAccessNode):
            self.update_member_assignment_intervals(
                value, intervals, assigned, assigned_members, target
            )
        elif isinstance(target, ArrayAccessNode):
            self.update_element_assignment_intervals(value, intervals, assigned, target)
