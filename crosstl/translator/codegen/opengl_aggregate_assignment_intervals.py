"""Interval updates for assignments to struct members and array elements, for ``OpenGLAssignmentIntervalMixin.update_named_assignment_intervals``."""

from ..ast import (
    BinaryOpNode,
)
from ..validation import (
    expression_debug_name,
)


class OpenGLAggregateAssignmentIntervalMixin:
    """Member and element members of ``OpenGLAssignmentIntervalMixin.update_named_assignment_intervals``."""

    def update_member_assignment_intervals(
        self,
        value,
        intervals,
        assigned,
        assigned_members,
        target,
    ):
        """Write or clear the interval of an assigned struct member."""
        self.clear_interval_targets(
            intervals,
            self.target_interval_keys(target),
        )
        component_index = self.generator.glsl_index_component_index(target.member)
        key = self.generator.glsl_index_component_interval_key(
            target.object,
            component_index,
        )
        field_key = self.generator.glsl_declared_struct_member_interval_key(target)
        if field_key is not None:
            key = field_key
            component_index = None
        if key is None:
            key = expression_debug_name(target)
        if key is not None:
            assigned_interval = self.checked_interval(assigned, intervals)
            operator = self.generator.map_operator(
                getattr(value, "operator", getattr(value, "op", "="))
            )
            current = getattr(
                value,
                "_glsl_control_flow_intervals",
                {},
            ).get(key)
            if operator == "=":
                replacement = assigned_interval
            elif operator in {"+=", "-="} and current and assigned_interval:
                replacement = self.checked_interval(
                    BinaryOpNode(target, operator[0], assigned),
                    {**intervals, key: current},
                )
            else:
                replacement = None
            target_type = self.expression_source_type(
                target.object if component_index is not None else target
            )
            target_info = self.generator.glsl_value_type_info(target_type)
            replacement = (
                self.generator.glsl_integer_storage_interval(
                    replacement,
                    target_type,
                    component=component_index is not None,
                )
                if target_info is not None and target_info["family"] in {"int", "uint"}
                else None
            )
            if replacement is None:
                intervals.pop(key, None)
            else:
                intervals[key] = replacement
            if operator == "=":
                for member, interval in assigned_members.items():
                    intervals[f"{key}.{member}"] = interval

    def update_element_assignment_intervals(
        self,
        value,
        intervals,
        assigned,
        target,
    ):
        """Write or clear the interval of an assigned array element."""
        index_bounds = self.checked_interval(target.index, intervals)
        array_name = self.generator.expression_name(target.array)
        source_bounds = self.checked_interval(target, intervals)
        operator = self.generator.map_operator(value.operator)
        assigned_bounds = self.checked_integer_interval(assigned, intervals)
        replacement = None
        extent = self.pointer_array_extent(target.array)
        if operator == "=" and extent is not None:
            replacement = self.checked_interval(assigned, intervals)
        if source_bounds is not None and assigned_bounds is not None:
            if operator == "+=":
                replacement = (
                    source_bounds[0] + assigned_bounds[0],
                    source_bounds[1] + assigned_bounds[1],
                )
            elif operator == "-=":
                replacement = (
                    source_bounds[0] - assigned_bounds[1],
                    source_bounds[1] - assigned_bounds[0],
                )
        key = (
            self.generator.glsl_pointer_array_interval_key(array_name, index_bounds[0])
            if index_bounds is not None
            and index_bounds[0] == index_bounds[1]
            and extent is not None
            and 0 <= index_bounds[0] < extent
            else None
        )
        known_element = key is not None and (key in intervals or operator == "=")
        self.clear_interval_targets(
            intervals,
            {key} if key is not None else self.target_interval_keys(target),
        )
        if (
            known_element
            and replacement is not None
            and (0 <= replacement[0] <= replacement[1] < (1 << 31))
        ):
            intervals[key] = replacement
