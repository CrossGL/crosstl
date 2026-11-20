"""Interval and alias transfer of declarations, assignments and expressions, for ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

from ..ast import (
    BinaryOpNode,
    IdentifierNode,
    UnaryOpNode,
    VariableNode,
)


class OpenGLWorkgroupPointerExpressionMixin:
    """Declaration, assignment and expression members of ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

    def visit_declaration(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a variable or array declaration and record the alias or interval it binds."""
        initial_value = getattr(value, "initial_value", None)
        self.visit(initial_value, active_aliases, active_intervals)
        name = getattr(value, "name", None)
        if not name:
            return
        binding = self.generator.glsl_workgroup_pointer_binding(
            initial_value,
            active_aliases,
            active_intervals,
        )
        if self.generator.glsl_workgroup_pointer_declaration(value, active_aliases):
            if binding is None:
                active_aliases.pop(name, None)
            else:
                active_aliases[name] = self.retain_refinable_offset(binding)
        else:
            active_aliases.pop(name, None)
        interval = self.generator.glsl_private_pointer_interval(
            initial_value,
            active_intervals,
            self.constants,
        )
        if interval is None:
            active_intervals.pop(name, None)
        else:
            active_intervals[name] = interval
        return

    def visit_assignment(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit an assignment and update or poison the aliases and intervals it writes."""
        target = getattr(value, "target", getattr(value, "left", None))
        assigned = getattr(value, "value", getattr(value, "right", None))
        self.visit(assigned, active_aliases, active_intervals)
        target_name = self.generator.expression_name(target)
        pointer_target = isinstance(target, (str, IdentifierNode, VariableNode))
        current = active_aliases.get(target_name) if pointer_target else None
        operator = self.generator.map_operator(
            getattr(value, "operator", getattr(value, "op", "="))
        )
        if current is not None and current.get("kind") == "workgroup-pointer":
            if operator == "=":
                replacement = self.generator.glsl_workgroup_pointer_binding(
                    assigned,
                    active_aliases,
                    active_intervals,
                )
            elif operator in {"+=", "-="}:
                delta = assigned if operator == "+=" else UnaryOpNode("-", assigned)
                replacement = self.generator.glsl_workgroup_pointer_binding(
                    BinaryOpNode(target, "+", delta),
                    active_aliases,
                    active_intervals,
                )
            else:
                replacement = None
            if replacement is None:
                active_aliases[target_name] = self.poison_alias(current)
            else:
                active_aliases[target_name] = self.retain_refinable_offset(replacement)
            return
        if target_name and isinstance(target, (str, IdentifierNode, VariableNode)):
            assigned_interval = self.generator.glsl_private_pointer_interval(
                assigned,
                active_intervals,
                self.constants,
            )
            current_interval = active_intervals.get(target_name)
            if operator == "=":
                replacement_interval = assigned_interval
            elif operator == "+=" and current_interval and assigned_interval:
                replacement_interval = (
                    current_interval[0] + assigned_interval[0],
                    current_interval[1] + assigned_interval[1],
                )
            elif operator == "-=" and current_interval and assigned_interval:
                replacement_interval = (
                    current_interval[0] - assigned_interval[1],
                    current_interval[1] - assigned_interval[0],
                )
            else:
                replacement_interval = None
            if replacement_interval is None:
                active_intervals.pop(target_name, None)
            else:
                active_intervals[target_name] = replacement_interval
        self.visit(target, active_aliases, active_intervals)
        return

    def visit_dereference(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a pointer dereference and record the element it reads."""
        reinterpretation = self.generator.glsl_storage_struct_reinterpret_node(value)
        if reinterpretation is not None:
            word_count = self.generator.glsl_workgroup_byte_array_copy_word_count(
                reinterpretation,
                active_aliases,
                active_intervals,
            )
            if word_count is not None:
                self.record_access(
                    reinterpretation.expression,
                    0,
                    active_aliases,
                    active_intervals,
                )
                self.record_access(
                    reinterpretation.expression,
                    word_count - 1,
                    active_aliases,
                    active_intervals,
                )
                return
        self.record_access(
            value.operand,
            0,
            active_aliases,
            active_intervals,
        )
        return

    def visit_increment(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit an increment or decrement and update the alias offset or interval it changes."""
        target_name = self.generator.expression_name(value.operand)
        current_alias = active_aliases.get(target_name)
        if (
            current_alias is not None
            and current_alias.get("kind") == "workgroup-pointer"
        ):
            delta = 1 if self.generator.map_operator(value.op) == "++" else -1
            replacement = self.generator.glsl_workgroup_pointer_binding(
                BinaryOpNode(value.operand, "+", delta),
                active_aliases,
                active_intervals,
            )
            active_aliases[target_name] = (
                self.retain_refinable_offset(replacement)
                if replacement is not None
                else self.poison_alias(current_alias)
            )
            return
        current_interval = active_intervals.get(target_name)
        if current_interval is None:
            active_intervals.pop(target_name, None)
        else:
            delta = 1 if self.generator.map_operator(value.op) == "++" else -1
            active_intervals[target_name] = (
                current_interval[0] + delta,
                current_interval[1] + delta,
            )
        self.generator.glsl_clear_index_component_intervals(
            active_intervals,
            target_name,
        )
        return

    def visit_ternary(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit both arms of a conditional expression and merge their states."""
        self.visit(value.condition, active_aliases, active_intervals)
        base_aliases = dict(active_aliases)
        branch_states = []
        for selected, branch in (
            (True, value.true_expr),
            (False, value.false_expr),
        ):
            branch_intervals = (
                self.generator.glsl_private_pointer_refined_condition_intervals(
                    value.condition,
                    active_intervals,
                    self.constants,
                    selected,
                    excluded_names=self.mutated_interval_names,
                )
            )
            if branch_intervals is None:
                continue
            branch_aliases = dict(base_aliases)
            self.visit(branch, branch_aliases, branch_intervals)
            branch_states.append((branch_aliases, branch_intervals))
        if branch_states:
            merged = branch_states[0][1]
            for _, branch_intervals in branch_states[1:]:
                merged = self.merge_intervals(merged, branch_intervals)
            active_intervals.clear()
            active_intervals.update(merged)
            active_aliases.clear()
            active_aliases.update(base_aliases)
            self.merge_alias_states(active_aliases, branch_states)
        return

    def visit_short_circuit(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a logical and/or whose right operand may not run, and merge the states."""
        self.visit(value.left, active_aliases, active_intervals)
        left_aliases = dict(active_aliases)
        left_intervals = dict(active_intervals)
        right_aliases = dict(left_aliases)
        right_intervals = dict(left_intervals)
        self.visit(value.right, right_aliases, right_intervals)
        active_intervals.clear()
        active_intervals.update(self.merge_intervals(left_intervals, right_intervals))
        active_aliases.clear()
        active_aliases.update(left_aliases)
        self.merge_alias_states(
            active_aliases,
            [
                (left_aliases, left_intervals),
                (right_aliases, right_intervals),
            ],
        )
        return
