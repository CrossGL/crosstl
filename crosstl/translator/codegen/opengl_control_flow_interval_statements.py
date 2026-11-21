"""Per-statement interval transfer of ``GLSLControlFlowIntervalAnalysis.visit``: declarations, assignments, calls, increments, blocks, branches and loops."""

from ..ast import (
    ArrayNode,
    DoWhileNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..unhandled import UNHANDLED
from .loop_guard_bounds import (
    guarded_additive_loop_entries,
)
from .opengl_assignment_intervals import OpenGLAssignmentIntervalMixin
from .opengl_loop_replay_intervals import OpenGLLoopReplayIntervalMixin


class OpenGLControlFlowIntervalStatementMixin(
    OpenGLAssignmentIntervalMixin, OpenGLLoopReplayIntervalMixin
):
    """Statement members of ``GLSLControlFlowIntervalAnalysis.visit``."""

    def visit_declaration(
        self,
        value,
        intervals,
    ):
        """Record the interval of a declared scalar or array variable."""
        initial_value = getattr(value, "initial_value", None)
        self.visit(initial_value, intervals)
        name = getattr(value, "name", None)
        if name:
            variable_type = (
                self.lexical["declarations"].get(id(value), {}).get("source_type")
            )
            self.clear_interval_targets(intervals, {name})
            interval = self.generator.glsl_integer_storage_interval(
                self.checked_interval(initial_value, intervals, variable_type),
                variable_type,
            )
            if (
                interval is None
                and self.generator.glsl_storage_pointer_element_type(value) is not None
            ):
                asserted = self.generator.glsl_index_asserted_range(
                    initial_value,
                    function_name=(
                        getattr(self.function, "_glsl_resource_source_name", None)
                        or getattr(self.function, "_generic_source_name", None)
                        or getattr(self.function, "name", None)
                    ),
                )
                if asserted is not None:
                    interval = (asserted.minimum, asserted.maximum)
            if interval is None:
                intervals.pop(name, None)
            else:
                intervals[name] = interval
            component_intervals = self.checked_components(
                initial_value,
                variable_type,
                intervals,
            )
            for component_index, component_interval in enumerate(
                component_intervals or ()
            ):
                key = self.generator.glsl_index_component_interval_key(
                    name,
                    component_index,
                )
                if key is not None:
                    intervals[key] = component_interval
            member_intervals = (
                self.generator.glsl_simple_struct_constructor_member_intervals(
                    initial_value,
                    intervals,
                    self.constants,
                )
            )
            for member, member_interval in member_intervals.items():
                intervals[f"{name}.{member}"] = member_interval
        return

    def visit_assignment(
        self,
        value,
        intervals,
    ):
        """Update the intervals an assignment writes, or clear the targets it makes unknown."""
        target = getattr(value, "target", getattr(value, "left", None))
        assigned = getattr(value, "value", getattr(value, "right", None))
        self.visit(assigned, intervals)
        assigned_members = (
            self.generator.glsl_simple_struct_constructor_member_intervals(
                assigned, intervals, self.constants
            )
        )
        target_name = self.generator.expression_name(target)
        self.update_named_assignment_intervals(
            value, intervals, assigned, assigned_members, target, target_name
        )
        self.visit(target, intervals)
        return

    def visit_call(
        self,
        value,
        intervals,
    ):
        """Visit a call's arguments and clear the intervals the callee may mutate."""
        function_expr = getattr(value, "function", None)
        self.visit(function_expr, intervals)
        arguments = list(value.arguments or [])
        for argument in arguments:
            self.visit(argument, intervals)
        argument_mutations = (
            [self.mutation_interval_keys(argument) for argument in arguments]
            if any(
                isinstance(argument, UnaryOpNode)
                and self.generator.map_operator(argument.op) == "&"
                for argument in arguments
            )
            else []
        )
        for argument_index, argument in enumerate(arguments):
            if self.pointer_array_extent(argument) is not None:
                # An array argument refers to the current slots, including
                # effects from evaluating other arguments of this call.
                self.annotate_expression(argument, intervals)
            elif (
                isinstance(argument, UnaryOpNode)
                and self.generator.map_operator(argument.op) == "&"
            ):
                other_mutations = set().union(
                    *(
                        mutations
                        for index, mutations in enumerate(argument_mutations)
                        if index != argument_index
                    )
                )
                argument._glsl_pointer_argument_mutated = (
                    self.generator.glsl_interval_key_sets_overlap(
                        self.target_interval_keys(argument.operand), other_mutations
                    )
                )
        updates = self.aggregate_updates(value, intervals)
        function_name = self.generator.function_call_name(value)
        for argument_index, argument in enumerate(arguments):
            self.clear_interval_targets(
                intervals,
                self.call_argument_mutation_keys(
                    function_name,
                    arguments,
                    argument_index,
                    argument,
                ),
            )
        if updates is not None:
            intervals.update(updates)
        return

    def visit_increment(
        self,
        value,
        intervals,
    ):
        """Step the interval of an incremented or decremented integer."""
        self.visit(value.operand, intervals)
        name = getattr(value.operand, "name", None)
        current = intervals.get(name)
        domain = self.lexical_integer_domain(value.operand)
        replacement = None
        if (
            current is not None
            and domain is not None
            and name not in self.escaped_names
            and domain.minimum <= current[0] <= current[1] <= domain.maximum
        ):
            step = self.unit_increment(value.op)
            candidate = (current[0] + step, current[1] + step)
            if domain.minimum <= candidate[0] <= candidate[1] <= domain.maximum:
                replacement = candidate
        self.clear_interval_targets(
            intervals,
            self.target_interval_keys(value.operand),
        )
        if replacement is not None:
            intervals[name] = replacement
        return

    def visit_block(
        self,
        value,
        intervals,
    ):
        """Visit a block's statements in order, recording unreachable tails."""
        outer_keys = set(intervals)
        block_intervals = dict(intervals)
        shadowed_values = {}
        block_names = set()
        statements = list(getattr(value, "statements", []) or [])
        continuing = True
        for index, statement in enumerate(statements):
            if isinstance(statement, (VariableNode, ArrayNode)):
                name = getattr(statement, "name", None)
                if name:
                    block_names.add(name)
                    for key in outer_keys:
                        if key == name or key.startswith(f"{name}."):
                            shadowed_values[key] = block_intervals.get(key)
            if self.visit(statement, block_intervals) is False:
                self.unreachable_tails.update(
                    id(item) for item in statements[index + 1 :]
                )
                continuing = False
                break
        for key in outer_keys:
            value_after_block = shadowed_values.get(
                key,
                block_intervals.get(key),
            )
            if value_after_block is not None:
                intervals[key] = value_after_block
            else:
                intervals.pop(key, None)
        for key, bounds in block_intervals.items():
            owner, separator, _ = key.partition(".[")
            if separator and owner not in block_names:
                intervals[key] = bounds
        return continuing

    def visit_if(
        self,
        value,
        intervals,
    ):
        """Refine intervals under an if condition and merge the branch results."""
        condition = getattr(value, "condition", getattr(value, "if_condition", None))
        self.visit(condition, intervals)
        static_known, static_body = self.generator.glsl_static_if_body(value)
        if static_known:
            return self.visit_sequence(static_body, intervals)
        branch_intervals = []
        then_intervals = self.refine_guard(
            condition,
            intervals,
            True,
        )
        if then_intervals is not None:
            if self.visit_sequence(value.if_body, then_intervals) is not False:
                branch_intervals.append(then_intervals)
        else:
            self.refuted_branch(value.if_body)
        fallthrough_intervals = self.refine_guard(
            condition,
            intervals,
            False,
        )
        for else_if_condition, else_if_body in zip(
            getattr(value, "else_if_conditions", []) or [],
            getattr(value, "else_if_bodies", []) or [],
        ):
            if fallthrough_intervals is None:
                self.refuted_branch(else_if_body)
                continue
            self.visit(else_if_condition, fallthrough_intervals)
            candidate = self.refine_guard(
                else_if_condition,
                fallthrough_intervals,
                True,
            )
            if candidate is not None:
                if self.visit_sequence(else_if_body, candidate) is not False:
                    branch_intervals.append(candidate)
            else:
                self.refuted_branch(else_if_body)
            fallthrough_intervals = self.refine_guard(
                else_if_condition,
                fallthrough_intervals,
                False,
            )
        if fallthrough_intervals is not None:
            else_intervals = dict(fallthrough_intervals)
            if (
                self.visit_sequence(getattr(value, "else_body", None), else_intervals)
                is not False
            ):
                branch_intervals.append(else_intervals)
        else:
            self.refuted_branch(getattr(value, "else_body", None))
        if not branch_intervals:
            return False
        merged = branch_intervals[0]
        for candidate in branch_intervals[1:]:
            merged = self.merge_intervals(merged, candidate)
        intervals.clear()
        intervals.update(merged)
        return

    def visit_switch(
        self,
        value,
        intervals,
    ):
        """Visit every case of a switch or match and merge the results."""
        self.visit(getattr(value, "expression", None), intervals)
        branches = list(getattr(value, "cases", getattr(value, "arms", [])) or [])
        default_case = getattr(value, "default_case", None)
        mutated = self.mutation_interval_keys([branches, default_case])
        branch_base = dict(intervals)
        self.clear_interval_targets(branch_base, mutated)
        for branch in branches:
            branch_intervals = dict(branch_base)
            self.visit(branch, branch_intervals)
        if default_case is not None:
            default_intervals = dict(branch_base)
            self.visit(default_case, default_intervals)
        self.clear_interval_targets(intervals, mutated)
        return

    def visit_for(
        self,
        value,
        intervals,
    ):
        """Bound a counted for loop and propagate the intervals that hold after it."""
        counter_name = getattr(value.init, "name", None)
        counter_before = {
            key: bounds
            for key, bounds in intervals.items()
            if key == counter_name or key.startswith(f"{counter_name}.")
        }

        def restore_counter(frame):
            if not isinstance(value.init, VariableNode):
                return
            self.clear_interval_targets(frame, {counter_name})
            frame.update(counter_before)

        loop_intervals = dict(intervals)
        mutated = self.mutation_interval_keys(
            [
                getattr(value, "init", None),
                getattr(value, "condition", None),
                getattr(value, "body", None),
                getattr(value, "update", None),
            ]
        )
        self.clear_interval_targets(loop_intervals, mutated)
        self.visit(getattr(value, "init", None), loop_intervals)
        if isinstance(getattr(value, "init", None), list):
            # for (int a = 0, b = n; ...) declares no single induction counter for the bounds below, so a
            # declared variable the loop writes has no proven range inside the loop.
            self.clear_interval_targets(loop_intervals, mutated)
        bounded = self.bounded_loop_bounds(value, loop_intervals, mutated)
        count = bounded[0] if bounded is not None else None
        if count == 0:
            self.bounded_zero_trip.add(id(value))
            return
        self.possibly_entered_loops.add(id(value))
        handled = self.replay_exact_for_iterations(
            value, intervals, count, loop_intervals, restore_counter
        )
        if handled is not UNHANDLED:
            return handled
        entries, exits = (
            self.additive_loop_intervals(value, intervals, mutated, count)
            if count is not None
            else ({}, {})
        )
        entries.update(
            guarded_additive_loop_entries(
                self.generator,
                self.function,
                value,
                intervals,
                entries,
                count,
                constants=self.constants,
                escaped=self.escaped_names,
                mutated=mutated,
                lexical=self.lexical,
                checked_integer=self.checked_integer_interval,
                checked_components=self.checked_components,
                mutation_keys=self.mutation_interval_keys,
            )
        )
        loop_intervals.update(entries)
        loop_name, loop_interval = self.generator.glsl_private_pointer_loop_interval(
            value, loop_intervals, self.constants
        )
        if loop_interval is None and count is not None:
            loop_name = value.init.name
            # An upper trip bound establishes entry values, not an exact
            # iteration sequence for pointer rebasing or exit state.
            loop_interval = bounded[1]
        if loop_name in self.mutation_interval_keys(
            [
                getattr(value, "condition", None),
                getattr(value, "body", None),
            ],
            (
                id(value.init)
                if isinstance(value.init, VariableNode) and value.init.name == loop_name
                else None
            ),
        ):
            loop_interval = None
        if loop_name:
            if loop_interval is None:
                loop_intervals.pop(loop_name, None)
            else:
                loop_intervals[loop_name] = loop_interval
        self.visit(getattr(value, "condition", None), loop_intervals)
        guarded = self.refine_guard(value.condition, loop_intervals, True)
        if guarded is not None:
            self.visit_sequence(getattr(value, "body", None), guarded)
            self.visit(getattr(value, "update", None), guarded)
        self.clear_interval_targets(intervals, mutated)
        intervals.update(exits)
        restore_counter(intervals)
        return

    def visit_for_in(
        self,
        value,
        intervals,
    ):
        """Visit a for-in loop with its pattern interval and merge the loop results."""
        loop_intervals = dict(intervals)
        mutated = self.mutation_interval_keys(
            [
                getattr(value, "iterable", None),
                getattr(value, "body", None),
            ]
        )
        self.clear_interval_targets(loop_intervals, mutated)
        getattr(value, "iterable", None)
        pattern_interval = self.generator.glsl_for_in_pattern_interval(
            value,
            loop_intervals,
            self.constants,
        )

        pattern = getattr(value, "pattern", None)
        if pattern:
            loop_intervals.pop(pattern, None)
            self.generator.glsl_clear_index_component_intervals(
                loop_intervals,
                pattern,
            )
            if pattern_interval is not None:
                loop_intervals[pattern] = pattern_interval
        self.visit_sequence(getattr(value, "body", None), loop_intervals)
        self.clear_interval_targets(intervals, mutated)
        return

    def visit_while(
        self,
        value,
        intervals,
    ):
        """Visit a while, do-while or loop statement and clear what its body mutates."""
        loop_intervals = dict(intervals)
        mutated = self.mutation_interval_keys(
            [
                getattr(value, "condition", None),
                getattr(value, "body", None),
            ]
        )
        self.clear_interval_targets(loop_intervals, mutated)
        if isinstance(value, DoWhileNode):
            self.visit_sequence(getattr(value, "body", None), loop_intervals)
            self.visit(getattr(value, "condition", None), loop_intervals)
        elif isinstance(value, WhileNode):
            self.visit(getattr(value, "condition", None), loop_intervals)
            guarded = self.refine_guard(value.condition, loop_intervals, True)
            if guarded is not None:
                self.visit_sequence(getattr(value, "body", None), guarded)
        else:
            self.visit_sequence(getattr(value, "body", None), loop_intervals)
        self.clear_interval_targets(intervals, mutated)
        return
