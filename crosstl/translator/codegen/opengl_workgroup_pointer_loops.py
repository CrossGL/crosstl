"""Interval and alias transfer of for, for-in, while, do-while and loop statements, for ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

from types import (
    SimpleNamespace,
)

from ..ast import (
    ArrayNode,
    DoWhileNode,
    VariableNode,
    WhileNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
)
from .opengl_workgroup_pointer_loop_replay import OpenGLWorkgroupPointerLoopReplayMixin


class OpenGLWorkgroupPointerLoopMixin(OpenGLWorkgroupPointerLoopReplayMixin):
    """Loop members of ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

    def visit_for(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a for loop, replaying exact iterations when the counter values are known."""
        original_aliases = dict(active_aliases)
        original_intervals = dict(active_intervals)
        loop_aliases = dict(active_aliases)
        loop_intervals = dict(active_intervals)
        init = getattr(value, "init", None)
        self.visit(init, loop_aliases, loop_intervals)
        entry_aliases = dict(loop_aliases)
        entry_intervals = dict(loop_intervals)
        loop_name, loop_interval = self.generator.glsl_private_pointer_loop_interval(
            value,
            loop_intervals,
            self.constants,
        )
        loop_mutations = self.generator.glsl_function_mutated_interval_names(
            SimpleNamespace(
                body=[
                    getattr(value, "condition", None),
                    getattr(value, "body", None),
                    getattr(value, "update", None),
                ]
            )
        )
        self.clear_interval_names(loop_intervals, loop_mutations)
        body_mutations = self.generator.glsl_function_mutated_interval_names(
            SimpleNamespace(
                body=[
                    getattr(value, "condition", None),
                    getattr(value, "body", None),
                ]
            )
        )
        if loop_name in body_mutations:
            loop_interval = None
        if loop_name:
            if loop_interval is None:
                loop_intervals.pop(loop_name, None)
            else:
                loop_intervals[loop_name] = loop_interval
        self.visit(
            getattr(value, "condition", None),
            loop_aliases,
            loop_intervals,
        )
        self.visit_scoped_sequence(
            getattr(value, "body", None),
            loop_aliases,
            loop_intervals,
        )
        self.visit(
            getattr(value, "update", None),
            loop_aliases,
            loop_intervals,
        )
        changed = self.changed_alias_names(
            entry_aliases,
            entry_intervals,
            loop_aliases,
            loop_intervals,
        )
        exact_values = (
            self.exact_classic_for_iteration_values(value) if changed else None
        )
        if exact_values is not None:
            post_aliases, post_intervals = self.replay_exact_for_iterations(
                value, entry_aliases, entry_intervals, exact_values, loop_name
            )
        else:
            if changed:
                repeat_aliases = dict(loop_aliases)
                for name in changed:
                    repeat_aliases[name] = self.poison_alias(
                        repeat_aliases.get(name, entry_aliases[name])
                    )
                repeat_intervals = dict(loop_intervals)
                self.visit(
                    getattr(value, "condition", None),
                    repeat_aliases,
                    repeat_intervals,
                )
                self.visit_scoped_sequence(
                    getattr(value, "body", None),
                    repeat_aliases,
                    repeat_intervals,
                )
                self.visit(
                    getattr(value, "update", None),
                    repeat_aliases,
                    repeat_intervals,
                )
                loop_aliases = repeat_aliases
                loop_intervals = repeat_intervals
            post_aliases = dict(entry_aliases)
            self.merge_alias_states(
                post_aliases,
                [
                    (entry_aliases, entry_intervals),
                    (loop_aliases, loop_intervals),
                ],
            )
            post_intervals = self.merge_intervals(
                entry_intervals,
                loop_intervals,
            )
            self.clear_interval_names(post_intervals, loop_mutations)
        init_name = (
            getattr(init, "name", None)
            if isinstance(init, (VariableNode, ArrayNode))
            else None
        )
        self.propagate_outer_state(
            active_aliases,
            active_intervals,
            original_aliases,
            original_intervals,
            post_aliases,
            post_intervals,
            shadow_names={init_name} if init_name else (),
        )
        return

    def visit_for_in(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a for-in loop, replaying exact iterations when the iterated values are known."""
        self.visit(
            getattr(value, "iterable", None),
            active_aliases,
            active_intervals,
        )
        original_aliases = dict(active_aliases)
        original_intervals = dict(active_intervals)
        loop_aliases = dict(active_aliases)
        loop_intervals = dict(active_intervals)
        pattern_interval = self.generator.glsl_for_in_pattern_interval(
            value,
            loop_intervals,
            self.constants,
        )
        pattern = getattr(value, "pattern", None)
        if pattern:
            loop_aliases.pop(pattern, None)
            loop_intervals.pop(pattern, None)
            self.generator.glsl_clear_index_component_intervals(
                loop_intervals,
                pattern,
            )
            if pattern_interval is not None:
                loop_intervals[pattern] = pattern_interval
        body_mutations = self.generator.glsl_function_mutated_interval_names(
            SimpleNamespace(body=getattr(value, "body", None))
        )
        body_mutations.discard(pattern)
        self.clear_interval_names(loop_intervals, body_mutations)
        exact_values = self.exact_for_in_iteration_values(value)
        if exact_values == ():
            post_aliases = dict(original_aliases)
            post_intervals = dict(original_intervals)
        else:
            self.visit_scoped_sequence(
                getattr(value, "body", None),
                loop_aliases,
                loop_intervals,
            )
            changed = self.changed_alias_names(
                {
                    name: binding
                    for name, binding in original_aliases.items()
                    if name != pattern
                },
                original_intervals,
                loop_aliases,
                loop_intervals,
            )
            if changed and exact_values is not None:
                loop_aliases = dict(original_aliases)
                if pattern:
                    loop_aliases.pop(pattern, None)
                loop_intervals = dict(original_intervals)
                if pattern:
                    loop_intervals.pop(pattern, None)
                    self.generator.glsl_clear_index_component_intervals(
                        loop_intervals,
                        pattern,
                    )
                for iteration_value in exact_values:
                    if pattern:
                        loop_intervals[pattern] = (
                            iteration_value,
                            iteration_value,
                        )
                    self.visit_scoped_sequence(
                        getattr(value, "body", None),
                        loop_aliases,
                        loop_intervals,
                    )
                post_aliases = loop_aliases
                post_intervals = loop_intervals
            else:
                if changed:
                    repeat_aliases = dict(loop_aliases)
                    for name in changed:
                        repeat_aliases[name] = self.poison_alias(
                            repeat_aliases.get(
                                name,
                                original_aliases[name],
                            )
                        )
                    repeat_intervals = dict(loop_intervals)
                    self.visit_scoped_sequence(
                        getattr(value, "body", None),
                        repeat_aliases,
                        repeat_intervals,
                    )
                    loop_aliases = repeat_aliases
                    loop_intervals = repeat_intervals
                post_aliases = dict(original_aliases)
                self.merge_alias_states(
                    post_aliases,
                    [
                        (original_aliases, original_intervals),
                        (loop_aliases, loop_intervals),
                    ],
                )
                post_intervals = self.merge_intervals(
                    original_intervals,
                    loop_intervals,
                )
                self.clear_interval_names(post_intervals, body_mutations)
        self.propagate_outer_state(
            active_aliases,
            active_intervals,
            original_aliases,
            original_intervals,
            post_aliases,
            post_intervals,
            shadow_names={pattern} if pattern else (),
        )
        return

    def visit_while(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit a while, do-while or loop statement until its state stops changing."""
        original_aliases = dict(active_aliases)
        original_intervals = dict(active_intervals)
        condition = getattr(value, "condition", None)
        condition_literal = (
            evaluate_literal_int_expression(condition, self.constants)
            if condition is not None
            else None
        )
        if isinstance(value, WhileNode) and condition_literal == 0:
            self.visit(condition, active_aliases, active_intervals)
            return
        loop_aliases = dict(active_aliases)
        loop_intervals = dict(active_intervals)
        loop_mutations = self.generator.glsl_function_mutated_interval_names(
            SimpleNamespace(body=[condition, getattr(value, "body", None)])
        )
        self.clear_interval_names(loop_intervals, loop_mutations)
        if isinstance(value, DoWhileNode):
            self.visit_scoped_sequence(
                getattr(value, "body", None),
                loop_aliases,
                loop_intervals,
            )
            self.visit(condition, loop_aliases, loop_intervals)
        elif isinstance(value, WhileNode):
            self.visit(condition, loop_aliases, loop_intervals)
            self.visit_scoped_sequence(
                getattr(value, "body", None),
                loop_aliases,
                loop_intervals,
            )
        else:
            self.visit_scoped_sequence(
                getattr(value, "body", None),
                loop_aliases,
                loop_intervals,
            )
        changed = self.changed_alias_names(
            original_aliases,
            original_intervals,
            loop_aliases,
            loop_intervals,
        )
        repeats = not (isinstance(value, DoWhileNode) and condition_literal == 0)
        if changed and repeats:
            repeat_aliases = dict(loop_aliases)
            for name in changed:
                repeat_aliases[name] = self.poison_alias(
                    repeat_aliases.get(name, original_aliases[name])
                )
            repeat_intervals = dict(loop_intervals)
            if isinstance(value, DoWhileNode):
                self.visit_scoped_sequence(
                    getattr(value, "body", None),
                    repeat_aliases,
                    repeat_intervals,
                )
                self.visit(condition, repeat_aliases, repeat_intervals)
            elif isinstance(value, WhileNode):
                self.visit(condition, repeat_aliases, repeat_intervals)
                self.visit_scoped_sequence(
                    getattr(value, "body", None),
                    repeat_aliases,
                    repeat_intervals,
                )
            else:
                self.visit_scoped_sequence(
                    getattr(value, "body", None),
                    repeat_aliases,
                    repeat_intervals,
                )
            loop_aliases = repeat_aliases
            loop_intervals = repeat_intervals
        if isinstance(value, DoWhileNode) and condition_literal == 0:
            post_aliases = loop_aliases
            post_intervals = loop_intervals
        else:
            post_aliases = dict(original_aliases)
            self.merge_alias_states(
                post_aliases,
                [
                    (original_aliases, original_intervals),
                    (loop_aliases, loop_intervals),
                ],
            )
            post_intervals = self.merge_intervals(
                original_intervals,
                loop_intervals,
            )
            self.clear_interval_names(post_intervals, loop_mutations)
        self.propagate_outer_state(
            active_aliases,
            active_intervals,
            original_aliases,
            original_intervals,
            post_aliases,
            post_intervals,
        )
        return
