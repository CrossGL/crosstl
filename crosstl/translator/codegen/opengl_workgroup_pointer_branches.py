"""Interval and alias transfer of if, switch and match statements, for ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

from types import (
    SimpleNamespace,
)


class OpenGLWorkgroupPointerBranchMixin:
    """Branch members of ``GLSLWorkgroupPointerAccessAnalysis.visit``."""

    def visit_if(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit both branches of an if statement and merge their states."""
        condition = getattr(
            value,
            "condition",
            getattr(value, "if_condition", None),
        )
        self.visit(
            condition,
            active_aliases,
            active_intervals,
        )
        static_known, static_body = self.generator.glsl_static_if_body(value)
        if static_known:
            self.visit_scoped_sequence(
                static_body,
                active_aliases,
                active_intervals,
            )
            return
        base_aliases = dict(active_aliases)
        branch_states = []
        then_aliases = dict(base_aliases)
        then_intervals = (
            self.generator.glsl_private_pointer_refined_condition_intervals(
                condition,
                active_intervals,
                self.constants,
                True,
                excluded_names=self.mutated_interval_names,
            )
        )
        if then_intervals is not None:
            self.visit_scoped_sequence(
                value.if_body,
                then_aliases,
                then_intervals,
            )
            branch_states.append((then_aliases, then_intervals))
        fallthrough_aliases = dict(base_aliases)
        fallthrough_intervals = (
            self.generator.glsl_private_pointer_refined_condition_intervals(
                condition,
                active_intervals,
                self.constants,
                False,
                excluded_names=self.mutated_interval_names,
            )
        )
        for else_if_condition, else_if_body in zip(
            getattr(value, "else_if_conditions", []) or [],
            getattr(value, "else_if_bodies", []) or [],
        ):
            if fallthrough_intervals is None:
                break
            self.visit(
                else_if_condition,
                fallthrough_aliases,
                fallthrough_intervals,
            )
            candidate_aliases = dict(fallthrough_aliases)
            candidate = self.generator.glsl_private_pointer_refined_condition_intervals(
                else_if_condition,
                fallthrough_intervals,
                self.constants,
                True,
                excluded_names=self.mutated_interval_names,
            )
            if candidate is not None:
                self.visit_scoped_sequence(
                    else_if_body,
                    candidate_aliases,
                    candidate,
                )
                branch_states.append((candidate_aliases, candidate))
            fallthrough_intervals = (
                self.generator.glsl_private_pointer_refined_condition_intervals(
                    else_if_condition,
                    fallthrough_intervals,
                    self.constants,
                    False,
                    excluded_names=self.mutated_interval_names,
                )
            )
        if fallthrough_intervals is not None:
            else_aliases = dict(fallthrough_aliases)
            else_intervals = dict(fallthrough_intervals)
            self.visit_scoped_sequence(
                getattr(value, "else_body", None),
                else_aliases,
                else_intervals,
            )
            branch_states.append((else_aliases, else_intervals))
        if branch_states:
            merged_intervals = branch_states[0][1]
            for _, candidate in branch_states[1:]:
                merged_intervals = self.merge_intervals(
                    merged_intervals,
                    candidate,
                )
            active_intervals.clear()
            active_intervals.update(merged_intervals)
            active_aliases.clear()
            active_aliases.update(base_aliases)
            self.merge_alias_states(active_aliases, branch_states)
        return

    def visit_switch(
        self,
        value,
        active_aliases,
        active_intervals,
    ):
        """Visit every case of a switch or match statement and merge their states."""
        self.visit(getattr(value, "expression", None), active_aliases, active_intervals)
        base_aliases = dict(active_aliases)
        base_intervals = dict(active_intervals)
        branches = list(getattr(value, "cases", getattr(value, "arms", [])) or [])
        default_case = getattr(value, "default_case", None)
        branch_states = []
        for branch in branches:
            branch_aliases = dict(base_aliases)
            branch_intervals = dict(base_intervals)
            self.visit(
                getattr(branch, "value", getattr(branch, "guard", None)),
                branch_aliases,
                branch_intervals,
            )
            self.visit_scoped_sequence(
                getattr(branch, "statements", getattr(branch, "body", None)),
                branch_aliases,
                branch_intervals,
            )
            branch_states.append((branch_aliases, branch_intervals))
        if default_case is not None:
            default_aliases = dict(base_aliases)
            default_intervals = dict(base_intervals)
            self.visit_scoped_sequence(
                getattr(default_case, "statements", default_case),
                default_aliases,
                default_intervals,
            )
            branch_states.append((default_aliases, default_intervals))
        else:
            branch_states.append((dict(base_aliases), dict(base_intervals)))
        changed_aliases = set()
        for branch_aliases, branch_intervals in branch_states:
            changed_aliases.update(
                self.changed_alias_names(
                    base_aliases,
                    base_intervals,
                    branch_aliases,
                    branch_intervals,
                )
            )
        switch_mutations = self.generator.glsl_function_mutated_interval_names(
            SimpleNamespace(body=[branches, default_case])
        )
        if changed_aliases or switch_mutations:
            conservative_aliases = dict(base_aliases)
            for name in changed_aliases:
                conservative_aliases[name] = self.poison_alias(
                    conservative_aliases[name]
                )
            conservative_intervals = dict(base_intervals)
            self.clear_interval_names(conservative_intervals, switch_mutations)
            for branch in branches:
                self.visit_scoped_sequence(
                    getattr(
                        branch,
                        "statements",
                        getattr(branch, "body", None),
                    ),
                    dict(conservative_aliases),
                    dict(conservative_intervals),
                )
            if default_case is not None:
                self.visit_scoped_sequence(
                    getattr(default_case, "statements", default_case),
                    dict(conservative_aliases),
                    dict(conservative_intervals),
                )
        merged_intervals = branch_states[0][1]
        for _, candidate in branch_states[1:]:
            merged_intervals = self.merge_intervals(
                merged_intervals,
                candidate,
            )
        self.clear_interval_names(merged_intervals, switch_mutations)
        active_intervals.clear()
        active_intervals.update(merged_intervals)
        active_aliases.clear()
        active_aliases.update(base_aliases)
        self.merge_alias_states(active_aliases, branch_states)
        for name in changed_aliases:
            active_aliases[name] = self.poison_alias(active_aliases[name])
        return
