"""AST walk that tracks visible constants and index hints for resource-array size inference."""

from ..ast import (
    AssignmentNode,
    BlockNode,
    CaseNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    IfNode,
    LoopNode,
    MatchArmNode,
    MatchNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)


class ResourceArrayConstantWalkMixin:
    """Methods of :class:`ResourceArraySizeInference`; see :mod:`.resource_array_size_inference`."""

    def walk_nodes_with_constants(self, root, constants, initial_index_hints=None):
        """Yield AST nodes with literal constants visible at that program point."""
        visited = set()

        def subtree_node_ids(value, seen=None):
            seen = seen or set()
            if value is None or isinstance(value, (str, int, float, bool)):
                return seen
            if isinstance(value, dict):
                for item in value.values():
                    subtree_node_ids(item, seen)
                return seen
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    subtree_node_ids(item, seen)
                return seen

            value_id = id(value)
            if value_id in seen:
                return seen
            seen.add(value_id)

            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    subtree_node_ids(child, seen)
            return seen

        def reset_visited_subtree(*values):
            for value in values:
                visited.difference_update(subtree_node_ids(value))

        def walk(value, active_constants, active_index_hints):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item, active_constants, active_index_hints)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item, active_constants, active_index_hints)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            if isinstance(value, BlockNode):
                yield from self.walk_block_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, VariableNode):
                yield from self.walk_variable_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, AssignmentNode):
                yield from self.walk_assignment_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, IfNode):
                yield from self.walk_if_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, ForNode):
                yield from self.walk_for_with_constants(
                    walk,
                    reset_visited_subtree,
                    value,
                    active_constants,
                    active_index_hints,
                )
                return

            if isinstance(value, ForInNode):
                yield from self.walk_for_in_with_constants(
                    walk,
                    reset_visited_subtree,
                    value,
                    active_constants,
                    active_index_hints,
                )
                return

            if isinstance(value, WhileNode):
                yield from self.walk_while_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, DoWhileNode):
                yield from self.walk_do_while_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, LoopNode):
                yield from self.walk_loop_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, SwitchNode):
                yield from self.walk_switch_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, CaseNode):
                yield from self.walk_case_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, MatchNode):
                yield from self.walk_match_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            if isinstance(value, MatchArmNode):
                yield from self.walk_match_arm_with_constants(
                    walk, value, active_constants, active_index_hints
                )
                return

            yield value, active_constants, active_index_hints
            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    yield from walk(child, active_constants, active_index_hints)

        yield from walk(root, constants, dict(initial_index_hints or {}))

    def walk_block_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``BlockNode`` for :meth:`walk_nodes_with_constants`."""
        block_constants = dict(active_constants)
        block_index_hints = dict(active_index_hints)
        outer_index_hint_names = set(active_index_hints)
        local_names = set()
        yield value, block_constants, block_index_hints
        for statement in getattr(value, "statements", []) or []:
            if isinstance(statement, VariableNode):
                name = getattr(statement, "name", None)
                if name:
                    local_names.add(name)
            yield from walk(statement, block_constants, block_index_hints)
        propagated_names = set(outer_index_hint_names).union(
            self.assigned_hint_base_names(getattr(value, "statements", []))
        )
        self.merge_existing_index_hints(
            active_index_hints,
            block_index_hints,
            self.outer_hint_names_without_locals(propagated_names, local_names),
        )

    def walk_variable_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``VariableNode`` for :meth:`walk_nodes_with_constants`."""
        name = getattr(value, "name", None)
        if name:
            active_constants.pop(name, None)
            self.remove_index_hints_for_name(active_index_hints, name)
        yield value, active_constants, active_index_hints
        for key, child in vars(value).items():
            if key in {"parent", "annotations", "name"}:
                continue
            yield from walk(child, active_constants, active_index_hints)
        initial_value = getattr(value, "initial_value", None)
        index_hint = self.resource_index_hint(
            initial_value, active_constants, active_index_hints
        )
        if name and index_hint is not None and index_hint >= 0:
            active_index_hints[name] = index_hint
        if name:
            self.record_vector_component_index_hints(
                name,
                initial_value,
                active_constants,
                active_index_hints,
            )
        if name and "const" in getattr(value, "qualifiers", []):
            const_value = self.literal_int_value(initial_value, active_constants)
            if const_value is not None:
                active_constants[name] = const_value

    def walk_assignment_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``AssignmentNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        target = getattr(value, "target", getattr(value, "left", None))
        assigned_value = getattr(value, "value", getattr(value, "right", None))
        yield from walk(target, active_constants, active_index_hints)
        yield from walk(assigned_value, active_constants, active_index_hints)

        target_component = self.expression_component_parts(target)
        if target_component is not None:
            target_name, component = target_component
            operator = getattr(value, "operator", None)
            current_hint = self.resource_index_hint(
                target, active_constants, active_index_hints
            )
            assignment_hint = None
            compound_hint = None
            if operator == "=":
                assignment_hint = self.resource_index_hint(
                    assigned_value, active_constants, active_index_hints
                )
            else:
                compound_hint = self.compound_assignment_index_hint(
                    operator,
                    current_hint,
                    assigned_value,
                    active_constants,
                    active_index_hints,
                )
            updated_alternatives = self.updated_component_assignment_alternatives(
                target_name,
                component,
                operator,
                assigned_value,
                active_constants,
                active_index_hints,
                None,
            )
            self.remove_component_index_hints(
                active_index_hints, target_name, component
            )
            if operator == "=":
                self.record_component_index_hint(
                    target_name,
                    component,
                    assignment_hint,
                    active_index_hints,
                )
            else:
                self.record_component_index_hint(
                    target_name, component, compound_hint, active_index_hints
                )
            if updated_alternatives:
                active_index_hints[self.vector_alternatives_key(target_name)] = (
                    updated_alternatives
                )
            return

        target_name = self.expression_name(target)
        if target_name:
            operator = getattr(value, "operator", None)
            current_hint = self.resource_index_hint(
                target, active_constants, active_index_hints
            )
            source_index_hints = dict(active_index_hints)
            assignment_hint = None
            compound_hint = None
            if operator == "=":
                assignment_hint = self.resource_index_hint(
                    assigned_value, active_constants, active_index_hints
                )
            else:
                compound_hint = self.compound_assignment_index_hint(
                    operator,
                    current_hint,
                    assigned_value,
                    active_constants,
                    active_index_hints,
                )
            active_constants.pop(target_name, None)
            self.remove_index_hints_for_name(active_index_hints, target_name)
            if operator == "=":
                if assignment_hint is not None and assignment_hint >= 0:
                    active_index_hints[target_name] = assignment_hint
                self.record_vector_component_index_hints(
                    target_name,
                    assigned_value,
                    active_constants,
                    active_index_hints,
                    source_index_hints=source_index_hints,
                )
            else:
                if compound_hint is not None and compound_hint >= 0:
                    active_index_hints[target_name] = compound_hint

    def walk_if_with_constants(self, walk, value, active_constants, active_index_hints):
        """Walk a ``IfNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        yield from walk(
            getattr(value, "condition", None),
            active_constants,
            active_index_hints,
        )
        then_index_hints = dict(active_index_hints)
        yield from walk(
            getattr(value, "then_branch", None),
            dict(active_constants),
            then_index_hints,
        )
        else_branch = getattr(value, "else_branch", None)
        else_index_hints = dict(active_index_hints)
        if else_branch is not None:
            yield from walk(
                else_branch,
                dict(active_constants),
                else_index_hints,
            )
        self.merge_branch_index_hints(
            active_index_hints, then_index_hints, else_index_hints
        )

    def walk_for_with_constants(
        self, walk, reset_visited_subtree, value, active_constants, active_index_hints
    ):
        """Walk a ``ForNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        outer_index_hint_names = set(active_index_hints)
        loop_constants = dict(active_constants)
        loop_index_hints = dict(active_index_hints)
        yield from walk(getattr(value, "init", None), loop_constants, loop_index_hints)
        yield from walk(
            getattr(value, "condition", None),
            loop_constants,
            loop_index_hints,
        )
        body = getattr(value, "body", None)
        update = getattr(value, "update", None)
        iteration_count = None
        if not self.subtree_contains_control_exit(body):
            iteration_count = self.simple_for_iteration_count(
                value, loop_constants, loop_index_hints
            )
        if iteration_count is not None:
            final_index_hints = dict(loop_index_hints)
            for _ in range(iteration_count):
                reset_visited_subtree(body, update)
                yield from walk(
                    body,
                    dict(loop_constants),
                    final_index_hints,
                )
                yield from walk(update, loop_constants, final_index_hints)
            propagated_names = set(outer_index_hint_names)
            if iteration_count > 0:
                propagated_names.update(self.assigned_hint_base_names(body, update))
                propagated_names.difference_update(
                    self.declared_variable_names(getattr(value, "init", None))
                )
            self.merge_existing_index_hints(
                active_index_hints,
                final_index_hints,
                propagated_names,
            )
            return
        self.approximate_loops.add(id(value))
        zero_iteration_hints = dict(loop_index_hints)
        body_index_hints = dict(loop_index_hints)
        yield from walk(
            body,
            dict(loop_constants),
            body_index_hints,
        )
        yield from walk(update, loop_constants, body_index_hints)
        self.merge_loop_index_hints(
            active_index_hints,
            zero_iteration_hints,
            body_index_hints,
            compound_targets=(
                self.compound_assignment_targets(body)
                + self.compound_assignment_targets(update)
            ),
        )

    def walk_for_in_with_constants(
        self, walk, reset_visited_subtree, value, active_constants, active_index_hints
    ):
        """Walk a ``ForInNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        outer_index_hint_names = set(active_index_hints)
        loop_constants = dict(active_constants)
        loop_index_hints = dict(active_index_hints)
        pattern = getattr(value, "pattern", None)
        if pattern:
            loop_constants.pop(pattern, None)
            loop_index_hints.pop(pattern, None)
        yield from walk(
            getattr(value, "iterable", None),
            active_constants,
            active_index_hints,
        )
        body = getattr(value, "body", None)
        loop_range = None
        if not self.subtree_contains_control_exit(body):
            loop_range = self.simple_for_in_range(value, loop_constants)
        if loop_range is not None:
            start, iteration_count = loop_range
            final_index_hints = dict(loop_index_hints)
            for iteration in range(iteration_count):
                reset_visited_subtree(body)
                if pattern:
                    final_index_hints[pattern] = start + iteration
                yield from walk(
                    body,
                    loop_constants,
                    final_index_hints,
                )
            if pattern:
                final_index_hints.pop(pattern, None)
            propagated_names = set(outer_index_hint_names)
            if iteration_count > 0:
                propagated_names.update(self.assigned_hint_base_names(body))
                if pattern:
                    propagated_names.discard(pattern)
            self.merge_existing_index_hints(
                active_index_hints,
                final_index_hints,
                propagated_names,
            )
            return
        self.approximate_loops.add(id(value))
        body_index_hints = dict(loop_index_hints)
        yield from walk(body, loop_constants, body_index_hints)
        self.merge_loop_index_hints(
            active_index_hints,
            loop_index_hints,
            body_index_hints,
            compound_targets=self.compound_assignment_targets(body),
        )

    def walk_while_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``WhileNode`` for :meth:`walk_nodes_with_constants`."""
        self.approximate_loops.add(id(value))
        yield value, active_constants, active_index_hints
        condition_index_hints = dict(active_index_hints)
        yield from walk(
            getattr(value, "condition", None),
            active_constants,
            condition_index_hints,
        )
        body_index_hints = dict(condition_index_hints)
        yield from walk(
            getattr(value, "body", None),
            dict(active_constants),
            body_index_hints,
        )
        body = getattr(value, "body", None)
        self.merge_loop_index_hints(
            active_index_hints,
            condition_index_hints,
            body_index_hints,
            compound_targets=self.compound_assignment_targets(body),
        )

    def walk_do_while_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``DoWhileNode`` for :meth:`walk_nodes_with_constants`."""
        self.approximate_loops.add(id(value))
        yield value, active_constants, active_index_hints
        body = getattr(value, "body", None)
        body_index_hints = dict(active_index_hints)
        yield from walk(
            body,
            dict(active_constants),
            body_index_hints,
        )
        condition_index_hints = dict(body_index_hints)
        yield from walk(
            getattr(value, "condition", None),
            active_constants,
            condition_index_hints,
        )
        self.merge_loop_index_hints(
            active_index_hints,
            body_index_hints,
            condition_index_hints,
            compound_targets=self.compound_assignment_targets(body),
        )

    def walk_loop_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``LoopNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        body = getattr(value, "body", None)
        body_index_hints = dict(active_index_hints)
        yield from walk(
            body,
            dict(active_constants),
            body_index_hints,
        )
        self.merge_loop_index_hints(
            active_index_hints,
            body_index_hints,
            compound_targets=self.compound_assignment_targets(body),
        )

    def walk_switch_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``SwitchNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        yield from walk(
            getattr(value, "expression", None),
            active_constants,
            active_index_hints,
        )
        branch_index_hints = []
        has_default = False
        for case in getattr(value, "cases", []) or []:
            case_index_hints = dict(active_index_hints)
            yield from walk(case, dict(active_constants), case_index_hints)
            branch_index_hints.append(case_index_hints)
            has_default = has_default or self.is_switch_default_case(case)
        default_case = getattr(value, "default_case", None)
        if default_case is not None:
            default_index_hints = dict(active_index_hints)
            yield from walk(
                default_case,
                dict(active_constants),
                default_index_hints,
            )
            branch_index_hints.append(default_index_hints)
            has_default = True
        if not has_default:
            branch_index_hints.append(dict(active_index_hints))
        self.merge_branch_index_hints(active_index_hints, *branch_index_hints)

    def walk_case_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``CaseNode`` for :meth:`walk_nodes_with_constants`."""
        case_constants = dict(active_constants)
        yield value, case_constants, active_index_hints
        yield from walk(
            getattr(value, "value", None),
            active_constants,
            active_index_hints,
        )
        for statement in getattr(value, "statements", []) or []:
            yield from walk(statement, case_constants, active_index_hints)

    def walk_match_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``MatchNode`` for :meth:`walk_nodes_with_constants`."""
        yield value, active_constants, active_index_hints
        yield from walk(
            getattr(value, "expression", None),
            active_constants,
            active_index_hints,
        )
        branch_index_hints = []
        has_default = False
        for arm in getattr(value, "arms", []) or []:
            arm_index_hints = dict(active_index_hints)
            yield from walk(arm, dict(active_constants), arm_index_hints)
            branch_index_hints.append(arm_index_hints)
            has_default = has_default or self.is_match_default_arm(arm)
        if not has_default:
            branch_index_hints.append(dict(active_index_hints))
        self.merge_branch_index_hints(active_index_hints, *branch_index_hints)

    def walk_match_arm_with_constants(
        self, walk, value, active_constants, active_index_hints
    ):
        """Walk a ``MatchArmNode`` for :meth:`walk_nodes_with_constants`."""
        arm_constants = dict(active_constants)
        yield value, arm_constants, active_index_hints
        yield from walk(
            getattr(value, "pattern", None),
            active_constants,
            active_index_hints,
        )
        yield from walk(
            getattr(value, "guard", None),
            active_constants,
            active_index_hints,
        )
        yield from walk(getattr(value, "body", None), arm_constants, active_index_hints)
