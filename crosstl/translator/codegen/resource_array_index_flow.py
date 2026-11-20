"""Loop, assignment and call-site index hints for resource-array size inference."""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    ReturnNode,
    VariableNode,
)
from .resource_array_index_alternatives import (
    MAX_STATIC_LOOP_ITERATIONS_FOR_HINTS,
)


class ResourceArrayIndexFlowMixin:
    """Methods of :class:`ResourceArraySizeInference`; see :mod:`.resource_array_size_inference`."""

    def assignment_target_name(self, node):
        target = getattr(node, "target", getattr(node, "left", None))
        return self.expression_name(target)

    def simple_assignment_step(self, node, name, constants, index_hints):
        if (
            not isinstance(node, AssignmentNode)
            or self.assignment_target_name(node) != name
        ):
            return None

        operator = getattr(node, "operator", None)
        assigned_value = getattr(node, "value", getattr(node, "right", None))
        if operator in {"+=", "-="}:
            step = self.literal_int_value(assigned_value, constants)
            if step is None:
                return None
            return step if operator == "+=" else -step

        if operator != "=" or not isinstance(assigned_value, BinaryOpNode):
            return None

        binary_operator = getattr(
            assigned_value, "operator", getattr(assigned_value, "op", None)
        )
        left_expr = getattr(assigned_value, "left", None)
        right_expr = getattr(assigned_value, "right", None)
        left_name = self.expression_name(left_expr)
        right_name = self.expression_name(right_expr)

        if left_name == name:
            step = self.literal_int_value(right_expr, constants)
            if step is None:
                return None
            if binary_operator == "+":
                return step
            if binary_operator == "-":
                return -step
        if right_name == name and binary_operator == "+":
            return self.literal_int_value(left_expr, constants)
        return None

    def simple_for_iteration_count(self, node, constants, index_hints):
        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None) or self.assignment_target_name(init)
        if not loop_name:
            return None

        start = self.known_index_value(init, constants, index_hints)
        if start is None:
            start = self.known_index_value(
                getattr(init, "initial_value", None), constants, index_hints
            )
        if start is None:
            start = self.known_index_value(
                getattr(init, "value", getattr(init, "right", None)),
                constants,
                index_hints,
            )
        if start is None:
            return None

        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return None

        operator = getattr(condition, "operator", getattr(condition, "op", None))
        left_expr = getattr(condition, "left", None)
        right_expr = getattr(condition, "right", None)
        left_name = self.expression_name(left_expr)
        right_name = self.expression_name(right_expr)
        if left_name == loop_name:
            bound = self.literal_int_value(right_expr, constants)
        elif right_name == loop_name:
            bound = self.literal_int_value(left_expr, constants)
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator)
        else:
            return None
        if bound is None:
            return None

        step = self.simple_assignment_step(
            getattr(node, "update", None), loop_name, constants, index_hints
        )
        if step is None or step == 0:
            return None

        count = None
        if step > 0:
            if operator == "<":
                count = 0 if start >= bound else (bound - start + step - 1) // step
            elif operator == "<=":
                count = 0 if start > bound else ((bound - start) // step) + 1
        else:
            step_abs = -step
            if operator == ">":
                count = (
                    0 if start <= bound else (start - bound + step_abs - 1) // step_abs
                )
            elif operator == ">=":
                count = 0 if start < bound else ((start - bound) // step_abs) + 1

        if count is None or count > MAX_STATIC_LOOP_ITERATIONS_FOR_HINTS:
            return None
        return count

    def simple_for_in_range(self, node, constants):
        iterable = getattr(node, "iterable", None)
        if iterable is None or "Range" not in iterable.__class__.__name__:
            return None

        start = self.literal_int_value(getattr(iterable, "start", None), constants)
        end = self.literal_int_value(getattr(iterable, "end", None), constants)
        if start is None or end is None:
            return None

        count = end - start
        if getattr(iterable, "inclusive", False):
            count += 1
        if count < 0 or count > MAX_STATIC_LOOP_ITERATIONS_FOR_HINTS:
            return None
        return start, count

    def subtree_contains_control_exit(self, value, seen=None):
        seen = seen or set()
        if value is None or isinstance(value, (str, int, float, bool)):
            return False
        if isinstance(value, dict):
            return any(
                self.subtree_contains_control_exit(item, seen)
                for item in value.values()
            )
        if isinstance(value, (list, tuple, set)):
            return any(self.subtree_contains_control_exit(item, seen) for item in value)

        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)

        class_name = value.__class__.__name__
        if class_name in {"BreakNode", "ContinueNode", "ReturnNode"}:
            return True
        if not hasattr(value, "__dict__"):
            return False
        return any(
            self.subtree_contains_control_exit(child, seen)
            for key, child in vars(value).items()
            if key not in {"parent", "annotations"}
        )

    def compound_assignment_targets(self, value):
        targets = []
        seen = set()

        def visit(item):
            if item is None or isinstance(item, (str, int, float, bool)):
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
                return
            if isinstance(item, (list, tuple, set)):
                for child in item:
                    visit(child)
                return

            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)

            if (
                isinstance(item, AssignmentNode)
                and getattr(item, "operator", None) != "="
            ):
                target = getattr(item, "target", getattr(item, "left", None))
                target_component = self.expression_component_parts(target)
                if target_component is not None:
                    targets.append(target_component)
                else:
                    target_name = self.expression_name(target)
                    if target_name:
                        targets.append((target_name, None))

            if hasattr(item, "__dict__"):
                for key, child in vars(item).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        visit(value)
        return targets

    def assigned_hint_base_names(self, *values):
        names = set()
        seen = set()

        def visit(item):
            if item is None or isinstance(item, (str, int, float, bool)):
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
                return
            if isinstance(item, (list, tuple, set)):
                for child in item:
                    visit(child)
                return

            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)

            if isinstance(item, AssignmentNode):
                target = getattr(item, "target", getattr(item, "left", None))
                target_component = self.expression_component_parts(target)
                if target_component is not None:
                    names.add(target_component[0])
                else:
                    target_name = self.expression_name(target)
                    if target_name:
                        names.add(target_name)

            if hasattr(item, "__dict__"):
                for key, child in vars(item).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        for value in values:
            visit(value)
        return names

    def declared_variable_names(self, *values):
        names = set()
        seen = set()

        def visit(item):
            if item is None or isinstance(item, (str, int, float, bool)):
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
                return
            if isinstance(item, (list, tuple, set)):
                for child in item:
                    visit(child)
                return

            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)

            if isinstance(item, VariableNode):
                name = getattr(item, "name", None)
                if name:
                    names.add(name)

            if hasattr(item, "__dict__"):
                for key, child in vars(item).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        for value in values:
            visit(value)
        return names

    def remove_compound_loop_targets(self, index_hints, targets):
        for name, component in targets:
            if component is None:
                self.remove_index_hints_for_name(index_hints, name)
            else:
                self.remove_component_index_hints(index_hints, name, component)

    def resource_index_hint(self, expr, constants, index_hints, call_stack=None):
        """Return a conservative known lower-bound index for resource access."""
        call_stack = call_stack or set()
        index = self.literal_int_value(expr, constants)
        if index is not None:
            return index

        if expr is None:
            return None

        name = self.expression_name(expr)
        if name in index_hints:
            return index_hints[name]

        component_key = self.expression_component_hint_key(expr)
        if component_key in index_hints:
            return index_hints[component_key]

        correlated_index = self.correlated_resource_index_hint(
            expr, constants, index_hints, call_stack
        )
        if correlated_index is not None:
            return correlated_index

        if isinstance(expr, FunctionCallNode):
            call_name = self.function_call_name(expr)
            call_index = self.call_return_index_hint(
                expr, constants, index_hints, call_stack
            )
            if call_index is not None:
                return call_index
            if call_name in self.return_index_hints:
                return self.return_index_hints[call_name]
            builtin_index = self.builtin_resource_index_hint(
                expr, call_name, constants, index_hints, call_stack
            )
            if builtin_index is not None:
                return builtin_index

        if isinstance(expr, BinaryOpNode):
            return self.binary_resource_index_hint(
                expr, constants, index_hints, call_stack
            )

        if "TernaryOp" not in expr.__class__.__name__:
            return None

        branch_indexes = []
        for attr in ("true_expr", "false_expr"):
            branch_index = self.resource_index_hint(
                getattr(expr, attr, None), constants, index_hints, call_stack
            )
            if branch_index is not None and branch_index >= 0:
                branch_indexes.append(branch_index)
        return max(branch_indexes) if branch_indexes else None

    def binary_resource_index_hint(self, expr, constants, index_hints, call_stack):
        """Infer a lower-bound index through monotonic integer arithmetic."""
        operator = getattr(expr, "operator", getattr(expr, "op", None))
        left_expr = getattr(expr, "left", None)
        right_expr = getattr(expr, "right", None)
        left = self.resource_index_hint(left_expr, constants, index_hints, call_stack)
        right = self.resource_index_hint(right_expr, constants, index_hints, call_stack)
        right_exact = self.literal_int_value(right_expr, constants)

        result = None
        if operator == "+" and left is not None and right is not None:
            result = left + right
        elif operator == "-" and left is not None and right_exact is not None:
            result = left - right_exact
        elif (
            operator == "*"
            and left is not None
            and right is not None
            and left >= 0
            and right >= 0
        ):
            result = left * right

        return result if result is not None and result >= 0 else None

    def builtin_resource_index_hint(
        self, call, call_name, constants, index_hints, call_stack
    ):
        """Infer bounds from common integer built-ins used for array indices."""
        args = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        if call_name == "min" and len(args) >= 2:
            exact_bounds = [
                value
                for value in (self.literal_int_value(arg, constants) for arg in args)
                if value is not None and value >= 0
            ]
            return min(exact_bounds) if exact_bounds else None

        if call_name == "max" and len(args) >= 2:
            lower_bounds = [
                value
                for value in (
                    self.resource_index_hint(arg, constants, index_hints, call_stack)
                    for arg in args
                )
                if value is not None and value >= 0
            ]
            return max(lower_bounds) if lower_bounds else None

        if call_name == "clamp" and len(args) >= 3:
            max_value = self.literal_int_value(args[2], constants)
            return max_value if max_value is not None and max_value >= 0 else None

        return None

    def call_index_context(self, call, callee, constants, index_hints, call_stack):
        call_constants = self.function_initial_constants(callee)
        call_index_hints = {}
        args = getattr(call, "arguments", getattr(call, "args", []))
        for index, param in enumerate(getattr(callee, "parameters", []) or []):
            if index >= len(args):
                break
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            arg = args[index]
            exact_value = self.literal_int_value(arg, constants)
            if exact_value is not None:
                call_constants[param_name] = exact_value
                call_index_hints[param_name] = exact_value
                continue
            arg_hint = self.resource_index_hint(arg, constants, index_hints, call_stack)
            if arg_hint is not None and arg_hint >= 0:
                call_index_hints[param_name] = arg_hint
            self.record_vector_component_index_hints(
                param_name,
                arg,
                constants,
                call_index_hints,
                call_stack,
                source_index_hints=index_hints,
            )
        return call_constants, call_index_hints

    def call_return_index_hint(self, call, constants, index_hints, call_stack):
        call_name = self.function_call_name(call)
        callee = self.functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return None

        call_constants, call_index_hints = self.call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        return_hints = []
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        for (
            node,
            visible_constants,
            visible_index_hints,
        ) in self.walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            return_hint = self.resource_index_hint(
                getattr(node, "value", None),
                visible_constants,
                visible_index_hints,
                next_call_stack,
            )
            if return_hint is not None and return_hint >= 0:
                return_hints.append(return_hint)
        return max(return_hints) if return_hints else None

    def call_vector_component_hints(self, call, constants, index_hints, call_stack):
        call_name = self.function_call_name(call)
        callee = self.functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return {}

        call_constants, call_index_hints = self.call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        component_values = {}
        for (
            node,
            visible_constants,
            visible_index_hints,
        ) in self.walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            component_hints = self.vector_component_index_hints(
                getattr(node, "value", None),
                visible_constants,
                visible_index_hints,
                next_call_stack,
            )
            for index, value in component_hints.items():
                if value is None or value < 0:
                    continue
                component_values[index] = max(value, component_values.get(index, 0))
        return component_values

    def call_vector_component_alternatives(
        self, call, constants, index_hints, call_stack
    ):
        call_name = self.function_call_name(call)
        callee = self.functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return ()

        call_constants, call_index_hints = self.call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        alternatives = []
        for (
            node,
            visible_constants,
            visible_index_hints,
        ) in self.walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            alternatives.extend(
                self.vector_component_alternatives(
                    getattr(node, "value", None),
                    visible_constants,
                    visible_index_hints,
                    next_call_stack,
                )
            )
        return self.normalize_vector_alternatives(alternatives)

    def merge_existing_index_hints(self, target, source, names):
        """Propagate updates to known outer index aliases through a nested block."""
        for name in names:
            if (
                isinstance(name, str)
                and "." not in name
                and not self.is_vector_alternatives_key(name)
            ):
                keys = [
                    key
                    for key in set(target).union(source)
                    if self.hint_key_base_name(key) == name
                ]
            else:
                keys = [name]
            for key in keys:
                if key in source:
                    target[key] = source[key]
                else:
                    target.pop(key, None)

    def merge_branch_index_hints(self, target, *branches):
        """Merge possible branch-local lower-bound index aliases."""
        names = set(target)
        for branch in branches:
            names.update(branch)

        for name in names:
            if self.is_vector_alternatives_key(name):
                continue

            branch_values = [
                branch[name]
                for branch in branches
                if name in branch and self.is_nonnegative_int_hint(branch[name])
            ]
            if branch_values:
                target[name] = max(branch_values)
            else:
                target.pop(name, None)
        self.merge_branch_vector_alternative_hints(target, *branches)

    def merge_loop_index_hints(self, target, *branches, compound_targets=None):
        """Merge loop-carried updates for aliases already visible after the loop."""
        for name in list(target):
            if self.is_vector_alternatives_key(name):
                alternatives = []
                alternatives.extend(
                    self.normalize_vector_alternatives(target.get(name))
                )
                for branch in branches:
                    alternatives.extend(
                        self.normalize_vector_alternatives(branch.get(name))
                    )
                alternatives = self.normalize_vector_alternatives(alternatives)
                if alternatives:
                    target[name] = alternatives
                else:
                    target.pop(name, None)
                continue

            values = [
                branch[name]
                for branch in branches
                if name in branch and self.is_nonnegative_int_hint(branch[name])
            ]
            if self.is_nonnegative_int_hint(target[name]):
                values.append(target[name])
            if values:
                target[name] = max(values)
            else:
                target.pop(name, None)
        self.remove_compound_loop_targets(target, compound_targets or ())

    def is_switch_default_case(self, case):
        return getattr(case, "value", None) is None

    def is_match_default_arm(self, arm):
        pattern = getattr(arm, "pattern", None)
        return pattern is None or "Wildcard" in pattern.__class__.__name__
