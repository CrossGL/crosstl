"""Index evaluation over vector-component alternatives for resource-array size inference."""

from ..ast import (
    BinaryOpNode,
    FunctionCallNode,
    ReturnNode,
)
from ..refusals import TargetRefusal

COMPONENT_ALIASES_BY_INDEX = (
    ("x", "r", "s"),
    ("y", "g", "t"),
    ("z", "b", "p"),
    ("w", "a", "q"),
)
COMPONENT_INDEX_BY_NAME = {
    component: index
    for index, aliases in enumerate(COMPONENT_ALIASES_BY_INDEX)
    for component in aliases
}
MAX_STATIC_LOOP_ITERATIONS_FOR_HINTS = 1024
MAX_VECTOR_ALTERNATIVE_BINDINGS = 64


class ResourceArrayIndexAlternativesMixin:
    """Methods of :class:`ResourceArraySizeInference`; see :mod:`.resource_array_size_inference`."""

    def assert_not_larger_than_known_fixed(self, name, fixed_size, required_size):
        """Raise when inferred usage exceeds a known fixed array size."""
        if required_size > fixed_size:
            raise TargetRefusal(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {fixed_size} and {required_size}"
            )

    def function_initial_constants(self, func):
        if self.initial_literal_int_constants is None:
            return dict(self.visible_literal_int_constants(func))
        return dict(self.initial_literal_int_constants(func))

    def component_hint_key(self, name, component):
        return f"{name}.{component}"

    def vector_alternatives_key(self, name):
        return ("vector_component_alternatives", name)

    def is_vector_alternatives_key(self, key):
        return (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "vector_component_alternatives"
        )

    def hint_key_base_name(self, key):
        if self.is_vector_alternatives_key(key):
            return key[1]
        if isinstance(key, str):
            return key.split(".", 1)[0]
        return key

    def is_nonnegative_int_hint(self, value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def remove_index_hints_for_name(self, index_hints, name):
        for key in list(index_hints):
            if self.hint_key_base_name(key) == name:
                index_hints.pop(key, None)

    def remove_component_index_hints(self, index_hints, name, component):
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return
        index_hints.pop(self.vector_alternatives_key(name), None)
        for alias in COMPONENT_ALIASES_BY_INDEX[component_index]:
            index_hints.pop(self.component_hint_key(name, alias), None)

    def outer_hint_names_without_locals(self, names, local_names):
        return {
            name
            for name in names
            if not any(
                self.hint_key_base_name(name) == local_name
                for local_name in local_names
            )
        }

    def expression_component_parts(self, expr):
        components = getattr(expr, "member", None)
        vector_expr = getattr(expr, "object_expr", None)
        if components is None:
            components = getattr(expr, "components", None)
            vector_expr = getattr(expr, "vector_expr", None)
        if (
            not isinstance(components, str)
            or len(components) != 1
            or components not in COMPONENT_INDEX_BY_NAME
        ):
            return None
        vector_name = self.expression_name(vector_expr)
        if not vector_name:
            return None
        return vector_name, components

    def expression_component_hint_key(self, expr):
        parts = self.expression_component_parts(expr)
        if parts is None:
            return None
        vector_name, component = parts
        return self.component_hint_key(vector_name, component)

    def expression_swizzle_parts(self, expr):
        components = getattr(expr, "member", None)
        vector_expr = getattr(expr, "object_expr", None)
        if components is None:
            components = getattr(expr, "components", None)
            vector_expr = getattr(expr, "vector_expr", None)
        if (
            not isinstance(components, str)
            or len(components) < 2
            or any(component not in COMPONENT_INDEX_BY_NAME for component in components)
        ):
            return None
        vector_name = self.expression_name(vector_expr)
        if not vector_name:
            return None
        return vector_name, components

    def component_hint_parts(self, key):
        if not isinstance(key, str) or "." not in key:
            return None
        name, component = key.rsplit(".", 1)
        if not name or component not in COMPONENT_INDEX_BY_NAME:
            return None
        return name, component

    def type_name(self, node):
        if isinstance(node, str):
            return node
        if node is None:
            return None
        name = getattr(node, "name", None)
        if isinstance(name, str):
            return name
        identifier = getattr(node, "identifier", None)
        if isinstance(identifier, str):
            return identifier
        return None

    def integer_vector_size(self, name):
        if name in {"int2", "uint2", "ivec2", "uvec2"}:
            return 2
        if name in {"int3", "uint3", "ivec3", "uvec3"}:
            return 3
        if name in {"int4", "uint4", "ivec4", "uvec4"}:
            return 4
        return None

    def existing_vector_component_hints(self, name, index_hints):
        hints = {}
        for index, aliases in enumerate(COMPONENT_ALIASES_BY_INDEX):
            values = [
                index_hints[self.component_hint_key(name, alias)]
                for alias in aliases
                if self.component_hint_key(name, alias) in index_hints
                and self.is_nonnegative_int_hint(
                    index_hints[self.component_hint_key(name, alias)]
                )
            ]
            if values:
                hints[index] = max(values)
        return hints

    def normalize_vector_alternatives(self, alternatives):
        normalized = []
        seen = set()
        for alternative in alternatives or ():
            values = list(alternative)[: len(COMPONENT_ALIASES_BY_INDEX)]
            values.extend([None] * (len(COMPONENT_ALIASES_BY_INDEX) - len(values)))
            values = tuple(values[: len(COMPONENT_ALIASES_BY_INDEX)])
            if values in seen or not any(value is not None for value in values):
                continue
            seen.add(values)
            normalized.append(values)
        return tuple(normalized)

    def vector_alternatives_from_component_hints(self, component_hints):
        values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
        for index, value in component_hints.items():
            if 0 <= index < len(values) and self.is_nonnegative_int_hint(value):
                values[index] = value
        return self.normalize_vector_alternatives((values,))

    def existing_vector_component_alternatives(self, name, index_hints):
        return self.normalize_vector_alternatives(
            index_hints.get(self.vector_alternatives_key(name))
        )

    def vector_hint_names(self, *hint_sets):
        names = set()
        for index_hints in hint_sets:
            for key in index_hints:
                if self.is_vector_alternatives_key(key):
                    names.add(key[1])
                    continue
                component_parts = self.component_hint_parts(key)
                if component_parts is not None:
                    names.add(component_parts[0])
        return names

    def branch_vector_alternatives(self, name, index_hints):
        alternatives = self.existing_vector_component_alternatives(name, index_hints)
        if alternatives:
            return alternatives
        return self.vector_alternatives_from_component_hints(
            self.existing_vector_component_hints(name, index_hints)
        )

    def vector_swizzle_alternatives(self, expr, index_hints):
        swizzle_parts = self.expression_swizzle_parts(expr)
        if swizzle_parts is None:
            return ()

        name, components = swizzle_parts
        alternatives = self.branch_vector_alternatives(name, index_hints)
        if not alternatives:
            return ()

        swizzled = []
        for alternative in alternatives:
            values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
            for index, component in enumerate(components[: len(values)]):
                source_index = COMPONENT_INDEX_BY_NAME[component]
                values[index] = alternative[source_index]
            swizzled.append(values)
        return self.normalize_vector_alternatives(swizzled)

    def merge_branch_vector_alternative_hints(self, target, *branches):
        for name in self.vector_hint_names(target, *branches):
            alternatives = []
            for branch in branches:
                branch_alternatives = self.branch_vector_alternatives(name, branch)
                if not branch_alternatives:
                    alternatives = []
                    break
                alternatives.extend(branch_alternatives)

            alternatives = self.normalize_vector_alternatives(alternatives)
            key = self.vector_alternatives_key(name)
            if alternatives:
                target[key] = alternatives
            else:
                target.pop(key, None)

    def vector_alternative_names_for_expr(self, expr, index_hints):
        names = set()
        seen = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    visit(child)
                return

            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)

            component_parts = self.expression_component_parts(value)
            if component_parts is not None:
                vector_name, _component = component_parts
                if self.existing_vector_component_alternatives(
                    vector_name, index_hints
                ):
                    names.add(vector_name)

            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        visit(expr)
        return names

    def alternative_bindings_for_names(self, names, index_hints):
        alternatives_by_name = []
        binding_count = 1
        for name in sorted(names):
            alternatives = self.existing_vector_component_alternatives(
                name, index_hints
            )
            if not alternatives:
                return None
            binding_count *= len(alternatives)
            if binding_count > MAX_VECTOR_ALTERNATIVE_BINDINGS:
                return None
            alternatives_by_name.append((name, alternatives))

        bindings = [{}]
        for name, alternatives in alternatives_by_name:
            next_bindings = []
            for binding in bindings:
                for alternative in alternatives:
                    next_binding = dict(binding)
                    next_binding[name] = alternative
                    next_bindings.append(next_binding)
            bindings = next_bindings
        return bindings

    def index_hints_with_alternative_bindings(self, index_hints, alternative_bindings):
        bound_hints = dict(index_hints)
        for name, alternative in alternative_bindings.items():
            normalized = self.normalize_vector_alternatives((alternative,))
            if not normalized:
                continue
            bound_hints[self.vector_alternatives_key(name)] = normalized
            for index, value in enumerate(normalized[0]):
                if value is None:
                    continue
                for alias in COMPONENT_ALIASES_BY_INDEX[index]:
                    bound_hints[self.component_hint_key(name, alias)] = value
        return bound_hints

    def evaluate_index_with_alternatives(
        self, expr, constants, index_hints, alternative_bindings, call_stack
    ):
        index = self.literal_int_value(expr, constants)
        if index is not None:
            return index
        if expr is None:
            return None

        component_parts = self.expression_component_parts(expr)
        if component_parts is not None:
            vector_name, component = component_parts
            component_index = COMPONENT_INDEX_BY_NAME.get(component)
            alternative = alternative_bindings.get(vector_name)
            if alternative is not None and component_index is not None:
                value = alternative[component_index]
                if value is not None:
                    return value
            component_key = self.component_hint_key(vector_name, component)
            component_hint = index_hints.get(component_key)
            return (
                component_hint if self.is_nonnegative_int_hint(component_hint) else None
            )

        name = self.expression_name(expr)
        if name in index_hints and self.is_nonnegative_int_hint(index_hints[name]):
            return index_hints[name]

        if isinstance(expr, BinaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            left_expr = getattr(expr, "left", None)
            right_expr = getattr(expr, "right", None)
            left = self.evaluate_index_with_alternatives(
                left_expr, constants, index_hints, alternative_bindings, call_stack
            )
            right = self.evaluate_index_with_alternatives(
                right_expr, constants, index_hints, alternative_bindings, call_stack
            )
            right_exact = self.literal_int_value(right_expr, constants)
            right_alternative_exact = right is not None and bool(
                self.vector_alternative_names_for_expr(right_expr, index_hints)
            )

            result = None
            if operator == "+" and left is not None and right is not None:
                result = left + right
            elif (
                operator == "-"
                and left is not None
                and (right_exact is not None or right_alternative_exact)
            ):
                result = left - (right_exact if right_exact is not None else right)
            elif (
                operator == "*"
                and left is not None
                and right is not None
                and left >= 0
                and right >= 0
            ):
                result = left * right
            return result if result is not None and result >= 0 else None

        if isinstance(expr, FunctionCallNode):
            call_name = self.function_call_name(expr)
            call_index = self.call_return_index_hint_with_alternatives(
                expr,
                constants,
                index_hints,
                alternative_bindings,
                call_stack,
            )
            if call_index is not None:
                return call_index

            args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
            arg_values = [
                self.evaluate_index_with_alternatives(
                    arg, constants, index_hints, alternative_bindings, call_stack
                )
                for arg in args
            ]
            if call_name == "min" and len(arg_values) >= 2:
                values = [value for value in arg_values if value is not None]
                return min(values) if values else None
            if call_name == "max" and len(arg_values) >= 2:
                if any(value is None for value in arg_values):
                    return None
                return max(arg_values)
            if call_name == "clamp" and len(args) >= 3:
                max_value = self.literal_int_value(args[2], constants)
                return max_value if max_value is not None and max_value >= 0 else None

        if "TernaryOp" in expr.__class__.__name__:
            branch_indexes = []
            for attr in ("true_expr", "false_expr"):
                branch_index = self.evaluate_index_with_alternatives(
                    getattr(expr, attr, None),
                    constants,
                    index_hints,
                    alternative_bindings,
                    call_stack,
                )
                if branch_index is not None and branch_index >= 0:
                    branch_indexes.append(branch_index)
            return max(branch_indexes) if branch_indexes else None

        return None

    def correlated_resource_index_hint(self, expr, constants, index_hints, call_stack):
        names = self.vector_alternative_names_for_expr(expr, index_hints)
        if not names:
            return None
        bindings = self.alternative_bindings_for_names(names, index_hints)
        if not bindings:
            return None

        indexes = []
        for binding in bindings:
            index = self.evaluate_index_with_alternatives(
                expr, constants, index_hints, binding, call_stack
            )
            if index is None or index < 0:
                return None
            indexes.append(index)
        return max(indexes) if indexes else None

    def call_index_context_with_alternatives(
        self,
        call,
        callee,
        constants,
        index_hints,
        alternative_bindings,
        call_stack,
    ):
        call_constants = self.function_initial_constants(callee)
        call_index_hints = {}
        bound_index_hints = self.index_hints_with_alternative_bindings(
            index_hints, alternative_bindings
        )
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
            else:
                alternative_names = self.vector_alternative_names_for_expr(
                    arg, index_hints
                )
                arg_hint = self.evaluate_index_with_alternatives(
                    arg,
                    constants,
                    bound_index_hints,
                    alternative_bindings,
                    call_stack,
                )
                if arg_hint is not None and arg_hint >= 0:
                    call_index_hints[param_name] = arg_hint
                    if alternative_names:
                        call_constants[param_name] = arg_hint

            self.record_vector_component_index_hints(
                param_name,
                arg,
                constants,
                call_index_hints,
                call_stack,
                source_index_hints=bound_index_hints,
            )

        return call_constants, call_index_hints

    def call_return_index_hint_with_alternatives(
        self, call, constants, index_hints, alternative_bindings, call_stack
    ):
        call_stack = call_stack or set()
        call_name = self.function_call_name(call)
        callee = self.functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return None

        call_constants, call_index_hints = self.call_index_context_with_alternatives(
            call,
            callee,
            constants,
            index_hints,
            alternative_bindings,
            call_stack,
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

    def component_assignment_alternative_value(
        self,
        operator,
        current_value,
        assigned_value,
        constants,
        index_hints,
        alternative_bindings,
        call_stack,
    ):
        if operator == "=":
            return self.evaluate_index_with_alternatives(
                assigned_value,
                constants,
                index_hints,
                alternative_bindings,
                call_stack,
            )

        if current_value is None or current_value < 0:
            return None

        rhs = self.evaluate_index_with_alternatives(
            assigned_value,
            constants,
            index_hints,
            alternative_bindings,
            call_stack,
        )
        rhs_exact = self.literal_int_value(assigned_value, constants)
        result = None
        if operator == "+=" and rhs is not None:
            result = current_value + rhs
        elif operator == "-=" and rhs_exact is not None:
            result = current_value - rhs_exact
        elif operator == "*=" and rhs is not None and current_value >= 0 and rhs >= 0:
            result = current_value * rhs

        return result if result is not None and result >= 0 else None

    def updated_component_assignment_alternatives(
        self,
        name,
        component,
        operator,
        assigned_value,
        constants,
        index_hints,
        call_stack,
    ):
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return ()

        alternatives = self.existing_vector_component_alternatives(name, index_hints)
        if not alternatives:
            return ()

        updated = []
        for alternative in alternatives:
            current_value = alternative[component_index]
            value = self.component_assignment_alternative_value(
                operator,
                current_value,
                assigned_value,
                constants,
                index_hints,
                {name: alternative},
                call_stack,
            )
            if value is None:
                return ()
            next_alternative = list(alternative)
            next_alternative[component_index] = value
            updated.append(next_alternative)

        return self.normalize_vector_alternatives(updated)

    def vector_component_alternatives(
        self, expr, constants, index_hints, call_stack=None
    ):
        call_stack = call_stack or set()
        if expr is None:
            return ()

        name = self.expression_name(expr)
        if name:
            alternatives = self.existing_vector_component_alternatives(
                name, index_hints
            )
            if alternatives:
                return alternatives
            component_hints = self.existing_vector_component_hints(name, index_hints)
            if component_hints:
                return self.vector_alternatives_from_component_hints(component_hints)

        alternatives = self.vector_swizzle_alternatives(expr, index_hints)
        if alternatives:
            return alternatives

        if "TernaryOp" in expr.__class__.__name__:
            alternatives = []
            for attr in ("true_expr", "false_expr"):
                alternatives.extend(
                    self.vector_component_alternatives(
                        getattr(expr, attr, None), constants, index_hints, call_stack
                    )
                )
            return self.normalize_vector_alternatives(alternatives)

        class_name = expr.__class__.__name__
        if "FunctionCall" in class_name:
            constructor = getattr(expr, "function", getattr(expr, "name", None))
            args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        elif "Constructor" in class_name:
            constructor = getattr(expr, "constructor_type", None)
            args = list(getattr(expr, "arguments", []) or [])
        else:
            return ()

        size = self.integer_vector_size(self.type_name(constructor))
        if size is None:
            if "FunctionCall" in class_name:
                return self.call_vector_component_alternatives(
                    expr, constants, index_hints, call_stack
                )
            return ()

        if len(args) == 1:
            copied_alternatives = self.vector_component_alternatives(
                args[0], constants, index_hints, call_stack
            )
            if copied_alternatives:
                alternatives = []
                for alternative in copied_alternatives:
                    values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
                    for index in range(size):
                        values[index] = alternative[index]
                    alternatives.append(values)
                return self.normalize_vector_alternatives(alternatives)

            names = self.vector_alternative_names_for_expr(args[0], index_hints)
            bindings = (
                self.alternative_bindings_for_names(names, index_hints)
                if names
                else [{}]
            )
            if not bindings:
                return ()
            alternatives = []
            for binding in bindings:
                value = self.evaluate_index_with_alternatives(
                    args[0], constants, index_hints, binding, call_stack
                )
                if value is None or value < 0:
                    return ()
                values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
                for index in range(size):
                    values[index] = value
                alternatives.append(values)
            return self.normalize_vector_alternatives(alternatives)

        names = set()
        for arg in args[:size]:
            names.update(self.vector_alternative_names_for_expr(arg, index_hints))
        bindings = (
            self.alternative_bindings_for_names(names, index_hints) if names else [{}]
        )
        if not bindings:
            return ()

        alternatives = []
        for binding in bindings:
            values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
            for index, arg in enumerate(args[:size]):
                value = self.evaluate_index_with_alternatives(
                    arg, constants, index_hints, binding, call_stack
                )
                if value is not None and value >= 0:
                    values[index] = value
                else:
                    fallback = self.resource_index_hint(
                        arg, constants, index_hints, call_stack
                    )
                    if fallback is not None and fallback >= 0:
                        values[index] = fallback
            alternatives.append(values)
        return self.normalize_vector_alternatives(alternatives)

    def vector_component_index_hints(
        self, expr, constants, index_hints, call_stack=None
    ):
        call_stack = call_stack or set()
        if expr is None:
            return {}

        name = self.expression_name(expr)
        if name:
            hints = self.existing_vector_component_hints(name, index_hints)
            if hints:
                return hints

        if "TernaryOp" in expr.__class__.__name__:
            branch_hints = [
                self.vector_component_index_hints(
                    getattr(expr, attr, None), constants, index_hints, call_stack
                )
                for attr in ("true_expr", "false_expr")
            ]
            merged = {}
            for index in set().union(*(branch.keys() for branch in branch_hints)):
                values = [
                    branch[index]
                    for branch in branch_hints
                    if index in branch
                    and branch[index] is not None
                    and branch[index] >= 0
                ]
                if values:
                    merged[index] = max(values)
            return merged

        class_name = expr.__class__.__name__
        if "FunctionCall" in class_name:
            constructor = getattr(expr, "function", getattr(expr, "name", None))
            args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        elif "Constructor" in class_name:
            constructor = getattr(expr, "constructor_type", None)
            args = list(getattr(expr, "arguments", []) or [])
        else:
            return {}

        size = self.integer_vector_size(self.type_name(constructor))
        if size is None:
            if "FunctionCall" in class_name:
                return self.call_vector_component_hints(
                    expr, constants, index_hints, call_stack
                )
            return {}

        hints = {}
        if len(args) == 1:
            value = self.resource_index_hint(
                args[0], constants, index_hints, call_stack
            )
            if value is not None and value >= 0:
                for index in range(size):
                    hints[index] = value
            return hints

        for index, arg in enumerate(args[:size]):
            value = self.resource_index_hint(arg, constants, index_hints, call_stack)
            if value is not None and value >= 0:
                hints[index] = value
        return hints

    def record_vector_component_index_hints(
        self,
        name,
        expr,
        constants,
        index_hints,
        call_stack=None,
        source_index_hints=None,
    ):
        source_hints = source_index_hints or index_hints
        component_hints = self.vector_component_index_hints(
            expr, constants, source_hints, call_stack
        )
        for index, value in component_hints.items():
            for alias in COMPONENT_ALIASES_BY_INDEX[index]:
                index_hints[self.component_hint_key(name, alias)] = value
        alternatives = self.vector_component_alternatives(
            expr, constants, source_hints, call_stack
        )
        if alternatives:
            index_hints[self.vector_alternatives_key(name)] = alternatives

    def record_component_index_hint(self, name, component, value, index_hints):
        if value is None or value < 0:
            return
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return
        for alias in COMPONENT_ALIASES_BY_INDEX[component_index]:
            index_hints[self.component_hint_key(name, alias)] = value

    def compound_assignment_index_hint(
        self,
        operator,
        current_value,
        assigned_value,
        constants,
        index_hints,
        call_stack=None,
    ):
        if current_value is None or current_value < 0:
            return None

        rhs = self.resource_index_hint(
            assigned_value, constants, index_hints, call_stack
        )
        rhs_exact = self.literal_int_value(assigned_value, constants)
        result = None
        if operator == "+=" and rhs is not None:
            result = current_value + rhs
        elif operator == "-=" and rhs_exact is not None:
            result = current_value - rhs_exact
        elif operator == "*=" and rhs is not None and rhs >= 0:
            result = current_value * rhs

        return result if result is not None and result >= 0 else None

    def known_index_value(self, expr, constants, index_hints):
        value = self.literal_int_value(expr, constants)
        if value is not None:
            return value

        name = self.expression_name(expr)
        if name in index_hints:
            return index_hints[name]
        return None
