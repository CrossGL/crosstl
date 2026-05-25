"""Helpers for collecting resource-array size hints from CrossGL AST nodes."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CaseNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchArmNode,
    MatchNode,
    ReturnNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)


def split_array_suffix(array_suffix):
    """Split a C-style array suffix into dimension strings."""
    dimensions = []
    offset = 0
    while offset < len(array_suffix):
        if array_suffix[offset] != "[":
            return None
        end = array_suffix.find("]", offset + 1)
        if end == -1:
            return None
        dimensions.append(array_suffix[offset + 1 : end])
        offset = end + 1
    return dimensions


def format_array_declarator(
    mapped_base,
    name,
    array_suffix,
    dynamic_array_as_pointer=True,
):
    """Format a typed C-style array declarator for generated code."""
    if not dynamic_array_as_pointer or "[]" not in array_suffix:
        return f"{mapped_base} {name}{array_suffix}"

    dimensions = split_array_suffix(array_suffix)
    if not dimensions:
        return f"{mapped_base} {name}{array_suffix}"

    if dimensions[0] == "":
        trailing_dimensions = dimensions[1:]
        if not trailing_dimensions:
            return f"{mapped_base}* {name}"
        if all(dimension != "" for dimension in trailing_dimensions):
            trailing_suffix = "".join(
                f"[{dimension}]" for dimension in trailing_dimensions
            )
            return f"{mapped_base} (*{name}){trailing_suffix}"

    collapsed_suffix = "".join(
        f"[{dimension}]" for dimension in dimensions if dimension != ""
    )
    return f"{mapped_base}* {name}{collapsed_suffix}"


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


def collect_resource_array_size_hints(
    *,
    global_arrays,
    function_arrays,
    fixed_global_array_sizes=None,
    fixed_function_array_sizes=None,
    functions,
    walk_nodes,
    expression_name,
    literal_int_value,
    visible_literal_int_constants,
    function_call_name,
    initial_size,
    format_size,
    initial_literal_int_constants=None,
):
    """Infer resource-array sizes from literal accesses and call propagation."""
    fixed_global_array_sizes = fixed_global_array_sizes or {}
    fixed_function_array_sizes = fixed_function_array_sizes or {}
    global_hints = {name: initial_size for name in global_arrays}
    function_hints = {
        func_name: {param_name: initial_size for param_name in params}
        for func_name, params in function_arrays.items()
    }
    fixed_requirements = {}
    functions_by_name = {getattr(func, "name", None): func for func in functions}
    functions_by_name = {name: func for name, func in functions_by_name.items() if name}
    return_index_hints = {}

    def assert_not_larger_than_known_fixed(name, fixed_size, required_size):
        """Raise when inferred usage exceeds a known fixed array size."""
        if required_size > fixed_size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {fixed_size} and {required_size}"
            )

    def function_initial_constants(func):
        if initial_literal_int_constants is None:
            return dict(visible_literal_int_constants(func))
        return dict(initial_literal_int_constants(func))

    def component_hint_key(name, component):
        return f"{name}.{component}"

    def vector_alternatives_key(name):
        return ("vector_component_alternatives", name)

    def is_vector_alternatives_key(key):
        return (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "vector_component_alternatives"
        )

    def hint_key_base_name(key):
        if is_vector_alternatives_key(key):
            return key[1]
        if isinstance(key, str):
            return key.split(".", 1)[0]
        return key

    def is_nonnegative_int_hint(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def remove_index_hints_for_name(index_hints, name):
        for key in list(index_hints):
            if hint_key_base_name(key) == name:
                index_hints.pop(key, None)

    def remove_component_index_hints(index_hints, name, component):
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return
        index_hints.pop(vector_alternatives_key(name), None)
        for alias in COMPONENT_ALIASES_BY_INDEX[component_index]:
            index_hints.pop(component_hint_key(name, alias), None)

    def outer_hint_names_without_locals(names, local_names):
        return {
            name
            for name in names
            if not any(
                hint_key_base_name(name) == local_name for local_name in local_names
            )
        }

    def expression_component_parts(expr):
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
        vector_name = expression_name(vector_expr)
        if not vector_name:
            return None
        return vector_name, components

    def expression_component_hint_key(expr):
        parts = expression_component_parts(expr)
        if parts is None:
            return None
        vector_name, component = parts
        return component_hint_key(vector_name, component)

    def expression_swizzle_parts(expr):
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
        vector_name = expression_name(vector_expr)
        if not vector_name:
            return None
        return vector_name, components

    def component_hint_parts(key):
        if not isinstance(key, str) or "." not in key:
            return None
        name, component = key.rsplit(".", 1)
        if not name or component not in COMPONENT_INDEX_BY_NAME:
            return None
        return name, component

    def type_name(node):
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

    def integer_vector_size(name):
        if name in {"int2", "uint2", "ivec2", "uvec2"}:
            return 2
        if name in {"int3", "uint3", "ivec3", "uvec3"}:
            return 3
        if name in {"int4", "uint4", "ivec4", "uvec4"}:
            return 4
        return None

    def existing_vector_component_hints(name, index_hints):
        hints = {}
        for index, aliases in enumerate(COMPONENT_ALIASES_BY_INDEX):
            values = [
                index_hints[component_hint_key(name, alias)]
                for alias in aliases
                if component_hint_key(name, alias) in index_hints
                and is_nonnegative_int_hint(
                    index_hints[component_hint_key(name, alias)]
                )
            ]
            if values:
                hints[index] = max(values)
        return hints

    def normalize_vector_alternatives(alternatives):
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

    def vector_alternatives_from_component_hints(component_hints):
        values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
        for index, value in component_hints.items():
            if 0 <= index < len(values) and is_nonnegative_int_hint(value):
                values[index] = value
        return normalize_vector_alternatives((values,))

    def existing_vector_component_alternatives(name, index_hints):
        return normalize_vector_alternatives(
            index_hints.get(vector_alternatives_key(name))
        )

    def vector_hint_names(*hint_sets):
        names = set()
        for index_hints in hint_sets:
            for key in index_hints:
                if is_vector_alternatives_key(key):
                    names.add(key[1])
                    continue
                component_parts = component_hint_parts(key)
                if component_parts is not None:
                    names.add(component_parts[0])
        return names

    def branch_vector_alternatives(name, index_hints):
        alternatives = existing_vector_component_alternatives(name, index_hints)
        if alternatives:
            return alternatives
        return vector_alternatives_from_component_hints(
            existing_vector_component_hints(name, index_hints)
        )

    def vector_swizzle_alternatives(expr, index_hints):
        swizzle_parts = expression_swizzle_parts(expr)
        if swizzle_parts is None:
            return ()

        name, components = swizzle_parts
        alternatives = branch_vector_alternatives(name, index_hints)
        if not alternatives:
            return ()

        swizzled = []
        for alternative in alternatives:
            values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
            for index, component in enumerate(components[: len(values)]):
                source_index = COMPONENT_INDEX_BY_NAME[component]
                values[index] = alternative[source_index]
            swizzled.append(values)
        return normalize_vector_alternatives(swizzled)

    def merge_branch_vector_alternative_hints(target, *branches):
        for name in vector_hint_names(target, *branches):
            alternatives = []
            for branch in branches:
                branch_alternatives = branch_vector_alternatives(name, branch)
                if not branch_alternatives:
                    alternatives = []
                    break
                alternatives.extend(branch_alternatives)

            alternatives = normalize_vector_alternatives(alternatives)
            key = vector_alternatives_key(name)
            if alternatives:
                target[key] = alternatives
            else:
                target.pop(key, None)

    def vector_alternative_names_for_expr(expr, index_hints):
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

            component_parts = expression_component_parts(value)
            if component_parts is not None:
                vector_name, _component = component_parts
                if existing_vector_component_alternatives(vector_name, index_hints):
                    names.add(vector_name)

            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        visit(expr)
        return names

    def alternative_bindings_for_names(names, index_hints):
        alternatives_by_name = []
        binding_count = 1
        for name in sorted(names):
            alternatives = existing_vector_component_alternatives(name, index_hints)
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

    def index_hints_with_alternative_bindings(index_hints, alternative_bindings):
        bound_hints = dict(index_hints)
        for name, alternative in alternative_bindings.items():
            normalized = normalize_vector_alternatives((alternative,))
            if not normalized:
                continue
            bound_hints[vector_alternatives_key(name)] = normalized
            for index, value in enumerate(normalized[0]):
                if value is None:
                    continue
                for alias in COMPONENT_ALIASES_BY_INDEX[index]:
                    bound_hints[component_hint_key(name, alias)] = value
        return bound_hints

    def evaluate_index_with_alternatives(
        expr, constants, index_hints, alternative_bindings, call_stack
    ):
        index = literal_int_value(expr, constants)
        if index is not None:
            return index
        if expr is None:
            return None

        component_parts = expression_component_parts(expr)
        if component_parts is not None:
            vector_name, component = component_parts
            component_index = COMPONENT_INDEX_BY_NAME.get(component)
            alternative = alternative_bindings.get(vector_name)
            if alternative is not None and component_index is not None:
                value = alternative[component_index]
                if value is not None:
                    return value
            component_key = component_hint_key(vector_name, component)
            component_hint = index_hints.get(component_key)
            return component_hint if is_nonnegative_int_hint(component_hint) else None

        name = expression_name(expr)
        if name in index_hints and is_nonnegative_int_hint(index_hints[name]):
            return index_hints[name]

        if isinstance(expr, BinaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            left_expr = getattr(expr, "left", None)
            right_expr = getattr(expr, "right", None)
            left = evaluate_index_with_alternatives(
                left_expr, constants, index_hints, alternative_bindings, call_stack
            )
            right = evaluate_index_with_alternatives(
                right_expr, constants, index_hints, alternative_bindings, call_stack
            )
            right_exact = literal_int_value(right_expr, constants)
            right_alternative_exact = right is not None and bool(
                vector_alternative_names_for_expr(right_expr, index_hints)
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
            call_name = function_call_name(expr)
            call_index = call_return_index_hint_with_alternatives(
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
                evaluate_index_with_alternatives(
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
                max_value = literal_int_value(args[2], constants)
                return max_value if max_value is not None and max_value >= 0 else None

        if "TernaryOp" in expr.__class__.__name__:
            branch_indexes = []
            for attr in ("true_expr", "false_expr"):
                branch_index = evaluate_index_with_alternatives(
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

    def correlated_resource_index_hint(expr, constants, index_hints, call_stack):
        names = vector_alternative_names_for_expr(expr, index_hints)
        if not names:
            return None
        bindings = alternative_bindings_for_names(names, index_hints)
        if not bindings:
            return None

        indexes = []
        for binding in bindings:
            index = evaluate_index_with_alternatives(
                expr, constants, index_hints, binding, call_stack
            )
            if index is None or index < 0:
                return None
            indexes.append(index)
        return max(indexes) if indexes else None

    def call_index_context_with_alternatives(
        call,
        callee,
        constants,
        index_hints,
        alternative_bindings,
        call_stack,
    ):
        call_constants = function_initial_constants(callee)
        call_index_hints = {}
        bound_index_hints = index_hints_with_alternative_bindings(
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
            exact_value = literal_int_value(arg, constants)
            if exact_value is not None:
                call_constants[param_name] = exact_value
                call_index_hints[param_name] = exact_value
            else:
                alternative_names = vector_alternative_names_for_expr(arg, index_hints)
                arg_hint = evaluate_index_with_alternatives(
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

            record_vector_component_index_hints(
                param_name,
                arg,
                constants,
                call_index_hints,
                call_stack,
                source_index_hints=bound_index_hints,
            )

        return call_constants, call_index_hints

    def call_return_index_hint_with_alternatives(
        call, constants, index_hints, alternative_bindings, call_stack
    ):
        call_stack = call_stack or set()
        call_name = function_call_name(call)
        callee = functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return None

        call_constants, call_index_hints = call_index_context_with_alternatives(
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
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            return_hint = resource_index_hint(
                getattr(node, "value", None),
                visible_constants,
                visible_index_hints,
                next_call_stack,
            )
            if return_hint is not None and return_hint >= 0:
                return_hints.append(return_hint)
        return max(return_hints) if return_hints else None

    def component_assignment_alternative_value(
        operator,
        current_value,
        assigned_value,
        constants,
        index_hints,
        alternative_bindings,
        call_stack,
    ):
        if operator == "=":
            return evaluate_index_with_alternatives(
                assigned_value,
                constants,
                index_hints,
                alternative_bindings,
                call_stack,
            )

        if current_value is None or current_value < 0:
            return None

        rhs = evaluate_index_with_alternatives(
            assigned_value,
            constants,
            index_hints,
            alternative_bindings,
            call_stack,
        )
        rhs_exact = literal_int_value(assigned_value, constants)
        result = None
        if operator == "+=" and rhs is not None:
            result = current_value + rhs
        elif operator == "-=" and rhs_exact is not None:
            result = current_value - rhs_exact
        elif operator == "*=" and rhs is not None and current_value >= 0 and rhs >= 0:
            result = current_value * rhs

        return result if result is not None and result >= 0 else None

    def updated_component_assignment_alternatives(
        name, component, operator, assigned_value, constants, index_hints, call_stack
    ):
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return ()

        alternatives = existing_vector_component_alternatives(name, index_hints)
        if not alternatives:
            return ()

        updated = []
        for alternative in alternatives:
            current_value = alternative[component_index]
            value = component_assignment_alternative_value(
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

        return normalize_vector_alternatives(updated)

    def vector_component_alternatives(expr, constants, index_hints, call_stack=None):
        call_stack = call_stack or set()
        if expr is None:
            return ()

        name = expression_name(expr)
        if name:
            alternatives = existing_vector_component_alternatives(name, index_hints)
            if alternatives:
                return alternatives
            component_hints = existing_vector_component_hints(name, index_hints)
            if component_hints:
                return vector_alternatives_from_component_hints(component_hints)

        alternatives = vector_swizzle_alternatives(expr, index_hints)
        if alternatives:
            return alternatives

        if "TernaryOp" in expr.__class__.__name__:
            alternatives = []
            for attr in ("true_expr", "false_expr"):
                alternatives.extend(
                    vector_component_alternatives(
                        getattr(expr, attr, None), constants, index_hints, call_stack
                    )
                )
            return normalize_vector_alternatives(alternatives)

        class_name = expr.__class__.__name__
        if "FunctionCall" in class_name:
            constructor = getattr(expr, "function", getattr(expr, "name", None))
            args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        elif "Constructor" in class_name:
            constructor = getattr(expr, "constructor_type", None)
            args = list(getattr(expr, "arguments", []) or [])
        else:
            return ()

        size = integer_vector_size(type_name(constructor))
        if size is None:
            if "FunctionCall" in class_name:
                return call_vector_component_alternatives(
                    expr, constants, index_hints, call_stack
                )
            return ()

        if len(args) == 1:
            copied_alternatives = vector_component_alternatives(
                args[0], constants, index_hints, call_stack
            )
            if copied_alternatives:
                alternatives = []
                for alternative in copied_alternatives:
                    values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
                    for index in range(size):
                        values[index] = alternative[index]
                    alternatives.append(values)
                return normalize_vector_alternatives(alternatives)

            names = vector_alternative_names_for_expr(args[0], index_hints)
            bindings = (
                alternative_bindings_for_names(names, index_hints) if names else [{}]
            )
            if not bindings:
                return ()
            alternatives = []
            for binding in bindings:
                value = evaluate_index_with_alternatives(
                    args[0], constants, index_hints, binding, call_stack
                )
                if value is None or value < 0:
                    return ()
                values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
                for index in range(size):
                    values[index] = value
                alternatives.append(values)
            return normalize_vector_alternatives(alternatives)

        names = set()
        for arg in args[:size]:
            names.update(vector_alternative_names_for_expr(arg, index_hints))
        bindings = alternative_bindings_for_names(names, index_hints) if names else [{}]
        if not bindings:
            return ()

        alternatives = []
        for binding in bindings:
            values = [None] * len(COMPONENT_ALIASES_BY_INDEX)
            for index, arg in enumerate(args[:size]):
                value = evaluate_index_with_alternatives(
                    arg, constants, index_hints, binding, call_stack
                )
                if value is not None and value >= 0:
                    values[index] = value
                else:
                    fallback = resource_index_hint(
                        arg, constants, index_hints, call_stack
                    )
                    if fallback is not None and fallback >= 0:
                        values[index] = fallback
            alternatives.append(values)
        return normalize_vector_alternatives(alternatives)

    def vector_component_index_hints(expr, constants, index_hints, call_stack=None):
        call_stack = call_stack or set()
        if expr is None:
            return {}

        name = expression_name(expr)
        if name:
            hints = existing_vector_component_hints(name, index_hints)
            if hints:
                return hints

        if "TernaryOp" in expr.__class__.__name__:
            branch_hints = [
                vector_component_index_hints(
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

        size = integer_vector_size(type_name(constructor))
        if size is None:
            if "FunctionCall" in class_name:
                return call_vector_component_hints(
                    expr, constants, index_hints, call_stack
                )
            return {}

        hints = {}
        if len(args) == 1:
            value = resource_index_hint(args[0], constants, index_hints, call_stack)
            if value is not None and value >= 0:
                for index in range(size):
                    hints[index] = value
            return hints

        for index, arg in enumerate(args[:size]):
            value = resource_index_hint(arg, constants, index_hints, call_stack)
            if value is not None and value >= 0:
                hints[index] = value
        return hints

    def record_vector_component_index_hints(
        name,
        expr,
        constants,
        index_hints,
        call_stack=None,
        source_index_hints=None,
    ):
        source_hints = source_index_hints or index_hints
        component_hints = vector_component_index_hints(
            expr, constants, source_hints, call_stack
        )
        for index, value in component_hints.items():
            for alias in COMPONENT_ALIASES_BY_INDEX[index]:
                index_hints[component_hint_key(name, alias)] = value
        alternatives = vector_component_alternatives(
            expr, constants, source_hints, call_stack
        )
        if alternatives:
            index_hints[vector_alternatives_key(name)] = alternatives

    def record_component_index_hint(name, component, value, index_hints):
        if value is None or value < 0:
            return
        component_index = COMPONENT_INDEX_BY_NAME.get(component)
        if component_index is None:
            return
        for alias in COMPONENT_ALIASES_BY_INDEX[component_index]:
            index_hints[component_hint_key(name, alias)] = value

    def compound_assignment_index_hint(
        operator, current_value, assigned_value, constants, index_hints, call_stack=None
    ):
        if current_value is None or current_value < 0:
            return None

        rhs = resource_index_hint(assigned_value, constants, index_hints, call_stack)
        rhs_exact = literal_int_value(assigned_value, constants)
        result = None
        if operator == "+=" and rhs is not None:
            result = current_value + rhs
        elif operator == "-=" and rhs_exact is not None:
            result = current_value - rhs_exact
        elif operator == "*=" and rhs is not None and rhs >= 0:
            result = current_value * rhs

        return result if result is not None and result >= 0 else None

    def known_index_value(expr, constants, index_hints):
        value = literal_int_value(expr, constants)
        if value is not None:
            return value

        name = expression_name(expr)
        if name in index_hints:
            return index_hints[name]
        return None

    def assignment_target_name(node):
        target = getattr(node, "target", getattr(node, "left", None))
        return expression_name(target)

    def simple_assignment_step(node, name, constants, index_hints):
        if not isinstance(node, AssignmentNode) or assignment_target_name(node) != name:
            return None

        operator = getattr(node, "operator", None)
        assigned_value = getattr(node, "value", getattr(node, "right", None))
        if operator in {"+=", "-="}:
            step = literal_int_value(assigned_value, constants)
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
        left_name = expression_name(left_expr)
        right_name = expression_name(right_expr)

        if left_name == name:
            step = literal_int_value(right_expr, constants)
            if step is None:
                return None
            if binary_operator == "+":
                return step
            if binary_operator == "-":
                return -step
        if right_name == name and binary_operator == "+":
            return literal_int_value(left_expr, constants)
        return None

    def simple_for_iteration_count(node, constants, index_hints):
        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None) or assignment_target_name(init)
        if not loop_name:
            return None

        start = known_index_value(init, constants, index_hints)
        if start is None:
            start = known_index_value(
                getattr(init, "initial_value", None), constants, index_hints
            )
        if start is None:
            start = known_index_value(
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
        left_name = expression_name(left_expr)
        right_name = expression_name(right_expr)
        if left_name == loop_name:
            bound = literal_int_value(right_expr, constants)
        elif right_name == loop_name:
            bound = literal_int_value(left_expr, constants)
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator)
        else:
            return None
        if bound is None:
            return None

        step = simple_assignment_step(
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

    def simple_for_in_range(node, constants):
        iterable = getattr(node, "iterable", None)
        if iterable is None or "Range" not in iterable.__class__.__name__:
            return None

        start = literal_int_value(getattr(iterable, "start", None), constants)
        end = literal_int_value(getattr(iterable, "end", None), constants)
        if start is None or end is None:
            return None

        count = end - start
        if getattr(iterable, "inclusive", False):
            count += 1
        if count < 0 or count > MAX_STATIC_LOOP_ITERATIONS_FOR_HINTS:
            return None
        return start, count

    def subtree_contains_control_exit(value, seen=None):
        seen = seen or set()
        if value is None or isinstance(value, (str, int, float, bool)):
            return False
        if isinstance(value, dict):
            return any(
                subtree_contains_control_exit(item, seen) for item in value.values()
            )
        if isinstance(value, (list, tuple, set)):
            return any(subtree_contains_control_exit(item, seen) for item in value)

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
            subtree_contains_control_exit(child, seen)
            for key, child in vars(value).items()
            if key not in {"parent", "annotations"}
        )

    def compound_assignment_targets(value):
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
                target_component = expression_component_parts(target)
                if target_component is not None:
                    targets.append(target_component)
                else:
                    target_name = expression_name(target)
                    if target_name:
                        targets.append((target_name, None))

            if hasattr(item, "__dict__"):
                for key, child in vars(item).items():
                    if key in {"parent", "annotations"}:
                        continue
                    visit(child)

        visit(value)
        return targets

    def assigned_hint_base_names(*values):
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
                target_component = expression_component_parts(target)
                if target_component is not None:
                    names.add(target_component[0])
                else:
                    target_name = expression_name(target)
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

    def declared_variable_names(*values):
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

    def remove_compound_loop_targets(index_hints, targets):
        for name, component in targets:
            if component is None:
                remove_index_hints_for_name(index_hints, name)
            else:
                remove_component_index_hints(index_hints, name, component)

    def resource_index_hint(expr, constants, index_hints, call_stack=None):
        """Return a conservative known lower-bound index for resource access."""
        call_stack = call_stack or set()
        index = literal_int_value(expr, constants)
        if index is not None:
            return index

        if expr is None:
            return None

        name = expression_name(expr)
        if name in index_hints:
            return index_hints[name]

        component_key = expression_component_hint_key(expr)
        if component_key in index_hints:
            return index_hints[component_key]

        correlated_index = correlated_resource_index_hint(
            expr, constants, index_hints, call_stack
        )
        if correlated_index is not None:
            return correlated_index

        if isinstance(expr, FunctionCallNode):
            call_name = function_call_name(expr)
            call_index = call_return_index_hint(
                expr, constants, index_hints, call_stack
            )
            if call_index is not None:
                return call_index
            if call_name in return_index_hints:
                return return_index_hints[call_name]
            builtin_index = builtin_resource_index_hint(
                expr, call_name, constants, index_hints, call_stack
            )
            if builtin_index is not None:
                return builtin_index

        if isinstance(expr, BinaryOpNode):
            return binary_resource_index_hint(expr, constants, index_hints, call_stack)

        if "TernaryOp" not in expr.__class__.__name__:
            return None

        branch_indexes = []
        for attr in ("true_expr", "false_expr"):
            branch_index = resource_index_hint(
                getattr(expr, attr, None), constants, index_hints, call_stack
            )
            if branch_index is not None and branch_index >= 0:
                branch_indexes.append(branch_index)
        return max(branch_indexes) if branch_indexes else None

    def binary_resource_index_hint(expr, constants, index_hints, call_stack):
        """Infer a lower-bound index through monotonic integer arithmetic."""
        operator = getattr(expr, "operator", getattr(expr, "op", None))
        left_expr = getattr(expr, "left", None)
        right_expr = getattr(expr, "right", None)
        left = resource_index_hint(left_expr, constants, index_hints, call_stack)
        right = resource_index_hint(right_expr, constants, index_hints, call_stack)
        right_exact = literal_int_value(right_expr, constants)

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
        call, call_name, constants, index_hints, call_stack
    ):
        """Infer bounds from common integer built-ins used for array indices."""
        args = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        if call_name == "min" and len(args) >= 2:
            exact_bounds = [
                value
                for value in (literal_int_value(arg, constants) for arg in args)
                if value is not None and value >= 0
            ]
            return min(exact_bounds) if exact_bounds else None

        if call_name == "max" and len(args) >= 2:
            lower_bounds = [
                value
                for value in (
                    resource_index_hint(arg, constants, index_hints, call_stack)
                    for arg in args
                )
                if value is not None and value >= 0
            ]
            return max(lower_bounds) if lower_bounds else None

        if call_name == "clamp" and len(args) >= 3:
            max_value = literal_int_value(args[2], constants)
            return max_value if max_value is not None and max_value >= 0 else None

        return None

    def call_index_context(call, callee, constants, index_hints, call_stack):
        call_constants = function_initial_constants(callee)
        call_index_hints = {}
        args = getattr(call, "arguments", getattr(call, "args", []))
        for index, param in enumerate(getattr(callee, "parameters", []) or []):
            if index >= len(args):
                break
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            arg = args[index]
            exact_value = literal_int_value(arg, constants)
            if exact_value is not None:
                call_constants[param_name] = exact_value
                call_index_hints[param_name] = exact_value
                continue
            arg_hint = resource_index_hint(arg, constants, index_hints, call_stack)
            if arg_hint is not None and arg_hint >= 0:
                call_index_hints[param_name] = arg_hint
            record_vector_component_index_hints(
                param_name,
                arg,
                constants,
                call_index_hints,
                call_stack,
                source_index_hints=index_hints,
            )
        return call_constants, call_index_hints

    def call_return_index_hint(call, constants, index_hints, call_stack):
        call_name = function_call_name(call)
        callee = functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return None

        call_constants, call_index_hints = call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        return_hints = []
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            return_hint = resource_index_hint(
                getattr(node, "value", None),
                visible_constants,
                visible_index_hints,
                next_call_stack,
            )
            if return_hint is not None and return_hint >= 0:
                return_hints.append(return_hint)
        return max(return_hints) if return_hints else None

    def call_vector_component_hints(call, constants, index_hints, call_stack):
        call_name = function_call_name(call)
        callee = functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return {}

        call_constants, call_index_hints = call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        component_values = {}
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            component_hints = vector_component_index_hints(
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

    def call_vector_component_alternatives(call, constants, index_hints, call_stack):
        call_name = function_call_name(call)
        callee = functions_by_name.get(call_name)
        if callee is None or call_name in call_stack:
            return ()

        call_constants, call_index_hints = call_index_context(
            call, callee, constants, index_hints, call_stack
        )
        next_call_stack = set(call_stack)
        next_call_stack.add(call_name)
        alternatives = []
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(callee, "body", []), call_constants, call_index_hints
        ):
            if not isinstance(node, ReturnNode):
                continue
            alternatives.extend(
                vector_component_alternatives(
                    getattr(node, "value", None),
                    visible_constants,
                    visible_index_hints,
                    next_call_stack,
                )
            )
        return normalize_vector_alternatives(alternatives)

    def merge_existing_index_hints(target, source, names):
        """Propagate updates to known outer index aliases through a nested block."""
        for name in names:
            if (
                isinstance(name, str)
                and "." not in name
                and not is_vector_alternatives_key(name)
            ):
                keys = [
                    key
                    for key in set(target).union(source)
                    if hint_key_base_name(key) == name
                ]
            else:
                keys = [name]
            for key in keys:
                if key in source:
                    target[key] = source[key]
                else:
                    target.pop(key, None)

    def merge_branch_index_hints(target, *branches):
        """Merge possible branch-local lower-bound index aliases."""
        names = set(target)
        for branch in branches:
            names.update(branch)

        for name in names:
            if is_vector_alternatives_key(name):
                continue

            branch_values = [
                branch[name]
                for branch in branches
                if name in branch and is_nonnegative_int_hint(branch[name])
            ]
            if branch_values:
                target[name] = max(branch_values)
            else:
                target.pop(name, None)
        merge_branch_vector_alternative_hints(target, *branches)

    def merge_loop_index_hints(target, *branches, compound_targets=None):
        """Merge loop-carried updates for aliases already visible after the loop."""
        for name in list(target):
            if is_vector_alternatives_key(name):
                alternatives = []
                alternatives.extend(normalize_vector_alternatives(target.get(name)))
                for branch in branches:
                    alternatives.extend(normalize_vector_alternatives(branch.get(name)))
                alternatives = normalize_vector_alternatives(alternatives)
                if alternatives:
                    target[name] = alternatives
                else:
                    target.pop(name, None)
                continue

            values = [
                branch[name]
                for branch in branches
                if name in branch and is_nonnegative_int_hint(branch[name])
            ]
            if is_nonnegative_int_hint(target[name]):
                values.append(target[name])
            if values:
                target[name] = max(values)
            else:
                target.pop(name, None)
        remove_compound_loop_targets(target, compound_targets or ())

    def is_switch_default_case(case):
        return getattr(case, "value", None) is None

    def is_match_default_arm(arm):
        pattern = getattr(arm, "pattern", None)
        return pattern is None or "Wildcard" in pattern.__class__.__name__

    def walk_nodes_with_constants(root, constants, initial_index_hints=None):
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
                    assigned_hint_base_names(getattr(value, "statements", []))
                )
                merge_existing_index_hints(
                    active_index_hints,
                    block_index_hints,
                    outer_hint_names_without_locals(propagated_names, local_names),
                )
                return

            if isinstance(value, VariableNode):
                name = getattr(value, "name", None)
                if name:
                    active_constants.pop(name, None)
                    remove_index_hints_for_name(active_index_hints, name)
                yield value, active_constants, active_index_hints
                for key, child in vars(value).items():
                    if key in {"parent", "annotations", "name"}:
                        continue
                    yield from walk(child, active_constants, active_index_hints)
                initial_value = getattr(value, "initial_value", None)
                index_hint = resource_index_hint(
                    initial_value, active_constants, active_index_hints
                )
                if name and index_hint is not None and index_hint >= 0:
                    active_index_hints[name] = index_hint
                if name:
                    record_vector_component_index_hints(
                        name,
                        initial_value,
                        active_constants,
                        active_index_hints,
                    )
                if name and "const" in getattr(value, "qualifiers", []):
                    const_value = literal_int_value(initial_value, active_constants)
                    if const_value is not None:
                        active_constants[name] = const_value
                return

            if isinstance(value, AssignmentNode):
                yield value, active_constants, active_index_hints
                target = getattr(value, "target", getattr(value, "left", None))
                assigned_value = getattr(value, "value", getattr(value, "right", None))
                yield from walk(target, active_constants, active_index_hints)
                yield from walk(assigned_value, active_constants, active_index_hints)

                target_component = expression_component_parts(target)
                if target_component is not None:
                    target_name, component = target_component
                    operator = getattr(value, "operator", None)
                    current_hint = resource_index_hint(
                        target, active_constants, active_index_hints
                    )
                    assignment_hint = None
                    compound_hint = None
                    if operator == "=":
                        assignment_hint = resource_index_hint(
                            assigned_value, active_constants, active_index_hints
                        )
                    else:
                        compound_hint = compound_assignment_index_hint(
                            operator,
                            current_hint,
                            assigned_value,
                            active_constants,
                            active_index_hints,
                        )
                    updated_alternatives = updated_component_assignment_alternatives(
                        target_name,
                        component,
                        operator,
                        assigned_value,
                        active_constants,
                        active_index_hints,
                        None,
                    )
                    remove_component_index_hints(
                        active_index_hints, target_name, component
                    )
                    if operator == "=":
                        record_component_index_hint(
                            target_name,
                            component,
                            assignment_hint,
                            active_index_hints,
                        )
                    else:
                        record_component_index_hint(
                            target_name, component, compound_hint, active_index_hints
                        )
                    if updated_alternatives:
                        active_index_hints[vector_alternatives_key(target_name)] = (
                            updated_alternatives
                        )
                    return

                target_name = expression_name(target)
                if target_name:
                    operator = getattr(value, "operator", None)
                    current_hint = resource_index_hint(
                        target, active_constants, active_index_hints
                    )
                    source_index_hints = dict(active_index_hints)
                    assignment_hint = None
                    compound_hint = None
                    if operator == "=":
                        assignment_hint = resource_index_hint(
                            assigned_value, active_constants, active_index_hints
                        )
                    else:
                        compound_hint = compound_assignment_index_hint(
                            operator,
                            current_hint,
                            assigned_value,
                            active_constants,
                            active_index_hints,
                        )
                    active_constants.pop(target_name, None)
                    remove_index_hints_for_name(active_index_hints, target_name)
                    if operator == "=":
                        if assignment_hint is not None and assignment_hint >= 0:
                            active_index_hints[target_name] = assignment_hint
                        record_vector_component_index_hints(
                            target_name,
                            assigned_value,
                            active_constants,
                            active_index_hints,
                            source_index_hints=source_index_hints,
                        )
                    else:
                        if compound_hint is not None and compound_hint >= 0:
                            active_index_hints[target_name] = compound_hint
                return

            if isinstance(value, IfNode):
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
                merge_branch_index_hints(
                    active_index_hints, then_index_hints, else_index_hints
                )
                return

            if isinstance(value, ForNode):
                yield value, active_constants, active_index_hints
                outer_index_hint_names = set(active_index_hints)
                loop_constants = dict(active_constants)
                loop_index_hints = dict(active_index_hints)
                yield from walk(
                    getattr(value, "init", None), loop_constants, loop_index_hints
                )
                yield from walk(
                    getattr(value, "condition", None),
                    loop_constants,
                    loop_index_hints,
                )
                body = getattr(value, "body", None)
                update = getattr(value, "update", None)
                iteration_count = None
                if not subtree_contains_control_exit(body):
                    iteration_count = simple_for_iteration_count(
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
                        propagated_names.update(assigned_hint_base_names(body, update))
                        propagated_names.difference_update(
                            declared_variable_names(getattr(value, "init", None))
                        )
                    merge_existing_index_hints(
                        active_index_hints,
                        final_index_hints,
                        propagated_names,
                    )
                    return
                zero_iteration_hints = dict(loop_index_hints)
                body_index_hints = dict(loop_index_hints)
                yield from walk(
                    body,
                    dict(loop_constants),
                    body_index_hints,
                )
                yield from walk(update, loop_constants, body_index_hints)
                merge_loop_index_hints(
                    active_index_hints,
                    zero_iteration_hints,
                    body_index_hints,
                    compound_targets=(
                        compound_assignment_targets(body)
                        + compound_assignment_targets(update)
                    ),
                )
                return

            if isinstance(value, ForInNode):
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
                if not subtree_contains_control_exit(body):
                    loop_range = simple_for_in_range(value, loop_constants)
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
                        propagated_names.update(assigned_hint_base_names(body))
                        if pattern:
                            propagated_names.discard(pattern)
                    merge_existing_index_hints(
                        active_index_hints,
                        final_index_hints,
                        propagated_names,
                    )
                    return
                body_index_hints = dict(loop_index_hints)
                yield from walk(body, loop_constants, body_index_hints)
                merge_loop_index_hints(
                    active_index_hints,
                    loop_index_hints,
                    body_index_hints,
                    compound_targets=compound_assignment_targets(body),
                )
                return

            if isinstance(value, WhileNode):
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
                merge_loop_index_hints(
                    active_index_hints,
                    condition_index_hints,
                    body_index_hints,
                    compound_targets=compound_assignment_targets(body),
                )
                return

            if isinstance(value, DoWhileNode):
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
                merge_loop_index_hints(
                    active_index_hints,
                    body_index_hints,
                    condition_index_hints,
                    compound_targets=compound_assignment_targets(body),
                )
                return

            if isinstance(value, LoopNode):
                yield value, active_constants, active_index_hints
                body = getattr(value, "body", None)
                body_index_hints = dict(active_index_hints)
                yield from walk(
                    body,
                    dict(active_constants),
                    body_index_hints,
                )
                merge_loop_index_hints(
                    active_index_hints,
                    body_index_hints,
                    compound_targets=compound_assignment_targets(body),
                )
                return

            if isinstance(value, SwitchNode):
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
                    has_default = has_default or is_switch_default_case(case)
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
                merge_branch_index_hints(active_index_hints, *branch_index_hints)
                return

            if isinstance(value, CaseNode):
                case_constants = dict(active_constants)
                yield value, case_constants, active_index_hints
                yield from walk(
                    getattr(value, "value", None),
                    active_constants,
                    active_index_hints,
                )
                for statement in getattr(value, "statements", []) or []:
                    yield from walk(statement, case_constants, active_index_hints)
                return

            if isinstance(value, MatchNode):
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
                    has_default = has_default or is_match_default_arm(arm)
                if not has_default:
                    branch_index_hints.append(dict(active_index_hints))
                merge_branch_index_hints(active_index_hints, *branch_index_hints)
                return

            if isinstance(value, MatchArmNode):
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
                yield from walk(
                    getattr(value, "body", None), arm_constants, active_index_hints
                )
                return

            yield value, active_constants, active_index_hints
            if hasattr(value, "__dict__"):
                for key, child in vars(value).items():
                    if key in {"parent", "annotations"}:
                        continue
                    yield from walk(child, active_constants, active_index_hints)

        yield from walk(root, constants, dict(initial_index_hints or {}))

    changed = True
    while changed:
        changed = False
        for func_name, func in functions_by_name.items():
            initial_constants = function_initial_constants(func)
            return_hints = []
            for (
                node,
                visible_constants,
                visible_index_hints,
            ) in walk_nodes_with_constants(
                getattr(func, "body", []), initial_constants
            ):
                if not isinstance(node, ReturnNode):
                    continue
                index = resource_index_hint(
                    getattr(node, "value", None),
                    visible_constants,
                    visible_index_hints,
                )
                if index is not None and index >= 0:
                    return_hints.append(index)
            if not return_hints:
                continue
            return_index = max(return_hints)
            if return_index_hints.get(func_name) != return_index:
                return_index_hints[func_name] = return_index
                changed = True

    def record_array_required_size(func_name, array_name, required_size):
        changed = False
        if array_name in global_hints:
            new_size = max(global_hints[array_name], required_size)
            if new_size != global_hints[array_name]:
                global_hints[array_name] = new_size
                changed = True
        if array_name in fixed_global_array_sizes:
            assert_not_larger_than_known_fixed(
                array_name, fixed_global_array_sizes[array_name], required_size
            )
        if array_name in function_hints.get(func_name, {}):
            new_size = max(function_hints[func_name][array_name], required_size)
            if new_size != function_hints[func_name][array_name]:
                function_hints[func_name][array_name] = new_size
                changed = True
        fixed_params = fixed_function_array_sizes.get(func_name, {})
        if array_name in fixed_params:
            assert_not_larger_than_known_fixed(
                array_name, fixed_params[array_name], required_size
            )
        return changed

    def scan_call_site_resource_requirements(
        func_name, func, constants, index_hints, call_stack
    ):
        """Apply scalar call-site hints to resource-array accesses in callees."""
        changed = False
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(func, "body", []), constants, index_hints
        ):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                array_name = expression_name(array_expr)
                index = resource_index_hint(
                    index_expr, visible_constants, visible_index_hints, call_stack
                )
                if array_name is not None and index is not None and index >= 0:
                    changed = (
                        record_array_required_size(func_name, array_name, index + 1)
                        or changed
                    )
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            callee_name = function_call_name(node)
            callee = functions_by_name.get(callee_name)
            if callee is None or callee_name in call_stack:
                continue
            call_constants, call_index_hints = call_index_context(
                node, callee, visible_constants, visible_index_hints, call_stack
            )
            next_call_stack = set(call_stack)
            next_call_stack.add(callee_name)
            changed = (
                scan_call_site_resource_requirements(
                    callee_name,
                    callee,
                    call_constants,
                    call_index_hints,
                    next_call_stack,
                )
                or changed
            )
        return changed

    for func_name, func in functions_by_name.items():
        initial_constants = function_initial_constants(func)
        for node, visible_constants, visible_index_hints in walk_nodes_with_constants(
            getattr(func, "body", []), initial_constants
        ):
            if not isinstance(node, ArrayAccessNode):
                continue
            array_expr = getattr(node, "array", getattr(node, "array_expr", None))
            index_expr = getattr(node, "index", getattr(node, "index_expr", None))
            array_name = expression_name(array_expr)
            index = resource_index_hint(
                index_expr, visible_constants, visible_index_hints
            )
            if array_name is None or index is None or index < 0:
                continue
            required_size = index + 1
            record_array_required_size(func_name, array_name, required_size)

    for func_name, func in functions_by_name.items():
        initial_constants = function_initial_constants(func)
        scan_call_site_resource_requirements(
            func_name, func, initial_constants, {}, {func_name}
        )

    def register_fixed_requirement(scope_key, size, current_size):
        """Record a fixed size requirement for a propagated array argument."""
        if size is None:
            return
        existing = fixed_requirements.get(scope_key)
        if existing is not None and existing != size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {existing} and {size}"
            )
        if current_size and current_size > size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {current_size} and {size}"
            )
        fixed_requirements[scope_key] = size

    def assert_not_larger_than_fixed(scope_key, required_size):
        """Raise when propagated usage exceeds a fixed array requirement."""
        fixed_size = fixed_requirements.get(scope_key)
        if fixed_size is not None and required_size > fixed_size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {fixed_size} and {required_size}"
            )

    changed = True
    while changed:
        changed = False
        for caller_name, func in functions_by_name.items():
            caller_param_hints = function_hints.get(caller_name, {})
            caller_fixed_hints = fixed_function_array_sizes.get(caller_name, {})
            for call in walk_nodes(getattr(func, "body", [])) or ():
                if not isinstance(call, FunctionCallNode):
                    continue
                callee_name = function_call_name(call)
                callee_param_hints = function_hints.get(callee_name, {})
                callee_fixed_hints = fixed_function_array_sizes.get(callee_name, {})
                if not callee_param_hints and not callee_fixed_hints:
                    continue
                callee = functions_by_name.get(callee_name)
                if callee is None:
                    continue
                callee_params = getattr(callee, "parameters", [])
                args = getattr(call, "arguments", getattr(call, "args", []))
                for index, arg in enumerate(args):
                    if index >= len(callee_params):
                        continue
                    callee_param_name = getattr(callee_params[index], "name", None)
                    required_size = (
                        callee_param_hints.get(callee_param_name)
                        if callee_param_name in callee_param_hints
                        else callee_fixed_hints.get(callee_param_name)
                    )
                    fixed_size = callee_fixed_hints.get(callee_param_name)
                    arg_name = expression_name(arg)
                    arg_size = None
                    arg_is_fixed = False
                    arg_scope_key = None
                    if arg_name in global_hints:
                        arg_scope_key = ("global", arg_name)
                        arg_size = global_hints[arg_name]
                    elif arg_name in fixed_global_array_sizes:
                        arg_size = fixed_global_array_sizes[arg_name]
                        arg_is_fixed = True
                        if fixed_size is not None and arg_size != fixed_size:
                            raise ValueError(
                                "Conflicting fixed resource array sizes for "
                                f"'{arg_name}': {arg_size} and {fixed_size}"
                            )
                    elif arg_name in caller_param_hints:
                        arg_scope_key = ("param", caller_name, arg_name)
                        arg_size = caller_param_hints[arg_name]
                    elif arg_name in caller_fixed_hints:
                        arg_size = caller_fixed_hints[arg_name]
                        arg_is_fixed = True
                        if fixed_size is not None and arg_size != fixed_size:
                            raise ValueError(
                                "Conflicting fixed resource array sizes for "
                                f"'{arg_name}': {arg_size} and {fixed_size}"
                            )
                    if arg_scope_key is not None:
                        register_fixed_requirement(arg_scope_key, fixed_size, arg_size)
                    if (
                        callee_param_name in callee_param_hints
                        and arg_size
                        and arg_size > callee_param_hints[callee_param_name]
                    ):
                        callee_param_hints[callee_param_name] = arg_size
                        changed = True
                    if not required_size:
                        continue
                    if (
                        arg_name in global_hints
                        and required_size > global_hints[arg_name]
                    ):
                        assert_not_larger_than_fixed(
                            ("global", arg_name), required_size
                        )
                        global_hints[arg_name] = required_size
                        changed = True
                    if (
                        arg_name in caller_param_hints
                        and required_size > caller_param_hints[arg_name]
                    ):
                        assert_not_larger_than_fixed(
                            ("param", caller_name, arg_name), required_size
                        )
                        caller_param_hints[arg_name] = required_size
                        changed = True
                    if arg_is_fixed and arg_size and required_size > arg_size:
                        raise ValueError(
                            "Conflicting fixed resource array sizes for "
                            f"'{arg_name}': {arg_size} and {required_size}"
                        )

    return (
        {name: format_size(size) for name, size in global_hints.items()},
        {
            func_name: {
                param_name: format_size(size)
                for param_name, size in param_hints.items()
            }
            for func_name, param_hints in function_hints.items()
        },
    )
