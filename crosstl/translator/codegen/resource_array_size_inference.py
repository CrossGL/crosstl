"""Resource-array size inference from literal accesses and call propagation."""

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import TargetRefusal
from .resource_array_constant_walk import ResourceArrayConstantWalkMixin
from .resource_array_index_alternatives import ResourceArrayIndexAlternativesMixin
from .resource_array_index_flow import ResourceArrayIndexFlowMixin


class ResourceArraySizeInference(
    ResourceArrayIndexAlternativesMixin,
    ResourceArrayIndexFlowMixin,
    ResourceArrayConstantWalkMixin,
):
    """One inference run of :func:`.resource_arrays.collect_resource_array_size_hints`.

    The keyword arguments are that function's parameters. Values the nested scopes shared are attributes.
    """

    def __init__(
        self,
        *,
        global_arrays,
        function_arrays,
        fixed_global_array_sizes,
        fixed_function_array_sizes,
        functions,
        walk_nodes,
        expression_name,
        literal_int_value,
        visible_literal_int_constants,
        function_call_name,
        initial_size,
        format_size,
        initial_literal_int_constants,
        strict_fixed_local_array_sizes,
        propagate_argument_sizes_to_callee,
        validate_fixed_argument_sizes,
        required_sizes,
    ):
        self.global_arrays = global_arrays
        self.function_arrays = function_arrays
        self.fixed_global_array_sizes = fixed_global_array_sizes
        self.fixed_function_array_sizes = fixed_function_array_sizes
        self.functions = functions
        self.walk_nodes = walk_nodes
        self.expression_name = expression_name
        self.literal_int_value = literal_int_value
        self.visible_literal_int_constants = visible_literal_int_constants
        self.function_call_name = function_call_name
        self.initial_size = initial_size
        self.format_size = format_size
        self.initial_literal_int_constants = initial_literal_int_constants
        self.strict_fixed_local_array_sizes = strict_fixed_local_array_sizes
        self.propagate_argument_sizes_to_callee = propagate_argument_sizes_to_callee
        self.validate_fixed_argument_sizes = validate_fixed_argument_sizes
        self.required_sizes = required_sizes

    def run(self):
        """Return ``(global_hints, function_hints)``; call once per instance."""
        self.resolved_accesses = {}
        self.approximate_loops = set()
        self.fixed_global_array_sizes = self.fixed_global_array_sizes or {}
        self.fixed_function_array_sizes = self.fixed_function_array_sizes or {}
        self.global_hints = {name: self.initial_size for name in self.global_arrays}
        self.function_hints = {
            func_name: {param_name: self.initial_size for param_name in params}
            for func_name, params in self.function_arrays.items()
        }
        self.fixed_requirements = {}
        if (
            not self.global_hints
            and not any(self.function_hints.values())
            and not self.fixed_global_array_sizes
            and not any(self.fixed_function_array_sizes.values())
        ):
            # No array to size or check: every scan below only reads these tables,
            # so they would stay empty. Skip walking every function body.
            return {}, {func_name: {} for func_name in self.function_hints}
        self.functions_by_name = {
            getattr(func, "name", None): func for func in self.functions
        }
        self.functions_by_name = {
            name: func for name, func in self.functions_by_name.items() if name
        }
        self.return_index_hints = {}
        self.collect_fixed_local_array_sizes()
        self.collect_return_index_hints()
        self.collect_access_requirements()
        self.propagate_call_argument_sizes()

        return (
            {name: self.format_size(size) for name, size in self.global_hints.items()},
            {
                func_name: {
                    param_name: self.format_size(size)
                    for param_name, size in param_hints.items()
                }
                for func_name, param_hints in self.function_hints.items()
            },
        )

    def collect_fixed_local_array_sizes(self):
        """Record each function's local arrays whose literal size is unambiguous."""
        self.fixed_local_array_sizes = {}
        for function_name, function in self.functions_by_name.items():
            local_sizes = {}
            ambiguous_names = set()
            initial_constants = self.function_initial_constants(function)
            for node, visible_constants, _index_hints in self.walk_nodes_with_constants(
                getattr(function, "body", []), initial_constants
            ):
                if not isinstance(node, VariableNode):
                    continue
                node_type = getattr(node, "var_type", getattr(node, "vtype", None))
                if "ArrayType" not in node_type.__class__.__name__:
                    continue
                name = getattr(node, "name", None)
                size_expression = getattr(node_type, "size", None)
                size = self.literal_int_value(size_expression, visible_constants)
                if not name or size is None or size <= 0:
                    if name:
                        ambiguous_names.add(name)
                    continue
                previous = local_sizes.get(name)
                if previous is not None and previous != size:
                    ambiguous_names.add(name)
                    continue
                local_sizes[name] = size
            for name in ambiguous_names:
                local_sizes.pop(name, None)
            self.fixed_local_array_sizes[function_name] = local_sizes

    def collect_return_index_hints(self):
        """Iterate the literal index each function returns to a fixed point."""
        self.fixed_local_parameter_sizes = {}
        changed = True
        while changed:
            changed = False
            for func_name, func in self.functions_by_name.items():
                initial_constants = self.function_initial_constants(func)
                return_hints = []
                for (
                    node,
                    visible_constants,
                    visible_index_hints,
                ) in self.walk_nodes_with_constants(
                    getattr(func, "body", []), initial_constants
                ):
                    if not isinstance(node, ReturnNode):
                        continue
                    index = self.resource_index_hint(
                        getattr(node, "value", None),
                        visible_constants,
                        visible_index_hints,
                    )
                    if index is not None and index >= 0:
                        return_hints.append(index)
                if not return_hints:
                    continue
                return_index = max(return_hints)
                if self.return_index_hints.get(func_name) != return_index:
                    self.return_index_hints[func_name] = return_index
                    changed = True

    def collect_access_requirements(self):
        """Record the sizes literal accesses, call sites and ``required_sizes`` need."""
        for func_name, func in self.functions_by_name.items():
            initial_constants = self.function_initial_constants(func)
            for (
                node,
                visible_constants,
                visible_index_hints,
            ) in self.walk_nodes_with_constants(
                getattr(func, "body", []), initial_constants
            ):
                if not isinstance(node, ArrayAccessNode):
                    continue
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                # Only an index into the declared array itself sizes it; ``a[i][j]``'s ``j`` indexes an element.
                array_name = (
                    None
                    if isinstance(array_expr, ArrayAccessNode)
                    else self.expression_name(array_expr)
                )
                index = self.resource_index_hint(
                    index_expr, visible_constants, visible_index_hints
                )
                self.resolved_accesses[id(node)] = self.resolved_accesses.get(
                    id(node), True
                ) and (index is not None and index >= 0)
                if array_name is None or index is None or index < 0:
                    continue
                required_size = index + 1
                self.record_array_required_size(func_name, array_name, required_size)

        for func_name, func in self.functions_by_name.items():
            initial_constants = self.function_initial_constants(func)
            self.scan_call_site_resource_requirements(
                func_name, func, initial_constants, {}, {func_name}
            )

        if callable(self.required_sizes):
            self.required_sizes = self.required_sizes(
                resolved=frozenset(
                    node_id
                    for node_id, resolved in self.resolved_accesses.items()
                    if resolved
                ),
                approximate_loops=frozenset(self.approximate_loops),
            )
        for array_name, required_size in sorted((self.required_sizes or {}).items()):
            self.record_array_required_size(None, array_name, required_size)

    def propagate_call_argument_sizes(self):
        """Propagate parameter sizes back to call arguments until nothing changes."""
        changed = True
        while changed:
            changed = False
            for caller_name, func in self.functions_by_name.items():
                caller_param_hints = self.function_hints.get(caller_name, {})
                caller_fixed_hints = self.fixed_function_array_sizes.get(
                    caller_name, {}
                )
                caller_local_fixed_hints = self.fixed_local_array_sizes.get(
                    caller_name, {}
                )
                for call in self.walk_nodes(getattr(func, "body", [])) or ():
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee_param_hints = self.function_hints.get(callee_name, {})
                    callee_fixed_hints = self.fixed_function_array_sizes.get(
                        callee_name, {}
                    )
                    if not callee_param_hints and not callee_fixed_hints:
                        continue
                    callee = self.functions_by_name.get(callee_name)
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
                        arg_name = self.expression_name(arg)
                        arg_size = None
                        arg_is_fixed = False
                        arg_scope_key = None
                        if arg_name in self.global_hints:
                            arg_scope_key = ("global", arg_name)
                            arg_size = self.global_hints[arg_name]
                        elif arg_name in self.fixed_global_array_sizes:
                            arg_size = self.fixed_global_array_sizes[arg_name]
                            arg_is_fixed = True
                            if fixed_size is not None and arg_size != fixed_size:
                                raise TargetRefusal(
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
                                raise TargetRefusal(
                                    "Conflicting fixed resource array sizes for "
                                    f"'{arg_name}': {arg_size} and {fixed_size}"
                                )
                        elif arg_name in caller_local_fixed_hints:
                            arg_size = caller_local_fixed_hints[arg_name]
                            arg_is_fixed = True
                            if fixed_size is not None and arg_size != fixed_size:
                                raise TargetRefusal(
                                    "Conflicting fixed resource array sizes for "
                                    f"'{arg_name}': {arg_size} and {fixed_size}"
                                )
                        if (
                            self.strict_fixed_local_array_sizes
                            and arg_name in caller_local_fixed_hints
                            and callee_param_name in callee_param_hints
                        ):
                            parameter_key = (callee_name, callee_param_name)
                            existing_size = self.fixed_local_parameter_sizes.get(
                                parameter_key
                            )
                            if existing_size is not None and existing_size != arg_size:
                                raise TargetRefusal(
                                    "Conflicting fixed local array sizes for private "
                                    f"pointer parameter '{callee_name}.{callee_param_name}': "
                                    f"{existing_size} and {arg_size}"
                                )
                            if callee_param_hints[callee_param_name] > arg_size:
                                raise TargetRefusal(
                                    "Private pointer parameter "
                                    f"'{callee_name}.{callee_param_name}' requires at least "
                                    f"{callee_param_hints[callee_param_name]} elements, but "
                                    f"caller array '{arg_name}' has {arg_size}"
                                )
                            self.fixed_local_parameter_sizes[parameter_key] = arg_size
                        if arg_scope_key is not None:
                            self.register_fixed_requirement(
                                arg_scope_key, fixed_size, arg_size
                            )
                        if (
                            self.propagate_argument_sizes_to_callee
                            and callee_param_name in callee_param_hints
                            and arg_size
                            and arg_size > callee_param_hints[callee_param_name]
                        ):
                            callee_param_hints[callee_param_name] = arg_size
                            changed = True
                        if not required_size:
                            continue
                        if (
                            arg_name in self.global_hints
                            and required_size > self.global_hints[arg_name]
                        ):
                            self.assert_not_larger_than_fixed(
                                ("global", arg_name), required_size
                            )
                            self.global_hints[arg_name] = required_size
                            changed = True
                        if (
                            arg_name in caller_param_hints
                            and required_size > caller_param_hints[arg_name]
                        ):
                            self.assert_not_larger_than_fixed(
                                ("param", caller_name, arg_name), required_size
                            )
                            caller_param_hints[arg_name] = required_size
                            changed = True
                        if (
                            self.validate_fixed_argument_sizes
                            and arg_is_fixed
                            and arg_size
                            and required_size > arg_size
                        ):
                            raise TargetRefusal(
                                "Conflicting fixed resource array sizes for "
                                f"'{arg_name}': {arg_size} and {required_size}"
                            )

    def record_array_required_size(self, func_name, array_name, required_size):
        changed = False
        if array_name in self.global_hints:
            new_size = max(self.global_hints[array_name], required_size)
            if new_size != self.global_hints[array_name]:
                self.global_hints[array_name] = new_size
                changed = True
        if array_name in self.fixed_global_array_sizes:
            self.assert_not_larger_than_known_fixed(
                array_name, self.fixed_global_array_sizes[array_name], required_size
            )
        if array_name in self.function_hints.get(func_name, {}):
            new_size = max(self.function_hints[func_name][array_name], required_size)
            if new_size != self.function_hints[func_name][array_name]:
                self.function_hints[func_name][array_name] = new_size
                changed = True
        fixed_params = self.fixed_function_array_sizes.get(func_name, {})
        if array_name in fixed_params:
            self.assert_not_larger_than_known_fixed(
                array_name, fixed_params[array_name], required_size
            )
        return changed

    def scan_call_site_resource_requirements(
        self, func_name, func, constants, index_hints, call_stack
    ):
        """Apply scalar call-site hints to resource-array accesses in callees."""
        changed = False
        for (
            node,
            visible_constants,
            visible_index_hints,
        ) in self.walk_nodes_with_constants(
            getattr(func, "body", []), constants, index_hints
        ):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                # Only an index into the declared array itself sizes it; ``a[i][j]``'s ``j`` indexes an element.
                array_name = (
                    None
                    if isinstance(array_expr, ArrayAccessNode)
                    else self.expression_name(array_expr)
                )
                index = self.resource_index_hint(
                    index_expr, visible_constants, visible_index_hints, call_stack
                )
                self.resolved_accesses[id(node)] = self.resolved_accesses.get(
                    id(node), True
                ) and (index is not None and index >= 0)
                if array_name is not None and index is not None and index >= 0:
                    changed = (
                        self.record_array_required_size(
                            func_name, array_name, index + 1
                        )
                        or changed
                    )
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            callee_name = self.function_call_name(node)
            callee = self.functions_by_name.get(callee_name)
            if callee is None or callee_name in call_stack:
                continue
            call_constants, call_index_hints = self.call_index_context(
                node, callee, visible_constants, visible_index_hints, call_stack
            )
            next_call_stack = set(call_stack)
            next_call_stack.add(callee_name)
            changed = (
                self.scan_call_site_resource_requirements(
                    callee_name,
                    callee,
                    call_constants,
                    call_index_hints,
                    next_call_stack,
                )
                or changed
            )
        return changed

    def register_fixed_requirement(self, scope_key, size, current_size):
        """Record a fixed size requirement for a propagated array argument."""
        if size is None:
            return
        existing = self.fixed_requirements.get(scope_key)
        if existing is not None and existing != size:
            raise TargetRefusal(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {existing} and {size}"
            )
        if current_size and current_size > size:
            raise TargetRefusal(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {current_size} and {size}"
            )
        self.fixed_requirements[scope_key] = size

    def assert_not_larger_than_fixed(self, scope_key, required_size):
        """Raise when propagated usage exceeds a fixed array requirement."""
        fixed_size = self.fixed_requirements.get(scope_key)
        if fixed_size is not None and required_size > fixed_size:
            raise TargetRefusal(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {fixed_size} and {required_size}"
            )
