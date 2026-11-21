"""Fixed array parameters in the GLSL generator: mutation analysis, mutated interval keys, write-back targets and storage pointer specialization views."""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..validation import expression_debug_name
from .opengl_values import (
    OpenGLFixedArrayResourceError,
    OpenGLPrivatePointerParameterError,
    OpenGLWorkgroupPointerError,
)
from .resource_helper_calls import is_read_only_helper_expression


class OpenGLFixedArrayParameterMixin:
    """Fixed array parameter members of ``GLSLCodeGen``."""

    def glsl_fixed_array_parameter_is_mutated(self, function, parameter_name):
        cache, active = self.glsl_fixed_array_mutation_analysis_state()
        cache_key = (id(function), parameter_name)
        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        parameter = next(
            (
                candidate
                for candidate in parameters
                if getattr(candidate, "name", None) == parameter_name
            ),
            None,
        )
        if cache_key in cache:
            return cache[cache_key]
        if cache_key in active:
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot prove fixed-array writeback through recursive "
                f"helper '{getattr(function, 'name', None)}'",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="fixed-array-mutation-cycle",
                source_location=getattr(function, "source_location", None),
            )

        source_shape = (
            self.glsl_fixed_array_parameter_shape(parameter)
            if parameter is not None
            else None
        )
        if source_shape is None:
            cache[cache_key] = False
            return False

        def mark_mutated():
            # Whether the callee writes the array is a fact about the callee alone. Only a caller that passes
            # caller-visible storage, such as ``&local``, needs those writes back
            # (:meth:`glsl_mark_fixed_array_writeback`); an unqualified array parameter is copied in.
            cache[cache_key] = True
            return True

        nodes = list(self.walk_ast(getattr(function, "body", [])))
        lexical_call_source_types = self.glsl_scalar_array_view_lexical_bindings(
            function
        )["call_source_types"]
        active.add(cache_key)
        try:
            for node in nodes:
                target = None
                if isinstance(node, AssignmentNode):
                    target = getattr(node, "target", getattr(node, "left", None))
                elif isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                    "++",
                    "--",
                }:
                    target = node.operand
                if (
                    target is not None
                    and self.glsl_fixed_array_target_root_name(target) == parameter_name
                ):
                    return mark_mutated()

            for call in nodes:
                if not isinstance(call, FunctionCallNode):
                    continue
                arguments = list(
                    getattr(call, "arguments", getattr(call, "args", [])) or []
                )
                forwarded = [
                    (index, kind)
                    for index, argument in enumerate(arguments)
                    if (
                        kind := self.glsl_fixed_array_forwarding_kind(
                            argument,
                            parameter_name,
                        )
                    )
                    is not None
                ]
                if not forwarded:
                    continue
                callee = self.glsl_fixed_array_mutation_callee(
                    function,
                    call,
                    parameter_name,
                    lexical_source_types=lexical_call_source_types.get(id(call), {}),
                )
                if callee is None:
                    continue
                callee_parameters = list(
                    getattr(
                        callee,
                        "parameters",
                        getattr(callee, "params", []),
                    )
                    or []
                )
                for argument_index, forwarding_kind in forwarded:
                    if argument_index >= len(callee_parameters):
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL fixed-array writeback resolved a helper with "
                            "an incompatible arity",
                            function_name=getattr(function, "name", None),
                            parameter_name=parameter_name,
                            reason="fixed-array-mutation-call-unresolved",
                            source_location=getattr(call, "source_location", None),
                        )
                    callee_parameter = callee_parameters[argument_index]
                    callee_qualifiers = set(
                        self.glsl_parameter_qualifiers(
                            callee_parameter, array_view_writeback=False
                        )
                    )
                    if callee_qualifiers.intersection({"out", "inout"}):
                        return mark_mutated()
                    callee_shape = self.glsl_fixed_array_parameter_shape(
                        callee_parameter
                    )
                    if callee_shape is None:
                        if forwarding_kind.startswith("addressed"):
                            raise OpenGLPrivatePointerParameterError(
                                "OpenGL cannot prove addressed fixed-array "
                                "writeback through a helper parameter without a "
                                "fixed-array or out/inout contract",
                                function_name=getattr(function, "name", None),
                                parameter_name=parameter_name,
                                reason="fixed-array-mutation-contract-unresolved",
                                source_location=getattr(call, "source_location", None),
                            )
                        continue
                    expected_extent = (
                        1
                        if forwarding_kind in {"element", "addressed-element"}
                        else source_shape.get("extent")
                    )
                    if callee_shape.get("extent") != expected_extent or (
                        callee_shape.get("element_type")
                        != source_shape.get("element_type")
                    ):
                        raise OpenGLPrivatePointerParameterError(
                            "OpenGL cannot preserve fixed-array writeback across "
                            "incompatible helper contracts",
                            function_name=getattr(function, "name", None),
                            parameter_name=parameter_name,
                            reason="fixed-array-mutation-contract-mismatch",
                            source_location=getattr(call, "source_location", None),
                        )
                    if self.glsl_fixed_array_parameter_is_mutated(
                        callee,
                        getattr(callee_parameter, "name", None),
                    ):
                        return mark_mutated()

            cache[cache_key] = False
            return False
        finally:
            active.discard(cache_key)

    def glsl_fixed_array_forwarded_parameters(self, function, parameter_name):
        """The fixed-array parameters of helpers that ``function`` passes its parameter ``parameter_name`` to."""
        nodes = list(self.walk_ast(getattr(function, "body", [])))
        lexical_call_source_types = self.glsl_scalar_array_view_lexical_bindings(
            function
        )["call_source_types"]
        forwarded = []
        for call in nodes:
            if not isinstance(call, FunctionCallNode):
                continue
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            indices = [
                index
                for index, argument in enumerate(arguments)
                if self.glsl_fixed_array_forwarding_kind(argument, parameter_name)
                is not None
            ]
            if not indices:
                continue
            callee = self.glsl_fixed_array_mutation_callee(
                function,
                call,
                parameter_name,
                lexical_source_types=lexical_call_source_types.get(id(call), {}),
            )
            if callee is None:
                continue
            callee_parameters = list(
                getattr(callee, "parameters", getattr(callee, "params", [])) or []
            )
            for index in indices:
                if index < len(callee_parameters) and (
                    self.glsl_fixed_array_parameter_shape(callee_parameters[index])
                    is not None
                ):
                    forwarded.append((callee, callee_parameters[index]))
        return forwarded

    def glsl_mark_fixed_array_writeback(self, function, parameter_name, visited=None):
        """Give ``function``'s fixed-array parameter ``parameter_name`` writeback when the function writes it.

        The caller passes storage it reads back, so the parameter becomes ``inout``, as does every fixed-array
        parameter of a helper the function forwards it to and that writes it in turn.
        """
        visited = set() if visited is None else visited
        key = (id(function), parameter_name)
        if key in visited:
            return
        visited.add(key)
        if not self.glsl_fixed_array_parameter_is_mutated(function, parameter_name):
            return
        parameters = getattr(function, "parameters", getattr(function, "params", []))
        for parameter in parameters or []:
            if getattr(parameter, "name", None) == parameter_name:
                parameter._glsl_scalar_array_view_inout = True
        for callee, callee_parameter in self.glsl_fixed_array_forwarded_parameters(
            function, parameter_name
        ):
            self.glsl_mark_fixed_array_writeback(
                callee, getattr(callee_parameter, "name", None), visited
            )

    def glsl_fixed_array_target_root_name(self, expression):
        current = expression
        while isinstance(current, (ArrayAccessNode, MemberAccessNode)):
            current = (
                getattr(current, "array", getattr(current, "array_expr", None))
                if isinstance(current, ArrayAccessNode)
                else current.object
            )
        if isinstance(current, PointerAccessNode):
            current = current.pointer_expr
        return self.expression_name(current)

    def glsl_fixed_array_storage_selection_bindings(self, expression, aliases):
        binding = self.glsl_storage_pointer_binding(expression, aliases)
        if binding is not None:
            return [binding]
        if not isinstance(expression, TernaryOpNode):
            return []
        return [
            *self.glsl_fixed_array_storage_selection_bindings(
                expression.true_expr, aliases
            ),
            *self.glsl_fixed_array_storage_selection_bindings(
                expression.false_expr, aliases
            ),
        ]

    def raise_glsl_fixed_array_resource_error(
        self,
        message,
        *,
        function,
        parameter,
        argument,
        binding=None,
        resource_names=None,
        fixed_extent=None,
        required_access=None,
        actual_access=None,
        reason,
    ):
        function_name = getattr(function, "name", None)
        parameter_name = getattr(parameter, "name", None)
        if resource_names is None and binding is not None:
            resource_names = [binding.get("root")]
        raise OpenGLFixedArrayResourceError(
            message,
            function_name=function_name,
            parameter_name=parameter_name,
            resource_names=[name for name in resource_names or () if name],
            fixed_extent=fixed_extent,
            required_access=required_access,
            actual_access=actual_access,
            reason=reason,
            source_location=(
                getattr(argument, "source_location", None)
                or getattr(parameter, "source_location", None)
            ),
        )

    def glsl_fixed_array_storage_binding(
        self, function, parameter, argument, aliases, contract
    ):
        binding = self.glsl_storage_pointer_binding(argument, aliases)
        if binding is None:
            candidates = self.glsl_fixed_array_storage_selection_bindings(
                argument, aliases
            )
            if candidates:
                resource_names = {
                    candidate.get("root")
                    for candidate in candidates
                    if candidate.get("root")
                }
                if not self.glsl_side_effect_free_expression(argument):
                    self.raise_glsl_fixed_array_resource_error(
                        "OpenGL cannot specialize a side-effecting runtime "
                        "storage-resource selection for fixed-array parameter "
                        f"'{getattr(function, 'name', None)}."
                        f"{getattr(parameter, 'name', None)}'",
                        function=function,
                        parameter=parameter,
                        argument=argument,
                        resource_names=resource_names,
                        fixed_extent=contract.get("extent"),
                        required_access=contract.get("required_access"),
                        reason="side-effecting-resource-selection",
                    )
                self.raise_glsl_fixed_array_resource_error(
                    "OpenGL cannot choose one concrete storage resource for "
                    "fixed-array parameter "
                    f"'{getattr(function, 'name', None)}."
                    f"{getattr(parameter, 'name', None)}'",
                    function=function,
                    parameter=parameter,
                    argument=argument,
                    resource_names=resource_names,
                    fixed_extent=contract.get("extent"),
                    required_access=contract.get("required_access"),
                    reason="ambiguous-resource-aliasing",
                )

            argument_type = self.glsl_source_expression_type(argument)
            if self.structured_buffer_type_name(argument_type) in {
                "StructuredBuffer",
                "RWStructuredBuffer",
            }:
                self.raise_glsl_fixed_array_resource_error(
                    "OpenGL cannot specialize a side-effecting or unresolved "
                    "runtime storage-resource expression for fixed-array "
                    f"parameter '{getattr(function, 'name', None)}."
                    f"{getattr(parameter, 'name', None)}'",
                    function=function,
                    parameter=parameter,
                    argument=argument,
                    fixed_extent=contract.get("extent"),
                    required_access=contract.get("required_access"),
                    reason="side-effecting-resource-selection",
                )
            return None

        if not self.glsl_side_effect_free_expression(argument):
            self.raise_glsl_fixed_array_resource_error(
                "OpenGL cannot move a side-effecting storage-resource base "
                "expression into a specialized fixed-array helper call",
                function=function,
                parameter=parameter,
                argument=argument,
                binding=binding,
                fixed_extent=contract.get("extent"),
                required_access=contract.get("required_access"),
                actual_access=binding.get("access"),
                reason="side-effecting-resource-selection",
            )
        return binding

    def glsl_checked_private_pointer_interval(
        self, expression, intervals, constants, *, require_arithmetic_types=False
    ):
        """Do not use mathematical bounds that depend on source integer overflow."""
        for node in self.walk_ast(expression):
            if not isinstance(node, (BinaryOpNode, UnaryOpNode)):
                continue
            bounds = self.glsl_private_pointer_interval(node, intervals, constants)
            domain = self.glsl_integer_type_domain(node, allow_wide=True)
            if require_arithmetic_types and (bounds is None or domain is None):
                return None
            if (
                bounds is not None
                and domain is not None
                and not (domain.minimum <= bounds[0] <= bounds[1] <= domain.maximum)
            ):
                return None
        return self.glsl_private_pointer_interval(expression, intervals, constants)

    def glsl_simple_struct_constructor_member_intervals(
        self,
        expression,
        intervals,
        constants,
        _active=None,
    ):
        """Summarize fields copied or returned by straight-line value helpers.

        Only local/by-value objects may be updated. Ambiguous calls, references,
        recursion, control flow and unmodeled effects cannot establish bounds.
        """

        if isinstance(
            expression, (str, IdentifierNode, VariableNode, MemberAccessNode)
        ):
            name = expression_debug_name(expression)
            prefix = f"{name}."
            return {
                key[len(prefix) :]: interval
                for key, interval in (intervals or {}).items()
                if name and key.startswith(prefix)
            }

        if not isinstance(expression, FunctionCallNode):
            return {}
        function_name = self.function_call_name(expression)
        arguments = list(expression.arguments or [])
        candidates = [
            candidate
            for candidate in self.glsl_function_overloads_by_name.get(function_name, ())
            if len(
                getattr(
                    candidate,
                    "parameters",
                    getattr(candidate, "params", []),
                )
                or []
            )
            == len(arguments)
            and getattr(candidate, "body", None) is not None
        ]
        fallback = self.function_definitions.get(function_name)
        if (
            fallback is not None
            and fallback not in candidates
            and len(
                getattr(fallback, "parameters", getattr(fallback, "params", [])) or []
            )
            == len(arguments)
            and getattr(fallback, "body", None) is not None
        ):
            candidates.append(fallback)
        unique_candidates = {}
        for candidate in candidates:
            signature = self.glsl_source_function_signature(candidate)
            unique_candidates.setdefault(signature, candidate)
        if len(unique_candidates) != 1:
            return {}
        constructor = next(iter(unique_candidates.values()))
        active = set() if _active is None else _active
        if id(constructor) in active:
            return {}
        parameters = list(
            getattr(
                constructor,
                "parameters",
                getattr(constructor, "params", []),
            )
            or []
        )
        body = getattr(constructor, "body", None)
        statements = list(getattr(body, "statements", body) or [])
        if not statements or not isinstance(statements[-1], ReturnNode):
            return {}
        used_names = self.glsl_interval_dependency_names(statements)
        for parameter in parameters:
            parameter_type = (
                self.type_name_string(
                    getattr(parameter, "param_type", getattr(parameter, "vtype", None))
                )
                or ""
            )
            if (
                set(self.glsl_parameter_qualifiers(parameter)).intersection(
                    {"out", "inout"}
                )
                or parameter_type.endswith("&")
                or parameter_type.endswith("*")
                and parameter.name in used_names
            ):
                return {}

        local_constants = self.initial_literal_int_constants(constructor)
        local_intervals = {}
        local_names = {parameter.name for parameter in parameters}
        local_types = {
            parameter.name: getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            for parameter in parameters
        }

        def members(value, bounds, literals):
            return self.glsl_simple_struct_constructor_member_intervals(
                value, bounds, literals, active
            )

        def effect_free(value, bounds, literals):
            return self.glsl_side_effect_free_expression(value) or (
                isinstance(value, FunctionCallNode)
                and bool(members(value, bounds, literals))
            )

        def scalar_interval(value, value_intervals, value_constants, value_types):
            with self.glsl_lexical_source_type_scope(value_types):
                return self.glsl_checked_private_pointer_interval(
                    value,
                    value_intervals,
                    value_constants,
                    require_arithmetic_types=True,
                )

        for parameter, argument in zip(parameters, arguments):
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                continue
            argument_members = members(argument, intervals, constants)
            if not (
                self.glsl_side_effect_free_expression(argument)
                or isinstance(argument, FunctionCallNode)
                and argument_members
                or is_read_only_helper_expression(self, argument)
            ):
                return {}
            for member, interval in argument_members.items():
                local_intervals[f"{parameter_name}.{member}"] = interval
            interval = scalar_interval(
                argument,
                intervals,
                constants,
                {},
            )
            with self.glsl_lexical_source_type_scope(local_types):
                domain = self.glsl_integer_type_domain(
                    IdentifierNode(parameter_name), allow_wide=True
                )
            if interval is not None and (
                domain is None
                or not (domain.minimum <= interval[0] <= interval[1] <= domain.maximum)
            ):
                interval = None
            if interval is not None:
                local_intervals[parameter_name] = interval
            parameter_type = getattr(
                parameter,
                "param_type",
                getattr(parameter, "vtype", None),
            )
            component_intervals = self.glsl_static_integer_vector_component_intervals(
                argument,
                parameter_type,
                intervals,
                constants,
            )
            for component_index, component_interval in enumerate(
                component_intervals or ()
            ):
                key = self.glsl_index_component_interval_key(
                    parameter_name,
                    component_index,
                )
                if key is not None:
                    local_intervals[key] = component_interval

        # Arguments are evaluated in the caller; only entering the body can recur.
        active = active | {id(constructor)}
        for statement in statements[:-1]:
            if isinstance(statement, VariableNode):
                name = getattr(statement, "name", None)
                if not name or name in local_names:
                    return {}
                local_names.add(name)
                local_constants.pop(name, None)
                variable_type = getattr(
                    statement, "var_type", getattr(statement, "vtype", None)
                )
                if (self.type_name_string(variable_type) or "").endswith(("*", "&")):
                    return {}
                local_types[name] = variable_type
                initial = getattr(statement, "initial_value", None)
                if initial is not None:
                    if not effect_free(initial, local_intervals, local_constants):
                        return {}
                    for member, interval in members(
                        initial, local_intervals, local_constants
                    ).items():
                        local_intervals[f"{name}.{member}"] = interval
                continue
            assignment = (
                statement.expression
                if isinstance(statement, ExpressionStatementNode)
                else statement
            )
            if not isinstance(assignment, AssignmentNode):
                return {}
            operator = self.map_operator(
                getattr(assignment, "operator", getattr(assignment, "op", "="))
            )
            if operator not in {"=", "+=", "-="}:
                return {}
            target = getattr(
                assignment,
                "target",
                getattr(assignment, "left", None),
            )
            key = expression_debug_name(target)
            if (
                not isinstance(target, MemberAccessNode)
                or key.split(".")[0] not in local_names
            ):
                return {}
            assigned = getattr(assignment, "value", getattr(assignment, "right", None))
            if not effect_free(assigned, local_intervals, local_constants):
                return {}
            interval = scalar_interval(
                assigned, local_intervals, local_constants, local_types
            )
            fields = (
                members(assigned, local_intervals, local_constants)
                if operator == "="
                else {}
            )
            previous = local_intervals.get(key)
            if operator == "+=":
                interval = (
                    (previous[0] + interval[0], previous[1] + interval[1])
                    if previous and interval
                    else None
                )
            elif operator == "-=":
                interval = (
                    (previous[0] - interval[1], previous[1] - interval[0])
                    if previous and interval
                    else None
                )
            if interval is not None:
                with self.glsl_lexical_source_type_scope(local_types):
                    domain = self.glsl_integer_type_domain(target, allow_wide=True)
                if domain is None or not (
                    domain.minimum <= interval[0] <= interval[1] <= domain.maximum
                ):
                    interval = None
            for old_key in list(local_intervals):
                if old_key == key or old_key.startswith(f"{key}."):
                    del local_intervals[old_key]
            if interval is not None:
                local_intervals[key] = interval
            for member, member_interval in fields.items():
                local_intervals[f"{key}.{member}"] = member_interval
        return members(statements[-1].value, local_intervals, local_constants)

    def glsl_workgroup_call_parameter_intervals(self, function, arguments):
        result = {}
        constants = self.glsl_active_literal_int_constants()
        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        for parameter, argument in zip(parameters, arguments or []):
            name = getattr(parameter, "name", None)
            if not name or self.glsl_workgroup_pointer_element_type(parameter):
                continue
            source_intervals = getattr(argument, "_glsl_control_flow_intervals", {})
            member_intervals = self.glsl_simple_struct_constructor_member_intervals(
                argument, source_intervals, constants
            )
            for member, interval in member_intervals.items():
                result[f"{name}.{member}"] = interval
            interval = self.glsl_checked_private_pointer_interval(
                argument,
                source_intervals,
                constants,
            )
            if interval is not None:
                result[name] = interval
            parameter_type = getattr(
                parameter,
                "param_type",
                getattr(parameter, "vtype", None),
            )
            component_intervals = self.glsl_static_integer_vector_component_intervals(
                argument,
                parameter_type,
                source_intervals,
                constants,
            )
            for component_index, component_interval in enumerate(
                component_intervals or ()
            ):
                key = self.glsl_index_component_interval_key(
                    name,
                    component_index,
                )
                if key is not None:
                    result[key] = component_interval
        return result

    def glsl_interval_dependency_names(self, expression):
        if isinstance(expression, str):
            return {expression} if re.fullmatch(r"[A-Za-z_]\w*", expression) else set()
        nodes = (
            expression.walk()
            if hasattr(expression, "walk")
            else self.walk_ast(expression)
        )
        return {
            name
            for node in nodes
            if isinstance(node, (IdentifierNode, VariableNode))
            and isinstance(name := getattr(node, "name", None), str)
        }

    def glsl_interval_expression_keys(self, expression):
        """Return component-sensitive names that an interval depends on."""

        field_key = self.glsl_declared_struct_member_interval_key(expression)
        if field_key is not None:
            return {field_key}
        if expression is None or isinstance(expression, (int, float, bool)):
            return set()
        if isinstance(expression, str):
            return (
                {expression}
                if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", expression)
                else set()
            )
        if isinstance(expression, (IdentifierNode, VariableNode)):
            name = getattr(expression, "name", None)
            return {name} if isinstance(name, str) else set()
        if isinstance(expression, MemberAccessNode):
            component_indices = tuple(
                self.glsl_index_component_index(component)
                for component in str(expression.member)
            )
            if component_indices and all(
                component_index is not None for component_index in component_indices
            ):
                return {
                    key
                    for component_index in component_indices
                    if (
                        key := self.glsl_index_component_interval_key(
                            expression.object,
                            component_index,
                        )
                    )
                    is not None
                }
            object_name = expression_debug_name(expression.object)
            if object_name:
                return {f"{object_name}.{expression.member}"}
        if isinstance(expression, ArrayAccessNode):
            component_index = self.literal_int_value(
                expression.index,
                self.glsl_active_literal_int_constants(),
            )
            if (
                isinstance(component_index, int)
                and not isinstance(component_index, bool)
                and component_index in range(4)
            ):
                key = self.glsl_index_component_interval_key(
                    expression.array,
                    component_index,
                )
                if key is not None:
                    return {key}

        dependencies = set()
        if hasattr(expression, "child_nodes"):
            for child in expression.child_nodes():
                dependencies.update(self.glsl_interval_expression_keys(child))
            return dependencies
        if hasattr(expression, "__dict__"):
            for field, child in vars(expression).items():
                if field not in {
                    "annotations",
                    "parent",
                    "source_location",
                }:
                    if isinstance(child, (list, tuple, set)):
                        for item in child:
                            dependencies.update(
                                self.glsl_interval_expression_keys(item)
                            )
                    else:
                        dependencies.update(self.glsl_interval_expression_keys(child))
        return dependencies

    def glsl_interval_mutation_target_keys(self, target):
        """Return exact interval keys invalidated by an lvalue mutation."""

        field_key = self.glsl_declared_struct_member_interval_key(target)
        if field_key is not None:
            return {field_key}
        if isinstance(target, (str, IdentifierNode, VariableNode)):
            name = self.expression_name(target)
            return {name} if name else set()
        if isinstance(target, MemberAccessNode):
            component_indices = tuple(
                self.glsl_index_component_index(component)
                for component in str(target.member)
            )
            if component_indices and all(
                component_index is not None for component_index in component_indices
            ):
                return {
                    key
                    for component_index in component_indices
                    if (
                        key := self.glsl_index_component_interval_key(
                            target.object,
                            component_index,
                        )
                    )
                    is not None
                }
            object_name = expression_debug_name(target.object)
            return {f"{object_name}.{target.member}"} if object_name else set()
        if isinstance(target, ArrayAccessNode):
            component_index = self.literal_int_value(
                target.index,
                self.glsl_active_literal_int_constants(),
            )
            if (
                isinstance(component_index, int)
                and not isinstance(component_index, bool)
                and component_index in range(4)
            ):
                key = self.glsl_index_component_interval_key(
                    target.array,
                    component_index,
                )
                if key is not None:
                    return {key}
            name = expression_debug_name(target.array)
            return {name} if name else set()
        return set()

    @staticmethod
    def glsl_interval_key_sets_overlap(dependencies, mutations):
        return any(
            dependency == mutation
            or dependency.startswith(f"{mutation}.")
            or mutation.startswith(f"{dependency}.")
            for dependency in dependencies
            for mutation in mutations
        )

    def glsl_function_mutated_interval_keys(self, function):
        """Collect component-sensitive direct and transitive interval mutations."""

        def structural_nodes(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    yield from structural_nodes(child)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    yield from structural_nodes(child)
                return
            if hasattr(value, "walk"):
                yield from value.walk()

        def argument_member_mutations(argument, parameter_name, callee_mutations):
            fallback = self.glsl_interval_mutation_target_keys(argument)
            if callee_mutations is None or parameter_name in callee_mutations:
                return fallback
            relevant = {
                key for key in callee_mutations if key.startswith(f"{parameter_name}.")
            }
            if not relevant:
                return set()
            argument_name = expression_debug_name(argument)
            if (
                not argument_name
                or re.fullmatch(
                    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
                    argument_name,
                )
                is None
            ):
                return fallback
            return {f"{argument_name}{key[len(parameter_name) :]}" for key in relevant}

        memo = {}
        active = set()

        def collect(current):
            current_id = id(current)
            if current_id in memo:
                return set(memo[current_id])
            if current_id in active:
                # A recursive inout edge cannot establish a finite member-only
                # mutation summary without a fixed-point proof.  Its caller
                # therefore invalidates the complete forwarded lvalue.
                return None

            active.add(current_id)
            mutated = set()
            try:
                for node in structural_nodes(getattr(current, "body", [])):
                    if isinstance(node, AssignmentNode):
                        mutated.update(
                            self.glsl_interval_mutation_target_keys(
                                getattr(
                                    node,
                                    "target",
                                    getattr(node, "left", None),
                                )
                            )
                        )
                        continue
                    if isinstance(node, UnaryOpNode) and (
                        self.map_operator(node.op) in {"++", "--"}
                        or node.op in {
                            "PRE_INCREMENT",
                            "PRE_DECREMENT",
                            "POST_INCREMENT",
                            "POST_DECREMENT",
                        }
                    ):
                        mutated.update(
                            self.glsl_interval_mutation_target_keys(node.operand)
                        )
                        continue
                    if (
                        isinstance(node, UnaryOpNode)
                        and self.map_operator(node.op) == "&"
                    ):
                        mutated.update(
                            self.glsl_interval_mutation_target_keys(node.operand)
                        )
                        continue
                    if not isinstance(node, FunctionCallNode):
                        continue
                    function_name = self.function_call_name(node)
                    arguments = list(node.arguments or [])
                    if function_name in self.GLSL_MEMORY_ATOMIC_FUNCTIONS and arguments:
                        mutated.update(
                            self.glsl_interval_mutation_target_keys(arguments[0])
                        )
                        if (
                            function_name == "atomicCompareExchangeWeak"
                            and len(arguments) > 1
                        ):
                            mutated.update(
                                self.glsl_interval_mutation_target_keys(arguments[1])
                            )
                    candidates = list(
                        self.glsl_function_overloads_by_name.get(function_name, ())
                    )
                    fallback = self.function_definitions.get(function_name)
                    if fallback is not None and fallback not in candidates:
                        candidates.append(fallback)
                    for argument_index, argument in enumerate(arguments):
                        for candidate in candidates:
                            parameters = list(
                                getattr(
                                    candidate,
                                    "parameters",
                                    getattr(candidate, "params", []),
                                )
                                or []
                            )
                            if argument_index >= len(parameters):
                                continue
                            parameter = parameters[argument_index]
                            if not set(
                                self.glsl_parameter_qualifiers(parameter)
                            ).intersection({"out", "inout"}):
                                continue
                            parameter_name = getattr(parameter, "name", None)
                            if (
                                not parameter_name
                                or getattr(candidate, "body", None) is None
                            ):
                                mutated.update(
                                    self.glsl_interval_mutation_target_keys(argument)
                                )
                                continue
                            mutated.update(
                                argument_member_mutations(
                                    argument,
                                    parameter_name,
                                    collect(candidate),
                                )
                            )
            finally:
                active.remove(current_id)

            memo[current_id] = frozenset(mutated)
            return set(mutated)

        return collect(function) or set()

    def glsl_mutation_target_names(self, target):
        if isinstance(target, (str, IdentifierNode, VariableNode)):
            name = self.expression_name(target)
            return {name} if name else set()
        if isinstance(target, ArrayAccessNode):
            return self.glsl_mutation_target_names(target.array)
        if isinstance(target, MemberAccessNode):
            names = self.glsl_mutation_target_names(target.object)
            object_name = expression_debug_name(target.object)
            if object_name:
                names.add(f"{object_name}.{target.member}")
            return names
        return set()

    def glsl_function_mutated_interval_names(self, function):
        mutated = set()

        def structural_nodes(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    yield from structural_nodes(child)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    yield from structural_nodes(child)
                return
            if hasattr(value, "walk"):
                yield from value.walk()

        for node in structural_nodes(getattr(function, "body", [])):
            if isinstance(node, AssignmentNode):
                mutated.update(
                    self.glsl_mutation_target_names(
                        getattr(node, "target", getattr(node, "left", None))
                    )
                )
                continue
            if isinstance(node, UnaryOpNode) and (
                self.map_operator(node.op) in {"++", "--"}
                or node.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }
            ):
                mutated.update(self.glsl_mutation_target_names(node.operand))
                continue
            if isinstance(node, UnaryOpNode) and self.map_operator(node.op) == "&":
                mutated.update(self.glsl_mutation_target_names(node.operand))
                continue
            if not isinstance(node, FunctionCallNode):
                continue
            function_name = self.function_call_name(node)
            arguments = list(node.arguments or [])
            if function_name in self.GLSL_MEMORY_ATOMIC_FUNCTIONS and arguments:
                mutated.update(self.glsl_mutation_target_names(arguments[0]))
                if function_name == "atomicCompareExchangeWeak" and len(arguments) > 1:
                    mutated.update(self.glsl_mutation_target_names(arguments[1]))
            candidates = list(
                self.glsl_function_overloads_by_name.get(function_name, ())
            )
            fallback = self.function_definitions.get(function_name)
            if fallback is not None and fallback not in candidates:
                candidates.append(fallback)
            for argument_index, argument in enumerate(arguments):
                for candidate in candidates:
                    parameters = list(
                        getattr(
                            candidate,
                            "parameters",
                            getattr(candidate, "params", []),
                        )
                        or []
                    )
                    if argument_index >= len(parameters):
                        continue
                    if set(
                        self.glsl_parameter_qualifiers(parameters[argument_index])
                    ).intersection({"out", "inout"}):
                        mutated.update(self.glsl_mutation_target_names(argument))
                        break
        return mutated

    def validate_glsl_workgroup_pointer_specialization(
        self,
        function,
        parameter,
        argument,
        binding,
        parameter_intervals,
    ):
        parameter_name = getattr(parameter, "name", None)
        access_intervals = self.glsl_workgroup_pointer_direct_access_intervals(
            function,
            parameter_intervals,
        )
        if parameter_name not in access_intervals:
            return

        function_name = getattr(
            function, "_glsl_resource_source_name", None
        ) or getattr(function, "name", None)
        backing_name = binding.get("backing_declaration") or binding.get("root")
        materialization_name = self.glsl_resource_specialization_name(
            function_name,
            {0: (parameter_name, binding)},
        )
        entry_owner = binding.get("entry_owner")
        entry_point = (
            str(entry_owner).partition(":")[0] if entry_owner is not None else None
        )
        asserted_matches = [
            assertion
            for assertion in self.workgroup_access_assertions
            if assertion.applies_to(entry_point, function_name, parameter_name)
        ]
        asserted_interval = None
        if asserted_matches:
            asserted_interval = (
                max(assertion.minimum for assertion in asserted_matches),
                min(assertion.maximum for assertion in asserted_matches),
            )
        offset_interval = self.glsl_workgroup_pointer_offset_interval(
            binding,
            getattr(argument, "_glsl_control_flow_intervals", {}),
        )
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        if offset_interval is not None and offset_interval[0] == offset_interval[1]:
            proven_offset = str(offset_interval[0])
            if proven_offset not in rendered_offset:
                rendered_offset = f"{rendered_offset} (proven {proven_offset})"
        elif offset_interval is not None:
            rendered_offset = (
                f"{rendered_offset} (proven {offset_interval[0]}..{offset_interval[1]})"
            )

        access_interval = access_intervals[parameter_name]
        static_interval = None
        if offset_interval is not None and access_interval is not None:
            static_interval = (
                offset_interval[0] + access_interval[0],
                offset_interval[1] + access_interval[1],
            )

        effective_interval = static_interval
        if asserted_interval is not None:
            if asserted_interval[0] > asserted_interval[1]:
                reason = "conflicting-access-assertions"
                message = (
                    "OpenGL workgroup access assertions conflict for "
                    f"'{function_name}.{parameter_name}' at entry point "
                    f"'{entry_point}'"
                )
            elif static_interval is not None and (
                max(static_interval[0], asserted_interval[0])
                > min(static_interval[1], asserted_interval[1])
            ):
                reason = "asserted-access-conflict"
                message = (
                    "OpenGL workgroup access assertion for "
                    f"'{function_name}.{parameter_name}' at entry point "
                    f"'{entry_point}' conflicts with the statically derived "
                    f"range {static_interval[0]} through {static_interval[1]}"
                )
            else:
                effective_interval = (
                    max(
                        asserted_interval[0],
                        (
                            static_interval[0]
                            if static_interval is not None
                            else asserted_interval[0]
                        ),
                    ),
                    min(
                        asserted_interval[1],
                        (
                            static_interval[1]
                            if static_interval is not None
                            else asserted_interval[1]
                        ),
                    ),
                )
                reason = None
                message = None
        elif offset_interval is None:
            reason = "unprovable-view-offset"
            message = (
                "OpenGL cannot prove the workgroup pointer view offset for "
                f"'{function_name}.{parameter_name}' into shared backing "
                f"'{backing_name}'"
            )
        elif access_interval is None:
            reason = "unprovable-view-access"
            message = (
                "OpenGL cannot prove the workgroup pointer access range for "
                f"'{function_name}.{parameter_name}' into shared backing "
                f"'{backing_name}'"
            )
        else:
            reason = None
            message = None

        if reason is None and effective_interval is not None:
            lower, upper = effective_interval
            extent = binding.get("array_size")
            if lower >= 0 and isinstance(extent, int) and upper < extent:
                binding["_glsl_validated_workgroup_access_interval"] = (
                    lower,
                    upper,
                )
                return
            reason = "view-out-of-bounds"
            message = (
                "OpenGL workgroup pointer view "
                f"'{function_name}.{parameter_name}' requires shared elements "
                f"{lower} through {upper}, but backing '{backing_name}' has "
                f"extent {extent}"
            )
        raise OpenGLWorkgroupPointerError(
            message,
            function_name=function_name,
            parameter_name=parameter_name,
            backing_name=backing_name,
            offset_expression=rendered_offset,
            materialization_name=materialization_name,
            reason=reason,
            entry_point=entry_point,
            asserted_range=asserted_interval,
            source_location=getattr(argument, "source_location", None),
        )

    @staticmethod
    def glsl_storage_pointer_specialization_view_key(binding):
        reinterpretation = binding.get("pointer_reinterpretation")
        byte_offset = str(binding.get("byte_offset", "0"))
        if reinterpretation is None:
            return ("byte-offset",) if byte_offset not in {"0", "0u"} else ("direct",)

        def layout_key(layout):
            return (
                layout.name,
                layout.kind,
                layout.bit_width,
                layout.signed,
            )

        return (
            "reinterpret",
            layout_key(reinterpretation["source_layout"]),
            layout_key(reinterpretation["target_layout"]),
        )
