"""Ray-tracing stages and ray queries in the Slang generator: parameter roles, call validation and result types."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix


class SlangRayTracingMixin:
    """Ray-tracing members of ``SlangCodeGen``."""

    def slang_ray_stage_types(self):
        return {
            "raygeneration",
            "intersection",
            "closesthit",
            "anyhit",
            "miss",
            "callable",
        }

    def slang_ray_role_from_name(self, name, shader_stage=None):
        if not name:
            return None

        normalized = str(name).lower()
        compact = normalized.replace("_", "")
        if compact in {"payload", "raypayloadext", "raypayloadinext"}:
            if shader_stage == "callable":
                return "callable_data"
            return "payload"
        if compact in {"hitattribute", "hitattributeext"}:
            return "hit_attribute"
        if compact in {"callabledata", "callabledataext", "callabledatainext"}:
            return "callable_data"
        return None

    def slang_ray_attribute_role_name(self, attr, shader_stage=None):
        return self.slang_ray_role_from_name(getattr(attr, "name", None), shader_stage)

    def slang_ray_semantic_role(self, parameter, shader_type=None):
        shader_stage = self.slang_shader_stage_name(shader_type)
        semantic = getattr(parameter, "semantic", None)
        if semantic:
            role = self.slang_ray_role_from_name(semantic, shader_stage)
            if role:
                return role

        for attr in getattr(parameter, "attributes", []) or []:
            role = self.slang_ray_attribute_role_name(attr, shader_stage)
            if role:
                return role
        return None

    def slang_ray_role_parameters(self, parameters, shader_type):
        role_parameters = {}
        for parameter in parameters or []:
            role = self.slang_ray_semantic_role(parameter, shader_type)
            if role:
                role_parameters.setdefault(role, []).append(parameter)
        return role_parameters

    def is_slang_stage_local_interface_parameter(self, parameter):
        if hasattr(parameter, "get_annotation"):
            return bool(parameter.get_annotation("slang_stage_local_interface"))
        return bool(
            getattr(parameter, "annotations", {}).get("slang_stage_local_interface")
        )

    def is_slang_ray_interface_value_type(self, base_type, role=None):
        """Whether Slang accepts ``base_type`` as a ray payload, hit attribute or
        callable data type.

        DXC requires a user-defined structure there, but Slang accepts any value
        type and wraps a non-structure in a structure itself when it writes HLSL.
        """
        if base_type in self.user_struct_names:
            return True
        if base_type == "BuiltInTriangleIntersectionAttributes":
            return role in {None, "hit_attribute"}
        return self.is_slang_ray_query_value_type(base_type)

    def validate_slang_ray_parameter_type(self, parameter, role):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return

        mapped_type = self.convert_type(type_name)
        base_type, _array_suffix = split_array_type_suffix(mapped_type)
        if not self.is_slang_ray_interface_value_type(base_type, role):
            raise SlangRefusal(
                f"Slang ray {role} parameter '{parameter.name}' must use a "
                "structure, scalar, vector or matrix type"
            )

    def validate_slang_ray_stage_parameters(self, func, shader_type, parameters):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage not in self.slang_ray_stage_types():
            return

        role_parameters = self.slang_ray_role_parameters(parameters, shader_type)
        allowed_stages = {
            "payload": {"closesthit", "anyhit", "miss"},
            "hit_attribute": {"closesthit", "anyhit"},
            "callable_data": {"callable"},
        }
        for role, role_params in role_parameters.items():
            if len(role_params) > 1:
                raise SlangRefusal(
                    f"Slang {shader_type} stage must declare at most one "
                    f"{role} parameter"
                )
            if shader_stage not in allowed_stages.get(role, set()):
                if not (
                    shader_stage == "raygeneration"
                    and role in {"payload", "callable_data"}
                    and self.is_slang_stage_local_interface_parameter(role_params[0])
                ):
                    raise SlangRefusal(
                        f"Slang {shader_type} stage cannot use {role} parameter "
                        f"'{role_params[0].name}'"
                    )
            self.validate_slang_ray_parameter_type(role_params[0], role)

    def slang_function_scope_variable_types(self, func):
        variables = {}
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            type_name = self.get_variable_type(node)
            if type_name is not None:
                variables[node.name] = type_name
        return variables

    def slang_ray_tracing_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call = self.slang_ray_tracing_call_parts(node)
            if call is not None:
                calls.append(call)
        return calls

    def slang_ray_tracing_intrinsic_names(self):
        return {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "AcceptHitAndEndSearch",
            "IgnoreHit",
        }

    def slang_ray_tracing_call_parts(self, node, include_user_functions=False):
        if isinstance(node, RayTracingOpNode):
            operation = getattr(node, "operation", None)
            if not include_user_functions and operation in self.user_function_names:
                return None
            return operation, self.normalized_slang_intrinsic_args(
                getattr(node, "arguments", [])
            )

        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        func_name = getattr(func_expr, "name", func_expr)
        if func_name not in self.slang_ray_tracing_intrinsic_names():
            return None
        if not include_user_functions and func_name in self.user_function_names:
            return None
        return func_name, self.normalized_slang_intrinsic_args(
            getattr(node, "arguments", getattr(node, "args", []))
        )

    def slang_function_call_name(self, node):
        if not isinstance(node, FunctionCallNode):
            return None
        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def slang_user_function_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call_name = self.slang_function_call_name(node)
            if call_name in self.user_functions_by_name:
                calls.append(call_name)
        return calls

    def slang_user_function_call_nodes(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call_name = self.slang_function_call_name(node)
            if call_name in self.user_functions_by_name:
                calls.append(node)
        return calls

    def slang_expression_root_identifier(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.slang_expression_root_identifier(array)
        if isinstance(expr, MemberAccessNode):
            obj = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.slang_expression_root_identifier(obj)
        if isinstance(expr, PointerAccessNode):
            pointer = getattr(expr, "pointer_expr", None)
            return self.slang_expression_root_identifier(pointer)
        return None

    def slang_resolve_alias_root(self, root_name, aliases):
        seen = set()
        while root_name in aliases and root_name not in seen:
            seen.add(root_name)
            root_name = aliases[root_name]
        return root_name

    def slang_parameter_alias_roots(self, func, parameter_names):
        aliases = {}

        def alias_root(expr):
            root_name = self.slang_expression_root_identifier(expr)
            return self.slang_resolve_alias_root(root_name, aliases)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                initial_value = getattr(
                    node, "initial_value", getattr(node, "value", None)
                )
                source_root = alias_root(initial_value)
                if source_root in parameter_names:
                    aliases[node.name] = source_root
                continue

            if isinstance(node, AssignmentNode):
                target = getattr(node, "left", getattr(node, "target", None))
                target_root = self.slang_expression_root_identifier(target)
                if target_root in parameter_names:
                    continue
                source_root = alias_root(
                    getattr(node, "right", getattr(node, "value", None))
                )
                if target_root and source_root in parameter_names:
                    aliases[target_root] = source_root

        return aliases

    def slang_ray_interface_argument_role(self, operation, args):
        if operation == "TraceRay" and args:
            return args[-1], "payload"
        if operation == "CallShader" and len(args) >= 2:
            return args[1], "callable data"
        if operation == "ReportHit" and len(args) >= 3:
            return args[2], "hit attribute"
        return None, None

    def slang_ray_interface_parameter_roles(self, func, visited_helpers=None):
        if visited_helpers is None:
            visited_helpers = set()

        func_name = getattr(func, "name", None)
        if func_name in visited_helpers:
            return {}
        if func_name:
            visited_helpers = visited_helpers | {func_name}

        parameters = getattr(func, "parameters", getattr(func, "params", []))
        parameter_names = {
            getattr(parameter, "name", None) for parameter in parameters or []
        }
        parameter_names.discard(None)
        roles = {}
        alias_roots = self.slang_parameter_alias_roots(func, parameter_names)

        for operation, args in self.slang_ray_tracing_calls(func):
            argument, role = self.slang_ray_interface_argument_role(operation, args)
            if argument is None:
                continue
            root_name = self.slang_expression_root_identifier(argument)
            root_name = self.slang_resolve_alias_root(root_name, alias_roots)
            if root_name in parameter_names and root_name not in roles:
                roles[root_name] = role

        for call_node in self.slang_user_function_call_nodes(func):
            helper_name = self.slang_function_call_name(call_node)
            if helper_name in visited_helpers:
                continue
            helper_func = self.user_functions_by_name.get(helper_name)
            if helper_func is None:
                continue
            helper_roles = self.slang_ray_interface_parameter_roles(
                helper_func, visited_helpers
            )
            if not helper_roles:
                continue
            helper_params = getattr(
                helper_func, "parameters", getattr(helper_func, "params", [])
            )
            args = getattr(call_node, "arguments", getattr(call_node, "args", []))
            for index, helper_param in enumerate(helper_params or []):
                helper_param_name = getattr(helper_param, "name", None)
                if helper_param_name not in helper_roles or index >= len(args):
                    continue
                root_name = self.slang_expression_root_identifier(args[index])
                root_name = self.slang_resolve_alias_root(root_name, alias_roots)
                if root_name in parameter_names and root_name not in roles:
                    roles[root_name] = helper_roles[helper_param_name]

        return roles

    def slang_expression_mapped_base_and_array_suffix(self, expr):
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return None, ""
        expr_type = self.reference_referent_type_name(expr_type) or expr_type
        mapped_type = self.convert_type(expr_type)
        return split_array_type_suffix(mapped_type)

    def slang_expression_is_lvalue(self, expr):
        if isinstance(expr, (VariableNode, IdentifierNode)):
            return True
        if isinstance(expr, UnaryOpNode):
            if getattr(expr, "op", None) == "*":
                return (
                    self.pointer_pointee_type_name(
                        self.expression_result_type(getattr(expr, "operand", None))
                    )
                    is not None
                )
            return False
        if isinstance(expr, ArrayAccessNode):
            return self.slang_expression_is_lvalue(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        if isinstance(expr, MemberAccessNode):
            return self.slang_expression_is_lvalue(
                getattr(expr, "object", getattr(expr, "object_expr", None))
            )
        if isinstance(expr, PointerAccessNode):
            return self.slang_expression_is_lvalue(getattr(expr, "pointer_expr", None))
        return False

    def validate_slang_ray_lvalue_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if self.slang_expression_is_lvalue(argument):
            return
        actual_type = f"{base_type}{array_suffix}"
        raise SlangRefusal(
            f"Slang {shader_type} {operation} {role} argument must be "
            f"an lvalue, got {actual_type}"
        )

    def validate_slang_ray_address_of_lvalue_argument(
        self, argument, shader_type, operation, role
    ):
        if not (
            isinstance(argument, UnaryOpNode)
            and getattr(argument, "op", None) == "&"
            and not getattr(argument, "is_postfix", False)
        ):
            return

        operand = getattr(argument, "operand", None)
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            operand
        )
        if base_type is None:
            return
        if self.slang_expression_is_lvalue(operand):
            return
        actual_type = f"{base_type}{array_suffix}"
        raise SlangRefusal(
            f"Slang {shader_type} {operation} {role} argument must take the "
            f"address of an lvalue, got {actual_type}"
        )

    def slang_parameter_requires_lvalue_argument(self, parameter):
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if directions & {"out", "inout"}:
            return True

        param_type = self.slang_parameter_type_name(parameter)
        return self.reference_referent_type_name(param_type) is not None

    def validate_slang_ray_exact_type_argument(
        self, argument, shader_type, operation, role, expected_type
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type != expected_type:
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"{expected_type}, got {actual_type}"
            )

    def validate_slang_ray_scalar_int_uint_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type not in {"int", "uint"}:
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"scalar int or uint, got {actual_type}"
            )

    def validate_slang_ray_scalar_float_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type not in {"float", "double"}:
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"scalar floating, got {actual_type}"
            )

    def validate_slang_ray_value_argument(self, argument, shader_type, operation, role):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if not self.is_slang_ray_interface_value_type(base_type):
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} {operation} {role} argument must use a "
                f"structure, scalar, vector or matrix type, got {actual_type}"
            )

    def validate_slang_call_shader_callable_data_type(self, argument, shader_type):
        if not self.slang_callable_data_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_callable_data_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_callable_data_parameter_types))
        raise SlangRefusal(
            f"Slang {shader_type} CallShader callable data argument type "
            f"{base_type} must match callable data parameter type {expected_label}"
        )

    def validate_slang_trace_ray_payload_type(self, argument, shader_type):
        if not self.slang_ray_payload_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_ray_payload_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_ray_payload_parameter_types))
        raise SlangRefusal(
            f"Slang {shader_type} TraceRay payload argument type {base_type} "
            f"must match ray payload parameter type {expected_label}"
        )

    def validate_slang_report_hit_attribute_type(self, argument, shader_type):
        if not self.slang_hit_attribute_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_hit_attribute_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_hit_attribute_parameter_types))
        raise SlangRefusal(
            f"Slang {shader_type} ReportHit hit attribute argument type {base_type} "
            f"must match hit attribute parameter type {expected_label}"
        )

    def validate_slang_trace_ray_arguments(self, args, shader_type):
        self.validate_slang_ray_exact_type_argument(
            args[0],
            shader_type,
            "TraceRay",
            "acceleration structure",
            "RaytracingAccelerationStructure",
        )
        for index, role in (
            (1, "ray flags"),
            (2, "instance inclusion mask"),
            (3, "ray contribution to hit group index"),
            (4, "geometry contribution multiplier"),
            (5, "miss shader index"),
        ):
            self.validate_slang_ray_scalar_int_uint_argument(
                args[index], shader_type, "TraceRay", role
            )
        if len(args) == 8:
            self.validate_slang_ray_exact_type_argument(
                args[6], shader_type, "TraceRay", "ray descriptor", "RayDesc"
            )
        else:
            self.validate_slang_ray_exact_type_argument(
                args[6], shader_type, "TraceRay", "origin", "float3"
            )
            self.validate_slang_ray_scalar_float_argument(
                args[7], shader_type, "TraceRay", "minimum distance"
            )
            self.validate_slang_ray_exact_type_argument(
                args[8], shader_type, "TraceRay", "direction", "float3"
            )
            self.validate_slang_ray_scalar_float_argument(
                args[9], shader_type, "TraceRay", "maximum distance"
            )
        self.validate_slang_ray_value_argument(
            args[-1], shader_type, "TraceRay", "payload"
        )
        self.validate_slang_trace_ray_payload_type(args[-1], shader_type)
        self.validate_slang_ray_lvalue_argument(
            args[-1], shader_type, "TraceRay", "payload"
        )

    def validate_slang_ray_tracing_call_arguments(self, operation, args, shader_type):
        if operation == "TraceRay":
            self.validate_slang_trace_ray_arguments(args, shader_type)
        elif operation == "CallShader":
            self.validate_slang_ray_scalar_int_uint_argument(
                args[0], shader_type, "CallShader", "shader index"
            )
            self.validate_slang_ray_value_argument(
                args[1], shader_type, "CallShader", "callable data"
            )
            self.validate_slang_call_shader_callable_data_type(args[1], shader_type)
            self.validate_slang_ray_lvalue_argument(
                args[1], shader_type, "CallShader", "callable data"
            )
        elif operation == "ReportHit":
            self.validate_slang_ray_scalar_float_argument(
                args[0], shader_type, "ReportHit", "hit distance"
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[1], shader_type, "ReportHit", "hit kind"
            )
            if len(args) == 3:
                self.validate_slang_ray_value_argument(
                    args[2], shader_type, "ReportHit", "hit attribute"
                )
                self.validate_slang_report_hit_attribute_type(args[2], shader_type)

    def validate_slang_ray_helper_call_arguments(self, call_node, shader_type):
        helper_name = self.slang_function_call_name(call_node)
        helper_func = self.user_functions_by_name.get(helper_name)
        if helper_func is None:
            return

        roles = self.slang_ray_interface_parameter_roles(helper_func)
        if not roles:
            return

        parameters = getattr(
            helper_func, "parameters", getattr(helper_func, "params", [])
        )
        args = getattr(call_node, "arguments", getattr(call_node, "args", []))
        for index, parameter in enumerate(parameters or []):
            param_name = getattr(parameter, "name", None)
            if param_name not in roles or index >= len(args):
                continue
            role = roles[param_name]
            expected_type = self.slang_parameter_type_name(parameter)
            if not expected_type:
                continue
            expected_compare_type = (
                self.reference_referent_type_name(expected_type) or expected_type
            )
            expected_base, expected_suffix = split_array_type_suffix(
                self.convert_type(expected_compare_type)
            )
            actual_base, actual_suffix = (
                self.slang_expression_mapped_base_and_array_suffix(args[index])
            )
            if actual_base is None:
                continue
            if actual_base == expected_base and actual_suffix == expected_suffix:
                self.validate_slang_ray_address_of_lvalue_argument(
                    args[index], shader_type, helper_name, role
                )
                if role in {
                    "payload",
                    "callable data",
                } and self.slang_parameter_requires_lvalue_argument(parameter):
                    self.validate_slang_ray_lvalue_argument(
                        args[index], shader_type, helper_name, role
                    )
                continue
            actual_type = f"{actual_base}{actual_suffix}"
            expected_label = self.convert_type(expected_type)
            raise SlangRefusal(
                f"Slang {shader_type} {helper_name} {role} "
                f"argument type {actual_type} must match parameter type "
                f"{expected_label}"
            )

    def validate_slang_ray_tracing_calls(
        self, func, shader_type, parameters, visited_helpers=None
    ):
        calls = self.slang_ray_tracing_calls(func)
        helper_call_nodes = self.slang_user_function_call_nodes(func)
        if not calls and not helper_call_nodes:
            return
        if visited_helpers is None:
            visited_helpers = set()

        shader_stage = self.slang_shader_stage_name(shader_type)
        allowed_stages = {
            "TraceRay": {"raygeneration", "closesthit", "miss"},
            "CallShader": {"raygeneration", "closesthit", "miss", "callable"},
            "ReportHit": {"intersection"},
            "AcceptHitAndEndSearch": {"anyhit"},
            "IgnoreHit": {"anyhit"},
        }
        expected_arg_counts = {
            "TraceRay": {8, 11},
            "CallShader": {2},
            "ReportHit": {2, 3},
            "AcceptHitAndEndSearch": {0},
            "IgnoreHit": {0},
        }
        saved_variable_types = self.variable_types.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        try:
            for parameter in parameters or []:
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for operation, args in calls:
                if operation not in allowed_stages:
                    continue
                if shader_stage not in allowed_stages[operation]:
                    valid_stages = ", ".join(sorted(allowed_stages[operation]))
                    raise SlangRefusal(
                        f"Slang {shader_type} stage cannot call {operation}; "
                        f"{operation} is only valid in: {valid_stages}"
                    )
                expected_counts = expected_arg_counts[operation]
                if len(args) not in expected_counts:
                    expected = " or ".join(
                        str(count) for count in sorted(expected_counts)
                    )
                    raise SlangRefusal(
                        f"Slang {shader_type} {operation} requires {expected} "
                        f"argument(s), got {len(args)}"
                    )
                self.validate_slang_ray_tracing_call_arguments(
                    operation, args, shader_type
                )
            for helper_call in helper_call_nodes:
                self.validate_slang_ray_helper_call_arguments(helper_call, shader_type)

            for helper_call in helper_call_nodes:
                helper_name = self.slang_function_call_name(helper_call)
                if helper_name in visited_helpers:
                    continue
                helper_func = self.user_functions_by_name.get(helper_name)
                if helper_func is None or helper_func is func:
                    continue
                helper_params = getattr(
                    helper_func, "parameters", getattr(helper_func, "params", [])
                )
                self.validate_slang_ray_tracing_calls(
                    helper_func,
                    shader_type,
                    helper_params,
                    visited_helpers | {helper_name},
                )
        finally:
            self.variable_types = saved_variable_types
            self.buffer_resource_types = saved_buffer_resource_types
            self.buffer_resource_accesses = saved_buffer_resource_accesses

    def slang_ray_tracing_result_type(self, operation):
        return {
            "TraceRay": "void",
            "CallShader": "void",
            "ReportHit": "bool",
            "AcceptHitAndEndSearch": "void",
            "IgnoreHit": "void",
        }.get(operation)

    def slang_ray_transform_call_type(self, func_name, args):
        """The matrix type of a DXR transform call such as ``ObjectToWorld3x4()``, or ``None``.

        Slang declares these as HLSL does (DXR spec, "ObjectToWorld3x4"): the unsuffixed names are aliases of the
        ``3x4`` ones. Knowing the type lets ``ObjectToWorld3x4() * p`` become ``mul(...)`` and an element index on
        the call swap to Slang's row-first order, as they do on a local of the same type.
        """
        if args or not isinstance(func_name, str):
            return None
        if func_name in self.user_function_names:
            return None
        return {
            "ObjectToWorld": "float3x4",
            "ObjectToWorld3x4": "float3x4",
            "ObjectToWorld4x3": "float4x3",
            "WorldToObject": "float3x4",
            "WorldToObject3x4": "float3x4",
            "WorldToObject4x3": "float4x3",
        }.get(func_name)

    def slang_ray_tracing_expected_target_type(self):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            return None
        expected_type = self.convert_type(expected_type)
        if expected_type == "auto":
            return None
        return expected_type

    def slang_ray_tracing_target_type_rejection_reason(self, result_type):
        expected_type = self.slang_ray_tracing_expected_target_type()
        result_type = self.type_name_string(result_type)
        if result_type:
            result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if result_type == "void":
            return f"returns void but target expects {expected_type}"
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def slang_ray_tracing_diagnostic_expression(self, operation, reason):
        expected_type = self.slang_ray_tracing_expected_target_type()
        fallback = self.zero_value_for_type(expected_type) if expected_type else "0"
        return self.refuse_lowering(
            f"Slang ray tracing intrinsic: {operation} {reason}", fallback
        )

    def slang_ray_query_method_return_type(self, operation):
        return {
            "Proceed": "bool",
            "Abort": "void",
            "TraceRayInline": "void",
            "CommitNonOpaqueTriangleHit": "void",
            "CommitProceduralPrimitiveHit": "void",
            "CandidateType": "uint",
            "CommittedStatus": "uint",
            "CandidatePrimitiveIndex": "uint",
            "CommittedPrimitiveIndex": "uint",
            "CandidateInstanceID": "uint",
            "CommittedInstanceID": "uint",
            "CandidateInstanceIndex": "uint",
            "CommittedInstanceIndex": "uint",
            "CandidateGeometryIndex": "uint",
            "CommittedGeometryIndex": "uint",
            "CandidateObjectRayOrigin": "float3",
            "CandidateObjectRayDirection": "float3",
            "CommittedObjectRayOrigin": "float3",
            "CommittedObjectRayDirection": "float3",
            "CandidateTriangleRayT": "float",
            "CommittedRayT": "float",
            "CandidateObjectRayTMin": "float",
            "CandidateTriangleBarycentrics": "float2",
            "CommittedTriangleBarycentrics": "float2",
            "CandidateTriangleFrontFace": "bool",
            "CommittedTriangleFrontFace": "bool",
            "CandidateObjectToWorld3x4": "float3x4",
            "CandidateWorldToObject3x4": "float3x4",
            "CommittedObjectToWorld3x4": "float3x4",
            "CommittedWorldToObject3x4": "float3x4",
        }.get(operation)

    def slang_ray_query_method_names(self):
        return {
            "Proceed",
            "Abort",
            "TraceRayInline",
            "CommitNonOpaqueTriangleHit",
            "CommitProceduralPrimitiveHit",
            "CandidateType",
            "CommittedStatus",
            "CandidatePrimitiveIndex",
            "CommittedPrimitiveIndex",
            "CandidateInstanceID",
            "CommittedInstanceID",
            "CandidateInstanceIndex",
            "CommittedInstanceIndex",
            "CandidateGeometryIndex",
            "CommittedGeometryIndex",
            "CandidateObjectRayOrigin",
            "CandidateObjectRayDirection",
            "CommittedObjectRayOrigin",
            "CommittedObjectRayDirection",
            "CandidateTriangleRayT",
            "CommittedRayT",
            "CandidateObjectRayTMin",
            "CandidateTriangleBarycentrics",
            "CommittedTriangleBarycentrics",
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
            "CandidateObjectToWorld3x4",
            "CandidateWorldToObject3x4",
            "CommittedObjectToWorld3x4",
            "CommittedWorldToObject3x4",
        }

    def generate_slang_ray_query_expression(self, operation, query_expr, args):
        result_type = self.slang_ray_query_method_return_type(operation)
        target_reason = self.slang_ray_query_target_type_rejection_reason(result_type)
        if target_reason is not None:
            return self.slang_ray_query_diagnostic_expression(operation, target_reason)

        query = self.generate_expression(query_expr)
        arg_list = ", ".join(self.generate_expression(arg) for arg in args)
        access = (
            "->"
            if self.slang_ray_query_pointer_pointee_type(query_expr) is not None
            else "."
        )
        return f"{query}{access}{operation}({arg_list})"

    def slang_ray_query_target_type_rejection_reason(self, result_type):
        expected_type = self.slang_ray_query_expected_target_type()
        result_type = self.type_name_string(result_type)
        if result_type:
            result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if result_type == "void":
            return f"returns void but target expects {expected_type}"
        if not self.is_slang_ray_query_value_type(result_type):
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def slang_ray_query_expected_target_type(self):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            return None
        expected_type = self.convert_type(expected_type)
        if expected_type == "auto":
            return None
        if self.is_slang_ray_query_value_type(expected_type):
            return expected_type
        return None

    def is_slang_ray_query_value_type(self, type_name):
        return (
            self.is_scalar_value_type(type_name)
            or self.is_vector_value_type(type_name)
            or self.is_matrix_value_type(type_name)
        )

    def slang_ray_query_diagnostic_expression(self, operation, reason):
        expected_type = self.slang_ray_query_expected_target_type()
        fallback = self.zero_value_for_type(expected_type) if expected_type else "0"
        return self.refuse_lowering(f"Slang RayQuery: {operation} {reason}", fallback)

    def slang_ray_query_call_parts(self, node):
        if isinstance(node, RayQueryOpNode):
            return (
                getattr(node, "operation", None),
                getattr(node, "query_expr", None),
                getattr(node, "arguments", []),
            )

        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if isinstance(func_expr, PointerAccessNode):
            operation = str(getattr(func_expr, "member", ""))
            if operation not in self.slang_ray_query_method_names():
                return None
            return (
                operation,
                getattr(func_expr, "pointer_expr", None),
                getattr(node, "arguments", getattr(node, "args", [])),
            )

        if not isinstance(func_expr, MemberAccessNode):
            return None

        operation = str(getattr(func_expr, "member", ""))
        if operation not in self.slang_ray_query_method_names():
            return None
        return (
            operation,
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None)),
            getattr(node, "arguments", getattr(node, "args", [])),
        )

    def slang_ray_query_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call = self.slang_ray_query_call_parts(node)
            if call is not None:
                calls.append(call)
        return calls

    def slang_ray_query_pointer_pointee_type(self, query_expr):
        pointer_type = self.expression_result_type(query_expr)
        pointee_type = self.pointer_pointee_type_name(pointer_type)
        if pointee_type is None:
            return None
        return self.reference_referent_type_name(pointee_type) or pointee_type

    def is_slang_ray_query_receiver_type(self, base_type, array_suffix):
        return not array_suffix and (
            base_type == "RayQuery"
            or (base_type.startswith("RayQuery<") and base_type.endswith(">"))
        )

    def validate_slang_ray_query_receiver(self, query_expr, shader_type, operation):
        pointee_type = self.slang_ray_query_pointer_pointee_type(query_expr)
        if pointee_type is not None:
            mapped_pointee = self.convert_type(pointee_type)
            base_type, array_suffix = split_array_type_suffix(mapped_pointee)
            if self.is_slang_ray_query_receiver_type(base_type, array_suffix):
                return
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery, got {actual_type}"
            )

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            query_expr
        )
        if base_type is None:
            return
        if not self.is_slang_ray_query_receiver_type(base_type, array_suffix):
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery, got {actual_type}"
            )
        if not self.slang_expression_is_lvalue(query_expr):
            actual_type = f"{base_type}{array_suffix}"
            raise SlangRefusal(
                f"Slang {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery lvalue, got {actual_type}"
            )

    def validate_slang_ray_query_call_arguments(
        self, operation, query_expr, args, shader_type
    ):
        self.validate_slang_ray_query_receiver(query_expr, shader_type, operation)

        expected_arg_counts = {
            "TraceRayInline": {4},
            "CommitProceduralPrimitiveHit": {1},
        }
        expected_counts = expected_arg_counts.get(operation, {0})
        if len(args) not in expected_counts:
            expected = " or ".join(str(count) for count in sorted(expected_counts))
            raise SlangRefusal(
                f"Slang {shader_type} RayQuery.{operation} requires {expected} "
                f"argument(s), got {len(args)}"
            )

        if operation == "TraceRayInline":
            self.validate_slang_ray_exact_type_argument(
                args[0],
                shader_type,
                "RayQuery.TraceRayInline",
                "acceleration structure",
                "RaytracingAccelerationStructure",
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[1], shader_type, "RayQuery.TraceRayInline", "ray flags"
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[2],
                shader_type,
                "RayQuery.TraceRayInline",
                "instance inclusion mask",
            )
            self.validate_slang_ray_exact_type_argument(
                args[3],
                shader_type,
                "RayQuery.TraceRayInline",
                "ray descriptor",
                "RayDesc",
            )
        elif operation == "CommitProceduralPrimitiveHit":
            self.validate_slang_ray_scalar_float_argument(
                args[0],
                shader_type,
                "RayQuery.CommitProceduralPrimitiveHit",
                "hit distance",
            )

    def validate_slang_ray_query_calls(self, func, shader_type, parameters):
        calls = self.slang_ray_query_calls(func)
        if not calls:
            return

        saved_variable_types = self.variable_types.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        try:
            for parameter in parameters or []:
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for operation, query_expr, args in calls:
                if operation not in self.slang_ray_query_method_names():
                    continue
                self.validate_slang_ray_query_call_arguments(
                    operation, query_expr, args, shader_type
                )
        finally:
            self.variable_types = saved_variable_types
            self.buffer_resource_types = saved_buffer_resource_types
            self.buffer_resource_accesses = saved_buffer_resource_accesses

    def slang_ray_stage_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage not in self.slang_ray_stage_types():
            return None

        role = self.slang_ray_semantic_role(parameter, shader_type)
        if role is None:
            return None

        qualifier = "in" if role == "hit_attribute" else "inout"
        return f"{qualifier} {declaration}"

    def slang_mesh_payload_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        if self.slang_shader_stage_name(shader_type) != "mesh":
            return None
        if not self.is_slang_mesh_payload_parameter(parameter):
            return None

        return (
            f"{self.slang_parameter_qualifier_prefix(parameter)}payload {declaration}"
        )
