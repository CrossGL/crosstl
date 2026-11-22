"""Mesh dispatch and groupshared payloads in the HLSL generator: function declarations, output payload parameters, helper counts, group constants and workgroup statements."""

import re

from ..ast import (
    ArrayAccessNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MeshOpNode,
    PointerType,
    ReturnNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    format_c_style_array_declaration,
    parse_array_type,
    split_array_type_suffix,
)
from .directx_values import (
    DirectXMeshOutputCountsError,
    DirectXWorkgroupPointerError,
)
from .stage_utils import (
    collect_stage_local_variables,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class DirectXMeshDispatchMixin:
    """Mesh dispatch members of ``HLSLCodeGen``."""

    def validate_hlsl_mesh_output_write_sequence_result(
        self,
        statements,
        set_mesh_output_counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        active_helper_calls=None,
        loop_exit_nodes=None,
        visible_bool_constants=None,
        visible_int_constants=None,
        role_parameters=None,
    ):
        if active_helper_calls is None:
            active_helper_calls = set()
        if loop_exit_nodes is None:
            loop_exit_nodes = ()
        if visible_bool_constants is None:
            visible_bool_constants = dict(self.literal_bool_constants)
        if visible_int_constants is None:
            visible_int_constants = dict(self.literal_int_constants)

        counts_seen = set_mesh_output_counts_seen
        for stmt in statements:
            if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                counts_seen, can_continue = (
                    self.validate_hlsl_mesh_output_write_sequence_result(
                        self.hlsl_statement_body_items(stmt),
                        counts_seen,
                        role_by_name,
                        declared_counts,
                        set_count_literals,
                        active_helper_calls,
                        loop_exit_nodes,
                        visible_bool_constants=dict(visible_bool_constants),
                        visible_int_constants=dict(visible_int_constants),
                        role_parameters=role_parameters,
                    )
                )
                if not can_continue:
                    return counts_seen, False
                continue

            if isinstance(stmt, IfNode):
                counts_seen, can_continue = (
                    self.validate_hlsl_mesh_output_write_if_result(
                        stmt,
                        counts_seen,
                        role_by_name,
                        declared_counts,
                        set_count_literals,
                        active_helper_calls,
                        loop_exit_nodes,
                        visible_bool_constants,
                        visible_int_constants,
                        role_parameters,
                    )
                )
                if not can_continue:
                    return counts_seen, False
                continue

            if isinstance(stmt, SwitchNode):
                counts_seen, can_continue = (
                    self.validate_hlsl_mesh_output_write_switch_result(
                        stmt,
                        counts_seen,
                        role_by_name,
                        declared_counts,
                        set_count_literals,
                        active_helper_calls,
                        loop_exit_nodes,
                        visible_bool_constants,
                        visible_int_constants,
                        role_parameters,
                    )
                )
                if not can_continue:
                    return counts_seen, False
                continue

            if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
                self.validate_hlsl_mesh_output_write_sequence_result(
                    self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                    counts_seen,
                    role_by_name,
                    declared_counts,
                    set_count_literals,
                    active_helper_calls,
                    (BreakNode, ContinueNode),
                    visible_bool_constants=dict(visible_bool_constants),
                    visible_int_constants=dict(visible_int_constants),
                    role_parameters=role_parameters,
                )
                continue

            assignment = self.hlsl_assignment_from_statement(stmt)
            if assignment is not None:
                self.validate_hlsl_mesh_output_assignment(
                    assignment,
                    counts_seen,
                    role_by_name,
                    declared_counts,
                    set_count_literals,
                    visible_int_constants,
                )

            for node in self.walk_ast(stmt):
                if not isinstance(node, FunctionCallNode):
                    continue
                self.validate_hlsl_mesh_output_helper_call(
                    node,
                    counts_seen,
                    role_by_name,
                    declared_counts,
                    set_count_literals,
                    active_helper_calls,
                    role_parameters,
                )

            count_calls = self.hlsl_set_mesh_output_count_args_in_node(stmt)
            if count_calls:
                if role_parameters is not None:
                    for args in count_calls:
                        self.hlsl_update_set_mesh_output_count_literals(
                            args,
                            role_parameters,
                            set_count_literals,
                            visible_int_constants,
                        )
                counts_seen = True

            self.hlsl_update_visible_bool_constants(stmt, visible_bool_constants)
            self.hlsl_update_visible_int_constants(stmt, visible_int_constants)

            if isinstance(stmt, ReturnNode):
                return counts_seen, False
            if loop_exit_nodes and isinstance(stmt, loop_exit_nodes):
                return counts_seen, False

        return counts_seen, True

    def validate_hlsl_mesh_output_write_if_result(
        self,
        if_node,
        counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        active_helper_calls,
        loop_exit_nodes=None,
        visible_bool_constants=None,
        visible_int_constants=None,
        role_parameters=None,
    ):
        if loop_exit_nodes is None:
            loop_exit_nodes = ()
        if visible_bool_constants is None:
            visible_bool_constants = dict(self.literal_bool_constants)
        if visible_int_constants is None:
            visible_int_constants = dict(self.literal_int_constants)

        then_statements = self.hlsl_statement_body_items(
            getattr(if_node, "then_branch", getattr(if_node, "if_body", None))
        )
        else_branch = getattr(
            if_node, "else_branch", getattr(if_node, "else_body", None)
        )
        condition_value = self.hlsl_bool_constant_value(
            getattr(if_node, "condition", getattr(if_node, "if_condition", None)),
            visible_bool_constants,
        )
        if condition_value is True:
            return self.validate_hlsl_mesh_output_write_sequence_result(
                then_statements,
                counts_seen,
                role_by_name,
                declared_counts,
                set_count_literals,
                active_helper_calls,
                loop_exit_nodes,
                visible_bool_constants=dict(visible_bool_constants),
                visible_int_constants=dict(visible_int_constants),
                role_parameters=role_parameters,
            )

        if condition_value is False:
            if else_branch is None:
                return counts_seen, True
            return self.validate_hlsl_mesh_output_write_sequence_result(
                self.hlsl_statement_body_items(else_branch),
                counts_seen,
                role_by_name,
                declared_counts,
                set_count_literals,
                active_helper_calls,
                loop_exit_nodes,
                visible_bool_constants=dict(visible_bool_constants),
                visible_int_constants=dict(visible_int_constants),
                role_parameters=role_parameters,
            )

        branch_results = [
            self.validate_hlsl_mesh_output_write_sequence_result(
                then_statements,
                counts_seen,
                role_by_name,
                declared_counts,
                set_count_literals,
                active_helper_calls,
                loop_exit_nodes,
                visible_bool_constants=dict(visible_bool_constants),
                visible_int_constants=dict(visible_int_constants),
                role_parameters=role_parameters,
            )
        ]
        if else_branch is not None:
            branch_results.append(
                self.validate_hlsl_mesh_output_write_sequence_result(
                    self.hlsl_statement_body_items(else_branch),
                    counts_seen,
                    role_by_name,
                    declared_counts,
                    set_count_literals,
                    active_helper_calls,
                    loop_exit_nodes,
                    visible_bool_constants=dict(visible_bool_constants),
                    visible_int_constants=dict(visible_int_constants),
                    role_parameters=role_parameters,
                )
            )
        else:
            branch_results.append((counts_seen, True))

        continuing_counts = [
            branch_counts
            for branch_counts, can_continue in branch_results
            if can_continue
        ]
        if not continuing_counts:
            return counts_seen, False

        return all(continuing_counts), True

    def validate_hlsl_mesh_output_writes(self, func, role_parameters):
        role_by_name = self.hlsl_mesh_output_role_by_parameter_name(role_parameters)
        declared_counts = {
            role: self.hlsl_mesh_output_parameter_literal_count(parameters[0])
            for role, parameters in role_parameters.items()
            if role in {"vertices", "indices", "primitives"}
        }

        set_count_literals = {}
        calls = self.hlsl_set_mesh_output_count_calls(func)
        if calls and len(calls[0]) == 2:
            vertex_count = self.hlsl_integer_constant_value(calls[0][0])
            primitive_count = self.hlsl_integer_constant_value(calls[0][1])
            set_count_literals = {
                "vertices": vertex_count,
                "indices": primitive_count,
                "primitives": primitive_count,
            }

        self.validate_hlsl_mesh_output_write_sequence(
            self.hlsl_statement_body_items(getattr(func, "body", [])),
            False,
            role_by_name,
            declared_counts,
            set_count_literals,
            visible_bool_constants=self.hlsl_initial_bool_constants(func),
            visible_int_constants=self.hlsl_initial_int_constants(func),
            role_parameters=role_parameters,
        )

    def validate_hlsl_mesh_output_parameters(self, func, parameters):
        role_parameters = {}
        for parameter in parameters:
            roles = self.hlsl_mesh_parameter_roles(parameter)
            if len(roles) > 1:
                raise DirectXRefusal(
                    f"DirectX mesh stage parameter '{parameter.name}' "
                    "can use only one mesh role qualifier"
                )
            if roles:
                role_parameters.setdefault(roles[0], []).append(parameter)

        mesh_output_roles = {"vertices", "indices", "primitives"}
        if not (set(role_parameters) & mesh_output_roles):
            if self.hlsl_set_mesh_output_count_calls(func):
                raise DirectXRefusal(
                    "DirectX mesh SetMeshOutputCounts requires mesh output "
                    "vertices and indices parameters"
                )
            return

        for role, role_params in role_parameters.items():
            if role not in mesh_output_roles:
                continue
            if len(role_params) > 1:
                raise DirectXRefusal(
                    f"DirectX mesh stage must declare at most one {role} "
                    "output parameter"
                )
            self.validate_hlsl_mesh_output_array_parameter(role_params[0], role)

        if "vertices" not in role_parameters:
            raise DirectXRefusal(
                "DirectX mesh stage output signature must declare an out vertices array"
            )
        if "indices" not in role_parameters:
            raise DirectXRefusal(
                "DirectX mesh stage output signature must declare an out indices array"
            )

        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        expected_index_types = {
            "line": "uint2",
            "triangle": "uint3",
        }
        expected_index_type = expected_index_types.get(topology)
        if expected_index_type is not None:
            index_param = role_parameters["indices"][0]
            index_base_type, _array_suffix = (
                self.hlsl_parameter_mapped_base_and_array_suffix(index_param)
            )
            if index_base_type != expected_index_type:
                raise DirectXRefusal(
                    f"DirectX mesh stage outputtopology '{topology}' requires "
                    f"indices parameter '{index_param.name}' to use "
                    f"{expected_index_type}, got {index_base_type}"
                )

        self.validate_hlsl_mesh_output_struct(
            role_parameters["vertices"][0], "vertices"
        )
        if "primitives" in role_parameters:
            self.validate_hlsl_mesh_output_struct(
                role_parameters["primitives"][0], "primitives"
            )

            index_count = self.hlsl_mesh_output_parameter_literal_count(
                role_parameters["indices"][0]
            )
            primitive_count = self.hlsl_mesh_output_parameter_literal_count(
                role_parameters["primitives"][0]
            )
            if (
                index_count is not None
                and primitive_count is not None
                and index_count != primitive_count
            ):
                raise DirectXRefusal(
                    "DirectX mesh stage primitives output array size must match "
                    "the indices output array size"
                )

        output_count_calls = self.hlsl_mesh_output_call_count(func)
        if output_count_calls != 1:
            raise DirectXMeshOutputCountsError(
                "DirectX mesh stage output signature must call "
                "SetMeshOutputCounts exactly once"
            )
        self.validate_hlsl_set_mesh_output_counts(func, role_parameters)
        self.validate_hlsl_mesh_output_writes(func, role_parameters)

    def hlsl_mesh_payload_parameters(self, parameters):
        return [
            parameter
            for parameter in parameters
            if "payload" in self.hlsl_mesh_parameter_roles(parameter)
        ]

    def hlsl_mesh_payload_type_from_parameters(self, parameters):
        payload_parameters = self.hlsl_mesh_payload_parameters(parameters)
        if len(payload_parameters) != 1:
            return None
        return self.hlsl_parameter_user_struct_type(payload_parameters[0])

    def validate_hlsl_mesh_payload_parameter(self, func, parameters):
        payload_parameters = self.hlsl_mesh_payload_parameters(parameters)
        if not payload_parameters:
            return
        if len(payload_parameters) > 1:
            raise DirectXRefusal(
                "DirectX mesh stage must declare at most one mesh payload parameter"
            )

        parameter = payload_parameters[0]
        directions = self.hlsl_parameter_direction_qualifiers(parameter)
        if "out" in directions or "inout" in directions:
            raise DirectXRefusal(
                f"DirectX mesh stage payload parameter '{parameter.name}' "
                "must be an input parameter"
            )
        if "in" not in directions:
            raise DirectXRefusal(
                f"DirectX mesh stage payload parameter '{parameter.name}' "
                "must use the in qualifier"
            )

        payload_type = self.hlsl_parameter_user_struct_type(parameter)
        if payload_type is None:
            raise DirectXRefusal(
                f"DirectX mesh stage payload parameter '{parameter.name}' "
                "must use a user-defined struct type"
            )

        dispatch_payload_types = {
            payload
            for payload in self.current_hlsl_dispatch_mesh_payload_types
            if payload is not None
        }
        if dispatch_payload_types and payload_type not in dispatch_payload_types:
            expected = ", ".join(sorted(dispatch_payload_types))
            raise DirectXRefusal(
                "DirectX mesh stage payload parameter type "
                f"'{payload_type}' must match DispatchMesh payload type(s): "
                f"{expected}"
            )
        if (
            self.current_hlsl_has_amplification_stage
            and None in self.current_hlsl_dispatch_mesh_payload_types
        ):
            raise DirectXRefusal(
                "DirectX mesh stage payload parameter requires amplification "
                "DispatchMesh calls to pass a payload argument"
            )

    def hlsl_function_visible_variable_types(self, func):
        variable_types = dict(getattr(self, "global_variable_types", {}) or {})
        for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []:
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            variable_types[parameter.name] = self.type_name_string(parameter_type)

        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            variable_type = self.local_variable_declared_type(node)
            variable_types[node.name] = self.type_name_string(variable_type)
        return variable_types

    def hlsl_expression_user_struct_type(self, expr, variable_types):
        name = self.expression_name(expr)
        if not name:
            return None
        type_name = variable_types.get(name)
        if not type_name:
            return None
        base_type = self.map_type(type_name).split("<", 1)[0].split("[", 1)[0].strip()
        if base_type not in self.structs_by_name:
            return None
        return base_type

    def hlsl_dispatch_mesh_calls(self, func):
        return self.hlsl_dispatch_mesh_args_in_node(getattr(func, "body", []))

    def hlsl_function_call_names(self, func):
        return list(self.hlsl_function_body_scan(func).call_names)

    def hlsl_function_call_summary(self, func):
        """Call names and ``DispatchMesh`` argument lists of ``func``'s body, from its body scan."""
        scan = self.hlsl_function_body_scan(func)
        return list(scan.call_names), list(scan.dispatch_mesh_args)

    def hlsl_reachable_helper_dispatch_mesh_calls(self, func):
        """``(function, DispatchMesh argument lists)`` for ``func`` and each helper it reaches, in visit order."""
        available_functions = self.current_hlsl_available_functions or {}
        reachable = []
        visited = set()

        def visit(current):
            if current is None:
                return
            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)
            names, dispatch_mesh_args = self.hlsl_function_call_summary(current)
            reachable.append((current, dispatch_mesh_args))

            for name in names:
                helper = available_functions.get(name)
                if helper is None or helper is current:
                    continue
                visit(helper)

        visit(func)
        return reachable

    def hlsl_function_and_reachable_helpers(self, func):
        return [
            current
            for current, _ in self.hlsl_reachable_helper_dispatch_mesh_calls(func)
        ]

    def validate_hlsl_dispatch_mesh_group_count_arguments(
        self, args, shader_type, visible_int_constants=None
    ):
        labels = ("ThreadGroupCountX", "ThreadGroupCountY", "ThreadGroupCountZ")
        literal_counts = []
        for label, argument in zip(labels, args[:3]):
            self.validate_hlsl_scalar_int_uint_expression(
                argument,
                f"DirectX {shader_type} DispatchMesh {label} argument",
            )
            literal_count = self.hlsl_integer_constant_value(
                argument, visible_int_constants
            )
            literal_counts.append(literal_count)
            if literal_count is None:
                continue
            if literal_count < 0:
                raise DirectXRefusal(
                    f"DirectX {shader_type} DispatchMesh {label} argument "
                    "must be non-negative"
                )
            if literal_count >= 65536:
                raise DirectXRefusal(
                    f"DirectX {shader_type} DispatchMesh {label} argument "
                    "must be less than 65536"
                )

        if (
            len(literal_counts) == 3
            and all(count is not None for count in literal_counts)
            and literal_counts[0] * literal_counts[1] * literal_counts[2] > (1 << 22)
        ):
            raise DirectXRefusal(
                f"DirectX {shader_type} DispatchMesh thread group count product "
                "must not exceed 4194304"
            )

    def hlsl_expression_is_dispatch_mesh(self, expr):
        if isinstance(expr, FunctionCallNode):
            return self.function_call_name(expr) == "DispatchMesh"
        if isinstance(expr, MeshOpNode):
            return getattr(expr, "operation", None) == "DispatchMesh"
        return False

    def hlsl_dispatch_mesh_args_in_node(self, root):
        calls = []
        for node in self.walk_ast(root):
            if not self.hlsl_expression_is_dispatch_mesh(node):
                continue
            if isinstance(node, FunctionCallNode):
                calls.append(getattr(node, "arguments", getattr(node, "args", [])))
            elif isinstance(node, MeshOpNode):
                calls.append(getattr(node, "arguments", []))
        return calls

    def hlsl_call_literal_int_parameter_constants(
        self, call, callee, visible_int_constants
    ):
        args = getattr(call, "arguments", getattr(call, "args", [])) or []
        parameters = getattr(callee, "parameters", getattr(callee, "params", [])) or []
        parameter_constants = {}
        for index, parameter in enumerate(parameters):
            if index >= len(args):
                break
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                continue
            value = self.hlsl_integer_constant_value(args[index], visible_int_constants)
            if value is not None:
                parameter_constants[parameter_name] = value
        return parameter_constants

    def validate_hlsl_dispatch_mesh_helper_group_count_arguments(
        self, stmt, shader_type, visible_int_constants, active_helper_calls
    ):
        available_functions = self.current_hlsl_available_functions or {}
        for node in self.walk_ast(stmt):
            if not isinstance(node, FunctionCallNode):
                continue

            helper_name = self.function_call_name(node)
            helper = available_functions.get(helper_name)
            if helper is None or not self.hlsl_function_contains_dispatch_mesh(helper):
                continue

            helper_parameter_constants = self.hlsl_call_literal_int_parameter_constants(
                node, helper, visible_int_constants
            )
            if not helper_parameter_constants:
                continue

            helper_id = id(helper)
            if helper_id in active_helper_calls:
                continue

            helper_visible_int_constants = self.hlsl_initial_int_constants(helper)
            helper_visible_int_constants.update(helper_parameter_constants)
            active_helper_calls.add(helper_id)
            try:
                self.validate_hlsl_dispatch_mesh_group_count_sequence(
                    self.hlsl_statement_body_items(getattr(helper, "body", [])),
                    shader_type,
                    helper_visible_int_constants,
                    active_helper_calls,
                )
            finally:
                active_helper_calls.remove(helper_id)

    def validate_hlsl_dispatch_mesh_group_count_sequence(
        self,
        statements,
        shader_type,
        visible_int_constants=None,
        active_helper_calls=None,
    ):
        if visible_int_constants is None:
            visible_int_constants = dict(self.literal_int_constants)
        if active_helper_calls is None:
            active_helper_calls = set()

        for stmt in self.hlsl_statement_body_items(statements):
            if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                self.validate_hlsl_dispatch_mesh_group_count_sequence(
                    self.hlsl_statement_body_items(stmt),
                    shader_type,
                    dict(visible_int_constants),
                    active_helper_calls,
                )
                continue

            if isinstance(stmt, IfNode):
                self.validate_hlsl_dispatch_mesh_group_count_sequence(
                    self.hlsl_statement_body_items(
                        getattr(stmt, "then_branch", getattr(stmt, "if_body", None))
                    ),
                    shader_type,
                    dict(visible_int_constants),
                    active_helper_calls,
                )
                else_branch = getattr(
                    stmt, "else_branch", getattr(stmt, "else_body", None)
                )
                if else_branch is not None:
                    self.validate_hlsl_dispatch_mesh_group_count_sequence(
                        self.hlsl_statement_body_items(else_branch),
                        shader_type,
                        dict(visible_int_constants),
                        active_helper_calls,
                    )
                continue

            if isinstance(stmt, SwitchNode):
                case_entries = self.hlsl_switch_case_entries(stmt)
                for start_index in range(len(case_entries)):
                    self.validate_hlsl_dispatch_mesh_group_count_sequence(
                        self.hlsl_switch_fallthrough_path_body(
                            case_entries, start_index
                        ),
                        shader_type,
                        dict(visible_int_constants),
                        active_helper_calls,
                    )
                continue

            if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
                self.validate_hlsl_dispatch_mesh_group_count_sequence(
                    self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                    shader_type,
                    dict(visible_int_constants),
                    active_helper_calls,
                )
                continue

            for args in self.hlsl_dispatch_mesh_args_in_node(stmt):
                self.validate_hlsl_dispatch_mesh_group_count_arguments(
                    args, shader_type, visible_int_constants
                )
            self.validate_hlsl_dispatch_mesh_helper_group_count_arguments(
                stmt, shader_type, visible_int_constants, active_helper_calls
            )

            self.hlsl_update_visible_int_constants(stmt, visible_int_constants)

    def hlsl_function_contains_dispatch_mesh(self, func, visited=None):
        if func is None:
            return False
        if visited is None:
            visited = set()
        func_id = id(func)
        if func_id in visited:
            return False
        visited.add(func_id)

        if self.hlsl_statement_contains_direct_dispatch_mesh(getattr(func, "body", [])):
            return True

        available_functions = self.current_hlsl_available_functions or {}
        for name in self.hlsl_function_call_names(func):
            helper = available_functions.get(name)
            if helper is None or helper is func:
                continue
            if self.hlsl_function_contains_dispatch_mesh(helper, visited):
                return True
        return False

    def hlsl_expression_is_dispatch_mesh_or_helper(self, expr):
        if self.hlsl_expression_is_dispatch_mesh(expr):
            return True
        if not isinstance(expr, FunctionCallNode):
            return False
        helper_name = self.function_call_name(expr)
        helper = (self.current_hlsl_available_functions or {}).get(helper_name)
        return self.hlsl_function_contains_dispatch_mesh(helper)

    def hlsl_call_thread_varying_parameter_names(
        self, call, callee, thread_varying_names
    ):
        if not thread_varying_names:
            return set()

        args = getattr(call, "arguments", getattr(call, "args", [])) or []
        parameters = getattr(callee, "parameters", getattr(callee, "params", [])) or []
        tainted_parameters = set()
        for index, parameter in enumerate(parameters):
            if index >= len(args):
                break
            if not self.hlsl_condition_uses_thread_varying_name(
                args[index], thread_varying_names
            ):
                continue
            parameter_name = getattr(parameter, "name", None)
            if parameter_name:
                tainted_parameters.add(parameter_name)
        return tainted_parameters

    def validate_hlsl_dispatch_mesh_helper_parameter_control_flow(
        self, stmt, thread_varying_names, shader_type, visited_helper_taints
    ):
        if not thread_varying_names:
            return

        available_functions = self.current_hlsl_available_functions or {}
        for node in self.walk_ast(stmt):
            if not isinstance(node, FunctionCallNode):
                continue

            helper_name = self.function_call_name(node)
            helper = available_functions.get(helper_name)
            if helper is None or not self.hlsl_function_contains_dispatch_mesh(helper):
                continue

            helper_thread_varying_names = self.hlsl_call_thread_varying_parameter_names(
                node, helper, thread_varying_names
            )
            if not helper_thread_varying_names:
                continue

            visit_key = (id(helper), tuple(sorted(helper_thread_varying_names)))
            if visit_key in visited_helper_taints:
                continue
            visited_helper_taints.add(visit_key)
            self.validate_hlsl_dispatch_mesh_control_flow(
                self.hlsl_statement_body_items(getattr(helper, "body", [])),
                helper_thread_varying_names,
                shader_type,
                visited_helper_taints=visited_helper_taints,
            )

    def hlsl_statement_contains_direct_dispatch_mesh(self, stmt):
        return any(
            self.hlsl_expression_is_dispatch_mesh(node) for node in self.walk_ast(stmt)
        )

    def hlsl_statement_contains_dispatch_mesh(self, stmt):
        return any(
            self.hlsl_expression_is_dispatch_mesh_or_helper(node)
            for node in self.walk_ast(stmt)
        )

    def hlsl_dispatch_mesh_invocation_site_count(self, func):
        return sum(
            1
            for node in self.walk_ast(getattr(func, "body", []))
            if self.hlsl_expression_is_dispatch_mesh_or_helper(node)
        )

    def validate_hlsl_dispatch_mesh_control_flow(
        self,
        statements,
        thread_varying_names,
        shader_type,
        in_loop=False,
        in_thread_varying_branch=False,
        visited_helper_taints=None,
    ):
        if visited_helper_taints is None:
            visited_helper_taints = set()

        current_thread_varying_names = set(thread_varying_names)
        for stmt in statements:
            self.hlsl_update_thread_varying_alias_names(
                stmt, current_thread_varying_names
            )
            contains_dispatch_mesh = self.hlsl_statement_contains_dispatch_mesh(stmt)
            if contains_dispatch_mesh and in_loop:
                raise DirectXRefusal(
                    f"DirectX {shader_type} DispatchMesh must not be called from "
                    "loop control flow"
                )
            if contains_dispatch_mesh and in_thread_varying_branch:
                raise DirectXRefusal(
                    f"DirectX {shader_type} DispatchMesh must not be called from "
                    "thread-varying control flow"
                )
            self.validate_hlsl_dispatch_mesh_helper_parameter_control_flow(
                stmt,
                current_thread_varying_names,
                shader_type,
                visited_helper_taints,
            )

            if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                self.validate_hlsl_dispatch_mesh_control_flow(
                    self.hlsl_statement_body_items(stmt),
                    current_thread_varying_names,
                    shader_type,
                    in_loop,
                    in_thread_varying_branch,
                    visited_helper_taints,
                )
                continue

            if isinstance(stmt, IfNode):
                condition = getattr(
                    stmt, "condition", getattr(stmt, "if_condition", None)
                )
                branch_is_thread_varying = (
                    in_thread_varying_branch
                    or self.hlsl_condition_uses_thread_varying_name(
                        condition, current_thread_varying_names
                    )
                )
                self.validate_hlsl_dispatch_mesh_control_flow(
                    self.hlsl_statement_body_items(
                        getattr(stmt, "then_branch", getattr(stmt, "if_body", None))
                    ),
                    current_thread_varying_names,
                    shader_type,
                    in_loop,
                    branch_is_thread_varying,
                    visited_helper_taints,
                )
                else_branch = getattr(
                    stmt, "else_branch", getattr(stmt, "else_body", None)
                )
                if else_branch is not None:
                    self.validate_hlsl_dispatch_mesh_control_flow(
                        self.hlsl_statement_body_items(else_branch),
                        current_thread_varying_names,
                        shader_type,
                        in_loop,
                        branch_is_thread_varying,
                        visited_helper_taints,
                    )
                continue

            if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
                self.validate_hlsl_dispatch_mesh_control_flow(
                    self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                    current_thread_varying_names,
                    shader_type,
                    True,
                    in_thread_varying_branch,
                    visited_helper_taints,
                )

    def validate_hlsl_dispatch_mesh_placement(self, func, shader_type):
        self.validate_hlsl_dispatch_mesh_control_flow(
            self.hlsl_statement_body_items(getattr(func, "body", [])),
            self.hlsl_thread_varying_mesh_condition_names(func),
            shader_type,
        )

    def validate_hlsl_dispatch_mesh_calls(self, func, shader_type):
        reachable = self.hlsl_reachable_helper_dispatch_mesh_calls(func)
        reachable_functions = [current for current, _ in reachable]
        calls = [
            args for _, dispatch_mesh_args in reachable for args in dispatch_mesh_args
        ]
        if not calls:
            return

        allowed_stages = {"task", "amplification", "object"}
        if shader_type not in allowed_stages:
            if shader_type is not None:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage cannot call DispatchMesh; "
                    "DispatchMesh is only valid in amplification/task/object stages"
                )
            return

        for reachable_func in reachable_functions:
            if self.hlsl_dispatch_mesh_invocation_site_count(reachable_func) > 1:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage must call DispatchMesh at most once"
                )

        for args in calls:
            if len(args) not in {3, 4}:
                raise DirectXRefusal(
                    f"DirectX {shader_type} DispatchMesh requires exactly "
                    "three thread group count arguments and an optional "
                    "mesh payload argument"
                )
        for reachable_func in reachable_functions:
            self.validate_hlsl_dispatch_mesh_group_count_sequence(
                self.hlsl_statement_body_items(getattr(reachable_func, "body", [])),
                shader_type,
                self.hlsl_initial_int_constants(reachable_func),
            )
            self.validate_hlsl_dispatch_mesh_placement(reachable_func, shader_type)

    def hlsl_dispatch_mesh_payload_types_for_function(self, func):
        payload_types = set()
        for reachable_func in self.hlsl_function_and_reachable_helpers(func):
            variable_types = self.hlsl_function_visible_variable_types(reachable_func)
            for args in self.hlsl_dispatch_mesh_calls(reachable_func):
                if len(args) < 4:
                    payload_types.add(None)
                    continue

                payload_type = self.hlsl_expression_user_struct_type(
                    args[3], variable_types
                )
                payload_types.add(payload_type)
        return payload_types

    def hlsl_function_visible_variable_declarations(self, func):
        declarations = {}
        for node in (
            getattr(self, "current_global_resource_declaration_nodes", []) or []
        ):
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if name:
                declarations[name] = node

        for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if getattr(parameter, "name", None):
                declarations[parameter.name] = parameter

        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if name:
                declarations[name] = node
        return declarations

    def hlsl_declaration_has_groupshared_qualifier(self, node):
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        return bool(qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"})

    def hlsl_workgroup_pointer_pointee_type(self, node):
        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        for attribute in getattr(node, "attributes", []) or []:
            name = getattr(attribute, "name", None)
            if name:
                qualifiers.add(str(name).lower())
        address_space = str(getattr(raw_type, "address_space", "") or "").lower()
        if not (
            qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"}
            or address_space in {"groupshared", "shared", "threadgroup", "workgroup"}
        ):
            return None
        if isinstance(raw_type, PointerType):
            return raw_type.pointee_type
        raw_type_name = self.type_name_string(raw_type)
        if isinstance(raw_type_name, str) and raw_type_name.rstrip().endswith("*"):
            return raw_type_name.rstrip()[:-1].strip()
        return None

    def hlsl_workgroup_pointer_element_type(self, node):
        pointee_type = self.hlsl_workgroup_pointer_pointee_type(node)
        if pointee_type is None:
            return None
        pointee_type_name = self.type_name_string(pointee_type)
        if self.is_hlsl_bfloat16_type(pointee_type_name):
            return self.hlsl_bfloat16_scalar_type_name(pointee_type_name)
        return self.map_type(pointee_type_name)

    def hlsl_workgroup_pointer_declaration(self, node, aliases=None):
        if self.hlsl_workgroup_pointer_pointee_type(node) is not None:
            return True
        if aliases is None:
            return False

        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        raw_type_name = self.type_name_string(raw_type)
        if not isinstance(raw_type_name, str) or raw_type_name.strip() != "auto":
            return False
        initial_value = getattr(node, "initial_value", None)
        if isinstance(initial_value, ArrayAccessNode):
            return False
        binding = self.hlsl_static_workgroup_pointer_binding(initial_value, aliases)
        return binding is not None

    def hlsl_groupshared_storage_binding(self, node, root_name):
        declared_type = self.hlsl_groupshared_declaration_type(node)
        declared_type_name = self.type_name_string(declared_type)
        element_type, array_suffix = split_array_type_suffix(
            str(declared_type_name or "")
        )
        if not array_suffix or array_suffix.count("[") != 1:
            return None

        constants = {
            **self.literal_int_constants,
            **self.hlsl_hoisted_groupshared_constants_by_declaration.get(id(node), {}),
        }
        raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
        array_size = self.literal_int_value(getattr(raw_type, "size", None), constants)
        if array_size is None:
            _element_type, array_size = parse_array_type(declared_type_name)
        if (
            not isinstance(array_size, int)
            or isinstance(array_size, bool)
            or array_size <= 0
        ):
            array_size = None

        binding_element_type = (
            self.hlsl_bfloat16_scalar_type_name(element_type)
            if self.is_hlsl_bfloat16_type(element_type)
            else self.map_type(element_type)
        )
        return {
            "kind": "workgroup-pointer",
            "root": root_name,
            "root_kind": "concrete",
            "offset": "0",
            "array_size": array_size,
            "element_type": binding_element_type,
            "source_element_type": element_type,
            "resource_type": "groupshared",
            "access": "read_write",
        }

    def hlsl_emitted_functions_for_groupshared_lowering(self, ast, target_stage=None):
        functions = []
        seen = set()

        def add(func):
            if func is None or id(func) in seen:
                return
            seen.add(id(func))
            functions.append(func)

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", []) or []
            qualifier = (
                qualifiers[0] if qualifiers else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if should_emit_qualified_function(target_stage, qualifier_name):
                add(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            for func in getattr(stage, "local_functions", []) or []:
                add(func)
            add(getattr(stage, "entry_point", None))

        return functions

    def collect_hlsl_function_local_groupshared_declarations(
        self, ast, target_stage=None
    ):
        emitted_stage_groupshared = collect_stage_local_variables(
            ast,
            target_stage,
            self.hlsl_declaration_has_groupshared_qualifier,
        )
        used_names = set(self.global_variable_types)
        used_names.update(
            getattr(node, "name", None)
            for node in emitted_stage_groupshared
            if getattr(node, "name", None)
        )
        used_names.update(self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES)
        declarations = []
        aliases_by_declaration = {}
        function_names_by_declaration = {}
        constants_by_declaration = {}
        declaration_ids = set()
        emitted_functions = self.hlsl_emitted_functions_for_groupshared_lowering(
            ast, target_stage
        )

        scope_names_by_declaration = {}
        for func in emitted_functions:
            declaration_constants = self.collect_hlsl_groupshared_declaration_constants(
                func
            )
            constants_by_declaration.update(declaration_constants)
            scope_names = frozenset(self.function_scope_variable_types(func))
            for declaration_id in declaration_constants:
                scope_names_by_declaration[declaration_id] = scope_names

        self.hlsl_hoisted_groupshared_constants_by_declaration = (
            constants_by_declaration
        )
        self.hlsl_hoisted_groupshared_scope_names_by_declaration = (
            scope_names_by_declaration
        )
        global_aliases = {}
        for node in emitted_stage_groupshared:
            name = getattr(node, "name", None)
            if not name:
                continue
            if self.hlsl_workgroup_pointer_declaration(node):
                raise DirectXWorkgroupPointerError(
                    "DirectX cannot declare a module-scope workgroup pointer",
                    parameter_name=name,
                    reason="pointer-storage-declaration",
                    source_location=getattr(node, "source_location", None),
                )
            binding = self.hlsl_groupshared_storage_binding(node, name)
            if binding is not None:
                global_aliases[name] = binding

        bindings_by_declaration = {}
        for func in emitted_functions:
            func_name = getattr(func, "name", None) or "shader"
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, VariableNode):
                    continue
                if not self.hlsl_declaration_has_groupshared_qualifier(node):
                    continue
                if self.hlsl_workgroup_pointer_declaration(node):
                    continue

                name = getattr(node, "name", None)
                if not name:
                    continue

                alias_base = f"{func_name}_{name}"
                alias = self.hlsl_unique_local_identifier(alias_base, used_names)
                used_names.add(alias)
                declared_type_text = self.hlsl_groupshared_declaration_type(node)
                self.global_variable_types[alias] = declared_type_text
                declarations.append((node, alias))
                aliases_by_declaration[id(node)] = alias
                function_names_by_declaration[id(node)] = getattr(func, "name", None)
                declaration_ids.add(id(node))
                binding = self.hlsl_groupshared_storage_binding(node, alias)
                if binding is not None:
                    bindings_by_declaration[id(node)] = binding

        self.hlsl_hoisted_groupshared_declarations = declarations
        self.hlsl_hoisted_groupshared_aliases_by_declaration = aliases_by_declaration
        self.hlsl_hoisted_groupshared_function_names_by_declaration = (
            function_names_by_declaration
        )
        self.hlsl_hoisted_groupshared_declaration_ids = declaration_ids
        self.hlsl_groupshared_pointer_bindings_by_declaration = bindings_by_declaration
        self.hlsl_global_groupshared_pointer_aliases = global_aliases

    def collect_hlsl_groupshared_declaration_constants(self, func):
        # A hoisted declaration must retain the constants visible at its original
        # lexical position, including aliases and nested shadowing.
        constants_by_declaration = {}
        self.collect_hlsl_groupshared_statement_constants(
            getattr(func, "body", []),
            self.hlsl_initial_int_constants(func),
            constants_by_declaration,
        )
        return constants_by_declaration

    def collect_hlsl_groupshared_statement_constants(
        self, statements, visible_int_constants, constants_by_declaration
    ):
        for stmt in self.hlsl_statement_body_items(statements):
            if isinstance(stmt, BlockNode):
                self.collect_hlsl_groupshared_statement_constants(
                    stmt,
                    dict(visible_int_constants),
                    constants_by_declaration,
                )
                continue

            if isinstance(stmt, IfNode):
                self.collect_hlsl_groupshared_statement_constants(
                    getattr(stmt, "then_branch", getattr(stmt, "if_body", None)),
                    dict(visible_int_constants),
                    constants_by_declaration,
                )
                else_branch = getattr(
                    stmt, "else_branch", getattr(stmt, "else_body", None)
                )
                if else_branch is not None:
                    self.collect_hlsl_groupshared_statement_constants(
                        else_branch,
                        dict(visible_int_constants),
                        constants_by_declaration,
                    )
                continue

            if isinstance(stmt, SwitchNode):
                for case in self.hlsl_switch_case_entries(stmt):
                    self.collect_hlsl_groupshared_statement_constants(
                        self.hlsl_switch_case_body_items(case),
                        dict(visible_int_constants),
                        constants_by_declaration,
                    )
                continue

            if isinstance(stmt, MatchNode):
                for arm in getattr(stmt, "arms", []) or []:
                    self.collect_hlsl_groupshared_statement_constants(
                        getattr(arm, "body", None),
                        dict(visible_int_constants),
                        constants_by_declaration,
                    )
                continue

            if isinstance(stmt, ForNode):
                loop_constants = dict(visible_int_constants)
                init = getattr(stmt, "init", None)
                if init is not None:
                    self.collect_hlsl_groupshared_statement_constants(
                        init,
                        loop_constants,
                        constants_by_declaration,
                    )
                self.collect_hlsl_groupshared_statement_constants(
                    getattr(stmt, "body", None),
                    loop_constants,
                    constants_by_declaration,
                )
                continue

            if isinstance(stmt, ForInNode):
                loop_constants = dict(visible_int_constants)
                loop_constants.pop(getattr(stmt, "pattern", None), None)
                self.collect_hlsl_groupshared_statement_constants(
                    getattr(stmt, "body", None),
                    loop_constants,
                    constants_by_declaration,
                )
                continue

            if isinstance(stmt, (WhileNode, DoWhileNode, LoopNode)):
                self.collect_hlsl_groupshared_statement_constants(
                    getattr(stmt, "body", None),
                    dict(visible_int_constants),
                    constants_by_declaration,
                )
                continue

            if isinstance(stmt, VariableNode) and (
                self.hlsl_declaration_has_groupshared_qualifier(stmt)
            ):
                constants_by_declaration[id(stmt)] = dict(visible_int_constants)

            self.hlsl_update_visible_int_constants(stmt, visible_int_constants)

    def hlsl_groupshared_declaration_type(self, node):
        declared_type = self.local_variable_declared_type(node)
        declared_type_text = self.type_name_string(declared_type)
        if not declared_type_text:
            return declared_type

        base_type, array_suffix = split_array_type_suffix(declared_type_text)
        if not array_suffix:
            return declared_type_text

        visible_constants = self.hlsl_hoisted_groupshared_constants_by_declaration.get(
            id(node)
        )
        if visible_constants is None:
            # A stage-scope declaration is already at global scope, where its extent was written.
            return declared_type_text
        scope_names = self.hlsl_hoisted_groupshared_scope_names_by_declaration.get(
            id(node), frozenset()
        )

        def resolve(match):
            name = match.group(0)
            if name in visible_constants:
                return str(visible_constants[name])
            if name in scope_names:
                # Moved to global scope, the extent would name the global of that name, or nothing.
                return self.refuse_lowering(
                    f"groupshared array '{getattr(node, 'name', '')}' moves to global scope, but its "
                    f"extent names '{name}', a parameter or local whose value is not a known integer constant",
                    name,
                )
            # Any other name is declared at global scope, which the hoisted declaration still sees.
            return name

        resolved_suffix = re.sub(r"\b[A-Za-z_]\w*\b", resolve, array_suffix)
        return f"{base_type}{resolved_suffix}"

    def activate_hlsl_hoisted_groupshared_declaration(self, node):
        alias = self.hlsl_hoisted_groupshared_aliases_by_declaration.get(id(node))
        if alias is None:
            return False

        name = getattr(node, "name", None)
        if not name:
            return True

        declared_type_text = self.hlsl_groupshared_declaration_type(node)
        self.current_identifier_aliases[name] = alias
        if declared_type_text:
            self.local_variable_types[name] = declared_type_text
            self.local_variable_types[alias] = declared_type_text
        binding = self.hlsl_groupshared_pointer_bindings_by_declaration.get(id(node))
        if binding is not None:
            self.current_hlsl_resource_pointer_aliases[name] = dict(binding)
            self.current_hlsl_resource_pointer_aliases[alias] = dict(binding)
        return True

    def hlsl_groupshared_declarations_to_emit(self, stage_groupshared_vars):
        declarations = [
            (node, getattr(node, "name", None)) for node in stage_groupshared_vars
        ]
        declarations.extend(
            (node, alias)
            for node, alias in self.hlsl_hoisted_groupshared_declarations
            if self.hlsl_hoisted_groupshared_function_names_by_declaration.get(id(node))
            not in self.hlsl_omitted_function_names
        )
        return declarations

    def generate_hlsl_hoisted_groupshared_declaration(self, node, alias):
        if self.hlsl_workgroup_pointer_declaration(node):
            raise DirectXWorkgroupPointerError(
                "DirectX workgroup pointer aliases cannot be emitted as "
                "groupshared pointer declarations",
                parameter_name=getattr(node, "name", None),
                reason="pointer-storage-declaration",
                source_location=getattr(node, "source_location", None),
            )
        vtype = self.hlsl_groupshared_declaration_type(node)
        name = alias or getattr(node, "name", None)
        if self.hlsl_type_contains_bfloat16(vtype):
            mapped_type = self.hlsl_bfloat16_groupshared_register_type(
                vtype,
                operation=f"groupshared variable '{name}'",
                source_location=getattr(node, "source_location", None),
            )
        else:
            mapped_type = self.map_type(vtype)
            base_type, array_suffix = split_array_type_suffix(str(mapped_type))
            if self.groupshared_bool_storage == "uint" and base_type.strip() == "bool":
                mapped_type = f"uint{array_suffix}"
        declaration = format_c_style_array_declaration(mapped_type, name)
        declaration = f"{self.local_variable_qualifier(node)}{declaration}"
        return f"{declaration};\n"

    def validate_hlsl_local_groupshared_declarations(self, func):
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            if not self.hlsl_declaration_has_groupshared_qualifier(node):
                continue
            if self.hlsl_workgroup_pointer_declaration(node):
                continue
            if id(node) in self.hlsl_hoisted_groupshared_declaration_ids:
                continue
            raise DirectXRefusal(
                "DirectX groupshared variables must be declared at shader/global scope"
            )
