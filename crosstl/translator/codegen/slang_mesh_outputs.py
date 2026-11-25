"""Mesh output parameters and payload, callable-data, ray-payload and hit-attribute parameter types in the Slang generator."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    MeshOpNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix


class SlangMeshOutputsMixin:
    """Mesh-output members of ``SlangCodeGen``."""

    def slang_mesh_output_parameter_role(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        if normalized in {"vertices", "indices", "primitives"}:
            return normalized
        return None

    def slang_mesh_output_role_from_parameter(self, parameter):
        roles = self.slang_mesh_output_roles_from_parameter(parameter)
        if len(roles) > 1:
            raise SlangRefusal(
                f"Slang mesh stage parameter '{parameter.name}' "
                "can use only one mesh role qualifier"
            )
        if roles:
            return roles[0]
        return None

    def slang_mesh_output_roles_from_parameter(self, parameter):
        roles = []
        for attr in getattr(parameter, "attributes", []) or []:
            role = self.slang_mesh_output_parameter_role(attr)
            if role is not None and role not in roles:
                roles.append(role)
        return roles

    def slang_parameter_mapped_base_and_array_suffix(self, parameter):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return None, None

        mapped_type = self.map_resource_type_with_format(type_name, parameter)
        return split_array_type_suffix(str(mapped_type))

    def slang_parameter_array_size_expression(self, parameter):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return None

        _base_type, array_suffix = split_array_type_suffix(str(type_name))
        if not array_suffix.startswith("["):
            return None

        closing_bracket = array_suffix.find("]")
        if closing_bracket < 0:
            return None
        size_expr = array_suffix[1:closing_bracket]
        if not size_expr:
            return None
        return size_expr

    def slang_parameter_array_count(self, parameter):
        size_expr = self.slang_parameter_array_size_expression(parameter)
        if size_expr is None:
            return None
        return self.slang_int_literal_value(size_expr)

    def validate_slang_mesh_output_array_parameter(self, parameter, role):
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if "out" not in directions or directions & {"in", "inout", "const"}:
            raise SlangRefusal(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "must use the out qualifier"
            )

        if self.slang_parameter_array_size_expression(parameter) is None:
            raise SlangRefusal(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "must declare a static array size"
            )

        array_count = self.slang_parameter_array_count(parameter)
        if array_count is not None and array_count <= 0:
            raise SlangRefusal(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "array size must be positive"
            )

    def slang_set_mesh_output_count_calls(self, func, active_helpers=None):
        if active_helpers is None:
            active_helpers = set()

        func_name = getattr(func, "name", None)
        if func_name in active_helpers:
            return []
        if func_name:
            active_helpers = active_helpers | {func_name}

        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call_name = self.slang_call_name(node)
            if call_name == "SetMeshOutputCounts":
                calls.append(getattr(node, "arguments", getattr(node, "args", [])))
                continue
            if not isinstance(node, FunctionCallNode):
                continue

            helper_func = self.user_functions_by_name.get(call_name)
            if helper_func is None:
                continue
            calls.extend(
                self.slang_set_mesh_output_count_calls(helper_func, active_helpers)
            )
        return calls

    def validate_slang_mesh_output_parameters(self, func, shader_type, parameters):
        if self.slang_shader_stage_name(shader_type) != "mesh":
            return

        role_parameters = {}
        for parameter in parameters or []:
            roles = self.slang_mesh_output_roles_from_parameter(parameter)
            if len(roles) > 1:
                raise SlangRefusal(
                    f"Slang mesh stage parameter '{parameter.name}' "
                    "can use only one mesh role qualifier"
                )
            if roles:
                role_parameters.setdefault(roles[0], []).append(parameter)

        if not role_parameters:
            return

        for role, role_params in role_parameters.items():
            if len(role_params) > 1:
                raise SlangRefusal(
                    f"Slang mesh stage must declare at most one {role} output parameter"
                )
            self.validate_slang_mesh_output_array_parameter(role_params[0], role)

        if "vertices" not in role_parameters:
            raise SlangRefusal(
                "Slang mesh stage output signature must declare an out vertices array"
            )
        if "indices" not in role_parameters:
            raise SlangRefusal(
                "Slang mesh stage output signature must declare an out indices array"
            )

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        expected_index_types = {
            "point": "uint",
            "line": "uint2",
            "triangle": "uint3",
        }
        expected_index_type = expected_index_types.get(topology)
        if expected_index_type is not None:
            index_param = role_parameters["indices"][0]
            index_base_type, _array_suffix = (
                self.slang_parameter_mapped_base_and_array_suffix(index_param)
            )
            if index_base_type != expected_index_type:
                raise SlangRefusal(
                    f"Slang mesh stage outputtopology '{topology}' requires "
                    f"indices parameter '{index_param.name}' to use "
                    f"{expected_index_type}, got {index_base_type}"
                )

        if "primitives" in role_parameters:
            index_count = self.slang_parameter_array_count(
                role_parameters["indices"][0]
            )
            primitive_count = self.slang_parameter_array_count(
                role_parameters["primitives"][0]
            )
            if (
                index_count is not None
                and primitive_count is not None
                and index_count != primitive_count
            ):
                raise SlangRefusal(
                    "Slang mesh stage primitives output array size must match "
                    "the indices output array size"
                )

        output_count_calls = self.slang_set_mesh_output_count_calls(func)
        if len(output_count_calls) != 1:
            raise SlangRefusal(
                "Slang mesh stage output signature must call "
                "SetMeshOutputCounts exactly once"
            )
        self.validate_slang_set_mesh_output_counts(
            output_count_calls[0], role_parameters
        )
        self.validate_slang_mesh_output_accesses(func, role_parameters)

    def validate_slang_set_mesh_output_counts(self, args, role_parameters):
        if len(args) != 2:
            raise SlangRefusal(
                "Slang mesh SetMeshOutputCounts requires exactly two arguments"
            )

        vertex_count = self.slang_int_literal_value(args[0])
        primitive_count = self.slang_int_literal_value(args[1])
        declared_counts = {
            role: self.slang_parameter_array_count(parameters[0])
            for role, parameters in role_parameters.items()
            if role in {"vertices", "indices", "primitives"}
        }

        if (
            vertex_count is not None
            and declared_counts.get("vertices") is not None
            and vertex_count > declared_counts["vertices"]
        ):
            raise SlangRefusal(
                "Slang mesh SetMeshOutputCounts vertex count exceeds the "
                "vertices output array size"
            )

        for role in ("indices", "primitives"):
            if (
                primitive_count is not None
                and declared_counts.get(role) is not None
                and primitive_count > declared_counts[role]
            ):
                raise SlangRefusal(
                    "Slang mesh SetMeshOutputCounts primitive count exceeds the "
                    f"{role} output array size"
                )

    def slang_call_name(self, node):
        if isinstance(node, MeshOpNode):
            return getattr(node, "operation", None)
        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def slang_statement_expression(self, statement):
        if isinstance(statement, ExpressionStatementNode):
            return getattr(statement, "expression", None)
        return statement

    def slang_statement_is_set_mesh_output_counts(self, statement):
        expr = self.slang_statement_expression(statement)
        call_name = self.slang_call_name(expr)
        if call_name == "SetMeshOutputCounts":
            return True
        if not isinstance(expr, FunctionCallNode):
            return False

        helper_func = self.user_functions_by_name.get(call_name)
        if helper_func is None:
            return False
        return bool(self.slang_set_mesh_output_count_calls(helper_func))

    def slang_mesh_output_role_by_parameter(self, role_parameters):
        role_by_name = {}
        for role, parameters in role_parameters.items():
            if not parameters:
                continue
            param_name = getattr(parameters[0], "name", None)
            if param_name:
                role_by_name[param_name] = role
        return role_by_name

    def slang_mesh_output_access(self, target, role_by_name):
        if isinstance(target, MemberAccessNode):
            obj = getattr(target, "object", getattr(target, "object_expr", None))
            return self.slang_mesh_output_access(obj, role_by_name)
        if isinstance(target, ArrayAccessNode):
            array = getattr(target, "array", getattr(target, "array_expr", None))
            array_name = self.identifier_name(array)
            if array_name in role_by_name:
                return (
                    role_by_name[array_name],
                    array_name,
                    getattr(target, "index", getattr(target, "index_expr", None)),
                )
            return self.slang_mesh_output_access(array, role_by_name)

        target_name = self.identifier_name(target)
        if target_name in role_by_name:
            return role_by_name[target_name], target_name, None
        return None

    def slang_expression_mentions_mesh_output(self, expr, role_by_name):
        for node in self.walk_ast(expr):
            if isinstance(node, IdentifierNode) and node.name in role_by_name:
                return True
            if self.slang_mesh_output_access(node, role_by_name) is not None:
                return True
        return False

    def validate_slang_mesh_output_accesses(self, func, role_parameters):
        role_by_name = self.slang_mesh_output_role_by_parameter(role_parameters)
        if not role_by_name:
            return

        declared_counts = {
            getattr(parameters[0], "name", None): self.slang_parameter_array_count(
                parameters[0]
            )
            for parameters in role_parameters.values()
            if parameters and getattr(parameters[0], "name", None)
        }

        output_counts_seen = False
        for statement in self.get_statements(getattr(func, "body", [])):
            if self.slang_statement_is_set_mesh_output_counts(statement):
                output_counts_seen = True
                continue

            for node in self.walk_ast(statement):
                if isinstance(node, ArrayAccessNode):
                    access = self.slang_mesh_output_access(node, role_by_name)
                    if access is not None:
                        self.validate_slang_mesh_output_literal_index(
                            access, declared_counts
                        )

                if isinstance(node, AssignmentNode):
                    access = self.slang_mesh_output_access(node.left, role_by_name)
                    if access is None:
                        continue
                    role, _name, index_expr = access
                    if index_expr is None:
                        raise SlangRefusal(
                            f"Slang mesh {role} output writes must target an "
                            "indexed output element"
                        )
                    if not output_counts_seen:
                        raise SlangRefusal(
                            "Slang mesh output writes must occur after "
                            "SetMeshOutputCounts"
                        )

                if output_counts_seen or not isinstance(
                    node, (FunctionCallNode, MeshOpNode)
                ):
                    continue
                if self.slang_call_name(node) == "SetMeshOutputCounts":
                    continue
                for arg in getattr(node, "arguments", getattr(node, "args", [])):
                    if self.slang_expression_mentions_mesh_output(arg, role_by_name):
                        raise SlangRefusal(
                            "Slang mesh output arrays cannot be passed to helper "
                            "calls before SetMeshOutputCounts"
                        )

    def validate_slang_mesh_output_literal_index(self, access, declared_counts):
        role, param_name, index_expr = access
        index_value = self.slang_int_literal_value(index_expr)
        if index_value is None:
            return

        declared_count = declared_counts.get(param_name)
        if declared_count is None:
            return
        if index_value < 0 or index_value >= declared_count:
            raise SlangRefusal(
                f"Slang mesh {role} output literal index {index_value} exceeds "
                f"the {role} output array size"
            )

    def slang_mesh_output_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        role = self.slang_mesh_output_role_from_parameter(parameter)
        if role is None:
            return None

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "mesh":
            directions = self.slang_parameter_direction_qualifiers(parameter)
            if "out" not in directions or directions & {"in", "inout", "const"}:
                raise SlangRefusal(
                    f"Slang mesh output parameter '{parameter.name}' "
                    "must use the out qualifier"
                )
            return f"out {role} {declaration}"

        if shader_type:
            raise SlangRefusal(
                f"Slang {shader_type} stage cannot declare mesh {role} output parameter"
            )

        return f"{self.slang_parameter_qualifier_prefix(parameter)}{declaration}"

    def slang_mesh_payload_parameter_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        if normalized == "mesh_payload":
            return "payload"
        return None

    def is_slang_mesh_payload_parameter(self, parameter):
        return any(
            self.slang_mesh_payload_parameter_attribute_name(attr)
            for attr in getattr(parameter, "attributes", []) or []
        )

    def slang_mesh_payload_parameters(self, parameters):
        return [
            parameter
            for parameter in parameters or []
            if self.is_slang_mesh_payload_parameter(parameter)
        ]

    def collect_slang_mesh_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) != "mesh":
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue

            parameters = getattr(
                entry_point, "parameters", getattr(entry_point, "params", [])
            )
            for parameter in self.slang_mesh_payload_parameters(parameters):
                payload_type = self.slang_parameter_user_struct_type(parameter)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def collect_slang_callable_data_parameter_types(self, stages):
        callable_data_types = set()
        if not isinstance(stages, dict):
            return callable_data_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) != "callable":
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if (
                        self.slang_ray_semantic_role(parameter, stage_name)
                        != "callable_data"
                    ):
                        continue
                    callable_type = self.slang_parameter_user_struct_type(parameter)
                    if callable_type is not None:
                        callable_data_types.add(callable_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if (
                    self.slang_ray_semantic_role(local_var, stage_name)
                    != "callable_data"
                ):
                    continue
                callable_type = self.slang_parameter_user_struct_type(local_var)
                if callable_type is not None:
                    callable_data_types.add(callable_type)

        return callable_data_types

    def collect_slang_ray_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if (
                self.slang_shader_stage_name(stage_name)
                not in self.slang_ray_stage_types()
            ):
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if self.slang_ray_semantic_role(parameter, stage_name) != "payload":
                        continue
                    payload_type = self.slang_parameter_user_struct_type(parameter)
                    if payload_type is not None:
                        payload_types.add(payload_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if self.slang_ray_semantic_role(local_var, stage_name) != "payload":
                    continue
                payload_type = self.slang_parameter_user_struct_type(local_var)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def collect_slang_hit_attribute_parameter_types(self, stages):
        hit_attribute_types = set()
        if not isinstance(stages, dict):
            return hit_attribute_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if (
                self.slang_shader_stage_name(stage_name)
                not in self.slang_ray_stage_types()
            ):
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if (
                        self.slang_ray_semantic_role(parameter, stage_name)
                        != "hit_attribute"
                    ):
                        continue
                    hit_attribute_type = self.slang_parameter_user_struct_type(
                        parameter
                    )
                    if hit_attribute_type is not None:
                        hit_attribute_types.add(hit_attribute_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if (
                    self.slang_ray_semantic_role(local_var, stage_name)
                    != "hit_attribute"
                ):
                    continue
                hit_attribute_type = self.slang_parameter_user_struct_type(local_var)
                if hit_attribute_type is not None:
                    hit_attribute_types.add(hit_attribute_type)

        return hit_attribute_types
