"""Geometry, mesh-payload and stage-body checks of the Slang generator, plus emission of numthreads and stage attributes."""

from ..ast import (
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix


class SlangGeometryStagesMixin:
    """Geometry-stage members of ``SlangCodeGen``."""

    def slang_geometry_parameter_type_base(self, parameter):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return None

        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return None
        return base_type.split("<", 1)[0].strip()

    def is_slang_geometry_stream_type(self, type_name):
        type_name = self.resource_base_type(self.type_name_string(type_name))
        if not isinstance(type_name, str):
            return False
        base_type = type_name.split("<", 1)[0].strip()
        return base_type in self.SLANG_GEOMETRY_STREAM_TYPES

    def is_slang_geometry_stream_parameter(self, parameter):
        return (
            self.slang_geometry_parameter_type_base(parameter)
            in self.SLANG_GEOMETRY_STREAM_TYPES
        )

    def slang_geometry_stream_element_type_from_type(self, type_name):
        type_name = self.resource_base_type(self.type_name_string(type_name))
        if (
            not isinstance(type_name, str)
            or "<" not in type_name
            or not type_name.endswith(">")
        ):
            return None

        base_type, element_type = type_name.split("<", 1)
        if base_type.strip() not in self.SLANG_GEOMETRY_STREAM_TYPES:
            return None

        element_type = element_type[:-1].strip()
        return element_type or None

    def slang_geometry_stream_element_type(self, parameter):
        param_type = getattr(
            parameter,
            "param_type",
            getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
        )
        name = getattr(param_type, "name", None)
        generic_args = getattr(param_type, "generic_args", []) or []
        if name in self.SLANG_GEOMETRY_STREAM_TYPES and generic_args:
            return self.convert_type_node_to_string(generic_args[0])

        return self.slang_geometry_stream_element_type_from_type(
            self.slang_parameter_type_name(parameter)
        )

    def slang_geometry_input_primitive_qualifier(self, parameter):
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in self.SLANG_GEOMETRY_INPUT_PRIMITIVE_ARITIES:
                return normalized
        return None

    def validate_slang_geometry_stage(self, func, shader_type, parameters):
        shader_stage = self.slang_shader_stage_name(shader_type)
        stream_parameters = [
            parameter
            for parameter in parameters or []
            if self.is_slang_geometry_stream_parameter(parameter)
        ]
        if shader_stage != "geometry":
            if stream_parameters:
                raise SlangRefusal(
                    f"Slang {shader_type} stage cannot declare a geometry stream "
                    "output parameter"
                )
            return

        if parameters:
            self.validate_slang_geometry_stage_parameters(parameters)
            self.validate_slang_geometry_stream_output_semantics(parameters)
        self.validate_slang_geometry_stream_calls(func, shader_type, parameters)

    def validate_slang_geometry_stage_parameters(self, parameters):
        stream_parameters = [
            parameter
            for parameter in parameters or []
            if self.is_slang_geometry_stream_parameter(parameter)
        ]
        if not stream_parameters:
            raise SlangRefusal(
                "Slang geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )
        if len(stream_parameters) > 1:
            raise SlangRefusal(
                "Slang geometry stage must declare at most one stream output parameter"
            )

        stream_parameter = stream_parameters[0]
        directions = self.slang_parameter_direction_qualifiers(stream_parameter)
        if directions != {"inout"}:
            raise SlangRefusal(
                f"Slang geometry stream parameter '{stream_parameter.name}' "
                "must use the inout qualifier"
            )
        if self.slang_geometry_stream_element_type(stream_parameter) is None:
            raise SlangRefusal(
                f"Slang geometry stream parameter '{stream_parameter.name}' "
                "requires an output element type"
            )

        input_parameters = [
            parameter
            for parameter in parameters or []
            if self.slang_geometry_input_primitive_qualifier(parameter) is not None
        ]
        if not input_parameters:
            raise SlangRefusal(
                "Slang geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or "
                "triangleadj"
            )
        if len(input_parameters) > 1:
            raise SlangRefusal(
                "Slang geometry stage must declare at most one input primitive "
                "parameter"
            )

        self.validate_slang_geometry_input_primitive_arity(input_parameters[0])

    def validate_slang_geometry_input_primitive_arity(self, parameter):
        primitive = self.slang_geometry_input_primitive_qualifier(parameter)
        expected_count = self.SLANG_GEOMETRY_INPUT_PRIMITIVE_ARITIES.get(primitive)
        if expected_count is None:
            return

        if self.slang_parameter_array_size_expression(parameter) is None:
            raise SlangRefusal(
                f"Slang geometry stage {primitive} input primitive parameter "
                f"'{parameter.name}' must be an array with {expected_count} "
                "element(s)"
            )

        array_count = self.slang_parameter_array_count(parameter)
        if array_count is None:
            return
        if array_count != expected_count:
            raise SlangRefusal(
                f"Slang geometry stage {primitive} input primitive parameter "
                f"'{parameter.name}' must have {expected_count} element(s), "
                f"got {array_count}"
            )

    def validate_slang_geometry_stream_output_semantics(self, parameters):
        for parameter in parameters or []:
            if not self.is_slang_geometry_stream_parameter(parameter):
                continue

            stream_type_name = self.slang_geometry_stream_element_type(parameter)
            if stream_type_name is None:
                continue

            stream_type_name = stream_type_name.split("<", 1)[0].split("[", 1)[0]
            stream_struct = self.structs_by_name.get(stream_type_name.strip())
            if stream_struct is None:
                continue

            for member in getattr(stream_struct, "members", []) or []:
                semantic = self.semantic_from_node(member)
                member_name = getattr(member, "name", "<anonymous>")
                context = (
                    f"output stream struct '{stream_type_name}.{member_name}' semantic"
                )
                self.validate_slang_builtin_semantic_type(
                    semantic,
                    self.slang_tess_factor_member_type_name(member),
                    context,
                )
                self.validate_slang_output_semantic_stage("geometry", semantic, context)

    def validate_slang_geometry_stream_calls(self, func, shader_type, parameters):
        if self.slang_shader_stage_name(shader_type) != "geometry":
            return

        saved_variable_types = self.variable_types.copy()
        try:
            for parameter in parameters or []:
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", None)
                if not isinstance(func_expr, MemberAccessNode):
                    continue

                receiver = getattr(
                    func_expr, "object", getattr(func_expr, "object_expr", None)
                )
                receiver_type = self.expression_result_type(receiver)
                if not self.is_slang_geometry_stream_type(receiver_type):
                    continue

                member = str(getattr(func_expr, "member", ""))
                self.validate_slang_geometry_stream_call(
                    member, getattr(node, "args", []), receiver_type
                )
        finally:
            self.variable_types = saved_variable_types

    def validate_slang_geometry_stream_call(self, member, args, receiver_type):
        expected_arities = self.SLANG_GEOMETRY_STREAM_METHOD_ARITIES.get(member)
        if expected_arities is None:
            valid = ", ".join(sorted(self.SLANG_GEOMETRY_STREAM_METHOD_ARITIES))
            raise SlangRefusal(
                f"Slang geometry stream method {member} is unsupported; "
                f"valid methods are: {valid}"
            )

        if len(args) not in expected_arities:
            expected = " or ".join(str(arity) for arity in sorted(expected_arities))
            raise SlangRefusal(
                f"Slang geometry stream {member} requires {expected} "
                f"argument(s), got {len(args)}"
            )

        if member == "Append":
            self.validate_slang_geometry_stream_append_argument(args[0], receiver_type)

    def validate_slang_geometry_stream_append_argument(self, argument, receiver_type):
        expected_type = self.slang_geometry_stream_element_type_from_type(receiver_type)
        actual_type = self.expression_result_type(argument)
        if expected_type is None or actual_type is None:
            return

        expected_base, expected_suffix = split_array_type_suffix(
            self.convert_type(expected_type)
        )
        actual_base, actual_suffix = split_array_type_suffix(
            self.convert_type(actual_type)
        )
        if expected_base == actual_base and expected_suffix == actual_suffix:
            return

        raise SlangRefusal(
            "Slang geometry stream Append argument type "
            f"{self.convert_type(actual_type)} must match stream element type "
            f"{self.convert_type(expected_type)}"
        )

    def validate_slang_mesh_payload_parameter(self, shader_type, parameters):
        payload_parameters = self.slang_mesh_payload_parameters(parameters)
        if not payload_parameters:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage != "mesh":
            raise SlangRefusal(
                f"Slang {shader_type} stage cannot declare a mesh payload parameter"
            )
        if len(payload_parameters) > 1:
            raise SlangRefusal(
                "Slang mesh stage must declare at most one mesh payload parameter"
            )

        parameter = payload_parameters[0]
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if "in" not in directions or directions & {"out", "inout"}:
            raise SlangRefusal(
                f"Slang mesh payload parameter '{parameter.name}' "
                "must use the in qualifier"
            )
        if self.slang_parameter_user_struct_type(parameter) is None:
            raise SlangRefusal(
                f"Slang mesh payload parameter '{parameter.name}' "
                "must use a user-defined struct type"
            )

    def validate_slang_stage_body_builtins(
        self, body_statements, stage_name, params, stage_role=None
    ):
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in {"hull", "domain"}:
            return

        unsupported_tess_factor_builtins = {
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
        }
        declared_names = {
            getattr(param, "name", None)
            for param in params or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        if shader_stage == "hull" and stage_role == "patch_constant":
            invalid_semantics = {"SV_OutputControlPointID"}
            for param in params or []:
                semantic = self.semantic_from_node(param)
                if (
                    semantic
                    and self.map_semantic(semantic, stage_name) in invalid_semantics
                ):
                    raise SlangRefusal(
                        "Slang patch constant functions cannot use "
                        "gl_InvocationID or SV_OutputControlPointID"
                    )

            for node in self.walk_ast(body_statements):
                if not isinstance(node, IdentifierNode):
                    continue
                if node.name == "gl_InvocationID" and node.name not in declared_names:
                    raise SlangRefusal(
                        "Slang patch constant functions cannot use "
                        "gl_InvocationID or SV_OutputControlPointID"
                    )

        for node in self.walk_ast(body_statements):
            if not isinstance(node, IdentifierNode):
                continue
            if node.name not in unsupported_tess_factor_builtins:
                continue
            if node.name in declared_names:
                continue
            raise SlangRefusal(
                "Slang tessellation factor built-ins gl_TessLevelOuter and "
                "gl_TessLevelInner require an explicit patch constant function "
                "return value using SV_TessFactor and SV_InsideTessFactor"
            )

    def generate_slang_stage_numthreads(self, func, stage_name, execution_config=None):
        if not self.slang_stage_uses_numthreads(stage_name):
            return ""

        arguments = self.slang_stage_attribute_arguments(func, "numthreads")
        if arguments:
            values = [
                self.slang_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            values.extend(["1"] * (3 - len(values)))
            return f"[numthreads({', '.join(values[:3])})]\n"

        return self.generate_compute_numthreads(execution_config)

    def generate_slang_stage_attributes(self, func, stage_name):
        if stage_name not in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
        }:
            return ""

        quoted_argument_attributes = {
            "domain",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        result = ""
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name is None or attr_name == "numthreads":
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            values = [
                self.slang_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            if attr_name == "domain":
                values = [
                    self.canonical_slang_tessellation_domain(value) or value
                    for value in values
                ]
            if attr_name in quoted_argument_attributes:
                values = [f'"{value}"' for value in values]
            result += f"[{attr_name}({', '.join(values)})]\n"

        return result

    def function_return_semantic(self, node):
        return self.semantic_from_node(node, skip_stage_attributes=True)

    @staticmethod
    def slang_function_is_precise(node):
        """Return whether a function's results must be computed precisely."""
        qualifiers = {
            str(item).lower() for item in getattr(node, "qualifiers", []) or []
        }
        return "precise" in qualifiers or any(
            str(getattr(attribute, "name", "")).lower() == "precise"
            for attribute in getattr(node, "attributes", []) or []
        )
