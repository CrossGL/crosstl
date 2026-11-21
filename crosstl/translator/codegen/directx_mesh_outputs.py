"""Mesh outputs in the HLSL generator: set constants and counts, semantics, switch targets, bool and int updates, visible struct statements and render targets."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    MeshOpNode,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import DirectXRefusal
from .array_utils import split_array_type_suffix
from .directx_values import DirectXMeshOutputCountsError
from .hlsl_semantic_attributes import hlsl_unmapped_semantic_name
from .stage_distance_layout import distance_semantic
from .stage_utils import normalize_stage_name


class DirectXMeshOutputMixin:
    """Mesh output members of ``HLSLCodeGen``."""

    # System values DXIL requires to be uint; Vulkan HLSL and GLSL (gl_Layer, gl_PrimitiveID) also take int.
    # GLSL declares the stencil export gl_FragStencilRefARB as int (GL_ARB_shader_stencil_export).
    HLSL_UNSIGNED_SYSTEM_VALUES = frozenset(
        {
            "SV_STENCILREF",
            "SV_RENDERTARGETARRAYINDEX",
            "SV_VIEWPORTARRAYINDEX",
            "SV_PRIMITIVEID",
            "SV_VERTEXID",
            "SV_INSTANCEID",
            "SV_SAMPLEINDEX",
            "SV_GSINSTANCEID",
            "SV_SHADINGRATE",
        }
    )

    def hlsl_unsigned_system_value_type(self, declaration_type, semantic):
        """``uint`` for an ``int`` member carrying a system value that DXIL delivers as ``uint``."""
        mapped = str(self.hlsl_canonical_semantic(semantic) or "").upper()
        if mapped in self.HLSL_UNSIGNED_SYSTEM_VALUES and declaration_type == "int":
            return "uint"
        return declaration_type

    def hlsl_output_builtin_semantic_type(self, mapped_semantic):
        if mapped_semantic is None:
            return None

        semantic_key = str(mapped_semantic).upper()
        output_types = {
            "SV_POSITION": "float4",
            "SV_DEPTH": "float",
            "SV_STENCILREF": "uint",
            "SV_COVERAGE": "uint",
            "PSIZE": "float",
        }
        if semantic_key in output_types:
            return output_types[semantic_key]
        if semantic_key.startswith("SV_TARGET"):
            suffix = semantic_key[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "float4"
        return None

    def hlsl_render_target_semantic(self, mapped_semantic):
        if mapped_semantic is None:
            return False
        semantic_key = str(mapped_semantic).upper()
        if not semantic_key.startswith("SV_TARGET"):
            return False
        suffix = semantic_key[len("SV_TARGET") :]
        return suffix == "" or suffix.isdigit()

    def hlsl_flexible_render_target_semantic(self, semantic, mapped_semantic):
        if not self.hlsl_render_target_semantic(mapped_semantic):
            return False
        if self.hlsl_gl_frag_data_semantic(semantic):
            return True
        if re.fullmatch(r"gl_FragColor\d*", str(semantic)):
            # A colour attachment (MSL [[color(N)]], HLSL SV_Target<N>) may be a scalar or a narrower vector, as in
            # float depth [[color(3)]] or float main() : SV_Target; DXC accepts any numeric 1-4 component type.
            return True
        return str(semantic).upper().startswith("SV_TARGET")

    def hlsl_gl_frag_data_semantic(self, semantic):
        semantic_text = str(semantic)
        prefix = "gl_FragData["
        suffix = "]"
        if not semantic_text.startswith(prefix) or not semantic_text.endswith(suffix):
            return False
        index = semantic_text[len(prefix) : -len(suffix)]
        return index.isdigit()

    def hlsl_render_target_type(self, actual_base_type):
        numeric_scalar_bases = (
            "min16float",
            "min16uint",
            "min16int",
            "uint16_t",
            "int16_t",
            "float16_t",
            "float",
            "double",
            "half",
            "uint",
            "int",
        )
        for scalar_base in numeric_scalar_bases:
            if not actual_base_type.startswith(scalar_base):
                continue
            suffix = actual_base_type[len(scalar_base) :]
            return suffix in {"", "2", "3", "4"}
        return False

    def validate_hlsl_stage_return_semantics(self, func, shader_type, semantic=None):
        if semantic is None:
            semantic = self.hlsl_function_return_semantic(func)
        self.validate_hlsl_stage_output_semantic(
            shader_type,
            semantic,
            "function return",
            getattr(func, "name", "<anonymous>"),
        )

        return_struct = self.hlsl_return_struct(func)
        if return_struct is None:
            return

        struct_name = getattr(return_struct, "name", "<anonymous>")
        for member in getattr(return_struct, "members", []) or []:
            self.validate_hlsl_stage_output_semantic(
                shader_type,
                self.semantic_from_struct_member(member),
                f"return struct '{struct_name}' member",
                getattr(member, "name", "<anonymous>"),
            )

    def validate_hlsl_stage_output_semantic(self, shader_type, semantic, context, name):
        if semantic is None:
            return

        mapped_semantic = self.hlsl_canonical_semantic(semantic)
        semantic_key = str(mapped_semantic).upper()
        if str(semantic).lower() == "gl_samplemaskin":
            raise DirectXRefusal(
                f"DirectX {shader_type} stage {context} '{name}' cannot use "
                f"input-only semantic '{mapped_semantic}'"
            )

        forbidden_description = None
        if shader_type == "vertex" and (
            semantic_key in {"SV_DEPTH", "SV_STENCILREF", "SV_COVERAGE"}
            or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "fragment output"
        elif shader_type in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
        } and (
            semantic_key in {"SV_DEPTH", "SV_STENCILREF", "SV_COVERAGE"}
            or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "fragment output"
        elif shader_type == "fragment" and semantic_key == "SV_POSITION":
            forbidden_description = "vertex position output"
        elif shader_type == "compute" and (
            semantic_key in {"SV_POSITION", "SV_DEPTH", "SV_STENCILREF", "SV_COVERAGE"}
            or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "graphics output"

        if forbidden_description is None:
            return

        raise DirectXRefusal(
            f"DirectX {shader_type} stage {context} '{name}' cannot use "
            f"{forbidden_description} semantic '{mapped_semantic}'"
        )

    def is_hlsl_target_semantic(self, semantic_key):
        if not semantic_key.startswith("SV_TARGET"):
            return False
        suffix = semantic_key[len("SV_TARGET") :]
        return suffix == "" or suffix.isdigit()

    def hlsl_canonical_semantic(self, semantic):
        if semantic is None:
            return None
        return self.hlsl_semantic_name(semantic)

    def hlsl_semantic_name(self, semantic):
        semantic_text = str(semantic)
        prefix = "gl_FragData["
        if semantic_text.startswith(prefix) and semantic_text.endswith("]"):
            index = semantic_text[len(prefix) : -1]
            if index.isdigit():
                return f"SV_TARGET{index}"
        distance = distance_semantic(semantic_text)
        if distance is not None and distance[1] > 0:
            return f"{self.semantic_map[distance[0]]}{distance[1]}"
        if semantic_text == "gl_InvocationID":
            # GLSL's invocation index is the hull control point in a
            # tessellation control stage and the instance in a geometry stage.
            stage = normalize_stage_name(self.current_hlsl_semantic_stage)
            stage_semantic = {
                "tessellation_control": "SV_OutputControlPointID",
                "geometry": "SV_GSInstanceID",
            }.get(stage)
            if stage_semantic is not None:
                return stage_semantic
        mapped = self.semantic_map.get(semantic_text)
        if mapped is not None:
            return mapped
        return hlsl_unmapped_semantic_name(self, semantic_text)

    def validate_hlsl_stage_parameter_requirements(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        self.validate_hlsl_set_mesh_output_count_stage_use(func, shader_type)
        self.validate_hlsl_dispatch_mesh_calls(func, shader_type)
        self.validate_hlsl_dispatch_mesh_payloads(func, shader_type)
        self.validate_hlsl_ray_tracing_calls(func, shader_type)
        self.validate_hlsl_ray_query_calls(func, shader_type)
        self.validate_hlsl_ray_stage_parameters(func, shader_type, parameters)
        if shader_type == "mesh":
            self.validate_hlsl_mesh_payload_parameter(func, parameters)
            self.validate_hlsl_mesh_output_parameters(func, parameters)
        if not parameters:
            return

        parameter_type_bases = {
            self.hlsl_parameter_type_base(parameter) for parameter in parameters
        }
        if shader_type == "geometry" and not (
            parameter_type_bases & {"PointStream", "LineStream", "TriangleStream"}
        ):
            raise DirectXRefusal(
                "DirectX geometry stage parameters must include an HLSL stream "
                "output parameter: PointStream<T>, LineStream<T>, or "
                "TriangleStream<T>"
            )
        if shader_type == "geometry" and not any(
            set(self.hlsl_parameter_qualifiers(parameter))
            & {"point", "line", "triangle", "lineadj", "triangleadj"}
            for parameter in parameters
        ):
            raise DirectXRefusal(
                "DirectX geometry stage parameters must include an HLSL input "
                "primitive qualifier: point, line, triangle, lineadj, or "
                "triangleadj"
            )
        if shader_type == "geometry":
            self.validate_hlsl_geometry_input_primitive_arity(parameters)
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "geometry",
                ("SV_PrimitiveID", "SV_GSInstanceID"),
            )
            self.validate_hlsl_geometry_stream_output_semantics(parameters)

        if shader_type == "fragment":
            self.validate_hlsl_fragment_system_value_types(parameters)

        if shader_type == "tessellation_control":
            if "InputPatch" not in parameter_type_bases:
                raise DirectXRefusal(
                    "DirectX tessellation_control stage parameters must include "
                    "an InputPatch<T, N> parameter"
                )
            self.validate_hlsl_patch_parameter_signatures(
                parameters, "InputPatch", "tessellation_control stage"
            )
            self.validate_hlsl_output_control_points(func, parameters)
            if not any(
                self.hlsl_parameter_has_semantic(parameter, "SV_OutputControlPointID")
                for parameter in parameters
            ):
                raise DirectXRefusal(
                    "DirectX tessellation_control stage parameters must include "
                    "an SV_OutputControlPointID parameter"
                )
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "tessellation_control",
                ("SV_OutputControlPointID", "SV_PrimitiveID"),
            )

        if shader_type == "tessellation_evaluation":
            if "OutputPatch" not in parameter_type_bases:
                raise DirectXRefusal(
                    "DirectX tessellation_evaluation stage parameters must include "
                    "an OutputPatch<T, N> parameter"
                )
            self.validate_hlsl_patch_parameter_signatures(
                parameters, "OutputPatch", "tessellation_evaluation stage"
            )
            self.validate_hlsl_domain_output_patch_control_points(parameters)
            self.validate_hlsl_domain_output_patch_element_type(parameters)
            if not any(
                self.hlsl_parameter_has_semantic(parameter, "SV_DomainLocation")
                for parameter in parameters
            ):
                raise DirectXRefusal(
                    "DirectX tessellation_evaluation stage parameters must include "
                    "an SV_DomainLocation parameter"
                )
            self.validate_hlsl_domain_location_type(parameters)
            self.validate_hlsl_domain_location_components(func, parameters)
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "tessellation_evaluation",
                ("SV_PrimitiveID",),
            )

        if shader_type == "compute":
            self.validate_hlsl_compute_system_value_types(parameters)

        if shader_type in {"mesh", "task", "amplification", "object"}:
            self.validate_hlsl_thread_system_value_types(parameters, shader_type)
            self.validate_hlsl_mesh_view_id_system_value(parameters, shader_type)

    def validate_hlsl_geometry_stream_output_semantics(self, parameters):
        stream_types = {"PointStream", "LineStream", "TriangleStream"}
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) not in stream_types:
                continue

            stream_type_name = self.hlsl_parameter_type_generic_argument(parameter, 0)
            if stream_type_name is None:
                continue

            stream_type_name = (
                stream_type_name.split("<", 1)[0].split("[", 1)[0].strip()
            )
            stream_struct = self.structs_by_name.get(stream_type_name)
            if stream_struct is None:
                continue

            for member in getattr(stream_struct, "members", []) or []:
                self.validate_hlsl_stage_output_semantic(
                    "geometry",
                    self.semantic_from_struct_member(member),
                    f"output stream struct '{stream_type_name}' member",
                    getattr(member, "name", "<anonymous>"),
                )

    def hlsl_parameter_array_size_expression(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if str(type(param_type)).find("ArrayType") != -1:
            return getattr(param_type, "size", None)

        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        _base_type, array_suffix = split_array_type_suffix(str(type_name))
        if not array_suffix:
            return None
        first_dimension = array_suffix[1:].split("]", 1)[0]
        return first_dimension if first_dimension else None

    def hlsl_parameter_user_struct_type(self, parameter):
        base_type, _array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None

        struct_name = base_type.split("<", 1)[0].strip()
        if struct_name not in self.structs_by_name:
            return None
        return struct_name

    def validate_hlsl_mesh_output_array_parameter(self, parameter, role):
        if "out" not in self.hlsl_parameter_direction_qualifiers(parameter):
            raise DirectXRefusal(
                f"DirectX mesh stage {role} parameter '{parameter.name}' "
                "must use the out qualifier"
            )

        if not self.hlsl_parameter_is_array(parameter):
            raise DirectXRefusal(
                f"DirectX mesh stage {role} parameter '{parameter.name}' "
                "must be a statically sized array"
            )

        if self.hlsl_parameter_array_size_expression(parameter) is None:
            raise DirectXRefusal(
                f"DirectX mesh stage {role} parameter '{parameter.name}' "
                "must declare a static array size"
            )

        array_count = self.hlsl_parameter_array_count(parameter)
        if array_count is not None and not 1 <= array_count <= 256:
            raise DirectXRefusal(
                f"DirectX mesh stage {role} parameter '{parameter.name}' "
                "array size must be in the range 1..256"
            )

    def hlsl_mesh_output_struct(self, parameter, role):
        base_type, _array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None

        struct_name = base_type.split("<", 1)[0].strip()
        struct_node = self.structs_by_name.get(struct_name)
        if struct_node is None:
            raise DirectXRefusal(
                f"DirectX mesh stage {role} parameter '{parameter.name}' "
                "must use a user-defined struct type"
            )
        return struct_node

    def hlsl_struct_member_type_name(self, member):
        if isinstance(member, ArrayNode):
            return self.type_name_string(
                getattr(member, "element_type", getattr(member, "vtype", None))
            )
        return self.type_name_string(
            getattr(member, "member_type", getattr(member, "vtype", None))
        )

    def validate_hlsl_mesh_output_struct(self, parameter, role):
        struct_node = self.hlsl_mesh_output_struct(parameter, role)
        if struct_node is None:
            return

        primitive_only_semantics = {
            "SV_CULLPRIMITIVE",
            "SV_RENDERTARGETARRAYINDEX",
            "SV_VIEWPORTARRAYINDEX",
            "SV_SHADINGRATE",
        }
        primitive_expected_types = {
            "SV_CULLPRIMITIVE": "bool",
            "SV_RENDERTARGETARRAYINDEX": "uint",
            "SV_VIEWPORTARRAYINDEX": "uint",
            "SV_SHADINGRATE": "uint",
        }

        has_position = False
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", "<anonymous>")
            semantic = self.semantic_from_struct_member(
                member
            ) or self.hlsl_default_mesh_output_member_semantics(struct_node).get(
                member_name
            )
            if semantic is None:
                raise DirectXRefusal(
                    f"DirectX mesh stage {role} output struct "
                    f"'{struct_node.name}' member '{member_name}' "
                    "must declare a semantic"
                )

            mapped_semantic = self.hlsl_canonical_semantic(semantic)
            semantic_key = str(mapped_semantic).upper()
            self.validate_hlsl_stage_output_semantic(
                "mesh",
                semantic,
                f"{role} output struct '{struct_node.name}' member",
                member_name,
            )

            if role == "vertices" and semantic_key in primitive_only_semantics:
                raise DirectXRefusal(
                    f"DirectX mesh stage vertices output struct "
                    f"'{struct_node.name}' member '{member_name}' cannot use "
                    f"per-primitive semantic '{mapped_semantic}'"
                )

            expected_type = primitive_expected_types.get(semantic_key)
            if role == "primitives" and expected_type is not None:
                actual_type = self.map_type(self.hlsl_struct_member_type_name(member))
                actual_base_type, array_suffix = split_array_type_suffix(actual_type)
                if array_suffix or actual_base_type != expected_type:
                    raise DirectXRefusal(
                        f"DirectX mesh stage primitives output struct "
                        f"'{struct_node.name}' semantic '{mapped_semantic}' "
                        f"requires member '{member_name}' to have type "
                        f"{expected_type}, got {actual_type}"
                    )

            if role == "vertices" and semantic_key == "SV_POSITION":
                has_position = True

        if role == "vertices" and not has_position:
            raise DirectXRefusal(
                f"DirectX mesh stage vertices output struct '{struct_node.name}' "
                "must declare an SV_Position member"
            )

    def hlsl_mesh_output_parameter_literal_count(self, parameter):
        return self.hlsl_parameter_array_count(parameter)

    def hlsl_set_mesh_output_count_args_in_node(self, root):
        calls = []
        for node in self.walk_ast(root):
            if not self.hlsl_expression_is_set_mesh_output_counts(node):
                continue
            if isinstance(node, FunctionCallNode):
                calls.append(getattr(node, "arguments", getattr(node, "args", [])))
            elif isinstance(node, MeshOpNode):
                calls.append(getattr(node, "arguments", []))
        return calls

    def hlsl_set_mesh_output_count_calls(self, func):
        return self.hlsl_set_mesh_output_count_args_in_node(getattr(func, "body", []))

    def hlsl_mesh_output_call_count(self, func):
        return len(self.hlsl_set_mesh_output_count_calls(func))

    def hlsl_integer_constant_value(self, expr, constants=None):
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            value = self.hlsl_integer_constant_value(
                getattr(expr, "operand", None), constants
            )
            if value is None:
                return None
            if operator == "-":
                return -value
            if operator == "+":
                return value
            return None
        return self.hlsl_constant_int_value(expr, constants)

    def hlsl_literal_bool_value(self, expr, constants=None):
        constants = constants or {}
        if expr is None:
            return None
        if isinstance(expr, bool):
            return expr
        if isinstance(expr, str):
            return constants.get(expr)
        if hasattr(expr, "value") and isinstance(expr.value, bool):
            return expr.value
        name = getattr(expr, "name", None)
        if isinstance(name, str) and name in constants:
            return constants[name]
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            value = self.hlsl_literal_bool_value(
                getattr(expr, "operand", None), constants
            )
            if value is None:
                return None
            if operator == "!":
                return not value
        return None

    def collect_hlsl_literal_bool_constants(self, constants):
        resolved = {}
        pending = list(constants or [])

        changed = True
        while changed:
            changed = False
            remaining = []
            for const in pending:
                name = getattr(const, "name", None)
                if not name or name in resolved:
                    continue

                value = self.hlsl_literal_bool_value(
                    getattr(const, "value", None), resolved
                )
                if value is None:
                    remaining.append(const)
                    continue

                resolved[name] = value
                changed = True
            pending = remaining

        return resolved

    def hlsl_bool_constant_value(self, expr, constants=None):
        if constants is None:
            constants = self.literal_bool_constants
        return self.hlsl_literal_bool_value(expr, constants)

    def hlsl_initial_bool_constants(self, func):
        visible_constants = dict(self.literal_bool_constants)
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def hlsl_initial_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def hlsl_initial_texture_offset_constants(self, func):
        visible_constants = dict(self.hlsl_texture_offset_global_constants)
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def hlsl_update_visible_bool_constants(self, stmt, visible_bool_constants):
        if isinstance(stmt, VariableNode):
            name = getattr(stmt, "name", None)
            if not name:
                return

            visible_bool_constants.pop(name, None)
            if "const" not in getattr(stmt, "qualifiers", []):
                return

            value = self.hlsl_bool_constant_value(
                getattr(stmt, "initial_value", None), visible_bool_constants
            )
            if value is not None:
                visible_bool_constants[name] = value
            return

        assignment = self.hlsl_assignment_from_statement(stmt)
        if assignment is None:
            return

        target_name = self.expression_name(
            getattr(assignment, "target", getattr(assignment, "left", None))
        )
        if target_name:
            visible_bool_constants.pop(target_name, None)

    def hlsl_update_visible_int_constants(self, stmt, visible_int_constants):
        if isinstance(stmt, VariableNode):
            name = getattr(stmt, "name", None)
            if not name:
                return

            visible_int_constants.pop(name, None)
            if "const" not in getattr(stmt, "qualifiers", []):
                return

            value = self.hlsl_integer_constant_value(
                getattr(stmt, "initial_value", None), visible_int_constants
            )
            if value is not None:
                visible_int_constants[name] = value
            return

        assignment = self.hlsl_assignment_from_statement(stmt)
        if assignment is None:
            return

        target_name = self.expression_name(
            getattr(assignment, "target", getattr(assignment, "left", None))
        )
        if target_name:
            visible_int_constants.pop(target_name, None)

    def hlsl_update_visible_texture_offset_constants(
        self, stmt, visible_offset_constants
    ):
        if isinstance(stmt, VariableNode):
            name = getattr(stmt, "name", None)
            if not name:
                return

            visible_offset_constants.pop(name, None)
            if "const" not in getattr(stmt, "qualifiers", []):
                return

            value = self.hlsl_texture_offset_constant_value(
                getattr(stmt, "initial_value", None),
                scalar_constants=self.hlsl_current_visible_int_constants(),
                offset_constants=visible_offset_constants,
            )
            if value is not None and len(value) > 1:
                visible_offset_constants[name] = value
            return

        assignment = self.hlsl_assignment_from_statement(stmt)
        if assignment is None:
            return

        target_name = self.expression_name(
            getattr(assignment, "target", getattr(assignment, "left", None))
        )
        if target_name:
            visible_offset_constants.pop(target_name, None)

    def hlsl_current_visible_int_constants(self):
        if self.current_hlsl_visible_int_constants is not None:
            return self.current_hlsl_visible_int_constants
        return self.literal_int_constants

    def hlsl_record_generated_statement_int_constants(self, stmt, generated_code):
        if self.current_hlsl_visible_int_constants is not None:
            self.hlsl_update_visible_int_constants(
                stmt, self.current_hlsl_visible_int_constants
            )
        if self.current_hlsl_texture_offset_constants is not None:
            self.hlsl_update_visible_texture_offset_constants(
                stmt, self.current_hlsl_texture_offset_constants
            )
        return generated_code

    def validate_hlsl_scalar_int_uint_expression(self, argument, context):
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return

        mapped_type = self.map_type(argument_type)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix or base_type not in {"int", "uint"}:
            raise DirectXRefusal(
                f"{context} must be scalar int or uint, got {mapped_type}"
            )

    def validate_hlsl_set_mesh_output_count_stage_use(self, func, shader_type):
        if shader_type in {"mesh", None}:
            return
        if not self.hlsl_set_mesh_output_count_calls(func):
            return
        raise DirectXRefusal(
            f"DirectX {shader_type} stage cannot call SetMeshOutputCounts; "
            "SetMeshOutputCounts is only valid in mesh stages"
        )

    def validate_hlsl_set_mesh_output_count_bounds(
        self, args, role_parameters, count_parameters, visible_int_constants=None
    ):
        for index, (label, max_count) in enumerate(count_parameters):
            literal_count = self.hlsl_integer_constant_value(
                args[index], visible_int_constants
            )
            if literal_count is None:
                continue

            if literal_count < 0:
                raise DirectXRefusal(
                    f"DirectX mesh SetMeshOutputCounts {label} argument "
                    "must be non-negative"
                )
            if max_count is not None and literal_count > max_count:
                role = "vertices" if index == 0 else "indices"
                parameter_name = role_parameters[role][0].name
                raise DirectXRefusal(
                    f"DirectX mesh SetMeshOutputCounts {label} argument "
                    f"({literal_count}) cannot exceed {role} output array "
                    f"'{parameter_name}' size ({max_count})"
                )

    def hlsl_set_mesh_output_count_bound_parameters(self, role_parameters):
        return (
            (
                "numVertices",
                self.hlsl_mesh_output_parameter_literal_count(
                    role_parameters["vertices"][0]
                ),
            ),
            (
                "numPrimitives",
                self.hlsl_mesh_output_parameter_literal_count(
                    role_parameters["indices"][0]
                ),
            ),
        )

    def hlsl_update_set_mesh_output_count_literals(
        self,
        args,
        role_parameters,
        set_count_literals,
        visible_int_constants=None,
    ):
        if len(args) != 2:
            return

        self.validate_hlsl_set_mesh_output_count_bounds(
            args,
            role_parameters,
            self.hlsl_set_mesh_output_count_bound_parameters(role_parameters),
            visible_int_constants,
        )
        vertex_count = self.hlsl_integer_constant_value(args[0], visible_int_constants)
        primitive_count = self.hlsl_integer_constant_value(
            args[1], visible_int_constants
        )
        set_count_literals.update(
            {
                "vertices": vertex_count,
                "indices": primitive_count,
                "primitives": primitive_count,
            }
        )

    def hlsl_thread_varying_mesh_condition_names(self, func):
        names = set()
        thread_varying_semantics = {
            "SV_DispatchThreadID",
            "SV_GroupIndex",
            "SV_GroupThreadID",
        }
        for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []:
            semantic = self.hlsl_canonical_semantic(self.semantic_from_node(parameter))
            if semantic in thread_varying_semantics:
                names.add(parameter.name)
        return names

    def hlsl_expression_identifier_names(self, expr):
        names = set()
        for node in self.walk_ast(expr):
            name = self.expression_name(node)
            if name:
                names.add(name)
        return names

    def hlsl_condition_uses_thread_varying_name(self, condition, thread_varying_names):
        if not thread_varying_names:
            return False
        return bool(
            self.hlsl_expression_identifier_names(condition) & thread_varying_names
        )

    def hlsl_update_thread_varying_alias_names(self, stmt, thread_varying_names):
        if isinstance(stmt, VariableNode):
            name = getattr(stmt, "name", None)
            if not name:
                return
            if self.hlsl_condition_uses_thread_varying_name(
                getattr(stmt, "initial_value", None), thread_varying_names
            ):
                thread_varying_names.add(name)
            else:
                thread_varying_names.discard(name)
            return

        assignment = self.hlsl_assignment_from_statement(stmt)
        if assignment is None:
            return

        target_name = self.expression_name(
            getattr(assignment, "target", getattr(assignment, "left", None))
        )
        if not target_name:
            return

        value_uses_thread_varying = self.hlsl_condition_uses_thread_varying_name(
            getattr(assignment, "value", getattr(assignment, "right", None)),
            thread_varying_names,
        )
        operator = getattr(assignment, "operator", "=")
        if value_uses_thread_varying or (
            operator != "=" and target_name in thread_varying_names
        ):
            thread_varying_names.add(target_name)
        else:
            thread_varying_names.discard(target_name)

    def validate_hlsl_set_mesh_output_counts_control_flow(
        self,
        statements,
        thread_varying_names,
        in_loop=False,
        in_thread_varying_branch=False,
    ):
        for stmt in statements:
            if isinstance(stmt, SwitchNode):
                contains_output_count = (
                    self.hlsl_statement_contains_set_mesh_output_counts(stmt)
                )
                if contains_output_count and in_loop:
                    raise DirectXMeshOutputCountsError(
                        "DirectX mesh SetMeshOutputCounts must not be called from "
                        "loop control flow"
                    )

                condition = getattr(stmt, "expression", None)
                branch_is_thread_varying = (
                    in_thread_varying_branch
                    or self.hlsl_condition_uses_thread_varying_name(
                        condition, thread_varying_names
                    )
                )
                if contains_output_count and branch_is_thread_varying:
                    raise DirectXMeshOutputCountsError(
                        "DirectX mesh SetMeshOutputCounts must not be called from "
                        "thread-varying control flow"
                    )

                for case in self.hlsl_switch_case_entries(stmt):
                    self.validate_hlsl_set_mesh_output_counts_control_flow(
                        self.hlsl_switch_case_body_items(case),
                        thread_varying_names,
                        in_loop,
                        branch_is_thread_varying,
                    )
                continue

            contains_output_count = self.hlsl_statement_contains_set_mesh_output_counts(
                stmt
            )
            if contains_output_count and in_loop:
                raise DirectXMeshOutputCountsError(
                    "DirectX mesh SetMeshOutputCounts must not be called from "
                    "loop control flow"
                )
            if contains_output_count and in_thread_varying_branch:
                raise DirectXMeshOutputCountsError(
                    "DirectX mesh SetMeshOutputCounts must not be called from "
                    "thread-varying control flow"
                )

            if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                self.validate_hlsl_set_mesh_output_counts_control_flow(
                    self.hlsl_statement_body_items(stmt),
                    thread_varying_names,
                    in_loop,
                    in_thread_varying_branch,
                )
                continue

            if isinstance(stmt, IfNode):
                condition = getattr(
                    stmt, "condition", getattr(stmt, "if_condition", None)
                )
                branch_is_thread_varying = (
                    in_thread_varying_branch
                    or self.hlsl_condition_uses_thread_varying_name(
                        condition, thread_varying_names
                    )
                )
                self.validate_hlsl_set_mesh_output_counts_control_flow(
                    self.hlsl_statement_body_items(
                        getattr(stmt, "then_branch", getattr(stmt, "if_body", None))
                    ),
                    thread_varying_names,
                    in_loop,
                    branch_is_thread_varying,
                )
                else_branch = getattr(
                    stmt, "else_branch", getattr(stmt, "else_body", None)
                )
                if else_branch is not None:
                    self.validate_hlsl_set_mesh_output_counts_control_flow(
                        self.hlsl_statement_body_items(else_branch),
                        thread_varying_names,
                        in_loop,
                        branch_is_thread_varying,
                    )
                continue

            if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
                self.validate_hlsl_set_mesh_output_counts_control_flow(
                    self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                    thread_varying_names,
                    True,
                    in_thread_varying_branch,
                )

    def validate_hlsl_set_mesh_output_counts_placement(self, func):
        self.validate_hlsl_set_mesh_output_counts_control_flow(
            self.hlsl_statement_body_items(getattr(func, "body", [])),
            self.hlsl_thread_varying_mesh_condition_names(func),
        )

    def validate_hlsl_set_mesh_output_counts(self, func, role_parameters):
        calls = self.hlsl_set_mesh_output_count_calls(func)
        if not calls:
            return

        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {
            **previous_local_variable_types,
            **self.function_scope_variable_types(func),
        }
        try:
            for args in calls:
                if len(args) != 2:
                    raise DirectXRefusal(
                        "DirectX mesh SetMeshOutputCounts requires exactly "
                        "two arguments: numVertices and numPrimitives"
                    )

                self.validate_hlsl_scalar_int_uint_expression(
                    args[0],
                    "DirectX mesh SetMeshOutputCounts numVertices argument",
                )
                self.validate_hlsl_scalar_int_uint_expression(
                    args[1],
                    "DirectX mesh SetMeshOutputCounts numPrimitives argument",
                )
                self.validate_hlsl_set_mesh_output_count_bounds(
                    args,
                    role_parameters,
                    self.hlsl_set_mesh_output_count_bound_parameters(role_parameters),
                )
        finally:
            self.local_variable_types = previous_local_variable_types
        self.validate_hlsl_set_mesh_output_counts_placement(func)

    def hlsl_mesh_output_role_by_parameter_name(self, role_parameters):
        role_by_name = {}
        for role, parameters in role_parameters.items():
            for parameter in parameters:
                role_by_name[parameter.name] = role
        return role_by_name

    def hlsl_expression_is_set_mesh_output_counts(self, expr):
        if isinstance(expr, FunctionCallNode):
            return self.function_call_name(expr) == "SetMeshOutputCounts"
        if isinstance(expr, MeshOpNode):
            return getattr(expr, "operation", None) == "SetMeshOutputCounts"
        return False

    def hlsl_statement_contains_set_mesh_output_counts(self, stmt):
        return bool(self.hlsl_set_mesh_output_count_args_in_node(stmt))

    def hlsl_statement_body_items(self, body):
        if body is None:
            return []
        if isinstance(body, BlockNode) or hasattr(body, "statements"):
            return getattr(body, "statements", []) or []
        if isinstance(body, list):
            return body
        return [body]

    def hlsl_switch_case_entries(self, switch_node):
        entries = list(
            getattr(switch_node, "ordered_cases", None)
            or getattr(switch_node, "cases", [])
            or []
        )
        default_case = getattr(switch_node, "default_case", None)
        has_default = any(getattr(case, "value", None) is None for case in entries)
        if default_case is not None and not has_default:
            entries.append(default_case)
        return entries

    def hlsl_switch_case_body_items(self, case):
        if case is None:
            return []
        if isinstance(case, list):
            return case
        return self.hlsl_statement_body_items(
            getattr(case, "statements", getattr(case, "body", case))
        )

    def hlsl_switch_case_terminates(self, case):
        body = self.hlsl_switch_case_body_items(case)
        return bool(body) and isinstance(
            body[-1], (BreakNode, ContinueNode, ReturnNode)
        )

    def hlsl_switch_fallthrough_path_body(self, case_entries, start_index):
        path_body = []
        for case in case_entries[start_index:]:
            path_body.extend(self.hlsl_switch_case_body_items(case))
            if self.hlsl_switch_case_terminates(case):
                break
        return path_body

    def hlsl_switch_possible_start_indices(
        self, switch_node, case_entries, visible_int_constants=None
    ):
        selector_value = self.hlsl_integer_constant_value(
            getattr(switch_node, "expression", None), visible_int_constants
        )
        default_index = None
        for index, case in enumerate(case_entries):
            case_value = getattr(case, "value", None)
            if case_value is None:
                default_index = index
                continue
            literal_value = self.hlsl_integer_constant_value(
                case_value, visible_int_constants
            )
            if selector_value is not None and literal_value == selector_value:
                return [index], True

        if selector_value is not None:
            if default_index is None:
                return [], True
            return [default_index], True

        return list(range(len(case_entries))), default_index is not None

    def validate_hlsl_mesh_output_write_switch(
        self,
        switch_node,
        set_mesh_output_counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        active_helper_calls=None,
        visible_bool_constants=None,
        visible_int_constants=None,
        role_parameters=None,
    ):
        counts_seen, _ = self.validate_hlsl_mesh_output_write_switch_result(
            switch_node,
            set_mesh_output_counts_seen,
            role_by_name,
            declared_counts,
            set_count_literals,
            active_helper_calls,
            visible_bool_constants=visible_bool_constants,
            visible_int_constants=visible_int_constants,
            role_parameters=role_parameters,
        )
        return counts_seen

    def validate_hlsl_mesh_output_write_switch_result(
        self,
        switch_node,
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

        case_entries = self.hlsl_switch_case_entries(switch_node)
        if not case_entries:
            return set_mesh_output_counts_seen, True

        start_indices, covers_all_paths = self.hlsl_switch_possible_start_indices(
            switch_node, case_entries, visible_int_constants
        )
        path_results = []
        case_loop_exit_nodes = tuple(
            exit_node for exit_node in loop_exit_nodes if exit_node is not BreakNode
        )
        for start_index in start_indices:
            path_results.append(
                self.validate_hlsl_mesh_output_write_sequence_result(
                    self.hlsl_switch_fallthrough_path_body(case_entries, start_index),
                    set_mesh_output_counts_seen,
                    role_by_name,
                    declared_counts,
                    set_count_literals,
                    active_helper_calls,
                    case_loop_exit_nodes,
                    visible_bool_constants=dict(visible_bool_constants),
                    visible_int_constants=dict(visible_int_constants),
                    role_parameters=role_parameters,
                )
            )

        if not covers_all_paths:
            path_results.append((set_mesh_output_counts_seen, True))
        if not path_results:
            return set_mesh_output_counts_seen, True

        continuing_counts = [
            counts_seen for counts_seen, can_continue in path_results if can_continue
        ]
        if not continuing_counts:
            return set_mesh_output_counts_seen, False

        return all(continuing_counts), True

    def hlsl_assignment_from_statement(self, stmt):
        if isinstance(stmt, AssignmentNode):
            return stmt
        expr = getattr(stmt, "expression", None)
        if isinstance(expr, AssignmentNode):
            return expr
        return None

    def hlsl_mesh_output_array_access_from_target(self, target):
        current = target
        member_path = []
        while isinstance(current, MemberAccessNode):
            member_path.append(str(getattr(current, "member", "")))
            current = getattr(current, "object", getattr(current, "object_expr", None))

        if isinstance(current, ArrayAccessNode):
            return current, list(reversed(member_path))
        return None, member_path

    def hlsl_mesh_output_assignment_target_info(self, target, role_by_name):
        array_access, member_path = self.hlsl_mesh_output_array_access_from_target(
            target
        )
        if array_access is None:
            return None

        array_name = self.expression_name(
            getattr(array_access, "array", getattr(array_access, "array_expr", None))
        )
        role = role_by_name.get(array_name)
        if role is None:
            return None
        parameter_name = array_name
        if isinstance(role, dict):
            parameter_name = role.get("name", array_name)
            role = role.get("role")
        if role is None:
            return None

        return {
            "role": role,
            "name": parameter_name,
            "index": getattr(
                array_access, "index", getattr(array_access, "index_expr", None)
            ),
            "member_path": member_path,
        }

    def hlsl_mesh_output_call_role_mapping(self, call, callee, role_by_name):
        args = getattr(call, "arguments", getattr(call, "args", [])) or []
        parameters = getattr(callee, "parameters", getattr(callee, "params", [])) or []
        callee_role_by_name = {}
        for index, parameter in enumerate(parameters):
            if index >= len(args):
                break

            parameter_name = getattr(parameter, "name", None)
            argument_name = self.expression_name(args[index])
            if not parameter_name or not argument_name:
                continue

            role = role_by_name.get(argument_name)
            if role is None:
                continue

            display_name = argument_name
            if isinstance(role, dict):
                display_name = role.get("name", argument_name)
                role = role.get("role")
            if role is None:
                continue

            callee_role_by_name[parameter_name] = {
                "role": role,
                "name": display_name,
            }
        return callee_role_by_name

    def validate_hlsl_mesh_output_helper_call(
        self,
        call,
        set_mesh_output_counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        active_helper_calls,
        role_parameters=None,
    ):
        helper_name = self.function_call_name(call)
        helper = (self.current_hlsl_available_functions or {}).get(helper_name)
        if helper is None:
            return

        helper_role_by_name = self.hlsl_mesh_output_call_role_mapping(
            call, helper, role_by_name
        )
        if not helper_role_by_name:
            return

        helper_id = id(helper)
        if helper_id in active_helper_calls:
            return

        active_helper_calls.add(helper_id)
        try:
            self.validate_hlsl_mesh_output_write_sequence(
                self.hlsl_statement_body_items(getattr(helper, "body", [])),
                set_mesh_output_counts_seen,
                helper_role_by_name,
                declared_counts,
                set_count_literals,
                active_helper_calls,
                visible_bool_constants=self.hlsl_initial_bool_constants(helper),
                visible_int_constants=self.hlsl_initial_int_constants(helper),
                role_parameters=role_parameters,
            )
        finally:
            active_helper_calls.remove(helper_id)

    def validate_hlsl_mesh_output_assignment(
        self,
        assignment,
        set_mesh_output_counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        visible_int_constants=None,
    ):
        target = getattr(assignment, "target", getattr(assignment, "left", None))
        target_info = self.hlsl_mesh_output_assignment_target_info(target, role_by_name)
        if target_info is None:
            return

        role = target_info["role"]
        parameter_name = target_info["name"]
        if not set_mesh_output_counts_seen:
            raise DirectXRefusal(
                f"DirectX mesh output array '{parameter_name}' must be written "
                "only after SetMeshOutputCounts"
            )

        if role == "indices" and target_info["member_path"]:
            raise DirectXRefusal(
                f"DirectX mesh indices output array '{parameter_name}' must be "
                "written as a whole uint2/uint3 element"
            )

        index_value = self.hlsl_integer_constant_value(
            target_info["index"], visible_int_constants
        )
        if index_value is None:
            return

        if index_value < 0:
            raise DirectXRefusal(
                f"DirectX mesh {role} output array '{parameter_name}' index "
                "must be non-negative"
            )

        declared_count = declared_counts.get(role)
        if declared_count is not None and index_value >= declared_count:
            raise DirectXRefusal(
                f"DirectX mesh {role} output array '{parameter_name}' index "
                f"({index_value}) must be less than declared array size "
                f"({declared_count})"
            )

        count_label = "numVertices" if role == "vertices" else "numPrimitives"
        set_count = set_count_literals.get(role)
        if set_count is not None and index_value >= set_count:
            raise DirectXRefusal(
                f"DirectX mesh {role} output array '{parameter_name}' index "
                f"({index_value}) must be less than SetMeshOutputCounts "
                f"{count_label} ({set_count})"
            )

    def validate_hlsl_mesh_output_write_sequence(
        self,
        statements,
        set_mesh_output_counts_seen,
        role_by_name,
        declared_counts,
        set_count_literals,
        active_helper_calls=None,
        visible_bool_constants=None,
        visible_int_constants=None,
        role_parameters=None,
    ):
        counts_seen, _ = self.validate_hlsl_mesh_output_write_sequence_result(
            statements,
            set_mesh_output_counts_seen,
            role_by_name,
            declared_counts,
            set_count_literals,
            active_helper_calls,
            visible_bool_constants=visible_bool_constants,
            visible_int_constants=visible_int_constants,
            role_parameters=role_parameters,
        )
        return counts_seen
