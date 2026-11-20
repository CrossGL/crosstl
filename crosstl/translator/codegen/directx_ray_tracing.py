"""Ray tracing and ray queries in the HLSL generator: system values and semantics, call arguments and types, mesh interpolation roles, exact scalars, returns and metadata attributes."""

from ..ast import (
    BlockNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    SwitchNode,
    WhileNode,
)
from ..refusals import DirectXRefusal
from .array_utils import split_array_type_suffix
from .directx_values import DirectXContextualConversionError
from .hlsl_extension_builtins import HLSL_NOPERSPECTIVE_BARYCENTRICS
from .hlsl_narrow_system_values import HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES
from .stage_refusals import RayStageInterfaceUnsupportedError


class DirectXRayTracingMixin:
    """Ray tracing members of ``HLSLCodeGen``."""

    def hlsl_interpolation_metadata_names(self, node):
        names = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            if qualifier is not None:
                names.append(str(qualifier).lower())
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is not None:
                names.append(str(attr_name).lower())
        return names

    def is_hlsl_interpolation_metadata_name(self, name):
        if name is None:
            return False
        normalized = str(name).lower()
        return (
            normalized in self.HLSL_INTERPOLATION_MODE_MODIFIERS
            or normalized in self.HLSL_INTERPOLATION_SAMPLING_MODIFIERS
            or normalized in {"smooth", "linear"}
        )

    def hlsl_interpolation_modifier_prefix(self, node):
        mode = None
        sampling = None
        for name in self.hlsl_interpolation_metadata_names(node):
            if mode is None:
                mode = self.HLSL_INTERPOLATION_MODE_MODIFIERS.get(name)
            if sampling is None:
                sampling = self.HLSL_INTERPOLATION_SAMPLING_MODIFIERS.get(name)
        if (
            mode is None
            and self.semantic_from_node(node) in HLSL_NOPERSPECTIVE_BARYCENTRICS
        ):
            mode = "noperspective"

        modifiers = []
        for modifier in (mode, sampling):
            if modifier and modifier not in modifiers:
                modifiers.append(modifier)
        return f"{' '.join(modifiers)} " if modifiers else ""

    def hlsl_mesh_parameter_role_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]
        if normalized in {"vertices", "indices", "primitives"}:
            return normalized
        return None

    def hlsl_mesh_payload_parameter_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized in {"mesh_payload", "hlsl_mesh_payload"}:
            return "payload"
        return None

    def hlsl_mesh_parameter_roles(self, parameter):
        roles = []
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"vertices", "indices", "primitives"}:
                roles.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            role = self.hlsl_mesh_parameter_role_attribute_name(attr)
            if role:
                roles.append(role)

            payload_role = self.hlsl_mesh_payload_parameter_attribute_name(attr)
            if payload_role:
                roles.append(payload_role)

        ordered_roles = []
        for role in ("payload", "vertices", "indices", "primitives"):
            if role in roles:
                ordered_roles.append(role)
        return ordered_roles

    def hlsl_parameter_direction_qualifiers(self, parameter):
        return {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
            if str(qualifier).lower() in {"const", "in", "out", "inout"}
        }

    def apply_hlsl_parameter_qualifiers(self, param_type, parameter):
        qualifiers = self.hlsl_parameter_qualifiers(parameter)
        if self.is_hlsl_global_resource_type(param_type):
            qualifiers = [qualifier for qualifier in qualifiers if qualifier != "const"]
        if self.hlsl_has_precise_modifier(parameter):
            qualifiers.append("precise")
        if not qualifiers:
            return param_type
        return f"{' '.join(qualifiers)} {param_type}"

    def hlsl_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.hlsl_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def validate_hlsl_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.hlsl_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.hlsl_parameter_is_array(parameter):
                raise DirectXRefusal(
                    "DirectX geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.hlsl_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise DirectXRefusal(
                    "DirectX geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_hlsl_domain_location_components(self, func, parameters):
        domain = self.canonical_hlsl_tessellation_domain(
            self.normalized_hlsl_stage_attribute_argument(func, "domain")
        )
        expected_counts = {
            "tri": 3,
            "quad": 2,
            "isoline": 2,
        }
        expected_count = expected_counts.get(domain)
        if expected_count is None:
            return

        for parameter in parameters:
            if not self.hlsl_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            component_count = self.hlsl_parameter_component_count(parameter)
            if component_count is None:
                continue
            if component_count != expected_count:
                raise DirectXRefusal(
                    "DirectX tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must have {expected_count} "
                    f"component(s) for {domain} domains, got {component_count}"
                )

    def validate_hlsl_domain_location_type(self, parameters):
        floating_scalar_bases = (
            "min16float",
            "min10float",
            "float",
            "half",
            "double",
        )
        for parameter in parameters:
            if not self.hlsl_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            base_type, array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue

            is_floating_type = False
            for scalar_base in floating_scalar_bases:
                if not base_type.startswith(scalar_base):
                    continue
                suffix = base_type[len(scalar_base) :]
                if suffix in {"", "2", "3", "4"}:
                    is_floating_type = True
                break

            if array_suffix or not is_floating_type:
                raise DirectXRefusal(
                    "DirectX tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must be a floating-point "
                    "scalar or vector"
                )

    def validate_hlsl_scalar_integer_semantic_type(
        self, parameters, shader_type, semantic
    ):
        for parameter in parameters:
            if not self.hlsl_parameter_has_semantic(parameter, semantic):
                continue

            base_type, array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue
            if array_suffix or base_type not in {"int", "uint"}:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    "must be scalar int or uint"
                )

    def validate_hlsl_scalar_integer_semantic_types(
        self, parameters, shader_type, semantics
    ):
        for semantic in semantics:
            self.validate_hlsl_scalar_integer_semantic_type(
                parameters, shader_type, semantic
            )

    def validate_hlsl_exact_semantic_type(
        self, parameters, shader_type, semantic, expected_type, expected_description
    ):
        for parameter in parameters:
            if not self.hlsl_parameter_has_semantic(parameter, semantic):
                continue

            base_type, array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue
            if array_suffix or base_type != expected_type:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_hlsl_semantic_type_one_of(
        self, parameters, shader_type, semantic, expected_types, expected_description
    ):
        for parameter in parameters:
            if not self.hlsl_parameter_has_semantic(parameter, semantic):
                continue

            base_type, array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue
            if array_suffix or base_type not in expected_types:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_hlsl_thread_system_value_types(self, parameters, shader_type):
        self.validate_hlsl_exact_semantic_type(
            parameters, shader_type, "SV_GroupIndex", "uint", "scalar uint"
        )
        for semantic in (
            "SV_GroupID",
            "SV_GroupThreadID",
            "SV_DispatchThreadID",
            "SV_DispatchMeshID",
        ):
            self.validate_hlsl_exact_semantic_type(
                parameters, shader_type, semantic, "uint3", "uint3"
            )

    def validate_hlsl_mesh_view_id_system_value(self, parameters, shader_type):
        if not any(
            self.hlsl_parameter_has_semantic(parameter, "SV_ViewID")
            for parameter in parameters
        ):
            return

        if shader_type in {"task", "amplification", "object"}:
            raise DirectXRefusal(
                f"DirectX {shader_type} stage cannot use SV_ViewID; "
                "SV_ViewID is only valid as a mesh stage input"
            )

        self.validate_hlsl_exact_semantic_type(
            parameters, shader_type, "SV_ViewID", "uint", "scalar uint"
        )

    def validate_hlsl_fragment_system_value_types(self, parameters):
        for parameter in parameters:
            semantic = self.semantic_from_node(parameter)
            if semantic is None:
                continue
            if str(semantic).lower() == "gl_samplemask":
                raise DirectXRefusal(
                    "DirectX fragment stage gl_SampleMask parameter "
                    f"'{parameter.name}' cannot use output-only sample mask "
                    "semantic"
                )

        self.validate_hlsl_exact_semantic_type(
            parameters, "fragment", "SV_Coverage", "uint", "scalar uint"
        )

    def validate_hlsl_compute_system_value_types(self, parameters):
        # A 16-bit unsigned parameter is declared with the 32-bit system value
        # and narrowed in the body (hlsl_narrow_system_values).
        narrow_scalars = {
            name
            for name in HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES
            if not name[-1].isdigit()
        }
        self.validate_hlsl_semantic_type_one_of(
            parameters,
            "compute",
            "SV_GroupIndex",
            {"uint"} | narrow_scalars,
            "scalar uint or a 16-bit unsigned scalar",
        )
        for semantic in ("SV_GroupID", "SV_GroupThreadID", "SV_DispatchThreadID"):
            # HLSL accepts a scalar uint for these compute system values as a
            # truncated (x-component) binding, matching Metal kernels that declare
            # a 1-D `uint` [[threadgroup_position_in_grid]] / [[thread_position...]]
            # parameter (e.g. softmax, rms_norm, logsumexp).
            self.validate_hlsl_semantic_type_one_of(
                parameters,
                "compute",
                semantic,
                {"uint", "uint2", "uint3"} | HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES,
                "scalar uint, uint2, or uint3, or a 16-bit unsigned form of one",
            )

    def hlsl_ray_stage_types(self):
        return {
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "closesthit",
            "anyhit",
            "miss",
            "callable",
        }

    def hlsl_ray_semantic_role(self, parameter):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return None
        mapped_semantic = self.hlsl_canonical_semantic(semantic)
        normalized = str(mapped_semantic).lower()
        if normalized in {"payload", "hit_attribute", "callable_data"}:
            return normalized
        return None

    def apply_hlsl_ray_parameter_direction(self, param_type, role):
        direction = {
            "payload": "inout",
            "hit_attribute": "in",
            "callable_data": "inout",
        }.get(role)
        if direction is None:
            return param_type
        return f"{direction} {param_type}"

    def hlsl_ray_role_parameters(self, parameters):
        role_parameters = {}
        for parameter in parameters:
            role = self.hlsl_ray_semantic_role(parameter)
            if role:
                role_parameters.setdefault(role, []).append(parameter)
        return role_parameters

    def validate_hlsl_ray_parameter_type(self, parameter, role):
        base_type, array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return

        allowed_builtin_types = (
            {"BuiltInTriangleIntersectionAttributes"}
            if role == "hit_attribute"
            else set()
        )
        expected_type_description = (
            "user-defined struct type or BuiltInTriangleIntersectionAttributes"
            if allowed_builtin_types
            else "user-defined struct type"
        )
        if array_suffix or (
            base_type not in self.structs_by_name
            and base_type not in allowed_builtin_types
        ):
            raise DirectXRefusal(
                f"DirectX ray {role} parameter '{parameter.name}' must use a "
                f"{expected_type_description}"
            )

    def validate_hlsl_ray_stage_parameters(self, func, shader_type, parameters):
        if shader_type not in self.hlsl_ray_stage_types():
            return

        role_parameters = self.hlsl_ray_role_parameters(parameters)
        allowed_stages = {
            "payload": {
                "ray_closest_hit",
                "ray_any_hit",
                "ray_miss",
                "closesthit",
                "anyhit",
                "miss",
            },
            "hit_attribute": {
                "ray_closest_hit",
                "ray_any_hit",
                "closesthit",
                "anyhit",
            },
            "callable_data": {"ray_callable", "callable"},
        }
        for role, role_params in role_parameters.items():
            if len(role_params) > 1:
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage must declare at most one "
                    f"{role} parameter"
                )
            if shader_type not in allowed_stages.get(role, set()):
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage cannot use {role} parameter "
                    f"'{role_params[0].name}'"
                )
            self.validate_hlsl_ray_parameter_type(role_params[0], role)

        required_roles = {
            "ray_closest_hit": {"payload", "hit_attribute"},
            "closesthit": {"payload", "hit_attribute"},
            "ray_any_hit": {"payload", "hit_attribute"},
            "anyhit": {"payload", "hit_attribute"},
            "ray_miss": {"payload"},
            "miss": {"payload"},
            "ray_callable": {"callable_data"},
            "callable": {"callable_data"},
        }
        for role in sorted(
            required_roles.get(shader_type, set()) - set(role_parameters)
        ):
            raise RayStageInterfaceUnsupportedError(
                "directx",
                shader_type,
                role,
                "the entry declares none and the program declares no ray interface "
                "variable (GLSL rayPayloadInEXT, hitAttributeEXT, callableDataInEXT) "
                "to pass as one; DXR requires it (DirectX Raytracing spec, shader "
                "signatures)",
            )

        exact_role_order = {
            "ray_generation": [],
            "ray_intersection": [],
            "ray_closest_hit": ["payload", "hit_attribute"],
            "ray_any_hit": ["payload", "hit_attribute"],
            "ray_miss": ["payload"],
            "ray_callable": ["callable_data"],
            "intersection": [],
            "closesthit": ["payload", "hit_attribute"],
            "anyhit": ["payload", "hit_attribute"],
            "miss": ["payload"],
            "callable": ["callable_data"],
        }
        expected_roles = exact_role_order.get(shader_type)
        if expected_roles is None:
            return

        extra_parameters = [
            parameter
            for parameter in parameters
            if self.hlsl_ray_semantic_role(parameter) is None
        ]
        if extra_parameters:
            if expected_roles:
                expected = ", ".join(expected_roles)
                raise DirectXRefusal(
                    f"DirectX {shader_type} stage parameter "
                    f"'{extra_parameters[0].name}' must be one of: {expected}"
                )
            raise DirectXRefusal(
                f"DirectX {shader_type} stage must not declare entry parameters"
            )

        role_sequence = [
            self.hlsl_ray_semantic_role(parameter) for parameter in parameters
        ]
        if role_sequence != expected_roles:
            expected = ", ".join(expected_roles) if expected_roles else "no parameters"
            actual = ", ".join(role_sequence) if role_sequence else "no parameters"
            raise DirectXRefusal(
                f"DirectX {shader_type} stage parameters must be declared as "
                f"{expected}, got {actual}"
            )

    def hlsl_ray_tracing_calls_in_node(self, root):
        calls = []
        for node in self.walk_ast(root):
            if isinstance(node, RayTracingOpNode):
                calls.append(
                    (getattr(node, "operation", None), getattr(node, "arguments", []))
                )
            elif isinstance(node, FunctionCallNode):
                name = self.function_call_name(node)
                if name in {
                    "TraceRay",
                    "CallShader",
                    "ReportHit",
                    "AcceptHitAndEndSearch",
                    "IgnoreHit",
                }:
                    calls.append(
                        (name, getattr(node, "arguments", getattr(node, "args", [])))
                    )
        return calls

    def hlsl_ray_tracing_calls(self, func):
        return self.hlsl_ray_tracing_calls_in_node(getattr(func, "body", []))

    def hlsl_ray_query_method_return_type(self, operation):
        return {
            "Proceed": "bool",
            "Abort": "void",
            "TraceRayInline": "void",
            "CommitNonOpaqueTriangleHit": "void",
            "CommitProceduralPrimitiveHit": "void",
            "CandidateType": "uint",
            "CommittedStatus": "uint",
            "CandidateProceduralPrimitiveNonOpaque": "bool",
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
            "CandidateInstanceContributionToHitGroupIndex": "uint",
            "CommittedInstanceContributionToHitGroupIndex": "uint",
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

    def hlsl_ray_query_method_names(self):
        return {
            *(
                "Proceed",
                "Abort",
                "TraceRayInline",
                "CommitNonOpaqueTriangleHit",
                "CommitProceduralPrimitiveHit",
                "CandidateType",
                "CommittedStatus",
                "CandidateProceduralPrimitiveNonOpaque",
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
                "CandidateInstanceContributionToHitGroupIndex",
                "CommittedInstanceContributionToHitGroupIndex",
                "CandidateObjectRayTMin",
                "CandidateTriangleBarycentrics",
                "CommittedTriangleBarycentrics",
                "CandidateTriangleFrontFace",
                "CommittedTriangleFrontFace",
                "CandidateObjectToWorld3x4",
                "CandidateWorldToObject3x4",
                "CommittedObjectToWorld3x4",
                "CommittedWorldToObject3x4",
            )
        }

    def hlsl_ray_query_call_parts(self, node):
        if isinstance(node, RayQueryOpNode):
            return (
                getattr(node, "operation", None),
                getattr(node, "query_expr", None),
                getattr(node, "arguments", []),
            )
        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if not isinstance(func_expr, MemberAccessNode):
            return None

        operation = str(getattr(func_expr, "member", ""))
        if operation not in self.hlsl_ray_query_method_names():
            return None
        return (
            operation,
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None)),
            getattr(node, "arguments", getattr(node, "args", [])),
        )

    def hlsl_ray_query_calls_in_node(self, root):
        calls = []
        for node in self.walk_ast(root):
            call = self.hlsl_ray_query_call_parts(node)
            if call is not None:
                calls.append(call)
        return calls

    def hlsl_ray_query_calls(self, func):
        return self.hlsl_ray_query_calls_in_node(getattr(func, "body", []))

    def validate_hlsl_ray_struct_argument(self, argument, shader_type, operation, role):
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return

        mapped_type = self.map_type(argument_type)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        allowed_builtin_types = (
            {"BuiltInTriangleIntersectionAttributes"}
            if role == "hit attribute"
            else set()
        )
        expected_type_description = (
            "user-defined struct type or BuiltInTriangleIntersectionAttributes"
            if allowed_builtin_types
            else "user-defined struct type"
        )
        if array_suffix or (
            base_type not in self.structs_by_name
            and base_type not in allowed_builtin_types
        ):
            raise DirectXRefusal(
                f"DirectX {shader_type} {operation} {role} argument must use "
                f"a {expected_type_description}, got {mapped_type}"
            )

    def hlsl_expression_mapped_base_and_array_suffix(self, expr):
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return None, ""

        mapped_type = self.map_type(expr_type)
        return split_array_type_suffix(mapped_type)

    def validate_hlsl_ray_exact_type_argument(
        self, argument, shader_type, operation, role, expected_type
    ):
        base_type, array_suffix = self.hlsl_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type != expected_type:
            actual_type = f"{base_type}{array_suffix}"
            raise DirectXRefusal(
                f"DirectX {shader_type} {operation} {role} argument must be "
                f"{expected_type}, got {actual_type}"
            )

    def validate_hlsl_trace_ray_exact_type_argument(
        self, argument, shader_type, role, expected_type
    ):
        self.validate_hlsl_ray_exact_type_argument(
            argument, shader_type, "TraceRay", role, expected_type
        )

    def validate_hlsl_ray_query_receiver(self, query_expr, shader_type, operation):
        base_type, array_suffix = self.hlsl_expression_mapped_base_and_array_suffix(
            query_expr
        )
        if base_type is None:
            return
        if not (
            not array_suffix
            and (base_type == "RayQuery" or base_type.startswith("RayQuery<"))
        ):
            actual_type = f"{base_type}{array_suffix}"
            raise DirectXRefusal(
                f"DirectX {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery, got {actual_type}"
            )

    def validate_hlsl_ray_scalar_float_argument(
        self, argument, shader_type, operation, role
    ):
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return
        if not self.is_scalar_floating_type(argument_type):
            raise DirectXRefusal(
                f"DirectX {shader_type} {operation} {role} argument must be "
                f"scalar floating, got {self.map_type(argument_type)}"
            )

    def validate_hlsl_ray_instance_inclusion_mask_argument(
        self, argument, shader_type, operation
    ):
        self.validate_hlsl_scalar_int_uint_expression(
            argument,
            f"DirectX {shader_type} {operation} instance inclusion mask argument",
        )
        mask_value = self.literal_int_value(
            argument, self.hlsl_current_visible_int_constants()
        )
        if mask_value is None or 0 <= mask_value <= 0xFF:
            return
        raise DirectXRefusal(
            f"DirectX {shader_type} {operation} instance inclusion mask argument "
            f"must be in the range 0 to 255, got {mask_value}"
        )

    def hlsl_ray_flag_literal_constants(self):
        constants = dict(self.HLSL_RAY_FLAG_VALUES)
        constants.update(self.hlsl_current_visible_int_constants())
        return constants

    def hlsl_ray_flag_literal_int_value(self, expr):
        constants = self.hlsl_ray_flag_literal_constants()
        value = self.literal_int_value(expr, constants)
        if value is not None:
            return value

        class_name = expr.__class__.__name__
        if "UnaryOp" in class_name:
            operand = self.hlsl_ray_flag_literal_int_value(
                getattr(expr, "operand", None)
            )
            if operand is None:
                return None
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "~":
                return ~operand
            return None

        if "BinaryOp" not in class_name:
            return None

        left = self.hlsl_ray_flag_literal_int_value(getattr(expr, "left", None))
        right = self.hlsl_ray_flag_literal_int_value(getattr(expr, "right", None))
        if left is None or right is None:
            return None

        operator = getattr(expr, "operator", getattr(expr, "op", None))
        if operator == "|":
            return left | right
        if operator == "&":
            return left & right
        if operator == "^":
            return left ^ right
        if operator == "<<":
            return left << right
        if operator == ">>":
            return left >> right
        return None

    def validate_hlsl_ray_flags_argument(self, argument, shader_type, operation):
        diagnostic = f"DirectX {shader_type} {operation} ray flags argument"
        self.validate_hlsl_scalar_int_uint_expression(argument, diagnostic)

        flags = self.hlsl_ray_flag_literal_int_value(argument)
        if flags is None:
            return
        if flags < 0 or flags & ~self.HLSL_RAY_FLAG_KNOWN_MASK:
            raise DirectXRefusal(
                f"{diagnostic} may only use known RAY_FLAG bits "
                f"0x0 to 0x{self.HLSL_RAY_FLAG_KNOWN_MASK:X}, got {flags}"
            )

        for group in self.HLSL_RAY_FLAG_MUTUALLY_EXCLUSIVE_GROUPS:
            active = [name for name in group if flags & self.HLSL_RAY_FLAG_VALUES[name]]
            if len(active) > 1:
                raise DirectXRefusal(
                    f"{diagnostic} combines mutually exclusive ray flags: "
                    f"{', '.join(active)}"
                )

    def validate_hlsl_trace_ray_arguments(self, args, shader_type):
        self.validate_hlsl_trace_ray_exact_type_argument(
            args[0],
            shader_type,
            "acceleration structure",
            "RaytracingAccelerationStructure",
        )
        self.validate_hlsl_ray_flags_argument(args[1], shader_type, "TraceRay")
        for index, role in (
            (3, "ray contribution to hit group index"),
            (4, "geometry contribution multiplier"),
            (5, "miss shader index"),
        ):
            self.validate_hlsl_scalar_int_uint_expression(
                args[index], f"DirectX {shader_type} TraceRay {role} argument"
            )
        self.validate_hlsl_ray_instance_inclusion_mask_argument(
            args[2], shader_type, "TraceRay"
        )
        if len(args) == 8:
            self.validate_hlsl_trace_ray_exact_type_argument(
                args[6], shader_type, "ray descriptor", "RayDesc"
            )
        else:
            self.validate_hlsl_trace_ray_exact_type_argument(
                args[6], shader_type, "origin", "float3"
            )
            self.validate_hlsl_ray_scalar_float_argument(
                args[7], shader_type, "TraceRay", "minimum distance"
            )
            self.validate_hlsl_trace_ray_exact_type_argument(
                args[8], shader_type, "direction", "float3"
            )
            self.validate_hlsl_ray_scalar_float_argument(
                args[9], shader_type, "TraceRay", "maximum distance"
            )
        self.validate_hlsl_ray_struct_argument(
            args[-1], shader_type, "TraceRay", "payload"
        )

    def validate_hlsl_call_shader_arguments(self, args, shader_type):
        self.validate_hlsl_scalar_int_uint_expression(
            args[0], f"DirectX {shader_type} CallShader shader index argument"
        )
        self.validate_hlsl_ray_struct_argument(
            args[1], shader_type, "CallShader", "callable data"
        )

    def validate_hlsl_report_hit_arguments(self, args, shader_type):
        self.validate_hlsl_ray_scalar_float_argument(
            args[0], shader_type, "ReportHit", "hit distance"
        )
        self.validate_hlsl_scalar_int_uint_expression(
            args[1], f"DirectX {shader_type} ReportHit hit kind argument"
        )
        if len(args) == 3:
            self.validate_hlsl_ray_struct_argument(
                args[2], shader_type, "ReportHit", "hit attribute"
            )

    def validate_hlsl_ray_query_call_arguments(
        self, operation, query_expr, args, shader_type
    ):
        self.validate_hlsl_ray_query_receiver(query_expr, shader_type, operation)

        expected_arg_counts = {
            # Seven operands are the GLSL rayQueryInitializeEXT shape, whose
            # ray operands are packed into a RayDesc on emission.
            "TraceRayInline": {4, 7},
            "CommitProceduralPrimitiveHit": {1},
        }
        default_expected_counts = {0}
        expected_counts = expected_arg_counts.get(operation, default_expected_counts)
        if len(args) not in expected_counts:
            expected = " or ".join(str(count) for count in sorted(expected_counts))
            raise DirectXRefusal(
                f"DirectX {shader_type} RayQuery.{operation} requires {expected} "
                f"argument(s), got {len(args)}"
            )

        if operation == "TraceRayInline":
            self.validate_hlsl_ray_exact_type_argument(
                args[0],
                shader_type,
                "RayQuery.TraceRayInline",
                "acceleration structure",
                "RaytracingAccelerationStructure",
            )
            self.validate_hlsl_ray_flags_argument(
                args[1],
                shader_type,
                "RayQuery.TraceRayInline",
            )
            self.validate_hlsl_ray_instance_inclusion_mask_argument(
                args[2],
                shader_type,
                "RayQuery.TraceRayInline",
            )
            if len(args) == 4:
                self.validate_hlsl_ray_exact_type_argument(
                    args[3],
                    shader_type,
                    "RayQuery.TraceRayInline",
                    "ray descriptor",
                    "RayDesc",
                )
        elif operation == "CommitProceduralPrimitiveHit":
            self.validate_hlsl_ray_scalar_float_argument(
                args[0],
                shader_type,
                "RayQuery.CommitProceduralPrimitiveHit",
                "hit distance",
            )

    def validate_hlsl_ray_tracing_call_arguments(self, operation, args, shader_type):
        if operation == "TraceRay":
            self.validate_hlsl_trace_ray_arguments(args, shader_type)
        elif operation == "CallShader":
            self.validate_hlsl_call_shader_arguments(args, shader_type)
        elif operation == "ReportHit":
            self.validate_hlsl_report_hit_arguments(args, shader_type)

    def hlsl_ray_tracing_allowed_stages(self):
        return {
            "TraceRay": {
                "ray_generation",
                "ray_closest_hit",
                "ray_miss",
                "closesthit",
                "miss",
            },
            "CallShader": {
                "ray_generation",
                "ray_closest_hit",
                "ray_miss",
                "ray_callable",
                "closesthit",
                "miss",
                "callable",
            },
            "ReportHit": {"ray_intersection", "intersection"},
            "AcceptHitAndEndSearch": {"ray_any_hit", "anyhit"},
            "IgnoreHit": {"ray_any_hit", "anyhit"},
        }

    def hlsl_ray_tracing_expected_arg_counts(self):
        return {
            "TraceRay": {8, 11},
            "CallShader": {2},
            "ReportHit": {2, 3},
            "AcceptHitAndEndSearch": {0},
            "IgnoreHit": {0},
        }

    def validate_hlsl_ray_tracing_call(
        self, operation, args, shader_type, allowed_stages, expected_arg_counts
    ):
        if operation not in allowed_stages:
            return
        if shader_type is not None and shader_type not in allowed_stages[operation]:
            valid_stages = ", ".join(sorted(allowed_stages[operation]))
            raise DirectXRefusal(
                f"DirectX {shader_type} stage cannot call {operation}; "
                f"{operation} is only valid in: {valid_stages}"
            )
        expected_counts = expected_arg_counts[operation]
        if len(args) not in expected_counts:
            expected = " or ".join(str(count) for count in sorted(expected_counts))
            raise DirectXRefusal(
                f"DirectX {shader_type} {operation} requires {expected} "
                f"argument(s), got {len(args)}"
            )
        self.validate_hlsl_ray_tracing_call_arguments(operation, args, shader_type)

    def validate_hlsl_ray_tracing_call_sequence(
        self,
        statements,
        shader_type,
        allowed_stages,
        expected_arg_counts,
        visible_int_constants,
    ):
        previous_visible_int_constants = self.current_hlsl_visible_int_constants
        self.current_hlsl_visible_int_constants = visible_int_constants
        try:
            for stmt in self.hlsl_statement_body_items(statements):
                if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                    self.validate_hlsl_ray_tracing_call_sequence(
                        self.hlsl_statement_body_items(stmt),
                        shader_type,
                        allowed_stages,
                        expected_arg_counts,
                        dict(visible_int_constants),
                    )
                    continue

                if isinstance(stmt, IfNode):
                    self.validate_hlsl_ray_tracing_call_sequence(
                        self.hlsl_statement_body_items(
                            getattr(stmt, "then_branch", getattr(stmt, "if_body", None))
                        ),
                        shader_type,
                        allowed_stages,
                        expected_arg_counts,
                        dict(visible_int_constants),
                    )
                    else_branch = getattr(
                        stmt, "else_branch", getattr(stmt, "else_body", None)
                    )
                    if else_branch is not None:
                        self.validate_hlsl_ray_tracing_call_sequence(
                            self.hlsl_statement_body_items(else_branch),
                            shader_type,
                            allowed_stages,
                            expected_arg_counts,
                            dict(visible_int_constants),
                        )
                    continue

                if isinstance(stmt, SwitchNode):
                    case_entries = self.hlsl_switch_case_entries(stmt)
                    for start_index in range(len(case_entries)):
                        self.validate_hlsl_ray_tracing_call_sequence(
                            self.hlsl_switch_fallthrough_path_body(
                                case_entries, start_index
                            ),
                            shader_type,
                            allowed_stages,
                            expected_arg_counts,
                            dict(visible_int_constants),
                        )
                    continue

                if isinstance(
                    stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)
                ):
                    self.validate_hlsl_ray_tracing_call_sequence(
                        self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                        shader_type,
                        allowed_stages,
                        expected_arg_counts,
                        dict(visible_int_constants),
                    )
                    continue

                for operation, args in self.hlsl_ray_tracing_calls_in_node(stmt):
                    self.validate_hlsl_ray_tracing_call(
                        operation,
                        args,
                        shader_type,
                        allowed_stages,
                        expected_arg_counts,
                    )
                self.hlsl_update_visible_int_constants(stmt, visible_int_constants)
        finally:
            self.current_hlsl_visible_int_constants = previous_visible_int_constants

    def validate_hlsl_ray_tracing_calls(self, func, shader_type):
        calls = self.hlsl_ray_tracing_calls(func)
        if not calls:
            return

        allowed_stages = self.hlsl_ray_tracing_allowed_stages()
        expected_arg_counts = self.hlsl_ray_tracing_expected_arg_counts()
        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {
            **previous_local_variable_types,
            **self.function_scope_variable_types(func),
        }
        try:
            self.validate_hlsl_ray_tracing_call_sequence(
                self.hlsl_statement_body_items(getattr(func, "body", [])),
                shader_type,
                allowed_stages,
                expected_arg_counts,
                self.hlsl_initial_int_constants(func),
            )
        finally:
            self.local_variable_types = previous_local_variable_types

    def validate_hlsl_ray_query_call_sequence(
        self, statements, shader_type, visible_int_constants
    ):
        previous_visible_int_constants = self.current_hlsl_visible_int_constants
        self.current_hlsl_visible_int_constants = visible_int_constants
        try:
            for stmt in self.hlsl_statement_body_items(statements):
                if isinstance(stmt, BlockNode) or hasattr(stmt, "statements"):
                    self.validate_hlsl_ray_query_call_sequence(
                        self.hlsl_statement_body_items(stmt),
                        shader_type,
                        dict(visible_int_constants),
                    )
                    continue

                if isinstance(stmt, IfNode):
                    self.validate_hlsl_ray_query_call_sequence(
                        self.hlsl_statement_body_items(
                            getattr(stmt, "then_branch", getattr(stmt, "if_body", None))
                        ),
                        shader_type,
                        dict(visible_int_constants),
                    )
                    else_branch = getattr(
                        stmt, "else_branch", getattr(stmt, "else_body", None)
                    )
                    if else_branch is not None:
                        self.validate_hlsl_ray_query_call_sequence(
                            self.hlsl_statement_body_items(else_branch),
                            shader_type,
                            dict(visible_int_constants),
                        )
                    continue

                if isinstance(stmt, SwitchNode):
                    case_entries = self.hlsl_switch_case_entries(stmt)
                    for start_index in range(len(case_entries)):
                        self.validate_hlsl_ray_query_call_sequence(
                            self.hlsl_switch_fallthrough_path_body(
                                case_entries, start_index
                            ),
                            shader_type,
                            dict(visible_int_constants),
                        )
                    continue

                if isinstance(
                    stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)
                ):
                    self.validate_hlsl_ray_query_call_sequence(
                        self.hlsl_statement_body_items(getattr(stmt, "body", None)),
                        shader_type,
                        dict(visible_int_constants),
                    )
                    continue

                for operation, query_expr, args in self.hlsl_ray_query_calls_in_node(
                    stmt
                ):
                    self.validate_hlsl_ray_query_call_arguments(
                        operation, query_expr, args, shader_type
                    )
                self.hlsl_update_visible_int_constants(stmt, visible_int_constants)
        finally:
            self.current_hlsl_visible_int_constants = previous_visible_int_constants

    def validate_hlsl_ray_query_calls(self, func, shader_type):
        calls = self.hlsl_ray_query_calls(func)
        if not calls:
            return

        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {
            **previous_local_variable_types,
            **self.function_scope_variable_types(func),
        }
        try:
            self.validate_hlsl_ray_query_call_sequence(
                self.hlsl_statement_body_items(getattr(func, "body", [])),
                shader_type,
                self.hlsl_initial_int_constants(func),
            )
        finally:
            self.local_variable_types = previous_local_variable_types

    def hlsl_nonuniform_resource_index_return_type(self, args):
        if len(args) != 1:
            return "uint"

        argument_type = self.expression_result_type(args[0])
        if argument_type is None:
            return "uint"

        mapped_type = self.map_type(argument_type)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        if not array_suffix and base_type in {"int", "uint"}:
            return base_type
        return "uint"

    def validate_hlsl_nonuniform_resource_index_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            if (
                self.hlsl_function_call_name(self.function_call_name(node))
                != "NonUniformResourceIndex"
            ):
                continue
            calls.append(getattr(node, "arguments", getattr(node, "args", [])))

        if not calls:
            return

        previous_local_variable_types = self.local_variable_types
        self.local_variable_types = {
            **previous_local_variable_types,
            **self.function_scope_variable_types(func),
        }
        try:
            for args in calls:
                if len(args) != 1:
                    raise DirectXRefusal(
                        "DirectX NonUniformResourceIndex requires exactly "
                        f"1 argument, got {len(args)}"
                    )

                argument_type = self.expression_result_type(args[0])
                if argument_type is None:
                    continue
                argument_type_name = self.type_name_string(argument_type)
                if argument_type_name in {"let", "auto"}:
                    continue

                mapped_type = self.map_type(argument_type_name)
                base_type, array_suffix = split_array_type_suffix(mapped_type)
                if array_suffix or base_type not in {"int", "uint"}:
                    raise DirectXContextualConversionError(
                        "DirectX NonUniformResourceIndex index argument must be "
                        f"scalar int or uint, got {mapped_type}",
                        source_type=mapped_type,
                        target_type="int or uint",
                        reason="non-uniform-resource-index-not-integer",
                    )
        finally:
            self.local_variable_types = previous_local_variable_types

    def validate_hlsl_function_return_semantic(
        self, func, shader_type, raw_return_type=None, semantic=None
    ):
        if semantic is None:
            semantic = self.hlsl_function_return_semantic(func)
        if semantic is None:
            return
        mapped_semantic = self.hlsl_canonical_semantic(semantic)
        function_name = getattr(func, "name", "<anonymous>")
        actual_type = self.map_type(
            raw_return_type or getattr(func, "return_type", None)
        )
        actual_base_type, array_suffix = split_array_type_suffix(actual_type)
        if not array_suffix and actual_base_type == "void":
            raise DirectXRefusal(
                f"DirectX {shader_type} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_IsFrontFace",
            "gl_FragCoord",
            "gl_FrontFacing",
            "gl_PointCoord",
            "gl_SampleMaskIn",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "SV_VertexID",
            "SV_InstanceID",
            "SV_PrimitiveID",
            "PRIMITIVE_ID",
            "FRONT_FACE",
            "SV_IsFrontFace",
            "SV_GroupIndex",
            "SV_GroupID",
            "SV_GroupThreadID",
            "SV_DispatchThreadID",
        }
        if (
            str(semantic) in invalid_return_semantics
            or mapped_semantic in invalid_return_semantics
        ):
            raise DirectXRefusal(
                f"DirectX {shader_type} semantic '{semantic}' cannot be used "
                "as a function return semantic"
            )

        if self.hlsl_flexible_render_target_semantic(semantic, mapped_semantic):
            if array_suffix or not self.hlsl_render_target_type(actual_base_type):
                raise DirectXRefusal(
                    f"DirectX {shader_type} return semantic '{semantic}' maps to "
                    f"'{mapped_semantic}' and requires function '{function_name}' "
                    "to return a scalar or vector numeric render target type, "
                    f"got {actual_type}"
                )
            return

        expected_type = self.hlsl_output_builtin_semantic_type(mapped_semantic)
        if expected_type is None:
            return

        if array_suffix or not self.hlsl_builtin_semantic_type_matches(
            mapped_semantic, actual_base_type, expected_type
        ):
            raise DirectXRefusal(
                f"DirectX {shader_type} return semantic '{semantic}' maps to "
                f"'{mapped_semantic}' and requires function '{function_name}' "
                f"to return {expected_type}, got {actual_type}"
            )

    def validate_hlsl_struct_member_semantic_type(
        self, struct_name, member_name, member_type, semantic
    ):
        if semantic is None:
            return

        mapped_semantic = self.hlsl_canonical_semantic(semantic)
        self.validate_hlsl_tess_factor_member_semantic_type(
            struct_name, member_name, member_type, semantic, mapped_semantic
        )
        expected_type = self.hlsl_output_builtin_semantic_type(mapped_semantic)
        if expected_type is None:
            return

        actual_type = self.map_type(member_type)
        actual_base_type, array_suffix = split_array_type_suffix(actual_type)
        if self.hlsl_flexible_render_target_semantic(semantic, mapped_semantic):
            if array_suffix or not self.hlsl_render_target_type(actual_base_type):
                raise DirectXRefusal(
                    f"DirectX struct '{struct_name}' semantic '{semantic}' maps to "
                    f"'{mapped_semantic}' and requires member '{member_name}' to "
                    "have a scalar or vector numeric render target type, "
                    f"got {actual_type}"
                )
            return

        if array_suffix or not self.hlsl_builtin_semantic_type_matches(
            mapped_semantic, actual_base_type, expected_type
        ):
            raise DirectXRefusal(
                f"DirectX struct '{struct_name}' semantic '{semantic}' maps to "
                f"'{mapped_semantic}' and requires member '{member_name}' to "
                f"have type {expected_type}, got {actual_type}"
            )

    # A render target written as a 16-bit four-component vector keeps its
    # meaning: SV_Target accepts any numeric vector, and only the width differs.
    HLSL_HALF_RENDER_TARGET_TYPES = frozenset({"float16_t4", "half4", "min16float4"})

    def hlsl_builtin_semantic_type_matches(
        self, mapped_semantic, actual_base_type, expected_type
    ):
        if actual_base_type == expected_type:
            return True
        return (
            expected_type == "float4"
            and self.hlsl_render_target_semantic(mapped_semantic)
            and actual_base_type in self.HLSL_HALF_RENDER_TARGET_TYPES
        )

    def validate_hlsl_tess_factor_member_semantic_type(
        self, struct_name, member_name, member_type, semantic, mapped_semantic=None
    ):
        if mapped_semantic is None:
            mapped_semantic = self.hlsl_canonical_semantic(semantic)
        if str(mapped_semantic).lower() not in {
            "sv_tessfactor",
            "sv_insidetessfactor",
        }:
            return

        actual_type = self.map_type(member_type)
        actual_base_type, array_suffix = split_array_type_suffix(actual_type)
        floating_scalar_bases = ("min16float", "float", "half", "double")

        if array_suffix:
            valid_type = actual_base_type in floating_scalar_bases
        else:
            valid_type = False
            for scalar_base in floating_scalar_bases:
                if not actual_base_type.startswith(scalar_base):
                    continue
                suffix = actual_base_type[len(scalar_base) :]
                valid_type = suffix in {"", "2", "3", "4"}
                break

        if not valid_type:
            raise DirectXRefusal(
                f"DirectX struct '{struct_name}' semantic '{semantic}' maps to "
                f"'{mapped_semantic}' and requires member '{member_name}' to "
                "have a floating-point scalar, vector, or scalar array type, "
                f"got {actual_type}"
            )
