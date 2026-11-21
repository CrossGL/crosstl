"""Call arguments and ray tracing in the SPIR-V generator: projected shadow references, integer vector constants, static overload scores, unsupported call initializers, value array arguments, lambda defaults, ray tracing result types and contexts, trace ray arguments, hit reports, and ray query intersections and state getters."""

import re
from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    ArrayLiteralNode,
    FunctionCallNode,
    LiteralNode,
    MemberAccessNode,
    RayTracingOpNode,
    UnaryOpNode,
)
from .spirv_atomic_load_store import SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS
from .spirv_literal_context import is_literal_operand
from .spirv_ray_locations import ray_location_variable
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvRayTracingMixin:
    """Ray tracing members of ``VulkanSPIRVCodeGen``."""

    def projected_shadow_reference(
        self, coord_id: SpirvId, depth_id: SpirvId, float_type: SpirvId
    ) -> SpirvId:
        """The Dref of a projected compare, divided by the coordinate's last component ``q``.

        CrossGL divides the reference by ``q`` as ``OpImageSampleProjDref*`` does (Vulkan, "Projection
        Operation"); the coordinate is projected by hand, so the reference is too.
        """
        component_type_name, source_count = self.vector_component_type_and_count(
            coord_id.type.base_type
        )
        component_type = self.register_primitive_type(component_type_name)
        projection = self.convert_scalar_to_type(
            self.composite_extract(coord_id, component_type, source_count - 1),
            float_type,
        )
        depth = self.convert_scalar_to_type(depth_id, float_type)
        return self.binary_operation("/", float_type, depth, projection)

    def call_projected_shadow_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        compare_args = self.shadow_compare_operands(
            function_name,
            args,
            self.projected_shadow_operand_counts(function_name),
        )
        if compare_args is None:
            return self.register_constant(0.0, self.register_primitive_type("float"))

        sampled_image_id, coord_id, depth_id, extra_args = compare_args
        metadata = self.resource_metadata_for_value(sampled_image_id)
        result_type = self.register_primitive_type("float")
        if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
            return self.default_value_for_type(result_type)

        if function_name in {
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
        }:
            if not self.validate_sampled_texture_lod_operand(
                function_name, extra_args, 0
            ):
                return self.default_value_for_type(result_type)
        if function_name in {
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }:
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[0], "dx"
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[1], "dy"
            ):
                return self.default_value_for_type(result_type)
        if "Offset" in function_name:
            if not self.validate_sampled_texture_offset_operand(
                function_name, metadata, extra_args[-1]
            ):
                return self.default_value_for_type(result_type)

        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        if projected_coord is None:
            return self.default_value_for_type(result_type)
        depth_id = self.projected_shadow_reference(coord_id, depth_id, result_type)

        image_operands = self.projected_shadow_image_operands(function_name, extra_args)
        id_value = self.get_id()
        opcode = (
            "OpImageSampleDrefExplicitLod"
            if image_operands and ("Lod" in image_operands or "Grad" in image_operands)
            else "OpImageSampleDrefImplicitLod"
        )
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{sampled_image_id.id} %{projected_coord.id} %{depth_id.id}"
            f"{(' ' + image_operands) if image_operands else ''}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def resource_offset_argument_indices(self, function_name: str):
        return {
            "textureOffset": {2, 3},
            "textureProjOffset": {2, 3},
            "textureProjLodOffset": {3, 4},
            "textureProjGradOffset": {4, 5},
            "textureLodOffset": {3, 4},
            "textureGradOffset": {4, 5},
            "textureGatherOffset": {2, 3},
            "textureGatherOffsets": {2, 3, 4, 5, 6},
            "texelFetchOffset": {3, 4},
            "textureCompareOffset": {3, 4},
            "textureCompareProjOffset": {3, 4},
            "textureCompareProjLodOffset": {4, 5},
            "textureCompareProjGradOffset": {5, 6},
            "textureCompareLodOffset": {4, 5},
            "textureCompareGradOffset": {5, 6},
            "textureGatherCompareOffset": {3, 4},
            "textureOffsetClamp": {2, 3},
            "textureGradOffsetClamp": {4, 5},
            "textureCompareOffsetClamp": {3, 4},
            "textureCompareGradOffsetClamp": {5, 6},
        }.get(function_name, set())

    def literal_integer_vector_constant(self, expr) -> Optional[SpirvId]:
        if not isinstance(expr, FunctionCallNode):
            return None

        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if hasattr(callee_expr, "name"):
            function_name = callee_expr.name
        elif isinstance(callee_expr, str):
            function_name = callee_expr
        else:
            return None

        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info is None:
            return None

        component_type_name, component_count = vector_info
        if component_type_name not in {"int", "uint"}:
            return None
        if len(expr.args) not in {1, component_count}:
            return None

        component_values = [
            self.literal_integer_value(arg, component_type_name) for arg in expr.args
        ]
        if any(value is None for value in component_values):
            return None
        # A single scalar operand is replicated into every component (GLSL 4.60, section 5.4.2).
        component_values *= component_count // len(component_values)

        component_type = self.register_primitive_type(component_type_name)
        vector_type = self.register_vector_type(component_type, component_count)
        components = [
            self.register_constant(value, component_type) for value in component_values
        ]
        return self.register_vector_constant(vector_type, components)

    def literal_integer_value(self, expr, component_type_name: str) -> Optional[int]:
        if isinstance(expr, UnaryOpNode):
            value = self.literal_integer_value(expr.operand, component_type_name)
            if value is None:
                return None
            if expr.op == "-":
                value = -value
            elif expr.op != "+":
                return None
            if component_type_name == "uint" and value < 0:
                return None
            return value

        if not isinstance(expr, LiteralNode):
            return None

        literal_type = self.normalize_primitive_name(
            self.convert_type_node_to_string(expr.literal_type)
        )
        if literal_type not in {"int", "uint"}:
            return None

        value = int(expr.value)
        if component_type_name == "uint" and value < 0:
            return None
        return value

    def expression_contains_empty_array_literal(self, expr) -> bool:
        if not isinstance(expr, ArrayLiteralNode):
            return False
        if not expr.elements:
            return True
        return any(
            self.expression_contains_empty_array_literal(element)
            for element in expr.elements
        )

    def static_function_argument_match_score(
        self, expected_type: SpirvId, arg
    ) -> Optional[int]:
        actual_type = self.infer_expression_result_type(arg)
        score = self.function_argument_type_match_score(expected_type, actual_type)
        if score is not None or actual_type is None:
            return score

        expected_pointee = self.pointer_type_pointee_type(expected_type)
        if expected_pointee is None:
            return None
        return self.function_argument_type_match_score(expected_pointee, actual_type)

    def contextual_call_parameter_types(
        self,
        function_name: str,
        call_args,
        skipped_arg_indices,
        mesh_output_arg_indices,
    ):
        emitted_arguments = [
            (arg_index, arg)
            for arg_index, arg in enumerate(call_args)
            if arg_index not in skipped_arg_indices
            and arg_index not in mesh_output_arg_indices
        ]
        struct_members = self.current_struct_members.get(function_name)
        if struct_members is not None and function_name in self.struct_types:
            contextual_types = {}
            for parameter_index, (arg_index, _arg) in enumerate(emitted_arguments):
                if parameter_index < len(struct_members):
                    contextual_types[arg_index] = struct_members[parameter_index][0]
            return contextual_types, set(), None, set()
        implicit_parameter_count = len(
            self.required_function_stage_object_argument_names(function_name)
        )
        scored_candidates = []
        for function_reference in self.function_reference_candidates(function_name):
            _function_id, (_return_type, param_types) = function_reference
            if len(param_types) != len(emitted_arguments) + implicit_parameter_count:
                continue

            score = 0
            compatible = True
            for parameter_index, (_arg_index, arg) in enumerate(emitted_arguments):
                argument_score = self.static_function_argument_match_score(
                    param_types[parameter_index], arg
                )
                if argument_score is None:
                    compatible = False
                    break
                score += argument_score
            if compatible:
                scored_candidates.append((score, function_reference))

        if not scored_candidates:
            return {}, set(), None, set()

        best_score = max(score for score, _candidate in scored_candidates)
        best_candidates = [
            candidate for score, candidate in scored_candidates if score == best_score
        ]
        if len(best_candidates) != 1:
            return (
                {},
                {arg_index for arg_index, _arg in emitted_arguments},
                None,
                set(),
            )

        selected_reference = best_candidates[0]
        function_id, (_return_type, selected_parameter_types) = selected_reference
        value_array_parameter_indices = (
            self.function_value_array_parameter_indices_by_id.get(function_id.id, set())
        )
        contextual_types = {}
        temporary_array_arguments = set()
        for parameter_index, (arg_index, _arg) in enumerate(emitted_arguments):
            contextual_types[arg_index] = selected_parameter_types[parameter_index]
            if parameter_index in value_array_parameter_indices:
                temporary_array_arguments.add(arg_index)
        return (
            contextual_types,
            set(),
            selected_reference,
            temporary_array_arguments,
        )

    def unsupported_call_initializer_type_error(
        self, function_name: str, arg_index: int, arg, ambiguous: bool
    ):
        source_location = getattr(arg, "source_location", None)
        reason = "ambiguous" if ambiguous else "unresolved"
        return UnsupportedSPIRVFeatureError(
            "contextual empty initializer",
            "SPIR-V cannot infer the destination type for an empty aggregate "
            f"initializer in {reason} call '{function_name}', argument "
            f"{arg_index + 1}; resolve the callee overload and preserve its "
            "concrete parameter type before code generation.",
            missing_capabilities=("spirv.empty_initializer_type_inference",),
            source_location=source_location,
        )

    def unsupported_call_initializer_binding_error(
        self, function_name: str, arg_index: int, arg
    ):
        source_location = getattr(arg, "source_location", None)
        return UnsupportedSPIRVFeatureError(
            "reference-bound empty initializer",
            "SPIR-V cannot bind an empty aggregate temporary to reference or "
            f"resource parameter {arg_index + 1} in call '{function_name}'; "
            "preserve an addressable source object before code generation.",
            missing_capabilities=("spirv.aggregate_initializer_reference_binding",),
            source_location=source_location,
        )

    def unsupported_reference_argument_error(
        self,
        function_name: str,
        arg_index: int,
        param_name: str,
        arg,
        *,
        feature: str,
        reason: str,
    ):
        return UnsupportedSPIRVFeatureError(
            feature,
            "SPIR-V pointer-preserving function inlining cannot bind "
            f"argument {arg_index + 1} in call '{function_name}' to reference "
            f"parameter '{param_name}': {reason}; preserve an addressable "
            "source object with the parameter's declared type.",
            missing_capabilities=("spirv.reference_parameter_aliasing",),
            source_location=getattr(arg, "source_location", None),
        )

    def process_call_argument(
        self,
        function_name,
        arg,
        arg_index,
        expected_type=None,
        allow_array_temporary=False,
    ):
        if arg_index in self.resource_offset_argument_indices(function_name):
            offset_constant = self.literal_integer_vector_constant(arg)
            if offset_constant is not None:
                return offset_constant

        if function_name in self.image_atomic_function_names() and arg_index == 0:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if (
            function_name in self.buffer_atomic_function_names()
            or function_name in SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS
        ) and arg_index == 0:
            pointer_arg = self.storage_reference_pointer(arg)
            if pointer_arg is not None:
                return pointer_arg

        if function_name in self.buffer_function_names() and arg_index == 0:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if self.function_parameter_requires_storage_image_pointer(
            function_name, arg_index
        ):
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        resource_array_params = self.resolve_function_resource_array_params(
            function_name
        )
        if arg_index in resource_array_params:
            pointer_arg = self.variable_pointer_from_expression(arg)
            if pointer_arg is not None:
                return pointer_arg

        if isinstance(arg, ArrayLiteralNode) and expected_type is not None:
            target_type = expected_type
            expected_pointee = self.pointer_type_pointee_type(expected_type)
            if expected_pointee is not None:
                if (
                    not allow_array_temporary
                    or expected_type.type.storage_class != "Function"
                    or self.array_type_info_from_type(expected_pointee) is None
                ):
                    raise self.unsupported_call_initializer_binding_error(
                        function_name, arg_index, arg
                    )
                value = self.process_array_literal(arg, expected_pointee)
                if value is None:
                    return None
                temporary = self.create_variable(expected_pointee, "Function")
                self.store_to_variable(temporary, value)
                return temporary

            return self.process_array_literal(arg, target_type)

        # A literal argument takes its parameter's type when it is known, never the enclosing expression's: in
        # uint(max(0, v)) the 0 stays int, so the maximum stays signed. Other arguments keep the context.
        previous_expected_type = self.current_expression_expected_type
        if is_literal_operand(arg):
            self.current_expression_expected_type = (
                self.type_name_string(expected_type)
                if expected_type is not None
                else None
            )
        try:
            return self.process_expression(arg)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def value_array_argument_object(self, value, expected_type):
        """Pass a by-value array held in a member or element through its own Function variable.

        Logical addressing accepts only memory object declarations as pointer operands of ``OpFunctionCall`` (SPIR-V
        1.6, section 2.16.1, "Universal Validation Rules"), so an ``OpAccessChain`` result cannot be the argument.
        The callee copies a by-value array parameter on entry, so a copy taken here keeps the same meaning.
        An array that is a value rather than a pointer (an ``OpConstantComposite`` such as a program-scope
        ``const`` array, a composite construct or a call result) is stored into a Function variable in the
        same way.
        """
        if (
            value is not None
            and expected_type is not None
            and value.type.storage_class is None
            and expected_type.type.storage_class == "Function"
        ):
            return self.spill_array_value_argument(value, expected_type)
        if (
            value is None
            or expected_type is None
            or value.id not in self.access_chain_ids
            or value.type.storage_class != "Function"
            or expected_type.type.storage_class != "Function"
        ):
            return value
        expected_pointee = self.pointer_type_pointee_type(expected_type)
        if (
            expected_pointee is None
            or self.array_type_info_from_type(expected_pointee) is None
        ):
            return value
        copied = self.copy_array_pointer_to_function_storage(value, expected_pointee)
        return value if copied is None else copied

    def spill_array_value_argument(self, value, expected_type):
        """Store an array value into a Function variable so it can be a by-value array argument."""
        expected_pointee = self.pointer_type_pointee_type(expected_type)
        if (
            expected_pointee is None
            or self.array_type_info_from_type(expected_pointee) is None
        ):
            return value
        converted = self.convert_value_to_type(value, expected_pointee)
        if not self.value_has_type(converted, expected_pointee):
            self.refuse_lowering(
                "SPIR-V cannot pass an array value of a different type "
                "to a by-value array parameter"
            )
            return value
        scratch_array = self.create_variable(expected_pointee, "Function")
        self.store_to_variable(scratch_array, converted)
        return scratch_array

    def contains_lambda_expression(self, expr) -> bool:
        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr
            else:
                callee_name = None
            if callee_name == "lambda":
                return True
            return any(self.contains_lambda_expression(arg) for arg in expr.args)

        for attr in (
            "left",
            "right",
            "condition",
            "true_expr",
            "false_expr",
            "operand",
            "object",
            "array",
            "index",
        ):
            child = getattr(expr, attr, None)
            if child is not None and self.contains_lambda_expression(child):
                return True

        elements = getattr(expr, "elements", None)
        if elements is not None:
            return any(self.contains_lambda_expression(element) for element in elements)

        return False

    def unsupported_lambda_default_value(
        self, context: str, result_type=None
    ) -> SpirvId:
        self.refuse_lowering(
            "SPIR-V backend does not support CrossGL "
            f"lambda expressions in {context}"
        )
        if result_type is None:
            result_type = self.register_primitive_type("float")
        return self.default_value_for_type(result_type)

    def represented_ir_diagnostic_default_value(
        self, category: str, operation: str
    ) -> SpirvId:
        result_type = self.represented_ir_diagnostic_result_type(category, operation)
        self.refuse_lowering(
            f"SPIR-V backend does not lower {category} operation {operation} yet"
        )
        return self.default_value_for_type(result_type)

    def represented_ir_diagnostic_result_type(
        self, category: str, operation: str
    ) -> SpirvId:
        if category == "ray tracing" and operation == "ReportHit":
            return self.register_primitive_type("bool")

        if category == "ray query":
            if operation == "Proceed":
                return self.register_primitive_type("bool")
            getter_info = self.ray_query_state_getter_info(operation)
            if getter_info is not None:
                _, result_kind = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            getter_info = self.ray_query_candidate_getter_info(operation)
            if getter_info is not None:
                _, result_kind = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            getter_info = self.ray_query_intersection_getter_info(operation)
            if getter_info is not None:
                _, result_kind, _ = getter_info
                return self.ray_query_result_type_for_kind(result_kind)
            return self.register_primitive_type("uint")

        return self.register_primitive_type("uint")

    def ray_tracing_operation_result_type(self, operation: str) -> SpirvId:
        if operation == "ReportHit":
            return self.register_primitive_type("bool")
        return self.register_primitive_type("uint")

    def ray_tracing_default_value(self, operation: str) -> SpirvId:
        return self.default_value_for_type(
            self.ray_tracing_operation_result_type(operation)
        )

    def ray_tracing_allowed_execution_models(self, operation: str):
        return {
            "TraceRay": {"RayGenerationKHR", "ClosestHitKHR", "MissKHR"},
            "CallShader": {
                "RayGenerationKHR",
                "ClosestHitKHR",
                "MissKHR",
                "CallableKHR",
            },
            "ReportHit": {"IntersectionKHR"},
            "IgnoreHit": {"AnyHitKHR"},
            "AcceptHitAndEndSearch": {"AnyHitKHR"},
        }.get(operation)

    def validate_ray_tracing_operation_context(self, operation: str) -> bool:
        allowed_models = self.ray_tracing_allowed_execution_models(operation)
        if allowed_models is None:
            return False
        if self.current_execution_model is None:
            self.refuse_lowering(
                f"SPIR-V ray tracing operation {operation} requires "
                "a ray tracing stage context"
            )
            return False
        if self.current_execution_model in allowed_models:
            return True

        allowed = ", ".join(sorted(allowed_models))
        self.refuse_lowering(
            f"SPIR-V ray tracing operation {operation} is only valid "
            f"in {allowed} stages"
        )
        return False

    def ray_tracing_acceleration_structure_value_from_expression(
        self, expr, operation: str
    ) -> Optional[SpirvId]:
        pointer = self.variable_pointer_from_expression(expr)
        if pointer is not None:
            if pointer.type.storage_class != "Function":
                self.mark_function_interface_variable(pointer)
            value = self.get_variable_value(pointer)
        else:
            value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V {operation} acceleration structure argument "
                "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        if not self.is_acceleration_structure_type_name(type_name):
            self.refuse_lowering(
                f"SPIR-V {operation} acceleration structure argument "
                f"must be accelerationStructureEXT, got {type_name}"
            )
            return None

        return value

    def ray_tracing_uint_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument could not be " "evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be a " "32-bit integer scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"int", "uint"}:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be a "
                f"32-bit integer scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def ray_tracing_float_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument could not be " "evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be a "
                "32-bit floating-point scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"float", "double", "int", "uint"}:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be a "
                f"32-bit floating-point scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("float"))

    def ray_tracing_vec3_operand(
        self, expr, operation: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument could not be " "evaluated"
            )
            return None

        float_type = self.register_primitive_type("float")
        vec3_type = self.register_vector_type(float_type, 3)
        value = self.convert_value_to_type(value, vec3_type)
        vector_info = self.vector_component_type_and_count(value.type.base_type)
        if vector_info != ("float", 3):
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be a "
                "32-bit floating-point vec3"
            )
            return None

        return value

    def ray_tracing_ray_desc_member_expression(
        self, ray_desc_expr, field_names, operation: str
    ) -> Optional[MemberAccessNode]:
        ray_desc_pointer = self.variable_pointer_from_expression(ray_desc_expr)
        if ray_desc_pointer is None:
            self.refuse_lowering(
                f"SPIR-V {operation} RayDesc argument must be an " "addressable value"
            )
            return None

        ray_desc_type = self.pointer_pointee_type(ray_desc_pointer)
        members = self.current_struct_members.get(
            ray_desc_type.type.base_type if ray_desc_type is not None else None, []
        )
        available_names = {member_name for _, member_name in members}
        for field_name in field_names:
            if field_name in available_names:
                return MemberAccessNode(ray_desc_expr, field_name)

        expected = "/".join(field_names)
        self.refuse_lowering(
            f"SPIR-V {operation} RayDesc argument does not provide " f"{expected}"
        )
        return None

    def trace_ray_argument_expressions(self, arguments):
        if len(arguments) == 11:
            return tuple(arguments)

        acceleration, ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index = (
            arguments[:6]
        )
        ray_desc = arguments[6]
        payload = arguments[7]
        origin = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("Origin", "origin", "rayOrigin", "RayOrigin"), "TraceRay"
        )
        tmin = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("TMin", "tMin", "Tmin", "tmin"), "TraceRay"
        )
        direction = self.ray_tracing_ray_desc_member_expression(
            ray_desc,
            ("Direction", "direction", "rayDirection", "RayDirection"),
            "TraceRay",
        )
        tmax = self.ray_tracing_ray_desc_member_expression(
            ray_desc, ("TMax", "tMax", "Tmax", "tmax"), "TraceRay"
        )
        if None in {origin, tmin, direction, tmax}:
            return None

        return (
            acceleration,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        )

    def ray_tracing_storage_variable_name(self, pointer: SpirvId, storage_class: str):
        base_name = pointer.name or f"value_{pointer.id}"
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", storage_class).lower()
        return f"{base_name}_{suffix}"

    def ray_tracing_storage_pointer(
        self, expr, storage_class: str, operation: str, role: str
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        pointer = self.variable_pointer_from_expression(expr)
        if pointer is None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument must be an " "addressable value"
            )
            return None, None

        value_type = self.pointer_pointee_type(pointer)
        if value_type is None:
            self.refuse_lowering(
                f"SPIR-V {operation} {role} argument has no value type"
            )
            return None, None

        if pointer.type.storage_class == storage_class:
            self.mark_function_interface_variable(pointer)
            return pointer, None

        key = (storage_class, pointer.id)
        storage_pointer = self.ray_tracing_storage_variables.get(key)
        if storage_pointer is None:
            storage_pointer = self.create_variable(
                value_type,
                storage_class,
                self.ray_tracing_storage_variable_name(pointer, storage_class),
            )
            self.ray_tracing_storage_variables[key] = storage_pointer
        self.mark_function_interface_variable(storage_pointer)

        value = self.get_variable_value(pointer)
        if value is not None:
            self.store_to_variable(
                storage_pointer, self.convert_value_to_type(value, value_type)
            )

        return storage_pointer, pointer

    def copy_ray_tracing_storage_back(
        self, storage_pointer: Optional[SpirvId], destination: Optional[SpirvId]
    ) -> None:
        if storage_pointer is None or destination is None:
            return

        destination_type = self.pointer_pointee_type(destination)
        value = self.get_variable_value(storage_pointer)
        if value is None:
            return
        if destination_type is not None:
            value = self.convert_value_to_type(value, destination_type)
        self.store_to_variable(destination, value)

    def process_trace_ray_operation(self, arguments) -> Optional[SpirvId]:
        expressions = self.trace_ray_argument_expressions(arguments)
        if expressions is None:
            return self.ray_tracing_default_value("TraceRay")

        (
            acceleration_expr,
            ray_flags_expr,
            cull_mask_expr,
            sbt_offset_expr,
            sbt_stride_expr,
            miss_index_expr,
            origin_expr,
            tmin_expr,
            direction_expr,
            tmax_expr,
            payload_expr,
        ) = expressions

        acceleration = self.ray_tracing_acceleration_structure_value_from_expression(
            acceleration_expr, "TraceRay"
        )
        ray_flags = self.ray_tracing_uint_operand(
            ray_flags_expr, "TraceRay", "ray flags"
        )
        cull_mask = self.ray_tracing_uint_operand(
            cull_mask_expr, "TraceRay", "cull mask"
        )
        sbt_offset = self.ray_tracing_uint_operand(
            sbt_offset_expr, "TraceRay", "shader binding table offset"
        )
        sbt_stride = self.ray_tracing_uint_operand(
            sbt_stride_expr, "TraceRay", "shader binding table stride"
        )
        miss_index = self.ray_tracing_uint_operand(
            miss_index_expr, "TraceRay", "miss shader index"
        )
        origin = self.ray_tracing_vec3_operand(origin_expr, "TraceRay", "origin")
        tmin = self.ray_tracing_float_operand(tmin_expr, "TraceRay", "Tmin")
        direction = self.ray_tracing_vec3_operand(
            direction_expr, "TraceRay", "direction"
        )
        tmax = self.ray_tracing_float_operand(tmax_expr, "TraceRay", "Tmax")
        payload_expr = (
            ray_location_variable(
                payload_expr, self.ray_storage_locations, "RayPayloadKHR"
            )
            or payload_expr
        )
        payload, payload_destination = self.ray_tracing_storage_pointer(
            payload_expr, "RayPayloadKHR", "TraceRay", "payload"
        )

        operands = {
            acceleration,
            ray_flags,
            cull_mask,
            sbt_offset,
            sbt_stride,
            miss_index,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        }
        if None in operands:
            return self.ray_tracing_default_value("TraceRay")

        self.emit(
            f"OpTraceRayKHR %{acceleration.id} %{ray_flags.id} %{cull_mask.id} "
            f"%{sbt_offset.id} %{sbt_stride.id} %{miss_index.id} %{origin.id} "
            f"%{tmin.id} %{direction.id} %{tmax.id} %{payload.id}"
        )
        self.copy_ray_tracing_storage_back(payload, payload_destination)
        return None

    def process_call_shader_operation(self, arguments) -> Optional[SpirvId]:
        shader_index = self.ray_tracing_uint_operand(
            arguments[0], "CallShader", "shader index"
        )
        callable_expr = (
            ray_location_variable(
                arguments[1], self.ray_storage_locations, "CallableDataKHR"
            )
            or arguments[1]
        )
        callable_data, callable_destination = self.ray_tracing_storage_pointer(
            callable_expr, "CallableDataKHR", "CallShader", "callable data"
        )
        if shader_index is None or callable_data is None:
            return self.ray_tracing_default_value("CallShader")

        self.emit(f"OpExecuteCallableKHR %{shader_index.id} %{callable_data.id}")
        self.copy_ray_tracing_storage_back(callable_data, callable_destination)
        return None

    def process_report_hit_operation(self, arguments) -> SpirvId:
        hit_t = self.ray_tracing_float_operand(arguments[0], "ReportHit", "hit T")
        hit_kind = self.ray_tracing_uint_operand(arguments[1], "ReportHit", "hit kind")
        if len(arguments) == 3:
            self.ray_tracing_storage_pointer(
                arguments[2], "HitAttributeKHR", "ReportHit", "hit attribute"
            )
        if hit_t is None or hit_kind is None:
            return self.ray_tracing_default_value("ReportHit")

        result_type = self.register_primitive_type("bool")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpReportIntersectionKHR %{result_type.id} "
            f"%{hit_t.id} %{hit_kind.id}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def process_ray_tracing_operation(
        self, expr: RayTracingOpNode
    ) -> Optional[SpirvId]:
        operation = expr.operation
        arguments = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        supported_argument_counts = {
            "TraceRay": {8, 11},
            "CallShader": {2},
            "ReportHit": {2, 3},
            "IgnoreHit": {0},
            "AcceptHitAndEndSearch": {0},
        }

        expected_counts = supported_argument_counts.get(operation)
        if expected_counts is None:
            return self.represented_ir_diagnostic_default_value(
                "ray tracing", operation
            )
        if len(arguments) not in expected_counts:
            expected = self.format_expected_argument_counts(expected_counts)
            self.refuse_lowering(
                f"SPIR-V ray tracing operation {operation} requires "
                f"{expected} arguments"
            )
            return self.ray_tracing_default_value(operation)
        if not self.validate_ray_tracing_operation_context(operation):
            return self.ray_tracing_default_value(operation)

        self.require_capability("RayTracingKHR")
        self.require_extension("SPV_KHR_ray_tracing")

        if operation == "TraceRay":
            return self.process_trace_ray_operation(arguments)
        if operation == "CallShader":
            return self.process_call_shader_operation(arguments)
        if operation == "ReportHit":
            return self.process_report_hit_operation(arguments)
        if operation == "IgnoreHit":
            self.emit("OpIgnoreIntersectionKHR")
            return None
        if operation == "AcceptHitAndEndSearch":
            self.emit("OpTerminateRayKHR")
            return None

        return self.represented_ir_diagnostic_default_value("ray tracing", operation)

    def ray_query_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        query_pointer = self.variable_pointer_from_expression(expr)
        if query_pointer is None:
            return None

        query_type = self.pointer_pointee_type(query_pointer)
        if query_type is None:
            return None

        if not self.is_ray_query_type_name(query_type.type.base_type):
            return None

        return query_pointer

    def ray_query_intersection_selector(self, operation: str) -> Optional[int]:
        if operation.startswith("Candidate"):
            return 0
        if operation.startswith("Committed"):
            return 1
        return None

    def ray_query_intersection_constant(self, operation: str) -> Optional[SpirvId]:
        selector = self.ray_query_intersection_selector(operation)
        if selector is None:
            return None

        return self.register_constant(selector, self.register_primitive_type("uint"))

    def ray_query_result_type_for_kind(self, result_kind: str) -> SpirvId:
        if result_kind == "bool":
            return self.register_primitive_type("bool")
        if result_kind == "float":
            return self.register_primitive_type("float")
        if result_kind == "vec2":
            return self.register_vector_type(self.register_primitive_type("float"), 2)
        if result_kind == "vec3":
            return self.register_vector_type(self.register_primitive_type("float"), 3)
        if result_kind == "vec3_array3":
            vec3_type = self.register_vector_type(
                self.register_primitive_type("float"), 3
            )
            return self.register_array_type(vec3_type, 3)
        if result_kind == "mat4x3":
            column_type = self.register_vector_type(
                self.register_primitive_type("float"), 3
            )
            return self.register_matrix_type(column_type, 4)
        return self.register_primitive_type("uint")

    def ray_query_state_getter_info(self, operation: str):
        getters = {
            "RayTMin": ("OpRayQueryGetRayTMinKHR", "float"),
            "RayFlags": ("OpRayQueryGetRayFlagsKHR", "uint"),
            "WorldRayOrigin": ("OpRayQueryGetWorldRayOriginKHR", "vec3"),
            "WorldRayDirection": ("OpRayQueryGetWorldRayDirectionKHR", "vec3"),
        }
        return getters.get(operation)

    def ray_query_state_getter_operations(self):
        return {
            "RayTMin",
            "RayFlags",
            "WorldRayOrigin",
            "WorldRayDirection",
        }

    def ray_query_candidate_getter_info(self, operation: str):
        getters = {
            # Negated after the read: DXR asks whether the candidate is
            # non-opaque, SPIR-V whether it is opaque.
            "CandidateProceduralPrimitiveNonOpaque": (
                "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR",
                "bool",
            ),
        }
        return getters.get(operation)

    def ray_query_candidate_getter_operations(self):
        return {
            "CandidateProceduralPrimitiveNonOpaque",
        }

    def ray_query_intersection_getter_info(self, operation: str):
        getters = {
            "CandidateType": ("OpRayQueryGetIntersectionTypeKHR", "uint"),
            "CommittedStatus": ("OpRayQueryGetIntersectionTypeKHR", "uint"),
            "CandidateTriangleRayT": ("OpRayQueryGetIntersectionTKHR", "float"),
            "CommittedRayT": ("OpRayQueryGetIntersectionTKHR", "float"),
            "CandidatePrimitiveIndex": (
                "OpRayQueryGetIntersectionPrimitiveIndexKHR",
                "uint",
            ),
            "CommittedPrimitiveIndex": (
                "OpRayQueryGetIntersectionPrimitiveIndexKHR",
                "uint",
            ),
            # HLSL semantics: InstanceID is the application-defined custom
            # index, InstanceIndex the instance's position (InstanceId).
            "CandidateInstanceID": (
                "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
                "uint",
            ),
            "CommittedInstanceID": (
                "OpRayQueryGetIntersectionInstanceCustomIndexKHR",
                "uint",
            ),
            "CandidateInstanceIndex": (
                "OpRayQueryGetIntersectionInstanceIdKHR",
                "uint",
            ),
            "CommittedInstanceIndex": (
                "OpRayQueryGetIntersectionInstanceIdKHR",
                "uint",
            ),
            "CandidateInstanceContributionToHitGroupIndex": (
                "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
                "uint",
            ),
            "CommittedInstanceContributionToHitGroupIndex": (
                "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR",
                "uint",
            ),
            "CandidateObjectRayOrigin": (
                "OpRayQueryGetIntersectionObjectRayOriginKHR",
                "vec3",
            ),
            "CommittedObjectRayOrigin": (
                "OpRayQueryGetIntersectionObjectRayOriginKHR",
                "vec3",
            ),
            "CandidateObjectRayDirection": (
                "OpRayQueryGetIntersectionObjectRayDirectionKHR",
                "vec3",
            ),
            "CommittedObjectRayDirection": (
                "OpRayQueryGetIntersectionObjectRayDirectionKHR",
                "vec3",
            ),
            "CandidateGeometryIndex": (
                "OpRayQueryGetIntersectionGeometryIndexKHR",
                "uint",
            ),
            "CommittedGeometryIndex": (
                "OpRayQueryGetIntersectionGeometryIndexKHR",
                "uint",
            ),
            "CandidateTriangleBarycentrics": (
                "OpRayQueryGetIntersectionBarycentricsKHR",
                "vec2",
            ),
            "CommittedTriangleBarycentrics": (
                "OpRayQueryGetIntersectionBarycentricsKHR",
                "vec2",
            ),
            "CandidateTriangleFrontFace": (
                "OpRayQueryGetIntersectionFrontFaceKHR",
                "bool",
            ),
            "CommittedTriangleFrontFace": (
                "OpRayQueryGetIntersectionFrontFaceKHR",
                "bool",
            ),
            "CandidateTriangleNormal": ("", "vec3"),
            "CommittedTriangleNormal": ("", "vec3"),
            "CandidateTriangleArea": ("", "float"),
            "CommittedTriangleArea": ("", "float"),
            "CandidateTriangleCentroid": ("", "vec3"),
            "CommittedTriangleCentroid": ("", "vec3"),
            "CandidateTriangleVertexPositions": (
                "OpRayQueryGetIntersectionTriangleVertexPositionsKHR",
                "vec3_array3",
            ),
            "CommittedTriangleVertexPositions": (
                "OpRayQueryGetIntersectionTriangleVertexPositionsKHR",
                "vec3_array3",
            ),
            "CandidateObjectToWorld": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CommittedObjectToWorld": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CandidateObjectToWorld3x4": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CommittedObjectToWorld3x4": (
                "OpRayQueryGetIntersectionObjectToWorldKHR",
                "mat4x3",
            ),
            "CandidateWorldToObject": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CommittedWorldToObject": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CandidateWorldToObject3x4": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
            "CommittedWorldToObject3x4": (
                "OpRayQueryGetIntersectionWorldToObjectKHR",
                "mat4x3",
            ),
        }
        getter = getters.get(operation)
        if getter is None:
            return None

        selector = self.ray_query_intersection_selector(operation)
        if selector is None:
            return None

        opcode, result_kind = getter
        return opcode, result_kind, selector

    def ray_query_intersection_getter_operations(self):
        return {
            "CandidateType",
            "CommittedStatus",
            "CandidateTriangleRayT",
            "CommittedRayT",
            "CandidatePrimitiveIndex",
            "CommittedPrimitiveIndex",
            "CandidateInstanceID",
            "CommittedInstanceID",
            "CandidateInstanceIndex",
            "CommittedInstanceIndex",
            "CandidateInstanceContributionToHitGroupIndex",
            "CommittedInstanceContributionToHitGroupIndex",
            "CandidateObjectRayOrigin",
            "CommittedObjectRayOrigin",
            "CandidateObjectRayDirection",
            "CommittedObjectRayDirection",
            "CandidateGeometryIndex",
            "CommittedGeometryIndex",
            "CandidateTriangleBarycentrics",
            "CommittedTriangleBarycentrics",
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
            "CandidateTriangleNormal",
            "CommittedTriangleNormal",
            "CandidateTriangleArea",
            "CommittedTriangleArea",
            "CandidateTriangleCentroid",
            "CommittedTriangleCentroid",
            "CandidateTriangleVertexPositions",
            "CommittedTriangleVertexPositions",
            "CandidateObjectToWorld",
            "CommittedObjectToWorld",
            "CandidateObjectToWorld3x4",
            "CommittedObjectToWorld3x4",
            "CandidateWorldToObject",
            "CommittedWorldToObject",
            "CandidateWorldToObject3x4",
            "CommittedWorldToObject3x4",
        }

    def process_ray_query_state_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_state_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        opcode, result_kind = getter_info
        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{query_pointer.id}")
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def process_ray_query_candidate_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_candidate_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        opcode, result_kind = getter_info
        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{query_pointer.id}")
        self.value_types[id_value] = result_type
        if operation == "CandidateProceduralPrimitiveNonOpaque":
            negated = self.get_id()
            self.emit(f"%{negated} = OpLogicalNot %{result_type.id} %{id_value}")
            self.value_types[negated] = result_type
            return SpirvId(negated, result_type.type)
        return SpirvId(id_value, result_type.type)

    def process_ray_query_intersection_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        getter_info = self.ray_query_intersection_getter_info(operation)
        if getter_info is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        if operation in {"CandidateTriangleNormal", "CommittedTriangleNormal"}:
            return self.process_ray_query_triangle_normal_getter(
                query_pointer, operation
            )
        if operation in {"CandidateTriangleArea", "CommittedTriangleArea"}:
            return self.process_ray_query_triangle_area_getter(query_pointer, operation)
        if operation in {"CandidateTriangleCentroid", "CommittedTriangleCentroid"}:
            return self.process_ray_query_triangle_centroid_getter(
                query_pointer, operation
            )

        opcode, result_kind, _ = getter_info
        if operation in {
            "CandidateTriangleVertexPositions",
            "CommittedTriangleVertexPositions",
        }:
            self.require_capability("RayQueryPositionFetchKHR")
            self.require_extension("SPV_KHR_ray_tracing_position_fetch")

        intersection = self.ray_query_intersection_constant(operation)
        if intersection is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        result_type = self.ray_query_result_type_for_kind(result_kind)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{query_pointer.id} %{intersection.id}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def emit_glsl_std450_instruction(
        self, instruction: str, result_type: SpirvId, args: List[SpirvId]
    ) -> SpirvId:
        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in args)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} "
            f"%{self.glsl_std450_id} {instruction} {arg_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def ray_query_triangle_vertices(
        self, query_pointer: SpirvId, operation: str
    ) -> Optional[Tuple[SpirvId, SpirvId, SpirvId, SpirvId]]:
        intersection = self.ray_query_intersection_constant(operation)
        if intersection is None:
            return None

        self.require_capability("RayQueryPositionFetchKHR")
        self.require_extension("SPV_KHR_ray_tracing_position_fetch")

        vec3_type = self.ray_query_result_type_for_kind("vec3")
        positions_type = self.ray_query_result_type_for_kind("vec3_array3")
        positions_id = self.get_id()
        self.emit(
            f"%{positions_id} = "
            f"OpRayQueryGetIntersectionTriangleVertexPositionsKHR "
            f"%{positions_type.id} %{query_pointer.id} %{intersection.id}"
        )
        self.value_types[positions_id] = positions_type
        positions = SpirvId(positions_id, positions_type.type)

        p0 = self.composite_extract(positions, vec3_type, 0)
        p1 = self.composite_extract(positions, vec3_type, 1)
        p2 = self.composite_extract(positions, vec3_type, 2)
        return vec3_type, p0, p1, p2

    def ray_query_triangle_edges(
        self, query_pointer: SpirvId, operation: str
    ) -> Optional[Tuple[SpirvId, SpirvId, SpirvId]]:
        vertices = self.ray_query_triangle_vertices(query_pointer, operation)
        if vertices is None:
            return None

        vec3_type, p0, p1, p2 = vertices
        edge0 = self.binary_operation("-", vec3_type, p1, p0)
        edge1 = self.binary_operation("-", vec3_type, p2, p0)
        return vec3_type, edge0, edge1

    def process_ray_query_triangle_normal_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        edges = self.ray_query_triangle_edges(query_pointer, operation)
        if edges is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, edge0, edge1 = edges
        cross = self.emit_glsl_std450_instruction("Cross", vec3_type, [edge0, edge1])
        return self.emit_glsl_std450_instruction("Normalize", vec3_type, [cross])

    def process_ray_query_triangle_area_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        edges = self.ray_query_triangle_edges(query_pointer, operation)
        if edges is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, edge0, edge1 = edges
        float_type = self.ray_query_result_type_for_kind("float")
        cross = self.emit_glsl_std450_instruction("Cross", vec3_type, [edge0, edge1])
        double_area = self.emit_glsl_std450_instruction("Length", float_type, [cross])
        half = self.register_constant(0.5, float_type)
        return self.binary_operation("*", float_type, double_area, half)

    def process_ray_query_triangle_centroid_getter(
        self, query_pointer: SpirvId, operation: str
    ) -> SpirvId:
        vertices = self.ray_query_triangle_vertices(query_pointer, operation)
        if vertices is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        vec3_type, p0, p1, p2 = vertices
        float_type = self.ray_query_result_type_for_kind("float")
        p0_plus_p1 = self.binary_operation("+", vec3_type, p0, p1)
        total = self.binary_operation("+", vec3_type, p0_plus_p1, p2)
        three = self.register_constant(3.0, float_type)
        return self.binary_operation("/", vec3_type, total, three)

    def format_expected_argument_counts(self, expected_counts) -> str:
        return " or ".join(str(count) for count in sorted(expected_counts))

    def registered_value_type(self, value_id: SpirvId) -> Optional[SpirvId]:
        return self.value_types.get(value_id.id) or self.find_registered_type_by_base(
            value_id.type.base_type
        )

    def acceleration_structure_value_from_expression(self, expr) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                "SPIR-V RayQuery.TraceRayInline acceleration structure "
                "argument could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        if not self.is_acceleration_structure_type_name(type_name):
            self.refuse_lowering(
                "SPIR-V RayQuery.TraceRayInline acceleration structure "
                f"argument must be accelerationStructureEXT, got {type_name}"
            )
            return None

        return value
