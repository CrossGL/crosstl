"""Ray query initialization and mesh outputs in the SPIR-V generator: ray query operands, initialization and calls, mesh operations and counts, geometry stream calls, mesh primitives and limits, and mesh output members, locations, primitive indices, positions, assignments and shadow variables."""

from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
    MeshOpNode,
    RayQueryOpNode,
)
from ..ray_query_vocabulary import (
    ray_flag_value,
    ray_query_type_flags,
)
from .spirv_exact_lookups import (
    spirv_mesh_output_builtin,
    spirv_mesh_output_refusal,
)
from .spirv_geometry_streams import STREAM_FUNCTIONS
from .spirv_shading_rate import require_shading_rate_capability
from .spirv_values import SpirvId
from .stage_io_semantic_locations import semantic_location


class SpirvMeshOutputMixin:
    """Mesh output members of ``VulkanSPIRVCodeGen``."""

    def ray_query_uint_operand(self, expr, role: str) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V RayQuery.TraceRayInline {role} argument "
                "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.refuse_lowering(
                f"SPIR-V RayQuery.TraceRayInline {role} argument "
                "must be a 32-bit integer scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"int", "uint"}:
            self.refuse_lowering(
                f"SPIR-V RayQuery.TraceRayInline {role} argument "
                f"must be a 32-bit integer scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def ray_query_float_operand(
        self, expr, role: str, operation: str = "TraceRayInline"
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V RayQuery.{operation} {role} argument " "could not be evaluated"
            )
            return None

        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            self.refuse_lowering(
                f"SPIR-V RayQuery.{operation} {role} argument "
                "must be a 32-bit floating-point scalar"
            )
            return None

        component_type = self.normalize_primitive_name(value_type.type.base_type)
        if component_type not in {"float", "double", "int", "uint"}:
            self.refuse_lowering(
                f"SPIR-V RayQuery.{operation} {role} argument "
                f"must be a 32-bit floating-point scalar, got {component_type}"
            )
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("float"))

    def ray_query_vec3_operand(self, expr, role: str) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"SPIR-V RayQuery.TraceRayInline {role} argument "
                "could not be evaluated"
            )
            return None

        float_type = self.register_primitive_type("float")
        vec3_type = self.register_vector_type(float_type, 3)
        value = self.convert_value_to_type(value, vec3_type)
        vector_info = self.vector_component_type_and_count(value.type.base_type)
        if vector_info != ("float", 3):
            self.refuse_lowering(
                f"SPIR-V RayQuery.TraceRayInline {role} argument "
                "must be a 32-bit floating-point vec3"
            )
            return None

        return value

    def ray_desc_member_expression(
        self, ray_desc_expr, field_names
    ) -> Optional[MemberAccessNode]:
        ray_desc_pointer = self.variable_pointer_from_expression(ray_desc_expr)
        if ray_desc_pointer is None:
            self.refuse_lowering(
                "SPIR-V RayQuery.TraceRayInline RayDesc argument "
                "must be an addressable value"
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
            "SPIR-V RayQuery.TraceRayInline RayDesc argument "
            f"does not provide {expected}"
        )
        return None

    def ray_query_initialize_argument_expressions(self, arguments):
        if len(arguments) == 7:
            return tuple(arguments)

        acceleration, ray_flags, cull_mask, ray_desc = arguments
        origin = self.ray_desc_member_expression(
            ray_desc, ("Origin", "origin", "rayOrigin", "RayOrigin")
        )
        tmin = self.ray_desc_member_expression(
            ray_desc, ("TMin", "tMin", "Tmin", "tmin")
        )
        direction = self.ray_desc_member_expression(
            ray_desc, ("Direction", "direction", "rayDirection", "RayDirection")
        )
        tmax = self.ray_desc_member_expression(
            ray_desc, ("TMax", "tMax", "Tmax", "tmax")
        )
        if None in {origin, tmin, direction, tmax}:
            return None
        return acceleration, ray_flags, cull_mask, origin, tmin, direction, tmax

    def ray_query_flags_with_type_flags(
        self, query_pointer: SpirvId, ray_flags: SpirvId
    ) -> Optional[SpirvId]:
        """*ray_flags* combined with the flags of the query's RayQuery<FLAGS> type."""
        query_type = self.pointer_pointee_type(query_pointer)
        type_name = getattr(getattr(query_type, "type", None), "base_type", "")
        type_flags = ray_query_type_flags(type_name)
        if type_flags is None:
            self.refuse_lowering(
                f"SPIR-V RayQuery type '{type_name}' has template flags that are not "
                "ray flags"
            )
            return None
        type_value = ray_flag_value(type_flags)
        if not type_value:
            return ray_flags
        uint_type = self.register_primitive_type("uint")
        operand_value = self.constant_values_by_id.get(ray_flags.id)
        if isinstance(operand_value, int) and not isinstance(operand_value, bool):
            return self.register_constant(operand_value | type_value, uint_type)
        type_constant = self.register_constant(type_value, uint_type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpBitwiseOr %{uint_type.id} %{ray_flags.id} "
            f"%{type_constant.id}"
        )
        self.value_types[id_value] = uint_type
        return SpirvId(id_value, uint_type.type)

    def process_ray_query_initialize(self, query_pointer: SpirvId, arguments) -> None:
        expressions = self.ray_query_initialize_argument_expressions(arguments)
        if expressions is None:
            return None

        (
            acceleration_expr,
            ray_flags_expr,
            cull_mask_expr,
            origin_expr,
            tmin_expr,
            direction_expr,
            tmax_expr,
        ) = expressions

        acceleration = self.acceleration_structure_value_from_expression(
            acceleration_expr
        )
        ray_flags = self.ray_query_uint_operand(ray_flags_expr, "ray flags")
        cull_mask = self.ray_query_uint_operand(cull_mask_expr, "cull mask")
        origin = self.ray_query_vec3_operand(origin_expr, "origin")
        tmin = self.ray_query_float_operand(tmin_expr, "Tmin")
        direction = self.ray_query_vec3_operand(direction_expr, "direction")
        tmax = self.ray_query_float_operand(tmax_expr, "Tmax")

        if None in {acceleration, ray_flags, cull_mask, origin, tmin, direction, tmax}:
            return None
        ray_flags = self.ray_query_flags_with_type_flags(query_pointer, ray_flags)
        if ray_flags is None:
            return None

        self.emit(
            f"OpRayQueryInitializeKHR %{query_pointer.id} %{acceleration.id} "
            f"%{ray_flags.id} %{cull_mask.id} %{origin.id} %{tmin.id} "
            f"%{direction.id} %{tmax.id}"
        )
        return None

    def ray_query_method_names(self):
        return (
            {
                "Abort",
                "CommitNonOpaqueTriangleHit",
                "CommitProceduralPrimitiveHit",
                "Proceed",
                "TraceRayInline",
            }
            | self.ray_query_state_getter_operations()
            | self.ray_query_candidate_getter_operations()
            | (self.ray_query_intersection_getter_operations())
        )

    def ray_query_call_from_function_call(self, expr) -> Optional[RayQueryOpNode]:
        if not isinstance(expr, FunctionCallNode):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(func_expr, MemberAccessNode):
            return None

        operation = str(getattr(func_expr, "member", ""))
        if operation not in self.ray_query_method_names():
            return None

        return RayQueryOpNode(
            operation,
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None)),
            getattr(expr, "arguments", getattr(expr, "args", [])),
        )

    def process_ray_query_operation(self, expr: RayQueryOpNode) -> SpirvId:
        operation = expr.operation
        arguments = getattr(expr, "args", getattr(expr, "arguments", [])) or []

        supported_argument_counts = {
            "Proceed": 0,
            "Abort": 0,
            "CommitNonOpaqueTriangleHit": 0,
            "CommitProceduralPrimitiveHit": 1,
            "TraceRayInline": {4, 7},
        }
        for getter_operation in self.ray_query_state_getter_operations():
            supported_argument_counts[getter_operation] = 0
        for getter_operation in self.ray_query_candidate_getter_operations():
            supported_argument_counts[getter_operation] = 0
        for getter_operation in self.ray_query_intersection_getter_operations():
            supported_argument_counts[getter_operation] = 0
        if operation not in supported_argument_counts:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        expected_counts = supported_argument_counts[operation]
        if isinstance(expected_counts, int):
            expected_counts = {expected_counts}
        if len(arguments) not in expected_counts:
            self.refuse_lowering(
                f"SPIR-V RayQuery.{operation} requires "
                f"{self.format_expected_argument_counts(expected_counts)} arguments"
            )
            if operation == "TraceRayInline":
                return None
            return self.default_value_for_type(
                self.represented_ir_diagnostic_result_type("ray query", operation)
            )

        query_pointer = self.ray_query_pointer_from_expression(expr.query_expr)
        if query_pointer is None:
            return self.represented_ir_diagnostic_default_value("ray query", operation)

        self.require_capability("RayQueryKHR")
        self.require_extension("SPV_KHR_ray_query")

        if operation == "TraceRayInline":
            return self.process_ray_query_initialize(query_pointer, arguments)

        if operation == "Abort":
            self.emit(f"OpRayQueryTerminateKHR %{query_pointer.id}")
            return None

        if operation == "CommitNonOpaqueTriangleHit":
            self.emit(f"OpRayQueryConfirmIntersectionKHR %{query_pointer.id}")
            return None

        if operation == "CommitProceduralPrimitiveHit":
            hit_t = self.ray_query_float_operand(
                arguments[0], "hit distance", operation=operation
            )
            if hit_t is None:
                return None
            self.emit(
                f"OpRayQueryGenerateIntersectionKHR %{query_pointer.id} %{hit_t.id}"
            )
            return None

        if operation == "Proceed":
            result_type = self.register_primitive_type("bool")
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpRayQueryProceedKHR %{result_type.id} "
                f"%{query_pointer.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if operation in self.ray_query_state_getter_operations():
            return self.process_ray_query_state_getter(query_pointer, operation)

        if operation in self.ray_query_candidate_getter_operations():
            return self.process_ray_query_candidate_getter(query_pointer, operation)

        if operation in self.ray_query_intersection_getter_operations():
            return self.process_ray_query_intersection_getter(query_pointer, operation)

        return self.represented_ir_diagnostic_default_value("ray query", operation)

    def process_mesh_operation(self, expr: MeshOpNode) -> Optional[SpirvId]:
        """Process represented mesh/task shader intrinsics."""
        operation = expr.operation
        if operation == "DispatchMesh":
            return self.process_dispatch_mesh_operation(expr)
        if operation != "SetMeshOutputCounts":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        if self.current_execution_model != "MeshEXT":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        arguments = getattr(expr, "arguments", []) or []
        if len(arguments) != 2:
            self.refuse_lowering(
                "SPIR-V mesh SetMeshOutputCounts requires exactly " "2 arguments"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        uint_type = self.register_primitive_type("uint")
        vertex_count = self.process_expression(arguments[0])
        primitive_count = self.process_expression(arguments[1])
        if vertex_count is None or primitive_count is None:
            self.refuse_lowering(
                "SPIR-V mesh SetMeshOutputCounts requires count operands"
            )
            return self.register_constant(0, uint_type)
        if not self.validate_mesh_count_operands(
            "SetMeshOutputCounts",
            [vertex_count, primitive_count],
            argument_exprs=arguments,
        ):
            return self.register_constant(0, uint_type)
        if not self.validate_mesh_set_output_count_limits(arguments):
            return self.register_constant(0, uint_type)

        vertex_count = self.convert_value_to_type(vertex_count, uint_type)
        primitive_count = self.convert_value_to_type(primitive_count, uint_type)

        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        self.emit(f"OpSetMeshOutputsEXT %{vertex_count.id} %{primitive_count.id}")

        if self.current_function_id is not None:
            previous_vertices, previous_primitives = (
                self.mesh_output_counts_by_function.get(
                    self.current_function_id, (None, None)
                )
            )
            observed_vertices = self.literal_int_argument(arguments[0])
            observed_primitives = self.literal_int_argument(arguments[1])
            self.mesh_output_counts_by_function[self.current_function_id] = (
                self.max_optional_int(previous_vertices, observed_vertices),
                self.max_optional_int(previous_primitives, observed_primitives),
            )

        return self.register_constant(0, uint_type)

    def validate_mesh_set_output_count_limits(self, arguments: List) -> bool:
        """Reject literal mesh output counts above declared stage limits."""
        vertex_limit = self.mesh_stage_limit(self.current_stage, "max_vertices")
        primitive_limit = self.mesh_stage_limit(self.current_stage, "max_primitives")
        requested_vertices = self.literal_int_argument(arguments[0])
        requested_primitives = self.literal_int_argument(arguments[1])
        valid = True

        if (
            vertex_limit is not None
            and requested_vertices is not None
            and requested_vertices > vertex_limit
        ):
            self.refuse_lowering(
                "SPIR-V mesh SetMeshOutputCounts vertex count "
                "exceeds declared max_vertices"
            )
            valid = False

        if (
            primitive_limit is not None
            and requested_primitives is not None
            and requested_primitives > primitive_limit
        ):
            self.refuse_lowering(
                "SPIR-V mesh SetMeshOutputCounts primitive count "
                "exceeds declared max_primitives"
            )
            valid = False

        return valid

    def validate_mesh_count_operands(
        self,
        operation: str,
        operands: List[SpirvId],
        count_label: str = "count",
        argument_exprs: Optional[List] = None,
    ) -> bool:
        """Require SPIR-V mesh/task count operands to be scalar integer values."""
        if all(
            self.integer_value_component_count(operand) == 1 for operand in operands
        ):
            if argument_exprs is not None and any(
                (literal := self.literal_int_argument(argument)) is not None
                and literal < 0
                for argument in argument_exprs
            ):
                self.refuse_lowering(
                    f"SPIR-V mesh {operation} {count_label} operands "
                    "must be non-negative integer values"
                )
                return False
            return True

        self.refuse_lowering(
            f"SPIR-V mesh {operation} {count_label} operands must be "
            "scalar integer values"
        )
        return False

    def process_mesh_output_function_call(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        """Lower represented mesh output helper calls to SPIR-V Output stores."""
        uint_type = self.register_primitive_type("uint")
        if self.current_execution_model != "MeshEXT":
            self.refuse_lowering(
                f"SPIR-V mesh {function_name} is only valid in mesh stages"
            )
            return self.register_constant(0, uint_type)

        if len(args) != 2:
            self.refuse_lowering(
                f"SPIR-V mesh {function_name} requires exactly 2 arguments"
            )
            return self.register_constant(0, uint_type)

        if function_name == "SetVertex":
            self.process_mesh_set_vertex(args)
        elif function_name == "SetPrimitive":
            self.process_mesh_set_primitive(args)

        return self.register_constant(0, uint_type)

    def process_geometry_stream_function_call(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        """Lower geometry stream-control calls to SPIR-V instructions."""
        if self.current_execution_model != "Geometry":
            self.refuse_lowering(
                f"SPIR-V geometry {function_name} is only valid in " "geometry stages"
            )
            return None

        if function_name in STREAM_FUNCTIONS:
            # The stream operand must be a constant instruction (SPIR-V OpEmitStreamVertex).
            stream = self.literal_int_argument(args[0]) if len(args) == 1 else None
            if stream is None or stream < 0:
                self.refuse_lowering(
                    f"SPIR-V geometry {function_name} requires one constant, "
                    "non-negative stream number"
                )
                return None
            self.require_capability("GeometryStreams")
            self.geometry_streams_used = True
            stream_id = self.register_constant(
                stream, self.register_primitive_type("int")
            )
            self.emit(f"{STREAM_FUNCTIONS[function_name]} %{stream_id.id}")
            return None

        if args:
            self.refuse_lowering(
                f"SPIR-V geometry {function_name} requires no arguments"
            )
            return None

        opcode = {
            "EmitVertex": "OpEmitVertex",
            "EndPrimitive": "OpEndPrimitive",
        }[function_name]
        self.emit(opcode)
        return None

    def process_mesh_set_vertex(self, args: List):
        """Lower SetVertex(index, value) to a Position builtin output store."""
        index = self.process_expression(args[0])
        if index is None:
            self.refuse_lowering("SPIR-V mesh SetVertex requires an index operand")
            return

        position_value = self.mesh_vertex_position_value(args[1])
        if position_value is None:
            return

        vertex_output = self.ensure_mesh_vertex_position_output(
            self.literal_int_argument(args[0])
        )
        if vertex_output is None:
            return

        int_type = self.register_primitive_type("int")
        member_index = self.register_constant(0, int_type)
        vec4_type = self.register_vector_type(self.register_primitive_type("float"), 4)
        ptr_type = self.register_pointer_type(vec4_type, "Output")
        access = self.access_chain(vertex_output, [index, member_index], ptr_type)
        self.variable_value_types[access.id] = vec4_type
        self.store_to_variable(access, position_value)

    def process_mesh_set_primitive(self, args: List):
        """Lower SetPrimitive(index, value) to primitive-index builtin output."""
        index = self.process_expression(args[0])
        if index is None:
            self.refuse_lowering("SPIR-V mesh SetPrimitive requires an index operand")
            return

        primitive_output, element_type = self.ensure_mesh_primitive_index_output(
            self.literal_int_argument(args[0])
        )
        if primitive_output is None or element_type is None:
            return

        value = self.process_expression(args[1])
        if value is None:
            self.refuse_lowering("SPIR-V mesh SetPrimitive requires a value operand")
            return

        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.refuse_lowering(
                "SPIR-V mesh SetPrimitive value does not match "
                f"{element_type.type.base_type}"
            )
            return

        ptr_type = self.register_pointer_type(element_type, "Output")
        access = self.access_chain(primitive_output, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.store_to_variable(access, value)

    def mesh_stage_current_output_limits(self) -> Tuple[int, int]:
        """Return current mesh output limits from observed counts and metadata."""
        observed_vertices, observed_primitives = (None, None)
        if self.current_function_id is not None:
            observed_vertices, observed_primitives = (
                self.mesh_output_counts_by_function.get(
                    self.current_function_id, (None, None)
                )
            )

        return self.mesh_stage_limits_with_observed_counts(
            self.current_stage, observed_vertices, observed_primitives
        )

    def mesh_stage_limits_with_observed_counts(
        self, stage, observed_vertices, observed_primitives
    ) -> Tuple[int, int]:
        """Mesh output limits from metadata, then output array sizes, raised to constant observed counts."""
        max_vertices = self.mesh_stage_limit(stage, "max_vertices")
        max_primitives = self.mesh_stage_limit(stage, "max_primitives")
        declared = self.mesh_output_array_limits(stage)
        if max_vertices is None:
            max_vertices = declared.get("vertices")
        if max_primitives is None:
            max_primitives = declared.get("primitives")
        if observed_vertices is not None:
            max_vertices = max(observed_vertices, max_vertices or 0)
        if observed_primitives is not None:
            max_primitives = max(observed_primitives, max_primitives or 0)
        return max(1, max_vertices or 1), max(1, max_primitives or 1)

    def mesh_output_literal_index_limit(
        self, role: str, info: Optional[dict] = None
    ) -> Optional[int]:
        if info is not None and info.get("count") is not None:
            return int(info["count"])

        max_vertices, max_primitives = self.mesh_stage_current_output_limits()
        if role == "vertices":
            return max_vertices
        if role in {"indices", "primitives"}:
            return max_primitives
        return None

    def validate_mesh_output_literal_index(
        self,
        role: str,
        literal_index: Optional[int],
        diagnostic_name: str,
        info: Optional[dict] = None,
    ) -> bool:
        if literal_index is None:
            return True
        if literal_index < 0:
            self.refuse_lowering(
                f"SPIR-V mesh {diagnostic_name} literal index " "must be non-negative"
            )
            return False

        limit = self.mesh_output_literal_index_limit(role, info)
        if limit is None or literal_index < limit:
            return True

        limit_label = "vertex" if role == "vertices" else "primitive"
        self.refuse_lowering(
            f"SPIR-V mesh {diagnostic_name} literal index exceeds "
            f"the declared mesh {limit_label} output limit"
        )
        return False

    def ensure_mesh_vertex_position_output(
        self, literal_index: Optional[int] = None
    ) -> Optional[SpirvId]:
        """Create the mesh Position output array used by SetVertex."""
        if not self.validate_mesh_output_literal_index(
            "vertices", literal_index, "SetVertex"
        ):
            return None

        minimum_size = (literal_index + 1) if literal_index is not None else 1
        max_vertices, _ = self.mesh_stage_current_output_limits()
        max_vertices = max(max_vertices, minimum_size)

        if self.mesh_vertex_output_variable is not None:
            if (
                self.mesh_vertex_output_limit is not None
                and minimum_size > self.mesh_vertex_output_limit
            ):
                self.refuse_lowering(
                    "SPIR-V mesh SetVertex literal index exceeds the "
                    "declared mesh vertex output limit"
                )
                return None
            self.mark_function_interface_variable(self.mesh_vertex_output_variable)
            return self.mesh_vertex_output_variable

        float_type = self.register_primitive_type("float")
        vec4_type = self.register_vector_type(float_type, 4)
        vertex_type = self.register_struct_type(
            "_CrossGLMeshVertexOutputEXT", [(vec4_type, "position")]
        )
        self.decorations.append(
            f"OpMemberDecorate %{vertex_type.id} 0 BuiltIn Position"
        )
        array_type = self.register_array_type(vertex_type, max_vertices)
        variable = self.create_variable(array_type, "Output", "_CrossGLMeshVerticesEXT")
        self.outputs.append(variable)
        self.mesh_vertex_output_variable = variable
        self.mesh_vertex_output_limit = max_vertices
        self.mark_function_interface_variable(variable)
        return variable

    def ensure_mesh_member_output_variable(
        self,
        info: dict,
        member_name: str,
        member_type: SpirvId,
        semantic: Optional[str],
        literal_index: Optional[int] = None,
    ) -> Optional[SpirvId]:
        """Create a mesh output array for a signature member assignment."""
        role = info["role"]
        if not self.validate_mesh_output_literal_index(
            role, literal_index, f"{info['name']} output", info
        ):
            return None

        if role == "vertices":
            max_vertices, _ = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_vertices, max_vertices)
        else:
            _, max_primitives = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_primitives, max_primitives)

        if literal_index is not None:
            element_count = max(element_count, literal_index + 1)

        builtin = self.mesh_output_member_builtin(role, semantic)
        refusal = spirv_mesh_output_refusal(role, semantic)
        if refusal is not None:
            self.refuse_lowering(f"{refusal} (member '{member_name}')")
        location = (
            None
            if builtin
            else self.mesh_output_member_location(semantic, role, member_name)
        )
        key = (
            role,
            member_name,
            member_type.id,
            builtin,
            location,
            element_count,
        )
        if key in self.mesh_output_member_variables:
            self.mark_function_interface_variable(
                self.mesh_output_member_variables[key]
            )
            return self.mesh_output_member_variables[key]

        array_type = self.register_array_type(member_type, element_count)
        variable_name = f"_CrossGLMesh_{role}_{member_name}"
        if not builtin:
            self.validate_user_defined_interface_type(
                array_type, "Output", variable_name
            )
        variable = self.create_variable(array_type, "Output", variable_name)
        if builtin:
            self.decorations.append(f"OpDecorate %{variable.id} BuiltIn {builtin}")
            require_shading_rate_capability(self, builtin)
        else:
            self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        if role == "primitives":
            self.decorations.append(f"OpDecorate %{variable.id} PerPrimitiveEXT")
        self.outputs.append(variable)
        self.mesh_output_member_variables[key] = variable
        self.mark_function_interface_variable(variable)
        return variable

    def mesh_output_member_builtin(
        self, role: str, semantic: Optional[str]
    ) -> Optional[str]:
        """Return a SPIR-V BuiltIn decoration for mesh output member semantics."""
        return spirv_mesh_output_builtin(role, semantic)

    def mesh_output_member_location(
        self, semantic: Optional[str], role: str, member_name: str
    ) -> int:
        """Return or allocate a Location decoration for user mesh outputs."""
        key = (role, member_name, str(semantic or ""))
        if key in self.mesh_output_member_locations:
            return self.mesh_output_member_locations[key]

        location = self.mesh_output_semantic_location(semantic)
        if location is None:
            location = self.next_output_location
            self.next_output_location += 1

        self.mesh_output_member_locations[key] = location
        return location

    def mesh_output_semantic_location(self, semantic: Optional[str]) -> Optional[int]:
        return semantic_location(semantic)

    def ensure_mesh_primitive_index_output(
        self, literal_index: Optional[int] = None
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        """Create the topology-specific primitive-index output array."""
        if not self.validate_mesh_output_literal_index(
            "indices", literal_index, "SetPrimitive"
        ):
            return None, None

        info = self.mesh_primitive_index_builtin_info()
        if info is None:
            self.refuse_lowering(
                "SPIR-V mesh SetPrimitive requires point, line, or "
                "triangle output topology"
            )
            return None, None

        builtin_name, element_type = info
        minimum_size = (literal_index + 1) if literal_index is not None else 1
        _, max_primitives = self.mesh_stage_current_output_limits()
        max_primitives = max(max_primitives, minimum_size)

        cached = self.mesh_primitive_index_outputs.get(builtin_name)
        if cached is not None:
            variable, cached_type, cached_limit = cached
            if minimum_size > cached_limit:
                self.refuse_lowering(
                    "SPIR-V mesh SetPrimitive literal index exceeds the "
                    "declared mesh primitive output limit"
                )
                return None, None
            self.mark_function_interface_variable(variable)
            return variable, cached_type

        array_type = self.register_array_type(element_type, max_primitives)
        variable = self.create_variable(
            array_type, "Output", f"_CrossGLMesh{builtin_name}"
        )
        self.decorations.append(f"OpDecorate %{variable.id} BuiltIn {builtin_name}")
        self.outputs.append(variable)
        self.mesh_primitive_index_outputs[builtin_name] = (
            variable,
            element_type,
            max_primitives,
        )
        self.mark_function_interface_variable(variable)
        return variable, element_type

    def mesh_primitive_index_builtin_info(
        self,
    ) -> Optional[Tuple[str, SpirvId]]:
        """Return primitive-index builtin metadata for the current topology."""
        mode = self.mesh_stage_topology_mode(self.current_stage)
        uint_type = self.register_primitive_type("uint")
        if mode == "OutputPoints":
            return "PrimitivePointIndicesEXT", uint_type
        if mode == "OutputLinesEXT":
            return (
                "PrimitiveLineIndicesEXT",
                self.register_vector_type(uint_type, 2),
            )
        if mode == "OutputTrianglesEXT":
            return (
                "PrimitiveTriangleIndicesEXT",
                self.register_vector_type(uint_type, 3),
            )
        return None

    def mesh_vertex_position_value(self, expr) -> Optional[SpirvId]:
        """Return a vec4 position value for SetVertex."""
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering("SPIR-V mesh SetVertex requires a value operand")
            return None

        position = self.mesh_vertex_position_from_value(value)
        if position is None:
            self.refuse_lowering(
                "SPIR-V mesh SetVertex value must be a vec3, vec4, "
                "or struct with a position member"
            )
        return position

    def mesh_vertex_position_from_value(self, value: SpirvId) -> Optional[SpirvId]:
        """Convert a SetVertex value to the Position builtin's vec4 type."""
        float_type = self.register_primitive_type("float")
        vec4_type = self.register_vector_type(float_type, 4)
        vector_info = self.vector_component_type_and_count(value.type.base_type)

        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_count == 4:
                converted = self.convert_value_to_type(value, vec4_type)
                return converted if self.value_has_type(converted, vec4_type) else None
            if component_count == 3:
                component_type = self.register_primitive_type(component_type_name)
                components = [
                    self.convert_scalar_to_type(
                        self.composite_extract(value, component_type, index),
                        float_type,
                    )
                    for index in range(3)
                ]
                components.append(self.register_constant(1.0, float_type))
                return self.composite_construct(vec4_type, components)
            return None

        member_info = self.struct_member_info(value.type.base_type, "position")
        if member_info is None:
            member_info = self.struct_member_info(value.type.base_type, "gl_Position")
        if member_info is None:
            return None

        member_index, member_type = member_info
        member_value = self.composite_extract(value, member_type, member_index)
        return self.mesh_vertex_position_from_value(member_value)

    def process_mesh_output_assignment(self, target, value_expr) -> bool:
        """Lower assignments to mesh output signature parameters."""
        access = self.mesh_output_assignment_access(target)
        if access is None:
            return False

        info, index_expr, member_name, member_component = access
        index = self.process_expression(index_expr)
        if index is None:
            self.refuse_lowering("SPIR-V mesh output assignment requires an index")
            return True

        literal_index = self.literal_int_argument(index_expr)
        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return True

        role = info["role"]
        if role == "indices":
            self.store_mesh_index_output(info, index, literal_index, value_expr)
            return True

        if member_component is not None:
            member_info = info["members"].get(member_name)
            if member_info is None:
                self.refuse_lowering(
                    f"SPIR-V mesh output {info['name']} has no " f"member {member_name}"
                )
                return True
            if not self.validate_mesh_output_member_component_target(
                info, member_name, member_component
            ):
                return True
            member_type = member_info["type"]
            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is not None:
                self.store_mesh_output_member_component(
                    info,
                    member_name,
                    member_component,
                    index,
                    literal_index,
                    value,
                )
            return True

        if member_name is not None:
            member_info = info["members"].get(member_name)
            if member_info is None:
                self.refuse_lowering(
                    f"SPIR-V mesh output {info['name']} has no " f"member {member_name}"
                )
                return True
            member_type = member_info["type"]
            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is not None:
                self.store_mesh_output_member(
                    info, member_name, index, literal_index, value
                )
            return True

        value = self.process_expression(value_expr)
        if value is None:
            return True

        element_type = info["element_type"]
        if element_type is None:
            self.refuse_lowering(
                f"SPIR-V mesh {info['name']} elements can only be written "
                "member by member"
            )
            return True
        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.refuse_lowering(
                f"SPIR-V mesh {info['name']} output assignment value "
                f"does not match {element_type.type.base_type}"
            )
            return True

        for member_name, member_info in info["members"].items():
            member_value = self.composite_extract(
                value, member_info["type"], member_info["index"]
            )
            self.store_mesh_output_member(
                info, member_name, index, literal_index, member_value
            )
        return True

    def validate_mesh_output_member_component_target(
        self, info: dict, member_name: str, component_name: str
    ) -> bool:
        member_info = info["members"][member_name]
        member_type = member_info["type"]

        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, _, _ = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.refuse_lowering(
                    f"Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return False
            return True

        if self.vector_member_info(member_type.type.base_type, component_name):
            return True

        if (
            self.vector_component_type_and_count(member_type.type.base_type) is not None
            and len(component_name) > 1
        ):
            self.refuse_lowering(
                f"Invalid vector swizzle {component_name} "
                f"for {member_type.type.base_type}"
            )
        else:
            self.refuse_lowering(
                f"SPIR-V mesh output {member_name} has no "
                f"vector component {component_name}"
            )
        return False

    def process_mesh_output_compound_assignment(
        self,
        target,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        """Lower compound assignments to mesh output signature parameters."""
        access = self.mesh_output_assignment_access(target)
        if access is None:
            return False

        info, index_expr, member_name, member_component = access
        index = self.process_expression(index_expr)
        if index is None:
            self.refuse_lowering("SPIR-V mesh output assignment requires an index")
            return True

        literal_index = self.literal_int_argument(index_expr)
        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return True

        if info["role"] == "indices":
            self.refuse_lowering(
                "Compound assignment to SPIR-V mesh indices output " "cannot be lowered"
            )
            return True
        if member_name is None:
            self.refuse_lowering(
                "Compound assignment to whole SPIR-V mesh output "
                "elements cannot be lowered"
            )
            return True

        member_info = info["members"].get(member_name)
        if member_info is None:
            self.refuse_lowering(
                f"SPIR-V mesh output {info['name']} has no " f"member {member_name}"
            )
            return True
        if member_component is not None:
            return self.process_mesh_output_member_component_compound_assignment(
                info,
                member_name,
                member_component,
                index,
                literal_index,
                value_expr,
                spv_operator,
                target_is_precise,
            )

        return self.process_mesh_output_member_compound_assignment(
            info,
            member_name,
            index,
            literal_index,
            value_expr,
            spv_operator,
            target_is_precise,
        )

    def process_mesh_output_member_compound_assignment(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        member_pointer, member_type = self.mesh_output_member_shadow_pointer(
            info, member_name, index, literal_index
        )
        if member_pointer is None or member_type is None:
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(member_pointer, member_type)
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            result = self.binary_operation(
                spv_operator, member_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_mesh_output_member(info, member_name, index, literal_index, result)
        return True

    def process_mesh_output_member_component_compound_assignment(
        self,
        info: dict,
        member_name: str,
        component_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        member_info = info["members"][member_name]
        member_type = member_info["type"]
        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, component_type, swizzle_type = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.refuse_lowering(
                    f"Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return True

            member_pointer, storage_member_type = (
                self.mesh_output_member_shadow_pointer(
                    info, member_name, index, literal_index
                )
            )
            if member_pointer is None or storage_member_type is None:
                return True

            if target_is_precise:
                self.precise_expression_depth += 1

            try:
                rhs_value = self.process_expression(value_expr)
                if rhs_value is None:
                    return True
                rhs_components = self.convert_vector_swizzle_assignment_components(
                    rhs_value, component_type, len(component_indices), component_name
                )
                if rhs_components is None:
                    return True

                rhs_vector = self.composite_construct(swizzle_type, rhs_components)
                member_value = self.load_from_variable(
                    member_pointer, storage_member_type
                )
                current_value = self.vector_shuffle(
                    member_value, swizzle_type, component_indices
                )
                result = self.binary_operation(
                    spv_operator, swizzle_type, current_value, rhs_vector
                )
            finally:
                if target_is_precise:
                    self.precise_expression_depth -= 1

            for result_index, component_index in enumerate(component_indices):
                component_value = self.composite_extract(
                    result, component_type, result_index
                )
                self.store_mesh_output_member_component_value(
                    info,
                    member_name,
                    index,
                    literal_index,
                    component_index,
                    component_type,
                    component_value,
                )
            return True

        component_info = self.vector_member_info(
            member_type.type.base_type, component_name
        )
        if component_info is None:
            if (
                self.vector_component_type_and_count(member_type.type.base_type)
                is not None
                and len(component_name) > 1
            ):
                self.refuse_lowering(
                    f"Invalid vector swizzle {component_name} "
                    f"for {member_type.type.base_type}"
                )
            else:
                self.refuse_lowering(
                    f"SPIR-V mesh output {member_name} has no "
                    f"vector component {component_name}"
                )
            return True

        component_index, component_type = component_info
        component_pointer = self.mesh_output_member_shadow_component_pointer(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
        )
        if component_pointer is None:
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(component_pointer, component_type)
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            rhs_value = self.convert_vector_component_value(
                rhs_value, component_type, component_name
            )
            if rhs_value is None:
                return True

            result = self.binary_operation(
                spv_operator, component_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_mesh_output_member_component_value(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
            result,
        )
        return True

    def mesh_output_assignment_access(self, target):
        """Return mesh output parameter assignment metadata for a target."""
        if isinstance(target, MemberAccessNode):
            member_access = self.mesh_output_member_access_info(target)
            if member_access is not None:
                info, index_expr, member_name = member_access
                return info, index_expr, member_name, None

            parent_access = self.mesh_output_member_access_info(
                getattr(target, "object", None)
            )
            if parent_access is not None:
                info, index_expr, member_name = parent_access
                return info, index_expr, member_name, str(target.member)

        if isinstance(target, ArrayAccessNode):
            access = self.mesh_output_array_access_info(target)
            if access is None:
                return None
            info, index_expr = access
            if info["role"] not in {"vertices", "indices", "primitives"}:
                return None
            return info, index_expr, None, None

        return None

    def mesh_output_member_shadow_variable(
        self,
        info: dict,
        member_name: str,
        literal_index: Optional[int],
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]

        if not self.validate_mesh_output_literal_index(
            info["role"], literal_index, f"{info['name']} output", info
        ):
            return None, None

        if info["role"] == "vertices":
            max_vertices, _ = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_vertices, max_vertices)
        else:
            _, max_primitives = self.mesh_stage_current_output_limits()
            element_count = max(info.get("count") or max_primitives, max_primitives)

        if literal_index is not None:
            element_count = max(element_count, literal_index + 1)

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            storage_member_type = self.register_vector_type(
                self.register_primitive_type("float"), 4
            )
        else:
            storage_member_type = member_type

        key = (
            info["role"],
            member_name,
            storage_member_type.id,
            element_count,
        )
        if key in self.mesh_output_member_shadow_variables:
            return self.mesh_output_member_shadow_variables[key], storage_member_type

        array_type = self.register_array_type(storage_member_type, element_count)
        variable = self.create_variable(
            array_type, "Private", f"_CrossGLMeshShadow_{info['role']}_{member_name}"
        )
        self.entry_point_private_variables.append(variable)
        self.mesh_output_member_shadow_variables[key] = variable
        return variable, storage_member_type

    def mesh_output_member_shadow_pointer(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
    ) -> Tuple[Optional[SpirvId], Optional[SpirvId]]:
        variable, member_type = self.mesh_output_member_shadow_variable(
            info, member_name, literal_index
        )
        if variable is None or member_type is None:
            return None, None

        ptr_type = self.register_pointer_type(member_type, "Private")
        access = self.access_chain(variable, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access, member_type

    def mesh_output_member_shadow_component_pointer(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        component_index: int,
        component_type: SpirvId,
    ) -> Optional[SpirvId]:
        variable, member_type = self.mesh_output_member_shadow_variable(
            info, member_name, literal_index
        )
        if variable is None or member_type is None:
            return None

        int_type = self.register_primitive_type("int")
        component_index_id = self.register_constant(component_index, int_type)
        ptr_type = self.register_pointer_type(component_type, "Private")
        access = self.access_chain(variable, [index, component_index_id], ptr_type)

        self.variable_value_types[access.id] = component_type
        return access

    def mesh_output_member_access_info(self, target):
        if not isinstance(target, MemberAccessNode):
            return None
        array_access = getattr(target, "object", None)
        if not isinstance(array_access, ArrayAccessNode):
            return None
        access = self.mesh_output_array_access_info(array_access)
        if access is None:
            return None
        info, index_expr = access
        if info["role"] not in {"vertices", "primitives"}:
            return None
        return info, index_expr, str(target.member)

    def mesh_output_array_access_info(self, node):
        array_expr = getattr(node, "array", getattr(node, "array_expr", None))
        name = self.expression_name(array_expr)
        info = self.current_mesh_output_parameters.get(name)
        if info is None:
            return None
        index_expr = getattr(node, "index", getattr(node, "index_expr", None))
        return info, index_expr

    def store_mesh_index_output(
        self, info: dict, index: SpirvId, literal_index: Optional[int], value_expr
    ):
        builtin_info = self.mesh_primitive_index_builtin_info()
        if builtin_info is None:
            self.refuse_lowering(
                "SPIR-V mesh SetPrimitive requires point, line, or "
                "triangle output topology"
            )
            return
        _, element_type = builtin_info

        value = self.process_expression(value_expr)
        if value is None:
            return
        value = self.convert_value_to_type(value, element_type)
        if not self.value_has_type(value, element_type):
            self.refuse_lowering(
                "SPIR-V mesh indices output assignment value does not "
                f"match {element_type.type.base_type}"
            )
            return

        primitive_output, _ = self.ensure_mesh_primitive_index_output(literal_index)
        if primitive_output is None:
            return

        ptr_type = self.register_pointer_type(element_type, "Output")
        access = self.access_chain(primitive_output, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.store_to_variable(access, value)
