"""Mesh output stores, synchronization and group operations in the SPIR-V generator: mesh output member stores, mesh payload dispatch, vector and matrix constructor flattening, scope constants, execution models, atomic fences, synchronization calls, and group non-uniform arithmetic, ballot, broadcast, shuffle, partitioned and quad operations."""

from typing import (
    List,
    Optional,
)

from ..ast import (
    BinaryOpNode,
    FunctionCallNode,
    MeshOpNode,
)
from .spirv_quad_lane_reads import lower_spirv_quad_lane_read
from .spirv_subgroup_barriers import emit_spirv_subgroup_barrier
from .spirv_values import (
    SPIRVAtomicFenceLoweringError,
    SpirvId,
    UnsupportedSPIRVFeatureError,
)
from .spirv_wave_extended import lower_spirv_wave_extension


class SpirvGroupOperationMixin:
    """Group operation members of ``VulkanSPIRVCodeGen``."""

    def store_mesh_output_member(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value: SpirvId,
    ):
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            value = self.mesh_vertex_position_from_value(value)
            if value is None:
                self.refuse_lowering(
                    "SPIR-V mesh vertex output position assignment "
                    "requires a vec3, vec4, or struct with a position member"
                )
                return
            vertex_output = self.ensure_mesh_vertex_position_output(literal_index)
            if vertex_output is None:
                return
            int_type = self.register_primitive_type("int")
            member_index = self.register_constant(0, int_type)
            vec4_type = self.register_vector_type(
                self.register_primitive_type("float"), 4
            )
            ptr_type = self.register_pointer_type(vec4_type, "Output")
            access = self.access_chain(vertex_output, [index, member_index], ptr_type)
            self.variable_value_types[access.id] = vec4_type
            self.store_to_variable(access, value)
            shadow_access, _ = self.mesh_output_member_shadow_pointer(
                info, member_name, index, literal_index
            )
            if shadow_access is not None:
                self.store_to_variable(shadow_access, value)
            return

        variable = self.ensure_mesh_member_output_variable(
            info, member_name, member_type, semantic, literal_index
        )
        if variable is None:
            return

        value = self.convert_value_to_type(value, member_type)
        if not self.value_has_type(value, member_type):
            self.refuse_lowering(
                f"SPIR-V mesh output {member_name} assignment value "
                f"does not match {member_type.type.base_type}"
            )
            return

        ptr_type = self.register_pointer_type(member_type, "Output")
        access = self.access_chain(variable, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        self.store_to_variable(access, value)
        shadow_access, _ = self.mesh_output_member_shadow_pointer(
            info, member_name, index, literal_index
        )
        if shadow_access is not None:
            self.store_to_variable(shadow_access, value)

    def store_mesh_output_member_component(
        self,
        info: dict,
        member_name: str,
        component_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        value: SpirvId,
    ):
        member_info = info["members"][member_name]
        member_type = member_info["type"]

        component_indices = None
        component_type = None
        component_values = None

        swizzle_info = self.vector_swizzle_info(
            member_type.type.base_type, component_name
        )
        if swizzle_info is not None:
            component_indices, component_type, _ = swizzle_info
            if len(set(component_indices)) != len(component_indices):
                self.refuse_lowering(
                    f"Cannot assign to vector swizzle {component_name} "
                    "with duplicate components"
                )
                return
            component_values = self.convert_vector_swizzle_assignment_components(
                value, component_type, len(component_indices), component_name
            )
        else:
            member_component_info = self.vector_member_info(
                member_type.type.base_type, component_name
            )
            if member_component_info is not None:
                component_index, component_type = member_component_info
                component_indices = [component_index]
                component_value = self.convert_vector_component_value(
                    value, component_type, component_name
                )
                component_values = (
                    [component_value] if component_value is not None else None
                )

        if component_indices is None or component_type is None:
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
            return
        if component_values is None:
            return

        for component_index, component_value in zip(
            component_indices, component_values
        ):
            self.store_mesh_output_member_component_value(
                info,
                member_name,
                index,
                literal_index,
                component_index,
                component_type,
                component_value,
            )

    def store_mesh_output_member_component_value(
        self,
        info: dict,
        member_name: str,
        index: SpirvId,
        literal_index: Optional[int],
        component_index: int,
        component_type: SpirvId,
        component_value: SpirvId,
    ):
        member_info = info["members"][member_name]
        semantic = member_info.get("semantic")
        member_type = member_info["type"]
        int_type = self.register_primitive_type("int")
        component_index_id = self.register_constant(component_index, int_type)
        ptr_type = self.register_pointer_type(component_type, "Output")

        if (
            info["role"] == "vertices"
            and self.mesh_output_member_builtin("vertices", semantic) == "Position"
        ):
            vertex_output = self.ensure_mesh_vertex_position_output(literal_index)
            if vertex_output is None:
                return
            member_index = self.register_constant(0, int_type)
            access = self.access_chain(
                vertex_output, [index, member_index, component_index_id], ptr_type
            )
        else:
            variable = self.ensure_mesh_member_output_variable(
                info, member_name, member_type, semantic, literal_index
            )
            if variable is None:
                return
            access = self.access_chain(variable, [index, component_index_id], ptr_type)

        self.variable_value_types[access.id] = component_type
        self.store_to_variable(access, component_value)
        shadow_access = self.mesh_output_member_shadow_component_pointer(
            info,
            member_name,
            index,
            literal_index,
            component_index,
            component_type,
        )
        if shadow_access is not None:
            self.store_to_variable(shadow_access, component_value)

    def process_dispatch_mesh_operation(self, expr: MeshOpNode) -> Optional[SpirvId]:
        """Lower task-shader DispatchMesh to the SPIR-V mesh-task terminator."""
        operation = expr.operation
        if self.current_execution_model != "TaskEXT":
            return self.represented_ir_diagnostic_default_value(
                "mesh shader", operation
            )

        arguments = getattr(expr, "arguments", []) or []
        if len(arguments) not in {3, 4}:
            self.refuse_lowering(
                "SPIR-V mesh DispatchMesh requires exactly 3 arguments, "
                "or 4 with a task payload"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        uint_type = self.register_primitive_type("uint")
        group_counts = []
        for argument in arguments[:3]:
            group_count = self.process_expression(argument)
            if group_count is None:
                self.refuse_lowering(
                    "SPIR-V mesh DispatchMesh requires group-count operands"
                )
                return self.register_constant(0, uint_type)
            if not self.validate_mesh_count_operands(
                "DispatchMesh",
                [group_count],
                "group-count",
                argument_exprs=[argument],
            ):
                return self.register_constant(0, uint_type)
            group_counts.append(self.convert_value_to_type(group_count, uint_type))

        payload_pointer = None
        if len(arguments) == 4:
            payload_pointer = self.dispatch_mesh_payload_pointer(arguments[3])
            if payload_pointer is None:
                self.refuse_lowering(
                    "SPIR-V mesh DispatchMesh payload argument requires "
                    "taskPayloadSharedEXT storage"
                )
                return self.register_constant(0, uint_type)
        elif self.task_payload_shared_variables:
            # GL_EXT_mesh_shader's EmitMeshTasksEXT(x, y, z) passes the stage's one taskPayloadSharedEXT variable,
            # which OpEmitMeshTasksEXT names; without it the mesh workgroups would read no payload.
            if len(self.task_payload_shared_variables) > 1:
                names = ", ".join(sorted(self.task_payload_shared_variables))
                self.refuse_lowering(
                    "SPIR-V mesh DispatchMesh without a payload argument is "
                    f"ambiguous between task payload variables: {names}"
                )
                return self.register_constant(0, uint_type)
            payload_pointer = next(iter(self.task_payload_shared_variables.values()))
            self.record_task_payload_interface(payload_pointer)

        self.require_capability("MeshShadingEXT")
        self.require_extension("SPV_EXT_mesh_shader")
        operands = [f"%{group_count.id}" for group_count in group_counts]
        if payload_pointer is not None:
            operands.append(f"%{payload_pointer.id}")
        self.emit("OpEmitMeshTasksEXT " + " ".join(operands))
        return None

    def dispatch_mesh_payload_pointer(self, payload_expr) -> Optional[SpirvId]:
        """Return a TaskPayloadWorkgroupEXT pointer for DispatchMesh payloads."""
        target = self.dispatch_mesh_payload_target(payload_expr)
        if target is None:
            return None

        if self.direct_expression_name(payload_expr) == target.name:
            self.record_task_payload_interface(target)
            return target

        payload_value = self.process_expression(payload_expr)
        if payload_value is None:
            return None

        target_type = self.pointer_pointee_type(target)
        if target_type is not None:
            payload_value = self.convert_value_to_type(payload_value, target_type)
            if not self.value_has_type(payload_value, target_type):
                self.refuse_lowering(
                    "SPIR-V mesh DispatchMesh payload argument type "
                    f"{payload_value.type.base_type} does not match "
                    f"{target_type.type.base_type}"
                )
                return None

        self.store_to_variable(target, payload_value)
        self.record_task_payload_interface(target)
        return target

    def record_task_payload_interface(self, payload_pointer: SpirvId):
        """Associate a task payload variable with the current task entry point."""
        if self.current_function_id is None:
            return
        self.task_payload_interface_by_function[self.current_function_id] = (
            payload_pointer
        )

    def dispatch_mesh_payload_target(self, payload_expr) -> Optional[SpirvId]:
        """Find the shared task payload variable used by DispatchMesh."""
        if not self.task_payload_shared_variables:
            return None

        direct_name = self.direct_expression_name(payload_expr)
        if direct_name in self.task_payload_shared_variables:
            return self.task_payload_shared_variables[direct_name]

        payload_type = self.dispatch_mesh_payload_expression_type(payload_expr)
        if payload_type is not None:
            matches = [
                variable
                for variable in self.task_payload_shared_variables.values()
                if (
                    self.pointer_pointee_type(variable) is not None
                    and self.pointer_pointee_type(variable).type.base_type
                    == payload_type.type.base_type
                )
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                names = ", ".join(sorted(variable.name or "" for variable in matches))
                self.refuse_lowering(
                    "SPIR-V mesh DispatchMesh payload target is "
                    f"ambiguous for {payload_type.type.base_type}: {names}"
                )
                return None

        if len(self.task_payload_shared_variables) == 1:
            return next(iter(self.task_payload_shared_variables.values()))

        names = ", ".join(sorted(self.task_payload_shared_variables))
        self.refuse_lowering(
            "SPIR-V mesh DispatchMesh payload target is ambiguous: " f"{names}"
        )
        return None

    def dispatch_mesh_payload_expression_type(self, expr) -> Optional[SpirvId]:
        """Return a statically known payload value type without emitting code."""
        direct_name = self.direct_expression_name(expr)
        if direct_name:
            variable = self.local_variables.get(
                direct_name
            ) or self.resolve_global_variable(direct_name)
            if variable is not None:
                value_type = self.variable_value_types.get(variable.id)
                if value_type is not None:
                    return value_type
                registered_type = self.value_types.get(variable.id)
                if registered_type is not None:
                    return registered_type

        if isinstance(expr, FunctionCallNode):
            callee_name = self.function_call_name(expr)
            function_signature = self.resolve_function_signature(callee_name)
            if function_signature is not None:
                return_type = function_signature[0]
                if return_type.type.base_type != "void":
                    return return_type
            if callee_name in self.struct_types:
                return self.struct_types[callee_name]

        return None

    def flatten_vector_constructor_args(
        self,
        function_name: str,
        args: List[SpirvId],
        component_type: SpirvId,
        component_count: int,
    ) -> List[SpirvId]:
        if (
            len(args) == 1
            and self.vector_component_type_and_count(args[0].type.base_type) is None
        ):
            component_arg = self.convert_vector_constructor_scalar(
                args[0], component_type, function_name
            )
            return [component_arg] * component_count

        flattened_args = []
        before_last_argument = 0
        for arg in args:
            before_last_argument = len(flattened_args)
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is None:
                flattened_args.append(
                    self.convert_vector_constructor_scalar(
                        arg, component_type, function_name
                    )
                )
                continue

            source_component_type_name, source_component_count = vector_info
            source_component_type = self.register_primitive_type(
                source_component_type_name
            )
            for index in range(source_component_count):
                source_value = self.composite_extract(arg, source_component_type, index)
                flattened_args.append(
                    self.convert_vector_constructor_scalar(
                        source_value, component_type, function_name
                    )
                )

        if len(flattened_args) < component_count:
            self.refuse_lowering(
                f"Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}"
            )
            default_value = self.default_value_for_type(component_type)
            flattened_args.extend(
                [default_value] * (component_count - len(flattened_args))
            )
        elif len(flattened_args) > component_count:
            # GLSL 4.60 section 5.4.2 drops the unused components of the last argument
            # (vec2(v3)); an argument that contributes nothing is an error.
            if before_last_argument >= component_count:
                self.refuse_lowering(
                    f"Constructor {function_name} expected {component_count} "
                    f"components but got {len(flattened_args)}"
                )
            flattened_args = flattened_args[:component_count]

        return flattened_args

    def convert_vector_constructor_scalar(
        self, value: SpirvId, component_type: SpirvId, function_name: str
    ) -> SpirvId:
        # Numeric lanes convert to bool by comparing against zero (GLSL 4.60 section 5.4.1),
        # which convert_scalar_to_type does.
        target_type = self.normalize_primitive_name(component_type.type.base_type)
        converted_value = self.convert_scalar_to_type(value, component_type)
        source_type = self.normalize_primitive_name(converted_value.type.base_type)
        if source_type != target_type:
            self.refuse_lowering(
                f"Constructor {function_name} cannot convert "
                f"{source_type} component to {target_type}"
            )
            return self.default_value_for_type(component_type)

        return converted_value

    def flatten_matrix_constructor_components(
        self,
        function_name: str,
        args: List[SpirvId],
        component_type: SpirvId,
        component_count: int,
    ) -> List[SpirvId]:
        flattened_args = []
        before_last_argument = 0
        for arg in args:
            before_last_argument = len(flattened_args)
            flattened_args.extend(
                self.flatten_matrix_constructor_arg(function_name, arg, component_type)
            )

        if len(flattened_args) < component_count:
            self.refuse_lowering(
                f"Constructor {function_name} expected {component_count} "
                f"components but got {len(flattened_args)}"
            )
            default_value = self.default_value_for_type(component_type)
            flattened_args.extend(
                [default_value] * (component_count - len(flattened_args))
            )
        elif len(flattened_args) > component_count:
            # As for vectors, only the last argument may contribute unused components.
            if before_last_argument >= component_count:
                self.refuse_lowering(
                    f"Constructor {function_name} expected {component_count} "
                    f"components but got {len(flattened_args)}"
                )
            flattened_args = flattened_args[:component_count]

        return flattened_args

    def flatten_matrix_constructor_arg(
        self, function_name: str, arg: SpirvId, component_type: SpirvId
    ) -> List[SpirvId]:
        vector_info = self.vector_component_type_and_count(arg.type.base_type)
        if vector_info is not None:
            source_component_type_name, source_component_count = vector_info
            source_component_type = self.register_primitive_type(
                source_component_type_name
            )
            return [
                self.convert_vector_constructor_scalar(
                    self.composite_extract(arg, source_component_type, index),
                    component_type,
                    function_name,
                )
                for index in range(source_component_count)
            ]

        registered_type = self.find_registered_type_by_base(arg.type.base_type)
        matrix_info = self.matrix_type_info_from_type(registered_type)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            column_info = self.vector_type_info_from_type(column_type)
            if column_info is None:
                return []

            source_component_type, row_count = column_info
            flattened_components = []
            for column_index in range(column_count):
                column_value = self.composite_extract(arg, column_type, column_index)
                for row_index in range(row_count):
                    flattened_components.append(
                        self.convert_vector_constructor_scalar(
                            self.composite_extract(
                                column_value, source_component_type, row_index
                            ),
                            component_type,
                            function_name,
                        )
                    )
            return flattened_components

        return [
            self.convert_vector_constructor_scalar(arg, component_type, function_name)
        ]

    def spirv_scope_constant(self, scope: str) -> SpirvId:
        scope_values = {
            "CrossDevice": 0,
            "Device": 1,
            "Workgroup": 2,
            "Subgroup": 3,
            "Invocation": 4,
        }
        if scope == "Device":
            self.uses_device_scope_memory_operation = True
            if "VulkanMemoryModel" in self.required_capabilities:
                self.require_capability("VulkanMemoryModelDeviceScopeKHR")
        return self.register_constant(
            scope_values[scope], self.register_primitive_type("uint")
        )

    def spirv_memory_semantics_constant(self, *semantics: str) -> SpirvId:
        semantic_values = {
            "AcquireRelease": 0x8,
            "UniformMemory": 0x40,
            "WorkgroupMemory": 0x100,
            "ImageMemory": 0x800,
        }
        value = 0
        for semantic in semantics:
            value |= semantic_values[semantic]
        return self.register_constant(value, self.register_primitive_type("uint"))

    def execution_models_for_function(self, function_node) -> set:
        if function_node is None:
            return set()
        models = set(self.function_execution_models.get(id(function_node), ()))
        source_function_id = getattr(function_node, "_generic_source_function_id", None)
        if source_function_id is not None:
            models.update(self.function_execution_models.get(source_function_id, ()))
        return models

    def current_synchronization_execution_models(self) -> set:
        execution_models = self.execution_models_for_function(
            self.current_function_node
        )
        if self.current_stage is None:
            if execution_models:
                return execution_models
        if self.current_execution_model is not None:
            return {self.current_execution_model}
        return execution_models

    def current_execution_model_label(self) -> str:
        execution_models = self.current_synchronization_execution_models()
        if execution_models:
            return ", ".join(sorted(execution_models))
        return "unknown"

    def can_emit_workgroup_synchronization(self) -> bool:
        execution_models = self.current_synchronization_execution_models()
        workgroup_execution_models = {
            "GLCompute",
            "MeshEXT",
            "TaskEXT",
        }
        return bool(execution_models) and execution_models.issubset(
            workgroup_execution_models
        )

    def atomic_fence_operand_identifier(self, expr) -> Optional[str]:
        name = self.expression_name(expr)
        if name is None:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def atomic_fence_operand_text(self, expr) -> str:
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.atomic_fence_operand_text(expr.left)} | "
                f"{self.atomic_fence_operand_text(expr.right)}"
            )
        name = self.atomic_fence_operand_identifier(expr)
        return name if name is not None else self.diagnostic_expression(expr)

    def collect_atomic_fence_memory_flags(self, expr) -> Optional[frozenset]:
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_atomic_fence_memory_flags(expr.left)
            right = self.collect_atomic_fence_memory_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        name = self.atomic_fence_operand_identifier(expr)
        if name in self.ATOMIC_FENCE_MEMORY_FLAGS:
            return frozenset({name})
        return None

    def atomic_fence_contract_error(
        self,
        reason: str,
        args: List,
        *,
        source_location=None,
    ) -> SPIRVAtomicFenceLoweringError:
        rendered = [self.atomic_fence_operand_text(arg) for arg in args]
        return SPIRVAtomicFenceLoweringError(
            reason,
            memory_flags=rendered[0] if rendered else None,
            memory_order=rendered[1] if len(rendered) > 1 else None,
            thread_scope=rendered[2] if len(rendered) > 2 else None,
            source_location=source_location,
        )

    def reject_atomic_thread_fence(self, call_node):
        args = list(getattr(call_node, "args", ()) or ())
        source_location = getattr(call_node, "source_location", None)
        if len(args) != 3:
            raise self.atomic_fence_contract_error(
                "invalid-argument-count",
                args,
                source_location=source_location,
            )

        flags = self.collect_atomic_fence_memory_flags(args[0])
        order = self.atomic_fence_operand_identifier(args[1])
        scope = self.atomic_fence_operand_identifier(args[2])
        if flags is None:
            raise self.atomic_fence_contract_error(
                "unknown-memory-flags",
                args,
                source_location=source_location,
            )
        if order not in self.ATOMIC_FENCE_MEMORY_ORDERS:
            raise self.atomic_fence_contract_error(
                "unknown-memory-order",
                args,
                source_location=source_location,
            )
        if scope not in self.ATOMIC_FENCE_THREAD_SCOPES:
            raise self.atomic_fence_contract_error(
                "unknown-thread-scope",
                args,
                source_location=source_location,
            )

        reason = (
            "unsupported-system-thread-scope"
            if scope == "thread_scope_system"
            else "atomic-fence-lowering-deferred"
        )
        raise self.atomic_fence_contract_error(
            reason,
            args,
            source_location=source_location,
        )

    def is_canonical_atomic_thread_fence_call(self, function_name: str) -> bool:
        return function_name == "atomicThreadFence" and not self.has_function_reference(
            function_name
        )

    def call_synchronization_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        synchronization_functions = {
            "allMemoryBarrier",
            "barrier",
            "deviceMemoryBarrier",
            "groupMemoryBarrier",
            "workgroupBarrier",
            "workgroupExecutionBarrier",
            "memoryBarrier",
            "memoryBarrierBuffer",
            "memoryBarrierImage",
            "memoryBarrierShared",
        }
        if emit_spirv_subgroup_barrier(self, function_name, args):
            return self.register_constant(0, self.register_primitive_type("int"))
        if function_name not in synchronization_functions:
            return None

        if args:
            self.refuse_lowering(
                f"synchronization builtin '{function_name}' " "requires 0 operands"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        workgroup_synchronization_functions = {
            "barrier",
            "groupMemoryBarrier",
            "memoryBarrierShared",
            "workgroupBarrier",
            "workgroupExecutionBarrier",
        }
        if (
            function_name == "barrier"
            and self.current_synchronization_execution_models()
            == {"TessellationControl"}
        ):
            # GLSL 4.60 section 8.16: a tessellation control barrier() orders the patch's
            # invocations; glslang lowers it to a workgroup control barrier without memory semantics.
            workgroup_scope = self.spirv_scope_constant("Workgroup")
            invocation_scope = self.spirv_scope_constant("Invocation")
            no_memory_semantics = self.spirv_memory_semantics_constant()
            self.emit(
                f"OpControlBarrier %{workgroup_scope.id} %{invocation_scope.id} "
                f"%{no_memory_semantics.id}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if (
            function_name in workgroup_synchronization_functions
            and not self.can_emit_workgroup_synchronization()
        ):
            self.refuse_lowering(
                f"synchronization builtin '{function_name}' requires "
                "a workgroup-capable execution model; current execution model: "
                f"{self.current_execution_model_label()}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if function_name == "workgroupExecutionBarrier":
            workgroup_scope = self.spirv_scope_constant("Workgroup")
            no_memory_semantics = self.spirv_memory_semantics_constant()
            self.emit(
                f"OpControlBarrier %{workgroup_scope.id} %{workgroup_scope.id} "
                f"%{no_memory_semantics.id}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if function_name in {"barrier", "workgroupBarrier"}:
            workgroup_scope = self.spirv_scope_constant("Workgroup")
            workgroup_semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "WorkgroupMemory"
            )
            self.emit(
                f"OpControlBarrier %{workgroup_scope.id} %{workgroup_scope.id} "
                f"%{workgroup_semantics.id}"
            )
            return self.register_constant(0, self.register_primitive_type("int"))

        if function_name == "memoryBarrierShared":
            scope = self.spirv_scope_constant("Workgroup")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "WorkgroupMemory"
            )
        elif function_name == "groupMemoryBarrier":
            # GLSL 4.60 section 8.16: groupMemoryBarrier orders all memory accesses within the workgroup;
            # glslang emits a Workgroup-scope barrier over uniform, workgroup and image memory.
            scope = self.spirv_scope_constant("Workgroup")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory", "WorkgroupMemory", "ImageMemory"
            )
        elif function_name == "deviceMemoryBarrier":
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory", "ImageMemory"
            )
        elif function_name == "memoryBarrierBuffer":
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory"
            )
        elif function_name == "memoryBarrierImage":
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "ImageMemory"
            )
        else:
            scope = self.spirv_scope_constant("Device")
            semantics = self.spirv_memory_semantics_constant(
                "AcquireRelease", "UniformMemory", "WorkgroupMemory", "ImageMemory"
            )

        self.emit(f"OpMemoryBarrier %{scope.id} %{semantics.id}")
        return self.register_constant(0, self.register_primitive_type("int"))

    def require_group_non_uniform(self, capability: Optional[str] = None):
        self.require_capability("GroupNonUniform")
        if capability is not None:
            self.require_capability(capability)

    def require_group_non_uniform_partitioned_nv(self):
        self.require_capability("GroupNonUniformPartitionedNV")
        self.require_extension("SPV_NV_shader_subgroup_partitioned")

    def subgroup_scope_id(self) -> SpirvId:
        self.require_group_non_uniform()
        return self.spirv_scope_constant("Subgroup")

    def wave_result_default(self, operation: str, args: List[SpirvId]) -> SpirvId:
        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveIsFirstLane",
        }:
            return self.default_value_for_type(self.register_primitive_type("bool"))
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            uint_type = self.register_primitive_type("uint")
            return self.default_value_for_type(self.register_vector_type(uint_type, 4))
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return self.default_value_for_type(self.register_primitive_type("uint"))
        if args:
            return self.default_value_for_type(
                self.ensure_registered_type(args[0].type)
            )
        return self.default_value_for_type(self.register_primitive_type("uint"))

    def call_wave_operation(
        self, operation: str, argument_exprs: List
    ) -> Optional[SpirvId]:
        args = []
        for argument in argument_exprs:
            value = self.process_expression(argument)
            if value is None:
                self.refuse_lowering(f"Failed to evaluate argument for {operation}")
                value = self.default_value_for_type(
                    self.register_primitive_type("uint")
                )
            args.append(value)

        extended = lower_spirv_wave_extension(self, operation, argument_exprs, args)
        if extended is not None:
            return extended

        if operation == "WaveGetLaneCount":
            if args:
                self.refuse_lowering("WaveGetLaneCount takes no arguments")
                return self.wave_result_default(operation, args)
            builtin = self.ensure_compute_builtin("gl_SubgroupSize")
            return self.get_variable_value(builtin) if builtin is not None else None

        if operation == "WaveGetLaneIndex":
            if args:
                self.refuse_lowering("WaveGetLaneIndex takes no arguments")
                return self.wave_result_default(operation, args)
            builtin = self.ensure_compute_builtin("gl_SubgroupInvocationID")
            return self.get_variable_value(builtin) if builtin is not None else None

        if operation == "WaveIsFirstLane":
            if args:
                self.refuse_lowering("WaveIsFirstLane takes no arguments")
                return self.wave_result_default(operation, args)
            self.require_group_non_uniform("GroupNonUniformVote")
            bool_type = self.register_primitive_type("bool")
            scope = self.subgroup_scope_id()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpGroupNonUniformElect %{bool_type.id} %{scope.id}"
            )
            spirv_id = SpirvId(id_value, bool_type.type)
            self.value_types[id_value] = bool_type
            return spirv_id

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WavePrefixSum",
            "WavePrefixProduct",
            "WavePrefixInclusiveSum",
            "WavePrefixInclusiveProduct",
        }:
            return self.call_group_non_uniform_arithmetic(operation, args)

        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return self.call_group_non_uniform_vote(operation, args)

        if operation == "WaveActiveBallot":
            return self.call_group_non_uniform_ballot(operation, args)

        if operation == "WaveReadLaneAt":
            return self.call_group_non_uniform_shuffle(operation, args)

        if operation == "WaveShuffleAndFillUp":
            return self.call_group_non_uniform_shuffle_and_fill_up(operation, args)

        if operation in {"WaveShuffleDown", "WaveShuffleUp", "WaveShuffleXor"}:
            return self.call_group_non_uniform_shuffle_relative(operation, args)

        if operation == "WaveReadLaneFirst":
            return self.call_group_non_uniform_broadcast_first(operation, args)

        if operation == "WaveMatch":
            return self.call_group_non_uniform_partition(args)

        if operation in {
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            return self.call_group_non_uniform_partitioned_arithmetic(operation, args)

        if operation in {
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
        }:
            return self.call_group_non_uniform_quad(operation, argument_exprs, args)

        self.refuse_lowering(f"{operation} is not supported by the SPIR-V backend yet")
        return self.wave_result_default(operation, args)

    def call_group_non_uniform_arithmetic(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.refuse_lowering(f"{operation} requires exactly one argument")
            return self.wave_result_default(operation, args)

        source_result_type = self.ensure_registered_type(args[0].type)
        result_type = source_result_type
        operand = args[0]
        floating_bitwise_type = self.floating_bitwise_wave_integer_type(
            operation, source_result_type
        )
        if floating_bitwise_type is not None:
            result_type = floating_bitwise_type
            operand = self.bitcast_wave_value(operand, result_type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if operation in {"WavePrefixSum", "WavePrefixProduct"}:
            group_operation = "ExclusiveScan"
        elif operation in {"WavePrefixInclusiveSum", "WavePrefixInclusiveProduct"}:
            group_operation = "InclusiveScan"
        else:
            group_operation = "Reduce"

        arithmetic_ops = {
            "WaveActiveSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WavePrefixSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WaveActiveProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WavePrefixProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WavePrefixInclusiveSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WavePrefixInclusiveProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WaveActiveMin": {
                "float": "OpGroupNonUniformFMin",
                "double": "OpGroupNonUniformFMin",
                "int": "OpGroupNonUniformSMin",
                "uint": "OpGroupNonUniformUMin",
            },
            "WaveActiveMax": {
                "float": "OpGroupNonUniformFMax",
                "double": "OpGroupNonUniformFMax",
                "int": "OpGroupNonUniformSMax",
                "uint": "OpGroupNonUniformUMax",
            },
            "WaveActiveBitAnd": {
                "int": "OpGroupNonUniformBitwiseAnd",
                "uint": "OpGroupNonUniformBitwiseAnd",
            },
            "WaveActiveBitOr": {
                "int": "OpGroupNonUniformBitwiseOr",
                "uint": "OpGroupNonUniformBitwiseOr",
            },
            "WaveActiveBitXor": {
                "int": "OpGroupNonUniformBitwiseXor",
                "uint": "OpGroupNonUniformBitwiseXor",
            },
        }
        opcode = arithmetic_ops[operation].get(component_type)
        if opcode is None:
            self.refuse_lowering(
                f"{operation} requires a compatible arithmetic or "
                f"bitwise operand; got {result_type.type.base_type}"
            )
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformArithmetic")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} %{scope.id} "
            f"{group_operation} %{operand.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        if floating_bitwise_type is not None:
            return self.bitcast_wave_value(spirv_id, source_result_type)
        return spirv_id

    def floating_bitwise_wave_integer_type(
        self, operation: str, value_type: SpirvId
    ) -> Optional[SpirvId]:
        if operation not in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
        }:
            return None

        vector_info = self.vector_component_type_and_count(value_type.type.base_type)
        component_type = (
            vector_info[0]
            if vector_info is not None
            else self.normalize_primitive_name(value_type.type.base_type)
        )
        if component_type != "float":
            return None

        integer_component = self.register_primitive_type("uint")
        if vector_info is None:
            return integer_component
        return self.register_vector_type(integer_component, vector_info[1])

    def bitcast_wave_value(self, value: SpirvId, target_type: SpirvId) -> SpirvId:
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpBitcast %{target_type.id} %{value.id}")
        result = SpirvId(id_value, target_type.type)
        self.value_types[id_value] = target_type
        if self.is_non_uniform_value(value):
            self.mark_non_uniform_result(result)
        return result

    def call_group_non_uniform_vote(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.refuse_lowering(f"{operation} requires exactly one bool argument")
            return self.wave_result_default(operation, args)

        bool_type = self.register_primitive_type("bool")
        if args[0].type.base_type != "bool":
            self.refuse_lowering(f"{operation} requires a scalar bool argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformVote")
        scope = self.subgroup_scope_id()
        opcode = (
            "OpGroupNonUniformAll"
            if operation == "WaveActiveAllTrue"
            else "OpGroupNonUniformAny"
        )
        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{bool_type.id} %{scope.id} %{args[0].id}")

        spirv_id = SpirvId(id_value, bool_type.type)
        self.value_types[id_value] = bool_type
        return spirv_id

    def call_group_non_uniform_ballot(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        ballot_type = self.register_vector_type(uint_type, 4)
        if len(args) != 1:
            self.refuse_lowering(f"{operation} requires exactly one bool argument")
            return self.wave_result_default(operation, args)

        if args[0].type.base_type != "bool":
            self.refuse_lowering(f"{operation} requires a scalar bool argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformBallot")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformBallot %{ballot_type.id} "
            f"%{scope.id} %{args[0].id}"
        )

        spirv_id = SpirvId(id_value, ballot_type.type)
        self.value_types[id_value] = ballot_type
        return spirv_id

    def call_group_non_uniform_shuffle(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 2:
            self.refuse_lowering(f"{operation} requires value and lane arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        lane = self.convert_wave_lane_operand(operation, args[1])
        if lane is None:
            return self.wave_result_default(operation, args)

        # WaveReadLaneAt reads from an arbitrary (possibly dynamic) lane, so it
        # lowers to OpGroupNonUniformShuffle (capability GroupNonUniformShuffle),
        # mirroring GLSL subgroupShuffle. OpGroupNonUniformBroadcast cannot be
        # used here because it requires a constant lane id before SPIR-V 1.5.
        self.require_group_non_uniform("GroupNonUniformShuffle")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformShuffle %{result_type.id} "
            f"%{scope.id} %{args[0].id} %{lane.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_broadcast_first(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 1:
            self.refuse_lowering(f"{operation} requires exactly one argument")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        self.require_group_non_uniform("GroupNonUniformBallot")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformBroadcastFirst %{result_type.id} "
            f"%{scope.id} %{args[0].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_shuffle_relative(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        # `WaveShuffleDown/Up(value, delta)` and `WaveShuffleXor(value, mask)`
        # read a value from a lane relative to the current one, mirroring Metal
        # `simd_shuffle_down/up/xor` and GLSL `subgroupShuffleDown/Up/Xor`.
        # ShuffleDown/Up need the GroupNonUniformShuffleRelative capability;
        # ShuffleXor lives under GroupNonUniformShuffle.
        if len(args) != 2:
            self.refuse_lowering(f"{operation} requires value and delta arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        # The delta/mask operand is an unsigned lane offset.
        delta = self.convert_wave_lane_operand(operation, args[1])
        if delta is None:
            return self.wave_result_default(operation, args)

        opcodes = {
            "WaveShuffleDown": (
                "OpGroupNonUniformShuffleDown",
                "GroupNonUniformShuffleRelative",
            ),
            "WaveShuffleUp": (
                "OpGroupNonUniformShuffleUp",
                "GroupNonUniformShuffleRelative",
            ),
            "WaveShuffleXor": ("OpGroupNonUniformShuffleXor", "GroupNonUniformShuffle"),
        }
        opcode, capability = opcodes[operation]
        self.require_group_non_uniform(capability)
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{scope.id} %{args[0].id} %{delta.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_shuffle_and_fill_up(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 3:
            self.refuse_lowering(
                f"{operation} requires exactly three arguments "
                "(value, fill, and delta)"
            )
            return self.wave_result_default(operation, args)

        value, fill, delta_value = args
        result_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        fill_type = self.registered_value_type(fill) or self.ensure_registered_type(
            fill.type
        )
        if result_type.type.base_type != fill_type.type.base_type:
            self.refuse_lowering(
                f"{operation} requires compatible value and fill types"
            )
            return self.wave_result_default(operation, args)

        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        component_type = (
            result_vector[0]
            if result_vector is not None
            else self.normalize_primitive_name(result_type.type.base_type)
        )
        if component_type not in {
            "bool",
            "float",
            "double",
            *self.INTEGER_TYPE_NAMES,
        }:
            self.refuse_lowering(
                f"{operation} requires a scalar or vector Boolean, "
                "integer, or floating-point value"
            )
            return self.wave_result_default(operation, args)

        delta_type = self.registered_value_type(
            delta_value
        ) or self.ensure_registered_type(delta_value.type)
        if (
            self.vector_component_type_and_count(delta_type.type.base_type) is not None
            or self.normalize_primitive_name(delta_type.type.base_type)
            not in self.INTEGER_TYPE_NAMES
        ):
            self.refuse_lowering(f"{operation} requires a scalar integer delta")
            return self.wave_result_default(operation, args)

        uint_type = self.register_primitive_type("uint")
        delta = self.convert_scalar_to_type(delta_value, uint_type)
        if not self.value_has_type(delta, uint_type):
            self.refuse_lowering(f"{operation} requires a scalar integer delta")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformShuffleRelative")
        scope = self.subgroup_scope_id()
        shuffled_id = self.get_id()
        self.emit(
            f"%{shuffled_id} = OpGroupNonUniformShuffleUp %{result_type.id} "
            f"%{scope.id} %{value.id} %{delta.id}"
        )
        shuffled = SpirvId(shuffled_id, result_type.type)
        self.value_types[shuffled_id] = result_type

        invocation_variable = self.ensure_compute_builtin("gl_SubgroupInvocationID")
        if invocation_variable is None:
            raise UnsupportedSPIRVFeatureError(
                "spirv.wave_shuffle_and_fill_up",
                f"{operation} requires the SubgroupLocalInvocationId built-in",
                missing_capabilities=(
                    "GroupNonUniform",
                    "GroupNonUniformShuffleRelative",
                ),
            )
        invocation_id = self.get_variable_value(invocation_variable)
        bool_type = self.register_primitive_type("bool")
        in_range = self.binary_operation(">=", bool_type, invocation_id, delta)
        if result_vector is not None:
            selector_type = self.register_vector_type(bool_type, result_vector[1])
            in_range = self.splat_scalar_to_vector(in_range, selector_type)

        return self.select_operation(result_type, in_range, shuffled, fill)

    def call_group_non_uniform_partition(self, args: List[SpirvId]) -> SpirvId:
        if len(args) != 1:
            self.refuse_lowering("WaveMatch requires exactly one argument")
            return self.wave_result_default("WaveMatch", args)

        uint_type = self.register_primitive_type("uint")
        result_type = self.register_vector_type(uint_type, 4)
        self.require_group_non_uniform_partitioned_nv()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformPartitionNV %{result_type.id} "
            f"%{args[0].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_group_non_uniform_partitioned_arithmetic(
        self, operation: str, args: List[SpirvId]
    ) -> SpirvId:
        if len(args) != 2:
            self.refuse_lowering(f"{operation} requires value and ballot arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        arithmetic_ops = {
            "WaveMultiPrefixSum": {
                "float": "OpGroupNonUniformFAdd",
                "double": "OpGroupNonUniformFAdd",
                "int": "OpGroupNonUniformIAdd",
                "uint": "OpGroupNonUniformIAdd",
            },
            "WaveMultiPrefixProduct": {
                "float": "OpGroupNonUniformFMul",
                "double": "OpGroupNonUniformFMul",
                "int": "OpGroupNonUniformIMul",
                "uint": "OpGroupNonUniformIMul",
            },
            "WaveMultiPrefixBitAnd": {
                "int": "OpGroupNonUniformBitwiseAnd",
                "uint": "OpGroupNonUniformBitwiseAnd",
            },
            "WaveMultiPrefixBitOr": {
                "int": "OpGroupNonUniformBitwiseOr",
                "uint": "OpGroupNonUniformBitwiseOr",
            },
            "WaveMultiPrefixBitXor": {
                "int": "OpGroupNonUniformBitwiseXor",
                "uint": "OpGroupNonUniformBitwiseXor",
            },
        }
        opcode = arithmetic_ops[operation].get(component_type)
        if opcode is None:
            self.refuse_lowering(
                f"{operation} requires a compatible arithmetic or "
                f"bitwise operand; got {result_type.type.base_type}"
            )
            return self.wave_result_default(operation, args)

        if self.vector_component_type_and_count(args[1].type.base_type) != ("uint", 4):
            self.refuse_lowering(f"{operation} requires a uvec4 ballot argument")
            return self.wave_result_default(operation, args)

        self.require_group_non_uniform("GroupNonUniformArithmetic")
        self.require_group_non_uniform_partitioned_nv()
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} %{scope.id} "
            f"PartitionedExclusiveScanNV %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def convert_wave_lane_operand(
        self, operation: str, lane: SpirvId
    ) -> Optional[SpirvId]:
        if self.vector_component_type_and_count(lane.type.base_type) is not None:
            self.refuse_lowering(f"{operation} requires a scalar lane index")
            return None

        uint_type = self.register_primitive_type("uint")
        converted = self.convert_scalar_to_type(lane, uint_type)
        if converted.type.base_type != "uint":
            self.refuse_lowering(f"{operation} requires an integer lane index")
            return None
        return converted

    def call_group_non_uniform_quad(
        self, operation: str, argument_exprs: List, args: List[SpirvId]
    ) -> SpirvId:
        if operation in {
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            if len(args) != 1:
                self.refuse_lowering(f"{operation} requires exactly one argument")
                return self.wave_result_default(operation, args)

            directions = {
                "QuadReadAcrossX": 0,
                "QuadReadAcrossY": 1,
                "QuadReadAcrossDiagonal": 2,
            }
            result_type = self.ensure_registered_type(args[0].type)
            direction = self.register_constant(
                directions[operation], self.register_primitive_type("uint")
            )
            self.require_group_non_uniform("GroupNonUniformQuad")
            scope = self.subgroup_scope_id()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpGroupNonUniformQuadSwap %{result_type.id} "
                f"%{scope.id} %{args[0].id} %{direction.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if len(args) != 2:
            self.refuse_lowering("QuadReadLaneAt requires value and lane arguments")
            return self.wave_result_default(operation, args)

        result_type = self.ensure_registered_type(args[0].type)
        lane_value = self.literal_integer_value(argument_exprs[1], "uint")
        if lane_value is None:
            return lower_spirv_quad_lane_read(self, args[0], args[1])
        if not 0 <= lane_value <= 3:
            self.refuse_lowering(
                "a literal QuadReadLaneAt lane index must be between 0 and 3"
            )
            return self.wave_result_default(operation, args)

        lane = self.register_constant(lane_value, self.register_primitive_type("uint"))
        self.require_group_non_uniform("GroupNonUniformQuad")
        scope = self.subgroup_scope_id()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpGroupNonUniformQuadBroadcast %{result_type.id} "
            f"%{scope.id} %{args[0].id} %{lane.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id
