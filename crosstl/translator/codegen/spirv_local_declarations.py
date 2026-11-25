"""Local declarations of the SPIR-V generator: declaration scopes and blocks, fragment and clip discards, expression result type inference, built-in limits, match expression types, variable declarations, local integer constants, local storage classes, workgroup variables, local resource and array pointer aliases, storage buffer pointer aliases and offset markers, and locals that need descriptor storage."""

import re
from typing import Optional

from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    PointerReinterpretNode,
    TernaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..refusals import VulkanRefusal
from .array_utils import evaluate_literal_int_expression
from .bitfield_builtins import BITFIELD_ARITIES
from .generic_function_utils import generic_function_call_name
from .pack_builtins import spirv_pack_result_type
from .pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)
from .precise_dataflow import declaration_is_precise
from .spirv_shading_rate import shading_rate_flag_value
from .spirv_single_evaluation import (
    recording_subscript_indices,
    subscript_index_scope,
)
from .spirv_storage_subscripts import storage_buffer_subscript_element_type_name
from .spirv_values import (
    PointerOffsetState,
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)
from .task_payload_variables import dispatch_mesh_payload_names


class SpirvLocalDeclarationMixin:
    """Local declaration members of ``VulkanSPIRVCodeGen``."""

    def restore_local_declaration_scope_state(self, state: dict):
        current_pointer_offset_aliases = self.storage_buffer_pointer_offset_aliases
        current_pointer_reinterpretations = self.storage_pointer_reinterpretations
        visible_pointers = {
            *state["local_variables"].values(),
            *self.global_variables.values(),
            *self.stage_global_variables.values(),
        }
        self.local_variables = state["local_variables"]
        self.local_variable_types = state["local_variable_types"]
        self.named_constants = state["named_constants"]
        self.precise_local_variables = state["precise_local_variables"]
        self.resource_alias_variables = state["resource_alias_variables"]
        self.current_function_literal_int_constants = state["literal_int_constants"]
        self.current_function_literal_int_constant_shadows = state[
            "literal_int_constant_shadows"
        ]
        self.storage_buffer_pointer_offset_aliases = dict(
            state["pointer_offset_aliases"]
        )
        self.storage_buffer_pointer_offset_aliases.update(
            {
                pointer: offset
                for pointer, offset in current_pointer_offset_aliases.items()
                if pointer in visible_pointers
            }
        )
        self.storage_pointer_reinterpretations = dict(
            state["pointer_reinterpretations"]
        )
        self.storage_pointer_reinterpretations.update(
            {
                pointer: metadata
                for pointer, metadata in current_pointer_reinterpretations.items()
                if pointer in visible_pointers
            }
        )

    def process_block_statement(self, block: BlockNode):
        previous_state = self.local_declaration_scope_state()
        try:
            self.process_statements(block.statements)
        finally:
            self.restore_local_declaration_scope_state(previous_state)

    def process_fragment_discard_statement(self, expr) -> bool:
        """Lower CrossGL/GLSL/HLSL fragment discard statements to SPIR-V."""
        if isinstance(expr, IdentifierNode) and expr.name == "discard":
            return self.emit_fragment_discard("discard")
        if isinstance(expr, str) and expr == "discard":
            return self.emit_fragment_discard("discard")
        if not isinstance(expr, FunctionCallNode):
            return False

        callee_name = self.function_call_name(expr)
        if not isinstance(callee_name, str):
            return False

        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        if (
            callee_name == "discard"
            and not args
            and not self.has_function_reference(callee_name)
        ):
            return self.emit_fragment_discard("discard")

        if callee_name != "clip" or self.has_function_reference(callee_name):
            return False
        if len(args) != 1:
            self.refuse_lowering("SPIR-V clip discard requires exactly one operand")
            return True

        return self.process_clip_discard_statement(args[0])

    def emit_fragment_discard(self, source_name: str) -> bool:
        """Emit OpKill for fragment discard, preserving SPIR-V stage validity."""
        if self.current_execution_model != "Fragment":
            self.refuse_lowering(
                f"SPIR-V {source_name} lowers to OpKill, which is "
                "only valid in the Fragment execution model"
            )
            return True

        self.emit("OpKill")
        return True

    def process_clip_discard_statement(self, value_expr) -> bool:
        """Lower HLSL-style clip(x) to conditional fragment discard."""
        if self.current_execution_model != "Fragment":
            self.refuse_lowering(
                "SPIR-V clip lowers to OpKill, which is only valid "
                "in the Fragment execution model"
            )
            return True

        value = self.process_expression(value_expr)
        if value is None:
            self.refuse_lowering("SPIR-V clip operand could not be evaluated")
            return True

        condition = self.clip_discard_condition(value)
        if condition is None:
            self.refuse_lowering(
                "SPIR-V clip requires a numeric scalar or vector operand"
            )
            return True

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        kill_label = SpirvId(self.get_id(), SpirvType("label"))
        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, kill_label, merge_label)

        self.emit(f"%{kill_label.id} = OpLabel")
        self.current_label = kill_label.id
        self.emit("OpKill")

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id
        return True

    def clip_discard_condition(self, value: SpirvId) -> Optional[SpirvId]:
        """Return the scalar bool condition for HLSL clip discard semantics."""
        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        type_name = value_type.type.base_type
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name == "bool":
                return None
            component_type = self.register_primitive_type(component_type_name)
            zero = self.clip_zero_value(component_type_name, component_type)
            zero_vector = self.splat_scalar_to_vector(zero, value_type)
            bool_type = self.register_primitive_type("bool")
            bool_vector_type = self.register_vector_type(bool_type, component_count)
            condition_vector = self.binary_operation(
                "<", bool_vector_type, value, zero_vector
            )
            return self.any_bool_vector_operation(condition_vector)

        scalar_type_name = self.normalize_primitive_name(type_name)
        if scalar_type_name not in {"float", "double"} | self.INTEGER_TYPE_NAMES:
            return None

        scalar_type = self.register_primitive_type(scalar_type_name)
        value = self.convert_value_to_type(value, scalar_type)
        zero = self.clip_zero_value(scalar_type_name, scalar_type)
        bool_type = self.register_primitive_type("bool")
        return self.binary_operation("<", bool_type, value, zero)

    def clip_zero_value(self, type_name: str, type_id: SpirvId) -> SpirvId:
        """Return a zero constant compatible with a clip operand component."""
        if type_name in {"float", "double"}:
            return self.register_constant(0.0, type_id)
        return self.register_constant(0, type_id)

    def infer_expression_result_type(self, expr) -> Optional[SpirvId]:
        if expr is None:
            return None
        if isinstance(expr, bool):
            return self.register_primitive_type("bool")
        if isinstance(expr, int):
            return self.register_primitive_type("int")
        if isinstance(expr, float):
            return self.register_primitive_type("float")
        if isinstance(expr, LiteralNode):
            literal_type = self.convert_type_node_to_string(expr.literal_type)
            primitive_type = self.normalize_primitive_name(literal_type)
            if primitive_type in self.INTEGER_TYPE_NAMES:
                primitive_type = self.integer_literal_type_for_value(
                    primitive_type, int(expr.value)
                )
                return self.register_primitive_type(primitive_type)
            return self.map_crossgl_type(expr.literal_type)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            name = expr if isinstance(expr, str) else expr.name
            local_type_name = self.local_variable_types.get(name)
            if local_type_name is not None:
                return self.map_resource_type_with_format(local_type_name)
            variable = self.local_variables.get(name) or self.resolve_global_variable(
                name
            )
            if variable is not None:
                wrapped_type = self.uniform_block_wrapped_member_type(variable)
                if wrapped_type is not None:
                    return wrapped_type
                return self.variable_value_types.get(
                    variable.id
                ) or self.value_types.get(variable.id)
            named_constant = self.named_constants.get(name)
            if named_constant is not None:
                constant_type = self.value_types.get(named_constant.id)
                if constant_type is not None:
                    return constant_type
            cbuffer_member = self.cbuffer_members.get(name)
            if cbuffer_member is not None:
                return cbuffer_member[1]
            builtin_type = self.infer_builtin_expression_result_type(name)
            if builtin_type is not None:
                return builtin_type
            return None
        if isinstance(expr, FunctionCallNode):
            callee_name = self.function_call_name(expr)
            if callee_name is None:
                return None
            vector_info = self.vector_component_type_and_count(callee_name)
            if vector_info is not None:
                component_type, component_count = vector_info
                return self.register_vector_type(
                    self.register_primitive_type(component_type), component_count
                )
            matrix_type_name = self.normalize_hlsl_matrix_type(callee_name)
            if re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_type_name):
                return self.map_crossgl_type(matrix_type_name)
            primitive_name = self.normalize_primitive_name(callee_name)
            if primitive_name in {
                "bool",
                "int",
                "uint",
                "i64",
                "u64",
                "float",
                "double",
                self.BFLOAT16_TYPE_NAME,
            }:
                return self.register_primitive_type(primitive_name)
            specialized_callee_name = generic_function_call_name(
                self, callee_name, expr.args
            )
            if specialized_callee_name is not None:
                signature = self.resolve_function_signature(specialized_callee_name)
                if signature is not None:
                    return signature[0]
                for specialized_func in (
                    self.generic_function_specializations or {}
                ).values():
                    if (
                        getattr(specialized_func, "name", None)
                        == specialized_callee_name
                    ):
                        return self.map_crossgl_type(
                            getattr(specialized_func, "return_type", "void")
                        )
            declared_result_type = self.declared_concrete_call_result_type(expr)
            if declared_result_type is not None:
                return declared_result_type
            signature = self.resolve_function_signature(callee_name)
            if signature is not None:
                return signature[0]
            if callee_name in self.struct_types:
                return self.struct_types[callee_name]
            bfloat16_bitcast_type = self.infer_bfloat16_bitcast_result_type(callee_name)
            if bfloat16_bitcast_type is not None:
                return bfloat16_bitcast_type
            explicit_bitcast_type = self.explicit_bitcast_target_type(callee_name)
            if explicit_bitcast_type is not None:
                return explicit_bitcast_type
            bitcast_type = self.infer_bitcast_builtin_result_type(
                callee_name, expr.args
            )
            if bitcast_type is not None:
                return bitcast_type
            integer_bit_type = self.infer_integer_bit_builtin_result_type(
                callee_name, expr.args
            )
            if integer_bit_type is not None:
                return integer_bit_type
            if callee_name in BITFIELD_ARITIES and expr.args:
                return self.infer_expression_result_type(expr.args[0])
            return spirv_pack_result_type(self, callee_name)
        if isinstance(expr, ArrayAccessNode):
            buffer_element_type = storage_buffer_subscript_element_type_name(self, expr)
            if buffer_element_type is not None:
                return self.map_crossgl_type(buffer_element_type)
            array_type = self.infer_expression_result_type(expr.array)
            if array_type is None:
                return None
            element_type = self.array_element_type_from_type(array_type)
            if element_type is not None:
                return element_type
            vector_info = self.vector_type_info_from_type(array_type)
            if vector_info is not None:
                return vector_info[0]
            matrix_info = self.matrix_type_info_from_type(array_type)
            if matrix_info is not None:
                return matrix_info[0]
            return None
        if isinstance(expr, MemberAccessNode):
            base_type = self.infer_expression_result_type(expr.object)
            if base_type is None:
                return None
            member_info = self.struct_member_info(base_type.type.base_type, expr.member)
            if member_info is not None:
                return member_info[1]
            vector_member = self.vector_member_info(
                base_type.type.base_type, expr.member
            )
            if vector_member is not None:
                return vector_member[1]
            swizzle_info = self.vector_swizzle_info(
                base_type.type.base_type, expr.member
            )
            if swizzle_info is not None:
                return swizzle_info[2]
            return None
        if isinstance(expr, AssignmentNode):
            target = getattr(
                expr, "name", getattr(expr, "target", getattr(expr, "left", None))
            )
            return self.infer_expression_result_type(
                target
            ) or self.infer_expression_result_type(expr.value)
        if isinstance(expr, BinaryOpNode):
            if expr.op == ",":
                return self.infer_expression_result_type(expr.right)
            left_type = self.infer_expression_result_type(expr.left)
            right_type = self.infer_expression_result_type(expr.right)
            return self.binary_expression_result_type(expr.op, left_type, right_type)
        if isinstance(expr, TernaryOpNode):
            true_type = self.infer_expression_result_type(expr.true_expr)
            false_type = self.infer_expression_result_type(expr.false_expr)
            if true_type is not None and false_type is not None:
                return self.ternary_result_type(
                    SpirvId(0, true_type.type), SpirvId(0, false_type.type)
                )
            return true_type or false_type
        if isinstance(expr, MatchNode):
            return self.infer_match_expression_result_type(expr)
        if isinstance(expr, WaveOpNode):
            if expr.operation == "WaveShuffleAndFillUp" and expr.arguments:
                return self.infer_expression_result_type(expr.arguments[0])
            return None
        return None

    def infer_builtin_expression_result_type(self, name: str) -> Optional[SpirvId]:
        builtin_type = self.infer_builtin_variable_result_type(name)
        if builtin_type is not None:
            return builtin_type

        if not isinstance(name, str) or "." not in name:
            return None

        base_name, member_name = name.rsplit(".", 1)
        base_type = self.infer_builtin_variable_result_type(base_name)
        if base_type is None:
            return None

        member_info = self.vector_member_info(base_type.type.base_type, member_name)
        if member_info is not None:
            return member_info[1]

        swizzle_info = self.vector_swizzle_info(base_type.type.base_type, member_name)
        if swizzle_info is not None:
            return swizzle_info[2]

        return None

    def infer_builtin_variable_result_type(self, name: str) -> Optional[SpirvId]:
        compute_info = self.compute_builtin_info(name)
        if compute_info is not None:
            return self.map_crossgl_type(compute_info[0])

        stage_info = self.stage_builtin_info(name)
        if stage_info is not None:
            return self.map_crossgl_type(stage_info[0])

        return None

    def glsl_builtin_limit_constant(self, name: str) -> Optional[SpirvId]:
        value = GLSL_BUILTIN_INT_LIMITS.get(name)
        if value is None:
            value = shading_rate_flag_value(name)
        if value is None:
            return None
        return self.register_constant(value, self.register_primitive_type("int"))

    def infer_match_expression_result_type(self, node: MatchNode) -> Optional[SpirvId]:
        for arm in getattr(node, "arms", []) or []:
            expression = self.match_arm_tail_expression(getattr(arm, "body", None))
            result_type = self.infer_expression_result_type(expression)
            if result_type is not None:
                return result_type
        return None

    def match_arm_tail_expression(self, body):
        statements = getattr(body, "statements", None)
        if statements is not None:
            if statements and getattr(statements[-1], "is_tail_expression", False):
                return getattr(statements[-1], "expression", None)
            return None
        if hasattr(body, "expression"):
            return getattr(body, "expression", None)
        return body

    def process_variable_declaration(self, node: VariableNode):
        """Process a local CrossGL variable declaration."""
        # spirv_single_evaluation.py: the alias probes and the value path share one index evaluation.
        with subscript_index_scope(self):
            return self.process_variable_declaration_body(node)

    def process_variable_declaration_body(self, node: VariableNode):
        var_type_source = self.current_cooperative_matrix_role_overrides.get(
            id(node), getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        storage_class = self.local_variable_storage_class(node)
        self.current_function_literal_int_constant_shadows.add(node.name)
        self.current_function_literal_int_constants.pop(node.name, None)
        self.register_function_local_int_constant(node, var_type_source)
        if self.type_name_from_value(var_type_source) == "auto":
            var_type_source = None
        var_type_name = self.type_name_from_value(var_type_source)
        infer_declared_type = var_type_source is None or var_type_name == "auto"
        # Alias probes build the initializer's access chain; the value path below reuses its index.
        with recording_subscript_indices(self):
            if self.process_local_resource_alias_declaration(
                node, var_type_source, var_type_name
            ) or self.process_local_array_pointer_alias_declaration(
                node, var_type_source, var_type_name
            ):
                return

        if self.local_variable_requires_descriptor_storage(node, var_type_name):
            raise VulkanRefusal(
                f"SPIR-V descriptor resource '{node.name}' cannot be declared "
                "inside a function; declare it at shader or stage scope"
            )

        unresolved_extent = self.unresolved_fixed_array_extent(var_type_source)
        if unresolved_extent is not None:
            raise UnsupportedSPIRVFeatureError(
                "unresolved fixed array extent",
                "SPIR-V requires a concrete fixed-array extent for local variable "
                f"{node.name}; could not resolve '{unresolved_extent}'",
                missing_capabilities=("spirv.fixed_array_extent",),
                source_location=getattr(node, "source_location", None),
            )

        initial_value = getattr(node, "initial_value", None)
        is_precise = (
            declaration_is_precise(node) or node.name in self.precise_flow_names
        )
        if (
            infer_declared_type
            and initial_value is not None
            and not isinstance(initial_value, (ArrayLiteralNode, MatchNode))
        ):
            rhs_value = self.process_expression_with_precision(
                initial_value, is_precise
            )
            if rhs_value is not None:
                var_type = self.ensure_registered_type(rhs_value.type)
                var_id = self.create_variable(var_type, "Function", node.name)
                self.local_variables[node.name] = var_id
                if is_precise:
                    self.precise_local_variables.add(node.name)
                self.store_to_variable(var_id, rhs_value)
                return

        if infer_declared_type and isinstance(initial_value, MatchNode):
            var_type = self.infer_match_expression_result_type(initial_value)
            if var_type is None:
                var_type = self.map_resource_type_with_format(var_type_source, node)
        else:
            var_type = self.map_resource_type_with_format(var_type_source, node)
        if self.type_contains_runtime_array(var_type):
            self.refuse_lowering(
                f"local variable {node.name} has a runtime-array "
                "aggregate type that cannot be materialized in SPIR-V"
            )
            return

        if storage_class == "Workgroup":
            var_id = self.get_or_create_workgroup_variable(node, var_type)
            self.local_variables[node.name] = var_id
            if is_precise:
                self.precise_local_variables.add(node.name)
            if initial_value is not None:
                if isinstance(initial_value, MatchNode):
                    self.process_match_expression_assignment(
                        initial_value, var_id, var_type
                    )
                elif isinstance(initial_value, ArrayLiteralNode):
                    rhs_value = self.process_array_literal(initial_value, var_type)
                    if rhs_value is not None:
                        self.store_to_variable(var_id, rhs_value)
                else:
                    rhs_value = (
                        self.process_expression_with_expected_type_and_precision(
                            initial_value, var_type_name, is_precise
                        )
                    )
                    if rhs_value is not None:
                        self.store_to_variable(var_id, rhs_value)
            return

        var_id = self.create_variable(var_type, "Function", node.name)
        self.local_variables[node.name] = var_id
        if is_precise:
            self.precise_local_variables.add(node.name)

        if initial_value is not None:
            if isinstance(initial_value, MatchNode):
                self.process_match_expression_assignment(
                    initial_value, var_id, var_type
                )
            elif isinstance(initial_value, ArrayLiteralNode):
                rhs_value = self.process_array_literal(initial_value, var_type)
                if rhs_value is not None:
                    self.store_to_variable(var_id, rhs_value)
            else:
                rhs_value = self.process_expression_with_expected_type_and_precision(
                    initial_value, var_type_name, is_precise
                )
                if rhs_value is not None:
                    self.store_to_variable(var_id, rhs_value)

    def register_function_local_int_constant(self, node, var_type_source):
        """Register a visible local integer constant at its declaration point."""
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if not qualifiers & {"const", "constexpr", "constant"}:
            return

        declared_type = self.normalize_primitive_name(
            self.type_name_from_value(var_type_source)
        )
        if declared_type not in self.INTEGER_TYPE_NAMES:
            return

        value = evaluate_literal_int_expression(
            getattr(node, "initial_value", None),
            self.active_literal_int_constants(),
        )
        if value is not None:
            self.current_function_literal_int_constants[node.name] = int(value)

    def local_variable_storage_class(self, node: VariableNode) -> str:
        metadata = self.declaration_metadata_names(node)
        if not metadata & {"groupshared", "shared", "threadgroup", "workgroup"}:
            return "Function"

        execution_models = self.current_synchronization_execution_models()
        allowed_models = {"GLCompute", "MeshEXT", "TaskEXT"}
        if execution_models and not execution_models.issubset(allowed_models):
            label = self.current_execution_model_label()
            raise UnsupportedSPIRVFeatureError(
                "workgroup storage execution model",
                "SPIR-V workgroup storage is only available to compute, mesh, "
                f"and task execution models; '{node.name}' is reachable from {label}",
                missing_capabilities=("spirv.workgroup_storage.execution_model",),
                source_location=getattr(node, "source_location", None),
            )
        return "Workgroup"

    def workgroup_variable_cache_key(
        self,
        node: VariableNode,
        var_type: SpirvId,
    ) -> tuple:
        type_bindings = tuple(
            sorted(
                (str(name), str(value))
                for name, value in self.current_generic_type_substitutions.items()
            )
        )
        value_bindings = tuple(
            sorted(
                (str(name), str(value))
                for name, value in self.current_generic_function_substitutions.items()
            )
        )
        return id(node), var_type.id, type_bindings, value_bindings

    def get_or_create_workgroup_variable(
        self,
        node: VariableNode,
        var_type: SpirvId,
    ) -> SpirvId:
        cache_key = self.workgroup_variable_cache_key(node, var_type)
        variable = self.workgroup_variables_by_declaration.get(cache_key)
        if variable is None:
            # task_payload_variables.py: a task entry's DispatchMesh payload is task-payload storage.
            payload = (
                self.current_execution_model == "TaskEXT"
                and node.name in dispatch_mesh_payload_names(
                    getattr(self.current_function_node, "body", None)
                )
            )
            storage_class = "TaskPayloadWorkgroupEXT" if payload else "Workgroup"
            variable = self.create_variable(var_type, storage_class, node.name)
            if payload:
                self.require_capability("MeshShadingEXT")
                self.require_extension("SPV_EXT_mesh_shader")
                self.task_payload_shared_variables[node.name] = variable
            self.workgroup_variables_by_declaration[cache_key] = variable
        self.mark_function_interface_variable(variable)
        return variable

    def process_local_resource_alias_declaration(
        self, node: VariableNode, var_type_source, var_type_name: str
    ) -> bool:
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return False

        source_pointer = self.variable_pointer_from_expression(initial_value)
        if source_pointer is None:
            return False

        source_metadata = self.resource_metadata_for_pointer(source_pointer)
        source_storage_buffer_metadata = (
            self.storage_buffer_access_metadata_for_pointer(source_pointer)
        )
        if source_metadata is None and source_storage_buffer_metadata is None:
            return False

        referenced_type_name = (
            self.referenced_value_type_name(var_type_name)
            if var_type_source is not None
            else None
        )
        if referenced_type_name is not None and isinstance(
            initial_value, (ArrayAccessNode, MemberAccessNode)
        ):
            self.bind_local_storage_element_reference(
                node.name, source_pointer, referenced_type_name
            )
            return True

        declared_alias_type_name = self.storage_buffer_resource_type_name(var_type_name)
        inferred_alias_type_name = self.storage_buffer_expression_type_name(
            initial_value
        )
        if self.storage_buffer_type_is_inferred_pointer(var_type_name):
            alias_type_name = inferred_alias_type_name or declared_alias_type_name
        else:
            alias_type_name = declared_alias_type_name or inferred_alias_type_name
        if var_type_source is not None:
            # A declared value type (``int s = a[i]``, ``S s = b[i]``) copies
            # the element; it must never alias the source storage.
            if self.declared_type_name_is_value(var_type_name):
                return False
            base_type_name = self.array_base_type_name(var_type_name)
            if not (
                self.is_resource_type_name(base_type_name)
                or self.is_acceleration_structure_type_name(base_type_name)
                or alias_type_name is not None
            ):
                return False

        alias_pointer = self.copy_storage_buffer_pointer_alias(
            source_pointer,
            node.name,
        )
        if isinstance(initial_value, PointerReinterpretNode):
            reinterpretation = self.storage_pointer_reinterpretation_metadata(
                initial_value,
                source_pointer,
                source_storage_buffer_metadata or source_metadata,
            )
            self.storage_pointer_reinterpretations[alias_pointer] = reinterpretation
            self.storage_buffer_pointer_offset_aliases[
                self.storage_buffer_pointer_alias_key(alias_pointer)
            ] = PointerOffsetState(
                reinterpretation["root_pointer"],
                None,
                "storage_buffer",
            )
        self.local_variables[node.name] = alias_pointer
        if alias_type_name is not None:
            self.local_variable_types[node.name] = alias_type_name
        self.resource_alias_variables.add(node.name)
        return True

    def storage_pointer_reinterpretation_metadata(
        self, expression, source_pointer, source_metadata
    ):
        existing = self.storage_pointer_reinterpretations.get(source_pointer)
        source_type_name = (
            existing["source_type"]
            if existing is not None
            else source_metadata.get("element_type_name")
            or source_metadata.get("element_type").type.base_type
        )
        view_type_name = (
            existing["target_type"] if existing is not None else source_type_name
        )
        target_type = getattr(expression.target_type, "pointee_type", None)
        target_type_name = self.type_name_from_value(target_type)
        source_layout = scalar_storage_layout(source_type_name)
        view_layout = scalar_storage_layout(view_type_name)
        target_layout = scalar_storage_layout(target_type_name)
        if (
            source_layout is None
            or view_layout is None
            or target_layout is None
            or source_layout.bit_width != 32
            or target_layout.bit_width not in {8, 16, 32}
            or target_layout.kind == "floating"
            and target_layout.bit_width != 32
        ):
            raise PointerReinterpretationError(
                "SPIR-V storage pointer reinterpretation requires a 32-bit "
                "scalar backing element and an 8-, 16-, or 32-bit scalar view",
                source_type=source_type_name,
                target_type=target_type_name,
                address_space="storage",
                alignment=target_layout.byte_width if target_layout else None,
                access="read" if source_metadata.get("readonly") else "read_write",
                reason="unsupported-scalar-layout",
                source_location=getattr(expression, "source_location", None),
            )

        uint_type = self.register_primitive_type("uint")
        offset_state = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(source_pointer)
        )
        root_pointer = (
            offset_state.root_pointer if offset_state is not None else source_pointer
        )
        element_offset = self.register_constant(0, uint_type)
        if offset_state is not None and offset_state.offset_variable is not None:
            offset_type = self.variable_value_types[offset_state.offset_variable.id]
            element_offset = self.load_from_variable(
                offset_state.offset_variable, offset_type
            )
            element_offset = self.convert_value_to_type(element_offset, uint_type)
        if view_layout.byte_width != 1:
            view_bytes = self.register_constant(view_layout.byte_width, uint_type)
            element_offset = self.binary_operation(
                "*", uint_type, element_offset, view_bytes
            )
        existing_byte_offset = (
            existing.get("byte_offset") if existing is not None else None
        )
        if existing_byte_offset is not None:
            element_offset = self.binary_operation(
                "+", uint_type, existing_byte_offset, element_offset
            )

        return {
            "source_type": source_type_name,
            "target_type": target_type_name,
            "source_layout": source_layout,
            "view_layout": view_layout,
            "target_layout": target_layout,
            "target_type_id": self.map_crossgl_type(target_type_name),
            "access": "read" if source_metadata.get("readonly") else "read_write",
            "byte_offset": element_offset,
            "root_pointer": root_pointer,
        }

    def process_local_array_pointer_alias_declaration(
        self, node: VariableNode, var_type_source, var_type_name: str
    ) -> bool:
        # `threadgroup/device T* p = &arr[i]` (or `= q`, where q already aliases a
        # local-array element): bind `p` to the element pointer so `p`, `p[j]`
        # and `p + j` resolve to `arr[i + j]` rather than squashing `p` to a
        # scalar Function variable. Only takes over when the initializer is an
        # offset-aliased local-array element pointer; resource / storage-buffer
        # pointer aliases keep their existing dedicated path, and any other shape
        # falls through to the ordinary variable handling unchanged.
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return False
        if self.pointer_pointee_type_name_from_string(var_type_name) is None:
            return False
        source_pointer = self.variable_pointer_from_expression(initial_value)
        if source_pointer is None:
            return False
        if (
            self.storage_buffer_pointer_alias_key(source_pointer)
            not in self.storage_buffer_pointer_offset_aliases
        ):
            return False
        alias_pointer = self.copy_storage_buffer_pointer_alias(
            source_pointer,
            node.name,
        )
        self.local_variables[node.name] = alias_pointer
        self.resource_alias_variables.add(node.name)
        return True

    def copy_storage_buffer_pointer_alias(
        self, source_pointer: SpirvId, name: Optional[str]
    ) -> SpirvId:
        """Create independent mutable offset state for a pointer value copy."""
        alias_pointer = SpirvId(source_pointer.id, source_pointer.type, name)
        source_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(source_pointer)
        )
        if source_alias is None or source_alias.offset_variable is None:
            root_pointer = (
                source_alias.root_pointer
                if source_alias is not None
                else source_pointer
            )
            domain = (
                source_alias.domain
                if source_alias is not None
                else self.storage_buffer_pointer_alias_domain(root_pointer)
            )
            reset_marker = self.next_pointer_offset_reset_marker()
            self.emit(reset_marker)
            self.storage_buffer_pointer_offset_aliases[
                self.storage_buffer_pointer_alias_key(alias_pointer)
            ] = PointerOffsetState(
                root_pointer,
                None,
                domain,
                reset_marker,
            )
            reinterpretation = self.storage_pointer_reinterpretations.get(
                source_pointer
            )
            if reinterpretation is not None:
                self.storage_pointer_reinterpretations[alias_pointer] = dict(
                    reinterpretation
                )
            return alias_pointer

        root_pointer = source_alias.root_pointer
        source_offset_variable = source_alias.offset_variable
        offset_type = self.variable_value_types[source_offset_variable.id]
        source_offset = self.load_from_variable(source_offset_variable, offset_type)
        zero = self.register_constant(0, offset_type)
        alias_offset_variable = self.create_variable(
            offset_type,
            "Function",
            initializer=zero,
        )
        self.store_to_variable(alias_offset_variable, source_offset)
        self.storage_buffer_pointer_offset_aliases[
            self.storage_buffer_pointer_alias_key(alias_pointer)
        ] = PointerOffsetState(
            root_pointer,
            alias_offset_variable,
            source_alias.domain,
        )
        reinterpretation = self.storage_pointer_reinterpretations.get(source_pointer)
        if reinterpretation is not None:
            self.storage_pointer_reinterpretations[alias_pointer] = dict(
                reinterpretation
            )
        return alias_pointer

    def storage_buffer_pointer_alias_domain(self, pointer: SpirvId) -> str:
        """Classify a pointer root for offset-aware element access."""
        if self.pointer_points_to_indexable_array(pointer):
            return "local_array"
        return "storage_buffer"

    def next_pointer_offset_reset_marker(self) -> str:
        """Reserve a source-order location for a deferred pointer-copy reset."""
        self.pointer_offset_reset_marker_count += 1
        return (
            f"{self.POINTER_OFFSET_RESET_MARKER_PREFIX}"
            f"{self.pointer_offset_reset_marker_count}__"
        )

    def materialize_storage_buffer_pointer_offset_alias(
        self,
        pointer: SpirvId,
        state: PointerOffsetState,
        offset_type: SpirvId,
    ) -> PointerOffsetState:
        """Create a typed offset cell and place any deferred reset at its copy."""
        if state.offset_variable is not None:
            return state

        zero = self.register_constant(0, offset_type)
        offset_variable = self.create_variable(
            offset_type,
            "Function",
            initializer=zero,
        )
        if state.reset_marker is not None:
            try:
                marker_index = self.code_lines.index(state.reset_marker)
            except ValueError as exc:
                raise AssertionError(
                    "Deferred pointer-offset reset marker is missing"
                ) from exc
            self.code_lines[marker_index] = f"OpStore %{offset_variable.id} %{zero.id}"

        materialized = PointerOffsetState(
            state.root_pointer,
            offset_variable,
            state.domain,
        )
        self.storage_buffer_pointer_offset_aliases[
            self.storage_buffer_pointer_alias_key(pointer)
        ] = materialized
        return materialized

    def local_variable_requires_descriptor_storage(
        self, node: VariableNode, type_name: str
    ) -> bool:
        if not type_name:
            return False
        base_type = self.array_base_type_name(type_name)
        return (
            self.is_resource_type_name(base_type)
            or self.is_acceleration_structure_type_name(base_type)
            or self.is_structured_buffer_type_name(base_type)
            or self.is_glsl_buffer_block_node(node)
        )
