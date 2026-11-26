"""CrossGL-to-Vulkan SPIR-V code generator."""

import re
from typing import List, Optional, Union

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    CooperativeMatrixOpNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..unhandled import UNHANDLED
from .array_utils import evaluate_literal_int_expression
from .comma_expressions import is_comma_expression, spirv_comma_expression
from .spirv_access_chains import SpirvAccessChainMixin
from .spirv_assignments import SpirvAssignmentMixin
from .spirv_bfloat16_math import SpirvBfloat16MathMixin
from .spirv_binary_operations import SpirvBinaryOperationMixin
from .spirv_boolean_reductions import BOOLEAN_REDUCTIONS, spirv_boolean_reduction
from .spirv_buffer_layouts import SpirvBufferLayoutMixin
from .spirv_buffer_methods import SpirvBufferMethodMixin
from .spirv_builtin_calls import SpirvBuiltinCallMixin
from .spirv_builtin_families import SpirvBuiltinFamilyMixin
from .spirv_call_expressions import SpirvCallExpressionMixin
from .spirv_control_flow import SpirvControlFlowMixin
from .spirv_cooperative_matrices import SpirvCooperativeMatrixMixin
from .spirv_default_uniforms import SpirvDefaultUniformMixin
from .spirv_entry_locations import (
    interface_location_state,
    restore_interface_location_state,
)
from .spirv_entry_point_parameters import SpirvEntryPointParameterMixin
from .spirv_entry_selection import SpirvEntrySelectionMixin
from .spirv_enums_and_structs import SpirvEnumStructMixin
from .spirv_exact_lookups import spirv_builtin_call_key
from .spirv_expression_kinds import SpirvExpressionKindMixin
from .spirv_extension_builtins import BUILTIN_ALIASES as SPIRV_EXTENSION_BUILTIN_ALIASES
from .spirv_float_operand_widths import widen_float_operands
from .spirv_float_semantics import SpirvFloatSemanticsMixin
from .spirv_function_parameters import (
    SpirvFunctionParameterMixin,
)
from .spirv_function_phases import SpirvFunctionPhaseMixin
from .spirv_function_prepass import SpirvFunctionPrepassMixin
from .spirv_global_declarations import SpirvGlobalDeclarationMixin
from .spirv_group_operations import SpirvGroupOperationMixin
from .spirv_inline_requirements import apply_inline_spirv_requirements
from .spirv_local_declarations import SpirvLocalDeclarationMixin
from .spirv_local_references import SpirvLocalReferenceMixin
from .spirv_math_polyfills import SpirvMathPolyfillMixin
from .spirv_matrix_builtins import SpirvMatrixBuiltinMixin
from .spirv_mesh_outputs import SpirvMeshOutputMixin
from .spirv_module_assembly import SpirvModuleAssemblyMixin
from .spirv_module_phases import SpirvModulePhaseMixin
from .spirv_pointer_parameters import SpirvPointerParameterMixin
from .spirv_ray_tracing import SpirvRayTracingMixin
from .spirv_resource_calls import SpirvResourceCallMixin
from .spirv_sampling_calls import SpirvSamplingCallMixin
from .spirv_single_evaluation import (
    recording_subscript_indices,
    subscript_index_scope,
)
from .spirv_stage_builtins import SpirvStageBuiltinMixin
from .spirv_storage_buffer_access import SpirvStorageBufferAccessMixin
from .spirv_texture_calls import SpirvTextureCallMixin
from .spirv_types_and_constants import SpirvTypeConstantMixin
from .spirv_values import (  # noqa: F401  (also imported from this module)
    CALLEE_IDENTIFIER,
    INDIRECT_CALL_DETAIL,
    InlineReturnContext,
    PointerOffsetState,
    SPIRVAtomicFenceLoweringError,
    SPIRVBuiltinLoweringError,
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class VulkanSPIRVCodeGen(
    SpirvTypeConstantMixin,
    SpirvAccessChainMixin,
    SpirvBinaryOperationMixin,
    SpirvTextureCallMixin,
    SpirvRayTracingMixin,
    SpirvMeshOutputMixin,
    SpirvGroupOperationMixin,
    SpirvBuiltinCallMixin,
    SpirvEntryPointParameterMixin,
    SpirvCooperativeMatrixMixin,
    SpirvEnumStructMixin,
    SpirvBufferLayoutMixin,
    SpirvLocalDeclarationMixin,
    SpirvGlobalDeclarationMixin,
    SpirvFunctionPrepassMixin,
    SpirvStorageBufferAccessMixin,
    SpirvPointerParameterMixin,
    SpirvBufferMethodMixin,
    SpirvAssignmentMixin,
    SpirvControlFlowMixin,
    SpirvStageBuiltinMixin,
    SpirvModuleAssemblyMixin,
    SpirvBfloat16MathMixin,
    SpirvDefaultUniformMixin,
    SpirvFloatSemanticsMixin,
    SpirvLocalReferenceMixin,
    SpirvMathPolyfillMixin,
    SpirvMatrixBuiltinMixin,
    SpirvResourceCallMixin,
    SpirvSamplingCallMixin,
    SpirvBuiltinFamilyMixin,
    SpirvExpressionKindMixin,
    SpirvCallExpressionMixin,
    SpirvFunctionPhaseMixin,
    SpirvFunctionParameterMixin,
    SpirvModulePhaseMixin,
    SpirvEntrySelectionMixin,
):
    """Generates SPIR-V code from a CrossGL shader AST."""

    POINTER_OFFSET_RESET_MARKER_PREFIX = "; __crosstl_pointer_offset_reset_"
    COOPERATIVE_MATRIX_KHR_PROFILE = "vulkan-khr-cooperative-matrix"
    # Execution models whose extensions (SPV_EXT_mesh_shader, SPV_KHR_ray_tracing) need a SPIR-V 1.4 module.
    SPIRV_1_4_EXECUTION_MODELS = frozenset(
        {
            "MeshEXT",
            "TaskEXT",
            "RayGenerationKHR",
            "IntersectionKHR",
            "AnyHitKHR",
            "ClosestHitKHR",
            "MissKHR",
            "CallableKHR",
        }
    )
    COOPERATIVE_MATRIX_SCOPE_VALUES = {"subgroup": 3}
    COOPERATIVE_MATRIX_USE_VALUES = {
        "matrix_a": 0,
        "matrixa": 0,
        "matrix_a_khr": 0,
        "matrixakhr": 0,
        "multiplicand_a": 0,
        "matrix_b": 1,
        "matrixb": 1,
        "matrix_b_khr": 1,
        "matrixbkhr": 1,
        "multiplicand_b": 1,
        "accumulator": 2,
        "matrix_accumulator": 2,
        "matrix_accumulator_khr": 2,
        "matrixaccumulatorkhr": 2,
        "matrix_c": 2,
        "result": 2,
        "matrix_result": 2,
    }
    COOPERATIVE_MATRIX_USE_NAMES = {
        0: "matrix_a",
        1: "matrix_b",
        2: "accumulator",
    }
    ATOMIC_FENCE_MEMORY_FLAGS = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    ATOMIC_FENCE_MEMORY_ORDERS = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    ATOMIC_FENCE_THREAD_SCOPES = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )

    SIGNED_INT32_MIN = -(1 << 31)
    SIGNED_INT32_MAX = (1 << 31) - 1
    UNSIGNED_INT32_MAX = (1 << 32) - 1
    SIGNED_INT64_MIN = -(1 << 63)
    SIGNED_INT64_MAX = (1 << 63) - 1
    UNSIGNED_INT64_MAX = (1 << 64) - 1
    INTEGER_TYPE_WIDTHS = {
        "int": 32,
        "uint": 32,
        "i64": 64,
        "u64": 64,
    }
    SIGNED_INTEGER_TYPES = {"int", "i64"}
    UNSIGNED_INTEGER_TYPES = {"uint", "u64"}
    INTEGER_TYPE_NAMES = SIGNED_INTEGER_TYPES | UNSIGNED_INTEGER_TYPES
    BFLOAT16_TYPE_NAME = "bfloat16"
    # Internal binary operator for GLSL's floored `mod` (OpFMod/OpSMod); the
    # source-level `%` operator is the truncated remainder.
    FLOORED_MOD_OPERATOR = "FLOORED_MOD"

    def __init__(
        self,
        *,
        include_resource_interface_variables: bool = False,
        target_profile: Optional[str] = None,
    ):
        """Initialize an empty SPIR-V module-generation state."""
        self.include_resource_interface_variables = include_resource_interface_variables
        self.target_profile = self.normalize_target_profile(target_profile)
        self.cooperative_matrix_khr_enabled = (
            self.target_profile == self.COOPERATIVE_MATRIX_KHR_PROFILE
        )
        self.reset_generation_state()

    def refuse_lowering(self, reason: str):
        """Record a construct with no faithful SPIR-V lowering; :meth:`generate` raises once the module is done.

        The caller may still return a placeholder so that generation can reach every other refusal in the
        module, but no module is returned while a refusal is pending.
        """
        self.refusals.record(reason)

    def unary_operation(
        self, op: str, result_type: Union[SpirvId, SpirvType], operand: SpirvId
    ) -> SpirvId:
        """Create a unary operation."""
        result_type = self.ensure_registered_type(result_type)

        if op == "+":
            spv_op = None
        elif op == "-":
            # Unary minus on a complex-number struct: SPIR-V's OpFNegate/OpSNegate
            # only accept scalar or vector numeric types, so negating a complex
            # struct (e.g. MLX's `Negative` functor `return -x;` instantiated for
            # complex64_t) must negate the real and imaginary members separately
            # and rebuild the struct rather than emit an invalid struct-typed
            # OpFNegate.
            complex_negation = self.complex_struct_negate_operation(operand)
            if complex_negation is not None:
                return complex_negation
            # OpFNegate takes no matrix, so a matrix is negated one column at a time.
            matrix_info = self.matrix_operand_info(self.registered_value_type(operand))
            if matrix_info is not None:
                columns = [
                    self.unary_operation(
                        "-",
                        matrix_info["column_type"],
                        self.composite_extract(
                            operand, matrix_info["column_type"], index
                        ),
                    )
                    for index in range(matrix_info["column_count"])
                ]
                return self.composite_construct(matrix_info["type"], columns)
            component_type = self.scalar_or_vector_component_type(result_type.type)
            # Unary minus on a bool result: SPIR-V has no negation opcode for
            # bool (OpFNegate/OpSNegate reject bool), and C++ promotes bool to int
            # for unary minus (e.g. MLX's `Negative` functor `return -x;`
            # instantiated for bool). Negate on a promoted integer type and
            # convert the result back to bool.
            bool_integer_type = self.bool_arithmetic_integer_type(result_type)
            if bool_integer_type is not None:
                integer_operand = self.convert_value_to_type(operand, bool_integer_type)
                negated = self.unary_operation("-", bool_integer_type, integer_operand)
                return self.convert_value_to_type(negated, result_type)
            if component_type == self.BFLOAT16_TYPE_NAME:
                vector_info = self.vector_component_type_and_count(
                    result_type.type.base_type
                )
                float_type = self.register_primitive_type("float")
                promoted_type = (
                    self.register_vector_type(float_type, vector_info[1])
                    if vector_info is not None
                    else float_type
                )
                promoted_operand = self.convert_value_to_type(operand, promoted_type)
                negated = self.unary_operation("-", promoted_type, promoted_operand)
                return self.convert_value_to_type(negated, result_type)
            # Unsigned integers of any width (uint, u64) have no signed-negate
            # opcode, so negate as `0 - x` (two's-complement wrap). Signed
            # integers of any width (int, i64) use OpSNegate; only genuine
            # floating types use OpFNegate. Matching against the width-agnostic
            # type sets keeps 64-bit operands (e.g. MLX's `Negative` functor for
            # int64/uint64) from falling through to an invalid `OpFNegate`.
            if component_type in self.UNSIGNED_INTEGER_TYPES:
                zero = self.default_value_for_type(result_type)
                return self.binary_operation("-", result_type, zero, operand)
            spv_op = (
                "OpSNegate"
                if component_type in self.SIGNED_INTEGER_TYPES
                else "OpFNegate"
            )
        elif op == "!":
            bool_type = self.register_primitive_type("bool")
            operand_vector = self.vector_component_type_and_count(
                operand.type.base_type
            )
            if operand_vector is not None:
                _, component_count = operand_vector
                result_type = self.register_vector_type(bool_type, component_count)
                operand = self.convert_value_to_type(operand, result_type)
                if not self.value_has_type(operand, result_type):
                    self.refuse_lowering(
                        "logical not requires a bool scalar or vector " "operand"
                    )
                    return self.default_value_for_type(result_type)
            else:
                operand = self.ensure_bool_value(operand)
                result_type = bool_type
            spv_op = "OpLogicalNot"
        else:
            spv_op = {
                "~": "OpNot",
            }.get(op)

        if spv_op is None:
            return operand

        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{operand.id}")
        self.decorate_no_contraction_result(id_value, spv_op, result_type)

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_resource_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        handled = self.spirv_delegated_resource_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_buffer_counter_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_buffer_element_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_storage_image_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_implicit_lod_sample_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_depth_compare_sample_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_offset_sample_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_gather_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_texel_fetch_and_size_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_sample_count_and_level_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_query_lod_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_explicit_lod_sample_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        return None

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        function_name = spirv_builtin_call_key(function_name)
        args = widen_float_operands(self, function_name, args)
        handled = self.spirv_delegated_builtin_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_scalar_builtin_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_geometric_builtin_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.spirv_vector_and_struct_constructor_call(function_name, args)
        if handled is not UNHANDLED:
            return handled

        matrix_function_name = self.normalize_hlsl_matrix_type(function_name)

        # Matrix constructors
        if re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name):
            return self.spirv_matrix_constructor_call(
                function_name, args, matrix_function_name
            )

        # Special case for dot product - use OpDot instead of OpExtInst
        elif function_name == "dot" and len(args) == 2:
            if not self.dot_operands_are_valid(args):
                raise self.builtin_lowering_error(
                    function_name,
                    "dot requires matching floating-point vector operands",
                )

            component_type = self.scalar_or_vector_component_type(args[0].type)
            result_type = self.register_primitive_type(component_type)

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpDot %{result_type.id} %{args[0].id} %{args[1].id}"
            )
            self.decorate_no_contraction_result(id_value, "OpDot", result_type)

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        elif function_name in BOOLEAN_REDUCTIONS and len(args) == 1:
            return spirv_boolean_reduction(self, function_name, args[0])

        elif function_name in {"fmod", "mod"} and len(args) == 2:
            result_type = self.ensure_registered_type(args[0].type)
            # `fmod` is C's truncated remainder (sign of x); `mod` is GLSL's
            # floored x - y * floor(x / y) (sign of y).
            operator = "%" if function_name == "fmod" else self.FLOORED_MOD_OPERATOR
            return self.binary_operation(operator, result_type, args[0], args[1])

        # GLSL standard library functions
        else:
            return self.spirv_glsl_std450_call(function_name, args)

    def process_function_node(self, function_node, stage=None):
        """Process a CrossGL function definition."""
        previous_cooperative_matrix_control_state = (
            self.cooperative_matrix_selection_depth,
            self.cooperative_matrix_loop_control_depth,
            self.cooperative_matrix_control_flow_diverged,
        )
        self.cooperative_matrix_selection_depth = 0
        self.cooperative_matrix_loop_control_depth = 0
        self.cooperative_matrix_control_flow_diverged = False
        previous_cooperative_matrix_role_overrides = (
            self.current_cooperative_matrix_role_overrides
        )
        previous_storage_buffer_pointer_offset_aliases = (
            self.storage_buffer_pointer_offset_aliases
        )
        self.storage_buffer_pointer_offset_aliases = {}
        previous_storage_pointer_reinterpretations = (
            self.storage_pointer_reinterpretations
        )
        self.storage_pointer_reinterpretations = {}
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        self.current_generic_function_substitutions = (
            getattr(function_node, "_generic_substitutions", {}) or {}
        )
        self.current_cooperative_matrix_role_overrides = (
            self.cooperative_matrix_role_overrides_for_function(function_node)
        )
        previous_function_literal_int_constant_shadows = (
            self.current_function_literal_int_constant_shadows
        )
        self.current_function_literal_int_constant_shadows = {
            param.name
            for param in self.function_parameters(function_node)
            if getattr(param, "name", None)
        }
        previous_function_literal_int_constants = (
            self.current_function_literal_int_constants
        )
        self.current_function_literal_int_constants = {}
        return_type = self.map_crossgl_type(function_node.return_type)
        self.function_nodes[function_node.name] = function_node
        previous_return_type = self.current_return_type
        previous_return_type_source = self.current_return_type_source
        previous_return_semantic_output = self.current_return_semantic_output
        previous_entry_point_return_outputs = self.current_entry_point_return_outputs
        previous_stage = self.current_stage
        previous_function_node = self.current_function_node
        previous_function_name = self.current_function_name
        self.current_return_type = return_type
        self.current_return_type_source = self.type_name_from_value(
            function_node.return_type
        )
        self.current_return_semantic_output = None
        self.current_entry_point_return_outputs = None
        self.current_function_node = function_node
        self.current_function_name = function_node.name
        if stage is not None:
            self.current_stage = stage

        execution_model_hint = self.execution_model_hint_for_function(
            function_node, stage
        )
        is_entry_point = self.function_is_entry_point(function_node, stage)
        previous_entry_resource_binding_state = None
        if is_entry_point and self.entry_point_resource_binding_baseline is not None:
            previous_entry_resource_binding_state = self.resource_binding_state()
            self.restore_resource_binding_state(
                self.entry_point_resource_binding_baseline
            )
        entry_interface_locations = (
            interface_location_state(self) if is_entry_point else None
        )
        program_cbuffer_members = self.cbuffer_members
        entry_cbuffer_members = (
            self.entry_cbuffer_members.get(function_node.name, {})
            if is_entry_point
            else {}
        )
        if entry_cbuffer_members:
            # The members of this kernel's own constant blocks (spirv_entry_resources).
            self.cbuffer_members = {**program_cbuffer_members, **entry_cbuffer_members}
        if execution_model_hint is None and is_entry_point and stage is None:
            execution_model_hint = self.spirv_execution_model(
                self.get_function_qualifier(function_node)
            )
        apply_inline_spirv_requirements(self, function_node)
        return_semantic = self.semantic_from_node(function_node)
        has_direct_return_semantic = is_entry_point and return_semantic is not None
        if return_semantic is not None:
            self.validate_spirv_return_semantic(
                execution_model_hint, function_node.return_type, return_semantic
            )
        entry_point_uses_void_signature = is_entry_point
        function_return_type = (
            self.register_primitive_type("void")
            if entry_point_uses_void_signature
            else return_type
        )
        has_mesh_output_parameters, signature = self.collect_spirv_function_parameters(
            function_node,
            entry_point_uses_void_signature,
            execution_model_hint,
            is_entry_point,
        )

        function_id = self.register_spirv_function(
            function_node, stage, function_return_type, signature
        )

        self.declare_spirv_function_parameters(
            entry_point_uses_void_signature, signature
        )

        previous_execution_model = self.current_execution_model
        previous_function_id = self.current_function_id
        previous_mesh_output_parameters = self.current_mesh_output_parameters
        self.bind_spirv_function_interface(
            function_node,
            entry_point_uses_void_signature,
            execution_model_hint,
            function_id,
            has_direct_return_semantic,
            has_mesh_output_parameters,
            return_semantic,
            return_type,
            signature,
        )

        # A precise function evaluates all of its floating-point arithmetic as written.
        self.emit_spirv_function_body(function_node, entry_point_uses_void_signature)

        self.current_execution_model = previous_execution_model
        self.current_function_id = previous_function_id
        self.current_mesh_output_parameters = previous_mesh_output_parameters
        self.current_function_node = previous_function_node
        self.current_function_name = previous_function_name
        self.current_stage = previous_stage
        self.current_return_type = previous_return_type
        self.current_return_type_source = previous_return_type_source
        self.current_return_semantic_output = previous_return_semantic_output
        self.current_entry_point_return_outputs = previous_entry_point_return_outputs
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.current_function_literal_int_constants = (
            previous_function_literal_int_constants
        )
        self.current_function_literal_int_constant_shadows = (
            previous_function_literal_int_constant_shadows
        )
        self.current_cooperative_matrix_role_overrides = (
            previous_cooperative_matrix_role_overrides
        )
        (
            self.cooperative_matrix_selection_depth,
            self.cooperative_matrix_loop_control_depth,
            self.cooperative_matrix_control_flow_diverged,
        ) = previous_cooperative_matrix_control_state
        if previous_entry_resource_binding_state is not None:
            self.restore_resource_binding_state(previous_entry_resource_binding_state)
        if entry_interface_locations is not None:
            restore_interface_location_state(self, entry_interface_locations)
        if entry_cbuffer_members:
            for name, member in self.cbuffer_members.items():
                if name not in entry_cbuffer_members:
                    program_cbuffer_members.setdefault(name, member)
            self.cbuffer_members = program_cbuffer_members
        self.local_variables.clear()
        self.precise_local_variables.clear()
        self.resource_alias_variables.clear()
        self.storage_buffer_pointer_offset_aliases = (
            previous_storage_buffer_pointer_offset_aliases
        )
        self.storage_pointer_reinterpretations = (
            previous_storage_pointer_reinterpretations
        )
        return function_id

    def scan_storage_buffer_requirement_node(
        self,
        node,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            for value in node.values():
                self.scan_storage_buffer_requirement_node(
                    value,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                self.scan_storage_buffer_requirement_node(
                    value,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return

        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, AssignmentNode):
            target = getattr(
                node, "target", getattr(node, "left", getattr(node, "name", None))
            )
            operator = getattr(node, "operator", "=")
            self.scan_storage_buffer_assignment_target_requirement(
                target,
                operator,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                getattr(node, "value", getattr(node, "right", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(node, ReturnNode):
            self.scan_storage_buffer_requirement_node(
                getattr(node, "value", None),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(node, VariableNode):
            self.scan_storage_buffer_requirement_node(
                getattr(node, "initial_value", None),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(
            node,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                FunctionCallNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ):
            self.scan_storage_buffer_requirement_expression(
                node,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if hasattr(node, "expression"):
            self.scan_storage_buffer_requirement_node(
                node.expression,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if hasattr(node, "__dict__"):
            for field_name, child in vars(node).items():
                if field_name in {"annotations", "parent"}:
                    continue
                self.scan_storage_buffer_requirement_node(
                    child,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )

    def nested_inline_return_node(self, function_node):
        visited = set()

        def find_nested_return(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return None
            if isinstance(value, ReturnNode):
                return value
            if isinstance(value, dict):
                for child in value.values():
                    nested_return = find_nested_return(child)
                    if nested_return is not None:
                        return nested_return
                return None
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    nested_return = find_nested_return(child)
                    if nested_return is not None:
                        return nested_return
                return None

            value_id = id(value)
            if value_id in visited:
                return None
            visited.add(value_id)

            if isinstance(value, IfNode):
                condition_has_name = any(
                    isinstance(node, (IdentifierNode, VariableNode))
                    for node in self.walk_ast_nodes(value.if_condition)
                )
                literal_condition = (
                    None
                    if condition_has_name
                    else evaluate_literal_int_expression(value.if_condition, {})
                )
                if literal_condition is not None:
                    selected = (
                        value.if_body if literal_condition != 0 else value.else_body
                    )
                    return find_nested_return(selected)
                nested_return = find_nested_return(value.if_body)
                if nested_return is not None:
                    return nested_return
                return find_nested_return(value.else_body)

            if not hasattr(value, "__dict__"):
                return None
            for child in vars(value).values():
                nested_return = find_nested_return(child)
                if nested_return is not None:
                    return nested_return
            return None

        body = getattr(function_node, "body", [])
        statements = (
            body.statements
            if isinstance(body, BlockNode)
            else body if isinstance(body, list) else [body]
        )
        for statement in statements:
            if isinstance(statement, ReturnNode):
                continue
            nested_return = find_nested_return(statement)
            if nested_return is not None:
                return nested_return
        return None

    def function_parameter_has_reachable_use(
        self,
        function_node,
        parameter_name: str,
    ) -> bool:
        visited = set()

        def references(value) -> bool:
            if value is None or isinstance(value, (str, int, float, bool)):
                return False
            if isinstance(value, dict):
                return any(references(child) for child in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(references(child) for child in value)

            value_id = id(value)
            if value_id in visited:
                return False
            visited.add(value_id)
            if isinstance(value, (IdentifierNode, VariableNode)) and (
                getattr(value, "name", None) == parameter_name
            ):
                return True
            if isinstance(value, IfNode):
                if references(value.if_condition):
                    return True
                literal_condition = evaluate_literal_int_expression(
                    value.if_condition,
                    self.active_literal_int_constants(),
                )
                if literal_condition is None:
                    return references(value.if_body) or references(value.else_body)
                selected = value.if_body if literal_condition != 0 else value.else_body
                return references(selected)
            if not hasattr(value, "__dict__"):
                return False
            return any(references(child) for child in vars(value).values())

        return references(getattr(function_node, "body", []))

    def process_expression(self, expr) -> Optional[SpirvId]:
        """Process a CrossGL expression."""
        if expr is None:
            return None

        if isinstance(expr, bool):
            bool_type = self.register_primitive_type("bool")
            return self.register_constant(expr, bool_type)
        elif isinstance(expr, int):
            expected_type = self.expected_primitive_type_name()
            primitive_type = (
                expected_type
                if expected_type in self.UNSIGNED_INTEGER_TYPES and expr >= 0
                else "int"
            )
            primitive_type = self.integer_literal_type_for_value(primitive_type, expr)
            int_type = self.register_primitive_type(primitive_type)
            return self.register_constant(expr, int_type)
        elif isinstance(expr, float):
            float_type = self.register_primitive_type("float")
            return self.register_constant(expr, float_type)

        elif isinstance(expr, str):
            return self.spirv_string_expression(expr)

        elif isinstance(expr, LiteralNode):
            return self.spirv_literal_expression(expr)

        elif isinstance(expr, IdentifierNode):
            return self.process_expression(expr.name)

        elif isinstance(expr, VariableNode):
            return self.spirv_variable_expression(expr)

        elif isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr)

        elif isinstance(expr, ArrayAccessNode):
            return self.spirv_array_access_expression(expr)

        elif isinstance(expr, AssignmentNode):
            target = getattr(
                expr, "name", getattr(expr, "target", getattr(expr, "left", None))
            )
            # spirv_single_evaluation.py: reading the target back reuses the index the store used.
            with subscript_index_scope(self):
                with recording_subscript_indices(self):
                    self.process_assignment(expr)
                return (
                    self.process_expression(target)
                    if isinstance(
                        target, (IdentifierNode, MemberAccessNode, ArrayAccessNode)
                    )
                    else None
                )

        elif isinstance(expr, BinaryOpNode):
            return self.spirv_binary_expression(expr)

        elif isinstance(expr, UnaryOpNode):
            return self.spirv_unary_expression(expr)

        elif isinstance(expr, TernaryOpNode):
            return self.spirv_ternary_expression(expr)

        elif isinstance(expr, CooperativeMatrixOpNode):
            if self.cooperative_matrix_khr_enabled:
                return self.process_cooperative_matrix_operation(expr)
            feature = "cooperative-matrix-lowering"
            missing_capabilities = ("spirv.cooperative_matrix.khr",)
            message = (
                "SPIR-V cooperative-matrix lowering is not available for "
                f"operation '{expr.operation}'; scalar substitution would "
                "change subgroup-distributed matrix semantics"
            )
            raise UnsupportedSPIRVFeatureError(
                feature,
                message,
                missing_capabilities=missing_capabilities,
                source_location=getattr(expr, "source_location", None),
            )

        elif isinstance(expr, WaveOpNode):
            return self.call_wave_operation(expr.operation, expr.arguments)

        elif isinstance(expr, RayTracingOpNode):
            return self.process_ray_tracing_operation(expr)

        elif isinstance(expr, RayQueryOpNode):
            return self.process_ray_query_operation(expr)

        elif isinstance(expr, MeshOpNode):
            return self.process_mesh_operation(expr)

        elif isinstance(expr, ConstructorNode):
            vector_list = self.process_vector_list_constructor(expr)
            if vector_list is not None:
                return vector_list
            constructed = self.process_struct_constructor_node(expr)
            if constructed is not None:
                return constructed
            self.refuse_lowering(
                f"Unsupported constructor {self.convert_type_node_to_string(expr.constructor_type)}"
            )
            return None

        elif isinstance(expr, FunctionCallNode):
            return self.spirv_function_call_expression(expr)

        elif isinstance(expr, MemberAccessNode):
            return self.spirv_member_access_expression(expr)

        elif is_comma_expression(expr):
            return spirv_comma_expression(self, expr)

        else:
            self.refuse_lowering(f"Unknown expression type {type(expr).__name__}")
            return None

    def spirv_builtin_alias(self, name: str) -> str:
        aliases = {
            "SV_DISPATCHTHREADID": "gl_GlobalInvocationID",
            "SV_GROUPTHREADID": "gl_LocalInvocationID",
            "SV_GROUPID": "gl_WorkGroupID",
            "SV_GROUPINDEX": "gl_LocalInvocationIndex",
            "SV_VERTEXID": "gl_VertexID",
            "GL_VERTEXINDEX": "gl_VertexID",
            "SV_INSTANCEID": "gl_InstanceID",
            "GL_INSTANCEINDEX": "gl_InstanceIndex",
            "SV_PRIMITIVEID": "gl_PrimitiveID",
            # Metal input writes [[position_in_patch]] as HLSL's domain location.
            "SV_DOMAINLOCATION": "gl_TessCoord",
            "SV_VIEWID": "gl_ViewIndex",
            "SV_ISFRONTFACE": "gl_FrontFacing",
            "SV_SAMPLEINDEX": "gl_SampleID",
            "SV_BARYCENTRICS": "gl_BaryCoordEXT",
            "SV_STENCILREF": "gl_FragStencilRefEXT",
            "SV_POSITION": (
                "gl_FragCoord"
                if self.current_execution_model == "Fragment"
                else "gl_Position"
            ),
            "SV_DEPTH": "gl_FragDepth",
            "SV_COVERAGE": "gl_SampleMask",
            "SV_SHADINGRATE": (
                "gl_ShadingRateEXT"
                if self.current_execution_model == "Fragment"
                else "gl_PrimitiveShadingRateEXT"
            ),
            **SPIRV_EXTENSION_BUILTIN_ALIASES,
        }
        return aliases.get(str(name).upper(), name)

    def _generate_module(self, ast):
        self.begin_spirv_module(ast)

        self.declare_spirv_module_types_and_constants(ast)

        global_specialization_constant_declarations, all_functions = (
            self.collect_spirv_module_functions(ast)
        )

        self.collect_spirv_function_metadata(ast, all_functions)

        self.declare_spirv_module_globals(
            ast, global_specialization_constant_declarations
        )

        top_level_entries = self.emit_spirv_helper_functions(ast)

        entry_points = self.emit_spirv_entry_functions(ast, top_level_entries)

        if entry_points:
            self.main_fn_id = entry_points[0][1].id
            for execution_model, function_id, entry_name, stage in entry_points:
                self.emit_entry_point(execution_model, function_id, entry_name, stage)

        return "\n".join(self.ordered_module_lines())
