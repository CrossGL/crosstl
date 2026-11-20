"""CrossGL-to-Vulkan SPIR-V code generator."""

import re
from typing import List, Optional, Union

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstantNode,
    ConstructorNode,
    CooperativeMatrixOpNode,
    EnumNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..host_workgroup_size import function_entry_execution_config
from ..unhandled import UNHANDLED
from .array_utils import collect_literal_int_constants, evaluate_literal_int_expression
from .comma_expressions import is_comma_expression, spirv_comma_expression
from .enum_utils import (
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_parameter_names,
)
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
    prepare_generic_function_specializations,
)
from .precise_dataflow import declaration_is_precise, precise_contributors
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
from .spirv_entry_bindings import reserve_entry_buffer_bindings
from .spirv_entry_locations import (
    interface_location_state,
    restore_interface_location_state,
)
from .spirv_entry_point_parameters import SpirvEntryPointParameterMixin
from .spirv_entry_workgroup_sizes import plan_entry_workgroup_sizes
from .spirv_enums_and_structs import SpirvEnumStructMixin
from .spirv_exact_lookups import spirv_builtin_call_key
from .spirv_expression_kinds import SpirvExpressionKindMixin
from .spirv_extension_builtins import BUILTIN_ALIASES as SPIRV_EXTENSION_BUILTIN_ALIASES
from .spirv_float_semantics import SpirvFloatSemanticsMixin
from .spirv_function_prepass import SpirvFunctionPrepassMixin
from .spirv_global_declarations import SpirvGlobalDeclarationMixin
from .spirv_group_operations import SpirvGroupOperationMixin
from .spirv_inline_requirements import apply_inline_spirv_requirements
from .spirv_local_declarations import SpirvLocalDeclarationMixin
from .spirv_local_references import SpirvLocalReferenceMixin
from .spirv_math_polyfills import SpirvMathPolyfillMixin
from .spirv_matrix_builtins import SpirvMatrixBuiltinMixin
from .spirv_mesh_builtins import (
    glsl_mesh_builtin_output_infos,
    is_glsl_mesh_builtin_output_declaration,
)
from .spirv_mesh_outputs import SpirvMeshOutputMixin
from .spirv_module_assembly import SpirvModuleAssemblyMixin
from .spirv_pointer_parameters import SpirvPointerParameterMixin
from .spirv_ray_tracing import SpirvRayTracingMixin
from .spirv_resource_calls import SpirvResourceCallMixin
from .spirv_sampling_calls import SpirvSamplingCallMixin
from .spirv_single_evaluation import (
    recording_subscript_indices,
    subscript_index_scope,
)
from .spirv_specialized_workgroup_size import module_compute_stage_config
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
        mesh_output_parameters = {}
        output_parameters = []
        runtime_parameters = []
        param_types = []
        param_value_types = []
        resource_array_param_indices = set()
        value_array_param_indices = set()
        patch_interface_parameters = []
        param_type_hints = self.resolve_function_resource_array_type_hints(
            function_node.name
        )
        storage_image_pointer_params = (
            self.resolve_function_storage_image_pointer_params(function_node.name)
        )
        parameters = getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        )
        parameters = list(parameters or [])
        if is_entry_point:
            reserve_entry_buffer_bindings(self, parameters)
        skipped_parameter_indices = (
            set()
            if is_entry_point
            else self.skipped_function_parameter_indices_for_node(function_node)
        )
        if skipped_parameter_indices:
            parameters = [
                param
                for index, param in enumerate(parameters)
                if index not in skipped_parameter_indices
            ]
        if not is_entry_point:
            required_stage_input = self.required_function_stage_input_type(
                function_node.name
            )
            required_stage_output = self.required_function_stage_output_type(
                function_node.name
            )
            if required_stage_input is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("input", required_stage_input)
                )
            if required_stage_output is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("output", required_stage_output)
                )
        has_mesh_output_parameters = any(
            self.is_mesh_output_parameter(param) for param in parameters
        )
        mesh_parameter_execution_model = execution_model_hint
        if mesh_parameter_execution_model is None and has_mesh_output_parameters:
            mesh_parameter_execution_model = "MeshEXT"
        mesh_output_parameter_indices = set()
        if is_entry_point and mesh_parameter_execution_model == "MeshEXT":
            mesh_output_parameters.update(glsl_mesh_builtin_output_infos(self))

        for source_param_index, param in enumerate(parameters):
            param_type_source = getattr(
                param, "param_type", getattr(param, "vtype", None)
            )
            param_name = getattr(param, "name", None)
            mesh_output_info = self.mesh_output_parameter_info(
                param, mesh_parameter_execution_model
            )
            if mesh_output_info is not None:
                mesh_output_parameters[param_name] = mesh_output_info
                mesh_output_parameter_indices.add(source_param_index)
                continue

            patch_info = self.patch_parameter_info(
                param, execution_model_hint, is_entry_point
            )
            if patch_info is not None:
                patch_interface_parameters.append((param, patch_info))
                continue

            if param_name in param_type_hints:
                param_type_source = param_type_hints[param_name]
            if is_entry_point and self.is_storage_resource_parameter(param):
                param_type = self.register_primitive_type("void")
                param_value_types.append(param_type)
                runtime_parameters.append((param, param_type, param_type))
                continue
            if param_type_source is not None:
                param_type = self.map_resource_type_with_format(
                    param_type_source, param
                )
            else:
                param_type = self.map_crossgl_type("float")

            if self.is_graphics_output_parameter(
                param, execution_model_hint, is_entry_point
            ):
                output_parameters.append((param, param_type))
                continue

            param_value_types.append(param_type)
            param_resource_metadata = self.resource_type_metadata.get(param_type.id)
            is_stage_object_pointer_param = bool(
                getattr(param, "_spirv_stage_object_pointer", False)
            )
            is_storage_image_param = (
                param_resource_metadata is not None
                and param_resource_metadata.get("kind") == "storage_image"
                and param_name in storage_image_pointer_params
            )
            is_value_array_param = (
                self.array_type_info_from_type(param_type) is not None
            )
            if (
                self.is_resource_array_type(param_type)
                or is_storage_image_param
                or is_value_array_param
                or is_stage_object_pointer_param
            ):
                resource_array_param_indices.add(len(param_types))
                if is_value_array_param and not (
                    self.is_resource_array_type(param_type)
                    or is_storage_image_param
                    or is_stage_object_pointer_param
                    or self.function_parameter_is_reference_like(param)
                ):
                    value_array_param_indices.add(len(param_types))
                storage_class = (
                    "UniformConstant"
                    if self.is_resource_array_type(param_type) or is_storage_image_param
                    else "Function"
                )
                param_type = self.register_pointer_type(param_type, storage_class)

            if not entry_point_uses_void_signature:
                param_types.append(param_type)
            runtime_parameters.append((param, param_type, param_value_types[-1]))

        had_global_function = function_node.name in self.functions
        previous_global_function = self.functions.get(function_node.name)
        previous_global_signature = self.function_signatures.get(function_node.name)
        function_id = self.create_function(
            function_node.name,
            function_return_type,
            param_types,
            register_global=stage is None,
        )
        if stage is not None:
            self.register_stage_local_function(
                stage,
                function_node.name,
                function_id,
                function_return_type,
                param_types,
            )
            if had_global_function:
                self.functions[function_node.name] = previous_global_function
                self.function_signatures[function_node.name] = previous_global_signature
            else:
                self.functions.pop(function_node.name, None)
                self.function_signatures.pop(function_node.name, None)
        self.function_resource_array_params[function_node.name] = (
            resource_array_param_indices
        )
        self.function_mesh_output_parameter_indices[function_node.name] = (
            mesh_output_parameter_indices
        )
        if stage is not None:
            key = self.stage_local_function_key(stage, function_node.name)
            self.stage_local_function_resource_array_params[key] = (
                resource_array_param_indices
            )
            self.stage_local_function_mesh_output_parameter_indices[key] = (
                mesh_output_parameter_indices
            )
        self.function_value_array_parameter_indices_by_id[function_id.id] = set(
            value_array_param_indices
        )

        for i, (param, param_type, param_value_type) in enumerate(runtime_parameters):
            if entry_point_uses_void_signature:
                continue
            if hasattr(param, "name"):
                param_name = param.name
            else:
                param_name = f"param{i}"

            param_id = self.create_function_parameter(param_type, param_name)
            self.local_variables[param_name] = param_id
            if i in resource_array_param_indices:
                self.variable_value_types[param_id.id] = param_value_type
            # A helper's image parameter is a value, not a memory object declaration, so it takes no
            # NonWritable or NonReadable decoration (SPIR-V 1.6 section 3.20); the caller's variable has them.
            self.register_declared_resource_metadata(
                param, param_id, param_value_type, decorate=False
            )

        self.begin_block()

        if not entry_point_uses_void_signature:
            for i, (param, _param_type, param_value_type) in enumerate(
                runtime_parameters
            ):
                if i not in value_array_param_indices:
                    continue
                if self.function_parameter_is_reference_like(param):
                    continue
                param_name = getattr(param, "name", f"param{i}")
                param_id = self.local_variables.get(param_name)
                if param_id is None:
                    continue
                local_copy = self.copy_array_pointer_to_function_storage(
                    param_id, param_value_type, name=param_name
                )
                if local_copy is not None:
                    self.local_variables[param_name] = local_copy

        previous_execution_model = self.current_execution_model
        previous_function_id = self.current_function_id
        previous_mesh_output_parameters = self.current_mesh_output_parameters
        self.current_function_id = function_id.id
        if execution_model_hint is not None:
            self.current_execution_model = execution_model_hint
        elif has_mesh_output_parameters:
            self.current_execution_model = "MeshEXT"
        elif self.current_execution_model is None:
            self.current_execution_model = self.execution_model_hint_for_function(
                function_node
            )
        self.current_mesh_output_parameters = mesh_output_parameters
        if has_direct_return_semantic:
            self.current_return_semantic_output = self.register_direct_return_output(
                function_node, return_semantic, return_type
            )
        elif entry_point_uses_void_signature and return_type.type.base_type != "void":
            self.current_entry_point_return_outputs = (
                self.register_entry_point_return_outputs(
                    function_node, return_type, execution_model_hint
                )
            )
        if entry_point_uses_void_signature:
            for param, param_value_type in output_parameters:
                param_name = getattr(param, "name", None) or "param"
                variable = self.register_entry_point_interface_variable(
                    execution_model_hint,
                    "Output",
                    self.entry_point_interface_name(
                        execution_model_hint, "output", param_name
                    ),
                    param_value_type,
                    param,
                )
                self.local_variables[param_name] = variable
                self.register_declared_resource_metadata(
                    param, variable, param_value_type
                )
        for param, patch_info in patch_interface_parameters:
            patch_variable = self.register_patch_parameter_interface_variable(
                param, patch_info
            )
            self.local_variables[patch_info["name"]] = patch_variable
            self.mark_function_interface_variable(patch_variable)

        if entry_point_uses_void_signature:
            self.initialize_entry_point_parameters(
                runtime_parameters, execution_model_hint
            )

        # A precise function evaluates all of its floating-point arithmetic as written.
        precise_function = self.has_attribute(
            function_node, "precise"
        ) or "precise" in {
            str(qualifier).lower()
            for qualifier in getattr(function_node, "qualifiers", None) or ()
        }
        self.precise_expression_depth += precise_function
        previous_precise_flow_names = self.precise_flow_names
        self.precise_flow_names = precise_contributors(
            function_node.body,
            seeds=sorted(
                set(self.precise_global_variables)
                | {
                    param.name
                    for param in getattr(function_node, "parameters", None) or ()
                    if isinstance(getattr(param, "name", None), str)
                    and declaration_is_precise(param)
                }
            ),
        )
        try:
            self.process_statements(function_node.body)
        finally:
            self.precise_expression_depth -= precise_function
            self.precise_flow_names = previous_precise_flow_names
        self.process_tessellation_patch_constant_function(function_node)

        if not self.current_block_has_terminator():
            if (
                entry_point_uses_void_signature
                or self.convert_type_node_to_string(function_node.return_type) == "void"
            ):
                self.create_return()
            else:
                self.create_unreachable()

        self.end_function()

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
        self.emit("; SPIR-V")
        self.emit("; Version: 1.0")
        self.emit("; Generator: CrossGL Vulkan SPIR-V Generator")
        self.emit("; Schema: 0")

        self.emit("OpCapability Shader")

        self.glsl_std450_id = self.get_id()
        self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        self.emit("OpMemoryModel Logical GLSL450")

        self.register_primitive_type("void")
        self.register_primitive_type("bool")
        self.register_primitive_type("int")
        self.register_primitive_type("float")

        literal_integer_declarations = [
            constant
            for constant in getattr(ast, "constants", []) or []
            if not self.spirv_specialization_constant_attributes(constant)
        ]
        self.literal_int_constants = collect_literal_int_constants(
            literal_integer_declarations
        )

        float_type = self.primitive_types["float"]
        for i in range(2, 5):
            self.register_vector_type(float_type, i)

        struct_declarations = list(getattr(ast, "structs", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            struct_declarations.extend(getattr(stage, "local_structs", []) or [])

        self.struct_declarations = {}
        self.enum_declarations = {}

        def collect_type_declarations(nodes):
            for node in nodes or []:
                if isinstance(node, StructNode):
                    self.struct_declarations.setdefault(node.name, node)
                    collect_type_declarations(getattr(node, "members", []) or [])
                elif isinstance(node, EnumNode):
                    self.enum_declarations.setdefault(node.name, node)

        collect_type_declarations(struct_declarations)
        # Generic parameters are unbound where a generic body is read; instances naming them stay generic.
        self.program_generic_parameter_names = frozenset(
            collect_generic_parameter_names(ast)
        ) - set(self.struct_declarations)
        self.generic_struct_instance_spellings = {}
        self.glsl_buffer_block_type_names = self.collect_glsl_buffer_block_type_names(
            ast
        )

        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            struct_declarations
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )

        preprocessed_specialization_constant_declarations = set()
        for constant in getattr(ast, "constants", []) or []:
            if not isinstance(constant, ConstantNode):
                continue
            if not self.spirv_specialization_constant_attributes(constant):
                continue
            self.process_named_constant_declaration(constant)
            preprocessed_specialization_constant_declarations.add(id(constant))
        for var in getattr(ast, "global_variables", []) or []:
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if declaration is None:
                continue
            self.process_named_constant_declaration(declaration)
            preprocessed_specialization_constant_declarations.add(id(declaration))

        self.collect_enum_metadata(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.collect_enum_metadata(getattr(stage, "local_structs", []) or [])
        self.register_generic_enum_metadata()

        self.process_enum_structs(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.process_enum_structs(getattr(stage, "local_structs", []) or [])

        for struct in ast.structs:
            if not isinstance(struct, StructNode):
                continue
            self.process_crossgl_struct(struct)
        for stage in (getattr(ast, "stages", None) or {}).values():
            for struct in getattr(stage, "local_structs", []) or []:
                if not isinstance(struct, StructNode):
                    continue
                self.process_crossgl_struct(struct)

        for constant in getattr(ast, "constants", []) or []:
            if (
                isinstance(constant, ConstantNode)
                and id(constant)
                not in preprocessed_specialization_constant_declarations
            ):
                self.process_named_constant_declaration(constant)

        self.source_variable_type_names = self.collect_source_variable_type_names(ast)

        global_specialization_constant_declarations = {
            id(declaration)
            for var in getattr(ast, "global_variables", []) or []
            for declaration in [
                self.spirv_specialization_constant_declaration_node(var)
            ]
            if declaration is not None
        }

        all_functions = self.collect_ast_functions(ast)
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
            global_types=self.source_variable_type_names,
            ast=ast,
        )
        if generic_function_specializations:
            all_functions = all_functions + list(
                generic_function_specializations.values()
            )
        self.reject_unsupported_generic_functions(ast)
        self.function_nodes_by_name = self.collect_functions_by_name(all_functions)

        self.function_resource_array_type_hints = (
            self.collect_resource_array_parameter_type_hints(ast)
        )
        self.collect_stage_local_resource_array_parameter_type_hints(ast)
        self.function_image_access_requirements = (
            self.collect_function_image_access_requirements_for_ast(ast)
        )
        self.collect_stage_local_function_image_access_metadata(ast)
        self.function_storage_buffer_access_requirements = (
            self.collect_function_storage_buffer_access_requirements_for_functions(
                all_functions
            )
        )
        self.collect_stage_local_function_storage_buffer_metadata(ast)
        self.inline_storage_buffer_functions = (
            self.collect_inline_storage_buffer_functions(all_functions)
        )
        self.function_execution_models = self.collect_function_execution_models(
            ast, all_functions
        )
        (
            self.spirv_skipped_function_parameter_indices,
            self.spirv_skipped_function_parameter_indices_by_id,
        ) = self.collect_unused_array_parameter_index_maps(all_functions)
        self.function_stage_input_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "input")
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "output")
        )
        self.function_storage_image_pointer_params = (
            self.collect_storage_image_pointer_parameters(ast)
        )
        self.collect_stage_local_storage_image_pointer_metadata(ast)
        self.reserve_explicit_resource_bindings(ast)
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
            if self.stage_key(stage_type) == "tessellation_control":
                self.tessellation_control_stage = stage
                break

        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.process_cbuffer_declaration(cbuffer)
        self.declare_default_uniform_block(ast)

        for var in getattr(ast, "global_variables", []):
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if (
                declaration is not None
                and id(declaration) in global_specialization_constant_declarations
            ):
                continue
            self.process_global_variable_declaration(var)
        if not getattr(ast, "stages", None):
            self.bind_default_uniform_block()

        top_level_entries = []
        helper_functions = []
        for func in ast.functions:
            if getattr(func, "body", None) is None:
                continue
            if generic_function_parameters(func):
                helper_functions.extend(generic_function_emission_list(self, func))
                continue
            qualifier = self.get_function_qualifier(func)

            if func.name == "main" or qualifier in [
                "vertex",
                "fragment",
                "compute",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "mesh",
                "task",
                "object",
                "amplification",
                "ray_generation",
                "ray_intersection",
                "ray_closest_hit",
                "ray_miss",
                "ray_any_hit",
                "ray_callable",
            ]:
                top_level_entries.append((func, qualifier))
            else:
                helper_functions.append(func)

        compute_entry_configs = [
            function_entry_execution_config(func)
            for func, qualifier in top_level_entries
            if qualifier == "compute"
        ]
        if compute_entry_configs and all(
            config == compute_entry_configs[0] for config in compute_entry_configs
        ):
            # Built-ins read outside a stage take the size of the module's compute entries.
            self.module_compute_entry_config = compute_entry_configs[0]
        elif not compute_entry_configs:
            self.module_compute_entry_config = module_compute_stage_config(self, ast)
        plan_entry_workgroup_sizes(self, ast, top_level_entries)

        if self.tessellation_control_stage is None:
            for func, qualifier in top_level_entries:
                if qualifier == "tessellation_control":
                    self.tessellation_control_stage = func
                    break

        for func in self.order_functions_by_dependencies(helper_functions):
            if self.function_has_storage_buffer_parameters(func):
                continue
            self.process_function_node(func)

        entry_points = []

        if getattr(ast, "stages", None):
            for stage_type, stage in ast.stages.items():
                previous_execution_model = self.current_execution_model
                previous_stage = self.current_stage
                self.current_stage = stage
                self.current_execution_model = self.spirv_execution_model(
                    self.stage_key(stage_type)
                )
                try:
                    for var in getattr(stage, "local_variables", []):
                        if is_glsl_mesh_builtin_output_declaration(self, var):
                            continue
                        if not self.global_variable_exists_for_current_stage(var.name):
                            self.process_global_variable_declaration(var)
                finally:
                    self.current_execution_model = previous_execution_model
                    self.current_stage = previous_stage

            self.bind_default_uniform_block()
            self.entry_point_resource_binding_baseline = self.resource_binding_state()

            processed_local_functions = set()
            for stage in ast.stages.values():
                local_functions = [
                    func
                    for func in getattr(stage, "local_functions", [])
                    if id(func) not in processed_local_functions
                    and getattr(func, "body", None) is not None
                ]
                for func in self.order_functions_by_dependencies(local_functions):
                    if id(func) not in processed_local_functions:
                        if generic_function_parameters(func):
                            for specialized_func in generic_function_emission_list(
                                self, func
                            ):
                                self.process_function_node(
                                    specialized_func, stage=stage
                                )
                            processed_local_functions.add(id(func))
                            continue
                        if self.function_has_storage_buffer_parameters(func):
                            processed_local_functions.add(id(func))
                            continue
                        self.process_function_node(func, stage=stage)
                        processed_local_functions.add(id(func))

            for stage_type, stage in ast.stages.items():
                entry_function = stage.entry_point
                function_id = self.process_function_node(entry_function, stage=stage)
                stage_name = self.stage_key(stage_type)
                execution_model = self.spirv_execution_model(stage_name)
                entry_points.append(
                    (execution_model, function_id, entry_function.name, stage)
                )
        else:
            self.entry_point_resource_binding_baseline = self.resource_binding_state()
            for func, qualifier in top_level_entries:
                function_id = self.process_function_node(func)
                execution_model = self.spirv_execution_model(qualifier)
                is_tessellation_entry = execution_model in {
                    "TessellationControl",
                    "TessellationEvaluation",
                }
                stage_metadata = (
                    func
                    if is_tessellation_entry or execution_model == "GLCompute"
                    else None
                )
                entry_points.append(
                    (execution_model, function_id, func.name, stage_metadata)
                )

        if entry_points:
            self.main_fn_id = entry_points[0][1].id
            for execution_model, function_id, entry_name, stage in entry_points:
                self.emit_entry_point(execution_model, function_id, entry_name, stage)

        return "\n".join(self.ordered_module_lines())
