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
from ..matrix_shape import crossgl_matrix_shape, length_method_object
from .array_utils import collect_literal_int_constants, evaluate_literal_int_expression
from .bitfield_builtins import BITFIELD_ARITIES, spirv_bitfield_call
from .comma_expressions import is_comma_expression, spirv_comma_expression
from .enum_utils import (
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_parameter_names,
)
from .extended_arithmetic_builtins import (
    EXTENDED_ARITHMETIC_FUNCTIONS,
    spirv_extended_arithmetic_call,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_definition_name,
    generic_function_emission_list,
    generic_function_parameters,
    generic_function_value_arguments,
    prepare_generic_function_specializations,
    reject_unresolved_generic_member_call,
)
from .pack_builtins import PACK_FUNCTIONS, spirv_pack_call
from .precise_dataflow import declaration_is_precise, precise_contributors
from .spirv_access_chains import SpirvAccessChainMixin
from .spirv_assignments import SpirvAssignmentMixin
from .spirv_atomic_load_store import (
    SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS,
    call_spirv_atomic_load_store,
)
from .spirv_bfloat16_math import SpirvBfloat16MathMixin
from .spirv_binary_operations import SpirvBinaryOperationMixin
from .spirv_boolean_reductions import BOOLEAN_REDUCTIONS, spirv_boolean_reduction
from .spirv_buffer_layouts import SpirvBufferLayoutMixin
from .spirv_buffer_methods import SpirvBufferMethodMixin
from .spirv_builtin_calls import SpirvBuiltinCallMixin
from .spirv_byte_address_atomics import process_byte_address_atomic_call
from .spirv_control_flow import SpirvControlFlowMixin
from .spirv_cooperative_matrices import SpirvCooperativeMatrixMixin
from .spirv_debug_printf import DEBUG_PRINTF_FUNCTIONS, spirv_debug_printf_call
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
from .spirv_exponent_builtins import (
    OUTPUT_OPERAND_FUNCTIONS,
    spirv_ldexp_call,
    spirv_output_operand_call,
)
from .spirv_extension_builtins import BUILTIN_ALIASES as SPIRV_EXTENSION_BUILTIN_ALIASES
from .spirv_float_semantics import SpirvFloatSemanticsMixin
from .spirv_function_prepass import SpirvFunctionPrepassMixin
from .spirv_geometry_streams import STREAM_FUNCTIONS
from .spirv_global_declarations import SpirvGlobalDeclarationMixin
from .spirv_group_operations import SpirvGroupOperationMixin
from .spirv_image_texels import emit_four_component_image_read, narrow_image_texel
from .spirv_inline_requirements import apply_inline_spirv_requirements
from .spirv_literal_context import is_literal_operand
from .spirv_local_declarations import SpirvLocalDeclarationMixin
from .spirv_local_references import SpirvLocalReferenceMixin
from .spirv_math_polyfills import SpirvMathPolyfillMixin
from .spirv_matrix_builtins import SpirvMatrixBuiltinMixin
from .spirv_matrix_constructors import one_argument_matrix_columns
from .spirv_mesh_builtins import (
    glsl_mesh_builtin_output_infos,
    is_glsl_mesh_builtin_output_declaration,
)
from .spirv_mesh_outputs import SpirvMeshOutputMixin
from .spirv_module_assembly import SpirvModuleAssemblyMixin
from .spirv_pointer_parameters import SpirvPointerParameterMixin
from .spirv_ray_builtins import ray_constant_value, transposed_ray_builtin
from .spirv_ray_tracing import SpirvRayTracingMixin
from .spirv_shader_clock import SPIRV_SHADER_CLOCKS, spirv_shader_clock_call
from .spirv_short_circuit import (
    SHORT_CIRCUIT_OPERATORS,
    lower_logical_intrinsic,
    lower_short_circuit_logical,
    lower_short_circuit_ternary,
)
from .spirv_single_evaluation import (
    evaluate_subscript_index,
    recording_subscript_indices,
    subscript_index_scope,
)
from .spirv_specialized_workgroup_size import module_compute_stage_config
from .spirv_stage_builtins import SpirvStageBuiltinMixin
from .spirv_standard_constants import spirv_standard_float_constant
from .spirv_storage_buffer_access import SpirvStorageBufferAccessMixin
from .spirv_texel_buffers import emit_texel_buffer_fetch
from .spirv_texture_calls import SpirvTextureCallMixin
from .spirv_texture_residency import (
    spirv_texture_residency_call,
    spirv_texture_resident_call,
)
from .spirv_types_and_constants import SpirvTypeConstantMixin
from .spirv_value_subscripts import vector_value_subscript
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
from .texture_lod_clamp import (
    is_texture_lod_clamp_operation,
    spirv_texture_lod_clamp_call,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
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
        if is_texture_lod_clamp_operation(function_name):
            sampler = (
                self.resource_metadata_for_value(args[1]) if len(args) > 1 else None
            )
            return spirv_texture_lod_clamp_call(
                self,
                function_name,
                args,
                bool(sampler) and sampler.get("kind") == "sampler",
            )
        if function_name in self.image_atomic_function_names():
            return self.call_image_atomic_function(function_name, args)
        if function_name in self.buffer_atomic_function_names():
            return self.call_buffer_atomic_function(function_name, args)
        if function_name in SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS:
            return call_spirv_atomic_load_store(self, function_name, args)
        if function_name in self.projected_texture_function_names():
            return self.call_projected_texture_function(function_name, args)
        if function_name in self.projected_shadow_function_names():
            return self.call_projected_shadow_function(function_name, args)

        byte_address_load_width = self.byte_address_helper_load_width(function_name)
        if byte_address_load_width is not None:
            return self.call_byte_address_buffer_load_helper(
                function_name, args, byte_address_load_width
            )

        byte_address_store_width = self.byte_address_helper_store_width(function_name)
        if byte_address_store_width is not None:
            return self.call_byte_address_buffer_store_helper(
                function_name, args, byte_address_store_width
            )

        if function_name == "buffer_append":
            if len(args) < 2:
                self.refuse_lowering("buffer_append requires buffer and value operands")
                return None
            if len(args) > 2:
                self.refuse_lowering(
                    "buffer_append accepts only buffer and value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_append requires an AppendStructuredBuffer " "operand"
                )
                return None
            return self.process_structured_buffer_append_call(
                args[0], metadata, [args[1]], "buffer_append"
            )[1]

        if function_name == "buffer_consume":
            if not args:
                self.refuse_lowering("buffer_consume requires a buffer operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )
            if len(args) > 1:
                self.refuse_lowering("buffer_consume accepts only a buffer operand")
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_consume requires a ConsumeStructuredBuffer " "operand"
                )
                return self.default_value_for_buffer_load_failure(args)
            return self.process_structured_buffer_consume_call(
                args[0], metadata, [], "buffer_consume"
            )[1]

        if function_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires a buffer operand")
                return self.structured_buffer_counter_default_value()
            if len(args) > 1:
                self.refuse_lowering(f"{function_name} accepts only a buffer operand")
                return self.structured_buffer_counter_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    f"{function_name} requires an RWStructuredBuffer " "operand"
                )
                return self.structured_buffer_counter_default_value()
            return self.process_structured_buffer_counter_method_call(
                args[0],
                metadata,
                [],
                function_name,
                function_name == "buffer_increment_counter",
            )[1]

        if function_name == "buffer_dimensions":
            if not args:
                self.refuse_lowering("buffer_dimensions requires a buffer operand")
                return self.structured_buffer_dimensions_default_value()
            if len(args) > 1:
                self.refuse_lowering(
                    "buffer_dimensions expression form accepts only a " "buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_dimensions requires a structured or "
                    "byte-address buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()
            return self.emit_structured_buffer_dimensions(
                args[0], metadata, "buffer_dimensions"
            )

        if function_name == "buffer_load":
            if len(args) < 2:
                self.refuse_lowering("buffer_load requires buffer and index operands")
                return self.default_value_for_buffer_load_failure(args)

            if len(args) > 2:
                self.refuse_lowering(
                    "buffer_load accepts only buffer and index operands"
                )
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None and metadata.get("writeonly"):
                self.refuse_lowering("buffer_load requires a readable buffer")
                return self.default_value_for_buffer_load_failure(args)
            if (
                metadata is not None
                and not metadata.get("byte_address")
                and metadata.get("buffer_kind")
                not in {
                    "StructuredBuffer",
                    "RWStructuredBuffer",
                }
            ):
                self.refuse_lowering("buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.refuse_lowering("buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_type = self.variable_value_types[element_pointer.id]
            return self.load_from_variable(element_pointer, element_type)

        if function_name == "buffer_store":
            if len(args) < 3:
                self.refuse_lowering(
                    "buffer_store requires buffer, index, and value operands"
                )
                return None

            if len(args) > 3:
                self.refuse_lowering(
                    "buffer_store accepts only buffer, index, and " "value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if (
                not metadata.get("byte_address")
                and metadata.get("buffer_kind") != "RWStructuredBuffer"
            ):
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if metadata.get("readonly"):
                self.refuse_lowering("buffer_store requires an RWStructuredBuffer")
                return None

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None

            self.store_to_variable(element_pointer, args[2])
            return None

        if function_name == "subpassLoad":
            if not args:
                self.refuse_lowering("subpassLoad requires an input attachment operand")
                result_type = self.register_vector_type(
                    self.register_primitive_type("float"), 4
                )
                return self.default_value_for_type(result_type)

            image_id = args[0]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "input_attachment":
                self.refuse_lowering("subpassLoad requires a subpassInput operand")
                result_type = self.register_vector_type(
                    self.register_primitive_type("float"), 4
                )
                return self.default_value_for_type(result_type)

            result_type = self.resource_access_result_type(metadata)
            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 2:
                    self.refuse_lowering(
                        "subpassLoad on subpassInputMS requires " "a sample operand"
                    )
                    return self.default_value_for_type(result_type)
                if not self.validate_storage_image_sample("subpassLoad", args[1]):
                    return self.default_value_for_type(result_type)
                image_operands = f" Sample %{args[1].id}"

            coord_id = self.input_attachment_zero_coordinate()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpImageRead %{result_type.id} "
                f"%{image_id.id} %{coord_id.id}{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "imageLoad":
            if len(args) < 2:
                self.refuse_lowering("imageLoad requires image and coordinate operands")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            image_id, coord_id = args[0], args[1]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.refuse_lowering("imageLoad requires a storage image operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            result_type = self.resource_access_result_type(metadata)
            if metadata.get("writeonly"):
                self.refuse_lowering("imageLoad requires a readable storage image")
                return self.default_value_for_type(result_type)

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 3:
                    self.refuse_lowering("imageLoad requires a sample operand")
                    return self.default_value_for_type(result_type)
                if not self.validate_storage_image_sample(function_name, args[2]):
                    return self.default_value_for_type(result_type)
                image_operands = f" Sample %{args[2].id}"

            self.require_storage_image_without_format_capability(
                metadata, "StorageImageReadWithoutFormat"
            )
            return emit_four_component_image_read(
                self,
                result_type,
                image_id,
                coord_id,
                image_operands,
                metadata.get("component_count"),
            )

        if function_name == "imageStore":
            if len(args) < 3:
                self.refuse_lowering(
                    "imageStore requires image, coordinate, and value operands"
                )
                return None

            image_id, coord_id, texel_id = args[0], args[1], args[2]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.refuse_lowering("imageStore requires a storage image operand")
                return None
            if metadata.get("readonly"):
                self.refuse_lowering("imageStore requires a writable storage image")
                return None

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return None

            texel_type = self.resource_access_result_type(metadata)
            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 4:
                    self.refuse_lowering("imageStore requires a sample operand")
                    return None
                sample_id, texel_id = args[2], args[3]
                if not self.validate_storage_image_sample(function_name, sample_id):
                    return None
                image_operands = f" Sample %{sample_id.id}"

            texel_id = self.convert_storage_image_value_operand(
                function_name,
                "value",
                narrow_image_texel(self, texel_id, texel_type),
                texel_type,
            )
            if texel_id is None:
                return None

            self.require_storage_image_without_format_capability(
                metadata, "StorageImageWriteWithoutFormat"
            )
            self.emit(
                f"OpImageWrite %{image_id.id} %{coord_id.id} %{texel_id.id}"
                f"{image_operands}"
            )
            return None

        if function_name in {"texture", "texture2D", "textureCube"}:
            sample_args = self.sampled_texture_operands(function_name, args)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 1:
                self.refuse_lowering(
                    f"{function_name} accepts only texture, optional "
                    "sampler, coordinate, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if extra_args:
                bias_operand = self.texture_bias_operand(function_name, extra_args[0])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.default_lod_operand()}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id}"
                    f"{(' ' + bias_operand) if bias_operand else ''}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareOffset",
        }:
            extra_arg_count = {
                "textureCompare": 0,
                "textureCompareLod": 1,
                "textureCompareLodOffset": 2,
                "textureCompareGrad": 2,
                "textureCompareGradOffset": 3,
                "textureCompareOffset": 1,
            }[function_name]
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            if compare_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            result_type = self.register_primitive_type("float")
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)

            if function_name in {"textureCompareLod", "textureCompareLodOffset"}:
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {"textureCompareGrad", "textureCompareGradOffset"}:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
            }:
                offset_index = -1
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[offset_index]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if function_name == "textureCompare":
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {self.default_lod_operand()}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id}"
                    )
            elif function_name == "textureCompareOffset":
                offset_operand = self.image_offset_operand(extra_args[0])
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} "
                        f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {offset_operand}"
                    )
            elif function_name == "textureCompareLod":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Lod %{extra_args[0].id}"
                )
            elif function_name == "textureCompareLodOffset":
                offset_operand = self.image_offset_operand(extra_args[1])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Lod %{extra_args[0].id}', offset_operand)}"
                )
            elif function_name == "textureCompareGrad":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}"
                )
            else:
                offset_operand = self.image_offset_operand(extra_args[2])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Grad %{extra_args[0].id} %{extra_args[1].id}', offset_operand)}"
                )

            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            extra_arg_count = 1 if function_name == "textureGatherCompareOffset" else 0
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 4)
            if compare_args is None:
                return self.default_value_for_type(result_type)

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)
            if function_name == "textureGatherCompareOffset":
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operands = (
                f" {self.image_offset_operand(extra_args[0], gather=True)}"
                if function_name == "textureGatherCompareOffset"
                else ""
            )
            self.emit(
                f"%{id_value} = OpImageDrefGather %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id}"
                f"{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureOffset":
            sample_args = self.sampled_texture_operands(function_name, args, 1)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            offset_id = extra_args[0]

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 2:
                self.refuse_lowering(
                    "textureOffset accepts only texture, optional "
                    "sampler, coordinate, offset, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_offset_operand(
                function_name, metadata, offset_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if len(extra_args) >= 2:
                bias_operand = self.texture_bias_operand(function_name, extra_args[1])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            offset_operand = self.image_offset_operand(offset_id)
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(offset_operand, bias_operand)}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            required_extra_count = 0 if function_name == "textureGather" else 1
            sample_args = self.sampled_texture_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            int_type = self.register_primitive_type("int")
            result_type = self.resource_access_result_type(metadata)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureGather":
                if len(extra_args) > 1:
                    self.refuse_lowering(
                        "textureGather accepts only texture, optional "
                        "sampler, coordinate, and optional component operands"
                    )
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[0] if extra_args else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)
                offset_id = None
            elif function_name == "textureGatherOffsets":
                if len(extra_args) not in {1, 2, 4, 5}:
                    self.refuse_lowering(
                        self.sampled_texture_excess_operand_reason(function_name)
                    )
                    return self.default_value_for_type(result_type)
                return self.emit_texture_gather_offsets(
                    sampled_image_id,
                    coord_id,
                    extra_args,
                    metadata,
                    int_type,
                )
            else:
                if len(extra_args) > 2:
                    self.refuse_lowering(
                        "textureGatherOffset accepts only texture, "
                        "optional sampler, coordinate, offset, and optional "
                        "component operands"
                    )
                    return self.default_value_for_type(result_type)
                offset_id = extra_args[0]
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                ):
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[1]
                    if len(extra_args) >= 2
                    else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)

            return self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                offset_id,
            )

        texel_buffer_fetch = emit_texel_buffer_fetch(self, function_name, args)
        if texel_buffer_fetch is not None:
            return texel_buffer_fetch

        if function_name in {"texelFetch", "texelFetchOffset"}:
            required_extra_count = 1 if function_name == "texelFetch" else 2
            # A texel fetch reads the image without filtering, so the sampler it names is irrelevant.
            sample_args = self.texel_fetch_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            operand_id = extra_args[0]
            offset_id = extra_args[1] if function_name == "texelFetchOffset" else None
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > required_extra_count:
                self.refuse_lowering(
                    self.sampled_texture_excess_operand_reason(function_name)
                )
                return self.default_value_for_type(result_type)

            if metadata.get("dim") == "Cube":
                self.refuse_lowering(f"{function_name} is not valid for cube images")
                return self.default_value_for_type(result_type)

            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id, integer=True
            ):
                return self.default_value_for_type(result_type)
            if metadata.get("multisampled"):
                if not self.validate_sampled_texture_fetch_operand(
                    function_name, operand_id, "sample"
                ):
                    return self.default_value_for_type(result_type)
            elif not self.validate_sampled_texture_fetch_operand(
                function_name, operand_id, "LOD"
            ):
                return self.default_value_for_type(result_type)

            if metadata.get("multisampled") and offset_id is not None:
                self.refuse_lowering(
                    "texelFetchOffset is not valid for multisample images"
                )
                return self.default_value_for_type(result_type)
            if (
                offset_id is not None
                and not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                )
            ):
                return self.default_value_for_type(result_type)

            image_id = self.image_operand_for_query(sampled_image_id, metadata)
            if image_id is None:
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operand = (
                f"Sample %{operand_id.id}"
                if metadata.get("multisampled")
                else f"Lod %{operand_id.id}"
            )
            if offset_id is not None:
                image_operand = self.image_operands(
                    image_operand,
                    self.image_offset_operand(offset_id),
                )
            self.emit(
                f"%{id_value} = OpImageFetch %{result_type.id} "
                f"%{image_id.id} %{coord_id.id} {image_operand}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureSize", "imageSize"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image" if function_name == "textureSize" else "storage_image"
            )
            if not metadata or not self.image_query_kind_matches(
                metadata, expected_kind
            ):
                self.refuse_lowering(
                    f"{function_name} requires a {expected_kind} operand"
                )
                return self.resource_query_size_default_value(metadata)

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.resource_query_size_default_value(metadata)

            result_type = self.resource_query_size_result_type(metadata)
            if function_name == "imageSize" and len(args) > 1:
                self.refuse_lowering("imageSize accepts only an image operand")
                return self.default_value_for_type(result_type)

            if function_name == "textureSize" and len(args) > 2:
                self.refuse_lowering(
                    "textureSize accepts only texture and optional " "LOD operands"
                )
                return self.default_value_for_type(result_type)

            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not self.validate_resource_query_lod_operand(function_name, args[1])
            ):
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not metadata.get("multisampled")
            ):
                self.emit(
                    f"%{id_value} = OpImageQuerySizeLod %{result_type.id} "
                    f"%{image_id.id} %{args[1].id}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageQuerySize %{result_type.id} "
                    f"%{image_id.id}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureSamples", "imageSamples"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.refuse_lowering(f"{function_name} accepts only an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image"
                if function_name == "textureSamples"
                else "storage_image"
            )
            if not metadata or not self.image_query_kind_matches(
                metadata, expected_kind
            ):
                self.refuse_lowering(
                    f"{function_name} requires a {expected_kind} operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("dim") != "2D" or not metadata.get("multisampled"):
                self.refuse_lowering(f"{function_name} requires a multisample 2D image")
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQuerySamples %{result_type.id} "
                f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureQueryLevels":
            if not args:
                self.refuse_lowering("textureQueryLevels requires a texture operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.refuse_lowering(
                    "textureQueryLevels accepts only a texture operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            sampled_image_id = args[0]
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not metadata or not self.image_query_kind_matches(
                metadata, "sampled_image"
            ):
                self.refuse_lowering(
                    "textureQueryLevels requires a sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("multisampled"):
                self.refuse_lowering(
                    "textureQueryLevels requires a non-multisample "
                    "sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(sampled_image_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQueryLevels %{result_type.id} " f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureQueryLod":
            if len(args) < 2:
                self.refuse_lowering(
                    "textureQueryLod requires texture and coordinate operands"
                )
                return self.resource_query_lod_default_value()

            sampled_image_id = args[0]
            coord_id = args[1]
            sampler_id = None
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                sampler_id = args[1]
                if len(args) < 3:
                    self.refuse_lowering(
                        "textureQueryLod requires a coordinate operand "
                        "after the sampler"
                    )
                    return self.resource_query_lod_default_value()
                if len(args) > 3:
                    self.refuse_lowering(
                        "textureQueryLod accepts only texture, sampler, "
                        "and coordinate operands"
                    )
                    return self.resource_query_lod_default_value()
                coord_id = args[2]
            elif len(args) > 2:
                self.refuse_lowering(
                    "textureQueryLod accepts only texture and coordinate "
                    "operands unless the second operand is a sampler"
                )
                return self.resource_query_lod_default_value()

            metadata = self.resource_metadata_for_value(sampled_image_id)
            if metadata and metadata.get("kind") == "texture" and sampler_id is None:
                self.refuse_lowering(
                    "textureQueryLod requires a sampler for separate texture operands"
                )
                return self.resource_query_lod_default_value()
            if sampler_id is not None:
                # The computed level depends on the sampler's filters and LOD clamps.
                sampled_image_id = self.pair_resource_with_explicit_sampler(
                    function_name, sampled_image_id, sampler_id, metadata
                )
                if sampled_image_id is None:
                    return self.resource_query_lod_default_value()
                metadata = self.resource_metadata_for_value(sampled_image_id)

            if not metadata or metadata.get("kind") != "sampled_image":
                self.refuse_lowering("textureQueryLod requires a sampled image operand")
                return self.resource_query_lod_default_value()

            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.resource_query_lod_default_value()

            if not self.validate_resource_query_lod_coordinate(
                function_name, metadata, coord_id
            ):
                return self.resource_query_lod_default_value()

            coord_id = self.trim_image_query_lod_coordinate(coord_id, metadata)
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 2)
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if self.requires_explicit_lod_sampling():
                self.require_compute_derivatives()
            self.emit(
                f"%{id_value} = OpImageQueryLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
        }:
            required_arg_count = {
                "textureLod": 3,
                "textureLodOffset": 4,
                "textureGrad": 4,
                "textureGradOffset": 5,
            }[function_name]
            extra_arg_count = required_arg_count - 2
            sample_args = self.sampled_texture_operands(
                function_name, args, extra_arg_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > extra_arg_count:
                self.refuse_lowering(
                    self.sampled_texture_excess_operand_reason(function_name)
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureLod":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Lod %{extra_args[0].id}"
            elif function_name == "textureLodOffset":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[1]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Lod %{extra_args[0].id}",
                    self.image_offset_operand(extra_args[1]),
                )
            elif function_name == "textureGrad":
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Grad %{extra_args[0].id} %{extra_args[1].id}"
            else:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[2]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}",
                    self.image_offset_operand(extra_args[2]),
                )

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} {image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        return None

    def call_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Call a built-in function."""
        function_name = spirv_builtin_call_key(function_name)
        if function_name in PACK_FUNCTIONS:
            return spirv_pack_call(self, function_name, args)
        if function_name in BITFIELD_ARITIES:
            return spirv_bitfield_call(self, function_name, args)
        if function_name == "ldexp":
            return spirv_ldexp_call(self, args)
        synchronization_call = self.call_synchronization_function(function_name, args)
        if synchronization_call is not None:
            return synchronization_call

        bfloat16_bitcast = self.call_bfloat16_bitcast_function(function_name, args)
        if bfloat16_bitcast is not None:
            return bfloat16_bitcast

        explicit_bitcast = self.call_explicit_bitcast_function(function_name, args)
        if explicit_bitcast is not None:
            return explicit_bitcast

        scalar_constructor = self.call_scalar_constructor(function_name, args)
        if scalar_constructor is not None:
            return scalar_constructor

        derivative_call = self.call_derivative_function(function_name, args)
        if derivative_call is not None:
            return derivative_call

        bitcast_call = self.call_bitcast_builtin_function(function_name, args)
        if bitcast_call is not None:
            return bitcast_call

        integer_bit_call = self.call_integer_bit_builtin_function(function_name, args)
        if integer_bit_call is not None:
            return integer_bit_call

        bfloat16_math_call = self.call_bfloat16_math_function(function_name, args)
        if bfloat16_math_call is not None:
            return bfloat16_math_call

        float_semantics_call = self.call_float_semantics_function(function_name, args)
        if float_semantics_call is not None:
            return float_semantics_call

        math_polyfill_call = self.call_math_polyfill_function(function_name, args)
        if math_polyfill_call is not None:
            return math_polyfill_call

        matrix_builtin_call = self.call_matrix_builtin_function(function_name, args)
        if matrix_builtin_call is not None:
            return matrix_builtin_call

        exact_operand_counts = {
            "min": 2,
            "max": 2,
            "clamp": 3,
            "mix": 3,
            "pow": 2,
            "fma": 3,
            "faceforward": 3,
            "refract": 3,
            "length": 1,
            "distance": 2,
            "normalize": 1,
            "reflect": 2,
            "step": 2,
            "smoothstep": 3,
        }
        expected_operand_count = exact_operand_counts.get(function_name)
        if expected_operand_count is not None and len(args) != expected_operand_count:
            operand_label = "operand" if expected_operand_count == 1 else "operands"
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires "
                f"{expected_operand_count} {operand_label}",
            )

        if function_name == "mix" and len(args) == 3:
            bool_mix = self.call_bool_mix_function(args)
            if bool_mix is not None:
                return bool_mix

        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        if function_name == "saturate" and len(args) == 1:
            saturated = self.call_saturate_function(args[0])
            if saturated is not None:
                return saturated

        if function_name == "sign" and len(args) == 1:
            signed = self.call_sign_function(args[0])
            if signed is not None:
                return signed

        if function_name == "abs" and len(args) == 1:
            absolute = self.call_abs_function(args[0])
            if absolute is not None:
                return absolute
            raise self.builtin_lowering_error(
                function_name,
                "abs requires floating-point or signed integer "
                "scalar or vector operand",
            )

        # C/MSL fabs (C99 7.12.7.2, MSL 6.5) is the floating-point absolute value.
        if function_name == "fabs" and len(args) == 1:
            if self._float_shape(args[0]) is not None:
                return self.call_abs_function(args[0])
            raise self.builtin_lowering_error(
                function_name,
                "fabs requires a floating-point scalar or vector operand",
            )

        if function_name in {"and", "or"} and len(args) == 2:
            return lower_logical_intrinsic(self, function_name, args)

        # MSL select(a, b, c) (MSL 6.4) yields c ? b : a per component, like mix(a, b, bvec).
        if function_name == "select" and len(args) == 3:
            selected = self.call_bool_mix_function(args)
            if selected is not None:
                return selected
            raise self.builtin_lowering_error(
                function_name,
                "select requires matching value operands and a bool condition "
                "that is scalar or matches their component count",
            )

        if function_name == "pow" and len(args) == 2:
            power = self.call_pow_function(args)
            if power is not None:
                return power
            raise self.builtin_lowering_error(
                function_name,
                "pow requires compatible 32-bit floating-point "
                "scalar or vector operands",
            )

        if function_name == "fma" and len(args) == 3:
            fused = self.call_fma_function(args)
            if fused is not None:
                return fused
            raise self.builtin_lowering_error(
                function_name,
                "fma requires compatible floating-point scalar " "or vector operands",
            )

        if function_name == "faceforward" and len(args) == 3:
            faced = self.call_faceforward_function(args)
            if faced is not None:
                return faced
            raise self.builtin_lowering_error(
                function_name,
                "faceforward requires compatible floating-point "
                "scalar or vector operands",
            )

        if function_name == "refract" and len(args) == 3:
            refracted = self.call_refract_function(args)
            if refracted is not None:
                return refracted
            raise self.builtin_lowering_error(
                function_name,
                "refract requires matching floating-point incident "
                "and normal operands plus scalar eta",
            )

        if function_name == "length" and len(args) == 1:
            measured = self.call_length_function(args[0])
            if measured is not None:
                return measured
            raise self.builtin_lowering_error(
                function_name,
                "length requires floating-point scalar or vector operand",
            )

        if function_name == "distance" and len(args) == 2:
            measured = self.call_distance_function(args)
            if measured is not None:
                return measured
            raise self.builtin_lowering_error(
                function_name,
                "distance requires matching floating-point scalar "
                "or vector operands",
            )

        if function_name == "normalize" and len(args) == 1:
            normalized = self.call_normalize_function(args[0])
            if normalized is not None:
                return normalized
            raise self.builtin_lowering_error(
                function_name,
                "normalize requires floating-point scalar or vector operand",
            )

        if function_name == "reflect" and len(args) == 2:
            reflected = self.call_reflect_function(args)
            if reflected is not None:
                return reflected
            raise self.builtin_lowering_error(
                function_name,
                "reflect requires matching floating-point scalar " "or vector operands",
            )

        if function_name == "step" and len(args) == 2:
            stepped = self.call_step_smoothstep_function("Step", args, 1)
            if stepped is not None:
                return stepped
            raise self.builtin_lowering_error(
                function_name,
                "step requires compatible floating-point scalar " "or vector operands",
            )

        if function_name == "smoothstep" and len(args) == 3:
            smoothed = self.call_step_smoothstep_function("SmoothStep", args, 2)
            if smoothed is not None:
                return smoothed
            raise self.builtin_lowering_error(
                function_name,
                "smoothstep requires compatible floating-point "
                "scalar or vector operands",
            )

        if (
            function_name in {"min", "max"}
            and len(args) == 2
            or function_name == "clamp"
            and len(args) == 3
        ):
            min_max_clamp = self.call_min_max_clamp_function(function_name, args)
            if min_max_clamp is not None:
                return min_max_clamp
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires compatible scalar or "
                "vector numeric operands",
            )

        if function_name == "mix" and len(args) == 3:
            numeric_mix = self.call_numeric_mix_function(args)
            if numeric_mix is not None:
                return numeric_mix
            raise self.builtin_lowering_error(
                function_name,
                "mix requires compatible value operands and selector",
            )

        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info:
            component_type_name, component_count = vector_info
            component_type = self.register_primitive_type(component_type_name)
            vector_type = self.register_vector_type(component_type, component_count)
            if not args:
                if component_type_name == "bool":
                    zero_value = False
                    one_value = True
                elif component_type_name in {"int", "uint"}:
                    zero_value = 0
                    one_value = 1
                elif component_type_name == self.BFLOAT16_TYPE_NAME:
                    zero_value = 0
                    one_value = self.bfloat16_bits_for_float(1.0)
                else:
                    zero_value = 0.0
                    one_value = 1.0

                # Preserve old defaults: first component one, rest zero.
                component_zero = self.register_constant(zero_value, component_type)
                component_one = self.register_constant(one_value, component_type)

                default_args = [component_zero] * component_count
                if component_count > 0:
                    default_args[0] = component_one

                constructor_args = default_args
            else:
                constructor_args = self.flatten_vector_constructor_args(
                    function_name, args, component_type, component_count
                )

            return self.composite_construct(vector_type, constructor_args)

        if function_name in self.struct_types:
            struct_type = self.struct_types[function_name]
            if len(args) == 1:
                converted = self.convert_value_to_type(args[0], struct_type)
                if self.value_has_type(converted, struct_type):
                    return converted
            constructor_args = self.struct_constructor_args(function_name, args)
            return self.composite_construct(struct_type, constructor_args)

        matrix_function_name = self.normalize_hlsl_matrix_type(function_name)

        # Matrix constructors
        if re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name):
            match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name)
            is_double, cols, rows = match.groups()
            cols = int(cols)
            rows = int(rows or cols)

            component_type = self.register_primitive_type(
                "double" if is_double else "float"
            )
            vector_type = self.register_vector_type(component_type, rows)
            matrix_type = self.register_matrix_type(vector_type, cols)

            if not args:
                zero_value = self.register_constant(0.0, component_type)
                one_value = self.register_constant(1.0, component_type)

                col_vectors = []
                for col in range(cols):
                    col_components = []
                    for row in range(rows):
                        if col == row:
                            col_components.append(one_value)
                        else:
                            col_components.append(zero_value)

                    col_vectors.append(
                        self.composite_construct(vector_type, col_components)
                    )
            elif len(args) == 1 and (
                col_vectors := one_argument_matrix_columns(
                    self,
                    function_name,
                    args[0],
                    component_type,
                    vector_type,
                    cols,
                    rows,
                )
            ):
                pass
            else:
                components = self.flatten_matrix_constructor_components(
                    function_name, args, component_type, cols * rows
                )
                col_vectors = []
                for col in range(cols):
                    start = col * rows
                    col_vectors.append(
                        self.composite_construct(
                            vector_type, components[start : start + rows]
                        )
                    )

            return self.composite_construct(matrix_type, col_vectors)

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
            float_type = self.primitive_types["float"]
            glsl_std450_map = {
                "sin": "Sin",
                "cos": "Cos",
                "tan": "Tan",
                "atan2": "Atan2",
                "asin": "Asin",
                "acos": "Acos",
                "atan": "Atan",
                "sinh": "Sinh",
                "cosh": "Cosh",
                "tanh": "Tanh",
                "asinh": "Asinh",
                "acosh": "Acosh",
                "atanh": "Atanh",
                "exp": "Exp",
                "log": "Log",
                "exp2": "Exp2",
                "log2": "Log2",
                "sqrt": "Sqrt",
                "inversesqrt": "InverseSqrt",
                "abs": "FAbs",
                # Metal and C spell the floating-point absolute value fabs.
                "fabs": "FAbs",
                "floor": "Floor",
                "ceil": "Ceil",
                "fract": "Fract",
                "trunc": "Trunc",
                "round": "Round",
                "roundEven": "RoundEven",
                "degrees": "Degrees",
                "radians": "Radians",
                "length": "Length",
                "distance": "Distance",
                "cross": "Cross",
                "normalize": "Normalize",
                "reflect": "Reflect",
                "refract": "Refract",
            }
            std450_function_name = (
                "atan2" if function_name == "atan" and len(args) == 2 else function_name
            )
            if std450_function_name not in glsl_std450_map:
                if not CALLEE_IDENTIFIER.fullmatch(function_name):
                    raise self.builtin_lowering_error(
                        function_name, INDIRECT_CALL_DETAIL, reason="indirect-call"
                    )
                raise self.builtin_lowering_error(
                    function_name,
                    "it is neither a declared function nor a built-in with a "
                    "SPIR-V lowering",
                    reason="unknown-function",
                )

            unary_std450_functions = {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
                "exp",
                "log",
                "exp2",
                "log2",
                "sqrt",
                "inversesqrt",
                "abs",
                "floor",
                "ceil",
                "fract",
                "trunc",
                "round",
                "roundEven",
                "degrees",
                "radians",
                "length",
                "normalize",
            }
            std450_operand_counts = {name: 1 for name in unary_std450_functions}
            std450_operand_counts.update(
                {
                    "atan2": 2,
                    "distance": 2,
                    "cross": 2,
                    "reflect": 2,
                    "refract": 3,
                }
            )
            expected_operand_count = std450_operand_counts.get(std450_function_name)
            if (
                expected_operand_count is not None
                and len(args) != expected_operand_count
            ):
                operand_label = "operand" if expected_operand_count == 1 else "operands"
                raise self.builtin_lowering_error(
                    function_name,
                    f"{function_name} requires "
                    f"{expected_operand_count} {operand_label}",
                )

            vector_math_diagnostic = self.vector_math_operand_diagnostic(
                std450_function_name, args
            )
            if vector_math_diagnostic is not None:
                raise self.builtin_lowering_error(function_name, vector_math_diagnostic)

            result_type = float_type.type

            if args:
                if std450_function_name in [
                    "sin",
                    "cos",
                    "tan",
                    "atan2",
                    "asin",
                    "acos",
                    "atan",
                    "sinh",
                    "cosh",
                    "tanh",
                    "asinh",
                    "acosh",
                    "atanh",
                    "exp",
                    "log",
                    "exp2",
                    "log2",
                    "sqrt",
                    "inversesqrt",
                    "abs",
                    "floor",
                    "ceil",
                    "fract",
                    "trunc",
                    "round",
                    "roundEven",
                    "degrees",
                    "radians",
                ]:
                    # These functions return the same type as their first argument
                    result_type = args[0].type
                elif std450_function_name in ["length", "distance"]:
                    # These functions return the scalar component type.
                    result_type = self.metric_result_type(args).type
                elif std450_function_name in ["normalize", "reflect", "refract"]:
                    # These functions return the same vector type as their first argument
                    result_type = args[0].type
                elif std450_function_name in ["cross"]:
                    # cross product returns a vec3
                    component_type_name = "float"
                    vector_info = self.vector_component_type_and_count(
                        args[0].type.base_type
                    )
                    if vector_info is not None:
                        component_type_name = vector_info[0]
                    component_type = self.register_primitive_type(component_type_name)
                    vector_type = self.register_vector_type(component_type, 3)
                    result_type = vector_type.type

            result_type_id = self.ensure_registered_type(result_type)
            return self.emit_glsl_std450_instruction(
                glsl_std450_map[std450_function_name], result_type_id, args
            )

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
            if expr.rsplit("::", 1)[-1] == "None":
                none_default = self.option_none_default_value()
                if none_default is not None:
                    return none_default

            if self.enum_variant_is_payload_path(expr):
                enum_value = self.process_enum_variant_constructor(expr, [])
                if enum_value is not None:
                    return enum_value
            else:
                enum_constant = self.enum_variant_constant(expr)
                if enum_constant is not None:
                    return enum_constant

            if expr in self.local_variables:
                var_id = self.local_variables[expr]
                return self.get_variable_value(var_id)
            generic_literal = self.current_generic_literal_int_constants().get(expr)
            if generic_literal is not None:
                primitive_type = self.integer_literal_type_for_value(
                    "int", generic_literal
                )
                return self.register_constant(
                    generic_literal,
                    self.register_primitive_type(primitive_type),
                )
            elif expr in self.named_constants:
                return self.named_constants[expr]
            else:
                var_id = self.resolve_global_variable(expr)
                if var_id is not None:
                    self.mark_interface_variable_if_needed(var_id)
                    self.mark_builtin_interface_variable(var_id)
                    return self.get_variable_value(var_id)

                if expr in self.cbuffer_members:
                    member_pointer = self.cbuffer_member_pointer(expr)
                    if member_pointer is not None:
                        return self.get_variable_value(member_pointer)

                patch_input = self.ensure_tessellation_patch_constant_input(expr)
                if patch_input is not None:
                    return self.get_variable_value(patch_input)

                builtin_component = self.process_dotted_builtin(expr)
                if builtin_component is not None:
                    return builtin_component

                ray_constant = ray_constant_value(expr)
                if ray_constant is not None:
                    return self.register_constant(
                        ray_constant, self.register_primitive_type("uint")
                    )

                transposed = transposed_ray_builtin(expr)
                if transposed is not None:
                    # gl_ObjectToWorld3x4EXT and gl_WorldToObject3x4EXT transpose a built-in (spirv_ray_builtins).
                    return self.process_expression(
                        FunctionCallNode(
                            IdentifierNode("transpose"), [IdentifierNode(transposed)]
                        )
                    )

                builtin = self.ensure_builtin_variable(expr)
                if builtin is not None:
                    return self.get_variable_value(builtin)

                builtin_limit = self.glsl_builtin_limit_constant(expr)
                if builtin_limit is not None:
                    return builtin_limit

                standard_constant = spirv_standard_float_constant(self, expr)
                if standard_constant is not None:
                    return standard_constant

                # Create a default float constant for missing variables in examples
                # This is to make the SPIR-V code valid even if we can't find the variable
                if expr.replace(".", "", 1).isdigit():  # Check if it's a numeric string
                    float_type = self.register_primitive_type("float")
                    try:
                        value = float(expr)
                        return self.register_constant(value, float_type)
                    except ValueError:
                        pass

                self.refuse_lowering(f"Unknown variable {expr}")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, LiteralNode):
            literal_type = self.convert_type_node_to_string(expr.literal_type)
            primitive_type_name = self.normalize_primitive_name(literal_type)
            if primitive_type_name in {"float", "double"}:
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(float(expr.value), literal_type_id)
            if primitive_type_name in self.INTEGER_TYPE_NAMES:
                literal_value = int(expr.value)
                if (
                    primitive_type_name in self.SIGNED_INTEGER_TYPES
                    and self.expected_primitive_type_name()
                    in self.UNSIGNED_INTEGER_TYPES
                    and literal_value >= 0
                ):
                    primitive_type_name = self.expected_primitive_type_name()
                primitive_type_name = self.integer_literal_type_for_value(
                    primitive_type_name, literal_value
                )
                literal_type_id = self.register_primitive_type(primitive_type_name)
                return self.register_constant(literal_value, literal_type_id)
            if primitive_type_name == "bool":
                literal_type_id = self.register_primitive_type("bool")
                if isinstance(expr.value, str):
                    value = expr.value.lower() == "true"
                else:
                    value = bool(expr.value)
                return self.register_constant(value, literal_type_id)
            return self.process_expression(expr.value)

        elif isinstance(expr, IdentifierNode):
            return self.process_expression(expr.name)

        elif isinstance(expr, VariableNode):
            if expr.name in self.local_variables:
                var_id = self.local_variables[expr.name]
                return self.get_variable_value(var_id)
            elif expr.name in self.named_constants:
                return self.named_constants[expr.name]
            else:
                var_id = self.resolve_global_variable(expr.name)
                if var_id is not None:
                    self.mark_interface_variable_if_needed(var_id)
                    self.mark_builtin_interface_variable(var_id)
                    return self.get_variable_value(var_id)

                if expr.name in self.cbuffer_members:
                    member_pointer = self.cbuffer_member_pointer(expr.name)
                    if member_pointer is not None:
                        return self.get_variable_value(member_pointer)

                patch_input = self.ensure_tessellation_patch_constant_input(expr.name)
                if patch_input is not None:
                    return self.get_variable_value(patch_input)

                builtin = self.ensure_builtin_variable(expr.name)
                if builtin is not None:
                    return self.get_variable_value(builtin)

                builtin_limit = self.glsl_builtin_limit_constant(expr.name)
                if builtin_limit is not None:
                    return builtin_limit

                standard_constant = spirv_standard_float_constant(self, expr.name)
                if standard_constant is not None:
                    return standard_constant

                self.refuse_lowering(f"Unknown variable {expr.name}")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

        elif isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr)

        elif isinstance(expr, ArrayAccessNode):
            index = evaluate_subscript_index(self, expr.index)

            if index is None:
                self.refuse_lowering(f"Failed to evaluate array access")
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            reinterpret_load = self.process_storage_pointer_reinterpret_load(
                expr.array, index
            )
            if reinterpret_load is not None:
                return reinterpret_load
            component = vector_value_subscript(self, expr.array, index, expr.index)
            if component is not None:
                return component

            access, element_type = self.create_array_element_access(
                expr.array, index, expr.index
            )

            if access is None or element_type is None:
                self.refuse_lowering(
                    "Could not determine array element type for "
                    f"{self.diagnostic_expression(expr.array)}"
                )
                element_type = self.primitive_types["float"]
                return self.register_constant(0.0, element_type)

            return self.load_from_variable(access, element_type)

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
            if expr.op == ",":
                if isinstance(expr.left, AssignmentNode):
                    self.process_assignment(expr.left)
                else:
                    self.process_expression(expr.left)
                return self.process_expression(expr.right)
            if expr.op in SHORT_CIRCUIT_OPERATORS:
                short_circuit = lower_short_circuit_logical(self, expr)
                if short_circuit is not None:
                    return short_circuit
            if expr.op in {"&", "|", "^"}:
                expected_type = self.bitwise_expression_operand_type(
                    self.infer_expression_result_type(expr.left),
                    self.infer_expression_result_type(expr.right),
                )
                if expected_type is not None:
                    left = self.process_expression_with_expected_type(
                        expr.left, expected_type
                    )
                    right = self.process_expression_with_expected_type(
                        expr.right, expected_type
                    )
                else:
                    left = self.process_expression(expr.left)
                    right = self.process_expression(expr.right)
            else:
                # A literal operand takes the other operand's type, or its own when that is unknown, never the
                # enclosing expression's: in uint(v % 3) and (v > 0) ? 1u : 0u the literal stays int, so the
                # operation stays signed. Operations without a literal operand keep the enclosing context, which
                # loads and constructors read.
                previous_expected_type = self.current_expression_expected_type
                if is_literal_operand(expr.left) or is_literal_operand(expr.right):
                    operand_type = self.bitwise_expression_operand_type(
                        self.infer_expression_result_type(expr.left),
                        self.infer_expression_result_type(expr.right),
                    )
                    self.current_expression_expected_type = (
                        self.type_name_string(operand_type)
                        if operand_type is not None
                        else None
                    )
                try:
                    left = self.process_expression(expr.left)
                    right = self.process_expression(expr.right)
                finally:
                    self.current_expression_expected_type = previous_expected_type

            if left is None or right is None:
                self.refuse_lowering(
                    f"Operand of {expr.op!r} has no value: "
                    f"{self.diagnostic_expression(expr.left if left is None else expr.right)}"
                )
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            left_type = self.registered_value_type(left) or self.ensure_registered_type(
                left.type
            )
            right_type = self.registered_value_type(
                right
            ) or self.ensure_registered_type(right.type)
            result_type = self.binary_expression_result_type(
                expr.op, left_type, right_type
            )
            if result_type is None:
                result_type = left_type

            return self.binary_operation(expr.op, result_type, left, right)

        elif isinstance(expr, UnaryOpNode):
            if expr.op in {"++", "--"}:
                return self.process_increment_expression(expr)
            if expr.op == "*" and not getattr(expr, "is_postfix", False):
                pointer = self.dereference_pointer_from_expression(expr.operand)
                value_type = (
                    self.variable_value_types.get(pointer.id)
                    if pointer is not None
                    else None
                )
                if pointer is None or value_type is None:
                    raise self.unsupported_pointer_dereference_error(expr)
                return self.load_from_variable(pointer, value_type)

            operand = self.process_expression(expr.operand)
            if operand is None:
                self.refuse_lowering(
                    f"Operand of {expr.op!r} has no value: "
                    f"{self.diagnostic_expression(expr.operand)}"
                )
                float_type = self.register_primitive_type("float")
                return self.register_constant(0.0, float_type)

            return self.unary_operation(expr.op, operand.type, operand)

        elif isinstance(expr, TernaryOpNode):
            short_circuit = lower_short_circuit_ternary(self, expr)
            if short_circuit is not None:
                return short_circuit
            condition = self.process_expression(expr.condition)
            true_value = self.process_expression(expr.true_expr)
            false_value = self.process_expression(expr.false_expr)

            if condition is None:
                self.refuse_lowering(
                    "Conditional operator condition has no value: "
                    f"{self.diagnostic_expression(expr.condition)}"
                )
                condition = self.register_constant(
                    False, self.register_primitive_type("bool")
                )
            if true_value is None or false_value is None:
                self.refuse_lowering(
                    "Conditional operator branch has no value: "
                    + self.diagnostic_expression(
                        expr.true_expr if true_value is None else expr.false_expr
                    )
                )
                float_type = self.register_primitive_type("float")
                fallback = self.register_constant(0.0, float_type)
                true_value = true_value or fallback
                false_value = false_value or fallback

            result_type, true_value, false_value = self.normalize_ternary_values(
                true_value, false_value
            )
            return self.select_operation(
                result_type, condition, true_value, false_value
            )

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
            reject_unresolved_generic_member_call(expr, "Vulkan SPIR-V")
            ray_query_call = self.ray_query_call_from_function_call(expr)
            if ray_query_call is not None:
                return self.process_ray_query_operation(ray_query_call)

            handled_byte_address_call, byte_address_result = (
                self.process_byte_address_buffer_method_call(expr)
            )
            if handled_byte_address_call:
                return byte_address_result
            handled_byte_address_call, byte_address_result = (
                process_byte_address_atomic_call(self, expr)
            )
            if handled_byte_address_call:
                return byte_address_result

            handled_length_call, length_result = self.process_block_member_length_call(
                expr
            )
            if handled_length_call:
                return length_result
            length_object = length_method_object(
                getattr(expr, "function", None), getattr(expr, "args", None) or []
            )
            matrix_shape = (
                crossgl_matrix_shape(self.expression_result_type(length_object))
                if length_object is not None
                else None
            )
            if matrix_shape is not None:
                # A matrix's length() is its column count (GLSL 4.60 section 5.10).
                return self.register_constant(
                    matrix_shape[1], self.register_primitive_type("int")
                )

            handled_structured_buffer_call, structured_buffer_result = (
                self.process_structured_buffer_method_call(expr)
            )
            if handled_structured_buffer_call:
                return structured_buffer_result

            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = None
            if hasattr(callee_expr, "name"):
                callee_name = callee_expr.name
            elif isinstance(callee_expr, str):
                callee_name = callee_expr

            if self.is_canonical_atomic_thread_fence_call(callee_name):
                return self.reject_atomic_thread_fence(expr)

            instance_constructor = self.process_generic_struct_constructor_call(
                callee_name, expr
            )
            if instance_constructor is not None:
                return instance_constructor

            if (
                isinstance(callee_name, str)
                and callee_name.rsplit("::", 1)[-1] == "Some"
            ):
                payload_type = self.option_or_expected_payload_type_name()
                if payload_type is not None and not expr.args:
                    return self.default_value_for_type(
                        self.map_crossgl_type(payload_type)
                    )
                if payload_type is not None:
                    payload_value = self.process_expression_with_expected_type(
                        expr.args[0],
                        payload_type,
                    )
                    if payload_value is None:
                        return None
                    return self.convert_value_to_type(
                        payload_value,
                        self.map_crossgl_type(payload_type),
                    )

            if isinstance(callee_name, str) and callee_name.lower() in {
                "nonuniformext",
                "nonuniformresourceindex",
            }:
                # HLSL's NonUniformResourceIndex and GLSL's nonuniformEXT both mark a divergent descriptor index.
                return self.process_non_uniform_function_call(callee_name, expr.args)

            if (
                callee_name in DEBUG_PRINTF_FUNCTIONS
                and not self.has_function_reference(callee_name)
            ):
                return spirv_debug_printf_call(self, callee_name, expr.args, SpirvId)

            if callee_name in SPIRV_SHADER_CLOCKS and not self.has_function_reference(
                callee_name
            ):
                return spirv_shader_clock_call(
                    self, callee_name, list(expr.args), SpirvId
                )

            if callee_name == TEXTURE_RESIDENCY_OPERATION and not (
                self.has_function_reference(callee_name)
            ):
                return spirv_texture_residency_call(self, list(expr.args), expr)
            if callee_name == TEXTURE_RESIDENT_OPERATION and not (
                self.has_function_reference(callee_name)
            ):
                return spirv_texture_resident_call(self, list(expr.args), expr)

            if (
                callee_name in OUTPUT_OPERAND_FUNCTIONS
                and not self.has_function_reference(callee_name)
            ):
                return spirv_output_operand_call(self, callee_name, list(expr.args))

            if (
                callee_name in EXTENDED_ARITHMETIC_FUNCTIONS
                and not self.has_function_reference(callee_name)
            ):
                return spirv_extended_arithmetic_call(
                    self, callee_name, list(expr.args)
                )

            if callee_name in {"SetVertex", "SetPrimitive"}:
                return self.process_mesh_output_function_call(callee_name, expr.args)

            if callee_name in {"EmitVertex", "EndPrimitive", *STREAM_FUNCTIONS}:
                return self.process_geometry_stream_function_call(
                    callee_name, expr.args
                )

            if (
                isinstance(callee_name, str)
                and self.enum_path_parts(callee_name) is not None
            ):
                enum_value = self.process_enum_variant_constructor(
                    callee_name,
                    list(getattr(expr, "args", []) or []),
                )
                if enum_value is not None:
                    return enum_value

            if callee_name == "lambda":
                return self.unsupported_lambda_default_value("lambda expression")

            if callee_name == "buffer_dimensions":
                return self.process_buffer_dimensions_function_call(expr)

            if any(self.contains_lambda_expression(arg) for arg in expr.args):
                result_type = None
                function_signature = self.resolve_function_signature(callee_name)
                if function_signature is not None:
                    result_type = function_signature[0]
                return self.unsupported_lambda_default_value(
                    f"call to {callee_name or 'unknown callee'}",
                    result_type,
                )

            inline_storage_buffer_function = (
                self.resolve_inline_storage_buffer_function(
                    callee_name, expr.args, call_node=expr
                )
            )
            if inline_storage_buffer_function is not None:
                return self.inline_storage_buffer_function_call(
                    inline_storage_buffer_function, expr.args, call_node=expr
                )

            generic_function_definition = self.generic_function_definitions.get(
                generic_function_definition_name(self, callee_name)
            )
            if (
                generic_function_definition is not None
                and not self.has_matching_concrete_function_candidate(
                    callee_name, expr.args
                )
            ):
                specialized_callee_name = generic_function_call_name(
                    self, callee_name, expr.args
                )
                if specialized_callee_name is None:
                    if isinstance(generic_function_definition, list):
                        generic_function_definition = generic_function_definition[0]
                    raise self.unsupported_generic_function_error(
                        generic_function_definition, call_node=expr
                    )
                call_args = generic_function_value_arguments(
                    self,
                    callee_name,
                    expr.args,
                )
                callee_name = specialized_callee_name
            else:
                call_args = list(expr.args)

            args = []
            has_errors = False
            skipped_arg_indices = self.skipped_function_parameter_indices(callee_name)
            mesh_output_arg_indices = (
                self.resolve_function_mesh_output_parameter_indices(callee_name)
            )
            contextual_parameter_types = {}
            ambiguous_contextual_arguments = set()
            selected_function_reference = None
            temporary_array_arguments = set()
            if any(isinstance(arg, ArrayLiteralNode) for arg in call_args):
                (
                    contextual_parameter_types,
                    ambiguous_contextual_arguments,
                    selected_function_reference,
                    temporary_array_arguments,
                ) = self.contextual_call_parameter_types(
                    callee_name,
                    call_args,
                    skipped_arg_indices,
                    mesh_output_arg_indices,
                )
            for arg_index, arg in enumerate(call_args):
                if arg_index in skipped_arg_indices:
                    continue
                if arg_index in mesh_output_arg_indices:
                    continue
                if self.expression_contains_empty_array_literal(arg):
                    if arg_index in ambiguous_contextual_arguments:
                        raise self.unsupported_call_initializer_type_error(
                            callee_name, arg_index, arg, ambiguous=True
                        )
                    if arg_index not in contextual_parameter_types:
                        raise self.unsupported_call_initializer_type_error(
                            callee_name, arg_index, arg, ambiguous=False
                        )
                arg_value = self.process_call_argument(
                    callee_name,
                    arg,
                    arg_index,
                    expected_type=contextual_parameter_types.get(arg_index),
                    allow_array_temporary=arg_index in temporary_array_arguments,
                )
                if arg_value is None:
                    self.refuse_lowering(
                        f"Failed to evaluate argument for {callee_name or callee_expr}"
                    )
                    has_errors = True
                    float_type = self.register_primitive_type("float")
                    arg_value = self.register_constant(0.0, float_type)
                args.append(arg_value)

            for stage_object_name in self.required_function_stage_object_argument_names(
                callee_name
            ):
                stage_object = self.local_variables.get(stage_object_name)
                if stage_object is None:
                    self.refuse_lowering(
                        "Failed to evaluate stage object argument "
                        f"{stage_object_name} for {callee_name or callee_expr}"
                    )
                    has_errors = True
                    continue
                args.append(stage_object)

            if has_errors and callee_name == "vec2":
                float_type = self.register_primitive_type("float")
                vector_type = self.register_vector_type(float_type, 2)
                id_value = self.get_id()

                while len(args) < 2:
                    args.append(self.register_constant(0.0, float_type))

                arg_list = " ".join([f"%{arg.id}" for arg in args[:2]])
                self.emit(
                    f"%{id_value} = OpCompositeConstruct %{vector_type.id} {arg_list}"
                )
                return SpirvId(id_value, vector_type.type)

            if callee_name is None:
                raise SPIRVBuiltinLoweringError(
                    "<expression>",
                    INDIRECT_CALL_DETAIL,
                    reason="indirect-call",
                    source_location=getattr(expr, "source_location", None),
                )

            if not self.validate_function_image_access_arguments(
                callee_name, expr.args
            ):
                result_type = None
                function_signature = self.resolve_function_signature(callee_name)
                if function_signature is not None:
                    result_type = function_signature[0]
                if result_type is None or result_type.type.base_type == "void":
                    return None
                return self.default_value_for_type(result_type)

            if self.is_resource_type_name(callee_name):
                constructed_resource = self.call_resource_constructor(callee_name, args)
                if constructed_resource is not None:
                    return constructed_resource

            if (
                callee_name in self.resource_function_names()
                and not self.has_function_reference(callee_name)
            ):
                return self.call_resource_function(callee_name, args)

            return self.call_function(
                callee_name,
                args,
                function_reference=selected_function_reference,
            )

        elif isinstance(expr, MemberAccessNode):
            member_name = expr.member
            base_pointer = self.variable_pointer_from_expression(expr.object)
            if base_pointer is not None:
                access = self.create_member_access_pointer(base_pointer, member_name)
                if access is not None:
                    member_type = self.variable_value_types.get(access.id)
                    return self.load_from_variable(access, member_type)

            base = self.process_expression(expr.object)
            if base is None:
                return None

            struct_type = base.type.base_type
            member_info = self.struct_member_info(struct_type, member_name)
            if member_info is not None:
                member_index, member_type = member_info
                return self.composite_extract(base, member_type, member_index)

            member_info = self.vector_member_info(struct_type, member_name)
            if member_info is not None:
                member_index, member_type = member_info
                return self.composite_extract(base, member_type, member_index)

            swizzle_info = self.vector_swizzle_info(struct_type, member_name)
            if swizzle_info is not None:
                indices, _, result_type = swizzle_info
                return self.vector_shuffle(base, result_type, indices)

            if (
                self.vector_component_type_and_count(struct_type) is not None
                and len(member_name) > 1
            ):
                self.refuse_lowering(
                    f"Invalid vector swizzle {member_name} for {struct_type}"
                )
                return None

            selected = self.scalar_swizzle(base, member_name)
            if selected is not None:
                return selected

            self.refuse_lowering(
                f"Could not find member {member_name} in {struct_type}"
            )
            return None

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
