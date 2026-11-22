"""CrossGL-to-Mojo code generator."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    AtomicOpNode,
    BinaryOpNode,
    BufferOpNode,
    BuiltinVariableNode,
    CastNode,
    ConstructorNode,
    EnumNode,
    FunctionCallNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    StructNode,
    SwizzleNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..float_limit_constants import float_limit_constant
from ..refusals import MojoRefusal
from ..semantics.declared_types import declared_type_names
from ..unhandled import UNHANDLED
from .array_utils import collect_literal_int_constants
from .comma_expressions import is_comma_expression, refuse_comma_expression
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .enum_utils import type_node_contains_generic_parameter
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
    iter_function_nodes,
    prepare_generic_function_specializations,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .mojo_arithmetic_conversions import MojoArithmeticConversionMixin
from .mojo_buffer_objects import MojoBufferObjectMixin
from .mojo_builtin_calls import MojoBuiltinCallMixin
from .mojo_call_expressions import MojoCallExpressionMixin
from .mojo_expression_nodes import MojoExpressionNodeMixin
from .mojo_expressions import MojoExpressionMixin
from .mojo_functions import MojoFunctionMixin
from .mojo_generic_types import MojoGenericTypeMixin
from .mojo_kernel_arguments import MojoKernelArgumentMixin
from .mojo_match_lowering import MojoMatchMixin
from .mojo_math_helpers import MojoMathHelperMixin
from .mojo_pointer_expressions import MojoPointerExpressionMixin
from .mojo_private_globals import MojoPrivateGlobalMixin
from .mojo_refusals import (  # noqa: F401  (registers the Mojo refusal)
    MojoLoweringRefused,
)
from .mojo_resource_access import MojoResourceAccessMixin
from .mojo_resource_calls import MojoResourceCallMixin
from .mojo_resource_helpers import MojoResourceHelperMixin
from .mojo_resource_types import MojoResourceTypeMixin
from .mojo_result_types import MojoResultTypeMixin
from .mojo_return_aliases import MojoReturnAliasMixin
from .mojo_semantics import MojoSemanticMixin
from .mojo_shared_memory import MojoSharedMemoryMixin
from .mojo_split_builtins import MojoSplitBuiltinMixin
from .mojo_stage_contracts import MojoStageContractMixin
from .mojo_statements import MojoStatementMixin
from .mojo_storage_blocks import lower_mojo_single_array_storage_blocks
from .mojo_tables import (
    MOJO_ATOMIC_OP_ALIASES,
    MOJO_BUFFER_COUNTER_METHODS,
    MOJO_BUFFER_LOAD_RESOURCE_TYPES,
    MOJO_BUFFER_OP_ALIASES,
    MOJO_BUFFER_RESOURCE_TYPES,
    MOJO_BUFFER_STORE_RESOURCE_TYPES,
    MOJO_BUILTIN_PLACEHOLDERS,
    MOJO_BYTE_ADDRESS_ATOMIC_METHODS,
    MOJO_BYTE_ADDRESS_BUFFER_TYPES,
    MOJO_BYTE_ADDRESS_LOAD_METHODS,
    MOJO_BYTE_ADDRESS_STORE_METHODS,
    MOJO_CONCRETE_RESOURCE_TYPES,
    MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES,
    MOJO_DTYPE_INFO,
    MOJO_DTYPE_SUFFIX,
    MOJO_FLOAT_DTYPES,
    MOJO_GENERIC_TEXTURE_BUILTINS,
    MOJO_GPU_BUILTIN_ALIASES,
    MOJO_GPU_BUILTIN_PARAMETER_ORDER,
    MOJO_HLSL_BUFFER_TYPE_ALIASES,
    MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES,
    MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING,
    MOJO_IMAGE_ATOMIC_BUILTINS,
    MOJO_IMAGE_RESOURCE_BASE_TYPES,
    MOJO_INPUT_ONLY_RETURN_SEMANTICS,
    MOJO_INTEGER_DTYPES,
    MOJO_INTEGER_INDEX_TYPES,
    MOJO_MAPPED_INTEGER_TYPES,
    MOJO_MATH_HELPERS,
    MOJO_MATRIX_TYPES,
    MOJO_MESH_STAGE_TYPES,
    MOJO_NON_SEMANTIC_ATTRIBUTES,
    MOJO_RAY_STAGE_TYPES,
    MOJO_REINTERPRET_ALIAS_MAP,
    MOJO_REINTERPRET_BUILTINS,
    MOJO_REINTERPRET_TARGET_TYPES,
    MOJO_RESERVED_RESOURCE_TYPE_NAMES,
    MOJO_RESOURCE_ACCESS_QUALIFIERS,
    MOJO_RESOURCE_COMPARE_COORDS,
    MOJO_RESOURCE_MEMORY_QUALIFIERS,
    MOJO_RESOURCE_OFFSET_COORDS,
    MOJO_RESOURCE_OPERATION_MEMBER_REQUIREMENTS,
    MOJO_RESOURCE_PROJECTED_COORDS,
    MOJO_RESOURCE_QUERY_MEMBER_REQUIREMENTS,
    MOJO_RESOURCE_SAMPLE_COORDS,
    MOJO_RESOURCE_SIZE_RETURNS,
    MOJO_RESOURCE_TEXEL_COORDS,
    MOJO_RESOURCE_TYPE_MAPPING,
    MOJO_SCALAR_ATOMIC_OPS,
    MOJO_SCALAR_ATOMIC_TYPES,
    MOJO_SCALAR_DTYPES,
    MOJO_SUPPORTED_STAGE_TYPES,
    MOJO_SYNC_BUILTINS,
    MOJO_TEXTURE_RESOURCE_BASE_TYPES,
    MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES,
    MOJO_TYPED_BUFFER_RESOURCE_TYPES,
    MOJO_TYPED_IMAGE_DTYPE_SUFFIX,
    MOJO_TYPED_IMAGE_SUFFIX_INFO,
    MOJO_TYPED_TEXTURE_DTYPE_SUFFIX,
    MOJO_TYPED_TEXTURE_SUFFIX_INFO,
    MOJO_VECTOR_ARITHMETIC_OPS,
    MOJO_VECTOR_COMPARISON_METHODS,
    MOJO_VECTOR_TYPES,
    MOJO_WAVE_BOOL_RESULT_OPS,
    MOJO_WAVE_OP_ARITIES,
    MOJO_WAVE_UINT_RESULT_OPS,
    MOJO_WAVE_UVEC4_RESULT_OPS,
    ROUND_EVEN_FLOAT_DTYPES,
    SWIZZLE_SETS,
    MojoScalarConversionError,
)
from .mojo_target_shapes import MojoTargetShapeMixin
from .mojo_type_declarations import MojoTypeDeclarationMixin
from .mojo_types import MojoTypeMixin
from .mojo_values import MojoValueMixin
from .mojo_vector_ops import MojoVectorOpMixin
from .mojo_wave_ops import MOJO_WAVE_IMPORTS, MojoWaveOpMixin
from .precise_evaluation import refuse_precise_evaluation
from .sampler_pairs import canonicalize_separate_textures
from .written_function_names import (
    MOJO_CALLABLE_FUNCTION_NAMES,
    MOJO_NON_CALLS,
    refuse_written_foreign_calls,
)

__all__ = [
    "MOJO_ATOMIC_OP_ALIASES",
    "MOJO_BUFFER_COUNTER_METHODS",
    "MOJO_BUFFER_LOAD_RESOURCE_TYPES",
    "MOJO_BUFFER_OP_ALIASES",
    "MOJO_BUFFER_RESOURCE_TYPES",
    "MOJO_BUFFER_STORE_RESOURCE_TYPES",
    "MOJO_BUILTIN_PLACEHOLDERS",
    "MOJO_BYTE_ADDRESS_ATOMIC_METHODS",
    "MOJO_BYTE_ADDRESS_BUFFER_TYPES",
    "MOJO_BYTE_ADDRESS_LOAD_METHODS",
    "MOJO_BYTE_ADDRESS_STORE_METHODS",
    "MOJO_CONCRETE_RESOURCE_TYPES",
    "MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES",
    "MOJO_DTYPE_INFO",
    "MOJO_DTYPE_SUFFIX",
    "MOJO_FLOAT_DTYPES",
    "MOJO_GENERIC_TEXTURE_BUILTINS",
    "MOJO_GPU_BUILTIN_ALIASES",
    "MOJO_GPU_BUILTIN_PARAMETER_ORDER",
    "MOJO_HLSL_BUFFER_TYPE_ALIASES",
    "MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES",
    "MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING",
    "MOJO_IMAGE_ATOMIC_BUILTINS",
    "MOJO_IMAGE_RESOURCE_BASE_TYPES",
    "MOJO_INPUT_ONLY_RETURN_SEMANTICS",
    "MOJO_INTEGER_DTYPES",
    "MOJO_INTEGER_INDEX_TYPES",
    "MOJO_MAPPED_INTEGER_TYPES",
    "MOJO_MATH_HELPERS",
    "MOJO_MATRIX_TYPES",
    "MOJO_MESH_STAGE_TYPES",
    "MOJO_NON_SEMANTIC_ATTRIBUTES",
    "MOJO_RAY_STAGE_TYPES",
    "MOJO_REINTERPRET_ALIAS_MAP",
    "MOJO_REINTERPRET_BUILTINS",
    "MOJO_REINTERPRET_TARGET_TYPES",
    "MOJO_RESERVED_RESOURCE_TYPE_NAMES",
    "MOJO_RESOURCE_ACCESS_QUALIFIERS",
    "MOJO_RESOURCE_COMPARE_COORDS",
    "MOJO_RESOURCE_MEMORY_QUALIFIERS",
    "MOJO_RESOURCE_OFFSET_COORDS",
    "MOJO_RESOURCE_OPERATION_MEMBER_REQUIREMENTS",
    "MOJO_RESOURCE_PROJECTED_COORDS",
    "MOJO_RESOURCE_QUERY_MEMBER_REQUIREMENTS",
    "MOJO_RESOURCE_SAMPLE_COORDS",
    "MOJO_RESOURCE_SIZE_RETURNS",
    "MOJO_RESOURCE_TEXEL_COORDS",
    "MOJO_RESOURCE_TYPE_MAPPING",
    "MOJO_SCALAR_ATOMIC_OPS",
    "MOJO_SCALAR_ATOMIC_TYPES",
    "MOJO_SCALAR_DTYPES",
    "MOJO_SUPPORTED_STAGE_TYPES",
    "MOJO_SYNC_BUILTINS",
    "MOJO_TEXTURE_RESOURCE_BASE_TYPES",
    "MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES",
    "MOJO_TYPED_BUFFER_RESOURCE_TYPES",
    "MOJO_TYPED_IMAGE_DTYPE_SUFFIX",
    "MOJO_TYPED_IMAGE_SUFFIX_INFO",
    "MOJO_TYPED_TEXTURE_DTYPE_SUFFIX",
    "MOJO_TYPED_TEXTURE_SUFFIX_INFO",
    "MOJO_VECTOR_ARITHMETIC_OPS",
    "MOJO_VECTOR_COMPARISON_METHODS",
    "MOJO_VECTOR_TYPES",
    "MOJO_WAVE_BOOL_RESULT_OPS",
    "MOJO_WAVE_OP_ARITIES",
    "MOJO_WAVE_UINT_RESULT_OPS",
    "MOJO_WAVE_UVEC4_RESULT_OPS",
    "MojoCodeGen",
    "MojoScalarConversionError",
    "ROUND_EVEN_FLOAT_DTYPES",
    "SWIZZLE_SETS",
]


class MojoCodeGen(
    MojoKernelArgumentMixin,
    MojoBufferObjectMixin,
    MojoPrivateGlobalMixin,
    MojoStageContractMixin,
    MojoReturnAliasMixin,
    MojoGenericTypeMixin,
    MojoResourceTypeMixin,
    MojoResourceAccessMixin,
    MojoSemanticMixin,
    MojoTypeDeclarationMixin,
    MojoFunctionMixin,
    MojoStatementMixin,
    MojoMatchMixin,
    MojoExpressionMixin,
    MojoWaveOpMixin,
    MojoBuiltinCallMixin,
    MojoTargetShapeMixin,
    MojoResourceCallMixin,
    MojoVectorOpMixin,
    MojoResourceHelperMixin,
    MojoMathHelperMixin,
    MojoTypeMixin,
    MojoValueMixin,
    MojoCallExpressionMixin,
    MojoExpressionNodeMixin,
    MojoPointerExpressionMixin,
    MojoSplitBuiltinMixin,
    MojoSharedMemoryMixin,
    MojoArithmeticConversionMixin,
    MojoResultTypeMixin,
):
    """Emit Mojo-like shader source from the shared CrossGL AST."""

    def __init__(self):
        """Initialize Mojo type maps and helper-generation state."""
        self.vector_constructor_info = MOJO_VECTOR_TYPES
        self.struct_types = {}
        self.function_return_types = {}
        self.function_parameter_types = {}
        self.function_overload_signatures = {}
        self.function_return_resource_aliases = {}
        self.function_return_resource_static_aliases = {}
        self.function_return_resource_field_aliases = {}
        self.function_return_resource_static_field_aliases = {}
        self.stage_local_function_extra_params = {}
        self.reset_mojo_private_globals()
        self.variable_types = {}
        self.enum_types = {}
        self.enum_variant_aliases = {}
        self.enum_variant_values = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_result_types = {}
        self.enum_variant_names = {}
        self.enum_unit_variant_constructors = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.generic_struct_type_params = {}
        self.struct_member_semantics = {}
        self.current_enum_value_aliases = {}
        self.resource_access_qualifiers = {}
        self.resource_access_qualifier_aliases = {}
        self.resource_access_path_qualifier_aliases = {}
        self.struct_resource_access_qualifiers = {}
        self.literal_int_constants = {}
        self.mojo_resource_binding_cursors = {}
        self.mojo_source_resource_bindings = {}
        self.mojo_used_resource_bindings = {}
        self.mojo_resource_binding_initializers = {}
        self.required_resource_types = set()
        self.required_resource_sample_types = set()
        self.required_resource_sample_sampler_types = set()
        self.required_resource_lod_types = set()
        self.required_resource_lod_sampler_types = set()
        self.required_resource_grad_types = set()
        self.required_resource_grad_sampler_types = set()
        self.required_resource_size_types = set()
        self.required_resource_size_helpers = set()
        self.required_resource_query_level_types = set()
        self.required_resource_sample_count_helpers = set()
        self.required_resource_texel_fetch_types = set()
        self.required_resource_texel_fetch_offset_types = set()
        self.required_image_load_types = set()
        self.required_image_store_types = set()
        self.required_resource_builtin_helpers = {}
        self.required_buffer_resource_types = set()
        self.required_buffer_load_helpers = set()
        self.required_buffer_store_helpers = set()
        self.required_buffer_append_helpers = set()
        self.required_buffer_consume_helpers = set()
        self.required_buffer_dimensions_helpers = set()
        self.required_byte_address_vector_load_helpers = set()
        self.required_byte_address_vector_store_helpers = set()
        self.required_geometry_stream_helpers = set()
        self.required_tessellation_patch_helpers = set()
        self.required_mesh_helpers = set()
        self.required_reinterpret_helpers = set()
        self.requires_bitcast_import = False
        self.requires_threadfence_import = False
        self.requires_atomic_import = False
        self.requires_shared_memory_imports = False
        self.required_bit_imports = set()
        self.required_wave_helpers = {}
        self.required_ray_helpers = set()
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.required_select_helpers = set()
        self.required_matrix_types = set()
        self.required_matrix_constructor_helpers = {}
        self.required_matrix_diagonal_helpers = set()
        self.required_math_helpers = set()
        self.required_fract_helpers = set()
        self.required_mod_helpers = set()
        self.required_saturate_helpers = set()
        self.required_builtin_placeholders = set()
        self.required_gpu_builtin_placeholders = set()
        self.current_return_type = None
        self.current_shader = None
        self.current_shader_type = None
        self.current_shader_stage_count = 0
        self.expression_identifier_replacements = {}
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_exit_alias_state_stack = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.lambda_counter = 0
        self.match_expression_counter = 0
        self.match_subject_counter = 0
        self.dimension_query_counter = 0
        self.scalar_atomic_counter = 0
        self.swizzle_assignment_counter = 0
        self.increment_expression_counter = 0
        self.expression_prelude_stack = []
        self.type_mapping = {
            # Scalar Types
            "void": "None",
            "int": "Int32",
            "i8": "Int8",
            "i16": "Int16",
            "i32": "Int32",
            "int8": "Int8",
            "int8_t": "Int8",
            "int16": "Int16",
            "int16_t": "Int16",
            "min12int": "Int16",
            "min16int": "Int16",
            "short": "Int16",
            "long": "Int64",
            "i64": "Int64",
            "int64": "Int64",
            "int64_t": "Int64",
            "uint": "UInt32",
            "u8": "UInt8",
            "u16": "UInt16",
            "u32": "UInt32",
            "uint8": "UInt8",
            "uint8_t": "UInt8",
            "uint16": "UInt16",
            "uint16_t": "UInt16",
            "min16uint": "UInt16",
            "ushort": "UInt16",
            "ulong": "UInt64",
            "u64": "UInt64",
            "uint64": "UInt64",
            "uint64_t": "UInt64",
            "f16": "Float16",
            "f32": "Float32",
            "f64": "Float64",
            "float": "Float32",
            "double": "Float64",
            "half": "Float16",
            "min10float": "Float16",
            "min16float": "Float16",
            "bool": "Bool",
            "string": "String",
            "str": "String",
            # CrossGL char is a signed 8-bit integer, not text.
            "char": "Int8",
            **{
                name: f"SIMD[{dtype}, {storage_width}]"
                for name, (dtype, _, storage_width, _) in MOJO_VECTOR_TYPES.items()
            },
            **{
                name: self.matrix_type_name(dtype, columns, rows)
                for name, (dtype, columns, rows) in MOJO_MATRIX_TYPES.items()
            },
            # Texture/resource placeholders for Mojo compile-smoke support.
            **MOJO_RESOURCE_TYPE_MAPPING,
            "RayDesc": "RayDesc",
            "RayQuery": "RayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "BuiltInTriangleIntersectionAttributes"
            ),
        }
        self.scalar_constructor_map = {
            name: mapped
            for name, mapped in self.type_mapping.items()
            if mapped in {
                "Bool",
                "Float16",
                "Float32",
                "Float64",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "String",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            }
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_WorkGroupID": "workgroup_id",
            # Fragment attributes
            "gl_FragColor": "color(0)",
            "gl_FragColor0": "color(0)",
            "gl_FragColor1": "color(1)",
            "gl_FragColor2": "color(2)",
            "gl_FragColor3": "color(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord0",
            "TEXCOORD1": "texcoord1",
            "TEXCOORD2": "texcoord2",
            "TEXCOORD3": "texcoord3",
            "COLOR": "color",
            "COLOR0": "color0",
            "COLOR1": "color1",
            "SV_Position": "position",
            "SV_Depth": "depth(any)",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_GroupID": "workgroup_id",
            "SV_GroupIndex": "group_index",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_InstanceID": "instance_id",
            "SV_IsFrontFace": "front_facing",
            "payload": "ray_payload",
            "rayPayloadEXT": "ray_payload",
            "rayPayloadInEXT": "ray_payload",
            "hit_attribute": "hit_attribute",
            "hitAttributeEXT": "hit_attribute",
            "callable_data": "callable_data",
            "callableDataEXT": "callable_data",
            "callableDataInEXT": "callable_data",
            "gl_LaunchIDEXT": "launch_id",
            "gl_LaunchSizeEXT": "launch_size",
            "gl_HitTEXT": "hit_t",
            "gl_HitKindEXT": "hit_kind",
            "SV_DispatchRaysIndex": "launch_id",
            "SV_DispatchRaysDimensions": "launch_size",
            "SV_Target": "color(0)",
            "SV_Target0": "color(0)",
            "SV_Target1": "color(1)",
            "SV_Target2": "color(2)",
            "SV_Target3": "color(3)",
            "SV_VertexID": "vertex_id",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "normalize": "normalize",
            "dot": "dot_product",
            "cross": "cross_product",
            "length": "magnitude",
            "reflect": "reflect",
            "refract": "refract",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "sqrt": "sqrt",
            "inversesqrt": "rsqrt",
            "inverseSqrt": "rsqrt",
            "pow": "power",
            "roundEven": "round_even",
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
            "TraceRay": "trace_ray",
            "CallShader": "call_shader",
            "ReportHit": "report_hit",
            "IgnoreHit": "ignore_hit",
            "AcceptHitAndEndSearch": "accept_hit_and_end_search",
        }

    def generate(self, ast):
        """Generate complete Mojo-like shader source for a CrossGL AST, or raise :class:`MojoLoweringRefused`.

        A call the generator has no lowering for is written as it stands, so the rendered program is refused when
        it still calls a built-in function of another shading language that Mojo does not define and the program
        does not declare. A built-in type of another language that Mojo has no mapping for, such as Metal's
        ``visible_function_table``, is refused the same way.
        """
        self.program_type_names = declared_type_names(ast)
        output = self.render_program(ast)
        refuse_written_foreign_calls(
            output, MOJO_CALLABLE_FUNCTION_NAMES, "Mojo", non_calls=MOJO_NON_CALLS
        )
        return output

    def render_program(self, ast):
        """Render complete Mojo-like shader source for a CrossGL AST."""
        refuse_cooperative_matrices(ast, label="Mojo", target="mojo")
        refuse_precise_evaluation(ast, label="Mojo", target="mojo")
        # A separate CrossGL texture is the same Mojo texture as a combined one sampled with explicit samplers.
        ast = canonicalize_separate_textures(ast)
        # A storage block holding one runtime-sized array is that array: a pointer argument of the kernel.
        lower_mojo_single_array_storage_blocks(ast)
        self.struct_types = {}
        self.function_return_types = {}
        self.function_parameter_types = {}
        self.function_overload_signatures = {}
        self.function_return_resource_aliases = {}
        self.function_return_resource_static_aliases = {}
        self.function_return_resource_field_aliases = {}
        self.function_return_resource_static_field_aliases = {}
        self.stage_local_function_extra_params = {}
        self.variable_types = {}
        self.reset_mojo_kernel_arguments(ast)
        self.reset_mojo_private_globals()
        self.enum_types = {}
        self.enum_variant_aliases = {}
        self.enum_variant_values = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_result_types = {}
        self.enum_variant_names = {}
        self.enum_unit_variant_constructors = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.generic_struct_type_params = {}
        self.struct_member_semantics = {}
        self.current_enum_value_aliases = {}
        self.resource_access_qualifiers = {}
        self.resource_access_qualifier_aliases = {}
        self.resource_access_path_qualifier_aliases = {}
        self.struct_resource_access_qualifiers = {}
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        self.mojo_resource_binding_cursors = {}
        self.mojo_source_resource_bindings = {}
        self.mojo_used_resource_bindings = {}
        self.mojo_resource_binding_initializers = {}
        self.required_resource_types = set()
        self.required_resource_sample_types = set()
        self.required_resource_sample_sampler_types = set()
        self.required_resource_lod_types = set()
        self.required_resource_lod_sampler_types = set()
        self.required_resource_grad_types = set()
        self.required_resource_grad_sampler_types = set()
        self.required_resource_size_types = set()
        self.required_resource_size_helpers = set()
        self.required_resource_query_level_types = set()
        self.required_resource_sample_count_helpers = set()
        self.required_resource_texel_fetch_types = set()
        self.required_resource_texel_fetch_offset_types = set()
        self.required_image_load_types = set()
        self.required_image_store_types = set()
        self.required_resource_builtin_helpers = {}
        self.required_buffer_resource_types = set()
        self.required_buffer_load_helpers = set()
        self.required_buffer_store_helpers = set()
        self.required_buffer_append_helpers = set()
        self.required_buffer_consume_helpers = set()
        self.required_buffer_dimensions_helpers = set()
        self.required_byte_address_vector_load_helpers = set()
        self.required_byte_address_vector_store_helpers = set()
        self.required_geometry_stream_helpers = set()
        self.required_tessellation_patch_helpers = set()
        self.required_mesh_helpers = set()
        self.required_reinterpret_helpers = set()
        self.requires_bitcast_import = False
        self.requires_threadfence_import = False
        self.requires_atomic_import = False
        self.requires_shared_memory_imports = False
        self.required_bit_imports = set()
        self.required_wave_helpers = {}
        self.required_ray_helpers = set()
        self.required_helpers = set()
        self.required_splat_helpers = set()
        self.required_swizzle_helpers = set()
        self.required_constructor_helpers = {}
        self.required_select_helpers = set()
        self.required_matrix_types = set()
        self.required_matrix_constructor_helpers = {}
        self.required_matrix_diagonal_helpers = set()
        self.required_math_helpers = set()
        self.required_fract_helpers = set()
        self.required_mod_helpers = set()
        self.required_saturate_helpers = set()
        self.required_builtin_placeholders = set()
        self.required_gpu_builtin_placeholders = set()
        self.current_return_type = None
        self.current_shader_type = None
        self.current_shader_stage_count = len(getattr(ast, "stages", {}) or {})
        self.expression_identifier_replacements = {}
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_exit_alias_state_stack = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.lambda_counter = 0
        self.match_expression_counter = 0
        self.match_subject_counter = 0
        self.dimension_query_counter = 0
        self.scalar_atomic_counter = 0
        self.swizzle_assignment_counter = 0
        self.increment_expression_counter = 0
        self.expression_prelude_stack = []

        header = "# Generated Mojo Shader Code\n"
        header += "from std.math import *\n"
        header += "from max.gpu import *\n"
        code = ""

        structs = getattr(ast, "structs", [])
        self.prepare_generic_enum_metadata(ast, structs)
        self.collect_function_return_types(ast)
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            list(iter_function_nodes(ast)),
            collect_constant_and_cbuffer_member_types(self, ast),
            ast=ast,
        )
        for func in generic_function_specializations.values():
            self.register_function_return_type(func)
        self.reject_unsupported_generic_functions(ast)
        self.reject_parameterized_generic_enum_specializations(ast)
        for node in structs:
            if isinstance(node, EnumNode):
                code += self.generate_enum(node)
        code += self.generate_generic_payload_enum_constants()
        code += self.generate_generic_payload_enum_structs()
        code += self.generate_generic_payload_enum_constructors()
        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                code += self.generate_struct(node)

        code += self.generate_constants(ast)

        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            code += self.generate_module_variable_declaration(node)

        cbuffers = self.get_cbuffer_nodes(ast)
        if cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        self.collect_function_return_types(ast)

        functions = getattr(ast, "functions", [])
        # A function that reads a module-scope buffer or uniform value takes it as an extra argument.
        self.stage_local_function_extra_params = self.mojo_kernel_argument_captures(
            [*functions, *self.mojo_stage_functions(ast)],
            {
                **self.mojo_module_kernel_arguments,
                **self.mojo_private_global_arguments(),
            },
        )
        for func in functions:
            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    code += self.generate_function(specialized_func)
                continue
            qualifier = self.function_stage_qualifier(func)

            if qualifier == "vertex":
                code += "# Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier == "fragment":
                code += "# Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier == "compute":
                code += "# Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            elif (
                qualifier in {
                    "geometry",
                    "tessellation_control",
                    "tessellation_evaluation",
                } | MOJO_MESH_STAGE_TYPES
            ):
                code += f"# {qualifier.title()} Shader\n"
                code += self.generate_function(func, shader_type=qualifier)
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            emitted_stage_local_variables = set()
            emitted_stage_local_structs = set(self.struct_types)
            emitted_local_functions = set()
            for stage_type, stage in ast.stages.items():
                if hasattr(stage, "entry_point"):
                    stage_name = self.stage_type_name(stage_type)
                    self.validate_stage_type(stage_name)
                    self.mojo_stage_kernel_arguments = {}
                    self.reset_mojo_stage_private_globals()
                    code += f"# {stage_name.title()} Shader\n"
                    code += self.generate_stage_local_declarations(
                        stage,
                        emitted_stage_local_variables,
                        emitted_stage_local_structs,
                    )
                    previous_extra_params = self.stage_local_function_extra_params
                    self.stage_local_function_extra_params = (
                        self.merge_mojo_extra_params(
                            previous_extra_params,
                            self.stage_local_extra_params(stage),
                            self.mojo_kernel_argument_captures(
                                [
                                    *functions,
                                    *self.stage_local_functions_to_emit(stage),
                                ],
                                {
                                    **dict(self.mojo_entry_kernel_arguments()),
                                    **self.mojo_stage_shared_arguments(stage),
                                    **self.mojo_private_global_arguments(),
                                },
                            ),
                        )
                    )
                    for func in self.stage_local_functions_to_emit(stage):
                        signature = self.function_signature_key(func)
                        if signature in emitted_local_functions:
                            continue
                        if generic_function_parameters(func):
                            for specialized_func in generic_function_emission_list(
                                self,
                                func,
                            ):
                                code += self.generate_function(specialized_func)
                            emitted_local_functions.add(signature)
                            continue
                        code += self.generate_function(func)
                        emitted_local_functions.add(signature)
                    code += self.generate_function(
                        stage.entry_point, shader_type=stage_name, stage_node=stage
                    )
                    self.stage_local_function_extra_params = previous_extra_params
        if self.required_reinterpret_helpers or self.requires_bitcast_import:
            header += "from std.memory import bitcast\n"
        if self.requires_shared_memory_imports:
            header += "from std.memory import stack_allocation\n"
            header += "from max.gpu.memory import AddressSpace\n"
        if self.requires_atomic_import:
            header += "from std.atomic import Atomic\n"
        if self.requires_threadfence_import:
            header += "from max.gpu.intrinsics import threadfence\n"
        if "StaticTuple[" in code:
            header += "from std.utils import StaticTuple\n"
        if self.required_bit_imports:
            header += (
                f"from std.bit import {', '.join(sorted(self.required_bit_imports))}\n"
            )
        wave_helpers = self.mojo_wave_helper_source()
        if wave_helpers:
            header += MOJO_WAVE_IMPORTS
        header += "\n"

        return (
            header
            + self.generate_required_helpers()
            + self.mojo_buffer_object_helper_source()
            + wave_helpers
            + code
        )

    def reject_parameterized_generic_enum_specializations(self, ast):
        if not self.generic_enum_struct_definitions:
            return
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            if not hasattr(value, "__dict__"):
                return
            name = getattr(value, "name", None)
            generic_args = getattr(value, "generic_args", None)
            if name in self.generic_enum_struct_definitions and generic_args:
                if any(
                    type_node_contains_generic_parameter(
                        arg, self.generic_parameter_names
                    )
                    for arg in generic_args
                ):
                    raise MojoRefusal(
                        "Mojo generic payload enum specializations must be concrete"
                    )
            for child in vars(value).values():
                visit(child)

        visit(ast)

    def generate_expression(self, expr, target_type=None, target_context=None):
        """Render a CrossGL expression as Mojo expression syntax, converted to a numeric ``target_type``."""
        code = self.generate_expression_value(expr, target_type, target_context)
        if target_type is None or isinstance(expr, (str, int, float, bool)):
            return code
        return self.mojo_converted_to_target(expr, code, target_type)

    def generate_expression_value(self, expr, target_type=None, target_context=None):
        """Render a CrossGL expression as Mojo expression syntax in its own type."""
        if is_comma_expression(expr):
            return refuse_comma_expression("Mojo", expr)
        if isinstance(expr, str):
            return self.generate_mojo_string_expression(expr, target_type)
        elif isinstance(expr, (int, float, bool)):
            return self.format_literal(expr)
        elif isinstance(expr, VariableNode):
            return self.generate_mojo_variable_expression(expr)
        elif isinstance(expr, BinaryOpNode):
            return self.generate_mojo_binary_expression(
                expr, target_type, target_context
            )
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            if target_type is not None and self.is_array_type_name(target_type):
                return self.generate_array_literal_expression(expr, target_type)
            return self.generate_array_literal_expression(expr)
        elif isinstance(expr, MatchNode):
            return self.generate_match_value_expression(
                expr, target_type, target_context or "match expression"
            )
        elif isinstance(expr, ConstructorNode):
            return self.generate_constructor_node(expr, target_type, target_context)
        elif isinstance(expr, CastNode):
            return self.generate_cast_node(expr)
        elif isinstance(expr, SwizzleNode):
            return self.generate_swizzle_node(expr)
        elif isinstance(expr, BuiltinVariableNode):
            return self.generate_builtin_variable_node(expr)
        elif isinstance(expr, TextureNode):
            return self.generate_texture_node(expr)
        elif isinstance(expr, TextureOpNode):
            return self.generate_texture_op_node(expr)
        elif isinstance(expr, AtomicOpNode):
            return self.generate_atomic_op_node(expr)
        elif isinstance(expr, BufferOpNode):
            return self.generate_buffer_op_node(expr)
        elif isinstance(expr, UnaryOpNode):
            op = self.map_operator(expr.op)
            if op in ["++", "--"]:
                return self.generate_increment_expression(expr)
            if op == "*":
                return self.generate_mojo_dereference_expression(expr)
            if op == "&":
                return self.generate_mojo_address_of_expression(expr)
            operand = self.generate_expression(expr.operand)
            if op == "not":
                operand_type = self.expression_result_type(expr.operand)
                info = self.vector_type_info(operand_type)
                if info is not None and info[0] == "DType.bool":
                    return f"(~{operand})"
                return f"(not {operand})"
            return f"({op}{operand})"
        elif (
            isinstance(expr, ArrayAccessNode)
            and hasattr(expr, "array")
            and hasattr(expr, "index")
        ):
            return self.generate_array_access_expression(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_wave_op(expr)
        elif isinstance(expr, RayTracingOpNode):
            return self.generate_ray_tracing_op(expr)
        elif isinstance(expr, RayQueryOpNode):
            return self.generate_ray_query_op(expr)
        elif isinstance(expr, MeshOpNode):
            return self.generate_mesh_op(expr)
        elif isinstance(expr, FunctionCallNode):
            # Extract function name properly (might be IdentifierNode)
            return self.generate_mojo_function_call_expression(
                expr, target_type, target_context
            )
        elif isinstance(expr, MemberAccessNode):
            return self.generate_mojo_member_access_expression(expr)
        elif isinstance(expr, PointerAccessNode):
            return self.generate_mojo_pointer_access_expression(expr)
        elif isinstance(expr, TernaryOpNode):
            return self.generate_mojo_ternary_expression(
                expr, target_type, target_context
            )
        elif "Literal" in str(expr.__class__) and hasattr(expr, "value"):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            return self.format_literal(expr.value, literal_type)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.generate_mojo_identifier_like_expression(expr, target_type)
        elif hasattr(expr, "__class__") and "ExpressionStatement" in str(
            expr.__class__
        ):
            if hasattr(expr, "expression"):
                return self.generate_expression(expr.expression)
            else:
                return self.generate_expression(expr)
        # The repr of a node is valid Mojo call syntax, so an unhandled node is refused rather than printed.
        raise MojoRefusal(
            f"Mojo has no lowering for the CrossGL expression node {type(expr).__name__}"
        )

    def generate_mesh_op(self, expr):
        operation = getattr(expr, "operation", "mesh intrinsic")
        arguments = list(getattr(expr, "arguments", []) or [])
        stage_name = self.current_shader_type
        self.validate_mesh_op_stage(operation, stage_name)
        self.validate_mesh_op_arguments(operation, arguments, stage_name)

        args = [self.generate_expression(argument) for argument in arguments]
        if operation == "SetMeshOutputCounts":
            self.required_mesh_helpers.add("set_mesh_output_counts")
            return f"_crossgl_set_mesh_output_counts({', '.join(args)})"
        if operation == "DispatchMesh":
            self.required_mesh_helpers.add("dispatch_mesh")
            return f"_crossgl_dispatch_mesh({', '.join(args)})"

        raise MojoRefusal(f"Unsupported Mojo mesh intrinsic {operation}")

    def generate_required_helpers(self):
        self.required_math_helpers = self.expanded_math_helpers(
            self.required_math_helpers
        )
        if (
            not self.required_helpers
            and not self.required_splat_helpers
            and not self.required_swizzle_helpers
            and not self.required_constructor_helpers
            and not self.required_select_helpers
            and not self.required_matrix_types
            and not self.required_matrix_constructor_helpers
            and not self.required_matrix_diagonal_helpers
            and not self.required_math_helpers
            and not self.required_fract_helpers
            and not self.required_mod_helpers
            and not self.required_saturate_helpers
            and not self.required_resource_types
            and not self.required_resource_sample_types
            and not self.required_resource_sample_sampler_types
            and not self.required_resource_lod_types
            and not self.required_resource_lod_sampler_types
            and not self.required_resource_grad_types
            and not self.required_resource_grad_sampler_types
            and not self.required_resource_size_types
            and not self.required_resource_query_level_types
            and not self.required_resource_sample_count_helpers
            and not self.required_resource_texel_fetch_types
            and not self.required_resource_texel_fetch_offset_types
            and not self.required_image_load_types
            and not self.required_image_store_types
            and not self.required_resource_builtin_helpers
            and not self.required_buffer_resource_types
            and not self.required_buffer_load_helpers
            and not self.required_buffer_store_helpers
            and not self.required_buffer_append_helpers
            and not self.required_buffer_consume_helpers
            and not self.required_buffer_dimensions_helpers
            and not self.required_byte_address_vector_load_helpers
            and not self.required_byte_address_vector_store_helpers
            and not self.required_geometry_stream_helpers
            and not self.required_tessellation_patch_helpers
            and not self.required_mesh_helpers
            and not self.required_reinterpret_helpers
            and not self.required_ray_helpers
            and not self.required_builtin_placeholders
            and not self.required_gpu_builtin_placeholders
        ):
            return ""

        code = ""
        if self.required_builtin_placeholders:
            code += "# CrossGL builtin placeholders\n"
            for name, type_name in sorted(self.required_builtin_placeholders):
                code += self.generate_builtin_placeholder(name, type_name)
            code += "\n"

        if self.required_gpu_builtin_placeholders:
            code += "# CrossGL GPU builtin placeholders\n"
            emitted_placeholder_types = set()
            for name, type_name in sorted(self.required_gpu_builtin_placeholders):
                code += self.generate_gpu_builtin_placeholder(
                    name, type_name, emitted_placeholder_types
                )
            code += "\n"

        resource_sampled_types = (
            self.required_resource_sample_types
            | self.required_resource_sample_sampler_types
            | self.required_resource_lod_types
            | self.required_resource_lod_sampler_types
            | self.required_resource_grad_types
            | self.required_resource_grad_sampler_types
            | self.required_resource_size_types
            | self.required_resource_query_level_types
            | {
                resource_type
                for _, resource_type in self.required_resource_sample_count_helpers
            }
            | self.required_resource_texel_fetch_types
            | self.required_resource_texel_fetch_offset_types
            | self.required_image_load_types
            | self.required_image_store_types
        )
        resource_builtin_sampler_required = any(
            "Sampler" in helper["arg_types"]
            for helper in self.required_resource_builtin_helpers.values()
        )
        resource_types = self.required_resource_types | resource_sampled_types
        if (
            self.required_resource_sample_sampler_types
            or self.required_resource_lod_sampler_types
            or self.required_resource_grad_sampler_types
            or resource_builtin_sampler_required
        ):
            resource_types.add("Sampler")
        if (
            self.required_resource_types
            or resource_sampled_types
            or self.required_resource_builtin_helpers
            or self.required_buffer_resource_types
            or self.required_buffer_load_helpers
            or self.required_buffer_store_helpers
            or self.required_buffer_append_helpers
            or self.required_buffer_consume_helpers
            or self.required_buffer_dimensions_helpers
            or self.required_byte_address_vector_load_helpers
            or self.required_byte_address_vector_store_helpers
        ):
            if self.resource_requirements_use_concrete_bindings(resource_types):
                code += "# CrossGL resource bindings\n"
                for resource_type in sorted(resource_types):
                    code += self.generate_concrete_resource_type(resource_type)
                for resource_type in sorted(
                    self.required_resource_sample_sampler_types
                ):
                    code += self.generate_concrete_resource_sample_sampler_helper(
                        resource_type
                    )
            else:
                code += "# CrossGL resource placeholders\n"
                for resource_type in sorted(resource_types):
                    code += self.generate_resource_type(resource_type)
                for buffer_type in sorted(self.required_buffer_resource_types):
                    code += self.generate_buffer_resource_type(buffer_type)
                for resource_type in sorted(self.required_resource_sample_types):
                    code += self.generate_resource_sample_helper(resource_type)
                for resource_type in sorted(
                    self.required_resource_sample_sampler_types
                ):
                    code += self.generate_resource_sample_sampler_helper(resource_type)
                for resource_type in sorted(self.required_resource_lod_types):
                    code += self.generate_resource_lod_helper(resource_type)
                for resource_type in sorted(self.required_resource_lod_sampler_types):
                    code += self.generate_resource_lod_sampler_helper(resource_type)
                for resource_type in sorted(self.required_resource_grad_types):
                    code += self.generate_resource_grad_helper(resource_type)
                for resource_type in sorted(self.required_resource_grad_sampler_types):
                    code += self.generate_resource_grad_sampler_helper(resource_type)
                if any(
                    helper_name == "texture_size"
                    for helper_name, _resource_type in (
                        self.required_resource_size_helpers
                    )
                ):
                    code += self.generate_mip_dimension_helper()
                for helper_name, resource_type in sorted(
                    self.required_resource_size_helpers
                ):
                    code += self.generate_resource_size_helper(
                        helper_name, resource_type
                    )
                for resource_type in sorted(self.required_resource_query_level_types):
                    code += self.generate_resource_query_levels_helper(resource_type)
                for helper_name, resource_type in sorted(
                    self.required_resource_sample_count_helpers
                ):
                    code += self.generate_resource_sample_count_helper(
                        helper_name, resource_type
                    )
                for resource_type in sorted(self.required_resource_texel_fetch_types):
                    code += self.generate_texel_fetch_helper(resource_type)
                for resource_type in sorted(
                    self.required_resource_texel_fetch_offset_types
                ):
                    code += self.generate_texel_fetch_offset_helper(resource_type)
                for resource_type in sorted(self.required_image_load_types):
                    code += self.generate_image_load_helper(resource_type)
                for resource_type in sorted(self.required_image_store_types):
                    code += self.generate_image_store_helper(resource_type)
                for key in sorted(self.required_resource_builtin_helpers):
                    code += self.generate_resource_builtin_helper(
                        self.required_resource_builtin_helpers[key]
                    )
                for key in sorted(self.required_buffer_load_helpers):
                    code += self.generate_buffer_load_helper(*key)
                for key in sorted(self.required_buffer_store_helpers):
                    code += self.generate_buffer_store_helper(*key)
                for key in sorted(self.required_buffer_append_helpers):
                    code += self.generate_buffer_append_helper(*key)
                for key in sorted(self.required_buffer_consume_helpers):
                    code += self.generate_buffer_consume_helper(*key)
                for key in sorted(self.required_buffer_dimensions_helpers):
                    code += self.generate_buffer_dimensions_helper(*key)
                for key in sorted(self.required_byte_address_vector_load_helpers):
                    code += self.generate_byte_address_vector_load_helper(*key)
                for key in sorted(self.required_byte_address_vector_store_helpers):
                    code += self.generate_byte_address_vector_store_helper(*key)
            code += "\n"

        if self.required_geometry_stream_helpers:
            code += "# CrossGL geometry stream placeholders\n"
            for stream_name in sorted(self.required_geometry_stream_helpers):
                code += self.generate_geometry_stream_helper(stream_name)
            code += "\n"

        if self.required_tessellation_patch_helpers:
            code += "# CrossGL tessellation patch placeholders\n"
            for patch_name in sorted(self.required_tessellation_patch_helpers):
                code += self.generate_tessellation_patch_helper(patch_name)
            code += "\n"

        if self.required_mesh_helpers:
            code += "# CrossGL mesh/task placeholders\n"
            if "set_mesh_output_counts" in self.required_mesh_helpers:
                code += self.generate_set_mesh_output_counts_helper()
            if "dispatch_mesh" in self.required_mesh_helpers:
                code += self.generate_dispatch_mesh_helper()
            code += "\n"

        if self.required_reinterpret_helpers:
            code += "# CrossGL reinterpret helpers\n"
            for key in sorted(self.required_reinterpret_helpers):
                code += self.generate_reinterpret_helper(*key)
            code += "\n"

        if self.required_ray_helpers:
            code += self.generate_ray_helpers()

        non_matrix_saturate_helpers = {
            key for key in self.required_saturate_helpers if key[0] != "matrix"
        }
        matrix_saturate_helpers = {
            key for key in self.required_saturate_helpers if key[0] == "matrix"
        }

        if (
            self.required_math_helpers
            or self.required_fract_helpers
            or self.required_mod_helpers
            or non_matrix_saturate_helpers
        ):
            code += "# CrossGL math helpers\n"
            for helper_name in sorted(self.required_math_helpers):
                code += self.generate_math_helper(helper_name)
            for key in sorted(self.required_fract_helpers):
                code += self.generate_fract_helper(key)
            for key in sorted(self.required_mod_helpers):
                code += self.generate_mod_helper(key)
            for key in sorted(non_matrix_saturate_helpers):
                code += self.generate_saturate_helper(key)
            code += "\n"

        if self.required_matrix_types:
            code += "# CrossGL matrix types\n"
            for key in sorted(self.required_matrix_types):
                code += self.generate_matrix_type(key)
            code += "\n"

        if matrix_saturate_helpers:
            code += "# CrossGL matrix math helpers\n"
            for key in sorted(matrix_saturate_helpers):
                code += self.generate_saturate_helper(key)
            code += "\n"

        if (
            self.required_helpers
            or self.required_splat_helpers
            or self.required_swizzle_helpers
            or self.required_constructor_helpers
            or self.required_select_helpers
            or self.required_matrix_constructor_helpers
            or self.required_matrix_diagonal_helpers
        ):
            code += "# CrossGL vector helpers\n"
        for dtype, op, helper_kind in sorted(self.required_helpers):
            code += self.generate_vector_binary_helper(dtype, op, helper_kind)
        for dtype in sorted(self.required_splat_helpers):
            code += self.generate_vec3_splat_helper(dtype)
        for dtype, source_width, member in sorted(self.required_swizzle_helpers):
            code += self.generate_swizzle_helper(dtype, source_width, member)
        for key in sorted(self.required_constructor_helpers):
            code += self.generate_constructor_helper(
                self.required_constructor_helpers[key]
            )
        for key in sorted(self.required_select_helpers):
            code += self.generate_select_helper(key)
        for key in sorted(self.required_matrix_constructor_helpers):
            code += self.generate_matrix_constructor_helper(
                self.required_matrix_constructor_helpers[key]
            )
        for key in sorted(self.required_matrix_diagonal_helpers):
            code += self.generate_matrix_diagonal_helper(*key)
        return code + "\n"

    def generate_dispatch_mesh_helper(self):
        return (
            "def _crossgl_dispatch_mesh[X: AnyType, Y: AnyType, Z: AnyType]("
            "x: X, y: Y, z: Z):\n"
            "    pass\n\n"
            "def _crossgl_dispatch_mesh[X: AnyType, Y: AnyType, Z: AnyType, "
            "Payload: AnyType](x: X, y: Y, z: Z, payload: Payload):\n"
            "    pass\n\n"
        )

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def expression_result_type(self, expr):
        handled = self.mojo_pointer_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        handled = self.mojo_value_expression_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        handled = self.mojo_function_call_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, ConstructorNode):
            return self.convert_type_node_to_string(expr.constructor_type)
        if isinstance(expr, CastNode):
            return self.convert_type_node_to_string(expr.target_type)
        if isinstance(expr, SwizzleNode):
            return self.swizzle_result_type(
                self.expression_result_type(expr.vector_expr),
                len(expr.components),
            )
        if isinstance(expr, BuiltinVariableNode):
            if expr.component:
                base_type = self.builtin_variable_result_type(expr.builtin_name)
                vector_info = self.vector_type_info(base_type)
                if vector_info is not None:
                    return MOJO_DTYPE_INFO[vector_info[0]][0]
                return base_type
            return self.builtin_variable_result_type(expr.builtin_name)
        if isinstance(expr, WaveOpNode):
            return self.mojo_wave_result_type(
                getattr(expr, "operation", ""), getattr(expr, "arguments", []) or []
            )
        if isinstance(expr, TextureNode):
            resource_type = self.map_type(
                self.expression_result_type(expr.texture_expr)
            )
            return self.texture_result_type_name(resource_type)
        handled = self.mojo_texture_operation_result_type(expr)
        if handled is not UNHANDLED:
            return handled
        if isinstance(expr, AtomicOpNode):
            target_type = self.type_name(self.expression_result_type(expr.target))
            if target_type in MOJO_SCALAR_ATOMIC_TYPES:
                return target_type
            resource_type = self.map_type(self.expression_result_type(expr.target))
            value_type = self.image_value_type(resource_type)
            if value_type == "UInt32":
                return "uint"
            return "int"
        if isinstance(expr, BufferOpNode):
            return self.buffer_op_result_type(expr)
        if isinstance(expr, MemberAccessNode):
            obj_type = self.expression_result_type(expr.object)
            obj_struct_type, obj_fields = self.struct_type_name_and_fields(obj_type)
            if obj_struct_type is not None and expr.member in obj_fields:
                return obj_fields.get(expr.member)
            if obj_struct_type is not None:
                return None
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                return self.swizzle_result_type(obj_type, len(swizzle_indices))
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", "")
            if name in MOJO_BUILTIN_PLACEHOLDERS and name not in self.variable_types:
                return MOJO_BUILTIN_PLACEHOLDERS[name][1]
            if name not in self.variable_types and float_limit_constant(name):
                return "float"
            return (
                self.identifier_replacement_result_type(name)
                or self.enum_variant_result_types.get(name)
                or self.variable_types.get(name)
            )
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        return None
