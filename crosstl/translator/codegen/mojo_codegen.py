"""CrossGL-to-Mojo code generator."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
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
from ..refusals import MojoRefusal
from .array_utils import collect_literal_int_constants
from .comma_expressions import is_comma_expression, refuse_comma_expression
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .enum_utils import type_node_contains_generic_parameter
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    iter_function_nodes,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
    reject_unresolved_generic_function_call,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .lowering_refusals import refuse_unsupported
from .mojo_bit_functions import generate_mojo_bit_function_call
from .mojo_builtin_calls import MojoBuiltinCallMixin
from .mojo_expressions import MojoExpressionMixin
from .mojo_functions import MojoFunctionMixin
from .mojo_generic_types import MojoGenericTypeMixin
from .mojo_integer_promotion import C_PROMOTING_BINARY_OPERATORS, promoted_operand
from .mojo_match_lowering import MojoMatchMixin
from .mojo_math_helpers import MojoMathHelperMixin
from .mojo_refusals import (  # noqa: F401  (registers the Mojo refusal)
    MojoLoweringRefused,
)
from .mojo_resource_access import MojoResourceAccessMixin
from .mojo_resource_calls import MojoResourceCallMixin
from .mojo_resource_helpers import MojoResourceHelperMixin
from .mojo_resource_types import MojoResourceTypeMixin
from .mojo_return_aliases import MojoReturnAliasMixin
from .mojo_semantics import MojoSemanticMixin
from .mojo_stage_contracts import MojoStageContractMixin
from .mojo_statements import MojoStatementMixin
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
from .precise_evaluation import refuse_precise_evaluation
from .sampler_pairs import canonicalize_separate_textures
from .shader_clock_refusals import SHADER_CLOCK_BUILTINS, refuse_shader_clock_call
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
    MojoBuiltinCallMixin,
    MojoTargetShapeMixin,
    MojoResourceCallMixin,
    MojoVectorOpMixin,
    MojoResourceHelperMixin,
    MojoMathHelperMixin,
    MojoTypeMixin,
    MojoValueMixin,
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
        self.required_bit_imports = set()
        self.required_sync_helpers = set()
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
            "f16": "Float16",
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
                "Int16",
                "Int32",
                "Int64",
                "String",
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
        does not declare.
        """
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
        self.required_bit_imports = set()
        self.required_sync_helpers = set()
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
        header += "from math import *\n"
        header += "from gpu import *\n"
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
            if isinstance(node, ArrayNode):
                code += self.generate_array_declaration(node)
            else:
                if hasattr(node, "initial_value") and node.initial_value is not None:
                    vtype = self.variable_declared_type(node)
                    self.register_variable_type(
                        node.name,
                        vtype or self.expression_result_type(node.initial_value),
                    )
                    if (
                        isinstance(node.initial_value, ArrayLiteralNode)
                        and vtype is not None
                        and self.is_array_type_name(vtype)
                    ):
                        init_expr = self.generate_array_literal_expression(
                            node.initial_value, vtype
                        )
                    else:
                        self.validate_expression_target_shape(
                            node.initial_value, vtype, f"global declaration {node.name}"
                        )
                        init_expr = self.generate_expression(node.initial_value)
                        self.validate_scalar_target_conversion(
                            node.initial_value, vtype, f"global declaration {node.name}"
                        )
                    var_name = self.mojo_identifier(node.name)
                    if vtype is None:
                        code += f"var {var_name} = {init_expr}\n"
                        continue
                    code += f"var {var_name}: {self.map_type(vtype)} = {init_expr}\n"
                    continue

                # Handle both old and new AST variable structures
                vtype = self.variable_declared_type(node) or "float"
                self.register_variable_type(node.name, vtype)
                resource_comment = self.generate_resource_metadata_comment(node, vtype)
                var_name = self.mojo_identifier(node.name)
                if self.is_array_type_name(vtype):
                    code += resource_comment
                    code += (
                        f"var {var_name} = "
                        f"{self.array_initial_value_for_type(vtype)}\n"
                    )
                elif self.is_struct_type_name(vtype):
                    code += resource_comment
                    code += f"var {var_name} = {self.zero_value_for_type(vtype)}\n"
                elif self.is_resource_type_name(vtype):
                    code += resource_comment
                    mapped_type = self.map_type(vtype)
                    init_value = self.resource_binding_initial_value(
                        node.name, mapped_type, vtype
                    )
                    code += f"var {var_name}: {mapped_type} = " f"{init_value}\n"
                else:
                    code += (
                        f"var {var_name}: {self.map_type(vtype)} = "
                        f"{self.zero_value_for_type(vtype)}\n"
                    )

        cbuffers = self.get_cbuffer_nodes(ast)
        if cbuffers:
            code += "# Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        self.collect_function_return_types(ast)

        functions = getattr(ast, "functions", [])
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
                    code += f"# {stage_name.title()} Shader\n"
                    code += self.generate_stage_local_declarations(
                        stage,
                        emitted_stage_local_variables,
                        emitted_stage_local_structs,
                    )
                    previous_extra_params = self.stage_local_function_extra_params
                    self.stage_local_function_extra_params = {
                        **previous_extra_params,
                        **self.stage_local_extra_params(stage),
                    }
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
        if self.required_reinterpret_helpers:
            header += "from memory import bitcast\n"
        if self.required_bit_imports:
            header += (
                f"from std.bit import {', '.join(sorted(self.required_bit_imports))}\n"
            )
        header += "\n"

        return header + self.generate_required_helpers() + code

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
        """Render a CrossGL expression as Mojo expression syntax."""
        if is_comma_expression(expr):
            return refuse_comma_expression("Mojo", expr)
        if isinstance(expr, str):
            if expr in self.expression_identifier_replacements:
                return self.expression_identifier_replacements[expr]
            ray_builtin = self.mojo_ray_builtin_expression(expr)
            if ray_builtin is not None and expr not in self.variable_types:
                return ray_builtin
            gpu_builtin = self.mojo_gpu_builtin_expression(expr)
            if gpu_builtin is not None:
                return gpu_builtin
            builtin = self.mojo_builtin_placeholder_expression(expr)
            if builtin is not None:
                return builtin
            mapped_reference = self.map_enum_variant_reference(expr, target_type)
            if mapped_reference != expr:
                return mapped_reference
            return self.mojo_identifier(expr)
        elif isinstance(expr, (int, float, bool)):
            return self.format_literal(expr)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "vtype") and expr.vtype and expr.name:
                if expr.name in self.expression_identifier_replacements:
                    return self.expression_identifier_replacements[expr.name]
                ray_builtin = self.mojo_ray_builtin_expression(expr.name)
                if ray_builtin is not None and expr.name not in self.variable_types:
                    return ray_builtin
                gpu_builtin = self.mojo_gpu_builtin_expression(expr.name)
                if gpu_builtin is not None:
                    return gpu_builtin
                builtin = self.mojo_builtin_placeholder_expression(expr.name)
                if builtin is not None:
                    return builtin
                return self.mojo_identifier(expr.name)
            elif hasattr(expr, "name"):
                if expr.name in self.expression_identifier_replacements:
                    return self.expression_identifier_replacements[expr.name]
                ray_builtin = self.mojo_ray_builtin_expression(expr.name)
                if ray_builtin is not None and expr.name not in self.variable_types:
                    return ray_builtin
                gpu_builtin = self.mojo_gpu_builtin_expression(expr.name)
                if gpu_builtin is not None:
                    return gpu_builtin
                builtin = self.mojo_builtin_placeholder_expression(expr.name)
                if builtin is not None:
                    return builtin
                return self.mojo_identifier(expr.name)
            else:
                return str(expr)
        elif isinstance(expr, BinaryOpNode):
            comparison = self.generate_vector_comparison(expr)
            if comparison is not None:
                return comparison
            boolean_vector_logical = self.generate_boolean_vector_logical_op(expr)
            if boolean_vector_logical is not None:
                return boolean_vector_logical
            vector_binary = self.generate_vector_binary_op(expr)
            if vector_binary is not None:
                return vector_binary
            numeric_vector_binary = self.generate_numeric_vector_binary_op(
                expr, target_type, target_context
            )
            if numeric_vector_binary is not None:
                return numeric_vector_binary
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            op = self.map_operator(expr.op)
            if op in C_PROMOTING_BINARY_OPERATORS:
                left = promoted_operand(self, expr.left, left)
                right = promoted_operand(self, expr.right, right)
            return f"({left} {op} {right})"
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
            operand = self.generate_expression(expr.operand)
            if op == "not":
                operand_type = self.expression_result_type(expr.operand)
                info = self.vector_type_info(operand_type)
                if info is not None and info[0] == "DType.bool":
                    return f"(~{operand})"
                return f"(not {operand})"
            return f"({op}{operand})"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access properly
            if hasattr(expr, "array") and hasattr(expr, "index"):
                return self.generate_array_access_expression(expr)
            else:
                # Fallback for malformed ArrayAccessNode
                return str(expr)
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
            func_expr = getattr(expr, "function", None)
            if func_expr is None:
                func_expr = expr.name
            func_name = None
            if isinstance(func_expr, MemberAccessNode):
                member_call = self.generate_member_function_call(func_expr, expr.args)
                if member_call is not None:
                    return member_call
                callee = self.generate_expression(func_expr)
            elif hasattr(func_expr, "name") and getattr(func_expr, "name", None):
                # It's an IdentifierNode, extract the name
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            if (
                func_name in SHADER_CLOCK_BUILTINS
                and not self.is_user_defined_function(func_name)
            ):
                return refuse_shader_clock_call("Mojo", func_name)

            numeric_trait_call = generate_numeric_trait_method_call(
                self,
                func_expr,
                expr.args,
            )
            if numeric_trait_call is not None:
                return numeric_trait_call

            static_generic_call = generate_static_generic_numeric_call(self, func_name)
            if static_generic_call is not None:
                return static_generic_call

            if func_name == "lambda":
                lambda_expr = self.generate_lambda_expression(expr.args)
                if lambda_expr is not None:
                    return lambda_expr
                # A lambda whose parameters or body the helper materialization cannot type has no Mojo spelling:
                # the CrossGL text of it is not a Mojo expression.
                refuse_unsupported(
                    "Mojo",
                    "lambda",
                    "lambda whose parameter list or body cannot be materialized as a typed Mojo helper function",
                    "lambda",
                )

            enum_constructor = self.generate_enum_variant_constructor_call(
                func_name, expr.args, target_type
            )
            if enum_constructor is not None:
                return enum_constructor

            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                expr.args,
            )
            if specialized_func_name is not None:
                func_name = specialized_func_name
                callee = specialized_func_name
            else:
                reject_unresolved_generic_function_call(self, func_name, "Mojo")

            if self.is_user_defined_function(func_name):
                args = self.generate_user_function_call_arguments(func_name, expr.args)
                call_name = (
                    self.mojo_identifier(func_name) if func_name is not None else callee
                )
                return f"{call_name}({args})"

            reduction = self.generate_boolean_vector_reduction(func_name, expr.args)
            if reduction is not None:
                return reduction

            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, expr.args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call

            if func_name in {"fract", "frac"}:
                return self.generate_fract_call(expr.args)
            if (
                func_name == "roundEven"
                and len(expr.args) == 1
                and not self.is_user_defined_function(func_name)
            ):
                return self.generate_round_even_call(expr.args[0])
            if func_name == "mod":
                return self.generate_mod_call(expr.args)
            if func_name == "saturate":
                saturate_call = self.generate_saturate_call(expr.args)
                if saturate_call is not None:
                    return saturate_call
            if func_name in {"degrees", "radians"}:
                return self.generate_angle_conversion_call(func_name, expr.args)
            if func_name == "mix":
                bool_mix_call = self.generate_bool_mix_call(expr.args)
                if bool_mix_call is not None:
                    return bool_mix_call
            if func_name == "clamp":
                clamp_call = self.generate_clamp_call(expr.args)
                if clamp_call is not None:
                    return clamp_call
            if func_name in {"min", "max"}:
                min_max_call = self.generate_min_max_call(
                    func_name, expr.args, target_type
                )
                if min_max_call is not None:
                    return min_max_call
            if func_name == "smoothstep":
                smoothstep_call = self.generate_smoothstep_call(expr.args, target_type)
                if smoothstep_call is not None:
                    return smoothstep_call
            if func_name == "atan" and len(expr.args) == 2:
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"atan2({args})"
            if func_name == "texture":
                return self.generate_texture_call(expr.args, "sample")
            if func_name == "textureLod":
                return self.generate_texture_call(expr.args, "sample_lod")
            if func_name == "textureGrad":
                return self.generate_texture_call(expr.args, "sample_grad")
            if func_name == "textureSize":
                return self.generate_resource_size_call(expr.args, "texture_size")
            if func_name == "textureQueryLevels":
                return self.generate_resource_query_levels_call(expr.args)
            if func_name == "textureSamples":
                return self.generate_resource_sample_count_call(
                    expr.args, "texture_samples"
                )
            if func_name == "imageSamples":
                return self.generate_resource_sample_count_call(
                    expr.args, "image_samples"
                )
            if func_name == "texelFetch":
                return self.generate_texel_fetch_call(expr.args)
            if func_name == "texelFetchOffset":
                return self.generate_texel_fetch_call(expr.args, with_offset=True)
            if func_name == "imageSize":
                return self.generate_resource_size_call(expr.args, "image_size")
            if func_name == "imageLoad":
                return self.generate_image_load_call(expr.args)
            if func_name == "imageStore":
                return self.generate_image_store_call(expr.args)
            if func_name == "buffer_load":
                return self.generate_buffer_load_call(expr.args)
            if func_name == "buffer_store":
                return self.generate_buffer_store_call(expr.args)
            if func_name == "buffer_append":
                return self.generate_buffer_append_call(expr.args)
            if func_name == "buffer_consume":
                return self.generate_buffer_consume_call(expr.args)
            if func_name == "buffer_dimensions":
                return self.generate_buffer_dimensions_call(expr.args)
            if func_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
                raise MojoRefusal(
                    f"Unsupported Mojo buffer counter call {func_name}; "
                    "structured-buffer counters are not available in the Mojo "
                    "placeholder backend"
                )
            reinterpret_name = self.mojo_reinterpret_builtin_name(func_name)
            if reinterpret_name is not None:
                return self.generate_reinterpret_call(reinterpret_name, expr.args)
            if func_name in MOJO_GENERIC_TEXTURE_BUILTINS:
                helper_base, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[func_name]
                return self.generate_resource_builtin_call(
                    expr.args, helper_base, return_kind
                )
            scalar_atomic = self.generate_scalar_atomic_call(func_name, expr.args)
            if scalar_atomic is not None:
                return scalar_atomic
            if func_name in MOJO_IMAGE_ATOMIC_BUILTINS:
                return self.generate_image_atomic_call(
                    expr.args, MOJO_IMAGE_ATOMIC_BUILTINS[func_name]
                )
            if (
                func_name in MOJO_SYNC_BUILTINS
                and not expr.args
                and not self.is_user_defined_function(func_name)
            ):
                helper_name = MOJO_SYNC_BUILTINS[func_name]
                self.required_sync_helpers.add(helper_name)
                return f"{helper_name}()"
            if func_name == "__shfl_down_sync":
                return self.generate_cuda_shuffle_down_call(expr.args)
            bit_function = generate_mojo_bit_function_call(self, func_name, expr.args)
            if bit_function is not None:
                return bit_function

            if func_name in self.function_map:
                func_name = self.function_map[func_name]
            # Otherwise a program function or a Mojo library function, spelled as it is; a built-in function of
            # another language left in the output is refused by generate.
            if func_name in MOJO_MATH_HELPERS:
                self.required_math_helpers.add(func_name)
            if func_name in self.scalar_constructor_map:
                scalar_constructor_name = func_name
                mapped_constructor_name = self.scalar_constructor_map[func_name]
                target_dtype = MOJO_SCALAR_DTYPES.get(
                    self.type_name(scalar_constructor_name)
                )
                if target_dtype is not None and len(expr.args) == 1:
                    arg = self.generate_scalar_constructor_argument(
                        expr.args[0],
                        target_dtype,
                        f"scalar constructor {scalar_constructor_name} argument 1",
                    )
                    return f"{mapped_constructor_name}({arg})"
                func_name = mapped_constructor_name

            vector_constructor_name = self.normalize_generic_vector_type_name(func_name)
            if vector_constructor_name in self.vector_constructor_info:
                return self.generate_vector_constructor(
                    vector_constructor_name, expr.args
                )

            if func_name in MOJO_MATRIX_TYPES:
                return self.generate_matrix_constructor(func_name, expr.args)

            if self.generic_struct_mapped_type(func_name) is not None:
                return self.generate_struct_function_constructor_call(
                    expr, func_name, target_context, target_type
                )

            if func_name in self.struct_types:
                return self.generate_struct_function_constructor_call(
                    expr, func_name, target_context, target_type
                )

            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            call_name = (
                self.mojo_identifier(func_name) if func_name is not None else callee
            )
            return f"{call_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            builtin_member = self.generate_builtin_member_access(expr)
            if builtin_member is not None:
                return builtin_member
            obj = self.generate_expression(expr.object)
            enum_variant = self.map_enum_variant_reference(f"{obj}.{expr.member}")
            if enum_variant != f"{obj}.{expr.member}":
                return enum_variant
            obj_type = self.expression_result_type(expr.object)
            obj_struct_type, obj_fields = self.struct_type_name_and_fields(obj_type)
            if obj_struct_type is not None and expr.member in obj_fields:
                return f"{obj}.{self.mojo_identifier(expr.member)}"
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                return self.generate_swizzle(
                    expr.object, obj, obj_type, expr.member, swizzle_indices
                )
            return f"{obj}.{self.mojo_identifier(expr.member)}"
        elif isinstance(expr, TernaryOpNode):
            bool_vector_select = self.generate_bool_vector_select_expression(
                expr.condition, expr.true_expr, expr.false_expr
            )
            if bool_vector_select is not None:
                return bool_vector_select
            branch_context = target_context or "ternary expression"
            condition = self.generate_expression(expr.condition)
            self.validate_condition_scalar(expr.condition, "ternary condition")
            true_expr = self.generate_expression(
                expr.true_expr, target_type, f"{branch_context} true branch"
            )
            false_expr = self.generate_expression(
                expr.false_expr, target_type, f"{branch_context} false branch"
            )
            return f"({true_expr} if {condition} else {false_expr})"
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", str(expr))
            if name in self.expression_identifier_replacements:
                return self.expression_identifier_replacements[name]
            ray_builtin = self.mojo_ray_builtin_expression(name)
            if ray_builtin is not None and name not in self.variable_types:
                return ray_builtin
            gpu_builtin = self.mojo_gpu_builtin_expression(name)
            if gpu_builtin is not None:
                return gpu_builtin
            builtin = self.mojo_builtin_placeholder_expression(name)
            if builtin is not None:
                return builtin
            mapped_reference = self.map_enum_variant_reference(name, target_type)
            if mapped_reference != name:
                return mapped_reference
            return self.mojo_identifier(name)
        elif hasattr(expr, "__class__") and "ExpressionStatement" in str(
            expr.__class__
        ):
            if hasattr(expr, "expression"):
                return self.generate_expression(expr.expression)
            else:
                return self.generate_expression(expr)
        else:
            # For unknown expression types, handle special cases
            expr_str = str(expr)
            # Check if this looks like an array declaration being misinterpreted
            if (
                "ArrayAccessNode" in expr_str
                and "array=" in expr_str
                and "index=" in expr_str
            ):
                # Try to extract array name and size for array declarations
                import re

                array_match = re.search(r"array=(\w+).*?index=(\w+)", expr_str)
                if array_match:
                    array_name = array_match.group(1)
                    array_match.group(2)
                    return f"{array_name}"  # Just return the array name for now
            return expr_str

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
            and not self.required_sync_helpers
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

        if self.required_sync_helpers:
            code += "# CrossGL synchronization placeholders\n"
            for helper_name in sorted(self.required_sync_helpers):
                code += self.generate_sync_helper(helper_name)
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
            "fn _crossgl_dispatch_mesh[X: AnyType, Y: AnyType, Z: AnyType]("
            "x: X, y: Y, z: Z):\n"
            "    pass\n\n"
            "fn _crossgl_dispatch_mesh[X: AnyType, Y: AnyType, Z: AnyType, "
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
        if isinstance(expr, str):
            if expr in MOJO_BUILTIN_PLACEHOLDERS and expr not in self.variable_types:
                return MOJO_BUILTIN_PLACEHOLDERS[expr][1]
            return self.identifier_replacement_result_type(
                expr
            ) or self.variable_types.get(expr)
        if isinstance(expr, VariableNode) and hasattr(expr, "name"):
            if (
                expr.name in MOJO_BUILTIN_PLACEHOLDERS
                and expr.name not in self.variable_types
            ):
                return MOJO_BUILTIN_PLACEHOLDERS[expr.name][1]
            return self.identifier_replacement_result_type(
                expr.name
            ) or self.variable_types.get(expr.name)
        if isinstance(expr, ArrayLiteralNode):
            element_type = self.infer_array_literal_element_type(expr)
            return self.array_type_name(element_type, len(expr.elements))
        if isinstance(expr, MatchNode):
            arms = getattr(expr, "arms", []) or []
            self.validate_match_expression_arms(
                arms, self.expression_result_type(getattr(expr, "expression", None))
            )
            for arm in arms:
                arm_expr, _ = self.match_arm_value_expression(arm)
                arm_type = self.expression_result_type(arm_expr)
                if arm_type is not None:
                    return arm_type
            return None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_result_type(expr.array)
            patch_info = self.tessellation_patch_type_info(array_type)
            if patch_info is not None:
                _patch_name, element_type, _point_count = patch_info
                return element_type
            array_element_type = self.array_element_type(array_type)
            if array_element_type is not None:
                return array_element_type
            pointer_element_type = self.pointer_element_type(array_type)
            if pointer_element_type is not None:
                return pointer_element_type
            matrix_info = self.matrix_type_info(array_type)
            if matrix_info is not None:
                dtype, _, rows = matrix_info
                return self.vector_type_name_for_dtype_width(dtype, rows)
            vector_info = self.vector_type_info(array_type)
            if vector_info is not None:
                return MOJO_DTYPE_INFO[vector_info[0]][0]
            return None
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(expr.op)
            operand_type = self.expression_result_type(expr.operand)
            if op == "not":
                info = self.vector_type_info(operand_type)
                if info is not None and info[0] == "DType.bool":
                    return operand_type
                return "bool"
            if op in {"+", "-", "~"}:
                return operand_type
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            return self.binary_expression_result_type(expr.op, left_type, right_type)
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            if true_type == false_type:
                return true_type
            if true_type is None:
                return false_type
            if false_type is None:
                return true_type
            return None
        if isinstance(expr, FunctionCallNode):
            member_result_type = self.member_function_result_type(expr)
            if member_result_type is not None:
                return member_result_type

            func_name = self.function_call_name(expr)
            if func_name in {"any", "all"} and not self.is_user_defined_function(
                func_name
            ):
                return "bool"
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type is not None:
                return numeric_result_type
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                expr.args,
            )
            if specialized_func_name in self.function_return_types:
                return self.selected_function_return_type(
                    specialized_func_name,
                    expr.args,
                )
            if func_name in self.enum_variant_result_types:
                return self.enum_variant_result_types[func_name]
            if func_name in self.struct_types:
                return func_name
            if func_name in self.vector_constructor_info:
                return func_name
            vector_func_name = self.normalize_generic_vector_type_name(func_name)
            if vector_func_name in self.vector_constructor_info:
                return vector_func_name
            if func_name in MOJO_MATRIX_TYPES:
                return func_name
            if func_name in self.function_return_types:
                return self.selected_function_return_type(func_name, expr.args)
            if func_name in self.scalar_constructor_map:
                return func_name
            if func_name in {"fract", "frac"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "roundEven" and len(expr.args) == 1:
                return self.round_even_result_type(expr.args[0])
            if func_name in {"inverse", "transpose"} and expr.args:
                return self.expression_result_type(expr.args[0])
            if func_name in {"length", "dot"}:
                return "float"
            if func_name == "normalize" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"inversesqrt", "inverseSqrt", "rsqrt"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"mix", "lerp"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"degrees", "radians"} and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"atan", "atan2"} and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "pow" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "mod" and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"min", "max", "clamp"} and expr.args:
                for arg in expr.args:
                    arg_type = self.expression_result_type(arg)
                    if self.vector_type_info(arg_type) is not None:
                        return arg_type
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name == "saturate" and expr.args:
                return self.expression_result_type(expr.args[0]) or "float"
            if func_name in {"texture", "textureLod", "textureGrad"}:
                args = self.strip_split_sampler_arg(expr.args)
                if args:
                    resource_type = self.map_type(self.expression_result_type(args[0]))
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name in {"texelFetch", "texelFetchOffset"}:
                if expr.args:
                    resource_type = self.map_type(
                        self.expression_result_type(expr.args[0])
                    )
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name == "textureQueryLevels":
                return "int"
            if func_name in {"imageSamples", "textureSamples"}:
                return "int"
            if func_name in {"textureSize", "imageSize"} and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                size_type = self.resource_size_return_type(resource_type)
                if size_type == "Int32":
                    return "int"
                if size_type == "SIMD[DType.int32, 2]":
                    return "ivec2"
                if size_type == "SIMD[DType.int32, 4]":
                    return "ivec3"
            if func_name == "imageLoad" and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                return self.image_result_type_name(resource_type)
            if func_name == "imageStore":
                return "void"
            if func_name == "buffer_load" and expr.args:
                info = self.buffer_resource_info(
                    self.expression_result_type(expr.args[0])
                )
                if info is not None:
                    return self.buffer_helper_element_type(*info)
            if func_name == "buffer_consume" and expr.args:
                info = self.buffer_resource_info(
                    self.expression_result_type(expr.args[0])
                )
                if info is not None:
                    return info[1]
            if func_name in {"buffer_store", "buffer_append", "buffer_dimensions"}:
                return "void"
            reinterpret_name = self.mojo_reinterpret_builtin_name(func_name)
            if reinterpret_name is not None:
                source_type = (
                    self.expression_result_type(expr.args[0]) if expr.args else None
                )
                return self.reinterpret_return_type_name(reinterpret_name, source_type)
            if func_name in MOJO_GENERIC_TEXTURE_BUILTINS:
                _, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[func_name]
                if return_kind == "float":
                    return "float"
                if return_kind == "vec2":
                    return "vec2"
                if expr.args:
                    resource_type = self.map_type(
                        self.expression_result_type(expr.args[0])
                    )
                    if self.is_texture_resource_type(resource_type):
                        return self.texture_result_type_name(resource_type)
                return "vec4"
            if func_name in MOJO_SCALAR_ATOMIC_OPS and expr.args:
                target_type = self.type_name(self.expression_result_type(expr.args[0]))
                if target_type in MOJO_SCALAR_ATOMIC_TYPES:
                    return target_type
            if func_name in MOJO_IMAGE_ATOMIC_BUILTINS and expr.args:
                resource_type = self.map_type(self.expression_result_type(expr.args[0]))
                value_type = self.image_value_type(resource_type)
                if value_type == "UInt32":
                    return "uint"
                return "int"
            if func_name == "__shfl_down_sync" and len(expr.args) >= 2:
                return self.expression_result_type(expr.args[1])
            return self.function_return_types.get(func_name)
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
        if isinstance(expr, TextureOpNode):
            operation = getattr(expr, "operation", "")
            if operation in {
                "SampleCmp",
                "SampleCmpLevel",
                "SampleCmpLevelZero",
                "SampleCmpGrad",
                "textureCompare",
                "textureCompareOffset",
                "textureCompareLod",
                "textureCompareLodOffset",
                "textureCompareGrad",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
            }:
                return "float"
            if operation == "textureQueryLod":
                return "vec2"
            if operation in MOJO_GENERIC_TEXTURE_BUILTINS:
                _, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[operation]
                if return_kind == "float":
                    return "float"
                if return_kind == "vec2":
                    return "vec2"
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                if self.is_texture_resource_type(resource_type):
                    return self.texture_result_type_name(resource_type)
                return "vec4"
            if operation == "textureQueryLevels":
                return "int"
            if operation in {"imageSamples", "textureSamples"}:
                return "int"
            if operation in {"GetDimensions", "textureSize"}:
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                size_type = self.resource_size_return_type(resource_type)
                if size_type == "Int32":
                    return "int"
                if size_type == "SIMD[DType.int32, 2]":
                    return "ivec2"
                if size_type == "SIMD[DType.int32, 4]":
                    return "ivec3"
            if operation in MOJO_ATOMIC_OP_ALIASES:
                resource_type = self.map_type(
                    self.expression_result_type(expr.texture_expr)
                )
                value_type = self.image_value_type(resource_type)
                if value_type == "UInt32":
                    return "uint"
                return "int"
            resource_type = self.map_type(
                self.expression_result_type(expr.texture_expr)
            )
            if self.is_texture_resource_type(resource_type):
                return self.texture_result_type_name(resource_type)
            return "vec4"
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
