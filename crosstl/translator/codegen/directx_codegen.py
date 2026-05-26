"""CrossGL-to-HLSL code generator."""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    ConstructorNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PreprocessorNode,
    RayQueryOpNode,
    RayTracingOpNode,
    RangeNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
    ArrayAccessNode,
    ArrayNode,
)
from .array_utils import (
    parse_array_type,
    format_c_style_array_declaration,
    split_array_type_suffix,
    get_array_size_from_node,
    evaluate_literal_int_expression,
    collect_literal_int_constants,
    collect_struct_member_types,
)
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gather_component_argument_index,
    texture_gradient_argument_indices,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
    texture_sample_index_argument_index,
)
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_matches,
)
from .resource_arrays import collect_resource_array_size_hints
from .enum_utils import (
    collect_enum_type_names,
    collect_enum_struct_variant_fields,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_enum_variant_constants,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_value_expression,
    generate_enum_constructor_expression,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_constants,
    generate_generic_enum_structs,
    generate_enum_constructor_call,
    generate_enum_constructor_functions,
    generate_enum_constants,
    generate_enum_structs,
    generic_enum_specialized_type_name,
    infer_enum_constructor_type,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generate_generic_structs,
    generate_struct_constructor_expression,
    generic_struct_specialized_type_name,
    infer_struct_constructor_type,
)
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .glsl_buffer_layout import (
    byte_offset_expression,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_compound_binary_operator,
    glsl_buffer_block_node_type,
    matrix_column_offsets,
)
from .image_access_contracts import (
    TEXTURE_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_INTRINSIC_NAMES,
    TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES,
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES,
    TEXTURE_SAMPLING_INTRINSIC_NAMES,
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    explicit_image_format,
    floating_coordinate_dimension_from_type_name,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_format_or_default_channel_count,
    image_format_channel_count,
    image_format_component_kind,
    image_format_result_type,
    image_format_component_type,
    image_format_vector_type,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    is_image_format_attribute,
    is_image_atomic_operation,
    is_image_resource_operation,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_projected_texture_operation,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_resource_access_attribute,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_compare_basic_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_compare_offset_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_non_projected_offset_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_single_offset_operation,
    is_texture_query_lod_operation,
    is_texture_query_levels_operation,
    is_texture_sample_basic_offset_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_offset_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_offset_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_operation,
    is_texture_sample_offset_operation,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    image_resource_metadata,
    record_explicit_image_metadata,
    resolve_image_atomic_component_kind,
    resource_query_scalar_constant_helper_descriptor,
    resource_query_scalar_helper_descriptor,
    resource_query_size_components_descriptor,
    resource_query_size_helper_descriptor,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_store_vector_constructor,
    storage_image_zero_values,
    supported_image_formats,
    projected_texture_extra_argument_count_error,
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
    texture_compare_argument_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_query_levels_multisample_expression,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_samples_query_expression,
    texture_sample_offset_extra_argument_count_error,
    texture_sample_offset_capability_error,
    texture_multisample_sample_type_error,
    validate_texture_operation_arity,
    requires_integer_coordinate,
    unsupported_image_atomic_expression,
    unsupported_multisample_image_atomic_expression,
    unsupported_multisample_image_store_expression,
    unsupported_cube_texel_fetch_expression,
    unsupported_multisample_texture_call_vector_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_multisample_texel_fetch_offset_expression,
    unsupported_projected_texture_call_expression,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_offset_call_expression,
    unsupported_texture_offset_operation_error,
    unsupported_projected_texture_operation_error,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)


class HLSLCodeGen:
    """Emit HLSL source from the shared CrossGL translator AST."""

    def __init__(self):
        """Initialize DirectX type maps and per-generation resource state."""
        self.texture_variables = set()
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_implicit_texture_samplers = {}
        self.current_implicit_texture_regular_samplers = {}
        self.current_implicit_texture_query_lod_samplers = {}
        self.global_implicit_texture_regular_samplers = {}
        self.global_implicit_texture_query_lod_samplers = {}
        self.required_texture_query_helpers = set()
        self.required_image_atomic_helpers = set()
        self.comparison_sampler_parameters = {}
        self.regular_sampler_parameters = {}
        self.implicit_texture_sampler_parameters = {}
        self.function_parameter_names = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.current_generic_function_substitutions = {}
        self.local_variable_types = {}
        self.struct_member_types = {}
        self.structs_by_name = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.current_hlsl_available_functions = {}
        self.current_hlsl_hull_output_control_points = None
        self.current_hlsl_hull_domain = None
        self.hlsl_temp_variable_index = 0
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_glsl_buffer_block_parameters = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.lowered_glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.comparison_sampler_struct_members = set()
        self.regular_sampler_struct_members = set()
        self.type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float3x2",
            "mat2x4": "float4x2",
            "mat3x2": "float2x3",
            "mat3x3": "float3x3",
            "mat3x4": "float4x3",
            "mat4x2": "float2x4",
            "mat4x3": "float3x4",
            "mat4x4": "float4x4",
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "float": "float",
            "packed_float2": "float2",
            "packed_float3": "float3",
            "packed_float4": "float4",
            "simd_float2": "float2",
            "simd_float3": "float3",
            "simd_float4": "float4",
            "half": "half",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "packed_half2": "half2",
            "packed_half3": "half3",
            "packed_half4": "half4",
            "float16": "half",
            "f16vec2": "half2",
            "f16vec3": "half3",
            "f16vec4": "half4",
            "double": "double",
            "str": "int",
            "char": "int",
            "signed char": "int",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "uint8": "uint",
            "uint8_t": "uint",
            "short": "min16int",
            "signed short": "min16int",
            "ushort": "min16uint",
            "unsigned short": "min16uint",
            "int16": "min16int",
            "int16_t": "min16int",
            "int32_t": "int",
            "int64": "int64_t",
            "int64_t": "int64_t",
            "long": "int64_t",
            "signed long": "int64_t",
            "ptrdiff_t": "int64_t",
            "uint16": "min16uint",
            "uint16_t": "min16uint",
            "uint32_t": "uint",
            "uint64": "uint64_t",
            "uint64_t": "uint64_t",
            "ulong": "uint64_t",
            "unsigned long": "uint64_t",
            "size_t": "uint64_t",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "packed_int2": "int2",
            "packed_int3": "int3",
            "packed_int4": "int4",
            "simd_int2": "int2",
            "simd_int3": "int3",
            "simd_int4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
            "packed_uint2": "uint2",
            "packed_uint3": "uint3",
            "packed_uint4": "uint4",
            "simd_uint2": "uint2",
            "simd_uint3": "uint3",
            "simd_uint4": "uint4",
            "i8vec2": "int2",
            "i8vec3": "int3",
            "i8vec4": "int4",
            "u8vec2": "uint2",
            "u8vec3": "uint3",
            "u8vec4": "uint4",
            "short2": "min16int2",
            "short3": "min16int3",
            "short4": "min16int4",
            "ushort2": "min16uint2",
            "ushort3": "min16uint3",
            "ushort4": "min16uint4",
            "i16vec2": "min16int2",
            "i16vec3": "min16int3",
            "i16vec4": "min16int4",
            "u16vec2": "min16uint2",
            "u16vec3": "min16uint3",
            "u16vec4": "min16uint4",
            "f16mat2": "half2x2",
            "f16mat3": "half3x3",
            "f16mat4": "half4x4",
            "f16mat2x2": "half2x2",
            "f16mat2x3": "half3x2",
            "f16mat2x4": "half4x2",
            "f16mat3x2": "half2x3",
            "f16mat3x3": "half3x3",
            "f16mat3x4": "half4x3",
            "f16mat4x2": "half2x4",
            "f16mat4x3": "half3x4",
            "f16mat4x4": "half4x4",
            "simd_float2x2": "float2x2",
            "simd_float2x3": "float2x3",
            "simd_float2x4": "float2x4",
            "simd_float3x2": "float3x2",
            "simd_float3x3": "float3x3",
            "simd_float3x4": "float3x4",
            "simd_float4x2": "float4x2",
            "simd_float4x3": "float4x3",
            "simd_float4x4": "float4x4",
            "sampler1D": "Texture1D",
            "sampler1DArray": "Texture1DArray",
            "sampler2D": "Texture2D",
            "sampler3D": "Texture3D",
            "samplerCube": "TextureCube",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "sampler2DMS": "Texture2DMS<float4>",
            "sampler2DMSArray": "Texture2DMSArray<float4>",
            "sampler2DShadow": "Texture2D",
            "sampler2DArrayShadow": "Texture2DArray",
            "samplerCubeShadow": "TextureCube",
            "samplerCubeArrayShadow": "TextureCubeArray",
            "iimage1D": "RWTexture1D<int>",
            "iimage1DArray": "RWTexture1DArray<int>",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "uimage1D": "RWTexture1D<uint>",
            "uimage1DArray": "RWTexture1DArray<uint>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "image1D": "RWTexture1D<float4>",
            "image1DArray": "RWTexture1DArray<float4>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "imageCube": "RWTextureCube<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
            "sampler": "SamplerState",
        }

        self.semantic_map = {
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_IsFrontFace": "FRONT_FACE",
            "gl_PrimitiveID": "PRIMITIVE_ID",
            "gl_ViewID": "SV_ViewID",
            "gl_Layer": "SV_RenderTargetArrayIndex",
            "gl_ViewportIndex": "SV_ViewportArrayIndex",
            "InstanceID": "INSTANCE_ID",
            "VertexID": "VERTEX_ID",
            "gl_Position": "SV_POSITION",
            "gl_PointSize": "SV_POINTSIZE",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_FragColor": "SV_TARGET",
            "gl_FragColor0": "SV_TARGET0",
            "gl_FragColor1": "SV_TARGET1",
            "gl_FragColor2": "SV_TARGET2",
            "gl_FragColor3": "SV_TARGET3",
            "gl_FragColor4": "SV_TARGET4",
            "gl_FragColor5": "SV_TARGET5",
            "gl_FragColor6": "SV_TARGET6",
            "gl_FragColor7": "SV_TARGET7",
            "gl_FragDepth": "SV_DEPTH",
            "payload": "payload",
            "hit_attribute": "hit_attribute",
            "callable_data": "callable_data",
            "shader_record": "shader_record",
        }

    def generate(self, ast):
        """Generate complete HLSL source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        """Generate HLSL source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to HLSL, optionally filtering stage entry points."""
        target_stage = normalize_stage_name(target_stage)

        self.texture_variables = set()
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.current_texture_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.current_implicit_texture_samplers = {}
        self.current_implicit_texture_regular_samplers = {}
        self.current_implicit_texture_query_lod_samplers = {}
        self.global_implicit_texture_regular_samplers = {}
        self.global_implicit_texture_query_lod_samplers = {}
        self.required_texture_query_helpers = set()
        self.required_image_atomic_helpers = set()
        self.comparison_sampler_parameters = {}
        self.regular_sampler_parameters = {}
        self.implicit_texture_sampler_parameters = {}
        self.function_image_access_requirements = {}
        self.unsupported_glsl_buffer_block_functions = {}
        self.unsupported_glsl_buffer_block_struct_names = set()
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.local_variable_types = {}
        self.current_global_resource_declaration_nodes = None
        self.current_hlsl_available_functions = {}
        self.current_hlsl_hull_output_control_points = None
        self.current_hlsl_hull_domain = None
        self.hlsl_temp_variable_index = 0
        self.current_glsl_buffer_block_parameters = {}
        self.unsupported_glsl_buffer_block_variables = set()
        self.unsupported_glsl_buffer_block_variable_types = {}
        self.current_unsupported_glsl_buffer_block_parameters = set()
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_glsl_buffer_block_parameter_failures = {}
        self.current_glsl_buffer_block_parameter_struct_failures = {}
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )
        self.generic_struct_specializations = collect_generic_struct_specializations(
            ast,
            self.generic_struct_definitions,
            self.type_name_string,
        )
        self.plain_enums = collect_plain_enums(structs)
        self.struct_payload_enums = collect_struct_payload_enums(structs)
        self.enum_type_names = collect_enum_type_names(self.plain_enums)
        self.enum_struct_type_names = (
            collect_enum_type_names(self.struct_payload_enums)
            | set(self.generic_enum_struct_definitions)
            | {
                specialization["struct_name"]
                for specialization in self.generic_enum_specializations.values()
            }
        )
        self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constructors = collect_enum_variant_constructors(
            self.struct_payload_enums
        )
        self.enum_variant_constructor_fields = collect_enum_variant_constructor_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constants = {
            **collect_enum_variant_constants(
                self.plain_enums + self.struct_payload_enums
            ),
            **collect_generic_enum_variant_constants(
                self.generic_enum_struct_definitions
            ),
        }
        global_vars = self.global_resource_declaration_nodes(ast, target_stage)
        self.current_global_resource_declaration_nodes = global_vars
        functions = self.collect_functions(ast)
        self.function_return_types = {
            func.name: self.type_name_string(getattr(func, "return_type", "void"))
            for func in functions
            if getattr(func, "name", None)
        }
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                list(global_vars) + self.collect_function_parameters(functions)
            )
        )
        (
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        ) = collect_lowered_glsl_buffer_blocks(
            global_vars,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="hlsl_type",
            unsupported_type_message=(
                "type is not supported by ByteAddressBuffer lowering"
            ),
        )
        self.lowered_glsl_buffer_block_struct_names = {
            block["type_name"] for block in self.lowered_glsl_buffer_blocks.values()
        }
        self.unsupported_glsl_buffer_block_struct_names = set(
            self.glsl_buffer_block_struct_lowering_failures
        )
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        self.struct_member_types.update(
            collect_generic_enum_specialization_member_types(
                self,
                self.generic_enum_specializations,
            )
        )
        self.struct_member_types.update(
            collect_generic_struct_specialization_member_types(
                self,
                self.generic_struct_specializations,
            )
        )
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            functions,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        self.comparison_sampler_parameters = self.collect_comparison_sampler_parameters(
            ast
        )
        self.regular_sampler_parameters = self.collect_regular_sampler_parameters(ast)
        self.comparison_sampler_struct_members = self.collect_sampler_struct_members(
            ast,
            self.comparison_sampler_parameters,
            self.comparison_texture_function_names(),
        )
        self.regular_sampler_struct_members = self.collect_sampler_struct_members(
            ast,
            self.regular_sampler_parameters,
            self.regular_texture_function_names(),
        )
        self.function_parameter_names = collect_function_parameter_names(functions)
        if generic_function_specializations:
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                functions,
                self.function_parameter_names,
                self.walk_ast,
                self.function_call_name,
                self.expression_name,
            )
        )
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        _, _, parameter_struct_failures = (
            self.collect_lowered_glsl_buffer_block_parameters(
                self.collect_function_parameters(functions)
            )
        )
        self.unsupported_glsl_buffer_block_struct_names.update(
            parameter_struct_failures
        )
        self.unsupported_glsl_buffer_block_functions = (
            self.collect_unsupported_glsl_buffer_block_functions(functions)
        )
        self.validate_global_resource_shadows(ast)
        self.validate_explicit_sampler_role_conflicts(ast)
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        for directive in preprocessors:
            if isinstance(directive, PreprocessorNode):
                line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line:
                code += f"{line}\n"

        code += generate_enum_constants(
            self, self.plain_enums + self.struct_payload_enums
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
        )
        code += self.generate_constants(ast)
        code += generate_generic_structs(self, self.generic_struct_specializations)
        code += generate_enum_structs(self, self.struct_payload_enums)
        code += generate_generic_enum_structs(self, self.generic_enum_specializations)
        code += generate_enum_constructor_functions(self, self.struct_payload_enums)
        code += generate_generic_enum_constructor_functions(
            self,
            self.generic_enum_specializations,
        )

        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                if node.name in self.generic_struct_definitions:
                    continue
                if node.name in self.lowered_glsl_buffer_block_struct_names:
                    continue
                if node.name in self.glsl_buffer_block_struct_names:
                    code += self.glsl_buffer_block_diagnostic(
                        "HLSL", node.name, None, None
                    )
                    code += self.unsupported_glsl_buffer_block_struct_placeholder(
                        "HLSL", node.name
                    )
                    continue
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        member_type = self.map_struct_member_type(
                            node.name, member.name, element_type
                        )
                        if member.size:
                            declaration_type = f"{member_type}[{member.size}]"
                        else:
                            # Dynamic arrays in HLSL
                            declaration_type = f"{member_type}[]"
                        semantic = self.semantic_from_array_node(member)
                        self.validate_hlsl_struct_member_semantic_type(
                            node.name, member.name, declaration_type, semantic
                        )
                        semantic_attr = self.map_semantic(semantic)
                        tess_factor_declaration = (
                            self.hlsl_tess_factor_member_declaration(
                                declaration_type, member.name, semantic
                            )
                        )
                        if tess_factor_declaration is not None:
                            code += f"    {tess_factor_declaration} {semantic_attr};\n"
                        else:
                            declaration = format_c_style_array_declaration(
                                declaration_type, member.name
                            )
                            code += f"    {declaration}{semantic_attr};\n"
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            # New AST structure - check if it's an ArrayType
                            if str(type(member.member_type)).find("ArrayType") != -1:
                                member_type_str = self.convert_type_node_to_string(
                                    member.member_type
                                )
                                member_type = self.map_struct_member_type(
                                    node.name, member.name, member_type_str
                                )
                                array_syntax = None
                            else:
                                # Regular type - pass TypeNode directly to map_type
                                member_type = self.map_struct_member_type(
                                    node.name, member.name, member.member_type
                                )
                                array_syntax = ""
                        elif hasattr(member, "vtype"):
                            # Old AST structure
                            member_type = self.map_struct_member_type(
                                node.name, member.name, member.vtype
                            )
                            array_syntax = ""
                        else:
                            member_type = "float"
                            array_syntax = ""

                        semantic = self.semantic_from_node(member)
                        if semantic is None and getattr(member, "name", "") in [
                            "view",
                            "layer",
                            "viewport",
                        ]:
                            # Fallback to name-based mapping for common multiview outputs
                            name_semantics = {
                                "view": "gl_ViewID",
                                "layer": "gl_Layer",
                                "viewport": "gl_ViewportIndex",
                            }
                            semantic = name_semantics.get(member.name)

                        self.validate_hlsl_struct_member_semantic_type(
                            node.name, member.name, member_type, semantic
                        )
                        semantic_attr = self.map_semantic(semantic)
                        tess_factor_declaration = (
                            self.hlsl_tess_factor_member_declaration(
                                member_type, member.name, semantic
                            )
                        )
                        if tess_factor_declaration is not None:
                            code += f"    {tess_factor_declaration} {semantic_attr};\n"
                            continue

                        if array_syntax is None:
                            declaration = format_c_style_array_declaration(
                                member_type, member.name
                            )
                            code += f"    {declaration}{semantic_attr};\n"
                        else:
                            code += f"    {member_type} {member.name}{array_syntax}{semantic_attr};\n"
                code += "};\n"

        comparison_texture_names, comparison_sampler_names = (
            self.collect_comparison_resources(ast)
        )
        self.implicit_texture_sampler_parameters = (
            self.collect_implicit_texture_sampler_parameters(ast)
        )
        comparison_sampler_names |= self.collect_comparison_sampler_arguments(
            ast, self.comparison_sampler_parameters
        )
        comparison_texture_names |= self.collect_implicit_comparison_texture_arguments(
            ast, self.implicit_texture_sampler_parameters
        )
        sampler_parameter_names = self.collect_sampler_parameter_names(ast)
        declared_sampler_names = set()
        explicit_sampler_names = set()
        for node in global_vars:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_sampler_type(var_type):
                declared_sampler_names.add(var_name)
                explicit_sampler_names.add(var_name)
        comparison_sampler_names |= {
            f"{texture_name}Sampler"
            for texture_name in comparison_texture_names
            if f"{texture_name}Sampler" in declared_sampler_names
        }
        global_implicit_sampler_texture_names = (
            self.collect_global_implicit_sampler_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        global_implicit_regular_sampler_texture_names = (
            self.collect_global_implicit_regular_sampler_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )
        global_implicit_query_lod_texture_names = (
            self.collect_global_implicit_query_lod_texture_names(
                ast,
                self.collect_global_texture_names(ast),
                declared_sampler_names | sampler_parameter_names,
                self.implicit_texture_sampler_parameters,
            )
        )

        texture_registers = {}
        sampler_registers = {}
        uav_registers = {}
        used_resource_registers = {}
        self.reserve_explicit_global_resource_registers(
            global_vars, used_resource_registers
        )
        for i, node in enumerate(global_vars):
            # Handle both old and new AST variable structures
            resource_count = 1
            if hasattr(node, "var_type"):
                if hasattr(node.var_type, "name") or hasattr(
                    node.var_type, "element_type"
                ):
                    # Check if it's an ArrayType and handle specially for global variables
                    if (
                        hasattr(node.var_type, "element_type")
                        and str(type(node.var_type)).find("ArrayType") != -1
                    ):  # ArrayType
                        base_type = self.convert_type_node_to_string(
                            node.var_type.element_type
                        )
                        array_size = (
                            self.generate_expression(node.var_type.size)
                            if node.var_type.size
                            else self.resource_array_size_hints.get(node.name, "")
                        )
                        vtype = base_type
                        array_suffix = f"[{array_size}]" if array_size else "[]"
                        resource_count = self.resource_array_count(
                            node.var_type.size if node.var_type.size else array_size
                        )
                    else:
                        # Use the proper type conversion for TypeNode objects
                        vtype = self.convert_type_node_to_string(node.var_type)
                        array_suffix = ""
                else:
                    vtype = str(node.var_type)
                    array_suffix = ""
            elif hasattr(node, "vtype"):
                vtype = node.vtype
                array_suffix = ""
            else:
                vtype = "float"
                array_suffix = ""

            if hasattr(node, "name"):
                var_name = node.name
            elif hasattr(node, "variable_name"):
                var_name = node.variable_name
            else:
                var_name = f"var{i}"

            lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
            if lowered_block is not None:
                mapped_type = (
                    "ByteAddressBuffer"
                    if lowered_block["readonly"]
                    else "RWByteAddressBuffer"
                )
                if lowered_block["readonly"]:
                    space = self.explicit_resource_register_space(node)
                    binding = self.explicit_resource_binding_index(
                        node, {"binding", "texture"}, ("t",)
                    )
                    if binding is None:
                        binding = self.next_available_resource_register(
                            used_resource_registers,
                            "t",
                            texture_registers,
                            space,
                            resource_count,
                        )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "t",
                        binding,
                        resource_count,
                        var_name,
                        space,
                    )
                    register = self.resource_register_suffix("t", binding, node)
                    self.advance_resource_register(
                        texture_registers, space, binding, resource_count
                    )
                else:
                    space = self.explicit_resource_register_space(node)
                    binding = self.explicit_resource_binding_index(
                        node, {"binding", "texture", "uav"}, ("u",)
                    )
                    if binding is None:
                        binding = self.next_available_resource_register(
                            used_resource_registers,
                            "u",
                            uav_registers,
                            space,
                            resource_count,
                        )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "u",
                        binding,
                        resource_count,
                        var_name,
                        space,
                    )
                    register = self.resource_register_suffix("u", binding, node)
                    self.advance_resource_register(
                        uav_registers, space, binding, resource_count
                    )
                declaration = format_c_style_array_declaration(
                    f"{mapped_type}{array_suffix}", var_name
                )
                qualifier = self.resource_memory_qualifier(mapped_type, node)
                code += f"{qualifier}{declaration}{register};\n"
                continue

            if self.is_glsl_buffer_block_variable(node, vtype):
                code += self.glsl_buffer_block_diagnostic("HLSL", vtype, var_name, node)
                self.unsupported_glsl_buffer_block_variables.add(var_name)
                self.unsupported_glsl_buffer_block_variable_types[var_name] = (
                    self.type_name_string(vtype)
                )
                code += self.unsupported_glsl_buffer_block_variable_placeholder(
                    "HLSL", vtype, var_name
                )
                continue

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if var_name in comparison_sampler_names and mapped_type == "SamplerState":
                mapped_type = "SamplerComparisonState"
            declaration_type = self.directx_resource_declaration_type(
                f"{mapped_type}{array_suffix}"
            )
            declaration = format_c_style_array_declaration(declaration_type, var_name)
            register = ""
            if mapped_type.startswith(
                "Texture"
            ) or self.is_multisample_storage_image_resource_type(mapped_type):
                self.texture_variables.add(var_name)
                self.texture_variable_types[var_name] = mapped_type
                if self.is_image_type(vtype):
                    record_explicit_image_metadata(
                        var_name,
                        node,
                        self.attribute_value_to_string,
                        image_formats=self.image_variable_formats,
                        image_accesses=self.image_variable_accesses,
                    )
                space = self.explicit_resource_register_space(node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture"}, ("t",)
                )
                if binding is None:
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "t",
                        texture_registers,
                        space,
                        resource_count,
                    )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "t",
                    binding,
                    resource_count,
                    var_name,
                    space,
                )
                register = self.resource_register_suffix("t", binding, node)
                self.advance_resource_register(
                    texture_registers, space, binding, resource_count
                )
            elif mapped_type.startswith("RWTexture"):
                self.texture_variable_types[var_name] = mapped_type
                record_explicit_image_metadata(
                    var_name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                    image_accesses=self.image_variable_accesses,
                )
                space = self.explicit_resource_register_space(node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture", "uav"}, ("u",)
                )
                if binding is None:
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "u",
                        uav_registers,
                        space,
                        resource_count,
                    )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "u",
                    binding,
                    resource_count,
                    var_name,
                    space,
                )
                register = self.resource_register_suffix("u", binding, node)
                self.advance_resource_register(
                    uav_registers, space, binding, resource_count
                )
            elif self.is_hlsl_uav_buffer_type(mapped_type):
                space = self.explicit_resource_register_space(node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture", "uav"}, ("u",)
                )
                if binding is None:
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "u",
                        uav_registers,
                        space,
                        resource_count,
                    )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "u",
                    binding,
                    resource_count,
                    var_name,
                    space,
                )
                register = self.resource_register_suffix("u", binding, node)
                self.advance_resource_register(
                    uav_registers, space, binding, resource_count
                )
            elif self.is_hlsl_readonly_buffer_type(mapped_type):
                space = self.explicit_resource_register_space(node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "texture"}, ("t",)
                )
                if binding is None:
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "t",
                        texture_registers,
                        space,
                        resource_count,
                    )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "t",
                    binding,
                    resource_count,
                    var_name,
                    space,
                )
                register = self.resource_register_suffix("t", binding, node)
                self.advance_resource_register(
                    texture_registers, space, binding, resource_count
                )
            elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
                self.sampler_variables.add(var_name)
                space = self.explicit_resource_register_space(node)
                binding = self.explicit_resource_binding_index(
                    node, {"binding", "sampler"}, ("s",)
                )
                if binding is None:
                    binding = self.next_available_resource_register(
                        used_resource_registers,
                        "s",
                        sampler_registers,
                        space,
                        resource_count,
                    )
                self.reserve_resource_register_range(
                    used_resource_registers,
                    "s",
                    binding,
                    resource_count,
                    var_name,
                    space,
                )
                register = self.resource_register_suffix("s", binding, node)
                self.advance_resource_register(
                    sampler_registers, space, binding, resource_count
                )

            qualifier = self.resource_memory_qualifier(mapped_type, node)
            code += f"{qualifier}{declaration}{register};\n"

            if mapped_type.startswith("Texture"):
                sampler_name = f"{var_name}Sampler"
                needs_implicit_sampler = (
                    var_name in global_implicit_sampler_texture_names
                )
                needs_query_lod_sampler = (
                    var_name in global_implicit_query_lod_texture_names
                )
                needs_regular_sampler = (
                    var_name in global_implicit_regular_sampler_texture_names
                )
                needs_comparison_sampler = var_name in comparison_texture_names
                if (
                    needs_implicit_sampler
                    and sampler_name not in explicit_sampler_names
                    and not self.is_multisample_sampler_type(vtype)
                ):
                    sampler_type = (
                        "SamplerComparisonState"
                        if needs_comparison_sampler
                        else "SamplerState"
                    )
                    self.sampler_variables.add(sampler_name)
                    sampler_binding = self.next_available_resource_register(
                        used_resource_registers,
                        "s",
                        sampler_registers,
                        space,
                        1,
                    )
                    self.reserve_resource_register_range(
                        used_resource_registers,
                        "s",
                        sampler_binding,
                        1,
                        sampler_name,
                        space,
                    )
                    sampler_register = self.resource_register_suffix_for_space(
                        "s", sampler_binding, space
                    )
                    code += f"{sampler_type} {sampler_name}{sampler_register};\n"
                    self.advance_resource_register(
                        sampler_registers, space, sampler_binding, 1
                    )
                if needs_regular_sampler and needs_comparison_sampler:
                    regular_sampler_name = self.implicit_regular_sampler_name(
                        var_name,
                        {
                            "comparison": needs_comparison_sampler,
                            "regular": needs_regular_sampler,
                            "sampler_name": sampler_name,
                        },
                    )
                    self.global_implicit_texture_regular_samplers[var_name] = (
                        regular_sampler_name
                    )
                    if (
                        regular_sampler_name not in explicit_sampler_names
                        and not self.is_multisample_sampler_type(vtype)
                    ):
                        self.sampler_variables.add(regular_sampler_name)
                        sampler_binding = self.next_available_resource_register(
                            used_resource_registers,
                            "s",
                            sampler_registers,
                            space,
                            1,
                        )
                        self.reserve_resource_register_range(
                            used_resource_registers,
                            "s",
                            sampler_binding,
                            1,
                            regular_sampler_name,
                            space,
                        )
                        sampler_register = self.resource_register_suffix_for_space(
                            "s", sampler_binding, space
                        )
                        code += (
                            f"SamplerState {regular_sampler_name}{sampler_register};\n"
                        )
                        self.advance_resource_register(
                            sampler_registers, space, sampler_binding, 1
                        )
                if needs_query_lod_sampler:
                    query_sampler_name = (
                        f"{var_name}QuerySampler"
                        if needs_comparison_sampler
                        else sampler_name
                    )
                    self.global_implicit_texture_query_lod_samplers[var_name] = (
                        query_sampler_name
                    )
                    if (
                        needs_comparison_sampler
                        and query_sampler_name not in explicit_sampler_names
                        and not self.is_multisample_sampler_type(vtype)
                    ):
                        self.sampler_variables.add(query_sampler_name)
                        sampler_binding = self.next_available_resource_register(
                            used_resource_registers,
                            "s",
                            sampler_registers,
                            space,
                            1,
                        )
                        self.reserve_resource_register_range(
                            used_resource_registers,
                            "s",
                            sampler_binding,
                            1,
                            query_sampler_name,
                            space,
                        )
                        sampler_register = self.resource_register_suffix_for_space(
                            "s", sampler_binding, space
                        )
                        code += (
                            f"SamplerState {query_sampler_name}{sampler_register};\n"
                        )
                        self.advance_resource_register(
                            sampler_registers, space, sampler_binding, 1
                        )

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        stage_entry_names = self.stage_entry_names(ast, target_stage)
        self.current_hlsl_hull_output_control_points = (
            self.hlsl_program_hull_output_control_points(ast)
        )
        self.current_hlsl_hull_domain = self.hlsl_program_hull_domain(ast)

        functions = getattr(ast, "functions", [])
        global_functions_by_name = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        self.current_hlsl_available_functions = global_functions_by_name
        functions_code = ""
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    functions_code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                functions_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                functions_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                functions_code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=stage_entry_names.get(id(func)),
                )
            else:
                functions_code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                local_functions = getattr(stage, "local_functions", []) or []
                stage_functions_by_name = dict(global_functions_by_name)
                stage_functions_by_name.update(
                    {
                        func.name: func
                        for func in local_functions
                        if getattr(func, "name", None)
                    }
                )

                for func in order_functions_by_dependencies(
                    local_functions,
                    self.walk_ast,
                    self.function_call_name,
                    FunctionCallNode,
                ):
                    if generic_function_parameters(func):
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            functions_code += self.generate_function(specialized_func)
                    else:
                        functions_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    previous_available_functions = self.current_hlsl_available_functions
                    self.current_hlsl_available_functions = {
                        name: func
                        for name, func in stage_functions_by_name.items()
                        if func is not stage.entry_point
                    }
                    try:
                        functions_code += self.generate_function(
                            stage.entry_point,
                            shader_type=stage_name,
                            execution_config=getattr(stage, "execution_config", None),
                            entry_name=stage_entry_names.get(id(stage.entry_point)),
                        )
                    finally:
                        self.current_hlsl_available_functions = (
                            previous_available_functions
                        )

        code += self.generate_texture_query_helpers()
        code += self.generate_image_atomic_helpers()
        code += functions_code

        return code

    def generate_constants(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            code += f"static const {self.map_type(const_type)} {name} = {value_code};\n"

        return f"{code}\n" if code else ""

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate cbuffer name(s) in DirectX output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise ValueError(
                "Cbuffer name(s) conflict with existing DirectX declaration(s): "
                f"{names}"
            )

        duplicate_members = collect_duplicate_cbuffer_member_names(cbuffers)
        if duplicate_members:
            names = ", ".join(sorted(duplicate_members))
            raise ValueError(
                f"Ambiguous cbuffer member name(s) in DirectX output: {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise ValueError(
                "Cbuffer member name(s) conflict with DirectX global declaration(s): "
                f"{names}"
            )
        used_cbuffer_registers = {}
        buffer_registers = {}
        for node in cbuffers:
            space = self.explicit_resource_register_space(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                continue
            self.reserve_resource_register_range(
                used_cbuffer_registers,
                "b",
                binding,
                1,
                node.name,
                space,
            )
        for node in cbuffers:
            space = self.explicit_resource_register_space(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b",)
            )
            if binding is None:
                binding = self.next_available_resource_register(
                    used_cbuffer_registers,
                    "b",
                    buffer_registers,
                    space,
                    1,
                )
            self.reserve_resource_register_range(
                used_cbuffer_registers,
                "b",
                binding,
                1,
                node.name,
                space,
            )
            buffer_registers[space] = max(buffer_registers.get(space, 0), binding + 1)
            register = self.resource_register_suffix("b", binding, node)
            if isinstance(node, StructNode):
                code += f"cbuffer {node.name}{register} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported, so we'll make it fixed size
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[1];\n"
                            )
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # Generic cbuffer handling
                code += f"cbuffer {node.name}{register} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[1];\n"
                            )
                    else:
                        # Handle both old and new AST member structures
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"
        return code

    def generate_compute_numthreads(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return f"[numthreads({x}, {y}, {z})]\n"

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
    ):
        """Render a function or stage entry point with HLSL semantics."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_access_parameters = {}
        image_format_parameters = {}
        comparison_sampler_parameters = self.comparison_sampler_parameters.get(
            getattr(func, "name", None), set()
        )
        implicit_texture_samplers = self.implicit_texture_sampler_parameters.get(
            getattr(func, "name", None), {}
        )
        implicit_existing_comparison_samplers = {
            data["sampler_name"]
            for data in implicit_texture_samplers.values()
            if data["comparison"] and not data["synthetic"]
        }
        param_names = {getattr(param, "name", None) for param in param_list}
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_glsl_buffer_block_parameters = (
            self.current_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_parameters = (
            self.current_unsupported_glsl_buffer_block_parameters
        )
        previous_unsupported_glsl_buffer_block_local_variables = (
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_glsl_buffer_block_parameter_failures = (
            self.current_glsl_buffer_block_parameter_failures
        )
        previous_glsl_buffer_block_parameter_struct_failures = (
            self.current_glsl_buffer_block_parameter_struct_failures
        )
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        self.local_variable_types = {}
        for p in param_list:
            if hasattr(p, "param_type"):
                raw_param_type = (
                    self.type_name_string(p.param_type)
                    if getattr(p.param_type, "generic_args", None)
                    else p.param_type
                )
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)

            param_type = self.map_resource_parameter_type_with_hint(
                raw_param_type, p, getattr(func, "name", None)
            )
            param_type = self.directx_resource_declaration_type(param_type)
            if self.is_texture_type(raw_param_type) or self.is_image_type(
                raw_param_type
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                if self.is_image_type(raw_param_type):
                    record_explicit_image_metadata(
                        p.name,
                        p,
                        self.attribute_value_to_string,
                        image_formats=image_format_parameters,
                        image_accesses=image_access_parameters,
                    )
            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                if (
                    p.name in comparison_sampler_parameters
                    or p.name in implicit_existing_comparison_samplers
                ):
                    param_type = param_type.replace(
                        "SamplerState", "SamplerComparisonState", 1
                    )

            semantic = self.semantic_from_node(p)

            param_type = self.apply_hlsl_parameter_qualifiers(param_type, p)
            declaration = format_c_style_array_declaration(param_type, p.name)
            semantic_attr = self.map_semantic(semantic)
            params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )
            if p.name in implicit_texture_samplers:
                sampler_info = implicit_texture_samplers[p.name]
                if sampler_info["synthetic"]:
                    sampler_type = (
                        "SamplerComparisonState"
                        if sampler_info["comparison"]
                        else "SamplerState"
                    )
                    sampler_name = sampler_info["sampler_name"]
                    params.append(f"{sampler_type} {sampler_name}")
                    sampler_parameters.add(sampler_name)
                regular_sampler_name = self.implicit_regular_sampler_name(
                    p.name, sampler_info
                )
                if (
                    sampler_info.get("regular")
                    and sampler_info.get("comparison")
                    and regular_sampler_name != sampler_info["sampler_name"]
                    and regular_sampler_name not in param_names
                ):
                    params.append(f"SamplerState {regular_sampler_name}")
                    sampler_parameters.add(regular_sampler_name)
                query_sampler_name = self.implicit_query_lod_sampler_name(
                    p.name, sampler_info
                )
                if (
                    sampler_info.get("query_lod")
                    and query_sampler_name != sampler_info["sampler_name"]
                    and query_sampler_name not in param_names
                ):
                    params.append(f"SamplerState {query_sampler_name}")
                    sampler_parameters.add(query_sampler_name)

        params_str = ", ".join(params)
        shader_map = {"vertex": "VSMain", "fragment": "PSMain", "compute": "CSMain"}
        shader_attr_map = {
            "geometry": "geometry",
            "tessellation_control": "hull",
            "tessellation_evaluation": "domain",
            "mesh": "mesh",
            "amplification": "amplification",
            "task": "amplification",
            "object": "amplification",
            "ray_generation": "raygeneration",
            "ray_intersection": "intersection",
            "ray_closest_hit": "closesthit",
            "ray_any_hit": "anyhit",
            "ray_miss": "miss",
            "ray_callable": "callable",
        }

        if hasattr(func, "return_type"):
            raw_return_type = self.type_name_string(func.return_type)
            if "[" in raw_return_type:
                raise ValueError(
                    "DirectX output does not support array return types; "
                    "wrap the array in a struct or use an output parameter"
                )
            return_type = self.map_type(raw_return_type)
        else:
            raw_return_type = "void"
            return_type = "void"
        self.current_function_return_type = raw_return_type
        parameter_diagnostics = self.glsl_buffer_block_parameter_diagnostics(
            "HLSL", param_list, indent
        )
        if parameter_diagnostics:
            code += parameter_diagnostics
            code += "  " * indent
        unsupported_function_reason = self.unsupported_glsl_buffer_block_functions.get(
            getattr(func, "name", None)
        )
        if unsupported_function_reason is not None:
            code += self.unsupported_glsl_buffer_block_function_placeholder(
                "HLSL", getattr(func, "name", None), unsupported_function_reason
            )
            self.current_function_return_type = previous_function_return_type
            self.local_variable_types = previous_local_variable_types
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.current_glsl_buffer_block_parameters = (
                previous_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_parameters = (
                previous_unsupported_glsl_buffer_block_parameters
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_glsl_buffer_block_local_variables
            )
            self.current_glsl_buffer_block_parameter_failures = (
                previous_glsl_buffer_block_parameter_failures
            )
            self.current_glsl_buffer_block_parameter_struct_failures = (
                previous_glsl_buffer_block_parameter_struct_failures
            )
            return code

        if hasattr(func, "qualifiers") and func.qualifiers:
            qualifier = func.qualifiers[0] if func.qualifiers else None
        else:
            qualifier = getattr(func, "qualifier", None)

        effective_shader_type = shader_type or qualifier

        return_semantic = self.hlsl_function_return_semantic(func)

        if effective_shader_type is not None:
            self.validate_hlsl_function_return_semantic(
                func, effective_shader_type, raw_return_type, return_semantic
            )
            self.validate_hlsl_stage_return_semantics(
                func, effective_shader_type, return_semantic
            )

        if effective_shader_type in shader_map:
            return_semantic_attr = self.map_semantic(return_semantic)
            code += f"// {effective_shader_type.capitalize()} Shader\n"
            if effective_shader_type == "compute":
                self.validate_hlsl_stage_parameter_requirements(
                    func, effective_shader_type
                )
                code += self.generate_compute_numthreads(execution_config)
            function_name = entry_name or shader_map[effective_shader_type]
            code += f"{return_type} {function_name}({params_str}){return_semantic_attr} {{\n"
        else:
            shader_attr = shader_attr_map.get(effective_shader_type)
            self.validate_hlsl_stage_requirements(func, effective_shader_type)
            code += self.generate_hlsl_stage_attributes(func, effective_shader_type)
            if shader_attr:
                code += f'[shader("{shader_attr}")]\n'
            function_name = entry_name or func.name
            return_semantic_attr = self.map_semantic(return_semantic)
            code += f"{return_type} {function_name}({params_str}){return_semantic_attr} {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_image_access_parameters = self.current_image_access_parameters
        previous_image_format_parameters = self.current_image_format_parameters
        previous_implicit_texture_samplers = self.current_implicit_texture_samplers
        previous_implicit_texture_regular_samplers = (
            self.current_implicit_texture_regular_samplers
        )
        previous_implicit_texture_query_lod_samplers = (
            self.current_implicit_texture_query_lod_samplers
        )
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = texture_parameters
        self.current_image_access_parameters = image_access_parameters
        self.current_image_format_parameters = image_format_parameters
        self.current_implicit_texture_samplers = {
            texture_name: sampler_info["sampler_name"]
            for texture_name, sampler_info in implicit_texture_samplers.items()
        }
        self.current_implicit_texture_regular_samplers = {
            texture_name: self.implicit_regular_sampler_name(texture_name, sampler_info)
            for texture_name, sampler_info in implicit_texture_samplers.items()
            if sampler_info.get("regular") and sampler_info.get("comparison")
        }
        self.current_implicit_texture_query_lod_samplers = {
            texture_name: self.implicit_query_lod_sampler_name(
                texture_name, sampler_info
            )
            for texture_name, sampler_info in implicit_texture_samplers.items()
            if sampler_info.get("query_lod")
        }
        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_image_access_parameters = previous_image_access_parameters
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_implicit_texture_samplers = previous_implicit_texture_samplers
        self.current_implicit_texture_regular_samplers = (
            previous_implicit_texture_regular_samplers
        )
        self.current_implicit_texture_query_lod_samplers = (
            previous_implicit_texture_query_lod_samplers
        )
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.current_glsl_buffer_block_parameters = (
            previous_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_parameters = (
            previous_unsupported_glsl_buffer_block_parameters
        )
        self.current_unsupported_glsl_buffer_block_local_variables = (
            previous_unsupported_glsl_buffer_block_local_variables
        )
        self.current_glsl_buffer_block_parameter_failures = (
            previous_glsl_buffer_block_parameter_failures
        )
        self.current_glsl_buffer_block_parameter_struct_failures = (
            previous_glsl_buffer_block_parameter_struct_failures
        )

        code += "  " * indent + "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as HLSL source."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            vtype = self.local_variable_declared_type(stmt)
            self.local_variable_types[stmt.name] = self.type_name_string(vtype)
            if self.is_unsupported_glsl_buffer_block_struct_type(vtype):
                self.current_unsupported_glsl_buffer_block_local_variables.add(
                    stmt.name
                )
                return (
                    f"{indent_str}"
                    f"{self.unsupported_glsl_buffer_block_local_variable_placeholder('HLSL', vtype, stmt.name)};\n"
                )

            declaration = format_c_style_array_declaration(
                self.map_type(vtype), stmt.name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    stmt.name,
                    vtype,
                    indent,
                    "HLSL",
                )
                return code
            if initial_value is not None:
                init_expr = self.generate_expression_with_expected(initial_value, vtype)
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"

        elif isinstance(stmt, ArrayNode):
            # Improved array node handling
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)

            if size is None:
                # HLSL dynamic arrays need a size, but can be accessed with buffer types
                # For basic shaders, use a fixed size as fallback
                return f"{indent_str}{element_type}[1024] {stmt.name};\n"
            else:
                return f"{indent_str}{element_type}[{size}] {stmt.name};\n"

        elif isinstance(stmt, AssignmentNode):
            return self.generate_statement_code(self.generate_assignment(stmt), indent)

        elif isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)

        elif isinstance(stmt, BreakNode):
            return f"{indent_str}break;\n"

        elif isinstance(stmt, ContinueNode):
            return f"{indent_str}continue;\n"

        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)

        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)

        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)

        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)

        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)

        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)

        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)

        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                # Handle both single values and lists
                if isinstance(stmt.value, list):
                    # Multiple return values
                    code = ""
                    for i, return_stmt in enumerate(stmt.value):
                        code += f"{self.generate_expression(return_stmt)}"
                        if i < len(stmt.value) - 1:
                            code += ", "
                    return f"{indent_str}return {code};\n"
                else:
                    # Single return value
                    return (
                        f"{indent_str}return "
                        f"{self.generate_expression_with_expected(stmt.value, self.current_function_return_type)};\n"
                    )
            else:
                # Void return
                return f"{indent_str}return;\n"

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            # Handle ExpressionStatementNode
            if hasattr(stmt, "expression"):
                tail_return = self.generate_tail_expression_statement(stmt, indent)
                if tail_return is not None:
                    return tail_return
                expression = self.generate_expression(stmt.expression)
                if isinstance(getattr(stmt, "expression", None), AssignmentNode):
                    return self.generate_statement_code(expression, indent)
                return f"{indent_str}{expression};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        else:
            # Try to generate as expression
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        indent_str = "    " * indent
        value = self.generate_expression_with_expected(stmt.expression, return_type)
        return f"{indent_str}return {value};\n"

    def local_variable_declared_type(self, stmt):
        vtype = getattr(stmt, "var_type", None)
        if vtype is None:
            vtype = getattr(stmt, "vtype", None)
        if vtype is None:
            vtype = self.expression_result_type(getattr(stmt, "initial_value", None))
        return vtype or "float"

    def local_variable_qualifier(self, node):
        return "const " if "const" in getattr(node, "qualifiers", []) else ""

    def generate_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        if not lines:
            return ""

        result = ""
        for line in lines:
            terminator = "" if line.endswith((";", "}")) else ";"
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "half",
            "min16float",
            "min10float",
            "double",
            "int",
            "min16int",
            "min12int",
            "uint",
            "min16uint",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float2",
            "float3",
            "float4",
            "half2",
            "half3",
            "half4",
            "min16float2",
            "min16float3",
            "min16float4",
            "min10float2",
            "min10float3",
            "min10float4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "min16int2",
            "min16int3",
            "min16int4",
            "min12int2",
            "min12int3",
            "min12int4",
            "uint2",
            "uint3",
            "uint4",
            "min16uint2",
            "min16uint3",
            "min16uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("min16float"):
            return "min16float"
        if mapped_type.startswith("min10float"):
            return "min10float"
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("min16uint"):
            return "min16uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("min16int"):
            return "min16int"
        if mapped_type.startswith("min12int"):
            return "min12int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            return self.local_variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        ):
            true_type = self.expression_result_type(getattr(expr, "true_expr", None))
            false_type = self.expression_result_type(getattr(expr, "false_expr", None))
            if self.is_vector_value_type(true_type):
                return true_type
            if self.is_vector_value_type(false_type):
                return false_type
            if true_type == false_type:
                return true_type
            if true_type == "float" or false_type == "float":
                return "float"
            return true_type or false_type
        if isinstance(expr, AssignmentNode):
            target = getattr(expr, "target", getattr(expr, "left", None))
            return self.expression_result_type(target)
        if isinstance(expr, ArrayAccessNode):
            block_access = self.glsl_buffer_block_array_access(expr)
            if block_access is not None:
                return block_access["type"]
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if not array_type:
                array_type = self.unsupported_glsl_buffer_block_expression_type(
                    expr.array
                )
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            block_access = self.glsl_buffer_block_member_access(expr)
            if block_access is not None:
                return block_access["type"]
            object_type = self.expression_result_type(
                expr.object
            ) or self.unsupported_glsl_buffer_block_expression_type(expr.object)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            if object_type:
                member_type = self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member)
                if member_type:
                    return member_type
            member_types = {
                self.type_name_string(members[member])
                for members in self.struct_member_types.values()
                if member in members
            }
            if len(member_types) == 1:
                return next(iter(member_types))
            return None
        if isinstance(expr, ConstructorNode):
            return infer_enum_constructor_type(
                self, expr
            ) or infer_struct_constructor_type(self, expr)
        if isinstance(expr, MatchNode):
            return infer_match_expression_result_type(self, expr)
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type:
                return numeric_result_type
            if func_name in {"normalize", "reflect"} and args:
                return self.expression_result_type(args[0])
            if func_name == "dot" and args:
                return (
                    self.vector_component_type(self.expression_result_type(args[0]))
                    or "float"
                )
            if func_name == "cross" and args:
                return self.expression_result_type(args[0])
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[specialized_func_name]
            if func_name in getattr(self, "function_return_types", {}):
                return self.function_return_types[func_name]
            unsupported_functions = getattr(
                self, "unsupported_glsl_buffer_block_functions", {}
            )
            if func_name in unsupported_functions:
                return unsupported_functions[func_name].get("return_type")
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            if func_name in {
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "int",
                "char",
                "signed char",
                "int8",
                "int8_t",
                "int16",
                "int16_t",
                "int32_t",
                "int64",
                "int64_t",
                "long",
                "signed long",
                "ptrdiff_t",
                "min16int",
                "min12int",
                "uint",
                "uchar",
                "unsigned char",
                "uint8",
                "uint8_t",
                "uint16",
                "uint16_t",
                "uint32_t",
                "uint64",
                "uint64_t",
                "ulong",
                "unsigned long",
                "size_t",
                "min16uint",
                "short",
                "signed short",
                "ushort",
                "unsigned short",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "packed_float2",
                "packed_float3",
                "packed_float4",
                "simd_float2",
                "simd_float3",
                "simd_float4",
                "int2",
                "int3",
                "int4",
                "packed_int2",
                "packed_int3",
                "packed_int4",
                "simd_int2",
                "simd_int3",
                "simd_int4",
                "uint2",
                "uint3",
                "uint4",
                "packed_uint2",
                "packed_uint3",
                "packed_uint4",
                "simd_uint2",
                "simd_uint3",
                "simd_uint4",
                "bool2",
                "bool3",
                "bool4",
                "half2",
                "half3",
                "half4",
                "packed_half2",
                "packed_half3",
                "packed_half4",
                "f16vec2",
                "f16vec3",
                "f16vec4",
                "f16mat2",
                "f16mat3",
                "f16mat4",
                "f16mat2x2",
                "f16mat2x3",
                "f16mat2x4",
                "f16mat3x2",
                "f16mat3x3",
                "f16mat3x4",
                "f16mat4x2",
                "f16mat4x3",
                "f16mat4x4",
                "simd_float2x2",
                "simd_float2x3",
                "simd_float2x4",
                "simd_float3x2",
                "simd_float3x3",
                "simd_float3x4",
                "simd_float4x2",
                "simd_float4x3",
                "simd_float4x4",
                "char2",
                "char3",
                "char4",
                "uchar2",
                "uchar3",
                "uchar4",
                "i8vec2",
                "i8vec3",
                "i8vec4",
                "u8vec2",
                "u8vec3",
                "u8vec4",
                "short2",
                "short3",
                "short4",
                "ushort2",
                "ushort3",
                "ushort4",
                "i16vec2",
                "i16vec3",
                "i16vec4",
                "u16vec2",
                "u16vec3",
                "u16vec4",
                "min16float2",
                "min16float3",
                "min16float4",
                "min10float2",
                "min10float3",
                "min10float4",
                "min16int2",
                "min16int3",
                "min16int4",
                "min12int2",
                "min12int3",
                "min12int4",
                "min16uint2",
                "min16uint3",
                "min16uint4",
            }:
                return str(func_name)
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return self.local_variable_types.get(getattr(expr, "name", None))
        return None

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            target = node.target
            value = node.value
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            target = node.left
            value = node.right
            op = getattr(node, "operator", "=")

        block_store = self.generate_glsl_buffer_block_store(target, value, op)
        if block_store is not None:
            return block_store
        unsupported_store = self.unsupported_glsl_buffer_block_assignment_diagnostic(
            target
        )
        if unsupported_store is not None:
            return unsupported_store

        lhs = self.generate_expression(target)
        rhs = self.generate_expression_with_expected(
            value, self.expression_result_type(target)
        )
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent

        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        then_body = getattr(node, "then_branch", getattr(node, "if_body", []))
        else_body = getattr(node, "else_branch", getattr(node, "else_body", []))

        code = f"{indent_str}if ({self.generate_expression(condition)}) {{\n"

        code += self.generate_scoped_statement_body(then_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
                code += self.generate_scoped_statement_body(else_if_body, indent + 1)
                code += f"{indent_str}}}"

        if else_body:
            code += " else {\n"
            code += self.generate_scoped_statement_body(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            # Handle for loop components
            init = ""
            condition = ""
            update = ""

            if hasattr(node, "init") and node.init:
                if isinstance(node.init, str):
                    init = node.init
                else:
                    init = self.generate_for_initializer(node.init)

            if hasattr(node, "condition") and node.condition:
                if isinstance(node.condition, str):
                    condition = node.condition
                else:
                    condition = (
                        self.generate_expression(node.condition).strip().rstrip(";")
                    )

            if hasattr(node, "update") and node.update:
                if isinstance(node.update, str):
                    update = node.update
                else:
                    update = self.generate_expression(node.update).strip().rstrip(";")

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

            body = getattr(node, "body", [])
            code += self.generate_scoped_statement_body(body, indent + 1)

            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_block(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}{{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "statements", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_scoped_statement_body(self, body, indent):
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )

        try:
            self.local_variable_types[pattern] = "int"
            self.current_unsupported_glsl_buffer_block_local_variables.discard(pattern)

            if isinstance(iterable_node, RangeNode):
                start = self.generate_expression(iterable_node.start)
                end = self.generate_expression(iterable_node.end)
                comparator = "<=" if iterable_node.inclusive else "<"
                code = (
                    f"{indent_str}for (int {pattern} = {start}; "
                    f"{pattern} {comparator} {end}; ++{pattern}) {{\n"
                )
            else:
                iterable = self.generate_expression(iterable_node)
                code = (
                    f"{indent_str}for (int {pattern} = 0; {pattern} < {iterable}; "
                    f"++{pattern}) {{\n"
                )

            code += self.generate_scoped_statement_body(
                getattr(node, "body", []), indent + 1
            )
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(getattr(node, "condition", ""))

        code = f"{indent_str}do {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}} while ({condition});\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for case in getattr(node, "cases", []) or []:
            value = getattr(case, "value", None)
            if value is None:
                label = "default"
            else:
                label = f"case {self.generate_expression(value)}"
            code += self.generate_switch_case(
                label, getattr(case, "statements", []), indent + 1
            )

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            code += self.generate_switch_case(
                "default",
                getattr(default_case, "statements", default_case),
                indent + 1,
            )

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        if is_switch_lowerable_match(node):
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "HLSL")

    def statement_body_terminates(self, body):
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]

        return bool(statements) and isinstance(
            statements[-1], (BreakNode, ContinueNode, ReturnNode)
        )

    def statement_body_has_statements(self, body):
        if hasattr(body, "statements"):
            return bool(body.statements)
        if isinstance(body, list):
            return bool(body)
        return body is not None

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_statement_body(self, body, indent):
        code = ""
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent)
        elif body is not None:
            code += self.generate_statement(body, indent)
        return code

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, VariableNode) or (
            hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__)
        ):
            return self.generate_statement(init, 0).strip().rstrip(";")
        return self.generate_expression(init).strip().rstrip(";")

    def generate_expression(self, expr):
        """Render a CrossGL AST expression into HLSL expression syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, (int, float)):
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                value = expr.value
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                if (
                    literal_type == "uint"
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    return f"{value}u"
                if isinstance(value, bool):
                    return "true" if value else "false"
                if isinstance(value, str) and not (
                    value.startswith('"') and value.endswith('"')
                ):
                    return f'"{value}"'
                return str(value)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            name = getattr(expr, "name", str(expr))
            return enum_value_expression(self, name)
        elif isinstance(expr, VariableNode):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            return enum_value_expression(self, expr.name)
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            left = self.generate_expression(getattr(expr, "left", ""))
            right = self.generate_expression(getattr(expr, "right", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"({left} {self.map_operator(op)} {right})"
        elif isinstance(expr, AssignmentNode):
            # Handle assignment as expression
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand = self.generate_expression(getattr(expr, "operand", ""))
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            return f"{self.map_operator(op)}{operand}"
        elif isinstance(expr, WaveOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, RayTracingOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, MeshOpNode):
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{expr.operation}({args_str})"
        elif isinstance(expr, RayQueryOpNode):
            query = self.generate_expression(expr.query_expr)
            args_str = ", ".join(
                self.generate_expression(arg) for arg in expr.arguments
            )
            return f"{query}.{expr.operation}({args_str})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_array_load(expr)
            if block_load is not None:
                return block_load
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]"
        elif isinstance(expr, ConstructorNode):
            enum_constructor = generate_enum_constructor_expression(self, expr)
            if enum_constructor is not None:
                return enum_constructor
            constructor = generate_struct_constructor_expression(self, expr)
            if constructor is not None:
                return constructor
            return str(expr)
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            numeric_trait_call = generate_numeric_trait_method_call(
                self,
                func_expr,
                args,
            )
            if numeric_trait_call is not None:
                return numeric_trait_call

            func_name = None
            if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)

            static_generic_call = generate_static_generic_numeric_call(self, func_name)
            if static_generic_call is not None:
                return static_generic_call

            enum_constructor = generate_enum_constructor_call(self, func_name, args)
            if enum_constructor is not None:
                return enum_constructor

            unsupported_call = self.unsupported_glsl_buffer_block_function_call(
                func_name
            )
            if unsupported_call is not None:
                return unsupported_call

            texture_call = self.generate_texture_call(func_name, args)
            if texture_call is not None:
                return texture_call

            buffer_call = self.generate_buffer_call(func_name, args)
            if buffer_call is not None:
                return buffer_call

            call_argument_func_name = func_name
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name is not None:
                callee = specialized_func_name
                func_name = specialized_func_name

            if func_name in [
                "float",
                "half",
                "float16",
                "min16float",
                "min10float",
                "double",
                "int",
                "char",
                "signed char",
                "int8",
                "int8_t",
                "int16",
                "int16_t",
                "int32_t",
                "int64",
                "int64_t",
                "long",
                "signed long",
                "ptrdiff_t",
                "min16int",
                "min12int",
                "uint",
                "uchar",
                "unsigned char",
                "uint8",
                "uint8_t",
                "uint16",
                "uint16_t",
                "uint32_t",
                "uint64",
                "uint64_t",
                "ulong",
                "unsigned long",
                "size_t",
                "min16uint",
                "short",
                "signed short",
                "ushort",
                "unsigned short",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "packed_float2",
                "packed_float3",
                "packed_float4",
                "simd_float2",
                "simd_float3",
                "simd_float4",
                "half2",
                "half3",
                "half4",
                "packed_half2",
                "packed_half3",
                "packed_half4",
                "f16vec2",
                "f16vec3",
                "f16vec4",
                "f16mat2",
                "f16mat3",
                "f16mat4",
                "f16mat2x2",
                "f16mat2x3",
                "f16mat2x4",
                "f16mat3x2",
                "f16mat3x3",
                "f16mat3x4",
                "f16mat4x2",
                "f16mat4x3",
                "f16mat4x4",
                "char2",
                "char3",
                "char4",
                "uchar2",
                "uchar3",
                "uchar4",
                "packed_int2",
                "packed_int3",
                "packed_int4",
                "simd_int2",
                "simd_int3",
                "simd_int4",
                "packed_uint2",
                "packed_uint3",
                "packed_uint4",
                "simd_uint2",
                "simd_uint3",
                "simd_uint4",
                "simd_float2x2",
                "simd_float2x3",
                "simd_float2x4",
                "simd_float3x2",
                "simd_float3x3",
                "simd_float3x4",
                "simd_float4x2",
                "simd_float4x3",
                "simd_float4x4",
                "i8vec2",
                "i8vec3",
                "i8vec4",
                "u8vec2",
                "u8vec3",
                "u8vec4",
                "short2",
                "short3",
                "short4",
                "ushort2",
                "ushort3",
                "ushort4",
                "i16vec2",
                "i16vec3",
                "i16vec4",
                "u16vec2",
                "u16vec3",
                "u16vec4",
                "min16float2",
                "min16float3",
                "min16float4",
                "min10float2",
                "min10float3",
                "min10float4",
                "min16int2",
                "min16int3",
                "min16int4",
                "min12int2",
                "min12int3",
                "min12int4",
                "min16uint2",
                "min16uint3",
                "min16uint4",
            ]:
                mapped_type = self.map_type(func_name)
                args_str = ", ".join(
                    self.generate_expression_with_expected(arg, None) for arg in args
                )
                return f"{mapped_type}({args_str})"
            self.validate_function_image_access_arguments(func_name, args)
            args_str = ", ".join(
                self.generate_call_arguments(call_argument_func_name, args)
            )
            return f"{callee}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
            if unsupported_value is not None:
                return unsupported_value
            block_load = self.generate_glsl_buffer_block_member_load(expr)
            if block_load is not None:
                return block_load
            obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
            member = getattr(expr, "member", "")
            obj = self.generate_expression_with_expected(obj_expr, None)
            return f"{obj}.{member}"
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            condition = self.generate_expression(getattr(expr, "condition", ""))
            true_expr = self.generate_expression(getattr(expr, "true_expr", ""))
            false_expr = self.generate_expression(getattr(expr, "false_expr", ""))
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def generate_buffer_call(self, func_name, args):
        """Render canonical CrossGL buffer operations as HLSL resource methods."""
        if func_name == "buffer_load" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}.Load({index})"
        if func_name == "buffer_store" and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}.Store({index}, {value})"
        if func_name == "buffer_append" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            value = self.generate_expression(args[1])
            return f"{buffer}.Append({value})"
        if func_name == "buffer_consume" and args:
            buffer = self.generate_expression(args[0])
            return f"{buffer}.Consume()"
        if func_name == "buffer_dimensions" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            dimensions = ", ".join(self.generate_expression(arg) for arg in args[1:])
            return f"{buffer}.GetDimensions({dimensions})"
        return None

    def collect_comparison_resources(self, root):
        texture_names = set()
        sampler_names = set()
        visited = set()
        global_resource_types = self.collect_global_texture_types(root)
        comparison_funcs = self.comparison_texture_function_names()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            if isinstance(value, FunctionCallNode):
                func_expr = getattr(value, "function", getattr(value, "name", None))
                func_name = self.expression_name(func_expr)
                args = getattr(value, "arguments", getattr(value, "args", []))
                if func_name in comparison_funcs and len(args) >= 3:
                    texture_name = self.expression_name(args[0])
                    texture_type = global_resource_types.get(texture_name)
                    if self.texture_call_is_diagnostic_only(func_name, texture_type):
                        return
                    if texture_name:
                        texture_names.add(texture_name)
                    if len(args) >= 4:
                        sampler_name = self.expression_name(args[1])
                        if sampler_name:
                            sampler_names.add(sampler_name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return texture_names, sampler_names

    def comparison_texture_function_names(self):
        return TEXTURE_COMPARE_INTRINSIC_NAMES | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES

    def regular_texture_function_names(self):
        return (
            TEXTURE_SAMPLING_INTRINSIC_NAMES
            | TEXTURE_GATHER_INTRINSIC_NAMES
            | TEXTURE_QUERY_LOD_INTRINSIC_NAMES
        )

    def collect_regular_sampler_parameters(self, root):
        regular_params = {}
        functions = self.collect_functions(root)
        global_resource_types = self.collect_global_texture_types(root)

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            sampler_params = {
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not sampler_params:
                continue

            previous_local_variable_types = self.local_variable_types
            self.local_variable_types = self.function_scope_variable_types(func)
            try:
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name_expr = self.expression_name(func_expr)
                    if func_name_expr not in self.regular_texture_function_names():
                        continue
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if not self.has_explicit_sampler_argument(
                        func_name_expr, args, sampler_params
                    ):
                        continue
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                    if self.texture_call_is_diagnostic_only(
                        func_name_expr, texture_type
                    ):
                        continue
                    sampler_name = self.expression_name(args[1])
                    if sampler_name in sampler_params:
                        regular_params.setdefault(func_name, set()).add(sampler_name)
            finally:
                self.local_variable_types = previous_local_variable_types

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                sampler_param_names = {
                    param.name
                    for param in getattr(
                        func, "parameters", getattr(func, "params", [])
                    )
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not sampler_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, regular_params, callee_name
                    )
                    if not required_indices:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name in sampler_param_names:
                            current = regular_params.setdefault(func_name, set())
                            if arg_name not in current:
                                current.add(arg_name)
                                changed = True

        return regular_params

    def collect_regular_sampler_arguments(self, root, regular_params):
        return self.collect_sampler_arguments(root, regular_params)

    def collect_sampler_arguments(self, root, sampler_params_by_function):
        sampler_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                required_indices = self.comparison_sampler_parameter_indices(
                    functions, sampler_params_by_function, callee_name
                )
                if not required_indices:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                for index in required_indices:
                    if index >= len(args):
                        continue
                    arg_name = self.expression_name(args[index])
                    if arg_name:
                        sampler_names.add(arg_name)

        return sampler_names

    def collect_explicit_sampler_resource_names(self, root, texture_funcs):
        sampler_names = set()
        global_sampler_names = self.collect_global_sampler_names(root)
        global_resource_types = self.collect_global_texture_types(root)
        previous_local_variable_types = self.local_variable_types
        try:
            for func in self.collect_functions(root):
                self.local_variable_types = self.function_scope_variable_types(func)
                sampler_scope = global_sampler_names | {
                    param.name
                    for param in getattr(
                        func, "parameters", getattr(func, "params", [])
                    )
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name = self.expression_name(func_expr)
                    if func_name not in texture_funcs:
                        continue
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if not self.has_explicit_sampler_argument(
                        func_name, args, sampler_scope
                    ):
                        continue
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                    if self.texture_call_is_diagnostic_only(func_name, texture_type):
                        continue
                    sampler_name = self.expression_name(args[1])
                    if sampler_name:
                        sampler_names.add(sampler_name)
        finally:
            self.local_variable_types = previous_local_variable_types
        return sampler_names

    def collect_sampler_struct_members(self, root, sampler_params, texture_funcs):
        members = set()
        previous_local_variable_types = self.local_variable_types
        functions = self.collect_functions(root)
        global_resource_types = self.collect_global_texture_types(root)
        try:
            for func in functions:
                self.local_variable_types = self.function_scope_variable_types(func)
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name = self.expression_name(func_expr)
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if (
                        func_name in texture_funcs
                        and self.has_explicit_sampler_argument(func_name, args, set())
                    ):
                        texture_type = self.texture_argument_analysis_type(
                            args[0], global_resource_types
                        )
                        if self.texture_call_is_diagnostic_only(
                            func_name, texture_type
                        ):
                            continue
                        member_ref = self.sampler_struct_member_reference(args[1])
                        if member_ref is not None:
                            members.add(member_ref)

                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, sampler_params, func_name
                    )
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        member_ref = self.sampler_struct_member_reference(args[index])
                        if member_ref is not None:
                            members.add(member_ref)
        finally:
            self.local_variable_types = previous_local_variable_types
        return members

    def function_scope_variable_types(self, func):
        variable_types = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            variable_types[param.name] = self.type_name_string(param_type)

        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            variable_types[name] = self.type_name_string(vtype)

        return variable_types

    def sampler_struct_member_reference(self, expr):
        while isinstance(expr, ArrayAccessNode):
            expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        if not isinstance(expr, MemberAccessNode):
            return None

        member_name = str(expr.member)
        object_type = self.type_name_string(self.expression_result_type(expr.object))
        if object_type:
            member_type = self.struct_member_types.get(object_type, {}).get(member_name)
            if member_type is not None and self.is_sampler_type(member_type):
                return object_type, member_name

        matching_structs = [
            struct_name
            for struct_name, members in self.struct_member_types.items()
            if member_name in members and self.is_sampler_type(members[member_name])
        ]
        if len(matching_structs) == 1:
            return matching_structs[0], member_name
        return None

    def collect_global_texture_names(self, root):
        texture_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            mapped_type = self.map_resource_type_with_format(var_type, node)
            if var_name and mapped_type.startswith("Texture"):
                texture_names.add(var_name)
        return texture_names

    def collect_global_texture_types(self, root):
        texture_types = {}
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            mapped_type = self.map_resource_type_with_format(var_type, node)
            if var_name and mapped_type.startswith(("Texture", "RWTexture")):
                texture_types[var_name] = mapped_type
        return texture_types

    def collect_global_sampler_names(self, root):
        sampler_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_sampler_type(var_type):
                sampler_names.add(var_name)
        return sampler_names

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_resource_parameter_type(var_type):
                resource_names.add(var_name)
        return resource_names

    def global_resource_declaration_nodes(self, root, target_stage=None):
        current_nodes = getattr(self, "current_global_resource_declaration_nodes", None)
        if current_nodes is not None and target_stage is None:
            return current_nodes

        global_vars = list(getattr(root, "global_variables", []) or [])
        stage_resource_vars = collect_stage_local_variables(
            root, target_stage, self.is_stage_local_resource_variable
        )
        return deduplicate_named_declarations(
            global_vars + stage_resource_vars, "DirectX resource"
        )

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        return (
            self.is_resource_parameter_type(vtype)
            or self.is_hlsl_readonly_buffer_type(vtype)
            or self.is_hlsl_uav_buffer_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_resource_parameter_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Non-resource local declaration(s) shadow DirectX global resource(s): "
                f"{names}"
            )

    def validate_explicit_sampler_role_conflicts(self, ast):
        global_sampler_names = self.collect_global_sampler_names(ast)
        comparison_sampler_names = self.collect_explicit_sampler_resource_names(
            ast, self.comparison_texture_function_names()
        ) | self.collect_comparison_sampler_arguments(
            ast, self.comparison_sampler_parameters
        )
        regular_sampler_names = self.collect_explicit_sampler_resource_names(
            ast, self.regular_texture_function_names()
        ) | self.collect_regular_sampler_arguments(ast, self.regular_sampler_parameters)

        conflicts = [
            name
            for name in sorted(
                (comparison_sampler_names & regular_sampler_names)
                & global_sampler_names
            )
        ]

        function_names = set(self.comparison_sampler_parameters) | set(
            self.regular_sampler_parameters
        )
        for func_name in sorted(function_names):
            for param_name in sorted(
                self.comparison_sampler_parameters.get(func_name, set())
                & self.regular_sampler_parameters.get(func_name, set())
            ):
                conflicts.append(f"{func_name}.{param_name}")

        for struct_name, member_name in sorted(
            self.comparison_sampler_struct_members & self.regular_sampler_struct_members
        ):
            conflicts.append(f"{struct_name}.{member_name}")

        if not conflicts:
            return

        names = ", ".join(conflicts)
        raise ValueError(
            "DirectX sampler(s) used for both regular sampling and shadow "
            f"comparison: {names}"
        )

    def is_projected_texture_function(self, func_name):
        return is_projected_texture_operation(func_name)

    def is_projected_texture_compare_function(self, func_name):
        return is_projected_texture_compare_operation(func_name)

    def projected_texture_call_is_diagnostic_only(self, func_name, texture_type):
        if not self.is_projected_texture_function(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        if texture_type == "TextureCubeArray":
            return True
        return texture_type == "TextureCube" and (
            is_projected_texture_basic_offset_operation(func_name)
            or is_projected_texture_lod_offset_operation(func_name)
            or is_projected_texture_grad_offset_operation(func_name)
        )

    def projected_texture_compare_call_is_diagnostic_only(
        self, func_name, texture_type
    ):
        if not self.is_projected_texture_compare_function(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        if texture_type == "TextureCubeArray":
            return True
        return texture_type == "TextureCube" and (
            is_texture_compare_offset_operation(func_name)
            or is_texture_compare_lod_offset_operation(func_name)
            or is_texture_compare_grad_offset_operation(func_name)
        )

    def diagnostic_texture_compare_sampler_parameter_is_comparison(
        self, func_name, texture_type
    ):
        if not self.is_projected_texture_compare_function(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return texture_type in {"TextureCube", "TextureCubeArray"}

    def texture_gather_compare_offset_call_is_diagnostic_only(
        self, func_name, texture_type
    ):
        if (
            not is_texture_gather_compare_offset_operation(func_name)
            or texture_type is None
        ):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_compare_offset_supported(texture_type)

    def texture_gather_call_is_diagnostic_only(self, func_name, texture_type):
        if func_name != "textureGather" or texture_type is None:
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_supported(texture_type)

    def texture_gather_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_gather_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_offset_supported(texture_type)

    def texture_sample_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_sample_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_sample_offset_supported(texture_type)

    def texture_compare_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_compare_non_projected_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_compare_offset_supported(texture_type)

    def is_multisample_texture_resource_type(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}

    def is_multisample_storage_image_resource_type(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return texture_type.startswith(("RWTexture2DMS<", "RWTexture2DMSArray<"))

    def directx_multisample_storage_image_srv_type(self, texture_type):
        """Return a legal HLSL SRV type for unsupported multisample images."""
        if texture_type is None:
            return None
        texture_text = str(texture_type)
        if "[" in texture_text and "]" in texture_text:
            base_type, array_suffix = split_array_type_suffix(texture_text)
        else:
            base_type, array_suffix = texture_text, ""
        base_type = self.resource_base_type(base_type)
        if "<" not in base_type or ">" not in base_type:
            return None
        component_type = base_type.split("<", 1)[1].split(">", 1)[0].strip()
        if base_type.startswith("RWTexture2DMSArray<"):
            return f"Texture2DMSArray<{component_type}>{array_suffix}"
        if base_type.startswith("RWTexture2DMS<"):
            return f"Texture2DMS<{component_type}>{array_suffix}"
        return None

    def directx_resource_declaration_type(self, texture_type):
        return (
            self.directx_multisample_storage_image_srv_type(texture_type)
            or texture_type
        )

    def multisample_texture_call_is_diagnostic_only(self, func_name, texture_type):
        if func_name not in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES:
            return False
        return self.is_multisample_texture_resource_type(texture_type)

    def texture_call_is_diagnostic_only(self, func_name, texture_type):
        if texture_type is None:
            return False
        if self.storage_image_texture_operation_expression(func_name, texture_type):
            return True
        if is_texture_query_lod_operation(
            func_name
        ) and self.is_storage_image_resource_type(texture_type):
            return True
        if self.projected_texture_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.projected_texture_compare_call_is_diagnostic_only(
            func_name, texture_type
        ):
            return True
        if self.texture_gather_compare_offset_call_is_diagnostic_only(
            func_name, texture_type
        ):
            return True
        if self.texture_gather_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_gather_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_sample_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_compare_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        return self.multisample_texture_call_is_diagnostic_only(func_name, texture_type)

    def texture_argument_analysis_type(self, texture_arg, global_resource_types=None):
        texture_name = self.expression_name(texture_arg)
        if texture_name and global_resource_types:
            texture_type = global_resource_types.get(texture_name)
            if texture_type is not None:
                return texture_type

        arg_type = self.expression_result_type(texture_arg)
        if arg_type is not None and self.is_texture_or_image_type(arg_type):
            return self.map_resource_type_with_format(self.resource_base_type(arg_type))
        return None

    def collect_global_implicit_sampler_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)
        texture_funcs = TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.projected_texture_compare_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_compare_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_compare_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param in callee_implicit:
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_global_implicit_regular_sampler_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)
        texture_funcs = (
            self.regular_texture_function_names() - TEXTURE_QUERY_LOD_INTRINSIC_NAMES
        )

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info.get("regular"):
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_global_implicit_query_lod_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if not is_texture_query_lod_operation(func_name):
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info.get("query_lod"):
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_sampler_parameter_names(self, root):
        sampler_names = set()
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            for param in getattr(value, "parameters", getattr(value, "params", [])):
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_sampler_type(param_type):
                    sampler_names.add(param.name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_names

    def collect_comparison_sampler_parameters(self, root):
        comparison_params = {}
        functions = self.collect_functions(root)
        global_resource_types = self.collect_global_texture_types(root)

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            resource_param_types = {
                param.name: getattr(param, "param_type", getattr(param, "vtype", None))
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_texture_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                or self.is_image_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            sampler_params = {
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not sampler_params:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                if self.expression_name(func_expr) not in {
                    *TEXTURE_COMPARE_INTRINSIC_NAMES,
                    *TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
                }:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 4:
                    continue
                texture_name = self.expression_name(args[0])
                texture_type = resource_param_types.get(texture_name)
                if texture_type is not None:
                    texture_type = self.map_resource_type_with_format(texture_type)
                else:
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                texture_func = self.expression_name(func_expr)
                if self.texture_call_is_diagnostic_only(
                    texture_func, texture_type
                ) and not self.diagnostic_texture_compare_sampler_parameter_is_comparison(
                    texture_func, texture_type
                ):
                    continue
                sampler_name = self.expression_name(args[1])
                if sampler_name in sampler_params:
                    comparison_params.setdefault(func_name, set()).add(sampler_name)

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                sampler_param_names = {
                    param.name
                    for param in params
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not sampler_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, comparison_params, callee_name
                    )
                    if not required_indices:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name in sampler_param_names:
                            current = comparison_params.setdefault(func_name, set())
                            if arg_name not in current:
                                current.add(arg_name)
                                changed = True

        return comparison_params

    def collect_comparison_sampler_arguments(self, root, comparison_params):
        comparison_sampler_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                required_indices = self.comparison_sampler_parameter_indices(
                    functions, comparison_params, callee_name
                )
                if not required_indices:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                for index in required_indices:
                    if index >= len(args):
                        continue
                    arg_name = self.expression_name(args[index])
                    if arg_name:
                        comparison_sampler_names.add(arg_name)

        return comparison_sampler_names

    def comparison_sampler_parameter_indices(
        self, functions, comparison_params, function_name
    ):
        if not function_name or function_name not in comparison_params:
            return set()

        for func in functions:
            if getattr(func, "name", None) != function_name:
                continue
            indices = set()
            params = getattr(func, "parameters", getattr(func, "params", []))
            for index, param in enumerate(params):
                if param.name in comparison_params[function_name]:
                    indices.add(index)
            return indices

        return set()

    def collect_implicit_texture_sampler_parameters(self, root):
        implicit_params = {}
        functions = self.collect_functions(root)
        texture_funcs = TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue

            params = getattr(func, "parameters", getattr(func, "params", []))
            param_names = {param.name for param in params}
            texture_param_types = {
                param.name: getattr(param, "param_type", getattr(param, "vtype", None))
                for param in params
                if self.is_texture_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            sampler_params = {
                param.name
                for param in params
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not texture_param_types:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                texture_func = self.expression_name(func_expr)
                if texture_func not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue

                texture_name = self.expression_name(args[0])
                if texture_name not in texture_param_types:
                    continue
                if self.has_explicit_sampler_argument(
                    texture_func, args, sampler_params
                ):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.projected_texture_compare_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_compare_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_compare_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue

                query_lod = is_texture_query_lod_operation(texture_func)
                regular = (
                    not query_lod
                    and texture_func in self.regular_texture_function_names()
                )
                comparison = (
                    not query_lod
                    and texture_func in self.comparison_texture_function_names()
                )
                self.add_implicit_texture_sampler_parameter(
                    implicit_params,
                    func_name,
                    texture_name,
                    comparison,
                    param_names,
                    query_lod=query_lod,
                    regular=regular,
                )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                param_names = {param.name for param in params}
                texture_param_names = {
                    param.name
                    for param in params
                    if self.is_texture_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not texture_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    callee_implicit = implicit_params.get(callee_name, {})
                    if not callee_implicit:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee_params = self.function_parameter_names.get(callee_name, [])
                    for texture_param, sampler_info in callee_implicit.items():
                        try:
                            texture_index = callee_params.index(texture_param)
                        except ValueError:
                            continue
                        if texture_index >= len(args):
                            continue
                        arg_name = self.expression_name(args[texture_index])
                        if arg_name not in texture_param_names:
                            continue
                        changed |= self.add_implicit_texture_sampler_parameter(
                            implicit_params,
                            func_name,
                            arg_name,
                            sampler_info["comparison"],
                            param_names,
                            query_lod=sampler_info.get("query_lod", False),
                            regular=sampler_info.get("regular", False),
                        )

        return implicit_params

    def add_implicit_texture_sampler_parameter(
        self,
        implicit_params,
        func_name,
        texture_name,
        comparison,
        param_names,
        query_lod=False,
        regular=False,
    ):
        sampler_name = f"{texture_name}Sampler"
        new_info = {
            "sampler_name": sampler_name,
            "comparison": comparison,
            "synthetic": sampler_name not in param_names,
            "query_lod": query_lod,
            "regular": regular,
        }
        current = implicit_params.setdefault(func_name, {}).get(texture_name)
        if current is None:
            implicit_params[func_name][texture_name] = new_info
            return True
        changed = False
        if comparison and not current["comparison"]:
            current["comparison"] = True
            changed = True
        if query_lod and not current.get("query_lod"):
            current["query_lod"] = True
            changed = True
        if regular and not current.get("regular"):
            current["regular"] = True
            changed = True
        return changed

    def has_explicit_sampler_argument(self, func_name, args, sampler_names):
        if func_name in {
            *TEXTURE_COMPARE_INTRINSIC_NAMES,
            *TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
        }:
            if len(args) < 4:
                return False
        elif len(args) < 3:
            return False

        sampler_name = self.expression_name(args[1])
        if sampler_name in sampler_names:
            return True
        sampler_type = self.expression_result_type(args[1])
        return sampler_type is not None and self.is_sampler_type(sampler_type)

    def collect_implicit_comparison_texture_arguments(self, root, implicit_params):
        texture_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info["comparison"]:
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name:
                        texture_names.add(texture_name)

        return texture_names

    def stage_entry_types(self):
        return {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }

    def stage_entry_base_name(self, stage_name, func):
        shader_map = {
            "vertex": "VSMain",
            "fragment": "PSMain",
            "compute": "CSMain",
            "geometry": "GSMain",
            "tessellation_control": "HSMain",
            "tessellation_evaluation": "DSMain",
            "mesh": "MSMain",
            "amplification": "ASMain",
            "task": "ASMain",
            "object": "ASMain",
            "ray_generation": "RayGenMain",
            "ray_intersection": "IntersectionMain",
            "ray_closest_hit": "ClosestHitMain",
            "ray_any_hit": "AnyHitMain",
            "ray_miss": "MissMain",
            "ray_callable": "CallableMain",
            "intersection": "IntersectionMain",
            "closesthit": "ClosestHitMain",
            "anyhit": "AnyHitMain",
            "miss": "MissMain",
            "callable": "CallableMain",
        }
        return shader_map.get(stage_name, getattr(func, "name", None) or "main")

    def hlsl_stage_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "domain",
            "maxvertexcount",
            "maxtessfactor",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        if normalized in valid_names:
            return normalized
        return None

    def hlsl_stage_attribute_names(self, func):
        names = set()
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if attr_name and getattr(attr, "arguments", []):
                names.add(attr_name)
        return names

    def hlsl_stage_attribute_argument(self, func, expected_name):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if attr_name != expected_name:
                continue

            arguments = getattr(attr, "arguments", []) or []
            if arguments:
                return self.hlsl_stage_attribute_value_to_string(arguments[0])
        return None

    def normalized_hlsl_stage_attribute_argument(self, func, expected_name):
        value = self.hlsl_stage_attribute_argument(func, expected_name)
        if value is None:
            return None
        return str(value).strip('"').lower()

    def canonical_hlsl_tessellation_domain(self, domain):
        if domain is None:
            return None
        normalized = str(domain).strip('"').lower()
        if normalized == "triangle":
            return "tri"
        return normalized

    def hlsl_int_literal_value(self, value):
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        text = str(value).strip().strip('"').replace("_", "")
        if not text:
            return None
        while text and text[-1] in {"u", "U", "l", "L"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def hlsl_stage_attribute_int_argument(self, func, expected_name):
        return self.hlsl_int_literal_value(
            self.hlsl_stage_attribute_argument(func, expected_name)
        )

    def hlsl_parameter_type_base(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        return type_name.split("<", 1)[0].strip()

    def hlsl_parameter_array_count(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if str(type(param_type)).find("ArrayType") != -1:
            return self.hlsl_int_literal_value(getattr(param_type, "size", None))

        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        _base_type, array_suffix = split_array_type_suffix(str(type_name))
        if not array_suffix:
            return None
        first_dimension = array_suffix[1:].split("]", 1)[0]
        return self.hlsl_int_literal_value(first_dimension)

    def hlsl_parameter_mapped_base_and_array_suffix(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None, None

        mapped_type = self.map_type(type_name)
        return split_array_type_suffix(str(mapped_type))

    def hlsl_parameter_is_array(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if str(type(param_type)).find("ArrayType") != -1:
            return True

        type_name = self.type_name_string(param_type)
        return bool(type_name and split_array_type_suffix(str(type_name))[1])

    def hlsl_parameter_component_count(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None

        base_type, _array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None
        scalar_bases = (
            "min16float",
            "min10float",
            "min16int",
            "min12int",
            "min16uint",
            "float",
            "half",
            "double",
            "int",
            "uint",
            "bool",
        )
        for scalar_base in scalar_bases:
            if not base_type.startswith(scalar_base):
                continue
            suffix = base_type[len(scalar_base) :]
            if suffix in {"2", "3", "4"}:
                return int(suffix)
            if suffix == "":
                return 1
        return None

    def split_hlsl_generic_argument_string(self, arguments):
        split_arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(arguments):
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                split_arguments.append(arguments[start:index].strip())
                start = index + 1
        split_arguments.append(arguments[start:].strip())
        return [argument for argument in split_arguments if argument]

    def hlsl_type_generic_arguments(self, vtype):
        generic_args = getattr(vtype, "generic_args", None)
        if generic_args:
            return generic_args

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return []
        arguments = type_name.split("<", 1)[1][:-1].strip()
        return self.split_hlsl_generic_argument_string(arguments)

    def hlsl_parameter_type_generic_argument(self, parameter, index):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        generic_args = self.hlsl_type_generic_arguments(param_type)
        if len(generic_args) <= index:
            return None
        return self.type_name_string(generic_args[index])

    def hlsl_patch_control_point_count(self, parameters, patch_type):
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) != patch_type:
                continue
            param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            generic_args = self.hlsl_type_generic_arguments(param_type)
            if len(generic_args) < 2:
                return None
            return self.hlsl_int_literal_value(generic_args[1])
        return None

    def hlsl_input_patch_control_point_count(self, parameters):
        return self.hlsl_patch_control_point_count(parameters, "InputPatch")

    def hlsl_output_patch_control_point_count(self, parameters):
        return self.hlsl_patch_control_point_count(parameters, "OutputPatch")

    def validate_hlsl_output_control_points(self, func, parameters):
        output_control_points = self.hlsl_stage_attribute_int_argument(
            func, "outputcontrolpoints"
        )
        input_patch_control_points = self.hlsl_input_patch_control_point_count(
            parameters
        )
        if output_control_points is None or input_patch_control_points is None:
            return
        if output_control_points != input_patch_control_points:
            raise ValueError(
                "DirectX tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must match the InputPatch control "
                f"point count ({input_patch_control_points})"
            )

    def validate_hlsl_output_control_points_value(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        output_control_points = self.hlsl_stage_attribute_int_argument(
            func, "outputcontrolpoints"
        )
        if output_control_points is None:
            return
        if output_control_points <= 0:
            raise ValueError(
                "DirectX tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must be positive"
            )

    def validate_hlsl_max_vertex_count_value(self, func, shader_type):
        if shader_type != "geometry":
            return

        max_vertex_count = self.hlsl_stage_attribute_int_argument(
            func, "maxvertexcount"
        )
        if max_vertex_count is None:
            return
        if max_vertex_count <= 0:
            raise ValueError(
                "DirectX geometry stage maxvertexcount "
                f"({max_vertex_count}) must be positive"
            )

    def validate_hlsl_domain_output_patch_control_points(self, parameters):
        hull_output_control_points = self.current_hlsl_hull_output_control_points
        output_patch_control_points = self.hlsl_output_patch_control_point_count(
            parameters
        )
        if hull_output_control_points is None or output_patch_control_points is None:
            return
        if output_patch_control_points != hull_output_control_points:
            raise ValueError(
                "DirectX tessellation_evaluation stage OutputPatch control "
                f"point count ({output_patch_control_points}) must match the "
                "tessellation_control outputcontrolpoints "
                f"({hull_output_control_points})"
            )

    def validate_hlsl_domain_matches_hull(self, func, shader_type):
        if shader_type != "tessellation_evaluation":
            return

        hull_domain = self.current_hlsl_hull_domain
        domain = self.canonical_hlsl_tessellation_domain(
            self.normalized_hlsl_stage_attribute_argument(func, "domain")
        )
        if hull_domain is None or domain is None:
            return
        if domain != hull_domain:
            raise ValueError(
                "DirectX tessellation_evaluation stage domain "
                f"'{domain}' must match tessellation_control domain "
                f"'{hull_domain}'"
            )

    def hlsl_parameter_has_semantic(self, parameter, expected_semantic):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        mapped_semantic = self.semantic_map.get(semantic, semantic)
        return str(mapped_semantic).lower() == expected_semantic.lower()

    def hlsl_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            primitive = self.hlsl_stage_attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        return qualifiers

    def apply_hlsl_parameter_qualifiers(self, param_type, parameter):
        qualifiers = self.hlsl_parameter_qualifiers(parameter)
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
                raise ValueError(
                    "DirectX geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.hlsl_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
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
                raise ValueError(
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
                raise ValueError(
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
                raise ValueError(
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
                raise ValueError(
                    f"DirectX {shader_type} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_hlsl_compute_system_value_types(self, parameters):
        self.validate_hlsl_exact_semantic_type(
            parameters, "compute", "SV_GroupIndex", "uint", "scalar uint"
        )
        for semantic in (
            "SV_GroupID",
            "SV_GroupThreadID",
            "SV_DispatchThreadID",
        ):
            self.validate_hlsl_exact_semantic_type(
                parameters, "compute", semantic, "uint3", "uint3"
            )

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
            raise ValueError(
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
            raise ValueError(
                f"DirectX {shader_type} semantic '{semantic}' cannot be used "
                "as a function return semantic"
            )

        expected_type = self.hlsl_output_builtin_semantic_type(mapped_semantic)
        if expected_type is None:
            return

        if array_suffix or actual_base_type != expected_type:
            raise ValueError(
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
        expected_type = self.hlsl_output_builtin_semantic_type(mapped_semantic)
        if expected_type is None:
            return

        actual_type = self.map_type(member_type)
        actual_base_type, array_suffix = split_array_type_suffix(actual_type)
        if array_suffix or actual_base_type != expected_type:
            raise ValueError(
                f"DirectX struct '{struct_name}' semantic '{semantic}' maps to "
                f"'{mapped_semantic}' and requires member '{member_name}' to "
                f"have type {expected_type}, got {actual_type}"
            )

    def hlsl_output_builtin_semantic_type(self, mapped_semantic):
        if mapped_semantic is None:
            return None

        semantic_key = str(mapped_semantic).upper()
        output_types = {
            "SV_POSITION": "float4",
            "SV_DEPTH": "float",
        }
        if semantic_key in output_types:
            return output_types[semantic_key]
        if semantic_key.startswith("SV_TARGET"):
            suffix = semantic_key[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "float4"
        return None

    def validate_hlsl_stage_return_semantics(self, func, shader_type, semantic=None):
        if semantic is None:
            semantic = self.hlsl_function_return_semantic(func)
        self.validate_hlsl_stage_output_semantic(
            shader_type,
            semantic,
            "function return",
            getattr(func, "name", "<anonymous>"),
        )

        return_struct = self.hlsl_return_struct(func)
        if return_struct is None:
            return

        struct_name = getattr(return_struct, "name", "<anonymous>")
        for member in getattr(return_struct, "members", []) or []:
            self.validate_hlsl_stage_output_semantic(
                shader_type,
                self.semantic_from_struct_member(member),
                f"return struct '{struct_name}' member",
                getattr(member, "name", "<anonymous>"),
            )

    def validate_hlsl_stage_output_semantic(self, shader_type, semantic, context, name):
        if semantic is None:
            return

        mapped_semantic = self.hlsl_canonical_semantic(semantic)
        semantic_key = str(mapped_semantic).upper()
        forbidden_description = None
        if shader_type == "vertex" and (
            semantic_key == "SV_DEPTH" or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "fragment output"
        elif shader_type in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
        } and (
            semantic_key == "SV_DEPTH" or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "fragment output"
        elif shader_type == "fragment" and semantic_key == "SV_POSITION":
            forbidden_description = "vertex position output"
        elif shader_type == "compute" and (
            semantic_key in {"SV_POSITION", "SV_DEPTH"}
            or self.is_hlsl_target_semantic(semantic_key)
        ):
            forbidden_description = "graphics output"

        if forbidden_description is None:
            return

        raise ValueError(
            f"DirectX {shader_type} stage {context} '{name}' cannot use "
            f"{forbidden_description} semantic '{mapped_semantic}'"
        )

    def is_hlsl_target_semantic(self, semantic_key):
        if not semantic_key.startswith("SV_TARGET"):
            return False
        suffix = semantic_key[len("SV_TARGET") :]
        return suffix == "" or suffix.isdigit()

    def hlsl_canonical_semantic(self, semantic):
        if semantic is None:
            return None
        return self.semantic_map.get(str(semantic), str(semantic))

    def validate_hlsl_stage_parameter_requirements(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        if not parameters:
            return

        parameter_type_bases = {
            self.hlsl_parameter_type_base(parameter) for parameter in parameters
        }
        if shader_type == "geometry" and not (
            parameter_type_bases & {"PointStream", "LineStream", "TriangleStream"}
        ):
            raise ValueError(
                "DirectX geometry stage parameters must include an HLSL stream "
                "output parameter: PointStream<T>, LineStream<T>, or "
                "TriangleStream<T>"
            )
        if shader_type == "geometry" and not any(
            set(self.hlsl_parameter_qualifiers(parameter))
            & {"point", "line", "triangle", "lineadj", "triangleadj"}
            for parameter in parameters
        ):
            raise ValueError(
                "DirectX geometry stage parameters must include an HLSL input "
                "primitive qualifier: point, line, triangle, lineadj, or "
                "triangleadj"
            )
        if shader_type == "geometry":
            self.validate_hlsl_geometry_input_primitive_arity(parameters)
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "geometry",
                ("SV_PrimitiveID", "SV_GSInstanceID"),
            )
            self.validate_hlsl_geometry_stream_output_semantics(parameters)

        if shader_type == "tessellation_control":
            if "InputPatch" not in parameter_type_bases:
                raise ValueError(
                    "DirectX tessellation_control stage parameters must include "
                    "an InputPatch<T, N> parameter"
                )
            self.validate_hlsl_output_control_points(func, parameters)
            if not any(
                self.hlsl_parameter_has_semantic(parameter, "SV_OutputControlPointID")
                for parameter in parameters
            ):
                raise ValueError(
                    "DirectX tessellation_control stage parameters must include "
                    "an SV_OutputControlPointID parameter"
                )
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "tessellation_control",
                ("SV_OutputControlPointID", "SV_PrimitiveID"),
            )

        if shader_type == "tessellation_evaluation":
            if "OutputPatch" not in parameter_type_bases:
                raise ValueError(
                    "DirectX tessellation_evaluation stage parameters must include "
                    "an OutputPatch<T, N> parameter"
                )
            self.validate_hlsl_domain_output_patch_control_points(parameters)
            if not any(
                self.hlsl_parameter_has_semantic(parameter, "SV_DomainLocation")
                for parameter in parameters
            ):
                raise ValueError(
                    "DirectX tessellation_evaluation stage parameters must include "
                    "an SV_DomainLocation parameter"
                )
            self.validate_hlsl_domain_location_type(parameters)
            self.validate_hlsl_domain_location_components(func, parameters)
            self.validate_hlsl_scalar_integer_semantic_types(
                parameters,
                "tessellation_evaluation",
                ("SV_PrimitiveID",),
            )

        if shader_type == "compute":
            self.validate_hlsl_compute_system_value_types(parameters)

    def validate_hlsl_geometry_stream_output_semantics(self, parameters):
        stream_types = {"PointStream", "LineStream", "TriangleStream"}
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) not in stream_types:
                continue

            stream_type_name = self.hlsl_parameter_type_generic_argument(parameter, 0)
            if stream_type_name is None:
                continue

            stream_type_name = (
                stream_type_name.split("<", 1)[0].split("[", 1)[0].strip()
            )
            stream_struct = self.structs_by_name.get(stream_type_name)
            if stream_struct is None:
                continue

            for member in getattr(stream_struct, "members", []) or []:
                self.validate_hlsl_stage_output_semantic(
                    "geometry",
                    self.semantic_from_struct_member(member),
                    f"output stream struct '{stream_type_name}' member",
                    getattr(member, "name", "<anonymous>"),
                )

    def validate_hlsl_patch_constant_function(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        patch_function_name = self.hlsl_stage_attribute_argument(
            func, "patchconstantfunc"
        )
        if not patch_function_name:
            return

        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        available_functions = self.current_hlsl_available_functions or {}
        patch_function = available_functions.get(patch_function_name)
        if patch_function is None or patch_function is func:
            if not parameters:
                return
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must reference an emitted helper "
                "function in the global or tessellation_control stage scope"
            )

        return_type = self.type_name_string(
            getattr(
                patch_function, "return_type", getattr(patch_function, "vtype", None)
            )
        )
        if return_type is None or self.map_type(return_type) == "void":
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return patch-constant data"
            )

        patch_parameters = (
            getattr(patch_function, "parameters", getattr(patch_function, "params", []))
            or []
        )
        if patch_parameters and "InputPatch" not in {
            self.hlsl_parameter_type_base(parameter) for parameter in patch_parameters
        }:
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' parameters must include "
                "InputPatch<T, N>"
            )
        self.validate_hlsl_scalar_integer_semantic_types(
            patch_parameters,
            "tessellation_control patchconstantfunc",
            ("SV_PrimitiveID",),
        )
        self.validate_hlsl_patch_constant_tess_factor_semantics(
            func, patch_function, patch_function_name
        )

    def hlsl_return_struct(self, func):
        return_type = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type:
            return None
        base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
        return self.structs_by_name.get(base_type)

    def hlsl_struct_semantics(self, struct_node):
        semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            mapped_semantic = self.semantic_map.get(semantic, semantic)
            semantics.add(str(mapped_semantic).lower())
        return semantics

    def hlsl_struct_semantic_member_counts(self, struct_node):
        counts = {}
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            mapped_semantic = self.semantic_map.get(semantic, semantic)
            semantic_key = str(mapped_semantic).lower()
            count = self.hlsl_tess_factor_member_count(member)
            if count is not None:
                counts[semantic_key] = counts.get(semantic_key, 0) + count
        return counts

    def hlsl_tess_factor_member_count(self, member):
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None and hasattr(member, "element_type"):
            member_type = member.element_type

        type_name = self.type_name_string(member_type)
        mapped_type = self.map_type(type_name)
        _base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            first_dimension = array_suffix[1:].split("]", 1)[0]
            try:
                return int(first_dimension)
            except ValueError:
                return None

        for scalar_base in ("min16float", "float", "half", "double"):
            if mapped_type.startswith(scalar_base):
                suffix = mapped_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    return int(suffix)
                if suffix == "":
                    return 1
        return None

    def validate_hlsl_patch_constant_tess_factor_semantics(
        self, hull_func, patch_function, patch_function_name
    ):
        return_struct = self.hlsl_return_struct(patch_function)
        if return_struct is None:
            return

        semantics = self.hlsl_struct_semantics(return_struct)
        if "sv_tessfactor" not in semantics:
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return a struct containing "
                "SV_TessFactor"
            )

        normalized_domain = self.normalized_hlsl_stage_attribute_argument(
            hull_func, "domain"
        )
        if normalized_domain in {"tri", "triangle", "quad"} and (
            "sv_insidetessfactor" not in semantics
        ):
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return a struct containing "
                f"SV_InsideTessFactor for {normalized_domain} domains"
            )
        factor_counts = self.hlsl_struct_semantic_member_counts(return_struct)
        expected_outer_counts = {
            "tri": 3,
            "triangle": 3,
            "quad": 4,
            "isoline": 2,
        }
        expected_inner_counts = {
            "tri": 1,
            "triangle": 1,
            "quad": 2,
        }
        expected_outer_count = expected_outer_counts.get(normalized_domain)
        expected_inner_count = expected_inner_counts.get(normalized_domain)
        outer_count = factor_counts.get("sv_tessfactor")
        inner_count = factor_counts.get("sv_insidetessfactor")

        if (
            expected_outer_count is not None
            and outer_count is not None
            and outer_count != expected_outer_count
        ):
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return {expected_outer_count} "
                f"SV_TessFactor value(s) for {normalized_domain} domains"
            )
        if (
            expected_inner_count is not None
            and inner_count is not None
            and inner_count != expected_inner_count
        ):
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return {expected_inner_count} "
                f"SV_InsideTessFactor value(s) for {normalized_domain} domains"
            )
        if expected_inner_count is None and inner_count is not None:
            raise ValueError(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must not return SV_InsideTessFactor "
                f"for {normalized_domain} domains"
            )

    def validate_hlsl_tessellation_domain_topology(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        domain = self.normalized_hlsl_stage_attribute_argument(func, "domain")
        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        if not domain or not topology:
            return

        if domain in {"tri", "triangle", "quad"} and topology == "line":
            raise ValueError(
                "DirectX tessellation_control stage domain "
                f"'{domain}' requires outputtopology triangle_cw or "
                "triangle_ccw"
            )
        if domain == "isoline" and topology in {"triangle_cw", "triangle_ccw"}:
            raise ValueError(
                "DirectX tessellation_control stage domain 'isoline' "
                "requires outputtopology line"
            )

    def validate_hlsl_tessellation_domain(self, func, shader_type):
        if shader_type not in {"tessellation_control", "tessellation_evaluation"}:
            return

        domain = self.normalized_hlsl_stage_attribute_argument(func, "domain")
        if not domain:
            return

        canonical_domain = self.canonical_hlsl_tessellation_domain(domain)
        valid_domains = {"tri", "quad", "isoline"}
        if canonical_domain not in valid_domains:
            valid_values = ", ".join(sorted(valid_domains))
            raise ValueError(
                f"DirectX {shader_type} stage domain '{domain}' must be "
                f"one of: {valid_values}"
            )

    def validate_hlsl_tessellation_output_topology(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        if not topology:
            return

        valid_topologies = {
            "point",
            "line",
            "triangle_cw",
            "triangle_ccw",
        }
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise ValueError(
                "DirectX tessellation_control stage outputtopology "
                f"'{topology}' must be one of: {valid_values}"
            )

    def validate_hlsl_tessellation_partitioning(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        partitioning = self.normalized_hlsl_stage_attribute_argument(
            func, "partitioning"
        )
        if not partitioning:
            return

        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if partitioning not in valid_partitioning:
            valid_values = ", ".join(sorted(valid_partitioning))
            raise ValueError(
                "DirectX tessellation_control stage partitioning "
                f"'{partitioning}' must be one of: {valid_values}"
            )

    def validate_hlsl_stage_requirements(self, func, shader_type):
        required_attributes = {
            "geometry": {"maxvertexcount"},
            "tessellation_control": {
                "domain",
                "outputcontrolpoints",
                "outputtopology",
                "partitioning",
                "patchconstantfunc",
            },
            "tessellation_evaluation": {"domain"},
        }.get(shader_type)
        if not required_attributes:
            return

        present_attributes = self.hlsl_stage_attribute_names(func)
        missing_attributes = sorted(required_attributes - present_attributes)
        if missing_attributes:
            missing = ", ".join(missing_attributes)
            raise ValueError(
                f"DirectX {shader_type} stage requires HLSL attribute(s): {missing}"
            )
        self.validate_hlsl_tessellation_domain(func, shader_type)
        self.validate_hlsl_tessellation_output_topology(func, shader_type)
        self.validate_hlsl_tessellation_domain_topology(func, shader_type)
        self.validate_hlsl_tessellation_partitioning(func, shader_type)
        self.validate_hlsl_output_control_points_value(func, shader_type)
        self.validate_hlsl_max_vertex_count_value(func, shader_type)
        self.validate_hlsl_domain_matches_hull(func, shader_type)
        self.validate_hlsl_stage_parameter_requirements(func, shader_type)
        self.validate_hlsl_patch_constant_function(func, shader_type)

    def hlsl_stage_attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)

    def generate_hlsl_stage_attributes(self, func, shader_type):
        if shader_type not in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        }:
            return ""

        quoted_argument_attributes = {
            "domain",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        code = ""
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if not attr_name:
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            argument_values = [
                self.hlsl_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            if attr_name == "domain":
                argument_values = [
                    self.canonical_hlsl_tessellation_domain(value) or value
                    for value in argument_values
                ]
            if attr_name in quoted_argument_attributes:
                argument_values = [f'"{value}"' for value in argument_values]
            code += f"[{attr_name}({', '.join(argument_values)})]\n"

        return code

    def stage_entry_names(self, ast, target_stage=None):
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, target_stage, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, target_stage, stage_entry_types
        )
        return assign_stage_entry_names(entries, used_names, self.stage_entry_base_name)

    def hlsl_stage_entry_functions(self, ast, expected_stage_name):
        functions = []
        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            if normalize_stage_name(qualifier) == expected_stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            if normalize_stage_name(stage_type) != expected_stage_name:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
        return functions

    def hlsl_program_hull_output_control_points(self, ast):
        counts = {
            self.hlsl_stage_attribute_int_argument(func, "outputcontrolpoints")
            for func in self.hlsl_stage_entry_functions(ast, "tessellation_control")
        }
        counts.discard(None)
        if len(counts) == 1:
            return next(iter(counts))
        return None

    def hlsl_program_hull_domain(self, ast):
        domains = {
            self.canonical_hlsl_tessellation_domain(
                self.normalized_hlsl_stage_attribute_argument(func, "domain")
            )
            for func in self.hlsl_stage_entry_functions(ast, "tessellation_control")
        }
        domains.discard(None)
        if len(domains) == 1:
            return next(iter(domains))
        return None

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def collect_function_parameters(self, functions):
        parameters = []
        for func in functions or []:
            parameters.extend(getattr(func, "parameters", getattr(func, "params", [])))
        return parameters

    def collect_resource_array_size_hints(self, ast):
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_resource_globals(ast),
            function_arrays=self.collect_unsized_resource_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=self.collect_functions(ast),
            walk_nodes=self.walk_ast,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=0,
            format_size=lambda size: str(size) if size > 1 else "",
            initial_literal_int_constants=self.initial_literal_int_constants,
        )

    def collect_unsized_resource_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if (
                name
                and self.is_unsized_resource_array_type(vtype)
                and not self.is_hlsl_buffer_resource_array_type(vtype)
            ):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

    def collect_unsized_resource_parameters(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                size = self.fixed_resource_array_size(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_array_hint_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_array_hint_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_array_hint_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_array_hint_type(base_type)

    def is_resource_array_hint_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            or self.is_hlsl_readonly_buffer_type(vtype)
            or self.is_hlsl_uav_buffer_type(vtype)
            or str(self.resource_base_type(vtype))
            in self.glsl_buffer_block_struct_names
        )

    def is_hlsl_buffer_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
        elif hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        else:
            type_string = str(vtype)
            if "[" not in type_string or "]" not in type_string:
                return False
            base_type, _ = parse_array_type(type_string)
        return self.is_hlsl_readonly_buffer_type(
            base_type
        ) or self.is_hlsl_uav_buffer_type(base_type)

    def walk_ast(self, root):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    yield from walk(child)

        yield from walk(root)

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def is_shadow_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def is_multisample_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_sampler_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return mapped_type in {"SamplerState", "SamplerComparisonState"}

    def is_texture_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith("Texture")

    def is_image_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith("RWTexture")

    def is_texture_or_image_type(self, vtype):
        return self.is_texture_type(vtype) or self.is_image_type(vtype)

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = texture_type.startswith("Texture2DMS")
        is_storage_image = texture_type.startswith("RWTexture")

        coordinate_dimension = None
        if texture_type and "Cube" not in texture_type:
            if texture_type.startswith("RWTexture1DArray<"):
                coordinate_dimension = 2
            elif texture_type.startswith("RWTexture1D<"):
                coordinate_dimension = 1
            elif texture_type.startswith("RWTexture2DArray<"):
                coordinate_dimension = 3
            elif texture_type.startswith("RWTexture2DMSArray<"):
                coordinate_dimension = 3
            elif texture_type.startswith("RWTexture3D<"):
                coordinate_dimension = 3
            elif texture_type.startswith("RWTexture2DMS<"):
                coordinate_dimension = 2
            elif texture_type.startswith("RWTexture2D<"):
                coordinate_dimension = 2
            elif texture_type.startswith("Texture2DMSArray"):
                coordinate_dimension = 3
            elif texture_type.startswith("Texture2DMS"):
                coordinate_dimension = 2
            else:
                coordinate_dimension = {
                    "Texture1D": 1,
                    "Texture1DArray": 2,
                    "Texture2D": 2,
                    "Texture2DArray": 3,
                    "Texture3D": 3,
                }.get(texture_type)

        offset_dimension = None
        if texture_type and "Cube" not in texture_type:
            offset_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 1,
                "Texture2D": 2,
                "Texture2DArray": 2,
                "Texture3D": 3,
            }.get(texture_type)

        gradient_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            gradient_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 1,
                "Texture2D": 2,
                "Texture2DArray": 2,
                "Texture3D": 3,
                "TextureCube": 3,
                "TextureCubeArray": 3,
            }.get(texture_type)

        query_lod_coordinate_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            query_lod_coordinate_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 2,
                "Texture2D": 2,
                "Texture2DArray": 3,
                "Texture3D": 3,
                "TextureCube": 3,
                "TextureCubeArray": 4,
            }.get(texture_type)

        return texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        key = texture_resource_offset_dimension_key(
            func_name, collapse_compare_offsets=True
        )
        return descriptor[key]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def is_resource_parameter_type(self, vtype):
        return (
            self.is_texture_type(vtype)
            or self.is_sampler_type(vtype)
            or self.is_image_type(vtype)
        )

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.texture_call_uses_explicit_sampler(args)

    def texture_call_uses_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        if (
            sampler_name in self.sampler_variables
            or sampler_name in self.current_sampler_parameters
        ):
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def global_resource_shape(self, node):
        resource_count = 1
        if hasattr(node, "var_type"):
            if hasattr(node.var_type, "name") or hasattr(node.var_type, "element_type"):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):
                    vtype = self.convert_type_node_to_string(node.var_type.element_type)
                    array_size = (
                        self.generate_expression(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    )
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
            else:
                vtype = str(node.var_type)
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            vtype = "float"
        return vtype, resource_count

    def global_resource_register_metadata(self, node):
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        lowered_block = self.lowered_glsl_buffer_blocks.get(var_name)
        if lowered_block is not None:
            if lowered_block["readonly"]:
                prefix = "t"
                attribute_names = {"binding", "texture"}
            else:
                prefix = "u"
                attribute_names = {"binding", "texture", "uav"}
        else:
            if self.is_glsl_buffer_block_variable(node, vtype):
                return None
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type.startswith(
                "Texture"
            ) or self.is_multisample_storage_image_resource_type(mapped_type):
                prefix = "t"
                attribute_names = {"binding", "texture"}
            elif mapped_type.startswith("RWTexture"):
                prefix = "u"
                attribute_names = {"binding", "texture", "uav"}
            elif self.is_hlsl_uav_buffer_type(mapped_type):
                prefix = "u"
                attribute_names = {"binding", "texture", "uav"}
            elif self.is_hlsl_readonly_buffer_type(mapped_type):
                prefix = "t"
                attribute_names = {"binding", "texture"}
            elif mapped_type in ["SamplerState", "SamplerComparisonState"]:
                prefix = "s"
                attribute_names = {"binding", "sampler"}
            else:
                return None

        binding = self.explicit_resource_binding_index(node, attribute_names, (prefix,))
        if binding is None:
            return None
        space = self.explicit_resource_register_space(node)
        return prefix, binding, space, resource_count, var_name

    def reserve_explicit_global_resource_registers(self, global_vars, used_registers):
        for node in global_vars:
            metadata = self.global_resource_register_metadata(node)
            if metadata is None:
                continue
            prefix, binding, space, resource_count, var_name = metadata
            self.reserve_resource_register_range(
                used_registers,
                prefix,
                binding,
                resource_count,
                var_name,
                space,
            )

    def next_resource_register(self, register_cursors, space):
        return register_cursors.get(space, 0)

    def next_available_resource_register(
        self, used_registers, register_prefix, register_cursors, space, count
    ):
        count = max(count or 1, 1)
        binding = self.next_resource_register(register_cursors, space)
        ranges = used_registers.get((register_prefix, space), [])
        while True:
            end = binding + count - 1
            conflict_end = None
            for used_start, used_end, _ in ranges:
                if binding <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding
            binding = conflict_end + 1

    def advance_resource_register(self, register_cursors, space, start, count):
        count = max(count or 1, 1)
        register_cursors[space] = max(register_cursors.get(space, 0), start + count)

    def reserve_resource_register_range(
        self, used_registers, register_prefix, start, count, name, space=None
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        namespace = (register_prefix, space)
        ranges = used_registers.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting DirectX resource binding for '{name}': "
                    f"{self.resource_register_range_label(register_prefix, start, end, space)} "
                    f"overlaps '{used_name}' "
                    f"{self.resource_register_range_label(register_prefix, used_start, used_end, space)}"
                )
        ranges.append((start, end, name))

    def resource_register_range_label(self, register_prefix, start, end, space=None):
        if start == end:
            label = f"{register_prefix}{start}"
        else:
            label = f"{register_prefix}{start}-{register_prefix}{end}"
        if space:
            return f"{label}, {space}"
        return label

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if not name:
                    continue

                visible_constants.pop(name, None)
                if "const" not in getattr(node, "qualifiers", []):
                    continue

                value = self.literal_int_value(
                    getattr(node, "initial_value", None), visible_constants
                )
                if value is not None:
                    visible_constants[name] = value

        return visible_constants

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None

    def texture_call_parts(self, args, func_name=None):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        texture_base_name = self.expression_name(args[0]) or texture_name
        if explicit_sampler:
            sampler_name = self.generate_expression(args[1])
        elif self.implicit_call_uses_regular_sampler(func_name):
            sampler_name = self.current_implicit_texture_regular_samplers.get(
                texture_base_name,
                self.global_implicit_texture_regular_samplers.get(
                    texture_base_name,
                    self.current_implicit_texture_samplers.get(
                        texture_base_name, f"{texture_base_name}Sampler"
                    ),
                ),
            )
        else:
            sampler_name = self.current_implicit_texture_samplers.get(
                texture_base_name, f"{texture_base_name}Sampler"
            )
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, sampler_name, coord, extra_args

    def generate_call_arguments(self, func_name, args):
        generated_args = []
        implicit_samplers = self.implicit_texture_sampler_parameters.get(func_name, {})
        param_names = self.function_parameter_names.get(func_name, [])

        for index, arg in enumerate(args):
            generated_args.append(self.generate_expression(arg))
            if index >= len(param_names):
                continue
            texture_param = param_names[index]
            if texture_param not in implicit_samplers:
                continue
            sampler_info = implicit_samplers[texture_param]
            if sampler_info["synthetic"]:
                generated_args.append(self.generate_implicit_sampler_argument(arg))
            regular_sampler_name = self.implicit_regular_sampler_name(
                texture_param, sampler_info
            )
            if (
                sampler_info.get("regular")
                and sampler_info.get("comparison")
                and regular_sampler_name != sampler_info["sampler_name"]
                and regular_sampler_name not in param_names
            ):
                generated_args.append(
                    self.generate_implicit_regular_sampler_argument(arg, sampler_info)
                )
            query_sampler_name = self.implicit_query_lod_sampler_name(
                texture_param, sampler_info
            )
            if (
                sampler_info.get("query_lod")
                and query_sampler_name != sampler_info["sampler_name"]
                and query_sampler_name not in param_names
            ):
                generated_args.append(
                    self.generate_implicit_query_lod_sampler_argument(arg, sampler_info)
                )

        return generated_args

    def generate_implicit_sampler_argument(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_samplers:
            return self.current_implicit_texture_samplers[texture_name]
        if texture_name:
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        return f"{texture_expr}Sampler"

    def implicit_call_uses_regular_sampler(self, func_name):
        return (
            func_name in self.regular_texture_function_names()
            and not is_texture_query_lod_operation(func_name)
        )

    def implicit_regular_sampler_name(self, texture_name, sampler_info):
        if sampler_info.get("regular") and sampler_info.get("comparison"):
            return f"{texture_name}RegularSampler"
        return sampler_info["sampler_name"]

    def generate_implicit_regular_sampler_argument(
        self, texture_arg, sampler_info=None
    ):
        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_regular_samplers:
            return self.current_implicit_texture_regular_samplers[texture_name]
        if texture_name in self.global_implicit_texture_regular_samplers:
            return self.global_implicit_texture_regular_samplers[texture_name]
        if texture_name:
            if sampler_info and sampler_info.get("comparison"):
                return f"{texture_name}RegularSampler"
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        if sampler_info and sampler_info.get("comparison"):
            return f"{texture_expr}RegularSampler"
        return f"{texture_expr}Sampler"

    def implicit_query_lod_sampler_name(self, texture_name, sampler_info):
        if sampler_info.get("query_lod") and sampler_info.get("comparison"):
            return f"{texture_name}QuerySampler"
        return sampler_info["sampler_name"]

    def generate_implicit_query_lod_sampler_argument(
        self, texture_arg, sampler_info=None
    ):
        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_query_lod_samplers:
            return self.current_implicit_texture_query_lod_samplers[texture_name]
        if texture_name in self.global_implicit_texture_query_lod_samplers:
            return self.global_implicit_texture_query_lod_samplers[texture_name]
        if texture_name:
            if sampler_info and sampler_info.get("comparison"):
                return f"{texture_name}QuerySampler"
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        if sampler_info and sampler_info.get("comparison"):
            return f"{texture_expr}QuerySampler"
        return f"{texture_expr}Sampler"

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

    def image_resource_access(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_access_parameters,
            self.image_variable_accesses,
        )

    def image_resource_format(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_texture_or_image_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"DirectX texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_resource_type(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"DirectX image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.image_resource_access(args[0])
        if access is None or access == "read_write":
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise ValueError(
                f"DirectX image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got writeonly"
            )
        if func_name == "imageStore" and access == "read":
            raise ValueError(
                f"DirectX image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got readonly"
            )
        if self.image_atomic_intrinsic(func_name):
            access_name = "readonly" if access == "read" else "writeonly"
            raise ValueError(
                f"DirectX image operation '{func_name}' requires read-write "
                f"storage image access for {texture_name}: got {access_name}"
            )

    def hlsl_storage_image_component_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        if "<" not in image_type or ">" not in image_type:
            return None
        return image_type.split("<", 1)[1].split(">", 1)[0].strip()

    def hlsl_storage_image_component_kind(self, image_type):
        component_type = self.hlsl_storage_image_component_type(image_type)
        component_kind = self.vector_component_type(component_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_atomic_value_arguments(self, func_name, args, image_type):
        has_sample = self.is_multisample_storage_image_resource_type(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, image_type, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args, image_type),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_atomic_value_kind_error(
                "DirectX",
                func_name,
                format_label,
                component_kind,
                expression_debug_name(value_arg),
                value_kind,
            )
        )

    def scalar_expected_kind(self):
        return numeric_scalar_type_kind(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_result_type(
        self, func_name, image_type, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_atomic_result_kind_error(
                "DirectX", func_name, format_label, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        component_kind = resolve_image_atomic_component_kind(
            func_name,
            image_format,
            self.hlsl_storage_image_component_type(image_type),
            "DirectX",
            self.resource_base_type(image_type),
        )
        self.validate_image_atomic_value_argument_types(
            func_name,
            args,
            image_type,
            component_kind,
            image_format,
        )
        self.validate_image_atomic_result_type(
            func_name,
            image_type,
            component_kind,
            image_format,
        )

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.image_resource_access(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise ValueError(
                f"DirectX function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got {actual_label}"
            )

    def validate_integer_coordinate_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        coord_type = self.expression_result_type(args[1])
        if coord_type is None or self.is_integer_coordinate_type(coord_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "resource",
                func_name,
                "an integer",
                "coordinate",
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_coordinate_dimension_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[1])
        coord_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(coord_type), self.map_type
        )
        if coord_dimension is None or coord_dimension == expected_dimension:
            return
        raise ValueError(
            operation_dimension_argument_error(
                "DirectX",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "coordinate",
                self.resource_base_type(texture_type),
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_offset_dimension_argument(self, func_name, args):
        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not offset_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for offset_index in offset_indices:
            offset_type = self.expression_result_type(args[offset_index])
            if offset_type is None:
                continue
            if not self.is_integer_coordinate_type(offset_type):
                raise ValueError(
                    operation_argument_type_error(
                        "DirectX",
                        "resource",
                        func_name,
                        "an integer",
                        "offset",
                        expression_debug_name(args[offset_index]),
                        self.type_name_string(offset_type),
                    )
                )
            offset_dimension = integer_coordinate_dimension_from_type_name(
                self.type_name_string(offset_type), self.map_type
            )
            if offset_dimension is None or offset_dimension == expected_dimension:
                continue
            raise ValueError(
                operation_dimension_argument_error(
                    "DirectX",
                    "resource",
                    func_name,
                    expected_dimension,
                    "integer",
                    "offset",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[offset_index]),
                    self.type_name_string(offset_type),
                )
            )

    def gradient_argument_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def query_lod_coordinate_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def validate_query_lod_coordinate_argument(self, func_name, args):
        coord_index = texture_query_lod_coordinate_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if coord_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_query_lod_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[coord_index])
        if coord_type is None:
            return
        coord_dimension = self.query_lod_coordinate_dimension(coord_type)
        if coord_dimension is None:
            raise ValueError(
                texture_query_lod_coordinate_type_error(
                    "DirectX",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise ValueError(
            texture_query_lod_coordinate_dimension_error(
                "DirectX",
                func_name,
                expected_dimension,
                self.resource_base_type(texture_type),
                expression_debug_name(args[coord_index]),
                self.type_name_string(coord_type),
            )
        )

    def validate_gradient_dimension_arguments(self, func_name, args):
        gradient_indices = texture_gradient_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not gradient_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_gradient_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for gradient_index in gradient_indices:
            gradient_type = self.expression_result_type(args[gradient_index])
            if gradient_type is None:
                continue
            gradient_dimension = self.gradient_argument_dimension(gradient_type)
            if gradient_dimension is None:
                raise ValueError(
                    operation_argument_type_error(
                        "DirectX",
                        "resource",
                        func_name,
                        "a floating",
                        "gradient",
                        expression_debug_name(args[gradient_index]),
                        self.type_name_string(gradient_type),
                    )
                )
            if gradient_dimension == expected_dimension:
                continue
            raise ValueError(
                operation_dimension_argument_error(
                    "DirectX",
                    "resource",
                    func_name,
                    expected_dimension,
                    "floating",
                    "gradient",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[gradient_index]),
                    self.type_name_string(gradient_type),
                )
            )

    def is_scalar_floating_type(self, vtype):
        return is_floating_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_numeric_type(self, vtype):
        return is_numeric_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_integer_type(self, vtype):
        return is_integer_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def texture_argument_diagnostic_type(self, arg):
        return shared_texture_argument_diagnostic_type(
            arg,
            self.texture_resource_type,
            self.expression_name,
            self.expression_result_type,
            self.sampler_variables | self.current_sampler_parameters,
        )

    def validate_compare_argument(self, func_name, args):
        compare_index = texture_compare_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if compare_index is None:
            return
        compare_type = self.expression_result_type(args[compare_index])
        if compare_type is None or self.is_scalar_floating_type(compare_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "texture compare",
                func_name,
                "a scalar floating",
                "compare",
                expression_debug_name(args[compare_index]),
                self.type_name_string(compare_type),
            )
        )

    def validate_lod_argument(self, func_name, args):
        lod_index = texture_lod_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if lod_index is None:
            return
        lod_type = self.texture_argument_diagnostic_type(args[lod_index])
        if lod_type is None or self.is_scalar_numeric_type(lod_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "texture LOD",
                func_name,
                "a scalar numeric",
                "lod",
                expression_debug_name(args[lod_index]),
                self.type_name_string(lod_type),
            )
        )

    def validate_bias_argument(self, func_name, args):
        bias_index = texture_bias_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if bias_index is None:
            return
        bias_type = self.texture_argument_diagnostic_type(args[bias_index])
        if bias_type is None or self.is_scalar_numeric_type(bias_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "texture bias",
                func_name,
                "a scalar numeric",
                "bias",
                expression_debug_name(args[bias_index]),
                self.type_name_string(bias_type),
            )
        )

    def validate_mip_level_argument(self, func_name, args):
        level_index = texture_mip_level_argument_index(func_name, len(args))
        if level_index is None:
            return
        level_type = self.texture_argument_diagnostic_type(args[level_index])
        if level_type is None or self.is_scalar_integer_type(level_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "resource",
                func_name,
                "a scalar integer",
                "mip/sample level",
                expression_debug_name(args[level_index]),
                self.type_name_string(level_type),
            )
        )

    def validate_sample_index_argument(self, func_name, args):
        sample_index = texture_sample_index_argument_index(func_name, len(args))
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource_type(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise ValueError(
            texture_multisample_sample_type_error(
                "DirectX",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_image_multisample_arguments(self, func_name, args):
        if not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        sample_index = image_multisample_sample_argument_index(
            func_name,
            len(args),
            self.is_multisample_storage_image_resource_type(texture_type),
            "DirectX",
        )
        if sample_index is None:
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        sample_type = image_multisample_sample_type_mismatch(
            sample_type, self.is_scalar_integer_type
        )
        if sample_type is None:
            return
        raise ValueError(
            image_multisample_sample_type_error(
                "DirectX",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_gather_component_argument(self, func_name, args):
        component_index = texture_gather_component_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if component_index is None:
            return
        component_type = self.texture_argument_diagnostic_type(args[component_index])
        if component_type is None or self.is_scalar_integer_type(component_type):
            return
        raise ValueError(
            operation_argument_type_error(
                "DirectX",
                "texture gather",
                func_name,
                "a scalar integer",
                "component",
                expression_debug_name(args[component_index]),
                self.type_name_string(component_type),
            )
        )

    def validate_texture_call_arity(self, func_name, args):
        validate_texture_operation_arity(
            "DirectX",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def four_component_image_store_constructor(self, texture_type):
        component_type = self.hlsl_storage_image_component_type(texture_type)
        return storage_image_store_vector_constructor(
            component_type,
            4,
            self.hlsl_storage_image_component_kind(texture_type),
        )

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def two_component_image_store_constructor(self, texture_type):
        component_type = self.hlsl_storage_image_component_type(texture_type)
        return storage_image_store_vector_constructor(
            component_type,
            2,
            self.hlsl_storage_image_component_kind(texture_type),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

    def image_store_expected_value_type(self, image_type, image_format, value_arg=None):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_kind = self.hlsl_storage_image_component_kind(image_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_load_result_type(self, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            return image_format_result_type(image_format)
        image_type = self.texture_argument_resource_type(image_arg)
        return self.hlsl_storage_image_component_type(image_type)

    def image_load_component_kind(self, image_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        return self.hlsl_storage_image_component_kind(image_type)

    def image_load_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            self.value_component_count(
                self.hlsl_storage_image_component_type(image_type)
            ),
        )

    def expected_component_kind(self):
        return numeric_component_kind_from_type(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def expected_component_count(self):
        return self.value_component_count(self.current_expression_expected_type)

    def expression_component_kind(self, expr):
        return numeric_expression_component_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def value_component_count(self, vtype):
        return numeric_component_count_from_type(
            vtype,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("x",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("x",),
        )

    def image_store_channel_count(self, image_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            self.value_component_count(
                self.hlsl_storage_image_component_type(image_type)
            ),
        )

    def validate_image_store_value_shape(self, image_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(image_type, image_format)
        value_channels = image_store_value_shape_mismatch(
            expected_channels, self.expression_component_count(value_arg)
        )
        if value_channels is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_store_value_shape_error(
                "DirectX",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, image_type, image_format, value_arg):
        self.validate_image_store_value_shape(image_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(image_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise ValueError(
            image_store_value_kind_error(
                "DirectX",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, image_type, image_format):
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(image_type, image_format)
        format_label = image_format or self.resource_base_type(image_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise ValueError(
                image_load_result_kind_error(
                    "DirectX", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(image_type, image_format)
        expected_channels = image_load_result_shape_mismatch(
            loaded_channels,
            expected_channels,
        )
        if expected_channels is None:
            return
        raise ValueError(
            image_load_result_shape_error(
                "DirectX", format_label, loaded_channels, expected_channels
            )
        )

    def texture_query_dimension(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type in {"Texture1D"}:
            return 1
        if texture_type in {"Texture1DArray"}:
            return 2
        if texture_type in {
            "Texture2D",
            "TextureCube",
            "Texture2DMS<float4>",
        }:
            return 2
        return 3

    def is_storage_image_resource_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type.startswith(
            (
                "RWTexture1D<",
                "RWTexture1DArray<",
                "RWTexture2D<",
                "RWTexture3D<",
                "RWTexture2DArray<",
                "RWTexture2DMS<",
                "RWTexture2DMSArray<",
                "RWTextureCube<",
            )
        )

    def texture_query_helper_key(self, helper_name, texture_type):
        if not texture_type:
            return None
        return helper_name, texture_type

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        storage_image = self.is_storage_image_resource_type(texture_type)
        return {
            "texture_type": texture_type,
            "storage_image": storage_image,
            "multisample": (
                texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}
                or self.is_multisample_storage_image_resource_type(texture_type)
            ),
            "size_descriptor": (
                self.image_size_helper_descriptor(texture_type)
                if storage_image
                else self.texture_size_helper_descriptor(texture_type)
            ),
            "levels_descriptor": (
                None
                if storage_image
                else self.texture_query_levels_helper_descriptor(texture_type)
            ),
            "samples_descriptor": self.texture_samples_helper_descriptor(texture_type),
        }

    def texture_query_size_expression(self, texture_arg, lod_arg=None):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            key = self.texture_query_helper_key("imageSize", texture_type)
            if key:
                self.required_texture_query_helpers.add(key)
            return f"imageSize({texture_name})"
        key = self.texture_query_helper_key("textureSize", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        if descriptor["multisample"]:
            return f"textureSize({texture_name})"
        lod = self.generate_expression(lod_arg) if lod_arg is not None else "0"
        return f"textureSize({texture_name}, {lod})"

    def texture_query_levels_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        key = self.texture_query_helper_key("textureQueryLevels", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        return f"textureQueryLevels({texture_name})"

    def texture_samples_expression(self, texture_arg):
        texture_name = self.generate_expression(texture_arg)
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if not descriptor["multisample"]:
            return unsupported_texture_samples_query_call_expression("DirectX")
        key = self.texture_query_helper_key("textureSamples", texture_type)
        if key:
            self.required_texture_query_helpers.add(key)
        return texture_samples_query_expression("DirectX", texture_name)

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("DirectX", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def is_array_expression(self, node):
        type_name = self.expression_result_type(node)
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_method(self, component_arg):
        if component_arg is None:
            return "Gather"

        methods = {
            0: "GatherRed",
            1: "GatherGreen",
            2: "GatherBlue",
            3: "GatherAlpha",
        }
        return methods.get(self.literal_int_value(component_arg))

    def texture_gather_component_expression(self, texture_name, method_args, component):
        arg_list = ", ".join(method_args)
        component_calls = [
            f"{texture_name}.{method}({arg_list})"
            for method in (
                "GatherRed",
                "GatherGreen",
                "GatherBlue",
                "GatherAlpha",
            )
        ]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("DirectX", func_name, reason)

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        gather_types = {
            "Texture2D",
            "Texture2DArray",
            "TextureCube",
            "TextureCubeArray",
        }
        offset_types = {"Texture2D", "Texture2DArray"}
        sample_offset_types = {
            "",
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture3D",
        }
        return {
            "texture_type": texture_type,
            "gather": texture_type in gather_types,
            "gather_offset": texture_type in offset_types,
            "sample_offset": texture_type in sample_offset_types,
            "compare_offset": texture_type in offset_types,
            "gather_compare_offset": texture_type in offset_types,
        }

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_gather_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_call_vector_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_query_lod_expression(
            "DirectX", texture_type
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_compare_scalar_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "DirectX", func_name, texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_texture_query_levels_expression("DirectX", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return unsupported_texture_query_lod_expression("DirectX", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_resource_type(texture_type):
            return None

        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "DirectX", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "DirectX", func_name, texture_type
            )

        return None

    def is_cube_texture_resource_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return texture_type in {"TextureCube", "TextureCubeArray"}

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_cube_texel_fetch_expression(
            "DirectX", func_name, texture_type
        )

    def texture_sample_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        return unsupported_texture_offset_call_expression("DirectX", func_name, reason)

    def generate_texture_sample_offset_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if not self.texture_sample_offset_supported(texture_type):
            return self.unsupported_texture_sample_offset_call(
                func_name, texture_sample_offset_capability_error("DirectX")
            )

        if is_texture_sample_basic_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            offset = self.generate_expression(extra_args[0])
            if len(extra_args) == 2:
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {coord}, {bias}, {offset})"
                )
            return f"{texture_name}.Sample({sampler_name}, {coord}, {offset})"

        if is_texture_sample_lod_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleLevel({sampler_name}, {coord}, {lod}, {offset})"
            )

        if is_texture_sample_grad_offset_operation(func_name):
            count_error = texture_sample_offset_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_sample_offset_call(
                    func_name, count_error
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {coord}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_sample_offset_call(
            func_name, unsupported_texture_offset_operation_error()
        )

    def unsupported_texture_projected_call(self, func_name, reason):
        return unsupported_projected_texture_call_expression(
            "DirectX", func_name, reason
        )

    def projected_texture_coord(self, texture_arg, coord_arg, coord):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        specs = {
            "Texture1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "Texture2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "Texture2DArray": {
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "Texture3D": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
            "TextureCube": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        texture_specs = specs.get(texture_type)
        if texture_specs is None:
            return None
        coord_spec = texture_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        projected_coord = (
            f"{self.vector_component(coord, numerator)} / "
            f"{self.vector_component(coord, divisor)}"
        )
        if texture_type == "Texture2DArray":
            return f"float3({projected_coord}, {self.vector_component(coord, 'z')})"
        return projected_coord

    def generate_texture_projected_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        projected_coord = self.projected_texture_coord(
            args[0], args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires 1D, 2D, or 3D projection coordinates"
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not extra_args:
                return f"{texture_name}.Sample({sampler_name}, {projected_coord})"
            if len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias})"
                )

        if is_projected_texture_basic_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return (
                    f"{texture_name}.Sample("
                    f"{sampler_name}, {projected_coord}, {offset})"
                )
            if len(extra_args) == 2:
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleBias("
                    f"{sampler_name}, {projected_coord}, {bias}, {offset})"
                )

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            lod = self.generate_expression(extra_args[0])
            return (
                f"{texture_name}.SampleLevel({sampler_name}, {projected_coord}, {lod})"
            )

        if is_projected_texture_lod_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleLevel("
                f"{sampler_name}, {projected_coord}, {lod}, {offset})"
            )

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {projected_coord}, {ddx}, {ddy})"
            )

        if is_projected_texture_grad_offset_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(func_name, count_error)
            if not self.texture_sample_offset_supported(texture_type):
                return self.unsupported_texture_projected_call(
                    func_name, texture_sample_offset_capability_error("DirectX")
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleGrad("
                f"{sampler_name}, {projected_coord}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error()
        )

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )
        if is_texture_gather_offset_operation(
            func_name
        ) and not self.texture_gather_offset_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_offset_capability_error()
            )

        offset_args = []
        component_arg = None

        if is_texture_gather_basic_operation(func_name):
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_count_error()
                )
            if extra_args:
                component_arg = extra_args[0]
        elif is_texture_gather_single_offset_operation(func_name):
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_argument_count_error()
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        method_args = [sampler_name, coord] + [
            self.generate_expression(offset_arg) for offset_arg in offset_args
        ]
        method = self.texture_gather_method(component_arg)
        if method is not None:
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if self.literal_int_value(component_arg) is not None:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_component_literal_error()
            )

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression(
            "DirectX", func_name, reason
        )

    def texture_compare_projected_coordinate(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "Texture2D":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            return f"{self.vector_component(coord, 'xy')} / {divisor}"

        if texture_type == "TextureCube":
            if coord_type not in {"vec4", "float4"}:
                return None
            return (
                f"{self.vector_component(coord, 'xyz')} / "
                f"{self.vector_component(coord, 'w')}"
            )

        if texture_type != "Texture2DArray" or coord_type not in {"vec4", "float4"}:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        return f"float3({projected_coord}, {self.vector_component(coord, 'z')})"

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            projected_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord
            )
            if projected_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("DirectX")
                )

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare})"
                )

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                offset = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleCmp("
                    f"{sampler_name}, {projected_coord}, {compare}, {offset})"
                )

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                lod = self.generate_expression(extra_args[1])
                return (
                    f"{texture_name}.SampleCmpLevel("
                    f"{sampler_name}, {projected_coord}, {compare}, {lod})"
                )

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                lod = self.generate_expression(extra_args[1])
                offset = self.generate_expression(extra_args[2])
                return (
                    f"{texture_name}.SampleCmpLevel("
                    f"{sampler_name}, {projected_coord}, {compare}, {lod}, {offset})"
                )

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return (
                    f"{texture_name}.SampleCmpGrad("
                    f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy})"
                )

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("DirectX")
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                offset = self.generate_expression(extra_args[3])
                return (
                    f"{texture_name}.SampleCmpGrad("
                    f"{sampler_name}, {projected_coord}, {compare}, {ddx}, {ddy}, {offset})"
                )

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        if is_texture_compare_basic_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleCmp({sampler_name}, {coord}, {compare}, {offset})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            lod = self.generate_expression(extra_args[1])
            return (
                f"{texture_name}.SampleCmpLevel("
                f"{sampler_name}, {coord}, {compare}, {lod})"
            )

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            lod = self.generate_expression(extra_args[1])
            offset = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleCmpLevel("
                f"{sampler_name}, {coord}, {compare}, {lod}, {offset})"
            )

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy})"
            )

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("DirectX")
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset = self.generate_expression(extra_args[3])
            return (
                f"{texture_name}.SampleCmpGrad("
                f"{sampler_name}, {coord}, {compare}, {ddx}, {ddy}, {offset})"
            )

        return self.unsupported_texture_compare_call(
            func_name, unsupported_texture_compare_operation_error()
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "DirectX", func_name, reason
        )

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args, func_name)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, sampler_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("DirectX")
            )
        offset = self.generate_expression(extra_args[1])
        return f"{texture_name}.GatherCmp({sampler_name}, {coord}, {compare}, {offset})"

    def generate_texture_query_helpers(self):
        if not self.required_texture_query_helpers:
            return ""

        helpers = []
        for helper_name, texture_type in sorted(self.required_texture_query_helpers):
            if helper_name == "textureSize":
                helpers.append(self.generate_texture_size_helper(texture_type))
            elif helper_name == "imageSize":
                helpers.append(self.generate_image_size_helper(texture_type))
            elif helper_name == "textureQueryLevels":
                helpers.append(self.generate_texture_query_levels_helper(texture_type))
            elif helper_name == "textureSamples":
                helpers.append(self.generate_texture_samples_helper(texture_type))

        return "".join(helper for helper in helpers if helper)

    def image_atomic_helper_name(self, operation, texture_type):
        descriptor = self.image_atomic_helper_descriptor(operation, texture_type)
        return descriptor["helper_name"] if descriptor else None

    def image_atomic_helper_return_type(self, texture_type):
        descriptor = self.image_atomic_helper_descriptor("imageAtomicAdd", texture_type)
        return descriptor["return_type"] if descriptor else None

    def image_atomic_helper_coord_type(self, texture_type):
        descriptor = self.image_atomic_helper_descriptor("imageAtomicAdd", texture_type)
        return descriptor["coord_type"] if descriptor else None

    def image_atomic_intrinsic(self, operation):
        return {
            "imageAtomicAdd": "InterlockedAdd",
            "imageAtomicMin": "InterlockedMin",
            "imageAtomicMax": "InterlockedMax",
            "imageAtomicAnd": "InterlockedAnd",
            "imageAtomicOr": "InterlockedOr",
            "imageAtomicXor": "InterlockedXor",
            "imageAtomicExchange": "InterlockedExchange",
            "imageAtomicCompSwap": "InterlockedCompareExchange",
        }.get(operation)

    def image_atomic_zero_value(self, image_type):
        return storage_image_atomic_zero_value(
            self.hlsl_storage_image_component_type(image_type)
        )

    def unsupported_image_atomic_call(self, operation, image_type):
        return unsupported_image_atomic_expression(
            "DirectX",
            operation,
            self.resource_base_type(image_type),
            self.image_atomic_zero_value(image_type),
        )

    def unsupported_multisample_image_atomic_call(self, operation, image_type):
        return unsupported_multisample_image_atomic_expression(
            "DirectX",
            operation,
            self.resource_base_type(image_type),
            self.image_atomic_zero_value(image_type),
        )

    def unsupported_multisample_image_store_call(self, image_type):
        return unsupported_multisample_image_store_expression(
            "DirectX", self.resource_base_type(image_type)
        )

    def image_atomic_helper_descriptor(self, operation, texture_type):
        intrinsic = self.image_atomic_intrinsic(operation)
        if not intrinsic:
            return None

        texture_type = self.resource_base_type(texture_type)
        if "<" not in texture_type or ">" not in texture_type:
            return None

        texture_family = texture_type.split("<", 1)[0]
        component_type = texture_type.split("<", 1)[1].split(">", 1)[0].strip()
        metadata = image_atomic_helper_resource_metadata(
            texture_family,
            {
                "RWTexture1D": "image1D",
                "RWTexture1DArray": "image1DArray",
                "RWTexture2D": "image2D",
                "RWTexture3D": "image3D",
                "RWTexture2DArray": "image2DArray",
                "RWTexture2DMS": "image2DMS",
                "RWTexture2DMSArray": "image2DMSArray",
            },
            {
                "RWTexture1D": "int",
                "RWTexture1DArray": "int2",
                "RWTexture2D": "int2",
                "RWTexture3D": "int3",
                "RWTexture2DArray": "int3",
                "RWTexture2DMS": "int2",
                "RWTexture2DMSArray": "int3",
            },
            sample_families={"RWTexture2DMS", "RWTexture2DMSArray"},
        )
        if metadata is None:
            return None

        descriptor = image_atomic_helper_descriptor_fields(
            operation,
            component_type,
            metadata["suffix_family"],
            metadata["coord_type"],
        )
        if descriptor is None:
            return None

        descriptor["intrinsic"] = intrinsic
        if metadata.get("has_sample"):
            descriptor["has_sample"] = True
        return descriptor

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None
        if operation == "imageAtomicCompSwap":
            if len(args) < 4:
                return None
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource_type(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    operation, image_type
                )
            descriptor = self.image_atomic_helper_descriptor(operation, image_type)
            if descriptor is None:
                return self.unsupported_image_atomic_call(operation, image_type)
            helper_name = descriptor["helper_name"]
            if descriptor.get("has_sample"):
                if len(args) < 5:
                    return None
                sample = self.generate_expression(args[2])
                compare = self.generate_expression(args[3])
                value = self.generate_expression(args[4])
                self.required_image_atomic_helpers.add((operation, image_type))
                return (
                    f"{helper_name}({image_name}, {coord}, {sample}, "
                    f"{compare}, {value})"
                )
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            self.required_image_atomic_helpers.add((operation, image_type))
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"
        if len(args) < 3:
            return None
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        image_type = self.texture_resource_type(args[0])
        if self.is_multisample_storage_image_resource_type(image_type):
            return self.unsupported_multisample_image_atomic_call(operation, image_type)
        descriptor = self.image_atomic_helper_descriptor(operation, image_type)
        if descriptor is None:
            return self.unsupported_image_atomic_call(operation, image_type)
        helper_name = descriptor["helper_name"]
        if descriptor.get("has_sample"):
            if len(args) < 4:
                return None
            sample = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            self.required_image_atomic_helpers.add((operation, image_type))
            return f"{helper_name}({image_name}, {coord}, {sample}, {value})"
        value = self.generate_expression(args[2])
        self.required_image_atomic_helpers.add((operation, image_type))
        return f"{helper_name}({image_name}, {coord}, {value})"

    def generate_image_atomic_helpers(self):
        if not self.required_image_atomic_helpers:
            return ""

        helpers = []
        for operation, texture_type in sorted(self.required_image_atomic_helpers):
            descriptor = self.image_atomic_helper_descriptor(operation, texture_type)
            if descriptor is None:
                continue
            helper_name = descriptor["helper_name"]
            return_type = descriptor["return_type"]
            coord_type = descriptor["coord_type"]
            intrinsic = descriptor["intrinsic"]
            has_sample = descriptor.get("has_sample")
            sample_param = ", int sample" if has_sample else ""
            target = "image[coord, sample]" if has_sample else "image[coord]"
            if operation == "imageAtomicCompSwap":
                helpers.append(
                    f"{return_type} {helper_name}({texture_type} image, {coord_type} coord{sample_param}, {return_type} compareValue, {return_type} value) {{\n"
                    f"    {return_type} original;\n"
                    f"    InterlockedCompareExchange({target}, compareValue, value, original);\n"
                    "    return original;\n"
                    "}\n\n"
                )
                continue
            helpers.append(
                f"{return_type} {helper_name}({texture_type} image, {coord_type} coord{sample_param}, {return_type} value) {{\n"
                f"    {return_type} original;\n"
                f"    {intrinsic}({target}, value, original);\n"
                "    return original;\n"
                "}\n\n"
            )

        return "".join(helpers)

    def generate_texture_size_helper(self, texture_type):
        descriptor = self.texture_size_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureSize", texture_type, descriptor
        )

    def generate_texture_query_dimension_helper(
        self, helper_name, texture_type, descriptor
    ):
        texture_type = self.resource_base_type(texture_type)
        parameter_type = self.directx_resource_declaration_type(texture_type)
        parameters = [f"{parameter_type} tex"]
        if descriptor["function_params"]:
            parameters.append(descriptor["function_params"])
        declarations = "".join(
            f"    uint {dimension};\n" for dimension in descriptor["dimensions"]
        )
        get_dimensions_args = descriptor["get_dimensions_args"]
        get_dimensions_call = ""
        if get_dimensions_args:
            get_dimensions_call = (
                f"    tex.GetDimensions({', '.join(get_dimensions_args)});\n"
            )
        return (
            f"{descriptor['return_type']} {helper_name}({', '.join(parameters)}) {{\n"
            f"{declarations}"
            f"{get_dimensions_call}"
            f"    return {descriptor['return_expr']};\n"
            "}\n\n"
        )

    def texture_size_helper_descriptor(self, texture_type):
        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "lod")
        return resource_query_size_helper_descriptor(descriptor)

    def texture_query_get_dimensions_descriptor(self, texture_type, lod_arg):
        texture_type = self.resource_base_type(texture_type)

        def mip_descriptor(return_type, size_components):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=("levels",),
                function_params="int lod",
                get_dimensions_prefix=(lod_arg,),
            )

        def sample_descriptor(return_type, size_components):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=("samples",),
            )

        if texture_type.startswith("RWTexture2DMSArray<"):
            return sample_descriptor("int3", ("width", "height", "elements"))
        if texture_type.startswith("RWTexture2DMS<"):
            return sample_descriptor("int2", ("width", "height"))
        descriptors = {
            "Texture1D": mip_descriptor("int", ("width",)),
            "Texture1DArray": mip_descriptor("int2", ("width", "elements")),
            "Texture2D": mip_descriptor("int2", ("width", "height")),
            "TextureCube": mip_descriptor("int2", ("width", "height")),
            "Texture2DArray": mip_descriptor("int3", ("width", "height", "elements")),
            "TextureCubeArray": mip_descriptor("int3", ("width", "height", "elements")),
            "Texture3D": mip_descriptor("int3", ("width", "height", "depth")),
            "Texture2DMS<float4>": sample_descriptor("int2", ("width", "height")),
            "Texture2DMSArray<float4>": sample_descriptor(
                "int3", ("width", "height", "elements")
            ),
        }
        return descriptors.get(texture_type)

    def generate_image_size_helper(self, texture_type):
        descriptor = self.image_size_helper_descriptor(texture_type)
        if descriptor is None:
            return ""

        parameter_type = self.directx_resource_declaration_type(texture_type)
        dimensions = descriptor["dimensions"]
        declarations = "".join(f"    uint {dimension};\n" for dimension in dimensions)
        dimension_args = ", ".join(dimensions)
        return (
            f"{descriptor['return_type']} imageSize({parameter_type} image) {{\n"
            f"{declarations}"
            f"    image.GetDimensions({dimension_args});\n"
            f"    return {descriptor['return_expr']};\n"
            "}\n\n"
        )

    def image_size_helper_descriptor(self, texture_type):
        if not self.is_storage_image_resource_type(texture_type):
            return None

        base_type = self.resource_base_type(texture_type)

        def size_descriptor(return_type, size_components, tail_dimensions=()):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=tail_dimensions,
            )

        descriptors = (
            (
                "RWTexture2DMSArray<",
                size_descriptor(
                    "int3",
                    ("width", "height", "elements"),
                    tail_dimensions=("samples",),
                ),
            ),
            (
                "RWTexture2DMS<",
                size_descriptor(
                    "int2", ("width", "height"), tail_dimensions=("samples",)
                ),
            ),
            (
                "RWTexture2DArray<",
                size_descriptor("int3", ("width", "height", "elements")),
            ),
            (
                "RWTexture1DArray<",
                size_descriptor("int2", ("width", "elements")),
            ),
            (
                "RWTexture1D<",
                size_descriptor("int", ("width",)),
            ),
            (
                "RWTexture3D<",
                size_descriptor("int3", ("width", "height", "depth")),
            ),
            (
                "RWTextureCube<",
                size_descriptor("int2", ("width", "height")),
            ),
        )
        for prefix, descriptor in descriptors:
            if base_type.startswith(prefix):
                return resource_query_size_helper_descriptor(
                    descriptor, include_function_fields=False
                )
        return resource_query_size_helper_descriptor(
            size_descriptor("int2", ("width", "height")),
            include_function_fields=False,
        )

    def generate_texture_query_levels_helper(self, texture_type):
        descriptor = self.texture_query_levels_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureQueryLevels", texture_type, descriptor
        )

    def texture_query_levels_helper_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
            return resource_query_scalar_constant_helper_descriptor(
                texture_query_levels_multisample_expression()
            )

        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "0")
        return resource_query_scalar_helper_descriptor(descriptor, "int(levels)")

    def generate_texture_samples_helper(self, texture_type):
        descriptor = self.texture_samples_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureSamples", texture_type, descriptor
        )

    def texture_samples_helper_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type not in {
            "Texture2DMS<float4>",
            "Texture2DMSArray<float4>",
        } and not self.is_multisample_storage_image_resource_type(texture_type):
            return None
        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "lod")
        return resource_query_scalar_helper_descriptor(descriptor, "int(samples)")

    def supported_image_formats(self):
        return supported_image_formats()

    def scalar_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) == 1
        }

    def vector_image_format_components(self):
        return {
            image_format: image_format_vector_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) in {2, 4}
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name"):
            return str(value.name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "readonly",
            "readwrite",
            "restrict",
            "volatile",
            "writeonly",
        }

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def register_space_value(self, value):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return f"space{raw_value}"
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return raw_value
        return None

    def explicit_resource_binding_index(
        self, node, attribute_names=(), register_prefixes=()
    ):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in attribute_names:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], register_prefixes)
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_resource_register_space(self, node):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name == "space" and arguments:
                space = self.register_space_value(arguments[0])
                if space is not None:
                    return space
            if attr_name == "register":
                for argument in arguments[1:]:
                    space = self.register_space_value(argument)
                    if space is not None:
                        return space
        return None

    def resource_register_suffix_for_space(self, register_prefix, binding, space=None):
        register = f"{register_prefix}{binding}"
        if space:
            return f" : register({register}, {space})"
        return f" : register({register})"

    def resource_register_suffix(self, register_prefix, binding, node):
        return self.resource_register_suffix_for_space(
            register_prefix,
            binding,
            self.explicit_resource_register_space(node),
        )

    def resource_memory_qualifier(self, mapped_type, node):
        if not (
            str(mapped_type).startswith(("RWTexture", "RWBuffer"))
            or self.is_hlsl_uav_buffer_type(mapped_type)
        ):
            return ""

        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        if qualifiers & {"coherent", "globallycoherent"} or attributes & {
            "coherent",
            "globallycoherent",
        }:
            return "globallycoherent "
        return ""

    def semantic_from_node(self, node):
        if hasattr(node, "semantic"):
            return node.semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def hlsl_function_return_semantic(self, func):
        semantic = getattr(func, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(func, "attributes"):
            return None
        for attr in func.attributes:
            if self.hlsl_stage_attribute_name(attr):
                continue
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def semantic_from_array_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def semantic_from_struct_member(self, member):
        if isinstance(member, ArrayNode):
            return self.semantic_from_array_node(member)
        return self.semantic_from_node(member)

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)
        lowered_block = self.current_glsl_buffer_block_parameters.get(param_name)
        if lowered_block is not None:
            mapped_type = (
                "ByteAddressBuffer"
                if lowered_block["readonly"]
                else "RWByteAddressBuffer"
            )
            return self.glsl_buffer_block_parameter_type(
                mapped_type, vtype, node, function_hints
            )

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_resource_array_hint_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return (
                    f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"
                )

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_resource_array_hint_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return (
                            f"{mapped_type}[{array_size}]"
                            if array_size
                            else f"{mapped_type}[]"
                        )
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def glsl_buffer_block_parameter_type(
        self, mapped_type, vtype, node=None, function_hints=None
    ):
        function_hints = function_hints or {}
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            array_size = (
                self.expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )
            return f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                _, array_suffix = split_array_type_suffix(type_string)
                if array_suffix == "[]":
                    array_size = function_hints.get(param_name, "")
                    return (
                        f"{mapped_type}[{array_size}]"
                        if array_size
                        else f"{mapped_type}[]"
                    )
                return f"{mapped_type}{array_suffix}"

        return mapped_type

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = (
            explicit_image_format(node, self.attribute_value_to_string)
            if node is not None
            else None
        )
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "RWTexture1D",
            "iimage1D": "RWTexture1D",
            "uimage1D": "RWTexture1D",
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "image1DArray": "RWTexture1DArray",
            "iimage1DArray": "RWTexture1DArray",
            "uimage1DArray": "RWTexture1DArray",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "iimage2DMS": "RWTexture2DMS",
            "uimage2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "iimage2DMSArray": "RWTexture2DMSArray",
            "uimage2DMSArray": "RWTexture2DMSArray",
            "imageCube": "RWTextureCube",
        }
        texture_type = texture_types.get(base_type)
        if component_type and texture_type:
            return f"{texture_type}<{component_type}>"
        return self.map_type(vtype)

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        arguments = getattr(attr, "arguments", []) if attr is not None else []
        if arguments:
            layout = self.attribute_value_to_string(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "glsl_buffer_block")

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        if self.glsl_buffer_block_attribute(node) is None:
            return False
        type_name = self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, global_vars):
        names = set()
        for node in global_vars:
            node_type = glsl_buffer_block_node_type(node)
            if not self.is_glsl_buffer_block_variable(node, node_type):
                continue
            type_name = self.resource_base_type(node_type)
            names.add(str(type_name))
        return names

    def collect_lowered_glsl_buffer_block_parameters(self, parameters):
        return collect_lowered_glsl_buffer_blocks(
            parameters,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="hlsl_type",
            unsupported_type_message=(
                "type is not supported by ByteAddressBuffer lowering"
            ),
        )

    def collect_unsupported_glsl_buffer_block_parameter_names(self, parameters):
        names = set()
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                param_name
                and param_name not in self.current_glsl_buffer_block_parameters
                and self.is_glsl_buffer_block_variable(param, param_type)
            ):
                names.add(param_name)
        return names

    def collect_unsupported_glsl_buffer_block_functions(self, functions):
        unsupported = {}
        omitted_struct_names = set(self.glsl_buffer_block_struct_names)
        if not omitted_struct_names:
            return unsupported

        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            return_type = self.type_name_string(getattr(func, "return_type", "void"))
            return_base = str(self.resource_base_type(return_type))
            if return_base in omitted_struct_names:
                unsupported[func_name] = {
                    "return_type": return_type,
                    "reason": (
                        f"return type references omitted GLSL buffer block struct {return_base}"
                    ),
                }
                continue

            param_list = getattr(func, "parameters", getattr(func, "params", []))
            lowered_params, _, _ = self.collect_lowered_glsl_buffer_block_parameters(
                param_list
            )
            for param in param_list:
                param_name = getattr(param, "name", None)
                param_type = getattr(
                    param,
                    "param_type",
                    getattr(param, "var_type", getattr(param, "vtype", None)),
                )
                param_base = str(self.resource_base_type(param_type))
                if (
                    param_base in omitted_struct_names
                    and param_name not in lowered_params
                ):
                    unsupported[func_name] = {
                        "return_type": return_type,
                        "reason": (
                            f"parameter {param_name} references omitted GLSL "
                            f"buffer block struct {param_base}"
                        ),
                    }
                    break
        return unsupported

    def glsl_buffer_block_lowering_failure_detail(self, type_name, var_name=None):
        if var_name:
            reason = self.current_glsl_buffer_block_parameter_failures.get(var_name)
            if reason:
                return reason
            reason = self.glsl_buffer_block_lowering_failures.get(var_name)
            if reason:
                return reason
        type_name = str(self.resource_base_type(type_name))
        return self.current_glsl_buffer_block_parameter_struct_failures.get(
            type_name
        ) or self.glsl_buffer_block_struct_lowering_failures.get(type_name)

    def glsl_buffer_block_parameter_diagnostics(self, target, parameters, indent=0):
        code = ""
        indent_str = "  " * indent
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                not param_name
                or param_name in self.current_glsl_buffer_block_parameters
                or param_name
                not in self.current_unsupported_glsl_buffer_block_parameters
                or not self.is_glsl_buffer_block_variable(param, param_type)
            ):
                continue
            code += indent_str + self.glsl_buffer_block_diagnostic(
                target, param_type, param_name, param, declaration_kind="parameter"
            )
        return code

    def glsl_buffer_block_member_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        var_name = self.expression_name(object_expr)
        if not var_name:
            return None
        block = self.current_glsl_buffer_block_parameters.get(
            var_name
        ) or self.lowered_glsl_buffer_blocks.get(var_name)
        if block is None:
            return None
        buffer_expr = self.generate_expression(object_expr)
        member_name = getattr(expr, "member", None)
        member = block["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": buffer_expr,
            "member": member_name,
            "readonly": block["readonly"],
            **member,
        }

    def glsl_buffer_block_array_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return None
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        member = self.glsl_buffer_block_member_access(array_expr)
        if member is None or not member.get("is_array"):
            return None
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
        index = self.generate_expression(index_expr)
        offset = byte_offset_expression(member["offset"], index, member["stride"])
        return {**member, "offset_expr": offset}

    def hlsl_byteaddress_load_method(self, components):
        return "Load" if components == 1 else f"Load{components}"

    def hlsl_byteaddress_store_method(self, components):
        return "Store" if components == 1 else f"Store{components}"

    def hlsl_byteaddress_load(self, buffer_name, offset, access):
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                load = (
                    f"{buffer_name}."
                    f"{self.hlsl_byteaddress_load_method(access['matrix_rows'])}"
                    f"({column_offset})"
                )
                columns.append(f"asfloat({load})")
            return f"{access['hlsl_type']}({', '.join(columns)})"

        load = (
            f"{buffer_name}."
            f"{self.hlsl_byteaddress_load_method(access['components'])}({offset})"
        )
        if access["component_type"] == "float":
            return f"asfloat({load})"
        if access["component_type"] == "int":
            return f"asint({load})"
        return load

    def hlsl_byteaddress_store_value(self, value, access):
        if access.get("layout_type") != access.get("type"):
            if access["component_type"] == "int":
                return f"asuint(int({value}))"
            if access["component_type"] == "uint":
                if access["components"] == 1:
                    return f"uint({value})"
                return f"uint{access['components']}({value})"
        if access["component_type"] in {"float", "int"}:
            return f"asuint({value})"
        return value

    def next_hlsl_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.hlsl_temp_variable_index}"
        self.hlsl_temp_variable_index += 1
        return name

    def hlsl_indexable_expression(self, expression):
        expression = str(expression)
        if expression.isidentifier():
            return expression
        if all(part.isidentifier() for part in expression.split(".")):
            return expression
        return f"({expression})"

    def hlsl_byteaddress_matrix_store(self, buffer_name, offset, value, access):
        value_expr = self.hlsl_indexable_expression(value)
        store_method = self.hlsl_byteaddress_store_method(access["matrix_rows"])
        lines = []
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            lines.append(
                f"{buffer_name}.{store_method}"
                f"({column_offset}, asuint({value_expr}[{column}]))"
            )
        return "\n".join(lines)

    def hlsl_byteaddress_matrix_compound_store(
        self, buffer_name, offset, value, op, access
    ):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            return (
                "/* unsupported HLSL GLSL buffer block matrix compound store: "
                "requires explicit matrix operation lowering */"
            )

        temp_name = self.next_hlsl_temp_variable("matrix_store")
        current = self.hlsl_byteaddress_load(buffer_name, offset, access)
        temp = f"{access['hlsl_type']} {temp_name} = ({current} {binary_op} {value})"
        stores = self.hlsl_byteaddress_matrix_store(
            buffer_name, offset, temp_name, access
        )
        return f"{temp}\n{stores}"

    def hlsl_byteaddress_compound_store_diagnostic(self, op, access):
        return (
            "/* unsupported HLSL GLSL buffer block compound store: "
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members */"
        )

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("runtime_array"):
            return None
        return self.hlsl_byteaddress_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        return self.hlsl_byteaddress_load(
            access["buffer"], access["offset_expr"], access
        )

    def generate_glsl_buffer_block_store(self, target, value, op):
        access = self.glsl_buffer_block_array_access(target)
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("runtime_array"):
                return None
            offset = access["offset"]
        else:
            offset = access["offset_expr"]

        if access.get("readonly"):
            return (
                "/* unsupported HLSL GLSL buffer block store: "
                "readonly ByteAddressBuffer cannot be written */"
            )

        rhs = self.generate_expression_with_expected(value, access["type"])
        if access.get("matrix_columns"):
            if op != "=":
                return self.hlsl_byteaddress_matrix_compound_store(
                    access["buffer"], offset, rhs, op, access
                )
            return self.hlsl_byteaddress_matrix_store(
                access["buffer"], offset, rhs, access
            )

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return self.hlsl_byteaddress_compound_store_diagnostic(op, access)
            current = self.hlsl_byteaddress_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        store_value = self.hlsl_byteaddress_store_value(rhs, access)
        store_method = self.hlsl_byteaddress_store_method(access["components"])
        return f"{access['buffer']}.{store_method}({offset}, {store_value})"

    def glsl_buffer_block_diagnostic(
        self, target, type_name, var_name=None, node=None, declaration_kind=None
    ):
        declaration = str(self.resource_base_type(type_name))
        if declaration_kind:
            declaration = f"{declaration_kind} {declaration}"
        if var_name:
            declaration += f" {var_name}"
        details = ""
        if node is not None:
            layout = self.glsl_buffer_block_layout(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer", "texture", "uav"}, ("b", "t", "u")
            )
            details = f" ({layout}"
            if binding is not None:
                details += f", binding = {binding}"
            details += ")"
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(
            type_name, var_name
        )
        reason = f"; {failure_detail}" if failure_detail else ""
        return (
            f"// unsupported {target} GLSL buffer block {declaration}{details}: "
            "mixed metadata/runtime-array layout requires ByteAddressBuffer "
            f"offset lowering{reason}\n"
        )

    def unsupported_glsl_buffer_block_struct_placeholder(self, target, type_name):
        type_name = str(self.resource_base_type(type_name))
        return (
            f"// unsupported {target} GLSL buffer block struct {type_name} "
            "omitted: no target-side fallback declaration emitted\n"
        )

    def unsupported_glsl_buffer_block_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        return (
            f"// unsupported {target} GLSL buffer block variable {declaration} "
            "omitted: no target-side fallback declaration emitted\n"
        )

    def unsupported_glsl_buffer_block_function_placeholder(
        self, target, func_name, info
    ):
        reason = info.get("reason", "signature references omitted GLSL buffer block")
        return (
            f"// unsupported {target} GLSL buffer block function {func_name} "
            f"omitted: {reason}\n"
        )

    def unsupported_glsl_buffer_block_function_call(self, func_name):
        if (
            not func_name
            or func_name not in self.unsupported_glsl_buffer_block_functions
        ):
            return None
        info = self.unsupported_glsl_buffer_block_functions[func_name]
        return_type = info.get("return_type") or self.current_expression_expected_type
        diagnostic = (
            f"unsupported HLSL GLSL buffer block function call {func_name}: "
            "target function omitted"
        )
        if self.map_type(return_type) == "void":
            return f"/* {diagnostic} */"
        fallback = self.diagnostic_zero_value_for_type(return_type)
        return f"{fallback} /* {diagnostic} */"

    def is_unsupported_glsl_buffer_block_struct_type(self, vtype):
        return str(self.resource_base_type(vtype)) in (
            self.unsupported_glsl_buffer_block_struct_names
        )

    def unsupported_glsl_buffer_block_local_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        return (
            f"/* unsupported {target} GLSL buffer block local variable {declaration} "
            "omitted: no target-side fallback declaration emitted */"
        )

    def is_unsupported_glsl_buffer_block_name(self, name):
        return name in self.unsupported_glsl_buffer_block_variables or (
            name in self.current_unsupported_glsl_buffer_block_parameters
            or name in self.current_unsupported_glsl_buffer_block_local_variables
        )

    def unsupported_glsl_buffer_block_name_type(self, name):
        if not self.is_unsupported_glsl_buffer_block_name(name):
            return None
        return self.unsupported_glsl_buffer_block_variable_types.get(
            name
        ) or self.local_variable_types.get(name)

    def unsupported_glsl_buffer_block_expression_type(self, expr):
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.unsupported_glsl_buffer_block_expression_type(
                object_expr
            )
            member_name = str(getattr(expr, "member", ""))
            if object_type and member_name:
                return self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member_name)
            return None

        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.unsupported_glsl_buffer_block_expression_type(array_expr)
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type

        expr_name = self.expression_name(expr)
        if expr_name:
            block_type = self.unsupported_glsl_buffer_block_name_type(expr_name)
            if block_type:
                return block_type
        return None

    def unsupported_glsl_buffer_block_access_name(self, expr):
        expr_name = self.expression_name(expr)
        if expr_name and self.is_unsupported_glsl_buffer_block_name(expr_name):
            return expr_name
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.unsupported_glsl_buffer_block_access_name(object_expr)
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.unsupported_glsl_buffer_block_access_name(array_expr)
        return None

    def diagnostic_zero_value_for_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type in {"float", "double", "int"}:
            return "0"
        if (
            mapped_type
            and mapped_type[0].isalpha()
            and any(char.isdigit() for char in mapped_type)
        ):
            return f"{mapped_type}(0)"
        return "0"

    def unsupported_glsl_buffer_block_access_value(self, expr):
        name = self.unsupported_glsl_buffer_block_access_name(expr)
        if not name:
            return None
        value_type = (
            self.expression_result_type(expr) or self.current_expression_expected_type
        )
        fallback = self.diagnostic_zero_value_for_type(value_type)
        return (
            f"{fallback} /* unsupported HLSL GLSL buffer block access {name}: "
            "no target-side fallback declaration emitted */"
        )

    def unsupported_glsl_buffer_block_assignment_diagnostic(self, target):
        name = self.unsupported_glsl_buffer_block_access_name(target)
        if not name:
            return None
        return (
            "/* unsupported HLSL GLSL buffer block assignment "
            f"{name}: no target-side fallback declaration emitted */"
        )

    def hlsl_resource_type_name(self, vtype):
        base_type = self.resource_base_type(vtype)
        return str(base_type).split("<", 1)[0]

    def is_hlsl_readonly_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) in {
            "Buffer",
            "StructuredBuffer",
            "ByteAddressBuffer",
        }

    def is_hlsl_uav_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) in {
            "RWBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "RWByteAddressBuffer",
        }

    def generate_texture_call(self, func_name, args):
        if not func_name:
            return None

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_image_multisample_arguments(func_name, args)
        self.validate_image_atomic_format_argument(func_name, args)
        self.validate_integer_coordinate_argument(func_name, args)
        self.validate_coordinate_dimension_argument(func_name, args)
        self.validate_query_lod_coordinate_argument(func_name, args)
        self.validate_compare_argument(func_name, args)
        self.validate_lod_argument(func_name, args)
        self.validate_bias_argument(func_name, args)
        self.validate_sample_index_argument(func_name, args)
        self.validate_mip_level_argument(func_name, args)
        self.validate_gradient_dimension_arguments(func_name, args)
        self.validate_offset_dimension_argument(func_name, args)
        self.validate_gather_component_argument(func_name, args)

        if is_resource_size_query_operation(func_name) and args:
            lod_arg = args[1] if len(args) > 1 else None
            return self.texture_query_size_expression(args[0], lod_arg)

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(args[0])

        if is_texture_query_lod_operation(func_name) and len(args) >= 2:
            parts = self.texture_call_parts(args, func_name)
            if parts is None:
                return None
            texture_name, sampler_name, coord, _ = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_resource_type(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            if not self.is_explicit_sampler_argument(args):
                sampler_name = self.generate_implicit_query_lod_sampler_argument(
                    args[0]
                )
            coord = self.texture_query_lod_coordinate(texture_type, coord)
            return (
                f"float2({texture_name}.CalculateLevelOfDetailUnclamped({sampler_name}, {coord}), "
                f"{texture_name}.CalculateLevelOfDetail({sampler_name}, {coord}))"
            )

        if is_image_atomic_operation(func_name):
            return self.image_atomic_expression(func_name, args)

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 3 and self.is_multisample_storage_image_resource_type(
                texture_type
            ):
                sample = self.generate_expression(args[2])
                load_expr = f"{image_name}.Load({coord}, {sample})"
            else:
                load_expr = f"{image_name}[{coord}]"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(texture_type, image_format)
            if self.four_component_image_store_constructor(
                texture_type
            ) and self.is_scalar_value_type(self.current_expression_expected_type):
                return f"{load_expr}.x"
            if self.two_component_image_store_constructor(
                texture_type
            ) and self.is_scalar_value_type(self.current_expression_expected_type):
                return f"{load_expr}.x"
            return load_expr

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_resource_type(
                texture_type
            ):
                value_arg = args[3]
            else:
                value_arg = args[2]
                store_target = f"{image_name}[{coord}]"
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(texture_type, image_format, value_arg)
            if self.is_multisample_storage_image_resource_type(texture_type):
                return self.unsupported_multisample_image_store_call(texture_type)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    texture_type, image_format, value_arg
                ),
            )
            four_component_constructor = self.four_component_image_store_constructor(
                texture_type
            )
            if four_component_constructor and self.is_scalar_value_type(
                self.expression_result_type(value_arg)
            ):
                value = f"{four_component_constructor}({value})"
            two_component_constructor = self.two_component_image_store_constructor(
                texture_type
            )
            if two_component_constructor and self.is_scalar_value_type(
                self.expression_result_type(value_arg)
            ):
                constructor, zero_value = two_component_constructor
                value = f"{constructor}({value}, {zero_value})"
            return f"{store_target} = {value}"

        if len(args) < 2:
            return None

        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(func_name, args)

        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(func_name, args)

        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(func_name, args)

        if is_texture_sample_offset_operation(func_name):
            return self.generate_texture_sample_offset_call(func_name, args)

        if is_projected_texture_operation(func_name):
            return self.generate_texture_projected_call(func_name, args)

        if is_texture_sample_operation(func_name):
            parts = self.texture_call_parts(args, func_name)
            if parts is None:
                return None
            texture_name, sampler_name, coord, extra_args = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_call(
                    func_name, texture_type
                )
            if is_texture_sample_basic_operation(func_name) and len(extra_args) == 1:
                bias = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleBias({sampler_name}, {coord}, {bias})"
            if is_texture_sample_basic_operation(func_name):
                method = "Sample"
            elif is_texture_sample_lod_operation(func_name):
                method = "SampleLevel"
            elif is_texture_sample_grad_operation(func_name):
                method = "SampleGrad"
            else:
                return None
            mapped_args = [coord] + [
                self.generate_expression(arg) for arg in extra_args
            ]
            return f"{texture_name}.{method}({sampler_name}, {', '.join(mapped_args)})"

        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
                return f"{texture_name}.Load({coord}, {lod})"
            load_coord_type = (
                "int4" if texture_type in {"Texture2DArray", "Texture3D"} else "int3"
            )
            return f"{texture_name}.Load({load_coord_type}({coord}, {lod}))"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            texture_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            lod = self.generate_expression(args[2])
            offset = self.generate_expression(args[3])
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if texture_type in {"Texture2DMS<float4>", "Texture2DMSArray<float4>"}:
                return unsupported_multisample_texel_fetch_offset_expression("DirectX")
            if texture_type == "Texture2DArray":
                coord_xy = self.vector_component(coord, "xy")
                layer = self.vector_component(coord, "z")
                return (
                    f"{texture_name}.Load("
                    f"int4(({coord_xy} + {offset}), {layer}, {lod}))"
                )
            if texture_type == "Texture3D":
                return f"{texture_name}.Load(int4(({coord} + {offset}), {lod}))"
            if texture_type == "Texture1D":
                return f"{texture_name}.Load(int2(({coord} + {offset}), {lod}))"
            if texture_type == "Texture1DArray":
                coord_x = self.vector_component(coord, "x")
                layer = self.vector_component(coord, "y")
                return (
                    f"{texture_name}.Load(int3(({coord_x} + {offset}), {layer}, {lod}))"
                )
            return f"{texture_name}.Load(int3(({coord} + {offset}), {lod}))"

        return None

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "value"):
            value = type_node.value
            return str(value).lower() if isinstance(value, bool) else str(value)
        generic_args = getattr(type_node, "generic_args", [])
        if hasattr(type_node, "name") and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg) for arg in generic_args
            )
            return f"{type_node.name}<{args}>"
        if hasattr(type_node, "name"):
            return type_node.name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"float{type_node.rows}x{type_node.rows}"
            return f"float{type_node.cols}x{type_node.rows}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return f"{element_type}[{type_node.size}]"
                    else:
                        size_str = self.expression_to_string(type_node.size)
                        return f"{element_type}[{size_str}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type == "uint":
                    return f"uint{size}"
                elif element_type == "bool":
                    return f"bool{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def map_type(self, vtype):
        """Map types to DirectX equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return vtype_str

        if "<" in vtype_str and vtype_str.endswith(">"):
            base_type, generic_args = vtype_str.split("<", 1)
            generic_args = generic_args[:-1].strip()
            if "," not in generic_args:
                return f"{base_type}<{self.map_type(generic_args)}>"

        return self.type_mapping.get(vtype_str, vtype_str)

    def map_struct_member_type(self, struct_name, member_name, vtype):
        mapped_type = self.map_type(vtype)
        if (
            member_name
            and self.is_sampler_type(vtype)
            and self.struct_member_uses_comparison_sampler(struct_name, member_name)
        ):
            return mapped_type.replace("SamplerState", "SamplerComparisonState", 1)
        return mapped_type

    def hlsl_tess_factor_member_declaration(self, member_type, member_name, semantic):
        mapped_semantic = self.semantic_map.get(semantic, semantic)
        if str(mapped_semantic).lower() not in {
            "sv_tessfactor",
            "sv_insidetessfactor",
        }:
            return None

        base_type, array_suffix = split_array_type_suffix(str(member_type))
        if array_suffix:
            return format_c_style_array_declaration(member_type, member_name)

        vector_base = None
        vector_size = None
        for scalar_base in ("float", "half", "double", "min16float"):
            if base_type.startswith(scalar_base):
                suffix = base_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    vector_base = scalar_base
                    vector_size = suffix
                    break

        if vector_base is None:
            return f"{member_type} {member_name}"
        return f"{vector_base} {member_name}[{vector_size}]"

    def struct_member_uses_comparison_sampler(self, struct_name, member_name):
        return (
            struct_name,
            member_name,
        ) in self.comparison_sampler_struct_members

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "ASSIGN_OR": "|=",
            "ASSIGN_XOR": "^=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to an HLSL semantic suffix."""
        if semantic:
            return f": {self.semantic_map.get(semantic, semantic)}"
        else:
            return ""  # Handle None by returning an empty string
