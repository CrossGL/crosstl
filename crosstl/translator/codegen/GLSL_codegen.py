"""CrossGL-to-GLSL code generator."""

import re
from copy import deepcopy
from types import SimpleNamespace

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerType,
    PreprocessorNode,
    PrimitiveType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
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
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    get_array_size_from_node,
    parse_array_type,
    split_array_type_suffix,
)
from .constant_ordering import partition_constants_by_struct_dependency
from .enum_utils import (
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_struct_fields,
    enum_value_expression,
    generate_enum_constants,
    generate_enum_constructor_call,
    generate_enum_constructor_expression,
    generate_enum_constructor_functions,
    generate_enum_structs,
    generate_generic_enum_constants,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_structs,
    generic_enum_specialized_fields,
    generic_enum_specialized_type_name,
    infer_enum_constructor_type,
    sanitize_type_name,
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
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generate_generic_structs,
    generate_struct_constructor_expression,
    generic_struct_specialized_fields,
    generic_struct_specialized_type_name,
    infer_struct_constructor_type,
)
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    default_storage_image_channel_count,
    floating_coordinate_dimension_from_type_name,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_explicit_format_component_kind,
    image_atomic_format_error,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
)
from .image_access_contracts import (
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
)
from .image_access_contracts import (
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_format_component_kind,
    image_format_or_default_channel_count,
    image_format_result_type,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_load_result_shape_mismatch,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    image_store_value_shape_mismatch,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_glsl_storage_image_type,
    is_image_atomic_operation,
    is_image_format_attribute,
    is_image_resource_operation,
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
    is_resource_access_attribute,
    is_resource_samples_query_operation,
    is_resource_size_query_operation,
    is_scalar_image_format,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_operation,
    is_texture_gather_single_offset_operation,
    is_texture_query_levels_operation,
    is_texture_query_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_samples_query_operation,
    is_texture_sampling_operation,
    is_two_component_image_format,
    normalized_image_access,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    projected_texture_extra_argument_count_error,
    record_explicit_image_metadata,
    requires_integer_coordinate,
    should_validate_image_load_result_shape,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_load_component_suffix,
    storage_image_store_constructors,
    storage_image_store_value_expression,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    supported_image_formats,
)
from .image_access_contracts import (
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
)
from .image_access_contracts import (
    texture_compare_argument_error,
    texture_compare_coordinate_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_compare_projected_lod_array_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_levels_multisample_expression,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_sample_offset_capability_error,
    texture_samples_query_expression,
)
from .image_access_contracts import (
    unsupported_image_atomic_expression as image_atomic_diagnostic_expression,
)
from .image_access_contracts import (
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_projected_texture_operation_error,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_compare_operation_error,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_gather_compare_call_expression,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
    validate_texture_operation_arity,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .resource_arrays import collect_resource_array_size_hints
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


class GLSLCodeGen:
    """Emit GLSL source from the shared CrossGL translator AST."""

    OPENGL_DESCRIPTOR_SET_BINDING_STRIDE = 1024
    METAL_ONLY_SYSTEM_INCLUDES = frozenset(
        {
            "metal_math",
            "metal_math.h",
            "metal_stdlib",
            "metal_stdlib.h",
        }
    )
    MESH_STAGE_NAMES = {"mesh", "task", "amplification", "object"}
    UNSIGNED_VECTOR_COMPUTE_BUILTINS = {"gl_GlobalInvocationID"}
    GLSL_STAGE_GUARD_MACROS = {
        "vertex": "GL_VERTEX_SHADER",
        "fragment": "GL_FRAGMENT_SHADER",
        "compute": "GL_COMPUTE_SHADER",
        "geometry": "GL_GEOMETRY_SHADER",
        "tessellation_control": "GL_TESS_CONTROL_SHADER",
        "tessellation_evaluation": "GL_TESS_EVALUATION_SHADER",
        "mesh": "GL_MESH_SHADER_EXT",
        "task": "GL_TASK_SHADER_EXT",
        "amplification": "GL_TASK_SHADER_EXT",
        "object": "GL_TASK_SHADER_EXT",
        "ray_generation": "GL_RAYGEN_SHADER_EXT",
        "ray_intersection": "GL_INTERSECTION_SHADER_EXT",
        "ray_any_hit": "GL_ANY_HIT_SHADER_EXT",
        "ray_closest_hit": "GL_CLOSEST_HIT_SHADER_EXT",
        "ray_miss": "GL_MISS_SHADER_EXT",
        "ray_callable": "GL_CALLABLE_SHADER_EXT",
    }
    TEXTURE_SIZE_NO_LOD_SUFFIXES = ("2DRect", "Buffer")
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
        "RWTexture1D": "image1D",
        "RWTexture1DArray": "image1DArray",
        "RWTexture2D": "image2D",
        "RWTexture2DArray": "image2DArray",
        "RWTexture2DMS": "image2DMS",
        "RWTexture2DMSArray": "image2DMSArray",
        "RWTexture3D": "image3D",
        "RWTextureCube": "imageCube",
        "RWTextureCubeArray": "imageCubeArray",
    }
    sampler_state_type_aliases = {
        "SamplerState": "sampler",
        "SamplerComparisonState": "comparison_sampler",
    }
    RAY_STAGE_NAMES = {
        "ray_generation",
        "ray_intersection",
        "ray_any_hit",
        "ray_closest_hit",
        "ray_miss",
        "ray_callable",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
    }
    RAY_QUERY_METHOD_MAP = {
        "Initialize": ("rayQueryInitializeEXT", None),
        "TraceRayInline": ("rayQueryInitializeEXT", None),
        "Proceed": ("rayQueryProceedEXT", None),
        "Abort": ("rayQueryTerminateEXT", None),
        "Terminate": ("rayQueryTerminateEXT", None),
        "GenerateIntersection": ("rayQueryGenerateIntersectionEXT", None),
        "ConfirmIntersection": ("rayQueryConfirmIntersectionEXT", None),
        "RayTMin": ("rayQueryGetRayTMinEXT", None),
        "RayFlags": ("rayQueryGetRayFlagsEXT", None),
        "WorldRayOrigin": ("rayQueryGetWorldRayOriginEXT", None),
        "WorldRayDirection": ("rayQueryGetWorldRayDirectionEXT", None),
        "CandidateType": ("rayQueryGetIntersectionTypeEXT", "false"),
        "CommittedType": ("rayQueryGetIntersectionTypeEXT", "true"),
        "CandidatePrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "false",
        ),
        "CommittedPrimitiveIndex": (
            "rayQueryGetIntersectionPrimitiveIndexEXT",
            "true",
        ),
        "CandidateInstanceID": ("rayQueryGetIntersectionInstanceIdEXT", "false"),
        "CommittedInstanceID": ("rayQueryGetIntersectionInstanceIdEXT", "true"),
        "CandidateGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "false",
        ),
        "CommittedGeometryIndex": (
            "rayQueryGetIntersectionGeometryIndexEXT",
            "true",
        ),
        "CandidateInstanceCustomIndex": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "false",
        ),
        "CommittedInstanceCustomIndex": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT",
            "true",
        ),
        "CandidateInstanceShaderBindingTableRecordOffset": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "false",
        ),
        "CommittedInstanceShaderBindingTableRecordOffset": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",
            "true",
        ),
        "CandidateObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "false",
        ),
        "CandidateObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "false",
        ),
        "CommittedObjectRayOrigin": (
            "rayQueryGetIntersectionObjectRayOriginEXT",
            "true",
        ),
        "CommittedObjectRayDirection": (
            "rayQueryGetIntersectionObjectRayDirectionEXT",
            "true",
        ),
        "CandidateRayT": ("rayQueryGetIntersectionTEXT", "false"),
        "CommittedRayT": ("rayQueryGetIntersectionTEXT", "true"),
        "CandidateTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "false",
        ),
        "CommittedTriangleBarycentrics": (
            "rayQueryGetIntersectionBarycentricsEXT",
            "true",
        ),
        "CandidateTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "false",
        ),
        "CommittedTriangleFrontFace": (
            "rayQueryGetIntersectionFrontFaceEXT",
            "true",
        ),
        "CandidateTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "false",
        ),
        "CommittedTriangleVertexPositions": (
            "rayQueryGetIntersectionTriangleVertexPositionsEXT",
            "true",
        ),
        "CandidateAABBOpaque": ("rayQueryGetIntersectionCandidateAABBOpaqueEXT", None),
        "CandidateObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "false",
        ),
        "CommittedObjectToWorld3x4": (
            "rayQueryGetIntersectionObjectToWorldEXT",
            "true",
        ),
        "CandidateWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
        "CandidateWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "false",
        ),
        "CommittedWorldToObject3x4": (
            "rayQueryGetIntersectionWorldToObjectEXT",
            "true",
        ),
    }
    RAY_QUERY_POSITION_FETCH_METHODS = {
        "CandidateTriangleVertexPositions",
        "CommittedTriangleVertexPositions",
    }
    RAY_QUERY_METHOD_ARITIES = {
        "Initialize": (4, 7),
        "TraceRayInline": (4, 7),
        "Proceed": (0,),
        "Abort": (0,),
        "Terminate": (0,),
        "GenerateIntersection": (1,),
        "ConfirmIntersection": (0,),
        "RayTMin": (0,),
        "RayFlags": (0,),
        "WorldRayOrigin": (0,),
        "WorldRayDirection": (0,),
        "CandidateType": (0,),
        "CommittedType": (0,),
        "CandidatePrimitiveIndex": (0,),
        "CommittedPrimitiveIndex": (0,),
        "CandidateInstanceID": (0,),
        "CommittedInstanceID": (0,),
        "CandidateGeometryIndex": (0,),
        "CommittedGeometryIndex": (0,),
        "CandidateInstanceCustomIndex": (0,),
        "CommittedInstanceCustomIndex": (0,),
        "CandidateInstanceShaderBindingTableRecordOffset": (0,),
        "CommittedInstanceShaderBindingTableRecordOffset": (0,),
        "CandidateObjectRayOrigin": (0,),
        "CandidateObjectRayDirection": (0,),
        "CommittedObjectRayOrigin": (0,),
        "CommittedObjectRayDirection": (0,),
        "CandidateRayT": (0,),
        "CommittedRayT": (0,),
        "CandidateTriangleBarycentrics": (0,),
        "CommittedTriangleBarycentrics": (0,),
        "CandidateTriangleFrontFace": (0,),
        "CommittedTriangleFrontFace": (0,),
        "CandidateTriangleVertexPositions": (1,),
        "CommittedTriangleVertexPositions": (1,),
        "CandidateAABBOpaque": (0,),
        "CandidateObjectToWorld": (0,),
        "CommittedObjectToWorld": (0,),
        "CandidateObjectToWorld3x4": (0,),
        "CommittedObjectToWorld3x4": (0,),
        "CandidateWorldToObject": (0,),
        "CommittedWorldToObject": (0,),
        "CandidateWorldToObject3x4": (0,),
        "CommittedWorldToObject3x4": (0,),
    }
    RAY_QUERY_RAY_DESC_FIELDS = (
        ("Origin", "origin", "rayOrigin", "RayOrigin"),
        ("TMin", "tMin", "Tmin", "tmin"),
        ("Direction", "direction", "rayDirection", "RayDirection"),
        ("TMax", "tMax", "Tmax", "tmax"),
    )
    GLSL_LAYOUT_ATTRIBUTE_NAMES = (
        "location",
        "component",
        "index",
        "stream",
        "xfb_buffer",
        "xfb_offset",
        "xfb_stride",
    )
    GLSL_BUFFER_BLOCK_MEMORY_LAYOUT_NAMES = {
        "packed",
        "scalar",
        "shared",
        "std140",
        "std430",
    }
    GLSL_BLEND_SUPPORT_LAYOUT_NAMES = {
        "blend_support_multiply",
        "blend_support_screen",
        "blend_support_overlay",
        "blend_support_darken",
        "blend_support_lighten",
        "blend_support_colordodge",
        "blend_support_colorburn",
        "blend_support_hardlight",
        "blend_support_softlight",
        "blend_support_difference",
        "blend_support_exclusion",
        "blend_support_hsl_hue",
        "blend_support_hsl_saturation",
        "blend_support_hsl_color",
        "blend_support_hsl_luminosity",
        "blend_support_all_equations",
    }
    GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES = {
        "depth_any",
        "depth_greater",
        "depth_less",
        "depth_unchanged",
        *GLSL_BLEND_SUPPORT_LAYOUT_NAMES,
    }
    GLSL_VARIABLE_QUALIFIER_ATTRIBUTE_NAMES = {
        "invariant",
        "precise",
        "patch",
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
        "in",
        "out",
        "inout",
        "lowp",
        "mediump",
        "highp",
    }
    GLSL_STORAGE_QUALIFIER_ATTRIBUTE_NAMES = {
        "shared",
        "groupshared",
        "workgroup",
        "taskpayloadshared",
        "taskpayloadsharedext",
        "task_payload_shared",
        "raypayload",
        "raypayloadext",
        "raypayloadin",
        "raypayloadinext",
        "hitattribute",
        "hitattributeext",
        "callabledata",
        "callabledataext",
        "callabledatain",
        "callabledatainext",
        "tileimage",
        "tileimageext",
    }
    GLSL_PRECISION_QUALIFIERS = {"lowp", "mediump", "highp"}
    GLSL_PARAMETER_QUALIFIERS = {"out", "inout"}
    GLSL_RESERVED_IDENTIFIERS = {"active", "input", "output"}
    GLSL_BFLOAT16_ALIASES = {"bfloat", "bfloat16", "bfloat16_t"}
    GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS = {
        "clamp",
        "floatBitsToInt",
        "floatBitsToUint",
        "intBitsToFloat",
        "mix",
        "uintBitsToFloat",
    }
    GLSL_BITCAST_FUNCTIONS = {"asfloat", "asint", "asuint"}
    GLSL_INTERPOLATION_FUNCTIONS = {
        "interpolateAtCentroid": 1,
        "interpolateAtSample": 2,
        "interpolateAtOffset": 2,
    }
    GLSL_DERIVATIVE_FUNCTIONS = {
        "dFdx": 1,
        "dFdy": 1,
        "fwidth": 1,
        "dFdxFine": 1,
        "dFdxCoarse": 1,
        "dFdyFine": 1,
        "dFdyCoarse": 1,
        "fwidthFine": 1,
        "fwidthCoarse": 1,
    }
    GLSL_DERIVATIVE_FUNCTION_ALIASES = {
        "ddx": "dFdx",
        "ddy": "dFdy",
        "fwidth": "fwidth",
        "ddx_fine": "dFdxFine",
        "ddx_coarse": "dFdxCoarse",
        "ddy_fine": "dFdyFine",
        "ddy_coarse": "dFdyCoarse",
        "fwidth_fine": "fwidthFine",
        "fwidth_coarse": "fwidthCoarse",
    }
    GLSL_WAVE_INTRINSIC_ARITIES = {
        "WaveGetLaneCount": 0,
        "WaveGetLaneIndex": 0,
        "WaveIsFirstLane": 0,
        "WaveActiveSum": 1,
        "WaveActiveProduct": 1,
        "WaveActiveBitAnd": 1,
        "WaveActiveBitOr": 1,
        "WaveActiveBitXor": 1,
        "WaveActiveMin": 1,
        "WaveActiveMax": 1,
        "WaveActiveAllTrue": 1,
        "WaveActiveAnyTrue": 1,
        "WaveActiveAllEqual": 1,
        "WaveActiveCountBits": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "WavePrefixCountBits": 1,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixCountBits": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
    }
    GLSL_WAVE_DIRECT_MAPPINGS = {
        "WaveActiveSum": "subgroupAdd",
        "WaveActiveProduct": "subgroupMul",
        "WaveActiveBitAnd": "subgroupAnd",
        "WaveActiveBitOr": "subgroupOr",
        "WaveActiveBitXor": "subgroupXor",
        "WaveActiveMin": "subgroupMin",
        "WaveActiveMax": "subgroupMax",
        "WaveActiveAllTrue": "subgroupAll",
        "WaveActiveAnyTrue": "subgroupAny",
        "WaveActiveAllEqual": "subgroupAllEqual",
        "WaveActiveBallot": "subgroupBallot",
        "WaveReadLaneAt": "subgroupShuffle",
        "WaveReadLaneFirst": "subgroupBroadcastFirst",
        "WavePrefixSum": "subgroupExclusiveAdd",
        "WavePrefixProduct": "subgroupExclusiveMul",
        "QuadReadAcrossX": "subgroupQuadSwapHorizontal",
        "QuadReadAcrossY": "subgroupQuadSwapVertical",
        "QuadReadAcrossDiagonal": "subgroupQuadSwapDiagonal",
        "QuadReadLaneAt": "subgroupQuadBroadcast",
    }
    GLSL_VECTOR_RELATIONAL_FUNCTIONS = {
        "<": "lessThan",
        "<=": "lessThanEqual",
        ">": "greaterThan",
        ">=": "greaterThanEqual",
    }
    GLSL_WAVE_EXTENSION_REQUIREMENTS = {
        "WaveGetLaneCount": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveGetLaneIndex": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveIsFirstLane": "#extension GL_KHR_shader_subgroup_basic : require",
        "WaveActiveAllTrue": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveAnyTrue": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveAllEqual": "#extension GL_KHR_shader_subgroup_vote : require",
        "WaveActiveSum": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveProduct": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitAnd": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitOr": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBitXor": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveMin": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveMax": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WavePrefixSum": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WavePrefixProduct": "#extension GL_KHR_shader_subgroup_arithmetic : require",
        "WaveActiveBallot": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveActiveCountBits": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WavePrefixCountBits": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveReadLaneAt": "#extension GL_KHR_shader_subgroup_shuffle : require",
        "WaveReadLaneFirst": "#extension GL_KHR_shader_subgroup_ballot : require",
        "QuadReadAcrossX": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadAcrossY": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadAcrossDiagonal": "#extension GL_KHR_shader_subgroup_quad : require",
        "QuadReadLaneAt": "#extension GL_KHR_shader_subgroup_quad : require",
        "WaveMatch": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixSum": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixCountBits": (
            "#extension GL_KHR_shader_subgroup_ballot : require"
        ),
        "WaveMultiPrefixProduct": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitAnd": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitOr": "#extension GL_KHR_shader_subgroup_ballot : require",
        "WaveMultiPrefixBitXor": "#extension GL_KHR_shader_subgroup_ballot : require",
    }
    GLSL_WAVE_ADDITIONAL_EXTENSION_REQUIREMENTS = {
        "WaveMatch",
        "WaveMultiPrefixSum",
        "WaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    GLSL_WAVE_HELPER_VALUE_TYPES = (
        "bool",
        "bvec2",
        "bvec3",
        "bvec4",
        "int",
        "ivec2",
        "ivec3",
        "ivec4",
        "uint",
        "uvec2",
        "uvec3",
        "uvec4",
        "float",
        "vec2",
        "vec3",
        "vec4",
        "double",
        "dvec2",
        "dvec3",
        "dvec4",
    )
    GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS = {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
    }
    GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS = {
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    GLSL_WAVE_MULTI_PREFIX_OPERATIONS = {
        *GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS,
        "WaveMultiPrefixCountBits",
        *GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS,
    }
    GLSL_WAVE_MULTI_PREFIX_HELPERS = {
        "WaveMultiPrefixSum": "crossglWaveMultiPrefixSum",
        "WaveMultiPrefixCountBits": "crossglWaveMultiPrefixCountBits",
        "WaveMultiPrefixProduct": "crossglWaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd": "crossglWaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr": "crossglWaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor": "crossglWaveMultiPrefixBitXor",
    }
    GLSL_WAVE_NUMERIC_OPERATIONS = {
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WavePrefixSum",
        "WavePrefixProduct",
    }
    GLSL_WAVE_INTEGER_OR_BOOLEAN_OPERATIONS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
    }
    GLSL_WAVE_BOOLEAN_OPERATIONS = {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
    }
    GLSL_WAVE_SCALAR_OR_VECTOR_OPERATIONS = {
        "WaveActiveAllEqual",
    }
    GLSL_WAVE_LANE_INDEX_ARGUMENTS = {
        "WaveReadLaneAt": 1,
        "QuadReadLaneAt": 1,
    }
    GLSL_TESSELLATION_FACTOR_BUILTINS = {
        "gl_TessLevelOuter",
        "gl_TessLevelInner",
    }
    GLSL_TESSELLATION_FACTOR_COMPONENT_COUNTS = {
        "gl_TessLevelOuter": 4,
        "gl_TessLevelInner": 2,
    }
    GLSL_TESSELLATION_CONTROL_BARRIER_BUILTINS = {
        "barrier",
        "workgroupBarrier",
    }
    GLSL_MESH_PRIMITIVE_INDEX_TYPES = {
        "points": "uint",
        "lines": "uvec2",
        "triangles": "uvec3",
    }
    GLSL_MEMORY_ATOMIC_FUNCTIONS = {
        "atomicAdd",
        "atomicMin",
        "atomicMax",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicExchange",
        "atomicCompSwap",
    }

    def __init__(self):
        """Initialize GLSL type maps and per-generation stage/resource state."""
        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.resource_variable_array_sizes = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.structured_buffer_variable_accesses = {}
        self.current_structured_buffer_access_parameters = {}
        self.glsl_buffer_block_variable_names = set()
        self.glsl_buffer_block_variable_types = {}
        self.glsl_buffer_block_variable_accesses = {}
        self.global_variable_types = {}
        self.function_sampler_parameter_indices = {}
        self.function_parameter_names = {}
        self.function_parameter_types = {}
        self.function_parameter_infos = {}
        self.function_return_types = {}
        self.function_image_access_requirements = {}
        self.function_structured_buffer_access_requirements = {}
        self.function_definitions = {}
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.emitted_glsl_resource_specialization_names = set()
        self.unsupported_structured_buffer_array_functions = {}
        self.resource_array_size_hints = {}
        self.function_resource_array_size_hints = {}
        self.literal_int_constants = {}
        self.literal_int_vector_constants = {}
        self.literal_int_vector_array_constants = {}
        self.current_compile_time_int_constants = {}
        self.current_compile_time_int_vector_constants = {}
        self.current_compile_time_int_vector_array_constants = {}
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.stage_parameter_output_aliases = {}
        self.stage_entry_functions_by_stage = {}
        self.current_identifier_aliases = {}
        self.current_mesh_output_parameters = {}
        self.current_mesh_output_topology = None
        self.current_glsl_extensions = set()
        self.current_glsl_version_line = None
        self.current_glsl_resource_binding_layouts_supported = True
        self.current_mesh_output_count_limits = {}
        self.task_payload_shared_variables = []
        self.current_target_stage = None
        self.emit_stage_guards = False
        self.stage_io_used_locations = {}
        self.stage_io_used_components = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.structs_by_name = {}
        self.vertex_input_struct_names = set()
        self.vertex_output_struct_names = set()
        self.combined_vertex_output_member_names = set()
        self.fragment_input_struct_names = set()
        self.fragment_input_member_names = set()
        self.fragment_output_struct_names = set()
        self.fragment_output_member_name_maps = {}
        self.fragment_output_member_layout_maps = {}
        self.fragment_direct_output_name_maps = {}
        self.vertex_input_member_names = set()
        self.current_function_return_type = None
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        self.function_fragment_only_requirements = {}
        self.current_expression_expected_type = None
        self.current_generic_function_substitutions = {}
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.struct_member_types = {}
        self.struct_member_name_maps = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.generic_function_definitions = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        self.glsl_buffer_block_read_validation_suppression = 0
        self.semantic_map = {
            "gl_VertexID": "gl_VertexID",
            "gl_VertexIndex": "gl_VertexID",
            "SV_VertexID": "gl_VertexID",
            "SV_VertexId": "gl_VertexID",
            "sv_vertex_id": "gl_VertexID",
            "sv_vertexid": "gl_VertexID",
            "gl_InstanceID": "gl_InstanceID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_InstanceId": "gl_InstanceID",
            "sv_instance_id": "gl_InstanceID",
            "sv_instanceid": "gl_InstanceID",
            "gl_IsFrontFace": "gl_FrontFacing",
            "gl_PrimitiveID": "gl_PrimitiveID",
            "Position": "layout(location = 0)",
            "POSITION": "layout(location = 0)",
            "Color": "layout(location = 13)",
            "COLOR": "layout(location = 13)",
            "COLOR0": "layout(location = 13)",
            "COLOR1": "layout(location = 14)",
            "COLOR2": "layout(location = 15)",
            "COLOR3": "layout(location = 16)",
            "COLOR4": "layout(location = 17)",
            "COLOR5": "layout(location = 18)",
            "COLOR6": "layout(location = 19)",
            "COLOR7": "layout(location = 20)",
            "Normal": "layout(location = 1)",
            "NORMAL": "layout(location = 1)",
            "Tangent": "layout(location = 2)",
            "TANGENT": "layout(location = 2)",
            "Binormal": "layout(location = 3)",
            "BINORMAL": "layout(location = 3)",
            "TexCoord": "layout(location = 4)",
            "TEXCOORD": "layout(location = 4)",
            "TexCoord0": "layout(location = 5)",
            "TEXCOORD0": "layout(location = 5)",
            "TexCoord1": "layout(location = 6)",
            "TEXCOORD1": "layout(location = 6)",
            "TexCoord2": "layout(location = 7)",
            "TEXCOORD2": "layout(location = 7)",
            "TexCoord3": "layout(location = 8)",
            "TEXCOORD3": "layout(location = 8)",
            "TexCoord4": "layout(location = 9)",
            "TEXCOORD4": "layout(location = 9)",
            "TexCoord5": "layout(location = 10)",
            "TEXCOORD5": "layout(location = 10)",
            "TexCoord6": "layout(location = 11)",
            "TEXCOORD6": "layout(location = 11)",
            "TexCoord7": "layout(location = 12)",
            "TEXCOORD7": "layout(location = 12)",
            # Vertex outputs
            "gl_Position": "gl_Position",
            "gl_PointSize": "gl_PointSize",
            "gl_ClipDistance": "gl_ClipDistance",
            "gl_CullDistance": "gl_CullDistance",
            # Fragment outputs
            "gl_FragColor": "layout(location = 0)",
            "gl_FragColor0": "layout(location = 0)",
            "gl_FragColor1": "layout(location = 1)",
            "gl_FragColor2": "layout(location = 2)",
            "gl_FragColor3": "layout(location = 3)",
            "gl_FragColor4": "layout(location = 4)",
            "gl_FragColor5": "layout(location = 5)",
            "gl_FragColor6": "layout(location = 6)",
            "gl_FragColor7": "layout(location = 7)",
            "gl_FragDepth": "gl_FragDepth",
            "gl_FragStencilRefEXT": "gl_FragStencilRefARB",
            "gl_SampleMask": "gl_SampleMask",
            "SV_Coverage": "gl_SampleMask",
            "sample_mask": "gl_SampleMask",
            # Additional fragment inputs
            "gl_FragCoord": "gl_FragCoord",
            "gl_FrontFacing": "gl_FrontFacing",
            "gl_PointCoord": "gl_PointCoord",
            "gl_SampleID": "gl_SampleID",
            "SV_SampleIndex": "gl_SampleID",
            "sv_sample_index": "gl_SampleID",
            "sv_sampleindex": "gl_SampleID",
            "sample_id": "gl_SampleID",
            "sample_index": "gl_SampleID",
            "gl_SamplePosition": "gl_SamplePosition",
            "sample_position": "gl_SamplePosition",
            "gl_SampleMaskIn": "gl_SampleMaskIn",
            "sample_mask_in": "gl_SampleMaskIn",
            # Compute shader specific
            "gl_GlobalInvocationID": "gl_GlobalInvocationID",
            "SV_DispatchThreadID": "gl_GlobalInvocationID",
            "SV_DispatchThreadId": "gl_GlobalInvocationID",
            "sv_dispatch_thread_id": "gl_GlobalInvocationID",
            "sv_dispatchthreadid": "gl_GlobalInvocationID",
            "gl_LocalInvocationID": "gl_LocalInvocationID",
            "SV_GroupThreadID": "gl_LocalInvocationID",
            "SV_GroupThreadId": "gl_LocalInvocationID",
            "sv_group_thread_id": "gl_LocalInvocationID",
            "sv_groupthreadid": "gl_LocalInvocationID",
            "gl_WorkGroupID": "gl_WorkGroupID",
            "SV_GroupID": "gl_WorkGroupID",
            "SV_GroupId": "gl_WorkGroupID",
            "sv_group_id": "gl_WorkGroupID",
            "sv_groupid": "gl_WorkGroupID",
            "gl_LocalInvocationIndex": "gl_LocalInvocationIndex",
            "SV_GroupIndex": "gl_LocalInvocationIndex",
            "sv_group_index": "gl_LocalInvocationIndex",
            "sv_groupindex": "gl_LocalInvocationIndex",
            "gl_WorkGroupSize": "gl_WorkGroupSize",
            "gl_NumWorkGroups": "gl_NumWorkGroups",
        }

        self.type_mapping = {
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "half": "float",
            "bfloat": "float",
            "bfloat16": "float",
            "bfloat16_t": "float",
            "half2": "vec2",
            "half3": "vec3",
            "half4": "vec4",
            "packed_half2": "vec2",
            "packed_half3": "vec3",
            "packed_half4": "vec4",
            "atomic<int>": "int",
            "atomic<uint>": "uint",
            "float16": "float",
            "f16": "float",
            "f32": "float",
            "f64": "double",
            "f16vec2": "vec2",
            "f16vec3": "vec3",
            "f16vec4": "vec4",
            "min16float": "float",
            "min10float": "float",
            "min16float2": "vec2",
            "min16float3": "vec3",
            "min16float4": "vec4",
            "min10float2": "vec2",
            "min10float3": "vec3",
            "min10float4": "vec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            "str": "int",
            "ivec2": "ivec2",
            "ivec3": "ivec3",
            "ivec4": "ivec4",
            "char": "int",
            "signed char": "int",
            "i8": "int",
            "i16": "int",
            "i32": "int",
            "i64": "int",
            "int8": "int",
            "int8_t": "int",
            "uchar": "uint",
            "unsigned char": "uint",
            "u8": "uint",
            "u16": "uint",
            "u32": "uint",
            "u64": "uint",
            "uint8": "uint",
            "uint8_t": "uint",
            "short": "int",
            "signed short": "int",
            "ushort": "uint",
            "unsigned short": "uint",
            "int16": "int",
            "int16_t": "int",
            "int32_t": "int",
            "int64": "int",
            "int64_t": "int",
            "long": "int",
            "signed long": "int",
            "ptrdiff_t": "int",
            "uint16": "uint",
            "uint16_t": "uint",
            "uint32_t": "uint",
            "uint64": "uint",
            "uint64_t": "uint",
            "ulong": "uint",
            "unsigned long": "uint",
            "size_t": "uint",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "char2": "ivec2",
            "char3": "ivec3",
            "char4": "ivec4",
            "i8vec2": "ivec2",
            "i8vec3": "ivec3",
            "i8vec4": "ivec4",
            "i16vec2": "ivec2",
            "i16vec3": "ivec3",
            "i16vec4": "ivec4",
            "uchar2": "uvec2",
            "uchar3": "uvec3",
            "uchar4": "uvec4",
            "u8vec2": "uvec2",
            "u8vec3": "uvec3",
            "u8vec4": "uvec4",
            "u16vec2": "uvec2",
            "u16vec3": "uvec3",
            "u16vec4": "uvec4",
            "short2": "ivec2",
            "short3": "ivec3",
            "short4": "ivec4",
            "ushort2": "uvec2",
            "ushort3": "uvec3",
            "ushort4": "uvec4",
            "min16int": "int",
            "min12int": "int",
            "min16int2": "ivec2",
            "min16int3": "ivec3",
            "min16int4": "ivec4",
            "min12int2": "ivec2",
            "min12int3": "ivec3",
            "min12int4": "ivec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
            "min16uint": "uint",
            "min16uint2": "uvec2",
            "min16uint3": "uvec3",
            "min16uint4": "uvec4",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
            "float2x2": "mat2",
            "float2x3": "mat3x2",
            "float2x4": "mat4x2",
            "float3x2": "mat2x3",
            "float3x3": "mat3",
            "float3x4": "mat4x3",
            "float4x2": "mat2x4",
            "float4x3": "mat3x4",
            "float4x4": "mat4",
            "simd_float2x2": "mat2",
            "simd_float2x3": "mat3x2",
            "simd_float2x4": "mat4x2",
            "simd_float3x2": "mat2x3",
            "simd_float3x3": "mat3",
            "simd_float3x4": "mat4x3",
            "simd_float4x2": "mat2x4",
            "simd_float4x3": "mat3x4",
            "simd_float4x4": "mat4",
            "half2x2": "mat2",
            "half2x3": "mat3x2",
            "half2x4": "mat4x2",
            "half3x2": "mat2x3",
            "half3x3": "mat3",
            "half3x4": "mat4x3",
            "half4x2": "mat2x4",
            "half4x3": "mat3x4",
            "half4x4": "mat4",
            "f16mat2": "mat2",
            "f16mat3": "mat3",
            "f16mat4": "mat4",
            "f16mat2x2": "mat2",
            "f16mat2x3": "mat2x3",
            "f16mat2x4": "mat2x4",
            "f16mat3x2": "mat3x2",
            "f16mat3x3": "mat3",
            "f16mat3x4": "mat3x4",
            "f16mat4x2": "mat4x2",
            "f16mat4x3": "mat4x3",
            "f16mat4x4": "mat4",
            "min16float2x2": "mat2",
            "min16float2x3": "mat3x2",
            "min16float2x4": "mat4x2",
            "min16float3x2": "mat2x3",
            "min16float3x3": "mat3",
            "min16float3x4": "mat4x3",
            "min16float4x2": "mat2x4",
            "min16float4x3": "mat3x4",
            "min16float4x4": "mat4",
            "min10float2x2": "mat2",
            "min10float2x3": "mat3x2",
            "min10float2x4": "mat4x2",
            "min10float3x2": "mat2x3",
            "min10float3x3": "mat3",
            "min10float3x4": "mat4x3",
            "min10float4x2": "mat2x4",
            "min10float4x3": "mat3x4",
            "min10float4x4": "mat4",
            "double2x2": "dmat2",
            "double2x3": "dmat3x2",
            "double2x4": "dmat4x2",
            "double3x2": "dmat2x3",
            "double3x3": "dmat3",
            "double3x4": "dmat4x3",
            "double4x2": "dmat2x4",
            "double4x3": "dmat3x4",
            "double4x4": "dmat4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "double": "double",
            "void": "void",
            "sampler": "sampler",
            "comparison_sampler": "sampler",
            "texture1D": "texture1D",
            "texture1DArray": "texture1DArray",
            "texture2D": "texture2D",
            "texture3D": "texture3D",
            "textureCube": "textureCube",
            "texture2DArray": "texture2DArray",
            "textureCubeArray": "textureCubeArray",
            "sampler1D": "sampler1D",
            "sampler1DArray": "sampler1DArray",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCube": "samplerCube",
            "sampler2DArray": "sampler2DArray",
            "samplerCubeArray": "samplerCubeArray",
            "sampler2DMS": "sampler2DMS",
            "sampler2DMSArray": "sampler2DMSArray",
            "samplerExternalOES": "samplerExternalOES",
            "__samplerExternal2DY2YEXT": "__samplerExternal2DY2YEXT",
            "sampler2DShadow": "sampler2DShadow",
            "sampler2DArrayShadow": "sampler2DArrayShadow",
            "samplerCubeShadow": "samplerCubeShadow",
            "samplerCubeArrayShadow": "samplerCubeArrayShadow",
            "isampler1D": "isampler1D",
            "isampler1DArray": "isampler1DArray",
            "isampler2D": "isampler2D",
            "isampler2DArray": "isampler2DArray",
            "isampler3D": "isampler3D",
            "isamplerCube": "isamplerCube",
            "isamplerCubeArray": "isamplerCubeArray",
            "isampler2DRect": "isampler2DRect",
            "isamplerBuffer": "isamplerBuffer",
            "isampler2DMS": "isampler2DMS",
            "isampler2DMSArray": "isampler2DMSArray",
            "usampler1D": "usampler1D",
            "usampler1DArray": "usampler1DArray",
            "usampler2D": "usampler2D",
            "usampler2DArray": "usampler2DArray",
            "usampler3D": "usampler3D",
            "usamplerCube": "usamplerCube",
            "usamplerCubeArray": "usamplerCubeArray",
            "usampler2DRect": "usampler2DRect",
            "usamplerBuffer": "usamplerBuffer",
            "usampler2DMS": "usampler2DMS",
            "usampler2DMSArray": "usampler2DMSArray",
            "iimage1D": "iimage1D",
            "iimage1DArray": "iimage1DArray",
            "iimage2D": "iimage2D",
            "iimage3D": "iimage3D",
            "iimageCube": "iimageCube",
            "iimageCubeArray": "iimageCubeArray",
            "iimage2DArray": "iimage2DArray",
            "iimage2DMS": "iimage2DMS",
            "iimage2DMSArray": "iimage2DMSArray",
            "uimage1D": "uimage1D",
            "uimage1DArray": "uimage1DArray",
            "uimage2D": "uimage2D",
            "uimage3D": "uimage3D",
            "uimageCube": "uimageCube",
            "uimageCubeArray": "uimageCubeArray",
            "uimage2DArray": "uimage2DArray",
            "uimage2DMS": "uimage2DMS",
            "uimage2DMSArray": "uimage2DMSArray",
            "image1D": "image1D",
            "image1DArray": "image1DArray",
            "image2D": "image2D",
            "image3D": "image3D",
            "imageCube": "imageCube",
            "imageCubeArray": "imageCubeArray",
            "image2DArray": "image2DArray",
            "image2DMS": "image2DMS",
            "image2DMSArray": "image2DMSArray",
            "accelerationStructureEXT": "accelerationStructureEXT",
        }

        self.function_map = {
            "atan2": "atan",
            "lerp": "mix",
            "frac": "fract",
            "saturate": "clamp",
            "tex2D": "texture",
            "tex2Dproj": "textureProj",
            "tex2Dlod": "textureLod",
            "tex2Dbias": "texture",
            "tex2Dgrad": "textureGrad",
            "tex2Doffset": "textureOffset",
            "texCUBE": "texture",
            "texCUBElod": "textureLod",
            "texCUBEbias": "texture",
            "texCUBEgrad": "textureGrad",
            "textureOffset": "textureOffset",
            "textureProj": "textureProj",
            "textureGatherOffset": "textureGatherOffset",
            "textureGatherOffsets": "textureGatherOffsets",
            "textureQueryLevels": "textureQueryLevels",
            "textureQueryLod": "textureQueryLod",
            "texelFetch": "texelFetch",
            "interpolate_at_centroid": "interpolateAtCentroid",
            "interpolate_at_sample": "interpolateAtSample",
            "interpolate_at_offset": "interpolateAtOffset",
            "interpolateAtCentroid": "interpolateAtCentroid",
            "interpolateAtSample": "interpolateAtSample",
            "interpolateAtOffset": "interpolateAtOffset",
            "TraceRay": "traceRayEXT",
            "ReportHit": "reportIntersectionEXT",
            "ReportIntersection": "reportIntersectionEXT",
            "CallShader": "executeCallableEXT",
            "ExecuteCallable": "executeCallableEXT",
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "atomicCounterIncrement": "atomicCounterIncrement",
            "atomicCounterDecrement": "atomicCounterDecrement",
            "atomicCounter": "atomicCounter",
            "atomicCounterAdd": "atomicCounterAdd",
            "ddx": "dFdx",
            "ddx_fine": "dFdxFine",
            "ddx_coarse": "dFdxCoarse",
            "ddy": "dFdy",
            "ddy_fine": "dFdyFine",
            "ddy_coarse": "dFdyCoarse",
            "fwidth_fine": "fwidthFine",
            "fwidth_coarse": "fwidthCoarse",
            "rsqrt": "inversesqrt",
            "sincos": "sin_cos",  # Custom function needed
            "log2": "log2",
            "exp2": "exp2",
            "pow": "pow",
            "sqrt": "sqrt",
            "abs": "abs",
            "sign": "sign",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "fmod": "mod",
            "trunc": "trunc",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "step": "step",
            "smoothstep": "smoothstep",
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            "all": "all",
            "any": "any",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
        }

    def generate(self, ast):
        """Generate complete GLSL source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_stage(self, ast, shader_type):
        """Generate GLSL source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def glsl_stage_names(self, ast, target_stage=None):
        stage_names = set()

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if should_emit_qualified_function(target_stage, stage_name):
                stage_names.add(stage_name)

        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                stage_names.add(stage_name)

        return stage_names

    def default_glsl_version_line(self, ast, target_stage=None):
        if self.glsl_stage_names(
            ast, target_stage
        ) & self.RAY_STAGE_NAMES or self.uses_ray_extension_type(ast, target_stage):
            return "#version 460 core"
        return "#version 450 core"

    def glsl_version_number(self, version_line):
        parts = str(version_line or "").strip().split()
        if len(parts) < 2 or parts[0] != "#version":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def glsl_resource_binding_layouts_supported(self, version_line):
        if self.glsl_extension_enabled("GL_ARB_shading_language_420pack"):
            return True

        if "es" in str(version_line or "").strip().split()[2:]:
            return True

        version_number = self.glsl_version_number(version_line)
        if version_number is None:
            return True
        return version_number >= 420

    def glsl_version_line_with_minimum(self, version_line, minimum_version):
        version_number = self.glsl_version_number(version_line)
        if version_number is None or version_number >= minimum_version:
            return version_line

        parts = str(version_line or "").strip().split()
        profile_parts = parts[2:] if len(parts) > 2 else []
        if "es" in profile_parts:
            es_minimum = 300 if minimum_version <= 330 else 320
            if version_number >= es_minimum:
                return version_line
            return f"#version {es_minimum} es"

        profile = next(
            (part for part in profile_parts if part in {"core", "compatibility"}),
            "core",
        )
        return f"#version {minimum_version} {profile}"

    def generated_stage_io_layout_minimum_version(self, code):
        minimum_version = None
        for line in str(code or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("layout("):
                continue

            end = stripped.find(")")
            if end == -1:
                continue
            layout_parts = {
                part.partition("=")[0].strip()
                for part in stripped[len("layout(") : end].split(",")
            }
            if "component" in layout_parts:
                minimum_version = max(minimum_version or 0, 440)
            if layout_parts & {"location", "index"}:
                minimum_version = max(minimum_version or 0, 330)
        return minimum_version

    def glsl_layout_minimum_satisfied_by_extensions(self, minimum_version):
        if minimum_version is None:
            return True
        if minimum_version <= 330 and self.glsl_extension_enabled(
            "GL_ARB_explicit_attrib_location"
        ):
            return True
        if minimum_version <= 440 and self.glsl_extension_enabled(
            "GL_ARB_enhanced_layouts"
        ):
            return True
        return False

    def ensure_glsl_version_supports_generated_layouts(self, code):
        minimum_version = self.generated_stage_io_layout_minimum_version(code)
        if self.glsl_layout_minimum_satisfied_by_extensions(minimum_version):
            return code

        lines = str(code).splitlines(keepends=True)
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#version"):
                continue
            upgraded = self.glsl_version_line_with_minimum(stripped, minimum_version)
            if upgraded == stripped:
                return code
            line_end = "\n" if line.endswith("\n") else ""
            lines[index] = f"{upgraded}{line_end}"
            self.current_glsl_version_line = upgraded
            self.current_glsl_resource_binding_layouts_supported = (
                self.glsl_resource_binding_layouts_supported(upgraded)
            )
            return "".join(lines)
        return code

    def should_emit_resource_binding_layouts(self):
        return getattr(
            self,
            "current_glsl_resource_binding_layouts_supported",
            True,
        )

    def glsl_resource_layout_prefix(self, *layout_parts, binding=None):
        parts = [
            str(part).strip()
            for part in layout_parts
            if part is not None and str(part).strip()
        ]
        if binding is not None and self.should_emit_resource_binding_layouts():
            parts.append(f"binding = {binding}")
        if not parts:
            return ""
        return f"layout({', '.join(parts)})"

    def glsl_preprocessor_extensions(self, ast):
        extensions = set()
        for directive in getattr(ast, "preprocessors", []) or []:
            if not isinstance(directive, PreprocessorNode):
                continue
            if directive.directive != "extension":
                continue

            content = str(getattr(directive, "content", "") or "").strip()
            if not content:
                continue

            extension_name, separator, behavior_text = content.partition(":")
            extension_name_parts = extension_name.strip().split()
            if not extension_name_parts:
                continue
            extension_name = extension_name_parts[0]
            behavior = behavior_text.strip().split()[0] if separator else ""
            if behavior in {"", "enable", "require"}:
                extensions.add(extension_name)
        return extensions

    def should_emit_glsl_preprocessor_directive(self, directive):
        if not isinstance(directive, PreprocessorNode):
            return True
        if str(directive.directive).lstrip("#") != "include":
            return True

        content = str(getattr(directive, "content", "") or "").strip()
        if not content:
            return True
        match = re.match(r'^[<"]([^>"]+)[>"]', content)
        include_name = match.group(1) if match else content.split()[0]
        normalized = include_name.strip().replace("\\", "/").lower()
        return normalized not in self.METAL_ONLY_SYSTEM_INCLUDES

    def glsl_extension_enabled(self, extension_name):
        return extension_name in self.current_glsl_extensions

    def glsl_texture_offset_non_const_enabled(self):
        return self.glsl_extension_enabled("GL_EXT_texture_offset_non_const")

    def glsl_stage_extension_lines(self, ast, target_stage=None):
        stage_names = self.glsl_stage_names(ast, target_stage)

        lines = []
        if self.uses_nonuniform_ext(ast):
            lines.append("#extension GL_EXT_nonuniform_qualifier : require")
        if stage_names & self.MESH_STAGE_NAMES:
            lines.append("#extension GL_EXT_mesh_shader : require")
        uses_ray_stage = bool(stage_names & self.RAY_STAGE_NAMES)
        if uses_ray_stage:
            lines.append("#extension GL_EXT_ray_tracing : require")
        if self.uses_ray_query(ast, target_stage) or (
            self.uses_acceleration_structure(ast, target_stage) and not uses_ray_stage
        ):
            lines.append("#extension GL_EXT_ray_query : require")
        if self.uses_ray_tracing_position_fetch(ast, target_stage):
            lines.append("#extension GL_EXT_ray_tracing_position_fetch : require")
        if self.uses_fragment_stencil_export(ast, target_stage):
            lines.append("#extension GL_ARB_shader_stencil_export : require")
        lines.extend(self.glsl_wave_extension_lines(ast, target_stage))
        return lines

    def uses_fragment_stencil_export(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        if target_stage is not None and target_stage != "fragment":
            return False
        if "fragment" not in self.glsl_stage_names(ast, target_stage):
            return False

        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                semantic = self.semantic_from_node(node)
                if semantic is None:
                    continue
                mapped_semantic = self.map_semantic(semantic)
                if mapped_semantic in {
                    "gl_FragStencilRefARB",
                    "gl_FragStencilRefEXT",
                }:
                    return True
        return False

    def glsl_wave_operations(self, ast, target_stage=None):
        operations = set()
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if isinstance(node, WaveOpNode):
                    operations.add(node.operation)
                elif isinstance(node, FunctionCallNode):
                    operation = self.function_call_name(node)
                    if operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
                        operations.add(operation)
        return operations

    def uses_nonuniform_ext(self, ast):
        for node in self.walk_ast(ast):
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node) == "nonuniformEXT"
            ):
                return True
        return False

    def glsl_wave_extension_lines(self, ast, target_stage=None):
        operations = self.glsl_wave_operations(ast, target_stage)
        lines = []
        if any(
            operation in self.GLSL_WAVE_EXTENSION_REQUIREMENTS
            for operation in operations
        ):
            lines.append("#extension GL_KHR_shader_subgroup_basic : require")
        for operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
            if operation not in operations:
                continue
            line = self.GLSL_WAVE_EXTENSION_REQUIREMENTS.get(operation)
            if line and line not in lines:
                lines.append(line)
            if operation in self.GLSL_WAVE_ADDITIONAL_EXTENSION_REQUIREMENTS:
                shuffle_line = "#extension GL_KHR_shader_subgroup_shuffle : require"
                if shuffle_line not in lines:
                    lines.append(shuffle_line)
        return lines

    def generate_glsl_wave_helpers(self, ast, target_stage=None):
        operations = self.glsl_wave_operations(ast, target_stage)
        needs_match = "WaveMatch" in operations
        multi_prefix_operations = [
            operation
            for operation in self.GLSL_WAVE_MULTI_PREFIX_HELPERS
            if operation in operations
        ]
        if not needs_match and not multi_prefix_operations:
            return ""

        code = self.glsl_wave_mask_contains_helper()
        if needs_match:
            code += self.glsl_wave_mask_set_helper()
            code += self.glsl_wave_match_helpers()
        for operation in multi_prefix_operations:
            code += self.glsl_wave_multi_prefix_helper(operation)
        return code

    def glsl_wave_mask_contains_helper(self):
        return (
            "bool crossglWaveMaskContains(uvec4 mask, uint lane) {\n"
            "    if (lane < 32u) {\n"
            "        return (mask.x & (1u << lane)) != 0u;\n"
            "    }\n"
            "    if (lane < 64u) {\n"
            "        return (mask.y & (1u << (lane - 32u))) != 0u;\n"
            "    }\n"
            "    if (lane < 96u) {\n"
            "        return (mask.z & (1u << (lane - 64u))) != 0u;\n"
            "    }\n"
            "    return (mask.w & (1u << (lane - 96u))) != 0u;\n"
            "}\n\n"
        )

    def glsl_wave_mask_set_helper(self):
        return (
            "void crossglWaveMaskSet(inout uvec4 mask, uint lane) {\n"
            "    if (lane < 32u) {\n"
            "        mask.x |= (1u << lane);\n"
            "    } else if (lane < 64u) {\n"
            "        mask.y |= (1u << (lane - 32u));\n"
            "    } else if (lane < 96u) {\n"
            "        mask.z |= (1u << (lane - 64u));\n"
            "    } else {\n"
            "        mask.w |= (1u << (lane - 96u));\n"
            "    }\n"
            "}\n\n"
        )

    def glsl_wave_match_helpers(self):
        code = ""
        for value_type in self.GLSL_WAVE_HELPER_VALUE_TYPES:
            condition = self.glsl_wave_match_condition(value_type)
            code += (
                f"uvec4 crossglWaveMatch({value_type} value) {{\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uvec4 result = uvec4(0u);\n"
                "    for (uint lane = 0u; lane < gl_SubgroupSize; ++lane) {\n"
                "        if (crossglWaveMaskContains(activeMask, lane) && "
                f"{condition}) {{\n"
                "            crossglWaveMaskSet(result, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def glsl_wave_match_condition(self, value_type):
        if value_type.startswith(("bvec", "ivec", "uvec", "vec", "dvec")):
            return "all(equal(subgroupShuffle(value, lane), value))"
        return "subgroupShuffle(value, lane) == value"

    def glsl_wave_multi_prefix_helper(self, operation):
        helper_name = self.GLSL_WAVE_MULTI_PREFIX_HELPERS[operation]
        if operation == "WaveMultiPrefixCountBits":
            return (
                f"uint {helper_name}(bool value, uvec4 mask) {{\n"
                "    uint laneValue = value ? 1u : 0u;\n"
                "    uint result = 0u;\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uint limit = min(gl_SubgroupInvocationID, gl_SubgroupSize);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (crossglWaveMaskContains(mask, lane) && "
                "crossglWaveMaskContains(activeMask, lane)) {\n"
                "            result += subgroupShuffle(laneValue, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )

        identity, assignment = self.glsl_wave_multi_prefix_identity_and_assignment(
            operation
        )
        code = ""
        for value_type in self.glsl_wave_multi_prefix_value_types(operation):
            code += (
                f"{value_type} {helper_name}({value_type} value, uvec4 mask) {{\n"
                f"    {value_type} result = {value_type}({identity});\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uint limit = min(gl_SubgroupInvocationID, gl_SubgroupSize);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (crossglWaveMaskContains(mask, lane) && "
                "crossglWaveMaskContains(activeMask, lane)) {\n"
                f"            result {assignment} subgroupShuffle(value, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def glsl_wave_multi_prefix_identity_and_assignment(self, operation):
        if operation == "WaveMultiPrefixProduct":
            return "1", "*="
        if operation == "WaveMultiPrefixBitAnd":
            return "~0", "&="
        if operation == "WaveMultiPrefixBitOr":
            return "0", "|="
        if operation == "WaveMultiPrefixBitXor":
            return "0", "^="
        return "0", "+="

    def glsl_wave_multi_prefix_value_types(self, operation):
        if operation in self.GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS:
            return (
                "int",
                "ivec2",
                "ivec3",
                "ivec4",
                "uint",
                "uvec2",
                "uvec3",
                "uvec4",
            )
        return (
            "int",
            "ivec2",
            "ivec3",
            "ivec4",
            "uint",
            "uvec2",
            "uvec3",
            "uvec4",
            "float",
            "vec2",
            "vec3",
            "vec4",
            "double",
            "dvec2",
            "dvec3",
            "dvec4",
        )

    def uses_ray_extension_type(self, ast, target_stage=None):
        return (
            self.uses_ray_query(ast, target_stage)
            or self.uses_acceleration_structure(ast, target_stage)
            or self.uses_ray_tracing_position_fetch(ast, target_stage)
        )

    def uses_ray_query(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if any(
                    self.is_ray_query_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def uses_ray_tracing_position_fetch(self, ast, target_stage=None):
        global_query_receivers = self.ray_query_declared_receiver_names(
            getattr(ast, "global_variables", []) or []
        )
        for root in self.ray_query_search_roots(ast, target_stage):
            query_receivers = (
                global_query_receivers | self.ray_query_declared_receiver_names(root)
            )
            for node in self.walk_ast(root):
                if isinstance(node, RayQueryOpNode) and (
                    node.operation in self.RAY_QUERY_POSITION_FETCH_METHODS
                ):
                    if self.ray_query_receiver_name(node.query_expr) in query_receivers:
                        return True
                if isinstance(node, FunctionCallNode):
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    if (
                        isinstance(func_expr, MemberAccessNode)
                        and func_expr.member in self.RAY_QUERY_POSITION_FETCH_METHODS
                    ):
                        if (
                            self.ray_query_receiver_name(func_expr.object)
                            in query_receivers
                        ):
                            return True
                    if (
                        self.function_call_name(node)
                        == "rayQueryGetIntersectionTriangleVertexPositionsEXT"
                    ):
                        return True
                if getattr(node, "name", None) == "gl_HitTriangleVertexPositionsEXT":
                    return True
        return False

    def uses_acceleration_structure(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if any(
                    self.is_acceleration_structure_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def ray_extension_type_nodes(self, node):
        return (
            getattr(node, "var_type", None),
            getattr(node, "vtype", None),
            getattr(node, "param_type", None),
            getattr(node, "return_type", None),
        )

    def ray_query_declared_receiver_names(self, root):
        roots = root if isinstance(root, (list, tuple)) else [root]
        names = set()
        for search_root in roots:
            for node in self.walk_ast(search_root):
                if any(
                    self.is_ray_query_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    name = getattr(node, "name", None)
                    if isinstance(name, str):
                        names.add(name)
        return names

    def ray_query_receiver_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.ray_query_receiver_name(expr.array)
        name = getattr(expr, "name", None)
        return name if isinstance(name, str) else None

    def ray_query_search_roots(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        if target_stage is None:
            return [ast]

        roots = []
        roots.extend(getattr(ast, "global_variables", []) or [])
        roots.extend(getattr(ast, "constants", []) or [])

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if should_emit_qualified_function(target_stage, stage_name):
                roots.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                roots.append(stage)

        return roots

    def glsl_stage_guard_macro(self, stage_name):
        return self.GLSL_STAGE_GUARD_MACROS.get(normalize_stage_name(stage_name))

    def should_emit_stage_guards(self, ast, target_stage=None):
        if target_stage is not None:
            return False

        stage_entry_types = self.combined_stage_entry_types()
        stage_names = {
            stage_name
            for _, stage_name, _ in collect_stage_entry_records(
                ast, None, stage_entry_types
            )
        }
        return len(stage_names) > 1

    def wrap_stage_guard(self, code, stage_name):
        if (
            not code
            or self.current_target_stage is not None
            or not self.emit_stage_guards
        ):
            return code

        macro = self.glsl_stage_guard_macro(stage_name)
        if macro is None:
            return code
        if not code.endswith("\n"):
            code += "\n"
        return f"#ifdef {macro}\n{code}#endif\n"

    def ordered_generic_struct_specializations(self):
        specializations = self.generic_struct_specializations or {}
        if not specializations:
            return {}

        type_text_by_struct_name = {
            specialization["struct_name"]: type_text
            for type_text, specialization in specializations.items()
        }
        dependencies = {type_text: set() for type_text in specializations}
        for type_text, specialization in specializations.items():
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                dependency_name = generic_struct_specialized_type_name(
                    self,
                    field_type,
                )
                dependency_type = type_text_by_struct_name.get(dependency_name)
                if dependency_type and dependency_type != type_text:
                    dependencies[type_text].add(dependency_type)

        ordered = {}
        visiting = set()
        visited = set()
        type_order = {
            type_text: index for index, type_text in enumerate(specializations)
        }

        def visit(type_text):
            if type_text in visited:
                return
            if type_text in visiting:
                return
            visiting.add(type_text)
            for dependency_type in sorted(
                dependencies[type_text],
                key=lambda item: type_order[item],
            ):
                visit(dependency_type)
            visiting.remove(type_text)
            visited.add(type_text)
            ordered[type_text] = specializations[type_text]

        for type_text in specializations:
            visit(type_text)
        return ordered

    def mapped_type_dependency_name(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type or ""))
        return base_type.strip() or None

    def struct_node_dependency_names(self, node):
        dependencies = []
        for member in getattr(node, "members", []) or []:
            if isinstance(member, ArrayNode):
                member_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "vtype", "float"),
                )
            elif hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")

            dependency = self.mapped_type_dependency_name(member_type)
            if dependency is not None:
                dependencies.append(dependency)
        return dependencies

    def order_struct_declaration_entries(self, entries):
        by_name = {}
        for entry in entries:
            by_name.setdefault(entry["name"], entry)

        order_index = {entry["name"]: index for index, entry in enumerate(entries)}
        dependencies = {}
        for entry in entries:
            dependencies[entry["name"]] = [
                dependency
                for dependency in entry["dependencies"]
                if dependency in by_name and dependency != entry["name"]
            ]

        ordered = []
        visiting = set()
        visited = set()

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                return
            visiting.add(name)
            for dependency in sorted(
                dependencies.get(name, []),
                key=lambda item: order_index[item],
            ):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(by_name[name])

        for entry in entries:
            visit(entry["name"])
        return ordered

    def generate_ordered_data_struct_declarations(self, data_struct_nodes):
        entries = []
        for node in data_struct_nodes:
            entries.append(
                {
                    "name": node.name,
                    "dependencies": self.struct_node_dependency_names(node),
                    "code": self.generate_struct(node) + "\n",
                }
            )

        for specialization in self.ordered_generic_struct_specializations().values():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in (
                            generic_struct_specialized_fields(
                                self.type_name_string,
                                specialization,
                            )
                        )
                    ],
                    "code": generate_generic_structs(
                        self,
                        {specialization["type_name"]: specialization},
                    ),
                }
            )

        for enum in self.struct_payload_enums:
            fields = enum_struct_fields(enum) or []
            entries.append(
                {
                    "name": enum.name,
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in fields
                    ],
                    "code": generate_enum_structs(self, [enum]),
                }
            )

        for type_text, specialization in self.generic_enum_specializations.items():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in generic_enum_specialized_fields(
                            self,
                            specialization,
                        )
                    ],
                    "code": generate_generic_enum_structs(
                        self,
                        {type_text: specialization},
                    ),
                }
            )

        code = ""
        for entry in self.order_struct_declaration_entries(entries):
            code += entry["code"]
        return code

    def default_value_expression_for_type(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return None

        fields = self.default_value_struct_field_types(base_type)
        if fields is None:
            return None

        field_defaults = [
            self.default_value_expression_for_type(field_type)
            or self.primitive_default_value_expression(field_type)
            for field_type in fields
        ]
        return f"{base_type}({', '.join(field_defaults)})"

    def default_value_struct_field_types(self, type_name):
        member_types = self.struct_member_types.get(type_name)
        if member_types is not None:
            return list(member_types.values())

        for enum in self.struct_payload_enums:
            if enum.name != type_name:
                continue
            fields = enum_struct_fields(enum) or []
            return ["int"] + [field_type for _field_name, field_type in fields]

        return None

    def primitive_default_value_expression(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return f"{base_type}(0)"
        if base_type == "bool":
            return "false"
        if base_type.startswith("bool") or base_type.startswith("bvec"):
            return f"{base_type}(false)"
        return f"{base_type}(0)"

    def generate_program(self, ast, target_stage=None):
        """Render an AST to GLSL, optionally filtering stage entry points."""
        target_stage = normalize_stage_name(target_stage)
        self.validate_stage_main_helper_conflict(ast, target_stage)

        self.sampler_variables = set()
        self.current_sampler_parameters = set()
        self.texture_variable_types = {}
        self.resource_variable_array_sizes = {}
        self.current_texture_parameters = {}
        self.current_resource_aliases = {}
        self.image_variable_formats = {}
        self.current_image_format_parameters = {}
        self.image_variable_accesses = {}
        self.current_image_access_parameters = {}
        self.structured_buffer_variable_accesses = {}
        self.current_structured_buffer_access_parameters = {}
        self.glsl_buffer_block_variable_names = set()
        self.glsl_buffer_block_variable_types = {}
        self.glsl_buffer_block_variable_accesses = {}
        self.global_variable_types = {}
        self.function_sampler_parameter_indices = (
            self.collect_function_sampler_parameter_indices(ast)
        )
        functions = self.collect_functions(ast)
        self.preserved_sampler_variables = set()
        self.preserved_sampler_parameters = {}
        self.function_return_types = {
            func.name: self.type_name_string(getattr(func, "return_type", "void"))
            for func in functions
            if getattr(func, "name", None)
        }
        self.function_parameter_names = collect_function_parameter_names(functions)
        self.function_parameter_types = self.collect_function_parameter_types(functions)
        self.function_parameter_infos = self.collect_function_parameter_infos(functions)
        self.function_definitions = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        self.collect_preserved_nonuniform_sampler_resources(ast, functions)
        self.validate_glsl_tessellation_control_barriers(ast, target_stage, functions)
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.emitted_glsl_resource_specialization_names = set()
        self.function_image_access_requirements = (
            collect_function_image_access_requirements(
                functions,
                self.function_parameter_names,
                self.walk_ast,
                self.function_call_name,
                self.expression_name,
            )
        )
        self.function_structured_buffer_access_requirements = (
            self.collect_function_structured_buffer_access_requirements(functions)
        )
        self.literal_int_constants = collect_literal_int_constants(
            getattr(ast, "constants", [])
        )
        self.literal_int_vector_constants = (
            self.collect_literal_int_vector_constants_from_variables(
                getattr(ast, "global_variables", []),
                self.literal_int_constants,
            )
        )
        self.literal_int_vector_array_constants = (
            self.collect_literal_int_vector_array_constants_from_variables(
                getattr(ast, "global_variables", []),
                self.literal_int_constants,
                self.literal_int_vector_constants,
            )
        )
        self.current_compile_time_int_constants = dict(self.literal_int_constants)
        self.current_compile_time_int_vector_constants = dict(
            self.literal_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = dict(
            self.literal_int_vector_array_constants
        )
        self.current_stage_output = None
        self.current_stage_inputs = {}
        self.current_stage_outputs = {}
        self.current_stage_output_member_map = {}
        self.current_stage_parameter_aliases = {}
        self.stage_entry_functions_by_stage = {}
        self.current_identifier_aliases = {}
        self.current_target_stage = target_stage
        self.current_glsl_extensions = self.glsl_preprocessor_extensions(ast)
        stage_names = {
            stage_name
            for stage_name in self.glsl_stage_names(ast, target_stage)
            if stage_name is not None
        }
        if target_stage is not None:
            self.current_global_stage_io_stage = target_stage
        elif len(stage_names) == 1:
            self.current_global_stage_io_stage = next(iter(stage_names))
        else:
            self.current_global_stage_io_stage = None
        self.emit_stage_guards = self.should_emit_stage_guards(ast, target_stage)
        self.task_payload_shared_variables = []
        self.stage_io_used_locations = {}
        self.stage_io_used_components = {}
        self.stage_io_declarations = {}
        self.flattened_stage_variables = set()
        self.fragment_output_member_layout_maps = {}
        self.fragment_direct_output_name_maps = {}
        self.fragment_blend_support_layout_parts = []
        self.current_function_return_type = None
        self.current_stage_return_type = None
        self.current_stage_entry_type = None
        self.function_fragment_only_requirements = {}
        self.current_expression_expected_type = None
        self.match_temp_variable_index = 0
        self.local_variable_types = {}
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_structured_buffer_access_parameters = {}
        self.structured_buffer_instance_members = {}
        self.structured_buffer_counter_members = {}
        self.structured_buffer_counter_instances = {}
        self.glsl_buffer_block_struct_names = set()
        structs = deduplicate_named_declarations(
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            "struct",
        )
        self.structs_by_name = {
            node.name: node for node in structs if isinstance(node, StructNode)
        }
        self.struct_member_types = collect_struct_member_types(
            structs, self.type_name_string
        )
        self.struct_member_name_maps = {}
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
        if generic_function_specializations:
            additional_generic_enum_specializations = (
                collect_generic_enum_specializations(
                    list(generic_function_specializations.values()),
                    self.generic_enum_struct_definitions,
                    self.type_name_string,
                )
            )
            if additional_generic_enum_specializations:
                self.generic_enum_specializations.update(
                    additional_generic_enum_specializations
                )
                self.struct_member_types.update(
                    collect_generic_enum_specialization_member_types(
                        self,
                        additional_generic_enum_specializations,
                    )
                )
            additional_generic_struct_specializations = (
                collect_generic_struct_specializations(
                    list(generic_function_specializations.values()),
                    self.generic_struct_definitions,
                    self.type_name_string,
                )
            )
            if additional_generic_struct_specializations:
                self.generic_struct_specializations.update(
                    additional_generic_struct_specializations
                )
                self.struct_member_types.update(
                    collect_generic_struct_specialization_member_types(
                        self,
                        additional_generic_struct_specializations,
                    )
                )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        if generic_function_specializations:
            self.function_definitions.update(
                {
                    func.name: func
                    for func in generic_function_specializations.values()
                    if getattr(func, "name", None)
                }
            )
            self.function_parameter_names.update(
                collect_function_parameter_names(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_types.update(
                self.collect_function_parameter_types(
                    generic_function_specializations.values()
                )
            )
            self.function_parameter_infos.update(
                self.collect_function_parameter_infos(
                    generic_function_specializations.values()
                )
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
        (
            self.resource_array_size_hints,
            self.function_resource_array_size_hints,
        ) = self.collect_resource_array_size_hints(ast)
        self.unsupported_structured_buffer_array_functions = (
            self.collect_unsupported_structured_buffer_array_functions(functions)
        )
        self.function_fragment_only_requirements = (
            self.collect_fragment_only_function_requirements(
                self.function_definitions.values()
            )
        )
        self.validate_global_resource_shadows(ast)
        code = "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        version_line = None
        extra_lines = []
        for directive in preprocessors:
            if not self.should_emit_glsl_preprocessor_directive(directive):
                continue
            if isinstance(directive, PreprocessorNode):
                if directive.directive == "precision":
                    line = (
                        f"precision {directive.content};"
                        if directive.content
                        else "precision;"
                    )
                else:
                    line = f"#{directive.directive} {directive.content}".strip()
            else:
                line = str(directive).strip()
            if line.startswith("#version") and version_line is None:
                version_line = line
            elif line:
                extra_lines.append(line)
        if version_line is None:
            version_line = self.default_glsl_version_line(ast, target_stage)
        self.current_glsl_version_line = version_line
        self.current_glsl_resource_binding_layouts_supported = (
            self.glsl_resource_binding_layouts_supported(version_line)
        )
        code += f"{version_line}\n"
        for line in self.glsl_stage_extension_lines(ast, target_stage):
            if line not in extra_lines:
                code += f"{line}\n"
        if extra_lines:
            code += "\n".join(extra_lines) + "\n"
        code += generate_enum_constants(
            self,
            self.plain_enums + self.struct_payload_enums,
            qualifier="const",
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
            qualifier="const",
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_glsl_wave_helpers(ast, target_stage)

        global_vars = [
            node
            for node in list(getattr(ast, "global_variables", []) or [])
            if not self.is_elided_bfloat16_type_alias_global(node)
        ]
        stage_local_resource_vars = collect_stage_local_variables(
            ast, target_stage, self.is_stage_local_resource_variable
        )
        stage_local_interface_vars = self.deduplicate_stage_interface_declarations(
            collect_stage_local_variables(
                ast, target_stage, self.is_stage_local_interface_variable
            )
        )
        self.fragment_blend_support_layout_parts = (
            self.glsl_fragment_output_blend_support_layout_parts(
                global_vars + stage_local_interface_vars
            )
        )
        self.task_payload_shared_variables = (
            self.glsl_task_payload_shared_variable_infos(
                global_vars + stage_local_interface_vars
            )
        )
        self.glsl_interface_block_structs_by_name = {
            node.name: node
            for node in structs
            if isinstance(node, StructNode)
            and self.is_glsl_interface_block_struct(node)
        }
        stage_resource_params = self.collect_stage_entry_resource_parameters(
            ast, target_stage
        )
        resource_declaration_nodes = self.deduplicate_resource_declaration_nodes(
            global_vars + stage_local_resource_vars + stage_resource_params
        )
        self.global_variable_types = self.collect_global_variable_types(
            global_vars
            + stage_local_resource_vars
            + stage_local_interface_vars
            + stage_resource_params
        )
        self.glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(resource_declaration_nodes)
        )
        self.vertex_input_struct_names = self.stage_parameter_struct_names(
            ast, "vertex"
        )
        self.vertex_output_struct_names = self.stage_return_struct_names(ast, "vertex")
        self.fragment_input_struct_names = self.stage_parameter_struct_names(
            ast, "fragment"
        )
        self.fragment_output_struct_names = self.stage_return_struct_names(
            ast, "fragment"
        )
        self.vertex_input_member_names = self.struct_member_names(
            self.vertex_input_struct_names
        )
        self.vertex_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "vertex")
        )
        self.combined_vertex_output_member_names = (
            self.vertex_output_declaration_names()
        )
        self.fragment_input_member_names = self.fragment_input_declaration_names(
            self.fragment_input_struct_names
        )
        self.fragment_input_member_names.update(
            self.stage_value_parameter_input_names(ast, "fragment")
        )
        self.fragment_output_member_name_maps = self.fragment_output_member_maps(ast)
        self.fragment_output_member_layout_maps = (
            self.fragment_output_member_layout_maps_for_outputs()
        )
        self.stage_entry_functions_by_stage = {
            stage_name: stage_functions[0]
            for stage_name in (
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
            )
            for stage_functions in [self.stage_functions(ast, stage_name)]
            if stage_functions
        }
        emit_vertex_io = target_stage in {None, "vertex"}
        emit_fragment_io = target_stage in {None, "fragment"}
        emit_graphics_io = target_stage in {None, "vertex", "fragment"}
        data_struct_nodes = []
        for node in structs:
            if isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                if node.name in self.generic_struct_definitions:
                    continue
                if self.is_glsl_interface_block_struct(node):
                    code += self.generate_glsl_interface_block_declaration(node)
                    continue
                if node.name in self.glsl_buffer_block_struct_names:
                    continue
                elif (
                    node.name == "VSInput"
                    or node.name in self.vertex_input_struct_names
                ):
                    if emit_vertex_io:
                        code += self.wrap_stage_guard(
                            self.generate_stage_input_declarations(node),
                            "vertex",
                        )
                    elif not emit_graphics_io:
                        data_struct_nodes.append(node)
                elif node.name == "VSOutput":
                    emitted_io = False
                    if node.name in self.vertex_output_struct_names and emit_vertex_io:
                        code += self.wrap_stage_guard(
                            self.generate_vertex_output_declarations(node),
                            "vertex",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if not emitted_io:
                        if emit_vertex_io:
                            code += self.wrap_stage_guard(
                                self.generate_legacy_output_declarations(node),
                                "vertex",
                            )
                        else:
                            data_struct_nodes.append(node)
                    elif node.name in self.fragment_output_struct_names:
                        data_struct_nodes.append(node)
                elif node.name in self.vertex_output_struct_names:
                    emitted_io = False
                    if emit_vertex_io:
                        code += self.wrap_stage_guard(
                            self.generate_vertex_output_declarations(node),
                            "vertex",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_input_struct_names
                        and emit_fragment_io
                    ):
                        code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if (
                        node.name in self.fragment_output_struct_names
                        and emit_fragment_io
                    ):
                        code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                        emitted_io = True
                    if self.should_emit_flattened_stage_io_struct(
                        ast, node.name, emitted_io, emit_graphics_io
                    ):
                        data_struct_nodes.append(node)
                elif node.name == "PSInput":
                    if emit_fragment_io:
                        code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                    elif not emit_graphics_io:
                        data_struct_nodes.append(node)
                elif node.name in self.fragment_input_struct_names:
                    if emit_fragment_io:
                        code += self.wrap_stage_guard(
                            self.generate_fragment_input_declarations(node),
                            "fragment",
                        )
                        if node.name in self.fragment_output_struct_names:
                            code += self.wrap_stage_guard(
                                self.generate_fragment_output_declarations(node),
                                "fragment",
                            )
                    if node.name in self.fragment_output_struct_names:
                        data_struct_nodes.append(node)
                    elif not emit_graphics_io:
                        data_struct_nodes.append(node)
                elif node.name in self.fragment_output_struct_names:
                    if emit_fragment_io:
                        code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                    if node.name != "PSOutput" or not emit_fragment_io:
                        data_struct_nodes.append(node)
                elif node.name == "PSOutput":
                    if emit_fragment_io:
                        code += self.wrap_stage_guard(
                            self.generate_fragment_output_declarations(node),
                            "fragment",
                        )
                    else:
                        data_struct_nodes.append(node)
                else:
                    data_struct_nodes.append(node)

        code += self.generate_ordered_data_struct_declarations(data_struct_nodes)
        code += self.generate_constants(ast, struct_dependent_constants)
        code += generate_enum_constructor_functions(self, self.struct_payload_enums)
        code += generate_generic_enum_constructor_functions(
            self,
            self.generic_enum_specializations,
        )

        code += self.generate_stage_parameter_input_declarations(ast, target_stage)
        code += self.generate_stage_parameter_output_declarations(ast, target_stage)

        resource_binding_cursors = {}
        used_resource_bindings = {}
        self.reserve_explicit_global_resource_bindings(
            resource_declaration_nodes, used_resource_bindings
        )
        for index, node in enumerate(resource_declaration_nodes):
            vtype, array_size, array_suffix, resource_count = (
                self.resource_declaration_shape(node)
            )

            if hasattr(node, "name"):
                var_name = node.name
            elif hasattr(node, "variable_name"):
                var_name = node.variable_name
            else:
                var_name = f"var{index}"

            if self.is_glsl_buffer_block_variable(node, vtype):
                self.record_glsl_buffer_block_variable(var_name, node, vtype)
                if self.is_shader_record_buffer_block(node):
                    if self.explicit_resource_binding_index(node) is not None:
                        raise ValueError(
                            "GLSL shaderRecordEXT buffer blocks cannot declare "
                            "binding layout qualifiers"
                        )
                    declaration = self.glsl_buffer_block_declaration(
                        node, vtype, var_name, None, array_suffix
                    )
                    if declaration is not None:
                        code += declaration
                        continue

                binding_namespace = "buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                declaration = self.glsl_buffer_block_declaration(
                    node, vtype, var_name, resource_binding, array_suffix
                )
                if declaration is not None:
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                        var_name,
                    )
                    code += declaration
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                    continue

            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                self.explicit_resource_binding_index(node)
                self.sampler_variables.add(var_name)
                if self.should_preserve_sampler_resource(var_name):
                    binding_namespace = "sampler binding"
                    resource_binding = self.opengl_resource_binding_for_declaration(
                        node,
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        resource_binding,
                        resource_count,
                        var_name,
                    )
                    declaration = format_c_style_array_declaration(
                        f"{mapped_type}{array_suffix}", var_name
                    )
                    layout = self.glsl_resource_layout_prefix(binding=resource_binding)
                    prefix = f"{layout} " if layout else ""
                    code += f"{prefix}uniform {declaration};\n"
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        resource_binding,
                        resource_count,
                    )
                continue
            if self.is_structured_buffer_type(vtype):
                self.record_structured_buffer_access_metadata(var_name, vtype, node)
                binding_namespace = "buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.structured_buffer_block_declaration(
                    vtype, var_name, resource_binding, array_size, node
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
                if self.structured_buffer_requires_counter(vtype):
                    counter_binding = self.next_available_resource_binding(
                        used_resource_bindings,
                        resource_binding_cursors,
                        binding_namespace,
                        resource_count,
                    )
                    counter_name = self.structured_buffer_counter_resource_name(
                        var_name
                    )
                    self.reserve_resource_binding_range(
                        used_resource_bindings,
                        "OpenGL",
                        binding_namespace,
                        counter_binding,
                        resource_count,
                        counter_name,
                    )
                    code += self.structured_buffer_counter_block_declaration(
                        var_name, counter_binding, array_size
                    )
                    self.advance_resource_binding(
                        resource_binding_cursors,
                        binding_namespace,
                        counter_binding,
                        resource_count,
                    )
                continue
            if self.is_opaque_resource_type(mapped_type):
                self.texture_variable_types[var_name] = mapped_type
                fixed_array_size = self.glsl_resource_array_size_value(array_size)
                if fixed_array_size is not None:
                    self.resource_variable_array_sizes[var_name] = fixed_array_size
                record_explicit_image_metadata(
                    var_name,
                    node,
                    self.attribute_value_to_string,
                    image_formats=self.image_variable_formats,
                    image_accesses=self.image_variable_accesses,
                )
            declaration = format_c_style_array_declaration(
                f"{mapped_type}{array_suffix}", var_name
            )
            if self.is_opaque_resource_type(mapped_type):
                binding_namespace = (
                    "image binding"
                    if self.is_storage_image_type(vtype)
                    else "texture binding"
                )
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                layout = self.opaque_resource_layout(
                    mapped_type, resource_binding, node
                )
                memory_qualifiers = self.resource_memory_qualifiers(node)
                precision_qualifiers = self.resource_precision_qualifiers(node)
                qualifier_parts = []
                if layout:
                    qualifier_parts.append(layout)
                if memory_qualifiers:
                    qualifier_parts.append(memory_qualifiers)
                qualifier_parts.append("uniform")
                if precision_qualifiers:
                    qualifier_parts.append(precision_qualifiers)
                code += f"{' '.join(qualifier_parts)} {declaration};\n"
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
            elif self.is_layout_bound_struct_uniform(node, vtype):
                binding_namespace = "uniform buffer binding"
                resource_binding = self.opengl_resource_binding_for_declaration(
                    node,
                    used_resource_bindings,
                    resource_binding_cursors,
                    binding_namespace,
                    resource_count,
                )
                self.reserve_resource_binding_range(
                    used_resource_bindings,
                    "OpenGL",
                    binding_namespace,
                    resource_binding,
                    resource_count,
                    var_name,
                )
                code += self.glsl_struct_uniform_block_declaration(
                    vtype,
                    var_name,
                    resource_binding,
                    array_suffix,
                )
                self.advance_resource_binding(
                    resource_binding_cursors,
                    binding_namespace,
                    resource_binding,
                    resource_count,
                )
            else:
                code += self.generate_global_variable_declaration(
                    node,
                    declaration,
                    vtype,
                    mapped_type_for_layout=f"{mapped_type}{array_suffix}",
                )

        for node in stage_local_interface_vars:
            code += self.generate_stage_local_interface_variable_declaration(node)

        cbuffers = getattr(ast, "cbuffers", [])
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        combined_stage_entry_names = self.combined_stage_entry_names(ast, target_stage)
        self.prepare_glsl_resource_function_specializations(ast)

        functions = getattr(ast, "functions", [])
        deferred_top_level_helpers_by_stage = (
            self.collect_stage_deferred_top_level_helpers(ast, functions, target_stage)
        )
        deferred_top_level_helper_ids = {
            id(func)
            for helpers in deferred_top_level_helpers_by_stage.values()
            for func in helpers
        }
        for func in functions:
            if id(func) in deferred_top_level_helper_ids:
                continue
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            resource_specializations = self.glsl_resource_function_emission_list(
                getattr(func, "name", None)
            )
            helper_stage_context = (
                target_stage
                if target_stage is not None and not qualifier_name
                else None
            )
            if resource_specializations and qualifier_name not in {
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
            }:
                for specialized_func in resource_specializations:
                    code += self.generate_function(
                        specialized_func,
                        stage_context=helper_stage_context,
                    )

            if generic_function_parameters(func):
                specialized_code = ""
                for specialized_func in generic_function_emission_list(self, func):
                    specialized_code += self.generate_function(
                        specialized_func,
                        stage_context=helper_stage_context,
                    )
                if qualifier_name in self.GLSL_STAGE_GUARD_MACROS:
                    code += self.wrap_stage_guard(specialized_code, qualifier_name)
                else:
                    code += specialized_code
                continue

            if qualifier_name == "vertex":
                stage_code = "// Vertex Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "vertex")
            elif qualifier_name == "fragment":
                stage_code = "// Fragment Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "fragment")
            elif qualifier_name == "compute":
                stage_code = "// Compute Shader\n"
                stage_code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=combined_stage_entry_names.get(id(func)),
                )
                code += self.wrap_stage_guard(stage_code, "compute")
            else:
                code += self.generate_function(
                    func,
                    stage_context=helper_stage_context,
                )

        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                stage_code = ""
                stage_local_functions = list(
                    getattr(stage, "local_functions", []) or []
                )
                deferred_top_level_helpers = deferred_top_level_helpers_by_stage.get(
                    stage_name, []
                )
                for func in order_functions_by_dependencies(
                    stage_local_functions + deferred_top_level_helpers,
                    self.walk_ast,
                    self.function_call_name,
                    FunctionCallNode,
                ):
                    resource_specializations = (
                        self.glsl_resource_function_emission_list(
                            getattr(func, "name", None)
                        )
                    )
                    if resource_specializations:
                        for specialized_func in resource_specializations:
                            stage_code += self.generate_function(
                                specialized_func,
                                stage_context=stage_name,
                            )

                    if generic_function_parameters(func):
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            stage_code += self.generate_function(
                                specialized_func,
                                stage_context=stage_name,
                            )
                    else:
                        stage_code += self.generate_function(
                            func,
                            stage_context=stage_name,
                        )

                if hasattr(stage, "entry_point"):
                    stage_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        stage_layout_qualifiers=getattr(
                            stage, "layout_qualifiers", None
                        ),
                        entry_name=combined_stage_entry_names.get(
                            id(stage.entry_point)
                        ),
                    )

                if stage_code:
                    stage_code = f"// {stage_name.title()} Shader\n" + stage_code
                    code += self.wrap_stage_guard(stage_code, stage_name)

        return self.ensure_glsl_version_supports_generated_layouts(code)

    def collect_stage_deferred_top_level_helpers(self, ast, functions, target_stage):
        if target_stage is None or not getattr(ast, "stages", None):
            return {}

        deferred_by_stage = {}
        for stage_type, stage in ast.stages.items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue

            local_functions = list(getattr(stage, "local_functions", []) or [])
            deferred = self.top_level_helpers_depending_on_stage_locals(
                functions,
                local_functions,
            )
            if deferred:
                deferred_by_stage[stage_name] = deferred

        return deferred_by_stage

    def top_level_helpers_depending_on_stage_locals(self, functions, local_functions):
        required_names = {
            getattr(func, "name", None)
            for func in local_functions
            if getattr(func, "name", None)
        }
        if not required_names:
            return []

        deferred_ids = set()
        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name or id(func) in deferred_ids:
                    continue
                if normalize_stage_name(getattr(func, "qualifier", None)):
                    continue
                qualifiers = getattr(func, "qualifiers", []) or []
                if qualifiers and normalize_stage_name(qualifiers[0]):
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    if self.function_call_name(node) not in required_names:
                        continue
                    deferred_ids.add(id(func))
                    required_names.add(func_name)
                    changed = True
                    break

        return [func for func in functions if id(func) in deferred_ids]

    def collect_fragment_only_function_requirements(self, functions):
        """Mark helpers whose call graph requires fragment-stage GLSL builtins."""
        functions_by_name = {}
        duplicate_names = set()
        for func in functions:
            name = getattr(func, "name", None)
            if not name:
                continue
            if name in functions_by_name:
                duplicate_names.add(name)
                continue
            functions_by_name[name] = func
        for name in duplicate_names:
            functions_by_name.pop(name, None)

        requirements = {
            name: self.function_body_uses_fragment_only_call(func)
            for name, func in functions_by_name.items()
        }

        changed = True
        while changed:
            changed = False
            for name, func in functions_by_name.items():
                if requirements.get(name):
                    continue
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    if requirements.get(callee_name):
                        requirements[name] = True
                        changed = True
                        break

        return requirements

    def function_body_uses_fragment_only_call(self, func):
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            if self.is_fragment_only_call_name(self.function_call_name(node)):
                return True
        return False

    def is_fragment_only_call_name(self, func_name):
        if not func_name:
            return False
        if func_name == "clip" and func_name not in self.function_return_types:
            return True
        mapped_name = self.function_map.get(func_name, func_name)
        return (
            mapped_name in self.GLSL_DERIVATIVE_FUNCTIONS
            or mapped_name in self.GLSL_INTERPOLATION_FUNCTIONS
        )

    def validate_fragment_only_helper_call(self, func_name):
        if (
            not func_name
            or not self.function_fragment_only_requirements.get(func_name)
            or self.current_stage_entry_type in (None, "fragment")
        ):
            return
        raise ValueError(
            f"OpenGL helper function '{func_name}' uses fragment-only operations "
            "and is only valid in fragment stages"
        )

    def prepare_glsl_resource_function_specializations(self, ast):
        """Create GLSL helper variants for concrete storage image arguments."""
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        pending = []

        def scan_function(func, aliases):
            before = set(self.glsl_resource_function_specializations)
            self.collect_glsl_resource_specializations_from_body(
                getattr(func, "body", []),
                aliases,
            )
            after = set(self.glsl_resource_function_specializations)
            pending.extend(after - before)

        for func in self.collect_functions(ast):
            scan_function(func, {})

        seen = set()
        while pending:
            key = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            specialized_func = self.glsl_resource_function_specializations.get(key)
            if specialized_func is None:
                continue
            scan_function(
                specialized_func,
                getattr(specialized_func, "_glsl_resource_aliases", {}) or {},
            )

        if not self.glsl_resource_function_specializations:
            return

        specialized_functions = list(
            self.glsl_resource_function_specializations.values()
        )
        self.function_definitions.update(
            {
                func.name: func
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_parameter_names.update(
            collect_function_parameter_names(specialized_functions)
        )
        self.function_parameter_types.update(
            self.collect_function_parameter_types(specialized_functions)
        )
        self.function_parameter_infos.update(
            self.collect_function_parameter_infos(specialized_functions)
        )
        self.function_sampler_parameter_indices.update(
            self.collect_function_sampler_parameter_indices(specialized_functions)
        )

    def collect_glsl_resource_specializations_from_body(self, body, aliases):
        for node in self.walk_ast(body):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if not func_name:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            self.ensure_glsl_resource_function_specialization(
                func_name,
                args,
                aliases,
            )
            self.ensure_glsl_dynamic_resource_function_specializations(
                func_name,
                args,
                aliases,
            )

    def glsl_resource_function_emission_list(self, source_name):
        if not source_name:
            return []
        functions = [
            func
            for func in self.glsl_resource_function_specializations.values()
            if getattr(func, "_glsl_resource_source_name", None) == source_name
        ]
        functions.sort(key=lambda func: getattr(func, "name", ""))
        result = []
        for func in functions:
            name = getattr(func, "name", None)
            if name in self.emitted_glsl_resource_specialization_names:
                continue
            self.emitted_glsl_resource_specialization_names.add(name)
            result.append(func)
        return result

    def glsl_resource_binding_info(self, arg, aliases):
        arg_name = self.expression_name(arg)
        is_array_access = isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        )
        alias_binding = aliases.get(arg_name)
        if alias_binding is not None and not is_array_access:
            return alias_binding

        resource_type = self.texture_argument_resource_type(arg)
        if not self.is_storage_image_type(resource_type) and alias_binding is not None:
            resource_type = alias_binding.get("type")
        if not self.is_storage_image_type(resource_type):
            return None

        expression = self.glsl_resource_argument_expression(arg, aliases)
        if expression is None:
            return None

        dynamic_parameters = self.glsl_resource_argument_dynamic_parameters(arg)
        specializable = self.glsl_resource_argument_is_specializable(arg)
        if alias_binding is not None:
            specializable = specializable and alias_binding.get("specializable", True)
            dynamic_parameters = [
                *(alias_binding.get("dynamic_parameters") or []),
                *(dynamic_parameters or []),
            ]

        if alias_binding is not None:
            return {
                **alias_binding,
                "expression": expression,
                "type": resource_type,
                "specializable": specializable,
                "dynamic_parameters": dynamic_parameters or [],
            }

        binding = {
            "expression": expression,
            "type": resource_type,
            "format": self.image_resource_format(arg),
            "access": self.image_resource_access(arg),
            "specializable": specializable,
            "dynamic_parameters": dynamic_parameters or [],
        }
        array_size = self.resource_variable_array_sizes.get(arg_name)
        if array_size is not None:
            binding["array_size"] = array_size
        return binding

    def glsl_resource_argument_expression(self, arg, aliases):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_code = self.glsl_resource_argument_expression(array_expr, aliases)
            if array_code is None:
                return None
            return f"{array_code}[{self.generate_expression(index_expr)}]"
        arg_name = self.expression_name(arg)
        if arg_name in aliases:
            return aliases[arg_name]["expression"]
        if isinstance(arg, str):
            return arg
        if hasattr(arg, "name") and isinstance(arg.name, str):
            return arg.name
        return None

    def glsl_resource_argument_dynamic_parameters(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            dynamic_parameters = self.glsl_resource_argument_dynamic_parameters(
                array_expr
            )
            if dynamic_parameters is None:
                return None
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            if isinstance(literal_index, int) and not isinstance(literal_index, bool):
                return dynamic_parameters
            index_name = self.expression_name(index_expr)
            if not index_name:
                return None
            index_expression = self.generate_expression(index_expr)
            dynamic_parameters.append(
                {
                    "name": index_name,
                    "type": self.dynamic_resource_index_parameter_type(index_expr),
                    "expression": index_expression,
                }
            )
            return dynamic_parameters
        return []

    def glsl_resource_argument_is_specializable(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            return isinstance(literal_index, int) and not isinstance(
                literal_index, bool
            )
        return True

    def glsl_resource_array_size_value(self, size):
        literal_size = self.literal_int_value(size, self.literal_int_constants)
        if isinstance(literal_size, int) and not isinstance(literal_size, bool):
            return literal_size if literal_size > 0 else None
        return None

    def glsl_resource_parameter_array_size(self, func_name, param_name, param_type):
        fixed_size = self.fixed_resource_array_size(param_type)
        if fixed_size is not None:
            return fixed_size
        hinted_size = self.function_resource_array_size_hints.get(func_name, {}).get(
            param_name
        )
        return self.glsl_resource_array_size_value(hinted_size)

    def glsl_dynamic_resource_array_access_info(self, arg, aliases):
        if not (
            isinstance(arg, ArrayAccessNode)
            or (hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__))
        ):
            return None
        if self.glsl_resource_argument_is_specializable(arg):
            return None

        array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
        index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
        array_name = self.expression_name(array_expr)
        if not array_name:
            return None

        alias_binding = aliases.get(array_name)
        if alias_binding is not None:
            array_size = self.glsl_resource_array_size_value(
                alias_binding.get("array_size")
            )
        else:
            array_size = self.resource_variable_array_sizes.get(array_name)
        if array_size is None:
            return None

        return {
            "array_expr": array_expr,
            "index_expr": index_expr,
            "array_size": array_size,
        }

    def glsl_static_array_access_argument(self, dynamic_info, index):
        return ArrayAccessNode(dynamic_info["array_expr"], index)

    def glsl_dynamic_resource_call_info(self, func_name, args, aliases):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        dynamic_arg = None
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_storage_image_type(param_type):
                continue
            binding = self.glsl_resource_binding_info(arg, aliases)
            if binding is None or binding.get("specializable", True):
                continue
            dynamic_info = self.glsl_dynamic_resource_array_access_info(arg, aliases)
            if dynamic_info is None:
                continue
            if dynamic_arg is not None:
                return None
            dynamic_arg = {
                "arg_index": index,
                **dynamic_info,
            }

        return dynamic_arg

    def ensure_glsl_dynamic_resource_function_specializations(
        self, func_name, args, aliases
    ):
        dynamic_info = self.glsl_dynamic_resource_call_info(func_name, args, aliases)
        if dynamic_info is None:
            return []

        specializations = []
        for index in range(dynamic_info["array_size"]):
            static_args = list(args or [])
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            specialized = self.ensure_glsl_resource_function_specialization(
                func_name,
                static_args,
                aliases,
            )
            if specialized is None:
                return []
            specializations.append((index, specialized, static_args))
        return specializations

    def dynamic_resource_index_parameter_type(self, index_expr):
        index_name = self.expression_name(index_expr)
        if index_name:
            return self.local_variable_types.get(index_name, "int")
        return "int"

    def glsl_resource_function_specialization_key(self, func_name, args, aliases):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return None, None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        access_requirements = self.function_image_access_requirements.get(func_name, {})
        bindings = {}
        key_parts = []
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_storage_image_type(param_type):
                continue
            binding = self.glsl_resource_binding_info(arg, aliases)
            if binding is None:
                continue
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            array_size = self.glsl_resource_parameter_array_size(
                func_name, param_name, param_type
            )
            if array_size is not None:
                binding = {**binding, "array_size": array_size}
            self.validate_storage_image_parameter_contract(
                func_name,
                param,
                arg,
                index,
                self.storage_image_binding_contract(binding),
            )
            required_access = access_requirements.get(param_name)
            if required_access is not None and not image_access_satisfies_requirement(
                required_access,
                binding.get("access"),
            ):
                return None, None
            if not binding.get("specializable", True):
                continue
            bindings[index] = (param_name, binding)
            key_parts.append((index, binding["expression"]))

        if not bindings:
            return None, None
        return (func_name, tuple(key_parts)), bindings

    def ensure_glsl_resource_function_specialization(self, func_name, args, aliases):
        key, bindings = self.glsl_resource_function_specialization_key(
            func_name,
            args,
            aliases,
        )
        if key is None:
            return None
        if key in self.glsl_resource_function_specializations:
            return self.glsl_resource_function_specializations[key]

        source_func = self.function_definitions.get(func_name)
        if source_func is None:
            return None

        clone = deepcopy(source_func)
        clone.name = self.glsl_resource_specialization_name(func_name, bindings)
        clone.parameters = [
            param
            for index, param in enumerate(
                getattr(clone, "parameters", getattr(clone, "params", []))
            )
            if index not in bindings
        ]
        remaining_param_names = {
            getattr(param, "name", None)
            for param in getattr(clone, "parameters", getattr(clone, "params", []))
        }
        dynamic_call_arguments = []
        for _, (_, binding) in sorted(bindings.items()):
            for dynamic_param in binding.get("dynamic_parameters") or []:
                param_name = dynamic_param.get("name")
                if not param_name or param_name in remaining_param_names:
                    continue
                remaining_param_names.add(param_name)
                clone.parameters.append(
                    SimpleNamespace(
                        name=param_name,
                        param_type=dynamic_param.get("type", "int"),
                        vtype=dynamic_param.get("type", "int"),
                        semantic=None,
                        attributes=[],
                        qualifiers=[],
                    )
                )
                dynamic_call_arguments.append(
                    dynamic_param.get("expression", param_name)
                )
        clone._glsl_resource_source_name = func_name
        clone._glsl_resource_bound_indices = set(bindings)
        clone._glsl_resource_dynamic_call_arguments = dynamic_call_arguments
        clone._glsl_resource_aliases = {
            param_name: binding for param_name, binding in bindings.values()
        }

        self.glsl_resource_function_specializations[key] = clone
        self.glsl_resource_function_call_names[key] = clone.name
        self.glsl_resource_specialized_source_names.add(func_name)
        return clone

    def glsl_resource_specialization_name(self, func_name, bindings):
        suffix_parts = []
        for _, (param_name, binding) in sorted(bindings.items()):
            suffix_parts.append(
                "{}_{}".format(
                    sanitize_type_name(param_name),
                    sanitize_type_name(binding["expression"]),
                )
            )
        suffix = "_".join(suffix_parts)
        return f"{sanitize_type_name(func_name)}__glsl_{suffix}"

    def glsl_resource_function_call_specialization(self, func_name, args):
        key, _ = self.glsl_resource_function_specialization_key(
            func_name,
            args,
            self.current_resource_aliases,
        )
        if key is None:
            return None
        return self.glsl_resource_function_specializations.get(key)

    def glsl_resource_specialized_call_arguments(self, specialized_func, args):
        bound_indices = getattr(specialized_func, "_glsl_resource_bound_indices", set())
        call_args = [
            arg for index, arg in enumerate(args or []) if index not in bound_indices
        ]
        call_args.extend(
            getattr(specialized_func, "_glsl_resource_dynamic_call_arguments", []) or []
        )
        return call_args

    def generate_constants(self, ast, constants=None):
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(value)
            constant_id = self.glsl_specialization_constant_id(node)
            declaration = format_c_style_array_declaration(
                self.map_type(const_type), name
            )
            if constant_id is not None:
                code += (
                    "/* CrossGL fallback: OpenGL source validation cannot preserve "
                    f"specialization constant id {constant_id} for "
                    f"'{name}'; using the default literal. */\n"
                )
            code += f"const {declaration} = {value_code};\n"

        return f"{code}\n" if code else ""

    def glsl_constant_layout_prefix(self, node):
        constant_id = self.glsl_specialization_constant_id(node)
        if constant_id is None:
            return ""
        return f"layout(constant_id = {constant_id}) "

    def glsl_specialization_constant_id(self, node):
        constant_id = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            if normalized != "constant_id":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            value = self.attribute_value_to_string(arguments[0])
            if constant_id is not None and constant_id != value:
                name = getattr(node, "name", "<unnamed>")
                raise ValueError(
                    "Conflicting OpenGL specialization constant ids for "
                    f"'{name}': {constant_id} differs from {value}"
                )
            constant_id = value
        return constant_id

    def generate_constant_expression(self, expr):
        value_code = self.generate_expression(expr)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def combined_stage_entry_types(self):
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
        }

    def combined_stage_entry_names(self, ast, target_stage=None):
        """Return distinct helper names for monolithic multi-stage output."""
        if target_stage is not None:
            return {}

        stage_entry_types = self.combined_stage_entry_types()
        entries = collect_stage_entry_records(ast, None, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, None, stage_entry_types
        )
        if (
            self.should_emit_stage_guards(ast, target_stage)
            and "main" not in used_names
        ):
            return {entry_id: "main" for entry_id, _stage_name, _func in entries}

        stage_names = {stage_name for _entry_id, stage_name, _func in entries}
        if len(entries) > 1 and len(stage_names) == 1 and "main" not in used_names:
            first_entry_id, _first_stage_name, _first_func = entries[0]
            assigned_names = assign_stage_entry_names(
                entries[1:],
                used_names | {"main"},
                lambda stage_name, _func: f"{stage_name}_main",
            )
            return {first_entry_id: "main", **assigned_names}

        return assign_stage_entry_names(
            entries,
            used_names,
            lambda stage_name, _func: f"{stage_name}_main",
            single_entry_default="main",
        )

    def validate_stage_main_helper_conflict(self, ast, target_stage):
        if target_stage is None:
            return

        stage_entry_types = self.combined_stage_entry_types()
        has_global_main_helper = False
        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name in stage_entry_types:
                continue
            if getattr(func, "name", None) == "main":
                has_global_main_helper = True
                break

        if not has_global_main_helper:
            return

        if collect_stage_entry_records(ast, target_stage, stage_entry_types):
            raise ValueError(
                "Global helper function 'main' conflicts with the GLSL "
                f"{target_stage} stage entry point"
            )

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = getattr(ast, "cbuffers", [])
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate cbuffer name(s) in OpenGL output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise ValueError(
                f"Cbuffer name(s) conflict with existing OpenGL declaration(s): {names}"
            )

        duplicate_members = collect_duplicate_cbuffer_member_names(cbuffers)
        if duplicate_members:
            names = ", ".join(sorted(duplicate_members))
            raise ValueError(
                f"Ambiguous cbuffer member name(s) in OpenGL output: {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise ValueError(
                "Cbuffer member name(s) conflict with OpenGL global declaration(s): "
                f"{names}"
            )
        cbuffer_binding_cursors = {}
        used_cbuffer_bindings = {}
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            if explicit_binding is None:
                continue
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                explicit_binding,
                1,
                node.name,
            )
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            resource_binding = (
                explicit_binding
                if explicit_binding is not None
                else self.next_available_resource_binding(
                    used_cbuffer_bindings,
                    cbuffer_binding_cursors,
                    "uniform buffer binding",
                    1,
                )
            )
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                resource_binding,
                1,
                node.name,
            )
            self.advance_resource_binding(
                cbuffer_binding_cursors,
                "uniform buffer binding",
                resource_binding,
                1,
            )
            layout = self.glsl_resource_layout_prefix(
                "std140", binding=resource_binding
            )
            if isinstance(node, StructNode):
                code += f"{layout} uniform {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
                            )
                    else:
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
            ):  # CbufferNode handling
                code += f"{layout} uniform {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += (
                                f"    {self.map_type(element_type)} {member.name}[];\n"
                            )
                    else:
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

    def generate_compute_layout(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return (
            f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n"
        )

    def generate_stage_layout(self, shader_type, func, stage_layout_qualifiers=None):
        if shader_type == "fragment":
            return self.generate_fragment_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "geometry":
            return self.generate_geometry_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "tessellation_control":
            return self.generate_tessellation_control_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "tessellation_evaluation":
            return self.generate_tessellation_evaluation_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "mesh":
            return self.generate_mesh_stage_layout(func, stage_layout_qualifiers)
        return ""

    def generate_fragment_stage_layout(self, func, stage_layout_qualifiers=None):
        code = ""
        if self.glsl_stage_bare_attribute(
            func, {"early_fragment_tests"}, stage_layout_qualifiers, "in"
        ):
            code += "layout(early_fragment_tests) in;\n"

        blend_support = self.glsl_fragment_blend_support_layout_parts(
            func, stage_layout_qualifiers
        )
        if blend_support:
            code += f"layout({', '.join(blend_support)}) out;\n"
        return code

    def glsl_fragment_blend_support_layout_parts(
        self, func, stage_layout_qualifiers=None
    ):
        parts = []
        seen = set()

        def add_if_blend_support(attr):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name not in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                return
            if getattr(attr, "arguments", []) or attr_name in seen:
                return
            seen.add(attr_name)
            parts.append(attr_name)

        for attr in getattr(func, "attributes", []) or []:
            add_if_blend_support(attr)
        for attr_name in getattr(self, "fragment_blend_support_layout_parts", []):
            if attr_name not in seen:
                seen.add(attr_name)
                parts.append(attr_name)
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, "out"):
            add_if_blend_support(attr)

        return parts

    def glsl_fragment_output_blend_support_layout_parts(self, nodes):
        parts = []
        seen = set()

        for node in nodes:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if "out" not in qualifiers:
                continue
            for attr in getattr(node, "attributes", []) or []:
                attr_name = self.glsl_stage_control_attribute_name(attr)
                if attr_name not in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                    continue
                if getattr(attr, "arguments", []) or attr_name in seen:
                    continue
                seen.add(attr_name)
                parts.append(attr_name)

        return parts

    def generate_geometry_stage_layout(self, func, stage_layout_qualifiers=None):
        input_bare_names = {
            "points",
            "lines",
            "lines_adjacency",
            "triangles",
            "triangles_adjacency",
        }
        output_bare_names = {"points", "line_strip", "triangle_strip"}
        if self.glsl_stage_has_attribute(
            func, "inputtopology", stage_layout_qualifiers, "in"
        ) or self.glsl_stage_has_bare_attribute(
            func,
            input_bare_names - {"points"},
            stage_layout_qualifiers,
            "in",
        ):
            input_bare_names.remove("points")
        if self.glsl_stage_has_attribute(
            func, "outputtopology", stage_layout_qualifiers, "out"
        ) or self.glsl_stage_has_bare_attribute(
            func,
            output_bare_names - {"points"},
            stage_layout_qualifiers,
            "out",
        ):
            output_bare_names.remove("points")

        input_primitive = self.glsl_stage_consistent_layout_value(
            "geometry",
            "input topology",
            func,
            explicit_attribute_name="inputtopology",
            bare_names=input_bare_names,
            value_mapper=self.glsl_geometry_input_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        output_primitive = self.glsl_stage_consistent_layout_value(
            "geometry",
            "output topology",
            func,
            explicit_attribute_name="outputtopology",
            bare_names=output_bare_names,
            value_mapper=self.glsl_geometry_output_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="out",
        )
        max_vertices = self.glsl_stage_positive_int_layout_argument(
            "geometry", func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_stage_positive_int_layout_argument(
                "geometry",
                func,
                "maxvertexcount",
                stage_layout_qualifiers,
                "out",
                layout_name="max_vertices",
            )
        invocations = self.glsl_stage_positive_int_layout_argument(
            "geometry", func, "invocations", stage_layout_qualifiers, "in"
        )

        code = ""
        input_layout = []
        if input_primitive:
            input_layout.append(input_primitive)
        if invocations is not None:
            input_layout.append(f"invocations = {invocations}")
        if input_layout:
            code += f"layout({', '.join(input_layout)}) in;\n"

        output_layout = []
        if output_primitive:
            output_layout.append(output_primitive)
        output_layout.extend(
            self.glsl_stage_stream_layout_parts(
                stage_layout_qualifiers, "out", skip_first=True
            )
        )
        if max_vertices is not None:
            output_layout.append(f"max_vertices = {max_vertices}")
        if output_layout:
            code += f"layout({', '.join(output_layout)}) out;\n"
        for stream_layout in self.glsl_stage_stream_layout_parts(
            stage_layout_qualifiers, "out", skip_first=False
        )[1:]:
            code += f"layout({stream_layout}) out;\n"
        return code

    def glsl_stage_stream_layout_parts(
        self, stage_layout_qualifiers=None, direction=None, skip_first=False
    ):
        stream_parts = []
        for attr in self.glsl_stage_attributes(
            None, "stream", stage_layout_qualifiers, direction
        ):
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            stream_parts.append(
                f"stream = {self.glsl_attribute_argument_source(arguments[0])}"
            )
        if skip_first:
            return stream_parts[:1]
        return stream_parts

    def glsl_stage_positive_int_layout_argument(
        self,
        stage_name,
        func,
        attribute_name,
        stage_layout_qualifiers=None,
        direction=None,
        layout_name=None,
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )

        argument = arguments[0]
        literal = self.literal_int_value(argument, self.literal_int_constants)
        if not isinstance(literal, int) or isinstance(literal, bool) or literal <= 0:
            layout_name = layout_name or attribute_name
            raise ValueError(
                f"GLSL {stage_name} {layout_name} layout requires a positive "
                "integer constant, got "
                f"{self.glsl_attribute_argument_source(argument)}"
            )
        return self.generate_expression(argument)

    def glsl_attribute_argument_source(self, argument):
        try:
            return self.generate_expression(argument)
        except Exception:
            return self.attribute_value_to_string(argument)

    def generate_tessellation_control_layout(self, func, stage_layout_qualifiers=None):
        self.validate_glsl_tessellation_patchconstantfunc(func, stage_layout_qualifiers)
        vertices = self.glsl_stage_positive_int_layout_argument(
            "tessellation_control",
            func,
            "vertices",
            stage_layout_qualifiers,
            "out",
        )
        if vertices is None:
            vertices = self.glsl_stage_positive_int_layout_argument(
                "tessellation_control",
                func,
                "outputcontrolpoints",
                stage_layout_qualifiers,
                "out",
                layout_name="vertices",
            )
        if vertices is None:
            return ""
        return f"layout(vertices = {vertices}) out;\n"

    def validate_glsl_tessellation_patchconstantfunc(
        self, func, stage_layout_qualifiers=None
    ):
        for attr in self.glsl_stage_attributes(
            func, "patchconstantfunc", stage_layout_qualifiers
        ):
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                raise ValueError(
                    "GLSL tessellation_control stage patchconstantfunc "
                    "requires exactly one function name"
                )
            patch_name = self.attribute_value_to_string(arguments[0])
            raise ValueError(
                "GLSL tessellation_control stage patchconstantfunc "
                f"'{patch_name}' is unsupported; write tessellation factors "
                "directly to gl_TessLevelOuter and gl_TessLevelInner in the "
                "tessellation_control entry point"
            )

    def generate_tessellation_evaluation_layout(
        self, func, stage_layout_qualifiers=None
    ):
        layout_parts = []
        mapped_domain = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "domain",
            func,
            explicit_attribute_name="domain",
            bare_names={"triangles", "quads", "isolines"},
            value_mapper=self.glsl_tessellation_domain,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        if mapped_domain:
            layout_parts.append(mapped_domain)

        partitioning = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "partitioning",
            func,
            explicit_attribute_name="partitioning",
            bare_names={
                "equal_spacing",
                "fractional_even_spacing",
                "fractional_odd_spacing",
            },
            value_mapper=self.glsl_tessellation_partitioning,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        if partitioning:
            layout_parts.append(partitioning)

        winding = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "winding",
            func,
            bare_names={"cw", "ccw"},
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        output_topology = self.glsl_single_stage_attribute_argument(
            func, "outputtopology", stage_layout_qualifiers
        )
        topology_winding, topology_point_mode = (
            self.glsl_tessellation_output_topology_layout(
                output_topology, mapped_domain
            )
        )
        if topology_winding:
            if winding and winding != topology_winding:
                raise ValueError(
                    "Conflicting GLSL tessellation outputtopology "
                    f"{output_topology} with winding {winding}"
                )
            winding = topology_winding
        if winding:
            layout_parts.append(winding)

        point_mode = self.glsl_stage_bare_attribute(
            func, {"point_mode"}, stage_layout_qualifiers, "in"
        )
        if point_mode or topology_point_mode:
            layout_parts.append("point_mode")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) in;\n"

    def generate_mesh_stage_layout(self, func, stage_layout_qualifiers=None):
        output_primitive = self.glsl_mesh_stage_output_topology_name(
            func, stage_layout_qualifiers
        )

        max_vertices = self.glsl_stage_consistent_positive_int_layout_argument(
            "mesh",
            func,
            ("max_vertices", "maxvertexcount"),
            stage_layout_qualifiers,
            "out",
            layout_name="max_vertices",
        )

        max_primitives = self.glsl_stage_consistent_positive_int_layout_argument(
            "mesh",
            func,
            ("max_primitives", "maxprimitivecount"),
            stage_layout_qualifiers,
            "out",
            layout_name="max_primitives",
        )

        layout_parts = []
        if output_primitive:
            layout_parts.append(output_primitive)
        if max_vertices is not None:
            layout_parts.append(f"max_vertices = {max_vertices}")
        if max_primitives is not None:
            layout_parts.append(f"max_primitives = {max_primitives}")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) out;\n"

    def glsl_mesh_stage_output_topology_name(self, func, stage_layout_qualifiers=None):
        return self.glsl_stage_consistent_layout_value(
            "mesh",
            "output topology",
            func,
            explicit_attribute_name="outputtopology",
            bare_names={"points", "lines", "triangles"},
            value_mapper=self.glsl_mesh_output_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="out",
        )

    def glsl_stage_bare_attribute(
        self, func, names, stage_layout_qualifiers=None, direction=None
    ):
        names = set(names)
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names:
                if getattr(attr, "arguments", []):
                    raise ValueError(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                return attr_name
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names:
                if getattr(attr, "arguments", []):
                    raise ValueError(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                return attr_name
        return None

    def glsl_stage_consistent_layout_value(
        self,
        stage_name,
        layout_name,
        func,
        explicit_attribute_name=None,
        bare_names=None,
        value_mapper=None,
        stage_layout_qualifiers=None,
        direction=None,
    ):
        bare_names = set(bare_names or ())
        choices = []
        for attr in self.glsl_stage_control_attributes(
            func, stage_layout_qualifiers, direction
        ):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if explicit_attribute_name is not None and (
                attr_name == explicit_attribute_name
            ):
                arguments = getattr(attr, "arguments", []) or []
                if len(arguments) != 1:
                    raise ValueError(
                        "GLSL stage attribute "
                        f"{explicit_attribute_name} requires exactly one argument"
                    )
                value = self.attribute_value_to_string(arguments[0])
                source = f"{attr_name} {value}"
            elif attr_name in bare_names:
                if getattr(attr, "arguments", []):
                    raise ValueError(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                value = attr_name
                source = attr_name
            else:
                continue

            mapped_value = value_mapper(value) if value_mapper else value
            choices.append((source, mapped_value))

        if not choices:
            return None

        first_source, first_mapped = choices[0]
        for source, mapped_value in choices[1:]:
            if mapped_value != first_mapped:
                raise ValueError(
                    f"Conflicting GLSL {stage_name} {layout_name} layout "
                    f"{first_source} with {source}"
                )
        return first_mapped

    def glsl_stage_consistent_positive_int_layout_argument(
        self,
        stage_name,
        func,
        attribute_names,
        stage_layout_qualifiers=None,
        direction=None,
        layout_name=None,
    ):
        layout_name = layout_name or attribute_names[0]
        choices = []
        for attribute_name in attribute_names:
            arguments = self.glsl_stage_attribute_arguments(
                func, attribute_name, stage_layout_qualifiers, direction
            )
            if not arguments:
                continue
            if len(arguments) != 1:
                raise ValueError(
                    f"GLSL stage attribute {attribute_name} requires exactly one argument"
                )

            argument = arguments[0]
            literal = self.literal_int_value(argument, self.literal_int_constants)
            source = self.glsl_attribute_argument_source(argument)
            if (
                not isinstance(literal, int)
                or isinstance(literal, bool)
                or literal <= 0
            ):
                raise ValueError(
                    f"GLSL {stage_name} {layout_name} layout requires a positive "
                    f"integer constant, got {source}"
                )
            choices.append(
                (
                    f"{attribute_name} {source}",
                    literal,
                    self.generate_expression(argument),
                )
            )

        if not choices:
            return None

        first_source, first_literal, first_expression = choices[0]
        for source, literal, _expression in choices[1:]:
            if literal != first_literal:
                raise ValueError(
                    f"Conflicting GLSL {stage_name} {layout_name} layout "
                    f"{first_source} with {source}"
                )
        return first_expression

    def glsl_stage_control_attributes(
        self, func, stage_layout_qualifiers=None, direction=None
    ):
        yield from getattr(func, "attributes", []) or []
        yield from self.glsl_stage_layout_entries(stage_layout_qualifiers, direction)

    def glsl_stage_has_attribute(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        return any(
            True
            for _ in self.glsl_stage_attributes(
                func, attribute_name, stage_layout_qualifiers, direction
            )
        )

    def glsl_stage_has_bare_attribute(
        self, func, names, stage_layout_qualifiers=None, direction=None
    ):
        names = set(names)
        for attr in self.glsl_stage_control_attributes(
            func, stage_layout_qualifiers, direction
        ):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names and not getattr(attr, "arguments", []):
                return True
        return False

    def glsl_single_stage_attribute_argument(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def glsl_stage_attribute_arguments(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        for attr in self.glsl_stage_attributes(
            func, attribute_name, stage_layout_qualifiers, direction
        ):
            return getattr(attr, "arguments", []) or []
        return []

    def glsl_stage_attributes(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        for attr in getattr(func, "attributes", []) or []:
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                yield attr
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                yield attr

    def glsl_stage_layout_entries(self, stage_layout_qualifiers, direction=None):
        if direction is not None:
            direction = str(direction).lower()
        for layout in stage_layout_qualifiers or []:
            layout_direction = getattr(layout, "direction", None)
            if layout_direction is not None:
                layout_direction = str(layout_direction).lower()
            if direction is not None and layout_direction != direction:
                continue
            yield from getattr(layout, "entries", []) or []

    def glsl_geometry_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "line_strip",
            "lines": "line_strip",
            "line_strip": "line_strip",
            "triangle": "triangle_strip",
            "triangles": "triangle_strip",
            "triangle_strip": "triangle_strip",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(
                f"GLSL geometry output topology cannot be lowered: {topology}"
            )
        return mapped

    def glsl_geometry_input_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "lines",
            "lines": "lines",
            "line_adjacency": "lines_adjacency",
            "lines_adjacency": "lines_adjacency",
            "triangle": "triangles",
            "triangles": "triangles",
            "triangle_adjacency": "triangles_adjacency",
            "triangles_adjacency": "triangles_adjacency",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(
                f"GLSL geometry input topology cannot be lowered: {topology}"
            )
        return mapped

    def glsl_mesh_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "lines",
            "lines": "lines",
            "triangle": "triangles",
            "triangles": "triangles",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise ValueError(f"GLSL mesh output topology cannot be lowered: {topology}")
        return mapped

    def map_mesh_intrinsic(self, operation):
        mesh_intrinsics = {
            "SetMeshOutputCounts": "SetMeshOutputsEXT",
            "DispatchMesh": "EmitMeshTasksEXT",
        }
        return mesh_intrinsics.get(operation, operation)

    def is_task_payload_shared_variable(self, node):
        return (
            "taskPayloadSharedEXT" in self.glsl_variable_qualifier_prefix(node).split()
        )

    def glsl_task_payload_shared_variable_infos(self, nodes):
        infos = []
        for node in nodes:
            if not self.is_task_payload_shared_variable(node):
                continue
            name = self.resource_node_name(node)
            if not name:
                continue
            vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
            infos.append(
                {
                    "name": name,
                    "type": self.map_type(vtype),
                    "raw_type": self.type_name_string(vtype),
                    "array_suffix": array_suffix,
                }
            )
        return infos

    def glsl_dispatch_mesh_payload_target(self, payload_expr):
        if not self.task_payload_shared_variables:
            raise ValueError(
                "GLSL DispatchMesh payload argument requires "
                "taskPayloadSharedEXT storage"
            )

        payload_name = self.expression_name(payload_expr)
        for info in self.task_payload_shared_variables:
            if payload_name == info["name"]:
                return info

        payload_type = self.expression_result_type(payload_expr)
        mapped_payload_type = self.map_type(payload_type) if payload_type else None
        if mapped_payload_type:
            matches = [
                info
                for info in self.task_payload_shared_variables
                if info["type"] == mapped_payload_type
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                names = ", ".join(sorted(info["name"] for info in matches))
                raise ValueError(
                    "Ambiguous GLSL DispatchMesh payload target for "
                    f"{mapped_payload_type}: {names}"
                )
            expected_types = ", ".join(
                sorted({info["type"] for info in self.task_payload_shared_variables})
            )
            raise ValueError(
                "GLSL DispatchMesh payload argument type "
                f"{mapped_payload_type} does not match taskPayloadSharedEXT "
                f"payload type {expected_types}"
            )

        if len(self.task_payload_shared_variables) == 1:
            return self.task_payload_shared_variables[0]

        names = ", ".join(
            sorted(info["name"] for info in self.task_payload_shared_variables)
        )
        raise ValueError(f"Ambiguous GLSL DispatchMesh payload target: {names}")

    def mesh_output_parameter_role(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            if normalized in {"vertices", "vertex", "mesh_vertices"}:
                return "vertices"
            if normalized in {"indices", "index", "primitive_indices"}:
                return "indices"
            if normalized in {"primitives", "primitive", "mesh_primitives"}:
                return "primitives"
        return None

    def is_mesh_output_parameter(self, node):
        if self.mesh_output_parameter_role(node) is None:
            return False
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        return bool(qualifiers & {"out", "inout"})

    def mesh_output_parameter_array_shape(self, node):
        node_type = self.resource_node_type(node)
        if (
            hasattr(node_type, "element_type")
            and str(type(node_type)).find("ArrayType") != -1
        ):
            return node_type.element_type, getattr(node_type, "size", None)
        return node_type, None

    def mesh_output_parameter_element_type(self, node):
        element_type, _ = self.mesh_output_parameter_array_shape(node)
        return self.type_name_string(element_type)

    def mesh_output_parameter_count(self, node):
        _, size = self.mesh_output_parameter_array_shape(node)
        return self.generate_expression(size) if size is not None else None

    def mesh_output_parameter_count_value(self, node):
        _, size = self.mesh_output_parameter_array_shape(node)
        if size is None:
            return None
        value = self.literal_int_value(size)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def mesh_primitive_index_builtin(self, topology):
        return {
            "points": "gl_PrimitivePointIndicesEXT",
            "lines": "gl_PrimitiveLineIndicesEXT",
            "triangles": "gl_PrimitiveTriangleIndicesEXT",
        }.get(topology)

    def glsl_stage_attribute_literal_int(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )
        value = self.literal_int_value(arguments[0])
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def glsl_mesh_output_count_limits(self, func, stage_layout_qualifiers=None):
        max_vertices = self.glsl_stage_attribute_literal_int(
            func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_stage_attribute_literal_int(
                func, "maxvertexcount", stage_layout_qualifiers, "out"
            )

        max_primitives = self.glsl_stage_attribute_literal_int(
            func, "max_primitives", stage_layout_qualifiers, "out"
        )
        if max_primitives is None:
            max_primitives = self.glsl_stage_attribute_literal_int(
                func, "maxprimitivecount", stage_layout_qualifiers, "out"
            )

        return {
            "vertices": max_vertices,
            "indices": max_primitives,
            "primitives": max_primitives,
        }

    def mesh_output_role_count_attribute(self, role):
        return "max_vertices" if role == "vertices" else "max_primitives"

    def validate_mesh_output_parameter_shape(
        self, node, role, element_type, count, count_value, count_limit, topology
    ):
        if count is None:
            raise ValueError(
                "GLSL mesh output parameter "
                f"@{role} '{node.name}' requires an array with explicit size"
            )

        if count_value is not None and count_limit is not None:
            if count_value != count_limit:
                attribute_name = self.mesh_output_role_count_attribute(role)
                raise ValueError(
                    "GLSL mesh output parameter "
                    f"@{role} '{node.name}' array size {count_value} must match "
                    f"{attribute_name} {count_limit}"
                )

        if role in {"vertices", "primitives"}:
            if element_type not in self.structs_by_name:
                raise ValueError(
                    "GLSL mesh output parameter "
                    f"@{role} '{node.name}' requires a struct element type, "
                    f"got {element_type}"
                )
            return

        expected_index_type = self.GLSL_MESH_PRIMITIVE_INDEX_TYPES.get(topology)
        if expected_index_type is not None and element_type != expected_index_type:
            raise ValueError(
                "GLSL mesh output parameter "
                f"@indices '{node.name}' for {topology} topology requires "
                f"{expected_index_type} elements, got {element_type}"
            )

    def mesh_output_parameter_infos(
        self, func, shader_type, stage_layout_qualifiers=None
    ):
        if shader_type != "mesh":
            return {}

        topology = self.glsl_mesh_stage_output_topology_name(
            func, stage_layout_qualifiers
        )
        count_limits = self.glsl_mesh_output_count_limits(func, stage_layout_qualifiers)
        infos = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if not self.is_mesh_output_parameter(param):
                continue

            role = self.mesh_output_parameter_role(param)
            element_type = self.mesh_output_parameter_element_type(param)
            count = self.mesh_output_parameter_count(param)
            count_value = self.mesh_output_parameter_count_value(param)
            self.validate_mesh_output_parameter_shape(
                param,
                role,
                element_type,
                count,
                count_value,
                count_limits.get(role),
                topology,
            )
            info = {
                "name": param.name,
                "role": role,
                "element_type": element_type,
                "count": count,
                "count_value": count_value,
                "topology": topology,
                "index_builtin": self.mesh_primitive_index_builtin(topology),
                "members": {},
            }

            struct = self.structs_by_name.get(element_type)
            if struct is not None:
                for member in getattr(struct, "members", []) or []:
                    mapped_semantic = self.map_semantic(self.semantic_from_node(member))
                    info["members"][member.name] = {
                        "node": member,
                        "type": self.member_type_name(member),
                        "output_name": member.name,
                        "semantic": mapped_semantic,
                    }

            infos[param.name] = info
        return infos

    def generate_mesh_output_parameter_declarations(self, mesh_outputs):
        code = ""
        emitted = set()
        for info in mesh_outputs.values():
            role = info["role"]
            if role == "indices":
                continue
            for member_name, member_info in info["members"].items():
                semantic = member_info["semantic"]
                if role == "vertices" and self.is_vertex_builtin_output(semantic):
                    continue
                if role == "primitives" and self.is_mesh_primitive_builtin(semantic):
                    continue

                member = member_info["node"]
                output_name = member_info["output_name"]
                declaration_key = (role, output_name)
                if declaration_key in emitted:
                    continue
                emitted.add(declaration_key)

                member_type = member_info["type"]
                count = info["count"]
                array_type = (
                    f"{member_type}[{count}]"
                    if count is not None
                    else f"{member_type}[]"
                )
                declaration = format_c_style_array_declaration(array_type, output_name)
                prefix = self.mesh_output_declaration_prefix(member, role)
                code += f"{prefix} {declaration};\n"
        return code

    def mesh_output_declaration_prefix(self, member, role):
        prefix = self.stage_io_declaration_prefix(member, "out")
        if role != "primitives":
            return prefix

        parts = prefix.split()
        if "perprimitiveEXT" in parts:
            return prefix
        if "out" in parts:
            parts.insert(parts.index("out"), "perprimitiveEXT")
            return " ".join(parts)
        return f"{prefix} perprimitiveEXT".strip()

    def is_mesh_primitive_builtin(self, semantic):
        return semantic in {
            "gl_PrimitiveID",
            "gl_Layer",
            "gl_ViewportIndex",
            "gl_CullPrimitiveEXT",
        }

    def map_ray_tracing_intrinsic(self, operation, args):
        ray_intrinsics = {
            "TraceRay": "traceRayEXT",
            "ReportHit": "reportIntersectionEXT",
            "CallShader": "executeCallableEXT",
            "AcceptHitAndEndSearch": "terminateRayEXT",
            "IgnoreHit": "ignoreIntersectionEXT",
        }
        mapped = ray_intrinsics.get(operation, operation)
        if mapped in {"ignoreIntersectionEXT", "terminateRayEXT"} and not args:
            return mapped
        return f"{mapped}({', '.join(args)})"

    def map_ray_query_intrinsic(self, operation, query, args):
        mapping = self.RAY_QUERY_METHOD_MAP.get(operation)
        query_expr = self.generate_expression(query)
        if mapping is None:
            generated_args = [self.generate_expression(arg) for arg in args]
            return f"{query_expr}.{operation}({', '.join(generated_args)})"
        if not self.can_lower_ray_query_receiver(operation, query):
            generated_args = [self.generate_expression(arg) for arg in args]
            return f"{query_expr}.{operation}({', '.join(generated_args)})"

        self.validate_ray_query_intrinsic_arguments(operation, args)
        function_name, committed = mapping
        call_args = [query_expr]
        if committed is not None:
            call_args.append(committed)
        args = self.ray_query_initialize_arguments(operation, args)
        args = [self.generate_expression(arg) for arg in args]
        call_args.extend(args)
        return f"{function_name}({', '.join(call_args)})"

    def can_lower_ray_query_receiver(self, operation, query):
        receiver_type = self.expression_result_type(query)
        if self.is_ray_query_type(receiver_type):
            return True
        if receiver_type is None:
            return False
        raise ValueError(
            f"GLSL ray query {operation} requires a RayQuery receiver, "
            f"got {self.type_name_string(receiver_type)}"
        )

    def validate_ray_query_intrinsic_arguments(self, operation, args):
        allowed_arities = self.RAY_QUERY_METHOD_ARITIES.get(operation)
        if allowed_arities is None:
            return
        actual_arity = len(args)
        if actual_arity in allowed_arities:
            return
        if operation in self.RAY_QUERY_POSITION_FETCH_METHODS:
            raise ValueError(
                f"GLSL ray query {operation} requires exactly one "
                f"output-array argument, got {actual_arity}"
            )
        raise ValueError(
            f"GLSL ray query {operation} expects "
            f"{self.ray_query_arity_label(allowed_arities)}, got {actual_arity}"
        )

    def ray_query_arity_label(self, allowed_arities):
        if len(allowed_arities) == 1:
            arity = allowed_arities[0]
            plural = "" if arity == 1 else "s"
            return f"{arity} argument{plural}"
        return f"{' or '.join(str(arity) for arity in allowed_arities)} arguments"

    def ray_query_initialize_arguments(self, operation, args):
        if operation not in {"Initialize", "TraceRayInline"} or len(args) != 4:
            return args

        acceleration, ray_flags, cull_mask, ray_desc = args
        return [
            acceleration,
            ray_flags,
            cull_mask,
            *[
                MemberAccessNode(
                    ray_desc,
                    self.ray_desc_field_name(ray_desc, field_names),
                )
                for field_names in self.RAY_QUERY_RAY_DESC_FIELDS
            ],
        ]

    def ray_desc_field_name(self, ray_desc, field_names):
        ray_desc_type = self.type_name_string(self.expression_result_type(ray_desc))
        members = self.struct_member_types.get(ray_desc_type, {})
        for field_name in field_names:
            if field_name in members:
                return field_name
        return field_names[0]

    def glsl_tessellation_domain(self, domain):
        domain_name = str(domain).strip().strip('"').lower()
        domain_map = {
            "tri": "triangles",
            "triangle": "triangles",
            "triangles": "triangles",
            "quad": "quads",
            "quads": "quads",
            "isoline": "isolines",
            "isolines": "isolines",
        }
        mapped = domain_map.get(domain_name)
        if mapped is None:
            raise ValueError(f"GLSL tessellation domain cannot be lowered: {domain}")
        return mapped

    def glsl_tessellation_partitioning(self, partitioning):
        partition_name = str(partitioning).strip().strip('"').lower()
        partition_map = {
            "equal": "equal_spacing",
            "equal_spacing": "equal_spacing",
            "fractional_even": "fractional_even_spacing",
            "fractional_even_spacing": "fractional_even_spacing",
            "fractional_odd": "fractional_odd_spacing",
            "fractional_odd_spacing": "fractional_odd_spacing",
        }
        mapped = partition_map.get(partition_name)
        if mapped is None:
            raise ValueError(
                f"GLSL tessellation partitioning cannot be lowered: {partitioning}"
            )
        return mapped

    def glsl_tessellation_output_topology_layout(self, topology, domain):
        if topology is None:
            return None, False

        topology_name = str(topology).strip().strip('"').lower()
        if topology_name in {"triangle_cw", "triangles_cw"}:
            if domain != "triangles":
                raise ValueError(
                    "GLSL tessellation outputtopology triangle_cw requires "
                    "triangles domain"
                )
            return "cw", False
        if topology_name in {"triangle_ccw", "triangles_ccw"}:
            if domain != "triangles":
                raise ValueError(
                    "GLSL tessellation outputtopology triangle_ccw requires "
                    "triangles domain"
                )
            return "ccw", False
        if topology_name in {"line", "lines"}:
            if domain != "isolines":
                raise ValueError(
                    "GLSL tessellation outputtopology line requires isolines domain"
                )
            return None, False
        if topology_name in {"point", "points"}:
            return None, True

        raise ValueError(
            f"GLSL tessellation outputtopology cannot be lowered: {topology}"
        )

    def structured_buffer_operation_access_requirement(self, func_name):
        if func_name == "buffer_load":
            return "read"
        if func_name in {"buffer_store", "buffer_append"}:
            return "write"
        if func_name == "buffer_consume":
            return "read_write"
        return None

    def merge_access_requirement(self, current, incoming):
        if incoming is None:
            return current
        if current is None or current == incoming:
            return incoming
        return "read_write"

    def collect_function_structured_buffer_access_requirements(self, functions):
        functions = list(functions)
        requirements = {
            getattr(func, "name", None): {}
            for func in functions
            if getattr(func, "name", None)
        }
        parameter_names = {
            func_name: list(names)
            for func_name, names in self.function_parameter_names.items()
        }
        parameter_sets = {
            func_name: set(names) for func_name, names in parameter_names.items()
        }

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_set = parameter_sets.get(func_name, set())
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                operation = self.function_call_name(node)
                required_access = self.structured_buffer_operation_access_requirement(
                    operation
                )
                if required_access is None:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if not args:
                    continue
                target_name = self.expression_name(args[0])
                if target_name not in parameter_set:
                    continue
                current = requirements[func_name].get(target_name)
                requirements[func_name][target_name] = self.merge_access_requirement(
                    current, required_access
                )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                parameter_set = parameter_sets.get(func_name, set())
                if not parameter_set:
                    continue
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    callee_requirements = requirements.get(callee_name)
                    if not callee_requirements:
                        continue
                    callee_parameters = parameter_names.get(callee_name, [])
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for callee_param, required_access in callee_requirements.items():
                        try:
                            index = callee_parameters.index(callee_param)
                        except ValueError:
                            continue
                        if index >= len(args):
                            continue
                        target_name = self.expression_name(args[index])
                        if target_name not in parameter_set:
                            continue
                        current = requirements[func_name].get(target_name)
                        merged = self.merge_access_requirement(current, required_access)
                        if merged != current:
                            requirements[func_name][target_name] = merged
                            changed = True

        return {name: reqs for name, reqs in requirements.items() if reqs}

    def validate_glsl_tessellation_control_barriers(
        self,
        ast,
        target_stage,
        functions,
    ):
        """Reject tessellation-control barriers GLSL cannot legally place."""
        if target_stage is not None and target_stage != "tessellation_control":
            return

        entry_records = collect_stage_entry_records(
            ast,
            target_stage,
            self.combined_stage_entry_types(),
        )
        entry_functions = [
            func
            for _func_id, stage_name, func in entry_records
            if stage_name == "tessellation_control"
        ]
        if not entry_functions:
            return

        entry_ids = {id(func) for func in entry_functions}
        for func in entry_functions:
            self.validate_glsl_tessellation_control_entry_barriers(func)

        for func in self.glsl_tessellation_control_helper_functions(
            ast,
            target_stage,
            functions,
            entry_ids,
        ):
            barrier_name = self.glsl_tessellation_control_first_native_barrier(
                getattr(func, "body", [])
            )
            if barrier_name is None:
                continue
            raise ValueError(
                "GLSL tessellation_control barrier() may only be emitted "
                "directly in main(); helper function "
                f"'{func.name}' reaches '{barrier_name}'"
            )

    def glsl_tessellation_control_helper_functions(
        self,
        ast,
        target_stage,
        functions,
        entry_ids,
    ):
        helpers = []
        seen = set()

        def add_helper(func):
            func_id = id(func)
            if func_id in entry_ids or func_id in seen:
                return
            seen.add(func_id)
            helpers.append(func)

        stage_entry_types = self.combined_stage_entry_types()
        top_level_function_ids = {
            id(func) for func in getattr(ast, "functions", []) or []
        }

        for func in functions:
            if id(func) not in top_level_function_ids:
                continue
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name in stage_entry_types:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                add_helper(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name != "tessellation_control" or not stage_matches(
                target_stage,
                stage_name,
            ):
                continue
            for func in getattr(stage, "local_functions", []) or []:
                add_helper(func)

        return helpers

    def validate_glsl_tessellation_control_entry_barriers(self, func):
        self.validate_glsl_tessellation_control_barrier_statement_sequence(
            self.glsl_statement_sequence(getattr(func, "body", [])),
            return_seen=False,
            control_depth=0,
        )

    def glsl_statement_sequence(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return list(body.statements)
        if isinstance(body, list):
            return body
        return [body]

    def validate_glsl_tessellation_control_barrier_statement_sequence(
        self,
        statements,
        return_seen,
        control_depth,
    ):
        for stmt in statements:
            return_seen = self.validate_glsl_tessellation_control_barrier_statement(
                stmt,
                return_seen,
                control_depth,
            )
        return return_seen

    def validate_glsl_tessellation_control_barrier_statement(
        self,
        stmt,
        return_seen,
        control_depth,
    ):
        barrier_name = self.glsl_tessellation_control_standalone_barrier_statement(stmt)
        if barrier_name is not None:
            if control_depth:
                raise ValueError(
                    "GLSL tessellation_control barrier() must not be emitted "
                    f"inside control flow; found '{barrier_name}'"
                )
            if return_seen:
                raise ValueError(
                    "GLSL tessellation_control barrier() cannot be emitted "
                    f"after a return statement in main(); found '{barrier_name}'"
                )
            return return_seen

        if isinstance(stmt, ReturnNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "value", None)
            )
            return True

        if isinstance(stmt, BlockNode):
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(stmt),
                return_seen,
                control_depth,
            )

        if isinstance(stmt, IfNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", getattr(stmt, "if_condition", None))
            )
            return_seen = (
                self.validate_glsl_tessellation_control_barrier_statement_sequence(
                    self.glsl_statement_sequence(
                        getattr(stmt, "if_body", getattr(stmt, "then_branch", []))
                    ),
                    return_seen,
                    control_depth + 1,
                )
            )
            for condition, body in zip(
                getattr(stmt, "else_if_conditions", []) or [],
                getattr(stmt, "else_if_bodies", []) or [],
            ):
                self.validate_glsl_tessellation_control_nonstandalone_barriers(
                    condition
                )
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(body),
                        return_seen,
                        control_depth + 1,
                    )
                )
            else_body = getattr(stmt, "else_body", getattr(stmt, "else_branch", None))
            if else_body is not None:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(else_body),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        if isinstance(stmt, ForNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "init", None)
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "update", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, ForInNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "iterable", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, WhileNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, DoWhileNode):
            return_seen = (
                self.validate_glsl_tessellation_control_barrier_statement_sequence(
                    self.glsl_statement_sequence(getattr(stmt, "body", [])),
                    return_seen,
                    control_depth + 1,
                )
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            return return_seen

        if isinstance(stmt, LoopNode):
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, SwitchNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "expression", None)
            )
            for case in getattr(stmt, "cases", []) or []:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(getattr(case, "statements", [])),
                        return_seen,
                        control_depth + 1,
                    )
                )
            default_case = getattr(stmt, "default_case", None)
            if default_case is not None:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(
                            getattr(default_case, "statements", default_case)
                        ),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        if isinstance(stmt, MatchNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "expression", None)
            )
            for arm in getattr(stmt, "arms", []) or []:
                self.validate_glsl_tessellation_control_nonstandalone_barriers(
                    getattr(arm, "guard", None)
                )
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(getattr(arm, "body", [])),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        self.validate_glsl_tessellation_control_nonstandalone_barriers(stmt)
        return return_seen

    def validate_glsl_tessellation_control_nonstandalone_barriers(self, node):
        barrier_name = self.glsl_tessellation_control_first_native_barrier(node)
        if barrier_name is None:
            return
        raise ValueError(
            "GLSL tessellation_control barrier() must be emitted as a "
            f"standalone statement in main(); found '{barrier_name}'"
        )

    def glsl_tessellation_control_standalone_barrier_statement(self, stmt):
        expression = (
            stmt.expression if isinstance(stmt, ExpressionStatementNode) else stmt
        )
        if not isinstance(expression, FunctionCallNode):
            return None
        return self.glsl_tessellation_control_native_barrier_call_name(expression)

    def glsl_tessellation_control_first_native_barrier(self, node):
        for child in self.walk_ast(node):
            if not isinstance(child, FunctionCallNode):
                continue
            barrier_name = self.glsl_tessellation_control_native_barrier_call_name(
                child
            )
            if barrier_name is not None:
                return barrier_name
        return None

    def glsl_tessellation_control_native_barrier_call_name(self, call):
        func_name = self.function_call_name(call)
        if func_name not in self.GLSL_TESSELLATION_CONTROL_BARRIER_BUILTINS:
            return None
        if func_name in self.function_return_types:
            return None
        args = getattr(call, "arguments", getattr(call, "args", [])) or []
        if args:
            return None
        return func_name

    def glsl_parameter_qualifiers(self, parameter):
        qualifiers = []
        seen = set()
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in self.GLSL_PARAMETER_QUALIFIERS and normalized not in seen:
                qualifiers.append(normalized)
                seen.add(normalized)
        return qualifiers

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        stage_layout_qualifiers=None,
        stage_context=None,
    ):
        """Render a function or GLSL ``main`` stage entry point."""
        code = ""
        code += "  " * indent

        param_list = getattr(func, "parameters", getattr(func, "params", []))
        params = []
        sampler_parameters = set()
        texture_parameters = {}
        image_format_parameters = {}
        image_access_parameters = {}
        resource_aliases = getattr(func, "_glsl_resource_aliases", {}) or {}
        unsupported_buffer_array_info = (
            self.unsupported_structured_buffer_array_functions.get(func.name, {})
        )
        unsupported_buffer_array_indices = unsupported_buffer_array_info.get(
            "indices", set()
        )
        previous_function_return_type = self.current_function_return_type
        previous_local_variable_types = self.local_variable_types
        previous_identifier_aliases = self.current_identifier_aliases
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        previous_structured_buffer_array_parameters = (
            self.current_structured_buffer_array_parameters
        )
        previous_structured_buffer_counter_parameters = (
            self.current_structured_buffer_counter_parameters
        )
        previous_structured_buffer_access_parameters = (
            self.current_structured_buffer_access_parameters
        )
        self.local_variable_types = {}
        self.current_identifier_aliases = dict(self.current_identifier_aliases)
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )
        self.current_structured_buffer_array_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_structured_buffer_access_parameters = {}
        for alias_name, binding in resource_aliases.items():
            self.local_variable_types[alias_name] = binding.get("type")
        for index, p in enumerate(param_list):
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
            self.validate_resource_access_metadata_operands(p)
            self.record_structured_buffer_access_metadata(
                p.name, raw_param_type, p, parameter=True
            )

            if index in unsupported_buffer_array_indices:
                continue

            if self.is_sampler_type(raw_param_type):
                self.explicit_resource_binding_index(p)
                sampler_parameters.add(p.name)
                if not self.should_preserve_sampler_parameter(
                    getattr(func, "name", None), p.name
                ):
                    continue

            buffer_array = self.structured_buffer_array_parameter_info(
                raw_param_type, p.name, getattr(func, "name", None)
            )
            if buffer_array is not None:
                expanded_names = [
                    f"{p.name}_{index}" for index in range(buffer_array["count"])
                ]
                buffer_info = {
                    **buffer_array,
                    "expanded_names": expanded_names,
                }
                if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                    counter_names = [
                        f"{p.name}Counter_{index}"
                        for index in range(buffer_array["count"])
                    ]
                    buffer_info["counter_expanded_names"] = counter_names
                self.current_structured_buffer_array_parameters[p.name] = buffer_info
                for expanded_name in expanded_names:
                    self.local_variable_types[expanded_name] = (
                        f"{buffer_array['element_type']}[]"
                    )
                    params.append(
                        format_c_style_array_declaration(
                            f"{buffer_array['element_type']}[]",
                            expanded_name,
                        )
                    )
                for counter_name in buffer_info.get("counter_expanded_names", []):
                    params.append(
                        format_c_style_array_declaration("uint[]", counter_name)
                    )
                continue

            param_type = self.map_resource_parameter_type_with_hint(
                raw_param_type, p, getattr(func, "name", None)
            )
            if self.is_opaque_resource_type(
                self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
            ):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                    image_accesses=image_access_parameters,
                )

            semantic = self.semantic_from_node(p)

            parameter_name = (
                self.glsl_parameter_identifier_name(p.name)
                if shader_type is None
                else p.name
            )
            declaration = format_c_style_array_declaration(param_type, parameter_name)
            if self.is_storage_image_type(param_type):
                memory_qualifiers = self.resource_memory_qualifiers(p)
                if memory_qualifiers:
                    declaration = f"{memory_qualifiers} {declaration}"
            parameter_qualifiers = self.glsl_parameter_qualifiers(p)
            if parameter_qualifiers:
                declaration = f"{' '.join(parameter_qualifiers)} {declaration}"
            semantic_attr = self.map_semantic(semantic)
            params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                params.append(format_c_style_array_declaration("uint[]", counter_name))

        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        elif shader_type is not None:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)

        params_str = ", ".join(params)
        if getattr(func, "body", []) is None and shader_type is None:
            return_type = self.map_type(
                self.type_name_string(getattr(func, "return_type", None)) or "void"
            )
            self.current_function_return_type = previous_function_return_type
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.current_structured_buffer_array_parameters = (
                previous_structured_buffer_array_parameters
            )
            self.current_structured_buffer_counter_parameters = (
                previous_structured_buffer_counter_parameters
            )
            self.current_structured_buffer_access_parameters = (
                previous_structured_buffer_access_parameters
            )
            return f"{code}{return_type} {func.name}({params_str});\n\n"

        stage_entry_types = {
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
        }

        if shader_type is not None:
            self.validate_function_return_semantic(func, shader_type)
            if shader_type == "tessellation_control":
                self.validate_glsl_tessellation_control_entry_barriers(func)
        elif stage_context == "tessellation_control":
            barrier_name = self.glsl_tessellation_control_first_native_barrier(
                getattr(func, "body", [])
            )
            if barrier_name is not None:
                raise ValueError(
                    "GLSL tessellation_control barrier() may only be emitted "
                    "directly in main(); helper function "
                    f"'{func.name}' reaches '{barrier_name}'"
                )

        mesh_output_parameters = self.mesh_output_parameter_infos(
            func, shader_type, stage_layout_qualifiers
        )
        mesh_output_topology = (
            self.glsl_mesh_stage_output_topology_name(func, stage_layout_qualifiers)
            if shader_type == "mesh"
            else None
        )
        stage_output = self.stage_return_output(func, shader_type)
        if stage_output and stage_output["declaration"]:
            code += f"{stage_output['declaration']}\n"
        if mesh_output_parameters:
            code += self.generate_mesh_output_parameter_declarations(
                mesh_output_parameters
            )

        stage_layout = self.generate_stage_layout(
            shader_type, func, stage_layout_qualifiers
        )
        if stage_layout:
            code += stage_layout

        is_native_stage_entry = (entry_name or "main") == "main"
        if shader_type in {"compute", "mesh", "task"} and is_native_stage_entry:
            code += self.generate_compute_layout(execution_config)

        if shader_type in stage_entry_types:
            code += f"void {entry_name or 'main'}() {{\n"
            self.current_function_return_type = "void"
        else:
            raw_return_type = self.type_name_string(getattr(func, "return_type", None))
            self.current_function_return_type = raw_return_type or "void"
            return_type = self.map_type(self.current_function_return_type)
            code += f"{return_type} {func.name}({params_str}) {{\n"

        previous_sampler_parameters = self.current_sampler_parameters
        previous_texture_parameters = self.current_texture_parameters
        previous_resource_aliases = self.current_resource_aliases
        previous_image_format_parameters = self.current_image_format_parameters
        previous_image_access_parameters = self.current_image_access_parameters
        previous_stage_output = self.current_stage_output
        previous_stage_inputs = self.current_stage_inputs
        previous_stage_outputs = self.current_stage_outputs
        previous_stage_output_member_map = self.current_stage_output_member_map
        previous_stage_parameter_aliases = self.current_stage_parameter_aliases
        previous_mesh_output_parameters = self.current_mesh_output_parameters
        previous_mesh_output_topology = self.current_mesh_output_topology
        previous_mesh_output_count_limits = self.current_mesh_output_count_limits
        previous_flattened_stage_variables = self.flattened_stage_variables
        previous_stage_return_type = self.current_stage_return_type
        previous_stage_entry_type = self.current_stage_entry_type
        previous_local_variable_types = self.local_variable_types
        previous_compile_time_int_constants = self.current_compile_time_int_constants
        previous_compile_time_int_vector_constants = (
            self.current_compile_time_int_vector_constants
        )
        previous_compile_time_int_vector_array_constants = (
            self.current_compile_time_int_vector_array_constants
        )
        self.current_sampler_parameters = sampler_parameters
        self.current_texture_parameters = {
            **texture_parameters,
            **{
                alias_name: binding.get("type")
                for alias_name, binding in resource_aliases.items()
            },
        }
        self.current_resource_aliases = resource_aliases
        self.current_image_format_parameters = {
            **image_format_parameters,
            **{
                alias_name: binding.get("format")
                for alias_name, binding in resource_aliases.items()
                if binding.get("format") is not None
            },
        }
        self.current_image_access_parameters = {
            **image_access_parameters,
            **{
                alias_name: binding.get("access")
                for alias_name, binding in resource_aliases.items()
                if binding.get("access") is not None
            },
        }
        self.current_stage_output = stage_output
        self.current_stage_inputs = self.stage_input_member_maps(func, shader_type)
        self.current_stage_output_member_map = self.stage_output_member_map(
            func, shader_type
        )
        self.current_stage_outputs = self.stage_output_member_maps(func, shader_type)
        self.current_stage_parameter_aliases = self.stage_parameter_aliases(
            func, shader_type
        )
        self.apply_stage_context_entry_aliases(func, shader_type, stage_context)
        self.current_mesh_output_parameters = mesh_output_parameters
        self.current_mesh_output_topology = mesh_output_topology
        self.current_mesh_output_count_limits = (
            self.glsl_mesh_output_count_limits(func, stage_layout_qualifiers)
            if shader_type == "mesh"
            else {}
        )
        self.flattened_stage_variables = set(self.current_stage_outputs)
        self.current_stage_return_type = self.type_node_name(
            getattr(func, "return_type", None)
        )
        self.current_stage_entry_type = shader_type or stage_context
        self.current_identifier_aliases = dict(self.current_identifier_aliases)
        self.current_compile_time_int_constants = dict(self.literal_int_constants)
        self.current_compile_time_int_vector_constants = dict(
            self.literal_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = dict(
            self.literal_int_vector_array_constants
        )
        for param in param_list:
            name = getattr(param, "name", None)
            self.current_compile_time_int_constants.pop(name, None)
            self.current_compile_time_int_vector_constants.pop(name, None)
            self.current_compile_time_int_vector_array_constants.pop(name, None)
        for declaration in self.stage_builtin_parameter_local_alias_declarations(
            func, shader_type
        ):
            code += f"    {declaration}\n"
        body = getattr(func, "body", [])
        if unsupported_buffer_array_info:
            code += self.unsupported_structured_buffer_array_function_body(
                func, unsupported_buffer_array_info, 1
            )
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, 1)
        self.current_sampler_parameters = previous_sampler_parameters
        self.current_texture_parameters = previous_texture_parameters
        self.current_resource_aliases = previous_resource_aliases
        self.current_image_format_parameters = previous_image_format_parameters
        self.current_image_access_parameters = previous_image_access_parameters
        self.current_stage_output = previous_stage_output
        self.current_stage_inputs = previous_stage_inputs
        self.current_stage_outputs = previous_stage_outputs
        self.current_stage_output_member_map = previous_stage_output_member_map
        self.current_stage_parameter_aliases = previous_stage_parameter_aliases
        self.current_mesh_output_parameters = previous_mesh_output_parameters
        self.current_mesh_output_topology = previous_mesh_output_topology
        self.current_mesh_output_count_limits = previous_mesh_output_count_limits
        self.flattened_stage_variables = previous_flattened_stage_variables
        self.current_stage_return_type = previous_stage_return_type
        self.current_stage_entry_type = previous_stage_entry_type
        self.local_variable_types = previous_local_variable_types
        self.current_identifier_aliases = previous_identifier_aliases
        self.current_compile_time_int_constants = previous_compile_time_int_constants
        self.current_compile_time_int_vector_constants = (
            previous_compile_time_int_vector_constants
        )
        self.current_compile_time_int_vector_array_constants = (
            previous_compile_time_int_vector_array_constants
        )
        self.current_function_return_type = previous_function_return_type
        self.local_variable_types = previous_local_variable_types
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        self.current_structured_buffer_array_parameters = (
            previous_structured_buffer_array_parameters
        )
        self.current_structured_buffer_counter_parameters = (
            previous_structured_buffer_counter_parameters
        )
        self.current_structured_buffer_access_parameters = (
            previous_structured_buffer_access_parameters
        )

        code += "}\n\n"
        return code

    def stage_functions(self, ast, stage_name):
        functions = []

        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", []) or []
            qualifier = (
                qualifiers[0] if qualifiers else getattr(func, "qualifier", None)
            )
            if normalize_stage_name(qualifier) == stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            current_stage = normalize_stage_name(stage_type)
            if current_stage == stage_name and hasattr(stage, "entry_point"):
                functions.append(stage.entry_point)

        return functions

    def stage_entry_function_ids(self, ast):
        stage_names = {
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
        }
        entry_ids = set()
        for stage_name in stage_names:
            entry_ids.update(id(func) for func in self.stage_functions(ast, stage_name))
        return entry_ids

    def type_node_references_name(self, type_node, struct_name):
        if type_node is None:
            return False
        if self.type_node_name(type_node) == struct_name:
            return True
        element_type = getattr(type_node, "element_type", None)
        if element_type is not None and self.type_node_references_name(
            element_type, struct_name
        ):
            return True
        return any(
            self.type_node_references_name(arg, struct_name)
            for arg in getattr(type_node, "generic_args", []) or []
        )

    def statement_declares_type(self, statement, struct_name):
        if isinstance(statement, VariableNode) and self.type_node_references_name(
            getattr(statement, "var_type", None), struct_name
        ):
            return True
        for child_name in ("body", "else_body", "cases", "default", "statements"):
            child = getattr(statement, child_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                if any(
                    self.statement_declares_type(item, struct_name) for item in child
                ):
                    return True
            elif self.statement_declares_type(child, struct_name):
                return True
        return False

    def function_signature_references_type(self, func, struct_name):
        if self.type_node_references_name(
            getattr(func, "return_type", None), struct_name
        ):
            return True
        return any(
            self.type_node_references_name(
                getattr(param, "param_type", None), struct_name
            )
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        )

    def function_body_declares_type(self, func, struct_name):
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        return any(
            self.statement_declares_type(stmt, struct_name) for stmt in statements
        )

    def struct_member_references_type(self, struct, struct_name):
        return any(
            self.type_node_references_name(
                getattr(member, "member_type", None), struct_name
            )
            for member in getattr(struct, "members", []) or []
        )

    def stage_io_struct_referenced_outside_stage_entries(self, ast, struct_name):
        stage_entry_ids = self.stage_entry_function_ids(ast)
        for func in getattr(ast, "functions", []) or []:
            if id(func) in stage_entry_ids:
                continue
            if self.function_signature_references_type(
                func, struct_name
            ) or self.function_body_declares_type(func, struct_name):
                return True

        for stage in getattr(ast, "stages", {}).values():
            for func in getattr(stage, "functions", []) or []:
                if id(func) in stage_entry_ids:
                    continue
                if self.function_signature_references_type(
                    func, struct_name
                ) or self.function_body_declares_type(func, struct_name):
                    return True

        for struct in getattr(ast, "structs", []) or []:
            if getattr(struct, "name", None) == struct_name:
                continue
            if self.struct_member_references_type(struct, struct_name):
                return True
        return False

    def should_emit_flattened_stage_io_struct(
        self, ast, struct_name, emitted_io, emit_graphics_io
    ):
        if not emit_graphics_io:
            return True
        if not emitted_io:
            return True
        if struct_name in self.fragment_output_struct_names:
            return True
        return self.stage_io_struct_referenced_outside_stage_entries(ast, struct_name)

    def stage_parameter_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            for param in parameters:
                type_name = self.type_node_name(getattr(param, "param_type", None))
                if type_name in self.structs_by_name:
                    struct_names.add(type_name)
        return struct_names

    def stage_return_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            type_name = self.type_node_name(getattr(func, "return_type", None))
            if type_name in self.structs_by_name:
                struct_names.add(type_name)
        return struct_names

    def collect_stage_entry_resource_parameters(self, ast, target_stage=None):
        stage_entry_types = {
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
        }
        parameters = []
        seen = set()

        def add_parameters(func):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if id(param) in seen:
                    continue
                if not self.is_stage_entry_resource_parameter(param):
                    continue
                parameters.append(param)
                seen.add(id(param))

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name not in stage_entry_types:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                add_parameters(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                add_parameters(entry_point)

        return parameters

    def collect_global_variable_types(self, nodes):
        variable_types = {}
        for node in nodes or []:
            name = self.resource_node_name(node)
            if not name:
                continue
            vtype, _array_size, array_suffix, _resource_count = (
                self.resource_declaration_shape(node)
            )
            variable_types[name] = f"{self.type_name_string(vtype)}{array_suffix}"
        return variable_types

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        base_type = self.resource_base_type(vtype)
        mapped_type = self.map_resource_type_with_format(base_type, node)
        return (
            bool(qualifiers & {"uniform", "buffer"})
            or mapped_type == "sampler"
            or self.is_opaque_resource_type(mapped_type)
            or self.is_structured_buffer_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def is_stage_local_interface_variable(self, node):
        if self.is_stage_local_resource_variable(node):
            return False
        return bool(
            self.stage_local_interface_direction(node)
            or self.glsl_variable_layout_prefix(node)
            or self.glsl_variable_qualifier_prefix(node)
        )

    def canonical_glsl_compile_time_type_text(self, type_text):
        type_text = str(type_text)
        for name, value in sorted(self.literal_int_constants.items()):
            type_text = type_text.replace(f"[{name}]", f"[{value}]")
        return type_text

    def deduplicate_stage_interface_declarations(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            if not name:
                declarations.append(node)
                continue

            vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
            signature = (
                self.canonical_glsl_compile_time_type_text(self.map_type(vtype)),
                self.canonical_glsl_compile_time_type_text(array_suffix),
                self.glsl_variable_layout_prefix(node),
                self.glsl_variable_qualifier_prefix(node),
            )
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = signature
                declarations.append(node)
                continue
            if existing != signature:
                raise ValueError(
                    "Conflicting OpenGL stage interface declaration for "
                    f"'{name}': {signature} differs from {existing}"
                )
        return declarations

    def deduplicate_resource_declaration_nodes(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            identity = self.resource_declaration_identity(node)
            if not name or identity is None:
                declarations.append(node)
                continue

            binding = self.explicit_resource_binding_index(node)
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = {
                    "identity": identity,
                    "binding": binding,
                    "index": len(declarations),
                }
                declarations.append(node)
                continue

            if existing["identity"] != identity:
                raise ValueError(
                    "Conflicting OpenGL resource declaration for "
                    f"'{name}': {identity} differs from {existing['identity']}"
                )

            existing_binding = existing["binding"]
            if (
                existing_binding is not None
                and binding is not None
                and existing_binding != binding
            ):
                raise ValueError(
                    "Conflicting OpenGL resource binding for "
                    f"'{name}': binding {binding} differs from "
                    f"existing binding {existing_binding}"
                )
            if existing_binding is None and binding is not None:
                declarations[existing["index"]] = node
                existing["binding"] = binding

        return declarations

    def is_stage_entry_resource_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.is_glsl_buffer_block_variable(param, raw_type):
            return True
        if self.is_structured_buffer_type(raw_type):
            return True
        mapped_type = self.map_resource_type_with_format(
            self.resource_base_type(raw_type), param
        )
        return self.is_opaque_resource_type(mapped_type)

    def is_stage_entry_value_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.is_stage_entry_resource_parameter(param):
            return False
        type_name = self.type_node_name(raw_type)
        return type_name not in self.structs_by_name

    def is_stage_builtin_semantic(self, semantic):
        mapped_semantic = self.map_semantic(semantic)
        return mapped_semantic.startswith("gl_")

    def parameter_has_mapped_semantic(
        self, parameter, expected_semantic, stage_name=None
    ):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        mapped = self.map_stage_input_semantic(semantic, stage_name)
        return mapped.lower() == expected_semantic.lower()

    def validate_exact_mapped_semantic_type(
        self, parameters, stage_name, semantic, expected_type, expected_description
    ):
        for parameter in parameters:
            if not self.parameter_has_mapped_semantic(parameter, semantic, stage_name):
                continue

            mapped_type = self.map_type(self.resource_node_type(parameter))
            base_type, array_suffix = split_array_type_suffix(str(mapped_type))
            if array_suffix or base_type != expected_type:
                if self.stage_builtin_parameter_local_alias_declaration(
                    parameter, stage_name
                ):
                    continue
                raise ValueError(
                    f"OpenGL {stage_name} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_compute_builtin_parameter_types(self, parameters):
        for semantic in (
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        ):
            self.validate_exact_mapped_semantic_type(
                parameters, "compute", semantic, "uvec3", "uvec3"
            )
        self.validate_exact_mapped_semantic_type(
            parameters,
            "compute",
            "gl_LocalInvocationIndex",
            "uint",
            "scalar uint",
        )

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        for semantic in ("gl_VertexID", "gl_InstanceID", "gl_PrimitiveID"):
            self.validate_exact_mapped_semantic_type(
                parameters, stage_name, semantic, "int", "scalar int"
            )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FrontFacing", "bool", "scalar bool"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FragCoord", "vec4", "vec4"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_PointCoord", "vec2", "vec2"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SampleID", "int", "scalar int"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SamplePosition", "vec2", "vec2"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SampleMaskIn", "int", "scalar int"
        )

    def validate_function_return_semantic(self, func, stage_name):
        semantic = self.function_return_semantic(func)
        if semantic is None:
            return
        mapped_semantic = self.map_semantic(semantic)
        return_type = self.function_return_type(func)
        return_base_type, array_suffix = split_array_type_suffix(str(return_type))
        if not array_suffix and return_base_type == "void":
            function_name = getattr(func, "name", "<anonymous>")
            raise ValueError(
                f"OpenGL {stage_name} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        self.validate_function_return_semantic_stage(stage_name, semantic)

        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_FrontFacing",
            "gl_FragCoord",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SamplePosition",
            "gl_SampleMaskIn",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if mapped_semantic in invalid_return_semantics:
            raise ValueError(
                f"OpenGL {stage_name} semantic '{semantic}' cannot be used as "
                "a function return semantic"
            )

    def validate_function_return_semantic_stage(self, stage_name, semantic):
        mapped_semantic = self.map_semantic(semantic)
        semantic_text = str(semantic)
        is_fragment_output = mapped_semantic in {
            "gl_FragDepth",
            "gl_FragStencilRefARB",
            "gl_SampleMask",
        } or semantic_text.startswith("gl_FragColor")
        is_vertex_output = self.is_vertex_builtin_output(mapped_semantic)

        if stage_name == "vertex" and is_fragment_output:
            raise ValueError(
                f"OpenGL vertex stage function return cannot use fragment output "
                f"semantic '{semantic}'"
            )
        if stage_name == "fragment" and is_vertex_output:
            raise ValueError(
                f"OpenGL fragment stage function return cannot use vertex output "
                f"semantic '{semantic}'"
            )
        if stage_name == "compute" and (is_fragment_output or is_vertex_output):
            raise ValueError(
                f"OpenGL compute stage function return cannot use graphics output "
                f"semantic '{semantic}'"
            )

    def generate_stage_parameter_input_declarations(self, ast, target_stage=None):
        declarations = []
        graphics_stages = {
            "vertex",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
        }

        def add_parameter(param, stage_name):
            if stage_name == "mesh" and self.is_mesh_output_parameter(param):
                return
            if self.is_graphics_stage_output_parameter(param, stage_name):
                return
            if not self.is_stage_entry_value_parameter(param):
                return
            semantic = self.semantic_from_node(param)
            if self.is_stage_builtin_semantic(semantic):
                return

            param_type = self.map_type(self.resource_node_type(param))
            layout = self.stage_io_layout_for_node(param)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                stage_name,
                "input",
                layout,
                param_type,
                param.name,
            )
            prefix = self.stage_io_declaration_prefix(param, "in")
            if self.requires_flat_stage_input(stage_name, param_type):
                prefix = self.with_flat_stage_input_qualifier(prefix)
            parameter_declaration = format_c_style_array_declaration(
                param_type, param.name
            )
            declaration = f"{prefix} {parameter_declaration};"
            if self.reserve_stage_io_declaration(
                stage_name, "input", param.name, declaration
            ):
                if target_stage is None:
                    declarations.append(
                        self.wrap_stage_guard(f"{declaration}\n", stage_name)
                    )
                else:
                    declarations.append(declaration)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name not in graphics_stages:
                continue
            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                add_parameter(param, qualifier_name)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in graphics_stages:
                continue
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            for param in (
                getattr(entry_point, "parameters", getattr(entry_point, "params", []))
                or []
            ):
                add_parameter(param, stage_name)

        if target_stage is None:
            return "".join(declarations)
        return "\n".join(declarations) + ("\n" if declarations else "")

    def generate_stage_parameter_output_declarations(self, ast, target_stage=None):
        declarations = []
        graphics_stages = {"vertex", "fragment"}

        def add_parameter(func, param, stage_name):
            if not self.is_graphics_stage_output_parameter(param, stage_name):
                return
            if not self.is_stage_entry_value_parameter(param):
                return

            param_type = self.map_type(self.resource_node_type(param))
            output_name = self.stage_output_parameter_name(func, param, stage_name)
            self.stage_parameter_output_aliases.setdefault(id(func), {})[
                param.name
            ] = output_name

            if self.is_fragment_builtin_output_target(output_name):
                return
            if self.is_vertex_builtin_output(output_name):
                return

            layout = self.stage_output_parameter_layout(param, stage_name)
            if layout:
                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    stage_name,
                    "output",
                    layout,
                    param_type,
                    output_name,
                )
            prefix = self.stage_io_declaration_prefix(param, "out", layout)
            prefix = self.stage_io_prefix_with_required_flat(prefix, param_type)
            parameter_declaration = format_c_style_array_declaration(
                param_type, output_name
            )
            declaration = f"{prefix} {parameter_declaration};"
            if self.reserve_stage_io_declaration(
                stage_name, "output", output_name, declaration
            ):
                if target_stage is None:
                    declarations.append(
                        self.wrap_stage_guard(f"{declaration}\n", stage_name)
                    )
                else:
                    declarations.append(declaration)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if hasattr(func, "qualifiers") and func.qualifiers
                else getattr(func, "qualifier", None)
            )
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name not in graphics_stages:
                continue
            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                add_parameter(func, param, qualifier_name)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in graphics_stages:
                continue
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            for param in (
                getattr(entry_point, "parameters", getattr(entry_point, "params", []))
                or []
            ):
                add_parameter(entry_point, param, stage_name)

        if target_stage is None:
            return "".join(declarations)
        return "\n".join(declarations) + ("\n" if declarations else "")

    def stage_io_layout_location(self, layout):
        value = self.stage_io_layout_value(layout, "location")
        return int(value) if value is not None and value.isdigit() else None

    def stage_io_layout_value(self, layout, name):
        if not layout.startswith("layout("):
            return None
        start = layout.find("(")
        end = layout.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return None

        for item in layout[start + 1 : end].split(","):
            key, separator, value = item.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        return None

    def stage_io_layout_int_value(self, layout, name):
        value = self.stage_io_layout_value(layout, name)
        if value is None:
            return None
        return int(value) if value.isdigit() else None

    def stage_io_layout_component(self, layout):
        return self.stage_io_layout_int_value(layout, "component")

    def stage_io_layout_index(self, layout):
        index = self.stage_io_layout_int_value(layout, "index")
        return 0 if index is None else index

    def stage_io_location_count(self, mapped_type):
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        array_count = self.stage_io_array_location_count(array_suffix)
        matrix_count = 1

        matrix_size = ""
        if base_type.startswith("dmat"):
            matrix_size = base_type[4:]
        elif base_type.startswith("mat"):
            matrix_size = base_type[3:]

        if matrix_size:
            columns = matrix_size.split("x", 1)[0]
            if columns.isdigit():
                matrix_count = int(columns)

        return max(matrix_count * array_count, 1)

    def stage_io_component_count(self, mapped_type):
        base_type, _ = split_array_type_suffix(str(mapped_type))
        normalized = str(base_type)
        if normalized in {
            "float",
            "int",
            "uint",
            "bool",
            "half",
            "short",
            "ushort",
        }:
            return 1
        if normalized == "double":
            return 2

        vector_prefix_widths = {
            "vec": 1,
            "ivec": 1,
            "uvec": 1,
            "bvec": 1,
            "dvec": 2,
        }
        for prefix, component_width in vector_prefix_widths.items():
            if normalized.startswith(prefix):
                size = normalized[len(prefix) :]
                if size.isdigit():
                    return int(size) * component_width
        return 4

    def stage_io_array_location_count(self, array_suffix):
        if not array_suffix:
            return 1

        count = 1
        index = 0
        while index < len(array_suffix):
            if array_suffix[index] != "[":
                return count
            close_index = array_suffix.find("]", index + 1)
            if close_index == -1:
                return count
            size = array_suffix[index + 1 : close_index].strip()
            if not size:
                index = close_index + 1
                continue
            try:
                dimension = int(size, 10)
            except ValueError:
                return count
            if dimension <= 0:
                return count
            count *= dimension
            index = close_index + 1
        return count

    def reserve_stage_io_layout(
        self, used_locations, stage_name, direction, layout, mapped_type, name
    ):
        start = self.stage_io_layout_location(layout)
        if start is None:
            return

        count = self.stage_io_location_count(mapped_type)
        end = start + count - 1
        component = self.stage_io_layout_component(layout)
        index = self.stage_io_layout_index(layout)
        if component is not None or self.stage_io_layout_has(layout, {"index"}):
            self.reserve_stage_io_component_layout(
                used_locations,
                stage_name,
                direction,
                start,
                end,
                component,
                index,
                mapped_type,
                name,
            )
            return

        ranges = used_locations.setdefault((stage_name, direction), [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_location_label(start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_location_label(used_start, used_end)}"
                )
        for (
            used_start,
            used_end,
            used_component_start,
            used_component_end,
            used_index,
            used_name,
        ) in self.stage_io_used_components.get((stage_name, direction), []):
            if used_index != 0:
                continue
            if start <= used_end and used_start <= end:
                raise ValueError(
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_location_label(start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_component_location_label(used_start, used_end, used_component_start, used_component_end)}"
                )
        ranges.append((start, end, name))

    def reserve_stage_io_component_layout(
        self,
        used_locations,
        stage_name,
        direction,
        start,
        end,
        component,
        index,
        mapped_type,
        name,
    ):
        component_start = 0 if component is None else component
        component_count = (
            4 if component is None else self.stage_io_component_count(mapped_type)
        )
        component_end = component_start + component_count - 1

        for used_start, used_end, used_name in used_locations.get(
            (stage_name, direction), []
        ):
            if start <= used_end and used_start <= end:
                if index != 0:
                    continue
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_component_location_label(start, end, component_start, component_end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_location_label(used_start, used_end)}"
                )

        entries = self.stage_io_used_components.setdefault((stage_name, direction), [])
        for (
            used_start,
            used_end,
            used_component_start,
            used_component_end,
            used_index,
            used_name,
        ) in entries:
            if start <= used_end and used_start <= end and index == used_index:
                if (
                    used_start == start
                    and used_end == end
                    and used_component_start == component_start
                    and used_component_end == component_end
                    and used_name == name
                ):
                    return
                if (
                    component_start <= used_component_end
                    and used_component_start <= component_end
                ):
                    raise ValueError(
                        f"Conflicting OpenGL {stage_name} {direction} location "
                        f"for '{name}': {self.stage_io_component_location_label(start, end, component_start, component_end)} "
                        f"overlaps '{used_name}' "
                        f"{self.stage_io_component_location_label(used_start, used_end, used_component_start, used_component_end)}"
                    )
        entries.append((start, end, component_start, component_end, index, name))

    def reserve_stage_io_declaration(self, stage_name, direction, name, declaration):
        key = (stage_name, direction, name)
        existing = self.stage_io_declarations.get(key)
        if existing is None:
            self.stage_io_declarations[key] = declaration
            return True
        if existing == declaration:
            return False
        raise ValueError(
            f"Conflicting OpenGL {stage_name} {direction} declaration for "
            f"'{name}': {declaration} differs from {existing}"
        )

    def stage_io_declared_names(self, stage_name, direction):
        return {
            name
            for stage, used_direction, name in self.stage_io_declarations
            if stage == stage_name and used_direction == direction
        }

    def stage_io_location_label(self, start, end):
        if start == end:
            return f"location {start}"
        return f"locations {start}-{end}"

    def stage_io_component_label(self, start, end):
        if start == end:
            return f"component {start}"
        return f"components {start}-{end}"

    def stage_io_component_location_label(
        self, start, end, component_start, component_end
    ):
        return (
            f"{self.stage_io_location_label(start, end)} "
            f"{self.stage_io_component_label(component_start, component_end)}"
        )

    def struct_member_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                names.add(member.name)
        return names

    def fragment_input_declaration_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                input_name = self.fragment_input_member_name(member, struct_name)
                if input_name is not None:
                    names.add(input_name)
        return names

    def vertex_output_declaration_names(self):
        names = set()
        for struct_name in self.vertex_output_struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                output_name = self.vertex_output_member_name(member)
                if not self.is_vertex_builtin_output(output_name):
                    names.add(output_name)
        return names

    def stage_value_parameter_input_names(self, ast, stage_name):
        names = set()
        for func in self.stage_functions(ast, stage_name):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if not self.is_stage_entry_value_parameter(param):
                    continue
                if self.is_graphics_stage_output_parameter(param, stage_name):
                    continue
                if self.is_stage_builtin_semantic(self.semantic_from_node(param)):
                    continue
                names.add(param.name)
        return names

    def parameter_qualifier_names(self, node=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        qualifiers.discard("")
        return qualifiers

    def is_graphics_stage_output_parameter(self, param, stage_name):
        if normalize_stage_name(stage_name) not in {"vertex", "fragment"}:
            return False
        return bool(self.parameter_qualifier_names(param) & {"out", "inout"})

    def stage_output_parameter_name(self, func, param, stage_name):
        semantic = self.semantic_from_node(param)
        mapped_semantic = self.map_semantic(semantic)
        if normalize_stage_name(stage_name) == "vertex":
            if self.is_vertex_builtin_output(mapped_semantic):
                return mapped_semantic
            reserved = set(self.vertex_input_member_names)
            reserved.update(self.stage_io_declared_names("vertex", "output"))
            return self.stage_io_name_avoiding_reserved(param.name, reserved)

        if self.is_fragment_builtin_output_target(mapped_semantic):
            return mapped_semantic
        if str(semantic or "").startswith("gl_FragColor"):
            return self.fragment_output_name(
                semantic, self.function_local_variable_names(func)
            )
        reserved = set(self.fragment_input_member_names)
        reserved.update(self.stage_io_declared_names("fragment", "output"))
        reserved.update(self.function_local_variable_names(func))
        return self.stage_io_name_avoiding_reserved(param.name, reserved)

    def stage_output_parameter_layout(self, param, stage_name):
        if normalize_stage_name(stage_name) != "fragment":
            return self.stage_io_layout_for_node(param)

        mapped_semantic = self.map_semantic(self.semantic_from_node(param))
        if mapped_semantic.startswith("layout("):
            return mapped_semantic
        layout = self.stage_io_layout_for_node(param)
        if layout.startswith("layout("):
            return layout
        return "layout(location = 0)"

    def stage_io_name_avoiding_reserved(self, preferred_name, reserved_names):
        if preferred_name not in reserved_names:
            return preferred_name

        base_name = f"out_{preferred_name}"
        candidate = base_name
        suffix = 1
        while candidate in reserved_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate

    def function_local_variable_names(self, func):
        body = getattr(func, "body", [])
        return {
            node.name
            for node in self.walk_ast(body)
            if isinstance(node, VariableNode) and getattr(node, "name", None)
        }

    def fragment_stage_local_variable_names(self, ast):
        names = set()
        for func in self.stage_functions(ast, "fragment"):
            names.update(self.function_local_variable_names(func))
        return names

    def fragment_output_member_maps(self, ast):
        maps = {}
        reserved_names = set(self.fragment_input_member_names)
        if self.current_target_stage is None:
            reserved_names.update(self.combined_vertex_output_member_names)
        reserved_names.update(self.fragment_stage_local_variable_names(ast))
        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_map = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(
                    member, reserved_names=reserved_names
                )
                member_map[member.name] = output_name
                if not self.is_fragment_builtin_output_target(output_name):
                    reserved_names.add(output_name)
            maps[struct_name] = member_map
        return maps

    def next_available_stage_io_layout(self, stage_name, direction, mapped_type):
        count = self.stage_io_location_count(mapped_type)
        ranges = sorted(self.stage_io_used_locations.get((stage_name, direction), []))
        ranges.extend(
            sorted(
                (start, end, name)
                for start, end, _, _, _, name in self.stage_io_used_components.get(
                    (stage_name, direction), []
                )
            )
        )
        location = 0

        while True:
            end = location + count - 1
            overlap = next(
                (
                    (used_start, used_end)
                    for used_start, used_end, _ in ranges
                    if location <= used_end and used_start <= end
                ),
                None,
            )
            if overlap is None:
                return f"layout(location = {location})"
            location = overlap[1] + 1

    def fragment_output_member_layout_maps_for_outputs(self):
        maps = {}

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(member, struct_name)
                layout = self.stage_io_layout_for_node(member)
                if self.is_fragment_builtin_output_target(
                    layout
                ) or self.is_fragment_builtin_output_target(output_name):
                    member_layouts[member.name] = output_name
                    continue
                if not layout.startswith("layout("):
                    continue

                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    layout,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = layout

            maps[struct_name] = member_layouts

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = maps.setdefault(struct_name, {})
            for member in getattr(struct, "members", []) or []:
                if member.name in member_layouts:
                    continue

                output_name = self.fragment_output_member_name(member, struct_name)
                layout = self.next_available_stage_io_layout(
                    "fragment", "output", self.member_type_name(member)
                )
                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    layout,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = layout

        return maps

    def type_node_name(self, type_node):
        if type_node is None:
            return None
        if hasattr(type_node, "name"):
            return type_node.name
        return str(type_node)

    def generate_stage_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            layout = self.stage_io_layout_for_node(member)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "input",
                layout,
                member_type,
                member.name,
            )
            prefix = self.stage_io_declaration_prefix(member, "in")
            member_decl = format_c_style_array_declaration(member_type, member.name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "input", member.name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_legacy_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_decl = format_c_style_array_declaration(
                self.member_type_name(member), member.name
            )
            declaration = f"out {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", member.name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_vertex_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                continue

            layout = self.stage_io_layout_for_node(member)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "output",
                layout,
                self.member_type_name(member),
                output_name,
            )
            prefix = self.stage_io_declaration_prefix(member, "out")
            member_type = self.member_type_name(member)
            prefix = self.stage_io_prefix_with_required_flat(prefix, member_type)
            member_decl = format_c_style_array_declaration(member_type, output_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_input_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            input_name = self.fragment_input_member_name(member, node.name)
            if input_name is None:
                continue

            layout = self.stage_io_layout_for_node(member)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "input",
                layout,
                self.member_type_name(member),
                input_name,
            )
            member_type = self.member_type_name(member)
            prefix = self.stage_io_declaration_prefix(member, "in")
            prefix = self.stage_io_prefix_with_required_flat(prefix, member_type)
            member_decl = format_c_style_array_declaration(member_type, input_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "fragment", "input", input_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            output_name = self.fragment_output_member_name(member, node.name)
            if self.is_fragment_builtin_output_target(output_name):
                continue
            layout = self.fragment_output_member_layout_maps.get(node.name, {}).get(
                member.name
            )
            if layout is None:
                layout = self.map_semantic(self.semantic_from_node(member))
                if not layout.startswith("layout("):
                    layout = "layout(location = 0)"
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "output",
                layout,
                member_type,
                output_name,
            )
            prefix = self.stage_io_declaration_prefix(member, "out", layout)
            member_decl = format_c_style_array_declaration(member_type, output_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "fragment", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def member_type_name(self, member):
        if hasattr(member, "member_type"):
            return self.map_type(member.member_type)
        return self.map_type(getattr(member, "vtype", "float"))

    def stage_io_prefix_with_required_flat(self, prefix, mapped_type):
        if not self.requires_flat_stage_io(mapped_type):
            return prefix
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        for direction in ("in", "out"):
            if direction in parts:
                parts.insert(parts.index(direction), "flat")
                return " ".join(parts)
        return f"{prefix} flat" if prefix else "flat"

    def requires_flat_stage_io(self, mapped_type):
        base_type, _ = split_array_type_suffix(str(mapped_type))
        return base_type in {
            "int",
            "uint",
            "bool",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def vertex_output_member_name(self, member):
        semantic = self.semantic_from_node(member)
        mapped_semantic = self.map_semantic(semantic)
        if self.is_vertex_builtin_output(mapped_semantic):
            return mapped_semantic
        if member.name in self.vertex_input_member_names:
            return f"out_{member.name}"
        return member.name

    def fragment_output_member_name(
        self, member, struct_name=None, reserved_names=None
    ):
        if struct_name in self.fragment_output_member_name_maps:
            return self.fragment_output_member_name_maps[struct_name].get(
                member.name, member.name
            )

        mapped_semantic = self.map_semantic(self.semantic_from_node(member))
        if mapped_semantic == "gl_FragDepth":
            return mapped_semantic
        if mapped_semantic == "gl_FragStencilRefARB":
            return mapped_semantic
        if mapped_semantic == "gl_SampleMask":
            return "gl_SampleMask[0]"

        if reserved_names is None:
            reserved_names = self.fragment_input_member_names
        return self.stage_io_name_avoiding_reserved(member.name, reserved_names)

    def is_fragment_builtin_output_target(self, target):
        return target in {
            "gl_FragDepth",
            "gl_FragStencilRefARB",
            "gl_SampleMask",
            "gl_SampleMask[0]",
        }

    def fragment_input_member_name(self, member, struct_name):
        if self.fragment_input_builtin_alias(member) is not None:
            return None
        if struct_name in self.vertex_output_struct_names:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_builtin_output(output_name):
                return None
            return self.combined_fragment_input_member_name(output_name)
        return self.combined_fragment_input_member_name(member.name)

    def fragment_input_builtin_alias(self, member):
        semantic = self.semantic_from_node(member)
        if semantic is None:
            return None
        mapped_semantic = self.map_stage_input_semantic(semantic, "fragment")
        if mapped_semantic not in {
            "gl_PrimitiveID",
            "gl_FrontFacing",
            "gl_FragCoord",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SamplePosition",
            "gl_SampleMaskIn",
        }:
            return None
        return self.stage_input_builtin_alias(semantic, "fragment")

    def combined_fragment_input_member_name(self, input_name):
        if (
            self.current_target_stage is not None
            or input_name not in self.combined_vertex_output_member_names
        ):
            return input_name
        candidate = f"in_{input_name}"
        suffix = 1
        reserved_names = set(self.combined_vertex_output_member_names)
        while candidate in reserved_names:
            suffix += 1
            candidate = f"in_{input_name}_{suffix}"
        return candidate

    def is_vertex_builtin_output(self, name):
        return name in {
            "gl_Position",
            "gl_PointSize",
            "gl_ClipDistance",
            "gl_CullDistance",
        }

    def stage_input_member_maps(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        maps = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            type_name = self.type_node_name(getattr(param, "param_type", None))
            struct = self.structs_by_name.get(type_name)
            if struct is None:
                continue
            if shader_type == "fragment":
                member_map = {}
                for member in getattr(struct, "members", []) or []:
                    builtin_alias = self.fragment_input_builtin_alias(member)
                    if builtin_alias is not None:
                        member_map[member.name] = builtin_alias
                        continue
                    input_name = self.fragment_input_member_name(member, type_name)
                    if input_name is not None:
                        member_map[member.name] = input_name
                maps[param.name] = member_map
            else:
                maps[param.name] = {
                    member.name: member.name
                    for member in getattr(struct, "members", [])
                }
        return maps

    def apply_stage_context_entry_aliases(self, func, shader_type, stage_context):
        if shader_type is not None:
            return
        if stage_context not in {"vertex", "fragment"}:
            return

        entry_func = self.stage_entry_functions_by_stage.get(stage_context)
        if entry_func is None:
            return

        parameter_names = {
            param.name
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
            if getattr(param, "name", None)
        }
        for name, member_map in self.stage_input_member_maps(
            entry_func,
            stage_context,
        ).items():
            if name not in parameter_names:
                self.current_stage_inputs.setdefault(name, member_map)

        for name, alias in self.stage_parameter_aliases(
            entry_func,
            stage_context,
        ).items():
            if name not in parameter_names:
                self.current_stage_parameter_aliases.setdefault(name, alias)

        if "output" not in parameter_names:
            output_member_map = self.stage_output_member_map(entry_func, stage_context)
            if output_member_map:
                self.current_stage_outputs.setdefault("output", output_member_map)

    def stage_parameter_aliases(self, func, shader_type):
        if shader_type is None:
            return {}

        aliases = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            output_alias = self.stage_parameter_output_aliases.get(id(func), {}).get(
                param.name
            )
            if output_alias is not None:
                aliases[param.name] = output_alias
                continue
            semantic = self.semantic_from_node(param)
            if self.is_stage_builtin_semantic(semantic):
                if self.stage_builtin_parameter_local_alias_declaration(
                    param, shader_type
                ):
                    continue
                aliases[param.name] = self.stage_input_builtin_alias(
                    semantic, shader_type
                )
        return aliases

    def stage_builtin_parameter_local_alias_declarations(self, func, shader_type):
        if shader_type is None:
            return []

        declarations = []
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            declaration = self.stage_builtin_parameter_local_alias_declaration(
                param, shader_type
            )
            if declaration is not None:
                declarations.append(declaration)
        return declarations

    def stage_builtin_parameter_local_alias_declaration(self, param, shader_type):
        semantic = self.semantic_from_node(param)
        if semantic is None:
            return None

        builtin = self.map_stage_input_semantic(semantic, shader_type)
        mapped_type = self.map_type(self.resource_node_type(param))
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return None

        expression = self.stage_builtin_parameter_local_alias_expression(
            builtin, base_type
        )
        if expression is None:
            return None
        return f"{base_type} {param.name} = {expression};"

    def stage_builtin_parameter_local_alias_expression(self, builtin, mapped_type):
        vector3_builtins = {"gl_GlobalInvocationID"}
        if builtin in vector3_builtins:
            if mapped_type == "uvec3":
                return None
            if mapped_type == "ivec3":
                return f"ivec3({builtin})"
            if mapped_type in {"uvec2", "ivec2"}:
                return f"{mapped_type}({builtin}.xy)"
            if mapped_type in {"uint", "int"}:
                return f"{mapped_type}({builtin}.x)"
            return None

        scalar_int_builtins = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_SampleID",
            "gl_SampleMaskIn",
        }
        if builtin in scalar_int_builtins:
            if mapped_type == "int":
                return None
            if mapped_type == "uint":
                return f"uint({builtin})"
            return None

        return None

    def stage_output_member_map(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        struct = self.structs_by_name.get(type_name)
        if struct is None:
            return {}

        if shader_type == "fragment":
            return {
                member.name: self.fragment_output_member_name(member, type_name)
                for member in getattr(struct, "members", []) or []
            }
        return {
            member.name: self.vertex_output_member_name(member)
            for member in getattr(struct, "members", []) or []
        }

    def stage_output_member_maps(self, func, shader_type):
        member_map = self.stage_output_member_map(func, shader_type)
        if not member_map:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        maps = {}
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        for stmt in statements:
            if not isinstance(stmt, VariableNode):
                continue
            if self.type_node_name(getattr(stmt, "var_type", None)) == type_name:
                maps[stmt.name] = member_map
        return maps

    def function_return_type(self, func):
        return_type = getattr(func, "return_type", None)
        if return_type is None:
            return "void"
        return self.map_type(return_type)

    def stage_return_output(self, func, shader_type):
        if shader_type == "fragment":
            return self.fragment_stage_output(func, shader_type)
        if shader_type == "vertex":
            return self.vertex_stage_output(func)
        return None

    def is_void_stage_entry_return_value(self):
        if self.current_stage_output is not None:
            return False
        if self.current_stage_entry_type is None:
            return False
        return self.current_function_return_type == "void"

    def vertex_stage_output(self, func):
        output_type = self.function_return_type(func)
        if output_type == "void":
            return None

        semantic = self.function_return_semantic(func)
        mapped_semantic = self.map_semantic(semantic)
        if not mapped_semantic and output_type == "vec4":
            mapped_semantic = "gl_Position"
        elif not mapped_semantic:
            if output_type in self.structs_by_name or output_type not in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "dvec2",
                "dvec3",
                "dvec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "mat2",
                "mat3",
                "mat4",
                "dmat2",
                "dmat3",
                "dmat4",
            }:
                return None
            raise ValueError(
                "OpenGL vertex stage entry point with non-void return type "
                f"'{output_type}' requires an output semantic"
            )

        if self.is_vertex_builtin_output(mapped_semantic):
            return {
                "name": mapped_semantic,
                "declaration": "",
            }

        layout = mapped_semantic
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"
        reserved_names = set(self.vertex_input_member_names)
        reserved_names.update(self.stage_io_declared_names("vertex", "output"))
        reserved_names.update(self.function_local_variable_names(func))
        output_name = self.stage_io_name_avoiding_reserved(
            "vertexOutput", reserved_names
        )
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "vertex",
            "output",
            layout,
            output_type,
            output_name,
        )
        declaration = f"{layout} out {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "vertex", "output", output_name, declaration
        ):
            declaration = ""
        return {
            "name": output_name,
            "declaration": declaration,
        }

    def fragment_stage_output(self, func, shader_type):
        if shader_type != "fragment":
            return None

        output_type = self.function_return_type(func)
        if output_type == "void":
            return None
        if output_type in self.structs_by_name:
            return None

        semantic = self.function_return_semantic(func) or "gl_FragColor"
        mapped_semantic = self.map_semantic(semantic)
        if mapped_semantic == "gl_FragDepth":
            return {
                "name": "gl_FragDepth",
                "declaration": "",
            }
        if mapped_semantic == "gl_FragStencilRefARB":
            return {
                "name": "gl_FragStencilRefARB",
                "declaration": "",
            }
        if mapped_semantic == "gl_SampleMask":
            return {
                "name": "gl_SampleMask[0]",
                "declaration": "",
            }

        layout = self.stage_io_layout_for_node(func)
        if not layout:
            layout = mapped_semantic
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"

        local_names = self.function_local_variable_names(func)
        output_key = self.fragment_direct_output_key(
            mapped_semantic, layout, output_type
        )
        output_name = self.fragment_direct_output_name_maps.get(output_key)
        if output_name in local_names:
            output_name = None
        if output_name is None:
            output_name = self.fragment_output_name(semantic, local_names)
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "fragment",
            "output",
            layout,
            output_type,
            output_name,
        )
        prefix = self.stage_io_declaration_prefix(func, "out", layout)
        declaration = f"{prefix} {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "fragment", "output", output_name, declaration
        ):
            declaration = ""
        self.fragment_direct_output_name_maps.setdefault(output_key, output_name)
        return {
            "name": output_name,
            "declaration": declaration,
        }

    def fragment_direct_output_key(self, mapped_semantic, layout, output_type):
        return (
            str(mapped_semantic or ""),
            str(layout or ""),
            str(output_type or ""),
        )

    def fragment_output_name(self, semantic, additional_reserved_names=None):
        if semantic and semantic.startswith("gl_FragColor"):
            suffix = semantic[len("gl_FragColor") :]
            output_name = f"fragColor{suffix}"
        else:
            output_name = "fragColor"

        reserved_names = set(self.fragment_input_member_names)
        for member_map in self.fragment_output_member_name_maps.values():
            for name in member_map.values():
                if not self.is_fragment_builtin_output_target(name):
                    reserved_names.add(name)
        reserved_names.update(self.stage_io_declared_names("fragment", "output"))
        reserved_names.update(additional_reserved_names or set())
        return self.stage_io_name_avoiding_reserved(output_name, reserved_names)

    def generate_stage_struct_constructor_return(self, expr, indent):
        if not self.current_stage_output_member_map:
            return None

        struct = self.structs_by_name.get(self.current_stage_return_type)
        if struct is None:
            return None

        members = getattr(struct, "members", []) or []
        field_values = self.stage_struct_constructor_field_values(expr, members)
        if field_values is None:
            return None

        indent_str = "    " * indent
        code = ""
        for member, value_expr in field_values:
            output_name = self.current_stage_output_member_map.get(member.name)
            if output_name is None:
                continue
            value = self.generate_expression_with_expected(
                value_expr, self.member_type_name(member)
            )
            code += f"{indent_str}{output_name} = {value};\n"
        code += f"{indent_str}return;\n"
        return code

    def stage_struct_constructor_field_values(self, expr, members):
        if isinstance(expr, FunctionCallNode):
            if self.function_call_name(expr) != self.current_stage_return_type:
                return None
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if len(args) != len(members):
                return None
            return list(zip(members, args))

        if isinstance(expr, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(expr, "constructor_type", None)
            )
            if constructor_type != self.current_stage_return_type:
                return None
            args = list(getattr(expr, "arguments", []) or [])
            named_args = dict(getattr(expr, "named_arguments", {}) or {})
            if len(args) > len(members):
                return None

            values = []
            for index, member in enumerate(members):
                if index < len(args):
                    values.append((member, args[index]))
                    continue
                if member.name not in named_args:
                    return None
                values.append((member, named_args[member.name]))
            return values

        return None

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL AST statement as GLSL source."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            if stmt.name in self.flattened_stage_variables:
                return ""
            var_type = self.local_variable_declared_type(stmt)
            if (
                self.current_stage_output is not None
                and stmt.name == self.current_stage_output["name"]
            ):
                initial_value = getattr(stmt, "initial_value", None)
                if initial_value is None:
                    return ""
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                return f"{indent_str}{stmt.name} = {init_expr};\n"
            local_name = self.glsl_local_identifier_name(stmt.name)
            self.local_variable_types[stmt.name] = var_type
            self.update_compile_time_offset_constants(stmt, var_type)

            declaration = format_c_style_array_declaration(
                self.map_type(var_type), local_name
            )
            declaration = f"{self.local_variable_qualifier(stmt)}{declaration}"
            initial_value = getattr(stmt, "initial_value", None)
            if initial_value is not None and self.type_contains_opaque_resource(
                var_type
            ):
                self.current_identifier_aliases[stmt.name] = (
                    self.generate_expression_with_expected(initial_value, var_type)
                )
                return ""
            if isinstance(initial_value, MatchNode):
                code = f"{indent_str}{declaration};\n"
                code += generate_match_expression_assignment(
                    self,
                    initial_value,
                    local_name,
                    var_type,
                    indent,
                    "GLSL",
                )
                return code
            if initial_value is not None:
                dynamic_init = (
                    self.generate_glsl_dynamic_resource_call_assignment_statement(
                        initial_value,
                        local_name,
                        var_type,
                        indent,
                    )
                )
                if dynamic_init is not None and not self.local_variable_qualifier(stmt):
                    return f"{indent_str}{declaration};\n{dynamic_init}"
                init_expr = self.generate_expression_with_expected(
                    initial_value, var_type
                )
                return f"{indent_str}{declaration} = {init_expr};\n"
            else:
                return f"{indent_str}{declaration};\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            mesh_assignment = self.generate_mesh_output_assignment_statement(
                stmt, indent
            )
            if mesh_assignment is not None:
                return mesh_assignment
            dynamic_assignment = self.generate_glsl_dynamic_resource_assignment_node(
                stmt,
                indent,
            )
            if dynamic_assignment is not None:
                return dynamic_assignment
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
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
            if getattr(stmt, "value", None) is None:
                return f"{indent_str}return;\n"
            dynamic_resource_return = (
                self.generate_glsl_dynamic_resource_call_return_statement(
                    stmt.value,
                    indent,
                )
            )
            if dynamic_resource_return is not None:
                return dynamic_resource_return
            return_value_name = self.expression_name(stmt.value)
            if return_value_name in self.flattened_stage_variables:
                return f"{indent_str}return;\n"
            if (
                self.current_stage_output is not None
                and return_value_name == self.current_stage_output["name"]
            ):
                return f"{indent_str}return;\n"
            stage_struct_return = self.generate_stage_struct_constructor_return(
                stmt.value, indent
            )
            if stage_struct_return is not None:
                return stage_struct_return
            if self.current_stage_output is not None:
                if isinstance(stmt.value, list):
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    value = values
                else:
                    value = self.generate_expression_with_expected(
                        stmt.value, self.current_function_return_type
                    )
                return (
                    f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                    f"{indent_str}return;\n"
                )
            if self.is_void_stage_entry_return_value():
                return f"{indent_str}return;\n"
            if isinstance(stmt.value, list):
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values};\n"
            else:
                return (
                    f"{indent_str}return "
                    f"{self.generate_expression_with_expected(stmt.value, self.current_function_return_type)};\n"
                )
        elif hasattr(stmt, "__class__") and "ExpressionStatementNode" in str(
            type(stmt)
        ):
            tail_return = self.generate_tail_expression_statement(stmt, indent)
            if tail_return is not None:
                return tail_return
            mesh_assignment = self.generate_mesh_output_assignment_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_assignment is not None:
                return mesh_assignment
            mesh_intrinsic = self.generate_mesh_intrinsic_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_intrinsic is not None:
                return mesh_intrinsic
            mesh_helper = self.generate_mesh_output_helper_call_statement(
                getattr(stmt, "expression", None), indent
            )
            if mesh_helper is not None:
                return mesh_helper
            expression = getattr(stmt, "expression", None)
            if isinstance(expression, AssignmentNode):
                dynamic_assignment = (
                    self.generate_glsl_dynamic_resource_assignment_node(
                        expression,
                        indent,
                    )
                )
                if dynamic_assignment is not None:
                    return dynamic_assignment
            dynamic_resource_statement = (
                self.generate_glsl_dynamic_resource_call_expression_statement(
                    expression,
                    indent,
                )
            )
            if dynamic_resource_statement is not None:
                return dynamic_resource_statement
            clip_statement = self.generate_clip_statement(expression, indent)
            if clip_statement is not None:
                return clip_statement
            expr_code = self.generate_expression_statement(stmt)
            return f"{indent_str}{expr_code};\n"
        else:
            mesh_intrinsic = self.generate_mesh_intrinsic_statement(stmt, indent)
            if mesh_intrinsic is not None:
                return mesh_intrinsic
            mesh_helper = self.generate_mesh_output_helper_call_statement(stmt, indent)
            if mesh_helper is not None:
                return mesh_helper
            dynamic_resource_statement = (
                self.generate_glsl_dynamic_resource_call_expression_statement(
                    stmt,
                    indent,
                )
            )
            if dynamic_resource_statement is not None:
                return dynamic_resource_statement
            clip_statement = self.generate_clip_statement(stmt, indent)
            if clip_statement is not None:
                return clip_statement
            expr_result = self.generate_expression(stmt)
            if expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def glsl_dynamic_resource_call_dispatch_info(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None
        func_name = self.function_call_name(expr)
        if not func_name:
            return None
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        dynamic_info = self.glsl_dynamic_resource_call_info(
            func_name,
            args,
            self.current_resource_aliases,
        )
        if dynamic_info is None:
            return None

        cases = []
        for index in range(dynamic_info["array_size"]):
            static_args = list(args)
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            specialized = self.glsl_resource_function_call_specialization(
                func_name,
                static_args,
            )
            if specialized is None:
                return None
            call_args = self.glsl_resource_specialized_call_arguments(
                specialized,
                static_args,
            )
            rendered_args = ", ".join(
                self.generate_function_call_arguments(specialized.name, call_args)
            )
            cases.append((index, f"{specialized.name}({rendered_args})"))

        return {
            "index_expr": dynamic_info["index_expr"],
            "cases": cases,
            "return_type": self.expression_result_type(expr),
        }

    def generate_glsl_dynamic_resource_switch_statement(
        self,
        dispatch,
        indent,
        case_statement,
        default_statement=None,
    ):
        indent_str = "    " * indent
        case_indent = "    " * (indent + 1)
        code = f"{indent_str}switch ({self.generate_expression(dispatch['index_expr'])}) {{\n"
        for index, call in dispatch["cases"]:
            code += f"{indent_str}case {index}:\n"
            code += f"{case_indent}{case_statement(call)}\n"
            code += f"{case_indent}break;\n"
        code += f"{indent_str}default:\n"
        if default_statement is not None:
            code += f"{case_indent}{default_statement}\n"
        code += f"{case_indent}break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_glsl_dynamic_resource_call_assignment_statement(
        self,
        expr,
        target,
        target_type,
        indent,
    ):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None
        return self.generate_glsl_dynamic_resource_switch_statement(
            dispatch,
            indent,
            lambda call: f"{target} = {call};",
            f"{target} = {self.zero_value_expression(target_type)};",
        )

    def generate_glsl_dynamic_resource_assignment_node(self, stmt, indent):
        left_node = getattr(stmt, "target", getattr(stmt, "left", None))
        right_node = getattr(stmt, "value", getattr(stmt, "right", None))
        op = self.map_operator(getattr(stmt, "operator", getattr(stmt, "op", "=")))
        if op != "=":
            return None
        if self.glsl_dynamic_resource_call_dispatch_info(right_node) is None:
            return None
        self.validate_glsl_buffer_block_assignment_target(left_node, op)
        expected_type = self.glsl_tessellation_factor_assignment_expected_type(
            left_node
        )
        if expected_type is not None:
            self.validate_glsl_tessellation_factor_assignment_value(
                left_node, right_node
            )
        else:
            expected_type = self.expression_result_type(left_node)
        left = self.generate_glsl_buffer_block_mutation_target(left_node)
        return self.generate_glsl_dynamic_resource_call_assignment_statement(
            right_node,
            left,
            expected_type,
            indent,
        )

    def generate_glsl_dynamic_resource_call_expression_statement(self, expr, indent):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None
        return self.generate_glsl_dynamic_resource_switch_statement(
            dispatch,
            indent,
            lambda call: f"{call};",
        )

    def generate_glsl_dynamic_resource_call_return_statement(self, expr, indent):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None

        indent_str = "    " * indent
        case_indent = "    " * (indent + 1)
        code = f"{indent_str}switch ({self.generate_expression(dispatch['index_expr'])}) {{\n"
        for index, call in dispatch["cases"]:
            code += f"{indent_str}case {index}:\n"
            code += f"{case_indent}return {call};\n"
        return_type = dispatch.get("return_type") or self.current_function_return_type
        mapped_return_type = self.map_type(return_type)
        code += f"{indent_str}default:\n"
        if mapped_return_type == "void":
            code += f"{case_indent}return;\n"
        else:
            code += f"{case_indent}return {self.zero_value_expression(return_type)};\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        value_expr = getattr(stmt, "expression", None)

        stage_struct_return = self.generate_stage_struct_constructor_return(
            value_expr,
            indent,
        )
        if stage_struct_return is not None:
            return stage_struct_return

        indent_str = "    " * indent
        if self.current_stage_output is not None:
            value = self.generate_expression_with_expected(
                value_expr,
                self.current_function_return_type,
            )
            return (
                f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                f"{indent_str}return;\n"
            )

        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        value = self.generate_expression_with_expected(value_expr, return_type)
        return f"{indent_str}return {value};\n"

    def local_variable_declared_type(self, stmt):
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        if var_type is None:
            var_type = self.expression_result_type(getattr(stmt, "initial_value", None))
        return self.type_name_string(var_type) or "float"

    def glsl_parameter_identifier_name(self, name):
        if name not in self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS:
            return name

        used_names = set(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS)

        alias = f"{name}_"
        suffix = 1
        while alias in used_names:
            suffix += 1
            alias = f"{name}_{suffix}"
        self.current_identifier_aliases[name] = alias
        return alias

    def glsl_local_identifier_name(self, name):
        reserved_names = self.current_stage_declared_names()
        if (
            name not in self.GLSL_RESERVED_IDENTIFIERS
            and name not in self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS
            and name not in reserved_names
        ):
            return name

        used_names = set(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(reserved_names)
        used_names.update(self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS)

        alias = f"{name}_"
        suffix = 1
        while alias in used_names:
            suffix += 1
            alias = f"{name}_{suffix}"
        self.current_identifier_aliases[name] = alias
        return alias

    def glsl_global_identifier_name(self, name):
        if name not in self.GLSL_RESERVED_IDENTIFIERS:
            return name

        used_names = set(self.global_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        alias = f"{name}_"
        suffix = 1
        while alias in used_names:
            suffix += 1
            alias = f"{name}_{suffix}"
        self.current_identifier_aliases[name] = alias
        return alias

    def match_binding_name(self, name):
        return self.glsl_local_identifier_name(name)

    def current_stage_declared_names(self):
        names = set(self.current_stage_parameter_aliases.values())
        for member_map in self.current_stage_inputs.values():
            names.update(member_map.values())
        for member_map in self.current_stage_outputs.values():
            names.update(member_map.values())
        names.update(self.current_stage_output_member_map.values())
        if self.current_stage_output is not None:
            names.add(self.current_stage_output["name"])
        return names

    def local_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.add(str(attr_name).lower())

        parts = []
        if "const" in qualifiers:
            parts.append("const")
        parts.extend(
            qualifier
            for qualifier in ("lowp", "mediump", "highp")
            if qualifier in qualifiers
        )
        return f"{' '.join(parts)} " if parts else ""

    def global_variable_qualifier(self, node):
        qualifier_prefix = self.glsl_variable_qualifier_prefix(node)
        if qualifier_prefix:
            return f"{qualifier_prefix} "

        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if qualifiers & {"shared", "groupshared", "workgroup", "threadgroup"}:
            return "shared "
        if "const" in qualifiers:
            return "const "
        return "uniform "

    def global_stage_io_layout_direction(self, qualifier):
        parts = qualifier.split()
        if "in" in parts:
            return "input"
        if "out" in parts:
            return "output"
        return None

    def reserve_global_stage_io_layout(self, node, qualifier, layout, mapped_type):
        direction = self.global_stage_io_layout_direction(qualifier)
        if direction is None:
            return

        stage_name = self.current_target_stage or "global"
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            stage_name,
            direction,
            layout,
            mapped_type,
            self.resource_node_name(node, "<unnamed>"),
        )

    def stage_io_layout_has(self, layout, names):
        if not layout.startswith("layout("):
            return False
        start = layout.find("(")
        end = layout.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return False

        for item in layout[start + 1 : end].split(","):
            key = item.partition("=")[0].strip()
            if key in names:
                return True
        return False

    def generate_global_variable_declaration(
        self, node, declaration, vtype, mapped_type_for_layout=None
    ):
        name = self.resource_node_name(node, "")
        builtin_output = self.fragment_output_variable_builtin_target(node)
        if builtin_output is not None:
            self.current_identifier_aliases[name] = builtin_output
            return ""

        qualifier = self.global_variable_qualifier(node)
        emitted_name = self.glsl_global_identifier_name(name)
        if emitted_name != name:
            declaration = format_c_style_array_declaration(
                mapped_type_for_layout if mapped_type_for_layout is not None else vtype,
                emitted_name,
            )
        initializer = ""
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            initializer = (
                f" = {self.generate_expression_with_expected(initial_value, vtype)}"
            )
        layout = self.glsl_variable_layout_prefix(node)
        mapped_type = (
            mapped_type_for_layout if mapped_type_for_layout is not None else vtype
        )
        stage_name = getattr(
            self, "current_global_stage_io_stage", self.current_target_stage
        )
        if self.global_stage_io_layout_direction(
            qualifier
        ) == "input" and self.requires_flat_stage_input(stage_name, mapped_type):
            qualifier = f"{self.with_flat_stage_input_qualifier(qualifier.strip())} "
        self.reserve_global_stage_io_layout(
            node,
            qualifier,
            layout,
            mapped_type,
        )
        return f"{layout}{qualifier}{declaration}{initializer};\n"

    def generate_stage_local_interface_variable_declaration(self, node):
        builtin_name = self.glsl_builtin_stage_interface_name(node)
        if builtin_name is not None:
            source_name = self.resource_node_name(node, "")
            if source_name and source_name != builtin_name:
                self.current_identifier_aliases[source_name] = builtin_name
            return ""

        vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
        mapped_type = f"{self.map_type(vtype)}{array_suffix}"
        declaration = format_c_style_array_declaration(
            mapped_type, self.resource_node_name(node, "")
        )
        direction = self.stage_local_interface_direction(node)
        if direction is not None:
            prefix = self.stage_io_declaration_prefix(node, direction)
            stage_name = getattr(
                self, "current_global_stage_io_stage", self.current_target_stage
            )
            if direction == "in" and self.requires_flat_stage_input(
                stage_name, mapped_type
            ):
                prefix = self.with_flat_stage_io_declaration_prefix(prefix)
            layout = self.glsl_variable_layout_prefix(node)
            self.reserve_global_stage_io_layout(node, f"{prefix} ", layout, mapped_type)
            return f"{prefix} {declaration};\n"

        return self.generate_global_variable_declaration(
            node, declaration, vtype, mapped_type_for_layout=mapped_type
        )

    def stage_local_interface_direction(self, node):
        names = {
            str(qualifier).lower().replace("-", "_")
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                names.add(str(attr_name).lower().replace("-", "_"))

        if names & {"input", "in"}:
            return "in"
        if names & {"output", "out"}:
            return "out"
        if "inout" in names:
            return "inout"
        return None

    def with_flat_stage_io_declaration_prefix(self, prefix):
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        for direction in ("in", "out", "inout"):
            marker = f" {direction}"
            if prefix.endswith(marker):
                return f"{prefix[: -len(marker)]} flat{marker}"
            marker = f" {direction} "
            if marker in prefix:
                return prefix.replace(marker, f" flat{marker}", 1)
        return f"flat {prefix}" if prefix else "flat"

    def glsl_builtin_stage_interface_name(self, node):
        mapped_semantic = self.map_semantic(self.semantic_from_node(node))
        if not mapped_semantic or mapped_semantic.startswith("layout("):
            return None
        if not mapped_semantic.startswith("gl_"):
            return None
        if self.glsl_variable_layout_prefix(node).strip():
            return None
        return mapped_semantic

    def fragment_output_variable_builtin_target(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "out" not in qualifiers:
            return None
        if self.map_semantic(self.semantic_from_node(node)) == "gl_SampleMask":
            return "gl_SampleMask[0]"
        return None

    def glsl_variable_qualifier_prefix(self, node):
        qualifiers = [
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        ]
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(str(attr_name).lower())

        emitted = []

        def add(qualifier):
            if qualifier not in emitted:
                emitted.append(qualifier)

        for qualifier in qualifiers:
            normalized = (
                qualifier[len("glsl_") :]
                if qualifier.startswith("glsl_")
                else qualifier
            )
            normalized = normalized.replace("-", "_")
            if normalized in {"perprimitive", "perprimitiveext"}:
                add("perprimitiveEXT")
            elif normalized in {
                "task_payload_shared",
                "taskpayloadshared",
                "taskpayloadsharedext",
            }:
                add("taskPayloadSharedEXT")
            elif normalized in {"raypayload", "raypayloadext"}:
                add("rayPayloadEXT")
            elif normalized in {"raypayloadin", "raypayloadinext"}:
                add("rayPayloadInEXT")
            elif normalized in {"hitattribute", "hitattributeext"}:
                add("hitAttributeEXT")
            elif normalized in {"callabledata", "callabledataext"}:
                add("callableDataEXT")
            elif normalized in {"callabledatain", "callabledatainext"}:
                add("callableDataInEXT")
            elif normalized in {"tileimage", "tileimageext"}:
                add("tileImageEXT")
            elif normalized in {"shared", "groupshared", "workgroup", "threadgroup"}:
                add("shared")
            elif normalized in {
                "invariant",
                "precise",
                "patch",
                "flat",
                "smooth",
                "noperspective",
                "centroid",
                "sample",
                "in",
                "out",
                "lowp",
                "mediump",
                "highp",
            }:
                add(normalized)

        if not emitted:
            return ""

        order = {
            "invariant": 0,
            "precise": 1,
            "patch": 2,
            "perprimitiveEXT": 3,
            "flat": 4,
            "smooth": 5,
            "noperspective": 6,
            "centroid": 7,
            "sample": 8,
            "in": 9,
            "out": 10,
            "tileImageEXT": 11,
            "lowp": 12,
            "mediump": 13,
            "highp": 14,
            "shared": 15,
            "taskPayloadSharedEXT": 16,
            "rayPayloadEXT": 17,
            "rayPayloadInEXT": 18,
            "hitAttributeEXT": 19,
            "callableDataEXT": 20,
            "callableDataInEXT": 21,
        }
        emitted.sort(key=lambda qualifier: order.get(qualifier, len(order)))
        return " ".join(emitted)

    def glsl_variable_layout_prefix(self, node):
        layout_parts = []
        seen_layout_parts = {}
        node_name = self.resource_node_name(node, "<unnamed>")
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            normalized = normalized.replace("-", "_")
            if normalized in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                continue
            if normalized in self.GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES:
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    continue
                previous = seen_layout_parts.get(normalized)
                if previous is not None:
                    raise ValueError(
                        "Duplicate OpenGL layout metadata for "
                        f"'{node_name}': {normalized}"
                    )
                seen_layout_parts[normalized] = normalized
                layout_parts.append(normalized)
                continue
            if normalized not in self.GLSL_LAYOUT_ATTRIBUTE_NAMES:
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            layout_part = (
                f"{normalized} = {self.attribute_value_to_string(arguments[0])}"
            )
            previous = seen_layout_parts.get(normalized)
            if previous == layout_part:
                raise ValueError(
                    "Duplicate OpenGL layout metadata for "
                    f"'{node_name}': {layout_part}"
                )
            if previous is not None:
                raise ValueError(
                    "Conflicting OpenGL layout metadata for "
                    f"'{node_name}': {previous} differs from {layout_part}"
                )
            seen_layout_parts[normalized] = layout_part
            layout_parts.append(layout_part)
        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) "

    def is_glsl_stage_io_metadata_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False

        normalized = str(attr_name).lower()
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]
        normalized = normalized.replace("-", "_")

        return (
            normalized in self.GLSL_LAYOUT_ATTRIBUTE_NAMES
            or normalized in self.GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES
            or normalized in self.GLSL_VARIABLE_QUALIFIER_ATTRIBUTE_NAMES
            or normalized in self.GLSL_STORAGE_QUALIFIER_ATTRIBUTE_NAMES
            or normalized
            in {
                "input",
                "output",
                "interface_block",
                "interface_instance",
                "interface_array",
            }
        )

    def stage_io_layout_for_node(self, node):
        explicit_layout = self.glsl_variable_layout_prefix(node).strip()
        if explicit_layout:
            return explicit_layout
        return self.map_semantic(self.semantic_from_node(node))

    def stage_io_declaration_prefix(self, node, direction, layout=None):
        parts = []
        if layout is None:
            layout = self.stage_io_layout_for_node(node)
        if layout.startswith("layout("):
            parts.append(layout)

        qualifiers = [
            qualifier
            for qualifier in self.glsl_variable_qualifier_prefix(node).split()
            if qualifier not in {"in", "out", "inout"}
        ]
        precision = [
            qualifier
            for qualifier in qualifiers
            if qualifier in self.GLSL_PRECISION_QUALIFIERS
        ]
        other_qualifiers = [
            qualifier
            for qualifier in qualifiers
            if qualifier not in self.GLSL_PRECISION_QUALIFIERS
        ]
        parts.extend(other_qualifiers)
        parts.append(direction)
        parts.extend(precision)
        return " ".join(parts)

    def requires_flat_stage_input(self, stage_name, mapped_type):
        if stage_name != "fragment":
            return False
        base_type, _ = split_array_type_suffix(str(mapped_type))
        return base_type in {
            "int",
            "uint",
            "bool",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def with_flat_stage_input_qualifier(self, prefix):
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        try:
            direction_index = parts.index("in")
        except ValueError:
            return f"flat {prefix}" if prefix else "flat"
        parts.insert(direction_index, "flat")
        return " ".join(parts)

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if (
            isinstance(vtype, (PointerType, ReferenceType))
            or hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
        ):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def signed_compute_builtin_member_cast(self, object_name, member_name):
        expected_type = self.map_type(self.current_expression_expected_type)
        if expected_type != "int":
            return None
        if object_name not in self.UNSIGNED_VECTOR_COMPUTE_BUILTINS:
            return None
        if member_name not in {"x", "y", "z"}:
            return None
        return f"int({object_name}.{member_name})"

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def is_matrix_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype).startswith(("mat", "dmat"))

    def type_contains_opaque_resource(self, vtype, seen=None):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return False

        base_type, _array_suffix = split_array_type_suffix(str(type_name))
        if self.is_sampler_type(base_type):
            return True
        mapped_resource_type = self.map_resource_type_with_format(base_type)
        if self.is_opaque_resource_type(mapped_resource_type):
            return True
        if self.is_structured_buffer_type(base_type):
            return True

        mapped_type = self.map_type(base_type)
        seen = set(seen or set())
        if mapped_type in seen:
            return False
        seen.add(mapped_type)

        struct = self.structs_by_name.get(mapped_type)
        if struct is None:
            return False
        for member_type in self.struct_member_types.get(mapped_type, {}).values():
            if self.type_contains_opaque_resource(member_type, seen):
                return True
        return False

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("dvec"):
            return "double"
        if mapped_type.startswith("uvec"):
            return "uint"
        if mapped_type.startswith("ivec"):
            return "int"
        if mapped_type.startswith("bvec"):
            return "bool"
        if mapped_type.startswith("vec"):
            return "float"
        return None

    def glsl_constructor_type(self, func_name):
        constructor = self.map_type(func_name) if isinstance(func_name, str) else None
        if constructor in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
            "dmat2",
            "dmat3",
            "dmat4",
            "dmat2x2",
            "dmat2x3",
            "dmat2x4",
            "dmat3x2",
            "dmat3x3",
            "dmat3x4",
            "dmat4x2",
            "dmat4x3",
            "dmat4x4",
        }:
            return constructor
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            return (
                self.local_variable_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
            )
        if isinstance(expr, (int, float)):
            return "float" if isinstance(expr, float) else "int"
        if isinstance(expr, BinaryOpNode):
            operator = self.map_operator(
                getattr(expr, "op", getattr(expr, "operator", None))
            )
            if operator in self.GLSL_VECTOR_RELATIONAL_FUNCTIONS:
                left_type = self.expression_result_type(expr.left)
                right_type = self.expression_result_type(expr.right)
                vector_type = self.glsl_relational_vector_type(left_type, right_type)
                if vector_type is not None:
                    return self.glsl_bool_vector_type(vector_type)
                return "bool"
            if operator in {"==", "!=", "&&", "||"}:
                return "bool"
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
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
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
        if isinstance(expr, WaveOpNode):
            if expr.operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
                return "uint"
            if expr.operation in {
                "WaveIsFirstLane",
                "WaveActiveAllTrue",
                "WaveActiveAnyTrue",
                "WaveActiveAllEqual",
            }:
                return "bool"
            if expr.operation in {
                "WaveActiveCountBits",
                "WavePrefixCountBits",
                "WaveMultiPrefixCountBits",
            }:
                return "uint"
            if expr.operation in {"WaveActiveBallot", "WaveMatch"}:
                return "uvec4"
            if expr.arguments:
                return self.expression_result_type(expr.arguments[0])
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            if func_name in self.GLSL_WAVE_INTRINSIC_ARITIES:
                return self.glsl_wave_result_type(func_name, args)
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
            if func_name in {"mix", "lerp"} and len(args) >= 2:
                left_type = self.expression_result_type(args[0])
                right_type = self.expression_result_type(args[1])
                if self.is_vector_value_type(left_type):
                    return left_type
                if self.is_vector_value_type(right_type):
                    return right_type
                return left_type or right_type
            if (
                func_name == "saturate"
                and args
                and func_name not in self.function_return_types
            ):
                return self.expression_result_type(args[0])
            if (
                func_name == "rcp"
                and args
                and func_name not in self.function_return_types
            ):
                return self.expression_result_type(args[0])
            if (
                func_name in self.GLSL_BITCAST_FUNCTIONS
                and args
                and func_name not in self.function_return_types
            ):
                return self.glsl_bitcast_result_type(func_name, args[0])
            specialized_func_name = generic_function_call_name(self, func_name, args)
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            if func_name in self.function_return_types:
                return self.function_return_types[func_name]
            if func_name == "imageLoad" and args:
                return self.image_load_result_type(args[0])
            if (
                func_name
                in {
                    "texture",
                    "textureLod",
                    "textureGrad",
                    "textureProj",
                    "textureProjLod",
                    "textureProjGrad",
                    "textureOffset",
                    "textureLodOffset",
                    "textureGradOffset",
                    "texelFetch",
                    "texelFetchOffset",
                    "textureGather",
                    "textureGatherOffset",
                    "textureGatherOffsets",
                }
                and args
            ):
                return self.texture_sampling_result_type(args[0], func_name)
            derivative_name = self.GLSL_DERIVATIVE_FUNCTION_ALIASES.get(
                func_name,
                func_name,
            )
            if derivative_name in self.GLSL_DERIVATIVE_FUNCTIONS and args:
                return self.expression_result_type(args[0])
            constructor = self.glsl_constructor_type(func_name)
            if constructor:
                return constructor
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", None)
            return (
                self.local_variable_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
            )
        return None

    def glsl_bool_vector_type(self, vector_type):
        mapped_type = self.map_type(vector_type)
        if self.vector_component_type(mapped_type) is None:
            return None
        if mapped_type.endswith(("2", "3", "4")):
            return f"bvec{mapped_type[-1]}"
        return None

    def glsl_relational_vector_type(self, left_type, right_type):
        if self.is_vector_value_type(left_type):
            return self.map_type(left_type)
        if self.is_vector_value_type(right_type):
            return self.map_type(right_type)
        return None

    def is_numeric_literal_expression(self, expr):
        if isinstance(expr, (int, float)) and not isinstance(expr, bool):
            return True
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            value = getattr(expr, "value", None)
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return False

    def glsl_vector_relational_operand(self, expr, expr_type, vector_type):
        generated = self.generate_expression(expr)
        if self.is_vector_value_type(expr_type):
            return generated
        is_scalar = self.is_scalar_numeric_type(expr_type)
        is_literal = self.is_numeric_literal_expression(expr)
        if is_scalar or is_literal:
            return f"{vector_type}({generated})"
        return generated

    def generate_vector_relational_expression(self, left_expr, operator, right_expr):
        function_name = self.GLSL_VECTOR_RELATIONAL_FUNCTIONS.get(operator)
        if function_name is None:
            return None

        left_type = self.expression_result_type(left_expr)
        right_type = self.expression_result_type(right_expr)
        vector_type = self.glsl_relational_vector_type(left_type, right_type)
        if vector_type is None:
            return None

        left = self.glsl_vector_relational_operand(left_expr, left_type, vector_type)
        right = self.glsl_vector_relational_operand(right_expr, right_type, vector_type)
        return f"{function_name}({left}, {right})"

    def validate_dispatch_mesh_count_argument(self, arg, index):
        axis = ("x", "y", "z")[index]
        arg_type = self.expression_result_type(arg)
        if arg_type is not None:
            mapped_type = self.map_type(arg_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise ValueError(
                    f"GLSL DispatchMesh {axis} group count requires a "
                    "non-negative scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_count = self.literal_int_value(arg, self.literal_int_constants)
        if literal_count is None:
            return
        if (
            not isinstance(literal_count, int)
            or isinstance(literal_count, bool)
            or literal_count < 0
        ):
            raise ValueError(
                f"GLSL DispatchMesh {axis} group count requires a "
                "non-negative scalar integer value, got "
                f"{self.glsl_attribute_argument_source(arg)}"
            )

    def validate_dispatch_mesh_count_arguments(self, args):
        for index, arg in enumerate(args[:3]):
            self.validate_dispatch_mesh_count_argument(arg, index)

    def validate_dispatch_mesh_arguments(self, args):
        if len(args) not in {3, 4}:
            raise ValueError(
                "GLSL DispatchMesh requires three group counts and optional "
                "payload argument"
            )
        self.validate_dispatch_mesh_count_arguments(args)

    def generate_mesh_intrinsic_statement(self, node, indent=0):
        if not isinstance(node, MeshOpNode):
            return None
        if node.operation != "DispatchMesh":
            return None

        if self.current_stage_entry_type not in {"task", "amplification", "object"}:
            return None

        args = getattr(node, "arguments", []) or []
        self.validate_dispatch_mesh_arguments(args)

        indent_str = "    " * indent
        dispatch_args = ", ".join(self.generate_expression(arg) for arg in args[:3])
        statements = []

        if len(args) == 4:
            payload_expr = args[3]
            target = self.glsl_dispatch_mesh_payload_target(payload_expr)
            if self.expression_name(payload_expr) != target["name"]:
                payload_value = self.generate_expression_with_expected(
                    payload_expr, target["raw_type"]
                )
                statements.append(f"{indent_str}{target['name']} = {payload_value};")

        statements.append(f"{indent_str}EmitMeshTasksEXT({dispatch_args});")
        return "\n".join(statements) + "\n"

    def mesh_output_role_count_limit(self, roles):
        for role in roles:
            for info in self.current_mesh_output_parameters.values():
                if info["role"] == role and info.get("count_value") is not None:
                    return info["count_value"], f"@{role} array size"

        for role in roles:
            limit = self.current_mesh_output_count_limits.get(role)
            if limit is not None:
                return limit, self.mesh_output_role_count_attribute(role)
        return None, None

    def validate_mesh_output_count_argument(self, args, index, roles, label):
        arg = args[index]
        count_type = self.expression_result_type(arg)
        if count_type is not None:
            mapped_type = self.map_type(count_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise ValueError(
                    "GLSL mesh SetMeshOutputCounts "
                    f"{label} count requires a non-negative scalar integer "
                    f"value, got {mapped_type}"
                )

        value = self.literal_int_value(arg, self.literal_int_constants)
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "GLSL mesh SetMeshOutputCounts "
                f"{label} count requires a non-negative scalar integer value, "
                f"got {self.glsl_attribute_argument_source(arg)}"
            )
        limit, limit_label = self.mesh_output_role_count_limit(roles)
        if limit is None or value <= limit:
            return
        raise ValueError(
            "GLSL mesh SetMeshOutputCounts "
            f"{label} count {value} exceeds {limit_label} {limit}"
        )

    def validate_mesh_output_index_expression(self, index_expr, context, limit):
        index_type = self.expression_result_type(index_expr)
        if index_type is not None:
            mapped_type = self.map_type(index_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise ValueError(
                    f"GLSL mesh {context} index requires a scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is None or limit is None:
            return
        if literal_index < 0 or literal_index >= limit:
            raise ValueError(
                f"GLSL mesh {context} index {literal_index} out of range; "
                f"valid range is 0..{limit - 1}"
            )

    def validate_mesh_output_helper_index(self, index_expr, roles, context):
        limit, _ = self.mesh_output_role_count_limit(roles)
        self.validate_mesh_output_index_expression(index_expr, context, limit)

    def generate_mesh_output_counts_call(self, func_name, args):
        if func_name != "SetMeshOutputCounts":
            return None
        if self.current_stage_entry_type != "mesh":
            return None
        if len(args) != 2:
            raise ValueError(
                "GLSL mesh SetMeshOutputCounts requires exactly two arguments "
                "(vertex count, primitive count)"
            )
        self.validate_mesh_output_count_argument(args, 0, ("vertices",), "vertex")
        self.validate_mesh_output_count_argument(
            args, 1, ("primitives", "indices"), "primitive"
        )
        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"SetMeshOutputsEXT({generated_args})"

    def generate_mesh_output_helper_call_statement(self, node, indent=0):
        if not isinstance(node, FunctionCallNode):
            return None
        if self.current_stage_entry_type != "mesh":
            return None

        func_name = self.function_call_name(node)
        args = getattr(node, "arguments", getattr(node, "args", [])) or []
        indent_str = "    " * indent
        helper = self.mesh_output_helper_call_components(func_name, args)
        if helper is None:
            return None

        if helper["kind"] == "comment":
            return f"{indent_str}{helper['text']};\n"
        if helper["kind"] == "vertex":
            return (
                f"{indent_str}gl_MeshVerticesEXT[{helper['index']}].gl_Position = "
                f"{helper['value']};\n"
            )
        if helper["kind"] == "primitive":
            return (
                f"{indent_str}{helper['builtin']}[{helper['index']}] = "
                f"{helper['value']};\n"
            )
        return None

    def mesh_output_helper_call_components(self, func_name, args):
        if func_name == "SetVertex":
            if len(args) != 2:
                raise ValueError(
                    "GLSL mesh SetVertex helper requires exactly two arguments "
                    "(vertex index, position)"
                )
            self.validate_mesh_output_helper_index(
                args[0], ("vertices",), "SetVertex vertex"
            )
            index = self.generate_expression(args[0])
            value = self.glsl_mesh_vertex_helper_value(args[1])
            return {"kind": "vertex", "index": index, "value": value}

        if func_name == "SetPrimitive":
            if len(args) != 2:
                raise ValueError(
                    "GLSL mesh SetPrimitive helper requires exactly two arguments "
                    "(primitive index, primitive indices)"
                )
            index_builtin = self.mesh_primitive_index_builtin(
                self.current_mesh_output_topology
            )
            if index_builtin is None:
                return {
                    "kind": "comment",
                    "text": "/* GLSL mesh SetPrimitive requires output topology */",
                }
            self.validate_mesh_output_helper_index(
                args[0], ("primitives", "indices"), "SetPrimitive primitive"
            )
            index = self.generate_expression(args[0])
            value = self.glsl_mesh_primitive_helper_value(args[1])
            return {
                "kind": "primitive",
                "builtin": index_builtin,
                "index": index,
                "value": value,
            }

        return None

    def reject_mesh_output_helper_expression_context(self, func_name, args):
        if self.current_stage_entry_type != "mesh":
            return
        if func_name not in {"SetVertex", "SetPrimitive"}:
            return
        self.mesh_output_helper_call_components(func_name, args)
        raise ValueError(
            f"GLSL mesh {func_name} helper can only be used as a statement"
        )

    def glsl_mesh_vertex_helper_value(self, expr):
        result_type = self.expression_result_type(expr)
        value_type = self.map_type(result_type) if result_type is not None else None
        if value_type is not None and value_type not in {"vec3", "vec4"}:
            raise ValueError(
                "GLSL mesh SetVertex position requires vec3 or vec4, "
                f"got {value_type}"
            )

        value = self.generate_expression_with_expected(expr, None)
        if value_type == "vec3":
            return f"vec4({value}, 1.0)"
        return value

    def glsl_mesh_primitive_helper_value(self, expr):
        expected_type = {
            "points": "uint",
            "lines": "uvec2",
            "triangles": "uvec3",
        }.get(self.current_mesh_output_topology)

        result_type = self.expression_result_type(expr)
        value_type = self.map_type(result_type) if result_type is not None else None
        if expected_type and value_type is not None and value_type != expected_type:
            raise ValueError(
                "GLSL mesh SetPrimitive for "
                f"{self.current_mesh_output_topology} topology requires "
                f"{expected_type} primitive indices, got {value_type}"
            )

        return self.generate_expression_with_expected(expr, expected_type)

    def generate_mesh_output_assignment_statement(self, node, indent=0):
        if (
            not isinstance(node, AssignmentNode)
            or not self.current_mesh_output_parameters
        ):
            return None

        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
        indent_str = "    " * indent

        target = self.mesh_output_assignment_target(left_node)
        if target is not None:
            right = self.generate_expression_with_expected(
                right_node, self.expression_result_type(left_node)
            )
            return f"{indent_str}{target} {op} {right};\n"

        expanded_targets = self.mesh_output_whole_assignment_targets(left_node)
        if expanded_targets is None or op != "=":
            return None

        access = self.mesh_output_array_access(left_node)
        constructor_values = (
            self.mesh_output_constructor_assignment_values(access[0], right_node)
            if access is not None
            else None
        )
        if constructor_values is None and access is not None:
            self.validate_mesh_output_whole_assignment_value(access[0], right_node)
        right = self.generate_expression_with_expected(
            right_node, self.expression_result_type(left_node)
        )
        code = ""
        for target, member_name, member_type in expanded_targets:
            value_expr = (
                constructor_values.get(member_name)
                if constructor_values is not None
                else None
            )
            if value_expr is None:
                value = f"{right}.{member_name}"
            else:
                value = self.generate_expression_with_expected(value_expr, member_type)
            code += f"{indent_str}{target} = {value};\n"
        return code

    def validate_mesh_output_whole_assignment_value(self, info, expr):
        expected_type = self.type_name_string(info.get("element_type"))
        if expected_type is None:
            return

        if self.mesh_output_constructor_for_type(expr, expected_type):
            raise ValueError(
                "GLSL mesh output "
                f"@{info['role']} '{info['name']}' assignment requires a "
                f"complete {expected_type} constructor value"
            )

        result_type = self.type_name_string(self.expression_result_type(expr))
        if result_type is None:
            return

        expected_label = self.map_type(expected_type)
        result_label = self.map_type(result_type)
        if result_label != expected_label:
            raise ValueError(
                "GLSL mesh output "
                f"@{info['role']} '{info['name']}' assignment requires "
                f"{expected_label} value, got {result_label}"
            )

    def mesh_output_constructor_for_type(self, expr, struct_name):
        if isinstance(expr, FunctionCallNode):
            return self.function_call_name(expr) == struct_name
        if not isinstance(expr, ConstructorNode):
            return False
        constructor_type = self.type_name_string(
            getattr(expr, "constructor_type", None)
        )
        return constructor_type == struct_name

    def mesh_output_constructor_assignment_values(self, info, expr):
        struct = self.structs_by_name.get(info["element_type"])
        if struct is None:
            return None

        members = getattr(struct, "members", []) or []
        field_values = self.mesh_output_constructor_field_values(
            expr, info["element_type"], members
        )
        if field_values is None:
            return None
        return {member.name: value_expr for member, value_expr in field_values}

    def mesh_output_constructor_field_values(self, expr, struct_name, members):
        if isinstance(expr, FunctionCallNode):
            if self.function_call_name(expr) != struct_name:
                return None
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if len(args) != len(members):
                return None
            return list(zip(members, args))

        if not isinstance(expr, ConstructorNode):
            return None

        constructor_type = self.type_name_string(
            getattr(expr, "constructor_type", None)
        )
        if constructor_type != struct_name:
            return None

        args = list(getattr(expr, "arguments", []) or [])
        named_args = dict(getattr(expr, "named_arguments", {}) or {})
        if len(args) > len(members):
            return None

        values = []
        for index, member in enumerate(members):
            if index < len(args):
                values.append((member, args[index]))
                continue
            if member.name not in named_args:
                return None
            values.append((member, named_args[member.name]))
        return values

    def mesh_output_assignment_target(self, node):
        if isinstance(node, MemberAccessNode):
            access = self.mesh_output_array_access(getattr(node, "object", None))
            if access is None:
                return None
            info, index = access
            if info["role"] not in {"vertices", "primitives"}:
                return None
            return self.mesh_output_member_target(info, index, str(node.member))

        access = self.mesh_output_array_access(node)
        if access is None:
            return None
        info, index = access
        if info["role"] != "indices":
            return None
        index_builtin = info.get("index_builtin")
        return f"{index_builtin}[{index}]" if index_builtin else None

    def mesh_output_whole_assignment_targets(self, node):
        access = self.mesh_output_array_access(node)
        if access is None:
            return None
        info, index = access
        if info["role"] not in {"vertices", "primitives"}:
            return None
        targets = []
        for member_name, member_info in info["members"].items():
            target = self.mesh_output_member_target(info, index, member_name)
            if target is None:
                return None
            targets.append((target, member_name, member_info["type"]))
        return targets

    def mesh_output_array_access(self, node):
        if not isinstance(node, ArrayAccessNode):
            return None
        array_expr = getattr(node, "array", getattr(node, "array_expr", None))
        name = self.expression_name(array_expr)
        info = self.current_mesh_output_parameters.get(name)
        if info is None:
            return None
        index_expr = getattr(node, "index", getattr(node, "index_expr", None))
        limit, _ = self.mesh_output_index_limit(info)
        self.validate_mesh_output_index_expression(
            index_expr, f"output @{info['role']} '{info['name']}'", limit
        )
        return info, self.generate_expression(index_expr)

    def mesh_output_index_limit(self, info):
        if info.get("count_value") is not None:
            return info["count_value"], f"@{info['role']} array size"
        limit = self.current_mesh_output_count_limits.get(info["role"])
        if limit is not None:
            return limit, self.mesh_output_role_count_attribute(info["role"])
        return None, None

    def mesh_output_member_target(self, info, index, member_name):
        member_info = info["members"].get(member_name)
        if member_info is None:
            return None

        semantic = member_info["semantic"]
        if info["role"] == "vertices":
            if self.is_vertex_builtin_output(semantic):
                return f"gl_MeshVerticesEXT[{index}].{semantic}"
            return f"{member_info['output_name']}[{index}]"

        if info["role"] == "primitives":
            if self.is_mesh_primitive_builtin(semantic):
                return f"gl_MeshPrimitivesEXT[{index}].{semantic}"
            return f"{member_info['output_name']}[{index}]"

        return None

    def glsl_tessellation_factor_assignment_expected_type(self, target):
        target_kind = self.glsl_tessellation_factor_assignment_target_kind(target)
        if target_kind is None:
            return None

        builtin_name, is_component, index_expr = target_kind
        if self.current_stage_entry_type not in {None, "tessellation_control"}:
            raise ValueError(
                "GLSL tessellation factor assignments are only valid in "
                "tessellation_control stages"
            )
        if not is_component:
            raise ValueError(
                f"GLSL tessellation factor {builtin_name} assignment requires "
                "an indexed scalar component target"
            )
        self.validate_glsl_tessellation_factor_assignment_index(
            builtin_name, index_expr
        )
        return "float"

    def glsl_tessellation_factor_assignment_target_kind(self, target):
        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            builtin_name = self.expression_name(array_expr)
            if builtin_name in self.GLSL_TESSELLATION_FACTOR_BUILTINS:
                index_expr = getattr(
                    target, "index", getattr(target, "index_expr", None)
                )
                return (
                    builtin_name,
                    not isinstance(array_expr, ArrayAccessNode),
                    index_expr,
                )
            return None

        builtin_name = self.expression_name(target)
        if builtin_name in self.GLSL_TESSELLATION_FACTOR_BUILTINS:
            return builtin_name, False, None
        return None

    def validate_glsl_tessellation_factor_assignment_index(
        self, builtin_name, index_expr
    ):
        index_type = self.expression_result_type(index_expr)
        if index_type is not None:
            mapped_type = self.map_type(index_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise ValueError(
                    f"GLSL tessellation factor {builtin_name} component index "
                    "requires a scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is None:
            return

        component_count = self.GLSL_TESSELLATION_FACTOR_COMPONENT_COUNTS[builtin_name]
        if literal_index < 0 or literal_index >= component_count:
            raise ValueError(
                f"GLSL tessellation factor {builtin_name} component index "
                f"{literal_index} out of range; valid range is "
                f"0..{component_count - 1}"
            )

    def validate_glsl_tessellation_factor_assignment_value(self, target, value):
        builtin_name = self.expression_name(target)
        value_type = self.expression_result_type(value)
        if value_type is None:
            return
        mapped_type = self.map_type(value_type)
        if not self.is_scalar_numeric_type(mapped_type):
            raise ValueError(
                f"GLSL tessellation factor {builtin_name} component assignment "
                f"requires a scalar numeric value, got {mapped_type}"
            )

    def generate_assignment(self, node, is_main=False):
        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
        self.validate_glsl_buffer_block_assignment_target(left_node, op)
        expected_type = self.glsl_tessellation_factor_assignment_expected_type(
            left_node
        )
        if expected_type is not None:
            self.validate_glsl_tessellation_factor_assignment_value(
                left_node, right_node
            )
        else:
            expected_type = self.expression_result_type(left_node)
        left = self.generate_glsl_buffer_block_mutation_target(left_node)
        right = self.generate_expression_with_expected(right_node, expected_type)
        return f"{left} {op} {right}"

    def generate_if(self, node, indent, is_main=False):
        indent_str = "    " * indent
        condition = self.generate_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = node.if_body
        code += self.generate_scoped_statement_body(if_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.generate_expression(else_if_condition)
                code += f" else if ({condition}) {{\n"
                code += self.generate_scoped_statement_body(else_if_body, indent + 1)
                code += f"{indent_str}}}"

        if hasattr(node, "else_body") and node.else_body:
            code += " else {\n"
            else_body = node.else_body
            code += self.generate_scoped_statement_body(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)

        try:
            init = self.generate_for_initializer(getattr(node, "init", None))
            condition = (
                self.generate_expression(node.condition)
                if getattr(node, "condition", None)
                else ""
            )
            update = (
                self.generate_expression(node.update)
                if getattr(node, "update", None)
                else ""
            )

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

            body = node.body
            code += self.generate_scoped_statement_body(body, indent + 1)

            code += f"{indent_str}}}\n"

            return code
        finally:
            self.local_variable_types = previous_local_variable_types

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)

        try:
            self.local_variable_types[pattern] = "int"

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
        expression_type = self.map_type(
            self.expression_result_type(getattr(node, "expression", None))
        )
        if is_switch_lowerable_match(node) and expression_type != "bool":
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "GLSL")

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
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases

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

    def glsl_target_builtin_identifier(self, name):
        if (
            normalize_stage_name(getattr(self, "current_stage_entry_type", None))
            == "vertex"
            and name == "gl_VertexIndex"
        ):
            return "gl_VertexID"
        return name

    def generate_expression(self, expr, is_main=False):
        """Render a CrossGL AST expression into GLSL expression syntax."""
        if expr is None:
            return ""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "VariableNode" in str(type(expr)):
            if hasattr(expr, "name"):
                if expr.name in self.current_resource_aliases:
                    return self.current_resource_aliases[expr.name]["expression"]
                if expr.name in getattr(self, "enum_variant_constants", {}):
                    return enum_value_expression(self, expr.name)
                if expr.name in self.current_identifier_aliases:
                    return self.current_identifier_aliases[expr.name]
                return self.current_stage_parameter_aliases.get(
                    expr.name, self.glsl_target_builtin_identifier(expr.name)
                )
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "IdentifierNode" in str(type(expr)):
            if expr.name in self.current_resource_aliases:
                return self.current_resource_aliases[expr.name]["expression"]
            if expr.name in getattr(self, "enum_variant_constants", {}):
                return enum_value_expression(self, expr.name)
            if expr.name in self.current_identifier_aliases:
                return self.current_identifier_aliases[expr.name]
            return self.current_stage_parameter_aliases.get(
                expr.name, self.glsl_target_builtin_identifier(expr.name)
            )
        elif isinstance(expr, ArrayLiteralNode):
            elements = ", ".join(
                self.generate_expression(element) for element in expr.elements
            )
            return f"{{ {elements} }}"
        elif hasattr(expr, "__class__") and "LiteralNode" in str(type(expr)):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type == "string":
                return f'"{expr.value}"'
            if literal_type == "char":
                return f"'{expr.value}'"
            if (
                literal_type == "uint"
                and isinstance(expr.value, int)
                and not isinstance(expr.value, bool)
            ):
                return f"{expr.value}u"
            if isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            return str(expr.value)
        elif hasattr(expr, "__class__") and "BinaryOpNode" in str(type(expr)):
            op = self.map_operator(expr.op)
            vector_relational = self.generate_vector_relational_expression(
                expr.left, op, expr.right
            )
            if vector_relational is not None:
                return vector_relational
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"({left} {op} {right})"
        elif hasattr(expr, "__class__") and "AssignmentNode" in str(type(expr)):
            return self.generate_assignment(expr)
        elif hasattr(expr, "__class__") and "UnaryOpNode" in str(type(expr)):
            op = self.map_operator(expr.op)
            if op in {"++", "--"} or expr.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }:
                self.validate_glsl_buffer_block_member_access(
                    expr.operand, "read_write"
                )
                operand = self.generate_glsl_buffer_block_mutation_target(expr.operand)
            else:
                operand = self.generate_expression(expr.operand)
            return f"({op}{operand})"
        elif isinstance(expr, WaveOpNode):
            return self.generate_glsl_wave_op_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            args = [self.generate_expression(arg) for arg in expr.arguments]
            return self.map_ray_tracing_intrinsic(expr.operation, args)
        elif isinstance(expr, MeshOpNode):
            mesh_output_counts_call = self.generate_mesh_output_counts_call(
                expr.operation, expr.arguments
            )
            if mesh_output_counts_call is not None:
                return mesh_output_counts_call
            arguments = expr.arguments
            if expr.operation == "DispatchMesh":
                if self.current_stage_entry_type in {
                    "task",
                    "amplification",
                    "object",
                }:
                    self.validate_dispatch_mesh_arguments(arguments)
                if len(arguments) == 4:
                    arguments = arguments[:3]
            args = ", ".join(self.generate_expression(arg) for arg in arguments)
            operation = self.map_mesh_intrinsic(expr.operation)
            return f"{operation}({args})"
        elif isinstance(expr, RayQueryOpNode):
            return self.map_ray_query_intrinsic(
                expr.operation, expr.query_expr, expr.arguments
            )
        elif hasattr(expr, "__class__") and "ArrayAccessNode" in str(type(expr)):
            if hasattr(expr, "array") and hasattr(expr, "index"):
                array = self.generate_expression(expr.array)
                index = self.glsl_resource_array_index_expression(expr)
                return f"{array}[{index}]"
            else:
                return str(expr)
        elif isinstance(expr, ConstructorNode):
            enum_constructor = generate_enum_constructor_expression(self, expr)
            if enum_constructor is not None:
                return enum_constructor
            constructor = generate_struct_constructor_expression(self, expr)
            if constructor is not None:
                return constructor
            return str(expr)
        elif hasattr(expr, "__class__") and "FunctionCallNode" in str(type(expr)):
            func_expr = getattr(expr, "function", getattr(expr, "name", expr))
            numeric_trait_call = generate_numeric_trait_method_call(
                self,
                func_expr,
                expr.args,
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
            original_func_name = func_name

            mesh_output_counts_call = self.generate_mesh_output_counts_call(
                original_func_name, expr.args
            )
            if mesh_output_counts_call is not None:
                return mesh_output_counts_call
            self.reject_mesh_output_helper_expression_context(
                original_func_name, expr.args
            )

            ray_query_member_call = self.ray_query_member_function_call(
                func_expr, expr.args
            )
            if ray_query_member_call is not None:
                return ray_query_member_call

            synchronization_call = self.synchronization_function_call(
                original_func_name, expr.args
            )
            if synchronization_call is not None:
                return synchronization_call

            if original_func_name in self.GLSL_WAVE_INTRINSIC_ARITIES:
                return self.generate_glsl_wave_operation(original_func_name, expr.args)

            glsl_memory_atomic_call = self.generate_glsl_memory_atomic_call(
                original_func_name, expr.args
            )
            if glsl_memory_atomic_call is not None:
                return glsl_memory_atomic_call

            saturate_call = self.generate_saturate_call(original_func_name, expr.args)
            if saturate_call is not None:
                return saturate_call

            reciprocal_call = self.generate_reciprocal_call(
                original_func_name, expr.args
            )
            if reciprocal_call is not None:
                return reciprocal_call

            bitcast_call = self.generate_bitcast_call(original_func_name, expr.args)
            if bitcast_call is not None:
                return bitcast_call

            mul_call = self.generate_mul_call(original_func_name, expr.args)
            if mul_call is not None:
                return mul_call

            mad_call = self.generate_mad_call(original_func_name, expr.args)
            if mad_call is not None:
                return mad_call

            if original_func_name not in self.function_return_types:
                func_name = self.function_map.get(func_name, func_name)
            self.validate_fragment_only_helper_call(original_func_name)

            static_generic_call = generate_static_generic_numeric_call(
                self,
                original_func_name,
            )
            if static_generic_call is not None:
                return static_generic_call

            enum_constructor = generate_enum_constructor_call(
                self, original_func_name, expr.args
            )
            if enum_constructor is not None:
                return enum_constructor

            texture_call = self.generate_texture_call(func_name, expr.args)
            if texture_call is not None:
                return texture_call

            interpolation_call = self.generate_interpolation_call(func_name, expr.args)
            if interpolation_call is not None:
                return interpolation_call

            derivative_call = self.generate_derivative_call(func_name, expr.args)
            if derivative_call is not None:
                return derivative_call

            buffer_call = self.generate_buffer_call(func_name, expr.args)
            if buffer_call is not None:
                return buffer_call

            specialized_func_name = generic_function_call_name(
                self,
                original_func_name,
                expr.args,
            )
            if specialized_func_name is not None:
                func_name = specialized_func_name
                callee = specialized_func_name

            constructor = self.glsl_constructor_type(func_name)
            if constructor:
                args = ", ".join(
                    self.generate_expression_with_expected(arg, None)
                    for arg in expr.args
                )
                return f"{constructor}({args})"

            self.validate_function_structured_buffer_access_arguments(
                func_name, expr.args
            )
            self.validate_function_image_parameter_contract_arguments(
                func_name, expr.args
            )
            self.validate_function_image_access_arguments(func_name, expr.args)
            resource_specialization = self.glsl_resource_function_call_specialization(
                func_name,
                expr.args,
            )
            argument_func_name = original_func_name
            if resource_specialization is not None:
                func_name = resource_specialization.name
                callee = resource_specialization.name
                argument_func_name = func_name
                call_args = self.glsl_resource_specialized_call_arguments(
                    resource_specialization,
                    expr.args,
                )
                call_args = self.filter_sampler_arguments(func_name, call_args)
            else:
                call_args = self.filter_sampler_arguments(original_func_name, expr.args)
            args = ", ".join(
                self.generate_function_call_arguments(argument_func_name, call_args)
            )
            return f"{func_name or callee}({args})"
        elif hasattr(expr, "__class__") and "MemberAccessNode" in str(type(expr)):
            flattened_member = self.flattened_stage_member_name(expr)
            if flattened_member is not None:
                return flattened_member
            if not self.glsl_buffer_block_read_validation_suppression:
                self.validate_glsl_buffer_block_member_access(expr, "read")
            obj = self.generate_expression_with_expected(expr.object, None)
            member_name = self.glsl_member_access_name(expr.object, str(expr.member))
            cast_member = self.signed_compute_builtin_member_cast(obj, member_name)
            if cast_member is not None:
                return cast_member
            return f"{obj}.{member_name}"
        elif hasattr(expr, "__class__") and "TernaryOpNode" in str(type(expr)):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        else:
            return str(expr)

    def synchronization_function_call(self, func_name, args):
        if args or func_name in self.function_return_types:
            return None
        if func_name == "workgroupBarrier":
            return "barrier()"
        return None

    def generate_glsl_wave_op_expression(self, node):
        return self.generate_glsl_wave_operation(node.operation, node.arguments)

    def generate_saturate_call(self, func_name, args):
        if func_name != "saturate" or func_name in self.function_return_types:
            return None
        if len(args) != 1:
            raise ValueError("OpenGL saturate alias requires 1 argument")

        value = self.generate_expression(args[0])
        zero, one = self.saturate_bound_literals(args[0])
        return f"clamp({value}, {zero}, {one})"

    def generate_reciprocal_call(self, func_name, args):
        if func_name != "rcp" or func_name in self.function_return_types:
            return None
        if len(args) != 1:
            raise ValueError("OpenGL rcp alias requires 1 argument")

        value_expr = args[0]
        value_type = self.map_type(self.expression_result_type(value_expr))
        component_type = self.vector_component_type(value_type) or value_type
        if component_type not in {"float", "double"}:
            return None

        value = self.generate_expression(value_expr)
        if self.is_scalar_value_type(value_type) or self.is_vector_value_type(
            value_type
        ):
            return f"(1.0 / {value})"
        return None

    def generate_bitcast_call(self, func_name, args):
        if (
            func_name not in self.GLSL_BITCAST_FUNCTIONS
            or func_name in self.function_return_types
        ):
            return None
        if len(args) != 1:
            raise ValueError(f"OpenGL {func_name} alias requires 1 argument")

        target = self.glsl_bitcast_target(func_name, args[0])
        if target is None:
            source_type = self.map_type(self.expression_result_type(args[0]))
            raise ValueError(
                f"OpenGL {func_name} alias cannot bitcast from {source_type}"
            )

        value = self.generate_expression(args[0])
        if target == "__crossgl_bfloat16_to_uint":
            return f"(floatBitsToUint({value}) >> 16u)"
        return f"{target}({value})"

    def glsl_bitcast_target(self, func_name, value_expr):
        source_type = self.expression_result_type(value_expr)
        source_type_name = self.type_name_string(source_type)
        source_component_type = (
            self.vector_component_type(source_type_name) or source_type_name
        )
        if (
            func_name == "asuint"
            and source_component_type in self.GLSL_BFLOAT16_ALIASES
        ):
            return "__crossgl_bfloat16_to_uint"

        value_type = self.map_type(source_type)
        component_type = self.vector_component_type(value_type) or value_type
        if func_name == "asfloat":
            if component_type == "int":
                return "intBitsToFloat"
            if component_type == "uint":
                return "uintBitsToFloat"
        if func_name == "asint" and component_type == "float":
            return "floatBitsToInt"
        if func_name == "asuint" and component_type == "float":
            return "floatBitsToUint"
        return None

    def glsl_bitcast_result_type(self, func_name, value_expr):
        value_type = self.map_type(self.expression_result_type(value_expr))
        if self.glsl_bitcast_target(func_name, value_expr) is None:
            return None

        if func_name == "asfloat":
            if value_type in {"int", "uint"}:
                return "float"
            if value_type.endswith(("2", "3", "4")):
                return f"vec{value_type[-1]}"
        if func_name == "asint":
            if value_type == "float":
                return "int"
            if value_type.endswith(("2", "3", "4")):
                return f"ivec{value_type[-1]}"
        if func_name == "asuint":
            if value_type == "float":
                return "uint"
            if value_type.endswith(("2", "3", "4")):
                return f"uvec{value_type[-1]}"
        return None

    def is_elided_bfloat16_type_alias_global(self, node):
        if not isinstance(node, VariableNode):
            return False
        if getattr(node, "name", None) not in self.GLSL_BFLOAT16_ALIASES:
            return False
        if getattr(node, "initial_value", None) is not None:
            return False
        if getattr(node, "attributes", None):
            return False
        if getattr(node, "qualifiers", None):
            return False
        return self.type_name_string(getattr(node, "var_type", None)) in {
            "f16",
            "float16",
            "bfloat",
        }

    def generate_mul_call(self, func_name, args):
        if func_name != "mul" or func_name in self.function_return_types:
            return None
        if len(args) != 2:
            raise ValueError("OpenGL mul alias requires 2 arguments")

        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        return f"({left} * {right})"

    def generate_mad_call(self, func_name, args):
        if func_name != "mad" or func_name in self.function_return_types:
            return None
        if len(args) != 3:
            raise ValueError("OpenGL mad alias requires 3 arguments")

        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        addend = self.generate_expression(args[2])
        return f"(({left} * {right}) + {addend})"

    def generate_clip_statement(self, expression, indent):
        if not isinstance(expression, FunctionCallNode):
            return None

        func_name = self.function_call_name(expression)
        if func_name != "clip" or func_name in self.function_return_types:
            return None

        args = getattr(expression, "arguments", getattr(expression, "args", [])) or []
        if len(args) != 1:
            raise ValueError("OpenGL clip alias requires 1 argument")
        if self.current_stage_entry_type not in (None, "fragment"):
            raise ValueError(
                "OpenGL clip alias lowers to discard and is only valid in "
                "fragment stages"
            )

        value_expr = args[0]
        value = self.generate_expression(value_expr)
        condition = self.clip_discard_condition(value_expr, value)
        indent_str = "    " * indent
        return (
            f"{indent_str}if ({condition}) {{\n"
            f"{indent_str}    discard;\n"
            f"{indent_str}}}\n"
        )

    def clip_discard_condition(self, value_expr, value):
        value_type = self.map_type(self.expression_result_type(value_expr))
        zero = self.clip_zero_literal(value_type)
        if self.is_vector_value_type(value_type):
            return f"any(lessThan({value}, {value_type}({zero})))"
        return f"{value} < {zero}"

    def clip_zero_literal(self, value_type):
        component_type = self.vector_component_type(value_type) or value_type
        if component_type == "uint":
            return "0u"
        if component_type == "int":
            return "0"
        return "0.0"

    def saturate_bound_literals(self, value_expr):
        value_type = self.expression_result_type(value_expr)
        mapped_type = self.map_type(value_type)
        component_type = self.vector_component_type(mapped_type) or mapped_type
        if component_type == "uint":
            return "0u", "1u"
        if component_type == "int":
            return "0", "1"
        return "0.0", "1.0"

    def generate_glsl_wave_operation(self, operation, arguments):
        expected_arity = self.GLSL_WAVE_INTRINSIC_ARITIES.get(operation)
        if expected_arity is None:
            return self.glsl_wave_diagnostic_expression(
                operation, "is not recognized by the OpenGL backend"
            )

        actual_arity = len(arguments)
        if actual_arity != expected_arity:
            return self.glsl_wave_diagnostic_expression(
                operation, f"expects {expected_arity} arguments, got {actual_arity}"
            )

        type_diagnostic = self.glsl_wave_type_diagnostic(operation, arguments)
        if type_diagnostic is not None:
            return type_diagnostic

        if operation == "WaveGetLaneCount":
            return "gl_SubgroupSize"
        if operation == "WaveGetLaneIndex":
            return "gl_SubgroupInvocationID"
        if operation == "WaveIsFirstLane":
            return "subgroupElect()"

        if operation == "WaveActiveCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"subgroupBallotBitCount(subgroupBallot({predicate}))"
        if operation == "WavePrefixCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"subgroupBallotExclusiveBitCount(subgroupBallot({predicate}))"
        if operation == "WaveMatch":
            value = self.generate_expression(arguments[0])
            return f"crossglWaveMatch({value})"
        if operation in self.GLSL_WAVE_MULTI_PREFIX_OPERATIONS:
            value = self.generate_expression(arguments[0])
            mask = self.generate_expression(arguments[1])
            helper = self.GLSL_WAVE_MULTI_PREFIX_HELPERS[operation]
            return f"{helper}({value}, {mask})"

        mapped = self.GLSL_WAVE_DIRECT_MAPPINGS.get(operation)
        if mapped is None:
            return self.glsl_wave_diagnostic_expression(
                operation, "is not recognized by the OpenGL backend"
            )

        args = ", ".join(self.generate_expression(arg) for arg in arguments)
        return f"{mapped}({args})"

    def glsl_wave_result_type(self, operation, arguments):
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return "uint"
        if operation in {
            "WaveIsFirstLane",
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveAllEqual",
        }:
            return "bool"
        if operation in {
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            return "uint"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uvec4"
        if arguments:
            return self.expression_result_type(arguments[0])
        return None

    def glsl_wave_type_diagnostic(self, operation, args):
        if operation in self.GLSL_WAVE_NUMERIC_OPERATIONS:
            return self.glsl_wave_validate_argument_kind(
                operation,
                args[0],
                "a numeric scalar or vector",
                "value",
                {"float", "double", "int", "uint"},
            )
        if operation in self.GLSL_WAVE_INTEGER_OR_BOOLEAN_OPERATIONS:
            return self.glsl_wave_validate_argument_kind(
                operation,
                args[0],
                "an integer or boolean scalar or vector",
                "value",
                {"int", "uint", "bool"},
            )
        if operation in self.GLSL_WAVE_BOOLEAN_OPERATIONS:
            return self.glsl_wave_validate_argument_type(
                operation,
                args[0],
                "a boolean scalar",
                "value",
                {"bool"},
                allow_vectors=False,
            )
        if operation in self.GLSL_WAVE_SCALAR_OR_VECTOR_OPERATIONS:
            return self.glsl_wave_validate_argument_type(
                operation,
                args[0],
                "a scalar or vector",
                "value",
                {"float", "double", "int", "uint", "bool"},
                allow_vectors=True,
            )
        if operation == "WaveMatch":
            return self.glsl_wave_validate_argument_type(
                operation,
                args[0],
                "a scalar or vector",
                "value",
                {"float", "double", "int", "uint", "bool"},
                allow_vectors=True,
            )
        if operation == "WaveMultiPrefixCountBits":
            diagnostic = self.glsl_wave_validate_argument_type(
                operation,
                args[0],
                "a boolean scalar",
                "value",
                {"bool"},
                allow_vectors=False,
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(operation, args[1])
        if operation in self.GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS:
            diagnostic = self.glsl_wave_validate_argument_kind(
                operation,
                args[0],
                "a numeric scalar or vector",
                "value",
                {"float", "double", "int", "uint"},
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(operation, args[1])
        if operation in self.GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS:
            diagnostic = self.glsl_wave_validate_argument_kind(
                operation,
                args[0],
                "an integer scalar or vector",
                "value",
                {"int", "uint"},
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(operation, args[1])
        index_argument = self.GLSL_WAVE_LANE_INDEX_ARGUMENTS.get(operation)
        if index_argument is not None:
            return self.glsl_wave_validate_argument_type(
                operation,
                args[index_argument],
                "a scalar integer",
                "lane",
                {"int", "uint"},
                allow_vectors=False,
            )
        return None

    def glsl_wave_validate_argument_kind(
        self, operation, arg, requirement, argument_label, allowed_kinds
    ):
        return self.glsl_wave_validate_argument_type(
            operation,
            arg,
            requirement,
            argument_label,
            allowed_kinds,
            allow_vectors=True,
        )

    def glsl_wave_validate_argument_type(
        self,
        operation,
        arg,
        requirement,
        argument_label,
        allowed_kinds,
        allow_vectors,
    ):
        result_type = self.expression_result_type(arg)
        if result_type is None:
            return None
        mapped_type = self.map_type(result_type)
        component_kind = self.vector_component_type(mapped_type)
        if component_kind is not None:
            if allow_vectors and component_kind in allowed_kinds:
                return None
        elif mapped_type in allowed_kinds:
            return None

        return self.glsl_wave_diagnostic_expression(
            operation,
            (
                f"requires {requirement} {argument_label} argument: "
                f"{expression_debug_name(arg)} has type {mapped_type}"
            ),
        )

    def glsl_wave_validate_mask_argument(self, operation, arg):
        result_type = self.expression_result_type(arg)
        if result_type is None:
            return None
        mapped_type = self.map_type(result_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix or mapped_type != "uvec4":
            return self.glsl_wave_diagnostic_expression(
                operation,
                (
                    "requires a uvec4 partition mask argument: "
                    f"{expression_debug_name(arg)} has type {mapped_type}"
                ),
            )
        return None

    def glsl_wave_diagnostic_expression(self, operation, reason):
        return (
            f"/* GLSL wave intrinsic diagnostic: {operation} {reason} */ "
            f"{self.glsl_wave_default_value(operation)}"
        )

    def glsl_wave_default_value(self, operation):
        expected_type = self.current_expression_expected_type
        if expected_type:
            return self.zero_value_expression(expected_type)
        if operation in {
            "WaveIsFirstLane",
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveAllEqual",
        }:
            return "false"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uvec4(0u)"
        return "0u"

    def generate_interpolation_call(self, func_name, args):
        expected_args = self.GLSL_INTERPOLATION_FUNCTIONS.get(func_name)
        if expected_args is None:
            return None
        if len(args) != expected_args:
            noun = "argument" if expected_args == 1 else "arguments"
            raise ValueError(
                f"OpenGL interpolation operation '{func_name}' requires "
                f"{expected_args} {noun}"
            )
        if self.current_stage_entry_type not in (None, "fragment"):
            raise ValueError(
                f"OpenGL interpolation operation '{func_name}' is only valid "
                "in fragment stages"
            )

        if func_name == "interpolateAtSample":
            sample_type = self.expression_result_type(args[1])
            if sample_type is not None and not self.is_scalar_integer_type(sample_type):
                raise ValueError(
                    operation_argument_type_error(
                        "OpenGL",
                        "interpolation",
                        func_name,
                        "a scalar integer",
                        "sample",
                        expression_debug_name(args[1]),
                        self.type_name_string(sample_type),
                    )
                )

        if func_name == "interpolateAtOffset":
            offset_type = self.expression_result_type(args[1])
            if offset_type is not None and self.map_type(offset_type) != "vec2":
                raise ValueError(
                    operation_argument_type_error(
                        "OpenGL",
                        "interpolation",
                        func_name,
                        "a vec2 floating",
                        "offset",
                        expression_debug_name(args[1]),
                        self.type_name_string(offset_type),
                    )
                )

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{func_name}({generated_args})"

    def generate_derivative_call(self, func_name, args):
        expected_args = self.GLSL_DERIVATIVE_FUNCTIONS.get(func_name)
        if expected_args is None:
            return None
        if len(args) != expected_args:
            raise ValueError(
                f"OpenGL derivative operation '{func_name}' requires "
                f"{expected_args} argument"
            )
        if self.current_stage_entry_type not in (None, "fragment"):
            raise ValueError(
                f"OpenGL derivative operation '{func_name}' is only valid "
                "in fragment stages"
            )

        value_type = self.expression_result_type(args[0])
        component_type = self.vector_component_type(value_type)
        if value_type is not None and not (
            self.is_scalar_floating_type(value_type)
            or component_type in {"float", "double"}
        ):
            raise ValueError(
                operation_argument_type_error(
                    "OpenGL",
                    "derivative",
                    func_name,
                    "a floating scalar or vector",
                    "value",
                    expression_debug_name(args[0]),
                    self.type_name_string(value_type),
                )
            )

        generated_arg = self.generate_expression(args[0])
        return f"{func_name}({generated_arg})"

    def ray_query_member_function_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        if func_expr.member not in self.RAY_QUERY_METHOD_MAP:
            return None
        return self.map_ray_query_intrinsic(func_expr.member, func_expr.object, args)

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            self.validate_structured_buffer_access_argument(func_name, args)
            index = self.generate_expression_with_expected(args[1], None)
            return f"{self.structured_buffer_access_expression(args[0], index)}"
        if func_name == "buffer_store" and len(args) >= 3:
            self.validate_structured_buffer_access_argument(func_name, args)
            index = self.generate_expression_with_expected(args[1], None)
            value = self.generate_expression(args[2])
            array_access = self.structured_buffer_array_parameter_access(args[0])
            if array_access is not None:
                info, selector = array_access
                return self.structured_buffer_array_select_expression(
                    info,
                    selector,
                    lambda name: f"({name}[{index}] = {value})",
                )
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_append" and len(args) >= 2:
            self.validate_structured_buffer_access_argument(func_name, args)
            value = self.generate_expression(args[1])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "/* unsupported GLSL buffer append: requires append/consume "
                    "counter buffer */"
                )
            index = f"atomicAdd({counter}, 1u)"
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_consume" and args:
            self.validate_structured_buffer_access_argument(func_name, args)
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return (
                    "0 /* unsupported GLSL buffer consume: requires append/consume "
                    "counter buffer */"
                )
            index = f"(atomicAdd({counter}, uint(-1)) - 1u)"
            return self.structured_buffer_access_expression(args[0], index)
        if func_name == "buffer_dimensions" and args:
            length_expr = self.structured_buffer_length_expression(args[0])
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {length_expr}"
            return length_expr
        return None

    def structured_buffer_access_expression(self, buffer_arg, element_index):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}[{element_index}]",
            )
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}[{element_index}]"
        return f"{buffer}[{element_index}]"

    def structured_buffer_length_expression(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}.length()",
            )
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}.length()"
        return f"{buffer}.length()"

    def structured_buffer_counter_reference(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: f"{name}[0]",
                )

        counter_data = self.structured_buffer_counter_data_argument(buffer_arg)
        if counter_data is None:
            return None
        return f"{counter_data}[0]"

    def flattened_stage_member_name(self, expr):
        object_name = self.expression_name(getattr(expr, "object", None))
        if object_name in self.current_stage_inputs:
            return self.current_stage_inputs[object_name].get(expr.member)
        if object_name in self.current_stage_outputs:
            return self.current_stage_outputs[object_name].get(expr.member)
        return None

    def glsl_member_access_name(self, object_expr, member_name):
        object_type = self.type_name_string(self.expression_result_type(object_expr))
        if not object_type:
            return member_name
        struct_name, _array_suffix = split_array_type_suffix(object_type)
        if struct_name not in self.structs_by_name:
            return member_name
        return self.glsl_struct_member_name(struct_name, member_name)

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

    def glsl_resource_array_index_expression(self, expr):
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if (
            isinstance(literal_index, int)
            and not isinstance(literal_index, bool)
            and literal_index < 0
            and self.is_resource_array_access(expr)
        ):
            return "0"
        return self.generate_expression_with_expected(index_expr, None)

    def is_resource_array_access(self, expr):
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        array_name = self.expression_name(array_expr)
        if not array_name:
            return False
        if array_name in self.texture_variable_types:
            return True
        if array_name in self.current_texture_parameters:
            return True
        array_type = self.expression_result_type(array_expr)
        return self.is_resource_array_hint_type(array_type)

    def collect_function_parameter_types(self, functions):
        parameter_types = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            types = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                types.append(self.type_name_string(raw_type))
            parameter_types[func_name] = types
        return parameter_types

    def collect_function_parameter_infos(self, functions):
        parameter_infos = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            infos = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                infos.append(
                    (getattr(param, "name", None), self.type_name_string(raw_type))
                )
            parameter_infos[func_name] = infos
        return parameter_infos

    def function_call_argument_pairs(self, func_name, call_args):
        parameter_infos = list(self.function_parameter_infos.get(func_name, []))
        if not parameter_infos:
            parameter_infos = [
                (None, param_type)
                for param_type in self.function_parameter_types.get(func_name, [])
            ]
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        filtered_parameter_infos = [
            param_info
            for index, param_info in enumerate(parameter_infos)
            if index not in skipped_indices
        ]
        pairs = []
        for index, arg in enumerate(call_args):
            param_name, param_type = (
                filtered_parameter_infos[index]
                if index < len(filtered_parameter_infos)
                else (None, None)
            )
            pairs.append((arg, param_name, param_type))
        return pairs

    def generate_function_call_arguments(self, func_name, call_args):
        args = []
        for arg, param_name, param_type in self.function_call_argument_pairs(
            func_name, call_args
        ):
            generated = self.generate_function_call_argument(
                func_name, arg, param_name, param_type
            )
            if isinstance(generated, list):
                args.extend(generated)
            else:
                args.append(generated)
        return args

    def generate_function_call_argument(self, func_name, arg, param_name, param_type):
        buffer_array = self.structured_buffer_array_parameter_info(
            param_type, param_name, func_name
        )
        if buffer_array is not None:
            generated_args = self.structured_buffer_array_data_arguments(
                arg, buffer_array["count"]
            )
            if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                generated_args += self.structured_buffer_array_counter_arguments(
                    arg, buffer_array["count"]
                )
            return generated_args
        if (
            param_type is not None
            and self.is_structured_buffer_type(param_type)
            and not self.is_array_type(param_type)
        ):
            data_arg = self.structured_buffer_data_argument(arg)
            if self.structured_buffer_requires_counter(param_type):
                return [data_arg, self.structured_buffer_counter_data_argument(arg)]
            return data_arg
        return self.generate_expression(arg)

    def structured_buffer_array_data_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            expanded_names = expanded["expanded_names"]
            return expanded_names[:count]

        arg_code = self.generate_expression(arg)
        instance_member = self.structured_buffer_instance_members.get(arg_name)
        if instance_member:
            return [f"{arg_code}[{index}].{instance_member}" for index in range(count)]
        return [f"{arg_code}[{index}]" for index in range(count)]

    def structured_buffer_data_argument(self, arg):
        arg_code = self.generate_expression(arg)
        buffer_name = self.expression_name(arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{arg_code}.{instance_member}"
        return arg_code

    def structured_buffer_array_counter_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            counter_names = expanded.get("counter_expanded_names", [])
            return counter_names[:count]

        instance_name = self.structured_buffer_counter_instances.get(arg_name)
        member_name = self.structured_buffer_counter_members.get(arg_name)
        if instance_name and member_name:
            return [f"{instance_name}[{index}].{member_name}" for index in range(count)]
        return [self.structured_buffer_counter_data_argument(arg)] * count

    def structured_buffer_counter_data_argument(self, arg):
        array_access = self.structured_buffer_array_parameter_access(arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: name,
                )

        arg_name = self.expression_name(arg)
        counter_parameter = self.current_structured_buffer_counter_parameters.get(
            arg_name
        )
        if counter_parameter is not None:
            return counter_parameter

        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            instance_name = self.structured_buffer_counter_instances.get(array_name)
            member_name = self.structured_buffer_counter_members.get(array_name)
            if instance_name and member_name:
                index = self.generate_expression(index_expr)
                return f"{instance_name}[{index}].{member_name}"

        member_name = self.structured_buffer_counter_members.get(arg_name)
        if member_name and arg_name not in self.structured_buffer_counter_instances:
            return member_name
        return None

    def structured_buffer_array_parameter_access(self, buffer_arg):
        if not isinstance(buffer_arg, ArrayAccessNode) and not (
            hasattr(buffer_arg, "__class__")
            and "ArrayAccess" in str(buffer_arg.__class__)
        ):
            return None
        array_expr = getattr(
            buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
        )
        array_name = self.expression_name(array_expr)
        info = self.current_structured_buffer_array_parameters.get(array_name)
        if info is None:
            return None
        selector = getattr(buffer_arg, "index", getattr(buffer_arg, "index_expr", None))
        return info, selector

    def structured_buffer_array_select_expression(self, info, selector, branch):
        expanded_names = info["expanded_names"]
        literal_index = self.literal_int_value(selector, self.literal_int_constants)
        if literal_index is not None and 0 <= literal_index < len(expanded_names):
            return branch(expanded_names[literal_index])

        selector_code = self.generate_expression(selector)
        selector_type = self.map_type(self.expression_result_type(selector))

        def selector_literal(index):
            return f"{index}u" if selector_type == "uint" else str(index)

        expression = branch(expanded_names[-1])
        for index in range(len(expanded_names) - 2, -1, -1):
            expression = (
                f"(({selector_code} == {selector_literal(index)}) ? "
                f"{branch(expanded_names[index])} : {expression})"
            )
        return expression

    def is_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return True
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        return "[" in type_string and "]" in type_string

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

    def texture_call_parts(self, args):
        explicit_sampler = self.is_explicit_sampler_argument(args)
        coord_index = 2 if explicit_sampler else 1
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        extra_args = args[coord_index + 1 :]
        return texture_name, coord, extra_args

    def texture_resource_type(self, texture_arg):
        return self.sampled_image_type(self.texture_declared_resource_type(texture_arg))

    def texture_declared_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        alias = self.current_resource_aliases.get(texture_name)
        if alias is not None:
            return alias.get("type")
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def sampled_image_type(self, texture_type):
        base_type = self.resource_base_type(texture_type)
        texture_to_sampler = {
            "texture1D": "sampler1D",
            "texture1DArray": "sampler1DArray",
            "texture2D": "sampler2D",
            "texture3D": "sampler3D",
            "textureCube": "samplerCube",
            "texture2DArray": "sampler2DArray",
            "textureCubeArray": "samplerCubeArray",
        }
        sampled_type = texture_to_sampler.get(base_type)
        if sampled_type is None:
            return texture_type
        if "[" not in str(texture_type):
            return sampled_type
        _, array_suffix = split_array_type_suffix(str(texture_type))
        return f"{sampled_type}{array_suffix}"

    def is_separate_texture_object_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "texture1D",
            "texture1DArray",
            "texture2D",
            "texture3D",
            "textureCube",
            "texture2DArray",
            "textureCubeArray",
        }

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        return self.expression_result_type(texture_arg)

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_inferable_resource_array_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"OpenGL texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_type(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise ValueError(
            f"OpenGL image operation '{func_name}' requires a storage "
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
                f"OpenGL image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got writeonly"
            )
        if func_name == "imageStore" and access == "read":
            raise ValueError(
                f"OpenGL image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got readonly"
            )
        if is_image_atomic_operation(func_name):
            access_name = "readonly" if access == "read" else "writeonly"
            raise ValueError(
                f"OpenGL image operation '{func_name}' requires read-write "
                f"storage image access for {texture_name}: got {access_name}"
            )

    def validate_structured_buffer_access_argument(self, func_name, args):
        required_access = self.structured_buffer_operation_access_requirement(func_name)
        if required_access is None or not args:
            return
        access = self.structured_buffer_resource_access(args[0])
        if image_access_satisfies_requirement(required_access, access):
            return
        buffer_name = expression_debug_name(args[0])
        required_label = image_access_requirement_label(required_access)
        actual_label = image_access_diagnostic_name(access)
        raise ValueError(
            f"OpenGL buffer operation '{func_name}' requires {required_label} "
            f"SSBO access for {buffer_name}: got {actual_label}"
        )

    def glsl_buffer_block_access_metadata(self, node):
        choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, choices)
        if not choices:
            return None
        return choices[0][1]

    def record_glsl_buffer_block_variable(self, name, node, vtype=None):
        if not name:
            return
        block_type = str(
            self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        )
        self.glsl_buffer_block_variable_names.add(name)
        self.glsl_buffer_block_variable_types[name] = block_type
        access = self.glsl_buffer_block_access_metadata(node)
        if access is None or access == "read_write":
            return
        self.glsl_buffer_block_variable_accesses[name] = access

    def glsl_buffer_block_member_base_name(self, expr):
        if isinstance(expr, MemberAccessNode):
            return self.glsl_buffer_block_member_base_name(expr.object)
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            return self.glsl_buffer_block_member_base_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        return self.expression_name(expr)

    def is_glsl_buffer_block_member_expression(self, expr):
        if isinstance(expr, MemberAccessNode):
            return True
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            return self.is_glsl_buffer_block_member_expression(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        return False

    def is_glsl_buffer_block_member_reference(self, expr):
        if not self.is_glsl_buffer_block_member_expression(expr):
            return False
        base_name = self.glsl_buffer_block_member_base_name(expr)
        return bool(base_name and base_name in self.glsl_buffer_block_variable_names)

    def glsl_buffer_block_member_access(self, expr):
        if not self.is_glsl_buffer_block_member_expression(expr):
            return None
        base_name = self.glsl_buffer_block_member_base_name(expr)
        if not base_name:
            return None
        return self.glsl_buffer_block_variable_accesses.get(base_name)

    def validate_glsl_buffer_block_member_access(self, expr, required_access):
        access = self.glsl_buffer_block_member_access(expr)
        if image_access_satisfies_requirement(required_access, access):
            return
        member_name = expression_debug_name(expr)
        required_label = image_access_requirement_label(required_access)
        actual_label = image_access_diagnostic_name(access)
        raise ValueError(
            f"OpenGL buffer block member access for {member_name} requires "
            f"{required_label} buffer block access: got {actual_label}"
        )

    def validate_glsl_buffer_block_assignment_target(self, target, operator):
        if operator == "=":
            self.validate_glsl_buffer_block_member_access(target, "write")
            return
        self.validate_glsl_buffer_block_member_access(target, "read_write")

    def validate_glsl_buffer_block_atomic_call(self, func_name, args):
        if func_name not in self.GLSL_MEMORY_ATOMIC_FUNCTIONS or not args:
            return
        target = args[0]
        self.validate_glsl_buffer_block_member_access(target, "read_write")
        if not self.is_glsl_buffer_block_member_reference(target):
            return
        target_type = self.expression_result_type(target)
        if target_type is None:
            return
        mapped_type = self.map_type(target_type)
        if mapped_type not in {"int", "uint"}:
            target_name = expression_debug_name(target)
            raise ValueError(
                f"OpenGL buffer block atomic '{func_name}' requires a scalar "
                f"int or uint buffer block member for {target_name}: got "
                f"{mapped_type}"
            )
        self.validate_glsl_buffer_block_atomic_value_arguments(
            func_name,
            args,
            target,
            mapped_type,
        )

    def validate_glsl_buffer_block_atomic_value_arguments(
        self, func_name, args, target, target_type
    ):
        target_name = expression_debug_name(target)
        for value_arg, label in self.glsl_buffer_block_atomic_value_arguments(
            func_name, args
        ):
            value_type = self.glsl_buffer_block_atomic_argument_type(value_arg)
            if value_type is None or value_type == target_type:
                continue
            if self.glsl_unsigned_atomic_literal_is_compatible(
                value_arg, value_type, target_type
            ):
                continue
            raise ValueError(
                f"OpenGL buffer block atomic '{func_name}' requires {target_type} "
                f"{label} argument for {target_name}: "
                f"{expression_debug_name(value_arg)} has type {value_type}"
            )

    def glsl_buffer_block_atomic_value_arguments(self, func_name, args):
        if func_name == "atomicCompSwap":
            for index, label in ((1, "compare"), (2, "value")):
                if len(args) > index:
                    yield args[index], label
            return
        if len(args) > 1:
            yield args[1], "value"

    def glsl_buffer_block_atomic_argument_type(self, expr):
        scalar_kind = self.scalar_expression_kind(expr)
        if scalar_kind is not None:
            return scalar_kind
        result_type = self.expression_result_type(expr)
        if result_type is None:
            return None
        return self.map_type(result_type)

    def glsl_unsigned_atomic_literal_is_compatible(
        self, value_arg, value_type, target_type
    ):
        if target_type != "uint" or value_type != "int":
            return False
        literal_value = self.literal_int_value(value_arg, self.literal_int_constants)
        return literal_value is not None and literal_value >= 0

    def generate_glsl_memory_atomic_call(self, func_name, args):
        if func_name not in self.GLSL_MEMORY_ATOMIC_FUNCTIONS or not args:
            return None

        self.validate_glsl_buffer_block_atomic_call(func_name, args)

        target_type = self.expression_result_type(args[0])
        target_type = self.map_type(target_type) if target_type is not None else None
        rendered_args = [self.generate_expression(args[0])]
        value_arg_ids = {
            id(value_arg)
            for value_arg, _ in self.glsl_buffer_block_atomic_value_arguments(
                func_name, args
            )
        }

        for arg in args[1:]:
            if id(arg) in value_arg_ids and target_type in {"int", "uint"}:
                rendered_args.append(
                    self.generate_glsl_memory_atomic_value_argument(arg, target_type)
                )
            else:
                rendered_args.append(self.generate_expression(arg))

        return f"{func_name}({', '.join(rendered_args)})"

    def generate_glsl_memory_atomic_value_argument(self, arg, target_type):
        value_type = self.glsl_buffer_block_atomic_argument_type(arg)
        if value_type == target_type:
            return self.generate_expression_with_expected(arg, target_type)
        if self.glsl_unsigned_atomic_literal_is_compatible(
            arg, value_type, target_type
        ):
            literal_value = self.literal_int_value(arg, self.literal_int_constants)
            if literal_value is not None:
                return f"{literal_value}u"
        return self.generate_expression(arg)

    def generate_glsl_buffer_block_mutation_target(self, expr):
        self.glsl_buffer_block_read_validation_suppression += 1
        try:
            return self.generate_expression(expr)
        finally:
            self.glsl_buffer_block_read_validation_suppression -= 1

    def image_atomic_value_arguments(self, func_name, args):
        image_type = self.texture_argument_resource_type(args[0])
        has_sample = self.is_multisample_storage_image_type(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        raise ValueError(
            image_atomic_value_kind_error(
                "OpenGL",
                func_name,
                image_format,
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
        self, func_name, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        raise ValueError(
            image_atomic_result_kind_error(
                "OpenGL", func_name, image_format, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_format = self.image_resource_format(args[0])
        if image_format is None:
            return
        component_kind = image_atomic_explicit_format_component_kind(
            func_name, image_format
        )
        if component_kind is not None:
            self.validate_image_atomic_value_argument_types(
                func_name,
                args,
                component_kind,
                image_format,
            )
            self.validate_image_atomic_result_type(
                func_name,
                component_kind,
                image_format,
            )
            return
        raise ValueError(image_atomic_format_error("OpenGL", func_name, image_format))

    def storage_image_effective_format(self, image_type, explicit_format=None):
        if explicit_format is not None:
            return explicit_format
        image_type = self.resource_base_type(image_type)
        if not self.is_storage_image_type(image_type):
            return None
        return self.image_format_qualifier(image_type)

    def storage_image_parameter_contract(self, param):
        raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if not self.is_storage_image_type(raw_type):
            return None

        base_type = self.resource_base_type(raw_type)
        expected_type = self.resource_base_type(
            self.map_resource_type_with_format(base_type, param)
        )
        explicit_format = self.explicit_glsl_image_format(param)
        access_choices = self.resource_access_metadata_choices(param)
        self.validate_resource_access_metadata_consistency(param, access_choices)
        return {
            "type": expected_type,
            "format": explicit_format,
            "access": access_choices[0][1] if access_choices else None,
        }

    def storage_image_argument_contract(self, arg):
        actual_type = self.texture_argument_resource_type(arg)
        if not self.is_storage_image_type(actual_type):
            return None
        explicit_format = self.image_resource_format(arg)
        return {
            "type": self.resource_base_type(actual_type),
            "format": self.storage_image_effective_format(actual_type, explicit_format),
            "access": self.image_resource_access(arg),
        }

    def storage_image_binding_contract(self, binding):
        if binding is None:
            return None
        actual_type = binding.get("type")
        if not self.is_storage_image_type(actual_type):
            return None
        explicit_format = binding.get("format")
        return {
            "type": self.resource_base_type(actual_type),
            "format": self.storage_image_effective_format(actual_type, explicit_format),
            "access": binding.get("access"),
        }

    def validate_storage_image_parameter_contract(
        self,
        func_name,
        param,
        arg,
        index,
        actual,
    ):
        expected = self.storage_image_parameter_contract(param)
        if expected is None or actual is None:
            return

        param_name = getattr(param, "name", None) or f"arg{index}"
        actual_name = expression_debug_name(arg)
        expected_format = expected.get("format")
        if expected_format is not None and actual.get("format") != expected_format:
            actual_format = actual.get("format") or "<unspecified>"
            raise ValueError(
                f"OpenGL function call '{func_name}' requires "
                f"{expected_format} storage image format for argument "
                f"{actual_name} passed to parameter {param_name}: got "
                f"{actual_format}"
            )

        expected_type = expected.get("type")
        if expected_type is not None and actual.get("type") != expected_type:
            actual_type = actual.get("type") or "<unknown>"
            raise ValueError(
                f"OpenGL function call '{func_name}' requires "
                f"{expected_type} storage image for argument {actual_name} "
                f"passed to parameter {param_name}: got {actual_type}"
            )

        expected_access = expected.get("access")
        actual_access = actual.get("access")
        if expected_access is not None and not image_access_satisfies_requirement(
            expected_access,
            actual_access,
        ):
            required_label = image_access_requirement_label(expected_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise ValueError(
                f"OpenGL function call '{func_name}' requires "
                f"{required_label} storage image access for argument "
                f"{actual_name} passed to parameter {param_name}: got "
                f"{actual_label}"
            )

    def validate_function_image_parameter_contract_arguments(self, func_name, args):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        for index, param in enumerate(params):
            if index >= len(args):
                break
            self.validate_storage_image_parameter_contract(
                func_name,
                param,
                args[index],
                index,
                self.storage_image_argument_contract(args[index]),
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
                f"OpenGL function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got {actual_label}"
            )

    def validate_function_structured_buffer_access_arguments(self, func_name, args):
        callee_requirements = self.function_structured_buffer_access_requirements.get(
            func_name
        )
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.structured_buffer_resource_access(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise ValueError(
                f"OpenGL function call '{func_name}' requires {required_label} "
                f"SSBO access for argument {actual_name} passed to parameter "
                f"{param_name}: got {actual_label}"
            )

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = self.is_multisample_texture_resource_type(texture_type)
        is_storage_image = self.is_storage_image_type(texture_type)

        coordinate_dimension = None
        if texture_type and "Cube" in texture_type and is_storage_image:
            coordinate_dimension = 3
        elif texture_type and "Cube" not in texture_type:
            for prefix in ("iimage", "uimage", "image"):
                if texture_type.startswith(f"{prefix}2DMSArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2DMS"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}2DArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}3D"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2D"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1DArray"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1D"):
                    coordinate_dimension = 1
                    break
            if coordinate_dimension is None:
                coordinate_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 2,
                    "sampler2D": 2,
                    "samplerExternalOES": 2,
                    "__samplerExternal2DY2YEXT": 2,
                    "sampler2DArray": 3,
                    "sampler2DMS": 2,
                    "sampler2DMSArray": 3,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 2,
                    "isampler2D": 2,
                    "isampler2DArray": 3,
                    "isampler2DMS": 2,
                    "isampler2DMSArray": 3,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 2,
                    "usampler2D": 2,
                    "usampler2DArray": 3,
                    "usampler2DMS": 2,
                    "usampler2DMSArray": 3,
                    "usampler3D": 3,
                }.get(texture_type)

        offset_dimension = None
        if texture_type and "Cube" not in texture_type:
            offset_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 1,
                "sampler2D": 2,
                "sampler2DArray": 2,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 2,
                "sampler3D": 3,
                "isampler1D": 1,
                "isampler1DArray": 1,
                "isampler2D": 2,
                "isampler2DArray": 2,
                "isampler3D": 3,
                "usampler1D": 1,
                "usampler1DArray": 1,
                "usampler2D": 2,
                "usampler2DArray": 2,
                "usampler3D": 3,
            }.get(texture_type)

        gradient_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            if "Cube" in texture_type:
                gradient_dimension = 3
            else:
                gradient_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 1,
                    "sampler2D": 2,
                    "sampler2DArray": 2,
                    "sampler2DShadow": 2,
                    "sampler2DArrayShadow": 2,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 1,
                    "isampler2D": 2,
                    "isampler2DArray": 2,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 1,
                    "usampler2D": 2,
                    "usampler2DArray": 2,
                    "usampler3D": 3,
                }.get(texture_type)

        query_lod_coordinate_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            query_lod_coordinate_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 2,
                "sampler2D": 2,
                "sampler2DArray": 3,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 3,
                "sampler3D": 3,
                "samplerCube": 3,
                "samplerCubeArray": 4,
                "samplerCubeShadow": 3,
                "samplerCubeArrayShadow": 4,
                "isampler1D": 1,
                "isampler1DArray": 2,
                "isampler2D": 2,
                "isampler2DArray": 3,
                "isampler3D": 3,
                "isamplerCube": 3,
                "isamplerCubeArray": 4,
                "usampler1D": 1,
                "usampler1DArray": 2,
                "usampler2D": 2,
                "usampler2DArray": 3,
                "usampler3D": 3,
                "usamplerCube": 3,
                "usamplerCubeArray": 4,
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
        return descriptor[texture_resource_offset_dimension_key(func_name)]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

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
                "OpenGL",
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
                "OpenGL",
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
                        "OpenGL",
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
                    "OpenGL",
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
                    "OpenGL",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise ValueError(
            texture_query_lod_coordinate_dimension_error(
                "OpenGL",
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
                        "OpenGL",
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
                    "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
            self.is_multisample_storage_image_type(texture_type),
            "OpenGL",
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
                "OpenGL",
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
                "OpenGL",
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
            "OpenGL",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_size_expected_argument_count(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not texture_type:
            return None
        if self.is_storage_image_type(texture_type):
            return None
        if self.is_multisample_texture_resource_type(texture_type):
            return 1
        if texture_type.endswith(self.TEXTURE_SIZE_NO_LOD_SUFFIXES):
            return 1
        if texture_type.startswith(("sampler", "isampler", "usampler")):
            return 2
        return None

    def validate_texture_size_argument_count(self, func_name, args):
        if func_name != "textureSize" or not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_count = self.texture_size_expected_argument_count(texture_type)
        if expected_count is None or len(args) == expected_count:
            return
        texture_type = self.resource_base_type(texture_type)
        if len(args) < expected_count:
            raise ValueError(
                f"OpenGL texture operation 'textureSize' requires "
                f"{expected_count} argument(s) for {texture_type}, got {len(args)}"
            )
        raise ValueError(
            f"OpenGL texture operation 'textureSize' accepts "
            f"{expected_count} argument(s) for {texture_type}, got {len(args)}"
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("GLSL", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offset_args = [
                ArrayAccessNode(
                    extra_args[0],
                    LiteralNode(index, PrimitiveType("int")),
                )
                for index in range(4)
            ]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_component_value(self, component_arg):
        if component_arg is None:
            return None
        return self.literal_int_value(component_arg, self.literal_int_constants)

    def texture_gather_call_expression(
        self, function_name, texture_name, coord, offset_arg=None, component=None
    ):
        args = [texture_name, coord]
        if offset_arg is not None:
            args.append(offset_arg)
        if component is not None:
            args.append(str(component))
        return f"{function_name}({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, coord, offset_args, component
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            gather = self.texture_gather_call_expression(
                "textureGatherOffset",
                texture_name,
                coord,
                self.generate_expression(offset_arg),
                component,
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"vec4({', '.join(component_values)})"

    def texture_gather_compare_offsets_args(self, extra_args):
        if not extra_args:
            return None, None

        compare_arg = extra_args[0]
        offset_args, component_arg = self.texture_gather_offsets_args(extra_args[1:])
        if component_arg is not None:
            return None, None
        return compare_arg, offset_args

    def texture_gather_compare_offsets_expression(
        self, texture_name, coord, compare, offset_args
    ):
        component_suffixes = ("x", "y", "z", "w")
        component_values = []
        for index, offset_arg in enumerate(offset_args):
            offset = self.generate_expression(offset_arg)
            gather = (
                f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"
            )
            component_values.append(f"{gather}.{component_suffixes[index]}")
        return f"vec4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(self, build_expression, component):
        component_calls = [build_expression(index) for index in range(4)]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("GLSL", func_name, reason)

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        gather_types = {
            "sampler2D",
            "sampler2DArray",
            "samplerCube",
            "samplerCubeArray",
            "isampler2D",
            "isampler2DArray",
            "isamplerCube",
            "isamplerCubeArray",
            "usampler2D",
            "usampler2DArray",
            "usamplerCube",
            "usamplerCubeArray",
        }
        gather_offset_types = {
            "sampler2D",
            "sampler2DArray",
            "isampler2D",
            "isampler2DArray",
            "usampler2D",
            "usampler2DArray",
        }
        sample_offset_types = {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler3D",
            "sampler2DArray",
            "isampler1D",
            "isampler1DArray",
            "isampler2D",
            "isampler3D",
            "isampler2DArray",
            "usampler1D",
            "usampler1DArray",
            "usampler2D",
            "usampler3D",
            "usampler2DArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }
        compare_offset_types = {"sampler2DShadow", "sampler2DArrayShadow"}
        compare_lod_types = {"sampler2DShadow"}
        compare_grad_types = {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
        }
        return {
            "texture_type": texture_type,
            "gather": texture_type in gather_types,
            "gather_offset": texture_type in gather_offset_types,
            "sample_offset": texture_type in sample_offset_types,
            "compare_offset": texture_type in compare_offset_types,
            "compare_lod": texture_type in compare_lod_types,
            "compare_grad": texture_type in compare_grad_types,
            "compare_lod_offset": texture_type in compare_lod_types,
            "compare_grad_offset": texture_type in compare_offset_types,
            "gather_compare_offset": texture_type in compare_offset_types,
        }

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_gather_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_sample_vector_zero_value(self, texture_type=None):
        expected_type = self.map_type(self.current_expression_expected_type)
        if expected_type in {"ivec4", "uvec4", "vec4"}:
            return self.zero_value_expression(expected_type)

        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("isampler"):
            return "ivec4(0)"
        if texture_type.startswith("usampler"):
            return "uvec4(0u)"
        return "vec4(0.0)"

    def texture_sampling_result_type(self, texture_arg, func_name=None):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        if texture_type.startswith("isampler"):
            return "ivec4"
        if texture_type.startswith("usampler"):
            return "uvec4"
        if "Shadow" in texture_type and func_name not in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return "float"
        return "vec4"

    def unsupported_texture_projected_call(self, func_name, reason, texture_type=None):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL projected texture: {func_name} {reason} */ "
            f"{zero_value}"
        )

    def projected_cube_texture_coordinate(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type not in {"samplerCube", "isamplerCube", "usamplerCube"}:
            return None

        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        if coord_type not in {"vec4", "float4"}:
            return None
        return (
            f"{self.vector_component(coord, 'xyz')} / "
            f"{self.vector_component(coord, 'w')}"
        )

    def generate_projected_cube_texture_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        texture_type = self.texture_resource_type(args[0])
        projected_coord = self.projected_cube_texture_coordinate(
            texture_type, args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name,
                "requires 1D, 2D, 2D-array, 3D, or cube projection coordinates",
                texture_type,
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            mapped_args = [texture_name, projected_coord]
            if extra_args:
                mapped_args.append(self.generate_expression(extra_args[0]))
            return f"texture({', '.join(mapped_args)})"

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            lod = self.generate_expression(extra_args[0])
            return f"textureLod({texture_name}, {projected_coord}, {lod})"

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"textureGrad({texture_name}, {projected_coord}, {ddx}, {ddy})"

        if (
            is_projected_texture_basic_offset_operation(func_name)
            or is_projected_texture_lod_offset_operation(func_name)
            or is_projected_texture_grad_offset_operation(func_name)
        ):
            return self.unsupported_texture_projected_call(
                func_name, texture_sample_offset_capability_error("GLSL"), texture_type
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error(), texture_type
        )

    def texture_sample_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        zero_value = self.texture_sample_vector_zero_value()
        return (
            f"/* unsupported GLSL texture offset: {func_name} {reason} */ "
            f"{zero_value}"
        )

    def is_multisample_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler2DMS",
            "isampler2DMSArray",
            "usampler2DMS",
            "usampler2DMSArray",
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def is_multisample_storage_image_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL multisample texture call: "
            f"{func_name} on {texture_type} */ {zero_value}"
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_compare_scalar_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_query_lod_expression(
            "GLSL", texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_levels_expression("GLSL", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_lod_expression("GLSL", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_type(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "GLSL", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "GLSL", func_name, texture_type
            )

        return None

    def unsupported_texture_samples_query_call(self):
        return unsupported_texture_samples_query_call_expression("GLSL")

    def image_size_expression(self, image_arg):
        image_name = self.generate_expression(image_arg)
        return f"imageSize({image_name})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not texture_type:
            return None
        return {
            "function_name": (
                "imageSize"
                if self.is_storage_image_type(texture_type)
                else "textureSize"
            ),
        }

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_type(texture_type),
            "multisample": self.is_multisample_texture_resource_type(texture_type),
            "size_descriptor": self.texture_query_size_descriptor(texture_type),
        }

    def texture_query_levels_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return texture_query_levels_multisample_expression()
        return None

    def texture_samples_expression(self, func_name, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if descriptor["texture_type"] and not descriptor["multisample"]:
            return self.unsupported_texture_samples_query_call()
        if (
            not is_texture_samples_query_operation(func_name)
            and descriptor["multisample"]
        ):
            texture_name = self.generate_expression(texture_arg)
            return texture_samples_query_expression("GLSL", texture_name)
        return None

    def texture_size_query_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        size_descriptor = descriptor["size_descriptor"]
        if size_descriptor and size_descriptor["function_name"] == "imageSize":
            return self.image_size_expression(texture_arg)
        return None

    def unsupported_multisample_texel_fetch_offset_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL texel fetch offset: multisample texture "
            f"{texture_type} does not support offsets */ {zero_value}"
        )

    def is_cube_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "samplerCube",
            "samplerCubeArray",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "isamplerCube",
            "isamplerCubeArray",
            "usamplerCube",
            "usamplerCubeArray",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL texel fetch: {func_name} on {texture_type} */ "
            f"{zero_value}"
        )

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
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

        if offset_args and not self.glsl_texel_offsets_are_compile_time_constants(
            offset_args
        ):
            return self.unsupported_dynamic_texel_offset_gather_call(
                func_name, texture_type
            )

        component = self.texture_gather_component_value(component_arg)
        if component is not None:
            if component not in {0, 1, 2, 3}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_literal_error()
                )
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, component
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, component
            )

        if component_arg is None:
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, None
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg
            )

        component_expr = self.generate_expression(component_arg)
        if is_texture_gather_multi_offset_operation(func_name):
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, option
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        function_name = (
            "textureGatherOffset"
            if is_texture_gather_single_offset_operation(func_name)
            else "textureGather"
        )
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, option
            ),
            component_expr,
        )

    def texture_compare_coordinate(self, texture_type, coord, compare):
        texture_type = self.resource_base_type(texture_type)
        if texture_type == "samplerCubeArrayShadow":
            return None
        constructor = (
            "vec4"
            if texture_type
            in {
                "sampler2DArrayShadow",
                "samplerCubeShadow",
            }
            else "vec3"
        )
        return f"{constructor}({coord}, {compare})"

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def texture_compare_lod_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod"]

    def texture_compare_grad_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad"]

    def texture_compare_lod_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod_offset"]

    def texture_compare_grad_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad_offset"]

    def texture_compare_projected_coordinate(
        self, texture_type, coord_arg, coord, compare
    ):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "sampler2DShadow":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
            return f"vec3({projected_coord}, {compare})"

        if texture_type == "samplerCubeShadow":
            if coord_type not in {"vec4", "float4"}:
                return None
            projected_coord = (
                f"{self.vector_component(coord, 'xyz')} / "
                f"{self.vector_component(coord, 'w')}"
            )
            return f"vec4({projected_coord}, {compare})"

        if texture_type != "sampler2DArrayShadow" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        projected_coord = (
            f"{self.vector_component(coord, 'xy')} / "
            f"{self.vector_component(coord, 'w')}"
        )
        layer = self.vector_component(coord, "z")
        return f"vec4({projected_coord}, {layer}, {compare})"

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression("GLSL", func_name, reason)

    def glsl_integer_vector_type_names(self):
        return {
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
        }

    def glsl_compile_time_int_constants(self):
        if self.current_compile_time_int_constants is not None:
            return self.current_compile_time_int_constants
        return self.literal_int_constants

    def glsl_compile_time_int_vector_constants(self):
        if self.current_compile_time_int_vector_constants is not None:
            return self.current_compile_time_int_vector_constants
        return self.literal_int_vector_constants

    def glsl_compile_time_int_vector_array_constants(self):
        if self.current_compile_time_int_vector_array_constants is not None:
            return self.current_compile_time_int_vector_array_constants
        return self.literal_int_vector_array_constants

    def glsl_integer_vector_literal_value(
        self,
        expr,
        int_constants=None,
        vector_constants=None,
        vector_array_constants=None,
    ):
        int_constants = (
            self.glsl_compile_time_int_constants()
            if int_constants is None
            else int_constants
        )
        if isinstance(expr, ArrayAccessNode):
            vector_array_constants = (
                self.glsl_compile_time_int_vector_array_constants()
                if vector_array_constants is None
                else vector_array_constants
            )
            array_name = self.expression_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
            values = vector_array_constants.get(array_name)
            index = self.literal_int_value(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                int_constants,
            )
            if (
                values is not None
                and isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(values)
            ):
                return values[index]
            return None

        vector_constants = (
            self.glsl_compile_time_int_vector_constants()
            if vector_constants is None
            else vector_constants
        )
        expr_name = self.expression_name(expr)
        if expr_name in vector_constants:
            return vector_constants[expr_name]

        class_name = expr.__class__.__name__ if expr is not None else ""
        if "FunctionCall" in class_name:
            constructor = getattr(expr, "function", getattr(expr, "name", None))
            arguments = getattr(expr, "arguments", getattr(expr, "args", []))
        elif "Constructor" in class_name:
            constructor = getattr(expr, "constructor_type", None)
            arguments = getattr(expr, "arguments", [])
        else:
            return None

        constructor_name = self.type_name_string(constructor)
        if constructor_name not in self.glsl_integer_vector_type_names():
            return None

        arguments = list(arguments or [])
        if not arguments:
            return None

        values = []
        for argument in arguments:
            literal_argument = self.literal_int_value(argument, int_constants)
            if not isinstance(literal_argument, int) or isinstance(
                literal_argument, bool
            ):
                return None
            values.append(literal_argument)

        vector_width = int(constructor_name[-1])
        if len(values) == 1:
            return tuple(values * vector_width)
        if len(values) == vector_width:
            return tuple(values)
        return None

    def collect_literal_int_vector_constants_from_variables(
        self, variables, int_constants
    ):
        vector_constants = {}
        for variable in variables or []:
            name = getattr(variable, "name", None)
            if not name or "const" not in getattr(variable, "qualifiers", []):
                continue
            var_type = self.type_name_string(
                getattr(variable, "var_type", getattr(variable, "vtype", None))
            )
            if var_type not in self.glsl_integer_vector_type_names():
                continue
            value = self.glsl_integer_vector_literal_value(
                getattr(variable, "initial_value", None),
                int_constants,
                vector_constants,
            )
            if value is not None:
                vector_constants[name] = value
        return vector_constants

    def glsl_integer_vector_array_literal_value(
        self, expr, int_constants=None, vector_constants=None
    ):
        if not isinstance(expr, ArrayLiteralNode):
            return None

        int_constants = (
            self.glsl_compile_time_int_constants()
            if int_constants is None
            else int_constants
        )
        vector_constants = (
            self.glsl_compile_time_int_vector_constants()
            if vector_constants is None
            else vector_constants
        )
        values = []
        for element in expr.elements:
            value = self.glsl_integer_vector_literal_value(
                element,
                int_constants,
                vector_constants,
            )
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    def collect_literal_int_vector_array_constants_from_variables(
        self, variables, int_constants, vector_constants
    ):
        vector_array_constants = {}
        for variable in variables or []:
            name = getattr(variable, "name", None)
            if not name or "const" not in getattr(variable, "qualifiers", []):
                continue

            var_type = getattr(variable, "var_type", getattr(variable, "vtype", None))
            if "ArrayType" not in var_type.__class__.__name__:
                continue
            element_type = self.type_name_string(
                getattr(var_type, "element_type", None)
            )
            if element_type not in self.glsl_integer_vector_type_names():
                continue

            value = self.glsl_integer_vector_array_literal_value(
                getattr(variable, "initial_value", None),
                int_constants,
                vector_constants,
            )
            if value is None:
                continue

            size = self.literal_int_value(
                getattr(var_type, "size", None), int_constants
            )
            if size is not None and len(value) != size:
                continue
            vector_array_constants[name] = value
        return vector_array_constants

    def update_compile_time_offset_constants(self, variable, var_type):
        name = getattr(variable, "name", None)
        if not name:
            return
        self.current_compile_time_int_constants.pop(name, None)
        self.current_compile_time_int_vector_constants.pop(name, None)
        self.current_compile_time_int_vector_array_constants.pop(name, None)
        if "const" not in getattr(variable, "qualifiers", []):
            return

        initial_value = getattr(variable, "initial_value", None)
        raw_var_type = var_type
        var_type = self.type_name_string(var_type)
        scalar_integer_types = {"int", "uint"}
        if var_type in scalar_integer_types:
            value = self.literal_int_value(
                initial_value, self.current_compile_time_int_constants
            )
            if isinstance(value, int) and not isinstance(value, bool):
                self.current_compile_time_int_constants[name] = value
            return

        if var_type not in self.glsl_integer_vector_type_names():
            if "[" not in str(var_type):
                return
            base_type, _ = split_array_type_suffix(var_type)
            if base_type not in self.glsl_integer_vector_type_names():
                return
            value = self.glsl_integer_vector_array_literal_value(
                initial_value,
                self.current_compile_time_int_constants,
                self.current_compile_time_int_vector_constants,
            )
            if value is None:
                return
            size = self.literal_int_value(
                getattr(raw_var_type, "size", None),
                self.current_compile_time_int_constants,
            )
            if size is not None and len(value) != size:
                return
            self.current_compile_time_int_vector_array_constants[name] = value
            return

        value = self.glsl_integer_vector_literal_value(
            initial_value,
            self.current_compile_time_int_constants,
            self.current_compile_time_int_vector_constants,
        )
        if value is not None:
            self.current_compile_time_int_vector_constants[name] = value

    def glsl_texel_offset_is_compile_time_constant(self, offset_arg):
        offset_type = self.type_name_string(self.expression_result_type(offset_arg))
        scalar_integer_types = {"int", "uint"}
        vector_integer_types = self.glsl_integer_vector_type_names()
        int_constants = self.glsl_compile_time_int_constants()

        literal_value = self.literal_int_value(offset_arg, int_constants)
        if isinstance(literal_value, int) and not isinstance(literal_value, bool):
            return offset_type is None or offset_type in scalar_integer_types

        if offset_type is not None and offset_type not in vector_integer_types:
            return False
        return (
            self.glsl_integer_vector_literal_value(offset_arg, int_constants)
            is not None
        )

    def glsl_texel_offsets_are_compile_time_constants(self, offset_args):
        return all(
            self.glsl_texel_offset_is_compile_time_constant(offset_arg)
            for offset_arg in offset_args
        )

    def glsl_texture_call_offsets_are_compile_time_constants(self, func_name, args):
        if self.glsl_texture_offset_non_const_enabled():
            return True

        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        return self.glsl_texel_offsets_are_compile_time_constants(
            args[offset_index] for offset_index in offset_indices
        )

    def dynamic_texel_offset_reason(self):
        return "texel offsets must be compile-time integer constants"

    def unsupported_dynamic_texel_offset_compare_call(self, func_name):
        return self.unsupported_texture_compare_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_offset_sample_call(self, func_name):
        return self.unsupported_texture_sample_offset_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_offset_projected_call(self, func_name, texture_type):
        return self.unsupported_texture_projected_call(
            func_name, self.dynamic_texel_offset_reason(), texture_type
        )

    def unsupported_dynamic_texel_offset_gather_call(self, func_name, texture_type):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL texture gather: {func_name} "
            f"{self.dynamic_texel_offset_reason()} */ {zero_value}"
        )

    def unsupported_dynamic_texel_offset_gather_compare_call(self, func_name):
        return self.unsupported_texture_gather_compare_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_fetch_offset_call(self, func_name, texture_type):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return (
            f"/* unsupported GLSL texel fetch offset: {func_name} "
            f"{self.dynamic_texel_offset_reason()} */ {zero_value}"
        )

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
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
            compare_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("GLSL")
                )

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return f"texture({texture_name}, {compare_coord})"

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("GLSL")
                    )
                offset_arg = extra_args[1]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return f"textureOffset({texture_name}, {compare_coord}, {offset})"

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD requires 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                return f"textureLod({texture_name}, {compare_coord}, {lod})"

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD offsets require 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                offset_arg = extra_args[2]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return (
                    f"textureLodOffset({texture_name}, {compare_coord}, "
                    f"{lod}, {offset})"
                )

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradient offsets require 2D or 2D-array shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                offset_arg = extra_args[3]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return (
                    f"textureGradOffset({texture_name}, {compare_coord}, "
                    f"{ddx}, {ddy}, {offset})"
                )

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        if is_texture_compare_basic_operation(func_name):
            if texture_type == "samplerCubeArrayShadow":
                return f"texture({texture_name}, {coord}, {compare})"
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            return f"texture({texture_name}, {compare_coord})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("GLSL")
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            offset_arg = extra_args[1]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return f"textureOffset({texture_name}, {compare_coord}, {offset})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD requires 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            return f"textureLod({texture_name}, {compare_coord}, {lod})"

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD offsets require 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            offset_arg = extra_args[2]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return f"textureLodOffset({texture_name}, {compare_coord}, {lod}, {offset})"

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradient offsets require 2D or 2D-array shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset_arg = extra_args[3]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return (
                f"textureGradOffset({texture_name}, {compare_coord}, "
                f"{ddx}, {ddy}, {offset})"
            )

        return None

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "GLSL", func_name, reason
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def validate_texture_gather_compare_offsets_argument(
        self, func_name, texture_type, offset_arg
    ):
        offset_type = self.expression_result_type(offset_arg)
        if offset_type is None:
            return
        if not self.is_integer_coordinate_type(offset_type):
            raise ValueError(
                operation_argument_type_error(
                    "OpenGL",
                    "resource",
                    func_name,
                    "an integer",
                    "offset",
                    expression_debug_name(offset_arg),
                    self.type_name_string(offset_type),
                )
            )
        expected_dimension = self.resource_offset_dimension(
            "textureGatherCompareOffset", texture_type
        )
        offset_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(offset_type), self.map_type
        )
        if offset_dimension is None or offset_dimension == expected_dimension:
            return
        raise ValueError(
            operation_dimension_argument_error(
                "OpenGL",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "offset",
                self.resource_base_type(texture_type),
                expression_debug_name(offset_arg),
                self.type_name_string(offset_type),
            )
        )

    def generate_texture_gather_compare_offsets_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare_arg, offset_args = self.texture_gather_compare_offsets_args(extra_args)
        if offset_args is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires compare and typed offsets array or four offsets"
            )
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("GLSL")
            )

        compare_type = self.expression_result_type(compare_arg)
        if compare_type is not None and not self.is_scalar_floating_type(compare_type):
            raise ValueError(
                operation_argument_type_error(
                    "OpenGL",
                    "texture compare",
                    func_name,
                    "a scalar floating",
                    "compare",
                    expression_debug_name(compare_arg),
                    self.type_name_string(compare_type),
                )
            )
        for offset_arg in offset_args:
            self.validate_texture_gather_compare_offsets_argument(
                func_name, texture_type, offset_arg
            )
        if not self.glsl_texel_offsets_are_compile_time_constants(offset_args):
            return self.unsupported_dynamic_texel_offset_gather_compare_call(func_name)

        compare = self.generate_expression(compare_arg)
        return self.texture_gather_compare_offsets_expression(
            texture_name, coord, compare, offset_args
        )

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
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
            return f"textureGather({texture_name}, {coord}, {compare})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("GLSL")
            )
        offset_arg = extra_args[1]
        if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
            return self.unsupported_dynamic_texel_offset_gather_compare_call(func_name)
        offset = self.generate_expression(offset_arg)
        return f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"

    def image_resource_format(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def image_resource_access(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_access_parameters,
            self.image_variable_accesses,
        )

    def image_atomic_parameter_requires_diagnostic(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name is None:
            return False
        if texture_name in self.current_resource_aliases:
            return False
        return (
            texture_name in self.current_texture_parameters
            and texture_name not in self.texture_variable_types
        )

    def structured_buffer_resource_access(self, buffer_arg):
        buffer_name = self.expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.current_structured_buffer_access_parameters.get(
            buffer_name,
            self.structured_buffer_variable_accesses.get(buffer_name),
        )

    def image_atomic_zero_value(self, texture_type, image_format):
        component_kind = image_format_component_kind(image_format)
        if component_kind is None:
            texture_type = self.resource_base_type(texture_type)
            if texture_type and texture_type.startswith("uimage"):
                component_kind = "uint"
            elif texture_type and texture_type.startswith("image"):
                component_kind = "float"
            else:
                component_kind = "int"
        return storage_image_atomic_zero_value(component_kind)

    def image_atomic_parameter_fallback_call(self, func_name, args):
        texture_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        zero_value = self.image_atomic_zero_value(texture_type, image_format)
        texture_type = self.resource_base_type(texture_type) or expression_debug_name(
            args[0]
        )
        return image_atomic_diagnostic_expression(
            "GLSL",
            func_name,
            f"{texture_type} parameter",
            zero_value,
        )

    def generate_image_atomic_call(self, func_name, args):
        if self.image_atomic_parameter_requires_diagnostic(args[0]):
            return self.image_atomic_parameter_fallback_call(func_name, args)
        return "{}({})".format(
            func_name,
            ", ".join(self.generate_expression(arg) for arg in args),
        )

    def is_integer_image_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_glsl_integer_image_type(texture_type) or texture_type in {
            "iimageCube",
            "iimageCubeArray",
            "uimageCube",
            "uimageCubeArray",
        }

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, texture_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(texture_type)

    def is_float_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_glsl_float_image_resource(texture_type) or texture_type in {
            "imageCube",
            "imageCubeArray",
        }

    def image_store_constructors_by_kind(self):
        return storage_image_store_constructors("vec4", "ivec4", "uvec4")

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def image_load_component_suffix(self, texture_type, image_format):
        return storage_image_load_component_suffix(
            image_format,
            expected_scalar=self.is_scalar_value_type(
                self.current_expression_expected_type
            ),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                texture_type, image_format
            )
            and self.expected_component_count() != 4,
            float_resource=self.is_float_image_resource(texture_type),
        )

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, texture_type):
        if texture_type in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
        }:
            return "ivec4"
        if texture_type in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }:
            return "uvec4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        return storage_image_two_component_store_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            self.image_store_constructors_by_kind(),
            self.image_store_zero_values_by_kind(),
        )

    def image_store_value_expression(
        self, texture_type, image_format, value, value_type=None
    ):
        if self.allows_native_glsl_image_four_component_value(
            self.image_store_channel_count(texture_type, image_format),
            self.value_component_count(value_type),
        ):
            return value
        return storage_image_store_value_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            scalar_integer_resource=self.is_scalar_integer_image_resource(
                texture_type, image_format
            ),
            float_resource=self.is_float_image_resource(texture_type),
            integer_constructor=self.integer_image_store_constructor(texture_type),
            float_constructor="vec4",
            constructors_by_kind=self.image_store_constructors_by_kind(),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

    def glsl_storage_image_component_kind(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("uimage"):
            return "uint"
        if texture_type.startswith("iimage"):
            return "int"
        if texture_type.startswith("image"):
            return "float"
        return None

    def image_store_expected_value_type(
        self, texture_type, image_format, value_arg=None
    ):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_kind = self.glsl_storage_image_component_kind(texture_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_load_result_type(self, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            return image_format_result_type(
                image_format,
                vector_prefixes={"float": "vec", "int": "ivec", "uint": "uvec"},
            )
        image_type = self.texture_argument_resource_type(image_arg)
        component_kind = self.glsl_storage_image_component_kind(image_type)
        if component_kind == "float":
            return "vec4"
        if component_kind == "int":
            return "ivec4"
        if component_kind == "uint":
            return "uvec4"
        return None

    def image_load_component_kind(self, texture_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        return self.glsl_storage_image_component_kind(texture_type)

    def image_load_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
            ),
        )

    def allows_native_glsl_image_four_component_value(
        self, image_channels, value_channels
    ):
        """Return true when scalar image formats use GLSL's native gvec4 value."""
        return value_channels == 4 and image_channels == 1

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
            excluded_type_markers=("mat",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("mat",),
        )

    def image_store_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
            ),
        )

    def validate_image_store_value_shape(self, texture_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(texture_type, image_format)
        value_channels = self.expression_component_count(value_arg)
        if self.allows_native_glsl_image_four_component_value(
            expected_channels,
            value_channels,
        ):
            return
        value_channels = image_store_value_shape_mismatch(
            expected_channels,
            value_channels,
        )
        if value_channels is None:
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise ValueError(
            image_store_value_shape_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, texture_type, image_format, value_arg):
        value_type = self.expression_result_type(value_arg)
        if self.is_matrix_value_type(value_type):
            format_label = image_format or self.resource_base_type(texture_type)
            mapped_type = self.map_type(self.type_name_string(value_type))
            raise ValueError(
                "OpenGL image store operation 'imageStore' requires scalar or "
                f"vector value for {format_label} images: "
                f"{expression_debug_name(value_arg)} has type {mapped_type}"
            )
        self.validate_image_store_value_shape(texture_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(texture_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise ValueError(
            image_store_value_kind_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, texture_type, image_format):
        if self.is_matrix_value_type(self.current_expression_expected_type):
            format_label = image_format or self.resource_base_type(texture_type)
            expected_type = self.map_type(self.current_expression_expected_type)
            raise ValueError(
                "OpenGL image load operation 'imageLoad' requires scalar or "
                f"vector result context for {format_label} images: "
                f"got {expected_type}"
            )
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(texture_type, image_format)
        format_label = image_format or self.resource_base_type(texture_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise ValueError(
                image_load_result_kind_error(
                    "OpenGL", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(texture_type, image_format)
        if self.allows_native_glsl_image_four_component_value(
            loaded_channels,
            expected_channels,
        ):
            return
        expected_channels = image_load_result_shape_mismatch(
            loaded_channels,
            expected_channels,
        )
        if expected_channels is None:
            return
        raise ValueError(
            image_load_result_shape_error(
                "OpenGL", format_label, loaded_channels, expected_channels
            )
        )

    def generate_texture_call(self, func_name, args):
        if not func_name:
            return None

        self.validate_texture_call_arity(func_name, args)
        self.validate_image_resource_argument(func_name, args)
        self.validate_image_access_argument(func_name, args)
        self.validate_texture_resource_argument(func_name, args)
        self.validate_texture_size_argument_count(func_name, args)
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

        if is_texture_query_levels_operation(func_name) and args:
            return self.texture_query_levels_expression(args[0])

        if is_resource_samples_query_operation(func_name) and args:
            return self.texture_samples_expression(func_name, args[0])

        if is_resource_size_query_operation(func_name) and args:
            return self.texture_size_query_expression(args[0])

        if len(args) < 2:
            return None

        if func_name == "imageLoad" and len(args) >= 2:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 3 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                load_expr = f"imageLoad({image_name}, {coord}, {sample})"
            else:
                load_expr = f"imageLoad({image_name}, {coord})"
            image_format = self.image_resource_format(args[0])
            self.validate_image_load_result_type(texture_type, image_format)
            return f"{load_expr}{self.image_load_component_suffix(texture_type, image_format)}"

        if func_name == "imageStore" and len(args) >= 3:
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            texture_type = self.texture_resource_type(args[0])
            if len(args) >= 4 and self.is_multisample_storage_image_type(texture_type):
                sample = self.generate_expression(args[2])
                value_arg = args[3]
            else:
                sample = None
                value_arg = args[2]
            image_format = self.image_resource_format(args[0])
            self.validate_image_store_value_type(texture_type, image_format, value_arg)
            value = self.generate_expression_with_expected(
                value_arg,
                self.image_store_expected_value_type(
                    texture_type, image_format, value_arg
                ),
            )
            value = self.image_store_value_expression(
                texture_type,
                image_format,
                value,
                self.expression_result_type(value_arg),
            )
            if sample is not None:
                return f"imageStore({image_name}, {coord}, {sample}, {value})"
            return f"imageStore({image_name}, {coord}, {value})"

        if is_image_atomic_operation(func_name) and args:
            return self.generate_image_atomic_call(func_name, args)

        texture_type = self.texture_resource_type(args[0])
        storage_image_operation = self.storage_image_texture_operation_expression(
            func_name, texture_type
        )
        if storage_image_operation is not None:
            return storage_image_operation

        if is_texture_compare_operation(func_name):
            return self.generate_texture_compare_call(func_name, args)

        if func_name == "textureGatherCompareOffsets":
            return self.generate_texture_gather_compare_offsets_call(func_name, args)

        if is_texture_gather_compare_operation(func_name):
            return self.generate_texture_gather_compare_call(func_name, args)

        if is_texture_gather_operation(func_name):
            return self.generate_texture_gather_call(func_name, args)

        if is_texture_query_lod_operation(func_name):
            parts = self.texture_call_parts(args)
            if parts is None:
                return None
            texture_name, coord, _ = parts
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_query_lod_call(texture_type)
            if self.is_storage_image_type(texture_type):
                return self.unsupported_texture_query_lod_call(texture_type)
            coord = self.texture_query_lod_coordinate(texture_type, coord)
            return f"textureQueryLod({texture_name}, {coord})"

        if is_texel_fetch_offset_operation(func_name) and len(args) >= 4:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texel_fetch_offset_call(
                    texture_type
                )
            if not self.glsl_texture_call_offsets_are_compile_time_constants(
                func_name, args
            ):
                return self.unsupported_dynamic_texel_fetch_offset_call(
                    func_name, texture_type
                )
            return None

        if is_texel_fetch_basic_operation(func_name) and len(args) >= 3:
            texture_type = self.texture_resource_type(args[0])
            if self.is_cube_texture_resource_type(texture_type):
                return self.unsupported_cube_texel_fetch_call(func_name, texture_type)
            return None

        if is_projected_texture_operation(func_name) and args:
            texture_type = self.resource_base_type(self.texture_resource_type(args[0]))
            if texture_type in {"samplerCube", "isamplerCube", "usamplerCube"}:
                return self.generate_projected_cube_texture_call(func_name, args)
            if texture_type in {
                "samplerCubeArray",
                "isamplerCubeArray",
                "usamplerCubeArray",
            }:
                return self.unsupported_texture_projected_call(
                    func_name,
                    "requires 1D, 2D, 2D-array, or 3D projection coordinates",
                    texture_type,
                )

        if is_texture_sampling_operation(func_name):
            texture_type = self.texture_resource_type(args[0])
            if self.is_multisample_texture_resource_type(texture_type):
                return self.unsupported_multisample_texture_call(
                    func_name, texture_type
                )
            if is_texture_sample_offset_operation(func_name):
                if not self.texture_sample_offset_supported(texture_type):
                    return self.unsupported_texture_sample_offset_call(
                        func_name, texture_sample_offset_capability_error("GLSL")
                    )
                if not self.glsl_texture_call_offsets_are_compile_time_constants(
                    func_name, args
                ):
                    return self.unsupported_dynamic_texel_offset_sample_call(func_name)
            if (
                is_projected_texture_basic_offset_operation(func_name)
                or is_projected_texture_lod_offset_operation(func_name)
                or is_projected_texture_grad_offset_operation(func_name)
            ):
                if not self.texture_sample_offset_supported(texture_type):
                    return self.unsupported_texture_projected_call(
                        func_name,
                        texture_sample_offset_capability_error("GLSL"),
                        texture_type,
                    )
                if not self.glsl_texture_call_offsets_are_compile_time_constants(
                    func_name, args
                ):
                    return self.unsupported_dynamic_texel_offset_projected_call(
                        func_name, texture_type
                    )

        if not is_texture_sampling_operation(
            func_name
        ) or not self.is_explicit_sampler_argument(args):
            return None

        parts = self.texture_call_parts(args)
        if parts is None:
            return None
        texture_name, coord, extra_args = parts
        sampler_name = self.generate_expression(args[1])
        declared_texture_type = self.texture_declared_resource_type(args[0])
        if self.is_separate_texture_object_type(declared_texture_type):
            sampled_type = self.sampled_image_type(declared_texture_type)
            sampled_type = self.resource_base_type(sampled_type)
            texture_name = f"{sampled_type}({texture_name}, {sampler_name})"
        mapped_args = [texture_name, coord] + [
            self.generate_expression(arg) for arg in extra_args
        ]
        return f"{func_name}({', '.join(mapped_args)})"

    def collect_function_sampler_parameter_indices(self, root):
        sampler_indices = {}
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

            name = getattr(value, "name", None)
            params = getattr(value, "parameters", getattr(value, "params", []))
            if name and params:
                indices = []
                for index, param in enumerate(params):
                    param_type = getattr(
                        param, "param_type", getattr(param, "vtype", None)
                    )
                    if self.is_sampler_type(param_type):
                        indices.append(index)
                if indices:
                    sampler_indices[name] = set(indices)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_indices

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            base_type = self.resource_base_type(var_type)
            mapped_type = self.map_resource_type_with_format(base_type, node)
            if var_name and (
                self.is_sampler_type(base_type)
                or self.is_opaque_resource_type(mapped_type)
            ):
                resource_names.add(var_name)
        return resource_names

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_inferable_resource_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Non-resource local declaration(s) shadow OpenGL global resource(s): "
                f"{names}"
            )

    def filter_sampler_arguments(self, func_name, args):
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        if not skipped_indices:
            return args
        return [arg for index, arg in enumerate(args) if index not in skipped_indices]

    def should_preserve_sampler_resource(self, name):
        return name in getattr(self, "preserved_sampler_variables", set())

    def should_preserve_sampler_parameter(self, func_name, param_name):
        if not func_name:
            return False
        return param_name in getattr(self, "preserved_sampler_parameters", {}).get(
            func_name, set()
        )

    def expression_uses_nonuniform_ext(self, expr):
        for node in self.walk_ast(expr):
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node) == "nonuniformEXT"
            ):
                return True
        return False

    def collect_preserved_nonuniform_sampler_resources(self, ast, functions):
        global_samplers = set()
        for node in getattr(ast, "global_variables", []) or []:
            name = self.resource_node_name(node)
            if name and self.is_sampler_type(self.resource_node_type(node)):
                global_samplers.add(name)

        sampler_parameters = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            sampler_parameters[func_name] = {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }

        preserved_globals = set()
        preserved_params = {name: set() for name in sampler_parameters}

        changed = True
        while changed:
            changed = False
            for func in functions or []:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    index_expr = getattr(
                        node, "index", getattr(node, "index_expr", None)
                    )
                    if isinstance(
                        node, ArrayAccessNode
                    ) and self.expression_uses_nonuniform_ext(index_expr):
                        array_name = self.expression_name(
                            getattr(node, "array", getattr(node, "array_expr", None))
                        )
                        if array_name in sampler_parameters.get(func_name, set()):
                            if array_name not in preserved_params[func_name]:
                                preserved_params[func_name].add(array_name)
                                changed = True
                        elif array_name in global_samplers:
                            if array_name not in preserved_globals:
                                preserved_globals.add(array_name)
                                changed = True

                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    callee_preserved = preserved_params.get(callee_name)
                    if not callee_preserved:
                        continue
                    callee_params = self.function_parameter_names.get(callee_name, [])
                    args = list(getattr(node, "arguments", getattr(node, "args", [])))
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            break
                        if callee_params[index] not in callee_preserved:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name in sampler_parameters.get(func_name, set()):
                            if arg_name not in preserved_params[func_name]:
                                preserved_params[func_name].add(arg_name)
                                changed = True
                        elif arg_name in global_samplers:
                            if arg_name not in preserved_globals:
                                preserved_globals.add(arg_name)
                                changed = True

        self.preserved_sampler_variables = preserved_globals
        self.preserved_sampler_parameters = {
            func_name: names for func_name, names in preserved_params.items() if names
        }

    def collect_resource_array_size_hints(self, ast):
        global_hints, function_hints = collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_sampled_texture_globals(ast),
            function_arrays=self.collect_unsized_sampled_texture_parameters(ast),
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
        global_hints.update(
            self.collect_structured_buffer_global_array_size_hints(ast, function_hints)
        )
        return global_hints, function_hints

    def collect_unsized_sampled_texture_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_sampled_texture_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_structured_buffer_global_array_size_hints(self, ast, function_hints):
        structured_globals = self.collect_unsized_structured_buffer_globals(ast)
        if not structured_globals:
            return {}

        required_sizes = {name: 0 for name in structured_globals}
        fixed_requirements = {}
        functions_by_name = {
            getattr(func, "name", None): func for func in self.collect_functions(ast)
        }
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }

        def raise_fixed_size_conflict(name, left, right):
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {left} and {right}"
            )

        def record_required_size(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and size > fixed_size:
                raise_fixed_size_conflict(name, size, fixed_size)
            required_sizes[name] = max(required_sizes[name], size)

        def record_fixed_requirement(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and fixed_size != size:
                raise_fixed_size_conflict(name, fixed_size, size)
            current_size = required_sizes.get(name, 0)
            if current_size and current_size > size:
                raise_fixed_size_conflict(name, current_size, size)
            fixed_requirements[name] = size
            required_sizes[name] = max(current_size, size)

        for node in self.walk_ast(ast):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                array_name = self.expression_name(array_expr)
                if array_name not in structured_globals:
                    continue
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                index = self.literal_int_value(index_expr, self.literal_int_constants)
                if index is not None and index >= 0:
                    record_required_size(array_name, index + 1)
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            callee_name = self.function_call_name(node)
            callee = functions_by_name.get(callee_name)
            if callee is None:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
            callee_hints = function_hints.get(callee_name, {})
            for index, arg in enumerate(args):
                if index >= len(params):
                    break
                arg_name = self.expression_name(arg)
                if arg_name not in structured_globals:
                    continue
                param = params[index]
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(param_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                param_name = getattr(param, "name", None)
                fixed_size = self.fixed_resource_array_size(param_type)
                if fixed_size is not None:
                    record_fixed_requirement(arg_name, fixed_size)
                else:
                    record_required_size(arg_name, callee_hints.get(param_name))

        def structured_buffer_param(params, index):
            if index >= len(params):
                return None, None, None
            param = params[index]
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            base_type, _ = self.structured_buffer_array_base_type(param_type)
            if base_type is None or not self.is_structured_buffer_type(base_type):
                return None, None, None
            return param, param_type, getattr(param, "name", None)

        def function_hint_size(func_name, param_name):
            return self.positive_int_value(
                function_hints.get(func_name, {}).get(param_name)
            )

        def record_function_required_size(func_name, param_name, size):
            size = self.positive_int_value(size)
            if not func_name or not param_name or size is None:
                return False
            hints = function_hints.setdefault(func_name, {})
            current_size = self.positive_int_value(hints.get(param_name))
            if current_size is not None and current_size >= size:
                return False
            hints[param_name] = str(size)
            return True

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions_by_name.items():
                caller_hints = function_hints.get(caller_name, {})
                for call in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue
                    params = list(
                        getattr(callee, "parameters", getattr(callee, "params", []))
                    )
                    args = list(
                        getattr(call, "arguments", getattr(call, "args", [])) or []
                    )
                    for index, arg in enumerate(args):
                        _, param_type, param_name = structured_buffer_param(
                            params, index
                        )
                        if (
                            param_type is None
                            or param_name is None
                            or self.fixed_resource_array_size(param_type) is not None
                        ):
                            continue

                        arg_name = self.expression_name(arg)
                        if not arg_name:
                            continue

                        callee_size = function_hint_size(callee_name, param_name)
                        if arg_name in required_sizes:
                            if callee_size is not None:
                                current_size = required_sizes.get(arg_name, 0)
                                record_required_size(arg_name, callee_size)
                                changed = (
                                    required_sizes.get(arg_name, 0) != current_size
                                    or changed
                                )
                            changed = (
                                record_function_required_size(
                                    callee_name,
                                    param_name,
                                    required_sizes.get(arg_name),
                                )
                                or changed
                            )
                            continue

                        caller_size = self.positive_int_value(
                            caller_hints.get(arg_name)
                        )
                        if caller_size is not None:
                            changed = (
                                record_function_required_size(
                                    callee_name, param_name, caller_size
                                )
                                or changed
                            )

        return {
            name: str(size)
            for name, size in required_sizes.items()
            if size is not None and size > 1
        }

    def collect_unsized_structured_buffer_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            base_type, declared_size = self.structured_buffer_array_base_type(vtype)
            if (
                name
                and base_type is not None
                and declared_size is None
                and self.is_structured_buffer_type(base_type)
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

    def collect_unsized_sampled_texture_parameters(self, ast):
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

    def is_unsized_sampled_texture_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_inferable_resource_array_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_inferable_resource_array_type(base_type)

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

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

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

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}

    def structured_buffer_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_structured_buffer_type(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_requires_counter(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "uint"
        element_type = type_name.split("<", 1)[1][:-1].strip()
        return self.map_type(element_type)

    def structured_buffer_array_parameter_info(
        self, vtype, param_name=None, function_name=None
    ):
        base_type, declared_size = self.structured_buffer_array_base_type(vtype)
        if base_type is None or not self.is_structured_buffer_type(base_type):
            return None

        count = declared_size
        if count is None and function_name and param_name:
            hint = self.function_resource_array_size_hints.get(function_name, {}).get(
                param_name
            )
            count = self.positive_int_value(hint)
        if count is None or count <= 0:
            return None
        return {
            "base_type": base_type,
            "element_type": self.structured_buffer_element_type(base_type),
            "count": count,
        }

    def structured_buffer_array_base_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return base_type, self.positive_int_value(vtype.size)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None, None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None, None
        base_type, size = parse_array_type(type_string)
        return base_type, self.positive_int_value(size)

    def collect_unsupported_structured_buffer_array_functions(self, functions):
        unsupported_functions = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            unsupported_params = []
            for index, param in enumerate(
                getattr(func, "parameters", getattr(func, "params", []))
            ):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(raw_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                if (
                    self.structured_buffer_array_parameter_info(
                        raw_type, getattr(param, "name", None), func_name
                    )
                    is not None
                ):
                    continue
                unsupported_params.append(
                    {"index": index, "name": getattr(param, "name", None)}
                )
            if unsupported_params:
                unsupported_functions[func_name] = {
                    "indices": {param["index"] for param in unsupported_params},
                    "params": unsupported_params,
                }
        return unsupported_functions

    def skipped_function_parameter_indices(self, func_name):
        skipped = set(self.function_sampler_parameter_indices.get(func_name, set()))
        preserved = getattr(self, "preserved_sampler_parameters", {}).get(
            func_name, set()
        )
        if preserved:
            parameter_names = self.function_parameter_names.get(func_name, [])
            skipped = {
                index
                for index in skipped
                if index >= len(parameter_names)
                or parameter_names[index] not in preserved
            }
        unsupported = self.unsupported_structured_buffer_array_functions.get(
            func_name, {}
        )
        skipped.update(unsupported.get("indices", set()))
        return skipped

    def unsupported_structured_buffer_array_function_body(
        self, func, unsupported_info, indent
    ):
        indent_str = "    " * indent
        code = ""
        func_name = getattr(func, "name", "<anonymous>")
        for param in unsupported_info.get("params", []):
            param_name = param.get("name") or "<anonymous>"
            code += (
                f"{indent_str}/* unsupported GLSL structured buffer array parameter "
                f"{func_name}.{param_name}: fixed array size is required for helper "
                "lowering */\n"
            )

        return_type = (
            self.current_function_return_type
            or self.type_name_string(getattr(func, "return_type", None))
            or "void"
        )
        mapped_return_type = self.map_type(return_type)
        if mapped_return_type == "void" or self.current_stage_output is not None:
            return f"{code}{indent_str}return;\n"
        return f"{code}{indent_str}return {self.zero_value_expression(return_type)};\n"

    def zero_value_expression(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "void":
            return ""
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type == "int":
            return "0"
        if mapped_type in {"float", "double"}:
            return "0.0"
        if mapped_type.startswith("bvec"):
            return f"{mapped_type}(false)"
        if mapped_type.startswith("uvec"):
            return f"{mapped_type}(0u)"
        if mapped_type.startswith("ivec"):
            return f"{mapped_type}(0)"
        if mapped_type.startswith(("vec", "dvec", "mat", "dmat")):
            return f"{mapped_type}(0.0)"
        struct = self.structs_by_name.get(mapped_type)
        if struct is not None:
            member_values = []
            for member in getattr(struct, "members", []) or []:
                member_type = getattr(
                    member, "member_type", getattr(member, "vtype", None)
                )
                member_values.append(self.zero_value_expression(member_type))
            return f"{mapped_type}({', '.join(member_values)})"
        return f"{mapped_type}(0)"

    def positive_int_value(self, value):
        literal = self.literal_int_value(value, self.literal_int_constants)
        if literal is None and isinstance(value, int) and not isinstance(value, bool):
            literal = value
        if literal is None and isinstance(value, str) and value.isdigit():
            literal = int(value)
        return literal if literal is not None and literal > 0 else None

    def structured_buffer_access_metadata(self, node):
        choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, choices)
        if not choices:
            return None
        return choices[0][1]

    def structured_buffer_effective_access(self, vtype, node=None):
        type_name = self.structured_buffer_type_name(vtype)
        explicit_access = self.structured_buffer_access_metadata(node)

        if type_name == "StructuredBuffer":
            if explicit_access in {"write", "read_write"}:
                access_name = image_access_diagnostic_name(explicit_access)
                node_name = self.resource_node_name(node, "<unnamed>")
                raise ValueError(
                    "OpenGL StructuredBuffer resource "
                    f"'{node_name}' cannot use {access_name} access metadata"
                )
            return "read"

        return explicit_access

    def structured_buffer_memory_qualifiers(self, vtype, node=None):
        effective_access = self.structured_buffer_effective_access(vtype, node)
        qualifiers = set()
        for qualifier in self.resource_memory_qualifiers(node).split():
            qualifiers.add(qualifier)
        if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
            qualifiers.add("readonly")
        elif effective_access == "read":
            qualifiers.add("readonly")
        elif effective_access == "write":
            qualifiers.add("writeonly")

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        order = ("coherent", "volatile", "restrict", "readonly", "writeonly")
        return " ".join(qualifier for qualifier in order if qualifier in qualifiers)

    def record_structured_buffer_access_metadata(
        self, name, vtype, node=None, parameter=False
    ):
        if not name or not self.is_structured_buffer_type(vtype):
            return
        access = self.structured_buffer_effective_access(vtype, node)
        if access is None or access == "read_write":
            return
        target = (
            self.current_structured_buffer_access_parameters
            if parameter
            else self.structured_buffer_variable_accesses
        )
        target[name] = access

    def structured_buffer_block_declaration(
        self, vtype, name, binding, array_size=None, node=None
    ):
        element_type = self.structured_buffer_element_type(vtype)
        memory_qualifiers = self.structured_buffer_memory_qualifiers(vtype, node)
        qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
        layout = self.glsl_resource_layout_prefix("std430", binding=binding)
        if array_size is not None:
            instance_member = "data"
            self.structured_buffer_instance_members[name] = instance_member
            array_suffix = f"[{array_size}]" if array_size else "[]"
            return (
                f"{layout} {qualifier_prefix}buffer "
                f"{name}Buffer {{ {element_type} {instance_member}[]; }} "
                f"{name}{array_suffix};\n"
            )
        return (
            f"{layout} {qualifier_prefix}buffer "
            f"{name}Buffer {{ {element_type} {name}[]; }};\n"
        )

    def structured_buffer_counter_resource_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_parameter_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_block_declaration(
        self, name, binding, array_size=None
    ):
        counter_member = self.structured_buffer_counter_resource_name(name)
        if array_size is not None:
            counter_member = "counter"
            counter_instance = f"{name}Counters"
            self.structured_buffer_counter_members[name] = counter_member
            self.structured_buffer_counter_instances[name] = counter_instance
            array_suffix = f"[{array_size}]" if array_size else "[]"
            layout = self.glsl_resource_layout_prefix("std430", binding=binding)
            return (
                f"{layout} buffer {name}CounterBuffer "
                f"{{ uint {counter_member}[]; }} {counter_instance}{array_suffix};\n"
            )

        self.structured_buffer_counter_members[name] = counter_member
        layout = self.glsl_resource_layout_prefix("std430", binding=binding)
        return (
            f"{layout} buffer {name}CounterBuffer " f"{{ uint {counter_member}[]; }};\n"
        )

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
            layout_parts = [
                self.attribute_value_to_string(argument) for argument in arguments
            ]
            layout_parts = [part for part in layout_parts if part]
            if layout_parts:
                self.validate_glsl_buffer_block_layout_parts(node, layout_parts)
                return ", ".join(layout_parts)
        return "std430"

    def validate_glsl_buffer_block_layout_parts(self, node, layout_parts):
        node_name = self.resource_node_name(node, "<unnamed>")
        seen = {}
        memory_layouts = []
        for layout_part in layout_parts:
            normalized = self.normalized_glsl_buffer_block_layout_part(layout_part)
            if normalized in seen:
                raise ValueError(
                    "Duplicate OpenGL buffer block layout metadata for "
                    f"'{node_name}': {layout_part}"
                )
            seen[normalized] = layout_part
            if normalized in self.GLSL_BUFFER_BLOCK_MEMORY_LAYOUT_NAMES:
                memory_layouts.append(layout_part)

        if len(memory_layouts) > 1:
            first_layout = memory_layouts[0]
            conflicting_layout = memory_layouts[1]
            raise ValueError(
                "Conflicting OpenGL buffer block memory layout metadata for "
                f"'{node_name}': {first_layout} differs from {conflicting_layout}"
            )

    def normalized_glsl_buffer_block_layout_part(self, layout_part):
        return str(layout_part).strip().lower().replace("_", "")

    def is_shader_record_buffer_block(self, node):
        layout_parts = {
            part.strip().lower().replace("_", "")
            for part in self.glsl_buffer_block_layout(node).split(",")
        }
        return "shaderrecordext" in layout_parts

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if (
            self.glsl_buffer_block_attribute(node) is None
            and "buffer" not in qualifiers
        ):
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

    def glsl_buffer_block_declaration(
        self, node, vtype, name, binding, array_suffix=""
    ):
        block_name = str(self.resource_base_type(vtype))
        struct = self.structs_by_name.get(block_name)
        if struct is None:
            return None

        layout = self.glsl_buffer_block_layout(node)
        memory_qualifiers = self.resource_memory_qualifiers(node)
        qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
        layout_prefix = self.glsl_resource_layout_prefix(layout, binding=binding)
        code = f"{layout_prefix} {qualifier_prefix}buffer " f"{block_name} {{\n"
        for member in getattr(struct, "members", []) or []:
            code += self.generate_struct_member_declaration(
                member,
                struct_name=block_name,
                preserve_unsized_arrays=True,
            )
        code += f"}} {name}{array_suffix};\n"
        return code

    def is_sampled_texture_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return (
            mapped_type != "sampler"
            and self.is_opaque_resource_type(mapped_type)
            and not mapped_type.startswith(("image", "iimage", "uimage"))
        )

    def is_storage_image_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return self.is_opaque_resource_type(mapped_type) and is_glsl_storage_image_type(
            mapped_type
        )

    def is_inferable_resource_array_type(self, vtype):
        return self.is_sampled_texture_type(vtype) or self.is_storage_image_type(vtype)

    def is_resource_array_hint_type(self, vtype):
        return self.is_inferable_resource_array_type(
            vtype
        ) or self.is_structured_buffer_type(vtype)

    def is_inferable_resource_type(self, vtype):
        return (
            self.is_sampler_type(vtype)
            or self.is_structured_buffer_type(vtype)
            or self.is_inferable_resource_array_type(vtype)
        )

    def canonical_sampled_resource_type(self, type_name):
        """Return GLSL sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return GLSL image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def canonical_resource_type(self, type_name):
        """Return canonical GLSL sampler/image/resource type for known aliases."""
        if isinstance(type_name, str):
            base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
            sampler_type = self.sampler_state_type_aliases.get(base_type)
            if sampler_type is not None:
                return sampler_type
        return self.canonical_sampled_resource_type(
            type_name
        ) or self.canonical_storage_resource_type(type_name)

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
            return self.canonical_resource_type(base_type) or base_type
        return self.canonical_resource_type(vtype) or vtype

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def resource_node_type(self, node):
        return getattr(
            node,
            "var_type",
            getattr(node, "param_type", getattr(node, "vtype", "float")),
        )

    def resource_node_name(self, node, default=None):
        return getattr(node, "name", getattr(node, "variable_name", default))

    def is_layout_bound_struct_uniform(self, node, vtype):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "uniform" not in qualifiers:
            return False
        return (
            str(self.resource_base_type(vtype)) in self.structs_by_name
            and self.explicit_resource_binding_index(node) is not None
        )

    def glsl_struct_uniform_block_declaration(
        self, vtype, var_name, binding, array_suffix=""
    ):
        block_name = str(self.resource_base_type(vtype))
        struct = self.structs_by_name[block_name]
        layout = self.glsl_resource_layout_prefix("std140", binding=binding)
        code = f"{layout} uniform {block_name} {{\n"
        for member in getattr(struct, "members", []) or []:
            code += self.generate_struct_member_declaration(
                member,
                struct_name=block_name,
                preserve_unsized_arrays=True,
            )
        code += f"}} {var_name}{array_suffix};\n"
        return code

    def pointer_buffer_structured_type(self, node, node_type):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" not in qualifiers or not hasattr(node_type, "pointee_type"):
            return None
        element_type = self.convert_type_node_to_string(node_type.pointee_type)
        return f"RWStructuredBuffer<{element_type}>"

    def resource_declaration_shape(self, node):
        node_type = self.resource_node_type(node)
        node_name = self.resource_node_name(node, "")
        resource_count = 1
        array_size = None
        array_suffix = ""

        pointer_buffer_type = self.pointer_buffer_structured_type(node, node_type)
        if pointer_buffer_type is not None:
            return pointer_buffer_type, None, "", resource_count

        if (
            hasattr(node_type, "element_type")
            and str(type(node_type)).find("ArrayType") != -1
        ):
            vtype = self.convert_type_node_to_string(node_type.element_type)
            array_size = (
                self.generate_expression(node_type.size)
                if node_type.size
                else (
                    self.resource_array_size_hints.get(node_name, "")
                    if self.is_resource_array_hint_type(vtype)
                    else ""
                )
            )
            if not array_size and self.is_inferable_resource_array_type(vtype):
                array_size = "1"
            array_suffix = f"[{array_size}]" if array_size else "[]"
            resource_count = self.resource_array_count(
                node_type.size if node_type.size else array_size
            )
            return vtype, array_size, array_suffix, resource_count

        if hasattr(node_type, "name") or hasattr(node_type, "element_type"):
            return self.convert_type_node_to_string(node_type), None, "", 1

        vtype = str(node_type) if node_type is not None else "float"
        if "[" in vtype and "]" in vtype:
            base_type, parsed_size = parse_array_type(vtype)
            array_size = parsed_size
            if array_size is None and self.is_inferable_resource_array_type(base_type):
                array_size = "1"
            array_suffix = f"[{parsed_size}]" if parsed_size else "[]"
            if array_size and not parsed_size:
                array_suffix = f"[{array_size}]"
            resource_count = self.resource_array_count(parsed_size)
            if array_size and not parsed_size:
                resource_count = self.resource_array_count(array_size)
            return base_type, array_size, array_suffix, resource_count

        return vtype, None, "", resource_count

    def resource_declaration_identity(self, node):
        self.validate_resource_access_metadata_operands(node)
        vtype, _, array_suffix, resource_count = self.resource_declaration_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            return (
                "buffer binding",
                "glsl_buffer_block",
                str(self.resource_base_type(vtype)),
                array_suffix,
                resource_count,
                self.glsl_buffer_block_layout(node),
                self.resource_memory_qualifiers(node),
            )
        if self.is_structured_buffer_type(vtype):
            return (
                "buffer binding",
                self.structured_buffer_type_name(vtype),
                self.structured_buffer_element_type(vtype),
                array_suffix,
                resource_count,
            )

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if mapped_type == "sampler" or not self.is_opaque_resource_type(mapped_type):
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if qualifiers & {"uniform", "buffer"}:
                return (
                    "qualified resource",
                    self.map_type(vtype),
                    array_suffix,
                    resource_count,
                    self.glsl_variable_layout_prefix(node),
                    self.glsl_variable_qualifier_prefix(node),
                )
            return None
        namespace = (
            "image binding" if self.is_storage_image_type(vtype) else "texture binding"
        )
        return (
            namespace,
            mapped_type,
            array_suffix,
            resource_count,
            self.image_format_qualifier(mapped_type, node),
            self.resource_memory_qualifiers(node),
            self.resource_precision_qualifiers(node),
        )

    def global_resource_shape(self, node):
        vtype, _, _, resource_count = self.resource_declaration_shape(node)
        return vtype, resource_count

    def global_resource_binding_metadata(self, node):
        var_name = self.resource_node_name(node)
        if not var_name:
            return None

        vtype, resource_count = self.global_resource_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            namespace = "buffer binding"
        elif self.is_structured_buffer_type(vtype):
            namespace = "buffer binding"
        elif self.is_layout_bound_struct_uniform(node, vtype):
            namespace = "uniform buffer binding"
        else:
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                if not self.should_preserve_sampler_resource(var_name):
                    return None
                namespace = "sampler binding"
            elif not self.is_opaque_resource_type(mapped_type):
                return None
            else:
                namespace = (
                    "image binding"
                    if self.is_storage_image_type(vtype)
                    else "texture binding"
                )

        binding = self.explicit_resource_binding_index(node)
        if binding is None:
            return None
        return namespace, binding, resource_count, var_name

    def reserve_explicit_global_resource_bindings(self, global_vars, used_bindings):
        used_source_bindings = {}
        for node in global_vars:
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            namespace, binding, resource_count, var_name = metadata
            source_binding = self.source_descriptor_binding_index(node)
            if source_binding is not None:
                self.reserve_resource_binding_range(
                    used_source_bindings,
                    "source descriptor",
                    "descriptor binding",
                    source_binding,
                    1,
                    var_name,
                )
            if self.should_remap_descriptor_array_target_binding(node, resource_count):
                continue
            self.reserve_resource_binding_range(
                used_bindings,
                "OpenGL",
                namespace,
                binding,
                resource_count,
                var_name,
            )

    def next_resource_binding(self, binding_cursors, namespace):
        return binding_cursors.get(namespace, 0)

    def next_available_resource_binding(
        self, used_bindings, binding_cursors, namespace, count
    ):
        count = max(count or 1, 1)
        binding = self.next_resource_binding(binding_cursors, namespace)
        return self.next_available_resource_binding_from(
            used_bindings, namespace, binding, count
        )

    def next_available_resource_binding_from(
        self, used_bindings, namespace, binding, count
    ):
        count = max(count or 1, 1)
        ranges = used_bindings.get(namespace, [])
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

    def opengl_resource_binding_for_declaration(
        self, node, used_bindings, binding_cursors, namespace, count
    ):
        explicit_binding = self.explicit_resource_binding_index(node)
        if explicit_binding is None:
            return self.next_available_resource_binding(
                used_bindings, binding_cursors, namespace, count
            )
        if self.should_remap_descriptor_array_target_binding(node, count):
            return self.next_available_resource_binding_from(
                used_bindings, namespace, explicit_binding, count
            )
        return explicit_binding

    def advance_resource_binding(self, binding_cursors, namespace, start, count):
        count = max(count or 1, 1)
        binding_cursors[namespace] = max(
            binding_cursors.get(namespace, 0), start + count
        )

    def reserve_resource_binding_range(
        self, used_bindings, target, namespace, start, count, name
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        ranges = used_bindings.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting {target} resource binding for '{name}': "
                    f"{self.resource_binding_range_label(namespace, start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.resource_binding_range_label(namespace, used_start, used_end)}"
                )
        ranges.append((start, end, name))

    def resource_binding_range_label(self, namespace, start, end):
        if start == end:
            return f"{namespace} {start}"
        return f"{namespace} {start}-{end}"

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

    def map_type(self, vtype):
        """Map types to GLSL equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        if isinstance(vtype, PointerType):
            return f"{self.map_type(vtype.pointee_type)}*"
        if isinstance(vtype, ReferenceType):
            return f"{self.map_type(vtype.referenced_type)}&"

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if self.is_ray_query_type_name(vtype_str):
            return "rayQueryEXT"

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        canonical_type = self.canonical_resource_type(vtype_str)
        if canonical_type is not None:
            return canonical_type

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

        return self.type_mapping.get(vtype_str, vtype_str)

    def is_ray_query_type(self, vtype):
        if vtype is None:
            return False
        return self.is_ray_query_type_name(self.type_name_string(vtype))

    def is_acceleration_structure_type(self, vtype):
        if vtype is None:
            return False
        return self.resource_base_type(vtype) == "accelerationStructureEXT"

    def is_ray_query_type_name(self, type_name):
        type_name = str(type_name)
        return (
            type_name == "rayQueryEXT"
            or type_name == "rayQuery"
            or type_name == "RayQuery"
            or type_name.startswith("RayQuery<")
        )

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" in qualifiers and isinstance(vtype, PointerType):
            return f"{self.map_type(vtype.pointee_type)}[]"
        if self.is_structured_buffer_type(vtype):
            return f"{self.structured_buffer_element_type(vtype)}[]"

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_inferable_resource_array_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return f"{mapped_type}[{array_size or 1}]"

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_inferable_resource_array_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return f"{mapped_type}[{array_size or 1}]"
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if (
            isinstance(vtype, (PointerType, ReferenceType))
            or hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
        ):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def invalid_glsl_image_format_message(self, node, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Invalid OpenGL image format metadata for "
            f"'{node_name}': format {source} is not a supported storage image format"
        )

    def glsl_image_format_choice_description(self, attr_name, source):
        if attr_name == "format":
            source = source if source is not None else "<missing>"
            return f"format {source}"
        return f"@{attr_name}"

    def duplicate_glsl_image_format_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        description = self.glsl_image_format_choice_description(attr_name, source)
        return (
            "Duplicate OpenGL image format metadata for "
            f"'{node_name}': {description}"
        )

    def conflicting_glsl_image_format_message(
        self, node, previous_attr, previous_source, attr_name, source
    ):
        node_name = self.resource_node_name(node, "<unnamed>")
        previous = self.glsl_image_format_choice_description(
            previous_attr, previous_source
        )
        current = self.glsl_image_format_choice_description(attr_name, source)
        return (
            "Conflicting OpenGL image format metadata for "
            f"'{node_name}': {previous} differs from {current}"
        )

    def explicit_glsl_image_format(self, node):
        if node is None or not hasattr(node, "attributes"):
            return None
        supported_formats = supported_image_formats()
        choices = []
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                choices.append((attr_name, attr_name, attr_name))
                continue
            if attr_name != "format":
                continue

            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            if source is None:
                raise ValueError(self.invalid_glsl_image_format_message(node, source))
            format_name = str(source).lower()
            if format_name not in supported_formats:
                raise ValueError(self.invalid_glsl_image_format_message(node, source))
            choices.append((attr_name, source, format_name))

        if not choices:
            return None

        seen_by_attribute = {}
        for attr_name, source, format_name in choices:
            previous = seen_by_attribute.get(attr_name)
            if previous is not None:
                previous_source, previous_format = previous
                if format_name == previous_format:
                    raise ValueError(
                        self.duplicate_glsl_image_format_message(
                            node, attr_name, source
                        )
                    )
                raise ValueError(
                    self.conflicting_glsl_image_format_message(
                        node, attr_name, previous_source, attr_name, source
                    )
                )
            seen_by_attribute[attr_name] = (source, format_name)

        first_attr, first_source, first_format = choices[0]
        for attr_name, source, format_name in choices[1:]:
            if format_name == first_format:
                continue
            raise ValueError(
                self.conflicting_glsl_image_format_message(
                    node, first_attr, first_source, attr_name, source
                )
            )
        return first_format

    def validate_glsl_image_format_target(self, node, vtype, explicit_format):
        if explicit_format is None or self.is_storage_image_type(vtype):
            return
        node_name = self.resource_node_name(node, "<unnamed>")
        type_name = self.type_name_string(self.resource_base_type(vtype))
        raise ValueError(
            "OpenGL image format metadata for "
            f"'{node_name}' applies only to storage image resources, "
            f"got {type_name}"
        )

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = self.explicit_glsl_image_format(node)
        self.validate_glsl_image_format_target(node, base_type, explicit_format)
        if explicit_format in {
            "r8",
            "r8_snorm",
            "r16",
            "r16_snorm",
            "r16f",
            "r32f",
            "r8i",
            "r16i",
            "r32i",
            "r8ui",
            "r16ui",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg8i",
            "rg16i",
            "rg8ui",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba32f",
            "rgba8i",
            "rgba16i",
            "rgba32i",
            "rgba8ui",
            "rgba16ui",
            "rgba32ui",
        }:
            if explicit_format in {
                "r8",
                "r8_snorm",
                "r16",
                "r16_snorm",
                "r16f",
                "rg8",
                "rg8_snorm",
                "rg16",
                "rg16_snorm",
                "rg16f",
                "rg32f",
                "rgba8",
                "rgba8_snorm",
                "rgba16",
                "rgba16_snorm",
                "rgba16f",
                "rgba32f",
            }:
                format_class = "r32f"
            elif explicit_format.endswith("ui"):
                format_class = "r32ui"
            elif explicit_format.endswith("i"):
                format_class = "r32i"
            else:
                format_class = explicit_format
            format_type_map = {
                "image1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "iimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "uimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "image2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "iimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "uimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "image3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "iimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "uimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "image1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "iimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "uimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "image2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "iimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "uimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "image2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "iimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "uimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "image2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "iimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "uimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "imageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "iimageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "uimageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "imageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
                "iimageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
                "uimageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
            }
            mapped_type = format_type_map.get(base_type, {}).get(format_class)
            if mapped_type:
                return mapped_type

        return self.map_type(vtype)

    def is_opaque_resource_type(self, vtype):
        return vtype in {
            "texture1D",
            "texture1DArray",
            "texture2D",
            "texture3D",
            "textureCube",
            "texture2DArray",
            "textureCubeArray",
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "samplerExternalOES",
            "__samplerExternal2DY2YEXT",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "sampler2DRect",
            "samplerBuffer",
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler1D",
            "isampler1DArray",
            "isampler2D",
            "isampler2DArray",
            "isampler3D",
            "isamplerCube",
            "isamplerCubeArray",
            "isampler2DRect",
            "isamplerBuffer",
            "isampler2DMS",
            "isampler2DMSArray",
            "usampler1D",
            "usampler1DArray",
            "usampler2D",
            "usampler2DArray",
            "usampler3D",
            "usamplerCube",
            "usamplerCubeArray",
            "usampler2DRect",
            "usamplerBuffer",
            "usampler2DMS",
            "usampler2DMSArray",
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DArray",
            "image2DMS",
            "image2DMSArray",
            "imageBuffer",
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
            "atomic_uint",
            "accelerationStructureEXT",
        }

    def supported_image_formats(self):
        return supported_image_formats()

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
            "readwrite",
        }

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

    def invalid_resource_binding_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        if attr_name == "register":
            requirement = "must use b/s/t/u register syntax or an integer binding"
        else:
            requirement = "must resolve to a concrete integer binding"
        return (
            "Invalid OpenGL resource binding metadata for "
            f"'{node_name}': {attr_name} {source} {requirement}"
        )

    def unsupported_resource_space_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Unsupported OpenGL resource binding metadata for "
            f"'{node_name}': {attr_name} {source} is not supported by OpenGL GLSL"
        )

    def resource_register_prefix(self, source):
        if source is None:
            return None
        raw_source = str(source).strip().lower()
        if len(raw_source) >= 2 and raw_source[0] in {"b", "s", "t", "u"}:
            suffix = raw_source[1:]
            if suffix.isdigit():
                return raw_source[0]
        return None

    def expected_resource_register_prefix(self, node):
        if (
            hasattr(node, "members")
            and not hasattr(node, "var_type")
            and not hasattr(node, "param_type")
        ):
            return "b", "uniform buffer binding"

        vtype = self.resource_node_type(node)

        if self.is_glsl_buffer_block_variable(node, vtype):
            return "u", "buffer binding"

        if self.is_structured_buffer_type(vtype):
            if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
                return "t", "buffer binding"
            return "u", "buffer binding"

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if self.is_opaque_resource_type(mapped_type):
            if self.is_storage_image_type(vtype):
                return "u", "image binding"
            return "t", "texture binding"

        if mapped_type == "sampler":
            return "s", "sampler binding"

        return None, None

    def accepts_hlsl_sampler_register_for_texture_binding(self, node):
        vtype = self.resource_node_type(node)
        mapped_type = self.map_resource_type_with_format(vtype, node)
        return (
            self.is_opaque_resource_type(mapped_type)
            and not self.is_storage_image_type(vtype)
            and mapped_type.startswith(("sampler", "isampler", "usampler"))
        )

    def validate_resource_register_prefix(self, node, source):
        actual_prefix = self.resource_register_prefix(source)
        if actual_prefix is None:
            return

        expected_prefix, namespace = self.expected_resource_register_prefix(node)
        if expected_prefix is None or actual_prefix == expected_prefix:
            return
        if (
            expected_prefix == "t"
            and actual_prefix == "s"
            and self.accepts_hlsl_sampler_register_for_texture_binding(node)
        ):
            return

        node_name = self.resource_node_name(node, "<unnamed>")
        raise ValueError(
            "Incompatible OpenGL resource register metadata for "
            f"'{node_name}': register {source} uses {actual_prefix}-register, "
            f"expected {expected_prefix}-register for {namespace}"
        )

    def has_descriptor_set_attribute(self, node):
        if not hasattr(node, "attributes"):
            return False
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() in {"set", "space"}:
                return True
        return False

    def has_explicit_opengl_target_binding_attribute(self, node):
        if not hasattr(node, "attributes"):
            return False
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() in {
                "buffer",
                "register",
                "sampler",
                "texture",
            }:
                return True
        return False

    def source_descriptor_binding_index(self, node):
        if not self.has_descriptor_set_attribute(node):
            return None
        choices = self.explicit_resource_binding_choices(node)
        binding_choices = [
            binding for attr_name, _, binding in choices if attr_name == "binding"
        ]
        if not binding_choices:
            return None
        self.explicit_resource_binding_index(node)
        return binding_choices[0]

    def should_remap_descriptor_array_target_binding(self, node, resource_count):
        return (
            resource_count > 1
            and self.source_descriptor_binding_index(node) is not None
            and not self.has_explicit_opengl_target_binding_attribute(node)
        )

    def invalid_resource_access_message(self, node, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Invalid OpenGL resource access metadata for "
            f"'{node_name}': access({source}) must be readonly, writeonly, "
            "or readwrite"
        )

    def explicit_resource_binding_choices(self, node):
        choices = []
        if not hasattr(node, "attributes"):
            return choices
        descriptor_set = 0
        descriptor_set_description = None
        node_name = self.resource_node_name(node, "<unnamed>")
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name not in {"set", "space"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            resource_space = (
                self.binding_index_value(arguments[0]) if arguments else None
            )
            if resource_space is None:
                raise ValueError(
                    self.invalid_resource_binding_message(node, attr_name, source)
                )
            description = f"{attr_name} {source if source is not None else '<missing>'}"
            if descriptor_set_description is not None:
                if resource_space == descriptor_set:
                    raise ValueError(
                        "Duplicate OpenGL resource binding metadata for "
                        f"'{node_name}': {description}"
                    )
                raise ValueError(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {descriptor_set_description} differs from "
                    f"{description}"
                )
            descriptor_set = resource_space
            descriptor_set_description = description

        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in {"set", "space"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture"}:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], ("b", "s", "t", "u"))
                self.validate_resource_register_prefix(node, source)
            else:
                continue
            if binding is None:
                raise ValueError(
                    self.invalid_resource_binding_message(node, attr_name, source)
                )
            if descriptor_set:
                binding += descriptor_set * self.OPENGL_DESCRIPTOR_SET_BINDING_STRIDE
            choices.append((attr_name, source, binding))
        return choices

    def explicit_resource_binding_choice_description(self, attr_name, source, binding):
        if attr_name == "binding":
            source = source if source is not None else binding
            return f"binding {source}"
        source = source if source is not None else binding
        return f"{attr_name} {source} binding {binding}"

    def explicit_resource_binding_index(self, node):
        choices = self.explicit_resource_binding_choices(node)
        if not choices:
            return None

        node_name = self.resource_node_name(node, "<unnamed>")
        seen_by_attribute = {}
        for attr_name, source, binding in choices:
            description = self.explicit_resource_binding_choice_description(
                attr_name, source, binding
            )
            previous = seen_by_attribute.get(attr_name)
            if previous is not None:
                previous_description, previous_binding = previous
                if binding == previous_binding:
                    raise ValueError(
                        "Duplicate OpenGL resource binding metadata for "
                        f"'{node_name}': {description}"
                    )
                raise ValueError(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {previous_description} differs from "
                    f"{description}"
                )
            seen_by_attribute[attr_name] = (description, binding)

        first_name, first_source, first_binding = choices[0]
        first_description = self.explicit_resource_binding_choice_description(
            first_name, first_source, first_binding
        )
        for attr_name, source, binding in choices[1:]:
            if binding != first_binding:
                current_description = self.explicit_resource_binding_choice_description(
                    attr_name, source, binding
                )
                raise ValueError(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {first_description} differs from "
                    f"{current_description}"
                )
        return first_binding

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if self.glsl_stage_control_attribute_name(attr):
                continue
            if (
                self.is_glsl_stage_io_metadata_attribute(attr)
                or is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_hlsl_metadata_attribute(attr)
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def glsl_stage_control_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]

        valid_names = {
            "cw",
            "ccw",
            "compute",
            "domain",
            "early_fragment_tests",
            "equal_spacing",
            "fragment",
            "fractional_even_spacing",
            "fractional_odd_spacing",
            "invocations",
            "inputtopology",
            "isolines",
            "layout",
            "local_size",
            "line_strip",
            "lines",
            "lines_adjacency",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "max_vertices",
            "max_primitives",
            "maxprimitivecount",
            "maxvertexcount",
            "numthreads",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
            "point_mode",
            "points",
            "quads",
            "stage_entry",
            "stream",
            "triangles",
            "triangles_adjacency",
            "triangle_strip",
            "vertex",
            "vertices",
            "workgroup_size",
        } | self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES
        if normalized in valid_names:
            return normalized
        return None

    def function_return_semantic(self, func):
        semantic = getattr(func, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(func, "attributes"):
            return None
        for attr in func.attributes:
            if self.glsl_stage_control_attribute_name(attr):
                continue
            if (
                self.is_glsl_stage_io_metadata_attribute(attr)
                or is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_hlsl_metadata_attribute(attr)
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def is_hlsl_metadata_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {"rootsignature", "hlsl_rootsignature"}

    def image_format_qualifier(self, vtype, node=None):
        explicit_format = self.explicit_glsl_image_format(node)
        self.validate_glsl_image_format_target(node, vtype, explicit_format)
        if explicit_format:
            return explicit_format
        if vtype in {
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DArray",
        }:
            return "rgba32f"
        if vtype in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
        }:
            return "r32i"
        if vtype in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
        }:
            return "r32ui"
        return None

    def opaque_resource_layout(self, vtype, binding, node=None):
        image_format = self.image_format_qualifier(vtype, node)
        return self.glsl_resource_layout_prefix(image_format, binding=binding)

    def resource_precision_qualifiers(self, node):
        qualifiers = [
            qualifier
            for qualifier in self.glsl_variable_qualifier_prefix(node).split()
            if qualifier in self.GLSL_PRECISION_QUALIFIERS
        ]
        return " ".join(qualifiers)

    def resource_memory_qualifiers(self, node):
        supported = {
            "coherent",
            "globallycoherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
        }
        qualifiers = set()
        access_choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, access_choices)

        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier = str(qualifier).lower()
            if qualifier in supported:
                qualifiers.add(qualifier)

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported:
                qualifiers.add(attr_name)

        for _, access in access_choices:
            if access == "read":
                qualifiers.add("readonly")
            elif access == "write":
                qualifiers.add("writeonly")

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        order = ("coherent", "volatile", "restrict", "readonly", "writeonly")
        return " ".join(qualifier for qualifier in order if qualifier in qualifiers)

    def resource_access_metadata_choices(self, node):
        choices = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            source = str(qualifier).lower()
            access = normalized_image_access(source)
            if access is not None:
                choices.append((source, access))

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                source = (
                    self.attribute_value_to_string(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if len(arguments) != 1:
                    raise ValueError(self.invalid_resource_access_message(node, source))
                source = self.attribute_value_to_string(arguments[0])
                access = normalized_image_access(source)
                if access is None:
                    raise ValueError(self.invalid_resource_access_message(node, source))
                choices.append((f"access({source})", access))
                continue

            access = normalized_image_access(attr_name)
            if access is not None:
                choices.append((attr_name, access))
        return choices

    def validate_resource_access_metadata_operands(self, node):
        self.resource_access_metadata_choices(node)

    def validate_resource_access_metadata_consistency(self, node, choices):
        if not choices:
            return
        first_source, first_access = choices[0]
        node_name = self.resource_node_name(node, "<unnamed>")
        seen_sources = {first_source}
        for source, access in choices[1:]:
            if source in seen_sources:
                raise ValueError(
                    "Duplicate OpenGL resource access metadata for "
                    f"'{node_name}': {self.resource_access_metadata_label(source)}"
                )
            seen_sources.add(source)
            if access != first_access:
                raise ValueError(
                    "Conflicting OpenGL resource access metadata for "
                    f"'{node_name}': {first_source} differs from {source}"
                )

    def resource_access_metadata_label(self, source):
        return f"@{source}"

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_OR": "|=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "ASSIGN_XOR": "^=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "MOD": "%",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "LOGICAL_AND": "&&",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the corresponding GLSL builtin or layout."""
        if semantic is not None:
            return f"{self.semantic_map.get(semantic, semantic)}"
        else:
            return ""

    def map_stage_input_semantic(self, semantic, stage_name=None):
        mapped = self.map_semantic(semantic)
        if normalize_stage_name(stage_name) == "fragment":
            if mapped == "gl_Position":
                return "gl_FragCoord"
            semantic_text = str(semantic).lower() if semantic is not None else ""
            if semantic_text in {
                "sv_coverage",
                "sample_mask",
                "gl_samplemask",
                "gl_samplemaskin",
                "sample_mask_in",
            }:
                return "gl_SampleMaskIn"
        return mapped

    def stage_input_builtin_alias(self, semantic, stage_name=None):
        mapped = self.map_stage_input_semantic(semantic, stage_name)
        if (
            normalize_stage_name(stage_name) == "fragment"
            and mapped == "gl_SampleMaskIn"
        ):
            return "gl_SampleMaskIn[0]"
        return mapped

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, PointerType):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if isinstance(type_node, ReferenceType):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"{referenced_type}&"
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
                return f"mat{type_node.rows}"
            return f"mat{type_node.cols}x{type_node.rows}"
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
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                elif element_type == "bool":
                    return f"bvec{size}"
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

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        semantic_attrs = [
            "position",
            "color",
            "texcoord",
            "normal",
            "tangent",
            "binormal",
            "POSITION",
            "COLOR",
            "TEXCOORD",
            "NORMAL",
            "TANGENT",
            "BINORMAL",
            "TEXCOORD0",
            "TEXCOORD1",
            "TEXCOORD2",
            "TEXCOORD3",
        ]

        for attr in attributes:
            if hasattr(attr, "name") and attr.name in semantic_attrs:
                return attr.name
        return None

    def generate_array_declaration(self, stmt, indent):
        indent_str = "    " * indent
        element_type = self.map_type(stmt.element_type)
        size = get_array_size_from_node(stmt)

        if size is None:
            # In GLSL, dynamic sized arrays need special handling
            # For instance in shader storage blocks, but for simple cases:
            return f"{indent_str}{element_type} {stmt.name}[];\n"
        else:
            return f"{indent_str}{element_type} {stmt.name}[{size}];\n"

    def glsl_attribute(self, node, name):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == name:
                return attr
        return None

    def is_glsl_interface_block_struct(self, node):
        return self.glsl_attribute(node, "glsl_interface_block") is not None

    def glsl_attribute_arguments(self, node, name):
        attr = self.glsl_attribute(node, name)
        if attr is None:
            return []
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def glsl_interface_block_qualifiers(self, node):
        return [
            self.attribute_value_to_string(argument)
            for argument in self.glsl_attribute_arguments(node, "glsl_interface_block")
        ]

    def glsl_interface_block_name(self, node):
        block_name_args = self.glsl_attribute_arguments(
            node, "glsl_interface_block_name"
        )
        if block_name_args:
            return self.attribute_value_to_string(block_name_args[0])
        return node.name

    def glsl_interface_block_instance_suffix(self, node):
        instance_args = self.glsl_attribute_arguments(node, "glsl_interface_instance")
        if not instance_args:
            return ""

        instance_name = self.attribute_value_to_string(instance_args[0])
        array_attr = self.glsl_attribute(node, "glsl_interface_array")
        array_suffix = ""
        if array_attr is not None:
            array_args = getattr(
                array_attr, "arguments", getattr(array_attr, "args", [])
            )
            if array_args:
                array_suffix = "".join(
                    f"[{self.attribute_value_to_string(arg)}]" for arg in array_args
                )
            else:
                array_suffix = "[]"
        return f" {instance_name}{array_suffix}"

    def glsl_struct_member_name_map(self, struct_name):
        if not struct_name:
            return {}
        if struct_name in self.struct_member_name_maps:
            return self.struct_member_name_maps[struct_name]

        struct = self.structs_by_name.get(struct_name)
        member_names = [
            getattr(member, "name", None)
            for member in getattr(struct, "members", []) or []
        ]
        member_names = [name for name in member_names if name]
        used_names = set()
        member_map = {}
        for name in member_names:
            candidate = name
            if candidate in self.GLSL_RESERVED_IDENTIFIERS or candidate in used_names:
                candidate = f"{name}_"
                suffix = 1
                while candidate in used_names:
                    suffix += 1
                    candidate = f"{name}_{suffix}"
            member_map[name] = candidate
            used_names.add(candidate)

        self.struct_member_name_maps[struct_name] = member_map
        return member_map

    def glsl_struct_member_name(self, struct_name, member_name):
        return self.glsl_struct_member_name_map(struct_name).get(
            member_name, member_name
        )

    def generate_glsl_interface_block_member_declaration(
        self, member, indent="    ", struct_name=None
    ):
        qualifier = self.glsl_variable_qualifier_prefix(member)
        qualifier = f"{qualifier} " if qualifier else ""
        member_name = self.glsl_struct_member_name(
            struct_name, getattr(member, "name", "")
        )

        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            if member.size:
                return (
                    f"{indent}{qualifier}{self.map_type(element_type)} "
                    f"{member_name}[{member.size}];\n"
                )
            return (
                f"{indent}{qualifier}{self.map_type(element_type)} {member_name}[];\n"
            )

        if hasattr(member, "member_type"):
            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_type(member_type_str)
            if str(type(member.member_type)).find("ArrayType") != -1:
                declaration = format_c_style_array_declaration(member_type, member_name)
                return f"{indent}{qualifier}{declaration};\n"
            return f"{indent}{qualifier}{member_type} {member_name};\n"

        if hasattr(member, "vtype"):
            member_type = self.map_type(member.vtype)
            return f"{indent}{qualifier}{member_type} {member_name};\n"

        return f"{indent}{qualifier}float {member_name};\n"

    def generate_glsl_interface_block_declaration(self, node):
        layout = self.glsl_variable_layout_prefix(node)
        qualifiers = self.glsl_interface_block_qualifiers(node)
        qualifier_prefix = " ".join(qualifiers)
        if qualifier_prefix:
            qualifier_prefix += " "

        block_name = self.glsl_interface_block_name(node)
        code = f"{layout}{qualifier_prefix}{block_name} {{\n"
        for member in getattr(node, "members", []) or []:
            code += self.generate_glsl_interface_block_member_declaration(
                member, struct_name=node.name
            )
        code += f"}}{self.glsl_interface_block_instance_suffix(node)};\n"
        return code

    def generate_struct_member_declaration(
        self,
        member,
        indent="    ",
        struct_name=None,
        preserve_unsized_arrays=False,
    ):
        member_name = self.glsl_struct_member_name(
            struct_name, getattr(member, "name", "")
        )
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            if member.size:
                return f"{indent}{self.map_type(element_type)} {member_name}[{member.size}];\n"
            if preserve_unsized_arrays:
                return f"{indent}{self.map_type(element_type)} {member_name}[];\n"
            return f"{indent}{self.map_type(element_type)} {member_name}[1024];\n"

        if hasattr(member, "member_type"):
            if str(type(member.member_type)).find("ArrayType") != -1:
                member_type_str = self.convert_type_node_to_string(member.member_type)
                member_type = self.map_type(member_type_str)
                if str(member_type).endswith("[]") and not preserve_unsized_arrays:
                    member_type = f"{str(member_type)[:-2]}[1024]"
                declaration = format_c_style_array_declaration(member_type, member_name)
                return f"{indent}{declaration};\n"

            member_type_str = self.convert_type_node_to_string(member.member_type)
            member_type = self.map_type(member_type_str)
            return f"{indent}{member_type} {member_name};\n"

        if hasattr(member, "vtype"):
            member_type = self.map_type(member.vtype)
            return f"{indent}{member_type} {member_name};\n"

        return f"{indent}float {member_name};\n"

    def generate_struct(self, node):
        code = f"struct {node.name} {{\n"
        for member in getattr(node, "members", []) or []:
            code += self.generate_struct_member_declaration(
                member, struct_name=node.name
            )
        code += "};\n"
        return code

    def generate_expression_statement(self, stmt):
        """Generate code for expression statements."""
        if hasattr(stmt, "expression"):
            expr = self.generate_expression(stmt.expression)
            return expr
        else:
            return self.generate_expression(stmt)
