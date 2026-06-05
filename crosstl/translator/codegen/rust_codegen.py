"""CrossGL-to-Rust code generator."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ConstantNode,
    ConstructorNode,
    ConstructorPatternNode,
    ContinueNode,
    DoWhileNode,
    EnumNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LambdaNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    ParameterNode,
    PointerAccessNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    StructNode,
    StructPatternNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import get_array_size_from_node
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .stage_utils import normalize_stage_name, stage_matches


class RustCodeGen:
    """Emit Rust-like GPU shader source from the shared CrossGL AST."""

    def __init__(self):
        """Initialize Rust type maps and expression-generation state."""
        self.current_shader = None
        self.current_shader_type = None
        self.type_mapping = {
            # Scalar Types
            "void": "()",
            "int": "i32",
            "short": "i16",
            "long": "i64",
            "uint": "u32",
            "ushort": "u16",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "bool": "bool",
            "string": "&'static str",
            "str": "&'static str",
            "char": "char",
            # Vector Types (using GPU-style vector types)
            "vec2<f32>": "Vec2<f32>",
            "vec3<f32>": "Vec3<f32>",
            "vec4<f32>": "Vec4<f32>",
            "vec2<f64>": "Vec2<f64>",
            "vec3<f64>": "Vec3<f64>",
            "vec4<f64>": "Vec4<f64>",
            "vec2<i32>": "Vec2<i32>",
            "vec3<i32>": "Vec3<i32>",
            "vec4<i32>": "Vec4<i32>",
            "vec2<u32>": "Vec2<u32>",
            "vec3<u32>": "Vec3<u32>",
            "vec4<u32>": "Vec4<u32>",
            "vec2<bool>": "Vec2<bool>",
            "vec3<bool>": "Vec3<bool>",
            "vec4<bool>": "Vec4<bool>",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "dvec2": "Vec2<f64>",
            "dvec3": "Vec3<f64>",
            "dvec4": "Vec4<f64>",
            "bvec2": "Vec2<bool>",
            "bvec3": "Vec3<bool>",
            "bvec4": "Vec4<bool>",
            "bool2": "Vec2<bool>",
            "bool3": "Vec3<bool>",
            "bool4": "Vec4<bool>",
            "float2": "Vec2<f32>",
            "float3": "Vec3<f32>",
            "float4": "Vec4<f32>",
            "int2": "Vec2<i32>",
            "int3": "Vec3<i32>",
            "int4": "Vec4<i32>",
            "uint2": "Vec2<u32>",
            "uint3": "Vec3<u32>",
            "uint4": "Vec4<u32>",
            # Matrix Types
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            "mat2x2": "Mat2<f32>",
            "mat2x3": "Mat2x3<f32>",
            "mat2x4": "Mat2x4<f32>",
            "mat3x2": "Mat3x2<f32>",
            "mat3x3": "Mat3<f32>",
            "mat3x4": "Mat3x4<f32>",
            "mat4x2": "Mat4x2<f32>",
            "mat4x3": "Mat4x3<f32>",
            "mat4x4": "Mat4<f32>",
            "dmat2": "Mat2<f64>",
            "dmat3": "Mat3<f64>",
            "dmat4": "Mat4<f64>",
            "dmat2x2": "Mat2<f64>",
            "dmat2x3": "Mat2x3<f64>",
            "dmat2x4": "Mat2x4<f64>",
            "dmat3x2": "Mat3x2<f64>",
            "dmat3x3": "Mat3<f64>",
            "dmat3x4": "Mat3x4<f64>",
            "dmat4x2": "Mat4x2<f64>",
            "dmat4x3": "Mat4x3<f64>",
            "dmat4x4": "Mat4<f64>",
            # Texture Types
            "sampler1D": "Texture1D<f32>",
            "sampler1DArray": "Texture1DArray<f32>",
            "sampler2D": "Texture2D<f32>",
            "sampler2DArray": "Texture2DArray<f32>",
            "sampler3D": "Texture3D<f32>",
            "samplerCube": "TextureCube<f32>",
            "samplerCubeArray": "TextureCubeArray<f32>",
            "sampler2DMS": "Texture2DMS<f32>",
            "sampler2DMSArray": "Texture2DMSArray<f32>",
            "sampler2DShadow": "DepthTexture2D<f32>",
            "sampler2DArrayShadow": "DepthTexture2DArray<f32>",
            "samplerCubeShadow": "DepthTextureCube<f32>",
            "samplerCubeArrayShadow": "DepthTextureCubeArray<f32>",
            "sampler": "Sampler",
            "SamplerState": "Sampler",
            "SamplerComparisonState": "Sampler",
            "Texture1D": "Texture1D<f32>",
            "Texture1DArray": "Texture1DArray<f32>",
            "Texture2D": "Texture2D<f32>",
            "Texture2DArray": "Texture2DArray<f32>",
            "Texture3D": "Texture3D<f32>",
            "TextureCube": "TextureCube<f32>",
            "TextureCubeArray": "TextureCubeArray<f32>",
            "Texture2DMS": "Texture2DMS<f32>",
            "Texture2DMSArray": "Texture2DMSArray<f32>",
            "Texture2DShadow": "DepthTexture2D<f32>",
            "Texture2DArrayShadow": "DepthTexture2DArray<f32>",
            "TextureCubeShadow": "DepthTextureCube<f32>",
            "TextureCubeArrayShadow": "DepthTextureCubeArray<f32>",
            "image1D": "Image1D<Vec4<f32>>",
            "image1DArray": "Image1DArray<Vec4<f32>>",
            "image2D": "Image2D<Vec4<f32>>",
            "image3D": "Image3D<Vec4<f32>>",
            "imageCube": "ImageCube<Vec4<f32>>",
            "image2DArray": "Image2DArray<Vec4<f32>>",
            "image2DMS": "Image2DMS<Vec4<f32>>",
            "image2DMSArray": "Image2DMSArray<Vec4<f32>>",
            "iimage1D": "Image1D<Vec4<i32>>",
            "iimage1DArray": "Image1DArray<Vec4<i32>>",
            "iimage2D": "Image2D<Vec4<i32>>",
            "iimage3D": "Image3D<Vec4<i32>>",
            "iimage2DArray": "Image2DArray<Vec4<i32>>",
            "iimage2DMS": "Image2DMS<Vec4<i32>>",
            "iimage2DMSArray": "Image2DMSArray<Vec4<i32>>",
            "uimage1D": "Image1D<Vec4<u32>>",
            "uimage1DArray": "Image1DArray<Vec4<u32>>",
            "uimage2D": "Image2D<Vec4<u32>>",
            "uimage3D": "Image3D<Vec4<u32>>",
            "uimage2DArray": "Image2DArray<Vec4<u32>>",
            "uimage2DMS": "Image2DMS<Vec4<u32>>",
            "uimage2DMSArray": "Image2DMSArray<Vec4<u32>>",
            "StructuredBuffer": "StructuredBuffer",
            "RWStructuredBuffer": "RWStructuredBuffer",
            "AppendStructuredBuffer": "AppendStructuredBuffer",
            "ConsumeStructuredBuffer": "ConsumeStructuredBuffer",
            "ByteAddressBuffer": "ByteAddressBuffer",
            "RWByteAddressBuffer": "RwByteAddressBuffer",
            "PointStream": "CglRustPointStream",
            "LineStream": "CglRustLineStream",
            "TriangleStream": "CglRustTriangleStream",
            "InputPatch": "CglRustInputPatch",
            "OutputPatch": "CglRustOutputPatch",
            "accelerationStructureEXT": "RayTracingAccelerationStructure",
            "AccelerationStructure": "RayTracingAccelerationStructure",
            "acceleration_structure": "RayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "RayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "RayTracingAccelerationStructure",
            "RayDesc": "RayDesc",
            "RayQuery": "RayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "BuiltInTriangleIntersectionAttributes"
            ),
        }

        self.semantic_map = {
            # Vertex attributes
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_BaseVertex": "start_vertex_location",
            "gl_BaseInstance": "start_instance_location",
            "gl_DrawID": "draw_id",
            "gl_Position": "position",
            "gl_PointSize": "point_size",
            "gl_ClipDistance": "clip_distance",
            "SV_Position": "position",
            "SV_VertexID": "vertex_id",
            "SV_InstanceID": "instance_id",
            "SV_StartVertexLocation": "start_vertex_location",
            "SV_StartInstanceLocation": "start_instance_location",
            "SV_DrawID": "draw_id",
            # Fragment attributes
            "gl_FragColor": "target(0)",
            "gl_FragColor0": "target(0)",
            "gl_FragColor1": "target(1)",
            "gl_FragColor2": "target(2)",
            "gl_FragColor3": "target(3)",
            "gl_FragDepth": "depth(any)",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_PrimitiveID": "primitive_id",
            "gl_SampleID": "sample_index",
            "SV_Target": "target(0)",
            "SV_Target0": "target(0)",
            "SV_Target1": "target(1)",
            "SV_Target2": "target(2)",
            "SV_Target3": "target(3)",
            "SV_Target4": "target(4)",
            "SV_Target5": "target(5)",
            "SV_Target6": "target(6)",
            "SV_Target7": "target(7)",
            "SV_Depth": "depth(any)",
            "SV_IsFrontFace": "front_facing",
            "SV_PrimitiveID": "primitive_id",
            "SV_SampleIndex": "sample_index",
            # Compute attributes
            "gl_WorkGroupID": "workgroup_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "SV_GroupID": "workgroup_id",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_GroupIndex": "local_invocation_index",
            # Tessellation attributes
            "gl_InvocationID": "output_control_point_id",
            "gl_PrimitiveIDIn": "primitive_id",
            "gl_TessCoord": "domain_location",
            "SV_OutputControlPointID": "output_control_point_id",
            "SV_DomainLocation": "domain_location",
            # Ray tracing attributes
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
            # Standard vertex semantics
            "POSITION": "position",
            "NORMAL": "normal",
            "TANGENT": "tangent",
            "BINORMAL": "binormal",
            "TEXCOORD": "texcoord",
            "TEXCOORD0": "texcoord(0)",
            "TEXCOORD1": "texcoord(1)",
            "TEXCOORD2": "texcoord(2)",
            "TEXCOORD3": "texcoord(3)",
            "COLOR": "color",
            "COLOR0": "color(0)",
            "COLOR1": "color(1)",
        }

        # Function mapping for common shader functions
        self.function_map = {
            "texture": "sample",
            "textureLod": "sample_lod",
            "textureLodOffset": "sample_lod_offset",
            "textureGrad": "sample_grad",
            "textureGradOffset": "sample_grad_offset",
            "textureOffset": "sample_offset",
            "textureProj": "sample_projected",
            "textureProjLod": "sample_projected_lod",
            "textureProjGrad": "sample_projected_grad",
            "textureProjOffset": "sample_projected_offset",
            "textureProjLodOffset": "sample_projected_lod_offset",
            "textureProjGradOffset": "sample_projected_grad_offset",
            "texelFetch": "texel_fetch",
            "texelFetchOffset": "texel_fetch_offset",
            "textureQueryLevels": "texture_query_levels",
            "textureQueryLod": "texture_query_lod",
            "textureSamples": "texture_samples",
            "imageLoad": "image_load",
            "imageStore": "image_store",
            "imageSize": "image_size",
            "imageSamples": "image_samples",
            "imageAtomicAdd": "image_atomic_add",
            "imageAtomicMin": "image_atomic_min",
            "imageAtomicMax": "image_atomic_max",
            "imageAtomicAnd": "image_atomic_and",
            "imageAtomicOr": "image_atomic_or",
            "imageAtomicXor": "image_atomic_xor",
            "imageAtomicExchange": "image_atomic_exchange",
            "imageAtomicCompSwap": "image_atomic_comp_swap",
            "TraceRay": "trace_ray",
            "CallShader": "call_shader",
            "ReportHit": "report_hit",
            "IgnoreHit": "ignore_hit",
            "AcceptHitAndEndSearch": "accept_hit_and_end_search",
            "textureCompare": "texture_compare",
            "textureCompareOffset": "texture_compare_offset",
            "textureCompareLod": "texture_compare_lod",
            "textureCompareLodOffset": "texture_compare_lod_offset",
            "textureCompareGrad": "texture_compare_grad",
            "textureCompareGradOffset": "texture_compare_grad_offset",
            "textureCompareProj": "texture_compare_projected",
            "textureCompareProjOffset": "texture_compare_projected_offset",
            "textureCompareProjLod": "texture_compare_projected_lod",
            "textureCompareProjLodOffset": "texture_compare_projected_lod_offset",
            "textureCompareProjGrad": "texture_compare_projected_grad",
            "textureCompareProjGradOffset": "texture_compare_projected_grad_offset",
            "textureGatherCompare": "texture_gather_compare",
            "textureGatherCompareOffset": "texture_gather_compare_offset",
            "normalize": "normalize",
            "dot": "dot",
            "cross": "cross",
            "length": "length",
            "distance": "distance",
            "reflect": "reflect",
            "refract": "refract",
            "faceforward": "faceforward",
            "outerProduct": "outer_product",
            "transpose": "transpose",
            "determinant": "determinant",
            "inverse": "inverse",
            "matrixCompMult": "matrix_comp_mult",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "atan2": "atan2",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
            "degrees": "degrees",
            "radians": "radians",
            "sqrt": "sqrt",
            "inversesqrt": "rsqrt",
            "pow": "pow",
            "trunc": "trunc",
            "roundEven": "round_even",
            "fma": "fma",
            "mad": "fma",
            "ldexp": "ldexp",
            "exp": "exp",
            "exp2": "exp2",
            "log": "log",
            "log2": "log2",
            "abs": "abs",
            "sign": "sign",
            "isnan": "isnan",
            "isinf": "isinf",
            "isfinite": "isfinite",
            "any": "any",
            "all": "all",
            "lessThan": "less_than",
            "lessThanEqual": "less_than_equal",
            "greaterThan": "greater_than",
            "greaterThanEqual": "greater_than_equal",
            "equal": "equal",
            "notEqual": "not_equal",
            "bitCount": "bit_count",
            "bitfieldReverse": "bitfield_reverse",
            "findLSB": "find_lsb",
            "findMSB": "find_msb",
            "bitfieldExtract": "bitfield_extract",
            "bitfieldInsert": "bitfield_insert",
            "floatBitsToInt": "float_bits_to_int",
            "floatBitsToUint": "float_bits_to_uint",
            "intBitsToFloat": "int_bits_to_float",
            "uintBitsToFloat": "uint_bits_to_float",
            "packUnorm2x16": "pack_unorm_2x16",
            "packSnorm2x16": "pack_snorm_2x16",
            "packUnorm4x8": "pack_unorm_4x8",
            "packSnorm4x8": "pack_snorm_4x8",
            "packHalf2x16": "pack_half_2x16",
            "packDouble2x32": "pack_double_2x32",
            "unpackUnorm2x16": "unpack_unorm_2x16",
            "unpackSnorm2x16": "unpack_snorm_2x16",
            "unpackUnorm4x8": "unpack_unorm_4x8",
            "unpackSnorm4x8": "unpack_snorm_4x8",
            "unpackHalf2x16": "unpack_half_2x16",
            "unpackDouble2x32": "unpack_double_2x32",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "mix": "lerp",
            "smoothstep": "smoothstep",
            "step": "step",
            "dFdx": "dfdx",
            "dFdy": "dfdy",
            "ddx": "dfdx",
            "ddy": "dfdy",
            "dFdxFine": "dfdx_fine",
            "dFdyFine": "dfdy_fine",
            "ddx_fine": "dfdx_fine",
            "ddy_fine": "dfdy_fine",
            "dFdxCoarse": "dfdx_coarse",
            "dFdyCoarse": "dfdy_coarse",
            "ddx_coarse": "dfdx_coarse",
            "ddy_coarse": "dfdy_coarse",
            "fwidth": "fwidth",
            "fwidthFine": "fwidth_fine",
            "fwidthCoarse": "fwidth_coarse",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "frac": "fract",
            "fract": "fract",
            "mod": "modulo",
            "barrier": "workgroup_barrier",
            "workgroupBarrier": "workgroup_barrier",
            "memoryBarrier": "memory_barrier",
            "memoryBarrierShared": "memory_barrier_shared",
            "memoryBarrierBuffer": "memory_barrier_buffer",
            "memoryBarrierImage": "memory_barrier_image",
            "groupMemoryBarrier": "group_memory_barrier",
            "allMemoryBarrier": "all_memory_barrier",
            "deviceMemoryBarrier": "device_memory_barrier",
            "GroupMemoryBarrier": "group_memory_barrier",
            "GroupMemoryBarrierWithGroupSync": "group_memory_barrier_with_group_sync",
            "DeviceMemoryBarrier": "device_memory_barrier",
            "DeviceMemoryBarrierWithGroupSync": "device_memory_barrier_with_group_sync",
            "AllMemoryBarrier": "all_memory_barrier",
            "AllMemoryBarrierWithGroupSync": "all_memory_barrier_with_group_sync",
            "WaveShuffle": "subgroup_shuffle",
            "WaveShuffleXor": "subgroup_shuffle_xor",
            "__shfl_down_sync": "subgroup_shuffle_down",
        }
        self.wave_op_map = {
            "WaveActiveSum": "subgroup_add",
            "WaveActiveProduct": "subgroup_mul",
            "WaveActiveMin": "subgroup_min",
            "WaveActiveMax": "subgroup_max",
            "WaveActiveAnyTrue": "subgroup_any",
            "WaveActiveAllTrue": "subgroup_all",
            "WaveActiveAllEqual": "subgroup_all_equal",
            "WaveActiveBallot": "subgroup_ballot",
            "WaveActiveCountBits": "subgroup_ballot_bit_count",
            "WaveActiveBitAnd": "subgroup_and",
            "WaveActiveBitOr": "subgroup_or",
            "WaveActiveBitXor": "subgroup_xor",
            "WaveGetLaneCount": "subgroup_size",
            "WaveGetLaneIndex": "subgroup_invocation_id",
            "WaveIsFirstLane": "subgroup_elect",
            "WaveReadLaneFirst": "subgroup_broadcast_first",
            "WaveReadLaneAt": "subgroup_broadcast",
            "WavePrefixSum": "subgroup_exclusive_add",
            "WavePrefixProduct": "subgroup_exclusive_mul",
            "WavePrefixCountBits": "subgroup_exclusive_ballot_bit_count",
            "WaveMatch": "subgroup_match",
            "WaveMultiPrefixSum": "subgroup_partitioned_add",
            "WaveMultiPrefixProduct": "subgroup_partitioned_mul",
            "WaveMultiPrefixCountBits": "subgroup_partitioned_ballot_bit_count",
            "WaveMultiPrefixBitAnd": "subgroup_partitioned_and",
            "WaveMultiPrefixBitOr": "subgroup_partitioned_or",
            "WaveMultiPrefixBitXor": "subgroup_partitioned_xor",
            "QuadReadLaneAt": "subgroup_quad_broadcast",
            "QuadReadAcrossX": "subgroup_quad_swap_horizontal",
            "QuadReadAcrossY": "subgroup_quad_swap_vertical",
            "QuadReadAcrossDiagonal": "subgroup_quad_swap_diagonal",
            "QuadAny": "subgroup_quad_any",
            "QuadAll": "subgroup_quad_all",
            "WaveShuffle": "subgroup_shuffle",
            "WaveShuffleXor": "subgroup_shuffle_xor",
        }
        self.sync_map = {
            "barrier": "workgroup_barrier",
            "workgroupBarrier": "workgroup_barrier",
            "memoryBarrier": "memory_barrier",
            "memoryBarrierShared": "memory_barrier_shared",
            "memoryBarrierBuffer": "memory_barrier_buffer",
            "memoryBarrierImage": "memory_barrier_image",
            "groupMemoryBarrier": "group_memory_barrier",
            "allMemoryBarrier": "all_memory_barrier",
            "deviceMemoryBarrier": "device_memory_barrier",
            "GroupMemoryBarrier": "group_memory_barrier",
            "GroupMemoryBarrierWithGroupSync": "group_memory_barrier_with_group_sync",
            "DeviceMemoryBarrier": "device_memory_barrier",
            "DeviceMemoryBarrierWithGroupSync": "device_memory_barrier_with_group_sync",
            "AllMemoryBarrier": "all_memory_barrier",
            "AllMemoryBarrierWithGroupSync": "all_memory_barrier_with_group_sync",
        }
        self.variable_types = {}
        self.local_variable_names = set()
        self.lazy_static_names = set()
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.rust_resource_binding_cursors = {}
        self.rust_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.struct_generic_params = {}
        self.static_variable_names = set()
        self.static_symbol_names = {}
        self.runtime_type_collisions = set()
        self.current_return_type = None
        self.current_shader_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.user_function_names = set()
        self.user_function_nodes = {}
        self.user_function_return_types = {}
        self.user_function_param_types = {}
        self.user_function_generic_constraint_cache = {}
        self.active_generic_constraint_functions = set()
        self.rust_function_capture_params = {}
        self.current_generic_param_names = set()
        self.current_function_generic_constraints = {}
        self.trait_methods = {}
        self.enum_variant_names = {}
        self.enum_variant_field_types = {}
        self.enum_variant_payload_type_map = {}
        self.enum_generic_params = {}
        self.non_copy_type_names = set()
        self.current_mutated_names = set()
        self.required_generic_math_traits = set()
        self.required_mesh_helpers = set()
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0
        self.lambda_capture_temp_counter = 0
        self.assignment_lhs_depth = 0
        self.member_object_depth = 0
        self.array_object_depth = 0
        self.return_move_blocked_roots = set()
        self.lambda_capture_alias_stack = []
        self.lambda_capture_place_alias_stack = []
        self.force_move_lambda_depth = 0
        self.nested_lambda_capture_preclone_depth = 0

    def generate(self, ast):
        """Generate complete Rust-like shader source for a CrossGL AST."""
        self.validate_supported_stage_types(ast)
        self.variable_types = {}
        self.local_variable_names = set()
        self.lazy_static_names = set()
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.rust_resource_binding_cursors = {}
        self.rust_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.struct_generic_params = {}
        self.static_variable_names = self.collect_static_variable_names(ast)
        self.static_symbol_names = self.build_static_symbol_names(
            self.static_variable_names
        )
        self.runtime_type_collisions = self.collect_runtime_type_collisions(ast)
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.loop_depth = 0
        self.do_while_counter = 0
        self.user_function_names = self.collect_user_function_names(ast)
        self.user_function_nodes = self.collect_user_function_nodes(ast)
        self.user_function_return_types = self.collect_user_function_return_types(ast)
        self.user_function_param_types = self.collect_user_function_param_types(ast)
        self.user_function_generic_constraint_cache = {}
        self.active_generic_constraint_functions = set()
        self.rust_function_capture_params = {}
        self.current_generic_param_names = set()
        self.current_function_generic_constraints = {}
        self.trait_methods = self.collect_trait_methods(ast)
        self.enum_variant_names = self.collect_enum_variant_names(ast)
        self.enum_variant_field_types = self.collect_enum_variant_field_types(ast)
        self.enum_variant_payload_type_map = self.collect_enum_variant_payload_type_map(
            ast
        )
        self.enum_generic_params = self.collect_enum_generic_params(ast)
        self.non_copy_type_names = self.collect_non_copy_type_names(ast)
        self.register_constant_types(ast)
        self.current_mutated_names = set()
        self.required_generic_math_traits = set()
        self.required_mesh_helpers = set()
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0
        self.lambda_capture_temp_counter = 0
        self.assignment_lhs_depth = 0
        self.member_object_depth = 0
        self.array_object_depth = 0
        self.return_move_blocked_roots = set()
        self.lambda_capture_alias_stack = []
        self.lambda_capture_place_alias_stack = []
        self.force_move_lambda_depth = 0
        self.nested_lambda_capture_preclone_depth = 0
        self.reserve_explicit_rust_resource_bindings(ast)
        code = "// Generated Rust GPU Shader Code\n"
        code += "use gpu::*;\n"
        code += "use math::*;\n\n"
        if self.has_geometry_stage(ast):
            code += self.generate_rust_geometry_stream_helpers()
        if self.has_tessellation_stage(ast):
            code += self.generate_rust_tessellation_patch_helpers()

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                if getattr(node, "is_trait", False):
                    code += self.generate_trait(node)
                else:
                    code += self.generate_struct(node)
            elif isinstance(node, EnumNode):
                code += self.generate_enum(node)

        code += self.generate_constant_declarations(ast)

        emitted_static_names = set()
        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if isinstance(node, ArrayNode):
                code += self.generate_global_array_declaration(node)
                emitted_static_names.add(node.name)
            else:
                code += self.generate_variable_static_declaration(node)
                emitted_static_names.add(node.name)

        cbuffers = self.get_cbuffer_nodes(ast)
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

            if qualifier_name == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier_name == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier_name == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            elif qualifier_name == "geometry":
                code += "// Geometry Shader\n"
                code += self.generate_function(func, shader_type="geometry")
            elif qualifier_name == "tessellation_control":
                code += "// Tessellation Control Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="tessellation_control",
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            elif qualifier_name == "tessellation_evaluation":
                code += "// Tessellation Evaluation Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="tessellation_evaluation",
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            elif qualifier_name in self.rust_mesh_stage_names():
                code += f"// {qualifier_name.title()} Shader\n"
                code += self.generate_function(
                    func,
                    shader_type=qualifier_name,
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            else:
                code += self.generate_function(func)

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            stage_entry_name_counts = self.stage_entry_name_counts(ast.stages)
            for stage_type, stage in ast.stages.items():
                for struct in getattr(stage, "local_structs", []) or []:
                    code += self.generate_struct(struct)

                saved_variable_types = self.variable_types.copy()
                for variable in getattr(stage, "local_variables", []) or []:
                    if self.is_rust_shared_variable(variable):
                        continue
                    self.register_variable_type(
                        variable.name,
                        self.get_variable_type(variable),
                        scope="static",
                    )
                    if variable.name not in emitted_static_names:
                        code += self.generate_variable_static_declaration(variable)
                        emitted_static_names.add(variable.name)

                local_captures = {}
                if hasattr(stage, "entry_point"):
                    stage_name = str(stage_type).split(".")[-1].lower()
                    function_name = self.stage_entry_function_name(
                        stage_name,
                        stage.entry_point,
                        stage_entry_name_counts,
                    )
                    local_captures = self.collect_stage_local_function_captures(stage)
                    self.rust_function_capture_params.update(local_captures)
                    code += f"// {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        function_name=function_name,
                        stage_node=stage,
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(
                            func,
                            extra_params=self.rust_function_capture_params.get(
                                getattr(func, "name", None), []
                            ),
                        )
                for captured_name in local_captures:
                    self.rust_function_capture_params.pop(captured_name, None)

                self.variable_types = saved_variable_types

        code += self.generate_required_rust_mesh_helpers()
        code += self.generate_required_generic_math_traits()
        return code

    def unsupported_stage_types(self):
        return set()

    def has_geometry_stage(self, ast_node, target_stage=None):
        """Return whether the AST contains a geometry stage to emit."""
        if not stage_matches(target_stage, "geometry"):
            return False

        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == "geometry":
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == "geometry"
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_rust_geometry_stream_helpers(self):
        """Emit compile-oriented placeholders for CrossGL geometry streams."""
        return (
            "pub struct CglRustPointStream<T> {\n"
            "    _marker: std::marker::PhantomData<T>,\n"
            "}\n\n"
            "impl<T> Default for CglRustPointStream<T> {\n"
            "    fn default() -> Self {\n"
            "        Self { _marker: std::marker::PhantomData }\n"
            "    }\n"
            "}\n\n"
            "impl<T> CglRustPointStream<T> {\n"
            "    pub fn Append(&self, value: T) {\n"
            "        let _ = value;\n"
            "    }\n\n"
            "    pub fn RestartStrip(&self) {}\n"
            "}\n\n"
            "pub struct CglRustLineStream<T> {\n"
            "    _marker: std::marker::PhantomData<T>,\n"
            "}\n\n"
            "impl<T> Default for CglRustLineStream<T> {\n"
            "    fn default() -> Self {\n"
            "        Self { _marker: std::marker::PhantomData }\n"
            "    }\n"
            "}\n\n"
            "impl<T> CglRustLineStream<T> {\n"
            "    pub fn Append(&self, value: T) {\n"
            "        let _ = value;\n"
            "    }\n\n"
            "    pub fn RestartStrip(&self) {}\n"
            "}\n\n"
            "pub struct CglRustTriangleStream<T> {\n"
            "    _marker: std::marker::PhantomData<T>,\n"
            "}\n\n"
            "impl<T> Default for CglRustTriangleStream<T> {\n"
            "    fn default() -> Self {\n"
            "        Self { _marker: std::marker::PhantomData }\n"
            "    }\n"
            "}\n\n"
            "impl<T> CglRustTriangleStream<T> {\n"
            "    pub fn Append(&self, value: T) {\n"
            "        let _ = value;\n"
            "    }\n\n"
            "    pub fn RestartStrip(&self) {}\n"
            "}\n\n"
        )

    def has_tessellation_stage(self, ast_node, target_stage=None):
        """Return whether the AST contains a tessellation stage to emit."""
        tessellation_stages = {"tessellation_control", "tessellation_evaluation"}
        if not any(
            stage_matches(target_stage, stage_name)
            for stage_name in tessellation_stages
        ):
            return False

        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) in tessellation_stages:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) in tessellation_stages
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_rust_tessellation_patch_helpers(self):
        """Emit compile-oriented placeholders for CrossGL tessellation patches."""
        return (
            "pub struct CglRustInputPatch<T, const N: usize> {\n"
            "    control_points: [T; N],\n"
            "}\n\n"
            "impl<T: Default, const N: usize> Default for "
            "CglRustInputPatch<T, N> {\n"
            "    fn default() -> Self {\n"
            "        Self { control_points: std::array::from_fn(|_| "
            "T::default()) }\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> std::ops::Index<usize> for "
            "CglRustInputPatch<T, N> {\n"
            "    type Output = T;\n\n"
            "    fn index(&self, index: usize) -> &Self::Output {\n"
            "        &self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> std::ops::IndexMut<usize> for "
            "CglRustInputPatch<T, N> {\n"
            "    fn index_mut(&mut self, index: usize) -> &mut Self::Output {\n"
            "        &mut self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "pub struct CglRustOutputPatch<T, const N: usize> {\n"
            "    control_points: [T; N],\n"
            "}\n\n"
            "impl<T: Default, const N: usize> Default for "
            "CglRustOutputPatch<T, N> {\n"
            "    fn default() -> Self {\n"
            "        Self { control_points: std::array::from_fn(|_| "
            "T::default()) }\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> std::ops::Index<usize> for "
            "CglRustOutputPatch<T, N> {\n"
            "    type Output = T;\n\n"
            "    fn index(&self, index: usize) -> &Self::Output {\n"
            "        &self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> std::ops::IndexMut<usize> for "
            "CglRustOutputPatch<T, N> {\n"
            "    fn index_mut(&mut self, index: usize) -> &mut Self::Output {\n"
            "        &mut self.control_points[index]\n"
            "    }\n"
            "}\n\n"
        )

    def rust_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def rust_ray_stage_metadata_name(self, shader_type):
        return {
            "ray_any_hit": "any_hit",
            "ray_callable": "callable",
            "ray_closest_hit": "closest_hit",
            "ray_generation": "ray_generation",
            "ray_intersection": "intersection",
            "ray_miss": "miss",
        }.get(shader_type, shader_type)

    def rust_mesh_stage_names(self):
        return {"amplification", "mesh", "object", "task"}

    def rust_mesh_stage_attribute_arguments(self, func, *attribute_names):
        for attribute_name in attribute_names:
            arguments = self.rust_stage_attribute_arguments(func, attribute_name)
            if arguments:
                return arguments
        return []

    def rust_mesh_stage_attribute_value(self, func, stage_name, *attribute_names):
        arguments = self.rust_mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise ValueError(
                f"Rust {stage_name} stage {display_name} requires exactly one argument"
            )
        return self.attribute_argument_text(arguments[0])

    def rust_mesh_stage_positive_attribute(self, func, stage_name, *attribute_names):
        arguments = self.rust_mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise ValueError(
                f"Rust {stage_name} stage {display_name} requires exactly one argument"
            )
        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                f"Rust {stage_name} stage {display_name} ({value}) must be positive"
            )
        return value_text

    def rust_mesh_stage_numthreads(self, func, stage_name):
        arguments = self.rust_mesh_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return None
        if len(arguments) > 3:
            raise ValueError(
                f"Rust {stage_name} stage numthreads requires at most three arguments"
            )
        values = [self.attribute_argument_text(arg) for arg in arguments]
        for index, argument in enumerate(arguments):
            value = self.literal_int_value(argument)
            if value is not None and value <= 0:
                raise ValueError(
                    f"Rust {stage_name} stage numthreads value {index + 1} "
                    "must be positive"
                )
        while len(values) < 3:
            values.append("1")
        return tuple(values[:3])

    def rust_mesh_stage_layout_local_size(self, stage_node, stage_name):
        if stage_node is None:
            return None
        sizes = {"local_size_x": None, "local_size_y": None, "local_size_z": None}
        for layout in getattr(stage_node, "layout_qualifiers", []) or []:
            for entry in getattr(layout, "entries", []) or []:
                name = getattr(entry, "name", None)
                if name not in sizes:
                    continue
                arguments = (
                    getattr(entry, "arguments", getattr(entry, "args", [])) or []
                )
                if len(arguments) != 1:
                    raise ValueError(
                        f"Rust {stage_name} stage {name} requires exactly one argument"
                    )
                value = self.literal_int_value(arguments[0])
                if value is not None and value <= 0:
                    raise ValueError(f"Rust {stage_name} stage {name} must be positive")
                sizes[name] = self.attribute_argument_text(arguments[0])
        if all(value is None for value in sizes.values()):
            return None
        return (
            sizes["local_size_x"] or "1",
            sizes["local_size_y"] or "1",
            sizes["local_size_z"] or "1",
        )

    def canonical_rust_mesh_output_topology(self, value):
        if value is None:
            return None
        normalized = str(value).strip().strip("\"'").lower()
        return {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
            "triangle_cw": "triangle_cw",
            "triangle_ccw": "triangle_ccw",
        }.get(normalized)

    def rust_mesh_output_topology(self, func, stage_name, required=False):
        value = self.rust_mesh_stage_attribute_value(func, stage_name, "outputtopology")
        if value is None:
            if required:
                raise ValueError(
                    f"Rust {stage_name} stage requires outputtopology attribute"
                )
            return None
        topology = self.canonical_rust_mesh_output_topology(value)
        if topology is None:
            raise ValueError(
                f"Rust {stage_name} stage outputtopology '{value}' must be point, "
                "line, triangle, triangle_cw, or triangle_ccw"
            )
        return topology

    def rust_mesh_parameter_role(self, parameter):
        role_aliases = {
            "mesh_payload": "mesh_payload",
            "meshpayload": "mesh_payload",
            "payload": "mesh_payload",
            "vertices": "vertices",
            "vertex": "vertices",
            "indices": "indices",
            "index": "indices",
            "primitives": "primitives",
            "primitive": "primitives",
        }
        for attr in getattr(parameter, "attributes", []) or []:
            name = getattr(attr, "name", None)
            if name is None:
                continue
            normalized = str(name).strip().lower().replace("-", "_")
            role = role_aliases.get(normalized)
            if role is not None:
                return role
        return None

    def rust_mesh_parameter_base_type(self, param_type):
        if self.is_array_type_name(param_type):
            base_type, _sizes = self.c_array_type_parts(param_type)
            return base_type
        return param_type

    def rust_mesh_param_infos(self, parameters):
        return [
            (parameter, self.function_parameter_type(parameter))
            for parameter in parameters
        ]

    def validate_rust_mesh_parameter_roles(self, stage_name, param_infos):
        seen_roles = {}
        for parameter, param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role is None:
                continue
            seen_roles.setdefault(role, []).append(parameter.name)

            if role == "mesh_payload" and stage_name != "mesh":
                raise ValueError(
                    "Rust mesh payload parameters are only valid on mesh stages"
                )
            if role in {"vertices", "indices", "primitives"}:
                if stage_name != "mesh":
                    raise ValueError(
                        f"Rust mesh output parameter role {role} is only valid "
                        "on mesh stages"
                    )
                if not self.rust_parameter_is_array(parameter):
                    raise ValueError(
                        f"Rust mesh {role} parameter '{parameter.name}' must be "
                        "an array"
                    )

        if len(seen_roles.get("mesh_payload", [])) > 1:
            raise ValueError(
                "Rust mesh stage accepts at most one mesh_payload parameter"
            )

    def validate_rust_mesh_output_parameter_bounds(self, func, stage_name, param_infos):
        if stage_name != "mesh":
            return
        max_vertices = self.literal_int_value(
            self.rust_mesh_stage_positive_attribute(
                func, stage_name, "maxvertices", "max_vertices"
            )
        )
        max_primitives = self.literal_int_value(
            self.rust_mesh_stage_positive_attribute(
                func, stage_name, "maxprimitives", "max_primitives"
            )
        )
        for parameter, _param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role not in {"vertices", "indices", "primitives"}:
                continue
            array_count = self.rust_parameter_array_count(parameter)
            if array_count is None:
                continue
            if (
                role == "vertices"
                and max_vertices is not None
                and array_count > max_vertices
            ):
                raise ValueError(
                    f"Rust mesh vertices parameter '{parameter.name}' count "
                    f"({array_count}) exceeds maxvertices ({max_vertices})"
                )
            if (
                role in {"indices", "primitives"}
                and max_primitives is not None
                and array_count > max_primitives
            ):
                raise ValueError(
                    f"Rust mesh {role} parameter '{parameter.name}' count "
                    f"({array_count}) exceeds maxprimitives ({max_primitives})"
                )

    def walk_rust_mesh_ops(self, node, visited=None):
        if visited is None:
            visited = set()
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, list):
            for item in node:
                yield from self.walk_rust_mesh_ops(item, visited)
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, MeshOpNode):
            yield node
            for argument in getattr(node, "arguments", []) or []:
                yield from self.walk_rust_mesh_ops(argument, visited)
            return

        for attribute_name in (
            "statements",
            "body",
            "expression",
            "value",
            "left",
            "right",
            "condition",
            "then_branch",
            "else_branch",
            "if_body",
            "args",
            "arguments",
        ):
            if hasattr(node, attribute_name):
                yield from self.walk_rust_mesh_ops(
                    getattr(node, attribute_name), visited
                )

    def validate_rust_mesh_op_stage(self, operation, stage_name):
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise ValueError(
                    f"Rust {stage_name or 'function'} stage cannot call "
                    "SetMeshOutputCounts; SetMeshOutputCounts is only valid in "
                    "mesh stages"
                )
            return
        if operation == "DispatchMesh":
            if stage_name not in {"task", "amplification", "object"}:
                raise ValueError(
                    f"Rust {stage_name or 'function'} stage cannot call "
                    "DispatchMesh; DispatchMesh is only valid in task, "
                    "amplification, or object stages"
                )
            return
        raise ValueError(f"Unsupported Rust mesh intrinsic {operation}")

    def rust_mesh_count_argument_is_integer_scalar(self, argument):
        if isinstance(argument, bool) or isinstance(argument, float):
            return False
        if isinstance(argument, int) and not isinstance(argument, bool):
            return True
        if self.literal_int_value(argument) is not None:
            return True
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return True
        return self.normalize_scalar_type(argument_type) in {
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
        }

    def validate_rust_mesh_op_arguments(self, operation, arguments, stage_name):
        if operation == "SetMeshOutputCounts":
            if len(arguments) != 2:
                raise ValueError(
                    "Rust mesh SetMeshOutputCounts requires exactly two arguments"
                )
            labels = ("vertex count", "primitive count")
        elif operation == "DispatchMesh":
            if len(arguments) not in {3, 4}:
                raise ValueError(
                    "Rust mesh DispatchMesh requires three group counts and an "
                    "optional payload argument"
                )
            labels = ("x count", "y count", "z count")
        else:
            return

        for label, argument in zip(labels, arguments):
            if not self.rust_mesh_count_argument_is_integer_scalar(argument):
                raise ValueError(
                    f"Rust {stage_name} {operation} {label} argument must be an "
                    "integer scalar"
                )
            value = self.literal_int_value(argument)
            if value is not None and value < 0:
                raise ValueError(
                    f"Rust {stage_name} {operation} {label} argument must be "
                    "non-negative"
                )

    def validate_rust_mesh_stage(self, func, parameters, shader_type, stage_node=None):
        self.rust_mesh_stage_numthreads(func, shader_type)
        self.rust_mesh_stage_layout_local_size(stage_node, shader_type)
        param_infos = self.rust_mesh_param_infos(parameters)
        if shader_type == "mesh":
            self.rust_mesh_output_topology(func, shader_type, required=True)
            self.rust_mesh_stage_positive_attribute(
                func, shader_type, "maxvertices", "max_vertices"
            )
            self.rust_mesh_stage_positive_attribute(
                func, shader_type, "maxprimitives", "max_primitives"
            )
        self.validate_rust_mesh_parameter_roles(shader_type, param_infos)
        self.validate_rust_mesh_output_parameter_bounds(func, shader_type, param_infos)

        for mesh_op in self.walk_rust_mesh_ops(getattr(func, "body", [])):
            operation = getattr(mesh_op, "operation", None)
            arguments = list(getattr(mesh_op, "arguments", []) or [])
            self.validate_rust_mesh_op_stage(operation, shader_type)
            self.validate_rust_mesh_op_arguments(operation, arguments, shader_type)

    def generate_rust_mesh_parameter_role_comments(self, param_infos):
        descriptions = []
        for parameter, param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role is None:
                continue
            base_type = self.rust_mesh_parameter_base_type(param_type)
            array_count = self.rust_parameter_array_count(parameter)
            suffix = f"[{array_count}]" if array_count is not None else ""
            descriptions.append(f"{role}:{parameter.name}->{base_type}{suffix}")
        if not descriptions:
            return ""
        return f"// CrossGL mesh parameter role: {', '.join(descriptions)}\n"

    def generate_rust_mesh_stage_comments(
        self, func, parameters, shader_type, stage_node=None
    ):
        lines = [f"// CrossGL mesh/task stage: stage={shader_type}\n"]
        numthreads = self.rust_mesh_stage_numthreads(func, shader_type)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {', '.join(numthreads)}\n")
        local_size = self.rust_mesh_stage_layout_local_size(stage_node, shader_type)
        if local_size is not None:
            lines.append(f"// CrossGL mesh/task local size: {', '.join(local_size)}\n")

        if shader_type == "mesh":
            topology = self.rust_mesh_output_topology(func, shader_type, required=True)
            lines.append(f"// CrossGL mesh output topology: {topology}\n")
        max_vertices = self.rust_mesh_stage_positive_attribute(
            func, shader_type, "maxvertices", "max_vertices"
        )
        if max_vertices is not None:
            lines.append(f"// CrossGL mesh max vertices: {max_vertices}\n")
        max_primitives = self.rust_mesh_stage_positive_attribute(
            func, shader_type, "maxprimitives", "max_primitives"
        )
        if max_primitives is not None:
            lines.append(f"// CrossGL mesh max primitives: {max_primitives}\n")

        role_comments = self.generate_rust_mesh_parameter_role_comments(
            self.rust_mesh_param_infos(parameters)
        )
        if role_comments:
            lines.append(role_comments)
        return "".join(lines)

    def generate_rust_mesh_op_expression(self, expr):
        operation = getattr(expr, "operation", "mesh intrinsic")
        arguments = list(getattr(expr, "arguments", []) or [])
        stage_name = self.current_shader_type
        self.validate_rust_mesh_op_stage(operation, stage_name)
        self.validate_rust_mesh_op_arguments(operation, arguments, stage_name)

        args = [self.generate_expression(argument) for argument in arguments]
        if operation == "SetMeshOutputCounts":
            self.required_mesh_helpers.add("set_mesh_output_counts")
            return f"_crossgl_set_mesh_output_counts({', '.join(args)})"
        if operation == "DispatchMesh":
            self.required_mesh_helpers.add("dispatch_mesh")
            if len(args) == 3:
                return f"_crossgl_dispatch_mesh_without_payload({', '.join(args)})"
            return f"_crossgl_dispatch_mesh({', '.join(args)})"

        raise ValueError(f"Unsupported Rust mesh intrinsic {operation}")

    def generate_required_rust_mesh_helpers(self):
        if not self.required_mesh_helpers:
            return ""
        code = "// CrossGL mesh/task placeholders\n"
        if "set_mesh_output_counts" in self.required_mesh_helpers:
            code += (
                "pub fn _crossgl_set_mesh_output_counts("
                "vertices: u32, primitives: u32) {\n"
                "    let _ = (vertices, primitives);\n"
                "}\n\n"
            )
        if "dispatch_mesh" in self.required_mesh_helpers:
            code += (
                "pub fn _crossgl_dispatch_mesh<T>("
                "x: u32, y: u32, z: u32, payload: T) {\n"
                "    let _ = (x, y, z, payload);\n"
                "}\n\n"
                "pub fn _crossgl_dispatch_mesh_without_payload("
                "x: u32, y: u32, z: u32) {\n"
                "    let _ = (x, y, z);\n"
                "}\n\n"
            )
        return code

    def rust_stage_attribute_arguments(self, func, attribute_name):
        """Return arguments for a Rust-relevant stage control attribute."""
        expected = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == expected:
                return self.attribute_arguments(attr)
        return []

    def rust_geometry_maxvertexcount(self, func):
        arguments = self.rust_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise ValueError("Rust geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise ValueError(
                "Rust geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                f"Rust geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def rust_parameter_qualifiers(self, parameter):
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
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            primitive = self.attribute_argument_text(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def rust_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.rust_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def rust_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if getattr(raw_type, "__class__", None).__name__ == "ArrayType":
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def rust_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if getattr(raw_type, "__class__", None).__name__ == "ArrayType":
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            return self.literal_int_value(size)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, sizes = self.c_array_type_parts(type_name)
        return sizes[0] if sizes else None

    def validate_rust_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.rust_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.rust_parameter_is_array(parameter):
                raise ValueError(
                    "Rust geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.rust_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
                    "Rust geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_rust_geometry_stage(self, func, parameters):
        self.rust_geometry_maxvertexcount(func)

        if not any(
            self.rust_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise ValueError(
                "Rust geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.rust_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise ValueError(
                "Rust geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_rust_geometry_input_primitive_arity(parameters)

    def generate_rust_geometry_stage_comments(self, func, parameters):
        maxvertexcount = self.rust_geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.rust_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.rust_geometry_stream_info(
                self.parameter_raw_type(parameter)
            )
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"// CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                "// CrossGL geometry input primitive: "
                f"{', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                f"// CrossGL geometry output stream: {', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def rust_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, args_text = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None

        generic_args = self.split_top_level_generic_args(args_text[:-1])
        if len(generic_args) != 1:
            return None
        output_type = generic_args[0].strip()
        if not output_type:
            return None
        return base_name, self.map_type(output_type)

    def rust_geometry_stream_mapped_type(self, type_name):
        stream_info = self.rust_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglRust{stream_name}<{output_type}>"

    def rust_stage_entry_function_name(self, stage_name, func):
        """Return the deterministic Rust entry name for special stages."""
        name = getattr(func, "name", None)
        if (
            normalize_stage_name(stage_name)
            in {
                "amplification",
                "mesh",
                "object",
                "task",
                "tessellation_control",
                "tessellation_evaluation",
            }
            and name == "main"
        ):
            return f"{normalize_stage_name(stage_name)}_{name}"
        return name

    def rust_tessellation_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.rust_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"Rust {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_argument_text(arguments[0])

    def rust_tessellation_output_control_points(self, func):
        attribute_name = "outputcontrolpoints"
        arguments = self.rust_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            arguments = self.rust_stage_attribute_arguments(func, "vertices")
        if not arguments:
            raise ValueError(
                "Rust tessellation_control stage requires outputcontrolpoints attribute"
            )
        if len(arguments) != 1:
            raise ValueError(
                "Rust tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )

        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is None:
            raise ValueError(
                "Rust tessellation_control stage outputcontrolpoints "
                f"'{value_text}' must be an integer literal"
            )
        if value <= 0:
            raise ValueError(
                "Rust tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        if value > 32:
            raise ValueError(
                "Rust tessellation_control stage outputcontrolpoints "
                f"({value}) must be at most 32"
            )
        return value_text, value

    def canonical_rust_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().strip("\"'").lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def rust_tessellation_domain(self, func, stage_name, required=False):
        value = self.rust_tessellation_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise ValueError(f"Rust {stage_name} stage requires domain attribute")
            return None

        domain = self.canonical_rust_tessellation_domain(value)
        if domain is None:
            raise ValueError(
                f"Rust {stage_name} stage domain '{value}' must be one of: "
                "isoline, quad, tri"
            )
        return domain

    def rust_tessellation_partitioning(self, func, stage_name, required=False):
        value = self.rust_tessellation_stage_attribute_value(
            func, "partitioning", stage_name
        )
        if value is None:
            if required:
                raise ValueError(
                    f"Rust {stage_name} stage requires partitioning attribute"
                )
            return None

        normalized = str(value).strip().strip("\"'").lower()
        partitioning = {
            "integer": "integer",
            "equal": "integer",
            "equal_spacing": "integer",
            "fractional_even": "fractional_even",
            "fractional_even_spacing": "fractional_even",
            "fractional_odd": "fractional_odd",
            "fractional_odd_spacing": "fractional_odd",
            "pow2": "pow2",
        }.get(normalized)
        if partitioning is None:
            raise ValueError(
                f"Rust {stage_name} stage partitioning '{value}' must be one of: "
                "fractional_even, fractional_odd, integer, pow2"
            )
        return partitioning

    def rust_tessellation_patch_info(self, type_name):
        base_name, args = self.generic_type_parts(type_name)
        patch_kinds = {
            "InputPatch": "InputPatch",
            "OutputPatch": "OutputPatch",
            "CglRustInputPatch": "InputPatch",
            "CglRustOutputPatch": "OutputPatch",
        }
        kind = patch_kinds.get(base_name)
        if kind is None:
            return None

        info = {
            "kind": kind,
            "valid": len(args) == 2,
            "element_type": None,
            "mapped_element_type": None,
            "count_text": None,
            "count_value": None,
        }
        if len(args) >= 1:
            element_type = args[0].strip()
            if element_type:
                info["element_type"] = element_type
                info["mapped_element_type"] = self.map_type(element_type)
        if len(args) >= 2:
            count_text = args[1].strip()
            info["count_text"] = count_text
            info["count_value"] = self.literal_int_value(count_text)
        return info

    def rust_tessellation_patch_parameters(self, parameters, patch_type):
        for parameter in parameters:
            info = self.rust_tessellation_patch_info(self.parameter_raw_type(parameter))
            if info is not None and info["kind"] == patch_type:
                yield parameter, info

    def validate_rust_tessellation_patch_parameter_signature(
        self, parameter, info, stage_name
    ):
        patch_type = info["kind"]
        parameter_name = getattr(parameter, "name", None)
        name_clause = (
            f" parameter '{parameter_name}'" if parameter_name else " parameter"
        )
        type_name = self.type_name_string(self.parameter_raw_type(parameter))

        if not info["valid"]:
            raise ValueError(
                f"Rust {stage_name} stage {patch_type}{name_clause} must use "
                f"{patch_type}<T, N>; found '{type_name or patch_type}'"
            )
        if info["element_type"] is None:
            raise ValueError(
                f"Rust {stage_name} stage {patch_type}{name_clause} must include "
                "an element type"
            )

        control_points = info["count_value"]
        if control_points is None:
            raise ValueError(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                f"point count '{info['count_text']}' must be an integer literal"
            )
        if control_points <= 0:
            raise ValueError(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                f"point count ({control_points}) must be positive"
            )
        if control_points > 32:
            raise ValueError(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                "point count "
                f"({control_points}) must be at most 32"
            )

    def validate_rust_tessellation_patch_parameter_signatures(
        self, parameters, patch_type, stage_name
    ):
        for parameter, info in self.rust_tessellation_patch_parameters(
            parameters, patch_type
        ):
            self.validate_rust_tessellation_patch_parameter_signature(
                parameter,
                info,
                stage_name,
            )

    def validate_rust_tessellation_control_stage(self, func, parameters):
        stage_name = "tessellation_control"
        self.rust_tessellation_domain(func, stage_name, required=True)
        self.rust_tessellation_partitioning(func, stage_name, required=True)
        self.rust_tessellation_output_control_points(func)

        input_patch_parameters = list(
            self.rust_tessellation_patch_parameters(parameters, "InputPatch")
        )
        if not input_patch_parameters:
            raise ValueError(
                "Rust tessellation_control stage parameters must include an "
                "InputPatch<T, N> parameter"
            )
        self.validate_rust_tessellation_patch_parameter_signatures(
            parameters,
            "InputPatch",
            stage_name,
        )

    def validate_rust_tessellation_evaluation_stage(self, func, parameters):
        stage_name = "tessellation_evaluation"
        self.rust_tessellation_domain(func, stage_name, required=True)
        self.rust_tessellation_partitioning(func, stage_name)

        output_patch_parameters = list(
            self.rust_tessellation_patch_parameters(parameters, "OutputPatch")
        )
        if not output_patch_parameters:
            raise ValueError(
                "Rust tessellation_evaluation stage parameters must include an "
                "OutputPatch<T, N> parameter"
            )
        self.validate_rust_tessellation_patch_parameter_signatures(
            parameters,
            "OutputPatch",
            stage_name,
        )

    def validate_rust_tessellation_stage(self, func, parameters, shader_type):
        if shader_type == "tessellation_control":
            self.validate_rust_tessellation_control_stage(func, parameters)
        elif shader_type == "tessellation_evaluation":
            self.validate_rust_tessellation_evaluation_stage(func, parameters)

    def generate_rust_tessellation_stage_comments(self, func, parameters, shader_type):
        metadata = []
        domain = self.rust_tessellation_domain(
            func,
            shader_type,
            required=shader_type == "tessellation_evaluation",
        )
        if domain is not None:
            metadata.append(f"domain={domain}")
        partitioning = self.rust_tessellation_partitioning(func, shader_type)
        if partitioning is not None:
            metadata.append(f"partitioning={partitioning}")
        output_topology = self.rust_tessellation_stage_attribute_value(
            func,
            "outputtopology",
            shader_type,
        )
        if output_topology is not None:
            metadata.append(f"outputtopology={output_topology}")
        if shader_type == "tessellation_control":
            output_control_points, _value = (
                self.rust_tessellation_output_control_points(func)
            )
            metadata.append(f"outputcontrolpoints={output_control_points}")
        patch_constant_function = self.rust_tessellation_stage_attribute_value(
            func,
            "patchconstantfunc",
            shader_type,
        )
        if patch_constant_function is not None:
            metadata.append(f"patchconstantfunc={patch_constant_function}")

        lines = [
            f"// CrossGL {shader_type} stage"
            f"{': ' + ', '.join(metadata) if metadata else ''}\n"
        ]

        patch_descriptions = []
        for parameter in parameters:
            info = self.rust_tessellation_patch_info(self.parameter_raw_type(parameter))
            if info is None or not info["valid"]:
                continue
            patch_descriptions.append(
                f"{info['kind']}:{parameter.name}->{info['mapped_element_type']}"
                f"[{info['count_text']}]"
            )
        if patch_descriptions:
            lines.append(
                "// CrossGL tessellation patch parameters: "
                f"{', '.join(patch_descriptions)}\n"
            )
        return "".join(lines)

    def rust_tessellation_patch_mapped_type(self, type_name):
        info = self.rust_tessellation_patch_info(type_name)
        if info is None or not info["valid"]:
            return None
        helper_name = {
            "InputPatch": "CglRustInputPatch",
            "OutputPatch": "CglRustOutputPatch",
        }[info["kind"]]
        return f"{helper_name}<{info['mapped_element_type']}, {info['count_text']}>"

    def validate_supported_stage_types(self, ast, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise ValueError(
                f"Rust output does not support stage type(s): {stage_list}"
            )

    def stage_entry_name_counts(self, stages):
        """Count entry-point names across shader stages."""
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        """Avoid duplicate Rust symbols when multiple stages use ``main``."""
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if (
            normalize_stage_name(stage_name)
            in {
                "amplification",
                "mesh",
                "object",
                "task",
                "tessellation_control",
                "tessellation_evaluation",
            }
            and name == "main"
        ):
            return f"{normalize_stage_name(stage_name)}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_function_nodes(self, node):
        functions = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode) and current.name:
                functions[current.name] = current
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return functions

    def collect_static_variable_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            for variable in getattr(current, "global_variables", []):
                name = getattr(variable, "name", None)
                if name:
                    names.add(name)
            for variable in getattr(current, "local_variables", []):
                if self.is_rust_shared_variable(variable):
                    continue
                name = getattr(variable, "name", None)
                if name:
                    names.add(name)
            for constant in getattr(current, "constants", []):
                name = getattr(constant, "name", None)
                if name:
                    names.add(name)
            for cbuffer in self.get_cbuffer_nodes(current):
                for member in getattr(cbuffer, "members", []) or []:
                    name = getattr(member, "name", None)
                    if name:
                        names.add(name)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return names

    def build_static_symbol_names(self, names):
        """Map source static names to Rust-style uppercase static symbols."""
        symbol_names = {}
        used_symbols = set()
        for name in sorted(names):
            symbol = self.uppercase_static_symbol_name(name)
            base_symbol = symbol
            suffix = 2
            while symbol in used_symbols:
                symbol = f"{base_symbol}_{suffix}"
                suffix += 1
            symbol_names[name] = symbol
            used_symbols.add(symbol)
        return symbol_names

    def uppercase_static_symbol_name(self, name):
        """Convert a CrossGL identifier to a Rust static identifier."""
        name = str(name)
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        words = re.sub(r"[^0-9A-Za-z_]+", "_", words)
        words = re.sub(r"_+", "_", words).strip("_")
        symbol = words.upper() or "STATIC"
        if symbol[0].isdigit():
            symbol = f"STATIC_{symbol}"
        return symbol

    def collect_runtime_type_collisions(self, node):
        """Find user declarations that shadow Rust math prelude type names."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, (StructNode, EnumNode)):
                name = getattr(current, "name", None)
                if name in self.runtime_math_type_names():
                    names.add(name)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return names

    def runtime_math_type_names(self):
        return {
            "Vec2",
            "Vec3",
            "Vec4",
            "Mat2",
            "Mat3",
            "Mat4",
            "Mat2x3",
            "Mat2x4",
            "Mat3x2",
            "Mat3x4",
            "Mat4x2",
            "Mat4x3",
        }

    def collect_trait_methods(self, node):
        traits = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode) and getattr(current, "is_trait", False):
                traits[current.name] = {
                    method.name
                    for method in getattr(current, "members", []) or []
                    if isinstance(method, FunctionNode)
                }
            for struct in getattr(current, "structs", []):
                collect(struct)
            for function in getattr(current, "functions", []):
                collect(function)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return traits

    def collect_enum_variant_names(self, node):
        """Collect enum variant names keyed by the Rust enum type name."""
        variants = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variants[name] = [
                variant.name
                for variant in getattr(enum_node, "variants", []) or []
                if getattr(variant, "name", None)
            ]

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return variants

    def collect_enum_variant_field_types(self, node):
        """Collect named payload fields for struct-like enum variants."""
        field_types = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variant_fields = {}
            for variant in getattr(enum_node, "variants", []) or []:
                data = self.enum_variant_data(variant)
                if not data or not all(
                    isinstance(item, tuple) and len(item) == 2 for item in data
                ):
                    continue
                variant_fields[variant.name] = {
                    field_name: self.convert_type_node_to_string(field_type)
                    for field_name, field_type in data
                }
            if variant_fields:
                field_types[name] = variant_fields

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return field_types

    def collect_enum_variant_payload_type_map(self, node):
        """Collect positional payload types for enum variant constructor calls."""
        payload_types = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variant_payloads = {}
            for variant in getattr(enum_node, "variants", []) or []:
                variant_payloads[variant.name] = [
                    self.convert_type_node_to_string(field_type)
                    for field_type in self.enum_variant_payload_types(variant)
                ]
            payload_types[name] = variant_payloads

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return payload_types

    def collect_enum_generic_params(self, node):
        """Collect generic parameter names for enums and enum-wrapper structs."""
        generic_params = {}

        def add_enum(name, generic_owner):
            params = self.generic_param_names(generic_owner)
            if name and params:
                generic_params[name] = params

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, current)
                    add_enum(wrapper_enum.name, current)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return generic_params

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                if current.name and return_type is not None:
                    return_types[current.name] = self.convert_type_node_to_string(
                        return_type
                    )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return return_types

    def collect_user_function_param_types(self, node):
        param_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode) and current.name:
                params = getattr(current, "parameters", getattr(current, "params", []))
                param_types[current.name] = [
                    self.function_parameter_type(param) for param in params
                ]
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return param_types

    def collect_stage_local_function_captures(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        entry_params = getattr(
            entry_point, "parameters", getattr(entry_point, "params", [])
        )
        entry_param_types = {
            param.name: self.function_parameter_type(param) for param in entry_params
        }
        if not entry_param_types:
            return {}

        captures = {}
        for function in getattr(stage, "local_functions", []) or []:
            if self.is_function_prototype(function):
                continue
            used_names = self.collect_free_identifier_names(function)
            captured_params = [
                ParameterNode(name, entry_param_types[name])
                for name in entry_param_types
                if name in used_names
            ]
            if captured_params:
                captures[function.name] = captured_params
        return captures

    def collect_free_identifier_names(self, function):
        bound_names = {
            param.name
            for param in getattr(
                function, "parameters", getattr(function, "params", [])
            )
        }
        used_names = set()

        def visit(node, bound):
            if node is None:
                return
            if isinstance(node, list):
                for item in node:
                    visit(item, bound)
                return
            if isinstance(node, IdentifierNode):
                if node.name not in bound:
                    used_names.add(node.name)
                return
            if isinstance(node, VariableNode):
                visit(getattr(node, "initial_value", None), bound)
                bound.add(node.name)
                return
            if isinstance(node, ArrayNode):
                visit(
                    getattr(node, "initial_value", getattr(node, "value", None)), bound
                )
                bound.add(node.name)
                return
            if isinstance(node, FunctionCallNode):
                for arg in getattr(node, "arguments", getattr(node, "args", [])) or []:
                    visit(arg, bound)
                return
            if isinstance(node, MemberAccessNode):
                visit(
                    getattr(node, "object_expr", getattr(node, "object", None)), bound
                )
                return
            if isinstance(node, AssignmentNode):
                visit(getattr(node, "target", getattr(node, "left", None)), bound)
                visit(getattr(node, "value", getattr(node, "right", None)), bound)
                return
            if isinstance(
                node, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode, IfNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)
                return
            if isinstance(node, ReturnNode):
                visit(getattr(node, "value", None), bound)
                return
            if isinstance(node, ExpressionStatementNode):
                visit(getattr(node, "expression", None), bound)
                return
            if isinstance(
                node, (BinaryOpNode, TernaryOpNode, UnaryOpNode, ArrayAccessNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)
                return
            if isinstance(
                node, (ConstructorNode, ArrayLiteralNode, BlockNode, MatchNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)

        visit(getattr(function, "body", None), set(bound_names))
        return used_names

    def derive_traits_for_members(self, members, include_default=False):
        traits = ["Debug", "Clone"]
        if self.members_are_copy_derivable(members):
            traits.append("Copy")
        if include_default and self.members_default_derive_supported(members):
            traits.append("Default")
        return ", ".join(traits)

    def members_default_derive_supported(self, members):
        return all(
            self.member_default_derive_supported(member)
            for member in members
            if not isinstance(member, EnumNode)
        )

    def member_default_derive_supported(self, member):
        member_type = self.member_type_for_copy_check(member)
        if member_type is None:
            return True
        return self.rust_type_default_derive_supported(self.map_type(member_type))

    def rust_type_default_derive_supported(self, rust_type):
        array_type = self.parse_rust_array_type(str(rust_type))
        if array_type is None:
            return True
        element_type, size = array_type
        if size is not None:
            try:
                if int(size) > 32:
                    return False
            except ValueError:
                return False
        return self.rust_type_default_derive_supported(element_type)

    def derive_traits_for_enum(self, node, generic_owner=None, include_default=False):
        traits = ["Debug", "Clone"]
        generic_names = set(self.generic_param_names(generic_owner or node))
        if self.enum_payloads_are_copy_derivable(node, generic_names):
            traits.append("Copy")
        if include_default:
            traits.append("Default")
        return ", ".join(traits)

    def collect_non_copy_type_names(self, node):
        declarations = []

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            if isinstance(current, EnumNode):
                declarations.append((current.name, "enum", current, current))
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    declarations.append((current.name, "enum", wrapper_enum, current))
                    declarations.append(
                        (wrapper_enum.name, "enum", wrapper_enum, current)
                    )
                else:
                    declarations.append((current.name, "struct", current, current))
                    for member in getattr(current, "members", []) or []:
                        if isinstance(member, EnumNode):
                            collect(member)

            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        for cbuffer in self.get_cbuffer_nodes(node):
            collect(cbuffer)

        non_copy_type_names = set()
        changed = True
        while changed:
            changed = False
            for name, declaration_kind, declaration, generic_owner in declarations:
                if not name or name in non_copy_type_names:
                    continue
                generic_names = set(self.generic_param_names(generic_owner))
                if declaration_kind == "enum":
                    copy_derivable = self.enum_payloads_are_copy_derivable(
                        declaration,
                        generic_names,
                        non_copy_type_names,
                    )
                else:
                    copy_derivable = self.members_are_copy_derivable(
                        getattr(declaration, "members", []),
                        generic_names,
                        non_copy_type_names,
                    )
                if not copy_derivable:
                    non_copy_type_names.add(name)
                    changed = True

        return non_copy_type_names

    def members_are_copy_derivable(
        self,
        members,
        generic_names=None,
        non_copy_type_names=None,
    ):
        generic_names = generic_names or set()
        for member in members:
            if isinstance(member, EnumNode):
                continue
            member_type = self.member_type_for_copy_check(member)
            if member_type is None:
                continue
            if not self.type_is_copy_derivable(
                member_type,
                generic_names,
                non_copy_type_names,
            ):
                return False
        return True

    def member_type_for_copy_check(self, member):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", None)
            )
            element_type = self.convert_type_node_to_string(element_type)
            size = self.format_array_size(getattr(member, "size", None))
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )

        if not hasattr(member, "member_type") and not hasattr(member, "vtype"):
            return None
        return self.get_member_type(member)

    def enum_payloads_are_copy_derivable(
        self,
        node,
        generic_names,
        non_copy_type_names=None,
    ):
        for variant in getattr(node, "variants", []) or []:
            for field_type in self.enum_variant_payload_types(variant):
                if not self.type_is_copy_derivable(
                    field_type,
                    generic_names,
                    non_copy_type_names,
                ):
                    return False
        return True

    def enum_variant_payload_types(self, variant):
        field_types = []
        for item in self.enum_variant_data(variant):
            if isinstance(item, tuple) and len(item) == 2:
                field_types.append(item[1])
            else:
                field_types.append(item)
        return field_types

    def type_is_copy_derivable(
        self,
        type_name,
        generic_names=None,
        non_copy_type_names=None,
    ):
        generic_names = generic_names or set()
        non_copy_type_names = non_copy_type_names or self.non_copy_type_names
        type_name = self.convert_type_node_to_string(type_name)

        if type_name in generic_names:
            return True

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            return all(
                size is not None for size in sizes
            ) and self.type_is_copy_derivable(
                base_type,
                generic_names,
                non_copy_type_names,
            )

        rust_type = self.unqualify_runtime_type_name(self.map_type(type_name))
        if rust_type.startswith("Vec<"):
            return False

        base_type, generic_args = self.generic_type_parts(type_name)
        if base_type in non_copy_type_names:
            return False
        if generic_args:
            return all(
                self.type_is_copy_derivable(
                    generic_arg,
                    generic_names,
                    non_copy_type_names,
                )
                for generic_arg in generic_args
            )

        return True

    def members_have_unsized_arrays(self, members):
        return any(self.member_has_unsized_array(member) for member in members)

    def member_has_unsized_array(self, member):
        if isinstance(member, ArrayNode):
            return getattr(member, "size", None) is None

        member_type = getattr(member, "member_type", None)
        if member_type is not None and member_type.__class__.__name__ == "ArrayType":
            return getattr(member_type, "size", None) is None

        vtype = getattr(member, "vtype", None)
        return isinstance(vtype, str) and vtype.endswith("[]")

    def generate_trait(self, node):
        """Render a simple CrossGL trait declaration as a Rust trait."""
        code = f"pub trait {node.name}{self.format_generic_params(node)}: Sized {{\n"
        for method in getattr(node, "members", []) or []:
            if isinstance(method, FunctionNode):
                code += self.generate_trait_method(method)
        code += "}\n\n"
        if self.is_auto_numeric_trait(node):
            code += self.generate_numeric_trait_primitive_impls(node.name)
        return code

    def is_auto_numeric_trait(self, node):
        """Detect the canonical value-style Numeric trait used by generic shaders."""
        if getattr(node, "name", None) != "Numeric":
            return False

        methods = {
            method.name: method
            for method in getattr(node, "members", []) or []
            if isinstance(method, FunctionNode)
        }
        if set(methods) != {"add", "mul", "zero", "one"}:
            return False

        return (
            self.is_binary_self_method(methods["add"])
            and self.is_binary_self_method(methods["mul"])
            and self.is_nullary_self_method(methods["zero"])
            and self.is_nullary_self_method(methods["one"])
        )

    def is_binary_self_method(self, method):
        params = getattr(method, "parameters", getattr(method, "params", []))
        return (
            len(params) == 2
            and params[0].name == "self"
            and self.function_parameter_type(params[0]) == "Self"
            and self.function_parameter_type(params[1]) == "Self"
            and self.convert_type_node_to_string(method.return_type) == "Self"
        )

    def is_nullary_self_method(self, method):
        params = getattr(method, "parameters", getattr(method, "params", []))
        return (
            len(params) == 0
            and self.convert_type_node_to_string(method.return_type) == "Self"
        )

    def generate_numeric_trait_primitive_impls(self, trait_name):
        """Provide primitive Rust implementations for the canonical Numeric trait."""
        primitive_types = ["f32", "f64", "i32", "u32", "i16", "u16", "i64", "u64"]
        code = ""
        for rust_type in primitive_types:
            one = "1.0" if rust_type in {"f32", "f64"} else "1"
            zero = "0.0" if rust_type in {"f32", "f64"} else "0"
            code += f"impl {trait_name} for {rust_type} {{\n"
            code += "    fn add(self, other: Self) -> Self {\n"
            code += "        self + other\n"
            code += "    }\n"
            code += "    fn mul(self, other: Self) -> Self {\n"
            code += "        self * other\n"
            code += "    }\n"
            code += "    fn zero() -> Self {\n"
            code += f"        {zero}\n"
            code += "    }\n"
            code += "    fn one() -> Self {\n"
            code += f"        {one}\n"
            code += "    }\n"
            code += "}\n\n"
        return code

    def generate_required_generic_math_traits(self):
        code = ""
        if "CglSqrt" in self.required_generic_math_traits:
            code += self.generate_cgl_sqrt_trait()
        return code

    def generate_cgl_sqrt_trait(self):
        code = "pub trait CglSqrt {\n"
        code += "    fn cgl_sqrt(self) -> Self;\n"
        code += "}\n\n"
        for rust_type in ["f32", "f64"]:
            code += f"impl CglSqrt for {rust_type} {{\n"
            code += "    fn cgl_sqrt(self) -> Self {\n"
            code += "        self.sqrt()\n"
            code += "    }\n"
            code += "}\n\n"
        for rust_type in ["i32", "u32", "i16", "u16", "i64", "u64"]:
            code += f"impl CglSqrt for {rust_type} {{\n"
            code += "    fn cgl_sqrt(self) -> Self {\n"
            code += f"        (self as f64).sqrt() as {rust_type}\n"
            code += "    }\n"
            code += "}\n\n"
        return code

    def generate_trait_method(self, method):
        param_list = getattr(method, "parameters", getattr(method, "params", []))
        params = []
        for param in param_list:
            param_type = self.function_parameter_type(param)
            if param.name == "self" and param_type == "Self":
                params.append("self")
            else:
                params.append(f"{param.name}: {self.map_type(param_type)}")

        return_type = getattr(method, "return_type", None)
        if return_type is not None:
            return_type = self.convert_type_node_to_string(return_type)
        else:
            return_type = "void"

        generic_params = self.format_generic_params(method)
        params_str = ", ".join(params)
        return (
            f"    fn {method.name}{generic_params}({params_str}) "
            f"-> {self.map_type(return_type)};\n"
        )

    def generate_struct(self, node):
        wrapper_enum = self.struct_enum_wrapper(node)
        if wrapper_enum is not None:
            self.struct_member_types[node.name] = {}
            self.struct_member_semantics[node.name] = {}
            self.struct_generic_params[node.name] = self.generic_param_names(node)
            return self.generate_enum(
                wrapper_enum,
                enum_name=node.name,
                generic_owner=node,
                derive_traits=self.derive_traits_for_enum(
                    wrapper_enum,
                    generic_owner=node,
                    include_default=self.enum_has_unit_variant(wrapper_enum),
                ),
                default_variant_name=self.enum_default_variant_name(wrapper_enum),
            )

        members = getattr(node, "members", [])
        derive_traits = self.derive_traits_for_members(members, include_default=True)
        code = f"#[repr(C)]\n#[derive({derive_traits})]\n"
        code += f"pub struct {node.name}{self.format_generic_params(node)} {{\n"
        member_types = {}
        member_semantics = {}

        for member in members:
            if isinstance(member, ArrayNode):
                member_name = self.rust_identifier(member.name)
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                member_types[member.name] = (
                    f"{self.convert_type_node_to_string(element_type)}[{member.size}]"
                    if member.size
                    else f"{self.convert_type_node_to_string(element_type)}[]"
                )
                if member.size:
                    code += (
                        f"    pub {member_name}: "
                        f"[{self.map_type_to_rust(element_type)}; {member.size}],\n"
                    )
                else:
                    code += (
                        f"    pub {member_name}: "
                        f"Vec<{self.map_type_to_rust(element_type)}>,\n"
                    )
            else:
                if not hasattr(member, "member_type") and not hasattr(member, "vtype"):
                    continue

                if hasattr(member, "member_type"):
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_types[member.name] = member_type

                semantic = self.node_semantic(member)
                if semantic:
                    member_semantics[member.name] = semantic
                    self.validate_rust_builtin_semantic_type(
                        semantic, member_type, "struct member semantic"
                    )
                semantic_comment = (
                    f"  // {self.map_semantic(semantic)}" if semantic else ""
                )
                member_name = self.rust_identifier(member.name)
                code += (
                    f"    pub {member_name}: "
                    f"{self.map_type(member_type)},{semantic_comment}\n"
                )

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.struct_generic_params[node.name] = self.generic_param_names(node)
        code += "}\n\n"
        code += self.generate_struct_constructor_impl(node, member_types)
        if not self.members_default_derive_supported(members):
            code += self.generate_struct_default_impl(node, member_types)
        return code

    def generate_struct_constructor_impl(self, node, member_types):
        """Emit a positional constructor for generated Rust structs."""
        if not member_types:
            return ""

        generic_decl = self.format_generic_params(node)
        generic_use = self.format_generic_argument_params(node)
        params = []
        fields = []
        for name, member_type in member_types.items():
            param_name = self.struct_constructor_param_name(name)
            field_ident = self.rust_identifier(name)
            param_ident = self.rust_identifier(param_name)
            params.append(f"{param_ident}: {self.map_type(member_type)}")
            if param_name == name:
                fields.append(field_ident)
            else:
                fields.append(f"{field_ident}: {param_ident}")

        code = f"impl{generic_decl} {node.name}{generic_use} {{\n"
        code += f"    pub fn new({', '.join(params)}) -> Self {{\n"
        code += f"        Self {{ {', '.join(fields)} }}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def generate_struct_default_impl(self, node, member_types):
        generic_decl = self.format_generic_params(node)
        generic_use = self.format_generic_argument_params(node)
        field_initializers = []
        for name, member_type in member_types.items():
            field_initializers.append(
                f"{self.rust_identifier(name)}: "
                f"{self.rust_runtime_default_initializer(member_type)}"
            )

        code = f"impl{generic_decl} Default for {node.name}{generic_use} {{\n"
        code += "    fn default() -> Self {\n"
        code += f"        Self {{ {', '.join(field_initializers)} }}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def struct_constructor_param_name(self, field_name):
        if field_name in self.static_variable_names:
            return f"{field_name}_value"
        return field_name

    def struct_enum_wrapper(self, node):
        """Return a nested enum when a legacy wrapper struct should lower as enum."""
        members = getattr(node, "members", []) or []
        enum_members = [member for member in members if isinstance(member, EnumNode)]
        data_members = [
            member for member in members if not isinstance(member, EnumNode)
        ]

        if len(enum_members) != 1 or len(data_members) != 1:
            return None

        variant_member = data_members[0]
        if getattr(variant_member, "name", None) != "variant":
            return None

        variant_type = self.get_member_type(variant_member)
        if variant_type != enum_members[0].name:
            return None

        return enum_members[0]

    def enum_has_unit_variant(self, node):
        return self.enum_default_variant_name(node) is not None

    def enum_default_variant_name(self, node):
        for variant in getattr(node, "variants", []) or []:
            if self.enum_variant_data(variant):
                continue
            if getattr(variant, "value", None) is not None:
                continue
            return variant.name
        return None

    def generate_enum(
        self,
        node,
        enum_name=None,
        generic_owner=None,
        derive_traits=None,
        default_variant_name=None,
    ):
        """Render a CrossGL enum declaration as a Rust enum."""
        if derive_traits is None:
            default_variant_name = self.enum_default_variant_name(node)
            derive_traits = self.derive_traits_for_enum(
                node,
                generic_owner=generic_owner,
                include_default=default_variant_name is not None,
            )

        derived_traits = {
            trait.strip() for trait in derive_traits.split(",") if trait.strip()
        }

        if derive_traits:
            code = f"#[derive({derive_traits})]\n"
        else:
            code = ""
        name = enum_name or node.name
        generic_node = generic_owner or node
        self.enum_generic_params[name] = self.generic_param_names(generic_node)
        code += f"pub enum {name}{self.format_generic_params(generic_node)} {{\n"
        for variant in getattr(node, "variants", []) or []:
            if variant.name == default_variant_name:
                code += "    #[default]\n"
            code += f"    {self.generate_enum_variant(variant)},\n"
        code += "}\n\n"

        if "Default" not in derived_traits:
            code += self.generate_enum_default_impl(node, name, generic_node)

        return code

    def generate_enum_default_impl(self, node, name, generic_node):
        variant = self.enum_default_impl_variant(node)
        if variant is None:
            return ""

        default_expression = self.enum_variant_default_expression(variant)
        generic_decl = self.format_default_impl_generic_params(generic_node)
        generic_use = self.format_generic_argument_params(generic_node)
        code = f"impl{generic_decl} Default for {name}{generic_use} {{\n"
        code += "    fn default() -> Self {\n"
        code += f"        {default_expression}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def enum_default_impl_variant(self, node):
        variants = getattr(node, "variants", []) or []
        for variant in variants:
            if not self.enum_variant_data(variant):
                return variant
        return variants[0] if variants else None

    def enum_variant_default_expression(self, variant):
        data = self.enum_variant_data(variant)
        if not data:
            return f"Self::{variant.name}"

        if all(isinstance(item, tuple) and len(item) == 2 for item in data):
            fields = ", ".join(
                f"{self.rust_identifier(name)}: Default::default()"
                for name, _field_type in data
            )
            return f"Self::{variant.name} {{ {fields} }}"

        args = ", ".join("Default::default()" for _field_type in data)
        return f"Self::{variant.name}({args})"

    def enum_variant_data(self, variant):
        data = getattr(variant, "data", None)
        if data:
            return data
        return getattr(variant, "fields", None) or []

    def format_default_impl_generic_params(self, node):
        generic_params = getattr(node, "generic_params", []) or []
        if not generic_params:
            return ""
        params = []
        for param in generic_params:
            name = self.generic_param_name(param)
            if not name:
                continue
            constraints = self.generic_param_constraints(param)
            if "Default" not in constraints:
                constraints.append("Default")
            params.append(f"{name}: {' + '.join(constraints)}")
        return f"<{', '.join(params)}>" if params else ""

    def generate_enum_variant(self, variant):
        data = self.enum_variant_data(variant)
        if data:
            if all(isinstance(item, tuple) and len(item) == 2 for item in data):
                fields = ", ".join(
                    f"{self.rust_identifier(name)}: "
                    f"{self.map_type(self.convert_type_node_to_string(field_type))}"
                    for name, field_type in data
                )
                return f"{variant.name} {{ {fields} }}"

            fields = ", ".join(
                self.map_type(self.convert_type_node_to_string(field_type))
                for field_type in data
            )
            return f"{variant.name}({fields})"

        value = getattr(variant, "value", None)
        if value is not None:
            return f"{variant.name} = {self.generate_expression(value)}"

        return variant.name

    def format_generic_params(self, node, extra_constraints=None):
        """Render AST generic parameters as a Rust generic parameter list."""
        extra_constraints = extra_constraints or {}
        params = []
        for param in getattr(node, "generic_params", []) or []:
            name = self.generic_param_name(param)
            if not name:
                continue
            constraints = self.generic_param_constraints(param)
            for constraint in extra_constraints.get(name, []):
                if constraint not in constraints:
                    constraints.append(constraint)
            if constraints:
                params.append(f"{name}: {' + '.join(constraints)}")
            else:
                params.append(name)
        return f"<{', '.join(params)}>" if params else ""

    def format_generic_argument_params(self, node):
        """Render only generic parameter names for a Rust type reference."""
        params = self.generic_param_names(node)
        return f"<{', '.join(params)}>" if params else ""

    def generic_param_names(self, node):
        """Return unique generic parameter names from an AST declaration."""
        params = []
        for param in getattr(node, "generic_params", []) or []:
            name = self.generic_param_name(param)

            if name and name not in params:
                params.append(name)

        return params

    def generic_param_name(self, param):
        if hasattr(param, "name"):
            return param.name
        if isinstance(param, (tuple, list)) and param:
            return param[0]
        return str(param)

    def generic_param_constraints(self, param):
        constraints = []
        for constraint in getattr(param, "constraints", []) or []:
            constraint_name = self.generic_constraint_name(constraint)
            if constraint_name and constraint_name not in constraints:
                constraints.append(constraint_name)
        return constraints

    def generic_constraint_name(self, constraint):
        if hasattr(constraint, "name") or hasattr(constraint, "element_type"):
            constraint = self.convert_type_node_to_string(constraint)
        return self.map_type(str(constraint))

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "value") and getattr(type_node, "name", None) is None:
            value = type_node.value
            return str(value).lower() if isinstance(value, bool) else str(value)

        type_class = type_node.__class__.__name__
        if type_class == "ReferenceType":
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            mutability = "mut " if type_node.is_mutable else ""
            return f"&{mutability}{referenced_type}"
        if type_class == "PointerType":
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            mutability = "mut" if type_node.is_mutable else "const"
            return f"*{mutability} {pointee_type}"
        if type_class == "FunctionType":
            params = ", ".join(
                self.convert_type_node_to_string(param_type)
                for param_type in type_node.param_types
            )
            return_type = self.convert_type_node_to_string(type_node.return_type)
            return f"fn({params}) -> {return_type}"
        if type_class == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return (
                f"{element_type}[{size}]" if size is not None else f"{element_type}[]"
            )
        if getattr(type_node, "name", None) is not None:
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            prefix = "dmat" if element_type == "double" else "mat"
            if type_node.rows == type_node.cols:
                return f"{prefix}{type_node.rows}"
            return f"{prefix}{type_node.rows}x{type_node.cols}"
        else:
            return str(type_node)

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        return size

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        for attr in attributes:
            if hasattr(attr, "name") and attr.name in self.semantic_map:
                return attr.name
        return None

    def node_semantic(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        return self.extract_semantic_from_attributes(getattr(node, "attributes", []))

    def rust_semantic_output_kind(self, semantic):
        if semantic is None:
            return None

        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        input_only_sources = {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_instanceid",
            "gl_pointcoord",
            "gl_vertexid",
        }
        input_only_sv_sources = {
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }
        if lower_name in input_only_sources or upper_name in input_only_sv_sources:
            return "input_only"

        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def is_rust_vec4_float_type(self, type_name):
        return self.map_type(type_name) == "Vec4<f32>"

    def is_rust_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "f32"

    def validate_rust_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.rust_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return

        if kind in {"position", "color"}:
            if self.is_rust_vec4_float_type(type_name):
                return
            raise ValueError(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "expected vec4-compatible type"
            )

        if kind == "depth" and not self.is_rust_float_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "expected float type"
            )

    def validate_rust_output_semantic_stage(self, shader_type, semantic, context):
        kind = self.rust_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise ValueError(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if shader_type is None:
            return

        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if shader_type not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise ValueError(
                f"Unsupported {semantic} {context} for Rust {shader_type} stage; "
                f"valid stage is {allowed}"
            )

    def validate_rust_return_semantic(self, shader_type, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "()":
            raise ValueError(
                f"Unsupported {semantic} return semantic for Rust codegen; "
                "void return type"
            )
        self.validate_rust_output_semantic_stage(
            shader_type, semantic, "return semantic"
        )
        self.validate_rust_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_rust_struct_return_semantics(self, shader_type, return_type):
        if shader_type is None:
            return
        base_type = str(return_type).split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        if not member_semantics:
            return
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_rust_output_semantic_stage(shader_type, semantic, context)
            self.validate_rust_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def rust_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def rust_stage_parameter_semantic_rules(self):
        ray_stages = self.rust_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "u32", {"vertex"}),
            "gl_instanceid": ("instance_id", "u32", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "i32", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "u32", {"vertex"}),
            "gl_drawid": ("draw_id", "u32", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "u32", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "u32", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "i32", {"vertex"}),
            "SV_STARTINSTANCELOCATION": ("start_instance_location", "u32", {"vertex"}),
            "SV_DRAWID": ("draw_id", "u32", {"vertex"}),
            "gl_position": ("position", "Vec4<f32>", {"fragment"}),
            "gl_fragcoord": ("position", "Vec4<f32>", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "Vec2<f32>", {"fragment"}),
            "gl_primitiveid": ("primitive_id", "u32", {"fragment"} | ray_hit_stages),
            "gl_sampleid": ("sample_index", "u32", {"fragment"}),
            "SV_POSITION": ("position", "Vec4<f32>", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": ("primitive_id", "u32", {"fragment"} | ray_hit_stages),
            "SV_SAMPLEINDEX": ("sample_index", "u32", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "Vec3<u32>", {"compute"}),
            "gl_localinvocationid": (
                "local_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "gl_globalinvocationid": (
                "global_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "gl_localinvocationindex": (
                "local_invocation_index",
                "u32",
                {"compute"},
            ),
            "SV_GROUPID": ("workgroup_id", "Vec3<u32>", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "Vec3<u32>", {"compute"}),
            "SV_DISPATCHTHREADID": (
                "global_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "SV_GROUPINDEX": ("local_invocation_index", "u32", {"compute"}),
            "gl_launchidext": ("launch_id", "Vec3<u32>", ray_stages),
            "gl_launchsizeext": ("launch_size", "Vec3<u32>", ray_stages),
            "gl_hittext": ("hit_t", "f32", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "u32", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "Vec3<u32>", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": (
                "launch_size",
                "Vec3<u32>",
                ray_stages,
            ),
        }

    def validate_rust_stage_parameter_semantic_type(
        self, parameter, semantic, expected_type
    ):
        actual_type = self.map_type(self.function_parameter_type(parameter))
        if actual_type == expected_type:
            return
        raise ValueError(
            f"Unsupported {semantic} stage parameter semantic for Rust codegen; "
            f"expected {expected_type} type"
        )

    def validate_rust_stage_parameter_semantics(self, shader_type, parameters):
        if shader_type is None:
            return

        rules = self.rust_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for parameter in parameters or []:
            semantic = self.node_semantic(parameter)
            if semantic is None:
                continue

            semantic_key = self.rust_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if shader_type not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise ValueError(
                        f"Unsupported {semantic} stage parameter semantic for Rust "
                        f"{shader_type} stage; valid stage is {allowed}"
                    )
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise ValueError(
                        f"Duplicate Rust stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{parameter.name}'"
                    )
                seen_system_semantics[mapped_semantic] = parameter.name
                self.validate_rust_stage_parameter_semantic_type(
                    parameter, semantic, expected_type
                )
                continue

            kind = self.rust_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise ValueError(
                    f"Unsupported {semantic} stage parameter semantic for Rust "
                    f"{shader_type} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def get_member_type(self, member):
        if hasattr(member, "member_type"):
            member_type = self.convert_type_node_to_string(member.member_type)
            return self.resource_type_with_attributes(member_type, member)
        if hasattr(member, "vtype"):
            return self.resource_type_with_attributes(member.vtype, member)
        return "float"

    def get_cbuffer_nodes(self, ast):
        nodes = []
        seen = set()
        for attr in ("cbuffers", "constants"):
            for node in getattr(ast, attr, None) or []:
                node_id = id(node)
                if node_id not in seen:
                    nodes.append(node)
                    seen.add(node_id)
        return nodes

    def map_type_to_rust(self, type_str):
        """Enhanced type mapping for Rust."""
        if type_str.startswith("float") and len(type_str) > 5:
            size = type_str[5:]
            if size.isdigit():
                return f"Vec{size}<f32>"
        elif type_str.startswith("int") and len(type_str) > 3:
            size = type_str[3:]
            if size.isdigit():
                return f"Vec{size}<i32>"

        # Standard type mapping
        type_map = {
            "void": "()",
            "bool": "bool",
            "int": "i32",
            "uint": "u32",
            "float": "f32",
            "double": "f64",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            "float2": "Vec2<f32>",
            "float3": "Vec3<f32>",
            "float4": "Vec4<f32>",
        }
        mapped_type = type_map.get(type_str, type_str)
        if mapped_type != type_str:
            return self.qualify_colliding_runtime_type(mapped_type)
        return mapped_type

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = self.get_cbuffer_nodes(ast)
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += self.rust_resource_metadata_comment(
                    node, getattr(node, "name", None), kind="cbuffer"
                )
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    member_name = self.rust_identifier(member.name)
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += (
                                f"    pub {member_name}: "
                                f"[{self.map_type(member.element_type)}; "
                                f"{member.size}],\n"
                            )
                        else:
                            code += (
                                f"    pub {member_name}: "
                                f"Vec<{self.map_type(member.element_type)}>,\n"
                            )
                    else:
                        code += (
                            f"    pub {member_name}: "
                            f"{self.map_type(self.get_member_type(member))},\n"
                        )
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += self.rust_resource_metadata_comment(
                    node, getattr(node, "name", None), kind="cbuffer"
                )
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    member_name = self.rust_identifier(member.name)
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += (
                                f"    pub {member_name}: "
                                f"[{self.map_type(member.element_type)}; "
                                f"{member.size}],\n"
                            )
                        else:
                            code += (
                                f"    pub {member_name}: "
                                f"Vec<{self.map_type(member.element_type)}>,\n"
                            )
                    else:
                        code += (
                            f"    pub {member_name}: "
                            f"{self.map_type(self.get_member_type(member))},\n"
                        )
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
        return code

    def generate_cbuffer_member_statics(self, members):
        code = ""
        for member in members:
            if isinstance(member, ArrayNode):
                if member.size:
                    member_type = (
                        f"[{self.map_type(member.element_type)}; {member.size}]"
                    )
                else:
                    member_type = f"Vec<{self.map_type(member.element_type)}>"
            else:
                member_type = self.map_type(self.get_member_type(member))
            initializer = self.rust_static_default_initializer(member_type)
            code += self.generate_static_declaration(
                member.name,
                member_type,
                initializer,
            )
        return code

    def generate_variable_static_declaration(self, node):
        """Render a module-level static for a global or stage-local variable."""
        var_type = self.get_variable_type(node)
        if var_type is None:
            var_type = "float"
        elif hasattr(var_type, "name") or hasattr(var_type, "element_type"):
            var_type = self.convert_type_node_to_string(var_type)

        self.register_glsl_buffer_block_metadata(node)
        self.register_variable_type(node.name, var_type, scope="static")
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        rust_type = self.map_type(var_type)

        if initial_value is not None:
            init_expr = self.generate_expression_with_type(
                initial_value,
                var_type,
                static_context=True,
            )
            lazy_lock = self.static_array_literal_requires_lazy_lock(
                var_type,
                initial_value,
            )
        else:
            init_expr = self.rust_static_default_initializer(rust_type)
            lazy_lock = False

        return (
            self.rust_resource_metadata_comment(node, var_type)
            + self.rust_resource_placeholder_diagnostic_comment(node, var_type)
            + self.generate_static_declaration(
                node.name,
                rust_type,
                init_expr,
                lazy_lock=lazy_lock,
            )
        )

    def generate_global_array_declaration(self, node):
        element_type_name = self.convert_type_node_to_string(
            getattr(node, "element_type", "float")
        )
        element_type_name = self.resource_type_with_attributes(element_type_name, node)
        element_type = self.map_type(element_type_name)
        size = self.format_array_size(getattr(node, "size", None))
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))

        if size is None:
            rust_type = f"Vec<{element_type}>"
            source_type = f"{element_type_name}[]"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = source_type
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False
        else:
            rust_type = f"[{element_type}; {size}]"
            source_type = f"{element_type_name}[{size}]"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = source_type
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False

        self.register_variable_type(node.name, source_type, scope="static")
        return (
            self.rust_resource_metadata_comment(node, source_type)
            + self.rust_resource_placeholder_diagnostic_comment(node, source_type)
            + self.generate_static_declaration(
                node.name, rust_type, initializer, lazy_lock=lazy_lock
            )
        )

    def register_constant_types(self, ast):
        for node in getattr(ast, "constants", []) or []:
            if not isinstance(node, ConstantNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            self.register_variable_type(name, const_type, scope="static")

    def generate_constant_declarations(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            if isinstance(node, ConstantNode):
                code += self.generate_constant_declaration(node)
        return f"{code}\n" if code else ""

    def generate_constant_declaration(self, node):
        name = getattr(node, "name", None)
        if not name:
            return ""

        const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
        if hasattr(const_type, "name") or hasattr(const_type, "element_type"):
            const_type = self.convert_type_node_to_string(const_type)
        else:
            const_type = str(const_type)

        self.register_variable_type(name, const_type, scope="static")
        rust_type = self.map_type(const_type)
        value = getattr(node, "value", None)
        if value is None:
            initializer = self.rust_static_default_initializer(rust_type)
        else:
            initializer = self.generate_expression_with_type(
                value,
                const_type,
                static_context=True,
            )
            initializer = self.normalize_assignment_rhs(
                const_type,
                value,
                initializer,
                "=",
            )

        if self.rust_constant_initializer_is_const(rust_type):
            symbol = self.static_symbol_name(name)
            return f"const {symbol}: {rust_type} = {initializer};\n"

        return self.generate_static_declaration(
            name,
            rust_type,
            initializer,
            lazy_lock=True,
        )

    def rust_constant_initializer_is_const(self, rust_type):
        return self.rust_scalar_default_literal(rust_type) is not None

    def generate_static_declaration(
        self, name, rust_type, initializer, lazy_lock=False
    ):
        symbol = self.static_symbol_name(name)
        if self.pointer_type_parts(rust_type) is not None:
            self.lazy_static_names.discard(name)
            initializer = self.rust_static_pointer_storage_initializer(initializer)
            return f"static {symbol}: usize = {initializer};\n"
        lazy_lock = lazy_lock or self.static_initializer_requires_lazy_lock(
            rust_type,
            initializer,
        )
        if lazy_lock:
            self.lazy_static_names.add(name)
            return (
                f"static {symbol}: std::sync::LazyLock<{rust_type}> = "
                f"std::sync::LazyLock::new(|| {initializer});\n"
            )
        self.lazy_static_names.discard(name)
        return f"static {symbol}: {rust_type} = {initializer};\n"

    def static_initializer_requires_lazy_lock(self, rust_type, initializer):
        return str(initializer) == "Default::default()"

    def rust_static_pointer_storage_initializer(self, initializer):
        initializer = str(initializer).strip()
        if initializer in {
            "",
            "0",
            "Default::default()",
            "std::ptr::null()",
            "std::ptr::null_mut()",
        }:
            return "0"
        return f"({initializer} as usize)"

    def static_array_literal_requires_lazy_lock(self, target_type, initial_value):
        if not isinstance(initial_value, ArrayLiteralNode):
            return False
        if not self.is_array_type_name(target_type):
            return False

        base_type, sizes = self.c_array_type_parts(target_type)
        size = sizes[0] if sizes else None
        if size is None:
            return True

        return self.is_rust_shader_pod_value_type(self.map_type(base_type))

    def rust_static_default_initializer(self, rust_type):
        rust_type = str(rust_type)

        if rust_type.startswith("Vec<"):
            return "Vec::new()"

        array_type = self.parse_rust_array_type(rust_type)
        if array_type is not None:
            element_type, size = array_type
            element_initializer = self.rust_scalar_default_literal(element_type)
            if element_initializer is not None:
                return f"[{element_initializer}; {size}]"
            if self.rust_type_is_zeroable(element_type):
                return "unsafe { std::mem::zeroed() }"
            if self.is_rust_shader_pod_value_type(element_type):
                return "unsafe { std::mem::zeroed() }"
            return "Default::default()"

        scalar_initializer = self.rust_scalar_default_literal(rust_type)
        if scalar_initializer is not None:
            return scalar_initializer

        return "Default::default()"

    def rust_runtime_default_initializer(self, type_name):
        rust_type = self.map_type(type_name)
        if self.parse_rust_array_type(
            rust_type
        ) is not None and self.rust_type_is_zeroable(rust_type):
            return "unsafe { std::mem::zeroed() }"
        return self.rust_static_default_initializer(rust_type)

    def local_array_default_initializer(self, rust_type):
        rust_type = str(rust_type)
        if rust_type.startswith("Vec<"):
            return "Vec::new()"
        if self.parse_rust_array_type(rust_type) is not None:
            if not self.rust_type_default_derive_supported(
                rust_type
            ) and self.rust_type_is_zeroable(rust_type):
                return "unsafe { std::mem::zeroed() }"
            return "std::array::from_fn(|_| Default::default())"
        return None

    def parse_rust_array_type(self, rust_type):
        if not (rust_type.startswith("[") and rust_type.endswith("]")):
            return None
        body = rust_type[1:-1]
        if ";" not in body:
            return None
        element_type, size = body.rsplit(";", 1)
        return element_type.strip(), size.strip()

    def rust_scalar_default_literal(self, rust_type):
        rust_type = str(rust_type)
        if rust_type in {"f16", "f32", "f64"}:
            return "0.0"
        if rust_type in {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
        }:
            return "0"
        if rust_type == "bool":
            return "false"
        if rust_type == "char":
            return r"'\0'"
        if rust_type == "&'static str":
            return '""'
        return None

    def rust_type_is_zeroable(self, rust_type):
        rust_type = str(rust_type).strip()
        if self.pointer_type_parts(rust_type) is not None:
            return True

        array_type = self.parse_rust_array_type(rust_type)
        if array_type is not None:
            element_type, _size = array_type
            return self.rust_type_is_zeroable(element_type)

        if self.rust_scalar_default_literal(rust_type) is not None:
            return rust_type != "&'static str"

        return self.is_rust_shader_pod_value_type(rust_type)

    def is_rust_shader_pod_value_type(self, rust_type):
        rust_type = self.unqualify_runtime_type_name(rust_type)
        return str(rust_type).startswith(
            (
                "Vec2<",
                "Vec3<",
                "Vec4<",
                "Mat2<",
                "Mat3<",
                "Mat4<",
                "Mat2x",
                "Mat3x",
                "Mat4x",
            )
        ) and str(rust_type).endswith(">")

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        function_name=None,
        stage_node=None,
        extra_params=None,
    ):
        """Render one CrossGL function or shader entry point as Rust code."""
        if self.is_function_prototype(func):
            return ""

        code = ""
        code += "  " * indent
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.local_variable_names = set()
        saved_return_type = self.current_return_type
        saved_shader_type = self.current_shader_type
        saved_generic_param_names = self.current_generic_param_names
        saved_function_generic_constraints = self.current_function_generic_constraints
        saved_mutated_names = self.current_mutated_names
        self.current_generic_param_names = set(self.generic_param_names(func))
        self.current_shader_type = shader_type
        param_list = list(getattr(func, "parameters", getattr(func, "params", [])))
        if extra_params:
            param_list.extend(extra_params)
        for p in param_list:
            self.register_glsl_buffer_block_metadata(p)
            self.register_variable_type(
                p.name,
                self.function_parameter_type(p),
                scope="local",
            )
        self.current_mutated_names = self.collect_mutated_binding_names(
            getattr(func, "body", None)
        )

        inferred_constraints = self.combined_function_generic_constraints(func)
        self.current_function_generic_constraints = inferred_constraints

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.current_return_type = return_type
        return_semantic = self.node_semantic(func)
        if shader_type == "geometry":
            self.validate_rust_geometry_stage(func, param_list)
        if shader_type in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_rust_tessellation_stage(func, param_list, shader_type)
        if shader_type in self.rust_mesh_stage_names():
            self.validate_rust_mesh_stage(func, param_list, shader_type, stage_node)
        self.validate_rust_return_semantic(shader_type, return_type, return_semantic)
        self.validate_rust_struct_return_semantics(shader_type, return_type)
        self.validate_rust_stage_parameter_semantics(shader_type, param_list)

        param_types = [self.function_parameter_type(p) for p in param_list]
        reference_lifetime = self.function_reference_return_lifetime(
            func,
            return_type,
            param_types,
        )

        params = []
        for p, param_type in zip(param_list, param_types):
            self.register_glsl_buffer_block_metadata(p)
            self.register_variable_type(p.name, param_type, scope="local")
            source_param_name = p.name
            param_name = self.rust_identifier(source_param_name)
            if source_param_name in self.current_mutated_names or (
                self.function_parameter_requires_mut_binding(
                    p,
                    inferred_constraints,
                )
            ):
                param_name = f"mut {param_name}"
            params.append(
                f"{param_name}: "
                f"{self.map_function_parameter_type_with_lifetime(param_type, reference_lifetime)}"
            )

        params_str = ", ".join(params) if params else ""

        if shader_type == "vertex":
            code += self.generate_rust_stage_attribute("vertex_shader")
        elif shader_type == "fragment":
            code += self.generate_rust_stage_attribute("fragment_shader")
        elif shader_type == "compute":
            code += self.generate_rust_stage_attribute("compute_shader")
        elif shader_type == "geometry":
            code += self.generate_rust_geometry_stage_comments(func, param_list)
        elif shader_type in {"tessellation_control", "tessellation_evaluation"}:
            code += self.generate_rust_tessellation_stage_comments(
                func,
                param_list,
                shader_type,
            )
        elif shader_type in self.rust_mesh_stage_names():
            code += f"// CrossGL shader stage: {shader_type}\n"
            code += self.generate_rust_mesh_stage_comments(
                func, param_list, shader_type, stage_node
            )
        elif shader_type in self.rust_ray_stage_names():
            code += (
                "// CrossGL ray stage: "
                f"{self.rust_ray_stage_metadata_name(shader_type)}\n"
            )
        if return_semantic:
            code += (
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}\n"
            )
        if shader_type:
            for param in param_list:
                param_semantic = self.node_semantic(param)
                if param_semantic:
                    code += (
                        f"// CrossGL parameter semantic: {param.name}: "
                        f"{self.map_semantic(param_semantic)}\n"
                    )

        generic_params = self.format_function_generic_params(
            func,
            extra_constraints=inferred_constraints,
            lifetime=reference_lifetime,
        )
        emitted_name = self.rust_identifier(function_name or func.name)
        code += (
            f"pub fn {emitted_name}{generic_params}({params_str}) "
            f"-> {self.map_type_with_lifetime(return_type, reference_lifetime)} {{\n"
        )

        if stage_node is not None:
            for variable in getattr(stage_node, "local_variables", []) or []:
                if self.is_rust_shared_variable(variable):
                    code += self.generate_statement(variable, indent + 1)

        body = getattr(func, "body", [])
        if hasattr(body, "statements"):
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)

        code += "  " * indent + "}\n\n"
        self.variable_types = saved_variable_types
        self.local_variable_names = saved_local_variable_names
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_return_type = saved_return_type
        self.current_shader_type = saved_shader_type
        self.current_generic_param_names = saved_generic_param_names
        self.current_function_generic_constraints = saved_function_generic_constraints
        self.current_mutated_names = saved_mutated_names
        return code

    def is_function_prototype(self, func):
        return isinstance(func, FunctionNode) and getattr(func, "body", None) is None

    def is_rust_shared_variable(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return bool(qualifiers & {"shared", "workgroup", "groupshared"})

    def generate_rust_stage_attribute(self, attribute_name):
        """Render a host-compatible Rust shader stage attribute."""
        return f'#[cfg_attr(feature = "crossgl_gpu", {attribute_name})]\n'

    def function_reference_return_lifetime(self, func, return_type, param_types):
        if self.reference_type_parts_for_type(return_type) is None:
            return None

        reference_param_count = sum(
            1
            for param_type in param_types
            if self.reference_type_parts_for_type(param_type) is not None
        )
        if reference_param_count <= 1:
            return None

        generic_names = set(self.generic_param_names(func))
        for lifetime in ("'a", "'b", "'cgl_ref"):
            if lifetime not in generic_names:
                return lifetime
        return "'cgl_ref"

    def format_function_generic_params(
        self,
        node,
        extra_constraints=None,
        lifetime=None,
    ):
        generic_params = self.format_generic_params(
            node,
            extra_constraints=extra_constraints,
        )
        if lifetime is None:
            return generic_params
        if not generic_params:
            return f"<{lifetime}>"
        return f"<{lifetime}, {generic_params[1:-1]}>"

    def map_function_parameter_type_with_lifetime(self, param_type, lifetime):
        if lifetime is not None and self.reference_type_parts_for_type(param_type):
            return self.map_type_with_lifetime(param_type, lifetime)
        return self.map_function_parameter_type(param_type)

    def combined_function_generic_constraints(self, func):
        """Return explicit and inferred generic constraints for one function."""
        getattr(func, "name", None)
        cache_key = id(func)
        if cache_key in self.user_function_generic_constraint_cache:
            return self.user_function_generic_constraint_cache[cache_key]
        if cache_key in self.active_generic_constraint_functions:
            return {}

        self.active_generic_constraint_functions.add(cache_key)
        try:
            constraints = {}
            for param in getattr(func, "generic_params", []) or []:
                name = self.generic_param_name(param)
                if name:
                    constraints[name] = self.generic_param_constraints(param)

            inferred = self.infer_function_generic_operator_constraints(func)
            for name, bounds in inferred.items():
                for bound in bounds:
                    self.add_generic_constraint(constraints, name, bound)
        finally:
            self.active_generic_constraint_functions.remove(cache_key)

        self.user_function_generic_constraint_cache[cache_key] = constraints
        return constraints

    def infer_function_generic_operator_constraints(self, func):
        """Infer Rust operator trait bounds needed by generic function bodies."""
        generic_names = set(self.generic_param_names(func))
        if not generic_names:
            return {}

        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        saved_generic_param_names = self.current_generic_param_names
        constraints = {name: [] for name in generic_names}

        try:
            self.variable_types = {}
            self.local_variable_names = set()
            self.current_generic_param_names = generic_names
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                self.register_variable_type(
                    param.name,
                    self.function_parameter_type(param),
                    scope="local",
                )
            self.collect_generic_operator_constraints(
                getattr(func, "body", []),
                generic_names,
                constraints,
            )
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names
            self.current_generic_param_names = saved_generic_param_names

        return {name: bounds for name, bounds in constraints.items() if bounds}

    def collect_mutated_binding_names(self, node):
        """Collect local binding names that need `mut` because they are assigned."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if hasattr(current, "statements"):
                collect(current.statements)
                return

            if isinstance(current, AssignmentNode):
                root_name = self.assignment_target_root_name(current.target)
                if root_name:
                    names.add(root_name)
                collect(current.value)
                return

            if isinstance(current, VariableNode):
                initial_value = getattr(current, "initial_value", None)
                declared_type = self.get_variable_type(current)
                reference_parts = self.reference_type_parts_for_type(declared_type)
                if reference_parts is not None and reference_parts[0]:
                    names.update(
                        self.mutable_reference_borrow_root_names(initial_value)
                    )
                collect(initial_value)
                return

            if isinstance(current, ExpressionStatementNode):
                collect(current.expression)
                return

            if isinstance(current, ReturnNode):
                collect(current.value)
                return

            if isinstance(current, IfNode):
                collect(current.condition)
                collect(current.then_branch)
                collect(current.else_branch)
                return

            if isinstance(current, ForNode):
                collect(current.init)
                collect(current.condition)
                collect(current.update)
                collect(current.body)
                return

            if isinstance(current, ForInNode):
                collect(current.iterable)
                collect(current.body)
                return

            if isinstance(current, (WhileNode, DoWhileNode)):
                collect(current.condition)
                collect(current.body)
                return

            if isinstance(current, LoopNode):
                collect(current.body)
                return

            if isinstance(current, MatchNode):
                collect(current.expression)
                for arm in getattr(current, "arms", []) or []:
                    collect(getattr(arm, "guard", None))
                    collect(getattr(arm, "body", None))
                return

            if isinstance(current, SwitchNode):
                collect(getattr(current, "expression", None))
                for case in getattr(current, "cases", []) or []:
                    collect(getattr(case, "value", None))
                    collect(getattr(case, "statements", []))
                collect(getattr(current, "default_case", None))
                return

            if isinstance(current, UnaryOpNode):
                operator = self.map_operator(
                    getattr(current, "operator", getattr(current, "op", None))
                )
                if operator in {"++", "--"}:
                    root_name = self.assignment_target_root_name(current.operand)
                    if root_name:
                        names.add(root_name)
                collect(current.operand)
                return

            if isinstance(current, BinaryOpNode):
                collect(current.left)
                collect(current.right)
                return

            if isinstance(current, TernaryOpNode):
                collect(current.condition)
                collect(current.true_expr)
                collect(current.false_expr)
                return

            if isinstance(current, FunctionCallNode):
                func_expr = getattr(current, "function", getattr(current, "name", None))
                func_name = self.function_call_name(func_expr)
                args = getattr(current, "arguments", getattr(current, "args", []))
                for index, argument in enumerate(args or []):
                    param_type = self.user_function_param_type(func_name, index)
                    reference_parts = self.reference_type_parts_for_type(param_type)
                    if reference_parts is None or not reference_parts[0]:
                        continue
                    names.update(self.mutable_reference_borrow_root_names(argument))
                collect(current.function)
                collect(args)
                return

            if isinstance(current, MemberAccessNode):
                collect(current.object_expr)
                return

            if isinstance(current, PointerAccessNode):
                collect(current.pointer_expr)
                return

            if isinstance(current, ArrayAccessNode):
                collect(current.array_expr)
                collect(current.index_expr)
                return

            if isinstance(current, ConstructorNode):
                collect(getattr(current, "arguments", []))
                collect(list((getattr(current, "named_arguments", {}) or {}).values()))
                return

            if isinstance(current, ArrayLiteralNode):
                collect(getattr(current, "elements", []))

        collect(node)
        return names

    def assignment_target_root_name(self, target):
        """Return the binding name mutated by an assignment target."""
        if isinstance(target, IdentifierNode):
            return target.name
        if isinstance(target, VariableNode):
            return target.name
        if isinstance(target, str):
            return target if target.isidentifier() else None
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root_name(target.object_expr)
        if isinstance(target, PointerAccessNode):
            return self.assignment_target_root_name(target.pointer_expr)
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root_name(target.array_expr)
        return None

    def mutable_reference_borrow_root_names(self, expr):
        """Return owned roots that must be mutable for an `&mut` borrow."""
        if expr is None:
            return set()

        source_reference = self.reference_type_parts_for_type(
            self.expression_result_type(expr)
        )
        if source_reference is not None:
            return set()

        if isinstance(expr, TernaryOpNode):
            return self.mutable_reference_borrow_root_names(
                expr.true_expr
            ) | self.mutable_reference_borrow_root_names(expr.false_expr)

        if isinstance(expr, MatchNode):
            names = set()
            subject_type = self.expression_result_type(
                getattr(expr, "expression", None)
            )
            for arm in getattr(expr, "arms", []) or []:
                saved_variable_types = self.variable_types.copy()
                saved_local_variable_names = self.local_variable_names.copy()
                try:
                    self.register_match_pattern_bindings(
                        getattr(arm, "pattern", None),
                        subject_type,
                    )
                    names.update(
                        self.mutable_reference_borrow_root_names(
                            self.match_arm_tail_expression(getattr(arm, "body", None))
                        )
                    )
                finally:
                    self.variable_types = saved_variable_types
                    self.local_variable_names = saved_local_variable_names
            return names

        if isinstance(expr, BlockNode):
            explicit_tail = getattr(expr, "expression", None)
            if explicit_tail is not None:
                return self.mutable_reference_borrow_root_names(explicit_tail)
            statements = self.statement_list(expr)
            if not statements:
                return set()
            tail_expression = self.block_tail_expression(statements[-1])
            if tail_expression is not None:
                return self.mutable_reference_borrow_root_names(tail_expression)
            return set()

        root_name = self.assignment_target_root_name(expr)
        return {root_name} if root_name else set()

    def match_arm_tail_expression(self, body):
        statements = self.statement_list(body)
        if not statements:
            return None
        return self.block_tail_expression(statements[-1])

    def generate_borrow_operand_expression(self, expr):
        """Render a borrow operand without value-context clones."""
        self.assignment_lhs_depth += 1
        try:
            return self.generate_expression(expr)
        finally:
            self.assignment_lhs_depth -= 1

    def collect_generic_operator_constraints(self, node, generic_names, constraints):
        if node is None:
            return

        if isinstance(node, list):
            for item in node:
                self.collect_generic_operator_constraints(
                    item, generic_names, constraints
                )
            return

        if hasattr(node, "statements"):
            for statement in node.statements:
                self.collect_generic_operator_constraints(
                    statement, generic_names, constraints
                )
            return

        if isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", None)
            self.collect_generic_operator_constraints(
                initial_value, generic_names, constraints
            )
            declared_type = self.get_variable_type(node)
            variable_type = declared_type or self.expression_result_type(initial_value)
            if variable_type is not None:
                self.register_variable_type(node.name, variable_type, scope="local")
            return

        if isinstance(node, ExpressionStatementNode):
            self.collect_generic_operator_constraints(
                node.expression, generic_names, constraints
            )
            return

        if isinstance(node, ReturnNode):
            self.collect_generic_operator_constraints(
                node.value, generic_names, constraints
            )
            return

        if isinstance(node, AssignmentNode):
            self.collect_generic_operator_constraints(
                node.target, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.value, generic_names, constraints
            )
            return

        if isinstance(node, IfNode):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.then_branch, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.else_branch, generic_names, constraints
            )
            return

        if isinstance(node, ForNode):
            self.collect_generic_operator_constraints(
                node.init, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.update, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, ForInNode):
            self.collect_generic_operator_constraints(
                node.iterable, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, (WhileNode, DoWhileNode)):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, LoopNode):
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, MatchNode):
            subject_expr = getattr(node, "expression", None)
            subject_type = self.expression_result_type(subject_expr)
            self.collect_generic_operator_constraints(
                subject_expr, generic_names, constraints
            )
            for arm in getattr(node, "arms", []) or []:
                saved_variable_types = self.variable_types.copy()
                saved_local_variable_names = self.local_variable_names.copy()
                try:
                    self.register_match_pattern_bindings(
                        getattr(arm, "pattern", None), subject_type
                    )
                    self.collect_generic_operator_constraints(
                        getattr(arm, "guard", None), generic_names, constraints
                    )
                    self.collect_generic_operator_constraints(
                        getattr(arm, "body", None), generic_names, constraints
                    )
                finally:
                    self.variable_types = saved_variable_types
                    self.local_variable_names = saved_local_variable_names
            return

        if isinstance(node, BinaryOpNode):
            self.collect_generic_operator_constraints(
                node.left, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.right, generic_names, constraints
            )
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", None))
            )
            self.add_generic_operator_constraints(
                operator,
                self.expression_result_type(node.left),
                self.expression_result_type(node.right),
                generic_names,
                constraints,
            )
            return

        if isinstance(node, UnaryOpNode):
            self.collect_generic_operator_constraints(
                node.operand, generic_names, constraints
            )
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", None))
            )
            operand_type = self.expression_result_type(node.operand)
            if operator == "-" and operand_type in generic_names:
                self.add_generic_constraint(
                    constraints,
                    operand_type,
                    f"std::ops::Neg<Output = {operand_type}>",
                )
            return

        if isinstance(node, TernaryOpNode):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.true_expr, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.false_expr, generic_names, constraints
            )
            return

        if isinstance(node, FunctionCallNode):
            self.collect_generic_operator_constraints(
                node.function, generic_names, constraints
            )
            for argument in getattr(node, "arguments", []) or []:
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            self.add_builtin_function_call_constraints(node, generic_names, constraints)
            self.add_user_function_call_constraints(node, generic_names, constraints)
            return

        if isinstance(node, MemberAccessNode):
            self.collect_generic_operator_constraints(
                node.object_expr, generic_names, constraints
            )
            return

        if isinstance(node, ConstructorNode):
            for argument in getattr(node, "arguments", []) or []:
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            for argument in (getattr(node, "named_arguments", {}) or {}).values():
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            return

        if isinstance(node, ArrayLiteralNode):
            for element in getattr(node, "elements", []) or []:
                self.collect_generic_operator_constraints(
                    element, generic_names, constraints
                )
            return

        if isinstance(node, ArrayAccessNode):
            self.collect_generic_operator_constraints(
                node.array_expr, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.index_expr, generic_names, constraints
            )

    def add_builtin_function_call_constraints(self, node, generic_names, constraints):
        func_name = self.function_call_name(getattr(node, "function", None))
        if self.is_user_defined_function(func_name):
            return

        mapped_name = self.function_map.get(func_name, func_name)
        if mapped_name != "sqrt":
            return

        arguments = getattr(node, "arguments", []) or []
        if len(arguments) != 1:
            return

        argument_type = self.expression_result_type(arguments[0])
        if argument_type in generic_names:
            self.add_generic_constraint(constraints, argument_type, "CglSqrt")
            self.required_generic_math_traits.add("CglSqrt")

    def add_user_function_call_constraints(self, node, generic_names, constraints):
        func_name = self.function_call_name(getattr(node, "function", None))
        callee = self.user_function_nodes.get(func_name)
        if callee is None:
            return

        substitutions = self.infer_call_generic_substitutions(callee, node)
        if not substitutions:
            return

        callee_constraints = self.combined_function_generic_constraints(callee)
        for callee_param, caller_type in substitutions.items():
            if caller_type not in generic_names:
                continue
            for bound in callee_constraints.get(callee_param, []):
                self.add_generic_constraint(
                    constraints,
                    caller_type,
                    self.substitute_generic_bound(bound, callee_param, caller_type),
                )

    def function_call_name(self, function_expr):
        if isinstance(function_expr, IdentifierNode):
            return function_expr.name
        if isinstance(function_expr, str):
            return function_expr
        return getattr(function_expr, "name", None)

    def infer_call_generic_substitutions(self, callee, call_node):
        callee_generic_names = self.generic_param_names(callee)
        if not callee_generic_names:
            return {}

        substitutions = {}
        param_types = [
            self.function_parameter_type(param)
            for param in getattr(callee, "parameters", getattr(callee, "params", []))
        ]
        arg_types = [
            self.expression_result_type(argument)
            for argument in getattr(call_node, "arguments", []) or []
        ]

        for param_type, arg_type in zip(param_types, arg_types):
            self.collect_generic_substitutions_from_type(
                param_type,
                arg_type,
                set(callee_generic_names),
                substitutions,
            )

        return substitutions

    def collect_generic_substitutions_from_type(
        self, param_type, arg_type, callee_generic_names, substitutions
    ):
        if param_type is None or arg_type is None:
            return
        param_type = (
            self.convert_type_node_to_string(param_type)
            if hasattr(param_type, "name") or hasattr(param_type, "element_type")
            else str(param_type)
        )
        arg_type = (
            self.convert_type_node_to_string(arg_type)
            if hasattr(arg_type, "name") or hasattr(arg_type, "element_type")
            else str(arg_type)
        )

        if param_type in callee_generic_names:
            substitutions.setdefault(param_type, arg_type)
            return

        param_base, param_args = self.generic_type_parts(param_type)
        arg_base, arg_args = self.generic_type_parts(arg_type)
        if param_base != arg_base or len(param_args) != len(arg_args):
            return

        for nested_param, nested_arg in zip(param_args, arg_args):
            self.collect_generic_substitutions_from_type(
                nested_param,
                nested_arg,
                callee_generic_names,
                substitutions,
            )

    def substitute_generic_bound(self, bound, source_name, target_name):
        return str(bound).replace(f"Output = {source_name}", f"Output = {target_name}")

    def add_generic_operator_constraints(
        self, operator, left_type, right_type, generic_names, constraints
    ):
        generic_types = []
        for type_name in (left_type, right_type):
            if type_name in generic_names and type_name not in generic_types:
                generic_types.append(type_name)

        for generic_type in generic_types:
            bound = self.generic_operator_bound(operator, generic_type)
            if bound is not None:
                self.add_generic_constraint(constraints, generic_type, bound)

    def generic_operator_bound(self, operator, generic_name):
        if operator in {"==", "!="}:
            return "PartialEq"
        if operator in {"<", ">", "<=", ">="}:
            return "PartialOrd"

        operator_traits = {
            "+": "Add",
            "-": "Sub",
            "*": "Mul",
            "/": "Div",
            "%": "Rem",
        }
        trait = operator_traits.get(operator)
        if trait is None:
            return None
        return f"std::ops::{trait}<Output = {generic_name}>"

    def add_generic_constraint(self, constraints, generic_name, bound):
        if generic_name not in constraints:
            constraints[generic_name] = []
        if bound not in constraints[generic_name]:
            constraints[generic_name].append(bound)

    def function_parameter_type(self, param):
        if hasattr(param, "param_type"):
            param_type = self.convert_type_node_to_string(param.param_type)
            return self.resource_type_with_attributes(param_type, param)
        if hasattr(param, "vtype"):
            vtype = param.vtype
            if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
                vtype = self.convert_type_node_to_string(vtype)
            return self.resource_type_with_attributes(vtype, param)
        return "float"

    def function_parameter_requires_mut_binding(self, param, generic_constraints):
        qualifiers = {str(q) for q in getattr(param, "qualifiers", []) or []}
        if "mut" in qualifiers or getattr(param, "is_mutable", False):
            return True

        param_type = self.function_parameter_type(param)
        if self.callable_type_uses_trait(param_type, "FnMut"):
            return True

        param_name = self.type_name_string(param_type)
        if not param_name:
            return False

        return any(
            self.callable_type_uses_trait(bound, "FnMut")
            for bound in generic_constraints.get(param_name, [])
        )

    def map_function_parameter_type(self, param_type):
        rust_type = self.map_type(param_type)
        if self.callable_signature_parts(rust_type) is None:
            return rust_type

        if rust_type.startswith(("impl ", "dyn ")):
            return rust_type
        return f"impl {rust_type}"

    def generate_param_attributes(self, param):
        """Generate Rust GPU parameter attributes based on semantic"""
        if not param.semantic:
            return ""

        semantic = param.semantic.lower()
        if "position" in semantic:
            return "#[location(0)] "
        elif "normal" in semantic:
            return "#[location(1)] "
        elif "texcoord" in semantic:
            if "texcoord0" in semantic:
                return "#[location(2)] "
            elif "texcoord1" in semantic:
                return "#[location(3)] "
            else:
                return "#[location(2)] "
        elif "color" in semantic:
            return "#[location(4)] "
        elif "gl_position" in semantic:
            return "#[builtin(position)] "
        elif "gl_fragcolor" in semantic:
            return "#[location(0)] "
        return ""

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL statement as Rust code."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            initial_value = getattr(stmt, "initial_value", None)
            vtype = self.variable_declaration_type(stmt, initial_value)
            self.register_variable_type(stmt.name, vtype, scope="local")
            binding_keyword = self.local_let_keyword(stmt)
            binding_name = self.rust_identifier(stmt.name)
            rust_type = self.map_type(vtype)
            default_array_initializer = self.local_array_default_initializer(rust_type)
            if initial_value is not None:
                increment_init = self.generate_increment_initializer_declaration(
                    stmt,
                    initial_value,
                    vtype,
                    indent,
                )
                if increment_init is not None:
                    return increment_init
                if isinstance(initial_value, MatchNode):
                    init_expr = self.generate_match_expression(
                        initial_value,
                        indent,
                        target_type=vtype,
                    )
                else:
                    init_expr = self.generate_expression_with_type(initial_value, vtype)
                    init_expr = self.normalize_assignment_rhs(
                        vtype, initial_value, init_expr, "="
                    )
                if self.expression_contains_block_node(initial_value):
                    init_expr = self.indent_multiline_expression(init_expr, indent)
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {init_expr};\n"
                )
            elif self.is_generated_struct_type(vtype):
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = Default::default();\n"
                )
            elif self.is_default_constructible_ray_runtime_type(rust_type):
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = Default::default();\n"
                )
            elif default_array_initializer is not None:
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {default_array_initializer};\n"
                )
            elif self.local_value_default_initializer(rust_type) is not None:
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {self.local_value_default_initializer(rust_type)};\n"
                )
            else:
                return f"{indent_str}{binding_keyword} {binding_name}: {rust_type};\n"

        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)

        elif isinstance(stmt, BlockNode):
            return (
                f"{indent_str}{self.generate_block_expression(stmt, indent=indent)}\n"
            )

        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"

        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)

        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)

        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)

        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)

        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)

        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)

        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)

        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                # Handle both single values and lists
                if isinstance(stmt.value, list):
                    # Multiple return values (tuple)
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    return f"{indent_str}return ({values});\n"
                else:
                    # Single return value
                    return_expr = self.generate_return_expression_with_type(
                        stmt.value, self.current_return_type
                    )
                    return_expr = self.normalize_assignment_rhs(
                        self.current_return_type, stmt.value, return_expr, "="
                    )
                    if self.expression_contains_block_node(stmt.value):
                        return_expr = self.indent_multiline_expression(
                            return_expr,
                            indent,
                        )
                    return f"{indent_str}return {return_expr};\n"
            else:
                # Void return
                return f"{indent_str}return;\n"

        elif isinstance(stmt, BreakNode):
            context = self.active_do_while_context()
            if context:
                break_flag = context["break_flag"]
                return f"{indent_str}{break_flag} = true;\n{indent_str}break;\n"
            return f"{indent_str}break;\n"

        elif isinstance(stmt, ContinueNode):
            if self.active_do_while_context():
                return f"{indent_str}break;\n"
            context = self.active_for_context()
            if context:
                update = context["update"]
                return f"{indent_str}{update};\n{indent_str}continue;\n"
            return f"{indent_str}continue;\n"

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            if hasattr(stmt, "expression"):
                expression = self.generate_expression(stmt.expression)
                if getattr(stmt, "is_tail_expression", False):
                    return f"{indent_str}{expression}\n"
                return f"{indent_str}{expression};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode as statement - likely misclassified
            return f"{indent_str}// Unhandled ArrayAccessNode: {stmt}\n"

        elif isinstance(stmt, SyncNode):
            return self.generate_sync_statement(stmt, indent)

        else:
            # Try to generate as expression
            expr_result = self.generate_expression(stmt)
            if expr_result and expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return f"{indent_str}// Unhandled statement: {type(stmt).__name__}\n"

    def generate_sync_statement(self, stmt, indent):
        """Render a SyncNode as a Rust synchronization call."""
        indent_str = "    " * indent
        sync_type = getattr(stmt, "sync_type", "")
        rust_name = self.sync_map.get(sync_type)
        if rust_name is None:
            return f"{indent_str}/* unsupported sync: {sync_type} */\n"
        return f"{indent_str}{rust_name}();\n"

    def local_value_default_initializer(self, rust_type):
        scalar_initializer = self.rust_scalar_default_literal(rust_type)
        if scalar_initializer is not None:
            return scalar_initializer
        if self.is_rust_shader_pod_value_type(rust_type):
            return "Default::default()"
        return None

    def generate_wave_op_expression(self, expr):
        """Render a WaveOpNode as a Rust subgroup intrinsic call."""
        operation = getattr(expr, "operation", "")
        rust_name = self.wave_op_map.get(operation)
        if rust_name is None:
            return f"/* unsupported wave op: {operation} */"
        args = ", ".join(self.generate_expression(arg) for arg in expr.arguments or [])
        return f"{rust_name}({args})"

    def wave_op_result_type(self, operation, arguments):
        """Infer the CrossGL result type for a Rust subgroup helper."""
        helper_name = self.wave_op_map.get(operation, operation)
        arg_types = [self.expression_result_type(arg) for arg in arguments or []]
        return self.wave_helper_result_type(helper_name, arg_types)

    def wave_helper_result_type(self, helper_name, arg_types):
        helper_name = str(helper_name or "")
        if helper_name in {
            "subgroup_size",
            "subgroup_invocation_id",
            "subgroup_ballot_bit_count",
            "subgroup_exclusive_ballot_bit_count",
            "subgroup_partitioned_ballot_bit_count",
        }:
            return "uint"
        if helper_name in {
            "subgroup_elect",
            "subgroup_any",
            "subgroup_all",
            "subgroup_all_equal",
            "subgroup_quad_any",
            "subgroup_quad_all",
        }:
            return "bool"
        if helper_name in {"subgroup_ballot", "subgroup_match"}:
            return "uvec4"
        if helper_name in {
            "subgroup_add",
            "subgroup_mul",
            "subgroup_min",
            "subgroup_max",
            "subgroup_and",
            "subgroup_or",
            "subgroup_xor",
            "subgroup_broadcast_first",
            "subgroup_broadcast",
            "subgroup_exclusive_add",
            "subgroup_exclusive_mul",
            "subgroup_partitioned_add",
            "subgroup_partitioned_mul",
            "subgroup_partitioned_and",
            "subgroup_partitioned_or",
            "subgroup_partitioned_xor",
            "subgroup_quad_broadcast",
            "subgroup_quad_swap_horizontal",
            "subgroup_quad_swap_vertical",
            "subgroup_quad_swap_diagonal",
            "subgroup_shuffle",
            "subgroup_shuffle_xor",
        }:
            return arg_types[0] if arg_types else None
        return None

    def local_let_keyword(self, stmt):
        """Return `let` or `let mut` for a generated local binding."""
        qualifiers = {str(q) for q in getattr(stmt, "qualifiers", []) or []}
        if (
            "mut" in qualifiers
            or getattr(stmt, "name", None) in self.current_mutated_names
        ):
            return "let mut"
        return "let"

    def generate_increment_initializer_declaration(
        self,
        stmt,
        initial_value,
        vtype,
        indent,
    ):
        if not isinstance(initial_value, UnaryOpNode):
            return None

        op = getattr(initial_value, "operator", getattr(initial_value, "op", ""))
        op = self.map_operator(op)
        if op not in {"++", "--"}:
            return None

        operand = self.generate_expression(getattr(initial_value, "operand", ""))
        assignment_op = "+=" if op == "++" else "-="
        increment = self.rust_increment_literal(
            self.expression_result_type(getattr(initial_value, "operand", None))
        )
        update = f"{'    ' * indent}{operand} {assignment_op} {increment};\n"
        declaration = (
            f"{'    ' * indent}{self.local_let_keyword(stmt)} "
            f"{self.rust_identifier(stmt.name)}: "
            f"{self.map_type(vtype)} = {operand};\n"
        )
        is_postfix = getattr(
            initial_value,
            "is_postfix",
            getattr(initial_value, "postfix", False),
        )
        if is_postfix:
            return declaration + update
        return update + declaration

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        arm_indent = "    " * (indent + 1)
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}match {expression} {{\n"
        has_default = False
        for case in getattr(node, "cases", []) or []:
            if not isinstance(case, CaseNode):
                continue
            value = getattr(case, "value", None)
            pattern = "_" if value is None else self.generate_expression(value)
            has_default = has_default or value is None
            code += f"{arm_indent}{pattern} => {{\n"
            code += self.generate_switch_case_body(
                getattr(case, "statements", []), indent + 2
            )
            code += f"{arm_indent}}},\n"

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            has_default = True
            code += f"{arm_indent}_ => {{\n"
            code += self.generate_switch_case_body(default_case, indent + 2)
            code += f"{arm_indent}}},\n"
        elif not has_default:
            code += f"{arm_indent}_ => {{}},\n"

        code += f"{indent_str}}}\n"
        return code

    def generate_match(
        self,
        node,
        indent,
        expression_target_type=None,
        return_context=False,
    ):
        indent_str = "    " * indent
        arm_indent = "    " * (indent + 1)
        subject_expr = getattr(node, "expression", "")
        subject_type = self.expression_result_type(subject_expr)
        expression = None
        if return_context or self.match_statement_terminates_all_arms(node):
            expression = self.generate_direct_return_move_expression(subject_expr)
        if expression is None:
            expression = self.generate_expression(subject_expr)

        code = f"{indent_str}match {expression} {{\n"
        has_unconditional_wildcard = False
        for arm in getattr(node, "arms", []) or []:
            if not self.is_supported_match_arm(arm):
                raise ValueError("Unsupported match arm for Rust codegen")

            pattern = getattr(arm, "pattern", None)
            guard = getattr(arm, "guard", None)
            body = getattr(arm, "body", [])
            unused_bindings = self.unused_match_pattern_binding_names(
                pattern,
                guard,
                body,
            )
            mutable_bindings = self.mutable_match_pattern_binding_names(pattern, body)
            arm_pattern = self.generate_match_pattern(
                pattern,
                unused_bindings,
                mutable_bindings,
            )
            saved_variable_types = self.variable_types.copy()
            saved_local_variable_names = self.local_variable_names.copy()
            try:
                self.register_match_pattern_bindings(pattern, subject_type)

                has_unconditional_wildcard = (
                    has_unconditional_wildcard
                    or guard is None
                    and isinstance(pattern, WildcardPatternNode)
                )
                if guard is not None:
                    arm_pattern = f"{arm_pattern} if {self.generate_expression(guard)}"

                code += f"{arm_indent}{arm_pattern} => {{\n"
                code += self.generate_match_arm_body(
                    body,
                    indent + 2,
                    expression_target_type=expression_target_type,
                    return_context=return_context,
                )
                code += f"{arm_indent}}},\n"
            finally:
                self.variable_types = saved_variable_types
                self.local_variable_names = saved_local_variable_names

        if not has_unconditional_wildcard and not self.match_arms_are_exhaustive(
            node, subject_type
        ):
            code += f"{arm_indent}_ => unreachable!(),\n"

        code += f"{indent_str}}}\n"
        return code

    def match_statement_terminates_all_arms(self, node):
        arms = getattr(node, "arms", []) or []
        return bool(arms) and all(
            self.match_arm_body_terminates(getattr(arm, "body", None)) for arm in arms
        )

    def match_arm_body_terminates(self, body):
        return self.statement_body_terminates(body)

    def statement_body_terminates(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False

        tail = statements[-1]
        return self.statement_terminates(tail)

    def statement_terminates(self, tail):
        if isinstance(tail, ReturnNode):
            return True
        if isinstance(tail, BlockNode):
            return self.statement_body_terminates(tail)
        if isinstance(tail, LoopNode):
            return self.loop_statement_terminates(tail)
        if isinstance(tail, MatchNode):
            return self.match_statement_terminates_all_arms(tail)
        if isinstance(tail, IfNode):
            return self.if_statement_terminates_all_paths(tail)
        return False

    def loop_statement_terminates(self, node):
        body = getattr(node, "body", None)
        if self.loop_body_contains_control_flow(body):
            return False
        return self.statement_body_terminates(body)

    def loop_body_contains_control_flow(self, body):
        for statement in self.statement_list(body):
            if isinstance(statement, (BreakNode, ContinueNode)):
                return True
            if isinstance(statement, IfNode):
                if self.loop_body_contains_control_flow(
                    getattr(
                        statement, "then_branch", getattr(statement, "if_body", None)
                    )
                ):
                    return True
                if self.loop_body_contains_control_flow(
                    getattr(
                        statement, "else_branch", getattr(statement, "else_body", None)
                    )
                ):
                    return True
            elif isinstance(statement, MatchNode):
                for arm in getattr(statement, "arms", []) or []:
                    if self.loop_body_contains_control_flow(getattr(arm, "body", None)):
                        return True
            elif isinstance(statement, BlockNode):
                if self.loop_body_contains_control_flow(statement):
                    return True
        return False

    def if_statement_terminates_all_paths(self, node):
        then_branch = getattr(node, "then_branch", getattr(node, "if_body", None))
        else_branch = getattr(node, "else_branch", getattr(node, "else_body", None))
        if else_branch is None:
            return False
        return self.statement_body_terminates(
            then_branch
        ) and self.statement_body_terminates(else_branch)

    def generate_match_expression(
        self,
        node,
        indent=0,
        target_type=None,
        return_context=False,
    ):
        """Render a match node where Rust expects an expression value."""
        code = self.generate_match(
            node,
            indent,
            expression_target_type=target_type,
            return_context=return_context,
        ).rstrip("\n")
        indent_str = "    " * indent
        if indent_str and code.startswith(indent_str):
            code = code[len(indent_str) :]
        return code

    def register_match_pattern_bindings(self, pattern, subject_type):
        """Track types introduced by a match pattern while rendering its arm."""
        if isinstance(pattern, IdentifierPatternNode):
            if pattern.name.isidentifier() and subject_type is not None:
                self.register_variable_type(
                    pattern.name,
                    subject_type,
                    scope="local",
                )
            return

        if isinstance(pattern, StructPatternNode):
            pattern_type = self.specialize_pattern_type(
                pattern.type_name,
                subject_type,
            )
            variant_name = self.match_pattern_variant_name_for_subject(
                pattern,
                subject_type,
            )
            variant_field_types = (
                self.resolve_enum_variant_field_types(subject_type, variant_name)
                if variant_name is not None
                else {}
            )
            for field_name, field_pattern in pattern.field_patterns.items():
                field_type = variant_field_types.get(field_name)
                if field_type is None:
                    field_type = self.resolve_struct_member_type(
                        pattern_type,
                        field_name,
                    )
                if field_type is None:
                    field_type = self.resolve_struct_member_type(
                        pattern.type_name,
                        field_name,
                    )
                self.register_match_pattern_bindings(field_pattern, field_type)
            return

        if isinstance(pattern, ConstructorPatternNode):
            for index, arg_pattern in enumerate(pattern.arguments):
                arg_type = self.constructor_pattern_argument_type(
                    pattern.type_name,
                    subject_type,
                    index,
                )
                self.register_match_pattern_bindings(arg_pattern, arg_type)

    def specialize_pattern_type(self, pattern_type, subject_type):
        """Prefer the scrutinee's generic arguments for matching struct patterns."""
        if subject_type is None:
            return pattern_type
        subject_base, _ = self.generic_type_parts(subject_type)
        pattern_base, _ = self.generic_type_parts(pattern_type)
        if subject_base == pattern_base:
            return subject_type
        return pattern_type

    def constructor_pattern_argument_type(self, pattern_type, subject_type, index):
        """Infer payload types for common generic enum-like constructor patterns."""
        if subject_type is None:
            return None
        base, args = self.generic_type_parts(subject_type)
        variant_base, variant = self.split_variant_path(
            self.normalize_turbofish_type_name(pattern_type)
        )
        if variant_base is not None:
            variant_base, _variant_args = self.generic_type_parts(variant_base)
            if variant_base != base:
                return None
        payload_types = self.resolve_enum_variant_payload_types(subject_type, variant)
        if index < len(payload_types):
            return payload_types[index]
        if base == "Option" and variant == "Some" and index == 0 and args:
            return args[0]
        if base == "Result" and args:
            if variant == "Ok" and index == 0:
                return args[0]
            if variant == "Err" and index == 0 and len(args) > 1:
                return args[1]
        return None

    def is_supported_match_arm(self, arm):
        pattern = getattr(arm, "pattern", None)
        return isinstance(
            pattern,
            (
                LiteralPatternNode,
                WildcardPatternNode,
                IdentifierPatternNode,
                ConstructorPatternNode,
                StructPatternNode,
            ),
        )

    def match_arms_are_exhaustive(self, node, subject_type):
        """Return whether known bool/enum match arms already cover all cases."""
        arms = getattr(node, "arms", []) or []
        if self.match_arms_cover_bool(arms, subject_type):
            return True
        if self.match_arms_cover_struct(arms, subject_type):
            return True

        variants = self.enum_variants_for_type(subject_type)
        if not variants:
            return False

        covered_variants = set()
        conditional_variant_patterns = {variant: [] for variant in variants}
        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            pattern = getattr(arm, "pattern", None)
            variant_name = self.match_pattern_variant_name_for_subject(
                pattern,
                subject_type,
            )
            if variant_name is None:
                continue
            if self.is_unconditional_variant_pattern(pattern):
                covered_variants.add(variant_name)
            else:
                conditional_variant_patterns[variant_name].append(pattern)

        for variant in variants:
            if variant in covered_variants:
                continue
            if self.variant_patterns_cover_nested_enum_field(
                conditional_variant_patterns.get(variant, []),
                subject_type,
                variant,
            ):
                continue
            return False

        return True

    def match_arms_cover_bool(self, arms, subject_type):
        if self.normalize_scalar_type(subject_type) != "bool":
            return False

        covered = set()
        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            value = self.bool_match_pattern_value(getattr(arm, "pattern", None))
            if value is not None:
                covered.add(value)

        return covered == {True, False}

    def match_arms_cover_struct(self, arms, subject_type):
        subject_base, _generic_args = self.generic_type_parts(subject_type)
        member_types = self.struct_member_types.get(subject_base)
        if not member_types:
            return False

        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            pattern = getattr(arm, "pattern", None)
            if not isinstance(pattern, StructPatternNode):
                continue
            pattern_base, _pattern_args = self.generic_type_parts(pattern.type_name)
            if pattern_base != subject_base:
                continue
            if not self.is_unconditional_variant_pattern(pattern):
                continue
            if getattr(pattern, "has_rest", False) or set(
                pattern.field_patterns
            ) == set(member_types):
                return True

        return False

    def bool_match_pattern_value(self, pattern):
        if isinstance(pattern, LiteralPatternNode):
            value = getattr(pattern.literal, "value", None)
            if isinstance(value, bool):
                return value
        if isinstance(pattern, IdentifierPatternNode):
            if pattern.name == "true":
                return True
            if pattern.name == "false":
                return False
        return None

    def enum_variants_for_type(self, subject_type):
        if subject_type is None:
            return []
        base_type, _generic_args = self.generic_type_parts(subject_type)
        return self.enum_variant_names.get(base_type, [])

    def match_pattern_variant_name_for_subject(self, pattern, subject_type):
        variant_path = self.match_pattern_variant_path(pattern)
        if variant_path is None:
            return None

        subject_base, _generic_args = self.generic_type_parts(subject_type)
        variant_base, variant_name = self.split_variant_path(variant_path)
        if variant_base is not None and variant_base != subject_base:
            return None

        variants = self.enum_variants_for_type(subject_type)
        if variant_name not in variants:
            return None

        return variant_name

    def unconditional_match_variant_name(self, pattern, subject_type):
        variant_name = self.match_pattern_variant_name_for_subject(
            pattern,
            subject_type,
        )
        if variant_name is None:
            return None
        if not self.is_unconditional_variant_pattern(pattern):
            return None

        return variant_name

    def variant_patterns_cover_nested_enum_field(
        self,
        patterns,
        subject_type,
        variant_name,
    ):
        if not patterns:
            return False

        subject_base, _generic_args = self.generic_type_parts(subject_type)
        variant_fields = self.enum_variant_field_types.get(subject_base, {}).get(
            variant_name,
            {},
        )
        if not variant_fields:
            return False

        for field_name, field_type in variant_fields.items():
            nested_variants = self.enum_variants_for_type(field_type)
            if not nested_variants:
                continue

            covered_nested_variants = set()
            for pattern in patterns:
                if not isinstance(pattern, StructPatternNode):
                    continue
                if not getattr(pattern, "has_rest", False) and set(
                    pattern.field_patterns
                ) != set(variant_fields):
                    continue
                if not self.struct_pattern_other_fields_are_unconditional(
                    pattern,
                    field_name,
                ):
                    continue
                field_pattern = pattern.field_patterns.get(field_name)
                if field_pattern is None:
                    if getattr(pattern, "has_rest", False):
                        return True
                    continue
                if self.is_unconditional_payload_pattern(field_pattern):
                    return True
                nested_variant = self.unconditional_match_variant_name(
                    field_pattern,
                    field_type,
                )
                if nested_variant is not None:
                    covered_nested_variants.add(nested_variant)

            if set(nested_variants).issubset(covered_nested_variants):
                return True

        return False

    def struct_pattern_other_fields_are_unconditional(self, pattern, skipped_field):
        for field_name, field_pattern in pattern.field_patterns.items():
            if field_name == skipped_field:
                continue
            if not self.is_unconditional_payload_pattern(field_pattern):
                return False
        return True

    def match_pattern_variant_path(self, pattern):
        if isinstance(pattern, ConstructorPatternNode):
            return pattern.type_name
        if isinstance(pattern, StructPatternNode):
            return pattern.type_name
        if isinstance(pattern, IdentifierPatternNode):
            return pattern.name
        return None

    def split_variant_path(self, path):
        path = str(path)
        if "::" not in path:
            return None, path
        base, variant_name = path.rsplit("::", 1)
        return base, variant_name

    def is_unconditional_variant_pattern(self, pattern):
        if isinstance(pattern, IdentifierPatternNode):
            return "::" in pattern.name
        if isinstance(pattern, ConstructorPatternNode):
            return all(
                self.is_unconditional_payload_pattern(argument)
                for argument in getattr(pattern, "arguments", []) or []
            )
        if isinstance(pattern, StructPatternNode):
            return all(
                self.is_unconditional_payload_pattern(field_pattern)
                for field_pattern in getattr(pattern, "field_patterns", {}).values()
            )
        return False

    def is_unconditional_payload_pattern(self, pattern):
        if isinstance(pattern, WildcardPatternNode):
            return True
        if isinstance(pattern, IdentifierPatternNode):
            return "::" not in pattern.name
        return False

    def unused_match_pattern_binding_names(self, pattern, guard, body):
        """Return pattern bindings that are not referenced by the arm."""
        binding_names = self.match_pattern_binding_names(pattern)
        if not binding_names:
            return set()

        used_names = self.collect_referenced_names([guard, body])
        return binding_names - used_names

    def mutable_match_pattern_binding_names(self, pattern, body):
        """Return pattern bindings that are reassigned inside the arm body."""
        binding_names = self.match_pattern_binding_names(pattern)
        if not binding_names:
            return set()

        return binding_names & self.collect_mutated_binding_names(body)

    def match_pattern_binding_names(self, pattern):
        """Collect identifiers introduced by a match pattern."""
        if isinstance(pattern, IdentifierPatternNode):
            if self.is_match_binding_identifier(pattern.name):
                return {pattern.name}
            return set()
        if isinstance(pattern, ConstructorPatternNode):
            names = set()
            for argument in getattr(pattern, "arguments", []) or []:
                names.update(self.match_pattern_binding_names(argument))
            return names
        if isinstance(pattern, StructPatternNode):
            names = set()
            for field_pattern in getattr(pattern, "field_patterns", {}).values():
                names.update(self.match_pattern_binding_names(field_pattern))
            return names
        return set()

    def is_match_binding_identifier(self, name):
        """Return whether a pattern identifier introduces a Rust binding."""
        return (
            isinstance(name, str)
            and name.isidentifier()
            and name not in {"true", "false"}
        )

    def collect_referenced_names(self, node):
        """Collect identifier references from statements and expressions."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if hasattr(current, "statements"):
                collect(current.statements)
                return

            if isinstance(current, IdentifierNode):
                names.add(current.name)
                return

            if isinstance(current, str):
                if current.isidentifier():
                    names.add(current)
                return

            if isinstance(current, VariableNode):
                collect(getattr(current, "initial_value", None))
                return

            if isinstance(current, AssignmentNode):
                collect(current.target)
                collect(current.value)
                return

            if isinstance(current, ExpressionStatementNode):
                collect(current.expression)
                return

            if isinstance(current, ReturnNode):
                collect(current.value)
                return

            if isinstance(current, IfNode):
                collect(current.condition)
                collect(current.then_branch)
                collect(current.else_branch)
                return

            if isinstance(current, ForNode):
                collect(current.init)
                collect(current.condition)
                collect(current.update)
                collect(current.body)
                return

            if isinstance(current, ForInNode):
                collect(current.iterable)
                collect(current.body)
                return

            if isinstance(current, (WhileNode, DoWhileNode)):
                collect(current.condition)
                collect(current.body)
                return

            if isinstance(current, LoopNode):
                collect(current.body)
                return

            if isinstance(current, MatchNode):
                collect(current.expression)
                for arm in getattr(current, "arms", []) or []:
                    collect(getattr(arm, "guard", None))
                    collect(getattr(arm, "body", None))
                return

            if isinstance(current, SwitchNode):
                collect(getattr(current, "expression", None))
                for case in getattr(current, "cases", []) or []:
                    collect(getattr(case, "value", None))
                    collect(getattr(case, "statements", []))
                collect(getattr(current, "default_case", None))
                return

            if isinstance(current, UnaryOpNode):
                collect(current.operand)
                return

            if isinstance(current, BinaryOpNode):
                collect(current.left)
                collect(current.right)
                return

            if isinstance(current, TernaryOpNode):
                collect(current.condition)
                collect(current.true_expr)
                collect(current.false_expr)
                return

            if isinstance(current, FunctionCallNode):
                collect(current.function)
                collect(getattr(current, "arguments", getattr(current, "args", [])))
                return

            if isinstance(current, MemberAccessNode):
                collect(current.object_expr)
                return

            if isinstance(current, PointerAccessNode):
                collect(current.pointer_expr)
                return

            if isinstance(current, ArrayAccessNode):
                collect(current.array_expr)
                collect(current.index_expr)
                return

            if isinstance(current, RangeNode):
                collect(current.start)
                collect(current.end)
                return

            if isinstance(current, ConstructorNode):
                collect(getattr(current, "arguments", []))
                collect(list((getattr(current, "named_arguments", {}) or {}).values()))
                return

            if isinstance(current, ArrayLiteralNode):
                collect(getattr(current, "elements", []))

        collect(node)
        return names

    def format_match_binding_identifier(
        self,
        name,
        unused_bindings,
        mutable_bindings=None,
    ):
        """Prefix unused pattern bindings to suppress Rust warnings."""
        mutable_bindings = mutable_bindings or set()
        binding_name = name
        if name in unused_bindings and not name.startswith("_"):
            binding_name = f"_{name}"
        else:
            binding_name = self.rust_identifier(binding_name)
        if name in mutable_bindings:
            return f"mut {binding_name}"
        return binding_name

    def generate_match_pattern(
        self,
        pattern,
        unused_bindings=None,
        mutable_bindings=None,
    ):
        unused_bindings = unused_bindings or set()
        mutable_bindings = mutable_bindings or set()
        if isinstance(pattern, WildcardPatternNode):
            return "_"
        if isinstance(pattern, LiteralPatternNode):
            return self.generate_expression(pattern.literal)
        if isinstance(pattern, IdentifierPatternNode):
            if self.is_match_binding_identifier(pattern.name):
                return self.format_match_binding_identifier(
                    pattern.name,
                    unused_bindings,
                    mutable_bindings,
                )
            return self.rust_identifier(pattern.name)
        if isinstance(pattern, ConstructorPatternNode):
            if not pattern.arguments:
                return pattern.type_name
            args = ", ".join(
                self.generate_match_pattern(arg, unused_bindings, mutable_bindings)
                for arg in pattern.arguments
            )
            return f"{pattern.type_name}({args})"
        if isinstance(pattern, StructPatternNode):
            fields = []
            for field_name, field_pattern in pattern.field_patterns.items():
                field_pattern_code = self.generate_match_pattern(
                    field_pattern,
                    unused_bindings,
                    mutable_bindings,
                )
                field_ident = self.rust_identifier(field_name)
                if (
                    isinstance(field_pattern, IdentifierPatternNode)
                    and field_pattern.name == field_name
                    and field_pattern_code == field_ident
                ):
                    fields.append(field_ident)
                else:
                    fields.append(f"{field_ident}: {field_pattern_code}")
            if getattr(pattern, "has_rest", False):
                fields.append("..")
            return f"{pattern.type_name} {{ {', '.join(fields)} }}"
        raise ValueError(f"Unsupported match pattern for Rust codegen: {pattern!r}")

    def generate_switch_case_body(self, body, indent):
        statements = self.statement_list(body)
        code = ""
        for stmt in statements:
            if isinstance(stmt, BreakNode):
                continue
            code += self.generate_statement(stmt, indent)
        return code

    def generate_match_arm_body(
        self,
        body,
        indent,
        expression_target_type=None,
        return_context=False,
    ):
        statements = self.statement_list(body)
        expression_context = return_context or expression_target_type is not None
        if not expression_context or not statements:
            return "".join(self.generate_statement(stmt, indent) for stmt in statements)

        code = "".join(
            self.generate_statement(stmt, indent) for stmt in statements[:-1]
        )
        tail = statements[-1]
        tail_expression = None
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail,
            "is_tail_expression",
            False,
        ):
            tail_expression = tail.expression
        elif isinstance(
            tail,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                BlockNode,
                ConstructorNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ) or isinstance(tail, str):
            tail_expression = tail

        if tail_expression is None:
            return code + self.generate_statement(tail, indent)

        if return_context:
            tail_value = self.generate_return_branch_expression_with_type(
                tail_expression,
                expression_target_type,
            )
        else:
            tail_value = self.generate_expression_with_type(
                tail_expression,
                expression_target_type,
            )
        tail_value = self.normalize_assignment_rhs(
            expression_target_type,
            tail_expression,
            tail_value,
            "=",
        )
        return code + f"{'    ' * indent}{tail_value}\n"

    def statement_list(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def block_tail_expression(self, tail):
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail,
            "is_tail_expression",
            False,
        ):
            return tail.expression
        if isinstance(
            tail,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                BlockNode,
                ConstructorNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ) or isinstance(tail, str):
            return tail
        return None

    def indent_multiline_expression(self, expression, indent, indent_first=False):
        if not isinstance(expression, str):
            return expression

        prefix = "    " * indent
        if "\n" not in expression:
            return f"{prefix}{expression}" if indent_first else expression

        lines = expression.split("\n")
        start_index = 0 if indent_first else 1
        for index in range(start_index, len(lines)):
            if lines[index]:
                lines[index] = f"{prefix}{lines[index]}"
        return "\n".join(lines)

    def expression_contains_block_node(self, node):
        if node is None:
            return False
        if isinstance(node, BlockNode):
            return True
        if isinstance(node, (list, tuple)):
            return any(self.expression_contains_block_node(item) for item in node)
        if isinstance(node, dict):
            return any(
                self.expression_contains_block_node(item) for item in node.values()
            )

        child_attrs = (
            "expression",
            "value",
            "initial_value",
            "target",
            "left",
            "right",
            "condition",
            "then_branch",
            "else_branch",
            "body",
            "guard",
            "operand",
            "object_expr",
            "object",
            "array_expr",
            "array",
            "index_expr",
            "index",
            "function",
            "arguments",
            "args",
            "named_arguments",
            "elements",
            "true_expr",
            "false_expr",
            "arms",
            "statements",
        )
        return any(
            self.expression_contains_block_node(getattr(node, attr, None))
            for attr in child_attrs
        )

    def generate_block_expression(
        self,
        node,
        target_type=None,
        return_context=False,
        indent=0,
    ):
        statements = self.statement_list(node)
        explicit_tail = getattr(node, "expression", None)
        if explicit_tail is not None:
            prefix_statements = statements
            tail_expression = explicit_tail
        elif statements:
            prefix_statements = statements[:-1]
            tail_expression = self.block_tail_expression(statements[-1])
        else:
            prefix_statements = []
            tail_expression = None

        indent_str = "    " * indent
        body_indent = indent + 1
        code = "{\n"

        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        try:
            for statement in prefix_statements:
                code += self.generate_statement(statement, body_indent)

            if tail_expression is not None:
                if return_context:
                    tail_value = self.generate_return_branch_expression_with_type(
                        tail_expression,
                        target_type,
                    )
                else:
                    tail_value = self.generate_expression_with_type(
                        tail_expression,
                        target_type,
                    )
                tail_value = self.normalize_assignment_rhs(
                    target_type,
                    tail_expression,
                    tail_value,
                    "=",
                )
                code += (
                    self.indent_multiline_expression(
                        tail_value,
                        body_indent,
                        indent_first=True,
                    )
                    + "\n"
                )
            elif statements:
                code += self.generate_statement(statements[-1], body_indent)
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

        code += f"{indent_str}}}"
        return code

    def active_do_while_context(self):
        if not self.do_while_contexts:
            return None
        context = self.do_while_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def active_for_context(self):
        if not self.for_contexts:
            return None
        context = self.for_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def get_variable_type(self, node):
        if hasattr(node, "var_type"):
            vtype = node.var_type
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            return None

        if self.is_inferred_declaration_type(vtype):
            return None
        type_name = self.type_name_string(vtype)
        return self.resource_type_with_attributes(type_name, node)

    def is_inferred_declaration_type(self, type_name):
        if type_name is None:
            return True
        type_name = self.type_name_string(type_name)
        return type_name.strip() in {"", "None", "auto"}

    def variable_declaration_type(self, node, initial_value=None):
        declared_type = self.get_variable_type(node)
        if declared_type is not None:
            return declared_type

        inferred_type = self.expression_result_type(initial_value)
        if inferred_type is not None:
            return inferred_type
        return "float"

    def statement_body_terminates_inner_loop(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type_name = self.convert_type_node_to_string(node.element_type)
        element_type = self.map_type(element_type_name)
        size = get_array_size_from_node(node)

        if size is None:
            rust_type = f"Vec<{element_type}>"
            source_type = f"{element_type_name}[]"
        else:
            rust_type = f"[{element_type}; {size}]"
            source_type = f"{element_type_name}[{size}]"

        self.register_variable_type(node.name, source_type, scope="local")
        initializer = self.local_array_default_initializer(rust_type)
        return (
            f"{indent_str}{self.local_let_keyword(node)} "
            f"{self.rust_identifier(node.name)}: "
            f"{rust_type} = {initializer};\n"
        )

    def generate_expression_with_type(self, expr, target_type, static_context=False):
        if self.reference_type_parts_for_type(target_type) is not None:
            if isinstance(expr, TernaryOpNode):
                return self.generate_ternary_expression(expr, target_type)
            if isinstance(expr, MatchNode):
                return self.generate_match_expression(expr, target_type=target_type)
            if isinstance(expr, BlockNode):
                return self.generate_block_expression(expr, target_type=target_type)
            generated = self.generate_borrow_operand_expression(expr)
            return self.normalize_reference_typed_expression(
                expr,
                generated,
                target_type,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(expr, target_type=target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(expr, target_type=target_type)
        if isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(
                expr, target_type, static_context=static_context
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_constructor_expression(expr, target_type)
        if isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, target_type)
        if isinstance(expr, TernaryOpNode):
            return self.generate_ternary_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_function_call_with_target(expr, target_type)
            if function_call is not None:
                return function_call
        return self.generate_expression(expr)

    def generate_return_expression_with_type(self, expr, target_type):
        direct_move = self.generate_direct_return_move_expression(expr)
        if direct_move is not None:
            return direct_move
        member_move = self.generate_return_member_access_expression(expr)
        if member_move is not None:
            return member_move
        if isinstance(expr, TernaryOpNode):
            return self.generate_return_ternary_expression(expr, target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_return_constructor_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                expr,
                target_type,
            )
            if function_call is not None:
                return function_call
        return self.generate_expression_with_type(expr, target_type)

    def generate_return_branch_expression_with_type(self, expr, target_type):
        direct_move = self.generate_direct_return_move_expression(expr)
        if direct_move is not None:
            return direct_move
        member_move = self.generate_return_member_access_expression(expr)
        if member_move is not None:
            return member_move
        if isinstance(expr, TernaryOpNode):
            return self.generate_return_ternary_expression(expr, target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_return_constructor_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                expr,
                target_type,
            )
            if function_call is not None:
                return function_call
        return self.generate_expression_with_type(expr, target_type)

    def generate_return_ternary_expression(self, expr, target_type):
        condition_expr = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")

        bool_vector_ternary = self.generate_bool_vector_ternary_expression(
            condition_expr,
            true_expr,
            false_expr,
            target_type,
        )
        if bool_vector_ternary is not None:
            return bool_vector_ternary

        condition = self.generate_condition_expression(condition_expr)
        branch_type = target_type or self.expression_result_type(expr)
        true_value = self.generate_return_branch_expression_with_type(
            true_expr,
            branch_type,
        )
        false_value = self.generate_return_branch_expression_with_type(
            false_expr,
            branch_type,
        )
        true_value = self.normalize_typed_expression_value(
            true_expr,
            true_value,
            branch_type,
        )
        false_value = self.normalize_typed_expression_value(
            false_expr,
            false_value,
            branch_type,
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_return_function_call_expression(
        self,
        expr,
        target_type=None,
        move_counts=None,
    ):
        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = self.function_call_name(func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        callee = func_name

        enum_variant = self.generate_enum_variant_call_with_typed_args(
            func_name,
            args,
            target_type=target_type,
            return_context=True,
            move_counts=move_counts,
        )
        if enum_variant is not None:
            return enum_variant

        if self.is_user_defined_function(func_name):
            param_types = self.user_function_param_types.get(func_name, [])
            generated_args = self.generate_return_call_args_with_types(
                args,
                param_types,
                func_name=func_name,
                move_counts=move_counts,
            )
            generated_args.extend(self.generate_captured_function_call_args(func_name))
            return f"{self.rust_callable_name(callee)}({', '.join(generated_args)})"

        struct_constructor = self.generate_return_struct_new_call_with_typed_args(
            func_name,
            args,
            move_counts=move_counts,
        )
        if struct_constructor is not None:
            return struct_constructor

        return None

    def generate_return_constructor_expression(
        self,
        expr,
        target_type=None,
        move_counts=None,
    ):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        rust_type = self.map_type(type_name)

        named_arguments = getattr(expr, "named_arguments", {}) or {}
        if named_arguments:
            variant_info = self.enum_variant_struct_constructor_info(
                type_name,
                target_type,
            )
            if variant_info is not None:
                enum_type, variant_name, field_types = variant_info
                constructor_path = self.enum_variant_call_path(enum_type, variant_name)
                fields = self.generate_return_named_constructor_fields(
                    named_arguments,
                    lambda name: field_types.get(name),
                    move_counts=move_counts,
                )
                return f"{constructor_path} {{ {', '.join(fields)} }}"

            constructor_path = self.rust_constructor_path(rust_type)
            fields = self.generate_return_named_constructor_fields(
                named_arguments,
                lambda name: self.resolve_struct_member_type(type_name, name),
                move_counts=move_counts,
            )
            return f"{constructor_path} {{ {', '.join(fields)} }}"

        arguments = getattr(expr, "arguments", []) or []
        member_types = self.resolve_struct_positional_member_types(type_name)
        args = self.generate_return_call_args_with_types(
            arguments,
            member_types,
            move_counts=move_counts,
        )
        return f"{self.rust_constructor_path(rust_type)}::new({', '.join(args)})"

    def generate_return_struct_new_call_with_typed_args(
        self,
        func_name,
        args,
        move_counts=None,
    ):
        type_name = self.struct_new_call_type_name(func_name)
        if type_name is None:
            return None

        member_types = self.resolve_struct_positional_member_types(type_name)
        if not member_types:
            return None

        generated_args = self.generate_return_call_args_with_types(
            args,
            member_types,
            move_counts=move_counts,
        )
        return f"{self.struct_new_call_path(type_name)}({', '.join(generated_args)})"

    def generate_return_named_constructor_fields(
        self,
        named_arguments,
        field_type_for_name,
        move_counts=None,
    ):
        argument_values = list(named_arguments.values())
        local_move_counts = (
            move_counts
            if move_counts is not None
            else self.return_move_place_counts(argument_values)
        )
        later_roots = self.return_call_arg_later_roots(argument_values)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(
            argument_values
        )
        fields = []
        for index, (name, value) in enumerate(named_arguments.items()):
            field_type = field_type_for_name(name)
            field_value = self.generate_with_lambda_capture_move_blocks(
                value,
                later_roots[index],
                lambda value=value, field_type=field_type: (
                    self.generate_return_argument_with_type(
                        value,
                        field_type,
                        local_move_counts,
                    )
                ),
                lambda_conflict_roots[index],
            )
            fields.append(f"{self.rust_identifier(name)}: {field_value}")
        return fields

    def generate_enum_variant_call_with_typed_args(
        self,
        func_name,
        args,
        target_type=None,
        return_context=False,
        move_counts=None,
    ):
        variant_info = self.enum_variant_call_info(func_name, target_type)
        if variant_info is None:
            return None

        enum_type, variant_name, payload_types = variant_info
        if len(args) == 0 and not payload_types:
            return self.enum_variant_call_path(enum_type, variant_name)
        if len(args) > 0 and not payload_types:
            return None

        if return_context:
            generated_args = self.generate_return_call_args_with_types(
                args,
                payload_types,
                move_counts=move_counts,
            )
        else:
            generated_args = self.generate_constructor_call_args_with_types(
                args,
                payload_types,
            )

        return (
            f"{self.enum_variant_call_path(enum_type, variant_name)}"
            f"({', '.join(generated_args)})"
        )

    def generate_return_call_args_with_types(
        self,
        args,
        target_types,
        func_name=None,
        move_counts=None,
    ):
        if move_counts is None:
            move_counts = self.return_move_place_counts(args)
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            target_type = target_types[index] if index < len(target_types) else None
            if func_name is not None:
                target_type = self.user_function_argument_target_type(
                    func_name,
                    arg,
                    target_type,
                )
            generated_args.append(
                self.generate_with_lambda_capture_move_blocks(
                    arg,
                    later_roots[index],
                    lambda: self.generate_return_argument_with_type(
                        arg,
                        target_type,
                        move_counts,
                    ),
                    lambda_conflict_roots[index],
                )
            )
        return generated_args

    def generate_return_argument_with_type(self, arg, target_type, move_counts):
        move_key = self.return_move_place_key(arg)
        if move_key is not None and move_key in move_counts:
            move_counts[move_key] -= 1
            if move_counts[move_key] == 0:
                direct_move = self.generate_direct_return_move_expression(arg)
                if direct_move is not None:
                    return direct_move
                member_move = self.generate_return_member_access_expression(arg)
                if member_move is not None:
                    return self.normalize_typed_expression_value(
                        arg,
                        member_move,
                        target_type,
                    )

        if isinstance(arg, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                arg,
                target_type,
                move_counts=move_counts,
            )
            if function_call is not None:
                return self.normalize_typed_expression_value(
                    arg,
                    function_call,
                    target_type,
                )

        if isinstance(arg, ConstructorNode):
            constructor = self.generate_return_constructor_expression(
                arg,
                target_type,
                move_counts=move_counts,
            )
            return self.normalize_typed_expression_value(
                arg,
                constructor,
                target_type,
            )

        if isinstance(arg, BlockNode):
            arg_expr = self.generate_return_branch_expression_with_type(
                arg,
                target_type,
            )
        else:
            arg_expr = self.generate_expression_with_type(arg, target_type)
        return self.normalize_typed_expression_value(arg, arg_expr, target_type)

    def return_move_place_counts(self, args):
        records = []
        for index, arg in enumerate(args):
            records.extend(self.return_move_place_records(arg, (index,)))

        if not records:
            return {}

        eligible_keys = set()
        roots = {record["root"] for record in records}
        for root in roots:
            root_records = [record for record in records if record["root"] == root]
            blocking_records = [
                record
                for record in root_records
                if record["kind"] in {"indexed", "lambda_capture"}
            ]
            movable_records = [
                record
                for record in root_records
                if record["kind"] in {"identifier", "member"}
            ]
            if blocking_records and movable_records:
                continue
            root_records = movable_records
            if not root_records:
                continue
            direct_records = [
                record for record in root_records if record["kind"] == "identifier"
            ]
            member_records = [
                record for record in root_records if record["kind"] == "member"
            ]

            if direct_records and member_records:
                last_direct_index = max(record["index"] for record in direct_records)
                last_member_index = max(record["index"] for record in member_records)
                if last_direct_index > last_member_index:
                    eligible_keys.update(record["key"] for record in direct_records)
                else:
                    member_keys = {record["key"] for record in member_records}
                    for member_key in member_keys:
                        last_key_index = max(
                            record["index"]
                            for record in member_records
                            if record["key"] == member_key
                        )
                        if last_key_index > last_direct_index:
                            eligible_keys.add(member_key)
            else:
                eligible_keys.update(record["key"] for record in root_records)

        counts = {}
        for record in records:
            move_key = record["key"]
            if move_key not in eligible_keys:
                continue
            counts[move_key] = counts.get(move_key, 0) + 1
        return counts

    def return_call_arg_later_roots(self, args):
        records_by_index = []
        for index, arg in enumerate(args):
            records_by_index.append(self.return_move_place_records(arg, (index,)))

        later_roots_by_index = []
        later_roots = set()
        for records in reversed(records_by_index):
            later_roots_by_index.append(set(later_roots))
            later_roots.update(
                record["root"] for record in records if record["root"] is not None
            )

        return list(reversed(later_roots_by_index))

    def return_call_arg_lambda_conflict_roots(self, args):
        capture_roots_by_index = []
        capture_root_counts = {}
        for index, arg in enumerate(args):
            capture_roots = {
                record["root"]
                for record in self.return_move_place_records(arg, (index,))
                if record["kind"] == "lambda_capture" and record["root"] is not None
            }
            capture_roots_by_index.append(capture_roots)
            for root in capture_roots:
                capture_root_counts[root] = capture_root_counts.get(root, 0) + 1

        return [
            {root for root in capture_roots if capture_root_counts[root] > 1}
            for capture_roots in capture_roots_by_index
        ]

    def generate_with_lambda_capture_move_blocks(
        self,
        arg,
        later_roots,
        generate_arg,
        lambda_conflict_roots=None,
    ):
        capture_roots = self.return_argument_lambda_capture_root_names(arg)
        candidate_roots = capture_roots & (lambda_conflict_roots or set())
        if self.loop_depth > 0 and isinstance(arg, LambdaNode):
            candidate_roots |= capture_roots
        if self.nested_lambda_capture_preclone_depth > 0:
            candidate_roots |= capture_roots & self.return_move_blocked_roots
        preclone_places = self.lambda_preclone_capture_roots(
            arg,
            candidate_roots,
        )
        if preclone_places:
            return self.generate_with_precloned_lambda_captures(
                arg,
                preclone_places,
                generate_arg,
            )

        blocked_roots = capture_roots & later_roots
        if lambda_conflict_roots is not None:
            blocked_roots |= capture_roots & lambda_conflict_roots
        if not blocked_roots:
            return generate_arg()

        saved_blocked_roots = self.return_move_blocked_roots
        self.return_move_blocked_roots = saved_blocked_roots | blocked_roots
        enable_nested_preclone = not isinstance(arg, LambdaNode)
        if enable_nested_preclone:
            self.nested_lambda_capture_preclone_depth += 1
        try:
            return generate_arg()
        finally:
            if enable_nested_preclone:
                self.nested_lambda_capture_preclone_depth -= 1
            self.return_move_blocked_roots = saved_blocked_roots

    def lambda_preclone_capture_roots(self, arg, candidate_roots):
        if not isinstance(arg, LambdaNode) or not candidate_roots:
            return set()

        saved_blocked_roots = self.return_move_blocked_roots
        self.return_move_blocked_roots = saved_blocked_roots - set(candidate_roots)
        try:
            body_records = self.return_move_place_records(
                getattr(arg, "body", None),
                (),
            )
        finally:
            self.return_move_blocked_roots = saved_blocked_roots

        preclone_places = []
        for root in sorted(candidate_roots):
            if self.is_static_reference(root):
                continue
            root_type = self.variable_types.get(root)
            if root_type is None or self.type_is_copy_derivable(
                root_type,
                self.current_generic_param_names,
            ):
                continue

            root_records = [
                record
                for record in body_records
                if record["root"] == root and record["kind"] in {"identifier", "member"}
            ]
            if len(root_records) == 1:
                record = root_records[0]
                key = record["key"]
                if record["kind"] == "member":
                    key = key[1]
                place_type = self.return_place_key_type(key)
                if place_type is None or self.type_is_copy_derivable(
                    place_type,
                    self.current_generic_param_names,
                ):
                    continue
                preclone_places.append(
                    {
                        "key": key,
                        "root": root,
                        "type": place_type,
                    }
                )
        return preclone_places

    def generate_with_precloned_lambda_captures(
        self,
        arg,
        preclone_places,
        generate_arg,
    ):
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        aliases = {}
        place_aliases = {}
        bindings = []
        for place in preclone_places:
            key = place["key"]
            place_type = place["type"]
            place_value = self.return_place_key_expression(key)
            if place_type is None or place_value is None:
                continue
            alias = self.next_lambda_capture_temp_name(
                self.lambda_capture_place_label(key)
            )
            if key[0] == "identifier":
                aliases[place["root"]] = alias
            else:
                place_aliases[key] = alias
            self.register_variable_type(alias, place_type, scope="local")
            bindings.append(f"let {alias} = {self.clone_value_expression(place_value)}")

        if not aliases and not place_aliases:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names
            return generate_arg()

        self.lambda_capture_alias_stack.append(aliases)
        self.lambda_capture_place_alias_stack.append(place_aliases)
        self.force_move_lambda_depth += 1
        try:
            closure = generate_arg()
        finally:
            self.force_move_lambda_depth -= 1
            self.lambda_capture_place_alias_stack.pop()
            self.lambda_capture_alias_stack.pop()
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

        return f"{{ {'; '.join(bindings)}; {closure} }}"

    def lambda_capture_place_label(self, key):
        if key[0] == "identifier":
            return key[1]
        if key[0] == "member":
            return f"{self.lambda_capture_place_label(key[1])}_{key[2]}"
        return "capture"

    def lambda_capture_place_alias_name(self, key):
        if key is None:
            return None
        for aliases in reversed(self.lambda_capture_place_alias_stack):
            alias = aliases.get(key)
            if alias is not None:
                return alias
        return None

    def return_place_key_type(self, key):
        if key is None:
            return None
        if key[0] == "identifier":
            return self.variable_types.get(key[1])
        if key[0] == "member":
            object_type = self.return_place_key_type(key[1])
            if object_type is None:
                return None
            return self.resolve_struct_member_type(object_type, key[2])
        return None

    def return_place_key_expression(self, key):
        if key is None:
            return None
        if key[0] == "identifier":
            return self.lazy_static_identifier_expression(key[1])
        if key[0] == "member":
            object_expr = self.return_place_key_expression(key[1])
            if object_expr is None:
                return None
            return f"{object_expr}.{key[2]}"
        return None

    def lambda_capture_alias_name(self, name):
        for aliases in reversed(self.lambda_capture_alias_stack):
            alias = aliases.get(name)
            if alias is not None:
                return alias
        return None

    def return_argument_lambda_capture_root_names(self, expr):
        return {
            record["root"]
            for record in self.return_move_place_records(expr, ())
            if record["kind"] == "lambda_capture" and record["root"] is not None
        }

    def lambda_capture_root_names(self, expr):
        if not isinstance(expr, LambdaNode):
            return set()

        roots = set()
        for capture in getattr(expr, "captures", []) or []:
            if isinstance(capture, str):
                name = capture if capture.isidentifier() else None
            else:
                name = self.return_place_root_name(capture)
            if name is not None:
                roots.add(name)
        return roots

    def return_move_place_records(self, expr, index_path):
        if isinstance(expr, LambdaNode):
            return [
                {
                    "index": index_path + (capture_index,),
                    "key": ("lambda_capture", name),
                    "kind": "lambda_capture",
                    "root": name,
                }
                for capture_index, name in enumerate(
                    sorted(self.lambda_capture_root_names(expr))
                )
            ]

        if isinstance(expr, ArrayAccessNode):
            root_name = self.return_place_root_name(expr)
            if root_name is not None and root_name in self.variable_types:
                return [
                    {
                        "index": index_path,
                        "key": ("indexed", root_name),
                        "kind": "indexed",
                        "root": root_name,
                    }
                ]

        move_key = self.return_move_place_key(expr)
        if move_key is not None:
            kind = move_key[0]
            root_name = (
                move_key[1]
                if kind == "identifier"
                else self.return_place_root_name(expr)
            )
            return [
                {
                    "index": index_path,
                    "key": move_key,
                    "kind": kind,
                    "root": root_name,
                }
            ]

        child_args = self.return_consumed_child_arguments(expr)
        records = []
        for child_index, child in enumerate(child_args):
            records.extend(
                self.return_move_place_records(
                    child,
                    index_path + (child_index,),
                )
            )
        return records

    def return_consumed_child_arguments(self, expr):
        if isinstance(expr, ConstructorNode):
            named_arguments = getattr(expr, "named_arguments", {}) or {}
            if named_arguments:
                return list(named_arguments.values())
            return list(getattr(expr, "arguments", []) or [])

        if not isinstance(expr, FunctionCallNode):
            return []

        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = self.function_call_name(func_expr)
        if not isinstance(func_name, str):
            return []

        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        if self.is_user_defined_function(func_name):
            return args
        if self.enum_variant_call_info(func_name) is not None:
            return args

        type_name = self.struct_new_call_type_name(func_name)
        if type_name is not None and self.resolve_struct_positional_member_types(
            type_name
        ):
            return args

        return []

    def return_move_place_key(self, expr):
        name = self.simple_identifier_expression_name(expr)
        if name is not None and self.should_move_direct_return_identifier(name):
            return ("identifier", name)

        if self.should_move_member_return_place(expr):
            place_key = self.return_place_expression_key(expr)
            if place_key is not None:
                return ("member", place_key)

        return None

    def return_place_expression_key(self, expr):
        if isinstance(expr, IdentifierNode):
            return ("identifier", expr.name)
        if isinstance(expr, VariableNode):
            return ("identifier", expr.name)
        if isinstance(expr, str):
            return ("identifier", expr) if expr.isidentifier() else None
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_key = self.return_place_expression_key(object_expr)
            if object_key is None:
                return None
            member = getattr(expr, "member", None)
            return ("member", object_key, member)
        return None

    def generate_constructor_call_args_with_types(self, args, target_types):
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            target_type = target_types[index] if index < len(target_types) else None
            generated_args.append(
                self.generate_with_lambda_capture_move_blocks(
                    arg,
                    later_roots[index],
                    lambda arg=arg, target_type=target_type: (
                        self.generate_constructor_argument_with_type(
                            arg,
                            target_type,
                        )
                    ),
                    lambda_conflict_roots[index],
                )
            )
        return generated_args

    def generate_constructor_argument_with_type(self, arg, target_type):
        arg_expr = self.generate_expression_with_type(arg, target_type)
        return self.normalize_typed_expression_value(arg, arg_expr, target_type)

    def generate_return_member_access_expression(self, expr):
        place_alias = self.lambda_capture_place_alias_name(
            self.return_place_expression_key(expr)
        )
        if place_alias is not None:
            return self.lazy_static_identifier_expression(place_alias)

        if not self.should_move_member_return_place(expr):
            return None

        swizzle_components = self.member_swizzle_components(expr)
        if swizzle_components is not None:
            return None

        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        self.member_object_depth += 1
        try:
            obj = self.generate_expression(obj_expr)
        finally:
            self.member_object_depth -= 1
        obj = self.lazy_static_object_expression(obj_expr, obj)
        return f"{obj}.{self.rust_identifier(member)}"

    def should_move_member_return_place(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return False

        member_type = self.expression_result_type(expr)
        if member_type is None or self.type_is_copy_derivable(
            member_type,
            self.current_generic_param_names,
        ):
            return False

        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        if self.return_place_contains_index_access(object_expr):
            return False

        root_name = self.return_place_root_name(object_expr)
        if root_name is None or self.is_static_reference(root_name):
            return False
        if root_name in self.return_move_blocked_roots:
            return False

        return root_name in self.variable_types

    def return_place_contains_index_access(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.return_place_contains_index_access(object_expr)
        return False

    def return_place_root_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr if expr.isidentifier() else None
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.return_place_root_name(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.return_place_root_name(array_expr)
        return None

    def generate_direct_return_move_expression(self, expr):
        name = self.simple_identifier_expression_name(expr)
        if name is None:
            return None
        move_name = self.lambda_capture_alias_name(name) or name
        if not self.should_move_direct_return_identifier(move_name):
            return None
        return self.lazy_static_identifier_expression(move_name)

    def should_move_direct_return_identifier(self, name):
        if name in self.return_move_blocked_roots:
            return False
        if self.is_static_reference(name):
            return False
        value_type = self.variable_types.get(name)
        if value_type is None or self.type_is_copy_derivable(
            value_type,
            self.current_generic_param_names,
        ):
            return False
        return True

    def simple_identifier_expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str) and expr.isidentifier():
            return expr
        return None

    def generate_binary_expression(self, expr, target_type=None):
        left_expr = getattr(expr, "left", "")
        right_expr = getattr(expr, "right", "")
        op = getattr(expr, "operator", getattr(expr, "op", "+"))
        mapped_op = self.map_operator(op)

        left_type = self.expression_result_type(left_expr)
        right_type = self.expression_result_type(right_expr)
        bool_vector_logical = self.generate_bool_vector_logical_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if bool_vector_logical is not None:
            return bool_vector_logical

        if mapped_op in {"&&", "||"}:
            left = self.generate_condition_expression(left_expr)
            right = self.generate_condition_expression(right_expr)
            return f"({left} {mapped_op} {right})"

        vector_comparison = self.generate_vector_comparison_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if vector_comparison is not None:
            return vector_comparison

        vector_arithmetic = self.generate_vector_arithmetic_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if vector_arithmetic is not None:
            return vector_arithmetic

        matrix_multiply = self.generate_matrix_multiply_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if matrix_multiply is not None:
            return matrix_multiply

        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, mapped_op, target_type
        )
        if matrix_vector_plan is not None:
            matrix_vector = self.generate_matrix_vector_expression(
                left_expr,
                right_expr,
                left_type,
                right_type,
                matrix_vector_plan,
            )
            if matrix_vector is not None:
                return matrix_vector
            left = self.generate_binary_composite_operand(
                left_expr, left_type, matrix_vector_plan["left_target_type"]
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, matrix_vector_plan["right_target_type"]
            )
            return f"({left} {mapped_op} {right})"

        scalar_left_vector = self.generate_scalar_left_vector_binary_expression(
            left_expr, right_expr, left_type, right_type, mapped_op
        )
        if scalar_left_vector is not None:
            return scalar_left_vector

        composite_operand_type = self.binary_composite_operand_type(
            left_type, right_type, mapped_op
        )
        if composite_operand_type is not None:
            left = self.generate_binary_composite_operand(
                left_expr, left_type, composite_operand_type
            )
            right = self.generate_binary_composite_operand(
                right_expr, right_type, composite_operand_type
            )
            return f"({left} {mapped_op} {right})"

        operand_type = self.binary_scalar_operand_type(
            left_type,
            right_type,
            mapped_op,
        )
        left = self.generate_expression(left_expr)
        right = self.generate_expression(right_expr)
        if operand_type is not None:
            left = self.normalize_binary_scalar_operand(
                left_expr,
                left,
                left_type,
                operand_type,
            )
            right = self.normalize_binary_scalar_operand(
                right_expr,
                right,
                right_type,
                operand_type,
            )

        return f"({left} {mapped_op} {right})"

    def generate_matrix_multiply_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.matrix_multiply_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.matrix_argument_lane_expressions(
            left_expr,
            plan["left_matrix"],
            temp_bindings,
            plan["component_type"],
        )
        right_lanes = self.matrix_argument_lane_expressions(
            right_expr,
            plan["right_matrix"],
            temp_bindings,
            plan["component_type"],
        )

        left_cell = self.matrix_lane_accessor(left_lanes, plan["left_matrix"])
        right_cell = self.matrix_lane_accessor(right_lanes, plan["right_matrix"])
        result_lanes = []
        for column in range(plan["result_columns"]):
            for row in range(plan["result_rows"]):
                terms = [
                    f"({left_cell(k, row)} * {right_cell(column, k)})"
                    for k in range(plan["shared_size"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))

        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            result_lanes,
            temp_bindings,
        )

    def generate_matrix_vector_expression(
        self, left_expr, right_expr, left_type, right_type, plan
    ):
        if not self.rust_matrix_vector_requires_lanes(plan):
            return None

        temp_bindings = []
        matrix_expr = left_expr if plan["matrix_on_left"] else right_expr
        vector_expr = right_expr if plan["matrix_on_left"] else left_expr
        matrix_type = left_type if plan["matrix_on_left"] else right_type
        vector_type = right_type if plan["matrix_on_left"] else left_type
        matrix_info = self.matrix_type_info(matrix_type)
        vector_info = self.vector_type_info(vector_type)
        if matrix_info is None or vector_info is None:
            return None

        result_info = self.vector_type_info(plan["result_type"])
        if result_info is None:
            return None
        component_type = result_info["component_type"]

        matrix_lanes = self.matrix_argument_lane_expressions(
            matrix_expr,
            matrix_info,
            temp_bindings,
            component_type,
        )
        vector_lanes = self.vector_argument_lane_expressions(
            vector_expr,
            vector_info,
            temp_bindings,
            component_type,
        )
        matrix_cell = self.matrix_lane_accessor(matrix_lanes, matrix_info)

        result_lanes = []
        if plan["matrix_on_left"]:
            for row in range(matrix_info["rows"]):
                terms = [
                    f"({matrix_cell(column, row)} * {vector_lanes[column]})"
                    for column in range(matrix_info["columns"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))
        else:
            for column in range(matrix_info["columns"]):
                terms = [
                    f"({vector_lanes[row]} * {matrix_cell(column, row)})"
                    for row in range(matrix_info["rows"])
                ]
                result_lanes.append(self.sum_rust_terms(terms))

        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            result_lanes,
            temp_bindings,
        )

    def matrix_lane_accessor(self, lanes, matrix_info):
        rows = matrix_info["rows"]
        return lambda column, row: lanes[column * rows + row]

    def sum_rust_terms(self, terms):
        if not terms:
            return "0"
        expression = terms[0]
        for term in terms[1:]:
            expression = f"({expression} + {term})"
        return expression

    def generate_vector_arithmetic_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.vector_arithmetic_plan(left_type, right_type, operator)
        if plan is None or not self.rust_vector_arithmetic_requires_lanes(plan):
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr,
            right_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]),
            lanes,
            temp_bindings,
        )

    def generate_scalar_left_vector_binary_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        if operator not in {"+", "-", "*", "/", "%"}:
            return None

        left_scalar = self.normalize_scalar_type(left_type)
        right_vector = self.vector_type_info(right_type)
        if left_scalar is None or right_vector is None:
            return None

        result_type = self.vector_type_for_promoted_scalar(right_vector, left_scalar)
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        component_type = result_info["component_type"]
        if self.normalize_scalar_type(component_type) == "bool":
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            component_type,
            result_info["size"],
            temp_bindings,
        )
        right_lanes = self.vector_argument_lane_expressions(
            right_expr,
            right_vector,
            temp_bindings,
            component_type,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(result_type),
            lanes,
            temp_bindings,
        )

    def generate_binary_composite_operand(self, expr, source_type, composite_type):
        if self.vector_type_info(source_type) or self.matrix_type_info(source_type):
            operand = self.generate_expression_with_type(expr, composite_type)
            return self.normalize_typed_expression_value(expr, operand, composite_type)

        component_type = self.composite_component_type(composite_type)
        if component_type is not None and self.normalize_scalar_type(source_type):
            operand = self.generate_expression_with_type(expr, component_type)
            return self.normalize_scalar_assignment_value(
                expr, operand, source_type, component_type
            )

        return self.generate_expression(expr)

    def composite_component_type(self, composite_type):
        vector_info = self.vector_type_info(composite_type)
        if vector_info is not None:
            return vector_info["component_type"]

        matrix_info = self.matrix_type_info(composite_type)
        if matrix_info is not None:
            return matrix_info["component_type"]

        return None

    def generate_bool_vector_logical_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.bool_vector_logical_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr, left_type, "bool", plan["size"], temp_bindings
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr, right_type, "bool", plan["size"], temp_bindings
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def generate_bool_vector_not_expression(self, expr, operator):
        if operator != "!":
            return None

        vector_info = self.vector_type_info(self.expression_result_type(expr))
        if vector_info is None or vector_info["component_type"] != "bool":
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, vector_info, temp_bindings, "bool"
        )
        lanes = [f"(!{lane})" for lane in lanes[: vector_info["size"]]]
        result_type = self.vector_type_for_components("bool", vector_info["size"])
        return self.generate_constructor_call(
            self.map_type(result_type), lanes, temp_bindings
        )

    def generate_vector_comparison_expression(
        self, left_expr, right_expr, left_type, right_type, operator
    ):
        plan = self.vector_comparison_plan(left_type, right_type, operator)
        if plan is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_comparison_operand_lanes(
            left_expr,
            left_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        right_lanes = self.vector_comparison_operand_lanes(
            right_expr,
            right_type,
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"({left_lane} {operator} {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def vector_comparison_operand_lanes(
        self, expr, source_type, target_component_type, size, temp_bindings
    ):
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            return self.vector_argument_lane_expressions(
                expr, source_info, temp_bindings, target_component_type
            )

        scalar_expr = self.generate_expression_with_type(expr, target_component_type)
        scalar_expr = self.normalize_scalar_assignment_value(
            expr, scalar_expr, source_type, target_component_type
        )
        if not self.is_repeat_safe_expression(expr):
            temp_name = self.next_vector_arg_temp_name()
            temp_bindings.append((temp_name, scalar_expr))
            scalar_expr = temp_name
        return [scalar_expr] * size

    def generate_condition_expression(self, expr):
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "")))
            if op == "!":
                operand = getattr(expr, "operand", "")
                return f"!({self.generate_condition_expression(operand)})"

        condition = self.generate_expression(expr)
        return self.normalize_condition_expression(expr, condition)

    def normalize_condition_expression(self, expr, generated_expr):
        condition_type = self.normalize_scalar_type(self.expression_result_type(expr))
        if condition_type is None or condition_type == "bool":
            return generated_expr

        zero_literal = "0.0" if condition_type in {"f16", "f32", "f64"} else "0"
        return f"({generated_expr} != {zero_literal})"

    def generate_ternary_expression(self, expr, target_type=None):
        condition_expr = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")

        bool_vector_ternary = self.generate_bool_vector_ternary_expression(
            condition_expr, true_expr, false_expr, target_type
        )
        if bool_vector_ternary is not None:
            return bool_vector_ternary

        condition = self.generate_condition_expression(condition_expr)
        branch_type = target_type or self.expression_result_type(expr)

        true_value = self.generate_expression_with_type(true_expr, branch_type)
        false_value = self.generate_expression_with_type(false_expr, branch_type)
        true_value = self.normalize_typed_expression_value(
            true_expr, true_value, branch_type
        )
        false_value = self.normalize_typed_expression_value(
            false_expr, false_value, branch_type
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_bool_vector_ternary_expression(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        plan = self.bool_vector_ternary_plan(
            condition_expr, true_expr, false_expr, target_type
        )
        if plan is None:
            return None

        temp_bindings = []
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        condition_lanes = self.vector_argument_lane_expressions(
            condition_expr, condition_info, temp_bindings, "bool"
        )
        true_lanes = self.vector_comparison_operand_lanes(
            true_expr,
            self.expression_result_type(true_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        false_lanes = self.vector_comparison_operand_lanes(
            false_expr,
            self.expression_result_type(false_expr),
            plan["component_type"],
            plan["size"],
            temp_bindings,
        )
        lanes = [
            f"(if {condition_lane} {{ {true_lane} }} else {{ {false_lane} }})"
            for condition_lane, true_lane, false_lane in zip(
                condition_lanes, true_lanes, false_lanes
            )
        ]
        return self.generate_constructor_call(
            self.map_type(plan["result_type"]), lanes, temp_bindings
        )

    def is_array_type_name(self, type_name):
        return type_name is not None and "[" in str(type_name) and "]" in str(type_name)

    def c_array_type_parts(self, type_name):
        if type_name is None:
            return None, []
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return type_name, []

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        suffix = type_name[open_bracket:]
        sizes = []
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                return type_name, []

            size_text = suffix[1:close_bracket].strip()
            if not size_text:
                sizes.append(None)
            else:
                try:
                    sizes.append(int(size_text))
                except ValueError:
                    sizes.append(None)
            suffix = suffix[close_bracket + 1 :]

        if suffix:
            return type_name, []
        return base_type, sizes

    def format_c_array_type(self, base_type, sizes):
        suffix = "".join(f"[{'' if size is None else size}]" for size in sizes)
        return f"{base_type}{suffix}"

    def c_array_outer_element_type_and_size(self, type_name):
        base_type, sizes = self.c_array_type_parts(type_name)
        if not sizes:
            return None, None
        element_type = (
            self.format_c_array_type(base_type, sizes[1:])
            if len(sizes) > 1
            else base_type
        )
        return element_type, sizes[0]

    def generate_array_literal_expression(
        self, expr, target_type=None, static_context=False
    ):
        element_target_type = None
        target_size = None

        if self.is_array_type_name(target_type):
            element_target_type, target_size = self.c_array_outer_element_type_and_size(
                target_type
            )

        elements = []
        for element in expr.elements:
            if element_target_type is None:
                elements.append(self.generate_expression(element))
                continue

            element_expr = self.generate_expression_with_type(
                element,
                element_target_type,
                static_context=static_context,
            )
            elements.append(
                self.normalize_typed_expression_value(
                    element,
                    element_expr,
                    element_target_type,
                )
            )

        if element_target_type is not None:
            if target_size is None:
                return f"vec![{', '.join(elements)}]"

            elements = elements[:target_size]
            padding = self.rust_array_padding_expression(
                element_target_type, static_context=static_context
            )
            while len(elements) < target_size:
                elements.append(padding)

        return f"[{', '.join(elements)}]"

    def generate_constructor_expression(self, expr, target_type=None):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        rust_type = self.map_type(type_name)

        named_arguments = getattr(expr, "named_arguments", {}) or {}
        if named_arguments:
            variant_info = self.enum_variant_struct_constructor_info(
                type_name,
                target_type,
            )
            if variant_info is not None:
                enum_type, variant_name, field_types = variant_info
                constructor_path = self.enum_variant_call_path(enum_type, variant_name)
                fields = self.generate_constructor_named_fields(
                    named_arguments,
                    lambda name: field_types.get(name),
                )
                return f"{constructor_path} {{ {', '.join(fields)} }}"

            constructor_path = self.rust_constructor_path(rust_type)
            fields = self.generate_constructor_named_fields(
                named_arguments,
                lambda name: self.resolve_struct_member_type(type_name, name),
            )
            return f"{constructor_path} {{ {', '.join(fields)} }}"

        arguments = getattr(expr, "arguments", []) or []
        member_types = self.resolve_struct_positional_member_types(type_name)
        args = self.generate_constructor_call_args_with_types(arguments, member_types)
        return f"{self.rust_constructor_path(rust_type)}::new({', '.join(args)})"

    def generate_constructor_named_fields(self, named_arguments, field_type_for_name):
        argument_values = list(named_arguments.values())
        later_roots = self.return_call_arg_later_roots(argument_values)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(
            argument_values
        )
        fields = []
        for index, (name, value) in enumerate(named_arguments.items()):
            field_type = field_type_for_name(name)
            field_value = self.generate_with_lambda_capture_move_blocks(
                value,
                later_roots[index],
                lambda value=value, field_type=field_type: (
                    self.generate_constructor_argument_with_type(
                        value,
                        field_type,
                    )
                ),
                lambda_conflict_roots[index],
            )
            fields.append(f"{self.rust_identifier(name)}: {field_value}")
        return fields

    def rust_array_padding_expression(self, base_type, static_context=False):
        if static_context:
            rust_type = self.map_type(base_type)
            scalar_initializer = self.rust_scalar_default_literal(rust_type)
            if scalar_initializer is not None:
                return scalar_initializer
            if self.is_rust_shader_pod_value_type(rust_type):
                return "unsafe { std::mem::zeroed() }"
        return "Default::default()"

    def generate_assignment(self, node):
        # Handle both old and new AST assignment structures
        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            target_expr = node.target
            value_expr = node.value
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            target_expr = node.left
            value_expr = node.right
            op = getattr(node, "operator", "=")

        swizzle_assignment = self.generate_swizzle_assignment(
            target_expr, value_expr, op
        )
        if swizzle_assignment is not None:
            return swizzle_assignment

        pointer_assignment = self.generate_pointer_index_assignment(
            target_expr, value_expr, op
        )
        if pointer_assignment is not None:
            return pointer_assignment

        glsl_block_store = self.generate_glsl_buffer_block_assignment(
            target_expr, value_expr, op
        )
        if glsl_block_store is not None:
            return glsl_block_store

        vector_compound_assignment = self.generate_vector_compound_assignment(
            target_expr,
            value_expr,
            op,
        )
        if vector_compound_assignment is not None:
            return vector_compound_assignment

        if hasattr(node, "target") and hasattr(node, "value"):
            # New AST structure
            lhs = self.generate_assignment_target_expression(node.target)
            lhs_type = self.expression_result_type(node.target)
            rhs = self.generate_expression_with_type(node.value, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.value, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        else:
            # Old AST structure
            lhs = self.generate_assignment_target_expression(node.left)
            lhs_type = self.expression_result_type(node.left)
            rhs = self.generate_expression_with_type(node.right, lhs_type)
            rhs = self.normalize_assignment_rhs(
                lhs_type, node.right, rhs, getattr(node, "operator", "=")
            )
            op = getattr(node, "operator", "=")
        return f"{lhs} {op} {rhs}"

    def generate_vector_compound_assignment(self, target, value, operator):
        binary_operator = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }.get(operator)
        if binary_operator is None:
            return None

        target_type = self.expression_result_type(target)
        if self.vector_type_info(target_type) is None:
            return None

        value_type = self.expression_result_type(value)
        if (
            self.vector_type_info(value_type) is None
            and self.normalize_scalar_type(value_type) is None
        ):
            return None

        lhs = self.generate_assignment_target_expression(target)
        rhs = self.generate_vector_arithmetic_expression(
            target,
            value,
            target_type,
            value_type,
            binary_operator,
        )
        if rhs is None:
            expression = BinaryOpNode(target, binary_operator, value)
            rhs = self.generate_binary_expression(expression, target_type)
            rhs = self.normalize_typed_expression_value(expression, rhs, target_type)
        return f"{lhs} = {rhs}"

    def generate_swizzle_assignment(self, target, value, operator):
        if not isinstance(target, MemberAccessNode):
            return None

        components = self.member_swizzle_components(target)
        if components is None or len(components) <= 1:
            return None

        target_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, target_type)
        rhs = self.normalize_assignment_rhs(target_type, value, rhs, operator)
        rhs_info = self.vector_type_info(self.expression_result_type(value))
        target_info = self.vector_type_info(target_type)
        if rhs_info is None and target_info is not None:
            rhs = self.normalize_typed_expression_value(value, rhs, target_type)

        object_expr = getattr(target, "object_expr", getattr(target, "object", None))
        self.assignment_lhs_depth += 1
        try:
            obj = self.generate_expression(object_expr)
        finally:
            self.assignment_lhs_depth -= 1

        rhs_name = self.next_swizzle_temp_name()
        assignments = []
        if self.vector_type_info(self.expression_result_type(value)) is not None:
            for component in components:
                assignments.append(
                    f"{obj}.{component} {operator} {rhs_name}.{component}"
                )
        else:
            for component in components:
                assignments.append(f"{obj}.{component} {operator} {rhs_name}")
        return f"{{ let {rhs_name} = {rhs}; {'; '.join(assignments)}; }}"

    def generate_pointer_index_assignment(self, target, value, operator):
        if not isinstance(target, ArrayAccessNode):
            return None

        array_expr = getattr(target, "array_expr", getattr(target, "array", None))
        pointer_type = self.pointer_type_parts(
            self.map_type(self.expression_result_type(array_expr))
        )
        if pointer_type is None:
            return None

        index_expr = getattr(target, "index_expr", getattr(target, "index", None))
        pointer = self.generate_expression(array_expr)
        index = self.generate_array_index_expression(index_expr)
        lhs_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, lhs_type)
        rhs = self.normalize_assignment_rhs(lhs_type, value, rhs, operator)
        return f"unsafe {{ *({pointer}).add({index}) {operator} {rhs} }}"

    def generate_glsl_buffer_block_assignment(self, target, value, op):
        access = self.glsl_buffer_block_access(target)
        if access is None:
            return None
        if access == "readonly":
            return self.unsupported_glsl_buffer_block_statement(
                "store", "readonly placeholder cannot be written"
            )
        root_name = self.glsl_buffer_block_root_name(target)
        if root_name in self.static_variable_names and root_name not in (
            self.local_variable_names
        ):
            return self.unsupported_glsl_buffer_block_statement(
                "store", "Rust placeholder storage is immutable"
            )
        lhs = self.generate_assignment_target_expression(target)
        lhs_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, lhs_type)
        rhs = self.normalize_assignment_rhs(lhs_type, value, rhs, op)
        return f"{lhs} {op} {rhs}"

    def generate_assignment_target_expression(self, target):
        self.assignment_lhs_depth += 1
        try:
            return self.generate_expression(target)
        finally:
            self.assignment_lhs_depth -= 1

    def normalize_assignment_rhs(self, lhs_type, rhs_expr, generated_rhs, operator):
        if operator == "=":
            return self.normalize_typed_expression_value(
                rhs_expr, generated_rhs, lhs_type
            )

        compound_ops = {
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "^=",
            "|=",
            "&=",
            "<<=",
            ">>=",
        }
        if operator not in compound_ops:
            return generated_rhs

        target_type = self.normalize_scalar_type(lhs_type)
        source_type = self.expression_result_type(rhs_expr)
        if target_type is None or self.normalize_scalar_type(source_type) is None:
            return generated_rhs
        return self.normalize_binary_scalar_operand(
            rhs_expr, generated_rhs, source_type, target_type
        )

    def normalize_typed_expression_value(self, expr, generated_expr, target_type):
        if isinstance(expr, TernaryOpNode):
            return generated_expr
        if self.generated_binary_expression_matches_target(expr, target_type):
            return generated_expr

        reference_expr = self.normalize_reference_typed_expression(
            expr,
            generated_expr,
            target_type,
        )
        if reference_expr is not None:
            return reference_expr

        vector_expr = self.normalize_vector_typed_expression(expr, target_type)
        if vector_expr is not None:
            return vector_expr
        matrix_expr = self.normalize_matrix_typed_expression(expr, target_type)
        if matrix_expr is not None:
            return matrix_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, self.expression_result_type(expr), target_type
        )

    def normalize_reference_typed_expression(self, expr, generated_expr, target_type):
        target_reference = self.reference_type_parts_for_type(target_type)
        if target_reference is None:
            return None
        if isinstance(expr, (MatchNode, BlockNode)):
            return generated_expr

        generated_text = str(generated_expr).strip()
        if not generated_text:
            return generated_expr

        target_is_mutable, _ = target_reference
        source_reference = self.reference_type_parts_for_type(
            self.expression_result_type(expr)
        )
        if source_reference is not None:
            source_is_mutable, _ = source_reference
            if not target_is_mutable or source_is_mutable:
                return generated_expr

        if generated_text.startswith("&mut "):
            return generated_expr
        if not target_is_mutable and generated_text.startswith("&"):
            return generated_expr

        borrow_prefix = "&mut " if target_is_mutable else "&"
        return f"{borrow_prefix}{generated_expr}"

    def generated_binary_expression_matches_target(self, expr, target_type):
        if not isinstance(expr, BinaryOpNode) or target_type is None:
            return False

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        operator = self.map_operator(
            getattr(expr, "operator", getattr(expr, "op", None))
        )
        matrix_vector_plan = self.binary_matrix_vector_plan(
            left_type, right_type, operator, target_type
        )
        if matrix_vector_plan is None:
            return False
        return self.type_names_match(matrix_vector_plan["result_type"], target_type)

    def type_names_match(self, left_type, right_type):
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)
        if left_scalar is not None or right_scalar is not None:
            return left_scalar == right_scalar
        return self.map_type(left_type) == self.map_type(right_type)

    def normalize_scalar_assignment_value(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if source_type is None or target_type is None or source_type == target_type:
            return generated_expr

        if target_type == "bool":
            zero_literal = "0.0" if source_type in {"f16", "f32", "f64"} else "0"
            return f"({generated_expr} != {zero_literal})"

        if source_type == "bool" and target_type in {"f16", "f32", "f64"}:
            return f"(if {generated_expr} {{ 1.0 }} else {{ 0.0 }})"

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{self.integer_literal_expression_value(expr)}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        return f"({generated_expr} as {target_type})"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_condition_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition} {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate else if by recursively generating the nested if with else if prefix
                elif_condition = self.generate_condition_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if {elif_condition} {{\n"

                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                code += f"{indent_str}}}"

                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another else if - recursively handle
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "else if"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", " else if ", 1
                            )
                        code += "\n".join(
                            remaining_lines[1:]
                        )  # Skip first line as we already handled it
                    else:
                        # Final else clause
                        code += f" else {{\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
                        code += f"{indent_str}}}"
            else:
                code += f" else {{\n"
                if hasattr(else_branch, "statements"):
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    code += self.generate_statement(else_branch, indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        if init.endswith(";"):
            init = init[:-1]
        condition = self.generate_condition_expression(node.condition)
        update = self.generate_expression(node.update)

        code = f"{indent_str}{init};\n"
        code += f"{indent_str}while {condition} {{\n"

        self.loop_depth += 1
        self.for_contexts.append({"loop_depth": self.loop_depth, "update": update})
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.for_contexts.pop()
            self.loop_depth -= 1

        code += f"{indent_str}    {update};\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        if isinstance(pattern, str):
            pattern = self.rust_identifier(pattern)
        iterable = self.generate_for_in_iterable(getattr(node, "iterable", None))

        code = f"{indent_str}for {pattern} in {iterable} {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(getattr(node, "body", [])):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_for_in_iterable(self, iterable_node):
        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            operator = "..=" if iterable_node.inclusive else ".."
            return f"{start}{operator}{end}"

        iterable = self.generate_expression(iterable_node)
        return f"0..{iterable}"

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_condition_expression(node.condition)

        code = f"{indent_str}while {condition} {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}loop {{\n"

        self.loop_depth += 1
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            self.loop_depth -= 1

        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        break_flag = f"__cgl_do_break_{self.do_while_counter}"
        self.do_while_counter += 1
        condition = self.generate_condition_expression(node.condition)

        code = f"{indent_str}let mut {break_flag}: bool = false;\n"
        code += f"{indent_str}loop {{\n"
        code += f"{indent_str}    loop {{\n"

        self.loop_depth += 1
        self.do_while_contexts.append(
            {"loop_depth": self.loop_depth, "break_flag": break_flag}
        )
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 2)
        finally:
            self.do_while_contexts.pop()
            self.loop_depth -= 1

        if not self.statement_body_terminates_inner_loop(node.body):
            code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if {break_flag} {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if !({condition}) {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_expression(self, expr):
        """Render a CrossGL expression as Rust expression syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", str(expr))
            return self.generate_identifier_value_expression(name)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "name"):
                return self.generate_identifier_value_expression(expr.name)
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            return self.generate_binary_expression(expr)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr)
        elif isinstance(expr, ConstructorNode):
            return self.generate_constructor_expression(expr)
        elif isinstance(expr, TextureNode):
            return self.generate_texture_node_expression(expr)
        elif isinstance(expr, TextureOpNode):
            return self.generate_texture_op_node_expression(expr)
        elif isinstance(expr, BlockNode):
            return self.generate_block_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            return self.generate_ray_tracing_op_expression(expr)
        elif isinstance(expr, RayQueryOpNode):
            return self.generate_ray_query_op_expression(expr)
        elif isinstance(expr, MeshOpNode):
            return self.generate_rust_mesh_op_expression(expr)
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr)
        elif isinstance(expr, MatchNode):
            return self.generate_match_expression(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand_expr = getattr(expr, "operand", "")
            op = getattr(expr, "operator", getattr(expr, "op", "+"))
            op = self.map_operator(op)
            if op in ["++", "--"]:
                operand = self.generate_expression(operand_expr)
                assignment_op = "+=" if op == "++" else "-="
                increment = self.rust_increment_literal(
                    self.expression_result_type(operand_expr)
                )
                return f"{operand} {assignment_op} {increment}"
            bool_vector_not = self.generate_bool_vector_not_expression(operand_expr, op)
            if bool_vector_not is not None:
                return bool_vector_not
            operand = self.generate_expression(operand_expr)
            return f"({op}{operand})"
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            self.array_object_depth += 1
            try:
                array = self.generate_expression(array_expr)
            finally:
                self.array_object_depth -= 1
            array = self.lazy_static_object_expression(array_expr, array)
            index = self.generate_array_index_expression(index_expr)
            if (
                self.pointer_type_parts(
                    self.map_type(self.expression_result_type(array_expr))
                )
                is not None
            ):
                return f"unsafe {{ *({array}).add({index}) }}"
            access = f"{array}[{index}]"
            if self.should_clone_array_access_value(expr):
                return self.clone_value_expression(access)
            return access
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
            func_name = None
            if getattr(func_expr, "name", None):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            if isinstance(func_expr, MemberAccessNode):
                texture_call = self.generate_texture_member_function_call(
                    func_expr,
                    args,
                )
                if texture_call is not None:
                    return texture_call

            qualified_method_call = self.generate_qualified_generic_trait_method_call(
                func_expr,
                args,
            )
            if qualified_method_call is not None:
                return qualified_method_call

            if func_name == "lambda":
                return self.generate_lambda_expression(args)

            if self.is_user_defined_function(func_name):
                args_str = ", ".join(
                    self.generate_user_function_call_args(func_name, args)
                )
                return f"{self.rust_callable_name(callee)}({args_str})"

            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name,
                args,
            )
            if ray_tracing_call is not None:
                return ray_tracing_call

            enum_variant = self.generate_enum_variant_call_with_typed_args(
                func_name,
                args,
            )
            if enum_variant is not None:
                return enum_variant

            if func_name == "mix" and len(args) == 3:
                bool_mix = self.generate_bool_mix_expression(args)
                if bool_mix is not None:
                    return bool_mix

            original_func_name = func_name
            func_name = self.mapped_function_name(func_name, len(args), args)
            cuda_shuffle_call = self.generate_cuda_shuffle_intrinsic_call(
                original_func_name, args
            )
            if cuda_shuffle_call is not None:
                return cuda_shuffle_call
            self.validate_rust_texture_helper_call(func_name, args)
            if func_name == "saturate" and len(args) == 1:
                arg = self.generate_expression(args[0])
                return f"clamp({arg}, 0.0, 1.0)"

            dot_call = self.generate_inline_dot_call(func_name, args)
            if dot_call is not None:
                return dot_call

            vector_intrinsic = self.generate_inline_vector_intrinsic_call(
                func_name,
                args,
            )
            if vector_intrinsic is not None:
                return vector_intrinsic

            generic_intrinsic = self.generate_generic_intrinsic_call(func_name, args)
            if generic_intrinsic is not None:
                return generic_intrinsic

            scalar_cast = self.generate_scalar_constructor_call(func_name, args)
            if scalar_cast is not None:
                return scalar_cast

            vector_info = self.vector_type_info(func_name)
            if vector_info:
                return self.generate_vector_constructor_call(
                    func_name, vector_info, args
                )

            if func_name in [
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
            ]:
                return self.generate_matrix_constructor_call(func_name, args)

            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{self.rust_callable_name(func_name or callee)}({args_str})"
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            return self.generate_member_access_expression(expr)
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            return self.generate_ternary_expression(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_wave_op_expression(expr)
        else:
            return str(expr)

    def rust_increment_literal(self, operand_type):
        scalar_type = self.normalize_scalar_type(operand_type)
        if scalar_type in {"f16", "f32", "f64"}:
            return "1.0"
        return "1"

    def generate_inline_dot_call(self, func_name, args):
        if func_name != "dot" or len(args or []) != 2:
            return None

        left_type = self.expression_result_type(args[0])
        right_type = self.expression_result_type(args[1])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if left_info["size"] != right_info["size"]:
            return None

        # The standalone Rust contract already has Vec3 dot; other vector widths
        # need concrete lane math instead of a runtime overload.
        if left_info["size"] == 3:
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"],
            right_info["component_type"],
        )
        if component_type is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_argument_lane_expressions(
            args[0],
            left_info,
            temp_bindings,
            component_type,
        )
        right_lanes = self.vector_argument_lane_expressions(
            args[1],
            right_info,
            temp_bindings,
            component_type,
        )
        products = [
            f"({left_lane} * {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        expression = " + ".join(products)
        if not temp_bindings:
            return expression

        bindings = " ".join(
            self.generate_temp_binding_statement(binding) for binding in temp_bindings
        )
        return f"{{ {bindings} {expression} }}"

    def generate_inline_vector_intrinsic_call(self, func_name, args):
        if len(args or []) != 1 or func_name not in {"length", "normalize"}:
            return None

        value_type = self.expression_result_type(args[0])
        value_info = self.vector_type_info(value_type)
        if value_info is None:
            return None

        # The standalone Rust contract already has Vec3 length/normalize.
        if value_info["size"] == 3:
            return None

        component_type = value_info["component_type"]
        temp_bindings = []
        value_expr = self.generate_expression(args[0])
        temp_name = self.next_vector_arg_temp_name()
        type_hint = self.map_type(value_type)
        temp_bindings.append((temp_name, value_expr, type_hint))

        components = ("x", "y", "z", "w")[: value_info["size"]]
        length_terms = [
            f"({temp_name}.{component} * {temp_name}.{component})"
            for component in components
        ]
        length_expr = f"({self.sum_rust_terms(length_terms)}).sqrt()"
        if func_name == "length":
            bindings = " ".join(
                self.generate_temp_binding_statement(binding)
                for binding in temp_bindings
            )
            return f"{{ {bindings} {length_expr} }}"

        length_name = self.next_vector_arg_temp_name()
        temp_bindings.append((length_name, length_expr, self.map_type(component_type)))
        lanes = [
            f"({temp_name}.{component} / {length_name})" for component in components
        ]
        return self.generate_constructor_call(
            self.map_type(value_type),
            lanes,
            temp_bindings,
        )

    def generate_generic_intrinsic_call(self, func_name, args):
        if func_name != "sqrt" or len(args) != 1:
            return None

        argument_type = self.expression_result_type(args[0])
        if argument_type not in self.current_generic_param_names:
            return None

        self.required_generic_math_traits.add("CglSqrt")
        return f"CglSqrt::cgl_sqrt({self.generate_expression(args[0])})"

    def generate_cuda_shuffle_intrinsic_call(self, func_name, args):
        if func_name != "__shfl_down_sync" or len(args or []) < 3:
            return None
        value = self.generate_expression(args[1])
        delta = self.generate_expression(args[2])
        return f"subgroup_shuffle_down({value}, {delta})"

    def generate_ray_tracing_op_expression(self, expr):
        operation = getattr(expr, "operation", "")
        if operation in self.user_function_names:
            args = ", ".join(
                self.generate_user_function_call_args(
                    operation,
                    getattr(expr, "arguments", []),
                )
            )
            return f"{self.rust_callable_name(operation)}({args})"

        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(expr, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, arguments):
        if operation not in {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "IgnoreHit",
            "AcceptHitAndEndSearch",
        }:
            return None

        func_name = self.function_map.get(operation, operation)
        args = ", ".join(self.generate_expression(arg) for arg in arguments or [])
        if operation == "TraceRay" and len(arguments or []) == 11:
            generated_args = [self.generate_expression(arg) for arg in arguments]
            ray_desc = (
                "RayDesc { "
                f"origin: {generated_args[6]}, "
                f"t_min: {generated_args[7]}, "
                f"direction: {generated_args[8]}, "
                f"t_max: {generated_args[9]} "
                "}"
            )
            args = ", ".join(generated_args[:6] + [ray_desc, generated_args[10]])
        elif operation == "ReportHit" and len(arguments or []) == 2:
            generated_args = [self.generate_expression(arg) for arg in arguments]
            args = ", ".join(generated_args + ["()"])
        return f"{func_name}({args})"

    def generate_ray_query_op_expression(self, expr):
        operation = getattr(expr, "operation", "")
        receiver = self.generate_expression(getattr(expr, "query_expr", None))
        args = [receiver]
        args.extend(
            self.generate_expression(arg)
            for arg in getattr(expr, "arguments", []) or []
        )
        func_name = f"ray_query_{self.rust_snake_case_name(operation)}"
        return f"{func_name}({', '.join(args)})"

    def generate_texture_node_expression(self, expr):
        args = [expr.texture_expr]
        if getattr(expr, "sampler_expr", None) is not None:
            args.append(expr.sampler_expr)
        args.append(expr.coordinates)
        operation = "texture"
        if getattr(expr, "level", None) is not None:
            args.append(expr.level)
            operation = "textureLod"
        if getattr(expr, "offset", None) is not None:
            args.append(expr.offset)
            operation = (
                "textureLodOffset" if operation == "textureLod" else "textureOffset"
            )
        return self.generate_texture_operation_call(operation, args)

    def generate_texture_op_node_expression(self, expr):
        operation = getattr(expr, "operation", "")
        sampler_expr = getattr(expr, "sampler_expr", None)
        arguments = getattr(expr, "arguments", []) or []
        if operation in self.rust_hlsl_texture_member_operations():
            member_args = []
            if sampler_expr is not None:
                member_args.append(sampler_expr)
            member_args.extend(arguments)
            call = self.generate_hlsl_texture_member_call(
                operation,
                expr.texture_expr,
                member_args,
            )
            if call is not None:
                return call

        args = [expr.texture_expr]
        if sampler_expr is not None:
            args.append(sampler_expr)
        args.extend(arguments)
        call = self.generate_texture_operation_call(operation, args)
        if call is not None:
            return call
        raise ValueError(f"Unsupported Rust texture operation {operation}")

    def generate_texture_member_function_call(self, func_expr, args):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        if not self.is_texture_resource_type(self.expression_result_type(obj_expr)):
            return None
        call = self.generate_hlsl_texture_member_call(member, obj_expr, args or [])
        if call is not None:
            return call
        raise ValueError(f"Unsupported Rust texture member operation {member}")

    def generate_texture_operation_call(self, operation, args):
        if operation in self.rust_hlsl_texture_member_operations():
            if not args:
                raise ValueError(f"Unsupported Rust texture operation {operation}")
            call = self.generate_hlsl_texture_member_call(
                operation,
                args[0],
                list(args[1:]),
            )
            if call is not None:
                return call
        if operation in {"Sample", "sample", "texture"}:
            helper_name = self.texture_sample_helper_name(args)
            return self.generate_texture_helper_call(helper_name, args)
        if operation == "SampleCmp":
            helper_name = (
                "texture_compare_sampler"
                if self.call_has_explicit_sampler_argument(args)
                else "texture_compare"
            )
            return self.generate_texture_helper_call(helper_name, args)
        if operation in self.function_map or str(operation).startswith("texture"):
            helper_name = self.mapped_function_name(operation, len(args or []), args)
            if helper_name != operation:
                return self.generate_texture_helper_call(helper_name, args)
        return None

    def texture_sample_helper_name(self, args):
        has_sampler = self.call_has_explicit_sampler_argument(args)
        shadow = self.call_uses_shadow_texture(args)
        if shadow:
            return "sample_shadow_sampler" if has_sampler else "sample_shadow"
        return "sample_sampler" if has_sampler else "sample"

    def texture_member_function_result_type(self, func_expr, args):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        texture_type = self.expression_result_type(obj_expr)
        if not self.is_texture_resource_type(texture_type):
            return None
        if member == "SampleCmp":
            return "float"
        if member in {
            "SampleCmpLevel",
            "SampleCmpLevelZero",
            "SampleCmpGrad",
            "CalculateLevelOfDetail",
            "CalculateLevelOfDetailUnclamped",
        }:
            return "float"
        if member == "GetDimensions":
            return "void"
        if member in {"Sample", "sample", "texture"}:
            return "float" if self.is_shadow_texture_type(texture_type) else "vec4"
        if member in self.rust_hlsl_texture_member_operations():
            return "vec4"
        return None

    def generate_texture_helper_call(self, helper_name, args):
        self.validate_rust_texture_helper_call(helper_name, args)
        generated_args = ", ".join(self.generate_expression(arg) for arg in args or [])
        return f"{helper_name}({generated_args})"

    def rust_hlsl_texture_member_operations(self):
        return {
            "Sample",
            "SampleLevel",
            "SampleLOD",
            "SampleGrad",
            "SampleBias",
            "SampleCmp",
            "SampleCmpLevel",
            "SampleCmpLevelZero",
            "SampleCmpGrad",
            "Gather",
            "GatherRed",
            "GatherGreen",
            "GatherBlue",
            "GatherAlpha",
            "GatherCmp",
            "GatherCmpRed",
            "GatherCmpGreen",
            "GatherCmpBlue",
            "GatherCmpAlpha",
            "Load",
            "GetDimensions",
            "CalculateLevelOfDetail",
            "CalculateLevelOfDetailUnclamped",
        }

    def generate_hlsl_texture_member_call(self, operation, texture_expr, member_args):
        member_args = list(member_args or [])
        texture_type = self.expression_result_type(texture_expr)

        def helper(helper_name, args):
            return self.generate_texture_helper_call(helper_name, args)

        if operation == "Sample":
            self.require_rust_texture_member_arg_count(operation, member_args, {2, 3})
            helper_name = (
                "sample_shadow_sampler"
                if self.is_shadow_texture_type(texture_type)
                else "sample_sampler"
            )
            if len(member_args) == 3:
                helper_name = "sample_offset_sampler"
            return helper(helper_name, [texture_expr, *member_args])

        if operation in {"SampleLevel", "SampleLOD"}:
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "sample_lod_offset_sampler"
                if len(member_args) == 4
                else "sample_lod_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleGrad":
            self.require_rust_texture_member_arg_count(operation, member_args, {4, 5})
            helper_name = (
                "sample_grad_offset_sampler"
                if len(member_args) == 5
                else "sample_grad_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleBias":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            sampler, coord, bias = member_args[:3]
            if len(member_args) == 4:
                offset = member_args[3]
                return helper(
                    "sample_offset_bias_sampler",
                    [texture_expr, sampler, coord, offset, bias],
                )
            return helper("sample_bias_sampler", [texture_expr, sampler, coord, bias])

        if operation == "SampleCmp":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpLevel":
            self.require_rust_texture_member_arg_count(operation, member_args, {4, 5})
            helper_name = (
                "texture_compare_lod_offset_sampler"
                if len(member_args) == 5
                else "texture_compare_lod_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpLevelZero":
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "SampleCmpGrad":
            self.require_rust_texture_member_arg_count(operation, member_args, {5, 6})
            helper_name = (
                "texture_compare_grad_offset_sampler"
                if len(member_args) == 6
                else "texture_compare_grad_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        gather_components = {
            "GatherRed": "0",
            "GatherGreen": "1",
            "GatherBlue": "2",
            "GatherAlpha": "3",
        }
        if operation in {"Gather", *gather_components}:
            self.require_rust_texture_member_arg_count(operation, member_args, {2, 3})
            if operation == "Gather":
                helper_name = (
                    "texture_gather_offset_sampler"
                    if len(member_args) == 3
                    else "texture_gather_sampler"
                )
                return helper(helper_name, [texture_expr, *member_args])

            component = gather_components[operation]
            sampler, coord = member_args[:2]
            if len(member_args) == 3:
                offset = member_args[2]
                return helper(
                    "texture_gather_offset_component_sampler",
                    [texture_expr, sampler, coord, offset, component],
                )
            return helper(
                "texture_gather_component_sampler",
                [texture_expr, sampler, coord, component],
            )

        if operation in {
            "GatherCmp",
            "GatherCmpRed",
            "GatherCmpGreen",
            "GatherCmpBlue",
            "GatherCmpAlpha",
        }:
            self.require_rust_texture_member_arg_count(operation, member_args, {3, 4})
            helper_name = (
                "texture_gather_compare_offset_sampler"
                if len(member_args) == 4
                else "texture_gather_compare_sampler"
            )
            return helper(helper_name, [texture_expr, *member_args])

        if operation == "Load":
            self.require_rust_texture_member_arg_count(operation, member_args, {1, 2})
            if len(member_args) == 2 and not self.is_multisample_texture_type(
                texture_type
            ):
                return helper(
                    "texel_fetch_offset",
                    [texture_expr, member_args[0], "0", member_args[1]],
                )
            lod_or_sample = member_args[1] if len(member_args) == 2 else "0"
            return helper("texel_fetch", [texture_expr, member_args[0], lod_or_sample])

        if operation == "GetDimensions":
            generated_args = ", ".join(
                self.generate_expression(arg) for arg in member_args
            )
            if len(member_args) == 1:
                generated_args += ","
            return (
                f"texture_dimensions({self.generate_expression(texture_expr)}, "
                f"({generated_args}))"
            )

        if operation in {"CalculateLevelOfDetail", "CalculateLevelOfDetailUnclamped"}:
            self.require_rust_texture_member_arg_count(operation, member_args, {2})
            return helper("texture_calculate_lod", [texture_expr, *member_args])

        return None

    def require_rust_texture_member_arg_count(self, operation, args, expected_counts):
        if len(args) not in expected_counts:
            expected = ", ".join(str(count) for count in sorted(expected_counts))
            raise ValueError(
                f"Unsupported Rust texture member operation {operation}; "
                f"expected {expected} argument(s), got {len(args)}"
            )

    def validate_rust_texture_helper_call(self, helper_name, args):
        if not args:
            return
        if not self.is_rust_texture_helper_name(helper_name):
            return

        texture_type = self.expression_result_type(args[0])
        if texture_type is not None and not self.is_texture_resource_type(texture_type):
            raise ValueError(
                f"Unsupported {helper_name} for Rust codegen; texture resource "
                f"required: {self.map_type(texture_type)}"
            )

        if helper_name.startswith(
            ("texture_compare", "texture_gather_compare")
        ) and not self.is_shadow_texture_type(texture_type):
            raise ValueError(
                f"Unsupported {helper_name} for Rust codegen; shadow texture "
                f"required: {self.map_type(texture_type)}"
            )

    def is_rust_texture_helper_name(self, helper_name):
        helper_name = str(helper_name)
        return helper_name.startswith(
            (
                "sample",
                "texel_fetch",
                "texture_size",
                "texture_query",
                "texture_samples",
                "texture_gather",
                "texture_compare",
                "texture_dimensions",
                "texture_calculate_lod",
            )
        )

    def rust_snake_case_name(self, name):
        name = str(name)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.replace("__", "_").strip("_").lower()

    def mapped_function_name(self, func_name, arg_count=None, arguments=None):
        if not isinstance(func_name, str):
            return func_name
        if arguments is not None:
            arg_count = len(arguments)
        has_sampler = self.call_has_explicit_sampler_argument(arguments)
        if func_name in {"textureSize", "textureDimensions"}:
            return "texture_size" if arg_count == 1 else "texture_size_lod"
        if func_name == "texture":
            if self.call_uses_shadow_texture(arguments):
                if has_sampler:
                    return (
                        "sample_shadow_bias_sampler"
                        if arg_count and arg_count >= 4
                        else "sample_shadow_sampler"
                    )
                return (
                    "sample_shadow_bias"
                    if arg_count and arg_count >= 3
                    else "sample_shadow"
                )
            if has_sampler:
                return (
                    "sample_bias_sampler"
                    if arg_count and arg_count >= 4
                    else "sample_sampler"
                )
            return "sample_bias" if arg_count and arg_count >= 3 else "sample"
        if func_name == "textureOffset":
            if has_sampler:
                return (
                    "sample_offset_bias_sampler"
                    if arg_count and arg_count >= 5
                    else "sample_offset_sampler"
                )
            return (
                "sample_offset_bias"
                if arg_count and arg_count >= 4
                else "sample_offset"
            )
        if func_name == "textureProj":
            if has_sampler:
                return (
                    "sample_projected_bias_sampler"
                    if arg_count and arg_count >= 4
                    else "sample_projected_sampler"
                )
            return (
                "sample_projected_bias"
                if arg_count and arg_count >= 3
                else "sample_projected"
            )
        if func_name == "textureProjOffset":
            if has_sampler:
                return (
                    "sample_projected_offset_bias_sampler"
                    if arg_count and arg_count >= 5
                    else "sample_projected_offset_sampler"
                )
            return (
                "sample_projected_offset_bias"
                if arg_count and arg_count >= 4
                else "sample_projected_offset"
            )
        sampler_texture_map = {
            "textureLod": "sample_lod_sampler",
            "textureLodOffset": "sample_lod_offset_sampler",
            "textureGrad": "sample_grad_sampler",
            "textureGradOffset": "sample_grad_offset_sampler",
            "textureProjLod": "sample_projected_lod_sampler",
            "textureProjLodOffset": "sample_projected_lod_offset_sampler",
            "textureProjGrad": "sample_projected_grad_sampler",
            "textureProjGradOffset": "sample_projected_grad_offset_sampler",
            "textureQueryLod": "texture_query_lod_sampler",
        }
        if has_sampler and func_name in sampler_texture_map:
            return sampler_texture_map[func_name]
        if func_name == "imageLoad" and arg_count == 3:
            return "image_load_sample"
        if func_name == "imageStore" and arg_count == 4:
            return "image_store_sample"
        sample_indexed_atomics = {
            "imageAtomicAdd": "image_atomic_add_sample",
            "imageAtomicMin": "image_atomic_min_sample",
            "imageAtomicMax": "image_atomic_max_sample",
            "imageAtomicAnd": "image_atomic_and_sample",
            "imageAtomicOr": "image_atomic_or_sample",
            "imageAtomicXor": "image_atomic_xor_sample",
            "imageAtomicExchange": "image_atomic_exchange_sample",
        }
        if func_name in sample_indexed_atomics and arg_count == 4:
            return sample_indexed_atomics[func_name]
        if func_name in {"imageAtomicCompSwap", "imageAtomicCompareExchange"} and (
            arg_count == 5
        ):
            return "image_atomic_comp_swap_sample"
        if func_name == "textureGather":
            if has_sampler:
                return (
                    "texture_gather_component_sampler"
                    if arg_count and arg_count >= 4
                    else "texture_gather_sampler"
                )
            return (
                "texture_gather_component"
                if arg_count and arg_count >= 3
                else "texture_gather"
            )
        if func_name == "textureGatherOffset":
            if has_sampler:
                return (
                    "texture_gather_offset_component_sampler"
                    if arg_count and arg_count >= 5
                    else "texture_gather_offset_sampler"
                )
            return (
                "texture_gather_offset_component"
                if arg_count and arg_count >= 4
                else "texture_gather_offset"
            )
        if func_name == "textureGatherOffsets":
            if has_sampler:
                return (
                    "texture_gather_offsets_component_sampler"
                    if arg_count in {5, 8}
                    else "texture_gather_offsets_sampler"
                )
            return (
                "texture_gather_offsets_component"
                if arg_count in {4, 7}
                else "texture_gather_offsets"
            )
        compare_sampler_thresholds = {
            "textureCompare": (4, "texture_compare_sampler"),
            "textureCompareOffset": (5, "texture_compare_offset_sampler"),
            "textureCompareLod": (5, "texture_compare_lod_sampler"),
            "textureCompareLodOffset": (6, "texture_compare_lod_offset_sampler"),
            "textureCompareGrad": (6, "texture_compare_grad_sampler"),
            "textureCompareGradOffset": (7, "texture_compare_grad_offset_sampler"),
            "textureCompareProj": (4, "texture_compare_projected_sampler"),
            "textureCompareProjOffset": (5, "texture_compare_projected_offset_sampler"),
            "textureCompareProjLod": (5, "texture_compare_projected_lod_sampler"),
            "textureCompareProjLodOffset": (
                6,
                "texture_compare_projected_lod_offset_sampler",
            ),
            "textureCompareProjGrad": (6, "texture_compare_projected_grad_sampler"),
            "textureCompareProjGradOffset": (
                7,
                "texture_compare_projected_grad_offset_sampler",
            ),
            "textureGatherCompare": (4, "texture_gather_compare_sampler"),
            "textureGatherCompareOffset": (5, "texture_gather_compare_offset_sampler"),
        }
        sampler_mapping = compare_sampler_thresholds.get(func_name)
        if sampler_mapping is not None:
            threshold, sampler_name = sampler_mapping
            if has_sampler or (arg_count and arg_count >= threshold):
                return sampler_name
        return self.function_map.get(func_name, func_name)

    def call_has_explicit_sampler_argument(self, arguments):
        if not arguments or len(arguments) < 2:
            return False
        return self.is_sampler_type(self.expression_result_type(arguments[1]))

    def call_uses_shadow_texture(self, arguments):
        if not arguments:
            return False
        return self.is_shadow_texture_type(self.expression_result_type(arguments[0]))

    def is_sampler_type(self, type_name):
        if type_name is None:
            return False
        type_name = str(type_name)
        return (
            type_name in {"sampler", "Sampler"} or self.map_type(type_name) == "Sampler"
        )

    def is_texture_resource_type(self, type_name):
        if type_name is None:
            return False
        mapped_type = self.map_type(type_name)
        base_type, _generic_args = self.generic_type_parts(mapped_type)
        base_name = base_type.rsplit("::", 1)[-1]
        return base_name.startswith(("Texture", "DepthTexture"))

    def is_shadow_texture_type(self, type_name):
        if type_name is None:
            return False
        source_type = str(type_name)
        mapped_type = self.map_type(source_type)
        names = {source_type, mapped_type}
        return any("Shadow" in name or "DepthTexture" in name for name in names)

    def is_multisample_texture_type(self, type_name):
        if type_name is None:
            return False
        source_type = str(type_name)
        mapped_type = self.map_type(source_type)
        names = {source_type, mapped_type}
        return any("MS" in name for name in names)

    def generate_qualified_generic_trait_method_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None

        method_name = getattr(func_expr, "member", "")
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        receiver_type = self.expression_result_type(object_expr)
        if receiver_type not in self.current_generic_param_names:
            return None

        trait_name = self.generic_trait_method_owner(receiver_type, method_name)
        if trait_name is None:
            return None

        receiver = self.generate_expression(object_expr)
        args_str = ", ".join(self.generate_expression(arg) for arg in args)
        if args_str:
            return f"{trait_name}::{method_name}({receiver}, {args_str})"
        return f"{trait_name}::{method_name}({receiver})"

    def generic_trait_method_owner(self, generic_name, method_name):
        for constraint in self.current_function_generic_constraints.get(
            generic_name, []
        ):
            trait_name = str(constraint).split("<", 1)[0]
            if method_name in self.trait_methods.get(trait_name, set()):
                return trait_name
        return None

    def generate_lambda_expression(self, args):
        """Render CrossGL's compact pseudo-lambda as a native Rust closure."""
        if not args:
            return "|| ()"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"|{params}| {body}"

    def generate_lambda_node_expression(self, node, target_type=None):
        """Render a first-class LambdaNode as a native Rust closure."""
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        try:
            params = []
            parameter_target_types = self.callable_parameter_target_types(target_type)
            for index, param in enumerate(getattr(node, "parameters", []) or []):
                param_type = self.function_parameter_type(param)
                if self.is_inferred_declaration_type(param_type):
                    if index < len(parameter_target_types):
                        param_type = parameter_target_types[index]
                self.register_variable_type(param.name, param_type, scope="local")
                params.append(self.generate_lambda_node_parameter(param, param_type))

            return_type = self.lambda_node_return_target_type(node, target_type)
            body = self.generate_lambda_node_body(
                getattr(node, "body", None),
                return_type=return_type,
            )
            move_prefix = "move " if self.force_move_lambda_depth > 0 else ""
            return f"{move_prefix}|{', '.join(params)}| {body}"
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

    def generate_lambda_node_parameter(self, param, param_type):
        if self.is_inferred_declaration_type(param_type):
            return param.name
        return f"{param.name}: {self.map_type(param_type)}"

    def lambda_node_return_target_type(self, node, target_type):
        explicit_type = getattr(node, "return_type", None)
        if explicit_type is None and hasattr(node, "get_annotation"):
            explicit_type = node.get_annotation("return_type")
        if explicit_type is not None:
            return self.type_name_string(explicit_type)

        return self.callable_return_target_type(target_type)

    def callable_return_target_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        return_type = self.split_callable_return_type(type_name)
        if return_type is None:
            return None
        return self.source_type_from_rust_type(return_type)

    def callable_parameter_target_types(self, type_name):
        signature = self.callable_signature_parts(type_name)
        if signature is None:
            return []

        _callable_trait, params_text, _return_type = signature
        if not params_text:
            return []

        return [
            self.source_type_from_rust_type(param_type.strip())
            for param_type in self.split_top_level_list(params_text)
            if param_type.strip()
        ]

    def callable_type_uses_trait(self, type_name, trait_name):
        signature = self.callable_signature_parts(type_name)
        if signature is None:
            return False

        callable_trait, _params_text, _return_type = signature
        return callable_trait == trait_name

    def callable_signature_parts(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        type_name = type_name.strip()
        match = re.match(r"^(?:dyn\s+|impl\s+)?(FnOnce|FnMut|Fn)\s*\(", type_name)
        if match is None:
            return None

        open_index = type_name.find("(", match.start())
        close_index = self.find_matching_delimiter(type_name, open_index, "(", ")")
        if close_index is None:
            return None

        params_text = type_name[open_index + 1 : close_index].strip()
        suffix = type_name[close_index + 1 :].strip()
        return_type = "()"
        if suffix.startswith("->"):
            return_type = suffix[2:].strip()

        return match.group(1), params_text, return_type

    def split_callable_return_type(self, type_name):
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        index = 0
        while index < len(type_name) - 1:
            char = type_name[index]
            next_char = type_name[index + 1]
            if (
                char == "-"
                and next_char == ">"
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return type_name[index + 2 :].strip()

            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            index += 1
        return None

    def type_name_string(self, type_name):
        if type_name is None:
            return None
        if self.is_type_node(type_name):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def is_type_node(self, value):
        return (
            value.__class__.__name__
            in {"ReferenceType", "PointerType", "FunctionType", "ArrayType"}
            or hasattr(value, "name")
            or hasattr(value, "element_type")
        )

    def source_type_from_rust_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        type_name = type_name.strip()

        array_parts = self.rust_array_type_parts(type_name)
        if array_parts is not None:
            base_type, sizes = array_parts
            return self.format_c_array_type(base_type, sizes)

        rust_scalar_aliases = {
            "i16": "short",
            "u16": "ushort",
            "i32": "int",
            "u32": "uint",
            "i64": "long",
            "u64": "ulong",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        return rust_scalar_aliases.get(type_name, type_name)

    def rust_array_type_parts(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        type_name = type_name.strip()
        if not (type_name.startswith("[") and type_name.endswith("]")):
            return None

        body = type_name[1:-1].strip()
        separator = self.find_top_level_separator(body, ";")
        if separator is None:
            return None

        element_type = body[:separator].strip()
        size_text = body[separator + 1 :].strip()
        try:
            size = int(size_text)
        except ValueError:
            size = None

        nested_parts = self.rust_array_type_parts(element_type)
        if nested_parts is not None:
            base_type, nested_sizes = nested_parts
            return base_type, [size, *nested_sizes]
        return self.source_type_from_rust_type(element_type), [size]

    def find_top_level_separator(self, text, separator):
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        for index, char in enumerate(text):
            if char == separator and all(
                depth == 0 for depth in (paren_depth, bracket_depth, angle_depth)
            ):
                return index
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
        return None

    def find_matching_delimiter(self, text, open_index, open_char, close_char):
        if open_index < 0 or open_index >= len(text) or text[open_index] != open_char:
            return None

        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return index
        return None

    def split_top_level_list(self, text, separator=","):
        parts = []
        current = []
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        for char in text:
            if char == separator and all(
                depth == 0 for depth in (paren_depth, bracket_depth, angle_depth)
            ):
                parts.append("".join(current))
                current = []
                continue
            current.append(char)

            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)

        if current:
            parts.append("".join(current))
        return parts

    def generate_lambda_node_body(self, body, return_type=None):
        saved_return_type = self.current_return_type
        self.current_return_type = return_type
        try:
            if isinstance(body, BlockNode):
                return self.generate_block_expression(
                    body,
                    target_type=return_type,
                    return_context=True,
                )

            return self.generate_return_branch_expression_with_type(body, return_type)
        finally:
            self.current_return_type = saved_return_type

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return raw or "_"

        type_name, param_name = typed_param
        return f"{self.rust_identifier(param_name)}: {self.map_type(type_name)}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw:
            return raw
        return self.generate_expression(arg)

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        if type_name in {"mut", "ref"}:
            return None
        return type_name, param_name

    def generate_bool_mix_expression(self, args):
        condition_type = self.expression_result_type(args[2])
        condition_info = self.vector_type_info(condition_type)
        if condition_info is not None:
            if condition_info["component_type"] != "bool":
                return None
            return self.generate_bool_vector_ternary_expression(
                args[2], args[1], args[0]
            )

        if self.normalize_scalar_type(condition_type) != "bool":
            return None

        result_type = self.promoted_bool_mix_scalar_type(args[0], args[1])
        if result_type is None:
            return None

        condition = self.generate_condition_expression(args[2])
        true_value = self.generate_expression_with_type(args[1], result_type)
        false_value = self.generate_expression_with_type(args[0], result_type)
        true_value = self.normalize_typed_expression_value(
            args[1], true_value, result_type
        )
        false_value = self.normalize_typed_expression_value(
            args[0], false_value, result_type
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_function_call_with_target(self, expr, target_type):
        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", []))
        enum_variant = self.generate_enum_variant_call_with_typed_args(
            func_name,
            args,
            target_type=target_type,
        )
        if enum_variant is not None:
            return enum_variant

        struct_constructor = self.generate_struct_new_call_with_typed_args(
            func_name,
            args,
        )
        if struct_constructor is not None:
            return struct_constructor

        vector_info = self.vector_type_info(func_name)
        if vector_info is None:
            return None

        target_rust_type = self.vector_constructor_target_type(
            target_type,
            vector_info,
        )
        if target_rust_type is None:
            return None

        generated_args, temp_bindings = self.generate_vector_constructor_args(
            vector_info,
            args,
        )
        return self.generate_constructor_call(
            target_rust_type,
            generated_args,
            temp_bindings,
        )

    def generate_struct_new_call_with_typed_args(self, func_name, args):
        type_name = self.struct_new_call_type_name(func_name)
        if type_name is None:
            return None

        member_types = self.resolve_struct_positional_member_types(type_name)
        if not member_types:
            return None

        generated_args = self.generate_constructor_call_args_with_types(
            args,
            member_types,
        )
        return f"{self.struct_new_call_path(type_name)}({', '.join(generated_args)})"

    def vector_constructor_target_type(self, target_type, source_vector_info):
        if target_type is None:
            return None

        target_info = self.vector_type_info(target_type)
        if target_info is None:
            return None
        if target_info["size"] != source_vector_info["size"]:
            return None

        return self.map_type(target_type)

    def promoted_bool_mix_scalar_type(self, false_expr, true_expr):
        false_type = self.expression_result_type(false_expr)
        true_type = self.expression_result_type(true_expr)
        if self.vector_type_info(false_type) or self.vector_type_info(true_type):
            return None
        return self.promoted_scalar_type(false_type, true_type)

    def struct_new_call_type_name(self, func_name):
        if not isinstance(func_name, str) or not func_name.endswith("::new"):
            return None

        type_name = func_name[: -len("::new")]
        return type_name.replace("::<", "<", 1)

    def struct_new_call_path(self, type_name):
        rust_type = self.map_type(type_name)
        return f"{self.rust_constructor_path(rust_type)}::new"

    def enum_variant_call_info(self, func_name, target_type=None):
        if not isinstance(func_name, str) or "::" not in func_name:
            return None

        enum_path, variant_name = func_name.rsplit("::", 1)
        if not enum_path or not variant_name:
            return None

        enum_type = self.normalize_turbofish_type_name(enum_path)
        enum_base, enum_args = self.generic_type_parts(enum_type)
        target_type = self.normalize_turbofish_type_name(target_type)
        target_base, target_args = (
            self.generic_type_parts(target_type) if target_type else (None, [])
        )
        if target_base == enum_base and target_args and not enum_args:
            enum_type = target_type

        enum_base, _enum_args = self.generic_type_parts(enum_type)
        variants = self.enum_variant_names.get(enum_base, [])
        if variant_name not in variants:
            return None

        payload_types = self.resolve_enum_variant_payload_types(
            enum_type,
            variant_name,
        )
        return enum_type, variant_name, payload_types

    def enum_variant_call_result_type(self, func_name):
        variant_info = self.enum_variant_call_info(func_name)
        if variant_info is None:
            return None
        enum_type, _variant_name, _payload_types = variant_info
        return enum_type

    def resolve_enum_variant_payload_types(self, enum_type, variant_name):
        if enum_type is None or variant_name is None:
            return []

        enum_type = self.normalize_turbofish_type_name(enum_type)
        exact_variants = self.enum_variant_payload_type_map.get(enum_type)
        if exact_variants is not None:
            return list(exact_variants.get(variant_name, []))

        enum_base, generic_args = self.generic_type_parts(enum_type)
        variant_payloads = self.enum_variant_payload_type_map.get(enum_base, {})
        payload_types = variant_payloads.get(variant_name)
        if payload_types is None:
            return []

        generic_params = self.enum_generic_params.get(
            enum_base,
            self.struct_generic_params.get(enum_base, []),
        )
        substitutions = dict(zip(generic_params, generic_args))
        return [
            self.substitute_type_parameters(payload_type, substitutions)
            for payload_type in payload_types
        ]

    def resolve_enum_variant_field_types(self, enum_type, variant_name):
        if enum_type is None or variant_name is None:
            return {}

        enum_type = self.normalize_turbofish_type_name(enum_type)
        exact_variants = self.enum_variant_field_types.get(enum_type)
        if exact_variants is not None:
            return dict(exact_variants.get(variant_name, {}))

        enum_base, generic_args = self.generic_type_parts(enum_type)
        variant_fields = self.enum_variant_field_types.get(enum_base, {}).get(
            variant_name,
        )
        if not variant_fields:
            return {}

        generic_params = self.enum_generic_params.get(
            enum_base,
            self.struct_generic_params.get(enum_base, []),
        )
        substitutions = dict(zip(generic_params, generic_args))
        return {
            field_name: self.substitute_type_parameters(field_type, substitutions)
            for field_name, field_type in variant_fields.items()
        }

    def enum_variant_struct_constructor_info(self, type_name, target_type=None):
        if not isinstance(type_name, str) or "::" not in type_name:
            return None

        enum_path, variant_name = self.split_variant_path(
            self.normalize_turbofish_type_name(type_name)
        )
        if enum_path is None or variant_name is None:
            return None

        enum_type = enum_path
        enum_base, enum_args = self.generic_type_parts(enum_type)
        target_type = self.normalize_turbofish_type_name(target_type)
        target_base, target_args = (
            self.generic_type_parts(target_type) if target_type else (None, [])
        )
        if target_base == enum_base and target_args and not enum_args:
            enum_type = target_type
            enum_base, enum_args = self.generic_type_parts(enum_type)

        variants = self.enum_variant_names.get(enum_base, [])
        if variant_name not in variants:
            return None

        field_types = self.resolve_enum_variant_field_types(enum_type, variant_name)
        if not field_types:
            return None

        return enum_type, variant_name, field_types

    def enum_variant_call_path(self, enum_type, variant_name):
        rust_type = self.map_type(enum_type)
        return f"{self.rust_constructor_path(rust_type)}::{variant_name}"

    def normalize_turbofish_type_name(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name.replace("::<", "<", 1)

    def generate_user_function_call_args(self, func_name, args):
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            param_type = self.user_function_param_type(func_name, index)
            target_type = self.user_function_argument_target_type(
                func_name,
                arg,
                param_type,
            )
            if target_type is None:
                generated_args.append(
                    self.generate_with_lambda_capture_move_blocks(
                        arg,
                        later_roots[index],
                        lambda: self.generate_expression(arg),
                        lambda_conflict_roots[index],
                    )
                )
                continue

            arg_expr = self.generate_with_lambda_capture_move_blocks(
                arg,
                later_roots[index],
                lambda: self.generate_expression_with_type(arg, target_type),
                lambda_conflict_roots[index],
            )
            arg_expr = self.normalize_user_function_call_arg(
                arg,
                arg_expr,
                target_type,
            )
            generated_args.append(arg_expr)
        generated_args.extend(self.generate_captured_function_call_args(func_name))
        return generated_args

    def generate_captured_function_call_args(self, func_name):
        captured_params = self.rust_function_capture_params.get(func_name, [])
        return [
            self.generate_expression_with_type(
                IdentifierNode(param.name), param.param_type
            )
            for param in captured_params
        ]

    def user_function_argument_target_type(self, func_name, arg, param_type):
        if not isinstance(arg, LambdaNode):
            return param_type

        callable_type = self.callable_parameter_target_type(func_name, param_type)
        return callable_type or param_type

    def callable_parameter_target_type(self, func_name, param_type):
        if self.callable_return_target_type(param_type) is not None:
            return param_type

        param_name = self.type_name_string(param_type)
        if not param_name:
            return None

        callee = self.user_function_nodes.get(func_name)
        if callee is None:
            return None

        constraints = self.combined_function_generic_constraints(callee)
        for bound in constraints.get(param_name, []):
            if self.callable_return_target_type(bound) is not None:
                return bound
        return None

    def normalize_user_function_call_arg(self, arg, generated_arg, param_type):
        return self.normalize_typed_expression_value(arg, generated_arg, param_type)

    def user_function_param_type(self, func_name, index):
        param_types = self.user_function_param_types.get(func_name, [])
        return param_types[index] if index < len(param_types) else None

    def normalize_vector_typed_expression(self, expr, target_type):
        target_info = self.vector_type_info(target_type)
        source_info = self.vector_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if target_info["size"] != source_info["size"]:
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        lanes = lanes[: target_info["size"]]
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

    def normalize_matrix_typed_expression(self, expr, target_type):
        target_info = self.matrix_type_info(target_type)
        source_info = self.matrix_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if (
            target_info["columns"] != source_info["columns"]
            or target_info["rows"] != source_info["rows"]
        ):
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.matrix_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

    def generate_scalar_constructor_call(self, func_name, args):
        rust_type = self.scalar_constructor_type(func_name)
        if rust_type is None or len(args) != 1:
            return None

        arg = args[0]
        arg_expr = self.generate_expression(arg)

        if rust_type == "bool":
            arg_type = self.expression_result_type(arg)
            if arg_type == "bool":
                return arg_expr
            zero_literal = "0.0" if arg_type in {"float", "double", "half"} else "0"
            return f"({arg_expr} != {zero_literal})"

        return f"({arg_expr} as {rust_type})"

    def generate_vector_constructor_call(self, func_name, vector_info, args):
        rust_type = self.map_type(func_name)
        generated_args, temp_bindings = self.generate_vector_constructor_args(
            vector_info, args
        )
        return self.generate_constructor_call(rust_type, generated_args, temp_bindings)

    def generate_matrix_constructor_call(self, func_name, args):
        rust_type = self.map_type(func_name)
        matrix_info = self.matrix_type_info(func_name)
        generated_args, temp_bindings = self.generate_matrix_constructor_args(
            matrix_info, args
        )
        return self.generate_constructor_call(rust_type, generated_args, temp_bindings)

    def generate_constructor_call(self, rust_type, generated_args, temp_bindings):
        args_str = ", ".join(generated_args)
        constructor = f"{self.rust_constructor_path(rust_type)}::new({args_str})"
        if not temp_bindings:
            return constructor

        bindings = " ".join(
            self.generate_temp_binding_statement(binding) for binding in temp_bindings
        )
        return f"{{ {bindings} {constructor} }}"

    def generate_temp_binding_statement(self, binding):
        name, expr, *type_hint = binding
        if type_hint and type_hint[0]:
            return f"let {name}: {type_hint[0]} = {expr};"
        return f"let {name} = {expr};"

    def generate_matrix_constructor_args(self, matrix_info, args):
        generated_args = []
        temp_bindings = []
        expected_count = matrix_info["columns"] * matrix_info["rows"]
        component_type = matrix_info["component_type"]

        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            arg_matrix_info = self.matrix_type_info(arg_type)
            if arg_matrix_info is not None:
                return self.generate_matrix_conversion_constructor_args(
                    args[0], arg_matrix_info, matrix_info, component_type
                )

            if self.normalize_scalar_type(arg_type) is not None:
                scalar_expr = self.generate_expression_with_type(
                    args[0], component_type
                )
                scalar_expr = self.normalize_scalar_assignment_value(
                    args[0], scalar_expr, arg_type, component_type
                )
                zero = (
                    self.rust_scalar_default_literal(self.map_type(component_type))
                    or "0"
                )
                for column in range(matrix_info["columns"]):
                    for row in range(matrix_info["rows"]):
                        generated_args.append(scalar_expr if row == column else zero)
                return generated_args, temp_bindings

        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= expected_count:
                return generated_args[:expected_count], temp_bindings

        return generated_args, temp_bindings

    def generate_matrix_conversion_constructor_args(
        self, arg, source_info, target_info, component_type
    ):
        temp_bindings = []
        source_expr = self.generate_expression(arg)
        source_expr = self.generate_matrix_lane_source(arg, source_expr, temp_bindings)
        source_expr = self.rust_member_base_expression(source_expr)
        components = ("x", "y", "z", "w")
        zero = self.rust_scalar_default_literal(self.map_type(component_type)) or "0"
        one = (
            "1.0"
            if self.normalize_scalar_type(component_type) in {"f16", "f32", "f64"}
            else "1"
        )
        generated_args = []
        for column in range(target_info["columns"]):
            for row in range(target_info["rows"]):
                if column < source_info["columns"] and row < source_info["rows"]:
                    generated_args.append(
                        self.normalize_constructor_scalar_lane(
                            None,
                            f"{source_expr}.c{column}.{components[row]}",
                            source_info["component_type"],
                            component_type,
                        )
                    )
                else:
                    generated_args.append(one if column == row else zero)
        return generated_args, temp_bindings

    def generate_vector_constructor_args(self, vector_info, args):
        component_type = vector_info["component_type"]
        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                temp_bindings = []
                arg_expr = self.generate_expression(args[0])
                arg_expr = self.normalize_constructor_scalar_lane(
                    args[0], arg_expr, arg_type, component_type
                )
                if not self.is_repeat_safe_expression(args[0]):
                    temp_name = self.next_vector_arg_temp_name()
                    temp_bindings.append((temp_name, arg_expr))
                    arg_expr = temp_name
                return [arg_expr] * vector_info["size"], temp_bindings

        generated_args = []
        temp_bindings = []
        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= vector_info["size"]:
                return generated_args[: vector_info["size"]], temp_bindings

        return generated_args, temp_bindings

    def vector_argument_lane_expressions(
        self, arg, arg_info, temp_bindings, target_component_type=None
    ):
        swizzle_components = self.member_swizzle_components(arg)
        if swizzle_components is not None:
            object_expr = getattr(arg, "object_expr", getattr(arg, "object", None))
            object_value = self.generate_vector_lane_source(
                object_expr, self.generate_expression(object_expr), temp_bindings
            )
            object_value = self.rust_member_base_expression(object_value)
            return [
                self.normalize_constructor_scalar_lane(
                    None,
                    f"{object_value}.{component}",
                    arg_info["component_type"],
                    target_component_type or arg_info["component_type"],
                )
                for component in swizzle_components
            ]

        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_vector_lane_source(arg, arg_expr, temp_bindings)
        arg_expr = self.rust_member_base_expression(arg_expr)
        components = ("x", "y", "z", "w")[: arg_info["size"]]
        return [
            self.normalize_constructor_scalar_lane(
                None,
                f"{arg_expr}.{component}",
                arg_info["component_type"],
                target_component_type or arg_info["component_type"],
            )
            for component in components
        ]

    def normalize_constructor_scalar_lane(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, float)
            and source_type in {"f32", "f64"}
            and target_type in {"f32", "f64"}
        ):
            return generated_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, source_type, target_type
        )

    def matrix_argument_lane_expressions(
        self, arg, matrix_info, temp_bindings, target_component_type=None
    ):
        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_matrix_lane_source(arg, arg_expr, temp_bindings)
        arg_expr = self.rust_member_base_expression(arg_expr)
        target_component_type = target_component_type or matrix_info["component_type"]
        components = ("x", "y", "z", "w")[: matrix_info["rows"]]
        lanes = []
        for column in range(matrix_info["columns"]):
            for component in components:
                lanes.append(
                    self.normalize_constructor_scalar_lane(
                        None,
                        f"{arg_expr}.c{column}.{component}",
                        matrix_info["component_type"],
                        target_component_type,
                    )
                )
        return lanes

    def generate_matrix_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_matrix_arg_temp_name()
        temp_bindings.append((temp_name, generated_expr))
        return temp_name

    def generate_vector_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_vector_arg_temp_name()
        type_hint = self.rust_temp_binding_type_hint(expr)
        if type_hint is None:
            temp_bindings.append((temp_name, generated_expr))
        else:
            temp_bindings.append((temp_name, generated_expr, type_hint))
        return temp_name

    def rust_temp_binding_type_hint(self, expr):
        func_name = (
            self.function_call_name(getattr(expr, "function", None))
            if isinstance(expr, FunctionCallNode)
            else None
        )
        if func_name is None:
            return None

        mapped_name = self.mapped_function_name(
            func_name,
            len(getattr(expr, "arguments", getattr(expr, "args", [])) or []),
            getattr(expr, "arguments", getattr(expr, "args", [])) or [],
        )
        if mapped_name not in {"texture_size", "texture_size_lod", "image_size"}:
            return None

        result_type = self.expression_result_type(expr)
        if result_type is None:
            return None
        return self.map_type(result_type)

    def rust_member_base_expression(self, generated_expr):
        generated_expr = str(generated_expr)
        if generated_expr.startswith("*") and not generated_expr.startswith("(*"):
            return f"({generated_expr})"
        return generated_expr

    def generate_member_access_expression(self, expr):
        access = self.glsl_buffer_block_access(expr)
        if access == "writeonly" and self.assignment_lhs_depth == 0:
            return self.unsupported_glsl_buffer_block_expression(
                "load",
                "writeonly placeholder cannot be read",
                self.expression_result_type(expr),
            )

        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        self.member_object_depth += 1
        try:
            obj = self.generate_expression(obj_expr)
        finally:
            self.member_object_depth -= 1
        obj = self.lazy_static_object_expression(obj_expr, obj)

        swizzle_components = self.member_swizzle_components(expr)
        if swizzle_components is None:
            access = f"{obj}.{self.rust_identifier(member)}"
            if self.should_clone_member_access_value(expr):
                return f"{access}.clone()"
            return access

        if len(swizzle_components) == 1:
            return f"{obj}.{swizzle_components[0]}"

        rust_type = self.swizzle_constructor_type(obj_expr, len(swizzle_components))
        if not self.is_repeat_safe_expression(obj_expr):
            temp_name = self.next_swizzle_temp_name()
            args = ", ".join(
                f"{temp_name}.{component}" for component in swizzle_components
            )
            return (
                f"{{ let {temp_name} = {obj}; "
                f"{self.rust_constructor_path(rust_type)}::new({args}) }}"
            )

        args = ", ".join(f"{obj}.{component}" for component in swizzle_components)
        return f"{self.rust_constructor_path(rust_type)}::new({args})"

    def should_clone_member_access_value(self, expr):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        member_type = self.expression_result_type(expr)
        if member_type is None:
            return False

        return not self.type_is_copy_derivable(
            member_type,
            self.current_generic_param_names,
        )

    def should_clone_array_access_value(self, expr):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        element_type = self.expression_result_type(expr)
        if element_type is None:
            return False

        return not self.type_is_copy_derivable(
            element_type,
            self.current_generic_param_names,
        )

    def generate_identifier_value_expression(self, name):
        name = self.lambda_capture_alias_name(name) or name
        builtin_value = self.rust_builtin_value_expression(name)
        if builtin_value is not None and name not in self.variable_types:
            return builtin_value
        value = self.lazy_static_identifier_expression(name)
        if self.should_clone_identifier_value(name):
            return self.clone_value_expression(value)
        return value

    def rust_builtin_value_expression(self, name):
        return {
            "gl_WorkGroupID": "workgroup_id()",
            "gl_LocalInvocationID": "local_invocation_id()",
            "gl_GlobalInvocationID": "global_invocation_id()",
            "gl_LocalInvocationIndex": "local_invocation_index()",
            "gl_NumWorkGroups": "num_workgroups()",
            "gl_SubgroupInvocationID": "subgroup_invocation_id()",
            "gl_SubgroupSize": "subgroup_size()",
            "gl_SubgroupID": "subgroup_id()",
            "gl_NumSubgroups": "num_subgroups()",
            "SV_GroupID": "workgroup_id()",
            "SV_GroupThreadID": "local_invocation_id()",
            "SV_DispatchThreadID": "global_invocation_id()",
            "SV_GroupIndex": "local_invocation_index()",
            "gl_LaunchIDEXT": "ray_launch_id()",
            "gl_LaunchSizeEXT": "ray_launch_size()",
            "gl_HitTEXT": "ray_hit_t()",
            "gl_HitKindEXT": "ray_hit_kind()",
            "gl_WorldRayOriginEXT": "ray_world_origin()",
            "gl_WorldRayDirectionEXT": "ray_world_direction()",
            "gl_ObjectRayOriginEXT": "ray_object_origin()",
            "gl_ObjectRayDirectionEXT": "ray_object_direction()",
            "gl_RayTminEXT": "ray_t_min()",
            "gl_IncomingRayFlagsEXT": "ray_incoming_flags()",
            "gl_InstanceCustomIndexEXT": "ray_instance_custom_index()",
            "gl_GeometryIndexEXT": "ray_geometry_index()",
        }.get(name)

    def rust_builtin_value_type(self, name):
        return {
            "gl_WorkGroupID": "uvec3",
            "gl_LocalInvocationID": "uvec3",
            "gl_GlobalInvocationID": "uvec3",
            "gl_LocalInvocationIndex": "uint",
            "gl_NumWorkGroups": "uvec3",
            "gl_SubgroupInvocationID": "uint",
            "gl_SubgroupSize": "uint",
            "gl_SubgroupID": "uint",
            "gl_NumSubgroups": "uint",
            "SV_GroupID": "uvec3",
            "SV_GroupThreadID": "uvec3",
            "SV_DispatchThreadID": "uvec3",
            "SV_GroupIndex": "uint",
            "gl_LaunchIDEXT": "uvec3",
            "gl_LaunchSizeEXT": "uvec3",
            "gl_HitTEXT": "float",
            "gl_HitKindEXT": "uint",
            "gl_WorldRayOriginEXT": "vec3",
            "gl_WorldRayDirectionEXT": "vec3",
            "gl_ObjectRayOriginEXT": "vec3",
            "gl_ObjectRayDirectionEXT": "vec3",
            "gl_RayTminEXT": "float",
            "gl_IncomingRayFlagsEXT": "uint",
            "gl_InstanceCustomIndexEXT": "uint",
            "gl_GeometryIndexEXT": "uint",
        }.get(name)

    def is_default_constructible_ray_runtime_type(self, rust_type):
        return str(rust_type) in {
            "BuiltInTriangleIntersectionAttributes",
            "RayDesc",
            "RayQuery",
            "RayTracingAccelerationStructure",
        }

    def should_clone_identifier_value(self, name):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        value_type = self.variable_types.get(name)
        if value_type is None:
            return False

        return not self.type_is_copy_derivable(
            value_type,
            self.current_generic_param_names,
        )

    def clone_value_expression(self, value):
        if isinstance(value, str) and value.startswith("*"):
            return f"({value}).clone()"
        return f"{value}.clone()"

    def swizzle_constructor_type(self, obj_expr, component_count):
        object_type = self.expression_result_type(obj_expr)
        vector_info = self.vector_type_info(object_type)
        component_type = vector_info["component_type"]
        result_type = self.vector_type_for_components(component_type, component_count)
        return self.map_type(result_type)

    def next_swizzle_temp_name(self):
        name = f"__cgl_swizzle_{self.swizzle_temp_counter}"
        self.swizzle_temp_counter += 1
        return name

    def next_vector_arg_temp_name(self):
        name = f"__cgl_vec_arg_{self.vector_arg_temp_counter}"
        self.vector_arg_temp_counter += 1
        return name

    def next_matrix_arg_temp_name(self):
        name = f"__cgl_mat_arg_{self.matrix_arg_temp_counter}"
        self.matrix_arg_temp_counter += 1
        return name

    def next_lambda_capture_temp_name(self, root):
        safe_root = re.sub(r"[^0-9A-Za-z_]+", "_", str(root)).strip("_")
        if not safe_root:
            safe_root = "capture"
        name = f"__cgl_capture_{safe_root}_{self.lambda_capture_temp_counter}"
        self.lambda_capture_temp_counter += 1
        return name

    def is_repeat_safe_expression(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode, LiteralNode)):
            return True
        if isinstance(expr, str):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.is_repeat_safe_expression(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
            return self.is_repeat_safe_expression(
                array_expr
            ) and self.is_repeat_safe_expression(index_expr)
        return False

    def generate_array_index_expression(self, index_expr):
        index = self.generate_expression(index_expr)
        if self.is_usize_compatible_index(index_expr, index):
            return index
        return f"{index} as usize"

    def is_usize_compatible_index(self, index_expr, generated_index):
        if isinstance(index_expr, LiteralNode) and isinstance(index_expr.value, int):
            return index_expr.value >= 0
        if isinstance(index_expr, int):
            return index_expr >= 0
        if isinstance(index_expr, str) and index_expr.isdigit():
            return True
        return generated_index.endswith(" as usize")

    def member_swizzle_components(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None

        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        if not self.vector_type_info(self.expression_result_type(object_expr)):
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(expr, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None
        return components

    def scalar_constructor_type(self, func_name):
        scalar_types = {
            "bool": "bool",
            "char": "char",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "i16": "i16",
            "u16": "u16",
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f16": "f16",
            "f32": "f32",
            "f64": "f64",
        }
        return scalar_types.get(func_name)

    def binary_scalar_result_type(self, left_type, right_type, operator=None):
        comparison_ops = {"<", ">", "<=", ">=", "==", "!="}
        logical_ops = {"&&", "||"}
        if operator in comparison_ops or operator in logical_ops:
            return "bool"

        return self.promoted_scalar_type(left_type, right_type)

    def binary_scalar_operand_type(self, left_type, right_type, operator=None):
        if operator in {"&&", "||"}:
            return None
        return self.promoted_scalar_type(left_type, right_type)

    def binary_composite_operand_type(self, left_type, right_type, operator=None):
        arithmetic_ops = {"+", "-", "*", "/", "%"}
        bitwise_ops = {"&", "|", "^", "<<", ">>"}

        vector_type = self.promoted_vector_type(left_type, right_type)
        if vector_type is not None:
            vector_info = self.vector_type_info(vector_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_type
            return None

        matrix_type = self.promoted_matrix_type(left_type, right_type)
        if matrix_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_type

        vector_scalar_type = self.promoted_vector_scalar_type(left_type, right_type)
        if vector_scalar_type is not None:
            vector_info = self.vector_type_info(vector_scalar_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_scalar_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_scalar_type
            return None

        matrix_scalar_type = self.promoted_matrix_scalar_type(left_type, right_type)
        if matrix_scalar_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_scalar_type
        return None

    def bool_vector_logical_plan(self, left_type, right_type, operator=None):
        if operator not in {"&&", "||"}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and left_vector["component_type"] != "bool":
            return None
        if right_vector is not None and right_vector["component_type"] != "bool":
            return None

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            size = left_vector["size"]
        elif left_vector is not None and right_scalar == "bool":
            size = left_vector["size"]
        elif right_vector is not None and left_scalar == "bool":
            size = right_vector["size"]
        else:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {"size": size, "result_type": result_type}

    def bool_vector_ternary_plan(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        size = condition_info["size"]
        true_type = self.expression_result_type(true_expr)
        false_type = self.expression_result_type(false_expr)
        result_type = self.bool_vector_ternary_result_type(
            true_type, false_type, size, target_type
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None or result_info["size"] != size:
            return None
        return {
            "size": size,
            "result_type": result_type,
            "component_type": result_info["component_type"],
        }

    def bool_vector_ternary_result_type(
        self, true_type, false_type, condition_size, target_type=None
    ):
        target_info = self.vector_type_info(target_type)
        if target_info is not None and target_info["size"] == condition_size:
            return target_type

        true_info = self.vector_type_info(true_type)
        false_info = self.vector_type_info(false_type)
        true_scalar = self.normalize_scalar_type(true_type)
        false_scalar = self.normalize_scalar_type(false_type)

        if true_info is not None and true_info["size"] == condition_size:
            if false_info is not None and false_info["size"] == condition_size:
                return self.promoted_vector_type(true_type, false_type)
            if false_scalar is not None:
                return self.vector_type_for_promoted_scalar(true_info, false_scalar)
            return true_type

        if false_info is not None and false_info["size"] == condition_size:
            if true_scalar is not None:
                return self.vector_type_for_promoted_scalar(false_info, true_scalar)
            return false_type

        if true_scalar is None or false_scalar is None:
            return None
        component_type = self.promoted_scalar_type(true_scalar, false_scalar)
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, condition_size)

    def vector_comparison_plan(self, left_type, right_type, operator=None):
        if operator not in {"<", ">", "<=", ">=", "==", "!="}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_vector["component_type"]
            )
            size = left_vector["size"]
        elif left_vector is not None and right_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_scalar
            )
            size = left_vector["size"]
        elif right_vector is not None and left_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_scalar, right_vector["component_type"]
            )
            size = right_vector["size"]
        else:
            return None

        if component_type is None:
            return None
        component_type = self.normalize_scalar_type(component_type)
        if component_type == "bool" and operator not in {"==", "!="}:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {
            "component_type": component_type,
            "size": size,
            "result_type": result_type,
        }

    def vector_arithmetic_plan(self, left_type, right_type, operator=None):
        if operator not in {"+", "-", "*", "/", "%"}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_vector["component_type"]
            )
            size = left_vector["size"]
            operand_shape = "vector_vector"
        elif left_vector is not None and right_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_scalar
            )
            size = left_vector["size"]
            operand_shape = "vector_scalar"
        elif right_vector is not None and left_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_scalar, right_vector["component_type"]
            )
            size = right_vector["size"]
            operand_shape = "scalar_vector"
        else:
            return None

        component_type = self.normalize_scalar_type(component_type)
        if component_type is None or component_type == "bool":
            return None

        result_type = self.vector_type_for_components(component_type, size)
        if result_type is None:
            return None

        return {
            "component_type": component_type,
            "operand_shape": operand_shape,
            "operator": operator,
            "result_type": result_type,
            "size": size,
        }

    def rust_vector_arithmetic_requires_lanes(self, plan):
        """Return whether Rust output should avoid relying on vector operator traits."""
        if plan["operand_shape"] != "vector_vector":
            return True

        if plan["size"] == 2:
            return True

        # The current standalone Rust contract only defines Vec4 addition.
        if plan["size"] == 4:
            return plan["operator"] != "+"

        if plan["size"] == 3:
            return plan["operator"] in {"/", "%"}

        return True

    def matrix_multiply_plan(self, left_type, right_type, operator=None):
        if operator != "*":
            return None

        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        if left_matrix is None or right_matrix is None:
            return None
        if left_matrix["columns"] != right_matrix["rows"]:
            return None

        component_type = self.promoted_scalar_type(
            left_matrix["component_type"], right_matrix["component_type"]
        )
        if component_type is None:
            return None

        result_type = self.matrix_type_for_dimensions(
            component_type,
            right_matrix["columns"],
            left_matrix["rows"],
        )
        if result_type is None:
            return None

        return {
            "component_type": component_type,
            "left_matrix": left_matrix,
            "result_columns": right_matrix["columns"],
            "result_rows": left_matrix["rows"],
            "result_type": result_type,
            "right_matrix": right_matrix,
            "shared_size": left_matrix["columns"],
        }

    def rust_matrix_vector_requires_lanes(self, plan):
        if not plan["matrix_on_left"]:
            return True

        matrix_info = self.matrix_type_info(plan["left_target_type"])
        vector_info = self.vector_type_info(plan["right_target_type"])
        if matrix_info is None or vector_info is None:
            return True

        component_type = self.normalize_scalar_type(matrix_info["component_type"])
        return not (
            component_type == "f32"
            and matrix_info["columns"] == matrix_info["rows"]
            and matrix_info["columns"] in {3, 4}
            and vector_info["size"] == matrix_info["columns"]
        )

    def binary_matrix_vector_plan(
        self, left_type, right_type, operator=None, target_type=None
    ):
        if operator != "*":
            return None

        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)

        if left_matrix is not None and right_vector is not None:
            if right_vector["size"] != left_matrix["columns"]:
                return None
            return self.build_matrix_vector_plan(
                left_matrix,
                right_vector,
                result_size=left_matrix["rows"],
                target_type=target_type,
                matrix_on_left=True,
            )

        if left_vector is not None and right_matrix is not None:
            if left_vector["size"] != right_matrix["rows"]:
                return None
            return self.build_matrix_vector_plan(
                right_matrix,
                left_vector,
                result_size=right_matrix["columns"],
                target_type=target_type,
                matrix_on_left=False,
            )

        return None

    def build_matrix_vector_plan(
        self, matrix_info, vector_info, result_size, target_type, matrix_on_left
    ):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], vector_info["component_type"]
        )
        if component_type is None:
            return None

        operation_component_type = self.promoted_component_with_target(
            component_type, target_type, result_size
        )
        result_type = self.vector_type_for_components(
            operation_component_type, result_size
        )
        matrix_type = self.matrix_type_for_dimensions(
            operation_component_type, matrix_info["columns"], matrix_info["rows"]
        )
        vector_type = self.vector_type_for_components(
            operation_component_type, vector_info["size"]
        )
        if result_type is None or matrix_type is None or vector_type is None:
            return None

        if matrix_on_left:
            left_target_type = matrix_type
            right_target_type = vector_type
        else:
            left_target_type = vector_type
            right_target_type = matrix_type

        return {
            "result_type": result_type,
            "left_target_type": left_target_type,
            "matrix_on_left": matrix_on_left,
            "right_target_type": right_target_type,
        }

    def promoted_component_with_target(self, component_type, target_type, result_size):
        target_info = self.vector_type_info(target_type)
        if target_info is None or target_info["size"] != result_size:
            return component_type

        target_component_type = self.normalize_scalar_type(
            target_info["component_type"]
        )
        promoted_type = self.promoted_scalar_type(component_type, target_component_type)
        if promoted_type == target_component_type:
            return promoted_type
        return component_type

    def promoted_vector_scalar_type(self, left_type, right_type):
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_scalar is not None:
            return self.vector_type_for_promoted_scalar(left_vector, right_scalar)
        if right_vector is not None and left_scalar is not None:
            return self.vector_type_for_promoted_scalar(right_vector, left_scalar)
        return None

    def vector_type_for_promoted_scalar(self, vector_info, scalar_type):
        component_type = self.promoted_scalar_type(
            vector_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, vector_info["size"])

    def promoted_matrix_scalar_type(self, left_type, right_type):
        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_matrix is not None and right_scalar is not None:
            return self.matrix_type_for_promoted_scalar(left_matrix, right_scalar)
        if right_matrix is not None and left_scalar is not None:
            return self.matrix_type_for_promoted_scalar(right_matrix, left_scalar)
        return None

    def matrix_type_for_promoted_scalar(self, matrix_info, scalar_type):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, matrix_info["columns"], matrix_info["rows"]
        )

    def promoted_scalar_type(self, left_type, right_type):
        left = self.normalize_scalar_type(left_type)
        right = self.normalize_scalar_type(right_type)
        if left is None or right is None:
            return None

        ranks = {
            "bool": 0,
            "i16": 1,
            "u16": 2,
            "i32": 3,
            "u32": 4,
            "i64": 5,
            "u64": 6,
            "f16": 7,
            "f32": 8,
            "f64": 9,
        }
        return left if ranks[left] >= ranks[right] else right

    def promoted_vector_type(self, left_type, right_type):
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if left_info["size"] != right_info["size"]:
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, left_info["size"])

    def promoted_matrix_type(self, left_type, right_type):
        left_info = self.matrix_type_info(left_type)
        right_info = self.matrix_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if (
            left_info["columns"] != right_info["columns"]
            or left_info["rows"] != right_info["rows"]
        ):
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, left_info["columns"], left_info["rows"]
        )

    def normalize_binary_scalar_operand(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        if source_type is None or source_type == target_type:
            return generated_expr

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{self.integer_literal_expression_value(expr)}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        return f"({generated_expr} as {target_type})"

    def is_integer_literal_expression(self, expr):
        if isinstance(expr, int) and not isinstance(expr, bool):
            return True
        return (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        )

    def integer_literal_expression_value(self, expr):
        return expr.value if isinstance(expr, LiteralNode) else expr

    def normalize_scalar_type(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        aliases = {
            "bool": "bool",
            "char": "i32",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "half": "f16",
            "float": "f32",
            "double": "f64",
            "i16": "i16",
            "u16": "u16",
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f16": "f16",
            "f32": "f32",
            "f64": "f64",
        }
        return aliases.get(type_name)

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def register_variable_type(self, name, type_name, scope="local"):
        if not name or type_name is None:
            return
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        self.variable_types[name] = type_name
        if scope == "local":
            self.local_variable_names.add(name)

    def lazy_static_identifier_expression(self, name):
        if self.is_static_reference(name):
            symbol = self.static_symbol_name(name)
            if self.is_lazy_static_reference(name):
                return f"*{symbol}"
            pointer_parts = self.pointer_type_parts(
                self.map_type(self.variable_types.get(name))
            )
            if pointer_parts is not None:
                is_mutable, pointee_type = pointer_parts
                pointer_keyword = "mut" if is_mutable else "const"
                return f"({symbol} as *{pointer_keyword} {pointee_type})"
            return symbol
        return self.rust_identifier(name)

    def lazy_static_object_expression(self, expr, generated_expr):
        name = self.get_expression_name(expr)
        if name is not None and self.is_lazy_static_reference(name):
            return f"({generated_expr})"
        return generated_expr

    def is_static_reference(self, name):
        return (
            name in self.static_variable_names and name not in self.local_variable_names
        )

    def is_lazy_static_reference(self, name):
        return name in self.lazy_static_names and name not in self.local_variable_names

    def static_symbol_name(self, name):
        return self.static_symbol_names.get(name, name)

    def rust_identifier(self, name):
        """Return a Rust-safe value/field identifier for source-level names."""
        name = str(name)
        if name in {"self", "Self", "super", "crate"}:
            return f"{name}_"
        if name in self.rust_reserved_identifiers():
            return f"{name}_"
        return name

    def rust_callable_name(self, name):
        if isinstance(name, str) and name.isidentifier():
            return self.rust_identifier(name)
        return name

    def rust_reserved_identifiers(self):
        return {
            "as",
            "async",
            "await",
            "become",
            "box",
            "break",
            "const",
            "continue",
            "do",
            "dyn",
            "else",
            "enum",
            "extern",
            "false",
            "final",
            "fn",
            "for",
            "if",
            "impl",
            "in",
            "let",
            "loop",
            "macro",
            "match",
            "mod",
            "move",
            "mut",
            "override",
            "priv",
            "pub",
            "ref",
            "return",
            "static",
            "struct",
            "trait",
            "true",
            "try",
            "type",
            "typeof",
            "union",
            "unsafe",
            "unsized",
            "use",
            "virtual",
            "where",
            "while",
            "yield",
        }

    def is_generated_struct_type(self, type_name):
        if type_name is None:
            return False
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name in self.struct_member_types

    def get_expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.get_expression_name(array_expr)
        return None

    def block_expression_result_type(self, expr):
        explicit_tail = getattr(expr, "expression", None)
        if explicit_tail is not None:
            return self.expression_result_type(explicit_tail)

        statements = self.statement_list(expr)
        if not statements:
            return None

        tail = statements[-1]
        tail_expression = self.block_tail_expression(tail)
        if tail_expression is not None:
            return self.expression_result_type(tail_expression)
        if isinstance(tail, ReturnNode):
            return self.expression_result_type(getattr(tail, "value", None))
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, int):
            return "int"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, ArrayAccessNode):
            return self.array_access_element_type(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = self.get_expression_name(expr)
            variable_type = self.variable_types.get(name)
            if variable_type is not None:
                return variable_type
            return self.rust_builtin_value_type(name)
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int):
                return "int"
            return None
        if isinstance(expr, BlockNode):
            return self.block_expression_result_type(expr)
        if isinstance(expr, ConstructorNode):
            return self.constructor_result_type(expr)
        if isinstance(expr, TextureNode):
            texture_type = self.expression_result_type(expr.texture_expr)
            return "float" if self.is_shadow_texture_type(texture_type) else "vec4"
        if isinstance(expr, TextureOpNode):
            operation = getattr(expr, "operation", "")
            if operation in {
                "SampleCmp",
                "SampleCmpLevel",
                "SampleCmpLevelZero",
                "SampleCmpGrad",
                "CalculateLevelOfDetail",
                "CalculateLevelOfDetailUnclamped",
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
            if operation in {"textureQueryLod"}:
                return "vec2"
            if operation in {"textureQueryLevels", "textureSamples"}:
                return "int"
            if operation in {"GetDimensions"}:
                return "void"
            if operation in {"textureDimensions"}:
                texture_type = self.expression_result_type(expr.texture_expr)
                return self.texture_dimensions_result_type(texture_type)
            if operation in {"textureSize"}:
                texture_type = self.expression_result_type(expr.texture_expr)
                return self.texture_size_result_type(texture_type)
            texture_type = self.expression_result_type(expr.texture_expr)
            if operation in {"Sample", "sample", "texture"} and (
                self.is_shadow_texture_type(texture_type)
            ):
                return "float"
            return "vec4"
        if isinstance(expr, WaveOpNode):
            return self.wave_op_result_type(
                getattr(expr, "operation", ""),
                getattr(expr, "arguments", []),
            )
        if isinstance(expr, MatchNode):
            return self.match_expression_result_type(expr)
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            arguments = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if isinstance(func_expr, MemberAccessNode):
                texture_member_type = self.texture_member_function_result_type(
                    func_expr,
                    arguments,
                )
                if texture_member_type is not None:
                    return texture_member_type
                receiver_type = self.expression_result_type(
                    getattr(
                        func_expr, "object_expr", getattr(func_expr, "object", None)
                    )
                )
                if receiver_type is not None and func_expr.member in {
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "rem",
                }:
                    return receiver_type
            func_name = getattr(func_expr, "name", func_expr)
            if isinstance(func_name, str) and self.vector_type_info(func_name):
                return func_name
            if isinstance(func_name, str) and self.matrix_type_info(func_name):
                return func_name
            enum_result_type = self.enum_variant_call_result_type(func_name)
            if enum_result_type is not None:
                return enum_result_type
            if isinstance(func_name, str) and "::" in func_name:
                base_name = func_name.split("::", 1)[0]
                if base_name in self.current_generic_param_names:
                    return base_name
            scalar_type = self.scalar_constructor_type(func_name)
            if scalar_type is not None:
                return scalar_type
            return_type = self.user_function_return_types.get(func_name)
            if return_type and return_type != "void":
                return return_type
            builtin_type = self.builtin_function_result_type(func_name, arguments)
            if builtin_type is not None:
                return builtin_type
            return None
        if isinstance(expr, BinaryOpNode):
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            operator = self.map_operator(
                getattr(expr, "operator", getattr(expr, "op", None))
            )
            bool_vector_logical_plan = self.bool_vector_logical_plan(
                left_type, right_type, operator
            )
            if bool_vector_logical_plan is not None:
                return bool_vector_logical_plan["result_type"]
            vector_comparison_plan = self.vector_comparison_plan(
                left_type, right_type, operator
            )
            if vector_comparison_plan is not None:
                return vector_comparison_plan["result_type"]
            matrix_multiply_plan = self.matrix_multiply_plan(
                left_type, right_type, operator
            )
            if matrix_multiply_plan is not None:
                return matrix_multiply_plan["result_type"]
            matrix_vector_plan = self.binary_matrix_vector_plan(
                left_type, right_type, operator
            )
            if matrix_vector_plan is not None:
                return matrix_vector_plan["result_type"]
            composite_type = self.binary_composite_operand_type(
                left_type, right_type, operator
            )
            if composite_type is not None:
                return composite_type
            if self.vector_type_info(left_type):
                return left_type
            if self.vector_type_info(right_type):
                return right_type
            if self.matrix_type_info(left_type):
                return left_type
            if self.matrix_type_info(right_type):
                return right_type
            scalar_type = self.binary_scalar_result_type(
                left_type,
                right_type,
                operator,
            )
            if scalar_type is not None:
                return scalar_type
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, TernaryOpNode):
            vector_ternary_plan = self.bool_vector_ternary_plan(
                expr.condition, expr.true_expr, expr.false_expr
            )
            if vector_ternary_plan is not None:
                return vector_ternary_plan["result_type"]
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            vector_type = self.promoted_vector_type(true_type, false_type)
            if vector_type is not None:
                return vector_type
            if self.vector_type_info(true_type):
                return true_type
            if self.vector_type_info(false_type):
                return false_type
            matrix_type = self.promoted_matrix_type(true_type, false_type)
            if matrix_type is not None:
                return matrix_type
            if self.matrix_type_info(true_type):
                return true_type
            if self.matrix_type_info(false_type):
                return false_type
            scalar_type = self.promoted_scalar_type(true_type, false_type)
            if scalar_type is not None:
                return scalar_type
            return true_type or false_type
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.expression_result_type(object_expr)
            object_type_name = (
                self.convert_type_node_to_string(object_type)
                if object_type is not None
                and (
                    hasattr(object_type, "name") or hasattr(object_type, "element_type")
                )
                else object_type
            )
            member = getattr(expr, "member", "")
            member_type = self.resolve_struct_member_type(object_type_name, member)
            if member_type is not None:
                return member_type

            vector_info = self.vector_type_info(object_type)
            if not vector_info:
                return None
            if len(member) == 1:
                return vector_info["component_type"]
            if all(component in "xyzwrgba" for component in member):
                return self.vector_type_for_components(
                    vector_info["component_type"], len(member)
                )
        return None

    def builtin_function_result_type(self, func_name, arguments):
        """Infer result types for shader intrinsics emitted as Rust prelude calls."""
        if not isinstance(func_name, str):
            return None

        mapped_name = self.mapped_function_name(func_name, len(arguments), arguments)
        arg_types = [self.expression_result_type(arg) for arg in arguments]

        wave_type = self.wave_helper_result_type(mapped_name, arg_types)
        if wave_type is not None:
            return wave_type

        if mapped_name in {
            "sample",
            "sample_bias",
            "sample_sampler",
            "sample_bias_sampler",
            "sample_lod",
            "sample_lod_sampler",
            "sample_lod_offset",
            "sample_lod_offset_sampler",
            "sample_grad",
            "sample_grad_sampler",
            "sample_grad_offset",
            "sample_grad_offset_sampler",
            "sample_offset",
            "sample_offset_bias",
            "sample_offset_sampler",
            "sample_offset_bias_sampler",
            "sample_projected",
            "sample_projected_bias",
            "sample_projected_sampler",
            "sample_projected_bias_sampler",
            "sample_projected_lod",
            "sample_projected_lod_sampler",
            "sample_projected_grad",
            "sample_projected_grad_sampler",
            "sample_projected_offset",
            "sample_projected_offset_bias",
            "sample_projected_offset_sampler",
            "sample_projected_offset_bias_sampler",
            "sample_projected_lod_offset",
            "sample_projected_lod_offset_sampler",
            "sample_projected_grad_offset",
            "sample_projected_grad_offset_sampler",
            "texel_fetch",
            "texel_fetch_offset",
            "texture_gather",
            "texture_gather_sampler",
            "texture_gather_component",
            "texture_gather_component_sampler",
            "texture_gather_offset",
            "texture_gather_offset_sampler",
            "texture_gather_offset_component",
            "texture_gather_offset_component_sampler",
            "texture_gather_offsets",
            "texture_gather_offsets_sampler",
            "texture_gather_offsets_component",
            "texture_gather_offsets_component_sampler",
        }:
            return "vec4"

        if mapped_name.startswith("sample_shadow"):
            return "float"

        if mapped_name in {"texture_size", "texture_size_lod"} and arg_types:
            if func_name == "textureDimensions":
                return self.texture_dimensions_result_type(arg_types[0])
            return self.texture_size_result_type(arg_types[0])

        if mapped_name in {"texture_query_levels", "texture_samples"}:
            return "int"

        if mapped_name in {"texture_query_lod", "texture_query_lod_sampler"}:
            return "vec2"

        if mapped_name.startswith("texture_compare"):
            return "float"

        if mapped_name.startswith("texture_gather_compare"):
            return "vec4"

        if mapped_name in {"image_load", "image_load_sample"} and arg_types:
            return self.storage_image_value_result_type(arg_types[0])

        if mapped_name == "image_size" and arg_types:
            return self.storage_image_size_result_type(arg_types[0])

        if mapped_name == "image_samples":
            return "int"

        if mapped_name.startswith("image_atomic_") and arg_types:
            return self.storage_image_atomic_result_type(arg_types[0])

        if mapped_name == "buffer_load" and arg_types:
            return self.buffer_element_result_type(arg_types[0])

        if mapped_name == "buffer_consume" and arg_types:
            return self.buffer_element_result_type(arg_types[0])

        if mapped_name in {
            "image_store",
            "image_store_sample",
            "buffer_store",
            "buffer_dimensions",
            "buffer_append",
        }:
            return "void"

        if (
            mapped_name in {"normalize", "reflect", "refract", "faceforward"}
            and arg_types
        ):
            return arg_types[0]

        if mapped_name == "cross" and len(arg_types) >= 2:
            return self.promoted_value_type(arg_types[0], arg_types[1]) or arg_types[0]

        if mapped_name in {"dot", "length", "distance"} and arg_types:
            return self.vector_or_scalar_component_type(arg_types[0])

        if mapped_name == "outer_product" and len(arg_types) >= 2:
            return self.outer_product_result_type(arg_types[0], arg_types[1])

        if mapped_name == "matrix_comp_mult" and len(arg_types) >= 2:
            return self.promoted_matrix_type(arg_types[0], arg_types[1])

        if mapped_name == "transpose" and arg_types:
            matrix_info = self.matrix_type_info(arg_types[0])
            if matrix_info is not None:
                return self.matrix_type_for_dimensions(
                    matrix_info["component_type"],
                    matrix_info["rows"],
                    matrix_info["columns"],
                )

        if mapped_name == "determinant" and arg_types:
            matrix_info = self.matrix_type_info(arg_types[0])
            if matrix_info is not None:
                return matrix_info["component_type"]

        if mapped_name == "inverse" and arg_types:
            matrix_info = self.matrix_type_info(arg_types[0])
            if matrix_info is not None:
                return arg_types[0]

        if (
            mapped_name
            in {
                "sqrt",
                "rsqrt",
                "abs",
                "floor",
                "ceil",
                "round",
                "trunc",
                "round_even",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "exp",
                "exp2",
                "log",
                "log2",
                "degrees",
                "radians",
                "fract",
                "sign",
            }
            and arg_types
        ):
            return arg_types[0]

        if mapped_name in {"isnan", "isinf", "isfinite"} and arg_types:
            return self.boolean_value_type(arg_types[0])

        if mapped_name in {"any", "all"} and arg_types:
            return "bool"

        relational_operator = {
            "less_than": "<",
            "less_than_equal": "<=",
            "greater_than": ">",
            "greater_than_equal": ">=",
            "equal": "==",
            "not_equal": "!=",
        }.get(mapped_name)
        if relational_operator is not None and len(arg_types) >= 2:
            return self.relational_function_result_type(
                arg_types[0], arg_types[1], relational_operator
            )

        if mapped_name in {"bit_count", "find_lsb", "find_msb"} and arg_types:
            return self.integer_index_value_type(arg_types[0])

        if mapped_name in {"bitfield_reverse", "bitfield_extract"} and arg_types:
            return arg_types[0]

        if mapped_name == "bitfield_insert" and len(arg_types) >= 2:
            return self.promoted_value_type(arg_types[0], arg_types[1]) or arg_types[0]

        bitcast_component_type = {
            "float_bits_to_int": "int",
            "float_bits_to_uint": "uint",
            "int_bits_to_float": "float",
            "uint_bits_to_float": "float",
        }.get(mapped_name)
        if bitcast_component_type is not None and arg_types:
            return self.bitcast_value_type(arg_types[0], bitcast_component_type)

        if mapped_name in {
            "pack_unorm_2x16",
            "pack_snorm_2x16",
            "pack_unorm_4x8",
            "pack_snorm_4x8",
            "pack_half_2x16",
        }:
            return "uint"

        if mapped_name == "pack_double_2x32":
            return "double"

        unpack_vector_type = {
            "unpack_unorm_2x16": "vec2",
            "unpack_snorm_2x16": "vec2",
            "unpack_half_2x16": "vec2",
            "unpack_double_2x32": "uvec2",
            "unpack_unorm_4x8": "vec4",
            "unpack_snorm_4x8": "vec4",
        }.get(mapped_name)
        if unpack_vector_type is not None:
            return unpack_vector_type

        if (
            mapped_name in {"min", "max", "pow", "modulo", "atan2"}
            and len(arg_types) >= 2
        ):
            return self.promoted_value_type(arg_types[0], arg_types[1])

        if mapped_name == "fma" and len(arg_types) >= 3:
            return self.promoted_argument_type(arg_types[:3])

        if mapped_name == "ldexp" and arg_types:
            return arg_types[0]

        if mapped_name == "clamp" and arg_types:
            return self.promoted_argument_type(arg_types[:3])

        if mapped_name == "lerp" and len(arg_types) >= 2:
            return self.promoted_value_type(arg_types[0], arg_types[1])

        if mapped_name == "smoothstep" and len(arg_types) >= 3:
            return self.promoted_argument_type(arg_types[:3])

        if mapped_name == "step" and len(arg_types) >= 2:
            return self.promoted_argument_type(arg_types[:2])

        if (
            mapped_name
            in {
                "dfdx",
                "dfdy",
                "dfdx_fine",
                "dfdy_fine",
                "dfdx_coarse",
                "dfdy_coarse",
                "fwidth",
                "fwidth_fine",
                "fwidth_coarse",
            }
            and arg_types
        ):
            return arg_types[0]

        return None

    def texture_size_result_type(self, texture_type):
        texture_name = str(texture_type or "")
        if texture_name in {"sampler1D", "Texture1D<f32>"}:
            return "int"
        if texture_name in {
            "sampler1DArray",
            "sampler2D",
            "samplerCube",
            "Texture1DArray<f32>",
            "Texture2D<f32>",
            "TextureCube<f32>",
            "sampler2DMS",
            "Texture2DMS<f32>",
            "sampler2DShadow",
            "samplerCubeShadow",
            "DepthTexture2D<f32>",
            "DepthTextureCube<f32>",
        }:
            return "ivec2"
        if texture_name in {
            "sampler2DArray",
            "sampler3D",
            "samplerCubeArray",
            "Texture2DArray<f32>",
            "Texture3D<f32>",
            "TextureCubeArray<f32>",
            "sampler2DMSArray",
            "Texture2DMSArray<f32>",
            "sampler2DArrayShadow",
            "samplerCubeArrayShadow",
            "DepthTexture2DArray<f32>",
            "DepthTextureCubeArray<f32>",
        }:
            return "ivec3"
        return "ivec2"

    def texture_dimensions_result_type(self, texture_type):
        size_type = self.texture_size_result_type(texture_type)
        size_info = self.vector_type_info(size_type)
        if size_info is not None:
            return self.vector_type_for_components("uint", size_info["size"])
        if self.normalize_scalar_type(size_type) is not None:
            return "uint"
        return size_type

    def storage_image_value_result_type(self, image_type):
        image_type = str(image_type or "")
        element_type = self.storage_image_runtime_element_type(image_type)
        if element_type is not None:
            return self.rust_storage_image_element_crossgl_type(element_type)
        if image_type.startswith("uimage") or "Vec4<u32>" in image_type:
            return "uvec4"
        if image_type.startswith("iimage") or "Vec4<i32>" in image_type:
            return "ivec4"
        return "vec4"

    def storage_image_size_result_type(self, image_type):
        image_base, _ = self.generic_type_parts(image_type)
        image_base = image_base.rsplit("::", 1)[-1]
        if image_base in {"image1D", "iimage1D", "uimage1D", "Image1D"}:
            return "int"
        if image_base in {
            "image1DArray",
            "iimage1DArray",
            "uimage1DArray",
            "image2D",
            "iimage2D",
            "uimage2D",
            "imageCube",
            "Image1DArray",
            "Image2D",
            "ImageCube",
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "Image2DMS",
        }:
            return "ivec2"
        if image_base in {
            "image3D",
            "iimage3D",
            "uimage3D",
            "image2DArray",
            "iimage2DArray",
            "uimage2DArray",
            "Image3D",
            "Image2DArray",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
            "Image2DMSArray",
        }:
            return "ivec3"
        return "ivec2"

    def storage_image_atomic_result_type(self, image_type):
        image_type = str(image_type or "")
        element_type = self.storage_image_runtime_element_type(image_type)
        if element_type is not None and "i32" in element_type:
            return "int"
        if image_type.startswith("iimage") or "Vec4<i32>" in image_type:
            return "int"
        return "uint"

    def storage_image_type_with_attributes(self, type_name, attributes):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            image_type = self.storage_image_type_with_attributes(base_type, attributes)
            if image_type is None:
                return None
            return self.format_c_array_type(image_type, sizes)

        mapped_type = self.type_mapping.get(type_name)
        if mapped_type is None or not mapped_type.startswith("Image"):
            return None

        format_name = self.storage_image_format_attribute(attributes)
        if format_name is None:
            return None

        element_type = self.storage_image_format_element_type(format_name)
        if element_type is None:
            return None

        runtime_base = mapped_type.split("<", 1)[0]
        return f"{runtime_base}<{element_type}>"

    def resource_type_with_attributes(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        glsl_buffer_type = self.glsl_buffer_array_resource_type(type_name, node)
        if glsl_buffer_type is not None:
            return glsl_buffer_type

        buffer_type = self.buffer_type_with_access(type_name, node)
        image_type = self.storage_image_type_with_attributes(
            buffer_type,
            getattr(node, "attributes", []),
        )
        return image_type or buffer_type

    def is_glsl_buffer_array_declaration(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" not in qualifiers:
            return False
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        if not attributes.intersection({"std140", "std430", "scalar"}):
            return False
        node_type = glsl_buffer_block_node_type(node)
        if getattr(node_type, "__class__", None).__name__ == "ArrayType":
            return True
        type_name = self.type_name_string(node_type)
        return isinstance(type_name, str) and type_name.endswith("[]")

    def glsl_buffer_array_resource_type(self, type_name, node):
        if not self.is_glsl_buffer_array_declaration(node):
            return None
        base_type, sizes = self.c_array_type_parts(type_name)
        if not sizes or len(sizes) != 1 or sizes[0] is not None:
            return None
        access = self.explicit_resource_access(node) or "readwrite"
        resource_type = (
            "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
        )
        return f"{resource_type}<{base_type}>"

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
            layout = self.attribute_argument_text(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_variable(self, node):
        return self.glsl_buffer_block_attribute(node) is not None

    def register_glsl_buffer_block_metadata(self, node):
        if not self.is_glsl_buffer_block_variable(node):
            return
        name = getattr(node, "name", None)
        if not name:
            return
        self.glsl_buffer_block_accesses[name] = (
            self.explicit_resource_access(node) or "readwrite"
        )
        self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)

    def glsl_buffer_block_metadata_comment(self, node):
        if not self.is_glsl_buffer_block_variable(node):
            return ""
        name = getattr(node, "name", "")
        access = self.explicit_resource_access(node) or "readwrite"
        layout = self.glsl_buffer_block_layout(node)
        return (
            f"// CrossGL resource metadata: name={name} "
            f"kind=glsl_buffer_block layout={layout} access={access}\n"
        )

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None)
        if isinstance(expr, str):
            return expr
        if isinstance(expr, MemberAccessNode):
            return self.glsl_buffer_block_root_name(
                getattr(expr, "object_expr", getattr(expr, "object", None))
            )
        if isinstance(expr, ArrayAccessNode):
            return self.glsl_buffer_block_root_name(
                getattr(expr, "array_expr", getattr(expr, "array", None))
            )
        return None

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if not root_name:
            return None
        return self.glsl_buffer_block_accesses.get(root_name)

    def unsupported_glsl_buffer_block_expression(self, operation, reason, result_type):
        fallback = self.rust_default_value_for_type(result_type)
        return (
            f"/* unsupported Rust GLSL buffer block: {operation} {reason} */ {fallback}"
        )

    def unsupported_glsl_buffer_block_statement(self, operation, reason):
        return f"/* unsupported Rust GLSL buffer block: {operation} {reason} */"

    def rust_default_value_for_type(self, type_name):
        rust_type = self.map_type(type_name)
        scalar = self.rust_scalar_default_literal(rust_type)
        if scalar is not None:
            return scalar
        return "Default::default()"

    def explicit_resource_access(self, node):
        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "") or "").lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                access = access_names.get(
                    str(self.attribute_argument_text(arguments[0])).lower()
                )
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def buffer_type_with_access(self, type_name, node):
        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type, sizes = (
            self.c_array_type_parts(type_name)
            if self.is_array_type_name(type_name)
            else (type_name, [])
        )
        generic_base, generic_args = self.generic_type_parts(base_type)
        mapped_base = None

        if generic_base in {"StructuredBuffer", "RWStructuredBuffer"}:
            mapped_base = (
                "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
            )
        elif generic_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )

        if mapped_base is None:
            return type_name

        if generic_args:
            mapped = f"{mapped_base}<{', '.join(generic_args)}>"
        else:
            mapped = mapped_base
        return self.format_c_array_type(mapped, sizes) if sizes else mapped

    def storage_image_format_attribute(self, attributes):
        for attr in attributes or []:
            name = str(getattr(attr, "name", "") or "")
            if name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_argument_text(arguments[0])
                continue
            if self.storage_image_format_element_type(name) is not None:
                return name
        return None

    def literal_int_value(self, value):
        """Return an integer value for literal integer AST nodes/text."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, UnaryOpNode) and not getattr(value, "is_postfix", False):
            operator = getattr(value, "operator", getattr(value, "op", None))
            if operator not in {"+", "-"}:
                return None
            operand_value = self.literal_int_value(value.operand)
            if operand_value is None:
                return None
            return -operand_value if operator == "-" else operand_value
        if hasattr(value, "value"):
            return self.literal_int_value(value.value)
        if hasattr(value, "name"):
            return None

        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"[uUlL]+$", "", text)
        try:
            return int(text, 0)
        except ValueError:
            return None

    def attribute_argument_text(self, argument):
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return str(argument)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
        raw_value = self.attribute_argument_text(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def resource_register_space_value(self, value):
        raw_value = self.attribute_argument_text(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_rust_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_rust_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if attr_name in {"set", "group"} and arguments:
                set_index = self.resource_binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.resource_register_space_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.resource_register_space_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def rust_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_argument_text(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def rust_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            count = 1
            for size in sizes:
                if size is not None:
                    count *= int(size)
            return base_type, count
        return type_name, 1

    def rust_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_variable(node):
            return "glsl_buffer_block"

        base_type, _count = self.rust_resource_base_type_and_count(type_name)
        generic_base, _generic_args = self.generic_type_parts(base_type)
        base_name = generic_base.rsplit("::", 1)[-1]
        if base_name in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "Buffer",
            "RwBuffer",
            "AppendBuffer",
            "ConsumeBuffer",
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RwByteAddressBuffer",
        }:
            return "buffer"

        mapped_type = self.type_mapping.get(base_name, base_name)
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        if mapped_base == "RayTracingAccelerationStructure":
            return "acceleration_structure"
        if mapped_base == "Sampler":
            return "sampler"
        if mapped_base.startswith(("Image", "IImage", "UImage")):
            return "image"
        if mapped_base.startswith(("Texture", "DepthTexture")):
            return "texture"
        return None

    def rust_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def rust_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.rust_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_rust_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.rust_resource_binding_cursors.get(key, 0)
        while self.rust_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.rust_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_rust_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.rust_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise ValueError(
                    "Conflicting Rust resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        ranges.append((binding, end, name))
        self.rust_resource_binding_cursors[key] = max(
            self.rust_resource_binding_cursors.get(key, 0), end + 1
        )

    def rust_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.rust_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return ""

        _base_type, count = self.rust_resource_base_type_and_count(type_name)
        namespace = self.rust_resource_binding_namespace(resource_kind)
        set_index = self.explicit_rust_resource_set_index(node)
        binding = self.explicit_rust_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_rust_resource_binding(
                namespace, set_index, count
            )
            binding_source = "automatic"
            self.reserve_rust_resource_binding(
                namespace, set_index, binding, count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_rust_resource_binding(
                namespace, set_index, binding, count, name
            )

        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
        ]
        if resource_kind == "glsl_buffer_block":
            parts.append(f"layout={self.glsl_buffer_block_layout(node)}")
            parts.append(f"access={self.explicit_resource_access(node) or 'readwrite'}")
        parts.extend(
            [
                f"set={set_index}",
                f"binding={binding}",
                f"binding_source={binding_source}",
            ]
        )
        if count != 1:
            parts.append(f"count={count}")
        register_metadata = self.rust_resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        return " ".join(parts) + "\n"

    def rust_resource_placeholder_diagnostic_comment(self, node, type_name):
        name = getattr(node, "name", None)
        resource_kind = self.rust_resource_kind(type_name, node=node)
        if not name or resource_kind is None:
            return ""
        return (
            f"// CrossGL Rust limitation: resource {name} is emitted as a "
            "compile-only placeholder static, not a rust-gpu resource binding; "
            "pass real spirv_std resources as #[spirv(...)] entry parameters "
            "when targeting rust-gpu.\n"
        )

    def reserve_explicit_rust_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.rust_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_rust_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.rust_resource_base_type_and_count(type_name)
        namespace = self.rust_resource_binding_namespace(resource_kind)
        set_index = self.explicit_rust_resource_set_index(node)
        self.reserve_rust_resource_binding(namespace, set_index, binding, count, name)

    def reserve_explicit_rust_resource_bindings(self, ast):
        def reserve_variable(node):
            if isinstance(node, ArrayNode):
                element_type = self.convert_type_node_to_string(
                    getattr(node, "element_type", "float")
                )
                size = self.format_array_size(getattr(node, "size", None))
                type_name = (
                    f"{element_type}[{size}]"
                    if size is not None
                    else f"{element_type}[]"
                )
            else:
                type_name = self.get_variable_type(node) or "float"
            type_name = self.resource_type_with_attributes(type_name, node)
            self.reserve_explicit_rust_resource_binding(node, type_name)

        for node in getattr(ast, "global_variables", []) or []:
            reserve_variable(node)
        for cbuffer in self.get_cbuffer_nodes(ast):
            self.reserve_explicit_rust_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        for stage in (
            getattr(ast, "stages", {}).values() if hasattr(ast, "stages") else []
        ):
            for node in getattr(stage, "local_variables", []) or []:
                reserve_variable(node)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_rust_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )

    def storage_image_format_element_type(self, format_name):
        format_name = str(format_name or "").lower()
        match = re.fullmatch(
            r"(rgba|rgb|rg|r)(?:8|16|32|64)?(ui|i|f|unorm|snorm)?",
            format_name,
        )
        if match is None:
            return None

        lanes = {"r": 1, "rg": 2, "rgb": 3, "rgba": 4}[match.group(1)]
        suffix = match.group(2) or "f"
        scalar_type = "u32" if suffix == "ui" else "i32" if suffix == "i" else "f32"
        if lanes == 1:
            return scalar_type
        return f"Vec{lanes}<{scalar_type}>"

    def storage_image_runtime_element_type(self, image_type):
        image_base, image_args = self.generic_type_parts(image_type)
        if not image_args or not image_base.rsplit("::", 1)[-1].startswith("Image"):
            return None
        return image_args[0]

    def rust_storage_image_element_crossgl_type(self, element_type):
        element_type = str(element_type or "")
        element_base, element_args = self.generic_type_parts(element_type)
        if element_args:
            component = element_args[0]
            vector_type_name = element_base.rsplit("::", 1)[-1]
            lane_count = (
                vector_type_name[3:]
                if vector_type_name.startswith("Vec")
                else vector_type_name
            )
            if component == "u32":
                return f"uvec{lane_count}"
            if component == "i32":
                return f"ivec{lane_count}"
            return f"vec{lane_count}"
        if element_type == "u32":
            return "uint"
        if element_type == "i32":
            return "int"
        return "float"

    def buffer_element_result_type(self, buffer_type):
        base_type, generic_args = self.generic_type_parts(buffer_type)
        if generic_args and base_type in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "Buffer",
            "RwBuffer",
            "AppendBuffer",
            "ConsumeBuffer",
        }:
            return generic_args[0]
        if base_type in {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RwByteAddressBuffer",
        }:
            return "uint"
        return None

    def vector_or_scalar_component_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"]
        return self.normalize_scalar_type(type_name)

    def boolean_value_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components("bool", vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return "bool"
        return None

    def outer_product_result_type(self, column_type, row_type):
        column_info = self.vector_type_info(column_type)
        row_info = self.vector_type_info(row_type)
        if column_info is None or row_info is None:
            return None

        component_type = self.promoted_scalar_type(
            column_info["component_type"], row_info["component_type"]
        )
        if component_type is None:
            return None

        return self.matrix_type_for_dimensions(
            component_type, row_info["size"], column_info["size"]
        )

    def relational_function_result_type(self, left_type, right_type, operator):
        vector_plan = self.vector_comparison_plan(left_type, right_type, operator)
        if vector_plan is not None:
            return vector_plan["result_type"]

        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)
        if left_scalar is None or right_scalar is None:
            return None

        component_type = self.promoted_scalar_type(left_scalar, right_scalar)
        if component_type is None:
            return None
        if self.normalize_scalar_type(component_type) == "bool" and operator not in {
            "==",
            "!=",
        }:
            return None
        return "bool"

    def integer_index_value_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components("int", vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return "int"
        return None

    def bitcast_value_type(self, type_name, component_type):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return self.vector_type_for_components(component_type, vector_info["size"])
        if self.normalize_scalar_type(type_name) is not None:
            return component_type
        return None

    def promoted_argument_type(self, arg_types):
        result_type = None
        for arg_type in arg_types:
            if arg_type is None:
                continue
            if result_type is None:
                result_type = arg_type
                continue
            result_type = self.promoted_value_type(result_type, arg_type)
        return result_type

    def promoted_value_type(self, left_type, right_type):
        for promoted in (
            self.promoted_vector_type(left_type, right_type),
            self.promoted_vector_scalar_type(left_type, right_type),
            self.promoted_matrix_type(left_type, right_type),
            self.promoted_matrix_scalar_type(left_type, right_type),
            self.promoted_scalar_type(left_type, right_type),
        ):
            if promoted is not None:
                return promoted
        return left_type or right_type

    def match_expression_result_type(self, expr):
        subject_type = self.expression_result_type(getattr(expr, "expression", None))
        result_type = None

        for arm in getattr(expr, "arms", []) or []:
            saved_variable_types = self.variable_types.copy()
            saved_local_variable_names = self.local_variable_names.copy()
            try:
                self.register_match_pattern_bindings(
                    getattr(arm, "pattern", None),
                    subject_type,
                )
                arm_type = self.match_arm_body_result_type(getattr(arm, "body", None))
            finally:
                self.variable_types = saved_variable_types
                self.local_variable_names = saved_local_variable_names

            if arm_type is None:
                continue
            if result_type is None:
                result_type = arm_type
                continue

            promoted_vector = self.promoted_vector_type(result_type, arm_type)
            if promoted_vector is not None:
                result_type = promoted_vector
                continue

            promoted_scalar = self.promoted_scalar_type(result_type, arm_type)
            if promoted_scalar is not None:
                result_type = promoted_scalar

        return result_type

    def match_arm_body_result_type(self, body):
        statements = self.statement_list(body)
        if not statements:
            return None

        tail = statements[-1]
        if hasattr(tail, "expression"):
            return self.expression_result_type(tail.expression)
        if isinstance(tail, ReturnNode):
            return self.expression_result_type(getattr(tail, "value", None))
        if isinstance(tail, MatchNode):
            return self.expression_result_type(tail)
        return None

    def constructor_result_type(self, expr):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        variant_info = self.enum_variant_struct_constructor_info(type_name)
        if variant_info is not None:
            enum_type, _variant_name, _field_types = variant_info
            return enum_type

        base_type, existing_args = self.generic_type_parts(type_name)
        if existing_args or base_type not in self.struct_generic_params:
            return type_name

        generic_params = self.struct_generic_params.get(base_type, [])
        if not generic_params:
            return type_name

        substitutions = {}
        named_arguments = getattr(expr, "named_arguments", {}) or {}
        for field_name, value in named_arguments.items():
            member_type = self.resolve_struct_member_type(base_type, field_name)
            value_type = self.expression_result_type(value)
            self.collect_type_parameter_bindings(
                member_type,
                value_type,
                substitutions,
                set(generic_params),
            )

        if not all(param in substitutions for param in generic_params):
            return type_name

        args = ", ".join(substitutions[param] for param in generic_params)
        return f"{base_type}<{args}>"

    def resolve_struct_member_type(self, object_type, member):
        if object_type is None:
            return None
        if hasattr(object_type, "name") or hasattr(object_type, "element_type"):
            object_type = self.convert_type_node_to_string(object_type)
        else:
            object_type = str(object_type)

        exact_members = self.struct_member_types.get(object_type, {})
        if member in exact_members:
            return exact_members[member]

        base_type, generic_args = self.generic_type_parts(object_type)
        member_types = self.struct_member_types.get(base_type, {})
        if member not in member_types:
            return None

        member_type = member_types[member]
        generic_params = self.struct_generic_params.get(base_type, [])
        substitutions = dict(zip(generic_params, generic_args))
        return self.substitute_type_parameters(member_type, substitutions)

    def resolve_struct_positional_member_types(self, object_type):
        if object_type is None:
            return []
        if hasattr(object_type, "name") or hasattr(object_type, "element_type"):
            object_type = self.convert_type_node_to_string(object_type)
        else:
            object_type = str(object_type)

        exact_members = self.struct_member_types.get(object_type)
        if exact_members is not None:
            return list(exact_members.values())

        base_type, generic_args = self.generic_type_parts(object_type)
        member_types = self.struct_member_types.get(base_type)
        if not member_types:
            return []

        generic_params = self.struct_generic_params.get(base_type, [])
        substitutions = dict(zip(generic_params, generic_args))
        return [
            self.substitute_type_parameters(member_type, substitutions)
            for member_type in member_types.values()
        ]

    def generic_type_parts(self, type_name):
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "<" not in type_name or not type_name.endswith(">"):
            return type_name, []

        base_type, args_text = type_name.split("<", 1)
        args_text = args_text[:-1]
        args = [arg.strip() for arg in self.split_top_level_generic_args(args_text)]
        return base_type, [arg for arg in args if arg]

    def substitute_type_parameters(self, type_name, substitutions):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if type_name in substitutions:
            return substitutions[type_name]

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            mapped_base = self.substitute_type_parameters(base_type, substitutions)
            return self.format_c_array_type(mapped_base, sizes)

        base_type, generic_args = self.generic_type_parts(type_name)
        if not generic_args:
            return type_name

        mapped_args = [
            self.substitute_type_parameters(arg, substitutions) for arg in generic_args
        ]
        return f"{base_type}<{', '.join(mapped_args)}>"

    def collect_type_parameter_bindings(
        self,
        expected_type,
        actual_type,
        substitutions,
        generic_params,
    ):
        if expected_type is None or actual_type is None:
            return

        expected_type = str(expected_type)
        actual_type = str(actual_type)
        if expected_type in generic_params:
            substitutions.setdefault(expected_type, actual_type)
            return

        expected_base, expected_args = self.generic_type_parts(expected_type)
        actual_base, actual_args = self.generic_type_parts(actual_type)
        if expected_base != actual_base or len(expected_args) != len(actual_args):
            return

        for expected_arg, actual_arg in zip(expected_args, actual_args):
            self.collect_type_parameter_bindings(
                expected_arg,
                actual_arg,
                substitutions,
                generic_params,
            )

    def array_access_element_type(self, expr):
        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        array_type = self.expression_result_type(array_expr)
        if array_type is None:
            array_name = self.get_expression_name(array_expr)
            array_type = self.variable_types.get(array_name)
            if array_type is None:
                return None
        if hasattr(array_type, "name") or hasattr(array_type, "element_type"):
            array_type = self.convert_type_node_to_string(array_type)
        else:
            array_type = str(array_type)
        pointer_parts = self.pointer_type_parts(self.map_type(array_type))
        if pointer_parts is not None:
            return pointer_parts[1]
        if "[" not in array_type or "]" not in array_type:
            patch_info = self.rust_tessellation_patch_info(array_type)
            if patch_info is not None and patch_info["valid"]:
                return patch_info["element_type"]
            return None
        element_type, _size = self.c_array_outer_element_type_and_size(array_type)
        return element_type or None

    def vector_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        mapped_type = self.map_type(type_name)
        mapped_type = self.unqualify_runtime_type_name(mapped_type)
        vector_details = {
            "Vec2<f32>": ("float", 2),
            "Vec3<f32>": ("float", 3),
            "Vec4<f32>": ("float", 4),
            "Vec2<f64>": ("double", 2),
            "Vec3<f64>": ("double", 3),
            "Vec4<f64>": ("double", 4),
            "Vec2<i32>": ("int", 2),
            "Vec3<i32>": ("int", 3),
            "Vec4<i32>": ("int", 4),
            "Vec2<u32>": ("uint", 2),
            "Vec3<u32>": ("uint", 3),
            "Vec4<u32>": ("uint", 4),
            "Vec2<bool>": ("bool", 2),
            "Vec3<bool>": ("bool", 3),
            "Vec4<bool>": ("bool", 4),
        }
        details = vector_details.get(mapped_type)
        if details is None:
            return None
        component_type, size = details
        return {"component_type": component_type, "size": size}

    def unqualify_runtime_type_name(self, type_name):
        type_name = str(type_name)
        if type_name.startswith("math::"):
            return type_name[len("math::") :]
        return type_name

    def matrix_type_info(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        matrix_details = {
            "mat2": (2, 2),
            "mat3": (3, 3),
            "mat4": (4, 4),
            "mat2x2": (2, 2),
            "mat2x3": (2, 3),
            "mat2x4": (2, 4),
            "mat3x2": (3, 2),
            "mat3x3": (3, 3),
            "mat3x4": (3, 4),
            "mat4x2": (4, 2),
            "mat4x3": (4, 3),
            "mat4x4": (4, 4),
            "dmat2": (2, 2),
            "dmat3": (3, 3),
            "dmat4": (4, 4),
            "dmat2x2": (2, 2),
            "dmat2x3": (2, 3),
            "dmat2x4": (2, 4),
            "dmat3x2": (3, 2),
            "dmat3x3": (3, 3),
            "dmat3x4": (3, 4),
            "dmat4x2": (4, 2),
            "dmat4x3": (4, 3),
            "dmat4x4": (4, 4),
        }
        details = matrix_details.get(type_name)
        if details is None:
            return None
        columns, rows = details
        component_type = "double" if type_name.startswith("dmat") else "float"
        return {"columns": columns, "rows": rows, "component_type": component_type}

    def scalar_type_for_type_constructor(self, scalar_type):
        scalar_type = self.normalize_scalar_type(scalar_type)
        aliases = {
            "bool": "bool",
            "i16": "short",
            "u16": "ushort",
            "i32": "int",
            "u32": "uint",
            "i64": "long",
            "u64": "ulong",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        return aliases.get(scalar_type)

    def matrix_type_for_dimensions(self, component_type, columns, rows):
        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type == "double":
            prefix = "dmat"
        elif component_type == "float":
            prefix = "mat"
        else:
            return None

        if columns == rows:
            return f"{prefix}{columns}"
        return f"{prefix}{columns}x{rows}"

    def vector_type_for_components(self, component_type, component_count):
        component_type = self.scalar_type_for_type_constructor(component_type)
        if component_type is None:
            return None
        if component_count < 2 or component_count > 4:
            return component_type
        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        prefix = prefixes.get(component_type)
        if prefix is None:
            return None
        return f"{prefix}{component_count}"

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def map_type(self, vtype):
        """Map a CrossGL type name or type node to a Rust type string."""
        if vtype is None:
            return "f32"

        vtype_str = self.type_name_string(vtype)

        reference_parts = self.reference_type_parts(vtype_str)
        if reference_parts is not None:
            is_mutable, referenced_type = reference_parts
            mutability = "mut " if is_mutable else ""
            return f"&{mutability}{self.map_reference_pointee_type(referenced_type)}"

        pointer_parts = self.pointer_type_parts(vtype_str)
        if pointer_parts is not None:
            is_mutable, pointee_type = pointer_parts
            mutability = "mut" if is_mutable else "const"
            return f"*{mutability} {self.map_reference_pointee_type(pointee_type)}"

        function_pointer = self.map_function_pointer_type(vtype_str)
        if function_pointer is not None:
            return function_pointer

        if self.callable_signature_parts(vtype_str) is not None:
            return vtype_str

        patch_type = self.rust_tessellation_patch_mapped_type(vtype_str)
        if patch_type is not None:
            return patch_type

        if "[" in vtype_str and "]" in vtype_str:
            base_type, sizes = self.c_array_type_parts(vtype_str)
            base_mapped = self.map_type(base_type)
            rust_type = self.qualify_colliding_runtime_type(base_mapped)
            for size in reversed(sizes):
                if size is None:
                    rust_type = f"Vec<{rust_type}>"
                else:
                    rust_type = f"[{rust_type}; {size}]"
            return rust_type

        mapped_type = self.type_mapping.get(vtype_str)
        if mapped_type is not None:
            return self.qualify_colliding_runtime_type(mapped_type)

        generic_type = self.map_generic_type_string(vtype_str)
        if generic_type is not None:
            return generic_type

        return vtype_str

    def map_type_with_lifetime(self, vtype, lifetime):
        if lifetime is None:
            return self.map_type(vtype)

        vtype_str = self.type_name_string(vtype)
        reference_parts = self.reference_type_parts(vtype_str)
        if reference_parts is None:
            return self.map_type(vtype)

        is_mutable, referenced_type = reference_parts
        mutability = "mut " if is_mutable else ""
        return (
            f"&{lifetime} {mutability}"
            f"{self.map_reference_pointee_type(referenced_type)}"
        )

    def reference_type_parts(self, type_name):
        type_name = str(type_name).strip()
        if type_name.startswith("&mut "):
            return True, type_name[len("&mut ") :].strip()
        if type_name.startswith("&") and not type_name.startswith("&'"):
            return False, type_name[1:].strip()
        return None

    def reference_type_parts_for_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        return self.reference_type_parts(type_name)

    def pointer_type_parts(self, type_name):
        type_name = str(type_name).strip()
        if type_name.startswith("*mut "):
            return True, type_name[len("*mut ") :].strip()
        if type_name.startswith("*const "):
            return False, type_name[len("*const ") :].strip()
        return None

    def map_reference_pointee_type(self, type_name):
        type_name = str(type_name).strip()
        if type_name in {"str", "string"}:
            return "str"
        return self.map_type(type_name)

    def map_function_pointer_type(self, type_name):
        type_name = str(type_name).strip()
        if not type_name.startswith("fn"):
            return None

        open_index = type_name.find("(")
        if type_name[:open_index].strip() != "fn":
            return None
        close_index = self.find_matching_delimiter(type_name, open_index, "(", ")")
        if close_index is None:
            return None

        params_text = type_name[open_index + 1 : close_index].strip()
        suffix = type_name[close_index + 1 :].strip()
        if not suffix.startswith("->"):
            return None

        return_type = suffix[2:].strip()
        params = []
        if params_text:
            params = [
                self.map_type(param_type.strip())
                for param_type in self.split_top_level_list(params_text)
                if param_type.strip()
            ]
        return f"fn({', '.join(params)}) -> {self.map_type(return_type)}"

    def qualify_colliding_runtime_type(self, rust_type):
        """Disambiguate built-in math types when user declarations reuse their names."""
        rust_type = str(rust_type)
        if rust_type.startswith("math::"):
            return rust_type
        base_type = rust_type.split("<", 1)[0]
        if base_type in self.runtime_type_collisions:
            return f"math::{rust_type}"
        return rust_type

    def map_generic_type_string(self, type_name):
        """Map primitive arguments inside a generic type string."""
        if "<" not in type_name or not type_name.endswith(">"):
            return None

        base_type, args_text = type_name.split("<", 1)
        args_text = args_text[:-1]
        args = self.split_top_level_generic_args(args_text)
        mapped_args = [self.map_type(arg.strip()) for arg in args if arg.strip()]
        if not mapped_args:
            return None

        mapped_base = self.type_mapping.get(base_type, base_type)
        if "<" in mapped_base:
            mapped_base = mapped_base.split("<", 1)[0]
        return f"{mapped_base}<{', '.join(mapped_args)}>"

    def split_top_level_generic_args(self, args_text):
        """Split generic arguments without breaking nested generic arguments."""
        args = []
        current = []
        depth = 0

        for char in args_text:
            if char == "<":
                depth += 1
            elif char == ">" and depth > 0:
                depth -= 1
            elif char == "," and depth == 0:
                args.append("".join(current))
                current = []
                continue

            current.append(char)

        if current:
            args.append("".join(current))

        return args

    def rust_constructor_path(self, rust_type):
        """Return a Rust path suitable for associated constructor calls."""
        rust_type = str(rust_type)
        if "<" not in rust_type:
            return rust_type
        return rust_type.replace("<", "::<", 1)

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
            "ASSIGN_XOR": "^=",
            "ASSIGN_OR": "|=",
            "ASSIGN_AND": "&=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
            "NOT": "!",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the Rust backend attribute name."""
        if semantic:
            return self.semantic_map.get(semantic, semantic)
        return ""
