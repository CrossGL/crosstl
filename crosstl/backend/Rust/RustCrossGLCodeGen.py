"""Reverse code generator that emits CrossGL from Rust AST nodes."""

import re

from .RustAst import *
from .RustLexer import *
from .RustParser import *

RUST_NUMERIC_LITERAL_RE = re.compile(
    r"^(?P<body>"
    r"0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*|"
    r"0[bB][01](?:_?[01])*|"
    r"0[oO][0-7](?:_?[0-7])*|"
    r"\d(?:_?\d)*(?:(?:\.\d(?:_?\d)*)|\.(?![._A-Za-z0-9]))?"
    r"(?:[eE][+-]?\d(?:_?\d)*)?"
    r")(?P<suffix>(?:[iu](?:8|16|32|64|128|size))|f(?:32|64))?$"
)
RUST_RAW_STRING_RE = re.compile(r'^r(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_BYTE_RAW_STRING_RE = re.compile(r'^br(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_C_RAW_STRING_RE = re.compile(r'^cr(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_STRING_RE = re.compile(r'^"((?:[^"\\]|\\(.|\n))*)"$', re.DOTALL)
RUST_BYTE_STRING_RE = re.compile(r'^b"((?:[^"\\]|\\.)*)"$', re.DOTALL)
RUST_C_STRING_RE = re.compile(r'^c"((?:[^"\\]|\\.)*)"$', re.DOTALL)
RUST_BYTE_CHAR_RE = re.compile(r"^b'((?:[^'\\]|\\.)*)'$", re.DOTALL)

TRANSPARENT_BLOCK_NODE_TYPES = (AsyncBlockNode, UnsafeBlockNode, ConstBlockNode)
BLOCK_EXPRESSION_NODE_TYPES = (BlockNode,) + TRANSPARENT_BLOCK_NODE_TYPES
try:
    from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
except ImportError:
    CROSSGL_KEYWORDS = {}

CROSSGL_RESERVED_IDENTIFIERS = set(CROSSGL_KEYWORDS) | {"true", "false"}


class RustToCrossGLConverter:
    """Serialize Rust backend AST nodes back into CrossGL source."""

    SCALAR_ZERO_ARG_METHOD_MAP = {
        "abs": "abs",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "trunc": "trunc",
        "round_ties_even": "roundEven",
        "fract": "fract",
        "sqrt": "sqrt",
        "exp": "exp",
        "exp2": "exp2",
        "ln": "log",
        "log2": "log2",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "to_degrees": "degrees",
        "to_radians": "radians",
        "signum": "sign",
        "is_nan": "isnan",
        "is_infinite": "isinf",
        "is_finite": "isfinite",
    }
    SCALAR_ONE_ARG_METHOD_MAP = {
        "min": "min",
        "max": "max",
        "powf": "pow",
        "powi": "pow",
        "atan2": "atan2",
        "rem_euclid": "mod",
        "step": "step",
    }
    SCALAR_TWO_ARG_METHOD_MAP = {
        "clamp": "clamp",
        "lerp": "mix",
        "mul_add": "fma",
    }
    RUST_GPU_DERIVATIVE_METHOD_MAP = {
        "dfdx": "dFdx",
        "dfdy": "dFdy",
        "dfdx_fine": "dFdxFine",
        "dfdy_fine": "dFdyFine",
        "dfdx_coarse": "dFdxCoarse",
        "dfdy_coarse": "dFdyCoarse",
        "fwidth": "fwidth",
        "fwidth_fine": "fwidthFine",
        "fwidth_coarse": "fwidthCoarse",
    }
    RUST_GPU_ASSOCIATED_INTRINSIC_TYPES = {"Derivative"}
    RESOURCE_METHOD_PREFIXES = ("sample", "texture_", "image_", "buffer_")
    RESOURCE_METHOD_NAMES = {
        "fetch",
        "fetch_with",
        "gather",
        "query_levels",
        "query_samples",
        "query_size",
        "query_size_lod",
        "read",
        "write",
    }
    RESOURCE_TYPE_PREFIXES = (
        "sampler",
        "image",
        "uimage",
        "iimage",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
    )
    RUST_GPU_SAMPLE_METHOD_MAP = {
        "gather": "textureGather",
        "sample_by_lod": "textureLod",
        "sample_by_gradient": "textureGrad",
        "sample_with_project_coordinate": "textureProj",
        "sample_with_project_coordinate_by_lod": "textureProjLod",
        "sample_with_project_coordinate_by_gradient": "textureProjGrad",
        "sample_depth_reference": "textureCompare",
        "sample_depth_reference_by_lod": "textureCompareLod",
        "sample_depth_reference_by_gradient": "textureCompareGrad",
        "sample_depth_reference_with_project_coordinate": "textureCompareProj",
        "sample_depth_reference_with_project_coordinate_by_lod": (
            "textureCompareProjLod"
        ),
        "sample_depth_reference_with_project_coordinate_by_gradient": (
            "textureCompareProjGrad"
        ),
    }
    RUST_GPU_SAMPLE_WITH_METHOD_MAP = {
        "sample_with": "texture",
        "sample_with_project_coordinate_with": "textureProj",
        "sample_depth_reference_with": "textureCompare",
        "sample_depth_reference_with_project_coordinate_with": "textureCompareProj",
    }
    VECTOR_EXTEND_CONSTRUCTOR_MAP = {
        "vec2": "vec3",
        "vec3": "vec4",
        "ivec2": "ivec3",
        "ivec3": "ivec4",
        "uvec2": "uvec3",
        "uvec3": "uvec4",
        "bvec2": "bvec3",
        "bvec3": "bvec4",
    }
    VECTOR_CONSTRUCTOR_RETURN_TYPES = {
        "vec2": "vec2",
        "vec3": "vec3",
        "vec4": "vec4",
        "ivec2": "ivec2",
        "ivec3": "ivec3",
        "ivec4": "ivec4",
        "uvec2": "uvec2",
        "uvec3": "uvec3",
        "uvec4": "uvec4",
        "bvec2": "bvec2",
        "bvec3": "bvec3",
        "bvec4": "bvec4",
    }
    VECTOR_SPLAT_CONSTRUCTORS = set(VECTOR_CONSTRUCTOR_RETURN_TYPES.values())
    VECTOR_COMPONENT_COUNTS = {
        "vec2": 2,
        "vec3": 3,
        "vec4": 4,
        "ivec2": 2,
        "ivec3": 3,
        "ivec4": 4,
        "uvec2": 2,
        "uvec3": 3,
        "uvec4": 4,
        "bvec2": 2,
        "bvec3": 3,
        "bvec4": 4,
    }
    SCALAR_ASSOCIATED_CONSTANTS = {
        "u8::MAX": "255",
        "u8::MIN": "0",
        "u16::MAX": "65535",
        "u16::MIN": "0",
        "u32::MAX": "4294967295",
        "u32::MIN": "0",
        "i8::MAX": "127",
        "i8::MIN": "(-128)",
        "i16::MAX": "32767",
        "i16::MIN": "(-32768)",
        "i32::MAX": "2147483647",
        "i32::MIN": "(-2147483648)",
        "f32::EPSILON": "1.1920929e-7",
        "f32::INFINITY": "1.0 / 0.0",
        "f32::NEG_INFINITY": "-1.0 / 0.0",
    }
    VECTOR_AXIS_CONSTANT_INDICES = {"X": 0, "Y": 1, "Z": 2, "W": 3}

    def __init__(self):
        self.type_map = {
            # Rust primitive types to CrossGL
            "()": "void",
            "f32": "float",
            "f64": "double",
            "i32": "int",
            "i64": "int64_t",
            "u32": "uint",
            "u64": "uint64_t",
            "i8": "int8_t",
            "u8": "uint8_t",
            "i16": "int16_t",
            "u16": "uint16_t",
            "bool": "bool",
            "usize": "uint",
            "isize": "int",
            # Rust vector types to CrossGL
            "Vec2<f32>": "vec2",
            "Vec3<f32>": "vec3",
            "Vec4<f32>": "vec4",
            "Vec2<i32>": "ivec2",
            "Vec3<i32>": "ivec3",
            "Vec4<i32>": "ivec4",
            "Vec2<u32>": "uvec2",
            "Vec3<u32>": "uvec3",
            "Vec4<u32>": "uvec4",
            "Vec2<bool>": "bvec2",
            "Vec3<bool>": "bvec3",
            "Vec4<bool>": "bvec4",
            # Simplified vector names
            "Vec2": "vec2",
            "Vec3": "vec3",
            "Vec3A": "vec3",
            "Vec4": "vec4",
            "IVec2": "ivec2",
            "IVec3": "ivec3",
            "IVec4": "ivec4",
            "UVec2": "uvec2",
            "UVec3": "uvec3",
            "UVec4": "uvec4",
            "BVec2": "bvec2",
            "BVec3": "bvec3",
            "BVec4": "bvec4",
            # Rust matrix types to CrossGL
            "Mat2<f32>": "mat2",
            "Mat3<f32>": "mat3",
            "Mat4<f32>": "mat4",
            "Mat2x3<f32>": "mat2x3",
            "Mat2x4<f32>": "mat2x4",
            "Mat3x2<f32>": "mat3x2",
            "Mat3x4<f32>": "mat3x4",
            "Mat4x2<f32>": "mat4x2",
            "Mat4x3<f32>": "mat4x3",
            "Mat2<f64>": "dmat2",
            "Mat3<f64>": "dmat3",
            "Mat4<f64>": "dmat4",
            "Mat2x3<f64>": "dmat2x3",
            "Mat2x4<f64>": "dmat2x4",
            "Mat3x2<f64>": "dmat3x2",
            "Mat3x4<f64>": "dmat3x4",
            "Mat4x2<f64>": "dmat4x2",
            "Mat4x3<f64>": "dmat4x3",
            "Mat2": "mat2",
            "Mat3": "mat3",
            "Mat4": "mat4",
            "Mat2x3": "mat2x3",
            "Mat2x4": "mat2x4",
            "Mat3x2": "mat3x2",
            "Mat3x4": "mat3x4",
            "Mat4x2": "mat4x2",
            "Mat4x3": "mat4x3",
            # GPU-specific types
            "Texture1D<f32>": "sampler1D",
            "Texture1DArray<f32>": "sampler1DArray",
            "Texture2D<f32>": "sampler2D",
            "Texture2DArray<f32>": "sampler2DArray",
            "Texture3D<f32>": "sampler3D",
            "TextureCube<f32>": "samplerCube",
            "TextureCubeArray<f32>": "samplerCubeArray",
            "Texture2DMS<f32>": "sampler2DMS",
            "Texture2DMSArray<f32>": "sampler2DMSArray",
            "DepthTexture2D<f32>": "sampler2DShadow",
            "DepthTexture2DArray<f32>": "sampler2DArrayShadow",
            "DepthTextureCube<f32>": "samplerCubeShadow",
            "DepthTextureCubeArray<f32>": "samplerCubeArrayShadow",
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "DepthTexture2D": "sampler2DShadow",
            "DepthTexture2DArray": "sampler2DArrayShadow",
            "DepthTextureCube": "samplerCubeShadow",
            "DepthTextureCubeArray": "samplerCubeArrayShadow",
            "Image1D": "image1D",
            "Image1DArray": "image1DArray",
            "Image2D": "image2D",
            "Image3D": "image3D",
            "ImageCube": "imageCube",
            "Image2DArray": "image2DArray",
            "Image2DMS": "image2DMS",
            "Image2DMSArray": "image2DMSArray",
            "ByteAddressBuffer": "ByteAddressBuffer",
            "RwByteAddressBuffer": "RWByteAddressBuffer",
            "Sampler": "sampler",
            # Reference types (stripped in CrossGL)
            "&f32": "float",
            "&i32": "int",
            "&u32": "uint",
            "&bool": "bool",
            "&mut f32": "float",
            "&mut i32": "int",
            "&mut u32": "uint",
            "&mut bool": "bool",
        }

        self.semantic_map = {
            # Rust shader attributes to CrossGL semantics
            "vertex_position": "Position",
            "vertex_normal": "Normal",
            "vertex_tangent": "Tangent",
            "vertex_binormal": "Binormal",
            "vertex_texcoord": "TexCoord",
            "vertex_texcoord0": "TexCoord0",
            "vertex_texcoord1": "TexCoord1",
            "vertex_texcoord2": "TexCoord2",
            "vertex_texcoord3": "TexCoord3",
            "vertex_color": "Color",
            "base_instance": "gl_BaseInstance",
            "base_vertex": "gl_BaseVertex",
            "device_index": "gl_DeviceIndex",
            "draw_index": "gl_DrawID",
            "instance_id": "InstanceID",
            "instance_index": "InstanceID",
            "vertex_id": "VertexID",
            "vertex_index": "VertexID",
            "clip_distance": "gl_ClipDistance",
            "cull_distance": "gl_CullDistance",
            "frag_depth": "gl_FragDepth",
            "frag_stencil_ref_ext": "gl_FragStencilRefEXT",
            "layer": "gl_Layer",
            "point_size": "gl_PointSize",
            "viewport_index": "gl_ViewportIndex",
            # Fragment shader semantics
            "bary_coord": "gl_BaryCoordEXT",
            "bary_coord_no_persp": "gl_BaryCoordNoPerspEXT",
            "bary_coord_no_persp_amd": "gl_BaryCoordNoPerspAMD",
            "bary_coord_no_persp_centroid_amd": "gl_BaryCoordNoPerspCentroidAMD",
            "bary_coord_no_persp_sample_amd": "gl_BaryCoordNoPerspSampleAMD",
            "bary_coord_pull_model_amd": "gl_BaryCoordPullModelAMD",
            "bary_coord_smooth_amd": "gl_BaryCoordSmoothAMD",
            "bary_coord_smooth_centroid_amd": "gl_BaryCoordSmoothCentroidAMD",
            "bary_coord_smooth_sample_amd": "gl_BaryCoordSmoothSampleAMD",
            "frag_coord": "gl_FragCoord",
            "frag_invocation_count_ext": "gl_FragInvocationCountEXT",
            "frag_size_ext": "gl_FragSizeEXT",
            "fragment_position": "gl_Position",
            "position": "gl_Position",
            "fragment_color": "gl_FragColor",
            "fragment_depth": "gl_FragDepth",
            "front_face": "gl_IsFrontFace",
            "front_facing": "gl_FrontFacing",
            "fully_covered_ext": "gl_FullyCoveredEXT",
            "helper_invocation": "gl_HelperInvocation",
            "primitive_id": "gl_PrimitiveID",
            "point_coord": "gl_PointCoord",
            "sample_id": "gl_SampleID",
            "sample_mask": "gl_SampleMask",
            "sample_position": "gl_SamplePosition",
            # Compute shader semantics
            "local_invocation_id": "gl_LocalInvocationID",
            "local_invocation_index": "gl_LocalInvocationIndex",
            "global_invocation_id": "gl_GlobalInvocationID",
            "workgroup_id": "gl_WorkGroupID",
            "workgroup_size": "gl_WorkGroupSize",
            "num_workgroups": "gl_NumWorkGroups",
            "view_index": "gl_ViewIndex",
            "subgroup_local_invocation_id": "gl_SubgroupInvocationID",
            "subgroup_invocation_id": "gl_SubgroupInvocationID",
            "subgroup_size": "gl_SubgroupSize",
            "subgroup_id": "gl_SubgroupID",
            "num_subgroups": "gl_NumSubgroups",
            "subgroup_eq_mask": "gl_SubgroupEqMask",
            "subgroup_ge_mask": "gl_SubgroupGeMask",
            "subgroup_gt_mask": "gl_SubgroupGtMask",
            "subgroup_le_mask": "gl_SubgroupLeMask",
            "subgroup_lt_mask": "gl_SubgroupLtMask",
            # Tessellation shader semantics
            "invocation_id": "gl_InvocationID",
            "patch_vertices": "gl_PatchVerticesIn",
            "tess_coord": "gl_TessCoord",
            "tess_level_inner": "gl_TessLevelInner",
            "tess_level_outer": "gl_TessLevelOuter",
            # NV mesh/stereo/SM builtins surfaced by rust-gpu compiletests
            "SMIDNV": "gl_SMIDNV",
            "smidnv": "gl_SMIDNV",
            "clip_distance_per_view_nv": "gl_ClipDistancePerViewNV",
            "cull_distance_per_view_nv": "gl_CullDistancePerViewNV",
            "layer_per_view_nv": "gl_LayerPerViewNV",
            "mesh_view_count_nv": "gl_MeshViewCountNV",
            "mesh_view_indices_nv": "gl_MeshViewIndicesNV",
            "position_per_view_nv": "gl_PositionPerViewNV",
            "primitive_count_nv": "gl_PrimitiveCountNV",
            "primitive_indices_nv": "gl_PrimitiveIndicesNV",
            "secondary_position_nv": "gl_SecondaryPositionNV",
            "secondary_viewport_mask_nv": "gl_SecondaryViewportMaskNV",
            "sm_count_nv": "gl_SMCountNV",
            "task_count_nv": "gl_TaskCountNV",
            "viewport_mask_nv": "gl_ViewportMaskNV",
            "viewport_mask_per_view_nv": "gl_ViewportMaskPerViewNV",
            "warp_id_nv": "gl_WarpIDNV",
            "warps_per_sm_nv": "gl_WarpsPerSMNV",
            # Ray tracing semantics
            "hit_kind": "gl_HitKindEXT",
            "incoming_ray_flags": "gl_IncomingRayFlagsEXT",
            "incoming_ray_payload": "rayPayloadInEXT",
            "instance_custom_index": "gl_InstanceCustomIndexEXT",
            "launch_id": "gl_LaunchIDEXT",
            "launch_size": "gl_LaunchSizeEXT",
            "object_ray_direction": "gl_ObjectRayDirectionEXT",
            "object_ray_origin": "gl_ObjectRayOriginEXT",
            "object_to_world": "gl_ObjectToWorldEXT",
            "ray_geometry_index": "gl_GeometryIndexEXT",
            "ray_payload": "rayPayloadEXT",
            "ray_tmax": "gl_RayTmaxEXT",
            "ray_tmin": "gl_RayTminEXT",
            "world_ray_direction": "gl_WorldRayDirectionEXT",
            "world_ray_origin": "gl_WorldRayOriginEXT",
            "world_to_object": "gl_WorldToObjectEXT",
        }
        self.interpolation_semantic_map = {
            "flat": "flat",
            "noperspective": "noperspective",
            "no_perspective": "noperspective",
            "centroid": "centroid",
            "sample": "sample",
            "invariant": "invariant",
        }
        self.spirv_stage_map = {
            "vertex": "vertex",
            "fragment": "fragment",
            "compute": "compute",
            "geometry": "geometry",
            "tessellation_control": "tessellation_control",
            "tessellation_evaluation": "tessellation_evaluation",
            "task_ext": "task",
            "mesh_ext": "mesh",
            "ray_generation": "ray_generation",
            "intersection": "intersection",
            "closest_hit": "ray_closest_hit",
            "miss": "ray_miss",
            "any_hit": "ray_any_hit",
            "callable": "callable",
        }

        self.function_map = {
            # Rust math functions to CrossGL
            "abs": "abs",
            "min": "min",
            "max": "max",
            "clamp": "clamp",
            "floor": "floor",
            "ceil": "ceil",
            "round": "round",
            "trunc": "trunc",
            "round_even": "roundEven",
            "sqrt": "sqrt",
            "rsqrt": "inversesqrt",
            "fract": "fract",
            "pow": "pow",
            "fma": "fma",
            "ldexp": "ldexp",
            "modulo": "mod",
            "step": "step",
            "smoothstep": "smoothstep",
            "dfdx": "dFdx",
            "dfdy": "dFdy",
            "dfdx_fine": "dFdxFine",
            "dfdy_fine": "dFdyFine",
            "dfdx_coarse": "dFdxCoarse",
            "dfdy_coarse": "dFdyCoarse",
            "fwidth": "fwidth",
            "fwidth_fine": "fwidthFine",
            "fwidth_coarse": "fwidthCoarse",
            "exp": "exp",
            "exp2": "exp2",
            "log": "log",
            "ln": "log",
            "log2": "log2",
            "sign": "sign",
            "isnan": "isnan",
            "isinf": "isinf",
            "isfinite": "isfinite",
            "kill": "discard",
            "any": "any",
            "all": "all",
            "less_than": "lessThan",
            "less_than_equal": "lessThanEqual",
            "greater_than": "greaterThan",
            "greater_than_equal": "greaterThanEqual",
            "equal": "equal",
            "not_equal": "notEqual",
            "bit_count": "bitCount",
            "bitfield_reverse": "bitfieldReverse",
            "find_lsb": "findLSB",
            "find_msb": "findMSB",
            "bitfield_extract": "bitfieldExtract",
            "bitfield_insert": "bitfieldInsert",
            "float_bits_to_int": "floatBitsToInt",
            "float_bits_to_uint": "floatBitsToUint",
            "int_bits_to_float": "intBitsToFloat",
            "uint_bits_to_float": "uintBitsToFloat",
            "pack_unorm_2x16": "packUnorm2x16",
            "pack_snorm_2x16": "packSnorm2x16",
            "pack_unorm_4x8": "packUnorm4x8",
            "pack_snorm_4x8": "packSnorm4x8",
            "pack_half_2x16": "packHalf2x16",
            "pack_double_2x32": "packDouble2x32",
            "unpack_unorm_2x16": "unpackUnorm2x16",
            "unpack_snorm_2x16": "unpackSnorm2x16",
            "unpack_unorm_4x8": "unpackUnorm4x8",
            "unpack_snorm_4x8": "unpackSnorm4x8",
            "unpack_half_2x16": "unpackHalf2x16",
            "unpack_double_2x32": "unpackDouble2x32",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "atan2": "atan2",
            "lerp": "mix",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
            "degrees": "degrees",
            "radians": "radians",
            "workgroup_memory_barrier_with_group_sync": (
                "GroupMemoryBarrierWithGroupSync"
            ),
            # Vector operations
            "dot": "dot",
            "cross": "cross",
            "length": "length",
            "normalize": "normalize",
            "distance": "distance",
            "reflect": "reflect",
            "refract": "refract",
            "faceforward": "faceforward",
            "outer_product": "outerProduct",
            # Matrix operations
            "transpose": "transpose",
            "determinant": "determinant",
            "inverse": "inverse",
            "matrix_comp_mult": "matrixCompMult",
            # Texture sampling
            "sample": "texture",
            "sample_bias": "texture",
            "sample_sampler": "texture",
            "sample_bias_sampler": "texture",
            "sample_lod": "textureLod",
            "sample_lod_sampler": "textureLod",
            "sample_level": "textureLod",
            "sample_lod_offset": "textureLodOffset",
            "sample_lod_offset_sampler": "textureLodOffset",
            "sample_grad": "textureGrad",
            "sample_grad_sampler": "textureGrad",
            "sample_grad_offset": "textureGradOffset",
            "sample_grad_offset_sampler": "textureGradOffset",
            "sample_offset": "textureOffset",
            "sample_offset_bias": "textureOffset",
            "sample_offset_sampler": "textureOffset",
            "sample_offset_bias_sampler": "textureOffset",
            "sample_projected": "textureProj",
            "sample_projected_bias": "textureProj",
            "sample_projected_sampler": "textureProj",
            "sample_projected_bias_sampler": "textureProj",
            "sample_projected_lod": "textureProjLod",
            "sample_projected_lod_sampler": "textureProjLod",
            "sample_projected_grad": "textureProjGrad",
            "sample_projected_grad_sampler": "textureProjGrad",
            "sample_projected_offset": "textureProjOffset",
            "sample_projected_offset_bias": "textureProjOffset",
            "sample_projected_offset_sampler": "textureProjOffset",
            "sample_projected_offset_bias_sampler": "textureProjOffset",
            "sample_projected_lod_offset": "textureProjLodOffset",
            "sample_projected_lod_offset_sampler": "textureProjLodOffset",
            "sample_projected_grad_offset": "textureProjGradOffset",
            "sample_projected_grad_offset_sampler": "textureProjGradOffset",
            "texel_fetch": "texelFetch",
            "texel_fetch_offset": "texelFetchOffset",
            "texture_size": "textureSize",
            "texture_size_lod": "textureSize",
            "texture_query_levels": "textureQueryLevels",
            "texture_query_lod": "textureQueryLod",
            "texture_query_lod_sampler": "textureQueryLod",
            "texture_samples": "textureSamples",
            "texture_gather": "textureGather",
            "texture_gather_sampler": "textureGather",
            "texture_gather_component": "textureGather",
            "texture_gather_component_sampler": "textureGather",
            "texture_gather_offset": "textureGatherOffset",
            "texture_gather_offset_sampler": "textureGatherOffset",
            "texture_gather_offset_component": "textureGatherOffset",
            "texture_gather_offset_component_sampler": "textureGatherOffset",
            "texture_gather_offsets": "textureGatherOffsets",
            "texture_gather_offsets_sampler": "textureGatherOffsets",
            "texture_gather_offsets_component": "textureGatherOffsets",
            "texture_gather_offsets_component_sampler": "textureGatherOffsets",
            "texture_compare": "textureCompare",
            "texture_compare_sampler": "textureCompare",
            "texture_compare_offset": "textureCompareOffset",
            "texture_compare_offset_sampler": "textureCompareOffset",
            "texture_compare_lod": "textureCompareLod",
            "texture_compare_lod_sampler": "textureCompareLod",
            "texture_compare_lod_offset": "textureCompareLodOffset",
            "texture_compare_lod_offset_sampler": "textureCompareLodOffset",
            "texture_compare_grad": "textureCompareGrad",
            "texture_compare_grad_sampler": "textureCompareGrad",
            "texture_compare_grad_offset": "textureCompareGradOffset",
            "texture_compare_grad_offset_sampler": "textureCompareGradOffset",
            "texture_compare_projected": "textureCompareProj",
            "texture_compare_projected_sampler": "textureCompareProj",
            "texture_compare_projected_offset": "textureCompareProjOffset",
            "texture_compare_projected_offset_sampler": "textureCompareProjOffset",
            "texture_compare_projected_lod": "textureCompareProjLod",
            "texture_compare_projected_lod_sampler": "textureCompareProjLod",
            "texture_compare_projected_lod_offset": "textureCompareProjLodOffset",
            "texture_compare_projected_lod_offset_sampler": (
                "textureCompareProjLodOffset"
            ),
            "texture_compare_projected_grad": "textureCompareProjGrad",
            "texture_compare_projected_grad_sampler": "textureCompareProjGrad",
            "texture_compare_projected_grad_offset": "textureCompareProjGradOffset",
            "texture_compare_projected_grad_offset_sampler": (
                "textureCompareProjGradOffset"
            ),
            "texture_gather_compare": "textureGatherCompare",
            "texture_gather_compare_sampler": "textureGatherCompare",
            "texture_gather_compare_offset": "textureGatherCompareOffset",
            "texture_gather_compare_offset_sampler": "textureGatherCompareOffset",
            "image_load": "imageLoad",
            "image_load_sample": "imageLoad",
            "image_store": "imageStore",
            "image_store_sample": "imageStore",
            "image_size": "imageSize",
            "image_samples": "imageSamples",
            "image_atomic_add": "imageAtomicAdd",
            "image_atomic_add_sample": "imageAtomicAdd",
            "image_atomic_min": "imageAtomicMin",
            "image_atomic_min_sample": "imageAtomicMin",
            "image_atomic_max": "imageAtomicMax",
            "image_atomic_max_sample": "imageAtomicMax",
            "image_atomic_and": "imageAtomicAnd",
            "image_atomic_and_sample": "imageAtomicAnd",
            "image_atomic_or": "imageAtomicOr",
            "image_atomic_or_sample": "imageAtomicOr",
            "image_atomic_xor": "imageAtomicXor",
            "image_atomic_xor_sample": "imageAtomicXor",
            "image_atomic_exchange": "imageAtomicExchange",
            "image_atomic_exchange_sample": "imageAtomicExchange",
            "image_atomic_comp_swap": "imageAtomicCompSwap",
            "image_atomic_comp_swap_sample": "imageAtomicCompSwap",
            "buffer_load": "buffer_load",
            "buffer_store": "buffer_store",
            "buffer_dimensions": "buffer_dimensions",
        }

        self.attribute_map = {
            # Rust shader attributes to CrossGL qualifiers
            "vertex_shader": "vertex",
            "fragment_shader": "fragment",
            "compute_shader": "compute",
            "binding": "binding",
            "location": "location",
            "group": "group",
            "builtin": "builtin",
            "stage": "stage",
            "workgroup_size": "workgroup_size",
        }

        self.indentation = 0
        self.code = []
        self.type_aliases = {}
        self.imported_type_aliases = {}
        self.imported_module_aliases = {}
        self.struct_member_types = {}
        self.struct_generics = {}
        self.return_result_target = object()
        self.labeled_control_counter = 0
        self.switch_break_counter = 0
        self.for_loop_index_counter = 0
        self.for_loop_step_counter = 0
        self.for_loop_bound_counter = 0
        self.for_loop_iterable_counter = 0
        self.match_chain_counter = 0
        self.match_subject_counter = 0
        self.match_payload_counter = 0
        self.match_or_counter = 0
        self.match_tuple_counter = 0
        self.match_array_counter = 0
        self.matches_result_counter = 0
        self.closure_param_counter = 0
        self.try_subject_counter = 0
        self.try_value_counter = 0
        self.try_block_value_counter = 0
        self.transparent_block_value_counter = 0
        self.inline_expression_value_counter = 0
        self.current_function_return_type = None
        self.user_function_names = set()
        self.function_return_types = {}
        self.function_type_signatures = {}
        self.impl_method_signatures = {}
        self.value_type_scopes = []
        self.closure_helper_counter = 0
        self.closure_helper_names = set()
        self.local_function_item_counter = 0
        self.local_function_item_names = set()
        self.local_function_item_helper_names = {}
        self.current_closure_helpers = None
        self.closure_helper_generation_depth = 0
        self.name_alias_scopes = []
        self.local_binding_name_scopes = []
        self.local_callable_scopes = []

    def get_indent(self):
        return "    " * self.indentation

    def visit(self, node):
        """Dispatch a Rust AST node to a visitor method when available."""
        if isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, FunctionNode):
            return self.visit_FunctionNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)
        elif isinstance(node, ImplNode):
            return self.visit_ImplNode(node)
        elif isinstance(node, UseNode):
            return self.visit_UseNode(node)
        elif isinstance(node, ConstNode):
            return self.visit_ConstNode(node)
        elif isinstance(node, StaticNode):
            return self.visit_StaticNode(node)

        # For other node types, use existing methods
        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        self.type_aliases = {}
        self.imported_type_aliases = {}
        self.imported_module_aliases = {}
        self.struct_member_types = self.collect_struct_member_types(ast)
        self.struct_generics = self.collect_struct_generics(ast)
        self.labeled_control_counter = 0
        self.switch_break_counter = 0
        self.for_loop_index_counter = 0
        self.for_loop_step_counter = 0
        self.for_loop_bound_counter = 0
        self.match_chain_counter = 0
        self.match_subject_counter = 0
        self.match_payload_counter = 0
        self.match_or_counter = 0
        self.match_tuple_counter = 0
        self.match_array_counter = 0
        self.matches_result_counter = 0
        self.closure_param_counter = 0
        self.try_subject_counter = 0
        self.try_value_counter = 0
        self.try_block_value_counter = 0
        self.transparent_block_value_counter = 0
        self.inline_expression_value_counter = 0
        self.current_function_return_type = None
        self.user_function_names = self.collect_user_function_names(ast)
        self.function_return_types = self.collect_function_return_types(ast)
        self.function_type_signatures = self.collect_function_type_signatures(ast)
        self.impl_method_signatures = self.collect_impl_method_signatures(ast)
        self.value_type_scopes = []
        self.closure_helper_counter = 0
        self.closure_helper_names = set()
        self.local_function_item_counter = 0
        self.local_function_item_names = set()
        self.local_function_item_helper_names = {}
        self.current_closure_helpers = None
        self.closure_helper_generation_depth = 0
        self.name_alias_scopes = []
        self.local_binding_name_scopes = []
        self.local_callable_scopes = []
        code = "shader main {\n"

        for use_stmt in ast.use_statements:
            self.register_use_alias(use_stmt)
            code += f"    // use {use_stmt.path}\n"

        for alias in getattr(ast, "type_aliases", []):
            self.register_type_alias(alias)

        for alias in getattr(ast, "type_aliases", []):
            alias_code = self.generate_type_alias(alias)
            if alias_code:
                code += alias_code

        for global_var in ast.global_variables:
            if isinstance(global_var, ConstNode):
                declarator = self.format_typed_declarator(
                    global_var.vtype, global_var.name
                )
                value = self.generate_expression(global_var.value)
                code += f"    const {declarator} = {value};\n"
            elif isinstance(global_var, StaticNode):
                mutability = "mut " if global_var.is_mutable else ""
                declarator = self.format_typed_declarator(
                    global_var.vtype, global_var.name
                )
                value = self.generate_expression(global_var.value)
                code += f"    static {mutability}{declarator} = {value};\n"

        for struct in ast.structs:
            if isinstance(struct, StructNode):
                code += f"    struct {struct.name} {{\n"
                for member in struct.members:
                    semantic = self.get_semantic_from_attributes(member.attributes)
                    declarator = self.format_typed_declarator(member.vtype, member.name)
                    code += f"        {declarator}{semantic};\n"
                code += "    }\n\n"

        for func in ast.functions:
            shader_type = self.get_shader_type_from_attributes(func.attributes)
            if shader_type:
                code += f"    // {shader_type.title()} Shader\n"
                code += self.generate_shader_stage_function(func, shader_type)
            else:
                code += self.generate_function(func, indent=1)

        for impl_block in ast.impl_blocks:
            code += f"    // Implementation for {impl_block.struct_name}\n"
            for func in impl_block.functions:
                code += self.generate_function(
                    func, indent=1, struct_name=impl_block.struct_name
                )

        code += "}\n"
        return code

    def collect_user_function_names(self, ast):
        names = {getattr(func, "name", None) for func in getattr(ast, "functions", [])}
        for impl_block in getattr(ast, "impl_blocks", []):
            for func in getattr(impl_block, "functions", []):
                names.add(getattr(func, "name", None))
        names.discard(None)
        return names

    def push_name_alias_scope(self, aliases=None):
        self.name_alias_scopes.append(dict(aliases or {}))

    def current_name_alias_scope(self):
        if not self.name_alias_scopes:
            self.push_name_alias_scope()
        return self.name_alias_scopes[-1]

    def pop_name_alias_scope(self):
        if self.name_alias_scopes:
            self.name_alias_scopes.pop()

    def push_local_callable_scope(self, entries=None):
        scope = {}
        if entries:
            for name, alias in entries:
                if name:
                    scope[name] = alias or name
        self.local_callable_scopes.append(scope)

    def pop_local_callable_scope(self):
        if self.local_callable_scopes:
            self.local_callable_scopes.pop()

    def add_local_callable(self, name, alias=None):
        if not name:
            return
        if not self.local_callable_scopes:
            self.push_local_callable_scope()
        self.local_callable_scopes[-1][name] = alias or name

    def lookup_local_callable_name(self, name):
        if not isinstance(name, str):
            return None
        for scope in reversed(self.local_callable_scopes):
            callable_name = scope.get(name)
            if callable_name is not None:
                return callable_name
        return None

    def get_local_function_item_helper_name(self, func):
        key = id(func)
        helper_name = self.local_function_item_helper_names.get(key)
        if helper_name is None:
            helper_name = self.next_local_function_item_name(func.name)
            self.local_function_item_helper_names[key] = helper_name
        return helper_name

    def predeclare_local_function_items(self, body):
        for stmt in body or []:
            if isinstance(stmt, FunctionNode):
                self.add_local_callable(
                    stmt.name,
                    self.get_local_function_item_helper_name(stmt),
                )

    def generate_scoped_function_body(
        self,
        body,
        indent=1,
        loop_contexts=None,
        allow_implicit_final_return=False,
    ):
        self.push_local_callable_scope()
        try:
            return self.generate_function_body(
                body,
                indent,
                loop_contexts,
                allow_implicit_final_return=allow_implicit_final_return,
            )
        finally:
            self.pop_local_callable_scope()

    def resolve_name_alias(self, name):
        if not isinstance(name, str):
            return name

        for scope in reversed(self.name_alias_scopes):
            alias = scope.get(name)
            if alias is not None:
                return alias
        return name

    def crossgl_identifier(self, name, forbidden=None):
        if not isinstance(name, str) or not name:
            return name

        forbidden = set(forbidden or ())
        if self.is_valid_crossgl_identifier(name) and name not in forbidden:
            return name

        base = re.sub(r"\W+", "_", name).strip("_") or "value"
        if base[0].isdigit():
            base = f"_{base}"

        candidate = f"{base}_"
        suffix = 1
        while not self.is_valid_crossgl_identifier(candidate) or candidate in forbidden:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def is_valid_crossgl_identifier(self, name):
        return (
            self.is_plain_identifier(name) and name not in CROSSGL_RESERVED_IDENTIFIERS
        )

    def prepare_function_parameters(
        self, params, struct_name=None, extra_forbidden=None
    ):
        original_names = {
            param.name for param in params if getattr(param, "name", None)
        }
        extra_forbidden = set(extra_forbidden or ())
        used_names = set()
        aliases = {}
        declarations = []
        param_types = []

        for param in params:
            forbidden = (original_names - {param.name}) | used_names | extra_forbidden
            name = self.crossgl_identifier(param.name, forbidden)
            aliases[param.name] = name
            used_names.add(name)
            semantic = self.get_semantic_from_attributes(
                getattr(param, "attributes", [])
            )
            param_type = self.normalize_parameter_type(param.vtype, struct_name)
            declarations.append(
                f"{self.format_typed_declarator(param_type, name)}{semantic}"
            )
            param_types.append((name, param_type))

        return ", ".join(declarations), param_types, aliases

    def local_aliasing_enabled(self):
        return bool(self.local_binding_name_scopes)

    def collect_simple_local_binding_names(self, statements):
        names = set()

        def visit_statement(stmt):
            if isinstance(stmt, LetNode):
                if isinstance(stmt.name, str):
                    names.add(stmt.name)
                for child in (stmt.value, stmt.else_body):
                    visit(child)
                return
            if isinstance(stmt, IfNode):
                visit(stmt.if_body)
                for _, body in getattr(stmt, "if_chain", []) or []:
                    visit(body)
                for _, body in getattr(stmt, "else_if_chain", []) or []:
                    visit(body)
                visit(stmt.else_body)
                return
            if isinstance(stmt, (ForNode, WhileNode, LoopNode)):
                pattern = getattr(stmt, "pattern", None)
                if isinstance(pattern, str):
                    names.add(pattern)
                visit(getattr(stmt, "body", None))
                return
            if isinstance(stmt, MatchNode):
                for arm in getattr(stmt, "arms", []) or []:
                    visit(getattr(arm, "body", None))

        def visit(node):
            if node is None:
                return
            if isinstance(node, list):
                for item in node:
                    visit_statement(item)
                return
            if isinstance(node, BlockNode):
                visit(node.statements)
                return
            visit_statement(node)

        visit(statements)
        return names

    def declare_local_alias(self, name):
        if not self.local_aliasing_enabled() or not isinstance(name, str):
            return name

        forbidden = set(self.local_binding_name_scopes[-1])
        forbidden.discard(name)
        for scope in self.name_alias_scopes:
            forbidden.update(scope.values())

        alias = self.crossgl_identifier(name, forbidden)
        self.current_name_alias_scope()[name] = alias
        return alias

    def collect_function_return_types(self, ast):
        return_types = {}
        for func in getattr(ast, "functions", []):
            name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if name and return_type:
                return_types[name] = return_type
        return return_types

    def collect_function_type_signatures(self, ast):
        signatures = {}
        for func in getattr(ast, "functions", []):
            name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if not name or not return_type:
                continue

            params = []
            for param in getattr(func, "params", []):
                param_type = getattr(param, "vtype", None)
                if param_type:
                    params.append((getattr(param, "name", None), param_type))

            signatures[name] = {
                "return_type": return_type,
                "params": params,
                "generic_names": [
                    self.generic_parameter_name(generic)
                    for generic in getattr(func, "generics", []) or []
                ],
            }
        return signatures

    def collect_impl_method_signatures(self, ast):
        signatures_by_type = {}
        for impl_block in getattr(ast, "impl_blocks", []):
            struct_name = getattr(impl_block, "struct_name", None)
            if not struct_name:
                continue

            signature = signatures_by_type.setdefault(
                struct_name,
                {
                    "struct_name": struct_name,
                    "call_prefix": self.impl_function_prefix(struct_name),
                    "generic_names": [
                        self.generic_parameter_name(generic)
                        for generic in getattr(impl_block, "generics", []) or []
                    ],
                    "methods": {},
                },
            )

            for func in getattr(impl_block, "functions", []):
                method_name = getattr(func, "name", None)
                return_type = getattr(func, "return_type", None)
                params = []
                for param in getattr(func, "params", []):
                    param_type = getattr(param, "vtype", None)
                    if param_type:
                        params.append((getattr(param, "name", None), param_type))

                if method_name and return_type:
                    signature["methods"][method_name] = {
                        "return_type": return_type,
                        "params": params,
                        "generic_names": [
                            self.generic_parameter_name(generic)
                            for generic in getattr(func, "generics", []) or []
                        ],
                    }
        return signatures_by_type

    def collect_struct_member_types(self, ast):
        member_types = {}
        for struct in getattr(ast, "structs", []):
            if not isinstance(struct, StructNode):
                continue

            members = {}
            for member in getattr(struct, "members", []):
                name = getattr(member, "name", None)
                type_name = getattr(member, "vtype", None)
                if name and type_name:
                    members[name] = type_name
            member_types[struct.name] = members

        return member_types

    def collect_struct_generics(self, ast):
        generics = {}
        for struct in getattr(ast, "structs", []):
            if not isinstance(struct, StructNode):
                continue
            generics[struct.name] = [
                self.generic_parameter_name(generic)
                for generic in getattr(struct, "generics", []) or []
            ]
        return generics

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def push_value_type_scope(self, entries=None):
        scope = {}
        if entries:
            for name, type_name in entries:
                if name and type_name:
                    scope[name] = type_name
        self.value_type_scopes.append(scope)

    def pop_value_type_scope(self):
        if self.value_type_scopes:
            self.value_type_scopes.pop()

    def add_value_type(self, name, type_name, struct_name=None):
        normalized_type = self.normalize_receiver_type(type_name, struct_name)
        if name and normalized_type and self.value_type_scopes:
            self.value_type_scopes[-1][name] = normalized_type

    def lookup_value_type(self, name):
        direct_type = self.lookup_direct_value_type(name)
        if direct_type:
            return direct_type

        if isinstance(name, str):
            return self.lookup_expression_value_type(name)

        return None

    def lookup_direct_value_type(self, name):
        for scope in reversed(self.value_type_scopes):
            type_name = scope.get(name)
            if type_name:
                return type_name
        return None

    def normalize_receiver_type(self, type_name, struct_name=None):
        if not type_name:
            return None

        type_name = self.strip_reference_type(type_name)
        if type_name == "Self" and struct_name:
            return struct_name
        return type_name

    def normalize_parameter_type(self, type_name, struct_name=None):
        typed_buffer_type = self.map_typed_buffer_parameter_type(type_name)
        if typed_buffer_type is not None:
            return typed_buffer_type
        return self.normalize_receiver_type(type_name, struct_name)

    def infer_value_type(self, expression):
        if isinstance(expression, StructInitializationNode):
            return expression.struct_name
        if isinstance(expression, BinaryOpNode):
            return self.infer_binary_expression_value_type(expression)
        if isinstance(expression, FunctionCallNode):
            return self.infer_function_call_return_type(expression)
        if isinstance(expression, TernaryOpNode):
            return self.common_inferred_value_type(
                [
                    self.infer_value_type(expression.true_expr),
                    self.infer_value_type(expression.false_expr),
                ]
            )
        if isinstance(expression, IfNode):
            return self.common_inferred_value_type(
                [
                    self.infer_branch_body_value_type(expression.if_body),
                    self.infer_branch_body_value_type(expression.else_body),
                ]
            )
        if isinstance(expression, MatchNode):
            return self.common_inferred_value_type(
                [
                    self.infer_branch_body_value_type(arm.body)
                    for arm in getattr(expression, "arms", [])
                ]
            )
        if self.is_block_expression_node(expression):
            return self.infer_block_expression_value_type(expression)
        if isinstance(expression, str):
            path_type = self.infer_path_value_type(expression)
            if path_type is not None:
                return path_type
        if isinstance(
            expression, (str, VariableNode, MemberAccessNode, ArrayAccessNode)
        ):
            return self.lookup_value_type(self.generate_expression(expression))
        return None

    def infer_binary_expression_value_type(self, expression):
        if expression.op in {"==", "!=", "<", ">", "<=", ">=", "&&", "||"}:
            return "bool"

        left_type = self.infer_value_type(expression.left)
        right_type = self.infer_value_type(expression.right)
        if left_type and right_type:
            return self.common_inferred_value_type([left_type, right_type])
        return left_type or right_type

    def infer_path_value_type(self, expression):
        if not isinstance(expression, str) or "::" not in expression:
            return None

        path = self.resolve_imported_module_path(expression)
        type_name = path.rsplit("::", 1)[0]
        mapped_type = self.map_type(type_name)
        if mapped_type == type_name:
            return None
        return mapped_type

    def infer_branch_body_value_type(self, body):
        if body is None:
            return None
        if isinstance(body, ReturnNode):
            return self.infer_value_type(body.value)
        if isinstance(body, list):
            if not body:
                return None
            return self.infer_branch_body_value_type(body[-1])
        return self.infer_value_type(body)

    def infer_block_expression_value_type(self, expression):
        block_node = self.get_block_expression_node(expression)
        if block_node is None:
            return None

        block_expression = self.get_block_expression(block_node)
        if block_expression is not None:
            return self.infer_value_type(block_expression)

        statements = getattr(block_node, "statements", []) or []
        if not statements:
            return None
        return self.infer_branch_body_value_type(statements[-1])

    def common_inferred_value_type(self, type_names):
        if not type_names or any(not type_name for type_name in type_names):
            return None

        first_type = self.normalize_receiver_type(type_names[0])
        if all(
            self.normalize_receiver_type(type_name) == first_type
            for type_name in type_names
        ):
            return first_type

        first_resource_type = self.map_resource_receiver_type(first_type)
        if not isinstance(
            first_resource_type, str
        ) or not first_resource_type.startswith(self.RESOURCE_TYPE_PREFIXES):
            return None

        for type_name in type_names[1:]:
            resource_type = self.map_resource_receiver_type(type_name)
            if resource_type != first_resource_type:
                return None
        return first_type

    def infer_function_call_return_type(self, expression):
        if isinstance(expression.name, MemberAccessNode):
            return self.infer_impl_method_call_return_type(expression)

        raw_function_name = self.function_call_name(expression.name)
        if raw_function_name is None:
            return None
        builtin_return_type = self.infer_builtin_function_return_type(raw_function_name)
        if builtin_return_type is not None:
            return builtin_return_type
        associated_return_type = self.infer_impl_associated_function_call_return_type(
            raw_function_name,
            expression.args,
        )
        if associated_return_type is not None:
            return associated_return_type

        function_name, explicit_type_args = self.split_function_type_arguments(
            raw_function_name
        )
        signature = self.lookup_user_function_signature(function_name)
        if signature is not None:
            return self.infer_user_function_call_return_type(
                signature,
                expression.args,
                explicit_type_args,
            )
        return self.lookup_user_function_return_type(function_name)

    def infer_impl_associated_function_call_return_type(self, function_name, args):
        match = self.lookup_impl_associated_function_match(function_name)
        if match is None:
            return None

        return self.infer_impl_associated_function_return_type(match, args)

    def infer_impl_method_call_return_type(self, expression):
        member_access = expression.name
        method_name = getattr(member_access, "member", None)
        builtin_return_type = self.infer_builtin_method_return_type(
            method_name,
            self.infer_value_type(member_access.object),
            expression.args,
        )
        if builtin_return_type is not None:
            return builtin_return_type

        obj = self.generate_expression(member_access.object)
        match = self.lookup_impl_method_receiver_match(obj, method_name)
        if match is None:
            return None

        return self.infer_impl_method_return_type(match, expression.args)

    def infer_builtin_function_return_type(self, function_name):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        if "::" in function_name:
            type_name, method_name = function_name.rsplit("::", 1)
            if method_name in {"new", "splat", "from"}:
                mapped_type = self.map_type(type_name)
                if mapped_type != type_name:
                    return mapped_type

        return self.VECTOR_CONSTRUCTOR_RETURN_TYPES.get(
            function_name.rsplit("::", 1)[-1]
        )

    def infer_builtin_method_return_type(self, method_name, receiver_type, args):
        if method_name == "extend" and len(args) == 1:
            return self.extended_vector_constructor(receiver_type)
        return None

    def infer_impl_method_return_type(self, match, arg_values):
        method = match["method"]
        return_type = match["method"].get("return_type")
        return_type = self.normalize_receiver_type(return_type, match["receiver_type"])
        substitutions = dict(match.get("substitutions") or {})
        generic_names = list(match.get("generic_names") or [])
        for generic_name in method.get("generic_names") or []:
            if generic_name not in generic_names:
                generic_names.append(generic_name)

        conflicts = set()
        for (_, param_type), arg in zip(
            self.method_value_params(method),
            arg_values,
        ):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            param_type = self.normalize_receiver_type(
                param_type,
                match["receiver_type"],
            )
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return_type = self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def method_value_params(self, method):
        return [
            (name, param_type)
            for name, param_type in method.get("params", [])
            if not self.is_self_parameter(name, param_type)
        ]

    def method_has_self_parameter(self, method):
        return any(
            self.is_self_parameter(name, param_type)
            for name, param_type in method.get("params", [])
        )

    def is_self_parameter(self, name, param_type):
        return name == "self"

    def infer_impl_associated_function_return_type(self, match, arg_values):
        method = match["method"]
        return_type = method.get("return_type")
        return_type = self.normalize_receiver_type(return_type, match["receiver_type"])
        substitutions = dict(match.get("substitutions") or {})
        generic_names = list(match.get("generic_names") or [])
        method_generic_names = method.get("generic_names") or []
        for generic_name in method.get("generic_names") or []:
            if generic_name not in generic_names:
                generic_names.append(generic_name)

        conflicts = set()
        for generic_name, type_arg in zip(
            method_generic_names,
            match.get("method_type_args") or [],
        ):
            self.bind_generic_type_substitution(
                generic_name,
                type_arg,
                substitutions,
                conflicts,
            )

        for (_, param_type), arg in zip(
            self.method_value_params(method),
            arg_values,
        ):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return_type = self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def function_call_name(self, name):
        if isinstance(name, str):
            return name
        if isinstance(name, VariableNode):
            return name.name
        return None

    def split_function_type_arguments(self, function_name):
        if not isinstance(function_name, str):
            return function_name, []

        generic = self.parse_generic_type(function_name)
        if generic is None:
            return function_name, []
        return generic

    def lookup_user_function_signature(self, function_name):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        signature = self.function_type_signatures.get(function_name)
        if signature is not None:
            return signature

        current_module_function = self.current_module_user_function_name(function_name)
        if current_module_function is None:
            return None
        return self.function_type_signatures.get(current_module_function)

    def infer_user_function_call_return_type(
        self,
        signature,
        args,
        explicit_type_args=None,
    ):
        return_type = signature.get("return_type")
        if not return_type:
            return None

        generic_names = signature.get("generic_names") or []
        if not generic_names:
            return return_type

        substitutions = {}
        conflicts = set()
        for generic_name, type_arg in zip(generic_names, explicit_type_args or []):
            self.bind_generic_type_substitution(
                generic_name,
                type_arg,
                substitutions,
                conflicts,
            )

        for (_, param_type), arg in zip(signature.get("params", []), args):
            arg_type = self.infer_call_argument_type(arg)
            if arg_type is None:
                continue
            self.match_generic_type_parameters(
                param_type,
                arg_type,
                generic_names,
                substitutions,
                conflicts,
            )

        for generic_name in conflicts:
            substitutions.pop(generic_name, None)
        if substitutions:
            return self.apply_type_substitutions(return_type, substitutions)
        return return_type

    def infer_call_argument_type(self, arg):
        if isinstance(arg, str):
            return self.lookup_value_type(arg)
        inferred_type = self.infer_value_type(arg)
        if inferred_type is not None:
            return inferred_type
        return self.lookup_value_type(self.generate_expression(arg))

    def match_generic_type_parameters(
        self,
        pattern_type,
        actual_type,
        generic_names,
        substitutions,
        conflicts,
    ):
        pattern_type = self.normalize_receiver_type(pattern_type)
        actual_type = self.normalize_receiver_type(actual_type)
        if not pattern_type or not actual_type:
            return

        if pattern_type in generic_names:
            self.bind_generic_type_substitution(
                pattern_type,
                actual_type,
                substitutions,
                conflicts,
            )
            return

        pattern_array = self.split_array_type(pattern_type)
        actual_array = self.split_array_type(actual_type)
        if pattern_array and actual_array:
            self.match_generic_type_parameters(
                pattern_array[0],
                actual_array[0],
                generic_names,
                substitutions,
                conflicts,
            )
            return

        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            pattern_generic = self.parse_generic_type(pattern_candidate)
            if pattern_generic is None:
                continue
            pattern_base, pattern_args = pattern_generic

            for actual_candidate in self.generic_type_match_candidates(actual_type):
                actual_generic = self.parse_generic_type(actual_candidate)
                if actual_generic is None:
                    continue
                actual_base, actual_args = actual_generic
                if not self.generic_type_bases_match(pattern_base, actual_base):
                    continue
                if len(pattern_args) != len(actual_args):
                    continue

                for pattern_arg, actual_arg in zip(pattern_args, actual_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        actual_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                return

    def bind_generic_type_substitution(
        self,
        generic_name,
        replacement,
        substitutions,
        conflicts,
    ):
        replacement = self.normalize_receiver_type(replacement)
        if not generic_name or not replacement or generic_name in conflicts:
            return

        existing = substitutions.get(generic_name)
        if existing is None or self.inferred_types_equivalent(existing, replacement):
            substitutions[generic_name] = replacement
            return

        conflicts.add(generic_name)
        substitutions.pop(generic_name, None)

    def generic_type_match_candidates(self, type_name):
        candidates = []

        def add_candidate(candidate):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add_candidate(type_name)
        resolved = self.resolve_imported_module_path(type_name)
        add_candidate(resolved)
        alias_target = self.resolve_type_alias_target(type_name)
        add_candidate(alias_target)
        if alias_target is not None:
            add_candidate(self.resolve_imported_module_path(alias_target))
        return candidates

    def generic_type_bases_match(self, left, right):
        return any(
            left_name == right_name
            for left_name in self.type_lookup_names(left)
            for right_name in self.type_lookup_names(right)
        )

    def inferred_types_equivalent(self, left, right):
        if left == right:
            return True
        left_resource = self.map_resource_receiver_type(left)
        right_resource = self.map_resource_receiver_type(right)
        if left_resource == right_resource:
            return True
        return self.map_type(left) == self.map_type(right)

    def lookup_user_function_return_type(self, function_name):
        if not isinstance(function_name, str):
            return None

        function_name = self.resolve_imported_module_path(function_name)
        return_type = self.function_return_types.get(function_name)
        if return_type is not None:
            return return_type

        current_module_function = self.current_module_user_function_name(function_name)
        if current_module_function is None:
            return None
        return self.function_return_types.get(current_module_function)

    def lookup_expression_value_type(self, expression):
        parts = self.split_member_expression(expression)
        if not parts:
            return None

        current_type = self.lookup_indexed_expression_part_type(parts[0])
        for member_part in parts[1:]:
            if current_type is None:
                return None

            member_name, index_count = self.split_indexed_expression_part(member_part)
            if not member_name:
                return None

            current_type = self.lookup_struct_member_type(current_type, member_name)
            for _ in range(index_count):
                current_type = self.array_element_type(current_type)
                if current_type is None:
                    return None

        return current_type

    def lookup_indexed_expression_part_type(self, expression):
        root, index_count = self.split_indexed_expression_part(expression)
        if not root:
            return None

        if len(self.split_member_expression(root)) > 1:
            current_type = self.lookup_expression_value_type(root)
        else:
            current_type = self.lookup_direct_value_type(root)
            if current_type is None:
                current_type = self.infer_generated_call_return_type(root)

        for _ in range(index_count):
            current_type = self.array_element_type(current_type)
            if current_type is None:
                return None

        return current_type

    def infer_generated_call_return_type(self, expression):
        parsed = self.parse_generated_call_expression(expression)
        if parsed is None:
            return None

        function_name, args = parsed
        return_type = self.infer_generated_impl_method_return_type(
            function_name,
            args,
        )
        if return_type is not None:
            return return_type

        return_type = self.infer_generated_impl_associated_function_return_type(
            function_name,
            args,
        )
        if return_type is not None:
            return return_type

        function_name, explicit_type_args = self.split_function_type_arguments(
            function_name
        )
        signature = self.lookup_user_function_signature(function_name)
        if signature is not None:
            return self.infer_user_function_call_return_type(
                signature,
                args,
                explicit_type_args,
            )
        return self.lookup_user_function_return_type(function_name)

    def parse_generated_call_expression(self, expression):
        if not isinstance(expression, str) or not expression.endswith(")"):
            return None

        open_index = self.find_matching_call_open(expression)
        if open_index is None or open_index == 0:
            return None

        function_name = expression[:open_index].strip()
        if not function_name:
            return None

        args_text = expression[open_index + 1 : -1]
        args = self.split_generic_arguments(args_text)
        return function_name, args

    def find_matching_call_open(self, expression):
        depth = 0
        for index in range(len(expression) - 1, -1, -1):
            char = expression[index]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def infer_generated_impl_method_return_type(self, function_name, args):
        if not args:
            return None

        for signature in self.impl_method_signatures.values():
            for method_name, method in signature["methods"].items():
                if function_name != f"{signature['call_prefix']}_{method_name}":
                    continue

                match = self.lookup_impl_method_receiver_match(args[0], method_name)
                if match is None:
                    return None
                return self.infer_impl_method_return_type(match, args[1:])

        return None

    def infer_generated_impl_associated_function_return_type(self, function_name, args):
        for signature in self.impl_method_signatures.values():
            for method_name, method in signature["methods"].items():
                if function_name != f"{signature['call_prefix']}_{method_name}":
                    continue
                if self.method_has_self_parameter(method):
                    continue

                match = {
                    "receiver_type": signature["struct_name"],
                    "call_prefix": signature["call_prefix"],
                    "method_name": method_name,
                    "generic_names": signature["generic_names"],
                    "method": method,
                    "substitutions": {},
                }
                return self.infer_impl_associated_function_return_type(match, args)

        return None

    def split_member_expression(self, expression):
        if not isinstance(expression, str):
            return []

        parts = []
        current = []
        depth = 0
        for char in expression:
            if char == "." and depth == 0:
                part = "".join(current).strip()
                if not part:
                    return []
                parts.append(part)
                current = []
                continue

            if char in "[({":
                depth += 1
            elif char in "])}":
                depth = max(0, depth - 1)
            current.append(char)

        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def split_indexed_expression_part(self, expression):
        if not isinstance(expression, str):
            return "", 0

        root = expression.strip()
        index_count = 0
        while root.endswith("]"):
            open_index = self.find_matching_index_open(root)
            if open_index is None or open_index == 0:
                break
            root = root[:open_index].strip()
            index_count += 1

        return root, index_count

    def find_matching_index_open(self, expression):
        depth = 0
        for index in range(len(expression) - 1, -1, -1):
            char = expression[index]
            if char == "]":
                depth += 1
            elif char == "[":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def array_element_type(self, type_name):
        if not type_name:
            return None

        normalized = self.normalize_receiver_type(type_name)
        array_parts = self.split_array_type(normalized)
        if array_parts:
            return array_parts[0]

        mapped = self.map_type(normalized)
        array_parts = self.split_array_type(mapped)
        if array_parts:
            return array_parts[0]

        return None

    def lookup_struct_member_type(self, struct_type, member_name):
        if not struct_type or not member_name:
            return None

        struct_type = self.normalize_receiver_type(struct_type)
        array_parts = self.split_array_type(struct_type)
        if array_parts:
            struct_type = array_parts[0]

        resolved = self.resolve_struct_type(struct_type)
        if resolved is None:
            return None

        struct_name, substitutions = resolved
        members = self.struct_member_types.get(struct_name, {})
        member_type = members.get(member_name)
        if member_type is None and self.is_tuple_field_member(member_name):
            member_type = members.get(self.tuple_field_member_name(member_name))
        if member_type is None:
            return None

        if substitutions:
            member_type = self.apply_type_substitutions(member_type, substitutions)
        return self.normalize_receiver_type(member_type)

    def resolve_member_access_name(self, obj, member_name):
        if not self.is_tuple_field_member(member_name):
            return member_name

        receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return member_name

        field_name = self.tuple_field_member_name(member_name)
        if self.lookup_struct_member_type(receiver_type, field_name) is None:
            return member_name
        return field_name

    def is_tuple_field_member(self, member_name):
        return isinstance(member_name, str) and member_name.isdigit()

    def tuple_field_member_name(self, member_name):
        return f"field{member_name}"

    def resolve_struct_type(self, type_name, seen=None):
        seen = seen or set()
        generic = self.parse_generic_type(type_name)
        if generic is not None:
            base_name, args = generic
            for candidate in self.type_lookup_names(base_name):
                if candidate in self.struct_member_types:
                    generics = self.struct_generics.get(candidate, [])
                    if len(generics) != len(args):
                        continue
                    substitutions = dict(zip(generics, args))
                    return candidate, substitutions

                if candidate in seen:
                    continue

                alias = self.type_aliases.get(candidate)
                if alias is None:
                    continue

                alias_type = self.apply_alias_type_arguments(alias, args)
                if alias_type is None:
                    continue

                seen.add(candidate)
                resolved = self.resolve_struct_type(alias_type, seen)
                if resolved is not None:
                    return resolved
            return None

        for candidate in self.type_lookup_names(type_name):
            if candidate in self.struct_member_types:
                return candidate, {}

            if candidate in seen:
                continue

            alias = self.type_aliases.get(candidate)
            if alias is not None and not getattr(alias, "generics", None):
                seen.add(candidate)
                resolved = self.resolve_struct_type(alias.alias_type, seen)
                if resolved is not None:
                    return resolved

        return None

    def apply_alias_type_arguments(self, alias, args):
        generics = getattr(alias, "generics", None) or []
        if not generics:
            return alias.alias_type if not args else None
        if len(generics) != len(args):
            return None

        substitutions = {
            self.generic_parameter_name(generic): arg
            for generic, arg in zip(generics, args)
        }
        return self.apply_type_substitutions(alias.alias_type, substitutions)

    def apply_type_substitutions(self, type_name, substitutions):
        result = type_name
        for name, replacement in substitutions.items():
            result = self.replace_type_identifier(result, name, replacement)
        return result

    def is_block_expression_node(self, node):
        return isinstance(node, BLOCK_EXPRESSION_NODE_TYPES)

    def is_transparent_block_expression_node(self, node):
        return isinstance(node, TRANSPARENT_BLOCK_NODE_TYPES)

    def get_block_expression_node(self, node):
        if isinstance(node, BlockNode):
            return node
        if self.is_transparent_block_expression_node(node):
            return node.block
        return None

    def register_type_alias(self, alias):
        if getattr(alias, "name", None):
            self.type_aliases[alias.name] = alias

    def register_use_alias(self, use_stmt):
        path = getattr(use_stmt, "path", None)
        items = getattr(use_stmt, "items", None)
        if items:
            self.register_grouped_use_aliases(path, items)
            return

        alias = getattr(use_stmt, "alias", None)
        if not alias or not path or "*" in path or "{" in path:
            return

        self.register_import_alias(alias, path)

    def register_grouped_use_aliases(self, path, items):
        if not path:
            return

        base_path = path.split("::{", 1)[0]
        if not base_path:
            return

        for item in items:
            alias = item.get("alias")
            item_path = item.get("path")
            if not alias or not item_path or "*" in item_path or "{" in item_path:
                continue

            self.register_import_alias(alias, f"{base_path}::{item_path}")

    def register_import_alias(self, alias, path):
        if self.is_imported_type_path(path):
            self.imported_type_aliases[alias] = path
        else:
            self.imported_module_aliases[alias] = path

    def is_imported_type_path(self, path):
        if self.map_builtin_type(path) is not None:
            return True

        type_name = path.rsplit("::", 1)[-1]
        if not type_name:
            return False

        return type_name[0].isupper() or type_name in self.type_map

    def generate_type_alias(self, alias):
        if getattr(alias, "generics", None) or not getattr(alias, "alias_type", None):
            return ""

        declarator = self.format_typed_declarator(alias.alias_type, alias.name)
        return f"    typedef {declarator};\n"

    def generate_function(self, func, indent=1, struct_name=None):
        """Render one Rust function node as a CrossGL function."""
        code = ""
        indent_str = "    " * indent

        local_binding_names = self.collect_simple_local_binding_names(func.body)
        params_str, param_types, name_aliases = self.prepare_function_parameters(
            func.params,
            struct_name=struct_name,
            extra_forbidden=local_binding_names,
        )
        display_return_type = self.normalize_receiver_type(
            func.return_type, struct_name
        )
        return_type = self.map_type(display_return_type)

        if struct_name:
            func_name = f"{self.impl_function_prefix(struct_name)}_{func.name}"
        else:
            func_name = func.name

        previous_return_type = self.current_function_return_type
        previous_helpers = self.current_closure_helpers
        self.current_closure_helpers = []
        self.current_function_return_type = func.return_type
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        self.push_name_alias_scope(name_aliases)
        self.local_binding_name_scopes.append(local_binding_names)
        try:
            body_code = self.generate_function_body(
                func.body,
                indent=indent + 1,
                allow_implicit_final_return=True,
            )
            helper_code = "".join(self.current_closure_helpers)
            code += helper_code
            code += f"{indent_str}{return_type} {func_name}({params_str}) {{\n"
            code += body_code
            code += f"{indent_str}}}\n\n"
        finally:
            self.current_function_return_type = previous_return_type
            self.current_closure_helpers = previous_helpers
            self.local_binding_name_scopes.pop()
            self.pop_name_alias_scope()
            self.pop_local_callable_scope()
            self.pop_value_type_scope()

        return code

    def impl_function_prefix(self, struct_name):
        generic = self.parse_generic_type(struct_name)
        type_name = generic[0] if generic is not None else struct_name
        return self.type_lookup_names(type_name)[-1]

    def generate_shader_stage_function(self, func, shader_type):
        """Render a Rust shader attribute as a named CrossGL stage entry."""
        local_binding_names = self.collect_simple_local_binding_names(func.body)
        params_str, param_types, name_aliases = self.prepare_function_parameters(
            func.params,
            extra_forbidden=local_binding_names,
        )
        return_type = self.map_type(func.return_type)
        entry_point_name = self.get_entry_point_name_from_attributes(func.attributes)
        stage_name = self.crossgl_identifier(entry_point_name or func.name)
        numthreads = self.get_numthreads_from_attributes(func.attributes)
        numthreads_suffix = (
            f" @numthreads({', '.join(numthreads)})" if numthreads else ""
        )
        stage_header = (
            shader_type if stage_name == "main" else f"{shader_type} {stage_name}"
        )

        previous_return_type = self.current_function_return_type
        previous_helpers = self.current_closure_helpers
        self.current_closure_helpers = []
        self.current_function_return_type = func.return_type
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        self.push_name_alias_scope(name_aliases)
        self.local_binding_name_scopes.append(local_binding_names)
        try:
            body_code = self.generate_function_body(
                func.body,
                indent=3,
                allow_implicit_final_return=True,
            )
            helper_code = "".join(self.current_closure_helpers)
        finally:
            self.current_function_return_type = previous_return_type
            self.current_closure_helpers = previous_helpers
            self.local_binding_name_scopes.pop()
            self.pop_name_alias_scope()
            self.pop_local_callable_scope()
            self.pop_value_type_scope()

        code = helper_code
        code += f"    {stage_header} {{\n"
        code += f"        {return_type} main({params_str}){numthreads_suffix} {{\n"
        code += body_code
        code += "        }\n"
        code += "    }\n\n"
        return code

    def function_returns_value(self):
        return_type = self.current_function_return_type
        if not return_type:
            return False

        return self.map_type(return_type) != "void"

    def is_implicit_final_return_statement(self, stmt, index, body):
        if index != len(body) - 1 or not self.function_returns_value():
            return False

        return not isinstance(
            stmt,
            (
                ConstNode,
                StaticNode,
                FunctionNode,
                LetNode,
                AssignmentNode,
                ReturnNode,
                ForNode,
                WhileNode,
                BreakNode,
                ContinueNode,
            ),
        )

    def generate_implicit_final_return_statement(
        self,
        stmt,
        indent,
        loop_contexts=None,
    ):
        if isinstance(stmt, MatchesMacroNode):
            return self.generate_matches_expression_return(stmt, indent, loop_contexts)

        return self.generate_expression_result(
            stmt,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_function_body(
        self,
        body,
        indent=1,
        loop_contexts=None,
        allow_implicit_final_return=False,
    ):
        code = ""
        indent_str = "    " * indent
        body = body or []
        loop_contexts = loop_contexts or []
        scoped_aliases = self.local_aliasing_enabled()
        if scoped_aliases:
            self.push_name_alias_scope()
        pushed_callable_scope = False
        if not self.local_callable_scopes:
            self.push_local_callable_scope()
            pushed_callable_scope = True

        try:
            self.predeclare_local_function_items(body)
            for index, stmt in enumerate(body):
                if (
                    allow_implicit_final_return
                    and self.is_implicit_final_return_statement(
                        stmt,
                        index,
                        body,
                    )
                ):
                    code += self.generate_implicit_final_return_statement(
                        stmt,
                        indent,
                        loop_contexts,
                    )
                elif isinstance(stmt, ConstNode):
                    code += self.generate_const_statement(stmt, indent)
                elif isinstance(stmt, StaticNode):
                    code += self.generate_static_statement(stmt, indent)
                elif isinstance(stmt, FunctionNode):
                    code += self.generate_local_function_item(stmt)
                elif isinstance(stmt, UseNode):
                    code += f"{indent_str}// use {stmt.path}\n"
                elif isinstance(stmt, LetNode):
                    code += self.generate_let_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, AssignmentNode):
                    code += self.generate_assignment_statement(
                        stmt, indent, loop_contexts
                    )
                elif isinstance(stmt, ReturnNode):
                    if isinstance(stmt.value, LoopNode):
                        code += self.generate_loop_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif self.is_block_expression_node(stmt.value):
                        code += self.generate_block_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, IfNode):
                        code += self.generate_if_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, MatchNode):
                        code += self.generate_match_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, MatchesMacroNode):
                        code += self.generate_matches_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, AssignmentNode):
                        code += self.generate_assignment_expression_result(
                            stmt.value,
                            indent,
                            self.return_result_target,
                            loop_contexts,
                        )
                    elif self.expression_contains_try(stmt.value):
                        code += self.generate_expression_result(
                            stmt.value,
                            indent,
                            self.return_result_target,
                            loop_contexts,
                        )
                    elif self.expression_contains_inline_result_expression(stmt.value):
                        prelude, value = self.generate_materialized_expression(
                            stmt.value,
                            indent,
                            loop_contexts,
                        )
                        code += prelude
                        code += f"{indent_str}return {value};\n"
                    elif stmt.value:
                        code += f"{indent_str}return {self.generate_expression(stmt.value)};\n"
                    else:
                        code += f"{indent_str}return;\n"
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent, loop_contexts)
                elif isinstance(stmt, WhileNode):
                    code += self.generate_while_loop(stmt, indent, loop_contexts)
                elif isinstance(stmt, LoopNode):
                    code += self.generate_loop(
                        stmt, indent, loop_contexts=loop_contexts
                    )
                elif isinstance(stmt, MatchNode):
                    code += self.generate_match_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, BreakNode):
                    code += self.generate_break_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, ContinueNode):
                    code += self.generate_continue_statement(
                        stmt, indent, loop_contexts
                    )
                elif isinstance(stmt, FunctionCallNode):
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )
                elif isinstance(stmt, BinaryOpNode):
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )
                elif isinstance(stmt, str):
                    code += f"{indent_str}{stmt};\n"
                else:
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )

                if self.statement_needs_labeled_control_propagation(
                    stmt,
                    loop_contexts,
                ):
                    code += self.generate_labeled_control_propagation(
                        loop_contexts,
                        indent,
                    )
        finally:
            if scoped_aliases:
                self.pop_name_alias_scope()
            if pushed_callable_scope:
                self.pop_local_callable_scope()

        return code

    def generate_local_function_item(self, func):
        helper_name = self.get_local_function_item_helper_name(func)
        self.add_local_callable(func.name, helper_name)
        helper_func = FunctionNode(
            func.return_type,
            helper_name,
            func.params,
            func.body,
            attributes=func.attributes,
            visibility=func.visibility,
            generics=func.generics,
            where_clauses=func.where_clauses,
            is_async=func.is_async,
            is_unsafe=func.is_unsafe,
            abi=func.abi,
            is_const=func.is_const,
        )
        helper_code = self.generate_function(helper_func, 0)

        if self.current_closure_helpers is not None:
            self.current_closure_helpers.append(helper_code)
            return ""
        return helper_code

    def generate_const_statement(self, node, indent):
        indent_str = "    " * indent
        declarator = self.format_typed_declarator(node.vtype, node.name)
        value = self.generate_expression(node.value)
        return f"{indent_str}const {declarator} = {value};\n"

    def generate_static_statement(self, node, indent):
        indent_str = "    " * indent
        mutability = "mut " if node.is_mutable else ""
        declarator = self.format_typed_declarator(node.vtype, node.name)
        value = self.generate_expression(node.value)
        return f"{indent_str}static {mutability}{declarator} = {value};\n"

    def generate_let_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        type_str = ""

        if getattr(stmt, "else_body", None) is not None:
            return self.generate_let_else_statement(stmt, indent, loop_contexts)

        if self.is_nontrivial_let_pattern(stmt.name):
            return self.generate_pattern_let_statement(stmt, indent, loop_contexts)

        if self.is_discard_pattern(stmt.name):
            return self.generate_discard_let_statement(stmt, indent, loop_contexts)
        if self.is_tuple_pattern(stmt.name):
            return self.generate_tuple_pattern_let_statement(
                stmt,
                indent,
                loop_contexts,
            )

        if isinstance(stmt.value, LoopNode):
            return self.generate_loop_expression_let(stmt, indent, loop_contexts)
        if self.is_block_expression_node(stmt.value):
            return self.generate_block_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, IfNode):
            return self.generate_if_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, MatchNode):
            return self.generate_match_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, MatchesMacroNode):
            return self.generate_matches_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, AssignmentNode):
            code = ""
            target_name = self.declare_local_alias(stmt.name)
            if stmt.vtype:
                code += (
                    f"{indent_str}"
                    f"{self.format_typed_declarator(stmt.vtype, target_name)};\n"
                )
            return code + self.generate_assignment_expression_result(
                stmt.value,
                indent,
                target_name,
                loop_contexts,
            )
        if self.expression_contains_try(stmt.value):
            return self.generate_try_let_statement(stmt, indent, loop_contexts)

        if self.expression_contains_inline_result_expression(stmt.value):
            prelude, value_str = self.generate_materialized_expression(
                stmt.value,
                indent,
                loop_contexts,
            )
            target_name = self.declare_local_alias(stmt.name)
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return prelude + f"{indent_str}{type_str} = {value_str};\n"

            inferred_type = self.infer_value_type(stmt.value)
            if inferred_type:
                self.add_value_type(target_name, inferred_type)
            mutability = "mut " if stmt.is_mutable else ""
            return (
                prelude + f"{indent_str}let {mutability}{target_name} = {value_str};\n"
            )

        if isinstance(stmt.value, ClosureNode):
            helper_name = self.try_generate_closure_helper(
                stmt.value,
                context_name=stmt.name,
            )
            if helper_name is not None:
                target_name = self.declare_local_alias(stmt.name)
                self.add_local_callable(stmt.name, target_name)
                mutability = "mut " if stmt.is_mutable else ""
                return f"{indent_str}let {mutability}{target_name} = {helper_name};\n"

        if stmt.vtype:
            type_str = self.format_typed_declarator(stmt.vtype, stmt.name)
        elif stmt.value:
            type_str = ""

        if stmt.value:
            value_str = self.generate_expression(stmt.value)
            target_name = self.declare_local_alias(stmt.name)
            if isinstance(stmt.value, ClosureNode):
                self.add_local_callable(stmt.name, target_name)
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return f"{indent_str}{type_str} = {value_str};\n"
            inferred_type = self.infer_value_type(stmt.value)
            if inferred_type:
                self.add_value_type(target_name, inferred_type)
            mutability = "mut " if stmt.is_mutable else ""
            return f"{indent_str}let {mutability}{target_name} = {value_str};\n"
        else:
            target_name = self.declare_local_alias(stmt.name)
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return f"{indent_str}{type_str};\n"
            return f"{indent_str}{target_name};\n"

    def generate_let_else_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""

        indent_str = "    " * indent
        subject, code = self.generate_let_else_subject(
            stmt.value,
            indent,
            loop_contexts,
        )
        binding_names = self.collect_pattern_binding_names(stmt.name)
        code += self.generate_let_else_binding_declarations(
            stmt,
            binding_names,
            indent,
        )

        matched_flag = self.next_match_chain_flag_name()
        code += f"{indent_str}bool {matched_flag} = false;\n"

        def success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n"

        match_code = self.generate_nested_pattern_match(
            subject,
            stmt.name,
            indent + 1,
            success,
        )
        match_code = self.rewrite_pattern_binding_declarations_as_assignments(
            match_code,
            binding_names,
        )

        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += match_code
        code += f"{indent_str}}}\n"
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_scoped_function_body(
            stmt.else_body,
            indent + 1,
            loop_contexts,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_let_else_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            expression,
            indent,
            loop_contexts,
        )

        if isinstance(subject, str) and not code:
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            subject = subject_name

        return subject, code

    def generate_let_else_binding_declarations(self, stmt, binding_names, indent):
        indent_str = "    " * indent
        code = ""

        for name in binding_names:
            if (
                isinstance(stmt.name, str)
                and stmt.name == name
                and stmt.vtype is not None
            ):
                code += (
                    f"{indent_str}{self.format_typed_declarator(stmt.vtype, name)};\n"
                )
            else:
                code += f"{indent_str}{name};\n"

        return code

    def rewrite_pattern_binding_declarations_as_assignments(self, code, binding_names):
        for name in sorted(binding_names, key=len, reverse=True):
            pattern = rf"(?m)^(\s*)auto {re.escape(name)} ="
            code = re.sub(pattern, rf"\1{name} =", code)
        return code

    def collect_pattern_binding_names(self, pattern):
        names = []

        def add_name(name):
            if not self.is_discard_pattern(name) and name not in names:
                names.append(name)

        def visit(node):
            if isinstance(node, MatchBindingPatternNode):
                add_name(node.name)
                visit(node.pattern)
                return

            if isinstance(node, ReferenceNode):
                visit(node.expression)
                return

            if isinstance(node, MatchOrPatternNode):
                for alternative in node.patterns:
                    visit(alternative)
                return

            if isinstance(node, TupleNode):
                for element in node.elements:
                    visit(element)
                return

            if isinstance(node, ArrayNode):
                for element in node.elements:
                    visit(element)
                return

            if isinstance(node, MatchStructPatternNode):
                for _, field_pattern in node.fields:
                    visit(field_pattern)
                return

            if isinstance(node, FunctionCallNode):
                for arg in node.args:
                    visit(arg)
                return

            if isinstance(node, str) and self.is_simple_pattern_binding(node):
                add_name(node)

        visit(pattern)
        return names

    def is_nontrivial_let_pattern(self, pattern):
        return isinstance(
            pattern,
            (
                ArrayNode,
                FunctionCallNode,
                MatchBindingPatternNode,
                MatchOrPatternNode,
                MatchStructPatternNode,
                ReferenceNode,
            ),
        )

    def generate_pattern_let_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""

        subject, code = self.generate_let_else_subject(
            stmt.value,
            indent,
            loop_contexts,
        )
        binding_names = self.collect_pattern_binding_names(stmt.name)
        code += self.generate_let_else_binding_declarations(
            stmt,
            binding_names,
            indent,
        )

        match_code = self.generate_nested_pattern_match(
            subject,
            stmt.name,
            indent,
            lambda success_indent: "",
        )
        match_code = self.rewrite_pattern_binding_declarations_as_assignments(
            match_code,
            binding_names,
        )
        return code + match_code

    def is_discard_pattern(self, pattern):
        return pattern == "_"

    def is_tuple_pattern(self, pattern):
        return isinstance(pattern, TupleNode)

    def generate_discard_let_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""
        return self.generate_discarded_expression(stmt.value, indent, loop_contexts)

    def generate_tuple_pattern_let_statement(self, stmt, indent, loop_contexts=None):
        if not isinstance(stmt.value, TupleNode):
            if stmt.value is None:
                return ""
            return self.generate_discarded_expression(
                stmt.value,
                indent,
                loop_contexts,
            )

        type_elements = self.get_tuple_pattern_type_elements(
            stmt.vtype,
            len(stmt.name.elements),
        )
        return self.generate_tuple_pattern_bindings(
            stmt.name,
            stmt.value,
            indent,
            loop_contexts,
            type_elements,
        )

    def generate_tuple_pattern_bindings(
        self, pattern, value, indent, loop_contexts=None, type_elements=None
    ):
        if len(pattern.elements) != len(value.elements):
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if type_elements is not None and len(type_elements) != len(pattern.elements):
            type_elements = None

        code = ""
        for index, (pattern_element, value_element) in enumerate(
            zip(pattern.elements, value.elements)
        ):
            type_element = type_elements[index] if type_elements is not None else None
            code += self.generate_pattern_binding(
                pattern_element,
                value_element,
                indent,
                loop_contexts,
                type_element,
            )
        return code

    def generate_pattern_binding(
        self, pattern, value, indent, loop_contexts=None, type_name=None
    ):
        if self.is_discard_pattern(pattern):
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if self.is_tuple_pattern(pattern):
            if isinstance(value, TupleNode):
                type_elements = self.get_tuple_pattern_type_elements(
                    type_name,
                    len(pattern.elements),
                )
                return self.generate_tuple_pattern_bindings(
                    pattern,
                    value,
                    indent,
                    loop_contexts,
                    type_elements,
                )
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if isinstance(pattern, str):
            if type_name:
                return self.generate_typed_pattern_binding(
                    pattern,
                    value,
                    type_name,
                    indent,
                    loop_contexts,
                )
            return self.generate_expression_result(
                value,
                indent,
                pattern,
                loop_contexts,
            )

        return self.generate_discarded_expression(value, indent, loop_contexts)

    def get_tuple_pattern_type_elements(self, type_name, expected_count):
        elements = self.split_tuple_type(type_name)
        if elements is None or len(elements) != expected_count:
            return None
        return elements

    def split_tuple_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        text = type_name.strip()
        if not (text.startswith("(") and text.endswith(")")):
            return None

        inner = text[1:-1].strip()
        if not inner:
            return []

        return self.split_generic_arguments(inner)

    def generate_typed_pattern_binding(
        self, name, value, type_name, indent, loop_contexts=None
    ):
        indent_str = "    " * indent
        declarator = self.format_typed_declarator(type_name, name)

        if isinstance(
            value,
            (
                LoopNode,
                IfNode,
                MatchNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            code = f"{indent_str}{declarator};\n"
            code += self.generate_expression_result(
                value,
                indent,
                name,
                loop_contexts,
            )
            return code

        if self.expression_contains_try(value):
            code = f"{indent_str}{declarator};\n"
            code += self.generate_expression_result(
                value,
                indent,
                name,
                loop_contexts,
            )
            return code

        return f"{indent_str}{declarator} = {self.generate_expression(value)};\n"

    def generate_loop_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_loop(
            stmt.value,
            indent,
            result_target=stmt.name,
            loop_contexts=loop_contexts,
        )
        return code

    def generate_loop_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_loop(
            stmt.right,
            indent,
            result_target=target,
            loop_contexts=loop_contexts,
        )

    def generate_loop_expression_return(self, loop_node, indent, loop_contexts=None):
        return self.generate_loop(
            loop_node,
            indent,
            result_target=self.return_result_target,
            loop_contexts=loop_contexts,
        )

    def generate_block_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        target_name = self.declare_local_alias(stmt.name)
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(target_name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(stmt.vtype, target_name)};\n"
            )
        elif inferred_type:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(inferred_type, target_name)};\n"
            )
        else:
            code += f"{indent_str}auto {target_name};\n"

        code += self.generate_block_expression_result(
            stmt.value,
            indent,
            target_name,
            loop_contexts,
        )
        return code

    def generate_block_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_block_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_block_expression_return(self, block_node, indent, loop_contexts=None):
        return self.generate_block_expression_result(
            block_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_if_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(stmt.name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_if_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_if_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_if_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_if_expression_return(self, if_node, indent, loop_contexts=None):
        return self.generate_if_expression_result(
            if_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_match_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(stmt.name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_match_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_match_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_match_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_match_expression_return(self, match_node, indent, loop_contexts=None):
        return self.generate_match_expression_result(
            match_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_matches_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_matches_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_matches_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_matches_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_matches_expression_return(
        self, matches_node, indent, loop_contexts=None
    ):
        return self.generate_matches_expression_result(
            matches_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_matches_expression_result(
        self,
        matches_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if result_target is self.return_result_target:
            result_name = self.next_matches_result_name()
            code = f"{indent_str}bool {result_name} = false;\n"
            code += self.generate_matches_pattern_assignment(
                matches_node,
                indent,
                result_name,
                loop_contexts,
            )
            code += f"{indent_str}return {result_name};\n"
            return code

        code = f"{indent_str}{result_target} = false;\n"
        code += self.generate_matches_pattern_assignment(
            matches_node,
            indent,
            result_target,
            loop_contexts,
        )
        return code

    def generate_matches_pattern_assignment(
        self,
        matches_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        subject, code = self.generate_let_else_subject(
            matches_node.expression,
            indent,
            loop_contexts,
        )

        def success(success_indent):
            success_indent_str = "    " * success_indent
            if matches_node.guard is None:
                return f"{success_indent_str}{result_target} = true;\n"

            guard_code, guard = self.generate_try_expression(
                matches_node.guard,
                success_indent,
                loop_contexts,
            )
            return (
                guard_code + f"{success_indent_str}if ({guard}) {{\n"
                f"{success_indent_str}    {result_target} = true;\n"
                f"{success_indent_str}}}\n"
            )

        code += self.generate_nested_pattern_match(
            subject,
            matches_node.pattern,
            indent,
            success,
        )
        return code

    def generate_matches_condition_flag(
        self,
        matches_node,
        indent,
        loop_contexts=None,
    ):
        result_name = self.next_matches_result_name()
        indent_str = "    " * indent
        code = f"{indent_str}bool {result_name} = false;\n"
        code += self.generate_matches_pattern_assignment(
            matches_node,
            indent,
            result_name,
            loop_contexts,
        )
        return result_name, code

    def generate_if_expression_result(
        self, if_node, indent, result_target, loop_contexts=None
    ):
        if isinstance(if_node.condition, ConditionChainNode):
            return self.generate_condition_chain_if_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        if isinstance(if_node.condition, MatchesMacroNode):
            return self.generate_matches_if_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        if isinstance(if_node.condition, LetPatternConditionNode):
            return self.generate_if_let_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            if_node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_result_branch(
            if_node.if_body, indent + 1, result_target, loop_contexts
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_result_branch(
                if_node.else_body, indent + 1, result_target, loop_contexts
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_condition_chain_if_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        def success(success_indent):
            return self.generate_result_branch(
                if_node.if_body,
                success_indent,
                result_target,
                loop_contexts,
            )

        def failure(failure_indent):
            return self.generate_result_branch(
                if_node.else_body,
                failure_indent,
                result_target,
                loop_contexts,
            )

        return self.generate_condition_chain_branch(
            if_node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_if_let_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        def success(success_indent):
            return self.generate_result_branch(
                if_node.if_body,
                success_indent,
                result_target,
                loop_contexts,
            )

        def failure(failure_indent):
            return self.generate_result_branch(
                if_node.else_body,
                failure_indent,
                result_target,
                loop_contexts,
            )

        return self.generate_let_pattern_condition_branch(
            if_node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_matches_if_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        condition, code = self.generate_matches_condition_flag(
            if_node.condition,
            indent,
            loop_contexts,
        )
        indent_str = "    " * indent

        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_result_branch(
            if_node.if_body,
            indent + 1,
            result_target,
            loop_contexts,
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_result_branch(
                if_node.else_body,
                indent + 1,
                result_target,
                loop_contexts,
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_result_branch(self, branch, indent, result_target, loop_contexts=None):
        if self.is_block_expression_node(branch):
            return self.generate_block_expression_result(
                branch, indent, result_target, loop_contexts
            )
        if isinstance(branch, list):
            return self.generate_scoped_function_body(
                branch,
                indent,
                loop_contexts,
                allow_implicit_final_return=(
                    result_target is self.return_result_target
                ),
            )
        if branch is not None:
            return self.generate_expression_result(
                branch, indent, result_target, loop_contexts
            )
        return ""

    def generate_match_expression_result(
        self, match_node, indent, result_target, loop_contexts=None
    ):
        if self.match_requires_if_chain(match_node):
            return self.generate_match_if_chain(
                match_node,
                indent,
                lambda body, branch_indent, contexts: self.generate_result_branch(
                    body,
                    branch_indent,
                    result_target,
                    contexts,
                ),
                loop_contexts,
            )

        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            match_node.expression,
            indent,
            loop_contexts,
        )
        expression = self.generate_match_subject_expression(subject)
        switch_break_flag = self.create_switch_break_flag(match_node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        code += self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in match_node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_result_branch(
                arm.body,
                indent + 2,
                result_target,
                arm_loop_contexts,
            )
            if self.match_arm_needs_switch_terminator(arm.body):
                code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        code += self.generate_switch_break_propagation(
            switch_break_flag,
            loop_contexts,
            indent,
        )
        return code

    def generate_block_expression_result(
        self, block_node, indent, result_target, loop_contexts=None
    ):
        block_node = self.get_block_expression_node(block_node)
        indent_str = "    " * indent
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(
                block_node.statements, indent, loop_contexts
            )
            expression = self.get_block_expression(block_node)

            if isinstance(expression, LoopNode):
                code += self.generate_loop(
                    expression,
                    indent,
                    result_target=result_target,
                    loop_contexts=loop_contexts,
                )
            elif isinstance(expression, IfNode):
                code += self.generate_if_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif isinstance(expression, MatchNode):
                code += self.generate_match_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif expression is not None:
                code += self.generate_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif (
                result_target is self.return_result_target
                and not self.branch_guarantees_control_transfer(block_node)
            ):
                code += f"{indent_str}return;\n"

            return code
        finally:
            self.pop_local_callable_scope()

    def generate_expression_result(
        self, expression, indent, result_target, loop_contexts=None
    ):
        indent_str = "    " * indent

        if isinstance(expression, LoopNode):
            return self.generate_loop(
                expression,
                indent,
                result_target=result_target,
                loop_contexts=loop_contexts,
            )
        if isinstance(expression, TryBlockNode):
            return self.generate_try_block_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, IfNode):
            return self.generate_if_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, MatchNode):
            return self.generate_match_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if self.is_block_expression_node(expression):
            return self.generate_block_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, AssignmentNode):
            return self.generate_assignment_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )

        if self.expression_contains_try(expression):
            prelude, value = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            if result_target is self.return_result_target:
                return prelude + f"{indent_str}return {value};\n"
            if result_target:
                return prelude + f"{indent_str}{result_target} = {value};\n"
            if value:
                return prelude + f"{indent_str}{value};\n"
            return prelude

        if self.expression_contains_inline_result_expression(expression):
            prelude, value = self.generate_materialized_expression(
                expression,
                indent,
                loop_contexts,
            )
            if result_target is self.return_result_target:
                return prelude + f"{indent_str}return {value};\n"
            if result_target:
                return prelude + f"{indent_str}{result_target} = {value};\n"
            if value:
                return prelude + f"{indent_str}{value};\n"
            return prelude

        value = self.generate_expression(expression)
        if result_target is self.return_result_target:
            return f"{indent_str}return {value};\n"
        if result_target:
            return f"{indent_str}{result_target} = {value};\n"
        if value:
            return f"{indent_str}{value};\n"
        return ""

    def generate_try_let_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}auto {stmt.name};\n"

        code += self.generate_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_assignment_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent

        if stmt.operator == "=" and isinstance(stmt.right, LoopNode):
            return self.generate_loop_expression_assignment(stmt, indent, loop_contexts)
        if stmt.operator == "=" and self.is_block_expression_node(stmt.right):
            return self.generate_block_expression_assignment(
                stmt, indent, loop_contexts
            )
        if stmt.operator == "=" and isinstance(stmt.right, IfNode):
            return self.generate_if_expression_assignment(stmt, indent, loop_contexts)
        if stmt.operator == "=" and isinstance(stmt.right, MatchNode):
            return self.generate_match_expression_assignment(
                stmt, indent, loop_contexts
            )
        if stmt.operator == "=" and isinstance(stmt.right, MatchesMacroNode):
            return self.generate_matches_expression_assignment(
                stmt, indent, loop_contexts
            )
        if self.expression_contains_try(stmt.right):
            return self.generate_try_assignment_statement(stmt, indent, loop_contexts)
        if self.expression_contains_inline_result_expression(stmt.right):
            prelude, value = self.generate_materialized_expression(
                stmt.right,
                indent,
                loop_contexts,
            )
            target = self.generate_expression(stmt.left)
            return prelude + f"{indent_str}{target} {stmt.operator} {value};\n"
        return f"{indent_str}{self.generate_assignment(stmt)};\n"

    def generate_assignment_expression_result(
        self, stmt, indent, result_target, loop_contexts=None
    ):
        indent_str = "    " * indent
        code = self.generate_assignment_statement(stmt, indent, loop_contexts)

        if result_target is self.return_result_target:
            return code + f"{indent_str}return;\n"
        if result_target:
            return code + f"{indent_str}{result_target} = ();\n"
        return code

    def generate_try_assignment_statement(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        if stmt.operator != "=":
            prelude, value = self.generate_try_expression(
                stmt.right,
                indent,
                loop_contexts,
            )
            indent_str = "    " * indent
            return prelude + f"{indent_str}{target} {stmt.operator} {value};\n"
        return self.generate_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_try_expression(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent

        if isinstance(expression, TryNode):
            return self.generate_try_node_expression(
                expression,
                indent,
                loop_contexts,
            )

        if self.is_transparent_block_expression_node(expression):
            value_name = self.next_transparent_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_block_expression_result(
                expression,
                indent,
                value_name,
                loop_contexts,
            )
            return code, value_name

        if isinstance(expression, TryBlockNode):
            value_name = self.next_try_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_try_block_expression_result(
                expression,
                indent,
                value_name,
                loop_contexts,
            )
            return code, value_name

        if not self.expression_contains_try(expression):
            return "", self.generate_expression(expression)

        if isinstance(expression, BinaryOpNode):
            left_code, left = self.generate_try_expression(
                expression.left,
                indent,
                loop_contexts,
            )
            right_code, right = self.generate_try_expression(
                expression.right,
                indent,
                loop_contexts,
            )
            return left_code + right_code, f"({left} {expression.op} {right})"

        if isinstance(expression, UnaryOpNode):
            code, operand = self.generate_try_expression(
                expression.operand,
                indent,
                loop_contexts,
            )
            return code, f"({expression.op}{operand})"

        if isinstance(expression, FunctionCallNode):
            return self.generate_try_function_call_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, MemberAccessNode):
            code, obj = self.generate_try_expression(
                expression.object,
                indent,
                loop_contexts,
            )
            member_name = self.resolve_member_access_name(obj, expression.member)
            return code, f"{obj}.{member_name}"

        if isinstance(expression, AwaitNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, ArrayAccessNode):
            array_code, array = self.generate_try_expression(
                expression.array,
                indent,
                loop_contexts,
            )
            index_code, index = self.generate_try_expression(
                expression.index,
                indent,
                loop_contexts,
            )
            return array_code + index_code, f"{array}[{index}]"

        if isinstance(expression, VectorConstructorNode):
            code, args = self.generate_try_argument_list(
                expression.args,
                indent,
                loop_contexts,
            )
            type_name = self.map_type(expression.type_name)
            return code, f"{type_name}({', '.join(args)})"

        if isinstance(expression, TernaryOpNode):
            return self.generate_try_ternary_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, CastNode):
            code, value = self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )
            return code, self.format_cast_expression(expression.target_type, value)

        if isinstance(expression, ReferenceNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, DereferenceNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, TupleNode):
            code, elements = self.generate_try_argument_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"({', '.join(elements)})"

        if isinstance(expression, ArrayNode):
            code, elements = self.generate_try_argument_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            if expression.size is not None and len(elements) == 1:
                size_code, size = self.generate_try_expression(
                    expression.size,
                    indent,
                    loop_contexts,
                )
                return code + size_code, f"_rust_repeat({elements[0]}, {size})"
            return code, "{" + ", ".join(elements) + "}"

        if isinstance(expression, RangeNode):
            start_code, start = self.generate_try_range_bound(
                expression.start,
                indent,
                loop_contexts,
            )
            end_code, end = self.generate_try_range_bound(
                expression.end,
                indent,
                loop_contexts,
            )
            return start_code + end_code, self.format_range_expression(
                start,
                end,
                expression.inclusive,
            )

        if isinstance(expression, StructInitializationNode):
            return self.generate_try_struct_initialization_expression(
                expression,
                indent,
                loop_contexts,
            )
        return "", self.generate_expression(expression)

    def generate_try_range_bound(self, bound, indent, loop_contexts=None):
        if bound is None:
            return "", ""
        return self.generate_try_expression(bound, indent, loop_contexts)

    def format_range_expression(self, start, end, inclusive=False):
        operator = "..=" if inclusive else ".."
        return f"{start or ''}{operator}{end or ''}"

    def generate_try_node_expression(self, try_node, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject, code = self.generate_try_subject(
            try_node.expression,
            indent,
            loop_contexts,
        )
        code += self.generate_try_propagation(subject, indent)

        value_name = self.next_try_value_name()
        code += (
            f"{indent_str}auto {value_name} = {self.generate_try_unwrap(subject)};\n"
        )
        return code, value_name

    def generate_try_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject_name = self.next_try_subject_name()

        if isinstance(expression, TryBlockNode):
            code = f"{indent_str}auto {subject_name};\n"
            code += self.generate_try_block_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        if isinstance(
            expression,
            (
                LoopNode,
                IfNode,
                MatchNode,
                MatchesMacroNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            code = f"{indent_str}auto {subject_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        prelude, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )
        code = prelude + f"{indent_str}auto {subject_name} = {value};\n"
        return subject_name, code

    def generate_try_propagation(self, subject, indent):
        indent_str = "    " * indent
        if self.current_try_kind() == "option":
            return (
                f"{indent_str}if (is_None({subject})) {{\n"
                f"{indent_str}    return None;\n"
                f"{indent_str}}}\n"
            )

        return (
            f"{indent_str}if (is_Err({subject})) {{\n"
            f"{indent_str}    return Err(unwrap_Err({subject}));\n"
            f"{indent_str}}}\n"
        )

    def generate_try_unwrap(self, subject):
        if self.current_try_kind() == "option":
            return f"unwrap_Some({subject})"
        return f"unwrap_Ok({subject})"

    def current_try_kind(self):
        return_type = self.current_function_return_type or ""
        return_type = self.strip_reference_type(return_type)
        generic = self.parse_generic_type(return_type)
        base_name = generic[0] if generic is not None else return_type
        base_name = base_name.rsplit("::", 1)[-1]

        if base_name == "Option":
            return "option"
        return "result"

    def generate_try_argument_list(self, args, indent, loop_contexts=None):
        code = ""
        values = []

        for arg in args:
            arg_code, arg_value = self.generate_try_expression(
                arg,
                indent,
                loop_contexts,
            )
            code += arg_code
            values.append(arg_value)

        return code, values

    def generate_try_function_call_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        if isinstance(expression.name, MemberAccessNode):
            object_code, obj = self.generate_try_expression(
                expression.name.object,
                indent,
                loop_contexts,
            )
            args_code, args = self.generate_try_argument_list(
                expression.args,
                indent,
                loop_contexts,
            )
            method_call = self.format_method_call_parts(
                expression.name.member,
                obj,
                args,
                expression.args,
                self.infer_value_type(expression.name.object),
            )
            if method_call is not None:
                return object_code + args_code, method_call
            return (
                object_code + args_code,
                f"{obj}.{expression.name.member}({', '.join(args)})",
            )

        args_code, args = self.generate_try_argument_list(
            expression.args,
            indent,
            loop_contexts,
        )

        if isinstance(expression.name, str):
            constructor = self.format_path_constructor_call_parts(
                expression.name,
                args,
            )
            if constructor is not None:
                return args_code, constructor
            associated_call = self.format_associated_impl_function_call(
                expression.name,
                args,
            )
            if associated_call is not None:
                return args_code, associated_call
            return args_code, f"{self.map_function(expression.name)}({', '.join(args)})"

        name_code, name = self.generate_try_expression(
            expression.name,
            indent,
            loop_contexts,
        )
        return name_code + args_code, f"{name}({', '.join(args)})"

    def format_method_call_parts(
        self,
        method_name,
        obj,
        args,
        arg_nodes,
        receiver_type=None,
    ):
        impl_call = self.format_user_impl_method_call(method_name, obj, args)
        if impl_call is not None:
            return impl_call

        resource_call = self.format_resource_method_call(
            method_name,
            obj,
            args,
            receiver_type,
        )
        if resource_call is not None:
            return resource_call

        if method_name == "len" and not args:
            return f"{obj}.length"

        derivative_method = self.RUST_GPU_DERIVATIVE_METHOD_MAP.get(method_name)
        if derivative_method is not None and not args:
            return f"{derivative_method}({obj})"

        if not args and method_name in self.SCALAR_ZERO_ARG_METHOD_MAP:
            return f"{self.SCALAR_ZERO_ARG_METHOD_MAP[method_name]}({obj})"

        if method_name == "recip" and not args:
            return f"(1.0 / {obj})"

        if len(args) == 1 and method_name in self.SCALAR_ONE_ARG_METHOD_MAP:
            return f"{self.SCALAR_ONE_ARG_METHOD_MAP[method_name]}({obj}, {args[0]})"

        if len(args) == 2 and method_name in self.SCALAR_TWO_ARG_METHOD_MAP:
            return (
                f"{self.SCALAR_TWO_ARG_METHOD_MAP[method_name]}"
                f"({obj}, {args[0]}, {args[1]})"
            )

        if method_name == "is_multiple_of" and len(args) == 1:
            return f"(({obj} % {args[0]}) == 0)"

        if method_name == "extend" and len(args) == 1:
            constructor = self.extended_vector_constructor(receiver_type)
            if constructor is not None:
                return f"{constructor}({obj}, {args[0]})"

        if method_name == "length" and not args:
            return f"length({obj})"

        if (
            method_name == "length_recip"
            and not args
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"(1.0 / length({obj}))"

        if (
            method_name == "length_squared"
            and not args
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"dot({obj}, {obj})"

        if method_name == "normalize" and not args:
            return f"normalize({obj})"

        mapped_receiver_type = self.map_type(receiver_type)
        if (
            method_name == "normalize_or_zero"
            and not args
            and mapped_receiver_type in self.VECTOR_COMPONENT_COUNTS
            and mapped_receiver_type.startswith("vec")
        ):
            zero, _ = self.vector_zero_one_literals(mapped_receiver_type)
            return (
                f"((length({obj}) > 0.0) ? normalize({obj}) : "
                f"{mapped_receiver_type}({zero}))"
            )

        if method_name in {"dot", "cross"} and len(args) == 1:
            return f"{method_name}({obj}, {args[0]})"

        if (
            method_name == "dot_into_vec"
            and len(args) == 1
            and mapped_receiver_type in self.VECTOR_COMPONENT_COUNTS
            and mapped_receiver_type.startswith("vec")
        ):
            return f"{mapped_receiver_type}(dot({obj}, {args[0]}))"

        if (
            method_name == "distance"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"distance({obj}, {args[0]})"

        if (
            method_name == "distance_squared"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            delta = f"({obj} - {args[0]})"
            return f"dot({delta}, {delta})"

        if (
            method_name == "reflect"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"reflect({obj}, {args[0]})"

        if (
            method_name == "refract"
            and len(args) == 2
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"refract({obj}, {args[0]}, {args[1]})"

        if (
            method_name == "perp_dot"
            and len(args) == 1
            and self.map_type(receiver_type) == "vec2"
        ):
            return f"(({obj}.x * {args[0]}.y) - ({obj}.y * {args[0]}.x))"

        if (
            method_name in {"map", "filter", "for_each", "any", "all"}
            and len(args) == 1
            and isinstance(arg_nodes[0], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                arg_nodes[0],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"{method_name}({obj}, {helper_name})"
            return f"{method_name}({obj}, {args[0]})"

        if (
            method_name == "fold"
            and len(args) == 2
            and isinstance(arg_nodes[1], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                arg_nodes[1],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"fold({obj}, {args[0]}, {helper_name})"
            return f"fold({obj}, {args[0]}, {args[1]})"

        if self.is_swizzle_member(method_name) and not args:
            return f"{obj}.{method_name}"

        return None

    def extended_vector_constructor(self, receiver_type):
        if not receiver_type:
            return None

        mapped_type = self.map_type(receiver_type)
        return self.VECTOR_EXTEND_CONSTRUCTOR_MAP.get(mapped_type)

    def format_resource_method_call(self, method_name, obj, args, receiver_type=None):
        if not self.is_resource_method_name(method_name):
            return None

        mapped_resource_type = self.map_resource_method_receiver_type(
            obj,
            receiver_type,
        )
        if mapped_resource_type is None:
            return None

        sample_with_call = self.format_rust_gpu_sample_with_method_call(
            method_name,
            obj,
            args,
            mapped_resource_type,
        )
        if sample_with_call is not None:
            return sample_with_call

        fetch_with_call = self.format_rust_gpu_fetch_with_method_call(
            method_name,
            obj,
            args,
            mapped_resource_type,
        )
        if fetch_with_call is not None:
            return fetch_with_call

        mapped = self.function_map.get(method_name)
        if mapped is None:
            mapped = self.map_rust_gpu_resource_method(
                method_name,
                mapped_resource_type,
            )
        if mapped is None:
            return None

        return f"{mapped}({', '.join([obj] + args)})"

    def format_rust_gpu_sample_with_method_call(
        self,
        method_name,
        obj,
        args,
        mapped_resource_type,
    ):
        if not mapped_resource_type.startswith("sampler"):
            return None

        base_intrinsic = self.RUST_GPU_SAMPLE_WITH_METHOD_MAP.get(method_name)
        if base_intrinsic is None or not args:
            return None

        parsed_operand = self.parse_rust_gpu_sample_with_operand(args[-1])
        if parsed_operand is None:
            return None

        operand_name, operand_args = parsed_operand
        intrinsic = self.rust_gpu_sample_with_intrinsic(base_intrinsic, operand_name)
        if intrinsic is None:
            return None

        call_args = [obj] + args[:-1] + operand_args
        return f"{intrinsic}({', '.join(call_args)})"

    def format_rust_gpu_fetch_with_method_call(
        self,
        method_name,
        obj,
        args,
        mapped_resource_type,
    ):
        if method_name != "fetch_with" or len(args) != 2:
            return None
        if not mapped_resource_type.startswith("sampler"):
            return None

        parsed_operand = self.parse_rust_gpu_sample_index_operand(args[-1])
        if parsed_operand is None:
            return None

        return f"texelFetch({', '.join([obj, args[0], parsed_operand])})"

    def parse_rust_gpu_sample_with_operand(self, operand):
        parsed = self.parse_generated_call_expression(operand.strip())
        if parsed is None:
            return None

        function_name, args = parsed
        function_name, _ = self.split_function_type_arguments(function_name)
        operand_name = function_name.rsplit("::", 1)[-1]
        if operand_name not in {"bias", "lod", "grad"}:
            return None

        if operand_name in {"bias", "lod"} and len(args) != 1:
            return None
        if operand_name == "grad" and len(args) != 2:
            return None

        return operand_name, args

    def parse_rust_gpu_sample_index_operand(self, operand):
        parsed = self.parse_generated_call_expression(operand.strip())
        if parsed is None:
            return None

        function_name, args = parsed
        function_name, _ = self.split_function_type_arguments(function_name)
        operand_name = function_name.rsplit("::", 1)[-1]
        if operand_name != "sample_index" or len(args) != 1:
            return None

        return args[0]

    def rust_gpu_sample_with_intrinsic(self, base_intrinsic, operand_name):
        if operand_name == "bias":
            if base_intrinsic.startswith("textureCompare"):
                return None
            return base_intrinsic
        if operand_name == "lod":
            return f"{base_intrinsic}Lod"
        if operand_name == "grad":
            return f"{base_intrinsic}Grad"
        return None

    def is_resource_method_name(self, method_name):
        return isinstance(method_name, str) and (
            method_name.startswith(self.RESOURCE_METHOD_PREFIXES)
            or method_name in self.RESOURCE_METHOD_NAMES
        )

    def is_resource_method_receiver(self, obj, receiver_type=None):
        return self.map_resource_method_receiver_type(obj, receiver_type) is not None

    def map_resource_method_receiver_type(self, obj, receiver_type=None):
        if receiver_type is None:
            receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return None

        mapped_type = self.map_resource_receiver_type(receiver_type)
        if isinstance(mapped_type, str) and mapped_type.startswith(
            self.RESOURCE_TYPE_PREFIXES
        ):
            return mapped_type
        return None

    def map_rust_gpu_resource_method(self, method_name, mapped_resource_type):
        if method_name == "read":
            return "imageLoad"
        if method_name == "write":
            return "imageStore"
        if method_name == "fetch":
            if mapped_resource_type.startswith("sampler"):
                return "texelFetch"
            return "imageLoad"
        mapped_sample = self.RUST_GPU_SAMPLE_METHOD_MAP.get(method_name)
        if mapped_sample is not None and mapped_resource_type.startswith("sampler"):
            return mapped_sample
        if method_name in {"query_size", "query_size_lod"}:
            if mapped_resource_type.startswith("sampler"):
                return "textureSize"
            return "imageSize"
        if method_name == "query_levels":
            if mapped_resource_type.startswith("sampler"):
                return "textureQueryLevels"
            return None
        if method_name == "query_samples":
            if mapped_resource_type.startswith("sampler"):
                return "textureSamples"
            return "imageSamples"
        return None

    def map_resource_receiver_type(self, receiver_type, seen=None):
        mapped_type = self.map_type(receiver_type)
        if isinstance(mapped_type, str) and mapped_type.startswith(
            self.RESOURCE_TYPE_PREFIXES
        ):
            return mapped_type

        alias_target = self.resolve_type_alias_target(receiver_type)
        if alias_target is None:
            return mapped_type

        seen = seen or set()
        if alias_target in seen:
            return mapped_type

        seen.add(alias_target)
        return self.map_resource_receiver_type(alias_target, seen)

    def format_user_impl_method_call(self, method_name, obj, args):
        match = self.lookup_impl_method_receiver_match(obj, method_name)
        if match is None:
            return None

        call_args = [obj] + args
        return f"{match['call_prefix']}_{method_name}({', '.join(call_args)})"

    def format_associated_impl_function_call(self, function_name, args):
        match = self.lookup_impl_associated_function_match(function_name)
        if match is None:
            return None

        call = f"{match['call_prefix']}_{match['method_name']}({', '.join(args)})"
        return_type = self.infer_impl_associated_function_return_type(match, args)
        if return_type is not None:
            self.add_value_type(call, return_type)
        return call

    def lookup_impl_method_receiver_match(self, obj, method_name):
        receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return None

        for signature in self.impl_method_signatures.values():
            method = signature["methods"].get(method_name)
            if method is None:
                continue
            if not self.method_has_self_parameter(method):
                continue

            substitutions = self.match_impl_receiver_type(
                signature["struct_name"],
                receiver_type,
                signature["generic_names"],
            )
            if substitutions is None:
                continue

            return {
                "receiver_type": receiver_type,
                "call_prefix": signature["call_prefix"],
                "generic_names": signature["generic_names"],
                "method": method,
                "substitutions": substitutions,
            }

        return None

    def lookup_impl_associated_function_match(self, function_name):
        parsed = self.parse_associated_function_path(function_name)
        if parsed is None:
            return None

        type_name, method_name, method_type_args = parsed
        for signature in self.impl_method_signatures.values():
            method = signature["methods"].get(method_name)
            if method is None:
                continue
            if self.method_has_self_parameter(method):
                continue

            receiver_type, substitutions = self.match_associated_impl_type(
                signature["struct_name"],
                type_name,
                signature["generic_names"],
            )
            if receiver_type is None:
                continue

            return {
                "receiver_type": receiver_type,
                "call_prefix": signature["call_prefix"],
                "method_name": method_name,
                "method_type_args": method_type_args,
                "generic_names": signature["generic_names"],
                "method": method,
                "substitutions": substitutions,
            }

        return None

    def parse_associated_function_path(self, function_name):
        if not isinstance(function_name, str) or "::" not in function_name:
            return None

        function_name = self.normalize_associated_type_path(function_name)
        type_name, method_name = function_name.rsplit("::", 1)
        method_name, method_type_args = self.split_function_type_arguments(method_name)
        if not type_name or not method_name:
            return None
        return type_name, method_name, method_type_args

    def normalize_associated_type_path(self, type_name):
        if not isinstance(type_name, str):
            return type_name
        return type_name.replace("::<", "<")

    def match_associated_impl_type(self, pattern_type, call_type, generic_names):
        call_type = self.resolve_imported_module_path(call_type)
        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            pattern_generic = self.parse_generic_type(pattern_candidate)
            for call_candidate in self.generic_type_match_candidates(call_type):
                if self.impl_type_names_match(pattern_candidate, call_candidate):
                    return call_candidate, {}

                call_generic = self.parse_generic_type(call_candidate)
                if pattern_generic is None:
                    if call_generic is None:
                        continue
                    call_base, _ = call_generic
                    if self.generic_type_bases_match(pattern_candidate, call_base):
                        return pattern_candidate, {}
                    continue

                pattern_base, pattern_args = pattern_generic
                if call_generic is None:
                    if self.generic_type_bases_match(pattern_base, call_candidate):
                        return pattern_candidate, {}
                    continue

                call_base, call_args = call_generic
                if not self.generic_type_bases_match(pattern_base, call_base):
                    continue
                if len(pattern_args) != len(call_args):
                    continue

                substitutions = {}
                conflicts = set()
                for pattern_arg, call_arg in zip(pattern_args, call_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        call_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                if conflicts:
                    continue
                return call_candidate, substitutions

        return None, {}

    def match_impl_receiver_type(self, pattern_type, receiver_type, generic_names):
        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            for receiver_candidate in self.generic_type_match_candidates(receiver_type):
                if self.impl_type_names_match(pattern_candidate, receiver_candidate):
                    return {}

                pattern_generic = self.parse_generic_type(pattern_candidate)
                receiver_generic = self.parse_generic_type(receiver_candidate)
                if pattern_generic is None or receiver_generic is None:
                    continue

                pattern_base, pattern_args = pattern_generic
                receiver_base, receiver_args = receiver_generic
                if not self.generic_type_bases_match(pattern_base, receiver_base):
                    continue
                if len(pattern_args) != len(receiver_args):
                    continue

                substitutions = {}
                conflicts = set()
                for pattern_arg, receiver_arg in zip(pattern_args, receiver_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        receiver_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                if conflicts:
                    continue
                return substitutions

        return None

    def impl_type_names_match(self, left, right):
        return any(
            left_name == right_name
            for left_name in self.type_lookup_names(left)
            for right_name in self.type_lookup_names(right)
        )

    def format_path_constructor_call_parts(self, function_name, args):
        if "::" not in function_name:
            return None

        type_name, constructor_name = function_name.rsplit("::", 1)
        if constructor_name not in {"new", "splat"}:
            return None

        mapped_type = self.map_type(type_name)
        if mapped_type == type_name:
            return None

        if constructor_name == "splat":
            if len(args) != 1 or mapped_type not in self.VECTOR_SPLAT_CONSTRUCTORS:
                return None
            return f"{mapped_type}({args[0]})"

        return f"{mapped_type}({', '.join(args)})"

    def format_vector_associated_constant(self, value):
        if not isinstance(value, str) or "::" not in value:
            return None

        type_name, constant_name = value.rsplit("::", 1)
        mapped_type = self.map_type(type_name)
        components = self.vector_associated_constant_components(
            mapped_type,
            constant_name,
        )
        if components is None:
            return None

        return f"{mapped_type}({', '.join(components)})"

    def vector_associated_constant_components(self, mapped_type, constant_name):
        count = self.VECTOR_COMPONENT_COUNTS.get(mapped_type)
        if count is None:
            return None

        zero, one = self.vector_zero_one_literals(mapped_type)
        if constant_name == "ZERO":
            return [zero] * count
        if constant_name == "ONE":
            return [one] * count
        if constant_name == "NEG_ONE":
            negative_one = self.vector_negative_one_literal(mapped_type)
            if negative_one is None:
                return None
            return [negative_one] * count

        axis_constant_name = constant_name
        axis_value = one
        if constant_name.startswith("NEG_"):
            negative_one = self.vector_negative_one_literal(mapped_type)
            if negative_one is None:
                return None
            axis_constant_name = constant_name[4:]
            axis_value = negative_one

        axis_index = self.VECTOR_AXIS_CONSTANT_INDICES.get(axis_constant_name)
        if axis_index is None or axis_index >= count:
            return None

        components = [zero] * count
        components[axis_index] = axis_value
        return components

    def format_scalar_associated_constant(self, value):
        if not isinstance(value, str):
            return None
        return self.SCALAR_ASSOCIATED_CONSTANTS.get(value)

    def vector_zero_one_literals(self, mapped_type):
        if mapped_type.startswith("bvec"):
            return "false", "true"
        if mapped_type.startswith("vec"):
            return "0.0", "1.0"
        return "0", "1"

    def vector_negative_one_literal(self, mapped_type):
        if mapped_type.startswith("vec"):
            return "-1.0"
        if mapped_type.startswith("ivec"):
            return "-1"
        return None

    def generate_try_ternary_expression(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            expression.condition,
            indent,
            loop_contexts,
        )
        result_name = self.next_try_value_name()
        code = condition_code
        code += f"{indent_str}auto {result_name};\n"
        code += f"{indent_str}if ({condition}) {{\n"
        true_code, true_value = self.generate_try_expression(
            expression.true_expr,
            indent + 1,
            loop_contexts,
        )
        code += true_code
        code += f"{indent_str}    {result_name} = {true_value};\n"
        code += f"{indent_str}}} else {{\n"
        false_code, false_value = self.generate_try_expression(
            expression.false_expr,
            indent + 1,
            loop_contexts,
        )
        code += false_code
        code += f"{indent_str}    {result_name} = {false_value};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_try_struct_initialization_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        code = ""
        fields = []

        for field_name, field_expression in expression.fields:
            field_code, field_value = self.generate_try_expression(
                field_expression,
                indent,
                loop_contexts,
            )
            code += field_code
            fields.append(f"{field_name}: {field_value}")

        return code, f"{expression.struct_name} {{ {', '.join(fields)} }}"

    def generate_try_block_expression(self, try_block):
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = previous_return_type or "Result"
        try:
            code = self.generate_try_block_body_code(try_block.block, indent=0)
        finally:
            self.current_function_return_type = previous_return_type

        return f"lambda({{ {self.compact_generated_block(code)} }})()"

    def generate_try_block_expression_result(
        self,
        try_block,
        indent,
        result_target,
        loop_contexts=None,
    ):
        if result_target is self.return_result_target:
            previous_return_type = self.current_function_return_type
            self.current_function_return_type = previous_return_type or "Result"
            try:
                return self.generate_try_block_body_code(try_block.block, indent)
            finally:
                self.current_function_return_type = previous_return_type

        if not result_target:
            result_target = self.next_try_block_value_name()

        indent_str = "    " * indent
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = previous_return_type or "Result"
        try:
            block_code = self.generate_try_block_body_code(try_block.block, indent + 1)
        finally:
            self.current_function_return_type = previous_return_type

        code = f"{indent_str}do {{\n"
        code += self.rewrite_try_block_returns_to_target(block_code, result_target)
        code += f"{indent_str}}} while (false);\n"
        return code

    def rewrite_try_block_returns_to_target(self, code, result_target):
        rewritten = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("return ") and stripped.endswith(";"):
                indent = line[: len(line) - len(line.lstrip())]
                value = stripped[len("return ") : -1]
                rewritten.append(f"{indent}{result_target} = {value};")
                rewritten.append(f"{indent}continue;")
            else:
                rewritten.append(line)
        return "\n".join(rewritten) + ("\n" if code.endswith("\n") else "")

    def generate_try_block_body_code(self, block_node, indent=0):
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(block_node.statements, indent=indent)
            expression = self.get_block_expression(block_node)
            code += self.generate_try_block_success_return(expression, indent)
            return code
        finally:
            self.pop_local_callable_scope()

    def generate_try_block_success_return(self, expression, indent):
        indent_str = "    " * indent

        if expression is None:
            return f"{indent_str}return {self.generate_try_success_value('()')};\n"

        if isinstance(
            expression,
            (
                LoopNode,
                IfNode,
                MatchNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            value_name = self.next_try_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                value_name,
            )
            code += (
                f"{indent_str}return "
                f"{self.generate_try_success_value(value_name)};\n"
            )
            return code

        prelude, value = self.generate_try_expression(expression, indent)
        return (
            prelude + f"{indent_str}return {self.generate_try_success_value(value)};\n"
        )

    def generate_try_success_value(self, value):
        if self.current_try_kind() == "option":
            return f"Some({value})"
        return f"Ok({value})"

    def expression_contains_try(self, expression):
        if expression is None:
            return False
        if isinstance(expression, TryNode):
            return True
        if isinstance(expression, TryBlockNode):
            return True
        if isinstance(expression, BinaryOpNode):
            return self.expression_contains_try(
                expression.left
            ) or self.expression_contains_try(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_contains_try(expression.operand)
        if isinstance(expression, FunctionCallNode):
            return self.expression_contains_try(expression.name) or any(
                self.expression_contains_try(arg) for arg in expression.args
            )
        if isinstance(expression, MemberAccessNode):
            return self.expression_contains_try(expression.object)
        if isinstance(expression, AwaitNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_contains_try(
                expression.array
            ) or self.expression_contains_try(expression.index)
        if isinstance(expression, VectorConstructorNode):
            return any(self.expression_contains_try(arg) for arg in expression.args)
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_contains_try(expression.condition)
                or self.expression_contains_try(expression.true_expr)
                or self.expression_contains_try(expression.false_expr)
            )
        if isinstance(expression, CastNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ReferenceNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, DereferenceNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, TupleNode):
            return any(
                self.expression_contains_try(elem) for elem in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_contains_try(elem) for elem in expression.elements
            ) or self.expression_contains_try(expression.size)
        if isinstance(expression, RangeNode):
            return self.expression_contains_try(
                expression.start
            ) or self.expression_contains_try(expression.end)
        if isinstance(expression, StructInitializationNode):
            return any(
                self.expression_contains_try(field_expression)
                for _, field_expression in expression.fields
            )
        if isinstance(expression, MatchesMacroNode):
            return self.expression_contains_try(
                expression.expression
            ) or self.expression_contains_try(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_contains_try(operand) for operand in expression.operands
            )
        if isinstance(expression, ClosureNode):
            return False
        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            return any(
                self.expression_contains_try(statement)
                for statement in block_node.statements
            ) or self.expression_contains_try(self.get_block_expression(block_node))
        if isinstance(expression, IfNode):
            return (
                self.expression_contains_try(expression.condition)
                or self.expression_contains_try(expression.if_body)
                or self.expression_contains_try(expression.else_body)
            )
        if isinstance(expression, MatchNode):
            return self.expression_contains_try(expression.expression) or any(
                self.expression_contains_try(arm.guard)
                or self.expression_contains_try(arm.body)
                for arm in expression.arms
            )
        if isinstance(expression, list):
            return any(self.expression_contains_try(item) for item in expression)
        if isinstance(expression, ReturnNode):
            return self.expression_contains_try(expression.value)
        if isinstance(expression, AssignmentNode):
            return self.expression_contains_try(
                expression.left
            ) or self.expression_contains_try(expression.right)
        if isinstance(expression, LetNode):
            return self.expression_contains_try(expression.value)
        return False

    def get_block_expression(self, block_node):
        return getattr(
            block_node, "expression", getattr(block_node, "returns_value", None)
        )

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"{left} {node.operator} {right}"

    def generate_if_statement(self, node, indent, loop_contexts=None):
        if isinstance(node.condition, ConditionChainNode):
            return self.generate_condition_chain_if_statement(
                node,
                indent,
                loop_contexts,
            )

        if isinstance(node.condition, MatchesMacroNode):
            return self.generate_matches_if_statement(node, indent, loop_contexts)

        if isinstance(node.condition, LetPatternConditionNode):
            return self.generate_if_let_statement(node, indent, loop_contexts)

        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.if_body, indent + 1, loop_contexts
        )
        code += f"{indent_str}}}"

        if node.else_body:
            if (
                isinstance(node.else_body, list)
                and len(node.else_body) == 1
                and isinstance(node.else_body[0], IfNode)
            ):
                code += " else "
                code += self.generate_if_statement(
                    node.else_body[0], 0, loop_contexts
                ).lstrip()
            else:
                code += " else {\n"
                if isinstance(node.else_body, list):
                    code += self.generate_scoped_function_body(
                        node.else_body, indent + 1, loop_contexts
                    )
                else:
                    code += self.generate_scoped_function_body(
                        [node.else_body], indent + 1, loop_contexts
                    )
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_condition_chain_if_statement(self, node, indent, loop_contexts=None):
        def success(success_indent):
            return self.generate_scoped_function_body(
                node.if_body,
                success_indent,
                loop_contexts,
            )

        failure = None
        if node.else_body is not None:

            def failure(failure_indent):
                if isinstance(node.else_body, list):
                    return self.generate_scoped_function_body(
                        node.else_body,
                        failure_indent,
                        loop_contexts,
                    )
                return self.generate_scoped_function_body(
                    [node.else_body],
                    failure_indent,
                    loop_contexts,
                )

        return self.generate_condition_chain_branch(
            node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_matches_if_statement(self, node, indent, loop_contexts=None):
        condition, code = self.generate_matches_condition_flag(
            node.condition,
            indent,
            loop_contexts,
        )
        indent_str = "    " * indent

        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.if_body, indent + 1, loop_contexts
        )
        code += f"{indent_str}}}"

        if node.else_body is not None:
            code += " else {\n"
            if isinstance(node.else_body, list):
                code += self.generate_scoped_function_body(
                    node.else_body,
                    indent + 1,
                    loop_contexts,
                )
            else:
                code += self.generate_scoped_function_body(
                    [node.else_body],
                    indent + 1,
                    loop_contexts,
                )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_if_let_statement(self, node, indent, loop_contexts=None):
        def success(success_indent):
            return self.generate_scoped_function_body(
                node.if_body,
                success_indent,
                loop_contexts,
            )

        failure = None
        if node.else_body is not None:

            def failure(failure_indent):
                if isinstance(node.else_body, list):
                    return self.generate_scoped_function_body(
                        node.else_body,
                        failure_indent,
                        loop_contexts,
                    )
                return self.generate_scoped_function_body(
                    [node.else_body],
                    failure_indent,
                    loop_contexts,
                )

        return self.generate_let_pattern_condition_branch(
            node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_let_pattern_condition_branch(
        self,
        condition,
        indent,
        success,
        failure=None,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            condition.expression,
            indent,
            loop_contexts,
        )

        if failure is None:
            code += self.generate_nested_pattern_match(
                subject,
                condition.pattern,
                indent,
                success,
            )
            return code

        matched_flag = self.next_match_chain_flag_name()
        code += f"{indent_str}bool {matched_flag} = false;\n"

        def matched_success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n" + success(
                success_indent
            )

        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_nested_pattern_match(
            subject,
            condition.pattern,
            indent + 1,
            matched_success,
        )
        code += f"{indent_str}}}\n"
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += failure(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_condition_chain_branch(
        self,
        chain,
        indent,
        success,
        failure=None,
        loop_contexts=None,
    ):
        if failure is None:
            return self.generate_condition_chain_operands(
                chain.operands,
                0,
                indent,
                success,
                loop_contexts,
            )

        matched_flag = self.next_match_chain_flag_name()
        indent_str = "    " * indent
        code = f"{indent_str}bool {matched_flag} = false;\n"

        def matched_success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n" + success(
                success_indent
            )

        code += self.generate_condition_chain_operands(
            chain.operands,
            0,
            indent,
            matched_success,
            loop_contexts,
        )
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += failure(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_condition_chain_operands(
        self,
        operands,
        index,
        indent,
        success,
        loop_contexts=None,
    ):
        if index >= len(operands):
            return success(indent)

        operand = operands[index]

        def continue_chain(next_indent):
            return self.generate_condition_chain_operands(
                operands,
                index + 1,
                next_indent,
                success,
                loop_contexts,
            )

        return self.generate_condition_chain_operand(
            operand,
            indent,
            continue_chain,
            loop_contexts,
        )

    def generate_condition_chain_operand(
        self,
        operand,
        indent,
        success,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if isinstance(operand, LetPatternConditionNode):
            subject, code = self.generate_match_subject(
                operand.expression,
                indent,
                loop_contexts,
            )
            code += self.generate_nested_pattern_match(
                subject,
                operand.pattern,
                indent,
                success,
            )
            return code

        if isinstance(operand, MatchesMacroNode):
            condition, code = self.generate_matches_condition_flag(
                operand,
                indent,
                loop_contexts,
            )
            code += f"{indent_str}if ({condition}) {{\n"
            code += success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        condition_code, condition = self.generate_try_expression(
            operand,
            indent,
            loop_contexts,
        )
        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += success(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_for_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        # Convert Rust for-in loop to C-style for loop
        code = self.generate_labeled_control_declarations(loop_context, indent)
        enumerate_loop = self.generate_enumerate_for_loop(
            node,
            indent,
            loop_contexts,
            nested_contexts,
        )
        if enumerate_loop is not None:
            return code + enumerate_loop

        pattern = self.generate_for_loop_pattern(node.pattern)
        code += self.generate_for_loop_header(
            pattern,
            node.iterable,
            indent,
            loop_contexts,
        )
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_for_loop_pattern(self, pattern):
        if isinstance(pattern, str) and not self.is_discard_pattern(pattern):
            return pattern
        return self.next_for_loop_index_name()

    def next_for_loop_index_name(self):
        name = f"_for_index_{self.for_loop_index_counter}"
        self.for_loop_index_counter += 1
        return name

    def next_for_loop_step_name(self):
        name = f"_for_step_{self.for_loop_step_counter}"
        self.for_loop_step_counter += 1
        return name

    def next_for_loop_bound_name(self):
        name = f"_for_bound_{self.for_loop_bound_counter}"
        self.for_loop_bound_counter += 1
        return name

    def next_for_loop_iterable_name(self):
        name = f"_for_iterable_{self.for_loop_iterable_counter}"
        self.for_loop_iterable_counter += 1
        return name

    def generate_enumerate_for_loop(
        self,
        node,
        indent,
        loop_contexts=None,
        nested_contexts=None,
    ):
        enumerate_parts = self.parse_for_enumerate_iterable(node.pattern, node.iterable)
        if enumerate_parts is None:
            return None

        index_pattern, value_pattern, collection = enumerate_parts
        indent_str = "    " * indent

        collection_setup, collection_expr = self.generate_for_loop_iterable_expression(
            collection,
            indent,
            loop_contexts,
        )
        index_name = self.generate_enumerate_index_name(index_pattern)

        code = collection_setup
        code += (
            f"{indent_str}for (int {index_name} = 0; "
            f"{index_name} < {collection_expr}.length; {index_name}++) {{\n"
        )
        code += self.generate_enumerate_value_binding(
            value_pattern,
            collection_expr,
            index_name,
            indent + 1,
        )
        code += self.generate_scoped_function_body(
            node.body,
            indent + 1,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"
        return code

    def parse_for_enumerate_iterable(self, pattern, iterable):
        if not (
            isinstance(pattern, TupleNode)
            and len(pattern.elements) == 2
            and isinstance(iterable, FunctionCallNode)
            and isinstance(iterable.name, MemberAccessNode)
            and iterable.name.member == "enumerate"
            and not iterable.args
        ):
            return None

        iterator = iterable.name.object
        if not (
            isinstance(iterator, FunctionCallNode)
            and isinstance(iterator.name, MemberAccessNode)
            and iterator.name.member in {"iter", "iter_mut", "into_iter"}
            and not iterator.args
        ):
            return None

        return pattern.elements[0], pattern.elements[1], iterator.name.object

    def generate_for_loop_iterable_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        setup, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )

        if setup:
            if isinstance(expression, TryNode):
                return setup, value
            name = self.next_for_loop_iterable_name()
            return setup + f"{indent_str}auto {name} = {value};\n", name

        if self.expression_has_side_effects(expression):
            name = self.next_for_loop_iterable_name()
            return f"{indent_str}auto {name} = {value};\n", name

        return "", value

    def generate_enumerate_index_name(self, index_pattern):
        if isinstance(index_pattern, str) and not self.is_discard_pattern(
            index_pattern
        ):
            return index_pattern
        return self.next_for_loop_index_name()

    def generate_enumerate_value_binding(
        self,
        value_pattern,
        collection_expr,
        index_name,
        indent,
    ):
        if not isinstance(value_pattern, str) or self.is_discard_pattern(value_pattern):
            return ""

        indent_str = "    " * indent
        return f"{indent_str}auto {value_pattern} = {collection_expr}[{index_name}];\n"

    def generate_for_loop_header(self, pattern, iterable, indent, loop_contexts=None):
        indent_str = "    " * indent
        (
            range_node,
            step_expression,
            is_reverse,
            align_reverse_step,
        ) = self.parse_for_range_iterable(iterable)

        if range_node is not None:
            start_setup, start = self.generate_for_loop_setup_expression(
                range_node.start if range_node.start is not None else "0",
                indent,
                self.next_for_loop_bound_name,
                loop_contexts,
            )
            setup = start_setup
            condition = ""
            step = None

            if range_node.end is not None:
                end_setup, end = self.generate_for_loop_setup_expression(
                    range_node.end,
                    indent,
                    self.next_for_loop_bound_name,
                    loop_contexts,
                )
                setup += end_setup
                if step_expression is not None:
                    step_setup, step = self.generate_for_loop_setup_expression(
                        step_expression,
                        indent,
                        self.next_for_loop_step_name,
                        loop_contexts,
                    )
                    setup += step_setup

                if is_reverse:
                    if step is not None and align_reverse_step:
                        initial_value = self.format_reverse_stepped_range_start(
                            start,
                            end,
                            step,
                            range_node.inclusive,
                        )
                    else:
                        initial_value = (
                            end
                            if range_node.inclusive
                            else self.format_reverse_exclusive_range_start(end)
                        )
                    condition = f"{pattern} >= {start}"
                else:
                    initial_value = start
                    comparison = "<=" if range_node.inclusive else "<"
                    condition = f"{pattern} {comparison} {end}"
            else:
                initial_value = start
                if step_expression is not None:
                    step_setup, step = self.generate_for_loop_setup_expression(
                        step_expression,
                        indent,
                        self.next_for_loop_step_name,
                        loop_contexts,
                    )
                    setup += step_setup

            if step is None:
                increment = f"{pattern}--" if is_reverse else f"{pattern}++"
            else:
                operator = "-=" if is_reverse else "+="
                increment = f"{pattern} {operator} {step}"

            return (
                setup + f"{indent_str}for (int {pattern} = {initial_value}; "
                f"{condition}; {increment}) {{\n"
            )

        setup, iterable_expr = self.generate_for_loop_setup_expression(
            iterable,
            indent,
            self.next_for_loop_bound_name,
            loop_contexts,
        )

        return (
            setup + f"{indent_str}for (int {pattern} = 0; "
            f"{pattern} < {iterable_expr}; {pattern}++) {{\n"
        )

    def generate_for_loop_setup_expression(
        self,
        expression,
        indent,
        name_factory,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        setup, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )

        if setup:
            if not isinstance(expression, TryNode):
                name = name_factory()
                setup += f"{indent_str}int {name} = {value};\n"
                return setup, name
            return setup, value

        if self.expression_has_side_effects(expression):
            name = name_factory()
            return f"{indent_str}int {name} = {value};\n", name

        return "", value

    def format_reverse_exclusive_range_start(self, end):
        if isinstance(end, str) and re.fullmatch(r"-?\d+", end):
            return str(int(end) - 1)
        return f"({end} - 1)"

    def format_reverse_stepped_range_start(self, start, end, step, inclusive):
        start_value = self.parse_integer_literal(start)
        end_value = self.parse_integer_literal(end)
        step_value = self.parse_integer_literal(step)

        if (
            start_value is not None
            and end_value is not None
            and step_value is not None
            and step_value != 0
        ):
            final_value = end_value if inclusive else end_value - 1
            step_count = (final_value - start_value) // step_value
            return str(start_value + (step_count * step_value))

        final_expr = (
            end if inclusive else self.format_reverse_exclusive_range_start(end)
        )
        if start == "0":
            return f"({final_expr} - ({final_expr} % {step}))"
        return f"({start} + ((({final_expr} - {start}) / {step}) * {step}))"

    def parse_integer_literal(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            return int(value)
        return None

    def parse_for_range_iterable(self, iterable):
        if isinstance(iterable, RangeNode):
            return iterable, None, False, False

        if not isinstance(iterable, FunctionCallNode):
            return None, None, False, False

        callee = iterable.name
        if (
            isinstance(callee, MemberAccessNode)
            and callee.member == "rev"
            and not iterable.args
        ):
            range_node, step_expression, _, align_reverse_step = (
                self.parse_for_range_iterable(callee.object)
            )
            if range_node is not None and range_node.end is not None:
                return (
                    range_node,
                    step_expression,
                    True,
                    step_expression is not None or align_reverse_step,
                )

        if (
            isinstance(callee, MemberAccessNode)
            and callee.member == "step_by"
            and len(iterable.args) == 1
        ):
            range_node, _, is_reverse, align_reverse_step = (
                self.parse_for_range_iterable(callee.object)
            )
            if range_node is not None and is_reverse:
                return range_node, iterable.args[0], True, align_reverse_step
            if isinstance(callee.object, RangeNode):
                return callee.object, iterable.args[0], False, False

        return None, None, False, False

    def generate_while_loop(self, node, indent, loop_contexts=None):
        if isinstance(node.condition, ConditionChainNode):
            return self.generate_condition_chain_while_loop(
                node,
                indent,
                loop_contexts,
            )

        if isinstance(node.condition, MatchesMacroNode):
            return self.generate_matches_while_loop(node, indent, loop_contexts)

        if isinstance(node.condition, LetPatternConditionNode):
            return self.generate_while_let_loop(node, indent, loop_contexts)

        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        if self.expression_contains_try(node.condition):
            code += f"{indent_str}while (true) {{\n"
            condition_code, condition = self.generate_try_expression(
                node.condition,
                indent + 1,
                nested_contexts,
            )
            code += condition_code
            code += f"{indent_str}    if (!{condition}) {{\n"
            code += f"{indent_str}        break;\n"
            code += f"{indent_str}    }}\n"
            code += self.generate_scoped_function_body(
                node.body, indent + 1, nested_contexts
            )
            code += f"{indent_str}}}\n"
            return code

        condition = self.generate_expression(node.condition)
        code += f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_condition_chain_while_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"

        def success(success_indent):
            return self.generate_scoped_function_body(
                node.body,
                success_indent,
                nested_contexts,
            )

        def failure(failure_indent):
            failure_indent_str = "    " * failure_indent
            return f"{failure_indent_str}break;\n"

        code += self.generate_condition_chain_branch(
            node.condition,
            indent + 1,
            success,
            failure,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_matches_while_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"
        condition, condition_code = self.generate_matches_condition_flag(
            node.condition,
            indent + 1,
            nested_contexts,
        )
        code += condition_code
        code += f"{indent_str}    if (!{condition}) {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_while_let_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"

        def success(success_indent):
            return self.generate_scoped_function_body(
                node.body,
                success_indent,
                nested_contexts,
            )

        def failure(failure_indent):
            failure_indent_str = "    " * failure_indent
            return f"{failure_indent_str}break;\n"

        code += self.generate_let_pattern_condition_branch(
            node.condition,
            indent + 1,
            success,
            failure,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_loop(self, node, indent, result_target=None, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(
            loop_contexts, node, result_target=result_target
        )
        loop_context = nested_contexts[-1]

        # Convert Rust infinite loop to while(true)
        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_match_statement(self, node, indent, loop_contexts=None):
        if self.match_requires_if_chain(node):
            return self.generate_match_if_chain(
                node,
                indent,
                self.generate_scoped_function_body,
                loop_contexts,
            )

        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            node.expression,
            indent,
            loop_contexts,
        )
        expression = self.generate_match_subject_expression(subject)
        switch_break_flag = self.create_switch_break_flag(node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        # Convert Rust match to switch statement
        code += self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_scoped_function_body(
                arm.body,
                indent + 2,
                arm_loop_contexts,
            )
            if self.match_arm_needs_switch_terminator(arm.body):
                code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        code += self.generate_switch_break_propagation(
            switch_break_flag,
            loop_contexts,
            indent,
        )
        return code

    def generate_match_case_label(self, pattern, indent):
        indent_str = "    " * indent
        if pattern == "_":
            return f"{indent_str}default:\n"

        return f"{indent_str}case {self.generate_expression(pattern)}:\n"

    def match_requires_if_chain(self, match_node):
        return any(
            isinstance(arm.pattern, MatchOrPatternNode)
            or arm.guard is not None
            or not self.is_switch_case_pattern(arm.pattern)
            for arm in match_node.arms
        )

    def is_switch_case_pattern(self, pattern):
        return pattern == "_" or isinstance(pattern, str)

    def next_match_chain_flag_name(self):
        name = f"_rust_match_matched_{self.match_chain_counter}"
        self.match_chain_counter += 1
        return name

    def next_match_subject_name(self):
        name = f"_rust_match_subject_{self.match_subject_counter}"
        self.match_subject_counter += 1
        return name

    def next_match_payload_name(self):
        name = f"_rust_match_payload_{self.match_payload_counter}"
        self.match_payload_counter += 1
        return name

    def next_match_or_flag_name(self):
        name = f"_rust_match_or_matched_{self.match_or_counter}"
        self.match_or_counter += 1
        return name

    def next_match_tuple_name(self):
        name = f"_rust_match_tuple_{self.match_tuple_counter}"
        self.match_tuple_counter += 1
        return name

    def next_match_array_name(self):
        name = f"_rust_match_array_{self.match_array_counter}"
        self.match_array_counter += 1
        return name

    def next_matches_result_name(self):
        name = f"_rust_matches_result_{self.matches_result_counter}"
        self.matches_result_counter += 1
        return name

    def next_closure_param_name(self):
        name = f"_rust_closure_arg_{self.closure_param_counter}"
        self.closure_param_counter += 1
        return name

    def next_try_subject_name(self):
        name = f"_rust_try_subject_{self.try_subject_counter}"
        self.try_subject_counter += 1
        return name

    def next_try_value_name(self):
        name = f"_rust_try_value_{self.try_value_counter}"
        self.try_value_counter += 1
        return name

    def next_try_block_value_name(self):
        name = f"_rust_try_block_value_{self.try_block_value_counter}"
        self.try_block_value_counter += 1
        return name

    def next_transparent_block_value_name(self):
        name = f"_rust_block_value_{self.transparent_block_value_counter}"
        self.transparent_block_value_counter += 1
        return name

    def next_inline_expression_value_name(self):
        name = f"_rust_expr_value_{self.inline_expression_value_counter}"
        self.inline_expression_value_counter += 1
        return name

    def generate_match_if_chain(
        self,
        match_node,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        subject, code = self.generate_match_subject(
            match_node.expression,
            indent,
            loop_contexts,
        )

        matched_flag = self.next_match_chain_flag_name()
        indent_str = "    " * indent
        code += f"{indent_str}bool {matched_flag} = false;\n"

        for arm in match_node.arms:
            code += self.generate_match_if_chain_arm(
                subject,
                arm,
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        return code

    def generate_match_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if isinstance(expression, TupleNode):
            return self.generate_match_tuple_subject(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, ArrayNode):
            return self.generate_match_array_subject(
                expression,
                indent,
                loop_contexts,
            )

        if self.match_subject_needs_statement_result(expression):
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        if self.expression_contains_try(expression):
            try_code, subject = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            if not isinstance(expression, TryNode) and self.expression_has_side_effects(
                expression
            ):
                subject_name = self.next_match_subject_name()
                try_code += f"{indent_str}auto {subject_name} = {subject};\n"
                subject = subject_name
            return subject, try_code

        subject = self.generate_expression(expression)
        if self.expression_has_side_effects(expression):
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            subject = subject_name

        return subject, code

    def match_subject_needs_statement_result(self, expression):
        return self.is_block_expression_node(expression) and (
            self.expression_contains_try(expression)
            or self.expression_has_side_effects(expression)
        )

    def generate_match_tuple_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        elements = []
        code = ""

        for element in expression.elements:
            if isinstance(element, TupleNode):
                nested_subject, nested_code = self.generate_match_tuple_subject(
                    element,
                    indent,
                    loop_contexts,
                )
                code += nested_code
                elements.append(nested_subject)
                continue

            element_name = self.next_match_tuple_name()
            element_code, element_value = self.generate_try_expression(
                element,
                indent,
                loop_contexts,
            )
            code += element_code
            code += f"{indent_str}auto {element_name} = " f"{element_value};\n"
            elements.append(element_name)

        return TupleNode(elements), code

    def generate_match_array_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        elements = []
        code = ""

        if expression.size is not None:
            subject_code, subject = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            subject_name = self.next_match_array_name()
            code += subject_code
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            return subject_name, code

        for element in expression.elements:
            if isinstance(element, ArrayNode):
                nested_subject, nested_code = self.generate_match_array_subject(
                    element,
                    indent,
                )
                code += nested_code
                elements.append(nested_subject)
                continue

            element_name = self.next_match_array_name()
            element_code, element_value = self.generate_try_expression(
                element,
                indent,
                loop_contexts,
            )
            code += element_code
            code += f"{indent_str}auto {element_name} = " f"{element_value};\n"
            elements.append(element_name)

        return ArrayNode(elements), code

    def generate_match_if_chain_arm(
        self,
        subject,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        if isinstance(arm.pattern, ReferenceNode):
            return self.generate_match_if_chain_arm(
                subject,
                MatchArmNode(arm.pattern.expression, arm.guard, arm.body),
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        if isinstance(arm.pattern, MatchBindingPatternNode) and isinstance(
            arm.pattern.pattern, MatchOrPatternNode
        ):
            code = ""
            for pattern in arm.pattern.pattern.patterns:
                code += self.generate_match_if_chain_arm(
                    subject,
                    MatchArmNode(
                        MatchBindingPatternNode(arm.pattern.name, pattern),
                        arm.guard,
                        arm.body,
                    ),
                    matched_flag,
                    indent,
                    branch_generator,
                    loop_contexts,
                )
            return code

        if isinstance(arm.pattern, MatchOrPatternNode):
            code = ""
            for pattern in arm.pattern.patterns:
                code += self.generate_match_if_chain_arm(
                    subject,
                    MatchArmNode(pattern, arm.guard, arm.body),
                    matched_flag,
                    indent,
                    branch_generator,
                    loop_contexts,
                )
            return code

        if self.pattern_has_nested_constructor(arm.pattern):
            return self.generate_nested_constructor_match_arm(
                subject,
                arm,
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        indent_str = "    " * indent
        pattern_condition = self.generate_match_pattern_condition(
            subject,
            arm.pattern,
        )
        if pattern_condition is None:
            arm_condition = f"!{matched_flag}"
        else:
            arm_condition = f"!{matched_flag} && {pattern_condition}"

        code = f"{indent_str}if ({arm_condition}) {{\n"
        code += self.generate_match_pattern_bindings(
            subject,
            arm.pattern,
            indent + 1,
        )

        if arm.guard is not None:
            guard_code, guard = self.generate_try_expression(
                arm.guard,
                indent + 1,
                loop_contexts,
            )
            code += guard_code
            code += f"{indent_str}    if ({guard}) {{\n"
            code += f"{indent_str}        {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 2, loop_contexts)
            code += f"{indent_str}    }}\n"
        else:
            code += f"{indent_str}    {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 1, loop_contexts)

        code += f"{indent_str}}}\n"
        return code

    def pattern_has_nested_constructor(self, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.pattern_has_nested_constructor(pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.pattern_has_nested_constructor(pattern.expression)

        if isinstance(pattern, TupleNode):
            return True

        if isinstance(pattern, ArrayNode):
            return True

        if isinstance(pattern, MatchStructPatternNode):
            return True

        if isinstance(pattern, MatchOrPatternNode):
            return any(
                isinstance(alternative, FunctionCallNode)
                or self.pattern_has_nested_constructor(alternative)
                or isinstance(alternative, TupleNode)
                or isinstance(alternative, ArrayNode)
                or isinstance(alternative, MatchStructPatternNode)
                for alternative in pattern.patterns
            )

        if not isinstance(pattern, FunctionCallNode):
            return False

        return any(
            isinstance(arg, FunctionCallNode)
            or self.pattern_has_nested_constructor(arg)
            for arg in pattern.args
        )

    def generate_nested_constructor_match_arm(
        self,
        subject,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        def success(success_indent):
            return self.generate_match_arm_success(
                arm,
                matched_flag,
                success_indent,
                branch_generator,
                loop_contexts,
            )

        code = f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_nested_pattern_match(
            subject,
            arm.pattern,
            indent + 1,
            success,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        if isinstance(pattern, MatchBindingPatternNode):

            def binding_success(success_indent):
                success_indent_str = "    " * success_indent
                binding = ""
                if not self.is_discard_pattern(pattern.name):
                    binding = (
                        f"{success_indent_str}auto {pattern.name} = "
                        f"{self.generate_match_subject_expression(subject)};\n"
                    )
                return binding + success(success_indent)

            return self.generate_nested_pattern_match(
                subject,
                pattern.pattern,
                indent,
                binding_success,
            )

        if isinstance(pattern, TupleNode):
            return self.generate_nested_tuple_pattern_match(
                subject,
                pattern,
                0,
                indent,
                success,
            )

        if isinstance(pattern, ArrayNode):
            return self.generate_nested_array_pattern_match(
                subject,
                pattern,
                indent,
                success,
            )

        if isinstance(pattern, MatchStructPatternNode):
            return self.generate_nested_struct_pattern_match(
                subject,
                pattern,
                indent,
                success,
            )

        if isinstance(pattern, ReferenceNode):
            return self.generate_nested_pattern_match(
                subject,
                pattern.expression,
                indent,
                success,
            )

        if isinstance(pattern, MatchOrPatternNode):
            matched_flag = self.next_match_or_flag_name()
            code = f"{indent_str}bool {matched_flag} = false;\n"
            for alternative in pattern.patterns:

                def alternative_success(success_indent, matched_flag=matched_flag):
                    success_indent_str = "    " * success_indent
                    return f"{success_indent_str}{matched_flag} = true;\n" + success(
                        success_indent
                    )

                code += f"{indent_str}if (!{matched_flag}) {{\n"
                code += self.generate_nested_pattern_match(
                    subject,
                    alternative,
                    indent + 1,
                    alternative_success,
                )
                code += f"{indent_str}}}\n"
            return code

        if not isinstance(pattern, FunctionCallNode):
            condition = self.generate_match_pattern_condition(subject, pattern)
            if condition is None:
                code = self.generate_match_pattern_bindings(subject, pattern, indent)
                code += success(indent)
                return code

            code = f"{indent_str}if ({condition}) {{\n"
            code += self.generate_match_pattern_bindings(subject, pattern, indent + 1)
            code += success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        if isinstance(pattern, MatchRestPatternNode):
            return success(indent)

        subject_expr = self.generate_match_subject_expression(subject)
        constructor = self.generate_constructor_helper_name(pattern.name)
        code = f"{indent_str}if (is_{constructor}({subject_expr})) {{\n"
        code += self.generate_nested_constructor_args(
            subject_expr,
            constructor,
            pattern.args,
            0,
            indent + 1,
            success,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_struct_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        subject_expr = self.generate_match_subject_expression(subject)

        def fields_success(success_indent):
            return self.generate_nested_struct_pattern_fields(
                subject_expr,
                pattern,
                0,
                success_indent,
                success,
            )

        if self.is_record_variant_pattern(pattern):
            helper = self.generate_constructor_helper_name(pattern.name)
            code = f"{indent_str}if (is_{helper}({subject_expr})) {{\n"
            code += fields_success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        return fields_success(indent)

    def generate_nested_struct_pattern_fields(
        self,
        subject,
        pattern,
        index,
        indent,
        success,
    ):
        if index >= len(pattern.fields):
            return success(indent)

        field_name, field_pattern = pattern.fields[index]
        field_subject = self.generate_struct_field_accessor(
            subject,
            pattern,
            field_name,
        )

        def continue_fields(next_indent):
            return self.generate_nested_struct_pattern_fields(
                subject,
                pattern,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            field_subject,
            field_pattern,
            indent,
            continue_fields,
        )

    def generate_nested_tuple_pattern_match(
        self,
        subject,
        pattern,
        index,
        indent,
        success,
    ):
        if index >= len(pattern.elements):
            return success(indent)

        if isinstance(subject, TupleNode) and index >= len(subject.elements):
            return ""

        element_subject = self.generate_tuple_element_accessor(subject, index)

        def continue_elements(next_indent):
            return self.generate_nested_tuple_pattern_match(
                subject,
                pattern,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            element_subject,
            pattern.elements[index],
            indent,
            continue_elements,
        )

    def generate_nested_array_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        rest_index = self.find_array_rest_index(pattern)
        length_condition = self.generate_array_length_condition(
            subject,
            pattern,
            rest_index,
        )

        def length_success(success_indent):
            return self.generate_nested_array_elements(
                subject,
                pattern,
                rest_index,
                success_indent,
                success,
            )

        if length_condition is None:
            return length_success(indent)

        code = f"{indent_str}if ({length_condition}) {{\n"
        code += length_success(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_array_elements(
        self,
        subject,
        pattern,
        rest_index,
        indent,
        success,
    ):
        entries = self.get_array_pattern_entries(subject, pattern, rest_index)

        def rest_success(success_indent):
            code = self.generate_array_rest_binding(
                subject,
                pattern,
                rest_index,
                success_indent,
            )
            code += success(success_indent)
            return code

        return self.generate_nested_array_element_entries(
            entries,
            0,
            indent,
            rest_success,
        )

    def generate_nested_array_element_entries(
        self,
        entries,
        index,
        indent,
        success,
    ):
        if index >= len(entries):
            return success(indent)

        element_pattern, element_subject = entries[index]

        def continue_entries(next_indent):
            return self.generate_nested_array_element_entries(
                entries,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            element_subject,
            element_pattern,
            indent,
            continue_entries,
        )

    def find_array_rest_index(self, pattern):
        for index, element in enumerate(pattern.elements):
            if self.is_array_rest_pattern(element):
                return index
        return None

    def is_array_rest_pattern(self, pattern):
        if isinstance(pattern, MatchRestPatternNode):
            return True
        return isinstance(pattern, MatchBindingPatternNode) and isinstance(
            pattern.pattern, MatchRestPatternNode
        )

    def generate_array_length_condition(self, subject, pattern, rest_index):
        fixed_count = len(pattern.elements) - (1 if rest_index is not None else 0)
        subject_len = self.get_static_array_subject_length(subject)

        if rest_index is None:
            if subject_len is not None:
                return None if subject_len == fixed_count else "false"
            return (
                f"({self.generate_array_length_expression(subject)} == {fixed_count})"
            )

        if subject_len is not None:
            return None if subject_len >= fixed_count else "false"
        return f"({self.generate_array_length_expression(subject)} >= {fixed_count})"

    def get_static_array_subject_length(self, subject):
        if isinstance(subject, ArrayNode):
            return len(subject.elements)
        return None

    def get_array_pattern_entries(self, subject, pattern, rest_index):
        entries = []
        elements = pattern.elements

        if rest_index is None:
            for index, element in enumerate(elements):
                entries.append(
                    (element, self.generate_array_element_accessor(subject, index))
                )
            return entries

        for index, element in enumerate(elements[:rest_index]):
            entries.append(
                (element, self.generate_array_element_accessor(subject, index))
            )

        suffix = elements[rest_index + 1 :]
        for offset, element in enumerate(suffix):
            entries.append(
                (
                    element,
                    self.generate_array_suffix_element_accessor(
                        subject,
                        len(suffix) - offset,
                    ),
                )
            )

        return entries

    def generate_array_rest_binding(self, subject, pattern, rest_index, indent):
        if rest_index is None:
            return ""

        rest_pattern = pattern.elements[rest_index]
        if not isinstance(rest_pattern, MatchBindingPatternNode):
            return ""
        if self.is_discard_pattern(rest_pattern.name):
            return ""

        indent_str = "    " * indent
        suffix_count = len(pattern.elements) - rest_index - 1
        rest_length = self.generate_array_rest_length_expression(
            subject,
            rest_index,
            suffix_count,
        )
        rest_slice = self.generate_array_slice_expression(
            subject,
            rest_index,
            rest_length,
        )
        return f"{indent_str}auto {rest_pattern.name} = {rest_slice};\n"

    def generate_array_rest_length_expression(self, subject, rest_index, suffix_count):
        subject_len = self.get_static_array_subject_length(subject)
        fixed_count = rest_index + suffix_count
        if subject_len is not None:
            return str(subject_len - fixed_count)

        length_expr = self.generate_array_length_expression(subject)
        if fixed_count == 0:
            return length_expr
        return f"({length_expr} - {fixed_count})"

    def generate_array_length_expression(self, subject):
        subject_len = self.get_static_array_subject_length(subject)
        if subject_len is not None:
            return str(subject_len)
        return f"length({self.generate_match_subject_expression(subject)})"

    def generate_array_slice_expression(self, subject, start, length):
        return (
            f"_rust_slice({self.generate_match_subject_expression(subject)}, "
            f"{start}, {length})"
        )

    def generate_nested_constructor_args(
        self,
        subject,
        constructor,
        args,
        index,
        indent,
        success,
    ):
        if index >= len(args):
            return success(indent)

        indent_str = "    " * indent
        arg = args[index]
        payload = self.generate_constructor_payload_accessor(
            subject,
            constructor,
            index,
            len(args),
        )

        def continue_args(next_indent):
            return self.generate_nested_constructor_args(
                subject,
                constructor,
                args,
                index + 1,
                next_indent,
                success,
            )

        if isinstance(arg, MatchOrPatternNode):
            payload_name = self.next_match_payload_name()
            code = f"{indent_str}auto {payload_name} = {payload};\n"
            code += self.generate_nested_pattern_match(
                payload_name,
                arg,
                indent,
                continue_args,
            )
            return code

        if isinstance(arg, MatchBindingPatternNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, TupleNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, ArrayNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, MatchStructPatternNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, ReferenceNode):
            return self.generate_nested_pattern_match(
                payload,
                arg.expression,
                indent,
                continue_args,
            )

        if self.is_discard_pattern(arg):
            return continue_args(indent)

        if self.is_simple_pattern_binding(arg):
            code = f"{indent_str}auto {arg} = {payload};\n"
            code += continue_args(indent)
            return code

        if isinstance(arg, FunctionCallNode):
            payload_name = self.next_match_payload_name()
            code = f"{indent_str}auto {payload_name} = {payload};\n"
            code += self.generate_nested_pattern_match(
                payload_name,
                arg,
                indent,
                continue_args,
            )
            return code

        if isinstance(arg, RangeNode):
            condition = self.generate_match_pattern_condition(payload, arg)
            code = f"{indent_str}if ({condition}) {{\n"
            code += continue_args(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        condition = f"({payload} == {self.generate_expression(arg)})"
        code = f"{indent_str}if ({condition}) {{\n"
        code += continue_args(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_match_arm_success(
        self,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        if arm.guard is not None:
            guard = self.generate_expression(arm.guard)
            code = f"{indent_str}if ({guard}) {{\n"
            code += f"{indent_str}    {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 1, loop_contexts)
            code += f"{indent_str}}}\n"
            return code

        code = f"{indent_str}{matched_flag} = true;\n"
        code += branch_generator(arm.body, indent, loop_contexts)
        return code

    def generate_match_pattern_condition(self, subject, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.generate_match_pattern_condition(subject, pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.generate_match_pattern_condition(subject, pattern.expression)

        if isinstance(pattern, MatchOrPatternNode):
            alternatives = []
            for alternative in pattern.patterns:
                condition = self.generate_match_pattern_condition(subject, alternative)
                alternatives.append(condition or "true")
            return " || ".join(f"({condition})" for condition in alternatives)

        conditions = self.generate_pattern_conditions(subject, pattern)
        if not conditions:
            return None

        return " && ".join(conditions)

    def generate_pattern_conditions(self, subject, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.generate_pattern_conditions(subject, pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.generate_pattern_conditions(subject, pattern.expression)

        if isinstance(pattern, TupleNode):
            if isinstance(subject, TupleNode) and len(pattern.elements) != len(
                subject.elements
            ):
                return ["false"]

            conditions = []
            for index, element in enumerate(pattern.elements):
                conditions.extend(
                    self.generate_pattern_conditions(
                        self.generate_tuple_element_accessor(subject, index),
                        element,
                    )
                )
            return conditions

        if isinstance(pattern, ArrayNode):
            rest_index = self.find_array_rest_index(pattern)
            length_condition = self.generate_array_length_condition(
                subject,
                pattern,
                rest_index,
            )
            conditions = [length_condition] if length_condition is not None else []
            for element, element_subject in self.get_array_pattern_entries(
                subject,
                pattern,
                rest_index,
            ):
                conditions.extend(
                    self.generate_pattern_conditions(element_subject, element)
                )
            return conditions

        if isinstance(pattern, MatchStructPatternNode):
            subject_expr = self.generate_match_subject_expression(subject)
            conditions = []
            if self.is_record_variant_pattern(pattern):
                helper = self.generate_constructor_helper_name(pattern.name)
                conditions.append(f"is_{helper}({subject_expr})")

            for field_name, field_pattern in pattern.fields:
                conditions.extend(
                    self.generate_pattern_conditions(
                        self.generate_struct_field_accessor(
                            subject_expr,
                            pattern,
                            field_name,
                        ),
                        field_pattern,
                    )
                )
            return conditions

        if isinstance(pattern, MatchRestPatternNode):
            return []

        if isinstance(pattern, MatchOrPatternNode):
            condition = self.generate_match_pattern_condition(subject, pattern)
            return [f"({condition})"] if condition else []

        if isinstance(pattern, RangeNode):
            subject_expr = self.generate_match_subject_expression(subject)
            start = self.generate_expression(pattern.start)
            end = self.generate_expression(pattern.end)
            end_operator = "<=" if pattern.inclusive else "<"
            return [
                f"({subject_expr} >= {start})",
                f"({subject_expr} {end_operator} {end})",
            ]

        if self.is_discard_pattern(pattern) or self.is_simple_pattern_binding(pattern):
            return []

        if isinstance(pattern, FunctionCallNode):
            subject_expr = self.generate_match_subject_expression(subject)
            constructor = self.generate_constructor_helper_name(pattern.name)
            conditions = [f"is_{constructor}({subject_expr})"]
            arg_count = len(pattern.args)
            for index, arg in enumerate(pattern.args):
                payload = self.generate_constructor_payload_accessor(
                    subject_expr,
                    constructor,
                    index,
                    arg_count,
                )
                conditions.extend(self.generate_pattern_conditions(payload, arg))
            return conditions

        return [
            f"({self.generate_match_subject_expression(subject)} == "
            f"{self.generate_expression(pattern)})"
        ]

    def generate_match_pattern_bindings(self, subject, pattern, indent):
        if isinstance(pattern, MatchBindingPatternNode):
            indent_str = "    " * indent
            binding = ""
            if not self.is_discard_pattern(pattern.name):
                binding += (
                    f"{indent_str}auto {pattern.name} = "
                    f"{self.generate_match_subject_expression(subject)};\n"
                )
            binding += self.generate_match_pattern_bindings(
                subject,
                pattern.pattern,
                indent,
            )
            return binding

        if isinstance(pattern, ReferenceNode):
            return self.generate_match_pattern_bindings(
                subject,
                pattern.expression,
                indent,
            )

        if isinstance(pattern, MatchOrPatternNode):
            return ""

        indent_str = "    " * indent
        if self.is_simple_pattern_binding(pattern):
            return (
                f"{indent_str}auto {pattern} = "
                f"{self.generate_match_subject_expression(subject)};\n"
            )

        if isinstance(pattern, TupleNode):
            code = ""
            for index, element in enumerate(pattern.elements):
                code += self.generate_match_pattern_bindings(
                    self.generate_tuple_element_accessor(subject, index),
                    element,
                    indent,
                )
            return code

        if isinstance(pattern, ArrayNode):
            code = ""
            rest_index = self.find_array_rest_index(pattern)
            for element, element_subject in self.get_array_pattern_entries(
                subject,
                pattern,
                rest_index,
            ):
                code += self.generate_match_pattern_bindings(
                    element_subject,
                    element,
                    indent,
                )
            code += self.generate_array_rest_binding(
                subject,
                pattern,
                rest_index,
                indent,
            )
            return code

        if isinstance(pattern, MatchStructPatternNode):
            subject_expr = self.generate_match_subject_expression(subject)
            code = ""
            for field_name, field_pattern in pattern.fields:
                code += self.generate_match_pattern_bindings(
                    self.generate_struct_field_accessor(
                        subject_expr,
                        pattern,
                        field_name,
                    ),
                    field_pattern,
                    indent,
                )
            return code

        if isinstance(pattern, MatchRestPatternNode):
            return ""

        if not isinstance(pattern, FunctionCallNode):
            return ""

        subject_expr = self.generate_match_subject_expression(subject)
        constructor = self.generate_constructor_helper_name(pattern.name)
        arg_count = len(pattern.args)
        code = ""
        for index, arg in enumerate(pattern.args):
            payload = self.generate_constructor_payload_accessor(
                subject_expr,
                constructor,
                index,
                arg_count,
            )
            code += self.generate_match_pattern_bindings(payload, arg, indent)
        return code

    def generate_match_subject_expression(self, subject):
        if isinstance(subject, TupleNode):
            return self.generate_expression(subject)
        if isinstance(subject, ArrayNode):
            return self.generate_expression(subject)
        return str(subject)

    def generate_tuple_element_accessor(self, subject, index):
        if isinstance(subject, TupleNode):
            element = subject.elements[index]
            if isinstance(element, TupleNode):
                return element
            return self.generate_expression(element)

        return f"_rust_tuple_{index}({self.generate_match_subject_expression(subject)})"

    def generate_array_element_accessor(self, subject, index):
        if isinstance(subject, ArrayNode):
            element = subject.elements[index]
            if isinstance(element, ArrayNode):
                return element
            return self.generate_expression(element)

        return f"{self.generate_match_subject_expression(subject)}[{index}]"

    def generate_array_suffix_element_accessor(self, subject, reverse_index):
        subject_len = self.get_static_array_subject_length(subject)
        if subject_len is not None:
            return self.generate_array_element_accessor(
                subject,
                subject_len - reverse_index,
            )

        length_expr = self.generate_array_length_expression(subject)
        if reverse_index == 1:
            index_expr = f"({length_expr} - 1)"
        else:
            index_expr = f"({length_expr} - {reverse_index})"
        return f"{self.generate_match_subject_expression(subject)}[{index_expr}]"

    def is_record_variant_pattern(self, pattern):
        return isinstance(pattern.name, str) and "::" in pattern.name

    def generate_struct_field_accessor(self, subject, pattern, field_name):
        subject_expr = self.generate_match_subject_expression(subject)
        if self.is_record_variant_pattern(pattern):
            helper = self.generate_constructor_helper_name(pattern.name)
            return f"unwrap_{helper}_{field_name}({subject_expr})"

        return f"{subject_expr}.{field_name}"

    def generate_constructor_payload_accessor(
        self,
        subject,
        constructor,
        index,
        arg_count,
    ):
        if arg_count == 1:
            return f"unwrap_{constructor}({subject})"
        return f"unwrap_{constructor}_{index}({subject})"

    def generate_constructor_helper_name(self, constructor):
        if isinstance(constructor, str):
            name = constructor
        else:
            name = self.generate_expression(constructor)

        helper_name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        return helper_name or "variant"

    def is_simple_pattern_binding(self, pattern):
        if not isinstance(pattern, str):
            return False
        if pattern == "_" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern):
            return False
        if pattern in {"true", "false", "None", "Some", "Ok", "Err"}:
            return False
        return pattern[0].islower()

    def match_arm_needs_switch_terminator(self, body):
        return not self.branch_guarantees_control_transfer(body)

    def branch_guarantees_control_transfer(self, body):
        if isinstance(body, (BreakNode, ContinueNode, ReturnNode)):
            return True
        if isinstance(body, list):
            if not body:
                return False
            return self.branch_guarantees_control_transfer(body[-1])
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            if block_node.statements:
                return self.branch_guarantees_control_transfer(
                    block_node.statements[-1]
                )
            return self.branch_guarantees_control_transfer(
                self.get_block_expression(block_node)
            )
        if isinstance(body, IfNode):
            if body.else_body is None:
                return False
            return self.branch_guarantees_control_transfer(
                body.if_body
            ) and self.branch_guarantees_control_transfer(body.else_body)
        if isinstance(body, MatchNode):
            return bool(body.arms) and all(
                self.branch_guarantees_control_transfer(arm.body) for arm in body.arms
            )
        return False

    def create_switch_break_flag(self, match_node, loop_contexts=None):
        if not self.match_break_targets_current_loop(match_node, loop_contexts):
            return None

        flag = f"_rust_switch_break_{self.switch_break_counter}"
        self.switch_break_counter += 1
        return flag

    def match_break_targets_current_loop(self, match_node, loop_contexts=None):
        contexts = list(loop_contexts or [])
        if not contexts:
            return False

        current_loop = contexts[-1]
        current_label = current_loop.get("label")
        return any(
            self.contains_current_loop_break_in_switch(
                arm.body,
                current_label,
                nested_loop_depth=0,
            )
            for arm in match_node.arms
        )

    def contains_current_loop_break_in_switch(
        self, node, current_label, nested_loop_depth=0
    ):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_current_loop_break_in_switch(
                    child,
                    current_label,
                    nested_loop_depth,
                )
                for child in node
            )
        if isinstance(node, BreakNode):
            if nested_loop_depth != 0:
                return False
            return node.label is None or node.label == current_label
        if isinstance(node, LoopNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, WhileNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, ForNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, IfNode):
            return self.contains_current_loop_break_in_switch(
                node.if_body,
                current_label,
                nested_loop_depth,
            ) or self.contains_current_loop_break_in_switch(
                node.else_body,
                current_label,
                nested_loop_depth,
            )
        if isinstance(node, MatchNode):
            return any(
                self.contains_current_loop_break_in_switch(
                    arm.body,
                    current_label,
                    nested_loop_depth,
                )
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_current_loop_break_in_switch(
                block_node.statements,
                current_label,
                nested_loop_depth,
            ) or self.contains_current_loop_break_in_switch(
                self.get_block_expression(block_node),
                current_label,
                nested_loop_depth,
            )
        return False

    def with_switch_break_flag(self, loop_contexts=None, switch_break_flag=None):
        contexts = list(loop_contexts or [])
        if not switch_break_flag or not contexts:
            return contexts

        contexts[-1] = dict(contexts[-1])
        contexts[-1]["switch_break_flag"] = switch_break_flag
        return contexts

    def current_switch_break_flag(self, loop_contexts=None):
        contexts = list(loop_contexts or [])
        if not contexts:
            return None
        return contexts[-1].get("switch_break_flag")

    def generate_switch_break_declaration(self, switch_break_flag, indent):
        if not switch_break_flag:
            return ""
        indent_str = "    " * indent
        return f"{indent_str}bool {switch_break_flag} = false;\n"

    def generate_switch_break_propagation(
        self, switch_break_flag, loop_contexts=None, indent=1
    ):
        if not switch_break_flag:
            return ""

        indent_str = "    " * indent
        outer_switch_break_flag = self.current_switch_break_flag(loop_contexts)
        if outer_switch_break_flag:
            return (
                f"{indent_str}if ({switch_break_flag}) {{\n"
                f"{indent_str}    {outer_switch_break_flag} = true;\n"
                f"{indent_str}    break;\n"
                f"{indent_str}}}\n"
            )

        return (
            f"{indent_str}if ({switch_break_flag}) {{\n"
            f"{indent_str}    break;\n"
            f"{indent_str}}}\n"
        )

    def generate_break_statement(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        labeled_context = self.resolve_labeled_loop_context(node.label, loop_contexts)
        if labeled_context is not None and not self.is_innermost_loop_context(
            labeled_context,
            loop_contexts,
        ):
            target = labeled_context["result_target"]

            if target is self.return_result_target and node.value is not None:
                return self.generate_expression_result(
                    node.value,
                    indent,
                    self.return_result_target,
                    loop_contexts,
                )

            if target and node.value is not None:
                code += self.generate_expression_result(
                    node.value,
                    indent,
                    target,
                    loop_contexts,
                )
            elif node.value is not None:
                code += self.generate_discarded_expression(
                    node.value,
                    indent,
                    loop_contexts,
                )

            if labeled_context.get("break_flag"):
                code += f"{indent_str}{labeled_context['break_flag']} = true;\n"
            code += f"{indent_str}break;\n"
            return code

        switch_break_flag = self.current_switch_break_flag(loop_contexts)
        if switch_break_flag and (
            node.label is None
            or self.is_innermost_loop_context(labeled_context, loop_contexts)
        ):
            target = self.resolve_break_value_target(node, loop_contexts)

            if target is self.return_result_target and node.value is not None:
                return self.generate_expression_result(
                    node.value,
                    indent,
                    self.return_result_target,
                    loop_contexts,
                )

            if target and node.value is not None:
                code += self.generate_expression_result(
                    node.value,
                    indent,
                    target,
                    loop_contexts,
                )
            elif node.value is not None:
                code += self.generate_discarded_expression(
                    node.value,
                    indent,
                    loop_contexts,
                )

            code += f"{indent_str}{switch_break_flag} = true;\n"
            code += f"{indent_str}break;\n"
            return code

        target = self.resolve_break_value_target(node, loop_contexts)

        if target is self.return_result_target and node.value is not None:
            return self.generate_expression_result(
                node.value,
                indent,
                self.return_result_target,
                loop_contexts,
            )

        if target and node.value is not None:
            code += self.generate_expression_result(
                node.value,
                indent,
                target,
                loop_contexts,
            )

        if target is None and node.value is not None:
            code += self.generate_discarded_expression(
                node.value,
                indent,
                loop_contexts,
            )

        code += f"{indent_str}break;\n"
        return code

    def generate_continue_statement(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        labeled_context = self.resolve_labeled_loop_context(node.label, loop_contexts)

        if labeled_context is None or self.is_innermost_loop_context(
            labeled_context,
            loop_contexts,
        ):
            return f"{indent_str}continue;\n"

        code = ""
        if labeled_context.get("continue_flag"):
            code += f"{indent_str}{labeled_context['continue_flag']} = true;\n"
        code += f"{indent_str}break;\n"
        return code

    def generate_discarded_expression(self, expression, indent, loop_contexts=None):
        if isinstance(expression, IfNode):
            return self.generate_discarded_if_expression(
                expression,
                indent,
                loop_contexts,
            )
        if isinstance(expression, MatchNode):
            return self.generate_discarded_match_expression(
                expression,
                indent,
                loop_contexts,
            )
        if self.is_block_expression_node(expression):
            return self.generate_discarded_block_expression(
                expression,
                indent,
                loop_contexts,
            )
        if isinstance(expression, LoopNode):
            return self.generate_loop(
                expression,
                indent,
                result_target=None,
                loop_contexts=loop_contexts,
            )
        if self.expression_has_side_effects(expression):
            return self.generate_expression_result(
                expression,
                indent,
                None,
                loop_contexts,
            )
        return ""

    def generate_discarded_if_expression(self, if_node, indent, loop_contexts=None):
        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            if_node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_discarded_result_branch(
            if_node.if_body,
            indent + 1,
            loop_contexts,
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_discarded_result_branch(
                if_node.else_body,
                indent + 1,
                loop_contexts,
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_discarded_match_expression(
        self, match_node, indent, loop_contexts=None
    ):
        if self.match_requires_if_chain(match_node):
            return self.generate_match_if_chain(
                match_node,
                indent,
                lambda body, branch_indent, contexts: self.generate_discarded_result_branch(
                    body,
                    branch_indent,
                    contexts,
                ),
                loop_contexts,
            )

        indent_str = "    " * indent
        expression = self.generate_expression(match_node.expression)
        switch_break_flag = self.create_switch_break_flag(match_node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        code = self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in match_node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_discarded_result_branch(
                arm.body,
                indent + 2,
                arm_loop_contexts,
            )
            if self.match_arm_needs_switch_terminator(arm.body):
                code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        code += self.generate_switch_break_propagation(
            switch_break_flag,
            loop_contexts,
            indent,
        )
        return code

    def generate_discarded_result_branch(self, branch, indent, loop_contexts=None):
        if self.is_block_expression_node(branch):
            return self.generate_discarded_block_expression(
                branch,
                indent,
                loop_contexts,
            )
        if isinstance(branch, list):
            return self.generate_scoped_function_body(branch, indent, loop_contexts)
        if branch is not None:
            return self.generate_discarded_expression(
                branch,
                indent,
                loop_contexts,
            )
        return ""

    def generate_discarded_block_expression(
        self, block_node, indent, loop_contexts=None
    ):
        block_node = self.get_block_expression_node(block_node)
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(
                block_node.statements, indent, loop_contexts
            )
            expression = self.get_block_expression(block_node)
            if expression is not None:
                code += self.generate_discarded_expression(
                    expression,
                    indent,
                    loop_contexts,
                )
            return code
        finally:
            self.pop_local_callable_scope()

    def expression_has_side_effects(self, expression):
        if isinstance(expression, FunctionCallNode):
            if self.is_pure_method_call(expression):
                return self.method_call_has_side_effects(expression)
            return True
        if isinstance(expression, TryNode):
            return True
        if isinstance(expression, TryBlockNode):
            return True
        if isinstance(expression, AwaitNode):
            return self.expression_has_side_effects(expression.expression)
        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            return bool(block_node.statements) or self.expression_has_side_effects(
                self.get_block_expression(block_node)
            )
        if isinstance(expression, AssignmentNode):
            return True
        if isinstance(expression, IfNode):
            return True
        if isinstance(expression, MatchNode):
            return True
        if isinstance(expression, MatchesMacroNode):
            return self.expression_has_side_effects(
                expression.expression
            ) or self.expression_has_side_effects(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_has_side_effects(operand)
                for operand in expression.operands
            )
        if isinstance(expression, ClosureNode):
            return False
        if isinstance(expression, LoopNode):
            return True
        if isinstance(expression, BinaryOpNode):
            return self.expression_has_side_effects(
                expression.left
            ) or self.expression_has_side_effects(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_has_side_effects(expression.operand)
        if isinstance(expression, CastNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, ReferenceNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, DereferenceNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, TupleNode):
            return any(
                self.expression_has_side_effects(element)
                for element in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_has_side_effects(element)
                for element in expression.elements
            ) or self.expression_has_side_effects(expression.size)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_has_side_effects(
                expression.array
            ) or self.expression_has_side_effects(expression.index)
        if isinstance(expression, MemberAccessNode):
            return self.expression_has_side_effects(expression.object)
        if isinstance(expression, VectorConstructorNode):
            return any(self.expression_has_side_effects(arg) for arg in expression.args)
        if isinstance(expression, StructInitializationNode):
            return any(
                self.expression_has_side_effects(field_expression)
                for _, field_expression in expression.fields
            )
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_has_side_effects(expression.condition)
                or self.expression_has_side_effects(expression.true_expr)
                or self.expression_has_side_effects(expression.false_expr)
            )
        return False

    def is_pure_method_call(self, expression):
        if not (
            isinstance(expression, FunctionCallNode)
            and isinstance(expression.name, MemberAccessNode)
        ):
            return False

        method_name = expression.name.member
        if method_name in {"len", "length", "normalize"}:
            return not expression.args
        if method_name in {"dot", "cross"}:
            return len(expression.args) == 1
        return not expression.args and self.is_swizzle_member(method_name)

    def method_call_has_side_effects(self, expression):
        return self.expression_has_side_effects(expression.name.object) or any(
            self.expression_has_side_effects(arg) for arg in expression.args
        )

    def is_swizzle_member(self, member):
        if not isinstance(member, str) or not 1 <= len(member) <= 4:
            return False
        return (
            all(char in "xyzw" for char in member)
            or all(char in "rgba" for char in member)
            or all(char in "stpq" for char in member)
        )

    def extend_loop_contexts(self, loop_contexts, node, result_target=None):
        contexts = list(loop_contexts or [])
        label = getattr(node, "label", None)
        break_flag = None
        continue_flag = None
        if label and self.loop_uses_labeled_control_flags(node):
            suffix = self.make_labeled_control_suffix(label)
            break_flag = f"_rust_break_{suffix}"
            continue_flag = f"_rust_continue_{suffix}"

        contexts.append(
            {
                "label": label,
                "result_target": result_target,
                "break_flag": break_flag,
                "continue_flag": continue_flag,
                "switch_break_flag": None,
            }
        )
        return contexts

    def loop_uses_labeled_control_flags(self, loop_node):
        label = getattr(loop_node, "label", None)
        if not label:
            return False
        return self.contains_labeled_control_target(
            getattr(loop_node, "body", []),
            label,
            nested_loop_depth=0,
        )

    def contains_labeled_control_target(self, node, label, nested_loop_depth=0):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_labeled_control_target(child, label, nested_loop_depth)
                for child in node
            )
        if isinstance(node, (BreakNode, ContinueNode)):
            return node.label == label and nested_loop_depth > 0
        if isinstance(node, LoopNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, WhileNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, ForNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, IfNode):
            return self.contains_labeled_control_target(
                node.if_body,
                label,
                nested_loop_depth,
            ) or self.contains_labeled_control_target(
                node.else_body,
                label,
                nested_loop_depth,
            )
        if isinstance(node, MatchNode):
            return any(
                self.contains_labeled_control_target(
                    arm.body,
                    label,
                    nested_loop_depth,
                )
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_labeled_control_target(
                block_node.statements,
                label,
                nested_loop_depth,
            ) or self.contains_labeled_control_target(
                self.get_block_expression(block_node),
                label,
                nested_loop_depth,
            )
        return False

    def make_labeled_control_suffix(self, label):
        clean = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip("'"))
        clean = clean.strip("_") or "label"
        suffix = f"{clean}_{self.labeled_control_counter}"
        self.labeled_control_counter += 1
        return suffix

    def generate_labeled_control_declarations(self, context, indent):
        if not context.get("break_flag"):
            return ""

        indent_str = "    " * indent
        return (
            f"{indent_str}bool {context['break_flag']} = false;\n"
            f"{indent_str}bool {context['continue_flag']} = false;\n"
        )

    def generate_labeled_control_propagation(self, loop_contexts=None, indent=1):
        contexts = list(loop_contexts or [])
        if not contexts:
            return ""

        indent_str = "    " * indent
        code = ""
        innermost_index = len(contexts) - 1

        for index, context in enumerate(contexts):
            break_flag = context.get("break_flag")
            continue_flag = context.get("continue_flag")
            if not break_flag:
                continue

            if index == innermost_index:
                code += (
                    f"{indent_str}if ({continue_flag}) {{\n"
                    f"{indent_str}    {continue_flag} = false;\n"
                    f"{indent_str}    continue;\n"
                    f"{indent_str}}}\n"
                    f"{indent_str}if ({break_flag}) {{\n"
                    f"{indent_str}    break;\n"
                    f"{indent_str}}}\n"
                )
            else:
                code += (
                    f"{indent_str}if ({continue_flag} || {break_flag}) {{\n"
                    f"{indent_str}    break;\n"
                    f"{indent_str}}}\n"
                )

        return code

    def statement_requires_labeled_control_propagation(self, stmt, loop_contexts=None):
        contexts = list(loop_contexts or [])
        innermost_index = len(contexts) - 1

        for index, context in enumerate(contexts):
            if not context.get("break_flag"):
                continue

            if index == innermost_index:
                if self.contains_labeled_control_target(
                    stmt,
                    context["label"],
                    nested_loop_depth=0,
                ):
                    return True
            elif self.contains_labeled_control_reference(stmt, context["label"]):
                return True

        return False

    def statement_needs_labeled_control_propagation(self, stmt, loop_contexts=None):
        if isinstance(stmt, (BreakNode, ContinueNode, ReturnNode)):
            return False

        if not self.statement_requires_labeled_control_propagation(
            stmt,
            loop_contexts,
        ):
            return False

        if (
            self.is_block_expression_node(stmt)
            and self.branch_guarantees_control_transfer(stmt)
            and not self.contains_switch_match(stmt)
        ):
            return False

        return True

    def contains_switch_match(self, node):
        if node is None:
            return False
        if isinstance(node, list):
            return any(self.contains_switch_match(child) for child in node)
        if isinstance(node, MatchNode):
            return not self.match_requires_if_chain(node) or any(
                self.contains_switch_match(arm.body) for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_switch_match(
                block_node.statements,
            ) or self.contains_switch_match(self.get_block_expression(block_node))
        if isinstance(node, IfNode):
            return self.contains_switch_match(
                node.if_body,
            ) or self.contains_switch_match(node.else_body)
        if isinstance(node, LoopNode):
            return self.contains_switch_match(node.body)
        if isinstance(node, WhileNode):
            return self.contains_switch_match(node.body)
        if isinstance(node, ForNode):
            return self.contains_switch_match(node.body)
        return False

    def contains_labeled_control_reference(self, node, label):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_labeled_control_reference(child, label) for child in node
            )
        if isinstance(node, (BreakNode, ContinueNode)):
            return node.label == label
        if isinstance(node, LoopNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, WhileNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, ForNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, IfNode):
            return self.contains_labeled_control_reference(
                node.if_body,
                label,
            ) or self.contains_labeled_control_reference(node.else_body, label)
        if isinstance(node, MatchNode):
            return any(
                self.contains_labeled_control_reference(arm.body, label)
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_labeled_control_reference(
                block_node.statements,
                label,
            ) or self.contains_labeled_control_reference(
                self.get_block_expression(block_node),
                label,
            )
        return False

    def resolve_labeled_loop_context(self, label, loop_contexts=None):
        if not label:
            return None

        for context in reversed(list(loop_contexts or [])):
            if context["label"] == label:
                return context
        return None

    def is_innermost_loop_context(self, context, loop_contexts=None):
        contexts = list(loop_contexts or [])
        return bool(contexts) and context is contexts[-1]

    def resolve_break_value_target(self, node, loop_contexts=None):
        if node.value is None:
            return None

        contexts = list(loop_contexts or [])
        if not contexts:
            return None

        if node.label:
            for context in reversed(contexts):
                if context["label"] == node.label:
                    return context["result_target"]
            return None

        return contexts[-1]["result_target"]

    def is_inline_result_expression(self, expression):
        if isinstance(expression, (LoopNode, IfNode, MatchNode, AssignmentNode)):
            return True

        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            if block_node is None:
                return False
            if block_node.statements:
                return True
            return self.expression_contains_inline_result_expression(
                self.get_block_expression(block_node)
            )

        return False

    def expression_contains_inline_result_expression(self, expression):
        if expression is None:
            return False

        if self.is_inline_result_expression(expression):
            return True

        if isinstance(expression, BinaryOpNode):
            return self.expression_contains_inline_result_expression(
                expression.left
            ) or self.expression_contains_inline_result_expression(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_contains_inline_result_expression(expression.operand)
        if isinstance(expression, FunctionCallNode):
            return self.expression_contains_inline_result_expression(
                expression.name
            ) or any(
                self.expression_contains_inline_result_expression(arg)
                for arg in expression.args
            )
        if isinstance(expression, MemberAccessNode):
            return self.expression_contains_inline_result_expression(expression.object)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_contains_inline_result_expression(
                expression.array
            ) or self.expression_contains_inline_result_expression(expression.index)
        if isinstance(expression, RangeNode):
            return self.expression_contains_inline_result_expression(
                expression.start
            ) or self.expression_contains_inline_result_expression(expression.end)
        if isinstance(expression, VectorConstructorNode):
            return any(
                self.expression_contains_inline_result_expression(arg)
                for arg in expression.args
            )
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_contains_inline_result_expression(expression.condition)
                or self.expression_contains_inline_result_expression(
                    expression.true_expr
                )
                or self.expression_contains_inline_result_expression(
                    expression.false_expr
                )
            )
        if isinstance(expression, AwaitNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, CastNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, ReferenceNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, DereferenceNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, TupleNode):
            return any(
                self.expression_contains_inline_result_expression(elem)
                for elem in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_contains_inline_result_expression(elem)
                for elem in expression.elements
            ) or self.expression_contains_inline_result_expression(expression.size)
        if isinstance(expression, StructInitializationNode):
            fields = (
                expression.fields.items()
                if isinstance(expression.fields, dict)
                else expression.fields
            )
            return any(
                self.expression_contains_inline_result_expression(field_value)
                for _, field_value in fields
            )
        if isinstance(expression, MatchesMacroNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            ) or self.expression_contains_inline_result_expression(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_contains_inline_result_expression(operand)
                for operand in expression.operands
            )

        return False

    def generate_materialized_expression(self, expression, indent, loop_contexts=None):
        if self.is_inline_result_expression(expression):
            return self.materialize_inline_result_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, BinaryOpNode):
            if expression.op in {"&&", "||"}:
                return self.generate_materialized_logical_expression(
                    expression,
                    indent,
                    loop_contexts,
                )
            left_code, left = self.generate_materialized_expression(
                expression.left,
                indent,
                loop_contexts,
            )
            right_code, right = self.generate_materialized_expression(
                expression.right,
                indent,
                loop_contexts,
            )
            return left_code + right_code, f"({left} {expression.op} {right})"

        if isinstance(expression, UnaryOpNode):
            code, operand = self.generate_materialized_expression(
                expression.operand,
                indent,
                loop_contexts,
            )
            return code, f"({expression.op}{operand})"

        if isinstance(expression, FunctionCallNode):
            return self.generate_materialized_function_call(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, MemberAccessNode):
            code, obj = self.generate_materialized_expression(
                expression.object,
                indent,
                loop_contexts,
            )
            member_name = self.resolve_member_access_name(obj, expression.member)
            return code, f"{obj}.{member_name}"

        if isinstance(expression, ArrayAccessNode):
            array_code, array = self.generate_materialized_expression(
                expression.array,
                indent,
                loop_contexts,
            )
            index_code, index = self.generate_materialized_expression(
                expression.index,
                indent,
                loop_contexts,
            )
            return array_code + index_code, f"{array}[{index}]"

        if isinstance(expression, RangeNode):
            start_code, start = self.generate_materialized_optional_expression(
                expression.start,
                indent,
                loop_contexts,
            )
            end_code, end = self.generate_materialized_optional_expression(
                expression.end,
                indent,
                loop_contexts,
            )
            return start_code + end_code, self.format_range_expression(
                start,
                end,
                expression.inclusive,
            )

        if isinstance(expression, VectorConstructorNode):
            code, args = self.generate_materialized_expression_list(
                expression.args,
                indent,
                loop_contexts,
            )
            type_name = self.map_type(expression.type_name)
            return code, f"{type_name}({', '.join(args)})"

        if isinstance(expression, TernaryOpNode):
            return self.generate_materialized_ternary_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, AwaitNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, CastNode):
            code, value = self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )
            return code, self.format_cast_expression(expression.target_type, value)

        if isinstance(expression, ReferenceNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, DereferenceNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, TupleNode):
            code, elements = self.generate_materialized_expression_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"({', '.join(elements)})"

        if isinstance(expression, ArrayNode):
            code, elements = self.generate_materialized_expression_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"{{{', '.join(elements)}}}"

        if isinstance(expression, StructInitializationNode):
            code = ""
            fields = []
            field_items = (
                expression.fields.items()
                if isinstance(expression.fields, dict)
                else expression.fields
            )
            for field_name, field_value in field_items:
                field_code, value = self.generate_materialized_expression(
                    field_value,
                    indent,
                    loop_contexts,
                )
                code += field_code
                fields.append(f"{field_name}: {value}")
            return code, f"{expression.struct_name} {{ {', '.join(fields)} }}"

        return "", self.generate_expression(expression)

    def generate_materialized_optional_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        if expression is None:
            return "", ""
        return self.generate_materialized_expression(expression, indent, loop_contexts)

    def generate_materialized_expression_list(
        self,
        expressions,
        indent,
        loop_contexts=None,
    ):
        code = ""
        values = []
        for expression in expressions:
            expression_code, value = self.generate_materialized_expression(
                expression,
                indent,
                loop_contexts,
            )
            code += expression_code
            values.append(value)
        return code, values

    def materialize_inline_result_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            if not block_node.statements:
                block_expression = self.get_block_expression(block_node)
                if block_expression is None:
                    return "", ""
                return self.generate_materialized_expression(
                    block_expression,
                    indent,
                    loop_contexts,
                )

        value_name = self.next_inline_expression_value_name()
        code = f"{indent_str}auto {value_name};\n"
        code += self.generate_expression_result(
            expression,
            indent,
            value_name,
            loop_contexts,
        )
        return code, value_name

    def generate_materialized_logical_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        left_code, left = self.generate_materialized_expression(
            expression.left,
            indent,
            loop_contexts,
        )
        result_name = self.next_inline_expression_value_name()
        code = left_code
        code += f"{indent_str}auto {result_name};\n"

        if expression.op == "||":
            code += f"{indent_str}if ({left}) {{\n"
            code += f"{indent_str}    {result_name} = true;\n"
            code += f"{indent_str}}} else {{\n"
        else:
            code += f"{indent_str}if (!{left}) {{\n"
            code += f"{indent_str}    {result_name} = false;\n"
            code += f"{indent_str}}} else {{\n"

        right_code, right = self.generate_materialized_expression(
            expression.right,
            indent + 1,
            loop_contexts,
        )
        code += right_code
        code += f"{indent_str}    {result_name} = {right};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_materialized_ternary_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        condition_code, condition = self.generate_materialized_expression(
            expression.condition,
            indent,
            loop_contexts,
        )
        result_name = self.next_inline_expression_value_name()
        code = condition_code
        code += f"{indent_str}auto {result_name};\n"
        code += f"{indent_str}if ({condition}) {{\n"
        true_code, true_expr = self.generate_materialized_expression(
            expression.true_expr,
            indent + 1,
            loop_contexts,
        )
        code += true_code
        code += f"{indent_str}    {result_name} = {true_expr};\n"
        code += f"{indent_str}}} else {{\n"
        false_code, false_expr = self.generate_materialized_expression(
            expression.false_expr,
            indent + 1,
            loop_contexts,
        )
        code += false_code
        code += f"{indent_str}    {result_name} = {false_expr};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_materialized_function_call(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        args_code, args = self.generate_materialized_expression_list(
            expression.args,
            indent,
            loop_contexts,
        )

        if isinstance(expression.name, MemberAccessNode):
            obj_code, obj = self.generate_materialized_expression(
                expression.name.object,
                indent,
                loop_contexts,
            )
            method_call = self.format_method_call_parts(
                expression.name.member,
                obj,
                args,
                expression.args,
                self.infer_value_type(expression.name.object),
            )
            if method_call is not None:
                return obj_code + args_code, method_call
            return (
                obj_code + args_code,
                f"{obj}.{expression.name.member}({', '.join(args)})",
            )

        if isinstance(expression.name, str):
            constructor = self.format_path_constructor_call_parts(
                expression.name,
                args,
            )
            if constructor is not None:
                return args_code, constructor
            associated_call = self.format_associated_impl_function_call(
                expression.name,
                args,
            )
            if associated_call is not None:
                return args_code, associated_call
            expression_call = self.format_unary_expression_intrinsic_call(
                expression.name,
                args,
            )
            if expression_call is not None:
                return args_code, expression_call
            func_name = self.map_function(expression.name)
            return args_code, f"{func_name}({', '.join(args)})"

        name_code, func_name = self.generate_materialized_expression(
            expression.name,
            indent,
            loop_contexts,
        )
        return name_code + args_code, f"{func_name}({', '.join(args)})"

    def generate_expression(self, expr):
        """Render a Rust backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            value = self.resolve_imported_module_path(self.normalize_rust_literal(expr))
            scalar_constant = self.format_scalar_associated_constant(value)
            if scalar_constant is not None:
                return scalar_constant
            vector_constant = self.format_vector_associated_constant(value)
            if vector_constant is not None:
                return vector_constant
            return self.resolve_name_alias(value)
        elif isinstance(expr, VariableNode):
            return self.resolve_name_alias(expr.name)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"({expr.op}{operand})"
        elif isinstance(expr, FunctionCallNode):
            method_call = self.format_method_call(expr)
            if method_call is not None:
                return method_call

            if isinstance(expr.name, str) and expr.name.endswith("!"):
                func_name = self.map_function(expr.name)
                args = ", ".join(self.generate_macro_argument(arg) for arg in expr.args)
                return f"{func_name}({args})"

            if isinstance(expr.name, str):
                constructor = self.format_path_constructor_call(expr.name, expr.args)
                if constructor is not None:
                    return constructor
                arg_values = [self.generate_expression(arg) for arg in expr.args]
                associated_call = self.format_associated_impl_function_call(
                    expr.name,
                    arg_values,
                )
                if associated_call is not None:
                    return associated_call
                expression_call = self.format_unary_expression_intrinsic_call(
                    expr.name,
                    arg_values,
                )
                if expression_call is not None:
                    return expression_call
                func_name = self.map_function(expr.name)
                args = ", ".join(arg_values)
            else:
                func_name = self.generate_expression(expr.name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            member_name = self.resolve_member_access_name(obj, expr.member)
            return f"{obj}.{member_name}"
        elif isinstance(expr, AwaitNode):
            return self.generate_expression(expr.expression)
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, RangeNode):
            start = "" if expr.start is None else self.generate_expression(expr.start)
            end = "" if expr.end is None else self.generate_expression(expr.end)
            return self.format_range_expression(start, end, expr.inclusive)
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            type_name = self.map_type(expr.type_name)
            return f"{type_name}({args})"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, (LoopNode, IfNode, MatchNode, AssignmentNode)):
            return self.generate_inline_result_expression(expr)
        elif self.is_block_expression_node(expr):
            return self.generate_inline_block_expression(expr)
        elif isinstance(expr, MatchesMacroNode):
            return self.generate_matches_inline_condition(expr)
        elif isinstance(expr, ConditionChainNode):
            return " && ".join(
                f"({self.generate_expression(operand)})"
                for operand in expr.operands
                if not isinstance(operand, LetPatternConditionNode)
            )
        elif isinstance(expr, ClosureNode):
            return self.generate_closure_expression(expr)
        elif isinstance(expr, TryBlockNode):
            return self.generate_try_block_expression(expr)
        elif isinstance(expr, TryNode):
            return self.generate_try_unwrap(self.generate_expression(expr.expression))
        elif isinstance(expr, CastNode):
            expression = self.generate_expression(expr.expression)
            return self.format_cast_expression(expr.target_type, expression)
        elif isinstance(expr, ReferenceNode):
            # References are handled differently in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, DereferenceNode):
            # Dereferences are typically not needed in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, TupleNode):
            # Tuples might not be directly supported, convert to struct or multiple variables
            elements = ", ".join(
                self.generate_expression(elem) for elem in expr.elements
            )
            return f"({elements})"
        elif isinstance(expr, ArrayNode):
            if expr.size is not None and len(expr.elements) == 1:
                repeated = self.expand_repeated_array_literal(
                    expr.elements[0], expr.size
                )
                if repeated is not None:
                    return repeated

            elements = ", ".join(
                self.generate_expression(elem) for elem in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, StructInitializationNode):
            fields = ", ".join(
                f"{field_name}: {self.generate_expression(field_value)}"
                for field_name, field_value in expr.fields
            )
            return f"{expr.struct_name} {{ {fields} }}"
        elif isinstance(expr, (int, float, bool)):
            return str(expr).lower() if isinstance(expr, bool) else str(expr)
        else:
            return str(expr)

    def generate_macro_argument(self, arg):
        if isinstance(arg, str):
            return self.normalize_macro_body(arg)
        return self.generate_expression(arg)

    def format_cast_expression(self, target_type, expression):
        if target_type == "_":
            return expression

        mapped_type = self.map_type(target_type)
        if mapped_type == "_":
            return expression
        return f"({mapped_type}){expression}"

    def normalize_macro_body(self, body):
        return body.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    def generate_inline_result_expression(self, expression):
        value_name = self.next_inline_expression_value_name()
        code = f"auto {value_name};\n"
        code += self.generate_expression_result(expression, 0, value_name)
        code += f"return {value_name};\n"
        return f"lambda({{ {self.compact_generated_block(code)} }})()"

    def generate_inline_block_expression(self, expression):
        block_node = self.get_block_expression_node(expression)
        block_expression = self.get_block_expression(block_node)

        if not block_node.statements:
            if block_expression is None:
                return ""
            return self.generate_expression(block_expression)

        return self.generate_inline_result_expression(expression)

    def try_generate_closure_helper(self, closure, context_name=None):
        if self.current_closure_helpers is None:
            return None
        if self.closure_helper_generation_depth:
            return None
        if not self.can_generate_closure_helper(closure):
            return None
        if self.closure_has_captures(closure):
            return None

        helper_name = self.next_closure_helper_name(context_name)
        helper_code = self.generate_closure_helper_function(helper_name, closure)
        self.current_closure_helpers.append(helper_code)
        self.closure_helper_names.add(helper_name)
        self.user_function_names.add(helper_name)
        return helper_name

    def can_generate_closure_helper(self, closure):
        if not getattr(closure, "return_type", None):
            return False
        if self.expression_contains_try(closure.body):
            return False
        if self.closure_body_contains_unsupported_helper_node(closure.body):
            return False

        for param in closure.params:
            if not param.param_type:
                return False
            if not self.is_helper_compatible_parameter_type(param.param_type):
                return False
            if not self.is_helper_supported_closure_parameter_pattern(param.pattern):
                return False
        return True

    def is_helper_compatible_parameter_type(self, param_type):
        mapped_type = self.map_type(param_type)
        if not mapped_type:
            return False
        if mapped_type.startswith("(") or "->" in mapped_type:
            return False
        return True

    def is_helper_supported_closure_parameter_pattern(self, pattern):
        if self.is_discard_pattern(pattern):
            return True
        if isinstance(pattern, str):
            return self.is_plain_identifier(pattern)
        if isinstance(pattern, ReferenceNode):
            return self.is_helper_supported_closure_parameter_pattern(
                pattern.expression
            )
        if isinstance(pattern, MatchBindingPatternNode):
            return self.is_plain_identifier(
                pattern.name
            ) and self.is_helper_supported_closure_parameter_pattern(pattern.pattern)
        if isinstance(pattern, TupleNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(element)
                for element in pattern.elements
            )
        if isinstance(pattern, MatchStructPatternNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(field_pattern)
                for _, field_pattern in pattern.fields
            )
        if isinstance(pattern, FunctionCallNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(arg)
                for arg in pattern.args
            )
        return False

    def closure_body_contains_unsupported_helper_node(self, node):
        unsupported_types = (
            ClosureNode,
            TryNode,
            TryBlockNode,
            AwaitNode,
            LoopNode,
            ForNode,
            WhileNode,
            MatchNode,
            MatchesMacroNode,
            LetPatternConditionNode,
            ConditionChainNode,
        )
        if node is None:
            return False
        if isinstance(node, unsupported_types):
            return True
        if isinstance(node, (str, int, float, bool)):
            return False
        if isinstance(node, list):
            return any(
                self.closure_body_contains_unsupported_helper_node(item)
                for item in node
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.closure_body_contains_unsupported_helper_node(
                block_node.statements
            ) or self.closure_body_contains_unsupported_helper_node(
                self.get_block_expression(block_node)
            )
        if isinstance(node, LetNode):
            return self.closure_body_contains_unsupported_helper_node(node.value)
        if isinstance(node, ReturnNode):
            return self.closure_body_contains_unsupported_helper_node(node.value)
        if isinstance(node, IfNode):
            return (
                self.closure_body_contains_unsupported_helper_node(node.condition)
                or self.closure_body_contains_unsupported_helper_node(node.if_body)
                or self.closure_body_contains_unsupported_helper_node(node.else_body)
            )
        if isinstance(node, BinaryOpNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.left
            ) or self.closure_body_contains_unsupported_helper_node(node.right)
        if isinstance(node, UnaryOpNode):
            return self.closure_body_contains_unsupported_helper_node(node.operand)
        if isinstance(node, FunctionCallNode):
            return self.closure_body_contains_unsupported_helper_node(node.name) or any(
                self.closure_body_contains_unsupported_helper_node(arg)
                for arg in node.args
            )
        if isinstance(node, MemberAccessNode):
            return self.closure_body_contains_unsupported_helper_node(node.object)
        if isinstance(node, AssignmentNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.left
            ) or self.closure_body_contains_unsupported_helper_node(node.right)
        if isinstance(node, CastNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, ReferenceNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, DereferenceNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, ArrayAccessNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.array
            ) or self.closure_body_contains_unsupported_helper_node(node.index)
        if isinstance(node, VectorConstructorNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(arg)
                for arg in node.args
            )
        if isinstance(node, TupleNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(element)
                for element in node.elements
            )
        if isinstance(node, ArrayNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(element)
                for element in node.elements
            ) or self.closure_body_contains_unsupported_helper_node(node.size)
        if isinstance(node, StructInitializationNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(value)
                for _, value in node.fields
            )
        if isinstance(node, TernaryOpNode):
            return (
                self.closure_body_contains_unsupported_helper_node(node.condition)
                or self.closure_body_contains_unsupported_helper_node(node.true_expr)
                or self.closure_body_contains_unsupported_helper_node(node.false_expr)
            )
        return False

    def closure_has_captures(self, closure):
        local_names = set()
        for param in closure.params:
            self.add_closure_local_bindings(param.pattern, local_names)
        captures = set()
        self.collect_closure_identifier_captures(
            closure.body,
            local_names,
            captures,
        )
        return bool(captures)

    def collect_closure_identifier_captures(self, node, local_names, captures):
        if node is None or isinstance(node, (int, float, bool)):
            return
        if isinstance(node, str):
            if self.is_capturable_identifier(node) and node not in local_names:
                captures.add(node)
            return
        if isinstance(node, list):
            for item in node:
                self.collect_closure_identifier_captures(item, local_names, captures)
            return
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            block_locals = set(local_names)
            for statement in block_node.statements:
                self.collect_closure_identifier_captures(
                    statement,
                    block_locals,
                    captures,
                )
            self.collect_closure_identifier_captures(
                self.get_block_expression(block_node),
                block_locals,
                captures,
            )
            return
        if isinstance(node, LetNode):
            self.collect_closure_identifier_captures(
                node.value,
                local_names,
                captures,
            )
            self.add_closure_local_bindings(node.name, local_names)
            return
        if isinstance(node, ReturnNode):
            self.collect_closure_identifier_captures(
                node.value,
                local_names,
                captures,
            )
            return
        if isinstance(node, IfNode):
            self.collect_closure_identifier_captures(
                node.condition,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.if_body,
                set(local_names),
                captures,
            )
            self.collect_closure_identifier_captures(
                node.else_body,
                set(local_names),
                captures,
            )
            return
        if isinstance(node, BinaryOpNode):
            self.collect_closure_identifier_captures(node.left, local_names, captures)
            self.collect_closure_identifier_captures(node.right, local_names, captures)
            return
        if isinstance(node, UnaryOpNode):
            self.collect_closure_identifier_captures(
                node.operand, local_names, captures
            )
            return
        if isinstance(node, FunctionCallNode):
            if isinstance(node.name, MemberAccessNode):
                self.collect_closure_identifier_captures(
                    node.name,
                    local_names,
                    captures,
                )
            elif isinstance(node.name, str):
                if (
                    node.name in local_names
                    or not self.is_closure_allowed_callable(node.name)
                ) and self.is_capturable_identifier(node.name):
                    if node.name not in local_names:
                        captures.add(node.name)
            else:
                self.collect_closure_identifier_captures(
                    node.name,
                    local_names,
                    captures,
                )
            for arg in node.args:
                self.collect_closure_identifier_captures(arg, local_names, captures)
            return
        if isinstance(node, MemberAccessNode):
            self.collect_closure_identifier_captures(node.object, local_names, captures)
            return
        if isinstance(node, AssignmentNode):
            self.collect_closure_identifier_captures(node.left, local_names, captures)
            self.collect_closure_identifier_captures(node.right, local_names, captures)
            return
        if isinstance(node, CastNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, ReferenceNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, DereferenceNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, ArrayAccessNode):
            self.collect_closure_identifier_captures(node.array, local_names, captures)
            self.collect_closure_identifier_captures(node.index, local_names, captures)
            return
        if isinstance(node, VectorConstructorNode):
            for arg in node.args:
                self.collect_closure_identifier_captures(arg, local_names, captures)
            return
        if isinstance(node, TupleNode):
            for element in node.elements:
                self.collect_closure_identifier_captures(
                    element,
                    local_names,
                    captures,
                )
            return
        if isinstance(node, ArrayNode):
            for element in node.elements:
                self.collect_closure_identifier_captures(
                    element,
                    local_names,
                    captures,
                )
            self.collect_closure_identifier_captures(node.size, local_names, captures)
            return
        if isinstance(node, StructInitializationNode):
            for _, value in node.fields:
                self.collect_closure_identifier_captures(value, local_names, captures)
            return
        if isinstance(node, TernaryOpNode):
            self.collect_closure_identifier_captures(
                node.condition,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.true_expr,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.false_expr,
                local_names,
                captures,
            )

    def add_closure_local_bindings(self, pattern, local_names):
        if isinstance(pattern, str):
            if self.is_plain_identifier(pattern) and not self.is_discard_pattern(
                pattern
            ):
                local_names.add(pattern)
            return

        for name in self.collect_pattern_binding_names(pattern):
            local_names.add(name)

    def is_capturable_identifier(self, value):
        if not isinstance(value, str):
            return False
        if "::" in value:
            return False
        if value in {"true", "false", "None", "Some", "Ok", "Err"}:
            return False
        return self.is_plain_identifier(value)

    def is_closure_allowed_callable(self, name):
        if not isinstance(name, str):
            return False
        base_name = name.rsplit("::", 1)[-1]
        if base_name in self.user_function_names:
            return True
        if base_name in self.closure_helper_names:
            return True
        if base_name in self.type_map:
            return True
        if base_name in {"Some", "Ok", "Err"}:
            return True
        return base_name[:1].isupper()

    def is_plain_identifier(self, value):
        return isinstance(value, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value)

    def next_closure_helper_name(self, context_name=None):
        context = self.sanitize_closure_helper_context(context_name)
        while True:
            if context:
                name = f"_rust_closure_{context}_{self.closure_helper_counter}"
            else:
                name = f"_rust_closure_{self.closure_helper_counter}"
            self.closure_helper_counter += 1
            if (
                name not in self.user_function_names
                and name not in self.closure_helper_names
            ):
                return name

    def next_local_function_item_name(self, context_name=None):
        context = self.sanitize_closure_helper_context(context_name)
        while True:
            if context:
                name = f"_rust_local_{context}_{self.local_function_item_counter}"
            else:
                name = f"_rust_local_{self.local_function_item_counter}"
            self.local_function_item_counter += 1
            if (
                name not in self.user_function_names
                and name not in self.closure_helper_names
                and name not in self.local_function_item_names
            ):
                self.local_function_item_names.add(name)
                return name

    def sanitize_closure_helper_context(self, context_name):
        if not context_name:
            return ""
        context = re.sub(r"[^A-Za-z0-9_]+", "_", str(context_name)).strip("_")
        if not context:
            return ""
        if context[0].isdigit():
            context = f"_{context}"
        return context

    def generate_closure_helper_function(self, helper_name, closure):
        parameter_infos = [
            self.prepare_closure_parameter(param) for param in closure.params
        ]
        params = [info["declarator"] for info in parameter_infos]
        param_types = [
            (info["subject"], param.param_type)
            for info, param in zip(parameter_infos, closure.params)
        ]
        pattern_params = [
            (info["subject"], info["pattern"])
            for info in parameter_infos
            if info["pattern"] is not None
        ]
        return_type = self.map_type(closure.return_type)
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = closure.return_type
        self.closure_helper_generation_depth += 1
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        try:
            if pattern_params:
                body = self.generate_closure_pattern_parameter_body(
                    pattern_params,
                    0,
                    2,
                    closure.body,
                )
            else:
                body = self.generate_closure_body_code(closure.body, indent=2)
        finally:
            self.pop_local_callable_scope()
            self.pop_value_type_scope()
            self.closure_helper_generation_depth -= 1
            self.current_function_return_type = previous_return_type

        return (
            f"    {return_type} {helper_name}({', '.join(params)}) {{\n"
            f"{body}"
            "    }\n\n"
        )

    def generate_closure_expression(self, closure):
        parameter_infos = [
            self.prepare_closure_parameter(param) for param in closure.params
        ]
        params = ", ".join(info["declarator"] for info in parameter_infos)
        param_types = [
            (info["subject"], param.param_type)
            for info, param in zip(parameter_infos, closure.params)
        ]
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        try:
            body = self.generate_closure_body(
                closure.body,
                [
                    (info["subject"], info["pattern"])
                    for info in parameter_infos
                    if info["pattern"] is not None
                ],
                closure.return_type,
            )
        finally:
            self.pop_local_callable_scope()
            self.pop_value_type_scope()
        if not params:
            return f"lambda({body})"
        return f"lambda({params}, {body})"

    def prepare_closure_parameter(self, param):
        if self.is_simple_closure_parameter(param):
            name = param.pattern
            return {
                "declarator": self.generate_closure_parameter_declarator(
                    name,
                    param.param_type,
                ),
                "subject": name,
                "pattern": None,
            }

        name = self.next_closure_param_name()
        pattern = None if self.is_discard_pattern(param.pattern) else param.pattern
        return {
            "declarator": self.generate_closure_parameter_declarator(
                name,
                param.param_type,
            ),
            "subject": name,
            "pattern": pattern,
        }

    def is_simple_closure_parameter(self, param):
        return (
            isinstance(param.pattern, str)
            and not self.is_discard_pattern(param.pattern)
            and self.is_simple_pattern_binding(param.pattern)
        )

    def generate_closure_parameter_declarator(self, name, param_type=None):
        if param_type:
            return self.format_typed_declarator(param_type, name)
        return name

    def generate_closure_body(self, body, pattern_params=None, return_type=None):
        pattern_params = pattern_params or []
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = self.get_closure_return_type(
            body,
            return_type,
            previous_return_type,
        )
        try:
            if pattern_params:
                code = self.generate_closure_pattern_parameter_body(
                    pattern_params,
                    0,
                    0,
                    body,
                )
                return f"{{ {self.compact_generated_block(code)} }}"

            if self.is_block_expression_node(body) or self.expression_contains_try(
                body
            ):
                code = self.generate_closure_body_code(body, indent=0)
                return f"{{ {self.compact_generated_block(code)} }}"
            return self.generate_expression(body)
        finally:
            self.current_function_return_type = previous_return_type

    def get_closure_return_type(self, body, return_type, fallback_return_type):
        if return_type:
            return return_type

        inferred = self.infer_closure_return_type(body)
        if inferred is not None:
            return inferred

        return fallback_return_type

    def infer_closure_return_type(self, body):
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            expression = self.get_block_expression(block_node)
            inferred = self.infer_closure_return_type_from_expression(expression)
            if inferred is not None:
                return inferred

            for statement in reversed(block_node.statements):
                inferred = self.infer_closure_return_type_from_expression(statement)
                if inferred is not None:
                    return inferred
            return None

        return self.infer_closure_return_type_from_expression(body)

    def infer_closure_return_type_from_expression(self, expression):
        if expression is None:
            return None

        if isinstance(expression, ReturnNode):
            return self.infer_closure_return_type_from_expression(expression.value)

        if self.is_block_expression_node(expression):
            return self.infer_closure_return_type(expression)

        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name,
            str,
        ):
            constructor = expression.name.rsplit("::", 1)[-1]
            if constructor in {"Ok", "Err"}:
                return "Result"
            if constructor == "Some":
                return "Option"

        if isinstance(expression, str) and expression.rsplit("::", 1)[-1] == "None":
            return "Option"

        if isinstance(expression, TernaryOpNode):
            return self.common_inferred_closure_return_type(
                self.infer_closure_return_type_from_expression(expression.true_expr),
                self.infer_closure_return_type_from_expression(expression.false_expr),
            )

        if isinstance(expression, IfNode):
            return self.common_inferred_closure_return_type(
                self.infer_closure_return_type_from_expression(expression.if_body),
                self.infer_closure_return_type_from_expression(expression.else_body),
            )

        if isinstance(expression, MatchNode):
            inferred = None
            has_inferred_arm = False
            for arm in expression.arms:
                arm_type = self.infer_closure_return_type_from_expression(arm.body)
                if arm_type is None:
                    continue
                if not has_inferred_arm:
                    inferred = arm_type
                    has_inferred_arm = True
                    continue
                if inferred != arm_type:
                    return None
            return inferred

        if isinstance(expression, list):
            for statement in reversed(expression):
                inferred = self.infer_closure_return_type_from_expression(statement)
                if inferred is not None:
                    return inferred

        return None

    def common_inferred_closure_return_type(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if left == right:
            return left
        return None

    def generate_closure_pattern_parameter_body(
        self,
        pattern_params,
        index,
        indent,
        body,
    ):
        if index >= len(pattern_params):
            return self.generate_closure_body_code(body, indent)

        subject, pattern = pattern_params[index]

        def success(success_indent):
            return self.generate_closure_pattern_parameter_body(
                pattern_params,
                index + 1,
                success_indent,
                body,
            )

        return self.generate_nested_pattern_match(
            subject,
            pattern,
            indent,
            success,
        )

    def generate_closure_body_code(self, body, indent=0):
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            self.push_local_callable_scope()
            try:
                code = self.generate_function_body(block_node.statements, indent=indent)
                expression = self.get_block_expression(block_node)
                if expression is not None:
                    code += self.generate_expression_result(
                        expression,
                        indent=indent,
                        result_target=self.return_result_target,
                    )
                return code
            finally:
                self.pop_local_callable_scope()

        return self.generate_expression_result(
            body,
            indent=indent,
            result_target=self.return_result_target,
        )

    def compact_generated_block(self, code):
        return " ".join(line.strip() for line in code.splitlines() if line.strip())

    def generate_matches_inline_condition(self, matches_node):
        condition = self.generate_match_pattern_condition(
            matches_node.expression,
            matches_node.pattern,
        )
        condition = condition or "true"

        if matches_node.guard is None:
            return f"({condition})"

        if self.collect_pattern_binding_names(matches_node.pattern):
            return f"({condition})"

        guard = self.generate_expression(matches_node.guard)
        return f"(({condition}) && ({guard}))"

    def normalize_rust_literal(self, value):
        byte_raw_string = RUST_BYTE_RAW_STRING_RE.match(value)
        if byte_raw_string:
            return self.normalize_raw_string_literal(byte_raw_string.group(2))

        c_raw_string = RUST_C_RAW_STRING_RE.match(value)
        if c_raw_string:
            return self.normalize_raw_string_literal(c_raw_string.group(2))

        raw_string = RUST_RAW_STRING_RE.match(value)
        if raw_string:
            return self.normalize_raw_string_literal(raw_string.group(2))

        string = RUST_STRING_RE.match(value)
        if string:
            return self.normalize_string_literal(string.group(1))

        byte_string = RUST_BYTE_STRING_RE.match(value)
        if byte_string:
            return self.normalize_byte_string_literal(byte_string.group(1))

        c_string = RUST_C_STRING_RE.match(value)
        if c_string:
            return self.normalize_string_literal(c_string.group(1))

        byte_char = RUST_BYTE_CHAR_RE.match(value)
        if byte_char:
            return self.normalize_byte_char_literal(byte_char.group(1))

        return self.normalize_numeric_literal(value)

    def normalize_numeric_literal(self, value):
        match = RUST_NUMERIC_LITERAL_RE.match(value)
        if not match:
            return value

        body = match.group("body").replace("_", "")
        if body.startswith(("0x", "0X")):
            return str(int(body, 16))
        if body.startswith(("0b", "0B")):
            return str(int(body, 2))
        if body.startswith(("0o", "0O")):
            return str(int(body, 8))
        return body

    def normalize_raw_string_literal(self, content):
        escaped = (
            content.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'

    def normalize_string_literal(self, content):
        escaped = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{escaped}"'

    def normalize_byte_string_literal(self, content):
        return self.normalize_string_literal(content)

    def normalize_byte_char_literal(self, content):
        if len(content) == 1:
            return str(ord(content))

        escape_values = {
            r"\0": 0,
            r"\n": 10,
            r"\r": 13,
            r"\t": 9,
            r"\\": 92,
            r"\'": 39,
            r"\"": 34,
        }
        if content in escape_values:
            return str(escape_values[content])

        if len(content) == 4 and content.startswith(r"\x"):
            return str(int(content[2:], 16))

        return f"b'{content}'"

    def format_method_call(self, expr):
        if not isinstance(expr.name, MemberAccessNode):
            return None

        method_name = expr.name.member
        obj = self.generate_expression(expr.name.object)
        receiver_type = self.infer_value_type(expr.name.object)

        if (
            method_name in {"map", "filter", "for_each", "any", "all"}
            and len(expr.args) == 1
            and isinstance(expr.args[0], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                expr.args[0],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"{method_name}({obj}, {helper_name})"
            return f"{method_name}({obj}, {self.generate_expression(expr.args[0])})"

        if (
            method_name == "fold"
            and len(expr.args) == 2
            and isinstance(expr.args[1], ClosureNode)
        ):
            initial_value = self.generate_expression(expr.args[0])
            helper_name = self.try_generate_closure_helper(
                expr.args[1],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"fold({obj}, {initial_value}, {helper_name})"
            return (
                f"fold({obj}, {initial_value}, "
                f"{self.generate_expression(expr.args[1])})"
            )

        args = [self.generate_expression(arg) for arg in expr.args]
        return self.format_method_call_parts(
            method_name,
            obj,
            args,
            expr.args,
            receiver_type,
        )

    def format_unary_expression_intrinsic_call(self, function_name, args):
        if len(args) != 1 or not isinstance(function_name, str):
            return None

        resolved_name = self.resolve_imported_module_path(function_name)
        if self.is_user_defined_function(resolved_name):
            return None

        current_module_function = self.current_module_user_function_name(resolved_name)
        if current_module_function is not None:
            return None

        if resolved_name == "recip":
            return f"(1.0 / {args[0]})"

        if "::" not in resolved_name:
            return None

        qualifier, intrinsic_name = resolved_name.rsplit("::", 1)
        if intrinsic_name != "recip":
            return None

        if self.is_module_qualified_function_path(qualifier.split("::")):
            return f"(1.0 / {args[0]})"

        return None

    def format_path_constructor_call(self, function_name, args):
        arg_values = [self.generate_expression(arg) for arg in args]
        return self.format_path_constructor_call_parts(function_name, arg_values)

    def map_type(self, rust_type):
        """Map a Rust type name to the closest CrossGL type name."""
        if not rust_type:
            return "void"

        lifetime_stripped_type = self.strip_lifetime_type_syntax(rust_type)
        if lifetime_stripped_type != rust_type:
            return self.map_type(lifetime_stripped_type)

        referenced_type = self.strip_reference_type(rust_type)
        if referenced_type != rust_type:
            return self.map_type(referenced_type)

        trait_object_type = self.strip_trait_object_type(rust_type)
        if trait_object_type != rust_type:
            return self.map_type(trait_object_type)

        expanded_type = self.resolve_imported_module_path(rust_type)
        if expanded_type != rust_type:
            return self.map_type(expanded_type)

        array_parts = self.split_array_type(rust_type)
        if array_parts:
            base_type, array_suffix = array_parts
            return f"{self.map_type(base_type)}{array_suffix}"

        resolved_alias = self.resolve_type_alias(rust_type)
        if resolved_alias is not None:
            return resolved_alias

        imported_alias = self.resolve_imported_type_alias(rust_type)
        if imported_alias is not None:
            return imported_alias

        builtin_type = self.map_builtin_type(rust_type)
        if builtin_type is not None:
            return builtin_type

        crossgl_type = self.format_unmapped_crossgl_type(rust_type)
        if crossgl_type is not None:
            return crossgl_type

        return rust_type

    def format_unmapped_crossgl_type(self, rust_type):
        if not isinstance(rust_type, str) or "::" not in rust_type:
            return None

        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            mapped_args = [self.map_type(arg) for arg in args]
            return (
                f"{self.crossgl_type_path_identifier(base_name)}"
                f"<{', '.join(mapped_args)}>"
            )

        return self.crossgl_type_path_identifier(rust_type)

    def crossgl_type_path_identifier(self, type_name):
        return self.crossgl_identifier(type_name.replace("::", "_"))

    def resolve_imported_type_alias(self, rust_type):
        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            target = self.imported_type_aliases.get(base_name)
            if target is None:
                return None

            target_type = f"{target}<{', '.join(args)}>"
            return self.resolve_imported_target_type(rust_type, target_type)

        target = self.imported_type_aliases.get(rust_type)
        if target is None:
            return None

        return self.resolve_imported_target_type(rust_type, target)

    def resolve_imported_target_type(self, rust_type, target_type):
        if target_type == rust_type:
            return None

        mapped = self.map_type(target_type)
        if mapped == target_type:
            return None
        if mapped == self.format_unmapped_crossgl_type(target_type):
            return rust_type
        return mapped

    def resolve_imported_module_path(self, path):
        if not isinstance(path, str) or "::" not in path:
            return path

        resolved = path
        seen = set()
        while "::" in resolved:
            root, remainder = resolved.split("::", 1)
            target = self.imported_module_aliases.get(root)
            if target is None or root in seen:
                return resolved

            seen.add(root)
            resolved = f"{target}::{remainder}"

        return resolved

    def map_builtin_type(self, rust_type):
        image_macro_type = self.map_image_macro_type(rust_type)
        if image_macro_type is not None:
            return image_macro_type

        for type_name in self.type_lookup_names(rust_type):
            mapped = self.type_map.get(type_name)
            if mapped is not None:
                return mapped

        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        for base_candidate in self.type_lookup_names(base_name):
            resource_type = self.map_resource_generic_type(base_candidate, args)
            if resource_type is not None:
                return resource_type

            candidate = f"{base_candidate}<{', '.join(args)}>"
            mapped = self.type_map.get(candidate)
            if mapped is not None:
                return mapped

            if base_candidate in ["Vec2", "Vec3", "Vec4"]:
                return self.type_map.get(base_candidate)

        return None

    def map_resource_generic_type(self, base_name, args):
        if not args:
            return None

        if base_name == "SampledImage":
            return self.map_sampled_image_generic_type(args[0])

        if base_name == "RuntimeArray":
            return f"{self.map_type(args[0])}[]"

        if base_name == "TypedBuffer":
            return self.format_typed_buffer_resource_type(args[0])

        sampled_texture_map = {
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
        if base_name in sampled_texture_map:
            return sampled_texture_map[base_name]

        depth_texture_map = {
            "DepthTexture2D": "sampler2DShadow",
            "DepthTexture2DArray": "sampler2DArrayShadow",
            "DepthTextureCube": "samplerCubeShadow",
            "DepthTextureCubeArray": "samplerCubeArrayShadow",
        }
        if base_name in depth_texture_map:
            return depth_texture_map[base_name]

        buffer_map = {
            "Buffer": "StructuredBuffer",
            "RwBuffer": "RWStructuredBuffer",
            "AppendBuffer": "AppendStructuredBuffer",
            "ConsumeBuffer": "ConsumeStructuredBuffer",
        }
        if base_name in buffer_map:
            return f"{buffer_map[base_name]}<{self.map_type(args[0])}>"

        image_suffixes = {
            "Image1D": "1D",
            "Image1DArray": "1DArray",
            "Image2D": "2D",
            "Image3D": "3D",
            "ImageCube": "Cube",
            "Image2DArray": "2DArray",
            "Image2DMS": "2DMS",
            "Image2DMSArray": "2DMSArray",
        }
        suffix = image_suffixes.get(base_name)
        if suffix is None:
            return None

        element_type = self.map_type(args[0])
        if element_type.startswith("u"):
            return f"uimage{suffix}"
        if element_type.startswith("i"):
            return f"iimage{suffix}"
        return f"image{suffix}"

    def map_typed_buffer_parameter_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        rust_type = self.strip_lifetime_type_syntax(rust_type.strip())
        if rust_type.startswith("&mut "):
            return self.map_typed_buffer_reference_type(rust_type[5:].strip(), True)
        if rust_type.startswith("&"):
            return self.map_typed_buffer_reference_type(rust_type[1:].strip(), False)
        return None

    def map_typed_buffer_reference_type(self, rust_type, writable):
        mapped_type = self.map_typed_buffer_type(rust_type, writable=writable)
        if mapped_type is not None:
            return mapped_type
        return None

    def map_typed_buffer_type(self, rust_type, writable=False):
        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        if "TypedBuffer" not in self.type_lookup_names(base_name):
            return None

        buffer_kind = "RWStructuredBuffer" if writable else "StructuredBuffer"
        return self.format_typed_buffer_resource_type(args[0], buffer_kind)

    def format_typed_buffer_resource_type(
        self,
        element_type,
        buffer_kind="StructuredBuffer",
    ):
        mapped_element_type = self.map_typed_buffer_element_type(element_type)
        return f"{buffer_kind}<{mapped_element_type}>"

    def map_typed_buffer_element_type(self, element_type):
        element_type = element_type.strip()
        if element_type.startswith("[") and element_type.endswith("]"):
            body = element_type[1:-1].strip()
            if ";" in body:
                inner_type, size = body.split(";", 1)
                return f"{self.map_type(inner_type.strip())}[{size.strip()}]"
            return self.map_type(body)
        return self.map_type(element_type)

    def map_sampled_image_generic_type(self, image_type):
        image_config = self.parse_image_macro_type(image_type)
        if image_config is not None:
            suffix = self.image_macro_dimension_suffix(image_config)
            if suffix is None:
                return None

            prefix = self.image_macro_sampler_prefix(image_config)
            if image_config["depth"] is True:
                return f"{prefix}{suffix}Shadow"
            return f"{prefix}{suffix}"

        mapped_image_type = self.map_type(image_type)
        sampler_type = self.sampled_image_type_from_mapped_image(mapped_image_type)
        if sampler_type is not None:
            return sampler_type

        return None

    def sampled_image_type_from_mapped_image(self, mapped_image_type):
        if not isinstance(mapped_image_type, str):
            return None

        if mapped_image_type.startswith("sampler"):
            return mapped_image_type
        if mapped_image_type.startswith("uimage"):
            return f"usampler{mapped_image_type[len('uimage') :]}"
        if mapped_image_type.startswith("iimage"):
            return f"isampler{mapped_image_type[len('iimage') :]}"
        if mapped_image_type.startswith("image"):
            return f"sampler{mapped_image_type[len('image') :]}"
        return None

    def map_image_macro_type(self, rust_type):
        image_config = self.parse_image_macro_type(rust_type)
        if image_config is None:
            return None

        subpass_type = self.map_image_macro_subpass_type(image_config)
        if subpass_type is not None:
            return subpass_type

        suffix = self.image_macro_dimension_suffix(image_config)
        if suffix is None:
            return None

        if image_config["sampled"] is True:
            prefix = self.image_macro_sampler_prefix(image_config)
            if image_config["depth"] is True:
                return f"{prefix}{suffix}Shadow"
            return f"{prefix}{suffix}"

        prefix = self.image_macro_storage_prefix(image_config)
        return f"{prefix}{suffix}"

    def parse_image_macro_type(self, rust_type):
        if not isinstance(rust_type, str):
            return None

        match = re.match(
            r"^(?:(?:[A-Za-z_][A-Za-z0-9_]*)::)*Image!\((?P<args>.*)\)$",
            rust_type,
        )
        if match is None:
            return None

        args = self.split_generic_arguments(match.group("args"))
        if not args:
            return None

        config = {
            "dimension": args[0].replace(" ", ""),
            "type": None,
            "format": None,
            "sampled": None,
            "multisampled": False,
            "arrayed": False,
            "depth": None,
        }

        for arg in args[1:]:
            key, value = self.split_image_macro_argument(arg)
            if key in {"type", "format"}:
                config[key] = value
            elif key in {"sampled", "multisampled", "arrayed", "depth"}:
                config[key] = self.parse_image_macro_bool(value)

        return config

    def split_image_macro_argument(self, arg):
        if "=" not in arg:
            return arg.strip(), "true"

        key, value = arg.split("=", 1)
        return key.strip(), value.strip()

    def parse_image_macro_bool(self, value):
        value = (value or "").strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
        return None

    def image_macro_dimension_suffix(self, config):
        dimension = config["dimension"].lower()
        if dimension == "buffer":
            if (
                config["multisampled"] is True
                or config["arrayed"] is True
                or config["depth"] is True
            ):
                return None
            return "Buffer"

        suffix_map = {
            "1d": "1D",
            "2d": "2D",
            "rect": "2DRect",
            "3d": "3D",
            "cube": "Cube",
        }
        suffix = suffix_map.get(dimension)
        if suffix is None:
            return None

        if config["multisampled"] is True:
            if suffix != "2D":
                return None
            suffix = f"{suffix}MS"

        if config["arrayed"] is True:
            suffix = f"{suffix}Array"

        return suffix

    def map_image_macro_subpass_type(self, config):
        if config["dimension"].lower() != "subpass":
            return None

        family = self.image_macro_numeric_family(config)
        if family == "u":
            subpass_type = "usubpassInput"
        elif family == "i":
            subpass_type = "isubpassInput"
        else:
            subpass_type = "subpassInput"

        if config["multisampled"] is True:
            subpass_type += "MS"
        if config["arrayed"] is True:
            subpass_type += "Array"
        return subpass_type

    def image_macro_sampler_prefix(self, config):
        if config["depth"] is True:
            return "sampler"

        family = self.image_macro_numeric_family(config)
        if family == "u":
            return "usampler"
        if family == "i":
            return "isampler"
        return "sampler"

    def image_macro_storage_prefix(self, config):
        family = self.image_macro_numeric_family(config)
        if family == "u":
            return "uimage"
        if family == "i":
            return "iimage"
        return "image"

    def image_macro_numeric_family(self, config):
        sample_type = config.get("type")
        if sample_type:
            sample_type = sample_type.rsplit("::", 1)[-1].strip().lower()
            if sample_type.startswith("u"):
                return "u"
            if sample_type.startswith("i"):
                return "i"
            return "f"

        image_format = config.get("format")
        if image_format:
            image_format = image_format.rsplit("::", 1)[-1].strip().lower()
            image_format = image_format.replace("_", "")
            if image_format.endswith("ui"):
                return "u"
            if image_format.endswith("i") and not image_format.endswith("ui"):
                return "i"

        return "f"

    def resolve_type_alias_target(self, rust_type):
        generic = self.parse_generic_type(rust_type)
        if generic is not None:
            base_name, args = generic
            for alias_name in self.type_lookup_names(base_name):
                alias = self.type_aliases.get(alias_name)
                if alias is None:
                    continue
                return self.apply_alias_type_arguments(alias, args)
            return None

        for alias_name in self.type_lookup_names(rust_type):
            alias = self.type_aliases.get(alias_name)
            if alias is not None and not getattr(alias, "generics", None):
                return alias.alias_type
        return None

    def resolve_type_alias(self, rust_type):
        for alias_name in self.type_lookup_names(rust_type):
            alias = self.type_aliases.get(alias_name)
            if alias is None:
                continue
            if getattr(alias, "generics", None):
                return None
            return alias.name

        generic = self.parse_generic_type(rust_type)
        if generic is None:
            return None

        base_name, args = generic
        alias = None
        for alias_name in self.type_lookup_names(base_name):
            alias = self.type_aliases.get(alias_name)
            if alias is not None:
                break
        if alias is None:
            return None

        generics = getattr(alias, "generics", []) or []
        if not generics or len(generics) != len(args):
            return None

        substituted = self.substitute_type_parameters(alias.alias_type, generics, args)
        if substituted == rust_type:
            return None
        return self.map_type(substituted)

    def type_lookup_names(self, type_name):
        names = [type_name]
        if "::" in type_name:
            names.append(type_name.rsplit("::", 1)[-1])
        return names

    def parse_generic_type(self, rust_type):
        if not rust_type or "<" not in rust_type or not rust_type.endswith(">"):
            return None

        base_name, remainder = rust_type.split("<", 1)
        base_name = base_name.strip()
        if not base_name:
            return None

        args_text = remainder[:-1]
        args = self.split_generic_arguments(args_text)
        if not args:
            return None
        return base_name, args

    def split_generic_arguments(self, args_text):
        args = []
        current = []
        depth = 0

        for char in args_text:
            if char == "," and depth == 0:
                arg = "".join(current).strip()
                if arg:
                    args.append(arg)
                current = []
                continue

            if char in "<[(":
                depth += 1
            elif char in ">])":
                depth = max(0, depth - 1)

            current.append(char)

        arg = "".join(current).strip()
        if arg:
            args.append(arg)
        return args

    def strip_lifetime_type_syntax(self, type_name):
        if not isinstance(type_name, str) or "'" not in type_name:
            return type_name

        generic = self.parse_generic_type(type_name)
        if generic is not None:
            base_name, args = generic
            stripped_args = []
            changed = False
            for arg in args:
                if self.is_lifetime_type_argument(arg):
                    changed = True
                    continue
                stripped_arg = self.strip_lifetime_type_syntax(arg)
                changed = changed or stripped_arg != arg
                stripped_args.append(stripped_arg)

            if changed:
                if not stripped_args:
                    return base_name
                return f"{base_name}<{', '.join(stripped_args)}>"

        return self.strip_reference_lifetime(type_name)

    def is_lifetime_type_argument(self, type_name):
        return bool(re.fullmatch(r"'[A-Za-z_][A-Za-z0-9_]*", type_name.strip()))

    def strip_reference_lifetime(self, type_name):
        return re.sub(
            r"(&\s*)'[A-Za-z_][A-Za-z0-9_]*\s*",
            r"\1",
            type_name,
        )

    def substitute_type_parameters(self, alias_type, generics, args):
        substitutions = {
            self.generic_parameter_name(generic): arg
            for generic, arg in zip(generics, args)
        }
        result = alias_type

        for name, replacement in substitutions.items():
            result = self.replace_type_identifier(result, name, replacement)

        return result

    def generic_parameter_name(self, generic):
        return generic.split(":", 1)[0].strip()

    def replace_type_identifier(self, text, name, replacement):
        result = []
        index = 0
        name_len = len(name)

        while index < len(text):
            if (
                text.startswith(name, index)
                and self.is_type_identifier_boundary(text, index - 1)
                and self.is_type_identifier_boundary(text, index + name_len)
            ):
                result.append(replacement)
                index += name_len
            else:
                result.append(text[index])
                index += 1

        return "".join(result)

    def is_type_identifier_boundary(self, text, index):
        if index < 0 or index >= len(text):
            return True
        return not (text[index].isalnum() or text[index] == "_")

    def strip_reference_type(self, type_name):
        if type_name.startswith("&mut "):
            return type_name[5:].strip()
        if type_name.startswith("&"):
            return type_name[1:].strip()
        return type_name

    def strip_trait_object_type(self, type_name):
        for prefix in ("impl ", "dyn "):
            if type_name.startswith(prefix):
                return type_name[len(prefix) :].strip()
        return type_name

    def split_array_type(self, type_name):
        if not type_name or "[" not in type_name:
            return None

        base_type = type_name.split("[", 1)[0]
        array_suffix = type_name[len(base_type) :]
        if not base_type or not array_suffix:
            return None
        return base_type, array_suffix

    def format_typed_declarator(self, type_name, name):
        mapped_type = self.map_type(type_name)
        array_parts = self.split_array_type(mapped_type)
        if array_parts:
            base_type, array_suffix = array_parts
            return f"{base_type} {name}{array_suffix}"
        return f"{mapped_type} {name}"

    def expand_repeated_array_literal(self, element, size):
        try:
            count = int(self.generate_expression(size))
        except (TypeError, ValueError):
            return None

        if count < 0:
            return None

        value = self.generate_expression(element)
        return "{" + ", ".join(value for _ in range(count)) + "}"

    def map_function(self, rust_func):
        stripped_func, type_args = self.split_function_type_arguments(rust_func)
        if type_args and self.lookup_user_function_signature(stripped_func) is not None:
            rust_func = stripped_func

        local_callable = self.lookup_local_callable_name(stripped_func)
        if local_callable is not None:
            return local_callable

        rust_func = self.resolve_imported_module_path(rust_func)
        if self.is_user_defined_function(rust_func):
            return rust_func

        current_module_function = self.current_module_user_function_name(rust_func)
        if current_module_function is not None:
            return current_module_function

        mapped = self.function_map.get(rust_func)
        if mapped is not None:
            return mapped

        if "::" not in rust_func:
            return rust_func

        segments = rust_func.split("::")
        mapped = self.RUST_GPU_DERIVATIVE_METHOD_MAP.get(segments[-1])
        if mapped is not None and self.is_rust_gpu_associated_intrinsic_path(
            segments[:-1]
        ):
            return mapped

        mapped = self.function_map.get(segments[-1])
        if mapped is None:
            return rust_func

        if self.is_module_qualified_function_path(segments[:-1]):
            return mapped
        return rust_func

    def is_rust_gpu_associated_intrinsic_path(self, qualifier_segments):
        if not qualifier_segments:
            return False

        type_segment = qualifier_segments[-1].split("<", 1)[0]
        if type_segment not in self.RUST_GPU_ASSOCIATED_INTRINSIC_TYPES:
            return False

        module_segments = qualifier_segments[:-1]
        return not module_segments or self.is_module_qualified_function_path(
            module_segments
        )

    def current_module_user_function_name(self, rust_func):
        if not isinstance(rust_func, str) or "::" not in rust_func:
            return None

        segments = rust_func.split("::")
        if len(segments) != 2 or segments[0] not in {"self", "crate"}:
            return None

        function_name = segments[-1]
        if self.is_user_defined_function(function_name):
            return function_name
        return None

    def is_module_qualified_function_path(self, qualifier_segments):
        if not qualifier_segments:
            return False

        namespace_roots = {"crate", "self", "super", "std"}
        if qualifier_segments[0] in namespace_roots:
            qualifier_segments = qualifier_segments[1:]

        if not qualifier_segments:
            return True

        return all(
            self.is_module_path_segment(segment) for segment in qualifier_segments
        )

    def is_module_path_segment(self, segment):
        if not segment or "<" in segment or ">" in segment:
            return False
        return not segment[0].isupper()

    def effective_attributes(self, attributes):
        for attr in attributes or []:
            yield attr
            if isinstance(attr, AttributeNode) and attr.name == "cfg_attr":
                yield from self.extract_cfg_attr_codegen_attributes(attr.args)

    def extract_cfg_attr_codegen_attributes(self, args):
        codegen_attribute_names = {
            "spirv",
            "vertex_shader",
            "fragment_shader",
            "compute_shader",
            "location",
            "binding",
        }
        extracted = []
        index = 0
        args = list(args or [])

        while index < len(args):
            name = args[index]
            if name not in codegen_attribute_names:
                index += 1
                continue

            if index + 1 < len(args) and args[index + 1] == "(":
                nested_args, index = self.collect_nested_cfg_attr_args(
                    args,
                    index + 1,
                )
                extracted.append(AttributeNode(name, nested_args))
                continue

            extracted.append(AttributeNode(name, []))
            index += 1

        return extracted

    def collect_nested_cfg_attr_args(self, args, open_index):
        nested_args = []
        depth = 0
        index = open_index + 1

        while index < len(args):
            token = args[index]
            if token == "(":
                depth += 1
                nested_args.append(token)
            elif token == ")":
                if depth == 0:
                    return nested_args, index + 1
                depth -= 1
                nested_args.append(token)
            else:
                nested_args.append(token)
            index += 1

        return nested_args, index

    def get_shader_type_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if isinstance(attr, AttributeNode):
                mapped = self.attribute_map.get(attr.name)
                if mapped in ["vertex", "fragment", "compute"]:
                    return mapped
                if attr.name == "spirv":
                    for arg in attr.args:
                        mapped = self.spirv_stage_map.get(arg)
                        if mapped:
                            return mapped
        return None

    def get_entry_point_name_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue

            entry_point_name = self.attribute_arg_value(attr.args, "entry_point_name")
            if entry_point_name is not None:
                return self.unquote_attribute_string(entry_point_name)

        return None

    def get_numthreads_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue
            args = list(attr.args or [])
            if "threads" not in args:
                continue
            threads_index = args.index("threads")
            if threads_index + 1 >= len(args) or args[threads_index + 1] != "(":
                continue

            values = []
            depth = 0
            for token in args[threads_index + 2 :]:
                if token == "(":
                    depth += 1
                    continue
                if token == ")":
                    if depth == 0:
                        break
                    depth -= 1
                    continue
                if depth == 0:
                    values.append(self.normalize_attribute_metadata_value(str(token)))

            if values:
                while len(values) < 3:
                    values.append("1")
                return values[:3]

        return None

    def unquote_attribute_string(self, value):
        if not isinstance(value, str):
            return value

        raw_string = RUST_RAW_STRING_RE.match(value)
        if raw_string:
            return raw_string.group(2)

        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]

        return value

    def get_semantic_from_attributes(self, attributes):
        if not attributes:
            return ""

        for attr in self.effective_attributes(attributes):
            if isinstance(attr, AttributeNode):
                if attr.name in self.semantic_map:
                    return f" @ {self.semantic_map[attr.name]}"
                if attr.name == "spirv":
                    semantics = []
                    for arg in attr.args:
                        if arg in self.semantic_map:
                            semantics.append(self.semantic_map[arg])
                        elif arg in self.interpolation_semantic_map:
                            semantics.append(self.interpolation_semantic_map[arg])

                    location = self.attribute_arg_value(attr.args, "location")
                    binding_index = self.attribute_arg_value(attr.args, "binding")
                    descriptor_set = self.attribute_arg_value(
                        attr.args, "descriptor_set"
                    )
                    spec_constant_id = None
                    if "spec_constant" in attr.args:
                        spec_constant_id = self.attribute_arg_value(attr.args, "id")
                    input_attachment_index = self.attribute_arg_value(
                        attr.args, "input_attachment_index"
                    )

                    if location is not None:
                        semantics.append(
                            f"location({self.normalize_attribute_metadata_value(location)})"
                        )
                    if descriptor_set is not None:
                        semantics.append(
                            f"set({self.normalize_attribute_metadata_value(descriptor_set)})"
                        )
                    if binding_index is not None:
                        semantics.append(
                            f"binding({self.normalize_attribute_metadata_value(binding_index)})"
                        )
                    if "push_constant" in attr.args:
                        semantics.append("push_constant")
                    if spec_constant_id is not None:
                        semantics.append(
                            "constant_id("
                            f"{self.normalize_attribute_metadata_value(spec_constant_id)}"
                            ")"
                        )
                    if input_attachment_index is not None:
                        semantics.append(
                            "input_attachment_index("
                            f"{self.normalize_attribute_metadata_value(input_attachment_index)}"
                            ")"
                        )
                    if "workgroup" in attr.args:
                        semantics.append("groupshared")
                    if semantics:
                        return "".join(f" @ {semantic}" for semantic in semantics)
                elif attr.name == "location" and attr.args:
                    return (
                        " @ location("
                        f"{self.normalize_attribute_metadata_value(attr.args[0])}"
                        ")"
                    )
                elif attr.name == "binding" and attr.args:
                    return (
                        " @ binding("
                        f"{self.normalize_attribute_metadata_value(attr.args[0])}"
                        ")"
                    )
        return ""

    def attribute_arg_value(self, args, key):
        for index, arg in enumerate(args):
            if arg != key:
                continue
            if index + 2 < len(args) and args[index + 1] == "=":
                return args[index + 2]
            if index + 1 < len(args):
                return args[index + 1]
        return None

    def normalize_attribute_metadata_value(self, value):
        if not isinstance(value, str):
            return value

        match = RUST_NUMERIC_LITERAL_RE.match(value)
        if not match:
            return value

        return match.group("body").replace("_", "")

    def visit_StructNode(self, node):
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = self.get_semantic_from_attributes(member.attributes)
            type_str = self.map_type(member.vtype)
            code += f"{self.get_indent()}{type_str} {member.name}{semantic};\n"

        self.indentation -= 1
        code += f"{self.get_indent()}}}\n"
        return code

    def visit_FunctionNode(self, node):
        shader_type = self.get_shader_type_from_attributes(node.attributes)
        if shader_type:
            body = self.generate_function_body(
                node.body,
                1,
                allow_implicit_final_return=True,
            )
            return f"{shader_type} {{\n{body}}}\n"
        else:
            return self.generate_function(node, 0)

    def visit_BinaryOpNode(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.generate_expression(node.operand)
        return f"({node.op}{operand})"

    def visit_ImplNode(self, node):
        code = f"// Implementation for {node.struct_name}\n"
        for func in node.functions:
            code += self.generate_function(func, 0, node.struct_name)
        return code

    def visit_UseNode(self, node):
        return f"// use {node.path}\n"

    def visit_ConstNode(self, node):
        type_str = self.map_type(node.vtype)
        value = self.generate_expression(node.value)
        return f"const {type_str} {node.name} = {value};\n"

    def visit_StaticNode(self, node):
        mutability = "mut " if node.is_mutable else ""
        type_str = self.map_type(node.vtype)
        value = self.generate_expression(node.value)
        return f"static {mutability}{type_str} {node.name} = {value};\n"
